"""The uniform model surface: SNNModel, Probe, the neuron factory and the architecture registry."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

import torch
from spikingjelly.activation_based import functional, neuron, surrogate
from torch import nn

from sbb.models.spec import ArchSpec
from sbb.utils.errors import SbbError
from sbb.utils.registry import Registry

ProbeKind = Literal["spikes", "membrane"]


def make_neuron(arch: ArchSpec) -> nn.Module:
    """Build one multi-step spiking neuron layer from the architecture spec."""
    spec = arch.neuron
    common = {
        "v_threshold": spec.v_threshold,
        "v_reset": spec.v_reset,
        "surrogate_function": {"atan": surrogate.ATan, "sigmoid": surrogate.Sigmoid}[
            spec.surrogate
        ](),
        "detach_reset": spec.detach_reset,
        "step_mode": "m",
        "backend": arch.backend,
    }
    if spec.model == "lif":
        return neuron.LIFNode(tau=spec.tau, **common)  # type: ignore[no-any-return]
    if spec.model == "plif":
        return neuron.ParametricLIFNode(init_tau=spec.tau, **common)  # type: ignore[no-any-return]
    return neuron.IFNode(**common)  # type: ignore[no-any-return]


class SNNModel(nn.Module):
    """Uniform surface over every architecture.

    forward takes x [B, T, P, H, W], resets neuron states, runs the wrapped network in
    multi-step mode on [T, B, P, H, W], and returns the mean over T of its output [B, K]. That
    output is a non-spiking linear readout (logits) or, in the papers' networks, voted spikes, so
    the mean is a firing rate per class.

    Args:
        spec: The architecture spec the model was built from.
        input_shape: Shape (P, H, W) of one frame.
        num_classes: Number of classes K.
        net: Network that maps [T, B, P, H, W] to [T, B, K].
    """

    def __init__(
        self, spec: ArchSpec, input_shape: tuple[int, int, int], num_classes: int, net: nn.Module
    ) -> None:
        super().__init__()
        self._spec = spec
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.net = net

    @property
    def spec(self) -> ArchSpec:
        """The architecture spec."""
        return self._spec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mean over T of the network output [B, K] for x [B, T, P, H, W]."""
        self.reset()
        out: torch.Tensor = self.net(x.transpose(0, 1))
        return out.mean(0)

    def reset(self) -> None:
        """Reset all neuron states."""
        functional.reset_net(self.net)

    def named_neurons(self) -> Iterator[tuple[str, neuron.BaseNode]]:
        """Yield (name, module) for every spiking neuron layer."""
        for name, module in self.named_modules():
            if isinstance(module, neuron.BaseNode):
                yield name, module

    def probe(self, layer: str, kind: ProbeKind, detach: bool = True) -> "Probe":
        """Return a probe that records one neuron layer. See named_neurons for layer names.

        With detach False the records keep the autograd graph, for defenses that optimize inputs
        through them (TMPBD).
        """
        neurons = dict(self.named_neurons())
        if layer not in neurons:
            raise SbbError(f"no neuron layer {layer!r}, known: {list(neurons)}")
        return Probe(neurons[layer], kind, detach)

    def activity(self) -> "Activity":
        """Return a context manager that counts spikes and synaptic operations (Activity)."""
        return Activity(self)


class Probe:
    """Context manager that records spikes or membrane potentials of one layer.

    Records are concatenated over the forward passes made inside the context. They are detached
    unless detach is False.
    """

    def __init__(self, layer: neuron.BaseNode, kind: ProbeKind, detach: bool = True) -> None:
        self.layer = layer
        self.kind = kind
        self.detach = detach
        self._records: list[torch.Tensor] = []

    def __enter__(self) -> "Probe":
        self._stored_v_seq = self.layer.store_v_seq
        self.layer.store_v_seq = self.kind == "membrane"
        self._handle = self.layer.register_forward_hook(self._record)
        return self

    def __exit__(self, *exc: object) -> None:
        self._handle.remove()
        self.layer.store_v_seq = self._stored_v_seq

    def _record(self, module: neuron.BaseNode, inputs: object, output: torch.Tensor) -> None:
        seq = module.v_seq if self.kind == "membrane" else output
        self._records.append((seq.detach() if self.detach else seq).transpose(0, 1))

    def _get(self, kind: ProbeKind) -> torch.Tensor:
        if kind != self.kind:
            raise SbbError(f"this probe records {self.kind}, not {kind}")
        return torch.cat(self._records)

    def spikes(self) -> torch.Tensor:
        """Spikes [B, T, ...]."""
        return self._get("spikes")

    def membrane(self) -> torch.Tensor:
        """Membrane potentials [B, T, ...]."""
        return self._get("membrane")

    def rates(self) -> torch.Tensor:
        """Mean spike rate over T, [B, ...]."""
        return self.spikes().mean(1)


@dataclass(frozen=True)
class ActivityCounts:
    """Spikes and synaptic operations of a model over some inputs, as NeuroBench defines them.

    Attributes:
        samples: Inputs seen.
        spikes: Spikes of every neuron layer.
        neuron_steps: Neurons times time steps times samples, the slots a spike can take.
        synops: Synaptic operations: the connections of every convolution and linear layer that
            carry a nonzero input (NeuroBench's effective MACs and ACs together).
    """

    samples: int
    spikes: float
    neuron_steps: float
    synops: float

    @property
    def spike_rate(self) -> float:
        """Spikes per neuron and time step, 1 minus NeuroBench's activation sparsity."""
        return self.spikes / self.neuron_steps if self.neuron_steps else 0.0

    @property
    def synops_per_sample(self) -> float:
        """Synaptic operations of one input over all its time steps."""
        return self.synops / self.samples if self.samples else 0.0


class Activity:
    """Context manager that counts the spikes and synaptic operations of a model's forward passes.

    The counts add up over every forward pass made inside the context. They follow NeuroBench
    (activation sparsity, synaptic operations) without depending on it.
    """

    def __init__(self, model: SNNModel) -> None:
        self.model = model
        self._samples = 0
        self._spikes = self._steps = self._synops = 0.0

    def __enter__(self) -> "Activity":
        self._handles = [self.model.register_forward_pre_hook(self._count_samples)]
        self._handles += [
            layer.register_forward_hook(self._count_spikes)
            for _, layer in self.model.named_neurons()
        ]
        self._handles += [
            module.register_forward_hook(self._count_synops)
            for module in self.model.modules()
            if isinstance(module, nn.Linear | nn.modules.conv._ConvNd)
        ]
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()

    def counts(self) -> ActivityCounts:
        """The counts so far."""
        return ActivityCounts(self._samples, self._spikes, self._steps, self._synops)

    def _count_samples(self, module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        self._samples += inputs[0].shape[0]  # x [B, T, P, H, W]

    def _count_spikes(self, module: nn.Module, inputs: object, output: torch.Tensor) -> None:
        self._spikes += float(output.detach().sum(dtype=torch.float64))
        self._steps += output.numel()

    @torch.no_grad()
    def _count_synops(
        self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        active = (inputs[0] != 0).to(output.dtype)
        if isinstance(module, nn.Linear):
            ops = active.sum(dtype=torch.float64) * module.out_features
        else:
            # The same convolution with unit weights counts, for each output, the nonzero inputs
            # that reach it, so padding and stride are exact. Multi-step inputs fold T into B.
            assert isinstance(module, nn.modules.conv._ConvNd)
            active = active.flatten(0, active.dim() - len(module.kernel_size) - 2)
            ones = torch.ones_like(module.weight)
            # ponytail: one more convolution per layer and pass, only on the evaluation passes.
            ops = module._conv_forward(active, ones, None).sum(dtype=torch.float64)
        self._synops += float(ops)


ArchBuilder = Callable[[ArchSpec, tuple[int, int, int], int], SNNModel]
ARCHITECTURES: Registry[ArchBuilder] = Registry("architecture")


def build_model(arch: ArchSpec, input_shape: tuple[int, int, int], num_classes: int) -> SNNModel:
    """Build the registered architecture arch.name."""
    return ARCHITECTURES.get(arch.name)(arch, input_shape, num_classes)
