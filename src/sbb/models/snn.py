"""The uniform model surface: SNNModel, Probe, the neuron factory and the architecture registry."""

from collections.abc import Callable, Iterator
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
    multi-step mode on [T, B, P, H, W], and returns logits [B, K] as the mean over T of
    the network's non-spiking linear readout.

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
        """Return logits [B, K] for x [B, T, P, H, W]."""
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

    def probe(self, layer: str, kind: ProbeKind) -> "Probe":
        """Return a probe that records one neuron layer. See named_neurons for layer names."""
        neurons = dict(self.named_neurons())
        if layer not in neurons:
            raise SbbError(f"no neuron layer {layer!r}, known: {list(neurons)}")
        return Probe(neurons[layer], kind)


class Probe:
    """Context manager that records spikes or membrane potentials of one layer.

    Records are concatenated over the forward passes made inside the context.
    """

    def __init__(self, layer: neuron.BaseNode, kind: ProbeKind) -> None:
        self.layer = layer
        self.kind = kind
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
        self._records.append(seq.detach().transpose(0, 1))

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


ArchBuilder = Callable[[ArchSpec, tuple[int, int, int], int], SNNModel]
ARCHITECTURES: Registry[ArchBuilder] = Registry("architecture")


def build_model(arch: ArchSpec, input_shape: tuple[int, int, int], num_classes: int) -> SNNModel:
    """Build the registered architecture arch.name."""
    return ARCHITECTURES.get(arch.name)(arch, input_shape, num_classes)
