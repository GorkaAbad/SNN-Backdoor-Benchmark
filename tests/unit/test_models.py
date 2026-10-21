from pathlib import Path

import pytest
import torch

from sbb.core.trainer import per_sample_loss
from sbb.models.checkpoint import load_checkpoint, save_checkpoint
from sbb.models.snn import SNNModel, build_model
from sbb.models.spec import ArchSpec, NeuronSpec
from sbb.utils.errors import ReproducibilityError, SbbError

ARCH = ArchSpec(name="tiny_lif_cnn", T=4)


def _model(arch: ArchSpec = ARCH) -> SNNModel:
    torch.manual_seed(0)
    return build_model(arch, (2, 16, 16), 4)


@pytest.mark.parametrize("neuron", ["lif", "plif", "if"])
def test_forward_shape_and_state_reset(neuron: str) -> None:
    model = _model(ArchSpec(name="tiny_lif_cnn", T=4, neuron=NeuronSpec(model=neuron)))
    x = torch.rand(3, 4, 2, 16, 16) * 3
    first = model(x)
    assert first.shape == (3, 4)
    assert torch.equal(first, model(x))  # forward resets neuron states


def test_flatten_size_follows_the_input() -> None:
    assert build_model(ARCH, (2, 34, 34), 10)(torch.rand(1, 4, 2, 34, 34)).shape == (1, 10)


def test_probes() -> None:
    model = _model()
    names = [name for name, _ in model.named_neurons()]
    assert len(names) == 2
    x = torch.rand(3, 4, 2, 16, 16) * 3
    with model.probe(names[0], "spikes") as spikes:
        model(x)
        model(x)
    assert spikes.spikes().shape == (6, 4, 16, 16, 16)
    assert set(spikes.spikes().unique().tolist()) <= {0.0, 1.0}
    assert spikes.rates().shape == (6, 16, 16, 16)
    with model.probe(names[1], "membrane") as membrane:
        model(x)
    assert membrane.membrane().shape == (3, 4, 32, 8, 8)
    with pytest.raises(SbbError):
        membrane.spikes()
    with pytest.raises(SbbError):
        model.probe("nope", "spikes")


def test_membrane_probe_can_keep_the_graph() -> None:
    # TMPBD optimizes inputs through membrane potentials. Its official code runs in train mode,
    # where the surrogate gradient flows through the spikes of earlier layers.
    model = _model().train()
    x = (torch.rand(2, 4, 2, 16, 16) * 3).requires_grad_()
    name = list(model.named_neurons())[-1][0]
    with model.probe(name, "membrane", detach=False) as probe:
        model(x)
    probe.membrane().sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    with model.probe(name, "membrane") as probe:
        model(x)
    assert not probe.membrane().requires_grad, "records are detached by default"


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = _model().eval()
    save_checkpoint(model, tmp_path, "k")
    loaded = load_checkpoint(tmp_path).eval()
    x = torch.rand(2, 4, 2, 16, 16)
    assert torch.equal(model(x), loaded(x)) and loaded.spec == ARCH
    (tmp_path / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ReproducibilityError):
        load_checkpoint(tmp_path)


def test_cupy_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="torch backend"):
        ArchSpec(name="tiny_lif_cnn", T=4, backend="cupy")


@pytest.mark.parametrize(
    ("name", "spikingjelly_class", "num_classes", "neuron_model", "side"),
    [
        ("dvsgesture_net", "DVSGestureNet", 11, "lif", 128),
        ("cifar10dvs_net", "CIFAR10DVSNet", 10, "lif", 128),
        ("nmnist_net", "NMNISTNet", 10, "if", 34),  # the papers use IF neurons on N-MNIST
    ],
)
def test_papers_networks_equal_spikingjelly(
    name: str, spikingjelly_class: str, num_classes: int, neuron_model: str, side: int
) -> None:
    from spikingjelly.activation_based import functional, neuron, surrogate
    from spikingjelly.activation_based.model import parametric_lif_net

    # Built exactly as Sneaky Spikes, Flashy Backdoor and TMPBD build them (models.py, get_model).
    node = {"lif": neuron.LIFNode, "if": neuron.IFNode}[neuron_model]
    theirs = getattr(parametric_lif_net, spikingjelly_class)(
        spiking_neuron=node, surrogate_function=surrogate.ATan(), detach_reset=True
    )
    functional.set_step_mode(theirs, "m")
    arch = ArchSpec(name=name, T=2, neuron=NeuronSpec(model=neuron_model))
    ours = build_model(arch, (2, side, side), num_classes)
    assert [type(m) for m in ours.net] == [type(m) for m in theirs.conv_fc]
    weights = {k.removeprefix("conv_fc."): v for k, v in theirs.state_dict().items()}
    ours.net.load_state_dict(weights)  # same parameter names and shapes, layer by layer
    x = torch.rand(2, 2, 2, side, side) * 4  # [B, T, P, H, W]

    # Eval mode. The untrained output layer is silent, so compare every layer, not the output only.
    outputs: dict[str, list[torch.Tensor]] = {"ours": [], "theirs": []}
    hooks = [
        module.register_forward_hook(lambda m, i, o, key=key: outputs[key].append(o))
        for key, net in (("ours", ours.net), ("theirs", theirs.conv_fc))
        for module in net
    ]
    ours.eval()
    theirs.eval()
    with torch.no_grad():
        expected = theirs(x.transpose(0, 1)).mean(0)  # the papers' firing-rate readout
        assert torch.equal(ours(x), expected)
    for hook in hooks:
        hook.remove()
    assert len(outputs["ours"]) == len(outputs["theirs"]) == len(theirs.conv_fc)
    for a, b in zip(outputs["ours"], outputs["theirs"], strict=True):
        assert torch.equal(a, b)
    first_spikes = next(
        o
        for m, o in zip(theirs.conv_fc, outputs["theirs"], strict=True)
        if isinstance(m, neuron.BaseNode)
    )
    assert first_spikes.sum() > 0  # in eval mode only the first block spikes

    # Train mode, one MSE step as the papers take it. The same seed gives the same dropout masks,
    # so outputs, every gradient and the batch-norm buffers must match. This also covers dropout,
    # batch-norm momentum, the surrogate and detach_reset, which an eval forward cannot see.
    ours.train()
    theirs.train()
    targets = torch.arange(2) % num_classes
    functional.reset_net(theirs)
    torch.manual_seed(0)
    theirs_out = theirs(x.transpose(0, 1)).mean(0)
    per_sample_loss("mse", theirs_out, targets).mean().backward()
    torch.manual_seed(0)
    ours_out = ours(x)
    per_sample_loss("mse", ours_out, targets).mean().backward()
    assert torch.equal(ours_out, theirs_out)
    for a, b in zip(ours.net.parameters(), theirs.parameters(), strict=True):
        assert a.grad is not None and b.grad is not None and torch.equal(a.grad, b.grad)
    for a, b in zip(ours.net.buffers(), theirs.buffers(), strict=True):
        assert torch.equal(a, b)
