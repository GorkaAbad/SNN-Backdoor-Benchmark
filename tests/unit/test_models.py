from pathlib import Path

import pytest
import torch

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
