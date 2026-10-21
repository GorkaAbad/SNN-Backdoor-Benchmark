from itertools import islice

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sbb.defenses.utils import channel_means, conv_blocks, superpose, zero_channels
from sbb.models.snn import SNNModel, build_model
from sbb.models.spec import ArchSpec
from sbb.utils.errors import SbbError

ARCH = ArchSpec(name="tiny_lif_cnn", T=4)


def _model() -> SNNModel:
    torch.manual_seed(0)
    return build_model(ARCH, (2, 16, 16), 4)


def _loader(n: int = 10) -> DataLoader[tuple[torch.Tensor, ...]]:
    x = torch.rand(n, 4, 2, 16, 16, generator=torch.Generator().manual_seed(1)) * 3
    return DataLoader(TensorDataset(x, torch.zeros(n), torch.arange(n)), batch_size=4)


def test_channel_means_average_over_time_batch_and_space() -> None:
    model = _model()
    conv = model.net[4]
    outputs: list[torch.Tensor] = []
    handle = conv.register_forward_hook(lambda m, i, o: outputs.append(o.detach()))
    with torch.no_grad():
        model.eval()
        for x, _, _ in _loader():
            model(x)
    handle.remove()
    expected = torch.cat([o.transpose(0, 1) for o in outputs]).mean(
        dim=(0, 1, 3, 4)
    )  # [B, T, C, H, W]

    model.train()
    means = channel_means(model, conv, _loader(), torch.device("cpu"))
    assert means.shape == (32,) and torch.allclose(means, expected.double())
    assert model.training, "the model's mode is restored"
    assert not conv._forward_hooks, "the hook is removed"
    # Any iterable of batches works, for example a part of a loader.
    first = channel_means(model, conv, islice(_loader(), 1), torch.device("cpu"))
    assert torch.allclose(first, outputs[0].transpose(0, 1).mean(dim=(0, 1, 3, 4)).double())


def test_channel_means_of_a_neuron_layer_are_firing_rates() -> None:
    model = _model()
    name, node = list(model.named_neurons())[0]
    with torch.no_grad(), model.probe(name, "spikes") as probe:
        model.eval()
        for x, _, _ in _loader():
            model(x)
    rates = probe.spikes().mean(dim=(0, 1, 3, 4))
    assert torch.allclose(
        channel_means(model, node, _loader(), torch.device("cpu")), rates.double()
    )


def test_zero_channels_prunes_in_the_state_dict() -> None:
    model = _model().eval()
    blocks = conv_blocks(model)
    assert [(b.conv.out_channels, b.bn.num_features) for b in blocks] == [(16, 16), (32, 32)]
    conv, bn = blocks[0].conv, blocks[0].bn  # the first block sees the frames, so it is not silent
    with torch.no_grad():
        bn.bias.fill_(0.5)  # a shift, so a zero output needs the bias pruned too
    zero_channels(conv, (0, 3))  # a tuple lists channels, it is not one index
    zero_channels(bn, torch.arange(16) == 5)  # a boolean mask
    fresh = _model()
    fresh.load_state_dict(model.state_dict())  # pruning survives the checkpoint the Game stores
    outputs: dict[str, torch.Tensor] = {}
    fresh_block = conv_blocks(fresh.eval())[0]
    fresh_conv, fresh_bn = fresh_block.conv, fresh_block.bn
    fresh_conv.register_forward_hook(lambda m, i, o: outputs.update(conv=o))
    fresh_bn.register_forward_hook(lambda m, i, o: outputs.update(bn=o))
    with torch.no_grad():
        fresh(torch.rand(2, 4, 2, 16, 16))
    assert (outputs["conv"][:, :, [0, 3]] == 0).all() and (outputs["conv"][:, :, 1] != 0).any()
    assert (outputs["bn"][:, :, 5] == 0).all() and (outputs["bn"][:, :, 4] != 0).any()
    with pytest.raises(SbbError, match="no weights"):
        zero_channels(nn.BatchNorm2d(4, affine=False), [0])


def test_conv_blocks_follow_the_networks() -> None:
    model = build_model(
        ArchSpec(name="dvsgesture_net", T=4, kwargs={"channels": 8}), (2, 32, 32), 11
    )
    blocks = conv_blocks(model)
    # Each block is [conv, batch norm, neuron, max pool]: the neuron follows the batch norm.
    layers = list(model.net)
    assert len(blocks) == 5
    for block in blocks:
        i = layers.index(block.conv)
        assert layers[i + 1] is block.bn and layers[i + 2] is block.neuron
    neurons = [node for _, node in model.named_neurons()]
    assert [b.neuron for b in blocks] == neurons[:5]  # the classifier's neurons are not in a block


def test_superpose_adds_frames() -> None:
    x, overlay = torch.rand(2, 4, 2, 8, 8), torch.rand(2, 4, 2, 8, 8)
    assert torch.equal(superpose(x, overlay), x + overlay)
