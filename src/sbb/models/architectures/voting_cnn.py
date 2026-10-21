"""The voting CNN layout of SpikingJelly's parametric_lif_net, used by the ported papers."""

from spikingjelly.activation_based import functional, layer
from torch import nn

from sbb.models.snn import SNNModel, make_neuron
from sbb.models.spec import ArchSpec
from sbb.utils.errors import SbbError


def voting_cnn(
    arch: ArchSpec,
    input_shape: tuple[int, int, int],
    num_classes: int,
    blocks: int,
    hidden: int = 512,
) -> SNNModel:
    """Build conv blocks of [conv 3x3, batch norm, neuron, max pool 2] and a spiking classifier.

    The classifier is dropout, a spiking layer of hidden units, dropout, and 10 spiking output
    neurons per class whose groups a voting layer averages, so the mean over T of the output is a
    firing rate per class. This is SpikingJelly's layout layer for layer. Only the input channels
    of the first conv, the flatten size and the number of output neurons follow the input shape and
    the number of classes. The width is arch.kwargs["channels"], 128 by default as in SpikingJelly.
    """
    P, H, W = input_shape
    if H >> blocks == 0 or W >> blocks == 0:
        raise SbbError(f"{arch.name} needs an input of at least {2**blocks}x{2**blocks}")
    channels = int(arch.kwargs.get("channels", 128))
    layers: list[nn.Module] = []
    for i in range(blocks):
        layers += [
            layer.Conv2d(P if i == 0 else channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            make_neuron(arch),
            layer.MaxPool2d(2, 2),
        ]
    net = nn.Sequential(
        *layers,
        layer.Flatten(),
        layer.Dropout(0.5),
        layer.Linear(channels * (H >> blocks) * (W >> blocks), hidden),
        make_neuron(arch),
        layer.Dropout(0.5),
        layer.Linear(hidden, 10 * num_classes),
        make_neuron(arch),
        layer.VotingLayer(10),
    )
    functional.set_step_mode(net, "m")
    return SNNModel(arch, input_shape, num_classes, net)
