"""tiny_lif_cnn, the lite-track architecture."""

from spikingjelly.activation_based import functional, layer
from torch import nn

from sbb.models.snn import ARCHITECTURES, SNNModel, make_neuron
from sbb.models.spec import ArchSpec


@ARCHITECTURES.register("tiny_lif_cnn")
def tiny_lif_cnn(arch: ArchSpec, input_shape: tuple[int, int, int], num_classes: int) -> SNNModel:
    """Two conv blocks (16 and 32 channels) with neurons and pooling, then a linear readout."""
    P, H, W = input_shape
    net = nn.Sequential(
        layer.Conv2d(P, 16, 3, padding=1, bias=False),
        layer.BatchNorm2d(16),
        make_neuron(arch),
        layer.MaxPool2d(2),
        layer.Conv2d(16, 32, 3, padding=1, bias=False),
        layer.BatchNorm2d(32),
        make_neuron(arch),
        layer.MaxPool2d(2),
        layer.Flatten(),
        layer.Linear(32 * (H // 4) * (W // 4), num_classes),
    )
    functional.set_step_mode(net, "m")
    return SNNModel(arch, input_shape, num_classes, net)
