"""nmnist_net, the N-MNIST network of the ported papers."""

from sbb.models.architectures.voting_cnn import voting_cnn
from sbb.models.snn import ARCHITECTURES, SNNModel
from sbb.models.spec import ArchSpec


@ARCHITECTURES.register("nmnist_net")
def nmnist_net(arch: ArchSpec, input_shape: tuple[int, int, int], num_classes: int) -> SNNModel:
    """SpikingJelly's NMNISTNet: two conv blocks, a 2048-unit layer and a voting readout.

    Sneaky Spikes trains it on 2x34x34 frames with T = 16 and IF neurons (neuron model "if").
    """
    return voting_cnn(arch, input_shape, num_classes, blocks=2, hidden=2048)
