"""cifar10dvs_net, the CIFAR10-DVS network of the ported papers."""

from sbb.models.architectures.voting_cnn import voting_cnn
from sbb.models.snn import ARCHITECTURES, SNNModel
from sbb.models.spec import ArchSpec


@ARCHITECTURES.register("cifar10dvs_net")
def cifar10dvs_net(arch: ArchSpec, input_shape: tuple[int, int, int], num_classes: int) -> SNNModel:
    """SpikingJelly's CIFAR10DVSNet: four conv blocks and a voting readout.

    Sneaky Spikes, Flashy Backdoor, TMPBD and T-Backdoor train it on 2x128x128 frames.
    """
    return voting_cnn(arch, input_shape, num_classes, blocks=4)
