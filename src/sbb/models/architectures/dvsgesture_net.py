"""dvsgesture_net, the DVS128 Gesture network of the ported papers."""

from sbb.models.architectures.voting_cnn import voting_cnn
from sbb.models.snn import ARCHITECTURES, SNNModel
from sbb.models.spec import ArchSpec


@ARCHITECTURES.register("dvsgesture_net")
def dvsgesture_net(arch: ArchSpec, input_shape: tuple[int, int, int], num_classes: int) -> SNNModel:
    """SpikingJelly's DVSGestureNet: five conv blocks and a voting readout.

    Sneaky Spikes, Flashy Backdoor and TMPBD train it on 2x128x128 frames with T = 16.
    """
    return voting_cnn(arch, input_shape, num_classes, blocks=5)
