import math

import pytest
import torch

from sbb.attacks.dynamic_trigger import DynamicTrigger
from sbb.utils.determinism import rng_for
from sbb.utils.errors import SbbError


def test_dynamic_trigger_adds_a_bounded_spiking_perturbation() -> None:
    torch.manual_seed(0)
    trigger = DynamicTrigger(gamma=0.01, channels=4)
    frames = torch.rand(4, 2, 16, 16) * 3
    out = trigger.apply(frames, rng_for(0, 0))
    # The generator ends in spiking neurons and tanh, so each entry moves by 0 or gamma tanh(1).
    steps = torch.unique(torch.round((out - frames) / (0.01 * math.tanh(1)), decimals=4))
    assert out.shape == frames.shape and set(steps.tolist()) <= {0.0, 1.0}
    assert not trigger.generator.training, "the trigger generates in eval mode"


def test_dynamic_trigger_keeps_the_frame_size() -> None:
    # The official Autoencoder needs sides that are multiples of 16, AutoencoderMNIST is for 34.
    with pytest.raises(SbbError, match="frame size"):
        DynamicTrigger(gamma=0.01, channels=2).apply(torch.rand(4, 2, 34, 34), rng_for(0, 0))
    mnist = DynamicTrigger(gamma=0.01, channels=2, autoencoder="mnist")
    assert mnist.apply(torch.rand(4, 2, 34, 34), rng_for(0, 0)).shape == (4, 2, 34, 34)
