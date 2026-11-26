"""The clean-label attacks of group D (docs/PLAN.md section 13.5, v3)."""

import math

import pytest
import torch

from sbb.attacks.frame_triggers import SigTrigger
from sbb.core.proxies import AttackerView
from sbb.core.registry import ATTACKS
from sbb.core.trainer import Trainer, TrainSpec
from sbb.data.module import DataModule
from sbb.data.poison import Budget
from sbb.models.snn import build_model
from sbb.utils.determinism import rng_for
from sbb.utils.errors import SbbError
from tests.contracts.test_plugins import ARCH, INFO

RNG = rng_for(0, 0)


def test_sig_adds_one_sinusoid_over_the_frames_to_every_pixel() -> None:
    frames = torch.rand(8, 2, 4, 4) * 4 + 1  # at least 1 everywhere, so nothing clips
    out = SigTrigger(delta=0.2, frequency=2).apply(frames, RNG)
    signal = 0.2 * float(frames.max()) * torch.sin(2 * math.pi * 2 * torch.arange(8) / 8)
    assert torch.allclose(out - frames, signal.view(8, 1, 1, 1).expand_as(frames), atol=1e-5)


def test_sig_clips_at_zero_events() -> None:
    frames = torch.zeros(8, 2, 4, 4)
    frames[0, 0, 0, 0] = 10.0  # the peak, so the amplitude is 5
    out = SigTrigger(delta=0.5, frequency=2).apply(frames, RNG)
    assert torch.allclose(out[1], torch.full((2, 4, 4), 5.0))  # sin(pi / 2) = 1
    assert torch.equal(out[3], torch.zeros(2, 4, 4))  # sin(3 pi / 2) = -1, clipped


def test_sig_refuses_a_frequency_the_frames_cannot_carry() -> None:
    with pytest.raises(SbbError, match="frequency"):  # 5 periods over 10 frames: sin(pi t) = 0
        SigTrigger(delta=0.1, frequency=5).apply(torch.zeros(10, 2, 4, 4), RNG)


def test_sig_poisons_only_the_target_class(datamodule: DataModule) -> None:
    cls = ATTACKS.get("sig")
    attack = cls(cls.card.hparams_for("synthetic", "tiny_lif_cnn"), target=1)
    trainer = Trainer(TrainSpec(epochs=1, batch_size=32), torch.device("cpu"))
    view = AttackerView(
        cls.card, INFO, 0, datamodule, ARCH, trainer, build_model(ARCH, (2, 16, 16), 4)
    )
    poison = attack.poison(view, Budget(), datamodule.defender_pool)
    assert len(poison.indices) > 0
    assert (datamodule.labels[poison.indices] == 1).all() and (poison.labels == 1).all()
