"""The adaptive attacks TaCT, Adap-Blend and Adap-Patch, and their shared parts."""

import numpy as np
import pytest
import torch

from sbb.attacks.frame_triggers import BlendTrigger, PatchesTrigger
from sbb.attacks.selectors import ClassSelector
from sbb.core.plugins import PoisoningAttack
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
LABELS = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])


def test_class_selector_picks_only_its_classes() -> None:
    picked = ClassSelector((1, 2)).select(LABELS, 10, 0, np.array([1]), rng_for(0, 0))
    assert sorted(picked.tolist()) == [2, 4, 5, 7, 8]
    assert len(ClassSelector((1,)).select(LABELS, 2, 0, np.array([]), rng_for(0, 0))) == 2


def test_blend_mixes_one_pattern_scaled_to_the_peak_into_every_frame() -> None:
    frames = torch.rand(3, 2, 8, 8) * 4
    out = BlendTrigger(alpha=0.2).apply(frames, RNG)
    pattern = (out - 0.8 * frames) / 0.2
    assert torch.allclose(pattern, pattern[:1].expand_as(pattern), atol=1e-4)
    peak = float(frames.max())
    on = torch.isclose(pattern, torch.tensor(peak), atol=1e-4)
    assert (on | torch.isclose(pattern, torch.tensor(0.0), atol=1e-4)).all() and on.any()


def test_blend_for_training_applies_half_of_the_pieces() -> None:
    frames = torch.ones(1, 2, 8, 8)
    frames[0, 0, 0, 0] = 2.0  # the peak, so the pattern is 0 or 2 and an untouched pixel stays 1
    out = BlendTrigger(alpha=1.0, grid=4, keep=0.5).apply(frames, RNG)
    kept = [
        bool((out[..., r : r + 2, c : c + 2] != 1).all())
        for r in range(0, 8, 2)
        for c in range(0, 8, 2)
    ]
    assert sum(kept) == 8


def test_patches_blend_each_square_with_its_opacity() -> None:
    patches = (
        {"size": 0.25, "position": "top-left", "polarity": 1, "alpha": 0.5},
        {"size": 0.25, "position": "bottom-right", "polarity": 2, "alpha": 1.0},
    )
    frames = torch.full((2, 2, 8, 8), 3.0)
    out = PatchesTrigger(patches).apply(frames, RNG)
    assert (out[:, 0, :2, :2] == 1.5).all() and (out[:, 1, :2, :2] == 2.0).all()
    assert (out[:, 0, 6:, 6:] == 1.0).all() and (out[:, 1, 6:, 6:] == 0.0).all()
    assert int((out != 3.0).sum()) == 2 * 2 * 4 * 2  # steps x channels x pixels x patches
    one = PatchesTrigger(patches, one=True).apply(frames, RNG)
    assert [(one[..., :2, :2] != 3).any().item(), (one[..., 6:, 6:] != 3).any().item()].count(
        True
    ) == 1


def _poison(name: str, datamodule: DataModule) -> tuple[PoisoningAttack, np.ndarray, np.ndarray]:
    cls = ATTACKS.get(name)
    attack = cls(cls.card.hparams_for("synthetic", "tiny_lif_cnn"), target=0)
    assert isinstance(attack, PoisoningAttack)
    trainer = Trainer(TrainSpec(epochs=1, batch_size=32), torch.device("cpu"))
    view = AttackerView(
        cls.card, INFO, 0, datamodule, ARCH, trainer, build_model(ARCH, (2, 16, 16), 4)
    )
    poison = attack.poison(view, Budget(), datamodule.defender_pool)
    return attack, poison.labels, datamodule.labels[poison.indices]


@pytest.mark.parametrize("name", ["tact", "adap_blend", "adap_patch"])
def test_payload_samples_go_to_the_target_and_cover_samples_keep_their_label(
    name: str, datamodule: DataModule
) -> None:
    attack, labels, true = _poison(name, datamodule)
    payload = labels == 0
    assert (true[payload] != 0).all() and np.array_equal(labels[~payload], true[~payload])
    n = Budget().max_poison(480)
    assert (~payload).sum() == round(attack.hparams["eta"] * n) and payload.sum() == n - (
        ~payload
    ).sum()


def test_tact_poisons_its_source_class_and_covers_with_its_cover_classes(
    datamodule: DataModule,
) -> None:
    attack, labels, true = _poison("tact", datamodule)
    assert set(true[labels == 0].tolist()) == {1} and set(true[labels != 0].tolist()) <= {2, 3}
    assert attack.source_classes == (1,)
    with pytest.raises(SbbError):
        ATTACKS.get("tact")(attack.hparams, target=1)  # the target is the source class
