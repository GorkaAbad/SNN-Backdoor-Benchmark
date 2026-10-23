import math

import pytest
import torch

from sbb.attacks import sneaky_dynamic
from sbb.attacks.dynamic_trigger import DynamicTrigger
from sbb.attacks.sneaky_dynamic import SneakyDynamic
from sbb.core.proxies import AttackerView
from sbb.core.trainer import Trainer, TrainSpec
from sbb.data.module import DataModule
from sbb.models.snn import build_model
from sbb.utils.determinism import rng_for
from sbb.utils.errors import SbbError
from tests.unit.test_core import ARCH, INFO


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


def test_dynamic_attack_keeps_the_epoch_where_both_accuracies_improve(
    datamodule: DataModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The official rule: an epoch replaces the best one only if clean and backdoor accuracy both
    # improve. The attacker scores the epochs on its own training data, never on the test set.
    scripted = iter([(0.5, 0.5), (0.9, 0.9), (0.95, 0.1)])
    monkeypatch.setattr(sneaky_dynamic, "_accuracies", lambda *args: next(scripted))
    card = SneakyDynamic.card
    trainer = Trainer(TrainSpec(epochs=3, batch_size=8), torch.device("cpu"))
    clean = build_model(ARCH, (2, 16, 16), 4)
    view = AttackerView(card, INFO, 0, datamodule, ARCH, trainer, clean)
    attack = SneakyDynamic(card.hparams_for("synthetic", "tiny_lif_cnn"), target=0)
    attack.make_model(view)
    assert attack.best_epoch == 1
