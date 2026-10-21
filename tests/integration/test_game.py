from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.utils.prune
from torch.utils.data import DataLoader

from sbb.attacks.frame_triggers import FlashTrigger
from sbb.attacks.labels import DirtyLabel
from sbb.attacks.no_attack import NoAttack
from sbb.attacks.selectors import RandomSelector
from sbb.attacks.static_patch import StaticPatch
from sbb.attacks.triggers import IdentityTrigger
from sbb.core.cards import AttackCard, DefenseCard
from sbb.core.game import Game
from sbb.core.plugins import (
    DataDefense,
    Defense,
    DetectionDefense,
    InputDefense,
    ModelAttack,
    ModelDefense,
    PoisoningAttack,
    TrainingDefense,
)
from sbb.core.proxies import AttackerView, DefenderView
from sbb.core.registry import ATTACKS, DEFENSES
from sbb.core.trainer import TrainerHooks, TrainState
from sbb.data.module import DataModule
from sbb.data.poison import Budget, PoisonSet
from sbb.models.checkpoint import load_checkpoint
from sbb.models.snn import SNNModel, build_model
from tests.cards import attack_card, defense_card
from tests.conftest import SCENARIO, make_cell, run_in_fresh_caches
from tests.contracts.test_plugins import check_attack, check_defense


def test_static_patch_backdoors_and_fine_tuning_runs(game: Game) -> None:
    clean = game.run(make_cell("no_attack", "no_defense"))
    attacked = game.run(make_cell("static_patch", "no_defense"))
    tuned = game.run(make_cell("static_patch", "fine_tuning"))
    assert [r.outcome for r in (clean, attacked, tuned)] == ["ok"] * 3, attacked.reason
    assert clean.metrics["ca"] > 0.9 and clean.metrics["asr"] < 0.1
    assert attacked.metrics["ca"] > 0.9 and attacked.metrics["asr"] > 0.9
    assert set(tuned.metrics) == {"ca", "asr", "ra", "asr_all"}
    assert attacked.artifacts["clean_model"] == clean.artifacts["clean_model"]
    assert attacked.artifacts["model"] == tuned.artifacts["model"]


def test_same_seed_gives_identical_results(tmp_path: Path) -> None:
    (first, first_weights), (second, second_weights) = run_in_fresh_caches(
        make_cell("static_patch", "fine_tuning"), tmp_path, torch.device("cpu")
    )
    assert first.metrics == second.metrics and first_weights == second_weights


class _Peeking(ModelDefense):
    card = defense_card(name="peeking")

    def repair(self, view: DefenderView) -> SNNModel:
        view.train_set  # noqa: B018  not declared
        return view.model


class _Sanitizer(DataDefense):
    card = defense_card(name="sanitizer", stage="pre_training", view={"resources": ["train_set"]})

    def sanitize(self, view: DefenderView) -> np.ndarray:
        return np.arange(len(view.train_set))


class _SupplyChain(NoAttack):
    card = attack_card(name="supply", kind="supply_chain", capabilities={})


@pytest.fixture
def test_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    for registry, name, cls in [
        (DEFENSES, "peeking", _Peeking),
        (DEFENSES, "sanitizer", _Sanitizer),
        (ATTACKS, "supply", _SupplyChain),
    ]:
        monkeypatch.setitem(registry._items, name, cls)


@pytest.mark.usefixtures("test_plugins")
def test_undeclared_resource_fails_the_cell(game: Game) -> None:
    result = game.run(make_cell("no_attack", "peeking"))
    assert result.outcome == "failed" and "ThreatModelViolation" in (result.reason or "")
    assert result.metrics == {}


@pytest.mark.usefixtures("test_plugins")
def test_data_defense_against_supply_chain_is_reported_not_applicable(game: Game) -> None:
    result = game.run(make_cell("supply", "sanitizer"))
    assert result.outcome == "not_applicable" and "train_set" in (result.reason or "")
    assert isinstance(_Sanitizer.card, DefenseCard) and isinstance(_SupplyChain.card, AttackCard)


class _LabelFlip(PoisoningAttack):
    card = attack_card(name="label_flip")

    def __init__(self, hparams: dict[str, object], target: int) -> None:
        super().__init__(hparams, target)
        self.trigger, self.selector, self.labels = IdentityTrigger(), RandomSelector(), DirtyLabel()


def test_attack_that_destroys_clean_accuracy_is_invalid(
    game: Game, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(ATTACKS._items, "label_flip", _LabelFlip)
    budget = SCENARIO.budget.model_copy(update={"poison_rate": 0.5})
    result = game.run(
        make_cell(
            "label_flip", "no_defense", scenario=SCENARIO.model_copy(update={"budget": budget})
        )
    )
    assert result.outcome == "invalid_attack", result.reason
    assert result.metrics["ca_backdoored"] < result.metrics["ca_clean"] - 0.03


class _Injection(ModelAttack):
    """Poisons the clean split with the static patch, then trains on it (test-only attack)."""

    card = attack_card(name="supply_injection", kind="supply_chain", capabilities={})
    received: dict[str, torch.Tensor] = {}  # the clean weights the last make_model was given

    def __init__(self, hparams: dict[str, object], target: int) -> None:
        super().__init__(hparams, target)
        hparams = StaticPatch.card.hparams_for("synthetic", "tiny_lif_cnn")
        self.poisoner = StaticPatch(hparams, target)
        self.trigger = self.poisoner.trigger

    def _poisoned(self, view: AttackerView) -> DataLoader[Any]:
        budget = Budget(poison_rate=0.1, target=self.target)
        poison = self.poisoner.poison(view, budget, np.empty(0, dtype=np.int64))
        return view.poisoned_loader(poison, batch_size=32)

    def make_model(self, view: AttackerView) -> SNNModel:
        model = view.clean_model
        _Injection.received = {k: v.clone() for k, v in model.state_dict().items()}
        return view.trainer.fit(model, self._poisoned(view), epochs=3)


class _TrainingControl(_Injection):
    card = attack_card(name="trains_own_model", kind="training_control", capabilities={})

    def make_model(self, view: AttackerView) -> SNNModel:
        model = build_model(view.arch, view.scenario.input_shape, view.scenario.num_classes)
        return view.trainer.fit(model, self._poisoned(view))


class _OtherSpec(_Injection):
    card = attack_card(name="other_spec", kind="supply_chain", capabilities={})

    def make_model(self, view: AttackerView) -> SNNModel:
        wider = view.arch.model_copy(update={"kwargs": {"unused": 1}})
        return build_model(wider, view.scenario.input_shape, view.scenario.num_classes)


class _OtherClasses(_Injection):
    card = attack_card(name="other_classes", kind="supply_chain", capabilities={})

    def make_model(self, view: AttackerView) -> SNNModel:
        return build_model(view.arch, view.scenario.input_shape, view.scenario.num_classes + 1)


class _LowThreshold(_Injection):
    """Changes a neuron parameter in place, which a checkpoint cannot keep."""

    card = attack_card(name="low_threshold", kind="supply_chain", capabilities={})

    def make_model(self, view: AttackerView) -> SNNModel:
        model = view.clean_model
        for _, node in model.named_neurons():
            node.v_threshold = 0.05
        return model


class _Pruned(_Injection):
    """Changes the state-dict layout (weight_orig and weight_mask), which would not load."""

    card = attack_card(name="pruned", kind="supply_chain", capabilities={})

    def make_model(self, view: AttackerView) -> SNNModel:
        model = view.clean_model
        torch.nn.utils.prune.l1_unstructured(model.net[0], "weight", amount=0.5)
        return model


class _Hybrid(PoisoningAttack, ModelAttack):
    """A supply-chain attack that reuses PoisoningAttack.poison, as a fine-tuning port could."""

    card = attack_card(name="hybrid", kind="supply_chain", capabilities={})
    made = 0

    def __init__(self, hparams: dict[str, object], target: int) -> None:
        super().__init__(hparams, target)
        self.trigger, self.selector, self.labels = IdentityTrigger(), RandomSelector(), DirtyLabel()

    def make_model(self, view: AttackerView) -> SNNModel:
        _Hybrid.made += 1
        return view.clean_model


MODEL_ATTACKS = (
    _Injection,
    _TrainingControl,
    _OtherSpec,
    _OtherClasses,
    _LowThreshold,
    _Pruned,
    _Hybrid,
)


@pytest.fixture
def model_attacks(monkeypatch: pytest.MonkeyPatch) -> None:
    for cls in MODEL_ATTACKS:
        monkeypatch.setitem(ATTACKS._items, cls.card.name, cls)


@pytest.mark.parametrize("cls", [_Injection, _TrainingControl])
def test_test_model_attacks_meet_the_attack_contract(
    cls: type[ModelAttack], datamodule: DataModule
) -> None:
    check_attack(cls, datamodule)


@pytest.mark.usefixtures("model_attacks")
@pytest.mark.parametrize("attack", ["supply_injection", "trains_own_model"])
def test_model_attacks_backdoor_the_victim(game: Game, attack: str) -> None:
    result = game.run(make_cell(attack, "no_defense"))
    assert result.outcome == "ok", result.reason
    assert result.metrics["ca"] > 0.9 and result.metrics["asr"] > 0.9
    assert set(result.artifacts) == {"clean_model", "model", "defended_model"}
    assert result.artifacts["model"] != result.artifacts["clean_model"]


@pytest.mark.usefixtures("model_attacks")
def test_supply_chain_attacker_receives_the_trained_clean_model(game: Game) -> None:
    result = game.run(make_cell("supply_injection", "no_defense"))
    clean = load_checkpoint(game.cache.local.root / result.artifacts["clean_model"])
    received = _Injection.received
    assert received.keys() == clean.state_dict().keys()
    assert all(torch.equal(received[k], v) for k, v in clean.state_dict().items())


@pytest.mark.usefixtures("model_attacks")
def test_model_attack_key_follows_the_target(game: Game) -> None:
    first = game.run(make_cell("supply_injection", "no_defense"))
    other_defense = game.run(make_cell("supply_injection", "fine_tuning"))
    budget = SCENARIO.budget.model_copy(update={"target": 1})
    other_target = game.run(
        make_cell(
            "supply_injection",
            "no_defense",
            scenario=SCENARIO.model_copy(update={"budget": budget}),
        )
    )
    assert first.artifacts["model"] == other_defense.artifacts["model"]  # one victim, two defenses
    assert other_target.artifacts["model"] != first.artifacts["model"]


@pytest.mark.usefixtures("model_attacks")
@pytest.mark.parametrize("attack", ["other_spec", "other_classes", "low_threshold", "pruned"])
def test_model_attack_must_return_the_scenario_architecture(game: Game, attack: str) -> None:
    result = game.run(make_cell(attack, "no_defense"))
    assert result.outcome == "failed" and "ThreatModelViolation" in (result.reason or "")
    assert not list(game.cache.local.root.glob(".staging-*"))  # nothing was cached


@pytest.mark.usefixtures("model_attacks")
def test_game_plays_the_card_kind_not_the_class(game: Game) -> None:
    result = game.run(make_cell("hybrid", "no_defense"))
    assert result.outcome == "ok", result.reason
    assert _Hybrid.made == 1 and "poison_set" not in result.artifacts


class _DropLabel0(DataDefense):
    """Keeps the training samples not labelled 0, which drops every dirty-label poison."""

    card = defense_card(
        name="drop_label_0", stage="pre_training", view={"resources": ["train_set"]}
    )

    def sanitize(self, view: DefenderView) -> np.ndarray:
        return np.flatnonzero(view.train_set.labels() != 0)


class _MaskLabel0(TrainerHooks):
    def loss(self, state: TrainState, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert state.per_sample_loss is not None
        keep = (targets != 0).float()
        return (state.per_sample_loss * keep).sum() / keep.sum().clamp(min=1.0)


class _IgnoreLabel0(TrainingDefense):
    """Leaves label 0 out of the training loss, so the poison never trains."""

    card = defense_card(
        name="ignore_label_0",
        stage="in_training",
        view={"resources": ["train_set", "training_loop"]},
    )

    def hooks(self, view: DefenderView) -> TrainerHooks:
        return _MaskLabel0()


class _PatchDetector(InputDefense):
    """Scores the events in the 3x3 corner where static_patch puts its square."""

    card = defense_card(
        name="patch_detector",
        stage="inference",
        view={"resources": ["model", "clean_subset"], "clean_fraction": 0.05},
    )

    def fit(self, view: DefenderView) -> None:
        x, _, _ = next(iter(view.clean_loader(batch_size=1000)))
        self.threshold = float(self.score(x).max())

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :, :3, :3].sum(dim=(1, 2, 3, 4))


class _WeightNorm(DetectionDefense):
    card = defense_card(name="weight_norm", stage="model_detection", view={"resources": ["model"]})
    calls = 0

    def verdict(self, view: DefenderView) -> float:
        _WeightNorm.calls += 1
        return float(sum(p.detach().abs().sum() for p in view.model.parameters()))


STAGE_DEFENSES = (_DropLabel0, _IgnoreLabel0, _PatchDetector, _WeightNorm)


@pytest.fixture
def stage_defenses(monkeypatch: pytest.MonkeyPatch) -> None:
    for cls in STAGE_DEFENSES:
        monkeypatch.setitem(DEFENSES._items, cls.card.name, cls)


@pytest.mark.parametrize("cls", STAGE_DEFENSES)
def test_test_defenses_meet_the_defense_contract(
    cls: type[Defense], datamodule: DataModule
) -> None:
    check_defense(cls, datamodule)


@pytest.mark.usefixtures("stage_defenses")
@pytest.mark.parametrize("defense", ["drop_label_0", "ignore_label_0"])
def test_retraining_defenses_retrain_the_victim(game: Game, defense: str) -> None:
    undefended = game.run(make_cell("static_patch", "no_defense"))
    defended = game.run(make_cell("static_patch", defense))
    assert defended.outcome == "ok", defended.reason
    assert undefended.metrics["asr"] > 0.9 and defended.metrics["asr"] < 0.1
    assert defended.artifacts["model"] == undefended.artifacts["model"]
    assert defended.artifacts["defended_model"] != defended.artifacts["model"]
    assert set(defended.metrics) == {"ca", "asr", "ra", "asr_all"}


@pytest.mark.usefixtures("stage_defenses")
def test_input_defense_rejects_triggered_inputs(game: Game) -> None:
    result = game.run(make_cell("static_patch", "patch_detector"))
    assert result.outcome == "ok", result.reason
    metrics = result.metrics
    assert set(metrics) == {"ca", "asr", "ra", "asr_all", "frr", "tpr", "auroc"}
    assert metrics["tpr"] > 0.9 and metrics["auroc"] > 0.9 and metrics["frr"] < 0.2
    assert metrics["asr"] < 0.1  # rejected triggered inputs do not count as successes
    assert "defended_model" not in result.artifacts  # the model itself is unchanged


@pytest.mark.usefixtures("stage_defenses")
def test_detection_defense_records_a_cached_verdict(game: Game) -> None:
    _WeightNorm.calls = 0
    first = game.run(make_cell("static_patch", "weight_norm"))
    again = game.run(make_cell("static_patch", "weight_norm"))
    undefended = game.run(make_cell("static_patch", "no_defense"))
    assert first.outcome == "ok", first.reason
    assert first.metrics == again.metrics and _WeightNorm.calls == 1
    assert first.metrics["verdict"] > 0
    assert {k: first.metrics[k] for k in ("ca", "asr", "ra", "asr_all")} == undefended.metrics


class _KeepAll(DataDefense):
    """Keeps every sample, so the retrained victim is the undefended one."""

    card = defense_card(name="keep_all", stage="pre_training", view={"resources": ["train_set"]})

    def sanitize(self, view: DefenderView) -> np.ndarray:
        return np.arange(len(view.train_set))


class _NoHooks(TrainingDefense):
    """Hooks that change nothing, so the retrained victim is the undefended one."""

    card = defense_card(
        name="no_hooks", stage="in_training", view={"resources": ["train_set", "training_loop"]}
    )

    def hooks(self, view: DefenderView) -> TrainerHooks:
        return TrainerHooks()


class _Duplicates(DataDefense):
    card = defense_card(name="duplicates", stage="pre_training", view={"resources": ["train_set"]})

    def sanitize(self, view: DefenderView) -> np.ndarray:
        return np.array([0, 0, 1])


class _NaNVerdict(DetectionDefense):
    card = defense_card(name="nan_verdict", stage="model_detection", view={"resources": ["model"]})

    def verdict(self, view: DefenderView) -> float:
        return float("nan")


@pytest.fixture
def neutral_defenses(monkeypatch: pytest.MonkeyPatch) -> None:
    for cls in (_KeepAll, _NoHooks, _Duplicates, _NaNVerdict):
        monkeypatch.setitem(DEFENSES._items, cls.card.name, cls)


@pytest.mark.usefixtures("neutral_defenses")
@pytest.mark.parametrize("defense", ["keep_all", "no_hooks"])
def test_retraining_uses_the_poisoned_split(game: Game, defense: str) -> None:
    # Retraining on what the victim received, with nothing removed, gives back the backdoor.
    result = game.run(make_cell("static_patch", defense))
    assert result.outcome == "ok", result.reason
    assert result.metrics["asr"] > 0.9


@pytest.mark.usefixtures("neutral_defenses")
def test_data_defense_positions_are_checked(game: Game) -> None:
    result = game.run(make_cell("static_patch", "duplicates"))
    assert result.outcome == "failed" and "unique positions" in (result.reason or "")


@pytest.mark.usefixtures("neutral_defenses")
def test_a_verdict_must_be_finite(game: Game) -> None:
    result = game.run(make_cell("static_patch", "nan_verdict"))
    assert result.outcome == "failed" and "finite" in (result.reason or "")
    assert not list(game.cache.local.root.rglob("verdict.json"))


class _Flash(PoisoningAttack):
    """Dirty-label poisoning with a flash of ON events in the first frame (test-only attack)."""

    card = attack_card(name="flash", capabilities={"labels": "dirty", "trigger_domain": "frames"})

    def __init__(self, hparams: dict[str, object], target: int) -> None:
        super().__init__(hparams, target)
        self.trigger = FlashTrigger(polarity=1, length=1)
        self.selector, self.labels = RandomSelector(), DirtyLabel()


class _FlashCalledRaw(_Flash):
    """Declares raw_events, but its trigger acts on frames."""

    card = attack_card(name="flash_called_raw")


class _RawTriggerCalledFrames(_Flash):
    """Declares frames, but its trigger acts on raw events."""

    card = attack_card(
        name="raw_called_frames", capabilities={"labels": "dirty", "trigger_domain": "frames"}
    )

    def __init__(self, hparams: dict[str, object], target: int) -> None:
        super().__init__(hparams, target)
        self.trigger = IdentityTrigger()


class _SwapsItsTrigger(StaticPatch):
    """Declares raw_events and passes the first check, then swaps in a frame trigger."""

    card = attack_card(name="swaps_its_trigger", hparams=StaticPatch.card.hparams)

    def poison(self, view: AttackerView, budget: Budget, pool: np.ndarray) -> PoisonSet:
        poison = super().poison(view, budget, pool)
        self.trigger = FlashTrigger(polarity=1, length=1)
        return poison


class _FramesBehindARawCard(PoisoningAttack):
    """Declares raw_events, but its poison set edits frames anyway."""

    card = attack_card(name="frames_behind_a_raw_card")

    def __init__(self, hparams: dict[str, object], target: int) -> None:
        super().__init__(hparams, target)
        self.trigger, self.selector, self.labels = IdentityTrigger(), RandomSelector(), DirtyLabel()

    def poison(self, view: AttackerView, budget: Budget, pool: np.ndarray) -> PoisonSet:
        poison = super().poison(view, budget, pool)
        return PoisonSet(poison.indices, poison.samples, poison.labels, lambda x, rng: x + 1)


@pytest.fixture
def frame_attacks(monkeypatch: pytest.MonkeyPatch) -> None:
    for cls in (
        _Flash,
        _FlashCalledRaw,
        _RawTriggerCalledFrames,
        _SwapsItsTrigger,
        _FramesBehindARawCard,
    ):
        monkeypatch.setitem(ATTACKS._items, cls.card.name, cls)


def test_frame_attack_meets_the_attack_contract(datamodule: DataModule) -> None:
    check_attack(_Flash, datamodule)


@pytest.mark.usefixtures("frame_attacks")
def test_frame_trigger_backdoors_the_victim(game: Game) -> None:
    result = game.run(make_cell("flash", "no_defense"))
    assert result.outcome == "ok", result.reason
    assert result.metrics["ca"] > 0.9 and result.metrics["asr"] > 0.9


@pytest.mark.usefixtures("frame_attacks")
@pytest.mark.parametrize(
    ("attack", "message"),
    [
        ("flash_called_raw", "but its trigger is a FlashTrigger"),
        ("raw_called_frames", "but its trigger is a IdentityTrigger"),
        ("swaps_its_trigger", "but its trigger is a FlashTrigger"),  # checked again before step 8
        ("frames_behind_a_raw_card", "edits frames"),
    ],
)
def test_trigger_must_act_in_the_declared_domain(game: Game, attack: str, message: str) -> None:
    result = game.run(make_cell(attack, "no_defense"))
    assert result.outcome == "failed", result.reason
    assert "ThreatModelViolation" in (result.reason or "") and message in (result.reason or "")


class _RecordsHparams(StaticPatch):
    card = attack_card(name="records_hparams", hparams=StaticPatch.card.hparams)
    seen: dict[str, Any] = {}

    def __init__(self, hparams: dict[str, Any], target: int) -> None:
        super().__init__(hparams, target)
        _RecordsHparams.seen = hparams


def test_cell_hparams_override_the_card(game: Game, monkeypatch: pytest.MonkeyPatch) -> None:
    # Replications run the paper's hyperparameters, given in the cell over the card's.
    monkeypatch.setitem(ATTACKS._items, "records_hparams", _RecordsHparams)
    plain = game.run(make_cell("records_hparams", "no_defense"))
    cell = make_cell("records_hparams", "no_defense").model_copy(
        update={"attack_hparams": {"trigger": {"size": 2}}}
    )
    changed = game.run(cell)
    card = StaticPatch.card.hparams["default"]["trigger"]
    assert _RecordsHparams.seen["trigger"] == {**card, "size": 2}
    assert changed.artifacts["model"] != plain.artifacts["model"]
    assert changed.key != plain.key


@pytest.mark.parametrize(
    ("defense", "extra"),
    [
        ("fine_pruning", set()),
        ("clp", set()),
        ("strip", {"frr", "tpr", "auroc"}),
        ("tmpbd", {"verdict"}),
    ],
)
def test_ported_defenses_run_in_the_game(game: Game, defense: str, extra: set[str]) -> None:
    result = game.run(make_cell("static_patch", defense))
    assert result.outcome == "ok", result.reason
    assert set(result.metrics) == {"ca", "asr", "ra", "asr_all"} | extra
