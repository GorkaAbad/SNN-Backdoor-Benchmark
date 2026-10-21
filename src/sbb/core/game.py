"""The security game that evaluates one cell (docs/PLAN.md sections 3.1 and 4.7)."""

import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from sbb.core.cards import ScenarioKind, Stage, compatible
from sbb.core.plugins import (
    STAGE_BASES,
    Attack,
    DataDefense,
    Defense,
    DetectionDefense,
    EvalContext,
    FrameTrigger,
    InputDefense,
    ModelAttack,
    ModelDefense,
    PoisoningAttack,
    TrainingDefense,
)
from sbb.core.proxies import AttackerView, DefenderView, ScenarioInfo, TrainSetView
from sbb.core.registry import ATTACKS, DEFENSES, METRICS
from sbb.core.result import CellResult, Outcome, collect_provenance
from sbb.core.scenario import Cell, Scenario, artifact_key
from sbb.core.trainer import Trainer
from sbb.data.module import DataModule
from sbb.data.poison import PoisonSet
from sbb.data.representations import FrameRepr
from sbb.data.sample import DATASETS
from sbb.data.splits import load_split, reserve_defender_pool
from sbb.models.checkpoint import load_checkpoint, save_checkpoint
from sbb.models.snn import SNNModel, build_model
from sbb.utils.determinism import seed_everything
from sbb.utils.errors import SbbError, ThreatModelViolation
from sbb.utils.storage import ArtifactCache

log = logging.getLogger(__name__)


@dataclass
class _Report:
    outcome: Outcome
    reason: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)


class Game:
    """Plays the security game for one cell. The only place that seeds and trains victims.

    Args:
        cache: Cache of frames and models.
        device: Device for training and evaluation.
        data_root: Directory with one subdirectory per dataset.
        site: SiteProfile name, recorded in the provenance.
        executor: Executor name, recorded in the provenance.
    """

    def __init__(
        self,
        cache: ArtifactCache,
        device: torch.device,
        data_root: Path,
        site: str = "local",
        executor: str = "in-process",
    ) -> None:
        self.cache = cache
        self.device = device
        self.data_root = data_root
        self.site = site
        self.executor = executor

    def run(self, cell: Cell) -> CellResult:
        """Evaluate one cell. Errors become a result with outcome "failed"."""
        started_at, start = datetime.now(UTC), time.perf_counter()
        try:
            report = self._play(cell)
        except Exception as error:
            log.exception("cell %s failed", cell.key())
            report = _Report("failed", f"{type(error).__name__}: {error}")
        provenance = collect_provenance(
            self.device, self.site, self.executor, started_at, time.perf_counter() - start
        )
        return CellResult(
            key=cell.key(),
            cell=cell,
            outcome=report.outcome,
            reason=report.reason,
            metrics=report.metrics,
            artifacts=report.artifacts,
            provenance=provenance,
        )

    def _play(self, cell: Cell) -> _Report:
        scenario, seed = cell.scenario, cell.seed
        budget, bs = scenario.budget, scenario.train.batch_size
        attack_cls, defense_cls = ATTACKS.get(cell.attack), DEFENSES.get(cell.defense)
        _check_version(cell.attack, attack_cls.card.version, cell.attack_version)
        _check_version(cell.defense, defense_cls.card.version, cell.defense_version)

        # 1. Compatibility
        ok, reasons = compatible(attack_cls.card, defense_cls.card, budget)
        if not ok:
            return _Report("not_applicable", "; ".join(reasons))
        kind = attack_cls.card.kind
        if kind == ScenarioKind.FEDERATED:
            raise NotImplementedError("the federated scenario kind lands in v3")
        poisoning = kind == ScenarioKind.DATA_POISONING
        base = PoisoningAttack if poisoning else ModelAttack
        if not issubclass(attack_cls, base):
            raise SbbError(f"{cell.attack} is a {kind} attack, so it must subclass {base.__name__}")
        stage = defense_cls.card.stage
        if not issubclass(defense_cls, STAGE_BASES[stage]):
            raise SbbError(
                f"{cell.defense} is a {stage} defense, so it must subclass "
                f"{STAGE_BASES[stage].__name__}"
            )

        # 2. Data and defender pool
        seed_everything(seed)
        data = self._prepare_data(scenario, seed)
        dataset = data.dataset
        trainer = Trainer(scenario.train, self.device)
        info = ScenarioInfo(
            dataset=dataset.name,
            sensor_size=dataset.sensor_size,
            num_classes=dataset.num_classes,
            T=scenario.repr.T,
            input_shape=data.input_shape,
        )

        # 3. Clean reference
        clean_key = artifact_key(data=scenario.data, **_recipe(cell))
        clean = self._model(
            clean_key,
            seed,
            lambda: trainer.fit(_fresh_model(scenario, data), data.train_loader(None, bs)),
        )

        # 4 and 5. Attack and victim
        attack_hparams = attack_cls.card.hparams_for(
            dataset.name, scenario.arch.name, cell.attack_hparams
        )
        attack = attack_cls(attack_hparams, budget.target)
        _check_domain(cell.attack, attack)
        attacker_view = AttackerView(
            attack_cls.card,
            info,
            seed,
            data,
            scenario.arch,
            Trainer(scenario.train, self.device),
            clean,
        )
        setting = _Setting(cell, data, trainer, clean, clean_key)
        poison: PoisonSet | None = None
        if poisoning:  # the card kind decides, even for a class that is both
            assert isinstance(attack, PoisoningAttack)
            victim, poison, keys = self._poisoned_victim(
                setting, attack, attack_hparams, attacker_view
            )
        else:
            assert isinstance(attack, ModelAttack)
            victim, keys = self._attacker_model(setting, attack, attack_hparams, attacker_view)
        artifacts = {"clean_model": clean_key, **keys}

        # 6. Attack validity
        test = data.test_loader(bs)
        ca_clean = trainer.evaluate(clean, test)["accuracy"]
        ca_victim = trainer.evaluate(victim, test)["accuracy"]
        if ca_victim < ca_clean - budget.delta_pp / 100:
            reason = f"CA dropped from {ca_clean:.4f} to {ca_victim:.4f}"
            metrics = {"ca_clean": ca_clean, "ca_backdoored": ca_victim}
            return _Report("invalid_attack", reason, metrics, artifacts)

        # 7. Defense
        defense_hparams = defense_cls.card.hparams_for(dataset.name, scenario.arch.name)
        defense = defense_cls(defense_hparams)
        defender_view = DefenderView(
            defense_cls.card,
            info,
            seed,
            Trainer(scenario.train, self.device),
            victim,
            TrainSetView(data, poison),
            data,
        )
        defense_key = artifact_key(
            model_key=artifacts["model"],
            defense=cell.defense,
            defense_version=cell.defense_version,
            hparams=defense_hparams,
            seed=seed,
        )
        defended = self._defend(setting, defense, defender_view, victim, poison, defense_key)
        artifacts.update(defended.artifacts)

        # 8. Metrics on the clean and triggered test sets
        _check_domain(cell.attack, attack)  # again, the attack may have replaced its trigger
        if isinstance(attack.trigger, FrameTrigger):
            triggered = data.triggered_loader(bs, frames=attack.trigger.apply)
        else:
            triggered = data.triggered_loader(bs, events=attack.trigger.apply)
        clean_pred, clean_true = trainer.predict(defended.model, test)
        triggered_pred, triggered_true = trainer.predict(defended.model, triggered)
        rejection: dict[str, np.ndarray] = {}
        if defended.detector is not None:
            # ponytail: scoring runs the loaders a second time. Merge it into predict if it is slow.
            detector = defended.detector
            clean_scores, triggered_scores = (
                self._scores(detector, test),
                self._scores(detector, triggered),
            )
            rejection = {
                "clean_scores": clean_scores,
                "triggered_scores": triggered_scores,
                "clean_accepted": clean_scores <= detector.threshold,
                "triggered_accepted": triggered_scores <= detector.threshold,
            }
        ctx = EvalContext(
            clean_pred, clean_true, triggered_pred, triggered_true, budget.target, **rejection
        )
        metrics = {
            name: metric().compute(ctx)
            for name in METRICS.names()
            if stage in (metric := METRICS.get(name)).stages
        }
        metrics.update(defended.metrics)
        return _Report("ok", None, metrics, artifacts)

    def _poisoned_victim(
        self,
        setting: "_Setting",
        attack: PoisoningAttack,
        hparams: dict[str, Any],
        view: AttackerView,
    ) -> tuple[SNNModel, PoisonSet, dict[str, str]]:
        """Data poisoning: the attack builds a poison set and the harness trains the victim."""
        cell, data = setting.cell, setting.data
        budget = cell.scenario.budget
        # ponytail: the PoisonSet is rebuilt in every cell and only its key is recorded, although
        # PLAN 4.8 rule 9 caches every stage. On N-MNIST that costs 30 to 40 s on 4 CPU cores per
        # static_patch cell (6 000 samples). Cache it with learned triggers in v1: a cached set
        # skips poison(), so trigger.fit too.
        poison = attack.poison(view, budget, data.defender_pool)
        if poison.frame_trigger is not None and attack.card.capabilities.trigger_domain != "frames":
            raise ThreatModelViolation(
                f"{cell.attack} edits frames, but its card declares trigger_domain raw_events"
            )
        label_policy = attack.card.capabilities.labels
        assert label_policy is not None
        poison.validate(
            budget, len(data.split.train), data.labels, label_policy, data.defender_pool
        )
        poison_key = artifact_key(
            data=cell.scenario.data,
            budget=budget,
            attack=cell.attack,
            attack_version=cell.attack_version,
            hparams=hparams,
            seed=cell.seed,
        )

        def train_victim() -> SNNModel:
            if len(poison.indices) == 0:  # an empty poison set trains exactly the clean model
                return setting.clean
            loader = data.train_loader(poison, cell.scenario.train.batch_size)
            return setting.trainer.fit(_fresh_model(cell.scenario, data), loader)

        model_key = artifact_key(poison_key=poison_key, **_recipe(cell))
        victim = self._model(model_key, cell.seed, train_victim)
        return victim, poison, {"poison_set": poison_key, "model": model_key}

    def _attacker_model(
        self,
        setting: "_Setting",
        attack: ModelAttack,
        hparams: dict[str, Any],
        view: AttackerView,
    ) -> tuple[SNNModel, dict[str, str]]:
        """training_control and supply_chain: the attack returns the victim itself."""
        cell, data = setting.cell, setting.data
        expected = (cell.scenario.arch, data.input_shape, data.dataset.num_classes)

        def make_victim() -> SNNModel:
            model = attack.make_model(view)
            returned = (model.spec, model.input_shape, model.num_classes)
            if returned != expected:
                raise ThreatModelViolation(
                    f"{cell.attack} must return the scenario architecture {expected}, "
                    f"it returned {returned}"
                )
            # Only the state dict is stored, so the victim is the scenario architecture with the
            # returned weights. Refuse anything a checkpoint would lose or could not load.
            deployed = _fresh_model(cell.scenario, data)
            try:
                deployed.load_state_dict(model.state_dict())
            except RuntimeError as error:
                raise ThreatModelViolation(
                    f"the weights {cell.attack} returned do not fit the scenario architecture"
                ) from error
            if repr(deployed) != repr(model):
                raise ThreatModelViolation(
                    f"{cell.attack} changed the model outside its weights, for example a neuron "
                    "parameter, which a checkpoint cannot keep"
                )
            return model

        # The clean key covers the data, representation, architecture, recipe and seed.
        model_key = artifact_key(
            clean_key=setting.clean_key,
            budget=cell.scenario.budget,
            attack=cell.attack,
            attack_version=cell.attack_version,
            hparams=hparams,
            seed=cell.seed,
        )
        return self._model(model_key, cell.seed, make_victim), {"model": model_key}

    def _defend(
        self,
        setting: "_Setting",
        defense: Defense,
        view: DefenderView,
        victim: SNNModel,
        poison: PoisonSet | None,
        key: str,
    ) -> "_Defended":
        """Apply the defense as its stage says (docs/PLAN.md 4.7). Model outputs are cached."""
        cell, data, trainer = setting.cell, setting.data, setting.trainer
        seed, batch_size = cell.seed, cell.scenario.train.batch_size
        stage = defense.card.stage
        if stage == Stage.POST_TRAINING:
            assert isinstance(defense, ModelDefense)
            model = self._model(key, seed, lambda: defense.repair(view))
            return _Defended(model, artifacts={"defended_model": key})
        if stage == Stage.PRE_TRAINING:
            assert isinstance(defense, DataDefense)

            def retrain_on_kept() -> SNNModel:
                keep = _positions(defense.sanitize(view), len(data.split.train), cell.defense)
                loader = data.train_loader(poison, batch_size, keep=keep)
                return trainer.fit(_fresh_model(cell.scenario, data), loader)

            model = self._model(key, seed, retrain_on_kept)
            return _Defended(model, artifacts={"defended_model": key})
        if stage == Stage.IN_TRAINING:
            assert isinstance(defense, TrainingDefense)

            def retrain_with_hooks() -> SNNModel:
                loader = data.train_loader(poison, batch_size)
                hooks = [defense.hooks(view)]
                return trainer.fit(_fresh_model(cell.scenario, data), loader, hooks=hooks)

            model = self._model(key, seed, retrain_with_hooks)
            return _Defended(model, artifacts={"defended_model": key})
        if stage == Stage.INFERENCE:
            assert isinstance(defense, InputDefense)
            # ponytail: the fitted detector is not cached, fit runs in every cell. Cache it when
            # a fit becomes slow.
            seed_everything(seed)
            defense.fit(view)
            return _Defended(victim, detector=defense)
        assert isinstance(defense, DetectionDefense)
        verdict = self._verdict(key, seed, lambda: defense.verdict(view))
        return _Defended(victim, metrics={"verdict": verdict}, artifacts={"verdict": key})

    def _verdict(self, key: str, seed: int, compute: Callable[[], float]) -> float:
        """Load the verdict cached under key, or seed, compute and cache it."""

        def build(directory: Path) -> None:
            seed_everything(seed)
            verdict = float(compute())
            if not math.isfinite(verdict):  # NaN would make the result unreadable
                raise SbbError(f"the verdict must be a finite number, got {verdict}")
            (directory / "verdict.json").write_text(json.dumps({"verdict": verdict}))

        path = self.cache.get_or_build(key, build)
        return float(json.loads((path / "verdict.json").read_text())["verdict"])

    @torch.no_grad()
    def _scores(self, detector: InputDefense, loader: DataLoader[Any]) -> np.ndarray:
        """The input defense's score of every sample, in loader order."""
        parts = [detector.score(x.to(self.device)).cpu() for x, _, _ in loader]
        return torch.cat(parts).numpy() if parts else np.empty(0)

    def _prepare_data(self, scenario: Scenario, seed: int) -> DataModule:
        """Prepare the dataset, load the split, reserve the defender pool and build the loaders."""
        dataset = DATASETS.get(scenario.data.name)(self.data_root / scenario.data.name)
        dataset.prepare()
        split = load_split(scenario.data.split)
        if split.dataset != scenario.data.name:
            raise SbbError(f"split {split.name} belongs to {split.dataset}")
        fraction = scenario.budget.defender_clean_fraction
        pool = reserve_defender_pool(dataset.labels(), split.train, fraction, seed)
        return DataModule(dataset, split, FrameRepr(scenario.repr), self.cache, pool, seed)

    def _model(self, key: str, seed: int, make: Callable[[], SNNModel]) -> SNNModel:
        """Load the model cached under key, or seed, make and cache it."""

        def build(directory: Path) -> None:
            seed_everything(seed)
            save_checkpoint(make(), directory, key)

        return load_checkpoint(self.cache.get_or_build(key, build), self.device)


@dataclass
class _Setting:
    """What the attack steps of one cell share."""

    cell: Cell
    data: DataModule
    trainer: Trainer
    clean: SNNModel
    clean_key: str


@dataclass
class _Defended:
    """What a defense leaves for the metrics step."""

    model: SNNModel  # the model to evaluate
    detector: InputDefense | None = None  # an input defense that rejects inputs
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)


def _check_domain(name: str, attack: Attack) -> None:
    """Refuse a trigger that acts in another domain than the card's trigger_domain."""
    domain = attack.card.capabilities.trigger_domain
    if isinstance(attack.trigger, FrameTrigger) != (domain == "frames"):
        raise ThreatModelViolation(
            f"{name} declares trigger_domain {domain}, "
            f"but its trigger is a {type(attack.trigger).__name__}"
        )


def _positions(keep: np.ndarray, n_train: int, defense: str) -> np.ndarray:
    """Check the positions a data defense keeps: unique integers in [0, n_train)."""
    keep = np.asarray(keep)
    valid = keep.ndim == 1 and np.issubdtype(keep.dtype, np.integer)
    valid = valid and len(np.unique(keep)) == len(keep)
    valid = valid and (len(keep) == 0 or (keep.min() >= 0 and keep.max() < n_train))
    if not valid:
        raise SbbError(f"{defense} must keep unique positions of the train set, 0 to {n_train - 1}")
    return keep


def _recipe(cell: Cell) -> dict[str, Any]:
    """The parts of a cell that determine how a model trains, for artifact keys."""
    scenario = cell.scenario
    return {
        "repr": scenario.repr,
        "arch": scenario.arch,
        "train": scenario.train,
        "seed": cell.seed,
    }


def _fresh_model(scenario: Scenario, data: DataModule) -> SNNModel:
    """A newly initialized model of the scenario architecture."""
    return build_model(scenario.arch, data.input_shape, data.dataset.num_classes)


def _check_version(name: str, registered: str, planned: str) -> None:
    if registered != planned:
        raise SbbError(f"{name} is registered at version {registered}, the cell needs {planned}")
