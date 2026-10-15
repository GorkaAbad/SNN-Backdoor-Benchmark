"""The security game that evaluates one cell (docs/PLAN.md sections 3.1 and 4.7)."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import torch

from sbb.core.cards import ScenarioKind, Stage, compatible
from sbb.core.plugins import EvalContext, ModelDefense, PoisoningAttack
from sbb.core.proxies import AttackerView, DefenderView, ScenarioInfo, TrainSetView
from sbb.core.registry import ATTACKS, DEFENSES, METRICS
from sbb.core.result import CellResult, Outcome, collect_provenance
from sbb.core.scenario import Cell, Scenario, artifact_key
from sbb.core.trainer import Trainer
from sbb.data.module import DataModule
from sbb.data.representations import FrameRepr
from sbb.data.sample import DATASETS
from sbb.data.splits import load_split, reserve_defender_pool
from sbb.models.checkpoint import load_checkpoint, save_checkpoint
from sbb.models.snn import SNNModel, build_model
from sbb.utils.determinism import seed_everything
from sbb.utils.errors import SbbError
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
        if attack_cls.card.kind != ScenarioKind.DATA_POISONING:
            raise NotImplementedError(f"scenario kind {attack_cls.card.kind} lands in v1")
        if defense_cls.card.stage != Stage.POST_TRAINING:
            raise NotImplementedError(f"defense stage {defense_cls.card.stage} lands in v1")
        assert issubclass(attack_cls, PoisoningAttack) and issubclass(defense_cls, ModelDefense)

        # 2. Data and defender pool
        seed_everything(seed)
        data = self._prepare_data(scenario, seed)
        dataset, pool = data.dataset, data.defender_pool
        trainer = Trainer(scenario.train, self.device)
        info = ScenarioInfo(
            dataset=dataset.name,
            sensor_size=dataset.sensor_size,
            num_classes=dataset.num_classes,
            T=scenario.repr.T,
        )

        def fresh_model() -> SNNModel:
            return build_model(scenario.arch, data.input_shape, dataset.num_classes)

        # 3. Clean reference
        recipe = {
            "repr": scenario.repr,
            "arch": scenario.arch,
            "train": scenario.train,
            "seed": seed,
        }
        clean_key = artifact_key(data=scenario.data, **recipe)
        clean = self._model(
            clean_key, seed, lambda: trainer.fit(fresh_model(), data.train_loader(None, bs))
        )

        # 4. Attack
        attack_hparams = attack_cls.card.hparams_for(dataset.name, scenario.arch.name)
        attack = attack_cls(attack_hparams, budget.target)
        attacker_view = AttackerView(
            attack_cls.card, info, seed, TrainSetView(data, None), scenario.arch
        )
        poison = attack.poison(attacker_view, budget, pool)
        label_policy = attack_cls.card.capabilities.labels
        assert label_policy is not None
        poison.validate(budget, len(data.split.train), data.labels, label_policy, pool)
        poison_key = artifact_key(
            data=scenario.data,
            budget=budget,
            attack=cell.attack,
            attack_version=cell.attack_version,
            hparams=attack_hparams,
            seed=seed,
        )

        # 5. Victim. An empty poison set trains exactly the clean model, so its weights are reused.
        def train_victim() -> SNNModel:
            if len(poison.indices) == 0:
                return clean
            return trainer.fit(fresh_model(), data.train_loader(poison, bs))

        model_key = artifact_key(poison_key=poison_key, **recipe)
        victim = self._model(model_key, seed, train_victim)
        artifacts = {"clean_model": clean_key, "poison_set": poison_key, "model": model_key}

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
            model_key=model_key,
            defense=cell.defense,
            defense_version=cell.defense_version,
            hparams=defense_hparams,
            seed=seed,
        )
        defended = self._model(defense_key, seed, lambda: defense.repair(defender_view))
        artifacts["defended_model"] = defense_key

        # 8. Metrics
        triggered = data.triggered_loader(attack.trigger_test, budget.target, bs)
        ctx = EvalContext(
            *trainer.predict(defended, test), *trainer.predict(defended, triggered), budget.target
        )
        metrics = {name: METRICS.get(name)().compute(ctx) for name in METRICS.names()}
        return _Report("ok", None, metrics, artifacts)

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


def _check_version(name: str, registered: str, planned: str) -> None:
    if registered != planned:
        raise SbbError(f"{name} is registered at version {registered}, the cell needs {planned}")
