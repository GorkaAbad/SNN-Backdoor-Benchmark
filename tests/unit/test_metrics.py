import numpy as np
import pytest

from sbb.core.cards import Stage
from sbb.core.plugins import EvalContext
from sbb.core.registry import METRICS
from sbb.metrics.detection import auroc


def _context(**overrides: object) -> EvalContext:
    fields: dict[str, object] = {
        "clean_pred": np.array([0, 1, 2, 3]),
        "clean_true": np.array([0, 1, 2, 0]),
        "triggered_pred": np.array([0, 0, 2, 0]),
        "triggered_true": np.array([1, 2, 2, 3]),
        "target": 0,
    }
    return EvalContext(**{**fields, **overrides})  # type: ignore[arg-type]


def _metric(name: str, ctx: EvalContext) -> float:
    return METRICS.get(name)().compute(ctx)


def test_without_an_input_defense_every_sample_is_accepted() -> None:
    ctx = _context()
    assert _metric("ca", ctx) == 0.75
    assert _metric("asr", ctx) == 0.75
    assert _metric("ra", ctx) == 0.25


def test_rejected_samples_count_as_neither_correct_nor_successful() -> None:
    ctx = _context(
        clean_accepted=np.array([True, True, False, True]),
        triggered_accepted=np.array([False, True, True, True]),
        clean_scores=np.array([0.1, 0.2, 0.9, 0.3]),
        triggered_scores=np.array([0.9, 0.2, 0.1, 0.4]),
    )
    assert _metric("ca", ctx) == 0.5  # sample 2 is correct but rejected
    assert _metric("asr", ctx) == 0.5  # sample 0 reaches the target but is rejected
    assert _metric("ra", ctx) == 0.25
    assert _metric("frr", ctx) == 0.25
    assert _metric("tpr", ctx) == 0.25
    # Higher scores mean more suspicious. Of the 16 (clean, triggered) pairs, the triggered sample
    # scores higher in 3 + 1 + 0 + 3 = 7, and the three ties (0.9, 0.2, 0.1) count half each.
    assert _metric("auroc", ctx) == pytest.approx(8.5 / 16)


def test_target_class_samples_count_only_in_asr_all() -> None:
    # The last triggered sample belongs to the target class. The papers count it in their ASR,
    # the benchmark's metrics leave it out.
    fields = {
        "triggered_pred": np.array([0, 0, 2, 0, 0]),
        "triggered_true": np.array([1, 2, 2, 3, 0]),
    }
    ctx = _context(**fields)
    assert _metric("asr", ctx) == 0.75 and _metric("ra", ctx) == 0.25
    assert _metric("asr_all", ctx) == 0.8
    rejecting = _context(
        **fields,
        clean_accepted=np.ones(4, dtype=bool),
        triggered_accepted=np.array([True, True, True, True, False]),
        clean_scores=np.array([0.1, 0.2, 0.3, 0.4]),
        triggered_scores=np.array([0.5, 0.6, 0.7, 0.8, 0.0]),
    )
    assert _metric("tpr", rejecting) == 0.0  # the only rejected sample is of the target class
    assert _metric("auroc", rejecting) == 1.0  # and its low score does not count
    assert _metric("asr_all", rejecting) == 0.6


def test_auroc_counts_ties_as_half() -> None:
    assert auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 1.0
    assert auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 0.0
    assert auroc(np.array([1.0, 1.0]), np.array([1.0])) == 0.5


def test_input_defense_metrics_apply_only_to_input_defenses() -> None:
    for name in ("frr", "tpr", "auroc"):
        assert METRICS.get(name).stages == frozenset({Stage.INFERENCE})
    for name in ("ca", "asr", "ra", "asr_all"):
        assert METRICS.get(name).stages == frozenset(Stage)
