"""Metrics of input defenses: rejection rates and how well scores separate triggered inputs."""

import numpy as np

from sbb.core.cards import Stage
from sbb.core.plugins import EvalContext, Metric
from sbb.core.registry import METRICS

_INFERENCE = frozenset({Stage.INFERENCE})


def auroc(negative: np.ndarray, positive: np.ndarray) -> float:
    """Probability that a positive scores higher than a negative, ties counting half."""
    sorted_negative = np.sort(negative)
    below = np.searchsorted(sorted_negative, positive, side="left")
    not_above = np.searchsorted(sorted_negative, positive, side="right")
    pairs = (below + (not_above - below) / 2).sum()
    return float(pairs / (len(negative) * len(positive)))


@METRICS.register("frr")
class FalseRejectionRate(Metric):
    """Clean test samples the input defense rejects."""

    name = "frr"
    stages = _INFERENCE

    def compute(self, ctx: EvalContext) -> float:
        """Fraction of clean samples rejected."""
        return float(np.mean(~ctx.clean_ok))


@METRICS.register("tpr")
class TruePositiveRate(Metric):
    """Triggered test samples the input defense rejects."""

    name = "tpr"
    stages = _INFERENCE

    def compute(self, ctx: EvalContext) -> float:
        """Fraction of triggered samples rejected."""
        return float(np.mean(~ctx.triggered_ok))


@METRICS.register("auroc")
class InputAUROC(Metric):
    """Area under the ROC curve of the input defense's scores, triggered samples positive."""

    name = "auroc"
    stages = _INFERENCE

    def compute(self, ctx: EvalContext) -> float:
        """AUROC of clean versus triggered scores."""
        assert ctx.clean_scores is not None and ctx.triggered_scores is not None
        return auroc(ctx.clean_scores, ctx.triggered_scores)
