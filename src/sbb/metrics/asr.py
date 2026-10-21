"""Attack success rate and robust accuracy on the triggered test set."""

import numpy as np

from sbb.core.plugins import EvalContext, Metric
from sbb.core.registry import METRICS


def _hits(ctx: EvalContext) -> np.ndarray:
    """Triggered samples accepted and predicted as the target."""
    hits: np.ndarray = ctx.triggered_ok & (ctx.triggered_pred == ctx.target)
    return hits


@METRICS.register("asr")
class AttackSuccessRate(Metric):
    """Triggered samples of other classes than the target, accepted and classified as the target."""

    name = "asr"

    def compute(self, ctx: EvalContext) -> float:
        """Fraction of triggered non-target samples accepted and predicted as the target."""
        return float(np.mean(_hits(ctx)[ctx.nontarget]))


@METRICS.register("asr_all")
class AttackSuccessRateAll(Metric):
    """ASR as the ported papers report it: every triggered sample, the target class included."""

    name = "asr_all"

    def compute(self, ctx: EvalContext) -> float:
        """Fraction of triggered samples accepted and predicted as the target."""
        return float(np.mean(_hits(ctx)))


@METRICS.register("ra")
class RobustAccuracy(Metric):
    """Triggered non-target samples accepted and classified as their true label."""

    name = "ra"

    def compute(self, ctx: EvalContext) -> float:
        """Fraction of triggered non-target samples accepted and predicted as their true label."""
        correct = ctx.triggered_ok & (ctx.triggered_pred == ctx.triggered_true)
        return float(np.mean(correct[ctx.nontarget]))
