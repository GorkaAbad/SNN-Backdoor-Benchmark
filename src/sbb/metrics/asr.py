"""Attack success rate and robust accuracy on the triggered test set."""

import numpy as np

from sbb.core.plugins import EvalContext, Metric
from sbb.core.registry import METRICS


@METRICS.register("asr")
class AttackSuccessRate(Metric):
    """Triggered samples accepted and classified as the target."""

    name = "asr"

    def compute(self, ctx: EvalContext) -> float:
        """Fraction of triggered samples accepted and predicted as the target."""
        return float(np.mean(ctx.triggered_ok & (ctx.triggered_pred == ctx.target)))


@METRICS.register("ra")
class RobustAccuracy(Metric):
    """Triggered samples accepted and classified as their true label."""

    name = "ra"

    def compute(self, ctx: EvalContext) -> float:
        """Fraction of triggered samples accepted and predicted as their true label."""
        return float(np.mean(ctx.triggered_ok & (ctx.triggered_pred == ctx.triggered_true)))
