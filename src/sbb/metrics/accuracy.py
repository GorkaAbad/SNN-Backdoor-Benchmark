"""Clean accuracy."""

import numpy as np

from sbb.core.plugins import EvalContext, Metric
from sbb.core.registry import METRICS


@METRICS.register("ca")
class CleanAccuracy(Metric):
    """Clean test samples accepted and classified correctly."""

    name = "ca"

    def compute(self, ctx: EvalContext) -> float:
        """Fraction of clean samples accepted and predicted correctly."""
        return float(np.mean(ctx.clean_ok & (ctx.clean_pred == ctx.clean_true)))
