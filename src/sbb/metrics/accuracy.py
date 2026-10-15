"""Clean accuracy."""

import numpy as np

from sbb.core.plugins import EvalContext, Metric
from sbb.core.registry import METRICS


@METRICS.register("ca")
class CleanAccuracy(Metric):
    """Clean test samples classified correctly."""

    name = "ca"

    def compute(self, ctx: EvalContext) -> float:
        """Fraction of correct clean predictions."""
        return float(np.mean(ctx.clean_pred == ctx.clean_true))
