"""Statistics of a metric over the seeds of a cell (docs/PLAN.md 13.4, task 5)."""

import math
from collections.abc import Sequence
from typing import TypedDict

import numpy as np
from scipy import stats


class SeedStats(TypedDict):
    """A metric over seeds: mean, population std, number of seeds, 95% half-width or None."""

    mean: float
    std: float
    n: int
    ci95: float | None


def seed_stats(values: Sequence[float]) -> SeedStats:
    """Mean, population std, number of seeds and the half-width of the 95% t-interval.

    The interval is Student's t with n - 1 degrees of freedom on the sample std (n - 1). With 3
    seeds it is wide, which is the honest picture. One seed has no interval (None).
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    ci95 = None if n < 2 else float(stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / math.sqrt(n))
    return {"mean": float(v.mean()), "std": float(v.std()), "n": n, "ci95": ci95}
