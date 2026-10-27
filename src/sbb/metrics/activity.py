"""Activity of the evaluated model: spike rates and synaptic operations (docs/PLAN.md section 8)."""

from sbb.core.plugins import EvalContext, Metric
from sbb.core.registry import METRICS
from sbb.models.snn import ActivityCounts
from sbb.utils.errors import SbbError


def _counts(counts: ActivityCounts | None) -> ActivityCounts:
    if counts is None:
        raise SbbError("no activity was recorded for this evaluation")
    return counts


@METRICS.register("spike_rate")
class SpikeRate(Metric):
    """Spikes per neuron and time step on the clean test set."""

    name = "spike_rate"

    def compute(self, ctx: EvalContext) -> float:
        """1 minus NeuroBench's activation sparsity, over every neuron layer."""
        return _counts(ctx.clean_activity).spike_rate


@METRICS.register("spike_rate_triggered")
class TriggeredSpikeRate(Metric):
    """Spikes per neuron and time step on the triggered test set."""

    name = "spike_rate_triggered"

    def compute(self, ctx: EvalContext) -> float:
        """The spike rate the trigger causes, to compare with spike_rate."""
        return _counts(ctx.triggered_activity).spike_rate


@METRICS.register("synops")
class SynapticOperations(Metric):
    """Synaptic operations per clean test sample, a raw count."""

    name = "synops"
    unit = "count"

    def compute(self, ctx: EvalContext) -> float:
        """Connections that carry a nonzero input, per sample over all time steps."""
        return _counts(ctx.clean_activity).synops_per_sample
