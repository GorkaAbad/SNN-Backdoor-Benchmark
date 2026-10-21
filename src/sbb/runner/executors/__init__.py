"""Executors decide where `sbb run-cell` runs (docs/PLAN.md section 6.9)."""

from sbb.runner.executors.base import Executor
from sbb.runner.executors.local import LocalExecutor
from sbb.runner.executors.slurm import SlurmExecutor
from sbb.runner.site import SiteProfile


def executor_for(site: SiteProfile) -> Executor:
    """The executor of the site's scheduler."""
    if site.scheduler == "slurm":
        return SlurmExecutor(site)
    if site.scheduler == "hfjobs":
        raise NotImplementedError("the hfjobs executor lands in v2")
    return LocalExecutor(site)


__all__ = ["Executor", "LocalExecutor", "SlurmExecutor", "executor_for"]
