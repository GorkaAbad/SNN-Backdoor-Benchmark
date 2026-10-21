"""The executor interface and the job directory every executor writes."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import yaml

from sbb.core.scenario import Cell
from sbb.runner.site import SiteProfile
from sbb.utils.hashing import sha256_hex


class Executor(ABC):
    """Runs cells somewhere. Every executor ends up running `sbb run-cell <cell.json> --out <dir>`.

    Args:
        site: The compute environment.
    """

    def __init__(self, site: SiteProfile) -> None:
        self.site = site

    @abstractmethod
    def submit(self, cells: Sequence[Cell], out_dir: Path) -> list[str]:
        """Start the cells, with their results going to out_dir, and return the job ids."""

    def write_job(self, cells: Sequence[Cell], out_dir: Path) -> Path:
        """Write the cells and the site under <out_dir>/jobs/<hash of the cell keys>/.

        Returns the job directory, with cells/<i>.json, site.yaml and logs/.
        """
        job = out_dir.resolve() / "jobs" / sha256_hex(sorted(c.key() for c in cells))[:12]
        (job / "cells").mkdir(parents=True, exist_ok=True)
        (job / "logs").mkdir(exist_ok=True)
        for i, cell in enumerate(cells):
            (job / "cells" / f"{i}.json").write_text(cell.model_dump_json())
        site = self.site.model_dump(mode="json")
        (job / "site.yaml").write_text(yaml.safe_dump(site, sort_keys=False))
        return job

    def command(self, job: Path, index: str, out_dir: Path) -> list[str]:
        """The run-cell command of one cell of a job, index being a number or a shell variable."""
        cell = f"{job}/cells/{index}.json"
        return ["sbb", "run-cell", cell, "--site", str(job / "site.yaml"), "--out", str(out_dir)]
