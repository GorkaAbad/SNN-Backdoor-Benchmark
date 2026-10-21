"""Runs cells on this machine, in a pool of `sbb run-cell` processes."""

import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sbb.core.scenario import Cell
from sbb.runner.executors.base import Executor


class LocalExecutor(Executor):
    """Runs up to site.max_parallel cells at once, each in its own process, and waits for them.

    Each cell's output goes to <job>/logs/<i>.log. The job ids are the cell keys.
    """

    def submit(self, cells: Sequence[Cell], out_dir: Path) -> list[str]:
        """Run the cells and return their keys once every process has finished."""
        job = self.write_job(cells, out_dir)
        out = out_dir.resolve()

        def run(i: int) -> None:
            # python -m sbb, so the processes use this interpreter's environment
            command = [sys.executable, "-m", *self.command(job, str(i), out)]
            with (job / "logs" / f"{i}.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)

        with ThreadPoolExecutor(self.site.max_parallel) as pool:
            list(pool.map(run, range(len(cells))))
        return [cell.key() for cell in cells]
