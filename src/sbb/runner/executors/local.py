"""Runs cells on this machine, in a pool of `sbb run-cell` processes."""

import os
import queue
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from sbb.core.scenario import Cell
from sbb.runner.executors.base import Executor


class LocalExecutor(Executor):
    """Runs up to site.max_parallel cells at once, each in its own process, and waits for them.

    On CUDA, each worker slot owns one visible GPU, round-robin, through CUDA_VISIBLE_DEVICES.
    A GPU in exclusive-process mode, common on clusters, takes one process only, so there
    max_parallel must not exceed the number of GPUs. Each cell's output goes to
    <job>/logs/<i>.log. The job ids are the cell keys.
    """

    def submit(self, cells: Sequence[Cell], out_dir: Path) -> list[str]:
        """Run the cells and return their keys once every process has finished."""
        job = self.write_job(cells, out_dir)
        out = out_dir.resolve()
        gpus = _visible_gpus() if self.site.device == "cuda" else []
        slots: queue.Queue[int] = queue.Queue()
        for slot in range(self.site.max_parallel):
            slots.put(slot)

        def run(i: int) -> None:
            # python -m sbb, so the processes use this interpreter's environment
            command = [sys.executable, "-m", *self.command(job, str(i), out)]
            slot = slots.get()
            env = dict(os.environ)
            if gpus:
                env["CUDA_VISIBLE_DEVICES"] = gpus[slot % len(gpus)]
            try:
                with (job / "logs" / f"{i}.log").open("w") as log:
                    subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, check=False, env=env
                    )
            finally:
                slots.put(slot)

        with ThreadPoolExecutor(self.site.max_parallel) as pool:
            list(pool.map(run, range(len(cells))))
        return [cell.key() for cell in cells]


def _visible_gpus() -> list[str]:
    """The GPUs this process may use: CUDA_VISIBLE_DEVICES if set, else every device."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [gpu for gpu in visible.split(",") if gpu]
    return [str(i) for i in range(torch.cuda.device_count())]
