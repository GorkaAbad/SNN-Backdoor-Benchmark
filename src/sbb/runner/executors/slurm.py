"""Submits cells as one Slurm job array, optionally inside an Apptainer image."""

import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

from sbb.core.scenario import Cell
from sbb.runner.executors.base import Executor
from sbb.utils.errors import SbbError


class SlurmExecutor(Executor):
    """One array job, at most site.max_parallel tasks at a time, one cell per task.

    site.slurm entries become `#SBATCH --<key>=<value>` lines, and its "extra" entry is copied as
    is, one line per line. With runtime apptainer, each task runs the site's image with
    `apptainer exec --nv` (on CUDA) and the cache bound. With runtime none, the tasks use the
    environment sbatch exports, so `sbb` must be on the PATH of the submitting shell.
    """

    def submit(self, cells: Sequence[Cell], out_dir: Path) -> list[str]:
        """Write the job, submit it with sbatch, and return the array's job id."""
        if not cells:
            return []
        job = self.write_job(cells, out_dir)
        script = job / "array.sbatch"
        script.write_text(self.script(job, len(cells), out_dir.resolve()))
        submitted = subprocess.run(
            ["sbatch", "--parsable", str(script)], capture_output=True, text=True, check=True
        )
        return [submitted.stdout.strip().split(";")[0]]

    def script(self, job: Path, n: int, out_dir: Path) -> str:
        """The sbatch script of a job of n cells."""
        options = {k: v for k, v in self.site.slurm.items() if k != "extra"}
        lines = [
            "#!/bin/bash",
            "#SBATCH --job-name=sbb",
            f"#SBATCH --array=0-{n - 1}%{self.site.max_parallel}",
            f"#SBATCH --output={job}/logs/%A_%a.out",
            *(f"#SBATCH --{key}={value}" for key, value in options.items()),
            *self.site.slurm.get("extra", "").splitlines(),
            "set -euo pipefail",
        ]
        command = shlex.join(self.command(job, "INDEX", out_dir))
        command = command.replace("INDEX", "${SLURM_ARRAY_TASK_ID}")
        return "\n".join([*lines, f"{self.prefix()}{command}", ""])

    def prefix(self) -> str:
        """The container part of the task command."""
        runtime = self.site.runtime
        if runtime == "none":
            return ""
        if runtime == "apptainer":
            if not self.site.image:
                raise SbbError(f"site {self.site.name} runs apptainer but sets no image")
            gpu = "--nv " if self.site.device == "cuda" else ""
            cache = shlex.quote(str(self.site.cache_dir))
            return f"apptainer exec {gpu}--bind {cache} {shlex.quote(self.site.image)} "
        raise SbbError(f"Slurm sites run apptainer or no container, not {runtime}")
