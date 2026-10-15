"""CellResult and its Provenance."""

import os
import platform
import subprocess
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel

from sbb import __version__
from sbb.core.scenario import Cell
from sbb.utils.hashing import file_sha256

Outcome = Literal["ok", "not_applicable", "invalid_attack", "failed"]


class Provenance(BaseModel):
    """Where, how and by whom a result was produced."""

    sbb_version: str
    git_commit: str
    image_digest: str | None
    lockfile_sha256: str
    python: str
    torch: str
    spikingjelly: str
    cuda: str | None
    device: str
    site: str
    executor: str
    deterministic: bool
    submitter: str
    submitter_org: str
    started_at: datetime
    duration_s: float


class CellResult(BaseModel):
    """The result of one cell. Only the Game writes metrics."""

    key: str
    cell: Cell
    outcome: Outcome
    reason: str | None = None
    metrics: dict[str, float] = {}
    artifacts: dict[str, str] = {}
    provenance: Provenance


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=Path(__file__).parent
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _commit() -> str:
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return "unknown"
    return commit + ("+dirty" if _git("status", "--porcelain") else "")


def _lockfile_sha256() -> str:
    root = _git("rev-parse", "--show-toplevel")
    lock = Path(root) / "uv.lock" if root else None
    return file_sha256(lock) if lock and lock.exists() else "unknown"


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if platform.system() == "Darwin":
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        if out.stdout.strip():
            return out.stdout.strip()
    return platform.processor() or platform.machine()


def collect_provenance(
    device: torch.device, site: str, executor: str, started_at: datetime, duration_s: float
) -> Provenance:
    """Describe the current environment. Submitter fields come from SBB_SUBMITTER(_ORG)."""
    cuda = device.type == "cuda"
    return Provenance(
        sbb_version=__version__,
        git_commit=_commit(),
        image_digest=os.environ.get("SBB_IMAGE_DIGEST"),
        lockfile_sha256=_lockfile_sha256(),
        python=platform.python_version(),
        torch=torch.__version__,
        spikingjelly=version("spikingjelly"),
        cuda=torch.version.cuda if cuda else None,
        device=torch.cuda.get_device_name(device) if cuda else _cpu_name(),
        site=site,
        executor=executor,
        deterministic=torch.are_deterministic_algorithms_enabled(),
        submitter=os.environ.get("SBB_SUBMITTER", "unknown"),
        submitter_org=os.environ.get("SBB_SUBMITTER_ORG", "unknown"),
        started_at=started_at,
        duration_s=duration_s,
    )
