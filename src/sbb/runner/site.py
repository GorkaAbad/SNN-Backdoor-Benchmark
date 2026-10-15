"""Site profiles describe a compute environment. Real ones live outside the repository."""

from pathlib import Path
from typing import Literal, Self

import torch
import yaml
from pydantic import BaseModel, field_validator

from sbb.core.game import Game
from sbb.utils.storage import ArtifactCache, LocalStore


class SiteProfile(BaseModel):
    """A compute environment (docs/PLAN.md section 7.4)."""

    name: str
    runtime: Literal["none", "docker", "apptainer"] = "none"
    scheduler: Literal["none", "slurm", "hfjobs"] = "none"
    image: str | None = None
    device: Literal["cpu", "cuda"] = "cuda"
    cache_dir: Path
    max_parallel: int = 1
    slurm: dict[str, str] = {}
    hfjobs: dict[str, str] = {}

    @field_validator("cache_dir")
    @classmethod
    def _expand(cls, path: Path) -> Path:
        return path.expanduser()

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load and validate a YAML profile."""
        return cls.model_validate(yaml.safe_load(path.read_text()))

    @classmethod
    def default(cls) -> Self:
        """Local profile with ~/.cache/sbb, on CUDA if available."""
        device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(name="default", device=device, cache_dir=Path("~/.cache/sbb").expanduser())

    def game(self, executor: str) -> Game:
        """Build the Game for this site. Datasets live in <cache_dir>/datasets."""
        cache = ArtifactCache(LocalStore(self.cache_dir / "artifacts"))
        return Game(
            cache, torch.device(self.device), self.cache_dir / "datasets", self.name, executor
        )
