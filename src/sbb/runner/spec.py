"""Benchmark spec (docs/PLAN.md section 7.3)."""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, model_validator

from sbb.core.scenario import Scenario

Track = Literal["lite", "full"]


class Battery(BaseModel):
    """Attacks and defenses evaluated against each other."""

    attacks: list[str]
    defenses: list[str]
    dev: list[str] = []


class Tolerance(BaseModel):
    """Reproduction tolerances in percentage points."""

    lite_pp: float = 1.0
    full_pp: float = 2.0
    full_sigma: float = 2.0


class TrackSpec(BaseModel):
    """Scenarios and seeds of one track."""

    scenarios: list[str]
    seeds: list[int]


class BenchmarkSpec(BaseModel):
    """A versioned benchmark."""

    benchmark: str = "sbb"
    version: str
    scenarios: dict[str, Scenario]
    tracks: dict[Track, TrackSpec]
    battery: Battery
    tolerance: Tolerance = Tolerance()

    @model_validator(mode="after")
    def _scenarios_resolve(self) -> Self:
        for key, scenario in self.scenarios.items():
            if key != scenario.name:
                raise ValueError(f"scenario key {key!r} differs from its name {scenario.name!r}")
        for track, spec in self.tracks.items():
            if unknown := set(spec.scenarios) - set(self.scenarios):
                raise ValueError(f"track {track} uses unknown scenarios {sorted(unknown)}")
        return self

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load and validate a YAML spec."""
        return cls.model_validate(yaml.safe_load(path.read_text()))
