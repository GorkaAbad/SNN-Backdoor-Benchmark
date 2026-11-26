"""Benchmark spec (docs/PLAN.md section 7.3)."""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, model_validator

from sbb.core.scenario import Scenario

Track = Literal["lite", "full"]


class Battery(BaseModel, extra="forbid"):
    """Attacks and defenses evaluated against each other."""

    attacks: list[str]
    defenses: list[str]
    dev: list[str] = []


class Tolerance(BaseModel, extra="forbid"):
    """Reproduction tolerances in percentage points."""

    lite_pp: float = 3.0
    full_pp: float = 2.0
    full_sigma: float = 2.0


class TrackSpec(BaseModel, extra="forbid"):
    """Scenarios and seeds of one track.

    Attributes:
        exclude: Methods of the battery this track leaves out, for example an attack too slow to
            verify on a CPU runner in the lite track.
    """

    scenarios: list[str]
    seeds: list[int]
    exclude: list[str] = []


class BenchmarkSpec(BaseModel, extra="forbid"):
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
            battery = {*self.battery.attacks, *self.battery.defenses}
            if unknown := set(spec.exclude) - battery:
                raise ValueError(f"track {track} excludes {sorted(unknown)}, not in the battery")
        # The site picks cells by version and scenario, without the track.
        listed = [s for spec in self.tracks.values() for s in spec.scenarios]
        if twice := sorted({s for s in listed if listed.count(s) > 1}):
            raise ValueError(f"scenarios {twice} are in more than one track")
        return self

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load and validate a YAML spec."""
        return cls.model_validate(yaml.safe_load(path.read_text()))
