"""Expands a benchmark spec into cells."""

from itertools import product

from sbb.core.registry import ATTACKS, DEFENSES
from sbb.core.scenario import Cell
from sbb.runner.spec import BenchmarkSpec


class Planner:
    """Lists the cells of a benchmark spec."""

    def __init__(self, spec: BenchmarkSpec) -> None:
        self.spec = spec

    def cells(
        self, track: str | None = None, attack: str | None = None, defense: str | None = None
    ) -> list[Cell]:
        """Return every (track, scenario, attack, defense, seed) cell, N/A cells included."""
        battery = self.spec.battery
        cells = []
        for track_name, track_spec in self.spec.tracks.items():
            if track not in (None, track_name):
                continue
            for scenario, a, d, seed in product(
                track_spec.scenarios, battery.attacks, battery.defenses, track_spec.seeds
            ):
                if attack not in (None, a) or defense not in (None, d):
                    continue
                cells.append(
                    Cell(
                        benchmark_version=self.spec.version,
                        track=track_name,
                        scenario=self.spec.scenarios[scenario],
                        attack=a,
                        attack_version=ATTACKS.get(a).card.version,
                        defense=d,
                        defense_version=DEFENSES.get(d).card.version,
                        seed=seed,
                    )
                )
        return cells
