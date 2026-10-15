"""Command line interface. The only module that prints."""

import logging
import warnings
from pathlib import Path
from typing import Annotated

# SpikingJelly triggers this on import. It must be filtered before the plugin imports below.
warnings.filterwarnings("ignore", message="`torch.jit.script` is deprecated")

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

import sbb.attacks  # noqa: F401  registers the plugins
import sbb.defenses  # noqa: F401
import sbb.metrics  # noqa: F401
from sbb.core.cards import compatible
from sbb.core.game import Game
from sbb.core.registry import ATTACKS, DEFENSES
from sbb.core.scenario import Cell
from sbb.results.store import LocalResultStore, ResultStore
from sbb.results.table import markdown_table
from sbb.runner.planner import Planner
from sbb.runner.site import SiteProfile
from sbb.runner.spec import BenchmarkSpec

app = typer.Typer(no_args_is_help=True, help="SNN backdoor benchmark.")
console = Console()

Site = Annotated[Path | None, typer.Option(help="SiteProfile YAML. Default: local, ~/.cache/sbb.")]
Out = Annotated[Path, typer.Option(help="Results directory.")]
TrackFilter = Annotated[str | None, typer.Option("--track", help="Only this track.")]
AttackFilter = Annotated[str | None, typer.Option("--attack", help="Only this attack.")]
DefenseFilter = Annotated[str | None, typer.Option("--defense", help="Only this defense.")]


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
        force=True,
    )


def _site(path: Path | None) -> SiteProfile:
    return SiteProfile.load(path) if path else SiteProfile.default()


def _done(store: ResultStore, cell: Cell) -> bool:
    result = store.get(cell.key())
    return result is not None and result.outcome != "failed"


def _run_cell(game: Game, cell: Cell, store: ResultStore) -> bool:
    """Run one cell through the Game, save the result, and return False if it failed."""
    console.print(f"[bold]{cell.scenario.name}[/] {cell.attack} × {cell.defense} seed {cell.seed}")
    result = game.run(cell)
    store.save(result)
    metrics = " ".join(f"{k}={v:.4f}" for k, v in result.metrics.items())
    console.print(f"  {result.outcome} {metrics} {result.reason or ''}")
    return result.outcome != "failed"


@app.command("run-cell")
def run_cell(cell_json: Path, out: Out = Path("results"), site: Site = None) -> None:
    """Run exactly one cell from its JSON. The entry point every executor calls."""
    cell = Cell.model_validate_json(cell_json.read_text())
    if not _run_cell(_site(site).game("run-cell"), cell, LocalResultStore(out)):
        raise typer.Exit(1)


@app.command()
def run(
    spec: Path,
    track: TrackFilter = None,
    attack: AttackFilter = None,
    defense: DefenseFilter = None,
    site: Site = None,
    out: Out = Path("results"),
) -> None:
    """Plan the cells of SPEC, run the pending ones in-process, and write table.md."""
    bench = BenchmarkSpec.load(spec)
    store = LocalResultStore(out)
    cells = Planner(bench).cells(track, attack, defense)
    pending = [c for c in cells if not _done(store, c)]
    console.print(f"{len(cells)} cells, {len(pending)} pending")
    game = _site(site).game("in-process")
    failures = sum(not _run_cell(game, cell, store) for cell in pending)
    table = out / bench.version / "table.md"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(markdown_table(store.query(benchmark_version=bench.version)))
    console.print(f"table written to {table}")
    if failures:
        console.print(f"[red]{failures} cells failed[/]")
        raise typer.Exit(1)


@app.command()
def plan(
    spec: Path,
    track: TrackFilter = None,
    attack: AttackFilter = None,
    defense: DefenseFilter = None,
    out: Out = Path("results"),
) -> None:
    """List the cells of SPEC with their status and N/A reasons."""
    store = LocalResultStore(out)
    table = Table("cell", "track", "scenario", "attack", "defense", "seed", "status")
    for cell in Planner(BenchmarkSpec.load(spec)).cells(track, attack, defense):
        attack_card, defense_card = ATTACKS.get(cell.attack).card, DEFENSES.get(cell.defense).card
        ok, reasons = compatible(attack_card, defense_card, cell.scenario.budget)
        result = store.get(cell.key())
        status = result.outcome if result else "pending" if ok else "N/A: " + "; ".join(reasons)
        table.add_row(
            cell.key()[:12],
            cell.track,
            cell.scenario.name,
            cell.attack,
            cell.defense,
            str(cell.seed),
            status,
        )
    console.print(table)


@app.command()
def test(plugin: Annotated[str | None, typer.Option(help="Only this plugin.")] = None) -> None:
    """Run the contract tests, optionally for one plugin."""
    import pytest

    args = ["-q", "tests/contracts"] + (["-k", plugin] if plugin else [])
    raise typer.Exit(pytest.main(args))
