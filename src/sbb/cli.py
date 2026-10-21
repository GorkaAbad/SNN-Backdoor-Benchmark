"""Command line interface. The only module that prints."""

import logging
import signal
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
from sbb.results.replication import ReplicationRecord
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
    """Configure logging, and turn SIGTERM into a normal exit."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
        force=True,
    )
    # Slurm sends SIGTERM on scancel and at the time limit. Exiting through SystemExit lets the
    # with blocks remove their staging directories, which can hold gigabytes.
    signal.signal(signal.SIGTERM, _exit_on_signal)


def _exit_on_signal(signum: int, frame: object) -> None:
    raise SystemExit(128 + signum)


def _site(path: Path | None) -> SiteProfile:
    return SiteProfile.load(path) if path else SiteProfile.default()


def _done(store: ResultStore, cell: Cell) -> bool:
    result = store.get(cell.key())
    return result is not None and result.outcome != "failed"


def _run_pending(cells: list[Cell], store: ResultStore, site: SiteProfile) -> int:
    """Run the cells without a result in-process, and return the number that failed."""
    pending = [c for c in cells if not _done(store, c)]
    console.print(f"{len(cells)} cells, {len(pending)} pending")
    game = site.game("in-process")
    return sum(not _run_cell(game, cell, store) for cell in pending)


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
    failures = _run_pending(Planner(bench).cells(track, attack, defense), store, _site(site))
    table = out / bench.version / "table.md"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(markdown_table(store.query(benchmark_version=bench.version)))
    console.print(f"table written to {table}")
    if failures:
        console.print(f"[red]{failures} cells failed[/]")
        raise typer.Exit(1)


@app.command()
def replicate(
    method: str,
    records: Annotated[Path, typer.Option(help="Directory of replication records.")] = Path(
        "replications"
    ),
    site: Site = None,
    out: Out = Path("results"),
    update: Annotated[
        bool, typer.Option(help="Also write the filled record into RECORDS.")
    ] = False,
) -> None:
    """Run the replication record of METHOD and compare our numbers with the paper's.

    The filled record goes to OUT/replications. Use --update only in a local checkout, never in
    a clone that must stay unmodified, such as one on a cluster.
    """
    path = records / f"{method}.yaml"
    record = ReplicationRecord.load(path)
    store = LocalResultStore(out)
    cells = record.cells()
    if _run_pending(cells, store, _site(site)):
        console.print("[red]some cells failed, the record is not updated[/]")
        raise typer.Exit(1)
    filled = record.with_results([store.get(cell.key()) for cell in cells])
    table = Table("setting", "metric", "paper %", "ours %")
    for target in filled.targets:
        assert target.ours is not None
        for metric, reported in target.reported.items():
            ours = target.ours[metric]
            row = (f"{reported:.2f}", f"{ours.mean:.2f} ± {ours.std:.2f}")
            table.add_row(target.setting.name, metric, *row)
    console.print(table)
    filled.save(out / "replications" / f"{method}.yaml")
    if update:
        filled.save(path)
    console.print(f"{method}: {filled.status} (tolerance {filled.tolerance_pp} pp)")


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
