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
from sbb.core.result import CellResult
from sbb.core.scenario import Cell
from sbb.results.bundle import make_manifest
from sbb.results.replication import ReplicationRecord
from sbb.results.store import LocalResultStore, ResultStore
from sbb.results.table import markdown_table
from sbb.results.validate import validate as validate_files
from sbb.results.verification import verify
from sbb.runner.executors import executor_for
from sbb.runner.planner import Planner
from sbb.runner.site import SiteProfile
from sbb.runner.spec import BenchmarkSpec, Tolerance
from sbb.site.builder import LeaderboardBuilder
from sbb.utils.errors import SbbError
from sbb.utils.storage import ArtifactCache, HubStore, LocalStore

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


def _execute(cells: list[Cell], store: ResultStore, site: SiteProfile, out: Path) -> bool:
    """Run or submit the cells without a result, and return whether their results are in.

    A local site with one worker runs them in this process, one with more workers in a pool of
    sbb run-cell processes. A site with a scheduler submits them and returns False.
    """
    pending = Planner.pending(cells, store)
    console.print(f"{len(cells)} cells, {len(pending)} pending")
    if not pending:
        return True
    if site.scheduler == "none" and site.max_parallel == 1:
        game = site.game("in-process")
        for cell in pending:
            _run_cell(game, cell, store)
        return True
    ids = executor_for(site).submit(pending, out)
    if site.scheduler != "none":
        console.print(f"submitted as job {', '.join(ids)}, run this command again when it ends")
        return False
    for cell in pending:
        if result := store.get(cell.key()):
            _print(result)
    return True


def _run_cell(game: Game, cell: Cell, store: ResultStore) -> bool:
    """Run one cell through the Game, save the result, and return False if it failed."""
    console.print(f"[bold]{cell.scenario.name}[/] {cell.attack} × {cell.defense} seed {cell.seed}")
    result = game.run(cell)
    store.save(result)
    _print(result)
    return result.outcome != "failed"


def _print(result: CellResult) -> None:
    metrics = " ".join(f"{k}={v:.4f}" for k, v in result.metrics.items())
    console.print(f"  {result.outcome} {metrics} {result.reason or ''}")


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
    """Plan the cells of SPEC, run or submit the pending ones, and write table.md when done."""
    bench = BenchmarkSpec.load(spec)
    store = LocalResultStore(out)
    cells = Planner(bench).cells(track, attack, defense)
    if not _execute(cells, store, _site(site), out):
        return
    failures = len(Planner.pending(cells, store))
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
    if not _execute(cells, store, _site(site), out):
        return
    if Planner.pending(cells, store):
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


zoo = typer.Typer(no_args_is_help=True, help="Share cached models through a Hugging Face Hub repo.")
app.add_typer(zoo, name="zoo")
# TODO([FILL]) default to <hf-org>/sbb-zoo once the organization exists (docs/PLAN.md 5.2).
Repo = Annotated[str, typer.Option(help="Hub repository of the zoo, <hf-org>/sbb-zoo.")]


@zoo.command("push")
def zoo_push(version: str, repo: Repo, out: Out = Path("results"), site: Site = None) -> None:
    """Upload, in one commit, the cached artifacts that the results of VERSION reference.

    Needs a Hub token with write access (HF_TOKEN, or hf auth login).
    """
    local = LocalStore(_site(site).cache_dir / "artifacts")
    results = LocalResultStore(out).query(benchmark_version=version)
    keys = sorted({key for result in results for key in result.artifacts.values()})
    found = {key: path for key in keys if (path := local.get(key))}
    if found:
        HubStore(repo, prefix=version).put_many(found)
    console.print(f"pushed {len(found)} of {len(keys)} artifacts to {repo} under {version}")


@zoo.command("pull")
def zoo_pull(
    version: str,
    repo: Repo,
    revision: Annotated[str, typer.Option(help="Commit or branch of the zoo.")] = "main",
    site: Site = None,
) -> None:
    """Copy the artifacts of VERSION from the Hub into the local cache, which runs look up first."""
    hub = HubStore(repo, revision, prefix=version)
    cache = ArtifactCache(LocalStore(_site(site).cache_dir / "artifacts"), hub)

    def missing(directory: Path) -> None:
        raise SbbError("listed on the Hub but not downloadable")

    keys = hub.keys()
    for key in keys:
        cache.get_or_build(key, missing)
    console.print(f"{len(keys)} artifacts of {version} are in the local cache")


@app.command()
def bundle(
    plugin: Annotated[str, typer.Option(help="The attack or defense whose results to bundle.")],
    out: Out = Path("results"),
    version: Annotated[str | None, typer.Option(help="Only this benchmark version.")] = None,
    models_repo: Annotated[str | None, typer.Option(help="Hub repository of the models.")] = None,
    models_revision: Annotated[str, typer.Option(help="Revision of models_repo.")] = "main",
) -> None:
    """Write the manifest of PLUGIN's results to OUT/manifests/<bundle_id>.json."""
    results = [
        r
        for r in LocalResultStore(out).query()
        if plugin in (r.cell.attack, r.cell.defense)
        and r.cell.track != "replication"
        and version in (None, r.cell.benchmark_version)
    ]
    if not results:
        console.print(f"[red]no results of {plugin} under {out}[/]")
        raise typer.Exit(1)
    manifest = make_manifest(out, results, models_repo, models_revision)
    path = out / "manifests" / f"{manifest.bundle_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n")
    console.print(f"{len(results)} results of {plugin} in {path}")


@app.command()
def validate(paths: list[Path]) -> None:
    """Validate the cards, results, manifests and records in PATHS, as CI does."""
    checked, problems = validate_files(paths)
    for path, found in problems.items():
        for problem in found:
            console.print(f"[red]{path}[/]: {problem}")
    console.print(f"{checked} files checked, {len(problems)} with problems")
    if problems:
        raise typer.Exit(1)


@app.command()
def reproduce(
    result_json: Path,
    site: Site = None,
    verifications: Annotated[Path, typer.Option(help="Root of the verification records.")] = Path(
        "verifications"
    ),
    verifier: Annotated[
        str, typer.Option(envvar="SBB_SUBMITTER", help="Your GitHub handle.")
    ] = "unknown",
    verifier_org: Annotated[
        str, typer.Option(envvar="SBB_SUBMITTER_ORG", help="Your organization.")
    ] = "unknown",
    tolerance_pp: Annotated[
        float | None, typer.Option(help="Allowed difference. Default: the track's.")
    ] = None,
) -> None:
    """Rerun the cell of RESULT_JSON on this site and write a Verification record."""
    original = CellResult.model_validate_json(result_json.read_text())
    rerun = _site(site).game("reproduce").run(original.cell)
    defaults = Tolerance()
    track_pp = defaults.lite_pp if original.cell.track == "lite" else defaults.full_pp
    check = verify(original, rerun, verifier, verifier_org, tolerance_pp or track_pp)
    path = check.path(verifications, original)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(check.model_dump_json(indent=2) + "\n")
    deltas = " ".join(f"{k}={v:+.4f}" for k, v in check.delta.items())
    console.print(f"{check.verdict} {deltas} {check.notes or ''}")
    console.print(f"record written to {path}")


site_app = typer.Typer(no_args_is_help=True, help="The leaderboard site.")
app.add_typer(site_app, name="site")


@site_app.command("build")
def site_build(
    out: Annotated[Path, typer.Option(help="Directory of the site.")] = Path("_site"),
    results: Annotated[Path, typer.Option(help="Results directory.")] = Path("results"),
    replications: Annotated[Path, typer.Option(help="Replication records.")] = Path("replications"),
) -> None:
    """Build the static, interactive leaderboard into OUT, for GitHub Pages."""
    records = [ReplicationRecord.load(p) for p in sorted(replications.glob("*.yaml"))]
    builder = LeaderboardBuilder(LocalResultStore(results).query(), records)
    builder.build(out)
    console.print(f"{len(builder.results)} results and {len(records)} records in {out}/index.html")


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
