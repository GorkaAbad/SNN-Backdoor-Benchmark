# CLAUDE.md

## Project

`sbb` (working name) is a benchmark for backdoor attacks and defenses on spiking neural networks with neuromorphic event data. Contributors submit attacks and defenses as plugins. A harness evaluates every compatible attack and defense pair under typed threat models, and a leaderboard publishes the results with their verification status.

The full design and roadmap is in `docs/PLAN.md`. Before writing code, read sections 0 to 6 (section 4.8 lists the design rules). Before starting a task, read the section of the current version.

**Current target: v1** (`docs/PLAN.md`, section 13.3), on the `v1` branch. Do not implement features of later versions.

## Invariants

1. **One code path.** Every evaluation goes through `Game.run(cell)`, reached from `sbb run-cell`. Executors only decide where that command runs.
2. **Threat models are enforced by code.** Defenses receive a `DefenderView`, attacks an `AttackerView`. Never pass raw datasets, models or loaders to plugins.
3. **One training loop.** All training goes through `sbb.core.trainer.Trainer`. Plugins that change training provide `TrainerHooks`.
4. **One data pipeline.** Only `DataModule` builds DataLoaders. Attacks return a `PoisonSet` overlay, never a copy of a dataset.
5. **Raw events first.** Triggers act on `EventSample` objects. Representations run afterwards. Only an attack whose card declares `trigger_domain: frames` uses a `FrameTrigger`, which the DataModule applies to the encoded frames.
6. **Cards hold the configuration.** Every plugin has a YAML card validated by pydantic. Hyperparameters live in the card, not in code.
7. **Provenance everywhere.** Every result is a `CellResult` with a full `Provenance`. No metric is ever written by hand.
8. **Determinism.** Only the Game calls `seed_everything`. Randomized triggers use `rng_for(seed, sample.index)`.
9. **No pickle.** Weights in safetensors, arrays in npz with `allow_pickle=False`, metadata in JSON. Never call `torch.load` on downloaded files.
10. **No site-specific code.** Cluster details live in SiteProfile files outside the repository.
11. **Layering.** import-linter enforces the layers of `docs/PLAN.md` section 4.1. Lower layers never import higher ones.

## Conventions

- Python 3.11, src layout (`src/sbb`), uv for environments and the lockfile.
- Type hints everywhere. mypy strict for `sbb.core`, `sbb.data`, `sbb.models`.
- pydantic v2 for specs, cards and records. Frozen models for anything that is hashed.
- ruff for lint and format, line length 100.
- Google-style docstrings on public classes and functions.
- Use `logging`. Only the CLI prints, through rich.
- Small modules. One plugin per file, registered with the registry decorator and imported explicitly in the package `__init__`.
- Tensor layout between DataModule and SNNModel: `[B, T, P, H, W]`. SNNModel handles any internal transposition.
- Prose in docs and comments: plain, direct sentences, no em dashes, no semicolons.

## Commands

- `uv sync` installs the environment.
- `uv run pytest -q` runs all tests. `uv run pytest -m "not slow"` skips the slow ones.
- `uv run ruff check . && uv run ruff format --check .`
- `uv run mypy src`
- `uv run lint-imports`
- `uv run sbb run benchmarks/smoke.yaml --site sites/examples/local-cpu.yaml` runs the smoke benchmark.
- `uv run sbb test --plugin <name>` runs the contract tests of one plugin.

## Testing rules

- Unit tests never use the network or real datasets. Use `SyntheticEvents` and the fixtures in `tests/conftest.py`.
- Every registered plugin is covered by `tests/contracts/`. Do not duplicate those checks in plugin-specific tests.
- Golden tests in `tests/golden/` freeze the metrics of tiny CPU runs. Never update them to make a test pass. A change there means a bug or a new benchmark version, so ask first.
- Mark tests longer than 30 seconds with `@pytest.mark.slow` and GPU tests with `@pytest.mark.gpu`.

## How to work

- Follow the task list of the current version in order. One task per pull request when possible.
- Write the contract or unit test first, then the implementation.
- The interfaces in `docs/PLAN.md` section 6 are contracts. If one looks wrong, stop and propose the change in `docs/PLAN.md` before changing code.
- Items marked [DECIDE] use their default. Items marked [FILL] need a maintainer: leave a `TODO([FILL])` comment and continue.
- Ports of published methods follow `docs/PLAN.md` section 12, including the replication record.
- When a version's definition of done passes, report it and stop. Do not start the next version without approval.
