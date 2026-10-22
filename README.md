# sbb: SNN backdoor benchmark

A benchmark for backdoor attacks and defenses on spiking neural networks with neuromorphic event data.
Attacks and defenses are plugins with YAML cards. The harness evaluates every compatible pair under typed threat models.

The design and the roadmap are in [docs/PLAN.md](docs/PLAN.md). The project is at v1, the public alpha (section 13.3).

| Doc | For |
|---|---|
| [docs/contributing.md](docs/contributing.md) | adding an attack or a defense, running it, opening a pull request, verifying a cell |
| [docs/threat-model.md](docs/threat-model.md) | what attacks may do and what defenses may see |
| [docs/porting.md](docs/porting.md) | porting a published method, replication records |
| [docs/sites.md](docs/sites.md) | site profiles, clusters with Slurm, the container image |
| [docs/measurements.md](docs/measurements.md) | runtimes and determinism checks |

## Install

```bash
uv sync    # CPU or CUDA, the locked torch wheel is the CUDA 13 build on Linux
```

## Run

```bash
uv run sbb run benchmarks/smoke.yaml --site sites/examples/local-cpu.yaml   # synthetic data, seconds
uv run sbb plan benchmarks/toy.yaml                                         # list cells and N/A reasons
uv run sbb run benchmarks/toy.yaml --site sites/examples/local-cpu.yaml     # N-MNIST, about 1.5 hours on 4 CPU cores
uv run sbb run-cell cell.json --out results                                 # one cell, the entry point of every executor
uv run sbb test --plugin static_patch                                       # contract tests of one plugin
```

`sbb run` writes one JSON per cell to `results/<version>/<key>.json` and a Markdown table to `results/<version>/table.md`.
Cells that already have a result are skipped.

N-MNIST (about 1.2 GB) is downloaded to `<cache_dir>/datasets/nmnist` on first use.
Frames and models are cached in `<cache_dir>/artifacts`.

## Run on a GPU cluster

Write a site profile in `sites/private/`, which git ignores, for example `sites/private/my-cluster.yaml`:

```yaml
name: my-cluster
device: cuda
cache_dir: /scratch/<user>/sbb-cache
```

Then run `uv run sbb run benchmarks/toy.yaml --site sites/private/my-cluster.yaml` inside a GPU job.
With `scheduler: slurm` in the profile, `sbb run` submits the cells as a Slurm job array instead, optionally in the container image ([docs/sites.md](docs/sites.md)).
To check that GPU training is deterministic, run `uv run pytest -m gpu -v`. The result is recorded in [docs/measurements.md](docs/measurements.md).

## Develop

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run lint-imports
```

A new attack or defense is one module, one class and one YAML card in `src/sbb/attacks` or `src/sbb/defenses`, imported in the package `__init__`.
The contract tests in `tests/contracts` cover it automatically. [docs/contributing.md](docs/contributing.md) walks through it.
