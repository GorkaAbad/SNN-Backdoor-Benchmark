# v0 measurements

Tasks 13 and 14 of docs/PLAN.md section 13.2. These numbers fix the lite-track recipe.

## Runtime of one N-MNIST lite cell

Scenario `nmnist_tiny` of `benchmarks/toy.yaml`: N-MNIST, T = 10, `tiny_lif_cnn`, 10 epochs, batch 64, Adam, SpikingJelly `torch` backend.

### CPU, 4 cores

Machine: Apple M-series laptop, `OMP_NUM_THREADS=4`, torch 2.14.1, SpikingJelly 0.0.0.0.14, 2026-10-15.

| Stage | Wall time | Notes |
|---|---|---|
| Frame cache, 70 000 samples | 5.1 min | once per (dataset, split, representation), 3.2 GB uint16 |
| One training epoch, 60 000 samples | 3.5 min | about 195 ms per batch of 64 |
| Clean model, 10 epochs | 35.3 min | shared by every cell of the scenario and seed |
| Backdoored model, 10 epochs | PENDING | shared by every defense of the attack |
| Fine-tuning, 5 epochs on 3 000 samples | PENDING | |
| Evaluation, clean and triggered test sets | PENDING | |
| Full toy benchmark, 4 cells, cold cache | PENDING | |

A cold `static_patch` x `fine_tuning` cell therefore costs about 75 minutes on 4 CPU cores, most of it in the two training runs.
With the clean and backdoored models cached, a post-training defense cell costs minutes.

Toy results (seed 0):

PENDING

### GPU

PENDING, needs the cluster. Run inside a GPU job:

```bash
uv run sbb run benchmarks/toy.yaml --site sites/private/<cluster>.yaml --out /tmp/toy-gpu
```

and copy the stage times from the log (`building artifact` and `epoch` lines) into a table like the one above.

## Determinism of the SpikingJelly `cupy` backend

PENDING, needs a CUDA device. Run:

```bash
uv sync --extra cuda
uv run pytest -m gpu -v
```

`tests/gpu/test_determinism_gpu.py` trains the same cell twice from empty caches with each backend and compares metrics and the sha256 of the weights.
Record here, for each backend, whether the two runs match, with the GPU model, CUDA version and cupy version.
If `cupy` does not match, it is allowed only for self-reported runs (docs/PLAN.md section 9).

On CPU, two runs with the same seed give identical metrics and identical weights (`tests/integration/test_game.py`).
