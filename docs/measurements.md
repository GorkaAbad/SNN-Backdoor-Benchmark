# v0 measurements

Tasks 13 and 14 of docs/PLAN.md section 13.2. These numbers fix the lite-track recipe.

## Runtime of one N-MNIST lite cell

Scenario `nmnist_tiny` of `benchmarks/toy.yaml`: N-MNIST, T = 10, `tiny_lif_cnn`, 10 epochs, batch 64, Adam, SpikingJelly `torch` backend.

### CPU, 4 cores

Machine: Apple M3 laptop, `OMP_NUM_THREADS=4`, torch 2.14.1, SpikingJelly 0.0.0.0.14, 2026-10-15.
Code: an uncommitted working tree from before the first commit (de8f2db). On this benchmark it
differs from commit 3daea23, used on GPU, in no way that changes a weight or a metric: a
`no_attack` victim now gets its own model key but still reuses the clean weights, and the
clean-subset cap of 3daea23 changes nothing here because the pool holds exactly 3 000 samples.
The clean and `static_patch` model keys are the same in both runs.

| Stage | Wall time | Notes |
|---|---|---|
| Frame cache, 70 000 samples | 5.1 min | once per (dataset, split, representation), 3.2 GB uint16 |
| One training epoch, 60 000 samples | 3.5 min | about 225 ms per batch of 64 (938 batches) |
| Clean model, 10 epochs | 35.3 min | shared by every cell of the scenario and seed |
| Backdoored model, 10 epochs | 39.2 min | shared by every defense of the attack, slower epochs because other jobs shared the machine |
| Fine-tuning, 5 epochs on 3 000 samples | 0.8 min | |
| Evaluation, clean and triggered test sets | about 1 min per cell | |
| Whole toy benchmark, 4 cells, cold cache | 86.8 min | dataset already downloaded |

A cold `static_patch` x `fine_tuning` cell costs about 80 minutes on 4 CPU cores, almost all of
it in the two training runs. With the clean and backdoored models cached, a post-training
defense cell costs a few minutes.

### GPU, one NVIDIA GH200

torch 2.14.1+cu130, CUDA 13.0, SpikingJelly 0.0.0.0.14, 8 CPU cores for the job, commit 3daea23,
2026-10-15.
The whole toy benchmark (4 cells, cold cache, dataset download included) took 7 min 40 s.

| Stage | Wall time |
|---|---|
| Download and extraction of N-MNIST (once) | 1.7 min |
| Frame cache, 70 000 samples | 51 s |
| One training epoch, 60 000 samples | about 12 s |
| Clean model, 10 epochs | 2.0 min |
| Backdoored model, 10 epochs | 2.0 min |
| Fine-tuning, 5 epochs on 3 000 samples | 6 s |

### Lite-track recipe

A cold lite cell takes about 80 minutes on 4 cores of an Apple M3 and a few minutes on one GPU,
so the recipe of `benchmarks/toy.yaml` (10 epochs, batch 64, T = 10) can stay. That it fits in the
6-hour limit of a GitHub-hosted runner (4 x86 vCPUs) is an extrapolation: no runner was measured,
and the 4.5x margin assumes a runner about as fast as the M3. The v1 definition of done checks it
on a runner.

### Toy results, seed 0

CPU (4 cores):

| attack \ defense | no_defense | fine_tuning |
|---|---|---|
| no_attack | CA 98.67 · ASR 0.12 · RA 98.57 | CA 98.52 · ASR 0.13 · RA 98.44 |
| static_patch | CA 98.65 · ASR 100.00 · RA 0.00 | CA 98.49 · ASR 99.89 · RA 0.11 |

GPU (GH200):

| attack \ defense | no_defense | fine_tuning |
|---|---|---|
| no_attack | CA 98.65 · ASR 0.13 · RA 98.57 | CA 98.42 · ASR 0.11 · RA 98.37 |
| static_patch | CA 98.66 · ASR 100.00 · RA 0.00 | CA 98.45 · ASR 99.93 · RA 0.07 |

CPU and GPU differ by at most 0.10 percentage points (CA of `no_attack` x `fine_tuning`), inside
the lite tolerance of 1.0. Each toy cell ran once on each machine. That a rerun on the same hardware
gives bit-identical weights is shown by the determinism tests below, on a small synthetic cell.

## Determinism on GPU and the SpikingJelly `cupy` backend

Checked on 2026-10-15 with `tests/gpu/test_determinism_gpu.py`, which trains the same cell twice
from empty caches and compares the metrics and the sha256 of the weights.
Machine: NVIDIA GH200 120GB (aarch64), torch 2.14.1+cu130, CUDA 13.0, SpikingJelly 0.0.0.0.14.

| Backend | Commit | Result |
|---|---|---|
| `torch` | 3daea23 | deterministic: identical metrics and identical weights |
| `cupy` | de8f2db | does not run: SpikingJelly calls `np.int` in `auto_cuda/base.py`, which NumPy 1.24 removed |

The `cupy` row comes from the version of the test at commit de8f2db, which also covered `cupy`
(cupy-cuda13x 14.2.0, NumPy 2.4.6). The test now covers only `torch`.

`ArchSpec` therefore accepts only the `torch` backend. Supporting `cupy` needs a SpikingJelly
release without `np.int` (2.0.0rc1 is on PyPI, untested). Pinning NumPy below 1.24 is not an
option: sbb requires NumPy 1.26 or later, and cupy-cuda13x 14.2.0 requires NumPy 2.0 or later.

On CPU, two runs with the same seed also give identical metrics and identical weights
(`tests/integration/test_game.py`).
