<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.png">
  <img alt="SNN Backdoor Benchmark" src="docs/assets/logo.png" width="560">
</picture>

**A reproducible benchmark for backdoor attacks and defenses on spiking neural networks with neuromorphic data.**

[![CI](https://github.com/GorkaAbad/SNN-Backdoor-Benchmark/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/GorkaAbad/SNN-Backdoor-Benchmark/actions/workflows/ci.yml)
[![Leaderboard](https://img.shields.io/badge/leaderboard-online-3559c7)](https://gorkaabad.github.io/SNN-Backdoor-Benchmark/)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20models-sbb--zoo-f5b800)](https://huggingface.co/sbb-zoo/checkpoints)
![Python 3.11](https://img.shields.io/badge/python-3.11-3776ab)
![Status](https://img.shields.io/badge/status-public%20alpha-e0a64a)
[![License: MIT](https://img.shields.io/badge/license-MIT-1f7a4d)](LICENSE)

[**Leaderboard**](https://gorkaabad.github.io/SNN-Backdoor-Benchmark/) ·
[**Methods**](#methods) ·
[**Quick start**](#quick-start) ·
[**Documentation**](#documentation) ·
[**Contributing**](docs/contributing.md)

</div>

---

Backdoor results on spiking neural networks are hard to compare today. Each paper brings its own
training loop, data pipeline, network and SNN adaptation of the defenses. **sbb** runs every attack
against every defense in one harness, under explicit threat models, and publishes each number with
its full provenance and its verification status.

- **One harness.** Every evaluation is the same security game: the same data pipeline, the same
  trainer, the same metrics, for every attack and defense pair.
- **Typed threat models.** An attack receives an attacker view and a defense a defender view, both
  built from the method's card. Using a resource the card does not declare fails the cell. Pairs
  whose threat models do not fit are reported as not applicable, with the reason.
- **Faithful ports.** Published SNN methods are ported on shared building blocks, checked
  against their official code, and replicated on their papers' own settings before they enter
  the benchmark. Attacks ported from ANNs say so in their card.
- **Verified results.** Every result carries the seeds, versions, device and site that produced
  it. Lite results are rerun on GitHub runners after merge, and the leaderboard shows which numbers
  were reproduced.

## Leaderboard

The [leaderboard](https://gorkaabad.github.io/SNN-Backdoor-Benchmark/) is an interactive page
built from the results in this repository. Filter by benchmark version, track, dataset, network,
attack, defense and metric, then explore:

- a matrix of attacks against defenses for any metric,
- one attack across defenses, or one defense across attacks,
- a sortable table with each cell's status, mean and standard deviation over seeds,
- each ported attack against the numbers of its paper,
- the defenses of each scenario as a Pareto front of clean accuracy against attack success,
- the card of every method, with its paper, code and SNN adaptation, and for a defense the
  adaptive attacks that break it.

Every view can be downloaded as CSV for your own analysis: the shown matrix or table, all results
with the mean and standard deviation over seeds
([results.csv](https://gorkaabad.github.io/SNN-Backdoor-Benchmark/results.csv)), and the run
status of every cell with its per-seed metrics
([status.csv](https://gorkaabad.github.io/SNN-Backdoor-Benchmark/status.csv)).

## Methods

Methods are listed oldest first, as on the leaderboard. Each one is a plugin with a YAML card that
declares its threat model and hyperparameters.

### Attacks

| Attack | Paper | Threat model | Trigger | Replication |
|---|---|---|---|---|
| TaCT, ported from ANNs (`tact`) | ANN attack, [Tang et al., USENIX Security 2021](https://arxiv.org/abs/1908.00686) | data poisoning, adaptive (STRIP) | patch that flips one source class | not applicable |
| Adap-Blend, ported from ANNs (`adap_blend`) | ANN attack, [Qi et al., ICLR 2023](https://arxiv.org/abs/2205.13613) | data poisoning, adaptive (activation clustering) | blended pattern, partial when poisoning | not applicable |
| Adap-Patch, ported from ANNs (`adap_patch`) | ANN attack, [Qi et al., ICLR 2023](https://arxiv.org/abs/2205.13613) | data poisoning, adaptive (activation clustering) | faint corner patches, opaque at test time | not applicable |
| Sneaky Spikes, static (`sneaky_static`) | [Abad et al., NDSS 2024](https://arxiv.org/abs/2302.06279) | data poisoning | static patch in every frame | replicated |
| Sneaky Spikes, moving (`sneaky_moving`) | [Abad et al., NDSS 2024](https://arxiv.org/abs/2302.06279) | data poisoning | patch moving across frames | explained |
| Sneaky Spikes, smart (`sneaky_smart`) | [Abad et al., NDSS 2024](https://arxiv.org/abs/2302.06279) | data poisoning | patch in the most active region | explained |
| Sneaky Spikes, dynamic (`sneaky_dynamic`) | [Abad et al., NDSS 2024](https://arxiv.org/abs/2302.06279) | training control | learned, input-specific | explained |
| Static patch on raw events (`static_patch`) | variant of Sneaky Spikes | data poisoning | patch added to the event stream | not applicable |
| Flashy Backdoor (`flashy`) | [Riaño et al., 2024](https://arxiv.org/abs/2411.03022) | data poisoning | full-frame flash | explained |
| T-Backdoor, rate, latency and jitter (`tbackdoor_*`) | [Miah et al., 2026](https://arxiv.org/abs/2609.30119) | data poisoning | reordered frames only | replicated |
| Fine-tuning injection (`finetune_injection`) | baseline | supply chain | static patch, fine-tuned into a clean model | not applicable |

A port is **replicated** when its mean over three seeds is within 3 percentage points of every
number its paper reports, and **explained** when a maintainer documented why it misses
([records](replications/)). T-Backdoor is part of benchmark 1.0. TaCT, Adap-Blend and
Adap-Patch are ANN image attacks ported to SNNs and event data, not attacks designed for SNNs.
Their papers have no event setting to replicate.

### Defenses

| Defense | Paper | Stage | The defender sees |
|---|---|---|---|
| Fine-pruning (`fine_pruning`) | [Liu et al., RAID 2018](https://arxiv.org/abs/1805.12185) | post-training | model, clean data |
| Spectral signatures (`spectral_signatures`) | [Tran et al., NeurIPS 2018](https://arxiv.org/abs/1811.00636) | pre-training | training set |
| Activation clustering (`activation_clustering`) | [Chen et al., 2018](https://arxiv.org/abs/1811.03728) | pre-training | training set |
| STRIP (`strip`) | [Gao et al., ACSAC 2019](https://arxiv.org/abs/1902.06531) | inference | model, clean data |
| STRIP per time step (`strip_per_step`) | sbb variant of STRIP | inference | model, clean data |
| NAD (`nad`) | [Li et al., ICLR 2021](https://arxiv.org/abs/2101.05930) | post-training | model, clean data |
| ABL (`abl`) | [Li et al., NeurIPS 2021](https://arxiv.org/abs/2110.11571) | in-training | training set, training loop |
| ANP (`anp`) | [Wu and Wang, NeurIPS 2021](https://arxiv.org/abs/2110.14430) | post-training | model, clean data |
| CLP (`clp`) | [Zheng et al., ECCV 2022](https://arxiv.org/abs/2208.03111) | post-training | model |
| TMPBD (`tmpbd`) | [Li et al., 2025](https://arxiv.org/abs/2510.06629) | model detection | model |
| NDSBM (`ndsbm`) | [Li et al., 2025](https://arxiv.org/abs/2510.06629) | post-training | model, clean data |
| Fine-tuning (`fine_tuning`) | baseline | post-training | model, clean data |
| Fewer time steps (`fewer_steps`) | SNN-native baseline | post-training | model, clean data |
| Threshold recalibration (`threshold_recalibration`) | SNN-native baseline | post-training | model, clean data |

Defenses designed for ANNs are adapted to SNNs once, with shared utilities, and the card of each
defense describes its adaptation. BadSNN waits for its official code.

### Datasets and networks

| Dataset | Classes | Track | Network |
|---|---|---|---|
| [N-MNIST](https://www.garrickorchard.com/datasets/n-mnist) | 10 | lite, on a CPU | `tiny_lif_cnn`, two conv blocks with LIF neurons |
| [DVS128 Gesture](https://research.ibm.com/interactive/dvsgesture/) | 11 | full, on a GPU | `dvsgesture_net`, the network of the ported papers |
| [CIFAR10-DVS](https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671) | 10 | full, on a GPU | `cifar10dvs_net`, the network of the ported papers |

Adapters read the raw event files and follow SpikingJelly's conventions exactly (sample order,
labels, splits), so results compare with the literature. Triggers act on raw events or on frames,
as each attack's card declares.

## Quick start

sbb needs Python 3.11 and [uv](https://docs.astral.sh/uv/). It runs on CPUs and NVIDIA GPUs.

```bash
git clone https://github.com/GorkaAbad/SNN-Backdoor-Benchmark
cd SNN-Backdoor-Benchmark
uv sync
```

Run the smoke benchmark, a few cells on synthetic data that finish in seconds:

```bash
uv run sbb run benchmarks/smoke.yaml --site sites/examples/local-cpu.yaml
```

List the cells of benchmark 1.0, with the reason for each cell that is not applicable:

```bash
uv run sbb plan benchmarks/v1.0.yaml
```

Add `--csv status.csv` to write every cell with its state (done, missing, not applicable or
failed), its verification and its metrics. The leaderboard publishes the same file for main.

Run the lite track on N-MNIST (downloaded on first use), then build the leaderboard locally:

```bash
uv run sbb run benchmarks/v1.0.yaml --track lite --site sites/examples/local-cpu.yaml
uv run sbb site build --out _site
python -m http.server -d _site 8000
```

Each cell is written to `results/<version>/<key>.json`, and cells that already have a result are
skipped. The trained models of the published results are on the Hugging Face Hub
([sbb-zoo/checkpoints](https://huggingface.co/sbb-zoo/checkpoints)): `uv run sbb zoo pull 1.0.0`
copies them into your cache, so defenses rerun on exactly those models. The full track needs a GPU. [docs/sites.md](docs/sites.md) covers workstations, Slurm
clusters and the container image.

## How it works

Every cell of the benchmark is one attack against one defense on one scenario and seed, played as
a security game ([threat model](docs/threat-model.md)):

1. The harness checks that the defense's resources fit the attack's threat model.
2. It trains a clean reference model and reserves a pool of clean data the attacker cannot touch.
3. The attacker poisons the training set, controls training or tampers with a trained model,
   through its view.
4. The defender acts at its stage, through its view.
5. The harness measures clean accuracy, attack success rate, robust accuracy and, for input
   detectors, rejection rates and AUROC, with the spike rates and synaptic operations of the
   evaluated model.

The benchmark has two tracks. The **lite** track runs on N-MNIST on a CPU, and GitHub Actions
reruns every lite result after merge to verify it. The **full** track runs the papers' networks on
DVS128 Gesture and CIFAR10-DVS on GPUs, and its results are self-reported until a group from
another organization reproduces them ([docs/verifying.md](docs/verifying.md)). Models, poisoned sets and frames are cached by content, so cells that
share a model train it once.

## Contributing

A new attack or defense is one Python file, one YAML card and one import line. It needs no change
to the core, and the contract tests cover it automatically.

```bash
uv run sbb test --plugin <name>                                   # contract tests on synthetic data
uv run sbb run benchmarks/v1.0.yaml --attack <name> --site <profile>
uv run sbb bundle --plugin <name>                                 # manifest of your results
```

Then open a pull request. [docs/contributing.md](docs/contributing.md) walks through it, and
[docs/porting.md](docs/porting.md) explains how to port a published method with its replication
record. Anyone can also verify a published cell with `sbb reproduce`.

## Documentation

| Document | Covers |
|---|---|
| [docs/contributing.md](docs/contributing.md) | adding an attack or a defense, running it, opening a pull request, verifying a cell |
| [docs/verifying.md](docs/verifying.md) | spot checks of full-track results, statuses, tolerance, submitting records |
| [docs/threat-model.md](docs/threat-model.md) | the security game, what attacks may do and what defenses may see, the metrics |
| [docs/porting.md](docs/porting.md) | porting a published method and its replication record |
| [docs/sites.md](docs/sites.md) | site profiles, Slurm clusters, Docker and Apptainer |
| [docs/measurements.md](docs/measurements.md) | runtimes, determinism and executor checks |
| [docs/PLAN.md](docs/PLAN.md) | design, interfaces and roadmap |

## Citing

If you use sbb, please cite the papers of the methods you evaluate (linked in the tables above and
in each card), and the benchmark. GitHub's "Cite this repository" and the Zenodo DOI of each
release come from [CITATION.cff](CITATION.cff).

<!-- TODO([FILL]): replace with the sbb paper once it exists. -->
```bibtex
@misc{sbb2026,
  title        = {{SNN Backdoor Benchmark}: A Reproducible Benchmark for Backdoor Attacks and
                  Defenses on Spiking Neural Networks},
  author       = {Gorka Abad},
  year         = {2026},
  howpublished = {\url{https://github.com/GorkaAbad/SNN-Backdoor-Benchmark}}
}
```

## License and acknowledgements

sbb is released under the [MIT License](LICENSE). Datasets keep their own licenses, and ported
methods state the license of their official code in their cards.

sbb builds on [SpikingJelly](https://github.com/fangwei123456/spikingjelly) and on the official
code of every ported method. Its metrics and conventions follow
[BackdoorBench](https://github.com/SCLBD/BackdoorBench) where they apply to SNNs.
