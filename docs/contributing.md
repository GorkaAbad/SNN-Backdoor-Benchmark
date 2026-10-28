# Contributing

sbb takes four kinds of contributions: a defense, an attack, a verification of a published
cell, and a fix to the harness. A method is a plugin: one Python file, one YAML card and one
import line. Adding a method never touches the core.

Read [threat-model.md](threat-model.md) first. It says what each kind of attack may do and what
each defense may see. Ports of published methods also follow [porting.md](porting.md).

## Set up

```bash
git clone https://github.com/GorkaAbad/SNN-Backdoor-Benchmark
cd SNN-Backdoor-Benchmark
uv sync
uv run pytest -q -m "not slow"
```

You need Python 3.11 and [uv](https://docs.astral.sh/uv/). The tests run on CPU with synthetic
data, so no GPU and no dataset download are needed.

## Add a defense

**1. Pick the stage.** The stage fixes the base class, the method you write and the resources
you may declare.

| Stage | Base class | You write | Resources you may declare |
|---|---|---|---|
| `pre_training` | `DataDefense` | `sanitize(view)` returns the positions of `train_set` to keep | `train_set` |
| `in_training` | `TrainingDefense` | `hooks(view)` returns `TrainerHooks` for the victim's training | `train_set`, `training_loop` |
| `post_training` | `ModelDefense` | `repair(view)` returns the repaired `SNNModel` | `model`, optional `clean_subset` |
| `inference` | `InputDefense` | `fit(view)` sets `self.threshold`, `score(x)` returns one score per input, higher is more suspicious | `model`, optional `clean_subset` |
| `model_detection` | `DetectionDefense` | `verdict(view)` returns a score, higher is more likely backdoored | `model`, optional `clean_subset` |

**2. Write the plugin**, `src/sbb/defenses/<name>.py`. This is the whole of `fine_tuning`:

```python
"""Fine-tuning on the clean subset."""

from pathlib import Path

from sbb.core.cards import DefenseCard
from sbb.core.plugins import ModelDefense
from sbb.core.proxies import DefenderView
from sbb.core.registry import DEFENSES
from sbb.models.snn import SNNModel


@DEFENSES.register("fine_tuning")
class FineTuning(ModelDefense):
    """Retrains the model on clean data with the scenario recipe."""

    card = DefenseCard.load(Path(__file__).parent / "cards" / "fine_tuning.yaml")

    def repair(self, view: DefenderView) -> SNNModel:
        """Fine-tune the model on the clean subset."""
        data = view.clean_loader(batch_size=self.hparams["batch_size"])
        return view.trainer.fit(view.model, data, epochs=self.hparams["epochs"])
```

**3. Write the card**, `src/sbb/defenses/cards/<name>.yaml`:

```yaml
name: fine_tuning
version: 1.0.0
title: Fine-tuning on clean data
stage: post_training
view:
  resources: [model, clean_subset]
  clean_fraction: 0.05          # the share of the defender pool you use
hparams:
  default: {epochs: 5, batch_size: 32}
  nmnist/tiny_lif_cnn: {epochs: 3}   # optional, merged over default for that dataset and model
```

A published method also sets `paper`, `code_reference` and `license_note`. A defense designed
for ANNs explains its SNN adaptation in `adaptation_note`.

**4. Register it.** Add the module to the import in `src/sbb/defenses/__init__.py`.

**5. Run its contract tests**: `uv run sbb test --plugin <name>`. They check the card, the
stage and the view on synthetic data.

The rules a defense follows:

- **Only the view.** A defense sees a `DefenderView` with exactly the declared resources:
  `view.model` (a copy of the victim), `view.train_set`, `view.clean_loader(batch_size)`, plus
  `view.trainer`, `view.rng` and `view.scenario`, which every defense gets. Reading an
  undeclared resource raises `ThreatModelViolation`.
- **One training loop.** Train with `view.trainer`. To change how the victim trains, return
  `TrainerHooks`.
- **Hyperparameters in the card.** `self.hparams` holds the values for the cell's dataset and
  model. Do not tune them per cell: the defender does not know the trigger.
- **Shared SNN parts.** Use `sbb.defenses.utils` (`channel_means`, `zero_channels`,
  `conv_blocks`, `superpose`) and `SNNModel.probe` for spikes and membrane potentials. Add a
  missing utility there, not inline.
- **Seeded randomness.** Draw from `view.rng`.

## Add an attack

**1. Pick the scenario kind** and what your attacker can do.

| Kind | The attacker | You subclass |
|---|---|---|
| `data_poisoning` | changes up to `poison_rate` of the train split, then the harness trains the victim | `PoisoningAttack`, usually `DirtyLabelPoisoning` |
| `training_control` | trains the victim itself | `ModelAttack` and write `make_model(view)` |
| `supply_chain` | receives a clean trained model and changes its weights | `ModelAttack` and write `make_model(view)` |

**2. Build on the shared parts.** Triggers live in `sbb.attacks.triggers` (raw events),
`sbb.attacks.frame_triggers` and `sbb.attacks.temporal_triggers`. Selectors are in
`sbb.attacks.selectors`, and label policies in `sbb.attacks.labels`. The static trigger of
Sneaky Spikes is this:

```python
@ATTACKS.register("sneaky_static")
class SneakyStatic(DirtyLabelPoisoning):
    """Sneaky Spikes, static trigger (port of the official code)."""

    card = AttackCard.load(Path(__file__).parent / "cards" / "sneaky_static.yaml")
    trigger_cls = FramePatchTrigger
```

A new trigger subclasses `Trigger`, with `apply(sample, rng)` on raw events, or `FrameTrigger`,
with `apply(frames, rng)` on encoded frames `[T, P, H, W]`. The card's `trigger_domain` says
which (`raw_events` or `frames`). Draw randomness only from the `rng` argument: the harness seeds
it per sample, so a sample gets the same trigger in every run. A trigger with weights, such as
a learned one, returns them from `learned()` so the harness can store them.

**3. Write the card**, `src/sbb/attacks/cards/<name>.yaml`, with `kind`, `capabilities`
(`labels`, `trigger_domain`), `knowledge` (`architecture`, `data`, `defense`) and `hparams`.
The harness checks the labels you declare against the poison set you return.

**4. Register and test** as for a defense: add the module to `src/sbb/attacks/__init__.py`, then
`uv run sbb test --plugin <name>`.

**5. A published attack needs a replication record** in `replications/<name>.yaml`
([porting.md](porting.md)).

The `AttackerView` gives every attack the clean train split and loaders over its poison set.
Model attacks also get `view.arch` and `view.trainer`, and `supply_chain` attacks get
`view.clean_model`. With `architecture: unknown`, a poisoning attack cannot read `view.arch`.

## Run your method on the benchmark

```bash
uv run sbb plan benchmarks/<version>.yaml --defense <name>     # its cells, with N/A reasons
uv run sbb run benchmarks/<version>.yaml --defense <name> --site <profile>
uv run sbb bundle --plugin <name>                              # writes results/manifests/<id>.json
uv run sbb validate results
```

Use `--attack <name>` for an attack. The planner picks every cell your method is compatible
with. A cold lite cell takes about 80 minutes on 4 CPU cores, and cells that share a cached
model take minutes. Full-track cells need a GPU.
[sites.md](sites.md) describes site profiles, clusters and the container image.

Set `SBB_SUBMITTER` (your GitHub handle) and `SBB_SUBMITTER_ORG` before running, so your results
name you. Results land in `results/<version>/<key>.json`. Never edit them: the key is the hash
of the cell, and `sbb validate` checks it.

## Open a pull request

Include the plugin, the card, the import line, the replication record of a published attack,
and your results with their manifest. The pull request template has the checklist. CI runs ruff,
mypy, import-linter, the tests with the contract suite, `sbb validate` and the smoke benchmark.
A maintainer reviews the code. Code review is the control against deliberate cheating, for
example an input detector that recognises the public test set.

After the merge, the `verify-lite` workflow reruns your lite cells on GitHub runners. A match
within 3 percentage points on every metric makes a cell `reproduced` on the leaderboard, a mismatch makes it
`disputed`. Full-track cells stay `self-reported` until a group from another organization
reproduces them ([verifying.md](verifying.md)).

## Attacks that arrive after a freeze

A benchmark version is frozen when it is released. An attack submitted after that waits one
benchmark version in the delayed battery, so defense authors cannot tune against it before it
is scored. Send it to the maintainers instead of opening a public pull request. They keep it in
a private copy of the repository, run it against every defense of the current version on their
own sites with the same harness, and keep the results private. When the next version is
released, the attack lands on `main` by an ordinary pull request with its results as they were
computed. The cell keys hash the method names, versions and hyperparameters, not the
repository, so the results match the published code. Their provenance names a commit of the
private copy. Anyone can still rerun them from the public release with `sbb reproduce`.

No attack is pending now (maintainer decision of 2026-10-28). The maintainers decide which
attacks are pending at each release.

## Verify a published cell

Anyone can rerun any published cell:

```bash
SBB_SUBMITTER=<handle> SBB_SUBMITTER_ORG=<org> \
  uv run sbb reproduce results/<version>/<key>.json --site <profile>
```

It writes `verifications/<version>/<key>/<handle>-<time>.json` with the verdict. Open a pull
request that adds the record. [verifying.md](verifying.md) covers spot checks of a submission,
the model zoo and the tolerance.

- **Reciprocity.** Submitting a full-track method commits you to verify 3 cells of another
  submission, which the maintainers assign.
- **Open challenges.** Anyone may rerun any published cell at any time and submit the record. A
  mismatch makes the cell `disputed` until the maintainers resolve it.

## Change the harness

Read `docs/PLAN.md` first. Its section 6 holds the interfaces, which are contracts: propose a
change there before changing code. The code follows a few conventions:

- Type hints everywhere, mypy strict for `sbb.core`, `sbb.data` and `sbb.models`.
- ruff for lint and format, line length 100.
- Google-style docstrings on public classes and functions.
- `logging` for messages. Only the CLI prints.
- A test before the code. Golden tests in `tests/golden/` freeze metrics: a change there is a
  bug or a new benchmark version, so ask before updating them.

Before pushing, run:

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run lint-imports
uv run pytest -q
```
