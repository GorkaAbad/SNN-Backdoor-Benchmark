# Threat model

Every cell of the benchmark is one attack against one defense, played as a security game. The
harness is the challenger. It gives the attacker and the defender exactly what their cards
declare, through views, and nothing else. This page is the reference for plugin authors.
`docs/PLAN.md` section 3 is the design record.

## The game

For a cell (attack, defense, scenario, seed), the harness:

1. **Checks compatibility.** If the defense needs a resource the scenario kind does not
   provide, the cell is N/A, with the missing resources as the reason. Nothing runs.
2. **Loads the data** and fixes the seed. It reserves the defender pool, a class-stratified
   share of the train split (the scenario's `defender_clean_fraction`, 5% by default) that the
   attacker cannot modify. The pool stays in the training set.
3. **Gets the clean model** of the scenario and seed. It is cached and shared by every cell.
4. **Lets the attacker act** through an `AttackerView`. A poisoning attack returns a poison set
   and a trigger. A model attack returns a model and a trigger.
5. **Trains the victim** on the poisoned train split, for a poisoning attack. The undefended
   backdoored model is cached and shared by every defense.
6. **Checks attack validity**: the backdoored model's clean accuracy may be at most 3
   percentage points below the clean model's. Otherwise the cell is `invalid_attack`.
7. **Lets the defender act** through a `DefenderView`, at its stage. A pre-training or
   in-training defense makes the harness train the victim again, on the kept samples or with
   the defense's hooks.
8. **Computes the metrics** on the clean test set and the triggered test set.

## The attacker

The scenario kind says what the attacker controls.

| Kind | The attacker | Its view also gives |
|---|---|---|
| `data_poisoning` | changes up to `poison_rate` of the train split, outside the defender pool. The harness trains the victim. | `arch` if the card knows the architecture |
| `training_control` | trains the victim itself, with any loss, schedule or neuron parameters, and returns the weights | `arch`, `trainer` |
| `supply_chain` | receives a clean trained model and changes its weights. Neuron parameters change only where they are weights, such as a trainable time constant. | `arch`, `trainer`, `clean_model` |

Every attacker sees the clean train split, a seeded `rng` and the public scenario description.

The card declares the rest:

- `capabilities.labels`: `dirty` (labels may change) or `clean` (they may not). The harness
  checks it against the poison set.
- `capabilities.trigger_domain`: `raw_events` (the default) or `frames`. A raw-event trigger
  edits the event stream before any representation. A frame trigger edits the encoded frames
  `[T, P, H, W]`, which assumes access after preprocessing, a stronger capability. The harness
  refuses a trigger of the other domain.
- `knowledge.architecture`: `known` or `unknown`. A poisoning attack that does not know the
  architecture cannot read `view.arch`.
- `knowledge.data`: `full`. Partial data knowledge is not supported yet.
- `knowledge.defense`: `oblivious`. Adaptive attacks, which declare `aware:<defense>` and run
  only against that defense, come with the adaptive track in v2.

## The defender

A defense declares the resources it needs:

- `train_set`: the training set as the victim received it, poisoned samples included. The
  defense never learns which samples are poisoned.
- `training_loop`: the right to hook into the victim's training run.
- `model`: a white-box copy of the trained victim.
- `clean_subset`: `clean_fraction` of the defender pool, clean and labelled.

Every defense also gets the shared `trainer`, a seeded `rng` and the public scenario
description (architecture, classes, sensor size, T). These are tools and public knowledge, not
resources. A defense that trains its own reference model with the trainer uses `train_set`, not
`training_loop`, because it does not hook into the victim's run.

| Stage | When it acts | Resources |
|---|---|---|
| `pre_training` | removes samples, then the victim trains on the rest | `train_set` |
| `in_training` | changes the victim's training through hooks | `train_set`, `training_loop` |
| `post_training` | repairs the trained model | `model`, optional `clean_subset` |
| `inference` | rejects suspicious test inputs | `model`, optional `clean_subset` |
| `model_detection` | decides whether a model is backdoored | `model`, optional `clean_subset` |

## Which cells exist

The scenario kind decides which resources a defender can have:

| Scenario kind | `train_set` | `training_loop` | `model` | `clean_subset` |
|---|---|---|---|---|
| `data_poisoning` | yes | yes | yes | yes |
| `training_control` | no | no | yes | yes |
| `supply_chain` | no | no | yes | yes |

A cell exists when the defense's resources are all available and its `clean_fraction` fits the
scenario's budget. Otherwise it is N/A, and the leaderboard shows why. A dataset defense cannot
face a training-control attack, for example, because the defender never sees that training
set. `uv run sbb plan <spec>` lists every cell with its reason.

## Rules

- **Kerckhoffs.** Defenses are public, hyperparameters included. An attack may adapt to one
  only by declaring `aware:<defense>` (v2).
- **Frozen hyperparameters.** Hyperparameters are fixed per dataset and model in the card. The
  defender cannot know the trigger, so tuning per cell is not allowed.
- **Attack validity.** An attack that costs more than 3 points of clean accuracy is reported as
  `invalid_attack` and left out of rankings.
- **Defense validity.** A defense counts as successful only if it costs at most 3 points of
  clean accuracy.

## Metrics

All metrics are fractions of the test set, shown as percentages. The triggered test set holds
every test sample with the trigger. Every metric but `asr_all` counts only the triggered samples
whose true label is not the target.

| Metric | Meaning | Cells |
|---|---|---|
| `ca` | clean test samples accepted and correctly classified | all |
| `asr` | triggered samples accepted and classified as the target | all |
| `asr_all` | ASR over every triggered sample, the target class included, as the attack papers report it | all |
| `ra` | triggered samples accepted and classified as their true label | all |
| `frr` | clean test samples the input defense rejects | inference |
| `tpr` | triggered samples the input defense rejects | inference |
| `auroc` | how well the input defense's scores separate clean from triggered samples | inference |
| `verdict` | the detection score of the model | model_detection |

A tied top score counts as no prediction: wrong for `ca` and `ra`, never a success for `asr`.
The voting readouts of the full-track networks tie often, and a silent network would otherwise
predict class 0. Replications keep the papers' `argmax`, which gives a tie to the first tied
class. Without an input defense, every sample is accepted. The leaderboard turns the verdicts into a
model AUROC: verdicts on clean models (`no_attack` cells) against verdicts on valid backdoored
models. `no_attack` cells give the clean reference and the natural ASR.

## What the views enforce, and what they do not

The views stop accidental violations. Reading an undeclared resource raises
`ThreatModelViolation`, and a train split poisoned in frames refuses raw-event access, because
re-encoding the clean events would reveal the poisoned samples. The views cannot stop deliberate
cheating, such as an input detector that flags anything not in the public test set. Code review
is the control for that.
