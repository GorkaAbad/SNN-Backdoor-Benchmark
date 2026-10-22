# Porting a published method

Port, do not wrap. An official repository brings its own training loop, data pipeline and SNN
adaptation, which is why numbers from different papers are not comparable today. A port
re-implements the method on the shared parts of sbb, so every method trains, loads data and
evaluates the same way. Read [contributing.md](contributing.md) for how a plugin is built.

## Steps

1. **Choose what to replicate.** For an attack, pick the main results of the paper: its clean
   accuracy and attack success rate tables or figures. Write them into a replication record
   (below) with the paper's setting.
2. **Read the paper and the official code, and check the license.** Code under a permissive
   license may be adapted, with the license named in the card's `license_note`. Code without a
   license can only be re-implemented from the paper, never copied.
3. **Implement on the shared parts.** Reuse the triggers, selectors, label policies, the
   `Trainer` with `TrainerHooks`, and `sbb.defenses.utils`. Write new code only for what is new
   in the method, and add a reusable part to the shared modules.
4. **Replicate** with three seeds: `uv run sbb replicate <method> --site <profile>`.
5. **Explain a miss.** If a result misses the tolerance, find out why and write the reason in
   the record. An attack enters the benchmark only with status `replicated` or `explained`.
6. **Ask the authors** to review the port and, for a defense, its adaptation note.

The card names the source: `paper`, `code_reference` (the repository and the functions you
ported) and `license_note`. Comments in the card say where the port differs from the official
code and why.

## Defenses

A defense has one sbb implementation: the SNN adaptation of its original paper. Every cell uses
it. Attack papers each adapt defenses in their own way, so the defense numbers they report are
not replicated, and a defense has no replication record. A variant can come later as a separate
method or version.

A defense designed for ANNs describes its adaptation in the card's `adaptation_note`, and
implements it with `sbb.defenses.utils`. Fine-pruning, for example, ranks the channels of the
last conv block by their spike rate on clean data, the SNN counterpart of the activations of the
original method:

```yaml
adaptation_note: >
  Neurons of the last conv block are ranked by their spike rate on clean data
  (channel_means on its neuron layer), the SNN counterpart of the activations of the original
  method. Pruning silences a channel by zeroing the scale and shift of its batch norm
  (zero_channels).
```

## The replication record

`replications/<method>.yaml` holds the paper's numbers and the setting that produced them.
Each target is a full scenario, which may differ from the benchmark scenarios: the paper's
dataset split, representation, network, training recipe and poison rate.

```yaml
method: sneaky_static
source: Abad et al., Sneaky Spikes, NDSS 2024, Table I, static trigger, baseline columns
confirmed: false                 # a maintainer confirmed the targets
hparams:                         # merged over the card's hparams for the replication cells
  selector: {include_target: true}
seeds: [0, 1, 2]
tolerance_pp: 2.0
targets:
  - setting:
      name: sneaky_spikes_cifar10dvs
      data: {name: cifar10dvs, split: cifar10dvs_9to1_v1}
      repr: {kind: frames, T: 16, split_by: number}
      arch: {name: cifar10dvs_net, T: 16}
      train: {epochs: 28, batch_size: 16, loss: mse, optimizer: adam, lr: 0.001, schedule: cosine}
      budget: {poison_rate: 0.1, target: 0, defender_clean_fraction: 0.0, delta_pp: 100.0}
    reported: {ca: 67.7, asr_all: 100.0}    # percent, as in the paper
status: pending
explanation: null
```

Points that matter:

- **Compare like with like.** Most attack papers report ASR over every triggered test sample,
  the target class included. That is `asr_all`. The benchmark's `asr` leaves the target class
  out.
- **No defender pool and a large `delta_pp`.** The papers have no defender pool, and the
  validity check must not hide the ASR of a setting that costs accuracy.
- **The paper's own hyperparameters** go into the record's `hparams`. The card keeps the
  benchmark's. Papers often poison every class, for example, while benchmark cells keep the
  target class out.
- **A missing network.** If the paper used a network that is not in the registry, register it
  under its own name in `sbb.models.architectures`. It is then available for the replication
  and, if useful, for the benchmark.
- **Numbers read from a plot** are marked as such in `source`, with a wider tolerance.

## Running it

```bash
uv run sbb replicate <method> --site <profile>            # filled record in results/replications/
uv run sbb replicate <method> --site <profile> --update   # also written back to replications/
```

`sbb replicate` runs one undefended cell per target and seed, on the track `replication`, and
fills `ours` with the mean and the population standard deviation over the seeds. The status
becomes `replicated` when every reported metric of every target is within `tolerance_pp` of our
mean, and `failed` otherwise. A maintainer can mark a failed record `explained`, with the
reason in `explanation`, for example a paper number that the official code does not reach
either. The leaderboard's Replications view shows the paper against ours for every record.
