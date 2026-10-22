## What this adds

<!-- One or two sentences. For a published method, the paper and the official code. -->

## Checklist

Delete the sections that do not apply.

**A new attack or defense** ([docs/contributing.md](../docs/contributing.md))

- [ ] One plugin file, one card, and the import line in the package `__init__`. No change to
      the core.
- [ ] Hyperparameters live in the card, fixed per dataset and model, not tuned per cell.
- [ ] The plugin uses only its view and the shared `Trainer`. No raw datasets, models or
      loaders.
- [ ] `uv run sbb test --plugin <name>` passes.
- [ ] A defense designed for ANNs explains its SNN adaptation in `adaptation_note`.

**A published method** ([docs/porting.md](../docs/porting.md))

- [ ] The card names `paper`, `code_reference` and `license_note`. No code was copied from a
      repository without a license.
- [ ] An attack has a replication record in `replications/`, with its result or the reason it
      misses.

**Results**

- [ ] Produced with `sbb run` and not edited. `SBB_SUBMITTER` and `SBB_SUBMITTER_ORG` set.
- [ ] `uv run sbb bundle --plugin <name>` wrote the manifest, included here.
- [ ] `uv run sbb validate results replications` passes.

**A verification**

- [ ] Produced with `sbb reproduce` on my own site, and not edited.

**A harness change**

- [ ] Tests first. Golden tests unchanged, or the change is discussed in this pull request.
- [ ] An interface change of `docs/PLAN.md` section 6 is proposed there.
- [ ] `ruff`, `mypy src`, `lint-imports` and `pytest` pass locally.
