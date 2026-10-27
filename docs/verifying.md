# Verifying results

A full-track result is `self-reported` until someone from another organization reruns it and
gets the same numbers. This page is for that person. A verifier reruns 3 cells of a submission,
picked at random by the tool, and opens a pull request with the records the tool writes.

## Statuses

`VerificationPolicy` (`sbb.results.policy`) computes every status from the results and the
records under `verifications/`. Nobody sets one by hand.

| Records of a result | Status |
|---|---|
| none that counts | `self-reported` |
| at least one independent match | `reproduced` |
| as many independent mismatches as matches, or more | `disputed` |

A record is independent when its `verifier_org` differs from the result's `submitter_org`. On
the lite track, the `verify-lite` workflow on GitHub runners is the verifier. A record counts
only while the result still has the numbers it checked, so a rerun under the same key needs a
new check. A cell over its seeds is `disputed` if one seed is, and a full-track cell is
`reproduced` when a spot check reproduced one of its seeds.

The policy judges each record from its stored differences, never from the stored verdict. On the
full track a metric may differ by `max(full_pp, full_sigma x std)`, the std taken over the
cell's valid seeds, with `full_pp` and `full_sigma` from the benchmark spec (2.0 and 2.0 in
`benchmarks/v0.9.yaml`). The lite track allows `lite_pp` (3.0).

## What you need

- One NVIDIA GPU, Docker with the NVIDIA container toolkit (or Apptainer, or Python 3.11 with
  uv), and room for the datasets and models.
- A GitHub account.
- An organization other than the submitter's.

On a GH200, a DVS128 Gesture cell takes about 12 minutes and a CIFAR10-DVS cell about 30 when
the published models are reused. A cell that retrains its model takes hours.

## Set up

Use exactly the release that produced the results. The provenance of each result names its
`sbb_version`, and the artifact keys of the model zoo include the package `major.minor`, so a
newer release finds none of the published models and retrains them.

```bash
git clone https://github.com/GorkaAbad/SNN-Backdoor-Benchmark.git
cd SNN-Backdoor-Benchmark
git checkout <tag>
docker pull ghcr.io/gorkaabad/sbb:<tag>     # or: uv sync --frozen
```

Write a site profile outside the repository ([sites.md](sites.md)). Its `name` is published with
every record, so pick a neutral one.

```yaml
name: uni-example-a100
runtime: none
scheduler: none
device: cuda
cache_dir: /cache
max_parallel: 1
```

Set who you are. Both values go into every record.

```bash
export SBB_SUBMITTER=<your GitHub handle>
export SBB_SUBMITTER_ORG=<your organization>
```

With Docker, run each `sbb` command below inside the image:

```bash
docker run --rm --gpus all -v "$PWD:/work" -v ~/sbb-cache:/cache \
  -e SBB_SUBMITTER -e SBB_SUBMITTER_ORG ghcr.io/gorkaabad/sbb:<tag> sbb <command>
```

Datasets download into `<cache_dir>/datasets` the first time a cell needs them. When a mirror
fails, the error says which files to fetch by hand and where to put them.

## Spot-check a submission

1. Pull the published models, so defense cells rerun on the exact models of the submission:

    ```bash
    sbb zoo pull <benchmark version> --site site.yaml
    ```

2. Let the tool pick the cells. The choice is seeded by the bundle id, so anyone gets the same
   picks:

    ```bash
    sbb spot-check results/manifests/<bundle_id>.json
    ```

3. Run each printed `sbb reproduce` command with `--site site.yaml`. Each writes
   `verifications/<version>/<key>/<handle>-<time>.json` and prints `match` or `mismatch` with
   the difference in every metric. The tolerance comes from the spec of the result's version.

Any published cell can be checked the same way with `sbb reproduce results/<version>/<key>.json`,
without a manifest.

## Submit the records

Fork the repository, add only the new files under `verifications/`, and open a pull request with
the cells and your GPU in its description. `sbb validate verifications` runs the check CI runs.
Submit a mismatch too: it is a valid outcome, and the maintainers look into it in the open.
