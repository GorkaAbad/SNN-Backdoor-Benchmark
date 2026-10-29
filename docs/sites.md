# Sites: where cells run

A site profile is a YAML file that describes one compute environment. Keep your real profiles
outside the repository, because they name accounts, partitions and paths. The examples in
`sites/examples/` are templates.

```yaml
name: my-workstation          # recorded in the provenance of every result
runtime: none                 # none, docker or apptainer
scheduler: none               # none or slurm
image: null                   # the container image, pinned by digest, when runtime is not none
device: cuda                  # cuda or cpu
cache_dir: ~/.cache/sbb       # datasets under datasets/, models and frame caches under artifacts/
max_parallel: 1               # cells at the same time
slurm: {}                     # #SBATCH options, see below
```

Every command that runs cells takes `--site <profile>`. Without it, sbb uses `~/.cache/sbb` and
CUDA when available.

## On one machine

With `scheduler: none`, `sbb run` and `sbb replicate` run the pending cells:

- `max_parallel: 1` runs them in the command's own process.
- `max_parallel: N` runs N `sbb run-cell` processes at a time, with one log per cell under
  `<cache_dir>/jobs/<id>/logs/`.
  On CUDA, each worker owns one visible GPU, round-robin. Cluster GPUs often run in
  exclusive-process mode, where a GPU takes one process only, so there set `max_parallel` to
  the number of GPUs of the job.

```bash
uv run sbb run benchmarks/smoke.yaml --site sites/examples/local-cpu.yaml --out results
```

## On a Slurm cluster

With `scheduler: slurm`, `sbb run` submits the pending cells as one job array and returns. Run the
same command again when the jobs end: it finds the results and writes the table.

```yaml
name: my-cluster
runtime: apptainer
scheduler: slurm
image: docker://ghcr.io/gorkaabad/sbb@sha256:<digest>
device: cuda
cache_dir: /scratch/<user>/sbb-cache
max_parallel: 16              # array tasks running at the same time
slurm:
  account: <account>
  partition: <gpu partition>
  time: "06:00:00"
  gres: "gpu:1"
  extra: "#SBATCH --qos=normal"   # copied into the script as is, one line per line
```

Each `slurm` entry becomes an `#SBATCH --<key>=<value>` line. The job and its cells are written to
`<cache_dir>/jobs/<id>/`, with one log per task. With `runtime: apptainer`, each task runs
`apptainer exec --nv --bind <cache_dir> <image> sbb run-cell ...`. With `runtime: none`, the tasks
use the environment `sbatch` exports, so `sbb` must be on the `PATH` of the shell that submits.

Some clusters do not allow Python on login nodes. There, submit a job that runs `sbb run` with a
profile that has `scheduler: none`: the cells then run inside that job.

## The container image

Every release tag `v*` publishes `ghcr.io/gorkaabad/sbb:<tag>` for `linux/amd64` and
`linux/arm64`, and its digest is in the release notes. It holds the environment of `uv.lock`, sbb
and every plugin of the release. Pin images by digest, so a result records exactly what ran.

Docker:

```bash
docker run --rm --gpus all -v "$PWD:/work" -v ~/.cache/sbb:/cache \
  ghcr.io/gorkaabad/sbb@sha256:<digest> \
  sbb run benchmarks/v1.0.yaml --site site.yaml --out results
```

with `cache_dir: /cache` in `site.yaml`.

Apptainer, which runs Docker images directly:

```bash
apptainer exec --nv --bind /scratch/<user>/sbb-cache \
  docker://ghcr.io/gorkaabad/sbb@sha256:<digest> \
  sbb run-cell cell.json --site site.yaml --out results
```

## The model zoo

The models behind the published results live on the Hugging Face Hub, in
[sbb-zoo/checkpoints](https://huggingface.co/sbb-zoo/checkpoints), as
`<version>/<key>/` folders named by the keys in each result's `artifacts`.
`sbb zoo pull VERSION --revision <commit>` copies the models of a benchmark version into
`cache_dir`, so a verifier reruns defense cells on exactly those models.
`sbb zoo push VERSION --repo <your repo>` uploads yours, with a Hugging Face token that can write to
that repository (`HF_TOKEN`, `HF_TOKEN_PATH` or `hf auth login`).
