---
license: mit
tags:
  - spiking-neural-networks
  - backdoor-attacks
  - neuromorphic
  - benchmark
  - safetensors
---

# sbb-zoo/checkpoints

The trained models of the [SNN Backdoor Benchmark](https://github.com/GorkaAbad/SNN-Backdoor-Benchmark),
a benchmark for backdoor attacks and defenses on spiking neural networks with neuromorphic data.
The [leaderboard](https://gorkaabad.github.io/SNN-Backdoor-Benchmark/) shows the results they produced.

> **Most of these models are backdoored on purpose.** They exist to reproduce and study the
> benchmark. Do not deploy them.

## Layout

Each artifact is a folder `<benchmark version>/<key>/`. The key is the hash of everything that
determines the artifact, and each result in the benchmark repository names its artifacts by key:

```json
"artifacts": {"clean_model": "8098...", "model": "3174...", "defended_model": "c19d..."}
```

A model folder holds `model.safetensors` and `checkpoint.json` (the architecture, input shape,
number of classes and sha256 of the weights). Some also hold `trigger.safetensors` (a learned trigger) or `defense.safetensors` (state a
defense adds to the model). Detection defenses store their verdict as JSON. There is no pickle file.

## Use

```bash
git clone https://github.com/GorkaAbad/SNN-Backdoor-Benchmark && cd SNN-Backdoor-Benchmark
uv sync
uv run sbb zoo pull 0.9.0 --revision <commit>   # copies the models into the local cache
uv run sbb reproduce results/0.9.0/<key>.json   # reruns a cell on the published models
```

Pin a commit with `--revision` to reproduce a result exactly.

## License

The code and these weights are released under the MIT License. The models were trained on
N-MNIST, DVS128 Gesture and CIFAR10-DVS, which keep their own licenses. Each attack and defense
card in the benchmark repository cites its paper.
