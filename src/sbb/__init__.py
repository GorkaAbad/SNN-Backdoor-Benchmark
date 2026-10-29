"""Benchmark for backdoor attacks and defenses on spiking neural networks."""

from importlib.metadata import version

__version__ = version("sbb")
# Hashed into every artifact key, under its old name sbb_minor. Up to 1.0 it was the package
# major.minor, so each minor release invalidated every cached model. Since 2.0 it moves only when
# a change alters what an artifact holds without a plugin version bump (the Trainer, the data
# pipeline, the networks). A golden metric that changes is the signal (PLAN 7.6).
ARTIFACT_EPOCH = "1.0"
