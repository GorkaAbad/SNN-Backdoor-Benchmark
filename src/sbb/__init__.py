"""Benchmark for backdoor attacks and defenses on spiking neural networks."""

from importlib.metadata import version

__version__ = version("sbb")
SBB_MINOR = ".".join(__version__.split(".")[:2])
