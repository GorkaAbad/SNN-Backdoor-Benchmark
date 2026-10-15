"""Seeding helpers. Only the Game calls seed_everything."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch, and make torch deterministic."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def rng_for(seed: int, index: int) -> np.random.Generator:
    """Return the RNG of one sample. The same (seed, index) always gives the same stream."""
    return np.random.default_rng([seed, index])
