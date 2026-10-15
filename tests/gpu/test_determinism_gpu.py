"""Is training on GPU deterministic? Run on a CUDA machine with `uv run pytest -m gpu -v`."""

from pathlib import Path

import pytest
import torch

from tests.conftest import make_cell, run_in_fresh_caches


@pytest.mark.gpu
def test_two_gpu_runs_give_identical_weights(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    (first, first_weights), (second, second_weights) = run_in_fresh_caches(
        make_cell("static_patch", "fine_tuning"), tmp_path, torch.device("cuda")
    )
    assert first.metrics == second.metrics
    assert first_weights == second_weights
