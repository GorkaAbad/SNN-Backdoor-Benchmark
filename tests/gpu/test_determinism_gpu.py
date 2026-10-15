"""Is training on GPU deterministic? Run on a CUDA machine with:

    uv run --extra cuda pytest -m gpu -v

and record the outcome in docs/measurements.md.
"""

from pathlib import Path

import pytest
import torch

from tests.conftest import SCENARIO, make_cell, run_in_fresh_caches


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["torch", "cupy"])
def test_two_gpu_runs_give_identical_weights(backend: str, tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if backend == "cupy":
        pytest.importorskip("cupy")
    arch = SCENARIO.arch.model_copy(update={"backend": backend})
    cell = make_cell(
        "static_patch", "fine_tuning", scenario=SCENARIO.model_copy(update={"arch": arch})
    )
    (first, first_weights), (second, second_weights) = run_in_fresh_caches(
        cell, tmp_path, torch.device("cuda")
    )
    assert first.metrics == second.metrics
    assert first_weights == second_weights
