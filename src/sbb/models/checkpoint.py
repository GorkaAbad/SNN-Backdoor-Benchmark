"""Safetensors checkpoints. Loading rebuilds the model from its spec and never unpickles."""

from pathlib import Path

import torch
from pydantic import BaseModel
from safetensors.torch import load_file, save_file

from sbb.models.snn import SNNModel, build_model
from sbb.models.spec import ArchSpec
from sbb.utils.errors import ReproducibilityError
from sbb.utils.hashing import file_sha256

_META = "checkpoint.json"


class Checkpoint(BaseModel):
    """Metadata stored next to the weights."""

    arch: ArchSpec
    input_shape: tuple[int, int, int]
    num_classes: int
    weights_file: str
    sha256: str
    model_key: str


def save_checkpoint(model: SNNModel, directory: Path, model_key: str) -> Checkpoint:
    """Write model.safetensors and checkpoint.json into directory."""
    weights = directory / "model.safetensors"
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, weights)
    checkpoint = Checkpoint(
        arch=model.spec,
        input_shape=model.input_shape,
        num_classes=model.num_classes,
        weights_file=weights.name,
        sha256=file_sha256(weights),
        model_key=model_key,
    )
    (directory / _META).write_text(checkpoint.model_dump_json(indent=2))
    return checkpoint


def load_checkpoint(directory: Path, device: torch.device | str = "cpu") -> SNNModel:
    """Rebuild the model from checkpoint.json and load the verified weights."""
    checkpoint = Checkpoint.model_validate_json((directory / _META).read_text())
    weights = directory / checkpoint.weights_file
    if file_sha256(weights) != checkpoint.sha256:
        raise ReproducibilityError(f"{weights} does not match its sha256")
    model = build_model(checkpoint.arch, checkpoint.input_shape, checkpoint.num_classes)
    model.load_state_dict(load_file(weights, device=str(device)))
    return model.to(device)
