"""Neuron and architecture specs."""

from typing import Any, Literal

from pydantic import BaseModel, field_validator


class NeuronSpec(BaseModel, frozen=True):
    """Spiking neuron configuration. v_reset None means soft reset."""

    model: Literal["lif", "plif", "if"] = "lif"
    tau: float = 2.0
    v_threshold: float = 1.0
    v_reset: float | None = 0.0
    surrogate: Literal["atan", "sigmoid"] = "atan"
    detach_reset: bool = True


class ArchSpec(BaseModel, frozen=True):
    """Architecture configuration. name is the key in ARCHITECTURES.

    Only the torch backend is supported: the cupy backend of SpikingJelly 0.0.0.0.14 uses np.int,
    which NumPy 1.24 removed (see docs/measurements.md).
    """

    name: str
    T: int
    neuron: NeuronSpec = NeuronSpec()
    backend: Literal["torch", "cupy"] = "torch"
    kwargs: dict[str, Any] = {}

    @field_validator("backend")
    @classmethod
    def _torch_only(cls, backend: str) -> str:
        if backend != "torch":
            raise ValueError("only the torch backend works with the pinned SpikingJelly and NumPy")
        return backend
