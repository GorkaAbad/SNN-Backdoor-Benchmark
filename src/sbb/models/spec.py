"""Neuron and architecture specs."""

from typing import Any, Literal

from pydantic import BaseModel


class NeuronSpec(BaseModel, frozen=True):
    """Spiking neuron configuration. v_reset None means soft reset."""

    model: Literal["lif", "plif", "if"] = "lif"
    tau: float = 2.0
    v_threshold: float = 1.0
    v_reset: float | None = 0.0
    surrogate: Literal["atan", "sigmoid"] = "atan"
    detach_reset: bool = True


class ArchSpec(BaseModel, frozen=True):
    """Architecture configuration. name is the key in ARCHITECTURES."""

    name: str
    T: int
    neuron: NeuronSpec = NeuronSpec()
    backend: Literal["torch", "cupy"] = "torch"
    kwargs: dict[str, Any] = {}
