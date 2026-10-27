"""The single training loop (docs/PLAN.md section 6.7)."""

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch.utils.data import DataLoader

from sbb.models.snn import SNNModel
from sbb.utils.errors import SbbError

log = logging.getLogger(__name__)
Loader = DataLoader[Any]


Loss = Literal["cross_entropy", "mse"]
# How predict counts a tied top score. The voting readouts of the full-track networks give
# multiples of 1 / (10 T), so silent or tied outputs are common. "none" makes a tie no
# prediction (NO_PREDICTION), the benchmark rule. "first" is argmax, the papers' rule.
Ties = Literal["none", "first"]
NO_PREDICTION = -1
# Below the smallest gap between two vote rates, so the same votes summed in another order tie.
_TIE_ATOL = 1e-6


def per_sample_loss(kind: Loss, output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the loss of each sample [B] for the model output [B, K] and class targets [B].

    "mse" compares the output with one-hot targets, averaged over classes. Its batch mean is the
    papers' F.mse_loss(firing_rate, one_hot).
    """
    if kind == "mse":
        one_hot = F.one_hot(targets, output.shape[1]).to(output.dtype)
        return ((output - one_hot) ** 2).mean(1)
    return F.cross_entropy(output, targets, reduction="none")


class TrainSpec(BaseModel, frozen=True):
    """Training recipe of a scenario. SGD uses momentum 0.9."""

    epochs: int
    batch_size: int
    loss: Loss = "cross_entropy"
    optimizer: Literal["adam", "sgd"] = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    schedule: Literal["none", "cosine"] = "cosine"
    augment: Literal["none"] = "none"
    amp: bool = False


@dataclass
class TrainState:
    """Mutable state passed to hooks."""

    epoch: int
    step: int
    model: SNNModel
    optimizer: torch.optim.Optimizer
    batch_indices: torch.Tensor
    per_sample_loss: torch.Tensor | None
    inputs: torch.Tensor  # the batch x [B, T, P, H, W] on the training device


class TrainerHooks:
    """No-op defaults. Override only what you need. At most one hook set may override the loss."""

    def on_epoch_start(self, state: TrainState) -> None:
        """Called before each epoch."""

    def loss(
        self, state: TrainState, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor | None:
        """Return a loss to replace the mean of state.per_sample_loss, or None."""
        return None

    def on_batch_end(self, state: TrainState) -> None:
        """Called after each optimizer step."""

    def on_epoch_end(self, state: TrainState) -> None:
        """Called after each epoch."""


class Trainer:
    """Trains and evaluates SNNModels with a scenario recipe.

    Args:
        spec: The training recipe.
        device: Device to train on.
    """

    def __init__(self, spec: TrainSpec, device: torch.device) -> None:
        if spec.amp:
            raise NotImplementedError("amp is not supported yet")
        self.spec = spec
        self.device = device

    def fit(
        self,
        model: SNNModel,
        loader: Loader,
        hooks: Sequence[TrainerHooks] = (),
        epochs: int | None = None,
    ) -> SNNModel:
        """Train model in place and return it.

        Args:
            model: The model to train.
            loader: Batches of (x, y, index).
            hooks: Hook sets called during training.
            epochs: Overrides the recipe's number of epochs.
        """
        epochs = self.spec.epochs if epochs is None else epochs
        model.to(self.device).train()
        optimizer = self._optimizer(model)
        schedule = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
            if self.spec.schedule == "cosine"
            else None
        )
        state = TrainState(0, 0, model, optimizer, torch.empty(0), None, torch.empty(0))
        for epoch in range(epochs):
            state.epoch = epoch
            for hook in hooks:
                hook.on_epoch_start(state)
            total = 0.0
            for x, y, index in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)
                state.batch_indices, state.inputs = index, x
                state.per_sample_loss = per_sample_loss(self.spec.loss, logits, y)
                loss = self._loss(hooks, state, logits, y)
                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                state.step += 1
                total += loss.item()
                for hook in hooks:
                    hook.on_batch_end(state)
            if schedule:
                schedule.step()
            log.info("epoch %d/%d, mean loss %.4f", epoch + 1, epochs, total / max(len(loader), 1))
            for hook in hooks:
                hook.on_epoch_end(state)
        return model

    @torch.no_grad()
    def predict(
        self, model: SNNModel, loader: Iterable[Any], ties: Ties = "none"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (predicted classes, labels) over the batches (x, y, index) of loader.

        With ties "none", an input whose top two scores tie gets NO_PREDICTION.
        """
        model.to(self.device).eval()
        predictions, labels = [torch.empty(0, dtype=torch.long)], [torch.empty(0, dtype=torch.long)]
        for x, y, _ in loader:
            out = model(x.to(self.device))
            predicted = out.argmax(1)
            if ties == "none":
                top = out.topk(2, dim=1).values
                predicted[top[:, 0] - top[:, 1] <= _TIE_ATOL] = NO_PREDICTION
            predictions.append(predicted.cpu())
            labels.append(y)
        return torch.cat(predictions).numpy(), torch.cat(labels).numpy()

    def evaluate(self, model: SNNModel, loader: Loader, ties: Ties = "none") -> dict[str, float]:
        """Return the accuracy over the loader. A tie is wrong unless ties is "first"."""
        predictions, labels = self.predict(model, loader, ties)
        return {"accuracy": float(np.mean(predictions == labels)) if len(labels) else 0.0}

    def _optimizer(self, model: SNNModel) -> torch.optim.Optimizer:
        params, lr, wd = model.parameters(), self.spec.lr, self.spec.weight_decay
        if self.spec.optimizer == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)

    @staticmethod
    def _loss(
        hooks: Sequence[TrainerHooks], state: TrainState, logits: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        overrides = [loss for hook in hooks if (loss := hook.loss(state, logits, y)) is not None]
        if len(overrides) > 1:
            raise SbbError("more than one hook set overrides the loss")
        assert state.per_sample_loss is not None
        return overrides[0] if overrides else state.per_sample_loss.mean()
