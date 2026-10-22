"""ABL, anti-backdoor learning (Li et al., NeurIPS 2021)."""

from pathlib import Path

import numpy as np
import torch

from sbb.core.cards import DefenseCard
from sbb.core.plugins import TrainingDefense
from sbb.core.proxies import DefenderView
from sbb.core.registry import DEFENSES
from sbb.core.trainer import Trainer, TrainerHooks, TrainState, per_sample_loss
from sbb.models.snn import SNNModel


@DEFENSES.register("abl")
class ABL(TrainingDefense):
    """Isolates the samples learned fastest, trains without them, then unlearns them."""

    card = DefenseCard.load(Path(__file__).parent / "cards" / "abl.yaml")

    def hooks(self, view: DefenderView) -> "ABLHooks":
        """The hooks that run ABL inside the victim's training."""
        return ABLHooks(view, **self.hparams)


class ABLHooks(TrainerHooks):
    """The three stages of ABL in one training run of the victim.

    The first isolation_epochs use local gradient ascent: the loss of a batch is sign(L - gamma) L
    of its mean loss L, which holds it near gamma, so the samples learned much faster, the
    poisoned ones, keep a lower loss. Then the isolation_ratio share of the train set with the
    lowest loss is isolated and the remaining epochs train without it. After the last epoch,
    unlearning_epochs of gradient ascent on the isolated samples unlearn the backdoor, through the
    shared Trainer with SGD (momentum 0.9, as in the official code).
    """

    def __init__(
        self,
        view: DefenderView,
        gamma: float,
        isolation_ratio: float,
        isolation_epochs: int,
        unlearning_epochs: int,
        unlearning_lr: float,
        unlearning_weight_decay: float,
    ) -> None:
        self.view = view
        self.gamma = gamma
        self.isolation_ratio = isolation_ratio
        self.last_epoch = view.trainer.spec.epochs - 1
        self.isolation_epochs = min(isolation_epochs, view.trainer.spec.epochs)
        self.unlearning_epochs = unlearning_epochs
        self.unlearning_lr = unlearning_lr
        self.unlearning_weight_decay = unlearning_weight_decay
        self.isolated = np.empty(0, dtype=np.int64)  # positions in view.train_set
        self._isolated_indices = torch.empty(0, dtype=torch.long)  # their dataset indices

    def loss(self, state: TrainState, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Local gradient ascent first, then the mean loss of the samples not isolated."""
        assert state.per_sample_loss is not None
        if state.epoch < self.isolation_epochs:
            mean = state.per_sample_loss.mean()
            return torch.sign(mean - self.gamma) * mean
        kept = ~torch.isin(state.batch_indices.cpu(), self._isolated_indices)
        kept = kept.to(state.per_sample_loss.device)
        return (state.per_sample_loss * kept).sum() / kept.sum().clamp(min=1)

    def on_epoch_end(self, state: TrainState) -> None:
        """Isolate after the ascent epochs, unlearn after the last epoch."""
        if state.epoch == self.isolation_epochs - 1:
            self._isolate(state.model)
        if state.epoch == self.last_epoch:
            self._unlearn(state.model)

    @torch.no_grad()
    def _isolate(self, model: SNNModel) -> None:
        """Isolate the isolation_ratio share of the train set with the lowest loss."""
        kind, device = self.view.trainer.spec.loss, self.view.trainer.device
        losses, indices = [], []
        model.eval()
        for x, y, index in self.view.train_set.loader(self.view.trainer.spec.batch_size):
            losses.append(per_sample_loss(kind, model(x.to(device)), y.to(device)).cpu())
            indices.append(index)
        model.train()
        order = torch.cat(losses).argsort(stable=True)[
            : int(len(self.view.train_set) * self.isolation_ratio)
        ]
        self.isolated = order.numpy()
        self._isolated_indices = torch.cat(indices)[order]

    def _unlearn(self, model: SNNModel) -> None:
        """Gradient ascent on the isolated samples, with the official SGD settings."""
        if not len(self.isolated):
            return
        spec = self.view.trainer.spec.model_copy(
            update={
                "epochs": self.unlearning_epochs,
                "optimizer": "sgd",
                "lr": self.unlearning_lr,
                "weight_decay": self.unlearning_weight_decay,
                "schedule": "none",
            }
        )
        loader = self.view.train_set.loader(spec.batch_size, indices=self.isolated, shuffle=True)
        Trainer(spec, self.view.trainer.device).fit(model, loader, hooks=[_Ascent()])


class _Ascent(TrainerHooks):
    """Gradient ascent: the negated mean loss of the batch."""

    def loss(self, state: TrainState, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Minus the mean loss."""
        assert state.per_sample_loss is not None
        return -state.per_sample_loss.mean()
