"""Sneaky Spikes, dynamic trigger (port of the official dynamic.py, MIT)."""

import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.func import functional_call

from sbb.attacks.dynamic_trigger import DynamicTrigger, perturb
from sbb.core.cards import AttackCard
from sbb.core.plugins import ModelAttack
from sbb.core.proxies import AttackerView
from sbb.core.registry import ATTACKS
from sbb.core.trainer import Loss, TrainerHooks, TrainState, per_sample_loss
from sbb.models.snn import SNNModel, build_model


class _Triggered(TrainerHooks):
    """The classifier's loss: alpha on the clean batch, 1 - alpha on its triggered copy."""

    def __init__(
        self, frozen: nn.Module, gamma: float, alpha: float, target: int, loss: Loss
    ) -> None:
        self.frozen = frozen
        self.gamma = gamma
        self.alpha = alpha
        self.target = target
        self.kind = loss

    def joint(
        self, model: nn.Module, clean: torch.Tensor, triggered: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Return alpha * loss(clean, y) + (1 - alpha) * loss(model(triggered), target)."""
        target = torch.full_like(y, self.target)
        backdoor = per_sample_loss(self.kind, model(triggered), target).mean()
        clean_loss = per_sample_loss(self.kind, clean, y).mean()
        joint: torch.Tensor = self.alpha * clean_loss + (1 - self.alpha) * backdoor
        return joint

    def loss(self, state: TrainState, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """The loss with the frozen generator of the previous epoch."""
        with torch.no_grad():
            triggered = perturb(self.frozen, state.inputs, self.gamma)
        return self.joint(state.model, logits, triggered, targets)


class _JointTraining(_Triggered):
    """Before the classifier's step, updates the generator through a one-step lookahead.

    As in the official train(): the generator's loss is the backdoor loss of the classifier after
    one step of plain gradient descent on the joint loss, so the generator learns a trigger the
    classifier is about to pick up.
    """

    def __init__(self, generator: nn.Module, optimizer: torch.optim.Optimizer, *args: Any):
        super().__init__(*args)
        self.generator, self.optimizer = generator, optimizer

    def loss(self, state: TrainState, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Update the generator, then return the classifier's loss."""
        x, model = state.inputs, state.model
        triggered = perturb(self.generator, x, self.gamma)
        joint = self.joint(model, model(x), triggered, targets)
        names, params = zip(*model.named_parameters(), strict=True)
        grads = torch.autograd.grad(joint, params, create_graph=True)
        lr = state.optimizer.param_groups[0]["lr"]
        ahead = {n: p - lr * g for n, p, g in zip(names, params, grads, strict=True)}
        buffers = {n: b.clone() for n, b in model.named_buffers()}  # the real statistics stay
        output = functional_call(
            model, {**ahead, **buffers}, (perturb(self.generator, x, self.gamma),)
        )
        self.optimizer.zero_grad()
        target = torch.full_like(targets, self.target)
        per_sample_loss(self.kind, output, target).mean().backward()  # type: ignore[no-untyped-call]
        self.optimizer.step()
        return super().loss(state, logits, targets)


@ATTACKS.register("sneaky_dynamic")
class SneakyDynamic(ModelAttack):
    """Trains the model jointly with a spiking autoencoder whose output is the trigger.

    Every epoch has a fresh classifier optimizer, and the frozen copy of the generator that the
    classifier sees is refreshed at the end, as in the official code. Then the classifier trains
    one more epoch with the learned trigger, the official evaluation's step. The official code
    keeps the epoch with the best test accuracies, which an attacker cannot see, so the last epoch
    is kept here.
    """

    card = AttackCard.load(Path(__file__).parent / "cards" / "sneaky_dynamic.yaml")
    trigger: DynamicTrigger

    def __init__(self, hparams: dict[str, Any], target: int) -> None:
        super().__init__(hparams, target)
        self.trigger = DynamicTrigger(hparams["gamma"], hparams["channels"])

    def make_model(self, view: AttackerView) -> SNNModel:
        """Train the scenario architecture and the generator together."""
        device, trainer = view.trainer.device, view.trainer
        model = build_model(view.arch, view.scenario.input_shape, view.scenario.num_classes)
        generator = copy.deepcopy(self.trigger.generator).to(device).train()
        frozen = copy.deepcopy(generator).eval()
        optimizer = torch.optim.Adam(generator.parameters(), lr=self.hparams["generator_lr"])
        common = (self.hparams["gamma"], self.hparams["alpha"], self.target, trainer.spec.loss)
        joint = _JointTraining(generator, optimizer, frozen, *common)
        loader = view.train_set.loader(self.hparams["batch_size"], shuffle=True)
        for _ in range(trainer.spec.epochs):
            trainer.fit(model, loader, hooks=[joint], epochs=1)
            frozen.load_state_dict(generator.state_dict())
        trainer.fit(model, loader, hooks=[_Triggered(frozen, *common)], epochs=1)
        self.trigger.generator.load_state_dict(frozen.state_dict())
        return model
