"""NAD, neural attention distillation (Li et al., ICLR 2021)."""

import copy
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn.functional as F

from sbb.core.cards import DefenseCard
from sbb.core.plugins import ModelDefense
from sbb.core.proxies import DefenderView
from sbb.core.registry import DEFENSES
from sbb.core.trainer import TrainerHooks, TrainState
from sbb.defenses.utils import conv_blocks
from sbb.models.snn import Probe, SNNModel


@DEFENSES.register("nad")
class NAD(ModelDefense):
    """Fine-tunes a teacher on clean data, then distills its attention into the model."""

    card = DefenseCard.load(Path(__file__).parent / "cards" / "nad.yaml")

    def repair(self, view: DefenderView) -> SNNModel:
        """Fine-tune a copy as the teacher, then train the model with attention distillation."""
        batch_size = self.hparams["batch_size"]
        teacher = copy.deepcopy(view.model)
        view.trainer.fit(
            teacher, view.clean_loader(batch_size), epochs=self.hparams["teacher_epochs"]
        )
        student = view.model
        hooks = AttentionDistillation(teacher, student, self.hparams["betas"], self.hparams["p"])
        try:
            loader = view.clean_loader(batch_size)
            return view.trainer.fit(student, loader, hooks=[hooks], epochs=self.hparams["epochs"])
        finally:
            hooks.close()


def attention_map(rates: torch.Tensor, p: float, eps: float = 1e-6) -> torch.Tensor:
    """The A2sum attention of a feature map [B, C, H, W]: sum over channels of |x|^p, unit norm."""
    am: torch.Tensor = rates.abs().pow(p).sum(1, keepdim=True)
    normalized: torch.Tensor = am / (am.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + eps)
    return normalized


class AttentionDistillation(TrainerHooks):
    """Adds sum_l beta_l MSE(A(student_l), A(teacher_l)) to the student's loss.

    The feature maps are the spike rates of the neurons of the last len(betas) conv blocks, the
    last beta going with the last block. The student's spikes keep their graph, so the term trains
    it through the surrogate gradients. Its probes are reopened after every batch, so their
    records never pile up. Call close when training ends.
    """

    def __init__(self, teacher: SNNModel, student: SNNModel, betas: list[float], p: float) -> None:
        layers = [b.neuron for b in conv_blocks(student) if b.neuron is not None]
        teacher_layers = [b.neuron for b in conv_blocks(teacher) if b.neuron is not None]
        depth = min(len(betas), len(layers))
        self.betas = betas[len(betas) - depth :]
        self.layers = layers[len(layers) - depth :]
        self.teacher_layers = teacher_layers[len(teacher_layers) - depth :]
        self.teacher = teacher.eval()
        self.p = p
        self._stack = ExitStack()
        self._open()

    def _open(self) -> None:
        self.probes = [
            self._stack.enter_context(Probe(layer, "spikes", detach=False)) for layer in self.layers
        ]

    def loss(self, state: TrainState, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """The task loss plus the attention distillation term."""
        assert state.per_sample_loss is not None
        with ExitStack() as stack, torch.no_grad():
            teacher_probes = [stack.enter_context(Probe(n, "spikes")) for n in self.teacher_layers]
            self.teacher(state.inputs)
        loss = state.per_sample_loss.mean()
        for beta, student, teacher in zip(self.betas, self.probes, teacher_probes, strict=True):
            student_map = attention_map(student.rates(), self.p)
            teacher_map = attention_map(teacher.rates(), self.p)
            loss = loss + beta * F.mse_loss(student_map, teacher_map)
        return loss

    def on_batch_end(self, state: TrainState) -> None:
        """Reopen the student's probes for the next batch."""
        self.close()
        self._open()

    def close(self) -> None:
        """Remove the student's probes."""
        self._stack.close()
