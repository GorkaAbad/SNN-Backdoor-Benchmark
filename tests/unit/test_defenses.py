"""Defense-specific logic. The registered defenses also meet tests/contracts."""

import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sbb.core.proxies import DefenderView, ScenarioInfo, TrainSetView
from sbb.core.trainer import Trainer, TrainSpec, TrainState, per_sample_loss
from sbb.data.module import DataModule
from sbb.defenses.abl import ABL
from sbb.defenses.activation_clustering import small_clusters
from sbb.defenses.anp import ANP, neuron_masks, perturb_bn
from sbb.defenses.clp import CLP, channel_lipschitz
from sbb.defenses.fine_pruning import FinePruning
from sbb.defenses.nad import NAD, AttentionDistillation, attention_map
from sbb.defenses.ndsbm import NDSBM, Clamps
from sbb.defenses.spectral_signatures import spectral_outliers
from sbb.defenses.strip import STRIP
from sbb.defenses.tmpbd import class_potentials, p_value
from sbb.defenses.utils import channel_means, conv_blocks
from sbb.models.snn import SNNModel, build_model
from sbb.models.spec import ArchSpec

INFO = ScenarioInfo(
    dataset="synthetic", sensor_size=(16, 16, 2), num_classes=4, T=4, input_shape=(2, 16, 16)
)
ARCH = ArchSpec(name="tiny_lif_cnn", T=4)
CPU = torch.device("cpu")


def _model() -> SNNModel:
    torch.manual_seed(0)
    model = build_model(ARCH, (2, 16, 16), 4)
    with torch.no_grad():  # batch norm statistics as after training, not the identity
        for block in conv_blocks(model):
            block.bn.running_var.uniform_(0.5, 2.0)
            block.bn.weight.uniform_(0.5, 1.5)
            block.bn.bias.uniform_(-0.2, 0.2)
    return model


def _view(cls: type, datamodule: DataModule, model: SNNModel) -> DefenderView:
    trainer = Trainer(TrainSpec(epochs=1, batch_size=8), CPU)
    train_set = TrainSetView(datamodule, None)
    return DefenderView(cls.card, INFO, 0, trainer, model, train_set, datamodule, ARCH)


def test_channel_lipschitz_matches_the_official_loop() -> None:
    block = conv_blocks(_model())[1]
    conv, bn = block.conv, block.bn
    std = bn.running_var.sqrt()
    official = [
        torch.linalg.svdvals(
            conv.weight[c].reshape(conv.weight.shape[1], -1) * (bn.weight[c] / std[c]).abs()
        ).max()
        for c in range(conv.out_channels)
    ]
    assert torch.allclose(channel_lipschitz(conv, bn), torch.stack(official))


def test_clp_prunes_the_outlier_channel(datamodule: DataModule) -> None:
    model = _model()
    with torch.no_grad():
        conv_blocks(model)[0].conv.weight[3] *= 50
    repaired = CLP({"u": 3.0}).repair(_view(CLP, datamodule, model))
    bn = conv_blocks(repaired)[0].bn
    pruned = torch.nonzero((bn.weight == 0) & (bn.bias == 0)).flatten().tolist()
    assert pruned == [3]


@pytest.mark.parametrize(("max_drop", "pruned"), [(1.0, 32), (-1.0, 0)])
def test_fine_pruning_prunes_the_dormant_channels_first(
    datamodule: DataModule, max_drop: float, pruned: int
) -> None:
    model = _model()
    view = _view(FinePruning, datamodule, model)
    block = conv_blocks(view.model)[-1]
    weight = block.bn.weight.detach().clone()
    clean = view.clean_loader(32)
    order = torch.argsort(channel_means(view.model, block.neuron, clean, CPU), stable=True)
    defense = FinePruning({"max_drop": max_drop, "epochs": 1, "batch_size": 32})
    repaired = defense.repair(view)
    bn = conv_blocks(repaired)[-1].bn
    zero = (bn.weight == 0) & (bn.bias == 0)
    # The pruned channels stay zero through fine-tuning, and they are the most dormant ones.
    assert zero.sum() == pruned and zero[order[:pruned]].all()
    if pruned == 0:  # the first channel already crossed the limit and was given back
        assert not zero.any() and not torch.equal(bn.weight, weight)  # fine-tuned, not pruned


def test_strip_rejects_the_chosen_fraction_of_clean_inputs(datamodule: DataModule) -> None:
    view = _view(STRIP, datamodule, _model())
    defense = STRIP({"overlays": 4, "frr": 0.25, "batch_size": 8})
    defense.fit(view)
    clean = torch.cat([x for x, _, _ in view.clean_loader(8)])
    with torch.no_grad():
        scores = defense.score(clean)
        overlay = defense.overlays[0]
        p = torch.softmax(defense.model(clean + overlay), dim=1)
    assert scores.shape == (len(clean),)
    assert np.mean((scores > defense.threshold).numpy()) <= 0.25
    first = -(p * torch.log(p)).sum(1)  # the entropy under one overlay enters the mean
    assert (first >= 0).all() and torch.isfinite(scores).all()


def test_tmpbd_p_value_flags_an_outlier_margin() -> None:
    rng = np.random.default_rng(0)
    clean = rng.normal(1.0, 0.1, 10)
    assert p_value(clean) > 0.05
    assert p_value(np.append(clean, 30.0)) < 0.05
    assert p_value(np.ones(11)) == 1.0  # nothing to fit, no evidence


def test_tmpbd_potentials_average_the_output_neurons() -> None:
    arch = ArchSpec(name="dvsgesture_net", T=4, kwargs={"channels": 8})
    model = build_model(arch, (2, 32, 32), 11).eval()  # no dropout, as in the verdict
    x = torch.rand(2, 4, 2, 32, 32) * 3
    name = list(model.named_neurons())[-1][0]
    with model.probe(name, "membrane") as probe:
        model(x)
    expected = F.avg_pool1d(probe.membrane().mean(1), 10, 10)  # 10 neurons vote per class
    assert torch.allclose(class_potentials(model, x).detach(), expected)
    tiny = _model()
    assert torch.equal(class_potentials(tiny, x[..., :16, :16]), tiny(x[..., :16, :16]))


def test_spectral_outliers_are_the_shifted_samples_of_their_class() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(210, 5))
    features[100:110, 0] += 8.0  # a planted cluster in class 0, as poisoned samples would be
    labels = np.repeat([0, 1], [110, 100])
    removed = spectral_outliers(features, labels, percentile=85)
    assert set(range(100, 110)) <= set(removed.tolist())
    # About 15% of each class, the paper's 1.5 epsilon. np.percentile interpolates, so 17 of the
    # 110 scores of class 0 lie above its 85th percentile, as in the official code.
    assert np.bincount(labels[removed]).tolist() == [17, 15]


def test_small_clusters_remove_a_planted_minority_only() -> None:
    rng = np.random.default_rng(0)
    clean = rng.normal(size=(90, 20))
    planted = rng.normal(size=(30, 20)) + 6.0  # 25% of class 0, as poisoned samples would be
    halves = np.concatenate([rng.normal(size=(50, 20)) - 3.0, rng.normal(size=(50, 20)) + 3.0])
    features = np.concatenate([clean, planted, halves])
    labels = np.repeat([0, 1], [120, 100])
    # Class 1 splits in two equal clusters, above max_share, so it keeps every sample.
    removed = small_clusters(features, labels, dims=10, max_share=0.35, seed=0)
    assert removed.tolist() == list(range(90, 120))


def _losses(model: SNNModel, view: DefenderView, positions: np.ndarray) -> torch.Tensor:
    with torch.no_grad():
        model.eval()
        x, y, _ = next(iter(view.train_set.loader(len(positions), indices=positions)))
        return per_sample_loss("cross_entropy", model(x), y)


def test_abl_ascends_near_gamma_isolates_the_easiest_and_unlearns_them(
    datamodule: DataModule,
) -> None:
    model = _model()
    view = _view(ABL, datamodule, model)
    hparams = {**ABL.card.hparams_for("synthetic", "tiny_lif_cnn"), "isolation_ratio": 0.1}
    hooks = ABL(hparams).hooks(view)
    state = TrainState(0, 0, model, None, torch.tensor([0, 1]), None, torch.empty(0))  # type: ignore[arg-type]
    for losses, expected in [([0.1, 0.3], -0.2), ([0.6, 0.8], 0.7)]:  # gamma 0.5
        state.per_sample_loss = torch.tensor(losses)
        assert hooks.loss(state, torch.empty(0), torch.empty(0)).item() == pytest.approx(expected)

    view.trainer.fit(model, view.train_set.loader(8, shuffle=True), epochs=2)
    hooks._isolate(model)  # what on_epoch_end does after the last ascent epoch
    everything = np.arange(len(view.train_set))
    lowest = _losses(model, view, everything).argsort(stable=True)[: len(everything) // 10]
    assert hooks.isolated.tolist() == lowest.tolist()

    _, _, indices = next(iter(view.train_set.loader(len(everything))))
    state.epoch, state.batch_indices = 1, indices[[int(hooks.isolated[0]), 0, 1]]
    state.per_sample_loss = torch.tensor([9.0, 1.0, 2.0])  # the isolated sample is left out
    assert hooks.loss(state, torch.empty(0), torch.empty(0)).item() == pytest.approx(1.5)

    before = _losses(model, view, hooks.isolated).mean()
    hooks._unlearn(model)
    assert _losses(model, view, hooks.isolated).mean() > before


def test_attention_map_sums_channels_and_has_unit_norm() -> None:
    rates = torch.rand(3, 8, 4, 4)
    am = attention_map(rates, p=2.0)
    assert am.shape == (3, 1, 4, 4)
    torch.testing.assert_close(am * am.flatten(1).norm(dim=1).view(3, 1, 1, 1), am)  # unit norm
    expected = (rates**2).sum(1, keepdim=True)
    torch.testing.assert_close(am, expected / expected.flatten(1).norm(dim=1).view(3, 1, 1, 1))


def test_nad_adds_attention_distillation_and_leaves_no_probe(datamodule: DataModule) -> None:
    student = _model()
    x, y, index = next(iter(datamodule.test_loader(4)))
    for teacher, more in [(copy.deepcopy(student), False), (_model().train(), True)]:
        if more:
            for p in teacher.parameters():
                p.data.add_(torch.randn_like(p) * 0.5)
        hooks = AttentionDistillation(teacher, student, betas=[500.0, 1000.0, 1000.0], p=2.0)
        logits = student.eval()(x)  # the teacher runs in eval mode, as in the official code
        state = TrainState(
            0, 0, student, None, index, per_sample_loss("cross_entropy", logits, y), x
        )  # type: ignore[arg-type]
        loss = hooks.loss(state, logits, y)
        hooks.close()
        assert (loss.item() > state.per_sample_loss.mean().item()) is more
        loss.backward()  # type: ignore[no-untyped-call]  # the distillation term reaches the student
    for module in student.modules():
        assert not module._forward_hooks
    view = _view(NAD, datamodule, _model())
    repaired = NAD(NAD.card.hparams_for("synthetic", "tiny_lif_cnn")).repair(view)
    assert all(not m._forward_hooks for m in repaired.modules())


def test_perturb_bn_matches_a_batch_norm_with_scaled_parameters() -> None:
    bn = conv_blocks(_model())[0].bn.eval()
    x = torch.randn(2, 3, 16, 8, 8)  # [T, B, C, H, W], as in multi-step mode
    weight_coeff, bias_coeff = torch.rand(16), torch.rand(16) + 0.5
    with torch.no_grad():
        expected = F.batch_norm(
            x.flatten(0, 1),
            bn.running_mean,
            bn.running_var,
            bn.weight * weight_coeff,
            bn.bias * bias_coeff,
            False,
            0.0,
            bn.eps,
        ).view_as(x)
        torch.testing.assert_close(perturb_bn(bn(x), bn, weight_coeff, bias_coeff), expected)


def test_anp_masks_lie_in_the_unit_interval_and_prune_only_the_scale(
    datamodule: DataModule,
) -> None:
    model = _model()
    view = _view(ANP, datamodule, model)
    masks = neuron_masks(
        copy.deepcopy(model),
        view.clean_loader(8),
        view.trainer,
        eps=0.4,
        alpha=0.2,
        lr=0.2,
        iterations=5,
    )
    assert [m.shape[0] for m in masks] == [b.bn.num_features for b in conv_blocks(model)]
    assert all(((m >= 0) & (m <= 1)).all() for m in masks)
    hparams = {
        **ANP.card.hparams_for("synthetic", "tiny_lif_cnn"),
        "iterations": 5,
        "threshold": 1.01,
    }
    repaired = ANP(hparams).repair(view)  # a threshold above 1 prunes every channel
    for block, original in zip(conv_blocks(repaired), conv_blocks(model), strict=True):
        assert (block.bn.weight == 0).all()
        torch.testing.assert_close(block.bn.bias, original.bn.bias)  # the shift stays


def test_clamps_bound_the_input_current_per_time_step_and_channel() -> None:
    model = _model().eval()
    clamps = Clamps(model, init=10.0)
    assert [p.shape for p in clamps.upper] == [(4, 16), (4, 32)]  # [T, C] per conv block
    x = torch.rand(2, 4, 2, 16, 16)
    with torch.no_grad():
        free = model(x)
        handles = clamps.attach(model)
        assert torch.equal(model(x), free)  # 1 +- 10 does not bind on these inputs
        for upper, lower in zip(clamps.upper, clamps.lower, strict=True):
            upper.fill_(0.0)
            lower.fill_(0.0)
        currents: list[torch.Tensor] = []
        hook = conv_blocks(model)[0].bn.register_forward_hook(lambda m, i, o: currents.append(o))
        model(x)
        hook.remove()
        for handle in handles:
            handle.remove()
    assert (currents[0] == 0).all()  # the clamp replaced the batch norm's output


def test_ndsbm_narrows_the_clamps_and_keeps_the_weights(datamodule: DataModule) -> None:
    model = _model()
    view = _view(NDSBM, datamodule, model)
    defense = NDSBM({**NDSBM.card.hparams_for("synthetic", "tiny_lif_cnn"), "epochs": 2})
    repaired = defense.repair(view)
    before, after = dict(model.named_parameters()), dict(repaired.named_parameters())
    for name in before:  # the batch-norm statistics follow the clean data, as in the official code
        torch.testing.assert_close(before[name], after[name])
    clamps = defense.extra(repaired)
    assert all((upper < 11.0).all() for upper in clamps.upper)  # the norm term pulls them in
    assert all(not m._forward_hooks for m in repaired.modules())  # the Game attaches them
