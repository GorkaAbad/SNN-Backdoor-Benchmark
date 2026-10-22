"""Defense-specific logic. The registered defenses also meet tests/contracts."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sbb.core.proxies import DefenderView, ScenarioInfo, TrainSetView
from sbb.core.trainer import Trainer, TrainSpec
from sbb.data.module import DataModule
from sbb.defenses.clp import CLP, channel_lipschitz
from sbb.defenses.fine_pruning import FinePruning
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
