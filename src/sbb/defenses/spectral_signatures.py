"""Spectral signatures (Tran, Li, Madry, NeurIPS 2018)."""

from pathlib import Path

import numpy as np

from sbb.core.cards import DefenseCard
from sbb.core.plugins import DataDefense
from sbb.core.proxies import DefenderView
from sbb.core.registry import DEFENSES
from sbb.defenses.utils import reference_features


@DEFENSES.register("spectral_signatures")
class SpectralSignatures(DataDefense):
    """Removes, in every class, the samples that stand out along the top singular direction."""

    card = DefenseCard.load(Path(__file__).parent / "cards" / "spectral_signatures.yaml")

    def sanitize(self, view: DefenderView) -> np.ndarray:
        """Train a reference model on the train set as received, then drop each class's outliers."""
        labels = view.train_set.labels()
        removed = spectral_outliers(reference_features(view), labels, self.hparams["percentile"])
        return np.setdiff1d(np.arange(len(labels)), removed)


def spectral_outliers(features: np.ndarray, labels: np.ndarray, percentile: float) -> np.ndarray:
    """Positions whose score is above the given percentile of their class, sorted.

    The score of a sample is |<v, R(x)>|, with v the top right singular vector of its class's
    centered representations, as in the official compute_corr.
    """
    removed = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        reps = features[members].astype(np.float64)
        top = np.linalg.svd(reps - reps.mean(0), full_matrices=False)[2][0]
        scores = np.abs(reps @ top)
        removed.append(members[scores > np.percentile(scores, percentile)])
    return np.sort(np.concatenate(removed))
