"""Activation clustering (Chen et al., arXiv 1811.03728)."""

from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA

from sbb.core.cards import DefenseCard
from sbb.core.plugins import DataDefense
from sbb.core.proxies import DefenderView
from sbb.core.registry import DEFENSES
from sbb.defenses.utils import reference_features


@DEFENSES.register("activation_clustering")
class ActivationClustering(DataDefense):
    """Removes, in every class, a small cluster of the reduced representations."""

    card = DefenseCard.load(Path(__file__).parent / "cards" / "activation_clustering.yaml")

    def sanitize(self, view: DefenderView) -> np.ndarray:
        """Train a reference model on the train set as received, then drop small clusters."""
        labels = view.train_set.labels()
        seed = int(view.rng.integers(2**31))
        dims, share = self.hparams["dims"], self.hparams["max_share"]
        removed = small_clusters(reference_features(view), labels, dims, share, seed)
        return np.setdiff1d(np.arange(len(labels)), removed)


def small_clusters(
    features: np.ndarray, labels: np.ndarray, dims: int, max_share: float, seed: int
) -> np.ndarray:
    """Positions in the smaller of two k-means clusters of their class, sorted.

    Each class's features are reduced to dims independent components (FastICA, with the
    max_iter and tol of ART), then split by 2-means. The smaller cluster counts as poisoned when
    it holds at most max_share of the class, the paper's relative size analysis. The features are
    whitened here rather than by FastICA (see _whitened), with no more directions than they vary
    in. A class whose features do not vary keeps every sample.
    """
    removed = [np.empty(0, dtype=np.int64)]
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        white = _whitened(features[members].astype(np.float64), dims)
        if white.shape[1] == 0:
            continue
        ica = FastICA(whiten=False, max_iter=1000, tol=0.005, random_state=seed)
        reduced = ica.fit_transform(white)
        clusters = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(reduced)
        sizes = np.bincount(clusters, minlength=2)
        small = int(np.argmin(sizes))
        if sizes[small] <= max_share * len(members):
            removed.append(members[clusters == small])
    return np.sort(np.concatenate(removed))


def _whitened(features: np.ndarray, dims: int) -> np.ndarray:
    """The centered features on their top principal directions, at most dims, with unit variance.

    This is FastICA's own whitening with another sign convention. FastICA signs each direction by
    its loading on the first feature, which is exactly zero when that feature does not vary (a
    neuron that never fires for the class), and a zero sign turned the run into NaN on the
    cluster's LAPACK. Here a direction's largest loading is positive, and directions without
    variance (as numpy's matrix_rank counts them) are left out.
    """
    centered = features - features.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(centered.T @ centered)
    tol = np.sqrt(max(eigvals.max(), 0.0)) * max(centered.shape) * np.finfo(np.float64).eps
    order = np.argsort(eigvals)[::-1]
    order = order[eigvals[order] > tol**2][:dims]
    directions = eigvecs[:, order]
    largest = np.abs(directions).argmax(axis=0)
    directions *= np.sign(directions[largest, np.arange(len(order))])
    whitened: np.ndarray = centered @ directions / np.sqrt(eigvals[order] / len(features))
    return whitened
