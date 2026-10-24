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
    it holds at most max_share of the class, the paper's relative size analysis. A class gets no
    more components than the rank of its centered features: FastICA divides by the spread of each
    component, which is zero beyond the rank. A class whose features do not vary keeps every
    sample.
    """
    removed = [np.empty(0, dtype=np.int64)]
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        class_features = features[members].astype(np.float64)
        rank = np.linalg.matrix_rank(class_features - class_features.mean(axis=0))
        components = min(dims, int(rank))
        if components == 0:
            continue
        ica = FastICA(n_components=components, max_iter=1000, tol=0.005, random_state=seed)
        reduced = ica.fit_transform(class_features)
        clusters = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(reduced)
        sizes = np.bincount(clusters, minlength=2)
        small = int(np.argmin(sizes))
        if sizes[small] <= max_share * len(members):
            removed.append(members[clusters == small])
    return np.sort(np.concatenate(removed))
