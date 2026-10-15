"""Label policies of poisoned samples."""

from sbb.core.plugins import LabelPolicy


class CleanLabel(LabelPolicy):
    """Keeps the true label."""

    name = "clean"

    def relabel(self, label: int, target: int) -> int:
        """Return label."""
        return label


class DirtyLabel(LabelPolicy):
    """Changes the label to the target."""

    name = "dirty"

    def relabel(self, label: int, target: int) -> int:
        """Return target."""
        return target
