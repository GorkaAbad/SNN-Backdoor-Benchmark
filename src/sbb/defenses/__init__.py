"""Defense plugins. Every plugin module is imported here so it registers."""

from sbb.defenses import (  # noqa: F401
    activation_clustering,
    clp,
    fine_pruning,
    fine_tuning,
    no_defense,
    spectral_signatures,
    strip,
    tmpbd,
)
