"""Defense plugins. Every plugin module is imported here so it registers."""

from sbb.defenses import (  # noqa: F401
    abl,
    activation_clustering,
    anp,
    clp,
    fewer_steps,
    fine_pruning,
    fine_tuning,
    nad,
    ndsbm,
    no_defense,
    spectral_signatures,
    strip,
    strip_per_step,
    threshold_recalibration,
    tmpbd,
)
