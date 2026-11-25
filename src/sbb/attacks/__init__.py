"""Attack plugins. Every plugin module is imported here so it registers."""

from sbb.attacks import (  # noqa: F401
    adap_blend,
    adap_patch,
    finetune_injection,
    flashy,
    no_attack,
    sneaky_dynamic,
    sneaky_moving,
    sneaky_smart,
    sneaky_static,
    static_patch,
    tact,
    tbackdoor_jitter,
    tbackdoor_latency,
    tbackdoor_rate,
)
