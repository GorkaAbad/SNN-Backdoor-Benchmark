"""Attack plugins. Every plugin module is imported here so it registers."""

from sbb.attacks import (  # noqa: F401
    finetune_injection,
    flashy,
    no_attack,
    sneaky_moving,
    sneaky_smart,
    sneaky_static,
    static_patch,
)
