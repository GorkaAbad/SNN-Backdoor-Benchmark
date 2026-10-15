"""Plugin registries of the core. Datasets and architectures have their own in their layers."""

from sbb.core.plugins import Attack, Defense, Metric
from sbb.utils.registry import Registry

ATTACKS: Registry[type[Attack]] = Registry("attack")
DEFENSES: Registry[type[Defense]] = Registry("defense")
METRICS: Registry[type[Metric]] = Registry("metric")
