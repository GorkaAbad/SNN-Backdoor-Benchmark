"""Exceptions raised by the harness."""


class SbbError(Exception):
    """Base class of every sbb error."""


class ThreatModelViolation(SbbError):
    """A plugin accessed a resource that its card does not declare."""


class BudgetViolation(SbbError):
    """An attack exceeded its budget or broke its declared label policy."""


class ReproducibilityError(SbbError):
    """An artifact does not match its recorded hash."""
