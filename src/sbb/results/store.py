"""Result storage as one JSON file per cell."""

from abc import ABC, abstractmethod
from pathlib import Path

from sbb.core.result import CellResult


class ResultStore(ABC):
    """Holds CellResults by key."""

    @abstractmethod
    def save(self, result: CellResult) -> None:
        """Store result, replacing any result with the same key."""

    @abstractmethod
    def get(self, key: str) -> CellResult | None:
        """Return the result of a cell key, or None."""

    @abstractmethod
    def query(self, **filters: str) -> list[CellResult]:
        """Return results whose cell fields equal the filters, e.g. attack="static_patch"."""


class LocalResultStore(ResultStore):
    """Results as <root>/<benchmark_version>/<key>.json."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def save(self, result: CellResult) -> None:
        """Write the result as JSON."""
        path = self.root / result.cell.benchmark_version / f"{result.key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(indent=2) + "\n")

    def get(self, key: str) -> CellResult | None:
        """Find the result in any benchmark version."""
        path = next(self.root.glob(f"*/{key}.json"), None)
        return CellResult.model_validate_json(path.read_text()) if path else None

    def query(self, **filters: str) -> list[CellResult]:
        """Scan all results and keep the matching ones."""
        results = [
            CellResult.model_validate_json(p.read_text()) for p in self.root.glob("*/*.json")
        ]
        return [r for r in results if all(str(getattr(r.cell, k)) == v for k, v in filters.items())]
