"""Canonical JSON and hashing (docs/PLAN.md section 7.6)."""

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _default(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    raise TypeError(f"cannot hash {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    """Serialize obj with sorted keys and no whitespace. Floats are written with repr."""
    return json.dumps(
        obj,
        default=_default,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_hex(obj: Any) -> str:
    """Return the sha256 hex digest of the canonical JSON of obj."""
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the sha256 hex digest of a file."""
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
