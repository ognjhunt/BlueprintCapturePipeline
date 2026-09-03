"""Host-side JSON Schema resource for the cross-runtime rigid success contract."""

from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any


SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "schemas"
    / "rigid_task_success_contract.v1.schema.json"
)


@lru_cache(maxsize=1)
def _loaded_schema() -> dict[str, Any]:
    import jsonschema

    value = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(value)
    return value


def rigid_task_success_contract_schema() -> dict[str, Any]:
    """Return an independent copy suitable for embedding under ``$defs``."""

    return deepcopy(_loaded_schema())


__all__ = ["SCHEMA_PATH", "rigid_task_success_contract_schema"]
