"""Host-side JSON Schema resources for the cross-runtime task success contracts.

The rigid and articulated contracts share an envelope but not their criteria,
so a boundary that carries either one embeds the union rather than assuming
the rigid lane.
"""

from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from .rigid_task_success_contract_schema import rigid_task_success_contract_schema

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "schemas"
    / "articulated_task_success_contract.v1.schema.json"
)


@lru_cache(maxsize=1)
def _loaded_schema() -> dict[str, Any]:
    import jsonschema

    value = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(value)
    return value


def articulated_task_success_contract_schema() -> dict[str, Any]:
    """Return an independent copy suitable for embedding under ``$defs``."""

    return deepcopy(_loaded_schema())


def task_success_contract_schema() -> dict[str, Any]:
    """Either admitted contract, discriminated by its own schema_version."""

    return {
        "oneOf": [
            rigid_task_success_contract_schema(),
            articulated_task_success_contract_schema(),
        ]
    }


__all__ = [
    "SCHEMA_PATH",
    "articulated_task_success_contract_schema",
    "task_success_contract_schema",
]
