"""Scene-envelope-independent digest for authored native task phases."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest


def native_task_construction_authored_contract_digest(
    phase_plan: Mapping[str, Any],
) -> str:
    value = json.loads(json.dumps(dict(phase_plan), allow_nan=False))
    if not isinstance(value.get("phases"), list) or not value["phases"]:
        raise ValueError("native_task_construction_authored_contract_invalid")
    value.pop("scene_plan_digest", None)
    value.pop("plan_digest", None)
    return canonical_digest(value)


__all__ = ["native_task_construction_authored_contract_digest"]
