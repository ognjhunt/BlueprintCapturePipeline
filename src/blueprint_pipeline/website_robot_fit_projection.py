"""The job's robot fit table as the site reads it (preparation status v5): a spec-bounded estimate.

Read-only. It projects the fit table the scene handoff wrote: the full route's
``robot_fit_table`` (``website_robot_fit``), or a geometry-only job's
``fit_check`` (``website_scene_handoff.geometry_only_fit_check``). The table must
be bound to the preparation the handoff names, and that preparation to the
current task context. The table is informational, so the projection never fails
a status read: anything it cannot verify is left out (None).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .website_preparation_outcome import _load

SCHEMA_VERSION = "website_robot_fit_projection.v1"
#: The WebApp accepts a status read of at most 8192 bytes; the projection keeps to half.
MAX_BYTES = 4096
#: Rows in this order, so a projection cut to MAX_BYTES keeps the robots that fit.
VERDICTS = ("fits", "fits_spec", "doesnt_fit", "unknown")
#: Why a geometry-only job's check did not run (``geometry_only_fit_check``, the handoff).
NOT_RUN_REASONS = frozenset({"preparation_blocked", "task_strategy_not_modelled", "subject_not_excised",
                             "task_family_not_modelled"})


def _projection(route: str, **fields: Any) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "route": route, "status": "not_run", "reason": None,
            "evidence_label": None, "counts": None, "models": [], "models_truncated": False,
            "table_digest": None, **fields}


def read_robot_fit(*, capture_root: Path, task_context_digest: str) -> dict[str, Any] | None:
    """The current job's robot fit table for the site, or None when there is none to show."""
    root = capture_root.absolute()
    path = root / "pipeline" / "website_scene_preparation" / "handoff.json"
    try:
        if not path.exists():
            return None
        handoff = _load(path, root, "digest")
        if (handoff.get("schema_version") != "website_scene_handoff.v1"
                or handoff.get("capture_id") != root.name or handoff.get("scene_id") != root.parent.parent.name):
            return None
        route = handoff.get("atlas_route", "full")
        if route == "geometry_only":
            fit = handoff.get("fit_check")
            if not isinstance(fit, Mapping):
                return None
            if fit.get("status") == "not_run":
                reason = fit.get("reason")
                return _projection(route, reason=reason) if reason in NOT_RUN_REASONS else None
            reference = fit.get("robot_fit_table")
        elif route == "full":
            reference = handoff.get("robot_fit_table")
        else:
            return None
        if not isinstance(reference, Mapping) or type(reference.get("path")) is not str:
            return None
        preparation = _load(Path(handoff["preparation_path"]), root, "digest")
        table = _load(Path(reference["path"]), root, "table_digest")
        if ((preparation.get("binding") or {}).get("task_context_digest") != task_context_digest
                or handoff.get("preparation_digest") != preparation["digest"]
                or table.get("schema_version") != "website_robot_fit_table.v1"
                or table.get("preparation_digest") != preparation["digest"]
                or table["table_digest"] != reference.get("digest")
                or table.get("evidence_label") != "spec_bounded_estimate"
                or table.get("physical_world_truth_claimed") is not False
                or any(row.get("verdict") not in VERDICTS for row in table["rows"])):
            return None
        counts = {verdict: table["counts"][verdict] for verdict in VERDICTS}
        if any(type(count) is not int or count < 0 for count in counts.values()):
            return None
        rows = sorted(table["rows"], key=lambda row: VERDICTS.index(row["verdict"]))
        value = _projection(route, status="completed", evidence_label="spec_bounded_estimate", counts=counts,
            models=[{"name": str(row["name"]), "manufacturer": str(row["manufacturer"]), "verdict": row["verdict"],
                     # What the spec sheet and the scene could not settle, or ruled out.
                     "open_checks": [str(check["check"]) for check in row["checks"]
                                     if check["result"] in {"fail", "unknown"}]} for row in rows],
            table_digest=table["table_digest"])
        while len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()) > MAX_BYTES:
            value["models"].pop()
            value["models_truncated"] = True
        return value
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
