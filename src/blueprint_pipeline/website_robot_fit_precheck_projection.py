"""The job's early robot fit as the site reads it (preparation status v6): a pre-spend estimate.

Read-only. It projects the precheck the scene handoff wrote (``website_robot_fit_precheck``). It lives
apart from the precheck, which reads the screening catalog, the GPU floor and the team registry, so the
status reader and the outcome reader stay off the screening launch and GPU provider import graph
(``test_live_pipeline_import_isolation``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "website_robot_fit_precheck.v1"
PROJECTION_SCHEMA = "website_robot_fit_precheck_projection.v1"
STATUSES = ("fit", "no_fit", "unknown")
VERDICTS = ("fit", "no_fit", "unknown", "unavailable")
#: The site's view stays small: the WebApp reads at most 8192 bytes, and v5's robot fit table takes 4096.
MAX_PROJECTION_BYTES = 1536


def read_robot_fit_precheck(*, capture_root: Path, task_context_digest: str) -> dict[str, Any] | None:
    """The current job's early robot fit for the site (preparation status v6), or None when there is none.

    Read-only, like ``website_robot_fit_projection``: the precheck must be bound to the preparation the
    handoff names, and that preparation to the current task context; anything that does not bind is left
    out, never a failed status read.
    """
    from .website_preparation_outcome import _load

    root = capture_root.absolute()
    path = root / "pipeline" / "website_scene_preparation" / "handoff.json"
    try:
        if not path.exists():
            return None
        handoff = _load(path, root, "digest")
        reference = handoff.get("robot_fit_precheck")
        if (handoff.get("schema_version") != "website_scene_handoff.v1" or handoff.get("capture_id") != root.name
                or handoff.get("scene_id") != root.parent.parent.name or not isinstance(reference, Mapping)
                or type(reference.get("path")) is not str):
            return None
        preparation = _load(Path(handoff["preparation_path"]), root, "digest")
        precheck = _load(Path(reference["path"]), root, "precheck_digest")
        if ((preparation.get("binding") or {}).get("task_context_digest") != task_context_digest
                or handoff.get("preparation_digest") != preparation["digest"]
                or precheck.get("schema_version") != SCHEMA
                or precheck.get("preparation_digest") != preparation["digest"]
                or precheck["precheck_digest"] != reference.get("digest")
                or precheck.get("status") not in STATUSES
                or precheck.get("physical_world_truth_claimed") is not False
                or (precheck["status"] == "no_fit") != isinstance(precheck.get("hold_reason"), str)):
            return None
        counts = {name: precheck["counts"][name] for name in VERDICTS}
        if any(type(count) is not int or count < 0 for count in counts.values()):
            return None
        rows = []
        for key, row in sorted(precheck["embodiments"].items()):
            candidates = [item for item in precheck["candidates"] if item.get("embodiment") == key]
            rows.append({"embodiment": str(row["embodiment"]), "verdict": str(row["verdict"]),
                         "reasons": [str(code) for code in row["reasons"]][:4],
                         "fit_candidates": sum(item["verdict"] == "fit" for item in candidates)})
        value = {"schema_version": PROJECTION_SCHEMA, "status": precheck["status"],
                 "hold_reason": precheck["hold_reason"], "task_kind": precheck.get("task_kind"), "counts": counts,
                 "embodiments": rows, "embodiments_truncated": False, "evidence_label": "pre_spend_estimate",
                 "precheck_digest": precheck["precheck_digest"]}
        while len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()) > MAX_PROJECTION_BYTES:
            value["embodiments"].pop()
            value["embodiments_truncated"] = True
        return value
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


__all__ = ["MAX_PROJECTION_BYTES", "PROJECTION_SCHEMA", "SCHEMA", "STATUSES", "VERDICTS", "read_robot_fit_precheck"]
