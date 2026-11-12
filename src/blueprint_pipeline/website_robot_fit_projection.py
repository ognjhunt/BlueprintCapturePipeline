"""The job's robot fit table as the site reads it (preparation status v5): a spec-bounded estimate.

Read-only. It projects the fit table the scene handoff wrote: the full route's
``robot_fit_table`` (``website_robot_fit``), or a geometry-only job's
``fit_check`` (``website_scene_handoff.geometry_only_fit_check``). The table must
be bound to the preparation the handoff names, and that preparation to the
current task context. The table is informational, so the projection never fails
a status read: anything it cannot verify is left out (None).

Two shapes. v1 (status v5 to v7) is frozen: the WebApp reads it strictly, so it reads
each row's ``verdict``, where a ``fits_if`` row is ``unknown`` as before. v2 (status v8)
reads ``first_look_verdict`` and carries ``fits_if`` with its conditions, each row's rated payload and payload limit, which checks failed, and what
is missing for an ``unknown`` row.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .website_preparation_outcome import _load

SCHEMA_VERSION = "website_robot_fit_projection.v1"
SCHEMA_VERSION_V2 = "website_robot_fit_projection.v2"
#: The WebApp accepts a status read of at most 8192 bytes; the projection keeps to half.
MAX_BYTES = 4096
#: v1's verdicts, in its row order, so a projection cut to MAX_BYTES keeps the robots that fit.
VERDICTS = ("fits", "fits_spec", "doesnt_fit", "unknown")
#: Fits if each of the row's conditions holds (``website_robot_fit``).
FITS_IF = "fits_if"
#: Every first-look verdict a table row may carry (``first_look_verdict``), in v2's row order.
FIRST_LOOK_VERDICTS = ("fits", "fits_spec", FITS_IF, "doesnt_fit", "unknown")
#: Where the table's item weight came from (``website_robot_fit._mass``).
WEIGHT_BASES = frozenset({"owner_stated", "estimated", "unavailable"})
#: Why a table check is ``unknown``. Only ``estimate_straddles_limit`` (the estimate's band straddles
#: the robot's limit) and ``item_mass_missing`` can make a check conditional; every other cause leaves
#: the row ``unknown`` and says what is missing or not done.
ESTIMATE_STRADDLES = "estimate_straddles_limit"
UNKNOWN_CAUSES = (ESTIMATE_STRADDLES, "item_mass_missing", "catalog_figure_missing", "tool_figure_missing",
                  "no_clear_station", "footprint_not_searched", "arm_pose_search_inconclusive",
                  "carry_check_undecided", "not_modelled", "not_checked")
#: The quantities a condition bounds, and their units.
QUANTITY_UNITS = {"item_mass_kg": "kg", "reach_m": "m", "cycle_time_s": "s", "route_clear": None}
#: A condition with no numbers: the route from the entrance, never checked, which the site or the
#: deployment company can judge. Its need, limit and unit are None and both bases ``not_checked``.
ROUTE_CLEAR = "route_clear"
#: Where a condition's need came from, and where its limit did.
NEED_BASES = frozenset({"estimated_from_video", "owner_stated", "unavailable", "not_checked"})
LIMIT_BASES = frozenset({"catalog", "owner_target", "not_checked"})
CHECKS = frozenset({"stance", "reach", "payload", "cycle_time", "access"})
#: Why a geometry-only job's check did not run (``geometry_only_fit_check``, the handoff).
NOT_RUN_REASONS = frozenset({"preparation_blocked", "task_strategy_not_modelled", "subject_not_excised",
                             "task_family_not_modelled"})


def _first_look(row: Mapping[str, Any]) -> Any:
    """A row's first-look verdict; a table written before it had one says only ``verdict``."""
    return row.get("first_look_verdict", row.get("verdict"))


def row_order(row: Mapping[str, Any]) -> tuple[int, float]:
    """The most useful rows first: fits, fits_spec, the conditional ones by ascending payload limit, then
    doesnt_fit and unknown. Used with a stable sort, which keeps catalog order within each."""
    verdict = _first_look(row)
    rank = FIRST_LOOK_VERDICTS.index(verdict) if verdict in FIRST_LOOK_VERDICTS else len(FIRST_LOOK_VERDICTS)
    limit = row.get("payload_limit_kg")
    usable = isinstance(limit, (int, float)) and not isinstance(limit, bool) and math.isfinite(limit)
    return rank, float(limit) if verdict == FITS_IF and usable else 0.0


def _projection(route: str, **fields: Any) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "route": route, "status": "not_run", "reason": None,
            "evidence_label": None, "counts": None, "models": [], "models_truncated": False,
            "table_digest": None, **fields}


def _verified_table(root: Path, task_context_digest: str) -> tuple[str, Any] | None:
    """The handoff's route and its bound fit table; ``(route, reason)`` when a geometry-only check did not
    run; None when there is no table or it cannot be verified. Raises on unreadable evidence."""
    path = root / "pipeline" / "website_scene_preparation" / "handoff.json"
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
            return (route, reason) if reason in NOT_RUN_REASONS else None
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
            or any(row.get("verdict") not in VERDICTS or _first_look(row) not in FIRST_LOOK_VERDICTS
                   or (_first_look(row) == FITS_IF and row["verdict"] != "unknown") for row in table["rows"])):
        return None
    return route, table


def robot_fit_rows(*, capture_root: Path, task_context_digest: str) -> list[dict[str, Any]] | None:
    """Each spec-catalog model's verdict for the current job, in the table's own order, or None.

    The screening planner's input for the known-good control's body
    (``beta_policy_screening_known_good_control``): ``model_id``, ``recipe_id`` and ``verdict`` only,
    from the same bound table the site reads. Anything it cannot verify is None, never a guess.
    """
    try:
        found = _verified_table(Path(capture_root).absolute(), task_context_digest)
        if found is None or not isinstance(found[1], Mapping):
            return None
        return [{"model_id": str(row["model_id"]),
                 "recipe_id": row["recipe_id"] if isinstance(row.get("recipe_id"), str) else None,
                 "verdict": row["verdict"]} for row in found[1]["rows"]]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def read_robot_fit(*, capture_root: Path, task_context_digest: str) -> dict[str, Any] | None:
    """The current job's robot fit table for the site, or None when there is none to show."""
    root = capture_root.absolute()
    try:
        found = _verified_table(root, task_context_digest)
        if found is None:
            return None
        route, table = found
        if not isinstance(table, Mapping):
            return _projection(route, reason=table)
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
        return _fit_bytes(value)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _first_look_counts(table: Mapping[str, Any]) -> dict[str, int]:
    """Each first-look verdict's count over the whole table, from its rows (older tables have none)."""
    return {verdict: sum(_first_look(row) == verdict for row in table["rows"]) for verdict in FIRST_LOOK_VERDICTS}


def _fit_bytes(value: dict[str, Any]) -> dict[str, Any]:
    while len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()) > MAX_BYTES:
        value["models"].pop()
        value["models_truncated"] = True
    return value


def _kg(value: Any) -> float | None:
    return (round(float(value), 3) if isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0 else None)


def _published_condition(row: Mapping[str, Any]) -> dict[str, Any]:
    """A table condition as v2 publishes it, every code from a fixed set (else ValueError)."""
    need = [row.get("need_min"), row.get("need_max")]
    if row.get("quantity") == ROUTE_CLEAR:
        if (row.get("check") != "access" or row.get("unit") is not None or need != [None, None]
                or row.get("limit") is not None or row.get("need_basis") != "not_checked"
                or row.get("limit_basis") != "not_checked"):
            raise ValueError("website_robot_fit_condition_unreadable")
        return {"quantity": ROUTE_CLEAR, "need_min": None, "need_max": None, "limit": None,
                "need_basis": "not_checked", "limit_basis": "not_checked"}
    if (row.get("check") not in CHECKS or row.get("quantity") not in QUANTITY_UNITS
            or QUANTITY_UNITS[row["quantity"]] != row.get("unit")
            or row.get("need_basis") not in NEED_BASES - {"not_checked"}
            or row.get("limit_basis") not in LIMIT_BASES - {"not_checked"}
            or _number(row.get("limit")) is None
            or (need != [None, None] and (None in map(_number, need) or need[0] > need[1]))):
        raise ValueError("website_robot_fit_condition_unreadable")
    # The quantity names its check and unit (``QUANTITY_UNITS``); v2 leaves both out to stay small.
    return {"quantity": row["quantity"], "need_min": _number(need[0]), "need_max": _number(need[1]), "limit": _number(row["limit"]),
            "need_basis": row["need_basis"], "limit_basis": row["limit_basis"]}


def _number(value: Any) -> float | None:
    return (round(float(value), 3) if isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) else None)


def _v2_model(row: Mapping[str, Any]) -> dict[str, Any]:
    """One table row as v2 publishes it; ValueError when it is not one a reader could word."""
    verdict = _first_look(row)
    hold = row.get("verdict_if_conditions_hold")
    conditions = [_published_condition(entry) for entry in row.get("conditions") or []]
    missing = [{"check": entry.get("check"), "cause": entry.get("cause")} for entry in row.get("missing") or []]
    if any(entry["check"] not in CHECKS or entry["cause"] not in UNKNOWN_CAUSES for entry in missing):
        raise ValueError("website_robot_fit_missing_unreadable")
    if verdict == FITS_IF and (hold not in ("fits", "fits_spec") or not conditions or missing):
        raise ValueError("website_robot_fit_conditional_row_unreadable")
    checks = row["checks"]
    return {"name": str(row["name"]), "manufacturer": str(row["manufacturer"]), "verdict": verdict,
            "verdict_if_conditions_hold": hold if verdict == FITS_IF else None,
            "rated_payload_kg": _kg(row.get("rated_payload_kg")), "payload_limit_kg": _kg(row.get("payload_limit_kg")),
            "failed_checks": [str(check["check"]) for check in checks if check["result"] == "fail"],
            "conditions": conditions if verdict == FITS_IF else [],
            "missing": missing if verdict == "unknown" else []}


def read_robot_fit_v2(*, capture_root: Path, task_context_digest: str) -> dict[str, Any] | None:
    """The current job's first look on paper for the site (status v8), or None when there is none to show.

    Each model says what it can carry (``payload_limit_kg``, the heaviest item it fits with). A
    ``fits_if`` row lists its ``conditions``: what the job seems to need, as a range, beside what the
    robot can do, each with its origin. ``failed_checks`` says why a ``doesnt_fit`` row does not fit;
    ``missing`` what leaves an ``unknown`` row open. Every scene figure is estimated from the video
    (``geometry_basis``), and the item weight's origin is ``weight_basis``: nothing here is measured.
    """
    root = capture_root.absolute()
    try:
        found = _verified_table(root, task_context_digest)
        if found is None:
            return None
        route, table = found
        if not isinstance(table, Mapping):
            return _projection(route, reason=table, schema_version=SCHEMA_VERSION_V2, geometry_basis=None,
                               weight_basis=None,
                               unknown_causes=None)
        counts = _first_look_counts(table)
        basis = (table.get("inputs") or {}).get("mass_basis")
        models = [_v2_model(row) for row in sorted(table["rows"], key=row_order)]
        # Every open check and cause of the whole table, counted: it survives a cut of the rows.
        causes: dict[str, int] = {}
        for model in models:
            for entry in model["missing"]:
                key = f"{entry['check']}:{entry['cause']}"
                causes[key] = causes.get(key, 0) + 1
        value = _projection(route, schema_version=SCHEMA_VERSION_V2, status="completed",
                            evidence_label="spec_bounded_estimate", counts=counts,
                            geometry_basis="estimated_from_video",
                            weight_basis=basis if basis in WEIGHT_BASES else None,
                            unknown_causes=dict(sorted(causes.items())), models=models,
                            table_digest=table["table_digest"])
        return _fit_bytes(value)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
