"""Stage two of autonomous candidate selection: measured reach on the reconstructed scene.

Beta journey: a website scene compiled from its reconstruction into the native
packet the screening runs. Each candidate whose robot registration declares a
fixed base and reach is checked with ``team_embodiment.station_reach_blockers``
against that packet's station and task points. Only that measured refusal makes
a candidate infeasible. A candidate with no registered body, a free base, or a
scene without a station or task points stays unassessed and keeps its place in
the plan; nothing here is a prior about which robot suits which task.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest

SCHEMA_VERSION = "website_candidate_feasibility.v1"
FILENAME = "candidate_feasibility.json"
MEASURED_REFUSALS = frozenset({"team_embodiment_station_out_of_reach"})


def candidate_feasibility(*, scene_plan: Mapping[str, Any], catalog: Mapping[str, Any],
                          profile: Mapping[str, Any]) -> dict[str, Any]:
    """Every catalog candidate's feasible, infeasible or unassessed status, with reasons."""
    from .beta_policy_screening_launch import _bound_path, _read
    from .team_embodiment import station_reach_blockers, validate_embodiment_manifest

    configured = profile.get("screening_candidates") or {}
    rows = {}
    for row in catalog["candidates"]:
        key = row["candidate_id"]
        settings = configured.get(key)
        registration = settings.get("robot_policy") if isinstance(settings, Mapping) else None
        if not isinstance(registration, Mapping) or not isinstance(registration.get("manifest"), Mapping):
            rows[key] = {"status": "unassessed", "blockers": ["embodiment_reach_manifest_not_registered"]}
            continue
        try:
            manifest = validate_embodiment_manifest(_read(_bound_path(registration["manifest"])))
        except (ValueError, OSError, KeyError, TypeError):
            rows[key] = {"status": "unassessed", "blockers": ["embodiment_reach_manifest_unreadable"]}
            continue
        if manifest["base"]["mode"] != "fixed":
            rows[key] = {"status": "unassessed", "blockers": ["embodiment_free_base_reach_not_modeled"],
                         "embodiment_id": manifest["embodiment_id"]}
            continue
        blockers = station_reach_blockers(scene_plan, manifest)
        status = ("feasible" if not blockers
                  else "infeasible" if set(blockers) <= MEASURED_REFUSALS else "unassessed")
        rows[key] = {"status": status, "blockers": list(blockers), "embodiment_id": manifest["embodiment_id"],
                     "reach_m": manifest["reach_m"]}
    value = {"schema_version": SCHEMA_VERSION, "scene_plan_digest": scene_plan.get("plan_digest"),
             "basis": "team_embodiment.station_reach_blockers_on_reconstructed_scene_packet",
             "candidates": rows, "feasibility_digest": ""}
    value["feasibility_digest"] = canonical_digest(value, digest_field="feasibility_digest")
    return value


def infeasible_blockers(feasibility: Mapping[str, Any]) -> dict[str, list[str]]:
    """The planner's input: measured refusals only."""
    return {key: list(row["blockers"]) for key, row in feasibility["candidates"].items()
            if row["status"] == "infeasible"}


__all__ = ["FILENAME", "SCHEMA_VERSION", "candidate_feasibility", "infeasible_blockers"]
