"""Stage two of autonomous candidate selection: measured reach on the reconstructed scene.

Beta journey: a website scene compiled from its reconstruction into the native
packet the screening runs. Each candidate whose robot registration declares a
fixed base and reach is checked against the packet's task points from that
body's own station, chosen from its own manifest when the scene was compiled
with it (``team_scene_packet.recorded_embodiment_stations``); a scene compiled
before the body was registered measures it from the packet's station, the one
its run is bound to (``team_embodiment.station_reach_blockers``). Only that
measured refusal makes a candidate infeasible. A candidate with no registered
body, a free base (its own floor station is recorded, its reach is undeclared),
or a scene without a station or task points stays unassessed and keeps its
place in the plan; nothing here is a prior about which robot suits which task.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest

SCHEMA_VERSION = "website_candidate_feasibility.v1"
FILENAME = "candidate_feasibility.json"
MEASURED_REFUSALS = frozenset({"team_embodiment_station_out_of_reach"})
OWN_STATION = "embodiment_own_station"


def _own(station: Mapping[str, Any] | None) -> dict[str, Any]:
    """The body's own station and what it measured there; nothing when the scene has none for it."""
    if station is None:
        return {}
    return {"station": station["station"], "station_basis": OWN_STATION,
            **{key: station[key] for key in ("base_mode", "reach", "grippers", "unassessed") if key in station}}


def _status(blockers: Any) -> str:
    return "feasible" if not blockers else "infeasible" if set(blockers) <= MEASURED_REFUSALS else "unassessed"


def candidate_feasibility(*, scene_plan: Mapping[str, Any], catalog: Mapping[str, Any],
                          profile: Mapping[str, Any]) -> dict[str, Any]:
    """Every catalog candidate's feasible, infeasible or unassessed status, with reasons."""
    from .beta_policy_screening_launch import _bound_path, _read
    from .team_embodiment import station_reach_blockers, validate_embodiment_manifest
    from .team_scene_packet import recorded_embodiment_stations

    configured = profile.get("screening_candidates") or {}
    stations = recorded_embodiment_stations(profile.get("team_scene_inputs_path"),
                                            plan_digest=scene_plan.get("plan_digest"))
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
        own = stations.get(manifest["embodiment_digest"])
        if manifest["base"]["mode"] != "fixed":
            rows[key] = {"status": "unassessed", "blockers": ["embodiment_free_base_reach_not_modeled"],
                         "embodiment_id": manifest["embodiment_id"],
                         **({"own_station": _own(own)} if own is not None else {})}
            continue
        # The status is measured where the run stands: a body with its own station on this scene
        # is bound there (own_station -> beta_policy_screening_team -> the binding's station), so
        # its own verdict decides; the packet's station is kept beside it for comparison. A body
        # registered after the scene was compiled has no own station and stands at the packet's.
        packet = station_reach_blockers(scene_plan, manifest)
        if own is not None:
            rows[key] = {"status": _status(own["blockers"]), "blockers": list(own["blockers"]),
                         "embodiment_id": manifest["embodiment_id"], "reach_m": manifest["reach_m"],
                         "station_basis": OWN_STATION,
                         "own_station": {**_own(own), "status": _status(own["blockers"]),
                                         "blockers": list(own["blockers"])},
                         "scene_packet_station": {"status": _status(packet), "blockers": list(packet)}}
            continue
        rows[key] = {"status": _status(packet), "blockers": list(packet),
                     "embodiment_id": manifest["embodiment_id"], "reach_m": manifest["reach_m"],
                     "station_basis": "scene_packet_station"}
    value = {"schema_version": SCHEMA_VERSION, "scene_plan_digest": scene_plan.get("plan_digest"),
             "basis": "team_embodiment.station_reach_blockers_on_reconstructed_scene_packet",
             "candidates": rows, "feasibility_digest": ""}
    value["feasibility_digest"] = canonical_digest(value, digest_field="feasibility_digest")
    return value


def station_scene_plan(scene_plan: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """The scene plan with a fixed body at its own station, for its run's binding; else the plan unchanged.

    ``team_embodiment.apply_team_embodiment`` holds a fixed base at the plan's robot station, so a run
    built from this plan stands where this body's feasibility was measured.
    """
    own = candidate.get("own_station") or {}
    station = own.get("station") if own.get("station_basis") == OWN_STATION else None
    if station is None or own.get("base_mode") != "fixed":
        return dict(scene_plan)
    robot = dict(scene_plan["robot"])
    robot["base_pose_world"] = {"position_world_m": list(station["position_world_m"]),
                                "orientation_xyzw": list(station["orientation_xyzw"])}
    value = {**scene_plan, "robot": robot, "plan_digest": ""}
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return value


def own_station(candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    """The fixed body's own station when its verdict was measured there; else ``None`` (the scene's)."""
    own = candidate.get("own_station") or {}
    if candidate.get("station_basis") != OWN_STATION or own.get("base_mode") != "fixed" or not own.get("station"):
        return None
    return {"position_world_m": list(own["station"]["position_world_m"]),
            "orientation_xyzw": list(own["station"]["orientation_xyzw"])}


def own_stance(candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    """A free base's own floor stance on this scene (``team_embodiment_stance.ground_stance``); else ``None``."""
    own = candidate.get("own_station") or {}
    stance = own.get("station")
    if own.get("base_mode") != "free" or not isinstance(stance, Mapping) or "floor_z_m" not in stance:
        return None
    return dict(stance)


def infeasible_blockers(feasibility: Mapping[str, Any]) -> dict[str, list[str]]:
    """The planner's input: measured refusals only."""
    return {key: list(row["blockers"]) for key, row in feasibility["candidates"].items()
            if row["status"] == "infeasible"}


__all__ = ["FILENAME", "SCHEMA_VERSION", "candidate_feasibility", "infeasible_blockers", "own_stance",
           "own_station", "station_scene_plan"]
