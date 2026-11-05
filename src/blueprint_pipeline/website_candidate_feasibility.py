"""Stage two of autonomous candidate selection: measured reach on the reconstructed scene.

Beta journey: a website scene compiled from its reconstruction into the native
packet the screening runs. Reach is decided here, at planning, before anything
is reserved, from evidence already computed for the packet, and each row says
which source decided it (``reach_source``):

- ``franka_chain_reach_bound``: a candidate on Blueprint's DROID Franka runs
  that arm at the packet's station. When every point of the object, at its
  start or wherever the task accepts it at its place, lies beyond the arm's
  chain reach bound from that station (the hard reach rule,
  ``robot_arm_path_clearance.rigid_reach_shortfall``), no grasp or policy
  reaches it there: the candidate is infeasible with
  ``robot_reach_insufficient:<phase>:<distance>m_vs_<bound>m``, the object's
  nearest point. The team half chose that station as one with the object in
  reach whenever any clear stance has it (``team_scene_packet``). Otherwise the
  candidate is feasible, and the packet's recorded station search
  (``robot_station_search``, the scripted carry) is kept as evidence: a station
  it cleared is ``packet_station_search``; one it could not clear carries the
  ``scripted_carry_path_not_found`` flag, since a learned policy chooses its own
  grasp, and is funded after unflagged rows (``reach_rank``).
- ``own_arm_model``: a registered fixed body with its robot model is solved
  through the task's authored phases at the station its run is bound to
  (``team_arm_reach``). ``out_of_reach`` (every arm has a phase beyond its
  chain's bound) is infeasible with the same typed reason; ``passed`` is
  feasible whatever its declared reach. A measured solve always outranks the
  declared figure.
- ``declared_reach_m``: without a measured verdict, a fixed body's declared
  reach against the task points from that body's own station, chosen from its
  own manifest when the scene was compiled with it
  (``team_scene_packet.recorded_embodiment_stations``); a scene compiled before
  the body was registered measures it from the packet's station, the one its
  run is bound to (``team_embodiment.station_reach_blockers``).

Only a measured refusal makes a candidate infeasible. A candidate with no
registered body or known arm, a free base (its own floor station is recorded,
its reach is undeclared), or a scene without a station or task points stays
unassessed and keeps its place in the plan; nothing here is a prior about which
robot suits which task, and unknown reach never blocks.

The same evidence orders the budget (``reach_rank``): a candidate whose solve
searched and did not reach (no proof that it cannot) is funded after every
unflagged one. A job's intake ranking otherwise stands as it is; on the
default-priority path learned policies go before scripted baselines and a solve
that reached the task before an unmeasured one. Ranking removes nothing.
"""
from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from .decision_evidence_contracts import canonical_digest

SCHEMA_VERSION = "website_candidate_feasibility.v1"
FILENAME = "candidate_feasibility.json"
MEASURED_REFUSALS = frozenset({"team_embodiment_station_out_of_reach"})
#: A measured reach shortfall's typed reason (``robot_arm_path_clearance.reach_insufficient_code``).
REACH_INSUFFICIENT_PREFIX = "robot_reach_insufficient:"
OWN_STATION = "embodiment_own_station"
PACKET_STATION = "scene_packet_station"


def is_measured_refusal(code: Any) -> bool:
    """A reason measured on this scene's geometry: the site sees it as infeasible, not a setup fault."""
    return isinstance(code, str) and (code in MEASURED_REFUSALS or code.startswith(REACH_INSUFFICIENT_PREFIX))


def _pose(value: Mapping[str, Any]) -> dict[str, list[float]]:
    return {"position_world_m": [float(v) for v in value["position_world_m"]],
            "orientation_xyzw": [float(v) for v in value["orientation_xyzw"]]}


def _packet_arm_reach(scene_plan: Mapping[str, Any], search: Mapping[str, Any]) -> dict[str, Any]:
    """Blueprint's DROID Franka at the packet's station: the reach proof, else the station search's verdict."""
    from .robot_arm_path_clearance import (
        SCRIPTED_CARRY_FLAG,
        RobotArmPathClearanceError,
        object_reach_inputs,
        reach_shortfall_code,
        rigid_reach_shortfall,
    )

    robot, task = scene_plan.get("robot") or {}, scene_plan.get("task_spec") or {}
    row: dict[str, Any] = {"robot_id": robot.get("robot_id"), "station_basis": PACKET_STATION}
    # The packet's rigid carry, as its arm-path gate measures one (``packet_arm_path_clearance``).
    articulated = any(isinstance(item, Mapping) and item.get("object_type") == "ARTICULATION"
                      for item in scene_plan.get("objects") or [])
    if articulated or task.get("manipulation_strategy") != "pick_and_place":
        return {**row, "status": "unassessed", "blockers": ["packet_arm_reach_task_unmodeled"], "reach_source": None}
    try:
        reach = rigid_reach_shortfall(robot_id=str(robot["robot_id"]), base_pose_world=_pose(robot["base_pose_world"]),
                                      **object_reach_inputs(task))
    except (KeyError, TypeError, ValueError, RobotArmPathClearanceError):
        return {**row, "status": "unassessed", "blockers": ["packet_arm_reach_inputs_invalid"], "reach_source": None}
    row["reach"] = reach
    if reach["beyond"]:
        # No point of the object is in reach at its start or its place: no grasp or policy reaches it.
        return {**row, "status": "infeasible", "reach_source": "franka_chain_reach_bound",
                "blockers": [reach_shortfall_code(reach)]}
    station = search.get("robot_station") or {}
    recorded = search.get("robot_station_search") or {}
    result = next((item for item in recorded.get("results") or []
                   if item.get("stance_digest") == station.get("stance_digest")), None)
    within = reach["status"] == "within_bound"
    if result is None or _pose(station) != _pose(robot["base_pose_world"]):
        if within:
            return {**row, "status": "feasible", "blockers": [], "reach_source": "franka_chain_reach_bound",
                    "flags": ["packet_station_search_not_recorded"]}
        return {**row, "status": "unassessed", "blockers": ["packet_station_search_not_recorded"], "reach_source": None}
    row["station_search"] = {"status": result["status"], "blockers": list(result["blockers"]),
                             "cleared": recorded.get("cleared")}
    if result["status"] == "clear":
        return {**row, "status": "feasible", "blockers": [], "reach_source": "packet_station_search"}
    if within:
        # The object is in reach; the scripted carry found no path. A learned policy chooses its own grasp:
        # planned and screened, with the miss recorded as a flag (and funded after unflagged rows, ``reach_rank``).
        return {**row, "status": "feasible", "blockers": [], "reach_source": "franka_chain_reach_bound",
                "flags": [SCRIPTED_CARRY_FLAG]}
    # The gate's search found no carry, but nothing proves there is none: unknown, never a refusal.
    return {**row, "status": "unassessed", "reach_source": None,
            "blockers": ["packet_station_search_blocked:" + code for code in result["blockers"]]}


def _own_arm(scene_plan: Mapping[str, Any], manifest: Mapping[str, Any], registration: Mapping[str, Any],
             station: Mapping[str, Any]) -> dict[str, Any] | None:
    """The fixed body's own arm through the task from ``station`` in its bound model, or ``None`` with no model."""
    if not isinstance(registration.get("model"), Mapping) or not isinstance(station, Mapping):
        return None
    from .beta_policy_screening_launch import _bound_path
    from .team_embodiment import ROBOT_ID

    try:
        from .team_arm_reach import team_arm_reach

        robot = {"robot_id": ROBOT_ID, "embodiment_id": manifest["embodiment_id"],
                 "embodiment_digest": manifest["embodiment_digest"], "base_mode": "fixed_at_station",
                 "usd_sha256": manifest["model"]["sha256"], "base_pose_world": _pose(station)}
        report = team_arm_reach({**scene_plan, "robot": robot}, model_path=_bound_path(registration["model"]),
                                manifest=manifest)
    except (ImportError, OSError, ArithmeticError, KeyError, TypeError, ValueError) as exc:
        return {"status": "unassessed", "reason": "own_arm_reach_inputs_invalid", "detail": type(exc).__name__}
    return {key: report[key] for key in ("status", "reason", "detail", "advisories", "reaching_arms", "beyond_reach",
                                         "claim_ceiling", "report_digest") if key in report}


def _decide_fixed(row: dict[str, Any], arm: Mapping[str, Any] | None) -> dict[str, Any]:
    """A measured solve decides over the declared reach; without one the declared verdict stands, and says so."""
    from .robot_arm_path_clearance import reach_insufficient_code

    if arm is None or arm.get("status") not in {"passed", "out_of_reach"}:
        return {**row, **({"own_arm": dict(arm)} if arm is not None else {}),
                "reach_source": "declared_reach_m" if row["status"] != "unassessed" else None}
    if arm["status"] == "passed":
        return {**row, "status": "feasible", "blockers": [], "own_arm": dict(arm), "reach_source": "own_arm_model"}
    beyond = arm.get("beyond_reach") or {}
    code = (reach_insufficient_code(beyond["phase_id"], beyond["distance_m"], beyond["reach_bound_m"])
            if beyond.get("distance_m") is not None and beyond.get("reach_bound_m") is not None
            else REACH_INSUFFICIENT_PREFIX + str(beyond.get("phase_id") or "task"))
    return {**row, "status": "infeasible", "blockers": [code], "own_arm": dict(arm), "reach_source": "own_arm_model"}


def _own(station: Mapping[str, Any] | None) -> dict[str, Any]:
    """The body's own station and what it measured there; nothing when the scene has none for it."""
    if station is None:
        return {}
    return {"station": station["station"], "station_basis": OWN_STATION,
            **{key: station[key] for key in ("base_mode", "reach", "grippers", "unassessed") if key in station}}


def _status(blockers: Any) -> str:
    return ("feasible" if not blockers else "infeasible" if all(is_measured_refusal(code) for code in blockers)
            else "unassessed")


def candidate_feasibility(*, scene_plan: Mapping[str, Any], catalog: Mapping[str, Any],
                          profile: Mapping[str, Any]) -> dict[str, Any]:
    """Every catalog candidate's feasible, infeasible or unassessed status, with reasons and its deciding source."""
    from .beta_policy_screening_launch import _bound_path, _read
    from .robot_arm_path_clearance import ROBOT_PROFILE_ARMS
    from .team_embodiment import station_reach_blockers, validate_embodiment_manifest
    from .team_scene_packet import recorded_embodiment_stations, recorded_station_search

    configured = profile.get("screening_candidates") or {}
    stations = recorded_embodiment_stations(profile.get("team_scene_inputs_path"),
                                            plan_digest=scene_plan.get("plan_digest"))
    packet_robot = str((scene_plan.get("robot") or {}).get("robot_id"))
    packet_arm: dict[str, Any] | None = None  # One verdict for every candidate on the packet's own arm.
    rows = {}
    for row in catalog["candidates"]:
        key = row["candidate_id"]
        settings = configured.get(key)
        registration = settings.get("robot_policy") if isinstance(settings, Mapping) else None
        if not isinstance(registration, Mapping) or not isinstance(registration.get("manifest"), Mapping):
            profile_id = (row.get("interface") or {}).get("robot_profile_id")
            if (not (isinstance(settings, Mapping) and "native_g1" in settings)
                    and ROBOT_PROFILE_ARMS.get(profile_id) == packet_robot):
                # The candidate runs the packet's own arm at the packet's station: measured there.
                if packet_arm is None:
                    packet_arm = _packet_arm_reach(scene_plan, recorded_station_search(
                        profile.get("team_scene_inputs_path"), plan_digest=scene_plan.get("plan_digest")))
                rows[key] = {**packet_arm, "robot_profile_id": profile_id}
                continue
            rows[key] = {"status": "unassessed", "blockers": ["embodiment_reach_manifest_not_registered"]}
            continue
        try:
            manifest = validate_embodiment_manifest(_read(_bound_path(registration["manifest"])))
        except (ValueError, OSError, KeyError, TypeError):
            rows[key] = {"status": "unassessed", "blockers": ["embodiment_reach_manifest_unreadable"]}
            continue
        own = stations.get(manifest["embodiment_digest"])
        if manifest["base"]["mode"] != "fixed":
            # A free base whose own arm reached the task from the floor stance it starts at is feasible
            # there (team_scene_packet._own_arm_stations); otherwise its reach stays unassessed.
            arm = ((own or {}).get("reach") or {}).get("own_arm") or {}
            blockers = [] if arm.get("status") == "passed" else ["embodiment_free_base_reach_not_modeled"]
            rows[key] = {"status": _status(blockers), "blockers": blockers,
                         "embodiment_id": manifest["embodiment_id"],
                         **({"own_station": _own(own)} if own is not None else {})}
            continue
        # The status is measured where the run stands: a body with its own station on this scene
        # is bound there (own_station -> beta_policy_screening_team -> the binding's station), so
        # its own verdict decides; the packet's station is kept beside it for comparison. A body
        # registered after the scene was compiled has no own station and stands at the packet's.
        packet = station_reach_blockers(scene_plan, manifest)
        if own is not None:
            measured = {"status": _status(own["blockers"]), "blockers": list(own["blockers"]),
                        "embodiment_id": manifest["embodiment_id"], "reach_m": manifest["reach_m"],
                        "station_basis": OWN_STATION,
                        "own_station": {**_own(own), "status": _status(own["blockers"]),
                                        "blockers": list(own["blockers"])},
                        "scene_packet_station": {"status": _status(packet), "blockers": list(packet)}}
            rows[key] = _decide_fixed(measured, _own_arm(scene_plan, manifest, registration, own["station"]))
            continue
        declared = {"status": _status(packet), "blockers": list(packet),
                    "embodiment_id": manifest["embodiment_id"], "reach_m": manifest["reach_m"],
                    "station_basis": PACKET_STATION}
        robot = scene_plan.get("robot") or {}
        station = robot.get("base_pose_world") if isinstance(robot.get("base_pose_world"), Mapping) else None
        rows[key] = _decide_fixed(declared, None if station is None else
                                  _own_arm(scene_plan, manifest, registration, station))
    value = {"schema_version": SCHEMA_VERSION, "scene_plan_digest": scene_plan.get("plan_digest"),
             "basis": "measured_reach_on_reconstructed_scene_packet",
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


def own_station_reached_by_own_arm(candidate: Mapping[str, Any]) -> bool:
    """Whether a fixed body's own station was chosen because its own arm reaches the task from there
    (``team_scene_packet._own_arm_stations``), so its run binds it there whatever its declared reach."""
    own = candidate.get("own_station") or {}
    reach = own.get("reach") or {}
    return (own_station(candidate) is not None
            and reach.get("rule") == "closest_clear_stance_own_arm_reaches_every_phase"
            and (reach.get("own_arm") or {}).get("status") == "passed")


def bound_station(candidate: Mapping[str, Any],
                  scene_plan: Mapping[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    """The station a fixed body's run binds (``None``: the scene's) and whether its own arm reaches from there.

    Its own station when the scene was compiled with one; when the arm was solved reaching every phase at
    planning (``reach_source`` ``own_arm_model``), the station it was solved at, so its declared reach no
    longer vetoes the binding (``team_embodiment.team_embodiment_binding``).
    """
    station = own_station(candidate)
    if own_station_reached_by_own_arm(candidate):
        return station, True
    if candidate.get("reach_source") == "own_arm_model" and (candidate.get("own_arm") or {}).get("status") == "passed":
        robot = scene_plan.get("robot") or {}
        return station or _pose(robot["base_pose_world"]), True
    return station, False


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


#: A row's reach tier by basis. The plan funds tier 3 last; without an intake ranking, learned before
#: scripted and then 1 before 2.
REACH_TIERS = {"own_arm_solve_passed": 1, "reach_unmeasured": 2, "own_arm_solve_flagged": 3}
REACH_RANK_FIELDS = frozenset({"tier", "basis", "feasibility_status", "reach_source", "solve_source",
                               "solve_status", "report_digest", "reasons", "feasibility_digest"})
_REASONS = 16


def _codes(values: Any) -> list[str]:
    return [value for value in values or [] if isinstance(value, str) and 1 <= len(value) <= 300][:_REASONS]


def _name(value: Any) -> str | None:
    return value if isinstance(value, str) and 1 <= len(value) <= 128 else None


def _sha(value: Any) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) else None


def reach_rank(row: Mapping[str, Any] | None, *, feasibility_digest: str | None = None) -> dict[str, Any]:
    """Where a feasible or unassessed candidate's reach evidence places it in the budget order.

    1 ``own_arm_solve_passed``: its own arm was solved through the task on this scene and reached it:
      a registered body's model (``team_arm_reach`` ``passed`` at planning, else where its own station
      was chosen) or the packet's DROID Franka (its recorded station search cleared).
    2 ``reach_unmeasured``: no solve decided either way: no model or recorded search, an unmodelled task,
      a free base (it can move from where its solve missed), or a candidate this feasibility never saw.
    3 ``own_arm_solve_flagged``: a fixed arm's solve searched and did not reach, without the proof that
      would make it infeasible (``team_arm_reach`` ``flagged``; the packet Franka's search blocked).
      Its declared reach or chain bound did not refuse it, so it stays eligible and is funded last.

    It only orders (``beta_policy_screening_plan.build_screening_plan``); no candidate is removed.
    """
    row = row or {}
    free = (row.get("own_station") or {}).get("base_mode") == "free"
    solves = [(source, arm) for source, arm in (
        ("own_arm_model", row.get("own_arm")),
        ("own_station_own_arm", ((row.get("own_station") or {}).get("reach") or {}).get("own_arm")))
        if isinstance(arm, Mapping)]
    search = row.get("station_search")
    basis, solve = "reach_unmeasured", None
    for source, arm in solves:  # A solve at planning decides over the one that chose the station.
        if arm.get("status") == "passed" or arm.get("status") == "flagged" and not free:
            basis = "own_arm_solve_passed" if arm["status"] == "passed" else "own_arm_solve_flagged"
            solve = (source, arm["status"], _codes(arm.get("advisories")), arm.get("report_digest"))
            break
    if solve is None and isinstance(search, Mapping) and isinstance(search.get("status"), str):
        basis = "own_arm_solve_passed" if search["status"] == "clear" else "own_arm_solve_flagged"
        solve = ("packet_station_search", search["status"], _codes(search.get("blockers")), None)
    if solve is None:
        source, arm = solves[0] if solves else (None, {})
        reasons = _codes([*(row.get("blockers") or []), arm.get("reason"), *(arm.get("advisories") or [])])
        solve = (source, arm.get("status"), reasons or ["candidate_reach_not_recorded"], arm.get("report_digest"))
    source, status, reasons, digest = solve
    return {"tier": REACH_TIERS[basis], "basis": basis,
            "feasibility_status": row.get("status") if row.get("status") in {"feasible", "unassessed"} else None,
            "reach_source": _name(row.get("reach_source")), "solve_source": source, "solve_status": _name(status),
            "report_digest": _sha(digest), "reasons": reasons, "feasibility_digest": _sha(feasibility_digest)}


def reach_ranks(feasibility: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Every candidate's reach rank but the measured refusals', which are never funded."""
    return {key: reach_rank(row, feasibility_digest=feasibility.get("feasibility_digest"))
            for key, row in feasibility["candidates"].items() if row["status"] != "infeasible"}


__all__ = ["FILENAME", "MEASURED_REFUSALS", "REACH_INSUFFICIENT_PREFIX", "REACH_RANK_FIELDS", "REACH_TIERS",
           "SCHEMA_VERSION", "bound_station", "candidate_feasibility", "infeasible_blockers", "is_measured_refusal",
           "own_stance", "own_station", "reach_rank", "reach_ranks", "station_scene_plan"]
