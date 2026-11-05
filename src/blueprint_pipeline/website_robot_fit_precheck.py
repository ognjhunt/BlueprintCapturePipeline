"""Before any spend, whether a robot we run can do a website job in its room: fit, no fit or unknown.

Beta journey step: L3 website scene preparation, at the scene handoff (``website_scene_handoff``). By then
Atlas has registered the room (its collider) and preparation has measured the task: the object's box, the
surface it stands on, the goal region and its surface, and the scale envelope the authored object must keep.
The job is not yet registered, so no ARTISAN authoring and no GPU scene build has started. Every candidate
of the screening catalog (``config/beta_policy_candidates.v1.json``, or the installed override) is read
with the late checks' own rules and code, on the CPU:

- the catalog row: what the screening plan refuses from the row alone (``catalog_blockers``: disabled,
  a scripted baseline, which a website job never runs, commercial use, an unpinned policy, a task kind
  it does not run), and a GPU memory floor the screening
  lane cannot rent (``candidate_gpu_blockers``), read with the lane's compute-capability ceiling lifted,
  since planning reads that ceiling in another process: refused only when no ceiling would rent it;
- the DROID Franka, on the station pedestal level with the lower of the task's supports, as the team
  packet stands it: the capability verdict's carry (``website_rigid_feasibility``, the carry gate's
  ``first_carrying_stance``), and the arm's own reach bound (``robot_arm_path_clearance.grasp_reach_bound``);
- the G1, on its own footprint at its pelvis height, as its packet stands it
  (``beta_policy_screening_g1_prepare.g1_stance_search``), with its nominal arm span;
- a registered team body (``team_scene_packet._embodiment_stations``): a fixed base's declared reach from
  the same pedestal stations; a free base declares none.

The early check is conservative: it says ``no_fit`` for a candidate only on a refusal the late checks
make on the same inputs, whatever the published scene's collider turns out to be: a catalog refusal, a
GPU floor no part meets under any ceiling, or a task the arm cannot reach from any station on the mount's
plane. That last one
is measured from the task's boxes, grown by the authored object's scale envelope, with no collider at
all: the farthest box is at least its height off the plane, and at least half the gap between the pick
and the place, from any station. A stance not found (reconstructed floors have holes, and the scene is
rebuilt), a carry not cleared from the stations tried, a G1 whose span misses (its lane refuses no reach:
it can walk, lean or crouch), a body whose own arm model may still reach, or a missing input is
``unknown``: flagged, and the job goes on. A candidate is ``fit`` when its embodiment's early check
passed: the Franka's carry cleared, the G1 stands within its nominal span, a fixed body's declared reach
covers the task from a station found. The Franka also fits when the capability verdict found the object
within its reach from a clear station and its scripted carry found no path: that miss is a flag
(``scripted_carry_path_not_found``), since a learned policy chooses its own grasp. That is an estimate from the
reconstructed room, not a promise.

When no candidate fits and none is unknown, the job holds before spend with
``website_robot_fit_none:<cause>`` (``HOLD_PREFIX``): a hold, never a failure. The handoff's stage is
accepted as held (``website_task_context_contracts.retains_robot_fit_hold``), and a change of the catalog
or the team registry it was decided against reruns it (``website_robot_fit_hold``). The team packet, the
screening plan and the carry gate stay the authoritative checks on the published scene.
"""
from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .usd_binary_read_cache import read_scoped
# The site's read of a precheck lives apart, so the status reader stays off this module's imports.
from .website_robot_fit_precheck_projection import (
    MAX_PROJECTION_BYTES, PROJECTION_SCHEMA, SCHEMA, VERDICTS, read_robot_fit_precheck,
)
from .website_task_context_contracts import ROBOT_FIT_HOLD_PREFIX as HOLD_PREFIX

FILENAME = "robot_fit_precheck.json"
CLAIM_CEILING = "development_only"
FRANKA_PROFILE = "droid_franka_panda_robotiq_2f85_v1"
G1_PROFILE = "unitree_g1_dex3_sonic_v1"
TEAM_EMBODIMENT = "registered_team_embodiment"
#: The task kind the team packet compiles each website strategy to.
TASK_KINDS = {"pick_and_place": "rigid_pick_place", "articulated_open_close": "articulated_open_close"}
#: Slack on every reach bound for the late check's own numbers: its IK position tolerance (2 mm), a mount
#: snapped up to the floor (``FLOOR_MOUNT_TOLERANCE_M``, 2 cm) and the task documents the published scene restates.
REACH_MARGIN_M = 0.05
REACH_RULE = "farthest_task_box_from_any_station_on_the_mount_plane"


def _r(value: float) -> float:
    return round(float(value), 6)


def _vector(value: Any, length: int = 3) -> list[float] | None:
    if (not isinstance(value, (list, tuple)) or len(value) != length
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
                   for item in value)):
        return None
    return [float(item) for item in value]


def task_geometry(preparation: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    """The task points, boxes and mount height preparation measured, or None and what is missing.

    The pick box is the object's own box; the place box is the same box at the goal, standing on it. Each
    is grown by the authored object's scale envelope (``metric_envelope.maximum_dimension_relative_error``),
    so wherever the published task holds the object inside the box it authored, the box holds it.
    """
    missing = []
    task = ((preparation.get("intake_request") or {}).get("task") or {})
    subject = preparation.get("subject") or {}
    lower, upper = _vector(subject.get("aabb_min_xyz")), _vector(subject.get("aabb_max_xyz"))
    if lower is None or upper is None or any(a > b for a, b in zip(lower, upper)):
        missing.append("subject_bounds")
    destination = preparation.get("destination") or {}
    goal = _vector(destination.get("position_world_m"))
    if goal is None:
        missing.append("destination_position")
    supports = [task.get("support")] + ([destination["support"]] if "support" in destination else [])
    tops = [_vector((row or {}).get("aabb_max_xyz")) if isinstance(row, Mapping) else None for row in supports]
    if any(top is None for top in tops):
        missing.append("support_bounds")
    error = ((preparation.get("authoring_inputs") or {}).get("metric_envelope") or {}).get(
        "maximum_dimension_relative_error")
    if isinstance(error, bool) or not isinstance(error, (int, float)) or not 0 <= error < 1:
        missing.append("metric_envelope")
    if missing:
        return None, missing
    assert lower is not None and upper is not None and goal is not None
    half = [(b - a) / 2.0 for a, b in zip(lower, upper)]
    start = [(a + b) / 2.0 for a, b in zip(lower, upper)]
    target = [goal[0], goal[1], goal[2] + half[2]]
    grow = [value * (1.0 + float(error)) for value in half]

    def box(centre: Sequence[float]) -> tuple[list[float], list[float]]:
        return [c - g for c, g in zip(centre, grow)], [c + g for c, g in zip(centre, grow)]

    return {"start_world_m": start, "target_world_m": target, "subject_minimum": lower, "subject_maximum": upper,
            "boxes": {"pick": box(start), "place": box(target)},
            "mount_z_m": min(float(top[2]) for top in tops if top is not None),
            "scale_envelope_fraction": float(error)}, []


def _gap(a: tuple[Sequence[float], Sequence[float]], b: tuple[Sequence[float], Sequence[float]]) -> float:
    return math.sqrt(sum(max(0.0, a[0][i] - b[1][i], b[0][i] - a[1][i]) ** 2 for i in range(3)))


def mount_reach_lower_bound(boxes: Mapping[str, tuple[Sequence[float], Sequence[float]]],
                            band_z: tuple[float, float]) -> float:
    """At most the farthest of ``boxes`` from any point whose height lies in ``band_z``: a lower bound.

    A point at that height is at least each box's height gap from it, and, by the triangle inequality,
    at least half the gap between any two boxes from the farther. No collider enters.
    """
    rows = list(boxes.values())
    vertical = max(max(0.0, low[2] - band_z[1], band_z[0] - high[2]) for low, high in rows)
    halves = [_gap(a, b) / 2.0 for a, b in itertools.combinations(rows, 2)]
    return max([vertical, *halves])


def _reach_bound(geometry: Mapping[str, Any], *, above_mount_m: float, bound_m: float, basis: str) -> dict[str, Any]:
    from .team_embodiment_stance import FLOOR_MOUNT_TOLERANCE_M

    mount = float(geometry["mount_z_m"])
    band = (mount + above_mount_m, mount + FLOOR_MOUNT_TOLERANCE_M + above_mount_m)
    lower = mount_reach_lower_bound(geometry["boxes"], band)
    return {"rule": REACH_RULE, "basis": basis, "mount_z_m": _r(mount), "measured_from_above_mount_m": _r(above_mount_m),
            "lower_bound_m": _r(lower), "reach_bound_m": _r(bound_m), "margin_m": REACH_MARGIN_M,
            "result": "beyond_reach" if lower > bound_m + REACH_MARGIN_M else "not_ruled_out"}


def _row(verdict: str, reasons: Sequence[str], **fields: Any) -> dict[str, Any]:
    return {"verdict": verdict, "reasons": list(reasons), **fields}


def _franka(geometry: Mapping[str, Any], verdict: Mapping[str, Any] | None) -> dict[str, Any]:
    """The Franka on the station pedestal: refused only beyond its arm's own reach bound from every station."""
    from .robot_arm_path_clearance import grasp_reach_bound

    shoulder, bound = grasp_reach_bound("franka_panda")
    reach = _reach_bound(geometry, above_mount_m=shoulder, bound_m=bound,
                         basis="robot_arm_path_clearance.grasp_reach_bound:franka_panda")
    fields = {"base": "fixed_on_station_pedestal_level_with_lower_support", "reach": reach}
    if reach["result"] == "beyond_reach":
        return _row("no_fit", ["reach"], **fields)
    carry = next((row for row in (verdict or {}).get("embodiments") or [] if row.get("robot_id") == "franka_panda"),
                 None)
    fields["carry"] = None if carry is None else {"status": carry.get("status"), "grasp": carry.get("grasp"),
                                                  "blockers": list(carry.get("blockers") or []),
                                                  **{key: carry[key] for key in ("object_reach", "flags")
                                                     if key in carry}}
    if carry is not None and carry.get("status") == "clear":
        return _row("fit", [], **fields)
    if carry is not None and carry.get("object_reach") == "within_bound":
        # The object is in the arm's reach from a clear station (the hard reach rule); the scripted carry's
        # miss is a flag, never a refusal: a learned policy chooses its own grasp.
        return _row("fit", [], **fields, flags=list(carry.get("flags") or []))
    if verdict is None:
        return _row("unknown", ["carry_not_checked_before_spend"], **fields)
    if any(str(code).startswith("robot_station_") for code in verdict.get("blockers") or []):
        return _row("unknown", ["no_clear_station_found"], **fields)
    if carry is not None and carry.get("status") == "blocked":
        # The scene is rebuilt before the carry gate runs again on it: not a refusal here.
        return _row("unknown", ["carry_blocked_from_stations_tried"], **fields)
    return _row("unknown", ["carry_undecided"], **fields)


def _g1(geometry: Mapping[str, Any], collision: Path | None, digest: str | None) -> dict[str, Any]:
    """The G1 as its own packet stands it; never refused here, since its lane refuses no reach."""
    from .beta_policy_screening_g1_prepare import g1_stance_search
    from .team_embodiment import TeamEmbodimentError

    fields: dict[str, Any] = {"base": "free_on_floor_at_pelvis_height", "reach_refuses": False}
    if collision is None:
        return _row("unknown", ["collider_unavailable"], **fields)
    try:
        stances, chosen, reach = g1_stance_search(
            collision_path=collision, collision_sha256=str(digest), offset=[0.0, 0.0, 0.0],
            start=geometry["start_world_m"], target=geometry["target_world_m"],
            bounds={"minimum": geometry["subject_minimum"], "maximum": geometry["subject_maximum"]})
    except TeamEmbodimentError as exc:
        return _row("unknown", ["no_clear_stance_found"], stance_blockers=list(exc.blockers), **fields)
    fields.update(clear_stances=len(stances), stance_position_world_m=[_r(v) for v in stances[chosen]["position_world_m"]],
                  stance_reach={key: reach.get(key) for key in ("rule", "within_nominal_arm_span",
                                                                "pick_nominal_shoulder_distance_m",
                                                                "place_nominal_shoulder_distance_m",
                                                                "nominal_arm_span_m", "status", "reason")
                                if key in reach})
    if reach.get("within_nominal_arm_span") is True:
        return _row("fit", [], **fields)
    return _row("unknown", ["nominal_arm_span_missed_body_motion_may_reach"
                            if reach.get("within_nominal_arm_span") is False else "stance_reach_unassessed"], **fields)


def _team(body: Mapping[str, Any], geometry: Mapping[str, Any], stations: Callable[[], tuple[list, str]]) -> dict[str, Any]:
    """A registered team body, as ``team_scene_packet._embodiment_stations`` stands and measures it."""
    manifest = body["manifest"]
    if manifest["base"]["mode"] != "fixed":
        return _row("unknown", ["embodiment_free_base_reach_not_modeled"], base="free_on_own_footprint")
    reach_m = float(manifest["reach_m"])
    reach = _reach_bound(geometry, above_mount_m=0.0, bound_m=reach_m, basis="embodiment_manifest.reach_m")
    fields = {"base": "fixed_on_station_pedestal_level_with_lower_support", "reach": reach,
              "own_arm_model_registered": bool(body.get("model"))}
    if reach["result"] == "beyond_reach":
        # A body whose own model is registered is measured with it (``_own_arm_stations``), which a
        # declared reach does not bound.
        return (_row("unknown", ["declared_reach_misses_own_arm_may_reach"], **fields) if body.get("model")
                else _row("no_fit", ["reach"], **fields))
    positions, source = stations()
    fields["stations"] = {"source": source, "count": len(positions)}
    if not positions:
        return _row("unknown", ["no_clear_station_found"], **fields)
    points = (geometry["start_world_m"], geometry["target_world_m"])
    if any(max(math.dist(position, point) for point in points) <= reach_m for position in positions):
        return _row("fit", [], **fields)
    return _row("unknown", ["declared_reach_misses_from_stations_found"], **fields)


def _embodiment(candidate: Mapping[str, Any], team: Mapping[str, Mapping[str, Any]]) -> str | None:
    body = team.get(candidate["candidate_id"])
    if body is not None:
        return "team:" + body["manifest"]["embodiment_digest"]
    profile = (candidate.get("interface") or {}).get("robot_profile_id")
    return profile if profile in (FRANKA_PROFILE, G1_PROFILE) else None


@read_scoped
def robot_fit_precheck(*, preparation: Mapping[str, Any], collision_path: str | Path | None = None,
                       verdict: Mapping[str, Any] | None = None, catalog: Mapping[str, Any] | None = None,
                       team_bodies: Mapping[str, Mapping[str, Any]] | None = None,
                       team_registry_digest: str | None = None) -> dict[str, Any]:
    """Every catalog candidate's early verdict for this prepared job, and the job's: fit, no fit or unknown.

    ``verdict`` is the capability verdict computed for the same preparation and collider, when one was;
    ``catalog`` a validated screening catalog (default ``load_default_catalog``); ``team_bodies`` each
    candidate's registered team body (``registered_team_bodies``) from the team registry
    ``team_registry_digest`` names. The catalog and the registry are what a hold is rechecked against
    when they change (``recheck_inputs``, ``website_robot_fit_hold``). CPU only; it buys nothing.
    """
    from .beta_policy_candidate_catalog import load_default_catalog, validate_catalog
    from .beta_policy_screening_gpu import CEILING_LIFTED, candidate_gpu_blockers
    from .beta_policy_screening_plan import TASK_KIND_BLOCKERS, catalog_blockers

    catalog = validate_catalog(load_default_catalog() if catalog is None else catalog)
    team = dict(team_bodies or {})
    collision = None if collision_path is None else Path(collision_path)
    digest = None
    if collision is not None and collision.is_file():
        sha = hashlib.sha256()
        with collision.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                sha.update(chunk)
        digest = "sha256:" + sha.hexdigest()
    elif collision is not None:
        collision = None
    if verdict is not None and (verdict.get("preparation_digest") != preparation.get("digest")
                                or verdict.get("collision_sha256") != digest):
        raise ValueError("website_robot_fit_precheck_verdict_binding_mismatch")
    strategy = ((preparation.get("intake_request") or {}).get("task") or {}).get("strategy")
    task_kind = TASK_KINDS.get(str(strategy))
    geometry, missing = task_geometry(preparation) if task_kind == "rigid_pick_place" else (None, [])
    flags = ["input_missing:" + name for name in missing]
    if task_kind is None:
        flags.append("task_strategy_not_modelled")
    elif collision is None and task_kind == "rigid_pick_place":
        flags.append("input_missing:collider")

    cache: dict[str, Any] = {}

    def stations() -> tuple[list, str]:
        # The pedestal stations every fixed base stands on (the verdict's, else the same search).
        if "stations" not in cache:
            from .website_robot_fit import _stations

            cache["stations"] = (([], "collider_unavailable") if collision is None
                                 else _stations(preparation, verdict or {}, collision, str(digest)))
        return cache["stations"]

    def measured(key: str) -> dict[str, Any]:
        if key not in cache:
            if task_kind != "rigid_pick_place":
                cache[key] = _row("unknown", ["task_geometry_not_modelled_before_authoring"])
            elif geometry is None:
                cache[key] = _row("unknown", ["task_inputs_missing"])
            elif key == FRANKA_PROFILE:
                cache[key] = _franka(geometry, verdict)
            elif key == G1_PROFILE:
                cache[key] = _g1(geometry, collision, digest)
            else:
                body = next(row for row in team.values() if "team:" + row["manifest"]["embodiment_digest"] == key)
                cache[key] = _team(body, geometry, stations)
        return cache[key]

    rows = []
    for candidate in catalog["candidates"]:
        key = candidate["candidate_id"]
        # A website job's own refusals: a scripted baseline is switched off there, so it never fits.
        refused = catalog_blockers(candidate, task_kind or "", website=True)
        kind = sorted(code for code in refused if code in TASK_KIND_BLOCKERS)
        other = sorted(set(refused) - set(kind))
        embodiment = _embodiment(candidate, team)
        row: dict[str, Any] = {"candidate_id": key, "embodiment": embodiment}
        if other:
            row.update(verdict="unavailable", causes=["catalog"], blockers=other)
            rows.append(row)
            continue
        if task_kind is None:
            row.update(verdict="unknown", causes=[], blockers=["task_strategy_not_modelled"])
            rows.append(row)
            continue
        # Every sound refusal is kept; a candidate fits only with none and nothing undecided.
        causes, blockers, open_ = [], [], []
        if kind:
            causes.append("task_kind")
            blockers += kind
        else:
            # Planning reads the compute-capability ceiling of the process it runs in. Lifted, a floor no
            # part meets is unrentable under any ceiling; one only this process's ceiling excludes is unknown.
            lifted = candidate_gpu_blockers(candidate, max_compute_cap=CEILING_LIFTED)
            if lifted:
                causes.append("gpu_memory")
                blockers += list(lifted)
            elif candidate_gpu_blockers(candidate):
                open_.append("gpu_memory_floor_rentable_only_above_this_compute_cap_ceiling")
            if embodiment is None:
                open_.append("embodiment_not_modeled_before_spend")
            else:
                measured_row = measured(embodiment)
                if measured_row["verdict"] == "no_fit":
                    causes += list(measured_row["reasons"])
                    blockers += list(measured_row["reasons"])
                elif measured_row["verdict"] == "unknown":
                    open_ += list(measured_row["reasons"])
        row.update(verdict="no_fit" if causes else "unknown" if open_ else "fit", causes=causes,
                   blockers=blockers if causes else open_)
        rows.append(row)

    counts = {name: sum(row["verdict"] == name for row in rows) for name in VERDICTS}
    fit = [row["candidate_id"] for row in rows if row["verdict"] == "fit"]
    status = "fit" if fit else "unknown" if counts["unknown"] else "no_fit"
    hold = None
    causes: dict[str, int] = {}
    for row in rows:
        for cause in row["causes"] if row["verdict"] == "no_fit" else ():
            causes[cause] = causes.get(cause, 0) + 1
    if status == "no_fit":
        # The cause that refused the most candidates names the hold; every cause is counted beside it.
        hold = HOLD_PREFIX + (min(causes, key=lambda cause: (-causes[cause], cause)) if causes
                              else "no_candidate_available")
    unknowns = sorted({code for row in rows if row["verdict"] == "unknown" for code in row["blockers"]})
    embodiments = {}
    for key, value in cache.items():
        if key == "stations":
            continue
        name = TEAM_EMBODIMENT if key.startswith("team:") else key
        embodiments[key] = {"embodiment": name, **value}
    value: dict[str, Any] = {
        "schema_version": SCHEMA, "preparation_digest": preparation.get("digest"), "collision_sha256": digest,
        "catalog_digest": catalog["catalog_digest"],
        "recheck_inputs": {"catalog_digest": catalog["catalog_digest"], "team_registry_digest": team_registry_digest},
        "verdict_digest": None if verdict is None else verdict.get("verdict_digest"),
        "strategy": strategy, "task_kind": task_kind, "status": status, "hold_reason": hold,
        "no_fit_causes": dict(sorted(causes.items())), "fit_candidates": fit, "counts": counts, "flags": sorted(set(flags) | {"unknown:" + code for code in unknowns}),
        "task": None if geometry is None else {
            "start_world_m": [_r(v) for v in geometry["start_world_m"]],
            "target_world_m": [_r(v) for v in geometry["target_world_m"]],
            "mount_z_m": _r(geometry["mount_z_m"]), "scale_envelope_fraction": geometry["scale_envelope_fraction"],
            "boxes": {name: [[_r(v) for v in low], [_r(v) for v in high]]
                      for name, (low, high) in geometry["boxes"].items()}},
        "embodiments": embodiments, "candidates": rows,
        "rule": "no_fit_only_on_refusals_the_late_checks_make_on_any_published_collider",
        "authoritative_checks": ["team_scene_packet", "robot_arm_path_clearance_carry_gate",
                                 "beta_policy_screening_g1_prepare", "beta_policy_screening_plan"],
        "claim_ceiling": CLAIM_CEILING, "physical_world_truth_claimed": False, "precheck_digest": ""}
    value["precheck_digest"] = canonical_digest(value, digest_field="precheck_digest")
    return value


def registered_team_bodies(registry: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Each screening candidate's registered team body, as the team packet reads it, keyed by candidate.

    ``screening_candidates.<id>.robot_policy`` under the operator team registry's derived defaults
    (``team_scene_packet_worker.registered_embodiments``): the validated manifest, and whether a robot
    model is registered with it, since that model, not the declared reach, then chooses its station.
    None configured is none registered; a row that does not read is left out (its candidate stays unknown).
    """
    from .beta_policy_screening_launch import _bound_path, _read
    from .team_candidate_quick10_registry import _derived_settings, load_registry
    from .team_embodiment import validate_embodiment_manifest

    registry = load_registry() if registry is None else registry
    if registry is None:
        return {}
    configured = ((_derived_settings(registry) or {}).get("defaults") or {}).get("screening_candidates") or {}
    bodies = {}
    for key, settings in sorted(configured.items()):
        registration = settings.get("robot_policy") if isinstance(settings, Mapping) else None
        if not isinstance(registration, Mapping) or not isinstance(registration.get("manifest"), Mapping):
            continue
        try:
            manifest = validate_embodiment_manifest(_read(_bound_path(registration["manifest"])))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        bodies[str(key)] = {"manifest": manifest, "model": isinstance(registration.get("model"), Mapping)}
    return bodies


__all__ = ["FILENAME", "HOLD_PREFIX", "MAX_PROJECTION_BYTES", "PROJECTION_SCHEMA", "SCHEMA",
           "mount_reach_lower_bound", "read_robot_fit_precheck", "registered_team_bodies",
           "robot_fit_precheck", "task_geometry"]
