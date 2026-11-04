"""The early robot fit precheck never refuses what the late checks accept, on the same synthetic scenes.

Each scene is checked twice. Early: ``website_robot_fit_precheck`` from the prepared job, before spend.
Late: the checks a built scene meets, called as they run: the team packet's Franka station and carry
(``team_scene_packet._carrying_station``, the carry gate), a registered body's own station and declared
reach (``team_scene_packet._embodiment_stations`` with ``website_candidate_feasibility``), the G1
packet's stance (``beta_policy_screening_g1_prepare.g1_stance_search``), and the screening plan's
refusals and GPU rule (``beta_policy_screening_plan``, ``beta_policy_screening_gpu``). Wherever the
early check says no fit, the late check refuses too; wherever the late check accepts, the early check
does not say no fit.
"""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_robot_fit_precheck.py
#   src/blueprint_pipeline/team_scene_packet.py
#   src/blueprint_pipeline/team_embodiment_stance.py
#   src/blueprint_pipeline/beta_policy_screening_g1_prepare.py
#   src/blueprint_pipeline/beta_policy_screening_plan.py
#   src/blueprint_pipeline/beta_policy_screening_gpu.py
#   src/blueprint_pipeline/robot_arm_path_clearance.py

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog
from blueprint_pipeline.beta_policy_screening_gpu import candidate_gpu_blockers
from blueprint_pipeline.beta_policy_screening_plan import _candidate_blockers
from blueprint_pipeline.team_embodiment import TeamEmbodimentError, validate_embodiment_manifest
from blueprint_pipeline.website_robot_fit_precheck import FRANKA_PROFILE, G1_PROFILE, robot_fit_precheck
from tests.test_website_robot_fit_precheck import fixed_arm_catalog, long_table_scene, shelf_scene, tray_scene

SCENES = {"tray_on_dining_table": tray_scene, "shelf_above_any_reach": shelf_scene,
          "carry_along_a_long_table": long_table_scene}
#: A team's fixed arm with a short and a long declared reach, registered without its own robot model.
REACHES = {"short_arm": 0.6, "long_arm": 2.5}


def _manifest(tmp_path: Path, name: str, reach: float) -> dict[str, Any]:
    from tests.test_team_embodiment import manifest

    model = tmp_path / f"{name}.usda"
    model.write_text("#usda 1.0\n")
    return validate_embodiment_manifest(manifest(model, embodiment_id=name, reach_m=reach))


def _task(preparation: dict[str, Any]) -> tuple[list[float], list[float]]:
    low, high = preparation["subject"]["aabb_min_xyz"], preparation["subject"]["aabb_max_xyz"]
    goal = preparation["destination"]["position_world_m"]
    return [(a + b) / 2 for a, b in zip(low, high)], [goal[0], goal[1], goal[2] + (high[2] - low[2]) / 2]


def _place(preparation: dict[str, Any], collision: Path) -> dict[str, Any]:
    """The team packet's station search (``carry_station_search``), less its ranking."""
    from blueprint_pipeline.website_rigid_feasibility import carry_station_search

    search = dict(carry_station_search(preparation))
    search.pop("ranked")
    return {"collision_path": collision, "collision_sha256": "sha256:" + hashlib.sha256(collision.read_bytes()).hexdigest(),
            **search, "task_motion": None}


def late_franka(preparation: dict[str, Any], collision: Path) -> bool:
    """Whether the team packet's Franka station carries the task through the carry gate."""
    from blueprint_pipeline.native_franka_action_math import grasp_orientation_contact_xyzw
    from blueprint_pipeline.robot_arm_path_clearance import CARRY_STATIONS
    from blueprint_pipeline.team_embodiment_stance import ground_stance
    from blueprint_pipeline.team_scene_packet import _carrying_station

    try:
        stations = ground_stance(**_place(preparation, collision), ranked=CARRY_STATIONS)
    except TeamEmbodimentError:
        return False  # The packet is refused by name: team_scene_station_inputs_invalid or a stance refusal.
    start, target = _task(preparation)
    task_spec = {"start_pose_world": [*start, 0.0, 0.0, 0.0, 1.0], "target_position_world_m": target,
                 "interaction_affordance": {
                     "contact_point_scoring_frame_m": [0.0, 0.0, 0.0], "approach_unit_scoring_frame": [0.0, 0.0, 1.0],
                     "gripper_orientation_scoring_frame_xyzw": grasp_orientation_contact_xyzw(
                         approach_axis=[0.0, 0.0, -1.0], jaw_axis=[0.0, 1.0, 0.0]),
                     "pregrasp_clearance_m": 0.1}}
    reset = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    profile = {"robot_joint_reset_positions_rad": {f"panda_joint{i}": value for i, value in enumerate(reset, 1)}}
    _, search = _carrying_station(stations, task_spec=task_spec, collision=collision, profile=profile)
    return search["cleared"]


def late_team_fixed(preparation: dict[str, Any], collision: Path, manifest: dict[str, Any]) -> bool:
    """Whether the team packet's own station for this body leaves its candidate feasible in screening."""
    from blueprint_pipeline.team_scene_packet import _embodiment_stations
    from blueprint_pipeline.website_candidate_feasibility import _status

    row = _embodiment_stations([manifest], place=_place(preparation, collision), points=list(_task(preparation)),
                               grasp=None)[manifest["embodiment_digest"]]
    return _status(row["blockers"]) != "infeasible"


def late_g1(preparation: dict[str, Any], collision: Path) -> bool:
    """Whether the G1 packet finds the stance it stands at; its reach is advisory there."""
    from blueprint_pipeline.beta_policy_screening_g1_prepare import g1_stance_search

    low, high = preparation["subject"]["aabb_min_xyz"], preparation["subject"]["aabb_max_xyz"]
    start, target = _task(preparation)
    try:
        g1_stance_search(collision_path=collision, collision_sha256="sha256:" + "0" * 64, offset=[0.0, 0.0, 0.0],
                         start=start, target=target, bounds={"minimum": low, "maximum": high})
    except TeamEmbodimentError:
        return False
    return True


@pytest.fixture(scope="module", params=sorted(SCENES))
def scene(request, tmp_path_factory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp(request.param)
    preparation, collision = SCENES[request.param](root)
    bodies = {name: _manifest(root, name, reach) for name, reach in REACHES.items()}
    return {"name": request.param, "preparation": preparation, "collision": collision, "bodies": bodies,
            "early": robot_fit_precheck(preparation=preparation, collision_path=collision),
            "late": {FRANKA_PROFILE: late_franka(preparation, collision), G1_PROFILE: late_g1(preparation, collision),
                     **{"team:" + manifest["embodiment_digest"]: late_team_fixed(preparation, collision, manifest)
                        for manifest in bodies.values()}}}


def _registered_candidate() -> str:
    """An enabled rigid candidate a team body is registered for here (the shipped catalog's team rows are off)."""
    return next(row["candidate_id"] for row in load_default_catalog()["candidates"]
                if (row.get("interface") or {}).get("robot_profile_id") == FRANKA_PROFILE)


def _team_early(scene: dict[str, Any], manifest: dict[str, Any], *, model: bool = False) -> dict[str, Any]:
    """The early verdict for a registered team body on this scene."""
    value = robot_fit_precheck(preparation=scene["preparation"], collision_path=scene["collision"],
                               team_bodies={_registered_candidate(): {"manifest": manifest, "model": model}})
    return value["embodiments"]["team:" + manifest["embodiment_digest"]]


def test_the_scenes_cover_fit_no_fit_and_unknown(scene):
    expected = {"tray_on_dining_table": (True, "not_ruled_out"), "shelf_above_any_reach": (False, "beyond_reach"),
                "carry_along_a_long_table": (False, "beyond_reach")}[scene["name"]]
    franka = scene["early"]["embodiments"][FRANKA_PROFILE]
    assert (scene["late"][FRANKA_PROFILE], franka["reach"]["result"]) == expected


def test_no_embodiment_the_late_check_accepts_is_refused_early(scene):
    for key, accepted in scene["late"].items():
        early = (scene["early"]["embodiments"].get(key) if not key.startswith("team:")
                 else _team_early(scene, next(m for m in scene["bodies"].values()
                                              if "team:" + m["embodiment_digest"] == key)))
        assert early is not None, key
        if accepted:
            assert early["verdict"] != "no_fit", (scene["name"], key, early)
        if early["verdict"] == "no_fit":
            assert not accepted, (scene["name"], key)


def test_the_g1_is_never_refused_early_because_its_lane_refuses_no_reach(scene):
    assert scene["early"]["embodiments"][G1_PROFILE]["verdict"] in {"fit", "unknown"}


def test_a_declared_reach_the_scene_rules_out_is_refused_by_both(scene):
    short = scene["bodies"]["short_arm"]
    early = _team_early(scene, short)
    late = scene["late"]["team:" + short["embodiment_digest"]]
    if scene["name"] == "shelf_above_any_reach":
        # 2.3 m above the floor mount, past a 0.6 m reach from any station: both refuse it.
        assert early["verdict"] == "no_fit" and late is False
    long = _team_early(scene, scene["bodies"]["long_arm"])
    if scene["late"]["team:" + scene["bodies"]["long_arm"]["embodiment_digest"]]:
        assert long["verdict"] in {"fit", "unknown"}


def test_a_body_whose_own_model_is_registered_is_never_refused_on_declared_reach(scene):
    # Its own arm, not its declared reach, chooses its station late (team_scene_packet._own_arm_stations).
    assert _team_early(scene, scene["bodies"]["short_arm"], model=True)["verdict"] != "no_fit"


def test_every_candidate_refused_early_is_refused_by_the_screening_plan_or_its_embodiment(scene):
    late_by_embodiment = scene["late"]
    catalog = {row["candidate_id"]: row for row in load_default_catalog()["candidates"]}
    for row in scene["early"]["candidates"]:
        if row["verdict"] not in {"no_fit", "unavailable"}:
            continue
        candidate = catalog[row["candidate_id"]]
        plan = _candidate_blockers(candidate, {"task_kind": scene["early"]["task_kind"]}, None, set())
        static = [code for code in plan if code not in {"runtime_candidate_binding_missing"}]
        assert row["causes"], row
        for cause in row["causes"]:
            if cause in ("catalog", "task_kind"):
                assert static and set(row["blockers"]) & set(static), row
            elif cause == "gpu_memory":
                assert candidate_gpu_blockers(candidate), row
            else:
                assert cause == "reach" and late_by_embodiment[row["embodiment"]] is False, row


@pytest.mark.parametrize("ceiling", [None, "0", "800", "860", "1200"])
def test_the_gpu_rule_never_refuses_early_what_screening_rents_under_its_own_ceiling(monkeypatch, ceiling):
    """Planning reads BLUEPRINT_VAST_MAX_COMPUTE_CAP in its own process; the precheck lifts it."""
    from blueprint_pipeline.beta_policy_screening_gpu import CEILING_LIFTED
    from blueprint_pipeline.vast_compute_capability import MAX_COMPUTE_CAP_ENV

    monkeypatch.delenv(MAX_COMPUTE_CAP_ENV, raising=False)
    candidates = [row for row in load_default_catalog()["candidates"]
                  if row["eligibility"]["enabled"] and row["runtime"] is not None]
    early = {row["candidate_id"]: candidate_gpu_blockers(row, max_compute_cap=CEILING_LIFTED) for row in candidates}
    if ceiling is not None:
        monkeypatch.setenv(MAX_COMPUTE_CAP_ENV, ceiling)
    # The early answer does not move with this process's environment.
    assert {row["candidate_id"]: candidate_gpu_blockers(row, max_compute_cap=CEILING_LIFTED)
            for row in candidates} == early
    for row in candidates:
        if early[row["candidate_id"]]:
            assert candidate_gpu_blockers(row), row["candidate_id"]  # screening, under this ceiling, refuses too


def test_a_fixed_arm_only_catalog_holds_exactly_where_the_late_franka_check_refuses(scene):
    value = robot_fit_precheck(preparation=scene["preparation"], collision_path=scene["collision"],
                               catalog=fixed_arm_catalog())
    if value["status"] == "no_fit":
        assert scene["late"][FRANKA_PROFILE] is False
    if scene["late"][FRANKA_PROFILE]:
        assert value["status"] in {"fit", "unknown"}
