"""A website navigation job's scene half: the base G1 packet at the compiled start and its sealed record.

Part C of docs/G1_NAVIGATION_FROM_COMPILED_SPEC.md. Real OpenUSD packet compilation with a tiny
G1-shaped robot (``g1_like_asset``) under a fixed test identity, a one-collider room and a captured
appearance; no simulator, provider, network or money.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

import numpy
import pytest

from blueprint_pipeline import native_g1_arena_robot_configuration as robot_config
from blueprint_pipeline.beta_policy_screening_files import _record
from blueprint_pipeline.beta_policy_screening_g1_prepare import HEAD_CAMERA
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_g1_navigation_goal import (
    NAVIGATION_CRITERION,
    compose_g1_navigation_goal_authority,
    navigation_task_success_contract,
    seal_g1_navigation_goal_authority,
    validate_g1_navigation_goal_authority,
    validate_navigation_task_success_contract,
)
from blueprint_pipeline.native_task_arena_bundle import verify_native_task_arena_packet
from blueprint_pipeline.scene_placement.robot_profile import UNITREE_G1_PROFILE
from blueprint_pipeline.website_navigation_scene_assets import SCHEMA as ASSETS_SCHEMA
from blueprint_pipeline.website_navigation_scene_inputs import (
    SOURCE_UNBOUND,
    WebsiteNavigationSceneInputsError,
    read_navigation_scene_inputs,
    screening_source,
    write_navigation_scene_inputs,
)
from tests.test_native_g1_navigation_goal import _authority_plan
from tests.test_native_g1_spawn_clearance import g1_like_asset
from tests.test_native_task_appearance_frame_alignment import write_appearance_usdz
from tests.test_native_task_navigation_scene import _navigation_task, _room_usd
from tests.test_website_navigation_packet import _context, _spec


def _scene_assets(directory: Path) -> dict:
    """The A2 record's shape (``navigation_scene_assets``) over a one-collider room and an appearance."""
    directory.mkdir(parents=True, exist_ok=True)
    collision = _room_usd(directory / "normalized_scene.usda")
    rng = numpy.random.default_rng(20261107)
    positions = numpy.stack(
        [rng.uniform(-1.0, 4.0, 512), rng.uniform(-2.0, 2.0, 512), rng.uniform(0.0, 2.5, 512)], axis=1
    ).astype(numpy.float32)
    appearance = write_appearance_usdz(directory / "background.usdz", positions)

    def row(path: Path) -> dict:
        record = _record(path)
        return {"path": record["path"], "sha256": record["sha256"], "size_bytes": record["size_bytes"],
                "receipt_digest": "sha256:" + "e" * 64}

    value = {"schema_version": ASSETS_SCHEMA, "claim_ceiling": "development_only",
             "frame": {"declared_up_axis": "Y", "declared_meters_per_unit": 1.0, "physical_scale_measured": False},
             "frame_digest": "sha256:" + "f" * 64, "appearance": row(appearance), "collision": row(collision),
             "source": {"splat_digest": "sha256:" + "1" * 64, "collision_mesh_digest": "sha256:" + "2" * 64},
             "paid_construction_performed": False, "scene_edited": False, "physical_scale_measured": False}
    value["digest"] = canonical_digest(value, digest_field="digest")
    return value


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch) -> dict:
    # A tiny G1-shaped articulation under a fixed test identity instead of the 38 MB pinned USD.
    (tmp_path / "robot").mkdir()
    asset = g1_like_asset(tmp_path / "robot" / "g1.usda")
    monkeypatch.setattr(robot_config, "G1_USD_SHA256", _record(asset)["sha256"][7:])
    monkeypatch.setattr(robot_config, "G1_USD_SIZE_BYTES", asset.stat().st_size)
    return {"navigation_task_spec": _navigation_task(), "scene_assets": _scene_assets(tmp_path / "navigation_scene"),
            "robot_record": _record(asset), "output_root": tmp_path / "navigation_scene_inputs"}


def _packet_documents(record: dict) -> tuple[Path, dict, dict]:
    root = Path(record["packet_root"])
    plan = json.loads((root / "native_task_arena_scene_plan.v1.json").read_text())
    request = json.loads((root / "native_task_arena_packet_request.v1.json").read_text())
    return root, plan, request


def _refused(path: Path, code: str) -> None:
    with pytest.raises(WebsiteNavigationSceneInputsError) as excinfo:
        read_navigation_scene_inputs(path)
    assert str(excinfo.value) == code


def _resealed(record: dict, tmp_path: Path, name: str, **changes) -> Path:
    value = {**copy.deepcopy(record), **changes}
    value["scene_inputs_digest"] = canonical_digest(value, digest_field="scene_inputs_digest")
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return path


# --- the packet -------------------------------------------------------------------------------


def test_the_packet_compiles_with_the_g1_standing_at_the_compiled_start(inputs) -> None:
    result = write_navigation_scene_inputs(**inputs)
    record = result["record"]
    task = inputs["navigation_task_spec"]
    root, plan, request = _packet_documents(record)
    _, receipt, _ = verify_native_task_arena_packet(root)

    assert plan["task_kind"] == "goal_region_navigation" and plan["robot"]["robot_id"] == "unitree_g1"
    assert [row["semantic_role"] for row in plan["objects"]] == ["scene_collision", "scene_appearance"]
    assert all(row["pose_world"]["position_world_m"] == [0.0, 0.0, 0.0] for row in plan["objects"])
    # x, y as compiled; the pelvis the G1 profile's height above the compiled floor; yaw faces the goal.
    x, y, z = plan["robot"]["base_pose_world"]["position_world_m"]
    start, goal = task["start_pose_world"], task["g1_navigation_goal"]["center_world_m"]
    assert [x, y] == start[:2] and z == pytest.approx(start[2] + UNITREE_G1_PROFILE.pelvis_height_m)
    qx, qy, qz, qw = plan["robot"]["base_pose_world"]["orientation_xyzw"]
    assert qx == qy == 0.0
    assert 2 * math.atan2(qz, qw) == pytest.approx(math.atan2(goal[1] - start[1], goal[0] - start[0]))
    assert record["robot"]["start_yaw_rad"] == pytest.approx(math.atan2(goal[1] - start[1], goal[0] - start[0]))
    # The G1 preparation's head camera and one review-only world overview, which the G1 contract requires.
    cameras = {row["role"]: row for row in plan["cameras"]}
    assert set(cameras) == {"head", "overview"}
    assert cameras["head"]["frame_from_camera_matrix"] == HEAD_CAMERA["frame_from_camera_matrix"]
    assert cameras["head"]["policy_input"] is True and cameras["overview"]["policy_input"] is False
    assert cameras["overview"]["pose_frame"] == "world" and cameras["overview"]["scoring_input"] is False
    # The A1 spec rides unchanged under its own seal, with the success contract beside it.
    assert {key: value for key, value in plan["task_spec"].items()
            if key not in {"task_success_contract", "task_success_contract_digest"}} == task
    assert plan["task_spec"]["task_success_contract"] == navigation_task_success_contract(task)
    assert request["scenario"]["context_document"]["factor_records"] == []
    assert "appearance_variant" not in request

    assert record["scene_revision_digest"] == record["plan_digest"] == plan["plan_digest"]
    assert record["claim_ceiling"] == "development_only" and record["task_kind"] == "goal_region_navigation"
    assert record["base_native_packet"] == record["construction_result"] == _record(
        root / "native_task_arena_packet_receipt.v1.json")
    assert record["packet_receipt_digest"] == receipt["receipt_digest"]
    assert record["runtime_contract_digest"] == receipt["runtime_contract_digest"]
    assert record["navigation_task_spec_digest"] == task["task_spec_digest"]
    assert record["scene_assets_digest"] == inputs["scene_assets"]["digest"]
    assert record["task_success_contract_digest"] == record["task_success_contract"]["contract_digest"]
    assert record["scene_inputs_digest"] == canonical_digest(record, digest_field="scene_inputs_digest")
    assert record["source"] is None and record["source_gap"]["code"] == SOURCE_UNBOUND
    assert any(item.startswith("g1_pelvis_0.79m_above_compiled_floor") for item in record["disclosed_assumptions"])
    assert "overview_camera_added_because_the_g1_contract_requires_one_review_only_not_visibility_checked" in (
        record["disclosed_assumptions"])
    assert record["native_application_claimed"] is False and record["paid_authority_granted"] is False
    assert read_navigation_scene_inputs(result["path"]) == record
    # The machine-composed goal authority (owner decision 4) binds this exact plan.
    authority = compose_g1_navigation_goal_authority(plan=plan, task_context=_context(_spec()))
    assert validate_g1_navigation_goal_authority(authority, plan=plan) == authority


def test_a_second_call_returns_the_same_record_without_rewriting(inputs) -> None:
    first = write_navigation_scene_inputs(**inputs)
    root = Path(first["path"]).parent
    before = {path: path.stat().st_mtime_ns for path in root.rglob("*")}
    second = write_navigation_scene_inputs(**inputs)
    assert second == first
    assert {path: path.stat().st_mtime_ns for path in root.rglob("*")} == before


def test_the_writer_refuses_another_kind_another_robot_and_a_changed_scene_file(inputs, tmp_path) -> None:
    rigid = {**inputs["navigation_task_spec"], "task_kind": "rigid_pick_place"}
    with pytest.raises(WebsiteNavigationSceneInputsError, match="^website_navigation_scene_inputs_task_kind_invalid$"):
        write_navigation_scene_inputs(**{**inputs, "navigation_task_spec": rigid})
    unsealed = {**inputs["navigation_task_spec"], "maximum_episode_seconds": 1.0}
    with pytest.raises(WebsiteNavigationSceneInputsError, match="^website_navigation_scene_inputs_task_spec_invalid$"):
        write_navigation_scene_inputs(**{**inputs, "navigation_task_spec": unsealed})
    other = tmp_path / "other.usda"
    other.write_text("#usda 1.0\n")
    with pytest.raises(WebsiteNavigationSceneInputsError,
                       match="^website_navigation_scene_inputs_robot_asset_mismatch$"):
        write_navigation_scene_inputs(**{**inputs, "robot_record": _record(other)})
    Path(inputs["scene_assets"]["collision"]["path"]).write_text("#usda 1.0\n")
    with pytest.raises(WebsiteNavigationSceneInputsError,
                       match="^website_navigation_scene_inputs_scene_asset_changed$"):
        write_navigation_scene_inputs(**inputs)
    assert not inputs["output_root"].exists()


# --- the success contract ---------------------------------------------------------------------


def test_the_success_contract_validates_and_tampering_refuses() -> None:
    task = _navigation_task()
    contract = navigation_task_success_contract(task)
    assert validate_navigation_task_success_contract(contract, task_spec=task) == contract
    assert contract["criterion"] == NAVIGATION_CRITERION == "root_xy_goal_arrival_and_terminal_hold"
    assert contract["compiled_task_spec_digest"] == task["compiled_task_spec_digest"]
    assert contract["obstacle_contact"] == {"recorded_per_step": True, "part_of_success": False,
                                            "compiled_predicate_ids": ["no_obstacle_contact"]}
    assert contract["claim_ceiling"] == "development_only" and contract["physical_outcome_claimed"] is False

    def refused(value, spec=task) -> None:
        with pytest.raises(ValueError, match="^g1_navigation_task_success_contract_invalid$"):
            validate_navigation_task_success_contract(value, task_spec=spec)

    scored_contact = copy.deepcopy(contract)
    scored_contact["obstacle_contact"]["part_of_success"] = True
    scored_contact["contract_digest"] = canonical_digest(scored_contact, digest_field="contract_digest")
    refused(scored_contact)
    refused({**contract, "criterion": "root_xy_inside_visible_goal"})
    refused({**contract, "contract_digest": "sha256:" + "0" * 64})
    # A goal moved under the same contract, a tampered compiled spec, or another kind of task.
    moved = copy.deepcopy(task)
    moved["g1_navigation_goal"]["center_world_m"][0] += 0.5
    moved["g1_navigation_goal"]["visible_target_marker"]["surface_position_world_m"][0] += 0.5
    refused(contract, moved)
    tampered = copy.deepcopy(task)
    tampered["compiled_task_spec"]["site_id"] = "site-other"
    refused(contract, tampered)
    refused(contract, {**task, "task_kind": "rigid_pick_place"})


# --- the reader -------------------------------------------------------------------------------


def test_the_reader_refuses_a_changed_packet_file(inputs) -> None:
    result = write_navigation_scene_inputs(**inputs)
    root = Path(result["record"]["packet_root"])
    plan = root / "native_task_arena_scene_plan.v1.json"
    original = plan.read_bytes()
    os.chmod(plan, 0o644)
    plan.write_bytes(original + b"\n")
    _refused(Path(result["path"]), "website_navigation_scene_inputs_packet_changed")
    plan.write_bytes(original)
    assert read_navigation_scene_inputs(result["path"]) == result["record"]
    appearance = next((root / "assets").glob("scene_appearance.*"))
    os.chmod(appearance, 0o644)
    with appearance.open("ab") as stream:
        stream.write(b"\0")
    _refused(Path(result["path"]), "website_navigation_scene_inputs_packet_changed")


def test_the_reader_refuses_a_changed_seal_another_kind_and_a_moved_binding(inputs, tmp_path) -> None:
    result = write_navigation_scene_inputs(**inputs)
    record = result["record"]
    changed = {**record, "claim_ceiling": "diagnostic_policy_execution"}
    unsealed = tmp_path / "unsealed.json"
    unsealed.write_text(json.dumps({**record, "scene_revision_digest": "sha256:" + "9" * 64}))
    _refused(unsealed, "website_navigation_scene_inputs_seal_invalid")
    _refused(_resealed(changed, tmp_path, "ceiling.json"), "website_navigation_scene_inputs_seal_invalid")
    _refused(_resealed(record, tmp_path, "rigid.json", task_kind="rigid_pick_place"),
             "website_navigation_scene_inputs_task_kind_invalid")
    _refused(_resealed(record, tmp_path, "revision.json", scene_revision_digest="sha256:" + "9" * 64),
             "website_navigation_scene_inputs_binding_changed")
    moved = copy.deepcopy(record)
    moved["robot"]["base_pose_world"]["position_world_m"][0] += 0.25
    _refused(_resealed(moved, tmp_path, "moved.json"), "website_navigation_scene_inputs_binding_changed")
    _refused(_resealed(record, tmp_path, "assets.json", scene_assets={**record["scene_assets"],
                                                                     "collision_sha256": "sha256:" + "3" * 64}),
             "website_navigation_scene_inputs_binding_changed")
    _refused(tmp_path / "missing.json", "website_navigation_scene_inputs_unreadable")


def test_the_screening_source_refuses_until_a_source_identity_is_decided(inputs) -> None:
    record = write_navigation_scene_inputs(**inputs)["record"]
    assert set(record["source_gap"]["missing"]) == {"capture_id", "scene_intent_id", "scene_intent_digest"}
    with pytest.raises(WebsiteNavigationSceneInputsError, match=f"^{SOURCE_UNBOUND}$"):
        screening_source(record)


# --- the rigid path ---------------------------------------------------------------------------


def test_the_rigid_goal_authority_still_seals_the_digest_it_did_before_this_module() -> None:
    plan = _authority_plan()
    authority = seal_g1_navigation_goal_authority(plan=plan, confirmed_by_team_id="team-a", human_reviewer="owner-a")
    # Computed on main before the navigation success contract was added to native_g1_navigation_goal.
    assert plan["plan_digest"] == "sha256:f83d648ffcde085496a70e6a45517d99781436841449146f3e666d7faba5598d"
    assert authority["authority_digest"] == (
        "sha256:23c0f49456532e2e2dfa056f69e412cbe21774ac9a6a9f585c9ab11f97d20cc7")
    with pytest.raises(ValueError, match="g1_navigation_task_success_contract_invalid"):
        navigation_task_success_contract(plan["task_spec"])
