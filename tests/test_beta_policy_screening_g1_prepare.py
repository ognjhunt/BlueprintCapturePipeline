"""Real CPU scene compilation and OpenUSD G1 packet preparation; no simulator."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal

import pytest

from blueprint_pipeline import beta_policy_screening_g1 as g1
from blueprint_pipeline import beta_policy_screening_g1_prepare as prepare
from blueprint_pipeline import native_g1_arena_robot_configuration as robot_config
from blueprint_pipeline.beta_policy_screening_launch import _record
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_bundle import verify_native_task_arena_packet
from tests.test_native_g1_arena_robot_configuration import _asset
from tests.test_team_scene_packet import _scene, _compile


@pytest.fixture
def prepared_scene(tmp_path, monkeypatch):
    from pxr import Usd, UsdGeom, UsdPhysics
    # Tiny declared synthetic robot; exercise all actual USD/packet/stance
    # algorithms with fixed test identity rather than downloading 38 MB in CI.
    asset = tmp_path / "robot.usda"
    _asset(asset)
    stage = Usd.Stage.Open(str(asset))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/G1/head_link").GetPrim())
    stage.GetRootLayer().Save()
    stage = None
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    monkeypatch.setattr(robot_config, "G1_USD_SHA256", digest)
    monkeypatch.setattr(robot_config, "G1_USD_SIZE_BYTES", asset.stat().st_size)
    scene = _scene(tmp_path, strategy="pick_and_place")
    compiled, request = _compile(tmp_path, scene)
    inputs = json.loads(Path(compiled["path"]).read_text())
    source = {"capture_id": "capture-fixture", "task_id": request["task_id"], "scene_intent_id": "intent-fixture",
        "task_context_digest": "sha256:" + "a" * 64, "scene_intent_digest": "sha256:" + "b" * 64,
        "scene_revision_digest": inputs["scene_revision_digest"]}
    args = {"base_packet_record": inputs["base_native_packet"], "source": source, "site_id": request["scene_id"],
        "output_root": Path(compiled["path"]).parent, "robot_asset_record": _record(asset),
        "task_success_contract": inputs["task_success_contract"]}
    return args, request, inputs


def test_real_compiled_task_prepares_native_g1_with_frozen_task_and_visible_assumptions(prepared_scene):
    args, original, inputs = prepared_scene
    base_root = Path(args["base_packet_record"]["path"]).parent
    original_files = {p.relative_to(base_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in base_root.rglob("*") if p.is_file()}
    result = prepare.prepare_native_g1_beta_packet(**args)
    root, receipt, _ = verify_native_task_arena_packet(Path(result["scene_packet"]["path"]).parent)
    request = json.loads((root / "native_task_arena_packet_request.v1.json").read_text())
    plan = json.loads((root / "native_task_arena_scene_plan.v1.json").read_text())
    assert result["status"] == "prepared_not_executed"
    assert result["candidate_policy_queried"] is False and result["provider_allocations_performed"] == 0
    assert result["current_consent_revalidated"] is False and result["new_human_review_claimed"] is False
    assert request["scenario"] == original["scenario"]  # No invented ADP context.
    assert request["task_spec"]["task_success_contract"] == inputs["task_success_contract"]
    for key, value in original["task_spec"].items():
        if key not in {"control_frequency_hz", "maximum_action_steps", "maximum_episode_seconds"}:
            assert request["task_spec"][key] == value
    assert request["task_spec"]["maximum_episode_seconds"] <= original["task_spec"]["maximum_episode_seconds"]
    assert request["task_spec"]["maximum_action_steps"] / 50 <= original["task_spec"]["maximum_action_steps"] / original["task_spec"]["control_frequency_hz"]
    assert request["physics_frequency_hz"] == 200
    assert request["cameras"][0] == prepare.HEAD_CAMERA
    assert request["cameras"][1] == next(c for c in original["cameras"] if c["role"] == "overview")
    assert len(plan["robot"]["joint_reset_positions_rad"]) == 43
    assumptions = result["assumptions"]
    assert assumptions == request["g1_scene_derivation"]["assumptions"]
    assert assumptions["stance_is_geometric_estimate"] is True
    assert assumptions["measured_robot_calibration_claimed"] is False
    # The actual fixture has an 8.1m support; retaining large collision bounds
    # prevents a seemingly clear G1 footprint from being placed inside it.
    x, y, z = assumptions["stance"]["position_world_m"]
    assert not (2.5 <= x <= 4.5 and -9.5 <= y <= -1.4)
    assert z == pytest.approx(0.79) and assumptions["large_collision_bounds_retained"] is True
    # The stance is the closest clear one the G1's nominal arm span covers the task from, else the closest.
    reach = assumptions["stance_reach"]
    assert reach["rule"] == "closest_clear_stance_within_nominal_arm_span" and reach["clear_stances_measured"] > 1
    assert reach["shoulder"] == assumptions["task_hand"] == "right"
    assert reach["within_nominal_arm_span"] is (reach["interpretation"] == "nominal_static_reach_candidate")
    assert request["robot_base_pose_world"]["position_world_m"] == assumptions["stance"]["position_world_m"]
    assert root.name == canonical_digest(args["source"])[7:]
    assert len(result["candidate_reasons"]) == 2 and all(k.endswith("vision_navi") for k in result["candidate_reasons"])
    assert not any("navigation_goal" in k for k in request["task_spec"])
    base_receipt = json.loads(Path(args["base_packet_record"]["path"]).read_text())
    assert [(r["semantic_role"], r["staged_sha256"]) for r in receipt["source_bindings"]] == [
        (r["semantic_role"], r["staged_sha256"]) for r in base_receipt["source_bindings"]]
    assert original_files == {p.relative_to(base_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in base_root.rglob("*") if p.is_file()}
    assert prepare.prepare_native_g1_beta_packet(**args) == result


def test_prepared_packet_reaches_dp_spec_without_hand_authored_scene_or_pi_assets(prepared_scene, tmp_path, monkeypatch):
    args, _, _ = prepared_scene
    result = prepare.prepare_native_g1_beta_packet(**args)
    path = tmp_path / "source-receipt.json"
    path.write_text("{}")
    # Test only the scene preparation/selection boundary; runtime source and
    # live current-consent verification have their own tests and no live call.
    monkeypatch.setattr("blueprint_pipeline.native_g1_publisher_source_stage.verify_g1_publisher_source", lambda root: {})
    monkeypatch.setattr("blueprint_pipeline.native_task_runtime_source_packet.verify_native_task_runtime_source_packet",
        lambda path: {"runtime_profile": "unitree_g1", "redistribution_permitted": True})
    monkeypatch.setattr("blueprint_pipeline.beta_policy_screening_worker.require_current_source", lambda *a, **k: {})
    # Only the CPU fixture asset check is substituted; no readiness is inferred.
    monkeypatch.setattr(g1, "ROBOT_USD_SHA256", args["robot_asset_record"]["sha256"])
    scene = json.loads((Path(args["base_packet_record"]["path"]).parent / "native_task_arena_scene_plan.v1.json").read_text())
    spec = g1.candidate_execution_spec(candidate_id=g1.CANDIDATES[0],
        registration={"scene_packet": result["scene_packet"], "publisher_source": _record(path), "runtime_source": _record(path)},
        scene_plan=scene, base_packet=args["base_packet_record"], source=args["source"],
        contract=args["task_success_contract"], site_id=args["site_id"])
    assert spec["candidate_id"] == "humanoidarena_dp_g1_dex3_sonic"
    assert spec["rights_authority"]["new_human_review_claimed"] is False
    assert spec["task_success_contract"] == args["task_success_contract"]


def test_corrected_source_context_uses_distinct_immutable_packet(prepared_scene):
    args, _, _ = prepared_scene
    first = prepare.prepare_native_g1_beta_packet(**args)
    changed = deepcopy(args)
    changed["source"]["task_context_digest"] = "sha256:" + "c" * 64
    second = prepare.prepare_native_g1_beta_packet(**changed)
    assert first["scene_packet"]["path"] != second["scene_packet"]["path"]
    assert prepare.prepare_native_g1_beta_packet(**args) == first


def test_wrong_current_task_or_changed_source_bytes_are_rejected(prepared_scene):
    args, _, _ = prepared_scene
    changed = deepcopy(args)
    changed["source"]["task_id"] = "different-task"
    with pytest.raises(ValueError, match="task_binding_mismatch"):
        prepare.prepare_native_g1_beta_packet(**changed)
    Path(args["robot_asset_record"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="input_bytes_changed"):
        prepare.prepare_native_g1_beta_packet(**args)


def test_interrupted_evidence_copy_never_publishes_partial_bytes(tmp_path, monkeypatch):
    source, destination = tmp_path / "source.usd", tmp_path / "evidence/robot.usd"
    source.write_bytes(b"complete verified robot fixture")
    expected = _record(source)
    real_copy = prepare.shutil.copyfile

    def interrupted(src, dst):
        Path(dst).write_bytes(Path(src).read_bytes()[:3])
        raise OSError("injected interrupted copy")

    monkeypatch.setattr(prepare.shutil, "copyfile", interrupted)
    with pytest.raises(OSError, match="injected interrupted copy"):
        prepare._copy(source, destination, sha256=expected["sha256"], size_bytes=expected["size_bytes"])
    assert not destination.exists()
    monkeypatch.setattr(prepare.shutil, "copyfile", real_copy)
    prepare._copy(source, destination, sha256=expected["sha256"], size_bytes=expected["size_bytes"])
    assert destination.read_bytes() == source.read_bytes()
    inode = destination.stat().st_ino
    monkeypatch.setattr(prepare.shutil, "copyfile", lambda *a: pytest.fail("valid evidence must not be recopied"))
    prepare._copy(source, destination, sha256=expected["sha256"], size_bytes=expected["size_bytes"])
    assert destination.stat().st_ino == inode


def test_legacy_partial_copy_is_preserved_before_exact_retry(tmp_path):
    source, destination = tmp_path / "source.usd", tmp_path / "robot.usd"
    source.write_bytes(b"complete verified source")
    destination.write_bytes(b"com")
    expected = _record(source)
    prepare._copy(source, destination, sha256=expected["sha256"], size_bytes=expected["size_bytes"])
    assert destination.read_bytes() == source.read_bytes()
    [retained] = list(tmp_path.glob(".robot.usd.incomplete-*/robot.usd"))
    assert retained.read_bytes() == b"com"


def test_killed_packet_compiler_releases_lock_and_retries_without_losing_partial_evidence(prepared_scene):
    args, _, _ = prepared_scene
    base = json.loads(Path(args["base_packet_record"]["path"]).read_text())
    final = args["output_root"] / "native-g1" / base["receipt_digest"][7:] / canonical_digest(args["source"])[7:]

    def killed_compiler():
        from blueprint_pipeline import native_task_arena_packet

        def interrupted(*, output_dir, **kwargs):
            output_dir.mkdir()
            (output_dir / "partial-build.bin").write_bytes(b"retained incomplete CPU fixture")
            os.kill(os.getpid(), signal.SIGKILL)

        native_task_arena_packet.materialize_native_task_arena_packet = interrupted
        prepare.prepare_native_g1_beta_packet(**args)

    child = multiprocessing.get_context("fork").Process(target=killed_compiler)
    child.start()
    try:
        child.join(timeout=20)
        assert child.exitcode == -signal.SIGKILL
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=5)
    assert (final / "partial-build.bin").is_file()
    assert not (final / "native_task_arena_packet_receipt.v1.json").exists()
    result = prepare.prepare_native_g1_beta_packet(**args)
    assert Path(result["scene_packet"]["path"]).parent == final
    verify_native_task_arena_packet(final)
    [retained] = list(final.parent.glob("." + final.name + ".incomplete-*/" + final.name + "/partial-build.bin"))
    assert retained.read_bytes() == b"retained incomplete CPU fixture"
    assert not (final / "partial-build.bin").exists()


def test_retained_packet_with_receipt_is_never_replaced_on_validation_failure(prepared_scene):
    args, _, _ = prepared_scene
    result = prepare.prepare_native_g1_beta_packet(**args)
    final = Path(result["scene_packet"]["path"]).parent
    plan = final / "native_task_arena_scene_plan.v1.json"
    inode = final.stat().st_ino
    receipt_bytes = Path(result["scene_packet"]["path"]).read_bytes()
    plan.write_text("corrupted retained packet")
    with pytest.raises(ValueError):
        prepare.prepare_native_g1_beta_packet(**args)
    assert final.stat().st_ino == inode
    assert plan.read_text() == "corrupted retained packet"
    assert Path(result["scene_packet"]["path"]).read_bytes() == receipt_bytes
    assert not list(final.parent.glob("." + final.name + ".incomplete-*"))
