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
from blueprint_pipeline import native_g1_spawn_clearance as spawn
from tests.test_native_g1_arena_robot_configuration import _asset
from tests.test_native_g1_spawn_clearance import g1_like_asset
from tests.test_team_scene_packet import _scene, _compile


def _unarticulated_asset(asset: Path) -> None:
    """The arena configuration fixture's joints and contact bodies, with no joint bodies or colliders."""
    from pxr import Usd, UsdGeom, UsdPhysics
    _asset(asset)
    stage = Usd.Stage.Open(str(asset))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/G1/head_link").GetPrim())
    stage.GetRootLayer().Save()


@pytest.fixture
def prepared_scene(tmp_path, monkeypatch, request):
    # Tiny declared synthetic robot (G1-shaped articulation with solid colliders); exercise all actual
    # USD/packet/stance/spawn algorithms with fixed test identity rather than downloading 38 MB in CI.
    # A test may name another builder by indirect parametrization.
    asset = tmp_path / "robot.usda"
    getattr(request, "param", g1_like_asset)(asset)
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


def _shared_sources(args, tmp_path, monkeypatch) -> dict:
    """The registration's shared source records. Test only the scene preparation/selection boundary;
    runtime source and live current-consent verification have their own tests and no live call."""
    path = tmp_path / "source-receipt.json"
    path.write_text("{}")
    monkeypatch.setattr("blueprint_pipeline.native_g1_publisher_source_stage.verify_g1_publisher_source", lambda root: {})
    monkeypatch.setattr("blueprint_pipeline.native_task_runtime_source_packet.verify_native_task_runtime_source_packet",
        lambda path: {"runtime_profile": "unitree_g1", "redistribution_permitted": True})
    monkeypatch.setattr("blueprint_pipeline.beta_policy_screening_worker.require_current_source", lambda *a, **k: {})
    # Only the CPU fixture asset check is substituted; no readiness is inferred.
    monkeypatch.setattr(g1, "ROBOT_USD_SHA256", args["robot_asset_record"]["sha256"])
    return {"publisher_source": _record(path), "runtime_source": _record(path)}


def _dp_spec(args, result, tmp_path, monkeypatch):
    scene = json.loads((Path(args["base_packet_record"]["path"]).parent / "native_task_arena_scene_plan.v1.json").read_text())
    return g1.candidate_execution_spec(candidate_id=g1.CANDIDATES[0],
        registration={"scene_packet": result["scene_packet"], **_shared_sources(args, tmp_path, monkeypatch)},
        scene_plan=scene, base_packet=args["base_packet_record"], source=args["source"],
        contract=args["task_success_contract"], site_id=args["site_id"])


def test_prepared_packet_reaches_dp_spec_without_hand_authored_scene_or_pi_assets(prepared_scene, tmp_path, monkeypatch):
    args, _, _ = prepared_scene
    result = prepare.prepare_native_g1_beta_packet(**args)
    spec = _dp_spec(args, result, tmp_path, monkeypatch)
    assert spec["candidate_id"] == "humanoidarena_dp_g1_dex3_sonic"
    assert spec["rights_authority"]["new_human_review_claimed"] is False
    assert spec["task_success_contract"] == args["task_success_contract"]


def test_the_chosen_stance_is_measured_whole_body_and_kept_beside_the_packet(prepared_scene):
    args, _, _ = prepared_scene
    result = prepare.prepare_native_g1_beta_packet(**args)
    root = Path(result["scene_packet"]["path"]).parent
    report = json.loads((root / spawn.FILENAME).read_text())
    plan = json.loads((root / "native_task_arena_scene_plan.v1.json").read_text())
    assert report["schema_version"] == "native_g1_spawn_clearance.v1" and report["status"] == "clear"
    assert report["scene_plan_digest"] == plan["plan_digest"] and report["simulator_executed"] is False
    assert report["base_pose_world"] == plan["robot"]["base_pose_world"]
    assert report["foot_support_tolerance_m"] == 0.02 and report["claim_boundary"] == spawn.CLAIM_BOUNDARY
    assert report["foot_support"]["bodies_found"] == ["left_ankle_roll_link", "right_ankle_roll_link"]
    assert report["stance_selection"]["fallback_used"] is False
    line = result["spawn_clearance"]
    assert line["status"] == "clear" and line["report"] == _record(root / spawn.FILENAME)
    assert line["disclosed_assumption"] == spawn.DISCLOSED_ASSUMPTION
    # The sealed request's assumptions keep the field set the door's registration pins.
    assert "spawn_clearance" not in result["assumptions"]
    # The same packet measured from its own files reads the same.
    assert spawn.packet_spawn_clearance(root)["status"] == "clear"


def _inside_the_table(monkeypatch, *, keep_searched: bool):
    """Put a stance inside the fixture's 2 x 8 m support first: its pelvis box crosses the table top."""
    real = prepare.g1_stance_search
    seen = {}

    def search(**kwargs):
        stances, chosen, reach = real(**kwargs)
        bad = {**deepcopy(stances[chosen]), "position_world_m": [3.5, -5.0, stances[chosen]["position_world_m"][2]]}
        bad["stance_digest"] = canonical_digest(bad, digest_field="stance_digest")
        seen.update(searched=stances[chosen])
        return ([bad, *stances] if keep_searched else [bad]), 0, reach

    monkeypatch.setattr(prepare, "g1_stance_search", search)
    return seen


def test_a_stance_whose_body_overlaps_the_room_falls_back_to_the_first_clearing_one(prepared_scene, monkeypatch):
    args, _, _ = prepared_scene
    seen = _inside_the_table(monkeypatch, keep_searched=True)
    result = prepare.prepare_native_g1_beta_packet(**args)
    root = Path(result["scene_packet"]["path"]).parent
    report = json.loads((root / spawn.FILENAME).read_text())
    assert report["status"] == "clear"
    assert result["assumptions"]["stance"] == seen["searched"]  # The clearing stance, the search's own pick.
    selection = report["stance_selection"]
    assert selection["fallback_used"] is True and selection["search_rank_chosen"] == 0
    assert selection["search_rank_used"] > 0 and selection["chosen_overlap"]["worst_bodies"][0]["body"] == "pelvis"
    assert result["assumptions"]["stance_reach"]["chosen_rank"] == selection["search_rank_used"]
    assert result["spawn_clearance"]["fallback_used"] is True


def test_with_no_clearing_stance_the_packet_is_written_with_overlap_and_never_funded(prepared_scene, tmp_path, monkeypatch):
    args, _, _ = prepared_scene
    _inside_the_table(monkeypatch, keep_searched=False)
    result = prepare.prepare_native_g1_beta_packet(**args)
    root, _, _ = verify_native_task_arena_packet(Path(result["scene_packet"]["path"]).parent)
    report = json.loads((root / spawn.FILENAME).read_text())
    assert report["status"] == "overlap" and result["spawn_clearance"]["status"] == "overlap"
    assert report["worst_bodies"][0]["body"] == "pelvis" and report["max_depth_m"] > 0.05
    assert report["clearing_push"]["distance_m"] > 0
    assert (root / prepare.FILENAME).is_file()
    with pytest.raises(ValueError, match="^g1_spawn_overlap$"):
        _dp_spec(args, result, tmp_path, monkeypatch)


@pytest.mark.parametrize("prepared_scene", [_unarticulated_asset], indirect=True)
def test_a_robot_usd_the_check_cannot_read_is_unmeasured_and_flagged_not_refused(prepared_scene, tmp_path, monkeypatch):
    args, _, _ = prepared_scene
    result = prepare.prepare_native_g1_beta_packet(**args)  # Still prepared: the packet and its receipt.
    root = Path(result["scene_packet"]["path"]).parent
    report = json.loads((root / spawn.FILENAME).read_text())
    assert report["status"] == "unmeasured" and report["reason"].startswith("scene_physics_")
    spec = _dp_spec(args, result, tmp_path, monkeypatch)  # Only a measured overlap refuses money.
    assert g1.setup_quality_flags(spec) == ["g1_spawn_clearance_unmeasured"]


def test_a_packet_without_its_spawn_report_is_flagged_until_measured_from_the_packet(prepared_scene, tmp_path, monkeypatch):
    args, _, _ = prepared_scene
    result = prepare.prepare_native_g1_beta_packet(**args)
    root = Path(result["scene_packet"]["path"]).parent
    original = json.loads((root / spawn.FILENAME).read_text())
    (root / spawn.FILENAME).unlink()  # A packet prepared before the check existed.
    assert g1.setup_quality_flags(_dp_spec(args, result, tmp_path, monkeypatch)) == ["g1_spawn_clearance_unmeasured"]
    measured = spawn.ensure_packet_spawn_clearance(root)
    assert measured["status"] == original["status"] == "clear"
    assert measured["scene_plan_digest"] == original["scene_plan_digest"]
    assert measured["stance_selection"]["source"] == "packet_robot_base_pose"
    assert g1.setup_quality_flags(_dp_spec(args, result, tmp_path, monkeypatch)) == []


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


@pytest.mark.parametrize("spawn_reading", ["overlap", "clear", "missing"])
def test_planning_funds_no_g1_row_whose_packet_spawns_inside_the_room(prepared_scene, tmp_path, monkeypatch,
                                                                       spawn_reading):
    from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog
    from blueprint_pipeline.beta_policy_screening_capabilities import build_planning_inputs
    from blueprint_pipeline.beta_policy_screening_launch import _write
    from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan
    from blueprint_pipeline.robot_free_probe_gate import (
        RECEIPT_FILENAME, REQUIRED_CHECKS, RIGID_RECEIPT_SCHEMA, task_success_contract_binding)
    from blueprint_pipeline.team_candidate_quick10_registry import scene_facts

    args, request, _ = prepared_scene
    if spawn_reading == "overlap":
        _inside_the_table(monkeypatch, keep_searched=False)
    result = prepare.prepare_native_g1_beta_packet(**args)
    if spawn_reading == "missing":
        (Path(result["scene_packet"]["path"]).parent / spawn.FILENAME).unlink()
    dp = g1.CANDIDATES[0]
    [inputs_path] = args["output_root"].glob("team_scene_inputs.v1.json")
    profile = {"capture_root": str(tmp_path / "capture"), "scene_id": args["site_id"], "task_id": request["task_id"],
               "team_scene_inputs_path": str(inputs_path), "robot_free_probe_root": str(tmp_path / "probe"),
               "open_loop_horizon": 8, "max_hourly_rate_usd": 0.8,
               "screening_candidates": {dp: {"native_g1": {"scene_packet": result["scene_packet"],
                                                           **_shared_sources(args, tmp_path, monkeypatch)}}}}
    facts = scene_facts(profile)
    task = facts["scene"]["task_spec"]
    # Hermetic admission receipt; this test establishes no physical or GPU claim.
    probe = {"schema_version": RIGID_RECEIPT_SCHEMA, "status": "passed", "learned_policy_evaluation_admitted": True,
             "configured_scene_revision_digest": facts["inputs"]["scene_revision_digest"],
             "task_spec_digest": canonical_digest(task),
             "task_success_contract_digest": task_success_contract_binding(task["task_kind"],
                                                                           facts["inputs"]["task_success_contract"]),
             "checks": dict.fromkeys(REQUIRED_CHECKS["rigid_pick_place"], True),
             "proof_boundary": {"robot_embodiment_used": False}, "receipt_digest": ""}
    probe["receipt_digest"] = canonical_digest(probe, digest_field="receipt_digest")
    _write(Path(profile["robot_free_probe_root"]) / RECEIPT_FILENAME, probe)
    catalog = load_default_catalog()
    planning = build_planning_inputs(catalog=catalog, profile=profile, capture_root=Path(profile["capture_root"]),
                                     source=args["source"], inputs_dir=tmp_path / "screening-inputs")
    capabilities = planning["capabilities"]
    plan = build_screening_plan(catalog, planning["scene"], capabilities=capabilities,
                                cost_quotes=planning["cost_quotes"], campaign_id="g1-spawn")
    row = next(row for row in plan["candidates"] if row["candidate_id"] == dp)
    if spawn_reading == "overlap":  # The only spawn reason that refuses money.
        assert capabilities["candidate_blockers"][dp] == ["g1_spawn_overlap"]
        assert dp not in capabilities["candidate_bindings"] and dp not in planning["cost_quotes"]
        assert row["status"] == "unavailable" and "g1_spawn_overlap" in row["blockers"] and "cost_quote" not in row
        return
    # Clear, or not measured: bound, quoted and selected; an unmeasured spawn travels as a flag.
    assert dp not in capabilities["candidate_blockers"] and dp in planning["cost_quotes"]
    assert row["status"] == "selected"
    flags = row["execution_spec"].get("setup_quality_flags")
    assert flags == (["g1_spawn_clearance_unmeasured"] if spawn_reading == "missing" else None)
