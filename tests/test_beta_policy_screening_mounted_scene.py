"""A real compiled installed articulation reaches the actual sealed preflight."""
from __future__ import annotations

import json
from pathlib import Path
import zipfile

from blueprint_pipeline import beta_policy_screening_launch as launch
from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog
from blueprint_pipeline.beta_policy_screening_capabilities import build_planning_inputs
from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.policy_canary_scene_resolution import resolved_scene_plan
from blueprint_pipeline.robot_free_probe_gate import RECEIPT_FILENAME, RECEIPT_SCHEMA, REQUIRED_CHECKS
from blueprint_pipeline.team_candidate_quick10_registry import scene_facts
from tests.test_native_task_arena_bundle import _runtime_source_packet
from tests.test_team_scene_packet import _articulated_scene, _compile


def test_installed_articulated_scene_plans_and_passes_real_sealed_bundle_preflight(tmp_path):
    from tests.test_task_evaluation_articulated_open_close_native_adapter import DEFINITION

    scene = _articulated_scene(tmp_path)
    definition_path = Path(scene["documents"][DEFINITION])
    definition = json.loads(definition_path.read_text())
    definition["assembly_mounting"] = {
        "mode": "fixed_in_scene", "source_frame_ids": ["fixture-source-view"],
        "reason": "Installed test cabinet", "body_envelope_excludes_environment": True,
    }
    definition_path = tmp_path / "mounted-definition.json"
    definition_path.write_text(json.dumps(definition, sort_keys=True) + "\n")
    scene["documents"][DEFINITION] = definition_path
    record = launch._record(definition_path)
    scene["revision"]["task_template"]["definition"].update(digest=record["sha256"], size_bytes=record["size_bytes"])
    scene["revision"]["revision_digest"] = canonical_digest(scene["revision"], digest_field="revision_digest")
    scene["runtime_source_receipt_path"] = _runtime_source_packet(tmp_path / "runtime")
    compiled, _ = _compile(tmp_path, scene)
    profile = {"capture_root": str(tmp_path / "capture"), "scene_id": "site-mounted-fixture",
               "task_id": compiled["task_id"], "team_scene_inputs_path": compiled["path"],
               "robot_free_probe_root": str(tmp_path / "probe"), "open_loop_horizon": 8,
               "max_hourly_rate_usd": 0.8}
    facts = scene_facts(profile)
    native_path = Path(facts["packet_dir"]) / "native_task_arena_scene_plan.v1.json"
    native = json.loads(native_path.read_text())
    subject = next(row for row in native["objects"] if row.get("task_subject"))
    assert native["task_kind"] == "articulated_open_close" and subject["object_type"] == "ARTICULATION"
    assert subject["articulation_adaptation"]["scene_mounting"]["mode"] == "fixed_in_scene"
    # Hermetic admission receipt; this test establishes no physical or GPU claim.
    probe = {"schema_version": RECEIPT_SCHEMA, "status": "passed",
             "configured_scene_revision_digest": facts["inputs"]["scene_revision_digest"],
             "task_spec_digest": canonical_digest(facts["scene"]["task_spec"]),
             "learned_policy_evaluation_admitted": True,
             "checks": dict.fromkeys(REQUIRED_CHECKS["articulated_open_close"], True),
             "proof_boundary": {"robot_embodiment_used": False}, "receipt_digest": ""}
    probe["receipt_digest"] = canonical_digest(probe, digest_field="receipt_digest")
    launch._write(Path(profile["robot_free_probe_root"]) / RECEIPT_FILENAME, probe)
    source = {"capture_id": "mounted-capture", "task_id": profile["task_id"], "scene_intent_id": "mounted-intent",
              "scene_intent_digest": "sha256:" + "a" * 64, "task_context_digest": "sha256:" + "b" * 64,
              "scene_revision_digest": facts["inputs"]["scene_revision_digest"]}
    catalog = load_default_catalog()
    planning = build_planning_inputs(catalog=catalog, profile=profile, capture_root=Path(profile["capture_root"]),
                                    source=source, inputs_dir=tmp_path / "screening-inputs")
    plan = build_screening_plan(catalog, planning["scene"], capabilities=planning["capabilities"],
                               cost_quotes=planning["cost_quotes"], campaign_id="mounted-campaign")
    assert planning["scene"]["native_scene_plan"] == native
    gaps = []
    for cell in plan["scenario_cells"]:
        parameters = cell["resolved_scenario"]["parameters"]
        assert not any(parameters.get(key) for key in
                       ("object_start_x_delta_m", "object_start_y_delta_m", "object_yaw_delta_degrees"))
        gaps.extend(cell["resolved_scenario"].get("runtime_coverage_gaps", []))
        resolved = resolved_scene_plan(native, cell, task_success_contract=facts["inputs"]["task_success_contract"])
        actual = next(row for row in resolved["objects"] if row.get("task_subject"))
        assert actual["pose_world"] == subject["pose_world"]
        assert actual["reset_state"]["root_pose_world"] == subject["reset_state"]["root_pose_world"]
    assert gaps and all(gap["qualified_variation_claim_permitted"] is False for gap in gaps)
    by_id = {row["candidate_id"]: row for row in plan["candidates"]}
    selected = [row for row in plan["candidates"] if row["status"] == "selected"]
    assert {row["candidate_id"] for row in selected} == {
        "cosmos3_edge_policy_droid", "groot_n17_droid", "pi05_droid"}
    # No GPU the screening lane may rent meets Cosmos Nano's 80 GB floor, so it is
    # listed infeasible with that reason and never funded; its slot goes on in plan order.
    assert by_id["cosmos3_nano_policy_droid"]["status"] == "infeasible"
    assert by_id["cosmos3_nano_policy_droid"]["blockers"] == ["screening_gpu_memory_floor_unrentable"]
    assert "cost_quote" not in by_id["cosmos3_nano_policy_droid"]
    assert plan["maximum_planned_cost_usd"] == "12.00" and plan["budget_usd"] == "15.00"
    assert all(row["max_cost_usd"] == row["cost_quote"]["maximum_cost_usd"] == "4.00" for row in selected)
    # The fourth compatible $4 candidate exceeds the remaining $3. Preserve
    # that individual reason and preflight a candidate the planner selected.
    # It is waitlisted, frozen with its quote, for money a selected candidate releases.
    assert by_id["pi0_droid_jointpos_100k_polaris"]["status"] == "waitlisted"
    assert by_id["pi0_droid_jointpos_100k_polaris"]["blockers"] == ["site_screening_budget_insufficient"]
    assert by_id["pi0_droid_jointpos_100k_polaris"]["cost_quote"]["maximum_cost_usd"] == "4.00"
    assert by_id["molmoact2_droid"]["status"] == "unavailable"
    assert by_id["molmoact2_droid"]["blockers"] == ["task_kind_not_supported_by_candidate"]
    assert all(row["blockers"] for row in plan["candidates"] if row["status"] != "selected")
    candidate = by_id["groot_n17_droid"]
    prepared = launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=tmp_path / "attempt")
    assert launch._read(tmp_path / "attempt" / "static_preflight.json")["status"] == "passed"
    receipt = launch._read(Path(prepared["bundle_receipt"]["path"]))
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        sealed = json.loads(archive.read("provider_runtime/runtime_inputs/policy_canary_runtime_inputs.json"))
        assert sealed["cells"][0]["resolved_scenario"] == plan["scenario_cells"][0]["resolved_scenario"]
        assert any(row["resolved_scenario"].get("runtime_coverage_gaps") for row in sealed["cells"])
    assert not (tmp_path / "attempt" / "allocator_launch_intent.json").exists()
