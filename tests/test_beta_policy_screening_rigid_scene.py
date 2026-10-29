"""Compiled rigid jobs reach canonical GPU screening after their exact probe.

Beta L5 regression: exercise the real compiler, probe recorder/scorer, planner
and sealed provider preflight without allocating or claiming GPU evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import zipfile

from blueprint_pipeline import beta_policy_screening_launch as launch
from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog
from blueprint_pipeline.beta_policy_screening_capabilities import build_planning_inputs
from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.robot_free_probe_gate import RECEIPT_FILENAME
from blueprint_pipeline.robot_free_readiness_retention import TRACE_FILENAME, retain_robot_free_readiness
from blueprint_pipeline.rigid_robot_free_probe_plan import physics_plan, probe_context
from blueprint_pipeline.rigid_robot_free_probe_recorder import record_rigid_robot_free_probe
from blueprint_pipeline.team_candidate_quick10_registry import scene_facts
from blueprint_pipeline.task_evaluation_scene_rigid_robot_free_readiness import rigid_task_for_revision
from blueprint_pipeline.task_evaluation_rigid_relocation_native_adapter import (
    DEFINITION_CONTRACT_PATH, NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH,
)
from tests.test_native_task_arena_bundle import _runtime_source_packet
from tests.test_rigid_robot_free_probe import FlatSupport, binding
from tests.test_team_scene_packet import _compile, _scene


def test_compiled_pick_place_reaches_canonical_gpu_screening_with_its_scored_probe(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    from blueprint_pipeline import beta_policy_screening_worker as worker
    from blueprint_pipeline import team_candidate_quick10_launch as process

    scene = _scene(tmp_path, strategy="pick_and_place")
    # A top grasp clears the table; the generic relocation fixture's side
    # grasp puts link7 through it and must continue to fail path admission.
    definition_path = scene["documents"][DEFINITION_CONTRACT_PATH]
    definition = launch._read(Path(definition_path))
    definition["interaction_affordance"].update(contact_point_scoring_frame_m=[0.0, 0.0, 0.06],
        approach_unit_scoring_frame=[0.0, 0.0, 1.0])
    definition_path = tmp_path / "top-grasp-definition.json"
    definition_path.write_text(json.dumps(definition) + "\n")
    scene["documents"][DEFINITION_CONTRACT_PATH] = definition_path
    definition_binding = launch._record(definition_path)
    scene["revision"]["task_template"]["definition"].update(digest=definition_binding["sha256"],
        size_bytes=definition_binding["size_bytes"])
    scene["revision"]["revision_digest"] = canonical_digest(scene["revision"], digest_field="revision_digest")
    scene["store"][scene["revision"]["task_template"]["definition"]["uri"]] = definition_path.read_bytes()
    revision_bytes = json.dumps(scene["revision"], sort_keys=True).encode()
    scene["store"][scene["revision_reference"]["uri"]] = revision_bytes
    scene["revision_reference"].update(digest="sha256:" + hashlib.sha256(revision_bytes).hexdigest(),
        size_bytes=len(revision_bytes))
    scene["runtime_source_receipt_path"] = _runtime_source_packet(tmp_path / "runtime")
    compiled, _ = _compile(tmp_path, scene)
    profile = {"capture_root": str(tmp_path / "capture"), "scene_id": "site-rigid-fixture",
               "task_id": compiled["task_id"], "team_scene_inputs_path": compiled["path"],
               "robot_free_probe_root": str(tmp_path / "probe"), "open_loop_horizon": 8,
               "max_hourly_rate_usd": 0.8}
    facts = scene_facts(profile)
    spec, inputs = facts["scene"]["task_spec"], facts["inputs"]
    assert spec["task_kind"] == "rigid_pick_place"
    assert spec["manipulation_strategy"] == "pick_and_place"
    probe_spec = rigid_task_for_revision(revision=scene["revision"], documents=scene["documents"])["task_spec"]
    assert probe_spec == spec
    asset_digest = launch._read(Path(scene["documents"][NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH])).get("asset_digest")
    trace = record_rigid_robot_free_probe(scene=FlatSupport(spec["start_pose_world"]),
        plan=physics_plan(probe_context(spec)),
        binding=binding(spec, inputs["scene_revision_digest"], asset_digest))
    probe_root = Path(profile["robot_free_probe_root"])
    launch._write(probe_root / TRACE_FILENAME, trace)
    retained = retain_robot_free_readiness(run_root=probe_root, revision_reference=scene["revision_reference"],
        revision_digest=inputs["scene_revision_digest"], fetcher=lambda ref: scene["store"][ref["uri"]])
    receipt = launch._read(probe_root / RECEIPT_FILENAME)
    assert retained["status"] == receipt["status"] == "passed", receipt.get("probe_error")
    source = {"capture_id": "rigid-capture", "task_id": profile["task_id"], "scene_intent_id": "rigid-intent",
              "scene_intent_digest": "sha256:" + "a" * 64, "task_context_digest": "sha256:" + "b" * 64,
              "scene_revision_digest": inputs["scene_revision_digest"]}
    catalog = load_default_catalog()
    planning = build_planning_inputs(catalog=catalog, profile=profile, capture_root=Path(profile["capture_root"]),
        source=source, inputs_dir=tmp_path / "screening-inputs")
    plan = build_screening_plan(catalog, planning["scene"], capabilities=planning["capabilities"],
        cost_quotes=planning["cost_quotes"], campaign_id="rigid-campaign")
    candidates = [row for row in plan["candidates"] if row["status"] == "selected"]
    assert {row["candidate_id"] for row in candidates} == {"pi05_droid", "groot_n17_droid", "cosmos3_edge_policy_droid"}
    assert plan["maximum_planned_cost_usd"] == "12.00"
    spawned = []

    def spawn(argv, **_):
        spawned.append(argv)
        return os.getpid()  # Retained live identity without spawning a provider process.

    def capacity(**kwargs):
        launch._validate_prepared(kwargs["plan"], kwargs["candidate"], kwargs["attempt_dir"], kwargs["prepared"])
        return {"capacity_available": True, "blockers": []}

    monkeypatch.setattr(process, "_start_allocator", spawn)
    monkeypatch.setattr(worker, "require_current_source", lambda *_, **__: {})
    monkeypatch.setattr(launch, "preflight_candidate_capacity", capacity)
    queue = tmp_path / "screening-queue"
    coordinator.enqueue_screening_campaign(plan, queue_root=queue)
    progressed = coordinator.process_screening_queue(queue_root=queue, max_launches=3, require_publication_ack=False)
    assert progressed["launches_started"] == 3 and len(spawned) == 3
    ledger = coordinator._ledger(queue)
    state = ledger["campaigns"][plan["campaign_id"]]
    assert state["site_budget"]["reserved_usd"] == "12"
    for candidate in candidates:
        row = state["candidates"][candidate["candidate_id"]]
        assert row["status"] == "running"
        attempt = queue / "attempts" / row["attempt"]["attempt_id"]
        prepared = row["attempt"]["prepared"]
        assert prepared["argv"][:3] == ["gpu-canary", "--provider", "vast"]
        assert launch._read(attempt / "static_preflight.json")["status"] == "passed"
        bundle = launch._read(Path(prepared["bundle_receipt"]["path"]))
        with zipfile.ZipFile(bundle["bundle_path"]) as archive:
            sealed = json.loads(archive.read("provider_runtime/runtime_inputs/policy_canary_runtime_inputs.json"))
            assert sealed["candidate_ids"] == [candidate["candidate_id"]]
            assert sealed["task_success_contract_digest"] == inputs["task_success_contract_digest"]
            assert sealed["beta_screening_admission"]["robot_free_probe_receipt_digest"] == receipt["receipt_digest"]
            if candidate["candidate_id"] == "cosmos3_edge_policy_droid":
                manifest = json.loads(archive.read("provider_runtime/adp_arena_provider_manifest.json"))
                assert manifest["minimum_gpu_memory_gb"] == 48
                assert "provider_runtime/cosmos_edge_policy_snapshot_manifest.json" in archive.namelist()
                native = json.loads(archive.read("provider_runtime/native_task_packet/native_task_arena_scene_plan.v1.json"))
                cameras = {row["role"]: row for row in native["cameras"]}
                assert cameras["exterior_right"]["policy_input"] is True
                assert cameras["overview"]["policy_input"] is False
                script = archive.read("provider_runtime/adp009d_policy_provisioning.cosmos3_edge_policy_droid.sh").decode()
                assert "--group cu128-train --group policy-server" in script
        assert (attempt / "allocator_launch_intent.json").is_file()
    replayed = coordinator.process_screening_queue(queue_root=queue, max_launches=3, require_publication_ack=False)
    assert replayed["launches_started"] == 0 and len(spawned) == 3
