"""Compiled rigid jobs reach canonical GPU screening after their exact probe.

Beta L5 regression: exercise the real compiler, probe recorder/scorer, planner
and sealed provider preflight without allocating or claiming GPU evidence.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import zipfile

import pytest

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
    from blueprint_pipeline import allocator_invocation

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
    assert {row["candidate_id"] for row in candidates} == {"pi05_droid", "groot_n17_droid"}
    assert plan["maximum_planned_cost_usd"] == "8.00"
    spawned = []

    def spawn(invocation_path, **_):
        invocation = allocator_invocation.manifest(invocation_path)
        spawned.append(invocation["argv"])
        return os.getpid()  # Retained live identity without spawning a provider process.

    def capacity(**kwargs):
        launch._validate_prepared(kwargs["plan"], kwargs["candidate"], kwargs["attempt_dir"], kwargs["prepared"])
        return {"capacity_available": True, "blockers": []}

    monkeypatch.setattr(allocator_invocation, "start_invocation", spawn)
    # Keep the actual control request signing and exact response validation.
    # The service wire checks a fixture-only token and echoes only that
    # invocation's binding; it grants no provider execution in this test.
    from blueprint_pipeline import safe_outbound_http, beta_policy_screening_control_contracts
    from tests.test_beta_policy_screening_control_contracts import response

    now = int(time.time())
    deadline = now + 3600
    monkeypatch.setattr(beta_policy_screening_control_contracts.time, "time", lambda: now)
    monkeypatch.setattr(worker, "require_current_source", lambda *_, **__: {"execution_deadline_at_ms": deadline * 1000})
    fixture_token = "hermetic-rigid-controller-token"
    token_file = tmp_path / "fixture-controller-token"
    token_file.write_text(fixture_token)
    token_file.chmod(0o600)
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://fixture.invalid/api/internal/pipeline/sync")
    control_requests = []

    def signed_control(url, **kwargs):
        assert url == "https://fixture.invalid/api/internal/pipeline/beta-policy-screening-runs/rigid-campaign/execution-control"
        assert kwargs["method"] == "POST"
        body, headers = kwargs["data"], kwargs["headers"]
        stamp = headers["X-Blueprint-Pipeline-Timestamp"]
        expected = hmac.new(fixture_token.encode(), stamp.encode() + b"." + body, hashlib.sha256).hexdigest()
        assert hmac.compare_digest(headers["X-Blueprint-Pipeline-Signature"], "sha256=" + expected)
        payload = json.loads(body)
        assert payload["schema_version"] == "blueprint.beta_policy_screening_execution_control_request.v1"
        assert payload["execution_deadline_at_ms"] == deadline * 1000
        scope = {key: payload[key] for key in beta_policy_screening_control_contracts.BINDING_FIELDS}
        control_requests.append(scope)
        return SimpleNamespace(body=json.dumps(response(scope=scope, now=now, deadline=deadline)).encode())

    monkeypatch.setattr(safe_outbound_http, "request", signed_control)
    monkeypatch.setattr(launch, "preflight_candidate_capacity", capacity)
    # Replace this module's subprocess binding, rather than the shared Python
    # module: source-commit reads and sealed-bundle CPU preflight still run,
    # while an obsolete spawn seam can never launch an allocator child.
    monkeypatch.setattr(allocator_invocation, "subprocess", SimpleNamespace(
        Popen=lambda *_, **__: pytest.fail("fixture attempted a real allocator subprocess"),
        STDOUT=subprocess.STDOUT))
    queue = tmp_path / "screening-queue"
    coordinator.enqueue_screening_campaign(plan, queue_root=queue)
    progressed = coordinator.process_screening_queue(queue_root=queue, max_launches=2, require_publication_ack=False)
    assert progressed["launches_started"] == 2 and len(spawned) == 2
    assert {request["candidate_id"] for request in control_requests} == {"pi05_droid", "groot_n17_droid"}
    ledger = coordinator._ledger(queue)
    state = ledger["campaigns"][plan["campaign_id"]]
    assert state["site_budget"]["reserved_usd"] == "8"
    for candidate in candidates:
        row = state["candidates"][candidate["candidate_id"]]
        assert row["status"] == "running"
        attempt = queue / "attempts" / row["attempt"]["attempt_id"]
        prepared = row["attempt"]["prepared"]
        observed = next(request for request in control_requests if request["candidate_id"] == candidate["candidate_id"])
        authority = launch._read(Path(prepared["authority"]["path"]))
        assert observed["source"] == source and observed["site_id"] == plan["site_id"]
        assert observed["plan_digest"] == plan["plan_digest"] and observed["configuration_digest"] == candidate["configuration_digest"]
        assert observed["prepared_digest"] == prepared["prepared_digest"]
        assert observed["authority_digest"] == authority["authority_digest"] and observed["resource_name"] == authority["resource_name"]
        assert prepared["argv"][:3] == ["gpu-canary", "--provider", "vast"]
        assert launch._read(attempt / "static_preflight.json")["status"] == "passed"
        bundle = launch._read(Path(prepared["bundle_receipt"]["path"]))
        with zipfile.ZipFile(bundle["bundle_path"]) as archive:
            sealed = json.loads(archive.read("provider_runtime/runtime_inputs/policy_canary_runtime_inputs.json"))
            assert sealed["candidate_ids"] == [candidate["candidate_id"]]
            assert sealed["task_success_contract_digest"] == inputs["task_success_contract_digest"]
            assert sealed["beta_screening_admission"]["robot_free_probe_receipt_digest"] == receipt["receipt_digest"]
        assert (attempt / "allocator_launch_intent.json").is_file()
    replayed = coordinator.process_screening_queue(queue_root=queue, max_launches=2, require_publication_ack=False)
    assert replayed["launches_started"] == 0 and len(spawned) == 2
