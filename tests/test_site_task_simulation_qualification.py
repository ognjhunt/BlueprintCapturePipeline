"""Hermetic contract evidence; synthetic samples never constitute live qualification."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest as digest
from blueprint_pipeline.site_task_simulation_qualification import (
    DIMENSIONS, SiteTaskQualificationError, main, qualification_for_execution, qualify_from_profile,
    validate_receipt,
)

NOW = datetime(2026, 10, 18, tzinfo=timezone.utc)
SHA = "sha256:" + "a" * 64


def fixture(root: Path, *, scene=None, model=None, binding=None):
    from tests.test_decision_evidence_router import _testbed
    from tests.test_capture_intake import _envelope
    from tests.test_team_candidate_quick10_admission import _inputs
    from blueprint_pipeline.capture_intake import validate_capture_intake_envelope
    root.mkdir(parents=True, exist_ok=True)
    scene = scene or _inputs()["scene"]
    contract = scene["task_success_contract"]
    if "contract_digest" not in contract:
        contract["contract_digest"] = canonical_digest(contract)
    testbed = _testbed()
    capture_bytes = b"fixture retained capture video; not a real site"
    (root / "capture.mp4").write_bytes(capture_bytes)
    capture = _envelope(capture_bytes)
    capture["intake_id"] = "capture-1"
    capture["scene_id"] = (binding or {}).get("site_id", "site-fixture")
    capture = validate_capture_intake_envelope(capture)
    testbed["source_capture_bundles"][0].update(envelope_digest=capture["envelope_digest"],
                                               version=capture["capture_authority_profile"])
    testbed["approved_task_definition"] = {"approved_task_id": (binding or {}).get("task_id", "task-fixture")}
    testbed["task_distribution"]["tasks"] = [testbed["approved_task_definition"]["approved_task_id"]]
    testbed["testbed_digest"] = canonical_digest(testbed, digest_field="testbed_digest")
    model = model or b"fixture robot model bytes; not operational evidence"
    bound = {"scene_request_id": "scene-request-fixture", "site_id": "site-fixture", "task_id": "task-fixture",
             "capture_id": "capture-fixture", "source_capture_digest": SHA,
             "testbed_digest_sha256": testbed["testbed_digest"], "scene_revision_digest": scene["scene_revision_digest"],
             "task_spec_digest": canonical_digest(scene["task_spec"]), "success_contract_digest": contract["contract_digest"],
             "robot_model_digest": "sha256:" + hashlib.sha256(model).hexdigest(), "policy_interface_id": "blueprint_policy_https.v1",
             **(binding or {})}
    profile = {"schema_version": "blueprint.site_task_fidelity_profile.v1", "checks": [
        {"id": dimension + "_fixture", "dimension": dimension, "unit": "ratio", "minimum_sample_count": 2,
         "maximum_absolute_error": 0.05, "maximum_rms_error": 0.04} for dimension in sorted(DIMENSIONS)]}
    observed = {"schema_version": "blueprint.site_task_measurement_samples.v1", "binding": bound,
                "producer_id": "fixture-independent-measurement", "observed_at_iso": "2026-10-17T10:00:00Z",
                "checks": [{"id": c["id"], "dimension": c["dimension"], "unit": c["unit"],
                            "samples": [{"sample_id": "point-1", "value": 1.0}, {"sample_id": "point-2", "value": 2.0}]}
                           for c in profile["checks"]]}
    simulated = copy.deepcopy(observed)
    simulated["producer_id"] = "fixture-simulator"
    for check in simulated["checks"]:
        check["samples"][0]["value"] += 0.01
    values = {"capture": capture, "testbed": testbed,
              "scene": {"task_spec": scene["task_spec"]}, "task_spec": scene["task_spec"],
              "success_contract": contract, "robot_model": model, "measurement": observed,
              "simulator_measurement": simulated}
    sources = []
    for role, value in values.items():
        raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode()
        filename = role + ".json"
        (root / filename).write_bytes(raw)
        sources.append({"role": role, "path": filename, "uri": "gs://blueprint-test-fixtures/qualification/" + filename,
                        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)})
    request = {"schema_version": "blueprint.site_task_qualification_request.v1", "binding": bound, "source_artifacts": sources}
    review = {"schema_version": "blueprint.site_task_qualification_review.v1", "request_digest": digest(request),
              "profile_digest": digest(profile), "decision": "accepted_for_bounded_simulation", "reviewed_by": "fixture-reviewer",
              "reviewed_at_iso": "2026-10-17T11:00:00Z", "expires_at_iso": "2099-01-01T00:00:00Z"}
    authority = {"allowed_reviewers": ["fixture-reviewer"], "allowed_measurement_producers": [observed["producer_id"]],
                 "allowed_simulator_producers": [simulated["producer_id"]],
                 "canonical_uri_prefixes": ["gs://blueprint-test-fixtures/qualification/"]}
    for role, value in (("request", request), ("profile", profile), ("review", review)):
        path = root / (role + ".json")
        path.write_text(json.dumps(value))
        authority[role + "_path"] = str(path)
        authority[role + "_digest"] = digest(value)
    return {"site_task_qualification": authority}


def _rewrite(profile, role, update):
    authority = profile["site_task_qualification"]
    path = Path(authority[role + "_path"])
    value = json.loads(path.read_text())
    update(value)
    path.write_text(json.dumps(value))
    authority[role + "_digest"] = digest(value)


def _reauthorize_review(profile):
    authority = profile["site_task_qualification"]
    _rewrite(profile, "review", lambda v: v.update(request_digest=authority["request_digest"], profile_digest=authority["profile_digest"]))


def test_qualification_consumes_raw_samples_and_emits_portable_receipts(tmp_path):
    profile = fixture(tmp_path)
    receipt = qualify_from_profile(profile, now=NOW)
    assert validate_receipt(receipt, now=NOW) == receipt
    assert len(receipt["source_artifacts"]) == 8
    assert all("path" not in source for source in receipt["source_artifacts"])
    assert receipt["physical_performance_claimed"] is False
    assert receipt["adp_qualified_evidence_eligible"] is False
    assert receipt["registration_receipt"]["checks"][0]["max_abs_error"] == pytest.approx(0.01)
    assert receipt["qualification"]["registration_receipt_digest"] == receipt["registration_receipt"]["receipt_digest"]


def test_relocating_identical_retained_evidence_preserves_portable_receipt_identity(tmp_path):
    profile = fixture(tmp_path / "original")
    receipt = qualify_from_profile(profile, now=NOW)
    shutil.copytree(tmp_path / "original", tmp_path / "restored")
    restored = copy.deepcopy(profile)
    for role in ("request", "profile", "review"):
        restored["site_task_qualification"][role + "_path"] = str(tmp_path / "restored" / (role + ".json"))
    assert qualify_from_profile(restored, now=NOW) == receipt


@pytest.mark.parametrize("role", ["capture", "testbed", "scene", "task_spec", "success_contract", "robot_model", "measurement", "simulator_measurement"])
def test_changed_raw_source_bytes_are_rejected(tmp_path, role):
    profile = fixture(tmp_path)
    (tmp_path / (role + ".json")).write_bytes(b"changed")
    with pytest.raises(SiteTaskQualificationError, match="source_bytes_changed"):
        qualify_from_profile(profile, now=NOW)


@pytest.mark.parametrize("mutation", ["unauthorized", "expired", "future", "wrong_request"])
def test_review_is_authorized_current_and_bound(tmp_path, mutation):
    profile = fixture(tmp_path)
    changes = {"unauthorized": {"reviewed_by": "someone-else"}, "expired": {"expires_at_iso": "2026-10-17T12:00:00Z"},
               "future": {"reviewed_at_iso": "2098-01-01T00:00:00Z"}, "wrong_request": {"request_digest": "sha256:" + "b" * 64}}
    _rewrite(profile, "review", lambda value: value.update(changes[mutation]))
    with pytest.raises(SiteTaskQualificationError, match="review_unauthorized"):
        qualify_from_profile(profile, now=NOW)


def test_a_self_asserted_review_and_looser_tolerance_do_not_change_operator_authority(tmp_path):
    profile = fixture(tmp_path)
    path = tmp_path / "profile.json"
    edited = json.loads(path.read_text())
    edited["checks"][0]["maximum_absolute_error"] = 100
    path.write_text(json.dumps(edited))
    with pytest.raises(SiteTaskQualificationError, match="profile_digest_mismatch"):
        qualify_from_profile(profile, now=NOW)


@pytest.mark.parametrize("mutation,reason", [("failure", "fidelity_failed"), ("nan", "samples_invalid"),
    ("missing", "samples_insufficient"), ("unpaired", "sample_pairs_mismatch"), ("same_producer", "independent_review_required")])
def test_evidence_must_pass_independent_matching_numeric_checks(tmp_path, mutation, reason):
    profile = fixture(tmp_path)
    path = tmp_path / "simulator_measurement.json"
    value = json.loads(path.read_text())
    if mutation == "failure":
        value["checks"][0]["samples"][0]["value"] = 10
    if mutation == "nan":
        value["checks"][0]["samples"][0]["value"] = float("nan")
    if mutation == "missing":
        value["checks"][0]["samples"].pop()
    if mutation == "unpaired":
        value["checks"][0]["samples"][0]["sample_id"] = "other"
    if mutation == "same_producer":
        value["producer_id"] = "fixture-independent-measurement"
        profile["site_task_qualification"]["allowed_simulator_producers"] = [value["producer_id"]]
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    def update(request):
        source = next(source for source in request["source_artifacts"] if source["role"] == "simulator_measurement")
        source.update(sha256="sha256:" + hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    _rewrite(profile, "request", update)
    _reauthorize_review(profile)
    with pytest.raises(SiteTaskQualificationError, match=reason):
        qualify_from_profile(profile, now=NOW)


def test_incomplete_dimension_coverage_cannot_qualify(tmp_path):
    profile = fixture(tmp_path)
    _rewrite(profile, "profile", lambda v: v["checks"].pop())
    _reauthorize_review(profile)
    with pytest.raises(SiteTaskQualificationError, match="coverage_incomplete"):
        qualify_from_profile(profile, now=NOW)


def test_relabelled_or_mixed_receipts_cannot_validate(tmp_path):
    receipt = qualify_from_profile(fixture(tmp_path), now=NOW)
    receipt["registration_receipt"]["binding"]["scene_revision_digest"] = "sha256:" + "b" * 64
    receipt["receipt_digest"] = digest(receipt, digest_field="receipt_digest")
    with pytest.raises(SiteTaskQualificationError):
        validate_receipt(receipt, now=NOW)


def test_diagnostic_admission_cannot_carry_a_qualification(tmp_path):
    receipt = qualify_from_profile(fixture(tmp_path), now=NOW)
    with pytest.raises(SiteTaskQualificationError, match="diagnostic_cannot_promote"):
        qualification_for_execution(row={"execution_admission": {"evidence_scope": "development_only", "site_task_qualification_receipt": receipt}},
            request={}, operator_profile={}, facts={}, robot_model_path=tmp_path / "model", now=NOW)


def test_local_command_requires_operator_registry_and_writes_without_overwriting(tmp_path, monkeypatch):
    from blueprint_pipeline.team_candidate_quick10_registry import REGISTRY_ENV, REGISTRY_SCHEMA
    profile = fixture(tmp_path / "evidence")
    profile["capture_root"] = str(tmp_path.resolve())
    registry = {"schema_version": REGISTRY_SCHEMA, "profiles": [profile]}
    registry["registry_digest"] = canonical_digest(registry)
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry))
    path.chmod(0o600)
    monkeypatch.setenv(REGISTRY_ENV, str(path))
    output = tmp_path / "receipt.json"
    assert main(["--capture-root", str(tmp_path), "--out", str(output)]) == 0
    assert json.loads(output.read_text())["schema_version"] == "blueprint.site_task_simulation_qualification.v1"
    original = output.read_bytes()
    assert main(["--capture-root", str(tmp_path), "--out", str(output)]) == 0
    assert output.read_bytes() == original
    output.write_text("{}")
    with pytest.raises(SiteTaskQualificationError, match="output_conflict"):
        main(["--capture-root", str(tmp_path), "--out", str(output)])
    assert output.read_text() == "{}"


def test_json_bytes_are_authenticated_and_parsed_from_one_buffer(tmp_path, monkeypatch):
    profile = fixture(tmp_path)
    path = tmp_path / "simulator_measurement.json"
    original_read = Path.read_bytes
    read_count = []
    def interleaved_read(self):
        raw = original_read(self)
        if self == path:
            read_count.append(self)
            changed = json.loads(raw)
            changed["checks"][0]["samples"][0]["value"] = 1.0
            self.write_text(json.dumps(changed))
        return raw
    monkeypatch.setattr(Path, "read_bytes", interleaved_read)
    receipt = qualify_from_profile(profile, now=NOW)
    # It parsed the authenticated original buffer, never the replacement's 0 error.
    assert len(read_count) == 1
    assert receipt["registration_receipt"]["checks"][0]["max_abs_error"] == pytest.approx(0.01)
    with pytest.raises(SiteTaskQualificationError, match="source_bytes_changed"):
        qualify_from_profile(profile, now=NOW)


def test_lost_publication_acknowledgement_can_retry_the_same_retained_output(tmp_path, monkeypatch):
    from blueprint_pipeline import agent_run_executor, task_evaluation_launch_webapp_sync
    from blueprint_pipeline.team_candidate_quick10_registry import REGISTRY_ENV, REGISTRY_SCHEMA
    profile = fixture(tmp_path / "evidence")
    profile["capture_root"] = str(tmp_path.resolve())
    registry = {"schema_version": REGISTRY_SCHEMA, "profiles": [profile]}
    registry["registry_digest"] = canonical_digest(registry)
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry))
    path.chmod(0o600)
    monkeypatch.setenv(REGISTRY_ENV, str(path))
    calls = []
    class Client:
        def __init__(self, **_kwargs): pass
        def _json(self, _path, *, method, payload):
            calls.append(payload["receipt_digest"])
            if len(calls) == 1:
                raise TimeoutError("synthetic lost acknowledgement")
            return {"receiptDigest": payload["receipt_digest"], "created": False}
    monkeypatch.setattr(agent_run_executor, "AgentRunWebAppClient", Client)
    monkeypatch.setattr(task_evaluation_launch_webapp_sync, "load_pipeline_sync_token", lambda **_kwargs: "synthetic")
    output = tmp_path / "receipt.json"
    argv = ["--capture-root", str(tmp_path), "--out", str(output), "--publish", "--webapp-url", "https://webapp.invalid",
            "--token-file", str(tmp_path / "unused-test-token")]
    with pytest.raises(TimeoutError):
        main(argv)
    original = output.read_bytes()
    assert main(argv) == 0
    assert len(calls) == 2 and len(set(calls)) == 1 and output.read_bytes() == original


def _qualified_world(tmp_path, monkeypatch):
    from tests.test_team_candidate_quick10_launch import _world
    from tests.test_team_embodiment import manifest, write_robot_usd
    from blueprint_pipeline.team_candidate_quick10_registry import profile_for_request, offer_eligibility
    model = write_robot_usd(tmp_path / "robot.usda")
    world = _world(tmp_path, monkeypatch, robot={"manifest": manifest(model, control_hz=15), "model": model})
    envelope = world["row"]["execution_admission"]
    request = envelope["canonical_execution_request"]
    operator = profile_for_request(request)
    facts, blockers = offer_eligibility(operator)
    assert not blockers
    bound = {"capture_id": request["site_package"]["capture_id"],
             "scene_request_id": envelope["binding"]["scene_request_id"]}
    source = fixture(tmp_path / "qualification", scene=facts["scene"], model=model.read_bytes(), binding=bound)
    raw = (Path(facts["packet_dir"]) / "native_task_arena_scene_plan.v1.json").read_bytes()
    (tmp_path / "qualification" / "scene.json").write_bytes(raw)
    def update(request):
        artifact = next(row for row in request["source_artifacts"] if row["role"] == "scene")
        artifact.update(sha256="sha256:" + hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    _rewrite(source, "request", update)
    _reauthorize_review(source)
    receipt = qualify_from_profile(source)
    source["site_task_qualification"]["receipt_digest"] = receipt["receipt_digest"]
    registry_path = world["operator"] / "registry.json"
    registry = json.loads(registry_path.read_text())
    registry["profiles"][0].update(source)
    registry["registry_digest"] = canonical_digest(registry, digest_field="registry_digest")
    registry_path.write_text(json.dumps(registry))
    envelope["evidence_scope"] = "simulation"
    envelope["site_task_qualification_receipt"] = receipt
    envelope["site_task_qualification"] = receipt["qualification"]
    envelope["binding"].update({key: receipt["binding"][key] for key in
                               ("scene_request_id", "site_id", "task_id", "capture_id", "testbed_digest_sha256")})
    envelope["binding"]["capture_digest_sha256"] = receipt["binding"]["source_capture_digest"]
    frozen = json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    world["row"].update(execution_admission_canonical_json=frozen,
                        execution_admission_digest="sha256:" + hashlib.sha256(frozen.encode()).hexdigest())
    return world, receipt


def test_qualified_team_run_rehearsal_carries_exact_evidence_from_admission_to_result(tmp_path, monkeypatch):
    from tests.test_team_candidate_quick10_launch import _staged_run_reported
    world, receipt = _qualified_world(tmp_path, monkeypatch)
    staged, reported = _staged_run_reported(world)
    assert staged["staged"] == reported["completed"] == 1, (staged, reported, world["client"].posted)
    admission = json.loads((world["job_dir"] / "team_candidate_admission.json").read_text())
    assert admission["evidence_scope"] == "simulation"
    assert admission["site_task_qualification_receipt"] == receipt
    result = world["client"].posted[0][1]
    assert result["episodes_run"] == 200
    assert result["scoring"]["claim_ceiling"] == "site_task_simulation"
    assert result["scoring"]["site_task_qualification"] == receipt["qualification"]
    # The inner worker's diagnostic scope is intentionally retained unchanged.
    inputs = json.loads(next(world["job_dir"].glob("shards/*/stage-*/runtime_inputs/policy_canary_runtime_inputs.json")).read_text())
    assert inputs["claim_ceiling"] == "diagnostic_policy_execution"


def test_qualified_source_drift_refuses_before_releasing_credentials(tmp_path, monkeypatch):
    from tests.test_team_candidate_quick10_launch import _Broker
    from blueprint_pipeline import agent_run_executor as executor
    world, _ = _qualified_world(tmp_path, monkeypatch)
    (tmp_path / "qualification" / "capture.mp4").write_bytes(b"altered raw capture")
    result = executor.poll_once(client=world["client"], capture_root=world["capture"], team_launch_hooks=world["hooks"])
    assert result["blocked"] == 1 and result["completed"] == 0
    assert world["client"].blocked == ["site_task_qualification_capture_original_changed"]
    assert not _Broker.leases and not world["allocator_calls"]


def test_qualified_source_drift_at_closeout_cannot_report_recommendation_evidence(tmp_path, monkeypatch):
    from blueprint_pipeline import agent_run_executor as executor
    from blueprint_pipeline import team_candidate_quick10_launch as launch
    world, _ = _qualified_world(tmp_path, monkeypatch)
    staged = executor.poll_once(client=world["client"], capture_root=world["capture"], team_launch_hooks=world["hooks"])
    assert staged["staged"] == 1
    (tmp_path / "qualification" / "capture.mp4").write_bytes(b"altered after admission")
    launch.run_queue(world["queue"], hooks=world["runner"])
    reported = executor.poll_once(client=world["client"], capture_root=world["capture"], team_launch_hooks=world["hooks"])
    assert reported["completed"] == 0
    assert not any(payload.get("scoring", {}).get("claim_ceiling") == "site_task_simulation"
                   for _path, payload in world["client"].posted)


def test_publication_is_explicit_and_requires_exact_acknowledgement(tmp_path):
    from blueprint_pipeline.site_task_simulation_qualification import publish_receipt
    receipt = qualify_from_profile(fixture(tmp_path), now=NOW)
    class Client:
        def _json(self, path, *, method, payload):
            assert path == "/api/internal/pipeline/site-task-qualifications" and method == "POST" and payload == receipt
            return {"receiptDigest": receipt["receipt_digest"], "created": True}
    assert publish_receipt(receipt, client=Client())["created"] is True
    class BadClient:
        def _json(self, *_args, **_kwargs): return {"receiptDigest": SHA, "created": True}
    with pytest.raises(SiteTaskQualificationError, match="acknowledgement_mismatch"):
        publish_receipt(receipt, client=BadClient())
