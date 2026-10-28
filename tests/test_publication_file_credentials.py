"""AUT-15: the existing publication transports share one private token contract."""
from __future__ import annotations

import json
import sys
import pytest

from blueprint_pipeline import capture_qa_webapp_sync as qa_sync
from blueprint_pipeline import capture_reconstruction_status_sync as reconstruction_sync
from blueprint_pipeline import site_task_testbed_webapp_sync as testbed_sync
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.site_task_testbed_compiler import compile_site_task_testbed
from tests import test_site_task_testbed_compiler as testbed_fixture


class _Response:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(self.value).encode()


def _case(kind):
    if kind == "qa":
        report = {
            "schema_version": "capture_qa_report.v1", "intake_id": "intake-1",
            "capture_authority_profile": "camera_360_equirectangular",
            "envelope_digest": "sha256:" + "a" * 64,
            "status": "accepted", "state": "capture_accepted", "checks": [],
            "recapture_plan": [], "comparative_policy_ranking_verdict": "thesis_not_supported",
            "claim_ceiling": {key: False for key in (
                "physical_task_success", "deployment_readiness", "safety_certification")},
        }
        report["qa_report_digest"] = canonical_digest(report, digest_field="qa_report_digest")
        return qa_sync, qa_sync.sync_capture_qa_to_webapp, {
            "capture_session_id": "capture-1", "report": report,
            "endpoint_url": "https://webapp.test/api/internal/pipeline/capture-qa"}, "qa_report_digest"
    if kind == "reconstruction":
        status = reconstruction_sync.build_terminal_status(
            capture_id="capture-1", capture_digest="sha256:" + "b" * 64,
            state="failed", blockers=["retained_scene_unavailable"],
            completed_at="2026-10-28T16:00:00Z")
        return reconstruction_sync, reconstruction_sync.sync_terminal_status_to_webapp, {
            "status": status, "endpoint_url": "https://webapp.test/api/internal/captures"}, "status_digest"
    envelope = testbed_fixture._envelope()
    plan = testbed_fixture._plan(testbed_fixture._qa(envelope))
    approved = testbed_fixture._approved_task()
    testbed = compile_site_task_testbed(
        testbed_id="file-token-testbed", version="1", capture_intake_envelope=envelope,
        capture_qa_report=testbed_fixture._qa(envelope), approved_task_definition=approved,
        reconstruction_plan=plan, reconstruction_results=[testbed_fixture._result(plan)],
        simready_decision=testbed_fixture._simready(),
        robot_placement_result=testbed_fixture._placement(), artifact_references=testbed_fixture._refs(),
        supported_condition_ranges={"lighting_lux": [300, 600]})
    return testbed_sync, testbed_sync.sync_site_task_testbed_to_webapp, {
        "capture_session_id": "capture-1", "intake_id": "intake-1",
        "approved_task_digest": approved["approved_task_digest"], "testbed": testbed,
        "decision_evidence_request": testbed_fixture._decision_request(testbed, approved),
        "endpoint_url": "https://webapp.test/api/internal/pipeline/capture-testbeds"}, "testbed_digest"


def _receipt(kind, body):
    if kind == "reconstruction":
        return {"schema_version": reconstruction_sync.SYNC_RECEIPT_SCHEMA_VERSION,
                **{key: body[key] for key in ("capture_id", "capture_digest", "status_digest")},
                "written": True, "already_synced": False}
    fields = (("capture_session_id", "intake_id", "approved_task_digest", "testbed_id",
               "version", "testbed_digest", "artifact_reference", "proof_boundary")
              if kind == "testbed" else ("capture_session_id", "intake_id", "capture_authority_profile",
                   "envelope_digest", "qa_report_digest", "status", "state", "proof_boundary"))
    result = {key: body[key] for key in fields}
    result.update({"schema_version": ("capture_site_task_testbed_publication_receipt.v1"
                   if kind == "testbed" else "capture_qa_publication_receipt.v1"), "already_exists": False})
    if kind == "testbed":
        result.update(status="testbed_ready",
                      request_digest=(body.get("decision_evidence_request") or {}).get("request_digest"))
    return result


def _transport(monkeypatch, module, kind, *, mutate=None):
    bodies, tokens = [], []
    headers = module._pipeline_sync_headers

    def sign(token, body):
        tokens.append(token)
        return headers(token, body)

    def send(request, **_):
        bodies.append(request.data)
        receipt = _receipt(kind, json.loads(request.data))
        if mutate:
            mutate(receipt)
        return _Response(receipt)

    monkeypatch.setattr(module, "_pipeline_sync_headers", sign)
    monkeypatch.setattr(module.urllib_request, "urlopen", send)
    return bodies, tokens


def _assert_succeeded(kind, result):
    assert result.get("written") is True if kind == "reconstruction" else result["status"] == "succeeded"


@pytest.mark.parametrize("kind", ["testbed", "qa", "reconstruction"])
@pytest.mark.parametrize("ambient", [False, True])
def test_file_only_transport_uses_canonical_precedence(tmp_path, monkeypatch, kind, ambient):
    module, sync, arguments, _ = _case(kind)
    private = tmp_path / "sync-token"
    private.write_text("private-fixture-token\n")
    private.chmod(0o600)
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN_FILE", str(private))
    if ambient:
        monkeypatch.setenv("PIPELINE_SYNC_TOKEN", "legacy-fixture-token")
    else:
        monkeypatch.delenv("PIPELINE_SYNC_TOKEN", raising=False)
    bodies, tokens = _transport(monkeypatch, module, kind)
    result = sync(**arguments)
    _assert_succeeded(kind, result)
    assert tokens == ["private-fixture-token"]
    assert len(bodies) == 1
    assert "private-fixture-token" not in json.dumps(result)


@pytest.mark.parametrize("kind", ["testbed", "qa", "reconstruction"])
@pytest.mark.parametrize("failure", ["unsafe", "missing", "empty", "oversized", "symlink", "non_utf8"])
def test_bad_configured_file_never_falls_back_to_ambient(tmp_path, monkeypatch, kind, failure):
    module, sync, arguments, _ = _case(kind)
    private = tmp_path / "private-token"
    private.write_bytes(b"private-fixture-token")
    private.chmod(0o600)
    selected = private
    if failure == "unsafe":
        private.chmod(0o644)
    elif failure == "missing":
        selected = tmp_path / "missing"
    elif failure == "empty":
        private.write_bytes(b" \n")
    elif failure == "oversized":
        private.write_bytes(b"x" * 4097)
    elif failure == "non_utf8":
        private.write_bytes(b"\xff")
    elif failure == "symlink":
        selected = tmp_path / "link"
        selected.symlink_to(private)
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN_FILE", str(selected))
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN", "ambient-must-not-be-used")
    bodies, tokens = _transport(monkeypatch, module, kind)
    if kind == "reconstruction":
        with pytest.raises(reconstruction_sync.CaptureReconstructionStatusError) as exc:
            sync(**arguments)
        diagnostic = str(exc.value)
    else:
        result = sync(**arguments)
        assert result["status"] == "failed" and result["attempts"] == 0
        diagnostic = json.dumps(result)
    assert "pipeline_sync_token_" in diagnostic
    assert str(private) not in diagnostic and "private-fixture-token" not in diagnostic
    assert bodies == tokens == []


@pytest.mark.parametrize("kind", ["testbed", "qa", "reconstruction"])
def test_explicit_override_and_recovered_file_reuse_identical_artifact(tmp_path, monkeypatch, kind):
    module, sync, arguments, _ = _case(kind)
    private = tmp_path / "sync-token"
    private.write_text("file-token")
    private.chmod(0o644)
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN_FILE", str(private))
    bodies, tokens = _transport(monkeypatch, module, kind)
    first = sync(**arguments, token="explicit-fixture-token")
    _assert_succeeded(kind, first)
    private.chmod(0o600)
    second = sync(**arguments)
    _assert_succeeded(kind, second)
    assert tokens == ["explicit-fixture-token", "file-token"]
    assert bodies[0] == bodies[1]


def test_failed_file_only_testbed_publication_recovers_retained_request_without_reconstruction(tmp_path, monkeypatch):
    module, sync, arguments, _ = _case("testbed")
    retained_testbed = json.dumps(arguments["testbed"], sort_keys=True)
    retained_request = json.dumps(arguments["decision_evidence_request"], sort_keys=True)
    private = tmp_path / "missing-until-recovery"
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN_FILE", str(private))
    monkeypatch.delenv("PIPELINE_SYNC_TOKEN", raising=False)
    bodies, tokens = _transport(monkeypatch, module, "testbed")
    refused = sync(**arguments)
    assert refused["status"] == "failed" and refused["attempts"] == 0
    assert bodies == tokens == []
    # The transport retries the already compiled scientific bytes; no compiler
    # or reconstruction step is used to recover a private transport credential.
    def no_compilation(**_):
        pytest.fail("unexpected compilation")
    # Guard the actual direct import used by _case, the compiler's canonical
    # export, and the fixture module alias; mutating recovery to rebuild cannot
    # hide behind a different imported name.
    monkeypatch.setattr(sys.modules[__name__], "compile_site_task_testbed", no_compilation)
    monkeypatch.setattr("blueprint_pipeline.site_task_testbed_compiler.compile_site_task_testbed", no_compilation)
    monkeypatch.setattr(testbed_fixture, "compile_site_task_testbed", no_compilation)
    private.write_text("recovered-file-token")
    private.chmod(0o600)
    accepted = sync(**arguments)
    assert accepted["status"] == "succeeded" and len(bodies) == 1
    offered = json.loads(bodies[0])
    assert json.dumps(offered["testbed"], sort_keys=True) == retained_testbed
    assert json.dumps(offered["decision_evidence_request"], sort_keys=True) == retained_request


@pytest.mark.parametrize("kind", ["testbed", "qa", "reconstruction"])
@pytest.mark.parametrize("mismatch", ["digest", "source"])
def test_file_credentials_do_not_weaken_receipt_binding(tmp_path, monkeypatch, kind, mismatch):
    module, sync, arguments, digest_field = _case(kind)
    private = tmp_path / "sync-token"
    private.write_text("file-token")
    private.chmod(0o600)
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN_FILE", str(private))
    def change(receipt):
        receipt[digest_field if mismatch == "digest" else (
            "capture_id" if kind == "reconstruction" else "capture_session_id")] = "wrong"
    _transport(monkeypatch, module, kind, mutate=change)
    if kind == "reconstruction":
        with pytest.raises(reconstruction_sync.CaptureReconstructionStatusError, match="response_binding_mismatch"):
            sync(**arguments)
    else:
        assert sync(**arguments)["reason"] == "response_binding_mismatch"
