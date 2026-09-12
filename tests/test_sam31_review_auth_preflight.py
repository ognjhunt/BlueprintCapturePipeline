"""Reject invalid scoped credentials before render review; replay stays offline."""
import json
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from blueprint_pipeline import openai_credential_preflight as auth
from blueprint_pipeline import task_evaluation_sam31_preparation_review_stages as stages
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _key(tmp_path):
    path = tmp_path / "review.key"
    path.write_text("fixture-secret-do-not-log")
    path.chmod(0o600)
    return path


def test_auth_probe_is_single_pinned_get_with_no_inference(tmp_path, monkeypatch):
    key = _key(tmp_path)
    calls = []
    monkeypatch.setenv("OPENAI_BASE_URL", "https://unrelated.example")
    def request(url, **kwargs):
        calls.append((url, kwargs))
        assert url == "https://api.openai.com/v1/models"
        assert kwargs["method"] == "GET" and "data" not in kwargs
        assert kwargs["headers"] == {"Authorization": "Bearer fixture-secret-do-not-log", "OpenAI-Project": "proj_fixture"}
        assert kwargs["timeout_seconds"] == 10 and kwargs["max_response_bytes"] == 1024 * 1024
        assert kwargs["policy"].allowed_hosts == frozenset({"api.openai.com"})
        assert kwargs["policy"].follow_same_origin_redirects is False
        return SimpleNamespace(status=200, body=b'{"object":"list","data":[]}')
    monkeypatch.setattr(auth.safe_outbound_http, "request", request)
    result = auth.check_openai_credential(api_key_file=key, project_id="proj_fixture")
    assert len(calls) == 1 and result["status"] == "authenticated"
    assert result["model_inference_performed"] is result["spending_authority_granted"] is False
    assert "fixture-secret" not in json.dumps(result)


@pytest.mark.parametrize("failure", [401, 403, 429, 500, "timeout", "body"])
def test_probe_errors_never_expose_provider_text_or_retry(tmp_path, monkeypatch, failure):
    key = _key(tmp_path)
    calls = []
    def request(*args, **kwargs):
        calls.append(True)
        if failure == "body":
            return SimpleNamespace(status=200, body=b'fixture-secret-do-not-log')
        if failure == "timeout":
            raise OSError("fixture-secret-do-not-log")
        raise HTTPError(auth.URL, failure, "fixture-secret-do-not-log", {}, None)
    monkeypatch.setattr(auth.safe_outbound_http, "request", request)
    with pytest.raises(auth.OpenAICredentialPreflightError) as error:
        auth.check_openai_credential(api_key_file=key, project_id="proj_fixture")
    assert len(calls) == 1
    assert "fixture-secret" not in str(error.value)
    assert "fixture-secret" not in stages._failure_blocker(error.value)


@pytest.mark.parametrize("fault", ["mode", "symlink", "missing", "project"])
def test_bad_secret_configuration_refuses_before_transport(tmp_path, monkeypatch, fault):
    key = _key(tmp_path)
    project = "proj_fixture"
    if fault == "mode":
        key.chmod(0o644)
    if fault == "missing":
        key.unlink()
    if fault == "symlink":
        pointer = tmp_path / "pointer"
        pointer.symlink_to(key)
        key = pointer
    if fault == "project":
        project = "proj_fixture\nInjected: value"
    monkeypatch.setattr(auth.safe_outbound_http, "request", lambda *a, **k: pytest.fail("unsafe credential reached HTTP"))
    with pytest.raises(auth.OpenAICredentialPreflightError):
        auth.check_openai_credential(api_key_file=key, project_id=project)


@pytest.mark.parametrize("mode", ["live", "replay", "completed", "legacy"])
def test_stage_auth_precedes_packet_and_render_work_only_for_live_scoped_key(tmp_path, monkeypatch, mode):
    key = _key(tmp_path)
    selection = tmp_path / "selection.json"
    selection.write_text("{}")
    profile = {"source_commit": "a" * 40, "sam31_visual_review": {
        "openai_api_key_file": str(key), "openai_project_id": "proj_fixture"}}
    if mode == "completed":
        profile["sam31_visual_review"]["completed_execution"] = {"retained": True}
    if mode == "legacy":
        profile["sam31_visual_review"].pop("openai_api_key_file")
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    job = {"stage_id": "sam31_review", "request": {"expected_production_commit": "a" * 40},
        "plan": {"source_commit": "a" * 40}, "server_profile": profile,
        "server_data_root": str(tmp_path), "output_root": str(tmp_path / "out"), "inputs": {}}
    if mode == "replay":
        job["diagnostic_replay_code_root"] = str(tmp_path / "candidate")
    calls = []
    def check(**kwargs):
        assert mode == "live", "replay/retained paths must not probe credentials"
        calls.append("auth")
        raise auth.OpenAICredentialPreflightError("openai_credential_preflight_http_401")
    def input_file(_inputs, name, _root):
        if name == "task_selection":
            return selection
        assert mode != "live", "invalid credential reached expensive input validation"
        calls.append("packet_boundary")
        raise stages.Sam31PreparationReviewStageError("fixture_packet_boundary")
    monkeypatch.setattr(auth, "check_openai_credential", check)
    monkeypatch.setattr(stages, "_input", input_file)
    monkeypatch.setattr(stages, "validate_removal_task_selection", lambda _: {"task_id": "fixture"})
    result = stages.execute_review_stage(job)
    assert result["status"] == "blocked"
    assert calls == (["auth"] if mode == "live" else ["packet_boundary"])
    assert not (tmp_path / "out/selection-inputs").exists()
    assert "fixture-secret" not in json.dumps(result)
