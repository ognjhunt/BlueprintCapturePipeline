"""A rejected first request permits bounded key repair, never unaccounted retries."""
import io
import json
from pathlib import Path
import urllib.error

import pytest

from blueprint_pipeline import task_evaluation_authoring_auth_recovery as auth
from tests.test_unentered_authoring_budget import archive, evidence, seal


def failed_request(tmp_path, identity=None):
    request, files, result = evidence(tmp_path)
    result.update(identity or {})
    result.update(provider_mutations_performed=0, continuing_spend_from_this_run=False)
    ids = {k: result[k] for k in ("run_id", "source_commit")}
    stage = files["stages/stage-3/producer/stage_production_input.v1.json"]
    stage.update(ids)
    stage["construction_envelope"]["request"] = request
    provider = files["task_evaluation_scene_configuration_provider_result.v1.json"]
    provider.update(ids)
    seal(provider)
    p = auth.PREFIX
    authored = seal({"run_id": ids["run_id"], "object_id": "blue-object"}, "request_digest")
    files[p + "authoring/request.json"] = authored
    files[p + "authoring/failure.json"] = {"request_digest": authored["request_digest"],
        "status": "blocked", "exception_type": "AuthenticationError",
        "blocker": "Error code: 401 - {'code': 'expired_secret_key'}"}
    files[p + "inference/inference_reservations/reserved/one.json"] = seal({
        "run_id": ids["run_id"], "capability": "blue-object_source_analysis", "max_turns": 1,
        "billing_status": "worst_case_reserved_before_provider_call", "projected_max_cost_usd": 1.42717},
        "inference_reservation_digest")
    scope = seal({"candidate_id": ids["run_id"], "api_key_id": "old-key", "project_id": "project",
                  "reserved_max_cost_usd": 5}, "cost_reservation_digest")
    official = seal({"run_id": ids["run_id"], "maximum_cost_usd": 5,
        "request_digest": "request", "candidate_digest": "candidate", "authorization_receipt_digest": "authority",
        "provider_reservation": scope}, "reservation_receipt_digest")
    files[p + "official_openai_cost/openai_official_cost_run_reservation.v1.json"] = official
    files[p + "official_openai_cost/openai_official_cost_run_completion.v1.json"] = seal({
        **{k: v for k, v in official.items() if k != "provider_reservation"},
        "cost_reservation_digest": scope["cost_reservation_digest"],
        "provider_call_performed": True, "runtime_exception_type": "AuthenticationError", "runtime_result_digest": None,
        "official_completion_snapshot": seal({"api_key_id": "old-key", "project_id": "project"}, "openai_cost_snapshot_digest")},
        "completion_receipt_digest")
    return files, result


def test_rejected_call_retains_its_entire_estimated_ceiling(tmp_path):
    files, result = failed_request(tmp_path)
    assert auth.initial_authentication_failure(archive(files, result)) == {
        "api_key_id": "old-key", "project_id": "project", "retained_spend_usd": 1.42717,
        "official_billing_final": False}


@pytest.mark.parametrize("fault", ["timeout", "completed_phase", "second_request", "later_stage", "paid_dependency",
    "foreign_request", "foreign_scope", "foreign_completion", "tamper", "duplicate", "unsealed_request", "exclusion",
    "provider_mutation", "pretraining", "adoption", "cost_above_cap", "cost_nan"])
def test_ambiguous_or_paid_work_does_not_release_allowance(tmp_path, fault):
    files, result = failed_request(tmp_path)
    p = auth.PREFIX
    if fault == "timeout":
        files[p + "authoring/failure.json"]["exception_type"] = "TimeoutError"
    elif fault == "completed_phase":
        files[p + "authoring/source_analysis.json"] = {}
    elif fault == "second_request":
        files[p + "inference/inference_reservations/reserved/two.json"] = {}
    elif fault == "later_stage":
        files["stages/stage-4/input.json"] = {}
    elif fault == "paid_dependency":
        dep = files["stages/stage-3/producer/dependency_results.v1.json"][0]
        dep["paid_execution_requested"] = True
        seal(dep, "stage_result_digest")
    elif fault in {"foreign_request", "unsealed_request"}:
        value = files[p + "authoring/request.json"]
        value["run_id"] = "other"
        if fault == "foreign_request":
            seal(value, "request_digest")
    elif fault in {"foreign_scope", "foreign_completion"}:
        value = files[p + "official_openai_cost/openai_official_cost_run_completion.v1.json"]
        if fault == "foreign_scope":
            snapshot = value["official_completion_snapshot"]
            snapshot["api_key_id"] = "another-key"
            seal(snapshot, "openai_cost_snapshot_digest")
        else:
            value["request_digest"] = "another-request"
        seal(value, "completion_receipt_digest")
    elif fault == "exclusion":
        files["provider_output_zip_exclusions.json"]["excluded_directory_names"].append("inference")
    elif fault == "provider_mutation":
        result["provider_mutations_performed"] = 1
    elif fault == "pretraining":
        result["api_pretraining"] = {"status": "completed"}
    elif fault == "adoption":
        files["stages/stage-3/producer/stage_production_input.v1.json"]["configuration"]["astra_phase_adoption"] = {}
    elif fault.startswith("cost_"):
        value = files[p + "inference/inference_reservations/reserved/one.json"]
        value["projected_max_cost_usd"] = "NaN" if fault == "cost_nan" else 6
        seal(value, "inference_reservation_digest")
    archive(files, result)
    if fault == "tamper":
        Path(result["provider_runtime_output_zip_path"]).write_bytes(b"changed")
    elif fault == "duplicate":
        import hashlib
        import zipfile
        with pytest.warns(UserWarning), zipfile.ZipFile(result["provider_runtime_output_zip_path"], "a") as z:
            z.writestr(p + "authoring/request.json", json.dumps(files[p + "authoring/request.json"]))
        result["provider_runtime_output_zip_sha256"] = "sha256:" + hashlib.sha256(Path(result["provider_runtime_output_zip_path"]).read_bytes()).hexdigest()
        seal(result)
    assert auth.initial_authentication_failure(result) is None


def test_key_must_change_in_same_project_and_authenticate(tmp_path, monkeypatch):
    files, result = failed_request(tmp_path)
    archive(files, result)
    calls = []
    monkeypatch.setattr(auth, "authenticate_key", calls.append)
    env = {"OPENAI_PROJECT_ID": "project", "OPENAI_CONTENT_AGENTS_API_KEY_ID": "old-key",
           "OPENAI_CONTENT_AGENTS_API_KEY_FILE": "/protected/new-key"}
    assert auth.replacement_admission(result, env)["status"] == "blocked"
    assert not calls
    env["OPENAI_CONTENT_AGENTS_API_KEY_ID"] = "replacement-key"
    assert auth.replacement_admission(result, env)["status"] == "admitted"
    assert calls == ["/protected/new-key"]
    env["OPENAI_PROJECT_ID"] = "foreign-project"
    assert auth.replacement_admission(result, env)["status"] == "blocked"
    env["OPENAI_PROJECT_ID"] = "project"
    def refuse(_path):
        raise ValueError("openai_key_authentication_http_401")
    monkeypatch.setattr(auth, "authenticate_key", refuse)
    assert auth.replacement_admission(result, env)["blockers"] == ["openai_key_authentication_http_401"]


def test_authentication_uses_read_only_endpoint_and_sanitizes_failures(tmp_path):
    key = tmp_path / "key"
    key.write_text("test-secret")
    def opener(request, **kwargs):
        assert request.full_url == "https://api.openai.com/v1/models"
        assert request.get_method() == "GET"
        assert request.headers["Authorization"] == "Bearer test-secret"
        return io.StringIO('{"data": []}')
    auth.authenticate_key(key, opener=opener)
    def denied(*args, **kwargs):
        raise urllib.error.HTTPError("url", 401, "test-secret", {}, None)
    with pytest.raises(ValueError, match="^openai_key_authentication_http_401$"):
        auth.authenticate_key(key, opener=denied)
    with pytest.raises(ValueError, match="^openai_key_authentication_unavailable$"):
        auth.authenticate_key(tmp_path / "missing", opener=opener)


def test_terminal_settlement_retains_rejected_call_ceiling_with_real_archive(tmp_path, monkeypatch):
    from tests.test_terminal_scene_attempt_settlement import (
        _website_preallocation_failure, _write, _settle, _reserve,
        validated_cancellation, settlement,
    )
    fx, receipt, result_path, _ = _website_preallocation_failure(tmp_path, monkeypatch)
    old = json.loads(result_path.read_text())
    files, result = failed_request(tmp_path, {k: old[k] for k in ('run_id', 'source_commit')})
    result.update(schema_version=old['schema_version'])
    _write(result_path, archive(files, result))
    launch_path = Path(receipt['execution_terminal']['launch_receipt']['path'])
    launch = json.loads(launch_path.read_text())
    launch['terminal_evidence']['result'] = {**settlement._file(result_path), 'exists': True}
    _write(launch_path, launch)
    (fx['directory'] / 'cancelled-unstarted-controls' / (receipt['attempt_id'] + '.json')).unlink()
    _settle(fx, source_factory=fx['factory'])
    attempt = json.loads((fx['directory'] / 'attempts' / (receipt['attempt_id'] + '.json')).read_text())
    receipt = validated_cancellation(fx['directory'], attempt)
    assert settlement.budget_retained_hold(receipt) == {
        'basis': 'rejected_initial_authoring_request_upper_bound',
        'retained_spend_usd': 1.42717, 'counts_as_attempt': False}
    assert _reserve(fx['root'], fx['intent'], 'successor', 17, now=300)['status'] == 'reserved'
    # Missing evidence restores the conservative full hold; it never creates credit.
    Path(result['provider_runtime_output_zip_path']).unlink()
    assert settlement.budget_retained_hold(receipt)['retained_spend_usd'] == 16.76


def test_recovery_revalidates_replacement_key_and_complete_launch_identity(tmp_path, monkeypatch):
    from tests.test_task_evaluation_scene_capacity_recovery import capacity_fixture, write
    from blueprint_pipeline import task_evaluation_scene_capacity_recovery as capacity
    from blueprint_pipeline.task_evaluation_retained_controls_evidence import _file
    _, first, observed, config, _ = capacity_fixture(tmp_path, monkeypatch)
    refs = observed['records']
    old = observed['values']['result']
    files, result = failed_request(tmp_path, {k: old[k] for k in ('run_id', 'source_commit')})
    result.update({k: old[k] for k in ('schema_version', 'bundle_sha256', 'authority_digest', 'provider_output_disk_requirements')})
    result['blockers'] = ['authoring_expired_key']
    refs['result'] = write(Path(refs['result']['path']), archive(files, result), 'result_digest')
    launch_path = Path(refs['launch']['path'])
    teardown_path = launch_path.parent / 'teardown.json'
    write(teardown_path, {'schema_version': 'vast_teardown_manifest.v1',
        'status': 'not_required_provider_adapter_never_invoked', 'vast_instance_ids': [],
        'continuing_spend_from_this_run': False})
    launch = observed['values']['launch']
    launch['terminal_evidence'] = {'artifacts': {'teardown_manifest_path': {**_file(teardown_path), 'exists': True}}}
    refs['launch'] = write(launch_path, launch, 'receipt_digest')
    zero = observed['values']['zero']
    zero['receipt_digest'] = launch['receipt_digest']
    refs['zero'] = write(Path(refs['zero']['path']), zero, 'provider_zero_receipt_digest')
    values = capacity.validate_source(refs, prior_attempt=first, kind=auth.KIND)
    observation = {'values': values, 'kind': auth.KIND}
    monkeypatch.setenv('OPENAI_CONTENT_AGENTS_API_KEY_ID', 'replacement')
    monkeypatch.setenv('OPENAI_PROJECT_ID', 'project')
    monkeypatch.setenv('OPENAI_CONTENT_AGENTS_API_KEY_FILE', '/test/reference')
    calls = []
    monkeypatch.setattr(auth, 'authenticate_key', lambda path: calls.append(path))
    assert capacity.capacity_admission(observation, config, 103)['status'] == 'admitted'
    assert calls == ['/test/reference']
    monkeypatch.setenv('OPENAI_CONTENT_AGENTS_API_KEY_ID', 'old-key')
    assert capacity.capacity_admission(observation, config, 104)['status'] == 'waiting_for_capacity'
