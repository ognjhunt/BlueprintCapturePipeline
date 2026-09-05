from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_sam31_preparation_execution as execution
from blueprint_pipeline import task_evaluation_launch_preparation_worker as worker
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_preparation_contract import launch_preparation_request_digest
from blueprint_pipeline.task_evaluation_launch_preparation_queue import stage_launch_preparation_request
from tests.test_task_evaluation_launch_preparation_worker import (
    SERVICE_ACCOUNT, _rebind_recipe, fetcher, production_request_with_fetchable_bytes,
)


def _ref(path):
    return {"path": str(path), "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "scope"
    root.mkdir()
    request, payloads = production_request_with_fetchable_bytes()
    plan_data = json.dumps({"schema_version": "test_plan.v1", "reviewer_kind": "ai"}).encode()
    plan_uri = "s3://blueprint-production-inputs/plan.json"
    plan_ref = {"uri": plan_uri, "digest": "sha256:" + hashlib.sha256(plan_data).hexdigest(),
                "size_bytes": len(plan_data)}
    payloads[plan_uri] = plan_data
    request["runtime"]["mounts"].append({"source": plan_ref, "container_path": "/inputs/sam31-plan.json",
                                         "mode": "read_only"})
    recipe = json.loads(payloads[request["construction"]["recipe"]["uri"]])
    stage_ref = recipe["stage_sequence"][0]["configuration"]
    stage = {"required_views": {"mask_source": "sam31_reviewed_calibrated_object_masks"},
             "sam31_review_kind": "ai", "sam31_preparation_plan": plan_ref}
    data = json.dumps(stage).encode()
    payloads[stage_ref["uri"]] = data
    stage_ref.update(digest="sha256:" + hashlib.sha256(data).hexdigest(), size_bytes=len(data))
    _rebind_recipe(request, payloads, recipe)
    parent_queue, input_root = root / "parent", root / "inputs"
    stage_launch_preparation_request(value=request, queue_root=parent_queue, submitted_by="blueprint-webapp")
    worker.process_launch_preparation_queue(
        queue_root=parent_queue, input_root=input_root,
        allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=SERVICE_ACCOUNT, source_commit=request["expected_production_commit"],
        fetcher=fetcher(payloads),
        sam31_preparation_advancer=lambda context: {"status": "waiting_for_child", "evidence_refs": []},
    )
    plan_path = input_root / "content-addressed" / "sha256" / plan_ref["digest"][7:]
    source = root / "source.json"
    source.write_text('{"source":"immutable"}')
    queue = root / "children"
    args = {"queue_root": queue, "parent_preparation_id": request["preparation_id"],
            "parent_request_digest": launch_preparation_request_digest(request),
            "expected_source_commit": request["expected_production_commit"], "plan_ref": _ref(plan_path),
            "phase": "source_selections", "inputs": {"source": _ref(source)}}
    monkeypatch.setattr(execution, "_verified_checkout_head", lambda: request["expected_production_commit"])
    process = {"queue_root": queue, "parent_queue_root": parent_queue,
               "preparation_input_root": input_root, "execution_root": root / "outputs",
               "approved_roots": (root,)}
    return root, args, process


def _complete(context):
    output = Path(context["output_root"]) / "result.json"
    output.write_text(json.dumps({"child_id": context["child_id"], "phase": context["phase"]}))
    return {"status": "completed", "artifacts": {"stage_result": _ref(output)}}


def _mutate_job(intake, **changes):
    path = Path(intake["job_path"])
    job = json.loads(path.read_text())
    job.update(changes)
    job["job_digest"] = canonical_digest(job, digest_field="job_digest")
    path.chmod(0o600)
    path.write_text(json.dumps(job))


def test_exact_enqueue_is_idempotent_and_complete_phase_wakes_parent(setup):
    _, args, process = setup
    first = execution.enqueue_sam31_phase(**args)
    again = execution.enqueue_sam31_phase(**args)
    assert again["status"] == "already_exists" and again["child_id"] == first["child_id"]
    calls = []
    result = execution.process_sam31_phase_queue(
        **process, phase_executor=lambda context: calls.append(context) or _complete(context))
    assert result["results"][0]["status"] == "completed"
    assert len(calls) == 1 and calls[0]["resume_only"] is False
    assert calls[0]["phase"] in execution.PHASES
    assert result["parent_wakeups"] == [first["child_id"]]
    assert json.loads(Path(first["result_path"]).read_text())["artifacts"]["stage_result"]
    assert execution.enqueue_sam31_phase(**args)["status"] == "already_exists"
    assert execution.process_sam31_phase_queue(
        **process, phase_executor=lambda context: pytest.fail("Must not run completed job"),
    )["results"] == []


def test_waiting_external_rechecks_same_child_and_marks_resume_only(setup):
    _, args, process = setup
    intake = execution.enqueue_sam31_phase(**args)
    calls = []
    def execute(context):
        calls.append((context["child_id"], context["resume_only"]))
        if not context["resume_only"]:
            return {"status": "waiting_for_external_result", "artifacts": {}, "launch_id": context["child_id"]}
        assert context["previous_progress"]["executor_result"]["launch_id"] == context["child_id"]
        return _complete(context)
    first = execution.process_sam31_phase_queue(**process, phase_executor=execute)
    assert first["results"][0]["status"] == "waiting_for_external_result"
    assert not Path(intake["result_path"]).exists()
    second = execution.process_sam31_phase_queue(**process, phase_executor=execute)
    assert second["results"][0]["status"] == "completed"
    assert calls == [(intake["child_id"], False), (intake["child_id"], True)]


def test_crash_after_retained_result_does_not_execute_phase_again(setup):
    _, args, process = setup
    intake = execution.enqueue_sam31_phase(**args)
    execution.process_sam31_phase_queue(**process, phase_executor=_complete)
    root = Path(args["queue_root"])
    name = intake["child_id"] + ".json"
    os.replace(root / "completed" / name, root / "processing" / name)
    result = execution.process_sam31_phase_queue(
        **process, phase_executor=lambda context: pytest.fail("Retained result prevents redispatch"),
    )
    assert result["results"][0]["status"] == "completed"


def test_fast_child_retains_wakeup_until_real_parent_progress_exists(setup):
    root, args, process = setup
    intake = execution.enqueue_sam31_phase(**args)
    parent = Path(process["parent_queue_root"])
    waiting = next((parent / "awaiting_source_preparation").glob("*.json"))
    os.replace(waiting, parent / "processing" / waiting.name)
    os.replace(parent / "source-progress", root / "saved-progress")
    result = execution.process_sam31_phase_queue(**process, phase_executor=_complete)
    assert result["parent_wakeups"] == []
    assert (Path(args["queue_root"]) / "wake-pending" / (intake["child_id"] + ".json")).exists()
    os.replace(root / "saved-progress", parent / "source-progress")
    os.replace(parent / "processing" / waiting.name, waiting)
    result = execution.process_sam31_phase_queue(
        **process, phase_executor=lambda context: pytest.fail("Only wakeup retries"),
    )
    assert result["parent_wakeups"] == [intake["child_id"]]
    signals = list((parent / "source-resume-pending").glob("*.json"))
    assert len(signals) == 1 and json.loads(signals[0].read_text())["progress_digest"].startswith("sha256:")


@pytest.mark.parametrize("mutation", ["argv", "phase", "input_bytes", "plan_path"])
def test_changed_jobs_and_unapproved_paths_never_reach_phase_executor(setup, mutation, tmp_path):
    _, args, process = setup
    intake = execution.enqueue_sam31_phase(**args)
    if mutation == "argv":
        _mutate_job(intake, argv=["python", "untrusted.py"])
    elif mutation == "phase":
        _mutate_job(intake, phase="allocate_any_gpu")
    elif mutation == "input_bytes":
        Path(args["inputs"]["source"]["path"]).write_text("changed")
    else:
        outside = tmp_path / "outside.json"
        outside.write_bytes(Path(args["plan_ref"]["path"]).read_bytes())
        _mutate_job(intake, plan_ref={**args["plan_ref"], "path": str(outside)})
    result = execution.process_sam31_phase_queue(
        **process, phase_executor=lambda context: pytest.fail("Invalid jobs cannot execute"),
    )
    assert result["results"][0]["status"] == "failed"
    receipt = json.loads(Path(intake["result_path"]).read_text())
    assert receipt["status"] == "failed" and receipt["artifacts"] == {}


def test_stale_source_commit_is_retained_as_failure_without_execution(setup, monkeypatch):
    _, args, process = setup
    intake = execution.enqueue_sam31_phase(**args)
    monkeypatch.setattr(execution, "_verified_checkout_head", lambda: "b" * 40)
    result = execution.process_sam31_phase_queue(
        **process, phase_executor=lambda context: pytest.fail("Wrong source commit"),
    )
    assert result["results"][0]["status"] == "failed"
    assert json.loads(Path(intake["result_path"]).read_text())["blocker"] == "sam31_phase_source_commit_mismatch"


def test_unknown_phase_is_rejected_before_enqueue(setup):
    _, args, _ = setup
    with pytest.raises(execution.Sam31PhaseExecutionError, match="phase_or_inputs_invalid"):
        execution.enqueue_sam31_phase(**{**args, "phase": "generic_shell"})


def test_completed_claim_requires_named_verified_artifacts(setup):
    _, args, process = setup
    intake = execution.enqueue_sam31_phase(**args)
    result = execution.process_sam31_phase_queue(
        **process, phase_executor=lambda context: {"status": "completed", "artifacts": {}},
    )
    assert result["results"][0]["status"] == "failed"
    assert json.loads(Path(intake["result_path"]).read_text())["blocker"] == "sam31_phase_completed_artifacts_missing"

def test_external_progress_artifacts_are_reverified_before_rechecking_handler(setup):
    _, args, process = setup
    intake = execution.enqueue_sam31_phase(**args)
    def waiting(context):
        path = Path(context["output_root"]) / "external.json"
        path.write_text('{"launch_id":"stable"}')
        return {"status": "waiting_for_external_result", "artifacts": {"external_status": _ref(path)}}
    execution.process_sam31_phase_queue(**process, phase_executor=waiting)
    progress_path = next((Path(args["queue_root"]) / "progress").rglob("*.json"))
    ref = json.loads(progress_path.read_text())["executor_result"]["artifacts"]["external_status"]
    Path(ref["path"]).write_text("changed")
    result = execution.process_sam31_phase_queue(
        **process, phase_executor=lambda context: pytest.fail("Changed external receipt"),
    )
    assert result["results"][0]["status"] == "failed"
    assert json.loads(Path(intake["result_path"]).read_text())["status"] == "failed"


def test_deployer_installs_and_authority_gates_exact_child_units():
    import ast

    repository = Path(__file__).resolve().parents[1]
    tree = ast.parse((repository / "scripts/deploy_control_plane_commit.py").read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {
                "DEFAULT_DEPLOYED_SYSTEMD_UNITS", "DEFAULT_ALWAYS_ARM_AUTHORITY_GATED_PATH_UNITS",
                "DEFAULT_ALWAYS_ARM_PATH_UNITS", "DEFAULT_ALWAYS_ARM_TIMER_UNITS",
            } and isinstance(node.value, ast.Tuple):
                constants[node.targets[0].id] = ast.literal_eval(node.value)
    stem = "blueprint-task-evaluation-sam31-preparation-execution"
    for suffix in ("service", "path", "timer"):
        name = f"{stem}.{suffix}"
        assert name in constants["DEFAULT_DEPLOYED_SYSTEMD_UNITS"]
        assert (repository / "deploy/systemd" / name).is_file()
    assert f"{stem}.path" in constants["DEFAULT_ALWAYS_ARM_AUTHORITY_GATED_PATH_UNITS"]
    assert f"{stem}.path" not in constants["DEFAULT_ALWAYS_ARM_PATH_UNITS"]
    assert f"{stem}.timer" in constants["DEFAULT_ALWAYS_ARM_TIMER_UNITS"]
    service = (repository / "deploy/systemd" / f"{stem}.service").read_text()
    assert "-m blueprint_pipeline.task_evaluation_sam31_preparation_execution" in service
    assert "--max-messages 1" in service
    assert "paid_resource_allocator" not in service
    assert "TimeoutStartSec=75m" in service


def test_real_source_selection_stage_crosses_outer_queue_with_canonical_frame_reference(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_sam31_preparation_stages as stages
    from tests.test_task_evaluation_sam31_preparation_cpu_stages import _fixture as source_fixture
    root = tmp_path/'real-source-chain'
    root.mkdir()
    source_root = root/'source'
    source_root.mkdir()
    source_job = source_fixture(source_root)
    commit = source_job['request']['expected_production_commit']
    profile = {'schema_version':stages.PROFILE_SCHEMA, 'source_commit':commit,
               'repo_root':source_job['repo_root'], 'server_data_root':str(root),
               'runtime_root':source_job['runtime_root']}
    profile['profile_digest'] = canonical_digest(profile, digest_field='profile_digest')
    profile_path = root/'profile.json'
    profile_path.write_text(json.dumps(profile))
    monkeypatch.setenv(stages.PROFILE_ENV, str(profile_path))
    plan = {**source_job['plan'], 'server_profile_sha256':_ref(profile_path)['sha256']}
    plan_data = json.dumps(plan).encode()
    plan_uri = 's3://blueprint-production-inputs/real-source-plan.json'
    plan_ref = {'uri':plan_uri, 'digest':'sha256:'+hashlib.sha256(plan_data).hexdigest(),
                'size_bytes':len(plan_data)}
    request, payloads = production_request_with_fetchable_bytes()
    request['expected_production_commit'] = commit
    payloads[plan_uri] = plan_data
    request['runtime']['mounts'].append({'source':plan_ref, 'container_path':'/inputs/sam31-plan.json',
                                       'mode':'read_only'})
    recipe = json.loads(payloads[request['construction']['recipe']['uri']])
    stage_ref = recipe['stage_sequence'][0]['configuration']
    stage = {'required_views':{'mask_source':'sam31_reviewed_calibrated_object_masks'},
             'sam31_review_kind':'ai', 'sam31_preparation_plan':plan_ref}
    data = json.dumps(stage).encode()
    payloads[stage_ref['uri']] = data
    stage_ref.update(digest='sha256:'+hashlib.sha256(data).hexdigest(), size_bytes=len(data))
    _rebind_recipe(request, payloads, recipe)
    parent, inputs, queue = root/'parents', root/'inputs', root/'children'
    stage_launch_preparation_request(value=request, queue_root=parent, submitted_by='fixture-webapp')
    worker.process_launch_preparation_queue(queue_root=parent, input_root=inputs,
        allowed_uri_prefixes=['s3://blueprint-production-inputs/'], service_account=SERVICE_ACCOUNT,
        source_commit=commit, fetcher=fetcher(payloads),
        sam31_preparation_advancer=lambda _: {'status':'waiting_for_child', 'evidence_refs':[]})
    plan_path = inputs/'content-addressed/sha256'/plan_ref['digest'][7:]
    intake = execution.enqueue_sam31_phase(queue_root=queue, parent_preparation_id=request['preparation_id'],
        parent_request_digest=launch_preparation_request_digest(request), expected_source_commit=commit,
        plan_ref=_ref(plan_path), phase='source_selections', inputs=plan['host_inputs'])
    source_preparation = Path(plan['host_inputs']['source_preparation_receipt']['path'])
    frame = source_preparation.parent/'shared_frame_candidate.json'
    original = frame.read_bytes()
    assert json.loads(original)['receipt_digest']
    monkeypatch.setattr(execution, '_verified_checkout_head', lambda: commit)
    result = execution.process_sam31_phase_queue(queue_root=queue, parent_queue_root=parent,
        preparation_input_root=inputs, execution_root=root/'executions', approved_roots=(root,))
    assert result['results'][0]['status'] == 'completed', Path(intake['result_path']).read_text()
    sealed = json.loads(Path(intake['result_path']).read_text())
    ref = sealed['artifacts']['registered_frame']
    assert set(ref) == {'path','sha256','size_bytes'}
    assert Path(ref['path']).read_bytes() == original
    assert ref == _ref(frame)
    assert json.loads(frame.read_text())['receipt_digest'] == json.loads(original)['receipt_digest']
