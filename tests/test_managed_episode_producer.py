"""The closeout producer queues once, then attaches the independently sealed result."""
import json
import time

import pytest

from blueprint_pipeline.agent_execution.episode_producer import schedule_episode_batch
from blueprint_pipeline.agent_execution.episode_tasks import interpreter_identity, ROLES
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.episode_interpretation_batch_authority import validate_episode_interpretation_batch_authority_shape
from tests.test_agent_episode_tasks import setup
from tests.test_episode_interpretation import _request, _output


def test_producer_waits_for_worker_and_publishes_receipt_without_repeating_inference(tmp_path):
    service, initial, api, data = setup(tmp_path)
    request = _request(data)
    profile = {"schema_version": "policy_canary_episode_interpreter_profile.v2", "status": "configured",
        "runtime": "openai_agents_api", "model": initial.task.model, "max_cost_usd": 2, "per_episode_budget_usd": 1}
    profile["profile_digest"] = canonical_digest(profile)
    policy = json.loads((tmp_path / "episode-rights.json").read_text())["agent_runtime_policy"]
    authority = {"schema_version": "policy_canary_episode_interpretation_batch_authority.v2", "status": "approved",
        "run_id": "batch-run", "interpreter": interpreter_identity("openai_agents_api", initial.task.model).__dict__,
        "interpreter_profile_digest": profile["profile_digest"], "allowed_artifact_roles": list(ROLES),
        "external_disclosure_authorized": True, "provider_training_authorized": False, "public_redistribution_authorized": False,
        "maximum_cost_usd": 2, "source_rights_admission_digest": canonical_digest({"owned_fixture": True}),
        "accepted_by": "fixture-owner", "accepted_on": "2026-09-10", "authority_reference": "fixture-only",
        "agent_runtime_policy": policy, "maximum_episodes": 2, "expires_at": time.time() + 1800}
    authority["authority_digest"] = canonical_digest(authority)
    result = {"run_id": "batch-run", "episodes": [{"episode": {"episode_id": request.episode_id},
        "candidate_id": request.candidate_policy_id}], "artifact_inventory": [], "task_succeeded": False}
    original = json.loads(json.dumps(result))
    kwargs = dict(requests=[({}, request)], result=result, evidence_root=data["root"], profile=profile, authority=authority, service=service)
    pending = schedule_episode_batch(**kwargs)
    assert pending["episode_interpretation"]["pending_count"] == 1
    assert not api.calls and result == original
    task_id = pending["episode_interpretation"]["receipts"][0]["task_id"]
    record = service.record(task_id)
    service.enqueue(task_id, "blueprint-webapp")
    runtime = service.runtime_for_task(record.task)
    runtime.transport = api
    runtime.step(task_id)
    calls = [("read_episode_context", {}),
        ("read_episode_trace", {"role": "state_trace", "start_step": 0, "end_step": 20, "limit": 100}),
        ("read_episode_trace", {"role": "contact_force_trace", "start_step": 0, "end_step": 20, "limit": 100}),
        ("inspect_episode_interval", {"start_seconds": 0, "end_seconds": 1, "max_observations": 8})]
    api.actions = [{"type": "function_call", "turn_id": "turn_1", "call_id": f"call_{i}", "name": name, "arguments": args}
                   for i, (name, args) in enumerate(calls)]
    runtime.step(task_id)
    api.actions = []; api.output = _output(data).model_dump(mode="json"); api.turn_status = "completed"
    runtime.step(task_id)
    before = len(api.calls)
    finished = schedule_episode_batch(**kwargs)
    assert finished["episode_interpretation"]["pending_count"] == 0
    assert finished["episodes"][0]["evidence_artifacts"]["episode_interpretation"]
    assert finished["task_succeeded"] is False and len(api.calls) == before and result == original
    assert schedule_episode_batch(**kwargs)["episode_interpretation"]["receipts"] == finished["episode_interpretation"]["receipts"]
    bad = {**authority, "schema_version": "policy_canary_episode_interpretation_batch_authority.v1"}
    bad["authority_digest"] = canonical_digest(bad, digest_field="authority_digest")
    with pytest.raises(ValueError, match="batch_authority_invalid"):
        validate_episode_interpretation_batch_authority_shape(bad)
