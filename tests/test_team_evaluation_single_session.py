"""ADP-009D/day-28 runtime precursor: 200 isolated cases inside one lease.

Provider/Isaac are stand-ins here. The actual session validator, parent worker,
per-case sealing, aggregation and first-batch spending gate execute unchanged.
The existing lifecycle rehearsal covers the real episode/media/client path.
"""
import copy
import json
import hashlib

import pytest

from blueprint_pipeline import native_task_arena_policy_canary_worker as worker
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_session import (
    PROVIDER_RESULT_FILENAME, build_session_authority, validate_session_authority,
)
from tests import test_team_candidate_quick10_rehearsal as rehearsal
from tests.test_team_candidate_quick10_launch import _provider_session_result


def _campaign(tmp_path, monkeypatch, *, interface_failures=0, missing_at=None):
    runtime, output, inputs, _admission = rehearsal._stage_team_runtime(tmp_path)
    templates = inputs["cells"]
    inputs["cells"] = [dict(copy.deepcopy(templates[i % len(templates)]),
                            cell_id=f"single-session-{i:03d}", seed=1000 + i)
                       for i in range(200)]
    activation_path = runtime / "runtime_inputs/task_evaluation_policy_campaign_activation.v1.json"
    activation = json.loads(activation_path.read_text())
    activation["campaign_units"] = [dict(activation["campaign_units"][0],
                                        campaign_unit_id=cell["cell_id"])
                                    for cell in inputs["cells"]]
    activation["campaign_unit_count"] = 200
    activation["activation_digest"] = canonical_digest(activation, digest_field="activation_digest")
    activation_path.write_text(json.dumps(activation))
    inputs["activation_digest"] = activation["activation_digest"]
    inputs["runtime_inputs_digest"] = canonical_digest(inputs, digest_field="runtime_inputs_digest")
    input_path = runtime / "runtime_inputs/policy_canary_runtime_inputs.json"
    input_path.write_text(json.dumps(inputs))
    def record(path):
        return {"path": str(path), "size_bytes": path.stat().st_size,
                "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}
    authority = build_session_authority(activation_manifest=activation, activation_record=record(activation_path),
        runtime_inputs=inputs, runtime_input_record=record(input_path),
        resource_name="blueprint-native-task-policy-canary-" + "1" * 32,
        hard_cap_usd=20, hard_ttl_seconds=57600, hard_deadline_epoch=4_000_000_000.)
    (runtime / "runtime_inputs/policy_canary_session_authority.json").write_text(json.dumps(authority))
    provider_path = runtime / "adp_arena_provider_manifest.json"
    provider = json.loads(provider_path.read_text())
    provider.update(authority_digest=authority["authority_digest"],
                    runtime_inputs_digest=inputs["runtime_inputs_digest"],
                    episodes_per_policy=200, learned_policy_rollout_count=200)
    provider["input_digest"] = canonical_digest(provider, digest_field="input_digest")
    provider_path.write_text(json.dumps(provider))
    spec = json.loads(next((runtime / "runtime_inputs").glob("policy_execution_spec.*.json")).read_text())
    rows = _provider_session_result(runtime, spec)["episodes"]
    seen = []
    monkeypatch.setattr(worker, "_write_policy_canary_telemetry", lambda *args, **kw: ({}, []))

    def spawn(*, index, child_root, **kwargs):
        assert index == len(seen)  # no case replay or simulator reuse
        if index:
            progress = json.loads((output / "team_evaluation_checkpoint.json").read_text())
            assert len(progress["completed_child_results"]) == index
            assert progress["result_digest"] == canonical_digest(progress, digest_field="result_digest")
            assert progress["authority_digest"] == authority["authority_digest"]
            assert progress["completed_attempt_replay_authorized"] is False
        seen.append(index)
        if index == missing_at:
            return 1
        row = dict(rows[index], family=inputs["cells"][index]["family"])
        if index < interface_failures:
            row.update(status="failed", episode={}, candidate_policy_query_attempted=True,
                       policy_response_status="timeout", failure_message="team_policy_https_timeout",
                       episode_failure_stage="after_first_observation", first_observation_retained=True,
                       policy_outcome_interpretable=False,
                       visual_evidence={"status": "complete", "human_review_available": True,
                                        "episode_terminal_status": "failed_after_first_observation",
                                        "terminal_observation_invented": False})
        else:
            row.update(status="completed", scoring_authority="deterministic_simulator_state",
                       episode={"score": {"status": "scored", "task_succeeded": False}})
        child = {"schema_version": "native_task_arena_policy_canary_session_result.v1",
                 "selected_cell_index": index, "status": "runtime_selected_cell_completed_pending_aggregation",
                 "task_success_contract": inputs["task_success_contract"],
                 "task_success_contract_digest": inputs["task_success_contract_digest"], "episodes": [row]}
        child["result_digest"] = canonical_digest(child, digest_field="result_digest")
        worker._seal_result(result_path=child_root / PROVIDER_RESULT_FILENAME, result=child)
        return 0

    return runtime, output, inputs, authority, seen, spawn


@pytest.mark.parametrize("failures, expected_cells", [(0, 200), (12, 200), (13, 25)])
def test_one_session_runs_the_full_matrix_or_stops_at_the_original_interface_gate(
        tmp_path, monkeypatch, failures, expected_cells):
    runtime, output, inputs, authority, seen, spawn = _campaign(
        tmp_path, monkeypatch, interface_failures=failures)
    rc = worker._run_isolated_cell_processes(runtime_root=runtime, output_root=output, run_cell_process=spawn)
    assert seen == list(range(expected_cells))
    assert authority["maximum_provider_allocations"] == 1 and authority["retry_cap"] == 0
    result = json.loads((output / PROVIDER_RESULT_FILENAME).read_text())
    assert len(result["episodes"]) == expected_cells
    assert len({row["cell_id"] for row in result["episodes"]}) == expected_cells
    assert result["warm_session_open_count"] == 1
    assert result["result_digest"] == canonical_digest(result, digest_field="result_digest")
    assert rc == (1 if failures == 13 else 0)
    if failures == 13:
        assert result["stopped_before_next_cell"] is True
        assert result["planned_episodes_per_policy"] == 200
        assert result["blockers"] == ["team_policy_interface_failures:25_episodes:policy_service_did_not_answer=13"]


def test_lost_child_preserves_completed_prefix_without_starting_another_case(tmp_path, monkeypatch):
    runtime, output, _inputs, _authority, seen, spawn = _campaign(tmp_path, monkeypatch, missing_at=30)
    with pytest.raises(RuntimeError, match="isolated_cell_result_missing:30"):
        worker._run_isolated_cell_processes(runtime_root=runtime, output_root=output, run_cell_process=spawn)
    result = json.loads((output / PROVIDER_RESULT_FILENAME).read_text())
    assert seen == list(range(31)) and len(result["episodes"]) == 30
    assert result["status"] == "blocked" and result["stopped_before_next_cell"] is False
    assert result["result_digest"] == canonical_digest(result, digest_field="result_digest")
    assert len(list((output / "cell_runs").glob("*/" + PROVIDER_RESULT_FILENAME))) == 30


@pytest.mark.parametrize("close_at", [0, 3])
def test_close_timeout_retains_completed_current_child_and_earlier_prefix(tmp_path, monkeypatch, close_at):
    import subprocess

    runtime, output, _inputs, _authority, seen, spawn = _campaign(tmp_path, monkeypatch)

    def close_timeout(**kwargs):
        exit_code = spawn(**kwargs)
        if kwargs["index"] == close_at:
            raise subprocess.TimeoutExpired("isaac-close", 60)
        return exit_code

    assert worker._run_isolated_cell_processes(
        runtime_root=runtime, output_root=output, run_cell_process=close_timeout,
    ) == 1
    result = json.loads((output / PROVIDER_RESULT_FILENAME).read_text())
    assert seen == list(range(close_at + 1))
    assert result["status"] == "blocked" and len(result["episodes"]) == close_at + 1
    assert result["episodes"][-1]["episode"]["score"]["status"] == "scored"
    assert result["interrupted_cell_index"] == close_at
    assert result["result_digest"] == canonical_digest(result, digest_field="result_digest")
    assert not (output / "cell_runs" / f"{close_at + 1:02d}").exists()


@pytest.mark.parametrize("deadline", [True, float("inf"), -1, "tomorrow"])
def test_authority_refuses_invalid_absolute_deadline(tmp_path, monkeypatch, deadline):
    _runtime, _output, _inputs, authority, _seen, _spawn = _campaign(tmp_path, monkeypatch)
    authority["hard_deadline_epoch"] = deadline
    if deadline != float("inf"):
        authority["authority_digest"] = canonical_digest(authority, digest_field="authority_digest")
    with pytest.raises(ValueError):
        validate_session_authority(authority)


@pytest.mark.parametrize('damage', ['status', 'contract', 'scenario', 'candidate', 'episodes', 'media', 'schema', 'kind', 'claims', 'identity', 'telemetry', 'json'])
def test_malformed_child_stops_before_the_next_paid_case_and_retains_valid_prefix(tmp_path, monkeypatch, damage):
    runtime, output, _inputs, _authority, seen, spawn = _campaign(tmp_path, monkeypatch)

    def damaged_spawn(**kwargs):
        result = spawn(**kwargs)
        if kwargs['index'] == 3:
            path = kwargs['child_root'] / PROVIDER_RESULT_FILENAME
            child = json.loads(path.read_text())
            if damage == 'json':
                malformed = path.with_suffix('.malformed')
                malformed.write_text('{incomplete')
                malformed.replace(path)
                return result
            if damage == 'schema':
                child['schema_version'] = 'invented'
            elif damage == 'status':
                child['status'] = 'invented'
            elif damage == 'contract':
                child['task_success_contract_digest'] = 'sha256:' + '0' * 64
            elif damage == 'scenario':
                child['episodes'][0]['seed'] = -1
            elif damage == 'candidate':
                child['episodes'][0]['candidate_id'] = 'another-policy'
            elif damage == 'episodes':
                child['episodes'] = []
            elif damage == 'telemetry':
                child['episodes'][0]['telemetry'] = 'malformed'
            elif damage == 'kind':
                child['episodes'][0]['run_kind'] = 'internal_policy_canary'
            elif damage == 'claims':
                child['episodes'][0]['ranking_eligible'] = True
            elif damage == 'identity':
                child['episodes'][0]['checkpoint_digest'] = 'sha256:' + '0' * 64
            elif damage == 'media':
                child['episodes'][0]['lossless_frame_manifest_digest'] = 'unbound'
            child['result_digest'] = canonical_digest(child, digest_field='result_digest')
            worker._seal_result(result_path=path, result=child)
        return result

    with pytest.raises((RuntimeError, ValueError)):
        worker._run_isolated_cell_processes(runtime_root=runtime, output_root=output, run_cell_process=damaged_spawn)
    assert seen == list(range(4))
    checkpoint = json.loads((output / 'team_evaluation_checkpoint.json').read_text())
    assert len(checkpoint['completed_child_results']) == 3
    result = json.loads((output / PROVIDER_RESULT_FILENAME).read_text())
    assert result['status'] == 'blocked' and len(result['episodes']) == 3
    assert result['planned_episodes_per_policy'] == 200
    assert (output / 'cell_runs/03' / PROVIDER_RESULT_FILENAME).is_file()


@pytest.mark.parametrize("failure", ["missing", "unreadable", "timeout", "recovery_failed", "all_prepolicy"])
def test_a_prefix_with_only_prepolicy_failures_is_retained_without_invented_episodes(tmp_path, monkeypatch, failure):
    import subprocess
    runtime, output, inputs, _authority, seen, spawn = _campaign(
        tmp_path, monkeypatch, missing_at=3 if failure == "missing" else None)
    from blueprint_pipeline import policy_canary_interrupted_cell_recovery as recovery
    def recover(**kwargs):
        if failure == 'recovery_failed':
            raise RuntimeError('recovery_unavailable')
        return {'candidate_policy_queried': None, 'episodes': []}
    monkeypatch.setattr(recovery, 'recover_interrupted_cell_result', recover)
    def prepolicy_spawn(**kwargs):
        index = kwargs['index']
        result = spawn(**kwargs)
        if index == 3 and failure == 'missing':
            return result
        path = kwargs['child_root'] / PROVIDER_RESULT_FILENAME
        if index == 3 and failure == 'unreadable':
            temporary = path.with_suffix('.malformed')
            temporary.write_text('{incomplete')
            temporary.replace(path)
            return result
        if index == 3 and failure in ('timeout', 'recovery_failed'):
            # Recovery is needed only if the child timed out before sealing.
            # A valid receipt followed by a close hang has separate coverage.
            path.unlink()
            raise subprocess.TimeoutExpired('isolated-child', 2700)
        child = {'schema_version': 'native_task_arena_policy_canary_session_result.v1',
                 'selected_cell_index': index, 'status': 'blocked',
                 'task_success_contract': inputs['task_success_contract'],
                 'task_success_contract_digest': inputs['task_success_contract_digest'],
                 'episodes': [], 'policy_loads': [], 'candidate_policy_queried': False,
                 'preload_observation_gate': None, 'session_closeout': {'runtime_closed': True},
                 'telemetry': {'message_count': 0, 'channel_message_counts': {'observations': 0, 'episode_envelopes': 0}},
                 'session_failure_type': 'CameraGateError',
                 'session_failure_source_trace': [{'function': 'prepolicy_observation_gate'}]}
        child['result_digest'] = canonical_digest(child, digest_field='result_digest')
        worker._seal_result(result_path=path, result=child)
        return result
    if failure in ('missing', 'unreadable', 'recovery_failed'):
        with pytest.raises((RuntimeError, ValueError)):
            worker._run_isolated_cell_processes(runtime_root=runtime, output_root=output, run_cell_process=prepolicy_spawn)
    else:
        assert worker._run_isolated_cell_processes(runtime_root=runtime, output_root=output, run_cell_process=prepolicy_spawn) == 1
    result = json.loads((output / PROVIDER_RESULT_FILENAME).read_text())
    expected = 25 if failure == 'all_prepolicy' else 3
    assert result['status'] == 'blocked'
    assert len(result['isolated_cell_failures']) == expected
    assert result['episodes'] == [] and result['learned_policy_rollout_count'] == 0
    if failure == 'all_prepolicy':
        assert result['policy_loads'] == [] and result['candidate_policy_queried'] is False
    else:
        assert result['policy_loads'] is None and result['candidate_policy_queried'] is None
        assert result['interrupted_cell_index'] == 3
    assert result['planned_episodes_per_policy'] == 200
    assert result['result_digest'] == canonical_digest(result, digest_field='result_digest')
    checkpoint = json.loads((output / 'team_evaluation_checkpoint.json').read_text())
    assert len(checkpoint['completed_child_results']) == expected
