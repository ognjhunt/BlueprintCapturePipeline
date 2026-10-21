"""CPU orchestration proof; synthetic receipts deliberately fail real media qualification."""

from __future__ import annotations

import copy
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.exact_workcell_evaluation_session import (
    _atomic,
    artifact_record,
    execute_evaluation_schedule,
)
from blueprint_pipeline.exact_workcell_native_session import (
    NativeCellProcessRunner,
    execute_native_evaluation_schedule,
)
from blueprint_pipeline.exact_workcell_variation_matrix import (
    ExactWorkcellVariationError,
    compile_variation_matrix,
)
from blueprint_pipeline.exact_workcell_variation_runtime import compile_evaluation_schedule
from tests.test_exact_workcell_variation_matrix import _request, _schedule_request


def _inputs(tmp_path):
    request = _request()
    matrix = compile_variation_matrix(request)
    schedule_request = _schedule_request(matrix)
    schedule = compile_evaluation_schedule(
        matrix, request=request, schedule_request=schedule_request
    )
    return dict(
        request=request,
        matrix=matrix,
        schedule_request=schedule_request,
        schedule=schedule,
        source_binding={
            "job_id": "job-a",
            "site_id": "scene_840920",
            "capture_id": "capture-a",
            "source_capture_digest": "sha256:" + "a" * 64,
            "source_plan_digest": "sha256:" + "b" * 64,
            "configuration_digest": "sha256:" + "c" * 64,
        },
        evidence_root=tmp_path,
    )


def _retain(root, row):
    path = root / "receipts" / (canonical_digest({"episode_id": row["episode_id"]})[7:] + ".json")
    # This is deliberately NOT an admissible simulator result. The real
    # collector must retain it as blocked instead of treating count400 as proof.
    _atomic(
        path,
        {
            "schema_version": "synthetic_software_receipt",
            "episode_id": row["episode_id"],
            "candidate_policy_queried": False,
        },
    )
    return {
        "episode_id": row["episode_id"],
        "episode_binding_digest": row["episode_binding_digest"],
        "artifact": artifact_record(root, path),
    }


def test_full100_schedule_runs400_once_and_actual_collector_refuses_fixture_evidence(tmp_path):
    args = _inputs(tmp_path)
    observed = []

    def run_cell(*, cell, rows, begin, commit):
        assert len(rows) == 4
        assert [r["subject_type"] for r in rows] == ["control", "control", "policy", "policy"]
        assert all(
            r["seed"] == cell["seed"] and r["reset_digest"] == cell["reset_digest"] for r in rows
        )
        for row in rows:
            begin(row)
            observed.append(row["episode_id"])
            commit(row, _retain(tmp_path, row))

    first = execute_evaluation_schedule(
        **args, run_cell=run_cell, assert_execution_allowed=lambda: None
    )
    assert len(observed) == len(set(observed)) == 400
    assert first["status"] == "blocked"
    assert first["completed_episode_count"] == 400
    assert first["evidence"]["qualification_authorized"] is False
    assert any("episode_evidence_incomplete" in code for code in first["evidence"]["blockers"])
    second = execute_evaluation_schedule(
        **args,
        run_cell=lambda **_: pytest.fail("completed effects must not replay"),
        assert_execution_allowed=lambda: pytest.fail("no effects to authorize"),
    )
    assert second["evidence"]["evidence_digest"] == first["evidence"]["evidence_digest"]


def test_unknown_effect_after_worker_interruption_never_replays(tmp_path):
    args = _inputs(tmp_path)
    calls = []

    def fail_after_effect(*, rows, begin, **_):
        begin(rows[0])
        calls.append(rows[0]["episode_id"])
        raise TimeoutError("provider acknowledgement ambiguous")

    first = execute_evaluation_schedule(
        **args, run_cell=fail_after_effect, assert_execution_allowed=lambda: None
    )
    assert first["status"] == "interrupted"
    second = execute_evaluation_schedule(
        **args,
        run_cell=lambda **_: pytest.fail("unknown effect replayed"),
        assert_execution_allowed=lambda: None,
    )
    assert second["status"] == "reconciliation_required"
    assert second["episode_ids"] == calls
    assert second["paid_execution_retried"] is False


def test_lost_receipt_acknowledgement_adopts_actual_retained_sidecar_without_repeating_effect(
    tmp_path,
):
    args = _inputs(tmp_path)
    first_id = args["schedule"]["rows"][0]["episode_id"]

    def lose_ack(*, rows, begin, **_):
        row = rows[0]
        begin(row)
        record = _retain(tmp_path, row)
        sidecar = (
            tmp_path
            / "receipts"
            / (canonical_digest({"episode_id": row["episode_id"]})[7:] + ".record.json")
        )
        _atomic(sidecar, record)
        raise TimeoutError()

    execute_evaluation_schedule(**args, run_cell=lose_ack, assert_execution_allowed=lambda: None)
    seen = []

    def resume(*, rows, begin, commit, **_):
        for row in rows:
            assert row["episode_id"] != first_id
            begin(row)
            seen.append(row["episode_id"])
            commit(row, _retain(tmp_path, row))

    resumed = execute_evaluation_schedule(
        **args, run_cell=resume, assert_execution_allowed=lambda: None
    )
    assert resumed["completed_episode_count"] == 400
    assert len(seen) == 399


@pytest.mark.parametrize("change", ["capture_id", "job_id", "configuration_digest"])
def test_revised_scope_cannot_reuse_an_existing_execution_directory(tmp_path, change):
    args = _inputs(tmp_path)
    execute_evaluation_schedule(
        **args,
        run_cell=lambda **_: (_ for _ in ()).throw(RuntimeError()),
        assert_execution_allowed=lambda: None,
    )
    modified = copy.deepcopy(args)
    modified["source_binding"][change] = (
        "other" if not change.endswith("digest") else "sha256:" + "d" * 64
    )
    with pytest.raises(ExactWorkcellVariationError, match="source_or_scope_changed"):
        execute_evaluation_schedule(
            **modified,
            run_cell=lambda **_: pytest.fail("cross-scope effect"),
            assert_execution_allowed=lambda: None,
        )


def test_revocation_is_rechecked_before_each_episode_and_pending_effects_stay_honest(tmp_path):
    args = _inputs(tmp_path)
    effects = []
    checks = []

    def authority():
        checks.append(True)
        if effects:
            raise PermissionError("revoked")

    def run_cell(*, rows, begin, commit, **_):
        for row in rows:
            begin(row)
            effects.append(row["episode_id"])
            commit(row, _retain(tmp_path, row))

    result = execute_evaluation_schedule(
        **args, run_cell=run_cell, assert_execution_allowed=authority
    )
    assert result["status"] == "interrupted" and result["completed_episode_count"] == 1
    state = json.loads((tmp_path / "exact_workcell_session_state.v1.json").read_text())
    assert all(row["state"] == "completed" for row in state["episodes"].values())
    assert len(checks) == 3


def test_changed_complete_runtime_inputs_cannot_resume_same_session(tmp_path):
    args = _inputs(tmp_path)
    execute_evaluation_schedule(
        **args,
        execution_inputs_digest="sha256:" + "1" * 64,
        run_cell=lambda **_: (_ for _ in ()).throw(RuntimeError()),
        assert_execution_allowed=lambda: None,
    )
    with pytest.raises(ExactWorkcellVariationError, match="source_or_scope_changed"):
        execute_evaluation_schedule(
            **args,
            execution_inputs_digest="sha256:" + "2" * 64,
            run_cell=lambda **_: pytest.fail("changed endpoint/authority/construction executed"),
            assert_execution_allowed=lambda: None,
        )


def test_tampered_retained_episode_is_refused_before_any_execution(tmp_path):
    args = _inputs(tmp_path)

    def one(*, rows, begin, commit, **_):
        begin(rows[0])
        record = _retain(tmp_path, rows[0])
        commit(rows[0], record)
        raise RuntimeError()

    execute_evaluation_schedule(**args, run_cell=one, assert_execution_allowed=lambda: None)
    receipt = next((tmp_path / "receipts").glob("*.json"))
    receipt.write_text("changed")
    with pytest.raises(ExactWorkcellVariationError, match="bytes_changed"):
        execute_evaluation_schedule(
            **args,
            run_cell=lambda **_: pytest.fail("tampered replay"),
            assert_execution_allowed=lambda: None,
        )


def test_native_entrypoint_binds_source_and_refuses_missing_cell_construction_before_spawn(
    tmp_path,
):
    args = _inputs(tmp_path)
    config = {
        **args,
        "source_scene_plan": {"plan_digest": args["source_binding"]["source_plan_digest"]},
        "cell_timeout_seconds": 10,
        "construction_results": {},
    }
    result = execute_native_evaluation_schedule(
        config=config, assert_execution_allowed=lambda: None
    )
    assert result["status"] == "blocked" and result["completed_episode_count"] == 0
    assert result["blockers"] == ["full_matrix_cell_construction_required"]
    config["source_binding"] = {**config["source_binding"], "site_id": "other-site"}
    with pytest.raises(ExactWorkcellVariationError, match="source_identity_mismatch"):
        execute_native_evaluation_schedule(config=config, assert_execution_allowed=lambda: None)
    assert NativeCellProcessRunner.__module__.endswith("exact_workcell_native_session")


def test_registered_native_adapter_refuses_generic_execution_gate_without_live_authority(tmp_path):
    from blueprint_pipeline.evaluation_run_contract import EvaluationRunSpec
    from blueprint_pipeline.evaluation_run_execution import (
        default_evaluation_run_execution_registry,
    )
    from tests.test_evaluation_run_execution import _spec

    registry = default_evaluation_run_execution_registry()
    adapter = registry.resolve("exact_workcell_native_droid")
    assert adapter is not None
    result = adapter.execute(
        spec=EvaluationRunSpec.from_mapping(_spec()), output_dir=tmp_path, context={}
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == ["existing_native_runtime_authority_and_immutable_inputs_required"]


def test_control_loop_retains_frozen_schedule_episode_identity(tmp_path):
    from blueprint_pipeline.adp009d_control_episode import run_task_neutral_control
    from tests.test_adp009d_task_neutral_controls import _plan, _task
    from tests.test_adp009d_control_episode import _ControlEnvironment

    task = _task("rigid_pick_place")
    result = run_task_neutral_control(
        environment=_ControlEnvironment(),
        task_spec=task,
        control_plan=_plan(task),
        control_id="zero_action_negative",
        gripper_open_command=1.0,
        gripper_closed_command=0.0,
        output_dir=tmp_path,
        episode_id="site-a.cell_000.zero_action_negative",
    )
    assert result["episode_id"] == "site-a.cell_000.zero_action_negative"
    assert result["candidate_policy_queried"] is False
    assert result["grader_authority"] == "deterministic_simulator_state"
    assert result["receipt_digest"] == canonical_digest(result, digest_field="receipt_digest")


@pytest.mark.parametrize("after_observation", [False, True])
def test_native_failure_reuses_media_sealer_and_retains_typed_gap(tmp_path, after_observation):
    from blueprint_pipeline.exact_workcell_native_session import retain_native_policy_failure

    args = _inputs(tmp_path)
    row = next(row for row in args["schedule"]["rows"] if row["subject_type"] == "policy")
    progress = {
        "first_observation_retained": after_observation,
        "candidate_policy_query_attempted": after_observation,
    }
    called = []
    if after_observation:

        def finalize(*, failure_reason):
            called.append(failure_reason)
            return {
                "status": "incomplete_after_first_observation",
                "media_gap": {"type": "retained_test_gap"},
            }, []

        progress["_failure_media_finalizer"] = finalize
    result = retain_native_policy_failure(
        root=tmp_path,
        cell_root=tmp_path / "cell",
        row=row,
        progress=progress,
        failure=TimeoutError("unconfirmed transport"),
        run_id="job-a",
    )
    assert result["episode_id"] == row["episode_id"]
    assert result["score"]["status"] == "not_scored"
    assert result["candidate_policy_query_attempted"] is after_observation
    assert len(called) == int(after_observation)
    assert (tmp_path / result["failure_evidence"]["relative_path"]).is_file()
    assert result["visual_evidence"]["media_gap"]["type"] == (
        "retained_test_gap" if after_observation else "before_first_observation"
    )
