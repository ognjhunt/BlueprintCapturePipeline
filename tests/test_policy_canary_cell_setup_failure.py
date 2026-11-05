"""A cell that stops before any policy runs is named, sealed, and ends the matrix.

The real parent and child worker run here over the lifecycle rehearsal's fake
Isaac; only the simulator, and the one failure each test injects, are faked.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
# Dotted imports on purpose: the impacted-test selector maps each changed module to these tests.
import blueprint_pipeline.native_task_arena_policy_canary_session as session
import blueprint_pipeline.native_task_arena_policy_canary_worker as worker
import blueprint_pipeline.policy_canary_cell_setup_failure as setup_failure
from tests.test_native_task_arena_policy_canary_lifecycle_rehearsal import (
    FakeIsaac, _rehearsal_runtime, _run_cell_in_process, _sealed_result, _stage_runtime_root,
)

RESULT = worker.PROVIDER_RESULT_FILENAME


def _deep(depth: int, error: Exception) -> None:
    if depth:
        _deep(depth - 1, error)
    raise error


def _matrix(tmp_path: Path, *, fail_at: int, fault: str) -> tuple[int, list[int], Path]:
    runtime_root, output = _stage_runtime_root(tmp_path)
    spawned: list[int] = []
    isaacs: list[FakeIsaac] = []

    def factory(isaac: FakeIsaac):
        runtime = _rehearsal_runtime(isaac)
        if len(spawned) - 1 != fail_at:
            return runtime
        if fault == "runtime_open":
            def launch(*_args, **_kwargs):
                raise RuntimeError("Isaac failed to start at /workspace/private/kit token=secret")
            return replace(runtime, launch_isaac=launch)
        if fault == "observation_gate":
            def gate(**_kwargs):
                value = {"schema_version": "policy_canary_runtime_observation_integrity_gate.v1",
                         "status": "blocked", "policy_observation_integrity_passed": False,
                         "candidate_policy_loaded": False, "candidate_policy_queried": False,
                         "blockers": ["wrist:task_object_not_visible"], "gate_digest": ""}
                value["gate_digest"] = canonical_digest(value, digest_field="gate_digest")
                return value
            return replace(runtime, prepolicy_camera_gate=gate)
        if fault == "policy_load":
            def client(*_args, **_kwargs):
                raise ConnectionError("policy server at 10.0.0.7 did not answer")
            return replace(runtime, policy_client=client)
        assert fault == "deep_gate_raise"

        def renderer(*, simulation_app: Any, plan: Any) -> dict[str, Any]:
            _deep(40, RuntimeError("appearance renderer unqualified"))
            raise AssertionError("unreachable")
        return replace(runtime, prepare_appearance_renderer=renderer)

    def spawn(**kwargs):
        spawned.append(kwargs["index"])
        return _run_cell_in_process(**kwargs, isaacs=isaacs, runtime_factory=factory)

    exit_code = worker._run_isolated_cell_processes(runtime_root=runtime_root, output_root=output,
                                                     run_cell_process=spawn)
    return exit_code, spawned, output


def test_isaac_that_never_starts_is_sealed_named_and_stops_the_matrix(tmp_path: Path) -> None:
    exit_code, spawned, output = _matrix(tmp_path, fail_at=0, fault="runtime_open")

    assert exit_code == 1
    assert spawned == [0] and not (output / "cell_runs/01").exists()
    child = _sealed_result(output / "cell_runs/00" / RESULT)
    # Sealed even though no SimulationApp existed to close.
    assert child["status"] == "blocked" and child["session_closeout"]["status"] == "not_opened"
    assert child["telemetry"]["message_count"] == 0
    result = _sealed_result(output / RESULT)
    assert result["status"] == "blocked"
    assert result["blockers"] == ["policy_canary_session_setup_failed:runtime_open"]
    assert result["episodes"] == [] and result["policy_loads"] == []
    assert result["candidate_policy_queried"] is False and result["learned_policy_rollout_count"] == 0
    assert result["planned_episodes_per_policy"] == 10
    assert result["execution_stopped_after_cell"] == 1 and result["stopped_before_next_cell"] is True
    failure = result["cell_setup_failure"]
    assert failure["stage"] == "runtime_open" and failure["cell_index"] == 0
    assert failure["failure_type"] == "RuntimeError"
    assert failure["progress_stages"] == ["static_preflight_passed", "isaac_launch_started"]
    assert any(frame["function"] == "open_session" for frame in failure["source_trace"])
    assert failure["child_result"]["sha256"] == worker._sha256(output / "cell_runs/00" / RESULT)
    assert failure["child_result"]["result_digest"] == child["result_digest"]
    assert "/workspace" not in json.dumps(result) and "secret" not in json.dumps(result)
    assert setup_failure.sealed_setup_failure_blocker(result) == "policy_canary_session_setup_failed:runtime_open"


@pytest.mark.parametrize("stage", ["observation_gate", "policy_load"])
def test_later_cell_setup_failure_keeps_the_measured_prefix_and_starts_no_next_cell(
        tmp_path: Path, stage: str) -> None:
    exit_code, spawned, output = _matrix(tmp_path, fail_at=3, fault=stage)

    assert exit_code == 1
    assert spawned == [0, 1, 2, 3] and not (output / "cell_runs/04").exists()
    result = _sealed_result(output / RESULT)
    assert result["status"] == "blocked"
    assert result["blockers"] == [f"policy_canary_session_setup_failed:{stage}"]
    assert len(result["episodes"]) == 6 and all(row["status"] == "completed" for row in result["episodes"])
    assert {row["cell_id"] for row in result["episodes"]} == {"cell-0", "cell-1", "cell-2"}
    assert result["execution_stopped_after_cell"] == 4 and result["isolated_simulation_process_count"] == 4
    failure = result["cell_setup_failure"]
    assert failure["stage"] == stage and failure["cell_index"] == 3 and failure["cell_id"] == "cell-3"
    if stage == "observation_gate":
        assert failure["observation_gate_verdict"] == "refused"
        assert failure["observation_gate_blockers"] == ["wrist:task_object_not_visible"]
        assert failure["progress_stages"][-1] == "isaac_launch_completed"
    else:
        assert failure["failure_type"] == "ConnectionError"
        assert failure["observation_gate_verdict"] == "passed"
        assert failure["progress_stages"][-1] == "observation_gate_passed"
        assert "10.0.0.7" not in json.dumps(result)
    assert setup_failure.sealed_setup_failure_blocker(result) == f"policy_canary_session_setup_failed:{stage}"


def test_raised_gate_refusal_keeps_its_retained_shape_and_the_matrix_continues(tmp_path: Path) -> None:
    # The refusal raises 40 frames below the gate: the gate frame must survive the trace bound.
    exit_code, spawned, output = _matrix(tmp_path, fail_at=2, fault="deep_gate_raise")

    assert exit_code == 0
    assert spawned == list(range(10))
    child = _sealed_result(output / "cell_runs/02" / RESULT)
    assert len(child["session_failure_source_trace"]) == 24
    assert any(frame["function"] == "prepolicy_observation_gate" for frame in child["session_failure_source_trace"])
    assert child["session_failure_source_trace"][-1]["function"] == "_deep"
    result = _sealed_result(output / RESULT)
    assert result["status"] == "runtime_partial_pending_closeout"
    assert [row["cell"]["cell_id"] for row in result["isolated_cell_failures"]] == ["cell-2"]
    assert len(result["episodes"]) == 18 and "cell_setup_failure" not in result


def test_source_trace_filters_before_bounding_and_keeps_the_boundary_frame() -> None:
    def prepolicy_observation_gate() -> None:
        _deep(60, RuntimeError("deep"))

    try:
        prepolicy_observation_gate()
    except RuntimeError as exc:
        frames = session._source_trace(exc)
    assert len(frames) == 24
    assert [frame["function"] for frame in frames[:2]] == [
        "test_source_trace_filters_before_bounding_and_keeps_the_boundary_frame", "prepolicy_observation_gate"]
    assert frames[-1]["function"] == "_deep"
    assert all(set(frame) == {"file", "function", "line"} for frame in frames)


def _blocked_child(inputs: dict[str, Any], index: int, **changes: Any) -> dict[str, Any]:
    child = {"schema_version": session.RESULT_SCHEMA_VERSION, "selected_cell_index": index, "status": "blocked",
             "task_success_contract": inputs["task_success_contract"],
             "task_success_contract_digest": inputs["task_success_contract_digest"],
             "episodes": [], "policy_loads": [], "candidate_policy_queried": False, "preload_observation_gate": None,
             "session_closeout": {"status": "not_opened"}, "session_failure_type": "RuntimeError",
             "session_failure_codes": ["isaac_kit_missing", "Not A Code"],
             "session_failure_source_trace": [{"file": "worker.py", "function": "open_session", "line": 9},
                                              {"file": "/abs/path.py", "function": "x", "line": 1}],
             **changes}
    child["result_digest"] = canonical_digest(child, digest_field="result_digest")
    return child


def _write_child(root: Path, index: int, child: dict[str, Any], stages: list[str]) -> Path:
    child_root = root / "cell_runs" / f"{index:02d}"
    child_root.mkdir(parents=True, exist_ok=True)
    (child_root / RESULT).write_text(json.dumps(child))
    (child_root / setup_failure.PROGRESS_FILENAME).write_text("".join(
        f"BLUEPRINT_POLICY_CANARY_PROGRESS:cell={index}:stage={stage}\n" for stage in stages)
        + "BLUEPRINT_POLICY_CANARY_PROGRESS:cell=7:stage=isaac_launch_completed\nnoise\n")
    return child_root


def _inputs(tmp_path: Path) -> dict[str, Any]:
    runtime_root, _output = _stage_runtime_root(tmp_path)
    return json.loads((runtime_root / "runtime_inputs/policy_canary_runtime_inputs.json").read_text())


def test_classification_uses_the_cells_own_markers_and_bounded_record(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    child = _blocked_child(inputs, 1)
    child_root = _write_child(tmp_path, 1, child, ["static_preflight_passed", "isaac_launch_started"])
    failure = setup_failure.cell_setup_failure(child=child, inputs=inputs, index=1, child_root=child_root,
                                               output_root=tmp_path)
    assert failure["blocker"] == "policy_canary_session_setup_failed:runtime_open"
    assert failure["progress_stages"] == ["static_preflight_passed", "isaac_launch_started"]
    assert failure["failure_codes"] == ["isaac_kit_missing"]
    assert failure["source_trace"] == [{"file": "worker.py", "function": "open_session", "line": 9}]
    assert failure["child_result"]["relative_path"] == f"cell_runs/01/{RESULT}"
    assert failure["record_digest"] == canonical_digest(failure, digest_field="record_digest")


@pytest.mark.parametrize("change", [
    {"status": "runtime_selected_cell_completed_pending_aggregation"},
    {"candidate_policy_queried": True},
    {"policy_loads": [{"candidate_id": "pi05_droid", "loaded_once": True}]},
    {"episodes": [{"fixture": "row"}]},
    {"execution_control_stop": {"reason": "beta_screening_cancelled"}},
    {"selected_cell_index": 2},
    {"session_failure_type": "has spaces"},
    {"task_success_contract_digest": "sha256:" + "0" * 64},
    # The retained pre-policy gate refusal keeps its existing per-cell handling.
    {"session_closeout": {"runtime_closed": True}, "telemetry": {
        "message_count": 0, "channel_message_counts": {"observations": 0, "episode_envelopes": 0}},
     "session_failure_source_trace": [{"function": "prepolicy_observation_gate"}]},
])
def test_only_a_sealed_no_policy_cell_is_a_setup_failure(tmp_path: Path, change: dict[str, Any]) -> None:
    inputs = _inputs(tmp_path)
    child = _blocked_child(inputs, 1, **change)
    child_root = _write_child(tmp_path, 1, child, ["static_preflight_passed", "isaac_launch_started"])
    assert setup_failure.cell_setup_failure(child=child, inputs=inputs, index=1, child_root=child_root,
                                            output_root=tmp_path) is None
    tampered = {**_blocked_child(inputs, 1), "session_failure_type": "OSError"}
    assert setup_failure.cell_setup_failure(child=tampered, inputs=inputs, index=1, child_root=child_root,
                                            output_root=tmp_path) is None


def test_diagnostic_continuation_first_cell_gate_keeps_cell_zero(tmp_path: Path) -> None:
    inputs = {**_inputs(tmp_path), "diagnostic_continuation_protocol": {"schema_version": "fixture"}}
    child = _blocked_child(inputs, 0)
    child_root = _write_child(tmp_path, 0, child, ["static_preflight_passed"])
    assert setup_failure.cell_setup_failure(child=child, inputs=inputs, index=0, child_root=child_root,
                                            output_root=tmp_path) is None


@pytest.mark.parametrize("forge", ["code", "stage", "record", "result", "index", "completed"])
def test_recognizer_returns_only_its_own_vocabulary_for_an_untampered_result(tmp_path: Path, forge: str) -> None:
    exit_code, _spawned, output = _matrix(tmp_path, fail_at=0, fault="runtime_open")
    assert exit_code == 1
    result = _sealed_result(output / RESULT)
    failure = result["cell_setup_failure"]
    if forge == "code":
        failure["blocker"] = "policy_canary_session_setup_failed:invented"
        result["blockers"] = [failure["blocker"]]
    elif forge == "stage":
        failure["stage"] = "policy_load"
    elif forge == "index":
        failure["cell_index"] = 4
    elif forge == "completed":
        result["status"] = "completed_unqualified"
    if forge != "record":
        failure["record_digest"] = canonical_digest(failure, digest_field="record_digest")
    if forge != "result":
        result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    else:
        result["episodes_per_policy"] = 5
    if forge == "record":
        failure["failure_type"] = "OSError"
        result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    assert setup_failure.sealed_setup_failure_blocker(result) is None


@pytest.mark.parametrize("case", ["beta_flagged", "flags_not_the_gates", "internal_canary"])
def test_a_beta_gate_the_session_continued_past_is_not_named_the_gate_stage(tmp_path: Path, case: str) -> None:
    from tests.test_beta_screening_worker_control import REAL_GATE_BLOCKERS, runtime_observation_gate
    inputs = _inputs(tmp_path)
    gate = runtime_observation_gate(backend_digest="sha256:" + "b" * 64)
    flags = sorted(REAL_GATE_BLOCKERS)
    # Stopped after the gate and before any policy load frame (for example a strict control gate).
    child = _blocked_child(
        inputs, 1, run_kind=session.RUN_KIND if case == "internal_canary" else session.BETA_RUN_KIND,
        preload_observation_gate=gate, observation_integrity_flagged=True,
        observation_quality_flags=flags[:1] if case == "flags_not_the_gates" else flags,
        session_failure_source_trace=[{"file": "native_task_arena_policy_canary_session.py",
                                       "function": "execute_paired_session", "line": 1}])
    child_root = _write_child(tmp_path, 1, child,
                              ["static_preflight_passed", "isaac_launch_started", "isaac_launch_completed"])
    failure = setup_failure.cell_setup_failure(child=child, inputs=inputs, index=1, child_root=child_root,
                                               output_root=tmp_path)
    assert failure["observation_gate_blockers"] == flags
    if case == "beta_flagged":
        assert failure["stage"] == "policy_load" and failure["observation_gate_verdict"] == "flagged"
        assert failure["observation_quality_flags"] == flags
    else:  # Unchanged: a refusal the session did not continue past names the gate.
        assert failure["stage"] == "observation_gate" and failure["observation_gate_verdict"] == "refused"
        assert "observation_quality_flags" not in failure



@pytest.mark.parametrize("exit_code, signal_number", [
    (139, 11), (-11, 11), (135, 7), (-7, 7), (134, 6), (132, 4), (136, 8), (1, None), (0, None),
    (137, None), (-9, None), (143, None), ("139", None), (None, None)])
def test_only_a_fault_signal_is_a_fatal_signal(exit_code, signal_number) -> None:
    assert setup_failure.fatal_signal(exit_code) == signal_number


def _crashed_cell(tmp_path: Path, stages: list[str]) -> Path:
    child_root = tmp_path / "cell_runs" / "00"
    child_root.mkdir(parents=True)
    (child_root / setup_failure.PROGRESS_FILENAME).write_text(
        "".join(f"BLUEPRINT_POLICY_CANARY_PROGRESS:cell=0:stage={stage}\n" for stage in stages))
    (child_root / "worker_console.log").write_text("[Fatal] terminating this process with exit code 139.\n")
    return child_root


@pytest.mark.parametrize("case", ["start_crash", "launched", "not_started", "exit_one", "child_sealed"])
def test_a_machine_fault_is_a_fatal_signal_while_isaac_was_starting_with_no_result(tmp_path: Path, case: str) -> None:
    inputs = _inputs(tmp_path)
    stages = ["static_preflight_passed", "isaac_launch_started"]
    if case == "launched":
        stages.append("isaac_launch_completed")
    elif case == "not_started":
        stages = ["static_preflight_passed"]
    child_root = _crashed_cell(tmp_path, stages)
    if case == "child_sealed":
        (child_root / RESULT).write_text("{}")
    fault = setup_failure.cell_machine_fault(exit_code=1 if case == "exit_one" else 139, inputs=inputs, index=0,
                                             child_root=child_root, output_root=tmp_path)
    if case != "start_crash":
        assert fault is None
        return
    assert fault["blocker"] == "policy_canary_session_machine_fault:simulator_start_crash"
    assert (fault["stage"], fault["fatal_signal"], fault["worker_exit_code"]) == ("simulator_start_crash", 11, 139)
    assert fault["progress_stages"] == stages and fault["candidate_policy_queried"] is False
    assert fault["worker_console"]["relative_path"] == "cell_runs/00/worker_console.log"
    assert fault["record_digest"] == canonical_digest(fault, digest_field="record_digest")
    assert setup_failure.MACHINE_FAULT_BLOCKERS.isdisjoint(setup_failure.SETUP_FAILURE_BLOCKERS)


@pytest.mark.parametrize("forge", ["none", "setup_prefix", "stage"])
def test_the_recognizer_returns_the_machine_fault_code_only_for_its_own_record(tmp_path: Path, forge: str) -> None:
    inputs = {**_inputs(tmp_path), "run_kind": session.BETA_RUN_KIND,
              "candidate_ids": ["pi05_droid"], "runtime_inputs_digest": "sha256:" + "e" * 64,
              "matrix_digest": "sha256:" + "f" * 64}
    child_root = _crashed_cell(tmp_path, ["static_preflight_passed", "isaac_launch_started"])
    sealed = {}
    assert setup_failure.seal_cell_machine_fault(
        exit_code=-11, inputs=inputs, index=0, child_root=child_root, output_root=tmp_path, children=[],
        authority={"authority_digest": "sha256:" + "d" * 64}, construction_lineage_mode="diagnostic",
        aggregate=lambda **_: None, seal_result=lambda *, result_path, result: sealed.update(result=result))
    result = sealed["result"]
    assert result["blockers"] == ["policy_canary_session_machine_fault:simulator_start_crash"]
    assert result["execution_stopped_after_cell"] == 1 and result["stopped_before_next_cell"] is True
    failure = result["cell_setup_failure"]
    if forge == "setup_prefix":
        failure["blocker"] = result["blockers"][0] = "policy_canary_session_setup_failed:simulator_start_crash"
    elif forge == "stage":
        failure["stage"] = "runtime_open"
    if forge != "none":
        failure["record_digest"] = canonical_digest(failure, digest_field="record_digest")
        result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    expected = "policy_canary_session_machine_fault:simulator_start_crash" if forge == "none" else None
    assert setup_failure.sealed_setup_failure_blocker(result) == expected
    # Outside a beta screening run nothing is sealed: the existing missing-result handling stands.
    assert not setup_failure.seal_cell_machine_fault(
        exit_code=139, inputs={**inputs, "run_kind": session.RUN_KIND}, index=0, child_root=child_root,
        output_root=tmp_path, children=[], authority={}, construction_lineage_mode="diagnostic",
        aggregate=lambda **_: None, seal_result=lambda **_: None)


def test_the_parent_seals_a_machine_fault_instead_of_raising(tmp_path: Path, monkeypatch) -> None:
    runtime_root, output = _stage_runtime_root(tmp_path)
    calls = []

    def seal(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(worker, "seal_cell_machine_fault", seal)

    def spawn(*, index, child_root, **_kwargs):
        (child_root / setup_failure.PROGRESS_FILENAME).write_text(
            f"BLUEPRINT_POLICY_CANARY_PROGRESS:cell={index}:stage=isaac_launch_started\n")
        return 139

    assert worker._run_isolated_cell_processes(runtime_root=runtime_root, output_root=output,
                                               run_cell_process=spawn) == 1
    [call] = calls
    assert (call["exit_code"], call["index"], call["children"]) == (139, 0, [])
    assert not (output / "cell_runs/01").exists()
