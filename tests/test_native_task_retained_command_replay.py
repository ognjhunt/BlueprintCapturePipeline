"""Bounded native command replay, reset refusal and process boundaries are CPU tested."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import json
import subprocess

import pytest

from blueprint_pipeline.native_task_retained_command_replay import (
    RetainedReplayAdapters,
    run_retained_command_replay,
    validate_retained_replay_request,
    REQUEST_SCHEMA,
)
from blueprint_pipeline.native_task_composition_diagnostic import seal
from blueprint_pipeline.native_task_combined_diagnostic_worker import (
    run_diagnostic_children,
    CHILDREN,
)
from blueprint_pipeline.policy_scientific_reset import seal_reset_readback
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def request():
    reset = seal_reset_readback(
        binding={
            "candidate_id": "pi05_droid",
            "cell_id": "retained-cell01",
            "seed": 1881096655,
            "task_spec_digest": "sha256:" + "1" * 64,
            "resolved_scenario_digest": "sha256:" + "2" * 64,
        },
        observed={
            "robot": {"joint_limits": [[[-2.9, 2.9]] * 7], "joint_pos": [[0.0] * 7]},
            **{
                key: {"fixture": True}
                for key in (
                    "objects",
                    "scene_assets",
                    "cameras",
                    "physics",
                    "lighting",
                    "colliders",
                    "contacts",
                )
            },
        },
        sources={
            key: "native_fixture"
            for key in (
                "robot",
                "objects",
                "scene_assets",
                "cameras",
                "physics",
                "lighting",
                "colliders",
                "contacts",
            )
        },
        gaps=[],
    )
    commands = [
        {
            "step_index": i,
            "environment_step_applied": True,
            "native_command_validated": True,
            "isaac_action": [0.1] * 7 + [float(i % 2)],
            "joint_position_target_rad": [0.1] * 7,
        }
        for i in range(1, 175)
    ]
    return seal(
        {
            "schema_version": REQUEST_SCHEMA,
            "source_result": {"sha256": "sha256:" + "a" * 64, "size_bytes": 1},
            "cell_id": "retained-cell01",
            "seed": 1881096655,
            "commands": commands,
            "command_count": 174,
            "command_tape_digest": canonical_digest({"commands": commands}),
            "expected_reset": reset,
            "policy_queries_permitted": 0,
            "model_calls_permitted": 0,
            "score_recomputation_permitted": False,
            "stop_on_first_native_joint_limit_violation": True,
        },
        "request_digest",
    )


def adapters(req, *, violation=174, mismatch=False):
    calls = []

    def state():
        joints = [0.1] * 7
        if violation and len(calls) >= violation:
            joints[4] = -4.004210948944092
        return {
            "joint_position_rad": joints,
            "joint_limits_rad": [[-2.9, 2.9]] * 7,
            "joint_position_target_rad": [0.1] * 7,
            "computed_torque_nm": [2.0] * 7,
            "applied_torque_nm": [1.0] * 7,
        }

    def reset():
        value = deepcopy(req["expected_reset"])
        if mismatch:
            value["observed"]["robot"]["joint_pos"][0][0] = 0.001
            value["scientific_state_digest"] = canonical_digest(value["observed"])
            seal(value, "receipt_digest")
        return value

    return RetainedReplayAdapters(
        reset,
        state,
        lambda action: calls.append(list(action)),
        lambda label, root: {"label": label},
    ), calls


def test_exact_174_native_commands_replayed_without_a_policy_and_violation_is_retained(tmp_path):
    req = request()
    before = deepcopy(req)
    binding, calls = adapters(req)
    result = run_retained_command_replay(req, output_root=tmp_path / "out", adapters=binding)
    assert result["status"] == "completed", result
    assert result["executed_commands"] == 174 and len(calls) == 174
    assert calls == [row["isaac_action"] for row in req["commands"]]
    assert req == before and result["policy_queries"] == result["model_calls"] == 0
    assert result["diagnostic_outcome"] == "native_joint_limit_violation_observed"
    assert result["native_joint_state_violation"]["step_index"] == 174
    assert result["native_states"][-1]["after"]["joint_position_rad"][4] == -4.004210948944092
    assert result["task_motion_executed"] is True and result["scoring_performed"] is False
    assert len((tmp_path / "out/native_states.jsonl").read_text().splitlines()) == 174
    assert (
        json.loads((tmp_path / "out/replay_progress.json").read_text())["executed_commands"] == 174
    )


def test_first_earlier_native_violation_stops_without_any_extra_tape_rows(tmp_path):
    req = request()
    binding, calls = adapters(req, violation=3)
    result = run_retained_command_replay(req, output_root=tmp_path / "out", adapters=binding)
    assert result["executed_commands"] == len(calls) == 3
    assert result["native_joint_state_violation"]["step_index"] == 3


def test_reset_mismatch_cannot_execute_even_one_retained_action(tmp_path):
    req = request()
    binding, calls = adapters(req, mismatch=True)
    result = run_retained_command_replay(req, output_root=tmp_path / "out", adapters=binding)
    assert result["status"] == "blocked" and result["reset_comparison"]["status"] == "mismatch"
    assert calls == [] and result["task_motion_executed"] is False


def test_no_violation_is_reported_as_negative_diagnostic_not_success(tmp_path):
    req = request()
    binding, calls = adapters(req, violation=None)
    result = run_retained_command_replay(req, output_root=tmp_path / "out", adapters=binding)
    assert (
        len(calls) == 174
        and result["diagnostic_outcome"] == "no_violation_in_174_retained_commands"
    )
    assert result["scoring_performed"] is False


@pytest.mark.parametrize("fault", ["longer_tape", "changed_action", "invalid_native_target"])
def test_tape_scope_and_unchanged_native_targets_fail_closed(fault):
    req = request()
    if fault == "longer_tape":
        req["commands"].append(deepcopy(req["commands"][-1]))
    elif fault == "changed_action":
        req["commands"][0]["isaac_action"][0] = 0.2
    else:
        req["commands"][0]["isaac_action"][0] = 4.0
        req["commands"][0]["joint_position_target_rad"][0] = 4.0
    req["command_tape_digest"] = canonical_digest({"commands": req["commands"]})
    seal(req, "request_digest")
    with pytest.raises(ValueError):
        validate_retained_replay_request(req)


def test_two_native_children_exit_in_order_on_same_resource_without_false_no_motion(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "adp_arena_provider_manifest.json").write_text(
        json.dumps(
            {
                "implementation_commit": "a" * 40,
                "container_image": "test",
                "input_digest": "sha256:" + "b" * 64,
            }
        )
    )
    calls = []

    def runner(command, **kwargs):
        module = command[2]
        calls.append(module)
        output = Path(kwargs["env"]["BLUEPRINT_ADP_ARENA_OUTPUT_DIR"])
        name = next(filename for _, m, filename in CHILDREN if m == module)
        assert kwargs["timeout"] == 600
        motion = module.endswith("native_task_retained_command_worker")
        (output / name).write_text(
            json.dumps(
                {
                    "status": "completed",
                    "candidate_policy_queried": False,
                    "task_motion_executed": motion,
                }
            )
        )
        return NS(returncode=0)

    result = run_diagnostic_children(
        runtime_root=runtime, output_root=tmp_path / "out", runner=runner
    )
    assert calls == [row[1] for row in CHILDREN]
    assert result["status"] == "completed" and result["task_motion_executed"] is True
    assert all(row["process_exited_before_next_child"] for row in result["children"])
    assert result["candidate_policy_queried"] is False


def test_isaac_timeout_does_not_start_another_child(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "adp_arena_provider_manifest.json").write_text(
        json.dumps(
            {
                "implementation_commit": "a" * 40,
                "container_image": "test",
                "input_digest": "sha256:" + "b" * 64,
            }
        )
    )
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, 600)

    result = run_diagnostic_children(
        runtime_root=runtime, output_root=tmp_path / "out", runner=runner
    )
    assert result["status"] == "blocked" and len(calls) == 1


def test_combined_payload_imports_in_isolated_interpreter_without_policy_or_simulator(tmp_path):
    import shutil
    import sys
    from blueprint_pipeline.native_task_composition_bundle import composition_runtime_sources

    package = tmp_path / "blueprint_pipeline"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for source in composition_runtime_sources(include_replay=True):
        shutil.copyfile(source, package / source.name)
    code = "import sys; sys.path.insert(0,sys.argv[1]); import blueprint_pipeline.native_task_combined_diagnostic_worker; import blueprint_pipeline.native_task_retained_command_worker; assert not any(k.startswith(('isaaclab.','openpi.','groot.')) for k in sys.modules)"
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
