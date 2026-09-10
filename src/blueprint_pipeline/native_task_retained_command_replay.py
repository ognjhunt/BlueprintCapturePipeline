"""One bounded native replay of the retained V27 cell-01 command tape, no policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Callable

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_composition_diagnostic import file_record, seal
from blueprint_pipeline.policy_scientific_reset import (
    validate_reset_readback,
    compare_reset_readbacks,
)

REQUEST_SCHEMA = "native_task_retained_command_replay_request.v1"
RESULT_SCHEMA = "native_task_retained_command_replay.v1"
MAX_COMMANDS = 174


def build_retained_replay_request(*, source_result_path, resolved_scene_plan, expected_cell):
    source_path = Path(source_result_path)
    source = json.loads(source_path.read_text())
    if (
        source.get("schema_version") != "native_task_arena_policy_canary_session_result.v1"
        or source.get("result_digest") != canonical_digest(source, digest_field="result_digest")
        or source.get("selected_cell_index") != 1
    ):
        raise ValueError("retained_replay_source_result_invalid")
    matches = [
        (index, row)
        for index, row in enumerate(source["episodes"])
        if row.get("candidate_id") == "pi05_droid"
        and row.get("cell_id") == expected_cell["cell_id"]
    ]
    if len(matches) != 1:
        raise ValueError("retained_replay_exact_episode_missing")
    index, episode = matches[0]
    reset = validate_reset_readback(episode["episode"]["scientific_reset"])
    if (
        reset["binding"]["seed"] != expected_cell["seed"]
        or reset["binding"]["task_spec_digest"]
        != canonical_digest(resolved_scene_plan["task_spec"])
        or resolved_scene_plan["scenario"]["cell_id"] != expected_cell["cell_id"]
    ):
        raise ValueError("retained_replay_reset_binding_mismatch")
    commands = episode["episode"]["commanded_actions"][:MAX_COMMANDS]
    if len(commands) != MAX_COMMANDS:
        raise ValueError("retained_replay_174_commands_required")
    request = seal(
        {
            "schema_version": REQUEST_SCHEMA,
            "source_result": file_record(source_path),
            "source_result_digest": source["result_digest"],
            "source_episode_json_pointer": f"/episodes/{index}",
            "source_candidate_id": "pi05_droid",
            "cell_id": expected_cell["cell_id"],
            "seed": expected_cell["seed"],
            "scene_plan_digest": resolved_scene_plan["plan_digest"],
            "expected_reset": reset,
            "commands": commands,
            "command_count": MAX_COMMANDS,
            "command_tape_digest": canonical_digest({"commands": commands}),
            "policy_queries_permitted": 0,
            "model_calls_permitted": 0,
            "score_recomputation_permitted": False,
            "stop_on_first_native_joint_limit_violation": True,
            "source_results_modified": False,
        },
        "request_digest",
    )
    validate_retained_replay_request(request)
    return request


def validate_retained_replay_request(request):
    if (
        request.get("schema_version") != REQUEST_SCHEMA
        or request.get("request_digest") != canonical_digest(request, digest_field="request_digest")
        or request.get("command_count") != MAX_COMMANDS
        or len(request.get("commands", [])) != MAX_COMMANDS
        or request.get("command_tape_digest") != canonical_digest({"commands": request["commands"]})
        or request.get("policy_queries_permitted") != 0
        or request.get("model_calls_permitted") != 0
        or request.get("score_recomputation_permitted") is not False
        or request.get("stop_on_first_native_joint_limit_violation") is not True
    ):
        raise ValueError("retained_replay_request_invalid")
    reset = validate_reset_readback(request["expected_reset"])
    if (
        reset["binding"]["cell_id"] != request["cell_id"]
        or reset["binding"]["seed"] != request["seed"]
    ):
        raise ValueError("retained_replay_reset_binding_mismatch")
    for index, row in enumerate(request["commands"], 1):
        values = row.get("isaac_action")
        if (
            row.get("step_index") != index
            or row.get("environment_step_applied") is not True
            or row.get("native_command_validated") is not True
            or not isinstance(values, list)
            or len(values) != 8
            or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in values
            )
            or values[:7] != row.get("joint_position_target_rad")
        ):
            raise ValueError("retained_replay_source_command_invalid")
        limits = reset["observed"]["robot"]["joint_limits"][0][:7]
        if any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(values[:7], limits, strict=True)
        ):
            raise ValueError("retained_replay_source_native_command_out_of_bounds")
    return request


@dataclass(frozen=True)
class RetainedReplayAdapters:
    reset_and_read: Callable[[], dict]
    read_state: Callable[[], dict]
    step: Callable[[list[float]], None]
    snapshot: Callable[[str, Path], dict]


def run_retained_command_replay(request, *, output_root, adapters: RetainedReplayAdapters):
    from blueprint_pipeline.adp009d_policy_episode import (
        NativeJointStateBoundsError,
        validate_native_joint_state,
    )

    validate_retained_replay_request(request)
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise ValueError("retained_replay_output_must_be_fresh")
    root.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "blocked",
        "request_digest": request["request_digest"],
        "cell_id": request["cell_id"],
        "seed": request["seed"],
        "source_result": request["source_result"],
        "command_tape_digest": request["command_tape_digest"],
        "maximum_commands": MAX_COMMANDS,
        "executed_commands": 0,
        "candidate_policy_queried": False,
        "policy_queries": 0,
        "model_calls": 0,
        "task_motion_executed": False,
        "arm_motion_observed": False,
        "scoring_performed": False,
        "source_results_modified": False,
        "native_states": [],
        "frames": [],
        "blockers": [],
    }
    try:
        actual_reset = adapters.reset_and_read()
        comparison = compare_reset_readbacks(request["expected_reset"], actual_reset)
        result.update(actual_reset=actual_reset, reset_comparison=comparison)
        (root / "actual_reset.json").write_text(
            json.dumps(actual_reset, indent=2, sort_keys=True) + "\n"
        )
        if comparison["status"] != "matched":
            raise ValueError("retained_replay_native_reset_not_exact")
        before = adapters.read_state()
        validate_native_joint_state(
            before["joint_position_rad"], before["joint_limits_rad"], phase="replay_reset"
        )
        result["initial_native_state"] = before
        result["frames"].append(adapters.snapshot("reset", root / "frames"))
        for source in request["commands"]:
            # The action array is copied without decoding, scaling, clamping,
            # interpolation, gripper remapping, or policy inference.
            action = list(source["isaac_action"])
            adapters.step(action)
            result["executed_commands"] += 1
            result["task_motion_executed"] = True
            progress = {
                "executed_commands": result["executed_commands"],
                "task_motion_executed": True,
                "policy_queries": 0,
                "model_calls": 0,
                "request_digest": request["request_digest"],
            }
            temporary = root / "replay_progress.tmp"
            temporary.write_text(json.dumps(progress, sort_keys=True) + "\n")
            temporary.replace(root / "replay_progress.json")
            after = adapters.read_state()
            row = {
                "step_index": source["step_index"],
                "source_action": action,
                "source_command_digest": canonical_digest(source),
                "before": before,
                "after": after,
            }
            result["native_states"].append(row)
            with (root / "native_states.jsonl").open("a") as journal:
                journal.write(json.dumps(row, sort_keys=True) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
            result["arm_motion_observed"] |= (
                before["joint_position_rad"] != after["joint_position_rad"]
            )
            try:
                if before["joint_limits_rad"] != after["joint_limits_rad"]:
                    raise ValueError("retained_replay_native_limits_changed")
                validate_native_joint_state(
                    after["joint_position_rad"],
                    after["joint_limits_rad"],
                    phase="retained_command_replay",
                )
            except NativeJointStateBoundsError as exc:
                result["native_joint_state_violation"] = {
                    **exc.readback,
                    "step_index": source["step_index"],
                }
                result["diagnostic_outcome"] = "native_joint_limit_violation_observed"
                break
            if source["step_index"] == MAX_COMMANDS - 1:
                result["frames"].append(
                    adapters.snapshot("before_original_boundary", root / "frames")
                )
            before = after
        result["frames"].append(adapters.snapshot("terminal", root / "frames"))
        result["diagnostic_outcome"] = result.get(
            "diagnostic_outcome", "no_violation_in_174_retained_commands"
        )
        result["status"] = "completed"
    except Exception as exc:
        result["blockers"].append(type(exc).__name__ + ":" + str(exc))
    seal(result)
    (root / "native_task_retained_command_replay.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
