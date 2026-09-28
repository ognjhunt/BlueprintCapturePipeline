"""Adopt one torn-down native construction result as initial feedback evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .native_task_arena_packet import validate_native_task_arena_packet_request
from .native_construction_terminal_feedback_contract import (
    BASELINE_SCHEMA_VERSION,
    SCHEMA_VERSION,
    validate_terminal_feedback_adoption,
)
from .task_evaluation_native_construction_feedback_controller import (
    summarize_native_construction_feedback,
    validate_native_construction_inventory,
)


class NativeConstructionTerminalFeedbackError(ValueError):
    """Terminal native evidence was not safe to adopt into a later launch."""


def _read(path: str | Path, *, blocker: str) -> dict[str, Any]:
    source = Path(path).expanduser()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NativeConstructionTerminalFeedbackError(blocker) from exc
    if source.is_symlink() or not isinstance(value, Mapping):
        raise NativeConstructionTerminalFeedbackError(blocker)
    return dict(value)


def materialize_native_construction_terminal_feedback_adoption(
    *,
    allocator_result: Mapping[str, Any],
    native_result: Mapping[str, Any],
    packet_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    allocator = json.loads(json.dumps(dict(allocator_result), allow_nan=False))
    native = json.loads(json.dumps(dict(native_result), allow_nan=False))
    packet = Path(packet_dir).expanduser().resolve()
    request = validate_native_task_arena_packet_request(
        _read(
            packet / "native_task_arena_packet_request.v1.json",
            blocker="terminal_feedback_packet_request_invalid",
        )
    )
    feedback_config = request.get("native_construction_feedback")
    universe = (
        feedback_config.get("candidate_universe")
        if isinstance(feedback_config, Mapping)
        else None
    )
    if not isinstance(universe, Mapping):
        raise NativeConstructionTerminalFeedbackError(
            "terminal_feedback_candidate_universe_missing"
        )
    inventory = validate_native_construction_inventory(
        universe,
        expected_run_id=str(universe.get("run_id") or ""),
        expected_round_index=0,
        expected_feedback_digest=None,
        maximum_candidates=64,
    )
    if (
        allocator.get("schema_version") != "native_task_arena_vast_run.v1"
        or allocator.get("status") != "blocked"
        or allocator.get("retry_cap") != 0
        or allocator.get("continuing_spend_from_this_run") is not False
        or allocator.get("warm_session") is not None
        or allocator.get("warm_session_receipt_path") is not None
        or allocator.get("native_control_result_digest") != native.get("result_digest")
        or not Path(str(allocator.get("native_control_result_path") or "")).is_absolute()
        or allocator.get("result_digest")
        != canonical_digest(allocator, digest_field="result_digest")
        or native.get("schema_version")
        != "native_task_arena_construction_result.v1"
        or native.get("status") != "blocked"
        or native.get("construction_gate_qualified") is not False
        or native.get("result_digest")
        != canonical_digest(native, digest_field="result_digest")
        or request.get("construction_feedback_candidate_binding") is not None
        or not str(feedback_config.get("selected_placement_candidate_id") or "")
    ):
        raise NativeConstructionTerminalFeedbackError(
            "terminal_feedback_evidence_invalid"
        )
    try:
        cost = float(allocator.get("estimated_cost_usd") or 0.0)
        runtime = float(allocator.get("runtime_seconds") or 0.0)
    except (TypeError, ValueError) as exc:
        raise NativeConstructionTerminalFeedbackError(
            "terminal_feedback_cost_runtime_invalid"
        ) from exc
    if not all(math.isfinite(value) and value >= 0.0 for value in (cost, runtime)):
        raise NativeConstructionTerminalFeedbackError(
            "terminal_feedback_cost_runtime_invalid"
        )
    feedback = summarize_native_construction_feedback(native)
    baseline: dict[str, Any] = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "baseline_kind": "cold_authored_baseline_not_feedback_candidate",
        "selected_placement_candidate_id": feedback_config.get(
            "selected_placement_candidate_id"
        ),
        "robot_base_pose_world": request.get("robot_base_pose_world"),
        "robot_joint_reset_positions_digest": canonical_digest(
            request.get("robot_joint_reset_positions_rad") or {}
        ),
        "camera_configuration_digest": canonical_digest(
            {"cameras": request.get("cameras") or []}
        ),
        "packet_request_digest": request["request_digest"],
        "candidate_universe_digest": inventory["inventory_digest"],
        "allocator_result_digest": allocator["result_digest"],
        "native_result_digest": native["result_digest"],
        "native_feedback_digest": feedback["feedback_digest"],
        "incremental_cost_upper_bound_usd": cost,
        "runtime_seconds": runtime,
        "optuna_trial_recorded": False,
        "candidate_digest": None,
        "binding_digest": "",
    }
    baseline["binding_digest"] = canonical_digest(
        baseline, digest_field="binding_digest"
    )
    checkpoint: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "accepted_for_feedback_bootstrap",
        "run_id": inventory["run_id"],
        "source_allocator_result_digest": allocator["result_digest"],
        "source_native_result_digest": native["result_digest"],
        "packet_request_digest": request["request_digest"],
        "candidate_universe_digest": inventory["inventory_digest"],
        "initial_native_feedback": feedback,
        "prior_attempted_baseline_binding": baseline,
        "prior_attempted_candidate_digests": [],
        "feedback_bootstrap_required": True,
        "baseline_physics_replay_required": False,
        "native_gates_or_thresholds_modified": False,
        "checkpoint_digest": "",
    }
    checkpoint["checkpoint_digest"] = canonical_digest(
        checkpoint, digest_field="checkpoint_digest"
    )
    destination = Path(output_path).expanduser()
    if destination.exists() or destination.is_symlink():
        raise NativeConstructionTerminalFeedbackError(
            "terminal_feedback_output_exists"
        )
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    write_json(destination, checkpoint)
    destination.chmod(0o440)
    return checkpoint


def validate_native_construction_terminal_feedback_adoption(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        return validate_terminal_feedback_adoption(value)
    except ValueError as exc:
        raise NativeConstructionTerminalFeedbackError(
            "terminal_feedback_checkpoint_invalid"
        ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocator-result", required=True)
    parser.add_argument("--native-result", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = materialize_native_construction_terminal_feedback_adoption(
            allocator_result=_read(
                args.allocator_result, blocker="terminal_feedback_allocator_invalid"
            ),
            native_result=_read(
                args.native_result, blocker="terminal_feedback_native_result_invalid"
            ),
            packet_dir=args.packet_dir,
            output_path=args.output,
        )
    except (NativeConstructionTerminalFeedbackError, OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "blockers": [str(exc)]}))
        return 2
    print(json.dumps({"status": result["status"], "output": args.output}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BASELINE_SCHEMA_VERSION",
    "NativeConstructionTerminalFeedbackError",
    "SCHEMA_VERSION",
    "main",
    "materialize_native_construction_terminal_feedback_adoption",
    "validate_native_construction_terminal_feedback_adoption",
]
