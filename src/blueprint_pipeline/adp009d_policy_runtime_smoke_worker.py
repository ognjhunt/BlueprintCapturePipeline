"""Seal one outcome-blind real policy-server inference smoke receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "adp009d_policy_runtime_smoke.v1"
EXPECTED_CANDIDATES = frozenset({"pi05_droid", "groot_n17_droid"})
FORBIDDEN_OUTCOME_KEYS = frozenset(
    {
        "candidate_result",
        "episode_success",
        "learned_outcome",
        "policy_result",
        "task_success",
    }
)


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("receipt_digest", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in FORBIDDEN_OUTCOME_KEYS:
                found.append(path)
            found.extend(_forbidden_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_paths(child, f"{prefix}[{index}]"))
    return found


def seal_policy_runtime_smoke(
    *,
    candidate_id: str,
    server_receipt_path: str | Path,
    provisioning_exit_code: int,
) -> dict[str, Any]:
    """Validate one real round trip without importing task or outcome state."""

    blockers: list[str] = []
    receipt_path = Path(server_receipt_path).expanduser().resolve()
    try:
        server = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        server = {}
        blockers.append("policy_runtime_smoke_server_receipt_missing_or_invalid")
    if candidate_id not in EXPECTED_CANDIDATES:
        blockers.append("policy_runtime_smoke_candidate_invalid")
    if provisioning_exit_code != 0:
        blockers.append("policy_runtime_smoke_provisioning_failed")
    if server.get("candidate_id") != candidate_id:
        blockers.append("policy_runtime_smoke_candidate_identity_mismatch")
    if server.get("status") != "ready" or server.get("round_trip_completed") is not True:
        blockers.append("policy_runtime_smoke_round_trip_not_completed")
    rows = server.get("action_chunk_rows")
    width = server.get("action_chunk_width")
    if (
        isinstance(rows, bool)
        or not isinstance(rows, int)
        or rows < 8
        or width != 8
    ):
        blockers.append("policy_runtime_smoke_action_shape_invalid")
    if server.get("server_stopped_after_round_trip") is not True:
        blockers.append("policy_runtime_smoke_server_not_stopped")
    pid = server.get("server_pid")
    if isinstance(pid, int) and not isinstance(pid, bool) and Path(f"/proc/{pid}").exists():
        blockers.append("policy_runtime_smoke_server_process_still_alive")
    forbidden = _forbidden_paths(server)
    if forbidden:
        blockers.append("policy_runtime_smoke_outcome_field_forbidden")

    receipt_sha256 = None
    if receipt_path.is_file() and not receipt_path.is_symlink():
        receipt_sha256 = "sha256:" + hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if not blockers else "blocked",
        "candidate_id": candidate_id,
        "candidate_policy_queried": server.get("round_trip_completed") is True,
        "candidate_outcomes_accessed": False,
        "synthetic_observation_query_count": (
            1 if server.get("round_trip_completed") is True else 0
        ),
        "server_receipt_sha256": receipt_sha256,
        "server_transport": server.get("transport"),
        "action_chunk_shape": [rows, width] if rows is not None else None,
        "server_stopped_after_round_trip": server.get(
            "server_stopped_after_round_trip"
        ),
        "claim_boundary": {
            "real_checkpoint_inference_observed": not blockers,
            "synthetic_static_observation_only": True,
            "actions_executed": False,
            "task_scene_loaded": False,
            "controls_executed": False,
            "policy_ranking": False,
            "task_success_claimed": False,
            "physical_robot_execution": False,
            "claim_ceiling": "development_only_runtime_smoke",
        },
        "blockers": sorted(set(blockers)),
        "raw_secret_values_recorded": False,
    }
    result["receipt_digest"] = _canonical_digest(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--server-receipt", required=True)
    parser.add_argument("--provisioning-exit-code", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = seal_policy_runtime_smoke(
        candidate_id=args.candidate_id,
        server_receipt_path=args.server_receipt,
        provisioning_exit_code=args.provisioning_exit_code,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"BLUEPRINT_ADP009D_POLICY_RUNTIME_SMOKE:{result['status']}")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA_VERSION", "seal_policy_runtime_smoke"]
