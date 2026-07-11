"""External full-episode semantic review adapter for G1 kitchen media."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


COMMAND_ENV = "BLUEPRINT_G1_KITCHEN_SEMANTIC_REVIEW_COMMAND"
ALLOW_ENV = "BLUEPRINT_ALLOW_G1_KITCHEN_SEMANTIC_REVIEW"
INPUT_ENV = "BLUEPRINT_G1_KITCHEN_SEMANTIC_REVIEW_INPUT"
OUTPUT_ENV = "BLUEPRINT_G1_KITCHEN_SEMANTIC_REVIEW_OUTPUT"
ATTESTATION_OUTPUT_ENV = "BLUEPRINT_G1_KITCHEN_SEMANTIC_REVIEW_ATTESTATION_OUTPUT"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def run_full_episode_semantic_review(
    *,
    scenario_dir: str | Path,
    expected_frame_count: int,
    command: str | None = None,
    allow: bool | None = None,
    timeout_seconds: float = 600.0,
    attestation_pins: Mapping[str, Any] | None = None,
    identity_binding: Mapping[str, Any] | None = None,
    step_bindings: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(scenario_dir).resolve()
    frames_dir = root / "frames"
    rows: list[dict[str, Any]] = []
    for role in ("overview", "robot_pov"):
        for path in sorted(frames_dir.glob(f"{role}_[0-9][0-9][0-9][0-9].png")):
            rows.append(
                {
                    "camera_role": role,
                    "frame_index": int(path.stem.rsplit("_", 1)[-1]),
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                }
            )
    request = {
        "schema_version": "g1_kitchen_full_episode_semantic_review_request.v1",
        "expected_frame_count_per_camera": int(expected_frame_count),
        "required_camera_roles": ["overview", "robot_pov"],
        "frames": rows,
        "identity_binding": dict(identity_binding or {}),
        "step_bindings": [dict(item) for item in (step_bindings or [])],
        "required_overview_fields": [
            "g1_visible",
            "target_visible",
            "floor_support_visible",
            "orientation_visible",
            "clearance_visible",
            "robot_pixel_occupancy",
            "target_pixel_occupancy",
        ],
        "required_robot_pov_fields": [
            "target_visible",
            "active_hand_wrist_chain_visible",
        ],
        "claim_boundary": {
            "semantic_review_is_separate_from_task_success": True,
            "semantic_review_may_abstain": True,
            "schematic_topdown_is_ineligible": True,
        },
    }
    request_path = root / "full_episode_semantic_review_request.json"
    raw_output_path = root / "full_episode_semantic_review_raw.json"
    attestation_path = root / "full_episode_semantic_review_raw.attestation.json"
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
    configured = str(command or os.environ.get(COMMAND_ENV) or "").strip()
    allowed = _truthy(os.environ.get(ALLOW_ENV)) if allow is None else bool(allow)
    blockers: list[str] = []
    if not allowed:
        blockers.append(f"missing_explicit_allow:{ALLOW_ENV}")
    argv = shlex.split(configured)
    if not argv:
        blockers.append(f"missing_semantic_review_command:{COMMAND_ENV}")
    if len(rows) != 2 * int(expected_frame_count):
        blockers.append("semantic_review_request_frame_count_mismatch")
    completed = None
    if not blockers:
        raw_output_path.unlink(missing_ok=True)
        attestation_path.unlink(missing_ok=True)
        completed = subprocess.run(
            argv,
            env={
                **os.environ,
                INPUT_ENV: str(request_path),
                OUTPUT_ENV: str(raw_output_path),
                ATTESTATION_OUTPUT_ENV: str(attestation_path),
            },
            capture_output=True,
            text=True,
            check=False,
            timeout=max(1.0, float(timeout_seconds)),
        )
        if completed.returncode != 0:
            blockers.append(f"semantic_review_command_exit:{completed.returncode}")
        if not raw_output_path.is_file():
            blockers.append("semantic_review_command_output_missing")
    raw: dict[str, Any] = {}
    if raw_output_path.is_file():
        try:
            value = json.loads(raw_output_path.read_text())
            raw = dict(value) if isinstance(value, Mapping) else {}
        except (OSError, json.JSONDecodeError):
            blockers.append("semantic_review_command_output_invalid_json")
    if attestation_pins is not None:
        from .g1_kitchen_proof_row_validation import verify_leaf_attestation

        try:
            attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            attestation = {}
        blockers.extend(
            "semantic_review_" + blocker
            for blocker in verify_leaf_attestation(
                data=raw_output_path.read_bytes() if raw_output_path.is_file() else b"",
                attestation=attestation,
                expected_role="semantic_review",
                pins=attestation_pins,
            )
        )
    if raw.get("status") != "passed":
        blockers.append("semantic_review_api_status_not_passed")
    if raw.get("abstained") is not False:
        blockers.append("semantic_review_api_abstained_or_missing")
    for field in ("review_runtime_id", "provider", "model"):
        if not str(raw.get(field) or "").strip():
            blockers.append(f"semantic_review_api_{field}_missing")
    request_sha256 = _canonical_sha256(request)
    if attestation_pins is not None:
        if raw.get("request_sha256") != request_sha256:
            blockers.append("semantic_review_api_request_sha256_mismatch")
        if dict(raw.get("identity_binding") or {}) != dict(identity_binding or {}):
            blockers.append("semantic_review_api_identity_binding_mismatch")
    reviews = raw.get("frame_reviews")
    reviews = list(reviews) if isinstance(reviews, Sequence) and not isinstance(reviews, (str, bytes)) else []
    expected_by_path = {row["path"]: row for row in rows}
    semantics: dict[str, dict[str, Any]] = {}
    for item in reviews:
        review = dict(item) if isinstance(item, Mapping) else {}
        path = str(Path(str(review.get("path") or "")).resolve())
        expected = expected_by_path.get(path)
        if expected is None or review.get("sha256") != expected["sha256"]:
            blockers.append("semantic_review_frame_identity_mismatch")
            continue
        required = (
            request["required_overview_fields"]
            if expected["camera_role"] == "overview"
            else request["required_robot_pov_fields"]
        )
        if any(field not in review for field in required):
            blockers.append(f"semantic_review_fields_missing:{expected['camera_role']}")
        semantics[path] = review
    if set(semantics) != set(expected_by_path):
        blockers.append("semantic_review_frame_coverage_incomplete")
    response_sha = _sha256(raw_output_path) if raw_output_path.is_file() else None
    normalized = {
        "schema_version": "g1_kitchen_full_episode_semantic_review.v1",
        "status": "passed" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "full_ordered_episode_reviewed": not blockers,
        "abstained": raw.get("abstained") is True,
        "review_runtime_id": raw.get("review_runtime_id"),
        "review_source": "external_semantic_review_api",
        "provider": raw.get("provider"),
        "model": raw.get("model"),
        "request_sha256": request_sha256,
        "response_sha256": response_sha,
        "frame_review_count": len(semantics),
        "command_result": {
            "returncode": completed.returncode if completed is not None else None,
            "stdout_sha256": hashlib.sha256(
                (completed.stdout if completed is not None else "").encode()
            ).hexdigest(),
            "stderr_sha256": hashlib.sha256(
                (completed.stderr if completed is not None else "").encode()
            ).hexdigest(),
        },
    }
    (root / "full_episode_frame_semantics.json").write_text(
        json.dumps({"frames": semantics}, indent=2, sort_keys=True) + "\n"
    )
    (root / "full_episode_semantic_review.json").write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n"
    )
    return normalized
