"""Validate a structured policy canary's output member without a local archive path.

Moved out of ``vast_provider_adapter`` so an archive read by range can be
inspected exactly as a local ZIP is. The adapter keeps
``_inspect_structured_policy_canary_output(path)`` as the path wrapper.
"""

from __future__ import annotations

import json
import math
import re
import zipfile
from typing import Any, Mapping

STRUCTURED_POLICY_CANARY_MEMBER = "policy_structured_canary.json"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def inspect_structured_policy_canary_archive(archive: zipfile.ZipFile | None) -> dict[str, Any]:
    """Validate a structured policy canary without requiring rollout video.

    ``None`` means the output archive is missing. A member that cannot be
    read or parsed is ``structured_policy_canary_member_invalid``.
    """

    blockers: list[str] = []
    payload: dict[str, Any] = {}
    if archive is None:
        blockers.append("structured_policy_canary_output_zip_missing")
    else:
        try:
            if STRUCTURED_POLICY_CANARY_MEMBER not in archive.namelist():
                blockers.append("structured_policy_canary_member_missing")
            else:
                value = json.loads(archive.read(STRUCTURED_POLICY_CANARY_MEMBER))
                payload = _mapping(value)
        except (OSError, UnicodeError, json.JSONDecodeError, zipfile.BadZipFile):
            blockers.append("structured_policy_canary_member_invalid")
    return structured_policy_canary_summary(payload, blockers)


def structured_policy_canary_summary(
    payload: Mapping[str, Any], blockers: list[str]
) -> dict[str, Any]:
    """Judge one parsed ``policy_structured_canary.json`` payload."""

    payload = _mapping(payload)
    blockers = list(blockers)
    native_action = payload.get("native_action")
    wam_prefix = payload.get("wam_prefix_action")
    executed_action = payload.get("executed_action")
    commanded_joint = payload.get("commanded_next_joint_position")
    commanded_gripper = payload.get("commanded_next_gripper_position")
    endpoint = _mapping(payload.get("policy_endpoint_evidence"))
    receipt = _mapping(payload.get("policy_request_receipt"))
    server_metadata = _mapping(endpoint.get("server_metadata"))
    if payload and payload.get("status") != "passed":
        blockers.append("structured_policy_canary_status_not_passed")
    if payload and endpoint.get("identity_verified") is not True:
        blockers.append("structured_policy_canary_identity_not_verified")
    if payload and endpoint.get("request_count") != 1:
        blockers.append("structured_policy_canary_request_count_invalid")
    native_shape_valid = bool(
        isinstance(native_action, list)
        and len(native_action) == 32
        and all(isinstance(row, list) and len(row) == 8 for row in native_action)
    )
    if payload and not native_shape_valid:
        blockers.append("structured_policy_canary_native_action_shape_invalid")
    if (
        payload
        and native_shape_valid
        and not all(
            type(value) in {int, float} and math.isfinite(float(value))
            for row in native_action
            for value in row
        )
    ):
        blockers.append("structured_policy_canary_native_action_not_finite")
    if payload and not (
        isinstance(wam_prefix, list)
        and len(wam_prefix) == 16
        and isinstance(native_action, list)
        and wam_prefix == native_action[:16]
    ):
        blockers.append("structured_policy_canary_wam_prefix_invalid")
    if payload and not (
        isinstance(executed_action, list)
        and len(executed_action) == 8
        and isinstance(native_action, list)
        and executed_action == native_action[:8]
    ):
        blockers.append("structured_policy_canary_executed_prefix_invalid")
    if payload and not (
        isinstance(native_action, list)
        and len(native_action) == 32
        and commanded_joint == native_action[7][:7]
        and commanded_gripper == [native_action[7][7]]
    ):
        blockers.append("structured_policy_canary_commanded_state_invalid")
    expected_receipt_shapes = {
        "native_action_shape": [32, 8],
        "wam_prefix_action_shape": [16, 8],
        "executed_prefix_steps": 8,
    }
    for key, expected in expected_receipt_shapes.items():
        if payload and receipt.get(key) != expected:
            blockers.append(f"structured_policy_canary_receipt_{key}_invalid")
    for key in (
        "server_identity_sha256",
        "observation_sha256",
        "native_action_sha256",
        "wam_prefix_action_sha256",
        "executed_prefix_action_sha256",
        "commanded_next_state_sha256",
        "receipt_sha256",
    ):
        if payload and not re.fullmatch(r"[0-9a-f]{64}", _string(receipt.get(key))):
            blockers.append(f"structured_policy_canary_receipt_{key}_invalid")

    return {
        "status": "passed" if payload and not blockers else "blocked",
        "blockers": blockers,
        "identity_verified": endpoint.get("identity_verified") is True,
        "request_count": endpoint.get("request_count"),
        "policy_id": server_metadata.get("policy_id"),
        "model_revision": server_metadata.get("model_revision"),
        "server_identity_sha256": receipt.get("server_identity_sha256"),
        "observation_sha256": receipt.get("observation_sha256"),
        "native_action_sha256": receipt.get("native_action_sha256"),
        "wam_prefix_action_sha256": receipt.get("wam_prefix_action_sha256"),
        "executed_prefix_action_sha256": receipt.get("executed_prefix_action_sha256"),
        "commanded_next_state_sha256": receipt.get("commanded_next_state_sha256"),
        "receipt_sha256": receipt.get("receipt_sha256"),
        "raw_secret_values_recorded": False,
    }


__all__ = [
    "STRUCTURED_POLICY_CANARY_MEMBER",
    "inspect_structured_policy_canary_archive",
    "structured_policy_canary_summary",
]
