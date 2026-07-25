"""Validation of the initial real policy observation for sealed episodes.

The sealed manipulation path must query the learned policy on the initial real
Isaac head-POV frame; substituting a deterministic fixture action for action
zero poisons every downstream identity binding, and in the qualification lane
the substituted walk token then crashes FK skeleton conditioning with
``unitree_g1_sonic_action_missing``.

The producer of this evidence is ``isaac_runtime_task_backend`` — its sealed
``initial_policy_observation.json`` nests ``camera_contract`` and
``visual_signal`` inside ``camera_projection_context`` and carries
``source_frame_artifact`` at the top level.  The #178 validator read all three
from the top level only, a shape no producer has ever emitted, so every
endpoint-configured episode since #178 silently fell back to the deterministic
walk policy (first observed live on Vast instance 45785933, attempt 066).
This module accepts both layouts: top level first, producer nesting second.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _resolved_section(evidence: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Read a contract section from the top level or the producer's nesting."""

    top = _mapping(evidence.get(key))
    if top:
        return top
    return _mapping(_mapping(evidence.get("camera_projection_context")).get(key))


def validated_initial_policy_observation(
    evidence: Mapping[str, Any] | None,
    *,
    start_frame_path: str | Path,
) -> dict[str, Any]:
    """Validate that the initial policy RGB is the hash-bound Isaac head POV."""

    context = _mapping(evidence)
    if not context:
        raise RuntimeError("initial_policy_observation_evidence_required")
    frame = _resolved_section(context, "source_frame_artifact")
    camera = _resolved_section(context, "camera_contract")
    visual_signal = _resolved_section(context, "visual_signal")
    resolved_start = Path(start_frame_path).expanduser().resolve()
    evidence_path = Path(_string(frame.get("path"))).expanduser().resolve()
    if (
        resolved_start.is_symlink()
        or not resolved_start.is_file()
        or evidence_path != resolved_start
    ):
        raise RuntimeError("initial_policy_observation_frame_binding_invalid")
    observed_sha256 = hashlib.sha256(resolved_start.read_bytes()).hexdigest()
    expected_sha256 = _string(frame.get("sha256")).strip().lower()
    if _SHA256_RE.fullmatch(expected_sha256) is None or expected_sha256 != observed_sha256:
        raise RuntimeError("initial_policy_observation_frame_sha256_mismatch")
    frame_resolution = [int(frame.get("width") or 0), int(frame.get("height") or 0)]
    if (
        frame.get("camera_role") != "robot_pov"
        or min(frame_resolution) <= 0
        or camera.get("available") is not True
        or camera.get("projection_token") != "perspective"
        or list(camera.get("resolution") or []) != frame_resolution
        or camera.get("viewpoint_mode") != "robot_head_mounted_egocentric"
        or camera.get("robot_mounted") is not True
        or camera.get("policy_observation_eligible") is not True
        or camera.get("mount_motion_model") != "rigid_head_local_transform"
        or camera.get("gaze_motion_model") != "inherits_head_orientation_no_task_reaim"
        or visual_signal.get("status") != "completed"
        or visual_signal.get("non_uniform") is not True
    ):
        raise RuntimeError("initial_policy_observation_camera_contract_invalid")
    return {
        "frame_path": str(resolved_start),
        "sha256": observed_sha256,
        "camera_role": "robot_pov",
        "viewpoint_mode": camera["viewpoint_mode"],
        "mount_motion_model": camera["mount_motion_model"],
        "gaze_motion_model": camera["gaze_motion_model"],
        "policy_observation_eligible": True,
        "third_person_overview_included": False,
        "camera_contract": dict(camera),
    }
