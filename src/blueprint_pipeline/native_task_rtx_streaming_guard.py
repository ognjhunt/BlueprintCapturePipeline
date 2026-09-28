"""Bound stale Isaac RTX streaming waits after a proven camera gate.

IsaacLab's RTX renderer waits up to 30 seconds whenever USD reports that stage
streaming is busy.  A missing optional MDL material can leave that status stuck
forever even though the exact task cameras have already rendered and passed the
policy observation gate.  Paying that 30-second wait on every simulator step
turned the second Quick-10 cell into a 15-minute timeout.

This guard is deliberately applied only *after* the lossless external, wrist,
and overview frames pass the native observation-integrity gate.  It preserves a
small bounded wait and the renderer's final app update; it does not skip camera
qualification or suppress later evidence.
"""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "policy_canary_post_gate_rtx_streaming_guard.v1"
MAXIMUM_POST_GATE_WAIT_SECONDS = 1.0


class RtxStreamingGuardError(RuntimeError):
    """The pinned renderer did not expose the expected bounded wait seam."""


def configure_post_gate_rtx_streaming_wait(
    *,
    observation_gate: Mapping[str, Any],
    output_path: str | Path,
    renderer_utils: Any | None = None,
    maximum_wait_seconds: float = MAXIMUM_POST_GATE_WAIT_SECONDS,
) -> dict[str, Any]:
    """Bound later streaming waits after the exact camera gate has passed."""

    gate = dict(observation_gate) if isinstance(observation_gate, Mapping) else {}
    if (
        gate.get("status") != "passed"
        or gate.get("policy_observation_integrity_passed") is not True
        or gate.get("candidate_policy_loaded") is not False
        or gate.get("candidate_policy_queried") is not False
        or gate.get("gate_digest")
        != canonical_digest(gate, digest_field="gate_digest")
    ):
        raise RtxStreamingGuardError("post_gate_rtx_streaming_observation_gate_invalid")
    try:
        requested = float(maximum_wait_seconds)
    except (TypeError, ValueError) as exc:
        raise RtxStreamingGuardError(
            "post_gate_rtx_streaming_wait_invalid"
        ) from exc
    if not math.isfinite(requested) or not 0.1 <= requested <= 2.0:
        raise RtxStreamingGuardError("post_gate_rtx_streaming_wait_invalid")
    module = renderer_utils or importlib.import_module(
        "isaaclab_physx.renderers.isaac_rtx_renderer_utils"
    )
    previous = getattr(module, "_STREAMING_WAIT_TIMEOUT_S", None)
    busy_probe = getattr(module, "_get_stage_streaming_busy", None)
    if (
        isinstance(previous, bool)
        or not isinstance(previous, (int, float))
        or not math.isfinite(float(previous))
        or float(previous) <= 0
        or not callable(busy_probe)
    ):
        raise RtxStreamingGuardError("post_gate_rtx_streaming_runtime_seam_invalid")
    busy = bool(busy_probe())
    applied = min(float(previous), requested)
    setattr(module, "_STREAMING_WAIT_TIMEOUT_S", applied)
    if float(getattr(module, "_STREAMING_WAIT_TIMEOUT_S", -1.0)) != applied:
        raise RtxStreamingGuardError("post_gate_rtx_streaming_wait_readback_failed")
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "configured",
        "observation_gate_digest": gate["gate_digest"],
        "streaming_busy_after_gate": busy,
        "previous_wait_timeout_seconds": float(previous),
        "maximum_wait_timeout_seconds": applied,
        "camera_qualification_skipped": False,
        "later_frames_remain_required": True,
        "claim_ceiling": "diagnostic_policy_execution",
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    destination = Path(output_path).expanduser()
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    except FileExistsError:
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != payload:
            raise RtxStreamingGuardError("post_gate_rtx_streaming_receipt_conflict")
    return receipt


__all__ = [
    "MAXIMUM_POST_GATE_WAIT_SECONDS",
    "RtxStreamingGuardError",
    "configure_post_gate_rtx_streaming_wait",
]
