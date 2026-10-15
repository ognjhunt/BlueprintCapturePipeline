"""A website scene admits a team evaluation only after its robot-free probe passed.

Beta journey step: L5 team policy evaluation. Owner decision 2026-10-12: a
configured scene becomes ready for robot teams' evaluation when, under the same
scorer their episodes are judged by, the scene left untouched does not succeed
and the task driven into its goal without a robot does
(``task_evaluation_scene_robot_free_readiness``). That receipt existed but no
admission read it, so a scene whose success contract could not tell success
from failure could still be evaluated, and a team could be scored against it.

This gate makes the receipt mandatory. It must be the sealed readiness receipt,
self-digest intact, ``passed`` with both checks true, scored without a robot,
and bound to the scene's *current* configured revision (whose digest binds the
task documents the receipt scored) and, when the caller knows it, to the exact
task specification the evaluation will score. Anything else fails closed with
a typed reason.

Two receipt kinds exist: articulated open/close
(``task_evaluation_scene_robot_free_readiness``) and rigid pick/place
(``task_evaluation_scene_rigid_robot_free_readiness``), each with its own
required checks. A receipt of one kind never admits the other, and every other
task kind is refused by name until it has a probe.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

# The readiness modules' SCHEMA and REQUIRED_CHECKS, kept literal so this gate
# imports no simulator-side scoring code; a test pins them together.
RECEIPT_SCHEMA = "task_evaluation_configured_scene_robot_free_readiness.v1"
RIGID_RECEIPT_SCHEMA = "task_evaluation_configured_scene_rigid_robot_free_readiness.v1"
RECEIPT_FILENAME = "robot_free_readiness.v1.json"
RECEIPT_TASK_KINDS = {RECEIPT_SCHEMA: "articulated_open_close", RIGID_RECEIPT_SCHEMA: "rigid_pick_place"}
REQUIRED_CHECKS = {
    "articulated_open_close": frozenset({"zero_action_not_success", "direct_drive_success"}),
    "rigid_pick_place": frozenset({
        "zero_action_not_success", "direct_placement_success", "contract_untouched_not_success",
        "contract_goal_success", "contract_short_hold_not_success", "contract_drift_not_success",
        "contract_outside_not_success"}),
}
PROBE_TASK_KINDS = frozenset(REQUIRED_CHECKS)
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class RobotFreeProbeGateError(ValueError):
    """The scene's robot-free probe does not admit a team evaluation."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__("team_evaluation_robot_free_probe_" + reason)


def require_robot_free_probe(receipt: Mapping[str, Any] | None, *, scene_revision_digest: str,
                             task_spec_digest: str | None = None,
                             task_kind: str | None = None,
                             task_success_contract_digest: str | None = None) -> dict[str, Any]:
    """Return the admitted receipt's binding, or raise with a typed reason."""

    if task_kind is not None and task_kind not in PROBE_TASK_KINDS:
        raise RobotFreeProbeGateError("unavailable_for_task_kind:" + str(task_kind))
    if not isinstance(scene_revision_digest, str) or not _DIGEST.fullmatch(scene_revision_digest):
        raise RobotFreeProbeGateError("scene_revision_unbound")
    if not isinstance(receipt, Mapping):
        raise RobotFreeProbeGateError("receipt_missing")
    kind = RECEIPT_TASK_KINDS.get(str(receipt.get("schema_version")))
    if kind is None:
        raise RobotFreeProbeGateError("receipt_schema_invalid")
    if receipt.get("receipt_digest") != canonical_digest(dict(receipt), digest_field="receipt_digest"):
        raise RobotFreeProbeGateError("receipt_digest_invalid")
    if task_kind is not None and kind != task_kind:
        raise RobotFreeProbeGateError("task_kind_mismatch")
    checks = receipt.get("checks")
    boundary = receipt.get("proof_boundary")
    if (receipt.get("status") != "passed" or receipt.get("learned_policy_evaluation_admitted") is not True
            or not isinstance(checks, Mapping) or not REQUIRED_CHECKS[kind] <= set(checks)
            or any(value is not True for value in checks.values())):
        raise RobotFreeProbeGateError("not_passed")
    if not isinstance(boundary, Mapping) or boundary.get("robot_embodiment_used") is not False:
        raise RobotFreeProbeGateError("robot_used")
    if receipt.get("configured_scene_revision_digest") != scene_revision_digest:
        raise RobotFreeProbeGateError("scene_revision_mismatch")
    if task_spec_digest is not None and receipt.get("task_spec_digest") != task_spec_digest:
        raise RobotFreeProbeGateError("task_spec_mismatch")
    if (task_success_contract_digest is not None
            and receipt.get("task_success_contract_digest") != task_success_contract_digest):
        raise RobotFreeProbeGateError("task_success_contract_mismatch")
    return {"receipt_digest": receipt["receipt_digest"], "scene_revision_digest": scene_revision_digest,
            "task_spec_digest": receipt.get("task_spec_digest")}


def retained_probe_receipt(root: Path) -> dict[str, Any] | None:
    """Read the receipt retained beside a source launch; never follow a symlink."""

    path = Path(root) / RECEIPT_FILENAME
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise RobotFreeProbeGateError("receipt_path_unsafe")
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RobotFreeProbeGateError("receipt_unreadable") from exc
    return value if isinstance(value, dict) else {}


__all__ = [
    "PROBE_TASK_KINDS",
    "RECEIPT_FILENAME",
    "RECEIPT_SCHEMA",
    "RECEIPT_TASK_KINDS",
    "REQUIRED_CHECKS",
    "RIGID_RECEIPT_SCHEMA",
    "RobotFreeProbeGateError",
    "require_robot_free_probe",
    "retained_probe_receipt",
]
