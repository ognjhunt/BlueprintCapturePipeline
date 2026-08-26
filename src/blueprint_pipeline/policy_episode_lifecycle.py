"""Fail-closed lifecycle contract for learned-policy episodes.

The scientific episode does not begin when a worker process starts.  It begins
only after the environment, policy control plane, evidence sink, and canonical
reset have all been exercised and retained in a readiness receipt.  Once that
boundary is crossed, an accepted terminal receipt has exactly three possible
meanings: the planned duration completed, a candidate action was refused by a
predeclared safety boundary, or a predeclared scientific boundary was reached.

An unexpected infrastructure exception after the boundary is deliberately not
converted into an episode result.  It is a lifecycle invariant violation and
remains a blocker requiring a durable repair.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:  # flat provider-bundle layout
    from decision_evidence_contracts import canonical_digest
except ModuleNotFoundError:  # repository package
    from .decision_evidence_contracts import canonical_digest


LIFECYCLE_SCHEMA_VERSION = "policy_episode_lifecycle.v1"
READINESS_SCHEMA_VERSION = "policy_episode_prestart_readiness.v1"

TERMINAL_PLANNED_DURATION = "planned_duration_complete"
TERMINAL_POLICY_SAFETY = "policy_safety_terminal"
TERMINAL_SCIENTIFIC = "scientific_terminal"
ALLOWED_TERMINAL_CLASSES = frozenset(
    {
        TERMINAL_PLANNED_DURATION,
        TERMINAL_POLICY_SAFETY,
        TERMINAL_SCIENTIFIC,
    }
)

REQUIRED_READINESS_CHECKS = (
    "environment_reset",
    "joint_limits_readback",
    "joint_state_readback",
    "task_state_readback",
    "policy_observation_built",
    "policy_control_plane_ready",
    "evidence_storage_reserved",
    "exact_media_write_readback",
    "multicamera_write_readback",
    "review_video_encode_readback",
    "environment_step_readback",
    "canonical_reset_restored",
)


class PolicyEpisodeLifecycleError(ValueError):
    """A receipt cannot support the episode lifecycle claim."""

    def __init__(self, errors: list[str] | tuple[str, ...]):
        self.errors = tuple(sorted({str(error) for error in errors if str(error)}))
        super().__init__(";".join(self.errors))


def seal_prestart_readiness(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and digest one outcome-blind pre-start rehearsal receipt."""

    receipt = dict(payload)
    receipt["schema_version"] = READINESS_SCHEMA_VERSION
    checks = receipt.get("checks")
    errors: list[str] = []
    if not isinstance(checks, Mapping):
        errors.append("policy_episode_readiness_checks_missing")
    else:
        for check in REQUIRED_READINESS_CHECKS:
            if checks.get(check) is not True:
                errors.append(f"policy_episode_readiness_check_failed:{check}")
    if receipt.get("candidate_policy_queried") is not False:
        errors.append("policy_episode_readiness_queried_candidate")
    if receipt.get("policy_state_advanced") is not False:
        errors.append("policy_episode_readiness_advanced_policy_state")
    if receipt.get("canonical_reset_restored") is not True:
        errors.append("policy_episode_readiness_canonical_reset_not_restored")
    storage = receipt.get("storage_reservation")
    if (
        not isinstance(storage, Mapping)
        or isinstance(storage.get("required_free_bytes"), bool)
        or not isinstance(storage.get("required_free_bytes"), int)
        or storage.get("required_free_bytes", 0) <= 0
        or isinstance(storage.get("observed_free_bytes"), bool)
        or not isinstance(storage.get("observed_free_bytes"), int)
        or storage.get("observed_free_bytes", -1)
        < storage.get("required_free_bytes", 0)
        or storage.get("projection_is_conservative") is not True
    ):
        errors.append("policy_episode_readiness_storage_reservation_invalid")
    control_plane = receipt.get("policy_control_plane")
    if (
        not isinstance(control_plane, Mapping)
        or control_plane.get("identity_verified") is not True
        or control_plane.get("candidate_policy_queried") not in {None, False}
        or control_plane.get("candidate_inference_performed") not in {None, False}
        or control_plane.get("policy_state_advanced") not in {None, False}
    ):
        errors.append("policy_episode_readiness_control_plane_invalid")
    visual = receipt.get("visual_evidence")
    if (
        not isinstance(visual, Mapping)
        or visual.get("status") != "complete"
        or set(visual.get("required_camera_ids") or ())
        != {"external", "wrist", "overview"}
        or set(visual.get("review_only_camera_ids") or ()) != {"overview"}
        or visual.get("terminal_observation_present") is not True
        or set(visual.get("videos") or {}) != {"external", "wrist", "overview"}
    ):
        errors.append("policy_episode_readiness_visual_evidence_invalid")
    if not isinstance(receipt.get("media_artifacts"), list) or not receipt.get(
        "media_artifacts"
    ):
        errors.append("policy_episode_readiness_media_artifacts_missing")
    if errors:
        raise PolicyEpisodeLifecycleError(errors)
    receipt["readiness_digest"] = canonical_digest(
        receipt, digest_field="readiness_digest"
    )
    return receipt


def build_lifecycle(
    *,
    readiness: Mapping[str, Any],
    terminal_class: str,
    planned_policy_queries: int,
    planned_action_steps: int,
    planned_settle_steps: int,
    actual_policy_queries: int,
    actual_action_steps: int,
    actual_settle_steps: int,
    terminal_reason: str,
    retained_terminal_result: bool,
) -> dict[str, Any]:
    """Build the lifecycle section shared by every accepted episode receipt."""

    readiness_receipt = dict(readiness)
    if (
        readiness_receipt.get("schema_version") != READINESS_SCHEMA_VERSION
        or readiness_receipt.get("readiness_digest")
        != canonical_digest(readiness_receipt, digest_field="readiness_digest")
    ):
        raise PolicyEpisodeLifecycleError(
            ["policy_episode_readiness_receipt_invalid"]
        )
    terminal = str(terminal_class)
    if terminal not in ALLOWED_TERMINAL_CLASSES:
        raise PolicyEpisodeLifecycleError(
            ["policy_episode_terminal_class_invalid"]
        )
    lifecycle = {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "readiness_receipt_digest": readiness_receipt["readiness_digest"],
        "episode_started": True,
        "start_boundary": (
            "after_outcome_blind_readiness_rehearsal_and_canonical_reset;"
            "before_first_scientific_policy_observation"
        ),
        "terminal_class": terminal,
        "terminal_reason": str(terminal_reason),
        "retained_terminal_result": bool(retained_terminal_result),
        "post_start_infrastructure_failure": False,
        "planned_policy_queries": int(planned_policy_queries),
        "planned_action_steps": int(planned_action_steps),
        "planned_settle_steps": int(planned_settle_steps),
        "actual_policy_queries": int(actual_policy_queries),
        "actual_action_steps": int(actual_action_steps),
        "actual_settle_steps": int(actual_settle_steps),
        "full_planned_duration_completed": terminal == TERMINAL_PLANNED_DURATION,
        "policy_safety_terminal": terminal == TERMINAL_POLICY_SAFETY,
        "scientific_terminal": terminal == TERMINAL_SCIENTIFIC,
    }
    lifecycle["lifecycle_digest"] = canonical_digest(
        lifecycle, digest_field="lifecycle_digest"
    )
    return lifecycle


def validate_policy_episode_lifecycle(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized lifecycle or fail closed on any claim mismatch."""

    lifecycle = receipt.get("lifecycle")
    readiness = receipt.get("prestart_readiness")
    errors: list[str] = []
    if not isinstance(lifecycle, Mapping):
        errors.append("policy_episode_lifecycle_missing")
        lifecycle = {}
    if not isinstance(readiness, Mapping):
        errors.append("policy_episode_prestart_readiness_missing")
        readiness = {}
    if lifecycle.get("schema_version") != LIFECYCLE_SCHEMA_VERSION:
        errors.append("policy_episode_lifecycle_schema_invalid")
    if lifecycle.get("lifecycle_digest") != canonical_digest(
        lifecycle, digest_field="lifecycle_digest"
    ):
        errors.append("policy_episode_lifecycle_digest_invalid")
    if readiness.get("schema_version") != READINESS_SCHEMA_VERSION:
        errors.append("policy_episode_readiness_schema_invalid")
    if readiness.get("readiness_digest") != canonical_digest(
        readiness, digest_field="readiness_digest"
    ):
        errors.append("policy_episode_readiness_digest_invalid")
    try:
        seal_prestart_readiness(readiness)
    except PolicyEpisodeLifecycleError as exc:
        errors.extend(exc.errors)
    if lifecycle.get("readiness_receipt_digest") != readiness.get(
        "readiness_digest"
    ):
        errors.append("policy_episode_lifecycle_readiness_binding_invalid")
    if lifecycle.get("episode_started") is not True:
        errors.append("policy_episode_started_not_proven")
    terminal = lifecycle.get("terminal_class")
    if terminal not in ALLOWED_TERMINAL_CLASSES:
        errors.append("policy_episode_terminal_class_invalid")
    if lifecycle.get("retained_terminal_result") is not True:
        errors.append("policy_episode_terminal_result_not_retained")
    if lifecycle.get("post_start_infrastructure_failure") is not False:
        errors.append("policy_episode_post_start_infrastructure_failure")
    if not str(lifecycle.get("terminal_reason") or "").strip():
        errors.append("policy_episode_terminal_reason_missing")
    terminal_flags = {
        TERMINAL_PLANNED_DURATION: lifecycle.get(
            "full_planned_duration_completed"
        ),
        TERMINAL_POLICY_SAFETY: lifecycle.get("policy_safety_terminal"),
        TERMINAL_SCIENTIFIC: lifecycle.get("scientific_terminal"),
    }
    if any(
        value is not (name == terminal)
        for name, value in terminal_flags.items()
    ):
        errors.append("policy_episode_terminal_flags_invalid")

    planned_queries = lifecycle.get("planned_policy_queries")
    planned_actions = lifecycle.get("planned_action_steps")
    planned_settle = lifecycle.get("planned_settle_steps")
    actual_queries = lifecycle.get("actual_policy_queries")
    actual_actions = lifecycle.get("actual_action_steps")
    actual_settle = lifecycle.get("actual_settle_steps")
    counts = (
        planned_queries,
        planned_actions,
        planned_settle,
        actual_queries,
        actual_actions,
        actual_settle,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts
    ):
        errors.append("policy_episode_lifecycle_counts_invalid")
    elif planned_queries < 1 or planned_actions < 1 or planned_settle < 1:
        errors.append("policy_episode_lifecycle_plan_invalid")
    elif terminal == TERMINAL_PLANNED_DURATION:
        if (
            actual_queries != planned_queries
            or actual_actions != planned_actions
            or actual_settle != planned_settle
            or lifecycle.get("full_planned_duration_completed") is not True
        ):
            errors.append("policy_episode_planned_duration_not_completed")
    elif terminal in {TERMINAL_POLICY_SAFETY, TERMINAL_SCIENTIFIC}:
        if actual_queries > planned_queries or actual_actions > planned_actions:
            errors.append("policy_episode_terminal_counts_exceed_plan")
        if lifecycle.get("full_planned_duration_completed") is not False:
            errors.append("policy_episode_early_terminal_marked_full_duration")

    visual = receipt.get("visual_evidence")
    if (
        not isinstance(visual, Mapping)
        or visual.get("status") != "complete"
        or set(visual.get("required_camera_ids") or ())
        != {"external", "wrist", "overview"}
        or set(visual.get("review_only_camera_ids") or ()) != {"overview"}
        or visual.get("terminal_observation_present") is not True
        or set(visual.get("videos") or {}) != {"external", "wrist", "overview"}
    ):
        errors.append("policy_episode_terminal_visual_evidence_incomplete")
    frames = receipt.get("candidate_exact_policy_input_frames")
    if (
        not isinstance(frames, list)
        or not frames
        or receipt.get("candidate_exact_policy_input_manifest_digest")
        != canonical_digest({"frames": frames})
        or receipt.get("observation_trace_digest")
        != canonical_digest({"observations": frames})
        or receipt.get("policy_observations_retained") != len(frames)
    ):
        errors.append("policy_episode_exact_policy_inputs_invalid")
    if not isinstance(receipt.get("media_artifacts"), list) or not receipt.get(
        "media_artifacts"
    ):
        errors.append("policy_episode_media_artifacts_missing")
    if receipt.get("receipt_digest") != canonical_digest(
        receipt, digest_field="receipt_digest"
    ):
        errors.append("policy_episode_receipt_digest_invalid")
    if errors:
        raise PolicyEpisodeLifecycleError(errors)
    return dict(lifecycle)


__all__ = [
    "ALLOWED_TERMINAL_CLASSES",
    "LIFECYCLE_SCHEMA_VERSION",
    "PolicyEpisodeLifecycleError",
    "READINESS_SCHEMA_VERSION",
    "REQUIRED_READINESS_CHECKS",
    "TERMINAL_PLANNED_DURATION",
    "TERMINAL_POLICY_SAFETY",
    "TERMINAL_SCIENTIFIC",
    "build_lifecycle",
    "seal_prestart_readiness",
    "validate_policy_episode_lifecycle",
]
