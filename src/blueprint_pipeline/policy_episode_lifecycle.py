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
import re
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
    "policy_input_frames_unsaturated",
    "policy_control_plane_ready",
    "evidence_storage_reserved",
    "exact_media_write_readback",
    "multicamera_write_readback",
    "review_video_encode_readback",
    "environment_step_readback",
    "canonical_reset_restored",
)


# A typed observation-quality flag: a lowercase code with at most three typed
# qualifiers, e.g. ``external:task_object:composition_support_...``.
OBSERVATION_QUALITY_FLAG = re.compile(r"[a-z][a-z0-9_]{0,95}(?::[a-z0-9_]{1,96}){0,3}")
OBSERVATION_QUALITY_FLAG_UNTYPED = "observation_gate_blocker_untyped"


def observation_quality_flag_codes(blockers: Any) -> list[str]:
    """The typed observation-quality flags of a gate's blocker list, sorted and unique.

    Doctrine stops a run only for privacy, rights and money, so a beta run
    carries an observation gate's blockers as flags instead of stopping.  A
    blocker that is not already a typed code keeps its leading code
    (``composition_gate_capture_failed:RuntimeError:<message>`` becomes
    ``composition_gate_capture_failed``), and one with no typed code at all is
    kept as ``observation_gate_blocker_untyped``.  No blocker is dropped and no
    message, path or other free text is carried.
    """
    codes: set[str] = set()
    for item in blockers if isinstance(blockers, (list, tuple)) else ():
        text, head = str(item), str(item).split(":", 1)[0]
        codes.add(text if OBSERVATION_QUALITY_FLAG.fullmatch(text)
                  else head if OBSERVATION_QUALITY_FLAG.fullmatch(head) else OBSERVATION_QUALITY_FLAG_UNTYPED)
    return sorted(codes)


def observation_quality_flags_valid(value: Any) -> bool:
    """A non-empty, sorted, unique list of typed observation-quality flags."""
    return (isinstance(value, list) and bool(value) and value == sorted(set(value))
            and all(isinstance(code, str) and OBSERVATION_QUALITY_FLAG.fullmatch(code) for code in value))


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


# --- What the robot's cameras really showed of the job before a beta try: three outcomes --
#
# The pre-policy runtime gate (``policy_canary_runtime_observation_integrity_gate.v1``,
# written by ``native_task_arena_policy_canary_worker``) measures, from the simulator's own
# semantic segmentation, how many pixels of the task object and of the destination marker
# each camera shows, and how many of those pixels the room's appearance (the Gaussian splat)
# covers. Its own verdict is all-or-nothing: one covered pixel blocks it. A beta session then
# continued past every blocked gate whose frames were usable, so on capture-447ae4f4 a gate
# that said ``blocked`` blocked nothing (39 tries ran and were graded).
#
# The functions below read the gate's measurements and decide:
#
# - ``visible``: the task object and the marker are each seen, clean, in a camera the policy
#   is given. Tries run and count.
# - ``visible_with_caveat``: each is seen enough in at least one policy camera, but one is partly
#   covered there, or a policy camera does not show it at the start. Tries run and count, and
#   the flags say what was hidden where.
# - ``not_visible``: the object or the marker is not seen enough in any policy camera. No try
#   runs; the reason is a fault in our scene, never a result for the policy.
#
# A gate that does not carry the measurements (an older receipt, or a composition capture
# that failed) is ``unmeasured`` and keeps the earlier handling: its blockers are flags.
#
# Thresholds, chosen on capture-447ae4f4's measurements (tray: 912 of 16,880 interior pixels
# covered in the external view; marker: 2,406 of 16,926):
#
# - a target counts as seen in a camera when at least ``MINIMUM_INTERIOR_PIXELS`` (64, the
#   composition check's own floor) of its unobstructed interior pixels exist there and at
#   least ``MINIMUM_VISIBLE_SHARE`` (half) of them are not covered;
# - it is clean when at most ``CLEAN_MAXIMUM_HIDDEN_SHARE`` (1 %) is covered; above that it is
#   partly hidden. On 447 the tray (5.4 % covered) and the marker (14.2 %) are partly hidden,
#   so 447 is ``visible_with_caveat``.
#
# The policy cameras are the external and the wrist camera; the overview camera is
# review-only (``adp009d_policy_episode``: ``review_only_camera_ids == ["overview"]``), so it
# is recorded but never decides. The composition check renders the external and overview
# cameras only; the wrist camera's task-object pixels come from the camera qualification
# (``camera_role_qualifications.wrist``), which measures pixels but not coverage. At the
# DROID reset pose the wrist camera often does not see the object at all (it looks past it:
# ``adp009d_approach_capture``), and DROID's reset is the start its policies learned from
# (``native_task_camera_start_configuration``), so a wrist view without the object at the
# start is a caveat, never a blocker. The marker is decided on the
# external camera alone.

VISIBILITY_VERDICT_SCHEMA_VERSION = "policy_observation_visibility_verdict.v1"
VISIBLE = "visible"
VISIBLE_WITH_CAVEAT = "visible_with_caveat"
NOT_VISIBLE = "not_visible"
UNMEASURED = "unmeasured"
#: Cameras whose images reach the policy. The overview camera is review-only.
POLICY_CAMERA_ROLES = ("external", "wrist")
MINIMUM_INTERIOR_PIXELS = 64
MINIMUM_VISIBLE_SHARE = 0.5
CLEAN_MAXIMUM_HIDDEN_SHARE = 0.01
#: Targets the gate decides on; a marker is required only when the plan drew one.
TASK_OBJECT = "task_object"
TARGET_MARKER = "task_target_marker"
#: Gate blockers that this verdict replaces once the gate is measured. Every other blocker
#: (a capture failure, a missing camera, a destination support not seen) stays a flag.
SUPERSEDED_BLOCKERS = frozenset({
    "droid_wrist_task_pixels_below_threshold",
    "policy_canary_task_semantic_visibility_failed",
    "policy_canary_external_task_visibility_failed",
    "policy_canary_wrist_task_visibility_failed",
    "policy_canary_overview_task_visibility_failed",
})
SUPERSEDED_COMPOSITION_REASONS = frozenset({
    "composition_support_background_occlusion_requires_review",
    "composition_support_interior_insufficient",
})
#: The typed fault a session stops with when the job is not visible; never a policy result.
NOT_VISIBLE_FAULT = "scene_task_not_visible"


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _composition_rows(gate: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    composition = gate.get("asset_composition_gate")
    if not isinstance(composition, Mapping):
        return None
    views = composition.get("views")
    if not isinstance(views, list):
        return None
    rows = []
    for view in views:
        if not isinstance(view, Mapping) or not isinstance(view.get("assessment"), Mapping):
            return None
        rows.append(view)
    return rows


def _measure(view: Mapping[str, Any]) -> dict[str, Any] | None:
    """Pixels of one target in one camera, from its composition assessment; ``None`` if unreadable."""
    assessment = view["assessment"]
    if assessment.get("scope") != "opaque_target_coverage":
        return None
    interior = _count(assessment.get("support_interior_pixels"))
    hidden = _count(assessment.get("interior_pixels_occluded_by_appearance"))
    if interior is None or hidden is None or hidden > interior:
        return None
    share = hidden / interior if interior else 1.0
    return {"interior_pixels": interior, "hidden_pixels": hidden, "hidden_share": round(share, 6),
            "seen": interior >= MINIMUM_INTERIOR_PIXELS and 1.0 - share >= MINIMUM_VISIBLE_SHARE,
            "clean": interior >= MINIMUM_INTERIOR_PIXELS and share <= CLEAN_MAXIMUM_HIDDEN_SHARE,
            "source": "composition_semantic_segmentation"}


def _wrist_object(gate: Mapping[str, Any]) -> dict[str, Any] | None:
    qualifications = gate.get("camera_role_qualifications")
    row = qualifications.get("wrist") if isinstance(qualifications, Mapping) else None
    if not isinstance(row, Mapping):
        return None
    pixels, minimum = _count(row.get("pixel_count")), _count(row.get("minimum_pixels"))
    if pixels is None or minimum is None or not isinstance(row.get("passed"), bool):
        return None
    # The qualification counts the object's own pixels; it does not measure coverage, so "clean"
    # is reported as not measured (None), never as true.
    return {"pixel_count": pixels, "minimum_pixels": minimum, "seen": row["passed"] is True,
            "clean": None, "coverage_measured": False, "source": "camera_qualification_semantic_segmentation"}


def visibility_verdict(gate: Any) -> dict[str, Any]:
    """The three-way verdict of a runtime gate's measurements (``unmeasured`` when it has none)."""
    unmeasured = {"schema_version": VISIBILITY_VERDICT_SCHEMA_VERSION, "outcome": UNMEASURED, "flags": [], "targets": {}}
    if not isinstance(gate, Mapping):
        return unmeasured
    rows = _composition_rows(gate)
    if rows is None:
        return unmeasured
    composition = gate["asset_composition_gate"]
    classes = composition.get("target_semantic_classes")
    if not isinstance(classes, list) or TASK_OBJECT not in classes:
        return unmeasured
    required = [TASK_OBJECT] + ([TARGET_MARKER] if TARGET_MARKER in classes else [])
    measured: dict[str, dict[str, Any]] = {target: {} for target in required}
    review: dict[str, dict[str, Any]] = {target: {} for target in required}
    for view in rows:
        target, role = view.get("target_semantic_class"), view.get("camera_role")
        if target not in measured:
            continue
        reading = _measure(view)
        if reading is None:
            return unmeasured
        (measured if role in POLICY_CAMERA_ROLES else review)[target][str(role)] = reading
    wrist = _wrist_object(gate)
    if wrist is None or "external" not in measured[TASK_OBJECT]:
        return unmeasured
    measured[TASK_OBJECT]["wrist"] = wrist
    if TARGET_MARKER in measured and "external" not in measured[TARGET_MARKER]:
        return unmeasured
    # Never a stop: the tries run on every outcome, with flags that name the camera and what is
    # hidden. A target none of the CHECKED policy cameras sees carries
    # ``<target>_not_in_view_of_checked_cameras``: the gate measures the external camera and the wrist
    # qualification only, so a policy that also reads ``exterior_right`` may still see it there.
    flags: set[str] = set()
    unseen: list[str] = []
    for target in required:
        cameras = measured[target]
        if not any(reading["seen"] for reading in cameras.values()):
            unseen.append(target)
            flags.add(f"{target}_not_in_view_of_checked_cameras")
        for role, reading in sorted(cameras.items()):
            if not reading["seen"]:
                flags.add(f"{target}_not_in_view:{role}")
            elif reading["clean"] is False:
                flags.add(f"{target}_partly_hidden:{role}")
        for role, reading in sorted(review[target].items()):
            if not reading["seen"]:
                flags.add(f"{target}_not_in_view:{role}")
            elif reading["clean"] is False:
                flags.add(f"{target}_partly_hidden:{role}")
    outcome = NOT_VISIBLE if unseen else VISIBLE_WITH_CAVEAT if flags else VISIBLE
    return {"schema_version": VISIBILITY_VERDICT_SCHEMA_VERSION, "outcome": outcome,
            "flags": sorted(flags), "faults": [], "not_seen_by_the_checked_cameras": sorted(unseen),
            "checked_cameras": sorted({role for target in required for role in measured[target]}
                                      | {role for target in required for role in review[target]}),
            "stops_the_tries": False,
            "targets": {target: {"policy_cameras": measured[target], "review_cameras": review[target]}
                        for target in required},
            "thresholds": {"minimum_interior_pixels": MINIMUM_INTERIOR_PIXELS,
                           "minimum_visible_share": MINIMUM_VISIBLE_SHARE,
                           "clean_maximum_hidden_share": CLEAN_MAXIMUM_HIDDEN_SHARE},
            "policy_camera_roles": list(POLICY_CAMERA_ROLES)}


def _superseded(code: str) -> bool:
    if code in SUPERSEDED_BLOCKERS:
        return True
    parts = code.split(":")
    return len(parts) == 3 and parts[1] in (TASK_OBJECT, TARGET_MARKER) and parts[2] in SUPERSEDED_COMPOSITION_REASONS


def gate_flag_codes(gate: Mapping[str, Any], blocker_codes: list[str]) -> list[str]:
    """The flags a beta try carries past a blocked gate (never a stop, whatever the cameras see).

    ``blocker_codes`` are the gate's blockers as typed codes (``observation_quality_flag_codes``).
    A measured gate replaces its visibility blockers with the verdict's own flags and keeps
    every other blocker; an unmeasured gate keeps its blockers unchanged.
    """
    verdict = visibility_verdict(gate)
    if verdict["outcome"] == UNMEASURED:
        return list(blocker_codes)
    replaced = sorted({code for code in blocker_codes if not _superseded(code)} | set(verdict["flags"]))
    # A clean verdict on a gate blocked only by a few covered pixels has nothing new to say:
    # its own blockers stay the flags, so a blocked gate always carries at least one.
    return replaced or list(blocker_codes)


__all__ = [
    "ALLOWED_TERMINAL_CLASSES",
    "CLEAN_MAXIMUM_HIDDEN_SHARE",
    "MINIMUM_INTERIOR_PIXELS",
    "MINIMUM_VISIBLE_SHARE",
    "NOT_VISIBLE",
    "NOT_VISIBLE_FAULT",
    "POLICY_CAMERA_ROLES",
    "UNMEASURED",
    "VISIBILITY_VERDICT_SCHEMA_VERSION",
    "VISIBLE",
    "VISIBLE_WITH_CAVEAT",
    "gate_flag_codes",
    "visibility_verdict",
    "LIFECYCLE_SCHEMA_VERSION",
    "OBSERVATION_QUALITY_FLAG",
    "OBSERVATION_QUALITY_FLAG_UNTYPED",
    "PolicyEpisodeLifecycleError",
    "READINESS_SCHEMA_VERSION",
    "REQUIRED_READINESS_CHECKS",
    "TERMINAL_PLANNED_DURATION",
    "TERMINAL_POLICY_SAFETY",
    "TERMINAL_SCIENTIFIC",
    "build_lifecycle",
    "observation_quality_flag_codes",
    "observation_quality_flags_valid",
    "seal_prestart_readiness",
    "validate_policy_episode_lifecycle",
]
