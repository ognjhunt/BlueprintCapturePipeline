"""Bounded scientific construction recovery on one warm native worker.

Allocator ``retry_cap=0`` means one provider allocation.  It does not require
throwing that allocation away after the first scientifically rejected
construction.  This module governs a bounded sequence of *different*,
precomputed construction candidates on the already-owned worker:

* native telemetry is reduced to measured collision/contact/task-pose/camera
  feedback without regrading it;
* a deterministic producer emits the next digest-bound candidate inventory;
* the OpenAI Agents SDK may rank that inventory, but returns only an exact
  candidate id and its existing digests;
* repeated candidates, changed candidate identities, new provider allocations,
  and spend/round/TTL overruns fail closed; and
* only a native construction pass invokes the controls continuation callback.

The controller is embodiment- and site-agnostic.  Candidate payloads carry an
exact base pose, reset, entry trajectory, and camera configuration.  Those
payloads are opaque to the model and are passed byte-for-byte to the native
executor; the deterministic inventory producer remains responsible for their
geometry/IK/camera admissibility.  Native gates and controls gates remain the
only qualification authorities.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_supervisor.agents_sdk import (
    AgentsSDKAgentSpec,
    AgentsSDKInvoker,
)


AUTHORITY_SCHEMA_VERSION = (
    "task_evaluation_native_construction_feedback_authority.v1"
)
INVENTORY_SCHEMA_VERSION = (
    "task_evaluation_native_construction_candidate_inventory.v1"
)
CANDIDATE_SCHEMA_VERSION = (
    "task_evaluation_native_construction_candidate.v1"
)
EXECUTION_SCHEMA_VERSION = (
    "task_evaluation_native_construction_candidate_execution.v1"
)
FEEDBACK_SCHEMA_VERSION = "task_evaluation_native_construction_feedback.v1"
CONTROLS_CONTINUATION_SCHEMA_VERSION = (
    "task_evaluation_native_controls_continuation.v1"
)
CONTROLLER_RECEIPT_SCHEMA_VERSION = (
    "task_evaluation_native_construction_feedback_controller_receipt.v1"
)

CONTROLLER_MODEL = "gpt-5.6-sol"
CONTROLLER_REASONING_EFFORT = "high"
CONTROLLER_MAX_OUTPUT_TOKENS = 4_000

# Candidate variants may change how a pre-authorized construction is attempted,
# but never its scientific acceptance criteria.  These names are rejected at
# any depth rather than trusting a caller to label a threshold as a variant.
FORBIDDEN_CANDIDATE_KEYS = frozenset(
    {
        "arrival_tolerance_m",
        "arrival_orientation_tolerance_rad",
        "collision_failure_minimum_force_n",
        "contact_failure_minimum_force_n",
        "gate",
        "gates",
        "gate_ids",
        "passed",
        "qualified",
        "score",
        "status",
        "success",
        "threshold",
        "thresholds",
    }
)


class NativeConstructionFeedbackControllerError(ValueError):
    """A feedback, inventory, selection, or warm execution was invalid."""


class NativeConstructionCandidateSelection(BaseModel):
    """The whole authority granted to the model: select one existing member."""

    model_config = ConfigDict(extra="forbid")

    inventory_digest: str = Field(min_length=71, max_length=71)
    candidate_id: str = Field(min_length=1, max_length=160)
    candidate_digest: str = Field(min_length=71, max_length=71)
    addressed_feedback_digest: str | None = Field(default=None, max_length=71)
    rationale: str = Field(min_length=1, max_length=4_000)


CandidateInventoryProducer = Callable[
    [Mapping[str, Any], Sequence[Mapping[str, Any]], int], Mapping[str, Any]
]
CandidateExecutor = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]
ControlsContinuation = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class CandidateGenerator(Protocol):
    """Swappable deterministic/collision-planner candidate source."""

    def generate(
        self,
        *,
        source_native_feedback: Mapping[str, Any] | None,
        prior_history: Sequence[Mapping[str, Any]],
        round_index: int,
        maximum_candidates: int,
    ) -> Mapping[str, Any]: ...


class SearchLedger(Protocol):
    """Persistent ask/tell evidence without candidate-authoring authority."""

    def record_inventory(
        self, *, inventory: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def record_attempt(
        self, *, round_record: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


class CompositeCandidateGenerator:
    """Try configured collision planners, then the deterministic baseline.

    Planner unavailability is recorded rather than hidden.  The fallback is
    still a full digest-bound CPU inventory and native execution remains the
    only grader, so losing an optional planner changes search quality, not a
    gate or claim.
    """

    def __init__(
        self,
        *,
        generators: Sequence[CandidateGenerator],
        deterministic_fallback: CandidateGenerator,
        fallback_on_generator_unavailable: bool = True,
    ) -> None:
        self._generators = tuple(generators)
        self._fallback = deterministic_fallback
        self._fallback_on_generator_unavailable = bool(
            fallback_on_generator_unavailable
        )

    def generate(
        self,
        *,
        source_native_feedback: Mapping[str, Any] | None,
        prior_history: Sequence[Mapping[str, Any]],
        round_index: int,
        maximum_candidates: int,
    ) -> Mapping[str, Any]:
        attempts: list[dict[str, Any]] = []
        selected: Mapping[str, Any] | None = None
        for generator in self._generators:
            identity = type(generator).__name__
            try:
                selected = generator.generate(
                    source_native_feedback=source_native_feedback,
                    prior_history=prior_history,
                    round_index=round_index,
                    maximum_candidates=maximum_candidates,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                if not self._fallback_on_generator_unavailable:
                    raise
                attempts.append(
                    {
                        "generator": identity,
                        "status_code": f"unavailable:{type(exc).__name__}",
                    }
                )
                continue
            attempts.append({"generator": identity, "status_code": "selected"})
            break
        if selected is None:
            selected = self._fallback.generate(
                source_native_feedback=source_native_feedback,
                prior_history=prior_history,
                round_index=round_index,
                maximum_candidates=maximum_candidates,
            )
            attempts.append(
                {
                    "generator": type(self._fallback).__name__,
                    "status_code": "selected_deterministic_baseline",
                }
            )
        inventory = _copy(selected)
        inventory["candidate_generator_chain"] = attempts
        inventory["inventory_digest"] = ""
        inventory["inventory_digest"] = canonical_digest(
            inventory, digest_field="inventory_digest"
        )
        return inventory


def _copy(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise NativeConstructionFeedbackControllerError(
            "native_construction_feedback_value_invalid"
        ) from exc


def _sha256(value: object) -> bool:
    text = str(value or "")
    return bool(
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _finite_vector(value: object, size: int, *, blocker: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != size
    ):
        raise NativeConstructionFeedbackControllerError(blocker)
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise NativeConstructionFeedbackControllerError(blocker) from exc
    if not all(math.isfinite(item) for item in result):
        raise NativeConstructionFeedbackControllerError(blocker)
    return result


def _forbidden_candidate_paths(value: Any, *, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            path = f"{prefix}.{name}" if prefix else name
            if name.lower() in FORBIDDEN_CANDIDATE_KEYS:
                found.append(path)
            found.extend(_forbidden_candidate_paths(child, prefix=path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(
                _forbidden_candidate_paths(child, prefix=f"{prefix}[{index}]")
            )
    return found


def _validate_variant(
    value: object, *, schema_version: str, digest_field: str, blocker: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeConstructionFeedbackControllerError(blocker)
    variant = _copy(value)
    if (
        variant.get("schema_version") != schema_version
        or variant.get(digest_field)
        != canonical_digest(variant, digest_field=digest_field)
    ):
        raise NativeConstructionFeedbackControllerError(blocker)
    return variant


def validate_native_construction_candidate(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one executable member without interpreting its scientific rank."""

    candidate = _copy(value)
    if _forbidden_candidate_paths(candidate):
        raise NativeConstructionFeedbackControllerError(
            "native_construction_candidate_gate_mutation_forbidden"
        )
    pose = candidate.get("robot_base_pose_world")
    if not isinstance(pose, Mapping):
        raise NativeConstructionFeedbackControllerError(
            "native_construction_candidate_pose_invalid"
        )
    position = _finite_vector(
        pose.get("position_world_m"),
        3,
        blocker="native_construction_candidate_pose_invalid",
    )
    orientation = _finite_vector(
        pose.get("orientation_xyzw"),
        4,
        blocker="native_construction_candidate_pose_invalid",
    )
    if not math.isclose(
        math.sqrt(math.fsum(item * item for item in orientation)),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-4,
    ):
        raise NativeConstructionFeedbackControllerError(
            "native_construction_candidate_pose_invalid"
        )
    candidate["robot_base_pose_world"] = {
        "position_world_m": position,
        "orientation_xyzw": orientation,
    }
    reset = _validate_variant(
        candidate.get("reset_variant"),
        schema_version="task_evaluation_native_robot_reset_variant.v1",
        digest_field="reset_variant_digest",
        blocker="native_construction_candidate_reset_invalid",
    )
    joints = reset.get("robot_joint_reset_positions_rad")
    if not isinstance(joints, Mapping) or not joints:
        raise NativeConstructionFeedbackControllerError(
            "native_construction_candidate_reset_invalid"
        )
    try:
        joint_values = {str(name): float(value) for name, value in joints.items()}
    except (TypeError, ValueError) as exc:
        raise NativeConstructionFeedbackControllerError(
            "native_construction_candidate_reset_invalid"
        ) from exc
    if not all(name and math.isfinite(value) for name, value in joint_values.items()):
        raise NativeConstructionFeedbackControllerError(
            "native_construction_candidate_reset_invalid"
        )
    entry = _validate_variant(
        candidate.get("entry_trajectory_variant"),
        schema_version="task_evaluation_native_entry_trajectory_variant.v1",
        digest_field="entry_trajectory_variant_digest",
        blocker="native_construction_candidate_entry_trajectory_invalid",
    )
    if not isinstance(entry.get("waypoints"), list) or not entry["waypoints"]:
        raise NativeConstructionFeedbackControllerError(
            "native_construction_candidate_entry_trajectory_invalid"
        )
    interaction_value = candidate.get("interaction_trajectory_variant")
    if interaction_value is not None:
        interaction = _validate_variant(
            interaction_value,
            schema_version=(
                "task_evaluation_native_interaction_trajectory_variant.v1"
            ),
            digest_field="interaction_trajectory_variant_digest",
            blocker="native_construction_candidate_interaction_trajectory_invalid",
        )
        interaction_waypoints = interaction.get("waypoints")
        if (
            not str(interaction.get("interaction_branch_id") or "")
            or not isinstance(interaction.get("solver_seed"), int)
            or not (
                _sha256(
                    interaction.get("source_native_phase_contract_digest")
                )
                or _sha256(
                    interaction.get("source_normalized_trajectory_digest")
                )
            )
            or interaction.get("preserves_authored_tcp_endpoints") is not True
            or not isinstance(interaction_waypoints, list)
            or not interaction_waypoints
            or any(not isinstance(row, Mapping) for row in interaction_waypoints)
        ):
            raise NativeConstructionFeedbackControllerError(
                "native_construction_candidate_interaction_trajectory_invalid"
            )
    camera = _validate_variant(
        candidate.get("camera_variant"),
        schema_version="task_evaluation_native_camera_variant.v1",
        digest_field="camera_variant_digest",
        blocker="native_construction_candidate_camera_invalid",
    )
    cameras = camera.get("cameras")
    if (
        not isinstance(cameras, list)
        or not cameras
        or any(not isinstance(row, Mapping) for row in cameras)
        or len({str(row.get("role") or "") for row in cameras}) != len(cameras)
        or any(not str(row.get("role") or "") for row in cameras)
    ):
        raise NativeConstructionFeedbackControllerError(
            "native_construction_candidate_camera_invalid"
        )
    if (
        candidate.get("schema_version") != CANDIDATE_SCHEMA_VERSION
        or not str(candidate.get("candidate_id") or "")
        or not str(candidate.get("support_surface_id") or "")
        or not isinstance(candidate.get("deterministic_rank"), int)
        or int(candidate["deterministic_rank"]) < 0
        or candidate.get("candidate_digest")
        != canonical_digest(candidate, digest_field="candidate_digest")
    ):
        raise NativeConstructionFeedbackControllerError(
            "native_construction_candidate_invalid"
        )
    try:
        maximum_cost = float(candidate.get("maximum_incremental_cost_usd"))
        maximum_runtime = float(candidate.get("maximum_runtime_seconds"))
    except (TypeError, ValueError) as exc:
        raise NativeConstructionFeedbackControllerError(
            "native_construction_candidate_budget_invalid"
        ) from exc
    if (
        not math.isfinite(maximum_cost)
        or maximum_cost <= 0.0
        or not math.isfinite(maximum_runtime)
        or maximum_runtime <= 0.0
    ):
        raise NativeConstructionFeedbackControllerError(
            "native_construction_candidate_budget_invalid"
        )
    return candidate


def validate_native_construction_inventory(
    value: Mapping[str, Any],
    *,
    expected_run_id: str,
    expected_round_index: int,
    expected_feedback_digest: str | None,
    maximum_candidates: int,
) -> dict[str, Any]:
    """Validate one deterministic candidate inventory and its feedback lineage."""

    inventory = _copy(value)
    raw_candidates = inventory.get("candidates")
    if (
        inventory.get("schema_version") != INVENTORY_SCHEMA_VERSION
        or inventory.get("run_id") != expected_run_id
        or inventory.get("round_index") != int(expected_round_index)
        or inventory.get("source_native_feedback_digest")
        != expected_feedback_digest
        or inventory.get("model_authored_candidates") is not False
        or not isinstance(raw_candidates, list)
        or not 1 <= len(raw_candidates) <= int(maximum_candidates)
        or inventory.get("inventory_digest")
        != canonical_digest(inventory, digest_field="inventory_digest")
    ):
        raise NativeConstructionFeedbackControllerError(
            "native_construction_candidate_inventory_invalid"
        )
    candidates = [
        validate_native_construction_candidate(row)
        for row in raw_candidates
        if isinstance(row, Mapping)
    ]
    if len(candidates) != len(raw_candidates):
        raise NativeConstructionFeedbackControllerError(
            "native_construction_candidate_inventory_invalid"
        )
    ids = [str(row["candidate_id"]) for row in candidates]
    digests = [str(row["candidate_digest"]) for row in candidates]
    if len(set(ids)) != len(ids) or len(set(digests)) != len(digests):
        raise NativeConstructionFeedbackControllerError(
            "native_construction_candidate_inventory_duplicate"
        )
    inventory["candidates"] = candidates
    return inventory


def _sample(row: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = row.get("task_sample")
    return _normalized_sample(raw)


def _normalized_sample(raw: object) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    native = raw.get("native_readback")
    if not isinstance(native, Mapping):
        return raw
    # Rigid readback keeps force/pose values at the sample root while some
    # runtimes additionally attach link-level sensor identities beneath
    # ``native_readback``.  Merge those namespaces so the controller never
    # drops either the measured force or the link that produced it.
    return {**dict(raw), **dict(native)}


def _phase_samples(row: Mapping[str, Any]) -> list[tuple[int, str, Mapping[str, Any]]]:
    sampled = row.get("task_samples")
    sampled = sampled if isinstance(sampled, list) else []
    result = [
        (index, "step", _normalized_sample(sample))
        for index, sample in enumerate(sampled)
        if isinstance(sample, Mapping)
    ]
    terminal = _normalized_sample(row.get("task_sample"))
    if terminal:
        result.append((len(sampled), "terminal", terminal))
    return result


def _pose_from_sample(sample: Mapping[str, Any]) -> list[float] | None:
    for key in (
        "task_scoring_pose_world",
        "task_object_pose_world",
        "task_root_pose_world",
        "asset_root_pose_world",
    ):
        value = sample.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            try:
                pose = [float(item) for item in value]
            except (TypeError, ValueError):
                continue
            if len(pose) >= 3 and all(math.isfinite(item) for item in pose):
                return pose
    return None


def _first_link_hint(sample: Mapping[str, Any], channel: str) -> str | None:
    """Return a measured sensor/body/link identity when the worker retained one."""

    queue: list[Any] = [sample]
    while queue:
        value = queue.pop(0)
        if isinstance(value, Mapping):
            identity = " ".join(
                str(value.get(key) or "")
                for key in ("logical_sensor_id", "sensor_id", "channel")
            )
            if channel in identity:
                for key in (
                    "body_path",
                    "body_prim_path",
                    "link_path",
                    "prim_path",
                    "sensor_instance_id",
                ):
                    if value.get(key):
                        return str(value[key])
            queue.extend(value.values())
        elif isinstance(value, list):
            queue.extend(value)
    return None


def _physics_objective_measurements(
    native: Mapping[str, Any], phase_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    plan = native.get("construction_phase_plan")
    if not isinstance(plan, Mapping) or plan.get("task_kind") != "rigid_pick_place":
        return None
    thresholds = plan.get("thresholds")
    if not isinstance(thresholds, Mapping):
        return None
    try:
        contact_threshold = float(thresholds["task_contact_minimum_force_n"])
    except (KeyError, TypeError, ValueError):
        return None
    by_id = {
        str(row.get("phase_id") or ""): row
        for row in phase_rows
        if isinstance(row, Mapping)
    }
    collision_samples = [
        (phase_id, index, kind, sample)
        for phase_id, row in by_id.items()
        for index, kind, sample in _phase_samples(row)
    ]
    robot_scene = []
    for phase_id, index, kind, sample in collision_samples:
        try:
            force = float(sample.get("robot_scene_contact_peak_force_n"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(force):
            robot_scene.append((phase_id, index, kind, force))
    positive = [row for row in robot_scene if row[3] > 0.0]
    first_force = positive[0][3] if positive else 0.0
    peak_force = max((row[3] for row in robot_scene), default=0.0)
    contact_ids = [
        phase_id
        for phase_id in by_id
        if phase_id == "push_contact"
        or (
            phase_id.startswith("push_")
            and phase_id not in {"push_detach", "push_release"}
        )
    ]
    contact_samples = [
        sample for phase_id in contact_ids for _index, _kind, sample in _phase_samples(by_id[phase_id])
    ]
    covered = sum(
        1
        for sample in contact_samples
        if float(sample.get("task_robot_contact_peak_force_n") or 0.0)
        >= contact_threshold
    )
    phase_plan = {
        str(row.get("phase_id") or ""): row
        for row in plan.get("phases") or []
        if isinstance(row, Mapping)
    }
    push_errors = []
    for phase_id in contact_ids:
        if phase_id == "push_contact":
            continue
        expected = (phase_plan.get(phase_id) or {}).get(
            "expected_scoring_position_world_m"
        )
        terminal = _pose_from_sample(_sample(by_id[phase_id]))
        if isinstance(expected, list) and terminal is not None:
            push_errors.append(math.dist(expected, terminal[:3]))
    destination = plan.get("destination_position_world_m")
    settle = by_id.get("settle_observe")
    final_pose = _pose_from_sample(_sample(settle)) if settle is not None else None
    destination_error = (
        math.dist(destination, final_pose[:3])
        if isinstance(destination, list) and final_pose is not None
        else None
    )
    result: dict[str, Any] = {
        "schema_version": (
            "task_evaluation_native_construction_physics_objective_measurements.v1"
        ),
        "forbidden_robot_scene_collision_peak_force_n": peak_force,
        "forbidden_robot_scene_collision_first_sample_force_n": first_force,
        "required_task_contact_covered_sample_count": covered,
        "required_task_contact_sample_count": len(contact_samples),
        "required_task_contact_coverage_fraction": (
            covered / len(contact_samples) if contact_samples else 0.0
        ),
        "push_path_tracking_error_m": max(push_errors, default=None),
        "destination_error_m": destination_error,
        "native_thresholds_changed": False,
        "native_verdict_recomputed": False,
        "measurement_only_not_native_grade": True,
        "native_result_digest": native.get("result_digest"),
        "measurement_digest": "",
    }
    result["measurement_digest"] = canonical_digest(
        result, digest_field="measurement_digest"
    )
    return result


def summarize_native_construction_feedback(
    native_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract measured recovery inputs from an immutable native result.

    No threshold is copied and no new verdict is calculated.  ``passed`` is the
    worker's sealed construction verdict; all remaining fields are measurements
    that help the deterministic inventory producer choose the next variants.
    """

    native = _copy(native_result)
    if (
        native.get("schema_version")
        != "native_task_arena_construction_result.v1"
        or native.get("result_digest")
        != canonical_digest(native, digest_field="result_digest")
        or not isinstance(native.get("blockers"), list)
    ):
        raise NativeConstructionFeedbackControllerError(
            "native_construction_feedback_result_invalid"
        )
    initial_raw = native.get("initial_readback")
    initial_sample_raw = (
        initial_raw.get("task_sample")
        if isinstance(initial_raw, Mapping)
        else None
    )
    initial_sample = (
        initial_sample_raw.get("native_readback")
        if isinstance(initial_sample_raw, Mapping)
        and isinstance(initial_sample_raw.get("native_readback"), Mapping)
        else initial_sample_raw
    )
    initial_pose = (
        _pose_from_sample(initial_sample)
        if isinstance(initial_sample, Mapping)
        else None
    )
    phases: list[dict[str, Any]] = []
    first_collision: dict[str, Any] | None = None
    peak_collision: dict[str, Any] | None = None
    first_failed_phase: str | None = None
    collision_channels = (
        "robot_scene_contact",
        "robot_task_forbidden_collision",
        "task_scene_collision",
    )
    for raw in native.get("phase_results") or []:
        if not isinstance(raw, Mapping):
            continue
        sample = _sample(raw)
        pose = _pose_from_sample(sample)
        displacement = (
            math.dist(initial_pose[:3], pose[:3])
            if initial_pose is not None and pose is not None
            else None
        )
        contacts: dict[str, float | None] = {}
        for channel in (
            "task_robot_contact",
            "task_support_contact",
            *collision_channels,
        ):
            raw_force = sample.get(f"{channel}_peak_force_n")
            try:
                force = float(raw_force)
            except (TypeError, ValueError):
                force = None
            contacts[f"{channel}_peak_force_n"] = (
                force if force is not None and math.isfinite(force) else None
            )
        phase_first_collision = None
        phase_peak_collision = None
        for sample_index, sample_kind, measured in _phase_samples(raw):
            for channel in collision_channels:
                try:
                    force = float(measured.get(f"{channel}_peak_force_n"))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(force) or force <= 0.0:
                    continue
                collision = {
                    "phase_id": str(raw.get("phase_id") or ""),
                    "sample_index": sample_index,
                    "sample_kind": sample_kind,
                    "channel": channel,
                    "peak_force_n": force,
                    "link_or_sensor_id": _first_link_hint(measured, channel),
                    "measurement_only_not_regraded": True,
                }
                if first_collision is None:
                    first_collision = collision
                if phase_first_collision is None:
                    phase_first_collision = collision
                if peak_collision is None or force > peak_collision["peak_force_n"]:
                    peak_collision = collision
                if (
                    phase_peak_collision is None
                    or force > phase_peak_collision["peak_force_n"]
                ):
                    phase_peak_collision = collision
        target_reached = raw.get("target_reached") is True
        if first_failed_phase is None and not target_reached:
            first_failed_phase = str(raw.get("phase_id") or "")
        phases.append(
            {
                "phase_id": str(raw.get("phase_id") or ""),
                "steps": raw.get("steps"),
                "target_reached": target_reached,
                "terminal_position_error_m": raw.get(
                    "terminal_position_error_m"
                ),
                "terminal_orientation_error_rad": raw.get(
                    "terminal_orientation_error_rad"
                ),
                "task_pose_world": pose,
                "task_displacement_from_reset_m": displacement,
                "contacts": contacts,
                "first_collision_sample": phase_first_collision,
                "peak_collision_sample": phase_peak_collision,
            }
        )
    cameras: dict[str, Any] = {}
    for role, raw in (native.get("camera_gates") or {}).items():
        if not isinstance(raw, Mapping):
            continue
        best = raw.get("best_observability")
        best = best if isinstance(best, Mapping) else {}
        render = best.get("render_evidence")
        render = render if isinstance(render, Mapping) else {}
        cameras[str(role)] = {
            "passed": raw.get("passed") is True,
            "best_snapshot_id": raw.get("best_snapshot_id"),
            "pixel_count": best.get("pixel_count"),
            "pixel_fraction": best.get("pixel_fraction"),
            "centroid_xy_fraction": best.get("centroid_xy_fraction"),
            "site_appearance_claimed": raw.get("site_appearance_claimed"),
            "site_rendered": render.get("site_rendered"),
            "dominant_rgb_pixel_fraction": render.get(
                "dominant_rgb_pixel_fraction"
            ),
            "blockers": list(best.get("blockers") or []),
        }
    gate_failure_codes = sorted(
        {
            "gate_failed:" + blocker.split(":", 1)[1]
            for blocker in native["blockers"]
            if isinstance(blocker, str)
            and blocker.startswith(
                (
                    "native_rigid_construction_gate_failed:",
                    "native_articulated_construction_gate_failed:",
                )
            )
        }
    )
    feedback: dict[str, Any] = {
        "schema_version": FEEDBACK_SCHEMA_VERSION,
        "native_result_digest": native["result_digest"],
        "passed": bool(
            native.get("status") == "completed"
            and native.get("construction_gate_qualified") is True
            and not native.get("blockers")
        ),
        "native_blockers": list(native["blockers"]),
        "initial_robot_root_pose_world": (
            initial_raw.get("robot_root_pose_world")
            if isinstance(initial_raw, Mapping)
            else None
        ),
        "first_failed_phase": first_failed_phase,
        "first_collision": first_collision,
        "peak_collision": peak_collision,
        "feedback_codes": gate_failure_codes,
        "phase_measurements": phases,
        "camera_measurements": cameras,
        "physics_objective_measurements": _physics_objective_measurements(
            native, native.get("phase_results") or []
        ),
        "claim_boundary": (
            "measured_native_feedback_only;does_not_change_or_recompute_any_"
            "construction_gate"
        ),
        "feedback_digest": "",
    }
    feedback["feedback_digest"] = canonical_digest(
        feedback, digest_field="feedback_digest"
    )
    return feedback


def bind_warm_native_construction_execution(
    *,
    candidate: Mapping[str, Any],
    inventory_digest: str,
    allocator_result: Mapping[str, Any],
    native_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the existing warm-executor output to one exact controller member.

    This is the production adapter between ``WarmNativePlacementExecutor`` (or
    another canonical warm Arena transport) and the controller's executor
    callback.  It copies no paths and accepts no candidate fields from the
    provider result, so the provider cannot rewrite the selected variant.
    """

    selected = validate_native_construction_candidate(candidate)
    if not _sha256(inventory_digest):
        raise NativeConstructionFeedbackControllerError(
            "native_construction_candidate_inventory_invalid"
        )
    allocator = _copy(allocator_result)
    feedback = summarize_native_construction_feedback(native_result)
    try:
        cost = float(allocator.get("incremental_cost_upper_bound_usd"))
    except (TypeError, ValueError) as exc:
        raise NativeConstructionFeedbackControllerError(
            "native_construction_warm_allocator_result_invalid"
        ) from exc
    if (
        allocator.get("provider_allocations_performed") != 0
        or not isinstance(allocator.get("provider_instance_id"), int)
        or not math.isfinite(cost)
        or cost < 0.0
    ):
        raise NativeConstructionFeedbackControllerError(
            "native_construction_warm_allocator_result_invalid"
        )
    result: dict[str, Any] = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "status": "passed" if feedback["passed"] else "rejected",
        "candidate_id": selected["candidate_id"],
        "candidate_digest": selected["candidate_digest"],
        "inventory_digest": inventory_digest,
        "provider_instance_id": allocator["provider_instance_id"],
        "provider_allocations_performed": 0,
        "runtime_seconds": allocator.get("runtime_seconds"),
        "incremental_cost_upper_bound_usd": cost,
        "native_result": _copy(native_result),
        "execution_result_digest": "",
    }
    result["execution_result_digest"] = canonical_digest(
        result, digest_field="execution_result_digest"
    )
    return result


def native_construction_feedback_codes(
    feedback: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    """Return deterministic, threshold-free labels for inventory ranking."""

    if feedback is None:
        return ()
    value = _copy(feedback)
    if (
        value.get("schema_version") != FEEDBACK_SCHEMA_VERSION
        or value.get("feedback_digest")
        != canonical_digest(value, digest_field="feedback_digest")
    ):
        raise NativeConstructionFeedbackControllerError(
            "native_construction_feedback_invalid"
        )
    codes: set[str] = {
        str(code) for code in value.get("feedback_codes") or [] if str(code)
    }
    for blocker in value.get("native_blockers") or []:
        text = str(blocker)
        for prefix in (
            "native_rigid_construction_gate_failed:",
            "native_articulated_construction_gate_failed:",
        ):
            if text.startswith(prefix) and len(text) > len(prefix):
                codes.add("gate_failed:" + text[len(prefix) :])
    failed = str(value.get("first_failed_phase") or "")
    if failed:
        codes.add(f"phase_unreached:{failed}")
    collision = value.get("first_collision")
    if isinstance(collision, Mapping):
        phase_id = str(collision.get("phase_id") or "")
        channel = str(collision.get("channel") or "")
        if phase_id and channel:
            codes.add(f"collision:{phase_id}:{channel}")
    for phase in value.get("phase_measurements") or []:
        if not isinstance(phase, Mapping):
            continue
        phase_id = str(phase.get("phase_id") or "")
        for field, force in (phase.get("contacts") or {}).items():
            try:
                observed = float(force) > 0.0
            except (TypeError, ValueError):
                observed = False
            if observed and phase_id:
                codes.add(f"contact_observed:{phase_id}:{field}")
    for role, camera in (value.get("camera_measurements") or {}).items():
        if not isinstance(camera, Mapping) or camera.get("passed") is True:
            continue
        codes.add(f"camera_failed:{role}")
        if camera.get("site_rendered") is False:
            codes.add(f"site_not_rendered:{role}")
        for blocker in camera.get("blockers") or []:
            codes.add(f"camera_blocker:{role}:{blocker}")
    return tuple(sorted(codes))


def build_next_native_construction_inventory(
    *,
    run_id: str,
    round_index: int,
    source_native_feedback: Mapping[str, Any] | None,
    prior_history: Sequence[Mapping[str, Any]],
    candidate_universe: Sequence[Mapping[str, Any]],
    maximum_candidates: int,
) -> dict[str, Any]:
    """Rank an immutable candidate universe from measured feedback.

    Variants are authored before this function (normally by deterministic
    geometry/IK/swept-volume and camera compilers).  This function never edits
    them: it excludes attempted digests and orders the remaining exact members
    by declared feedback coverage, deterministic rank, then candidate id.
    """

    if not run_id or round_index < 0 or not 1 <= int(maximum_candidates) <= 64:
        raise NativeConstructionFeedbackControllerError(
            "native_construction_candidate_inventory_invalid"
        )
    feedback_digest = (
        str(source_native_feedback.get("feedback_digest"))
        if source_native_feedback is not None
        else None
    )
    codes = set(native_construction_feedback_codes(source_native_feedback))
    attempted = {
        str((row.get("candidate") or {}).get("candidate_digest") or "")
        for row in prior_history
        if isinstance(row, Mapping) and isinstance(row.get("candidate"), Mapping)
    }
    candidates = [
        validate_native_construction_candidate(row) for row in candidate_universe
    ]
    remaining = [
        row for row in candidates if str(row["candidate_digest"]) not in attempted
    ]
    remaining.sort(
        key=lambda row: (
            -len(codes.intersection(set(row.get("addressed_feedback_codes") or []))),
            int(row["deterministic_rank"]),
            str(row["candidate_id"]),
        )
    )
    selected = remaining[: int(maximum_candidates)]
    if not selected:
        raise NativeConstructionFeedbackControllerError(
            "native_construction_candidate_inventory_exhausted"
        )
    inventory: dict[str, Any] = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "run_id": run_id,
        "round_index": int(round_index),
        "source_native_feedback_digest": feedback_digest,
        "source_native_feedback_codes": sorted(codes),
        "model_authored_candidates": False,
        "candidates": selected,
        "inventory_digest": "",
    }
    inventory["inventory_digest"] = canonical_digest(
        inventory, digest_field="inventory_digest"
    )
    return inventory


def construction_phase_plan_for_candidate(
    *,
    scene_plan: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Prepend one candidate's CPU-checked entry to the authored task plan.

    The authored phases, gate ids, thresholds, and task criteria are copied
    unchanged.  Entry phases carry no success gate; they merely move from the
    candidate reset to the original first phase and remain subject to native
    IK/collision readback.  The shared total action budget is adjusted, never
    widened beyond the scene plan's existing maximum.
    """

    from .native_task_construction_plan import (
        materialize_native_task_construction_phase_plan,
        native_task_construction_authored_contract_digest,
    )

    selected = validate_native_construction_candidate(candidate)
    plan = materialize_native_task_construction_phase_plan(scene_plan)
    authored_contract_digest = native_task_construction_authored_contract_digest(
        plan
    )
    variant = selected["entry_trajectory_variant"]
    entry_phases: list[dict[str, Any]] = []
    seen: set[str] = set()
    raw_waypoints = list(variant["waypoints"])
    solver_path = any(
        isinstance(row, Mapping)
        and isinstance(row.get("robot_joint_positions_rad"), Mapping)
        for row in raw_waypoints
    )
    rows_to_materialize: list[Mapping[str, Any]] = []
    if solver_path:
        for stage_kind in ("entry", "approach"):
            stage_rows = [
                row
                for row in raw_waypoints
                if isinstance(row, Mapping)
                and row.get("stage_kind") == stage_kind
            ]
            if not stage_rows:
                raise NativeConstructionFeedbackControllerError(
                    "native_construction_candidate_entry_trajectory_invalid"
                )
            terminal = stage_rows[-1]
            rows_to_materialize.append(
                {
                    "waypoint_id": f"curobo-{stage_kind}",
                    "position_world_m": terminal.get("target_position_world_m"),
                    "orientation_world_xyzw": terminal.get(
                        "target_orientation_world_xyzw"
                    ),
                    "solver_stage_kind": stage_kind,
                    "solver_joint_waypoint_sequence_rad": [
                        dict(row["robot_joint_positions_rad"])
                        for row in stage_rows
                    ],
                }
            )
    else:
        rows_to_materialize = [
            row for row in raw_waypoints if isinstance(row, Mapping)
        ]
        if len(rows_to_materialize) != len(raw_waypoints):
            raise NativeConstructionFeedbackControllerError(
                "native_construction_candidate_entry_trajectory_invalid"
            )
    for index, raw in enumerate(rows_to_materialize):
        if not isinstance(raw, Mapping):
            raise NativeConstructionFeedbackControllerError(
                "native_construction_candidate_entry_trajectory_invalid"
            )
        phase_id = f"feedback_entry_{index:02d}_{str(raw.get('waypoint_id') or index)}"
        if phase_id in seen or any(
            row.get("phase_id") == phase_id for row in plan["phases"]
        ):
            raise NativeConstructionFeedbackControllerError(
                "native_construction_candidate_entry_trajectory_invalid"
            )
        seen.add(phase_id)
        position = _finite_vector(
            raw.get("position_world_m"),
            3,
            blocker="native_construction_candidate_entry_trajectory_invalid",
        )
        orientation = _finite_vector(
            raw.get("orientation_world_xyzw"),
            4,
            blocker="native_construction_candidate_entry_trajectory_invalid",
        )
        phase = {
                "phase_id": phase_id,
                "position_world_m": position,
                "orientation_world_xyzw": orientation,
                "gripper_state": "open",
                "gate_ids": [],
                "feedback_entry_only": True,
            }
        if solver_path:
            sequence = raw.get("solver_joint_waypoint_sequence_rad")
            if not isinstance(sequence, list) or not sequence:
                raise NativeConstructionFeedbackControllerError(
                    "native_construction_candidate_entry_trajectory_invalid"
                )
            try:
                normalized_sequence = [
                    {str(name): float(value) for name, value in row.items()}
                    for row in sequence
                    if isinstance(row, Mapping)
                ]
            except (TypeError, ValueError) as exc:
                raise NativeConstructionFeedbackControllerError(
                    "native_construction_candidate_entry_trajectory_invalid"
                ) from exc
            if len(normalized_sequence) != len(sequence) or any(
                not row or not all(math.isfinite(value) for value in row.values())
                for row in normalized_sequence
            ):
                raise NativeConstructionFeedbackControllerError(
                    "native_construction_candidate_entry_trajectory_invalid"
                )
            phase["solver_stage_kind"] = raw["solver_stage_kind"]
            phase["solver_joint_waypoint_sequence_rad"] = normalized_sequence
            phase["solver_path_execution_required"] = True
        entry_phases.append(phase)
    if solver_path:
        interaction_variant = selected.get("interaction_trajectory_variant")
        task_solver_source = (
            interaction_variant.get("waypoints")
            if isinstance(interaction_variant, Mapping)
            else raw_waypoints
        )
        task_solver_rows = [
            row
            for row in task_solver_source
            if isinstance(row, Mapping)
            and row.get("stage_kind") in {"contact", "release", "retreat"}
        ]
        assigned: set[int] = set()
        for authored_phase in plan["phases"]:
            matches = [
                (index, row)
                for index, row in enumerate(task_solver_rows)
                if row.get("source_native_phase_id")
                == authored_phase.get("phase_id")
            ]
            if not matches:
                continue
            terminal = matches[-1][1]
            if (
                terminal.get("target_position_world_m")
                != authored_phase.get("position_world_m")
                or terminal.get("target_orientation_world_xyzw")
                != authored_phase.get("orientation_world_xyzw")
            ):
                raise NativeConstructionFeedbackControllerError(
                    "native_construction_candidate_authored_tcp_endpoint_mismatch"
                )
            try:
                sequence = [
                    {
                        str(name): float(value)
                        for name, value in row[
                            "robot_joint_positions_rad"
                        ].items()
                    }
                    for _index, row in matches
                ]
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise NativeConstructionFeedbackControllerError(
                    "native_construction_candidate_entry_trajectory_invalid"
                ) from exc
            if any(
                not values
                or not all(math.isfinite(value) for value in values.values())
                for values in sequence
            ):
                raise NativeConstructionFeedbackControllerError(
                    "native_construction_candidate_entry_trajectory_invalid"
                )
            assigned.update(index for index, _row in matches)
            authored_phase["solver_stage_kind"] = matches[0][1]["stage_kind"]
            authored_phase["solver_joint_waypoint_sequence_rad"] = sequence
            authored_phase["solver_path_execution_required"] = True
        if len(assigned) != len(task_solver_rows):
            raise NativeConstructionFeedbackControllerError(
                "native_construction_candidate_task_solver_path_unbound"
            )
    execution = plan.get("execution_parameters")
    cadence = scene_plan.get("cadence")
    if not isinstance(execution, Mapping) or not isinstance(cadence, Mapping):
        raise NativeConstructionFeedbackControllerError(
            "native_construction_candidate_entry_trajectory_invalid"
        )
    stable_samples = int(execution.get("stable_samples") or 0)
    added_steps = len(entry_phases) * stable_samples
    if solver_path:
        added_steps += sum(
            len(row["solver_joint_waypoint_sequence_rad"])
            for row in entry_phases
        )
    existing_budget = int(execution.get("maximum_construction_total_steps") or 0)
    action_cap = int(cadence.get("maximum_action_steps") or 0)
    if added_steps <= 0 or existing_budget + added_steps > action_cap:
        raise NativeConstructionFeedbackControllerError(
            "native_construction_candidate_entry_budget_invalid"
        )
    plan["phases"] = [*entry_phases, *plan["phases"]]
    plan["phase_count"] = len(plan["phases"])
    plan["execution_parameters"] = {
        **dict(execution),
        "maximum_construction_total_steps": existing_budget + added_steps,
    }
    plan["construction_feedback_candidate_digest"] = selected["candidate_digest"]
    plan["entry_trajectory_variant_digest"] = variant[
        "entry_trajectory_variant_digest"
    ]
    interaction_variant = selected.get("interaction_trajectory_variant")
    if isinstance(interaction_variant, Mapping) and solver_path:
        if (
            interaction_variant.get("source_native_phase_contract_digest")
            != authored_contract_digest
        ):
            raise NativeConstructionFeedbackControllerError(
                "native_construction_candidate_phase_plan_digest_mismatch"
            )
        plan["interaction_trajectory_variant_digest"] = interaction_variant[
            "interaction_trajectory_variant_digest"
        ]
    plan["authored_gate_contract_unchanged"] = True
    plan["plan_digest"] = ""
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def _validate_authority(value: Mapping[str, Any], *, now: float) -> dict[str, Any]:
    authority = _copy(value)
    try:
        maximum_cost = float(authority.get("maximum_incremental_cost_usd"))
        deadline = float(authority.get("deadline_unix_s"))
        maximum_rounds = int(authority.get("maximum_rounds"))
        maximum_candidates = int(authority.get("maximum_candidates_per_round"))
    except (TypeError, ValueError) as exc:
        raise NativeConstructionFeedbackControllerError(
            "native_construction_feedback_authority_invalid"
        ) from exc
    if (
        authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION
        or not str(authority.get("run_id") or "")
        or not isinstance(authority.get("expected_provider_instance_id"), int)
        or not _sha256(authority.get("warm_session_digest"))
        or authority.get("allocator_retry_cap") != 0
        or not 1 <= maximum_rounds <= 8
        or not 1 <= maximum_candidates <= 64
        or not math.isfinite(maximum_cost)
        or maximum_cost <= 0.0
        or not math.isfinite(deadline)
        or deadline <= now
        or authority.get("authority_digest")
        != canonical_digest(authority, digest_field="authority_digest")
    ):
        raise NativeConstructionFeedbackControllerError(
            "native_construction_feedback_authority_invalid"
        )
    return authority


def _selection_input(
    *,
    authority: Mapping[str, Any],
    inventory: Mapping[str, Any],
    feedback: Mapping[str, Any] | None,
    attempted_candidate_digests: Sequence[str],
) -> dict[str, Any]:
    candidates = [
        {
            "candidate_id": row["candidate_id"],
            "candidate_digest": row["candidate_digest"],
            "deterministic_rank": row["deterministic_rank"],
            "addressed_feedback_codes": list(
                row.get("addressed_feedback_codes") or []
            ),
            "robot_base_pose_world": row["robot_base_pose_world"],
            "support_surface_id": row["support_surface_id"],
            "reset_variant_digest": row["reset_variant"]["reset_variant_digest"],
            "entry_trajectory_variant_digest": row["entry_trajectory_variant"][
                "entry_trajectory_variant_digest"
            ],
            "interaction_trajectory_variant_digest": (
                row.get("interaction_trajectory_variant") or {}
            ).get("interaction_trajectory_variant_digest"),
            "camera_variant_digest": row["camera_variant"][
                "camera_variant_digest"
            ],
            "maximum_incremental_cost_usd": row[
                "maximum_incremental_cost_usd"
            ],
            "maximum_runtime_seconds": row["maximum_runtime_seconds"],
        }
        for row in inventory["candidates"]
    ]
    return {
        "schema_version": "task_evaluation_native_construction_selection_prompt.v1",
        "run_id": authority["run_id"],
        "round_index": inventory["round_index"],
        "inventory_digest": inventory["inventory_digest"],
        "source_native_feedback": feedback,
        "attempted_candidate_digests": list(attempted_candidate_digests),
        "candidates": candidates,
        "authority_boundary": {
            "select_exact_inventory_member_only": True,
            "candidate_content_not_model_authored": True,
            "model_may_not_mutate_base_reset_entry_interaction_or_cameras": True,
            "model_may_not_change_gates_or_thresholds": True,
            "native_worker_is_sole_construction_grader": True,
        },
    }


def _validated_ledger_receipt(
    value: Mapping[str, Any],
    *,
    event: str,
    run_id: str,
    round_index: int,
    inventory_digest: str,
    candidate_digest: str | None = None,
    execution_result_digest: str | None = None,
    feedback_digest: str | None = None,
) -> dict[str, Any]:
    receipt = _copy(value)
    if (
        not str(receipt.get("schema_version") or "").endswith(".v1")
        or receipt.get("event") != event
        or receipt.get("run_id") != run_id
        or receipt.get("round_index") != round_index
        or receipt.get("inventory_digest") != inventory_digest
        or receipt.get("candidate_digest") != candidate_digest
        or receipt.get("execution_result_digest") != execution_result_digest
        or receipt.get("native_feedback_digest") != feedback_digest
        or receipt.get("candidate_authoring_performed") is not False
        or receipt.get("grading_performed") is not False
        or receipt.get("ledger_receipt_digest")
        != canonical_digest(receipt, digest_field="ledger_receipt_digest")
    ):
        raise NativeConstructionFeedbackControllerError(
            "native_construction_search_ledger_receipt_invalid"
        )
    return receipt


def _select_candidate(
    *,
    invoker: AgentsSDKInvoker,
    authority: Mapping[str, Any],
    inventory: Mapping[str, Any],
    feedback: Mapping[str, Any] | None,
    attempted_candidate_digests: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = invoker.invoke(
        AgentsSDKAgentSpec(
            run_id=str(authority["run_id"]),
            capability="native_construction_feedback_candidate_selection",
            name="Blueprint Native Construction Feedback Controller",
            instructions=(
                "Select one exact candidate from the supplied deterministic "
                "inventory. Prefer the member whose declared addressed feedback "
                "matches the measured native failure while preserving the task. "
                "Return the inventory digest, candidate id, candidate digest, and "
                "feedback digest exactly as supplied. Never invent, interpolate, "
                "or modify a base pose, reset, entry trajectory, camera, gate, or "
                "threshold. You advise search order only; native execution grades."
            ),
            model=CONTROLLER_MODEL,
            max_turns=1,
            max_output_tokens=CONTROLLER_MAX_OUTPUT_TOKENS,
            max_input_tokens=120_000,
            reasoning_effort=CONTROLLER_REASONING_EFFORT,
            output_type=NativeConstructionCandidateSelection,
        ),
        [
            {
                "role": "user",
                "content": json.dumps(
                    _selection_input(
                        authority=authority,
                        inventory=inventory,
                        feedback=feedback,
                        attempted_candidate_digests=attempted_candidate_digests,
                    ),
                    sort_keys=True,
                ),
            }
        ],
    )
    selection = NativeConstructionCandidateSelection.model_validate(
        result.output
    ).model_dump(mode="json")
    expected_feedback = feedback["feedback_digest"] if feedback is not None else None
    members = {
        str(row["candidate_id"]): row for row in inventory["candidates"]
    }
    candidate = members.get(selection["candidate_id"])
    if candidate is None:
        raise NativeConstructionFeedbackControllerError(
            "native_construction_selection_unknown_candidate"
        )
    if (
        selection["inventory_digest"] != inventory["inventory_digest"]
        or selection["candidate_digest"] != candidate["candidate_digest"]
        or selection["addressed_feedback_digest"] != expected_feedback
    ):
        raise NativeConstructionFeedbackControllerError(
            "native_construction_selection_member_mutated"
        )
    selection.update(
        {
            "provider": result.provider,
            "model": result.model,
            "sdk_version": result.sdk_version,
            "usage": dict(result.usage),
            "trace_id": result.trace_id,
        }
    )
    return _copy(candidate), selection


def _validate_execution(
    value: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    inventory: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    execution = _copy(value)
    try:
        cost = float(execution.get("incremental_cost_upper_bound_usd"))
    except (TypeError, ValueError) as exc:
        raise NativeConstructionFeedbackControllerError(
            "native_construction_candidate_execution_invalid"
        ) from exc
    native = execution.get("native_result")
    if not isinstance(native, Mapping):
        raise NativeConstructionFeedbackControllerError(
            "native_construction_candidate_execution_invalid"
        )
    feedback = summarize_native_construction_feedback(native)
    expected_status = "passed" if feedback["passed"] else "rejected"
    if (
        execution.get("schema_version") != EXECUTION_SCHEMA_VERSION
        or execution.get("status") != expected_status
        or execution.get("candidate_id") != candidate["candidate_id"]
        or execution.get("candidate_digest") != candidate["candidate_digest"]
        or execution.get("inventory_digest") != inventory["inventory_digest"]
        or execution.get("provider_instance_id")
        != authority["expected_provider_instance_id"]
        or execution.get("provider_allocations_performed") != 0
        or not math.isfinite(cost)
        or cost < 0.0
        or cost > float(candidate["maximum_incremental_cost_usd"])
        or execution.get("execution_result_digest")
        != canonical_digest(execution, digest_field="execution_result_digest")
    ):
        raise NativeConstructionFeedbackControllerError(
            "native_construction_candidate_execution_invalid"
        )
    return execution, feedback


def _validate_controls_continuation(
    value: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    construction_qualification_digest: str,
    candidate_digest: str,
) -> dict[str, Any]:
    result = _copy(value)
    if (
        result.get("schema_version") != CONTROLS_CONTINUATION_SCHEMA_VERSION
        or result.get("status") not in {"queued", "completed"}
        or result.get("run_id") != authority["run_id"]
        or result.get("construction_qualification_digest")
        != construction_qualification_digest
        or result.get("qualified_candidate_digest") != candidate_digest
        or result.get("provider_instance_id")
        != authority["expected_provider_instance_id"]
        or result.get("provider_allocations_performed") != 0
        or result.get("controls_continuation_digest")
        != canonical_digest(result, digest_field="controls_continuation_digest")
    ):
        raise NativeConstructionFeedbackControllerError(
            "native_construction_controls_continuation_invalid"
        )
    return result


def run_native_construction_feedback_controller(
    *,
    invoker: AgentsSDKInvoker,
    authority: Mapping[str, Any],
    initial_inventory: Mapping[str, Any],
    produce_next_inventory: CandidateInventoryProducer | None,
    execute_candidate: CandidateExecutor,
    continue_to_controls: ControlsContinuation,
    candidate_generator: CandidateGenerator | None = None,
    search_ledger: SearchLedger | None = None,
    initial_native_feedback: Mapping[str, Any] | None = None,
    prior_attempted_candidate_digests: Sequence[str] = (),
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Execute bounded native candidates on one worker and continue on pass."""

    now = float(clock())
    admitted = _validate_authority(authority, now=now)
    maximum_rounds = int(admitted["maximum_rounds"])
    maximum_candidates = int(admitted["maximum_candidates_per_round"])
    starting_feedback = (
        _copy(initial_native_feedback)
        if initial_native_feedback is not None
        else None
    )
    if starting_feedback is not None and (
        starting_feedback.get("schema_version") != FEEDBACK_SCHEMA_VERSION
        or starting_feedback.get("feedback_digest")
        != canonical_digest(starting_feedback, digest_field="feedback_digest")
        or starting_feedback.get("passed") is not False
    ):
        raise NativeConstructionFeedbackControllerError(
            "native_construction_feedback_initial_feedback_invalid"
        )
    inventory = validate_native_construction_inventory(
        initial_inventory,
        expected_run_id=str(admitted["run_id"]),
        expected_round_index=0,
        expected_feedback_digest=(
            str(starting_feedback["feedback_digest"])
            if starting_feedback is not None
            else None
        ),
        maximum_candidates=maximum_candidates,
    )
    attempted_digests: list[str] = [
        str(value) for value in prior_attempted_candidate_digests
    ]
    if any(not _sha256(value) for value in attempted_digests) or len(
        set(attempted_digests)
    ) != len(attempted_digests):
        raise NativeConstructionFeedbackControllerError(
            "native_construction_feedback_prior_attempts_invalid"
        )
    candidate_id_bindings: dict[str, str] = {}
    history: list[dict[str, Any]] = []
    total_cost = 0.0
    final_feedback: dict[str, Any] | None = starting_feedback
    qualified_candidate: dict[str, Any] | None = None

    for round_index in range(maximum_rounds):
        if float(clock()) >= float(admitted["deadline_unix_s"]):
            break
        for candidate in inventory["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            candidate_digest = str(candidate["candidate_digest"])
            previous = candidate_id_bindings.setdefault(candidate_id, candidate_digest)
            if previous != candidate_digest:
                raise NativeConstructionFeedbackControllerError(
                    "native_construction_candidate_id_rebound"
                )
        if any(
            row["candidate_digest"] in attempted_digests
            for row in inventory["candidates"]
        ):
            raise NativeConstructionFeedbackControllerError(
                "native_construction_candidate_repeated"
            )
        active_inventory = inventory
        inventory_ledger_receipt = None
        if search_ledger is not None:
            inventory_ledger_receipt = _validated_ledger_receipt(
                search_ledger.record_inventory(inventory=active_inventory),
                event="inventory_recorded",
                run_id=str(admitted["run_id"]),
                round_index=round_index,
                inventory_digest=str(active_inventory["inventory_digest"]),
            )
        candidate, selection = _select_candidate(
            invoker=invoker,
            authority=admitted,
            inventory=active_inventory,
            feedback=final_feedback,
            attempted_candidate_digests=attempted_digests,
        )
        if candidate["candidate_digest"] in attempted_digests:
            raise NativeConstructionFeedbackControllerError(
                "native_construction_candidate_repeated"
            )
        if (
            total_cost + float(candidate["maximum_incremental_cost_usd"])
            > float(admitted["maximum_incremental_cost_usd"])
        ):
            raise NativeConstructionFeedbackControllerError(
                "native_construction_feedback_cost_cap_exceeded"
            )
        if (
            float(clock()) + float(candidate["maximum_runtime_seconds"])
            >= float(admitted["deadline_unix_s"])
        ):
            raise NativeConstructionFeedbackControllerError(
                "native_construction_feedback_ttl_exhausted"
            )
        execution_binding = {
            "run_id": admitted["run_id"],
            "round_index": round_index,
            "warm_session_digest": admitted["warm_session_digest"],
            "expected_provider_instance_id": admitted[
                "expected_provider_instance_id"
            ],
            "inventory_digest": active_inventory["inventory_digest"],
            "candidate_digest": candidate["candidate_digest"],
        }
        execution, feedback = _validate_execution(
            execute_candidate(_copy(candidate), execution_binding),
            candidate=candidate,
            inventory=active_inventory,
            authority=admitted,
        )
        attempted_digests.append(str(candidate["candidate_digest"]))
        # Candidate ceiling covers both remote candidate generation and native
        # execution on the continuously billed warm worker. The allocator
        # result measures execution only, so charging merely that value would
        # make cuRobo planning time disappear from the controller budget.
        total_cost += max(
            float(execution["incremental_cost_upper_bound_usd"]),
            float(candidate["maximum_incremental_cost_usd"]),
        )
        if total_cost > float(admitted["maximum_incremental_cost_usd"]):
            raise NativeConstructionFeedbackControllerError(
                "native_construction_feedback_cost_cap_exceeded"
            )
        round_record = {
                "round_index": round_index,
                "inventory_digest": active_inventory["inventory_digest"],
                "source_native_feedback_digest": active_inventory[
                    "source_native_feedback_digest"
                ],
                "selection": selection,
                "inventory_ledger_receipt": inventory_ledger_receipt,
                "candidate": candidate,
                "execution": execution,
                "native_feedback": feedback,
                "controller_search_state": (
                    "qualified"
                    if feedback["passed"]
                    else "exhausted_round_cap"
                    if round_index + 1 >= maximum_rounds
                    else "continuing"
                ),
                "attempt_ledger_receipt": None,
            }
        if search_ledger is not None:
            round_record["attempt_ledger_receipt"] = _validated_ledger_receipt(
                search_ledger.record_attempt(round_record=round_record),
                event="attempt_recorded",
                run_id=str(admitted["run_id"]),
                round_index=round_index,
                inventory_digest=str(active_inventory["inventory_digest"]),
                candidate_digest=str(candidate["candidate_digest"]),
                execution_result_digest=str(execution["execution_result_digest"]),
                feedback_digest=str(feedback["feedback_digest"]),
            )
        history.append(round_record)
        final_feedback = feedback
        if feedback["passed"]:
            qualified_candidate = candidate
            break
        if round_index + 1 >= maximum_rounds:
            break
        if candidate_generator is not None:
            next_inventory = candidate_generator.generate(
                source_native_feedback=feedback,
                prior_history=tuple(history),
                round_index=round_index + 1,
                maximum_candidates=maximum_candidates,
            )
        elif produce_next_inventory is not None:
            next_inventory = produce_next_inventory(
                feedback, tuple(history), round_index + 1
            )
        else:
            raise NativeConstructionFeedbackControllerError(
                "native_construction_candidate_generator_missing"
            )
        inventory = validate_native_construction_inventory(
            next_inventory,
            expected_run_id=str(admitted["run_id"]),
            expected_round_index=round_index + 1,
            expected_feedback_digest=str(feedback["feedback_digest"]),
            maximum_candidates=maximum_candidates,
        )

    status: Literal["construction_passed", "exhausted"] = (
        "construction_passed" if qualified_candidate is not None else "exhausted"
    )
    blockers = (
        []
        if qualified_candidate is not None
        else [
            "native_construction_feedback_candidates_exhausted"
            if float(clock()) < float(admitted["deadline_unix_s"])
            else "native_construction_feedback_ttl_exhausted"
        ]
    )
    construction_qualification_digest = (
        canonical_digest(
            {
                "run_id": admitted["run_id"],
                "authority_digest": admitted["authority_digest"],
                "warm_session_digest": admitted["warm_session_digest"],
                "provider_instance_id": admitted[
                    "expected_provider_instance_id"
                ],
                "history": history,
                "qualified_candidate_digest": qualified_candidate[
                    "candidate_digest"
                ],
                "final_native_feedback_digest": final_feedback[
                    "feedback_digest"
                ],
            }
        )
        if qualified_candidate is not None and final_feedback is not None
        else None
    )
    receipt: dict[str, Any] = {
        "schema_version": CONTROLLER_RECEIPT_SCHEMA_VERSION,
        "status": status,
        "run_id": admitted["run_id"],
        "authority_digest": admitted["authority_digest"],
        "warm_session_digest": admitted["warm_session_digest"],
        "provider_instance_id": admitted["expected_provider_instance_id"],
        "provider_allocations_performed": 0,
        "allocator_retry_cap": 0,
        "round_cap": maximum_rounds,
        "round_count": len(history),
        "attempted_candidate_digests": attempted_digests,
        "incremental_cost_upper_bound_usd": total_cost,
        "history": history,
        "qualified_candidate_digest": (
            qualified_candidate["candidate_digest"]
            if qualified_candidate is not None
            else None
        ),
        "construction_qualification_digest": construction_qualification_digest,
        "final_native_feedback_digest": (
            final_feedback["feedback_digest"] if final_feedback is not None else None
        ),
        "controls_continuation_required": qualified_candidate is not None,
        "controls_continuation": None,
        "blockers": blockers,
        "gates_or_thresholds_modified": False,
        "model_graded_construction": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    if qualified_candidate is not None:
        continuation = _validate_controls_continuation(
            continue_to_controls(_copy(receipt)),
            authority=admitted,
            construction_qualification_digest=str(
                construction_qualification_digest
            ),
            candidate_digest=str(qualified_candidate["candidate_digest"]),
        )
        receipt["controls_continuation"] = continuation
        receipt["controls_continuation_required"] = False
        receipt["status"] = (
            "controls_completed"
            if continuation["status"] == "completed"
            else "controls_continuation_queued"
        )
        receipt["receipt_digest"] = ""
        receipt["receipt_digest"] = canonical_digest(
            receipt, digest_field="receipt_digest"
        )
    return receipt


__all__ = [
    "AUTHORITY_SCHEMA_VERSION",
    "CANDIDATE_SCHEMA_VERSION",
    "CONTROLLER_RECEIPT_SCHEMA_VERSION",
    "CONTROLS_CONTINUATION_SCHEMA_VERSION",
    "EXECUTION_SCHEMA_VERSION",
    "FEEDBACK_SCHEMA_VERSION",
    "INVENTORY_SCHEMA_VERSION",
    "NativeConstructionCandidateSelection",
    "NativeConstructionFeedbackControllerError",
    "CandidateGenerator",
    "CompositeCandidateGenerator",
    "SearchLedger",
    "bind_warm_native_construction_execution",
    "build_next_native_construction_inventory",
    "construction_phase_plan_for_candidate",
    "native_construction_feedback_codes",
    "run_native_construction_feedback_controller",
    "summarize_native_construction_feedback",
    "validate_native_construction_candidate",
    "validate_native_construction_inventory",
]
