"""Materialize the CPU-first configured-scene controls continuation.

Turn one immutable launch profile into a controls plan after scene publication.
CPU geometry and trajectory gates enumerate exact candidates; the production
OpenAI Agents SDK reviewer may select but not mutate one inventory member. This
module never allocates a GPU resource or submits a launch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .openai_official_cost_gate import (
    RUN_COMPLETION_SCHEMA_VERSION,
    RUN_RESERVATION_SCHEMA_VERSION,
)
from .task_evaluation_configured_controls_plan import (
    materialize_configured_controls_plan,
)
from .task_evaluation_configured_scene_revision import (
    validate_configured_scene_revision,
)
from .task_evaluation_robot_placement_agent_cli import run_robot_placement_cli
from .task_evaluation_robot_placement_agent import (
    _reject_infeasible_orientation_slew,
    validate_robot_placement_receipt,
)
from .task_evaluation_robot_placement_agent import (
    ROBOT_PLACEMENT_AGENT_MODEL,
    ROBOT_PLACEMENT_AGENT_REASONING_EFFORT,
)
from .task_evaluation_robot_placement_readiness_candidate import (
    materialize_robot_placement_readiness_candidate,
)
from .task_evaluation_robot_placement_geometry import (
    validate_robot_placement_trajectory_position_ik,
)
from .task_evaluation_robot_placement_trajectory import (
    placement_trajectory_from_native_plan,
    validate_robot_placement_trajectory,
)
from .task_evaluation_shared_mutation_window import (
    TaskEvaluationSharedMutationWindowError,
    validate_shared_mutation_window_template,
)
from .task_evaluation_configured_controls_openai_placement import (
    PAID_RESOURCE_CLASS as OPENAI_PLACEMENT_PAID_RESOURCE_CLASS,
    VISUAL_REVIEW_CREDENTIAL_ROLE,
    configured_controls_robot_placement_openai_gate,
    exclusive_visual_review_cost_scope,
)
from .task_evaluation_native_construction_feedback_controller import (
    CANDIDATE_SCHEMA_VERSION as NATIVE_FEEDBACK_CANDIDATE_SCHEMA_VERSION,
    build_next_native_construction_inventory,
)
from .task_evaluation_native_interaction_variants import (
    build_native_interaction_variants,
)
from .task_evaluation_openai_inference_usage import (
    build_placement_inference_usage_packet,
    sync_inference_usage_to_webapp,
)

INTENT_SCHEMA_VERSION = "task_evaluation_configured_controls_autostart_intent.v2"
RESULT_SCHEMA_VERSION = "task_evaluation_configured_controls_autostart.v3"
DEFAULT_MAX_PLACEMENT_INFERENCE_COST_USD = 2.56
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_FIXED_PATHS = {
    "robot_asset_usd_path",
    "robot_mount_interface_path",
    "scene_camera_calibration_path",
    "native_trajectory_plan_path",
    "cameras_path",
    "runtime_binding_path",
}
_PHASE_PATHS = {
    "construction": {
        "release_window_template_path",
        "lineage_path",
        "authorization_path",
        "launch_authority_path",
    },
    "controls": {
        "release_window_template_path",
        "authorization_path",
        "launch_authority_path",
    },
}


class TaskEvaluationConfiguredControlsAutostartError(RuntimeError):
    """Automatic continuation input or CPU evidence was unsafe."""


PlacementRunner = Callable[..., Mapping[str, Any]]
ReadinessMaterializer = Callable[..., Mapping[str, Any]]
PlanMaterializer = Callable[..., Mapping[str, Any]]
_PLACEMENT_CHECKPOINT_SCHEMA_VERSION = (
    "task_evaluation_configured_controls_cpu_placement_checkpoint.v1"
)
_BOUND_CPU_PLACEMENT_CHECKPOINT_SCHEMA_VERSION = (
    "task_evaluation_configured_controls_cpu_placement_checkpoint.v2"
)
_CPU_PLACEMENT_BINDING_SCHEMA_VERSION = (
    "task_evaluation_configured_controls_cpu_placement_binding.v1"
)
_AGENT_PLACEMENT_CHECKPOINT_SCHEMA_VERSION = (
    "task_evaluation_configured_controls_agent_placement_checkpoint.v1"
)
_PLACEMENT_CAMERA_SCHEMA_VERSION = (
    "task_evaluation_placement_aware_camera_candidates.v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, *, blocker: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationConfiguredControlsAutostartError(blocker) from exc
    if path.is_symlink() or not path.is_file() or not isinstance(value, Mapping):
        raise TaskEvaluationConfiguredControlsAutostartError(blocker)
    return dict(value)


def _finite_vector(value: Any, length: int, *, blocker: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        raise TaskEvaluationConfiguredControlsAutostartError(blocker)
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise TaskEvaluationConfiguredControlsAutostartError(blocker) from exc
    if not all(math.isfinite(item) for item in result):
        raise TaskEvaluationConfiguredControlsAutostartError(blocker)
    return result


def _normalize_vector(value: Sequence[float], *, blocker: str) -> list[float]:
    norm = math.sqrt(sum(float(item) ** 2 for item in value))
    if not math.isfinite(norm) or norm <= 1.0e-9:
        raise TaskEvaluationConfiguredControlsAutostartError(blocker)
    return [float(item) / norm for item in value]


def _look_at_matrix(
    position: Sequence[float], target: Sequence[float]
) -> list[float]:
    forward = _normalize_vector(
        [float(target[index]) - float(position[index]) for index in range(3)],
        blocker="configured_controls_autostart_camera_aim_degenerate",
    )
    right = _normalize_vector(
        [forward[1], -forward[0], 0.0],
        blocker="configured_controls_autostart_camera_aim_degenerate",
    )
    down = _normalize_vector(
        [
            forward[1] * right[2] - forward[2] * right[1],
            forward[2] * right[0] - forward[0] * right[2],
            forward[0] * right[1] - forward[1] * right[0],
        ],
        blocker="configured_controls_autostart_camera_aim_degenerate",
    )
    return [
        right[0], down[0], forward[0], float(position[0]),
        right[1], down[1], forward[1], float(position[1]),
        right[2], down[2], forward[2], float(position[2]),
        0.0, 0.0, 0.0, 1.0,
    ]


def _world_camera_candidate(
    role: str,
    *,
    position: Sequence[float],
    target: Sequence[float],
    intrinsics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "role": role,
        "policy_input": role == "external",
        "scoring_input": False,
        "pose_frame": "world",
        "parent_prim_path": "{ENV_REGEX_NS}",
        "optical_convention": "opencv",
        "frame_from_camera_matrix": _look_at_matrix(position, target),
        "intrinsics": json.loads(json.dumps(dict(intrinsics), allow_nan=False)),
    }


def _placement_aware_camera_candidates(
    *,
    camera_template: Mapping[str, Any],
    accepted_pose: Mapping[str, Any],
    selected_candidate_id: str,
    trajectory: Mapping[str, Any],
    source_commit: str,
) -> dict[str, Any]:
    """Derive world cameras after CPU placement; retain only the wrist mount.

    A prelaunch profile can bind immutable DROID intrinsics and its robot-body
    wrist mount.  Its world cameras cannot be authoritative because the exact
    Franka base does not exist until deterministic trajectory placement has
    selected one inventory member.  Recompute those two cameras from the exact
    selected pose and full trajectory, and leave native observability as the
    final authority.
    """

    validated_trajectory = validate_robot_placement_trajectory(trajectory)
    rows = camera_template.get("cameras")
    by_role = {
        str(row.get("role") or ""): json.loads(json.dumps(dict(row), allow_nan=False))
        for row in rows or []
        if isinstance(row, Mapping)
    }
    wrist = by_role.get("wrist") or {}
    intrinsics = (by_role.get("external") or {}).get("intrinsics")
    if (
        not isinstance(rows, list)
        or len(rows) != 3
        or set(by_role) != {"external", "wrist", "overview"}
        or wrist.get("pose_frame") != "robot_body"
        or wrist.get("parent_prim_path")
        != "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link"
        or wrist.get("policy_input") is not True
        or wrist.get("scoring_input") is not False
        or wrist.get("optical_convention") != "opencv"
        or not isinstance(intrinsics, Mapping)
        or intrinsics != wrist.get("intrinsics")
    ):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_camera_template_invalid"
        )
    base = _finite_vector(
        accepted_pose.get("position_world_m"),
        3,
        blocker="configured_controls_autostart_camera_base_pose_invalid",
    )
    orientation = _finite_vector(
        accepted_pose.get("orientation_xyzw"),
        4,
        blocker="configured_controls_autostart_camera_base_pose_invalid",
    )
    if not math.isclose(
        sum(item * item for item in orientation),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_camera_base_pose_invalid"
        )
    phases = validated_trajectory["phases"]
    points = [
        _finite_vector(
            row.get("position_world_m"),
            3,
            blocker="configured_controls_autostart_camera_trajectory_invalid",
        )
        for row in phases
    ]
    focus = [
        sum(point[index] for point in points) / len(points) for index in range(3)
    ]
    longest = max(
        (
            (
                math.hypot(right[0] - left[0], right[1] - left[1]),
                left,
                right,
            )
            for left in points
            for right in points
        ),
        key=lambda row: row[0],
    )
    if longest[0] <= 1.0e-9:
        base_to_focus = [focus[0] - base[0], focus[1] - base[1], 0.0]
        if math.hypot(base_to_focus[0], base_to_focus[1]) <= 1.0e-9:
            x, y, z, w = orientation
            base_to_focus = [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y + w * z),
                0.0,
            ]
        direction = _normalize_vector(
            base_to_focus,
            blocker="configured_controls_autostart_camera_trajectory_invalid",
        )
    else:
        direction = _normalize_vector(
            [
                longest[2][0] - longest[1][0],
                longest[2][1] - longest[1][1],
                0.0,
            ],
            blocker="configured_controls_autostart_camera_trajectory_invalid",
        )
    lateral = [-direction[1], direction[0], 0.0]
    external_position = [base[0], base[1], base[2] + 1.35]
    overview_position = [
        focus[0] + 0.9 * lateral[0],
        focus[1] + 0.9 * lateral[1],
        max(base[2], max(point[2] for point in points)) + 1.45,
    ]
    cameras = [
        _world_camera_candidate(
            "external",
            position=external_position,
            target=focus,
            intrinsics=intrinsics,
        ),
        wrist,
        _world_camera_candidate(
            "overview",
            position=overview_position,
            target=focus,
            intrinsics=intrinsics,
        ),
    ]
    cameras[2]["policy_input"] = False
    result: dict[str, Any] = {
        "schema_version": _PLACEMENT_CAMERA_SCHEMA_VERSION,
        "status": "candidate_pending_native_observability_readback",
        "source_commit": source_commit,
        "selected_candidate_id": selected_candidate_id,
        "accepted_pose": {
            "position_world_m": base,
            "orientation_xyzw": orientation,
        },
        "trajectory_digest": validated_trajectory["trajectory_digest"],
        "camera_template_digest": canonical_digest(camera_template),
        "derivation_method": "selected_base_and_full_trajectory_look_at",
        "world_camera_positions_depend_on_selected_base": True,
        "wrist_mount_copied_from_immutable_profile": True,
        "camera_configuration_qualified": False,
        "native_observability_readback_required": True,
        "cameras": cameras,
        "document_digest": "",
    }
    result["document_digest"] = canonical_digest(
        result, digest_field="document_digest"
    )
    return result


def _materialize_placement_aware_cameras(
    *,
    root: Path,
    camera_template_path: Path,
    accepted_pose: Mapping[str, Any],
    selected_candidate_id: str,
    trajectory: Mapping[str, Any],
    source_commit: str,
) -> Path:
    value = _placement_aware_camera_candidates(
        camera_template=_read(
            camera_template_path,
            blocker="configured_controls_autostart_camera_template_invalid",
        ),
        accepted_pose=accepted_pose,
        selected_candidate_id=selected_candidate_id,
        trajectory=trajectory,
        source_commit=source_commit,
    )
    # The document embeds source_commit, so the filename must carry it too;
    # otherwise each redeploy collides with its predecessor bytes.
    destination = (
        root / f"placement-aware-camera-candidates-{source_commit[:12]}.v1.json"
    )
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
        destination.chmod(0o440)
    except FileExistsError:
        if destination.is_symlink() or destination.read_bytes() != payload:
            raise TaskEvaluationConfiguredControlsAutostartError(
                "configured_controls_autostart_camera_candidate_conflict"
            ) from None
    return destination


def _native_feedback_candidate_universe(
    *,
    run_id: str,
    inventory: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    camera_template: Mapping[str, Any],
    source_commit: str,
    maximum_candidates: int,
) -> dict[str, Any]:
    """Compile bounded base/reset/entry/camera variants before GPU spend."""

    validated_trajectory = validate_robot_placement_trajectory(trajectory)
    phases = list(validated_trajectory["phases"])
    first = phases[0]
    authored_positions = [list(row["position_world_m"]) for row in phases]
    authored_ids = [str(row["phase_id"]) for row in phases]
    authored_orientations = [
        list(row["orientation_world_xyzw"]) for row in phases
    ]
    candidates: list[dict[str, Any]] = []
    for base_rank, raw in enumerate(inventory.get("candidates") or []):
        if not isinstance(raw, Mapping):
            continue
        proposal = {
            "candidate_id": str(raw.get("candidate_id") or ""),
            "pose": json.loads(json.dumps(raw.get("pose"))),
            "support_surface_id": str(raw.get("support_surface_id") or ""),
        }
        # The compact inventory deliberately retains only the digest of the
        # already-passed geometry gate.  Recompute the task-aware orientation
        # report without pretending that compact row is the full gate document.
        gate = _reject_infeasible_orientation_slew(
            gate={
                "status": "passed",
                "blockers": [],
                "source_geometry_gate_digest": raw.get("geometry_gate_digest"),
            },
            proposal=proposal,
            trajectory=validated_trajectory,
            robot_id="franka_panda",
            maximum_steps_per_phase=int(
                validated_trajectory["maximum_steps_per_phase"]
            ),
        )
        orientation_gate = gate.get("orientation_slew_feasibility")
        reset = (
            orientation_gate.get("task_aware_reset")
            if isinstance(orientation_gate, Mapping)
            else None
        )
        joints = (
            reset.get("joint_positions_rad")
            if isinstance(reset, Mapping)
            else None
        )
        if (
            gate.get("status") != "passed"
            or not isinstance(joints, list)
            or len(joints) != 7
        ):
            continue
        reset_variant: dict[str, Any] = {
            "schema_version": "task_evaluation_native_robot_reset_variant.v1",
            "robot_joint_reset_positions_rad": {
                f"panda_joint{index}": float(value)
                for index, value in enumerate(joints, start=1)
            },
            "source_orientation_slew_feasibility_digest": canonical_digest(
                orientation_gate
            ),
            "reset_variant_digest": "",
        }
        reset_variant["reset_variant_digest"] = canonical_digest(
            reset_variant, digest_field="reset_variant_digest"
        )
        camera_document = _placement_aware_camera_candidates(
            camera_template=camera_template,
            accepted_pose=proposal["pose"],
            selected_candidate_id=proposal["candidate_id"],
            trajectory=validated_trajectory,
            source_commit=source_commit,
        )
        camera_variant: dict[str, Any] = {
            "schema_version": "task_evaluation_native_camera_variant.v1",
            "cameras": camera_document["cameras"],
            "source_camera_document_digest": camera_document["document_digest"],
            "camera_variant_digest": "",
        }
        camera_variant["camera_variant_digest"] = canonical_digest(
            camera_variant, digest_field="camera_variant_digest"
        )
        base = list(proposal["pose"]["position_world_m"])
        precontact = list(first["position_world_m"])
        vector = [base[index] - precontact[index] for index in range(3)]
        planar_norm = math.hypot(vector[0], vector[1])
        radial = (
            [vector[0] / planar_norm, vector[1] / planar_norm, 0.0]
            if planar_norm > 1.0e-9
            else [0.0, 0.0, 0.0]
        )
        entry_options = (
            (
                "direct",
                [
                    {
                        "waypoint_id": "direct-precontact",
                        "position_world_m": precontact,
                        "orientation_world_xyzw": list(
                            first["orientation_world_xyzw"]
                        ),
                    }
                ],
                [],
            ),
            (
                "overhead",
                [
                    {
                        "waypoint_id": "overhead-clearance",
                        "position_world_m": [
                            precontact[0], precontact[1], precontact[2] + 0.12
                        ],
                        "orientation_world_xyzw": list(
                            first["orientation_world_xyzw"]
                        ),
                    },
                    {
                        "waypoint_id": "overhead-precontact",
                        "position_world_m": precontact,
                        "orientation_world_xyzw": list(
                            first["orientation_world_xyzw"]
                        ),
                    },
                ],
                [
                    "collision:precontact:robot_task_forbidden_collision",
                    "collision:precontact:robot_scene_contact",
                ],
            ),
            (
                "radial_standoff",
                [
                    {
                        "waypoint_id": "radial-standoff",
                        "position_world_m": [
                            precontact[0] + 0.10 * radial[0],
                            precontact[1] + 0.10 * radial[1],
                            precontact[2] + 0.08,
                        ],
                        "orientation_world_xyzw": list(
                            first["orientation_world_xyzw"]
                        ),
                    },
                    {
                        "waypoint_id": "radial-precontact",
                        "position_world_m": precontact,
                        "orientation_world_xyzw": list(
                            first["orientation_world_xyzw"]
                        ),
                    },
                ],
                [
                    "collision:precontact:robot_task_forbidden_collision",
                    f"phase_unreached:{first['phase_id']}",
                ],
            ),
        )
        interaction_options = build_native_interaction_variants(
            phases=phases,
            first_phase_id=str(first["phase_id"]),
            base_rank=base_rank,
            trajectory_digest=validated_trajectory["trajectory_digest"],
        )
        paired_options = (
            entry_options[0],
            entry_options[1],
            entry_options[2],
            entry_options[0],
        )
        for option_index, (
            (option_id, entry_waypoints, entry_feedback_codes),
            interaction,
        ) in enumerate(zip(paired_options, interaction_options, strict=True)):
            interaction_id = interaction["branch_id"]
            interaction_feedback_codes = interaction["feedback_codes"]
            combined_positions = [
                *[list(row["position_world_m"]) for row in entry_waypoints],
                *authored_positions,
            ]
            combined_ids = [
                *[f"feedback_{row['waypoint_id']}" for row in entry_waypoints],
                *authored_ids,
            ]
            combined_orientations = [
                *[list(row["orientation_world_xyzw"]) for row in entry_waypoints],
                *authored_orientations,
            ]
            entry_gate = validate_robot_placement_trajectory_position_ik(
                proposal=proposal,
                trajectory_waypoints_world_m=combined_positions,
                trajectory_phase_ids=combined_ids,
                trajectory_orientations_world_xyzw=combined_orientations,
            )
            if entry_gate.get("status") != "passed":
                continue
            entry_variant: dict[str, Any] = {
                "schema_version": "task_evaluation_native_entry_trajectory_variant.v1",
                "entry_strategy": option_id,
                "joins_authored_phase_id": str(first["phase_id"]),
                "waypoints": entry_waypoints,
                "cpu_position_ik_gate_digest": entry_gate[
                    "trajectory_position_ik_gate_digest"
                ],
                "native_swept_collision_and_orientation_required": True,
                "entry_trajectory_variant_digest": "",
            }
            entry_variant["entry_trajectory_variant_digest"] = canonical_digest(
                entry_variant, digest_field="entry_trajectory_variant_digest"
            )
            interaction_variant = interaction["variant"]
            candidate: dict[str, Any] = {
                "schema_version": NATIVE_FEEDBACK_CANDIDATE_SCHEMA_VERSION,
                "candidate_id": (
                    f"{proposal['candidate_id']}--{option_id}--{interaction_id}"
                ),
                "deterministic_rank": base_rank * 4 + option_index,
                "robot_base_pose_world": proposal["pose"],
                "support_surface_id": proposal["support_surface_id"],
                "reset_variant": reset_variant,
                "entry_trajectory_variant": entry_variant,
                "interaction_trajectory_variant": interaction_variant,
                "camera_variant": camera_variant,
                "source_placement_candidate_id": proposal["candidate_id"],
                "source_placement_geometry_gate_digest": raw.get(
                    "geometry_gate_digest"
                ),
                "addressed_feedback_codes": sorted(
                    set(entry_feedback_codes) | set(interaction_feedback_codes)
                ),
                "maximum_incremental_cost_usd": 0.12,
                "maximum_runtime_seconds": 360.0,
                "candidate_digest": "",
            }
            candidate["candidate_digest"] = canonical_digest(
                candidate, digest_field="candidate_digest"
            )
            candidates.append(candidate)
    return build_next_native_construction_inventory(
        run_id=run_id,
        round_index=0,
        source_native_feedback=None,
        prior_history=(),
        candidate_universe=candidates,
        maximum_candidates=min(int(maximum_candidates), 64),
    )


def _materialize_native_feedback_candidate_universe(
    *,
    root: Path,
    run_id: str,
    inventory: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    camera_template_path: Path,
    source_commit: str,
    maximum_candidates: int,
) -> tuple[Path, dict[str, Any]]:
    value = _native_feedback_candidate_universe(
        run_id=run_id,
        inventory=inventory,
        trajectory=trajectory,
        camera_template=_read(
            camera_template_path,
            blocker="configured_controls_autostart_camera_template_invalid",
        ),
        source_commit=source_commit,
        maximum_candidates=maximum_candidates,
    )
    destination = root / (
        f"native-construction-feedback-candidates-{source_commit[:12]}.v1.json"
    )
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
        destination.chmod(0o440)
    except FileExistsError:
        if destination.is_symlink() or destination.read_bytes() != payload:
            raise TaskEvaluationConfiguredControlsAutostartError(
                "configured_controls_autostart_native_candidate_universe_conflict"
            ) from None
    return destination, value


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_artifact_invalid"
        )
    metadata = path.stat()
    return {
        "path": str(path),
        "digest": _sha256(path),
        "size_bytes": metadata.st_size,
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
    }


def _intent_paths(value: Mapping[str, Any]) -> dict[str, Path]:
    paths = value.get("paths")
    phases = value.get("phases")
    if (
        not isinstance(paths, Mapping)
        or set(paths) != _FIXED_PATHS | {"overview_image_paths"}
        or not isinstance(paths.get("overview_image_paths"), list)
        or not paths["overview_image_paths"]
        or not isinstance(phases, Mapping)
        or set(phases) != set(_PHASE_PATHS)
    ):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_intent_paths_invalid"
        )
    flattened = {name: Path(str(paths[name])).expanduser() for name in _FIXED_PATHS}
    for index, item in enumerate(paths["overview_image_paths"]):
        flattened[f"overview_image_paths.{index}"] = Path(str(item)).expanduser()
    for phase, expected in _PHASE_PATHS.items():
        row = phases.get(phase)
        if not isinstance(row, Mapping) or set(row) != expected:
            raise TaskEvaluationConfiguredControlsAutostartError(
                "configured_controls_autostart_intent_paths_invalid"
            )
        for name in expected:
            flattened[f"phases.{phase}.{name}"] = Path(str(row[name])).expanduser()
    if any(not path.is_absolute() for path in flattened.values()):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_intent_paths_invalid"
        )
    return flattened


def validate_configured_controls_autostart_intent(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    intent = json.loads(json.dumps(dict(value), allow_nan=False))
    target = intent.get("target_position_world_m")
    placement = intent.get("placement")
    placement_authority = (
        placement.get("official_cost_authority")
        if isinstance(placement, Mapping)
        else None
    )
    adoption = intent.get("configuration_adoption")
    if (
        intent.get("schema_version") != INTENT_SCHEMA_VERSION
        or intent.get("enabled") is not True
        or _COMMIT.fullmatch(str(intent.get("expected_production_commit") or "")) is None
        or _COMMIT.fullmatch(str(intent.get("configuration_source_commit") or "")) is None
        or not str(intent.get("submitted_by") or "").strip()
        or not str(intent.get("team_namespace") or "").strip()
        or not str(intent.get("scene_id") or "").strip()
        or not str(intent.get("task_id") or "").strip()
        or not isinstance(target, list)
        or len(target) != 3
        or not all(math.isfinite(float(item)) for item in target)
        or not isinstance(placement, Mapping)
        or set(placement)
        != {
            "robot_id",
            "max_rounds",
            "candidate_inventory_cap",
            "max_input_tokens",
            "max_inference_cost_usd",
            "expected_proposal_reuse_probability",
            "expected_visual_review_reuse_probability",
            "expected_proposal_reuse_count",
            "expected_visual_review_reuse_count",
            "agent_selection_required",
            "agent_model",
            "reasoning_effort",
            "official_cost_authority",
        }
        or placement.get("robot_id") != "franka_panda"
        or not 1 <= int(placement.get("max_rounds", 0)) <= 8
        or not 1 <= int(placement.get("candidate_inventory_cap", 0)) <= 128
        or not 1 <= int(placement.get("max_input_tokens", 0)) <= 1_000_000
        or float(placement.get("max_inference_cost_usd", -1.0))
        != DEFAULT_MAX_PLACEMENT_INFERENCE_COST_USD
        or not 0
        <= float(placement.get("expected_proposal_reuse_probability", -1.0))
        <= 1
        or not 0 <= int(placement.get("expected_proposal_reuse_count", -1)) <= 20
        or not 0
        <= int(placement.get("expected_visual_review_reuse_count", -1))
        <= 20
        or not 0
        <= float(placement.get("expected_visual_review_reuse_probability", -1.0))
        <= 1
        or placement.get("agent_selection_required") is not True
        or placement.get("agent_model") != ROBOT_PLACEMENT_AGENT_MODEL
        or placement.get("reasoning_effort")
        != ROBOT_PLACEMENT_AGENT_REASONING_EFFORT
        or not isinstance(placement_authority, Mapping)
        or set(placement_authority)
        != {
            "provider_id",
            "credential_role",
            "project_id",
            "api_key_id",
            "paid_resource_class",
            "maximum_cost_usd",
        }
        or placement_authority.get("provider_id") != "openai"
        or placement_authority.get("credential_role")
        != VISUAL_REVIEW_CREDENTIAL_ROLE
        or not str(placement_authority.get("project_id") or "").strip()
        or not str(placement_authority.get("api_key_id") or "").strip()
        or placement_authority.get("paid_resource_class")
        != OPENAI_PLACEMENT_PAID_RESOURCE_CLASS
        or float(placement_authority.get("maximum_cost_usd", -1.0))
        != float(placement["max_inference_cost_usd"])
        or not Path(str(intent.get("profile_dir") or "")).is_absolute()
        or intent.get("provider_mutation_performed") is not False
        or intent.get("paid_execution_requested") is not True
        or not isinstance(adoption, Mapping)
        or intent.get("intent_digest")
        != canonical_digest(intent, digest_field="intent_digest")
    ):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_intent_invalid"
        )
    if adoption.get("mode") == "same_commit_automatic":
        if (
            set(adoption) != {"mode"}
            or intent["configuration_source_commit"]
            != intent["expected_production_commit"]
        ):
            raise TaskEvaluationConfiguredControlsAutostartError(
                "configured_controls_autostart_adoption_invalid"
            )
    elif adoption.get("mode") == "explicit_terminal_adoption":
        if (
            set(adoption)
            != {
                "mode",
                "source_launch_id",
                "source_launch_receipt_digest",
                "terminal_result_digest",
                "configured_scene_revision_digest",
                "publication_result_digest",
                "webapp_sync_result_digest",
                "provider_zero_receipt_digest",
            }
            or not str(adoption.get("source_launch_id") or "").strip()
            or any(
                _DIGEST.fullmatch(str(adoption.get(field) or "")) is None
                for field in (
                    "source_launch_receipt_digest",
                    "terminal_result_digest",
                    "configured_scene_revision_digest",
                    "publication_result_digest",
                    "webapp_sync_result_digest",
                    "provider_zero_receipt_digest",
                )
            )
        ):
            raise TaskEvaluationConfiguredControlsAutostartError(
                "configured_controls_autostart_adoption_invalid"
            )
    else:
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_adoption_invalid"
        )
    flattened = _intent_paths(intent)
    inventory = intent.get("artifact_inventory")
    if not isinstance(inventory, Mapping) or set(inventory) != set(flattened):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_inventory_invalid"
        )
    for name, path in flattened.items():
        row = inventory.get(name)
        if not isinstance(row, Mapping) or dict(row) != _artifact(path):
            raise TaskEvaluationConfiguredControlsAutostartError(
                "configured_controls_autostart_inventory_invalid"
            )
    for phase in ("construction", "controls"):
        try:
            validate_shared_mutation_window_template(
                _read(
                    flattened[f"phases.{phase}.release_window_template_path"],
                    blocker="configured_controls_autostart_release_window_template_invalid",
                ),
                team_namespace=str(intent["team_namespace"]),
                expected_production_commit=str(intent["expected_production_commit"]),
            )
        except TaskEvaluationSharedMutationWindowError as exc:
            raise TaskEvaluationConfiguredControlsAutostartError(
                "configured_controls_autostart_release_window_template_invalid"
            ) from exc
    return intent


def materialize_configured_controls_autostart_intent(
    *,
    expected_production_commit: str,
    submitted_by: str,
    team_namespace: str,
    scene_id: str,
    task_id: str,
    target_position_world_m: Sequence[float],
    paths: Mapping[str, Any],
    phases: Mapping[str, Any],
    profile_dir: str | Path,
    output_path: str | Path,
    max_rounds: int = 2,
    candidate_inventory_cap: int = 24,
    max_input_tokens: int = 120_000,
    max_inference_cost_usd: float = DEFAULT_MAX_PLACEMENT_INFERENCE_COST_USD,
    expected_proposal_reuse_probability: float = 0.0,
    expected_visual_review_reuse_probability: float = 0.0,
    expected_proposal_reuse_count: int = 0,
    expected_visual_review_reuse_count: int = 0,
    openai_project_id: str,
    openai_api_key_id: str,
    configuration_source_commit: str | None = None,
    configuration_adoption: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal all fixed downstream bytes before the configuration launch."""

    draft: dict[str, Any] = {
        "schema_version": INTENT_SCHEMA_VERSION,
        "enabled": True,
        "expected_production_commit": expected_production_commit,
        "configuration_source_commit": (
            configuration_source_commit or expected_production_commit
        ),
        "configuration_adoption": json.loads(
            json.dumps(
                dict(configuration_adoption)
                if configuration_adoption is not None
                else {"mode": "same_commit_automatic"}
            )
        ),
        "submitted_by": submitted_by,
        "team_namespace": team_namespace,
        "scene_id": scene_id,
        "task_id": task_id,
        "target_position_world_m": [float(item) for item in target_position_world_m],
        "placement": {
            "robot_id": "franka_panda",
            "max_rounds": int(max_rounds),
            "candidate_inventory_cap": int(candidate_inventory_cap),
            "max_input_tokens": int(max_input_tokens),
            "max_inference_cost_usd": float(max_inference_cost_usd),
            "expected_proposal_reuse_probability": float(
                expected_proposal_reuse_probability
            ),
            "expected_visual_review_reuse_probability": float(
                expected_visual_review_reuse_probability
            ),
            "expected_proposal_reuse_count": int(expected_proposal_reuse_count),
            "expected_visual_review_reuse_count": int(
                expected_visual_review_reuse_count
            ),
            "agent_selection_required": True,
            "agent_model": ROBOT_PLACEMENT_AGENT_MODEL,
            "reasoning_effort": ROBOT_PLACEMENT_AGENT_REASONING_EFFORT,
            "official_cost_authority": {
                "provider_id": "openai",
                "credential_role": VISUAL_REVIEW_CREDENTIAL_ROLE,
                "project_id": str(openai_project_id),
                "api_key_id": str(openai_api_key_id),
                "paid_resource_class": OPENAI_PLACEMENT_PAID_RESOURCE_CLASS,
                "maximum_cost_usd": float(max_inference_cost_usd),
            },
        },
        "profile_dir": str(Path(profile_dir).expanduser()),
        "paths": json.loads(json.dumps(dict(paths))),
        "phases": json.loads(json.dumps(dict(phases))),
        "artifact_inventory": {},
        "provider_mutation_performed": False,
        "paid_execution_requested": True,
        "intent_digest": "",
    }
    flattened = _intent_paths(draft)
    draft["artifact_inventory"] = {
        name: _artifact(path) for name, path in sorted(flattened.items())
    }
    draft["intent_digest"] = canonical_digest(draft, digest_field="intent_digest")
    validate_configured_controls_autostart_intent(draft)
    destination = Path(output_path).expanduser()
    payload = (json.dumps(draft, sort_keys=True, separators=(",", ":")) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
        destination.chmod(0o440)
    except FileExistsError:
        if destination.is_symlink() or destination.read_bytes() != payload:
            raise TaskEvaluationConfiguredControlsAutostartError(
                "configured_controls_autostart_intent_conflict"
            ) from None
    return draft


def configured_controls_autostart_registry_name(
    *, team_namespace: str, scene_id: str, task_id: str
) -> str:
    identity = canonical_digest(
        {
            "team_namespace": team_namespace,
            "scene_id": scene_id,
            "task_id": task_id,
        }
    ).removeprefix("sha256:")
    return f"{identity}.json"


def configured_controls_autostart_adoption_registry_name(
    *, team_namespace: str, scene_id: str, task_id: str, source_launch_id: str
) -> str:
    identity = canonical_digest(
        {
            "team_namespace": team_namespace,
            "scene_id": scene_id,
            "task_id": task_id,
            "source_launch_id": source_launch_id,
        }
    ).removeprefix("sha256:")
    return f"adoption-{identity}.json"


def stage_configured_controls_autostart_intent(
    *,
    source_path: str | Path,
    expected_production_commit: str,
    team_namespace: str,
    scene_id: str,
    task_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate a registry intent and copy its exact bytes into a launch set."""

    source = Path(source_path).expanduser()
    value = validate_configured_controls_autostart_intent(
        _read(source, blocker="configured_controls_autostart_intent_invalid")
    )
    if (
        value["expected_production_commit"] != expected_production_commit
        or value["team_namespace"] != team_namespace
        or value["scene_id"] != scene_id
        or value["task_id"] != task_id
    ):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_intent_identity_mismatch"
        )
    destination = Path(output_path).expanduser()
    payload = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
        destination.chmod(0o440)
    except FileExistsError:
        if destination.is_symlink() or destination.read_bytes() != payload:
            raise TaskEvaluationConfiguredControlsAutostartError(
                "configured_controls_autostart_intent_conflict"
            ) from None
    return {
        "status": "staged",
        "intent_path": str(destination.resolve()),
        "intent_digest": value["intent_digest"],
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
    }


def _configured_collision(
    *, revision: Mapping[str, Any], revision_path: Path, output_root: Path
) -> Path:
    bundle = revision_path.parent / "configured_scene_bundle.v1.zip"
    reference = revision["configured_scene_bundle"]
    if (
        bundle.is_symlink()
        or not bundle.is_file()
        or _sha256(bundle) != reference["digest"]
        or bundle.stat().st_size != reference["size_bytes"]
    ):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_bundle_invalid"
        )
    expected = revision["geometry"]["configured_collision"]
    try:
        with zipfile.ZipFile(bundle) as archive:
            matches = [
                row for row in archive.infolist()
                if PurePosixPath(row.filename).name.startswith("collision.")
            ]
            if len(matches) != 1 or matches[0].is_dir() or matches[0].flag_bits & 0x1:
                raise ValueError("member")
            payload = archive.read(matches[0])
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as exc:
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_bundle_invalid"
        ) from exc
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    if digest != expected["digest"] or len(payload) != expected["size_bytes"]:
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_collision_binding_invalid"
        )
    destination = output_root / ("configured_collision" + PurePosixPath(matches[0].filename).suffix)
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
        destination.chmod(0o440)
    except FileExistsError:
        if destination.is_symlink() or destination.read_bytes() != payload:
            raise TaskEvaluationConfiguredControlsAutostartError(
                "configured_controls_autostart_collision_conflict"
            ) from None
    return destination


def _profile_intent(
    run_root: Path, *, intent_path_override: str | Path | None = None
) -> tuple[dict[str, Any], Path]:
    profile = _read(
        run_root / "launch_profile.json",
        blocker="configured_controls_autostart_profile_invalid",
    )
    matches = [
        row for row in profile.get("immutable_inputs") or []
        if isinstance(row, Mapping)
        and row.get("name") == "configured_controls_autostart_intent"
    ]
    if intent_path_override is not None:
        if matches:
            raise TaskEvaluationConfiguredControlsAutostartError(
                "configured_controls_autostart_adoption_profile_conflict"
            )
        path = Path(intent_path_override).expanduser()
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise TaskEvaluationConfiguredControlsAutostartError(
                "configured_controls_autostart_intent_binding_invalid"
            )
        return profile, path
    if len(matches) != 1:
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_intent_missing"
        )
    path = Path(str(matches[0].get("path") or "")).expanduser()
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or _DIGEST.fullmatch(str(matches[0].get("digest") or "")) is None
        or _sha256(path) != matches[0]["digest"]
    ):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_intent_binding_invalid"
        )
    return profile, path


def _validate_configuration_adoption(
    *,
    adoption: Mapping[str, Any],
    source_launch_id: str,
    terminal: Mapping[str, Any],
    receipt: Mapping[str, Any],
    revision: Mapping[str, Any],
    publication: Mapping[str, Any],
    sync: Mapping[str, Any],
    zero: Mapping[str, Any],
) -> None:
    if (
        adoption.get("mode") != "explicit_terminal_adoption"
        or adoption.get("source_launch_id") != source_launch_id
        or adoption.get("source_launch_receipt_digest") != receipt.get("receipt_digest")
        or adoption.get("terminal_result_digest") != terminal.get("result_digest")
        or adoption.get("configured_scene_revision_digest") != revision.get("revision_digest")
        or adoption.get("publication_result_digest") != publication.get("result_digest")
        or adoption.get("webapp_sync_result_digest") != sync.get("sync_result_digest")
        or adoption.get("provider_zero_receipt_digest")
        != zero.get("provider_zero_receipt_digest")
    ):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_adoption_evidence_invalid"
        )


def _artifact_record_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    try:
        path = Path(str(value.get("path") or ""))
        return dict(value) == _artifact(path)
    except (OSError, TaskEvaluationConfiguredControlsAutostartError):
        return False


def _validate_result(
    value: Mapping[str, Any],
    *,
    expected_intent_digest: str,
    expected_scene_binding_digest: str,
    expected_task_binding_digest: str,
    expected_cpu_checkpoint_binding_digest: str,
) -> dict[str, Any]:
    result = json.loads(json.dumps(dict(value), allow_nan=False))
    openai_evidence = result.get("official_openai_cost_evidence")
    inference_usage = result.get("openai_inference_usage_packet")
    inference_sync = result.get("openai_inference_usage_webapp_sync")
    native_universe = result.get("native_construction_candidate_universe")
    native_universe_path = (
        Path(str(native_universe.get("path") or ""))
        if isinstance(native_universe, Mapping)
        else Path()
    )
    if (
        result.get("schema_version") != RESULT_SCHEMA_VERSION
        or result.get("status") != "agent_binding_accepted_plan_materialized"
        or not str(result.get("source_launch_id") or "")
        or result.get("intent_digest") != expected_intent_digest
        or result.get("scene_binding_digest")
        != expected_scene_binding_digest
        or result.get("task_binding_digest") != expected_task_binding_digest
        or result.get("cpu_placement_checkpoint_binding_digest")
        != expected_cpu_checkpoint_binding_digest
        or _DIGEST.fullmatch(
            str(result.get("configured_scene_revision_digest") or "")
        )
        is None
        or _DIGEST.fullmatch(str(result.get("trajectory_digest") or "")) is None
        or _DIGEST.fullmatch(
            str(result.get("candidate_inventory_digest") or "")
        )
        is None
        or not str(result.get("selected_candidate_id") or "")
        or _DIGEST.fullmatch(
            str(result.get("cpu_inventory_ranker_receipt_digest") or "")
        )
        is None
        or _DIGEST.fullmatch(
            str(result.get("placement_agent_receipt_digest") or "")
        )
        is None
        or result.get("placement_agent_model") != ROBOT_PLACEMENT_AGENT_MODEL
        or result.get("placement_agent_reasoning_effort")
        != ROBOT_PLACEMENT_AGENT_REASONING_EFFORT
        or result.get("placement_agent_selected_exact_inventory_member") is not True
        or result.get("placement_agent_visual_review_completed") is not True
        or not isinstance(openai_evidence, Mapping)
        or set(openai_evidence)
        != {
            "reservation",
            "completion",
            "exclusive_lock",
            "exclusive_lock_release",
            "inference_reservations",
        }
        or not all(_artifact_record_valid(row) for row in openai_evidence.values())
        or not isinstance(inference_usage, Mapping)
        or not _artifact_record_valid(inference_usage)
        or not isinstance(inference_sync, Mapping)
        or not isinstance(inference_sync.get("required"), bool)
        or (
            inference_sync.get("required") is True
            and inference_sync.get("status") != "succeeded"
        )
        or (
            inference_sync.get("required") is False
            and inference_sync.get("status") not in {"succeeded", "skipped"}
        )
        or not _artifact_record_valid(inference_sync.get("artifact"))
        or _DIGEST.fullmatch(str(inference_sync.get("packet_digest") or "")) is None
        or not 1 <= int(inference_sync.get("call_count") or 0) <= 8
        or not Path(str(result.get("base_pose_candidate_path") or "")).is_absolute()
        or not isinstance(native_universe, Mapping)
        or not Path(str(native_universe.get("path") or "")).is_absolute()
        or native_universe_path.is_symlink()
        or not native_universe_path.is_file()
        or _sha256(native_universe_path) != native_universe.get("file_sha256")
        or _DIGEST.fullmatch(str(native_universe.get("file_sha256") or ""))
        is None
        or _DIGEST.fullmatch(str(native_universe.get("inventory_digest") or ""))
        is None
        or not 1 <= int(native_universe.get("candidate_count") or 0) <= 64
        or not Path(str(result.get("plan_path") or "")).is_absolute()
        or _DIGEST.fullmatch(str(result.get("plan_digest") or "")) is None
        or result.get("cpu_position_ik_qualified") is not True
        or result.get(
            "native_orientation_collision_contact_camera_and_execution_required"
        )
        is not True
        or result.get("provider_mutation_performed") is not False
        or result.get("paid_execution_requested") is not True
        or result.get("result_digest")
        != canonical_digest(result, digest_field="result_digest")
    ):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_result_invalid"
        )
    return result


def _placement_checkpoint(
    *,
    root: Path,
    placement_runner: PlacementRunner,
    runner_kwargs: Mapping[str, Any],
    expected_scene_binding_digest: str,
    expected_task_binding_digest: str,
    expected_checkpoint_binding_digest: str | None = None,
    attempts_dir_name: str = "placement-attempts",
    checkpoint_file_name: str = "cpu-placement-checkpoint.v1.json",
    checkpoint_schema_version: str = _PLACEMENT_CHECKPOINT_SCHEMA_VERSION,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Run in a fresh attempt and publish one immutable completed checkpoint."""

    if (
        expected_checkpoint_binding_digest is not None
        and _DIGEST.fullmatch(expected_checkpoint_binding_digest) is None
    ):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_placement_checkpoint_invalid"
        )

    attempts_root = root / attempts_dir_name
    attempts_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    checkpoint_path = root / checkpoint_file_name

    def reopen() -> tuple[dict[str, Any], dict[str, Any], Path]:
        checkpoint = _read(
            checkpoint_path,
            blocker="configured_controls_autostart_placement_checkpoint_invalid",
        )
        if (
            checkpoint.get("schema_version") != checkpoint_schema_version
            or checkpoint.get("status") != "complete"
            or (
                expected_checkpoint_binding_digest is not None
                and checkpoint.get("checkpoint_binding_digest")
                != expected_checkpoint_binding_digest
            )
            or checkpoint.get("checkpoint_digest")
            != canonical_digest(checkpoint, digest_field="checkpoint_digest")
        ):
            raise TaskEvaluationConfiguredControlsAutostartError(
                "configured_controls_autostart_placement_checkpoint_invalid"
            )
        attempt = Path(str(checkpoint.get("attempt_root") or ""))
        try:
            attempt.relative_to(attempts_root.resolve())
        except (OSError, ValueError) as exc:
            raise TaskEvaluationConfiguredControlsAutostartError(
                "configured_controls_autostart_placement_checkpoint_invalid"
            ) from exc
        receipt_path = Path(str(checkpoint.get("receipt_path") or ""))
        inventory_path = Path(str(checkpoint.get("inventory_path") or ""))
        if (
            attempt.is_symlink()
            or not attempt.is_dir()
            or receipt_path.parent != attempt
            or inventory_path.parent != attempt
            or receipt_path.is_symlink()
            or inventory_path.is_symlink()
            or not receipt_path.is_file()
            or not inventory_path.is_file()
            or _sha256(receipt_path) != checkpoint.get("receipt_sha256")
            or _sha256(inventory_path) != checkpoint.get("inventory_sha256")
        ):
            raise TaskEvaluationConfiguredControlsAutostartError(
                "configured_controls_autostart_placement_checkpoint_invalid"
            )
        receipt = validate_robot_placement_receipt(
            _read(
                receipt_path,
                blocker="configured_controls_autostart_placement_receipt_invalid",
            ),
            expected_scene_binding_digest=expected_scene_binding_digest,
            expected_task_binding_digest=expected_task_binding_digest,
        )
        inventory = _read(
            inventory_path,
            blocker="configured_controls_autostart_inventory_missing",
        )
        if (
            inventory.get("candidate_inventory_digest")
            != receipt.get("candidate_inventory_digest")
            or inventory.get("checkpoint_digest")
            != canonical_digest(inventory, digest_field="checkpoint_digest")
        ):
            raise TaskEvaluationConfiguredControlsAutostartError(
                "configured_controls_autostart_placement_checkpoint_invalid"
            )
        return receipt, inventory, attempt

    if checkpoint_path.exists():
        return reopen()

    attempt_root: Path | None = None
    for index in range(1_000):
        candidate = attempts_root / f"attempt_{index:03d}"
        try:
            candidate.mkdir(mode=0o750)
        except FileExistsError:
            continue
        attempt_root = candidate.resolve()
        break
    if attempt_root is None:
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_placement_attempts_exhausted"
        )
    receipt = dict(
        placement_runner(output_dir=attempt_root, **dict(runner_kwargs))
    )
    receipt_path = (
        attempt_root / "task_evaluation_robot_placement_receipt.v1.json"
    )
    inventory_path = (
        attempt_root
        / "task_evaluation_robot_placement_candidate_inventory.v1.json"
    )
    validated_receipt = validate_robot_placement_receipt(
        receipt,
        expected_scene_binding_digest=expected_scene_binding_digest,
        expected_task_binding_digest=expected_task_binding_digest,
    )
    inventory = _read(
        inventory_path, blocker="configured_controls_autostart_inventory_missing"
    )
    if (
        receipt_path.is_symlink()
        or not receipt_path.is_file()
        or inventory.get("candidate_inventory_digest")
        != validated_receipt.get("candidate_inventory_digest")
        or inventory.get("checkpoint_digest")
        != canonical_digest(inventory, digest_field="checkpoint_digest")
    ):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_placement_checkpoint_invalid"
        )
    checkpoint = {
        "schema_version": checkpoint_schema_version,
        "status": "complete",
        "attempt_root": str(attempt_root),
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
        "inventory_path": str(inventory_path),
        "inventory_sha256": _sha256(inventory_path),
        "checkpoint_digest": "",
    }
    if expected_checkpoint_binding_digest is not None:
        checkpoint["checkpoint_binding_digest"] = (
            expected_checkpoint_binding_digest
        )
    checkpoint["checkpoint_digest"] = canonical_digest(
        checkpoint, digest_field="checkpoint_digest"
    )
    payload = (
        json.dumps(checkpoint, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    try:
        with checkpoint_path.open("xb") as stream:
            stream.write(payload)
        checkpoint_path.chmod(0o440)
    except FileExistsError:
        return reopen()
    return validated_receipt, inventory, attempt_root


def _cpu_placement_checkpoint_binding_digest(
    *,
    intent_digest: str,
    scene_binding_digest: str,
    task_binding_digest: str,
) -> str:
    if any(
        _DIGEST.fullmatch(value) is None
        for value in (intent_digest, scene_binding_digest, task_binding_digest)
    ):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_placement_checkpoint_binding_invalid"
        )
    return canonical_digest(
        {
            "schema_version": _CPU_PLACEMENT_BINDING_SCHEMA_VERSION,
            "intent_digest": intent_digest,
            "scene_binding_digest": scene_binding_digest,
            "task_binding_digest": task_binding_digest,
        }
    )


def _cpu_placement_checkpoint_root(
    *, root: Path, checkpoint_binding_digest: str
) -> Path:
    if _DIGEST.fullmatch(checkpoint_binding_digest) is None:
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_placement_checkpoint_binding_invalid"
        )
    return (
        root
        / "cpu-placement-checkpoints"
        / checkpoint_binding_digest.removeprefix("sha256:")
    )


def _autostart_result_path(*, root: Path, intent_digest: str) -> Path:
    """Bind the autostart result to the intent that produced it.

    The result is validated against the intent digest on reopen, so a shared
    filename does not silently reuse a stale result -- it raises, and keeps
    raising for every successor intent. Give each intent its own destination so
    a redeploy derives fresh while its predecessor survives as evidence.
    """

    if _DIGEST.fullmatch(intent_digest) is None:
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_result_binding_invalid"
        )
    token = intent_digest.removeprefix("sha256:")[:16]
    return root / f"{RESULT_SCHEMA_VERSION}-{token}.json"


def _validated_agent_openai_evidence(
    *,
    cost_root: Path,
    agent_attempt_root: Path,
    intent: Mapping[str, Any],
    receipt: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    reservation_path = cost_root / "openai_official_cost_run_reservation.v1.json"
    completion_path = cost_root / "openai_official_cost_run_completion.v1.json"
    lock_path = cost_root / "openai_scope_lock_acquired.v1.json"
    release_path = cost_root / "openai_scope_lock_released.v1.json"
    inference_path = agent_attempt_root / "inference_reservations" / "manifest.json"
    reservation = _read(
        reservation_path,
        blocker="configured_controls_autostart_openai_evidence_invalid",
    )
    completion = _read(
        completion_path,
        blocker="configured_controls_autostart_openai_evidence_invalid",
    )
    lock = _read(
        lock_path,
        blocker="configured_controls_autostart_openai_evidence_invalid",
    )
    release = _read(
        release_path,
        blocker="configured_controls_autostart_openai_evidence_invalid",
    )
    inference = _read(
        inference_path,
        blocker="configured_controls_autostart_openai_evidence_invalid",
    )
    placement = intent["placement"]
    if (
        reservation.get("schema_version") != RUN_RESERVATION_SCHEMA_VERSION
        or reservation.get("status") != "reserved_before_openai_call"
        or reservation.get("request_digest")
        != canonical_digest(
            {
                "intent_digest": intent["intent_digest"],
                "candidate_inventory_checkpoint_digest": inventory[
                    "checkpoint_digest"
                ],
                "scene_binding_digest": receipt["scene_binding_digest"],
                "task_binding_digest": receipt["task_binding_digest"],
            }
        )
        or reservation.get("candidate_digest")
        != inventory.get("candidate_inventory_digest")
        or reservation.get("authorization_receipt_digest")
        != intent.get("intent_digest")
        or float(reservation.get("maximum_cost_usd", -1.0))
        != float(placement["max_inference_cost_usd"])
        or reservation.get("reservation_receipt_digest")
        != canonical_digest(reservation, digest_field="reservation_receipt_digest")
        or completion.get("schema_version") != RUN_COMPLETION_SCHEMA_VERSION
        or completion.get("status") != "official_cost_reporting_pending"
        or completion.get("reservation_receipt_digest")
        != reservation.get("reservation_receipt_digest")
        or completion.get("provider_call_performed") is not True
        or completion.get("runtime_result_digest") != receipt.get("receipt_digest")
        or completion.get("completion_receipt_digest")
        != canonical_digest(completion, digest_field="completion_receipt_digest")
        or lock.get("status") != "acquired"
        or lock.get("all_vast_launch_slots_held") is not True
        or lock.get("credential_role") != VISUAL_REVIEW_CREDENTIAL_ROLE
        or lock.get("lock_receipt_digest")
        != canonical_digest(lock, digest_field="lock_receipt_digest")
        or release.get("status") != "released"
        or release.get("all_vast_launch_slots_released") is not True
        or release.get("acquisition_receipt_digest")
        != lock.get("lock_receipt_digest")
        or release.get("release_receipt_digest")
        != canonical_digest(release, digest_field="release_receipt_digest")
        or inference.get("run_id") != receipt.get("run_id")
        or not 1 <= int(inference.get("reservation_count", 0)) <= 4
        or inference.get("in_flight_unknown_count") != 0
        or float(inference.get("reserved_max_cost_usd", -1.0))
        > float(placement["max_inference_cost_usd"])
        or inference.get("inference_reservation_manifest_digest")
        != canonical_digest(
            inference, digest_field="inference_reservation_manifest_digest"
        )
    ):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_openai_evidence_invalid"
        )
    return {
        "reservation": _artifact(reservation_path),
        "completion": _artifact(completion_path),
        "exclusive_lock": _artifact(lock_path),
        "exclusive_lock_release": _artifact(release_path),
        "inference_reservations": _artifact(inference_path),
    }


def materialize_configured_controls_autostart(
    *,
    source_launch_id: str,
    launch_state_root: str | Path,
    progression_root: str | Path,
    plan_root: str | Path,
    placement_runner: PlacementRunner = run_robot_placement_cli,
    agent_placement_runner: PlacementRunner | None = None,
    readiness_materializer: ReadinessMaterializer = materialize_robot_placement_readiness_candidate,
    plan_materializer: PlanMaterializer = materialize_configured_controls_plan,
    intent_path_override: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    openai_gate_builder: Callable[..., Any] = (
        configured_controls_robot_placement_openai_gate
    ),
    openai_scope_lock: Callable[..., Any] = exclusive_visual_review_cost_scope,
    require_inference_usage_webapp_sync: bool = True,
) -> dict[str, Any]:
    """Produce an exact plan from one completed, Website-synced configuration."""

    from .task_evaluation_configured_controls_progression_worker import _validate_source

    launch_root = Path(launch_state_root).expanduser()
    run_root = launch_root / source_launch_id
    terminal, receipt, _ = _validate_source(run_root)
    profile, intent_path = _profile_intent(
        run_root, intent_path_override=intent_path_override
    )
    intent = validate_configured_controls_autostart_intent(
        _read(intent_path, blocker="configured_controls_autostart_intent_invalid")
    )
    task_run = profile.get("task_evaluation_run")
    if (
        receipt.get("source_commit") != intent["configuration_source_commit"]
        or not isinstance(task_run, Mapping)
        or task_run.get("run_mode") != "scene_configuration"
        or task_run.get("team_namespace") != intent["team_namespace"]
        or task_run.get("scene_id") != intent["scene_id"]
        or task_run.get("task_id") != intent["task_id"]
    ):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_source_binding_invalid"
        )
    revision_path = Path(str(terminal.get("configured_scene_revision_path") or ""))
    revision = validate_configured_scene_revision(
        _read(revision_path, blocker="configured_controls_autostart_revision_invalid")
    )
    if (
        revision["source_commit"] != intent["configuration_source_commit"]
        or revision["team_namespace"] != intent["team_namespace"]
        or revision["scene_identity"]["id"] != intent["scene_id"]
        or revision["task_template"]["identity"]["id"] != intent["task_id"]
    ):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_revision_binding_invalid"
        )
    adoption = intent["configuration_adoption"]
    if intent_path_override is not None:
        sync = _read(
            run_root / "webapp_sync_succeeded.json",
            blocker="configured_controls_autostart_adoption_evidence_invalid",
        )
        zero = _read(
            run_root / "post_teardown_provider_zero_receipt.json",
            blocker="configured_controls_autostart_adoption_evidence_invalid",
        )
        publication = _read(
            Path(str(terminal.get("publication_result_path") or "")),
            blocker="configured_controls_autostart_adoption_evidence_invalid",
        )
        _validate_configuration_adoption(
            adoption=adoption,
            source_launch_id=source_launch_id,
            terminal=terminal,
            receipt=receipt,
            revision=revision,
            publication=publication,
            sync=sync,
            zero=zero,
        )
    elif adoption.get("mode") != "same_commit_automatic":
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_adoption_mode_invalid"
        )
    root = Path(progression_root).expanduser() / source_launch_id / "cpu-robot-binding"
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    paths = intent["paths"]
    trajectory = placement_trajectory_from_native_plan(
        _read(Path(paths["native_trajectory_plan_path"]), blocker="configured_controls_autostart_trajectory_invalid")
    )
    scene_binding = {
        "schema_version": "task_evaluation_robot_placement_scene_binding.v1",
        "scene_identity": revision["scene_identity"],
        "configured_scene_revision_digest": revision["revision_digest"],
        "robot_mount_interface_digest": revision["registration"]["robot_mount_interface"]["digest"],
        "workspace_clearance_digest": revision["registration"]["workspace_clearance"]["digest"],
        "collision_asset_digest": revision["geometry"]["configured_collision"]["digest"],
    }
    task_binding = {
        "schema_version": "task_evaluation_robot_placement_task_binding.v1",
        "task_identity": revision["task_template"]["identity"],
        "task_definition_digest": revision["task_template"]["definition"]["digest"],
        "robot_id": "franka_panda",
        "target_position_world_m": intent["target_position_world_m"],
        "trajectory_digest": trajectory["trajectory_digest"],
    }
    scene_binding_digest = canonical_digest(scene_binding)
    task_binding_digest = canonical_digest(task_binding)
    cpu_checkpoint_binding_digest = _cpu_placement_checkpoint_binding_digest(
        intent_digest=str(intent["intent_digest"]),
        scene_binding_digest=scene_binding_digest,
        task_binding_digest=task_binding_digest,
    )
    result_path = _autostart_result_path(
        root=root, intent_digest=str(intent["intent_digest"])
    )
    result_validation_kwargs = {
        "expected_intent_digest": str(intent["intent_digest"]),
        "expected_scene_binding_digest": scene_binding_digest,
        "expected_task_binding_digest": task_binding_digest,
        "expected_cpu_checkpoint_binding_digest": (
            cpu_checkpoint_binding_digest
        ),
    }
    if result_path.is_file():
        return _validate_result(
            _read(
                result_path,
                blocker="configured_controls_autostart_result_invalid",
            ),
            **result_validation_kwargs,
        )
    collision = _configured_collision(
        revision=revision, revision_path=revision_path, output_root=root
    )
    placement = intent["placement"]
    cpu_placement_receipt, inventory, _placement_root = _placement_checkpoint(
        root=_cpu_placement_checkpoint_root(
            root=root,
            checkpoint_binding_digest=cpu_checkpoint_binding_digest,
        ),
        placement_runner=placement_runner,
        expected_scene_binding_digest=scene_binding_digest,
        expected_task_binding_digest=task_binding_digest,
        expected_checkpoint_binding_digest=cpu_checkpoint_binding_digest,
        checkpoint_file_name="cpu-placement-checkpoint.v2.json",
        checkpoint_schema_version=(
            _BOUND_CPU_PLACEMENT_CHECKPOINT_SCHEMA_VERSION
        ),
        runner_kwargs={
            "run_id": f"{terminal['run_id']}-cpu-placement",
            "scene_collision_usd": collision,
            "robot_asset_usd": Path(paths["robot_asset_usd_path"]),
            "target_position_world_m": intent["target_position_world_m"],
            "scene_binding": scene_binding,
            "task_binding": task_binding,
            "overview_image_paths": [
                Path(item) for item in paths["overview_image_paths"]
            ],
            "max_rounds": placement["max_rounds"],
            "candidate_inventory_cap": placement["candidate_inventory_cap"],
            "max_input_tokens": placement["max_input_tokens"],
            "max_inference_cost_usd": placement["max_inference_cost_usd"],
            "expected_proposal_reuse_probability": placement[
                "expected_proposal_reuse_probability"
            ],
            "expected_visual_review_reuse_probability": placement[
                "expected_visual_review_reuse_probability"
            ],
            "expected_proposal_reuse_count": placement[
                "expected_proposal_reuse_count"
            ],
            "expected_visual_review_reuse_count": placement[
                "expected_visual_review_reuse_count"
            ],
            "allow_live_invocation": False,
            "tracing_disabled": True,
            "robot_id": "franka_panda",
            "task_trajectory": trajectory,
            "deterministic_selection": True,
        },
    )
    selected_agent_runner = agent_placement_runner or placement_runner
    selected_environment = dict(os.environ if environment is None else environment)
    agent_request_digest = canonical_digest(
        {
            "intent_digest": intent["intent_digest"],
            "candidate_inventory_checkpoint_digest": inventory["checkpoint_digest"],
            "scene_binding_digest": scene_binding_digest,
            "task_binding_digest": task_binding_digest,
        }
    )
    intent_token = intent["intent_digest"].removeprefix("sha256:")[:16]

    def reviewed_placement_runner(*, output_dir: Path, **runner_kwargs: Any):
        cost_root = (
            root
            / "agent-official-openai-cost"
            / output_dir.parent.name
            / output_dir.name
        )
        with openai_scope_lock(
            environment=selected_environment,
            output_root=cost_root,
        ):
            cost_gate = openai_gate_builder(
                environment=selected_environment,
                placement_authority=placement["official_cost_authority"],
                run_id=str(runner_kwargs["run_id"]),
                request_digest=agent_request_digest,
                candidate_digest=str(inventory["candidate_inventory_digest"]),
                authorization_receipt_digest=str(intent["intent_digest"]),
                output_root=cost_root,
            )
            cost_gate.reserve()
            try:
                agent_receipt = dict(
                    selected_agent_runner(
                        output_dir=output_dir,
                        record_inference_reservations=True,
                        **runner_kwargs,
                    )
                )
            except Exception as exc:
                cost_gate.complete(
                    provider_call_performed=True,
                    runtime_result_digest=None,
                    runtime_exception_type=type(exc).__name__,
                )
                raise
            cost_gate.complete(
                provider_call_performed=True,
                runtime_result_digest=str(agent_receipt.get("receipt_digest") or ""),
                runtime_exception_type=None,
            )
            return agent_receipt

    placement_receipt, agent_inventory, agent_placement_root = _placement_checkpoint(
        root=root,
        placement_runner=reviewed_placement_runner,
        expected_scene_binding_digest=scene_binding_digest,
        expected_task_binding_digest=task_binding_digest,
        attempts_dir_name=f"agent-placement-attempts-{intent_token}",
        checkpoint_file_name=f"agent-placement-checkpoint-{intent_token}.v1.json",
        checkpoint_schema_version=_AGENT_PLACEMENT_CHECKPOINT_SCHEMA_VERSION,
        runner_kwargs={
            "run_id": f"{terminal['run_id']}-agent-placement",
            "scene_collision_usd": collision,
            "robot_asset_usd": Path(paths["robot_asset_usd_path"]),
            "target_position_world_m": intent["target_position_world_m"],
            "scene_binding": scene_binding,
            "task_binding": task_binding,
            "overview_image_paths": [
                Path(item) for item in paths["overview_image_paths"]
            ],
            "max_rounds": placement["max_rounds"],
            "candidate_inventory_cap": placement["candidate_inventory_cap"],
            "max_input_tokens": placement["max_input_tokens"],
            "max_inference_cost_usd": placement["max_inference_cost_usd"],
            "expected_proposal_reuse_probability": placement[
                "expected_proposal_reuse_probability"
            ],
            "expected_visual_review_reuse_probability": placement[
                "expected_visual_review_reuse_probability"
            ],
            "expected_proposal_reuse_count": placement[
                "expected_proposal_reuse_count"
            ],
            "expected_visual_review_reuse_count": placement[
                "expected_visual_review_reuse_count"
            ],
            "allow_live_invocation": True,
            "tracing_disabled": True,
            "robot_id": "franka_panda",
            "task_trajectory": trajectory,
            "candidate_inventory_checkpoint": inventory,
            "deterministic_selection": False,
        },
    )
    if (
        agent_inventory.get("checkpoint_digest") != inventory.get("checkpoint_digest")
        or placement_receipt.get("candidate_inventory_digest")
        != cpu_placement_receipt.get("candidate_inventory_digest")
    ):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_agent_inventory_binding_invalid"
        )
    openai_evidence = _validated_agent_openai_evidence(
        cost_root=(
            root
            / "agent-official-openai-cost"
            / agent_placement_root.parent.name
            / agent_placement_root.name
        ),
        agent_attempt_root=agent_placement_root,
        intent=intent,
        receipt=placement_receipt,
        inventory=inventory,
    )
    inference_usage_packet = build_placement_inference_usage_packet(
        placement_receipt=placement_receipt,
        packet_run_id=str(
            receipt.get("run_id") or terminal.get("run_id") or source_launch_id
        ),
        launch_id=source_launch_id,
        source_commit=str(intent["expected_production_commit"]),
    )
    inference_usage_packet_path = (
        agent_placement_root / "openai_inference_usage_packet.v1.json"
    )
    inference_usage_packet_path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    inference_usage_payload = (
        json.dumps(
            inference_usage_packet,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    with inference_usage_packet_path.open("xb") as stream:
        stream.write(inference_usage_payload)
    inference_usage_packet_path.chmod(0o440)
    inference_usage_sync = sync_inference_usage_to_webapp(
        packet=inference_usage_packet
    )
    if (
        require_inference_usage_webapp_sync
        and inference_usage_sync.get("status") != "succeeded"
    ):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_inference_usage_sync_required"
        )
    inference_usage_sync_path = (
        agent_placement_root / "openai_inference_usage_webapp_sync.v1.json"
    )
    with inference_usage_sync_path.open("xb") as stream:
        stream.write(
            (
                json.dumps(
                    inference_usage_sync,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        )
    inference_usage_sync_path.chmod(0o440)
    native_universe_path, native_universe = (
        _materialize_native_feedback_candidate_universe(
            root=root,
            run_id=f"{terminal['run_id']}-native-construction-feedback",
            inventory=inventory,
            trajectory=trajectory,
            camera_template_path=Path(paths["cameras_path"]),
            source_commit=intent["expected_production_commit"],
            maximum_candidates=min(placement["candidate_inventory_cap"] * 4, 64),
        )
    )
    native_universe_reference = {
        "path": str(native_universe_path.resolve()),
        "file_sha256": _sha256(native_universe_path),
        "inventory_digest": native_universe["inventory_digest"],
        "candidate_count": len(native_universe["candidates"]),
    }
    camera_candidates_path = _materialize_placement_aware_cameras(
        root=root,
        camera_template_path=Path(paths["cameras_path"]),
        accepted_pose=placement_receipt["accepted_pose"],
        selected_candidate_id=str(placement_receipt["accepted_candidate_id"]),
        trajectory=trajectory,
        source_commit=intent["expected_production_commit"],
    )
    # Bind the readiness candidate to the intent that produced it. A shared
    # filename silently reuses a previous intent's accepted pose.
    base_path = root / (
        f"task_evaluation_robot_placement_readiness_candidate-{intent_token}.v1.json"
    )
    if not base_path.is_file():
        readiness_materializer(
            configured_revision=revision,
            scene_binding=scene_binding,
            task_binding=task_binding,
            placement_receipt=placement_receipt,
            candidate_inventory=inventory,
            output_path=base_path,
            native_construction_candidate_universe_reference=(
                native_universe_reference
            ),
        )
    bindings = {
        "robot_mount_interface_path": paths["robot_mount_interface_path"],
        "scene_camera_calibration_path": paths["scene_camera_calibration_path"],
        "base_pose_candidate_path": str(base_path),
        "cameras_path": str(camera_candidates_path),
        "runtime_binding_path": paths["runtime_binding_path"],
        "phases": intent["phases"],
    }
    plan = dict(plan_materializer(
        source_launch_id=source_launch_id,
        launch_state_root=launch_root,
        expected_production_commit=intent["expected_production_commit"],
        submitted_by=intent["submitted_by"],
        bindings=bindings,
        plan_root=plan_root,
        profile_dir=intent["profile_dir"],
    ))
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "agent_binding_accepted_plan_materialized",
        "source_launch_id": source_launch_id,
        "intent_digest": intent["intent_digest"],
        "scene_binding_digest": scene_binding_digest,
        "task_binding_digest": task_binding_digest,
        "cpu_placement_checkpoint_binding_digest": (
            cpu_checkpoint_binding_digest
        ),
        "configured_scene_revision_digest": revision["revision_digest"],
        "trajectory_digest": trajectory["trajectory_digest"],
        "candidate_inventory_digest": placement_receipt["candidate_inventory_digest"],
        "selected_candidate_id": placement_receipt["accepted_candidate_id"],
        "cpu_inventory_ranker_receipt_digest": cpu_placement_receipt[
            "receipt_digest"
        ],
        "placement_agent_receipt_digest": placement_receipt["receipt_digest"],
        "placement_agent_model": ROBOT_PLACEMENT_AGENT_MODEL,
        "placement_agent_reasoning_effort": (
            ROBOT_PLACEMENT_AGENT_REASONING_EFFORT
        ),
        "placement_agent_selected_exact_inventory_member": True,
        "placement_agent_visual_review_completed": True,
        "official_openai_cost_evidence": openai_evidence,
        "openai_inference_usage_packet": _artifact(inference_usage_packet_path),
        "openai_inference_usage_webapp_sync": {
            "artifact": _artifact(inference_usage_sync_path),
            "status": inference_usage_sync["status"],
            "required": require_inference_usage_webapp_sync,
            "reason": inference_usage_sync.get("reason"),
            "packet_digest": inference_usage_packet["packet_digest"],
            "call_count": len(inference_usage_packet["calls"]),
        },
        "base_pose_candidate_path": str(base_path),
        "native_construction_candidate_universe": native_universe_reference,
        "plan_path": plan["plan_path"],
        "plan_digest": plan["plan_digest"],
        "cpu_position_ik_qualified": True,
        "native_orientation_collision_contact_camera_and_execution_required": True,
        "provider_mutation_performed": False,
        "paid_execution_requested": True,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    _validate_result(result, **result_validation_kwargs)
    payload = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with result_path.open("xb") as stream:
        stream.write(payload)
    result_path.chmod(0o440)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent", required=True)
    parser.add_argument("--expected-production-commit", required=True)
    parser.add_argument("--team-namespace", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = stage_configured_controls_autostart_intent(
            source_path=args.intent,
            expected_production_commit=args.expected_production_commit,
            team_namespace=args.team_namespace,
            scene_id=args.scene_id,
            task_id=args.task_id,
            output_path=args.output,
        )
    except (OSError, ValueError, TaskEvaluationConfiguredControlsAutostartError) as exc:
        print(json.dumps({"status": "blocked", "blockers": [str(exc)]}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "INTENT_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "TaskEvaluationConfiguredControlsAutostartError",
    "materialize_configured_controls_autostart",
    "materialize_configured_controls_autostart_intent",
    "configured_controls_autostart_adoption_registry_name",
    "configured_controls_autostart_registry_name",
    "stage_configured_controls_autostart_intent",
    "validate_configured_controls_autostart_intent",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
