"""Materialize the CPU-first configured-scene controls continuation.

The scene-configuration envelope has always declared that automatic progression
is required, but the production timer previously consumed only a hand-written
plan.  This module turns one immutable launch-profile input into that plan after
the configured revision is published.  It performs deterministic
geometry/trajectory placement on CPU, then binds the exact accepted inventory
member.  It never invokes a model, allocates a provider resource, or submits a
launch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_configured_controls_plan import (
    materialize_configured_controls_plan,
)
from .task_evaluation_configured_scene_revision import (
    validate_configured_scene_revision,
)
from .task_evaluation_robot_placement_agent_cli import run_robot_placement_cli
from .task_evaluation_robot_placement_agent import validate_robot_placement_receipt
from .task_evaluation_robot_placement_readiness_candidate import (
    materialize_robot_placement_readiness_candidate,
)
from .task_evaluation_robot_placement_trajectory import (
    placement_trajectory_from_native_plan,
    validate_robot_placement_trajectory,
)
from .task_evaluation_shared_mutation_window import (
    TaskEvaluationSharedMutationWindowError,
    validate_shared_mutation_window_template,
)


INTENT_SCHEMA_VERSION = "task_evaluation_configured_controls_autostart_intent.v1"
RESULT_SCHEMA_VERSION = "task_evaluation_configured_controls_autostart.v1"
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
    destination = root / "placement-aware-camera-candidates.v1.json"
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
        }
        or placement.get("robot_id") != "franka_panda"
        or not 1 <= int(placement.get("max_rounds", 0)) <= 8
        or not 1 <= int(placement.get("candidate_inventory_cap", 0)) <= 128
        or not 1 <= int(placement.get("max_input_tokens", 0)) <= 1_000_000
        or float(placement.get("max_inference_cost_usd", -1.0)) != 0.0
        or not Path(str(intent.get("profile_dir") or "")).is_absolute()
        or intent.get("provider_mutation_performed") is not False
        or intent.get("paid_execution_requested") is not False
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
    max_inference_cost_usd: float = 0.0,
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
        },
        "profile_dir": str(Path(profile_dir).expanduser()),
        "paths": json.loads(json.dumps(dict(paths))),
        "phases": json.loads(json.dumps(dict(phases))),
        "artifact_inventory": {},
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
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


def _validate_result(value: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(dict(value), allow_nan=False))
    if (
        result.get("schema_version") != RESULT_SCHEMA_VERSION
        or result.get("status") != "cpu_binding_accepted_plan_materialized"
        or not str(result.get("source_launch_id") or "")
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
        or not Path(str(result.get("base_pose_candidate_path") or "")).is_absolute()
        or not Path(str(result.get("plan_path") or "")).is_absolute()
        or _DIGEST.fullmatch(str(result.get("plan_digest") or "")) is None
        or result.get("cpu_position_ik_qualified") is not True
        or result.get(
            "native_orientation_collision_contact_camera_and_execution_required"
        )
        is not True
        or result.get("provider_mutation_performed") is not False
        or result.get("paid_execution_requested") is not False
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
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Run in a fresh attempt and publish one immutable completed checkpoint."""

    attempts_root = root / "placement-attempts"
    attempts_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    checkpoint_path = root / "cpu-placement-checkpoint.v1.json"

    def reopen() -> tuple[dict[str, Any], dict[str, Any], Path]:
        checkpoint = _read(
            checkpoint_path,
            blocker="configured_controls_autostart_placement_checkpoint_invalid",
        )
        if (
            checkpoint.get("schema_version") != _PLACEMENT_CHECKPOINT_SCHEMA_VERSION
            or checkpoint.get("status") != "complete"
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
        "schema_version": _PLACEMENT_CHECKPOINT_SCHEMA_VERSION,
        "status": "complete",
        "attempt_root": str(attempt_root),
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
        "inventory_path": str(inventory_path),
        "inventory_sha256": _sha256(inventory_path),
        "checkpoint_digest": "",
    }
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


def materialize_configured_controls_autostart(
    *,
    source_launch_id: str,
    launch_state_root: str | Path,
    progression_root: str | Path,
    plan_root: str | Path,
    placement_runner: PlacementRunner = run_robot_placement_cli,
    readiness_materializer: ReadinessMaterializer = materialize_robot_placement_readiness_candidate,
    plan_materializer: PlanMaterializer = materialize_configured_controls_plan,
    intent_path_override: str | Path | None = None,
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
    result_path = root / f"{RESULT_SCHEMA_VERSION}.json"
    if result_path.is_file():
        return _validate_result(
            _read(result_path, blocker="configured_controls_autostart_result_invalid")
        )
    collision = _configured_collision(
        revision=revision, revision_path=revision_path, output_root=root
    )
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
    placement = intent["placement"]
    placement_receipt, inventory, _placement_root = _placement_checkpoint(
        root=root,
        placement_runner=placement_runner,
        expected_scene_binding_digest=canonical_digest(scene_binding),
        expected_task_binding_digest=canonical_digest(task_binding),
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
            "allow_live_invocation": False,
            "tracing_disabled": True,
            "robot_id": "franka_panda",
            "task_trajectory": trajectory,
            "deterministic_selection": True,
        },
    )
    camera_candidates_path = _materialize_placement_aware_cameras(
        root=root,
        camera_template_path=Path(paths["cameras_path"]),
        accepted_pose=placement_receipt["accepted_pose"],
        selected_candidate_id=str(placement_receipt["accepted_candidate_id"]),
        trajectory=trajectory,
        source_commit=intent["expected_production_commit"],
    )
    base_path = root / "task_evaluation_robot_placement_readiness_candidate.v1.json"
    if not base_path.is_file():
        readiness_materializer(
            configured_revision=revision,
            scene_binding=scene_binding,
            task_binding=task_binding,
            placement_receipt=placement_receipt,
            candidate_inventory=inventory,
            output_path=base_path,
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
        "status": "cpu_binding_accepted_plan_materialized",
        "source_launch_id": source_launch_id,
        "configured_scene_revision_digest": revision["revision_digest"],
        "trajectory_digest": trajectory["trajectory_digest"],
        "candidate_inventory_digest": placement_receipt["candidate_inventory_digest"],
        "selected_candidate_id": placement_receipt["accepted_candidate_id"],
        "base_pose_candidate_path": str(base_path),
        "plan_path": plan["plan_path"],
        "plan_digest": plan["plan_digest"],
        "cpu_position_ik_qualified": True,
        "native_orientation_collision_contact_camera_and_execution_required": True,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    _validate_result(result)
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
