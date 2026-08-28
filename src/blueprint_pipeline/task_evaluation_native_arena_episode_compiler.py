"""Closed production compiler for configured-scene episode evaluations.

Robot teams submit independent, immutable scene, robot, controller, sensor, and
runtime references through the Website contract.  This module joins those
verified bytes into the legacy native-Arena packet only inside Blueprint's
production preparation service.  It never allocates a provider and it never
accepts a customer-built Arena packet.
"""

from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_packet import materialize_native_task_arena_packet
from .task_evaluation_configured_scene_revision import (
    validate_configured_scene_revision,
)
from .task_evaluation_native_arena_preparation_adapter import (
    build_task_evaluation_adapter_bundle,
    materialize_native_arena_adapter,
)
from .task_evaluation_rigid_relocation_native_adapter import (
    adapt_rigid_relocation_task_template,
)


OUTPUT_SCHEMA_VERSION = "task_evaluation_episode_compiler_output.v1"


class TaskEvaluationNativeArenaEpisodeCompilerError(RuntimeError):
    """Verified robot-team inputs could not be compiled fail-closed."""


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _reference_path(
    references: Mapping[str, Mapping[str, Any]], contract_path: str
) -> Path:
    row = references.get(contract_path)
    unresolved_path = Path(
        str((row or {}).get("materialized_path") or "")
    ).expanduser()
    path = unresolved_path.resolve()
    if (
        row is None
        or unresolved_path.is_symlink()
        or not path.is_file()
        or _sha256_and_size(path) != (row.get("digest"), row.get("size_bytes"))
        or row.get("full_byte_service_account_readback_passed") is not True
    ):
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            f"episode_compiler_reference_invalid:{contract_path}"
        )
    return path


def _json_reference(
    references: Mapping[str, Mapping[str, Any]],
    contract_path: str,
    schema_version: str,
) -> dict[str, Any]:
    path = _reference_path(references, contract_path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            f"episode_compiler_reference_json_invalid:{contract_path}"
        ) from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != schema_version:
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            f"episode_compiler_reference_contract_invalid:{contract_path}"
        )
    return dict(value)


def _safe_member(value: Any) -> PurePosixPath:
    member = PurePosixPath(str(value or ""))
    if (
        member.is_absolute()
        or member.as_posix() in {"", ".", ".."}
        or ".." in member.parts
    ):
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_configured_bundle_member_invalid"
        )
    return member


def _extract_configured_assets(
    bundle_path: Path, *, output_root: Path
) -> dict[str, Path]:
    destination = output_root / "configured-scene"
    destination.mkdir(mode=0o750)
    try:
        archive = zipfile.ZipFile(bundle_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_configured_bundle_invalid"
        ) from exc
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        manifest_name = "configured_scene_bundle_candidate.v1.json"
        if len(names) != len(set(names)) or manifest_name not in names:
            raise TaskEvaluationNativeArenaEpisodeCompilerError(
                "episode_compiler_configured_bundle_invalid"
            )
        for info in infos:
            mode = info.external_attr >> 16
            _safe_member(info.filename)
            if info.is_dir() or info.flag_bits & 0x1 or stat.S_ISLNK(mode):
                raise TaskEvaluationNativeArenaEpisodeCompilerError(
                    "episode_compiler_configured_bundle_member_invalid"
                )
        try:
            manifest = json.loads(archive.read(manifest_name).decode("utf-8"))
        except (KeyError, UnicodeError, json.JSONDecodeError) as exc:
            raise TaskEvaluationNativeArenaEpisodeCompilerError(
                "episode_compiler_configured_bundle_manifest_invalid"
            ) from exc
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("schema_version")
            != "task_evaluation_configured_scene_bundle_candidate.v1"
            or manifest.get("status")
            != "assembled_pending_control_plane_publication"
            or manifest.get("robot_neutral") is not True
            or manifest.get("robot_specific_base_registration_included") is not False
            or manifest.get("manifest_digest")
            != canonical_digest(manifest, digest_field="manifest_digest")
        ):
            raise TaskEvaluationNativeArenaEpisodeCompilerError(
                "episode_compiler_configured_bundle_manifest_invalid"
            )
        rows = manifest.get("assets")
        if not isinstance(rows, list) or {row.get("role") for row in rows if isinstance(row, Mapping)} != {
            "appearance",
            "collision",
            "replacement",
        }:
            raise TaskEvaluationNativeArenaEpisodeCompilerError(
                "episode_compiler_configured_bundle_asset_set_invalid"
            )
        extracted: dict[str, Path] = {}
        expected_names = {manifest_name}
        for row in rows:
            if not isinstance(row, Mapping):
                raise TaskEvaluationNativeArenaEpisodeCompilerError(
                    "episode_compiler_configured_bundle_asset_set_invalid"
                )
            role = str(row["role"])
            member = _safe_member(row["relative_path"])
            expected_names.add(member.as_posix())
            target = destination / member.name
            if target.exists():
                raise TaskEvaluationNativeArenaEpisodeCompilerError(
                    "episode_compiler_configured_bundle_asset_set_invalid"
                )
            try:
                payload = archive.read(member.as_posix())
            except KeyError as exc:
                raise TaskEvaluationNativeArenaEpisodeCompilerError(
                    "episode_compiler_configured_bundle_asset_missing"
                ) from exc
            target.write_bytes(payload)
            if _sha256_and_size(target) != (row.get("digest"), row.get("size_bytes")):
                raise TaskEvaluationNativeArenaEpisodeCompilerError(
                    "episode_compiler_configured_bundle_asset_identity_mismatch"
                )
            target.chmod(0o440)
            extracted[role] = target
        if set(names) != expected_names:
            raise TaskEvaluationNativeArenaEpisodeCompilerError(
                "episode_compiler_configured_bundle_member_set_invalid"
            )
    return extracted


def _identity(value: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    if value.get("identity") != expected:
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            f"episode_compiler_identity_mismatch:{label}"
        )


def compile_native_arena_episode(
    *,
    envelope: Mapping[str, Any],
    materialized_references: Mapping[str, Mapping[str, Any]],
    output_root: str | Path,
) -> dict[str, Any]:
    """Compile one production-owned native packet without provider mutation."""

    request = envelope["request"]
    root = Path(output_root).resolve()
    revision = validate_configured_scene_revision(
        _json_reference(
            materialized_references,
            "scene.configured_revision",
            "task_evaluation_configured_scene_revision.v1",
        )
    )
    if (
        revision["revision_digest"] != envelope["configured_scene_revision_digest"]
        or revision["scene_identity"] != request["scene"]["identity"]
        or revision["task_template"]["identity"] != request["task"]["identity"]
        or revision["replacement"]["identity"]
        != request["task"]["subject"]["identity"]
    ):
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_configured_revision_binding_mismatch"
        )
    robot = _json_reference(
        materialized_references,
        "robot.configuration",
        "task_evaluation_native_robot_configuration.v1",
    )
    kinematics = _json_reference(
        materialized_references,
        "robot.kinematics",
        "task_evaluation_native_robot_kinematics.v1",
    )
    joint_bounds = _json_reference(
        materialized_references,
        "robot.joint_bounds",
        "task_evaluation_native_robot_joint_bounds.v1",
    )
    base = _json_reference(
        materialized_references,
        "robot.base_registration",
        "task_evaluation_robot_to_scene_registration.v1",
    )
    controller = _json_reference(
        materialized_references,
        "controller.configuration",
        "task_evaluation_native_controller_configuration.v1",
    )
    sensors = _json_reference(
        materialized_references,
        "sensors.configuration",
        "task_evaluation_native_sensor_configuration.v1",
    )
    task_adapter = adapt_rigid_relocation_task_template(
        request=request,
        configured_revision=revision,
        materialized_references=materialized_references,
    )
    task_definition = task_adapter["native_task_definition"]
    success = task_adapter["native_success_criteria"]
    execution = task_adapter["native_episode_execution"]
    for label, value in (
        ("robot.configuration", robot),
        ("robot.kinematics", kinematics),
        ("robot.joint_bounds", joint_bounds),
    ):
        _identity(value, request["robot"]["identity"], label=label)
    _identity(controller, request["controller"]["identity"], label="controller")
    if (
        controller.get("kind") != request["controller"]["kind"]
        or base.get("robot_identity") != request["robot"]["identity"]
        or base.get("scene_identity") != request["scene"]["identity"]
        or base.get("robot_mount_interface_digest")
        != revision["registration"]["robot_mount_interface"]["digest"]
        or sensors.get("scene_camera_calibration_digest")
        != revision["registration"]["camera_calibration"]["digest"]
        or task_definition.get("identity") != request["task"]["identity"]
        or success.get("identity") != request["task"]["identity"]
        or execution.get("identity") != request["task"]["identity"]
    ):
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_request_binding_mismatch"
        )
    task_spec = task_definition.get("task_spec")
    if not isinstance(task_spec, Mapping):
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_task_definition_invalid"
        )
    task_spec = dict(task_spec)
    task_spec["subject_asset_id"] = request["task"]["subject"]["identity"]["id"]
    task_spec["manipulation_strategy"] = request["task"]["strategy"]
    task_spec["success_criteria"] = success.get("criteria")
    configured_assets = _extract_configured_assets(
        _reference_path(
            materialized_references,
            "scene.configured_revision.configured_scene_bundle",
        ),
        output_root=root,
    )
    object_pose = task_definition.get("task_object_pose_world")
    packet_assets = []
    for role, semantic_role in (
        ("appearance", "scene_appearance"),
        ("collision", "scene_collision"),
        ("replacement", "task_object"),
    ):
        path = configured_assets[role]
        digest, size = _sha256_and_size(path)
        row: dict[str, Any] = {
            "semantic_role": semantic_role,
            "filename": path.name,
            "source": {
                "root": "evidence",
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": size,
                "sha256": digest,
            },
            "pose_world": object_pose if role == "replacement" else {
                "position_world_m": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }
        if role == "replacement":
            row.update(
                asset_id=request["task"]["subject"]["identity"]["id"],
                object_type="RIGID",
                reset_state={"root_pose_world": object_pose, "joint_positions": {}},
            )
        packet_assets.append(row)
    packet_request: dict[str, Any] = {
        "schema_version": "native_task_arena_packet_request.v1",
        "scene_id": request["scene"]["identity"]["id"],
        "task_id": request["task"]["identity"]["id"],
        "task_spec": task_spec,
        "task_joint_bindings": task_definition.get("task_joint_bindings") or [],
        "task_state_binding": task_definition.get("task_state_binding"),
        "assets": packet_assets,
        "robot_base_pose_world": base.get("pose_world"),
        "robot_joint_reset_positions_rad": robot.get("joint_reset_positions_rad"),
        "cameras": sensors.get("cameras"),
        "scenario": execution.get("scenario"),
        "physics_frequency_hz": execution.get("physics_frequency_hz"),
        "configured_task_template_adapter": task_adapter,
        "request_digest": "",
    }
    packet_request["request_digest"] = canonical_digest(
        packet_request, digest_field="request_digest"
    )
    packet_root = root / "native-task-packet"
    materialize_native_task_arena_packet(
        request=packet_request,
        evidence_root=root,
        output_dir=packet_root,
    )
    packet_zip = root / "native-task-arena-bundle.zip"
    build_task_evaluation_adapter_bundle(
        source_root=packet_root,
        output_path=packet_zip,
        request=request,
        role="construction_packet",
    )
    digest, size = _sha256_and_size(packet_zip)
    adapter_result = materialize_native_arena_adapter(
        request=request,
        compiled_episode_packet_path=packet_zip,
        compiled_episode_packet_reference={
            "uri": "production-internal://compiled-episode-packet",
            "digest": digest,
            "size_bytes": size,
        },
        configured_revision=revision,
        runtime_source_bundle_path=_reference_path(
            materialized_references,
            "execution_adapter.runtime_source_bundle",
        ),
        output_root=root / "native-arena-adapter",
    )
    output: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "completed",
        "run_id": envelope["run_id"],
        "configured_scene_revision_digest": revision["revision_digest"],
        "configured_task_template_adapter": {
            "schema_version": task_adapter["schema_version"],
            "adapter_digest": task_adapter["adapter_digest"],
            "source_documents_digest": task_adapter["source_documents"][
                "source_documents_digest"
            ],
            "manipulation_strategy": task_adapter["manipulation_strategy"],
        },
        "compiled_episode_packet": {
            "format": "native_task_arena_bundle_zip",
            "path": str(packet_zip),
            "digest": digest,
            "size_bytes": size,
        },
        "adapter_result": {
            "path": str(
                root
                / "native-arena-adapter"
                / "task_evaluation_native_arena_adapter_result.v1.json"
            ),
            "digest": adapter_result["result_digest"],
            "packet_receipt_digest": adapter_result["packet_receipt_digest"],
            "runtime_source_receipt_digest": adapter_result[
                "runtime_source_receipt_digest"
            ],
        },
        "compiled_by_production": True,
        "customer_supplied_prebuilt_episode_packet": False,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "raw_secret_values_recorded": False,
        "compiler_output_digest": "",
    }
    output["compiler_output_digest"] = canonical_digest(
        output, digest_field="compiler_output_digest"
    )
    return output


__all__ = [
    "OUTPUT_SCHEMA_VERSION",
    "TaskEvaluationNativeArenaEpisodeCompilerError",
    "compile_native_arena_episode",
]
