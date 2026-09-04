"""Seal scene-specific native placement evidence for a passive destination."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest

OBSERVATION_SCHEMA = "task_evaluation_rigid_destination_native_observation.v1"
OUTPUT_SCHEMA = "task_evaluation_rigid_destination_placement_qualification.v1"
PRODUCER = "native_task_arena_destination_qualification"


class RigidDestinationPlacementQualificationError(ValueError):
    """The native destination-placement evidence was incomplete or inconsistent."""


def _digest(value: Any) -> bool:
    text = str(value or "")
    return (
        text.startswith("sha256:")
        and len(text) == 71
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _document(path: str | Path, *, schema: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RigidDestinationPlacementQualificationError(
            f"rigid_destination_placement_document_invalid:{schema}"
        ) from exc
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise RigidDestinationPlacementQualificationError(
            f"rigid_destination_placement_document_invalid:{schema}"
        )
    return source, value


def _pose(value: Any, *, field: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 7
    ):
        raise RigidDestinationPlacementQualificationError(
            f"rigid_destination_placement_pose_invalid:{field}"
        )
    try:
        pose = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise RigidDestinationPlacementQualificationError(
            f"rigid_destination_placement_pose_invalid:{field}"
        ) from exc
    if not all(math.isfinite(item) for item in pose) or not math.isclose(
        sum(item * item for item in pose[3:]),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise RigidDestinationPlacementQualificationError(
            f"rigid_destination_placement_pose_invalid:{field}"
        )
    return pose


def _quaternion_angle(left: Sequence[float], right: Sequence[float]) -> float:
    dot = abs(sum(a * b for a, b in zip(left, right, strict=True)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _positive(mapping: Mapping[str, Any], field: str) -> float:
    try:
        value = float(mapping[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise RigidDestinationPlacementQualificationError(
            f"rigid_destination_placement_limit_invalid:{field}"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise RigidDestinationPlacementQualificationError(
            f"rigid_destination_placement_limit_invalid:{field}"
        )
    return value


def materialize_rigid_destination_placement_qualification(
    *,
    observation_path: str | Path,
    configured_scene_collision_path: str | Path,
    destination_asset_path: str | Path,
    destination_static_qualification_path: str | Path,
    destination_native_import_qualification_path: str | Path,
    destination_geometry_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Derive the qualification from native samples instead of asserted booleans."""

    _observation_file, observation = _document(
        observation_path, schema=OBSERVATION_SCHEMA
    )
    static_file, static = _document(
        destination_static_qualification_path,
        schema="task_evaluation_rigid_replacement_static_qualification.v1",
    )
    native_file, native = _document(
        destination_native_import_qualification_path,
        schema="task_evaluation_replacement_native_import_result.v1",
    )
    _geometry_file, geometry = _document(
        destination_geometry_path,
        schema="task_evaluation_rigid_destination_geometry.v1",
    )
    collision_file = Path(configured_scene_collision_path).resolve()
    asset_file = Path(destination_asset_path).resolve()
    identity = geometry.get("destination_identity")
    expected_pose_mapping = geometry.get("pose_world")
    if not isinstance(expected_pose_mapping, Mapping):
        raise RigidDestinationPlacementQualificationError(
            "rigid_destination_placement_geometry_pose_invalid"
        )
    expected_pose = _pose(
        [
            *(expected_pose_mapping.get("position_world_m") or []),
            *(expected_pose_mapping.get("orientation_xyzw") or []),
        ],
        field="expected",
    )
    limits = observation.get("qualification_limits")
    if not isinstance(limits, Mapping):
        raise RigidDestinationPlacementQualificationError(
            "rigid_destination_placement_limits_missing"
        )
    maximum_penetration = _positive(limits, "maximum_penetration_m")
    settle_translation_tolerance = _positive(
        limits, "settle_translation_tolerance_m"
    )
    settle_rotation_tolerance = _positive(
        limits, "settle_rotation_tolerance_rad"
    )
    reset_translation_tolerance = _positive(
        limits, "reset_translation_tolerance_m"
    )
    reset_rotation_tolerance = _positive(limits, "reset_rotation_tolerance_rad")
    minimum_camera_pixels = limits.get("minimum_camera_pixels")
    if not isinstance(minimum_camera_pixels, Mapping):
        raise RigidDestinationPlacementQualificationError(
            "rigid_destination_placement_camera_limits_missing"
        )
    actual = {
        "configured_scene_collision_digest": _sha256(collision_file),
        "destination_asset_digest": _sha256(asset_file),
        "destination_static_qualification_digest": _sha256(static_file),
        "destination_native_import_qualification_digest": _sha256(native_file),
        "destination_geometry_digest": geometry.get("geometry_digest"),
    }
    if (
        observation.get("status") != "completed"
        or observation.get("producer") != PRODUCER
        or observation.get("native_isaac_executed") is not True
        or observation.get("destination_identity") != identity
        or not _digest(observation.get("configured_scene_revision_digest"))
        or static.get("status") != "authored_structure_statically_qualified"
        or static.get("replacement_identity") != identity
        or (static.get("replacement_usd") or {}).get("sha256")
        != actual["destination_asset_digest"]
        or (static.get("replacement_usd") or {}).get("size_bytes")
        != asset_file.stat().st_size
        or static.get("result_digest")
        != canonical_digest(static, digest_field="result_digest")
        or native.get("status") != "qualified"
        or native.get("replacement_identity") != identity
        or native.get("asset_digest") != actual["destination_asset_digest"]
        or native.get("static_qualification_digest")
        != actual["destination_static_qualification_digest"]
        or native.get("native_isaac_executed") is not True
        or native.get("native_simulator_import_qualified") is not True
        or native.get("result_digest")
        != canonical_digest(native, digest_field="result_digest")
        or geometry.get("status") != "qualified"
        or geometry.get("destination_static_qualification_digest")
        != actual["destination_static_qualification_digest"]
        or geometry.get("geometry_digest")
        != canonical_digest(geometry, digest_field="geometry_digest")
        or observation.get("pose_world") != expected_pose_mapping
        or observation.get("observation_digest")
        != canonical_digest(observation, digest_field="observation_digest")
        or any(observation.get(field) != digest for field, digest in actual.items())
    ):
        raise RigidDestinationPlacementQualificationError(
            "rigid_destination_placement_binding_invalid"
        )
    settle_samples = observation.get("settle_samples")
    reset_samples = observation.get("reset_samples")
    camera_samples = observation.get("camera_observations")
    if (
        not isinstance(settle_samples, list)
        or len(settle_samples) < 3
        or not isinstance(reset_samples, list)
        or len(reset_samples) < 3
        or not isinstance(camera_samples, list)
    ):
        raise RigidDestinationPlacementQualificationError(
            "rigid_destination_placement_samples_missing"
        )

    def pose_errors(row: Mapping[str, Any], *, field: str) -> tuple[float, float]:
        observed = _pose(row.get("destination_pose_world"), field=field)
        return (
            math.dist(observed[:3], expected_pose[:3]),
            _quaternion_angle(observed[3:], expected_pose[3:]),
        )

    settle_errors = [
        pose_errors(row, field=f"settle:{index}")
        for index, row in enumerate(settle_samples)
        if isinstance(row, Mapping)
    ]
    reset_errors = [
        pose_errors(row, field=f"reset:{index}")
        for index, row in enumerate(reset_samples)
        if isinstance(row, Mapping)
    ]
    if len(settle_errors) != len(settle_samples) or len(reset_errors) != len(
        reset_samples
    ):
        raise RigidDestinationPlacementQualificationError(
            "rigid_destination_placement_samples_invalid"
        )
    try:
        penetration = [
            float(row["maximum_penetration_m"]) for row in settle_samples
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise RigidDestinationPlacementQualificationError(
            "rigid_destination_placement_penetration_readback_invalid"
        ) from exc
    nonpenetration_passed = all(
        math.isfinite(value) and 0.0 <= value <= maximum_penetration
        for value in penetration
    )
    support_stability_passed = all(
        row.get("support_contact_observed") is True
        for row in settle_samples
    ) and all(
        translation <= settle_translation_tolerance
        and rotation <= settle_rotation_tolerance
        for translation, rotation in settle_errors
    )
    cameras: dict[str, bool] = {}
    for role in ("external", "wrist", "overview"):
        rows = [
            row
            for row in camera_samples
            if isinstance(row, Mapping) and row.get("role") == role
        ]
        minimum = minimum_camera_pixels.get(role)
        cameras[role] = (
            len(rows) == 1
            and isinstance(minimum, int)
            and not isinstance(minimum, bool)
            and minimum > 0
            and isinstance(rows[0].get("task_support_pixel_count"), int)
            and rows[0]["task_support_pixel_count"] >= minimum
        )
    maximum_reset_translation = max(value[0] for value in reset_errors)
    maximum_reset_rotation = max(value[1] for value in reset_errors)
    reset_passed = (
        maximum_reset_translation <= reset_translation_tolerance
        and maximum_reset_rotation <= reset_rotation_tolerance
    )
    if not nonpenetration_passed or not support_stability_passed or not all(
        cameras.values()
    ) or not reset_passed:
        raise RigidDestinationPlacementQualificationError(
            "rigid_destination_placement_native_qualification_failed"
        )
    result: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "qualified",
        "producer": "task_evaluation_rigid_destination_placement_qualification",
        "native_observation_digest": observation["observation_digest"],
        "destination_identity": identity,
        "configured_scene_revision_digest": observation[
            "configured_scene_revision_digest"
        ],
        **actual,
        "pose_world": expected_pose_mapping,
        "nonpenetration_passed": True,
        "support_stability_passed": True,
        "camera_visibility": cameras,
        "repeated_reset_readback": {
            "repeat_count": len(reset_samples),
            "maximum_translation_error_m": maximum_reset_translation,
            "maximum_rotation_error_rad": maximum_reset_rotation,
            "translation_tolerance_m": reset_translation_tolerance,
            "rotation_tolerance_rad": reset_rotation_tolerance,
        },
        "placement_qualification_digest": "",
    }
    result["placement_qualification_digest"] = canonical_digest(
        result, digest_field="placement_qualification_digest"
    )
    destination = Path(output_path).resolve()
    if destination.exists() or destination.is_symlink():
        raise RigidDestinationPlacementQualificationError(
            "rigid_destination_placement_output_exists"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, result)
    return json.loads(json.dumps(result))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", required=True, type=Path)
    parser.add_argument("--configured-scene-collision", required=True, type=Path)
    parser.add_argument("--destination-asset", required=True, type=Path)
    parser.add_argument("--destination-static-qualification", required=True, type=Path)
    parser.add_argument("--destination-native-import-qualification", required=True, type=Path)
    parser.add_argument("--destination-geometry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = materialize_rigid_destination_placement_qualification(
        observation_path=args.observation,
        configured_scene_collision_path=args.configured_scene_collision,
        destination_asset_path=args.destination_asset,
        destination_static_qualification_path=args.destination_static_qualification,
        destination_native_import_qualification_path=(
            args.destination_native_import_qualification
        ),
        destination_geometry_path=args.destination_geometry,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
