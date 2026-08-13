"""Materialize calibrated native-render requests from paired-target preflight.

The paired-target preflight binds the repaired NuRec appearance, SimReady
candidate, collision scene, and source camera trajectory for one to five task
objects.  This module turns that sealed trajectory into the small camera
interface consumed by the existing Isaac NuRec renderer.  It does not copy the
large assets, allocate a provider, import Isaac, or claim that any native render
has happened.

The source trajectory uses the common OpenGL camera-to-world convention: local
``-Z`` is forward and local ``+Y`` is up.  The Isaac runner accepts a position,
look-at target, up vector, and vertical field of view.  Conversion is therefore
deterministic and retains the exact source pose and focal length in every row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .dual_task_rehearsal_contract import MAX_REPLACEMENT_OBJECTS


SCHEMA_VERSION = "paired_target_native_render_request.v1"
PREFLIGHT_SCHEMA_VERSION = "paired_target_native_preflight.v1"
FROZEN_CANDIDATES = ("pi05_droid", "groot_n17_droid")


class PairedTargetNativeRenderRequestError(ValueError):
    """Stable, fail-closed calibrated native-render request failure."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_mapping(path: str | Path, code: str) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path).expanduser().resolve()
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedTargetNativeRenderRequestError(code) from exc
    if candidate.is_symlink() or not isinstance(value, dict):
        raise PairedTargetNativeRenderRequestError(code)
    return candidate, value


def _verified_file_record(record: Any, code: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, Mapping):
        raise PairedTargetNativeRenderRequestError(code)
    candidate = Path(str(record.get("path") or "")).expanduser().resolve()
    if (
        candidate.is_symlink()
        or not candidate.is_file()
        or candidate.stat().st_size != record.get("size_bytes")
        or _sha256(candidate) != record.get("sha256")
    ):
        raise PairedTargetNativeRenderRequestError(code)
    return candidate, {
        "path": str(candidate),
        "size_bytes": candidate.stat().st_size,
        "sha256": _sha256(candidate),
    }


def _finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _vector(values: Sequence[Any], *, length: int, code: str) -> list[float]:
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or len(values) != length
        or any(not _finite(item) for item in values)
    ):
        raise PairedTargetNativeRenderRequestError(code)
    return [float(item) for item in values]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _camera_row(frame: Mapping[str, Any]) -> dict[str, Any]:
    code = "paired_target_native_camera_invalid"
    matrix = frame.get("transform_matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in matrix)
    ):
        raise PairedTargetNativeRenderRequestError(code)
    rows = [_vector(row, length=4, code=code) for row in matrix]
    if any(abs(value) > 1.0e-9 for value in rows[3][:3]) or abs(rows[3][3] - 1.0) > 1.0e-9:
        raise PairedTargetNativeRenderRequestError(code)

    right = [rows[index][0] for index in range(3)]
    up = [rows[index][1] for index in range(3)]
    backward = [rows[index][2] for index in range(3)]
    axes = (right, up, backward)
    if any(abs(_norm(axis) - 1.0) > 1.0e-6 for axis in axes) or any(
        abs(_dot(axes[left], axes[right_index])) > 1.0e-6
        for left, right_index in ((0, 1), (0, 2), (1, 2))
    ):
        raise PairedTargetNativeRenderRequestError(code)
    right_cross_up = [
        right[1] * up[2] - right[2] * up[1],
        right[2] * up[0] - right[0] * up[2],
        right[0] * up[1] - right[1] * up[0],
    ]
    if any(abs(right_cross_up[index] - backward[index]) > 1.0e-6 for index in range(3)):
        raise PairedTargetNativeRenderRequestError(code)

    try:
        width = int(frame["w"])
        height = int(frame["h"])
        fx = float(frame["fl_x"])
        fy = float(frame["fl_y"])
        cx = float(frame["cx"])
        cy = float(frame["cy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PairedTargetNativeRenderRequestError(code) from exc
    if (
        frame.get("camera_model") != "OPENCV"
        or width <= 0
        or height <= 0
        or fx <= 0.0
        or fy <= 0.0
        or not math.isclose(fx, fy, rel_tol=1.0e-9, abs_tol=1.0e-9)
        or not math.isclose(cx, width / 2.0, rel_tol=0.0, abs_tol=1.0e-6)
        or not math.isclose(cy, height / 2.0, rel_tol=0.0, abs_tol=1.0e-6)
        or any(float(frame.get(key, 0.0)) != 0.0 for key in ("k1", "k2", "p1", "p2"))
    ):
        raise PairedTargetNativeRenderRequestError(code)

    camera_id = str(frame.get("camera_id") or "")
    physical_index = frame.get("physical_camera_index")
    if (
        not camera_id
        or PurePosixPath(camera_id).name != camera_id
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in camera_id
        )
        or not isinstance(physical_index, int)
        or isinstance(physical_index, bool)
        or physical_index < 0
    ):
        raise PairedTargetNativeRenderRequestError(code)
    position = [rows[index][3] for index in range(3)]
    target = [position[index] - backward[index] for index in range(3)]
    vertical_fov_degrees = math.degrees(2.0 * math.atan(height / (2.0 * fy)))
    return {
        "id": camera_id,
        "spec": {
            "pos": position,
            "target": target,
            "up": up,
            "fov": vertical_fov_degrees,
        },
        "source": {
            "physical_camera_index": physical_index,
            "camera_model": "OPENCV",
            "transform_matrix_camera_to_world_opengl": rows,
            "width": width,
            "height": height,
            "fl_x": fx,
            "fl_y": fy,
            "cx": cx,
            "cy": cy,
            "conversion": "position=column_3,target=position-column_2,up=column_1",
            "field_of_view": "vertical_2_atan_height_over_2_fy",
        },
    }


def _materialize_camera_spec(
    *, trajectory_path: Path, expected_ids: Sequence[str], destination: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedTargetNativeRenderRequestError(
            "paired_target_native_camera_trajectory_invalid"
        ) from exc
    frames = trajectory.get("frames") if isinstance(trajectory, Mapping) else None
    if not isinstance(frames, list) or len(frames) != len(expected_ids):
        raise PairedTargetNativeRenderRequestError("paired_target_native_camera_trajectory_invalid")
    rows = [_camera_row(frame) for frame in frames if isinstance(frame, Mapping)]
    if (
        len(rows) != len(frames)
        or [row["id"] for row in rows] != list(expected_ids)
        or [row["source"]["physical_camera_index"] for row in rows] != list(range(len(rows)))
    ):
        raise PairedTargetNativeRenderRequestError("paired_target_native_camera_order_mismatch")
    dimensions = {(row["source"]["width"], row["source"]["height"]) for row in rows}
    if len(dimensions) != 1:
        raise PairedTargetNativeRenderRequestError(
            "paired_target_native_camera_dimensions_mismatch"
        )
    destination.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record = {
        "relative_path": destination.name,
        "size_bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
        "camera_spec_digest": canonical_digest({"cameras": rows}),
        "camera_ids": [row["id"] for row in rows],
        "width": next(iter(dimensions))[0],
        "height": next(iter(dimensions))[1],
    }
    return rows, record


def materialize_paired_target_native_render_request(
    *, preflight_path: str | Path, output_root: str | Path
) -> dict[str, Any]:
    """Reverify one preflight and write per-task calibrated render requests."""

    preflight_file, preflight = _read_mapping(
        preflight_path, "paired_target_native_preflight_invalid"
    )
    tasks = preflight.get("tasks")
    if (
        preflight.get("schema_version") != PREFLIGHT_SCHEMA_VERSION
        or preflight.get("receipt_digest")
        != canonical_digest(preflight, digest_field="receipt_digest")
        or preflight.get("native_isaac_import_executed") is not False
        or preflight.get("generated_output_is_capture_or_physical_evidence") is not False
        or tuple(preflight.get("candidate_ids") or ()) != FROZEN_CANDIDATES
        or not isinstance(tasks, list)
        or not 1 <= len(tasks) <= MAX_REPLACEMENT_OBJECTS
        or preflight.get("replacement_object_count") != len(tasks)
    ):
        raise PairedTargetNativeRenderRequestError("paired_target_native_preflight_invalid")

    collision_path, collision = _verified_file_record(
        preflight.get("collision_scene"), "paired_target_native_collision_invalid"
    )
    output = Path(output_root).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise PairedTargetNativeRenderRequestError("paired_target_native_render_output_exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    task_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        replacement_assets: list[dict[str, Any]] = []
        for task in tasks:
            if not isinstance(task, Mapping):
                raise PairedTargetNativeRenderRequestError("paired_target_native_task_invalid")
            task_id = str(task.get("task_id") or "")
            asset_id = str(task.get("asset_id") or "")
            simready_path, simready = _verified_file_record(
                task.get("simready_usd"),
                f"paired_target_native_simready_invalid:{task_id}",
            )
            visual_path, visual = _verified_file_record(
                task.get("registered_replacement_usd"),
                f"paired_target_native_visual_invalid:{task_id}",
            )
            registration = task.get("asset_frame_registration")
            registered_static_path, registered_static = _verified_file_record(
                task.get("registered_static_qualification"),
                f"paired_target_native_registered_static_invalid:{task_id}",
            )
            registered_static["receipt_digest"] = task.get(
                "registered_static_qualification", {}
            ).get("receipt_digest")
            appearance_contract = task.get("appearance_contract")
            if (
                not isinstance(registration, Mapping)
                or not isinstance(registered_static, Mapping)
                or not isinstance(appearance_contract, Mapping)
                or appearance_contract.get("agent_authored_display_colors_preserved") is not True
                or appearance_contract.get("neutral_fallback_permitted") is not False
            ):
                raise PairedTargetNativeRenderRequestError(
                    f"paired_target_native_visual_invalid:{task_id}"
                )
            if (
                not task_id
                or not asset_id
                or task_id in {row["task_id"] for row in replacement_assets}
                or asset_id in {row["asset_id"] for row in replacement_assets}
            ):
                raise PairedTargetNativeRenderRequestError(
                    "paired_target_native_replacement_set_invalid"
                )
            replacement_assets.append(
                {
                    "task_id": task_id,
                    "asset_id": asset_id,
                    "simready_usd": simready,
                    "source_path": simready_path,
                    "visual_usd": visual,
                    "visual_source_path": visual_path,
                    "asset_frame_registration": dict(registration),
                    "registered_static_qualification": dict(registered_static),
                    "registered_static_source_path": registered_static_path,
                }
            )
        for task in tasks:
            if not isinstance(task, Mapping):
                raise PairedTargetNativeRenderRequestError("paired_target_native_task_invalid")
            task_id = str(task.get("task_id") or "")
            if not task_id or task_id in seen or PurePosixPath(task_id).name != task_id:
                raise PairedTargetNativeRenderRequestError("paired_target_native_task_invalid")
            seen.add(task_id)
            appearance_path, appearance = _verified_file_record(
                task.get("isaac_nurec_usdz"),
                f"paired_target_native_appearance_invalid:{task_id}",
            )
            active_replacement = next(
                row for row in replacement_assets if row["task_id"] == task_id
            )
            simready_path = active_replacement["source_path"]
            simready = active_replacement["simready_usd"]
            trajectory_path, trajectory = _verified_file_record(
                task.get("camera_trajectory"),
                f"paired_target_native_camera_trajectory_invalid:{task_id}",
            )
            _, camera_index = _verified_file_record(
                task.get("camera_index"),
                f"paired_target_native_camera_index_invalid:{task_id}",
            )
            expected_ids = (task.get("camera_index") or {}).get("camera_ids")
            if not isinstance(expected_ids, list) or len(expected_ids) != 8:
                raise PairedTargetNativeRenderRequestError(
                    f"paired_target_native_camera_index_invalid:{task_id}"
                )
            task_dir = output / task_id
            task_dir.mkdir()
            _, camera_spec = _materialize_camera_spec(
                trajectory_path=trajectory_path,
                expected_ids=[str(value) for value in expected_ids],
                destination=task_dir / "fixed_cameras.json",
            )
            task_rows.append(
                {
                    "task_id": task_id,
                    "asset_id": str(task.get("asset_id") or ""),
                    "appearance_usdz": appearance,
                    "simready_usd": simready,
                    "visual_usd": active_replacement["visual_usd"],
                    "asset_frame_registration": active_replacement["asset_frame_registration"],
                    "registered_static_qualification": active_replacement[
                        "registered_static_qualification"
                    ],
                    "co_present_replacements": [
                        {
                            "task_id": row["task_id"],
                            "asset_id": row["asset_id"],
                            "simready_usd": row["simready_usd"],
                            "visual_usd": row["visual_usd"],
                            "asset_frame_registration": row["asset_frame_registration"],
                            "registered_static_qualification": row[
                                "registered_static_qualification"
                            ],
                            "task_subject": row["task_id"] == task_id,
                            "passive_co_present": row["task_id"] != task_id,
                        }
                        for row in replacement_assets
                    ],
                    "collision_scene": collision,
                    "source_camera_trajectory": trajectory,
                    "source_camera_index": camera_index,
                    "fixed_camera_spec": {
                        **camera_spec,
                        "relative_path": f"{task_id}/{camera_spec['relative_path']}",
                    },
                    "appearance_native_import_executed": False,
                    "simready_native_import_executed": False,
                    "calibrated_native_renders_executed": False,
                    "reachability_executed": False,
                }
            )
            # Recheck source identities after materialization so a concurrent
            # source mutation cannot be hidden behind the output receipt.
            for path, record, code in (
                (appearance_path, appearance, "appearance"),
                (simready_path, simready, "simready"),
                (trajectory_path, trajectory, "trajectory"),
                (collision_path, collision, "collision"),
            ):
                if path.stat().st_size != record["size_bytes"] or _sha256(path) != record["sha256"]:
                    raise PairedTargetNativeRenderRequestError(
                        f"paired_target_native_source_changed:{task_id}:{code}"
                    )
            for row in replacement_assets:
                source_path = row["source_path"]
                record = row["simready_usd"]
                if (
                    source_path.stat().st_size != record["size_bytes"]
                    or _sha256(source_path) != record["sha256"]
                ):
                    raise PairedTargetNativeRenderRequestError(
                        "paired_target_native_source_changed:"
                        f"{task_id}:co_present:{row['asset_id']}"
                    )
                visual_source_path = row["visual_source_path"]
                visual_record = row["visual_usd"]
                if (
                    visual_source_path.stat().st_size != visual_record["size_bytes"]
                    or _sha256(visual_source_path) != visual_record["sha256"]
                ):
                    raise PairedTargetNativeRenderRequestError(
                        "paired_target_native_source_changed:"
                        f"{task_id}:co_present_visual:{row['asset_id']}"
                    )
                registered_static_source_path = row["registered_static_source_path"]
                registered_static_record = row["registered_static_qualification"]
                if (
                    registered_static_source_path.stat().st_size
                    != registered_static_record["size_bytes"]
                    or _sha256(registered_static_source_path)
                    != registered_static_record["sha256"]
                ):
                    raise PairedTargetNativeRenderRequestError(
                        "paired_target_native_source_changed:"
                        f"{task_id}:registered_static:{row['asset_id']}"
                    )

        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "native_render_requests_materialized_pending_isaac_execution",
            "program_id": "arm-decision-proof-v1",
            "scene_id": str(preflight["scene_id"]),
            "preflight": {
                "path": str(preflight_file),
                "size_bytes": preflight_file.stat().st_size,
                "sha256": _sha256(preflight_file),
                "receipt_digest": preflight["receipt_digest"],
            },
            "replacement_object_count": len(task_rows),
            "maximum_replacement_objects": MAX_REPLACEMENT_OBJECTS,
            "tasks": task_rows,
            "candidate_ids": list(FROZEN_CANDIDATES),
            "required_controls": ["zero_action_negative", "scripted_positive"],
            "provider_allocation_performed": False,
            "paid_execution_authorized_by_request": False,
            "source_assets_copied_or_mutated": False,
            "native_isaac_executed": False,
            "generated_output_is_capture_or_physical_evidence": False,
            "claim_boundary": (
                "calibrated_native_render_requests_only_pending_separate_isaac_"
                "appearance_and_simready_import_render_reachability_controls_and_policy_gates"
            ),
            "receipt_digest": "",
        }
        result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
        receipt_path = output / "paired_target_native_render_request.v1.json"
        receipt_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return json.loads(json.dumps(result))
    except Exception:
        shutil.rmtree(output)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = materialize_paired_target_native_render_request(
        preflight_path=args.preflight,
        output_root=args.output_root,
    )
    print(json.dumps({"receipt_digest": result["receipt_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
