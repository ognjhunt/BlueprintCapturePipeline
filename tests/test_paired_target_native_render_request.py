from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paired_target_native_render_request import (
    PairedTargetNativeRenderRequestError,
    materialize_paired_target_native_render_request,
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, **extra: object) -> dict:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha(path),
        **extra,
    }


def _trajectory(path: Path, *, prefix: str) -> tuple[dict, list[str]]:
    ids = [f"{prefix}_{index}" for index in range(8)]
    frames = []
    for index, camera_id in enumerate(ids):
        angle = index * math.pi / 4.0
        right = [math.cos(angle), math.sin(angle), 0.0]
        up = [0.0, 0.0, 1.0]
        backward = [math.sin(angle), -math.cos(angle), 0.0]
        position = [1.0 + index, 2.0, 3.0]
        frames.append(
            {
                "camera_id": camera_id,
                "physical_camera_index": index,
                "camera_model": "OPENCV",
                "w": 1536,
                "h": 1024,
                "fl_x": 1024.0,
                "fl_y": 1024.0,
                "cx": 768.0,
                "cy": 512.0,
                "k1": 0.0,
                "k2": 0.0,
                "p1": 0.0,
                "p2": 0.0,
                "transform_matrix": [
                    [right[0], up[0], backward[0], position[0]],
                    [right[1], up[1], backward[1], position[1]],
                    [right[2], up[2], backward[2], position[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        )
    value = {"frames": frames}
    path.write_text(json.dumps(value), encoding="utf-8")
    return value, ids


def _fixture(tmp_path: Path, *, task_count: int = 2) -> Path:
    collision = tmp_path / "collision.usda"
    collision.write_text("collision", encoding="utf-8")
    tasks = []
    for task_index in range(task_count):
        task_id = f"task_{task_index}"
        task_root = tmp_path / task_id
        task_root.mkdir()
        appearance = task_root / "appearance.usdz"
        simready = task_root / "replacement.usda"
        visual = task_root / "replacement_visual.usda"
        index_path = task_root / "camera_index.json"
        trajectory_path = task_root / "review_transforms.json"
        appearance.write_bytes(b"usdz" + bytes([task_index]))
        simready.write_text(f"simready {task_index}", encoding="utf-8")
        visual.write_text(f"visual {task_index}", encoding="utf-8")
        registered_static_path = task_root / "registered_static.json"
        registered_static_path.write_text("{}", encoding="utf-8")
        _, camera_ids = _trajectory(trajectory_path, prefix=task_id)
        index_path.write_text(json.dumps({"frames": camera_ids}), encoding="utf-8")
        tasks.append(
            {
                "task_id": task_id,
                "asset_id": f"asset_{task_index}",
                "isaac_nurec_usdz": _record(appearance),
                "simready_usd": _record(simready),
                "registered_replacement_usd": _record(visual),
                "appearance_contract": {
                    "agent_authored_display_colors_preserved": True,
                    "generated_texture_maps_present": False,
                    "neutral_fallback_permitted": False,
                },
                "asset_frame_registration": {
                    "registration_digest": "sha256:" + str(task_index) * 64,
                    "T_observed_world_axes_from_asset_local_axes": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
                "registered_static_qualification": {
                    **_record(registered_static_path),
                    "receipt_digest": "sha256:" + str(task_index) * 64,
                },
                "camera_trajectory": _record(trajectory_path),
                "camera_index": _record(index_path, camera_ids=camera_ids),
            }
        )
    preflight = {
        "schema_version": "paired_target_native_preflight.v1",
        "scene_id": "scene",
        "replacement_object_count": task_count,
        "tasks": tasks,
        "collision_scene": _record(collision),
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "native_isaac_import_executed": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "receipt_digest": "",
    }
    preflight["receipt_digest"] = canonical_digest(preflight, digest_field="receipt_digest")
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(preflight), encoding="utf-8")
    return path


def test_materializes_two_task_calibrated_requests_without_copying_assets(
    tmp_path: Path,
) -> None:
    preflight = _fixture(tmp_path)
    output = tmp_path / "out"

    result = materialize_paired_target_native_render_request(
        preflight_path=preflight,
        output_root=output,
    )

    assert result["replacement_object_count"] == 2
    assert result["maximum_replacement_objects"] == 5
    assert result["candidate_ids"] == ["pi05_droid", "groot_n17_droid"]
    assert result["required_controls"] == [
        "zero_action_negative",
        "scripted_positive",
    ]
    assert result["provider_allocation_performed"] is False
    assert result["native_isaac_executed"] is False
    assert result["source_assets_copied_or_mutated"] is False
    assert not list(output.rglob("*.usdz"))
    assert not list(output.rglob("*.usda"))
    for task in result["tasks"]:
        assert len(task["co_present_replacements"]) == 2
        assert sum(row["task_subject"] is True for row in task["co_present_replacements"]) == 1
        assert (
            sum(row["passive_co_present"] is True for row in task["co_present_replacements"]) == 1
        )
        subject = next(row for row in task["co_present_replacements"] if row["task_subject"])
        assert subject["task_id"] == task["task_id"]
        spec_path = output / task["fixed_camera_spec"]["relative_path"]
        rows = json.loads(spec_path.read_text())
        assert len(rows) == 8
        assert rows[0]["spec"]["pos"] == [1.0, 2.0, 3.0]
        assert rows[0]["spec"]["target"] == [1.0, 3.0, 3.0]
        assert rows[0]["spec"]["up"] == [0.0, 0.0, 1.0]
        assert rows[0]["spec"]["fov"] == pytest.approx(math.degrees(2.0 * math.atan(0.5)))
        assert rows[0]["source"]["transform_matrix_camera_to_world_opengl"] == [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, 2.0],
            [0.0, 1.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        assert task["appearance_native_import_executed"] is False
        assert task["simready_native_import_executed"] is False
        assert task["calibrated_native_renders_executed"] is False


def test_scales_to_five_and_rejects_tampered_or_ambiguous_cameras(tmp_path: Path) -> None:
    preflight = _fixture(tmp_path, task_count=5)
    result = materialize_paired_target_native_render_request(
        preflight_path=preflight,
        output_root=tmp_path / "five",
    )
    assert len(result["tasks"]) == 5
    assert all(len(task["co_present_replacements"]) == 5 for task in result["tasks"])
    assert all(
        sum(row["task_subject"] is True for row in task["co_present_replacements"]) == 1
        for task in result["tasks"]
    )

    other = tmp_path / "tamper"
    other.mkdir()
    tampered_preflight = _fixture(other)
    preflight_value = json.loads(tampered_preflight.read_text())
    trajectory = Path(preflight_value["tasks"][0]["camera_trajectory"]["path"])
    trajectory_value = json.loads(trajectory.read_text())
    trajectory_value["frames"][0]["cx"] = 700.0
    trajectory.write_text(json.dumps(trajectory_value), encoding="utf-8")
    preflight_value["tasks"][0]["camera_trajectory"] = _record(trajectory)
    preflight_value["receipt_digest"] = canonical_digest(
        preflight_value, digest_field="receipt_digest"
    )
    tampered_preflight.write_text(json.dumps(preflight_value), encoding="utf-8")
    with pytest.raises(
        PairedTargetNativeRenderRequestError,
        match="paired_target_native_camera_invalid",
    ):
        materialize_paired_target_native_render_request(
            preflight_path=tampered_preflight,
            output_root=other / "out",
        )
