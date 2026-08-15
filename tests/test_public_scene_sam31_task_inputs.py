from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_sam31_task_inputs import (
    PublicSceneSam31InputError,
    materialize_public_scene_sam31_task_inputs,
)
from blueprint_pipeline.scene_placement.semantic_gaussian_lifting import (
    canonical_json_digest,
)
from blueprint_pipeline.scene_placement.sam31_source_track_provider import (
    _validate_request,
)
from tests.test_public_scene_calibrated_object_masks import _task
from tests.test_sam31_source_track_provider import _request as _sam_request


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    source = tmp_path / "source"
    source.mkdir()
    cameras = []
    image_rows = []
    for index, camera_id in enumerate(("front", "right")):
        image_path = source / f"{camera_id}.png"
        Image.new("RGB", (8, 6), (30 + index, 60, 90)).save(image_path)
        camera = {
            "camera_id": camera_id,
            "T_world_camera_provider_frame": [
                [1.0, 0.0, 0.0, float(index)],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "intrinsics": {
                "model": "PINHOLE",
                "fx": 4.0,
                "fy": 4.0,
                "cx": 4.0,
                "cy": 3.0,
                "width": 8,
                "height": 6,
            },
        }
        cameras.append(camera)
        image_rows.append(
            {
                "camera_id": camera_id,
                "relative_path": image_path.name,
                "size_bytes": image_path.stat().st_size,
                "sha256": _sha(image_path),
            }
        )
    camera_path = source / "cameras.v1.json"
    camera_path.write_text(json.dumps(cameras), encoding="utf-8")
    receipt = {
        "schema_version": "public_scene_interiorgs_edit_input_receipt.v2",
        "status": "render_derived_input_packet_materialized",
        "scene": {"task_id": "task_a"},
        "derived_artifacts": {
            "cameras": {
                "relative_path": camera_path.name,
                "size_bytes": camera_path.stat().st_size,
                "sha256": _sha(camera_path),
            },
            "images": image_rows,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = source / "public_scene_interiorgs_edit_input_receipt.v2.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(_task("task_a", 1)), encoding="utf-8")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_sam_request()["provider_profile"]), encoding="utf-8")
    prompts_path = tmp_path / "prompts.json"
    prompts_path.write_text(
        json.dumps([{"prompt_id": "washer", "text": "washer", "output_label": "washer"}]),
        encoding="utf-8",
    )
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_text("fixture", encoding="utf-8")
    return {
        "receipt": receipt_path,
        "task": task_path,
        "profile": profile_path,
        "prompts": prompts_path,
        "ffmpeg": ffmpeg,
        "source": source,
    }


def test_materializes_task_local_sam_request_without_hand_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)

    def fake_encode(*, output_path: Path, **_: object) -> list[str]:
        output_path.write_bytes(b"lossless-sequence-fixture")
        return ["fixture-ffmpeg", "ffv1"]

    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_sam31_task_inputs._encode_lossless_sequence",
        fake_encode,
    )
    result = materialize_public_scene_sam31_task_inputs(
        calibrated_view_receipt_path=fixture["receipt"],
        task_freeze_path=fixture["task"],
        provider_profile_path=fixture["profile"],
        prompts_path=fixture["prompts"],
        output_root=tmp_path / "output",
        ffmpeg_executable=fixture["ffmpeg"],
    )

    assert result["status"] == "prepared_no_upload_no_execution"
    assert result["camera_count"] == 2
    assert result["paid_execution_started"] is False
    assert result["provider_mutations_performed"] == 0
    assert set(result["camera_frame_map"]) == {"front", "right"}
    request_path = tmp_path / "output/semantic_sam31_source_track_run_request.v1.json"
    request = json.loads(request_path.read_text())
    assert result["run_request"]["request_digest"] == canonical_json_digest(request)
    assert request["frame_registry"][0]["source_frame_digest"] == _sha(
        tmp_path / "output/lossless_frames/000000.png"
    )
    assert request["bindings"]["retained_video_digest"] == _sha(
        tmp_path / "output/retained_calibrated_sequence.mkv"
    )
    materialized = tmp_path / "model-frames"
    materialized.mkdir()
    for index, artifact in enumerate(request["frame_artifacts"]):
        Path(artifact["path"]).replace(materialized / f"{index:06d}.jpg")
    _, _, _, blockers = _validate_request(request, materialized)
    assert blockers == []


def test_rejects_tampered_calibrated_image(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    (fixture["source"] / "front.png").write_bytes(b"tampered")
    with pytest.raises(PublicSceneSam31InputError, match="image_bytes_changed"):
        materialize_public_scene_sam31_task_inputs(
            calibrated_view_receipt_path=fixture["receipt"],
            task_freeze_path=fixture["task"],
            provider_profile_path=fixture["profile"],
            prompts_path=fixture["prompts"],
            output_root=tmp_path / "output",
            ffmpeg_executable=fixture["ffmpeg"],
        )


def test_resolves_explicit_anchor_camera_to_model_frame_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    fixture["prompts"].write_text(
        json.dumps(
            [
                {
                    "prompt_id": "washer",
                    "text": "washer",
                    "output_label": "washer",
                    "anchor_camera_id": "right",
                }
            ]
        ),
        encoding="utf-8",
    )

    def fake_encode(*, output_path: Path, **_: object) -> list[str]:
        output_path.write_bytes(b"lossless-sequence-fixture")
        return ["fixture-ffmpeg", "ffv1"]

    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_sam31_task_inputs._encode_lossless_sequence",
        fake_encode,
    )
    result = materialize_public_scene_sam31_task_inputs(
        calibrated_view_receipt_path=fixture["receipt"],
        task_freeze_path=fixture["task"],
        provider_profile_path=fixture["profile"],
        prompts_path=fixture["prompts"],
        output_root=tmp_path / "output",
        ffmpeg_executable=fixture["ffmpeg"],
    )

    request = json.loads(
        (tmp_path / "output/semantic_sam31_source_track_run_request.v1.json").read_text()
    )
    assert result["prompts"] == request["prompts"]
    assert request["prompts"] == [
        {
            "prompt_id": "washer",
            "text": "washer",
            "output_label": "washer",
            "anchor_camera_id": "right",
            "anchor_frame_index": 1,
        }
    ]


def test_rejects_unknown_prompt_anchor_camera(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["prompts"].write_text(
        json.dumps(
            [
                {
                    "prompt_id": "washer",
                    "text": "washer",
                    "output_label": "washer",
                    "anchor_camera_id": "missing",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(PublicSceneSam31InputError, match="prompt_anchor_unknown"):
        materialize_public_scene_sam31_task_inputs(
            calibrated_view_receipt_path=fixture["receipt"],
            task_freeze_path=fixture["task"],
            provider_profile_path=fixture["profile"],
            prompts_path=fixture["prompts"],
            output_root=tmp_path / "output",
            ffmpeg_executable=fixture["ffmpeg"],
        )
