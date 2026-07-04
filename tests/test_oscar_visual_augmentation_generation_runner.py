from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

from blueprint_pipeline.oscar_visual_augmentation_generation_runner import (
    main,
    run_visual_augmentation_generation,
)
from blueprint_pipeline.oscar_visual_augmentation_packet import (
    build_oscar_visual_augmentation_packet,
)


pytestmark = [pytest.mark.slow, pytest.mark.integration]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_bytes(path: Path, payload: bytes = b"asset") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "storage" / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    (capture_root / "raw").mkdir(parents=True)
    _write_json(
        capture_root / "capture_descriptor.json",
        {"scene_id": "scene-1", "capture_id": "capture-1"},
    )
    _write_json(
        capture_root / "raw" / "manifest.json",
        {"scene_id": "scene-1", "capture_id": "capture-1"},
    )
    return capture_root


def _write_reviewable_first_frame(path: Path) -> None:
    pytest.importorskip("cv2")
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(0, 255, 640, dtype=np.uint8)[None, :]
    frame[:, :, 1] = np.linspace(255, 0, 480, dtype=np.uint8)[:, None]
    for offset in range(0, 640, 48):
        cv2.line(frame, (offset, 0), (639 - offset // 2, 479), (255, 255, 255), 2)
    cv2.rectangle(frame, (250, 260), (360, 370), (30, 220, 80), -1)
    cv2.imwrite(str(path), frame)


def _ready_packet(tmp_path: Path, *, variant_count: int = 2) -> tuple[Path, Path]:
    capture_root = _capture_root(tmp_path)
    job_dir = tmp_path / "job"
    first_frame = job_dir / "first_frame.png"
    skeleton_video = job_dir / "skeleton.mp4"
    camera = job_dir / "camera_calibration_quality_gate.json"
    skeleton_trace = job_dir / "g1_projected_skeleton_trace.jsonl"
    variants = tmp_path / "variants.json"
    output_dir = job_dir / "oscar_visual_augmentation_packet"
    _write_reviewable_first_frame(first_frame)
    _write_bytes(skeleton_video, b"mp4")
    _write_json(camera, {"schema_version": "camera_calibration_quality_gate.v1", "status": "ready"})
    skeleton_trace.write_text('{"projected_landmark_count": 3}\n', encoding="utf-8")
    _write_json(
        variants,
        {
            "variants": [
                {
                    "variant_id": f"variant_{index}",
                    "prompt": f"same motion in environment {index}",
                    "environment_tags": ["fixture", f"env_{index}"],
                }
                for index in range(variant_count)
            ]
        },
    )
    manifest = build_oscar_visual_augmentation_packet(
        capture_root=capture_root,
        job_dir=job_dir,
        output_dir=output_dir,
        first_frame=first_frame,
        skeleton_video=skeleton_video,
        camera_provenance=camera,
        skeleton_provenance=skeleton_trace,
        variant_specs=variants,
    )
    assert manifest["status"] == "ready_for_model_backend_generation"
    return output_dir / "oscar_visual_augmentation_packet_manifest.json", output_dir


def test_visual_augmentation_runner_fixture_outputs_are_plumbing_not_training_data(
    tmp_path: Path,
) -> None:
    packet_manifest, packet_dir = _ready_packet(tmp_path, variant_count=2)

    result = run_visual_augmentation_generation(
        packet_manifest=packet_manifest,
        backend_mode="fixture",
        allow_fixture_backend=True,
    )

    assert result["status"] == "completed_fixture_test_outputs"
    assert result["generated_video_count"] == 2
    assert result["fixture_backend_used"] is True
    assert result["refreshed_packet_status"] == "completed_with_generated_support_videos_pending_model_truth"
    assert result["claim_boundary"]["generation_run_is_not_contact_physics_proof"] is True
    readiness = json.loads(
        (packet_dir / "visual_augmentation_training_readiness_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert readiness["status"] == "fixture_plumbing_only_not_training_data"
    assert readiness["training_ready_without_review"] is False
    assert readiness["usable_for_policy_pretraining_without_real_or_sim_truth_mix"] is False
    refreshed_packet = json.loads(packet_manifest.read_text(encoding="utf-8"))
    assert refreshed_packet["generated_video_count"] == 2
    assert refreshed_packet["generated_videos"][0]["model_derived"] is False
    assert refreshed_packet["generated_videos"][0]["raw_capture_evidence"] is False


def test_visual_augmentation_runner_command_backend_attaches_model_derived_outputs(
    tmp_path: Path,
) -> None:
    pytest.importorskip("cv2")
    packet_manifest, packet_dir = _ready_packet(tmp_path, variant_count=1)
    backend_script = tmp_path / "fake_visual_backend.py"
    backend_script.write_text(
        """
from __future__ import annotations
import json
import os
from pathlib import Path
import cv2
import numpy as np

request = json.loads(Path(os.environ["BLUEPRINT_VISUAL_AUGMENTATION_REQUEST"]).read_text())
output_video = Path(os.environ["BLUEPRINT_VISUAL_AUGMENTATION_OUTPUT_VIDEO"])
output_result = Path(os.environ["BLUEPRINT_VISUAL_AUGMENTATION_OUTPUT"])
output_video.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (640, 480))
for frame_index in range(24):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(0, 255, 640, dtype=np.uint8)[None, :]
    frame[:, :, 1] = np.linspace(255, 0, 480, dtype=np.uint8)[:, None]
    frame[:, :, 2] = (frame_index * 9) % 255
    for offset in range(0, 640, 42):
        cv2.line(frame, (offset, 0), (639 - offset // 2, 479), (255, 255, 255), 2)
    cv2.rectangle(frame, (220 + frame_index * 3, 260), (320 + frame_index * 3, 360), (30, 220, 80), -1)
    writer.write(frame)
writer.release()
output_result.write_text(json.dumps({
    "status": "completed",
    "variant_id": request["variant_id"],
    "generated_video_path": str(output_video),
    "model_derived": True,
    "learned_model_ran": True,
    "truth_boundary": {
        "generated_video_is_model_output": True,
        "contact_physics_proven": False,
        "deployment_safety_proven": False
    },
    "raw_credentials_written_to_artifacts": False,
    "secret_hashes_written_to_artifacts": False
}, indent=2))
""",
        encoding="utf-8",
    )
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(backend_script))}"

    result = run_visual_augmentation_generation(
        packet_manifest=packet_manifest,
        backend_mode="command",
        backend_command=command,
        timeout_seconds=30,
    )

    assert result["status"] == "completed_with_model_derived_outputs"
    assert result["generated_video_count"] == 1
    assert result["refreshed_packet_status"] == "completed_with_model_derived_generated_videos"
    readiness = json.loads(
        (packet_dir / "visual_augmentation_training_readiness_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert readiness["status"] == "review_ready_model_derived_training_candidate"
    assert readiness["training_ready_without_review"] is False
    dataset = json.loads(
        (packet_dir / "visual_augmentation_training_dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert dataset["status"] == "candidate_dataset_written_requires_review"
    refreshed_packet = json.loads(packet_manifest.read_text(encoding="utf-8"))
    assert refreshed_packet["generated_videos"][0]["model_derived"] is True
    assert refreshed_packet["generated_videos"][0]["contact_physics_proven"] is False


def test_visual_augmentation_runner_blocks_without_backend_or_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet_manifest, _packet_dir = _ready_packet(tmp_path, variant_count=1)
    monkeypatch.delenv("BLUEPRINT_VISUAL_AUGMENTATION_BACKEND_COMMAND", raising=False)
    monkeypatch.delenv("BLUEPRINT_OSCAR_VISUAL_AUGMENTATION_COMMAND", raising=False)
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_COMMAND", raising=False)
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_PROVIDER_COMMAND", raising=False)

    result = run_visual_augmentation_generation(packet_manifest=packet_manifest)

    assert result["status"] == "blocked"
    assert "backend_command_missing_and_fixture_not_authorized" in result["blockers"]
    assert result["generated_video_count"] == 0


def test_visual_augmentation_generation_cli_fixture_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    packet_manifest, _packet_dir = _ready_packet(tmp_path, variant_count=1)

    assert (
        main(
            [
                "--packet-manifest",
                str(packet_manifest),
                "--backend-mode",
                "fixture",
                "--allow-fixture-backend",
            ]
        )
        == 0
    )
    assert "status=completed_fixture_test_outputs" in capsys.readouterr().out
