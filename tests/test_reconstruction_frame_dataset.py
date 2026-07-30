from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.reconstruction_frame_dataset import (
    ReconstructionFrameDatasetError,
    compile_frozen_frame_dataset,
)


CAPTURE_DIGEST = "sha256:" + "a" * 64
RUNTIME_DIGEST = "sha256:" + "b" * 64
IMPLEMENTATION_DIGEST = "sha256:" + "c" * 64
SOURCE_COMMIT = "d" * 40


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _frames(root: Path, count: int = 10) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        path = root / "frames" / f"decoded-{index:09d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"frame-{index}".encode())
        rows.append(
            {
                "frame_id": f"decoded-{index:09d}",
                "decoded_frame_index": index,
                # Deliberately variable-rate to prove splits bind actual PTS.
                "t_video_sec": [0.0, 0.041, 0.083, 0.15, 0.231, 0.31, 0.45, 0.57, 0.8, 1.1][
                    index
                ],
                "source_pts_seconds": 4.0 + index,
                "source_dts_seconds": None,
                "duration_seconds": None,
                "key_frame": index in {0, 9},
                "artifact_relative_path": path.relative_to(root).as_posix(),
                "digest": _digest(path),
                "image_metadata": {
                    "width": 8,
                    "height": 8,
                    "pixel_orientation": "encoded_source_no_autorotate",
                },
                "quality_signals": {
                    "mean_luma_0_255": float(index),
                    "gradient_energy": float(index + 1),
                },
            }
        )
    return rows


def _compile(root: Path, frames: list[dict], *, timestamp: str = "2026-07-30T12:00:00Z") -> dict:
    return compile_frozen_frame_dataset(
        artifact_root=root,
        intake_id="intake-1",
        capture_digest=CAPTURE_DIGEST,
        capture_authority_profile="camera_360_equirectangular",
        source_video_relative_path="retained/source.mov",
        source_video_digest="sha256:" + "e" * 64,
        decoded_frame_count=24,
        selected_frames=frames,
        stream_metadata={
            "width": 3840,
            "height": 1920,
            "pix_fmt": "yuv420p10le",
            "display_rotation_degrees": 90.0,
        },
        runtime_identity="ffmpeg_ffprobe_local",
        runtime_digest=RUNTIME_DIGEST,
        implementation_digest=IMPLEMENTATION_DIGEST,
        source_commit_sha=SOURCE_COMMIT,
        rights_and_retention={"rights": "accepted", "external_processing_allowed": False},
        timestamp=timestamp,
    )


def _load_ref(root: Path, dataset: dict, name: str) -> dict:
    relative = dataset["artifact_references"][name]["relative_path"]
    return json.loads((root / relative).read_text(encoding="utf-8"))


def _schema() -> dict:
    return json.loads(
        (
            Path(__file__).parents[1]
            / "docs"
            / "schemas"
            / "reconstruction_frame_dataset.v1.schema.json"
        ).read_text(encoding="utf-8")
    )


def test_compiler_is_idempotent_and_isolates_hidden_heldout_pixels(tmp_path: Path) -> None:
    frames = _frames(tmp_path)

    first = _compile(tmp_path, frames)
    second = _compile(tmp_path, frames, timestamp="2099-01-01T00:00:00Z")

    assert first == second
    assert first["timestamp"] == "2026-07-30T12:00:00Z"
    assert first["raw_capture_bytes_remain_authoritative"] is True
    assert first["claim_ceiling"] == "decoded_observation_availability"
    assert first["candidate_dataset_contains_hidden_heldout_pixels"] is False
    split = _load_ref(tmp_path, first, "frozen_split_manifest")
    candidate = _load_ref(tmp_path, first, "candidate_dataset_manifest")
    heldout = _load_ref(tmp_path, first, "hidden_heldout_evaluator_manifest")
    assignments = {row["frame_id"]: row["split"] for row in split["assignments"]}
    candidate_ids = {row["frame_id"] for row in candidate["frames"]}
    heldout_ids = {row["frame_id"] for row in heldout["frames"]}
    assert {assignments[frame_id] for frame_id in candidate_ids} == {"training", "validation"}
    assert heldout_ids
    assert candidate_ids.isdisjoint(heldout_ids)
    assert all("held_out" not in row["candidate_relative_path"] for row in candidate["frames"])
    assert heldout["candidate_method_access_allowed"] is False
    assert split["candidate_can_change_assignments"] is False

    validator = jsonschema.Draft202012Validator(_schema())
    for artifact in (first, split, candidate, heldout, _load_ref(tmp_path, first, "retained_frame_selection_manifest")):
        validator.validate(artifact)


def test_compiler_replays_after_manifest_write_interruption(tmp_path: Path) -> None:
    frames = _frames(tmp_path)
    first = _compile(tmp_path, frames)
    manifest_path = next(tmp_path.glob("frozen_dataset_*/reconstruction_dataset_manifest.json"))
    manifest_path.unlink()

    replay = _compile(tmp_path, frames)

    assert replay == first
    candidate = _load_ref(tmp_path, replay, "candidate_dataset_manifest")
    assert candidate["candidate_dataset_digest"] == first["output_digests"][
        "candidate_dataset_digest"
    ]


def test_compiler_replay_rejects_tampered_split_artifact(tmp_path: Path) -> None:
    frames = _frames(tmp_path)
    dataset = _compile(tmp_path, frames)
    split_path = tmp_path / dataset["artifact_references"]["frozen_split_manifest"][
        "relative_path"
    ]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["assignments"][0]["split"] = "training"
    split_path.write_text(json.dumps(split), encoding="utf-8")

    with pytest.raises(ReconstructionFrameDatasetError, match="artifact_digest_mismatch"):
        _compile(tmp_path, frames)


def test_compiler_configuration_binds_pts_and_stream_metadata(tmp_path: Path) -> None:
    frames = _frames(tmp_path)
    first = _compile(tmp_path, frames)
    changed = [dict(row) for row in frames]
    changed[1]["t_video_sec"] = 0.05

    second = _compile(tmp_path, changed)

    assert first["deterministic_configuration_digest"] != second[
        "deterministic_configuration_digest"
    ]
    assert first["dataset_manifest_digest"] != second["dataset_manifest_digest"]


def test_compiler_rejects_duplicate_pts_and_frame_path_escape(tmp_path: Path) -> None:
    frames = _frames(tmp_path, count=4)
    frames[1]["t_video_sec"] = frames[0]["t_video_sec"]
    with pytest.raises(ReconstructionFrameDatasetError, match="pts_invalid_or_duplicate"):
        _compile(tmp_path, frames)

    frames = _frames(tmp_path, count=4)
    frames[0]["artifact_relative_path"] = "../private.png"
    with pytest.raises(ReconstructionFrameDatasetError, match="relative_path_unsafe"):
        _compile(tmp_path, frames)


def test_compiler_rejects_symlinked_selected_frame(tmp_path: Path) -> None:
    frames = _frames(tmp_path, count=4)
    external = tmp_path / "external.png"
    external.write_bytes(b"external")
    selected = tmp_path / frames[0]["artifact_relative_path"]
    selected.unlink()
    selected.symlink_to(external)
    frames[0]["digest"] = _digest(external)

    with pytest.raises(
        ReconstructionFrameDatasetError,
        match="selected_frame_source_(invalid|digest_mismatch)",
    ):
        _compile(tmp_path, frames)


def test_compiler_records_typed_blocker_when_disjoint_split_is_impossible(
    tmp_path: Path,
) -> None:
    dataset = _compile(tmp_path, _frames(tmp_path, count=2))
    split = _load_ref(tmp_path, dataset, "frozen_split_manifest")

    assert dataset["blockers"] == [
        "insufficient_selected_frames_for_disjoint_validation_and_hidden_heldout"
    ]
    assert {row["split"] for row in split["assignments"]} == {"training"}
    assert dataset["output_digests"]["hidden_heldout_digest"]
