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


def _grouped_lens_frames(root: Path, pair_count: int = 5) -> list[dict]:
    rows: list[dict] = []
    for pair_index in range(pair_count):
        for lens_id in ("front", "rear"):
            path = root / "lens-frames" / lens_id / f"pair-{pair_index:04d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{lens_id}-pair-{pair_index}".encode())
            rows.append(
                {
                    "frame_id": f"{lens_id}-pair-{pair_index:04d}",
                    "decoded_frame_index": pair_index,
                    "t_video_sec": pair_index * 0.033333,
                    "source_pts_seconds": pair_index * 0.033333,
                    "source_dts_seconds": pair_index * 0.033333,
                    "duration_seconds": 0.033333,
                    "key_frame": pair_index == 0,
                    "artifact_relative_path": path.relative_to(root).as_posix(),
                    "digest": _digest(path),
                    "image_metadata": {"width": 8, "height": 8},
                    "quality_signals": {"gradient_energy": 1.0},
                    "source_camera_identity": lens_id,
                    "observation_group_id": f"pair-{pair_index:04d}",
                }
            )
    return rows


def _compile(
    root: Path,
    frames: list[dict],
    *,
    timestamp: str = "2026-07-30T12:00:00Z",
    supporting_artifact_references: tuple[dict, ...] = (),
    source_video_references: tuple[dict, ...] | None = None,
    implementation_digest: str = IMPLEMENTATION_DIGEST,
    source_commit_sha: str = SOURCE_COMMIT,
) -> dict:
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
        implementation_digest=implementation_digest,
        source_commit_sha=source_commit_sha,
        rights_and_retention={"rights": "accepted", "external_processing_allowed": False},
        timestamp=timestamp,
        supporting_artifact_references=supporting_artifact_references,
        source_video_references=source_video_references,
    )


def _compile_grouped(root: Path, frames: list[dict]) -> dict:
    return compile_frozen_frame_dataset(
        artifact_root=root,
        intake_id="native-360-intake",
        capture_digest=CAPTURE_DIGEST,
        capture_authority_profile="camera_360_native",
        source_video_relative_path="native/capture.insv",
        source_video_digest="sha256:" + "e" * 64,
        decoded_frame_count=len(frames),
        selected_frames=frames,
        stream_metadata={
            "camera_representation": "calibrated_dual_fisheye_rig",
            "source_camera_identities": ["front", "rear"],
            "shared_physical_observation_groups": True,
        },
        runtime_identity="ffmpeg_dual_lens_fixture",
        runtime_digest=RUNTIME_DIGEST,
        implementation_digest=IMPLEMENTATION_DIGEST,
        source_commit_sha=SOURCE_COMMIT,
        rights_and_retention={
            "rights": "accepted",
            "external_processing_allowed": False,
        },
        parent_artifact={"dual_fisheye_binding_digest": "sha256:" + "f" * 64},
        timestamp="2026-07-30T12:00:00Z",
        camera_calibration_binding={
            "camera_360_rig_declaration_digest": "sha256:" + "1" * 64
        },
        coordinate_frame_declaration={
            "units": "meters",
            "handedness": "right_handed",
            "rig_frame": "front_lens_optical_center",
        },
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
    legacy_dataset = dict(first)
    legacy_dataset["producing_method"] = "deterministic_retained_frame_compiler.v1"
    validator.validate(legacy_dataset)


def test_compiler_preserves_complete_multisegment_source_set(tmp_path: Path) -> None:
    references = (
        {"relative_path": "retained/source.mov", "digest": "sha256:" + "e" * 64},
        {"relative_path": "retained/source_001.mov", "digest": "sha256:" + "f" * 64},
    )

    dataset = _compile(
        tmp_path,
        _frames(tmp_path),
        source_video_references=references,
    )
    selection = _load_ref(tmp_path, dataset, "retained_frame_selection_manifest")

    assert dataset["original_file_references"] == list(references)
    assert selection["source_video_references"] == list(references)
    assert dataset["input_digests"]["source_video_reference_set_digest"] == (
        dataset["deterministic_configuration"]["source_video_reference_set_digest"]
    )
    jsonschema.Draft202012Validator(_schema()).validate(dataset)
    jsonschema.Draft202012Validator(_schema()).validate(selection)


@pytest.mark.parametrize(
    "references, expected",
    [
        (
            (
                {"relative_path": "retained/source.mov", "digest": "sha256:" + "e" * 64},
                {"relative_path": "retained/source.mov", "digest": "sha256:" + "f" * 64},
            ),
            "dataset_source_video_reference_invalid:1",
        ),
        (
            ({"relative_path": "../escape.mov", "digest": "sha256:" + "e" * 64},),
            "dataset_source_video_reference_invalid:0",
        ),
        (
            ({"relative_path": "retained/other.mov", "digest": "sha256:" + "f" * 64},),
            "dataset_primary_source_video_reference_missing",
        ),
    ],
)
def test_compiler_rejects_invalid_multisegment_source_sets(
    tmp_path: Path, references: tuple[dict, ...], expected: str
) -> None:
    with pytest.raises(ReconstructionFrameDatasetError, match=expected):
        _compile(
            tmp_path,
            _frames(tmp_path),
            source_video_references=references,
        )


def test_compiler_freezes_synchronized_camera_groups_without_counterpart_leakage(
    tmp_path: Path,
) -> None:
    dataset = _compile_grouped(tmp_path, _grouped_lens_frames(tmp_path))
    split = _load_ref(tmp_path, dataset, "frozen_split_manifest")
    candidate = _load_ref(tmp_path, dataset, "candidate_dataset_manifest")
    heldout = _load_ref(tmp_path, dataset, "hidden_heldout_evaluator_manifest")
    selection = _load_ref(tmp_path, dataset, "retained_frame_selection_manifest")

    splits_by_group: dict[str, set[str]] = {}
    cameras_by_group: dict[str, set[str]] = {}
    for row in split["assignments"]:
        splits_by_group.setdefault(row["observation_group_id"], set()).add(
            row["split"]
        )
        cameras_by_group.setdefault(row["observation_group_id"], set()).add(
            row["source_camera_identity"]
        )
    assert all(values == {"front", "rear"} for values in cameras_by_group.values())
    assert all(len(values) == 1 for values in splits_by_group.values())

    candidate_groups = {row["observation_group_id"] for row in candidate["frames"]}
    heldout_groups = {row["observation_group_id"] for row in heldout["frames"]}
    assert heldout_groups
    assert candidate_groups.isdisjoint(heldout_groups)
    assert {
        row["source_camera_identity"]
        for row in heldout["frames"]
        if row["observation_group_id"] in heldout_groups
    } == {"front", "rear"}
    assert dataset["candidate_dataset_contains_hidden_heldout_pixels"] is False
    assert dataset["claim_ceiling"] == "decoded_observation_availability"
    assert dataset["camera_calibration_binding"] == {
        "camera_360_rig_declaration_digest": "sha256:" + "1" * 64
    }
    assert dataset["coordinate_frame_declaration"]["units"] == "meters"

    validator = jsonschema.Draft202012Validator(_schema())
    for artifact in (dataset, split, candidate, heldout, selection):
        validator.validate(artifact)


def test_compiler_allows_equal_pts_across_cameras_but_rejects_incomplete_groups(
    tmp_path: Path,
) -> None:
    frames = _grouped_lens_frames(tmp_path)
    dataset = _compile_grouped(tmp_path, frames)
    assert dataset["blockers"] == []

    incomplete_root = tmp_path / "incomplete"
    incomplete = [dict(row) for row in _grouped_lens_frames(incomplete_root)]
    incomplete[0].pop("observation_group_id")
    with pytest.raises(
        ReconstructionFrameDatasetError,
        match="selected_frame_group_binding_incomplete",
    ):
        _compile_grouped(incomplete_root, incomplete)

    duplicate_root = tmp_path / "duplicate-camera"
    duplicate = _grouped_lens_frames(duplicate_root)
    duplicate[1]["source_camera_identity"] = "front"
    with pytest.raises(
        ReconstructionFrameDatasetError,
        match="selected_frame_group_camera_duplicate",
    ):
        _compile_grouped(duplicate_root, duplicate)


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


def test_compiler_binds_supporting_artifact_and_rejects_tamper(tmp_path: Path) -> None:
    support = tmp_path / "support/decode.json"
    support.parent.mkdir(parents=True)
    support.write_text('{"schema_version":"frame_decode_receipt.v1"}\n')
    reference = {
        "relative_path": support.relative_to(tmp_path).as_posix(),
        "digest": _digest(support),
        "artifact_type": "frame_decode_receipt.v1",
    }
    dataset = _compile(
        tmp_path,
        _frames(tmp_path),
        supporting_artifact_references=(reference,),
    )
    assert dataset["supporting_artifact_references"] == [reference]

    support.write_text("tampered\n")
    with pytest.raises(
        ReconstructionFrameDatasetError,
        match="dataset_supporting_artifact_invalid",
    ):
        _compile(
            tmp_path,
            _frames(tmp_path),
            supporting_artifact_references=(reference,),
        )


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


def test_split_membership_is_stable_across_compiler_provenance_changes(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path)
    first = _compile(tmp_path, frames)
    changed = _compile(
        tmp_path,
        frames,
        implementation_digest="sha256:" + "9" * 64,
        source_commit_sha="8" * 40,
    )
    first_split = _load_ref(tmp_path, first, "frozen_split_manifest")
    changed_split = _load_ref(tmp_path, changed, "frozen_split_manifest")

    assert first["deterministic_configuration_digest"] != changed[
        "deterministic_configuration_digest"
    ]
    assert first["dataset_manifest_digest"] != changed["dataset_manifest_digest"]
    assert first_split == changed_split
    assert first["train_heldout_split_digest"] == changed[
        "train_heldout_split_digest"
    ]


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
