from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.adp_episode_evidence_index import (
    EpisodeEvidenceIndexError,
    HTML_FILENAME,
    INDEX_FILENAME,
    materialize_episode_evidence_index,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(root: Path, relative_path: str, content: bytes) -> dict[str, object]:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "relative_path": relative_path,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _receipt(
    root: Path,
    *,
    episode_id: str,
    subject_id: str,
    learned: bool,
) -> Path:
    artifacts = []
    manifest = _artifact(
        root,
        f"media/{episode_id}/multicamera_frame_manifest.json",
        b'{"schema_version":"fixture"}\n',
    )
    artifacts.append(
        {"role": "multicamera_observation_frame_manifest", **manifest}
    )
    videos = {}
    for camera_id in ("external", "wrist", "overview"):
        artifact = _artifact(
            root, f"media/{episode_id}/{camera_id}.mp4", camera_id.encode("utf-8")
        )
        artifacts.append(
            {"role": "camera_review_video", "camera_id": camera_id, **artifact}
        )
        videos[camera_id] = {
            **artifact,
            "camera_id": camera_id,
            "derived_from_frame_manifest_digest": "sha256:fixture",
        }
    receipt = {
        "schema_version": (
            "adp009d_policy_episode.v3" if learned else "adp009d_control_episode.v2"
        ),
        "episode_id": episode_id,
        ("candidate_id" if learned else "control_id"): subject_id,
        "score": {
            "status": "scored",
            "outcome": "placed" if learned else "never_moved",
            "task_succeeded": learned,
            "outcome_rank": 4 if learned else 0,
        },
        "grader_authority": "deterministic_simulator_state",
        "visual_evidence": {
            "status": "complete",
            "required_camera_ids": ["external", "wrist", "overview"],
            "review_only_camera_ids": ["overview"],
            "videos": videos,
        },
        "media_artifacts": artifacts,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    path = root / "receipts" / f"{episode_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("scene_id", "task_id"),
    [
        ("840313", "canned_beverage_pick_place"),
        ("840796", "upper_refrigerator_door_open"),
    ],
)
def test_portable_episode_index_covers_original_and_second_scene_fixtures(
    tmp_path: Path, scene_id: str, task_id: str
) -> None:
    zero = _receipt(
        tmp_path,
        episode_id=f"{scene_id}-canonical-zero",
        subject_id="zero_action_negative",
        learned=False,
    )
    learned = _receipt(
        tmp_path,
        episode_id=f"{scene_id}-canonical-pi05",
        subject_id="pi05_droid",
        learned=True,
    )
    result = materialize_episode_evidence_index(
        run_root=tmp_path,
        episode_receipt_paths=[learned, zero],
        run_identity={
            "scene_id": scene_id,
            "task_id": task_id,
            "scenario_suite_digest": "sha256:frozen-suite",
        },
    )

    assert result["index"]["episode_count"] == 2
    assert result["index"]["run_identity"]["scene_id"] == scene_id
    assert result["index"]["overview_is_review_only"] is True
    assert (tmp_path / INDEX_FILENAME).is_file()
    html = (tmp_path / HTML_FILENAME).read_text(encoding="utf-8")
    assert f"media/{scene_id}-canonical-pi05/external.mp4" in html
    assert f"media/{scene_id}-canonical-pi05/wrist.mp4" in html
    assert f"media/{scene_id}-canonical-pi05/overview.mp4" in html
    assert "deterministic simulator state" in html


def test_portable_episode_index_rejects_tampered_video(tmp_path: Path) -> None:
    receipt = _receipt(
        tmp_path,
        episode_id="canonical-pi05",
        subject_id="pi05_droid",
        learned=True,
    )
    (tmp_path / "media/canonical-pi05/wrist.mp4").write_bytes(b"tampered")

    with pytest.raises(
        EpisodeEvidenceIndexError,
        match="episode_artifact_(digest|size)_mismatch:canonical-pi05:wrist",
    ):
        materialize_episode_evidence_index(
            run_root=tmp_path,
            episode_receipt_paths=[receipt],
            run_identity={
                "scene_id": "840796",
                "task_id": "upper_refrigerator_door_open",
                "scenario_suite_digest": "sha256:frozen-suite",
            },
        )


def test_portable_episode_index_rejects_overview_as_policy_input(tmp_path: Path) -> None:
    receipt_path = _receipt(
        tmp_path,
        episode_id="canonical-groot",
        subject_id="groot_n17_droid",
        learned=True,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["visual_evidence"]["review_only_camera_ids"] = []
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(
        EpisodeEvidenceIndexError,
        match="episode_overview_not_review_only:canonical-groot",
    ):
        materialize_episode_evidence_index(
            run_root=tmp_path,
            episode_receipt_paths=[receipt_path],
            run_identity={
                "scene_id": "840796",
                "task_id": "upper_refrigerator_door_open",
                "scenario_suite_digest": "sha256:frozen-suite",
            },
        )
