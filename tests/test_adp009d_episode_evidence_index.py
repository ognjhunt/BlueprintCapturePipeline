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
    materialize_supporting_evidence_inventory,
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


def test_portable_index_represents_terminal_abstention_without_fake_episodes(
    tmp_path: Path,
) -> None:
    abstention = {
        "schema_version": "adp_task_evaluation_run_abstention.v1",
        "status": "typed_evidence_backed_abstention",
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "smallest_missing_capability": "joint_agent_local_ovrtx_renderer_not_ready",
        "controls_executed": False,
        "learned_candidate_episodes_executed": False,
        "receipt_digest": "",
    }
    abstention["receipt_digest"] = canonical_digest(
        abstention, digest_field="receipt_digest"
    )

    result = materialize_episode_evidence_index(
        run_root=tmp_path,
        episode_receipt_paths=[],
        run_identity={
            "scene_id": "840796",
            "task_id": "upper_refrigerator_door_open",
            "scenario_suite_digest": "not_materialized_before_abstention",
        },
        abstention_receipt=abstention,
    )

    assert result["index"]["episode_count"] == 0
    assert result["index"]["typed_abstention"] == abstention
    html = (tmp_path / HTML_FILENAME).read_text(encoding="utf-8")
    assert "No control or learned-policy episode exists" in html
    assert "joint_agent_local_ovrtx_renderer_not_ready" in html


def test_abstention_index_verifies_and_links_supporting_receipts(
    tmp_path: Path,
) -> None:
    external_root = tmp_path / "external"
    package_root = tmp_path / "package"
    external_root.mkdir()
    package_root.mkdir()
    mask = _artifact(external_root, "removal/masks/front.png", b"mask")
    teardown = _artifact(
        external_root,
        "removal/run/teardown.json",
        b'{"schema_version":"vast_teardown_manifest.v1"}\n',
    )
    inventory = materialize_supporting_evidence_inventory(
        source_root=external_root,
        output_root=package_root,
        output_relative_path="supporting_evidence_inventory.v1.json",
        source_root_id="rights_bounded_construction_root",
        artifacts=[
            {"role": "source_mask", **mask},
            {"role": "paid_teardown", **teardown},
        ],
        disclosure_class="digest_receipt_only",
    )
    recovery = {
        "schema_version": "adp_gaussian_excision_recovery_readiness.v1",
        "status": "ready_for_new_authority_not_executed",
        "receipt_digest": "",
    }
    recovery["receipt_digest"] = canonical_digest(
        recovery, digest_field="receipt_digest"
    )
    recovery_path = package_root / "recovery.json"
    recovery_path.write_text(json.dumps(recovery), encoding="utf-8")
    abstention = {
        "schema_version": "adp_task_evaluation_run_abstention.v1",
        "status": "typed_evidence_backed_abstention",
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "smallest_missing_capability": (
            "fresh_paid_authority_for_qualified_gaussian_contribution_missing"
        ),
        "controls_executed": False,
        "learned_candidate_episodes_executed": False,
        "receipt_digest": "",
    }
    abstention["receipt_digest"] = canonical_digest(
        abstention, digest_field="receipt_digest"
    )

    result = materialize_episode_evidence_index(
        run_root=package_root,
        episode_receipt_paths=[],
        run_identity={
            "scene_id": "fixture_scene",
            "task_id": "fixture_task",
            "scenario_suite_digest": "sha256:frozen-suite",
        },
        abstention_receipt=abstention,
        supporting_receipt_paths=[
            "supporting_evidence_inventory.v1.json",
            "recovery.json",
        ],
    )

    assert inventory["artifact_count"] == 2
    assert len(result["index"]["supporting_evidence"]) == 2
    html = (package_root / HTML_FILENAME).read_text(encoding="utf-8")
    assert "Supporting construction evidence" in html
    assert "supporting_evidence_inventory.v1.json" in html


def test_supporting_inventory_rejects_tampered_external_artifact(
    tmp_path: Path,
) -> None:
    external_root = tmp_path / "external"
    package_root = tmp_path / "package"
    external_root.mkdir()
    package_root.mkdir()
    artifact = _artifact(external_root, "mask.png", b"mask")
    (external_root / "mask.png").write_bytes(b"tampered")

    with pytest.raises(
        EpisodeEvidenceIndexError,
        match="supporting_evidence_artifact_(digest|size)_mismatch:source_mask",
    ):
        materialize_supporting_evidence_inventory(
            source_root=external_root,
            output_root=package_root,
            output_relative_path="supporting_evidence_inventory.v1.json",
            source_root_id="rights_bounded_construction_root",
            artifacts=[{"role": "source_mask", **artifact}],
            disclosure_class="digest_receipt_only",
        )


def test_supporting_inventory_and_index_reject_symlinks(tmp_path: Path) -> None:
    external_root = tmp_path / "external"
    package_root = tmp_path / "package"
    external_root.mkdir()
    package_root.mkdir()
    target = external_root / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = external_root / "linked.json"
    link.symlink_to(target)
    record = {
        "role": "linked_receipt",
        "relative_path": "linked.json",
        "sha256": "sha256:" + _sha256(target),
        "size_bytes": target.stat().st_size,
    }

    with pytest.raises(
        EpisodeEvidenceIndexError, match="episode_artifact_symlink_forbidden"
    ):
        materialize_supporting_evidence_inventory(
            source_root=external_root,
            output_root=package_root,
            output_relative_path="inventory.json",
            source_root_id="fixture_root",
            artifacts=[record],
            disclosure_class="digest_receipt_only",
        )

    with pytest.raises(
        EpisodeEvidenceIndexError, match="supporting_receipt_symlink_forbidden"
    ):
        abstention = {
            "schema_version": "adp_task_evaluation_run_abstention.v1",
            "status": "typed_evidence_backed_abstention",
            "smallest_missing_capability": "fixture_blocker",
            "controls_executed": False,
            "learned_candidate_episodes_executed": False,
            "candidate_ids": ["pi05_droid", "groot_n17_droid"],
            "receipt_digest": "",
        }
        abstention["receipt_digest"] = canonical_digest(
            abstention, digest_field="receipt_digest"
        )
        materialize_episode_evidence_index(
            run_root=external_root,
            episode_receipt_paths=[],
            run_identity={
                "scene_id": "fixture_scene",
                "task_id": "fixture_task",
                "scenario_suite_digest": "sha256:frozen-suite",
            },
            abstention_receipt=abstention,
            supporting_receipt_paths=["linked.json"],
        )
