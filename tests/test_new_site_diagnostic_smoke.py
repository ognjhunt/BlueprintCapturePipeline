from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from blueprint_pipeline.new_site_diagnostic_smoke import (
    EXPERIMENT_ID,
    assess_canary,
    build_protocol,
    build_ranking,
)
from blueprint_pipeline import new_site_diagnostic_smoke


ROOT = Path(__file__).resolve().parents[1]


def test_freeze_uses_development_scene_and_excludes_confirmation_sessions() -> None:
    protocol = build_protocol(ROOT)
    assert protocol["experiment_id"] == EXPERIMENT_ID
    assert protocol["scene"]["scene_id"].startswith("interiorgs_0787")
    assert len(protocol["policy_freeze"]["policies"]) == 3
    assert len(protocol["confirmation_exclusion"]["reserved_session_ids"]) == 17
    assert protocol["confirmation_exclusion"]["diagnostic_scene_outside_reserved_split"]
    assert protocol["arms"]["cosmos"]["admitted_arm_id"] is None
    assert protocol["paid_execution_admitted"] is False
    implementation_paths = {
        row["path"] for row in protocol["implementation"]["files"]
    }
    assert "src/blueprint_pipeline/new_site_diagnostic_canary_gpu.py" in implementation_paths
    assert "src/blueprint_pipeline/openpi_policy_ranking_gpu_admission.py" in implementation_paths


def test_freeze_hash_links_local_decoded_scene(tmp_path: Path) -> None:
    splat = tmp_path / "scene.ply"
    splat.write_bytes(b"ply\nsynthetic-test-fixture")
    protocol = build_protocol(ROOT, splat)
    scene_input = protocol["scene"]["local_scene_input"]
    assert scene_input["sha256"] == hashlib.sha256(splat.read_bytes()).hexdigest()
    assert scene_input["authoritative_source_replacement"] is False


def test_successor_version_requires_immediate_parent_and_disclosure() -> None:
    first = build_protocol(ROOT)
    second = build_protocol(
        ROOT,
        experiment_id=EXPERIMENT_ID.removesuffix("_v1") + "_v2",
        parent_protocol=first,
        disclosures=["v1 render blocked: splat_transform_cli_unavailable"],
    )
    assert second["lineage"]["parent_protocol_sha256"] == first["protocol_sha256"]
    assert second["lineage"]["version"] == 2
    with pytest.raises(
        ValueError, match="diagnostic_successor_requires_parent_and_disclosure"
    ):
        build_protocol(
            ROOT,
            experiment_id=EXPERIMENT_ID.removesuffix("_v1") + "_v2",
        )


def test_completed_scene_reference_is_hash_bound(tmp_path: Path) -> None:
    frame_paths = []
    for camera_id in ("external", "wrist"):
        frame = tmp_path / f"{camera_id}.png"
        frame.write_bytes(camera_id.encode())
        frame_paths.append(
            {
                "id": camera_id,
                "path": str(frame),
                "nonblank": True,
            }
        )
    video = tmp_path / "scene.mp4"
    video.write_bytes(b"scene-video")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "completed",
                "rendered_by": "reference_spark_renderer",
                "nonblank_camera_count": 2,
                "cameras": frame_paths,
                "mp4": {"mp4": str(video)},
                "proof_boundary": {"physics_or_navigation_proven": False},
            }
        ),
        encoding="utf-8",
    )
    protocol = build_protocol(ROOT, scene_reference_manifest=manifest)
    reference = protocol["scene"]["supporting_reference"]
    assert reference["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert reference["mp4"]["sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()


def test_canary_requires_all_three_label_free_gates() -> None:
    protocol = build_protocol(ROOT)
    evidence = {
        "arm_id": "oscar",
        "protocol_sha256": protocol["protocol_sha256"],
        "label_free": True,
        "ranking_outputs_accessed": False,
        "attempt_stage": "rollout",
        "model_invoked": True,
        "scene_id": protocol["scene"]["scene_id"],
        "task_instruction": protocol["scene"]["task_instruction"],
        "policy_id": protocol["canary_rule"]["frozen_policy_id"],
        "variant": "center",
        "observation_validity_passed": True,
        "motion_passed": True,
        "collapse_checks_passed": False,
        "blockers": ["first_future_frame_collapse"],
    }
    result = assess_canary(protocol, evidence)
    assert result["status"] == "failed"
    assert result["advance_to_complete_episodes"] is False
    with pytest.raises(ValueError, match="ranking_forbidden_for_failed_canary"):
        build_ranking(protocol, result, [], ROOT)


def test_ranking_reports_uncertainty_and_abstains_on_overlap(tmp_path: Path) -> None:
    protocol = build_protocol(ROOT)
    canary = assess_canary(
        protocol,
        {
            "arm_id": "skeleton_only",
            "protocol_sha256": protocol["protocol_sha256"],
            "label_free": True,
            "ranking_outputs_accessed": False,
            "attempt_stage": "rollout",
            "model_invoked": True,
            "scene_id": protocol["scene"]["scene_id"],
            "task_instruction": protocol["scene"]["task_instruction"],
            "policy_id": protocol["canary_rule"]["frozen_policy_id"],
            "variant": "center",
            "observation_validity_passed": True,
            "motion_passed": True,
            "collapse_checks_passed": True,
            "blockers": [],
        },
    )
    scores = {
        protocol["policy_freeze"]["policies"][0]["policy_id"]: [0.7, 0.5, 0.6],
        protocol["policy_freeze"]["policies"][1]["policy_id"]: [0.55, 0.45, 0.5],
        protocol["policy_freeze"]["policies"][2]["policy_id"]: [0.4, 0.3, 0.35],
    }
    episodes = []
    media_refs = {}
    for view in ("external", "wrist"):
        path = tmp_path / f"{view}.mp4"
        path.write_bytes(f"diagnostic-{view}".encode())
        media_refs[view] = {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    for policy_id, values in scores.items():
        for variant, score in zip(protocol["policy_freeze"]["variants"], values, strict=True):
            episodes.append(
                {
                    "arm_id": "skeleton_only",
                    "policy_id": policy_id,
                    "variant": variant,
                    "score": score,
                    "camera_videos": media_refs,
                    "failure_detected": False,
                    "failure_media": [],
                }
            )
    result = build_ranking(protocol, canary, episodes, tmp_path)
    assert result["abstained"] is True
    assert result["strict_total_ranking_emitted"] is False
    assert result["abstention_reasons"] == ["frozen_variant_intervals_overlap"]
    assert all(item["variant_count"] == 3 for item in result["policy_summaries"])


def test_cli_outputs_are_append_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = build_protocol(ROOT)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "new-site-smoke",
            "freeze",
            "--repo-root",
            str(ROOT),
            "--output",
            str(path),
        ],
    )
    with pytest.raises(FileExistsError, match="diagnostic_evidence_overwrite_forbidden"):
        new_site_diagnostic_smoke.main()


def test_ranking_rejects_duplicate_variant_even_when_count_is_three(tmp_path: Path) -> None:
    protocol = build_protocol(ROOT)
    canary_evidence = {
        "arm_id": "skeleton_only",
        "protocol_sha256": protocol["protocol_sha256"],
        "label_free": True,
        "ranking_outputs_accessed": False,
        "attempt_stage": "rollout",
        "model_invoked": True,
        "scene_id": protocol["scene"]["scene_id"],
        "task_instruction": protocol["scene"]["task_instruction"],
        "policy_id": protocol["canary_rule"]["frozen_policy_id"],
        "variant": "center",
        "observation_validity_passed": True,
        "motion_passed": True,
        "collapse_checks_passed": True,
        "blockers": [],
    }
    canary = assess_canary(protocol, canary_evidence)
    rows = []
    for policy in protocol["policy_freeze"]["policies"]:
        for _ in range(3):
            rows.append(
                {
                    "arm_id": "skeleton_only",
                    "policy_id": policy["policy_id"],
                    "variant": "center",
                    "score": 0.5,
                    "camera_videos": {},
                }
            )
    with pytest.raises(ValueError, match="ranking_duplicate_policy_variant"):
        build_ranking(protocol, canary, rows, tmp_path)


def test_pre_model_block_is_not_reported_as_rollout_failure() -> None:
    protocol = build_protocol(ROOT)
    result = assess_canary(
        protocol,
        {
            "arm_id": "oscar",
            "protocol_sha256": protocol["protocol_sha256"],
            "label_free": True,
            "ranking_outputs_accessed": False,
            "attempt_stage": "paid_admission",
            "model_invoked": False,
            "scene_id": protocol["scene"]["scene_id"],
            "task_instruction": protocol["scene"]["task_instruction"],
            "policy_id": protocol["canary_rule"]["frozen_policy_id"],
            "variant": "center",
            "observation_validity_passed": False,
            "motion_passed": False,
            "collapse_checks_passed": False,
            "blockers": ["exclusive_campaign_lease_unavailable"],
        },
    )
    assert result["status"] == "blocked_before_model"
    assert result["advance_to_complete_episodes"] is False
