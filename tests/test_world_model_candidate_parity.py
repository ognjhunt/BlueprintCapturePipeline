from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

import blueprint_pipeline.geometry_stage as geometry_stage
import blueprint_pipeline.materialization as materialization
from blueprint_pipeline.alpha_readiness import build_alpha_readiness_summary
from blueprint_pipeline.capture_bridge import CaptureDescriptor
from blueprint_pipeline.local_capture import LocalCaptureContext
from blueprint_pipeline.materialization import materialize_capture_bundle
from blueprint_pipeline.qualification import _should_run_default_geometry_stage


def _rehash_raw_bundle(raw_root: Path) -> None:
    artifacts = {
        path.relative_to(raw_root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(raw_root.rglob("*"))
        if path.is_file() and path.name != "hashes.json"
    }
    canonical = "\n".join(f"{name}:{artifacts[name]}" for name in sorted(artifacts))
    (raw_root / "hashes.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "bundle_sha256": sha256(canonical.encode("utf-8")).hexdigest(),
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )


def _write_capture(
    root: Path,
    *,
    frame_rows: list[dict[str, object]],
    pose_rows: list[dict[str, object]],
    include_site_identity: bool = True,
) -> dict[str, object]:
    bucket = "local-blueprint"
    scene_id = "scene-parity"
    capture_id = "capture-parity"
    raw_root = root / bucket / "scenes" / scene_id / "captures" / capture_id / "raw"
    raw_root.mkdir(parents=True)
    (raw_root / "arkit" / "depth").mkdir(parents=True)

    manifest = {
        "schema_version": "v3",
        "scene_id": scene_id,
        "capture_id": capture_id,
        "video_uri": "walkthrough.mov",
        "width": 1920,
        "height": 1080,
        "device_model": "iPhone 15 Pro",
        "os_version": "18.0",
        "fps_source": 30.0,
        "capture_start_epoch_ms": 1_700_000_000_000,
        "has_lidar": True,
        "capture_schema_version": "3.0.0",
        "capture_source": "iphone",
        "capture_tier_hint": "tier1_iphone",
        "capture_profile_id": "iphone_arkit_lidar",
        "capture_capabilities": {"camera_pose": True, "camera_intrinsics": True, "depth": True},
        "coordinate_frame_session_id": "cfs-parity-1",
        "app_version": "1.0.0",
        "app_build": "1",
        "ios_version": "18.0",
        "ios_build": "22A",
        "hardware_model_identifier": "iPhone16,1",
        "device_model_marketing": "iPhone 15 Pro",
        "depth_supported": True,
        "rights_profile": "documented_permission",
        "requested_outputs": ["qualification", "preview_simulation", "deeper_evaluation"],
        "capture_rights": {
            "derived_scene_generation_allowed": True,
            "data_licensing_allowed": False,
            "capture_contributor_payout_eligible": True,
            "consent_status": "documented",
            "consent_scope": ["alpha-site"],
            "permission_document_uri": "gs://bucket/rights/doc.pdf",
            "consent_notes": [],
        },
        "capture_mode": {
            "requested_mode": "site_world_candidate",
            "resolved_mode": "site_world_candidate",
        },
    }
    if include_site_identity:
        manifest["site_identity"] = {
            "site_id": "site-parity",
            "site_id_source": "test_fixture",
        }
    (raw_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (raw_root / "intake_packet.json").write_text(
        json.dumps(
            {
                "workflowName": "Open capture",
                "taskSteps": ["Walk the site"],
                "zone": "main floor",
                "owner": "ops",
            }
        ),
        encoding="utf-8",
    )
    (raw_root / "capture_context.json").write_text(
        json.dumps(
            {
                "sceneId": scene_id,
                "captureId": capture_id,
                "captureSource": "iphone",
                "captureModality": "iphone_arkit_lidar",
            }
        ),
        encoding="utf-8",
    )
    (raw_root / "walkthrough.mov").write_bytes(b"video")
    (raw_root / "capture_upload_complete.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "sceneId": scene_id,
                "captureId": capture_id,
                "raw_prefix": f"scenes/{scene_id}/captures/{capture_id}/raw",
                "completed_at": "2026-07-09T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (raw_root / "task_hypothesis.json").write_text(
        json.dumps(
            {"workflow_name": "Open capture", "task_steps": ["Walk the site"], "status": "accepted"}
        ),
        encoding="utf-8",
    )
    (raw_root / "arkit" / "frames.jsonl").write_text(
        "\n".join(json.dumps(row) for row in frame_rows) + "\n",
        encoding="utf-8",
    )
    (raw_root / "arkit" / "poses.jsonl").write_text(
        "\n".join(json.dumps(row) for row in pose_rows) + "\n",
        encoding="utf-8",
    )
    (raw_root / "arkit" / "intrinsics.json").write_text(
        json.dumps(
            {"width": 1920, "height": 1080, "fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0}
        ),
        encoding="utf-8",
    )
    (raw_root / "arkit" / "depth" / "000001.png").write_bytes(b"depth")
    _rehash_raw_bundle(raw_root)
    return {"bucket": bucket, "scene_id": scene_id, "capture_id": capture_id}


def _set_manifest_pose_alignment_declaration(
    root: Path,
    ids: dict[str, object],
    *,
    pose_match_rate: float,
    p95_pose_delta_sec: float,
) -> None:
    raw_root = (
        root
        / str(ids["bucket"])
        / "scenes"
        / str(ids["scene_id"])
        / "captures"
        / str(ids["capture_id"])
        / "raw"
    )
    manifest_path = raw_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pose_match_rate"] = pose_match_rate
    manifest["p95_pose_delta_sec"] = p95_pose_delta_sec
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _rehash_raw_bundle(raw_root)


def test_materialization_promotes_aligned_iphone_capture_to_site_world_candidate(
    tmp_path: Path,
) -> None:
    ids = _write_capture(
        tmp_path,
        frame_rows=[
            {"frame_id": "000001", "t_device_sec": 0.0},
            {"frame_id": "000002", "t_device_sec": 0.1},
        ],
        pose_rows=[
            {
                "frame_id": "000001",
                "t_device_sec": 0.0,
                "T_world_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            },
            {
                "frame_id": "000002",
                "t_device_sec": 0.1,
                "T_world_camera": [[1, 0, 0, 0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            },
        ],
    )

    materialized = materialize_capture_bundle(gcs_root=tmp_path, **ids)
    payload = dict(materialized["descriptor"])

    assert payload["quality"]["pose_alignment_ok"] is True
    assert payload["quality"]["world_model_candidate"] is True
    assert payload["quality"]["pose_match_rate"] == 1.0
    assert payload["metadata"]["capture_mode"]["resolved_mode"] == "site_world_candidate"
    assert (
        "site_id_present:True"
        in payload["metadata"]["scene_memory_capture"]["world_model_candidate_reasoning"]
    )
    assert (
        "pose_alignment_ok:True"
        in payload["metadata"]["scene_memory_capture"]["world_model_candidate_reasoning"]
    )


def test_materialization_downgrades_site_world_candidate_without_stable_site_id(
    tmp_path: Path,
) -> None:
    ids = _write_capture(
        tmp_path,
        frame_rows=[
            {"frame_id": "000001", "t_device_sec": 0.0},
            {"frame_id": "000002", "t_device_sec": 0.1},
        ],
        pose_rows=[
            {
                "frame_id": "000001",
                "t_device_sec": 0.0,
                "T_world_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            },
            {
                "frame_id": "000002",
                "t_device_sec": 0.1,
                "T_world_camera": [[1, 0, 0, 0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            },
        ],
        include_site_identity=False,
    )

    materialized = materialize_capture_bundle(gcs_root=tmp_path, **ids)
    payload = dict(materialized["descriptor"])

    assert payload["quality"]["pose_alignment_ok"] is True
    assert payload["quality"]["world_model_candidate"] is False
    assert payload["metadata"]["capture_mode"]["resolved_mode"] == "qualification_only"
    assert payload["metadata"]["capture_mode"]["downgrade_reason"] == "missing_site_id"
    assert (
        "site_id_present:False"
        in payload["metadata"]["scene_memory_capture"]["world_model_candidate_reasoning"]
    )


def test_materialization_preserves_meta_raw_video_descriptor_lineage(tmp_path: Path) -> None:
    bucket = "local-blueprint"
    scene_id = "scene-meta"
    capture_id = "capture-meta"
    raw_root = tmp_path / bucket / "scenes" / scene_id / "captures" / capture_id / "raw"
    glasses_root = raw_root / "glasses"
    glasses_root.mkdir(parents=True)
    (raw_root / "walkthrough.mov").write_bytes(b"video")
    (glasses_root / "frame_timestamps.jsonl").write_text(
        json.dumps({"frame_index": 0, "timestamp_seconds": 0.0}) + "\n",
        encoding="utf-8",
    )
    (glasses_root / "stream_metadata.json").write_text(
        json.dumps({"device_model": "Ray-Ban Meta", "fps": 30.0, "width": 1280, "height": 720}),
        encoding="utf-8",
    )
    (raw_root / "manifest.json").write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "capture_id": capture_id,
                "video_uri": "walkthrough.mov",
                "capture_source": "meta_glasses",
                "source_device": "meta_glasses",
                "capture_profile_id": "glasses_pov",
                "width": 1280,
                "height": 720,
                "fps_source": 30.0,
                "capture_start_epoch_ms": 1_700_000_000_000,
                "site_identity": {"site_id": "site-meta", "site_id_source": "fixture"},
                "capture_topology": {
                    "capture_session_id": "session-meta-1",
                    "route_id": "route-main",
                    "pass_id": "pass-1",
                    "pass_index": 1,
                },
                "capture_rights": {
                    "derived_scene_generation_allowed": True,
                    "consent_status": "documented",
                    "permission_document_uri": "gs://bucket/rights/meta.pdf",
                },
                "privacy_lineage": {"status": "raw_unprocessed", "source": "field_capture"},
                "provenance_lineage": {
                    "capture_app": "BlueprintCapture",
                    "original_media_sha256": "abc",
                },
                "capture_mode": {
                    "requested_mode": "site_world_candidate",
                    "resolved_mode": "site_world_candidate",
                },
                "requested_outputs": ["qualification", "retrieval_index"],
                "disable_default_preview": True,
            }
        ),
        encoding="utf-8",
    )
    (raw_root / "intake_packet.json").write_text(
        json.dumps({"workflowName": "Walk", "taskSteps": ["Walk route"], "zone": "main"}),
        encoding="utf-8",
    )
    (raw_root / "capture_context.json").write_text(
        json.dumps({"captureSource": "meta_glasses", "captureModality": "glasses_video_only"}),
        encoding="utf-8",
    )
    (raw_root / "capture_upload_complete.json").write_text(
        json.dumps({"ok": True}), encoding="utf-8"
    )

    materialized = materialize_capture_bundle(
        bucket=bucket,
        scene_id=scene_id,
        capture_id=capture_id,
        gcs_root=tmp_path,
    )
    payload = dict(materialized["descriptor"])

    assert payload["capture_source"] == "glasses"
    assert payload["source_device"] == "meta_glasses"
    assert payload["capture_modality"] == "glasses_pov"
    assert payload["raw_video_uri"].endswith("/raw/walkthrough.mov")
    assert payload["media_metadata"]["original_video_path"].endswith("raw/walkthrough.mov")
    assert payload["media_metadata"]["frame_timestamps_uri"].endswith(
        "raw/glasses/frame_timestamps.jsonl"
    )
    assert payload["media_metadata"]["stream_metadata_uri"].endswith(
        "raw/glasses/stream_metadata.json"
    )
    assert payload["metadata"]["capture_topology"]["capture_session_id"] == "session-meta-1"
    assert payload["metadata"]["capture_topology"]["route_id"] == "route-main"
    assert payload["metadata"]["capture_topology"]["pass_id"] == "pass-1"
    assert payload["metadata"]["privacy_lineage"]["status"] == "raw_unprocessed"
    assert payload["metadata"]["provenance_lineage"]["original_media_sha256"] == "abc"
    assert payload["quality"]["world_model_candidate"] is False
    assert payload["metadata"]["capture_mode"]["downgrade_reason"] == "awaiting_geometry_stage"


def test_materialization_downgrades_misaligned_iphone_capture_locally(tmp_path: Path) -> None:
    ids = _write_capture(
        tmp_path,
        frame_rows=[
            {"frame_id": "000001", "t_device_sec": 0.0},
            {"frame_id": "000002", "t_device_sec": 0.1},
        ],
        pose_rows=[
            {
                "frame_id": "001001",
                "t_device_sec": 2.0,
                "T_world_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            },
            {
                "frame_id": "001002",
                "t_device_sec": 2.1,
                "T_world_camera": [[1, 0, 0, 0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            },
        ],
    )

    materialized = materialize_capture_bundle(gcs_root=tmp_path, **ids)
    payload = dict(materialized["descriptor"])

    assert payload["quality"]["pose_alignment_ok"] is False
    assert payload["quality"]["world_model_candidate"] is False
    assert payload["quality"]["pose_match_rate"] == 0.0
    assert payload["quality"]["p95_pose_delta_sec"] is None
    assert payload["metadata"]["temporal_alignment"]["metrics"]["matched_count"] == 0
    assert all(
        row["reason"] in {"no_pose_within_delta", "pose_not_joined"}
        for row in payload["metadata"]["temporal_alignment"]["drop_ledger"]
    )
    assert payload["metadata"]["capture_mode"]["resolved_mode"] == "qualification_only"
    assert (
        payload["metadata"]["capture_mode"]["downgrade_reason"] == "insufficient_spatial_evidence"
    )
    assert (
        "pose_alignment_ok:False"
        in payload["metadata"]["scene_memory_capture"]["world_model_candidate_reasoning"]
    )


def test_forged_perfect_manifest_alignment_cannot_promote_zero_join_capture(
    tmp_path: Path,
) -> None:
    ids = _write_capture(
        tmp_path,
        frame_rows=[
            {"frame_id": "000001", "t_device_sec": 0.0},
            {"frame_id": "000002", "t_device_sec": 0.1},
        ],
        pose_rows=[
            {
                "frame_id": "001001",
                "t_device_sec": 2.0,
                "T_world_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            },
            {
                "frame_id": "001002",
                "t_device_sec": 2.1,
                "T_world_camera": [[1, 0, 0, 0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            },
        ],
    )
    _set_manifest_pose_alignment_declaration(
        tmp_path,
        ids,
        pose_match_rate=1.0,
        p95_pose_delta_sec=0.0,
    )

    materialized = materialize_capture_bundle(gcs_root=tmp_path, **ids)
    payload = dict(materialized["descriptor"])
    authority = payload["metadata"]["temporal_alignment_authority"]

    assert payload["quality"]["raw_bundle_integrity_verified"] is True
    assert payload["metadata"]["temporal_alignment"]["metrics"]["matched_count"] == 0
    assert payload["quality"]["pose_match_rate"] == 0.0
    assert payload["quality"]["p95_pose_delta_sec"] is None
    assert payload["quality"]["pose_alignment_ok"] is False
    assert authority["authoritative_source"] == "canonical_recomputed_frame_pose_join"
    assert authority["manifest_declaration"] == {
        "source": "raw_manifest",
        "authority": "non_authoritative_declaration",
        "used_for_candidacy": False,
        "pose_match_rate": 1.0,
        "p95_pose_delta_sec": 0.0,
    }
    assert payload["quality"]["world_model_candidate"] is False
    assert payload["metadata"]["world_model_candidacy"]["candidate"] is False
    assert payload["metadata"]["scene_memory_capture"]["world_model_candidate"] is False
    assert materialized["qa_report"]["scene_memory_readiness"]["world_model_candidate"] is False
    assert payload["metadata"]["capture_mode"]["resolved_mode"] == "qualification_only"


def test_pessimistic_manifest_alignment_cannot_downgrade_valid_join_capture(
    tmp_path: Path,
) -> None:
    ids = _write_capture(
        tmp_path,
        frame_rows=[
            {"frame_id": "000001", "t_device_sec": 0.0},
            {"frame_id": "000002", "t_device_sec": 0.1},
        ],
        pose_rows=[
            {
                "frame_id": "000001",
                "t_device_sec": 0.0,
                "T_world_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            },
            {
                "frame_id": "000002",
                "t_device_sec": 0.1,
                "T_world_camera": [[1, 0, 0, 0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            },
        ],
    )
    _set_manifest_pose_alignment_declaration(
        tmp_path,
        ids,
        pose_match_rate=0.0,
        p95_pose_delta_sec=99.0,
    )

    materialized = materialize_capture_bundle(gcs_root=tmp_path, **ids)
    payload = dict(materialized["descriptor"])

    assert payload["quality"]["pose_match_rate"] == 1.0
    assert payload["quality"]["p95_pose_delta_sec"] == 0.0
    assert payload["quality"]["pose_alignment_ok"] is True
    assert payload["quality"]["world_model_candidate"] is True
    assert payload["metadata"]["world_model_candidacy"]["candidate"] is True
    assert payload["metadata"]["scene_memory_capture"]["world_model_candidate"] is True
    assert materialized["qa_report"]["scene_memory_readiness"]["world_model_candidate"] is True
    assert payload["metadata"]["capture_mode"]["resolved_mode"] == "site_world_candidate"


# --- rights-boolean normalization parity across every consuming surface ---

_RIGHTS_MISSING = object()
_RIGHTS_COERCION_FORMS = [
    pytest.param(False, False, id="bool-false"),
    pytest.param("false", False, id="str-false"),
    pytest.param("0", False, id="str-zero"),
    pytest.param("no", False, id="str-no"),
    pytest.param("off", False, id="str-off"),
    pytest.param("", False, id="empty-str"),
    pytest.param(_RIGHTS_MISSING, False, id="missing"),
    pytest.param(True, True, id="bool-true"),
    pytest.param("true", True, id="str-true"),
    pytest.param("1", True, id="str-one"),
    pytest.param("yes", True, id="str-yes"),
    pytest.param("on", True, id="str-on"),
]

_DESCRIPTOR_BASE = {
    "schema_version": "v1",
    "scene_id": "scene-parity",
    "capture_id": "capture-parity",
    "capture_source": "meta_glasses",
    "raw_prefix_uri": "gs://bucket/scenes/scene-parity/captures/capture-parity/raw",
    "frames_index_uri": "gs://bucket/scenes/scene-parity/captures/capture-parity/frames.jsonl",
}


def _rights_mapping(rights_value: object) -> dict[str, object]:
    if rights_value is _RIGHTS_MISSING:
        return {}
    return {"derived_scene_generation_allowed": rights_value}


@pytest.mark.parametrize(("rights_value", "expected"), _RIGHTS_COERCION_FORMS)
def test_rights_boolean_normalization_agrees_across_surfaces(
    tmp_path: Path, rights_value: object, expected: bool
) -> None:
    rights = _rights_mapping(rights_value)
    manifest = {
        "site_identity": {"site_id": "site-parity"},
        "capture_mode": {
            "requested_mode": "site_world_candidate",
            "resolved_mode": "site_world_candidate",
        },
        "capture_rights": rights,
    }

    # 1. Canonical candidacy.
    candidate = materialization._canonical_world_model_candidate(
        manifest=manifest,
        arkit_poses_uri="gs://bucket/poses",
        arkit_intrinsics_uri="gs://bucket/intrinsics",
        arkit_depth_prefix_uri="gs://bucket/depth",
        intake_complete=True,
        evidence_tier="qualified_metric_capture",
        capture_source="iphone",
        pose_match_rate=0.95,
        p95_pose_delta_sec=0.02,
        geometry_ready=True,
        geometry_source="arkit",
    )
    assert candidate is expected

    # 2. Normalized rights projection.
    block = materialization._capture_rights_block(manifest)
    assert block["derived_scene_generation_allowed"] is expected

    # 3. Geometry descriptor patching.
    capture_root = tmp_path / "bucket" / "scenes" / "scene-parity" / "captures" / "capture-parity"
    geometry_root = capture_root / "pipeline" / "geometry"
    geometry_root.mkdir(parents=True)
    context = LocalCaptureContext(
        capture_root=capture_root,
        raw_root=capture_root / "raw",
        pipeline_root=capture_root / "pipeline",
        descriptor_path=capture_root / "capture_descriptor.json",
        raw_complete_path=capture_root / "raw" / "raw_complete.json",
        storage_root=tmp_path / "bucket",
        bucket="bucket",
        scene_id="scene-parity",
        capture_id="capture-parity",
    )
    descriptor = CaptureDescriptor.from_dict(
        {
            **_DESCRIPTOR_BASE,
            "metadata": {
                "capture_mode": {"requested_mode": "site_world_candidate"},
                "capture_rights": rights,
            },
        }
    )
    geometry_stage._patch_descriptor_with_geometry(
        context=context,
        descriptor=descriptor,
        geometry_source="video_to_world",
        ready_for_world_model=True,
        contract_ready_for_world_model=True,
        internal_fallback_ready=False,
        geometry_live_ready=True,
        external_market_ready=True,
        site_faithful_market_ready=False,
        provider_native_result=True,
        fallback_used=False,
        fallback_kind=None,
        launch_blockers=[],
        coordinate_frame_session_id="session-parity",
        summary_path=geometry_root / "geometry_summary.json",
        manifest_path=geometry_root / "geometry_manifest.json",
    )
    patched = json.loads(context.descriptor_path.read_text(encoding="utf-8"))
    assert patched["world_model_candidate"] is expected

    # 4. Legacy alpha-readiness capture_mode inference.
    alpha_root = tmp_path / "alpha-capture"
    alpha_root.mkdir()
    (alpha_root / "capture_descriptor.json").write_text(
        json.dumps(
            {
                **_DESCRIPTOR_BASE,
                "quality": {"geometry_ready": True},
                "metadata": {"capture_rights": rights},
            }
        ),
        encoding="utf-8",
    )
    summary = build_alpha_readiness_summary(capture_root=alpha_root, env={})
    assert summary["capture_mode"] == (
        "site_world_candidate" if expected else "qualification_only"
    )

    # 5. Default geometry-stage routing.
    routing_descriptor = CaptureDescriptor.from_dict(
        {**_DESCRIPTOR_BASE, "metadata": {"capture_rights": rights}}
    )
    assert _should_run_default_geometry_stage(routing_descriptor) is expected
