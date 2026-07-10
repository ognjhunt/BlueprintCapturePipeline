from __future__ import annotations

import json
import itertools
import subprocess
from pathlib import Path
from typing import Any

import pytest

import blueprint_pipeline.materialization as m


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _raw_root(tmp_path: Path, *, scene_id: str = "scene-1", capture_id: str = "capture-1") -> Path:
    raw = tmp_path / "bucket" / "scenes" / scene_id / "captures" / capture_id / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    return raw


def _minimal_ready_capture(tmp_path: Path, *, scene_id: str = "scene-1", capture_id: str = "capture-1") -> Path:
    raw = _raw_root(tmp_path, scene_id=scene_id, capture_id=capture_id)
    _write_json(raw / "manifest.json", {"scene_id": scene_id, "width": 640, "height": 480})
    (raw / "walkthrough.mov").write_bytes(b"video")
    return raw


def test_small_normalizers_and_fault_tolerant_readers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(m, "read_json", lambda _path: (_ for _ in ()).throw(RuntimeError("bad json")))
    assert m._read_optional_json(broken) == {}

    assert m._string_list(" one ") == ["one"]
    assert m._string_list(42) == ["42"]
    assert m._dict_float({"": 1, "ok": "2.5", "bad": "nan?"}) == {"ok": 2.5}

    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text('\n{"ok": 1}\nnot-json\n[]\n', encoding="utf-8")
    assert m._read_json_lines(jsonl) == [{"ok": 1}]

    class OSErrorPath:
        def is_file(self) -> bool:
            return True

        def open(self, *_args: Any, **_kwargs: Any) -> Any:
            raise OSError("cannot read")

    assert m._read_json_lines(OSErrorPath()) == []
    assert m._normalized_frame_id(0) == "000001"
    assert m._normalized_frame_id(None) is None
    assert m._time_value({"t_device_sec": None, "tCaptureSec": "bad", "timestamp": None}) is None

    assert m._percentile([], 50) is None
    assert m._percentile([2.0, 1.0], 0) == 1.0
    assert m._percentile([2.0, 1.0], 100) == 2.0
    assert m._percentile([7.0], 95) == 7.0

    assert m._nearest_pose_time([], 1.0) is None
    assert m._nearest_pose_time([3.0], 1.0) == 3.0
    assert m._nearest_pose_time([1.0, 2.0, 5.0], 1.8) == 2.0
    assert m._nearest_pose_time([1.0, 2.0, 5.0], 1.4) == 1.0


def test_pose_alignment_and_manifest_normalization_edges(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_jsonl(
        raw / "arkit" / "poses.jsonl",
        [{"t_device_sec": None}, {"timestamp_sec": 1.0, "frame_id": 0}],
    )
    _write_jsonl(
        raw / "arkit" / "frames.jsonl",
        [{}, {"timestamp_sec": 1.05, "frame_id": 0}],
    )

    alignment = m._inspect_pose_alignment(raw)

    assert alignment["matched_pose_count"] == 1.0
    assert alignment["frame_count"] == 1.0
    assert alignment["temporal_alignment_status"] == "blocked"
    assert "frames:row_0:timestamp_missing_or_ambiguous" in alignment[
        "temporal_alignment_blockers"
    ]
    identity = m._normalized_site_identity(
        {
            "site_identity": {
                "site_id": " site-1 ",
                "geo": {"latitude": 1.0, "longitude": 2.0, "accuracy_m": 3.0},
            }
        }
    )
    assert identity and identity["geo"] == {"latitude": 1.0, "longitude": 2.0, "accuracy_m": 3.0}
    assert m._normalized_route_anchors({"route_anchors": ["bad", {"anchorId": "a"}]})["route_anchors"] == [
        {
            "anchor_id": "a",
            "anchor_type": None,
            "label": None,
            "expected_observation": None,
            "required_in_primary_pass": False,
            "required_in_revisit_pass": False,
        }
    ]
    assert m._normalized_checkpoint_events({"checkpointEvents": ["bad", {"anchorId": "a"}]})["checkpoint_events"][0]["anchor_id"] == "a"
    assert m._normalized_relocalization_events({"relocalizationEvents": ["bad", {"eventId": "e"}]})["relocalization_events"][0]["event_id"] == "e"


def test_world_model_downgrade_and_lane_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = {"site_identity": {"site_id": "site-1"}, "capture_mode": {"requested_mode": "site_world_candidate"}}
    assert m._world_model_candidate_downgrade_reason(
        manifest=manifest,
        arkit_poses_uri="gs://bucket/poses",
        arkit_intrinsics_uri=None,
        arkit_depth_prefix_uri="gs://bucket/depth",
        intake_complete=True,
        capture_source="iphone",
        pose_match_rate=1.0,
        p95_pose_delta_sec=0.01,
        pose_alignment_valid=True,
        geometry_ready=False,
    ) == "missing_arkit_intrinsics"
    assert m._world_model_candidate_downgrade_reason(
        manifest=manifest,
        arkit_poses_uri="gs://bucket/poses",
        arkit_intrinsics_uri="gs://bucket/intrinsics",
        arkit_depth_prefix_uri=None,
        intake_complete=True,
        capture_source="iphone",
        pose_match_rate=1.0,
        p95_pose_delta_sec=0.01,
        pose_alignment_valid=True,
        geometry_ready=False,
    ) == "missing_lidar_depth"
    assert m._world_model_candidate_downgrade_reason(
        manifest=manifest,
        arkit_poses_uri="gs://bucket/poses",
        arkit_intrinsics_uri="gs://bucket/intrinsics",
        arkit_depth_prefix_uri="gs://bucket/depth",
        intake_complete=False,
        capture_source="iphone",
        pose_match_rate=1.0,
        p95_pose_delta_sec=0.01,
        pose_alignment_valid=True,
        geometry_ready=False,
    ) == "missing_complete_intake"
    assert m._world_model_candidate_downgrade_reason(
        manifest=manifest,
        arkit_poses_uri="gs://bucket/poses",
        arkit_intrinsics_uri="gs://bucket/intrinsics",
        arkit_depth_prefix_uri="gs://bucket/depth",
        intake_complete=True,
        capture_source="iphone",
        pose_match_rate=1.0,
        p95_pose_delta_sec=0.01,
        pose_alignment_valid=True,
        geometry_ready=False,
    ) == "derived_scene_generation_not_allowed"
    assert m._world_model_candidate_downgrade_reason(
        manifest={**manifest, "capture_rights": {"derived_scene_generation_allowed": True}},
        arkit_poses_uri=None,
        arkit_intrinsics_uri=None,
        arkit_depth_prefix_uri=None,
        intake_complete=True,
        capture_source="android",
        pose_match_rate=None,
        p95_pose_delta_sec=None,
        pose_alignment_valid=None,
        geometry_ready=True,
    ) == "site_world_candidate_gates_not_met"

    assert m._default_requested_lanes(
        {"disable_default_preview": True},
        {"sim_only_beta_default_task_eval": "true"},
    ) == ["qualification", "evaluation_prep", "simulation_automation"]
    assert m._default_requested_lanes({"requested_outputs": ["review_intake"]}, {}) == ["qualification"]
    assert m._requested_lanes_override({"requested_lanes": []}, {}) == ["qualification"]

    monkeypatch.setenv("BLUEPRINT_SIM_ONLY_BETA_DEFAULT_TASK_EVAL", "0")


def test_orientation_and_source_fallback_edges(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert m._first_int("bad", "", None) is None
    assert m._normalize_rotation_degrees("bad") is None
    assert m._normalize_rotation_degrees(359) == 359
    assert m._infer_display_orientation(10, 10) == "square"
    assert m._orientation_payload(
        encoded_width=10,
        encoded_height=20,
        declared_capture_width=None,
        declared_capture_height=None,
        display_rotation_degrees=0,
        display_orientation="",
        normalization_applied=False,
        source="test",
    )["display_orientation"] == "portrait"

    assert m._capture_orientation_from_metadata(
        manifest={},
        context={"width": 1920, "height": 1080, "captureOrientation": {"displaySize": {"width": 720, "height": 1280}}},
    )["display_orientation"] == "portrait"
    assert m._capture_orientation_from_metadata(
        manifest={},
        context={"width": 1920, "height": 1080, "captureOrientation": {"rotationDegrees": 90}},
    )["display_orientation"] == "portrait"
    assert m._capture_orientation_from_metadata(
        manifest={},
        context={"captureOrientation": {}},
    ) == {}
    assert m._capture_orientation_from_metadata(
        manifest={},
        context={"captureOrientation": {"ignored": "value"}},
    ) == {}
    assert m._capture_orientation_from_metadata(
        manifest={},
        context={"captureOrientation": {"displayWidth": 100}},
    )["display_orientation"] == "unknown"

    class Completed:
        def __init__(self, stdout: str, returncode: int = 0) -> None:
            self.stdout = stdout
            self.returncode = returncode

    video = tmp_path / "video.mov"
    video.write_bytes(b"video")
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: Completed("not-json"))
    assert m._ffprobe_capture_orientation(video) == {}
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: Completed(json.dumps({"streams": {}})))
    assert m._ffprobe_capture_orientation(video) == {}
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: Completed(json.dumps({"streams": [{"codec_type": "audio"}]})))
    assert m._ffprobe_capture_orientation(video) == {}
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: Completed(
            json.dumps(
                {
                    "streams": [
                        {
                            "codec_type": "video",
                            "width": 1920,
                            "height": 1080,
                            "side_data_list": [{"rotation": 90}, "bad"],
                        }
                    ]
                }
            )
        ),
    )
    assert m._ffprobe_capture_orientation(video)["display_rotation_degrees"] == 90

    assert m._capture_source({"capture_profile_id": "android_depth"}, {}) == "android"
    assert m._capture_source({"capture_profile_id": "iphone_arkit_lidar"}, {}) == "iphone"
    assert m._capture_source({"capture_source": "ray-ban_meta"}, {}) == "glasses"
    assert m._capture_source({"capture_source": "iphonevideo"}, {}) == "iphone"
    assert m._source_device({}, {}, "glasses") == "non_arkit_video"


def test_capture_modality_fallbacks() -> None:
    assert m._capture_modality({"capture_profile_id": "android_arcore_depth"}, {}, "android", [], False) == "android_arcore_depth"
    assert m._capture_modality({"has_lidar": True}, {}, "iphone", [], False) == "iphone_arkit_lidar"
    assert m._capture_modality({}, {}, "iphone", [], False) == "iphone_video_only"
    assert m._capture_modality({}, {}, "glasses", ["scale"], False) == "glasses_plus_scaffolding"
    assert m._capture_modality({}, {}, "glasses", [], False) == "glasses_video_only"
    assert m._capture_modality({}, {}, "android", ["scale"], False) == "android_plus_scaffolding"
    assert m._capture_modality({}, {}, "android", [], False) == "android_video_only"


def test_readiness_invalid_manifest_and_optional_descriptor_uris(tmp_path: Path) -> None:
    missing_manifest_raw = _raw_root(tmp_path, scene_id="scene-missing-manifest", capture_id="capture-missing-manifest")
    (missing_manifest_raw / "walkthrough.mov").write_bytes(b"video")
    assert m.capture_materialization_readiness(
        bucket="bucket",
        scene_id="scene-missing-manifest",
        capture_id="capture-missing-manifest",
        gcs_root=tmp_path,
    )["issues"] == [
        "missing_manifest",
        "raw_bundle_quarantined:missing_required_file:manifest.json",
    ]

    missing_video_raw = _raw_root(tmp_path, scene_id="scene-missing-video-ready", capture_id="capture-missing-video-ready")
    _write_json(missing_video_raw / "manifest.json", {"scene_id": "scene-missing-video-ready"})
    assert m.capture_materialization_readiness(
        bucket="bucket",
        scene_id="scene-missing-video-ready",
        capture_id="capture-missing-video-ready",
        gcs_root=tmp_path,
    )["issues"] == ["missing_raw_video"]
    with pytest.raises(RuntimeError, match="capture_not_ready:missing_raw_video"):
        m.assert_capture_materialization_ready(
            bucket="bucket",
            scene_id="scene-missing-video-ready",
            capture_id="capture-missing-video-ready",
            gcs_root=tmp_path,
        )

    raw = _raw_root(tmp_path)
    (raw / "manifest.json").write_text("{not-json", encoding="utf-8")
    (raw / "walkthrough.mov").write_bytes(b"video")
    readiness = m.capture_materialization_readiness(
        bucket="bucket",
        scene_id="scene-1",
        capture_id="capture-1",
        gcs_root=tmp_path,
    )
    assert readiness["issues"] == [
        "invalid_manifest",
        "raw_bundle_quarantined:invalid_json:manifest.json:JSONDecodeError",
    ]

    raw = _minimal_ready_capture(tmp_path, scene_id="scene-2", capture_id="capture-2")
    for path in (
        raw / "arkit" / "confidence" / ".keep",
        raw / "arcore" / "poses.jsonl",
        raw / "arcore" / "session_intrinsics.json",
        raw / "arcore" / "frames.jsonl",
        raw / "arcore" / "depth_manifest.json",
        raw / "arcore" / "confidence_manifest.json",
        raw / "arcore" / "depth" / ".keep",
        raw / "arcore" / "confidence" / ".keep",
        raw / "arcore" / "point_cloud.jsonl",
        raw / "arcore" / "planes.jsonl",
        raw / "arcore" / "tracking_state.jsonl",
        raw / "arcore" / "light_estimates.jsonl",
        raw / "companion_phone" / "poses.jsonl",
        raw / "companion_phone" / "session_intrinsics.json",
        raw / "companion_phone" / "calibration.json",
        raw / "object_index.json",
        raw / "motion.jsonl",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    result = m.build_capture_bundle_records(
        bucket="bucket",
        scene_id="scene-2",
        capture_id="capture-2",
        gcs_root=tmp_path,
        write_frames_index=False,
    )

    descriptor = result["descriptor"]
    assert descriptor["arkit_confidence_prefix_uri"] == "gs://bucket/scenes/scene-2/captures/capture-2/raw/arkit/confidence"
    assert descriptor["arcore_poses_uri"].endswith("/arcore/poses.jsonl")
    assert descriptor["arcore_confidence_prefix_uri"].endswith("/arcore/confidence")
    assert descriptor["companion_phone_calibration_uri"].endswith("/companion_phone/calibration.json")
    assert descriptor["object_index_uri"].endswith("/object_index.json")
    assert descriptor["motion_log_uri"].endswith("/motion.jsonl")


def test_uncertainty_edges_for_missing_video_and_unvalidated_scaffolding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_root(tmp_path, scene_id="scene-missing-video", capture_id="capture-missing-video")
    _write_json(raw / "manifest.json", {"scene_id": "scene-missing-video"})
    monkeypatch.setattr(m, "assert_capture_materialization_ready", lambda **_kwargs: {"ready": True, "issues": []})
    no_video = m.build_capture_bundle_records(
        bucket="bucket",
        scene_id="scene-missing-video",
        capture_id="capture-missing-video",
        gcs_root=tmp_path,
        write_frames_index=False,
    )
    assert no_video["descriptor"]["raw_video_uri"] is None
    assert no_video["qa_report"]["uncertainty_score"] >= 0.4

    raw = _minimal_ready_capture(tmp_path, scene_id="scene-scaffold", capture_id="capture-scaffold")
    _write_json(raw / "manifest.json", {"scene_id": "scene-scaffold", "capture_source": "rayban_meta"})
    _write_json(raw / "capture_context.json", {"scaffoldingUsed": ["scale"], "calibrationAssets": ["calib"]})
    scaffolded = m.build_capture_bundle_records(
        bucket="bucket",
        scene_id="scene-scaffold",
        capture_id="capture-scaffold",
        gcs_root=tmp_path,
        write_frames_index=False,
    )
    assert scaffolded["descriptor"]["capture_modality"] == "glasses_plus_scaffolding"
    assert scaffolded["qa_report"]["scaffolding_validation"]["validated_metric_bundle"] is False


def test_discover_raw_sidecars_arkit_geometry_ready(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    raw_prefix = "gs://bucket/scenes/sc/captures/cap/raw"
    # A full ARKit metric bundle: poses + intrinsics + depth dir => geometry ready.
    _write_jsonl(raw / "arkit" / "poses.jsonl", [{"timestamp_sec": 1.0, "frame_id": 0}, {"timestamp_sec": 2.0, "frame_id": 1}])
    _write_json(raw / "arkit" / "intrinsics.json", {"fx": 1.0})
    _write_jsonl(raw / "arkit" / "frames.jsonl", [{"timestamp_sec": 1.05, "frame_id": 0}, {"timestamp_sec": 2.05, "frame_id": 1}])
    (raw / "arkit" / "depth").mkdir(parents=True)
    (raw / "arkit" / "confidence").mkdir(parents=True)
    (raw / "object_index.json").write_text("{}", encoding="utf-8")
    (raw / "walkthrough.mov").write_bytes(b"video")

    sidecars = m._discover_raw_sidecars(
        raw_root=raw,
        raw_prefix_uri=raw_prefix,
        manifest={"width": 1920, "height": 1080},
        source="iphone",
        source_device="iphone",
    )

    assert sidecars["has_metric_arkit_bundle"] is True
    assert sidecars["arkit_geometry_ready"] is True
    assert sidecars["geometry_source"] == "arkit"
    assert sidecars["arkit_poses_uri"] == f"{raw_prefix}/arkit/poses.jsonl"
    assert sidecars["arkit_depth_prefix_uri"] == f"{raw_prefix}/arkit/depth"
    assert sidecars["arkit_confidence_prefix_uri"] == f"{raw_prefix}/arkit/confidence"
    assert sidecars["object_index_uri"] == f"{raw_prefix}/object_index.json"
    assert sidecars["raw_video_uri"] == f"{raw_prefix}/walkthrough.mov"
    assert sidecars["media_metadata"]["original_video_uri"] == f"{raw_prefix}/walkthrough.mov"
    assert sidecars["media_metadata"]["video_metadata"]["width"] == 1920.0
    # ARKit present so iPhone pose alignment ok by default (no manifest overrides).
    assert sidecars["pose_alignment_ok"] is True


def test_discover_raw_sidecars_arcore_only_and_video_uri_fallback(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    raw_prefix = "gs://bucket/scenes/sc/captures/cap/raw"
    _write_jsonl(raw / "arcore" / "poses.jsonl", [{"timestamp": 1.0}])
    _write_json(raw / "arcore" / "session_intrinsics.json", {})
    _write_json(raw / "arcore" / "depth_manifest.json", {})

    sidecars = m._discover_raw_sidecars(
        raw_root=raw,
        raw_prefix_uri=raw_prefix,
        # No on-disk video, but manifest declares an explicit video_uri => fallback.
        manifest={"video_uri": "gs://elsewhere/video.mov"},
        source="android",
        source_device="android_arcore",
    )

    assert sidecars["has_metric_arkit_bundle"] is False
    assert sidecars["arkit_geometry_ready"] is False
    assert sidecars["arcore_geometry_present"] is True
    assert sidecars["geometry_source"] == "arcore"
    assert sidecars["arcore_poses_uri"] == f"{raw_prefix}/arcore/poses.jsonl"
    assert sidecars["arcore_depth_manifest_uri"] == f"{raw_prefix}/arcore/depth_manifest.json"
    assert sidecars["raw_video_uri"] == "gs://elsewhere/video.mov"
    assert sidecars["media_metadata"]["original_video_path"] is None
    # Non-iphone source short-circuits the iPhone pose-alignment gate to True.
    assert sidecars["pose_alignment_ok"] is True


def test_resolve_world_model_candidacy_bundles_policy_outputs() -> None:
    manifest = {
        "site_identity": {"site_id": "site-1"},
        "capture_mode": {"requested_mode": "site_world_candidate"},
        "capture_rights": {"derived_scene_generation_allowed": True},
    }
    raw_prefix = "gs://bucket/scenes/sc/captures/cap/raw"
    sidecars = {
        "arkit_poses_uri": f"{raw_prefix}/arkit/poses.jsonl",
        "arkit_intrinsics_uri": f"{raw_prefix}/arkit/intrinsics.json",
        "arkit_depth_prefix_uri": f"{raw_prefix}/arkit/depth",
        "arkit_geometry_ready": True,
        "geometry_source": "arkit",
        "pose_match_rate": 0.95,
        "p95_pose_delta_sec": 0.02,
    }
    candidacy = m._resolve_world_model_candidacy(
        manifest=manifest,
        sidecars=sidecars,
        intake_complete=True,
        evidence_tier="qualified_metric_capture",
        source="iphone",
    )

    assert set(candidacy) == {
        "world_model_candidate",
        "world_model_candidate_reasoning",
        "capture_mode",
        "readiness_world_model_candidate",
        "decision",
    }
    # The bundle must equal the underlying helpers called with the same inputs,
    # proving the extraction is a pure projection (no behavior change).
    assert candidacy["world_model_candidate"] == m._canonical_world_model_candidate(
        manifest=manifest,
        arkit_poses_uri=sidecars["arkit_poses_uri"],
        arkit_intrinsics_uri=sidecars["arkit_intrinsics_uri"],
        arkit_depth_prefix_uri=sidecars["arkit_depth_prefix_uri"],
        intake_complete=True,
        evidence_tier="qualified_metric_capture",
        capture_source="iphone",
        pose_match_rate=0.95,
        p95_pose_delta_sec=0.02,
        geometry_ready=True,
        geometry_source="arkit",
    )
    assert candidacy["capture_mode"] == m._normalized_capture_mode(
        manifest=manifest,
        arkit_poses_uri=sidecars["arkit_poses_uri"],
        arkit_intrinsics_uri=sidecars["arkit_intrinsics_uri"],
        arkit_depth_prefix_uri=sidecars["arkit_depth_prefix_uri"],
        intake_complete=True,
        evidence_tier="qualified_metric_capture",
        capture_source="iphone",
        pose_match_rate=0.95,
        p95_pose_delta_sec=0.02,
        geometry_ready=True,
        geometry_source="arkit",
    )
    assert candidacy["readiness_world_model_candidate"] == m._canonical_world_model_candidate(
        manifest=manifest,
        arkit_poses_uri=sidecars["arkit_poses_uri"],
        arkit_intrinsics_uri=sidecars["arkit_intrinsics_uri"],
        arkit_depth_prefix_uri=sidecars["arkit_depth_prefix_uri"],
        intake_complete=True,
        evidence_tier="qualified_metric_capture",
        capture_source="iphone",
        pose_match_rate=0.95,
        p95_pose_delta_sec=0.02,
        geometry_ready=True,
        geometry_source="arkit",
    )
    assert candidacy["world_model_candidate"] == candidacy["readiness_world_model_candidate"]
    assert candidacy["decision"]["candidate"] == candidacy["world_model_candidate"]


def test_canonical_candidacy_projection_agrees_across_source_and_gate_combinations() -> None:
    for source, site_present, requested, rights, intake, geometry_ready, alignment_ok in itertools.product(
        ("iphone", "android", "glasses"),
        (False, True),
        ("qualification_only", "site_world_candidate"),
        (False, True),
        (False, True),
        (False, True),
        (False, True),
    ):
        manifest = {
            "site_identity": {"site_id": "site-1"} if site_present else {},
            "capture_mode": {
                "requested_mode": requested,
                "resolved_mode": requested,
            },
            "capture_rights": {"derived_scene_generation_allowed": rights},
        }
        sidecars = {
            "arkit_poses_uri": "gs://bucket/poses" if geometry_ready else None,
            "arkit_intrinsics_uri": "gs://bucket/intrinsics" if geometry_ready else None,
            "arkit_depth_prefix_uri": "gs://bucket/depth" if geometry_ready else None,
            "arkit_geometry_ready": geometry_ready,
            "geometry_source": "arkit" if geometry_ready else None,
            "pose_match_rate": 1.0 if alignment_ok else 0.0,
            "p95_pose_delta_sec": 0.0 if alignment_ok else 1.0,
        }

        projection = m._resolve_world_model_candidacy(
            manifest=manifest,
            sidecars=sidecars,
            intake_complete=intake,
            evidence_tier="qualified_metric_capture",
            source=source,
        )

        assert projection["world_model_candidate"] == projection["readiness_world_model_candidate"]
        assert projection["decision"]["candidate"] == projection["world_model_candidate"]
        assert projection["decision"]["capture_mode"] == projection["capture_mode"]


def test_build_records_reuses_candidacy_consistently(tmp_path: Path) -> None:
    # The descriptor, metadata scene_memory_capture, and qa scene_memory_readiness
    # must all surface the same candidacy projection after the refactor.
    raw = _raw_root(tmp_path, scene_id="sc-cand", capture_id="cap-cand")
    _write_json(
        raw / "manifest.json",
        {
            "scene_id": "sc-cand",
            "width": 1920,
            "height": 1080,
            "has_lidar": True,
            "capture_profile_id": "iphone_arkit_lidar",
            "capture_source": "iphone",
            "capture_mode": {"requested_mode": "site_world_candidate"},
            "capture_rights": {"derived_scene_generation_allowed": True},
            "site_identity": {"site_id": "site-cand"},
        },
    )
    (raw / "walkthrough.mov").write_bytes(b"video")
    _write_json(raw / "intake_packet.json", {"workflowName": "wf", "taskSteps": ["a"], "zone": "z", "owner": "o"})
    _write_jsonl(raw / "arkit" / "poses.jsonl", [{"timestamp": 1.0, "frame_id": 0}])
    _write_json(raw / "arkit" / "intrinsics.json", {"fx": 1.0})
    (raw / "arkit" / "depth").mkdir(parents=True)

    result = m.build_capture_bundle_records(
        bucket="bucket",
        scene_id="sc-cand",
        capture_id="cap-cand",
        gcs_root=tmp_path,
        write_frames_index=False,
    )
    descriptor = result["descriptor"]
    qa_report = result["qa_report"]
    smc = descriptor["metadata"]["scene_memory_capture"]

    # The same candidacy projection surfaces in descriptor.quality and the
    # scene_memory_capture block (single computation reused).
    assert (
        descriptor["quality"]["world_model_candidate"]
        == smc["world_model_candidate"]
    )
    assert isinstance(smc["world_model_candidate_reasoning"], list)
    assert smc["world_model_candidate_reasoning"]
    # capture_mode lives in metadata; readiness candidate is exposed in qa_report.
    assert "capture_mode" in descriptor["metadata"]
    assert "world_model_candidate" in qa_report["scene_memory_readiness"]
