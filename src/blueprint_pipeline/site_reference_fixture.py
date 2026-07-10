"""Deterministic local fixture for Site Reference Database v1.

The fixture exercises the local staged-capture path and uses only local
geometry plus deterministic embeddings. It does not call model providers,
storage services, or WebApp/Firebase.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from .common import read_json, write_json
from .local_bundle_workflow import run_local_bundle_workflow
from .retrieval_index_stage import run_retrieval_index_stage
from .site_reference_database import assert_summary_projection_safe

FIXTURE_BUCKET = "local-blueprint-fixture"
FIXTURE_SCENE_ID = "site-reference-fixture-scene"
FIXTURE_CAPTURE_ID = "site-reference-fixture-capture"
FIXTURE_SITE_ID = "site-reference-fixture-site"
FIXTURE_GENERATED_AT = "2026-05-24T00:00:00+00:00"


class DeterministicFixtureEmbeddingModel:
    """Small local embedding provider used only for deterministic fixtures."""

    dimension = 1024

    def encode(self, image_paths: Iterable[Path]) -> List[np.ndarray]:
        vectors: List[np.ndarray] = []
        for image_path in image_paths:
            path = Path(image_path)
            seed = hashlib.sha256(path.name.encode("utf-8") + b"\0" + path.read_bytes()).digest()
            values = np.frombuffer(seed * (self.dimension // len(seed)), dtype=np.uint8).astype(
                "float32"
            )
            values = values[: self.dimension]
            norm = float(np.linalg.norm(values)) or 1.0
            vectors.append(values / norm)
        return vectors


def build_site_reference_database_v1_fixture(output_root: str | Path) -> Dict[str, Any]:
    """Build a local staged-capture fixture through the WebApp-safe projection."""

    root = Path(output_root).expanduser().resolve()
    fixture_root = root / "site_reference_database_v1_fixture"
    if fixture_root.exists():
        shutil.rmtree(fixture_root)
    source_bundle = fixture_root / "source_bundle"
    storage_root = fixture_root / "storage"

    _write_source_bundle(source_bundle)
    workflow = run_local_bundle_workflow(
        source_bundle=source_bundle,
        storage_root=storage_root,
        bucket=FIXTURE_BUCKET,
        mode="copy",
        force=True,
        run_qualification=False,
    )
    capture_root = Path(str(workflow["capture_root"]))
    _write_privacy_artifact(capture_root)

    stage_result = run_retrieval_index_stage(
        capture_root=capture_root,
        embedding_model=DeterministicFixtureEmbeddingModel(),
    )
    site_root = storage_root / FIXTURE_BUCKET / "sites" / FIXTURE_SITE_ID / "reference_memory"
    site_index_path = site_root / "site_reference_index.jsonl"
    summary_projection_path = site_root / "site_reference_summary_projection.json"
    validation_path = site_root / "retrieval_validation.json"

    projection = read_json(summary_projection_path)
    assert_summary_projection_safe(projection)
    reference_rows = _read_jsonl(site_index_path)

    return {
        "schema_version": "site_reference_database_v1_fixture.v1",
        "fixture_root": str(fixture_root),
        "source_bundle": str(source_bundle),
        "storage_root": str(storage_root),
        "bucket": FIXTURE_BUCKET,
        "scene_id": FIXTURE_SCENE_ID,
        "capture_id": FIXTURE_CAPTURE_ID,
        "site_id": FIXTURE_SITE_ID,
        "capture_root": str(capture_root),
        "site_reference_root": str(site_root),
        "site_reference_index_path": str(site_index_path),
        "site_reference_manifest_path": str(site_root / "site_reference_manifest.json"),
        "summary_projection_path": str(summary_projection_path),
        "retrieval_validation_path": str(validation_path),
        "stage_result": stage_result,
        "reference_ids": [str(row.get("reference_id")) for row in reference_rows],
    }


def _write_source_bundle(source_bundle: Path) -> None:
    raw_root = source_bundle / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    common_identity = {
        "scene_id": FIXTURE_SCENE_ID,
        "capture_id": FIXTURE_CAPTURE_ID,
    }
    write_json(
        raw_root / "manifest.json",
        {
            **common_identity,
            "schema_version": "v3",
            "capture_schema_version": "3.0.0",
            "generated_at": FIXTURE_GENERATED_AT,
            "capture_source": "iphone",
            "capture_tier_hint": "tier1_iphone",
            "source_device": "iphone_arkit_fixture",
            "capture_modality": "iphone_arkit_lidar",
            "capture_profile_id": "iphone_arkit_lidar",
            "capture_capabilities": {
                "camera_pose": True,
                "camera_intrinsics": True,
                "depth": True,
            },
            "coordinate_frame_session_id": "fixture-coordinate-frame-001",
            "capture_start_epoch_ms": 1_700_000_000_000,
            "app_version": "1.0.0-test",
            "app_build": "1",
            "ios_version": "18.0-test",
            "ios_build": "22A-test",
            "hardware_model_identifier": "iPhoneFixture1,1",
            "device_model_marketing": "iPhone ARKit fixture",
            "has_lidar": True,
            "depth_supported": True,
            "rights_profile": "fixture_documented",
            "video_uri": "walkthrough.mov",
            "width": 640,
            "height": 480,
            "fps_source": 30,
            "requested_output": "site_world_candidate",
            "requested_outputs": ["scene_memory"],
            "disable_default_preview": True,
            "site_identity": {
                "site_id": FIXTURE_SITE_ID,
                "site_id_source": "deterministic_local_fixture",
                "site_name": "Site Reference Fixture Lab",
                "address_full": "Local fixture only",
                "building_id": "fixture-building",
                "floor_id": "fixture-floor-1",
                "zone_id": "fixture-zone-a",
            },
            "capture_topology": {
                "capture_session_id": "fixture-session-001",
                "route_id": "fixture-route-001",
                "pass_id": "fixture-pass-001",
                "pass_index": 1,
                "intended_pass_role": "primary",
                "entry_anchor_id": "fixture-entry",
                "return_anchor_id": "fixture-return",
                "site_visit_id": "fixture-visit-001",
                "coordinate_frame_session_id": "fixture-coordinate-frame-001",
            },
            "capture_mode": {
                "requested_mode": "site_world_candidate",
                "requested_output": "site_world_candidate",
                "resolved_mode": "site_world_candidate",
            },
            "capture_rights": {
                "derived_scene_generation_allowed": True,
                "data_licensing_allowed": True,
                "capture_contributor_payout_eligible": False,
                "consent_status": "fixture_documented",
                "consent_scope": ["local_contract_test"],
            },
        },
    )
    write_json(
        raw_root / "capture_context.json",
        {
            **common_identity,
            "captureSource": "iphone",
            "captureModality": "iphone_arkit_lidar",
            "intake_complete": True,
            "disableDefaultPreview": True,
        },
    )
    write_json(
        raw_root / "intake_packet.json",
        {
            "workflowName": "Fixture route walkthrough",
            "taskSteps": ["enter fixture zone", "pause at entry anchor", "walk fixture lane"],
            "zone": "fixture-zone-a",
            "owner": "Blueprint local fixture",
            "targetKPI": "contract projection only",
        },
    )
    write_json(
        raw_root / "capture_upload_complete.json",
        {
            **common_identity,
            "schema_version": "v1",
            "status": "complete",
            "completed_at": FIXTURE_GENERATED_AT,
            "raw_prefix": (
                f"scenes/{FIXTURE_SCENE_ID}/captures/{FIXTURE_CAPTURE_ID}/raw"
            ),
        },
    )
    write_json(
        raw_root / "route_anchors.json",
        {
            "schema_version": "v1",
            "route_anchors": [
                {
                    "anchor_id": "fixture-entry",
                    "anchor_type": "entry",
                    "label": "Fixture entry",
                    "expected_observation": "pause_and_pan",
                    "required_in_primary_pass": True,
                    "required_in_revisit_pass": True,
                }
            ],
        },
    )
    write_json(
        raw_root / "checkpoint_events.json",
        {
            "schema_version": "v1",
            "checkpoint_events": [
                {
                    "anchor_id": "fixture-entry",
                    "pass_id": "fixture-pass-001",
                    "t_capture_sec": 0.0,
                    "hold_duration_sec": 1.0,
                    "completed": True,
                }
            ],
        },
    )
    write_json(
        raw_root / "relocalization_events.json",
        {
            "schema_version": "v1",
            "relocalization_events": [
                {
                    "event_id": "fixture-relocalization-001",
                    "pass_id": "fixture-pass-001",
                    "route_id": "fixture-route-001",
                    "t_capture_sec": 0.0,
                    "status": "reference_origin",
                    "anchor_id": "fixture-entry",
                    "coordinate_frame_session_id": "fixture-coordinate-frame-001",
                }
            ],
        },
    )
    (raw_root / "walkthrough.mov").write_bytes(b"blueprint-site-reference-fixture-video\n")
    _write_arkit_geometry_fixture(raw_root)
    _write_raw_hash_manifest(raw_root)


def _write_arkit_geometry_fixture(raw_root: Path) -> None:
    arkit_root = raw_root / "arkit"
    frames_dir = arkit_root / "frames"
    depth_dir = arkit_root / "depth"
    confidence_dir = arkit_root / "confidence"
    frames_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    confidence_dir.mkdir(parents=True, exist_ok=True)
    intrinsics = {
        "fx": 520.0,
        "fy": 520.0,
        "cx": 32.0,
        "cy": 24.0,
        "width": 64,
        "height": 48,
    }
    write_json(arkit_root / "intrinsics.json", intrinsics)
    pose_rows: List[Dict[str, Any]] = []
    frame_rows: List[Dict[str, Any]] = []
    for frame_index in range(4):
        frame_id = f"{frame_index:06d}"
        checker = (np.indices((48, 64)).sum(axis=0) % 2).astype(np.float32) * 255.0
        image = np.repeat(checker[:, :, None], 3, axis=2)
        np.save(frames_dir / f"{frame_id}.npy", image)
        (depth_dir / f"{frame_id}.png").write_bytes(b"fixture-depth")
        (confidence_dir / f"{frame_id}.png").write_bytes(b"fixture-confidence")
        pose_rows.append(
            {
                "frameIndex": frame_index,
                "frame_id": frame_id,
                "t_device_sec": float(frame_index) * 0.5,
                "T_world_camera": [
                    [1.0, 0.0, 0.0, frame_index * 0.2],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        )
        frame_rows.append(
            {
                "frameIndex": frame_index,
                "frame_id": frame_id,
                "t_device_sec": float(frame_index) * 0.5,
                "trackingState": "normal",
                "sharpnessScore": 120.0,
                "relocalizationEvent": False,
                "worldMappingStatus": "mapped",
                "intrinsics": intrinsics,
                "imageResolution": [64, 48],
                "anchorObservations": ["fixture-entry"] if frame_index == 0 else [],
            }
        )
    (arkit_root / "poses.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in pose_rows),
        encoding="utf-8",
    )
    (arkit_root / "frames.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in frame_rows),
        encoding="utf-8",
    )


def _write_raw_hash_manifest(raw_root: Path) -> None:
    artifacts = {
        path.relative_to(raw_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(raw_root.rglob("*"))
        if path.is_file() and path.name != "hashes.json"
    }
    canonical = "\n".join(f"{name}:{artifacts[name]}" for name in sorted(artifacts))
    write_json(
        raw_root / "hashes.json",
        {
            "schema_version": "v1",
            "bundle_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "artifacts": artifacts,
        },
    )


def _write_privacy_artifact(capture_root: Path) -> None:
    privacy_root = capture_root / "privacy"
    privacy_root.mkdir(parents=True, exist_ok=True)
    (privacy_root / "final_walkthrough.mov").write_bytes(
        b"blueprint-site-reference-fixture-privacy-video\n"
    )


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows
