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
            "generated_at": FIXTURE_GENERATED_AT,
            "capture_source": "glasses",
            "source_device": "meta_glasses",
            "capture_modality": "glasses_video_only",
            "capture_profile_id": "glasses_pov",
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
            "captureSource": "glasses",
            "captureModality": "glasses_video_only",
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
            "status": "complete",
            "completed_at": FIXTURE_GENERATED_AT,
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
