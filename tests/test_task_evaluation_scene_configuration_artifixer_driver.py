from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from blueprint_pipeline.public_scene_artifixer3d_candidate_inputs import (
    materialize_artifixer3d_candidate_inputs,
)
from blueprint_pipeline.task_evaluation_scene_configuration_artifixer_driver import (
    _VISUAL_REVIEW_COST_SCOPE,
    _materialize_preflight,
    _semantic_rights_and_request,
    _write_execution_authority,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path) -> tuple[dict, dict]:
    rights = {
        "schema_version": "task_evaluation_scene_rights_admission.v1",
        "status": "admitted_for_internal_development",
        "scene_id": "839873",
        "publisher_scene_id": "839873",
        "private_provider_processing_allowed": True,
        "provider_training_allowed": False,
        "public_redistribution_allowed": False,
    }
    rights_path = tmp_path / "rights.json"
    rights_path.write_text(json.dumps(rights), encoding="utf-8")
    frame = tmp_path / "camera-0.png"
    mask = tmp_path / "camera-0-mask.png"
    Image.new("RGB", (1024, 1024), color=(90, 80, 70)).save(frame)
    Image.new("L", (1024, 1024), color=255).save(mask)
    retained = tmp_path / "retained.ply"
    write_standard_3dgs_ply(
        SplatData(
            count=2,
            xyz=np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
            opacity=np.zeros(2, dtype=np.float32),
            f_dc=np.zeros((2, 3), dtype=np.float32),
            scales=np.zeros((2, 3), dtype=np.float32),
            quats=np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float32),
            properties=(),
        ),
        retained,
    )
    calibration = tmp_path / "cameras.json"
    calibration.write_text(
        json.dumps(
            [
                {
                    "id": "camera-0",
                    "spec": {
                        "pose": {
                            "T_world_camera_opencv": [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 1.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        },
                        "intrinsics": {
                            "model": "PINHOLE",
                            "fx": 700.0,
                            "fy": 700.0,
                            "cx": 512.0,
                            "cy": 512.0,
                            "width": 1024,
                            "height": 1024,
                        },
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    envelope = {
        "materialized_references": [
            {
                "contract_path": "scene.rights.admission",
                "materialized_path": str(rights_path),
                "digest": _sha256(rights_path),
                "size_bytes": rights_path.stat().st_size,
                "full_byte_service_account_readback_passed": True,
            }
        ],
        "render_inputs_result": {
            "camera_calibration": {"path": str(calibration)},
            "derived_frames": [
                {
                    "camera_id": "camera-0",
                    "path": str(frame),
                    "source_object_mask": {"path": str(mask)},
                }
            ],
            "derived_gaussian_cutout": {
                "retained_scene_without_source_object": {"path": str(retained)},
                "retained_count": 2,
            },
        },
    }
    configuration = {
        "schema_version": "observed_appearance_object_removal_configuration.v1",
        "source_object": {"publisher_instance_id": "104"},
        "human_authority": {
            "accepted_by": "project-owner",
            "accepted_on": "2026-08-25",
            "authority_reference": "website-scene-configuration-consent-v1",
            "private_derived_frame_disclosure_authorized": True,
            "provider_retention_terms_accepted": True,
            "provider_training_terms_accepted": True,
            "provider_training_authorized": False,
        },
    }
    return envelope, configuration


def test_generic_render_contract_feeds_released_artifixer_inputs(tmp_path: Path) -> None:
    envelope, configuration = _inputs(tmp_path)
    authority_path = tmp_path / "authority.json"
    authority, _rights_path, scene_id = _write_execution_authority(
        envelope=envelope,
        configuration=configuration,
        destination=authority_path,
    )
    preflight_path = tmp_path / "preflight.json"
    preflight, task_id = _materialize_preflight(
        envelope=envelope,
        configuration=configuration,
        authority=authority,
        authority_path=authority_path,
        output_path=preflight_path,
    )
    candidate_root = tmp_path / "candidate"
    candidate = materialize_artifixer3d_candidate_inputs(
        calibrated_residual_preflight_path=preflight_path,
        output_root=candidate_root,
    )

    assert scene_id == "839873"
    assert task_id == "remove-source-object-104"
    assert preflight["replacement_object_count"] == 1
    assert candidate["publisher_scene_id"] == "839873"
    assert candidate["tasks"][0]["camera_count"] == 1
    assert candidate["receipt_digest"] == canonical_digest(candidate, digest_field="receipt_digest")


def test_generic_candidate_feeds_existing_semantic_teacher_packet(tmp_path: Path) -> None:
    envelope, configuration = _inputs(tmp_path)
    authority_path = tmp_path / "authority.json"
    authority, _rights_path, scene_id = _write_execution_authority(
        envelope=envelope,
        configuration=configuration,
        destination=authority_path,
    )
    preflight_path = tmp_path / "preflight.json"
    _materialize_preflight(
        envelope=envelope,
        configuration=configuration,
        authority=authority,
        authority_path=authority_path,
        output_path=preflight_path,
    )
    candidate_root = tmp_path / "candidate"
    candidate = materialize_artifixer3d_candidate_inputs(
        calibrated_residual_preflight_path=preflight_path,
        output_root=candidate_root,
    )
    candidate_path = candidate_root / "public_scene_artifixer3d_candidate_inputs.v3.json"
    packet_root = _semantic_rights_and_request(
        candidate=candidate,
        candidate_path=candidate_path,
        registry_path=(
            Path(__file__).resolve().parents[1]
            / "docs/arm_decision_proof_v1/manifests/image_editor_backends.v1.json"
        ),
        configuration=configuration,
        publisher_scene_id=scene_id,
        output_root=tmp_path,
    )
    packet = json.loads(
        (packet_root / "fresh_scene_semantic_teacher_image_edit_packet.v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert packet["task_count"] == 1
    assert packet["request_count"] == 1
    assert packet["backend"]["registry_entry"]["backend_id"].startswith("openai_gpt_image_2")
    assert packet["raw_nonredistributable_source_bytes_included"] is False


def test_visual_review_uses_the_scene_lanes_exclusive_cost_scope() -> None:
    source = Path(
        __import__(
            "blueprint_pipeline.task_evaluation_scene_configuration_artifixer_driver",
            fromlist=["__file__"],
        ).__file__
    ).read_text(encoding="utf-8")

    assert _VISUAL_REVIEW_COST_SCOPE == (
        "task_evaluation_scene_configuration_artifixer_visual_review"
    )
    assert "cost_lane_id=_VISUAL_REVIEW_COST_SCOPE" in source
    assert "paid_resource_class=_VISUAL_REVIEW_COST_SCOPE" in source
