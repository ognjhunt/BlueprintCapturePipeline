from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.public_scene_aura_exact_residual_preflight import (
    AuraExactResidualPreflightError,
    SCHEMA_VERSION,
    materialize_aura_exact_residual_preflight,
)
from blueprint_pipeline.public_scene_residual_inpainting_packet import (
    materialize_residual_inpainting_input_packet,
)
from tests.test_public_scene_residual_inpainting_packet import _packet_inputs


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _admit_exact_aura_backend(
    request_path: Path, *, publisher_scene_id: str = "840920"
) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    backend_path = Path(request["backend_admission_path"])
    backend = json.loads(backend_path.read_text(encoding="utf-8"))
    backend.update(
        {
            "backend_id": "aurafusion360_exact_residual_multiview",
            "strict_exact_residual_masks_required": True,
            "mask_dilation_pixels": 0,
            "outside_mask_pixel_delta_required": 0,
            "multi_view_consistency_required": True,
            "execution_authorized": False,
        }
    )
    attestation: dict[str, object] = {
        "schema_version": "third_scene_released_code_noncommercial_use_attestation.v1",
        "reviewer_role": "authorized_rights_holder",
        "authorization_kind": "explicit_user_direction_in_current_goal",
        "authorized_by": "fixture_user",
        "execution_authority_sha256": "",
        "execution_authority_digest": "",
        "internal_noncommercial_use_only": True,
        "private_derived_upload_authorized": True,
        "raw_dataset_bytes_upload_authorized": False,
        "provider_training_authorized": False,
        "noncommercial_research_evaluation_use_authorized": True,
        "commercial_use_authorized": False,
        "redistribution_authorized": False,
        "publication_authorized": False,
        "attestation_digest": "",
    }
    attestation_path = backend_path.parent / "aura-noncommercial-attestation.json"
    authority_path = backend_path.parent / "aura-execution-authority.json"
    authority: dict[str, object] = {
        "schema_version": "third_scene_dual_task_execution_authority.v1",
        "program_id": "arm-decision-proof-v1",
        "publisher_scene_id": publisher_scene_id,
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authorized_by": "fixture_user",
        "terms": {
            "interiorgs_commercial_use_authorized": False,
            "interiorgs_redistribution_authorized": False,
        },
        "authority_digest": "",
    }
    authority["authority_digest"] = canonical_digest(
        authority, digest_field="authority_digest"
    )
    _write(authority_path, authority)
    attestation["execution_authority_sha256"] = "sha256:" + hashlib.sha256(
        authority_path.read_bytes()
    ).hexdigest()
    attestation["execution_authority_digest"] = authority["authority_digest"]
    attestation["attestation_digest"] = canonical_digest(
        attestation, digest_field="attestation_digest"
    )
    _write(attestation_path, attestation)
    backend["noncommercial_research_evaluation_attestation"] = {
        "path": str(attestation_path),
        "size_bytes": attestation_path.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(attestation_path.read_bytes()).hexdigest(),
        "attestation_digest": attestation["attestation_digest"],
        "reviewer_role": attestation["reviewer_role"],
    }
    backend["execution_authority"] = {
        "path": str(authority_path),
        "size_bytes": authority_path.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(authority_path.read_bytes()).hexdigest(),
        "authority_digest": authority["authority_digest"],
    }
    backend["receipt_digest"] = canonical_digest(backend, digest_field="receipt_digest")
    _write(backend_path, backend)


def _bind_big_lama_prerequisite(tmp_path: Path, request_path: Path) -> None:
    archive = tmp_path / "methods" / "Inpaint360GS" / "LaMa" / "big-lama.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"pinned-big-lama")
    archive_sha256 = "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest()
    prerequisite: dict[str, object] = {
        "schema_version": "public_scene_method_prerequisite_receipt.v1",
        "methods": {
            "inpaint360_author_smoke": {
                "artifacts": [
                    {
                        "artifact_id": "big_lama_author_linked_archive",
                        "relative_path": "methods/Inpaint360GS/LaMa/big-lama.zip",
                        "rights_authority_id": "big_lama_apache_2_0",
                        "rights_established": True,
                        "role": "method_checkpoint",
                        "size_bytes": archive.stat().st_size,
                        "sha256": archive_sha256,
                    }
                ],
                "rights_authorities": [
                    {
                        "authority_id": "big_lama_apache_2_0",
                        "established": True,
                        "license_id": "Apache-2.0",
                        "repository": "https://github.com/advimman/lama",
                        "revision": "786f5936b27fb3dacd2b1ad799e4de968ea697e7",
                        "repository_tree": "25f9902ca0c2ec4bf6c31c2b4427f0a4f05f2fd1",
                    }
                ],
            }
        },
        "receipt_digest": "",
    }
    prerequisite["receipt_digest"] = canonical_digest(
        prerequisite, digest_field="receipt_digest"
    )
    path = tmp_path / "methods" / "prerequisite.json"
    _write(path, prerequisite)
    backend_path = Path(json.loads(request_path.read_text())["backend_admission_path"])
    backend = json.loads(backend_path.read_text())
    backend["method_prerequisite"] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "receipt_digest": prerequisite["receipt_digest"],
    }
    backend["receipt_digest"] = canonical_digest(backend, digest_field="receipt_digest")
    _write(backend_path, backend)


def _packet(
    tmp_path: Path, *, count: int = 2, publisher_scene_id: str = "840920"
) -> Path:
    request_path, _candidate = _packet_inputs(tmp_path, count=count)
    _admit_exact_aura_backend(
        request_path, publisher_scene_id=publisher_scene_id
    )
    _bind_big_lama_prerequisite(tmp_path, request_path)
    materialize_residual_inpainting_input_packet(
        request_path=request_path, output_root=tmp_path / "packet"
    )
    return tmp_path / "packet" / "public_scene_residual_inpainting_input_packet.v1.json"


def test_prepares_one_shared_direct_aura_plan_for_five_replacements(tmp_path: Path) -> None:
    packet = _packet(tmp_path, count=5)

    receipt = materialize_aura_exact_residual_preflight(
        input_packet_path=packet, output_path=tmp_path / "preflight.json"
    )

    assert receipt["schema_version"] == SCHEMA_VERSION
    assert receipt["status"] == "prepared_no_upload_no_execution"
    assert receipt["backend_admission"]["execution_authority"][
        "publisher_scene_id"
    ] == "840920"
    assert receipt["replacement_object_count"] == 5
    assert len(receipt["lanes"]) == 5
    assert receipt["aura_workflow"]["released_entrypoints"] == [
        "train.py",
        "remove.py",
        "inpaint.py",
    ]
    assert receipt["aura_workflow"]["shared_retained_scene_is_not_direct_aura_2dgs_input"] is True
    assert receipt["aura_workflow"]["excluded_stock_stages"] == [
        "utils/sam2_utils.py",
        "utils/LeftRefill/sdedit_utils.py",
    ]
    assert receipt["aura_workflow"]["workflow_stages"][0] == {
        "stage": "train_retained_scene_aura_2dgs",
        "entrypoint": "train.py",
        "input_frames": "sealed_retained_scene_before_frames",
        "input_masks": "exact_residual_masks_only",
        "mask_dilation_pixels": 0,
        "train_dilate_mask_kernel_size": 1,
        "train_dilate_mask_iter": 0,
    }
    assert receipt["aura_workflow"]["workflow_stages"][1][
        "remove_exports_generated_unseen_masks"
    ] is False
    assert receipt["aura_workflow"]["workflow_stages"][1][
        "convex_hull_expansion_performed"
    ] is True
    assert receipt["required_result_checks"][
        "aura_remove_convex_hull_expansion_evidence_required"
    ] is True
    assert receipt["aura_workflow"]["workflow_stages"][2]["transforms_performed"] is False
    assert receipt["aura_workflow"]["inpaint_dilate_mask_iter"] == 0
    assert receipt["reference_completion"]["backend_provenance"]["rights_authority"][
        "license_id"
    ] == "Apache-2.0"
    assert receipt["reference_completion"]["backend_provenance"][
        "stock_inpaint360gs_code_or_author_data_used"
    ] is False
    assert [
        (row["task_id"], row["camera_id"])
        for row in receipt["reference_completion"]["references"]
    ] == [("task_1", "camera_1"), ("task_2", "camera_2"), ("task_3", "camera_3"),
          ("task_4", "camera_4"), ("task_5", "camera_5")]
    assert receipt["required_result_checks"]["outside_mask_pixel_delta_required"] == 0
    assert receipt["execution"]["provider_mutations_performed"] == 0
    assert receipt["preflight_digest"] == canonical_digest(
        receipt, digest_field="preflight_digest"
    )


def test_rejects_aura_backend_that_allows_mask_dilation(tmp_path: Path) -> None:
    packet_path = _packet(tmp_path)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    backend_path = Path(packet["backend_admission"]["path"])
    backend = json.loads(backend_path.read_text(encoding="utf-8"))
    backend["mask_dilation_pixels"] = 1
    backend["receipt_digest"] = canonical_digest(backend, digest_field="receipt_digest")
    _write(backend_path, backend)
    packet["backend_admission"]["size_bytes"] = backend_path.stat().st_size
    packet["backend_admission"]["sha256"] = "sha256:" + __import__("hashlib").sha256(
        backend_path.read_bytes()
    ).hexdigest()
    packet["backend_admission"]["receipt_digest"] = backend["receipt_digest"]
    packet["packet_digest"] = canonical_digest(packet, digest_field="packet_digest")
    _write(packet_path, packet)

    with pytest.raises(AuraExactResidualPreflightError, match="backend_invalid"):
        materialize_aura_exact_residual_preflight(
            input_packet_path=packet_path, output_path=tmp_path / "preflight.json"
        )


def test_rejects_historical_backend_receipt_without_noncommercial_attestation(
    tmp_path: Path,
) -> None:
    packet_path = _packet(tmp_path)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    backend_path = Path(packet["backend_admission"]["path"])
    backend = json.loads(backend_path.read_text(encoding="utf-8"))
    del backend["noncommercial_research_evaluation_attestation"]
    backend["receipt_digest"] = canonical_digest(backend, digest_field="receipt_digest")
    _write(backend_path, backend)
    packet["backend_admission"]["size_bytes"] = backend_path.stat().st_size
    packet["backend_admission"]["sha256"] = "sha256:" + hashlib.sha256(
        backend_path.read_bytes()
    ).hexdigest()
    packet["backend_admission"]["receipt_digest"] = backend["receipt_digest"]
    packet["packet_digest"] = canonical_digest(packet, digest_field="packet_digest")
    _write(packet_path, packet)

    with pytest.raises(
        AuraExactResidualPreflightError,
        match="backend_noncommercial_attestation_missing",
    ):
        materialize_aura_exact_residual_preflight(
            input_packet_path=packet_path, output_path=tmp_path / "preflight.json"
        )


def test_rejects_attestation_not_bound_to_the_execution_authority(tmp_path: Path) -> None:
    packet_path = _packet(tmp_path)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    backend_path = Path(packet["backend_admission"]["path"])
    backend = json.loads(backend_path.read_text(encoding="utf-8"))
    attestation_path = Path(backend["noncommercial_research_evaluation_attestation"]["path"])
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["execution_authority_digest"] = "sha256:" + "0" * 64
    attestation["attestation_digest"] = canonical_digest(
        attestation, digest_field="attestation_digest"
    )
    _write(attestation_path, attestation)
    backend["noncommercial_research_evaluation_attestation"]["size_bytes"] = (
        attestation_path.stat().st_size
    )
    backend["noncommercial_research_evaluation_attestation"]["sha256"] = "sha256:" + hashlib.sha256(
        attestation_path.read_bytes()
    ).hexdigest()
    backend["noncommercial_research_evaluation_attestation"]["attestation_digest"] = (
        attestation["attestation_digest"]
    )
    backend["receipt_digest"] = canonical_digest(backend, digest_field="receipt_digest")
    _write(backend_path, backend)
    packet["backend_admission"]["size_bytes"] = backend_path.stat().st_size
    packet["backend_admission"]["sha256"] = "sha256:" + hashlib.sha256(
        backend_path.read_bytes()
    ).hexdigest()
    packet["backend_admission"]["receipt_digest"] = backend["receipt_digest"]
    packet["packet_digest"] = canonical_digest(packet, digest_field="packet_digest")
    _write(packet_path, packet)

    with pytest.raises(
        AuraExactResidualPreflightError,
        match="backend_noncommercial_attestation_invalid",
    ):
        materialize_aura_exact_residual_preflight(
            input_packet_path=packet_path, output_path=tmp_path / "preflight.json"
        )


def test_rejects_lane_that_drops_co_present_asset_coverage(tmp_path: Path) -> None:
    packet_path = _packet(tmp_path)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    lane_path = Path(packet["lanes"][0]["path"])
    lane = json.loads(lane_path.read_text(encoding="utf-8"))
    lane["co_present_replacement_asset_ids"] = [lane["replacement_asset_id"]]
    lane["lane_digest"] = canonical_digest(lane, digest_field="lane_digest")
    _write(lane_path, lane)
    packet["lanes"][0]["size_bytes"] = lane_path.stat().st_size
    packet["lanes"][0]["sha256"] = "sha256:" + __import__("hashlib").sha256(
        lane_path.read_bytes()
    ).hexdigest()
    packet["lanes"][0]["lane_digest"] = lane["lane_digest"]
    packet["packet_digest"] = canonical_digest(packet, digest_field="packet_digest")
    _write(packet_path, packet)

    with pytest.raises(AuraExactResidualPreflightError, match="lane_invalid"):
        materialize_aura_exact_residual_preflight(
            input_packet_path=packet_path, output_path=tmp_path / "preflight.json"
        )


def test_rejects_big_lama_checkpoint_bytes_changed(tmp_path: Path) -> None:
    packet_path = _packet(tmp_path)
    archive = tmp_path / "methods" / "Inpaint360GS" / "LaMa" / "big-lama.zip"
    archive.write_bytes(b"changed")

    with pytest.raises(AuraExactResidualPreflightError, match="big_lama_archive_invalid"):
        materialize_aura_exact_residual_preflight(
            input_packet_path=packet_path, output_path=tmp_path / "preflight.json"
        )


def test_rejects_source_render_with_camera_calibration_mismatch(tmp_path: Path) -> None:
    packet_path = _packet(tmp_path)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    lane_path = Path(packet["lanes"][0]["path"])
    lane = json.loads(lane_path.read_text(encoding="utf-8"))
    black_path = Path(lane["source_layer_black_render"]["path"])
    black = json.loads(black_path.read_text(encoding="utf-8"))
    black["calibrated_cameras"][0]["spec"]["intrinsics"]["fx"] = 3.0
    black["sealed_camera_render_manifest_digest"] = canonical_digest(
        black, digest_field="sealed_camera_render_manifest_digest"
    )
    _write(black_path, black)
    lane["source_layer_black_render"]["size_bytes"] = black_path.stat().st_size
    lane["source_layer_black_render"]["sha256"] = "sha256:" + hashlib.sha256(
        black_path.read_bytes()
    ).hexdigest()
    lane["lane_digest"] = canonical_digest(lane, digest_field="lane_digest")
    _write(lane_path, lane)
    packet["lanes"][0]["size_bytes"] = lane_path.stat().st_size
    packet["lanes"][0]["sha256"] = "sha256:" + hashlib.sha256(lane_path.read_bytes()).hexdigest()
    packet["lanes"][0]["lane_digest"] = lane["lane_digest"]
    packet["packet_digest"] = canonical_digest(packet, digest_field="packet_digest")
    _write(packet_path, packet)

    with pytest.raises(
        AuraExactResidualPreflightError,
        match="source_black_calibration_mismatch",
    ):
        materialize_aura_exact_residual_preflight(
            input_packet_path=packet_path, output_path=tmp_path / "preflight.json"
        )
