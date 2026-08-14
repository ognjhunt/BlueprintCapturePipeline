from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import (
    canonical_digest,
    canonical_json,
)
from blueprint_pipeline.fresh_scene_artifixer_candidate_preparation import (
    REQUEST_SCHEMA_VERSION as CANDIDATE_REQUEST_SCHEMA_VERSION,
    materialize_fresh_scene_artifixer_candidate_preparation,
)
from blueprint_pipeline.fresh_scene_semantic_teacher_image_edit import (
    PROMPT_POLICY,
    REQUEST_SCHEMA_VERSION,
    RIGHTS_SCHEMA_VERSION,
    SemanticTeacherImageEditError,
    materialize_semantic_teacher_image_edit_packet,
)
from blueprint_pipeline.image_editor_backend_registry import (
    ARTIFIXER_DIRECT_CAPABILITY,
    REGISTRY_SCHEMA_VERSION,
    SEMANTIC_TEACHER_IMAGE_EDIT_CAPABILITY,
)
from tests.test_public_scene_segment_mask_repair_preflight import _fixture


def _candidate(tmp_path: Path) -> tuple[Path, dict]:
    cutout, authority, _masks = _fixture(tmp_path / "source", task_count=2)
    request = {
        "schema_version": CANDIDATE_REQUEST_SCHEMA_VERSION,
        "segment_cutout_set_path": str(cutout),
        "execution_authority_path": str(authority),
        "selected_task_ids": ["task_1", "task_2"],
        "object_absent_reference_receipt_paths": [],
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    root = tmp_path / "candidate"
    materialize_fresh_scene_artifixer_candidate_preparation(
        request=request, output_root=root
    )
    path = root / "candidate_inputs/public_scene_artifixer3d_candidate_inputs.v3.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _backend(
    backend_id: str,
    *,
    transport_kind: str,
    mask_encoding: str,
    external_disclosure_required: bool,
) -> dict:
    hosted = transport_kind == "hosted_image_edit"
    return {
        "backend_id": backend_id,
        "capability": SEMANTIC_TEACHER_IMAGE_EDIT_CAPABILITY,
        "model_identity": f"Replaceable editor {backend_id}",
        "license": "commercial-provider-terms" if hosted else "Apache-2.0",
        "license_url": "https://example.invalid/editor-terms",
        "commercial_use_permitted": True,
        "recorded_on": "2026-08-13",
        "execution": {
            "transport_kind": transport_kind,
            "adapter_id": f"{backend_id}_adapter_v1",
            "provider_id": "example_hosted" if hosted else "bounded_gpu_allocator",
            "model_snapshot": f"{backend_id}-immutable-snapshot",
            "endpoint": "https://example.invalid/v1/images/edits" if hosted else None,
            "masked_image_edit_supported": True,
            "high_fidelity_input_supported": True,
            "output_formats": ["png"],
            "mask_encoding": mask_encoding,
            "external_disclosure_required": external_disclosure_required,
            "default_options": {"quality": "high"},
        },
    }


def test_direct_editor_capability_is_refused_by_semantic_teacher_lane(
    tmp_path: Path,
) -> None:
    candidate_path, candidate = _candidate(tmp_path)
    registry_path, rows = _registry(tmp_path)
    backend = rows["next_quarter_hosted_editor"]
    backend["capability"] = ARTIFIXER_DIRECT_CAPABILITY
    registry_path.write_text(
        canonical_json(
            {"schema_version": REGISTRY_SCHEMA_VERSION, "backends": list(rows.values())}
        )
        + "\n",
        encoding="utf-8",
    )
    rights = _rights(tmp_path, candidate=candidate, backend=backend)
    request = _request(
        candidate_path=candidate_path,
        registry_path=registry_path,
        rights_path=rights,
        backend_id=backend["backend_id"],
    )

    with pytest.raises(SemanticTeacherImageEditError, match="execution_contract"):
        materialize_semantic_teacher_image_edit_packet(
            request=request, output_root=tmp_path / "must-not-write"
        )


def _registry(tmp_path: Path) -> tuple[Path, dict[str, dict]]:
    rows = {
        "next_quarter_hosted_editor": _backend(
            "next_quarter_hosted_editor",
            transport_kind="hosted_image_edit",
            mask_encoding="rgba_alpha_zero_edit_region_png",
            external_disclosure_required=True,
        ),
        "small_open_editor": _backend(
            "small_open_editor",
            transport_kind="local_gpu_image_edit",
            mask_encoding="binary_white_edit_region_png",
            external_disclosure_required=False,
        ),
    }
    path = tmp_path / "image_editor_backends.json"
    path.write_text(
        canonical_json(
            {"schema_version": REGISTRY_SCHEMA_VERSION, "backends": list(rows.values())}
        )
        + "\n",
        encoding="utf-8",
    )
    return path, rows


def test_semantic_teacher_refuses_a_direct_editor_capability(tmp_path: Path) -> None:
    candidate_path, candidate = _candidate(tmp_path)
    backend = _backend(
        "wrong_capability",
        transport_kind="hosted_image_edit",
        mask_encoding="rgba_alpha_zero_edit_region_png",
        external_disclosure_required=True,
    )
    backend["capability"] = "artifixer_direct"
    registry_path = tmp_path / "wrong-capability-registry.json"
    registry_path.write_text(
        canonical_json(
            {"schema_version": REGISTRY_SCHEMA_VERSION, "backends": [backend]}
        )
        + "\n",
        encoding="utf-8",
    )
    rights_path = _rights(tmp_path, candidate=candidate, backend=backend)

    with pytest.raises(
        SemanticTeacherImageEditError,
        match="semantic_teacher_backend_execution_contract_invalid",
    ):
        materialize_semantic_teacher_image_edit_packet(
            request=_request(
                candidate_path=candidate_path,
                registry_path=registry_path,
                rights_path=rights_path,
                backend_id=backend["backend_id"],
            ),
            output_root=tmp_path / "wrong-capability-packet",
        )


def _rights(
    tmp_path: Path,
    *,
    candidate: dict,
    backend: dict,
    disclose: bool | None = None,
) -> Path:
    execution = backend["execution"]
    external = execution["external_disclosure_required"]
    value = {
        "schema_version": RIGHTS_SCHEMA_VERSION,
        "status": "accepted_for_private_derived_semantic_edit",
        "source_candidate_inputs_receipt_digest": candidate["receipt_digest"],
        "publisher_scene_id": candidate["publisher_scene_id"],
        "backend_id": backend["backend_id"],
        "backend_entry_digest": canonical_digest(backend),
        "provider_id": execution["provider_id"],
        "model_snapshot": execution["model_snapshot"],
        "private_derived_frame_disclosure_authorized": (
            external if disclose is None else disclose
        ),
        "provider_retention_terms_accepted": external,
        "provider_training_terms_accepted": external,
        "local_private_derived_use_authorized": not external,
        "raw_nonredistributable_source_bytes_included": False,
        "issued_by_agent": False,
        "accepted_by": "fixture_human",
        "accepted_on": "2026-08-13",
        "attestation_digest": "",
    }
    value["attestation_digest"] = canonical_digest(
        value, digest_field="attestation_digest"
    )
    path = tmp_path / f"{backend['backend_id']}_rights.json"
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")
    return path


def _request(
    *, candidate_path: Path, registry_path: Path, rights_path: Path, backend_id: str
) -> dict:
    value = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "source_candidate_inputs_receipt_path": str(candidate_path),
        "backend_registry_path": str(registry_path),
        "backend_id": backend_id,
        "rights_attestation_path": str(rights_path),
        "selected_task_ids": ["task_1", "task_2"],
        "prompt_policy": PROMPT_POLICY,
        "output_format": "png",
        "retry_count": 0,
        "request_digest": "",
    }
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    return value


@pytest.mark.parametrize(
    ("backend_id", "edit_test"),
    [
        ("next_quarter_hosted_editor", lambda image: image[..., 3] == 0),
        ("small_open_editor", lambda image: image > 0),
    ],
)
def test_registry_selected_backend_materializes_exact_masks_without_execution(
    tmp_path: Path, backend_id: str, edit_test
) -> None:
    candidate_path, candidate = _candidate(tmp_path)
    registry_path, rows = _registry(tmp_path)
    rights_path = _rights(
        tmp_path, candidate=candidate, backend=rows[backend_id]
    )

    result = materialize_semantic_teacher_image_edit_packet(
        request=_request(
            candidate_path=candidate_path,
            registry_path=registry_path,
            rights_path=rights_path,
            backend_id=backend_id,
        ),
        output_root=tmp_path / f"packet-{backend_id}",
    )

    assert result["backend"]["registry_entry"]["backend_id"] == backend_id
    assert result["backend"]["backend_entry_digest"] == canonical_digest(
        rows[backend_id]
    )
    assert result["task_count"] == 2
    assert result["request_count"] == sum(
        task["camera_count"] for task in result["tasks"]
    )
    assert result["provider_mutations_performed"] == 0
    assert result["provider_upload_performed"] is False
    assert result["provider_inference_performed"] is False
    for task in result["tasks"]:
        for frame in task["frames"]:
            exact = np.asarray(
                Image.open(frame["exact_repair_mask"]["path"]).convert("L"),
                dtype=np.uint8,
            ) > 0
            staged_path = (
                tmp_path
                / f"packet-{backend_id}"
                / frame["staged_edit_mask"]["relative_path"]
            )
            with Image.open(staged_path) as image:
                pixels = np.asarray(
                    image.convert("RGBA" if image.mode == "RGBA" else "L"),
                    dtype=np.uint8,
                )
            assert np.array_equal(edit_test(pixels), exact)


def test_hosted_editor_refuses_missing_human_disclosure_authority(
    tmp_path: Path,
) -> None:
    candidate_path, candidate = _candidate(tmp_path)
    registry_path, rows = _registry(tmp_path)
    backend = rows["next_quarter_hosted_editor"]
    rights_path = _rights(
        tmp_path, candidate=candidate, backend=backend, disclose=False
    )

    with pytest.raises(
        SemanticTeacherImageEditError,
        match="semantic_teacher_rights_attestation_invalid",
    ):
        materialize_semantic_teacher_image_edit_packet(
            request=_request(
                candidate_path=candidate_path,
                registry_path=registry_path,
                rights_path=rights_path,
                backend_id=backend["backend_id"],
            ),
            output_root=tmp_path / "packet",
        )


def test_registry_row_change_invalidates_the_bound_rights_attestation(
    tmp_path: Path,
) -> None:
    candidate_path, candidate = _candidate(tmp_path)
    registry_path, rows = _registry(tmp_path)
    backend = rows["next_quarter_hosted_editor"]
    rights_path = _rights(tmp_path, candidate=candidate, backend=backend)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["backends"][0]["execution"]["model_snapshot"] = "different-snapshot"
    registry_path.write_text(canonical_json(registry) + "\n", encoding="utf-8")

    with pytest.raises(
        SemanticTeacherImageEditError,
        match="semantic_teacher_rights_attestation_invalid",
    ):
        materialize_semantic_teacher_image_edit_packet(
            request=_request(
                candidate_path=candidate_path,
                registry_path=registry_path,
                rights_path=rights_path,
                backend_id=backend["backend_id"],
            ),
            output_root=tmp_path / "packet",
        )


def test_fresh_scene_packet_module_contains_no_model_name_literal() -> None:
    from blueprint_pipeline import fresh_scene_semantic_teacher_image_edit as module

    source = Path(module.__file__).read_text(encoding="utf-8").lower()
    assert "gpt-image-2" not in source
