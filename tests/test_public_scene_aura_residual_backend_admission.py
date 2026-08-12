from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.public_scene_aura_residual_backend_admission import (
    AURA_COMMIT,
    AURA_REPOSITORY,
    AURA_TREE,
    AuraResidualBackendAdmissionError,
    REQUIRED_CHECKPOINTS,
    build_aura_residual_backend_admission_request,
    materialize_aura_residual_backend_abstention,
    materialize_aura_residual_backend_admission,
    materialize_aura_residual_backend_admission_request,
    materialize_aura_residual_noncommercial_attestation,
    materialize_aura_residual_packet_rights_abstention,
)


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> Path:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")
    return path


def _authority(path: Path) -> Path:
    value: dict[str, object] = {
        "schema_version": "third_scene_dual_task_execution_authority.v1",
        "program_id": "arm-decision-proof-v1",
        "publisher_scene_id": "840920",
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authorized_by": "fixture_authorized_rights_holder",
        "private_rights_admitted_scene_derived_uploads_authorized": True,
        "raw_interiorgs_upload_authorized": False,
        "training_authorized": False,
        "public_dataset_bytes_publication_authorized": False,
        "terms": {
            "interiorgs_commercial_use_authorized": False,
            "interiorgs_redistribution_authorized": False,
        },
        "retention": "bounded_to_goal_then_provider_zero",
        "paid_compute": {
            "provider": "vast",
            "zero_retry": True,
            "provider_zero_required_for_lane": True,
        },
        "authority_digest": "",
    }
    value["authority_digest"] = canonical_digest(value, digest_field="authority_digest")
    return _write_json(path, value)


def _prerequisite(path: Path) -> Path:
    value: dict[str, object] = {
        "schema_version": "public_scene_method_prerequisite_receipt.v1",
        "methods": {
            "aurafusion360_quality_challenger": {
                "author_data_rights_established": True,
                "checkpoint_rights_established": True,
                "remote_snapshots": [
                    {"artifact_id": artifact_id, "rights_established": True}
                    for artifact_id in sorted(REQUIRED_CHECKPOINTS)
                ],
            }
        },
        "receipt_digest": "",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    return _write_json(path, value)


def _source_archive_and_spec(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, str, list[dict[str, str]]]:
    nested_components = {
        "submodules/diff-surfel-rasterization/LICENSE.md": (
            b"Gaussian-Splatting License\nnon-commercially\n"
        ),
        "submodules/simple-knn/LICENSE.md": (
            b"Gaussian-Splatting License\nnon-commercially\n"
        ),
    }
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_aura_residual_backend_admission.NESTED_COMPONENT_LICENSES",
        {name: _sha256_bytes(content) for name, content in nested_components.items()},
    )
    members = {
        "LICENSE": b"Apache License\nunit-test\n",
        "inpaint.py": b"print('released source')\n",
        "arguments/__init__.py": b"class Params: pass\n",
        **nested_components,
    }
    archive = root / "aurafusion360_source.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for name, content in members.items():
            output.writestr(name, content)
    spec: dict[str, object] = {
        "schema_version": "adp_aura_interiorgs_spec.v1",
        "source_repository": AURA_REPOSITORY,
        "source_commit": AURA_COMMIT,
        "source_tree": AURA_TREE,
        "source_files": [
            {"path": name, "size_bytes": len(content), "sha256": _sha256_bytes(content)}
            for name, content in members.items()
        ],
    }
    nested = [
        {
            "path": name,
            "sha256": _sha256_bytes(content),
            "license": "Gaussian-Splatting-noncommercial-research-evaluation",
        }
        for name, content in sorted(nested_components.items())
    ]
    return (
        archive,
        _write_json(root / "source_identity.json", spec),
        _sha256_bytes(members["LICENSE"]),
        nested,
    )


def _attestation(
    *,
    authority: Path,
    archive: Path,
    source_identity: Path,
    nested: list[dict[str, str]],
    path: Path,
) -> Path:
    spec = source_identity.read_bytes()
    authority_value = __import__("json").loads(authority.read_text(encoding="utf-8"))
    value: dict[str, object] = {
        "schema_version": "third_scene_released_code_noncommercial_use_attestation.v1",
        "program_id": "arm-decision-proof-v1",
        "publisher_scene_id": "840920",
        "reviewer_role": "authorized_rights_holder",
        "source_repository": AURA_REPOSITORY,
        "source_revision": AURA_COMMIT,
        "source_tree": AURA_TREE,
        "source_archive_sha256": "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_identity_spec_sha256": "sha256:" + hashlib.sha256(spec).hexdigest(),
        "source_identity_spec_source_file_count": 5,
        "nested_component_licenses": nested,
        "authorization_kind": authority_value["authority_kind"],
        "authorized_by": authority_value["authorized_by"],
        "execution_authority_sha256": _sha256_bytes(authority.read_bytes()),
        "execution_authority_digest": authority_value["authority_digest"],
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
    value["attestation_digest"] = canonical_digest(value, digest_field="attestation_digest")
    return _write_json(path, value)


def _request(
    *,
    authority: Path,
    prerequisite: Path,
    archive: Path,
    source_identity: Path,
    attestation: Path,
    lock: Path,
) -> dict[str, object]:
    return {
        "schema_version": "public_scene_aura_residual_backend_admission_request.v1",
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009D",
        "frozen_before_inpainting_execution": True,
        "learned_policy_outcomes_accessed": False,
        "strict_exact_residual_masks_required": True,
        "outside_mask_pixel_delta_required": 0,
        "multi_view_consistency_required": True,
        "execution_authority_path": str(authority),
        "prerequisite_receipt_path": str(prerequisite),
        "source_archive_path": str(archive),
        "source_identity_spec_path": str(source_identity),
        "noncommercial_attestation_path": str(attestation),
        "environment_lock_path": str(lock),
        "private_derived_upload_policy": {
            "raw_dataset_bytes_upload": False,
            "private_derived_upload": True,
            "maximum_retention_days": 7,
            "provider_training": False,
            "publication": False,
        },
    }


def test_materializes_private_derived_aura_admission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    authority = _authority(tmp_path / "authority.json")
    prerequisite = _prerequisite(tmp_path / "prerequisite.json")
    archive, source_identity, license_digest, nested = _source_archive_and_spec(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_aura_residual_backend_admission.AURA_LICENSE_SHA256",
        license_digest,
    )
    lock = (tmp_path / "pip-freeze.txt")
    lock.write_text("torch==2.0.1\n", encoding="utf-8")
    attestation = _attestation(
        authority=authority,
        archive=archive,
        source_identity=source_identity,
        nested=nested,
        path=tmp_path / "attestation.json",
    )
    request = build_aura_residual_backend_admission_request(
        _request(
            authority=authority,
            prerequisite=prerequisite,
            archive=archive,
            source_identity=source_identity,
            attestation=attestation,
            lock=lock,
        )
    )
    request_path = tmp_path / "request.json"
    materialized_request = materialize_aura_residual_backend_admission_request(
        value=request, output_path=request_path
    )
    assert materialized_request == request

    receipt = materialize_aura_residual_backend_admission(
        request_path=request_path, output_path=tmp_path / "receipt.json"
    )

    assert receipt["status"] == "rights_admitted_for_private_derived_inpainting"
    assert receipt["mask_dilation_pixels"] == 0
    assert receipt["claim_boundary"]["raw_dataset_bytes_upload_authorized"] is False
    assert receipt["source_identity_provenance"]["source_member_count"] == 5
    assert receipt["source_archive"]["noncommercial_research_evaluation_attestation_required"] is True


def test_materializes_internal_noncommercial_attestation_from_bound_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority(tmp_path / "authority.json")
    archive, source_identity, license_digest, _nested = _source_archive_and_spec(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_aura_residual_backend_admission.AURA_LICENSE_SHA256",
        license_digest,
    )

    receipt = materialize_aura_residual_noncommercial_attestation(
        execution_authority_path=authority,
        source_archive_path=archive,
        source_identity_spec_path=source_identity,
        output_path=tmp_path / "attestation.json",
    )

    assert receipt["authorized_by"] == "fixture_authorized_rights_holder"
    assert receipt["internal_noncommercial_use_only"] is True
    assert receipt["execution_authority_sha256"] == _sha256_bytes(authority.read_bytes())
    assert receipt["attestation_digest"] == canonical_digest(
        receipt, digest_field="attestation_digest"
    )


def test_internal_noncommercial_attestation_refuses_broader_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority(tmp_path / "authority.json")
    authority_value = __import__("json").loads(authority.read_text(encoding="utf-8"))
    authority_value["terms"]["interiorgs_commercial_use_authorized"] = True
    authority_value["authority_digest"] = canonical_digest(
        authority_value, digest_field="authority_digest"
    )
    _write_json(authority, authority_value)
    archive, source_identity, license_digest, _nested = _source_archive_and_spec(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_aura_residual_backend_admission.AURA_LICENSE_SHA256",
        license_digest,
    )

    with pytest.raises(
        AuraResidualBackendAdmissionError,
        match="execution_authority_internal_use_invalid",
    ):
        materialize_aura_residual_noncommercial_attestation(
            execution_authority_path=authority,
            source_archive_path=archive,
            source_identity_spec_path=source_identity,
            output_path=tmp_path / "attestation.json",
        )


def test_rejects_archive_that_does_not_match_pinned_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority(tmp_path / "authority.json")
    prerequisite = _prerequisite(tmp_path / "prerequisite.json")
    archive, source_identity, license_digest, nested = _source_archive_and_spec(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_aura_residual_backend_admission.AURA_LICENSE_SHA256",
        license_digest,
    )
    with zipfile.ZipFile(archive, "a") as output:
        output.writestr("extra.py", b"not-pinned")
    lock = tmp_path / "pip-freeze.txt"
    lock.write_text("torch==2.0.1\n", encoding="utf-8")
    attestation = _attestation(
        authority=authority,
        archive=archive,
        source_identity=source_identity,
        nested=nested,
        path=tmp_path / "attestation.json",
    )
    request_path = _write_json(
        tmp_path / "request.json",
        build_aura_residual_backend_admission_request(
            _request(
                authority=authority,
                prerequisite=prerequisite,
                archive=archive,
                source_identity=source_identity,
                attestation=attestation,
                lock=lock,
            )
        ),
    )

    with pytest.raises(AuraResidualBackendAdmissionError) as raised:
        materialize_aura_residual_backend_admission(
            request_path=request_path, output_path=tmp_path / "receipt.json"
        )

    assert raised.value.codes == ("aura_residual_source_archive_manifest_mismatch",)


def test_rejects_provider_training_permission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    authority = _authority(tmp_path / "authority.json")
    prerequisite = _prerequisite(tmp_path / "prerequisite.json")
    archive, source_identity, _license_digest, nested = _source_archive_and_spec(tmp_path, monkeypatch)
    lock = tmp_path / "pip-freeze.txt"
    lock.write_text("torch==2.0.1\n", encoding="utf-8")
    attestation = _attestation(
        authority=authority,
        archive=archive,
        source_identity=source_identity,
        nested=nested,
        path=tmp_path / "attestation.json",
    )
    request = _request(
        authority=authority,
        prerequisite=prerequisite,
        archive=archive,
        source_identity=source_identity,
        attestation=attestation,
        lock=lock,
    )
    policy = request["private_derived_upload_policy"]
    assert isinstance(policy, dict)
    policy["provider_training"] = True

    with pytest.raises(AuraResidualBackendAdmissionError) as raised:
        build_aura_residual_backend_admission_request(request)

    assert raised.value.codes == ("aura_residual_request_private_upload_policy_invalid",)


def test_seals_typed_abstention_when_noncommercial_rights_attestation_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority(tmp_path / "authority.json")
    prerequisite = _prerequisite(tmp_path / "prerequisite.json")
    archive, source_identity, license_digest, _nested = _source_archive_and_spec(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_aura_residual_backend_admission.AURA_LICENSE_SHA256",
        license_digest,
    )
    lock = tmp_path / "pip-freeze.txt"
    lock.write_text("torch==2.0.1\n", encoding="utf-8")
    request = build_aura_residual_backend_admission_request(
        _request(
            authority=authority,
            prerequisite=prerequisite,
            archive=archive,
            source_identity=source_identity,
            attestation=tmp_path / "missing-attestation.json",
            lock=lock,
        )
    )
    request_path = _write_json(tmp_path / "request.json", request)

    abstention = materialize_aura_residual_backend_abstention(
        request_path=request_path, output_path=tmp_path / "abstention.json"
    )

    assert abstention["status"] == "abstained_rights_admission_missing"
    assert abstention["provider_mutations_performed"] == 0
    assert abstention["blockers"] == [
        "aura_nested_gaussian_splatting_noncommercial_use_attestation_missing"
    ]


def test_prohibits_execution_from_a_preexisting_packet_when_rights_are_withdrawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority(tmp_path / "authority.json")
    prerequisite = _prerequisite(tmp_path / "prerequisite.json")
    archive, source_identity, license_digest, _nested = _source_archive_and_spec(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_aura_residual_backend_admission.AURA_LICENSE_SHA256",
        license_digest,
    )
    lock = tmp_path / "pip-freeze.txt"
    lock.write_text("torch==2.0.1\n", encoding="utf-8")
    request_path = _write_json(
        tmp_path / "request.json",
        build_aura_residual_backend_admission_request(
            _request(
                authority=authority,
                prerequisite=prerequisite,
                archive=archive,
                source_identity=source_identity,
                attestation=tmp_path / "missing-attestation.json",
                lock=lock,
            )
        ),
    )
    backend_abstention = materialize_aura_residual_backend_abstention(
        request_path=request_path, output_path=tmp_path / "backend-abstention.json"
    )
    legacy_backend: dict[str, object] = {
        "schema_version": "public_scene_released_code_inpainting_admission.v1",
        "status": "rights_admitted_for_private_derived_inpainting",
        "backend_id": "aurafusion360_exact_residual_multiview",
        "source_archive_sha256": backend_abstention["source_archive"]["sha256"],
        "receipt_digest": "",
    }
    legacy_backend["receipt_digest"] = canonical_digest(
        legacy_backend, digest_field="receipt_digest"
    )
    legacy_backend_path = _write_json(tmp_path / "legacy-backend.json", legacy_backend)
    packet: dict[str, object] = {
        "schema_version": "public_scene_residual_inpainting_input_packet.v1",
        "status": "exact_mask_contained_inpainting_input_packet_materialized",
        "replacement_object_count": 2,
        "backend_admission": {
            "path": str(legacy_backend_path),
            "size_bytes": legacy_backend_path.stat().st_size,
            "sha256": "sha256:" + hashlib.sha256(legacy_backend_path.read_bytes()).hexdigest(),
            "receipt_digest": legacy_backend["receipt_digest"],
        },
        "claim_boundary": {"released_code_inpainting_executed": False},
        "packet_digest": "",
    }
    packet["packet_digest"] = canonical_digest(packet, digest_field="packet_digest")
    packet_path = _write_json(tmp_path / "packet.json", packet)

    receipt = materialize_aura_residual_packet_rights_abstention(
        input_packet_path=packet_path,
        backend_abstention_path=tmp_path / "backend-abstention.json",
        output_path=tmp_path / "packet-abstention.json",
    )

    assert receipt["status"] == "inpainting_execution_prohibited_rights_admission_missing"
    assert receipt["replacement_object_count"] == 2
    assert receipt["provider_mutations_performed"] == 0
