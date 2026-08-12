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
    materialize_aura_residual_backend_admission,
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
        "private_rights_admitted_scene_derived_uploads_authorized": True,
        "raw_interiorgs_upload_authorized": False,
        "training_authorized": False,
        "public_dataset_bytes_publication_authorized": False,
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


def _source_archive_and_spec(root: Path) -> tuple[Path, Path, str]:
    members = {
        "LICENSE": b"Apache License\nunit-test\n",
        "inpaint.py": b"print('released source')\n",
        "arguments/__init__.py": b"class Params: pass\n",
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
    return archive, _write_json(root / "source_identity.json", spec), _sha256_bytes(members["LICENSE"])


def _request(
    *, authority: Path, prerequisite: Path, archive: Path, source_identity: Path, lock: Path
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
    archive, source_identity, license_digest = _source_archive_and_spec(tmp_path)
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_aura_residual_backend_admission.AURA_LICENSE_SHA256",
        license_digest,
    )
    lock = (tmp_path / "pip-freeze.txt")
    lock.write_text("torch==2.0.1\n", encoding="utf-8")
    request = build_aura_residual_backend_admission_request(
        _request(
            authority=authority,
            prerequisite=prerequisite,
            archive=archive,
            source_identity=source_identity,
            lock=lock,
        )
    )
    request_path = _write_json(tmp_path / "request.json", request)

    receipt = materialize_aura_residual_backend_admission(
        request_path=request_path, output_path=tmp_path / "receipt.json"
    )

    assert receipt["status"] == "rights_admitted_for_private_derived_inpainting"
    assert receipt["mask_dilation_pixels"] == 0
    assert receipt["claim_boundary"]["raw_dataset_bytes_upload_authorized"] is False
    assert receipt["source_identity_provenance"]["source_member_count"] == 3


def test_rejects_archive_that_does_not_match_pinned_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority(tmp_path / "authority.json")
    prerequisite = _prerequisite(tmp_path / "prerequisite.json")
    archive, source_identity, license_digest = _source_archive_and_spec(tmp_path)
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_aura_residual_backend_admission.AURA_LICENSE_SHA256",
        license_digest,
    )
    with zipfile.ZipFile(archive, "a") as output:
        output.writestr("extra.py", b"not-pinned")
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
                lock=lock,
            )
        ),
    )

    with pytest.raises(AuraResidualBackendAdmissionError) as raised:
        materialize_aura_residual_backend_admission(
            request_path=request_path, output_path=tmp_path / "receipt.json"
        )

    assert raised.value.codes == ("aura_residual_source_archive_manifest_mismatch",)


def test_rejects_provider_training_permission(tmp_path: Path) -> None:
    authority = _authority(tmp_path / "authority.json")
    prerequisite = _prerequisite(tmp_path / "prerequisite.json")
    archive, source_identity, _ = _source_archive_and_spec(tmp_path)
    lock = tmp_path / "pip-freeze.txt"
    lock.write_text("torch==2.0.1\n", encoding="utf-8")
    request = _request(
        authority=authority,
        prerequisite=prerequisite,
        archive=archive,
        source_identity=source_identity,
        lock=lock,
    )
    policy = request["private_derived_upload_policy"]
    assert isinstance(policy, dict)
    policy["provider_training"] = True

    with pytest.raises(AuraResidualBackendAdmissionError) as raised:
        build_aura_residual_backend_admission_request(request)

    assert raised.value.codes == ("aura_residual_request_private_upload_policy_invalid",)
