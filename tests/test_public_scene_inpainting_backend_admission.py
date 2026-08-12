from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_inpainting_backend_admission import (
    ABSTENTION_SCHEMA,
    ADMISSION_SCHEMA,
    REQUEST_SCHEMA,
    InpaintingBackendAdmissionError,
    materialize_inpainting_backend_admission,
)
from blueprint_pipeline.public_scene_residual_inpainting_packet import (
    BACKEND_ADMISSION_SCHEMA,
    ResidualInpaintingInputPacketError,
    _validate_backend_admission,
)


APACHE_TEXT = (
    "Apache License\nVersion 2.0, January 2004\nhttp://www.apache.org/licenses/\n"
    "Licensed under the Apache License, Version 2.0 (the \"License\");\n"
)
MIT_TEXT = (
    "MIT License\n\nPermission is hereby granted, free of charge, to any person "
    "obtaining a copy of this software and associated documentation files\n"
)
NON_COMMERCIAL_TEXT = (
    "Attribution-NonCommercial 4.0 International\n"
    "You may not use the material for commercial purposes.\n"
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _artifact(
    root: Path,
    artifact_id: str,
    kind: str,
    *,
    declared_license: str | None,
    license_text: str | None,
    license_name: str = "LICENSE.txt",
) -> dict[str, object]:
    document = None
    if license_text is not None:
        document = str(_write(root / artifact_id / license_name, license_text))
    return {
        "artifact_id": artifact_id,
        "kind": kind,
        "declared_license": declared_license,
        "license_document_path": document,
    }


def _request(root: Path, artifacts: list[dict[str, object]]) -> dict[str, object]:
    archive = _write(root / "source" / "archive.tar", "source-archive-bytes")
    lock = _write(root / "source" / "environment.lock", "torch==2.4.0\n")
    return {
        "schema_version": REQUEST_SCHEMA,
        "backend_id": "released-code-inpainting-backend",
        "source_repository": "https://github.com/example/backend",
        "source_revision": "b" * 40,
        "source_archive_path": str(archive),
        "environment_lock_path": str(lock),
        "model_identity": "backend-weights-v1",
        "maximum_retention_days": 30,
        "required_runtime_artifacts": artifacts,
    }


def _permissive_artifacts(root: Path) -> list[dict[str, object]]:
    return [
        _artifact(
            root, "source_code", "code", declared_license="Apache-2.0", license_text=APACHE_TEXT
        ),
        _artifact(
            root, "weights", "weights", declared_license="MIT", license_text=MIT_TEXT
        ),
    ]


def _privacy() -> dict[str, object]:
    return {"maximum_retention_days": 30}


def test_admits_backend_when_every_runtime_artifact_carries_permissive_license(
    tmp_path: Path,
) -> None:
    output = tmp_path / "admission.json"

    receipt = materialize_inpainting_backend_admission(
        _request(tmp_path, _permissive_artifacts(tmp_path)), output_path=output
    )

    assert receipt["schema_version"] == ADMISSION_SCHEMA
    assert receipt["status"] == "rights_admitted_for_private_derived_inpainting"
    assert json.loads(output.read_text(encoding="utf-8")) == receipt


def test_admitted_receipt_is_accepted_by_the_residual_inpainting_packet(
    tmp_path: Path,
) -> None:
    output = tmp_path / "admission.json"
    materialize_inpainting_backend_admission(
        _request(tmp_path, _permissive_artifacts(tmp_path)), output_path=output
    )

    admission, record = _validate_backend_admission(output, privacy=_privacy())

    assert ADMISSION_SCHEMA == BACKEND_ADMISSION_SCHEMA
    assert admission["backend_id"] == "released-code-inpainting-backend"
    assert record["sha256"] == _sha256(output)


def test_abstains_when_a_runtime_artifact_declares_no_license(tmp_path: Path) -> None:
    artifacts = _permissive_artifacts(tmp_path)
    artifacts.append(
        _artifact(
            tmp_path, "checkpoint", "weights", declared_license=None, license_text=None
        )
    )
    output = tmp_path / "admission.json"

    receipt = materialize_inpainting_backend_admission(
        _request(tmp_path, artifacts), output_path=output
    )

    assert receipt["schema_version"] == ABSTENTION_SCHEMA
    assert receipt["status"] == "abstained_backend_rights_not_admitted"
    assert receipt["blockers"] == ["checkpoint_license_document_missing"]
    assert "checkpoint" in receipt["smallest_missing_capability"]


def test_abstains_when_a_runtime_artifact_forbids_commercial_use(tmp_path: Path) -> None:
    artifacts = _permissive_artifacts(tmp_path)
    artifacts.append(
        _artifact(
            tmp_path,
            "author_dataset",
            "dataset",
            declared_license="CC-BY-NC-4.0",
            license_text=NON_COMMERCIAL_TEXT,
        )
    )
    output = tmp_path / "admission.json"

    receipt = materialize_inpainting_backend_admission(
        _request(tmp_path, artifacts), output_path=output
    )

    assert receipt["status"] == "abstained_backend_rights_not_admitted"
    assert receipt["blockers"] == ["author_dataset_license_not_private_derived_admissible"]


def test_abstains_when_a_dataset_reuses_the_source_code_license_document(
    tmp_path: Path,
) -> None:
    """A code license never propagates to the weights or dataset a backend needs."""
    code = _artifact(
        tmp_path, "source_code", "code", declared_license="Apache-2.0", license_text=APACHE_TEXT
    )
    dataset = {
        "artifact_id": "author_dataset",
        "kind": "dataset",
        "declared_license": "Apache-2.0",
        "license_document_path": code["license_document_path"],
    }
    output = tmp_path / "admission.json"

    receipt = materialize_inpainting_backend_admission(
        _request(tmp_path, [code, dataset]), output_path=output
    )

    assert receipt["status"] == "abstained_backend_rights_not_admitted"
    assert receipt["blockers"] == [
        "author_dataset_license_inherited_from_source_code_license"
    ]


def test_abstains_when_the_license_document_contradicts_the_declared_identifier(
    tmp_path: Path,
) -> None:
    artifacts = [
        _artifact(
            tmp_path, "source_code", "code", declared_license="Apache-2.0", license_text=APACHE_TEXT
        ),
        _artifact(
            tmp_path,
            "weights",
            "weights",
            declared_license="MIT",
            license_text=NON_COMMERCIAL_TEXT,
        ),
    ]
    output = tmp_path / "admission.json"

    receipt = materialize_inpainting_backend_admission(
        _request(tmp_path, artifacts), output_path=output
    )

    assert receipt["status"] == "abstained_backend_rights_not_admitted"
    assert receipt["blockers"] == ["weights_license_document_does_not_support_declared_license"]


def test_abstains_without_requiring_exact_bytes_of_a_refused_backend(
    tmp_path: Path,
) -> None:
    """A backend we are about to refuse need not be sealed byte-exact first."""
    artifacts = _permissive_artifacts(tmp_path)
    artifacts.append(
        _artifact(tmp_path, "checkpoint", "weights", declared_license=None, license_text=None)
    )
    request = _request(tmp_path, artifacts)
    request["source_archive_path"] = str(tmp_path / "absent" / "archive.tar")
    request["environment_lock_path"] = str(tmp_path / "absent" / "environment.lock")

    receipt = materialize_inpainting_backend_admission(
        request, output_path=tmp_path / "abstention.json"
    )

    assert receipt["status"] == "abstained_backend_rights_not_admitted"
    assert receipt["blockers"] == ["checkpoint_license_document_missing"]


def test_admission_still_requires_exact_source_bytes(tmp_path: Path) -> None:
    request = _request(tmp_path, _permissive_artifacts(tmp_path))
    request["source_archive_path"] = str(tmp_path / "absent" / "archive.tar")

    with pytest.raises(InpaintingBackendAdmissionError) as excinfo:
        materialize_inpainting_backend_admission(
            request, output_path=tmp_path / "admission.json"
        )

    assert excinfo.value.codes == ("source_archive_unreadable",)


def test_rejects_a_request_that_asserts_its_own_admission(tmp_path: Path) -> None:
    request = _request(tmp_path, _permissive_artifacts(tmp_path))
    request["status"] = "rights_admitted_for_private_derived_inpainting"

    with pytest.raises(InpaintingBackendAdmissionError) as excinfo:
        materialize_inpainting_backend_admission(
            request, output_path=tmp_path / "admission.json"
        )

    assert excinfo.value.codes == ("caller_asserted_admission_forbidden",)


def test_rejects_a_request_without_any_runtime_artifact(tmp_path: Path) -> None:
    request = _request(tmp_path, [])

    with pytest.raises(InpaintingBackendAdmissionError) as excinfo:
        materialize_inpainting_backend_admission(
            request, output_path=tmp_path / "admission.json"
        )

    assert excinfo.value.codes == ("required_runtime_artifacts_missing",)


def test_rejects_a_request_without_a_code_artifact(tmp_path: Path) -> None:
    artifacts = [
        _artifact(tmp_path, "weights", "weights", declared_license="MIT", license_text=MIT_TEXT)
    ]
    request = _request(tmp_path, artifacts)

    with pytest.raises(InpaintingBackendAdmissionError) as excinfo:
        materialize_inpainting_backend_admission(
            request, output_path=tmp_path / "admission.json"
        )

    assert excinfo.value.codes == ("released_code_artifact_missing",)


def test_admission_receipt_digest_binds_the_license_evidence(tmp_path: Path) -> None:
    output = tmp_path / "admission.json"
    receipt = materialize_inpainting_backend_admission(
        _request(tmp_path, _permissive_artifacts(tmp_path)), output_path=output
    )

    tampered = dict(receipt)
    tampered["license_evidence"] = [
        {**row, "declared_license": "MIT"} for row in receipt["license_evidence"]
    ]

    assert tampered["receipt_digest"] != canonical_digest(
        tampered, digest_field="receipt_digest"
    )
    output.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ResidualInpaintingInputPacketError):
        _validate_backend_admission(output, privacy=_privacy())


def test_abstention_never_satisfies_the_residual_inpainting_packet(tmp_path: Path) -> None:
    artifacts = _permissive_artifacts(tmp_path)
    artifacts.append(
        _artifact(tmp_path, "checkpoint", "weights", declared_license=None, license_text=None)
    )
    output = tmp_path / "abstention.json"
    materialize_inpainting_backend_admission(_request(tmp_path, artifacts), output_path=output)

    with pytest.raises(ResidualInpaintingInputPacketError) as excinfo:
        _validate_backend_admission(output, privacy=_privacy())

    assert "residual_inpainting_backend_admission_invalid" in excinfo.value.codes


def test_retention_beyond_the_packet_privacy_budget_is_rejected_downstream(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, _permissive_artifacts(tmp_path))
    request["maximum_retention_days"] = 45
    output = tmp_path / "admission.json"
    receipt = materialize_inpainting_backend_admission(request, output_path=output)

    assert receipt["private_derived_upload_policy"]["maximum_retention_days"] == 45
    with pytest.raises(ResidualInpaintingInputPacketError):
        _validate_backend_admission(output, privacy=_privacy())
