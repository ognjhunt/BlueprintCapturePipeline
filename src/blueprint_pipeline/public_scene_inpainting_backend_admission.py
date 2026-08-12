"""Mint the only sanctioned released-code inpainting rights admission.

``public_scene_residual_inpainting_packet`` refuses to build a residual
inpainting input packet without a file-backed
``public_scene_released_code_inpainting_admission.v1`` receipt.  Nothing in the
repository produced one, so the only way to satisfy that consumer was to
hand-author the receipt -- a caller assertion wearing a receipt's shape.  This
module closes that hole by deriving the receipt from license documents on disk.

The rule the three surveyed backends each violate differently is the same one:
*a runtime artifact never inherits the license of the repository that ships the
code that loads it.*  Inpaint360GS publishes Apache-2.0 code and an author
dataset with no established license authority; InFusion publishes an Apache-2.0
adapter and a checkpoint declaring no license at all.  Admission therefore
requires every required runtime artifact -- code, weights, and dataset alike --
to carry its own license document, and it rejects a non-code artifact that
points back at the code license file.

When any artifact fails, this module does not raise: an unadmitted backend is an
expected outcome, so it seals the smallest typed abstention instead.  It raises
only for a request that is malformed or that asserts its own admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json


REQUEST_SCHEMA = "public_scene_inpainting_backend_admission_request.v1"
ADMISSION_SCHEMA = "public_scene_released_code_inpainting_admission.v1"
ABSTENTION_SCHEMA = "public_scene_inpainting_backend_admission_abstention.v1"

# Identifiers that unambiguously permit private derivative use.  Anything else --
# non-commercial, research-only, or undeclared -- fails closed.
PRIVATE_DERIVED_ADMISSIBLE_LICENSES: dict[str, tuple[str, ...]] = {
    "Apache-2.0": ("apache license", "version 2.0"),
    "MIT": ("permission is hereby granted, free of charge",),
    "BSD-2-Clause": ("redistribution and use in source and binary forms",),
    "BSD-3-Clause": ("redistribution and use in source and binary forms",),
    "CC0-1.0": ("cc0",),
}

# The producer fixes these; a caller cannot widen its own handling terms.
_FIXED_UPLOAD_POLICY = {
    "raw_dataset_bytes_upload": False,
    "private_derived_upload": True,
    "provider_training": False,
    "publication": False,
}
_CALLER_ASSERTED_KEYS = frozenset(
    {
        "status",
        "admit",
        "admitted",
        "rights_admitted",
        "receipt_digest",
        "private_derived_upload_policy",
        "license_evidence",
    }
)


class InpaintingBackendAdmissionError(ValueError):
    """Stable fail-closed errors for an unbound or self-asserting request."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _file(value: Any, *, code: str) -> Path:
    path = Path(str(value or "")).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise InpaintingBackendAdmissionError([code])
    return path


def _text(value: Any) -> str:
    return str(value or "").strip()


def _validate_request(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InpaintingBackendAdmissionError(["request_not_an_object"])
    if value.get("schema_version") != REQUEST_SCHEMA:
        raise InpaintingBackendAdmissionError(["request_schema_invalid"])
    if _CALLER_ASSERTED_KEYS.intersection(value):
        raise InpaintingBackendAdmissionError(["caller_asserted_admission_forbidden"])
    if not all(
        _text(value.get(key))
        for key in ("backend_id", "source_repository", "source_revision", "model_identity")
    ):
        raise InpaintingBackendAdmissionError(["backend_identity_incomplete"])

    retention = value.get("maximum_retention_days")
    if isinstance(retention, bool) or not isinstance(retention, int) or retention <= 0:
        raise InpaintingBackendAdmissionError(["maximum_retention_days_invalid"])

    artifacts = value.get("required_runtime_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise InpaintingBackendAdmissionError(["required_runtime_artifacts_missing"])
    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or not all(
            _text(artifact.get(key)) for key in ("artifact_id", "kind")
        ):
            raise InpaintingBackendAdmissionError(["runtime_artifact_identity_incomplete"])
    if not any(_text(artifact.get("kind")) == "code" for artifact in artifacts):
        raise InpaintingBackendAdmissionError(["released_code_artifact_missing"])

    try:
        request = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise InpaintingBackendAdmissionError(["request_not_json_serializable"]) from exc
    return request


def _code_license_digests(artifacts: Sequence[Mapping[str, Any]]) -> set[str]:
    digests: set[str] = set()
    for artifact in artifacts:
        if _text(artifact.get("kind")) != "code":
            continue
        document = artifact.get("license_document_path")
        if not _text(document):
            continue
        path = Path(str(document)).expanduser().resolve()
        if path.is_file() and not path.is_symlink():
            digests.add(_sha256(path))
    return digests


def _evaluate_artifact(
    artifact: Mapping[str, Any], *, code_license_digests: set[str]
) -> tuple[dict[str, Any], str | None]:
    """Return the artifact's license evidence and its blocker, if any."""
    artifact_id = _text(artifact.get("artifact_id"))
    kind = _text(artifact.get("kind"))
    declared = _text(artifact.get("declared_license"))
    document = _text(artifact.get("license_document_path"))
    evidence: dict[str, Any] = {
        "artifact_id": artifact_id,
        "kind": kind,
        "declared_license": declared or None,
        "license_document": None,
    }

    if not document or not declared:
        return evidence, f"{artifact_id}_license_document_missing"
    path = Path(document).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        return evidence, f"{artifact_id}_license_document_unreadable"
    try:
        body = path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return evidence, f"{artifact_id}_license_document_unreadable"

    digest = _sha256(path)
    evidence["license_document"] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
    }

    markers = PRIVATE_DERIVED_ADMISSIBLE_LICENSES.get(declared)
    if markers is None:
        return evidence, f"{artifact_id}_license_not_private_derived_admissible"
    if kind != "code" and digest in code_license_digests:
        return evidence, f"{artifact_id}_license_inherited_from_source_code_license"
    if not all(marker in body for marker in markers):
        return evidence, f"{artifact_id}_license_document_does_not_support_declared_license"
    return evidence, None


def materialize_inpainting_backend_admission(
    request: Mapping[str, Any], *, output_path: Path
) -> dict[str, Any]:
    """Seal an admission when every runtime artifact clears, else an abstention."""
    validated = _validate_request(request)
    artifacts = validated["required_runtime_artifacts"]

    code_license_digests = _code_license_digests(artifacts)
    license_evidence: list[dict[str, Any]] = []
    blockers: list[str] = []
    for artifact in artifacts:
        evidence, blocker = _evaluate_artifact(
            artifact, code_license_digests=code_license_digests
        )
        license_evidence.append(evidence)
        if blocker:
            blockers.append(blocker)

    receipt: dict[str, Any] = {
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009D",
        "backend_id": validated["backend_id"],
        "source_repository": validated["source_repository"],
        "source_revision": validated["source_revision"],
        "model_identity": validated["model_identity"],
        "license_evidence": license_evidence,
        "source_code_license_not_inherited_by_runtime_artifacts": True,
        "raw_secret_values_recorded": False,
    }

    if blockers:
        unadmitted = sorted({blocker.rsplit("_license_", 1)[0] for blocker in blockers})
        receipt.update(
            {
                "schema_version": ABSTENTION_SCHEMA,
                "status": "abstained_backend_rights_not_admitted",
                "blockers": sorted(set(blockers)),
                "smallest_missing_capability": (
                    "a released-code inpainting backend whose required runtime artifacts "
                    f"({', '.join(unadmitted)}) each carry a file-backed license admitting "
                    "private derived use"
                ),
                "claim_ceiling": "backend_rights_unadmitted_no_inpainting_authorized",
            }
        )
    else:
        # Exact bytes matter only for a backend we are about to authorize.
        archive = _file(validated.get("source_archive_path"), code="source_archive_unreadable")
        lock = _file(validated.get("environment_lock_path"), code="environment_lock_unreadable")
        receipt.update(
            {
                "schema_version": ADMISSION_SCHEMA,
                "status": "rights_admitted_for_private_derived_inpainting",
                "source_archive_sha256": _sha256(archive),
                "environment_lock_sha256": _sha256(lock),
                "private_derived_upload_policy": {
                    **_FIXED_UPLOAD_POLICY,
                    "maximum_retention_days": validated["maximum_retention_days"],
                },
                "claim_ceiling": "private_derived_inpainting_inputs_only",
            }
        )

    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    receipt = materialize_inpainting_backend_admission(request, output_path=args.output)
    print(canonical_json({"status": receipt["status"], "receipt_digest": receipt["receipt_digest"]}))
    return 0 if receipt["schema_version"] == ADMISSION_SCHEMA else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ABSTENTION_SCHEMA",
    "ADMISSION_SCHEMA",
    "PRIVATE_DERIVED_ADMISSIBLE_LICENSES",
    "REQUEST_SCHEMA",
    "InpaintingBackendAdmissionError",
    "materialize_inpainting_backend_admission",
]
