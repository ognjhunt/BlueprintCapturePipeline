from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.portable_evidence_receipt_mirror import (
    RECEIPT_MIRROR_DIGEST_FIELD,
    RECEIPT_MIRROR_SCHEMA_VERSION,
    PortableEvidenceReceiptMirrorError,
    materialize_portable_evidence_receipt_mirror,
)


def _receipt(schema_version: str, digest_field: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": schema_version,
        "status": "sealed",
        digest_field: "",
    }
    payload[digest_field] = canonical_digest(payload, digest_field=digest_field)
    return payload


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_mirror_copies_only_verified_receipts_and_seals_manifest(tmp_path: Path) -> None:
    source = tmp_path / "rights_bounded"
    output = tmp_path / "portable"
    source.mkdir()
    output.mkdir()
    attempt = _receipt("example_attempt.v1", "receipt_digest")
    replay = _receipt("example_replay.v1", "replay_digest")
    _write(source / "task_a/attempt.json", attempt)
    _write(source / "task_a/replay.json", replay)

    manifest = materialize_portable_evidence_receipt_mirror(
        source_root=source,
        output_root=output,
        source_root_id="rights_bounded_fixture",
        receipt_relative_paths=["task_a/attempt.json", "task_a/replay.json"],
        admitted_schema_digest_fields={
            "example_attempt.v1": "receipt_digest",
            "example_replay.v1": "replay_digest",
        },
        output_relative_path="shared/receipt_mirror.v1.json",
    )

    assert manifest["schema_version"] == RECEIPT_MIRROR_SCHEMA_VERSION
    assert manifest["receipt_mirror_digest"] == canonical_digest(
        manifest, digest_field=RECEIPT_MIRROR_DIGEST_FIELD
    )
    assert manifest["raw_dataset_bytes_copied"] is False
    assert manifest["scene_media_copied"] is False
    assert (output / "task_a/attempt.json").read_bytes() == (
        source / "task_a/attempt.json"
    ).read_bytes()
    assert (output / "task_a/replay.json").read_bytes() == (
        source / "task_a/replay.json"
    ).read_bytes()


def test_mirror_rejects_unsealed_or_non_json_input(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    invalid = _receipt("example_attempt.v1", "receipt_digest")
    invalid["status"] = "tampered"
    _write(source / "invalid.json", invalid)
    (source / "raw.ply").write_bytes(b"not allowed")

    with pytest.raises(
        PortableEvidenceReceiptMirrorError,
        match="portable_receipt_mirror_digest_invalid",
    ):
        materialize_portable_evidence_receipt_mirror(
            source_root=source,
            output_root=output,
            source_root_id="fixture",
            receipt_relative_paths=["invalid.json"],
            admitted_schema_digest_fields={"example_attempt.v1": "receipt_digest"},
            output_relative_path="mirror.json",
        )

    with pytest.raises(
        PortableEvidenceReceiptMirrorError,
        match="portable_receipt_mirror_non_json_forbidden",
    ):
        materialize_portable_evidence_receipt_mirror(
            source_root=source,
            output_root=output,
            source_root_id="fixture",
            receipt_relative_paths=["raw.ply"],
            admitted_schema_digest_fields={"example_attempt.v1": "receipt_digest"},
            output_relative_path="mirror.json",
        )


def test_mirror_rejects_duplicate_and_outside_receipt_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    receipt = _receipt("example_attempt.v1", "receipt_digest")
    _write(source / "receipt.json", receipt)

    common = {
        "source_root": source,
        "output_root": output,
        "source_root_id": "fixture",
        "admitted_schema_digest_fields": {"example_attempt.v1": "receipt_digest"},
        "output_relative_path": "mirror.json",
    }
    with pytest.raises(
        PortableEvidenceReceiptMirrorError,
        match="portable_receipt_mirror_duplicate_path",
    ):
        materialize_portable_evidence_receipt_mirror(
            **common,
            receipt_relative_paths=["receipt.json", "receipt.json"],
        )
    with pytest.raises(
        PortableEvidenceReceiptMirrorError,
        match="portable_receipt_mirror_path_outside_root",
    ):
        materialize_portable_evidence_receipt_mirror(
            **common,
            receipt_relative_paths=["../receipt.json"],
        )
