from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.adp009d_840313_runtime_bundle import (
    RUNTIME_BUNDLE_ID,
    RuntimeBundleError,
    validate_runtime_bundle_manifest,
    verify_materialized_runtime_inputs,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


REPO = Path(__file__).resolve().parents[1]
MANIFEST = (
    REPO
    / "docs/arm_decision_proof_v1/manifests/adp009d_840313_franka_runtime_bundle.v1.json"
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_runtime_bundle_is_canonical_and_explicit_about_legacy_lineage() -> None:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert validate_runtime_bundle_manifest(value) == []
    assert value["runtime_bundle_digest"] == canonical_digest(
        value, digest_field="runtime_bundle_digest"
    )
    assert "does_not_bind_source_digest" not in value["construction_lineage"][
        "legacy_receipt_limit"
    ]
    assert "predates source-digest binding" in value["construction_lineage"][
        "legacy_receipt_limit"
    ]
    assert value["construction_lineage"]["claim_ceiling"] == "development_only"


def test_runtime_input_verifier_checks_every_byte_and_confines_paths(tmp_path: Path) -> None:
    runtime = tmp_path / RUNTIME_BUNDLE_ID
    source = tmp_path / "source"
    repo = tmp_path / "repo"
    runtime.mkdir()
    source.mkdir()
    repo.mkdir()
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for row in value["materialized_artifacts"]:
        path = runtime / Path(row["production_path"]).name
        path.write_bytes(row["role"].encode())
        row["size_bytes"] = path.stat().st_size
        row["sha256"] = _digest(path)
    for row in value["repository_inputs"]:
        path = repo / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(row["role"].encode())
        row["size_bytes"] = path.stat().st_size
        row["sha256"] = _digest(path)
    for row in value["source_bundle_inputs"]:
        path = source / row["filename"]
        path.write_bytes(row["role"].encode())
        row["size_bytes"] = path.stat().st_size
        row["sha256"] = _digest(path)
    value["runtime_bundle_digest"] = canonical_digest(
        value, digest_field="runtime_bundle_digest"
    )

    verified = verify_materialized_runtime_inputs(
        value,
        runtime_input_root=runtime,
        source_input_root=source,
        repo_root=repo,
    )
    assert len(verified) == 11

    (runtime / "aura_ghost_removed_appearance.usdz").write_bytes(b"drift")
    with pytest.raises(RuntimeBundleError, match="aura_nurec_appearance"):
        verify_materialized_runtime_inputs(
            value,
            runtime_input_root=runtime,
            source_input_root=source,
            repo_root=repo,
        )


def test_runtime_manifest_rejects_missing_role_and_digest_drift() -> None:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    value["materialized_artifacts"].pop()

    blockers = validate_runtime_bundle_manifest(value)

    assert "runtime_bundle_materialized_artifacts_roles_invalid" in blockers
    assert "runtime_bundle_digest_invalid" in blockers
