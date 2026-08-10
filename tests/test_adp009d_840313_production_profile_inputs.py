import hashlib
import json
from pathlib import Path

from blueprint_pipeline.evaluation_run_contract import validate_evaluation_run_spec


REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "docs" / "arm_decision_proof_v1" / "manifests"
SOURCE_BUNDLE = MANIFESTS / "adp009d_840313_interiorgs_sage_source_bundle.v1.json"
EVALUATION_RUN = MANIFESTS / "adp009d_840313_evaluation_run.v1.json"
RUNTIME_READINESS = MANIFESTS / "adp009d_840313_runtime_readiness.v1.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_digest(value: dict, field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def test_source_bundle_binds_exact_admitted_interiorgs_and_sage_bytes() -> None:
    bundle = _read(SOURCE_BUNDLE)

    assert bundle["bundle_digest"] == _canonical_digest(bundle, "bundle_digest")
    assert bundle["status"] == "admitted_development_only"
    assert bundle["claim_ceiling"] == "development_only"
    assert bundle["physical_evidence"] is False
    assert bundle["rights"]["interiorgs_redistribution_allowed"] is False
    expected_component_digests = {
        "interiorgs_appearance_scene": (
            "sha256:7105f5ffdd717cde9f4f3ba6f5f48bfd81cd6e53d28d1c95e7b19625f9a6decd"
        ),
        "sage3d_collision_companion": (
            "sha256:9ec03f57dec52c4998e35ba859ab931ef34a010caddf29e6f3fb8a1e21d3d9bc"
        ),
    }
    for component in bundle["components"]:
        path = REPO / component["manifest_path"]
        assert path.is_file()
        assert (
            "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            == (expected_component_digests[component["role"]])
        )
        assert component["manifest_file_digest"] == expected_component_digests[component["role"]]
    artifacts = {row["role"]: row for row in bundle["materialized_artifacts"]}
    assert artifacts["appearance_3dgs"]["sha256"] == (
        "sha256:57c71edcb450f2323a5b8ad290b5672b437fc73b9283a7485804ce607da12254"
    )
    assert artifacts["static_collision_geometry"]["sha256"] == (
        "sha256:b265706c24f6a8ace3ee6743fd138583c4e21d83f61b99a06fd435e6ac2d6b41"
    )
    assert all(
        row["production_path"].startswith(
            "/var/lib/blueprint/task-evaluation-inputs/adp009d-840313-interiorgs-sage-v1/"
        )
        for row in artifacts.values()
    )


def test_evaluation_run_spec_is_valid_but_explicitly_dry_only() -> None:
    bundle = _read(SOURCE_BUNDLE)
    spec = _read(EVALUATION_RUN)

    validation = validate_evaluation_run_spec(spec)

    assert validation["status"] == "passed"
    assert validation["errors"] == []
    assert spec["scene_bundle"]["content_digest"] == bundle["bundle_digest"]
    assert spec["policy_adapter"]["candidate_ids"] == [
        "pi05_droid",
        "groot_n17_droid",
    ]
    assert spec["runtime_provider_profile"]["providers"] == ["vast"]
    assert spec["runtime_provider_profile"]["retry_cap"] == 0
    assert spec["metadata"]["publication_state"] == "dry_only"
    assert set(spec["metadata"]["live_blockers"]) == {
        "exact_adp009d_runtime_adapter_not_on_protected_main",
        "scripted_positive_control_not_passed",
        "allocator_artifact_manifest_not_emitted",
    }


def test_runtime_readiness_receipt_binds_the_same_typed_live_blockers() -> None:
    bundle = _read(SOURCE_BUNDLE)
    spec = _read(EVALUATION_RUN)
    readiness = _read(RUNTIME_READINESS)
    validation = validate_evaluation_run_spec(spec)

    assert readiness["receipt_digest"] == _canonical_digest(
        readiness, "receipt_digest"
    )
    assert readiness["status"] == "blocked"
    assert readiness["live_execution_enabled"] is False
    assert readiness["provider_mutation_performed"] is False
    assert readiness["source_bundle_digest"] == bundle["bundle_digest"]
    assert readiness["evaluation_run_spec_digest"] == validation["spec_digest"]
    assert readiness["blockers"] == spec["metadata"]["live_blockers"]
