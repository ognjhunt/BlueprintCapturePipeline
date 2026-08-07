from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_hybrid_replacement_seal import (
    HybridReplacementSealError,
    materialize_hybrid_replacement_seal,
)


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _record(path: Path, relative_path: str) -> dict[str, object]:
    return {
        "relative_path": relative_path,
        "size_bytes": path.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _canonical(path: Path, value: dict[str, object], field: str) -> None:
    value[field] = canonical_digest(value, digest_field=field)
    _write(path, value)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    asset_path = repo / "asset.usda"
    asset_path.write_bytes(b"exact-simready-v2")
    asset = _record(asset_path, "asset.usda")
    composition_path = data / "composition.usda"
    composition_path.write_bytes(b"sage-plus-replacement")
    composition = _record(composition_path, "composition.usda")
    composition.update(
        {
            "source_target_collider_active": False,
            "support_collider_active": True,
            "replacement_rigid_body_active": True,
            "replacement_collision_api_present": True,
            "unresolved_dependency_count": 0,
        }
    )

    aura: dict[str, object] = {
        "schema_version": "adp009b_aura_human_visual_review.v1",
        "status": "human_accepted_visual_candidate_for_internal_hybrid_control",
        "scene": {"publisher_scene_id": "840313", "target_instance_id": "ins160"},
        "bindings": {
            "aura_execution_receipt_digest": "sha256:aura",
            "native_visual_review_receipt_digest": "sha256:native",
        },
        "observed_quality": {"human_visual_acceptance": True, "view_count": 8},
        "technical_admission": False,
        "receipt_digest": "",
    }
    _canonical(repo / "aura.json", aura, "receipt_digest")

    replacement: dict[str, object] = {
        "schema_version": "adp009b_simready_replacement_receipt.v1",
        "status": "composed_static_candidate",
        "scene": {
            "publisher_scene_id": "840313",
            "target_instance_id": "160",
            "target_semantic_label": "canned_beverage",
        },
        "bindings": {
            "aura_execution_receipt_digest": "sha256:aura",
            "exact_simready_asset": asset,
        },
        "composition": composition,
        "receipt_digest": "",
    }
    _canonical(repo / "replacement.json", replacement, "receipt_digest")

    static: dict[str, object] = {
        "status": "statically_validated",
        "checks": {
            "simready_foundation_profile_passed": True,
            "usd_readback_passed": True,
        },
        "usd": asset,
        "receipt_digest": "",
    }
    _canonical(repo / "static.json", static, "receipt_digest")

    manifest: dict[str, object] = {
        "scene_mapping": {
            "publisher_scene_id": "840313",
            "interiorgs_instance_id": "160",
        },
        "observed_evidence": {
            "probe_summaries": [
                {"probe": name, "passed": True}
                for name in ("drop", "slide", "tip", "gripper")
            ],
            "provider_zero_verified": True,
            "object_store_zero_verified": True,
            "runtime_result_digest": "sha256:runtime",
        },
        "manifest_digest": "",
    }
    _canonical(repo / "manifest.json", manifest, "manifest_digest")
    receipt: dict[str, object] = {
        "role": "exact_simready_object",
        "status": "admitted",
        "component_manifest_digest": manifest["manifest_digest"],
        "checks": {
            "cad_materialization_receipt_digest_verified": True,
            "simready_foundation_profile_passed": True,
            "isaac_dynamic_probes_passed": True,
            "teardown_and_object_cleanup_verified": True,
            "usd_bytes_verified": True,
        },
        "receipt_digest": "",
    }
    _canonical(repo / "component_receipt.json", receipt, "receipt_digest")
    request: dict[str, object] = {
        "schema_version": "adp009b_hybrid_replacement_seal_request.v1",
        "aura_human_review_receipt_path": "aura.json",
        "replacement_receipt_path": "replacement.json",
        "static_validation_receipt_path": "static.json",
        "exact_simready_component_manifest_path": "manifest.json",
        "exact_simready_component_receipt_path": "component_receipt.json",
    }
    request_path = repo / "request.json"
    _write(request_path, request)
    return repo, data, request_path, repo / "seal.json"


def _materialize(fixture: tuple[Path, Path, Path, Path]) -> dict[str, object]:
    repo, data, request, output = fixture
    return materialize_hybrid_replacement_seal(
        request_path=request, repo_root=repo, data_root=data, output_path=output
    )


def test_seals_exact_internal_hybrid_without_overclaiming(tmp_path: Path) -> None:
    receipt = _materialize(_fixture(tmp_path))

    assert receipt["status"] == "sealed_internal_hybrid_replacement_control"
    assert receipt["observed_gates"]["four_native_isaac_probes_passed"] is True
    assert receipt["adp009b_complete"] is False
    assert receipt["technical_inpainting_admitted"] is False
    assert receipt["physical_evidence"] is False
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_rejects_changed_composition_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    (fixture[1] / "composition.usda").write_bytes(b"changed")

    with pytest.raises(HybridReplacementSealError, match="composition_bytes_changed"):
        _materialize(fixture)


def test_rejects_scene_identity_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    replacement_path = fixture[0] / "replacement.json"
    replacement = json.loads(replacement_path.read_text(encoding="utf-8"))
    replacement["scene"]["publisher_scene_id"] = "different"
    _canonical(replacement_path, replacement, "receipt_digest")

    with pytest.raises(HybridReplacementSealError, match="scene_or_target_identity_mismatch"):
        _materialize(fixture)


def test_rejects_caller_asserted_completion(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    request = json.loads(fixture[2].read_text(encoding="utf-8"))
    request["complete"] = True
    _write(fixture[2], request)

    with pytest.raises(HybridReplacementSealError, match="caller_asserted_seal_forbidden"):
        _materialize(fixture)
