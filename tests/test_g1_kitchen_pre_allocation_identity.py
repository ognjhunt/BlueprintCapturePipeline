from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path
from typing import Any

from blueprint_pipeline.g1_kitchen_bundle_compatibility import (
    build_bundle_compatibility,
)
from blueprint_pipeline.g1_kitchen_pre_allocation_identity import (
    REGISTRY_EVIDENCE_SCHEMA_VERSION,
    enforce_pre_allocation_identity_gate,
    revalidate_attempt_artifact_bytes,
)
from blueprint_pipeline.g1_kitchen_worker_image_evidence import (
    assemble_worker_image_runtime_evidence,
    validate_worker_image_runtime_evidence,
)
from blueprint_pipeline.kitchen_attempt_lineage import build_attempt_input_manifest

IMAGE_HASH = "d" * 64
SOURCE_COMMIT = "b" * 40
DIRTY_PATCH = "e" * 64

SEALED_HEALTHCHECK_PATH = (
    Path(__file__).resolve().parents[1]
    / "deploy"
    / "docker"
    / "robot_eval_worker"
    / "groot_oscar_closed_loop"
    / "groot_oscar_closed_loop_image_healthcheck.py"
)


def _load_sealed_healthcheck_module():
    spec = importlib.util.spec_from_file_location(
        "groot_oscar_closed_loop_image_healthcheck", SEALED_HEALTHCHECK_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sealed_runtime_metadata() -> dict[str, Any]:
    module = _load_sealed_healthcheck_module()
    return module.build_runtime_metadata(
        env={
            "BLUEPRINT_WORKER_IMAGE_FAMILY": "isaac-eval-worker",
            "BLUEPRINT_WORKER_IMAGE_VARIANT": "groot-oscar-closed-loop",
            "BLUEPRINT_SIMULATOR_FRAMEWORK": "isaac_sim",
            "BLUEPRINT_ISAAC_SIM_MAJOR_VERSION": "6",
        },
        isaac_imported=True,
        g1_exists=True,
        asset_binding_valid=True,
        official_assets_exist=True,
        source_commit=SOURCE_COMMIT,
        dirty_patch=DIRTY_PATCH,
    )


def _canary(name: str) -> dict[str, Any]:
    canary = {
        "status": "passed",
        "image_digest": f"sha256:{IMAGE_HASH}",
        "provider_allocation_id": "do-1",
        "run_id": "run-1",
        "attempt_id": "attempt-1",
        "launch_nonce": "nonce-1",
    }
    if name == "review":
        canary.update({"width": 640, "height": 480})
    return canary


def _worker_evidence() -> dict[str, Any]:
    return assemble_worker_image_runtime_evidence(
        image_digest=f"sha256:{IMAGE_HASH}",
        source_commit=SOURCE_COMMIT,
        source_dirty_patch_sha256=DIRTY_PATCH,
        build_healthcheck={
            "status": "passed",
            "runtime_metadata": _sealed_runtime_metadata(),
        },
        fast_canary=_canary("fast"),
        review_canary=_canary("review"),
        teardown={"api_confirmed": True, "terminal_state": "terminated"},
        final_inventory={"api_confirmed": True, "live_resource_count": 0},
    )


def _write_fixture(tmp_path: Path) -> dict[str, Any]:
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema_version": "kitchen_random_task_selection.v1",
                "selected_task_id": "microwave_door",
            }
        ),
        encoding="utf-8",
    )
    selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    linked = {"source_selection_sha256": selection_sha}
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps(linked), encoding="utf-8")
    route_path = tmp_path / "route.json"
    route_path.write_text(json.dumps(linked), encoding="utf-8")
    contract_path = tmp_path / "task_success_contract.json"
    contract_path.write_text(
        json.dumps({"task_id": "microwave_door", **linked}), encoding="utf-8"
    )
    kitchen_path = tmp_path / "kitchen_inventory.json"
    kitchen_path.write_text(json.dumps({"members": []}), encoding="utf-8")
    bundle_path = tmp_path / "payload_bundle.zip"
    bundle_manifest = {
        "schema_version": "groot_oscar_closed_loop_payload_bundle.v1",
        "source_tree_identity": {
            "source_commit": SOURCE_COMMIT,
            "source_dirty_patch_sha256": DIRTY_PATCH,
        },
        "compatibility": build_bundle_compatibility(),
    }
    with zipfile.ZipFile(bundle_path, "w") as zf:
        zf.writestr("bundle_manifest.json", json.dumps(bundle_manifest, indent=2))
    evidence_path = tmp_path / "worker_image_runtime_evidence.json"
    evidence = _worker_evidence()
    assert evidence["status"] == "passed", evidence["blockers"]
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    manifest = build_attempt_input_manifest(
        run_id="run-1",
        attempt_id="attempt-1",
        launch_nonce="nonce-1",
        provider="digitalocean",
        artifacts={
            "selection": selection_path,
            "scenario": scenario_path,
            "route": route_path,
            "task_success_contract": contract_path,
            "kitchen_inventory": kitchen_path,
            "bundle": bundle_path,
            "worker_image_runtime_evidence": evidence_path,
        },
        image_digest=f"sha256:{IMAGE_HASH}",
        source_commit=SOURCE_COMMIT,
        source_dirty_patch_sha256=DIRTY_PATCH,
    )
    manifest_path = tmp_path / "attempt_input_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "manifest_path": manifest_path,
        "bundle_path": bundle_path,
        "launch_image_ref": (
            f"docker.io/nijelhunt/blueprint-groot-oscar-eval@sha256:{IMAGE_HASH}"
        ),
        "registry_evidence": {
            "schema_version": REGISTRY_EVIDENCE_SCHEMA_VERSION,
            "image_ref": (
                f"docker.io/nijelhunt/blueprint-groot-oscar-eval@sha256:{IMAGE_HASH}"
            ),
            "digest": f"sha256:{IMAGE_HASH}",
            "source": "registry_api",
        },
        "expected_source_identity": {
            "source_commit": SOURCE_COMMIT,
            "source_dirty_patch_sha256": DIRTY_PATCH,
        },
    }


def _gate(fixture: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "attempt_input_manifest_file": fixture["manifest_path"],
        "launch_image_ref": fixture["launch_image_ref"],
        "expected_source_identity": fixture["expected_source_identity"],
    }
    arguments.update(overrides)
    supplied = overrides.get("registry_image_evidence", fixture["registry_evidence"])
    arguments.pop("registry_image_evidence", None)
    arguments["registry_evidence_resolver"] = lambda _image_ref: supplied or {}
    return enforce_pre_allocation_identity_gate(**arguments)


def test_complete_identity_fixture_passes_pre_spend(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    result = _gate(fixture)
    assert result["blockers"] == []
    assert result["status"] == "PASS"
    assert result["identity"]["image_digest"] == IMAGE_HASH


def test_tag_only_launch_ref_blocks(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    result = _gate(
        fixture,
        launch_image_ref="docker.io/nijelhunt/blueprint-groot-oscar-eval:latest",
    )
    assert result["status"] == "BLOCKED"
    assert any("launch_image_ref_not_digest_pinned" in item for item in result["blockers"])


def test_stale_registry_digest_blocks(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    stale = dict(fixture["registry_evidence"])
    stale["digest"] = "sha256:" + "0" * 64
    result = _gate(fixture, registry_image_evidence=stale)
    assert result["status"] == "BLOCKED"
    assert any("registry_digest_mismatch" in item for item in result["blockers"])


def test_missing_registry_evidence_blocks(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    result = _gate(fixture, registry_image_evidence=None)
    assert result["status"] == "BLOCKED"
    assert any(
        "registry_image_evidence_missing" in item for item in result["blockers"]
    )


def test_revalidation_detects_post_gate_artifact_mutation(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    assert _gate(fixture)["status"] == "PASS"
    fixture["bundle_path"].write_bytes(b"mutated-after-capacity-probe")
    assert revalidate_attempt_artifact_bytes(fixture["manifest_path"]) == [
        "attempt_artifact_sha256_mismatch:bundle"
    ]


def test_old_commit_and_wrong_patch_hash_block(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    wrong_commit = _gate(
        fixture,
        expected_source_identity={
            "source_commit": "f" * 40,
            "source_dirty_patch_sha256": DIRTY_PATCH,
        },
    )
    assert wrong_commit["status"] == "BLOCKED"
    assert any(
        "source_commit_mismatch" in item for item in wrong_commit["blockers"]
    )
    wrong_patch = _gate(
        fixture,
        expected_source_identity={
            "source_commit": SOURCE_COMMIT,
            "source_dirty_patch_sha256": "0" * 64,
        },
    )
    assert wrong_patch["status"] == "BLOCKED"
    assert any(
        "source_dirty_patch_mismatch" in item for item in wrong_patch["blockers"]
    )


def test_swapped_bundle_bytes_block(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    with zipfile.ZipFile(fixture["bundle_path"], "a") as zf:
        zf.writestr("extra.txt", "swapped")
    result = _gate(fixture)
    assert result["status"] == "BLOCKED"
    assert any(
        "attempt_artifact_sha256_mismatch:bundle" in item
        for item in result["blockers"]
    )


def test_incompatible_bundle_schema_blocks(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    compatibility = build_bundle_compatibility()
    del compatibility["required_schemas"]["review_media"]
    manifest = {
        "schema_version": "groot_oscar_closed_loop_payload_bundle.v1",
        "source_tree_identity": {
            "source_commit": SOURCE_COMMIT,
            "source_dirty_patch_sha256": DIRTY_PATCH,
        },
        "compatibility": compatibility,
    }
    fixture["bundle_path"].unlink()
    with zipfile.ZipFile(fixture["bundle_path"], "w") as zf:
        zf.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))
    payload = json.loads(fixture["manifest_path"].read_text())
    payload["artifacts"]["bundle"]["sha256"] = hashlib.sha256(
        fixture["bundle_path"].read_bytes()
    ).hexdigest()
    fixture["manifest_path"].write_text(json.dumps(payload), encoding="utf-8")
    result = _gate(fixture)
    assert result["status"] == "BLOCKED"
    assert any(
        "g1_bundle_schema_incompatible:review_media" in item
        for item in result["blockers"]
    )


def test_sealed_healthcheck_payload_passes_real_evidence_validator() -> None:
    metadata = _sealed_runtime_metadata()
    assert metadata["configured_g1_usd_exists"] is True
    assert metadata["configured_g1_asset_binding_valid"] is True
    evidence = _worker_evidence()
    validation = validate_worker_image_runtime_evidence(
        evidence,
        expected_image_digest=f"sha256:{IMAGE_HASH}",
        expected_source_commit=SOURCE_COMMIT,
        expected_dirty_patch_sha256=DIRTY_PATCH,
    )
    assert validation["blockers"] == []
    assert validation["status"] == "passed"


def test_sealed_healthcheck_claims_stay_distinct() -> None:
    module = _load_sealed_healthcheck_module()
    metadata = module.build_runtime_metadata(
        env={
            "BLUEPRINT_WORKER_IMAGE_FAMILY": "isaac-eval-worker",
            "BLUEPRINT_SIMULATOR_FRAMEWORK": "isaac_sim",
            "BLUEPRINT_ISAAC_SIM_MAJOR_VERSION": "6",
        },
        isaac_imported=True,
        g1_exists=True,
        asset_binding_valid=False,
        official_assets_exist=True,
        source_commit=SOURCE_COMMIT,
        dirty_patch=DIRTY_PATCH,
    )
    assert metadata["configured_g1_usd_exists"] is True
    assert metadata["configured_g1_asset_binding_valid"] is False
