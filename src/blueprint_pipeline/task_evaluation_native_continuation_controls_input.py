"""Reuse completed native construction outputs as diagnostic robot inputs.

This consumes retained bytes and closed allocator receipts. It neither reruns
construction nor inherits the old run's deadline, spending grant or publication
authority. The resulting inputs remain development-only.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_provider_stage_chain import _completed_stage_chain_valid


CLAIM_CEILING = "development_only_downstream_construction_and_controls_diagnostic"


class NativeContinuationControlsInputError(ValueError):
    """Retained construction bytes do not support a downstream diagnostic."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise NativeContinuationControlsInputError("native_continuation_controls_" + reason)


def _identity(path: Path) -> tuple[str, int]:
    _require(path.is_file() and not path.is_symlink(), "source_file_invalid")
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest(), path.stat().st_size


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    _require(isinstance(value, Mapping), "source_document_invalid")
    return dict(value)


def materialize_native_continuation_controls_input(
    *, continuation_result_path: str | Path, allocator_result_path: str | Path,
    continuation_bundle_path: str | Path, configured_assets_root: str | Path,
    static_qualification_path: str | Path, native_import_qualification_path: str | Path,
    output_dir: str | Path,
    materialized_reference_overrides: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Verify the completed source chain and bind reusable downstream inputs."""
    result_path = Path(continuation_result_path).expanduser()
    allocator_path = Path(allocator_result_path).expanduser()
    bundle_path = Path(continuation_bundle_path).expanduser()
    native = _read(result_path)
    allocator = _read(allocator_path)
    _require(
        native.get("schema_version") == "scene_configuration_native_continuation_result.v1"
        and native.get("status") == "completed"
        and native.get("claim_ceiling") == "development_only"
        and native.get("qualification_eligible") is False
        and native.get("candidate_policy_queried") is False
        and native.get("provider_zero_required_after_return") is True
        and native.get("result_digest") == canonical_digest(native, digest_field="result_digest")
        and allocator.get("status") == "completed"
        and allocator.get("native_control_result_digest") == native["result_digest"]
        and allocator.get("continuing_spend_from_this_run") is False
        and allocator.get("all_staged_objects_absent") is True
        and allocator.get("provider_closeout", {}).get("provider_zero_confirmed") is True
        and allocator.get("blockers") == [], "completed_closed_source_required",
    )
    bundle_digest, _ = _identity(bundle_path)
    _require(bundle_digest == allocator.get("bundle_sha256"), "source_bundle_changed")
    with zipfile.ZipFile(bundle_path) as archive:
        manifest_name = "provider_runtime/native_continuation_manifest.v1.json"
        envelope_name = "provider_runtime/construction_envelope.v1.json"
        _require(all(archive.namelist().count(name) == 1 for name in (manifest_name, envelope_name)), "bundle_members_invalid")
        manifest = json.loads(archive.read(manifest_name))
        envelope = json.loads(archive.read(envelope_name))
    _require(
        manifest.get("schema_version") == "scene_configuration_native_continuation.v1"
        and manifest.get("diagnostic_kind") == "scene_configuration_native_continuation"
        and manifest.get("input_digest") == canonical_digest(manifest, digest_field="input_digest")
        and manifest.get("qualification_eligible") is False
        and manifest.get("configured_revision_publication_permitted") is False
        and manifest.get("implementation_commit") == native.get("implementation_commit")
        and envelope.get("envelope_digest") == canonical_digest(envelope, digest_field="envelope_digest")
        and envelope["envelope_digest"] == manifest.get("construction_envelope_digest"),
        "source_binding_invalid",
    )
    chain = native.get("stage_chain")
    _require(_completed_stage_chain_valid(chain, provider_result={"run_id": envelope["run_id"]}, diagnostic_only=False), "stage_chain_invalid")
    _require([row["stage_id"] for row in chain["stage_results"]]
             == [row["stage_id"] for row in envelope["recipe"]["stage_sequence"]], "stage_sequence_changed")
    for stage in chain["stage_results"]:
        configuration = manifest["configurations"].get(stage["stage_id"])
        _require(isinstance(configuration, Mapping) and configuration.get("sha256") == stage.get("configuration_digest"), "stage_configuration_changed")

    rows: list[dict[str, Any]] = []

    def bind(contract_path: str, path: Path, reference: Mapping[str, Any]) -> None:
        digest, size = _identity(path)
        _require((digest, size) == (reference.get("digest"), reference.get("size_bytes")), "bound_input_changed:" + contract_path)
        rows.append({"contract_path": contract_path, "path": str(path.resolve()),
                     "uri": reference.get("uri") or "retained-native-output://" + envelope["run_id"] + "/" + contract_path,
                     "digest": digest, "size_bytes": size, "full_byte_readback_passed": True})

    source_references = {row["contract_path"]: row for row in envelope["materialized_references"]}
    required_sources = (
        "task.definition", "task.success_criteria", "task.execution", "task.subject.source_object",
        "scene.registration.support_plane", "scene.registration.workspace_clearance",
    )
    for contract_path in required_sources:
        reference = source_references.get(contract_path)
        _require(isinstance(reference, Mapping) and reference.get("full_byte_service_account_readback_passed") is True, "source_reference_missing:" + contract_path)
        path = Path((materialized_reference_overrides or {}).get(contract_path) or reference["materialized_path"]).expanduser()
        bind(contract_path, path, reference)

    def artifact(role: str) -> dict[str, Any]:
        matches = [row for stage in chain["stage_results"] for row in stage["output_artifacts"] if row.get("role") == role]
        _require(len(matches) == 1, "stage_artifact_missing:" + role)
        return matches[0]

    bind("diagnostic_output.static_qualification_receipt.v1.json", Path(static_qualification_path), artifact("static_qualification_receipt"))
    bind("diagnostic_output.native_import_qualification_receipt.v1.json", Path(native_import_qualification_path), artifact("native_import_qualification_receipt"))
    assets_root = Path(configured_assets_root).expanduser()
    manifest_path = assets_root / "configured_scene_bundle_candidate.v1.json"
    manifest_reference = artifact("configured_scene_bundle_candidate_manifest")
    bind("diagnostic_output.configured_scene_bundle_candidate.v1.json", manifest_path, manifest_reference)
    scene_manifest = _read(manifest_path)
    _require(scene_manifest.get("manifest_digest") == canonical_digest(scene_manifest, digest_field="manifest_digest")
             and scene_manifest.get("robot_neutral") is True
             and scene_manifest.get("robot_specific_base_registration_included") is False,
             "scene_manifest_invalid")
    assets = scene_manifest.get("assets") or []
    _require(len(assets) == 3 and {row.get("role") for row in assets} == {"appearance", "collision", "replacement"}, "scene_assets_invalid")
    contracts = {"appearance": "diagnostic_output.appearance.usdz", "collision": "diagnostic_output.collision.usda", "replacement": "diagnostic_output.replacement.usdz"}
    for row in assets:
        relative = Path(row["relative_path"])
        _require(not relative.is_absolute() and len(relative.parts) == 1 and relative.name not in {".", ".."}, "scene_asset_path_invalid")
        if row["role"] == "replacement":
            _require((row["digest"], row["size_bytes"]) == (
                artifact("native_qualified_replacement_asset")["digest"],
                artifact("native_qualified_replacement_asset")["size_bytes"],
            ), "native_qualified_asset_mismatch")
        bind(contracts[row["role"]], assets_root / relative, row)
    native_digest, native_size = _identity(result_path)
    bind("diagnostic_output.scene_configuration_native_continuation_result.v1.json", result_path, {"digest": native_digest, "size_bytes": native_size})
    allocator_digest, allocator_size = _identity(allocator_path)
    value = {
        "schema_version": "task_evaluation_configured_scene_diagnostic_controls_input.v1",
        "status": "materialized", "source_configuration_run_id": envelope["run_id"],
        "source_provider_output_digest": native["result_digest"],
        "source_kind": "completed_native_continuation",
        "source_bundle_digest": bundle_digest,
        "source_allocator_reference": {"digest": allocator_digest, "size_bytes": allocator_size},
        "construction_envelope_digest": envelope["envelope_digest"],
        "materialized_inputs": rows, "claim_ceiling": CLAIM_CEILING,
        "qualification_eligible": False, "configured_revision_publication_permitted": False,
        "evaluation_ready_promotion_permitted": False, "prior_spend_authority_inherited": False,
        "prior_deadline_extended": False, "provider_mutation_performed": False,
        "paid_execution_requested": False, "receipt_digest": "",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    root = Path(output_dir).expanduser()
    _require(not root.exists(), "output_exists")
    root.mkdir(parents=True)
    target = root / "diagnostic_controls_input.v1.json"
    target.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return value
