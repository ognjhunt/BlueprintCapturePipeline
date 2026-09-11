"""Same-root checkpoints for resuming an interrupted Astra construction stage."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_object_astra_authoring import AssetAuthoringError


def _hash(path):
    if path.is_symlink() or not path.is_file():
        raise AssetAuthoringError("astra_stage_resume_artifact_invalid")
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def _seal(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise AssetAuthoringError("astra_stage_resume_immutable_binding_changed")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".resume-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def bind_same_root_resume(root, envelope, configurations, parent_deadline_epoch):
    value = {"schema_version": "astra_same_run_stage_resume_binding.v1", "output_root": str(root),
        "run_id": envelope["run_id"], "envelope_digest": canonical_digest(envelope),
        "configuration_digests": {name: _hash(path) for name, (_, path) in configurations.items()},
        "parent_deadline_epoch": parent_deadline_epoch, "new_paid_allocation_authorized": False}
    value["binding_digest"] = canonical_digest(value, digest_field="binding_digest")
    path = root / "astra_same_run_resume_binding.json"
    if not path.exists() and any((root / name).exists() for name in configurations):
        raise AssetAuthoringError("astra_stage_resume_legacy_prefix_not_checkpointed")
    _seal(path, value)
    return value


def _artifacts(root, records):
    if not isinstance(records, list):
        raise AssetAuthoringError("astra_stage_resume_artifact_inventory_invalid")
    for record in records:
        path = Path(record["path"])
        if (not path.resolve().is_relative_to(root.resolve()) or _hash(path) != record.get("digest")
                or path.stat().st_size != record.get("size_bytes")):
            raise AssetAuthoringError("astra_stage_resume_artifact_changed")


def _checkpoint(stage, binding, previous, result):
    value = {"schema_version": "astra_same_run_stage_checkpoint.v1", "stage_id": stage["stage_id"],
        "binding_digest": binding["binding_digest"], "dependency_result_digests": [row["stage_result_digest"] for row in previous],
        "stage_result": result}
    value["checkpoint_digest"] = canonical_digest(value, digest_field="checkpoint_digest")
    return value


def load_completed_stage(stage_root, stage, binding, previous):
    path = stage_root / "completed_stage_checkpoint.json"
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    result = saved.get("stage_result") or {}
    if (saved != _checkpoint(stage, binding, previous, result)
            or result.get("schema_version") != "task_evaluation_scene_configuration_stage_result.v1"
            or result.get("status") != "completed" or result.get("stage_id") != stage["stage_id"]
            or result.get("configuration_digest") != binding["configuration_digests"][stage["stage_id"]]
            or result.get("stage_result_digest") != canonical_digest(result, digest_field="stage_result_digest")
            or result.get("provider_mutations_performed") != 0 or result.get("paid_execution_requested") is not False
            or result.get("executed_inside_parent_configuration_run") is not True
            or result.get("canonical_allocator") is not None or result.get("retry_cap") != 0
            or result.get("executed_inside_one_parent_provider_run") is False
            or result.get("diagnostic_only") is True or result.get("qualification_eligible") is False):
        raise AssetAuthoringError("astra_stage_resume_completed_checkpoint_invalid")
    _artifacts(stage_root, result.get("output_artifacts"))
    return result


def save_completed_stage(stage_root, stage, binding, previous, result):
    _artifacts(stage_root, result.get("output_artifacts"))
    _seal(stage_root / "completed_stage_checkpoint.json", _checkpoint(stage, binding, previous, result))


def retained_astra_production(producer_root, stage, envelope, configuration_path):
    path = producer_root / "task_evaluation_scene_configuration_stage_production.v1.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    original = json.loads((producer_root / "stage_production_input.v1.json").read_text())
    if (original.get("construction_envelope") != envelope or original.get("configuration_sha256") != _hash(configuration_path)
            or original.get("stage") != stage or result.get("source_commit") != original.get("source_commit")
            or result.get("toolchain_digest") != original.get("toolchain_digest")
            or result.get("schema_version") != "task_evaluation_scene_configuration_stage_production.v1"
            or result.get("status") != "completed" or result.get("stage_id") != stage["stage_id"]
            or result.get("adapter_id") != "content_agents_rigid_replacement"
            or result.get("capability") != stage["capability"]
            or result.get("production_result_digest") != canonical_digest(result, digest_field="production_result_digest")
            or result.get("provider_mutations_performed") != 0 or result.get("paid_execution_requested") is not False
            or result.get("executed_inside_parent_configuration_run") is not True
            or {row.get("role") for row in result.get("artifacts", [])} != {
                "replacement_asset", "replacement_authoring_receipt", "replacement_graph_spec"}):
        raise AssetAuthoringError("astra_stage_resume_production_receipt_invalid")
    _artifacts(producer_root, result.get("artifacts"))
    return tuple(result["artifacts"])
