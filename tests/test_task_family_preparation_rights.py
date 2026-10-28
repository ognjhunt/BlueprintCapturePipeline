# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_family_preparation.py
"""Denied capture rights stop source reads while retaining the accepted job."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import subprocess

import pytest

from blueprint_pipeline import task_family_preparation as family
from blueprint_pipeline.core.handoff_lease import HandoffLeaseError, bind_handoff_lease
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_task_context_contracts import website_task_context_digest
from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation
from tests.test_task_family_layout_authoring import _native_records


def _refused(*args, **kwargs):
    pytest.fail("Denied rights must not read source bytes or author native assets")


def _deny_source_work(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _refused)
    monkeypatch.setattr(subprocess, "Popen", _refused)
    monkeypatch.setattr(family, "_sha256_file", _refused)
    monkeypatch.setattr("blueprint_pipeline.task_family_layout_authoring.emit_registered_task_layout", _refused)
    monkeypatch.setattr("blueprint_pipeline.task_family_button_authoring.author_button_family_inputs", _refused)
    monkeypatch.setattr("blueprint_pipeline.native_task_runtime_contract.materialize_native_task_runtime_contract", _refused)
    monkeypatch.setattr(Path, "read_bytes", _refused)
    monkeypatch.setattr(Path, "read_text", _refused)


@pytest.mark.parametrize("entry", [compile_website_scene_preparation, family.prepare_task_family])
@pytest.mark.parametrize("denial", [
    {"derived_scene_generation_allowed": False}, {"consent_revoked": True},
    {"consent_status": "revoked"}, {"future_processing_allowed": False},
    {"consent_revoked": 1}, {"consent_revoked": "false"}, {"consent_revoked": None},
])
def test_rights_denial_precedes_missing_source_files_and_native_authoring(tmp_path, monkeypatch, entry, denial):
    args, references = _native_records(tmp_path, monkeypatch)
    args["task_context"]["capture_rights"].update(denial)
    args["task_context"]["context_digest"] = website_task_context_digest(args["task_context"])
    # No geometry/asset input file can make this path accidentally pass. The
    # receipt may retain supplied digests but cannot claim those bytes validated.
    for reference in references.values():
        Path(reference["path"]).unlink()
    for key in ("splat_path", "collision_mesh_path"):
        Path(args["base_scene"][key]).unlink()
    args["base_scene"] = {}
    with monkeypatch.context() as patch:
        _deny_source_work(patch)
        result = entry(**args)
    assert result["schema_version"] == "website_scene_preparation.v1"
    assert result["status"] == "needs_input" and result["job_disposition"] == "accepted"
    assert result["blockers"] == ["website_scene_processing_rights_required"]
    assert result["digest"] == canonical_digest(result, digest_field="digest")
    assert "native_task_preparation_gap" not in result
    assert result["intake_request"] is None and result["provider_mutation_performed"] is False
    inputs = result["task_evaluation_inputs"]
    assert inputs["task_family"] == "button_press"
    assert inputs["scope"]["task_context_digest"] == args["task_context"]["context_digest"]
    assert inputs["requirements"]["button_press"]["native_observables"]
    assert inputs["task_success_contract"] is None and inputs["native_runtime_contract"] is None
    assert inputs["automatic_paid_execution_authorized"] is False
    assert list(args["output_root"].iterdir()) == [args["output_root"] / "preparation.json"]
    assert json.loads((args["output_root"] / "preparation.json").read_text()) == result


@pytest.mark.parametrize("entry", [compile_website_scene_preparation, family.prepare_task_family])
def test_confirmed_legacy_cpu_preparation_keeps_explicit_derived_rights_without_paid_grant(tmp_path, monkeypatch, entry):
    args, _ = _native_records(tmp_path, monkeypatch)
    args["task_context"]["capture_rights"] = {"derived_scene_generation_allowed": True}
    args["task_context"]["context_digest"] = website_task_context_digest(args["task_context"])
    args["spend"] = {}
    monkeypatch.setattr(socket.socket, "connect", _refused)
    monkeypatch.setattr(subprocess, "Popen", _refused)
    result = entry(**args)
    assert result["status"] == "task_evaluation_ready"
    assert "website_scene_processing_rights_required" not in result["blockers"]
    assert result["job_disposition"] == "accepted"
    assert result["intake_request"] is None


@pytest.mark.parametrize("entry", [compile_website_scene_preparation, family.prepare_task_family])
def test_unconfirmed_family_keeps_full_preparation_authority_validation_before_source(tmp_path, monkeypatch, entry):
    args, _ = _native_records(tmp_path, monkeypatch)
    args["task_context"].update(confirmed=False, confirmed_at=None, purpose="scene_preparation")
    args["task_context"]["context_digest"] = website_task_context_digest(args["task_context"])
    args["spend"] = {}
    with monkeypatch.context() as patch:
        _deny_source_work(patch)
        with pytest.raises(ValueError, match="website_preparation_authority_invalid"):
            entry(**args)
    assert not (args["output_root"] / "preparation.json").exists()


def test_rights_receipt_cannot_promote_after_lease_loss(tmp_path, monkeypatch):
    args, _ = _native_records(tmp_path, monkeypatch)
    args["task_context"]["capture_rights"]["consent_revoked"] = True
    args["task_context"]["context_digest"] = website_task_context_digest(args["task_context"])
    seen = []
    def lost_on_write(boundary):
        seen.append(boundary)
        if boundary == "shared_artifact_promotion":
            raise HandoffLeaseError("lost")
    with monkeypatch.context() as patch, bind_handoff_lease(lost_on_write):
        _deny_source_work(patch)
        with pytest.raises(HandoffLeaseError, match="lease_lost"):
            family.prepare_task_family(**args)
    assert seen == ["website_task_family_preparation_start", "shared_artifact_promotion"]
    assert not (args["output_root"] / "preparation.json").exists()
