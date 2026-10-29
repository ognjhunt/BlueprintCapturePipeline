# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_task_context_contracts.py
#   src/blueprint_pipeline/website_task_preparation.py
#   src/blueprint_pipeline/task_family_authoring.py
#   src/blueprint_pipeline/website_task_family_packet.py
#   src/blueprint_pipeline/website_task_semantics.py
#   src/blueprint_pipeline/website_native_background.py
#   src/blueprint_pipeline/website_operation_coverage.py
#   src/blueprint_pipeline/task_evaluation_scene_configuration_submission_inputs.py
"""Explicit cross-runtime contexts keep source identity and legacy bytes exact."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import (
    canonical_digest, cross_runtime_canonical_digest, cross_runtime_canonical_json,
)
from blueprint_pipeline.task_evaluation_scene_configuration_submission_inputs import checked_file, read
from blueprint_pipeline.task_object_authoring_contracts import file_record
from blueprint_pipeline.website_task_context_contracts import (
    validate_website_task_context, website_task_context_digest,
)
from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation
from tests.test_task_family_button_authoring import _fixture


def _flagged_arguments(tmp_path, monkeypatch, *, sequence=False):
    args, _row = _fixture(tmp_path, monkeypatch, sequence=sequence, general=sequence)
    context = args["task_context"]
    context["canonicalization"] = "rfc8785"
    context["confirmed_at"] = "2026-10-28T00:00:00Z"
    context["success_criteria"] = {"successDefinition": "Press the button: é e\u0301 \U0001f7e5",
        "successRate": 1e-7, "cycleTimeSeconds": 1.0, "unknown": False}
    facts = context["operator_answers"]["family_task_facts"]
    facts["canonicalization"] = "rfc8785"
    facts["task_description_digest"] = cross_runtime_canonical_digest({"description": context["description"]})
    facts["task_success_criteria_digest"] = cross_runtime_canonical_digest({"success_criteria": context["success_criteria"]})
    facts["facts_digest"] = cross_runtime_canonical_digest(facts, digest_field="facts_digest")
    context["context_digest"] = cross_runtime_canonical_digest(context, digest_field="context_digest")
    if sequence:
        from blueprint_pipeline.website_task_semantics import task_semantics_from_receipt
        receipt = deepcopy(args["task_semantics"]["planner_receipt"])
        receipt["binding"]["task_context_digest"] = context["context_digest"]
        receipt["binding_digest"] = canonical_digest(receipt["binding"])
        args["task_semantics"] = task_semantics_from_receipt(receipt=receipt, task_context=context,
            source_video_digest=args["source_geometry"]["binding"]["source_video_digest"])
    return args


@pytest.mark.parametrize("sequence", [False, True])
def test_flagged_context_reaches_actual_family_preparation_and_packet_read(tmp_path, monkeypatch, sequence):
    from blueprint_pipeline.native_task_arena_bundle import verify_native_task_arena_packet
    from tests.test_task_family_authoring_evidence import _packet

    args = _flagged_arguments(tmp_path, monkeypatch, sequence=sequence)
    context = args["task_context"]
    assert context["context_digest"] != canonical_digest(context, digest_field="context_digest")
    assert validate_website_task_context(context, request_id=context["request_id"],
        scene_id=context["scene_id"], capture_id=context["capture_id"]) == context
    prepared = compile_website_scene_preparation(**args)
    assert prepared["status"] == "intake_ready", prepared
    assert prepared["binding"]["task_context_digest"] == context["context_digest"]
    native_path = tmp_path / "authored-native.json"
    write_json(native_path, prepared["task_evaluation_inputs"]["native_runtime_contract"])
    packet = _packet(tmp_path, args, prepared, native_path)
    verify_native_task_arena_packet(packet["packet_root"])
    context_path = tmp_path / "task-context.json"
    write_json(context_path, context)
    before = file_record(context_path)
    assert read(checked_file(context_path, before), digest_field="context_digest") == context
    assert file_record(context_path) == before
    from blueprint_pipeline.website_scene_dispatch import register_website_preparation
    registration = register_website_preparation(preparation_path=args["output_root"] / "preparation.json",
        runtime_inputs_path=native_path, task_context_path=context_path, root=tmp_path / "registered-source", now=args["now"])
    retained = read(checked_file(registration["path"], registration), digest_field="registration_digest")
    assert retained["execution_authority_granted"] is False
    assert retained["references"]["task_context"] == before


def test_unflagged_context_keeps_legacy_digest_and_raw_source_bytes(tmp_path, monkeypatch):
    args, _row = _fixture(tmp_path, monkeypatch)
    context = args["task_context"]
    assert "canonicalization" not in context
    assert website_task_context_digest(context) == context["context_digest"]
    path = tmp_path / "legacy-context.json"
    write_json(path, context)
    before = path.read_bytes(), file_record(path)
    assert read(path, digest_field="context_digest") == context
    assert (path.read_bytes(), file_record(path)) == before


@pytest.mark.parametrize("mode", [None, "unknown", True, 1, [], {}])
def test_unknown_or_non_scalar_context_mode_refuses_without_legacy_fallback(tmp_path, monkeypatch, mode):
    args, _row = _fixture(tmp_path, monkeypatch)
    context = args["task_context"]
    context["canonicalization"] = mode
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    with pytest.raises(ValueError, match="context_canonicalization_invalid"):
        website_task_context_digest(context)
    path = tmp_path / "invalid-mode.json"
    write_json(path, context)
    with pytest.raises(ValueError, match="context_canonicalization_invalid"):
        read(path, digest_field="context_digest")


def test_flagged_context_cannot_change_source_capture_identity(tmp_path, monkeypatch):
    args = _flagged_arguments(tmp_path, monkeypatch)
    context = args["task_context"]
    with pytest.raises(ValueError, match="identity_mismatch"):
        validate_website_task_context(context, request_id=context["request_id"],
            scene_id=context["scene_id"], capture_id="another-capture")
    context["success_criteria"]["successRate"] = 2e-7
    with pytest.raises(ValueError, match="digest_mismatch"):
        validate_website_task_context(context, request_id=context["request_id"],
            scene_id=context["scene_id"], capture_id=context["capture_id"])


def test_generic_reader_does_not_reinterpret_other_schemas(tmp_path):
    value = {"schema_version": "another_legacy_context.v1", "canonicalization": "rfc8785", "number": 1e-7}
    value["context_digest"] = canonical_digest(value, digest_field="context_digest")
    path = tmp_path / "other-schema.json"
    write_json(path, value)
    assert read(path, digest_field="context_digest") == value


def test_actual_node_webapp_context_and_confirmation_reach_pipeline_consumers(tmp_path):
    from blueprint_pipeline.task_family_button_authoring import _facts

    # Exact Node bytes from WebApp's projectWebsiteTaskContext and
    # confirmFamilyTaskAnswer integration test, not a Python-built substitute.
    fixture_path = Path(__file__).parent / "fixtures" / "website_family_facts_node_context.v1.json"
    fixture = json.loads(fixture_path.read_text())
    assert fixture["evidence_class"] == "hermetic_fixture_no_source_or_runtime_admission"
    context = fixture["context"]
    assert context["canonicalization"] == "rfc8785"
    assert context["context_digest"] != canonical_digest(context, digest_field="context_digest")
    assert validate_website_task_context(context, request_id=context["request_id"],
        scene_id=context["scene_id"], capture_id=context["capture_id"]) == context
    facts = _facts(context, fixture["facts"]["source_video_digest"])
    assert facts["operation-001"]["minimum_hold_seconds"] == 1e-7
    assert context["success_criteria"]["cycleTimeSeconds"] == 1.0
    assert "café 🦾" in context["description"]
    context_path = tmp_path / "node-context.json"
    write_json(context_path, context)
    before = file_record(context_path)
    assert read(checked_file(context_path, before), digest_field="context_digest") == context
    assert file_record(context_path) == before


def test_actual_node_vector_preserves_utf16_key_order_and_unicode_normal_forms():
    fixture_path = Path(__file__).parent / "fixtures" / "website_family_facts_node_context.v1.json"
    vector = json.loads(fixture_path.read_text())["canonicalizer_vector"]
    assert cross_runtime_canonical_json(vector["value"]) == vector["canonical_json"]
    assert cross_runtime_canonical_digest(vector["value"]) == vector["digest"]
    assert canonical_digest(vector["value"]) != vector["digest"]
    assert vector["value"]["é"] == "composed" and vector["value"]["e\u0301"] == "decomposed"
