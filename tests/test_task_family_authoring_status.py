# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_family_authoring_status.py
"""Bounded callback projection from real compiler receipts using local fixtures."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import socket
import subprocess

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.task_family_authoring_status import project_family_authoring_status
from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation
from tests.test_task_family_button_authoring import _fixture, _reseal_facts


def _prepared(tmp_path, monkeypatch, *, sequence=False):
    args, _row = _fixture(tmp_path, monkeypatch, sequence=sequence)
    facts = args["task_context"]["operator_answers"]["family_task_facts"]
    facts["operations"][0].pop("minimum_hold_seconds")
    _reseal_facts(args)
    if sequence:
        from tests.test_task_family_button_authoring import _rebind_semantics
        _rebind_semantics(args)
    return args, compile_website_scene_preparation(**args)


def _project(args, prepared):
    return project_family_authoring_status(preparation=prepared, task_context=args["task_context"],
                                           pipeline_source_commit="a" * 40)


def _reseal(prepared):
    request = prepared["task_evaluation_inputs"]["authoring"]["request"]
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    prepared["digest"] = canonical_digest(prepared, digest_field="digest")
    return request


@pytest.mark.parametrize("sequence", [False, True])
def test_actual_authoring_receipt_projects_exact_source_and_one_task_fact(tmp_path, monkeypatch, sequence):
    args, prepared = _prepared(tmp_path, monkeypatch, sequence=sequence)
    before = deepcopy(prepared)
    result = _project(args, prepared)
    request = prepared["task_evaluation_inputs"]["authoring"]["request"]
    assert result["source_video_digest"] == args["source_geometry"]["binding"]["source_video_digest"]
    assert result["task_context_digest"] == args["task_context"]["context_digest"]
    assert result["authoring_request_digest"] == request["request_digest"]
    assert result["task_description_digest"] == cross_runtime_canonical_digest({"description": args["task_context"]["description"]})
    assert result["next_action"]["quantity"] == "minimum_hold_seconds"
    assert result["next_action"] in result["requirements"]
    assert result["next_action"]["owner"] == "task_owner"
    assert [row["operation_id"] for row in result["operations"]] == [row["stage_id"] for row in request["operations"]]
    assert [row["order"] for row in result["operations"]] == list(range(1, len(result["operations"]) + 1))
    assert result["automatic_paid_execution_authorized"] is False
    assert result["projection_digest"] == cross_runtime_canonical_digest(result, digest_field="projection_digest")
    assert "/private/" not in json.dumps(result) and '"path"' not in json.dumps(result)
    assert prepared == before


@pytest.mark.parametrize("fault,reason", [
    ("source_binding", "source_binding_changed"), ("video_crc", "source_video_required"),
    ("operation_order", "operation_order_invalid"), ("duplicate_operation", "operation_id_duplicate"),
    ("duplicate_target", "target_id_duplicate"), ("question_limit", "question_invalid"),
    ("wrong_owner", "requirement_invalid"), ("unrelated_next_action", "next_action_changed"),
    ("authority", "receipt_changed"),
])
def test_incoherent_or_overclaiming_resealed_receipt_is_not_reported(tmp_path, monkeypatch, fault, reason):
    args, prepared = _prepared(tmp_path, monkeypatch)
    request = prepared["task_evaluation_inputs"]["authoring"]["request"]
    if fault == "source_binding":
        request["binding"]["source_geometry_digest"] = "sha256:" + "b" * 64
    elif fault == "video_crc":
        request["binding"]["source_video_digest"] = "crc32:12345678"
    elif fault == "operation_order":
        request["operations"][0]["order"] = 2
    elif fault == "duplicate_operation":
        request["operations"][0]["source_operation"] = {"task_quote": args["task_context"]["description"]}
        request["operations"].append({**request["operations"][0], "order": 2})
    elif fault == "duplicate_target":
        request["source_targets"].append(deepcopy(request["source_targets"][0]))
    elif fault == "question_limit":
        request["requirements"][0]["question"] = "x" * 501
    elif fault == "wrong_owner":
        request["requirements"][0]["owner"] = "runtime_operator"
    elif fault == "unrelated_next_action":
        request["next_action"] = {"quantity": "unrelated"}
    else:
        request["automatic_paid_execution_authorized"] = True
    _reseal(prepared)
    with pytest.raises(ValueError, match=reason):
        _project(args, prepared)


def test_projection_excludes_unlisted_paths_and_never_reads_or_sends(tmp_path, monkeypatch):
    args, prepared = _prepared(tmp_path, monkeypatch)
    request = prepared["task_evaluation_inputs"]["authoring"]["request"]
    request["private_path"] = "/private/retained-source"
    request["requirements"][0]["reference"] = {"path": "/private/retained-source"}
    request["next_action"] = request["requirements"][0]
    _reseal(prepared)
    def refused(*args, **kwargs):
        pytest.fail("pure authoring projection must not read files, connect or execute")
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", refused)
        patch.setattr(socket.socket, "connect", refused)
        patch.setattr(subprocess, "Popen", refused)
        result = _project(args, prepared)
    assert "retained-source" not in json.dumps(result)


def test_projection_byte_limit_includes_utf8_expansion(tmp_path, monkeypatch):
    args, prepared = _prepared(tmp_path, monkeypatch)
    request = prepared["task_evaluation_inputs"]["authoring"]["request"]
    row = request["requirements"][0]
    request["requirements"] = [{**row, "requirement_id": f"request-{index}", "question": "\U0001f7e5" * 500}
                               for index in range(64)]
    request["next_action"] = request["requirements"][0]
    _reseal(prepared)
    with pytest.raises(ValueError, match="payload_limit"):
        _project(args, prepared)


def test_projection_requires_an_explicit_trusted_pipeline_commit(tmp_path, monkeypatch):
    args, prepared = _prepared(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="pipeline_commit_required"):
        project_family_authoring_status(preparation=prepared, task_context=args["task_context"], pipeline_source_commit=None)


def test_missing_multistep_source_quotes_cannot_be_filled_with_the_whole_job(tmp_path, monkeypatch):
    args, prepared = _prepared(tmp_path, monkeypatch, sequence=True)
    request = prepared["task_evaluation_inputs"]["authoring"]["request"]
    request["operations"][1].pop("source_operation")
    _reseal(prepared)
    with pytest.raises(ValueError, match="operation_quote_required"):
        _project(args, prepared)
