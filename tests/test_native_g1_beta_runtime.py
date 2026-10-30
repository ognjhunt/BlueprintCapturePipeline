"""Partial native evidence survives failures without fabricating a score."""
from copy import deepcopy
import json
import time

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import native_g1_beta_runtime as runtime
from blueprint_pipeline.beta_policy_screening_g1 import _episode_outcome
from tests.test_beta_policy_screening_g1 import session_result, SHA, DP


def _seal(value, field):
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _stage_query(output, cell, query):
    path = output / "cells" / cell["cell_id"] / "episode/query_checkpoints/query_0000.v1.json"
    runtime._write(path, query)
    return path


def test_worker_failure_retains_actual_query_bytes_and_load_proof(tmp_path):
    plan, candidate, source_result = session_result()
    source = source_result["episodes"][0]
    cell = candidate["scenario_cells"][0]
    query_path = _stage_query(tmp_path, cell, source["query_checkpoints"][0])
    worker = _seal({"schema_version": "native_g1_development_worker_result.v1", "candidate_id": DP,
        "request_digest": SHA, "scene_plan_digest": SHA, "status": "blocked", "phase_reached": "episode",
        "blocker": {"message": "g1_beta_checkpoint_load_unverified"}}, "result_digest")
    cell_root = query_path.parents[2]
    runtime._write(cell_root / runtime.WORKER_RESULT, worker)
    result = runtime.collect_cell(root=tmp_path, cell_root=cell_root, cell=cell, scene_plan_digest=SHA,
        candidate_id=DP, request_digest=SHA, returncode=None, reason=None)
    assert result["status"] == "blocked"
    assert result["reason"] == "g1_beta_checkpoint_load_unverified"
    assert result["query_checkpoints"] == source["query_checkpoints"]
    assert len(result["artifact_records"]) == 2
    outcome, _reason, queried = _episode_outcome(result, candidate_id=DP,
        contract_digest=candidate["execution_spec"]["task_success_contract_digest"])
    assert outcome == "void" and queried


def test_timeout_without_worker_final_keeps_response_and_explicit_reason(tmp_path):
    _plan, candidate, source_result = session_result()
    cell = candidate["scenario_cells"][0]
    path = _stage_query(tmp_path, cell, source_result["episodes"][0]["query_checkpoints"][0])
    result = runtime.collect_cell(root=tmp_path, cell_root=path.parents[2], cell=cell,
        scene_plan_digest=SHA, candidate_id=DP, request_digest=SHA, returncode=None,
        reason="g1_beta_native_worker_deadline")
    assert result["reason"] == "g1_beta_native_worker_deadline"
    assert result["status"] == "blocked" and "worker_result" not in result
    assert len(result["query_checkpoints"]) == 1


def test_query_file_closure_and_scene_binding_are_not_self_asserted(tmp_path):
    _plan, candidate, source_result = session_result()
    cell = candidate["scenario_cells"][0]
    query = deepcopy(source_result["episodes"][0]["query_checkpoints"][0])
    query["scene_plan_digest"] = "sha256:" + "f" * 64
    _seal(query, "checkpoint_digest")
    path = _stage_query(tmp_path, cell, query)
    with pytest.raises(ValueError, match="checkpoint_binding_invalid"):
        runtime.collect_cell(root=tmp_path, cell_root=path.parents[2], cell=cell, scene_plan_digest=SHA,
            candidate_id=DP, request_digest=SHA, returncode=None, reason=None)
    path.write_text(json.dumps(source_result["episodes"][0]["query_checkpoints"][0]))
    (path.parent / "unexpected.json").write_text("{}")
    with pytest.raises(ValueError, match="checkpoint_closure_invalid"):
        runtime.collect_cell(root=tmp_path, cell_root=path.parents[2], cell=cell, scene_plan_digest=SHA,
            candidate_id=DP, request_digest=SHA, returncode=None, reason=None)


def test_unstarted_cells_are_individual_void_rows_and_no_query_claim():
    plan, candidate, _source_result = session_result()
    authority = {"candidate_ids": [DP], "cells": candidate["scenario_cells"],
        "authority_digest": SHA, "runtime_inputs_digest": SHA,
        "task_success_contract_digest": candidate["execution_spec"]["task_success_contract_digest"],
        "beta_screening_admission": {"plan_digest": plan["plan_digest"], "configuration_digest": candidate["configuration_digest"],
            "admission_digest": candidate["admission_digest"], "runtime_identity_digest": candidate["execution_spec"]["runtime_digest"],
            "source": plan["source"], "scenario_plan_digest": plan["scenario_plan_digest"]}}
    rows = [runtime._empty_cell(cell) for cell in authority["cells"]]
    result = runtime._result(authority, rows, stage="policy_dependencies", blockers=["g1_beta_policy_dependencies_failed"])
    assert len(result["episodes"]) == 10 and result["candidate_policy_queried"] is False
    assert all(row["status"] == "not_attempted" and row["reason"] for row in result["episodes"])
    assert "session_closeout" not in result
    assert result["physical_outcome_claimed"] is False


def test_missing_image_git_retains_each_cell_reason_without_install_or_model_fetch(tmp_path, monkeypatch):
    from tests.test_native_g1_beta_session import inputs
    from blueprint_pipeline.native_g1_beta_session import build_session_authority
    plan, candidate, spec = inputs(tmp_path, monkeypatch)
    authority = build_session_authority(plan=plan, candidate=candidate, spec=spec)
    monkeypatch.setattr(runtime, "_inputs", lambda root: (authority, spec, tmp_path / "native"))
    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)
    monkeypatch.setenv(runtime.DEADLINE_ENV, str(time.time() + 1800))
    monkeypatch.setattr(runtime, "_build_runtime", lambda *args: pytest.fail("unexpected bootstrap"))
    monkeypatch.setattr("blueprint_pipeline.native_g1_beta_download.stage_selected_assets", lambda **kwargs: pytest.fail("unexpected model fetch"))
    output = tmp_path / "output"
    output.mkdir()
    result = runtime.run_session(tmp_path, output)
    assert result["candidate_policy_queried"] is False
    assert len(result["episodes"]) == 10
    for row in result["episodes"]:
        assert row["reason"] == "g1_beta_image_git_unavailable"
        assert _episode_outcome(row, candidate_id=DP, contract_digest=spec["task_success_contract_digest"])[1] == row["reason"]
