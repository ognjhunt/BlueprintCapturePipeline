"""A6: the terminal-result index gathers the paid run's real sealed producer
receipts into the reconciler's owner directory and produces the sealed result
publication; the REAL reconciler then completes the owner status from them.

This closes the producer gap the reconciler tests otherwise papered over: here the
six receipts are written to a SOURCE directory (their real, scattered producer
locations) and the index -- not the test -- files them for the reconciler.
"""
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_scene_terminal_result_index as index
from blueprint_pipeline import task_evaluation_scene_terminal_reconciler as reconciler
from tests.test_task_evaluation_scene_terminal_reconciler import (
    _env, _completed_projection, _readback, _closure, _launch_request, _profile, _publication, _write,
)


def _source_outputs(tmp_path, env):
    """The paid run's sealed producer receipts at a SOURCE dir (not the reconciler
    terminal_result_root)."""
    src = tmp_path / "run-outputs"
    src.mkdir()
    projection = _completed_projection()
    profile = _profile(env["binding"])
    _write(src / "policy_canary_result_projection.json", projection)
    _write(src / "launch_profile.json", profile)
    _write(src / "launch_request.json", _launch_request(profile["profile_digest"]))
    _write(src / "policy_canary_webapp_sync.json", _readback(projection))
    _write(src / "post_teardown_provider_zero_receipt.json", _closure(profile["profile_digest"]))
    return src, projection, _publication(projection)


def _index(env, src, publication):
    return index.index_terminal_owner_result(
        intent_id=env["intent_id"], terminal_result_root=env["config"]["terminal_result_root"],
        projection_path=src / "policy_canary_result_projection.json",
        webapp_sync_path=src / "policy_canary_webapp_sync.json",
        post_teardown_provider_zero_path=src / "post_teardown_provider_zero_receipt.json",
        launch_request_path=src / "launch_request.json", launch_profile_path=src / "launch_profile.json",
        result_uri=publication["uri"], result_size_bytes=publication["size_bytes"])


def test_index_then_reconcile_round_trip_completes(tmp_path):
    env = _env(tmp_path)
    src, projection, publication = _source_outputs(tmp_path, env)
    receipt = _index(env, src, publication)
    assert receipt["status"] == "terminal_result_indexed"
    assert receipt["provider_mutation_performed"] is False
    assert set(receipt["files"]) == {
        "policy_canary_result_projection.json", "policy_canary_webapp_sync.json", "provider_zero_closure.json",
        "launch_request.json", "launch_profile.json", "terminal_result_publication.json"}
    # The REAL reconciler now completes the owner status from the indexed receipts.
    result = reconciler.reconcile_terminal_owner_result(
        intent=env["intent"], config=env["config"], release=env["release"], now=env["now"], output=env["output"])
    assert result["terminal"] is True and result["status"] == "completed"
    assert result["result_reference"]["digest"] == projection["projection_digest"]
    # Idempotent: re-indexing the same sealed inputs is byte-identical.
    assert _index(env, src, publication) == receipt


def test_index_refuses_off_contract_receipt(tmp_path):
    env = _env(tmp_path)
    src, projection, publication = _source_outputs(tmp_path, env)
    bad = json.loads((src / "policy_canary_webapp_sync.json").read_text())
    bad["schema_version"] = "something_else.v1"  # not the reconciler's contract
    (src / "policy_canary_webapp_sync.json").write_text(json.dumps(bad))
    with pytest.raises(index.TerminalResultIndexError, match="receipt_schema_invalid"):
        _index(env, src, publication)


def test_index_refuses_invalid_result_uri(tmp_path):
    env = _env(tmp_path)
    src, projection, publication = _source_outputs(tmp_path, env)
    with pytest.raises(index.TerminalResultIndexError, match="result_uri_invalid"):
        index.index_terminal_owner_result(
            intent_id=env["intent_id"], terminal_result_root=env["config"]["terminal_result_root"],
            projection_path=src / "policy_canary_result_projection.json",
            webapp_sync_path=src / "policy_canary_webapp_sync.json",
            post_teardown_provider_zero_path=src / "post_teardown_provider_zero_receipt.json",
            launch_request_path=src / "launch_request.json", launch_profile_path=src / "launch_profile.json",
            result_uri="ftp://not-durable/result", result_size_bytes=publication["size_bytes"])
