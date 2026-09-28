# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_inventory_seed.py
#   src/blueprint_pipeline/task_evaluation_scene_preparation_lineage.py
#   src/blueprint_pipeline/task_evaluation_scene_source_attempt_lineage.py
"""Tiny resource and cold-purity proofs for the supplied inventory boundary."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.test_scene_inventory_history import ROLES, api, event, fixture, project, refuses
from tests.test_scene_inventory_preparations import fixture as downstream, replace_record


@pytest.mark.parametrize("constant", ["MAX_RECORDS", "MAX_RECORD_BYTES", "MAX_TOTAL_BYTES"])
def test_input_limits_refuse_before_record_hash_or_json(monkeypatch, constant):
    args = fixture()
    module = api()
    monkeypatch.setattr(module, constant, 0)
    monkeypatch.setattr(module.retained, "_record", lambda *args: pytest.fail("parsed before input cap"))
    refuses(args)


@pytest.mark.parametrize("raw", [b'{"same":1,"same":2}', b'{"bad":NaN}', b'{"bad":1e999}', br'{"bad":"\ud800"}',
                                b'[]', b'null', b'\xff', b''])
def test_malformed_raw_inputs_have_fixed_bounded_errors(raw):
    args = fixture()
    args["records"]["intent"] = (args["records"]["intent"][0], raw)
    refuses(args)


@pytest.mark.parametrize("value", [bytearray(b'{}'), "{}", 1, None])
def test_only_original_bytes_are_input_provenance(value):
    args = fixture()
    args["records"]["intent"] = (args["records"]["intent"][0], value)
    refuses(args)


@pytest.mark.parametrize("path", ["relative", "/a/../b", "/a//b", "/a/./b", "/a\\b", "/a\x00b", "/\ud800", "/" + "x" * 4096])
def test_invalid_lexical_input_paths_refuse(path):
    args = fixture()
    args["records"]["intent"] = (path, args["records"]["intent"][1])
    refuses(args)


def test_raw_input_byte_cap_is_inclusive(monkeypatch):
    args = fixture()
    size = len(args["records"]["intent"][1])
    module = api()
    monkeypatch.setattr(module, "MAX_TOTAL_BYTES", size)
    monkeypatch.setattr(module, "MAX_RECORD_BYTES", size)
    assert module.join_retained_scene_inventory_seed(**args)["mutations"] == 0
    monkeypatch.setattr(module, "MAX_TOTAL_BYTES", size - 1)
    refuses(args, "bytes_limit")


def test_successful_output_is_never_truncated_to_fit_caps(monkeypatch):
    args = downstream(configuration=True)
    result = api().join_retained_scene_inventory_seed(**args)
    size = len(api().retained._encoded(result))
    monkeypatch.setattr(api(), "MAX_OUTPUT_BYTES", size)
    assert api().join_retained_scene_inventory_seed(**args) == result
    monkeypatch.setattr(api(), "MAX_OUTPUT_BYTES", size - 1)
    refuses(args, "output_limit")
    monkeypatch.setattr(api(), "MAX_OUTPUT_BYTES", size)
    monkeypatch.setattr(api(), "MAX_ROWS", 0)
    refuses(args, "rows_limit")


def test_unknown_historical_role_path_stays_unproven_even_when_bytes_absent():
    args = fixture()
    event(args, {"factory": {"path": "/unrelated/shared-cache/receipt.json", "sha256": "sha256:" + "b" * 64, "size_bytes": 1}})
    result = api().join_retained_scene_inventory_seed(**args)
    row = result["obligations"][0]
    assert row["reason"] == "historical_reference_role_unproven" and row["status"] == "kept_deferred"
    assert result["members"] == []


def test_all_supplied_extra_or_wrong_owned_proof_refuses_even_when_other_edges_missing():
    for role in ("preparation_envelopes", "preparation_results", "configuration_progressions", "activation_envelopes"):
        args = downstream(configuration=True)
        path, raw = args["records"][role][0]
        args["records"][role][0] = ("/unrelated/" + Path(path).name, raw)
        args["records"]["preparation_envelopes"] = [] if role != "preparation_envelopes" else args["records"][role]
        refuses(args)


def test_resealed_paid_runtime_mismatch_is_not_hidden_by_available_request():
    args = downstream(activation=True)
    replace_record(args, "attempts", lambda row: row.update(runtime_digest="sha256:" + "c" * 64), "attempt_digest", cross=True)
    from tests.test_scene_inventory_history import ref
    replace_record(args, "preparation_links", lambda row: row.update(scene_configuration_attempt=ref(args["records"]["attempts"][0])), "link_digest", position=1)
    refuses(args, "attempt_runtime_invalid")


def test_deferred_branch_evidence_is_never_silent_full_coverage():
    args = fixture()
    project(args, event(args, {"preparation_failure": {"anything": "not recursively joined"}, "lookahead": {"path": "/not-owned"}}))
    result = api().join_retained_scene_inventory_seed(**args)
    assert result["deferred_branch_count"] == 2 and not result["remaining_branches_complete"]
    assert result["members"] == []


def test_event_launch_branch_is_deferred_without_launch_membership():
    args = fixture()
    project(args, event(args, {"launch": {"path": "/not-owned"}}))
    result = api().join_retained_scene_inventory_seed(**args)
    assert result["deferred_branch_count"] == 1 and result["members"] == []


def test_launch_projection_is_not_admitted_into_activation_projection_role():
    args = downstream(configuration=True)
    path, raw = args["records"]["configuration_progressions"][0]
    args["records"]["configuration_progressions"][0] = (path.replace("activation_progression.json", "launch_progression.json"), raw)
    refuses(args)


def test_downstream_ordering_is_deterministic_across_variants_and_versions():
    args = downstream(activation=True, configuration=True)
    path, raw = args["records"]["preparation_results"][0]
    value = json.loads(raw)
    value["references"][0]["content_addressed_reuse"] = True
    from tests.test_scene_inventory_history import pair, seal
    args["records"]["preparation_results"].append(pair(path, seal(value, "result_digest")))
    expected = api().join_retained_scene_inventory_seed(**args)
    for role in ROLES:
        args["records"][role].reverse()
    assert api().join_retained_scene_inventory_seed(**args) == expected


def test_no_mutation_calls_occur_during_supplied_join(monkeypatch):
    args = downstream(configuration=True)
    module = api()
    def forbidden(*args, **kwargs):
        pytest.fail("pure inventory touched filesystem or process")
    import builtins
    for owner, names in ((builtins, ("open",)), (os, ("open", "stat", "lstat", "mkdir", "rename", "replace", "unlink")),
                         (Path, ("resolve", "stat", "lstat", "read_bytes", "read_text", "write_bytes", "write_text", "glob", "mkdir")),
                         (subprocess, ("run", "Popen"))):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    assert module.join_retained_scene_inventory_seed(**args)["mutations"] == 0


@pytest.mark.slow
@pytest.mark.parametrize("activation", [False, True])
def test_first_inventory_join_in_fresh_process_has_no_runtime_import_or_filesystem_effect(tmp_path, activation):
    args = downstream(activation=activation, configuration=True)
    packet = tmp_path / "packet.json"
    serializable = copy.deepcopy(args)
    for role in ("intent", "projection"):
        row = serializable["records"][role]
        if row is not None:
            serializable["records"][role] = [row[0], row[1].decode()]
    for role in ROLES:
        serializable["records"][role] = [[path, raw.decode()] for path, raw in args["records"][role]]
    packet.write_text(json.dumps(serializable))
    script = r'''
import json, pathlib, sys
packet=json.loads(pathlib.Path(sys.argv[1]).read_text())
for role in ('intent','projection'):
 row=packet['records'][role]
 if row is not None: packet['records'][role]=(row[0],row[1].encode())
for role,rows in packet['records'].items():
 if role not in ('intent','projection'): packet['records'][role]=[(p,r.encode()) for p,r in rows]
def forbidden(*args,**kwargs): raise AssertionError('cold filesystem effect')
for name in ('resolve','stat','lstat','read_bytes','read_text','write_bytes','write_text','glob','mkdir'):
 setattr(pathlib.Path,name,forbidden)
from blueprint_pipeline.task_evaluation_scene_inventory_seed import join_retained_scene_inventory_seed
result=join_retained_scene_inventory_seed(**packet)
assert result['mutations']==0 and result['cleanup_authorized'] is False
for name in ('task_evaluation_scene_intake','task_evaluation_scene_progression_state','task_evaluation_scene_progression',
 'task_evaluation_launch_preparation_worker','task_evaluation_scene_configuration_activation_automation',
 'task_evaluation_launch_preparation_contract','task_evaluation_controls_worker'):
 assert 'blueprint_pipeline.'+name not in sys.modules,name
print('pure')
'''
    result = subprocess.run([sys.executable, "-c", script, str(packet)], env={**os.environ, "PYTHONPATH": "src"},
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "pure"
