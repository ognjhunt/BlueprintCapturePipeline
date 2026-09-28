"""Retained census annotations are proposals, never filesystem authority."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_scratch_decisions.py
#   scripts/lane_scratch_census.py

from __future__ import annotations

import hashlib
import json
import os

import pytest


def _json(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def _inventory(rows=()):
    return {"schema_version": "control_plane_lane_scratch_census.v1",
            "status": "complete", "observed_at_epoch": 900,
            "rows": list(rows), "candidate_count": len(rows), "entries_visited": len(rows),
            "unique_allocated_bytes": sum(row["allocated_bytes"] for row in rows),
            "scan_errors": [], "mutations": 0}


def _annotations(census, decisions=()):
    return _json({"schema_version": "control_plane_lane_scratch_annotations.v1",
                  "census_digest": "sha256:" + hashlib.sha256(census).hexdigest(),
                  "decisions": list(decisions)})


def _validate(census, annotations, **kwargs):
    from blueprint_pipeline.control_plane_lane_scratch_decisions import validate_census_annotations

    return validate_census_annotations(census, annotations, now=1000,
                                       allowed_roots=("/work", "/inputs"), **kwargs)


def _refused(code, census, annotations, **kwargs):
    from blueprint_pipeline.control_plane_lane_scratch_decisions import CensusDecisionError

    with pytest.raises(CensusDecisionError, match=f"^{code}$"):
        _validate(census, annotations, **kwargs)


def test_empty_census_binds_exact_retained_bytes():
    census = _json(_inventory())
    annotations = _annotations(census)
    result = _validate(census, annotations)
    assert result["census_digest"] == "sha256:" + hashlib.sha256(census).hexdigest()
    assert result["annotations_digest"] == "sha256:" + hashlib.sha256(annotations).hexdigest()
    assert result["mutations"] == 0 and result["execution_authorized"] is False
    assert result["requires_fresh_reference_check"] is True
    assert result["decisions"] == []
    _refused("census_identity_mismatch", census + b" ", annotations)


@pytest.mark.parametrize("payload", [b'{"status":1,"status":2}', b"[]", b"not json",
                                    b'{"x":NaN}', b'{"x":Infinity}', b'"\xff"'])
def test_ambiguous_or_nonfinite_json_refuses(payload):
    _refused("census_json_invalid", payload, _annotations(payload))


def test_input_and_output_bounds_are_injected_without_large_payloads():
    census = _json(_inventory())
    annotations = _annotations(census)
    _refused("census_input_too_large", census, annotations, max_input_bytes=16)
    _refused("census_validation_output_too_large", census, annotations, max_output_bytes=16)


def test_reader_reads_regular_file_without_mutation(tmp_path):
    from blueprint_pipeline.control_plane_lane_scratch_decisions import read_census_input

    path = tmp_path / "census.json"
    path.write_bytes(b"{}")
    assert read_census_input(path) == b"{}"
    assert path.read_bytes() == b"{}"


@pytest.mark.parametrize("kind", ["file_symlink", "parent_symlink", "directory", "missing"])
def test_reader_refuses_unsafe_or_missing_inputs(tmp_path, kind):
    from blueprint_pipeline.control_plane_lane_scratch_decisions import (
        CensusDecisionError, read_census_input,
    )

    real = tmp_path / "real"
    real.mkdir()
    source = real / "census.json"
    source.write_bytes(b"{}")
    if kind == "file_symlink":
        path = tmp_path / "linked.json"
        path.symlink_to(source)
    elif kind == "parent_symlink":
        parent = tmp_path / "linked"
        parent.symlink_to(real, target_is_directory=True)
        path = parent / "census.json"
    elif kind == "directory":
        path = real
    else:
        path = tmp_path / "missing"
    with pytest.raises(CensusDecisionError):
        read_census_input(path)


def test_reader_refuses_oversized_or_growing_input(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_lane_scratch_decisions as decisions

    path = tmp_path / "census.json"
    path.write_bytes(b"{}")
    with pytest.raises(decisions.CensusDecisionError, match="census_input_too_large"):
        decisions.read_census_input(path, max_bytes=1)
    real_read = os.read
    changed = False

    def grow(descriptor, count):
        nonlocal changed
        if not changed:
            changed = True
            path.write_bytes(b"{} ")
        return real_read(descriptor, count)

    monkeypatch.setattr(decisions.os, "read", grow)
    with pytest.raises(decisions.CensusDecisionError, match="census_input_unsafe"):
        decisions.read_census_input(path)


def test_reader_refuses_unreadable_input_with_typed_code(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_lane_scratch_decisions as decisions

    path = tmp_path / "census.json"
    path.write_bytes(b"{}")
    real_open = os.open

    def refuse(name, *args, **kwargs):
        if name == path.name:
            raise PermissionError("do not echo this private path")
        return real_open(name, *args, **kwargs)

    monkeypatch.setattr(decisions.os, "open", refuse)
    with pytest.raises(decisions.CensusDecisionError, match="^census_input_unreadable$"):
        decisions.read_census_input(path)


def _row(path='/work/experiment', references=()):
    return dict(path=path, family='experiment', owner_guess='unknown',
                owner_guess_basis='filename', allocated_bytes=4,
                newest_mtime_epoch=None, age_seconds=None, unreadable=0,
                shared_names=0, references=list(references), owner_decision=None,
                approved_expiry=None)


def _decision(action='keep', path='/work/experiment', **metadata):
    defaults = {'keep': {'expires_at_epoch': 1100},
                'register': dict(lane='ops', name='experiment', reason='retention_review',
                                 class_intent='cache', cleanup='owner_review', ttl_seconds=100,
                                 run_ref='run-1', size_budget_bytes=4),
                'delete': {'reason': 'owner_review'}, 'offload': {'reason': 'owner_review'}}
    return dict(path=path, action=action, owner='nijel') | defaults.get(action, {}) | metadata


@pytest.mark.parametrize('action', ['keep', 'register', 'delete', 'offload'])
def test_four_actions_preserve_exact_retained_reference_observations_without_authority(action, monkeypatch):
    from pathlib import Path
    from blueprint_pipeline import control_plane_lane_scratch as leases

    def forbidden(*args, **kwargs):
        raise AssertionError('pure validation inspected or mutated a target')

    census = _json(_inventory([_row()]))
    with monkeypatch.context() as context:
        context.setattr(Path, 'stat', forbidden)
        context.setattr(Path, 'resolve', forbidden)
        context.setattr(leases, 'create_lane_scratch', forbidden)
        result = _validate(census, _annotations(census, [_decision(action)]))
    assert result['decision_counts'][action] == 1
    assert result['decisions'][0]['path'] == '/work/experiment'
    assert result['decisions'][0]['references'] == []
    assert not result['execution_authorized'] and result['mutations'] == 0
    assert 'lease_digest' not in result['decisions'][0]


@pytest.mark.parametrize('patch,code', [
    ({'status': 'incomplete'}, 'census_inventory_incomplete'),
    ({'scan_errors': ['unreadable']}, 'census_inventory_incomplete'),
    ({'mutations': 1}, 'census_inventory_invalid'),
    ({'mutations': False}, 'census_inventory_invalid'),
    ({'candidate_count': 2}, 'census_inventory_invalid'),
    ({'entries_visited': True}, 'census_inventory_invalid'),
    ({'unique_allocated_bytes': 5}, 'census_inventory_invalid'),
    ({'observed_at_epoch': -1}, 'census_inventory_invalid'),
])
def test_inventory_consistency_refuses(patch, code):
    census = _json(_inventory([_row()]) | patch)
    _refused(code, census, _annotations(census, [_decision()]))


@pytest.mark.parametrize('patch,code', [
    ({'allocated_bytes': -1}, 'census_inventory_invalid'),
    ({'allocated_bytes': True}, 'census_inventory_invalid'),
    ({'unreadable': 1}, 'census_inventory_incomplete'),
    ({'newest_mtime_epoch': False}, 'census_inventory_invalid'),
    ({'age_seconds': -1}, 'census_inventory_invalid'),
    ({'owner_decision': 'delete'}, 'census_inventory_invalid'),
    ({'approved_expiry': 1100}, 'census_inventory_invalid'),
    ({'references': None}, 'census_reference_invalid'),
    ({'references': ['pin', 'pin']}, 'census_reference_invalid'),
    ({'references': ['future_ref']}, 'census_reference_invalid'),
])
def test_row_measurements_placeholders_and_references_refuse(patch, code):
    census = _json(_inventory([_row() | patch]))
    _refused(code, census, _annotations(census, [_decision()]))


@pytest.mark.parametrize('paths', [ ['/work'], ['/elsewhere/x'], ['/work/a/../b'],
    ['/work//a'], ['/work/<redacted>'], ['/work/a\\tb'],
    ['/work/a', '/work/a'], ['/work/a', '/work/a/child'] ])
def test_ambiguous_duplicate_or_overlapping_paths_refuse(paths):
    census = _json(_inventory([_row(path) for path in paths]))
    _refused('census_row_ambiguous', census, _annotations(census, [_decision(path=p) for p in paths]))


@pytest.mark.parametrize('decisions,code', [
    ([], 'census_decision_missing'),
    ([_decision(), _decision()], 'census_decision_duplicate'),
    ([_decision(path='/work/other')], 'census_decision_unknown_target'),
    ([_decision('move')], 'census_decision_action_invalid'),
    ([_decision(owner='bad owner')], 'census_decision_metadata_invalid'),
    ([_decision(expires_at_epoch=1000)], 'census_decision_metadata_invalid'),
    ([_decision(expires_at_epoch=1000+14*86400+1)], 'census_decision_metadata_invalid'),
    ([_decision(expires_at_epoch=True)], 'census_decision_metadata_invalid'),
    ([_decision(reason='wrong_action')], 'census_decision_metadata_invalid'),
    ([_decision(references=[])], 'census_decision_metadata_invalid'),
    ([_decision('register', scene_ref='scene-1')], 'census_decision_metadata_invalid'),
    ([_decision('register', run_ref=None)], 'census_decision_metadata_invalid'),
    ([_decision('register', ttl_seconds=14*86400+1)], 'census_decision_metadata_invalid'),
    ([_decision('register', size_budget_bytes=True)], 'census_decision_metadata_invalid'),
    ([_decision('delete', reason='unbounded reason text')], 'census_decision_metadata_invalid'),
])
def test_incomplete_or_contradictory_annotation_metadata_refuses(decisions, code):
    census = _json(_inventory([_row()]))
    _refused(code, census, _annotations(census, decisions))


@pytest.mark.parametrize('action', ['keep', 'register', 'delete', 'offload'])
def test_retained_references_block_cleanup_proposals_only(action):
    census = _json(_inventory([_row(references=['pin', 'active_run'])]))
    annotations = _annotations(census, [_decision(action)])
    if action in ('delete', 'offload'):
        _refused('census_decision_referenced', census, annotations)
    else:
        assert _validate(census, annotations)['decisions'][0]['references'] == ['pin', 'active_run']


def test_lane_layout_registration_metadata_matches_retained_path():
    path = '/work/lanes/ops/experiment'
    census = _json(_inventory([_row(path)]))
    assert _validate(census, _annotations(census, [_decision('register', path=path)]))['decision_count'] == 1
    _refused('census_decision_metadata_invalid', census,
             _annotations(census, [_decision('register', path=path, name='other')]))


@pytest.mark.parametrize('roots,now', [(['/work/../work'],1000), (['work'],1000),
                                     ([],1000), (['/work'],True), (['/work'],float('inf'))])
def test_validation_context_must_be_finite_and_lexically_canonical(roots, now):
    from blueprint_pipeline.control_plane_lane_scratch_decisions import CensusDecisionError, validate_census_annotations
    census = _json(_inventory())
    with pytest.raises(CensusDecisionError):
        validate_census_annotations(census, _annotations(census), now=now, allowed_roots=roots)


def test_huge_integer_measurement_is_typed_refusal():
    census = _json(_inventory([_row() | {'newest_mtime_epoch': 10**400}]))
    _refused('census_inventory_invalid', census, _annotations(census, [_decision()]))


def test_nonadjacent_lexical_overlap_and_missing_reference_accounting_refuse():
    paths = ['/work/a', '/work/a-other', '/work/a/child']
    census = _json(_inventory([_row(p) for p in paths]))
    _refused('census_row_ambiguous', census, _annotations(census, [_decision(path=p) for p in paths]))
    row = _row()
    del row['references']
    census = _json(_inventory([row]))
    _refused('census_reference_invalid', census, _annotations(census, [_decision()]))


def test_maximum_decision_rows_and_deterministic_path_order(monkeypatch):
    from blueprint_pipeline import control_plane_lane_scratch_decisions as module
    rows = [_row('/work/z'), _row('/work/a')]
    census = _json(_inventory(rows))
    annotations = _annotations(census, [_decision(path=r['path']) for r in rows])
    result = _validate(census, annotations)
    assert [d['path'] for d in result['decisions']] == ['/work/a', '/work/z']
    monkeypatch.setattr(module, 'MAX_ROWS', 1)
    _refused('census_inventory_invalid', census, annotations)
