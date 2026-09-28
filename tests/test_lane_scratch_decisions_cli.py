"""Retained validation CLI publishes reports without rescanning or applying proposals."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_scratch_decisions.py
#   scripts/lane_scratch_census.py

from __future__ import annotations

import hashlib
import json
import os

import pytest

from scripts import lane_scratch_census as cli


def _inputs(tmp_path):
    census = tmp_path / 'census.json'
    annotation = tmp_path / 'annotations.json'
    payload = (json.dumps(dict(schema_version='control_plane_lane_scratch_census.v1',
        status='complete', observed_at_epoch=900, rows=[], candidate_count=0,
        entries_visited=0, unique_allocated_bytes=0, scan_errors=[], mutations=0))+'\n').encode()
    census.write_bytes(payload)
    annotation.write_text(json.dumps(dict(schema_version='control_plane_lane_scratch_annotations.v1',
        census_digest='sha256:'+hashlib.sha256(payload).hexdigest(), decisions=[])))
    return census, annotation


def _args(census, annotation):
    return ['--validate-census', str(census), '--annotations', str(annotation),
            '--work-root', '/work', '--inputs-root', '/inputs']


def _snapshot(tmp_path):
    return {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_cli_validation_is_deterministic_and_has_no_scan_or_target_mutation(tmp_path, monkeypatch, capsys):
    from blueprint_pipeline import control_plane_lane_scratch as leases
    census, annotations = _inputs(tmp_path)
    before = _snapshot(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError('validation called scan or target mutation')

    with monkeypatch.context() as patch:
        patch.setattr(cli, 'build_census', forbidden)
        for name in ('create_lane_scratch', 'renew_lane_scratch', 'release_lane_scratch', '_publish_scratch_folder'):
            patch.setattr(leases, name, forbidden)
        patch.setattr(os, 'unlink', forbidden)
        patch.setattr(os, 'rename', forbidden)
        patch.setattr(os, 'replace', forbidden)
        assert cli.main(_args(census, annotations)) == 0
        first = capsys.readouterr().out
        assert cli.main(_args(census, annotations)) == 0
        second = capsys.readouterr().out
    assert first == second
    report = json.loads(first)
    assert report['execution_authorized'] is False and report['mutations'] == 0
    assert report['requires_fresh_reference_check'] is True
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize('extra', [['--process-root', '/proc'], ['--pins-root', '/pins'],
    ['--release-link', '/release'], ['--queue-root', '/queue'], ['--queue-inventory-empty'],
    ['--active-run-root', '/runs'], ['--active-run-inventory-empty'], ['--max-seconds=240'],
    ['--max-s=1'], ['--process', '/tmp/proc']])
def test_cli_scan_arguments_are_refused_in_validation_mode(tmp_path, extra, capsys):
    census, annotations = _inputs(tmp_path)
    assert cli.main(_args(census, annotations)+extra) != 0
    report = json.loads(capsys.readouterr().out)
    assert report['blockers'] == ['census_annotations_invalid']
    assert report['mutations'] == 0


@pytest.mark.parametrize('flag', ['--validate-census', '--annotations'])
def test_cli_validation_argument_pairing_is_typed(tmp_path, flag, capsys):
    census, annotations = _inputs(tmp_path)
    assert cli.main([flag, str(census if flag == '--validate-census' else annotations)]) != 0
    output = capsys.readouterr().out
    assert len(output.encode()) < 4096
    assert json.loads(output)['blockers'] == ['census_annotations_invalid']


def test_cli_failure_is_bounded_typed_and_never_writes_artifact(tmp_path, capsys):
    census, annotations = _inputs(tmp_path)
    annotations.write_bytes(b'{"private_path":"/secret/host/path", "decisions":NaN}')
    target = tmp_path / 'artifact.json'
    assert cli.main(_args(census, annotations)+['--json-out', str(target)]) != 0
    output = capsys.readouterr().out
    assert 'private_path' not in output and '/secret' not in output
    assert len(output.encode()) < 4096
    assert json.loads(output)['blockers'] == ['census_json_invalid']
    assert not target.exists()


def test_cli_artifact_matches_bounded_stdout_without_changing_inputs(tmp_path, capsys):
    census, annotations = _inputs(tmp_path)
    before = (census.read_bytes(), annotations.read_bytes())
    target = tmp_path / 'validation.json'
    assert cli.main(_args(census, annotations)+['--json-out', str(target)]) == 0
    assert target.read_bytes() == capsys.readouterr().out.encode()
    assert (census.read_bytes(), annotations.read_bytes()) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ['annotations.json','census.json','validation.json']


@pytest.mark.parametrize('kind', ['census', 'annotations', 'hardlink', 'symlink', 'linked_parent'])
def test_cli_artifact_refuses_input_aliases_and_linked_output_parents(tmp_path, kind, capsys):
    census, annotations = _inputs(tmp_path)
    target = tmp_path / 'output.json'
    if kind == 'census':
        target = census
    elif kind == 'annotations':
        target = annotations
    elif kind == 'hardlink':
        os.link(census, target)
    elif kind == 'symlink':
        target.symlink_to(census)
    else:
        parent = tmp_path / 'linked'
        parent.symlink_to(tmp_path, target_is_directory=True)
        target = parent / 'output.json'
    before = (census.read_bytes(), annotations.read_bytes())
    assert cli.main(_args(census, annotations)+['--json-out', str(target)]) != 0
    assert json.loads(capsys.readouterr().out)['blockers'] == ['census_input_unsafe']
    assert (census.read_bytes(), annotations.read_bytes()) == before


def test_cli_output_parent_replacement_cannot_redirect_over_source(tmp_path, monkeypatch, capsys):
    census, annotations = _inputs(tmp_path)
    output_parent = tmp_path / 'reports'
    output_parent.mkdir()
    old_parent = tmp_path / 'retained-reports'
    target = output_parent / census.name
    before = (census.read_bytes(), annotations.read_bytes())
    real_open = os.open
    replaced = False

    def race(name, flags, *args, **kwargs):
        nonlocal replaced
        if isinstance(name, str) and name.endswith('.tmp') and not replaced:
            replaced = True
            output_parent.rename(old_parent)
            output_parent.symlink_to(tmp_path, target_is_directory=True)
        return real_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(os, 'open', race)
    assert cli.main(_args(census, annotations)+['--json-out', str(target)]) == 0
    assert replaced
    assert (census.read_bytes(), annotations.read_bytes()) == before
    assert (old_parent / census.name).read_bytes() == capsys.readouterr().out.encode()
