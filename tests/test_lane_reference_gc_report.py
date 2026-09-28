# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_storage_gc.py
#   src/blueprint_pipeline/control_plane_lane_reference_collection.py
"""Exact target is an optional nested report, including with apply authorized."""
from dataclasses import asdict

import pytest

from blueprint_pipeline import control_plane_storage_gc as gc
from blueprint_pipeline import control_plane_lane_reference_collection as collector


@pytest.mark.parametrize('apply', [False, True])
def test_exact_target_report_cannot_route_to_lane_apply(tmp_path, monkeypatch, apply):
    captured = []
    target = '/mnt/blueprint-work/lanes/g1/pair'
    result = collector.LaneReferenceCollection(collector.TargetProbe('kept', target), 'collector_process_effective_only',
                                              (), (), (), ('consumer_participation_unproven',), False)
    def collect(path, **options):
        captured.append((path, options))
        return result
    monkeypatch.setattr(collector, 'collect_gc_lane_references', collect)
    monkeypatch.setattr(gc, 'reconcile_terminal_cache_pins', lambda **options: {'status': 'applied'})
    monkeypatch.setattr(gc, 'observe_lane_scratch_retention', lambda *args, **options: {'status': 'report_only', 'complete': True})
    for name in ('build_scratch_manifest', 'apply_scratch_manifest', 'build_derived_directory_manifest', 'build_workspace_bundle_manifest'):
        monkeypatch.setattr(gc, name, lambda **options: pytest.fail('target routed to applying phase'))
    report = gc.run_storage_gc(content_store_roots=(), derived_roots=(), queue_roots=('/selected/queue',), pins_root=str(tmp_path),
                               lane_reference_target=target, apply=apply, ack=gc.RUN_ACK if apply else '', now=lambda: 110)
    assert report['lane_scratch']['reference_collection'] == asdict(result)
    assert captured[0][1]['queue_roots'] == ('/selected/queue',)
    assert not report.get('phase_errors')
    assert 'lane_reference_collection_incomplete' in report['alerts']
    assert not report['lane_scratch']['reference_collection']['references_clear']


def test_default_gc_does_not_collect_processes_or_add_new_output(tmp_path, monkeypatch):
    monkeypatch.setattr(collector, 'collect_gc_lane_references', lambda *args, **options: pytest.fail('default collected proc'))
    monkeypatch.setattr(gc, 'reconcile_terminal_cache_pins', lambda **options: {'status': 'applied'})
    report = gc.run_storage_gc(content_store_roots=(), derived_roots=(), queue_roots=(), pins_root=tmp_path, now=lambda: 110)
    assert 'reference_collection' not in report['lane_scratch']


@pytest.mark.slow
def test_cli_forwards_one_explicit_target_and_ordinary_default(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(gc, 'running_release_commit', lambda: '')
    monkeypatch.setattr(gc, 'run_storage_gc', lambda **options: captured.append(options) or {})
    assert gc.main(['run', '--pins-root', str(tmp_path), '--lane-reference-target', '/mnt/blueprint-work/lanes/g1/pair']) == 0
    assert captured[-1]['lane_reference_target'] == '/mnt/blueprint-work/lanes/g1/pair'
    assert gc.main(['run', '--pins-root', str(tmp_path)]) == 0
    assert captured[-1]['lane_reference_target'] is None


@pytest.mark.slow
@pytest.mark.parametrize('arguments', [
    ['--lane-reference-target'],
    ['--lane-reference-target', '/private-secret', '--unknown-secret'],
    ['--lane-reference-t', '/private-secret', '--hot-window-seconds', 'secret-invalid'],
])
def test_selected_report_cli_parse_errors_never_echo_raw_arguments(arguments, capsys, monkeypatch):
    monkeypatch.setattr(gc, 'running_release_commit', lambda: '')
    assert gc.main(['run', *arguments]) == 2
    captured = capsys.readouterr()
    assert captured.err == ''
    assert 'secret' not in captured.out and 'usage:' not in captured.out
    assert 'lane_reference_parameters_invalid' in captured.out


@pytest.mark.slow
def test_ordinary_gc_parser_keeps_its_existing_error_behavior(capsys, monkeypatch):
    monkeypatch.setattr(gc, 'running_release_commit', lambda: '')
    with pytest.raises(SystemExit) as error:
        gc.main(['run', '--unknown-ordinary'])
    assert error.value.code == 2
    assert 'usage:' in capsys.readouterr().err
