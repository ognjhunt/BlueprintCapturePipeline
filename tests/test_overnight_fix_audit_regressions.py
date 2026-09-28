"""Contracts from the 2026-09-13 overnight-fixes audit (hermetic, no paid calls).

* settling a retired source attempt never disables its own authorized recovery;
* a terminal launch's settlement keeps its unproven spend held instead of refunding it;
* workspace-bundle cleanup honours live storage pins regardless of file age.
"""
import json
import pytest

from blueprint_pipeline import control_plane_storage_gc as gc
from blueprint_pipeline.control_plane_storage_pins import write_storage_pin, live_pinned_paths
from tests.test_control_plane_storage_gc import _workspace, _noclass
from tests.test_task_evaluation_scene_recovery import setup
from tests.test_terminal_scene_attempt_settlement import _fixture, _settle, _reserve
from blueprint_pipeline import task_evaluation_scene_progression as progression
from blueprint_pipeline import task_evaluation_scene_progression_recovery as recovery
from blueprint_pipeline.task_evaluation_scene_intake import SceneIntakeError
from blueprint_pipeline.task_evaluation_sam31_prefix_adoption import record


def test_settling_retired_source_must_not_disable_its_authorized_recovery(tmp_path, monkeypatch):
    intent, prior, evidence = setup(tmp_path)
    directory = tmp_path / intent['intent_id']
    # External observations are already captured by the existing recovery fixture.
    # Drive the actual production recovery function, settlement, and reservation.
    monkeypatch.setattr(recovery, 'retain_failure', lambda **kw: tmp_path / 'failure.json')
    monkeypatch.setattr(recovery, 'reconcile_ownership', lambda **kw: evidence)
    full_intent = json.loads((directory / 'intent.json').read_text())
    state = {'binding_digest': 'sha256:' + 'f' * 64,
             'attempt': record(directory / 'attempts/a1.json')}
    assert progression._recover(directory=directory, intent=full_intent, state=state,
        attempt=prior, link={}, output=tmp_path / 'out', now=104,
        config={'intent_root': tmp_path, 'child_queue_root': tmp_path / 'children',
                'launch_execution_root': tmp_path / 'launch-runs',
                'launch_queue_root': tmp_path / 'launches'},
        release={'source_commit': 'c' * 40, 'runtime_digest': 'sha256:' + 'e' * 64},
        machinery={'maximum_preparation_spend_usd': 2}) is True


def test_paid_terminal_launch_must_not_be_refunded_without_cost_reconciliation(tmp_path):
    fixture = _fixture(tmp_path, launch_status='completed')
    _settle(fixture)
    # Only a completed launch status exists: no zero-cost proof or actual-cost
    # settlement. The reservation must not vanish from cumulative spend/count.
    with pytest.raises(SceneIntakeError, match='spend_cap_exhausted'):
        _reserve(fixture['root'], fixture['intent'], 'new-full-budget-attempt', 26.0, now=300)


def test_workspace_gc_must_honor_a_live_pin_even_when_files_are_old(tmp_path):
    root, pins = tmp_path / 'semantic-pretraining', tmp_path / 'pins'
    now = 5_000_000.0
    workspace = _workspace(root, 'still-needed', age=7 * 3600, now=now)
    write_storage_pin(pins_root=pins, kind='activation', owner_id='active-launch',
                      paths=[workspace], now=lambda: now)
    assert str(workspace) in live_pinned_paths(pins, now=lambda: now)
    report = gc.run_storage_gc(content_store_roots=[], derived_roots=[], queue_roots=[],
        pins_root=pins, workspace_bundle_roots=[root], apply=True, ack=gc.RUN_ACK,
        now=lambda: now, classifier=_noclass)
    assert report['workspace_bundles']['removed_count'] == 0
    assert (workspace / 'bundle/provider_runtime/runtime.bin').exists()
