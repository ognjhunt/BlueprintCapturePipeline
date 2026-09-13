"""Workspace pins and live consumers must survive bundle reclamation."""
from blueprint_pipeline import control_plane_storage_gc as gc
from blueprint_pipeline.control_plane_storage_pins import write_storage_pin, live_pinned_paths
from tests.test_control_plane_storage_gc import _workspace, _noclass


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



def test_gc_rechecks_pin_created_after_manifest_and_holds_producer_lock(tmp_path, monkeypatch):
    from blueprint_pipeline.control_plane_workspace_lock import workspace_lock
    monkeypatch.setattr(gc, 'workspace_process_active', lambda root: False)
    root, pins = tmp_path / 'workspaces', tmp_path / 'pins'
    now = 5_000_000.0
    workspace = _workspace(root, 'in-use', age=7 * 3600, now=now)
    manifest = gc.build_workspace_bundle_manifest(workspace_roots=[root], pins_root=pins,
        now=lambda: now, classifier=_noclass)
    with workspace_lock(workspace):
        outcome = gc.apply_workspace_bundle_manifest(manifest, ack=gc.WORKSPACE_BUNDLE_ACK, now=lambda: now)
    assert outcome['removed_count'] == 0
    write_storage_pin(pins_root=pins, kind='activation', owner_id='new-reader',
                      paths=[workspace / 'bundle'], now=lambda: now)
    outcome = gc.apply_workspace_bundle_manifest(manifest, ack=gc.WORKSPACE_BUNDLE_ACK, now=lambda: now)
    assert outcome['removed_count'] == 0
    assert (workspace / 'bundle').exists()



def test_legacy_workspace_without_producer_lock_is_retained(tmp_path, monkeypatch):
    monkeypatch.setattr(gc, 'workspace_process_active', lambda root: False)
    root, pins = tmp_path / 'workspaces', tmp_path / 'pins'
    now = 5_000_000.0
    workspace = _workspace(root, 'legacy', age=7 * 3600, now=now)
    (root / '.workspace-locks/legacy.lock').unlink()
    report = gc.build_workspace_bundle_manifest(workspace_roots=[root], pins_root=pins,
        now=lambda: now, classifier=_noclass)
    assert report['candidate_count'] == 0
    assert (workspace / 'bundle').exists()


def test_queued_workspace_is_retained_without_a_pin(tmp_path, monkeypatch):
    monkeypatch.setattr(gc, 'workspace_process_active', lambda root: False)
    root, pins, queue = tmp_path / 'workspaces', tmp_path / 'pins', tmp_path / 'queue'
    now = 5_000_000.0
    workspace = _workspace(root, 'queued-workspace', age=7 * 3600, now=now)
    (queue / 'pending').mkdir(parents=True)
    (queue / 'pending/job.json').write_text('{"workspace": "queued-workspace"}')
    report = gc.build_workspace_bundle_manifest(workspace_roots=[root], pins_root=pins,
        queue_roots=[queue], now=lambda: now, classifier=_noclass)
    assert report['candidate_count'] == 0 and (workspace / 'bundle').exists()
