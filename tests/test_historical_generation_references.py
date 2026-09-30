"""Current historical reference tables; no deletion authority follows."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_historical_references.py
import os
import fcntl
import json
from pathlib import Path

import pytest

from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
from tests.test_historical_generation_authority import historical_installation  # noqa: F401
from tests.test_registered_experiment_issuer import installation, encoded  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401


def scan(root, target):
    from blueprint_pipeline.control_plane_lane_historical_references import _table
    budget = ReferenceCollectionBudget()
    try:
        return _table(Path(root), Path(target), budget)
    finally:
        budget.close()


def test_pending_row_naming_generation_is_a_reference(tmp_path):
    queue = tmp_path / 'queue'
    (queue / 'pending').mkdir(parents=True)
    target = tmp_path / 'selected'
    (queue / 'pending/job.json').write_text('{"source":"' + str(target / 'input.bin') + '"}')
    with pytest.raises(ValueError, match='table_reference'):
        scan(queue, target)


def test_closed_unchanged_tables_bind_exact_namespace_and_bytes(tmp_path):
    (tmp_path / 'processing').mkdir()
    (tmp_path / 'processing/job.json').write_text('{"run":"other"}')
    before = scan(tmp_path, Path('/unrelated/selected'))
    assert len(before) == 3
    (tmp_path / 'processing/job.json').write_text('{"run":"changed"}')
    assert scan(tmp_path, Path('/unrelated/selected')) != before


@pytest.mark.parametrize('kind', ['symlink', 'hardlink', 'malformed', 'fifo'])
def test_unknown_reference_rows_are_not_absence(tmp_path, kind):
    row = tmp_path / 'job.json'
    if kind == 'symlink':
        row.symlink_to(tmp_path / 'absent')
    elif kind == 'hardlink':
        row.write_text('{}')
        os.link(row, tmp_path / 'alias.json')
    elif kind == 'fifo':
        os.mkfifo(row)
    else:
        row.write_text('{bad json')
    with pytest.raises(ValueError, match='table_unknown'):
        scan(tmp_path, Path('/unrelated/selected'))


@pytest.fixture
def reference_installation(historical_installation, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_legacy_owner as legacy
    from blueprint_pipeline.control_plane_storage_pins import PIN_KINDS
    config, target, *_ = historical_installation
    root = config.parent
    for name in ('control-state', 'queue', 'active-runs', 'settlement', 'pins', 'release'):
        (root / name).mkdir()
    for kind in PIN_KINDS:
        (root / 'pins' / kind).mkdir()
    (root / 'active').symlink_to(root / 'release')
    unit, environment = root / 'gc.service', root / 'gc.env'
    unit.write_bytes(b'[Service]\n')
    settings = {
        'BLUEPRINT_CONTROL_PLANE_GC_QUEUE_ROOTS': root / 'queue',
        'BLUEPRINT_CONTROL_PLANE_GC_EVIDENCE_ROOTS': root / 'active-runs',
        'BLUEPRINT_CONTROL_PLANE_GC_SETTLEMENT_ROOTS': root / 'settlement',
        'BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT': root / 'pins'}
    environment.write_text('\n'.join(key + '=' + str(value) for key, value in settings.items()) + '\n')
    unit.chmod(0o600)
    environment.chmod(0o600)
    configured = json.loads(config.read_bytes()) | dict(control_plane_state=str(root / 'control-state'),
        experiment_gc_environment_file=str(environment), active_release_link=str(root / 'active'))
    config.write_bytes(encoded(configured))
    monkeypatch.setattr(legacy, '_GC_UNIT', unit)
    return config, target, root


def reference_call(installed, callback):
    from blueprint_pipeline import control_plane_lane_historical_authority as authority
    from blueprint_pipeline.control_plane_lane_historical_references import historical_reference_fence
    config, target, _ = installed
    with authority._session(config, authority._Operation(1030, lambda: 0)) as (files, settings, _):
        with historical_reference_fence(files, settings, target, observed_at=1030) as guard:
            return callback(guard)


def test_current_reference_fence_holds_real_publisher_locks(reference_installation):
    root = reference_installation[2]
    def check(guard):
        for path in (root / 'control-state', root / 'pins'):
            descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
            finally:
                os.close(descriptor)
        guard()
    reference_call(reference_installation, check)


@pytest.mark.parametrize('changed', ['queue', 'release', 'pin'])
def test_current_reference_change_is_detected_before_effect(reference_installation, changed):
    _, target, root = reference_installation
    def change(guard):
        if changed == 'queue':
            (root / 'queue/job.json').write_text(json.dumps({'input': str(target / 'one.log')}))
        elif changed == 'release':
            (root / 'active').unlink()
            (root / 'active').symlink_to(target)
        else:
            from blueprint_pipeline.control_plane_storage_pins import PIN_KINDS, SCHEMA_VERSION
            kind = sorted(PIN_KINDS)[0]
            (root / 'pins' / kind / 'new.json').write_text(json.dumps(dict(schema_version=SCHEMA_VERSION,
                kind=kind, owner_id='new', paths=[str(target)], depends_on=[], created_at_epoch=1020,
                expires_at_epoch=1100, released_at_epoch=None)))
        guard()
    with pytest.raises(ValueError):
        reference_call(reference_installation, change)
    assert (target / 'one.log').read_bytes() == b'original owner diagnostics\n'
