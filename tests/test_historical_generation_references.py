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
from tests.test_historical_generation_authority import packet, decision
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


@pytest.mark.parametrize('encoding', ['dash', 'entire_path'])
def test_encoded_pending_file_uri_is_an_actual_current_reference(tmp_path, encoding):
    from urllib.parse import unquote, urlparse
    queue = tmp_path / 'queue'
    queue.mkdir()
    target = tmp_path / 'selected-diagnostics'
    target.mkdir()
    payload = target / 'input.bin'
    payload.write_bytes(b'original owner bytes')
    selected = str(payload)
    encoded_path = selected.replace('-', '%2D') if encoding == 'dash' else ''.join(
        '/' if byte == 47 else '%' + format(byte, '02X') for byte in os.fsencode(selected))
    uri = 'file://' + encoded_path
    parsed = urlparse(uri)
    assert parsed.scheme == 'file' and parsed.netloc == ''
    assert Path(unquote(parsed.path)) == payload
    row = queue / 'job.json'
    original = json.dumps({'source_uri': uri}).encode()
    row.write_bytes(original)
    assert target.name not in encoded_path and str(target) not in encoded_path
    with pytest.raises(ValueError, match='historical_generation_table_reference'):
        scan(queue, target)
    assert row.read_bytes() == original and payload.read_bytes() == b'original owner bytes'


def test_closed_unchanged_tables_bind_exact_namespace_and_bytes(tmp_path):
    (tmp_path / 'processing').mkdir()
    (tmp_path / 'processing/job.json').write_text('{"run":"other"}')
    before = scan(tmp_path, Path('/unrelated/selected'))
    assert len(before) == 3
    (tmp_path / 'processing/job.json').write_text('{"run":"changed"}')
    assert scan(tmp_path, Path('/unrelated/selected')) != before


def test_table_checks_combined_descriptor_capacity_before_native_acquisition(tmp_path):
    from blueprint_pipeline.control_plane_lane_historical_references import _table
    (tmp_path / 'selected.json').write_text('{"run":"other"}')
    budget = ReferenceCollectionBudget()
    observed = []
    def bound(count):
        observed.append(count)
        if count > 2:
            raise ValueError('combined original descriptors exhausted')
    try:
        with pytest.raises(ValueError, match='table_unknown'):
            _table(tmp_path, Path('/unrelated/selected'), budget, _descriptor_check=bound)
        assert observed == [1, 2, 3]
        assert (tmp_path / 'selected.json').read_text() == '{"run":"other"}'
    finally:
        budget.close()


@pytest.mark.parametrize('kind', ['symlink', 'hardlink', 'malformed', 'malformed_uri', 'fifo'])
def test_unknown_reference_rows_are_not_absence(tmp_path, kind):
    row = tmp_path / 'job.json'
    if kind == 'symlink':
        row.symlink_to(tmp_path / 'absent')
    elif kind == 'hardlink':
        row.write_text('{}')
        os.link(row, tmp_path / 'alias.json')
    elif kind == 'fifo':
        os.mkfifo(row)
    elif kind == 'malformed_uri':
        row.write_text('{"source_uri":"file:///other/%FF/input.bin"}')
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


def test_initial_encoded_reference_refuses_held_fence_without_effect(reference_installation):
    _, target, root = reference_installation
    selected = 'file://' + ''.join('/' if byte == 47 else '%' + format(byte, '02X')
                                    for byte in os.fsencode(target / 'one.log'))
    row = root / 'queue/job.json'
    original = json.dumps({'source_uri': selected}).encode()
    row.write_bytes(original)
    with pytest.raises(ValueError, match='historical_generation_table_reference'):
        reference_call(reference_installation, lambda guard: guard())
    assert row.read_bytes() == original
    assert (target / 'one.log').read_bytes() == b'original owner diagnostics\n'


def test_unrelated_encoded_reference_does_not_select_target(tmp_path):
    queue, target, other = (tmp_path / name for name in ('queue', 'selected-diagnostics', 'other-generation'))
    queue.mkdir()
    target.mkdir()
    other.mkdir()
    payload = other / 'input.bin'
    payload.write_bytes(b'disjoint source')
    (queue / 'job.json').write_text(json.dumps({'source_uri': 'file://localhost' + str(payload).replace('-', '%2D')}))
    assert any(row[1] == 'local_uri' for row in scan(queue, target))


@pytest.mark.parametrize('alias_kind', ['directory_symlink', 'leaf_symlink', 'hardlink'])
def test_actual_local_uri_alias_is_never_reference_absence(tmp_path, alias_kind):
    queue, target = tmp_path / 'queue', tmp_path / 'selected-diagnostics'
    queue.mkdir()
    target.mkdir()
    payload = target / 'input.bin'
    payload.write_bytes(b'original selected source')
    if alias_kind == 'directory_symlink':
        alias = tmp_path / 'current-source'
        alias.symlink_to(target, target_is_directory=True)
        source = alias / payload.name
    else:
        source = tmp_path / 'current-input.bin'
        if alias_kind == 'leaf_symlink':
            source.symlink_to(payload)
        else:
            os.link(payload, source)
    row = queue / 'job.json'
    original = json.dumps({'source_uri': source.as_uri()}).encode()
    row.write_bytes(original)
    assert str(target) not in original.decode() and target.name not in original.decode()
    assert source.read_bytes() == payload.read_bytes()
    with pytest.raises(ValueError, match='historical_generation_table_unknown'):
        scan(queue, target)
    assert row.read_bytes() == original and payload.read_bytes() == b'original selected source'


@pytest.mark.parametrize('suffix', ['missing.bin', '../other/input.bin', '%00input.bin', '%FFinput.bin'])
def test_uninspectable_local_uri_is_unknown(tmp_path, suffix):
    queue, target = tmp_path / 'queue', tmp_path / 'selected-diagnostics'
    queue.mkdir()
    target.mkdir()
    (queue / 'job.json').write_text(json.dumps({'source_uri': 'file://localhost' + str(tmp_path) + '/' + suffix}))
    with pytest.raises(ValueError, match='historical_generation_table_unknown'):
        scan(queue, target)


@pytest.mark.parametrize('change', ['leaf', 'ancestor'])
def test_unchanged_queue_cannot_hide_external_local_source_drift(reference_installation, change):
    _, _, root = reference_installation
    source_root = root / 'other-source'
    source_root.mkdir()
    payload = source_root / 'input.bin'
    payload.write_bytes(b'other original source')
    row = root / 'queue/job.json'
    original = json.dumps({'source_uri': payload.as_uri()}).encode()
    row.write_bytes(original)
    def change_source(guard):
        if change == 'leaf':
            payload.write_bytes(b'changed current source')
        else:
            source_root.rename(root / 'original-source')
            source_root.mkdir()
            payload.write_bytes(b'other original source')
        guard()
    with pytest.raises(ValueError, match='historical_generation_table_unknown'):
        reference_call(reference_installation, change_source)
    assert row.read_bytes() == original


@pytest.mark.parametrize('bound', ['descriptors', 'roots'])
def test_local_uri_observation_preserves_original_limits_and_closes_fds(tmp_path, monkeypatch, bound):
    from blueprint_pipeline import control_plane_lane_historical_references as references
    queue, target, other = (tmp_path / name for name in ('queue', 'selected-diagnostics', 'other'))
    for directory in (queue, target, other):
        directory.mkdir()
    payload = other / 'input.bin'
    payload.write_bytes(b'other source')
    row = queue / 'job.json'
    original = json.dumps({'source_uri': payload.as_uri()}).encode()
    row.write_bytes(original)
    actual_open, actual_close = os.open, os.close
    opened, observations = set(), []
    def acquire(*args, **kwargs):
        fd = actual_open(*args, **kwargs)
        opened.add(fd)
        return fd
    def close(fd):
        actual_close(fd)
        opened.remove(fd)
    monkeypatch.setattr(references.os, 'open', acquire)
    monkeypatch.setattr(references.os, 'close', close)
    limit = len(queue.parts) + len(target.parts) + 1
    def capacity(count):
        observations.append((count, len(opened)))
        assert count == len(opened) + 1
        if count > limit:
            raise ValueError('original descriptor limit')
    budget = ReferenceCollectionBudget()
    deadline = None
    try:
        if bound == 'roots':
            budget.charge('roots', budget.limits['roots'] - 1)
        budget.tick()
        deadline = budget.deadline
        with pytest.raises(ValueError, match='table_unknown'):
            references._table(queue, target, budget, _descriptor_check=capacity if bound == 'descriptors' else None)
        assert not opened and budget.deadline == deadline
        if bound == 'roots':
            assert budget.counts['roots'] == budget.limits['roots']
            assert budget.failure == 'reference_roots_limit'
        else:
            assert observations[-1][0] == limit + 1
        assert row.read_bytes() == original and payload.read_bytes() == b'other source'
    finally:
        budget.close()


def test_local_source_named_ancestor_replacement_during_scan_is_unknown(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_lane_historical_references as references
    queue, target, other = (tmp_path / name for name in ('queue', 'selected-diagnostics', 'other'))
    for directory in (queue, target, other):
        directory.mkdir()
    payload = other / 'input.bin'
    payload.write_bytes(b'other source')
    row = queue / 'job.json'
    original = json.dumps({'source_uri': payload.as_uri()}).encode()
    row.write_bytes(original)
    actual_open = os.open
    def acquire(name, *args, **kwargs):
        fd = actual_open(name, *args, **kwargs)
        if name == payload.name:
            other.rename(tmp_path / 'original-other')
            other.mkdir()
            payload.write_bytes(b'other source')
        return fd
    monkeypatch.setattr(references.os, 'open', acquire)
    with pytest.raises(ValueError, match='historical_generation_(table_unknown|changed)'):
        scan(queue, target)
    assert row.read_bytes() == original


def test_production_lazy_target_parent_cannot_exceed_combined_original_fd_limit(tmp_path, monkeypatch):
    from contextlib import ExitStack
    from blueprint_pipeline.control_plane_lane_experiment_publication import _BirthFiles
    from blueprint_pipeline import control_plane_lane_historical_references as references
    target = tmp_path / 'selected-diagnostics'
    target.mkdir()
    budget = ReferenceCollectionBudget()
    actual_open, actual_close = os.open, os.close
    opened, peaks = set(), []
    def acquire(*args, **kwargs):
        fd = actual_open(*args, **kwargs)
        opened.add(fd)
        peaks.append(len(opened))
        return fd
    def close(fd):
        actual_close(fd)
        opened.remove(fd)
    monkeypatch.setattr(references.os, 'open', acquire)
    monkeypatch.setattr(references.os, 'close', close)
    files = _BirthFiles(budget)
    try:
        with ExitStack() as scope:
            for _ in range(103):
                files.open('/', os.O_RDONLY | os.O_DIRECTORY)
            for _ in range(25):
                scope.callback(os.close, os.open('/', os.O_RDONLY | os.O_DIRECTORY))
            assert len(opened) == 128
            def capacity(transient):
                budget.tick()
                if len(files.owned) + len(files.probe_owned) + transient > 128:
                    raise ValueError('combined original descriptor limit')
            with pytest.raises(ValueError):
                references._target_observation(files, target, 25, capacity)
            assert max(peaks) == 128
    finally:
        files.finish()
        budget.close()
    assert not opened


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


def test_authenticated_worker_frame_supports_every_held_reference_recheck(
        reference_installation, historical_installation):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_historical_authority as authority
    from blueprint_pipeline.control_plane_lane_historical_dispatch import _selection
    from blueprint_pipeline.control_plane_lane_historical_references import historical_reference_fence
    config, target, _ = reference_installation
    approved = decision(historical_installation, packet(historical_installation))
    with authority._session(config, authority._Operation(1030, lambda: 0)) as (files, settings, store):
        selected = _selection(files, settings, store, config, approved['action_id'], 1030)
        # The actual worker does the initial, before-effect, after-effect and
        # context-exit rechecks while retaining the SAME metadata acquisition.
        with historical_reference_fence(files, settings, target, observed_at=1030) as guard:
            guard()
            guard()
        assert _selection(files, settings, store, config, approved['action_id'], 1030) == selected
        assert files.budget.counts['roots'] > 16
    assert (target / 'one.log').read_bytes() == b'original owner diagnostics\n'


def test_historical_recheck_allowance_is_fixed_and_keeps_original_deadline():
    from blueprint_pipeline.control_plane_lane_historical_authority import _historical_reference_budget
    from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudgetError
    elapsed = [0]
    budget = _historical_reference_budget(monotonic=lambda: elapsed[0])
    generic = ReferenceCollectionBudget(monotonic=lambda: elapsed[0])
    try:
        assert generic.limits['roots'] == 16
        budget.charge('roots', 32)
        with pytest.raises(ReferenceCollectionBudgetError, match='reference_roots_limit'):
            budget.charge('roots')
    finally:
        budget.close()
        generic.close()
    timed = _historical_reference_budget(monotonic=lambda: elapsed[0])
    try:
        timed.tick()
        elapsed[0] = 5
        with pytest.raises(ReferenceCollectionBudgetError, match='reference_deadline_exceeded'):
            timed.tick()
    finally:
        timed.close()


@pytest.mark.parametrize('changed', ['queue', 'encoded_queue', 'release', 'pin'])
def test_current_reference_change_is_detected_before_effect(reference_installation, changed):
    _, target, root = reference_installation
    def change(guard):
        if changed in ('queue', 'encoded_queue'):
            selected = str(target / 'one.log')
            if changed == 'encoded_queue':
                selected = 'file://' + ''.join('/' if byte == 47 else '%' + format(byte, '02X') for byte in os.fsencode(selected))
            (root / 'queue/job.json').write_text(json.dumps({'input': selected}))
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
