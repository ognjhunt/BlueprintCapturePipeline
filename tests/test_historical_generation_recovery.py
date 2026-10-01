"""Recovery must bind original bytes and only journaled ownership changes."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_historical_recovery.py
import copy
import stat

import pytest


def fixture():
    rows = [dict(path='', kind='directory', version=[1, 2, stat.S_IFDIR | 0o700, 65534, 65534,
            2, 4096, 100, 100, 8], sha256=None, size_bytes=0),
        dict(path='one', kind='file', version=[1, 3, stat.S_IFREG | 0o640, 65534, 65534,
            1, 4, 100, 100, 8], sha256='sha256:' + 'a' * 64, size_bytes=4)]
    manifest = dict(target_path='/var/work/one', parent_path='/var/work', root_version=[1] * 10,
                    member_count=2, members=rows)
    fenced = list(rows[0]['version'])
    fenced[3:5], fenced[8] = [0, 0], 101
    events = [dict(kind='intent', body={}),
        dict(kind='fence_intent', body=dict(path='', version=rows[0]['version'], uid=0, gid=0, mode=0o700)),
        dict(kind='fenced', body=dict(path='', version=fenced))]
    observed = copy.deepcopy(manifest)
    observed['members'][0]['version'] = fenced
    return manifest, events, observed


def test_resume_exact_logged_root_fence_preserves_original_payload_binding():
    from blueprint_pipeline.control_plane_lane_historical_recovery import recover_fence
    manifest, events, observed = fixture()
    result = recover_fence(manifest, events, observed)
    assert result['completed'] == {''}
    assert result['manifest'] == observed
    assert result['manifest']['members'][1]['sha256'] == manifest['members'][1]['sha256']


@pytest.mark.parametrize('phase', ['unchanged', 'chown', 'chmod'])
def test_resume_last_fence_intent_accepts_only_exact_own_rights_transition(phase):
    from blueprint_pipeline.control_plane_lane_historical_recovery import recover_fence
    manifest, events, observed = fixture()
    row = manifest['members'][1]
    events.append(dict(kind='fence_intent', body=dict(path='one', version=row['version'], uid=0, gid=0, mode=0o600)))
    if phase != 'unchanged':
        observed['members'][1]['version'][3:5] = [0, 0]
        observed['members'][1]['version'][8] = 101
    if phase == 'chmod':
        observed['members'][1]['version'][2] = stat.S_IFREG | 0o600
    result = recover_fence(manifest, events, observed)
    assert result['completed'] == {''}
    assert result['pending'] == 'one'


@pytest.mark.parametrize('drift', ['bytes', 'inode', 'mtime', 'blocks', 'unlogged', 'parent', 'missing', 'extra', 'mode'])
def test_recovery_refuses_payload_namespace_and_unlogged_rights_changes(drift):
    from blueprint_pipeline.control_plane_lane_historical_recovery import recover_fence
    manifest, events, observed = fixture()
    if drift == 'bytes':
        observed['members'][1]['sha256'] = 'sha256:' + 'b' * 64
    elif drift == 'inode':
        observed['members'][0]['version'][1] += 1
    elif drift == 'mtime':
        observed['members'][0]['version'][7] += 1
    elif drift == 'blocks':
        observed['members'][0]['version'][9] += 1
    elif drift == 'unlogged':
        observed['members'][1]['version'][3:5] = [0, 0]
    elif drift == 'parent':
        observed['root_version'][1] += 1
    elif drift == 'missing':
        observed['members'].pop()
    elif drift == 'extra':
        observed['members'].append(dict(observed['members'][1], path='extra'))
    elif drift == 'mode':
        observed['members'][0]['version'][2] |= 0o020
    with pytest.raises(ValueError, match='recovery_changed'):
        recover_fence(manifest, events, observed)


def test_pending_intent_does_not_adopt_chmod_without_root_ownership():
    from blueprint_pipeline.control_plane_lane_historical_recovery import recover_fence
    manifest, events, observed = fixture()
    row = manifest['members'][1]
    events.append(dict(kind='fence_intent', body=dict(path='one', version=row['version'], uid=0, gid=0, mode=0o600)))
    observed['members'][1]['version'][2] = stat.S_IFREG | 0o600
    observed['members'][1]['version'][8] += 1
    with pytest.raises(ValueError, match='recovery_changed'):
        recover_fence(manifest, events, observed)


@pytest.mark.parametrize('kind', ['removed', 'removal_intent', 'final', 'fenced'])
def test_fence_resume_cannot_adopt_unsupported_or_out_of_order_events(kind):
    from blueprint_pipeline.control_plane_lane_historical_recovery import recover_fence
    manifest, events, observed = fixture()
    events.append(dict(kind=kind, body={}))
    with pytest.raises(ValueError, match='recovery_changed'):
        recover_fence(manifest, events, observed)


def removal_fixture():
    manifest, events, observed = fixture()
    row = manifest['members'][1]
    current = list(row['version'])
    current[2:5], current[8] = [stat.S_IFREG | 0o600, 0, 0], 101
    events.extend([dict(kind='fence_intent', body=dict(path='one', version=row['version'], uid=0, gid=0, mode=0o600)),
        dict(kind='fenced', body=dict(path='one', version=current)),
        dict(kind='removal_intent', body=dict(path='one', kind='file', version=current,
            sha256=row['sha256'], size_bytes=row['size_bytes']))])
    observed['members'][1]['version'] = current
    return manifest, events, observed


def removed_parent(observed):
    result = list(observed['members'][0]['version'])
    result[7:9] = [102, 102]
    return result


def test_pending_removal_keeps_present_original_and_reuses_intent():
    from blueprint_pipeline.control_plane_lane_historical_recovery import recover_action
    manifest, events, observed = removal_fixture()
    result = recover_action(manifest, events, observed)
    assert result['pending_removal'] == 'one'
    assert result['removed'] == result['uncertain'] == set()
    assert result['manifest'] == observed


def test_unlogged_absence_is_reconciled_with_zero_removal_credit():
    from blueprint_pipeline.control_plane_lane_historical_recovery import recover_action
    manifest, events, observed = removal_fixture()
    observed['members'].pop()
    observed['member_count'] = 1
    observed['members'][0]['version'] = removed_parent(observed)
    result = recover_action(manifest, events, observed)
    assert result['uncertain'] == {'one'} and not result['removed']
    assert result['pending_removal'] is None
    assert result['reconcile']['observed_removed_allocated_bytes'] == 0
    assert result['reconcile']['parent_version'] == observed['members'][0]['version']


def test_only_exact_logged_removal_retains_observed_credit():
    from blueprint_pipeline.control_plane_lane_historical_recovery import recover_action
    manifest, events, observed = removal_fixture()
    parent = removed_parent(observed)
    events.append(dict(kind='removed', body=dict(path='one', kind='file', physical_identity=[1, 3],
        logical_bytes=4, observed_removed_allocated_bytes=4096, parent_path='', parent_version=parent)))
    observed['members'].pop()
    observed['member_count'] = 1
    observed['members'][0]['version'] = parent
    result = recover_action(manifest, events, observed)
    assert result['removed'] == {'one'} and not result['uncertain']
    assert result['reconcile'] is None


@pytest.mark.parametrize('drift', ['extra', 'unplanned-absence', 'parent-inode', 'parent-mode', 'rewritten', 'credit'])
def test_interrupted_removal_refuses_unaccounted_effects_and_bytes(drift):
    from blueprint_pipeline.control_plane_lane_historical_recovery import recover_action
    manifest, events, observed = removal_fixture()
    if drift == 'extra':
        observed['members'].append(dict(observed['members'][1], path='extra'))
        observed['member_count'] += 1
    elif drift == 'unplanned-absence':
        events.pop()
        observed['members'].pop()
        observed['member_count'] = 1
    elif drift.startswith('parent-'):
        observed['members'].pop()
        observed['member_count'] = 1
        observed['members'][0]['version'][1 if drift == 'parent-inode' else 2] += 1
    elif drift == 'rewritten':
        observed['members'][1]['sha256'] = 'sha256:' + 'b' * 64
    elif drift == 'credit':
        events.append(dict(kind='removed', body=dict(path='one', kind='file', physical_identity=[1, 3],
            logical_bytes=4, observed_removed_allocated_bytes=99999, parent_path='', parent_version=removed_parent(observed))))
    with pytest.raises(ValueError, match='recovery_changed'):
        recover_action(manifest, events, observed)
