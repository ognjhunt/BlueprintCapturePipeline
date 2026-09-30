"""ADP-009D/day28: pre-write history is not a birth or permission to adopt files.

Parser classification only. The worker must separately prove the unchanged
physical tombstone and absent snapshot under the original approved operation.
"""
# Covers: src/blueprint_pipeline/control_plane_lane_historical_restore_worker.py
import pytest


ACTION = 'a' * 32


def events():
    return [dict(kind='intent', body={}),
        dict(kind='restore_intent', body=dict(phase='reservation')),
        dict(kind='restore_intent', body=dict(phase='directory', path='',
             stage_path='.historical-restore-' + ACTION))]


def test_only_original_pre_write_steps_can_be_classified_as_unwritten():
    from blueprint_pipeline.control_plane_lane_historical_restore_worker import _unwritten_restore_attempt
    original = events()
    assert _unwritten_restore_attempt(original, ACTION) is True
    # Repeated pre-write attempts retain their original intent, not fresh ones.
    assert _unwritten_restore_attempt(original + original[1:], ACTION) is True


@pytest.mark.parametrize('change', ['birth', 'member', 'child_directory', 'foreign_stage',
    'publication', 'owner_rights', 'final', 'second_intent'])
def test_any_possible_write_or_unknown_stage_requires_full_recovery(change):
    from blueprint_pipeline.control_plane_lane_historical_restore_worker import _unwritten_restore_attempt
    current = events()
    if change == 'birth':
        current.append(dict(kind='restore_directory', body=dict(path='')))
    elif change == 'member':
        current.append(dict(kind='restore_member', body=dict(path='one')))
    elif change == 'child_directory':
        current[-1]['body']['path'] = 'nested'
    elif change == 'foreign_stage':
        current[-1]['body']['stage_path'] = '.historical-restore-' + 'b' * 32
    elif change in ('publication', 'owner_rights'):
        current.append(dict(kind='restore_intent', body=dict(phase=change)))
    elif change == 'final':
        current.append(dict(kind='restore_final', body={}))
    else:
        current.append(dict(kind='intent', body={}))
    assert _unwritten_restore_attempt(current, ACTION) is False
