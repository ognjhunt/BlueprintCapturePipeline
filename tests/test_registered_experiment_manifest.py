"""Full compact decode accounting only; these supplied records grant no action."""
import pytest

from blueprint_pipeline import control_plane_lane_experiment_actions as actions
from blueprint_pipeline.control_plane_lane_experiment_publication import _BirthFiles
from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget


def document(count=4096):
    reference = {'sha256': 'sha256:' + 'a'*64, 'size_bytes': 128}
    binding = dict(generation='b'*32, birth=reference, target_identity={'dev': 1, 'ino': 2, 'type': 'directory'},
                   lease=reference, completion=reference)
    value = dict(schema_version=actions.MANIFEST_SCHEMA, **binding,
                 members=[[f'f-{index:04d}', 'file', f'1:{index+3}:r', '384:501:20:1:0:1:1', 'sha256:'+'c'*64]
                          for index in range(count)], logical_bytes=0, allocated_bytes=count*4096)
    return binding, value


def encoded(value):
    return actions._encoded(value, 'manifest_digest', 1048576)


def test_actual_full_4096_manifest_decoder_conserves_native_and_typed_value_allowance():
    binding, value = document()
    budget = ReferenceCollectionBudget(values_limit=100000)
    files = _BirthFiles(budget)
    try:
        result = actions._manifest_record(files, encoded(value), binding)
        assert len(result['members']) == 4096
        assert 94000 < budget.counts['values'] < 100000
        assert budget.failure is None
    finally:
        files.finish()
        budget.close()


@pytest.mark.parametrize('change', ['duplicate', 'path', 'kind', 'identity', 'mode', 'link', 'size', 'hash', 'binding', 'extra'])
def test_root_selected_compact_manifest_invalid_shape_refuses(change):
    binding, value = document(2)
    row = value['members'][0]
    if change == 'duplicate':
        value['members'][1][0] = row[0]
    elif change == 'path':
        row[0] = '../escape'
    elif change == 'kind':
        row[1] = 'directory'
    elif change == 'identity':
        row[2] = '1:+3:r'
    elif change == 'mode':
        row[3] = '33152:501:20:1:0:1:1'
    elif change == 'link':
        row[3] = '384:501:20:2:0:1:1'
    elif change == 'size':
        row[3] = '384:501:20:1:1:1:1'
    elif change == 'hash':
        row[4] = None
    elif change == 'binding':
        value['generation'] = 'd'*32
    else:
        value['permission'] = 'delete'
    files = _BirthFiles(ReferenceCollectionBudget(values_limit=100000))
    try:
        with pytest.raises(ValueError, match='experiment_manifest'):
            actions._manifest_record(files, encoded(value), binding)
    finally:
        files.finish()
        files.budget.close()
