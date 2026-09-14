import copy
import hashlib
import json
import shutil

import pytest

from blueprint_pipeline.control_plane_storage_pins import pin_path, write_storage_pin
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.task_evaluation_scene_storage_release import release_terminal_scene_activation_pin


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def seal(value, cross=False):
    result = copy.deepcopy(value)
    result['receipt_digest'] = (cross_runtime_canonical_digest if cross else canonical_digest)(
        result, digest_field='receipt_digest')
    return result


@pytest.fixture
def case(tmp_path):
    owner = 'scene-test-scene-configuration-activation-auto'
    root = tmp_path/'task-evaluation-launch-runs'/(owner+'-launch')
    cache = tmp_path/'launch-activations'/owner
    cache.mkdir(parents=True)
    source = cache/'bundle.zip'
    source.write_bytes(b'kept evidence bytes')
    staged = root/'immutable_inputs'/'bundle.input'
    staged.parent.mkdir(parents=True)
    staged.write_bytes(source.read_bytes())
    digest = 'sha256:' + hashlib.sha256(source.read_bytes()).hexdigest()
    pins = tmp_path/'storage-pins'
    write_storage_pin(pins_root=pins, kind='activation', owner_id=owner, paths=[cache])
    queue = tmp_path/'task-evaluation-launches'
    for state in ('pending', 'processing'):
        (queue/state).mkdir(parents=True)
    staging = seal({'schema_version': 'task_evaluation_immutable_input_staging.v1',
        'status': 'staged', 'profile_id': 'profile', 'profile_digest': 'sha256:'+'a'*64,
        'inputs': [{'source_path': str(source), 'staged_path': str(staged),
                    'expected_digest': digest, 'staged_digest': digest,
                    'staged_size_bytes': staged.stat().st_size}]})
    write(root/'immutable_input_staging_receipt.json', staging)
    receipt = seal({'schema_version': 'task_evaluation_launch_receipt.v1', 'status': 'blocked',
        'launch_id': root.name, 'run_id': root.name, 'execute_requested': True,
        'launch_profile_digest': staging['profile_digest'],
        'immutable_input_staging': {'receipt_digest': staging['receipt_digest']},
        'provider_mutation_attempted': True}, cross=True)
    provider = root/'allocator/scene-configuration-job/vast_provider_run'
    teardown = {'schema_version': 'vast_teardown_manifest.v1', 'status': 'completed',
        'vast_instance_ids': [123], 'runner_gpu_teardown_completed': True,
        'continuing_spend_from_this_run': False,
        'teardown_actions_performed': [{'instance_id': 123, 'action': 'destroy_instance',
            'status': 'completed', 'http_status_code': 200}]}
    write(provider/'vast_teardown_manifest.json', teardown)
    write(provider/'vast_budget_ledger.json', {'schema_version': 'vast_budget_ledger.v1',
        'vast_instance_ids': [123], 'continuing_spend_from_this_run': False})
    return dict(run_root=root, receipt=receipt, pins_root=pins, queue_root=queue,
                cache=cache, source=source, staged=staged, staging=staging, teardown=teardown, provider=provider, owner=owner)


def release(case):
    return release_terminal_scene_activation_pin(**{k:case[k] for k in ('run_root','receipt','pins_root','queue_root')})


def pin(case):
    return json.loads(pin_path(case['pins_root'], 'activation', case['owner']).read_text())


@pytest.mark.parametrize('status', ['blocked', 'completed'])
def test_closed_launch_releases_pin_but_never_removes_data(case, status):
    case['receipt']['status'] = status
    case['receipt'] = seal(case['receipt'], cross=True)
    assert release(case)['status'] == 'released'
    assert pin(case)['released_at_epoch'] is not None
    assert case['source'].read_bytes() == case['staged'].read_bytes() == b'kept evidence bytes'
    assert release(case)['status'] == 'already_released'


@pytest.mark.parametrize('mutation', ['dry', 'receipt_tamper', 'staging_tamper', 'wrong_pin',
                                    'missing_copy', 'same_size_corruption', 'symlinked_copy_parent', 'symlinked_pin_target',
                                    'warm', 'wrong_instance',
                                    'teardown_failure', 'other_pending', 'missing_queue'])
def test_uncertain_or_shared_launch_keeps_pin(case, mutation):
    if mutation == 'dry':
        case['receipt']['execute_requested'] = False
        case['receipt'] = seal(case['receipt'], cross=True)
    elif mutation == 'receipt_tamper':
        case['receipt']['run_id'] = 'other'
    elif mutation == 'staging_tamper':
        case['staging']['profile_digest'] = 'sha256:'+'c'*64
        write(case['run_root']/'immutable_input_staging_receipt.json', case['staging'])
    elif mutation == 'wrong_pin':
        p = pin(case)
        p['paths'] = [str(case['cache'].parent/'other')]
        write(pin_path(case['pins_root'], 'activation', case['owner']), p)
    elif mutation == 'missing_copy':
        case['staged'].unlink()
    elif mutation == 'same_size_corruption':
        case['staged'].write_bytes(b'X' * case['staged'].stat().st_size)
    elif mutation == 'symlinked_pin_target':
        original = case['cache']
        moved = original.with_name('moved-cache')
        original.rename(moved)
        original.symlink_to(moved, target_is_directory=True)
    elif mutation == 'symlinked_copy_parent':
        original = case['staged'].parent
        moved = original.with_name('moved-inputs')
        original.rename(moved)
        original.symlink_to(moved, target_is_directory=True)
    elif mutation == 'other_pending':
        write(case['queue_root']/'pending/another.json', {'launch_id':'another', 'launch_profile_id':'profile'})
    elif mutation == 'missing_queue':
        case['queue_root'] = case['queue_root'].with_name('absent-queue')
    else:
        t = case['teardown']
        if mutation == 'warm':
            t['continuing_spend_from_this_run'] = True
        elif mutation == 'wrong_instance':
            t['vast_instance_ids'] = [999]
        else:
            t['teardown_actions_performed'][0]['status'] = 'failed'
        write(case['provider']/'vast_teardown_manifest.json', t)
    assert release(case)['status'] != 'released'
    assert pin(case)['released_at_epoch'] is None


def test_same_processing_request_does_not_keep_its_own_finished_cache(case):
    write(case['queue_root']/'processing/current.json', {'launch_id':case['run_root'].name,
                                                      'launch_profile_id':'profile'})
    assert release(case)['status'] == 'released'


def test_pre_admission_refusal_can_release_without_provider_receipts(case):
    case['receipt']['provider_mutation_attempted'] = False
    case['receipt']['provider_mutation_evidence'] = {'status':'absent_before_paid_admission'}
    case['receipt'] = seal(case['receipt'], cross=True)
    shutil.rmtree(case['provider'])
    assert release(case)['status'] == 'released'


def test_absence_claim_cannot_override_recorded_provider_evidence(case):
    case['receipt']['provider_mutation_attempted'] = False
    case['receipt']['provider_mutation_evidence'] = {'status':'absent_before_paid_admission'}
    case['receipt'] = seal(case['receipt'], cross=True)
    assert release(case)['status'] == 'retained'
    assert pin(case)['released_at_epoch'] is None
