"""Real adapter refusal/teardown behavior inside both production sealed environments."""
from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from pathlib import Path

import pytest

from blueprint_pipeline import adp_retained_scene_render_vast as render_lane
from blueprint_pipeline import task_evaluation_scene_configuration_vast as scene_lane
from blueprint_pipeline import vast_provider_adapter as adapter
from tests.test_vast_provider_adapter import _configure_live_gates, _created_instance_detail, _paid_grant


@pytest.mark.parametrize('lane', [render_lane, scene_lane], ids=['source_render', 'scene_construction'])
@pytest.mark.parametrize('case', ['definite_refusal', 'refusal_limit', 'empty_unverified', 'empty_zero',
                                  'empty_created', 'definite_unverified', 'definite_created',
                                  'malformed_inventory', 'prior_label_created', 'ambiguous_success', 'timeout',
                                  'unknown_status_created', 'stopped_created', 'malformed_map_inventory', 'malformed_inventory_created',
                                  'inventory_valueerror'])
def test_sealed_lane_never_reselects_after_ambiguous_or_observed_creation(tmp_path: Path, monkeypatch, lane, case):
    secret = _configure_live_gates(tmp_path, monkeypatch)
    creates, labels, destroyed, consumptions = [], [], [], []
    live, maximum_live = set(), [0]
    clock = [adapter.time.time()]
    monkeypatch.setattr(adapter.time, 'time', lambda: clock[0])
    monkeypatch.setattr(adapter.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    offers = [{'id': n, 'ask_contract_id': n, 'gpu_name': 'RTX A6000', 'gpu_ram': 49152,
               'dph_total': 0.25 + i/100, 'driver_version': '580.159.03', 'machine_id': 9400+n,
               'num_gpus': 1, 'rentable': True} for i, n in enumerate([401, 402, 403])]

    def api(*, method, path, api_key, payload=None, timeout_seconds=30):
        assert api_key == secret
        if method == 'GET' and path == '/instances/':
            if not creates:
                return 200, {'instances': []}
            if case in ['empty_unverified', 'definite_unverified']:
                raise urllib.error.HTTPError('https://vast.invalid/instances/', 503, 'unavailable', {}, BytesIO(b''))
            if case == 'inventory_valueerror':
                raise ValueError(secret)
            if case == 'malformed_inventory':
                return 200, {}
            if case == 'malformed_map_inventory':
                return 200, {'instances': {'unknown': 'unparsed'}}
            if case == 'malformed_inventory_created':
                return 200, {'instances': [{'id': 4010, 'label': labels[0]}, 'unparsed']}
            if live:
                return 200, {'instances': [{'id': n, 'label': labels[0],
                                         **({} if case == 'unknown_status_created' else
                                            {'actual_status': 'exited' if case == 'stopped_created' else 'running'})}
                                        for n in live]}
            return 200, {'instances': []}
        if method == 'POST' and path == '/bundles/':
            return 200, {'offers': [] if 'id' in payload else offers}
        if method == 'PUT' and path.startswith('/asks/'):
            creates.append(path)
            labels.append(payload['label'])
            if case == 'timeout':
                raise TimeoutError('create response unavailable')
            if case == 'ambiguous_success':
                return 200, {'success': True}
            if (case in ['empty_created', 'definite_created', 'unknown_status_created', 'stopped_created', 'malformed_inventory_created']
                    or (case == 'prior_label_created' and len(creates) == 2)):
                live.add(4010)
                maximum_live[0] = max(maximum_live[0], len(live))
            if (case == 'definite_refusal' and len(creates) == 2):
                live.add(4020)
                maximum_live[0] = max(maximum_live[0], len(live))
                return 200, {'success': True, 'new_contract': 4020}
            body = b'' if case.startswith('empty_') else b'{"success":false,"msg":"no_such_ask"}'
            raise urllib.error.HTTPError('https://vast.invalid'+path, 400, 'bad request', {}, BytesIO(body))
        if method == 'GET' and path == '/instances/4020/':
            return 200, _created_instance_detail(dph_total=0.26)
        if method == 'PUT' and path == '/instances/request_logs/4020':
            return 200, {'success': True, 'result_url': 'https://logs.invalid/fake'}
        if method == 'DELETE' and path.startswith('/instances/'):
            identifier = int(path.strip('/').split('/')[-1])
            destroyed.append(identifier)
            live.remove(identifier)
            return 200, {'success': True}
        raise AssertionError((method, path))

    def consume():
        consumptions.append('consumed')
        return {'status': 'consumed'}

    monkeypatch.setattr(adapter, '_api_json', api)
    monkeypatch.setattr(adapter, '_fetch_text', lambda *_args, **_kwargs:
                        'BLUEPRINT_VAST_HEARTBEAT_OK\nRTX A6000, 580.159.03, 49140 MiB\nBLUEPRINT_VAST_GPU_SANITY_OK\n')
    with lane._authority_environment():
        result = adapter.run_vast_provider_adapter(
            job_dir=tmp_path, mode='live-startup-probe', paid_resource_admission_grant=_paid_grant(),
            allow_vast_api_call=True, allow_instance_launch=True, poll_interval_seconds=0,
            startup_timeout_seconds=20, session_max_live_minutes=None, pre_provider_mutation_hook=consume)
    expected_creates = 3 if case == 'refusal_limit' else 2 if case in ['definite_refusal', 'prior_label_created'] else 1
    assert len(creates) == expected_creates
    assert consumptions == ['consumed']
    assert secret not in json.dumps(result)
    assert maximum_live[0] <= 1 and not live
    assert result['status'] == ('completed' if case == 'definite_refusal' else 'failed')
    if case in ['empty_created', 'definite_created', 'prior_label_created', 'unknown_status_created', 'stopped_created', 'malformed_inventory_created']:
        assert destroyed == [4010]
        diagnosis = result['create_failure_diagnosis']
        assert diagnosis['matching_attempt_instance_ids'] == [4010]
        assert diagnosis['create_produced_no_instance'] is False
        if case == 'prior_label_created':
            assert len(set(labels)) == 2
            assert diagnosis['attempted_labels'] == sorted(set(labels))
    if case == 'empty_zero':
        diagnosis = result['create_failure_diagnosis']
        assert diagnosis['create_inventory_verified'] is True
        assert diagnosis['create_produced_no_instance'] is True
        assert diagnosis['definite_create_refusal'] is False
    if case == 'definite_refusal':
        manifest = json.loads((tmp_path/'vast_offer_selection_manifest.json').read_text())
        assert len(manifest['create_retry_attempts']) == 1
        assert destroyed == [4020]
