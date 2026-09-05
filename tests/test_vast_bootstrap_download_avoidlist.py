"""Provider bootstrap timeouts are learned without misclassifying other evidence gaps."""
from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline import vast_provider_adapter as adapter
from blueprint_pipeline.provider_machine_avoidlist import avoidlist_machine_ids

TIMEOUT_BLOCKERS = ['provider_remote_blocker:download_failed:28',
                    'provider_bundle_download_marker_missing', 'provider_entrypoint_start_marker_missing']
REASON = 'vast_provider_bootstrap_input_download_transport_timeout'


def test_observed_bootstrap_timeout_is_persisted_and_excluded_from_next_selection(tmp_path: Path):
    path = tmp_path/'avoidlist.json'
    path.write_text('{"machine_ids":[31726],"entries":[]}')
    assert adapter._machine_avoidlist_reason(TIMEOUT_BLOCKERS) == REASON
    offer = {'machine_id': 20166, 'ask_contract_id': 49588631, 'gpu_name': 'RTX 4090',
             'driver_version': '580.126.09'}
    result = adapter._record_machine_avoidlist_entry(
        path=path, generated_at='2026-09-05T19:00:00Z', selected_offer=offer,
        instance_id=49987963, blockers=TIMEOUT_BLOCKERS, reason=REASON)
    assert avoidlist_machine_ids(path) == {20166, 31726}
    assert result['entries'][-1]['retry_policy'] == 'exclude_persistently_across_sibling_jobs_until_manual_review'
    assert result['entries'][-1]['instance_id'] == 49987963
    candidates = [{'id': i, 'ask_contract_id': i, 'machine_id': m, 'gpu_name': 'RTX A6000',
                   'gpu_ram': 49152, 'dph_total': .25, 'driver_version': '580.159.03',
                   'num_gpus': 1, 'rentable': True} for i, m in [(1, 20166), (2, 99999)]]
    selected = adapter._select_offer(candidates, max_hourly_rate=1., min_gpu_ram_mb=16000,
                                     excluded_machine_ids=avoidlist_machine_ids(path))
    assert selected['machine_id'] == 99999


@pytest.mark.parametrize('blockers', [
    ['provider_output_get_url_download_failed'],
    ['provider_remote_blocker:download_failed:22', *TIMEOUT_BLOCKERS[1:]],
    ['provider_remote_blocker:download_failed:60', *TIMEOUT_BLOCKERS[1:]],
    [TIMEOUT_BLOCKERS[0]],
    [TIMEOUT_BLOCKERS[0], TIMEOUT_BLOCKERS[1]],
    ['provider_runtime_result_missing_from_output_zip'],
])
def test_unrelated_output_auth_certificate_and_runtime_failures_do_not_blame_machine(blockers):
    assert adapter._machine_avoidlist_reason(blockers) is None
