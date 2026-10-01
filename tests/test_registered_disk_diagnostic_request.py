"""ADP-009D/day28: fixed ordinary capacity request; no completion authority."""
# Covers: src/blueprint_pipeline/control_plane_lane_disk_diagnostic.py
from copy import deepcopy
import hashlib
import json

import pytest

from tests.test_registered_experiment_issuer import installation  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_registered_experiment_birth import prepare


def _request(installation):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_disk_diagnostic as producer
    return producer.build_request(installed_config_path=installation[0], run_ref='run1')


def test_request_binds_actual_installed_configuration_and_fixed_producer_sources(installation):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_disk_diagnostic as producer
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    prepare(installation)
    before = {p.name: p.read_bytes() for p in installation[2].iterdir()}
    request = _request(installation)
    assert request['schema_version'] == 'control_plane_lane_disk_diagnostic_request.v1'
    assert request['run_ref'] == 'run1'
    raw = installation[0].read_bytes()
    assert request['config'] == {'sha256': 'sha256:' + hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}
    assert request['roots'] == {key: installation[1]['lane_scratch_' + key + '_root'] for key in ('work', 'inputs')}
    assert set(request['installed_sources']) == producer.SOURCE_MODULES
    assert request['request_digest'] == canonical_digest(request, digest_field='request_digest')
    assert {p.name: p.read_bytes() for p in installation[2].iterdir()} == before
    assert not any(p.name.startswith('registered-') for p in installation[0].parent.rglob('*'))


@pytest.mark.parametrize('change', ['config', 'root', 'root_identity', 'boolean_identity',
                                   'source', 'run', 'extra', 'seal'])
def test_request_semantic_drift_refuses_before_any_payload_birth(installation, change):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_disk_diagnostic as producer
    from blueprint_pipeline.control_plane_lane_experiment_publication import _BirthFiles
    from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
    from blueprint_pipeline.control_plane_lane_experiment_retirement import _configuration
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    prepare(installation)
    request = deepcopy(_request(installation))
    if change == 'config':
        request['config']['sha256'] = 'sha256:' + 'f' * 64
    elif change == 'root':
        request['roots']['work'] = '/arbitrary/root'
    elif change == 'root_identity':
        request['root_identities']['work']['ino'] += 1
    elif change == 'boolean_identity':
        request['root_identities']['work']['dev'] = True
    elif change == 'source':
        request['installed_sources'][next(iter(producer.SOURCE_MODULES))] = 'sha256:' + 'f' * 64
    elif change == 'run':
        request['run_ref'] = 'unselected-run'
    elif change == 'extra':
        request['command'] = 'arbitrary-command'
    request['request_digest'] = canonical_digest(request, digest_field='request_digest')
    if change == 'seal':
        request['request_digest'] = 'sha256:' + 'f' * 64
    files = _BirthFiles(ReferenceCollectionBudget(values_limit=10000))
    try:
        config = _configuration(files, installation[0])
        with pytest.raises(ValueError, match='diagnostic_request_'):
            producer.validate_request(files, json.dumps(request).encode(), config=config,
                installed_config_path=installation[0], run_ref='run1')
    finally:
        files.finish()
        files.budget.close()
    assert not any(p.name.startswith('registered-') for p in installation[0].parent.rglob('*'))
