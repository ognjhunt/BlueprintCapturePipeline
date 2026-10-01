"""ADP-009D/day28: fixed ordinary capacity request; no completion authority."""
# Covers: src/blueprint_pipeline/control_plane_lane_disk_diagnostic.py
from copy import deepcopy
import hashlib
import json
from pathlib import Path

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
    assert set(request['installed_sources']) == producer.SOURCE_MODULES | {'operator_door.__init__', 'operator_door.config'}
    assert request['request_digest'] == canonical_digest(request, digest_field='request_digest')
    assert {p.name: p.read_bytes() for p in installation[2].iterdir()} == before
    assert not any(p.name.startswith('registered-') for p in installation[0].parent.rglob('*'))


@pytest.mark.parametrize('changed_source', ['config.py', '__init__.py', 'consumer',
                                          'control_plane_storage_pins', 'control_plane_disk_budget',
                                          'control_plane_lane_historical_restore_limits'])
def test_compatible_installed_acquisition_source_change_invalidates_request(installation, monkeypatch, changed_source):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_disk_diagnostic as producer
    from blueprint_pipeline import control_plane_lane_owner_consents as owners
    from blueprint_pipeline.control_plane_lane_experiment_publication import _BirthFiles
    from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
    from blueprint_pipeline.control_plane_lane_experiment_retirement import _configuration
    prepare(installation)
    if changed_source == 'consumer' or changed_source.startswith('control_plane_'):
        # Disposable protected source mirror, no mutation of the shared checkout.
        mirror = installation[0].parent / 'producer-source'
        mirror.mkdir(mode=0o700)
        source = Path(producer.__file__).parent
        for name in producer.SOURCE_MODULES | {'control_plane_lane_experiment_consumer'}:
            target = mirror / (name + '.py')
            target.write_bytes((source / (name + '.py')).read_bytes())
            target.chmod(0o600)
        monkeypatch.setattr(producer, '__file__', str(mirror / 'control_plane_lane_disk_diagnostic.py'))
        changed = mirror / ((changed_source if changed_source != 'consumer'
                             else 'control_plane_lane_experiment_consumer') + '.py')
    else:
        changed = owners.INSTALLED_PACKAGE_ROOT / 'operator_door' / changed_source
    request = _request(installation)
    changed.write_bytes(changed.read_bytes() + b'\n# compatible bytes still change the installed source identity\n')
    files = _BirthFiles(ReferenceCollectionBudget(values_limit=10000))
    try:
        config = _configuration(files, installation[0])
        with pytest.raises(ValueError, match='diagnostic_request_changed'):
            producer.validate_request(files, json.dumps(request).encode(), config=config,
                installed_config_path=installation[0], run_ref='run1')
    finally:
        files.finish()
        files.budget.close()


def test_same_admission_retains_original_source_fds_and_rejects_later_source_rewrite(installation, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_disk_diagnostic as producer
    from blueprint_pipeline.control_plane_lane_experiment_publication import _BirthFiles
    from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
    mirror = installation[0].parent / 'original-source-readback'
    mirror.mkdir(mode=0o700)
    source = Path(producer.__file__).parent
    for name in producer.SOURCE_MODULES:
        target = mirror / (name + '.py')
        target.write_bytes((source / (name + '.py')).read_bytes())
        target.chmod(0o600)
    monkeypatch.setattr(producer, '__file__', str(mirror / 'control_plane_lane_disk_diagnostic.py'))
    files = _BirthFiles(ReferenceCollectionBudget())
    try:
        selected = producer._sources(files)
        before = dict(files.owned), dict(files.budget.counts)
        assert producer._sources(files) == selected
        assert dict(files.owned) == before[0]
        assert files.budget.counts['raw_bytes'] == before[1]['raw_bytes']
        changed = mirror / 'control_plane_lane_disk_diagnostic.py'
        changed.write_bytes(changed.read_bytes() + b'\n# actual disposable source mutation\n')
        with pytest.raises(ValueError):
            producer._sources(files)
        assert dict(files.owned) == before[0]
    finally:
        files.finish()
        files.budget.close()


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
