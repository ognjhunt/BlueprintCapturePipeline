"""Authentic root-selected readable bootstrap; no private bytes are exported."""
import hashlib
import json
import os

import pytest

from tests.test_registered_experiment_issuer import installation, issue  # noqa: F401
from tests.test_registered_experiment_birth import prepare, birth
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_native_g1_development_pair import _paired_requests


@pytest.fixture
def bootstrap_installation(installation, tmp_path, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_birth as native
    prepare(installation)
    request_root = installation[0].parent / 'producer-requests'
    request_root.mkdir(mode=0o700)
    paths, _ = _paired_requests(request_root)
    for path in paths:
        path.chmod(0o640)
    selectors = tuple((path, {'sha256': 'sha256:'+hashlib.sha256(path.read_bytes()).hexdigest(),
                              'size_bytes': path.stat().st_size}) for path in paths)
    grant = issue(installation, participant_profile='g1_local_contained_completed.v1', request_records=selectors)
    monkeypatch.setattr(native, '_blueprint_identity', lambda: (0, 0))
    monkeypatch.setattr(os, 'fchown', lambda *a: None)
    born = birth(installation, grant)
    return installation, grant, born, paths


def publish(value):
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root
    value_installation, grant, _, paths = value
    return root.issue_experiment_producer_bootstrap(grant['intent_id'],
        expected_intent_sha256=grant['intent']['sha256'], expected_intent_size_bytes=grant['intent']['size_bytes'],
        request_paths=paths, installed_config_path=value_installation[0], now=lambda: 1100)


def test_actual_root_bootstrap_is_readable_current_bound_and_immutable(bootstrap_installation):
    value, grant, born, paths = bootstrap_installation
    selected = publish(bootstrap_installation)
    public = value[2].parents[1]/'experiment-authority'
    path = public/(grant['intent_id']+'.producer-bootstrap.json')
    raw = path.read_bytes()
    record = json.loads(raw)
    assert selected == {'sha256': 'sha256:'+hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}
    assert path.stat().st_mode & 0o777 == 0o640
    assert record['generation'] == born['generation'] and record['birth'] == born['birth']
    assert record['intent'] == grant['intent']
    assert [row['path'] for row in record['requests']] == [str(p) for p in paths]
    assert record['participant_profile'] == 'g1_local_contained_completed.v1'
    assert not any(key in record for key in ('principal', 'policy', 'config', 'private_store'))
    with pytest.raises(ValueError, match='experiment_'):
        publish(bootstrap_installation)
    assert path.read_bytes() == raw


def test_actual_bootstrap_refuses_changed_authorized_request_before_publication(bootstrap_installation):
    _, grant, _, paths = bootstrap_installation
    paths[0].write_text('{}')
    with pytest.raises(ValueError, match='experiment_|owner_'):
        publish(bootstrap_installation)
    public = bootstrap_installation[0][2].parents[1]/'experiment-authority'
    assert not (public/(grant['intent_id']+'.producer-bootstrap.json')).exists()
