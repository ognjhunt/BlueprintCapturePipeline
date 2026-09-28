"""Actual G1 pair/worker payload is protected by authenticated current target SH."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_experiment_consumer.py
#   src/blueprint_pipeline/native_g1_development_pair.py
#   src/blueprint_pipeline/native_g1_development_worker.py

import fcntl
import hashlib
import json
import os
from pathlib import Path

import pytest

from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_registered_experiment_issuer import installation, issue  # noqa: F401
from tests.test_registered_experiment_birth import birth, prepare
from tests.test_native_g1_development_pair import _paired_requests


@pytest.mark.parametrize("caller", ["pair", "worker"])
def test_registered_missing_use_refuses_before_request_or_payload_read(tmp_path, monkeypatch, caller):
    from blueprint_pipeline import native_g1_development_pair as pair
    from blueprint_pipeline import native_g1_development_worker as worker
    root = tmp_path / "lanes"
    target = root / "g1" / ("registered-" + "a" * 32)
    monkeypatch.setattr(pair, "LANE_SCRATCH_ROOTS", (root,))
    from blueprint_pipeline import control_plane_scratch_lifetime as lifetime
    monkeypatch.setattr(lifetime, "LANE_ROOTS", (root,))
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: pytest.fail("unadmitted payload read"))
    with pytest.raises(ValueError, match="experiment_consumer_authority_required"):
        if caller == "pair":
            pair.run_g1_development_pair(request_paths=[tmp_path / "absent"], output_dir=target)
        else:
            worker.run_g1_development_worker(request={}, output_dir=target / "candidate")


def test_actual_registered_pair_and_worker_preflight_output_holds_target_sh(installation, tmp_path, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_birth as issuer
    from blueprint_pipeline import control_plane_lane_experiment_consumer as consumer
    from blueprint_pipeline import native_g1_development_pair as pair
    from blueprint_pipeline import native_g1_development_worker as worker
    prepare(installation)
    paths, _ = _paired_requests(tmp_path)
    selectors = tuple((path, {"sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                             "size_bytes": path.stat().st_size}) for path in paths)
    grant = issue(installation, participant_profile="g1_local_prelaunch_block.v1", request_records=selectors)
    monkeypatch.setattr(issuer, "_blueprint_identity", lambda: (0, 0))
    monkeypatch.setattr(os, "fchown", lambda *a: None)
    born = birth(installation, grant)
    root = Path(installation[1]["lane_scratch_work_root"])
    monkeypatch.setattr(consumer, "LANE_ROOTS", (root, Path(installation[1]["lane_scratch_inputs_root"])))
    monkeypatch.setattr(consumer, "AUTHORITY_ROOT", installation[2].parents[1] / "experiment-authority")
    monkeypatch.setattr(consumer, "_blueprint_gid", lambda: 0)
    monkeypatch.setattr(pair, "LANE_SCRATCH_ROOTS", (root,))
    from blueprint_pipeline import control_plane_scratch_lifetime as lifetime
    monkeypatch.setattr(lifetime, "LANE_ROOTS", consumer.LANE_ROOTS)
    use = consumer.RegisteredExperimentUse.admit(Path(born["path"]), expected_birth=born["birth"],
        expected_generation=born["generation"], now=lambda: 1200, _producer_request_paths=paths, _producer_config_path=installation[0])
    entered = []
    def preflight(request):
        entered.append(True)
        fd = os.open(born["path"], os.O_RDONLY | os.O_DIRECTORY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
        raise ValueError("tiny_fixture_preflight_refused")
    monkeypatch.setattr(worker, "_preflight_inputs", preflight)
    result = pair.run_g1_development_pair(request_paths=paths, output_dir=Path(born["path"]),
        mode="local", _registered_use=use)
    assert result["status"] == "blocked" and entered == [True]
    attempt = result["attempts"][0]
    receipt = json.loads(Path(attempt["worker_result_path"]).read_bytes())
    assert receipt["phase_reached"] == "preflight" and receipt["teardown"] == {"environment": "not_started", "simulator": "not_started"}
    assert use._closed
    fd = os.open(born["path"], os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)


def _registered_fixture(installation, tmp_path, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_birth as issuer
    from blueprint_pipeline import control_plane_lane_experiment_consumer as consumer
    prepare(installation)
    paths, _ = _paired_requests(tmp_path)
    selectors = tuple((path, {"sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                             "size_bytes": path.stat().st_size}) for path in paths)
    grant = issue(installation, participant_profile="g1_local_prelaunch_block.v1", request_records=selectors)
    monkeypatch.setattr(issuer, "_blueprint_identity", lambda: (0, 0))
    monkeypatch.setattr(os, "fchown", lambda *a: None)
    born = birth(installation, grant)
    monkeypatch.setattr(consumer, "LANE_ROOTS", tuple(Path(installation[1][key]) for key in
                        ("lane_scratch_work_root", "lane_scratch_inputs_root")))
    monkeypatch.setattr(consumer, "AUTHORITY_ROOT", installation[2].parents[1] / "experiment-authority")
    monkeypatch.setattr(consumer, "_blueprint_gid", lambda: 0)
    monkeypatch.setattr(consumer, "PRODUCER_CONFIG_PATH", installation[0])
    return consumer, Path(born["path"]), born, paths


@pytest.mark.parametrize("fault", ["expired", "wrong_generation", "changed_request"])
def test_failed_real_admission_releases_target_sh(installation, tmp_path, monkeypatch, fault):  # noqa: F811
    consumer, target, born, paths = _registered_fixture(installation, tmp_path, monkeypatch)
    if fault == "changed_request":
        paths[0].write_bytes(b"{}")
    with pytest.raises(ValueError):
        consumer.RegisteredExperimentUse.admit(target, expected_birth=born["birth"],
            expected_generation="f" * 32 if fault == "wrong_generation" else born["generation"],
            now=lambda: 2900 if fault == "expired" else 1200, _producer_request_paths=paths)
    fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)


def test_plain_read_admission_is_not_root_producer_permission(installation, tmp_path, monkeypatch):  # noqa: F811
    consumer, target, born, paths = _registered_fixture(installation, tmp_path, monkeypatch)
    with consumer.RegisteredExperimentUse.admit(target, expected_birth=born["birth"], now=lambda: 1200) as use:
        with pytest.raises(ValueError, match="experiment_producer_authority_required"):
            use.authorize_g1_pair(paths)


def test_admitted_producer_rechecks_original_request_before_payload(installation, tmp_path, monkeypatch):  # noqa: F811
    consumer, target, born, paths = _registered_fixture(installation, tmp_path, monkeypatch)
    with consumer.RegisteredExperimentUse.admit(target, expected_birth=born["birth"],
             now=lambda: 1200, _producer_request_paths=paths) as use:
        paths[0].write_bytes(b"{}")
        with pytest.raises(ValueError, match="experiment_producer_request_changed"):
            use.authorize_g1_pair(paths)


def test_actual_prelaunch_pair_closure_publishes_authenticated_completion_for_offload(
        installation, tmp_path, monkeypatch):  # noqa: F811
    from blueprint_pipeline import native_g1_development_pair as pair
    from blueprint_pipeline import native_g1_development_worker as worker
    consumer, target, born, paths = _registered_fixture(installation, tmp_path, monkeypatch)
    monkeypatch.setattr(pair, 'LANE_SCRATCH_ROOTS', consumer.LANE_ROOTS)
    from blueprint_pipeline import control_plane_scratch_lifetime as lifetime
    monkeypatch.setattr(lifetime, 'LANE_ROOTS', consumer.LANE_ROOTS)
    def preflight(_request):
        raise ValueError('development_only_prelaunch_refusal')
    monkeypatch.setattr(worker, '_preflight_inputs', preflight)
    use = consumer.RegisteredExperimentUse.admit(target, expected_birth=born['birth'],
        expected_generation=born['generation'], now=lambda: 1200, _producer_request_paths=paths,
        _producer_config_path=installation[0])
    result = pair.run_g1_development_pair(request_paths=paths, output_dir=target, _registered_use=use)
    assert result['status'] == 'blocked' and use._closed and not use.files.owned
    public = consumer.AUTHORITY_ROOT
    head = json.loads((public / 'HEAD.json').read_bytes())
    current = json.loads((public / head['record_name']).read_bytes())
    entry = current['enrollments'][0]
    assert entry['completion'] is not None and entry['state'] == 'active'
    completion = installation[2] / (entry['intent_id'] + '.producer-completion.json')
    raw = completion.read_bytes()
    assert entry['completion'] == dict(sha256='sha256:' + hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    value = json.loads(raw)
    assert value['participant_profile'] == 'g1_local_prelaunch_block.v1'
    assert value['pair']['sha256'] == 'sha256:' + hashlib.sha256((target / (pair.SCHEMA + '.json')).read_bytes()).hexdigest()
    assert value['lifetime_closed'] is True and value['child_execution_started'] is False


@pytest.mark.parametrize('tampered', [False, True])
def test_real_completed_prelaunch_offload_intent_requires_exact_closure_record(
        installation, tmp_path, monkeypatch, tampered):  # noqa: F811
    from tests.test_registered_experiment_issuer import encoded
    from blueprint_pipeline import native_g1_development_pair as pair
    from blueprint_pipeline import native_g1_development_worker as worker
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root
    policy = json.loads(installation[3].read_bytes())
    policy['principals'][0]['allowed_actions'].append('offload')
    installation[3].write_bytes(encoded(policy))
    config, settings, _, _ = installation
    settings['experiment_retirement_enabled'] = True
    config.write_bytes(encoded(settings))
    consumer, target, born, paths = _registered_fixture(installation, tmp_path, monkeypatch)
    monkeypatch.setattr(pair, 'LANE_SCRATCH_ROOTS', consumer.LANE_ROOTS)
    from blueprint_pipeline import control_plane_scratch_lifetime as lifetime
    monkeypatch.setattr(lifetime, 'LANE_ROOTS', consumer.LANE_ROOTS)
    def preflight(_request):
        raise ValueError('development_only_prelaunch_refusal')
    monkeypatch.setattr(worker, '_preflight_inputs', preflight)
    use = consumer.RegisteredExperimentUse.admit(target, now=lambda: 1200, _producer_request_paths=paths)
    pair.run_g1_development_pair(request_paths=paths, output_dir=target, _registered_use=use)
    intent_id = use.entry['intent_id']
    if tampered:
        record = installation[2] / (intent_id + '.producer-completion.json')
        record.write_bytes(record.read_bytes() + b' ')
        with pytest.raises(ValueError, match='experiment_completion_changed'):
            root.issue_experiment_action_intent(intent_id, principal='operator', owner='owner', action='offload',
                expires_at_epoch=3500, installed_config_path=config, now=lambda: 2900)
    else:
        action = root.issue_experiment_action_intent(intent_id, principal='operator', owner='owner', action='offload',
            expires_at_epoch=3500, installed_config_path=config, now=lambda: 2900)
        assert action['action_intent']['size_bytes'] > 0
        assert json.loads((installation[2] / (action['action_id'] + '.action.json')).read_bytes())['completion'] is not None
