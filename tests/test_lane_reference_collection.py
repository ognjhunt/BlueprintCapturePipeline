# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_reference_collection.py
#   src/blueprint_pipeline/control_plane_reference_budget.py
"""Hermetic selected roots/proc/config reports never grant reference clearance."""
import hashlib
import json
import os
from dataclasses import asdict

import pytest

from blueprint_pipeline import control_plane_lane_reference_collection as collector
from blueprint_pipeline import control_plane_reference_budget as budgets
from tests.test_leased_scratch_lifetime import folder, open_use
from tests.test_queue_auxiliary_layouts import root_for


@pytest.fixture(autouse=True)
def installed_test_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "LANE_ROOTS", (tmp_path / "lanes",))


def proc_fixture(tmp_path, *, command=None, environment=None, link=None):
    root = tmp_path / "proc"
    pid = root / "123"
    (pid / "fd").mkdir(parents=True)
    (pid / "stat").write_bytes(b'123 (fake worker with spaces) S ' + b'0 ' * 18 + b'42 0 0\n')
    (pid / "cmdline").write_bytes(b'\0'.join(value.encode() for value in (command or ['/fake/python', '-m', 'blueprint_pipeline.task_evaluation_launch_preparation_worker'])) + b'\0')
    (pid / "environ").write_bytes(b'\0'.join(value.encode() for value in (environment or [])) + b'\0')
    (pid / "cwd").symlink_to(link or "/unrelated")
    if link:
        (pid / "fd" / "4").symlink_to(link)
    return root


def selected_roots(tmp_path):
    pins = tmp_path / "pins"
    pins.mkdir()
    preparation = root_for(tmp_path, "preparation")
    for name in collector.PREPARATION_STATES:
        (preparation / name).mkdir(exist_ok=True)
    activation = tmp_path / "activation"
    activation.mkdir()
    for name in (*collector.ACTIVATION_STATES, 'identities', 'results'):
        (activation / name).mkdir()
    return pins, preparation, activation


def test_empty_selected_scopes_are_narrow_and_report_only(tmp_path):
    root, target = folder(tmp_path)
    pins, preparation, activation = selected_roots(tmp_path)
    proc = proc_fixture(tmp_path)
    result = collector.collect_lane_references(str(target), pins_root=str(pins),
        preparation_roots=(str(preparation),), activation_roots=(str(activation),), proc_root=str(proc),
        observed_at_epoch=110, monotonic=lambda: 0)
    assert result.target_probe.status == "observed_exclusive_interval"
    assert result.target_probe.lease_class == 'evidence'
    assert result.target_probe.cleanup_policy == 'owner_review'
    assert result.source_origin == "supplied_only"
    assert result.mutations == 0 and result.candidate_bytes is None
    assert result.collector_owned_descriptor_handling == 'included_in_observed_channel_positives'
    for flag in ('references_clear', 'consumer_fence_checked', 'general_reference_inventory_complete',
                 'general_process_inventory_complete', 'execution_authorized', 'apply_supported'):
        assert getattr(result, flag) is False
    assert result.primary.complete and result.auxiliary.complete and result.pins.complete
    assert 'child_consumer_participation_unproven' in result.kept_reasons
    assert 'producer_configuration_unproven' in result.kept_reasons
    with open_use(root):
        pass  # Probe lifetime really ended before return.


def test_busy_target_is_local_keep_and_selected_positive_sources_remain(tmp_path):
    root, target = folder(tmp_path)
    pins, preparation, activation = selected_roots(tmp_path)
    proc = proc_fixture(tmp_path, link=str(target))
    with open_use(root):
        result = collector.collect_lane_references(str(target), pins_root=str(pins),
            preparation_roots=(str(preparation),), activation_roots=(str(activation),), proc_root=str(proc),
            observed_at_epoch=110, monotonic=lambda: 0)
    assert result.target_probe.status == "kept"
    assert 'lane_scratch_consumer_busy' in result.kept_reasons
    assert any(row.channel == 'fd_link' for row in result.process_references)
    assert result.primary.complete


def test_process_links_hints_and_secrets_have_separate_fixed_projection(tmp_path):
    _, target = folder(tmp_path)
    pins, preparation, activation = selected_roots(tmp_path)
    proc = proc_fixture(tmp_path, command=['/unknown/wrapper', 'private-secret ' + str(target)],
                        environment=['SECRET_KEY=do-not-echo ' + str(target)], link=str(target) + ' (deleted)')
    result = collector.collect_lane_references(str(target), pins_root=str(pins),
        preparation_roots=(str(preparation),), activation_roots=(str(activation),), proc_root=str(proc),
        observed_at_epoch=110, monotonic=lambda: 0)
    assert {row.channel for row in result.process_references} == {'cwd_link', 'fd_link', 'cmdline_hint', 'environ_hint'}
    assert any(row.deleted_suffix_observed for row in result.process_references)
    rendered = json.dumps(asdict(result))
    assert 'SECRET_KEY' not in rendered and 'do-not-echo' not in rendered and 'private-secret' not in rendered
    assert 'producer_configuration_unproven' in result.kept_reasons


def test_global_budget_exhaustion_never_publishes_complete_empty_report(tmp_path, monkeypatch):
    _, target = folder(tmp_path)
    pins, preparation, activation = selected_roots(tmp_path)
    proc = proc_fixture(tmp_path)
    monkeypatch.setattr(budgets, 'MAX_ENTRIES', 1)
    result = collector.collect_lane_references(str(target), pins_root=str(pins),
        preparation_roots=(str(preparation),), activation_roots=(str(activation),), proc_root=str(proc),
        observed_at_epoch=110, monotonic=lambda: 0)
    assert not result.complete_selected_observations
    assert any('limit' in code for code in result.kept_reasons)
    assert not result.references_clear


def test_unsupported_target_refuses_before_any_observation(tmp_path, monkeypatch):
    monkeypatch.setattr(collector.os, 'open', lambda *args, **kwargs: pytest.fail('opened unsupported target'))
    with pytest.raises(collector.LaneReferenceCollectionError, match='parameters_invalid'):
        collector.collect_lane_references('/arbitrary/evidence', pins_root='/pins', proc_root='/proc',
                                          observed_at_epoch=110, monotonic=lambda: 0)


def seal_config(value):
    value = {**value, 'schema_version': 'task_evaluation_scene_progression_config.v1'}
    value['config_digest'] = 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    return value


def test_historical_config_is_never_live_producer_authority(tmp_path):
    _, target = folder(tmp_path)
    pins, preparation, activation = selected_roots(tmp_path)
    config = tmp_path / 'config.json'
    value = seal_config({'intent_root': '/intents', 'preparation_queue_root': str(preparation),
                         'child_queue_root': '/sam', 'submission_enabled': False})
    config.write_text(json.dumps(value, indent=2))
    result = collector.collect_lane_references(str(target), pins_root=str(pins),
        preparation_roots=(str(preparation),), activation_roots=(str(activation),), proc_root=str(proc_fixture(tmp_path)),
        supplied_config_path=str(config), observed_at_epoch=110, monotonic=lambda: 0)
    assert result.configurations[0].origin == 'supplied_historical_config'
    assert result.configurations[0].raw_sha256 != result.configurations[0].canonical_digest
    assert 'producer_configuration_unproven' in result.kept_reasons
    assert any(row.path == '/sam' for row in result.selected_sources)


def test_selected_live_progression_config_retains_pid_identity_and_disabled_routes(tmp_path):
    _, target = folder(tmp_path)
    pins, preparation, activation = selected_roots(tmp_path)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(seal_config({'intent_root': '/intents', 'preparation_queue_root': str(preparation),
                         'child_queue_root': '/sam', 'activation_enabled': False})))
    proc = proc_fixture(tmp_path, command=['/fake/python', '-m', 'blueprint_pipeline.task_evaluation_scene_progression', '--config', str(config)])
    result = collector.collect_lane_references(str(target), pins_root=str(pins),
        preparation_roots=(str(preparation),), activation_roots=(str(activation),), proc_root=str(proc),
        observed_at_epoch=110, monotonic=lambda: 0)
    assert result.configurations[0].origin == 'live_selected_producer_config'
    assert (result.configurations[0].pid, result.configurations[0].start_time) == (123, 42)
    assert any(row.path == '/sam' for row in result.selected_sources)
    assert 'configuration_additional_sources_unobserved' in result.kept_reasons


def test_observed_pid_reuse_is_unknown_and_does_not_erase_positives(tmp_path, monkeypatch):
    _, target = folder(tmp_path)
    pins, preparation, activation = selected_roots(tmp_path)
    proc = proc_fixture(tmp_path, link=str(target))
    original = collector._Collector.verify
    def verify(self):
        path = proc / '123' / 'stat'
        path.write_bytes(path.read_bytes().replace(b'42', b'43'))
        return original(self)
    monkeypatch.setattr(collector._Collector, 'verify', verify)
    result = collector.collect_lane_references(str(target), pins_root=str(pins),
        preparation_roots=(str(preparation),), activation_roots=(str(activation),), proc_root=str(proc),
        observed_at_epoch=110, monotonic=lambda: 0)
    assert 'reference_process_changed' in result.kept_reasons
    assert not result.fixed_process_channels_complete
    assert result.process_references


def test_final_encoding_is_proved_before_asdict_and_budget_close(tmp_path, monkeypatch):
    _, target = folder(tmp_path)
    pins, preparation, activation = selected_roots(tmp_path)
    proc = proc_fixture(tmp_path)
    measured = []
    original = budgets.ReferenceCollectionBudget.measure
    def measure(self, value, **options):
        if isinstance(value, collector.LaneReferenceCollection):
            assert not self.closed
            measured.append(True)
        return original(self, value, **options)
    monkeypatch.setattr(budgets.ReferenceCollectionBudget, 'measure', measure)
    result = collector.collect_lane_references(str(target), pins_root=str(pins),
        preparation_roots=(str(preparation),), activation_roots=(str(activation),), proc_root=str(proc),
        observed_at_epoch=110, monotonic=lambda: 0)
    assert measured and not result.references_clear


def test_proc_token_global_limit_precedes_command_split(tmp_path, monkeypatch):
    _, target = folder(tmp_path)
    pins, preparation, activation = selected_roots(tmp_path)
    proc = proc_fixture(tmp_path, command=['x'] * 64)
    monkeypatch.setattr(budgets, 'MAX_VALUES', 60)
    called = []
    original = collector._Collector.producer
    def producer(self, *args):
        called.append(True)
        return original(self, *args)
    monkeypatch.setattr(collector._Collector, 'producer', producer)
    result = collector.collect_lane_references(str(target), pins_root=str(pins),
        preparation_roots=(str(preparation),), activation_roots=(str(activation),), proc_root=str(proc),
        observed_at_epoch=110, monotonic=lambda: 0)
    assert 'reference_values_limit' in result.kept_reasons and not result.fixed_process_channels_complete


def test_unknown_initial_acquisition_descriptor_is_never_closed_as_foreign(monkeypatch):
    scan = collector._Acquisition((), 110, lambda: 0, 5, budgets.ReferenceCollectionBudget(monotonic=lambda: 0))
    closed = []
    monkeypatch.setattr(os, 'open', lambda *args, **kwargs: 900)
    from blueprint_pipeline import control_plane_scratch_lifetime as lifetime
    monkeypatch.setattr(lifetime, '_identity', lambda fd: (_ for _ in ()).throw(OSError('unproven')))
    monkeypatch.setattr(os, 'close', closed.append)
    with pytest.raises(OSError):
        scan.open('/owned', os.O_RDONLY)
    with pytest.raises(Exception, match='ownership_unproven'):
        scan.close_all()
    assert not closed


@pytest.mark.parametrize('fault', ['missing', 'symlink', 'malformed', 'seal', 'array', 'null'])
def test_failed_selected_config_is_explicit_unknown_not_empty_authority(tmp_path, fault):
    _, target = folder(tmp_path)
    pins, preparation, activation = selected_roots(tmp_path)
    config = tmp_path / 'config.json'
    if fault == 'symlink':
        foreign = tmp_path / 'foreign.json'
        foreign.write_text(json.dumps(seal_config({'intent_root': '/intents'})))
        config.symlink_to(foreign)
    elif fault == 'malformed':
        config.write_text('private-not-json')
    elif fault == 'seal':
        config.write_text(json.dumps({**seal_config({'intent_root': '/intents'}), 'config_digest': 'sha256:' + '0' * 64}))
    elif fault in {'array', 'null'}:
        config.write_text('[]' if fault == 'array' else 'null')
    result = collector.collect_lane_references(str(target), pins_root=str(pins),
        preparation_roots=(str(preparation),), activation_roots=(str(activation),), proc_root=str(proc_fixture(tmp_path)),
        supplied_config_path=str(config), observed_at_epoch=110, monotonic=lambda: 0)
    assert not result.configurations and not result.references_clear
    assert not result.complete_selected_observations
    assert any(code in result.kept_reasons for code in ('reference_config_unavailable', 'queue_row_invalid', 'reference_config_seal_invalid', 'reference_config_shape_invalid'))


def test_config_changed_during_later_queue_observation_is_rechecked(tmp_path, monkeypatch):
    _, target = folder(tmp_path)
    pins, preparation, activation = selected_roots(tmp_path)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(seal_config({'intent_root': str(tmp_path / 'intents')})))
    original = collector.queues.observe_queue_states
    def observe(*args, **kwargs):
        value = original(*args, **kwargs)
        config.write_text(json.dumps(seal_config({'intent_root': str(tmp_path / 'changed')})))
        return value
    monkeypatch.setattr(collector.queues, 'observe_queue_states', observe)
    result = collector.collect_lane_references(str(target), pins_root=str(pins),
        preparation_roots=(str(preparation),), activation_roots=(str(activation),), proc_root=str(proc_fixture(tmp_path)),
        supplied_config_path=str(config), observed_at_epoch=110, monotonic=lambda: 0)
    assert 'reference_config_changed' in result.kept_reasons
    assert not result.complete_selected_observations
    assert result.target_probe.status == 'kept_changed'


def test_proc_stream_sentinel_ignores_advertised_zero_size(tmp_path, monkeypatch):
    _, target = folder(tmp_path)
    pins, preparation, activation = selected_roots(tmp_path)
    proc = proc_fixture(tmp_path, environment=['secret=' + 'x' * 20])
    original = collector.os.fstat
    def metadata(fd):
        value = original(fd)
        # Stream size0 must not suppress reading or its small injected cap.
        if stat_is_regular(value):
            fields = list(value)
            fields[6] = 0
            return os.stat_result(fields)
        return value
    # Target lease uses ns stat identity, so isolate exact binary helper instead
    # of overriding every filesystem metadata representation in the collector.
    budget = budgets.ReferenceCollectionBudget(monotonic=lambda: 0)
    state = collector._Collector(str(target), 110, budget, 'supplied_only')
    directory = state.scan.walk(str(proc / '123'))[-1][0]
    monkeypatch.setattr(collector.os, 'fstat', metadata)
    try:
        with pytest.raises(collector.queues._Blocked, match='') as refusal:
            state.binary(directory, 'environ', cap=8)
        assert refusal.value.code == 'reference_record_bytes_limit'
    finally:
        state.close()


def stat_is_regular(value):
    import stat
    return stat.S_ISREG(value.st_mode)


@pytest.mark.parametrize('clock', [lambda: float('nan'), lambda: True, lambda: 'private-secret'])
def test_invalid_clock_returns_only_fixed_empty_incomplete_packet(tmp_path, clock):
    _, target = folder(tmp_path)
    result = collector.collect_lane_references(str(target), pins_root=str(tmp_path / 'pins'),
                                              observed_at_epoch=110, monotonic=clock)
    assert not result.selected_sources and not result.process_references and not result.complete_selected_observations
    assert 'reference_clock_invalid' in result.kept_reasons
    assert 'private-secret' not in json.dumps(asdict(result))


def test_final_output_limit_is_preencoding_fixed_empty_refusal(tmp_path, monkeypatch):
    _, target = folder(tmp_path)
    pins, preparation, activation = selected_roots(tmp_path)
    monkeypatch.setattr(budgets, 'MAX_OUTPUT_BYTES', 20)
    result = collector.collect_lane_references(str(target), pins_root=str(pins),
        preparation_roots=(str(preparation),), activation_roots=(str(activation),), proc_root=str(proc_fixture(tmp_path)),
        observed_at_epoch=110, monotonic=lambda: 0)
    assert not result.selected_sources and 'reference_output_bytes_limit' in result.kept_reasons


def test_one_shot_acquisition_close_failure_still_finalizes_every_known_descriptor(tmp_path, monkeypatch):
    budget = budgets.ReferenceCollectionBudget(monotonic=lambda: 0)
    state = collector._Collector(str(tmp_path / 'target'), 110, budget, 'supplied_only')
    state.scan.walk(str(tmp_path))
    owned = tuple(state.scan.owner._owned)
    original, original_stat = os.close, os.fstat
    failed = []
    def close(fd):
        if fd in owned and not failed:
            failed.append(fd)
            raise OSError('private close fault')
        return original(fd)
    monkeypatch.setattr(os, 'close', close)
    state.close()
    assert not state.scan.owner._owned
    for fd in owned:
        with pytest.raises(OSError):
            original_stat(fd)


def test_report_contains_raw_row_identity_without_copying_input_document(tmp_path):
    _, target = folder(tmp_path)
    pins, preparation, activation = selected_roots(tmp_path)
    path = preparation / 'pending' / 'unknown.json'
    path.write_text('{"private_metadata":"secret-row-value"}')
    result = collector.collect_lane_references(str(target), pins_root=str(pins),
        preparation_roots=(str(preparation),), activation_roots=(str(activation),), proc_root=str(proc_fixture(tmp_path)),
        observed_at_epoch=110, monotonic=lambda: 0)
    rendered = json.dumps(asdict(result))
    assert 'secret-row-value' not in rendered and 'raw_text' not in rendered
    assert result.primary.rows[0].raw_sha256 == 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('channel', ['cmdline', 'environ', 'cwd', 'fd'])
def test_proc_channel_changes_with_same_pid_and_membership_are_unknown(tmp_path, monkeypatch, channel):
    _, target = folder(tmp_path)
    pins, preparation, activation = selected_roots(tmp_path)
    proc = proc_fixture(tmp_path, link=str(target))
    changed = [False]
    real_link, real_observe = os.readlink, collector.queues.observe_queue_states
    def observe(*args, **kwargs):
        result = real_observe(*args, **kwargs)
        changed[0] = True
        if channel in {'cmdline', 'environ'}:
            (proc / '123' / channel).write_bytes(b'private changed\0')
        return result
    def link(name, *args, **options):
        value = real_link(name, *args, **options)
        if changed[0] and ((channel == 'cwd' and name == 'cwd') or (channel == 'fd' and name == '4')):
            return str(tmp_path / 'changed')
        return value
    monkeypatch.setattr(collector.queues, 'observe_queue_states', observe)
    monkeypatch.setattr(os, 'readlink', link)
    result = collector.collect_lane_references(str(target), pins_root=str(pins),
        preparation_roots=(str(preparation),), activation_roots=(str(activation),), proc_root=str(proc),
        observed_at_epoch=110, monotonic=lambda: 0)
    assert 'reference_process_changed' in result.kept_reasons
    assert not result.fixed_process_channels_complete


def test_size_zero_channel_refuses_before_os_read_when_global_raw_allowance_exhausted(tmp_path, monkeypatch):
    budget = budgets.ReferenceCollectionBudget(monotonic=lambda: 0)
    state = collector._Collector(str(tmp_path / 'target'), 110, budget, 'supplied_only')
    path = tmp_path / 'empty'
    path.write_bytes(b'')
    directory = state.scan.walk(str(tmp_path))[-1][0]
    budget.charge('raw_bytes', budget.limits['raw_bytes'])
    monkeypatch.setattr(os, 'read', lambda *args: pytest.fail('allocated read after global byte exhaustion'))
    try:
        with pytest.raises(budgets.ReferenceCollectionBudgetError, match='raw_bytes_limit'):
            state.binary(directory, 'empty')
    finally:
        state.close()


def test_descriptor_capacity_refuses_before_acquiring_untracked_handle(monkeypatch):
    state = collector._Collector('/target', 110, budgets.ReferenceCollectionBudget(monotonic=lambda: 0), 'supplied_only')
    state.scan.owner._owned = {fd: (1, fd) for fd in range(768)}
    monkeypatch.setattr(os, 'open', lambda *args, **kwargs: pytest.fail('opened beyond owned descriptor cap'))
    with pytest.raises(Exception, match='descriptor_invalid'):
        state.scan.open('/target', os.O_RDONLY)


def test_complete_actual_config_paths_are_declared_without_scanning_unsupported_routes(tmp_path):
    _, target = folder(tmp_path)
    pins, preparation, activation = selected_roots(tmp_path)
    keys = ('factory_output_root', 'completed_source_machinery_path', 'activation_intent_root',
            'terminal_result_root', 'release_binding_root', 'deployment_receipt_root',
            'public_source_catalog_path', 'website_source_binding_root', 'website_source_machinery_path')
    paths = {key: str(tmp_path / ('missing-' + key)) for key in keys}
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(seal_config(paths)))
    result = collector.collect_lane_references(str(target), pins_root=str(pins),
        preparation_roots=(str(preparation),), activation_roots=(str(activation),), proc_root=str(proc_fixture(tmp_path)),
        supplied_config_path=str(config), observed_at_epoch=110, monotonic=lambda: 0)
    assert {(row.role, row.path) for row in result.selected_sources if row.role in keys} == set(paths.items())
    assert all(not os.path.exists(path) for path in paths.values())
    assert 'configuration_additional_sources_unobserved' in result.kept_reasons
    assert result.primary.complete and result.auxiliary.complete


def test_actual_intake_environment_is_declared_with_live_pid_provenance(tmp_path):
    _, target = folder(tmp_path)
    pins, preparation, activation = selected_roots(tmp_path)
    path = str(tmp_path / 'intake')
    proc = proc_fixture(tmp_path, environment=['BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT=' + path])
    result = collector.collect_lane_references(str(target), pins_root=str(pins),
        preparation_roots=(str(preparation),), activation_roots=(str(activation),), proc_root=str(proc),
        observed_at_epoch=110, monotonic=lambda: 0)
    row = next(row for row in result.selected_sources if row.role == 'intent_root' and row.path == path)
    assert (row.origin, row.pid, row.start_time) == ('live_selected_producer_environment', 123, 42)


def test_missing_required_selected_family_never_claims_complete_observations(tmp_path):
    _, target = folder(tmp_path)
    pins, _, activation = selected_roots(tmp_path)
    result = collector.collect_lane_references(str(target), pins_root=str(pins),
        activation_roots=(str(activation),), proc_root=str(proc_fixture(tmp_path)),
        observed_at_epoch=110, monotonic=lambda: 0)
    assert 'selected_reference_family_unconfigured' in result.kept_reasons
    assert not result.complete_selected_observations


def test_canonical_equal_target_lease_raw_version_change_invalidates_interval(tmp_path, monkeypatch):
    _, target = folder(tmp_path)
    pins, preparation, activation = selected_roots(tmp_path)
    original = collector.queues.observe_queue_states
    def observe(*args, **options):
        value = original(*args, **options)
        lease = target / collector.LEASE_FILE
        lease.write_text(json.dumps(json.loads(lease.read_text()), indent=3))
        return value
    monkeypatch.setattr(collector.queues, 'observe_queue_states', observe)
    result = collector.collect_lane_references(str(target), pins_root=str(pins),
        preparation_roots=(str(preparation),), activation_roots=(str(activation),), proc_root=str(proc_fixture(tmp_path)),
        observed_at_epoch=110, monotonic=lambda: 0)
    assert 'reference_target_lease_changed' in result.kept_reasons
    assert result.target_probe.status == 'kept_changed'
    assert not result.complete_selected_observations


@pytest.mark.slow
def test_first_collector_call_cold_has_no_runtime_payload_reads_or_mutations(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    _, target = folder(tmp_path)
    pins, preparation, activation = selected_roots(tmp_path)
    proc = proc_fixture(tmp_path)
    script = r'''
import importlib.abc, os, sys
from pathlib import Path
allowed = {
 'control_plane_lane_reference_collection', 'control_plane_queue_observation',
 'control_plane_queue_auxiliary_observation', 'control_plane_preparation_activation_references',
 'control_plane_reference_budget', 'control_plane_scratch_lifetime', 'control_plane_lane_scratch',
 'control_plane_storage_pin_observation', 'control_plane_storage_pins', 'decision_evidence_contracts',
}
class Reject(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.startswith('blueprint_pipeline.') and fullname.split('.')[-1] not in allowed:
   raise AssertionError('runtime import attempted ' + fullname)
sys.meta_path.insert(0, Reject())
def refuse(*args, **kwargs):
 raise AssertionError('mutation attempted')
for name in ('mkdir','unlink','remove','replace','rename','write','link','symlink'):
 setattr(os,name,refuse)
original = os.open
def read_only(name, flags, *args, **kwargs):
 assert not flags & (os.O_CREAT|os.O_WRONLY|os.O_RDWR|os.O_TRUNC|os.O_APPEND)
 return original(name,flags,*args,**kwargs)
os.open = read_only
from blueprint_pipeline import control_plane_lane_reference_collection as c
c.LANE_ROOTS = (Path(sys.argv[1]).parent.parent,)
result = c.collect_lane_references(sys.argv[1], pins_root=sys.argv[2], preparation_roots=(sys.argv[3],),
 activation_roots=(sys.argv[4],), proc_root=sys.argv[5], observed_at_epoch=110, monotonic=lambda:0)
assert result.primary.complete and result.auxiliary.complete
assert result.target_probe.status == 'observed_exclusive_interval'
assert not result.references_clear and not result.execution_authorized and result.mutations == 0
'''
    completed = subprocess.run([sys.executable, '-c', script, *map(str, (target, pins, preparation, activation, proc))],
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')},
        capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
