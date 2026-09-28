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
