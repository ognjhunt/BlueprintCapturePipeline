"""ADP-009D/day28: genuine sealed diagnostic + separate exact owner action.

Portable metadata fixtures prove issuance/manifest/default-off boundaries. They
do not substitute for installed Linux native reference and GC acceptance.
"""
import json
from pathlib import Path

import pytest

from tests.test_registered_experiment_issuer import installation, encoded  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_registered_disk_diagnostic_producer import enrolled, current
from tests.test_registered_experiment_retirement_flow import _gc


def configure(installation, *, enabled=True):  # noqa: F811
    config, settings, _, policy_path = installation
    pins = config.parent / "pins"
    pins.mkdir(mode=0o700)
    environment = config.parent / "gc.env"
    environment.write_text("BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT=" + str(pins) + "\n")
    environment.chmod(0o600)
    settings.update(experiment_retirement_enabled=enabled, experiment_gc_environment_file=str(environment))
    config.write_bytes(encoded(settings))
    policy = json.loads(policy_path.read_bytes())
    policy["principals"][0]["allowed_actions"] = ["register", "delete", "offload", "keep"]
    policy_path.write_bytes(encoded(policy))


def seal(installation, monkeypatch, profile):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_disk_diagnostic as diagnostic
    public, request, grant, born = enrolled(installation, monkeypatch, profile)
    result = diagnostic.run_registered_disk_diagnostic(grant["intent_id"], expected_intent=grant["intent"],
        request_path=request, installed_config_path=installation[0], now=lambda: 1200)
    return public, request, grant, born, result


@pytest.mark.parametrize("profile,action", [("root_disk_diagnostic_disposable.v1", "delete"),
                                           ("root_disk_diagnostic_evidence.v1", "offload")])
def test_exact_owner_can_issue_only_sealed_original_report_manifest(installation, monkeypatch, profile, action):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_retirement as issuer
    from blueprint_pipeline import control_plane_lane_disk_diagnostic as diagnostic
    configure(installation)
    public, _, grant, born, completed = seal(installation, monkeypatch, profile)
    original = (Path(born["path"]) / diagnostic.REPORT_NAME).read_bytes()
    decision = issuer.issue_experiment_action_intent(grant["intent_id"], principal="operator", owner="owner",
        action=action, expires_at_epoch=3500, installed_config_path=installation[0], now=lambda: 2900)
    record = json.loads((installation[2] / (decision["action_id"] + ".action.json")).read_bytes())
    manifest = json.loads((installation[2] / (decision["action_id"] + ".manifest.json")).read_bytes())
    assert record["action"] == action and record["owner"] == "owner"
    assert record["completion"] == completed["completion"] and record["generation"] == born["generation"]
    assert len(manifest["members"]) == 1 and manifest["members"][0][0] == diagnostic.REPORT_NAME
    assert manifest["logical_bytes"] == len(original)
    assert (Path(born["path"]) / diagnostic.REPORT_NAME).read_bytes() == original
    assert current(public)["state"] == "active"
    assert current(public)["operation_id"] == decision["action_id"]
    # Explicit timer opt-in is still independent of the protected owner packet.
    report = _gc(installation, enabled=False)
    assert report["registered_experiments"]["enabled"] is False
    assert (Path(born["path"]) / diagnostic.REPORT_NAME).read_bytes() == original


@pytest.mark.parametrize("profile,action", [("root_disk_diagnostic_disposable.v1", "delete"),
                                           ("root_disk_diagnostic_evidence.v1", "offload")])
def test_unsealed_actual_birth_has_no_owner_cleanup_eligibility(installation, monkeypatch, profile, action):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_retirement as issuer
    configure(installation)
    public, _, grant, born = enrolled(installation, monkeypatch, profile)
    before = {path.name: path.read_bytes() for path in Path(born["path"]).iterdir()}
    with pytest.raises(ValueError, match="experiment_completion_required"):
        issuer.issue_experiment_action_intent(grant["intent_id"], principal="operator", owner="owner",
            action=action, expires_at_epoch=3500, installed_config_path=installation[0], now=lambda: 2900)
    assert current(public)["state"] == "active" and current(public)["completion"] is None
    assert {path.name: path.read_bytes() for path in Path(born["path"]).iterdir()} == before


@pytest.mark.parametrize("change", ["foreign_member", "report_bytes", "report_mode", "identical_replacement", "identical_rewrite"])
def test_completion_does_not_authorize_changed_or_extra_payload(installation, monkeypatch, change):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_retirement as issuer
    from blueprint_pipeline import control_plane_lane_disk_diagnostic as diagnostic
    configure(installation)
    public, _, grant, born, _ = seal(installation, monkeypatch, "root_disk_diagnostic_evidence.v1")
    target = Path(born["path"])
    if change == "foreign_member":
        (target / "foreign").write_bytes(b"keep")
    elif change == "report_bytes":
        (target / diagnostic.REPORT_NAME).write_bytes(b"changed")
    elif change == "report_mode":
        (target / diagnostic.REPORT_NAME).chmod(0o640)
    else:
        report = target / diagnostic.REPORT_NAME
        original = report.read_bytes()
        if change == "identical_replacement":
            replacement = target / "owned-test-replacement"
            replacement.write_bytes(original)
            replacement.chmod(0o600)
            replacement.replace(report)
        else:
            report.write_bytes(original)
    before = {path.name: path.read_bytes() for path in target.iterdir()}
    with pytest.raises(ValueError, match="diagnostic_manifest_changed|experiment_completion_changed"):
        issuer.issue_experiment_action_intent(grant["intent_id"], principal="operator", owner="owner",
            action="offload", expires_at_epoch=3500, installed_config_path=installation[0], now=lambda: 2900)
    assert current(public)["state"] == "active"
    assert {path.name: path.read_bytes() for path in target.iterdir()} == before


def test_actual_seal_does_not_enable_default_off_retirement(installation, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_retirement as issuer
    configure(installation, enabled=False)
    public, _, grant, _, _ = seal(installation, monkeypatch, "root_disk_diagnostic_disposable.v1")
    with pytest.raises(ValueError, match="experiment_retirement_disabled"):
        issuer.issue_experiment_action_intent(grant["intent_id"], principal="operator", owner="owner",
            action="delete", expires_at_epoch=3500, installed_config_path=installation[0], now=lambda: 2900)
    assert current(public)["state"] == "active"


def test_shipped_gc_grants_only_the_two_diagnostic_lane_prefixes():
    unit = (Path(__file__).parents[1] / 'deploy/systemd/blueprint-control-plane-storage-gc.service').read_text()
    allowed = {item.removeprefix('-') for line in unit.splitlines() if line.startswith('ReadWritePaths=')
               for item in line.split('=', 1)[1].split()}
    assert {'/mnt/blueprint-work/lanes/diagnostics',
            '/var/lib/blueprint/task-evaluation-inputs/lanes/diagnostics'} <= allowed
    assert '/mnt/blueprint-work/lanes' not in allowed
    assert '/var/lib/blueprint/task-evaluation-inputs/lanes' not in allowed


def test_fixed_cli_executes_only_the_authenticated_diagnostic(installation, monkeypatch, capsys):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_retirement as issuer
    configure(installation)
    public, request, grant, born = enrolled(installation, monkeypatch)
    monkeypatch.setattr(issuer, 'INSTALLED_CONFIG_PATH', installation[0])
    monkeypatch.setattr(issuer.time, 'time', lambda: 1200)
    code = issuer.main(['run-diagnostic', grant['intent_id'], '--sha256', grant['intent']['sha256'],
                        '--size-bytes', str(grant['intent']['size_bytes']), '--request-path', str(request)])
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result['decision'] == 'completed'
    assert current(public)['completion'] == result['result']['completion']
    assert (Path(born['path']) / 'disk-capacity-report.v1.json').exists()


@pytest.mark.parametrize('mode', [0o777, 0o750])
def test_sealed_target_rights_are_required_when_issuing_cleanup(installation, monkeypatch, mode):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_retirement as issuer
    configure(installation)
    _, _, grant, born, _ = seal(installation, monkeypatch, 'root_disk_diagnostic_disposable.v1')
    target = Path(born['path'])
    before = {path.name: path.read_bytes() for path in target.iterdir()}
    target.chmod(mode)
    with pytest.raises(ValueError, match='experiment_diagnostic_rights_changed'):
        issuer.issue_experiment_action_intent(grant['intent_id'], principal='operator', owner='owner',
            action='delete', expires_at_epoch=3500, installed_config_path=installation[0], now=lambda: 2900)
    assert {path.name: path.read_bytes() for path in target.iterdir()} == before


def test_new_member_after_owner_issue_refuses_before_any_payload_removal(installation, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_retirement as issuer
    from blueprint_pipeline import control_plane_lane_experiment_actions as actions
    configure(installation)
    _, _, grant, born, _ = seal(installation, monkeypatch, 'root_disk_diagnostic_disposable.v1')
    selected = issuer.issue_experiment_action_intent(grant['intent_id'], principal='operator', owner='owner',
        action='delete', expires_at_epoch=3500, installed_config_path=installation[0], now=lambda: 2900)
    target = Path(born['path'])
    (target / 'late-foreign-member').write_bytes(b'preserve me')
    before = {path.name: path.read_bytes() for path in target.iterdir()}
    with pytest.raises(ValueError, match='experiment_diagnostic_namespace_changed'):
        actions.run_action(selected['action_id'], expected_action_intent=selected['action_intent'],
            installed_config_path=installation[0], now=lambda: 2901, _pins_root=installation[0].parent / 'pins')
    assert {path.name: path.read_bytes() for path in target.iterdir()} == before


def test_sealed_report_without_complete_reference_settings_cannot_delete(installation, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_retirement as issuer
    from blueprint_pipeline import control_plane_lane_experiment_actions as actions
    configure(installation)
    _, _, grant, born, _ = seal(installation, monkeypatch, 'root_disk_diagnostic_disposable.v1')
    selected = issuer.issue_experiment_action_intent(grant['intent_id'], principal='operator', owner='owner',
        action='delete', expires_at_epoch=3500, installed_config_path=installation[0], now=lambda: 2900)
    target = Path(born['path'])
    before = {path.name: path.read_bytes() for path in target.iterdir()}
    with pytest.raises(ValueError, match='experiment_diagnostic_references_unknown'):
        actions.run_action(selected['action_id'], expected_action_intent=selected['action_intent'],
            installed_config_path=installation[0], now=lambda: 2901, _pins_root=installation[0].parent / 'pins')
    assert {path.name: path.read_bytes() for path in target.iterdir()} == before


@pytest.mark.parametrize('reference_kind', ['literal', 'encoded', 'directory_alias'])
def test_actual_current_queue_reference_preserves_the_sealed_report(installation, monkeypatch, reference_kind):  # noqa: F811
    """Actual URI/table bytes and alias; portable owners, no kernel clearance."""
    from blueprint_pipeline import control_plane_lane_experiment_retirement as issuer
    from blueprint_pipeline import control_plane_lane_experiment_actions as actions
    from blueprint_pipeline import control_plane_lane_legacy_owner as legacy
    configure(installation)
    config, settings, _, _ = installation
    root = config.parent
    selected = {}
    for kind in ('queue', 'evidence', 'settlement'):
        path = root / kind
        path.mkdir(mode=0o700)
        selected[kind] = path
    unit = root / 'gc.service'
    unit.write_bytes(b'[Service]\n')
    unit.chmod(0o600)
    monkeypatch.setattr(legacy, '_GC_UNIT', unit)
    environment = root / 'gc.env'
    environment.write_text(environment.read_text() + ''.join(
        'BLUEPRINT_CONTROL_PLANE_GC_' + key.upper() + '_ROOTS=' + str(selected[key]) + '\n'
        for key in selected))
    release = root / 'active-release'
    release.symlink_to(root / 'installed')
    settings.update(control_plane_state=settings['state_root'], active_release_link=str(release))
    config.write_bytes(encoded(settings))
    _, _, grant, born, _ = seal(installation, monkeypatch, 'root_disk_diagnostic_disposable.v1')
    selected_action = issuer.issue_experiment_action_intent(grant['intent_id'], principal='operator', owner='owner',
        action='delete', expires_at_epoch=3500, installed_config_path=config, now=lambda: 2900)
    target = Path(born['path'])
    report = target / 'disk-capacity-report.v1.json'
    if reference_kind == 'encoded':
        reference = 'file://' + ''.join('/' if byte == 47 else '%' + format(byte, '02X')
                                        for byte in bytes(str(report), 'utf-8'))
        assert target.name not in reference
    elif reference_kind == 'directory_alias':
        alias = root / 'current-source'
        alias.symlink_to(target, target_is_directory=True)
        reference = (alias / report.name).as_uri()
        assert (alias / report.name).read_bytes() == report.read_bytes()
        assert target.name not in reference
    else:
        reference = str(report)
    record = selected['queue'] / 'active.json'
    record.write_bytes(encoded({'request': {'path': reference}}))
    record.chmod(0o600)
    before = {path.name: path.read_bytes() for path in target.iterdir()}
    reason = ('experiment_diagnostic_references_unknown' if reference_kind == 'directory_alias'
              else 'experiment_diagnostic_queue_reference')
    with pytest.raises(ValueError, match=reason):
        actions.run_action(selected_action['action_id'], expected_action_intent=selected_action['action_intent'],
            installed_config_path=config, now=lambda: 2901, _pins_root=root / 'pins')
    assert {path.name: path.read_bytes() for path in target.iterdir()} == before


def test_actual_diagnostic_delete_reference_accounting(installation, monkeypatch):  # noqa: F811
    """Portable accounting only; native process clearance is separately mandatory."""
    from blueprint_pipeline import control_plane_lane_experiment_retirement as issuer
    from blueprint_pipeline import control_plane_lane_experiment_actions as actions
    from blueprint_pipeline import control_plane_lane_legacy_owner as legacy
    from blueprint_pipeline import control_plane_lane_disk_diagnostic_references as references
    configure(installation)
    config, settings, _, _ = installation
    root = config.parent
    tables = {}
    for kind in ('queue', 'evidence', 'settlement'):
        tables[kind] = root / kind
        tables[kind].mkdir(mode=0o700)
    unit = root / 'gc.service'
    unit.write_bytes(b'[Service]\n')
    unit.chmod(0o600)
    monkeypatch.setattr(legacy, '_GC_UNIT', unit)
    environment = root / 'gc.env'
    environment.write_text(environment.read_text() + ''.join(
        'BLUEPRINT_CONTROL_PLANE_GC_' + key.upper() + '_ROOTS=' + str(path) + '\n'
        for key, path in tables.items()))
    release = root / 'active-release'
    release.symlink_to(root / 'installed')
    settings.update(control_plane_state=settings['state_root'], active_release_link=str(release))
    config.write_bytes(encoded(settings))
    _, _, grant, born, _ = seal(installation, monkeypatch, 'root_disk_diagnostic_disposable.v1')
    selected = issuer.issue_experiment_action_intent(grant['intent_id'], principal='operator', owner='owner',
        action='delete', expires_at_epoch=3500, installed_config_path=config, now=lambda: 2900)
    target = Path(born['path'])
    report = target / 'disk-capacity-report.v1.json'
    original = report.read_bytes()
    monkeypatch.setattr(references.DiagnosticReferences, '_own_handles', lambda *args: None)
    monkeypatch.setattr(references, 'refuse_historical_process_references', lambda *args, **kwargs: None)
    checks = []
    original_guard = references.DiagnosticReferences.guard
    def guard(reader, **kwargs):
        original_guard(reader, **kwargs)
        checks.append((reader.budget, dict(reader.budget.counts), reader.budget.deadline))
    monkeypatch.setattr(references.DiagnosticReferences, 'guard', guard)
    outcome = actions.run_action(selected['action_id'], expected_action_intent=selected['action_intent'],
        installed_config_path=config, now=lambda: 2901, _pins_root=root / 'pins')
    assert outcome['decision'] == 'retired' and outcome['removed_logical_bytes'] == len(original)
    assert outcome['receipt'] and not report.exists()
    assert len(checks) > 16
    assert all(budget is checks[0][0] and deadline == checks[0][2]
               and counts['roots'] == checks[0][1]['roots'] < 16 and counts['groups'] == 3
               for budget, counts, deadline in checks)
    assert all(later[1]['entries'] > earlier[1]['entries']
               and later[1]['raw_bytes'] > earlier[1]['raw_bytes']
               for earlier, later in zip(checks, checks[1:]))


def test_final_named_member_check_cannot_cross_original_delete_expiry(installation, monkeypatch):  # noqa: F811
    """Portable clock model only; this supplies no kernel reader clearance."""
    import sys
    from blueprint_pipeline import control_plane_lane_experiment_retirement as issuer
    from blueprint_pipeline import control_plane_lane_experiment_actions as actions
    from blueprint_pipeline import control_plane_lane_disk_diagnostic_references as references
    configure(installation)
    _, _, grant, born, _ = seal(installation, monkeypatch, 'root_disk_diagnostic_disposable.v1')
    selected = issuer.issue_experiment_action_intent(grant['intent_id'], principal='operator', owner='owner',
        action='delete', expires_at_epoch=3500, installed_config_path=installation[0], now=lambda: 2900)
    target = Path(born['path'])
    report = target / 'disk-capacity-report.v1.json'
    original = report.read_bytes()
    clock = [2901]
    class PortableReferences:
        def __init__(self, *args, **kwargs):
            from types import SimpleNamespace
            self.files = SimpleNamespace(owned=[], probe_owned=[])
            self.budget = SimpleNamespace(tick=lambda: None)
        def guard(self, **kwargs): pass
        def close(self): pass
    monkeypatch.setattr(references, 'DiagnosticReferences', PortableReferences)
    actual_stat = actions.os.stat
    def delayed_stat(path, *args, **kwargs):
        info = actual_stat(path, *args, **kwargs)
        caller = sys._getframe(1)
        if caller.f_code.co_name == 'run_action' and str(path) == report.name:
            clock[0] = 3501
        return info
    monkeypatch.setattr(actions.os, 'stat', delayed_stat)
    with pytest.raises(ValueError, match='experiment_action_expired|experiment_work_refused'):
        actions.run_action(selected['action_id'], expected_action_intent=selected['action_intent'],
            installed_config_path=installation[0], now=lambda: clock[0], _pins_root=installation[0].parent / 'pins')
    assert clock[0] == 3501 and report.read_bytes() == original
