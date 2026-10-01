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
