"""ADP-009D/day28: actual default-off authenticated experiment intent issuance."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_experiment_retirement.py
#   deploy/operator-door/operator_door/config.py

import hashlib
import json
import os
from pathlib import Path

import pytest

from tests.test_owner_target_version_publication import root_metadata  # noqa: F401


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


@pytest.fixture
def installation(tmp_path, monkeypatch, root_metadata):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_owner_consents as owners
    package = tmp_path / "installed" / "operator_door"
    package.mkdir(parents=True, mode=0o700)
    source = Path(__file__).parents[1] / "deploy/operator-door/operator_door"
    for name in ("__init__.py", "config.py"):
        (package / name).write_bytes((source / name).read_bytes())
        (package / name).chmod(0o600)
    state = tmp_path / "state"
    store = state / "requests/experiment-records"
    store.mkdir(parents=True, mode=0o700)
    (store / ".experiment-authority.lock").write_bytes(b"")
    (store / ".experiment-authority.lock").chmod(0o600)
    policy = tmp_path / "policy.json"
    policy.write_bytes(encoded(dict(schema_version=owners.POLICY_SCHEMA, enabled=True,
        principals=[dict(principal="operator", owners=["owner"], allowed_actions=["register"],
                         max_consent_seconds=3600)])))
    policy.chmod(0o600)
    config = tmp_path / "door.json"
    settings = dict(state_root=str(state), experiment_creation_enabled=True,
                    lane_owner_policy_file=str(policy), lane_scratch_work_root=str(tmp_path / "work/lanes"),
                    lane_scratch_inputs_root=str(tmp_path / "inputs/lanes"))
    config.write_bytes(encoded(settings))
    config.chmod(0o600)
    from blueprint_pipeline import control_plane_lane_experiment_consumer as consumer
    # Disposable fixture repins the compiled installed namespace, never a body
    # root override; every positive issue and later reader uses these same roots.
    monkeypatch.setattr(consumer, 'LANE_ROOTS', (Path(settings['lane_scratch_work_root']), Path(settings['lane_scratch_inputs_root'])))
    monkeypatch.setattr(owners, "INSTALLED_PACKAGE_ROOT", package.parent)
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    # This hermetic root-metadata fixture does not claim kernel boot proof.
    # Mandatory distinct-UID Linux acceptance bypasses this fixture entirely.
    from blueprint_pipeline import control_plane_lane_experiment_work as work
    monkeypatch.setattr(work, '_controller_boot_id', lambda files: '12345678-1234-1234-1234-123456789abc')
    return config, settings, store, policy


def issue(installation, **changes):
    from blueprint_pipeline.control_plane_lane_experiment_retirement import issue_experiment_creation_intent
    config, _, _, _ = installation
    options = dict(installed_config_path=config, principal="operator", owner="owner", root="work", reference_value="run1",
        lease_ttl_seconds=1800, participant_profile="local_root_disposable.v1", request_records=(), now=lambda: 1000)
    return issue_experiment_creation_intent(**(options | changes))


def test_default_flags_and_private_public_paths_are_derived(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "deploy/operator-door"))
    from operator_door.config import config_from_mapping
    config = config_from_mapping({"state_root": "/state"})
    assert config.experiment_creation_enabled is False and config.experiment_retirement_enabled is False
    assert config.experiment_record_store == "/state/requests/experiment-records"
    assert config.experiment_authority_root == "/state/experiment-authority"


@pytest.mark.parametrize("field", ["experiment_creation_enabled", "experiment_retirement_enabled"])
@pytest.mark.parametrize("bad", [0, 1, "true", None])
def test_new_enablement_requires_actual_bool(field, bad, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "deploy/operator-door"))
    from operator_door.config import DoorConfigError, config_from_mapping
    with pytest.raises(DoorConfigError):
        config_from_mapping({field: bad})


def test_issue_authentic_immutable_intent_does_not_create_payload(installation, monkeypatch):
    from blueprint_pipeline import control_plane_lane_owner_consents as old
    monkeypatch.setattr(old, "_publish", lambda *a, **kw: pytest.fail("unsafe publisher"))
    monkeypatch.setattr(os, "replace", lambda *a, **kw: pytest.fail("replace"))
    result = issue(installation)
    _, settings, store, policy = installation
    path = store / (result["intent_id"] + ".json")
    record = json.loads(path.read_bytes())
    assert record["name"] == "registered-" + record["intent_id"]
    assert record["generation"] != record["intent_id"]
    assert record["issuer_uid"] == 0 and record["principal"] == "operator"
    assert record["class_intent"] == "scratch" and record["cleanup"] == "delete"
    assert record["request_records"] == [] and record["lease_ttl_seconds"] == 1800
    assert record["expires_at_epoch"] == 2800
    assert record["policy"] == {"sha256": "sha256:" + hashlib.sha256(policy.read_bytes()).hexdigest(),
                                 "size_bytes": policy.stat().st_size}
    assert result["intent"] == {"sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                                 "size_bytes": path.stat().st_size}
    assert not Path(settings["lane_scratch_work_root"]).exists()
    assert len(list(store.iterdir())) == 2


@pytest.mark.parametrize("condition,code", [("disabled", "experiment_creation_disabled"),
    ("uid", "experiment_issuer_required"), ("owner", "owner_consent_owner_unmapped"),
    ("ttl", "experiment_creation_invalid"), ("policy", "owner_consent_action_unapproved"),
    ("store_mode", "owner_consent_store_unsafe")])
def test_refusals_precede_publication(installation, monkeypatch, condition, code):
    config, settings, store, policy = installation
    options = {}
    if condition == "disabled":
        config.write_bytes(encoded(settings | {"experiment_creation_enabled": False}))
    if condition == "uid":
        monkeypatch.setattr(os, "geteuid", lambda: 501)
    if condition == "owner":
        value = json.loads(policy.read_bytes())
        value["principals"][0]["owners"] = ["other"]
        policy.write_bytes(encoded(value))
    if condition == "ttl":
        options["expires_at_epoch"] = 1000
    if condition == "policy":
        value = json.loads(policy.read_bytes())
        value["principals"][0]["allowed_actions"] = ["keep"]
        policy.write_bytes(encoded(value))
    if condition == "store_mode":
        store.chmod(0o777)
    with pytest.raises(ValueError, match=code):
        issue(installation, **options)
    assert list(store.glob("*.json")) == []


def test_reused_random_id_is_not_reissued_or_overwritten(installation, monkeypatch):
    from blueprint_pipeline import control_plane_lane_experiment_retirement as feature
    monkeypatch.setattr(feature.secrets, "token_hex", lambda _: "a" * 32)
    # Distinct generation is mandatory even if the random source is faulty.
    with pytest.raises(ValueError, match="experiment_creation_invalid"):
        issue(installation)
    assert list(installation[2].glob("*.json")) == []


def test_previous_durable_intent_id_collision_keeps_old_grant(installation, monkeypatch):
    from blueprint_pipeline import control_plane_lane_experiment_retirement as feature
    ids = iter(["a" * 32, "b" * 32, "c" * 32, "a" * 32, "d" * 32])
    monkeypatch.setattr(feature.secrets, "token_hex", lambda size: next(ids) if size == 16 else "e" * 16)
    result = issue(installation)
    path = installation[2] / (result["intent_id"] + ".json")
    previous = path.read_bytes()
    with pytest.raises(ValueError, match="owner_target_publication_destination_exists"):
        issue(installation)
    assert path.read_bytes() == previous
    assert json.loads(previous)["generation"] == "b" * 32
    assert len(list(installation[2].glob("*.json"))) == 1


def test_uninstalled_experiment_root_refuses_before_private_grant(installation):
    config, settings, store, _ = installation
    config.write_bytes(encoded(settings | {'lane_scratch_work_root': str(config.parent/'uninstalled/lanes')}))
    before = {p.name:p.read_bytes() for p in store.iterdir()}
    with pytest.raises(ValueError, match='experiment_installed_namespace_changed'):
        issue(installation)
    assert {p.name:p.read_bytes() for p in store.iterdir()} == before
