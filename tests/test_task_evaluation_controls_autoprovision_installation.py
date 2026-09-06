"""The installer materializes a config the REAL autoprovisioner consumes.

Only external publication is replaced; ``process_config`` runs the canonical
producer and ``_preparation_context`` against the completed-scene-factory scene
intent and its retained preparation (construction) result.
"""
import grp
import json
import os
import pwd
import time as _time
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_controls_autoprovision as worker
# Dotted (non-aliased) import so the impacted-test selector maps this module.
import blueprint_pipeline.task_evaluation_controls_autoprovision_installation as installer
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline.task_evaluation_scene_progression_state import atomic_json
from tests.test_task_evaluation_controls_autoprovision import _configured_scene, _corrupt_preparation_task_id
from tests.test_task_evaluation_configured_controls_continuation_provisioning import (
    COMMIT, NOW, TASK_ID,
)


@pytest.fixture(autouse=True)
def trusted_scene_issuer(monkeypatch):
    monkeypatch.setenv(intake.CLIENTS_ENV, "webapp")


def _identity():
    return pwd.getpwuid(os.getuid()).pw_name, grp.getgrgid(os.getegid()).gr_name


def _bootstrap_value(tmp_path, kwargs, config_root, *, bindings=None):
    user, group = _identity()
    Path(kwargs["profile_dir"]).mkdir(parents=True, exist_ok=True)
    if bindings is None:
        binding = dict(kwargs["catalog"]["bindings"]["franka-droid"])
        binding.pop("expected_production_commit", None)
        bindings = {"franka-droid": binding}
    return installer.build_bootstrap(
        robot_catalog_bindings=bindings,
        scene_root=str(kwargs["scene_root"]),
        preparation_queue_root=str(kwargs["preparation_queue_root"]),
        controls_root=str(kwargs["controls_root"]),
        intent_root=str(kwargs["intent_root"]),
        profile_dir=str(kwargs["profile_dir"]),
        config_root=str(config_root),
        trusted_clients=list(kwargs["trusted_clients"]),
        service_account=user, service_group=group,
    )


def _install(tmp_path, kwargs, config_root, *, bindings=None):
    bootstrap = _bootstrap_value(tmp_path, kwargs, config_root, bindings=bindings)
    bootstrap_path = config_root / "controls-autoprovision-bootstrap.json"
    atomic_json(bootstrap_path, bootstrap)
    return installer.install_controls_autoprovision(bootstrap_path=bootstrap_path)


def _fake_clock(monkeypatch, offset=1.0):
    monkeypatch.setattr(_time, "time", lambda: NOW.timestamp() + offset)


def test_installed_config_drives_real_autoprovisioner_to_installed_intent(tmp_path, monkeypatch):
    kwargs = _configured_scene(tmp_path)
    config_root = tmp_path / "etc"
    config_root.mkdir()
    receipt = _install(tmp_path, kwargs, config_root)
    assert receipt["status"] == "installed"
    assert receipt["provider_mutation_performed"] is False
    assert receipt["execution_activation_enabled"] is False

    config_path = Path(receipt["config"]["path"])
    # The env pointer names a config that already exists and is readable -- the
    # file is materialized before the variable that fails progression closed.
    env_text = Path(receipt["environment"]["path"]).read_text()
    assert env_text.splitlines()[1] == f"{worker.CONFIG_ENV}={config_path}"
    assert config_path.is_file()
    # The stored catalog stays the release-independent content schema.
    catalog = json.loads(Path(receipt["catalog"]["path"]).read_text())
    assert catalog["schema_version"] == worker.CONTENT_CATALOG_SCHEMA
    assert "expected_production_commit" not in catalog["bindings"]["franka-droid"]

    # Run the REAL config consumer; only publication is faked.
    monkeypatch.setattr(worker.producer, "provision_configured_controls_continuation",
                        kwargs["provisioner"])
    _fake_clock(monkeypatch)
    rows = worker.process_config(str(config_path), expected_production_commit=COMMIT)
    assert [row["status"] for row in rows] == ["installed"]
    provisioning = rows[0]["provisioning"]
    assert provisioning["task_id"] == TASK_ID
    assert provisioning["expected_production_commit"] == COMMIT
    installed = json.loads(Path(rows[0]["installation"]["registry_path"]).read_text())
    assert installed["intent_digest"] == provisioning["intent_digest"]
    # The receipt itself is digest-bound.
    assert rows[0]["receipt_digest"] == worker.canonical_digest(rows[0], digest_field="receipt_digest")

    # Idempotent installer: byte-identical config/catalog/env on the next tick.
    receipt_again = _install(tmp_path, kwargs, config_root)
    assert receipt_again["config"] == receipt["config"]
    assert receipt_again["catalog"] == receipt["catalog"]
    assert receipt_again["environment"] == receipt["environment"]
    # Idempotent consumer: same receipt, no new reservations.
    assert worker.process_config(str(config_path), expected_production_commit=COMMIT) == rows
    attempts = list((kwargs["scene_root"] / kwargs["intent_id"] / "attempts").glob("*.json"))
    assert len(attempts) == 3


@pytest.mark.parametrize("asset_field", ["robot_asset_usd", "embodiment_camera_template"])
def test_installer_rejects_changed_asset_bytes_before_writing_env(tmp_path, asset_field):
    kwargs = _configured_scene(tmp_path)
    config_root = tmp_path / "etc"
    config_root.mkdir()
    bootstrap = _bootstrap_value(tmp_path, kwargs, config_root)
    bootstrap_path = config_root / "controls-autoprovision-bootstrap.json"
    atomic_json(bootstrap_path, bootstrap)
    # Change the retained bytes AFTER sealing the bootstrap.
    Path(bootstrap["robot_catalog_bindings"]["franka-droid"][asset_field]["path"]).write_text("changed")
    with pytest.raises(ValueError):
        installer.install_controls_autoprovision(bootstrap_path=bootstrap_path)
    # Fail closed: no config and no env pointer are left behind.
    assert not (config_root / Path(installer.DEFAULT_CONFIG).name).exists()
    assert not (config_root / Path(installer.DEFAULT_ENVIRONMENT).name).exists()


def test_installer_rejects_changed_runtime_bytes(tmp_path):
    kwargs = _configured_scene(tmp_path)
    config_root = tmp_path / "etc"
    config_root.mkdir()
    bootstrap = _bootstrap_value(tmp_path, kwargs, config_root)
    bootstrap_path = config_root / "controls-autoprovision-bootstrap.json"
    atomic_json(bootstrap_path, bootstrap)
    runtime = Path(bootstrap["robot_catalog_bindings"]["franka-droid"]["runtime_source_payload_dir"])
    (runtime / "extra-file").write_text("changed")
    with pytest.raises(ValueError):
        installer.install_controls_autoprovision(bootstrap_path=bootstrap_path)
    assert not (config_root / Path(installer.DEFAULT_ENVIRONMENT).name).exists()


@pytest.mark.parametrize("mutation,blocker", [
    ("expired", "authority_expired"),
    ("construction", "task_mismatch"),
])
def test_installed_config_consumer_refuses_expiry_and_mismatched_construction(
    tmp_path, monkeypatch, mutation, blocker
):
    kwargs = _configured_scene(tmp_path)
    config_root = tmp_path / "etc"
    config_root.mkdir()
    receipt = _install(tmp_path, kwargs, config_root)
    config_path = Path(receipt["config"]["path"])

    def unexpected(**_):
        pytest.fail("producer must not run for a refused scene")

    monkeypatch.setattr(worker.producer, "provision_configured_controls_continuation", unexpected)
    if mutation == "expired":
        _fake_clock(monkeypatch, offset=7201.0)
    else:
        _fake_clock(monkeypatch)
        _corrupt_preparation_task_id(kwargs, "someone-elses-task")
    rows = worker.process_config(str(config_path), expected_production_commit=COMMIT)
    assert rows[0]["status"] == "controls_autoprovision_refused"
    assert blocker in rows[0]["blocker"]
    # No reservation or installed intent is left behind by a refused scene.
    assert not (kwargs["scene_root"] / kwargs["intent_id"] / "attempts").exists()


def test_installer_rejects_operator_owned_config_conflict(tmp_path):
    kwargs = _configured_scene(tmp_path)
    config_root = tmp_path / "etc"
    config_root.mkdir()
    # An operator-owned (unmanaged) file at the config path must never be clobbered.
    (config_root / Path(installer.DEFAULT_CONFIG).name).write_text(json.dumps({"operator": "owned"}))
    with pytest.raises(ValueError, match="unmanaged_file"):
        _install(tmp_path, kwargs, config_root)


def test_installer_refuses_world_writable_bootstrap(tmp_path):
    kwargs = _configured_scene(tmp_path)
    config_root = tmp_path / "etc"
    config_root.mkdir()
    bootstrap = _bootstrap_value(tmp_path, kwargs, config_root)
    bootstrap_path = config_root / "controls-autoprovision-bootstrap.json"
    atomic_json(bootstrap_path, bootstrap)
    bootstrap_path.chmod(0o646)
    with pytest.raises(ValueError, match="bootstrap_writable"):
        installer.install_controls_autoprovision(bootstrap_path=bootstrap_path)
