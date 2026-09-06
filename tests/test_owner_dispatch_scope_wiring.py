"""The owner-scope wiring ships in the units, and scene_store() resolves the owner.

Spec C installs ``persistent_owner_only`` and the shared owner-store identity in
both dispatchers and controls progression.  These tests pin the shipped unit
wiring and the ``scene_store()`` contract the wiring depends on, so a future edit
cannot silently drop owner mode back to ``all_authorized`` or leave the store
unresolvable.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import task_evaluation_scene_policy_binding as binding

SYSTEMD = Path(__file__).resolve().parents[1] / "deploy" / "systemd"
SCOPE = "Environment=BLUEPRINT_TASK_EVALUATION_DISPATCH_OWNER_SCOPE=persistent_owner_only"
STORE_ENV_FILE = "EnvironmentFile=-/etc/blueprint/task-evaluation-scene-progression.env"
CONFIG_ENV_FILE = "EnvironmentFile=-/etc/blueprint/task-evaluation-controls-autoprovision.env"
UNITS = (
    "blueprint-task-evaluation-launch-dispatcher.service",
    "blueprint-task-evaluation-policy-canary-dispatcher.service",
    "blueprint-task-evaluation-configured-controls-progression.service",
)


def _unit(name):
    return (SYSTEMD / name).read_text()


@pytest.mark.parametrize("unit", UNITS)
def test_every_authority_unit_installs_persistent_owner_scope_and_store(unit):
    text = _unit(unit)
    assert SCOPE in text, f"{unit} must install persistent_owner_only"
    # The shared owner store is loaded optionally ("-"): its absence must never
    # block the unit start, only leave selection with nothing to resolve.
    assert STORE_ENV_FILE in text, f"{unit} must load the shared owner-store env"


def test_controls_progression_loads_autoprovision_config_optionally():
    text = _unit("blueprint-task-evaluation-configured-controls-progression.service")
    # Optional load: a missing file leaves the legacy lane, never fails closed.
    assert CONFIG_ENV_FILE in text
    # The autoprovision worker installs intents into the registry, so it must be
    # writable, not read-only, for this unit.
    read_write = [line for line in text.splitlines() if line.startswith("ReadWritePaths=")]
    read_only = [line for line in text.splitlines() if line.startswith("ReadOnlyPaths=")]
    intents = "/etc/blueprint/task-evaluation-configured-controls-intents"
    assert any(intents in line for line in read_write)
    assert not any(intents in line for line in read_only)


def test_scope_is_never_downgraded_to_all_authorized():
    for unit in UNITS:
        assert "all_authorized" not in _unit(unit)


def test_deploy_materializes_controls_autoprovision_gated_on_bootstrap_after_registry():
    import scripts.deploy_control_plane_commit as deploy

    src = Path(deploy.__file__).read_text(encoding="utf-8")
    assert "controls_autoprovision_bootstrap_file" in src
    assert "install_controls_autoprovision" in src
    assert '"controls_autoprovision_installation": controls_autoprovision_installation' in src
    # The config must be materialized AFTER the autostart registry (so its
    # group-writable permission wins) and gated on the operator bootstrap so a
    # missing bootstrap leaves the legacy lane rather than a broken config env.
    assert src.index("_install_configured_controls_autostart_registry(") < src.index(
        "install_controls_autoprovision(\n                bootstrap_path=controls_autoprovision_bootstrap"
    )
    assert 'controls_autoprovision_bootstrap.exists()' in src


def _clear(monkeypatch):
    monkeypatch.delenv(intake.ROOT_ENV, raising=False)
    monkeypatch.delenv(intake.CLIENTS_ENV, raising=False)
    monkeypatch.delenv("BLUEPRINT_TASK_EVALUATION_CONTROLS_AUTOPROVISION_CONFIG", raising=False)


def test_scene_store_resolves_from_scene_intake_root(tmp_path, monkeypatch):
    _clear(monkeypatch)
    root = tmp_path / "scene-intents"
    root.mkdir()
    monkeypatch.setenv(intake.ROOT_ENV, str(root))
    monkeypatch.setenv(intake.CLIENTS_ENV, "blueprint-webapp")
    resolved, clients = binding.scene_store()
    assert resolved == root and clients == {"blueprint-webapp"}


def test_scene_store_resolves_from_autoprovision_config(tmp_path, monkeypatch):
    _clear(monkeypatch)
    root = tmp_path / "scene-intents"
    root.mkdir()
    config = tmp_path / "controls-autoprovision.json"
    config.write_text(json.dumps({"scene_root": str(root), "trusted_clients": ["blueprint-webapp"]}))
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_CONTROLS_AUTOPROVISION_CONFIG", str(config))
    resolved, clients = binding.scene_store()
    assert resolved == root and clients == {"blueprint-webapp"}


def test_scene_store_refuses_when_no_owner_store_is_configured(monkeypatch):
    _clear(monkeypatch)
    with pytest.raises(binding.ScenePolicyBindingError, match="owner_store_missing"):
        binding.scene_store()
