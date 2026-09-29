# ruff: noqa: F811

"""Actual fixed default-off installation state, never an owner grant."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_experiment_retirement.py
#   src/blueprint_pipeline/control_plane_lane_experiment_installation.py
#   src/blueprint_pipeline/control_plane_lane_experiment_publication.py
import json
import os
from pathlib import Path

import pytest

from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_registered_experiment_issuer import installation  # noqa: F401


def setup(installation, monkeypatch):
    from blueprint_pipeline import control_plane_lane_experiment_birth as birth

    config, settings, store, _ = installation
    settings.update(experiment_creation_enabled=False, experiment_retirement_enabled=False)
    config.write_text(json.dumps(settings))
    (store / ".experiment-authority.lock").unlink()
    store.rmdir()
    monkeypatch.setattr(birth, "_blueprint_identity", lambda: (0, 0))
    monkeypatch.setattr(os, "fchown", lambda *args: None)
    return config, Path(settings["state_root"])


def test_actual_default_off_prepare_creates_only_fixed_metadata_and_preserves_reinstall(
    installation, monkeypatch
):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_retirement as code

    config, state = setup(installation, monkeypatch)
    before = config.read_bytes()
    result = code.prepare_registered_experiment_state(installed_config_path=config)
    assert result == {
        "decision": "prepared",
        "creation_enabled": False,
        "retirement_enabled": False,
        "cache_creation_enabled": False,
    }
    modes = {
        "requests/experiment-records": 0o700,
        "experiment-authority": 0o750,
        "requests/needed-checkpoint-cache-records": 0o700,
        "needed-checkpoint-cache-registration": 0o755,
        "needed-checkpoint-cache-registration/authority": 0o750,
    }
    locks = {
        "requests/experiment-records/.experiment-authority.lock": 0o600,
        "experiment-authority/.authority.lock": 0o640,
        "requests/needed-checkpoint-cache-records/.cache-store.lock": 0o600,
        "needed-checkpoint-cache-registration/authority/.authority.lock": 0o640,
    }
    for name, mode in (modes | locks).items():
        assert (state / name).stat().st_mode & 0o777 == mode
    snapshot = {name: (state / name).stat().st_ino for name in modes | locks}
    for name in locks:
        assert (state / name).read_bytes() == b""
    assert not list(state.rglob("*.json"))
    code.prepare_registered_experiment_state(installed_config_path=config)
    assert snapshot == {name: (state / name).stat().st_ino for name in snapshot}
    assert config.read_bytes() == before


@pytest.mark.parametrize("drift", ["symlink", "mode", "nonempty_lock"])
def test_prepare_refuses_existing_unsafe_state_without_repair(installation, monkeypatch, drift):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_retirement as code

    config, state = setup(installation, monkeypatch)
    public = state / "experiment-authority"
    if drift == "symlink":
        foreign = state / "foreign"
        foreign.mkdir()
        public.symlink_to(foreign, target_is_directory=True)
    else:
        public.mkdir(mode=0o777 if drift == "mode" else 0o750)
        public.chmod(0o777 if drift == "mode" else 0o750)
        if drift == "nonempty_lock":
            (public / ".authority.lock").write_bytes(b"foreign")
            (public / ".authority.lock").chmod(0o640)
    before = public.lstat()
    with pytest.raises(ValueError):
        code.prepare_registered_experiment_state(installed_config_path=config)
    assert public.lstat().st_ino == before.st_ino and public.lstat().st_mode == before.st_mode
    assert not (state / "requests/experiment-records").exists()


def test_fixed_cli_prepare_has_no_root_or_enablement_arguments(installation, monkeypatch, capsys):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_retirement as code

    config, _ = setup(installation, monkeypatch)
    monkeypatch.setattr(code, "INSTALLED_CONFIG_PATH", config)
    assert code.main(["prepare"]) == 0
    assert json.loads(capsys.readouterr().out)["result"]["decision"] == "prepared"
    assert code.main(["prepare", "--root", "/tmp/other"]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "experiment_cli_arguments_invalid"


def test_prepared_state_connects_actual_issuer_and_native_birth_without_manual_metadata(
    installation, monkeypatch
):
    from blueprint_pipeline import control_plane_lane_experiment_retirement as code
    from blueprint_pipeline.control_plane_lane_experiment_birth import create_registered_experiment

    config, state = setup(installation, monkeypatch)
    code.prepare_registered_experiment_state(installed_config_path=config)
    settings = json.loads(config.read_bytes())
    # An explicit protected owner setting is still required. Preparation did not
    # enable anything or create a grant. Native pre-existing lane roots are the
    # actual installation prerequisite; the provisioner never touches payload.
    settings["experiment_creation_enabled"] = True
    config.write_text(json.dumps(settings))
    for selected in ("lane_scratch_work_root", "lane_scratch_inputs_root"):
        root = Path(settings[selected])
        root.mkdir(parents=True, mode=0o750)
        (root / ".lane-scratch.lock").write_bytes(b"")
        (root / ".lane-scratch.lock").chmod(0o600)
        (root / "g1").mkdir(mode=0o750)
    grant = code.issue_experiment_creation_intent(
        principal="operator",
        owner="owner",
        root="work",
        reference_value="prepared-run",
        lease_ttl_seconds=1800,
        participant_profile="local_root_disposable.v1",
        request_records=(),
        installed_config_path=config,
        now=lambda: 1000,
    )
    born = create_registered_experiment(
        grant["intent_id"],
        expected_intent=grant["intent"],
        installed_config_path=config,
        now=lambda: 1100,
    )
    assert Path(born["path"]).is_dir()
    assert (state / "experiment-authority/HEAD.json").is_file()
    assert (state / "experiment-authority" / (grant["intent_id"] + ".birth.json")).is_file()
    assert (
        json.loads((Path(born["path"]) / ".lane-scratch.v1.json").read_bytes())["owner"] == "owner"
    )
