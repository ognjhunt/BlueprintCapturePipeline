"""A dated public machinery path survives installation without rewriting history."""
import json
import os
from pathlib import Path
import pwd

import pytest

from blueprint_pipeline import task_evaluation_scene_preparation_installation as installation
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

ACCOUNT = pwd.getpwuid(os.geteuid()).pw_name


def _bootstrap(tmp_path, monkeypatch, **options):
    # Destination geometry is orthogonal and has its own real-install tests.
    monkeypatch.setattr(installation, "validate_destination_catalog", lambda rows: rows)
    return installation.build_bootstrap(destination_catalog=[], config_root=tmp_path / "etc",
        state_root=tmp_path / "state", inputs_root=tmp_path / "inputs", capture_store_root=tmp_path / "captures",
        running_repo_root=tmp_path / "repo", service_account=ACCOUNT, **options)


def _save(tmp_path, bootstrap):
    bootstrap["bootstrap_digest"] = canonical_digest(bootstrap, digest_field="bootstrap_digest")
    path = tmp_path / "bootstrap.json"
    path.write_text(json.dumps(bootstrap))
    path.chmod(0o640)
    return path


def test_dated_machinery_is_installed_idempotently_without_changing_old_files_or_defaults(tmp_path, monkeypatch):
    root = tmp_path / "etc"
    root.mkdir()
    old = root / "task-evaluation-public-scene-machinery.json"
    new = root / "task-evaluation-public-scene-machinery-20260912.json"
    old.write_text('{"retained": "old credential binding"}')
    new.write_text('{"retained": "new credential binding"}')
    before = {p: p.read_bytes() for p in (old, new)}
    bootstrap = _bootstrap(tmp_path, monkeypatch, public_scene_enabled=True, public_scene_machinery_path=new)
    assert bootstrap["public_scene_machinery_path"] == str(new)
    path = _save(tmp_path, bootstrap)
    receipt = installation.install_scene_preparation(bootstrap_path=path)
    assert installation.install_scene_preparation(bootstrap_path=path) == receipt
    config = json.loads(Path(receipt["config"]["path"]).read_text())
    assert config["machinery_path"] == str(new)
    assert config["activation_enabled"] is False
    assert config["maximum_http_submission_attempts"] == 2
    assert config["require_whole_chain_capacity"] is True
    assert config["submission_transport"] == "local_owned_queue"
    assert json.loads(Path(receipt["machinery"]["path"]).read_text())["maximum_preparation_spend_usd"] == 0
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_absent_override_retains_original_canonical_path(tmp_path, monkeypatch):
    bootstrap = _bootstrap(tmp_path, monkeypatch, public_scene_enabled=True)
    assert "public_scene_machinery_path" not in bootstrap
    receipt = installation.install_scene_preparation(bootstrap_path=_save(tmp_path, bootstrap))
    config = json.loads(Path(receipt["config"]["path"]).read_text())
    assert config["machinery_path"] == str(tmp_path / "etc/task-evaluation-public-scene-machinery.json")


@pytest.mark.parametrize("fault", ["relative", "missing", "directory", "symlink", "parent_symlink", "disabled"])
@pytest.mark.parametrize("boundary", ["build", "install"])
def test_invalid_override_refuses_before_installer_writes(tmp_path, monkeypatch, fault, boundary):
    target = tmp_path / "machinery.json"
    target.write_text("{}")
    path = target
    if fault == "relative":
        path = Path("relative.json")
    elif fault == "missing":
        path = tmp_path / "missing.json"
    elif fault == "directory":
        path = tmp_path
    elif fault == "symlink":
        path = tmp_path / "symlink.json"
        path.symlink_to(target)
    elif fault == "parent_symlink":
        directory = tmp_path / "linked-parent"
        directory.symlink_to(tmp_path, target_is_directory=True)
        path = directory / target.name
    enabled = fault != "disabled"
    if boundary == "build":
        with pytest.raises(ValueError, match="(path_unsafe|public_scene_machinery)"):
            _bootstrap(tmp_path, monkeypatch, public_scene_enabled=enabled, public_scene_machinery_path=path)
    else:
        bootstrap = _bootstrap(tmp_path, monkeypatch, public_scene_enabled=enabled)
        bootstrap["public_scene_machinery_path"] = str(path)
        saved = _save(tmp_path, bootstrap)
        monkeypatch.setattr(installation, "_managed_json", lambda *a, **k: pytest.fail("invalid override wrote config"))
        with pytest.raises(ValueError, match="(path_unsafe|public_scene_machinery)"):
            installation.install_scene_preparation(bootstrap_path=saved)
        assert not (tmp_path / "state").exists()


def test_cli_updates_existing_bootstrap_with_new_path_only(tmp_path, monkeypatch):
    machinery = tmp_path / "machinery-20260912.json"
    machinery.write_text("{}")
    original = _bootstrap(tmp_path, monkeypatch, public_scene_enabled=True, activation_authorized=True)
    path = _save(tmp_path, original)
    installation.main(["--bootstrap", str(path), "--public-scene-machinery-path", str(machinery)])
    updated = json.loads(path.read_text())
    assert updated == {**original, "public_scene_machinery_path": str(machinery),
                       "bootstrap_digest": updated["bootstrap_digest"]}
    assert updated["bootstrap_digest"] == canonical_digest(updated, digest_field="bootstrap_digest")
