"""Prove installed configuration drives the real service without paid reservations."""
from __future__ import annotations

import json
import os
from pathlib import Path
import pwd

import pytest

from blueprint_pipeline import task_evaluation_scene_preparation_installation as installation
from blueprint_pipeline import task_evaluation_scene_configuration_submission_publication as publication
from blueprint_pipeline import task_evaluation_launch_preparation_worker as worker
from blueprint_pipeline.task_evaluation_scene_preparation_service import run_preparation_service
from blueprint_pipeline.task_evaluation_scene_intake import stage_scene_intent
from blueprint_pipeline.task_evaluation_public_scene_attempt_factory import record
from tests.test_task_evaluation_completed_scene_progression import _config
from tests.test_task_evaluation_scene_configuration_submission import SHA
from tests.test_task_evaluation_scene_configuration_submission_publication import Store

ACCOUNT = pwd.getpwuid(os.geteuid()).pw_name
pytestmark = pytest.mark.slow


def _installed(tmp_path, monkeypatch):
    old_config_path, intent_id, old_intents, now = _config(tmp_path, monkeypatch, source_kind="mesh", real_destination=True)
    old = json.loads(old_config_path.read_text())
    release = json.loads(Path(old["release_binding_path"]).read_text())
    machinery = json.loads(Path(old["completed_source_machinery_path"]).read_text())
    state = tmp_path / "production/pipeline-control-plane"
    inputs = tmp_path / "production/task-evaluation-inputs"
    bootstrap = installation.build_bootstrap(destination_catalog=machinery["destination_catalog"],
        config_root=tmp_path / "etc", state_root=state, inputs_root=inputs,
        capture_store_root=old["capture_store_root"], running_repo_root=tmp_path / "repo", service_account=ACCOUNT)
    path = tmp_path / "bootstrap.json"
    path.write_text(json.dumps(bootstrap))
    path.chmod(0o640)
    receipt = installation.install_scene_preparation(bootstrap_path=path)
    config_path = Path(receipt["config"]["path"])
    config = json.loads(config_path.read_text())
    for line in Path(receipt["environment"]["path"]).read_text().splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            monkeypatch.setenv(key, value)
    deploy_path = Path(release["deploy_receipt"]["path"])
    deploy = json.loads(deploy_path.read_text())
    deploy.update(status="deployed", release_path=release["repo_root"],
        scene_configuration_environment={**release["release_environment"], "credential_values_recorded": False})
    deploy["release_provenance"].update(release["release_provenance"])
    deploy_path.write_text(json.dumps(deploy))
    original = json.loads((old_intents / intent_id / "intent.json").read_text())
    stage_scene_intent(value=original["request"], queue_root=config["intent_root"], authenticated_client="blueprint-webapp",
                       trusted_clients={"blueprint-webapp"}, now=now)
    return path, receipt, config, intent_id


def test_installer_is_idempotent_and_refuses_unmanaged_configuration(tmp_path, monkeypatch):
    path, receipt, config, _ = _installed(tmp_path, monkeypatch)
    assert installation.install_scene_preparation(bootstrap_path=path) == receipt
    assert config["activation_enabled"] is False
    assert config["submission_transport"] == "local_owned_queue"
    assert "source_commit" not in config and "release_binding_path" not in config
    config_path = Path(receipt["config"]["path"])
    config_path.write_text('{"user_owned":true}')
    with pytest.raises(ValueError, match="unmanaged_file"):
        installation.install_scene_preparation(bootstrap_path=path)
    assert config_path.read_text() == '{"user_owned":true}'


def test_installed_service_prepares_a_mesh_then_restarts_without_a_paid_attempt(tmp_path, monkeypatch):
    _, receipt, config, intent_id = _installed(tmp_path, monkeypatch)
    class ReadbackStore(Store):
        def get_object(self, *, Bucket, Key):
            result = super().get_object(Bucket=Bucket, Key=Key)
            return {**result, "ContentLength": len(self.objects[Key])}
    store = ReadbackStore()
    monkeypatch.setattr(publication, "_verified_checkout_head", lambda: SHA)
    monkeypatch.setattr(publication, "_s3_client", lambda _: store)
    monkeypatch.setattr(worker, "_s3_client", lambda _: store)
    result = run_preparation_service(config_path=receipt["config"]["path"])
    assert result["scene_progression"]["results"][0]["phase"] == "construction_prepared", result
    intent = Path(config["intent_root"]) / intent_id
    assert list((intent / "attempts").glob("*.json")) == []
    attempts = list((intent / "preparation-attempts").glob("*.json"))
    assert len(attempts) == 1
    attempt = json.loads(attempts[0].read_text())
    assert attempt["maximum_spend_usd"] == 0 and attempt["paid_authority_granted"] is False
    before = list(store.puts)
    projection = record(intent / "progression.json")
    second = run_preparation_service(config_path=receipt["config"]["path"])
    assert second["scene_progression"]["results"][0]["phase"] == "construction_prepared"
    assert record(intent / "progression.json") == projection
    assert store.puts == before
    assert list((intent / "preparation-attempts").glob("*.json")) == attempts


def test_public_scene_mode_config_enables_the_public_source_path(tmp_path, monkeypatch):
    """With public_scene_enabled the installed config admits public_scene source
    intents (Spec A): supported_source_kinds gains public_scene, and the config
    carries public_source_binding_root + machinery_path so the scene-progression
    _source resolver can bind a public-scene intent. The owner-upload default
    (public_scene_enabled off) is unchanged and still refuses public sources.
    """
    machinery_catalog = installation.DEFAULT_PHYSICS_BOUNDS  # sanity: module import
    assert machinery_catalog is not None
    from tests.test_task_evaluation_completed_scene_progression import _config
    old_config_path, _intent_id, _intents, _now = _config(tmp_path, monkeypatch, source_kind="mesh", real_destination=True)
    old = json.loads(old_config_path.read_text())
    machinery = json.loads(Path(old["completed_source_machinery_path"]).read_text())
    state = tmp_path / "prod/pipeline-control-plane"
    inputs = tmp_path / "prod/task-evaluation-inputs"
    bootstrap = installation.build_bootstrap(destination_catalog=machinery["destination_catalog"],
        config_root=tmp_path / "etc", state_root=state, inputs_root=inputs,
        capture_store_root=old["capture_store_root"], running_repo_root=tmp_path / "repo",
        service_account=ACCOUNT, public_scene_enabled=True)
    assert bootstrap["supported_source_kinds"] == ["mesh", "gaussian_splat", "public_scene"]
    path = tmp_path / "bootstrap.json"
    path.write_text(json.dumps(bootstrap)); path.chmod(0o640)
    receipt = installation.install_scene_preparation(bootstrap_path=path)
    config = json.loads(Path(receipt["config"]["path"]).read_text())
    assert "public_scene" in config["supported_source_kinds"]
    assert config.get("public_source_binding_root")
    assert config.get("machinery_path")
    assert Path(config["public_source_binding_root"]).is_dir()


def test_owner_upload_default_still_refuses_public_scene(tmp_path, monkeypatch):
    """The default (public_scene_enabled off) config must not admit public_scene,
    so a public source under a mesh-only config fails with the typed refusal."""
    from tests.test_task_evaluation_completed_scene_progression import _config
    old_config_path, _intent_id, _intents, _now = _config(tmp_path, monkeypatch, source_kind="mesh", real_destination=True)
    old = json.loads(old_config_path.read_text())
    machinery = json.loads(Path(old["completed_source_machinery_path"]).read_text())
    bootstrap = installation.build_bootstrap(destination_catalog=machinery["destination_catalog"],
        config_root=tmp_path / "etc", state_root=tmp_path / "s", inputs_root=tmp_path / "i",
        capture_store_root=old["capture_store_root"], running_repo_root=tmp_path / "repo", service_account=ACCOUNT)
    assert bootstrap["supported_source_kinds"] == ["mesh", "gaussian_splat"]
    assert "public_scene" not in bootstrap["supported_source_kinds"]


def test_cli_public_scene_enabled_flag_builds_a_public_scene_bootstrap(tmp_path, monkeypatch):
    """`--public-scene-enabled` must reach build_bootstrap so the public_scene
    config is reproducible from the CLI (not hand-generated)."""
    captured = {}

    def recording_build(**kwargs):
        captured.update(kwargs)
        return {"schema_version": installation.BOOTSTRAP_SCHEMA, "service_account": ACCOUNT}

    monkeypatch.setattr(installation, "build_bootstrap", recording_build)
    # Stub the simready read so the CLI reaches build_bootstrap without a real asset.
    monkeypatch.setattr(installation, "read", lambda *a, **k: {"destination_identity": {"id": "tray", "version": "v1"}})
    monkeypatch.setattr(installation, "record", lambda p: {"path": str(p), "sha256": "sha256:" + "0" * 64, "size_bytes": 1})
    monkeypatch.setattr(installation, "_managed_json", lambda *a, **k: None)
    sim = tmp_path / "sim.json"; sim.write_text("{}")
    installation.main(["--bootstrap", str(tmp_path / "bootstrap.json"),
                       "--destination-simready", str(sim), "--public-scene-enabled"])
    assert captured.get("public_scene_enabled") is True


def _install_bootstrap(tmp_path, machinery, capture_store_root, authorized, root):
    bootstrap = installation.build_bootstrap(destination_catalog=machinery["destination_catalog"],
        config_root=root / "etc", state_root=root / "state", inputs_root=root / "inputs",
        capture_store_root=capture_store_root, running_repo_root=root / "repo",
        service_account=ACCOUNT, activation_authorized=authorized)
    path = root / "bootstrap.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bootstrap))
    path.chmod(0o640)
    return installation.install_scene_preparation(bootstrap_path=path)


def test_activation_authorized_enables_activation_on_the_progression_config(tmp_path, monkeypatch):
    """A3: production runs `task_evaluation_scene_progression --config <this file>`
    directly (blueprint-task-evaluation-scene-progression.service), so activation is
    a typed, separately-admitted mode ON THAT ONE config. Without authorization the
    config is preparation-only (activation_enabled False, no activation roots);
    with --activation-authorized it carries activation_enabled True +
    activation_intent_root + project_spend_current_path, so _advance_intent
    provisions the scene-configuration activation intent. Off by default."""
    old_config_path, _iid, _intents, _now = _config(tmp_path, monkeypatch, source_kind="mesh", real_destination=True)
    old = json.loads(old_config_path.read_text())
    machinery = json.loads(Path(old["completed_source_machinery_path"]).read_text())

    default_config = json.loads(Path(_install_bootstrap(
        tmp_path, machinery, old["capture_store_root"], False, tmp_path / "default")["config"]["path"]).read_text())
    assert default_config["activation_enabled"] is False
    assert "activation_intent_root" not in default_config
    assert "project_spend_current_path" not in default_config

    auth_config = json.loads(Path(_install_bootstrap(
        tmp_path, machinery, old["capture_store_root"], True, tmp_path / "authorized")["config"]["path"]).read_text())
    assert auth_config["activation_enabled"] is True
    assert Path(auth_config["activation_intent_root"]).is_dir()
    assert auth_config.get("project_spend_current_path")
    # Config-driven: no separate config / wrapper pointer.
    assert "activation_service_config" not in auth_config


def test_preparation_service_wrapper_refuses_the_authorized_activation_config(tmp_path, monkeypatch):
    """The preparation-only service wrapper (a test convenience -- NOT the
    production entrypoint) refuses an activation-enabled config: production runs
    activation via blueprint-task-evaluation-scene-progression.service, which
    execs process_scene_intents on the authorized config directly."""
    import pytest
    from blueprint_pipeline import task_evaluation_scene_preparation_service as svc
    old_config_path, _iid, _intents, _now = _config(tmp_path, monkeypatch, source_kind="mesh", real_destination=True)
    old = json.loads(old_config_path.read_text())
    machinery = json.loads(Path(old["completed_source_machinery_path"]).read_text())
    receipt = _install_bootstrap(tmp_path, machinery, old["capture_store_root"], True, tmp_path / "authorized")
    auth_config = json.loads(Path(receipt["config"]["path"]).read_text())
    assert auth_config["activation_enabled"] is True
    with pytest.raises(ValueError, match="preparation_service_scope_invalid"):
        svc.run_preparation_service(config_path=receipt["config"]["path"])
