"""Owner-mode preflight: selection scope, owner store, and autoprovision assets.

The chain look-ahead must name a missing scope, an unresolvable owner store, and
a missing/broken controls-autoprovision config or its assets as blockers before a
dispatch selects a wrong or empty row.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_production_chain_preflight as preflight

LAUNCH = "blueprint-task-evaluation-launch-dispatcher.service"
POLICY = "blueprint-task-evaluation-policy-canary-dispatcher.service"
CONTROLS = "blueprint-task-evaluation-configured-controls-progression.service"
RECONCILER = "blueprint-task-evaluation-launch-reconciler.service"
TERMINAL_INDEX_ENV = {
    "BLUEPRINT_TASK_EVALUATION_POLICY_CANARY_DISPATCH_ROOT": "/var/lib/blueprint/pipeline-control-plane/task-evaluation-policy-canaries",
    "BLUEPRINT_TASK_EVALUATION_TERMINAL_RESULT_ROOT": "/var/lib/blueprint/task-evaluation-inputs/task-evaluation-terminal-results",
    "BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT": "/var/lib/blueprint/pipeline-control-plane/task-evaluation-scene-intents",
}
IDS = (os.getuid(), os.getgid())


@pytest.fixture(autouse=True)
def _stub_active_release(monkeypatch):
    # R7: the controls preflight binds the catalog at the active release via the
    # shared resolve_robot_catalog. On a dev host there is no deployed release, so
    # stub a valid running commit; the catalog binds by content identity to it.
    monkeypatch.setattr(preflight, "active_release", lambda: (None, "a" * 40, []))


def _write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def _unit(env, env_files=()):
    return {"effective_environment": dict(env), "properties": {"EnvironmentFiles": list(env_files)}}


def _controls_config(tmp_path: Path) -> Path:
    assets = tmp_path / "assets"
    robot = assets / "robot.usd"
    camera = assets / "camera.json"
    runtime = assets / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    robot.write_text("#usda 1.0\n")
    camera.write_text("{}")
    (runtime / "packet.json").write_text("{}")
    # R7: the installed catalog is the sealed CONTENT catalog the real installer
    # writes (task_evaluation_controls_robot_content_catalog.v1) -- NOT a resolved
    # catalog. The preflight must bind it with the same resolver the consumer uses
    # (accepting the content schema + validating the seal + asset/runtime digests).
    import hashlib
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.task_evaluation_controls_autoprovision import payload_digest

    def _sha(path):
        return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()

    content = {"schema_version": "task_evaluation_controls_robot_content_catalog.v1",
               "bindings": {"franka-droid": {
                   "robot_asset_usd": {"path": str(robot), "digest": _sha(robot), "size_bytes": robot.stat().st_size},
                   "embodiment_camera_template": {"path": str(camera), "digest": _sha(camera),
                                                  "size_bytes": camera.stat().st_size},
                   "runtime_source_payload_dir": str(runtime), "runtime_digest": payload_digest(runtime),
                   "openai_project_id": "p", "openai_api_key_id": "k"}}}
    content["catalog_digest"] = canonical_digest(content, digest_field="catalog_digest")
    catalog_path = _write_json(tmp_path / "etc" / "catalog.json", content)
    scene_root = tmp_path / "store" / "scene-intents"
    scene_root.mkdir(parents=True, exist_ok=True)
    (tmp_path / "store" / "preps").mkdir(parents=True, exist_ok=True)
    controls_root = tmp_path / "var" / "controls"
    controls_root.mkdir(parents=True, exist_ok=True)
    intent_root = tmp_path / "etc" / "intents"
    intent_root.mkdir(parents=True, exist_ok=True)
    intent_root.chmod(0o770)
    profile_dir = tmp_path / "etc" / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    config = {"scene_root": str(scene_root), "preparation_queue_root": str(tmp_path / "store" / "preps"),
              "controls_root": str(controls_root), "intent_root": str(intent_root),
              "profile_dir": str(profile_dir), "robot_catalog_path": str(catalog_path),
              "trusted_clients": ["blueprint-webapp"], "service_group": "blueprint"}
    return _write_json(tmp_path / "etc" / "controls-autoprovision.json", config)


def _wired_units(tmp_path, config_path):
    scene_root = json.loads(config_path.read_text())["scene_root"]
    store_env = {"BLUEPRINT_TASK_EVALUATION_DISPATCH_OWNER_SCOPE": "persistent_owner_only",
                 "BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT": scene_root,
                 "BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS": "blueprint-webapp"}
    return {
        LAUNCH: _unit(store_env),
        POLICY: _unit(store_env),
        CONTROLS: _unit({**store_env,
                         "BLUEPRINT_TASK_EVALUATION_CONTROLS_AUTOPROVISION_CONFIG": str(config_path)}),
    }


def _blockers(findings):
    return [f["code"] for f in findings if f["severity"] == "blocker"]


def test_owner_mode_fully_wired_has_no_blockers(tmp_path):
    config_path = _controls_config(tmp_path)
    units = _wired_units(tmp_path, config_path)
    assert _blockers(preflight.owner_scope_checks(units, IDS)) == []


def test_default_scope_is_a_blocker_naming_the_override_source(tmp_path):
    config_path = _controls_config(tmp_path)
    units = _wired_units(tmp_path, config_path)
    hold = tmp_path / "override.env"
    hold.write_text("BLUEPRINT_TASK_EVALUATION_DISPATCH_OWNER_SCOPE=all_authorized\n")
    units[LAUNCH]["effective_environment"]["BLUEPRINT_TASK_EVALUATION_DISPATCH_OWNER_SCOPE"] = "all_authorized"
    units[LAUNCH]["properties"]["EnvironmentFiles"] = [f"{hold} (ignore_errors=yes)"]
    findings = preflight.owner_scope_checks(units, IDS)
    [scope] = [f for f in findings if f["code"] == "dispatch_owner_scope_not_persistent_owner_only"]
    assert scope["unit"] == LAUNCH and scope["value"] == "all_authorized"
    assert scope["overridden_by"] == [str(hold)]


def test_missing_owner_store_refuses_before_allocator(tmp_path):
    # Neither the scene-intake root nor an autoprovision config: scene_store() cannot resolve.
    units = {LAUNCH: _unit({"BLUEPRINT_TASK_EVALUATION_DISPATCH_OWNER_SCOPE": "persistent_owner_only"})}
    codes = _blockers(preflight.owner_scope_checks(units, IDS))
    assert "owner_store_unresolvable" in codes
    assert "owner_store_trusted_clients_unset" in codes


def test_controls_config_unset_in_owner_mode_is_a_blocker(tmp_path):
    config_path = _controls_config(tmp_path)
    units = _wired_units(tmp_path, config_path)
    del units[CONTROLS]["effective_environment"]["BLUEPRINT_TASK_EVALUATION_CONTROLS_AUTOPROVISION_CONFIG"]
    assert "controls_autoprovision_config_unset" in _blockers(preflight.owner_scope_checks(units, IDS))


def test_controls_config_set_but_missing_file_fails_closed_blocker(tmp_path):
    config_path = _controls_config(tmp_path)
    units = _wired_units(tmp_path, config_path)
    units[CONTROLS]["effective_environment"]["BLUEPRINT_TASK_EVALUATION_CONTROLS_AUTOPROVISION_CONFIG"] = str(
        tmp_path / "etc" / "does-not-exist.json")
    assert "controls_autoprovision_config_unreadable_by_service" in _blockers(
        preflight.owner_scope_checks(units, IDS))


def test_controls_missing_asset_is_a_blocker(tmp_path):
    config_path = _controls_config(tmp_path)
    units = _wired_units(tmp_path, config_path)
    catalog_path = Path(json.loads(config_path.read_text())["robot_catalog_path"])
    catalog = json.loads(catalog_path.read_text())
    Path(catalog["bindings"]["franka-droid"]["robot_asset_usd"]["path"]).unlink()
    findings = preflight.owner_scope_checks(units, IDS)
    asset = [f for f in findings if f["code"] == "controls_autoprovision_asset_missing"]
    assert asset and asset[0]["asset"] == "robot_asset_usd"


def test_unsealed_catalog_is_a_blocker(tmp_path):
    # R7: an unsealed catalog is refused by the shared resolver (unbindable). A
    # SEALED content catalog is accepted (that is what the installer writes).
    config_path = _controls_config(tmp_path)
    units = _wired_units(tmp_path, config_path)
    catalog_path = Path(json.loads(config_path.read_text())["robot_catalog_path"])
    catalog = json.loads(catalog_path.read_text())
    catalog.pop("catalog_digest", None)  # unsealed
    catalog_path.write_text(json.dumps(catalog))
    assert "controls_autoprovision_robot_catalog_unbindable" in _blockers(preflight.owner_scope_checks(units, IDS))


def test_tampered_seal_is_a_blocker(tmp_path):
    # R7: a catalog whose recorded seal does not match its bytes is refused.
    config_path = _controls_config(tmp_path)
    units = _wired_units(tmp_path, config_path)
    catalog_path = Path(json.loads(config_path.read_text())["robot_catalog_path"])
    catalog = json.loads(catalog_path.read_text())
    catalog["catalog_digest"] = "sha256:" + "0" * 64  # seal no longer matches
    catalog_path.write_text(json.dumps(catalog))
    assert "controls_autoprovision_robot_catalog_unbindable" in _blockers(preflight.owner_scope_checks(units, IDS))


def test_asset_digest_mismatch_is_a_blocker(tmp_path):
    # R7: an asset whose file bytes no longer match the sealed digest is refused by
    # the shared resolver (resolve_robot_catalog._asset), surfaced as unbindable.
    config_path = _controls_config(tmp_path)
    units = _wired_units(tmp_path, config_path)
    catalog_path = Path(json.loads(config_path.read_text())["robot_catalog_path"])
    catalog = json.loads(catalog_path.read_text())
    Path(catalog["bindings"]["franka-droid"]["robot_asset_usd"]["path"]).write_text("#usda 1.0\n; tampered\n")
    assert "controls_autoprovision_robot_catalog_unbindable" in _blockers(preflight.owner_scope_checks(units, IDS))


@pytest.mark.skipif(os.getuid() == 0, reason="root bypasses discretionary mode bits")
def test_controls_intent_root_not_writable_is_a_blocker(tmp_path):
    config_path = _controls_config(tmp_path)
    units = _wired_units(tmp_path, config_path)
    Path(json.loads(config_path.read_text())["intent_root"]).chmod(0o500)
    try:
        assert "controls_autoprovision_root_not_writable_by_service" in _blockers(
            preflight.owner_scope_checks(units, IDS))
    finally:
        Path(json.loads(config_path.read_text())["intent_root"]).chmod(0o770)


def test_launch_reconciler_with_terminal_index_roots_has_no_blockers(tmp_path):
    # R8: the launch reconciler tick files owner terminal receipts; a fully wired
    # unit (dispatch root + terminal result root + intake root) is clean.
    config_path = _controls_config(tmp_path)
    units = {**_wired_units(tmp_path, config_path), RECONCILER: _unit(TERMINAL_INDEX_ENV)}
    assert _blockers(preflight.owner_scope_checks(units, IDS)) == []


@pytest.mark.parametrize("missing", sorted(TERMINAL_INDEX_ENV))
def test_launch_reconciler_missing_a_terminal_index_root_is_a_blocker(tmp_path, missing):
    # Without any one root the retention duty is explicitly not configured and a
    # completed owner run never closes out -- named before a dispatch, not after.
    config_path = _controls_config(tmp_path)
    env = {name: value for name, value in TERMINAL_INDEX_ENV.items() if name != missing}
    units = {**_wired_units(tmp_path, config_path), RECONCILER: _unit(env)}
    findings = [f for f in preflight.owner_scope_checks(units, IDS) if f["code"] == "terminal_index_root_unset"]
    assert [f["variable"] for f in findings] == [missing]
    assert findings[0]["unit"] == RECONCILER and findings[0]["severity"] == "blocker"
