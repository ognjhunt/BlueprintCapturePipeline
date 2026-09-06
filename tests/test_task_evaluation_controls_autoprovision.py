"""Real producer/registry rehearsal, with only external publication replaced."""
import json
from functools import partial
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_controls_autoprovision as worker
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_task_evaluation_configured_controls_continuation_provisioning import (
    COMMIT, NOW, TEAM, SCENE_ID, TASK_ID, _preparation, _write,
    _embodiment_camera_template, _payload, _Publisher, _provider_zero,
)
from tests.test_task_evaluation_scene_intake import request
from tests.test_project_spend_reconciliation import _human_baseline
from blueprint_pipeline.project_spend_reconciliation import materialize_project_spend_reconciliation


@pytest.fixture(autouse=True)
def trusted_scene_issuer(monkeypatch):
    monkeypatch.setenv(intake.CLIENTS_ENV, "webapp")


def setup(tmp_path, *, cap=20, attempts=8):
    moment = NOW.timestamp()
    owner = request()
    owner["task"].update(task_id=TASK_ID, robot_binding_id="franka-droid")
    owner["execution"].update(max_total_spend_usd=cap, max_paid_attempts=attempts,
        expires_at_epoch=moment + 7200, allowed_providers=["vast", "openai"])
    owner["consent"]["accepted_at_epoch"] = moment - 1
    scene_root = tmp_path / "scenes"
    staged = intake.stage_scene_intent(value=owner, queue_root=scene_root,
        authenticated_client="webapp", trusted_clients={"webapp"}, now=moment)
    result_path = _preparation(tmp_path)
    envelope_path = tmp_path / "preparations" / "materialized" / result_path.name
    envelope = json.loads(envelope_path.read_text())
    envelope["request"]["scene_intent_digest"] = staged["intent_digest"]
    digest = canonical_digest(envelope["request"])
    envelope["request_digest"] = digest
    envelope["envelope_digest"] = canonical_digest(envelope, digest_field="envelope_digest")
    name = envelope["request"]["preparation_id"] + "-" + digest.removeprefix("sha256:") + ".json"
    _write(envelope_path.parent / name, envelope)
    _write(result_path.parent / name, json.loads(result_path.read_text()))
    link = worker.build_preparation_link(intent_id=staged["intent_id"], intent_digest=staged["intent_digest"],
        preparation_id=envelope["request"]["preparation_id"], request_digest=digest,
        expected_production_commit=COMMIT, team_namespace=TEAM, scene_id=SCENE_ID,
        task_id=TASK_ID, result_filename=name)
    link_path = _write(scene_root / staged["intent_id"] / "preparation-link.json", link)
    def asset(path):
        return {"path": str(path), "digest": worker.producer._sha256(path)}
    baseline, _ = _human_baseline(tmp_path / "baseline.json")
    spend_path = tmp_path / "spend.json"
    materialize_project_spend_reconciliation(baseline_authority_path=baseline,
        posted_reconciliation_paths=[], expected_coverage_ids=[], completeness_reference="fixture",
        authorized_by="u1", authorized_on=NOW.isoformat(), output_path=spend_path)
    runtime = _payload(tmp_path)
    binding = {"robot_asset_usd": asset(_write(tmp_path / "robot.usd", b"#usda 1.0\n")),
        "embodiment_camera_template": asset(_embodiment_camera_template(tmp_path)),
        "runtime_source_payload_dir": str(runtime), "runtime_digest": worker.payload_digest(runtime),
        "expected_production_commit": COMMIT,
        "project_spend_reconciliation": asset(spend_path),
        "project_spend_observed_at_epoch": moment,
        "openai_project_id": "proj_test", "openai_api_key_id": "key_test"}
    catalog = worker._seal({"schema_version": worker.CATALOG_SCHEMA,
        "bindings": {"franka-droid": binding}}, "catalog_digest")
    publisher = _Publisher()
    return dict(link_path=link_path, scene_root=scene_root,
        preparation_queue_root=tmp_path / "preparations", catalog=catalog,
        controls_root=tmp_path / "controls", intent_root=tmp_path / "registry",
        profile_dir=tmp_path / "profiles", expected_production_commit=COMMIT,
        trusted_clients={"webapp"}, now=moment, service_group=None,
        provisioner=partial(worker.producer.provision_configured_controls_continuation,
            artifact_publisher=publisher.artifact, layer_publisher=publisher.layers_of,
            provider_zero_collector=_provider_zero))


def test_real_preparation_to_installed_controls_and_retry(tmp_path):
    kwargs = setup(tmp_path)
    receipt = worker.provision_link(**kwargs)
    assert receipt["status"] == "installed"
    installed = json.loads(Path(receipt["installation"]["registry_path"]).read_text())
    assert installed["intent_digest"] == receipt["provisioning"]["intent_digest"]
    for phase in installed["phases"].values():
        authority = json.loads(Path(phase["authorization_path"]).read_text())
        assert authority["authorized_by"] == "u1"
        assert authority["reference"] == "scene-intent:" + receipt["owner_intent_digest"]
        assert authority["standing_authorization_expires_at"] == "2026-09-05T06:05:00+00:00"
    rows = list((kwargs["link_path"].parent / "attempts").glob("*.json"))
    assert len(rows) == 3
    assert sum(json.loads(p.read_text())["maximum_spend_usd"] for p in rows) == pytest.approx(6.56)
    kwargs["now"] += 60
    assert worker.provision_link(**kwargs) == receipt
    assert len(list((kwargs["link_path"].parent / "attempts").glob("*.json"))) == 3


@pytest.mark.parametrize("mutation,error", [
    ("revoked", "authority_revoked"), ("expired", "authority_expired"),
    ("release", "release_mismatch"), ("catalog", "catalog_invalid"),
    ("robot", "asset_invalid"), ("runtime", "runtime_digest_mismatch"),
    ("issuer", "owner_intent_invalid"), ("owner", "preparation_identity_mismatch"),
    ("result", "digest_invalid"), ("missing_camera", "asset_invalid"),
    ("stale_spend", "project_spend_stale"),
])
def test_refuses_before_provisioning(tmp_path, mutation, error):
    kwargs = setup(tmp_path)
    directory = kwargs["link_path"].parent
    binding = kwargs["catalog"]["bindings"]["franka-droid"]
    if mutation == "revoked":
        _write(directory / "revoked.json", {})
    elif mutation == "expired":
        kwargs["now"] += 7201
    elif mutation == "release":
        kwargs["expected_production_commit"] = "d" * 40
    elif mutation == "catalog":
        binding["openai_project_id"] = "wrong"
    elif mutation == "robot":
        Path(binding["robot_asset_usd"]["path"]).write_text("changed")
    elif mutation == "missing_camera":
        Path(binding["embodiment_camera_template"]["path"]).unlink()
    elif mutation == "runtime":
        _write(Path(binding["runtime_source_payload_dir"]) / "extra", b"changed")
    elif mutation == "issuer":
        kwargs["trusted_clients"] = {"another"}
    elif mutation == "owner":
        link = json.loads(kwargs["link_path"].read_text())
        link["scene_id"] = "another"
        _write(kwargs["link_path"], worker._seal(link, "link_digest"))
    elif mutation == "result":
        link = json.loads(kwargs["link_path"].read_text())
        _write(kwargs["preparation_queue_root"] / "results" / link["result_filename"], {})
    elif mutation == "stale_spend":
        kwargs["now"] += 901
    def unexpected(**_):
        pytest.fail("producer must not run")
    kwargs["provisioner"] = unexpected
    with pytest.raises((ValueError, OSError), match=error):
        worker.provision_link(**kwargs)
    assert not (directory / "attempts").exists()


def test_partial_install_recovery_keeps_reservations_and_authority(tmp_path):
    kwargs = setup(tmp_path)
    def fail_install(**_):
        raise OSError("interrupted before install")
    with pytest.raises(OSError, match="interrupted"):
        worker.provision_link(**kwargs, installer=fail_install)
    kwargs["now"] += 60
    assert worker.provision_link(**kwargs)["status"] == "installed"
    assert len(list((kwargs["link_path"].parent / "attempts").glob("*.json"))) == 3


def test_allocation_caps_cover_all_three_consumers(tmp_path):
    kwargs = setup(tmp_path, cap=4, attempts=3)
    with pytest.raises(ValueError, match="spend_cap_exhausted"):
        worker.provision_link(**kwargs)
    assert not kwargs["intent_root"].exists()


def test_absent_result_waits_without_reservation(tmp_path):
    kwargs = setup(tmp_path)
    link = json.loads(kwargs["link_path"].read_text())
    (kwargs["preparation_queue_root"] / "results" / link["result_filename"]).unlink()
    assert worker.provision_link(**kwargs)["status"] == "waiting_for_preparation_result"
    assert not (kwargs["link_path"].parent / "attempts").exists()


def test_link_rejects_path_escape(tmp_path):
    kwargs = setup(tmp_path)
    link = json.loads(kwargs["link_path"].read_text())
    link["result_filename"] = "../result.json"
    with pytest.raises(ValueError, match="filename_invalid"):
        worker.validate_preparation_link(worker._seal(link, "link_digest"))


def test_current_spend_pointer_is_read_when_preparation_arrives(tmp_path):
    kwargs = setup(tmp_path)
    binding = kwargs["catalog"]["bindings"]["franka-droid"]
    pointer = worker._seal({"schema_version": "task_evaluation_project_spend_current.v1",
        **binding.pop("project_spend_reconciliation"),
        "observed_at_epoch": binding.pop("project_spend_observed_at_epoch")}, "receipt_digest")
    binding["project_spend_current_path"] = str(_write(tmp_path / "current-spend.json", pointer))
    kwargs["catalog"] = worker._seal(kwargs["catalog"], "catalog_digest")
    receipt = worker.provision_link(**kwargs)
    # A refreshed current pointer cannot alter this attempt's retained baseline.
    _write(tmp_path / "current-spend.json", {"invalid": "new pointer"})
    assert worker.provision_link(**kwargs) == receipt


def test_production_default_and_worker_order(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_configured_controls_progression_worker as progression
    kwargs = setup(tmp_path)
    real_with_external_fakes = kwargs.pop("provisioner")
    monkeypatch.setattr(worker.producer, "provision_configured_controls_continuation", real_with_external_fakes)
    # With no producer argument, the worker reaches the canonical producer seam.
    assert worker.provision_link(**kwargs)["status"] == "installed"
    monkeypatch.setenv(worker.CONFIG_ENV, "catalog-config.json")
    monkeypatch.setattr(progression, "running_release_commit", lambda: COMMIT)
    events = []
    def process(path, **kw):
        events.append("provision")
        assert kw["expected_production_commit"] == COMMIT
        return [{"status": "installed"}]
    monkeypatch.setattr(worker, "process_config", process)
    monkeypatch.setattr(progression.scene_configuration_activation, "progression_rows",
        lambda **_: events.append("activation") or [])
    result = progression.process_plans(plan_root=tmp_path / "plans",
        launch_state_root=tmp_path / "launches", progression_root=tmp_path / "progression",
        preparation_queue_root=tmp_path / "preparations", activation_queue_root=tmp_path / "activation",
        scene_configuration_activation_intent_root=tmp_path / "scene-activations")
    assert result["status"] == "completed"
    assert events == ["provision", "activation"]


def test_expired_installed_intent_cannot_continue_progression(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_configured_controls_progression_worker as progression
    monkeypatch.setenv(worker.CONFIG_ENV, "catalog-config.json")
    monkeypatch.setattr(progression, "running_release_commit", lambda: COMMIT)
    monkeypatch.setattr(worker, "process_config", lambda *a, **kw: [
        {"status": "controls_autoprovision_refused", "blocker": "authority_expired",
         "blocked_scene_key": [TEAM, SCENE_ID, TASK_ID]}])
    events = []
    monkeypatch.setattr(progression.scene_configuration_activation, "progression_rows",
        lambda **kw: events.append(kw["blocked_scene_keys"]) or [])
    _write(tmp_path / "launches" / "revoked" / "launch_profile.json", {
        "task_evaluation_run": {"team_namespace": TEAM, "scene_id": SCENE_ID, "task_id": TASK_ID}})
    _write(tmp_path / "launches" / "valid" / "launch_profile.json", {
        "task_evaluation_run": {"team_namespace": TEAM, "scene_id": "other-scene", "task_id": TASK_ID}})
    _write(tmp_path / "plans" / "revoked.json", {"source_launch_id": "revoked"})
    _write(tmp_path / "plans" / "valid.json", {"source_launch_id": "valid"})
    monkeypatch.setattr(progression, "advance_configured_controls_plan",
        lambda **kw: events.append(Path(kw["plan_path"]).stem) or {"status": "awaiting_controls"})
    result = progression.process_plans(plan_root=tmp_path / "plans",
        launch_state_root=tmp_path / "launches", scene_configuration_activation_intent_root=tmp_path / "intents",
        preparation_queue_root=tmp_path / "preparations", activation_queue_root=tmp_path / "activations",
        progression_root=tmp_path / "progression")
    assert result["status"] == "blocked"
    assert events == [{(TEAM, SCENE_ID, TASK_ID)}, "valid"]


def test_queued_dispatch_owner_guard_reopens_current_consent(tmp_path):
    kwargs = setup(tmp_path)
    config = _write(tmp_path / "config.json", {"scene_root": str(kwargs["scene_root"]), "trusted_clients": ["webapp"]})
    digest = json.loads(kwargs["link_path"].read_text())["intent_digest"]
    assert worker.owner_authority_blocker(config, scene_intent_digest=digest, now=NOW.timestamp()) is None
    _write(kwargs["link_path"].parent / "revoked.json", {})
    assert "authority_revoked" in worker.owner_authority_blocker(config, scene_intent_digest=digest, now=NOW.timestamp())


def test_activation_filter_allows_other_scene(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_configuration_activation_automation as activation
    queue = tmp_path / "preparations"
    files = []
    for scene in (SCENE_ID, "other-scene"):
        files.append(_write(queue / "results" / (scene + ".json"), {}))
        _write(queue / "materialized" / (scene + ".json"), {"request": {
            "team_namespace": TEAM, "scene": {"identity": {"id": scene}},
            "task": {"identity": {"id": TASK_ID}}}})
    monkeypatch.setattr(activation, "_awaiting_scene_configurations", lambda root: files)
    observed = []
    monkeypatch.setattr(activation, "advance_scene_configuration_activation", lambda **kw:
        observed.append(kw["preparation_result_path"].stem) or {"status": "awaiting_configuration"})
    rows = activation.process_scene_configuration_activations(preparation_queue_root=queue,
        activation_queue_root=tmp_path / "activations", progression_root=tmp_path / "progression",
        intent_root=tmp_path / "intents", profile_dir=tmp_path / "profiles",
        standing_authorization_dir=tmp_path / "authorizations", blocked_scene_keys={(TEAM, SCENE_ID, TASK_ID)})
    assert observed == ["other-scene"]
    assert rows[0]["blockers"] == ["scene_configuration_owner_authority_refused"]


def test_scene_intent_uses_cross_runtime_number_encoding(tmp_path, monkeypatch):
    from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
    monkeypatch.setattr(intake, "canonical_digest", cross_runtime_canonical_digest)
    kwargs = setup(tmp_path, cap=20.0)
    receipt = worker.provision_link(**kwargs)
    assert receipt["status"] == "installed"
    assert worker.provision_link(**kwargs) == receipt
