"""Development-only fixed scripted batch registration; no providers or native execution."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.test_operator_door_public_policy_registration import contract, installer, write, ROOT, PROFILE_ROOT
from operator_door import scripted_policy_registration_contract as scripts
from operator_door.requests import validate_request, RequestRefused, required_scope, enqueue
from operator_door.config import DoorConfig
from operator_door.hostinfo import CommandResult
from operator_door.spool_runner import process_spool
from blueprint_pipeline.scripted_robot_baseline_runtime import source_identity
from blueprint_pipeline.scripted_robot_runtime_dispatch import build_policy_spec
from tests.test_scripted_robot_grasp_frames import development_manifest


def request(**changes):
    value = {"schema_version": contract.SCRIPTED_SCHEMA, "installation_id": "development-only-scripted",
        "release_commit": "a"*40, "preparation_authorization_id": "development-only-assets",
        "preparation_publication_digest": "sha256:"+"b"*64, "team_id": contract.SCRIPTED_TEAM_ID,
        "policy_profile": contract.SCRIPTED_PROFILE_ID, "expected_registry_digest": "sha256:"+"c"*64,
        "authorization_reference": "development_only explicit simulation baseline request"}
    value.update(changes)
    value["request_digest"] = contract.digest(value, "request_digest")
    return value


def installed_profile(tmp_path):
    target = tmp_path / "installed-profile"
    target.mkdir()
    for name in ("scripted_robot_baseline_registration_profile.json", "scripted_robot_baseline_license.txt"):
        shutil.copyfile(PROFILE_ROOT / name, target / name)
    code = target / "scripted-baseline-source"
    code.mkdir()
    for name in set(scripts.CODE_FILES) | set(scripts.VARIANT_CODE_FILES) | set(scripts.INDUSTRIAL_CODE_FILES) | set(scripts.YASKAWA_CODE_FILES) | set(scripts.KUKA_CODE_FILES) | set(scripts.PUBLIC_ARM_CODE_FILES):
        shutil.copyfile(ROOT / "src/blueprint_pipeline" / name, code / name)
    return target


@pytest.fixture()
def batch(tmp_path):
    profile_root = installed_profile(tmp_path)
    profile = scripts.load_profile(profile_root)
    models, state = tmp_path / "models", tmp_path / "proposals"
    publication = {"status": "completed", "claim_ceiling": "robot_asset_import_only", "candidate_policy_queried": False,
        "task_executed": False, "results": []}
    originals, model_records = {}, {}
    for body in contract.SCRIPTED_BODIES:
        directory = models / body / ("d"*64)
        prepared = directory / "prepared"
        prepared.mkdir(parents=True)
        model = prepared / "development-only.usd"
        model.write_bytes(b"development_only synthetic model; not an actual source import")
        record = contract.file_record(model)
        imported = {"model": {**record, "path": model.name}, "claim_ceiling": "development_only"}
        write(prepared / "asset-import.json", imported)
        original = {"schema_version": "embodiment_runtime_binding_inputs.v1", "robot_embodiment": development_manifest(body, model),
            "model_record": {**record, "dependency_closure": {"claim_ceiling": "development_only"},
                "source_import_provenance": imported}}
        write(directory / "binding-inputs.json", original)
        originals[body], model_records[body] = original, record
        publication["results"].append({"embodiment_id": body, "status": "retained", "directory": str(prepared),
            "binding_inputs_path": str(directory / "binding-inputs.json"),
            "binding_inputs_sha256": contract.file_record(directory / "binding-inputs.json")["sha256"]})
    publication["publication_digest"] = contract.digest(publication)
    write(models / "batches/development-only-assets.json", publication)
    registry = {"schema_version": "blueprint.team_candidate_quick10_registry.v1", "profiles": [{"capture_root": "/development-only",
        "screening_candidates": {"existing_learned": {"keep": True}}}], "derived_profiles": {"defaults": {"budget_usd": 15}},
        "other": {"preserve": True}}
    registry["registry_digest"] = contract.digest(registry)
    registry_path = tmp_path / "etc/registry.json"
    write(registry_path, registry)
    req = request(preparation_publication_digest=publication["publication_digest"], expected_registry_digest=registry["registry_digest"])
    destination = state / req["installation_id"]
    write(destination / "request.json", req)
    registrations, results = {}, []
    for body, original in originals.items():
        manifest = original["robot_embodiment"]
        path = destination / body / (manifest["embodiment_digest"][7:] + ".robot_manifest.json")
        write(path, manifest)
        registrations["blueprint_scripted_"+body] = {"team_id": contract.SCRIPTED_TEAM_ID, "checkpoint_id": scripts.CHECKPOINT_ID,
            "display_name": "Blueprint scripted pick/place — " + body, "model": model_records[body],
            "manifest": contract.file_record(path), "policy_spec": build_policy_spec(manifest),
            "rights_binding": scripts.rights_record(req, profile, body), "dependency_closure": original["model_record"]["dependency_closure"],
            "source_import_provenance": original["model_record"]["source_import_provenance"], "minimum_gpu_memory_gb": 46,
            "deployment": scripts.deployment(body)}
        results.append(scripts.result_row(body, "registered"))
    proposal = {"schema_version": "scripted_robot_baseline_install_proposal.v1", "request_digest": req["request_digest"],
        "registrations": registrations, "results": results, "candidate_policy_queried": False, "scripted_controller_invoked": False,
        "provider_allocations_performed": 0}
    write(destination / "proposal.json", proposal)
    return SimpleNamespace(request=req, profile=profile, originals=originals, publication=publication, registry=registry,
        destination=destination, proposal=proposal, options={"registry_path": registry_path, "state_root": state,
            "model_root": models, "profile_root": profile_root, "audit_root": tmp_path / "audit"})


def test_installed_profile_is_exact_source_inventory_license_and_all_ten_frames(batch):
    from blueprint_pipeline.scripted_robot_grasp_frames import _FRAMES, _IDENTITIES
    assert batch.profile["source_identity"] == source_identity()
    assert {key: batch.profile["frames"][key] for key in _FRAMES} == json.loads(json.dumps(_FRAMES))
    assert {key: batch.profile["bodies"][key] for key in _IDENTITIES} == _IDENTITIES
    from blueprint_pipeline.scripted_robot_robotiq_runtime import source_identity as variant_source
    assert batch.profile["variant_source_identity"] == variant_source()
    from blueprint_pipeline.scripted_robot_industrial_robotiq_runtime import source_identity as industrial_source
    assert batch.profile["industrial_source_identity"] == industrial_source()
    from blueprint_pipeline.scripted_robot_yaskawa_robotiq_runtime import source_identity as yaskawa_source
    assert batch.profile["yaskawa_source_identity"] == yaskawa_source()
    assert (PROFILE_ROOT / "scripted_robot_baseline_license.txt").read_bytes() == (ROOT / "LICENSE").read_bytes()
    assert required_scope("install-scripted-robot-baselines") == "operate"
    assert validate_request({"kind": "install-scripted-robot-baselines", "request": batch.request})["request"] == batch.request
    for original in batch.originals.values():
        manifest = original["robot_embodiment"]
        assert scripts.validated_policy(manifest, build_policy_spec(manifest), batch.profile) == build_policy_spec(manifest)


@pytest.mark.parametrize("changes", [{"candidate_id": "arbitrary"}, {"embodiment_ids": ["almond_axol"]}, {"checkpoint_id": "fake"},
    {"model_path": "/tmp/model"}, {"url": "https://example.invalid"}, {"command": "gpu-canary"}, {"credential_env": "SECRET"},
    {"team_id": "vendor"}, {"installation_id": "../escape"}, {"policy_profile": "learned"}])
def test_fixed_batch_shape_cannot_select_code_paths_authority_or_individual_candidates(changes):
    with pytest.raises(RequestRefused):
        validate_request({"kind": "install-scripted-robot-baselines", "request": request(**changes)})


def test_batch_cas_preserves_other_fields_and_exactly_ten_typed_registrations(batch):
    receipt = installer.apply(batch.request, **batch.options)
    assert receipt["registered_count"] == len(contract.SCRIPTED_BODIES) and len(receipt["results"]) == len(contract.SCRIPTED_BODIES)
    assert receipt["provider_allocations_performed"] == 0 and receipt["candidate_policy_queried"] is False
    assert receipt["scripted_controller_invoked"] is False and receipt["native_policy_execution_validated"] is False
    assert receipt["claim_ceiling"] == "scripted_baseline_registration_only"
    after = contract.read(batch.options["registry_path"])
    for settings in [after["derived_profiles"]["defaults"], *after["profiles"]]:
        for key, value in batch.proposal["registrations"].items():
            assert settings["screening_candidates"].pop(key) == {"robot_policy": value}
    after["derived_profiles"]["defaults"].pop("screening_candidates")
    after["registry_digest"] = batch.registry["registry_digest"]
    assert after == batch.registry
    from blueprint_pipeline.team_candidate_quick10_admission import validate_candidate_rights
    for key, value in batch.proposal["registrations"].items():
        rights = value["rights_binding"]
        validate_candidate_rights(rights, team_id=contract.SCRIPTED_TEAM_ID, checkpoint_id=scripts.CHECKPOINT_ID, candidate_id=key)
        assert rights["license_id"] == "MIT" and rights["private_vendor_authorization_claimed"] is False


def test_partial_batch_retains_individual_import_and_tool_reasons(batch):
    # Missing source/imports grant no registration; successful bodies still register.
    for index, reason in ((0, "prepared_body_verification_failed"), (1, "tool_profile_unavailable"),
                          (2, "source_or_import_verification_failed")):
        body = contract.SCRIPTED_BODIES[index]
        batch.proposal["results"][index] = scripts.result_row(body, "unavailable", reason)
        del batch.proposal["registrations"]["blueprint_scripted_"+body]
    write(batch.destination / "proposal.json", batch.proposal)
    receipt = installer.apply(batch.request, **batch.options)
    assert receipt["registered_count"] == len(contract.SCRIPTED_BODIES) - 3 and receipt["results"] == batch.proposal["results"]


@pytest.mark.parametrize("fault", ["endpoint", "credentials", "command", "translation", "rotation", "private_license", "source", "body", "manifest"])
def test_root_reconstructs_fixed_policy_and_rights_instead_of_trusting_proposal(batch, fault):
    row = next(iter(batch.proposal["registrations"].values()))
    if fault in {"endpoint", "credentials", "command"}:
        row["policy_spec"][fault] = "arbitrary"
    elif fault in {"translation", "rotation"}:
        transform = row["policy_spec"]["grasp_bindings"]["left"]["tcp_from_grasp"]
        transform["position_m" if fault == "translation" else "orientation_xyzw"][0] += .01
    elif fault == "private_license":
        row["rights_binding"]["private_vendor_authorization_claimed"] = True
    elif fault == "source":
        row["policy_spec"]["model_identity_digest"] = "sha256:"+"f"*64
    elif fault == "body":
        row["policy_spec"]["embodiment_id"] = "boston_dynamics_atlas"
    else:
        row["manifest"]["path"] = "/tmp/arbitrary.json"
    write(batch.destination / "proposal.json", batch.proposal)
    with pytest.raises(ValueError):
        installer.apply(batch.request, **batch.options)
    assert contract.read(batch.options["registry_path"]) == batch.registry


@pytest.mark.parametrize("fault", ["code", "license", "sidecar", "symlink", "model", "publication"])
def test_source_license_import_and_complete_installed_code_identity_cannot_drift(batch, fault):
    root = batch.options["profile_root"]
    if fault == "code":
        (root / "scripted-baseline-source" / scripts.CODE_FILES[0]).write_text("changed script")
    elif fault == "license":
        (root / "scripted_robot_baseline_license.txt").write_text("changed license")
    elif fault == "sidecar":
        (root / "scripted-baseline-source/arbitrary.py").write_text("pass")
    elif fault == "symlink":
        path = root / "scripted-baseline-source" / scripts.CODE_FILES[0]
        path.unlink()
        path.symlink_to(ROOT / "src/blueprint_pipeline" / scripts.CODE_FILES[0])
    elif fault == "model":
        Path(next(iter(batch.proposal["registrations"].values()))["model"]["path"]).write_bytes(b"tampered")
    else:
        path = batch.options["model_root"] / "batches/development-only-assets.json"
        value = contract.read(path)
        value["task_executed"] = True
        write(path, value)
    with pytest.raises(ValueError):
        installer.apply(batch.request, **batch.options)
    assert contract.read(batch.options["registry_path"]) == batch.registry


def test_batch_serializes_retries_and_recovers_after_replace_without_reapplying(batch, monkeypatch):
    write_once = installer._write_once
    def interrupted(path, value):
        if path.name == "receipt.json":
            raise OSError("development_only lost response")
        write_once(path, value)
    monkeypatch.setattr(installer, "_write_once", interrupted)
    with pytest.raises(OSError):
        installer.apply(batch.request, **batch.options)
    installed = batch.options["registry_path"].read_bytes()
    monkeypatch.setattr(installer, "_write_once", write_once)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: installer.apply(batch.request, **batch.options), range(3)))
    assert results[0] == results[1] == results[2] and results[0]["registered_count"] == len(contract.SCRIPTED_BODIES)
    assert batch.options["registry_path"].read_bytes() == installed
    with pytest.raises(ValueError, match="audit_identity_conflict"):
        installer.apply(request(**{**batch.request, "authorization_reference": "changed"}), **batch.options)


def test_batch_compare_detects_external_update_before_write(batch, monkeypatch):
    write_once = installer._write_once
    def update_after_intent(path, value):
        write_once(path, value)
        if path.name == "intent.json":
            changed = deepcopy(batch.registry)
            changed["external"] = "preserve"
            changed["registry_digest"] = contract.digest(changed, "registry_digest")
            write(batch.options["registry_path"], changed)
    monkeypatch.setattr(installer, "_write_once", update_after_intent)
    with pytest.raises(ValueError, match="registry_compare_failed"):
        installer.apply(batch.request, **batch.options)
    assert contract.read(batch.options["registry_path"])["external"] == "preserve"
    assert not (batch.options["audit_root"] / batch.request["installation_id"] / "receipt.json").exists()


def test_batch_spool_is_networkless_fixed_builder_and_no_compute(tmp_path):
    config = DoorConfig(state_root=str(tmp_path / "door"))
    for sub in ("pending", "processing", "completed", "results"):
        (Path(config.spool_root) / sub).mkdir(parents=True)
    req = request()
    enqueue(config, {"kind": "install-scripted-robot-baselines", "request": req}, requested_by="development-only",
        operation_key="development-only-scripted")
    calls = []
    process_spool(config, runner=SimpleNamespace(run=lambda argv, timeout: calls.append(argv) or CommandResult(0, "", "")))
    launch, = calls
    assert launch[-1] == config.install_root + "/door-install-public-piper-policy.sh"
    assert "--property=PrivateNetwork=yes" in launch and "--property=RuntimeMaxSec=10min" in launch
    env = dict(value.removeprefix("--setenv=").split("=", 1) for value in launch if value.startswith("--setenv="))
    assert env["DOOR_PUBLIC_POLICY_OPERATION"] == "install-scripted-robot-baselines"
    assert env["DOOR_SERVICE_USER"] == "blueprint" and json.loads(env["DOOR_PUBLIC_POLICY_REQUEST"]) == req
    assert not any("TOKEN" in key or "KEY" in key for key in env)


def test_actual_isolated_installer_can_load_scripted_contract_and_profile(batch):
    # Same fixed import mechanism as python3 -I -S privileged entrypoint;
    # run profile validation in an isolated stdlib subprocess, never execute a script.
    code = "import importlib.util; from pathlib import Path; s=importlib.util.spec_from_file_location('installed'," + repr(str(ROOT /
        "deploy/operator-door/operator_door/public_policy_registration.py")) + "); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
    code += "p=m._scripted_contract().load_profile(Path(" + repr(str(batch.options["profile_root"])) + ")); assert len(p['bodies'])==24"
    result = subprocess.run([sys.executable, "-I", "-S", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_service_builder_attempts_every_retained_body_and_retries_without_rebuilding(batch, monkeypatch):
    from blueprint_pipeline import scripted_robot_registration_install as builder
    from blueprint_pipeline import deployment_embodiment_runtime_configuration as runtime
    from blueprint_pipeline import scripted_robot_registration as registration
    (batch.destination / "proposal.json").unlink()
    calls = []
    def bind(configuration, prepared, url, runtime_settings):
        body = configuration.parent.parent.name
        calls.append(body)
        if body == contract.SCRIPTED_BODIES[0]:
            raise ValueError("development_only import proof unavailable")
        assert runtime_settings == {"control_hz": 60, "action_chunk_rows": 48}
        return batch.originals[body]
    def register(**kwargs):
        # Only the expensive native import/provenance check is substituted;
        # root reconstruction, frame math, rights and publication stay real.
        body = kwargs["prepared"]["robot_embodiment"]["embodiment_id"]
        return batch.proposal["registrations"]["blueprint_scripted_"+body]
    monkeypatch.setattr(runtime, "bind_prepared_configuration", bind)
    monkeypatch.setattr(registration, "registration_from_prepared_inputs", register)
    opts = {key: batch.options[key] for key in ("state_root", "model_root", "profile_root")}
    result = builder.prepare(batch.request, release_commit=batch.request["release_commit"], **opts)
    assert calls == list(contract.SCRIPTED_BODIES) and result["registered_count"] == len(contract.SCRIPTED_BODIES) - 1
    assert result["results"][0]["reason"] == "source_or_import_verification_failed"
    assert installer.apply(batch.request, **batch.options)["registered_count"] == len(contract.SCRIPTED_BODIES) - 1
    assert builder.prepare(batch.request, release_commit=batch.request["release_commit"], **opts) == result
    assert len(calls) == len(contract.SCRIPTED_BODIES)


def test_absent_and_failed_imports_keep_individual_reasons_without_registration(batch):
    absent, failed = contract.SCRIPTED_BODIES[:2]
    publication = batch.publication
    publication["results"] = [row for row in publication["results"] if row["embodiment_id"] != absent]
    next(row for row in publication["results"] if row["embodiment_id"] == failed)["status"] = "import_failed"
    publication["publication_digest"] = contract.digest(publication, "publication_digest")
    write(batch.options["model_root"] / "batches/development-only-assets.json", publication)
    req = request(**{**batch.request, "preparation_publication_digest": publication["publication_digest"]})
    write(batch.destination / "request.json", req)
    batch.proposal["request_digest"] = req["request_digest"]
    for index, body in enumerate((absent, failed)):
        batch.proposal["results"][index] = scripts.result_row(body, "unavailable", "prepared_body_unavailable")
        del batch.proposal["registrations"]["blueprint_scripted_"+body]
    write(batch.destination / "proposal.json", batch.proposal)
    receipt = installer.apply(req, **batch.options)
    assert receipt["registered_count"] == len(contract.SCRIPTED_BODIES) - 2 and receipt["results"][:2] == batch.proposal["results"][:2]


def test_stale_batch_registry_digest_never_changes_registry(batch):
    req = request(**{**batch.request, "expected_registry_digest": "sha256:"+"f"*64})
    with pytest.raises(ValueError, match="registry_compare_failed"):
        installer.apply(req, **batch.options)
    assert contract.read(batch.options["registry_path"]) == batch.registry


def test_scripted_identity_cannot_cross_public_model_operation():
    for kind in ("install-public-piper-policy", "install-public-yam-policy"):
        with pytest.raises(RequestRefused):
            validate_request({"kind": kind, "request": request()})


def test_service_builder_rejects_changed_release_and_source_before_any_proposal(batch, monkeypatch):
    from blueprint_pipeline import scripted_robot_registration_install as builder
    options = {key: batch.options[key] for key in ("state_root", "model_root", "profile_root")}
    with pytest.raises(ValueError, match="installed_release_mismatch"):
        builder.prepare(batch.request, release_commit="f"*40, **options)
    path = batch.options["profile_root"] / "scripted_robot_baseline_registration_profile.json"
    value = contract.read(path)
    value["source_identity"]["files"][0]["sha256"] = "sha256:"+"f"*64
    write(path, value)
    with pytest.raises(ValueError, match="script_source_changed"):
        builder.prepare(batch.request, release_commit=batch.request["release_commit"], **options)
    assert contract.read(batch.options["registry_path"]) == batch.registry


def test_no_eligible_body_retains_ten_reasons_without_any_registry_change(batch):
    batch.proposal["registrations"] = {}
    batch.proposal["results"] = [scripts.result_row(body, "unavailable", "source_or_import_verification_failed")
                                 for body in contract.SCRIPTED_BODIES]
    write(batch.destination / "proposal.json", batch.proposal)
    before = batch.options["registry_path"].read_bytes()
    receipt = installer.apply(batch.request, **batch.options)
    assert receipt["registered_count"] == 0 and len(receipt["results"]) == len(contract.SCRIPTED_BODIES)
    assert batch.options["registry_path"].read_bytes() == before
    assert receipt["before_registry_digest"] == receipt["after_registry_digest"]
