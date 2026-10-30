"""Fixed community TRLC registration; development-only artifacts, no external calls."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from tests.test_operator_door_public_policy_registration import (
    contract,
    installer,
    write,
    ROOT,
    PROFILE_ROOT,
    validate_request,
    RequestRefused,
    DoorConfig,
    CommandResult,
    enqueue,
    process_spool,
)
from tests.test_scripted_robot_grasp_frames import development_manifest
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.team_embodiment import validate_embodiment_manifest
from blueprint_pipeline.trlc_pi05_runtime import (
    build_policy_spec,
    configure_prepared_inputs,
    checkpoint_identity,
)

NOTICES = ROOT / "src/blueprint_pipeline/_vendor/trlc_pi05_notices"


def request(**changes):
    identity = contract.IDENTITIES[contract.TRLC_SCHEMA]
    value = {
        "schema_version": contract.TRLC_SCHEMA,
        "installation_id": "development-only-trlc",
        "release_commit": "a" * 40,
        "preparation_authorization_id": "development-only-assets",
        "preparation_publication_digest": "sha256:" + "b" * 64,
        **{
            key: identity[key]
            for key in ("candidate_id", "team_id", "checkpoint_id", "policy_profile")
        },
        "expected_registry_digest": "sha256:" + "c" * 64,
        "authorization_reference": "development_only current internal public-policy simulation request",
    }
    value.update(changes)
    value["request_digest"] = contract.digest(value, "request_digest")
    return value


def manifest(model):
    value = development_manifest(contract.TRLC_EMBODIMENT_ID, model)
    for side in ("left", "right"):
        for index in range(6):
            old, new = f"{side}_development_joint_{index}", f"{side}_joint{index + 1}"
            value = json.loads(json.dumps(value).replace(old, new))
    camera = value["cameras"][0]
    value["cameras"] = [
        dict(camera),
        {**camera, "name": "left_wrist"},
        {**camera, "name": "right_wrist"},
    ]
    value.pop("embodiment_digest", None)
    return validate_embodiment_manifest(value)


@pytest.fixture()
def retained(tmp_path, monkeypatch):
    profile_root = tmp_path / "installed-profile"
    profile_root.mkdir()
    shutil.copytree(NOTICES, profile_root / "trlc_pi05_notices")
    profile = json.loads((PROFILE_ROOT / "trlc_pi05_registration_profile.json").read_text())
    inventory = deepcopy(profile["tokenizer_inventory"])
    tokenizer = tmp_path / "development-only-tokenizer"
    tokenizer.mkdir()
    # Tiny explicit test inventory, not public tokenizer bytes or access evidence.
    for row in inventory["files"]:
        raw = ("development_only:" + row["path"]).encode()
        (tokenizer / row["path"]).write_bytes(raw)
    # Rebuild in canonical publisher order after replacing the development payloads.
    inventory["files"] = []
    for index, path in enumerate(sorted(tokenizer.iterdir())):
        raw = path.read_bytes()
        row = {"path": path.name, "size_bytes": len(raw)}
        if index % 2:
            row["sha256"] = hashlib.sha256(raw).hexdigest()
        else:
            row["git_blob_sha1"] = hashlib.sha1(
                f"blob {len(raw)}\0".encode() + raw, usedforsecurity=False
            ).hexdigest()
        inventory["files"].append(row)
    inventory_path = profile_root / "trlc_pi05_tokenizer_inventory.v1.json"
    write(inventory_path, inventory)
    profile.update(
        tokenizer_inventory=inventory,
        tokenizer_inventory_sha256=contract.file_record(inventory_path)["sha256"],
    )
    write(profile_root / "trlc_pi05_registration_profile.json", profile)
    monkeypatch.setattr(contract, "TRLC_TOKENIZER_ROOT", tokenizer)
    model_root, state = tmp_path / "models", tmp_path / "proposals"
    body = model_root / contract.TRLC_EMBODIMENT_ID / ("d" * 64)
    prepared = body / "prepared"
    prepared.mkdir(parents=True)
    model = prepared / "development-only.usda"
    model.write_text('#usda 1.0\n\ndef Xform "DevelopmentOnly" {}\n')
    model_record = contract.file_record(model)
    imported = {"model": {**model_record, "path": model.name}, "claim_ceiling": "development_only"}
    write(prepared / "asset-import.json", imported)
    original = {
        "schema_version": "embodiment_runtime_binding_inputs.v1",
        "robot_embodiment": manifest(model),
        "model_record": {
            **model_record,
            "dependency_closure": {"claim_ceiling": "development_only"},
            "source_import_provenance": imported,
        },
    }
    write(body / "binding-inputs.json", original)
    publication = {
        "status": "completed",
        "claim_ceiling": "robot_asset_import_only",
        "candidate_policy_queried": False,
        "task_executed": False,
        "results": [
            {
                "embodiment_id": contract.TRLC_EMBODIMENT_ID,
                "status": "retained",
                "directory": str(prepared),
                "binding_inputs_path": str(body / "binding-inputs.json"),
                "binding_inputs_sha256": contract.file_record(body / "binding-inputs.json")[
                    "sha256"
                ],
            }
        ],
    }
    publication["publication_digest"] = contract.digest(publication)
    write(model_root / "batches/development-only-assets.json", publication)
    registry = {
        "schema_version": "blueprint.team_candidate_quick10_registry.v1",
        "profiles": [{"capture_root": "/development-only"}],
        "derived_profiles": {"defaults": {"screening_candidates": {"prior": {"keep": True}}}},
        "preserve": [1, 2],
    }
    registry["registry_digest"] = contract.digest(registry)
    registry_path = tmp_path / "etc/registry.json"
    write(registry_path, registry)
    req = request(
        preparation_publication_digest=publication["publication_digest"],
        expected_registry_digest=registry["registry_digest"],
    )
    destination = state / req["installation_id"]
    configured = contract.expected_manifest(original["robot_embodiment"], profile)
    manifest_path = destination / (configured["embodiment_digest"][7:] + ".robot_manifest.json")
    write(manifest_path, configured)
    write(destination / "request.json", req)
    registration = {
        "team_id": contract.TEAM_ID,
        "checkpoint_id": contract.TRLC_CHECKPOINT_ID,
        "display_name": profile["display_name"],
        "model": model_record,
        "manifest": contract.file_record(manifest_path),
        "policy_spec": contract.expected_policy(configured, profile),
        "rights_binding": contract.rights_record(req, profile),
        "dependency_closure": original["model_record"]["dependency_closure"],
        "source_import_provenance": imported,
        "minimum_gpu_memory_gb": 48,
        "deployment": profile["deployment"],
        "offline_tokenizer": contract.trlc_tokenizer_record(profile),
    }
    proposal = {
        "schema_version": "trlc_pi05_policy_install_proposal.v1",
        "request_digest": req["request_digest"],
        "registration": registration,
        "candidate_policy_queried": False,
        "provider_allocations_performed": 0,
    }
    write(destination / "proposal.json", proposal)
    return SimpleNamespace(
        request=req,
        profile=profile,
        original=original,
        publication=publication,
        registry=registry,
        model=model,
        tokenizer=tokenizer,
        destination=destination,
        proposal=proposal,
        registration=registration,
        options={
            "registry_path": registry_path,
            "state_root": state,
            "model_root": model_root,
            "profile_root": profile_root,
            "audit_root": tmp_path / "audit",
        },
    )


def test_fixed_public_profile_binds_real_checkpoint_terms_and_offline_inventory(tmp_path):
    req = request()
    assert (
        validate_request({"kind": "install-public-trlc-policy", "request": req})["request"] == req
    )
    profile = contract.load_profile(req, PROFILE_ROOT, license_root=NOTICES)
    model = tmp_path / "development-only.usda"
    model.write_text("#usda 1.0\n")
    original = manifest(model)
    configured = configure_prepared_inputs(
        {"schema_version": "embodiment_runtime_binding_inputs.v1", "robot_embodiment": original}
    )
    assert contract.expected_manifest(original, profile) == configured["robot_embodiment"]
    assert contract.expected_policy(configured["robot_embodiment"], profile) == build_policy_spec(
        configured["robot_embodiment"]
    )
    assert profile["policy_spec_template"]["model_identity_digest"] == canonical_digest(
        checkpoint_identity()
    )
    assert (
        profile["tokenizer_inventory_sha256"]
        == "sha256:5020414c0d782bb33cdf1f53b220569c83aff55356d2d920f7741657eb0fe830"
    )
    rights = contract.rights_record(req, profile)
    assert rights["license_id"] == "Apache-2.0 AND Gemma-Terms-of-Use"
    assert rights["usage_scope"] == "internal_out_of_distribution_rigid_task_simulation_diagnostic"
    assert all(
        rights[name] is False
        for name in (
            "old_g1_task_authority_reused",
            "new_human_review_claimed",
            "model_redistribution_authorized",
            "public_model_service_authorized",
            "cloth_benchmark_claimed",
            "vendor_policy_claimed",
        )
    )
    assert (
        profile["deployment"]["companies"] == []
        and profile["deployment"]["policy_route_id"] == "pi05_custom_checkpoint"
    )
    assert "PaliGemma" in (NOTICES / "gemma-terms.txt").read_text()
    assert (
        "Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms"
        in (NOTICES / "NOTICE.txt").read_text()
    )


def test_exact_registration_cas_preserves_other_candidates_and_retries(retained):
    receipt = installer.apply(retained.request, **retained.options)
    assert receipt["candidate_id"] == contract.TRLC_CANDIDATE_ID
    assert (
        receipt["native_policy_execution_validated"] is False
        and receipt["provider_allocations_performed"] == 0
    )
    assert installer.apply(retained.request, **retained.options) == receipt
    updated = contract.read(retained.options["registry_path"])
    for settings in [updated["derived_profiles"]["defaults"], *updated["profiles"]]:
        assert settings["screening_candidates"].pop(contract.TRLC_CANDIDATE_ID) == {
            "robot_policy": retained.registration
        }
    updated["profiles"][0].pop("screening_candidates")
    updated["registry_digest"] = retained.registry["registry_digest"]
    assert updated == retained.registry


@pytest.mark.parametrize(
    "fault",
    [
        "body",
        "camera",
        "endpoint",
        "rights",
        "tokenizer_receipt",
        "model_bytes",
        "tokenizer_bytes",
        "tokenizer_missing",
        "terms",
        "inventory",
        "stale_registry",
    ],
)
def test_tampered_or_unavailable_inputs_do_not_change_registry(retained, fault):
    row = retained.proposal["registration"]
    if fault == "body":
        row["policy_spec"]["embodiment_id"] = "i2rt_yam_bimanual"
    elif fault == "camera":
        row["policy_spec"]["binding"]["camera_names"].reverse()
    elif fault == "endpoint":
        row["endpoint_url"] = "https://untrusted.invalid"
    elif fault == "rights":
        row["rights_binding"]["old_g1_task_authority_reused"] = True
    elif fault == "tokenizer_receipt":
        row["offline_tokenizer"]["inventory_sha256"] = "sha256:" + "0" * 64
    elif fault == "model_bytes":
        retained.model.write_bytes(b"changed")
    elif fault == "tokenizer_bytes":
        (retained.tokenizer / "added_tokens.json").write_bytes(b"changed")
    elif fault == "tokenizer_missing":
        (retained.tokenizer / "tokenizer.json").unlink()
    elif fault == "terms":
        (retained.options["profile_root"] / "trlc_pi05_notices/NOTICE.txt").write_text("changed")
    elif fault == "inventory":
        (retained.options["profile_root"] / "trlc_pi05_tokenizer_inventory.v1.json").write_text(
            "{}"
        )
    else:
        retained.request["expected_registry_digest"] = "sha256:" + "0" * 64
        retained.request["request_digest"] = contract.digest(retained.request, "request_digest")
        write(retained.destination / "request.json", retained.request)
    write(retained.destination / "proposal.json", retained.proposal)
    with pytest.raises((ValueError, OSError)):
        installer.apply(retained.request, **retained.options)
    assert contract.read(retained.options["registry_path"]) == retained.registry


@pytest.mark.parametrize(
    "changes",
    [
        {"tokenizer_dir": "/tmp/arbitrary"},
        {"command": "run"},
        {"candidate_id": "humanoidarena_pi05_g1_dex3_sonic"},
        {"policy_profile": "other"},
        {"preparation_authorization_id": "../../other"},
    ],
)
def test_caller_cannot_choose_paths_authority_or_another_policy(changes):
    with pytest.raises(RequestRefused):
        validate_request({"kind": "install-public-trlc-policy", "request": request(**changes)})


def test_networkless_fixed_spool_and_no_secret_environment(tmp_path):
    req = request()
    config = DoorConfig(state_root=str(tmp_path / "door"))
    for sub in ("pending", "processing", "completed", "results"):
        (Path(config.spool_root) / sub).mkdir(parents=True)
    enqueue(
        config,
        {"kind": "install-public-trlc-policy", "request": req},
        requested_by="development-only",
        operation_key="development-only-trlc",
    )
    commands = []
    process_spool(
        config,
        runner=SimpleNamespace(
            run=lambda argv, timeout: commands.append(argv) or CommandResult(0, "", "")
        ),
    )
    (launch,) = commands
    assert launch[-1] == config.install_root + "/door-install-public-piper-policy.sh"
    assert "--property=PrivateNetwork=yes" in launch
    assert "--property=RestrictAddressFamilies=AF_UNIX" in launch
    assert "--property=RuntimeMaxSec=5min" in launch
    env = dict(
        v.removeprefix("--setenv=").split("=", 1) for v in launch if v.startswith("--setenv=")
    )
    assert env["DOOR_PUBLIC_POLICY_OPERATION"] == "install-public-trlc-policy"
    assert not any("TOKEN" in key or "KEY" in key for key in env)


def test_service_builder_uses_fixed_runtime_then_privileged_reconstruction(retained, monkeypatch):
    from blueprint_pipeline import openwam_piper_registration as builder
    from blueprint_pipeline import deployment_embodiment_runtime_configuration as configuration
    from blueprint_pipeline import beta_policy_screening_team as team

    monkeypatch.setattr(builder._contract, "TRLC_TOKENIZER_ROOT", retained.tokenizer)
    calls = []

    def bound(config, prepared, url, *, runtime_settings):
        assert config.parent == retained.model.parent.parent
        assert prepared == retained.model.parent
        assert url == retained.original["robot_embodiment"]["model"]["url"]
        assert runtime_settings == {"control_hz": 60, "action_chunk_rows": 50}
        calls.append("source_binding")
        return deepcopy(retained.original)

    def registration(**kwargs):
        assert kwargs["candidate_id"] == contract.TRLC_CANDIDATE_ID
        assert kwargs["model_path"] == str(retained.model)
        assert kwargs["policy_spec"] == retained.registration["policy_spec"]
        assert kwargs["rights"] == retained.registration["rights_binding"]
        assert kwargs["prepared"]["robot_embodiment"] == contract.expected_manifest(
            retained.original["robot_embodiment"], retained.profile
        )
        calls.append("registration_binding")
        return {
            key: deepcopy(value)
            for key, value in retained.registration.items()
            if key not in {"minimum_gpu_memory_gb", "deployment", "offline_tokenizer"}
        }

    monkeypatch.setattr(configuration, "bind_prepared_configuration", bound)
    monkeypatch.setattr(team, "registration_from_prepared_inputs", registration)
    result = builder.prepare(
        retained.request,
        state_root=retained.options["state_root"],
        model_root=retained.options["model_root"],
        profile_root=retained.options["profile_root"],
        release_commit=retained.request["release_commit"],
    )
    assert calls == ["source_binding", "registration_binding"]
    assert result["status"] == "prepared" and result["provider_allocations_performed"] == 0
    assert installer.apply(retained.request, **retained.options)["status"] == "installed"


@pytest.mark.parametrize("name", ["added_tokens.json", "tokenizer.json"])
def test_same_size_tokenizer_tamper_fails_both_git_blob_and_sha256(retained, name):
    path = retained.tokenizer / name
    raw = path.read_bytes()
    path.write_bytes(b"X" + raw[1:])
    with pytest.raises(ValueError, match="trlc_offline_tokenizer_bytes_changed"):
        contract.trlc_tokenizer_record(retained.profile)


def test_tokenizer_symlink_is_not_a_retained_asset(retained, tmp_path):
    path = retained.tokenizer / "tokenizer.json"
    target = tmp_path / "unbound-tokenizer.json"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink_path_refused"):
        contract.trlc_tokenizer_record(retained.profile)


def test_default_community_row_has_precise_missing_inputs_and_preserves_vendor_row():
    from blueprint_pipeline.beta_policy_candidate_catalog import (
        catalog_candidate,
        load_default_catalog,
    )
    from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan
    from tests.test_beta_policy_screening_plan import scene, NOW

    catalog = load_default_catalog()
    row = catalog_candidate(catalog, contract.TRLC_CANDIDATE_ID)
    assert row["eligibility"]["enabled"] is False
    assert row["interface"] is row["runtime"] is None
    assert row["eligibility"]["requires_credentials"] == []
    assert row["eligibility"]["blockers"] == [
        "prepared_robot_policy_registration_missing",
        "trlc_offline_tokenizer_access_unverified",
    ]
    assert row["deployment"]["companies"] == []
    assert row["deployment"]["policy_route_id"] == "pi05_custom_checkpoint"
    vendor = catalog_candidate(catalog, "trlc_dk1_lerobot_controller")
    assert (
        vendor["candidate_id"] != row["candidate_id"] and vendor["eligibility"]["enabled"] is False
    )
    plan = build_screening_plan(catalog, scene(), campaign_id="development-only-trlc", now=NOW)
    actual = next(
        candidate
        for candidate in plan["candidates"]
        if candidate["candidate_id"] == row["candidate_id"]
    )
    assert actual["status"] == "unavailable" and set(row["eligibility"]["blockers"]) <= set(
        actual["blockers"]
    )
