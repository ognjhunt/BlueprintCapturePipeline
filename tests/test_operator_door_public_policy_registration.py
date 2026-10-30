"""Development-only fixed registration, no provider, simulator or allocation."""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy/operator-door"))
from operator_door import public_policy_registration as installer  # noqa: E402
from operator_door import public_policy_registration_contract as contract  # noqa: E402
from operator_door.config import DoorConfig  # noqa: E402
from operator_door.hostinfo import CommandResult  # noqa: E402
from operator_door.requests import RequestRefused, enqueue, required_scope, validate_request  # noqa: E402
from operator_door.spool_runner import process_spool  # noqa: E402
from blueprint_pipeline.decision_evidence_contracts import canonical_digest  # noqa: E402

PROFILE_ROOT = ROOT / "config/policy_runtimes"


def test_actual_isolated_privileged_entrypoint_loads_only_its_installed_sibling():
    result = subprocess.run([sys.executable, "-I", "-S", str(ROOT /
        "deploy/operator-door/operator_door/public_policy_registration.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--request-json" in result.stdout


def request(*, yam=False, **changes):
    identity = contract.IDENTITIES[contract.YAM_SCHEMA if yam else contract.SCHEMA]
    value = {"schema_version": contract.YAM_SCHEMA if yam else contract.SCHEMA, "installation_id": "development-only",
        "release_commit": "a" * 40, "preparation_authorization_id": "development-only-assets",
        "preparation_publication_digest": "sha256:" + "b" * 64,
        **{key: identity[key] for key in ("candidate_id", "team_id", "checkpoint_id", "policy_profile")},
        "expected_registry_digest": "sha256:" + "c" * 64,
        "authorization_reference": "development_only explicit request to test public models"}
    value.update(changes)
    value["request_digest"] = contract.digest(value, "request_digest")
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    path.chmod(0o600)


@pytest.fixture()
def retained(tmp_path):
    return retained_fixture(tmp_path)


@pytest.fixture()
def retained_yam(tmp_path):
    return retained_fixture(tmp_path, yam=True)


def retained_fixture(tmp_path, *, yam=False):
    """Minimal fixed-source records; fake USD bytes never claim an actual import."""
    models = tmp_path / "models"
    identity = contract.IDENTITIES[contract.YAM_SCHEMA if yam else contract.SCHEMA]
    body = models / identity["embodiment_id"] / ("d" * 64)
    prepared = body / "prepared"
    prepared.mkdir(parents=True)
    model = prepared / "development-only.usd"
    model.write_bytes(b"development_only model record; no native physics proof")
    model_record = contract.file_record(model)
    model_source = {**model_record, "path": model.name}
    imported = {"model": model_source, "claim_ceiling": "development_only"}
    write(prepared / "asset-import.json", imported)
    original = {"schema_version": "embodiment_runtime_binding_inputs.v1", "robot_embodiment": {
        "embodiment_id": identity["embodiment_id"], "control_hz": 30, "action_chunk_rows": 8,
        "cameras": [{"name": name, "width": 320, "height": 176, "horizontal_fov_deg": 90.0,
                     "mount_link": None if name == "overhead" else name.removesuffix("_wrist") + "_link_6",
                     "position_m": [0.1, 0.2, 0.3], "orientation_xyzw": [0., 0., 0., 1.]}
                    for name in ("overhead", "left_wrist", "right_wrist")],
        "model": {"format": "usd", "url": "https://assets.example.invalid/development-only.usd",
                  "sha256": model_record["sha256"]}},
        "model_record": {"sha256": model_record["sha256"], "size_bytes": model_record["size_bytes"],
            "dependency_closure": {"claim_ceiling": "development_only"}, "source_import_provenance": imported}}
    original["robot_embodiment"]["embodiment_digest"] = contract.digest(original["robot_embodiment"])
    write(body / "binding-inputs.json", original)
    publication = {"status": "completed", "claim_ceiling": "robot_asset_import_only",
        "candidate_policy_queried": False, "task_executed": False, "results": [{
            "embodiment_id": identity["embodiment_id"], "status": "retained", "directory": str(prepared),
            "binding_inputs_path": str(body / "binding-inputs.json"),
            "binding_inputs_sha256": contract.file_record(body / "binding-inputs.json")["sha256"]}]}
    publication["publication_digest"] = contract.digest(publication)
    write(models / "batches/development-only-assets.json", publication)
    registry = {"schema_version": "blueprint.team_candidate_quick10_registry.v1",
        "profiles": [{"capture_root": "/development-only/scene", "screening_candidates": {"existing": {"keep": True}},
                      "team_rights": {"existing": "preserve"}}],
        "derived_profiles": {"root": "/development-only/profiles", "defaults": {
            "allowed_founder_approvers": ["development-only"], "max_hourly_rate_usd": 0.8,
            "open_loop_horizon": 8, "screening_candidates": {"existing": {"keep": True}}}},
        "unrelated": {"do_not_change": [1, 2, 3]}}
    registry["registry_digest"] = contract.digest(registry)
    registry_path = tmp_path / "etc/registry.json"
    write(registry_path, registry)
    req = request(yam=yam, preparation_publication_digest=publication["publication_digest"],
                  expected_registry_digest=registry["registry_digest"])
    state = tmp_path / "proposals"
    destination = state / req["installation_id"]
    profile = contract.load_profile(req, PROFILE_ROOT)
    manifest = contract.expected_manifest(original["robot_embodiment"], profile)
    manifest_path = destination / (manifest["embodiment_digest"][7:] + ".robot_manifest.json")
    write(manifest_path, manifest)
    write(destination / "request.json", req)
    registration = {"team_id": identity["team_id"], "checkpoint_id": identity["checkpoint_id"],
        "display_name": profile["display_name"], "model": model_record,
        "manifest": contract.file_record(manifest_path), "policy_spec": contract.expected_policy(manifest, profile),
        "rights_binding": contract.rights_record(req, profile),
        "dependency_closure": original["model_record"]["dependency_closure"],
        "source_import_provenance": original["model_record"]["source_import_provenance"],
        "minimum_gpu_memory_gb": profile["minimum_gpu_memory_gb"], "deployment": profile["deployment"]}
    proposal = {"schema_version": identity["proposal_schema"], "request_digest": req["request_digest"],
        "registration": registration, "candidate_policy_queried": False, "provider_allocations_performed": 0}
    write(destination / "proposal.json", proposal)
    return SimpleNamespace(request=req, original=original, profile=profile, registry=registry, model=model,
        registry_path=registry_path, destination=destination, proposal=proposal, registration=registration,
        options={"registry_path": registry_path, "state_root": state, "model_root": models,
                 "audit_root": tmp_path / "audit", "profile_root": PROFILE_ROOT})


def test_exact_request_and_public_license_provenance_have_no_private_grant():
    req = request(authorization_reference="development_only café $(false) `false`")
    assert req["request_digest"] == canonical_digest(req, digest_field="request_digest")
    assert validate_request({"kind": "install-public-piper-policy", "request": req})["request"] == req
    assert required_scope("install-public-piper-policy") == "operate"
    profile = contract.read(PROFILE_ROOT / "openwam_piper_registration_profile.json")
    assert contract.file_record(PROFILE_ROOT / "openwam_piper_model_card.md")["sha256"] == profile["model_card_sha256"]
    rights = contract.rights_record(req, profile)
    assert rights["license_id"] == "Apache-2.0" and not rights["private_vendor_authorization_claimed"]
    assert rights["authorization_reference"] == req["authorization_reference"]
    from blueprint_pipeline.team_candidate_quick10_admission import validate_candidate_rights
    validate_candidate_rights(rights, team_id=contract.TEAM_ID, checkpoint_id=contract.CHECKPOINT_ID,
                              candidate_id=contract.CANDIDATE_ID)


@pytest.mark.parametrize("changes", [
    {"installation_id": "../escape"}, {"preparation_authorization_id": "/tmp/assets"},
    {"release_commit": "A" * 40}, {"candidate_id": "pi05_droid"}, {"team_id": "DreamLabs"},
    {"checkpoint_id": "different"}, {"policy_profile": "hosted"}, {"expected_registry_digest": "missing"},
    {"authorization_reference": "line\nbreak"}, {"url": "https://arbitrary.invalid"},
    {"registry_path": "/tmp/arbitrary"}, {"command": "gpu-canary"}, {"credential_env": "SECRET"},
])
def test_no_caller_selected_code_paths_providers_or_authority(changes):
    with pytest.raises(RequestRefused):
        validate_request({"kind": "install-public-piper-policy", "request": request(**changes)})


def test_registry_cas_preserves_every_other_field_and_retries_idempotently(retained):
    before = copy.deepcopy(retained.registry)
    receipt = installer.apply(retained.request, **retained.options)
    assert receipt["status"] == "installed" and receipt["provider_allocations_performed"] == 0
    assert receipt["native_policy_execution_validated"] is False
    after = contract.read(retained.registry_path)
    assert after["registry_digest"] == canonical_digest(after, digest_field="registry_digest")
    for settings in [after["derived_profiles"]["defaults"], *after["profiles"]]:
        assert settings["screening_candidates"].pop(contract.CANDIDATE_ID) == {"robot_policy": retained.registration}
    after["registry_digest"] = before["registry_digest"]
    assert after == before
    assert installer.apply(retained.request, **retained.options) == receipt
    conflicting = request(**{**retained.request, "authorization_reference": "changed"})
    with pytest.raises(ValueError, match="audit_identity_conflict"):
        installer.apply(conflicting, **retained.options)


def test_stale_registry_refuses_before_mutation(retained):
    req = request(**{**retained.request, "expected_registry_digest": "sha256:" + "f" * 64})
    with pytest.raises(ValueError, match="registry_compare_failed"):
        installer.apply(req, **retained.options)
    assert contract.read(retained.registry_path) == retained.registry


@pytest.mark.parametrize("fault", ["endpoint", "credential", "command", "frame", "license", "model", "manifest", "body"])
def test_privileged_boundary_reconstructs_exact_registration_instead_of_trusting_helper(retained, fault):
    registration = retained.proposal["registration"]
    if fault in {"endpoint", "credential", "command"}:
        registration[{"endpoint": "endpoint_url", "credential": "credential_env", "command": "launch_command"}[fault]] = "arbitrary"
    elif fault == "frame":
        registration["policy_spec"]["binding"]["native_tcp_from_policy_tcp"]["left"]["position_m"][0] = 1
    elif fault == "license":
        registration["rights_binding"]["private_vendor_authorization_claimed"] = True
    elif fault == "model":
        registration["model"]["path"] = "/etc/blueprint/secrets/key"
    elif fault == "manifest":
        registration["manifest"]["path"] = "/tmp/caller-selected.json"
    else:
        registration["policy_spec"]["embodiment_id"] = "universal_robots_ur5e"
    write(retained.destination / "proposal.json", retained.proposal)
    with pytest.raises(ValueError, match="fixed_registration_changed"):
        installer.apply(retained.request, **retained.options)
    assert contract.read(retained.registry_path) == retained.registry


def test_retained_model_bytes_and_parent_symlinks_are_verified(retained, tmp_path):
    retained.model.write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="prepared_model_changed"):
        installer.apply(retained.request, **retained.options)
    alias = tmp_path / "alias"
    alias.symlink_to(retained.options["state_root"], target_is_directory=True)
    with pytest.raises(ValueError, match="symlink_path_refused"):
        contract.read(alias / retained.request["installation_id"] / "request.json")


def test_crash_after_registry_replace_resumes_receipt_without_second_mutation(retained, monkeypatch):
    original = installer._write_once
    def interrupted(path, value):
        if path.name == "receipt.json":
            raise OSError("development_only crash after replace")
        original(path, value)
    monkeypatch.setattr(installer, "_write_once", interrupted)
    with pytest.raises(OSError, match="development_only"):
        installer.apply(retained.request, **retained.options)
    installed = retained.registry_path.read_bytes()
    monkeypatch.setattr(installer, "_write_once", original)
    assert installer.apply(retained.request, **retained.options)["status"] == "installed"
    assert retained.registry_path.read_bytes() == installed


def test_concurrent_same_request_has_one_durable_registry_transition(retained):
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: installer.apply(retained.request, **retained.options), range(4)))
    assert all(value == results[0] for value in results)
    audit = retained.options["audit_root"] / retained.request["installation_id"]
    assert sorted(path.name for path in audit.iterdir()) == ["after.json", "before.json", "intent.json", "receipt.json", "request.json"]


def test_privileged_storage_and_lock_cannot_be_replaced_by_service_user(retained):
    retained.registry_path.parent.chmod(0o777)
    with pytest.raises(ValueError, match="privileged_directory_ownership_invalid"):
        installer.apply(retained.request, **retained.options)
    retained.registry_path.parent.chmod(0o755)
    audit = retained.options["audit_root"]
    (audit / ".lock").symlink_to(retained.registry_path)
    with pytest.raises(OSError):
        installer.apply(retained.request, **retained.options)
    assert contract.read(retained.registry_path) == retained.registry


def test_audit_is_group_readable_under_actual_private_transient_umask(retained):
    from operator_door.fsview import FileView
    audit = retained.options["audit_root"]
    audit.mkdir(mode=0o750)
    audit.chmod(0o2750)  # The installer provisions blueprint-door group here.
    previous = os.umask(0o077)
    try:
        installer.apply(retained.request, **retained.options)
    finally:
        os.umask(previous)
    destination = audit / retained.request["installation_id"]
    assert destination.stat().st_mode & 0o2777 == 0o2750
    receipt = destination / "receipt.json"
    assert receipt.stat().st_mode & 0o777 == 0o640
    view = FileView(DoorConfig(read_roots=(str(audit),)))
    payload, _ = view.read_range(str(receipt))
    assert json.loads(payload)["status"] == "installed"


@pytest.mark.parametrize("yam", [False, True])
def test_spool_uses_fixed_networkless_privilege_separation(tmp_path, yam):
    config = DoorConfig(state_root=str(tmp_path / "door"))
    for sub in ("pending", "processing", "completed", "results"):
        (Path(config.spool_root) / sub).mkdir(parents=True)
    req = request(yam=yam)
    kind = "install-public-yam-policy" if yam else "install-public-piper-policy"
    body = {"kind": kind, "request": req}
    first = enqueue(config, body, requested_by="development-only", operation_key="install-" + req["installation_id"])
    assert enqueue(config, body, requested_by="development-only", operation_key="install-" + req["installation_id"]) == first
    calls = []
    process_spool(config, runner=SimpleNamespace(run=lambda argv, timeout: calls.append(argv) or CommandResult(0, "", "")))
    launch, = calls
    assert launch[-1] == config.install_root + "/door-install-public-piper-policy.sh"
    assert "--property=PrivateNetwork=yes" in launch
    env = dict(value.removeprefix("--setenv=").split("=", 1) for value in launch if value.startswith("--setenv="))
    assert env["DOOR_SERVICE_USER"] == "blueprint"
    assert env["DOOR_PUBLIC_POLICY_OPERATION"] == kind
    assert json.loads(env["DOOR_PUBLIC_POLICY_REQUEST"]) == req
    assert not any("TOKEN" in key or "KEY" in key for key in env)


def test_yam_request_cannot_cross_piper_operation_or_impersonate_other_checkpoint():
    value = request(yam=True)
    assert validate_request({"kind": "install-public-yam-policy", "request": value})["request"] == value
    assert required_scope("install-public-yam-policy") == "operate"
    for kind, body in (("install-public-piper-policy", value), ("install-public-yam-policy", request()),
                       ("install-public-yam-policy", request(yam=True, checkpoint_id=contract.CHECKPOINT_ID)),
                       ("install-public-yam-policy", request(yam=True, team_id="DreamLabs")),
                       ("install-public-yam-policy", request(yam=True, profile_file="/tmp/arbitrary"))):
        with pytest.raises(RequestRefused):
            validate_request({"kind": kind, "request": body})


def test_yam_rights_reference_verified_publisher_statement_not_missing_hf_license():
    req = request(yam=True)
    profile = contract.load_profile(req, PROFILE_ROOT)
    rights = contract.rights_record(req, profile)
    assert rights["license_basis"] == "publisher_model_license_statement_in_pinned_repository_readme"
    assert rights["hf_model_card_license_field"] is None
    assert "model_card_sha256" not in rights and "model_card_url" not in rights
    assert rights["license_declaration"] == "This model is licensed under Apache 2.0."
    assert "research and educational" in rights["publisher_intended_use"]
    assert "6070080a20321b4f498ab30f28e1d09ac465edb7/README.md" in rights["license_evidence_url"]
    assert rights["private_vendor_authorization_claimed"] is False
    from blueprint_pipeline.team_candidate_quick10_admission import validate_candidate_rights
    validate_candidate_rights(rights, team_id=contract.TEAM_ID, checkpoint_id=contract.YAM_CHECKPOINT_ID,
                              candidate_id=contract.YAM_CANDIDATE_ID)


def test_yam_manifest_changes_only_declared_camera_optics_and_native_clock(retained_yam):
    original = retained_yam.original["robot_embodiment"]
    result = contract.expected_manifest(original, retained_yam.profile)
    assert result["control_hz"] == 60 and result["action_chunk_rows"] == 30
    assert [camera["horizontal_fov_deg"] for camera in result["cameras"]] == [69.4, 87.0, 87.0]
    for current, previous in zip(result["cameras"], original["cameras"], strict=True):
        assert (current["width"], current["height"]) == (640, 360)
        for field in ("mount_link", "position_m", "orientation_xyzw"):
            assert current[field] == previous[field]
    restored = copy.deepcopy(result)
    for field in ("cameras", "control_hz", "action_chunk_rows", "embodiment_digest"):
        restored[field] = original[field]
    assert restored == original
    assert "not_publisher_or_site_calibration" in retained_yam.registration["policy_spec"]["binding"]["camera_binding_reference"]


def test_yam_install_preserves_piper_and_has_independent_retry_receipt(retained_yam):
    fixture = retained_yam
    piper = {"robot_policy": {"development_only_existing_piper": "preserve_exactly"}}
    for settings in [fixture.registry["derived_profiles"]["defaults"], *fixture.registry["profiles"]]:
        settings["screening_candidates"][contract.CANDIDATE_ID] = piper
    fixture.registry["registry_digest"] = contract.digest(fixture.registry, "registry_digest")
    write(fixture.registry_path, fixture.registry)
    fixture.request["expected_registry_digest"] = fixture.registry["registry_digest"]
    fixture.request["request_digest"] = contract.digest(fixture.request, "request_digest")
    write(fixture.destination / "request.json", fixture.request)
    fixture.proposal["request_digest"] = fixture.request["request_digest"]
    write(fixture.destination / "proposal.json", fixture.proposal)
    receipt = installer.apply(fixture.request, **fixture.options)
    assert receipt["schema_version"] == "molmoact2_yam_policy_install_receipt.v1"
    assert receipt["candidate_id"] == contract.YAM_CANDIDATE_ID
    assert receipt["provider_allocations_performed"] == 0 and receipt["native_policy_execution_validated"] is False
    assert installer.apply(fixture.request, **fixture.options) == receipt
    current = contract.read(fixture.registry_path)
    for settings in [current["derived_profiles"]["defaults"], *current["profiles"]]:
        assert settings["screening_candidates"][contract.CANDIDATE_ID] == piper
        assert settings["screening_candidates"].pop(contract.YAM_CANDIDATE_ID) == {"robot_policy": fixture.registration}
    current["registry_digest"] = fixture.registry["registry_digest"]
    assert current == fixture.registry


@pytest.mark.parametrize("name", ["molmoact2_yam_publisher_readme.md", "molmoact2_yam_publisher_license.txt"])
def test_yam_publisher_license_evidence_cannot_be_replaced(retained_yam, tmp_path, name):
    evidence = tmp_path / "installed-evidence"
    shutil.copytree(PROFILE_ROOT, evidence)
    (evidence / name).write_text("development_only changed publisher license statement")
    with pytest.raises(ValueError, match="model_license_evidence_changed"):
        installer.apply(retained_yam.request, **{**retained_yam.options, "profile_root": evidence})
    assert contract.read(retained_yam.registry_path) == retained_yam.registry


@pytest.mark.parametrize("field", ["camera_pose", "model", "private_rights", "credentials", "policy"])
def test_yam_helper_proposal_cannot_broaden_fixed_registration(retained_yam, field):
    registration = retained_yam.proposal["registration"]
    if field == "camera_pose":
        manifest_path = Path(registration["manifest"]["path"])
        manifest = contract.read(manifest_path)
        manifest["cameras"][0]["position_m"] = [9, 9, 9]
        write(manifest_path, manifest)
    elif field == "model":
        registration["model"]["path"] = "/tmp/unrelated-robot.usd"
    elif field == "private_rights":
        registration["rights_binding"]["private_vendor_authorization_claimed"] = True
    elif field == "credentials":
        registration["credential_env"] = "DREAM_API_KEY"
    else:
        registration["policy_spec"]["endpoint_url"] = "https://arbitrary.invalid/infer"
    write(retained_yam.destination / "proposal.json", retained_yam.proposal)
    with pytest.raises(ValueError, match="fixed_(manifest|registration)_changed"):
        installer.apply(retained_yam.request, **retained_yam.options)
    assert contract.read(retained_yam.registry_path) == retained_yam.registry
