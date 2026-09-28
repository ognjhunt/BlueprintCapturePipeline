"""Private credentials must belong to the exact currently approved team policy."""

from __future__ import annotations

import json
import os

import pytest

from blueprint_pipeline import native_g1_team_policy_credentials as credentials
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest as digest
from blueprint_pipeline.native_g1_team_policy_authority import verify_g1_team_policy_authority
from tests.test_native_g1_team_provider_bundle import _inputs
from tests.test_native_g1_team_policy_run_request import NOW


TOKEN = "private-test-token-do-not-print"


def _write(path, value):
    value["registry_digest"] = digest(value, digest_field="registry_digest")
    path.write_text(json.dumps(value))
    path.chmod(0o600)


def _binding(tmp_path, monkeypatch):
    monkeypatch.setattr(credentials.time, "time", lambda: NOW)
    args, _ = _inputs(tmp_path, monkeypatch, endpoint=True)
    authority_args = args["authority_arguments"]
    authority = verify_g1_team_policy_authority(**authority_args)
    profile = authority["intent"]["request"]["policy_profile"]
    root = tmp_path / "private-registry"
    root.mkdir(mode=0o700)
    (root / "credentials").mkdir(mode=0o700)
    secret = root / "credentials/policy-token"
    secret.write_text(TOKEN + "\n")
    secret.chmod(0o600)
    registry = {
        "schema_version": credentials.SCHEMA,
        "entries": [{
            "status": "active", "owner": profile["owner"],
            "profile_digest": profile["profile_digest"],
            "secret_ref": profile["delivery"]["auth_secret_ref"],
            "approved_origin": "https://policy.example.org",
            "credential_filename": "policy-token", "expires_at_epoch": NOW + 1800,
        }],
    }
    path = root / "registry.json"
    _write(path, registry)
    return {"registry_path": path, "authority_arguments": authority_args,
            "now_epoch": NOW}, registry, secret


def test_resolves_only_bound_file_and_exposes_secret_free_metadata(tmp_path, monkeypatch):
    args, _, secret = _binding(tmp_path, monkeypatch)
    binding = credentials.resolve_g1_team_policy_credential(**args)
    assert binding.credential_file == secret
    receipt = binding.safe_receipt()
    assert receipt["status"] == "resolved_not_staged"
    assert receipt["provider_mutation_performed"] is False
    assert receipt["credential_value_included"] is False
    assert TOKEN not in json.dumps(receipt) + repr(binding)
    assert str(secret) not in json.dumps(receipt) + repr(binding)
    assert binding.recheck() == binding.safe_receipt()


@pytest.mark.parametrize("field,value", [
    ("owner", {"type": "robot_team", "id": "foreign-team"}),
    ("profile_digest", "sha256:" + "f" * 64),
    ("approved_origin", "https://foreign.example.org"),
    ("secret_ref", "secretref:foreign/policy"),
    ("status", "revoked"),
    ("expires_at_epoch", NOW),
    ("expires_at_epoch", True),
    ("credential_filename", "../policy-token"),
])
def test_rejects_resealed_foreign_revoked_expired_or_traversing_entry(tmp_path, monkeypatch, field, value):
    args, registry, _ = _binding(tmp_path, monkeypatch)
    registry["entries"][0][field] = value
    _write(args["registry_path"], registry)
    with pytest.raises(ValueError, match="g1_team_credential_"):
        credentials.resolve_g1_team_policy_credential(**args)


def test_duplicate_reference_and_unknown_fields_refuse(tmp_path, monkeypatch):
    args, registry, _ = _binding(tmp_path, monkeypatch)
    registry["entries"].append(dict(registry["entries"][0]))
    _write(args["registry_path"], registry)
    with pytest.raises(ValueError, match="registry_invalid"):
        credentials.resolve_g1_team_policy_credential(**args)


@pytest.mark.parametrize("target", ["registry", "secret", "directory"])
def test_unsafe_permissions_refuse(tmp_path, monkeypatch, target):
    args, _, secret = _binding(tmp_path, monkeypatch)
    path = {"registry": args["registry_path"], "secret": secret, "directory": secret.parent}[target]
    path.chmod(0o777)
    with pytest.raises(ValueError, match="path_invalid"):
        credentials.resolve_g1_team_policy_credential(**args)


@pytest.mark.parametrize("target", ["registry", "secret", "directory"])
def test_symlinked_files_or_ancestor_refuse(tmp_path, monkeypatch, target):
    args, _, secret = _binding(tmp_path, monkeypatch)
    path = {"registry": args["registry_path"], "secret": secret, "directory": secret.parent}[target]
    moved = path.with_name(path.name + "-real")
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=target == "directory")
    with pytest.raises(ValueError, match="path_invalid"):
        credentials.resolve_g1_team_policy_credential(**args)


def test_hardlinked_secret_refuses(tmp_path, monkeypatch):
    args, _, secret = _binding(tmp_path, monkeypatch)
    os.link(secret, secret.with_name("alias"))
    with pytest.raises(ValueError, match="path_invalid"):
        credentials.resolve_g1_team_policy_credential(**args)


@pytest.mark.parametrize("value", ["", "space token", "line\nbreak", "x" * 4097, "nul\0token"])
def test_malformed_token_refuses_without_exposing_value(tmp_path, monkeypatch, value):
    args, _, secret = _binding(tmp_path, monkeypatch)
    secret.write_text(value)
    with pytest.raises(ValueError, match="value_invalid") as error:
        credentials.resolve_g1_team_policy_credential(**args)
    assert str(error.value) == "g1_team_credential_value_invalid"


def test_recheck_reopens_operator_approval_and_registry(tmp_path, monkeypatch):
    args, registry, _ = _binding(tmp_path, monkeypatch)
    binding = credentials.resolve_g1_team_policy_credential(**args)
    registry["entries"][0]["status"] = "revoked"
    _write(args["registry_path"], registry)
    with pytest.raises(ValueError, match="g1_team_credential_"):
        binding.recheck()
    registry["entries"][0]["status"] = "active"
    _write(args["registry_path"], registry)
    approval = args["authority_arguments"]["approval_path"]
    value = json.loads(approval.read_text())
    value["site_observation_exchange_authorized"] = False
    value["approval_digest"] = canonical_digest(value, digest_field="approval_digest")
    approval.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="approval"):
        binding.recheck()


def test_recheck_detects_replaced_secret_and_changed_valid_authority(tmp_path, monkeypatch):
    args, _, secret = _binding(tmp_path, monkeypatch)
    binding = credentials.resolve_g1_team_policy_credential(**args)
    secret.write_text("another-valid-token")
    with pytest.raises(ValueError, match="changed"):
        binding.recheck()


def test_recheck_uses_current_time_and_binds_changed_valid_approval(tmp_path, monkeypatch):
    args, _, _ = _binding(tmp_path, monkeypatch)
    binding = credentials.resolve_g1_team_policy_credential(**args)
    monkeypatch.setattr(credentials.time, "time", lambda: NOW + 1800)
    with pytest.raises(ValueError, match="g1_team_credential_"):
        binding.recheck()
    monkeypatch.setattr(credentials.time, "time", lambda: NOW)
    approval_path = args["authority_arguments"]["approval_path"]
    approval = json.loads(approval_path.read_text())
    approval["operator_reviewer"] = "changed but valid reviewer"
    approval["approval_digest"] = canonical_digest(approval, digest_field="approval_digest")
    approval_path.write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="binding_changed"):
        binding.recheck()
