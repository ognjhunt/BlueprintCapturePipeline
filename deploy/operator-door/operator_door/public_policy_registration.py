"""Fixed registry CAS; stdlib-only installed code, never active pipeline imports."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import sys

if __package__:
    from .public_policy_registration_contract import (
        CANDIDATE_ID, CHECKPOINT_ID, MODEL_ROOT, REGISTRY_PATH, STATE_ROOT, TEAM_ID,
        decode, digest, expected_manifest, expected_policy, file_record, read, require,
        retained_body, rights_record, safe_path, validate_request,
    )
else:
    # -I excludes the script directory. Load only this reviewed installed
    # sibling, never PYTHONPATH or the active pipeline checkout.
    _source = Path(__file__).resolve().with_name("public_policy_registration_contract.py")
    _spec = importlib.util.spec_from_file_location("public_policy_registration_contract", _source)
    assert _spec is not None and _spec.loader is not None
    _contract = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_contract)
    sys.modules[_spec.name] = _contract
    from public_policy_registration_contract import (
        CANDIDATE_ID, CHECKPOINT_ID, MODEL_ROOT, REGISTRY_PATH, STATE_ROOT, TEAM_ID,
        decode, digest, expected_manifest, expected_policy, file_record, read, require,
        retained_body, rights_record, safe_path, validate_request,
    )

AUDIT_ROOT = Path("/var/lib/blueprint-operator-door/policy-registrations")


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_once(path: Path, value: dict) -> None:
    safe_path(path)
    payload = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if path.exists():
        require(path.read_bytes() == payload, "audit_identity_conflict")
        return
    fd, temporary = tempfile.mkstemp(prefix=".install-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            os.fchmod(stream.fileno(), 0o640)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def verified_registration(request: dict, *, state_root: Path, model_root: Path, profile_root: Path) -> dict:
    profile = read(profile_root / "openwam_piper_registration_profile.json")
    require(file_record(profile_root / "openwam_piper_model_card.md")["sha256"] == profile["model_card_sha256"],
            "model_license_evidence_changed")
    _, original, model = retained_body(request, model_root=model_root)
    manifest = expected_manifest(original["robot_embodiment"])
    destination = safe_path(state_root / request["installation_id"])
    require(read(destination / "request.json") == request, "retained_request_changed")
    manifest_path = destination / (manifest["embodiment_digest"][7:] + ".robot_manifest.json")
    require(read(manifest_path) == manifest, "fixed_manifest_changed")
    # Construct every permitted value independently. Extra keys (including any
    # endpoint, credential, command or caller-selected path) cannot survive.
    expected = {"team_id": TEAM_ID, "checkpoint_id": CHECKPOINT_ID, "display_name": profile["display_name"],
        "model": model, "manifest": file_record(manifest_path), "policy_spec": expected_policy(manifest, profile),
        "rights_binding": rights_record(request, profile),
        "dependency_closure": original["model_record"]["dependency_closure"],
        "source_import_provenance": original["model_record"]["source_import_provenance"],
        "minimum_gpu_memory_gb": profile["minimum_gpu_memory_gb"], "deployment": profile["deployment"]}
    proposal = read(destination / "proposal.json")
    require(proposal == {"schema_version": "openwam_piper_policy_install_proposal.v1",
        "request_digest": request["request_digest"], "registration": expected,
        "candidate_policy_queried": False, "provider_allocations_performed": 0}, "fixed_registration_changed")
    return expected


def apply(request: dict, *, registry_path: Path = REGISTRY_PATH, state_root: Path = STATE_ROOT,
          model_root: Path = MODEL_ROOT, audit_root: Path = AUDIT_ROOT,
          profile_root: Path = Path(__file__).resolve().parent) -> dict:
    request = validate_request(request)
    safe_path(registry_path)
    safe_path(audit_root)
    # Registry parent and audit storage are controlled by the privileged caller,
    # never by the blueprint proposal builder. Refuse symlink lock substitution.
    audit_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    for directory in (registry_path.parent, audit_root):
        info = directory.stat()
        require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and not info.st_mode & 0o022,
                "privileged_directory_ownership_invalid")
    lock_fd = os.open(audit_root / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(lock_fd, "r+") as lock:
        info = os.fstat(lock.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and not info.st_mode & 0o022,
                "lock_ownership_invalid")
        fcntl.flock(lock, fcntl.LOCK_EX)
        destination = safe_path(audit_root / request["installation_id"])
        destination.mkdir(mode=0o750, exist_ok=True)
        info = destination.stat()
        require(info.st_uid == os.geteuid() and not info.st_mode & 0o022, "audit_ownership_invalid")
        # The transient uses umask0077. Preserve the installer-provisioned
        # blueprint-door group through setgid and make audit traversal explicit.
        destination.chmod(0o2750)
        _write_once(destination / "request.json", request)
        registry = read(registry_path)
        require(registry.get("schema_version") == "blueprint.team_candidate_quick10_registry.v1"
                and registry.get("registry_digest") == digest(registry, "registry_digest")
                and isinstance(registry.get("profiles"), list), "registry_invalid")
        if (destination / "receipt.json").exists():
            return read(destination / "receipt.json")
        if (destination / "intent.json").exists():
            intent = read(destination / "intent.json")
            require(intent["request_digest"] == request["request_digest"], "audit_identity_conflict")
            updated = read(destination / "after.json")
            require(updated["registry_digest"] == intent["after_registry_digest"] == digest(updated, "registry_digest"),
                    "audit_identity_conflict")
        else:
            require(registry["registry_digest"] == request["expected_registry_digest"], "registry_compare_failed")
            registration = verified_registration(request, state_root=state_root, model_root=model_root,
                                                  profile_root=profile_root)
            updated = json.loads(json.dumps(registry))
            defaults = (updated.get("derived_profiles") or {}).get("defaults")
            require(isinstance(defaults, dict), "registry_derived_defaults_missing")
            for settings in [defaults, *updated["profiles"]]:
                require(isinstance(settings, dict), "registry_profile_invalid")
                candidates = settings.setdefault("screening_candidates", {})
                require(isinstance(candidates, dict), "registry_candidates_invalid")
                candidates[CANDIDATE_ID] = {"robot_policy": registration}
            updated["registry_digest"] = digest(updated, "registry_digest")
            intent = {"request_digest": request["request_digest"],
                      "before_registry_digest": registry["registry_digest"],
                      "after_registry_digest": updated["registry_digest"]}
            _write_once(destination / "before.json", registry)
            _write_once(destination / "after.json", updated)
            _write_once(destination / "intent.json", intent)
        require(registry["registry_digest"] in {intent["before_registry_digest"], intent["after_registry_digest"]},
                "registry_compare_failed")
        if registry["registry_digest"] != intent["after_registry_digest"]:
            info = registry_path.stat()
            fd, temporary = tempfile.mkstemp(prefix=".public-policy-", dir=registry_path.parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write((json.dumps(updated, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode())
                    os.fchmod(stream.fileno(), stat.S_IMODE(info.st_mode))
                    os.fchown(stream.fileno(), info.st_uid, info.st_gid)
                    stream.flush()
                    os.fsync(stream.fileno())
                # Detect already-observed external writes; arbitrary root tools
                # must not race this final comparison and atomic replacement.
                require(read(registry_path)["registry_digest"] == intent["before_registry_digest"], "registry_compare_failed")
                os.replace(temporary, registry_path)
                _sync_directory(registry_path.parent)
            finally:
                Path(temporary).unlink(missing_ok=True)
        receipt = {"schema_version": "openwam_piper_policy_install_receipt.v1", "status": "installed",
            "installation_id": request["installation_id"], **intent, "candidate_id": CANDIDATE_ID,
            "preparation_authorization_id": request["preparation_authorization_id"],
            "preparation_publication_digest": request["preparation_publication_digest"],
            "native_policy_execution_validated": False, "candidate_policy_queried": False,
            "provider_allocations_performed": 0, "claim_ceiling": "public_policy_registration_only"}
        _write_once(destination / "receipt.json", receipt)
        return receipt


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", required=True)
    args = parser.parse_args(argv)
    require(os.geteuid() == 0, "privileged_installer_required")
    print(json.dumps(apply(decode(args.request_json)), sort_keys=True))


if __name__ == "__main__":
    main()
