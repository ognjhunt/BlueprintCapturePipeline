"""Atomic, site-scoped G1 registration using only root-installed stdlib code."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile


def _sibling(name):
    spec = importlib.util.spec_from_file_location("_installed_" + name, Path(__file__).with_name(name + ".py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = _sibling("native_g1_registration_contract")
common = contract.common
atomic = _sibling("public_policy_registration")


def apply(request, *, registry_path=common.REGISTRY_PATH, state_root=common.STATE_ROOT,
          audit_root=atomic.AUDIT_ROOT, profile_root=Path(__file__).parent,
          shared_registry_path=contract.SHARED_REGISTRY, source_queue=contract.SOURCE_QUEUE):
    request = contract.validate_request(request)
    registry_path, audit_root = common.safe_path(registry_path), common.safe_path(audit_root)
    audit_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    for directory in (registry_path.parent, audit_root):
        info = directory.stat()
        contract.require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and not info.st_mode & 0o022,
                         "privileged_directory_ownership_invalid")
    # Exactly the existing public-policy lock: the two installer kinds cannot
    # replace the same registry concurrently.
    descriptor = os.open(audit_root / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(descriptor, "r+") as lock:
        info = os.fstat(lock.fileno())
        contract.require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and not info.st_mode & 0o022, "lock_invalid")
        fcntl.flock(lock, fcntl.LOCK_EX)
        destination = common.safe_path(audit_root / request["installation_id"])
        destination.mkdir(exist_ok=True, mode=0o750)
        info = destination.stat()
        contract.require(info.st_uid == os.geteuid() and not info.st_mode & 0o022, "audit_ownership_invalid")
        destination.chmod(0o2750)
        atomic._write_once(destination / "request.json", request)
        registry = common.read(registry_path)
        contract.require(registry.get("schema_version") == "blueprint.team_candidate_quick10_registry.v1"
            and registry.get("registry_digest") == common.digest(registry, "registry_digest")
            and isinstance(registry.get("profiles"), list), "registry_invalid")
        if (destination / "receipt.json").exists():
            return common.read(destination / "receipt.json")
        if (destination / "intent.json").exists():
            intent = common.read(destination / "intent.json")
            updated = common.read(destination / "after.json")
            contract.require(intent["request_digest"] == request["request_digest"]
                and updated["registry_digest"] == intent["after_registry_digest"] == common.digest(updated, "registry_digest"), "audit_identity_conflict")
        else:
            contract.require(registry["registry_digest"] == request["expected_registry_digest"], "registry_compare_failed")
            proposal = contract.verify_proposal(request, registry, state_root=state_root, profile_root=profile_root,
                shared_registry_path=shared_registry_path, source_queue=source_queue)
            updated = json.loads(common.canonical(registry))
            registrations = proposal["registrations"]
            if registrations:
                sites = updated.setdefault("site_screening_candidates", {})
                contract.require(isinstance(sites, dict), "site_registry_invalid")
                previous = sites.get(request["site_id"])
                if previous is not None:
                    contract.require(isinstance(previous, dict) and set(previous) == {"source", "screening_candidates"}
                        and isinstance(previous["screening_candidates"], dict), "site_registry_invalid")
                # Never propagate old task records into a new revision.
                candidates = (dict(previous["screening_candidates"]) if previous and previous["source"] == request["source"] else {})
                candidates.update(registrations)
                sites[request["site_id"]] = {"source": request["source"], "screening_candidates": candidates}
            updated["registry_digest"] = common.digest(updated, "registry_digest")
            intent = {"request_digest": request["request_digest"], "before_registry_digest": registry["registry_digest"],
                "after_registry_digest": updated["registry_digest"], "registered_count": len(registrations), "results": proposal["results"]}
            atomic._write_once(destination / "before.json", registry)
            atomic._write_once(destination / "after.json", updated)
            atomic._write_once(destination / "intent.json", intent)
        contract.require(registry["registry_digest"] in {intent["before_registry_digest"], intent["after_registry_digest"]}, "registry_compare_failed")
        if registry["registry_digest"] != intent["after_registry_digest"]:
            info = registry_path.stat()
            descriptor, temporary = tempfile.mkstemp(prefix=".native-g1-", dir=registry_path.parent)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write((json.dumps(updated, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode())
                    os.fchmod(stream.fileno(), stat.S_IMODE(info.st_mode))
                    os.fchown(stream.fileno(), info.st_uid, info.st_gid)
                    stream.flush()
                    os.fsync(stream.fileno())
                contract.require(common.read(registry_path)["registry_digest"] == intent["before_registry_digest"], "registry_compare_failed")
                os.replace(temporary, registry_path)
                atomic._sync_directory(registry_path.parent)
            finally:
                Path(temporary).unlink(missing_ok=True)
        receipt = {"schema_version": contract.RECEIPT_SCHEMA, "status": "installed",
            "installation_id": request["installation_id"], "site_id": request["site_id"], **intent,
            "claim_ceiling": "native_g1_source_registration_only", "execution_authority_granted": False,
            "live_task_api_queried": False, "native_policy_execution_validated": False,
            "candidate_policy_queried": False, "provider_allocations_performed": 0}
        atomic._write_once(destination / "receipt.json", receipt)
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", required=True)
    args = parser.parse_args(argv)
    contract.require(os.geteuid() == 0, "privileged_installer_required")
    print(json.dumps(apply(common.decode(args.request_json)), sort_keys=True))


if __name__ == "__main__":
    main()
