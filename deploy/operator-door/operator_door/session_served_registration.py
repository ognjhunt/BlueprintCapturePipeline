"""Add or remove one session-served policy entry in the root-owned registry; stdlib-only installed code.

Plan 2.1's registry row (``blueprint_pipeline.session_served_policy``): a policy Blueprint serves itself
over the standard connection is one ``screening_candidates.<id>.session_served_policy`` entry. This
installer follows the public-policy registration pattern (``public_policy_registration``): the request
names the registry digest it was prepared against and an authorization reference; the change is a
compare-and-swap of ``/etc/blueprint/team-evaluation-registry.json`` with ``before``, ``after``,
``intent`` and ``receipt`` kept under the door's audit root. Root never imports the active pipeline.

The entry is checked against a fixed copy of the pipeline's schema (``SERVERS``, kept equal by a test):
known server, a native candidate that server serves, bounded memory, startup and port. An entry may never
take a packaged candidate's id (``PACKAGED_CANDIDATE_IDS``, which includes every native id any server
re-serves) nor an id the registry already uses for something else. Removing touches only that entry.
Planning reads the registry on the next pass; with the never-twice rule
(``beta_policy_screening_tested_pairs``) policies that already have results are not run again.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any

REQUEST_SCHEMA = "blueprint.session_served_policy_registration_request.v1"
RECEIPT_SCHEMA = "blueprint.session_served_policy_registration_receipt.v1"
ENTRY_SCHEMA = "blueprint.session_served_policy.v1"
FIELD = "session_served_policy"
REGISTRY_PATH = Path("/etc/blueprint/team-evaluation-registry.json")
REGISTRY_SCHEMA = "blueprint.team_candidate_quick10_registry.v1"
AUDIT_ROOT = Path("/var/lib/blueprint-operator-door/session-served-registrations")
#: The pipeline's ``session_served_policy.SERVERS``: server -> the native candidates it may re-serve.
SERVERS = {"openpi_droid_contract_server.v1": ("pi05_droid", "pi0_droid_jointpos_polaris",
                                               "pi0_droid_jointpos_100k_polaris", "pi0_fast_droid_jointpos_polaris")}
#: Ids a session-served entry may never shadow: every native candidate a server re-serves.
PACKAGED_CANDIDATE_IDS = frozenset(row for rows in SERVERS.values() for row in rows)
ENTRY_KEYS = frozenset({"schema_version", "server", "native_candidate_id", "display_name",
                        "minimum_gpu_memory_gb", "startup_timeout_seconds", "port"})
REQUEST_KEYS = frozenset({"schema_version", "action", "candidate_id", "entry", "expected_registry_digest",
                          "authorization_reference", "request_digest"})
_CANDIDATE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def require(condition: Any, reason: str) -> None:
    if not condition:
        raise ValueError("session_served_registration_" + reason)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value: dict, field: str = "") -> str:
    return "sha256:" + hashlib.sha256(canonical({k: v for k, v in value.items() if k != field} if field else value)).hexdigest()


def validate_entry(entry: Any, *, candidate_id: str) -> dict:
    require(isinstance(entry, dict) and not set(entry) - ENTRY_KEYS and entry.get("schema_version") == ENTRY_SCHEMA
            and entry.get("server") in SERVERS, "entry_invalid")
    require(entry.get("native_candidate_id") in SERVERS[entry["server"]], "native_candidate_unsupported")
    require(candidate_id not in PACKAGED_CANDIDATE_IDS, "candidate_id_is_packaged")
    memory = entry.get("minimum_gpu_memory_gb")
    require(memory is None or type(memory) in (int, float) and 0 < memory <= 1024, "gpu_memory_invalid")
    startup = entry.get("startup_timeout_seconds", 900)
    require(type(startup) is int and 60 <= startup <= 900, "startup_bound_invalid")
    port = entry.get("port", 8765)
    require(type(port) is int and 1024 <= port <= 65535 and port not in {8000, 5555}, "port_invalid")
    display = entry.get("display_name", candidate_id)
    require(isinstance(display, str) and display.strip() and len(display) <= 200 and display.isprintable(),
            "display_name_invalid")
    return dict(entry)


def validate_request(value: Any) -> dict:
    require(isinstance(value, dict) and set(value) == REQUEST_KEYS and value.get("schema_version") == REQUEST_SCHEMA,
            "request_schema_invalid")
    require(value["action"] in {"add", "remove"}, "action_invalid")
    require(isinstance(value["candidate_id"], str) and _CANDIDATE.fullmatch(value["candidate_id"]),
            "candidate_id_invalid")
    require(value["candidate_id"] not in PACKAGED_CANDIDATE_IDS, "candidate_id_is_packaged")
    if value["action"] == "add":
        validate_entry(value["entry"], candidate_id=value["candidate_id"])
    else:
        require(value["entry"] is None, "remove_takes_no_entry")
    require(isinstance(value["expected_registry_digest"], str) and _DIGEST.fullmatch(value["expected_registry_digest"]),
            "expected_registry_digest_invalid")
    reference = value["authorization_reference"]
    require(isinstance(reference, str) and 1 <= len(reference) <= 300 and reference.isprintable()
            and reference == reference.strip(), "authorization_reference_invalid")
    require(value["request_digest"] == digest(value, "request_digest"), "request_digest_invalid")
    return dict(value)


def seal_request(value: dict) -> dict:
    """The request with its digest (the client and tests build requests this way)."""
    return {**value, "request_digest": digest(value, "request_digest")}


def updated_registry(registry: dict, request: dict) -> dict:
    """The registry after the one change; refuses a change that would touch anything else."""
    require(registry.get("schema_version") == REGISTRY_SCHEMA and registry.get("registry_digest")
            == digest(registry, "registry_digest") and isinstance(registry.get("profiles"), list), "registry_invalid")
    updated = json.loads(canonical(registry))
    defaults = (updated.get("derived_profiles") or {}).get("defaults")
    require(isinstance(defaults, dict), "registry_derived_defaults_missing")
    key, found = request["candidate_id"], False
    for settings in [defaults, *updated["profiles"]]:
        require(isinstance(settings, dict), "registry_profile_invalid")
        candidates = settings.get("screening_candidates") or {}
        require(isinstance(candidates, dict), "registry_candidates_invalid")
        existing = candidates.get(key)
        # An id the registry already uses for anything but a session-served entry is never taken over.
        require(existing is None or isinstance(existing, dict) and set(existing) == {FIELD}, "candidate_id_taken")
        if request["action"] == "add":
            settings.setdefault("screening_candidates", {})[key] = {FIELD: dict(request["entry"])}
        elif existing is not None:
            found = True
            del candidates[key]
    require(request["action"] == "add" or found, "entry_absent")
    updated["registry_digest"] = digest(updated, "registry_digest")
    return updated


def _safe(path: Path) -> Path:
    require(path.is_absolute() and not any(part.is_symlink() for part in (path, *path.parents)), "symlink_path_refused")
    return path


def _read(path: Path) -> dict:
    fd = os.open(_safe(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= 8 * 1024**2 and not info.st_mode & 0o022,
                "record_file_invalid")
        value = json.loads(stream.read())
    require(isinstance(value, dict), "record_invalid")
    return value


def _write_once(path: Path, value: dict) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if path.exists():
        require(path.read_bytes() == payload, "audit_identity_conflict")
        return
    fd, temporary = tempfile.mkstemp(prefix=".served-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            os.fchmod(stream.fileno(), 0o640)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def packaged_ids(catalog_path: Path) -> frozenset[str]:
    """Every candidate id the active release's catalog ships (its JSON, read as data)."""
    catalog = json.loads(_safe(catalog_path).read_text(encoding="utf-8"))
    rows = catalog.get("candidates") if isinstance(catalog, dict) else None
    require(isinstance(rows, list) and rows and all(isinstance(row, dict) and isinstance(row.get("candidate_id"), str)
                                                    for row in rows), "catalog_unreadable")
    return frozenset(row["candidate_id"] for row in rows)


def apply(request: dict, *, registry_path: Path = REGISTRY_PATH, audit_root: Path = AUDIT_ROOT,
          catalog_path: Path | None = None, require_owner: bool = True) -> dict:
    request = validate_request(request)
    if catalog_path is not None:
        require(request["candidate_id"] not in packaged_ids(catalog_path), "candidate_id_is_packaged")
    _safe(registry_path)
    _safe(audit_root)
    audit_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    if require_owner:
        for directory in (registry_path.parent, audit_root):
            info = directory.stat()
            require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and not info.st_mode & 0o022,
                    "privileged_directory_ownership_invalid")
    lock_fd = os.open(audit_root / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock_fd, "r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        destination = audit_root / request["request_digest"][7:39]
        destination.mkdir(mode=0o750, exist_ok=True)
        _write_once(destination / "request.json", request)
        if (destination / "receipt.json").exists():
            return _read(destination / "receipt.json")
        registry = _read(registry_path)
        if (destination / "intent.json").exists():
            intent = _read(destination / "intent.json")
            updated = _read(destination / "after.json")
        else:
            require(registry.get("registry_digest") == request["expected_registry_digest"], "registry_compare_failed")
            updated = updated_registry(registry, request)
            intent = {"request_digest": request["request_digest"], "before_registry_digest": registry["registry_digest"],
                      "after_registry_digest": updated["registry_digest"]}
            _write_once(destination / "before.json", registry)
            _write_once(destination / "after.json", updated)
            _write_once(destination / "intent.json", intent)
        require(registry["registry_digest"] in {intent["before_registry_digest"], intent["after_registry_digest"]},
                "registry_compare_failed")
        if registry["registry_digest"] != intent["after_registry_digest"]:
            info = registry_path.stat()
            fd, temporary = tempfile.mkstemp(prefix=".served-policy-", dir=registry_path.parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write((json.dumps(updated, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode())
                    os.fchmod(stream.fileno(), stat.S_IMODE(info.st_mode))
                    if os.geteuid() == 0:
                        os.fchown(stream.fileno(), info.st_uid, info.st_gid)
                    stream.flush()
                    os.fsync(stream.fileno())
                require(_read(registry_path)["registry_digest"] == intent["before_registry_digest"],
                        "registry_compare_failed")
                os.replace(temporary, registry_path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        receipt = {"schema_version": RECEIPT_SCHEMA, "status": "added" if request["action"] == "add" else "removed",
                   "candidate_id": request["candidate_id"], **intent,
                   "authorization_reference": request["authorization_reference"],
                   "candidate_policy_queried": False, "provider_allocations_performed": 0,
                   "claim_ceiling": "session_served_policy_registration_only"}
        _write_once(destination / "receipt.json", receipt)
        return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--catalog", required=True, help="the active release's config/beta_policy_candidates.v1.json")
    args = parser.parse_args(argv)
    try:
        require(os.geteuid() == 0, "privileged_installer_required")
        print(json.dumps(apply(json.loads(args.request_json), catalog_path=Path(args.catalog)), sort_keys=True))
        return 0
    except ValueError as exc:
        code = str(exc) if re.fullmatch(r"[a-z][a-z0-9_]{0,199}", str(exc)) else "session_served_registration_failed"
        print(json.dumps({"status": "refused", "code": code}, sort_keys=True))
        return 3


if __name__ == "__main__":
    sys.exit(main())
