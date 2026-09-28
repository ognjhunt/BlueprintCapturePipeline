"""Resolve a protected endpoint credential for one currently approved policy.

The registry is operator controlled, not supplied by the team request. Only a
basename under its private credentials directory can be selected. This module
does not disclose credentials, contact an endpoint, or allocate resources.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest as digest
from .native_g1_team_policy_authority import verify_g1_team_policy_authority


SCHEMA = "native_g1_team_policy_credential_registry.v1"
_FIELDS = frozenset({"schema_version", "entries", "registry_digest"})
_ENTRY_FIELDS = frozenset({
    "status", "owner", "profile_digest", "secret_ref", "approved_origin",
    "credential_filename", "expires_at_epoch",
})
_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _check_path(path: Path, *, directory: bool = False) -> os.stat_result:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("g1_team_credential_path_invalid")
    try:
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError("g1_team_credential_path_invalid")
        info = path.stat()
    except OSError:
        raise ValueError("g1_team_credential_path_invalid") from None
    if (
        info.st_uid not in {0, os.geteuid()}
        or (directory and (not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o022))
        or (not directory and (
            not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_nlink != 1
        ))
    ):
        raise ValueError("g1_team_credential_path_invalid")
    return info


def _snapshot(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_private(path: Path, *, maximum_bytes: int, allow_empty: bool = False) -> tuple[bytes, tuple[int, ...]]:
    info = _check_path(path)
    if info.st_size > maximum_bytes or (not allow_empty and info.st_size == 0):
        raise ValueError("g1_team_credential_registry_invalid")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            value = stream.read(maximum_bytes + 1)
            after = os.fstat(stream.fileno())
        if (_snapshot(before) != _snapshot(info) or _snapshot(after) != _snapshot(info)
                or len(value) > maximum_bytes or len(value) != info.st_size):
            raise ValueError("g1_team_credential_file_changed")
    except OSError:
        raise ValueError("g1_team_credential_path_invalid") from None
    return value, _snapshot(info)


def _validated_token(raw: bytes) -> str:
    token = raw.removesuffix(b"\n")
    if not 1 <= len(token) <= 4096 or any(not 33 <= byte <= 126 for byte in token):
        raise ValueError("g1_team_credential_value_invalid")
    return token.decode("ascii")


@dataclass(frozen=True, repr=False)
class G1TeamPolicyCredentialBinding:
    """Private transport input; only safe_receipt may enter durable run records."""

    credential_file: Path = field(repr=False)
    _registry_path: Path = field(repr=False)
    _authority_arguments: Mapping[str, Any] = field(repr=False)
    _file_snapshot: tuple[int, ...] = field(repr=False)
    _receipt: Mapping[str, Any] = field(repr=False)

    def __repr__(self) -> str:
        return "G1TeamPolicyCredentialBinding(private_file_bound=True)"

    def safe_receipt(self) -> dict[str, Any]:
        # Roundtrip also prevents callers changing the stored nested owner map.
        return json.loads(json.dumps(self._receipt))

    def recheck(self) -> dict[str, Any]:
        current = resolve_g1_team_policy_credential(
            registry_path=self._registry_path, authority_arguments=self._authority_arguments,
        )
        if (current._file_snapshot != self._file_snapshot
                or current.credential_file != self.credential_file
                or current.safe_receipt() != self.safe_receipt()):
            raise ValueError("g1_team_credential_binding_changed")
        return current.safe_receipt()

    def read_for_endpoint_probe(self) -> str:
        """Private HTTPS input only; re-open authority and the bound file."""
        self.recheck()
        raw, snapshot = _read_private(self.credential_file, maximum_bytes=16384, allow_empty=True)
        if snapshot != self._file_snapshot:
            raise ValueError("g1_team_credential_binding_changed")
        token = _validated_token(raw)
        self.recheck()
        return token


def resolve_g1_team_policy_credential(
    *, registry_path: Path, authority_arguments: Mapping[str, Any],
    now_epoch: float | None = None,
) -> G1TeamPolicyCredentialBinding:
    """Reopen rights and resolve one exact active operator-owned secret reference."""

    now = time.time() if now_epoch is None else now_epoch
    if type(now) not in (int, float) or not math.isfinite(now):
        raise ValueError("g1_team_credential_time_invalid")
    arguments = dict(authority_arguments)
    arguments.pop("now_epoch", None)
    authority = verify_g1_team_policy_authority(**arguments, now_epoch=now)
    profile = authority["intent"]["request"]["policy_profile"]
    approval = authority["operator_approval"]
    binding = approval["runtime_binding"]
    if profile["delivery"]["mode"] != "authenticated_endpoint":
        raise ValueError("g1_team_credential_endpoint_required")
    path = Path(registry_path)
    _check_path(path.parent, directory=True)
    _check_path(path.parent / "credentials", directory=True)
    raw, _ = _read_private(path, maximum_bytes=1024 * 1024)
    try:
        registry = json.loads(raw)
        if (not isinstance(registry, dict) or set(registry) != _FIELDS
                or registry.get("schema_version") != SCHEMA
                or registry.get("registry_digest") != digest(registry, digest_field="registry_digest")
                or not isinstance(registry.get("entries"), list)
                or not 1 <= len(registry["entries"]) <= 1000):
            raise ValueError("g1_team_credential_registry_invalid")
    except (UnicodeError, json.JSONDecodeError, TypeError):
        raise ValueError("g1_team_credential_registry_invalid") from None
    entries = registry["entries"]
    references = []
    for entry in entries:
        if (not isinstance(entry, dict) or set(entry) != _ENTRY_FIELDS
                or not isinstance(entry.get("secret_ref"), str)):
            raise ValueError("g1_team_credential_registry_invalid")
        references.append(entry["secret_ref"])
    if len(set(references)) != len(references):
        raise ValueError("g1_team_credential_registry_invalid")
    selected = [entry for entry in entries if entry["secret_ref"] == binding["resolved_secret_ref"]]
    if len(selected) != 1:
        raise ValueError("g1_team_credential_binding_invalid")
    entry = selected[0]
    expiry = entry["expires_at_epoch"]
    filename = entry["credential_filename"]
    if (entry["status"] != "active" or entry["owner"] != profile["owner"]
            or entry["profile_digest"] != profile["profile_digest"]
            or entry["approved_origin"] != binding["approved_origin"]
            or type(expiry) not in (int, float) or not math.isfinite(expiry) or now >= expiry
            or not isinstance(filename, str) or _FILENAME.fullmatch(filename) is None):
        raise ValueError("g1_team_credential_binding_invalid")
    secret_path = path.parent / "credentials" / filename
    raw, snapshot = _read_private(secret_path, maximum_bytes=16384, allow_empty=True)
    # One final newline is permitted in a canonical token file. Never strip
    # arbitrary whitespace into a different credential than the operator wrote.
    _validated_token(raw)
    receipt = {
        "schema_version": "native_g1_team_policy_credential_binding.v1",
        "status": "resolved_not_staged", "owner": profile["owner"],
        "intent_digest": authority["intent"]["intent_digest"],
        "profile_digest": profile["profile_digest"],
        "operator_approval_digest": approval["approval_digest"],
        "secret_ref": entry["secret_ref"], "approved_origin": entry["approved_origin"],
        "credential_entry_digest": digest(entry),
        "expires_at_epoch": expiry, "credential_value_included": False,
        "provider_mutation_performed": False, "claim_ceiling": "development_only",
    }
    receipt["receipt_digest"] = digest(receipt, digest_field="receipt_digest")
    return G1TeamPolicyCredentialBinding(secret_path, path, arguments, snapshot, receipt)
