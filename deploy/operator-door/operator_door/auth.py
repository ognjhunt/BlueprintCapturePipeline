"""Bearer-token identities for the operator door.

Only SHA-256 hashes live on the host. A token is issued on the operator's own
machine, its hash is copied here, and the plaintext goes straight into the
cloud environment's API-credential store, where the egress proxy attaches it to
requests without the agent's VM ever seeing it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "blueprint_operator_door_tokens.v1"
SCOPES = frozenset({"read", "operate", "deploy"})
MIN_TOKEN_LENGTH = 32
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class TokenStoreError(ValueError):
    pass


@dataclass(frozen=True)
class TokenIdentity:
    name: str
    scopes: frozenset[str]


def hash_token(token: str) -> str:
    return "sha256:" + hashlib.sha256(token.encode("utf-8")).hexdigest()


def _parse(document: Any) -> list[tuple[str, str, frozenset[str]]]:
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise TokenStoreError("token_file_schema_invalid")
    entries = document.get("tokens")
    if not isinstance(entries, list):
        raise TokenStoreError("token_file_schema_invalid")
    parsed: list[tuple[str, str, frozenset[str]]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise TokenStoreError("token_entry_invalid")
        name, digest, scopes = entry.get("name"), entry.get("sha256"), entry.get("scopes")
        if not isinstance(name, str) or not _NAME.match(name):
            raise TokenStoreError("token_name_invalid")
        if not isinstance(digest, str) or not _HASH.match(digest):
            raise TokenStoreError("token_hash_invalid")
        if not isinstance(scopes, list) or not scopes:
            raise TokenStoreError("token_scopes_invalid")
        for scope in scopes:
            if scope not in SCOPES:
                raise TokenStoreError(f"token_scope_unknown:{scope}")
        parsed.append((name, digest, frozenset(scopes)))
    return parsed


class TokenStore:
    """Verifies ``Authorization`` headers against the hashed token file."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._signature: tuple[int, int, int] | None = None
        self._entries: list[tuple[str, str, frozenset[str]]] = []

    def _load(self) -> list[tuple[str, str, frozenset[str]]]:
        try:
            info = self._path.stat()
        except FileNotFoundError:
            self._signature, self._entries = None, []
            return []
        if info.st_mode & 0o027:
            raise TokenStoreError("token_file_mode_too_open")
        signature = (info.st_mtime_ns, info.st_size, info.st_ino)
        if signature != self._signature:
            document = json.loads(self._path.read_text(encoding="utf-8"))
            self._entries = _parse(document)
            self._signature = signature
        return self._entries

    def verify(self, header: str | None) -> TokenIdentity | None:
        entries = self._load()
        parts = (header or "").split()
        if len(parts) != 2 or parts[0].lower() != "bearer" or len(parts[1]) < MIN_TOKEN_LENGTH:
            return None
        presented = hash_token(parts[1])
        match: TokenIdentity | None = None
        for name, digest, scopes in entries:
            # Compare against every entry so timing does not reveal position.
            if hmac.compare_digest(presented, digest) and match is None:
                match = TokenIdentity(name=name, scopes=scopes)
        return match


def _read_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema": SCHEMA, "tokens": []}
    document = json.loads(path.read_text(encoding="utf-8"))
    _parse(document)
    return document


def _write_document(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tokens.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def add_token(path: str | os.PathLike[str], *, name: str, sha256: str, scopes: Iterable[str]) -> None:
    target = Path(path)
    document = _read_document(target)
    entry = {"name": name, "sha256": sha256, "scopes": sorted(set(scopes))}
    _parse({"schema": SCHEMA, "tokens": [entry]})
    if any(existing["name"] == name for existing in document["tokens"]):
        raise TokenStoreError("token_name_exists")
    document["tokens"].append(entry)
    _write_document(target, document)


def revoke_token(path: str | os.PathLike[str], *, name: str) -> None:
    target = Path(path)
    document = _read_document(target)
    remaining = [entry for entry in document["tokens"] if entry["name"] != name]
    if len(remaining) == len(document["tokens"]):
        raise TokenStoreError("token_name_unknown")
    document["tokens"] = remaining
    _write_document(target, document)


def list_tokens(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    document = _read_document(Path(path))
    return [{"name": entry["name"], "scopes": entry["scopes"]} for entry in document["tokens"]]
