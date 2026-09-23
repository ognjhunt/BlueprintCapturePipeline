"""Bearer tokens: hashes on disk, constant-time checks, explicit scopes."""

# Covers (for impacted-test selection):
#   deploy/operator-door/operator_door/auth.py

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "operator-door"))

from operator_door.auth import (  # noqa: E402
    TokenIdentity,
    TokenStore,
    TokenStoreError,
    add_token,
    hash_token,
    list_tokens,
    revoke_token,
)

GOOD = "a" * 43
OTHER = "b" * 43


def _store(tmp_path: Path, tokens: list[dict], mode: int = 0o640) -> Path:
    path = tmp_path / "tokens.json"
    path.write_text(
        json.dumps({"schema": "blueprint_operator_door_tokens.v1", "tokens": tokens}),
        encoding="utf-8",
    )
    os.chmod(path, mode)
    return path


def test_hash_is_prefixed_sha256_hex() -> None:
    digest = hash_token(GOOD)
    assert digest.startswith("sha256:") and len(digest) == len("sha256:") + 64


def test_verifies_the_matching_bearer_and_returns_scopes(tmp_path: Path) -> None:
    path = _store(tmp_path, [{"name": "cloud", "sha256": hash_token(GOOD), "scopes": ["read", "deploy"]}])
    identity = TokenStore(path).verify(f"Bearer {GOOD}")
    assert identity == TokenIdentity(name="cloud", scopes=frozenset({"read", "deploy"}))


@pytest.mark.parametrize(
    "header",
    [None, "", "Bearer", f"Basic {GOOD}", f"Bearer {OTHER}", "Bearer short", f"bearer  {GOOD} extra"],
)
def test_rejects_missing_malformed_or_unknown_bearers(tmp_path: Path, header: str | None) -> None:
    path = _store(tmp_path, [{"name": "cloud", "sha256": hash_token(GOOD), "scopes": ["read"]}])
    assert TokenStore(path).verify(header) is None


def test_scheme_is_case_insensitive(tmp_path: Path) -> None:
    path = _store(tmp_path, [{"name": "cloud", "sha256": hash_token(GOOD), "scopes": ["read"]}])
    assert TokenStore(path).verify(f"bearer {GOOD}") is not None


def test_missing_store_rejects_everything(tmp_path: Path) -> None:
    assert TokenStore(tmp_path / "absent.json").verify(f"Bearer {GOOD}") is None


def test_unknown_scope_in_file_is_refused(tmp_path: Path) -> None:
    path = _store(tmp_path, [{"name": "cloud", "sha256": hash_token(GOOD), "scopes": ["root"]}])
    with pytest.raises(TokenStoreError, match="token_scope_unknown:root"):
        TokenStore(path).verify(f"Bearer {GOOD}")


def test_world_readable_store_is_refused(tmp_path: Path) -> None:
    path = _store(tmp_path, [{"name": "cloud", "sha256": hash_token(GOOD), "scopes": ["read"]}], 0o644)
    with pytest.raises(TokenStoreError, match="token_file_mode_too_open"):
        TokenStore(path).verify(f"Bearer {GOOD}")


def test_store_reloads_when_the_file_changes(tmp_path: Path) -> None:
    path = _store(tmp_path, [{"name": "cloud", "sha256": hash_token(GOOD), "scopes": ["read"]}])
    store = TokenStore(path)
    assert store.verify(f"Bearer {GOOD}") is not None
    _store(tmp_path, [{"name": "other", "sha256": hash_token(OTHER), "scopes": ["read"]}])
    os.utime(path, (1, 2_000_000_000))
    assert store.verify(f"Bearer {GOOD}") is None
    assert store.verify(f"Bearer {OTHER}") is not None


def test_add_token_writes_hash_only_with_tight_mode(tmp_path: Path) -> None:
    path = tmp_path / "tokens.json"
    add_token(path, name="cloud", sha256=hash_token(GOOD), scopes=["read", "operate"])
    text = path.read_text(encoding="utf-8")
    assert GOOD not in text
    assert oct(path.stat().st_mode & 0o777) == oct(0o640)
    assert TokenStore(path).verify(f"Bearer {GOOD}").scopes == frozenset({"read", "operate"})


def test_add_token_refuses_duplicates_bad_hashes_and_bad_names(tmp_path: Path) -> None:
    path = tmp_path / "tokens.json"
    add_token(path, name="cloud", sha256=hash_token(GOOD), scopes=["read"])
    with pytest.raises(TokenStoreError, match="token_name_exists"):
        add_token(path, name="cloud", sha256=hash_token(OTHER), scopes=["read"])
    with pytest.raises(TokenStoreError, match="token_hash_invalid"):
        add_token(path, name="x", sha256="sha256:nothex", scopes=["read"])
    with pytest.raises(TokenStoreError, match="token_name_invalid"):
        add_token(path, name="bad name", sha256=hash_token(OTHER), scopes=["read"])


def test_list_tokens_shows_names_and_scopes_but_no_hashes(tmp_path: Path) -> None:
    path = tmp_path / "tokens.json"
    add_token(path, name="cloud", sha256=hash_token(GOOD), scopes=["deploy", "read"])
    assert list_tokens(path) == [{"name": "cloud", "scopes": ["deploy", "read"]}]


def test_revoke_token_removes_it(tmp_path: Path) -> None:
    path = tmp_path / "tokens.json"
    add_token(path, name="cloud", sha256=hash_token(GOOD), scopes=["read"])
    revoke_token(path, name="cloud")
    assert TokenStore(path).verify(f"Bearer {GOOD}") is None
    with pytest.raises(TokenStoreError, match="token_name_unknown"):
        revoke_token(path, name="cloud")


def test_token_file_written_as_root_takes_the_directory_group(tmp_path: Path, monkeypatch) -> None:
    import operator_door.auth as auth

    calls: list[tuple[str, int, int]] = []
    monkeypatch.setattr(auth.os, "geteuid", lambda: 0)
    monkeypatch.setattr(auth.os, "chown", lambda path, uid, gid: calls.append((str(path), uid, gid)))
    path = tmp_path / "tokens.json"
    add_token(path, name="cloud", sha256=hash_token(GOOD), scopes=["read"])
    assert calls and calls[-1][1:] == (0, tmp_path.stat().st_gid)
