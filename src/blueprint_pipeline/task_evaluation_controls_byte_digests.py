"""Persisted digests of retained controls bytes, in a module that imports almost nothing.

A stored verdict is keyed on the code that computed it, and that includes every data
import of the computing module. Computed in ``task_evaluation_controls_autoprovision``,
a plain sha256 of these bytes depended on 1,236 modules, so most deploys discarded it and
hashed the multi-GB robot runtime twice, once as root and once as the service account:
291 s and 324 s against 9.4 s on the deploy receipts of 2026-10-30 and 31, in step with
whether one of those modules had changed. Only this module and the hashing it calls can
change these results.

Persisted byte-digest verdicts: the controls tick re-hashed the runtime payload and every
catalog asset on every tick (2026-09-13: about two of every six minutes per tick), although
nothing had moved. A stored digest is reused only while every consulted file keeps its exact
stat identity (large files) or re-hashes equal (small files); see validation_verdict_store.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_controls_context_contracts import _require

PAYLOAD_DIGEST_VERDICT = "controls_runtime_payload_digest"
ASSET_DIGEST_VERDICT = "controls_asset_sha256"


def _persisted_digest(name: str, key: Mapping[str, Any], compute: Callable[[], Any]) -> Any:
    """Reuse a byte-derived verdict while its code closure and every consulted byte are unchanged."""
    from .task_evaluation_release_identity import running_release_commit
    from .validation_file_digests import touched_files
    from .validation_verdict_store import executed_code_identity, lookup_entry, store
    found, stored = lookup_entry(name=name, key=key)
    if found:
        return stored
    with touched_files() as touched:
        verdict, code = executed_code_identity(compute, always=[__name__])
    try:
        release = running_release_commit()
    except (OSError, ValueError):
        release = ""
    store(name=name, key=key, files=touched, verdict=verdict, source_commit=release, code=code)
    return verdict


def _consulted_sha256(path: Path) -> str:
    """Hash through the evidence reader so the byte is recorded as consulted by the verdict."""
    from .validation_file_digests import _identity, record_touched, sha256_file
    try:
        return sha256_file(path)
    except ValueError:
        # The reader refuses symlinked parents; keep the historical hash and still
        # record the identity so the verdict re-proves this byte before reuse.
        with path.open("rb") as stream:
            digest = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
        record_touched(path, _identity(path.lstat()), digest)
        return digest


def payload_digest(root: Path) -> str:
    _require(root.is_dir() and not root.is_symlink(), "runtime_missing")
    listing = []
    for path in sorted(root.rglob("*")):
        _require(not path.is_symlink(), "runtime_symlink")
        if path.is_file():
            listing.append(path.relative_to(root).as_posix())
    _require(bool(listing), "runtime_empty")

    def compute() -> str:
        return canonical_digest({relative: _consulted_sha256(root / relative) for relative in listing})

    # The listing is part of the key: an added or removed file is a different payload.
    return str(_persisted_digest(PAYLOAD_DIGEST_VERDICT, {"root": str(root), "files": listing}, compute))


def asset_digest(path: Path) -> str:
    return str(_persisted_digest(ASSET_DIGEST_VERDICT, {"path": str(path)}, lambda: _consulted_sha256(path)))
