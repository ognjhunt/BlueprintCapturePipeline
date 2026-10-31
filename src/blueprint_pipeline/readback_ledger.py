"""Remember which published runtime inodes the service account already read in full.

Most of a scene-configuration toolchain is the same inodes release after release:
the ARTISAN vendor reuse plan hardlinks about 38,500 vendored files (10.1 GB)
from the previous toolchain. Each deploy still digested those bytes about ten
times (component copy, package, validation, toolchain inventory, service-account
readback, post-publish validation), because every hardlink and unlink moves the
inode's link count and ctime, which the per-publication digest cache keys on:
about 20 of the deploy's 21 provisioning minutes (2026-10-29 receipts).

The ledger written beside a publication records the sealed identity
(``immutable_file_digest.sealed_identity``) of every inode the service account
read in full under that tree, or opened there after an earlier ledger proved
it. The next publication admits those inodes as proven while each still sits at
its recorded path in a retained tree with exactly that identity: their digests
are not recomputed and the account opens them at the new path instead of
reading every byte again. The tree, its manifest and its digest are unchanged;
a missing, stale or edited ledger only means the bytes are read again.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .immutable_file_digest import admit_proven_digest, sealed_identity

SCHEMA_VERSION = "runtime_tree_readback_ledger.v1"
DIRECTORY = "readback-ledgers"
IDENTITY_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_mode", "st_uid", "st_gid")
_LEDGER_NAME = re.compile(r"([0-9a-f]{40})\.json")


def ledger_path(tree: str | Path) -> Path:
    """``<runtime root>/readback-ledgers/<component>/<tree>.json``.

    Outside the component roots, so release retirement's managed children are
    unchanged; a ledger whose tree is gone is pruned by the next write.
    """
    tree = Path(tree)
    return tree.parent.parent / DIRECTORY / tree.parent.name / f"{tree.name}.json"


def admit_ledger(
    tree: str | Path, *, manifest_name: str, digest_field: str, readback_actor: str,
) -> int:
    """Admit the inodes ``tree``'s ledger proved; return how many files were admitted."""
    tree = Path(tree)
    try:
        path = ledger_path(tree)
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o222:
            return 0
        ledger = json.loads(path.read_bytes())
        manifest = json.loads((tree / manifest_name).read_bytes())
        if (ledger.get("schema_version") != SCHEMA_VERSION
                or ledger.get("ledger_digest") != canonical_digest(ledger, digest_field="ledger_digest")
                or ledger.get("readback_actor") != readback_actor
                or manifest.get(digest_field) != canonical_digest(manifest, digest_field=digest_field)
                or ledger.get("tree_digest") != manifest[digest_field]):
            return 0
        inodes = ledger["inodes"]
        admitted = 0
        for row in manifest["files"]:
            recorded = inodes.get(row["relative_path"])
            if recorded is None:
                continue
            try:
                current = (tree / row["relative_path"]).lstat()
            except OSError:
                continue  # retired or replaced since; read it again
            if (list(sealed_identity(current)) == recorded and current.st_size == row["size_bytes"]
                    and admit_proven_digest(current, row["sha256"])):
                admitted += 1
        return admitted
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return 0


def write_ledger(
    tree: str | Path, *, tree_digest: str, readback_actor: str,
    inodes: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    """Record the inodes this publication's readback proved, for the next publication.

    ``inodes`` maps each relative path to the sealed identity the account read
    (or opened) it under; only those still installed at that path in the
    published ``tree`` are recorded. Best effort: a ledger that cannot be
    written leaves the next publication to read these bytes again, so a failure
    is reported, never raised.
    """
    tree = Path(tree)
    path = ledger_path(tree)
    try:
        value: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "tree_digest": tree_digest,
            "readback_actor": readback_actor,
            "identity_fields": list(IDENTITY_FIELDS),
            "inodes": {relative: list(identity) for relative, identity in sorted(inodes.items())
                       if sealed_identity((tree / relative).lstat()) == tuple(identity)},
            "ledger_digest": "",
        }
        value["ledger_digest"] = canonical_digest(value, digest_field="ledger_digest")
        path.parent.mkdir(parents=True, exist_ok=True)
        for name in os.listdir(path.parent):
            match = _LEDGER_NAME.fullmatch(name)
            if match is not None and not (tree.parent / match.group(1)).is_dir():
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(path.parent / name)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                os.fchmod(stream.fileno(), 0o444)
                stream.write((canonical_json(value) + "\n").encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise
    except OSError as exc:
        return {"status": f"not_written:{type(exc).__name__}", "path": str(path)}
    return {"status": "written", "path": str(path), "ledger_digest": value["ledger_digest"],
            "file_count": len(value["inodes"])}
