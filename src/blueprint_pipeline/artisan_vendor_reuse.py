"""Bind sealed vendor bytes before admitting a release's temporary disk use.

Each plan row records the stat identity its vendor source file held while hashing
equal to the published digest. Hashing those 10 GB took two of a deploy's passes,
one at admission and one in the build, on files nothing had touched. A source is read
again only when its identity moved; the change time in it is kernel-owned, so a write
always moves it. A missing or stale plan only means the bytes are read again.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import time

from .decision_evidence_contracts import canonical_digest
from .immutable_file_digest import _STABLE_AGE_NS, immutable_file_digest_cache, sha256_file
from .readback_ledger import admit_ledger

SITE_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pth", "__editable__*", "artisan_assets-*.dist-info")
PACKAGE_PREFIX = "components/content_agents_rigid_replacement/package/"


def _files(root, *, ignore=None):
    root = Path(root).resolve()

    def walk(directory, ancestors):
        resolved = directory.resolve()
        if not resolved.is_relative_to(root) or resolved in ancestors:
            raise ValueError("artisan_vendor_reuse_source_link_invalid")
        entries = sorted(directory.iterdir())
        excluded = ignore(str(directory), [p.name for p in entries]) if ignore else set()
        for path in entries:
            if path.name in excluded:
                continue
            if path.is_dir():
                yield from walk(path, ancestors | {resolved})
            elif path.is_file() and (not path.is_symlink() or path.resolve().is_relative_to(root)):
                yield path  # its directory is already known to resolve inside the root
            else:
                raise ValueError("artisan_vendor_reuse_source_file_invalid")

    yield from walk(root, set())


def _identity(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_uid, value.st_gid, value.st_mtime_ns, value.st_ctime_ns)


def _links_above(path, known):
    """Whether a directory above ``path`` is a link, asking the filesystem once per directory."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent not in known:
        known[parent] = os.path.realpath(parent) != parent
    return known[parent]


def _recorded_identity(identity):
    """An identity may stand in for the bytes only once a same-tick write can no longer hide."""
    return list(identity) if time.time_ns() - identity[7] >= _STABLE_AGE_NS else None


def _remembered_sources(directory, sources):
    """The rows of the newest plan this deployer wrote for these vendor roots, by source file."""
    if directory is None:
        return {}
    try:
        plans = sorted(Path(directory).glob("*.json"), key=lambda path: path.lstat().st_mtime_ns, reverse=True)
        for path in plans[:3]:
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o222:
                continue
            plan = json.loads(path.read_bytes())
            if (plan.get("schema_version") == "artisan_vendor_reuse.v1" and plan.get("sources") == sources
                    and plan.get("digest") == canonical_digest(plan, digest_field="digest")):
                return {(row["kind"], row["relative_path"]): row for row in plan["files"]}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return {}


@immutable_file_digest_cache()
def prepare_vendor_reuse(*, toolchain_root, site_packages_root, blender_root, staging_device,
                         previous_plans=None):
    """Return a full-byte plan, or None and retain the ordinary full-copy budget.

    Published inodes the toolchain's own readback ledger proved are not digested
    again, nor is a vendor source file holding the identity it hashed equal under
    in the newest plan in ``previous_plans``.
    """
    root = Path(toolchain_root)
    if root.is_symlink() or not root.is_dir():
        return None
    try:
        manifest_name = "task_evaluation_scene_configuration_toolchain.v1.json"
        manifest = json.loads((root / manifest_name).read_text())
        if manifest.get("toolchain_digest") != canonical_digest(manifest, digest_field="toolchain_digest"):
            return None
        admit_ledger(root, manifest_name=manifest_name, digest_field="toolchain_digest",
                     readback_actor=manifest.get("readback_actor"))
        declared = {row["relative_path"]: row for row in manifest["files"]}
        sources = {"site_packages": str(Path(site_packages_root).resolve()),
                   "blender": str(Path(blender_root).resolve())}
        remembered = _remembered_sources(previous_plans, sources)
        directories = {}
        rows = []
        hashed = {"file_count": 0, "bytes": 0}
        for kind, source in sources.items():
            for path in _files(source, ignore=SITE_IGNORE if kind == "site_packages" else None):
                relative = os.fspath(path)[len(source) + 1:]
                published = PACKAGE_PREFIX + f"runtime/{kind}/" + relative
                prior = root / published
                row = declared[published]
                before, source_before = _identity(prior), _identity(path)
                earlier = remembered.get((kind, relative), {})
                source_read = not (earlier.get("source_identity") == list(source_before)
                                   and earlier.get("sha256") == row["sha256"])
                mode = stat.S_IMODE(prior.lstat().st_mode)
                if (not stat.S_ISREG(prior.lstat().st_mode) or mode & 0o222
                        or _links_above(prior, directories)
                        or before[0] != staging_device or row["size_bytes"] != source_before[3]
                        or before[3] != source_before[3]
                        or mode != (0o555 if source_before[2] & 0o111 else 0o444)
                        or (source_read and sha256_file(path) != row["sha256"])
                        or sha256_file(prior) != row["sha256"]
                        or before != _identity(prior) or source_before != _identity(path)):
                    return None
                if source_read:
                    hashed["file_count"] += 1
                    hashed["bytes"] += row["size_bytes"]
                rows.append({"kind": kind, "relative_path": relative, "published_path": published,
                             "size_bytes": row["size_bytes"], "sha256": row["sha256"],
                             "source_identity": _recorded_identity(source_before)})
        if not rows:
            return None
        value = {"schema_version": "artisan_vendor_reuse.v1", "toolchain_root": str(root.absolute()),
                 "staging_root": str(root.parent.absolute()),
                 "toolchain_digest": manifest["toolchain_digest"], "sources": sources, "files": rows,
                 "shared_bytes": sum(row["size_bytes"] for row in rows),
                 "source_files_hashed": hashed["file_count"], "source_bytes_hashed": hashed["bytes"]}
        value["digest"] = canonical_digest(value, digest_field="digest")
        return value
    except (OSError, ValueError, KeyError, TypeError):
        return None


def vendor_copy_function(plan, *, kind, source_root):
    """A budgeted reuse never falls back to an unreserved SDK copy."""
    if (plan.get("schema_version") != "artisan_vendor_reuse.v1"
            or plan.get("digest") != canonical_digest(plan, digest_field="digest")
            or str(Path(source_root).resolve()) != plan["sources"][kind]):
        raise ValueError("artisan_vendor_reuse_plan_invalid")
    rows = {row["relative_path"]: row for row in plan["files"] if row["kind"] == kind}
    root = Path(plan["toolchain_root"])
    copied = set()
    directories = {}

    def copy(source, destination):
        original, target = Path(source), Path(destination)
        try:
            row = rows[os.path.relpath(original, source_root)]
            prior = root / row["published_path"]
            before, source_before = _identity(prior), _identity(original)
            if (prior.is_symlink() or _links_above(prior, directories)
                    or before[2] & 0o222 or before[3] != row["size_bytes"]
                    or stat.S_IMODE(before[2]) != (0o555 if source_before[2] & 0o111 else 0o444)
                    or before[0] != target.parent.stat().st_dev
                    or (row.get("source_identity") != list(source_before)
                        and sha256_file(original) != row["sha256"])
                    or sha256_file(prior) != row["sha256"]
                    or before != _identity(prior) or source_before != _identity(original)):
                raise ValueError("artisan_vendor_reuse_bytes_changed")
            os.link(prior, target, follow_symlinks=False)
            if target.stat().st_ino != before[1] or sha256_file(target) != row["sha256"]:
                raise ValueError("artisan_vendor_reuse_link_changed")
            copied.add(row["relative_path"])
            return str(target)
        except (OSError, KeyError) as exc:
            raise ValueError("artisan_vendor_reuse_unavailable") from exc

    def require_complete():
        if copied != set(rows):
            raise ValueError("artisan_vendor_reuse_source_set_changed")

    copy.require_complete = require_complete
    return copy
