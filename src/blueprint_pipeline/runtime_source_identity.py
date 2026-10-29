"""Read an installed module's admitted source identity without Git or network.

The root installer joins signed source inventory to exact snapshot rows. Only
its protected v2 records prove this stronger identity; Git HEAD metadata and
legacy snapshots remain unproven here. This grants no execution authority.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

_RUNTIME_ROOT = Path("/mnt/blueprint-work/scene-retirement-runtime")
_BOOT_ROOT = Path("/usr/lib/blueprint/scene-retirement-runtime")
_OWNER = 0
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_MAX_RECORD = 16 * 1024**2
_MAX_FILES = 32768
_MAX_BYTES = 4 * 1024**3


def _require(value):
    if not value:
        raise ValueError("runtime_source_identity_unproven")


def _version(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _protected(info, directory):
    _require(info.st_uid in {0, _OWNER} and not info.st_mode & 0o022
        and (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)))
    if not directory:
        _require(info.st_nlink == 1 and stat.S_IMODE(info.st_mode) in {0o444, 0o555, 0o644, 0o755})


def _read(path, maximum):
    """Pin every no-follow ancestor and leaf through one bounded read."""
    _require(path.is_absolute() and ".." not in path.parts)
    opened = []
    try:
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        opened.append((fd, _version(os.fstat(fd))))
        _protected(os.fstat(fd), True)
        for index, name in enumerate(path.parts[1:]):
            directory = index < len(path.parts) - 2
            before = os.stat(name, dir_fd=fd, follow_symlinks=False)
            _protected(before, directory)
            child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
                | (os.O_DIRECTORY if directory else 0), dir_fd=fd)
            opened.append((child, _version(os.fstat(child))))
            _require(_version(os.fstat(child)) == _version(before))
            fd = child
        before = os.fstat(fd)
        _require(0 <= before.st_size <= maximum)
        raw = bytearray()
        while len(raw) <= before.st_size:
            chunk = os.read(fd, min(65536, before.st_size + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        _require(len(raw) == before.st_size
            and _version(path.lstat()) == _version(before))
        return bytes(raw), _version(before)
    finally:
        changed = False
        for fd, version in reversed(opened):
            try:
                changed |= _version(os.fstat(fd)) != version
            finally:
                os.close(fd)
        _require(not changed)


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _unique(pairs):
    value = {}
    for key, item in pairs:
        _require(key not in value)
        value[key] = item
    return value


def _record(path, maximum):
    raw, version = _read(path, maximum)
    value = json.loads(raw, object_pairs_hook=_unique)
    _require(type(value) is dict and _encoded(value) == raw)
    return value, raw, version


def _source_rows(value):
    _require(type(value) is dict and 0 < len(value) <= _MAX_FILES)
    total = 0
    for name, row in value.items():
        _require(type(name) is str and "\\" not in name
            and all(part not in {"", ".", ".."} for part in name.split("/"))
            and any(name.startswith(prefix) for prefix in ("src/blueprint_pipeline/", "scripts/", "deploy/systemd/"))
            and type(row) is dict and set(row) == {"size", "sha256", "mode"}
            and type(row["size"]) is int and row["size"] >= 0
            and type(row["sha256"]) is str and _HEX.fullmatch(row["sha256"])
            and type(row["mode"]) is int and row["mode"] in {0o644, 0o755})
        total += row["size"]
    _require(total <= _MAX_BYTES)
    return value


def _admitted_rows(identity):
    """Reconstruct the exact runtime subset of the existing signed tree claim."""
    items = identity['signed_entries']
    _require(type(items) is list and 0 < len(items) <= _MAX_FILES
        and hashlib.sha256(_encoded(items)).hexdigest() == identity['signed_tree_sha256'])
    rows, names, seen, total = {}, [], set(), 0
    for item in items:
        _require(type(item) is list and len(item) == 5)
        mode, oid, name, size, sha256 = item
        _require(mode in {'100644', '100755'} and type(oid) is str and re.fullmatch(r'[0-9a-f]{40}', oid)
            and type(name) is str and name not in seen and '\\' not in name
            and all(part not in {'', '.', '..'} for part in name.split('/'))
            and type(size) is int and size >= 0 and type(sha256) is str and _HEX.fullmatch(sha256))
        names.append(name)
        seen.add(name)
        total += size
        if any(name.startswith(prefix) for prefix in ('src/blueprint_pipeline/', 'scripts/', 'deploy/systemd/')):
            rows[name] = {'size': size, 'sha256': sha256, 'mode': 0o644}
        else:
            _require(name in {'uv.lock', 'pyproject.toml'})
    _require(total <= _MAX_BYTES and names == sorted(names)
        and {'uv.lock', 'src/blueprint_pipeline/__init__.py', 'scripts/scene_retirement_continuous_bootstrap.py'} <= set(names))
    return rows


def running_source_identity(module_path: str | Path | None = None) -> dict:
    """Prove the owning protected snapshot, not whichever cohort is now selected."""
    unknown = {"schema_version": "runtime_source_identity.v1", "state": "unproven",
        "reason": "runtime_source_identity_unproven", "source_commit": None}
    try:
        module = Path(module_path or __file__).absolute()
        relative = module.relative_to(_RUNTIME_ROOT)
        generation = None
        if relative.parts[0] == "generations":
            generation = relative.parts[1]
            _require(_HEX.fullmatch(generation))
            root = _RUNTIME_ROOT / "generations" / generation
            record = _RUNTIME_ROOT / "manifests" / ("source-" + generation + ".json")
            plan, raw, version = _record(record, _MAX_RECORD)
            _require(set(plan) == {"schema", "kind", "generation", "rows", "source_identity"}
                and plan["schema"] == "scene-retirement-runtime-generation.v2"
                and plan["kind"] == "source" and plan["generation"] == generation)
            rows = _source_rows(plan["rows"])
        else:
            root, record = _RUNTIME_ROOT, _BOOT_ROOT / "installation.json"
            plan, raw, version = _record(record, _MAX_RECORD)
            _require(set(plan) == {"schema", "runtime", "rows", "bootstrap", "source_identity"}
                and plan["schema"] == "scene-retirement-runtime-install.v2"
                and plan["runtime"] == str(root) and type(plan["rows"]) is dict
                and len(plan["rows"]) <= _MAX_FILES)
            rows = _source_rows({name: row for name, row in plan["rows"].items() if not name.startswith("dependencies/")})
        identity = plan["source_identity"]
        _require(type(identity) is dict and set(identity) == {"schema", "commit", "signed_tree_sha256", "signed_claim_sha256", "signed_entries"}
            and identity["schema"] == "scene-retirement-runtime-source-identity.v1"
            and type(identity["commit"]) is str and re.fullmatch(r"[0-9a-f]{40}", identity["commit"])
            and all(type(identity[key]) is str and _HEX.fullmatch(identity[key])
                    for key in ("signed_tree_sha256", "signed_claim_sha256")))
        _require(rows == _admitted_rows(identity))
        source_digest = hashlib.sha256(_encoded({"rows": rows, "source_identity": identity})).hexdigest()
        _require(generation is None or generation == source_digest)
        claim_path = _RUNTIME_ROOT.with_name(_RUNTIME_ROOT.name + "-sdk-inputs") / "release-inputs" / (identity["commit"] + ".claim.json")
        claim, claim_raw, claim_version = _record(claim_path, 4096)
        _require(claim == {"schema": "scene-retirement-signed-release.v1", "commit": identity["commit"],
            "tree_sha256": identity["signed_tree_sha256"]}
            and hashlib.sha256(claim_raw).hexdigest() == identity["signed_claim_sha256"])
        row = rows[str(module.relative_to(root))]
        body, module_version = _read(module, 1024**2)
        _require(len(body) == row["size"] and hashlib.sha256(body).hexdigest() == row["sha256"]
            and (0o755 if module_version[2] & 0o111 else 0o644) == row["mode"])
        _require(_read(record, _MAX_RECORD) == (raw, version)
            and _read(claim_path, 4096) == (claim_raw, claim_version)
            and _read(module, 1024**2) == (body, module_version))
        return {"schema_version": "runtime_source_identity.v1", "state": "proven",
            "source_commit": identity["commit"], "source_digest": source_digest,
            "basis": "protected_admitted_source_snapshot"}
    except (OSError, ValueError, KeyError, TypeError, IndexError, UnicodeError, RecursionError):
        return unknown
