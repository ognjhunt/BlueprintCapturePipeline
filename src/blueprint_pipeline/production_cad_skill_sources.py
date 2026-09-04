"""Provision and verify the pinned CAD skill sources used by production runs."""

from __future__ import annotations

import hashlib
import json
import os
import pwd
import shutil
import stat
import subprocess  # nosec B404 - fixed git argv and pinned public repositories
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "production_cad_skill_sources.v1"
DEFAULT_ROOT = (
    "/var/lib/blueprint/task-evaluation-inputs/sources/cad-authoring"
)
DEFAULT_SERVICE_ACCOUNT = "blueprint"
SOURCE_SPECS = (
    {
        "id": "text-to-cad",
        "repository": "https://github.com/earthtojake/text-to-cad",
        "commit": "4fd71ea75fbb8a80b0d7c76862e0fd73c52a8989",
        "tree": "a5bf8c8eb14561557f44064c59c254d031665c65",
        "license": "MIT",
        "license_sha256": (
            "sha256:c122f8e06704acaab9f2715b9940cdfdf1b35cf575088b4890970c6259027361"
        ),
        "skills": (
            "cad",
            "cad-viewer",
            "urdf",
            "sdf",
            "srdf",
            "step-parts",
            "implicit-cad",
            "dxf",
            "gcode",
        ),
    },
    {
        "id": "multi-agent-cad",
        "repository": "https://github.com/Pan-Chera/Multi-Agent-CAD",
        "commit": "42737c408534e7c00c63081d73ce7565a9464e56",
        "tree": "ff698ed1bbe4566df694b365c4ce15db721ac888",
        "license": "MIT",
        "license_sha256": (
            "sha256:b0d5bf2263928b3c6e536e1791e1820b991ee4c6c74568e5710026c62b8c8c29"
        ),
        "skills": ("multi-agent-cad",),
    },
)


class ProductionCadSkillSourcesError(RuntimeError):
    """The production CAD skill source closure was absent or identity-drifted."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(  # nosec B603 - fixed git executable and argv
        ["git", "-c", f"safe.directory={root}", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise ProductionCadSkillSourcesError(
            "production_cad_skill_source_git_failed"
        )
    return completed.stdout.strip()


def _source_record(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    license_path = root / "LICENSE"
    if (
        root.is_symlink()
        or not root.is_dir()
        or _git(root, "rev-parse", "HEAD") != spec["commit"]
        or _git(root, "rev-parse", "HEAD^{tree}") != spec["tree"]
        or _git(root, "status", "--porcelain=v1")
        or not license_path.is_file()
        or _sha256(license_path) != spec["license_sha256"]
    ):
        raise ProductionCadSkillSourcesError(
            f"production_cad_skill_source_invalid:{spec['id']}"
        )
    if spec["id"] == "text-to-cad":
        for skill in spec["skills"]:
            if not (root / "skills" / str(skill) / "SKILL.md").is_file():
                raise ProductionCadSkillSourcesError(
                    f"production_cad_skill_missing:{skill}"
                )
    else:
        for relative in (
            "multi_agent_cad/WORKFLOW.md",
            "multi_agent_cad/graph.py",
            "environment.yml",
        ):
            if not (root / relative).is_file():
                raise ProductionCadSkillSourcesError(
                    f"production_cad_skill_source_incomplete:{relative}"
                )
    return {
        "id": spec["id"],
        "repository": spec["repository"],
        "commit": spec["commit"],
        "tree": spec["tree"],
        "license": spec["license"],
        "license_sha256": spec["license_sha256"],
        "skills": list(spec["skills"]),
        "path": str(root),
    }


def validate_production_cad_skill_sources(
    root: str | Path,
    *,
    specs: Sequence[Mapping[str, Any]] = SOURCE_SPECS,
    service_account: str | None = None,
) -> dict[str, Any]:
    source_root = Path(root).expanduser().resolve()
    rows = [
        _source_record(
            source_root / f"{spec['id']}-{str(spec['commit'])[:8]}", spec
        )
        for spec in specs
    ]
    if service_account is not None and _service_account_ids(service_account) is not None:
        # The control-plane CAD authoring stage runs as the service account;
        # a root-only checkout passes every byte check and still blocks it.
        for spec, row in zip(specs, rows, strict=True):
            if service_account_tree_read_blocker(Path(row["path"]), service_account):
                raise ProductionCadSkillSourcesError(
                    "production_cad_skill_source_unreadable_by_service_account:"
                    f"{spec['id']}"
                )
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "source_count": len(rows),
        "skill_count": sum(len(row["skills"]) for row in rows),
        "sources": rows,
        "raw_secret_values_recorded": False,
        "provider_mutation_performed": False,
        "receipt_digest": "",
    }
    value["receipt_digest"] = canonical_digest(
        value, digest_field="receipt_digest"
    )
    return value


def _read_only_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        mode = stat.S_IMODE(path.stat().st_mode)
        path.chmod(mode & ~0o222)
    root.chmod(stat.S_IMODE(root.stat().st_mode) & ~0o222)


def _service_account_ids(account: str) -> tuple[int, int] | None:
    try:
        record = pwd.getpwnam(account)
    except KeyError:
        return None
    return record.pw_uid, record.pw_gid


def _account_can_read(metadata: os.stat_result, *, uid: int, gid: int) -> bool:
    """Read (and traverse, for directories) without root's bypass of mode bits."""

    want = stat.S_IRUSR | (stat.S_IXUSR if stat.S_ISDIR(metadata.st_mode) else 0)
    if metadata.st_uid == uid:
        return metadata.st_mode & want == want
    if metadata.st_gid == gid:
        want_group = want >> 3
        return metadata.st_mode & want_group == want_group
    want_other = want >> 6
    return metadata.st_mode & want_other == want_other


def _tree_paths(root: Path) -> list[Path]:
    return [root, *sorted(root.rglob("*"))]


def service_account_tree_read_blocker(root: Path, account: str) -> str | None:
    """Return the first path the service account could not read, or ``None``."""

    ids = _service_account_ids(account)
    if ids is None:
        return None
    uid, gid = ids
    for path in _tree_paths(root):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            continue
        if not _account_can_read(metadata, uid=uid, gid=gid):
            return str(path)
    return None


def grant_service_account_tree_read(root: Path, account: str) -> dict[str, Any]:
    """Make a read-only source tree readable by the service account's group.

    Ownership stays with the provisioning root user so the service account
    cannot rewrite pinned sources; the group gains read (and traverse) only.
    """

    ids = _service_account_ids(account)
    if ids is None:
        return {"account": account, "status": "not_applicable_no_service_account"}
    _uid, gid = ids
    for path in _tree_paths(root):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            continue
        if metadata.st_gid != gid:
            os.chown(path, -1, gid, follow_symlinks=False)
        mode = stat.S_IMODE(metadata.st_mode) & ~0o222
        mode |= stat.S_IRGRP | (stat.S_IXGRP if stat.S_ISDIR(metadata.st_mode) else 0)
        path.chmod(mode)
    blocker = service_account_tree_read_blocker(root, account)
    if blocker is not None:
        raise ProductionCadSkillSourcesError(
            f"production_cad_skill_source_unreadable_by_service_account:{blocker}"
        )
    return {"account": account, "status": "readable"}


def provision_production_cad_skill_sources(
    root: str | Path = DEFAULT_ROOT,
    *,
    specs: Sequence[Mapping[str, Any]] = SOURCE_SPECS,
    service_account: str | None = DEFAULT_SERVICE_ACCOUNT,
) -> dict[str, Any]:
    """Clone missing pinned sources atomically; never replace existing bytes.

    Every provisioned tree is left read-only and readable by the service
    account, which is the identity the control-plane CAD stage runs under.
    """

    destination = Path(root).expanduser().absolute()
    if destination.is_symlink():
        raise ProductionCadSkillSourcesError(
            "production_cad_skill_root_invalid"
        )
    destination.mkdir(parents=True, exist_ok=True, mode=0o755)
    for spec in specs:
        target = destination / f"{spec['id']}-{str(spec['commit'])[:8]}"
        if target.exists():
            _source_record(target, spec)
            if service_account is not None:
                grant_service_account_tree_read(target, service_account)
            continue
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{spec['id']}-", dir=destination
            )
        )
        try:
            completed = subprocess.run(  # nosec B603 - pinned public git source
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    str(spec["repository"]),
                    str(staging),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if completed.returncode != 0:
                raise ProductionCadSkillSourcesError(
                    f"production_cad_skill_clone_failed:{spec['id']}"
                )
            _git(staging, "checkout", "--detach", str(spec["commit"]))
            _source_record(staging, spec)
            try:
                os.rename(staging, target)
            except FileExistsError:
                _source_record(target, spec)
            else:
                _read_only_tree(target)
                if service_account is not None:
                    grant_service_account_tree_read(target, service_account)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
    result = validate_production_cad_skill_sources(
        destination, specs=specs, service_account=service_account
    )
    receipt_path = destination / f"{SCHEMA_VERSION}.json"
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    if receipt_path.exists():
        if receipt_path.is_symlink() or receipt_path.read_text() != payload:
            raise ProductionCadSkillSourcesError(
                "production_cad_skill_receipt_conflict"
            )
    else:
        receipt_path.write_text(payload, encoding="utf-8")
        receipt_path.chmod(0o444)
    access = (
        {"account": service_account, "status": "not_applicable_no_service_account"}
        if service_account is None or _service_account_ids(service_account) is None
        else {"account": service_account, "status": "readable"}
    )
    return {**result, "service_account_access": access}


__all__ = [
    "DEFAULT_ROOT",
    "DEFAULT_SERVICE_ACCOUNT",
    "grant_service_account_tree_read",
    "service_account_tree_read_blocker",
    "ProductionCadSkillSourcesError",
    "SCHEMA_VERSION",
    "SOURCE_SPECS",
    "provision_production_cad_skill_sources",
    "validate_production_cad_skill_sources",
]
