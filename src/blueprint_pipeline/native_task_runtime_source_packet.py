"""Package exact released Isaac Lab and Arena sources for paid native tasks.

The qualified Isaac Sim image is intentionally treated as a simulator base, not
as evidence that companion Python packages are installed.  This module exports
the small, task-neutral source closure used by native Panda construction from
exact git objects.  Reading blobs with ``git show`` prevents dirty working-copy
bytes from entering a paid bundle while retaining repository, revision, tree,
license, file, and archive identities.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from datetime import datetime, timezone
from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "native_task_runtime_source_packet.v1"
MANIFEST_SCHEMA_VERSION = "native_task_runtime_source_manifest.v1"
ISAACLAB_REPOSITORY = "https://github.com/isaac-sim/IsaacLab.git"
ISAACLAB_COMMIT = "e57379c634b42db5a0fe9f754341be6e2a7c7c43"
ISAACLAB_TREE = "454115265327a80acabd07cbd36e10071fc0c065"
ARENA_REPOSITORY = "https://github.com/isaac-sim/IsaacLab-Arena.git"
ARENA_COMMIT = "8b4a3a47fc53de23e8205089d71109a2e2348acd"
ARENA_TREE = "03f31f3dd56c56d00f24dbfb09711ec0ab345de8"
ISAACLAB_PACKAGE_NAMES = (
    "isaaclab",
    "isaaclab_physx",
    "isaaclab_assets",
    "isaaclab_tasks",
    "isaaclab_teleop",
)
INSTALL_ROOTS = tuple(
    f"runtime_sources/isaaclab/source/{name}" for name in ISAACLAB_PACKAGE_NAMES
) + ("runtime_sources/arena",)
RUNTIME_DEPENDENCY_WHEELS = (
    {
        "filename": "gymnasium-1.2.1-py3-none-any.whl",
        "package": "gymnasium",
        "version": "1.2.1",
        "license_spdx": "MIT",
    },
    {
        "filename": "lazy_loader-0.4-py3-none-any.whl",
        "package": "lazy_loader",
        "version": "0.4",
        "license_spdx": "BSD-3-Clause",
    },
    {
        "filename": "cloudpickle-3.1.1-py3-none-any.whl",
        "package": "cloudpickle",
        "version": "3.1.1",
        "license_spdx": "BSD-3-Clause",
    },
    {
        "filename": "Farama_Notifications-0.0.4-py3-none-any.whl",
        "package": "farama-notifications",
        "version": "0.0.4",
        "license_spdx": "MIT",
    },
    {
        "filename": "packaging-25.0-py3-none-any.whl",
        "package": "packaging",
        "version": "25.0",
        "license_spdx": "Apache-2.0 OR BSD-2-Clause",
    },
    {
        "filename": "typing_extensions-4.15.0-py3-none-any.whl",
        "package": "typing-extensions",
        "version": "4.15.0",
        "license_spdx": "PSF-2.0",
    },
)


class NativeTaskRuntimeSourcePacketError(ValueError):
    """Stable source identity or packet validation failure."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=not binary,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise NativeTaskRuntimeSourcePacketError(
            ["native_task_runtime_source_git_read_failed"]
        ) from exc
    return completed.stdout if binary else completed.stdout.strip()


def _tracked_paths(repo: Path, commit: str, prefixes: Sequence[str]) -> list[str]:
    output = _git(repo, "ls-tree", "-r", "--name-only", commit, "--", *prefixes)
    assert isinstance(output, str)
    paths = sorted(line.strip() for line in output.splitlines() if line.strip())
    if not paths:
        raise NativeTaskRuntimeSourcePacketError(
            ["native_task_runtime_source_required_paths_missing"]
        )
    return paths


def _tracked_blobs(
    repo: Path, commit: str, prefixes: Sequence[str]
) -> dict[str, bytes]:
    """Read the selected commit in one git process, never from the worktree."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "archive", "--format=tar", commit, "--", *prefixes],
            check=True,
            capture_output=True,
        )
        blobs: dict[str, bytes] = {}
        with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:") as archive:
            for member in archive.getmembers():
                if member.isdir():
                    continue
                if not member.isfile():
                    raise NativeTaskRuntimeSourcePacketError(
                        ["native_task_runtime_source_nonregular_git_object"]
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    raise NativeTaskRuntimeSourcePacketError(
                        ["native_task_runtime_source_git_archive_invalid"]
                    )
                blobs[member.name] = stream.read()
        return blobs
    except (OSError, subprocess.CalledProcessError, tarfile.TarError) as exc:
        raise NativeTaskRuntimeSourcePacketError(
            ["native_task_runtime_source_git_archive_failed"]
        ) from exc


def _repository_rows(
    *,
    repo: Path,
    repository: str,
    commit: str,
    expected_tree: str,
    license_id: str,
    license_path: str,
    archive_namespace: str,
    prefixes: Sequence[str],
) -> tuple[dict[str, Any], list[tuple[str, bytes]]]:
    if not repo.is_dir():
        raise NativeTaskRuntimeSourcePacketError(
            [f"native_task_runtime_source_repository_missing:{archive_namespace}"]
        )
    observed_commit = _git(repo, "rev-parse", "HEAD")
    observed_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    if observed_commit != commit:
        raise NativeTaskRuntimeSourcePacketError(
            [f"native_task_runtime_source_commit_mismatch:{archive_namespace}"]
        )
    if observed_tree != expected_tree:
        raise NativeTaskRuntimeSourcePacketError(
            [f"native_task_runtime_source_tree_mismatch:{archive_namespace}"]
        )
    paths = _tracked_paths(repo, commit, prefixes)
    if license_path not in paths:
        raise NativeTaskRuntimeSourcePacketError(
            [f"native_task_runtime_source_license_missing:{archive_namespace}"]
        )
    tracked_blobs = _tracked_blobs(repo, commit, prefixes)
    if set(paths) != set(tracked_blobs):
        raise NativeTaskRuntimeSourcePacketError(
            ["native_task_runtime_source_git_archive_members_invalid"]
        )
    blobs: list[tuple[str, bytes]] = []
    rows: list[dict[str, Any]] = []
    for source_path in paths:
        data = tracked_blobs[source_path]
        archive_path = f"runtime_sources/{archive_namespace}/{source_path}"
        blobs.append((archive_path, data))
        rows.append(
            {
                "source_path": source_path,
                "archive_path": archive_path,
                "size_bytes": len(data),
                "sha256": _sha256_bytes(data),
            }
        )
    license_row = next(row for row in rows if row["source_path"] == license_path)
    return (
        {
            "repository": repository,
            "commit": commit,
            "tree": expected_tree,
            "archive_namespace": archive_namespace,
            "license": {
                "spdx_id": license_id,
                "source_path": license_path,
                "sha256": license_row["sha256"],
                "redistribution_permitted": True,
            },
            "file_count": len(rows),
            "files": rows,
        },
        blobs,
    )


def _runtime_dependency_rows(
    wheel_dir: Path,
) -> tuple[list[dict[str, Any]], list[tuple[str, bytes]]]:
    expected = {row["filename"] for row in RUNTIME_DEPENDENCY_WHEELS}
    observed = {path.name for path in wheel_dir.glob("*.whl")} if wheel_dir.is_dir() else set()
    if observed != expected:
        raise NativeTaskRuntimeSourcePacketError(
            ["native_task_runtime_dependency_wheel_set_mismatch"]
        )
    rows: list[dict[str, Any]] = []
    blobs: list[tuple[str, bytes]] = []
    for contract in RUNTIME_DEPENDENCY_WHEELS:
        path = wheel_dir / contract["filename"]
        data = path.read_bytes()
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                wheel_members = [name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")]
                if len(wheel_members) != 1:
                    raise NativeTaskRuntimeSourcePacketError(
                        ["native_task_runtime_dependency_wheel_metadata_invalid"]
                    )
                wheel_metadata = archive.read(wheel_members[0]).decode("utf-8")
        except (OSError, zipfile.BadZipFile, UnicodeDecodeError) as exc:
            raise NativeTaskRuntimeSourcePacketError(
                ["native_task_runtime_dependency_wheel_invalid"]
            ) from exc
        if "Root-Is-Purelib: true" not in wheel_metadata or "Tag: py3-none-any" not in wheel_metadata:
            raise NativeTaskRuntimeSourcePacketError(
                ["native_task_runtime_dependency_wheel_not_pure_python"]
            )
        archive_path = f"runtime_dependencies/wheels/{path.name}"
        row = {
            **contract,
            "source": f"https://pypi.org/project/{contract['package']}/{contract['version']}/",
            "archive_path": archive_path,
            "size_bytes": len(data),
            "sha256": _sha256_bytes(data),
            "pure_python": True,
            "redistribution_permitted": True,
        }
        rows.append(row)
        blobs.append((archive_path, data))
    return rows, blobs


def materialize_native_task_runtime_source_packet(
    *,
    output_dir: str | Path,
    isaaclab_repo: str | Path,
    arena_repo: str | Path,
    dependency_wheel_dir: str | Path,
    generated_at: str | None = None,
    isaaclab_commit: str = ISAACLAB_COMMIT,
    isaaclab_tree: str = ISAACLAB_TREE,
    arena_commit: str = ARENA_COMMIT,
    arena_tree: str = ARENA_TREE,
) -> dict[str, Any]:
    """Create one deterministic, digest-bound released-source packet."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    isaaclab_prefixes = ["LICENSE"] + [
        f"source/{name}" for name in ISAACLAB_PACKAGE_NAMES
    ]
    isaaclab, isaaclab_blobs = _repository_rows(
        repo=Path(isaaclab_repo).expanduser().resolve(),
        repository=ISAACLAB_REPOSITORY,
        commit=isaaclab_commit,
        expected_tree=isaaclab_tree,
        license_id="BSD-3-Clause",
        license_path="LICENSE",
        archive_namespace="isaaclab",
        prefixes=isaaclab_prefixes,
    )
    arena, arena_blobs = _repository_rows(
        repo=Path(arena_repo).expanduser().resolve(),
        repository=ARENA_REPOSITORY,
        commit=arena_commit,
        expected_tree=arena_tree,
        license_id="Apache-2.0",
        license_path="LICENSE.md",
        archive_namespace="arena",
        prefixes=("LICENSE.md", "setup.py", "pyproject.toml", "extension.toml", "isaaclab_arena"),
    )
    dependency_rows, dependency_blobs = _runtime_dependency_rows(
        Path(dependency_wheel_dir).expanduser().resolve()
    )
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now_iso(),
        "status": "ready",
        "repositories": [isaaclab, arena],
        "install_roots": list(INSTALL_ROOTS),
        "runtime_dependency_wheels": dependency_rows,
        "source_file_count": isaaclab["file_count"] + arena["file_count"],
        "released_source_only": True,
        "scene_bytes_included": False,
        "policy_bytes_included": False,
        "redistribution_permitted": True,
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    packet_path = destination / "native_task_runtime_sources.zip"
    with zipfile.ZipFile(
        packet_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        entries = [
            ("native_task_runtime_source_manifest.v1.json", manifest_bytes),
            *isaaclab_blobs,
            *arena_blobs,
            *dependency_blobs,
        ]
        for relative, data in sorted(entries):
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": manifest["generated_at"],
        "status": "ready",
        "manifest_digest": manifest["manifest_digest"],
        "repositories": [
            {
                "repository": row["repository"],
                "commit": row["commit"],
                "tree": row["tree"],
                "license": row["license"],
            }
            for row in (isaaclab, arena)
        ],
        "install_roots": list(INSTALL_ROOTS),
        "runtime_dependency_wheels": dependency_rows,
        "source_file_count": manifest["source_file_count"],
        "packet_path": str(packet_path),
        "packet_size_bytes": packet_path.stat().st_size,
        "packet_sha256": _sha256_file(packet_path),
        "scene_bytes_included": False,
        "policy_bytes_included": False,
        "redistribution_permitted": True,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    _write_json(destination / "native_task_runtime_source_packet.v1.json", receipt)
    return receipt


def verify_native_task_runtime_source_packet(
    receipt_path: str | Path,
    *,
    packet_path_override: str | Path | None = None,
) -> dict[str, Any]:
    """Fail closed on receipt, archive, manifest, member, or license drift."""

    path = Path(receipt_path).expanduser().resolve()
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeTaskRuntimeSourcePacketError(
            ["native_task_runtime_source_receipt_invalid"]
        ) from exc
    errors: list[str] = []
    if not isinstance(receipt, Mapping) or receipt.get("schema_version") != SCHEMA_VERSION:
        errors.append("native_task_runtime_source_receipt_invalid")
    elif receipt.get("receipt_digest") != canonical_digest(
        receipt, digest_field="receipt_digest"
    ):
        errors.append("native_task_runtime_source_receipt_digest_invalid")
    packet_path = Path(
        packet_path_override
        if packet_path_override is not None
        else str(receipt.get("packet_path") or "")
    ).expanduser().resolve()
    if not packet_path.is_file():
        errors.append("native_task_runtime_source_packet_missing")
    elif (
        receipt.get("packet_size_bytes") != packet_path.stat().st_size
        or receipt.get("packet_sha256") != _sha256_file(packet_path)
    ):
        errors.append("native_task_runtime_source_packet_identity_mismatch")
    if errors:
        raise NativeTaskRuntimeSourcePacketError(errors)
    try:
        with zipfile.ZipFile(packet_path) as archive:
            names = archive.namelist()
            for name in names:
                pure = PurePosixPath(name)
                if pure.is_absolute() or ".." in pure.parts:
                    errors.append("native_task_runtime_source_archive_path_invalid")
            manifest = json.loads(
                archive.read("native_task_runtime_source_manifest.v1.json").decode("utf-8")
            )
            if (
                not isinstance(manifest, Mapping)
                or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
                or manifest.get("manifest_digest") != canonical_digest(
                    manifest, digest_field="manifest_digest"
                )
                or manifest.get("manifest_digest") != receipt.get("manifest_digest")
            ):
                errors.append("native_task_runtime_source_manifest_invalid")
            expected_names = {"native_task_runtime_source_manifest.v1.json"}
            for repository in manifest.get("repositories") or []:
                for row in repository.get("files") or []:
                    name = str(row.get("archive_path") or "")
                    expected_names.add(name)
                    try:
                        data = archive.read(name)
                    except KeyError:
                        errors.append(f"native_task_runtime_source_member_missing:{name}")
                        continue
                    if row.get("size_bytes") != len(data) or row.get("sha256") != _sha256_bytes(data):
                        errors.append(f"native_task_runtime_source_member_identity_mismatch:{name}")
            for row in manifest.get("runtime_dependency_wheels") or []:
                name = str(row.get("archive_path") or "")
                expected_names.add(name)
                try:
                    data = archive.read(name)
                except KeyError:
                    errors.append(f"native_task_runtime_dependency_wheel_missing:{name}")
                    continue
                if row.get("size_bytes") != len(data) or row.get("sha256") != _sha256_bytes(data):
                    errors.append(f"native_task_runtime_dependency_wheel_identity_mismatch:{name}")
            if manifest.get("runtime_dependency_wheels") != receipt.get(
                "runtime_dependency_wheels"
            ):
                errors.append("native_task_runtime_dependency_receipt_manifest_mismatch")
            if set(names) != expected_names:
                errors.append("native_task_runtime_source_archive_members_invalid")
    except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise NativeTaskRuntimeSourcePacketError(
            ["native_task_runtime_source_archive_invalid"]
        ) from exc
    if errors:
        raise NativeTaskRuntimeSourcePacketError(errors)
    verified = dict(receipt)
    verified["verified_packet_path"] = str(packet_path)
    return verified


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--isaaclab-repo", required=True)
    parser.add_argument("--arena-repo", required=True)
    parser.add_argument("--dependency-wheel-dir", required=True)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)
    receipt = materialize_native_task_runtime_source_packet(
        output_dir=args.output_dir,
        isaaclab_repo=args.isaaclab_repo,
        arena_repo=args.arena_repo,
        dependency_wheel_dir=args.dependency_wheel_dir,
        generated_at=args.generated_at,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI seam
    raise SystemExit(main())


__all__ = [
    "ARENA_COMMIT",
    "ARENA_REPOSITORY",
    "ARENA_TREE",
    "INSTALL_ROOTS",
    "ISAACLAB_COMMIT",
    "ISAACLAB_PACKAGE_NAMES",
    "ISAACLAB_REPOSITORY",
    "ISAACLAB_TREE",
    "MANIFEST_SCHEMA_VERSION",
    "NativeTaskRuntimeSourcePacketError",
    "SCHEMA_VERSION",
    "RUNTIME_DEPENDENCY_WHEELS",
    "materialize_native_task_runtime_source_packet",
    "main",
    "verify_native_task_runtime_source_packet",
]
