"""Package exact released Isaac Lab and Arena sources for paid native tasks.

The digest-pinned Isaac Lab image owns the complete Python/CUDA dependency
environment.  This module exports only the task-neutral released-source closure
needed by native Panda construction.  Keeping dependencies owned by the paired
image prevents a source packet from shadowing a mutually constrained runtime;
the pre-app matrix independently imports and versions that environment before
Kit starts.  Reading exact git objects prevents dirty working-copy bytes from
entering a paid bundle while retaining repository, revision, tree, license,
file, and archive identities.
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
ISAACLAB_COMMIT = "ffff603eafc6b74264a5261cc0183d6a65390d78"
ISAACLAB_TREE = "2f82f1afb2cfaf6816b328e03c7b3ddc12069658"
# Use one released source tree for both the Python APIs and the Kit
# experiences.  Keeping these aliases preserves the packet-builder API while
# making a mixed compatibility revision a typed construction error.
ISAACLAB_RUNTIME_COMPATIBILITY_COMMIT = ISAACLAB_COMMIT
ISAACLAB_RUNTIME_COMPATIBILITY_TREE = ISAACLAB_TREE
ISAACLAB_RUNTIME_COMPATIBILITY_UPSTREAM_FIXES = (
    # Fixes the Isaac Sim 6.0 SimulationManager/PhysxManager shared-view
    # lifecycle and PhysX tensors API compatibility.
    "ca84d35e009e93b03924073e468449c7977a9499",
)
RUNTIME_EXPERIENCE_RELATIVE_PATH = (
    "runtime_sources/isaaclab/apps/isaaclab.python.headless.rendering.kit"
)
RUNTIME_DEPENDENCY_MANIFEST_RELATIVE_PATH = "runtime_sources/isaaclab/source/isaaclab/setup.py"
ARENA_REPOSITORY = "https://github.com/isaac-sim/IsaacLab-Arena.git"
ARENA_COMMIT = "8b82dca224f2b5af08f339f987613c59ce9cdbaa"
ARENA_TREE = "a52514015a8573ac03b6448688bfa61f9cea18a9"
ARENA_ISAACLAB_SUBMODULE_PATH = "submodules/IsaacLab"
ARENA_DOCKERFILE_PATH = "docker/Dockerfile.isaaclab_arena"
ISAAC_SIM_BASE_IMAGE = "nvcr.io/nvidia/isaac-sim:6.0.1"
# The official Isaac Lab beta-2 image contains the complete Python/CUDA runtime
# installed by ``isaaclab.sh --install``.  Its OCI rootfs begins with all 19
# layers of the exact qualified Isaac Sim 6.0.1 linux/amd64 manifest below.
# Shipping only the simulator base caused paid hosts to discover core packages
# (first Torch) one at a time before Kit could start.  Bind both the multi-arch
# index and resolved linux/amd64 manifests so provider selection cannot change
# the stack behind a stable tag.
ISAAC_SIM_BASE_RUNTIME_IMAGE = (
    "nvcr.io/nvidia/isaac-sim:6.0.1@"
    "sha256:783444c706538aa76cf5126e911ddc5e618779e6105305ad4af4260362a30aa9"
)
ISAAC_SIM_BASE_AMD64_MANIFEST_DIGEST = (
    "sha256:b1c542b2ecc549b3d1ebb78c25664aa3bacba1709e6ad8e0a68e09426d57dedb"
)
ISAACLAB_RUNTIME_IMAGE = (
    "nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1@"
    "sha256:ae9c938a16df856effad6dab92115ee0dce2a8813f56847eeeccbebc008d02c4"
)
ISAACLAB_RUNTIME_AMD64_MANIFEST_DIGEST = (
    "sha256:ef451c70084abdf17af3e65fafbb2a8eae1c25d356b418efa285b942f783703f"
)
ISAACLAB_RUNTIME_CONFIG_DIGEST = (
    "sha256:663d4a37ac3019ae3b19418df062a72032db9ce4c6dfb82e894c0f0931807978"
)
# Backward-compatible name used by the generic native-task bundle contract.
# It now means the complete paired runtime, not a bare simulator image.
ISAAC_SIM_RUNTIME_IMAGE = ISAACLAB_RUNTIME_IMAGE
ISAACLAB_PACKAGE_NAMES = (
    "isaaclab",
    "isaaclab_assets",
    "isaaclab_contrib",
    "isaaclab_experimental",
    "isaaclab_mimic",
    "isaaclab_newton",
    "isaaclab_ov",
    "isaaclab_physx",
    "isaaclab_rl",
    "isaaclab_tasks",
    "isaaclab_tasks_experimental",
    "isaaclab_teleop",
    "isaaclab_visualizers",
)
INSTALL_ROOTS = tuple(
    f"runtime_sources/isaaclab/source/{name}" for name in ISAACLAB_PACKAGE_NAMES
) + ("runtime_sources/arena",)
# This tuple is deliberately empty for the complete, digest-pinned Isaac Lab
# runtime image.  The image owns its Python/CUDA dependency environment; adding
# ad-hoc wheels ahead of it can silently replace mutually constrained packages
# (Torch, packaging, typing-extensions, tqdm, and others).  The full pre-app
# dependency matrix imports and versions every required module in one pass.  A
# future overlay is admissible only from that retained image-specific evidence,
# with its complete dependency closure and a hermetic collision test.
RUNTIME_DEPENDENCY_WHEELS: tuple[dict[str, Any], ...] = ()


class NativeTaskRuntimeSourcePacketError(ValueError):
    """Stable source identity or packet validation failure."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _tracked_blobs(repo: Path, commit: str, prefixes: Sequence[str]) -> dict[str, bytes]:
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
    require_head: bool = True,
) -> tuple[dict[str, Any], list[tuple[str, bytes]]]:
    if not repo.is_dir():
        raise NativeTaskRuntimeSourcePacketError(
            [f"native_task_runtime_source_repository_missing:{archive_namespace}"]
        )
    observed_commit = _git(repo, "rev-parse", "HEAD")
    if require_head and observed_commit != commit:
        raise NativeTaskRuntimeSourcePacketError(
            [f"native_task_runtime_source_commit_mismatch:{archive_namespace}"]
        )
    observed_tree = _git(repo, "rev-parse", f"{commit}^{{tree}}")
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


def _arena_pairing_contract(*, repo: Path, commit: str, isaaclab_commit: str) -> dict[str, Any]:
    """Prove Arena, Isaac Lab, and the simulator image are one upstream pair."""

    try:
        submodule_row = _git(repo, "ls-tree", commit, "--", ARENA_ISAACLAB_SUBMODULE_PATH)
        assert isinstance(submodule_row, str)
        mode, object_type, observed_commit, observed_path = submodule_row.split(None, 3)
        dockerfile_bytes = _git(repo, "show", f"{commit}:{ARENA_DOCKERFILE_PATH}", binary=True)
        gitmodules_bytes = _git(repo, "show", f"{commit}:.gitmodules", binary=True)
        assert isinstance(dockerfile_bytes, bytes)
        assert isinstance(gitmodules_bytes, bytes)
        dockerfile = dockerfile_bytes.decode("utf-8")
        gitmodules = gitmodules_bytes.decode("utf-8")
    except (AssertionError, UnicodeDecodeError, ValueError) as exc:
        raise NativeTaskRuntimeSourcePacketError(
            ["native_task_runtime_arena_pairing_contract_invalid"]
        ) from exc
    errors: list[str] = []
    if (
        mode != "160000"
        or object_type != "commit"
        or observed_path != ARENA_ISAACLAB_SUBMODULE_PATH
        or observed_commit != isaaclab_commit
    ):
        errors.append("native_task_runtime_arena_isaaclab_pair_mismatch")
    expected_image_line = f"ARG BASE_IMAGE={ISAAC_SIM_BASE_IMAGE}"
    if expected_image_line not in dockerfile.splitlines():
        errors.append("native_task_runtime_arena_simulator_image_mismatch")
    if (
        f"path = {ARENA_ISAACLAB_SUBMODULE_PATH}" not in gitmodules
        or "github.com:isaac-sim/IsaacLab.git" not in gitmodules
    ):
        errors.append("native_task_runtime_arena_submodule_contract_invalid")
    if errors:
        raise NativeTaskRuntimeSourcePacketError(errors)
    return {
        "arena_repository": ARENA_REPOSITORY,
        "arena_revision": commit,
        "isaaclab_repository": ISAACLAB_REPOSITORY,
        "isaaclab_revision": isaaclab_commit,
        "isaaclab_submodule_path": ARENA_ISAACLAB_SUBMODULE_PATH,
        "simulator_base_image": ISAAC_SIM_BASE_IMAGE,
        "simulator_base_runtime_image": ISAAC_SIM_BASE_RUNTIME_IMAGE,
        "simulator_base_amd64_manifest_digest": (ISAAC_SIM_BASE_AMD64_MANIFEST_DIGEST),
        "simulator_runtime_image": ISAAC_SIM_RUNTIME_IMAGE,
        "runtime_image_kind": "official_isaac_lab_complete_runtime",
        "runtime_image_amd64_manifest_digest": (ISAACLAB_RUNTIME_AMD64_MANIFEST_DIGEST),
        "runtime_image_config_digest": ISAACLAB_RUNTIME_CONFIG_DIGEST,
        "runtime_image_base_layer_prefix_count": 19,
        "runtime_image_layer_count": 40,
        "simulator_dockerfile_path": ARENA_DOCKERFILE_PATH,
        "dockerfile_sha256": _sha256_bytes(dockerfile_bytes),
        "gitmodules_sha256": _sha256_bytes(gitmodules_bytes),
    }


def _runtime_dependency_rows(
    wheel_dir: Path | None,
) -> tuple[list[dict[str, Any]], list[tuple[str, bytes]]]:
    expected = {row["filename"] for row in RUNTIME_DEPENDENCY_WHEELS}
    observed = (
        {path.name for path in wheel_dir.glob("*.whl")}
        if wheel_dir is not None and wheel_dir.is_dir()
        else set()
    )
    if observed != expected:
        raise NativeTaskRuntimeSourcePacketError(
            ["native_task_runtime_dependency_wheel_set_mismatch"]
        )
    rows: list[dict[str, Any]] = []
    blobs: list[tuple[str, bytes]] = []
    for contract in RUNTIME_DEPENDENCY_WHEELS:
        assert wheel_dir is not None
        path = wheel_dir / contract["filename"]
        data = path.read_bytes()
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                wheel_members = [
                    name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")
                ]
                if len(wheel_members) != 1:
                    raise NativeTaskRuntimeSourcePacketError(
                        ["native_task_runtime_dependency_wheel_metadata_invalid"]
                    )
                wheel_metadata = archive.read(wheel_members[0]).decode("utf-8")
        except (OSError, zipfile.BadZipFile, UnicodeDecodeError) as exc:
            raise NativeTaskRuntimeSourcePacketError(
                ["native_task_runtime_dependency_wheel_invalid"]
            ) from exc
        pure_python = bool(contract.get("pure_python", True))
        expected_root = f"Root-Is-Purelib: {str(pure_python).lower()}"
        expected_tag = str(contract.get("wheel_tag", "py3-none-any"))
        if expected_root not in wheel_metadata or f"Tag: {expected_tag}" not in wheel_metadata:
            raise NativeTaskRuntimeSourcePacketError(
                ["native_task_runtime_dependency_wheel_platform_contract_invalid"]
            )
        archive_path = f"runtime_dependencies/wheels/{path.name}"
        row = {
            **contract,
            "source": contract.get("source")
            or f"https://pypi.org/project/{contract['package']}/{contract['version']}/",
            "archive_path": archive_path,
            "size_bytes": len(data),
            "sha256": _sha256_bytes(data),
            "pure_python": pure_python,
            "wheel_tag": expected_tag,
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
    dependency_wheel_dir: str | Path | None = None,
    generated_at: str | None = None,
    isaaclab_commit: str = ISAACLAB_COMMIT,
    isaaclab_tree: str = ISAACLAB_TREE,
    isaaclab_runtime_compatibility_repo: str | Path | None = None,
    isaaclab_runtime_compatibility_commit: str = (ISAACLAB_RUNTIME_COMPATIBILITY_COMMIT),
    isaaclab_runtime_compatibility_tree: str = ISAACLAB_RUNTIME_COMPATIBILITY_TREE,
    arena_commit: str = ARENA_COMMIT,
    arena_tree: str = ARENA_TREE,
) -> dict[str, Any]:
    """Create one deterministic, digest-bound released-source packet."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # Isaac Lab resolves its Kit experience files relative to the repository
    # root at import time.  They are runtime inputs, not optional developer
    # metadata, so keep the complete released ``apps`` directory in the exact
    # source closure alongside the Python packages.
    isaaclab_prefixes = ["LICENSE", "apps"] + [f"source/{name}" for name in ISAACLAB_PACKAGE_NAMES]
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
    dependency_manifest = next(
        (
            data
            for archive_path, data in isaaclab_blobs
            if archive_path == RUNTIME_DEPENDENCY_MANIFEST_RELATIVE_PATH
        ),
        b"",
    )
    if b"warp-lang==1.13.0" not in dependency_manifest:
        raise NativeTaskRuntimeSourcePacketError(
            ["native_task_runtime_dependency_source_contract_invalid:warp-lang"]
        )
    dependency_basis = {
        "package": "warp-lang",
        "version": "1.13.0",
        "requirement": "warp-lang==1.13.0",
        "runtime_owner": "official_isaac_lab_complete_runtime",
        "runtime_image": ISAACLAB_RUNTIME_IMAGE,
        "packet_overlay_required": False,
        "qualification_gate": "native_task_pre_app_dependency_matrix.v2",
        "source_repository": ISAACLAB_REPOSITORY,
        "source_revision": isaaclab_commit,
        "source_tree": isaaclab_tree,
        "relative_path": RUNTIME_DEPENDENCY_MANIFEST_RELATIVE_PATH,
        "sha256": _sha256_bytes(dependency_manifest),
    }
    compatibility_repo = (
        Path(isaaclab_runtime_compatibility_repo or isaaclab_repo).expanduser().resolve()
    )
    if (
        compatibility_repo != Path(isaaclab_repo).expanduser().resolve()
        or isaaclab_runtime_compatibility_commit != isaaclab_commit
        or isaaclab_runtime_compatibility_tree != isaaclab_tree
    ):
        raise NativeTaskRuntimeSourcePacketError(
            ["native_task_runtime_mixed_isaaclab_source_revisions"]
        )
    arena, arena_blobs = _repository_rows(
        repo=Path(arena_repo).expanduser().resolve(),
        repository=ARENA_REPOSITORY,
        commit=arena_commit,
        expected_tree=arena_tree,
        license_id="Apache-2.0",
        license_path="LICENSE.md",
        archive_namespace="arena",
        prefixes=(
            ".gitmodules",
            "LICENSE.md",
            ARENA_DOCKERFILE_PATH,
            "setup.py",
            "pyproject.toml",
            "extension.toml",
            "isaaclab_arena",
        ),
    )
    paired_stack = _arena_pairing_contract(
        repo=Path(arena_repo).expanduser().resolve(),
        commit=arena_commit,
        isaaclab_commit=isaaclab_commit,
    )
    dependency_rows, dependency_blobs = _runtime_dependency_rows(
        (
            Path(dependency_wheel_dir).expanduser().resolve()
            if dependency_wheel_dir is not None
            else None
        )
    )
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now_iso(),
        "status": "ready",
        "repositories": [isaaclab, arena],
        "paired_stack": paired_stack,
        "runtime_experience": {
            "relative_path": RUNTIME_EXPERIENCE_RELATIVE_PATH,
            "repository": ISAACLAB_REPOSITORY,
            "source_revision": isaaclab_commit,
            "source_tree": isaaclab_tree,
            "upstream_fix_revisions": list(ISAACLAB_RUNTIME_COMPATIBILITY_UPSTREAM_FIXES),
            "sha256": next(
                row["sha256"]
                for row in isaaclab["files"]
                if row["archive_path"] == RUNTIME_EXPERIENCE_RELATIVE_PATH
            ),
        },
        "install_roots": list(INSTALL_ROOTS),
        "runtime_dependency_wheels": dependency_rows,
        "runtime_dependency_basis": dependency_basis,
        "source_file_count": (isaaclab["file_count"] + arena["file_count"]),
        "released_source_only": True,
        "scene_bytes_included": False,
        "policy_bytes_included": False,
        "redistribution_permitted": True,
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
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
        "runtime_experience": manifest["runtime_experience"],
        "paired_stack": paired_stack,
        "install_roots": list(INSTALL_ROOTS),
        "runtime_dependency_wheels": dependency_rows,
        "runtime_dependency_basis": dependency_basis,
        "source_file_count": manifest["source_file_count"],
        "packet_path": str(packet_path),
        "packet_size_bytes": packet_path.stat().st_size,
        "packet_sha256": _sha256_file(packet_path),
        "scene_bytes_included": False,
        "policy_bytes_included": False,
        "redistribution_permitted": True,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
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
    elif receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest"):
        errors.append("native_task_runtime_source_receipt_digest_invalid")
    packet_path = (
        Path(
            packet_path_override
            if packet_path_override is not None
            else str(receipt.get("packet_path") or "")
        )
        .expanduser()
        .resolve()
    )
    if not packet_path.is_file():
        errors.append("native_task_runtime_source_packet_missing")
    elif receipt.get("packet_size_bytes") != packet_path.stat().st_size or receipt.get(
        "packet_sha256"
    ) != _sha256_file(packet_path):
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
                or manifest.get("manifest_digest")
                != canonical_digest(manifest, digest_field="manifest_digest")
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
                    if row.get("size_bytes") != len(data) or row.get("sha256") != _sha256_bytes(
                        data
                    ):
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
            if manifest.get("runtime_dependency_basis") != receipt.get("runtime_dependency_basis"):
                errors.append("native_task_runtime_dependency_basis_receipt_manifest_mismatch")
            if manifest.get("runtime_experience") != receipt.get("runtime_experience"):
                errors.append("native_task_runtime_experience_receipt_manifest_mismatch")
            if manifest.get("paired_stack") != receipt.get("paired_stack"):
                errors.append("native_task_runtime_paired_stack_receipt_manifest_mismatch")
            repositories = {
                str(row.get("repository") or ""): row
                for row in manifest.get("repositories") or []
                if isinstance(row, Mapping)
            }
            lab = repositories.get(ISAACLAB_REPOSITORY, {})
            arena = repositories.get(ARENA_REPOSITORY, {})
            paired = manifest.get("paired_stack") or {}
            runtime_experience = manifest.get("runtime_experience") or {}
            dependency_basis = manifest.get("runtime_dependency_basis") or {}
            arena_files = {
                str(row.get("source_path") or ""): row
                for row in arena.get("files") or []
                if isinstance(row, Mapping)
            }
            if (
                len(repositories) != 2
                or paired.get("arena_repository") != ARENA_REPOSITORY
                or paired.get("arena_revision") != arena.get("commit")
                or paired.get("isaaclab_repository") != ISAACLAB_REPOSITORY
                or paired.get("isaaclab_revision") != lab.get("commit")
                or paired.get("isaaclab_submodule_path") != ARENA_ISAACLAB_SUBMODULE_PATH
                or paired.get("simulator_base_image") != ISAAC_SIM_BASE_IMAGE
                or paired.get("simulator_base_runtime_image") != ISAAC_SIM_BASE_RUNTIME_IMAGE
                or paired.get("simulator_base_amd64_manifest_digest")
                != ISAAC_SIM_BASE_AMD64_MANIFEST_DIGEST
                or paired.get("simulator_runtime_image") != ISAAC_SIM_RUNTIME_IMAGE
                or paired.get("runtime_image_kind") != "official_isaac_lab_complete_runtime"
                or paired.get("runtime_image_amd64_manifest_digest")
                != ISAACLAB_RUNTIME_AMD64_MANIFEST_DIGEST
                or paired.get("runtime_image_config_digest") != ISAACLAB_RUNTIME_CONFIG_DIGEST
                or paired.get("runtime_image_base_layer_prefix_count") != 19
                or paired.get("runtime_image_layer_count") != 40
                or paired.get("simulator_dockerfile_path") != ARENA_DOCKERFILE_PATH
                or paired.get("dockerfile_sha256")
                != arena_files.get(ARENA_DOCKERFILE_PATH, {}).get("sha256")
                or paired.get("gitmodules_sha256")
                != arena_files.get(".gitmodules", {}).get("sha256")
                or runtime_experience.get("source_revision") != lab.get("commit")
                or runtime_experience.get("source_tree") != lab.get("tree")
                or dependency_basis.get("source_revision") != lab.get("commit")
                or dependency_basis.get("source_tree") != lab.get("tree")
            ):
                errors.append("native_task_runtime_paired_stack_contract_invalid")
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
    parser.add_argument("--dependency-wheel-dir")
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
    "ARENA_DOCKERFILE_PATH",
    "ARENA_ISAACLAB_SUBMODULE_PATH",
    "ARENA_REPOSITORY",
    "ARENA_TREE",
    "INSTALL_ROOTS",
    "ISAACLAB_COMMIT",
    "ISAACLAB_PACKAGE_NAMES",
    "ISAACLAB_REPOSITORY",
    "ISAACLAB_TREE",
    "ISAAC_SIM_BASE_IMAGE",
    "MANIFEST_SCHEMA_VERSION",
    "NativeTaskRuntimeSourcePacketError",
    "SCHEMA_VERSION",
    "RUNTIME_DEPENDENCY_WHEELS",
    "materialize_native_task_runtime_source_packet",
    "main",
    "verify_native_task_runtime_source_packet",
]
