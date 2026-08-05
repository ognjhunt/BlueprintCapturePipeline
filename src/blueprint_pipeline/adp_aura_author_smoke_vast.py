"""Canonical zero-retry Vast execution for AuraFusion360's author smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .paid_resource_admission import PaidResourceAdmissionGrant
from .provider_runtime_bundle_contract import provider_runtime_contract_blockers
from .public_scene_method_prerequisites import _hf_repository_info
from .model_access_env import normalize_model_access_env
from .vast_provider_adapter import run_vast_provider_adapter
from .vast_session_budget_contract import attempt_estimated_cost, attempt_runtime_seconds
from .wam_provider_object_store import (
    cleanup_staged_wam_provider_objects,
    stage_wam_provider_bundle_object_store,
)


PROBE_KIND = "adp-aurafusion360-author-smoke"
PROVIDER_BUNDLE_KIND = "adp_aura_smoke"
RESULT_SCHEMA_VERSION = "adp_aura_author_smoke_vast_run.v1"
AUTHOR_DATA_RECEIPT_SCHEMA_VERSION = "adp_aura_author_data_materialization.v1"
SOURCE_COMMIT = "f23b26c44ba84608306ba952510533ebf4c7877d"
SOURCE_TREE = "cc8447c66448b29bb4d39fec29c031df63d4b179"
SOURCE_REPOSITORY = "https://github.com/kkennethwu/AuraFusion360_official"
SAM2_SOURCE_REPOSITORY = "https://github.com/facebookresearch/sam2"
SAM2_SOURCE_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_SOURCE_TREE = "64becbca23f880e0056449377496da248a74da43"
SAM2_LICENSE_SHA256 = (
    "sha256:c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
)
WONDERWORLD_SOURCE_REPOSITORY = "https://github.com/KovenYu/WonderWorld"
WONDERWORLD_SOURCE_COMMIT = "cae41f9a24a9c5513a7eea8939ee14fa0576162d"
WONDERWORLD_SOURCE_TREE = "c9882769e0d8a6823055f76d66b9586fa8433003"
WONDERWORLD_MARIGOLD_LICENSE_SHA256 = (
    "sha256:0cec06e0e55fbc3dc5cee4fca9b607f66cb8f4e4dbcf3b3c013594dd156732e9"
)
WONDERWORLD_MARIGOLD_RUNTIME_FILES = {
    "utils/utils/LICENSE.txt": "marigold_module/LICENSE.txt",
    "utils/utils/batchsize.py": "marigold_lcm/util/batchsize.py",
    "utils/utils/ensemble.py": "marigold_lcm/util/ensemble.py",
    "utils/utils/image_util.py": "marigold_lcm/util/image_util.py",
}
SUBMODULES = {
    "submodules/diff-surfel-rasterization": "e0ed0207b3e0669960cfad70852200a4a5847f61",
    "submodules/diff-surfel-rasterization/third_party/glm": (
        "5c46b9c07008ae65cb81ab79cd677ecc1934b903"
    ),
    "submodules/simple-knn": "86710c2d4b46680c02301765dd79e465819c8f19",
}
DEFAULT_IMAGE = (
    "docker.io/nvidia/cuda@"
    "sha256:5645fec64549cc35930eee9d85aafd2b0006c0c3f22632be5a1d85e2604e9749"
)
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/aurafusion360-author-smoke"
PREREQUISITE_RECEIPT_DIGEST = (
    "sha256:026ab58fb71d1bb7eb0fb2530f9e181f9d32d2e7b9a526ca5156a7c3fc4b2c17"
)
_VAST_MUTATION_ENV = (
    "BLUEPRINT_ALLOW_VAST_API_CALLS",
    "BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH",
)
_VAST_SINGLE_ATTEMPT_ENV = "BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS"

_AUTHOR_DATA = {
    "repository": "kkennethwu/360-USID",
    "revision": "1c5a7863a284efe60fa33c2e4f0da19e50e55a0f",
    "path_prefix": "360-USID/sunflower/",
    "snapshot_digest": "sha256:e81ad45f416349cc6364edecdb156e7ca48f8d62b25443d5ea3225c4a8af07fd",
}
_EXPECTED_OUTPUT = {
    "repository": "kkennethwu/AuraFusion360_Results",
    "revision": "9682454f9829c6df013d3d22c55f3a9c53dd783f",
    "path_prefix": "360-USID/sunflower/",
    "snapshot_digest": "sha256:b6c92fae7bc42253f7e5ce4afa4f4adc32c7748783e5ec54616d76de638e45cc",
    "expected_ply_path": (
        "360-USID/sunflower/point_cloud/iteration_object_inpaint_init/point_cloud.ply"
    ),
    "expected_ply_size_bytes": 206_112_053,
    "expected_ply_sha256": (
        "sha256:0b493f2f15ea942507faa179e6cc9f5250ff985d6059272dd676bd7a92dde9da"
    ),
}
_RUNTIME_MODELS = [
    {
        "repository": "facebook/sam2-hiera-large",
        "revision": "e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251",
        "snapshot_digest": (
            "sha256:e873cf416a96e5656a1f9641325c299399c81b28612362c4ed14375cb82169c6"
        ),
    },
    {
        "repository": "prs-eth/marigold-depth-v1-0",
        "revision": "f4fc453d7d217cbe30ddcad3eb311d1ad9a11c4c",
        "snapshot_digest": (
            "sha256:d8c2d2dfdc297008d7f4314593b8dcd381fa6c120ce9780e93f1a9c15c206e74"
        ),
    },
]
_SD2 = {
    "repository": "sd2-community/stable-diffusion-2-inpainting",
    "revision": "5f74973cbb64c8568780732c17f43eb269d63a0d",
    "path": "512-inpainting-ema.ckpt",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_executable(path: Path, source: Path) -> None:
    shutil.copy2(source, path)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _tracked_files(repo: Path, prefix: str = "") -> list[tuple[str, Path]]:
    raw = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    rows: list[tuple[str, Path]] = []
    for value in raw.split(b"\0"):
        if not value:
            continue
        relative = value.decode("utf-8")
        path = repo / relative
        if path.is_file():
            rows.append(((prefix + relative).lstrip("/"), path))
    return rows


def _source_files(source: Path) -> list[tuple[str, Path]]:
    rows = _tracked_files(source)
    for relative in SUBMODULES:
        rows.extend(_tracked_files(source / relative, prefix=relative + "/"))
    by_name = {name: path for name, path in rows}
    return sorted(by_name.items())


def _source_manifest(rows: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    return [
        {"path": name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for name, path in rows
    ]


def _deterministic_zip_files(rows: list[tuple[str, Path]], destination: Path) -> None:
    with zipfile.ZipFile(destination, "w") as archive:
        for name, path in rows:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
            archive.writestr(
                info,
                path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def _deterministic_zip_directory(source: Path, destination: Path) -> None:
    _deterministic_zip_files(
        [
            (path.relative_to(source.parent).as_posix(), path)
            for path in sorted(source.rglob("*"))
            if path.is_file()
        ],
        destination,
    )


def _publisher_file_rows(info: Mapping[str, Any], prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sibling in info.get("siblings") or []:
        if not isinstance(sibling, Mapping):
            continue
        path = str(sibling.get("rfilename") or "")
        if not path.startswith(prefix):
            continue
        lfs_value = sibling.get("lfs")
        lfs = lfs_value if isinstance(lfs_value, Mapping) else {}
        rows.append(
            {
                "path": path,
                "size_bytes": int(sibling.get("size") or lfs.get("size") or 0),
                "lfs_sha256": lfs.get("sha256"),
                "git_blob_id": sibling.get("blobId"),
            }
        )
    return sorted(rows, key=lambda row: row["path"])


def _git_blob_id(path: Path) -> str:
    content = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(content)}\0".encode("ascii") + content,
        usedforsecurity=False,
    ).hexdigest()


def _download_author_snapshot(destination: Path) -> None:
    normalize_model_access_env()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    previous = os.environ.get("HF_HUB_DISABLE_XET")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=_AUTHOR_DATA["repository"],
            repo_type="dataset",
            revision=_AUTHOR_DATA["revision"],
            allow_patterns=[_AUTHOR_DATA["path_prefix"] + "*"],
            local_dir=destination,
            max_workers=1,
            token=token,
        )
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_DISABLE_XET", None)
        else:
            os.environ["HF_HUB_DISABLE_XET"] = previous


def materialize_aura_author_data(
    *,
    prerequisite_receipt_path: str | Path,
    output_root: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Download the exact author scene once and bind every materialized file."""

    prerequisite = _read_json(Path(prerequisite_receipt_path).expanduser().resolve())
    snapshots = _validate_prerequisite(prerequisite)
    author_snapshot = snapshots["aurafusion360_sunflower_author_scene"].get("publisher")
    if (
        not isinstance(author_snapshot, Mapping)
        or author_snapshot.get("repository") != _AUTHOR_DATA["repository"]
        or author_snapshot.get("revision") != _AUTHOR_DATA["revision"]
        or author_snapshot.get("path_prefix") != _AUTHOR_DATA["path_prefix"]
        or author_snapshot.get("snapshot_digest") != _AUTHOR_DATA["snapshot_digest"]
    ):
        raise ValueError("adp_aura_author_data_prerequisite_identity_invalid")
    root = Path(output_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("adp_aura_author_data_output_root_not_empty")
    data = root / "data"
    ensure_dir(data)
    info = _hf_repository_info(
        repo_type="dataset",
        repository=_AUTHOR_DATA["repository"],
        revision=_AUTHOR_DATA["revision"],
    )
    publisher_files = _publisher_file_rows(info, _AUTHOR_DATA["path_prefix"])
    if canonical_digest({"files": publisher_files}) != _AUTHOR_DATA["snapshot_digest"]:
        raise ValueError("adp_aura_author_data_publisher_snapshot_changed")
    _download_author_snapshot(data)
    cache = data / ".cache"
    if cache.is_dir():
        shutil.rmtree(cache)
    actual_paths = {
        path.relative_to(data).as_posix()
        for path in data.rglob("*")
        if path.is_file()
    }
    expected_paths = {str(row["path"]) for row in publisher_files}
    if actual_paths != expected_paths:
        raise ValueError("adp_aura_author_data_materialized_file_set_mismatch")
    files: list[dict[str, Any]] = []
    for publisher in publisher_files:
        path = data / str(publisher["path"])
        sha256 = _sha256(path)
        lfs_sha256 = publisher.get("lfs_sha256")
        git_blob_id = publisher.get("git_blob_id")
        if not lfs_sha256 and not git_blob_id:
            raise ValueError("adp_aura_author_data_publisher_file_identity_missing")
        if path.stat().st_size != publisher["size_bytes"]:
            raise ValueError("adp_aura_author_data_materialized_size_mismatch")
        if lfs_sha256 and sha256 != "sha256:" + str(lfs_sha256):
            raise ValueError("adp_aura_author_data_materialized_lfs_hash_mismatch")
        if not lfs_sha256 and git_blob_id and _git_blob_id(path) != git_blob_id:
            raise ValueError("adp_aura_author_data_materialized_git_blob_mismatch")
        files.append({**publisher, "sha256": sha256})
    receipt: dict[str, Any] = {
        "schema_version": AUTHOR_DATA_RECEIPT_SCHEMA_VERSION,
        "generated_at": generated_at or utc_now_iso(),
        "status": "completed",
        "publisher": {
            "repository": _AUTHOR_DATA["repository"],
            "revision": _AUTHOR_DATA["revision"],
            "path_prefix": _AUTHOR_DATA["path_prefix"],
            "snapshot_digest": _AUTHOR_DATA["snapshot_digest"],
        },
        "prerequisite_receipt_digest": prerequisite["receipt_digest"],
        "files": files,
        "file_count": len(files),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in files),
        "rights_established": True,
        "raw_secret_values_recorded": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(root / "adp_aura_author_data_materialization_receipt.json", receipt)
    return receipt


def _validate_materialized_author_data(
    receipt: Mapping[str, Any], *, data_root: Path
) -> tuple[list[tuple[str, Path]], list[dict[str, Any]]]:
    if (
        receipt.get("schema_version") != AUTHOR_DATA_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "completed"
        or canonical_digest(receipt, digest_field="receipt_digest")
        != receipt.get("receipt_digest")
        or receipt.get("prerequisite_receipt_digest") != PREREQUISITE_RECEIPT_DIGEST
        or receipt.get("rights_established") is not True
    ):
        raise ValueError("adp_aura_author_data_receipt_invalid")
    publisher = receipt.get("publisher")
    if not isinstance(publisher, Mapping) or any(
        publisher.get(key) != _AUTHOR_DATA[key]
        for key in ("repository", "revision", "path_prefix", "snapshot_digest")
    ):
        raise ValueError("adp_aura_author_data_receipt_publisher_mismatch")
    files = receipt.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("adp_aura_author_data_receipt_files_missing")
    if receipt.get("file_count") != len(files) or receipt.get(
        "total_size_bytes"
    ) != sum(int(item.get("size_bytes") or 0) for item in files if isinstance(item, Mapping)):
        raise ValueError("adp_aura_author_data_receipt_aggregate_mismatch")
    publisher_files: list[dict[str, Any]] = []
    rows: list[tuple[str, Path]] = []
    expected_paths: set[str] = set()
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("adp_aura_author_data_receipt_file_invalid")
        relative = str(item.get("path") or "")
        path = (data_root / relative).resolve()
        if not relative or (data_root != path and data_root not in path.parents):
            raise ValueError("adp_aura_author_data_path_outside_root")
        if (
            not path.is_file()
            or path.stat().st_size != item.get("size_bytes")
            or _sha256(path) != item.get("sha256")
        ):
            raise ValueError("adp_aura_author_data_materialized_bytes_changed")
        lfs_sha256 = item.get("lfs_sha256")
        git_blob_id = item.get("git_blob_id")
        if lfs_sha256:
            if item.get("sha256") != "sha256:" + str(lfs_sha256):
                raise ValueError("adp_aura_author_data_receipt_lfs_hash_mismatch")
        elif not git_blob_id or _git_blob_id(path) != git_blob_id:
            raise ValueError("adp_aura_author_data_receipt_git_blob_mismatch")
        expected_paths.add(relative)
        rows.append((relative, path))
        publisher_files.append(
            {
                "path": relative,
                "size_bytes": item.get("size_bytes"),
                "lfs_sha256": item.get("lfs_sha256"),
                "git_blob_id": item.get("git_blob_id"),
            }
        )
    actual_paths = {
        path.relative_to(data_root).as_posix()
        for path in data_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError("adp_aura_author_data_materialized_file_set_changed")
    if canonical_digest({"files": publisher_files}) != _AUTHOR_DATA["snapshot_digest"]:
        raise ValueError("adp_aura_author_data_receipt_snapshot_digest_mismatch")
    return sorted(rows), [dict(item) for item in files]


def _validate_prerequisite(receipt: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if (
        receipt.get("receipt_digest") != PREREQUISITE_RECEIPT_DIGEST
        or canonical_digest(receipt, digest_field="receipt_digest")
        != PREREQUISITE_RECEIPT_DIGEST
    ):
        raise ValueError("adp_aura_prerequisite_receipt_digest_mismatch")
    method = (receipt.get("methods") or {}).get("aurafusion360_quality_challenger")
    if not isinstance(method, Mapping):
        raise ValueError("adp_aura_prerequisite_role_missing")
    if (
        method.get("checkpoint_rights_established") is not True
        or method.get("author_data_rights_established") is not True
    ):
        raise ValueError("adp_aura_prerequisite_rights_missing")
    snapshots = {
        str(item.get("artifact_id")): item
        for item in method.get("remote_snapshots") or []
        if isinstance(item, Mapping)
    }
    required = {
        "aurafusion360_sunflower_author_scene",
        "aurafusion360_sunflower_expected_output",
        "aurafusion360_sam2_hiera_large",
        "aurafusion360_marigold_depth_v1_0",
        "aurafusion360_sd2_inpainting_exact_checkpoint",
    }
    if set(snapshots) != required or any(
        item.get("rights_established") is not True for item in snapshots.values()
    ):
        raise ValueError("adp_aura_prerequisite_snapshot_binding_invalid")
    sd2_publisher = snapshots["aurafusion360_sd2_inpainting_exact_checkpoint"].get(
        "publisher"
    )
    if not isinstance(sd2_publisher, Mapping):
        raise ValueError("adp_aura_sd2_publisher_identity_missing")
    sd2_identity = sd2_publisher.get("single_file_identity")
    if (
        sd2_publisher.get("repository") != _SD2["repository"]
        or sd2_publisher.get("revision") != _SD2["revision"]
        or sd2_publisher.get("path_prefix") != _SD2["path"]
        or not isinstance(sd2_identity, Mapping)
        or sd2_identity.get("path") != _SD2["path"]
        or int(sd2_identity.get("size_bytes") or 0) <= 0
        or not str(sd2_identity.get("lfs_sha256") or "").startswith("sha256:")
    ):
        raise ValueError("adp_aura_sd2_publisher_identity_invalid")
    return snapshots


def build_aura_author_smoke_vast_bundle(
    *,
    repo_root: str | Path,
    aura_root: str | Path,
    sam2_root: str | Path,
    wonderworld_root: str | Path,
    prerequisite_receipt_path: str | Path,
    author_data_root: str | Path,
    author_data_receipt_path: str | Path,
    job_dir: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build an immutable bundle from locally inspected author-scene bytes."""

    repo = Path(repo_root).expanduser().resolve()
    source = Path(aura_root).expanduser().resolve()
    sam2_source = Path(sam2_root).expanduser().resolve()
    wonderworld_source = Path(wonderworld_root).expanduser().resolve()
    prerequisite_path = Path(prerequisite_receipt_path).expanduser().resolve()
    author_root = Path(author_data_root).expanduser().resolve()
    author_receipt_path = Path(author_data_receipt_path).expanduser().resolve()
    job = Path(job_dir).expanduser().resolve()
    if job.exists() and any(job.iterdir()):
        raise ValueError("adp_aura_bundle_job_dir_not_empty")
    runtime = job / "provider_runtime"
    ensure_dir(runtime)
    if (
        _git(source, "rev-parse", "HEAD") != SOURCE_COMMIT
        or _git(source, "rev-parse", "HEAD^{tree}") != SOURCE_TREE
        or _git(source, "status", "--porcelain")
    ):
        raise ValueError("adp_aura_source_identity_mismatch")
    observed_submodules = {
        relative: _git(source / relative, "rev-parse", "HEAD") for relative in SUBMODULES
    }
    if observed_submodules != SUBMODULES or any(
        _git(source / relative, "status", "--porcelain") for relative in SUBMODULES
    ):
        raise ValueError("adp_aura_submodule_identity_mismatch")
    if (
        _git(sam2_source, "rev-parse", "HEAD") != SAM2_SOURCE_COMMIT
        or _git(sam2_source, "rev-parse", "HEAD^{tree}") != SAM2_SOURCE_TREE
        or _git(sam2_source, "status", "--porcelain")
    ):
        raise ValueError("adp_aura_sam2_source_identity_mismatch")
    sam2_license = sam2_source / "LICENSE"
    if not sam2_license.is_file() or _sha256(sam2_license) != SAM2_LICENSE_SHA256:
        raise ValueError("adp_aura_sam2_source_license_mismatch")
    if (
        _git(wonderworld_source, "rev-parse", "HEAD") != WONDERWORLD_SOURCE_COMMIT
        or _git(wonderworld_source, "rev-parse", "HEAD^{tree}")
        != WONDERWORLD_SOURCE_TREE
        or _git(wonderworld_source, "status", "--porcelain")
    ):
        raise ValueError("adp_aura_wonderworld_source_identity_mismatch")
    wonderworld_license = wonderworld_source / "marigold_module/LICENSE.txt"
    if (
        not wonderworld_license.is_file()
        or _sha256(wonderworld_license) != WONDERWORLD_MARIGOLD_LICENSE_SHA256
    ):
        raise ValueError("adp_aura_wonderworld_marigold_license_mismatch")
    wonderworld_rows = [
        (archive_path, wonderworld_source / source_path)
        for archive_path, source_path in sorted(WONDERWORLD_MARIGOLD_RUNTIME_FILES.items())
    ]
    if any(not path.is_file() for _, path in wonderworld_rows):
        raise ValueError("adp_aura_wonderworld_marigold_runtime_file_missing")
    prerequisite = _read_json(prerequisite_path)
    prerequisite_snapshots = _validate_prerequisite(prerequisite)
    sd2_identity = prerequisite_snapshots[
        "aurafusion360_sd2_inpainting_exact_checkpoint"
    ]["publisher"]["single_file_identity"]
    sd2_checkpoint = {
        **_SD2,
        "size_bytes": sd2_identity["size_bytes"],
        "sha256": sd2_identity["lfs_sha256"],
    }
    author_receipt = _read_json(author_receipt_path)
    author_rows, author_manifest = _validate_materialized_author_data(
        author_receipt, data_root=author_root
    )

    rows = _source_files(source)
    source_manifest = _source_manifest(rows)
    _deterministic_zip_files(rows, runtime / "aurafusion360_source.zip")
    sam2_rows = _tracked_files(sam2_source)
    sam2_manifest = _source_manifest(sam2_rows)
    _deterministic_zip_files(sam2_rows, runtime / "sam2_source.zip")
    wonderworld_manifest = _source_manifest(wonderworld_rows)
    _deterministic_zip_files(
        wonderworld_rows, runtime / "wonderworld_marigold_runtime.zip"
    )
    _deterministic_zip_files(author_rows, runtime / "author_data.zip")
    author_archive_sha256 = _sha256(runtime / "author_data.zip")
    smoke_spec = {
        "schema_version": "adp_aura_author_smoke_spec.v1",
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "submodules": SUBMODULES,
        "source_files": source_manifest,
        "sam2_source": {
            "repository": SAM2_SOURCE_REPOSITORY,
            "commit": SAM2_SOURCE_COMMIT,
            "tree": SAM2_SOURCE_TREE,
            "license": "Apache-2.0",
            "license_sha256": SAM2_LICENSE_SHA256,
            "source_files": sam2_manifest,
        },
        "wonderworld_marigold_runtime": {
            "repository": WONDERWORLD_SOURCE_REPOSITORY,
            "commit": WONDERWORLD_SOURCE_COMMIT,
            "tree": WONDERWORLD_SOURCE_TREE,
            "license": "Apache-2.0",
            "license_sha256": WONDERWORLD_MARIGOLD_LICENSE_SHA256,
            "archive": "wonderworld_marigold_runtime.zip",
            "archive_sha256": _sha256(
                runtime / "wonderworld_marigold_runtime.zip"
            ),
            "source_files": wonderworld_manifest,
        },
        "author_data": {
            **_AUTHOR_DATA,
            "materialized": True,
            "archive": "author_data.zip",
            "archive_sha256": author_archive_sha256,
            "materialization_receipt_digest": author_receipt["receipt_digest"],
            "files": author_manifest,
        },
        "expected_output": _EXPECTED_OUTPUT,
        "runtime_models": _RUNTIME_MODELS,
        "sd2_checkpoint": sd2_checkpoint,
        "author_commands": [
            ["python", "train.py", "--config", "configs/360-USID/sunflower/train.config"],
            [
                "python", "render.py", "-s", "data/360-USID/sunflower", "-m",
                "output/360-USID/sunflower", "--skip_mesh", "--render_path",
                "--iteration", "30000",
            ],
            ["python", "remove.py", "--config", "configs/360-USID/sunflower/remove.config"],
            ["python", "utils/sam2_utils.py", "--dataset", "360-USID", "--scene", "sunflower"],
            ["python", "inpaint.py", "--config", "configs/360-USID/sunflower/inpaint.config"],
        ],
        "scope": "unchanged_author_workflow_through_inpaint_init",
        "depth_anything3_used": False,
    }
    write_json(runtime / "smoke_spec.json", smoke_spec)
    scripts = repo / "scripts"
    _write_executable(
        runtime / "run_adp_aura_author_smoke_provider_runtime.sh",
        scripts / "run_adp_aura_author_smoke_provider_runtime.sh",
    )
    shutil.copy2(
        scripts / "adp_aura_author_smoke_provider_runner.py",
        runtime / "adp_aura_author_smoke_provider_runner.py",
    )
    entrypoint = runtime / "run_adp_aura_author_smoke_provider_runtime.sh"
    runner = runtime / "adp_aura_author_smoke_provider_runner.py"
    blockers = provider_runtime_contract_blockers(
        provider_bundle_kind=PROVIDER_BUNDLE_KIND,
        entrypoint_text=entrypoint.read_text(encoding="utf-8"),
        runner_text=runner.read_text(encoding="utf-8"),
    )
    generated = generated_at or utc_now_iso()
    readiness = {
        "schema_version": "adp_aura_author_smoke_provider_bundle.v1",
        "generated_at": generated,
        "status": "ready" if not blockers else "blocked",
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "submodules": SUBMODULES,
        "source_archive_sha256": _sha256(runtime / "aurafusion360_source.zip"),
        "source_manifest_digest": canonical_digest({"files": source_manifest}),
        "sam2_source_archive_sha256": _sha256(runtime / "sam2_source.zip"),
        "sam2_source_manifest_digest": canonical_digest({"files": sam2_manifest}),
        "wonderworld_source_commit": WONDERWORLD_SOURCE_COMMIT,
        "wonderworld_source_tree": WONDERWORLD_SOURCE_TREE,
        "wonderworld_marigold_runtime_archive_sha256": _sha256(
            runtime / "wonderworld_marigold_runtime.zip"
        ),
        "wonderworld_marigold_runtime_manifest_digest": canonical_digest(
            {"files": wonderworld_manifest}
        ),
        "wonderworld_marigold_runtime_license": "Apache-2.0",
        "author_data_archive_sha256": author_archive_sha256,
        "author_data_materialization_receipt_digest": author_receipt["receipt_digest"],
        "author_data_file_count": len(author_manifest),
        "author_data_total_size_bytes": sum(
            int(item["size_bytes"]) for item in author_manifest
        ),
        "prerequisite_receipt_digest": prerequisite["receipt_digest"],
        "container_image": DEFAULT_IMAGE,
        "container_platform": "linux/amd64",
        "python_version": "3.10",
        "cuda_toolkit": "12.4.1",
        "torch_version": "2.5.1+cu124",
        "sam2_source_commit": SAM2_SOURCE_COMMIT,
        "sam2_source_tree": SAM2_SOURCE_TREE,
        "sam2_source_license": "Apache-2.0",
        "sd2_checkpoint_identity": sd2_checkpoint,
        "runtime_entrypoint": (
            "provider_runtime/run_adp_aura_author_smoke_provider_runtime.sh"
        ),
        "expected_output_filename": "adp_aura_author_smoke_result.json",
        "smoke_scope": "unchanged_author_workflow_through_inpaint_init",
        "full_aurafusion360_workflow_executed": False,
        "depth_anything3_used": False,
        "provider_zero_required_after_return": True,
        "retry_cap": 0,
        "local_bundle_ready_for_remote_staging": not blockers,
        "blockers": blockers,
        "raw_secret_values_recorded": False,
    }
    write_json(runtime / "adp_aura_author_smoke_provider_manifest.json", readiness)
    bundle = job / "adp_aura_author_smoke_provider_runtime_bundle.zip"
    _deterministic_zip_directory(runtime, bundle)
    receipt = {
        **readiness,
        "bundle_path": str(bundle),
        "bundle_sha256": _sha256(bundle),
        "bundle_size_bytes": bundle.stat().st_size,
    }
    write_json(job / "adp_aura_author_smoke_bundle_receipt.json", receipt)
    return receipt


def _remaining_minutes(
    *, job: Path, hard_cap_usd: float, hard_ttl_seconds: int, max_hourly_rate_usd: float
) -> int:
    ledger = _read_json(job / "adp_aura_vast_session_budget.json")
    attempts = [row for row in ledger.get("attempts") or [] if isinstance(row, Mapping)]
    prior_seconds = sum(attempt_runtime_seconds(row) for row in attempts)
    prior_cost = sum(attempt_estimated_cost(row) for row in attempts)
    runtime_minutes = math.floor(max(0.0, hard_ttl_seconds - prior_seconds) / 60.0)
    spend_minutes = math.floor(
        max(0.0, hard_cap_usd - prior_cost) * 60.0 / max_hourly_rate_usd
    )
    return max(0, min(runtime_minutes, spend_minutes))


def _extract(path: Path, destination: Path) -> dict[str, Any]:
    blockers: list[str] = []
    if not path.is_file():
        return {"status": "blocked", "blockers": ["aura_provider_output_zip_missing"]}
    if destination.exists() and any(destination.iterdir()):
        return {"status": "blocked", "blockers": ["aura_output_destination_not_empty"]}
    ensure_dir(destination)
    root = destination.resolve()
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if root not in target.parents and target != root:
                    blockers.append("aura_provider_output_zip_path_traversal")
            if not blockers:
                archive.extractall(destination)
    except (OSError, zipfile.BadZipFile):
        blockers.append("aura_provider_output_zip_invalid")
    result_path = destination / "adp_aura_author_smoke_result.json"
    execution = _read_json(result_path)
    if not execution:
        blockers.append("aura_provider_result_missing")
    return {
        "status": "completed" if not blockers else "blocked",
        "result_path": str(result_path),
        "execution": execution,
        "blockers": sorted(set(blockers)),
    }


@contextmanager
def _authority_environment():
    names = (*_VAST_MUTATION_ENV, _VAST_SINGLE_ATTEMPT_ENV)
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in _VAST_MUTATION_ENV:
            os.environ[name] = "1"
        os.environ[_VAST_SINGLE_ATTEMPT_ENV] = "0"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_aura_author_smoke_vast(
    *,
    job_dir: str | Path,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    prepared_bundle: Mapping[str, Any],
    max_hourly_rate_usd: float = 1.5,
    hard_cap_usd: float = 5.0,
    hard_ttl_seconds: int = 10_800,
    public_image: str = DEFAULT_IMAGE,
    machine_avoidlist_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one attempt and require both Vast and staged-object provider zero."""

    job = Path(job_dir).expanduser().resolve()
    ensure_dir(job)
    bundle = dict(prepared_bundle)
    bundle_path = Path(str(bundle.get("bundle_path") or "")).resolve()
    if public_image != DEFAULT_IMAGE:
        raise ValueError("adp_aura_container_image_not_frozen")
    if (
        bundle.get("status") != "ready"
        or not bundle_path.is_file()
        or _sha256(bundle_path) != bundle.get("bundle_sha256")
    ):
        raise ValueError("adp_aura_prepared_bundle_binding_invalid")
    if not execute:
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "dry_run_ready",
            "bundle": bundle,
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "blockers": [],
        }
        write_json(job / "adp_aura_author_smoke_vast_result.json", result)
        return result
    if paid_resource_admission_grant is None:
        raise ValueError("adp_aura_paid_resource_admission_grant_missing")
    remaining_minutes = _remaining_minutes(
        job=job,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        max_hourly_rate_usd=max_hourly_rate_usd,
    )
    if remaining_minutes < 90:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "blockers": ["adp_aura_budget_below_minimum_live_window"],
        }
    staging_dir = job / "object_store_staging"
    staging = stage_wam_provider_bundle_object_store(
        job_dir=staging_dir,
        bundle_path=str(bundle_path),
        key_prefix=DEFAULT_KEY_PREFIX,
        expiration_seconds=max(hard_ttl_seconds + 1800, 18_000),
    )
    if staging.get("status") != "completed":
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "blockers": staging.get("blockers") or ["aura_object_store_staging_blocked"],
        }
    provider_run = job / "vast_provider_run"
    output_zip = provider_run / "vast_provider_runtime_output.zip"
    adapter: dict[str, Any] = {}
    try:
        with _authority_environment():
            adapter = run_vast_provider_adapter(
                job_dir=provider_run,
                mode="live-startup-probe",
                allow_vast_api_call=True,
                allow_instance_launch=True,
                max_hourly_rate=max_hourly_rate_usd,
                target_spend_usd=hard_cap_usd,
                hard_cap_usd=hard_cap_usd,
                max_live_minutes=remaining_minutes,
                session_max_live_minutes=hard_ttl_seconds // 60,
                public_image=public_image,
                isaac_image=public_image,
                ngc_image_login_mode="never",
                provider_bundle=bundle_path,
                provider_bundle_url=(staging_dir / "provider_bundle_url.txt").read_text().strip(),
                provider_output_put_url=(
                    staging_dir / "provider_output_put_url.txt"
                ).read_text().strip(),
                provider_output_get_url=(
                    staging_dir / "provider_output_get_url.txt"
                ).read_text().strip(),
                provider_runtime_output_zip=output_zip,
                enable_isaac_smoke=False,
                enable_blueprint_bundle=True,
                provider_bundle_kind=PROVIDER_BUNDLE_KIND,
                vast_launch_mode="ssh_direct",
                allow_cold_isaac_image_pull=False,
                disk_gb=128,
                min_gpu_ram_mb=24_000,
                poll_interval_seconds=15,
                startup_timeout_seconds=remaining_minutes * 60,
                heartbeat_no_progress_seconds=2700,
                session_budget_ledger_path=job / "adp_aura_vast_session_budget.json",
                verify_staging_urls=True,
                require_known_supported_isaac_driver=False,
                preferred_gpu_keywords=("RTX 4090", "RTX A6000", "L40S", "A100"),
                prefer_isaac_rt=False,
                machine_avoidlist_path=machine_avoidlist_path,
                instance_label_prefix="blueprint-adp-aura-smoke-",
                forward_hf_token=False,
                paid_resource_admission_grant=paid_resource_admission_grant,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        adapter = {
            "status": "blocked",
            "blockers": [f"adp_aura_vast_adapter_failed:{type(exc).__name__}"],
            "raw_secret_values_recorded": False,
        }
    finally:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
    extracted = _extract(output_zip, job / "immutable_execution")
    execution = dict(extracted.get("execution") or {})
    teardown = _read_json(provider_run / "vast_teardown_manifest.json")
    blockers = list(adapter.get("blockers") or []) + list(extracted.get("blockers") or [])
    if execution.get("status") != "completed":
        blockers.extend(execution.get("blockers") or ["aura_inpaint_init_not_completed"])
    if teardown.get("continuing_spend_from_this_run") is not False:
        blockers.append("aura_vast_provider_zero_not_proven")
    if cleanup.get("all_objects_absent") is not True:
        blockers.append("aura_object_store_provider_zero_not_proven")
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "completed" if not blockers else "blocked",
        "bundle_sha256": bundle["bundle_sha256"],
        "execution_result_path": extracted.get("result_path"),
        "adapter_result_path": str(provider_run / "vast_provider_adapter_result.json"),
        "teardown_manifest_path": str(provider_run / "vast_teardown_manifest.json"),
        "estimated_cost_usd": adapter.get("estimated_cost_usd"),
        "hard_cap_usd": hard_cap_usd,
        "hard_ttl_seconds": hard_ttl_seconds,
        "retry_cap": 0,
        "continuing_spend_from_this_run": teardown.get("continuing_spend_from_this_run"),
        "all_staged_objects_absent": cleanup.get("all_objects_absent"),
        "blockers": sorted(set(str(item) for item in blockers if str(item))),
        "raw_secret_values_recorded": False,
    }
    write_json(job / "adp_aura_author_smoke_vast_result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialize-author-data", action="store_true")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--aura-root")
    parser.add_argument("--sam2-root")
    parser.add_argument("--wonderworld-root")
    parser.add_argument("--prerequisite-receipt", required=True)
    parser.add_argument("--author-data-root")
    parser.add_argument("--author-data-receipt")
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args(argv)
    if args.materialize_author_data:
        receipt = materialize_aura_author_data(
            prerequisite_receipt_path=args.prerequisite_receipt,
            output_root=args.job_dir,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    missing = [
        name
        for name in (
            "aura_root",
            "sam2_root",
            "wonderworld_root",
            "author_data_root",
            "author_data_receipt",
        )
        if not getattr(args, name)
    ]
    if missing:
        parser.error("bundle build requires " + ", ".join("--" + name.replace("_", "-") for name in missing))
    receipt = build_aura_author_smoke_vast_bundle(
        repo_root=args.repo_root,
        aura_root=args.aura_root,
        sam2_root=args.sam2_root,
        wonderworld_root=args.wonderworld_root,
        prerequisite_receipt_path=args.prerequisite_receipt,
        author_data_root=args.author_data_root,
        author_data_receipt_path=args.author_data_receipt,
        job_dir=args.job_dir,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt.get("status") == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
