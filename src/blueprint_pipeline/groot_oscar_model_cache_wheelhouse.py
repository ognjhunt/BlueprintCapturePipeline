"""Build the locked Linux/Python 3.12 wheelhouse for model-cache preparation."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Mapping

from packaging.markers import Marker
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import parse_wheel_filename

from .common import write_json
from .safe_outbound_http import pinned_api_policy, request


SCHEMA_VERSION = "blueprint_python_wheelhouse.v1"
TARGET_PYTHON_VERSION = "3.12"
TARGET_PLATFORM_TAGS = ("manylinux2014_x86_64", "manylinux_2_17_x86_64")
ROOT_DISTRIBUTIONS = ("boto3", "huggingface-hub")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_SUPPORTED_TAG_ORDER = tuple(
    dict.fromkeys(
        [
            *cpython_tags(
                python_version=(3, 12),
                platforms=list(TARGET_PLATFORM_TAGS),
            ),
            *compatible_tags(
                python_version=(3, 12),
                interpreter="cp312",
                platforms=list(TARGET_PLATFORM_TAGS),
            ),
        ]
    )
)
_SUPPORTED_TAGS = frozenset(_SUPPORTED_TAG_ORDER)
_SUPPORTED_TAG_RANK = {tag: index for index, tag in enumerate(_SUPPORTED_TAG_ORDER)}


def _normalize(name: object) -> str:
    return re.sub(r"[-_.]+", "-", str(name or "").strip().lower())


def _marker_applies(marker: object) -> bool:
    if not marker:
        return True
    environment = {
        "implementation_name": "cpython",
        "implementation_version": "3.12.0",
        "os_name": "posix",
        "platform_machine": "x86_64",
        "platform_python_implementation": "CPython",
        "platform_release": "",
        "platform_system": "Linux",
        "platform_version": "",
        "python_full_version": "3.12.0",
        "python_version": TARGET_PYTHON_VERSION,
        "sys_platform": "linux",
        "extra": "",
    }
    return Marker(str(marker)).evaluate(environment)


def _wheel_compatible(filename: str) -> bool:
    try:
        _name, _version, _build, tags = parse_wheel_filename(filename)
    except (ValueError, TypeError):
        return False
    return bool(tags & _SUPPORTED_TAGS)


def _wheel_rank(filename: str) -> int | None:
    try:
        _name, _version, _build, tags = parse_wheel_filename(filename)
    except (ValueError, TypeError):
        return None
    ranks = [_SUPPORTED_TAG_RANK[tag] for tag in tags if tag in _SUPPORTED_TAG_RANK]
    return min(ranks) if ranks else None


def _select_locked_wheel(
    wheels: object, *, distribution: str
) -> tuple[str, Mapping[str, Any]]:
    rows = wheels if isinstance(wheels, list) else []
    compatible: list[tuple[int, str, Mapping[str, Any]]] = []
    for wheel in rows:
        if not isinstance(wheel, Mapping):
            continue
        filename = Path(urllib.parse.urlparse(str(wheel.get("url") or "")).path).name
        rank = _wheel_rank(filename)
        if rank is not None:
            compatible.append((rank, filename, wheel))
    compatible.sort(key=lambda item: (item[0], item[1]))
    if not compatible or sum(row[0] == compatible[0][0] for row in compatible) != 1:
        raise ValueError(
            "model_cache_wheelhouse_target_wheel_ambiguous:" + distribution
        )
    _rank, filename, wheel = compatible[0]
    return filename, wheel


def _download(url: str, *, maximum_bytes: int) -> bytes:
    response = request(
        url,
        policy=pinned_api_policy(
            "https://files.pythonhosted.org", max_response_bytes=maximum_bytes
        ),
        timeout_seconds=600,
        max_response_bytes=maximum_bytes,
    )
    if response.status != 200:
        raise RuntimeError("model_cache_wheel_download_http_status_invalid")
    return response.body


def plan_model_cache_wheelhouse(lock_bytes: bytes) -> dict[str, Any]:
    """Derive the canonical target closure and wheel metadata without I/O."""

    lock = tomllib.loads(lock_bytes.decode("utf-8"))
    packages = lock.get("package") if isinstance(lock, Mapping) else None
    packages = packages if isinstance(packages, list) else []
    by_name: dict[str, list[Mapping[str, Any]]] = {}
    for package in packages:
        if isinstance(package, Mapping):
            by_name.setdefault(_normalize(package.get("name")), []).append(package)
    selected: dict[str, Mapping[str, Any]] = {}
    pending: list[tuple[str, str | None]] = [
        (name, None) for name in ROOT_DISTRIBUTIONS
    ]
    while pending:
        name, version = pending.pop()
        normalized = _normalize(name)
        if normalized in selected:
            if version and str(selected[normalized].get("version")) != version:
                raise ValueError("model_cache_wheelhouse_dependency_version_conflict")
            continue
        candidates = by_name.get(normalized, [])
        if version:
            candidates = [
                row for row in candidates if str(row.get("version")) == version
            ]
        if len(candidates) != 1:
            raise ValueError(
                "model_cache_wheelhouse_locked_package_ambiguous:" + normalized
            )
        package = candidates[0]
        selected[normalized] = package
        dependencies = package.get("dependencies")
        dependencies = dependencies if isinstance(dependencies, list) else []
        for dependency in dependencies:
            if not isinstance(dependency, Mapping) or not _marker_applies(
                dependency.get("marker")
            ):
                continue
            pending.append(
                (
                    str(dependency.get("name") or ""),
                    str(dependency.get("version"))
                    if dependency.get("version")
                    else None,
                )
            )
    requirements = [
        {"name": name, "version": str(package.get("version") or "")}
        for name, package in sorted(selected.items())
    ]
    wheels: list[dict[str, Any]] = []
    observed_filenames: set[str] = set()
    for name, package in sorted(selected.items()):
        filename, wheel = _select_locked_wheel(
            package.get("wheels"), distribution=name
        )
        if filename in observed_filenames:
            raise ValueError("model_cache_wheelhouse_duplicate_filename")
        observed_filenames.add(filename)
        digest = str(wheel.get("hash") or "").removeprefix("sha256:")
        size = wheel.get("size")
        url = str(wheel.get("url") or "")
        if (
            _HEX64.fullmatch(digest) is None
            or type(size) is not int
            or size <= 0
            or not url.startswith("https://files.pythonhosted.org/")
        ):
            raise ValueError("model_cache_wheelhouse_locked_wheel_metadata_invalid")
        wheels.append(
            {
                "distribution": name,
                "version": str(package.get("version") or ""),
                "filename": filename,
                "sha256": digest,
                "bytes": size,
                "url": url,
            }
        )
    closure_bytes = (
        json.dumps(requirements, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    return {
        "requirements": requirements,
        "requirements_closure_sha256": hashlib.sha256(closure_bytes).hexdigest(),
        "wheels": wheels,
    }


def build_model_cache_wheelhouse(
    *,
    lockfile_path: str | Path,
    output_dir: str | Path,
    downloader: Callable[..., bytes] = _download,
) -> dict[str, Any]:
    """Materialize the exact target closure from uv.lock without resolving anew."""

    lockfile = Path(lockfile_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise ValueError("model_cache_wheelhouse_output_already_exists")
    output.mkdir(parents=True, mode=0o700)
    wheel_directory = output / "wheels"
    wheel_directory.mkdir(mode=0o700)
    lock_bytes = lockfile.read_bytes()
    plan = plan_model_cache_wheelhouse(lock_bytes)
    requirements = plan["requirements"]
    closure_bytes = (
        json.dumps(requirements, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    wheel_rows: list[dict[str, Any]] = []
    for planned in plan["wheels"]:
        name = planned["distribution"]
        filename = planned["filename"]
        digest = planned["sha256"]
        size = planned["bytes"]
        url = planned["url"]
        try:
            body = downloader(url, maximum_bytes=size + 1)
        except Exception:
            write_json(
                output / "wheelhouse_manifest.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "blocked",
                    "blockers": ["model_cache_wheelhouse_download_failed"],
                    "failed_distribution": name,
                    "raw_secret_values_recorded": False,
                },
            )
            raise
        if len(body) != size or hashlib.sha256(body).hexdigest() != digest:
            write_json(
                output / "wheelhouse_manifest.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "blocked",
                    "blockers": ["model_cache_wheelhouse_download_digest_mismatch"],
                    "failed_distribution": name,
                    "raw_secret_values_recorded": False,
                },
            )
            raise ValueError("model_cache_wheelhouse_download_digest_mismatch")
        destination = wheel_directory / filename
        destination.write_bytes(body)
        wheel_rows.append(
            {
                "distribution": name,
                "version": planned["version"],
                "filename": filename,
                "sha256": digest,
                "bytes": size,
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "python_version": TARGET_PYTHON_VERSION,
        "implementation": "cpython",
        "platform_tags": list(TARGET_PLATFORM_TAGS),
        "lockfile_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "requirements": requirements,
        "requirements_closure_sha256": hashlib.sha256(closure_bytes).hexdigest(),
        "wheels": wheel_rows,
        "sdists_allowed": False,
        "network_resolution_performed": False,
        "raw_secret_values_recorded": False,
        "wheelhouse_path": str(wheel_directory),
        "manifest_path": str(output / "wheelhouse_manifest.json"),
    }
    write_json(output / "wheelhouse_manifest.json", manifest)
    return manifest
