"""Build the locked Python runtime missing from the Isaac provider image.

The exact Isaac Sim 6.0.1 image supplies PIL, NumPy, SciPy, and PyYAML to its
Python launcher, but its USD bindings are not importable until Kit starts.  The
scene-configuration import preflight and Content Agents driver need standalone
USD before that point, in addition to the OpenAI Agents SDK and its Pydantic
closure.  Resolve both roots from ``uv.lock`` before a GPU is rented, download
only hash-bound wheels, and ship the resulting wheelhouse inside the immutable
component package.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tomllib
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from packaging.markers import Marker
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import parse_wheel_filename

from .decision_evidence_contracts import canonical_digest
from .safe_outbound_http import pinned_api_policy, request


SCHEMA_VERSION = "task_evaluation_scene_configuration_python_wheelhouse.v1"
TARGET_PYTHON_VERSION = "3.12"
TARGET_PLATFORM_TAGS = (
    "manylinux_2_17_x86_64",
    "manylinux2014_x86_64",
    "manylinux_2_28_x86_64",
    "manylinux_2_35_x86_64",
)
ROOT_DISTRIBUTIONS = ("openai-agents", "usd-core")
MANIFEST_NAME = f"{SCHEMA_VERSION}.json"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_WHEEL_BYTES = 256 * 1024**2
_SUPPORTED_TAG_ORDER = tuple(
    dict.fromkeys(
        [
            *cpython_tags(
                python_version=(3, 12), platforms=list(TARGET_PLATFORM_TAGS)
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
_SUPPORTED_TAG_RANK = {
    tag: index for index, tag in enumerate(_SUPPORTED_TAG_ORDER)
}
Downloader = Callable[..., bytes]


def _normalize(value: object) -> str:
    return re.sub(r"[-_.]+", "-", str(value or "").strip().lower())


def _marker_environment() -> dict[str, str]:
    return {
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


def _marker_applies(value: object) -> bool:
    return not value or Marker(str(value)).evaluate(_marker_environment())


def _package_applies(package: Mapping[str, Any]) -> bool:
    markers = package.get("resolution-markers")
    if not markers:
        return True
    return isinstance(markers, list) and any(_marker_applies(row) for row in markers)


def _wheel_rank(filename: str) -> int | None:
    try:
        _name, _version, _build, tags = parse_wheel_filename(filename)
    except (TypeError, ValueError):
        return None
    ranks = [_SUPPORTED_TAG_RANK[tag] for tag in tags if tag in _SUPPORTED_TAGS]
    return min(ranks) if ranks else None


def _select_wheel(
    wheels: object, *, distribution: str
) -> tuple[str, Mapping[str, Any]]:
    compatible: list[tuple[int, str, Mapping[str, Any]]] = []
    for wheel in wheels if isinstance(wheels, list) else []:
        if not isinstance(wheel, Mapping):
            continue
        filename = Path(
            urllib.parse.urlparse(str(wheel.get("url") or "")).path
        ).name
        rank = _wheel_rank(filename)
        if rank is not None:
            compatible.append((rank, filename, wheel))
    compatible.sort(key=lambda row: (row[0], row[1]))
    if not compatible or sum(
        row[0] == compatible[0][0] for row in compatible
    ) != 1:
        raise ValueError(
            "scene_configuration_python_wheel_ambiguous:" + distribution
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
        raise RuntimeError(
            "scene_configuration_python_wheel_download_http_status_invalid"
        )
    return response.body


def plan_scene_configuration_python_wheelhouse(
    lock_bytes: bytes,
    *,
    root_distributions: Sequence[str] = ROOT_DISTRIBUTIONS,
) -> dict[str, Any]:
    """Derive the exact CPython 3.12 Linux closure without network access."""

    lock = tomllib.loads(lock_bytes.decode("utf-8"))
    packages = lock.get("package") if isinstance(lock, Mapping) else None
    by_name: dict[str, list[Mapping[str, Any]]] = {}
    for package in packages if isinstance(packages, list) else []:
        if isinstance(package, Mapping) and _package_applies(package):
            by_name.setdefault(_normalize(package.get("name")), []).append(package)
    selected: dict[str, Mapping[str, Any]] = {}
    pending: list[tuple[str, str | None]] = [
        (str(name), None) for name in root_distributions
    ]
    while pending:
        unresolved, version = pending.pop()
        name = _normalize(unresolved)
        if not name:
            raise ValueError("scene_configuration_python_dependency_name_invalid")
        if name in selected:
            if version and str(selected[name].get("version")) != version:
                raise ValueError(
                    "scene_configuration_python_dependency_version_conflict"
                )
            continue
        candidates = by_name.get(name, [])
        if version:
            candidates = [
                row for row in candidates if str(row.get("version")) == version
            ]
        if len(candidates) != 1:
            raise ValueError(
                "scene_configuration_python_locked_package_ambiguous:" + name
            )
        package = candidates[0]
        selected[name] = package
        dependencies = package.get("dependencies")
        for dependency in dependencies if isinstance(dependencies, list) else []:
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
    filenames: set[str] = set()
    for name, package in sorted(selected.items()):
        filename, wheel = _select_wheel(package.get("wheels"), distribution=name)
        digest = str(wheel.get("hash") or "").removeprefix("sha256:")
        size = wheel.get("size")
        url = str(wheel.get("url") or "")
        if (
            filename in filenames
            or _HEX64.fullmatch(digest) is None
            or type(size) is not int
            or size <= 0
            or size > _MAX_WHEEL_BYTES
            or not url.startswith("https://files.pythonhosted.org/")
        ):
            raise ValueError(
                "scene_configuration_python_locked_wheel_metadata_invalid"
            )
        filenames.add(filename)
        wheels.append(
            {
                "distribution": name,
                "version": str(package.get("version") or ""),
                "filename": filename,
                "sha256": "sha256:" + digest,
                "size_bytes": size,
                "url": url,
            }
        )
    return {"requirements": requirements, "wheels": wheels}


def build_scene_configuration_python_wheelhouse(
    *,
    lockfile_path: str | Path,
    output_root: str | Path,
    downloader: Downloader = _download,
) -> dict[str, Any]:
    """Download and verify the exact closure before provider allocation."""

    lockfile = Path(lockfile_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise ValueError("scene_configuration_python_wheelhouse_output_exists")
    output.mkdir(parents=True, mode=0o700)
    wheels_root = output / "wheels"
    wheels_root.mkdir(mode=0o700)
    try:
        lock_bytes = lockfile.read_bytes()
        plan = plan_scene_configuration_python_wheelhouse(lock_bytes)
        sealed_rows: list[dict[str, Any]] = []
        for row in plan["wheels"]:
            try:
                body = downloader(
                    row["url"], maximum_bytes=int(row["size_bytes"]) + 1
                )
            except Exception as exc:
                raise RuntimeError(
                    "scene_configuration_python_wheel_download_failed:"
                    + str(row["distribution"])
                ) from exc
            if (
                len(body) != row["size_bytes"]
                or "sha256:" + hashlib.sha256(body).hexdigest()
                != row["sha256"]
            ):
                raise ValueError(
                    "scene_configuration_python_wheel_download_digest_mismatch:"
                    + str(row["distribution"])
                )
            (wheels_root / row["filename"]).write_bytes(body)
            sealed_rows.append({key: value for key, value in row.items() if key != "url"})
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "ready",
            "python_version": TARGET_PYTHON_VERSION,
            "implementation": "cpython",
            "platform": "linux-x86_64",
            "platform_tags": list(TARGET_PLATFORM_TAGS),
            "lockfile_sha256": "sha256:" + hashlib.sha256(lock_bytes).hexdigest(),
            "root_distributions": list(ROOT_DISTRIBUTIONS),
            "requirements": plan["requirements"],
            "wheels": sealed_rows,
            "sdists_allowed": False,
            "provider_network_install_required": False,
            "manifest_digest": "",
        }
        manifest["manifest_digest"] = canonical_digest(
            manifest, digest_field="manifest_digest"
        )
        (output / MANIFEST_NAME).write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return manifest
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def validate_scene_configuration_python_wheelhouse(
    *, root: str | Path
) -> dict[str, Any]:
    """Reopen every sealed wheel byte before it can enter a provider bundle."""

    wheelhouse = Path(root).expanduser().resolve()
    manifest_path = wheelhouse / MANIFEST_NAME
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "scene_configuration_python_wheelhouse_manifest_invalid"
        ) from exc
    if not isinstance(value, Mapping):
        raise ValueError("scene_configuration_python_wheelhouse_manifest_invalid")
    manifest = dict(value)
    rows = manifest.get("wheels")
    if (
        wheelhouse.is_symlink()
        or not wheelhouse.is_dir()
        or manifest_path.is_symlink()
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "ready"
        or manifest.get("python_version") != TARGET_PYTHON_VERSION
        or manifest.get("implementation") != "cpython"
        or manifest.get("platform") != "linux-x86_64"
        or manifest.get("root_distributions") != list(ROOT_DISTRIBUTIONS)
        or manifest.get("sdists_allowed") is not False
        or manifest.get("provider_network_install_required") is not False
        or manifest.get("manifest_digest")
        != canonical_digest(manifest, digest_field="manifest_digest")
        or not isinstance(rows, list)
        or not rows
    ):
        raise ValueError("scene_configuration_python_wheelhouse_manifest_invalid")
    expected: set[str] = set()
    wheels_root = wheelhouse / "wheels"
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(
                "scene_configuration_python_wheelhouse_inventory_invalid"
            )
        filename = str(row.get("filename") or "")
        path = wheels_root / filename
        if (
            not filename
            or Path(filename).name != filename
            or filename in expected
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            != row.get("sha256")
        ):
            raise ValueError(
                "scene_configuration_python_wheelhouse_inventory_invalid"
            )
        expected.add(filename)
    observed = {
        path.name
        for path in wheels_root.iterdir()
        if path.is_file() and not path.is_symlink()
    } if wheels_root.is_dir() and not wheels_root.is_symlink() else set()
    if observed != expected:
        raise ValueError("scene_configuration_python_wheelhouse_inventory_incomplete")
    return manifest


__all__ = [
    "MANIFEST_NAME",
    "ROOT_DISTRIBUTIONS",
    "SCHEMA_VERSION",
    "TARGET_PLATFORM_TAGS",
    "TARGET_PYTHON_VERSION",
    "build_scene_configuration_python_wheelhouse",
    "plan_scene_configuration_python_wheelhouse",
    "validate_scene_configuration_python_wheelhouse",
]
