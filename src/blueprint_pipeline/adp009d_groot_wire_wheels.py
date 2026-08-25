"""Materialize the exact GR00T wire wheels without mutating Isaac.

The policy episode imports three small codec distributions from a staged
directory beside :mod:`groot_n17_wire_client`.  Version constraints are not an
immutable dependency identity: the provider would still resolve current index
state during a paid run.  This worker downloads three exact publisher wheel
objects, verifies their size and SHA-256 before installation, and makes uv
install only those local bytes with index/network access disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

try:  # flat provider-bundle layout
    from decision_evidence_contracts import canonical_digest
except ModuleNotFoundError:  # repository package
    from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "adp009d_groot_wire_wheel_materialization.v1"
SOURCE_ORIGIN = "https://files.pythonhosted.org"
STAGED_WIRE_DEPS_DIRNAME = "groot_wire_deps"
WHEEL_DOWNLOAD_DIRNAME = "groot_wire_wheels"
RECEIPT_FILENAME = "adp009d_groot_wire_wheel_materialization.v1.json"

# Exact PyPI publisher objects for Linux CPython 3.12 x86_64, the interpreter
# and platform sealed by the digest-pinned Isaac 6.0.1 container.  URLs, sizes,
# and SHA-256 values come from PyPI's release JSON for each exact version.
GROOT_WIRE_WHEEL_ARTIFACTS: tuple[Mapping[str, Any], ...] = (
    {
        "distribution": "pyzmq",
        "version": "27.0.1",
        "filename": ("pyzmq-27.0.1-cp312-abi3-manylinux_2_26_x86_64.manylinux_2_28_x86_64.whl"),
        "url": (
            "https://files.pythonhosted.org/packages/7e/0a/"
            "2356305c423a975000867de56888b79e44ec2192c690ff93c3109fd78081/"
            "pyzmq-27.0.1-cp312-abi3-manylinux_2_26_x86_64."
            "manylinux_2_28_x86_64.whl"
        ),
        "size_bytes": 839_751,
        "sha256": ("sha256:f5b6133c8d313bde8bd0d123c169d22525300ff164c2189f849de495e1344577"),
    },
    {
        "distribution": "msgpack",
        "version": "1.1.0",
        "filename": ("msgpack-1.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"),
        "url": (
            "https://files.pythonhosted.org/packages/f1/54/"
            "65af8de681fa8255402c80eda2a501ba467921d5a7a028c9c22a2c2eedb5/"
            "msgpack-1.1.0-cp312-cp312-manylinux_2_17_x86_64."
            "manylinux2014_x86_64.whl"
        ),
        "size_bytes": 401_403,
        "sha256": ("sha256:17fb65dd0bec285907f68b15734a993ad3fc94332b5bb21b0435846228de1f39"),
    },
    {
        "distribution": "msgpack-numpy",
        "version": "0.4.8",
        "filename": "msgpack_numpy-0.4.8-py2.py3-none-any.whl",
        "url": (
            "https://files.pythonhosted.org/packages/9b/5d/"
            "f25ac7d4fb77cbd53ddc6d05d833c6bf52b12770a44fa9a447eed470ca9a/"
            "msgpack_numpy-0.4.8-py2.py3-none-any.whl"
        ),
        "size_bytes": 6_919,
        "sha256": ("sha256:773c19d4dfbae1b3c7b791083e2caf66983bb19b40901646f61d8731554ae3da"),
    },
)


def expected_artifact_rows() -> list[dict[str, Any]]:
    """Return a mutable JSON-safe copy of the frozen artifact identities."""

    return [dict(row) for row in GROOT_WIRE_WHEEL_ARTIFACTS]


def expected_artifacts_digest() -> str:
    return canonical_digest({"source_origin": SOURCE_ORIGIN, "artifacts": expected_artifact_rows()})


def _normalized_distribution(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _download_verified(
    artifact: Mapping[str, Any],
    *,
    destination: Path,
    opener: Callable[..., Any],
) -> dict[str, Any]:
    url = str(artifact["url"])
    if not url.startswith(SOURCE_ORIGIN + "/"):
        raise ValueError("groot_wire_wheel_source_origin_invalid")
    temporary = destination.with_name(destination.name + ".part")
    if destination.exists() or temporary.exists():
        raise FileExistsError("groot_wire_wheel_download_overwrite_forbidden")
    request = Request(url, headers={"User-Agent": "blueprint-adp009d-wire-wheel/1"})
    digest = hashlib.sha256()
    size = 0
    try:
        with opener(request, timeout=120) as response, temporary.open("xb") as output:
            final_url = str(response.geturl())
            if final_url != url:
                raise ValueError("groot_wire_wheel_redirect_forbidden")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        observed_sha256 = "sha256:" + digest.hexdigest()
        if size != int(artifact["size_bytes"]) or observed_sha256 != artifact["sha256"]:
            raise ValueError("groot_wire_wheel_identity_mismatch")
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        **dict(artifact),
        "observed_size_bytes": size,
        "observed_sha256": observed_sha256,
        "identity_verified": True,
    }


def _installed_distributions(target: Path) -> dict[str, str]:
    installed: dict[str, str] = {}
    for distribution in metadata.distributions(path=[str(target)]):
        name = _normalized_distribution(str(distribution.metadata.get("Name") or ""))
        if not name or name in installed:
            raise ValueError("groot_wire_staged_distribution_metadata_invalid")
        installed[name] = str(distribution.version)
    return dict(sorted(installed.items()))


def _atomic_write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("groot_wire_wheel_receipt_overwrite_forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("groot_wire_wheel_receipt_temporary_conflict")
    content = json.dumps(dict(receipt), indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != receipt:
        raise RuntimeError("groot_wire_wheel_receipt_readback_mismatch")


def materialize_wire_wheels(
    *,
    uv_executable: Path,
    python_executable: Path,
    runtime_dir: Path,
    output_path: Path,
    opener: Callable[..., Any] = urlopen,
    runner: Callable[..., Any] = subprocess.run,
    artifacts: Sequence[Mapping[str, Any]] = GROOT_WIRE_WHEEL_ARTIFACTS,
) -> dict[str, Any]:
    """Download, verify, install offline, and receipt the exact wheel bytes."""

    runtime = runtime_dir.expanduser().resolve()
    download_root = runtime / WHEEL_DOWNLOAD_DIRNAME
    target = runtime / STAGED_WIRE_DEPS_DIRNAME
    if download_root.exists() or target.exists():
        raise FileExistsError("groot_wire_wheel_runtime_target_not_fresh")
    download_root.mkdir(parents=True)
    observed_artifacts: list[dict[str, Any]] = []
    wheel_paths: list[Path] = []
    for artifact in artifacts:
        destination = download_root / str(artifact["filename"])
        observed_artifacts.append(
            _download_verified(artifact, destination=destination, opener=opener)
        )
        wheel_paths.append(destination)

    command = [
        str(uv_executable),
        "pip",
        "install",
        "--python",
        str(python_executable),
        "--no-deps",
        "--only-binary=:all:",
        "--offline",
        "--target",
        str(target),
        *(str(path) for path in wheel_paths),
    ]
    environment = dict(os.environ)
    environment["UV_NO_INDEX"] = "1"
    runner(command, check=True, env=environment)

    installed = _installed_distributions(target)
    expected_rows = [dict(row) for row in artifacts]
    expected_versions = {
        _normalized_distribution(str(row["distribution"])): str(row["version"]) for row in artifacts
    }
    if installed != dict(sorted(expected_versions.items())):
        raise RuntimeError("groot_wire_staged_distribution_set_mismatch")
    if "numpy" in installed:
        raise RuntimeError("groot_wire_numpy_staged_forbidden")
    observed_identity_rows = [
        {
            "distribution": row["distribution"],
            "version": row["version"],
            "filename": row["filename"],
            "url": row["url"],
            "size_bytes": row["observed_size_bytes"],
            "sha256": row["observed_sha256"],
        }
        for row in observed_artifacts
    ]
    expected_identity_digest = canonical_digest(
        {"source_origin": SOURCE_ORIGIN, "artifacts": expected_rows}
    )
    observed_identity_digest = canonical_digest(
        {"source_origin": SOURCE_ORIGIN, "artifacts": observed_identity_rows}
    )
    if observed_identity_digest != expected_identity_digest:
        raise RuntimeError("groot_wire_observed_artifact_identity_mismatch")
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "verified",
        "source_origin": SOURCE_ORIGIN,
        "expected_artifacts": expected_rows,
        "expected_artifacts_digest": expected_identity_digest,
        "observed_artifacts": observed_artifacts,
        "observed_artifacts_digest": observed_identity_digest,
        "staged_target": str(target),
        "installed_distributions": installed,
        "numpy_distribution_staged": False,
        "isaac_environment_mutated": False,
        "installer_network_access": False,
        "installer_index_access": False,
        "dependency_resolution_allowed": False,
        "blockers": [],
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _atomic_write_receipt(output_path.expanduser().resolve(), receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uv", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt = materialize_wire_wheels(
        uv_executable=Path(args.uv),
        python_executable=Path(args.python),
        runtime_dir=Path(args.runtime_dir),
        output_path=Path(args.output),
    )
    print(
        "BLUEPRINT_ADP009D_GROOT_WIRE_WHEELS_VERIFIED:" + str(receipt["expected_artifacts_digest"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GROOT_WIRE_WHEEL_ARTIFACTS",
    "RECEIPT_FILENAME",
    "SCHEMA_VERSION",
    "SOURCE_ORIGIN",
    "STAGED_WIRE_DEPS_DIRNAME",
    "WHEEL_DOWNLOAD_DIRNAME",
    "expected_artifact_rows",
    "expected_artifacts_digest",
    "materialize_wire_wheels",
]
