#!/usr/bin/env python3
"""Verify external GR00T + OSCAR foundation inputs before paid builds.

The static check binds this verifier to the canonical Dockerfile and model-cache
contract.  ``--live`` performs only read-only network requests and downloads no
model weights or container layers.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, cast


ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop"
FOUNDATION = IMAGE_ROOT / "Foundation.Dockerfile"
MODEL_CACHE = ROOT / "src/blueprint_pipeline/groot_oscar_model_cache.py"
ISAAC_ASSET_MANIFEST = IMAGE_ROOT / "isaac_6_g1_assets.sha256"
UV_BOOTSTRAP = IMAGE_ROOT / "requirements_uv_bootstrap.txt"
OSCAR_FOUNDATION_LOCK = IMAGE_ROOT / "requirements_oscar_foundation.lock"
ROBOT_RUNTIME_REQUIREMENTS = IMAGE_ROOT / "requirements_robot_runtime.txt"
THIN_REMOTE_BUILD_PACKET = (
    ROOT / "src/blueprint_pipeline/groot_oscar_thin_remote_build_packet.py"
)

SCHEMA = "groot_oscar_live_prerequisites.v1"
CUDA_REPOSITORY = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64"
ARTIFACTS = (
    (
        f"{CUDA_REPOSITORY}/cuda-keyring_1.1-1_all.deb",
        "d2a6b11c096396d868758b86dab1823b25e14d70333f1dfa74da5ddaf6a06dba",
    ),
    (
        "https://github.com/casey/just/releases/download/1.43.0/"
        "just-1.43.0-x86_64-unknown-linux-musl.tar.gz",
        "a1bc93654f31669fd964ea3011a5e5e9676b9b6f8adcd762606e5140632ea72d",
    ),
    (
        "https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/"
        "onnxruntime-linux-x64-1.16.3.tgz",
        "b072f989d6315ac0e22dcb4771b083c5156d974a3496ac3504c77f4062eb248e",
    ),
    (
        "https://files.pythonhosted.org/packages/71/a9/"
        "2735cc9dc39457c9cf64d1ce2ba5a9a8ecbb103d0fb64b052bf33ba3d669/"
        "uv-0.10.7-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "89de2504407dcf04aece914c6ca3b9d8e60cf9ff39a13031c1df1f7c040cea81",
    ),
    (
        "https://github.com/anchore/syft/releases/download/v1.44.0/"
        "syft_1.44.0_linux_amd64.tar.gz",
        "0e91737aee2b5baf1d255b959630194a302335d848ff97bb07921eb6205b5f5a",
    ),
)
SOURCE_PINS = (
    (
        "NVIDIA/Isaac-GR00T",
        "e5749287857afd97b78f1147166137de29746392",
    ),
    (
        "wuzy2115/oscar-public",
        "4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb",
    ),
    (
        "NVlabs/GR00T-WholeBodyControl",
        "6d8e931b9b10a4db2d8e7aba3ad6d5da3529ff3b",
    ),
)
SOURCE_REQUIRED_FILES = {
    "NVIDIA/Isaac-GR00T": (
        "pyproject.toml",
        "uv.lock",
        "gr00t/policy/gr00t_policy.py",
    ),
    "wuzy2115/oscar-public": (
        "requirements_minimal.txt",
        "inference/inference_oscar.py",
    ),
    "NVlabs/GR00T-WholeBodyControl": (
        "gear_sonic_deploy/scripts/setup_env.sh",
        "gear_sonic_deploy/.justfile",
    ),
}
TENSORRT_VERSION = "10.4.0.26-1+cuda12.6"
CUDA_CUDART_VERSION = "12.6.77-1"
TENSORRT_PACKAGES = (
    "libnvinfer-headers-dev",
    "libnvinfer-headers-plugin-dev",
    "libnvinfer10",
    "libnvinfer-plugin10",
    "libnvonnxparsers10",
    "libnvinfer-dev",
    "libnvinfer-plugin-dev",
    "libnvonnxparsers-dev",
)
ISAAC_ASSET_BASE_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/"
    "Isaac/6.0/Isaac/Robots/Unitree/G1/"
)
ISAAC_BASE_REPOSITORY = "nvidia/isaac-sim"
ISAAC_BASE_DIGEST = "68735a60b6c15c85e0dd0098570c6d2cc79e928f2d068ce2790aa43284ac165d"
OCI_MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)
Fetch = Callable[[str], bytes]
HeadSize = Callable[[str], int]
ALLOWED_HTTPS_HOSTS = frozenset(
    {
        "api.github.com",
        "developer.download.nvidia.com",
        "files.pythonhosted.org",
        "github.com",
        "huggingface.co",
        "nvcr.io",
        "omniverse-content-production.s3-us-west-2.amazonaws.com",
        "raw.githubusercontent.com",
        "release-assets.githubusercontent.com",
    }
)
AuthorizedFetch = Callable[[str, str], bytes]


def _assigned_literal(module: Path, name: str) -> object:
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    assignments: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and node.value is not None:
                    assignments[target.id] = node.value

    def evaluate(node: ast.expr) -> object:
        if isinstance(node, ast.Name) and node.id in assignments:
            return evaluate(assignments[node.id])
        if isinstance(node, ast.Tuple):
            return tuple(evaluate(item) for item in node.elts)
        if isinstance(node, ast.Dict):
            return {
                evaluate(key): evaluate(value)
                for key, value in zip(node.keys, node.values)
                if key is not None
            }
        return ast.literal_eval(node)

    if name not in assignments:
        raise ValueError(f"missing_literal:{name}")
    return evaluate(assignments[name])


def _allowed_https_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in ALLOWED_HTTPS_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
    ):
        raise ValueError("prerequisite_url_not_allowlisted")
    return url


class _AllowlistedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-apply the outbound allowlist to every redirect target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return super().redirect_request(
            req, fp, code, msg, headers, _allowed_https_url(newurl)
        )


def _open_allowlisted(request: urllib.request.Request, *, timeout: int):
    _allowed_https_url(request.full_url)
    return urllib.request.build_opener(_AllowlistedRedirectHandler()).open(
        request, timeout=timeout
    )


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(
        _allowed_https_url(url),
        headers={
            "Accept": "application/vnd.github+json, application/json",
            "User-Agent": "blueprint-foundation-prerequisite-verifier/1",
        },
    )
    with _open_allowlisted(request, timeout=60) as response:
        return response.read()


def _head_size(url: str) -> int:
    request = urllib.request.Request(
        _allowed_https_url(url),
        method="HEAD",
        headers={"User-Agent": "blueprint-foundation-prerequisite-verifier/1"},
    )
    with _open_allowlisted(request, timeout=30) as response:
        content_length = response.headers.get("Content-Length")
    if content_length is None:
        raise ValueError("content_length_absent")
    return int(content_length)


def _authorized_fetch(url: str, token: str) -> bytes:
    request = urllib.request.Request(
        _allowed_https_url(url),
        headers={
            "Accept": OCI_MANIFEST_ACCEPT,
            "Authorization": f"Bearer {token}",
            "User-Agent": "blueprint-foundation-prerequisite-verifier/1",
        },
    )
    with _open_allowlisted(request, timeout=60) as response:
        return response.read()


def _asset_rows() -> list[tuple[str, int, str]]:
    rows: list[tuple[str, int, str]] = []
    for line in ISAAC_ASSET_MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        digest, size, relative_path = line.split(maxsplit=2)
        rows.append((digest, int(size), relative_path))
    return rows


def verify_static() -> list[str]:
    blockers: list[str] = []
    dockerfile = FOUNDATION.read_text(encoding="utf-8")
    bootstrap_contract = "\n".join(
        (
            dockerfile,
            UV_BOOTSTRAP.read_text(encoding="utf-8"),
            THIN_REMOTE_BUILD_PACKET.read_text(encoding="utf-8"),
        )
    )
    for url, digest in ARTIFACTS:
        if url not in bootstrap_contract or digest not in bootstrap_contract:
            blockers.append(f"dockerfile_artifact_pin_mismatch:{url}")
    for repository, revision in SOURCE_PINS:
        if revision not in dockerfile or repository.split("/", 1)[1] not in dockerfile:
            blockers.append(f"dockerfile_source_pin_mismatch:{repository}")
    if f"ARG TENSORRT_VERSION={TENSORRT_VERSION}" not in dockerfile:
        blockers.append("dockerfile_tensorrt_version_mismatch")
    if f"ARG CUDA_CUDART_VERSION={CUDA_CUDART_VERSION}" not in dockerfile:
        blockers.append("dockerfile_cuda_cudart_version_mismatch")
    if "cuda-cudart-12-6=${CUDA_CUDART_VERSION}" not in dockerfile:
        blockers.append("dockerfile_cuda_cudart_package_unpinned")
    if "BLUEPRINT_GROOT_OSCAR_REQUIRED_CUDA_VERSION=12.8" not in dockerfile:
        blockers.append("dockerfile_host_cuda_version_must_cover_torch_cu128")
    for package in TENSORRT_PACKAGES:
        if f"{package}=${{TENSORRT_VERSION}}" not in dockerfile:
            blockers.append(f"dockerfile_tensorrt_package_unpinned:{package}")
    if ISAAC_ASSET_BASE_URL not in dockerfile:
        blockers.append("dockerfile_isaac_asset_base_url_mismatch")
    expected_base = f"nvcr.io/{ISAAC_BASE_REPOSITORY}:6.0.0@sha256:{ISAAC_BASE_DIGEST}"
    if f"ARG ISAAC_SIM_BASE_IMAGE={expected_base}" not in dockerfile:
        blockers.append("dockerfile_isaac_base_image_pin_mismatch")

    lock_text = OSCAR_FOUNDATION_LOCK.read_text(encoding="utf-8")
    runtime_digest = hashlib.sha256(ROBOT_RUNTIME_REQUIREMENTS.read_bytes()).hexdigest()
    if (
        f"# blueprint-input-sha256 requirements-robot-runtime {runtime_digest}"
        not in lock_text
    ):
        blockers.append("oscar_foundation_lock_runtime_input_digest_stale")
    if "# blueprint-target cpython-3.10 linux-x86_64 torch-cu128 uv-0.10.7" not in lock_text:
        blockers.append("oscar_foundation_lock_target_missing")

    try:
        assets = _asset_rows()
    except (OSError, ValueError) as exc:
        blockers.append(f"isaac_asset_manifest_unreadable:{exc}")
    else:
        if not assets:
            blockers.append("isaac_asset_manifest_empty")
        for digest, size, relative_path in assets:
            if not re.fullmatch(r"[0-9a-f]{64}", digest) or size <= 0:
                blockers.append(f"isaac_asset_manifest_invalid:{relative_path}")

    model_pins = cast(
        tuple[tuple[str, str, str], ...], _assigned_literal(MODEL_CACHE, "MODEL_PINS")
    )
    required_files = cast(
        dict[str, tuple[str, ...]],
        _assigned_literal(MODEL_CACHE, "REQUIRED_MODEL_FILES"),
    )
    if {row[0] for row in model_pins} != set(required_files):
        blockers.append("model_pin_required_file_coverage_mismatch")
    for name, _repository, revision in model_pins:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            blockers.append(f"model_revision_not_immutable:{name}")
        if not required_files.get(name):
            blockers.append(f"model_required_files_empty:{name}")
    return sorted(set(blockers))


def _json_object(payload: bytes) -> Mapping[str, Any]:
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("response_not_object")
    return cast(Mapping[str, Any], decoded)


def summarize_required_model_metadata(
    siblings: object, required_paths: tuple[str, ...]
) -> tuple[dict[str, Mapping[str, Any]], list[str], list[str], int]:
    """Return exact required-file availability and sizing from HF metadata."""

    rows = siblings if isinstance(siblings, list) else []
    available = {
        str(row.get("rfilename")): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("rfilename"), str)
    }
    missing = sorted(set(required_paths) - set(available))
    invalid_sizes = sorted(
        path
        for path in required_paths
        if path in available
        and (
            type(available[path].get("size")) is not int
            or available[path]["size"] <= 0
        )
    )
    required_bytes = sum(
        int(available[path]["size"])
        for path in required_paths
        if path in available and path not in invalid_sizes
    )
    return available, missing, invalid_sizes, required_bytes


def _packages_by_version(packages_gz: bytes) -> set[tuple[str, str]]:
    text = gzip.decompress(packages_gz).decode("utf-8")
    rows: set[tuple[str, str]] = set()
    for stanza in text.split("\n\n"):
        fields = dict(
            line.split(": ", 1)
            for line in stanza.splitlines()
            if ": " in line and line.split(": ", 1)[0] in {"Package", "Version"}
        )
        if "Package" in fields and "Version" in fields:
            rows.add((fields["Package"], fields["Version"]))
    return rows


def _verify_isaac_base_image(
    fetch: Fetch, authorized_fetch: AuthorizedFetch
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    check: dict[str, Any] = {}
    try:
        token_payload = _json_object(
            fetch("https://nvcr.io/proxy_auth?scope=repository%3Anvidia%2Fisaac-sim%3Apull")
        )
        token = token_payload.get("token")
        if not isinstance(token, str) or not token:
            raise ValueError("anonymous_pull_token_absent")
        manifest = authorized_fetch(
            f"https://nvcr.io/v2/nvidia/isaac-sim/manifests/sha256:{ISAAC_BASE_DIGEST}",
            token,
        )
        actual_digest = hashlib.sha256(manifest).hexdigest()
        metadata = _json_object(manifest)
        manifest_rows = metadata.get("manifests", [])
        linux_amd64_present = any(
            isinstance(row, dict)
            and isinstance(row.get("platform"), dict)
            and row["platform"].get("os") == "linux"
            and row["platform"].get("architecture") == "amd64"
            for row in manifest_rows
        )
        check = {
            "repository": ISAAC_BASE_REPOSITORY,
            "digest": f"sha256:{actual_digest}",
            "manifest_bytes": len(manifest),
            "linux_amd64_present": linux_amd64_present,
        }
        if actual_digest != ISAAC_BASE_DIGEST:
            blockers.append("isaac_base_image_digest_mismatch")
        if not linux_amd64_present:
            blockers.append("isaac_base_image_linux_amd64_manifest_absent")
    except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
        blockers.append(f"isaac_base_image_manifest_unavailable:{type(exc).__name__}")
    return blockers, check


def verify_live(
    fetch: Fetch = _fetch,
    head_size: HeadSize = _head_size,
    authorized_fetch: AuthorizedFetch = _authorized_fetch,
) -> tuple[list[str], dict[str, Any]]:
    blockers = verify_static()
    checks: dict[str, Any] = {}

    base_blockers, base_check = _verify_isaac_base_image(fetch, authorized_fetch)
    blockers.extend(base_blockers)
    checks["isaac_base_image"] = base_check

    for url, expected_digest in ARTIFACTS:
        label = url.rsplit("/", 1)[-1]
        try:
            payload = fetch(url)
            actual_digest = hashlib.sha256(payload).hexdigest()
            checks[f"artifact:{label}"] = {
                "bytes": len(payload),
                "sha256": actual_digest,
            }
            if actual_digest != expected_digest:
                blockers.append(f"artifact_checksum_mismatch:{label}")
        except (OSError, ValueError, urllib.error.URLError) as exc:
            blockers.append(f"artifact_unavailable:{label}:{type(exc).__name__}")

    try:
        package_rows = _packages_by_version(fetch(f"{CUDA_REPOSITORY}/Packages.gz"))
        missing_packages = sorted(
            package
            for package in TENSORRT_PACKAGES
            if (package, TENSORRT_VERSION) not in package_rows
        )
        checks["tensorrt_packages"] = {
            "version": TENSORRT_VERSION,
            "required": len(TENSORRT_PACKAGES),
            "missing": missing_packages,
        }
        blockers.extend(f"tensorrt_package_unavailable:{row}" for row in missing_packages)
        cuda_cudart_missing = (
            ("cuda-cudart-12-6", CUDA_CUDART_VERSION) not in package_rows
        )
        checks["cuda_cudart_package"] = {
            "version": CUDA_CUDART_VERSION,
            "missing": cuda_cudart_missing,
        }
        if cuda_cudart_missing:
            blockers.append("cuda_cudart_package_unavailable:cuda-cudart-12-6")
    except (OSError, ValueError, gzip.BadGzipFile, urllib.error.URLError) as exc:
        blockers.append(f"nvidia_package_index_unavailable:{type(exc).__name__}")

    source_file_payloads: dict[tuple[str, str], bytes] = {}
    for repository, revision in SOURCE_PINS:
        try:
            metadata = _json_object(
                fetch(f"https://api.github.com/repos/{repository}/commits/{revision}")
            )
            actual_revision = metadata.get("sha")
            checks[f"source:{repository}"] = {"revision": actual_revision}
            if actual_revision != revision:
                blockers.append(f"source_revision_mismatch:{repository}")
        except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
            blockers.append(f"source_revision_unavailable:{repository}:{type(exc).__name__}")
        for path in SOURCE_REQUIRED_FILES[repository]:
            try:
                content = fetch(
                    f"https://raw.githubusercontent.com/{repository}/{revision}/{path}"
                )
                if not content:
                    raise ValueError("source_file_empty")
                source_file_payloads[(repository, path)] = content
                checks[f"source_file:{repository}:{path}"] = {
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            except (OSError, ValueError, urllib.error.URLError) as exc:
                blockers.append(
                    f"source_file_unavailable:{repository}:{path}:{type(exc).__name__}"
                )

    groot_metadata = source_file_payloads.get(
        ("NVIDIA/Isaac-GR00T", "pyproject.toml"), b""
    ).decode("utf-8", errors="replace")
    oscar_requirements = source_file_payloads.get(
        ("wuzy2115/oscar-public", "requirements_minimal.txt"), b""
    ).decode("utf-8", errors="replace")
    oscar_requirements_digest = hashlib.sha256(oscar_requirements.encode()).hexdigest()
    lock_text = OSCAR_FOUNDATION_LOCK.read_text(encoding="utf-8")
    if (
        f"# blueprint-input-sha256 oscar-requirements-minimal {oscar_requirements_digest}"
        not in lock_text
    ):
        blockers.append("oscar_foundation_lock_upstream_input_digest_stale")
    groot_torch = re.search(r'"torch==([^";]+)', groot_metadata)
    oscar_torch = re.search(r"(?m)^torch==([^\s#]+)", oscar_requirements)
    groot_torch_version = groot_torch.group(1) if groot_torch else None
    oscar_torch_version = oscar_torch.group(1) if oscar_torch else None
    checks["python_environment_compatibility"] = {
        "groot_torch_version": groot_torch_version,
        "oscar_torch_version": oscar_torch_version,
        "shared_torch_environment_supported": (
            bool(groot_torch_version)
            and groot_torch_version == oscar_torch_version
        ),
        "isolated_environments_required": groot_torch_version != oscar_torch_version,
    }
    if groot_torch_version != "2.7.1":
        blockers.append("groot_declared_torch_version_drifted")
    if oscar_torch_version != "2.10.0":
        blockers.append("oscar_declared_torch_version_drifted")

    asset_failures: list[str] = []
    asset_bytes = 0
    for _digest, expected_size, relative_path in _asset_rows():
        try:
            actual_size = head_size(ISAAC_ASSET_BASE_URL + relative_path)
            asset_bytes += actual_size
            if actual_size != expected_size:
                asset_failures.append(relative_path)
                blockers.append(f"isaac_asset_size_mismatch:{relative_path}")
        except (OSError, ValueError, urllib.error.URLError) as exc:
            asset_failures.append(relative_path)
            blockers.append(f"isaac_asset_unavailable:{relative_path}:{type(exc).__name__}")
    checks["isaac_assets"] = {
        "required": len(_asset_rows()),
        "total_remote_bytes": asset_bytes,
        "failures": sorted(asset_failures),
    }

    model_pins = cast(
        tuple[tuple[str, str, str], ...], _assigned_literal(MODEL_CACHE, "MODEL_PINS")
    )
    required_files = cast(
        dict[str, tuple[str, ...]],
        _assigned_literal(MODEL_CACHE, "REQUIRED_MODEL_FILES"),
    )
    required_model_cache_bytes = 0
    required_model_cache_files = 0
    for name, repository, revision in model_pins:
        try:
            metadata = _json_object(
                fetch(
                    f"https://huggingface.co/api/models/{repository}/revision/"
                    f"{revision}?blobs=true"
                )
            )
            actual_revision = metadata.get("sha")
            _available, missing, invalid_sizes, model_bytes = (
                summarize_required_model_metadata(
                    metadata.get("siblings", []), required_files[name]
                )
            )
            required_model_cache_bytes += model_bytes
            required_model_cache_files += len(required_files[name])
            checks[f"model:{name}"] = {
                "repository": repository,
                "revision": actual_revision,
                "required": len(required_files[name]),
                "required_bytes": model_bytes,
                "missing": missing,
                "invalid_sizes": invalid_sizes,
            }
            if actual_revision != revision:
                blockers.append(f"model_revision_mismatch:{name}")
            blockers.extend(f"model_file_unavailable:{name}:{path}" for path in missing)
            blockers.extend(
                f"model_file_size_unavailable:{name}:{path}" for path in invalid_sizes
            )
        except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
            blockers.append(f"model_revision_unavailable:{name}:{type(exc).__name__}")

    checks["model_cache_plan"] = {
        "required_files": required_model_cache_files,
        "required_bytes": required_model_cache_bytes,
        "minimum_volume_size_gib": 30,
        "recommended_volume_size_gib": 50,
        "provider_volume_verified": False,
        "model_bytes_downloaded_or_hashed": False,
        "claim_boundary": "metadata sizing is not offline model-cache verification",
    }

    return sorted(set(blockers)), checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.live:
        blockers, checks = verify_live()
    else:
        blockers, checks = verify_static(), {}
    payload = {
        "schema": SCHEMA,
        "status": "blocked" if blockers else "ready",
        "live": args.live,
        "blockers": blockers,
        "checks": checks,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 2 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
