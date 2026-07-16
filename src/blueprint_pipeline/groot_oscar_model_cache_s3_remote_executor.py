"""Closed remote handler for an admitted CPU model-cache S3 packet.

There is deliberately no CLI. The canonical DigitalOcean adapter invokes the
fixed packet entrypoint after local admission and binds it to a one-time secret,
the verified archive digest, droplet identity, launch-bound host key, retained
volume watchdog, and exact result paths.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import json
import os
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request
import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .common import ensure_dir, write_json
from .groot_oscar_runpod_carrier_volume import (
    DEFAULT_RUNTIME_ARCHIVE_PATH,
    DEFAULT_RUNTIME_MANIFEST_PATH,
    DEFAULT_RUNTIME_ROOT,
    MIN_CARRIER_VOLUME_GIB,
    RUNTIME_ARCHIVE_ROOTS,
    RUNTIME_CARRIER_ENV,
    build_runtime_bundle_manifest,
    is_nvidia_driver_soname,
)
from .groot_oscar_model_cache import (
    COSMOS_RUNTIME_MODEL_RELATIVE_PATH,
    VERIFICATION_SCHEMA_VERSION,
    prepare_model_cache,
    verify_model_cache,
)
from .groot_oscar_runpod_s3_model_cache import (
    DEFAULT_REMOTE_PREFIX,
    RUNPOD_S3_VOLUME_DATA_CENTER_IDS,
    _issue_transport_execution_capability,
    _upload_and_verify_model_cache_impl,
)


PACKET_SCHEMA_VERSION = "groot_oscar_model_cache_s3_remote_packet.v1"
EXECUTION_SCHEMA_VERSION = "groot_oscar_model_cache_s3_remote_execution.v1"
PARENT_BINDING_SCHEMA_VERSION = "groot_oscar_model_cache_s3_parent_binding.v1"
RUNTIME_CACHE_ROOT = Path("/workspace/.blueprint-model-cache/blueprint-groot-oscar-v1")
RUNTIME_COSMOS_MODEL_ROOT = RUNTIME_CACHE_ROOT / COSMOS_RUNTIME_MODEL_RELATIVE_PATH
VERIFICATION_ROOT = Path("/workspace/.blueprint-model-cache-verification")
REMOTE_PACKET_ROOT = Path("/root/blueprint-build/run/groot_oscar_model_cache_s3_remote")
PACKET_PATH = REMOTE_PACKET_ROOT / "packet.json"
CONTEXT_ROOT = REMOTE_PACKET_ROOT / "context"
CONTEXT_MANIFEST_PATH = REMOTE_PACKET_ROOT / "context_manifest.json"
DEPENDENCY_MANIFEST_PATH = REMOTE_PACKET_ROOT / "dependency_manifest.json"
DEPENDENCY_LOCK_PATH = REMOTE_PACKET_ROOT / "uv.lock"
REQUIREMENTS_CLOSURE_PATH = REMOTE_PACKET_ROOT / "requirements_closure.json"
OUTPUT_DIR = REMOTE_PACKET_ROOT / "results"
PARENT_BINDING_PATH = Path("/root/blueprint-build/model_cache_parent_binding.json")
PARENT_CAPABILITY_PATH = Path("/root/.blueprint-secrets/model_cache_parent_capability")
CONSUMED_CAPABILITY_PATH = Path("/root/.blueprint-secrets/model_cache_parent_capability.consumed")
EXECUTION_LOCK_PATH = Path("/root/blueprint-build/model_cache_execution.lock")
RUNTIME_BUNDLE_ROOT = Path(DEFAULT_RUNTIME_ROOT)
RUNTIME_BUNDLE_BUILD_ROOT = Path("/workspace/.blueprint-runtime-build")
RUNTIME_EMBEDDED_MODEL_PATHS = (
    "opt/blueprint/ckpts",
    "opt/blueprint/hf_home",
    "opt/blueprint/models",
)
HF_TOKEN_PATH = Path("/root/.blueprint-secrets/hf_token")
S3_ACCESS_KEY_PATH = Path("/root/.blueprint-secrets/runpod_s3_access_key")
S3_SECRET_KEY_PATH = Path("/root/.blueprint-secrets/runpod_s3_secret_key")
VENV_SITE_PACKAGES = Path("/root/blueprint-build/model-cache-venv/lib/python3.12/site-packages")
DEPENDENCY_RESULT_PATH = Path("/root/blueprint-build/model_cache_dependency_verification.json")
TRANSPORT_RESULT_NAME = "runpod_s3_model_cache_transport_result.json"
CANARY_VERIFICATION_NAME = "external_model_cache_verification.json"
EXECUTION_RESULT_NAME = "model_cache_s3_remote_execution_result.json"
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_SAFE_NONCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{7,127}")
_SAFE_DROPLET_ID = re.compile(r"[0-9]{1,32}")
_MISSING_PYTHON_MODULE = re.compile(
    r"No module named ['\"]([A-Za-z0-9_.-]+)['\"]"
)
_PACKET_KEYS = frozenset(
    {
        "schema_version",
        "packet_kind",
        "source_commit",
        "source_patch_sha256",
        "runtime_cache_root",
        "verification_root",
        "remote_prefix",
        "data_center_id",
        "allocation_nonce",
        "volume_evidence",
        "volume_watchdog_handoff",
        "runtime_bundle_request",
        "context_manifest_sha256",
        "dependency_manifest_sha256",
        "dependency_lock_sha256",
        "requirements_closure_sha256",
        "result_files",
        "raw_secret_values_recorded",
    }
)
_BINDING_KEYS = frozenset(
    {
        "schema_version",
        "packet_kind",
        "tarball_sha256",
        "capability_sha256",
        "droplet_id",
        "name",
        "region",
        "ssh_host_key_sha256",
        "provider_volume_id",
        "allocation_nonce",
        "builder_deadline_epoch",
        "volume_watchdog_deadline_epoch",
        "archive_members",
        "binding_hmac_sha256",
        "raw_secret_values_recorded",
    }
)


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("typed_model_cache_json_not_object")
    return payload


def _normalize_distribution_name(value: object) -> str:
    return re.sub(r"[-_.]+", "-", str(value or "").strip().lower())


def _verified_installed_dependencies(
    dependency_manifest: Mapping[str, Any],
) -> dict[str, str]:
    """Prove the fixed venv contains exactly the locked application closure."""

    rows = dependency_manifest.get("requirements")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("typed_model_cache_dependency_requirements_invalid")
    expected: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"name", "version"}:
            raise RuntimeError("typed_model_cache_dependency_requirements_invalid")
        name = _normalize_distribution_name(row.get("name"))
        version = str(row.get("version") or "")
        if not name or not version or name in expected:
            raise RuntimeError("typed_model_cache_dependency_requirements_invalid")
        expected[name] = version
    installed = {
        _normalize_distribution_name(distribution.metadata.get("Name")): distribution.version
        for distribution in importlib.metadata.distributions(path=[str(VENV_SITE_PACKAGES)])
        if distribution.metadata.get("Name")
    }
    bootstrap = {"pip", "setuptools", "wheel"}
    application_installed = {
        name: version for name, version in installed.items() if name not in bootstrap
    }
    if application_installed != expected:
        raise RuntimeError("typed_model_cache_dependency_inventory_mismatch")
    return dict(sorted(application_installed.items()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_pinned_image(value: object) -> bool:
    ref = str(value or "").strip()
    _name, marker, digest = ref.rpartition("@sha256:")
    return bool(marker and _HEX64.fullmatch(digest))


def _run_command(
    runner: Any,
    argv: list[str],
    *,
    timeout: int = 7200,
) -> subprocess.CompletedProcess[str]:
    return runner(
        argv,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _pull_and_verify_image(runner: Any, image_ref: str) -> None:
    _run_command(runner, ["docker", "pull", image_ref])
    inspected = _run_command(
        runner,
        ["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", image_ref],
        timeout=120,
    )
    try:
        repo_digests = json.loads(inspected.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("typed_runtime_bundle_image_inspect_invalid") from exc
    expected = image_ref.rpartition("@")[2]
    if not isinstance(repo_digests, list) or not any(
        str(item).rpartition("@")[2] == expected for item in repo_digests
    ):
        raise RuntimeError("typed_runtime_bundle_image_digest_unverified")


def _runtime_payload_tree_safe(payload_root: Path) -> bool:
    for path in payload_root.rglob("*"):
        mode = path.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode) or stat.S_ISLNK(mode)):
            return False
        if not stat.S_ISLNK(mode):
            continue
        member = path.relative_to(payload_root).as_posix()
        target = os.readlink(path)
        resolved = (
            posixpath.normpath(target).lstrip("/")
            if target.startswith("/")
            else posixpath.normpath(posixpath.join(posixpath.dirname(member), target))
        )
        if resolved == ".." or resolved.startswith("../"):
            return False
        if not any(
            resolved == root or resolved.startswith(root + "/") for root in RUNTIME_ARCHIVE_ROOTS
        ):
            return False
    return True


def _remove_embedded_model_payloads(payload_root: Path) -> list[str]:
    removed: list[str] = []
    for relative in RUNTIME_EMBEDDED_MODEL_PATHS:
        path = payload_root / relative
        if path.is_symlink() or path.is_file():
            path.unlink()
            removed.append(relative)
        elif path.is_dir():
            shutil.rmtree(path)
            removed.append(relative)
    return removed


class RuntimeCarrierValidationError(RuntimeError):
    """Typed failure carrying only allowlisted, secret-free audit evidence."""

    def __init__(self, *, failed_checks: list[str], evidence: Mapping[str, Any]) -> None:
        super().__init__(
            "typed_runtime_bundle_carrier_validation_failed:"
            + ":".join(failed_checks)
        )
        self.evidence = dict(evidence)


def _missing_soname_tokens(text: str) -> set[str]:
    """Extract only bounded shared-library names without backtracking regexes."""

    allowed = frozenset(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_+.-"
    )
    tokens: set[str] = set()
    for line in text.splitlines():
        candidate = ""
        if "=> not found" in line:
            left = line.split("=> not found", 1)[0].strip()
            candidate = left.split()[-1] if left else ""
        elif ": cannot open shared object file" in line:
            left = line.split(": cannot open shared object file", 1)[0]
            segment = left.rsplit(":", 1)[-1].strip()
            candidate = segment.split()[-1] if segment else ""
        if (
            0 < len(candidate) <= 256
            and ".so" in candidate
            and all(character in allowed for character in candidate)
        ):
            tokens.add(candidate)
    return tokens


def _validation_dependency_tokens(exc: BaseException) -> set[str]:
    text = "\n".join(
        value
        for value in (getattr(exc, "stdout", ""), getattr(exc, "stderr", ""))
        if isinstance(value, str)
    )
    return {
        *_missing_soname_tokens(text),
        *(_MISSING_PYTHON_MODULE.findall(text)),
    }


def _validation_diagnostic_tokens(exc: BaseException) -> list[str]:
    return sorted(_validation_dependency_tokens(exc))[:256]


def _validate_runtime_inside_carrier(
    runner: Any, *, carrier_ref: str, payload_root: Path
) -> dict[str, Any]:
    validations = (
        (
            "all_archived_elf_linkage",
            r"""
missing="$(mktemp)"
deferred="$(mktemp)"
elf_count=0
while IFS= read -r -d '' candidate; do
  magic="$(head -c 4 "$candidate" 2>/dev/null || true)"
  [[ "$magic" == $'\x7fELF' ]] || continue
  elf_count=$((elf_count + 1))
  linkage="$(ldd "$candidate" 2>&1 || true)"
  while IFS= read -r missing_line; do
    soname="${missing_line%%=>*}"
    soname="${soname#"${soname%%[![:space:]]*}"}"
    soname="${soname%"${soname##*[![:space:]]}"}"
    case "$soname" in
      libnvidia-*.so*|libcuda.so|libcuda.so.*|libnvcuvid.so|libnvcuvid.so.*|libnvoptix.so|libnvoptix.so.*|libGLX_nvidia.so|libGLX_nvidia.so.*|libEGL_nvidia.so|libEGL_nvidia.so.*|libGLESv1_CM_nvidia.so|libGLESv1_CM_nvidia.so.*|libGLESv2_nvidia.so|libGLESv2_nvidia.so.*)
        printf '%s\n' "$soname" >>"$deferred"
        ;;
      *)
        {
          printf 'ELF %s\n' "$candidate"
          printf '%s\n' "$missing_line"
        } >>"$missing"
        ;;
    esac
  done < <(grep -F 'not found' <<<"$linkage" || true)
done < <(find \
  /isaac-sim \
  /opt/OSCAR \
  /opt/blueprint \
  /opt/gr00t \
  /opt/gr00t-venv \
  /opt/onnxruntime \
  /opt/oscar-venv \
  /opt/runpod-serverless-venv \
  /opt/uv-python \
  /opt/wbc \
  -xdev -type f -print0)
test "$elf_count" -gt 0
if [[ -s "$missing" ]]; then
  cat "$missing"
  exit 91
fi
if [[ -s "$deferred" ]]; then
  while IFS= read -r soname; do
    printf 'BLUEPRINT_GPU_DRIVER_DEFERRED_SONAME %s\n' "$soname"
  done < <(sort -u "$deferred")
fi
printf 'BLUEPRINT_ALL_ARCHIVED_ELF_LINKAGE_OK count=%s\n' "$elf_count"
""",
            900,
        ),
        (
            "gr00t_import",
            "/opt/gr00t-venv/bin/python -c 'import numpy; import safetensors; "
            "import torch; import transformers; "
            "from gr00t.policy.gr00t_policy import Gr00tPolicy'",
            300,
        ),
        (
            "oscar_import_matrix",
            "/opt/oscar-venv/bin/python -c 'import blueprint_pipeline; import diffusers; "
            "import imageio; import msgpack; import mujoco; import numpy; import PIL; "
            "import safetensors; import torch; import transformers; import yaml; import zmq; "
            "import inference.inference_oscar; "
            "from transformer_engine.pytorch.attention import apply_rotary_pos_emb'",
            300,
        ),
        (
            "serverless_import",
            "/opt/runpod-serverless-venv/bin/python -m "
            "blueprint_pipeline.groot_oscar_runpod_serverless_worker "
            "--verify-serverless-runtime",
            300,
        ),
        (
            "isaac_import_matrix",
            "/isaac-sim/python.sh -c 'import blueprint_pipeline; import carb; import isaacsim; "
            "from isaacsim import SimulationApp; from isaacsim.core.prims import SingleArticulation; "
            "import omni.kit.app; import omni.timeline; import omni.usd; "
            "import blueprint_pipeline.isaac_runtime_task_backend'",
            300,
        ),
        (
            "wbc_binary_executable",
            "test -x /opt/wbc/gear_sonic_deploy/target/release/g1_deploy_onnx_ref",
            300,
        ),
    )
    carrier_env_argv = [
        item
        for key, value in RUNTIME_CARRIER_ENV.items()
        for item in ("--env", f"{key}={value}")
    ]
    failed_checks: list[str] = []
    checks: list[dict[str, Any]] = []
    archived_elf_file_count = 0
    gpu_driver_deferred_sonames: set[str] = set()
    for check_name, validation, timeout_seconds in validations:
        try:
            completed = _run_command(
                runner,
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    *carrier_env_argv,
                    "--entrypoint",
                    "/bin/bash",
                    "--mount",
                    f"type=bind,src={payload_root / 'opt'},dst=/opt,readonly",
                    "--mount",
                    f"type=bind,src={payload_root / 'isaac-sim'},dst=/isaac-sim,readonly",
                    carrier_ref,
                    "-o",
                    "pipefail",
                    "-c",
                    validation,
                ],
                timeout=timeout_seconds,
            )
            if check_name == "all_archived_elf_linkage":
                match = re.search(r"count=([1-9][0-9]*)", completed.stdout or "")
                if match is None:
                    failed_checks.append(check_name)
                    checks.append(
                        {
                            "name": check_name,
                            "status": "failed",
                            "diagnostic_tokens": ["elf_count_evidence_missing"],
                        }
                    )
                    continue
                archived_elf_file_count = int(match.group(1))
                invalid_deferred_soname = False
                for line in (completed.stdout or "").splitlines():
                    marker = "BLUEPRINT_GPU_DRIVER_DEFERRED_SONAME "
                    if not line.startswith(marker):
                        continue
                    soname = line.removeprefix(marker).strip()
                    if not is_nvidia_driver_soname(soname):
                        invalid_deferred_soname = True
                        break
                    gpu_driver_deferred_sonames.add(soname)
                if invalid_deferred_soname:
                    failed_checks.append(check_name)
                    checks.append(
                        {
                            "name": check_name,
                            "status": "failed",
                            "diagnostic_tokens": ["driver_deferred_soname_invalid"],
                        }
                    )
                    continue
                checks.append(
                    {
                        "name": check_name,
                        "status": "passed",
                        "diagnostic_tokens": [],
                    }
                )
                continue
            checks.append(
                {
                    "name": check_name,
                    "status": "passed",
                    "diagnostic_tokens": [],
                }
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            dependency_tokens = _validation_dependency_tokens(exc)
            if dependency_tokens and all(
                is_nvidia_driver_soname(token) for token in dependency_tokens
            ):
                gpu_driver_deferred_sonames.update(dependency_tokens)
                checks.append(
                    {
                        "name": check_name,
                        "status": "deferred_to_gpu_driver_bootstrap",
                        "diagnostic_tokens": sorted(dependency_tokens)[:256],
                    }
                )
                continue
            failed_checks.append(check_name)
            checks.append(
                {
                    "name": check_name,
                    "status": "failed",
                    "diagnostic_tokens": _validation_diagnostic_tokens(exc),
                }
            )
    evidence = {
        "schema_version": "groot_oscar_runtime_carrier_compatibility_audit.v1",
        "status": (
            "failed"
            if failed_checks
            else (
                "passed_with_gpu_driver_deferred"
                if gpu_driver_deferred_sonames
                else "passed"
            )
        ),
        "failed_checks": failed_checks,
        "checks": checks,
        "archived_root_count": len(RUNTIME_ARCHIVE_ROOTS),
        "archived_elf_file_count": archived_elf_file_count,
        "gpu_driver_deferred_sonames": sorted(gpu_driver_deferred_sonames),
        "gpu_driver_resolution_required_at_bootstrap": bool(
            gpu_driver_deferred_sonames
        ),
        "all_failures_collected_before_blocking": True,
        "claim_boundary": (
            "This validates archived ELF linkage and import surfaces inside the exact "
            "carrier on CPU. It does not prove GPU, CUDA-driver, Isaac rendering, policy "
            "execution, artifact completion, or semantic task success."
        ),
        "raw_secret_values_recorded": False,
    }
    if failed_checks:
        raise RuntimeCarrierValidationError(
            failed_checks=failed_checks,
            evidence=evidence,
        )
    return evidence


def prepare_runtime_bundle(
    request: Mapping[str, Any],
    *,
    runtime_root: Path = RUNTIME_BUNDLE_ROOT,
    build_root: Path = RUNTIME_BUNDLE_BUILD_ROOT,
    runner: Any = subprocess.run,
    generated_at: str | None = None,
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy only the release runtime allowlist into a POSIX-preserving archive."""

    progress_evidence = progress if progress is not None else {}

    def record_progress(phase: str, *, allowlisted_root: str = "") -> None:
        progress_evidence.clear()
        progress_evidence.update(
            {
                "schema_version": "groot_oscar_runtime_bundle_progress.v1",
                "phase": phase,
                "allowlisted_root": allowlisted_root or None,
                "raw_secret_values_recorded": False,
            }
        )

    if request.get("enabled") is not True:
        return {
            "schema_version": "groot_oscar_runtime_bundle_preparation.v1",
            "status": "not_requested",
            "additional_artifacts": [],
            "gpu_compute_allocated": False,
            "raw_secret_values_recorded": False,
        }
    if set(request) != {"enabled", "source_release_image_ref", "carrier_image_ref"}:
        raise RuntimeError("typed_runtime_bundle_request_fields_invalid")
    source_ref = str(request.get("source_release_image_ref") or "").strip()
    carrier_ref = str(request.get("carrier_image_ref") or "").strip()
    if not _digest_pinned_image(source_ref) or not _digest_pinned_image(carrier_ref):
        raise RuntimeError("typed_runtime_bundle_image_ref_not_digest_pinned")
    if runtime_root.exists() or build_root.exists():
        raise RuntimeError("typed_runtime_bundle_output_not_fresh")
    if shutil.which("docker") is None:
        raise RuntimeError("typed_runtime_bundle_docker_missing")
    ensure_dir(runtime_root)
    payload_root = build_root / "payload"
    ensure_dir(payload_root / "opt")
    container_id = ""
    preparation_succeeded = False
    try:
        record_progress("pulling_source_release")
        try:
            _pull_and_verify_image(runner, source_ref)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("typed_runtime_bundle_source_image_command_failed") from exc
        record_progress("pulling_carrier")
        try:
            _pull_and_verify_image(runner, carrier_ref)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("typed_runtime_bundle_carrier_image_command_failed") from exc
        record_progress("creating_source_container")
        try:
            created = _run_command(runner, ["docker", "create", source_ref, "true"], timeout=120)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("typed_runtime_bundle_source_container_create_failed") from exc
        container_id = created.stdout.strip()
        if re.fullmatch(r"[0-9a-f]{12,64}", container_id) is None:
            raise RuntimeError("typed_runtime_bundle_container_id_invalid")
        for root in RUNTIME_ARCHIVE_ROOTS:
            source_path = "/" + root
            destination = payload_root / str(Path(root).parent)
            ensure_dir(destination)
            record_progress("copying_allowlisted_root", allowlisted_root=root)
            try:
                _run_command(
                    runner,
                    [
                        "docker",
                        "cp",
                        f"{container_id}:{source_path}",
                        str(destination) + "/",
                    ],
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(
                    f"typed_runtime_bundle_allowlisted_root_copy_failed:{root}"
                ) from exc
        record_progress("validating_payload_tree")
        if not all(
            (payload_root / root).exists() and not (payload_root / root).is_symlink()
            for root in RUNTIME_ARCHIVE_ROOTS
        ):
            raise RuntimeError("typed_runtime_bundle_allowlisted_root_missing")
        removed_embedded_model_paths = _remove_embedded_model_payloads(payload_root)
        if not _runtime_payload_tree_safe(payload_root):
            raise RuntimeError("typed_runtime_bundle_payload_tree_unsafe")
        record_progress("validating_runtime_inside_carrier")
        carrier_validation = _validate_runtime_inside_carrier(
            runner, carrier_ref=carrier_ref, payload_root=payload_root
        )
        record_progress("archiving_runtime_bundle")
        archive_path = runtime_root / Path(DEFAULT_RUNTIME_ARCHIVE_PATH).name
        with tarfile.open(archive_path, "w:gz", compresslevel=6) as archive:
            for root in RUNTIME_ARCHIVE_ROOTS:
                archive.add(payload_root / root, arcname=root, recursive=True)
        manifest = build_runtime_bundle_manifest(
            source_release_image_ref=source_ref,
            carrier_image_ref=carrier_ref,
            archive_sha256=_sha256(archive_path),
            archive_size_bytes=archive_path.stat().st_size,
            healthcheck_argv=(
                (
                    "/opt/gr00t-venv/bin/python",
                    "-c",
                    "import numpy; import safetensors; import torch; import transformers; "
                    "from gr00t.policy.gr00t_policy import Gr00tPolicy",
                ),
                (
                    "/opt/oscar-venv/bin/python",
                    "-c",
                    "import blueprint_pipeline; import diffusers; import imageio; import msgpack; "
                    "import mujoco; import numpy; import PIL; import safetensors; import torch; "
                    "import transformers; import yaml; import zmq; import inference.inference_oscar; "
                    "from transformer_engine.pytorch.attention import apply_rotary_pos_emb",
                ),
                (
                    "/opt/runpod-serverless-venv/bin/python",
                    "-m",
                    "blueprint_pipeline.groot_oscar_runpod_serverless_worker",
                    "--verify-serverless-runtime",
                ),
                (
                    "/isaac-sim/python.sh",
                    "-c",
                    "import blueprint_pipeline; import carb; import isaacsim; "
                    "from isaacsim import SimulationApp; "
                    "from isaacsim.core.prims import SingleArticulation; "
                    "import omni.kit.app; import omni.timeline; import omni.usd; "
                    "import blueprint_pipeline.isaac_runtime_task_backend",
                ),
            ),
            generated_at=generated_at or datetime.now(timezone.utc).isoformat(),
            gpu_driver_deferred_sonames=carrier_validation[
                "gpu_driver_deferred_sonames"
            ],
        )
        if manifest["status"] != "complete":
            raise RuntimeError("typed_runtime_bundle_manifest_blocked")
        manifest_path = runtime_root / Path(DEFAULT_RUNTIME_MANIFEST_PATH).name
        write_json(manifest_path, manifest)
        shutil.rmtree(build_root)
        preparation_succeeded = True
        record_progress("completed")
        return {
            "schema_version": "groot_oscar_runtime_bundle_preparation.v1",
            "status": "completed",
            "source_release_image_ref": source_ref,
            "carrier_image_ref": carrier_ref,
            "runtime_root": str(runtime_root),
            "archive_path": str(archive_path),
            "archive_sha256": _sha256(archive_path),
            "archive_size_bytes": archive_path.stat().st_size,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "additional_artifacts": [
                {
                    "local_path": str(archive_path),
                    "remote_key": DEFAULT_RUNTIME_ARCHIVE_PATH.removeprefix("/workspace/"),
                    "sha256": _sha256(archive_path),
                },
                {
                    "local_path": str(manifest_path),
                    "remote_key": DEFAULT_RUNTIME_MANIFEST_PATH.removeprefix("/workspace/"),
                    "sha256": _sha256(manifest_path),
                },
            ],
            "source_and_carrier_registry_digests_verified": True,
            "embedded_model_paths_excluded": list(RUNTIME_EMBEDDED_MODEL_PATHS),
            "embedded_model_paths_present_and_removed": removed_embedded_model_paths,
            "models_supplied_only_by_verified_external_cache": True,
            "runtime_imports_and_wbc_linkage_verified_in_exact_carrier": not bool(
                carrier_validation["gpu_driver_deferred_sonames"]
            ),
            "runtime_cpu_compatible_with_gpu_driver_deferred_linkage": True,
            "gpu_driver_deferred_resolution_unproven": bool(
                carrier_validation["gpu_driver_deferred_sonames"]
            ),
            "carrier_validation": carrier_validation,
            "gpu_compute_allocated": False,
            "raw_secret_values_recorded": False,
        }
    finally:
        if container_id:
            try:
                _run_command(runner, ["docker", "rm", "-f", container_id], timeout=120)
            except Exception:  # noqa: BLE001 - caller fails if primary preparation failed
                if preparation_succeeded:
                    raise RuntimeError("typed_runtime_bundle_source_container_cleanup_failed")


def _secret(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_mode & 0o077:
        raise ValueError("typed_model_cache_secret_file_missing_or_not_private")
    value = resolved.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError("typed_model_cache_secret_file_empty")
    return value


def _context_blockers(packet: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if _sha256(CONTEXT_MANIFEST_PATH) != packet.get("context_manifest_sha256"):
        blockers.append("typed_model_cache_context_manifest_digest_mismatch")
        return blockers
    if _sha256(DEPENDENCY_MANIFEST_PATH) != packet.get("dependency_manifest_sha256"):
        blockers.append("typed_model_cache_dependency_manifest_digest_mismatch")
    if _sha256(DEPENDENCY_LOCK_PATH) != packet.get("dependency_lock_sha256"):
        blockers.append("typed_model_cache_dependency_lock_digest_mismatch")
    if _sha256(REQUIREMENTS_CLOSURE_PATH) != packet.get("requirements_closure_sha256"):
        blockers.append("typed_model_cache_dependency_closure_digest_mismatch")
    manifest = _load_object(CONTEXT_MANIFEST_PATH)
    rows = manifest.get("files") if isinstance(manifest.get("files"), list) else []
    expected = {
        str(row.get("path")): str(row.get("sha256")) for row in rows if isinstance(row, Mapping)
    }
    actual: dict[str, str] = {}
    for path in CONTEXT_ROOT.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            blockers.append("typed_model_cache_context_nonregular_file")
            continue
        actual[path.relative_to(REMOTE_PACKET_ROOT).as_posix()] = _sha256(path)
    if actual != expected:
        blockers.append("typed_model_cache_context_inventory_mismatch")
    return blockers


def _live_host_identity() -> dict[str, str]:
    def metadata(name: str) -> str:
        url = f"http://169.254.169.254/metadata/v1/{name}"
        with urllib.request.urlopen(url, timeout=5) as response:  # nosec B310
            return response.read(4096).decode("utf-8").strip()

    public = Path("/etc/ssh/ssh_host_ed25519_key.pub").read_text(encoding="utf-8").split()
    if len(public) < 2 or public[0] != "ssh-ed25519":
        raise ValueError("typed_model_cache_live_host_key_invalid")
    key_blob = base64.b64decode(public[1], validate=True)
    fingerprint = base64.b64encode(hashlib.sha256(key_blob).digest()).decode().rstrip("=")
    return {
        "droplet_id": metadata("id"),
        "name": metadata("hostname"),
        "region": metadata("region"),
        "ssh_host_key_sha256": "SHA256:" + fingerprint,
    }


def _contract_blockers(
    packet: Mapping[str, Any], binding: Mapping[str, Any], capability: bytes
) -> list[str]:
    blockers: list[str] = []
    volume = packet.get("volume_evidence")
    volume = volume if isinstance(volume, Mapping) else {}
    handoff = packet.get("volume_watchdog_handoff")
    handoff = handoff if isinstance(handoff, Mapping) else {}
    if set(packet) != _PACKET_KEYS:
        blockers.append("typed_model_cache_packet_fields_invalid")
    if set(binding) != _BINDING_KEYS:
        blockers.append("typed_model_cache_parent_binding_fields_invalid")
    if packet.get("schema_version") != PACKET_SCHEMA_VERSION:
        blockers.append("typed_model_cache_packet_schema_invalid")
    if packet.get("packet_kind") != "model_cache_s3":
        blockers.append("typed_model_cache_packet_kind_invalid")
    if binding.get("schema_version") != PARENT_BINDING_SCHEMA_VERSION:
        blockers.append("typed_model_cache_parent_binding_schema_invalid")
    if binding.get("packet_kind") != "model_cache_s3":
        blockers.append("typed_model_cache_parent_binding_kind_invalid")
    if packet.get("raw_secret_values_recorded") is not False:
        blockers.append("typed_model_cache_packet_secret_contract_invalid")
    if binding.get("raw_secret_values_recorded") is not False:
        blockers.append("typed_model_cache_parent_binding_secret_contract_invalid")
    if len(capability) < 32:
        blockers.append("typed_model_cache_parent_capability_too_short")
    if _HEX40.fullmatch(str(packet.get("source_commit") or "")) is None:
        blockers.append("typed_model_cache_source_commit_invalid")
    if packet.get("source_patch_sha256") != hashlib.sha256(b"").hexdigest():
        blockers.append("typed_model_cache_source_patch_not_clean")
    if _HEX64.fullmatch(str(binding.get("tarball_sha256") or "")) is None:
        blockers.append("typed_model_cache_tarball_digest_invalid")
    if hashlib.sha256(capability).hexdigest() != binding.get("capability_sha256"):
        blockers.append("typed_model_cache_parent_capability_invalid")
    signed_binding = {key: value for key, value in binding.items() if key != "binding_hmac_sha256"}
    expected_hmac = hmac.new(
        capability,
        json.dumps(signed_binding, sort_keys=True, separators=(",", ":")).encode(),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected_hmac, str(binding.get("binding_hmac_sha256") or "")):
        blockers.append("typed_model_cache_parent_binding_hmac_invalid")
    if _SAFE_DROPLET_ID.fullmatch(str(binding.get("droplet_id") or "")) is None:
        blockers.append("typed_model_cache_parent_droplet_id_invalid")
    if not str(binding.get("ssh_host_key_sha256") or "").startswith("SHA256:"):
        blockers.append("typed_model_cache_parent_host_key_invalid")
    try:
        live_host = _live_host_identity()
    except Exception:  # noqa: BLE001 - live identity must fail closed
        blockers.append("typed_model_cache_live_host_identity_unavailable")
    else:
        for field in ("droplet_id", "name", "region", "ssh_host_key_sha256"):
            if live_host[field] != str(binding.get(field) or ""):
                blockers.append(f"typed_model_cache_live_host_{field}_mismatch")
    nonce = str(packet.get("allocation_nonce") or "")
    if _SAFE_NONCE.fullmatch(nonce) is None:
        blockers.append("typed_model_cache_allocation_nonce_invalid")
    data_center = str(packet.get("data_center_id") or "")
    if data_center not in RUNPOD_S3_VOLUME_DATA_CENTER_IDS:
        blockers.append("typed_model_cache_data_center_invalid")
    if packet.get("runtime_cache_root") != str(RUNTIME_CACHE_ROOT):
        blockers.append("typed_model_cache_runtime_root_invalid")
    if packet.get("verification_root") != str(VERIFICATION_ROOT):
        blockers.append("typed_model_cache_verification_root_invalid")
    prefix = str(packet.get("remote_prefix") or "")
    if prefix != DEFAULT_REMOTE_PREFIX or f"/workspace/{prefix}" != str(RUNTIME_CACHE_ROOT):
        blockers.append("typed_model_cache_runtime_prefix_mapping_invalid")
    if packet.get("result_files") != [
        TRANSPORT_RESULT_NAME,
        CANARY_VERIFICATION_NAME,
        EXECUTION_RESULT_NAME,
    ]:
        blockers.append("typed_model_cache_result_contract_invalid")
    runtime_request = packet.get("runtime_bundle_request")
    runtime_request = runtime_request if isinstance(runtime_request, Mapping) else {}
    if runtime_request.get("enabled") is True:
        if (
            set(runtime_request) != {"enabled", "source_release_image_ref", "carrier_image_ref"}
            or not _digest_pinned_image(runtime_request.get("source_release_image_ref"))
            or not _digest_pinned_image(runtime_request.get("carrier_image_ref"))
        ):
            blockers.append("typed_model_cache_runtime_bundle_request_invalid")
        if (
            type(volume.get("size_bytes")) is not int
            or int(volume.get("size_bytes") or 0) < MIN_CARRIER_VOLUME_GIB * 1024**3
        ):
            blockers.append("typed_model_cache_runtime_volume_below_120_gib")
    elif runtime_request != {"enabled": False}:
        blockers.append("typed_model_cache_runtime_bundle_request_invalid")
    if volume.get("schema_version") != "groot_oscar_runpod_network_volume_evidence.v1":
        blockers.append("typed_model_cache_volume_schema_invalid")
    if volume.get("status") != "verified" or volume.get("provider_api_verified") is not True:
        blockers.append("typed_model_cache_volume_not_verified")
    if (
        volume.get("id") != binding.get("provider_volume_id")
        or volume.get("data_center_id") != data_center
        or volume.get("allocation_nonce") != nonce
        or volume.get("allocation_name_verified") is not True
        or nonce not in str(volume.get("name") or "")
    ):
        blockers.append("typed_model_cache_volume_identity_mismatch")
    if binding.get("allocation_nonce") != nonce:
        blockers.append("typed_model_cache_parent_nonce_mismatch")
    deadline_raw = binding.get("builder_deadline_epoch")
    deadline = (
        float(deadline_raw)
        if isinstance(deadline_raw, (int, float)) and not isinstance(deadline_raw, bool)
        else None
    )
    if deadline is None or deadline <= time.time() + 600:
        blockers.append("typed_model_cache_builder_deadline_too_near")
    volume_deadline_raw = handoff.get("watchdog_deadline_epoch")
    volume_deadline = (
        float(volume_deadline_raw)
        if isinstance(volume_deadline_raw, (int, float))
        and not isinstance(volume_deadline_raw, bool)
        else None
    )
    if (
        handoff.get("schema_version") != "groot_oscar_model_volume_watchdog_handoff.v1"
        or handoff.get("status") != "storage_preparation_watchdog_armed"
        or handoff.get("volume_id") != volume.get("id")
        or handoff.get("teardown_owner") != "independent_model_volume_watchdog"
        or volume_deadline is None
        or deadline is None
        or volume_deadline <= (deadline or 0) + 2100
        or binding.get("volume_watchdog_deadline_epoch") != volume_deadline_raw
    ):
        blockers.append("typed_model_cache_volume_watchdog_handoff_invalid")
    blockers.extend(_context_blockers(packet))
    return blockers


def _canary_verification(
    *,
    packet: Mapping[str, Any],
    local: Mapping[str, Any],
    transport: Mapping[str, Any],
    transport_result_sha256: str,
    runtime_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    volume = packet["volume_evidence"]
    manifest_digest = local.get("model_manifest_digest")
    blockers: list[str] = []
    if transport.get("status") != "completed":
        blockers.append("typed_model_cache_transport_not_completed")
    if not manifest_digest or transport.get("model_manifest_digest") != manifest_digest:
        blockers.append("typed_model_cache_transport_manifest_digest_mismatch")
    if transport.get("remote_model_manifest_digest") != manifest_digest:
        blockers.append("typed_model_cache_remote_manifest_digest_mismatch")
    if _HEX64.fullmatch(
        str(transport.get("model_manifest_file_sha256") or "")
    ) is None or transport.get("remote_model_manifest_file_sha256") != transport.get(
        "model_manifest_file_sha256"
    ):
        blockers.append("typed_model_cache_manifest_file_sha256_mismatch")
    if transport.get("provider_volume_id") != volume.get("id"):
        blockers.append("typed_model_cache_transport_volume_mismatch")
    if transport.get("remote_provider_volume_id") != volume.get("id"):
        blockers.append("typed_model_cache_remote_volume_mismatch")
    if transport.get("remote_prefix") != packet.get("remote_prefix"):
        blockers.append("typed_model_cache_transport_prefix_mismatch")
    if transport.get("verification_method") != (
        "full_s3_redownload_and_sha256_manifest_verification"
    ):
        blockers.append("typed_model_cache_transport_verification_method_invalid")
    if (
        not isinstance(transport.get("verified_size_bytes"), int)
        or int(transport.get("verified_size_bytes") or 0) <= 0
    ):
        blockers.append("typed_model_cache_transport_verified_bytes_invalid")
    if (
        not isinstance(transport.get("remote_verified_file_count"), int)
        or int(transport.get("remote_verified_file_count") or 0) <= 0
    ):
        blockers.append("typed_model_cache_remote_verified_file_count_invalid")
    if _HEX64.fullmatch(str(transport.get("remote_verification_sha256") or "")) is None:
        blockers.append("typed_model_cache_remote_verification_digest_invalid")
    if _HEX64.fullmatch(transport_result_sha256) is None:
        blockers.append("typed_model_cache_transport_result_digest_invalid")
    runtime_status = runtime_bundle.get("status")
    runtime_verification = transport.get("additional_artifact_verification")
    runtime_verification = runtime_verification if isinstance(runtime_verification, list) else []
    if runtime_status == "completed":
        expected = {
            str(runtime_bundle.get("archive_sha256") or ""),
            str(runtime_bundle.get("manifest_sha256") or ""),
        }
        observed = {
            str(row.get("sha256") or "")
            for row in runtime_verification
            if isinstance(row, Mapping) and row.get("full_redownload_sha256_verified") is True
        }
        if (
            transport.get("additional_artifact_verification_method")
            != "full_s3_redownload_and_sha256"
            or observed != expected
        ):
            blockers.append("typed_runtime_bundle_transport_verification_invalid")
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "passed" if not blockers else "blocked",
        "blockers": blockers,
        "model_manifest_digest": manifest_digest,
        "expected_model_manifest_digest": manifest_digest,
        "model_manifest_file_sha256": transport.get("model_manifest_file_sha256"),
        "cache_root": str(RUNTIME_CACHE_ROOT),
        "provider_volume_id": volume.get("id"),
        "verified_file_count": transport.get("remote_verified_file_count") if not blockers else 0,
        "verified_size_bytes": transport.get("verified_size_bytes") if not blockers else 0,
        "checks": {"models_cached_offline": not blockers},
        "remote_prefix": packet.get("remote_prefix"),
        "runtime_path_mapping_verified": not blockers,
        "transport_result_sha256": transport_result_sha256,
        "remote_verification_sha256": transport.get("remote_verification_sha256"),
        "runtime_bundle": dict(runtime_bundle),
        "raw_secret_values_recorded": False,
    }


def execute_remote_packet() -> dict[str, Any]:
    """Execute only at the canonical adapter's fixed paths and capability."""

    result: dict[str, Any] = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "status": "failed",
        "blockers": ["typed_model_cache_remote_execution_failed"],
        "outer_volume_deletion_required": True,
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
    }
    transport: dict[str, Any] = {}
    runtime_bundle_progress: dict[str, Any] = {}
    terminal_exists = False
    try:
        ensure_dir(OUTPUT_DIR)
        terminal_exists = (OUTPUT_DIR / EXECUTION_RESULT_NAME).exists()
        if terminal_exists:
            result["blockers"] = ["typed_model_cache_terminal_result_already_exists"]
            raise RuntimeError("typed_model_cache_terminal_result_already_exists")
        lock_fd = os.open(
            EXECUTION_LOCK_PATH,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        os.close(lock_fd)
        if CONSUMED_CAPABILITY_PATH.exists():
            raise RuntimeError("typed_model_cache_capability_already_consumed")
        PARENT_CAPABILITY_PATH.replace(CONSUMED_CAPABILITY_PATH)
        capability = CONSUMED_CAPABILITY_PATH.read_bytes()
        packet = _load_object(PACKET_PATH)
        binding = _load_object(PARENT_BINDING_PATH)
        blockers = _contract_blockers(packet, binding, capability)
        if blockers:
            result = {
                **result,
                "status": "blocked",
                "blockers": sorted(set(blockers)),
            }
            raise RuntimeError("typed_model_cache_contract_blocked")
        if sys.version_info[:2] != (3, 12) or not VENV_SITE_PACKAGES.is_dir():
            raise RuntimeError("typed_model_cache_python_runtime_mismatch")
        dependency_manifest = _load_object(DEPENDENCY_MANIFEST_PATH)
        if dependency_manifest.get("schema_version") != "blueprint_python_wheelhouse.v1":
            raise RuntimeError("typed_model_cache_dependency_manifest_invalid")
        installed = _verified_installed_dependencies(dependency_manifest)
        sys.path.append(str(VENV_SITE_PACKAGES))
        import boto3
        import botocore
        import huggingface_hub

        write_json(
            DEPENDENCY_RESULT_PATH,
            {
                "schema_version": "groot_oscar_model_cache_dependency_verification.v1",
                "status": "verified",
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "dependency_manifest_sha256": packet["dependency_manifest_sha256"],
                "installed_distributions": dict(sorted(installed.items())),
                "required_import_versions": {
                    "boto3": boto3.__version__,
                    "botocore": botocore.__version__,
                    "huggingface-hub": huggingface_hub.__version__,
                },
                "dependency_imports_after_parent_contract_validation": True,
                "raw_secret_values_recorded": False,
            },
        )
        hf_token = _secret(HF_TOKEN_PATH)
        _secret(S3_ACCESS_KEY_PATH)
        _secret(S3_SECRET_KEY_PATH)
        runtime_bundle = prepare_runtime_bundle(
            packet["runtime_bundle_request"], progress=runtime_bundle_progress
        )
        manifest = prepare_model_cache(RUNTIME_CACHE_ROOT, token=hf_token)
        sonic = _load_object(RUNTIME_CACHE_ROOT / "sonic/config.json")
        if sonic.get("model_name") != str(RUNTIME_COSMOS_MODEL_ROOT):
            raise RuntimeError("typed_model_cache_sonic_runtime_path_mismatch")
        local = verify_model_cache(RUNTIME_CACHE_ROOT)
        if local.get("status") != "passed" or local.get("model_manifest_digest") != manifest.get(
            "manifest_digest"
        ):
            raise RuntimeError("typed_model_cache_local_verification_failed")
        volume = packet["volume_evidence"]
        transport = _upload_and_verify_model_cache_impl(
            cache_root=RUNTIME_CACHE_ROOT,
            verification_root=VERIFICATION_ROOT,
            volume_id=str(volume["id"]),
            data_center_id=str(packet["data_center_id"]),
            access_key_file=S3_ACCESS_KEY_PATH,
            secret_key_file=S3_SECRET_KEY_PATH,
            volume_evidence=volume,
            allocation_nonce=str(packet["allocation_nonce"]),
            remote_prefix=str(packet["remote_prefix"]),
            live_probe_attempts=12,
            live_probe_interval_seconds=5.0,
            execution_capability=_issue_transport_execution_capability(
                remote_parent_binding=binding,
                remote_parent_capability=capability,
                remote_packet=packet,
            ),
            additional_artifacts=runtime_bundle.get("additional_artifacts", []),
        )
        write_json(OUTPUT_DIR / TRANSPORT_RESULT_NAME, transport)
        transport_result_sha256 = _sha256(OUTPUT_DIR / TRANSPORT_RESULT_NAME)
        canary = _canary_verification(
            packet=packet,
            local=local,
            transport=transport,
            transport_result_sha256=transport_result_sha256,
            runtime_bundle=runtime_bundle,
        )
        write_json(OUTPUT_DIR / CANARY_VERIFICATION_NAME, canary)
        if canary["status"] != "passed":
            raise RuntimeError("typed_model_cache_canary_verification_blocked")
        result = {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "status": "completed",
            "blockers": [],
            "packet_kind": "model_cache_s3",
            "source_commit": packet["source_commit"],
            "source_patch_sha256": packet["source_patch_sha256"],
            "tarball_sha256": binding["tarball_sha256"],
            "droplet_id": binding["droplet_id"],
            "provider_volume_id": volume["id"],
            "model_manifest_digest": local["model_manifest_digest"],
            "runtime_bundle": runtime_bundle,
            "runtime_carrier_validation": runtime_bundle.get("carrier_validation"),
            "provider_mutations_performed": transport["provider_mutations_performed"],
            "outer_volume_deletion_required": False,
            "gpu_compute_allocated": False,
            "raw_secret_values_recorded": False,
        }
    except Exception as exc:  # noqa: BLE001 - terminal secret-free evidence
        transport_digest = (
            _sha256(OUTPUT_DIR / TRANSPORT_RESULT_NAME)
            if (OUTPUT_DIR / TRANSPORT_RESULT_NAME).is_file()
            else None
        )
        if result.get("status") != "blocked":
            error_code = str(exc)
            if re.fullmatch(r"typed_[a-z0-9_:-]+", error_code) is None:
                error_code = "typed_model_cache_remote_execution_unclassified"
            result.update(
                {
                    "status": "failed",
                    "blockers": ["typed_model_cache_remote_execution_failed"],
                    "error_type": type(exc).__name__,
                    "error_code": error_code,
                    "runtime_bundle_progress": dict(runtime_bundle_progress),
                }
            )
            if isinstance(exc, RuntimeCarrierValidationError):
                result["runtime_carrier_validation"] = exc.evidence
        result.update(
            {
                "provider_mutations_performed": int(
                    transport.get("provider_mutations_performed") or 0
                ),
                "transport_result_sha256": transport_digest,
                "partial_upload_cleanup_verified": transport.get("partial_upload_cleanup_verified"),
                "upload_attempt_count": int(transport.get("upload_attempt_count") or 0),
                "upload_success_count": int(transport.get("upload_success_count") or 0),
                "cleanup_delete_attempt_count": int(
                    transport.get("cleanup_delete_attempt_count") or 0
                ),
                "cleanup_delete_success_count": int(
                    transport.get("cleanup_delete_success_count") or 0
                ),
                "final_provider_observed_prefix_empty": transport.get(
                    "final_provider_observed_prefix_empty"
                ),
                "outer_volume_deletion_required": True,
                "gpu_compute_allocated": False,
            }
        )
    finally:
        cleanup = {}
        for path in (
            HF_TOKEN_PATH,
            S3_ACCESS_KEY_PATH,
            S3_SECRET_KEY_PATH,
            PARENT_CAPABILITY_PATH,
            CONSUMED_CAPABILITY_PATH,
        ):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                cleanup[str(path)] = False
            else:
                cleanup[str(path)] = not path.exists()
        result["secret_cleanup_verified"] = bool(cleanup and all(cleanup.values()))
        if not result["secret_cleanup_verified"]:
            result["status"] = "failed"
            result["outer_volume_deletion_required"] = True
            result["blockers"] = sorted(
                set([*result.get("blockers", []), "typed_model_cache_secret_cleanup_unverified"])
            )
    if not terminal_exists:
        write_json(OUTPUT_DIR / EXECUTION_RESULT_NAME, result)
    return result
