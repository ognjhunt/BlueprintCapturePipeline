"""Fail-closed contract for a small RunPod carrier plus a prepared volume.

The network volume is prepared without GPU compute.  It contains a tar archive
of the exact runtime copied from a digest-pinned release image and the already
verified GR00T/OSCAR model cache.  This module deliberately does not prepare or
launch either resource; it validates the handoff and renders the in-container
bootstrap that re-verifies the attached files before provider code runs.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import shlex
from typing import Any, Mapping, Sequence


RUNTIME_BUNDLE_MANIFEST_SCHEMA_VERSION = "groot_oscar_runtime_bundle_manifest.v2"
CARRIER_VOLUME_ADMISSION_SCHEMA_VERSION = "groot_oscar_runpod_carrier_volume_admission.v3"
LEGACY_CARRIER_VOLUME_ADMISSION_SCHEMA_VERSION = "groot_oscar_runpod_carrier_volume_admission.v2"
THIN_RUNTIME_SOURCE_EVIDENCE_SCHEMA_VERSION = "groot_oscar_thin_remote_build_result.v1"
RUNTIME_SOURCE_RELEASE_VERIFICATION_SCHEMA_VERSION = (
    "groot_oscar_runtime_source_release_verification.v1"
)
DEFAULT_RUNTIME_ROOT = "/workspace/.blueprint-runtime/blueprint-groot-oscar-v1"
DEFAULT_RUNTIME_ARCHIVE_PATH = f"{DEFAULT_RUNTIME_ROOT}/groot_oscar_runtime.tar.gz"
DEFAULT_RUNTIME_MANIFEST_PATH = f"{DEFAULT_RUNTIME_ROOT}/runtime_bundle_manifest.json"
DEFAULT_MODEL_CACHE_ROOT = "/workspace/.blueprint-model-cache/blueprint-groot-oscar-v1"
DEFAULT_MODEL_CACHE_MANIFEST_PATH = (
    f"{DEFAULT_MODEL_CACHE_ROOT}/groot_oscar_model_cache_manifest.json"
)
MIN_CARRIER_VOLUME_GIB = 120
RUNTIME_CARRIER_PYTHONPATH = "/opt/wbc:/opt/OSCAR"
RUNTIME_ELF_SYMLINK_FARM = "/opt/blueprint/runtime-libs"
RUNTIME_CARRIER_LD_LIBRARY_PATH = (
    "/opt/wbc/gear_sonic_deploy/thirdparty_runtime/lib:/opt/onnxruntime/lib:"
    "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:"
    f"/usr/local/nvidia/lib64:{RUNTIME_ELF_SYMLINK_FARM}"
)
RUNTIME_ARCHIVE_ROOTS = (
    "isaac-sim",
    "opt/OSCAR",
    "opt/blueprint",
    "opt/gr00t",
    "opt/gr00t-venv",
    "opt/onnxruntime",
    "opt/oscar-venv",
    "opt/runpod-serverless-venv",
    "opt/uv-python",
    "opt/wbc",
)
RUNTIME_CARRIER_ENV = {
    "PYTHONPATH": RUNTIME_CARRIER_PYTHONPATH,
    "LD_LIBRARY_PATH": RUNTIME_CARRIER_LD_LIBRARY_PATH,
}

_SHA256 = re.compile(r"[0-9a-f]{64}")
_VOLUME_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{2,127}")
_DATA_CENTER_ID = re.compile(r"[A-Z]{2,8}(?:-[A-Z0-9]+)+")
_COMMIT = re.compile(r"[0-9a-f]{40}")
NVIDIA_DRIVER_SYSTEM_PATH_PREFIXES = (
    "/lib/",
    "/lib64/",
    "/usr/lib/",
    "/usr/lib64/",
    "/usr/local/cuda/compat/",
    "/usr/local/nvidia/",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def is_nvidia_driver_soname(value: Any) -> bool:
    """Return whether a SONAME is supplied by the NVIDIA host driver injection."""

    soname = _string(value)
    if not soname or len(soname) > 256 or any(
        character.isspace() or character in "/\\\x00" for character in soname
    ):
        return False
    exact_stems = (
        "libcuda.so",
        "libnvcuvid.so",
        "libnvoptix.so",
        "libGLX_nvidia.so",
        "libEGL_nvidia.so",
        "libGLESv1_CM_nvidia.so",
        "libGLESv2_nvidia.so",
    )
    return (
        soname.startswith("libnvidia-") and ".so" in soname
    ) or any(soname == stem or soname.startswith(stem + ".") for stem in exact_stems)


def is_nvidia_driver_system_path(value: Any) -> bool:
    """Return whether an exact absolute path can be GPU-driver verified at bootstrap."""

    path = _string(value)
    if (
        not path
        or len(path) > 512
        or not path.startswith("/")
        or any(character.isspace() or character in "\\\x00" for character in path)
    ):
        return False
    normalized = posixpath.normpath(path)
    if normalized != path or not any(
        normalized.startswith(prefix) for prefix in NVIDIA_DRIVER_SYSTEM_PATH_PREFIXES
    ):
        return False
    return is_nvidia_driver_soname(posixpath.basename(normalized))


def _sha256_digest(value: Any) -> str:
    digest = _string(value).lower()
    return digest if _SHA256.fullmatch(digest) else ""


def _digest_pinned_image(value: Any) -> str:
    ref = _string(value)
    _, separator, digest = ref.rpartition("@sha256:")
    return ref if separator and _sha256_digest(digest) else ""


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def verify_runtime_source_release_evidence(
    value: Mapping[str, Any],
    *,
    expected_release_image_ref: str,
    expected_cuda_version: str = "12.8",
) -> dict[str, Any]:
    """Prove the runtime source is the exact thin, model-externalized release."""

    evidence = _mapping(value)
    contract = _mapping(evidence.get("thin_release_contract"))
    blockers: list[str] = []
    expected_ref = _digest_pinned_image(expected_release_image_ref)
    release_ref = _digest_pinned_image(
        evidence.get("resolved_digest_ref") or evidence.get("release_image_ref")
    )
    if evidence.get("schema_version") != THIN_RUNTIME_SOURCE_EVIDENCE_SCHEMA_VERSION:
        blockers.append("runtime_source_release_evidence_schema_invalid")
    if evidence.get("status") != "completed" or evidence.get("blockers") not in ([], ()):
        blockers.append("runtime_source_release_evidence_not_completed")
    if not expected_ref:
        blockers.append("runtime_source_expected_ref_not_digest_pinned")
    if not release_ref:
        blockers.append("runtime_source_release_ref_not_digest_pinned")
    if release_ref != expected_ref:
        blockers.append("runtime_source_release_ref_mismatch")
    if evidence.get("release_image_ref") != release_ref:
        blockers.append("runtime_source_release_declared_ref_mismatch")
    if evidence.get("runnable_platform") != "linux/amd64":
        blockers.append("runtime_source_release_platform_invalid")
    if _string(evidence.get("required_cuda_version")) != expected_cuda_version:
        blockers.append("runtime_source_release_cuda_version_mismatch")
    source_commit = _string(evidence.get("source_commit"))
    if not _COMMIT.fullmatch(source_commit):
        blockers.append("runtime_source_release_commit_invalid")
    if evidence.get("models_embedded") is not False:
        blockers.append("runtime_source_release_models_embedded")
    if evidence.get("thin_release_contract_status") != "passed":
        blockers.append("runtime_source_thin_contract_not_passed")
    if contract.get("schema_version") != "groot_oscar_thin_release_image_contract.v1":
        blockers.append("runtime_source_thin_contract_schema_invalid")
    if contract.get("status") != "passed" or contract.get("blockers") not in ([], ()):
        blockers.append("runtime_source_thin_contract_blocked")
    if contract.get("release_image_ref") != release_ref:
        blockers.append("runtime_source_thin_contract_ref_mismatch")
    if contract.get("models_externalized") is not True:
        blockers.append("runtime_source_thin_contract_models_not_externalized")
    if contract.get("release_delta_budget_passed") is not True:
        blockers.append("runtime_source_thin_contract_delta_budget_not_passed")
    if evidence.get("raw_secret_values_recorded") is not False:
        blockers.append("runtime_source_release_raw_secret_boundary_invalid")
    return {
        "schema_version": RUNTIME_SOURCE_RELEASE_VERIFICATION_SCHEMA_VERSION,
        "status": "blocked" if blockers else "verified",
        "blockers": sorted(set(blockers)),
        "release_image_ref": release_ref,
        "source_commit": source_commit,
        "required_cuda_version": _string(evidence.get("required_cuda_version")),
        "thin_release_contract_sha256": (canonical_json_sha256(contract) if contract else ""),
        "models_externalized": contract.get("models_externalized") is True,
        "claim_boundary": (
            "This verifies the exact model-externalized thin release used as the runtime "
            "copy source. It does not prove volume transfer, provider startup, inference, "
            "or semantic task success."
        ),
        "raw_secret_values_recorded": False,
    }


def build_runtime_bundle_manifest(
    *,
    source_release_image_ref: str,
    carrier_image_ref: str,
    archive_sha256: str,
    archive_size_bytes: int,
    healthcheck_argv: Sequence[Sequence[str]],
    generated_at: str,
    gpu_driver_deferred_sonames: Sequence[str] = (),
    gpu_driver_deferred_system_paths: Sequence[str] = (),
) -> dict[str, Any]:
    """Describe a deterministic runtime archive without claiming it was uploaded."""

    blockers: list[str] = []
    source_ref = _digest_pinned_image(source_release_image_ref)
    carrier_ref = _digest_pinned_image(carrier_image_ref)
    archive_digest = _sha256_digest(archive_sha256)
    if not source_ref:
        blockers.append("runtime_source_release_image_not_digest_pinned")
    if not carrier_ref:
        blockers.append("runtime_carrier_image_not_digest_pinned")
    if not archive_digest:
        blockers.append("runtime_archive_sha256_invalid")
    if type(archive_size_bytes) is not int or archive_size_bytes <= 0:
        blockers.append("runtime_archive_size_invalid")
    checks: list[list[str]] = []
    for row in healthcheck_argv:
        argv = [_string(item) for item in row]
        if not argv or any(not item or "\x00" in item for item in argv):
            blockers.append("runtime_healthcheck_argv_invalid")
            continue
        executable = argv[0]
        executable_in_archive = any(
            executable == f"/{root}" or executable.startswith(f"/{root}/")
            for root in RUNTIME_ARCHIVE_ROOTS
        )
        if not executable_in_archive or "/../" in executable:
            blockers.append("runtime_healthcheck_executable_outside_opt")
            continue
        checks.append(argv)
    if not checks:
        blockers.append("runtime_healthchecks_missing")
    deferred_sonames: list[str] = []
    for value in gpu_driver_deferred_sonames:
        soname = _string(value)
        if not is_nvidia_driver_soname(soname):
            blockers.append("runtime_gpu_driver_deferred_soname_invalid")
            continue
        deferred_sonames.append(soname)
    deferred_sonames = sorted(set(deferred_sonames))
    deferred_system_paths: list[str] = []
    for value in gpu_driver_deferred_system_paths:
        path = _string(value)
        if not is_nvidia_driver_system_path(path):
            blockers.append("runtime_gpu_driver_deferred_system_path_invalid")
            continue
        if posixpath.basename(path) not in deferred_sonames:
            blockers.append("runtime_gpu_driver_deferred_system_path_soname_missing")
            continue
        deferred_system_paths.append(path)
    deferred_system_paths = sorted(set(deferred_system_paths))
    manifest = {
        "schema_version": RUNTIME_BUNDLE_MANIFEST_SCHEMA_VERSION,
        "generated_at": _string(generated_at),
        "status": "blocked" if blockers else "complete",
        "source_release_image_ref": source_ref or _string(source_release_image_ref),
        "carrier_image_ref": carrier_ref or _string(carrier_image_ref),
        "archive": {
            "path": DEFAULT_RUNTIME_ARCHIVE_PATH,
            "sha256": archive_digest or _string(archive_sha256),
            "size_bytes": archive_size_bytes,
            "format": "tar.gz",
            "member_roots": list(RUNTIME_ARCHIVE_ROOTS),
        },
        "healthcheck_argv": checks,
        "gpu_driver_deferred_sonames": deferred_sonames,
        "gpu_driver_deferred_system_paths": deferred_system_paths,
        "runtime_env": dict(RUNTIME_CARRIER_ENV),
        "blockers": sorted(set(blockers)),
        "claim_boundary": (
            "This manifest describes runtime bytes copied from an exact image. It does not "
            "prove S3 transfer, network-volume attachment, GPU execution, policy validity, "
            "learned-WAM validity, or semantic task success."
        ),
        "raw_secret_values_recorded": False,
    }
    return manifest


def verify_carrier_volume_admission(
    value: Mapping[str, Any],
    *,
    expected_carrier_image_ref: str = "",
) -> dict[str, Any]:
    """Validate storage, runtime, model, and immutable-image evidence together."""

    admission = _mapping(value)
    volume = _mapping(admission.get("network_volume"))
    runtime = _mapping(admission.get("runtime_bundle"))
    runtime_source = _mapping(admission.get("runtime_source_release"))
    model = _mapping(admission.get("model_cache"))
    transfer = _mapping(admission.get("s3_transfer_verification"))
    blockers: list[str] = []
    admission_schema = _string(admission.get("schema_version"))
    if admission_schema not in {
        CARRIER_VOLUME_ADMISSION_SCHEMA_VERSION,
        LEGACY_CARRIER_VOLUME_ADMISSION_SCHEMA_VERSION,
    }:
        blockers.append("carrier_volume_admission_schema_invalid")
    if admission.get("status") != "verified":
        blockers.append("carrier_volume_admission_not_verified")
    volume_id = _string(volume.get("id"))
    data_center_id = _string(volume.get("data_center_id")).upper()
    if not _VOLUME_ID.fullmatch(volume_id):
        blockers.append("carrier_network_volume_id_invalid")
    if not _DATA_CENTER_ID.fullmatch(data_center_id):
        blockers.append("carrier_network_volume_data_center_invalid")
    size_gib = volume.get("size_gib")
    if type(size_gib) is not int or size_gib < MIN_CARRIER_VOLUME_GIB:
        blockers.append("carrier_network_volume_below_120_gib")
    carrier_ref = _digest_pinned_image(admission.get("carrier_image_ref"))
    source_ref = _digest_pinned_image(runtime.get("source_release_image_ref"))
    if not carrier_ref:
        blockers.append("carrier_image_not_digest_pinned")
    if expected_carrier_image_ref and carrier_ref != expected_carrier_image_ref:
        blockers.append("carrier_image_admission_mismatch")
    if not source_ref:
        blockers.append("runtime_source_release_image_not_digest_pinned")
    if runtime_source.get("schema_version") != RUNTIME_SOURCE_RELEASE_VERIFICATION_SCHEMA_VERSION:
        blockers.append("runtime_source_release_verification_schema_invalid")
    if runtime_source.get("status") != "verified":
        blockers.append("runtime_source_release_evidence_not_verified")
    if runtime_source.get("release_image_ref") != source_ref:
        blockers.append("runtime_source_release_evidence_ref_mismatch")
    if not _COMMIT.fullmatch(_string(runtime_source.get("source_commit"))):
        blockers.append("runtime_source_release_evidence_commit_invalid")
    if not _sha256_digest(runtime_source.get("thin_release_contract_sha256")):
        blockers.append("runtime_source_release_contract_sha256_invalid")
    if runtime_source.get("models_externalized") is not True:
        blockers.append("runtime_source_release_models_not_externalized")
    if runtime.get("manifest_schema_version") != RUNTIME_BUNDLE_MANIFEST_SCHEMA_VERSION:
        blockers.append("runtime_bundle_manifest_schema_invalid")
    if runtime.get("root") != DEFAULT_RUNTIME_ROOT:
        blockers.append("runtime_bundle_root_invalid")
    if runtime.get("archive_path") != DEFAULT_RUNTIME_ARCHIVE_PATH:
        blockers.append("runtime_bundle_archive_path_invalid")
    if runtime.get("manifest_path") != DEFAULT_RUNTIME_MANIFEST_PATH:
        blockers.append("runtime_bundle_manifest_path_invalid")
    archive_digest = _sha256_digest(runtime.get("archive_sha256"))
    manifest_digest = _sha256_digest(runtime.get("manifest_sha256"))
    if not archive_digest:
        blockers.append("runtime_bundle_archive_sha256_invalid")
    if not manifest_digest:
        blockers.append("runtime_bundle_manifest_sha256_invalid")
    if model.get("status") != "verified":
        blockers.append("carrier_model_cache_not_verified")
    if model.get("root") != DEFAULT_MODEL_CACHE_ROOT:
        blockers.append("carrier_model_cache_root_invalid")
    model_manifest_digest = _sha256_digest(model.get("manifest_sha256"))
    if not model_manifest_digest:
        blockers.append("carrier_model_cache_manifest_sha256_invalid")
    model_content_digest = _string(model.get("manifest_digest"))
    if model_content_digest and not (
        model_content_digest.startswith("sha256:")
        and _sha256_digest(model_content_digest.removeprefix("sha256:"))
    ):
        blockers.append("carrier_model_cache_content_digest_invalid")
    if admission_schema == CARRIER_VOLUME_ADMISSION_SCHEMA_VERSION and not model_content_digest:
        blockers.append("carrier_model_cache_content_digest_missing_from_v3")
    if transfer.get("upload_completed") is not True:
        blockers.append("carrier_volume_s3_upload_not_complete")
    if transfer.get("full_redownload_sha256_verified") is not True:
        blockers.append("carrier_volume_s3_full_redownload_not_verified")
    if transfer.get("provider_volume_id") != volume_id:
        blockers.append("carrier_volume_s3_volume_binding_mismatch")
    if _string(transfer.get("data_center_id")).upper() != data_center_id:
        blockers.append("carrier_volume_s3_data_center_binding_mismatch")
    return {
        "schema_version": admission_schema or CARRIER_VOLUME_ADMISSION_SCHEMA_VERSION,
        "status": "blocked" if blockers else "verified",
        "blockers": sorted(set(blockers)),
        "network_volume_id": volume_id,
        "data_center_id": data_center_id,
        "size_gib": size_gib,
        "carrier_image_ref": carrier_ref,
        "source_release_image_ref": source_ref,
        "source_release_commit": _string(runtime_source.get("source_commit")),
        "source_release_contract_sha256": _sha256_digest(
            runtime_source.get("thin_release_contract_sha256")
        ),
        "runtime_root": DEFAULT_RUNTIME_ROOT,
        "runtime_archive_path": DEFAULT_RUNTIME_ARCHIVE_PATH,
        "runtime_manifest_path": DEFAULT_RUNTIME_MANIFEST_PATH,
        "runtime_archive_sha256": archive_digest,
        "runtime_manifest_sha256": manifest_digest,
        "model_cache_root": DEFAULT_MODEL_CACHE_ROOT,
        "model_cache_manifest_path": DEFAULT_MODEL_CACHE_MANIFEST_PATH,
        "model_manifest_sha256": model_manifest_digest,
        "model_manifest_digest": model_content_digest or None,
        "requires_external_model_manifest_digest_binding": not bool(model_content_digest),
        "claim_boundary": (
            "This admission proves a digest-bound request may attach preverified volume bytes. "
            "The carrier must still reverify and activate them after container start. It does "
            "not prove provider attachment, inference, rollout quality, or semantic success."
        ),
        "raw_secret_values_recorded": False,
    }


def runtime_bootstrap_shell_prefix() -> str:
    """Return a secret-free bootstrap that validates and activates the runtime archive."""

    runtime_exports = "\n".join(
        f"export {key}={shlex.quote(value)}" for key, value in RUNTIME_CARRIER_ENV.items()
    )
    script = runtime_exports + r"""
echo BLUEPRINT_RUNPOD_CARRIER_RUNTIME_BOOTSTRAP_STARTED
mkdir -p "$WORK_DIR/runtime_output"
export BLUEPRINT_RUNPOD_CARRIER_BOOTSTRAP_RESULT="$WORK_DIR/runtime_output/runpod_carrier_runtime_bootstrap.json"
export PYTHONUNBUFFERED=1
export PIP_NO_CACHE_DIR=1
export MUJOCO_GL=osmesa
export BLUEPRINT_GROOT_OSCAR_REQUIRED_CUDA_VERSION=12.8
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export BLUEPRINT_GROOT_OSCAR_OSCAR_REPO=/opt/OSCAR
export BLUEPRINT_GROOT_OSCAR_GROOT_VENV_PYTHON=/opt/gr00t-venv/bin/python
export BLUEPRINT_GROOT_OSCAR_GROOT_ROOT=/opt/gr00t
export BLUEPRINT_GEAR_SONIC_ROOT=/opt/wbc
export BLUEPRINT_GEAR_SONIC_ROBOT_MODEL=/opt/wbc/gear_sonic_deploy/g1/g1_29dof_with_hand.xml
export BLUEPRINT_GEAR_SONIC_EXECUTOR_COMMAND="/opt/oscar-venv/bin/python -m blueprint_pipeline.gear_sonic_official_zmq_executor"
export BLUEPRINT_ISAAC_PYTHON=/isaac-sim/python.sh
export BLUEPRINT_ISAAC_UNITREE_G1_USD=/isaac-sim/Isaac/Robots/Unitree/G1/g1.usd
if ! python - <<'PY'
import ctypes
import hashlib
import json
import os
import subprocess
import tarfile
import posixpath
from pathlib import Path, PurePosixPath

result_path = Path(os.environ["BLUEPRINT_RUNPOD_CARRIER_BOOTSTRAP_RESULT"])
archive_path = Path(os.environ["BLUEPRINT_RUNTIME_ARCHIVE_PATH"])
manifest_path = Path(os.environ["BLUEPRINT_RUNTIME_MANIFEST_PATH"])
allowed_roots = tuple(filter(None, os.environ["BLUEPRINT_RUNTIME_ARCHIVE_ROOTS"].split(":")))

def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

result = {
    "schema_version": "groot_oscar_runpod_carrier_runtime_bootstrap.v1",
    "status": "blocked",
    "blockers": [],
    "raw_secret_values_recorded": False,
}
try:
    if digest(manifest_path) != os.environ["BLUEPRINT_RUNTIME_MANIFEST_SHA256"]:
        raise RuntimeError("runtime_manifest_sha256_mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "groot_oscar_runtime_bundle_manifest.v2":
        raise RuntimeError("runtime_manifest_schema_invalid")
    expected_runtime_env = {
        key: os.environ.get(key) for key in ("PYTHONPATH", "LD_LIBRARY_PATH")
    }
    if manifest.get("runtime_env") != expected_runtime_env:
        raise RuntimeError("runtime_manifest_env_mismatch")
    driver_sonames = manifest.get("gpu_driver_deferred_sonames")
    driver_stems = (
        "libcuda.so",
        "libnvcuvid.so",
        "libnvoptix.so",
        "libGLX_nvidia.so",
        "libEGL_nvidia.so",
        "libGLESv1_CM_nvidia.so",
        "libGLESv2_nvidia.so",
    )
    def driver_soname_allowed(value):
        return (
            isinstance(value, str)
            and value
            and len(value) <= 256
            and not any(character.isspace() or character in "/\\\x00" for character in value)
            and (
                (value.startswith("libnvidia-") and ".so" in value)
                or any(value == stem or value.startswith(stem + ".") for stem in driver_stems)
            )
        )
    if not isinstance(driver_sonames, list) or any(
        not driver_soname_allowed(value) for value in driver_sonames
    ):
        raise RuntimeError("runtime_gpu_driver_deferred_sonames_invalid")
    # This field was added to v2 manifests after the schema was already in use.
    # Absence means the older producer deferred no exact driver paths.
    driver_system_paths = manifest.get("gpu_driver_deferred_system_paths", [])
    driver_system_prefixes = (
        "/lib/",
        "/lib64/",
        "/usr/lib/",
        "/usr/lib64/",
        "/usr/local/cuda/compat/",
        "/usr/local/nvidia/",
    )
    def driver_system_path_allowed(value):
        return (
            isinstance(value, str)
            and value
            and len(value) <= 512
            and value.startswith("/")
            and not any(character.isspace() or character in "\\\x00" for character in value)
            and posixpath.normpath(value) == value
            and any(value.startswith(prefix) for prefix in driver_system_prefixes)
            and driver_soname_allowed(posixpath.basename(value))
            and posixpath.basename(value) in driver_sonames
        )
    if not isinstance(driver_system_paths, list) or any(
        not driver_system_path_allowed(value) for value in driver_system_paths
    ):
        raise RuntimeError("runtime_gpu_driver_deferred_system_paths_invalid")
    if digest(archive_path) != os.environ["BLUEPRINT_RUNTIME_ARCHIVE_SHA256"]:
        raise RuntimeError("runtime_archive_sha256_mismatch")
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            normalized = name.as_posix()
            while normalized.startswith("./"):
                normalized = normalized[2:]
            if name.is_absolute() or ".." in name.parts:
                raise RuntimeError("runtime_archive_unsafe_member")
            if not any(normalized == root or normalized.startswith(root + "/") for root in allowed_roots):
                raise RuntimeError("runtime_archive_member_outside_allowlist")
            if member.ischr() or member.isblk() or member.isfifo():
                raise RuntimeError("runtime_archive_special_member_disallowed")
            if member.issym() or member.islnk():
                target = member.linkname
                if member.islnk():
                    if target.startswith("/"):
                        raise RuntimeError("runtime_archive_unsafe_hardlink")
                    resolved_target = posixpath.normpath(target)
                elif target.startswith("/"):
                    resolved_target = posixpath.normpath(target).lstrip("/")
                else:
                    resolved_target = posixpath.normpath(
                        posixpath.join(posixpath.dirname(normalized), target)
                    )
                if resolved_target == ".." or resolved_target.startswith("../"):
                    raise RuntimeError("runtime_archive_unsafe_link")
                if not any(
                    resolved_target == root or resolved_target.startswith(root + "/")
                    for root in allowed_roots
                ):
                    raise RuntimeError("runtime_archive_link_outside_allowlist")
        archive.extractall(path="/", filter=lambda member, _path: member)
    for soname in driver_sonames:
        try:
            ctypes.CDLL(soname)
        except OSError as exc:
            raise RuntimeError("runtime_gpu_driver_soname_unresolved") from exc
    for system_path in driver_system_paths:
        try:
            ctypes.CDLL(system_path)
        except OSError as exc:
            raise RuntimeError("runtime_gpu_driver_system_path_unresolved") from exc
    for argv in manifest.get("healthcheck_argv", []):
        healthcheck_timeout = 300 if argv[0] == "/isaac-sim/python.sh" else 120
        completed = subprocess.run(
            argv,
            check=False,
            timeout=healthcheck_timeout,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError("runtime_healthcheck_failed")
    wbc_binary = Path("/opt/wbc/gear_sonic_deploy/target/release/g1_deploy_onnx_ref")
    linkage = subprocess.run(
        ["ldd", str(wbc_binary)], check=False, timeout=120, capture_output=True, text=True
    )
    if linkage.returncode != 0 or "not found" in linkage.stdout:
        raise RuntimeError("runtime_wbc_dynamic_linkage_failed")
    model_root = Path(os.environ["BLUEPRINT_MODEL_CACHE_ROOT"])
    model_manifest_path = Path(os.environ["BLUEPRINT_MODEL_CACHE_MANIFEST_PATH"])
    if digest(model_manifest_path) != os.environ["BLUEPRINT_MODEL_CACHE_MANIFEST_SHA256"]:
        raise RuntimeError("model_cache_manifest_sha256_mismatch")
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if model_manifest.get("schema_version") != "groot_oscar_external_model_cache.v2":
        raise RuntimeError("model_cache_manifest_schema_invalid")
    declared_model_paths = set()
    for row in model_manifest.get("files", []):
        relative = PurePosixPath(str(row.get("path") or ""))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise RuntimeError("model_cache_manifest_file_path_invalid")
        relative_text = relative.as_posix()
        if relative_text in declared_model_paths:
            raise RuntimeError("model_cache_manifest_file_path_duplicate")
        declared_model_paths.add(relative_text)
        model_path = model_root.joinpath(*relative.parts)
        if not model_path.is_file():
            raise RuntimeError("model_cache_declared_file_missing")
        if model_path.stat().st_size != int(row.get("size_bytes") or -1):
            raise RuntimeError("model_cache_declared_file_size_mismatch")
        if digest(model_path) != str(row.get("sha256") or ""):
            raise RuntimeError("model_cache_declared_file_sha256_mismatch")
    if not declared_model_paths:
        raise RuntimeError("model_cache_manifest_file_inventory_missing")
    gear_links = {
        "gear_sonic/model_encoder.onnx": "/opt/wbc/gear_sonic_deploy/policy/release/model_encoder.onnx",
        "gear_sonic/model_decoder.onnx": "/opt/wbc/gear_sonic_deploy/policy/release/model_decoder.onnx",
        "gear_sonic/observation_config.yaml": "/opt/wbc/gear_sonic_deploy/policy/release/observation_config.yaml",
        "gear_sonic/planner_sonic.onnx": "/opt/wbc/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx",
    }
    for source_relative, destination_text in gear_links.items():
        source = model_root / source_relative
        destination = Path(destination_text)
        if not source.is_file() or destination.parent.is_symlink():
            raise RuntimeError("model_cache_gear_sonic_link_source_or_parent_invalid")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source)
    result.update({
        "status": "ready",
        "blockers": [],
        "runtime_archive_sha256_verified": True,
        "runtime_manifest_sha256_verified": True,
        "model_manifest_sha256_verified": True,
        "all_declared_model_file_sha256_verified": True,
        "declared_model_file_count": len(declared_model_paths),
        "external_model_cache_bound_to_runtime": True,
        "runtime_healthchecks_passed": True,
        "gpu_driver_deferred_sonames_resolved": True,
        "gpu_driver_deferred_soname_count": len(driver_sonames),
        "gpu_driver_deferred_system_paths_resolved": True,
        "gpu_driver_deferred_system_path_count": len(driver_system_paths),
    })
except Exception as exc:
    result["blockers"] = [str(exc)]
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise
result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
then
  python - <<'PY'
import os
import urllib.request
import zipfile
from pathlib import Path
result = Path(os.environ["BLUEPRINT_RUNPOD_CARRIER_BOOTSTRAP_RESULT"])
archive = result.parent / "runpod_carrier_runtime_bootstrap_blocked.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
    output.write(result, "runpod_carrier_runtime_bootstrap.json")
target = (
    os.environ.get("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL")
    or os.environ.get("BLUEPRINT_ARTIFACT_OUTPUT_URI")
)
if target:
    request = urllib.request.Request(
        target,
        data=archive.read_bytes(),
        method="PUT",
        headers={"Content-Type": "application/zip"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        response.read()
PY
  exit 86
fi
export BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT="/opt/gr00t"
export BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT="/opt/wbc"
export BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT="$BLUEPRINT_MODEL_CACHE_ROOT/sonic"
export BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT="$BLUEPRINT_MODEL_CACHE_ROOT/sonic"
export BLUEPRINT_GROOT_OSCAR_SONIC_CHECKPOINT="$BLUEPRINT_MODEL_CACHE_ROOT/sonic"
export BLUEPRINT_GROOT_OSCAR_OSCAR_CHECKPOINT="$BLUEPRINT_MODEL_CACHE_ROOT/oscar"
export BLUEPRINT_OSCAR_WAM_CHECKPOINT="$BLUEPRINT_MODEL_CACHE_ROOT/oscar"
echo BLUEPRINT_RUNPOD_CARRIER_RUNTIME_BOOTSTRAP_READY
"""
    return script
