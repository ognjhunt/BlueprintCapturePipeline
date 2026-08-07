"""Immutable OVRTX bundle and canonical capped Vast transport for ADP-009D."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import zipfile
from pathlib import Path
from typing import Any, Mapping

from .adp_isaac_lab_arena_vast import run_arena_native_control_vast
from .common import ensure_dir, write_json
from .decision_evidence_contracts import canonical_digest
from .paid_resource_admission import PaidResourceAdmissionGrant


PROBE_KIND = "adp009d-aura-ovrtx-live-camera"
PROVIDER_BUNDLE_KIND = "adp009d_ovrtx"
RESULT_SCHEMA_VERSION = "adp009d_ovrtx_vast_run.v1"
DEFAULT_IMAGE = (
    "docker.io/nvidia/cuda@"
    "sha256:cff3a0d82d2c2b47bab252d67fa9b34a20ef4c50781d98501b5c7367ea9afd10"
)
OVRTX_REVISION = "4b9a5fe6f8becf6c5ff031e167cd4201054a96ce"
OVRTX_VERSION = "0.4.0.346409"
OVSTAGE_VERSION = "0.1.0.346039"
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/adp009d-aura-ovrtx"

ENTRYPOINT = r'''#!/usr/bin/env bash
set -u
RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${BLUEPRINT_ADP009D_OVRTX_OUTPUT_DIR:-${RUNTIME_DIR}/../runtime_output}"
RESULT_PATH="${OUTPUT_DIR}/adp009d_ovrtx_live_camera_result.json"
mkdir -p "${OUTPUT_DIR}"
write_missing_result() {
  local blocker="$1"
  python3 - "${RESULT_PATH}" "${blocker}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    path.write_text(json.dumps({
        "schema_version": "adp009d_ovrtx_live_camera_result.v1",
        "status": "blocked",
        "blockers": [sys.argv[2], "adp009d_ovrtx_runner_failed_without_runtime_result"],
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "provider_zero_required_after_return": True
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}
python3 -m pip install --disable-pip-version-check --no-cache-dir uv==0.10.7
if [ $? -ne 0 ]; then write_missing_result "adp009d_ovrtx_uv_install_failed"; exit 2; fi
UV_BIN="$(command -v uv)"
"${UV_BIN}" python install 3.12
if [ $? -ne 0 ]; then write_missing_result "adp009d_ovrtx_python312_install_failed"; exit 2; fi
OVRTX_ENV="${RUNTIME_DIR}/.ovrtx_venv"
"${UV_BIN}" venv "${OVRTX_ENV}" --python 3.12
if [ $? -ne 0 ]; then write_missing_result "adp009d_ovrtx_venv_failed"; exit 2; fi
"${UV_BIN}" pip install --python "${OVRTX_ENV}/bin/python" --extra-index-url https://pypi.nvidia.com \
  "ovrtx==0.4.0.346409" "ovstage==0.1.0.346039" "numpy>=1.26,<3" \
  "Pillow>=10,<13" "nvidia-ml-py>=12,<14"
if [ $? -ne 0 ]; then write_missing_result "adp009d_ovrtx_dependency_install_failed"; exit 2; fi
"${OVRTX_ENV}/bin/python" -c 'import importlib.metadata as m; import ovrtx, ovstage; assert m.version("ovrtx") == "0.4.0.346409" and m.version("ovstage") == "0.1.0.346039"'
if [ $? -ne 0 ]; then write_missing_result "adp009d_ovrtx_dependency_identity_failed"; exit 2; fi
if ! ldconfig -p | grep -q 'libGLX_nvidia.so.0'; then
  write_missing_result "adp009d_ovrtx_libglx_nvidia_missing"
  exit 2
fi
if ! command -v xvfb-run >/dev/null 2>&1 || ! command -v vulkaninfo >/dev/null 2>&1; then
  write_missing_result "adp009d_ovrtx_headless_graphics_tools_missing"
  exit 2
fi
xvfb-run -a -s '-screen 0 1280x720x24' sh -c \
  'vulkaninfo --summary >"$1/ovrtx_vulkan_preflight.log" 2>&1' sh "${OUTPUT_DIR}"
if [ $? -ne 0 ]; then write_missing_result "adp009d_ovrtx_vulkan_preflight_failed"; exit 2; fi
xvfb-run -a -s '-screen 0 1280x720x24' python3 "${RUNTIME_DIR}/adp009d_ovrtx_provider_runner.py" \
  --runtime-dir "${RUNTIME_DIR}" --output-dir "${OUTPUT_DIR}" \
  --ovrtx-python "${OVRTX_ENV}/bin/python"
runner_rc=$?
if [ ${runner_rc} -ne 0 ] && [ ! -f "${RESULT_PATH}" ]; then
  write_missing_result "adp009d_ovrtx_runner_failed_without_runtime_result"
fi
exit ${runner_rc}
'''


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def build_ovrtx_live_camera_bundle(
    *,
    job_dir: str | Path,
    probe_manifest_path: str | Path,
    implementation_commit: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    if len(implementation_commit) != 40 or any(
        char not in "0123456789abcdef" for char in implementation_commit
    ):
        raise ValueError("adp009d_ovrtx_implementation_commit_invalid")
    job = Path(job_dir).resolve()
    if job.exists():
        shutil.rmtree(job)
    runtime = job / "provider_runtime"
    assets = runtime / "assets"
    configs = runtime / "configs"
    ensure_dir(assets)
    ensure_dir(configs)
    probe_path = Path(probe_manifest_path).resolve()
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    if (
        probe.get("schema_version") != "adp009d_ovrtx_live_camera_probe.v1"
        or probe.get("status") != "materialized_unexecuted"
        or probe.get("manifest_digest")
        != canonical_digest(probe, digest_field="manifest_digest")
        or probe.get("metric_depth_aov") != "DistanceToCameraSD"
        or probe.get("unitless_depth_sd_used") is not False
    ):
        raise ValueError("adp009d_ovrtx_probe_manifest_invalid")
    particlefield = Path(probe["particlefield_path"]).resolve()
    if _sha256(particlefield) != probe.get("particlefield_sha256"):
        raise ValueError("adp009d_ovrtx_particlefield_digest_mismatch")
    shutil.copy2(particlefield, assets / "aura_gaussian_surflets.usdc")
    config_rows = []
    for row in probe.get("camera_configs", []):
        camera_id = str(row.get("camera_id"))
        if camera_id not in {"external", "wrist"}:
            raise ValueError("adp009d_ovrtx_camera_id_invalid")
        source = Path(row["configuration_path"]).resolve()
        if _sha256(source) != row.get("configuration_sha256"):
            raise ValueError("adp009d_ovrtx_camera_config_digest_mismatch")
        destination = configs / f"{camera_id}.ovrtx.json"
        shutil.copy2(source, destination)
        config_rows.append(
            {
                "camera_id": camera_id,
                "configuration_sha256": _sha256(destination),
            }
        )
    if {row["camera_id"] for row in config_rows} != {"external", "wrist"}:
        raise ValueError("adp009d_ovrtx_camera_set_invalid")
    source_dir = Path(__file__).resolve().parent
    repo_root = source_dir.parents[1]
    shutil.copy2(
        source_dir / "adp009d_ovrtx_provider_runner.py",
        runtime / "adp009d_ovrtx_provider_runner.py",
    )
    shutil.copy2(
        repo_root / "scripts/run_ovrtx_preflight_worker.py",
        runtime / "run_ovrtx_preflight_worker.py",
    )
    _write_executable(runtime / "run_adp009d_ovrtx_provider_runtime.sh", ENTRYPOINT)
    manifest: dict[str, Any] = {
        "schema_version": "adp009d_ovrtx_provider_manifest.v1",
        "status": "ready",
        "program_id": "arm-decision-proof-v1",
        "probe_kind": PROBE_KIND,
        "implementation_commit": implementation_commit,
        "container_image": DEFAULT_IMAGE,
        "ovrtx_repository": "https://github.com/NVIDIA-Omniverse/ovrtx",
        "ovrtx_revision": OVRTX_REVISION,
        "ovrtx_version": OVRTX_VERSION,
        "ovstage_version": OVSTAGE_VERSION,
        "particlefield_sha256": _sha256(assets / "aura_gaussian_surflets.usdc"),
        "camera_configs": sorted(config_rows, key=lambda row: row["camera_id"]),
        "metric_depth_aov": "DistanceToCameraSD",
        "rtpt_warmup_frames": 40,
        "headless_graphics_backend": "xvfb",
        "vulkan_preflight_required": True,
        "retry_cap": 0,
        "candidate_policy_queried": False,
        "private_data_uploaded": False,
        "provider_zero_required_after_return": True,
        "blockers": [],
    }
    if generated_at is not None:
        manifest["generated_at"] = generated_at
    manifest["input_digest"] = canonical_digest(manifest, digest_field="input_digest")
    write_json(runtime / "adp009d_ovrtx_provider_manifest.json", manifest)
    bundle_path = job / "adp009d_ovrtx_live_camera_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", allowZip64=True) as archive:
        for path in sorted(runtime.rglob("*")):
            if path.is_file():
                info = zipfile.ZipInfo(
                    path.relative_to(job).as_posix(), date_time=(1980, 1, 1, 0, 0, 0)
                )
                info.create_system = 3
                info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
                archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_STORED)
    receipt = {
        **manifest,
        "bundle_path": str(bundle_path),
        "bundle_sha256": _sha256(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
    }
    write_json(job / "adp009d_ovrtx_live_camera_bundle_receipt.json", receipt)
    return receipt


def run_ovrtx_live_camera_vast(
    *,
    job_dir: str | Path,
    prepared_bundle: Mapping[str, Any],
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    machine_avoidlist_path: str | Path | None = None,
    max_hourly_rate_usd: float = 1.0,
    hard_cap_usd: float = 1.25,
    hard_ttl_seconds: int = 2700,
) -> dict[str, Any]:
    return run_arena_native_control_vast(
        approval_path=".",
        job_dir=job_dir,
        paid_resource_admission_grant=paid_resource_admission_grant,
        execute=execute,
        prepared_bundle=prepared_bundle,
        machine_avoidlist_path=machine_avoidlist_path,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        expected_output_filename="adp009d_ovrtx_live_camera_result.json",
        container_image=DEFAULT_IMAGE,
        provider_bundle_kind=PROVIDER_BUNDLE_KIND,
        result_schema_version=RESULT_SCHEMA_VERSION,
        object_store_key_prefix=DEFAULT_KEY_PREFIX,
        instance_label_prefix="blueprint-adp009d-ovrtx-",
        blocker_prefix="adp009d_ovrtx",
        min_gpu_ram_mb=46_000,
        minimum_driver_version="580.95.05",
        preferred_gpu_keywords=("L40S", "RTX 6000 Ada", "RTX A6000"),
    )


__all__ = [
    "PROBE_KIND",
    "build_ovrtx_live_camera_bundle",
    "run_ovrtx_live_camera_vast",
]
