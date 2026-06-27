"""Provider-neutral WAM compute wrappers.

This module keeps the WAM compute contract stable while delegating provider-specific
launch, poll, teardown, URL handling, and dud detection to the existing RunPod and Vast
async runners. Generated WAM media remains a downstream support artifact; a completed
provider run here does not prove capture truth, visual usefulness, collision truth,
safety validation, physical readiness, or deployment approval.
"""

from __future__ import annotations

import os
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .runpod_provider_adapter import (
    RUNPOD_API_GATE_ENV,
    RUNPOD_API_KEY_ENV,
    RUNPOD_API_KEY_FILE_ENV,
)
from .runpod_wam_async_runner import (
    DEFAULT_GPU_TYPE_IDS as RUNPOD_DEFAULT_GPU_TYPE_IDS,
    RUNPOD_POD_LAUNCH_GATE_ENV,
    create_runpod_wam_async_run,
    poll_runpod_wam_async_run,
)
from .vast_provider_adapter import _inspect_provider_runtime_output_zip
from .vast_wam_async_runner import (
    DEFAULT_HARD_CAP_USD,
    DEFAULT_MAX_HOURLY_RATE,
    DEFAULT_TARGET_SPEND_USD,
    DEFAULT_WAM_PUBLIC_IMAGE,
    destroy_async_vast_wam_run,
    create_async_vast_wam_run,
    poll_async_vast_wam_run,
)


SCHEMA_VERSION = "wam_compute_providers.v1"
RESULT_SCHEMA_VERSION = "wam_compute_run_result.v1"
DEFAULT_PROVIDER_ORDER = ("runpod", "vast")
PROVIDER_ORDER_ENV = "BLUEPRINT_WAM_COMPUTE_PROVIDER_ORDER"
VAST_WAM_PAID_LAUNCH_GATE_ENV = "BLUEPRINT_ALLOW_PAID_VAST_WAM_PROVIDER_LAUNCH"
PROVIDER_BUNDLE_KINDS = ("wam", "unitree_unifolm", "unitree_groot_n17_sonic")


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value if str(item)]
    return []


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _env_truthy(name: str) -> bool:
    return _string(os.getenv(name)).lower() in {"1", "true", "yes", "on", "y"}


def _existing_path(*candidates: Path) -> str | None:
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


@contextmanager
def _temporary_env(env: Mapping[str, str] | None):
    previous: dict[str, str | None] = {}
    try:
        for key, value in (env or {}).items():
            text = _string(value)
            if not text:
                continue
            previous[key] = os.environ.get(key)
            os.environ[key] = text
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@dataclass(frozen=True)
class WamComputeLaunchSpec:
    """Provider-neutral WAM compute launch request."""

    name: str
    bundle_path: str | Path
    provider_bundle_kind: str = "wam"
    image: str = DEFAULT_WAM_PUBLIC_IMAGE
    env: Mapping[str, str] = field(default_factory=dict)
    public_base_url: str = ""
    provider_bundle_url: str = ""
    provider_output_put_url: str = ""
    provider_output_get_url: str = ""
    provider_bundle_url_file: str | Path | None = None
    provider_output_put_url_file: str | Path | None = None
    provider_output_get_url_file: str | Path | None = None
    token_file: str | Path | None = None
    secret_env_file: str | Path | None = None
    output_zip_path: str | Path | None = None
    expected_video_count: int = 1
    max_wait_seconds: int = 60
    retry_interval_seconds: int = 5
    gpu_type_ids: Sequence[str] = RUNPOD_DEFAULT_GPU_TYPE_IDS
    cloud_type: str = "SECURE"
    allowed_cuda_versions: Sequence[str] = ()
    container_disk_gb: int = 80
    volume_gb: int = 20
    min_vcpu_per_gpu: int = 2
    min_ram_per_gpu: int = 8
    max_hourly_rate_usd: float = DEFAULT_MAX_HOURLY_RATE
    target_spend_usd: float = DEFAULT_TARGET_SPEND_USD
    hard_cap_usd: float = DEFAULT_HARD_CAP_USD
    allow_target_spend_overrun: bool = False
    max_live_minutes: int = 30
    session_max_live_minutes: int | None = 45
    min_gpu_ram_mb: int = 0
    excluded_machine_ids: Sequence[int] = ()
    startup_poll_seconds: int = 90
    public_staging_verify_max_wait_seconds: int = 120
    public_staging_verify_retry_interval_seconds: float = 5.0
    public_staging_verify_timeout_seconds: float = 20.0
    public_staging_required_consecutive_successes: int = 2
    verify_output_put_url: bool = False
    skip_public_staging_verification: bool = False
    vast_launch_mode: str = "ssh_direct"
    disk_gb: int = 80
    heartbeat_url: str = ""
    claim_boundary: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.provider_bundle_kind not in PROVIDER_BUNDLE_KINDS:
            raise ValueError(f"unsupported_provider_bundle_kind:{self.provider_bundle_kind}")
        if int(self.expected_video_count) < 0:
            raise ValueError("expected_video_count_must_be_non_negative")

    def output_path_for(self, provider: str, job_dir: Path) -> Path:
        if self.output_zip_path:
            return Path(self.output_zip_path).expanduser().resolve()
        if provider == "runpod":
            return job_dir / "runpod_provider_runtime_output.zip"
        if provider == "vast":
            return job_dir / "vast_provider_runtime_output.zip"
        return job_dir / "provider_runtime_output.zip"


@dataclass
class WamComputeRunResult:
    provider: str
    status: str
    provider_command_status: str = ""
    instance_id: str | None = None
    output_zip_path: str | None = None
    output_zip_present: bool = False
    mp4_count: int = 0
    extracted_video_paths: list[str] = field(default_factory=list)
    runtime_result_status: str = ""
    runtime_result_blockers: list[str] = field(default_factory=list)
    phase_log_path: str | None = None
    budget_ledger_path: str | None = None
    teardown_manifest_path: str | None = None
    teardown_status: str = "not_requested"
    teardown_performed: bool = False
    continuing_spend_from_this_run: bool = False
    blockers: list[str] = field(default_factory=list)
    provider_phase: str = ""
    output_availability: str = "not_available"
    raw_secret_values_recorded: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": self.details.get("generated_at") or utc_now_iso(),
            "provider": self.provider,
            "status": self.status,
            "provider_command_status": self.provider_command_status,
            "instance_id": self.instance_id,
            "output_zip_path": self.output_zip_path,
            "output_zip_present": self.output_zip_present,
            "mp4_count": self.mp4_count,
            "extracted_video_paths": list(self.extracted_video_paths),
            "runtime_result_status": self.runtime_result_status,
            "runtime_result_blockers": list(self.runtime_result_blockers),
            "phase_log_path": self.phase_log_path,
            "budget_ledger_path": self.budget_ledger_path,
            "teardown_manifest_path": self.teardown_manifest_path,
            "teardown_status": self.teardown_status,
            "teardown_performed": self.teardown_performed,
            "continuing_spend_from_this_run": self.continuing_spend_from_this_run,
            "blockers": sorted(set(self.blockers)),
            "provider_phase": self.provider_phase,
            "output_availability": self.output_availability,
            "raw_secret_values_recorded": False,
            "claim_boundary": {
                "wam_compute_result_is_provider_package_runtime_state_only": True,
                "generated_video_success_is_not_visual_usefulness": True,
                "generated_world_rank_fidelity_result_proven": False,
                "capture_truth": False,
                "collision_truth": False,
                "safety_validation": False,
                "physical_robot_readiness": False,
                "deployment_approval": False,
            },
            "details": dict(self.details),
        }


class WamComputeProvider:
    name = "base"

    def available(self) -> dict[str, Any]:
        raise NotImplementedError

    def build_request(self, spec: WamComputeLaunchSpec, job_dir: Path) -> dict[str, Any]:
        raise NotImplementedError

    def create(
        self,
        spec: WamComputeLaunchSpec,
        job_dir: Path,
        *,
        allow_paid_launch: bool,
    ) -> WamComputeRunResult:
        raise NotImplementedError

    def poll(
        self,
        job_dir: Path,
        *,
        max_wait_seconds: int,
        teardown: bool,
    ) -> WamComputeRunResult:
        raise NotImplementedError

    def teardown(self, job_dir: Path, instance_id: str | None = None) -> WamComputeRunResult:
        raise NotImplementedError

    def inspect_output(self, job_dir: Path, output_zip_path: str | Path) -> dict[str, Any]:
        output_path = Path(output_zip_path).expanduser().resolve()
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            return {
                "status": "not_available",
                "zip_present": False,
                "output_zip_path": str(output_path),
                "mp4_count": 0,
                "extracted_video_paths": [],
                "runtime_result_status": "not_available",
                "runtime_result_blockers": ["provider_runtime_output_zip_missing_or_empty"],
                "raw_secret_values_recorded": False,
            }
        if not zipfile.is_zipfile(output_path):
            return {
                "status": "blocked",
                "zip_present": False,
                "output_zip_path": str(output_path),
                "mp4_count": 0,
                "extracted_video_paths": [],
                "runtime_result_status": "blocked",
                "runtime_result_blockers": ["provider_runtime_output_zip_invalid"],
                "raw_secret_values_recorded": False,
            }
        inspected = _inspect_provider_runtime_output_zip(
            output_path,
            video_extract_dir=job_dir / f"{self.name}_wam_compute_output_videos",
            expected_video_count=0,
        )
        return {
            **inspected,
            "status": "completed" if inspected.get("zip_present") else "not_available",
            "output_zip_path": str(output_path),
            "raw_secret_values_recorded": False,
        }

    def _blocked_no_paid_launch(
        self,
        *,
        job_dir: Path,
        spec: WamComputeLaunchSpec,
        blockers: Sequence[str],
    ) -> WamComputeRunResult:
        result = WamComputeRunResult(
            provider=self.name,
            status="blocked",
            provider_command_status="blocked",
            output_zip_path=str(spec.output_path_for(self.name, job_dir)),
            blockers=list(blockers),
            provider_phase="prelaunch_paid_gate",
            output_availability="not_available",
            continuing_spend_from_this_run=False,
            details={
                "generated_at": utc_now_iso(),
                "job_dir": str(job_dir),
                "request": self.build_request(spec, job_dir),
                "raw_secret_values_recorded": False,
            },
        )
        write_json(job_dir / "wam_compute_run_result.json", result.to_dict())
        return result


def _output_availability(
    *,
    output_zip_present: bool,
    mp4_count: int,
    expected_video_count: int,
    runtime_result_status: str,
) -> str:
    if not output_zip_present:
        return "not_available"
    if expected_video_count > 0 and mp4_count < expected_video_count:
        return "zip_present_but_expected_generated_videos_missing"
    if runtime_result_status and runtime_result_status not in {"completed", "not_available"}:
        return "zip_present_but_runtime_not_completed"
    return "available"


def _result_from_manifest(
    *,
    provider: str,
    job_dir: Path,
    manifest: Mapping[str, Any],
    spec: WamComputeLaunchSpec | None = None,
    output_inspection: Mapping[str, Any] | None = None,
) -> WamComputeRunResult:
    expected_video_count = int(spec.expected_video_count if spec else 0)
    details = {
        "generated_at": manifest.get("generated_at") or utc_now_iso(),
        "job_dir": str(job_dir),
    }
    output_path = (
        _string(manifest.get("provider_runtime_output_zip_path"))
        or _string(manifest.get("output_path"))
        or (str(spec.output_path_for(provider, job_dir)) if spec else "")
    )
    instance_id = (
        _string(manifest.get("instance_id"))
        or _string(manifest.get("pod_id"))
        or _string((manifest.get("vast_instance_ids") or [None])[0])
    )
    inspection = dict(output_inspection or {})
    mp4_count = int(inspection.get("mp4_count") or manifest.get("mp4_count") or 0)
    output_zip_present = bool(
        inspection.get("zip_present")
        if "zip_present" in inspection
        else manifest.get("output_zip_present")
    )
    runtime_status = _string(
        inspection.get("runtime_result_status") or manifest.get("runtime_result_status")
    )
    runtime_blockers = _string_list(
        inspection.get("runtime_result_blockers") or manifest.get("runtime_result_blockers")
    )
    provider_blockers = _string_list(
        manifest.get("provider_command_blockers") or manifest.get("blockers")
    )
    extracted_video_paths = _string_list(inspection.get("extracted_video_paths"))
    if not extracted_video_paths:
        validation_files = _mapping(inspection.get("mp4_validation")).get("files")
        if isinstance(validation_files, Sequence) and not isinstance(
            validation_files,
            (bytes, bytearray, str),
        ):
            extracted_video_paths = [
                _string(_mapping(row).get("path") or _mapping(row).get("file"))
                for row in validation_files
                if _string(_mapping(row).get("path") or _mapping(row).get("file"))
            ]
    output_available = _output_availability(
        output_zip_present=output_zip_present,
        mp4_count=mp4_count,
        expected_video_count=expected_video_count,
        runtime_result_status=runtime_status,
    )
    blockers = list(provider_blockers)
    if output_zip_present and output_available != "available":
        blockers.append(output_available)
    if not output_zip_present and _string(manifest.get("status")) == "completed":
        blockers.append("provider_completed_without_valid_output_zip")
    manifest_status = _string(manifest.get("status")) or "blocked"
    status_aliases = {
        "instance_created": "running",
        "pod_created": "running",
        "teardown_completed": "teardown_completed",
    }
    status = status_aliases.get(manifest_status, manifest_status)
    if status not in {"planned", "blocked", "running", "completed", "teardown_completed"}:
        status = "blocked"
    if status == "completed" and blockers:
        status = "blocked"
    teardown_path = (
        job_dir / "runpod_wam_async_delete_manifest.json"
        if provider == "runpod"
        else job_dir / "vast_teardown_manifest.json"
    )
    teardown_payload = {}
    if teardown_path.is_file():
        try:
            import json

            teardown_payload = json.loads(teardown_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            teardown_payload = {}
    return WamComputeRunResult(
        provider=provider,
        status=status,
        provider_command_status=_string(manifest.get("provider_command_status")) or status,
        instance_id=instance_id or None,
        output_zip_path=output_path or None,
        output_zip_present=output_zip_present,
        mp4_count=mp4_count,
        extracted_video_paths=extracted_video_paths,
        runtime_result_status=runtime_status or "not_available",
        runtime_result_blockers=runtime_blockers,
        phase_log_path=_existing_path(job_dir / "vast_runtime_phase_log.jsonl"),
        budget_ledger_path=_existing_path(job_dir / "vast_budget_ledger.json"),
        teardown_manifest_path=str(teardown_path) if teardown_path.is_file() else None,
        teardown_status=_string(teardown_payload.get("status")) or (
            "completed" if manifest.get("teardown_performed") else "not_requested"
        ),
        teardown_performed=bool(manifest.get("teardown_performed")) or (
            teardown_payload.get("status") == "completed"
        ),
        continuing_spend_from_this_run=bool(manifest.get("continuing_spend_from_this_run")),
        blockers=blockers,
        provider_phase=_string(manifest.get("pod_status"))
        or _string(manifest.get("instance_status"))
        or _string(manifest.get("status")),
        output_availability=output_available,
        raw_secret_values_recorded=False,
        details=details,
    )


class VastWamComputeProvider(WamComputeProvider):
    name = "vast"

    def available(self) -> dict[str, Any]:
        key_file = Path(
            os.getenv("VAST_API_KEY_FILE", "~/.blueprint-secrets/vast_api_key")
        ).expanduser()
        return {
            "provider": self.name,
            "available": key_file.is_file(),
            "reason": None if key_file.is_file() else "vast_api_key_file_missing",
            "raw_secret_values_recorded": False,
        }

    def build_request(self, spec: WamComputeLaunchSpec, job_dir: Path) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "provider": self.name,
            "name": spec.name,
            "bundle_path": str(Path(spec.bundle_path).expanduser()),
            "provider_bundle_kind": spec.provider_bundle_kind,
            "image": spec.image,
            "output_zip_path": str(spec.output_path_for(self.name, job_dir)),
            "max_hourly_rate_usd": spec.max_hourly_rate_usd,
            "target_spend_usd": spec.target_spend_usd,
            "hard_cap_usd": spec.hard_cap_usd,
            "max_live_minutes": spec.max_live_minutes,
            "session_max_live_minutes": spec.session_max_live_minutes,
            "min_gpu_ram_mb": spec.min_gpu_ram_mb,
            "excluded_machine_ids": list(spec.excluded_machine_ids),
            "raw_secret_values_recorded": False,
        }

    def create(
        self,
        spec: WamComputeLaunchSpec,
        job_dir: Path,
        *,
        allow_paid_launch: bool,
    ) -> WamComputeRunResult:
        ensure_dir(job_dir)
        if not allow_paid_launch:
            return self._blocked_no_paid_launch(
                job_dir=job_dir,
                spec=spec,
                blockers=["paid_wam_compute_launch_not_authorized:vast"],
            )
        if not _env_truthy(VAST_WAM_PAID_LAUNCH_GATE_ENV):
            return self._blocked_no_paid_launch(
                job_dir=job_dir,
                spec=spec,
                blockers=[f"missing_env_{VAST_WAM_PAID_LAUNCH_GATE_ENV}"],
            )
        output_path = spec.output_path_for(self.name, job_dir)
        with _temporary_env(spec.env):
            manifest = create_async_vast_wam_run(
                job_dir=job_dir,
                bundle_path=spec.bundle_path,
                public_base_url=spec.public_base_url,
                provider_bundle_url=spec.provider_bundle_url,
                provider_output_put_url=spec.provider_output_put_url,
                provider_output_get_url=spec.provider_output_get_url,
                provider_bundle_url_file=spec.provider_bundle_url_file,
                provider_output_put_url_file=spec.provider_output_put_url_file,
                provider_output_get_url_file=spec.provider_output_get_url_file,
                token_file=spec.token_file,
                secret_env_file=spec.secret_env_file,
                output_path=output_path,
                allow_paid_vast_launch=True,
                max_hourly_rate=spec.max_hourly_rate_usd,
                target_spend_usd=spec.target_spend_usd,
                hard_cap_usd=spec.hard_cap_usd,
                allow_target_spend_overrun=spec.allow_target_spend_overrun,
                max_live_minutes=spec.max_live_minutes,
                session_max_live_minutes=spec.session_max_live_minutes,
                min_gpu_ram_mb=spec.min_gpu_ram_mb,
                excluded_machine_ids=spec.excluded_machine_ids,
                startup_poll_seconds=spec.startup_poll_seconds,
                public_staging_verify_max_wait_seconds=(
                    spec.public_staging_verify_max_wait_seconds
                ),
                public_staging_verify_retry_interval_seconds=(
                    spec.public_staging_verify_retry_interval_seconds
                ),
                public_staging_verify_timeout_seconds=(
                    spec.public_staging_verify_timeout_seconds
                ),
                public_staging_required_consecutive_successes=(
                    spec.public_staging_required_consecutive_successes
                ),
                verify_output_put_url=spec.verify_output_put_url,
                public_image=spec.image,
                vast_launch_mode=spec.vast_launch_mode,
                disk_gb=spec.disk_gb,
                heartbeat_url=spec.heartbeat_url,
            )
        result = _result_from_manifest(
            provider=self.name,
            job_dir=job_dir,
            manifest=manifest,
            spec=spec,
        )
        write_json(job_dir / "wam_compute_run_result.json", result.to_dict())
        return result

    def poll(
        self,
        job_dir: Path,
        *,
        max_wait_seconds: int,
        teardown: bool,
    ) -> WamComputeRunResult:
        manifest = poll_async_vast_wam_run(
            job_dir=job_dir,
            max_wait_seconds=max_wait_seconds,
            retry_interval_seconds=5,
            teardown=teardown,
        )
        output_path = _string(manifest.get("provider_runtime_output_zip_path")) or _string(
            manifest.get("output_path")
        )
        if not output_path:
            output_path = str(job_dir / "vast_provider_runtime_output.zip")
        inspection = self.inspect_output(job_dir, output_path) if output_path else {}
        result = _result_from_manifest(
            provider=self.name,
            job_dir=job_dir,
            manifest=manifest,
            output_inspection=inspection,
        )
        write_json(job_dir / "wam_compute_run_result.json", result.to_dict())
        return result

    def teardown(self, job_dir: Path, instance_id: str | None = None) -> WamComputeRunResult:
        manifest = destroy_async_vast_wam_run(job_dir=job_dir)
        result = _result_from_manifest(provider=self.name, job_dir=job_dir, manifest=manifest)
        write_json(job_dir / "wam_compute_run_result.json", result.to_dict())
        return result


class RunPodWamComputeProvider(WamComputeProvider):
    name = "runpod"

    def available(self) -> dict[str, Any]:
        key_file = Path(os.getenv(RUNPOD_API_KEY_FILE_ENV, "")).expanduser()
        has_key = bool(os.getenv(RUNPOD_API_KEY_ENV) or (str(key_file) and key_file.is_file()))
        gates = {
            RUNPOD_API_GATE_ENV: _env_truthy(RUNPOD_API_GATE_ENV),
            RUNPOD_POD_LAUNCH_GATE_ENV: _env_truthy(RUNPOD_POD_LAUNCH_GATE_ENV),
        }
        return {
            "provider": self.name,
            "available": bool(has_key and all(gates.values())),
            "reason": None if has_key and all(gates.values()) else "runpod_api_key_or_gate_missing",
            "gates": gates,
            "raw_secret_values_recorded": False,
        }

    def build_request(self, spec: WamComputeLaunchSpec, job_dir: Path) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "provider": self.name,
            "name": spec.name,
            "bundle_path": str(Path(spec.bundle_path).expanduser()),
            "provider_bundle_kind": spec.provider_bundle_kind,
            "image": spec.image,
            "output_zip_path": str(spec.output_path_for(self.name, job_dir)),
            "gpu_type_ids": list(spec.gpu_type_ids),
            "cloud_type": spec.cloud_type,
            "container_disk_gb": spec.container_disk_gb,
            "volume_gb": spec.volume_gb,
            "min_vcpu_per_gpu": spec.min_vcpu_per_gpu,
            "min_ram_per_gpu": spec.min_ram_per_gpu,
            "raw_secret_values_recorded": False,
        }

    def create(
        self,
        spec: WamComputeLaunchSpec,
        job_dir: Path,
        *,
        allow_paid_launch: bool,
    ) -> WamComputeRunResult:
        ensure_dir(job_dir)
        if not allow_paid_launch:
            return self._blocked_no_paid_launch(
                job_dir=job_dir,
                spec=spec,
                blockers=["paid_wam_compute_launch_not_authorized:runpod"],
            )
        output_path = spec.output_path_for(self.name, job_dir)
        with _temporary_env(spec.env):
            manifest = create_runpod_wam_async_run(
                job_dir=job_dir,
                bundle_path=spec.bundle_path,
                public_base_url=spec.public_base_url,
                provider_bundle_url=spec.provider_bundle_url,
                provider_output_put_url=spec.provider_output_put_url,
                provider_output_get_url=spec.provider_output_get_url,
                provider_bundle_url_file=spec.provider_bundle_url_file,
                provider_output_put_url_file=spec.provider_output_put_url_file,
                provider_output_get_url_file=spec.provider_output_get_url_file,
                token_file=spec.token_file,
                secret_env_file=spec.secret_env_file,
                output_path=output_path,
                allow_paid_runpod_launch=True,
                skip_public_staging_verification=spec.skip_public_staging_verification,
                verify_output_put_url=spec.verify_output_put_url,
                gpu_type_ids=spec.gpu_type_ids,
                image_name=spec.image,
                provider_bundle_kind=spec.provider_bundle_kind,
                container_disk_gb=spec.container_disk_gb,
                volume_gb=spec.volume_gb,
                cloud_type=spec.cloud_type,
                allowed_cuda_versions=spec.allowed_cuda_versions,
                min_vcpu_per_gpu=spec.min_vcpu_per_gpu,
                min_ram_per_gpu=spec.min_ram_per_gpu,
            )
        result = _result_from_manifest(
            provider=self.name,
            job_dir=job_dir,
            manifest=manifest,
            spec=spec,
        )
        write_json(job_dir / "wam_compute_run_result.json", result.to_dict())
        return result

    def poll(
        self,
        job_dir: Path,
        *,
        max_wait_seconds: int,
        teardown: bool,
    ) -> WamComputeRunResult:
        manifest = poll_runpod_wam_async_run(
            job_dir=job_dir,
            max_wait_seconds=max_wait_seconds,
            retry_interval_seconds=5,
            teardown=teardown,
        )
        output_path = _string(manifest.get("provider_runtime_output_zip_path"))
        inspection = self.inspect_output(job_dir, output_path) if output_path else {}
        result = _result_from_manifest(
            provider=self.name,
            job_dir=job_dir,
            manifest=manifest,
            output_inspection=inspection,
        )
        write_json(job_dir / "wam_compute_run_result.json", result.to_dict())
        return result

    def teardown(self, job_dir: Path, instance_id: str | None = None) -> WamComputeRunResult:
        return self.poll(job_dir, max_wait_seconds=0, teardown=True)


def _provider_order_from_env() -> list[str]:
    configured = _string(os.getenv(PROVIDER_ORDER_ENV))
    values = [
        item.strip().lower()
        for item in configured.replace(";", ",").split(",")
        if item.strip()
    ]
    return values or list(DEFAULT_PROVIDER_ORDER)


def get_wam_compute_provider(name: str | None) -> WamComputeProvider:
    key = _string(name).lower()
    if key in {"", "auto"}:
        key = _provider_order_from_env()[0]
    if key == "vast":
        return VastWamComputeProvider()
    if key == "runpod":
        return RunPodWamComputeProvider()
    raise ValueError("unknown_wam_compute_provider:%r (known: runpod, vast, auto)" % name)


def list_wam_compute_providers() -> list[dict[str, Any]]:
    return [RunPodWamComputeProvider().available(), VastWamComputeProvider().available()]


def _provider_order(provider_order: Sequence[str] | None) -> list[str]:
    order = list(provider_order or _provider_order_from_env())
    normalized: list[str] = []
    for item in order:
        key = _string(item).lower()
        if key == "auto":
            for default in _provider_order_from_env():
                if default not in normalized:
                    normalized.append(default)
            continue
        if key and key not in normalized:
            normalized.append(key)
    return normalized or list(DEFAULT_PROVIDER_ORDER)


def _failover_allowed(blockers: Sequence[str], allowed: Sequence[str]) -> bool:
    if not blockers or not allowed:
        return False
    return any(any(marker in blocker for marker in allowed) for blocker in blockers)


def run_wam_compute_job(
    *,
    spec: WamComputeLaunchSpec,
    job_dir: str | Path,
    provider_order: Sequence[str] | None = None,
    allow_paid_launch: bool = False,
    failover_on_blockers: Sequence[str] = (),
    teardown: bool = True,
) -> WamComputeRunResult:
    root = Path(job_dir).expanduser().resolve()
    ensure_dir(root)
    bundle_path = Path(spec.bundle_path).expanduser().resolve()
    if not bundle_path.is_file():
        result = WamComputeRunResult(
            provider="none",
            status="blocked",
            provider_command_status="blocked",
            blockers=["wam_compute_bundle_missing"],
            provider_phase="local_preflight",
            output_availability="not_available",
            details={
                "generated_at": utc_now_iso(),
                "job_dir": str(root),
                "bundle_path": str(bundle_path),
            },
        )
        write_json(root / "wam_compute_run_result.json", result.to_dict())
        return result
    attempts: list[dict[str, Any]] = []
    last_result: WamComputeRunResult | None = None
    for provider_name in _provider_order(provider_order):
        provider = get_wam_compute_provider(provider_name)
        provider_job_dir = (
            root
            if root.name.endswith(f"{provider.name}_provider_run")
            else root / f"{provider.name}_provider_run"
        )
        ensure_dir(provider_job_dir)
        create_result = provider.create(
            spec,
            provider_job_dir,
            allow_paid_launch=allow_paid_launch,
        )
        attempts.append({"provider": provider.name, "phase": "create", **create_result.to_dict()})
        if create_result.status == "blocked":
            last_result = create_result
            if _failover_allowed(create_result.blockers, failover_on_blockers):
                continue
            break
        poll_result = provider.poll(
            provider_job_dir,
            max_wait_seconds=spec.max_wait_seconds,
            teardown=teardown,
        )
        if (
            poll_result.output_zip_present
            and spec.expected_video_count > 0
            and poll_result.mp4_count < spec.expected_video_count
        ):
            poll_result.status = "blocked"
            poll_result.output_availability = (
                "zip_present_but_expected_generated_videos_missing"
            )
            poll_result.blockers.append(poll_result.output_availability)
        attempts.append({"provider": provider.name, "phase": "poll", **poll_result.to_dict()})
        last_result = poll_result
        if poll_result.status == "completed":
            break
        if (
            poll_result.status == "blocked"
            and not poll_result.continuing_spend_from_this_run
            and _failover_allowed(poll_result.blockers, failover_on_blockers)
        ):
            continue
        break
    if last_result is None:
        last_result = WamComputeRunResult(
            provider="none",
            status="blocked",
            provider_command_status="blocked",
            blockers=["wam_compute_provider_order_empty"],
            provider_phase="provider_selection",
        )
    last_payload = last_result.to_dict()
    last_payload["attempted_providers"] = attempts
    write_json(root / "wam_compute_run_result.json", last_payload)
    last_result.details["attempted_providers"] = attempts
    return last_result
