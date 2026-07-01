"""Provider-neutral WAM compute wrappers.

This module keeps the WAM compute contract stable while delegating provider-specific
launch, poll, teardown, URL handling, and dud detection to the existing RunPod and Vast
async runners. Generated WAM media remains a downstream support artifact; a completed
provider run here does not prove capture truth, visual usefulness, collision truth,
safety validation, physical readiness, or deployment approval.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
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
from .wam_generated_video_review import (
    validate_generated_mp4_for_review,
    visual_smoke_generated_rollouts_for_review,
)


SCHEMA_VERSION = "wam_compute_providers.v1"
RESULT_SCHEMA_VERSION = "wam_compute_run_result.v1"
DEFAULT_PROVIDER_ORDER = ("deepinfra", "runpod", "vast")
PROVIDER_ORDER_ENV = "BLUEPRINT_WAM_COMPUTE_PROVIDER_ORDER"
VAST_WAM_PAID_LAUNCH_GATE_ENV = "BLUEPRINT_ALLOW_PAID_VAST_WAM_PROVIDER_LAUNCH"
PROVIDER_BUNDLE_KINDS = ("wam", "unitree_unifolm", "unitree_groot_n17_sonic")
DEEPINFRA_PROVIDER_NAME = "deepinfra"
DEEPINFRA_MODEL_ID = "nvidia/Cosmos3-Nano"
DEEPINFRA_API_BASE_URL_ENV = "BLUEPRINT_DEEPINFRA_API_BASE_URL"
DEEPINFRA_MODEL_ID_ENV = "BLUEPRINT_DEEPINFRA_COSMOS3_MODEL_ID"
DEEPINFRA_API_KEY_ENV = "DEEPINFRA_API_KEY"
DEEPINFRA_API_KEY_FILE_ENV = "DEEPINFRA_API_KEY_FILE"
DEEPINFRA_API_GATE_ENV = "BLUEPRINT_ALLOW_DEEPINFRA_API_CALLS"
DEEPINFRA_PROMPT_ENV = "BLUEPRINT_DEEPINFRA_COSMOS3_PROMPT"
DEEPINFRA_OUTPUT_TYPE_ENV = "BLUEPRINT_DEEPINFRA_COSMOS3_OUTPUT_TYPE"
DEEPINFRA_RESOLUTION_ENV = "BLUEPRINT_DEEPINFRA_COSMOS3_RESOLUTION"
DEEPINFRA_ASPECT_RATIO_ENV = "BLUEPRINT_DEEPINFRA_COSMOS3_ASPECT_RATIO"
DEEPINFRA_DURATION_SECONDS_ENV = "BLUEPRINT_DEEPINFRA_COSMOS3_DURATION_SECONDS"
DEEPINFRA_SEED_ENV = "BLUEPRINT_DEEPINFRA_COSMOS3_SEED"
DEEPINFRA_MAX_COST_USD_ENV = "BLUEPRINT_DEEPINFRA_COSMOS3_MAX_COST_USD"
DEEPINFRA_TIMEOUT_SECONDS_ENV = "BLUEPRINT_DEEPINFRA_COSMOS3_TIMEOUT_SECONDS"
DEEPINFRA_PRICE_PER_GENERATED_SECOND_480P_USD = 0.0108
DEEPINFRA_720P_MULTIPLIER = 2.25


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


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number and number not in {float("inf"), float("-inf")} else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_secret_value(*, env_name: str, file_env_name: str, default_file: str = "") -> str:
    direct = _string(os.getenv(env_name))
    if direct:
        return direct
    file_text = _string(os.getenv(file_env_name)) or default_file
    if not file_text:
        return ""
    path = Path(file_text).expanduser()
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _deepinfra_model_id() -> str:
    return _string(os.getenv(DEEPINFRA_MODEL_ID_ENV)) or DEEPINFRA_MODEL_ID


def _deepinfra_api_base_url() -> str:
    return (
        _string(os.getenv(DEEPINFRA_API_BASE_URL_ENV))
        or "https://api.deepinfra.com/v1/inference"
    ).rstrip("/")


def _deepinfra_api_url() -> str:
    return f"{_deepinfra_api_base_url()}/{_deepinfra_model_id()}"


def _deepinfra_runtime_manifest_from_bundle(bundle_path: str | Path) -> dict[str, Any]:
    path = Path(bundle_path).expanduser()
    if not path.is_file() or not zipfile.is_zipfile(path):
        return {}
    candidates = (
        "provider_runtime/wam_provider_runtime_manifest.json",
        "wam_provider_runtime_manifest.json",
    )
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            for candidate in candidates:
                if candidate not in names:
                    continue
                value = json.loads(archive.read(candidate).decode("utf-8"))
                return dict(value) if isinstance(value, Mapping) else {}
    except (OSError, ValueError, zipfile.BadZipFile):
        return {}
    return {}


def _deepinfra_duration_seconds(runtime_manifest: Mapping[str, Any]) -> float:
    configured = _safe_float(os.getenv(DEEPINFRA_DURATION_SECONDS_ENV), 0.0)
    if configured > 0:
        return round(min(max(configured, 1.0), 8.0), 3)
    num_frames = _safe_float(runtime_manifest.get("num_frames"), 0.0)
    fps = _safe_float(runtime_manifest.get("fps"), 0.0)
    if num_frames > 0 and fps > 0:
        return round(min(max(num_frames / fps, 1.0), 8.0), 3)
    return 5.0


def _deepinfra_resolution(runtime_manifest: Mapping[str, Any]) -> str:
    configured = _string(os.getenv(DEEPINFRA_RESOLUTION_ENV)).lower()
    if configured in {"480p", "720p"}:
        return configured
    width = _safe_int(runtime_manifest.get("width"), 0)
    height = _safe_int(runtime_manifest.get("height"), 0)
    if width >= 1280 or height >= 720:
        return "720p"
    return "480p"


def _deepinfra_estimated_cost_usd(*, duration_seconds: float, resolution: str) -> float:
    multiplier = DEEPINFRA_720P_MULTIPLIER if resolution == "720p" else 1.0
    return round(duration_seconds * DEEPINFRA_PRICE_PER_GENERATED_SECOND_480P_USD * multiplier, 6)


def _deepinfra_actual_cost_usd(response: Mapping[str, Any]) -> float | None:
    candidates = [
        response.get("cost"),
        response.get("cost_usd"),
        _mapping(response.get("inference_status")).get("cost"),
        _mapping(response.get("inference_status")).get("cost_usd"),
        _mapping(response.get("billing")).get("cost"),
        _mapping(response.get("billing")).get("cost_usd"),
    ]
    for value in candidates:
        cost = _safe_float(value, -1.0)
        if cost >= 0:
            return round(cost, 6)
    return None


def _deepinfra_video_url(response: Mapping[str, Any]) -> str:
    for key in ("video_url", "output_url", "url"):
        value = _string(response.get(key))
        if value:
            return value
    output = response.get("output")
    if isinstance(output, str):
        return output
    if isinstance(output, Mapping):
        for key in ("video_url", "url"):
            value = _string(output.get(key))
            if value:
                return value
    if isinstance(output, Sequence) and not isinstance(output, (bytes, bytearray, str)):
        for item in output:
            if isinstance(item, str) and item:
                return item
            if isinstance(item, Mapping):
                value = _string(item.get("video_url") or item.get("url"))
                if value:
                    return value
    return ""


def _deepinfra_prompt(runtime_manifest: Mapping[str, Any]) -> str:
    prompt = _string(os.getenv(DEEPINFRA_PROMPT_ENV)) or _string(runtime_manifest.get("prompt"))
    return prompt or "Predict the next robot-scene frames from Blueprint action conditioning."


def _deepinfra_request_payload(runtime_manifest: Mapping[str, Any]) -> dict[str, Any]:
    resolution = _deepinfra_resolution(runtime_manifest)
    duration_seconds = _deepinfra_duration_seconds(runtime_manifest)
    payload: dict[str, Any] = {
        "prompt": _deepinfra_prompt(runtime_manifest),
        "output_type": _string(os.getenv(DEEPINFRA_OUTPUT_TYPE_ENV)) or "video",
        "resolution": resolution,
        "aspect_ratio": _string(os.getenv(DEEPINFRA_ASPECT_RATIO_ENV)) or "16:9",
        "duration_seconds": duration_seconds,
    }
    seed = _safe_int(os.getenv(DEEPINFRA_SEED_ENV) or runtime_manifest.get("seed"), 0)
    if seed:
        payload["seed"] = seed
    return payload


def _deepinfra_post_json(
    *,
    url: str,
    api_key: str,
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(dict(payload)).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "BlueprintCapturePipeline/DeepInfraCosmos3NanoAdapter",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        body = response.read().decode("utf-8")
    value = json.loads(body)
    return dict(value) if isinstance(value, Mapping) else {"output": value}


def _deepinfra_download_file(
    *,
    url: str,
    target_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    ensure_dir(target_path.parent)
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:  # noqa: S310
        with target_path.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
    return {
        "path": str(target_path),
        "size_bytes": target_path.stat().st_size if target_path.is_file() else 0,
        "sha256": _sha256_file(target_path) if target_path.is_file() else None,
    }


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
    allowed_machine_ids: Sequence[int] = ()
    min_reliability: float = 0.0
    require_direct_port: bool = False
    preferred_gpu_keywords: Sequence[str] = ()
    preferred_geolocation_regex: str = ""
    prefer_isaac_rt: bool = False
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
        if provider == DEEPINFRA_PROVIDER_NAME:
            return job_dir / "deepinfra_provider_runtime_output.zip"
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


class DeepInfraCosmos3NanoProvider(WamComputeProvider):
    name = DEEPINFRA_PROVIDER_NAME

    def available(self) -> dict[str, Any]:
        key_file_text = (
            _string(os.getenv(DEEPINFRA_API_KEY_FILE_ENV))
            or "~/.blueprint-secrets/deepinfra_api_key"
        )
        key_file = Path(key_file_text).expanduser()
        has_key = bool(os.getenv(DEEPINFRA_API_KEY_ENV) or key_file.is_file())
        gate = _env_truthy(DEEPINFRA_API_GATE_ENV)
        return {
            "provider": self.name,
            "available": bool(has_key and gate),
            "reason": None if has_key and gate else "deepinfra_api_key_or_gate_missing",
            "model_id": _deepinfra_model_id(),
            "api_gate": {DEEPINFRA_API_GATE_ENV: gate},
            "raw_secret_values_recorded": False,
        }

    def build_request(self, spec: WamComputeLaunchSpec, job_dir: Path) -> dict[str, Any]:
        runtime_manifest = _deepinfra_runtime_manifest_from_bundle(spec.bundle_path)
        payload = _deepinfra_request_payload(runtime_manifest)
        duration = _safe_float(payload.get("duration_seconds"), 5.0)
        resolution = _string(payload.get("resolution")) or "480p"
        key_file = Path(
            _string(os.getenv(DEEPINFRA_API_KEY_FILE_ENV))
            or "~/.blueprint-secrets/deepinfra_api_key"
        ).expanduser()
        estimated_cost = _deepinfra_estimated_cost_usd(
            duration_seconds=duration,
            resolution=resolution,
        )
        return {
            "schema_version": "deepinfra_cosmos3_request_manifest.v1",
            "provider": self.name,
            "model_id": _deepinfra_model_id(),
            "api_endpoint": _deepinfra_api_url(),
            "name": spec.name,
            "bundle_path_name": Path(spec.bundle_path).expanduser().name,
            "provider_bundle_kind": spec.provider_bundle_kind,
            "output_zip_path": str(spec.output_path_for(self.name, job_dir)),
            "request_payload": payload,
            "estimated_cost_usd": estimated_cost,
            "pricing": {
                "price_per_generated_second_480p_usd": (
                    DEEPINFRA_PRICE_PER_GENERATED_SECOND_480P_USD
                ),
                "resolution_720p_multiplier": DEEPINFRA_720P_MULTIPLIER,
                "pricing_source": "https://deepinfra.com/nvidia/Cosmos3-Nano",
            },
            "api_key_configured": bool(
                os.getenv(DEEPINFRA_API_KEY_ENV)
                or _string(os.getenv(DEEPINFRA_API_KEY_FILE_ENV))
                or key_file.is_file()
            ),
            "raw_secret_values_recorded": False,
            "provider_transport_urls_recorded": False,
        }

    def _write_cost_ledger(
        self,
        *,
        job_dir: Path,
        request_manifest: Mapping[str, Any],
        status: str,
        api_call_performed: bool,
        actual_cost_usd: float | None,
        blockers: Sequence[str],
    ) -> dict[str, Any]:
        estimated_cost = _safe_float(request_manifest.get("estimated_cost_usd"), 0.0)
        payload = {
            "schema_version": "deepinfra_cosmos3_cost_control_ledger.v1",
            "generated_at": utc_now_iso(),
            "provider": self.name,
            "model_id": _deepinfra_model_id(),
            "status": status,
            "api_call_performed": bool(api_call_performed),
            "live_provider_calls_performed": bool(api_call_performed),
            "continuing_spend_from_this_run": False,
            "estimated_cost_usd": estimated_cost,
            "actual_cost_usd": actual_cost_usd,
            "cost_source": "deepinfra_response" if actual_cost_usd is not None else "estimate",
            "budget": {
                "requested_budget_usd": _safe_float(
                    os.getenv(DEEPINFRA_MAX_COST_USD_ENV),
                    _safe_float(request_manifest.get("max_cost_usd"), 0.0),
                ),
                "provider_pricing_per_generated_second_480p_usd": (
                    DEEPINFRA_PRICE_PER_GENERATED_SECOND_480P_USD
                ),
                "resolution_720p_multiplier": DEEPINFRA_720P_MULTIPLIER,
            },
            "blockers": sorted(set(str(item) for item in blockers if str(item))),
            "claim_boundary": {
                "cost_ledger_is_spend_telemetry_only": True,
                "api_cost_is_not_model_quality_or_task_success": True,
                "raw_secret_values_recorded": False,
            },
            "raw_secret_values_recorded": False,
        }
        write_json(job_dir / "deepinfra_cosmos3_cost_control_ledger.json", payload)
        write_json(job_dir / "wam_provider_cost_control_ledger.json", payload)
        return payload

    def _write_execution_manifest(
        self,
        *,
        job_dir: Path,
        request_manifest: Mapping[str, Any],
        status: str,
        provider_command_status: str,
        blockers: Sequence[str],
        output_zip_path: Path,
        mp4_path: Path | None = None,
        response_payload: Mapping[str, Any] | None = None,
        validation: Mapping[str, Any] | None = None,
        visual_report_path: Path | None = None,
        api_call_performed: bool = False,
        elapsed_seconds: float | None = None,
    ) -> dict[str, Any]:
        payload = {
            "schema_version": "deepinfra_cosmos3_execution_manifest.v1",
            "generated_at": utc_now_iso(),
            "provider": self.name,
            "model_id": _deepinfra_model_id(),
            "status": status,
            "provider_command_status": provider_command_status,
            "provider_command_blockers": sorted(
                set(str(item) for item in blockers if str(item))
            ),
            "blockers": sorted(set(str(item) for item in blockers if str(item))),
            "api_call_performed": bool(api_call_performed),
            "live_provider_calls_performed": bool(api_call_performed),
            "elapsed_seconds": elapsed_seconds,
            "provider_runtime_output_zip_path": str(output_zip_path),
            "output_zip_present": output_zip_path.is_file(),
            "mp4_count": 1 if mp4_path and mp4_path.is_file() else 0,
            "generated_video_path": str(mp4_path) if mp4_path and mp4_path.is_file() else None,
            "runtime_result_status": "completed" if status == "completed" else status,
            "runtime_result_blockers": sorted(
                set(str(item) for item in blockers if str(item))
            ),
            "visual_quality_report_path": (
                str(visual_report_path) if visual_report_path and visual_report_path.is_file() else None
            ),
            "downloaded_mp4_validation_status": _mapping(validation).get("status"),
            "response_metadata": {
                "request_id": _string(_mapping(response_payload).get("request_id")),
                "inference_status": _mapping(
                    _mapping(response_payload).get("inference_status")
                ),
                "video_url_present": bool(_deepinfra_video_url(_mapping(response_payload))),
            },
            "teardown_performed": False,
            "continuing_spend_from_this_run": False,
            "raw_secret_values_recorded": False,
            "claim_boundary": _deepinfra_claim_boundary(),
        }
        write_json(job_dir / "deepinfra_cosmos3_execution_manifest.json", payload)
        return payload

    def _blocked_result(
        self,
        *,
        job_dir: Path,
        spec: WamComputeLaunchSpec,
        request_manifest: Mapping[str, Any],
        blockers: Sequence[str],
        api_call_performed: bool = False,
    ) -> WamComputeRunResult:
        output_zip_path = spec.output_path_for(self.name, job_dir)
        self._write_cost_ledger(
            job_dir=job_dir,
            request_manifest=request_manifest,
            status="blocked",
            api_call_performed=api_call_performed,
            actual_cost_usd=None,
            blockers=blockers,
        )
        manifest = self._write_execution_manifest(
            job_dir=job_dir,
            request_manifest=request_manifest,
            status="blocked",
            provider_command_status="blocked",
            blockers=blockers,
            output_zip_path=output_zip_path,
            api_call_performed=api_call_performed,
        )
        result = _result_from_manifest(
            provider=self.name,
            job_dir=job_dir,
            manifest=manifest,
            spec=spec,
        )
        write_json(job_dir / "wam_compute_run_result.json", result.to_dict())
        return result

    def create(
        self,
        spec: WamComputeLaunchSpec,
        job_dir: Path,
        *,
        allow_paid_launch: bool,
    ) -> WamComputeRunResult:
        ensure_dir(job_dir)
        output_zip_path = spec.output_path_for(self.name, job_dir)
        request_manifest = self.build_request(spec, job_dir)
        write_json(job_dir / "deepinfra_cosmos3_request_manifest.json", request_manifest)

        api_key = _read_secret_value(
            env_name=DEEPINFRA_API_KEY_ENV,
            file_env_name=DEEPINFRA_API_KEY_FILE_ENV,
            default_file="~/.blueprint-secrets/deepinfra_api_key",
        )
        blockers: list[str] = []
        if not allow_paid_launch:
            blockers.append("paid_wam_compute_launch_not_authorized:deepinfra")
        if not _env_truthy(DEEPINFRA_API_GATE_ENV):
            blockers.append(f"missing_env_{DEEPINFRA_API_GATE_ENV}")
        if not api_key:
            blockers.append("deepinfra_api_key_missing")
        max_cost = _safe_float(
            os.getenv(DEEPINFRA_MAX_COST_USD_ENV),
            _safe_float(spec.hard_cap_usd, DEFAULT_HARD_CAP_USD),
        )
        estimated_cost = _safe_float(request_manifest.get("estimated_cost_usd"), 0.0)
        request_manifest["max_cost_usd"] = max_cost
        write_json(job_dir / "deepinfra_cosmos3_request_manifest.json", request_manifest)
        if max_cost > 0 and estimated_cost > max_cost and not spec.allow_target_spend_overrun:
            blockers.append("deepinfra_estimated_cost_exceeds_hard_cap")
        if blockers:
            return self._blocked_result(
                job_dir=job_dir,
                spec=spec,
                request_manifest=request_manifest,
                blockers=blockers,
            )

        response_payload: dict[str, Any] = {}
        mp4_path: Path | None = None
        validation: dict[str, Any] = {}
        visual_report_path: Path | None = None
        api_call_performed = False
        started = time.monotonic()
        try:
            response_payload = _deepinfra_post_json(
                url=_deepinfra_api_url(),
                api_key=api_key,
                payload=_mapping(request_manifest.get("request_payload")),
                timeout_seconds=_safe_float(os.getenv(DEEPINFRA_TIMEOUT_SECONDS_ENV), 120.0),
            )
            api_call_performed = True
            video_url = _deepinfra_video_url(response_payload)
            if not video_url:
                blockers.append("deepinfra_response_missing_video_url")
            else:
                download_url = _resolve_deepinfra_download_url(video_url)
                mp4_path = job_dir / "deepinfra_cosmos3_generated_rollout.mp4"
                _deepinfra_download_file(
                    url=download_url,
                    target_path=mp4_path,
                    timeout_seconds=_safe_float(
                        os.getenv(DEEPINFRA_TIMEOUT_SECONDS_ENV),
                        120.0,
                    ),
                )
                validation = validate_generated_mp4_for_review(mp4_path)
                if validation.get("status") != "completed":
                    blockers.extend(_string_list(validation.get("blockers")))
                    blockers.append("deepinfra_generated_video_not_reviewable")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            blockers.append(f"deepinfra_api_request_failed:{type(exc).__name__}")

        status = "completed" if mp4_path and mp4_path.is_file() and not blockers else "blocked"
        visual_report = _write_deepinfra_visual_quality_report(
            job_dir=job_dir,
            generated_video_path=mp4_path,
            validation=validation,
            generated_at=utc_now_iso(),
            provider_status=status,
        )
        visual_report_path = job_dir / "wam_rollout_visual_quality_report.json"
        deepinfra_visual_report_path = job_dir / "deepinfra_cosmos3_visual_quality_report.json"
        runtime_result = _write_deepinfra_runtime_result(
            job_dir=job_dir,
            request_manifest=request_manifest,
            status=status,
            blockers=blockers,
            mp4_path=mp4_path,
            visual_report=visual_report,
        )
        provider_payload = _write_deepinfra_provider_output(
            job_dir=job_dir,
            status=status,
            blockers=blockers,
            request_manifest=request_manifest,
            mp4_path=mp4_path,
            runtime_result=runtime_result,
        )
        actual_cost = _deepinfra_actual_cost_usd(response_payload)
        self._write_cost_ledger(
            job_dir=job_dir,
            request_manifest=request_manifest,
            status=status,
            api_call_performed=api_call_performed,
            actual_cost_usd=actual_cost,
            blockers=blockers,
        )
        manifest = self._write_execution_manifest(
            job_dir=job_dir,
            request_manifest=request_manifest,
            status=status,
            provider_command_status=status,
            blockers=blockers,
            output_zip_path=output_zip_path,
            mp4_path=mp4_path,
            response_payload=response_payload,
            validation=validation,
            visual_report_path=visual_report_path,
            api_call_performed=api_call_performed,
            elapsed_seconds=round(time.monotonic() - started, 6),
        )
        checksums = _write_deepinfra_checksums(
            job_dir=job_dir,
            paths=[
                path
                for path in (
                    mp4_path,
                    job_dir / "wam_provider_output.json",
                    job_dir / "wam_runtime_result.json",
                    visual_report_path,
                    deepinfra_visual_report_path,
                    job_dir / "deepinfra_cosmos3_request_manifest.json",
                    job_dir / "deepinfra_cosmos3_execution_manifest.json",
                    job_dir / "deepinfra_cosmos3_cost_control_ledger.json",
                )
                if path is not None
            ],
        )
        _write_deepinfra_output_zip(
            output_zip_path=output_zip_path,
            paths=[
                path
                for path in (
                    mp4_path,
                    job_dir / "wam_provider_output.json",
                    job_dir / "wam_runtime_result.json",
                    visual_report_path,
                    deepinfra_visual_report_path,
                    job_dir / "deepinfra_cosmos3_request_manifest.json",
                    job_dir / "deepinfra_cosmos3_execution_manifest.json",
                    job_dir / "deepinfra_cosmos3_cost_control_ledger.json",
                    job_dir / "deepinfra_cosmos3_artifact_checksums.json",
                )
                if path is not None
            ],
        )
        manifest = self._write_execution_manifest(
            job_dir=job_dir,
            request_manifest=request_manifest,
            status=status,
            provider_command_status=status,
            blockers=blockers,
            output_zip_path=output_zip_path,
            mp4_path=mp4_path,
            response_payload=response_payload,
            validation=validation,
            visual_report_path=visual_report_path,
            api_call_performed=api_call_performed,
            elapsed_seconds=round(time.monotonic() - started, 6),
        )
        checksums = _write_deepinfra_checksums(
            job_dir=job_dir,
            paths=[
                path
                for path in (
                    mp4_path,
                    output_zip_path,
                    job_dir / "wam_provider_output.json",
                    job_dir / "wam_runtime_result.json",
                    visual_report_path,
                    deepinfra_visual_report_path,
                    job_dir / "deepinfra_cosmos3_request_manifest.json",
                    job_dir / "deepinfra_cosmos3_execution_manifest.json",
                    job_dir / "deepinfra_cosmos3_cost_control_ledger.json",
                )
                if path is not None
            ],
            existing=checksums,
        )
        inspection = self.inspect_output(job_dir, output_zip_path)
        result = _result_from_manifest(
            provider=self.name,
            job_dir=job_dir,
            manifest={
                **manifest,
                "mp4_count": inspection.get("mp4_count"),
                "output_zip_present": inspection.get("zip_present"),
            },
            spec=spec,
            output_inspection=inspection,
        )
        result.details.update(
            {
                "request_manifest_path": str(job_dir / "deepinfra_cosmos3_request_manifest.json"),
                "execution_manifest_path": str(
                    job_dir / "deepinfra_cosmos3_execution_manifest.json"
                ),
                "cost_ledger_path": str(
                    job_dir / "deepinfra_cosmos3_cost_control_ledger.json"
                ),
                "artifact_checksums_path": str(
                    job_dir / "deepinfra_cosmos3_artifact_checksums.json"
                ),
                "provider_payload_status": provider_payload.get("status"),
                "artifact_checksum_count": checksums.get("artifact_count"),
            }
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
        manifest = _read_json_file(job_dir / "deepinfra_cosmos3_execution_manifest.json")
        if not manifest:
            result = WamComputeRunResult(
                provider=self.name,
                status="blocked",
                provider_command_status="blocked",
                blockers=["deepinfra_execution_manifest_missing"],
                provider_phase="poll",
                output_availability="not_available",
                continuing_spend_from_this_run=False,
                details={"generated_at": utc_now_iso(), "job_dir": str(job_dir)},
            )
            write_json(job_dir / "wam_compute_run_result.json", result.to_dict())
            return result
        output_path = _string(manifest.get("provider_runtime_output_zip_path")) or str(
            job_dir / "deepinfra_provider_runtime_output.zip"
        )
        inspection = self.inspect_output(job_dir, output_path)
        result = _result_from_manifest(
            provider=self.name,
            job_dir=job_dir,
            manifest=manifest,
            output_inspection=inspection,
        )
        write_json(job_dir / "wam_compute_run_result.json", result.to_dict())
        return result

    def teardown(self, job_dir: Path, instance_id: str | None = None) -> WamComputeRunResult:
        manifest = {
            "schema_version": "deepinfra_cosmos3_teardown_manifest.v1",
            "generated_at": utc_now_iso(),
            "provider": self.name,
            "status": "not_required",
            "reason": "deepinfra_api_call_has_no_blueprint_owned_live_instance",
            "teardown_performed": False,
            "continuing_spend_from_this_run": False,
            "raw_secret_values_recorded": False,
        }
        write_json(job_dir / "deepinfra_cosmos3_teardown_manifest.json", manifest)
        return self.poll(job_dir, max_wait_seconds=0, teardown=False)


def _deepinfra_claim_boundary() -> dict[str, Any]:
    return {
        "deepinfra_cosmos3_output_is_model_derived_support_artifact": True,
        "deepinfra_api_success_is_not_task_success": True,
        "valid_mp4_or_provider_completed_is_not_visual_success": True,
        "visual_quality_gate_required_before_review_claim": True,
        "external_episode_consistency_or_human_scorer_required_for_success": True,
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "capture_truth": False,
        "collision_truth": False,
        "safety_validation": False,
        "physical_robot_readiness": False,
        "deployment_approval": False,
        "raw_secret_values_recorded": False,
    }


def _resolve_deepinfra_download_url(video_url: str) -> str:
    if urllib.parse.urlparse(video_url).scheme:
        return video_url
    parsed = urllib.parse.urlparse(_deepinfra_api_base_url())
    origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else ""
    return urllib.parse.urljoin(origin or "https://api.deepinfra.com", video_url)


def _write_deepinfra_visual_quality_report(
    *,
    job_dir: Path,
    generated_video_path: Path | None,
    validation: Mapping[str, Any],
    generated_at: str,
    provider_status: str,
) -> dict[str, Any]:
    rollouts: list[dict[str, Any]] = []
    if generated_video_path and generated_video_path.is_file():
        rollouts.append(
            {
                "rollout_id": "deepinfra_cosmos3_rollout_0001",
                "generated_video_path": str(generated_video_path),
                "policy_id": "deepinfra_cosmos3_nano",
            }
        )
    visual_smoke = visual_smoke_generated_rollouts_for_review(
        rollouts=rollouts,
        output_dir=job_dir / "deepinfra_generated_rollout_visual_smoke",
        generated_at=generated_at,
        require_review_quality_profile=True,
    )
    passed = (
        validation.get("status") == "completed"
        and visual_smoke.get("status") == "passed_visual_quality_smoke"
    )
    blockers = sorted(
        set(
            [
                *[str(item) for item in validation.get("blockers", []) or [] if str(item)],
                *[str(item) for item in visual_smoke.get("blockers", []) or [] if str(item)],
                *[
                    str(item)
                    for item in visual_smoke.get("review_usefulness_blockers", []) or []
                    if str(item)
                ],
            ]
        )
    )
    report = {
        "schema_version": "deepinfra_cosmos3_visual_quality_report.v1",
        "generated_at": generated_at,
        "provider": DEEPINFRA_PROVIDER_NAME,
        "model_id": _deepinfra_model_id(),
        "status": "passed_visual_quality_gate" if passed else "failed_visual_quality_gate",
        "visual_success": passed,
        "visually_useful_rollout": passed,
        "provider_status": provider_status,
        "provider_completed": provider_status == "completed",
        "provider_completed_visual_quality_failed": bool(
            provider_status == "completed" and not passed
        ),
        "generated_video_review_validation": dict(validation),
        "generated_rollout_visual_smoke": visual_smoke,
        "generated_rollout_visual_smoke_status": visual_smoke.get("status"),
        "generated_rollout_review_usefulness_status": visual_smoke.get(
            "review_usefulness_status"
        ),
        "generated_rollout_review_usefulness_blockers": [
            str(item) for item in visual_smoke.get("review_usefulness_blockers", []) or []
        ],
        "review_video_path": str(generated_video_path) if generated_video_path else None,
        "blockers": blockers,
        "claim_boundary": _deepinfra_claim_boundary(),
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(job_dir / "deepinfra_cosmos3_visual_quality_report.json", report)
    write_json(job_dir / "wam_rollout_visual_quality_report.json", report)
    return report


def _write_deepinfra_runtime_result(
    *,
    job_dir: Path,
    request_manifest: Mapping[str, Any],
    status: str,
    blockers: Sequence[str],
    mp4_path: Path | None,
    visual_report: Mapping[str, Any],
) -> dict[str, Any]:
    request_payload = _mapping(request_manifest.get("request_payload"))
    video_downloaded = bool(mp4_path and mp4_path.is_file())
    runtime_result = {
        "schema_version": "deepinfra_cosmos3_runtime_result.v1",
        "generated_at": utc_now_iso(),
        "status": status,
        "runtime": "deepinfra_cosmos3_nano_api",
        "model_id": _deepinfra_model_id(),
        "runtime_settings": {
            "duration_seconds": request_payload.get("duration_seconds"),
            "resolution": request_payload.get("resolution"),
            "aspect_ratio": request_payload.get("aspect_ratio"),
            "seed": request_payload.get("seed"),
            "output_type": request_payload.get("output_type"),
        },
        "learned_wam_model_ran": video_downloaded,
        "api_model_inference_performed": video_downloaded,
        "generated_video_path": str(mp4_path) if video_downloaded else None,
        "visual_quality_report_path": str(job_dir / "wam_rollout_visual_quality_report.json"),
        "visual_quality_status": visual_report.get("status"),
        "blockers": sorted(set(str(item) for item in blockers if str(item))),
        "truth_boundary": {
            "generated_video_is_model_output": video_downloaded,
            "generated_video_is_model_derived_support_artifact": True,
            "generated_video_success_label_proven": False,
            "external_episode_consistency_scored": False,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
        },
        "claim_boundary": _deepinfra_claim_boundary(),
        "raw_secret_values_recorded": False,
    }
    write_json(job_dir / "wam_runtime_result.json", runtime_result)
    return runtime_result


def _write_deepinfra_provider_output(
    *,
    job_dir: Path,
    status: str,
    blockers: Sequence[str],
    request_manifest: Mapping[str, Any],
    mp4_path: Path | None,
    runtime_result: Mapping[str, Any],
) -> dict[str, Any]:
    request_payload = _mapping(request_manifest.get("request_payload"))
    rollouts: list[dict[str, Any]] = []
    if mp4_path and mp4_path.is_file():
        rollouts.append(
            {
                "rollout_id": "deepinfra_cosmos3_rollout_0001",
                "policy_id": "deepinfra_cosmos3_nano_api",
                "scenario_eval_run_id": "deepinfra_cosmos3_api_support_rollout",
                "task_prompt": request_payload.get("prompt"),
                "generated_video_path": "/workspace/runtime_output/deepinfra_cosmos3_generated_rollout.mp4",
                "provider_original_generated_video_name": "deepinfra_cosmos3_generated_rollout.mp4",
                "success_label_source": "generated_video_requires_external_review",
            }
        )
    model_output = bool(
        runtime_result.get("learned_wam_model_ran")
        and _mapping(runtime_result.get("truth_boundary")).get("generated_video_is_model_output")
    )
    provider_output = {
        "schema_version": "deepinfra_cosmos3_provider_output.v1",
        "status": status,
        "adapter_id": "deepinfra_cosmos3_nano_api",
        "provider": DEEPINFRA_PROVIDER_NAME,
        "model_id": _deepinfra_model_id(),
        "rollouts": rollouts,
        "blockers": sorted(set(str(item) for item in blockers if str(item))),
        "fresh_model_run_claimed": bool(status == "completed" and model_output),
        "fresh_provider_model_run_claimed": bool(status == "completed" and model_output),
        "fresh_model_command_executed_this_invocation": bool(
            status == "completed" and model_output
        ),
        "fresh_provider_launch_attempted": True,
        "provider_output_replayed": False,
        "api_request_model_id": _deepinfra_model_id(),
        "visual_quality_report": "wam_rollout_visual_quality_report.json",
        "claim_boundary": _deepinfra_claim_boundary(),
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(job_dir / "wam_provider_output.json", provider_output)
    return provider_output


def _write_deepinfra_checksums(
    *,
    job_dir: Path,
    paths: Sequence[Path | None],
    existing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        if path is None:
            continue
        resolved = Path(path).expanduser()
        key = str(resolved)
        if key in seen or not resolved.is_file():
            continue
        seen.add(key)
        artifacts.append(
            {
                "path": str(resolved),
                "name": resolved.name,
                "size_bytes": resolved.stat().st_size,
                "sha256": _sha256_file(resolved),
            }
        )
    payload = {
        "schema_version": "deepinfra_cosmos3_artifact_checksums.v1",
        "generated_at": utc_now_iso(),
        "provider": DEEPINFRA_PROVIDER_NAME,
        "model_id": _deepinfra_model_id(),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "previous_artifact_count": _safe_int(_mapping(existing).get("artifact_count"), 0),
        "claim_boundary": {
            "checksums_prove_local_artifact_integrity_only": True,
            "checksums_are_not_visual_quality_or_task_success": True,
            "raw_secret_values_recorded": False,
        },
        "raw_secret_values_recorded": False,
    }
    write_json(job_dir / "deepinfra_cosmos3_artifact_checksums.json", payload)
    return payload


def _write_deepinfra_output_zip(*, output_zip_path: Path, paths: Sequence[Path | None]) -> None:
    ensure_dir(output_zip_path.parent)
    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path is None:
                continue
            resolved = Path(path).expanduser()
            if not resolved.is_file():
                continue
            archive_name = resolved.name
            if archive_name == "deepinfra_cosmos3_generated_rollout.mp4":
                archive_name = "deepinfra_cosmos3_generated_rollout.mp4"
            archive.write(resolved, archive_name)


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
    if provider == "runpod":
        teardown_path = job_dir / "runpod_wam_async_delete_manifest.json"
        stop_teardown_path = job_dir / "runpod_wam_async_stop_manifest.json"
        if not teardown_path.is_file() and stop_teardown_path.is_file():
            teardown_path = stop_teardown_path
    elif provider == DEEPINFRA_PROVIDER_NAME:
        teardown_path = job_dir / "deepinfra_cosmos3_teardown_manifest.json"
    else:
        teardown_path = job_dir / "vast_teardown_manifest.json"
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
        budget_ledger_path=_existing_path(
            job_dir / "vast_budget_ledger.json",
            job_dir / "deepinfra_cosmos3_cost_control_ledger.json",
            job_dir / "wam_provider_cost_control_ledger.json",
        ),
        teardown_manifest_path=str(teardown_path) if teardown_path.is_file() else None,
        teardown_status=_string(teardown_payload.get("status")) or (
            "completed"
            if manifest.get("teardown_performed")
            else ("not_required" if provider == DEEPINFRA_PROVIDER_NAME else "not_requested")
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
            "allowed_machine_ids": list(spec.allowed_machine_ids),
            "min_reliability": spec.min_reliability,
            "require_direct_port": spec.require_direct_port,
            "preferred_gpu_keywords": list(spec.preferred_gpu_keywords),
            "preferred_geolocation_regex": spec.preferred_geolocation_regex,
            "prefer_isaac_rt": spec.prefer_isaac_rt,
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
                allowed_machine_ids=spec.allowed_machine_ids,
                min_reliability=spec.min_reliability,
                require_direct_port=spec.require_direct_port,
                preferred_gpu_keywords=spec.preferred_gpu_keywords,
                preferred_geolocation_regex=spec.preferred_geolocation_regex,
                prefer_isaac_rt=spec.prefer_isaac_rt,
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
    if key in {"deepinfra", "deepinfra_cosmos3", "deepinfra_cosmos3_nano"}:
        return DeepInfraCosmos3NanoProvider()
    if key == "vast":
        return VastWamComputeProvider()
    if key == "runpod":
        return RunPodWamComputeProvider()
    raise ValueError(
        "unknown_wam_compute_provider:%r (known: deepinfra, runpod, vast, auto)" % name
    )


def list_wam_compute_providers() -> list[dict[str, Any]]:
    return [
        DeepInfraCosmos3NanoProvider().available(),
        RunPodWamComputeProvider().available(),
        VastWamComputeProvider().available(),
    ]


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
