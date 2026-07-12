"""Live adapters for the Isaac startup supervisor and worker startup gates.

The supervisor owns allocation/spend/quarantine/teardown.  This module adapts
the existing GPU provider surface to that contract and validates the artifacts
uploaded by the *same* worker before the full kitchen runner is accepted.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import uuid
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping


STARTUP_GATE_SUMMARY_FILENAME = "supervised_startup_gates.json"
FAST_PREFLIGHT_FILENAME = "isaac_worker_runtime_preflight.json"
KITCHEN_GATE_FILENAME = "kitchen_asset_gate/kitchen_asset_startup_gate.json"
REVIEW_CANARY_FILENAME = "review_renderer_canary/isaac_review_renderer_canary.json"


def resolve_image_manifest_checksum(
    diagnostic: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Bind supervision to the exact registry diagnostic file on disk."""
    detail = dict(diagnostic) if isinstance(diagnostic, Mapping) else {}
    path_text = str(detail.get("path") or "").strip()
    path = Path(path_text).expanduser() if path_text else None
    blockers: list[str] = []
    if detail.get("metadata_available_for_selected_image") is not True:
        blockers.append("startup_image_manifest_diagnostic_not_for_selected_image")
    if path is None or not path.is_file():
        blockers.append("startup_image_manifest_diagnostic_file_missing")
    expected_digest = str(detail.get("diagnostic_sha256") or "").strip()
    expected_bytes = detail.get("diagnostic_bytes")
    if len(expected_digest) != 64:
        blockers.append("startup_image_manifest_diagnostic_identity_missing")
    digest = ""
    if not blockers and path is not None:
        try:
            raw = path.read_bytes()
        except OSError:
            blockers.append("startup_image_manifest_diagnostic_file_unreadable")
        else:
            digest = hashlib.sha256(raw).hexdigest()
            if digest != expected_digest or len(raw) != expected_bytes:
                blockers.append("startup_image_manifest_diagnostic_identity_changed")
    return {
        "status": "passed" if not blockers else "blocked",
        "path": str(path) if path is not None else None,
        "sha256": digest or None,
        "blockers": blockers,
    }


def provider_gpu_types(
    *,
    provider_name: str,
    request: Mapping[str, Any],
    capacity: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    """Resolve the ordered provider-native GPU/size candidates for supervision."""
    values: list[str] = []
    provider = str(provider_name or "").strip().lower()
    capacity_detail = dict(capacity) if isinstance(capacity, Mapping) else {}
    if provider == "digitalocean":
        for row in capacity_detail.get("viable_size_regions") or []:
            if isinstance(row, Mapping):
                values.append(str(row.get("size") or "").strip())
        values.append(str(request.get("size") or "").strip())
    elif provider == "runpod":
        for row in capacity_detail.get("viable_gpu_types") or []:
            if isinstance(row, Mapping):
                values.append(str(row.get("gpu_type_id") or "").strip())
        raw = request.get("gpuTypeIds")
        if isinstance(raw, (list, tuple)):
            values.extend(str(item).strip() for item in raw)
    ordered: list[str] = []
    for value in values:
        if value and value not in ordered:
            ordered.append(value)
    return tuple(ordered)


def _json_member(archive: zipfile.ZipFile, name: str) -> dict[str, Any] | None:
    try:
        value = json.loads(archive.read(name).decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def inspect_bootstrap_archive(
    data: bytes,
    *,
    expected_launch_session_id: str,
) -> dict[str, Any]:
    """Classify the early worker marker while rejecting stale output."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError):
        return {"status": "waiting", "reason": "provider_output_archive_unreadable"}
    with archive:
        bootstrap = _json_member(archive, "bootstrap.json")
    if bootstrap is None:
        return {"status": "waiting", "reason": "bootstrap_marker_missing"}
    observed = str(bootstrap.get("launch_session_id") or "").strip()
    if observed != str(expected_launch_session_id or "").strip():
        return {"status": "stale_marker", "reason": "bootstrap_launch_session_mismatch"}
    return {
        "status": "marker_verified",
        "phase": bootstrap.get("phase"),
        "launch_session_id": observed,
    }


def wait_for_startup_marker(
    *,
    fetch_archive: Callable[[], bytes],
    expected_launch_session_id: str,
    timeout_seconds: float,
    poll_seconds: float = 15.0,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Poll the attempt-bound bootstrap marker without accepting stale output."""
    started = clock()
    last: dict[str, Any] = {"status": "waiting", "reason": "not_polled"}
    stale_seen = False
    while clock() - started < max(0.0, float(timeout_seconds)):
        try:
            last = inspect_bootstrap_archive(
                fetch_archive(),
                expected_launch_session_id=expected_launch_session_id,
            )
        except Exception as exc:
            last = {
                "status": "waiting",
                "reason": "provider_output_fetch_failed",
                "error_type": type(exc).__name__,
            }
        if last.get("status") == "marker_verified":
            last["elapsed_seconds"] = round(clock() - started, 3)
            return last
        stale_seen = stale_seen or last.get("status") == "stale_marker"
        sleeper(max(0.01, float(poll_seconds)))
    return {
        "status": "stale_marker" if stale_seen else "no_runtime",
        "reason": (
            "only_stale_bootstrap_marker_observed"
            if stale_seen
            else "bootstrap_marker_timeout"
        ),
        "last_observation": last,
        "elapsed_seconds": round(clock() - started, 3),
    }


def _preflight_identity(preflight: Mapping[str, Any] | None) -> dict[str, Any]:
    detail = dict(preflight) if isinstance(preflight, Mapping) else {}
    gpu_name = None
    driver_version = None
    for row in detail.get("checks") or []:
        if not isinstance(row, Mapping):
            continue
        for candidate in row.get("gpu_inventory") or []:
            if isinstance(candidate, Mapping):
                gpu_name = gpu_name or candidate.get("gpu_name")
                driver_version = driver_version or candidate.get("driver_version")
    return {"gpu_name": gpu_name, "driver_version": driver_version}


def _startup_failure_class(
    *,
    blockers: list[str],
    preflight: Mapping[str, Any] | None,
) -> str:
    all_blockers = set(blockers)
    if "isaac_sim_6_rtx_driver_unsupported" in all_blockers:
        return "driver_incompatible"
    if all_blockers.intersection({"nvidia_smi_unavailable", "nvidia_smi_failed"}):
        return "cuda_unavailable"
    if any("blank_black" in item or "flat" in item for item in all_blockers):
        return "empty_frame"
    if any("rtx" in item for item in all_blockers):
        return "rtx_init_failed"
    if isinstance(preflight, Mapping) and preflight.get("status") != "passed":
        return "gpu_unhealthy"
    # Asset/contract failures are deliberately not machine-quarantined.
    return "canary_blocked"


def validate_startup_gate_archive(
    data: bytes,
    *,
    expected_launch_session_id: str,
    expected_image_digest: str,
) -> dict[str, Any]:
    """Validate one uploaded worker archive without trusting its summary alone."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError):
        return {"status": "waiting", "reason": "provider_output_archive_unreadable"}
    with archive:
        bootstrap = _json_member(archive, "bootstrap.json") or {}
        observed_nonce = str(bootstrap.get("launch_session_id") or "")
        if observed_nonce and observed_nonce != expected_launch_session_id:
            return {"status": "waiting", "reason": "stale_launch_session_archive"}
        summary = _json_member(archive, STARTUP_GATE_SUMMARY_FILENAME)
        if summary is None:
            phase = str(bootstrap.get("phase") or "")
            if phase == "startup_gates_blocked":
                return {
                    "status": "blocked",
                    "failure_class": "review_renderer_failed",
                    "blockers": list(bootstrap.get("blockers") or ["startup_gates_blocked"]),
                }
            return {"status": "waiting", "reason": "startup_gate_summary_missing"}

        blockers: list[str] = []
        if str(summary.get("launch_session_id") or "") != expected_launch_session_id:
            blockers.append("startup_gate_launch_session_mismatch")
        if str(summary.get("image_digest") or "") != expected_image_digest:
            blockers.append("startup_gate_image_digest_mismatch")

        preflight = _json_member(archive, FAST_PREFLIGHT_FILENAME)
        kitchen = _json_member(archive, KITCHEN_GATE_FILENAME)
        review = _json_member(archive, REVIEW_CANARY_FILENAME)
        if preflight is None or preflight.get("status") != "passed":
            blockers.append("fast_startup_canary_not_passed")
        if preflight is not None:
            if str(preflight.get("launch_session_id") or "") != expected_launch_session_id:
                blockers.append("fast_startup_canary_launch_session_mismatch")
            if str(preflight.get("image_digest") or "") != expected_image_digest:
                blockers.append("fast_startup_canary_image_digest_mismatch")
        if kitchen is None or kitchen.get("status") != "completed":
            blockers.append("kitchen_asset_startup_gate_not_passed")
        if kitchen is not None:
            if str(kitchen.get("launch_session_id") or "") != expected_launch_session_id:
                blockers.append("kitchen_asset_gate_launch_session_mismatch")
            if str(kitchen.get("image_digest") or "") != expected_image_digest:
                blockers.append("kitchen_asset_gate_image_digest_mismatch")
        if review is None or review.get("status") != "passed":
            blockers.append("review_renderer_canary_not_passed")
        if review is not None:
            if str(review.get("launch_session_id") or "") != expected_launch_session_id:
                blockers.append("review_renderer_launch_session_mismatch")
            if str(review.get("image_digest") or "") != expected_image_digest:
                blockers.append("review_renderer_image_digest_mismatch")
            if review.get("isaac_review_renderer_operational") is not True:
                blockers.append("review_renderer_not_operational")
        blockers.extend(str(item) for item in (summary.get("blockers") or []) if str(item))
        blockers = sorted(set(blockers))
        identity = _preflight_identity(preflight)
        return {
            "status": "passed" if not blockers else "blocked",
            "blockers": blockers,
            "failure_class": (
                None
                if not blockers
                else _startup_failure_class(blockers=blockers, preflight=preflight)
            ),
            **identity,
            "launch_session_id": expected_launch_session_id,
            "image_digest": expected_image_digest,
            "checks": {
                "fast_startup_canary": preflight,
                "kitchen_asset_startup_gate": kitchen,
                "review_renderer_canary": review,
            },
            "evidence_paths": [],
        }


def wait_for_startup_gates(
    *,
    fetch_archive: Callable[[], bytes],
    expected_launch_session_id: str,
    expected_image_digest: str,
    timeout_seconds: float,
    poll_seconds: float = 15.0,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    started = clock()
    last: dict[str, Any] = {"status": "waiting", "reason": "not_polled"}
    while clock() - started < max(0.0, float(timeout_seconds)):
        try:
            last = validate_startup_gate_archive(
                fetch_archive(),
                expected_launch_session_id=expected_launch_session_id,
                expected_image_digest=expected_image_digest,
            )
        except Exception as exc:  # provider object may not exist yet / upload in flight
            last = {
                "status": "waiting",
                "reason": "provider_output_fetch_failed",
                "error_type": type(exc).__name__,
            }
        if last.get("status") in {"passed", "blocked"}:
            last["elapsed_seconds"] = round(clock() - started, 3)
            return last
        sleeper(max(0.01, float(poll_seconds)))
    return {
        "status": "blocked",
        "failure_class": "no_runtime",
        "blockers": ["supervised_startup_gate_timeout"],
        "last_observation": last,
        "elapsed_seconds": round(clock() - started, 3),
    }


class SupervisedProviderAdapter:
    """Adapt ``GpuRenderProvider.launch`` to ``StartupSupervisor.allocate``."""

    def __init__(
        self,
        *,
        provider: Any,
        job_dir: str | Path,
        base_request: Mapping[str, Any],
        resource_name_prefix: str,
    ) -> None:
        self.provider = provider
        self.name = str(getattr(provider, "name", "unknown"))
        self.job_dir = Path(job_dir)
        self.base_request = deepcopy(dict(base_request))
        self.resource_name_prefix = str(resource_name_prefix)

    @staticmethod
    def _bind_nonce(request: dict[str, Any], nonce: str) -> None:
        env = request.setdefault("env", {})
        if isinstance(env, dict):
            env["BLUEPRINT_LAUNCH_SESSION_ID"] = nonce
        create_payload = request.get("create_payload")
        if isinstance(create_payload, dict):
            create_env = create_payload.setdefault("env", {})
            if isinstance(create_env, dict):
                create_env["BLUEPRINT_LAUNCH_SESSION_ID"] = nonce

    def allocate(self, *, gpu_type: str, attempt_id: str, launch_nonce: str) -> dict:
        request = deepcopy(self.base_request)
        safe_attempt = "".join(c if c.isalnum() or c == "-" else "-" for c in attempt_id)
        request["name"] = f"{self.resource_name_prefix}-{safe_attempt}"[:63]
        if isinstance(request.get("gpuTypeIds"), list):
            request["gpuTypeIds"] = [str(gpu_type)]
        if self.name == "digitalocean":
            request["size"] = str(gpu_type)
        self._bind_nonce(request, launch_nonce)
        capacity = dict(self.provider.capacity_preflight(request))
        # Catalog/plan visibility is advisory and never a reservation.  Actual
        # create is the authoritative capacity result (and a create failure
        # without an allocation id is explicitly no-spend).
        (self.job_dir / "launch_session_nonce.txt").write_text(launch_nonce, encoding="utf-8")
        result = dict(
            self.provider.launch(
                self.job_dir,
                request,
                cold=True,
                allow_cold_fallback=False,
            )
        )
        result["capacity_preflight"] = capacity
        result["catalog_capacity_was_advisory"] = True
        result["authoritative_capacity_source"] = "provider_create_response"
        return result

    def capacity_preflight(
        self, *, gpu_type: str, attempt_id: str, launch_nonce: str
    ) -> dict[str, Any]:
        """Read-only attempt-specific probe used by the supervisor before allocation."""
        request = deepcopy(self.base_request)
        request["name"] = f"{self.resource_name_prefix}-{attempt_id}"[:63]
        if isinstance(request.get("gpuTypeIds"), list):
            request["gpuTypeIds"] = [str(gpu_type)]
        if self.name == "digitalocean":
            request["size"] = str(gpu_type)
        self._bind_nonce(request, launch_nonce)
        return dict(self.provider.capacity_preflight(request))

    def inspect(self, instance_id: str) -> dict:
        return dict(self.provider.inspect(instance_id))

    def terminate(self, instance_id: str) -> dict:
        return dict(self.provider.terminate(instance_id))

    def inventory(self) -> dict[str, Any]:
        result = dict(
            self.provider.billable_inventory(name_prefix=self.resource_name_prefix)
        )
        if result.get("api_confirmed") is not True:
            # The supervisor interprets any positive count as fail-closed.  Do
            # not let an unavailable inventory API silently become zero.
            result["live_resource_count"] = 1
            result.setdefault("blockers", []).append(
                "provider_billable_inventory_not_api_confirmed"
            )
        return result

def run_parity_startup_transaction(
    *,
    provider: Any,
    job_dir: Path,
    request_body: Mapping[str, Any],
    out_dir: Path,
    image_ref: str,
    image_manifest_diagnostic: Mapping[str, Any],
    capacity: Mapping[str, Any] | None,
    max_attempts: int,
    marker_timeout_seconds: int,
    max_seconds: int,
    post_marker_progress_timeout: int,
    prelaunch_spend_guard: Mapping[str, Any],
) -> dict[str, Any]:
    """Run startup gates and the full job on one supervisor-owned allocation."""
    from .isaac_g1_kitchen_parity_job import (
        MAX_WARM_READINESS_ARCHIVE_BYTES,
        _fetch_provider_artifact_bytes,
        _teardown_proof_from_watch_result,
        close_pending_teardown,
        watch_and_collect,
    )
    from .isaac_startup_supervisor import (
        TERMINAL_MODE_PROMOTE,
        StartupSupervisorRequest,
        run_startup_supervisor,
    )
    run_id = f"isaac-g1-{uuid.uuid4().hex}"
    launch_nonce = uuid.uuid4().hex
    resource_prefix = "blueprint-isaac-g1-supervised"
    resolved_gpu_types = provider_gpu_types(
        provider_name=str(getattr(provider, "name", "unknown")),
        request=request_body,
        capacity=capacity,
    )
    checksum = resolve_image_manifest_checksum(image_manifest_diagnostic)
    early_blockers = list(checksum.get("blockers") or [])
    if not resolved_gpu_types:
        early_blockers.append("supervised_startup_provider_gpu_types_missing")
    if early_blockers:
        return {
            "startup_supervisor": {
                "status": "blocked",
                "blockers": early_blockers,
                "image_manifest_checksum": checksum,
            },
            "launch": {"status": "blocked", "blockers": early_blockers},
            "watch": None,
        }
    adapter = SupervisedProviderAdapter(
        provider=provider,
        job_dir=job_dir,
        base_request=request_body,
        resource_name_prefix=resource_prefix,
    )
    digest = image_ref.split("@", 1)[1]
    diagnostic_checksum = str(checksum["sha256"])
    hourly_rate = float(prelaunch_spend_guard.get("max_hourly_rate_usd") or 0.0)
    spend_cap = float(prelaunch_spend_guard.get("requested_budget_usd") or 0.0)
    reserved_seconds = float(
        prelaunch_spend_guard.get("render_budget_seconds") or 0.0
    ) + float(
        prelaunch_spend_guard.get("startup_budget_per_attempt_seconds") or 0.0
    ) + float(
        prelaunch_spend_guard.get(
            "teardown_reconciliation_grace_per_attempt_seconds"
        )
        or 0.0
    )
    result_holder: dict[str, Any] = {}

    def inventory() -> Mapping[str, Any]:
        return adapter.inventory()

    def wait_for_marker(*, launch_nonce: str, **_: Any) -> dict[str, Any]:
        get_url = (job_dir / "provider_output_get_url.txt").read_text(
            encoding="utf-8"
        ).strip()
        return wait_for_startup_marker(
            fetch_archive=lambda: _fetch_provider_artifact_bytes(
                get_url,
                timeout=60,
                max_bytes=MAX_WARM_READINESS_ARCHIVE_BYTES,
            ),
            expected_launch_session_id=launch_nonce,
            timeout_seconds=marker_timeout_seconds,
            poll_seconds=15.0,
        )

    def run_canary(*, launch_nonce: str, **_: Any) -> dict[str, Any]:
        get_url = (job_dir / "provider_output_get_url.txt").read_text(encoding="utf-8").strip()
        return wait_for_startup_gates(
            fetch_archive=lambda: _fetch_provider_artifact_bytes(
                get_url,
                timeout=60,
                max_bytes=MAX_WARM_READINESS_ARCHIVE_BYTES,
            ),
            expected_launch_session_id=launch_nonce,
            expected_image_digest=digest,
            timeout_seconds=max(60, marker_timeout_seconds),
            poll_seconds=15.0,
        )

    def full_job(allocation: Mapping[str, Any]) -> Mapping[str, Any]:
        instance_id = str(allocation["instance_id"])
        render_out = out_dir / "render_output"
        watch = watch_and_collect(
            job_dir,
            render_out,
            instance_id,
            provider=provider,
            max_seconds=max_seconds,
            stop_on_success=False,
            preserve_instance=False,
            preserve_blocked_instance=False,
            progress_timeout_seconds=post_marker_progress_timeout,
        )
        pending_path = str(allocation.get("pending_teardown_record") or "")
        if pending_path:
            proof = _teardown_proof_from_watch_result(
                provider_name=str(getattr(provider, "name", "unknown")),
                instance_id=instance_id,
                watch=watch,
            )
            closure = close_pending_teardown(pending_path, proof)
            watch["teardown_proof"] = proof
            watch["pending_teardown_record"] = pending_path
            watch["pending_teardown_status"] = closure.get("status")
        post_inventory = adapter.inventory()
        watch["post_full_job_inventory"] = post_inventory
        result_holder["watch"] = watch
        if watch.get("status") != "completed":
            raise RuntimeError("supervised_full_job_blocked")
        if int(post_inventory.get("live_resource_count") or 0) != 0:
            raise RuntimeError("supervised_full_job_residual_billable_resource")
        return {"status": "completed", "watch": watch}

    request = StartupSupervisorRequest(
        run_id=run_id,
        launch_nonce=launch_nonce,
        provider=str(getattr(provider, "name", "unknown")),
        gpu_types=resolved_gpu_types,
        image_ref=image_ref,
        image_manifest_checksum=diagnostic_checksum,
        max_attempts=max(1, int(max_attempts)),
        wall_clock_cap_seconds=max(1.0, reserved_seconds),
        total_spend_cap_usd=spend_cap,
        hourly_rate_usd=hourly_rate,
        per_attempt_reserved_seconds=max(1.0, reserved_seconds),
        terminal_mode=TERMINAL_MODE_PROMOTE,
        pending_teardown_max_age_seconds=max(
            7200, int(reserved_seconds * max(1, int(max_attempts)) + 1800)
        ),
        full_job_request={"job": "isaac_g1_kitchen_parity"},
    )
    supervisor_out = out_dir / "startup_supervisor"
    supervisor = run_startup_supervisor(
        request=request,
        provider_client=adapter,
        out_dir=supervisor_out,
        inventory=inventory,
        wait_for_marker=wait_for_marker,
        run_canary=run_canary,
        upload_failure_artifacts=lambda **_: {
            "status": "provider_output_preserved",
            "job_dir": str(job_dir),
        },
        full_job=full_job,
        quarantine_registry_dir=out_dir / "machine_quarantine",
        # Supervised descendants share the process-independent global registry
        # with the parent paid-lane lease.  A job-local registry would hide an
        # open child teardown obligation from subsequent agents.
        teardown_registry_dir=None,
    )
    receipt = supervisor.get("ownership_transfer_receipt") or {}
    launch = {
        "status": "launched" if supervisor.get("status") == "passed_and_promoted" else "blocked",
        "instance_id": receipt.get("instance_id"),
        "mode": "supervised_same_allocation",
        "pending_teardown_record": receipt.get("pending_teardown_record"),
        "blockers": list(supervisor.get("blockers") or []),
    }
    return {
        "startup_supervisor": supervisor,
        "launch": launch,
        "watch": result_holder.get("watch"),
    }




def run_live_startup_supervisor(
    *,
    provider: Any,
    job_dir: str | Path,
    base_request: Mapping[str, Any],
    out_dir: str | Path,
    run_id: str,
    launch_nonce: str,
    image_ref: str,
    image_manifest_checksum: str,
    gpu_types: tuple[str, ...],
    max_attempts: int,
    wall_clock_cap_seconds: float,
    total_spend_cap_usd: float,
    hourly_rate_usd: float,
    per_attempt_reserved_seconds: float,
    marker_timeout_seconds: float,
    canary_timeout_seconds: float,
    fetch_archive: Callable[[], bytes],
    marker_poll_seconds: float = 15.0,
    canary_poll_seconds: float = 15.0,
    quarantine_registry_dir: str | Path | None = None,
    teardown_registry_dir: str | Path | None = None,
    full_job: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Invoke the real supervisor with the provider and object-store adapters."""
    from .isaac_startup_supervisor import (
        TERMINAL_MODE_PROMOTE,
        StartupSupervisorRequest,
        run_startup_supervisor,
    )

    resource_prefix = str(base_request.get("name") or "blueprint-isaac-g1-kitchen-parity")
    adapter = SupervisedProviderAdapter(
        provider=provider,
        job_dir=job_dir,
        base_request=base_request,
        resource_name_prefix=resource_prefix,
    )
    request = StartupSupervisorRequest(
        run_id=str(run_id),
        launch_nonce=str(launch_nonce),
        provider=adapter.name,
        gpu_types=tuple(gpu_types),
        image_ref=str(image_ref),
        image_manifest_checksum=str(image_manifest_checksum),
        max_attempts=int(max_attempts),
        wall_clock_cap_seconds=float(wall_clock_cap_seconds),
        total_spend_cap_usd=float(total_spend_cap_usd),
        hourly_rate_usd=float(hourly_rate_usd),
        per_attempt_reserved_seconds=float(per_attempt_reserved_seconds),
        terminal_mode=TERMINAL_MODE_PROMOTE,
        full_job_request={"reuse_passing_warm_allocation": True},
    )

    def _marker(**context: Any) -> dict[str, Any]:
        return wait_for_startup_marker(
            fetch_archive=fetch_archive,
            expected_launch_session_id=str(context["launch_nonce"]),
            timeout_seconds=marker_timeout_seconds,
            poll_seconds=marker_poll_seconds,
        )

    def _canary(**context: Any) -> dict[str, Any]:
        return wait_for_startup_gates(
            fetch_archive=fetch_archive,
            expected_launch_session_id=str(context["launch_nonce"]),
            expected_image_digest=request.image_digest,
            timeout_seconds=canary_timeout_seconds,
            poll_seconds=canary_poll_seconds,
        )

    def _promote(context: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "status": "accepted",
            "instance_id": context.get("instance_id"),
            "launch_nonce": context.get("launch_nonce"),
            "pending_teardown_record": context.get("pending_teardown_record"),
            "same_warm_allocation_reused": True,
            "second_cold_launch_performed": False,
        }

    return run_startup_supervisor(
        request=request,
        provider_client=adapter,
        out_dir=out_dir,
        inventory=adapter.inventory,
        wait_for_marker=_marker,
        run_canary=_canary,
        full_job=full_job or _promote,
        quarantine_registry_dir=quarantine_registry_dir,
        teardown_registry_dir=teardown_registry_dir,
    )
