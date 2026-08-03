"""Grant-gated Vast lifecycle for typed reconstruction pose/trainer operations."""

from __future__ import annotations

import hashlib
import base64
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence
import urllib.error

from .common import ensure_dir, utc_now_iso, write_json
from .canonical_3dgs_transport import (
    Canonical3DGSTransportError,
    validate_canonical_3dgs_transport_receipt,
)
from .canonical_3dgs_vast_output import validate_canonical_3dgs_vast_output_bundle
from .decision_evidence_contracts import canonical_digest
from .gpu_render_providers import GpuRenderProvider, RenderLaunchSpec
from .paid_lane_guard import (
    bind_pending_teardown_instance,
    cancel_pending_teardown,
    close_pending_teardown,
    load_pending_teardowns,
    mark_pending_teardown_ambiguous,
    open_pending_teardown,
)
from .paid_provider_lane_lease import (
    acquire_paid_provider_lane_lease,
    build_paid_provider_lane_reconciliation,
    release_paid_provider_lane_lease,
)
from .paid_resource_admission import (
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
)
from .reconstruction_gpu_operation_bootstrap import MAX_OUTPUT_BUNDLE_BYTES
from .reconstruction_gpu_operation_bundle import (
    ReconstructionGpuOperationBundleError,
    validate_reconstruction_gpu_operation_bundle_receipt,
)
from .reconstruction_gpu_operation_output import (
    ReconstructionGpuOperationOutputError,
    validate_reconstruction_gpu_operation_output_bundle,
)
from .safe_outbound_http import (
    SafeOutboundHttpError,
    SafeHttpFileTransfer,
    download_file_observed,
    presigned_transfer_policy,
    validate_outbound_url,
)


SCHEMA_VERSION = "reconstruction_vast_operation_execution.v1"
REPLAY_SCHEMA_VERSION = "reconstruction_vast_operation_replay.v1"
TEARDOWN_SCHEMA_VERSION = "reconstruction_vast_operation_teardown.v1"
PROVIDER_ZERO_SCHEMA_VERSION = "reconstruction_vast_operation_provider_zero.v1"
PAID_LANE = "reconstruction_gpu_canary"
NAME_PREFIX = "blueprint-reconstruction-"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_EXPECTED_RESULTS = {
    "pose_canary": "pose_estimation_result.v1",
    "trainer_canary": "reconstruction_training_result.v1",
}


class ReconstructionVastOperationError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_receipt_file(root: Path, receipt: Mapping[str, Any]) -> tuple[Path, str]:
    path = root / "operation_bundle_receipt.json"
    payload = (
        json.dumps(dict(receipt), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    return path, "sha256:" + hashlib.sha256(payload).hexdigest()


def _watchdog_valid(
    watchdog: Mapping[str, Any], *, now_epoch: float, hard_ttl_seconds: int
) -> bool:
    try:
        pid = int(watchdog.get("pid") or 0)
        deadline = float(watchdog.get("deadline_epoch") or 0)
    except (TypeError, ValueError):
        return False
    if (
        watchdog.get("status") != "armed"
        or watchdog.get("independent_process") is not True
        or str(watchdog.get("name_prefix") or "") != NAME_PREFIX
        or pid <= 0
        or deadline < now_epoch + hard_ttl_seconds
    ):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _default_output_fetcher(url: str, destination: Path) -> SafeHttpFileTransfer:
    try:
        return download_file_observed(
            url,
            output_path=destination,
            max_bytes=MAX_OUTPUT_BUNDLE_BYTES,
            timeout_seconds=300,
            policy=presigned_transfer_policy(url),
        )
    except urllib.error.HTTPError as exc:
        if exc.code in {404, 409}:
            raise FileNotFoundError("reconstruction_operation_output_not_ready") from exc
        raise


def _bootstrap_script(*, canonical_splatfacto: bool = False) -> str:
    if canonical_splatfacto:
        return """set -euo pipefail
python - <<'PY'
import hashlib, json, os, pathlib, urllib.request, zipfile
bundle = pathlib.Path('/tmp/canonical_3dgs_transport.zip')
receipt = pathlib.Path('/tmp/canonical_3dgs_transport_receipt.json')
for url_name, path in (
    ('BLUEPRINT_RECONSTRUCTION_INPUT_BUNDLE_GET_URL', bundle),
    ('BLUEPRINT_RECONSTRUCTION_INPUT_RECEIPT_GET_URL', receipt),
):
    with urllib.request.urlopen(os.environ[url_name], timeout=300) as source, path.open('xb') as target:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk)
def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return 'sha256:' + value.hexdigest()
if digest(bundle) != os.environ['BLUEPRINT_RECONSTRUCTION_INPUT_BUNDLE_DIGEST']:
    raise SystemExit('canonical_transport_digest_mismatch')
if digest(receipt) != os.environ['BLUEPRINT_RECONSTRUCTION_INPUT_RECEIPT_FILE_DIGEST']:
    raise SystemExit('canonical_receipt_file_digest_mismatch')
control = json.loads(receipt.read_text())
wheel_name = control.get('worker_wheel_filename')
wheel_member = control.get('worker_wheel_archive_path')
wheel_digest = control.get('worker_wheel_digest')
if not wheel_name or wheel_member != 'worker/' + wheel_name:
    raise SystemExit('canonical_worker_wheel_binding_missing')
wheel_path = pathlib.Path('/tmp') / wheel_name
with zipfile.ZipFile(bundle) as archive:
    if wheel_member not in archive.namelist():
        raise SystemExit('canonical_worker_wheel_missing')
    wheel_path.write_bytes(archive.read(wheel_member))
if digest(wheel_path) != wheel_digest:
    raise SystemExit('canonical_worker_wheel_digest_mismatch')
pathlib.Path('/tmp/canonical_worker_wheel_path').write_text(str(wheel_path))
PY
python -m pip install --no-deps "$(cat /tmp/canonical_worker_wheel_path)"
export BLUEPRINT_CANONICAL_BUNDLE_PATH=/tmp/canonical_3dgs_transport.zip
export BLUEPRINT_CANONICAL_RECEIPT_PATH=/tmp/canonical_3dgs_transport_receipt.json
python -m blueprint_pipeline.canonical_3dgs_vast_bootstrap
"""
    return """set -euo pipefail
health=/tmp/blueprint_reconstruction_operation_health.json
python -m blueprint_pipeline.reconstruction_worker_image_healthcheck --output "$health"
python -m blueprint_pipeline.reconstruction_gpu_operation_bootstrap
"""


def _load_object(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionVastOperationError([code]) from exc
    if not isinstance(value, Mapping):
        raise ReconstructionVastOperationError([code])
    return dict(value)


def _validate_bindings(
    *, bound_request: Mapping[str, Any], bundle_receipt: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = json.loads(json.dumps(dict(bound_request)))
    canonical_splatfacto = (
        request.get("execution_adapter_id") == "canonical_splatfacto_vast_v1"
    )
    try:
        receipt = (
            validate_canonical_3dgs_transport_receipt(bundle_receipt)
            if canonical_splatfacto
            else validate_reconstruction_gpu_operation_bundle_receipt(bundle_receipt)
        )
    except (ReconstructionGpuOperationBundleError, Canonical3DGSTransportError) as exc:
        codes = getattr(exc, "codes", (type(exc).__name__,))
        raise ReconstructionVastOperationError(
            [f"reconstruction_vast_operation_receipt_invalid:{code}" for code in codes]
        ) from exc
    operation = str(request.get("operation") or "")
    blockers: list[str] = []
    if request.get("bound_request_digest") != canonical_digest(
        request, digest_field="bound_request_digest"
    ):
        blockers.append("reconstruction_vast_operation_bound_request_digest_mismatch")
    expected_runtime_schema = (
        "canonical_3dgs_vast_runtime_result.v1"
        if canonical_splatfacto
        else _EXPECTED_RESULTS.get(operation)
    )
    if (
        request.get("bound_provider") != "vast"
        or request.get("provider_mutation_authorized") is not True
        or operation not in _EXPECTED_RESULTS
        or request.get("expected_runtime_result_schema") != expected_runtime_schema
        or request.get("candidate_may_read_hidden_heldout") is not False
        or request.get("trainer_may_grade_heldout") is not False
        or request.get("proof_effect") != "none"
    ):
        blockers.append("reconstruction_vast_operation_bound_request_not_executable")
    bindings = [
        ("operation", "operation"),
        ("operation_request_digest", "operation_request_digest"),
        ("operation_input_bundle_digest", "operation_input_bundle_digest"),
        ("worker_image_digest", "worker_image_digest"),
        ("source_commit_sha", "source_commit_sha"),
        ("reconstruction_dataset_digest", "reconstruction_dataset_digest"),
        ("frozen_split_digest", "frozen_split_digest"),
        ("calibration_digest", "calibration_digest"),
    ]
    if canonical_splatfacto:
        bindings = [
            ("operation_request_digest", "canonical_3dgs_execution_plan_digest"),
            ("operation_input_bundle_digest", "transport_bundle_digest"),
            ("source_commit_sha", "source_commit_sha"),
            ("reconstruction_dataset_digest", "colmap_training_dataset_digest"),
            ("frozen_split_digest", "frozen_split_digest"),
        ]
    for request_key, receipt_key in bindings:
        if request.get(request_key) != receipt.get(receipt_key):
            blockers.append(f"reconstruction_vast_operation_{request_key}_mismatch")
    if blockers:
        raise ReconstructionVastOperationError(blockers)
    return request, receipt


def run_reconstruction_vast_operation(
    *,
    bound_request: Mapping[str, Any],
    bundle_receipt: Mapping[str, Any],
    preflight: Mapping[str, Any],
    job_dir: str | Path,
    input_bundle_get_url: str,
    input_receipt_get_url: str,
    output_bundle_put_url: str,
    output_bundle_get_url: str,
    provider: GpuRenderProvider,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    output_fetcher: Callable[[str, Path], SafeHttpFileTransfer] = _default_output_fetcher,
    output_validator: Callable[..., tuple[dict[str, Any], dict[str, Any]]] = (
        validate_reconstruction_gpu_operation_output_bundle
    ),
    allocator_admission: Mapping[str, Any] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
    watchdog_validator: Callable[[Mapping[str, Any], float, int], bool] | None = None,
) -> dict[str, Any]:
    """Run one operation and validate retrieved outputs before guaranteed teardown."""

    require_paid_resource_admission_grant(
        paid_resource_admission_grant, resource_class="gpu_render"
    )
    request, receipt = _validate_bindings(
        bound_request=bound_request, bundle_receipt=bundle_receipt
    )
    canonical_splatfacto = (
        request.get("execution_adapter_id") == "canonical_splatfacto_vast_v1"
    )
    if canonical_splatfacto and not isinstance(allocator_admission, Mapping):
        raise ReconstructionVastOperationError(
            ["canonical_splatfacto_allocator_admission_missing"]
        )
    try:
        for url in (
            input_bundle_get_url,
            input_receipt_get_url,
            output_bundle_put_url,
            output_bundle_get_url,
        ):
            validate_outbound_url(url, policy=presigned_transfer_policy(url))
    except SafeOutboundHttpError as exc:
        raise ReconstructionVastOperationError(
            ["reconstruction_vast_operation_transport_url_invalid"]
        ) from exc
    if provider.name != "vast":
        raise ReconstructionVastOperationError(
            ["reconstruction_vast_operation_vast_first_required"]
        )
    operation = str(request["operation"])
    request_digest = str(request["request_digest"])
    operation_request_digest = str(request["operation_request_digest"])
    input_bundle_digest = str(request["operation_input_bundle_digest"])
    worker_image = str(request["worker_image_digest"])
    source_commit = str(request["source_commit_sha"])
    hard_ttl = int(request.get("hard_ttl_seconds") or 0)
    max_spend = float(request.get("max_spend_usd") or 0)
    retry_cap = int(request.get("retry_cap") or 0)
    if (
        _DIGEST.fullmatch(request_digest) is None
        or hard_ttl <= 0
        or max_spend <= 0
        or retry_cap < 0
    ):
        raise ReconstructionVastOperationError(
            ["reconstruction_vast_operation_execution_bounds_invalid"]
        )

    root = Path(job_dir)
    ensure_dir(root)
    pending_dir = root / "pending_teardowns"
    lease_dir = root / "leases"
    attempts_dir = root / "retrieval_attempts"
    for path in (pending_dir, lease_dir, attempts_dir):
        ensure_dir(path)
    receipt_path, receipt_file_digest = _canonical_receipt_file(root, receipt)
    if _sha256(receipt_path) != receipt_file_digest:
        raise ReconstructionVastOperationError(
            ["reconstruction_vast_operation_receipt_file_digest_mismatch"]
        )
    started_at = float(clock())
    watchdog = preflight.get("watchdog")
    watchdog = watchdog if isinstance(watchdog, Mapping) else {}
    validator = watchdog_validator or (
        lambda value, now, ttl: _watchdog_valid(
            value, now_epoch=now, hard_ttl_seconds=ttl
        )
    )
    if not validator(watchdog, started_at, hard_ttl):
        raise ReconstructionVastOperationError(
            ["reconstruction_vast_operation_watchdog_not_live"]
        )
    scoped_before = provider.billable_inventory(name_prefix=NAME_PREFIX)
    global_before = provider.billable_inventory(name_prefix="")
    if not all(
        row.get("api_confirmed") is True and row.get("live_resource_count") == 0
        for row in (scoped_before, global_before)
    ):
        raise ReconstructionVastOperationError(
            ["reconstruction_vast_operation_provider_not_zero_before_launch"]
        )
    reconciliation = build_paid_provider_lane_reconciliation(
        provider="vast",
        lane=PAID_LANE,
        provider_inventory=global_before,
        open_pending_teardowns=load_pending_teardowns(registry_dir=pending_dir),
    )
    lease = acquire_paid_provider_lane_lease(
        provider="vast",
        lane=PAID_LANE,
        job_dir=str(root),
        ttl_seconds=hard_ttl,
        lease_dir=lease_dir,
        reconciliation=reconciliation,
    )
    if lease.get("status") != "acquired":
        raise ReconstructionVastOperationError(
            ["reconstruction_vast_operation_paid_lane_not_acquired"]
        )
    name = f"{NAME_PREFIX}{operation.replace('_canary', '')}-{request_digest[7:19]}"
    pending = open_pending_teardown(
        provider="vast",
        lane=PAID_LANE,
        run_id=request_digest,
        resource_name=name,
        job_dir=root,
        max_age_seconds=hard_ttl + 1800,
        registry_dir=pending_dir,
    )
    pending_path = Path(str(pending["path"]))
    instance_id: str | None = None
    launch_result: dict[str, Any] = {}
    validated_output_receipt: dict[str, Any] | None = None
    runtime_result: dict[str, Any] | None = None
    retrieved_transfer: SafeHttpFileTransfer | None = None
    output_retrieved_before_teardown = False
    provider_mutations = 0
    blockers: list[str] = []
    invalid_digests: set[str] = set()
    fetch_attempts = 0
    try:
        worker_environment = {
                "BLUEPRINT_RECONSTRUCTION_OPERATION": operation,
                "BLUEPRINT_RECONSTRUCTION_INPUT_BUNDLE_GET_URL": input_bundle_get_url,
                "BLUEPRINT_RECONSTRUCTION_INPUT_RECEIPT_GET_URL": input_receipt_get_url,
                "BLUEPRINT_RECONSTRUCTION_OUTPUT_BUNDLE_PUT_URL": output_bundle_put_url,
                "BLUEPRINT_RECONSTRUCTION_INPUT_BUNDLE_DIGEST": input_bundle_digest,
                "BLUEPRINT_RECONSTRUCTION_INPUT_RECEIPT_FILE_DIGEST": receipt_file_digest,
                "BLUEPRINT_RECONSTRUCTION_OPERATION_REQUEST_DIGEST": (
                    operation_request_digest
                ),
                "BLUEPRINT_CONTAINER_IMAGE_DIGEST": worker_image,
                "BLUEPRINT_SOURCE_COMMIT": source_commit,
        }
        if canonical_splatfacto:
            assert allocator_admission is not None
            worker_environment.update(
                {
                    "BLUEPRINT_CANONICAL_ALLOCATOR_ADMISSION_B64": base64.b64encode(
                        (json.dumps(dict(allocator_admission), sort_keys=True, separators=(",", ":")) + "\n").encode()
                    ).decode(),
                    "BLUEPRINT_CANONICAL_AUTHORITY_ID": str(request["authority_id"]),
                    "BLUEPRINT_CANONICAL_MAX_SPEND_USD": str(max_spend),
                    "BLUEPRINT_CANONICAL_HARD_TTL_SECONDS": str(hard_ttl),
                }
            )
        spec = RenderLaunchSpec(
            name=name,
            image=worker_image,
            env=worker_environment,
            bootstrap_argv=["-lc", _bootstrap_script(canonical_splatfacto=canonical_splatfacto)],
            entrypoint=["bash"],
            container_disk_gb=max(
                100, int(preflight.get("container_disk_bytes") or 0) // 1024**3
            ),
            volume_gb=0,
            max_hourly_rate_usd=float(
                preflight.get("on_demand_price_usd_per_hour") or 0
            ),
            min_gpu_ram_mb=max(
                24_000, int(preflight.get("gpu_memory_bytes") or 0) // 1_000_000
            ),
            requires_rtx=False,
            vast_launch_mode="args",
        )
        provider_request = provider.build_request(spec, root)
        provider_request["prelaunch_spend_guard"] = {
            "schema_version": "reconstruction_gpu_prelaunch_spend_guard.v1",
            "required_before_provider_launch": True,
            "can_launch": True,
            "blockers": [],
            "max_spend_usd": max_spend,
            "hard_ttl_seconds": hard_ttl,
            "retry_cap": retry_cap,
            "request_digest": request_digest,
            "operation": operation,
        }
        launch_result = dict(
            provider.launch(
                root,
                provider_request,
                cold=True,
                allow_cold_fallback=False,
                paid_resource_admission_grant=paid_resource_admission_grant,
            )
        )
        if launch_result.get("allocation_outcome_ambiguous") is True:
            provider_mutations += 1
            mark_pending_teardown_ambiguous(
                pending_path,
                reason="reconstruction_vast_operation_create_ambiguous",
                evidence={"blockers": list(launch_result.get("blockers") or [])},
            )
            blockers.append("reconstruction_vast_operation_create_ambiguous")
        elif launch_result.get("status") != "launched" or not launch_result.get(
            "instance_id"
        ):
            cancel_pending_teardown(
                pending_path,
                reason="provider_confirmed_no_allocation",
                evidence={"status": launch_result.get("status")},
            )
            blockers.append("reconstruction_vast_operation_instance_not_created")
        else:
            instance_id = str(launch_result["instance_id"])
            provider_mutations += 1
            bind_pending_teardown_instance(pending_path, instance_id)
            while float(clock()) - started_at <= hard_ttl:
                fetch_attempts += 1
                attempt_path = attempts_dir / f"output_{fetch_attempts:04d}.zip"
                try:
                    retrieved_transfer = output_fetcher(
                        output_bundle_get_url, attempt_path
                    )
                except (FileNotFoundError, TimeoutError):
                    if float(clock()) - started_at >= hard_ttl:
                        break
                    sleeper(
                        min(
                            5.0,
                            max(0.0, hard_ttl - (float(clock()) - started_at)),
                        )
                    )
                    continue
                except Exception as exc:  # noqa: BLE001
                    blockers.append(
                        f"reconstruction_vast_operation_output_fetch_failed:{type(exc).__name__}"
                    )
                    break
                try:
                    validator = (
                        validate_canonical_3dgs_vast_output_bundle
                        if canonical_splatfacto
                        else output_validator
                    )
                    validated_output_receipt, runtime_result = validator(
                        bundle_path=attempt_path,
                        expected_operation=operation,
                        expected_operation_request_digest=operation_request_digest,
                        expected_worker_image_digest=worker_image,
                        expected_source_commit_sha=source_commit,
                    )
                    if (
                        validated_output_receipt.get(
                            "operation_output_bundle_digest"
                        )
                        != retrieved_transfer.sha256
                    ):
                        raise ReconstructionGpuOperationOutputError(
                            [
                                "reconstruction_operation_output_transport_digest_mismatch"
                            ]
                        )
                except ReconstructionGpuOperationOutputError as exc:
                    rejection = {
                        "attempt": fetch_attempts,
                        "observed_digest": retrieved_transfer.sha256,
                        "blockers": list(exc.codes),
                        "failed_evidence_preserved": True,
                    }
                    write_json(attempt_path.with_suffix(".rejection.json"), rejection)
                    if retrieved_transfer.sha256 in invalid_digests:
                        blockers.append(
                            "reconstruction_vast_operation_repeated_identical_blocker"
                        )
                        break
                    invalid_digests.add(retrieved_transfer.sha256)
                    if float(clock()) - started_at >= hard_ttl:
                        break
                    sleeper(
                        min(
                            5.0,
                            max(0.0, hard_ttl - (float(clock()) - started_at)),
                        )
                    )
                    continue
                output_retrieved_before_teardown = True
                write_json(root / "validated_output_bundle_receipt.json", validated_output_receipt)
                write_json(root / "provider_runtime_result.json", runtime_result)
                break
            if validated_output_receipt is None:
                blockers.append("reconstruction_vast_operation_output_not_accepted")
    except Exception as exc:  # noqa: BLE001 - normalized before teardown.
        blockers.append(
            f"reconstruction_vast_operation_controller_failure:{type(exc).__name__}"
        )
        if instance_id is None and launch_result.get("allocation_outcome_ambiguous") is not True:
            cancel_pending_teardown(
                pending_path,
                reason="controller_failed_before_confirmed_allocation",
                evidence={"error_type": type(exc).__name__},
            )
    finally:
        terminate_result: dict[str, Any] = {
            "status": "not_required" if instance_id is None else "not_attempted"
        }
        if instance_id is not None:
            terminate_result = dict(provider.terminate(instance_id))
            provider_mutations += 1
        scoped_after = provider.billable_inventory(name_prefix=NAME_PREFIX)
        global_after = provider.billable_inventory(name_prefix="")
        provider_zero = all(
            row.get("api_confirmed") is True and row.get("live_resource_count") == 0
            for row in (scoped_after, global_after)
        )
        teardown_passed = bool(
            provider_zero
            and (
                instance_id is None
                or terminate_result.get("status")
                in {"stopped", "terminated", "deleted"}
            )
        )
        teardown = {
            "schema_version": TEARDOWN_SCHEMA_VERSION,
            "status": "PASS" if teardown_passed else "FAIL",
            "provider": "vast",
            "operation": operation,
            "request_digest": request_digest,
            "bound_request_digest": request.get("bound_request_digest"),
            "worker_image_digest": worker_image,
            "instance_id": instance_id,
            "output_retrieved_before_teardown": output_retrieved_before_teardown,
            "terminate_result": terminate_result,
            "provider_zero_verified": provider_zero,
            "timestamp": utc_now_iso(),
        }
        teardown["teardown_receipt_digest"] = canonical_digest(
            teardown, digest_field="teardown_receipt_digest"
        )
        write_json(root / "teardown_receipt.json", teardown)
        if instance_id is not None and teardown_passed:
            close_pending_teardown(pending_path, teardown)
        elif (
            instance_id is None
            and launch_result.get("allocation_outcome_ambiguous") is True
            and teardown_passed
        ):
            cancel_pending_teardown(
                pending_path,
                reason="provider_zero_resolved_ambiguous_create",
                evidence={"teardown_receipt_digest": teardown["teardown_receipt_digest"]},
            )
        terminal_reconciliation = build_paid_provider_lane_reconciliation(
            provider="vast",
            lane=PAID_LANE,
            provider_inventory=global_after,
            open_pending_teardowns=load_pending_teardowns(registry_dir=pending_dir),
        )
        lease_release = release_paid_provider_lane_lease(
            lease,
            reason="reconstruction_operation_terminal",
            provider_mutation_started=instance_id is not None,
            terminal_reconciliation=terminal_reconciliation,
            lease_dir=lease_dir,
        )
        if not teardown_passed:
            blockers.append("reconstruction_vast_operation_teardown_verification_failed")
        if instance_id is not None and lease_release.get("released") is not True:
            blockers.append("reconstruction_vast_operation_paid_lane_release_blocked")
        provider_zero_receipt = {
            "schema_version": PROVIDER_ZERO_SCHEMA_VERSION,
            "status": "PASS" if provider_zero else "FAIL",
            "provider": "vast",
            "operation": operation,
            "request_digest": request_digest,
            "bound_request_digest": request.get("bound_request_digest"),
            "scoped_live_resource_count": scoped_after.get("live_resource_count"),
            "global_live_resource_count": global_after.get("live_resource_count"),
            "api_confirmed": bool(
                scoped_after.get("api_confirmed") is True
                and global_after.get("api_confirmed") is True
            ),
            "timestamp": utc_now_iso(),
        }
        provider_zero_receipt["provider_zero_digest"] = canonical_digest(
            provider_zero_receipt, digest_field="provider_zero_digest"
        )
        write_json(root / "provider_zero_verification.json", provider_zero_receipt)

    duration = max(0.0, float(clock()) - started_at)
    hourly = float(preflight.get("on_demand_price_usd_per_hour") or 0)
    cost = hourly * duration / 3600.0 if instance_id else 0.0
    if cost > max_spend:
        blockers.append("reconstruction_vast_operation_budget_exhausted")
    runtime_digest = None
    runtime_status = None
    if runtime_result is not None:
        runtime_digest = runtime_result.get("pose_estimation_result_digest") or runtime_result.get(
            "reconstruction_training_result_digest"
        )
        runtime_status = runtime_result.get("status")
    execution = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "completed"
            if validated_output_receipt is not None and not blockers
            else "failed"
        ),
        "operation": operation,
        "request_digest": request_digest,
        "bound_request_digest": request.get("bound_request_digest"),
        "operation_request_digest": operation_request_digest,
        "operation_input_bundle_digest": input_bundle_digest,
        "worker_image_digest": worker_image,
        "source_commit_sha": source_commit,
        "provider": "vast",
        "instance_id": instance_id,
        "fetch_attempts": fetch_attempts,
        "operation_output_bundle_digest": (
            retrieved_transfer.sha256 if retrieved_transfer else None
        ),
        "output_bundle_receipt_digest": (
            validated_output_receipt.get("output_bundle_receipt_digest")
            if validated_output_receipt
            else None
        ),
        "provider_runtime_result_digest": runtime_digest,
        "operation_result_status": runtime_status,
        "operation_scientific_success_inferred": False,
        "output_retrieved_before_teardown": output_retrieved_before_teardown,
        "duration_seconds": duration,
        "cost_usd": cost,
        "provider_mutations_performed": provider_mutations,
        "provider_mutation_outcome_ambiguous": bool(
            launch_result.get("allocation_outcome_ambiguous") is True
        ),
        "blockers": sorted(set(blockers)),
        "teardown_receipt_digest": teardown["teardown_receipt_digest"],
        "provider_zero_digest": provider_zero_receipt["provider_zero_digest"],
        "provider_zero_verified": provider_zero_receipt["status"] == "PASS",
        "scientific_qualification_inferred": False,
        "proof_effect": "none",
        "claim_ceiling": "provider_execution_and_candidate_transport_only",
    }
    execution["execution_result_digest"] = canonical_digest(
        execution, digest_field="execution_result_digest"
    )
    write_json(root / "reconstruction_vast_operation_execution.json", execution)
    return execution


def replay_reconstruction_vast_operation(
    *,
    job_dir: str | Path,
    bound_request: Mapping[str, Any],
    output_validator: Callable[..., tuple[dict[str, Any], dict[str, Any]]] = (
        validate_reconstruction_gpu_operation_output_bundle
    ),
) -> dict[str, Any]:
    """Replay an accepted operation, outputs, teardown, and provider-zero offline."""

    root = Path(job_dir)
    request = json.loads(json.dumps(dict(bound_request)))
    blockers: list[str] = []
    try:
        execution = _load_object(
            root / "reconstruction_vast_operation_execution.json",
            code="reconstruction_vast_operation_replay_execution_missing",
        )
        teardown = _load_object(
            root / "teardown_receipt.json",
            code="reconstruction_vast_operation_replay_teardown_missing",
        )
        provider_zero = _load_object(
            root / "provider_zero_verification.json",
            code="reconstruction_vast_operation_replay_provider_zero_missing",
        )
        recorded_receipt = _load_object(
            root / "validated_output_bundle_receipt.json",
            code="reconstruction_vast_operation_replay_output_receipt_missing",
        )
        recorded_runtime = _load_object(
            root / "provider_runtime_result.json",
            code="reconstruction_vast_operation_replay_runtime_missing",
        )
    except ReconstructionVastOperationError as exc:
        blockers.extend(exc.codes)
        execution = {}
        teardown = {}
        provider_zero = {}
        recorded_receipt = {}
        recorded_runtime = {}
    operation = str(request.get("operation") or "")
    attempts = sorted((root / "retrieval_attempts").glob("output_*.zip"))
    accepted_path = next(
        (
            path
            for path in reversed(attempts)
            if _sha256(path) == execution.get("operation_output_bundle_digest")
        ),
        None,
    )
    validated_receipt: dict[str, Any] = {}
    validated_runtime: dict[str, Any] = {}
    if accepted_path is None:
        blockers.append("reconstruction_vast_operation_replay_output_bundle_missing")
    else:
        try:
            validated_receipt, validated_runtime = output_validator(
                bundle_path=accepted_path,
                expected_operation=operation,
                expected_operation_request_digest=str(
                    request.get("operation_request_digest") or ""
                ),
                expected_worker_image_digest=str(request.get("worker_image_digest") or ""),
                expected_source_commit_sha=str(request.get("source_commit_sha") or ""),
            )
        except ReconstructionGpuOperationOutputError as exc:
            blockers.extend(
                f"reconstruction_vast_operation_replay_output_invalid:{code}"
                for code in exc.codes
            )
    for value, field, code in (
        (execution, "execution_result_digest", "execution_digest_mismatch"),
        (teardown, "teardown_receipt_digest", "teardown_digest_mismatch"),
        (provider_zero, "provider_zero_digest", "provider_zero_digest_mismatch"),
        (recorded_receipt, "output_bundle_receipt_digest", "output_receipt_digest_mismatch"),
    ):
        if value and value.get(field) != canonical_digest(value, digest_field=field):
            blockers.append(f"reconstruction_vast_operation_replay_{code}")
    if (
        request.get("bound_request_digest")
        != canonical_digest(request, digest_field="bound_request_digest")
        or execution.get("status") != "completed"
        or execution.get("bound_request_digest") != request.get("bound_request_digest")
        or execution.get("output_retrieved_before_teardown") is not True
        or execution.get("scientific_qualification_inferred") is not False
        or execution.get("output_bundle_receipt_digest")
        != recorded_receipt.get("output_bundle_receipt_digest")
        or recorded_receipt != validated_receipt
        or recorded_runtime != validated_runtime
    ):
        blockers.append("reconstruction_vast_operation_replay_binding_mismatch")
    if (
        teardown.get("status") != "PASS"
        or teardown.get("output_retrieved_before_teardown") is not True
        or teardown.get("provider_zero_verified") is not True
        or provider_zero.get("status") != "PASS"
        or provider_zero.get("api_confirmed") is not True
        or provider_zero.get("scoped_live_resource_count") != 0
        or provider_zero.get("global_live_resource_count") != 0
    ):
        blockers.append("reconstruction_vast_operation_replay_teardown_not_verified")
    replay = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "status": "replay_verified" if not blockers else "replay_rejected",
        "operation": operation,
        "request_digest": request.get("request_digest"),
        "bound_request_digest": request.get("bound_request_digest"),
        "execution_result_digest": execution.get("execution_result_digest"),
        "operation_result_status": execution.get("operation_result_status"),
        "blockers": sorted(set(blockers)),
        "live_provider_accessed": False,
        "scientific_qualification_inferred": False,
        "proof_effect": "none",
        "claim_ceiling": "recorded_provider_execution_and_candidate_transport_only",
    }
    replay["replay_digest"] = canonical_digest(replay, digest_field="replay_digest")
    return replay


__all__ = [
    "ReconstructionVastOperationError",
    "replay_reconstruction_vast_operation",
    "run_reconstruction_vast_operation",
]
