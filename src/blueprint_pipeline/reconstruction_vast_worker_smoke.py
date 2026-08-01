"""Grant-gated one-instance Vast lifecycle for the reconstruction worker smoke."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .common import ensure_dir, utc_now_iso, write_json
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
from .reconstruction_worker_image_healthcheck import SCHEMA_VERSION as HEALTHCHECK_SCHEMA_VERSION
from .safe_outbound_http import presigned_transfer_policy, request as safe_http_request


RESULT_SCHEMA_VERSION = "reconstruction_vast_worker_smoke_result.v1"
TEARDOWN_SCHEMA_VERSION = "reconstruction_vast_teardown_receipt.v1"
PROVIDER_ZERO_SCHEMA_VERSION = "reconstruction_vast_provider_zero_verification.v1"
PAID_LANE = "reconstruction_gpu_canary"
NAME_PREFIX = "blueprint-reconstruction-"
OUTPUT_PUT_ENV = "BLUEPRINT_RECONSTRUCTION_SMOKE_OUTPUT_PUT_URL"
REQUEST_DIGEST_ENV = "BLUEPRINT_RECONSTRUCTION_SMOKE_REQUEST_DIGEST"
IMAGE_DIGEST_ENV = "BLUEPRINT_CONTAINER_IMAGE_DIGEST"
MAX_RESULT_BYTES = 2 * 1024 * 1024


class ReconstructionVastSmokeError(ValueError):
    pass


def _default_result_fetcher(url: str) -> Mapping[str, Any]:
    response = safe_http_request(
        url,
        method="GET",
        timeout_seconds=30,
        policy=presigned_transfer_policy(url, max_response_bytes=MAX_RESULT_BYTES),
        max_response_bytes=MAX_RESULT_BYTES,
    )
    if response.status != 200:
        raise FileNotFoundError(f"reconstruction_smoke_output_http:{response.status}")
    value = json.loads(response.body)
    if not isinstance(value, Mapping):
        raise ReconstructionVastSmokeError("reconstruction_smoke_output_not_object")
    return dict(value)


def _load_recorded_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionVastSmokeError(
            f"reconstruction_replay_artifact_unreadable:{path.name}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ReconstructionVastSmokeError(f"reconstruction_replay_artifact_not_object:{path.name}")
    return dict(value)


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
        or str(watchdog.get("name_prefix") or watchdog.get("pod_name_prefix") or "") != NAME_PREFIX
        or pid <= 0
        or deadline < now_epoch + hard_ttl_seconds
    ):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _bootstrap_script() -> str:
    """Run only the worker healthcheck and upload its digest-bound envelope."""

    return r"""set -euo pipefail
result_path=/tmp/blueprint_reconstruction_worker_healthcheck.json
envelope_path=/tmp/blueprint_reconstruction_worker_smoke_result.json
python -m blueprint_pipeline.reconstruction_worker_image_healthcheck --output "$result_path"
python - "$result_path" "$envelope_path" <<'PY'
import json, os, sys
from pathlib import Path
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
health = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = {
    "schema_version": "reconstruction_vast_worker_smoke_result.v1",
    "status": "passed" if health.get("status") == "passed" else "failed",
    "request_digest": os.environ["BLUEPRINT_RECONSTRUCTION_SMOKE_REQUEST_DIGEST"],
    "worker_image_digest": os.environ["BLUEPRINT_CONTAINER_IMAGE_DIGEST"],
    "healthcheck": health,
    "hidden_heldout_observations_accessed": False,
    "scientific_qualification_inferred": False,
    "proof_effect": "none",
    "claim_ceiling": "worker_image_compatibility_only",
}
value["runtime_result_digest"] = canonical_digest(value, digest_field="runtime_result_digest")
Path(sys.argv[2]).write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
PY
python - "$envelope_path" <<'PY'
import os, sys
from pathlib import Path
from blueprint_pipeline.safe_outbound_http import presigned_transfer_policy, request
url = os.environ["BLUEPRINT_RECONSTRUCTION_SMOKE_OUTPUT_PUT_URL"]
payload = Path(sys.argv[1]).read_bytes()
response = request(
    url,
    method="PUT",
    data=payload,
    headers={"Content-Type": "application/json"},
    timeout_seconds=90,
    policy=presigned_transfer_policy(url, max_response_bytes=1024),
    max_response_bytes=1024,
)
if response.status not in {200, 201, 204}:
    raise SystemExit(f"reconstruction_smoke_upload_failed:{response.status}")
PY
"""


def validate_worker_smoke_result(
    value: Mapping[str, Any],
    *,
    request_digest: str,
    worker_image_digest: str,
    source_commit_sha: str,
) -> dict[str, Any]:
    result = json.loads(json.dumps(dict(value)))
    supplied = result.pop("runtime_result_digest", None)
    expected = canonical_digest(result, digest_field="runtime_result_digest")
    result["runtime_result_digest"] = supplied
    blockers: list[str] = []
    if supplied != expected:
        blockers.append("reconstruction_smoke_result_digest_mismatch")
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        blockers.append("reconstruction_smoke_result_schema_invalid")
    if result.get("request_digest") != request_digest:
        blockers.append("reconstruction_smoke_request_binding_mismatch")
    if result.get("worker_image_digest") != worker_image_digest:
        blockers.append("reconstruction_smoke_image_binding_mismatch")
    if result.get("hidden_heldout_observations_accessed") is not False:
        blockers.append("reconstruction_smoke_hidden_heldout_access_forbidden")
    if result.get("scientific_qualification_inferred") is not False:
        blockers.append("reconstruction_smoke_scientific_claim_forbidden")
    if result.get("proof_effect") != "none":
        blockers.append("reconstruction_smoke_proof_effect_invalid")
    health = result.get("healthcheck")
    health = dict(health) if isinstance(health, Mapping) else {}
    health_supplied = health.pop("healthcheck_digest", None)
    health_expected = canonical_digest(health, digest_field="healthcheck_digest")
    health["healthcheck_digest"] = health_supplied
    if health_supplied != health_expected:
        blockers.append("reconstruction_smoke_healthcheck_digest_mismatch")
    if health.get("schema_version") != HEALTHCHECK_SCHEMA_VERSION:
        blockers.append("reconstruction_smoke_healthcheck_schema_invalid")
    if health.get("status") != "passed" or health.get("mode") != "gpu_runtime":
        blockers.append("reconstruction_smoke_healthcheck_not_passed")
    if health.get("display_attached") is not False:
        blockers.append("reconstruction_smoke_display_attached")
    runtime_identity = health.get("runtime_identity")
    runtime_identity = runtime_identity if isinstance(runtime_identity, Mapping) else {}
    if runtime_identity.get("container_image_digest") != worker_image_digest:
        blockers.append("reconstruction_smoke_runtime_image_mismatch")
    if runtime_identity.get("source_commit_sha") != source_commit_sha:
        blockers.append("reconstruction_smoke_runtime_source_commit_mismatch")
    checks = health.get("checks")
    checks = checks if isinstance(checks, list) else []
    if not any(
        isinstance(row, Mapping)
        and row.get("check_id") == "nvidia_runtime"
        and row.get("status") == "passed"
        for row in checks
    ):
        blockers.append("reconstruction_smoke_nvidia_runtime_not_passed")
    if blockers:
        raise ReconstructionVastSmokeError(";".join(sorted(set(blockers))))
    return result


def run_reconstruction_vast_worker_smoke(
    *,
    bound_request: Mapping[str, Any],
    preflight: Mapping[str, Any],
    job_dir: str | Path,
    output_put_url: str,
    output_get_url: str,
    provider: GpuRenderProvider,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    result_fetcher: Callable[[str], Mapping[str, Any]] = _default_result_fetcher,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
    watchdog_validator: Callable[[Mapping[str, Any], float, int], bool] | None = None,
) -> dict[str, Any]:
    """Execute one smoke and always attempt exact teardown after allocation."""

    require_paid_resource_admission_grant(
        paid_resource_admission_grant, resource_class="gpu_render"
    )
    request = json.loads(json.dumps(dict(bound_request)))
    expected_bound_digest = canonical_digest(request, digest_field="bound_request_digest")
    if request.get("bound_request_digest") != expected_bound_digest:
        raise ReconstructionVastSmokeError("reconstruction_bound_request_digest_mismatch")
    if (
        request.get("bound_provider") != "vast"
        or request.get("provider_mutation_authorized") is not True
        or request.get("operation") != "worker_smoke"
        or request.get("expected_runtime_result_schema") != RESULT_SCHEMA_VERSION
    ):
        raise ReconstructionVastSmokeError("reconstruction_bound_request_not_executable")
    for key in ("operation_request_digest", "operation_input_bundle_digest"):
        value = request.get(key)
        if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
            raise ReconstructionVastSmokeError(
                f"reconstruction_bound_request_{key}_invalid"
            )
    if provider.name != "vast":
        raise ReconstructionVastSmokeError("reconstruction_vast_first_required")
    worker_image = str(request.get("worker_image_digest") or "")
    request_digest = str(request.get("request_digest") or "")
    hard_ttl = int(request.get("hard_ttl_seconds") or 0)
    max_spend = float(request.get("max_spend_usd") or 0)
    retry_cap = int(request.get("retry_cap") or 0)
    if retry_cap < 0 or hard_ttl <= 0 or max_spend <= 0:
        raise ReconstructionVastSmokeError("reconstruction_execution_bounds_invalid")
    root = Path(job_dir)
    pending_dir = root / "pending_teardowns"
    lease_dir = root / "leases"
    ensure_dir(root)
    ensure_dir(pending_dir)
    ensure_dir(lease_dir)
    name = f"{NAME_PREFIX}{request_digest.removeprefix('sha256:')[:12]}"
    started_at = float(clock())
    watchdog = preflight.get("watchdog")
    watchdog = watchdog if isinstance(watchdog, Mapping) else {}
    validator = watchdog_validator or (
        lambda value, now, ttl: _watchdog_valid(value, now_epoch=now, hard_ttl_seconds=ttl)
    )
    if not validator(watchdog, started_at, hard_ttl):
        raise ReconstructionVastSmokeError("reconstruction_independent_watchdog_not_live")
    scoped_before = provider.billable_inventory(name_prefix=NAME_PREFIX)
    global_before = provider.billable_inventory(name_prefix="")
    if not all(
        row.get("api_confirmed") is True and row.get("live_resource_count") == 0
        for row in (scoped_before, global_before)
    ):
        raise ReconstructionVastSmokeError("reconstruction_provider_not_zero_before_launch")
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
        raise ReconstructionVastSmokeError("reconstruction_paid_lane_not_acquired")
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
    raw_result: dict[str, Any] | None = None
    validated_result: dict[str, Any] | None = None
    execution_blockers: list[str] = []
    provider_mutations = 0
    try:
        spec = RenderLaunchSpec(
            name=name,
            image=worker_image,
            env={
                OUTPUT_PUT_ENV: output_put_url,
                REQUEST_DIGEST_ENV: request_digest,
                IMAGE_DIGEST_ENV: worker_image,
                "BLUEPRINT_SOURCE_COMMIT": str(request.get("source_commit_sha") or ""),
            },
            bootstrap_argv=["-lc", _bootstrap_script()],
            entrypoint=["bash"],
            container_disk_gb=max(100, int(preflight.get("container_disk_bytes") or 0) // 1024**3),
            volume_gb=0,
            max_hourly_rate_usd=float(preflight.get("on_demand_price_usd_per_hour") or 0),
            min_gpu_ram_mb=max(24_000, int(preflight.get("gpu_memory_bytes") or 0) // 1_000_000),
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
                reason="reconstruction_vast_create_outcome_ambiguous",
                evidence={"blockers": list(launch_result.get("blockers") or [])},
            )
            execution_blockers.append("reconstruction_vast_create_outcome_ambiguous")
        elif launch_result.get("status") != "launched" or not launch_result.get("instance_id"):
            cancel_pending_teardown(
                pending_path,
                reason="provider_confirmed_no_allocation",
                evidence={"status": launch_result.get("status")},
            )
            execution_blockers.append("reconstruction_vast_instance_not_created")
        else:
            instance_id = str(launch_result["instance_id"])
            provider_mutations += 1
            bind_pending_teardown_instance(pending_path, instance_id)
            fetch_attempts = 0
            while float(clock()) - started_at <= hard_ttl:
                fetch_attempts += 1
                try:
                    raw_result = dict(result_fetcher(output_get_url))
                    break
                except (FileNotFoundError, TimeoutError):
                    if float(clock()) - started_at >= hard_ttl:
                        break
                    sleeper(min(5.0, max(0.0, hard_ttl - (float(clock()) - started_at))))
            if raw_result is None:
                execution_blockers.append("reconstruction_smoke_output_timeout")
            else:
                write_json(root / "provider_runtime_result.json", raw_result)
                try:
                    validated_result = validate_worker_smoke_result(
                        raw_result,
                        request_digest=request_digest,
                        worker_image_digest=worker_image,
                        source_commit_sha=str(request.get("source_commit_sha") or ""),
                    )
                except ReconstructionVastSmokeError as exc:
                    execution_blockers.extend(str(exc).split(";"))
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
                or (terminate_result.get("status") in {"stopped", "terminated", "deleted"})
            )
        )
        teardown_receipt = {
            "schema_version": TEARDOWN_SCHEMA_VERSION,
            "status": "PASS" if teardown_passed else "FAIL",
            "provider": "vast",
            "request_digest": request_digest,
            "bound_request_digest": request.get("bound_request_digest"),
            "worker_image_digest": worker_image,
            "instance_id": instance_id,
            "terminate_result": terminate_result,
            "provider_zero_verified": provider_zero,
            "timestamp": utc_now_iso(),
        }
        teardown_receipt["teardown_receipt_digest"] = canonical_digest(
            teardown_receipt, digest_field="teardown_receipt_digest"
        )
        write_json(root / "teardown_receipt.json", teardown_receipt)
        if instance_id is not None and teardown_passed:
            close_pending_teardown(pending_path, teardown_receipt)
        elif (
            instance_id is None
            and launch_result.get("allocation_outcome_ambiguous") is True
            and teardown_passed
        ):
            cancel_pending_teardown(
                pending_path,
                reason="provider_zero_resolved_ambiguous_create",
                evidence={"provider_zero_digest": teardown_receipt["teardown_receipt_digest"]},
            )
        terminal_reconciliation = build_paid_provider_lane_reconciliation(
            provider="vast",
            lane=PAID_LANE,
            provider_inventory=global_after,
            open_pending_teardowns=load_pending_teardowns(registry_dir=pending_dir),
        )
        lease_release = release_paid_provider_lane_lease(
            lease,
            reason="reconstruction_worker_smoke_terminal",
            provider_mutation_started=instance_id is not None,
            terminal_reconciliation=terminal_reconciliation,
            lease_dir=lease_dir,
        )
        if not teardown_passed:
            execution_blockers.append("reconstruction_teardown_verification_failed")
        if instance_id is not None and lease_release.get("released") is not True:
            execution_blockers.append("reconstruction_paid_lane_release_blocked")
        provider_zero_receipt = {
            "schema_version": PROVIDER_ZERO_SCHEMA_VERSION,
            "status": "PASS" if provider_zero else "FAIL",
            "provider": "vast",
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
        execution_blockers.append("reconstruction_budget_exhausted")
    result = {
        "schema_version": "reconstruction_vast_worker_smoke_execution.v1",
        "status": "completed"
        if validated_result is not None and not execution_blockers
        else "failed",
        "request_digest": request_digest,
        "bound_request_digest": request.get("bound_request_digest"),
        "worker_image_digest": worker_image,
        "source_commit_sha": request.get("source_commit_sha"),
        "provider": "vast",
        "instance_id": instance_id,
        "provider_runtime_result_digest": (
            validated_result.get("runtime_result_digest") if validated_result else None
        ),
        "duration_seconds": duration,
        "cost_usd": cost,
        "provider_mutations_performed": provider_mutations,
        "provider_mutation_outcome_ambiguous": bool(
            launch_result.get("allocation_outcome_ambiguous") is True
        ),
        "blockers": sorted(set(execution_blockers)),
        "teardown_receipt_digest": teardown_receipt["teardown_receipt_digest"],
        "provider_zero_digest": provider_zero_receipt["provider_zero_digest"],
        "provider_zero_verified": provider_zero_receipt["status"] == "PASS",
        "scientific_qualification_inferred": False,
        "proof_effect": "none",
        "claim_ceiling": "worker_image_compatibility_only",
    }
    result["execution_result_digest"] = canonical_digest(
        result, digest_field="execution_result_digest"
    )
    write_json(root / "reconstruction_vast_worker_smoke_execution.json", result)
    return result


def replay_reconstruction_vast_worker_smoke(
    *, job_dir: str | Path, bound_request: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify an accepted worker smoke entirely from recorded artifacts."""

    root = Path(job_dir)
    request = json.loads(json.dumps(dict(bound_request)))
    blockers: list[str] = []
    if request.get("bound_request_digest") != canonical_digest(
        request, digest_field="bound_request_digest"
    ):
        blockers.append("reconstruction_replay_bound_request_digest_mismatch")
    try:
        execution = _load_recorded_object(root / "reconstruction_vast_worker_smoke_execution.json")
        teardown = _load_recorded_object(root / "teardown_receipt.json")
        provider_zero = _load_recorded_object(root / "provider_zero_verification.json")
        runtime = _load_recorded_object(root / "provider_runtime_result.json")
    except ReconstructionVastSmokeError as exc:
        blockers.extend(str(exc).split(";"))
        execution = {}
        teardown = {}
        provider_zero = {}
        runtime = {}

    for value, field, blocker in (
        (execution, "execution_result_digest", "reconstruction_replay_execution_digest_mismatch"),
        (teardown, "teardown_receipt_digest", "reconstruction_replay_teardown_digest_mismatch"),
        (
            provider_zero,
            "provider_zero_digest",
            "reconstruction_replay_provider_zero_digest_mismatch",
        ),
    ):
        if value and value.get(field) != canonical_digest(value, digest_field=field):
            blockers.append(blocker)

    request_digest = str(request.get("request_digest") or "")
    bound_digest = str(request.get("bound_request_digest") or "")
    image_digest = str(request.get("worker_image_digest") or "")
    source_commit = str(request.get("source_commit_sha") or "")
    if execution:
        if execution.get("status") != "completed" or execution.get("blockers") != []:
            blockers.append("reconstruction_replay_execution_not_accepted")
        if execution.get("request_digest") != request_digest:
            blockers.append("reconstruction_replay_request_binding_mismatch")
        if execution.get("bound_request_digest") != bound_digest:
            blockers.append("reconstruction_replay_bound_request_binding_mismatch")
        if execution.get("worker_image_digest") != image_digest:
            blockers.append("reconstruction_replay_image_binding_mismatch")
        if execution.get("source_commit_sha") != source_commit:
            blockers.append("reconstruction_replay_source_commit_binding_mismatch")
        if execution.get("teardown_receipt_digest") != teardown.get("teardown_receipt_digest"):
            blockers.append("reconstruction_replay_teardown_reference_mismatch")
        if execution.get("provider_zero_digest") != provider_zero.get("provider_zero_digest"):
            blockers.append("reconstruction_replay_provider_zero_reference_mismatch")
        if execution.get("scientific_qualification_inferred") is not False:
            blockers.append("reconstruction_replay_scientific_claim_forbidden")
    if teardown:
        if (
            teardown.get("status") != "PASS"
            or teardown.get("provider_zero_verified") is not True
            or teardown.get("request_digest") != request_digest
            or teardown.get("bound_request_digest") != bound_digest
            or teardown.get("worker_image_digest") != image_digest
        ):
            blockers.append("reconstruction_replay_teardown_not_verified")
    if provider_zero:
        if (
            provider_zero.get("status") != "PASS"
            or provider_zero.get("api_confirmed") is not True
            or provider_zero.get("scoped_live_resource_count") != 0
            or provider_zero.get("global_live_resource_count") != 0
            or provider_zero.get("request_digest") != request_digest
            or provider_zero.get("bound_request_digest") != bound_digest
        ):
            blockers.append("reconstruction_replay_provider_zero_not_verified")
    if runtime:
        try:
            validated_runtime = validate_worker_smoke_result(
                runtime,
                request_digest=request_digest,
                worker_image_digest=image_digest,
                source_commit_sha=source_commit,
            )
            if execution.get("provider_runtime_result_digest") != validated_runtime.get(
                "runtime_result_digest"
            ):
                blockers.append("reconstruction_replay_runtime_reference_mismatch")
        except ReconstructionVastSmokeError as exc:
            blockers.extend(str(exc).split(";"))

    result = {
        "schema_version": "reconstruction_vast_worker_smoke_replay.v1",
        "status": "replay_verified" if not blockers else "replay_rejected",
        "request_digest": request_digest,
        "bound_request_digest": bound_digest,
        "execution_result_digest": execution.get("execution_result_digest"),
        "blockers": sorted(set(blockers)),
        "live_provider_accessed": False,
        "scientific_qualification_inferred": False,
        "proof_effect": "none",
        "claim_ceiling": "recorded_worker_image_compatibility_only",
    }
    result["replay_digest"] = canonical_digest(result, digest_field="replay_digest")
    return result


__all__ = [
    "NAME_PREFIX",
    "PAID_LANE",
    "RESULT_SCHEMA_VERSION",
    "ReconstructionVastSmokeError",
    "replay_reconstruction_vast_worker_smoke",
    "run_reconstruction_vast_worker_smoke",
    "validate_worker_smoke_result",
]
