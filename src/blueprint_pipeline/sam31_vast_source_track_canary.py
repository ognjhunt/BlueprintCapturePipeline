"""Grant-gated one-instance Vast lifecycle for a SAM 3.1 source-track canary."""

from __future__ import annotations

import json
import os
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
from .safe_outbound_http import presigned_transfer_policy, request as safe_http_request
from .sam31_gpu_admission import (
    CHECKPOINT_DIGEST,
    CHECKPOINT_REPOSITORY_REVISION,
    OFFICIAL_CODE_REVISION,
    OPERATION,
)
from .sam31_source_track_canary_worker import RUNTIME_RESULT_SCHEMA_VERSION
from .vast_independent_watchdog_control import write_started_vast_instance_id


EXECUTION_SCHEMA_VERSION = "semantic_sam31_vast_source_track_execution.v1"
TEARDOWN_SCHEMA_VERSION = "semantic_sam31_vast_teardown_receipt.v1"
PROVIDER_ZERO_SCHEMA_VERSION = "semantic_sam31_vast_provider_zero.v1"
PAID_LANE = "semantic_sam31_gpu_canary"
NAME_PREFIX = "blueprint-sam31-source-tracks-"
MAX_RESULT_BYTES = 64 * 1024**2

INPUT_GET_ENV = "BLUEPRINT_SAM31_INPUT_BUNDLE_GET_URL"
OUTPUT_PUT_ENV = "BLUEPRINT_SAM31_OUTPUT_PUT_URL"
HF_TOKEN_ENV = "HF_TOKEN"


class Sam31VastCanaryError(ValueError):
    pass


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return {
            *(str(key).lower() for key in value),
            *(key for child in value.values() for key in _nested_keys(child)),
        }
    if isinstance(value, list):
        return {key for child in value for key in _nested_keys(child)}
    return set()


def _default_result_fetcher(url: str) -> Mapping[str, Any]:
    response = safe_http_request(
        url,
        method="GET",
        timeout_seconds=30,
        policy=presigned_transfer_policy(url, max_response_bytes=MAX_RESULT_BYTES),
        max_response_bytes=MAX_RESULT_BYTES,
    )
    if response.status != 200:
        raise FileNotFoundError(f"sam31_canary_output_http:{response.status}")
    value = json.loads(response.body)
    if not isinstance(value, Mapping):
        raise Sam31VastCanaryError("sam31_canary_output_not_object")
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
        or not str(
            watchdog.get("name_prefix") or watchdog.get("pod_name_prefix") or ""
        ).startswith(NAME_PREFIX)
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
    """Download exact inputs, fetch and verify the checkpoint, run, then upload."""

    return r"""set -euo pipefail
bundle_path=/work/sam31_input_bundle.zip
result_path=/work/sam31_source_track_result.json
checkpoint_path=/models/sam31/sam3.1_multiplex.pt
python - "$bundle_path" <<'PY'
import hashlib, os, sys
from pathlib import Path
from blueprint_pipeline.safe_outbound_http import presigned_transfer_policy, request
url = os.environ["BLUEPRINT_SAM31_INPUT_BUNDLE_GET_URL"]
response = request(
    url,
    method="GET",
    timeout_seconds=180,
    policy=presigned_transfer_policy(url, max_response_bytes=536870912),
    max_response_bytes=536870912,
)
if response.status != 200:
    raise SystemExit(f"sam31_input_download_failed:{response.status}")
expected = os.environ["BLUEPRINT_SAM31_INPUT_BUNDLE_DIGEST"].removeprefix("sha256:")
if hashlib.sha256(response.body).hexdigest() != expected:
    raise SystemExit("sam31_input_bundle_digest_mismatch")
Path(sys.argv[1]).write_bytes(response.body)
PY
python - "$checkpoint_path" <<'PY'
import hashlib, os, shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("sam31_hf_token_missing")
downloaded = Path(hf_hub_download(
    repo_id="facebook/sam3.1",
    filename="sam3.1_multiplex.pt",
    revision=os.environ["BLUEPRINT_SAM31_CHECKPOINT_REPOSITORY_REVISION"],
    token=token,
))
expected = os.environ["BLUEPRINT_SAM31_EXPECTED_CHECKPOINT_DIGEST"].removeprefix("sha256:")
if hashlib.sha256(downloaded.read_bytes()).hexdigest() != expected:
    raise SystemExit("sam31_checkpoint_digest_mismatch")
target = Path(sys.argv[1])
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(downloaded, target)
if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
    raise SystemExit("sam31_checkpoint_copy_digest_mismatch")
PY
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export BLUEPRINT_SAM31_CHECKPOINT_PATH="$checkpoint_path"
set +e
python -m blueprint_pipeline.sam31_source_track_canary_worker run \
    --input-bundle "$bundle_path" \
    --output "$result_path"
worker_status=$?
set -e
if [ ! -s "$result_path" ]; then
    exit "$worker_status"
fi
python - "$result_path" <<'PY'
import os, sys
from pathlib import Path
from blueprint_pipeline.safe_outbound_http import presigned_transfer_policy, request
url = os.environ["BLUEPRINT_SAM31_OUTPUT_PUT_URL"]
payload = Path(sys.argv[1]).read_bytes()
response = request(
    url,
    method="PUT",
    data=payload,
    headers={"Content-Type": "application/json"},
    timeout_seconds=180,
    policy=presigned_transfer_policy(url, max_response_bytes=1024),
    max_response_bytes=1024,
)
if response.status not in {200, 201, 204}:
    raise SystemExit(f"sam31_output_upload_failed:{response.status}")
PY
exit "$worker_status"
"""


def validate_sam31_runtime_result(
    value: Mapping[str, Any], *, bound_request: Mapping[str, Any]
) -> dict[str, Any]:
    result = json.loads(json.dumps(dict(value)))
    supplied = result.pop("runtime_result_digest", None)
    expected = canonical_digest(result, digest_field="runtime_result_digest")
    result["runtime_result_digest"] = supplied
    blockers: list[str] = []
    if supplied != expected:
        blockers.append("sam31_runtime_result_digest_mismatch")
    expected_fields = {
        "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
        "status": "passed",
        "request_digest": bound_request.get("request_digest"),
        "bound_request_digest": bound_request.get("bound_request_digest"),
        "worker_image_digest": bound_request.get("worker_image_digest"),
        "input_bundle_digest": bound_request.get("input_bundle_digest"),
        "source_track_run_request_digest": bound_request.get("source_track_run_request_digest"),
        "checkpoint_digest": CHECKPOINT_DIGEST,
        "source_frame_bytes_returned": False,
        "raw_secret_values_recorded": False,
        "network_access_during_inference": False,
        "directly_observed_object_fact": False,
        "metric_box_ready": False,
        "collision_ready": False,
        "physics_ready": False,
        "physical_task_success_established": False,
        "model_self_grading_permitted": False,
        "scientific_qualification_inferred": False,
        "proof_effect": "none",
        "claim_ceiling": "source_bound_2d_binary_mask_tracks_only",
        "comparative_policy_ranking_verdict": "thesis_not_supported",
    }
    for field, expected_value in expected_fields.items():
        if result.get(field) != expected_value:
            blockers.append(f"sam31_runtime_{field}_mismatch")
    runtime = result.get("runtime")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    if runtime.get("cuda_available") is not True or int(runtime.get("cuda_device_count") or 0) < 1:
        blockers.append("sam31_runtime_cuda_not_verified")
    stage = result.get("stage_run_result")
    stage = stage if isinstance(stage, Mapping) else {}
    if stage.get("status") not in {"completed", "abstained"}:
        blockers.append("sam31_runtime_stage_not_terminal")
    if stage.get("comparative_policy_ranking_verdict") != "thesis_not_supported":
        blockers.append("sam31_runtime_policy_verdict_invalid")
    forbidden_keys = {"hf_token", "hugging_face_hub_token", "signed_url", "api_key"}
    if _nested_keys(result) & forbidden_keys:
        blockers.append("sam31_runtime_secret_field_forbidden")
    if blockers:
        raise Sam31VastCanaryError(";".join(sorted(set(blockers))))
    return result


def run_sam31_vast_source_track_canary(
    *,
    bound_request: Mapping[str, Any],
    preflight: Mapping[str, Any],
    job_dir: str | Path,
    input_bundle_get_url: str,
    output_put_url: str,
    output_get_url: str,
    hf_token: str,
    provider: GpuRenderProvider,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    result_fetcher: Callable[[str], Mapping[str, Any]] = _default_result_fetcher,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
    watchdog_validator: Callable[[Mapping[str, Any], float, int], bool] | None = None,
) -> dict[str, Any]:
    """Execute one bounded canary and always attempt exact teardown."""

    require_paid_resource_admission_grant(
        paid_resource_admission_grant, resource_class="gpu_render"
    )
    request = json.loads(json.dumps(dict(bound_request)))
    if request.get("bound_request_digest") != canonical_digest(
        request, digest_field="bound_request_digest"
    ):
        raise Sam31VastCanaryError("sam31_bound_request_digest_mismatch")
    if (
        request.get("bound_provider") != "vast"
        or request.get("provider_mutation_authorized") is not True
        or request.get("operation") != OPERATION
        or provider.name != "vast"
    ):
        raise Sam31VastCanaryError("sam31_bound_request_not_executable")
    if not str(hf_token).strip():
        raise Sam31VastCanaryError("sam31_hf_token_missing")
    hard_ttl = int(request.get("hard_ttl_seconds") or 0)
    max_spend = float(request.get("max_spend_usd") or 0)
    retry_cap = int(request.get("retry_cap") or 0)
    if retry_cap < 0 or hard_ttl <= 0 or max_spend <= 0:
        raise Sam31VastCanaryError("sam31_execution_bounds_invalid")

    root = Path(job_dir)
    pending_dir = root / "pending_teardowns"
    lease_dir = root / "leases"
    ensure_dir(root)
    ensure_dir(pending_dir)
    ensure_dir(lease_dir)
    request_digest = str(request.get("request_digest") or "")
    image = str(request.get("worker_image_digest") or "")
    started_at = float(clock())
    watchdog = preflight.get("watchdog")
    watchdog = watchdog if isinstance(watchdog, Mapping) else {}
    validator = watchdog_validator or (
        lambda value, now, ttl: _watchdog_valid(value, now_epoch=now, hard_ttl_seconds=ttl)
    )
    if not validator(watchdog, started_at, hard_ttl):
        raise Sam31VastCanaryError("sam31_independent_watchdog_not_live")
    watchdog_prefix = str(
        watchdog.get("name_prefix") or watchdog.get("pod_name_prefix") or ""
    )
    name = f"{watchdog_prefix}{request_digest.removeprefix('sha256:')[:12]}"
    scoped_before = provider.billable_inventory(name_prefix=NAME_PREFIX)
    global_before = provider.billable_inventory(name_prefix="")
    if not all(
        row.get("api_confirmed") is True and row.get("live_resource_count") == 0
        for row in (scoped_before, global_before)
    ):
        raise Sam31VastCanaryError("sam31_provider_not_zero_before_launch")

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
        raise Sam31VastCanaryError("sam31_paid_lane_not_acquired")
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
    validated_result: dict[str, Any] | None = None
    blockers: list[str] = []
    provider_mutations = 0
    try:
        env = {
            INPUT_GET_ENV: input_bundle_get_url,
            OUTPUT_PUT_ENV: output_put_url,
            HF_TOKEN_ENV: hf_token,
            "BLUEPRINT_SAM31_CANARY_REQUEST_DIGEST": request_digest,
            "BLUEPRINT_SAM31_BOUND_REQUEST_DIGEST": str(request.get("bound_request_digest") or ""),
            "BLUEPRINT_CONTAINER_IMAGE_DIGEST": image,
            "BLUEPRINT_SAM31_RUNTIME_DIGEST": image,
            "BLUEPRINT_SAM31_INPUT_BUNDLE_DIGEST": str(request.get("input_bundle_digest") or ""),
            "BLUEPRINT_SAM31_SOURCE_TRACK_REQUEST_DIGEST": str(
                request.get("source_track_run_request_digest") or ""
            ),
            "BLUEPRINT_SAM31_EXPECTED_CHECKPOINT_DIGEST": CHECKPOINT_DIGEST,
            "BLUEPRINT_SAM31_CHECKPOINT_REPOSITORY_REVISION": (CHECKPOINT_REPOSITORY_REVISION),
            "BLUEPRINT_SAM31_OFFICIAL_CODE_REVISION": OFFICIAL_CODE_REVISION,
        }
        spec = RenderLaunchSpec(
            name=name,
            image=image,
            env=env,
            bootstrap_argv=["-lc", _bootstrap_script()],
            entrypoint=["bash"],
            container_disk_gb=max(40, int(preflight.get("container_disk_bytes") or 0) // 1024**3),
            volume_gb=0,
            max_hourly_rate_usd=float(preflight.get("on_demand_price_usd_per_hour") or 0),
            min_gpu_ram_mb=max(24_000, int(preflight.get("gpu_memory_bytes") or 0) // 1_000_000),
            requires_rtx=False,
            vast_launch_mode="args",
        )
        provider_request = provider.build_request(spec, root)
        provider_request["prelaunch_spend_guard"] = {
            "schema_version": "semantic_sam31_gpu_prelaunch_spend_guard.v1",
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
                reason="sam31_vast_create_outcome_ambiguous",
                evidence={"blockers": list(launch_result.get("blockers") or [])},
            )
            blockers.append("sam31_vast_create_outcome_ambiguous")
        elif launch_result.get("status") != "launched" or not launch_result.get("instance_id"):
            cancel_pending_teardown(
                pending_path,
                reason="provider_confirmed_no_allocation",
                evidence={"status": launch_result.get("status")},
            )
            blockers.append("sam31_vast_instance_not_created")
        else:
            instance_id = str(launch_result["instance_id"])
            provider_mutations += 1
            bind_pending_teardown_instance(pending_path, instance_id)
            watchdog_instance_value = str(watchdog.get("started_instance_id_path") or "")
            if watchdog_instance_value:
                watchdog_instance_path = Path(watchdog_instance_value).expanduser()
                if not watchdog_instance_path.is_absolute() or watchdog_instance_path.is_symlink():
                    blockers.append("sam31_watchdog_instance_binding_path_invalid")
                else:
                    write_started_vast_instance_id(watchdog_instance_path, int(instance_id))
            raw_result: dict[str, Any] | None = None
            while float(clock()) - started_at <= hard_ttl:
                try:
                    raw_result = dict(result_fetcher(output_get_url))
                    break
                except (FileNotFoundError, TimeoutError):
                    if float(clock()) - started_at >= hard_ttl:
                        break
                    sleeper(
                        min(
                            5.0,
                            max(0.0, hard_ttl - (float(clock()) - started_at)),
                        )
                    )
            if raw_result is None:
                blockers.append("sam31_canary_output_timeout")
            else:
                write_json(root / "provider_runtime_result.json", raw_result)
                try:
                    validated_result = validate_sam31_runtime_result(
                        raw_result, bound_request=request
                    )
                except Sam31VastCanaryError as exc:
                    blockers.extend(str(exc).split(";"))
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
                or terminate_result.get("status") in {"stopped", "terminated", "deleted"}
            )
        )
        teardown = {
            "schema_version": TEARDOWN_SCHEMA_VERSION,
            "status": "PASS" if teardown_passed else "FAIL",
            "provider": "vast",
            "request_digest": request_digest,
            "bound_request_digest": request.get("bound_request_digest"),
            "worker_image_digest": image,
            "instance_id": instance_id,
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
                evidence={"teardown_digest": teardown["teardown_receipt_digest"]},
            )
        reconciliation_after = build_paid_provider_lane_reconciliation(
            provider="vast",
            lane=PAID_LANE,
            provider_inventory=global_after,
            open_pending_teardowns=load_pending_teardowns(registry_dir=pending_dir),
        )
        release = release_paid_provider_lane_lease(
            lease,
            reason="sam31_source_track_canary_terminal",
            provider_mutation_started=instance_id is not None,
            terminal_reconciliation=reconciliation_after,
            lease_dir=lease_dir,
        )
        if not teardown_passed:
            blockers.append("sam31_teardown_verification_failed")
        if instance_id is not None and release.get("released") is not True:
            blockers.append("sam31_paid_lane_release_blocked")
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
        blockers.append("sam31_budget_exhausted")
    result = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "status": "completed" if validated_result is not None and not blockers else "failed",
        "request_digest": request_digest,
        "bound_request_digest": request.get("bound_request_digest"),
        "worker_image_digest": image,
        "source_commit_sha": request.get("source_commit_sha"),
        "input_bundle_digest": request.get("input_bundle_digest"),
        "source_track_run_request_digest": request.get("source_track_run_request_digest"),
        "checkpoint_digest": CHECKPOINT_DIGEST,
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
        "blockers": sorted(set(blockers)),
        "teardown_receipt_digest": teardown["teardown_receipt_digest"],
        "provider_zero_digest": provider_zero_receipt["provider_zero_digest"],
        "provider_zero_verified": provider_zero_receipt["status"] == "PASS",
        "raw_secret_values_recorded": False,
        "scientific_qualification_inferred": False,
        "proof_effect": "none",
        "claim_ceiling": "source_bound_2d_binary_mask_tracks_only"
        if validated_result is not None
        else "none",
        "comparative_policy_ranking_verdict": "thesis_not_supported",
    }
    result["execution_result_digest"] = canonical_digest(
        result, digest_field="execution_result_digest"
    )
    write_json(root / "sam31_vast_source_track_execution.json", result)
    return result


__all__ = [
    "EXECUTION_SCHEMA_VERSION",
    "NAME_PREFIX",
    "PAID_LANE",
    "Sam31VastCanaryError",
    "run_sam31_vast_source_track_canary",
    "validate_sam31_runtime_result",
]
