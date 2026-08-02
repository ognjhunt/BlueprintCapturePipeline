"""Grant-gated Vast lifecycle for exact-package Isaac compatibility checks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
import urllib.error

from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .external_provider_nurec import (
    ExternalProviderNuRecError,
    build_provider_nurec_isaac_request,
    normalize_provider_nurec_isaac_verification,
)
from .external_scene_isaac_verification import (
    ExternalSceneIsaacVerificationError,
    build_external_scene_isaac_verification_request,
    normalize_external_scene_isaac_verification,
)
from .gpu_render_providers import GpuRenderProvider, RenderLaunchSpec
from .isaac_reconstruction_verification import (
    IsaacReconstructionVerificationError,
    build_isaac_asset_verification_request,
    normalize_isaac_reconstruction_verification,
)
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
from .reconstruction_isaac_bootstrap import (
    MAX_INPUT_BUNDLE_BYTES,
    MAX_OUTPUT_BUNDLE_BYTES,
    MAX_RECEIPT_BYTES,
)
from .reconstruction_isaac_output_bundle import (
    IsaacVerificationOutputBundleError,
    validate_and_extract_isaac_verification_output_bundle,
)
from .reconstruction_isaac_worker_bundle import (
    PROVIDER_ISAAC_WORKER_BUNDLE_SCHEMA,
    IsaacWorkerBundleError,
    extract_isaac_verification_worker_bundle,
    validate_isaac_verification_worker_bundle_receipt,
)
from .safe_outbound_http import (
    SafeHttpFileTransfer,
    SafeOutboundHttpError,
    download_file,
    download_file_observed,
    presigned_transfer_policy,
    validate_outbound_url,
)


SCHEMA_VERSION = "reconstruction_isaac_vast_execution.v1"
REPLAY_SCHEMA_VERSION = "reconstruction_isaac_vast_replay.v1"
TEARDOWN_SCHEMA_VERSION = "reconstruction_isaac_vast_teardown.v1"
PROVIDER_ZERO_SCHEMA_VERSION = "reconstruction_isaac_vast_provider_zero.v1"
QUALIFICATION_SCHEMA_VERSION = "reconstruction_isaac_independent_qualification.v1"
PAID_LANE = "reconstruction_gpu_canary"
NAME_PREFIX = "blueprint-reconstruction-"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class ReconstructionIsaacVastError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionIsaacVastError([code]) from exc
    if not isinstance(value, Mapping):
        raise ReconstructionIsaacVastError([code])
    return dict(value)


def _canonical_receipt(root: Path, receipt: Mapping[str, Any]) -> tuple[Path, str]:
    path = root / "isaac_input_bundle_receipt.json"
    payload = (json.dumps(dict(receipt), sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
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


def _exact_fetcher(
    url: str, destination: Path, expected_digest: str, maximum_bytes: int
) -> SafeHttpFileTransfer:
    return download_file(
        url,
        output_path=destination,
        expected_sha256=expected_digest,
        max_bytes=maximum_bytes,
        timeout_seconds=3600,
        policy=presigned_transfer_policy(url),
    )


def _output_fetcher(url: str, destination: Path) -> SafeHttpFileTransfer:
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
            raise FileNotFoundError("isaac_output_not_ready") from exc
        raise


def _bootstrap_script() -> str:
    return """set -euo pipefail
if [ -x /isaac-sim/python.sh ]; then
  exec /isaac-sim/python.sh -m blueprint_pipeline.reconstruction_isaac_bootstrap
fi
if command -v python3 >/dev/null 2>&1; then
  exec python3 -m blueprint_pipeline.reconstruction_isaac_bootstrap
fi
echo BLUEPRINT_RECONSTRUCTION_ISAAC_BLOCKED:python_runtime_missing >&2
exit 127
"""


def _validate_bindings(
    *, bound_request: Mapping[str, Any], bundle_receipt: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = json.loads(json.dumps(dict(bound_request)))
    try:
        receipt = validate_isaac_verification_worker_bundle_receipt(bundle_receipt)
    except IsaacWorkerBundleError as exc:
        raise ReconstructionIsaacVastError(
            [f"reconstruction_isaac_vast_receipt_invalid:{code}" for code in exc.codes]
        ) from exc
    blockers: list[str] = []
    provider_bundle = receipt.get("schema_version") == PROVIDER_ISAAC_WORKER_BUNDLE_SCHEMA
    external_scene_bundle = (
        receipt.get("verification_request_member")
        == "external_scene_isaac_verification_request.v1.json"
    )
    expected_operation = (
        "provider_nurec_isaac_canary"
        if provider_bundle
        else ("external_scene_isaac_canary" if external_scene_bundle else "isaac_canary")
    )
    expected_runtime_schema = (
        "provider_nurec_isaac_runtime_result.v1"
        if provider_bundle
        else "isaac_splat_nurec_render_result.v3"
    )
    if request.get("bound_request_digest") != canonical_digest(
        request, digest_field="bound_request_digest"
    ):
        blockers.append("reconstruction_isaac_vast_bound_request_digest_mismatch")
    if (
        request.get("operation") != expected_operation
        or request.get("expected_runtime_result_schema") != expected_runtime_schema
        or request.get("bound_provider") != "vast"
        or request.get("provider_mutation_authorized") is not True
        or request.get("bound_checkout_clean") is not True
        or request.get("candidate_may_read_hidden_heldout") is not False
        or request.get("trainer_may_grade_heldout") is not False
        or request.get("proof_effect") != "none"
    ):
        blockers.append("reconstruction_isaac_vast_request_not_executable")
    for request_key, receipt_key in (
        ("operation_request_digest", "isaac_verification_request_digest"),
        ("operation_input_bundle_digest", "bundle_digest"),
        ("worker_image_digest", "runtime_container_image_digest"),
        ("source_commit_sha", "source_commit_sha"),
    ):
        if request.get(request_key) != receipt.get(receipt_key):
            blockers.append(f"reconstruction_isaac_vast_{request_key}_mismatch")
    if request.get("bound_checkout_source_commit") != request.get("source_commit_sha"):
        blockers.append("reconstruction_isaac_vast_checkout_sha_mismatch")
    if _DIGEST.fullmatch(str(request.get("isaac_image_release_digest") or "")) is None:
        blockers.append("reconstruction_isaac_vast_image_release_unbound")
    if blockers:
        raise ReconstructionIsaacVastError(blockers)
    return request, receipt


def _safe_package_root(materialized: Path, request: Mapping[str, Any], root: Path) -> Path:
    reference = PurePosixPath(str(request.get("package_artifact_reference") or ""))
    if (
        reference.is_absolute()
        or any(part in {"", ".", ".."} for part in reference.parts)
        or ":" in reference.parts[0]
    ):
        raise ReconstructionIsaacVastError(["reconstruction_isaac_vast_package_reference_unsafe"])
    source = materialized / "reconstruction.usdz"
    destination = root.joinpath(*reference.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if _sha256(destination) != request.get("package_digest"):
        raise ReconstructionIsaacVastError(
            ["reconstruction_isaac_vast_staged_package_digest_mismatch"]
        )
    return root


def _independent_qualification(
    *,
    verification_request: Mapping[str, Any],
    runtime_result: Mapping[str, Any],
    package_root: Path,
    runtime_root: Path,
    normalizer: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    blockers: list[str] = []
    qualified: dict[str, Any] | None = None
    try:
        qualified = normalizer(
            verification_request=verification_request,
            runtime_result=runtime_result,
            package_artifact_root=package_root,
            runtime_artifact_root=runtime_root,
        )
    except (
        ExternalProviderNuRecError,
        ExternalSceneIsaacVerificationError,
        IsaacReconstructionVerificationError,
    ) as exc:
        blockers.extend(exc.codes)
    result = {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "status": ("verified_compatibility_only" if qualified is not None else "abstained"),
        "isaac_verification_request_digest": verification_request.get(
            "isaac_verification_request_digest"
        ),
        "runtime_result_digest": runtime_result.get("isaac_runtime_result_digest"),
        "qualified_result_digest": (
            (
                qualified.get("provider_isaac_verification_result_digest")
                or qualified.get("isaac_verification_result_digest")
            )
            if qualified
            else None
        ),
        "blockers": sorted(set(blockers)),
        "simulator_task_success_proven": False,
        "physical_success_proven": False,
        "deployment_readiness_proven": False,
        "proof_effect": ("isaac_load_render_physics_presence_only" if qualified else "none"),
        "claim_ceiling": (
            "isaac_load_render_compatibility" if qualified else "runtime_evidence_only"
        ),
    }
    result["qualification_digest"] = canonical_digest(result, digest_field="qualification_digest")
    return result


def run_reconstruction_isaac_vast_operation(
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
    exact_fetcher: Callable[[str, Path, str, int], SafeHttpFileTransfer] = _exact_fetcher,
    output_fetcher: Callable[[str, Path], SafeHttpFileTransfer] = _output_fetcher,
    output_validator: Callable[..., tuple[dict[str, Any], dict[str, Any], Path]] = (
        validate_and_extract_isaac_verification_output_bundle
    ),
    normalizer: Callable[..., dict[str, Any]] = normalize_isaac_reconstruction_verification,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
    watchdog_validator: Callable[[Mapping[str, Any], float, int], bool] | None = None,
) -> dict[str, Any]:
    """Stage, execute, retrieve, independently qualify, and tear down Isaac."""

    require_paid_resource_admission_grant(
        paid_resource_admission_grant, resource_class="gpu_render"
    )
    request, receipt = _validate_bindings(
        bound_request=bound_request, bundle_receipt=bundle_receipt
    )
    provider_bundle = receipt.get("schema_version") == PROVIDER_ISAAC_WORKER_BUNDLE_SCHEMA
    external_scene_bundle = (
        receipt.get("verification_request_member")
        == "external_scene_isaac_verification_request.v1.json"
    )
    operation = str(request["operation"])
    try:
        for url in (
            input_bundle_get_url,
            input_receipt_get_url,
            output_bundle_put_url,
            output_bundle_get_url,
        ):
            validate_outbound_url(url, policy=presigned_transfer_policy(url))
    except SafeOutboundHttpError as exc:
        raise ReconstructionIsaacVastError(
            ["reconstruction_isaac_vast_transport_url_invalid"]
        ) from exc
    if provider.name != "vast":
        raise ReconstructionIsaacVastError(["reconstruction_isaac_vast_first_required"])
    request_digest = str(request.get("request_digest") or "")
    worker_image = str(request["worker_image_digest"])
    source_commit = str(request["source_commit_sha"])
    hard_ttl = int(request.get("hard_ttl_seconds") or 0)
    max_spend = float(request.get("max_spend_usd") or 0)
    retry_cap = int(request.get("retry_cap") or 0)
    if (
        _DIGEST.fullmatch(request_digest) is None
        or not 300 <= hard_ttl <= 14_400
        or max_spend <= 0
        or not 0 <= retry_cap <= 2
    ):
        raise ReconstructionIsaacVastError(["reconstruction_isaac_vast_execution_bounds_invalid"])

    root = Path(job_dir)
    ensure_dir(root)
    for path in (
        root / "pending_teardowns",
        root / "leases",
        root / "retrieval_attempts",
        root / "staged_inputs",
    ):
        ensure_dir(path)
    local_receipt, receipt_file_digest = _canonical_receipt(root, receipt)
    remote_receipt = root / "staged_inputs/remote_receipt.json"
    staged_bundle = root / "staged_inputs/isaac_input.zip"
    try:
        receipt_transfer = exact_fetcher(
            input_receipt_get_url,
            remote_receipt,
            receipt_file_digest,
            MAX_RECEIPT_BYTES,
        )
        remote_value = validate_isaac_verification_worker_bundle_receipt(
            _load(
                remote_receipt,
                code="reconstruction_isaac_vast_remote_receipt_invalid",
            )
        )
        if remote_value != receipt or _sha256(local_receipt) != receipt_transfer.sha256:
            raise ReconstructionIsaacVastError(
                ["reconstruction_isaac_vast_remote_receipt_mismatch"]
            )
        exact_fetcher(
            input_bundle_get_url,
            staged_bundle,
            receipt["bundle_digest"],
            MAX_INPUT_BUNDLE_BYTES,
        )
        extraction = extract_isaac_verification_worker_bundle(
            bundle_path=staged_bundle,
            bundle_receipt=receipt,
            output_root=root / "staged_inputs/materialized",
        )
        materialized = root / "staged_inputs/materialized" / receipt["bundle_digest"][7:]
        request_member = str(receipt["verification_request_member"])
        request_value = _load(
            materialized / request_member,
            code="reconstruction_isaac_vast_verification_request_invalid",
        )
        verification_request = (
            build_provider_nurec_isaac_request(request_value)
            if provider_bundle
            else (
                build_external_scene_isaac_verification_request(request_value)
                if external_scene_bundle
                else build_isaac_asset_verification_request(request_value)
            )
        )
        if (
            verification_request["isaac_verification_request_digest"]
            != receipt["isaac_verification_request_digest"]
        ):
            raise ReconstructionIsaacVastError(
                ["reconstruction_isaac_vast_verification_request_mismatch"]
            )
        package_root = _safe_package_root(
            materialized, verification_request, root / "staged_package"
        )
    except (
        SafeOutboundHttpError,
        ExternalProviderNuRecError,
        ExternalSceneIsaacVerificationError,
        IsaacWorkerBundleError,
        IsaacReconstructionVerificationError,
    ) as exc:
        raise ReconstructionIsaacVastError(
            ["reconstruction_isaac_vast_prelaunch_staging_invalid"]
        ) from exc
    staging = {
        "status": "validated_before_provider_launch",
        "bundle_digest": receipt["bundle_digest"],
        "receipt_file_digest": receipt_file_digest,
        "extraction_receipt_digest": extraction["extraction_receipt_digest"],
        "package_digest": receipt["package_digest"],
        "provider_mutations_performed": 0,
        "proof_effect": "none",
    }
    staging["staging_digest"] = canonical_digest(staging, digest_field="staging_digest")
    write_json(root / "prelaunch_staging_receipt.json", staging)

    started_at = float(clock())
    watchdog = preflight.get("watchdog")
    watchdog = watchdog if isinstance(watchdog, Mapping) else {}
    validator = watchdog_validator or (
        lambda value, now, ttl: _watchdog_valid(value, now_epoch=now, hard_ttl_seconds=ttl)
    )
    if not validator(watchdog, started_at, hard_ttl):
        raise ReconstructionIsaacVastError(["reconstruction_isaac_vast_watchdog_not_live"])
    scoped_before = provider.billable_inventory(name_prefix=NAME_PREFIX)
    global_before = provider.billable_inventory(name_prefix="")
    if not all(
        row.get("api_confirmed") is True and row.get("live_resource_count") == 0
        for row in (scoped_before, global_before)
    ):
        raise ReconstructionIsaacVastError(
            ["reconstruction_isaac_vast_provider_not_zero_before_launch"]
        )

    pending_dir = root / "pending_teardowns"
    lease_dir = root / "leases"
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
        raise ReconstructionIsaacVastError(["reconstruction_isaac_vast_paid_lane_not_acquired"])
    name = f"{NAME_PREFIX}isaac-{request_digest[7:19]}"
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
    validated_receipt: dict[str, Any] | None = None
    runtime_result: dict[str, Any] | None = None
    qualification: dict[str, Any] | None = None
    retrieved: SafeHttpFileTransfer | None = None
    output_retrieved_before_teardown = False
    provider_mutations = 0
    blockers: list[str] = []
    invalid_digests: set[str] = set()
    fetch_attempts = 0
    try:
        spec = RenderLaunchSpec(
            name=name,
            image=worker_image,
            env={
                "ACCEPT_EULA": "Y",
                "PRIVACY_CONSENT": "Y",
                "CUDA_VISIBLE_DEVICES": "0",
                "BLUEPRINT_ISAAC_INPUT_BUNDLE_GET_URL": input_bundle_get_url,
                "BLUEPRINT_ISAAC_INPUT_RECEIPT_GET_URL": input_receipt_get_url,
                "BLUEPRINT_ISAAC_OUTPUT_BUNDLE_PUT_URL": output_bundle_put_url,
                "BLUEPRINT_ISAAC_INPUT_BUNDLE_DIGEST": receipt["bundle_digest"],
                "BLUEPRINT_ISAAC_INPUT_RECEIPT_FILE_DIGEST": receipt_file_digest,
                "BLUEPRINT_ISAAC_VERIFICATION_REQUEST_DIGEST": receipt[
                    "isaac_verification_request_digest"
                ],
                "BLUEPRINT_CONTAINER_IMAGE_DIGEST": worker_image,
                "BLUEPRINT_SOURCE_COMMIT": source_commit,
                "BLUEPRINT_RECONSTRUCTION_HARD_TTL_SECONDS": str(hard_ttl),
            },
            bootstrap_argv=["-lc", _bootstrap_script()],
            entrypoint=["bash"],
            container_disk_gb=max(100, int(preflight.get("container_disk_bytes") or 0) // 1024**3),
            volume_gb=0,
            max_hourly_rate_usd=float(preflight.get("on_demand_price_usd_per_hour") or 0),
            min_gpu_ram_mb=max(24_000, int(preflight.get("gpu_memory_bytes") or 0) // 1_000_000),
            requires_rtx=True,
            vast_launch_mode="args",
        )
        provider_request = provider.build_request(spec, root)
        provider_request["prelaunch_spend_guard"] = {
            "schema_version": "reconstruction_isaac_prelaunch_spend_guard.v1",
            "required_before_provider_launch": True,
            "can_launch": True,
            "max_spend_usd": max_spend,
            "hard_ttl_seconds": hard_ttl,
            "retry_cap": retry_cap,
            "request_digest": request_digest,
            "gpu_count": 1,
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
                reason="reconstruction_isaac_vast_create_ambiguous",
                evidence={"blockers": list(launch_result.get("blockers") or [])},
            )
            blockers.append("reconstruction_isaac_vast_create_ambiguous")
        elif launch_result.get("status") != "launched" or not launch_result.get("instance_id"):
            cancel_pending_teardown(
                pending_path,
                reason="provider_confirmed_no_allocation",
                evidence={"status": launch_result.get("status")},
            )
            blockers.append("reconstruction_isaac_vast_instance_not_created")
        else:
            instance_id = str(launch_result["instance_id"])
            provider_mutations += 1
            bind_pending_teardown_instance(pending_path, instance_id)
            while float(clock()) - started_at <= hard_ttl:
                fetch_attempts += 1
                attempt = root / f"retrieval_attempts/output_{fetch_attempts:04d}.zip"
                try:
                    retrieved = output_fetcher(output_bundle_get_url, attempt)
                except (FileNotFoundError, TimeoutError):
                    if float(clock()) - started_at >= hard_ttl:
                        break
                    sleeper(min(5.0, max(0.0, hard_ttl - (float(clock()) - started_at))))
                    continue
                except Exception as exc:  # noqa: BLE001
                    blockers.append(
                        f"reconstruction_isaac_vast_output_fetch_failed:{type(exc).__name__}"
                    )
                    break
                try:
                    validated_receipt, runtime_result, runtime_root = output_validator(
                        bundle_path=attempt,
                        expected_input_receipt=receipt,
                        expected_source_commit_sha=source_commit,
                        output_root=root / f"retrieval_attempts/validated_{fetch_attempts:04d}",
                    )
                    if validated_receipt.get("output_bundle_digest") != retrieved.sha256:
                        raise IsaacVerificationOutputBundleError(
                            ["isaac_output_transport_digest_mismatch"]
                        )
                except IsaacVerificationOutputBundleError as exc:
                    write_json(
                        attempt.with_suffix(".rejection.json"),
                        {
                            "attempt": fetch_attempts,
                            "observed_digest": retrieved.sha256,
                            "blockers": list(exc.codes),
                            "failed_evidence_preserved": True,
                        },
                    )
                    if retrieved.sha256 in invalid_digests:
                        blockers.append("reconstruction_isaac_vast_repeated_identical_blocker")
                        break
                    invalid_digests.add(retrieved.sha256)
                    if len(invalid_digests) > retry_cap:
                        blockers.append("reconstruction_isaac_vast_retry_cap_exhausted")
                        break
                    continue
                qualification = _independent_qualification(
                    verification_request=verification_request,
                    runtime_result=runtime_result,
                    package_root=package_root,
                    runtime_root=runtime_root,
                    normalizer=(
                        normalize_provider_nurec_isaac_verification
                        if provider_bundle
                        else (
                            normalize_external_scene_isaac_verification
                            if external_scene_bundle
                            else normalizer
                        )
                    ),
                )
                output_retrieved_before_teardown = True
                write_json(root / "validated_output_bundle_receipt.json", validated_receipt)
                write_json(root / "provider_runtime_result.json", runtime_result)
                write_json(root / "independent_isaac_qualification.json", qualification)
                break
            if validated_receipt is None:
                blockers.append("reconstruction_isaac_vast_output_not_accepted")
    except Exception as exc:  # noqa: BLE001
        blockers.append(f"reconstruction_isaac_vast_controller_failure:{type(exc).__name__}")
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
                or terminate_result.get("status") in {"stopped", "terminated", "deleted"}
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
        terminal = build_paid_provider_lane_reconciliation(
            provider="vast",
            lane=PAID_LANE,
            provider_inventory=global_after,
            open_pending_teardowns=load_pending_teardowns(registry_dir=pending_dir),
        )
        lease_release = release_paid_provider_lane_lease(
            lease,
            reason="reconstruction_isaac_terminal",
            provider_mutation_started=instance_id is not None,
            terminal_reconciliation=terminal,
            lease_dir=lease_dir,
        )
        if not teardown_passed:
            blockers.append("reconstruction_isaac_vast_teardown_verification_failed")
        if instance_id is not None and lease_release.get("released") is not True:
            blockers.append("reconstruction_isaac_vast_paid_lane_release_blocked")
        provider_zero_receipt = {
            "schema_version": PROVIDER_ZERO_SCHEMA_VERSION,
            "status": "PASS" if provider_zero else "FAIL",
            "provider": "vast",
            "operation": operation,
            "request_digest": request_digest,
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
        blockers.append("reconstruction_isaac_vast_budget_exhausted")
    execution = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if validated_receipt is not None and not blockers else "failed",
        "operation": operation,
        "request_digest": request_digest,
        "bound_request_digest": request.get("bound_request_digest"),
        "isaac_verification_request_digest": receipt["isaac_verification_request_digest"],
        "input_bundle_digest": receipt["bundle_digest"],
        "worker_image_digest": worker_image,
        "source_commit_sha": source_commit,
        "provider": "vast",
        "instance_id": instance_id,
        "gpu_count": 1,
        "fetch_attempts": fetch_attempts,
        "output_bundle_digest": retrieved.sha256 if retrieved else None,
        "output_bundle_receipt_digest": (
            validated_receipt.get("output_bundle_receipt_digest") if validated_receipt else None
        ),
        "runtime_result_digest": (
            runtime_result.get("isaac_runtime_result_digest") if runtime_result else None
        ),
        "runtime_status": runtime_result.get("status") if runtime_result else None,
        "independent_qualification_status": (
            qualification.get("status") if qualification else None
        ),
        "independent_qualification_digest": (
            qualification.get("qualification_digest") if qualification else None
        ),
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
        "simulator_task_success_proven": False,
        "physical_success_proven": False,
        "deployment_readiness_proven": False,
        "proof_effect": (
            "isaac_load_render_physics_presence_only"
            if qualification and qualification.get("status") == "verified_compatibility_only"
            else "none"
        ),
        "claim_ceiling": (
            "isaac_load_render_compatibility"
            if qualification and qualification.get("status") == "verified_compatibility_only"
            else "provider_execution_and_runtime_evidence_only"
        ),
    }
    execution["execution_result_digest"] = canonical_digest(
        execution, digest_field="execution_result_digest"
    )
    write_json(root / "reconstruction_isaac_vast_execution.json", execution)
    return execution


def replay_reconstruction_isaac_vast_operation(
    *,
    job_dir: str | Path,
    bound_request: Mapping[str, Any],
    bundle_receipt: Mapping[str, Any],
    output_validator: Callable[..., tuple[dict[str, Any], dict[str, Any], Path]] = (
        validate_and_extract_isaac_verification_output_bundle
    ),
    normalizer: Callable[..., dict[str, Any]] = normalize_isaac_reconstruction_verification,
) -> dict[str, Any]:
    """Reproduce qualification and teardown status without live agent/provider access."""

    root = Path(job_dir)
    request, receipt = _validate_bindings(
        bound_request=bound_request, bundle_receipt=bundle_receipt
    )
    provider_bundle = receipt.get("schema_version") == PROVIDER_ISAAC_WORKER_BUNDLE_SCHEMA
    external_scene_bundle = (
        receipt.get("verification_request_member")
        == "external_scene_isaac_verification_request.v1.json"
    )
    operation = str(request["operation"])
    blockers: list[str] = []
    try:
        execution = _load(
            root / "reconstruction_isaac_vast_execution.json",
            code="reconstruction_isaac_vast_replay_execution_missing",
        )
        teardown = _load(
            root / "teardown_receipt.json",
            code="reconstruction_isaac_vast_replay_teardown_missing",
        )
        provider_zero = _load(
            root / "provider_zero_verification.json",
            code="reconstruction_isaac_vast_replay_provider_zero_missing",
        )
        recorded_output = _load(
            root / "validated_output_bundle_receipt.json",
            code="reconstruction_isaac_vast_replay_output_receipt_missing",
        )
        recorded_runtime = _load(
            root / "provider_runtime_result.json",
            code="reconstruction_isaac_vast_replay_runtime_missing",
        )
        recorded_qualification = _load(
            root / "independent_isaac_qualification.json",
            code="reconstruction_isaac_vast_replay_qualification_missing",
        )
        local_receipt = validate_isaac_verification_worker_bundle_receipt(
            _load(
                root / "isaac_input_bundle_receipt.json",
                code="reconstruction_isaac_vast_replay_input_receipt_missing",
            )
        )
        if local_receipt != receipt:
            blockers.append("reconstruction_isaac_vast_replay_input_receipt_mismatch")
        staged_bundle = root / "staged_inputs/isaac_input.zip"
        extraction = extract_isaac_verification_worker_bundle(
            bundle_path=staged_bundle,
            bundle_receipt=receipt,
            output_root=root / "staged_inputs/materialized",
        )
        materialized = root / "staged_inputs/materialized" / receipt["bundle_digest"][7:]
        request_value = _load(
            materialized / str(receipt["verification_request_member"]),
            code="reconstruction_isaac_vast_replay_request_missing",
        )
        verification_request = (
            build_provider_nurec_isaac_request(request_value)
            if provider_bundle
            else (
                build_external_scene_isaac_verification_request(request_value)
                if external_scene_bundle
                else build_isaac_asset_verification_request(request_value)
            )
        )
        package_root = root / "staged_package"
    except (
        ReconstructionIsaacVastError,
        ExternalProviderNuRecError,
        ExternalSceneIsaacVerificationError,
        IsaacWorkerBundleError,
        IsaacReconstructionVerificationError,
    ) as exc:
        blockers.extend(
            exc.codes
            if hasattr(exc, "codes")
            else ["reconstruction_isaac_vast_replay_input_invalid"]
        )
        execution = {}
        teardown = {}
        provider_zero = {}
        recorded_output = {}
        recorded_runtime = {}
        recorded_qualification = {}
        extraction = {}
        verification_request = {}
        package_root = root / "staged_package"

    attempts = sorted((root / "retrieval_attempts").glob("output_*.zip"))
    accepted = next(
        (
            path
            for path in reversed(attempts)
            if _sha256(path) == execution.get("output_bundle_digest")
        ),
        None,
    )
    validated_output: dict[str, Any] = {}
    validated_runtime: dict[str, Any] = {}
    replayed_qualification: dict[str, Any] = {}
    temporary_root = Path(tempfile.mkdtemp(prefix="blueprint-isaac-replay-"))
    try:
        if accepted is None:
            blockers.append("reconstruction_isaac_vast_replay_output_bundle_missing")
        else:
            try:
                validated_output, validated_runtime, runtime_root = output_validator(
                    bundle_path=accepted,
                    expected_input_receipt=receipt,
                    expected_source_commit_sha=str(request.get("source_commit_sha") or ""),
                    output_root=temporary_root,
                )
                replayed_qualification = _independent_qualification(
                    verification_request=verification_request,
                    runtime_result=validated_runtime,
                    package_root=package_root,
                    runtime_root=runtime_root,
                    normalizer=(
                        normalize_provider_nurec_isaac_verification
                        if provider_bundle
                        else (
                            normalize_external_scene_isaac_verification
                            if external_scene_bundle
                            else normalizer
                        )
                    ),
                )
            except (IsaacVerificationOutputBundleError, OSError) as exc:
                codes = getattr(exc, "codes", [type(exc).__name__])
                blockers.extend(
                    f"reconstruction_isaac_vast_replay_output_invalid:{code}" for code in codes
                )
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    for value, field, code in (
        (execution, "execution_result_digest", "execution_digest_mismatch"),
        (teardown, "teardown_receipt_digest", "teardown_digest_mismatch"),
        (provider_zero, "provider_zero_digest", "provider_zero_digest_mismatch"),
        (
            recorded_output,
            "output_bundle_receipt_digest",
            "output_receipt_digest_mismatch",
        ),
        (
            recorded_qualification,
            "qualification_digest",
            "qualification_digest_mismatch",
        ),
    ):
        if value and value.get(field) != canonical_digest(value, digest_field=field):
            blockers.append(f"reconstruction_isaac_vast_replay_{code}")
    if (
        execution.get("status") != "completed"
        or execution.get("bound_request_digest") != request.get("bound_request_digest")
        or execution.get("output_retrieved_before_teardown") is not True
        or execution.get("output_bundle_receipt_digest")
        != recorded_output.get("output_bundle_receipt_digest")
        or recorded_output != validated_output
        or recorded_runtime != validated_runtime
        or recorded_qualification != replayed_qualification
        or not extraction
    ):
        blockers.append("reconstruction_isaac_vast_replay_binding_mismatch")
    if (
        teardown.get("status") != "PASS"
        or teardown.get("output_retrieved_before_teardown") is not True
        or teardown.get("provider_zero_verified") is not True
        or provider_zero.get("status") != "PASS"
        or provider_zero.get("api_confirmed") is not True
        or provider_zero.get("scoped_live_resource_count") != 0
        or provider_zero.get("global_live_resource_count") != 0
    ):
        blockers.append("reconstruction_isaac_vast_replay_teardown_not_verified")
    replay = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "status": "replay_verified" if not blockers else "replay_rejected",
        "operation": operation,
        "request_digest": request.get("request_digest"),
        "bound_request_digest": request.get("bound_request_digest"),
        "execution_result_digest": execution.get("execution_result_digest"),
        "runtime_status": execution.get("runtime_status"),
        "independent_qualification_status": execution.get("independent_qualification_status"),
        "blockers": sorted(set(blockers)),
        "live_provider_accessed": False,
        "live_agent_accessed": False,
        "simulator_task_success_proven": False,
        "physical_success_proven": False,
        "deployment_readiness_proven": False,
        "proof_effect": recorded_qualification.get("proof_effect", "none"),
        "claim_ceiling": recorded_qualification.get(
            "claim_ceiling", "recorded_runtime_evidence_only"
        ),
    }
    replay["replay_digest"] = canonical_digest(replay, digest_field="replay_digest")
    write_json(root / "reconstruction_isaac_vast_replay.json", replay)
    return replay


__all__ = [
    "ReconstructionIsaacVastError",
    "replay_reconstruction_isaac_vast_operation",
    "run_reconstruction_isaac_vast_operation",
]
