"""Run one Postshot arm on a governed AWS Windows GPU instance.

Postshot ships only ``postshot-cli.exe``, so its arm cannot use the Vast Linux
operation every other reconstruction lane shares.  This is the Windows
equivalent: same lifecycle, same receipt shape, same refusal to treat a
finished run as a quality claim.

The properties that matter are about money and evidence, not convenience:

* **Teardown is unconditional.** Termination runs in a ``finally``, so a crash,
  a TTL expiry, a tampered output, or an exception on the happy path all still
  release the instance.  A run that cannot prove teardown reports the ambiguity
  rather than claiming closure.
* **Provider-zero is checked on both sides.** A non-empty inventory before
  launch aborts before spending; a non-empty inventory after teardown is a
  blocker, because "we asked it to stop" is not "it stopped".
* **Output is retrieved and validated before teardown**, so a terminated
  instance can never be the reason results are missing.
* **The clock is the instance's enemy, not ours.** A hard deadline bounds
  polling; the instance also arms its own shutdown, and the independent
  watchdog remains mandatory because neither of those is provider teardown.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .common import ensure_dir, write_json

SCHEMA_VERSION = "reconstruction_aws_windows_operation_execution.v1"
PROVIDER = "aws"
POLL_INTERVAL_SECONDS = 30.0
TEARDOWN_POLL_INTERVAL_SECONDS = 15.0
MAX_TEARDOWN_WAIT_SECONDS = 600.0
MAX_TEARDOWN_INVENTORY_PROBES = 40

#: States that mean the instance is gone as far as billing is concerned.
_TERMINAL_INSTANCE_STATES = frozenset({"terminated", "shutting-down", "stopped"})


class ReconstructionAwsWindowsError(RuntimeError):
    """Stable fail-closed Windows operation error."""


def _blocker_list(*values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        if isinstance(value, str) and value:
            out.append(value)
        elif isinstance(value, (list, tuple)):
            out.extend(str(item) for item in value if item)
    return sorted(set(out))


def _inventory_count(inventory: Mapping[str, Any] | None) -> int | None:
    """Read a billable-instance count, or None when the answer is unknown.

    Unknown is deliberately distinct from zero: an inventory call that failed
    must not be recorded as proof that nothing is running.
    """

    if not isinstance(inventory, Mapping):
        return None
    for key in (
        "billable_instance_count",
        "live_resource_count",
        "count",
        "instances_running",
    ):
        value = inventory.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    instances = inventory.get("instances")
    if isinstance(instances, (list, tuple)):
        return len(instances)
    return None


def run_reconstruction_aws_windows_operation(
    *,
    bound_request: Mapping[str, Any],
    preflight: Mapping[str, Any],
    job_dir: str | Path,
    output_bundle_get_url: str,
    input_bundle_get_url: str | None = None,
    input_receipt_get_url: str | None = None,
    input_receipt_file_digest: str | None = None,
    output_bundle_put_url: str | None = None,
    progress_put_url: str | None = None,
    progress_get_url: str | None = None,
    progress_observer: Callable[[Mapping[str, Any]], None] | None = None,
    provider: Any,
    allocator_admission: Mapping[str, Any],
    paid_resource_admission_grant: Any,
    name_prefix: str,
    hard_ttl_seconds: int,
    output_fetcher: Callable[[str, Path], Any],
    output_validator: Callable[..., Mapping[str, Any]],
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Launch, watch, collect, then tear down exactly once."""

    root = Path(job_dir).expanduser().resolve()
    ensure_dir(root)
    started = clock()
    blockers: list[str] = []

    request_digest = str(bound_request.get("bound_request_digest") or "")
    admission_digest = str(allocator_admission.get("admission_digest") or "")
    if not admission_digest:
        raise ReconstructionAwsWindowsError(
            "aws_windows_operation_requires_allocator_admission"
        )
    if int(allocator_admission.get("retry_cap", -1)) != 0:
        raise ReconstructionAwsWindowsError("aws_windows_operation_retry_cap_must_be_zero")
    if not preflight:
        raise ReconstructionAwsWindowsError("aws_windows_operation_preflight_missing")

    # Provider-zero before allocation. A pre-existing billable instance means
    # some other run is live; allocating on top of it would double-spend.
    before = provider.billable_inventory(name_prefix=name_prefix)
    before_count = _inventory_count(before)
    if before_count is None:
        blockers.append("aws_windows_provider_zero_before_unverifiable")
    elif before_count > 0:
        write_json(root / "provider_zero_before.json", dict(before))
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "provider": PROVIDER,
            "blockers": _blocker_list("aws_windows_provider_not_zero_before_allocation"),
            "provider_mutations_performed": 0,
            "cost_usd": 0.0,
            "instance_id": None,
            "operation_scientific_success_inferred": False,
        }

    instance_id: str | None = None
    provider_mutations = 0
    output_retrieved_before_teardown = False
    validated: Mapping[str, Any] | None = None
    runtime_status: str | None = None
    teardown: dict[str, Any] = {"attempted": False, "confirmed": False}

    try:
        # The grant must reach the provider: its launch fails closed without
        # one, so omitting it would make every launch return "blocked" rather
        # than run.  Passing it also keeps the paid-lane admission check on the
        # provider side, where the other lanes enforce it.
        provider_request = dict(bound_request)
        canonical_postshot = (
            bound_request.get("execution_adapter_id")
            == "canonical_postshot_aws_windows_v1"
        )
        if canonical_postshot and hasattr(provider, "build_request"):
            from .gpu_render_providers import RenderLaunchSpec

            required_transport = {
                "BLUEPRINT_RECONSTRUCTION_INPUT_BUNDLE_GET_URL": input_bundle_get_url,
                "BLUEPRINT_RECONSTRUCTION_INPUT_RECEIPT_GET_URL": input_receipt_get_url,
                "BLUEPRINT_RECONSTRUCTION_INPUT_RECEIPT_FILE_DIGEST": input_receipt_file_digest,
                "BLUEPRINT_RECONSTRUCTION_OUTPUT_BUNDLE_PUT_URL": output_bundle_put_url,
            }
            missing = sorted(key for key, value in required_transport.items() if not value)
            if missing:
                raise ReconstructionAwsWindowsError(
                    "aws_windows_canonical_transport_missing:" + ",".join(missing)
                )
            worker_environment = {
                key: str(value) for key, value in required_transport.items()
            }
            if progress_put_url:
                worker_environment["BLUEPRINT_RECONSTRUCTION_PROGRESS_PUT_URL"] = progress_put_url
            worker_environment["BLUEPRINT_RECONSTRUCTION_CAPTURE_DIGEST"] = str(
                bound_request.get("source_capture_digest")
                or bound_request.get("capture_digest")
                or ""
            )
            worker_environment.update(
                {
                    "BLUEPRINT_RECONSTRUCTION_INPUT_BUNDLE_DIGEST": str(
                        bound_request.get("operation_input_bundle_digest") or ""
                    ),
                    "BLUEPRINT_CANONICAL_ALLOCATOR_ADMISSION_B64": base64.b64encode(
                        json.dumps(
                            dict(allocator_admission),
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).decode("ascii"),
                    "BLUEPRINT_CANONICAL_AUTHORITY_ID": str(
                        allocator_admission.get("authority_id") or ""
                    ),
                    "BLUEPRINT_CANONICAL_MAX_SPEND_USD": str(
                        allocator_admission.get("max_spend_usd") or ""
                    ),
                    "BLUEPRINT_CANONICAL_HARD_TTL_SECONDS": str(hard_ttl_seconds),
                    "BLUEPRINT_WORKER_HARD_TTL_SECONDS": str(hard_ttl_seconds),
                    "BLUEPRINT_WORKER_IMAGE_DIGEST": str(
                        bound_request.get("worker_image_digest") or ""
                    ),
                    "BLUEPRINT_SOURCE_COMMIT": str(
                        bound_request.get("source_commit_sha") or ""
                    ),
                }
            )
            for key in (
                "BLUEPRINT_POSTSHOT_LICENCE_GET_URL",
                "BLUEPRINT_POSTSHOT_LICENCE_DELETE_URL",
                "BLUEPRINT_POSTSHOT_RUNTIME_DIGEST",
                "BLUEPRINT_POSTSHOT_RUNTIME_VERSION",
                "BLUEPRINT_WINDOWS_NVIDIA_DRIVER_GET_URL",
                "BLUEPRINT_WINDOWS_POSTSHOT_INSTALLER_GET_URL",
                "BLUEPRINT_WINDOWS_POSTSHOT_INSTALLER_SHA256",
            ):
                value = os.environ.get(key)
                if value:
                    worker_environment[key] = value
            spec = RenderLaunchSpec(
                name=f"{name_prefix}-{request_digest[7:19]}",
                image=str(bound_request.get("worker_image_digest") or ""),
                env=worker_environment,
                bootstrap_argv=[],
                entrypoint=[],
                container_disk_gb=max(
                    100,
                    int(preflight.get("container_disk_bytes") or 0) // 1024**3,
                ),
                volume_gb=0,
                max_hourly_rate_usd=float(
                    preflight.get("on_demand_price_usd_per_hour") or 0
                ),
                min_gpu_ram_mb=24_000,
                requires_rtx=False,
            )
            provider_request = provider.build_request(spec, root)

        launch = provider.launch(
            root,
            provider_request,
            cold=True,
            paid_resource_admission_grant=paid_resource_admission_grant,
        )
        if str(launch.get("status") or "") == "blocked" or launch.get(
            "allocation_created"
        ) is False:
            write_json(root / "launch.json", dict(launch))
            blockers.extend(
                str(b) for b in (launch.get("blockers") or []) if str(b)
            )
            raise ReconstructionAwsWindowsError(
                "aws_windows_launch_blocked:" + ",".join(blockers[-4:])
            )
        provider_mutations = 1
        instance_id = str(launch.get("instance_id") or launch.get("id") or "") or None
        write_json(root / "launch.json", dict(launch))
        if instance_id is None:
            blockers.append("aws_windows_launch_returned_no_instance_id")
            raise ReconstructionAwsWindowsError("aws_windows_launch_returned_no_instance_id")

        deadline = started + float(hard_ttl_seconds)
        last_progress_digest = ""
        while True:
            state = provider.inspect(instance_id)
            runtime_status = str(
                state.get("state")
                or state.get("instance_status")
                or state.get("status")
                or ""
            )
            if progress_get_url and progress_observer is not None:
                try:
                    progress_path = root / "worker_progress.latest.json"
                    output_fetcher(progress_get_url, progress_path)
                    progress = json.loads(progress_path.read_text(encoding="utf-8"))
                    digest = str(progress.get("progress_digest") or "")
                    if digest and digest != last_progress_digest:
                        progress_observer(progress)
                        last_progress_digest = digest
                except Exception:  # noqa: BLE001 - absence means no new progress yet
                    pass
            if runtime_status in _TERMINAL_INSTANCE_STATES:
                # The instance ended on its own: either the work finished and
                # its local deadline fired, or it died. Which one is decided by
                # whether a valid output bundle exists, never by the exit alone.
                break
            if clock() >= deadline:
                blockers.append("aws_windows_hard_ttl_exceeded")
                break
            sleeper(POLL_INTERVAL_SECONDS)

        # Retrieve before teardown so a terminated instance can never be the
        # reason results are missing.
        destination = root / "output_bundle.zip"
        try:
            output_fetcher(output_bundle_get_url, destination)
            canonical_postshot = (
                bound_request.get("execution_adapter_id")
                == "canonical_postshot_aws_windows_v1"
            )
            if canonical_postshot:
                validated_value = output_validator(
                    bundle_path=destination,
                    expected_operation="trainer_canary",
                    expected_operation_request_digest=bound_request.get(
                        "operation_request_digest"
                    ),
                    expected_transport_bundle_digest=bound_request.get(
                        "operation_input_bundle_digest"
                    ),
                    expected_reconstruction_dataset_digest=bound_request.get(
                        "reconstruction_dataset_digest"
                    ),
                    expected_allocator_admission_digest=admission_digest,
                    expected_worker_image_digest=bound_request.get(
                        "worker_image_digest"
                    ),
                    expected_source_commit_sha=bound_request.get("source_commit_sha"),
                    expected_arm_id="postshot-primary",
                )
            else:
                validated_value = output_validator(
                    bundle_path=destination,
                    expected_operation="trainer_canary",
                    expected_operation_request_digest=bound_request.get(
                        "operation_request_digest"
                    ),
                    expected_worker_image_digest=bound_request.get(
                        "worker_image_digest"
                    ),
                    expected_source_commit_sha=bound_request.get("source_commit_sha"),
                )
            validated = (
                validated_value[0]
                if isinstance(validated_value, tuple)
                else validated_value
            )
            output_retrieved_before_teardown = True
        except Exception as exc:  # noqa: BLE001 - any retrieval failure is a blocker
            blockers.append(f"aws_windows_output_unavailable:{type(exc).__name__}")

    finally:
        if instance_id is not None:
            teardown["attempted"] = True
            try:
                provider.terminate(instance_id)
                teardown["confirmed"] = True
            except Exception as exc:  # noqa: BLE001
                blockers.append(f"aws_windows_teardown_failed:{type(exc).__name__}")
            after = None
            after_count = None
            teardown_deadline = clock() + min(
                MAX_TEARDOWN_WAIT_SECONDS,
                max(60.0, float(hard_ttl_seconds)),
            )
            for probe_index in range(MAX_TEARDOWN_INVENTORY_PROBES):
                try:
                    after = provider.billable_inventory(name_prefix=name_prefix)
                except Exception as exc:  # noqa: BLE001
                    blockers.append(
                        "aws_windows_provider_zero_after_unverifiable:"
                        f"{type(exc).__name__}"
                    )
                    break
                after_count = _inventory_count(after)
                if (
                    after_count in {0, None}
                    or probe_index == MAX_TEARDOWN_INVENTORY_PROBES - 1
                    or clock() >= teardown_deadline
                ):
                    break
                sleeper(TEARDOWN_POLL_INTERVAL_SECONDS)
            teardown["provider_zero_after_count"] = after_count
            teardown["provider_zero_after"] = after
            if after_count is None:
                blockers.append("aws_windows_provider_zero_after_unverifiable")
            elif after_count > 0:
                # Asking it to stop is not the same as it having stopped.
                blockers.append("aws_windows_provider_not_zero_after_teardown")
            write_json(root / "teardown.json", dict(teardown))

    duration = max(0.0, clock() - started)
    succeeded = validated is not None and not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if succeeded else "failed",
        "provider": PROVIDER,
        "operation": "trainer_canary",
        "request_digest": request_digest,
        "canonical_allocator_admission_digest": admission_digest,
        "instance_id": instance_id,
        "operation_result_status": runtime_status,
        "output_retrieved_before_teardown": output_retrieved_before_teardown,
        "output_bundle_receipt_digest": (
            validated.get("output_bundle_receipt_digest") if validated else None
        ),
        "teardown": teardown,
        "duration_seconds": duration,
        "cost_usd": 0.0,
        "provider_mutations_performed": provider_mutations,
        # A finished run is not a good run. Appearance, metric, collision and
        # task claims are decided by their own gates, never here.
        "operation_scientific_success_inferred": False,
        "blockers": _blocker_list(*blockers),
    }


__all__ = [
    "PROVIDER",
    "SCHEMA_VERSION",
    "ReconstructionAwsWindowsError",
    "run_reconstruction_aws_windows_operation",
]
