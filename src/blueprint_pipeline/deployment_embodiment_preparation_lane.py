"""Canonical allocator admission for one explicit operator-funded asset batch."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

from .decision_evidence_contracts import canonical_digest
from .deployment_embodiment_preparation_bundle import load_verified_bundle
from .deployment_embodiment_preparation_contract import (
    BUNDLE_KIND, INSTANCE_PREFIX, MAX_OUTPUT_BYTES, PROBE_KIND, RESULT_FILENAME, STATE_ROOT,
    read, require, resource_name, sealed, validate_request,
)
from .paid_resource_admission import (PAID_LANE_ADMISSION_SCHEMA_VERSION, PaidResourceAdmissionBlocked,
    build_paid_lane_admission, require_paid_resource_admission)
from .spend_authority_consumption_root import prepare_consumption_root

# Conservative transfer allowance for the pinned Isaac image, cold setup and
# provider pull overhead; not a measured image size. The adapter prices this,
# the complete input bundle, output cap and requested disk before creating.
COLD_IMAGE_SETUP_DOWNLOAD_BYTES = 64 * 1024**3


def consumption_path(request: dict) -> Path:
    identity = hashlib.sha256(request["authorization_id"].encode()).hexdigest()
    return prepare_consumption_root() / ("embodiment-preparation-" + identity + ".json")


def consume_once(request: dict, bundle: dict, job: Path) -> dict:
    """Claim the stable authorization ID durably before any provider create."""
    value = sealed({"schema_version": "embodiment_asset_preparation_consumption.v1", "status": "consumed",
        "authorization_id": request["authorization_id"], "request_digest": request["request_digest"],
        "bundle_sha256": bundle["bundle_sha256"], "inputs_digest": bundle["inputs_digest"],
        "resource_name": resource_name(request), "allocator_job": str(job.resolve()),
        "max_spend_usd": request["max_spend_usd"], "consumed_at_epoch": time.time()}, "consumption_digest")
    path = consumption_path(request)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        raise ValueError("embodiment_preparation_authority_already_consumed") from None
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_DIRECTORY | os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return value


def dispatch(args, control_context, *, transport=None, trusted_root: Path = STATE_ROOT) -> int:
    from .common import write_json
    from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_MINIMUM_DRIVER_VERSION
    from .rigid_robot_free_probe_lane import driver_ceiling_blockers
    from .vast_independent_watchdog_control import validate_independent_vast_watchdog_names
    from .adp_isaac_lab_arena_vast import run_arena_native_control_vast

    required = ("embodiment_preparation_request", "embodiment_preparation_bundle", "embodiment_preparation_job_dir",
                "admission_out", "adapter_output")
    require(all(getattr(args, key, None) for key in required), "allocator_arguments_missing")
    blockers, identity = control_context
    blockers = list(blockers)
    request = bundle = None
    job = Path(args.embodiment_preparation_job_dir)
    try:
        require(args.provider == "vast", "provider_invalid")
        request = validate_request(read(Path(args.embodiment_preparation_request)))
        trusted = read(trusted_root / "authorizations" / (request["authorization_id"] + ".json"))
        require(trusted == request, "operator_authority_missing")
        commit = identity.get("orchestrator_source_commit")
        require(request["release_commit"] == commit, "installed_release_mismatch")
        bundle = load_verified_bundle(args.embodiment_preparation_bundle, expected_implementation_commit=commit)
        require(bundle["request_digest"] == request["request_digest"], "authority_bundle_mismatch")
        require(not consumption_path(request).exists(), "authority_already_consumed")
        validate_independent_vast_watchdog_names(pod_name_prefix=INSTANCE_PREFIX, resource_name_exact=resource_name(request))
        blockers.extend(driver_ceiling_blockers())
    except (OSError, ValueError, KeyError) as exc:
        blockers.append(str(exc))
    binding = {"program_id": "blueprint-free-beta-evaluated-pilot", "probe_kind": PROBE_KIND,
        "request_digest": request["request_digest"] if request else None,
        "bundle_sha256": bundle["bundle_sha256"] if bundle else None,
        "inputs_digest": bundle["inputs_digest"] if bundle else None,
        "resource_name": resource_name(request) if request else None,
        "maximum_provider_allocations": 1, "retry_cap": 0, "candidate_policy_queried": False}
    admission = build_paid_lane_admission(resource_class="vast_provider_adapter", blockers=blockers)
    admission.update(allocation_binding=binding, allocation_binding_digest=canonical_digest(binding),
                     spend_authority="explicit_operator_asset_preparation_grant", control_plane_identity=dict(identity))
    write_json(Path(args.admission_out), admission)
    grant = None
    if args.execute and not blockers:
        try:
            grant = require_paid_resource_admission(admission, resource_class="vast_provider_adapter",
                                                    expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
        except PaidResourceAdmissionBlocked as exc:
            blockers.extend(exc.blockers)
    if blockers or request is None or bundle is None:
        result = {"schema_version": "embodiment_asset_preparation_vast_run.v1", "status": "blocked",
                  "blockers": blockers, "provider_mutations_performed": 0, "continuing_spend_from_this_run": False}
    else:
        consumption = consume_once(request, bundle, job) if args.execute else None
        result = (transport or run_arena_native_control_vast)(
            approval_path=".", job_dir=job, paid_resource_admission_grant=grant, execute=args.execute,
            prepared_bundle=bundle, max_hourly_rate_usd=request["max_hourly_rate_usd"],
            hard_cap_usd=request["max_spend_usd"], hard_ttl_seconds=request["ttl_seconds"],
            expected_output_filename=RESULT_FILENAME, container_image=bundle["container_image"],
            provider_bundle_kind=BUNDLE_KIND, result_schema_version="embodiment_asset_preparation_vast_run.v1",
            object_store_key_prefix="blueprint/robot-asset-preparation", instance_label_prefix=INSTANCE_PREFIX,
            instance_label_exact=resource_name(request), blocker_prefix=BUNDLE_KIND, min_gpu_ram_mb=24_000,
            minimum_driver_version=NATIVE_TASK_ARENA_MINIMUM_DRIVER_VERSION,
            require_known_supported_isaac_driver=True, require_independent_watchdog=True,
            hard_deadline_epoch=request["expires_at_epoch"], authorization_consumption=consumption,
            stale_offer_create_retry_limit=0, candidate_policy_query_expected=False,
            expected_provider_download_bytes=bundle["bundle_size_bytes"] + COLD_IMAGE_SETUP_DOWNLOAD_BYTES,
            expected_provider_upload_bytes=MAX_OUTPUT_BYTES, allowed_geolocation_country_codes=("US",))
    write_json(Path(args.adapter_output), result)
    return 0 if result.get("status") in {"completed", "dry_run_ready"} else 2
