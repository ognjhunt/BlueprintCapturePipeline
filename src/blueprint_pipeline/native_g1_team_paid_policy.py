"""Canonical allocation boundary for one approved selected G1 policy episode.

HTTP intake never calls this worker. The canonical paid allocator supplies its
verified release identity. This controller reuses spend admission, launch
coordination, independent watchdog and provider teardown, then reopens native
score/media bytes. Official billing and owner review delivery are later steps.
"""

from __future__ import annotations

import json
import math
import os
import zipfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

from .adp_isaac_lab_arena_vast import run_arena_native_control_vast
from .common import write_json
from .control_plane_disk_budget import (
    DEFAULT_RESERVATION_ROOT, ControlPlaneDiskBudgetError, disk_headroom,
    reserve_control_plane_disk,
)
from .control_plane_disk_reservation_heartbeat import keep_reservation_live
from .decision_evidence_contracts import canonical_digest
from .native_g1_paid_campaign import (
    _controller_release_authority, _early_spend_lock_blockers,
    _early_provider_credit_blockers, _hold_pre_stage_launch_gate,
)
from .native_g1_provider_bundle import _sha256
from .native_g1_team_dispatch_preflight import verify_g1_team_dispatch_inputs
from .native_g1_team_paid_output import verify_g1_team_paid_output
from .native_g1_team_provider_bundle import PACKET_RELATIVE_PATH, PROVIDER_BUNDLE_KIND, RESULT_FILENAME
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION, build_paid_lane_admission,
    require_paid_resource_admission, PaidResourceAdmissionBlocked,
)


PROBE_KIND = "native-g1-team-policy"
RESULT_SCHEMA = "native_g1_team_paid_policy_result.v1"
CONSUMPTION_FILENAME = "native_g1_team_paid_attempt_consumption.v1.json"
INSTANCE_LABEL_PREFIX = "blueprint-native-task-arena-g1-team-"
PROVIDER_INPUT_FORECAST_BYTES = 12_000_000_000
PROVIDER_OUTPUT_FORECAST_BYTES = 10_000_000_000
COLLECTION_FORECAST_BYTES = PROVIDER_INPUT_FORECAST_BYTES + 2 * PROVIDER_OUTPUT_FORECAST_BYTES


def add_g1_team_policy_allocator_arguments(parser: Any) -> None:
    for name in ("bundle-receipt", "intent", "registry", "approval", "credential-registry"):
        parser.add_argument("--g1-team-" + name)
    parser.add_argument("--g1-team-trusted-client", action="append", default=[])


def dispatch_g1_team_policy_allocator_cli(args: Any, *, control_recheck: Callable) -> int:
    blockers, identity = control_recheck()
    result = dispatch_g1_team_paid_policy(
        args, control_identity=identity, control_blockers=blockers,
        control_recheck=control_recheck,
    )
    success = result.get("status") in {"dry_run_ready", "completed"}
    print(json.dumps({"success": success}, sort_keys=True))
    return 0 if success else 2


def _verify_output(result: dict[str, Any], bundle: dict[str, Any], *, job: Path) -> dict[str, Any]:
    """Reopen immutable selected inputs and actual native episode media."""
    archive_path = Path(bundle["bundle_path"])
    if archive_path.is_symlink() or _sha256(archive_path) != bundle["bundle_sha256"]:
        raise ValueError("g1_team_paid_bundle_changed_after_run")
    with zipfile.ZipFile(archive_path) as archive:
        packet = json.loads(archive.read(PACKET_RELATIVE_PATH))
    attempt = Path(str(result.get("attempt_root") or ""))
    if (not attempt.is_absolute() or attempt.is_symlink()
            or attempt.resolve() != attempt
            or attempt.parent != job / "attempts"
            or not attempt.name.startswith("attempt_")):
        raise ValueError("g1_team_paid_output_path_invalid")
    root = attempt / "immutable_execution"
    if packet["request"]["policy_profile"]["delivery"]["mode"] != "authenticated_endpoint":
        # A guest score cannot stand in for policy-host/relay/container closure.
        # Keep this mandatory even while paired launch admission is refused.
        from .native_g1_team_vm_bundle_support import BOOTSTRAP_RESULT_FILENAME
        from .native_g1_team_vm_output import _receipt, verify_g1_team_vm_host_output
        bootstrap = _receipt(root / BOOTSTRAP_RESULT_FILENAME, digest_field="receipt_digest")
        if (bootstrap.get("schema_version") != "native_g1_team_vm_bootstrap_entrypoint.v1"
                or bootstrap.get("status") != "host_exited"
                or bootstrap.get("stage_reached") != "vm-host"
                or type(bootstrap.get("runner_exit_code")) is not int
                or bootstrap["runner_exit_code"] != 0
                or bootstrap.get("implementation_commit") != bundle["implementation_commit"]
                or bootstrap.get("provider_mutation_performed") is not False
                or bootstrap.get("gpu_runtime_qualified") is not False
                or bootstrap.get("claim_ceiling") != "development_only"):
            raise ValueError("g1_team_paid_vm_bootstrap_result_invalid")
        verified = verify_g1_team_vm_host_output(
            output_dir=root, execution_packet=packet,
            scene_plan_digest=bundle["scene_plan_digest"],
            scene_packet_receipt_digest=bundle["scene_packet_receipt_digest"],
        )
        return {
            **verified["verified_output"],
            "isolated_policy_host_verification": {
                **verified, "bootstrap_entrypoint_digest": bootstrap["receipt_digest"],
            },
        }
    path = root / RESULT_FILENAME
    if path.is_symlink() or not path.is_file():
        raise ValueError("g1_team_paid_native_result_missing")
    native = json.loads(path.read_text())
    if (native.get("schema_version") != "native_g1_team_provider_result.v1"
            or native.get("status") != "completed_development_only"
            or native.get("result_digest") != canonical_digest(native, digest_field="result_digest")
            or native.get("execution_packet_digest") != packet["packet_digest"]
            or native.get("worker_output_relative_path") != "selected-worker/worker"
            or native.get("candidate_policy_queried") is not True
            or native.get("public_redistribution_authorized") is not False):
        raise ValueError("g1_team_paid_native_result_invalid")
    verified = verify_g1_team_paid_output(
        output_dir=root / "selected-worker/worker", execution_packet=packet,
        scene_plan_digest=bundle["scene_plan_digest"],
        scene_packet_receipt_digest=bundle["scene_packet_receipt_digest"],
    )
    if native.get("verified_output") != verified:
        raise ValueError("g1_team_paid_native_verification_changed")
    return verified


def dispatch_g1_team_paid_policy(
    args: Any, *, control_identity: dict[str, Any], control_blockers: list[str],
    control_recheck: Callable[[], tuple[list[str], dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Run one selected episode only through the canonical paid adapter."""
    blockers = list(control_blockers)
    commit = str(control_identity.get("orchestrator_source_commit") or "")
    release = _controller_release_authority(commit, control_identity)
    if release is None:
        blockers.append("g1_team_paid_controller_not_exact_release")
    if args.provider != "vast":
        blockers.append("g1_team_paid_provider_must_be_vast")
    required = (
        "g1_team_bundle_receipt", "g1_team_intent", "g1_team_registry",
        "g1_team_approval", "g1_team_trusted_client", "adp_job_dir",
        "admission_out", "adapter_output",
    )
    if any(not getattr(args, name, None) for name in required):
        blockers.append("g1_team_paid_inputs_missing")
    rate, cap, ttl = args.adp_max_hourly_rate_usd, args.adp_max_spend_usd, args.adp_hard_ttl_seconds
    if (type(rate) not in (int, float) or not math.isfinite(rate) or not 0 < rate <= 5
            or type(cap) not in (int, float) or not math.isfinite(cap) or not 0 < cap <= 12
            or type(ttl) is not int or not 1800 <= ttl <= 14400):
        blockers.append("g1_team_paid_budget_invalid")
    if any(type(value) is not int or value <= 0 for value in args.adp_allowed_active_vast_instance_id):
        blockers.append("g1_team_paid_allowed_active_instance_invalid")
    job = Path(args.adp_job_dir) if args.adp_job_dir else None
    if job is not None and (not job.is_absolute() or job.is_symlink()):
        blockers.append("g1_team_paid_job_path_invalid")
    if job is not None and (job / CONSUMPTION_FILENAME).exists():
        blockers.append("g1_team_paid_attempt_already_consumed")
    if args.execute and not blockers:
        blockers.extend(_early_spend_lock_blockers())
        if not blockers:
            blockers.extend(_early_provider_credit_blockers(cap))
    gate = None
    if args.execute and not blockers:
        gate, gate_blocker = _hold_pre_stage_launch_gate()
        if gate_blocker:
            blockers.append(gate_blocker)
    try:
        return _dispatch(
            args, job=job, commit=commit, release=release, blockers=blockers,
            identity=control_identity, control_recheck=control_recheck, gate=gate,
        )
    finally:
        if gate is not None:
            gate.release()


def _dispatch(args, *, job, commit, release, blockers, identity, control_recheck, gate):
    result = None
    try:
        with ExitStack() as stack:
            result = _dispatch_selected(
                args, job=job, commit=commit, release=release, blockers=blockers,
                identity=identity, control_recheck=control_recheck, gate=gate, stack=stack,
            )
    except ControlPlaneDiskBudgetError:
        # Renewal can fail while the adapter owns a paid resource, or surface
        # only during context exit. Preserve its exact closeout if it returned;
        # an unreturned adapter must be reconciled, never asserted provider-zero.
        if result is None:
            result = {"schema_version": RESULT_SCHEMA,
                      "provider_mutation_status": "unproven_reconcile_exact_attempt",
                      "provider_teardown_verified": False, "claim_ceiling": "development_only",
                      "public_redistribution_authorized": False}
        result.update(status="blocked", blockers=[*result.get("blockers", []),
                                                  "g1_team_paid_collection_reservation_lost"])
        if args.adapter_output:
            write_json(Path(args.adapter_output), result)
    return result


def _dispatch_selected(args, *, job, commit, release, blockers, identity, control_recheck, gate, stack):
    selected = None
    health = None
    synthetic = {"status": "not_checked_static_dry" if not args.execute else "not_checked_blocked"}
    collection = {"status": "not_checked", "forecast_bytes": COLLECTION_FORECAST_BYTES,
                  "forecast_is_output_upper_bound": False}
    if not blockers:
        try:
            selected = verify_g1_team_dispatch_inputs(
                bundle_receipt_path=Path(args.g1_team_bundle_receipt),
                authority_arguments={
                    "intent_path": Path(args.g1_team_intent), "registry_path": Path(args.g1_team_registry),
                    "approval_path": Path(args.g1_team_approval), "trusted_clients": set(args.g1_team_trusted_client),
                },
                expected_implementation_commit=commit,
                credential_registry_path=(Path(args.g1_team_credential_registry) if args.g1_team_credential_registry else None),
                max_hourly_rate_usd=args.adp_max_hourly_rate_usd,
                hard_cap_usd=args.adp_max_spend_usd, hard_ttl_seconds=args.adp_hard_ttl_seconds,
            )
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
            blockers.append("g1_team_paid_input_preflight_failed")
    if not blockers:
        ledger = os.getenv("BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT", str(DEFAULT_RESERVATION_ROOT))
        try:
            headroom = disk_headroom(target_root=job, reservation_root=ledger)
            collection.update(status="observed", available_bytes=headroom["available_bytes"])
            if headroom["available_bytes"] < COLLECTION_FORECAST_BYTES:
                blockers.append("g1_team_paid_collection_capacity_insufficient")
                collection["status"] = "blocked"
            elif args.execute:
                reservation = reserve_control_plane_disk(
                    "policy_canary_dispatch", target_root=job, reservation_root=ledger,
                    expected_bytes=COLLECTION_FORECAST_BYTES,
                    ttl_seconds=args.adp_hard_ttl_seconds + 3600,
                )
                stack.enter_context(reservation)
                health = stack.enter_context(keep_reservation_live(reservation))
                collection.update(status="reserved", reservation=reservation.receipt())
        except (OSError, KeyError, ControlPlaneDiskBudgetError):
            blockers.append("g1_team_paid_collection_admission_failed")
            collection["status"] = "blocked"
    if args.execute and not blockers and selected is not None:
        try:
            if health is not None:
                health.check()
            synthetic = selected.probe_synthetic_endpoint()
        except (OSError, ValueError, ControlPlaneDiskBudgetError):
            blockers.append("g1_team_paid_endpoint_synthetic_preflight_failed")
            synthetic = {"status": "blocked_before_allocation"}
    binding = {
        "program_id": "arm-decision-proof-v1", "probe_kind": PROBE_KIND,
        "provider": "vast", "orchestrator_source_commit": commit,
        "release_authority": release,
        "selected_inputs": selected.safe_receipt() if selected else None,
        "collection_capacity": collection,
        "endpoint_preallocation_conformance": synthetic,
        "retry_cap": 0,
    }
    binding_digest = canonical_digest(binding)
    admission = build_paid_lane_admission(resource_class="vast_provider_adapter", blockers=blockers)
    admission.update({
        "probe_kind": PROBE_KIND, "control_plane_identity": identity,
        "allocation_binding": binding, "allocation_binding_digest": binding_digest,
        "claim_ceiling": "development_only", "retry_cap": 0,
    })
    if args.admission_out:
        write_json(Path(args.admission_out), admission)
    result = {"schema_version": RESULT_SCHEMA, "status": "blocked", "blockers": blockers,
              "provider_mutations_performed": 0}
    if not blockers and selected is not None:
        assert job is not None
        job.mkdir(parents=True, exist_ok=True, mode=0o750)
        grant = None
        transport_entered = False
        try:
            if args.execute:
                grant = require_paid_resource_admission(
                    admission, resource_class="vast_provider_adapter",
                    expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                )
            def before_create():
                if health is not None:
                    try:
                        health.check()
                    except ControlPlaneDiskBudgetError:
                        return {"status": "blocked", "blockers": ["g1_team_paid_collection_reservation_lost"]}
                if control_recheck is not None:
                    fresh_blockers, fresh_identity = control_recheck()
                    if (fresh_blockers or fresh_identity.get("orchestrator_source_commit") != commit
                            or _controller_release_authority(commit, fresh_identity) != release):
                        return {"status": "blocked", "blockers": ["g1_team_paid_controller_changed_before_create"]}
                try:
                    selected.recheck()
                except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
                    return {"status": "blocked", "blockers": ["g1_team_paid_inputs_changed_before_create"]}
                consumed = {
                    "schema_version": "native_g1_team_paid_attempt_consumption.v1", "status": "consumed",
                    "allocation_binding_digest": binding_digest, "orchestrator_source_commit": commit,
                    "execution_packet_digest": selected.bundle["execution_packet_digest"], "retry_cap": 0,
                }
                try:
                    with (job / CONSUMPTION_FILENAME).open("x") as stream:
                        json.dump(consumed, stream, indent=2, sort_keys=True)
                        stream.write("\n")
                except FileExistsError:
                    return {"status": "blocked", "blockers": ["g1_team_paid_attempt_already_consumed"]}
                if gate is not None:
                    gate.release()
                return consumed
            # The ZIP manifest remains sealed_not_admitted. Ready here denotes
            # the verified transport projection, bound to the paid grant above.
            transport = {**selected.bundle, "status": "ready", "allocation_binding_digest": binding_digest}
            transport_entered = True
            result = run_arena_native_control_vast(
                approval_path=args.g1_team_approval, job_dir=job,
                prepared_bundle=transport, paid_resource_admission_grant=grant, execute=args.execute,
                pre_provider_mutation_hook=before_create if args.execute else None,
                runtime_secret_file_paths=selected.runtime_secret_file_paths(),
                machine_avoidlist_path=args.adp_machine_avoidlist,
                max_hourly_rate_usd=args.adp_max_hourly_rate_usd, hard_cap_usd=args.adp_max_spend_usd,
                hard_ttl_seconds=args.adp_hard_ttl_seconds,
                expected_output_filename=RESULT_FILENAME, container_image=NATIVE_TASK_ARENA_IMAGE,
                provider_bundle_kind=PROVIDER_BUNDLE_KIND, result_schema_version=RESULT_SCHEMA,
                instance_label_prefix=INSTANCE_LABEL_PREFIX, blocker_prefix="native_g1_team_policy",
                min_gpu_ram_mb=48_000, candidate_policy_query_expected=True,
                require_independent_watchdog=True, allowed_active_instance_ids=args.adp_allowed_active_vast_instance_id,
                expected_provider_download_bytes=PROVIDER_INPUT_FORECAST_BYTES,
                expected_provider_upload_bytes=PROVIDER_OUTPUT_FORECAST_BYTES,
            )
            if args.execute and result.get("status") == "completed":
                result["g1_team_output_verification"] = _verify_output(result, dict(selected.bundle), job=job)
                result["official_billing_reconciled"] = False
                result["private_review_status"] = "pending_verified_ingest"
        except PaidResourceAdmissionBlocked as exc:
            result = {"schema_version": RESULT_SCHEMA, "status": "blocked", "blockers": exc.blockers,
                      "provider_mutations_performed": 0}
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            # Preserve adapter closeout if verification failed after teardown.
            if transport_entered and args.execute and result.get("provider_mutations_performed") == 0:
                result.pop("provider_mutations_performed")
                result["provider_mutation_status"] = "unproven_reconcile_exact_attempt"
                result["provider_teardown_verified"] = False
            result.update({"status": "blocked", "blockers": [*result.get("blockers", []),
                           "g1_team_paid_transport_or_output_failed:" + type(exc).__name__]})
    result.update({"allocation_binding_digest": binding_digest, "claim_ceiling": "development_only",
                   "collection_capacity": collection,
                   "endpoint_preallocation_conformance": synthetic,
                   "launch_readiness": "static_transport_only" if not args.execute else "paid_admission_required",
                   "public_redistribution_authorized": False})
    if args.adapter_output:
        write_json(Path(args.adapter_output), result)
    return result
