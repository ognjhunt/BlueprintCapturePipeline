"""Canonical paid admission for one sealed G1 beta session; no retry launcher."""
from __future__ import annotations

from contextlib import ExitStack
from decimal import Decimal
import json
import os
from pathlib import Path
import zipfile

from .adp_isaac_lab_arena_vast import run_arena_native_control_vast
from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .native_g1_beta_bundle import load_verified_bundle
from .native_g1_beta_session import (MAX_OUTPUT_BYTES, PROBE_KIND, PROVIDER_KIND, RESOURCE_PREFIX,
    RESULT_FILENAME, consume_session_authority_once, validate_session_authority)
from .native_g1_paid_campaign import (_controller_release_authority, _early_spend_lock_blockers,
    _early_provider_credit_blockers, _hold_pre_stage_launch_gate)
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
from .paid_resource_admission import (PAID_LANE_ADMISSION_SCHEMA_VERSION, build_paid_lane_admission,
    require_paid_resource_admission, PaidResourceAdmissionBlocked)

RESULT_SCHEMA = "native_g1_beta_paid_session_result.v1"


def _read(path):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("g1_beta_paid_input_path_invalid")
    return json.loads(path.read_text())


def _inputs(args, commit):
    authority = validate_session_authority(_read(args.g1_beta_session_authority))
    bundle = load_verified_bundle(Path(args.g1_beta_session_bundle_receipt), expected_implementation_commit=commit)
    if (bundle["authority_digest"] != authority["authority_digest"]
            or authority["implementation_commit"] != commit
            or Decimal(str(args.adp_max_hourly_rate_usd)) != Decimal(authority["max_hourly_rate_usd"])
            or Decimal(str(args.adp_max_spend_usd)) != Decimal(str(authority["hard_cap_usd"]))
            or args.adp_hard_ttl_seconds != authority["hard_ttl_seconds"]):
        raise ValueError("g1_beta_paid_bound_values_changed")
    with zipfile.ZipFile(bundle["bundle_path"]) as archive:
        if json.loads(archive.read("provider_runtime/inputs/session_authority.json")) != authority:
            raise ValueError("g1_beta_paid_archive_authority_changed")
    return authority, bundle


def dispatch_g1_beta_allocator_cli(args, *, control_recheck):
    blockers, identity = control_recheck()
    result = dispatch_session(args, control_identity=identity, control_blockers=blockers, control_recheck=control_recheck)
    print(json.dumps({"status": result.get("status")}))
    return 0 if result.get("status") in {"completed", "dry_run_ready"} else 2


def dispatch_session(args, *, control_identity, control_blockers, control_recheck):
    from .control_plane_disk_budget import (DEFAULT_RESERVATION_ROOT, ControlPlaneDiskBudgetError,
        disk_headroom, reserve_control_plane_disk)
    from .control_plane_disk_reservation_heartbeat import keep_reservation_live
    blockers = list(control_blockers)
    commit = str(control_identity.get("orchestrator_source_commit") or "")
    release = _controller_release_authority(commit, control_identity)
    if release is None:
        blockers.append("g1_beta_paid_exact_release_missing")
    if args.provider != "vast":
        blockers.append("g1_beta_paid_provider_invalid")
    if any(not getattr(args, name, None) for name in ("g1_beta_session_authority", "g1_beta_session_bundle_receipt",
            "adp_job_dir", "admission_out", "adapter_output")):
        blockers.append("g1_beta_paid_inputs_missing")
    job = Path(args.adp_job_dir) if args.adp_job_dir else None
    if job is None or not job.is_absolute() or job.is_symlink() or job.resolve() != job:
        blockers.append("g1_beta_paid_job_invalid")
    if job is not None and (job / "authorization_consumption.json").exists():
        blockers.append("g1_beta_paid_authority_already_consumed")
    authority, bundle = None, None
    if not blockers:
        try:
            authority, bundle = _inputs(args, commit)
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
            blockers.append("g1_beta_paid_input_verification_failed")
    gate = None
    if args.execute and not blockers:
        blockers.extend(_early_spend_lock_blockers())
        if not blockers:
            blockers.extend(_early_provider_credit_blockers(authority["hard_cap_usd"]))
        if not blockers:
            gate, blocked = _hold_pre_stage_launch_gate()
            if blocked:
                blockers.append(blocked)
    result = {"schema_version": RESULT_SCHEMA, "status": "blocked", "blockers": blockers,
              "provider_mutations_performed": 0, "claim_ceiling": "development_only"}
    entered = False
    try:
        with ExitStack() as stack:
            collection = {"status": "not_checked"}
            health = None
            if not blockers:
                amount = 4 * MAX_OUTPUT_BYTES + bundle["bundle_size_bytes"]
                ledger = os.getenv("BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT", str(DEFAULT_RESERVATION_ROOT))
                headroom = disk_headroom(target_root=job, reservation_root=ledger)
                collection = {"status": "observed", "expected_bytes": amount, "available_bytes": headroom["available_bytes"]}
                if headroom["available_bytes"] < amount:
                    blockers.append("g1_beta_collection_capacity_insufficient")
                elif args.execute:
                    reservation = reserve_control_plane_disk("policy_canary_dispatch", target_root=job,
                        reservation_root=ledger, expected_bytes=amount, ttl_seconds=authority["hard_ttl_seconds"] + 3600)
                    stack.enter_context(reservation)
                    health = stack.enter_context(keep_reservation_live(reservation))
                    collection.update(status="reserved", reservation=reservation.receipt())
            binding = {"program_id": "blueprint_beta_policy_screening", "probe_kind": PROBE_KIND,
                "provider": "vast", "orchestrator_source_commit": commit, "release_authority": release,
                "authority_digest": authority["authority_digest"] if authority else None,
                "bundle_sha256": bundle["bundle_sha256"] if bundle else None,
                "transfer_quote": bundle["transfer_quote"] if bundle else None,
                "collection_capacity": collection, "retry_cap": 0}
            admission = build_paid_lane_admission(resource_class="vast_provider_adapter", blockers=blockers)
            admission.update(probe_kind=PROBE_KIND, control_plane_identity=control_identity,
                allocation_binding=binding, allocation_binding_digest=canonical_digest(binding), claim_ceiling="development_only")
            if args.admission_out:
                write_json(Path(args.admission_out), admission)
            if not blockers:
                job.mkdir(parents=True, exist_ok=True)
                grant, consumed = None, None
                if args.execute:
                    grant = require_paid_resource_admission(admission, resource_class="vast_provider_adapter",
                        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
                    # This receipt fixes the absolute lifetime before transport.
                    # A crash or uncertain create can never get another deadline.
                    consumed = consume_session_authority_once(authority, consumption_path=job / "authorization_consumption.json")
                    if consumed["status"] != "consumed":
                        raise ValueError("g1_beta_paid_authority_already_consumed")
                create_checked = False
                def before_create():
                    nonlocal create_checked
                    if create_checked:
                        return {"status": "blocked", "blockers": ["g1_beta_paid_create_already_checked"]}
                    create_checked = True
                    try:
                        if health is not None:
                            health.check()
                        fresh_blockers, fresh_identity = control_recheck()
                        if fresh_blockers or _controller_release_authority(commit, fresh_identity) != release:
                            raise ValueError("g1_beta_paid_release_changed")
                        if _inputs(args, commit) != (authority, bundle):
                            raise ValueError("g1_beta_paid_inputs_changed")
                        from .beta_policy_screening_worker import require_current_source
                        admission_binding = authority["beta_screening_admission"]
                        require_current_source(admission_binding["source"], site_id=admission_binding["site_id"])
                    except (OSError, ValueError, TypeError, KeyError, ControlPlaneDiskBudgetError):
                        return {"status": "blocked", "blockers": ["g1_beta_paid_precreate_recheck_failed"]}
                    if gate is not None:
                        gate.release()
                    return consumed
                entered = True
                result = run_arena_native_control_vast(approval_path=args.g1_beta_session_authority, job_dir=job,
                    prepared_bundle=bundle, paid_resource_admission_grant=grant, execute=args.execute,
                    pre_provider_mutation_hook=before_create if args.execute else None,
                    machine_avoidlist_path=args.adp_machine_avoidlist,
                    max_hourly_rate_usd=float(authority["max_hourly_rate_usd"]), hard_cap_usd=authority["hard_cap_usd"],
                    hard_ttl_seconds=authority["hard_ttl_seconds"], expected_output_filename=RESULT_FILENAME,
                    container_image=NATIVE_TASK_ARENA_IMAGE, provider_bundle_kind=PROVIDER_KIND,
                    result_schema_version=RESULT_SCHEMA, instance_label_prefix=RESOURCE_PREFIX,
                    instance_label_exact=authority["resource_name"], blocker_prefix="native_g1_beta",
                    min_gpu_ram_mb=48_000, candidate_policy_query_expected=True, require_independent_watchdog=True,
                    allowed_active_instance_ids=args.adp_allowed_active_vast_instance_id,
                    expected_provider_download_bytes=bundle["transfer_quote"]["expected_provider_download_bytes"],
                    expected_provider_upload_bytes=bundle["transfer_quote"]["expected_provider_upload_bytes"],
                    hard_deadline_epoch=consumed["hard_deadline_epoch"] if consumed else None,
                    authorization_consumption=consumed, stale_offer_create_retry_limit=0)
                if args.execute and result.get("native_control_result_path"):
                    from .native_g1_beta_output import verify_output
                    path = Path(result["native_control_result_path"])
                    if not path.is_absolute() or not path.resolve().is_relative_to(job.resolve()):
                        raise ValueError("g1_beta_paid_foreign_output_path")
                    native = verify_output(output_dir=path.parent, bundle=bundle)
                    if native["source_result_digest"] != result.get("native_control_result_digest"):
                        raise ValueError("g1_beta_paid_output_digest_changed")
                    result["native_g1_beta_output_verified"] = True
            result.update(allocation_binding_digest=canonical_digest(binding), collection_capacity=collection)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, PaidResourceAdmissionBlocked, ControlPlaneDiskBudgetError) as exc:
        if entered and args.execute and result.get("provider_mutations_performed") == 0:
            result.pop("provider_mutations_performed", None)
            result.update(provider_mutation_status="unproven_reconcile_exact_attempt", provider_teardown_verified=False)
        result.update(status="blocked", blockers=[*result.get("blockers", []), "g1_beta_paid_dispatch_failed:" + type(exc).__name__])
    finally:
        if gate is not None:
            gate.release()
    result.update(claim_ceiling="development_only", public_media_publication_authorized=False,
                  model_redistribution_authorized=False, physical_outcome_claimed=False)
    if args.adapter_output:
        write_json(Path(args.adapter_output), result)
    return result
