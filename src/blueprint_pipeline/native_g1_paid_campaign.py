"""Controller admission and terminal evidence for one paid G1 campaign."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Callable

from .adp_isaac_lab_arena_vast import run_arena_native_control_vast
from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .episode_visual_evidence import validate_multicamera_frame_manifest
from .native_g1_development_pair import (
    EPISODE_FILENAME,
    PAIR_ORDER,
    TRACE_FILENAME,
    _score_from_episode,
    _verified_review_media,
)
from .native_g1_provider_bundle import (
    PROVIDER_BUNDLE_KIND,
    build_g1_provider_bundle,
    load_verified_g1_provider_bundle,
)
from .native_g1_provider_runtime import RESULT_FILENAME
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionBlocked,
    build_paid_lane_admission,
    require_paid_resource_admission,
)


PROBE_KIND = "native-g1-development-campaign"
RESULT_SCHEMA = "native_g1_paid_campaign_result.v1"


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("g1_paid_campaign_evidence_missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("g1_paid_campaign_evidence_invalid")
    return value


def verify_g1_paid_output(result: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    """Verify all four scored episodes and review videos after provider zero."""

    if result.get("status") != "completed" or result.get("continuing_spend_from_this_run") is not False:
        raise ValueError("g1_paid_campaign_transport_or_provider_zero_incomplete")
    attempt_root = Path(str(result.get("attempt_root") or ""))
    root = attempt_root / "immutable_execution"
    terminal = _read(root / RESULT_FILENAME)
    if (
        terminal.get("schema_version") != "native_g1_provider_campaign_result.v1"
        or terminal.get("result_digest") != canonical_digest(terminal, digest_field="result_digest")
        or terminal.get("status") != "completed"
        or terminal.get("claim_ceiling") != "development_only"
        or terminal.get("campaign_plan_digest") != bundle.get("campaign_plan_digest")
        or terminal.get("publisher_source_receipt_digest")
        != bundle.get("publisher_source_receipt_digest")
        or terminal.get("runtime_source_packet_sha256")
        != (bundle.get("runtime_source_packet") or {}).get("packet_sha256")
        or terminal.get("ranking_eligible") is not False
        or terminal.get("physical_outcome_claimed") is not False
        or list(terminal.get("policy_query_counts") or {}) != list(PAIR_ORDER)
        or any(
            not isinstance(value, int) or value < 1
            for value in (terminal.get("policy_query_counts") or {}).values()
        )
    ):
        raise ValueError("g1_paid_campaign_terminal_result_invalid")
    pairs = terminal.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 2:
        raise ValueError("g1_paid_campaign_pairs_missing")
    verified: list[dict[str, Any]] = []
    for index, (packet_name, objective, candidates) in enumerate(
        (
            ("manipulation", "task_success", PAIR_ORDER[:2]),
            ("movement", "g1_navigation_goal", PAIR_ORDER[2:]),
        )
    ):
        pair_root = root / (packet_name + "_pair")
        pair = _read(pair_root / "native_g1_development_pair.v1.json")
        row = pairs[index]
        if (
            not isinstance(row, dict)
            or row.get("objective_id") != objective
            or row.get("pair_relative_path")
            != packet_name + "_pair/native_g1_development_pair.v1.json"
            or row.get("pair_result_digest") != pair.get("result_digest")
            or pair.get("result_digest") != canonical_digest(pair, digest_field="result_digest")
            or pair.get("status") != "completed_development_only"
            or pair.get("candidate_ids") != list(candidates)
            or pair.get("objective_id") != objective
            or pair.get("ranking_eligible") is not False
            or pair.get("physical_outcome_claimed") is not False
        ):
            raise ValueError("g1_paid_campaign_pair_invalid:" + packet_name)
        attempts = pair.get("attempts")
        if not isinstance(attempts, list) or len(attempts) != 2:
            raise ValueError("g1_paid_campaign_attempts_incomplete:" + packet_name)
        for candidate, attempt in zip(candidates, attempts, strict=True):
            candidate_root = pair_root / candidate / "episode"
            worker = _read(pair_root / candidate / "native_g1_development_worker_result.v1.json")
            episode_path = candidate_root / EPISODE_FILENAME
            episode = _read(episode_path)
            trace = _read(candidate_root / TRACE_FILENAME)
            if (
                attempt.get("candidate_id") != candidate
                or attempt.get("status") != "completed_development_only"
                or attempt.get("worker_result_digest") != worker.get("result_digest")
                or worker.get("result_digest") != canonical_digest(worker, digest_field="result_digest")
                or trace.get("policy_query_count")
                != terminal["policy_query_counts"][candidate]
            ):
                raise ValueError("g1_paid_campaign_candidate_invalid:" + candidate)
            score = _score_from_episode(episode_path, worker=worker, objective_id=objective)
            media = _verified_review_media(
                episode_path, episode=episode, pair_root=pair_root
            )
            validate_multicamera_frame_manifest(
                _read(pair_root / media["frame_manifest"]["relative_path"]),
                output_dir=candidate_root,
                verify_files=True,
            )
            if attempt.get("score") != score or attempt.get("review_media") != media:
                raise ValueError("g1_paid_campaign_score_or_media_invalid:" + candidate)
            verified.append({
                "candidate_id": candidate,
                "objective_id": objective,
                "policy_query_count": trace["policy_query_count"],
                "score": score,
                "review_videos": media["review_videos"],
                "frame_manifest_digest": media["frame_manifest_digest"],
            })
    return {
        "schema_version": "native_g1_paid_output_verification.v1",
        "status": "verified_development_only",
        "campaign_plan_digest": bundle["campaign_plan_digest"],
        "terminal_result_digest": terminal["result_digest"],
        "episodes": verified,
        "public_redistribution_authorized": False,
        "physical_outcome_claimed": False,
    }


def dispatch_g1_paid_campaign(
    args: Any,
    *,
    control_identity: dict[str, Any],
    control_blockers: list[str],
    control_recheck: Callable[[], tuple[list[str], dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Build or load exact bytes, admit spend, launch, and verify terminal output."""

    blockers = list(control_blockers)
    commit = str(control_identity.get("orchestrator_source_commit") or "")
    if (
        not commit
        or commit != control_identity.get("origin_main_commit")
        or commit != control_identity.get("remote_main_commit")
    ):
        blockers.append("g1_paid_campaign_controller_not_exact_main")
    if args.provider != "vast":
        blockers.append("g1_paid_campaign_provider_must_be_vast")
    if not args.adp_job_dir or not args.admission_out or not args.adapter_output:
        blockers.append("g1_paid_campaign_output_paths_missing")
    rate = args.adp_max_hourly_rate_usd
    cap = args.adp_max_spend_usd
    ttl = args.adp_hard_ttl_seconds
    if (
        isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 < rate <= 5
        or isinstance(cap, bool) or not isinstance(cap, (int, float)) or not 0 < cap <= 20
        or isinstance(ttl, bool) or not isinstance(ttl, int) or not 1800 <= ttl <= 14400
    ):
        blockers.append("g1_paid_campaign_budget_invalid")
    if any(value <= 0 for value in args.adp_allowed_active_vast_instance_id):
        blockers.append("g1_paid_campaign_allowed_active_instance_id_invalid")
    if args.execute and not args.g1_campaign_bundle_receipt:
        blockers.append("g1_paid_campaign_execute_requires_dry_run_bundle_receipt")
    if args.adp_job_dir:
        job = Path(args.adp_job_dir)
        if not job.is_absolute() or job.is_symlink():
            blockers.append("g1_paid_campaign_job_path_invalid")
        elif not blockers:
            job.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, Any] | None = None
    if not blockers:
        try:
            if args.g1_campaign_bundle_receipt:
                bundle = load_verified_g1_provider_bundle(
                    Path(args.g1_campaign_bundle_receipt),
                    expected_implementation_commit=commit,
                )
            elif not args.execute:
                required = (
                    "g1_campaign_manipulation_packet", "g1_campaign_movement_packet",
                    "g1_campaign_book_handoff", "g1_campaign_navigation_authority",
                    "g1_campaign_publisher_source", "g1_campaign_runtime_source_receipt",
                )
                missing = [name for name in required if not getattr(args, name)]
                rights = {}
                for raw in args.g1_campaign_rights_review:
                    candidate, separator, path = raw.partition("=")
                    if not separator or candidate in rights:
                        raise ValueError("g1_paid_campaign_rights_argument_invalid")
                    rights[candidate] = Path(path)
                if missing:
                    raise ValueError("g1_paid_campaign_inputs_missing:" + ",".join(missing))
                bundle = build_g1_provider_bundle(
                    job_dir=Path(args.adp_job_dir) / "bundle",
                    manipulation_packet=Path(args.g1_campaign_manipulation_packet),
                    movement_packet=Path(args.g1_campaign_movement_packet),
                    book_handoff=Path(args.g1_campaign_book_handoff),
                    rights_review_paths=rights,
                    navigation_authority=Path(args.g1_campaign_navigation_authority),
                    publisher_source=Path(args.g1_campaign_publisher_source),
                    runtime_source_receipt=Path(args.g1_campaign_runtime_source_receipt),
                    implementation_commit=commit,
                )
            else:
                raise ValueError("g1_paid_campaign_bundle_receipt_missing")
        except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
            blockers.append("g1_paid_campaign_bundle_preparation_failed:" + type(exc).__name__)
    allocation_binding = {
        "program_id": "arm-decision-proof-v1",
        "probe_kind": PROBE_KIND,
        "provider": "vast",
        "orchestrator_source_commit": commit,
        "bundle_sha256": bundle.get("bundle_sha256") if bundle else None,
        "campaign_plan_digest": bundle.get("campaign_plan_digest") if bundle else None,
        "publisher_source_receipt_digest": (
            bundle.get("publisher_source_receipt_digest") if bundle else None
        ),
        "runtime_source_packet_sha256": (
            (bundle.get("runtime_source_packet") or {}).get("packet_sha256")
            if bundle else None
        ),
        "max_hourly_rate_usd": rate,
        "hard_cap_usd": cap,
        "hard_ttl_seconds": ttl,
        "retry_cap": 0,
    }
    allocation_binding_digest = canonical_digest(allocation_binding)
    admission = build_paid_lane_admission(
        resource_class="vast_provider_adapter", blockers=blockers
    )
    admission.update({
        "program_id": "arm-decision-proof-v1",
        "probe_kind": PROBE_KIND,
        "control_plane_identity": control_identity,
        "bundle_sha256": bundle.get("bundle_sha256") if bundle else None,
        "campaign_plan_digest": bundle.get("campaign_plan_digest") if bundle else None,
        "max_hourly_rate_usd": rate,
        "hard_cap_usd": cap,
        "hard_ttl_seconds": ttl,
        "retry_cap": 0,
        "allocation_binding": allocation_binding,
        "allocation_binding_digest": allocation_binding_digest,
        "claim_ceiling": "development_only",
        "authority": "owner_approved_g1_841757_development_simulation_and_bounded_gpu_compute",
        "private_data_uploaded": True,
        "physical_outcome_values_uploaded": False,
    })
    if args.admission_out:
        write_json(Path(args.admission_out), admission)
    if blockers or bundle is None:
        result = {"schema_version": RESULT_SCHEMA, "status": "blocked", "blockers": blockers,
                  "provider_mutations_performed": 0}
    else:
        grant = None
        if args.execute:
            try:
                grant = require_paid_resource_admission(
                    admission,
                    resource_class="vast_provider_adapter",
                    expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                )
            except PaidResourceAdmissionBlocked as exc:
                result = {"schema_version": RESULT_SCHEMA, "status": "blocked",
                          "blockers": exc.blockers, "provider_mutations_performed": 0}
                if args.adapter_output:
                    write_json(Path(args.adapter_output), result)
                return result
        def before_provider_create() -> dict[str, Any]:
            if control_recheck is not None:
                fresh_blockers, fresh_identity = control_recheck()
                if (
                    fresh_blockers
                    or fresh_identity.get("orchestrator_source_commit") != commit
                    or fresh_identity.get("origin_main_commit") != commit
                    or fresh_identity.get("remote_main_commit") != commit
                ):
                    return {
                        "status": "blocked",
                        "blockers": ["g1_paid_campaign_controller_identity_changed_before_create"],
                    }
            load_verified_g1_provider_bundle(
                Path(args.g1_campaign_bundle_receipt),
                expected_implementation_commit=commit,
            )
            consumption_path = Path(args.adp_job_dir) / "native_g1_paid_attempt_consumption.v1.json"
            consumption = {
                "schema_version": "native_g1_paid_attempt_consumption.v1",
                "status": "consumed",
                "allocation_binding_digest": allocation_binding_digest,
                "bundle_sha256": bundle["bundle_sha256"],
                "orchestrator_source_commit": commit,
                "provider": "vast",
                "retry_cap": 0,
            }
            try:
                with consumption_path.open("x", encoding="utf-8") as stream:
                    json.dump(consumption, stream, indent=2, sort_keys=True)
                    stream.write("\n")
            except FileExistsError:
                return {
                    "status": "blocked",
                    "blockers": ["g1_paid_campaign_attempt_already_consumed"],
                }
            return consumption
        result = run_arena_native_control_vast(
            pre_provider_mutation_hook=before_provider_create if args.execute else None,
            approval_path=args.g1_campaign_bundle_receipt or args.g1_campaign_book_handoff,
            job_dir=args.adp_job_dir,
            paid_resource_admission_grant=grant,
            execute=args.execute,
            prepared_bundle=bundle,
            machine_avoidlist_path=args.adp_machine_avoidlist,
            max_hourly_rate_usd=rate,
            hard_cap_usd=cap,
            hard_ttl_seconds=ttl,
            expected_output_filename=RESULT_FILENAME,
            container_image=NATIVE_TASK_ARENA_IMAGE,
            provider_bundle_kind=PROVIDER_BUNDLE_KIND,
            result_schema_version=RESULT_SCHEMA,
            instance_label_prefix="blueprint-g1-841757-",
            blocker_prefix="native_g1_campaign",
            min_gpu_ram_mb=48_000,
            candidate_policy_query_expected=True,
            require_independent_watchdog=True,
            allowed_active_instance_ids=args.adp_allowed_active_vast_instance_id,
            expected_provider_download_bytes=12_000_000_000,
            expected_provider_upload_bytes=10_000_000_000,
        )
        if args.execute and result.get("status") == "completed":
            try:
                result["g1_output_verification"] = verify_g1_paid_output(result, bundle)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                result["status"] = "blocked"
                result["blockers"] = sorted(set([
                    *(result.get("blockers") or []),
                    "g1_paid_campaign_output_verification_failed:" + type(exc).__name__,
                ]))
    if args.adapter_output:
        write_json(Path(args.adapter_output), result)
    return result
