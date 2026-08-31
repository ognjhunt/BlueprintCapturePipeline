"""Canonical allocator boundary for the scene-configuration Vast probe."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .host_resident_launch_inputs import (
    HostResidentInputError,
    resolve_host_resident_bundle_receipt,
)
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionBlocked,
    PaidResourceAdmissionGrant,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from .task_evaluation_scene_configuration_bundle import (
    PROBE_KIND,
    load_scene_configuration_provider_bundle_receipt,
)
from .task_evaluation_scene_configuration_diagnostic_release import (
    SceneConfigurationDiagnosticReleaseError,
    validate_scene_configuration_diagnostic_release_receipt,
)
from .task_evaluation_scene_configuration_paid_authority import (
    validate_scene_configuration_paid_authority,
)
from .task_evaluation_scene_configuration_vast import run_scene_configuration_vast


ControlPlaneIdentityProbe = Callable[[], tuple[list[str], dict[str, object]]]
ExpectedSourceCommitProbe = Callable[
    [str, Mapping[str, object]], tuple[list[str], str]
]
ROOT = Path(__file__).resolve().parents[2]


def _run_scene_configuration_warm_action(args: argparse.Namespace) -> int:
    from .task_evaluation_scene_configuration_warm_diagnostic import (  # noqa: PLC0415
        close_scene_configuration_warm_session,
        run_scene_configuration_warm_iteration,
        scene_configuration_warm_closeout_allocation_binding,
        scene_configuration_warm_iteration_allocation_binding,
        validate_scene_configuration_warm_iteration_authority,
        validate_scene_configuration_warm_session,
    )

    action = str(args.scene_configuration_warm_action or "")
    blockers: list[str] = []
    session_root_value = str(
        getattr(args, "scene_configuration_warm_session_root", "") or ""
    ).strip()
    if args.provider != "vast":
        blockers.append("scene_configuration_provider_must_be_vast")
    if not session_root_value:
        blockers.append("scene_configuration_warm_session_root_missing")
    session: dict[str, Any] | None = None
    state: dict[str, Any] | None = None
    authority: dict[str, Any] | None = None
    try:
        if session_root_value:
            session, state = validate_scene_configuration_warm_session(
                session_root=Path(session_root_value).expanduser(),
                require_iteration_window=action == "iterate",
                allowed_state_statuses=(
                    frozenset({"ready", "iteration_failed"})
                    if action == "iterate"
                    else frozenset(
                        {
                            "ready",
                            "iteration_failed",
                            "iteration_running",
                            "teardown_required",
                        }
                    )
                ),
            )
        if action == "iterate":
            authority_value = str(
                getattr(args, "scene_configuration_warm_iteration_authority", "")
                or ""
            ).strip()
            if not authority_value:
                blockers.append("scene_configuration_warm_iteration_authority_missing")
            elif session_root_value:
                session, state, authority = (
                    validate_scene_configuration_warm_iteration_authority(
                        session_root=Path(session_root_value).expanduser(),
                        authority_path=Path(authority_value).expanduser(),
                    )
                )
            if not getattr(args, "scene_configuration_job_dir", None):
                blockers.append("scene_configuration_job_dir")
        elif action == "closeout":
            if not getattr(args, "scene_configuration_warm_closeout_receipt", None):
                blockers.append("scene_configuration_warm_closeout_receipt_missing")
        else:
            blockers.append("scene_configuration_warm_action_invalid")
    except (OSError, TypeError, ValueError):
        blockers.append(f"scene_configuration_warm_{action}_authority_invalid")
    binding = (
        scene_configuration_warm_iteration_allocation_binding(
            session=session or {}, authority=authority or {}
        )
        if action == "iterate"
        else scene_configuration_warm_closeout_allocation_binding(
            session=session or {}
        )
    )
    admission = build_paid_lane_admission(
        resource_class="vast_provider_adapter",
        blockers=sorted(set(blockers)),
    )
    admission.update(
        {
            "program_id": "arm-decision-proof-v1",
            "probe_kind": PROBE_KIND,
            "authority": "bounded_scene_configuration_warm_action",
            "warm_action": action,
            "provider_allocations_permitted": 0,
            "diagnostic_only": True,
            "qualification_eligible": False,
            "configured_revision_publication_permitted": False,
            "offering_publication_permitted": False,
            "terminal_e2e_completion_permitted": False,
            "retry_cap": 0,
            "allocation_binding": binding,
            "allocation_binding_digest": canonical_digest(binding),
        }
    )
    write_json(Path(args.admission_out), admission)
    grant: PaidResourceAdmissionGrant | None = None
    if args.execute:
        try:
            grant = require_paid_resource_admission(
                admission,
                resource_class="vast_provider_adapter",
                expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
            )
        except PaidResourceAdmissionBlocked as exc:
            result = {
                "status": "blocked",
                "blockers": exc.blockers,
                "provider_allocations_performed": 0,
                "diagnostic_only": True,
            }
            write_json(Path(args.adapter_output), result)
            print(json.dumps({"success": False}, sort_keys=True))
            return 2
    if blockers or session is None:
        result = {
            "status": "blocked",
            "blockers": sorted(set(blockers)),
            "provider_allocations_performed": 0,
            "diagnostic_only": True,
        }
    elif action == "iterate" and authority is not None:
        result = run_scene_configuration_warm_iteration(
            session_root=session_root_value,
            authority_path=args.scene_configuration_warm_iteration_authority,
            job_dir=args.scene_configuration_job_dir,
            paid_resource_admission_grant=grant,
            execute=args.execute,
        )
    else:
        result = close_scene_configuration_warm_session(
            session_root=session_root_value,
            output_path=args.scene_configuration_warm_closeout_receipt,
            paid_resource_admission_grant=grant,
            execute=args.execute,
        )
    write_json(Path(args.adapter_output), result)
    success = result.get("status") in {
        "dry_run_ready",
        "completed",
        "completed_diagnostic_only",
    }
    print(json.dumps({"success": success}, sort_keys=True))
    return 0 if success else 2


def run_scene_configuration_allocator_probe(
    args: argparse.Namespace,
    *,
    control_plane_identity_probe: ControlPlaneIdentityProbe,
    expected_source_commit_probe: ExpectedSourceCommitProbe,
) -> int:
    """Admit and optionally execute one Website-bound scene configuration."""

    if getattr(args, "scene_configuration_warm_action", None):
        return _run_scene_configuration_warm_action(args)

    diagnostic_only = bool(
        getattr(args, "scene_configuration_diagnostic_only", False)
    )
    retain_warm_session = bool(
        getattr(args, "scene_configuration_retain_warm_session", False)
    )
    allowed_machine_values = (
        getattr(args, "scene_configuration_allowed_vast_machine_id", ()) or ()
    )
    allowed_machine_ids: tuple[int, ...] = ()
    missing = [
        name
        for name in (
            "scene_configuration_bundle_receipt",
            "scene_configuration_attempt_authority",
            "scene_configuration_job_dir",
        )
        if not getattr(args, name, None)
    ]
    control_blockers, control_identity = control_plane_identity_probe()
    source_blockers, expected_source_commit = expected_source_commit_probe(
        args.expected_source_commit or "", control_identity
    )
    blockers = [*missing, *control_blockers, *source_blockers]
    try:
        if any(isinstance(value, bool) for value in allowed_machine_values):
            raise ValueError("boolean machine id")
        allowed_machine_ids = tuple(
            sorted({int(value) for value in allowed_machine_values})
        )
        if any(value <= 0 for value in allowed_machine_ids):
            raise ValueError("non-positive machine id")
    except (TypeError, ValueError):
        blockers.append("scene_configuration_allowed_vast_machine_ids_invalid")
        allowed_machine_ids = ()
    if allowed_machine_ids and not diagnostic_only:
        blockers.append(
            "scene_configuration_machine_allowlist_requires_diagnostic_only"
        )
    diagnostic_release: dict[str, Any] | None = None
    if diagnostic_only:
        diagnostic_release_path = str(
            getattr(args, "release_evidence", "") or ""
        ).strip()
        if not diagnostic_release_path:
            blockers.append("scene_configuration_diagnostic_release_receipt_missing")
        else:
            try:
                diagnostic_release = (
                    validate_scene_configuration_diagnostic_release_receipt(
                        Path(diagnostic_release_path).expanduser(),
                        expected_source_commit=expected_source_commit,
                        expected_release_path=ROOT,
                    )
                )
            except (OSError, SceneConfigurationDiagnosticReleaseError):
                blockers.append("scene_configuration_diagnostic_release_receipt_invalid")
    warm_session_authority: dict[str, Any] | None = None
    warm_session_authority_path: Path | None = None
    warm_session_output_root = str(
        getattr(args, "scene_configuration_warm_session_output_root", "") or ""
    ).strip()
    if retain_warm_session:
        if not diagnostic_only:
            blockers.append("scene_configuration_warm_session_requires_diagnostic_only")
        warm_path_value = str(
            getattr(args, "scene_configuration_warm_session_authority", "") or ""
        ).strip()
        if not warm_path_value:
            blockers.append("scene_configuration_warm_session_authority_missing")
        if not warm_session_output_root:
            blockers.append("scene_configuration_warm_session_output_root_missing")
        if warm_path_value:
            try:
                from .task_evaluation_scene_configuration_warm_diagnostic import (  # noqa: PLC0415
                    validate_scene_configuration_warm_session_authority,
                )

                warm_session_authority_path = Path(warm_path_value).expanduser()
                warm_session_authority = (
                    validate_scene_configuration_warm_session_authority(
                        warm_session_authority_path
                    )
                )
            except (OSError, TypeError, ValueError):
                blockers.append("scene_configuration_warm_session_authority_invalid")
    if args.provider != "vast":
        blockers.append("scene_configuration_provider_must_be_vast")
    receipt_path: Path | None = None
    prepared_bundle: dict[str, Any] | None = None
    if args.scene_configuration_bundle_receipt:
        receipt_path = Path(
            args.scene_configuration_bundle_receipt
        ).expanduser().resolve()
        try:
            resolution = resolve_host_resident_bundle_receipt(receipt_path)
            blockers.extend(resolution["blockers"])
            prepared_bundle = load_scene_configuration_provider_bundle_receipt(
                receipt_path,
                expected_source_commit=expected_source_commit or None,
                diagnostic_only=diagnostic_only,
            )
        except (
            HostResidentInputError,
            OSError,
            ValueError,
            json.JSONDecodeError,
        ):
            blockers.append("scene_configuration_bundle_receipt_invalid")
    authority_path: Path | None = None
    authority: dict[str, Any] | None = None
    if args.scene_configuration_attempt_authority:
        authority_path = Path(
            args.scene_configuration_attempt_authority
        ).expanduser().resolve()
        try:
            authority = _load_json(authority_path)
            if prepared_bundle is None:
                raise ValueError("bundle unavailable")
            validate_scene_configuration_paid_authority(
                authority, bundle_receipt=prepared_bundle
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            blockers.append("scene_configuration_paid_authority_invalid")
            authority = None
    if authority is not None and args.pod_name != authority.get("resource_name"):
        blockers.append("scene_configuration_resource_name_mismatch")
    if warm_session_authority is not None and (
        prepared_bundle is None
        or authority is None
        or warm_session_authority.get("bundle_sha256")
        != prepared_bundle.get("bundle_sha256")
        or warm_session_authority.get("paid_attempt_authority_digest")
        != authority.get("authority_digest")
        or warm_session_authority.get("source_commit")
        != expected_source_commit
    ):
        blockers.append("scene_configuration_warm_session_authority_binding_mismatch")
    allocation_binding = {
        "program_id": "arm-decision-proof-v1",
        "probe_kind": PROBE_KIND,
        "orchestrator_source_commit": control_identity.get(
            "orchestrator_source_commit"
        ),
        "expected_source_commit": expected_source_commit or None,
        "run_id": prepared_bundle.get("run_id") if prepared_bundle else None,
        "bundle_sha256": (
            prepared_bundle.get("bundle_sha256") if prepared_bundle else None
        ),
        "portable_construction_envelope_digest": (
            prepared_bundle.get("portable_construction_envelope_digest")
            if prepared_bundle
            else None
        ),
        "toolchain_digest": (
            prepared_bundle.get("toolchain_digest") if prepared_bundle else None
        ),
        "authority_digest": authority.get("authority_digest") if authority else None,
        "resource_name": authority.get("resource_name") if authority else None,
        "container_image": authority.get("container_image") if authority else None,
        "max_hourly_rate_usd": (
            authority.get("maximum_hourly_rate_usd") if authority else None
        ),
        "hard_cap_usd": (
            authority.get("hard_attempt_spend_cap_usd") if authority else None
        ),
        "hard_ttl_seconds": (
            authority.get("maximum_single_resource_ttl_seconds")
            if authority
            else None
        ),
        "allowed_active_vast_instance_ids": [],
        "allowed_vast_machine_ids": list(allowed_machine_ids),
        "retry_cap": 0,
        "diagnostic_only": diagnostic_only,
        "retain_warm_session": retain_warm_session,
        "warm_session_authority_digest": (
            warm_session_authority.get("authority_digest")
            if warm_session_authority
            else None
        ),
        "source_diagnostic_checkpoint_digest": (
            prepared_bundle.get("source_diagnostic_checkpoint_digest")
            if prepared_bundle
            else None
        ),
        "carried_completed_stage_count": (
            prepared_bundle.get("carried_completed_stage_count")
            if prepared_bundle
            else None
        ),
        "diagnostic_bootstrap_mode": (
            prepared_bundle.get("diagnostic_bootstrap_mode")
            if prepared_bundle
            else None
        ),
        "diagnostic_scientific_binding_digest": (
            prepared_bundle.get("diagnostic_scientific_binding_digest")
            if prepared_bundle
            else None
        ),
        "diagnostic_stage_sequence_ids": (
            prepared_bundle.get("diagnostic_stage_sequence_ids")
            if prepared_bundle
            else None
        ),
    }
    if diagnostic_only:
        allocation_binding.update(
            {
                "diagnostic_release_receipt_digest": (
                    diagnostic_release.get("receipt_digest")
                    if diagnostic_release
                    else None
                ),
                "diagnostic_remote_ref": (
                    diagnostic_release.get("remote_ref")
                    if diagnostic_release
                    else None
                ),
                "diagnostic_remote_ref_tip_commit": (
                    diagnostic_release.get("remote_ref_tip_commit")
                    if diagnostic_release
                    else None
                ),
            }
        )
    admission = build_paid_lane_admission(
        resource_class="vast_provider_adapter",
        blockers=sorted(set(blockers)),
    )
    admission.update(
        {
            "program_id": "arm-decision-proof-v1",
            "probe_kind": PROBE_KIND,
            "control_plane_identity": control_identity,
            "authority": "user_authorized_scene_configuration",
            "raw_interiorgs_bytes_uploaded": False,
            "derived_rendered_views_only": True,
            "evaluation_episode_authorized": False,
            "single_parent_allocation": True,
            "diagnostic_only": diagnostic_only,
            "qualification_eligible": not diagnostic_only,
            "configured_revision_publication_permitted": not diagnostic_only,
            "offering_publication_permitted": not diagnostic_only,
            "terminal_e2e_completion_permitted": not diagnostic_only,
            "retry_cap": 0,
            "allocation_binding": allocation_binding,
            "allocation_binding_digest": canonical_digest(allocation_binding),
        }
    )
    write_json(Path(args.admission_out), admission)
    grant: PaidResourceAdmissionGrant | None = None
    if args.execute:
        try:
            grant = require_paid_resource_admission(
                admission,
                resource_class="vast_provider_adapter",
                expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
            )
        except PaidResourceAdmissionBlocked as exc:
            result = {
                "status": "blocked",
                "blockers": exc.blockers,
                "provider_mutations_performed": 0,
                "retry_cap": 0,
            }
            if diagnostic_only:
                result.update(
                    {
                        "diagnostic_only": True,
                        "qualification_eligible": False,
                        "configured_revision_publication_permitted": False,
                        "offering_publication_permitted": False,
                        "terminal_e2e_completion_permitted": False,
                    }
                )
            write_json(Path(args.adapter_output), result)
            print(json.dumps({"success": False}, sort_keys=True))
            return 2
    if (
        blockers
        or prepared_bundle is None
        or receipt_path is None
        or authority is None
        or authority_path is None
    ):
        result = {
            "status": "blocked",
            "blockers": sorted(set(blockers)),
            "provider_mutations_performed": 0,
            "retry_cap": 0,
        }
        if diagnostic_only:
            result.update(
                {
                    "diagnostic_only": True,
                    "qualification_eligible": False,
                    "configured_revision_publication_permitted": False,
                    "offering_publication_permitted": False,
                    "terminal_e2e_completion_permitted": False,
                }
            )
    else:
        result = run_scene_configuration_vast(
            job_dir=args.scene_configuration_job_dir,
            bundle_receipt_path=receipt_path,
            paid_attempt_authority_path=authority_path,
            paid_resource_admission_grant=grant,
            execute=args.execute,
            diagnostic_only=diagnostic_only,
            retain_warm_session=retain_warm_session,
            allowed_machine_ids=allowed_machine_ids,
            warm_session_authority_path=warm_session_authority_path,
            warm_session_output_root=(
                Path(warm_session_output_root).expanduser()
                if warm_session_output_root
                else None
            ),
            scene_construction_queue_root=(
                None
                if diagnostic_only
                else (
                    getattr(args, "scene_configuration_queue_root", None)
                    or os.getenv(
                        "BLUEPRINT_TASK_EVALUATION_SCENE_CONSTRUCTION_QUEUE_ROOT"
                    )
                )
            ),
        )
    write_json(Path(args.adapter_output), result)
    success = result.get("status") in {
        "dry_run_ready",
        "completed",
        "dry_run_ready_diagnostic_only",
        "completed_diagnostic_only",
        "retained_warm_diagnostic_session",
    }
    print(json.dumps({"success": success}, sort_keys=True))
    return 0 if success else 2


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("json_object_required")
    return dict(value)


__all__ = ["PROBE_KIND", "run_scene_configuration_allocator_probe"]
