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
from .task_evaluation_scene_configuration_paid_authority import (
    validate_scene_configuration_paid_authority,
)
from .task_evaluation_scene_configuration_vast import run_scene_configuration_vast


ControlPlaneIdentityProbe = Callable[[], tuple[list[str], dict[str, object]]]
ExpectedSourceCommitProbe = Callable[
    [str, Mapping[str, object]], tuple[list[str], str]
]


def run_scene_configuration_allocator_probe(
    args: argparse.Namespace,
    *,
    control_plane_identity_probe: ControlPlaneIdentityProbe,
    expected_source_commit_probe: ExpectedSourceCommitProbe,
) -> int:
    """Admit and optionally execute one Website-bound scene configuration."""

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
        "retry_cap": 0,
    }
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
    else:
        result = run_scene_configuration_vast(
            job_dir=args.scene_configuration_job_dir,
            bundle_receipt_path=receipt_path,
            paid_attempt_authority_path=authority_path,
            paid_resource_admission_grant=grant,
            execute=args.execute,
            scene_construction_queue_root=os.getenv(
                "BLUEPRINT_TASK_EVALUATION_SCENE_CONSTRUCTION_QUEUE_ROOT"
            ),
        )
    write_json(Path(args.adapter_output), result)
    success = result.get("status") in {"dry_run_ready", "completed"}
    print(json.dumps({"success": success}, sort_keys=True))
    return 0 if success else 2


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("json_object_required")
    return dict(value)


__all__ = ["PROBE_KIND", "run_scene_configuration_allocator_probe"]
