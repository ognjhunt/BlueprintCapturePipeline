"""Materialize exact Joint Agent packet and narrow topology-only authority.

The historical public-scene authority bundled Aura, inpainting, simulator, and
policy permissions into one document.  A Scene 840920 Joint run needs none of
those.  This module derives one authority from the canonical dual-task
admission and the retained public-scene rights record, then materializes the
released Joint Agent packet without contacting a model, object store, or GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_joint_agent_admission import (
    READY_STATUS,
    validate_dual_task_joint_agent_admission,
)
from .usd_content_joint_agent_packet import (
    build_joint_agent_packet,
    inspect_joint_agent_checkout,
)


SCHEMA_VERSION = "joint_agent_topology_execution_authority.v1"
COMPOSITION_SCHEMA_VERSION = "joint_agent_topology_launch_inputs.v1"
RIGHTS_SCHEMA_VERSION = "public_scene_rights_authority.v1"
DATASET_CLAIM = (
    "internal_noncommercial_private_processing_authority_only;"
    "does_not_change_publisher_nonredistribution_terms"
)
PROVIDER_SCOPE = ("object_store", "openai", "vast")
PURPOSE_SCOPE = ("articulation_topology_inference", "construction_preview_rendering")
FORBIDDEN_LEGACY_FIELDS = frozenset(
    {
        "aura_adapter_receipt_digest",
        "derived_aura_adapter_upload_authorized",
        "attempt_authority",
    }
)


class JointAgentTopologyAuthorityError(ValueError):
    """Stable fail-closed topology authority/composition errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _clone(value: Mapping[str, Any], *, code: str) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise JointAgentTopologyAuthorityError([code]) from exc
    if not isinstance(result, dict):
        raise JointAgentTopologyAuthorityError([code])
    return result


def _validate_rights_authority(
    value: Mapping[str, Any],
    *,
    publisher_scene_id: str,
    sage_collision_sha256: str,
) -> dict[str, Any]:
    rights = _clone(value, code="joint_agent_rights_authority_invalid")
    authorized = rights.get("authorized_source_sha256")
    if (
        rights.get("schema_version") != RIGHTS_SCHEMA_VERSION
        or rights.get("scene_id") != publisher_scene_id
        or rights.get("program_id") != "arm-decision-proof-v1"
        or rights.get("reviewer_status") != "approved_for_declared_use"
        or rights.get("declared_use_scope") != "noncommercial_internal_research"
        or rights.get("agent_accepted_terms") is not False
        or rights.get("raw_dataset_redistribution_allowed") is not False
        or rights.get("commercial_use_allowed") is not False
        or not isinstance(authorized, list)
        or sage_collision_sha256 not in authorized
    ):
        raise JointAgentTopologyAuthorityError(
            ["joint_agent_rights_authority_invalid"]
        )
    return rights


def build_joint_agent_topology_execution_authority(
    *,
    dual_task_admission: Mapping[str, Any],
    rights_authority: Mapping[str, Any],
    rights_authority_file_sha256: str,
    rights_authority_size_bytes: int,
    authorized_by: str,
    authority_reference: str,
    authorized_on: str,
    hard_total_spend_cap_usd: float,
    maximum_single_resource_ttl_seconds: int,
    model_backend: str = "openai",
) -> dict[str, Any]:
    """Derive one exact, zero-retry Joint-only execution authority."""

    admission = validate_dual_task_joint_agent_admission(dual_task_admission)
    task = admission.get("task") or {}
    source = admission.get("source") or {}
    normalized_freeze = admission.get("normalized_freeze") or {}
    if (
        admission.get("status") != READY_STATUS
        or admission.get("paid_joint_agent_execution_permitted") is not True
    ):
        raise JointAgentTopologyAuthorityError(
            ["joint_agent_dual_task_admission_not_paid_applicable"]
        )
    rights = _validate_rights_authority(
        rights_authority,
        publisher_scene_id=str(task.get("publisher_scene_id") or ""),
        sage_collision_sha256=str(source.get("sage_collision_usd_sha256") or ""),
    )
    spend = hard_total_spend_cap_usd
    ttl = maximum_single_resource_ttl_seconds
    if (
        isinstance(spend, bool)
        or not isinstance(spend, (int, float))
        or not math.isfinite(float(spend))
        or not 0 < float(spend) <= 25
    ):
        raise JointAgentTopologyAuthorityError(
            ["joint_agent_topology_authority_spend_cap_invalid"]
        )
    if isinstance(ttl, bool) or not isinstance(ttl, int) or not 5_400 <= ttl <= 14_400:
        raise JointAgentTopologyAuthorityError(
            ["joint_agent_topology_authority_ttl_invalid"]
        )
    if model_backend != "openai":
        raise JointAgentTopologyAuthorityError(
            ["joint_agent_topology_authority_model_backend_invalid"]
        )
    if (
        not _digest(rights_authority_file_sha256)
        or isinstance(rights_authority_size_bytes, bool)
        or not isinstance(rights_authority_size_bytes, int)
        or rights_authority_size_bytes <= 0
        or not str(authorized_by).strip()
        or not str(authority_reference).strip()
        or not str(authorized_on).strip()
    ):
        raise JointAgentTopologyAuthorityError(
            ["joint_agent_topology_authority_identity_invalid"]
        )
    authority: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": str(authority_reference).strip(),
        "authorized_by": str(authorized_by).strip(),
        "authorized_on": str(authorized_on).strip(),
        "program_id": "arm-decision-proof-v1",
        "publisher_scene_id": task["publisher_scene_id"],
        "task_id": task["task_id"],
        "target_instance_id": source["target_instance_id"],
        "task_freeze_digest": task["task_freeze_digest"],
        "dual_task_admission_digest": admission["admission_digest"],
        "freeze_digest": normalized_freeze["freeze_digest"],
        "joint_agent_source_receipt_digest": source["source_receipt_digest"],
        "joint_agent_source_asset_digest": source["source_asset_sha256"],
        "sage_collision_usd_sha256": source["sage_collision_usd_sha256"],
        "prior_rights_authority": {
            "schema_version": rights["schema_version"],
            "authority_reference": rights["authority_reference"],
            "sha256": rights_authority_file_sha256,
            "size_bytes": rights_authority_size_bytes,
            "document": rights,
        },
        "provider_scope": list(PROVIDER_SCOPE),
        "purpose_scope": list(PURPOSE_SCOPE),
        "model_backend": model_backend,
        "hard_total_spend_cap_usd": float(spend),
        "maximum_single_resource_ttl_seconds": ttl,
        "maximum_paid_attempts": 1,
        "maximum_automatic_retries": 0,
        "one_instance_at_a_time": True,
        "maximum_concurrent_paid_instances": 1,
        "remote_upload_authorized": True,
        "paid_compute_authorized": True,
        "sage_cc_by_nc_derived_asset_upload_authorized": True,
        "raw_interiorgs_downloaded_bytes_upload_authorized": False,
        "public_disclosure_authorized": False,
        "model_training_authorized": False,
        "commercial_use_authorized": False,
        "provider_zero_required_before_and_after": True,
        "teardown_required": True,
        "retention_policy": "bounded_to_goal_then_provider_zero",
        "dataset_claim": DATASET_CLAIM,
        "claim_boundary": {
            "joint_agent_output_is_optional_topology_candidate": True,
            "non_task_joint_behavior_exercised": False,
            "simready_qualified": False,
            "physical_equivalence_proven": False,
        },
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    return authority


def validate_joint_agent_topology_execution_authority(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    authority = _clone(value, code="joint_agent_topology_authority_invalid")
    errors: list[str] = []
    if authority.get("schema_version") != SCHEMA_VERSION:
        errors.append("joint_agent_topology_authority_schema_invalid")
    if authority.get("authority_kind") != "explicit_user_direction_in_current_goal":
        errors.append("joint_agent_topology_authority_kind_invalid")
    for field in (
        "authority_reference",
        "authorized_by",
        "authorized_on",
        "publisher_scene_id",
        "task_id",
        "target_instance_id",
    ):
        if not isinstance(authority.get(field), str) or not authority[field].strip():
            errors.append(f"joint_agent_topology_authority_{field}_invalid")
    for field in (
        "task_freeze_digest",
        "dual_task_admission_digest",
        "freeze_digest",
        "joint_agent_source_receipt_digest",
        "joint_agent_source_asset_digest",
        "sage_collision_usd_sha256",
    ):
        if not _digest(authority.get(field)):
            errors.append(f"joint_agent_topology_authority_{field}_invalid")
    rights = authority.get("prior_rights_authority")
    if (
        not isinstance(rights, Mapping)
        or rights.get("schema_version") != RIGHTS_SCHEMA_VERSION
        or not isinstance(rights.get("authority_reference"), str)
        or not rights["authority_reference"].strip()
        or not _digest(rights.get("sha256"))
        or isinstance(rights.get("size_bytes"), bool)
        or not isinstance(rights.get("size_bytes"), int)
        or rights.get("size_bytes", 0) <= 0
        or not isinstance(rights.get("document"), Mapping)
    ):
        errors.append("joint_agent_topology_authority_rights_invalid")
    else:
        try:
            _validate_rights_authority(
                rights["document"],
                publisher_scene_id=str(authority.get("publisher_scene_id") or ""),
                sage_collision_sha256=str(
                    authority.get("sage_collision_usd_sha256") or ""
                ),
            )
        except JointAgentTopologyAuthorityError:
            errors.append("joint_agent_topology_authority_rights_invalid")
    spend = authority.get("hard_total_spend_cap_usd")
    ttl = authority.get("maximum_single_resource_ttl_seconds")
    claim = authority.get("claim_boundary")
    exact_values = {
        "program_id": "arm-decision-proof-v1",
        "provider_scope": list(PROVIDER_SCOPE),
        "purpose_scope": list(PURPOSE_SCOPE),
        "model_backend": "openai",
        "maximum_paid_attempts": 1,
        "maximum_automatic_retries": 0,
        "one_instance_at_a_time": True,
        "maximum_concurrent_paid_instances": 1,
        "remote_upload_authorized": True,
        "paid_compute_authorized": True,
        "sage_cc_by_nc_derived_asset_upload_authorized": True,
        "raw_interiorgs_downloaded_bytes_upload_authorized": False,
        "public_disclosure_authorized": False,
        "model_training_authorized": False,
        "commercial_use_authorized": False,
        "provider_zero_required_before_and_after": True,
        "teardown_required": True,
        "retention_policy": "bounded_to_goal_then_provider_zero",
        "dataset_claim": DATASET_CLAIM,
    }
    for field, expected in exact_values.items():
        if authority.get(field) != expected:
            errors.append(f"joint_agent_topology_authority_{field}_invalid")
    if FORBIDDEN_LEGACY_FIELDS.intersection(authority):
        errors.append("joint_agent_topology_authority_legacy_scope_present")
    if (
        isinstance(spend, bool)
        or not isinstance(spend, (int, float))
        or not math.isfinite(float(spend))
        or not 0 < float(spend) <= 25
    ):
        errors.append("joint_agent_topology_authority_spend_cap_invalid")
    if isinstance(ttl, bool) or not isinstance(ttl, int) or not 5_400 <= ttl <= 14_400:
        errors.append("joint_agent_topology_authority_ttl_invalid")
    if claim != {
        "joint_agent_output_is_optional_topology_candidate": True,
        "non_task_joint_behavior_exercised": False,
        "simready_qualified": False,
        "physical_equivalence_proven": False,
    }:
        errors.append("joint_agent_topology_authority_claim_boundary_invalid")
    if authority.get("authorization_digest") != canonical_digest(
        authority, digest_field="authorization_digest"
    ):
        errors.append("joint_agent_topology_authority_digest_invalid")
    if errors:
        raise JointAgentTopologyAuthorityError(errors)
    return authority


def materialize_joint_agent_topology_launch_inputs(
    *,
    dual_task_admission_path: str | Path,
    source_asset_path: str | Path,
    source_receipt_path: str | Path,
    rights_authority_path: str | Path,
    joint_agent_checkout: str | Path,
    output_dir: str | Path,
    authorized_by: str,
    authority_reference: str,
    authorized_on: str,
    hard_total_spend_cap_usd: float,
    maximum_single_resource_ttl_seconds: int,
) -> dict[str, Any]:
    """Write packet, authority, and a self-digesting no-spend composition."""

    admission_file = Path(dual_task_admission_path).expanduser().resolve()
    source_asset = Path(source_asset_path).expanduser().resolve()
    source_receipt_file = Path(source_receipt_path).expanduser().resolve()
    rights_file = Path(rights_authority_path).expanduser().resolve()
    checkout = Path(joint_agent_checkout).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise JointAgentTopologyAuthorityError(
            ["joint_agent_topology_output_not_empty"]
        )
    try:
        admission = validate_dual_task_joint_agent_admission(
            json.loads(admission_file.read_text(encoding="utf-8"))
        )
        source_receipt = json.loads(source_receipt_file.read_text(encoding="utf-8"))
        rights = json.loads(rights_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise JointAgentTopologyAuthorityError(
            ["joint_agent_topology_input_invalid"]
        ) from exc
    if (
        admission_file.is_symlink()
        or source_asset.is_symlink()
        or source_receipt_file.is_symlink()
        or rights_file.is_symlink()
        or not source_asset.is_file()
        or not isinstance(source_receipt, Mapping)
        or admission.get("source_receipt") != source_receipt
        or source_asset.stat().st_size != admission["source"]["source_asset_size_bytes"]
        or _sha256(source_asset) != admission["source"]["source_asset_sha256"]
    ):
        raise JointAgentTopologyAuthorityError(
            ["joint_agent_topology_source_bytes_invalid"]
        )
    inspect_joint_agent_checkout(checkout)
    authority = build_joint_agent_topology_execution_authority(
        dual_task_admission=admission,
        rights_authority=rights,
        rights_authority_file_sha256=_sha256(rights_file),
        rights_authority_size_bytes=rights_file.stat().st_size,
        authorized_by=authorized_by,
        authority_reference=authority_reference,
        authorized_on=authorized_on,
        hard_total_spend_cap_usd=hard_total_spend_cap_usd,
        maximum_single_resource_ttl_seconds=maximum_single_resource_ttl_seconds,
    )
    packet = build_joint_agent_packet(
        source_asset_path=source_asset,
        source_receipt_path=source_receipt_file,
        joint_agent_checkout=checkout,
        output_dir=destination / "packet",
        external_disclosure_authorized=True,
        paid_execution_authorized=True,
    )
    if (
        packet.get("status") != "ready_for_dry_run_only"
        or (packet.get("execution_admission") or {}).get("blockers") != []
        or (packet.get("source_asset") or {}).get("sha256")
        != authority["joint_agent_source_asset_digest"]
        or (packet.get("source_asset") or {}).get("source_receipt_digest")
        != authority["joint_agent_source_receipt_digest"]
    ):
        raise JointAgentTopologyAuthorityError(
            ["joint_agent_topology_packet_authority_join_invalid"]
        )
    destination.mkdir(parents=True, exist_ok=True)
    authority_path = destination / "joint_agent_topology_execution_authority.json"
    authority_path.write_text(canonical_json(authority) + "\n", encoding="utf-8")
    packet_path = destination / "packet/joint_agent_packet.json"
    composition: dict[str, Any] = {
        "schema_version": COMPOSITION_SCHEMA_VERSION,
        "status": "ready_for_bundle_construction_no_remote_execution",
        "dual_task_admission": {
            "path": str(admission_file),
            "sha256": _sha256(admission_file),
            "admission_digest": admission["admission_digest"],
        },
        "packet": {
            "path": str(packet_path),
            "sha256": _sha256(packet_path),
            "packet_digest": packet["packet_digest"],
        },
        "execution_authority": {
            "path": str(authority_path),
            "sha256": _sha256(authority_path),
            "authorization_digest": authority["authorization_digest"],
        },
        "released_code": packet["released_code"],
        "provider_mutation_performed": False,
        "paid_resource_allocated": False,
        "claim_boundary": authority["claim_boundary"],
        "composition_digest": "",
    }
    composition["composition_digest"] = canonical_digest(
        composition, digest_field="composition_digest"
    )
    composition_path = destination / "joint_agent_topology_launch_inputs.json"
    composition_path.write_text(canonical_json(composition) + "\n", encoding="utf-8")
    return composition


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual-task-admission", required=True)
    parser.add_argument("--source-asset", required=True)
    parser.add_argument("--source-receipt", required=True)
    parser.add_argument("--rights-authority", required=True)
    parser.add_argument("--joint-agent-checkout", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--authorized-by", required=True)
    parser.add_argument("--authority-reference", required=True)
    parser.add_argument("--authorized-on", required=True)
    parser.add_argument("--max-spend-usd", required=True, type=float)
    parser.add_argument("--hard-ttl-seconds", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        result = materialize_joint_agent_topology_launch_inputs(
            dual_task_admission_path=args.dual_task_admission,
            source_asset_path=args.source_asset,
            source_receipt_path=args.source_receipt,
            rights_authority_path=args.rights_authority,
            joint_agent_checkout=args.joint_agent_checkout,
            output_dir=args.output_dir,
            authorized_by=args.authorized_by,
            authority_reference=args.authority_reference,
            authorized_on=args.authorized_on,
            hard_total_spend_cap_usd=args.max_spend_usd,
            maximum_single_resource_ttl_seconds=args.hard_ttl_seconds,
        )
    except (OSError, JointAgentTopologyAuthorityError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "COMPOSITION_SCHEMA_VERSION",
    "FORBIDDEN_LEGACY_FIELDS",
    "JointAgentTopologyAuthorityError",
    "SCHEMA_VERSION",
    "build_joint_agent_topology_execution_authority",
    "materialize_joint_agent_topology_launch_inputs",
    "validate_joint_agent_topology_execution_authority",
]
