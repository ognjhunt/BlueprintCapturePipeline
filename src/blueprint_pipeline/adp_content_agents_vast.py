"""Canonical zero-retry Vast execution for the bounded ADP-009A Content Agents case."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from .task_evaluation_artifact_manifest import (
    seal_lane_terminal_artifacts,
    seal_unallocated_provider_teardown,
)
from .common import ensure_dir, utc_now_iso, write_json, redacted_failure_detail
from .gpu_render_providers import _read_secret as _read_provider_secret
from .adp_content_agents_bundle_matrix import (
    SCHEMA_VERSION as AGENT_CAD_BUNDLE_MATRIX_V2_SCHEMA,
    validate_agent_cad_content_agents_bundle_matrix,
)
from .content_agents_execution_route import (
    ContentAgentsExecutionRouteError,
    nvidia_content_agents_required,
    validate_content_agents_execution_route,
)
from .content_agents_model_compatibility import (
    materialize_content_agents_model_compatibility_plan,
)
from .decision_evidence_contracts import canonical_digest
from .paid_attempt_authority import (
    active_instance_allowlist_metadata_error,
    flatten_active_instance_allowlist,
    normalize_active_instance_allowlist,
    validate_bound_lane_prior_spend,
)
from .paid_resource_admission import PaidResourceAdmissionGrant
from .paired_target_native_construction_bindings import (
    PairedTargetNativeConstructionBindingsError,
    validate_paired_target_native_construction_bindings,
)
from .public_scene_host_input_intake import RIGHTS_RECEIPT_SCHEMA
from .openai_successor_models import (
    OPENAI_IMAGE_MODEL,
    OPENAI_REASONING_EFFORT,
    OPENAI_TEXT_MODEL,
)
from .openai_api_geography import OPENAI_API_SUPPORTED_COUNTRY_CODES
from .public_scene_simready_native import materialize_native_probe
from .provider_runtime_bundle_contract import provider_runtime_contract_blockers
from .provider_bundle_rehearsal import (
    provider_bundle_rehearsal_blockers,
    rehearse_provider_bundle_entrypoint,
)
from .simready_cad_agent_contract import (
    ADMITTED_BACKENDS,
    SimReadyCadAgentContractError,
    validate_cad_agent_output,
)
from .vast_independent_watchdog_control import (
    arm_independent_vast_watchdog,
    close_independent_vast_watchdog,
)
from .vast_provider_adapter import run_vast_provider_adapter
from .vast_session_budget_contract import attempt_estimated_cost, attempt_runtime_seconds
from .wam_provider_object_store import (
    cleanup_staged_wam_provider_objects,
    stage_wam_provider_bundle_object_store,
)


from .spend_authority_consumption_root import consumption_root

PROBE_KIND = "adp-usd-content-agents"
RESULT_SCHEMA_VERSION = "adp_content_agents_vast_run.v1"
PAID_ATTEMPT_AUTHORITY_SCHEMA = "adp_content_agents_paid_attempt_authority.v1"
EXECUTION_READINESS_SCHEMA = "adp_content_agents_execution_readiness.v1"
SOURCE_COMMIT = "36dbf3f274f8e256637230a05a085853f65cc175"
SOURCE_TREE = "d36ddaed4c3ea44ab81c9f8178ab40d2eb0f8fe3"
SOURCE_VERSION = "0.5.2"
CONTENT_LLM_MODEL = OPENAI_TEXT_MODEL
CONTENT_LLM_REASONING_EFFORT = OPENAI_REASONING_EFFORT
CONTENT_IMAGE_MODEL = OPENAI_IMAGE_MODEL
DEFAULT_IMAGE = (
    "docker.io/nvidia/cuda@"
    "sha256:cff3a0d82d2c2b47bab252d67fa9b34a20ef4c50781d98501b5c7367ea9afd10"
)
REFERENCE_IMAGE_SHA256 = (
    "sha256:80954198df572d782e095d8670e0d4e8ceea530c8fe53c8476a487d1aebe137f"
)
MATCH_V2_RECEIPT_RELATIVE_PATH = (
    "docs/arm_decision_proof_v1/manifests/"
    "adp009a_840313_canned_beverage_match_v2_receipt.v1.json"
)
MATCH_V2_REPLACEMENT_RECEIPT_RELATIVE_PATH = (
    "docs/arm_decision_proof_v1/manifests/"
    "adp009b_simready_replacement_match_v2_receipt.v1.json"
)
MATCH_V2_HUMAN_REVIEW_RELATIVE_PATH = (
    "docs/arm_decision_proof_v1/manifests/"
    "adp009b_simready_match_v2_human_review_receipt.v1.json"
)
ARTICULATED_V1_MANIFEST_RELATIVE_PATH = (
    "docs/arm_decision_proof_v1/manifests/"
    "second_scene_840796_deterministic_simready_candidate.v1.json"
)
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/content-agents"
CONTENT_AGENTS_INSTANCE_LABEL_PREFIX = "blueprint-adp-content-agents-"
_VAST_MUTATION_ENV = (
    "BLUEPRINT_ALLOW_VAST_API_CALLS",
    "BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH",
)
_VAST_SINGLE_ATTEMPT_ENV = "BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS"
_FORWARDED_SECRET_NAMES = (
    "OPENAI_API_KEY",
)
_APPROVED_REFERENCE_RIGHTS_STATES = frozenset(
    {
        "accepted_for_declared_local_import_only",
        "approved_for_declared_use",
        "approved_for_internal_use",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _under(path: Path, root: Path, *, error: str) -> Path:
    resolved = path.expanduser().resolve()
    root = root.expanduser().resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(error)
    return resolved


def _verified_canonical_receipt(path: Path, *, error: str) -> dict[str, Any]:
    receipt = _read_json(path)
    supplied = receipt.get("receipt_digest")
    if not receipt or supplied != canonical_digest(receipt, digest_field="receipt_digest"):
        raise ValueError(error)
    return receipt


def _file_record_matches_current_bytes(record: Any) -> bool:
    if not isinstance(record, Mapping):
        return False
    path = Path(str(record.get("path") or "")).expanduser().resolve()
    return (
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == int(record.get("size_bytes", -1))
        and _sha256(path) == record.get("sha256")
    )


def materialize_content_agents_execution_readiness(
    *,
    content_agents_bundle_matrix: Mapping[str, Any],
    output_path: str | Path,
    config_preflight_receipts: (
        Mapping[str, str | Path | Sequence[str | Path]] | None
    ) = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Record exact local readiness and the next missing paid-execution inputs.

    This is a no-provider receipt. It validates prepared bundle bytes and
    records whether each candidate has the local config-preflight and explicit
    paid-attempt authority required before `--execute` can be admitted.
    """

    matrix = dict(content_agents_bundle_matrix)
    if matrix.get("schema_version") == AGENT_CAD_BUNDLE_MATRIX_V2_SCHEMA:
        try:
            matrix = validate_agent_cad_content_agents_bundle_matrix(matrix)
        except Exception as exc:
            raise ValueError(
                "adp_content_agents_readiness_bundle_matrix_invalid"
            ) from exc
    elif (
        matrix.get("schema_version")
        != "third_scene_agent_cad_content_agents_bundle_matrix.v1"
        or matrix.get("receipt_digest")
        != canonical_digest(matrix, digest_field="receipt_digest")
    ):
        raise ValueError("adp_content_agents_readiness_bundle_matrix_invalid")
    if matrix.get("input_variant") != "agent_cad_v1":
        raise ValueError("adp_content_agents_readiness_bundle_matrix_invalid")
    if matrix.get("claim_boundary") != {
        "content_agents_bundles_built": True,
        "exact_entrypoint_rehearsed": True,
        "content_agents_executed": False,
        "simready_qualified": False,
        "native_simulator_import_qualified": False,
        "physical_equivalence": False,
    }:
        raise ValueError("adp_content_agents_readiness_claim_boundary_invalid")
    capacity = matrix.get("replacement_object_capacity")
    items = matrix.get("items")
    if (
        not isinstance(capacity, Mapping)
        or capacity.get("minimum") != 1
        or capacity.get("maximum") != 5
        or not isinstance(capacity.get("sealed_slots"), int)
        or not 1 <= capacity["sealed_slots"] <= 5
        or not isinstance(items, list)
        or not items
        or len(items) > 10
        or matrix.get("candidate_count") != len(items)
    ):
        raise ValueError("adp_content_agents_readiness_matrix_capacity_invalid")
    expected_backends = set(ADMITTED_BACKENDS)
    sealed_slots = int(capacity["sealed_slots"])
    if matrix.get("candidate_count") != sealed_slots * len(expected_backends):
        raise ValueError("adp_content_agents_readiness_matrix_capacity_invalid")
    slot_rows: dict[int, dict[str, Any]] = {}
    seen_item_keys: set[tuple[int, str]] = set()
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("adp_content_agents_readiness_item_invalid")
        slot = item.get("replacement_slot")
        backend_id = str(item.get("cad_agent_backend_id") or "")
        task_id = str(item.get("task_id") or "")
        asset_id = str(item.get("asset_id") or "")
        if (
            not isinstance(slot, int)
            or isinstance(slot, bool)
            or slot < 1
            or slot > sealed_slots
            or backend_id not in expected_backends
            or not task_id
            or not asset_id
        ):
            raise ValueError("adp_content_agents_readiness_matrix_item_identity_invalid")
        item_key = (slot, backend_id)
        if item_key in seen_item_keys:
            raise ValueError("adp_content_agents_readiness_matrix_item_identity_invalid")
        seen_item_keys.add(item_key)
        prior = slot_rows.setdefault(
            slot,
            {"task_id": task_id, "asset_id": asset_id, "backends": set()},
        )
        if prior["task_id"] != task_id or prior["asset_id"] != asset_id:
            raise ValueError("adp_content_agents_readiness_matrix_item_identity_invalid")
        prior["backends"].add(backend_id)
    if set(slot_rows) != set(range(1, sealed_slots + 1)) or any(
        row["backends"] != expected_backends for row in slot_rows.values()
    ):
        raise ValueError("adp_content_agents_readiness_matrix_item_identity_invalid")

    preflights = dict(config_preflight_receipts or {})
    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("adp_content_agents_readiness_item_invalid")
        bundle_record = item.get("bundle")
        receipt_record = item.get("bundle_receipt")
        if not isinstance(bundle_record, Mapping) or not isinstance(
            receipt_record, Mapping
        ):
            raise ValueError("adp_content_agents_readiness_bundle_record_invalid")
        bundle_path = Path(str(bundle_record.get("path") or "")).expanduser().resolve()
        receipt_path = Path(str(receipt_record.get("path") or "")).expanduser().resolve()
        if (
            not bundle_path.is_file()
            or not receipt_path.is_file()
            or _sha256(bundle_path) != bundle_record.get("sha256")
            or _sha256(receipt_path) != receipt_record.get("sha256")
            or bundle_path.stat().st_size != int(bundle_record.get("size_bytes", -1))
            or receipt_path.stat().st_size
            != int(receipt_record.get("size_bytes", -1))
        ):
            raise ValueError("adp_content_agents_readiness_bundle_bytes_invalid")
        receipt = _read_json(receipt_path)
        if (
            receipt.get("schema_version")
            != "adp_content_agents_provider_bundle.v1"
            or receipt.get("status") != "ready"
            or receipt.get("bundle_path") != str(bundle_path)
            or receipt.get("bundle_sha256") != bundle_record.get("sha256")
            or receipt.get("source_commit") != SOURCE_COMMIT
            or receipt.get("source_tree") != SOURCE_TREE
            or receipt.get("container_image") != DEFAULT_IMAGE
            or receipt.get("retry_cap") != 0
            or receipt.get("blockers") not in ([], None)
            or provider_bundle_rehearsal_blockers(
                receipt.get("exact_bundle_entrypoint_rehearsal"),
                bundle_sha256=str(bundle_record.get("sha256") or ""),
                entrypoint_relative_path=(
                    "provider_runtime/run_adp_content_agents_provider_runtime.sh"
                ),
            )
        ):
            raise ValueError("adp_content_agents_readiness_bundle_receipt_invalid")
        bindings = receipt.get("input_variant_bindings")
        if not isinstance(bindings, Mapping) or receipt.get("input_variant") != "agent_cad_v1":
            raise ValueError("adp_content_agents_readiness_bundle_binding_invalid")
        input_normalization = receipt.get("input_usd_normalization")
        mesh_count = item.get("mesh_count")
        if (
            not isinstance(input_normalization, Mapping)
            or not isinstance(mesh_count, int)
            or isinstance(mesh_count, bool)
            or mesh_count < 1
            or input_normalization.get("mesh_count") != mesh_count
        ):
            raise ValueError("adp_content_agents_readiness_mesh_count_mismatch")
        expected_binding_fields = {
            "task_id": item.get("task_id"),
            "asset_id": item.get("asset_id"),
            "replacement_slot": item.get("replacement_slot"),
            "cad_agent_backend_id": item.get("cad_agent_backend_id"),
            "cad_agent_output_receipt_digest": item.get(
                "cad_agent_output_receipt_digest"
            ),
            "cad_agent_request_digest": item.get("cad_agent_request_digest"),
            "cad_agent_reference_manifest_object_digest": item.get(
                "cad_agent_reference_manifest_object_digest"
            ),
            "mesh_projection_receipt_digest": item.get(
                "mesh_projection_receipt_digest"
            ),
            "mesh_packet_digest": item.get("mesh_packet_digest"),
            "candidate_step_sha256": item.get("candidate_step_sha256"),
        }
        if any(
            bindings.get(field) != expected_value
            for field, expected_value in expected_binding_fields.items()
        ):
            raise ValueError("adp_content_agents_readiness_bundle_binding_mismatch")
        if not all(
            _file_record_matches_current_bytes(bindings.get(field))
            for field in (
                "cad_agent_output_manifest",
                "cad_agent_reference_manifest",
                "cad_agent_selected_reference_image",
                "mesh_projection_receipt",
            )
        ):
            raise ValueError(
                "adp_content_agents_readiness_bundle_binding_file_mismatch"
            )
        reference_image_records = bindings.get("cad_agent_reference_images")
        reference_image_sha256s = receipt.get("reference_image_sha256s")
        runtime_reference_bindings = receipt.get("runtime_reference_image_bindings")
        if (
            not isinstance(reference_image_records, list)
            or not reference_image_records
            or not isinstance(reference_image_sha256s, list)
            or not isinstance(runtime_reference_bindings, list)
            or len(reference_image_records) != len(reference_image_sha256s)
            or len(reference_image_records) != len(runtime_reference_bindings)
            or any(
                not _file_record_matches_current_bytes(record)
                for record in reference_image_records
            )
            or [record.get("sha256") for record in reference_image_records]
            != reference_image_sha256s
            or [record.get("sha256") for record in reference_image_records]
            != [row.get("sha256") for row in runtime_reference_bindings]
            or any(
                not isinstance(row, Mapping)
                or not str(row.get("relative_path") or "").startswith(
                    "input/reference"
                )
                for row in runtime_reference_bindings
            )
        ):
            raise ValueError(
                "adp_content_agents_readiness_reference_image_bindings_invalid"
            )
        projection_receipt_path = Path(
            str((bindings.get("mesh_projection_receipt") or {}).get("path") or "")
        ).expanduser().resolve()
        projection_receipt = _read_json(projection_receipt_path)
        if (
            projection_receipt.get("schema_version")
            != "cad_agent_mesh_usd_projection.v1"
            or projection_receipt.get("mesh_count") != mesh_count
            or projection_receipt.get("receipt_digest")
            != item.get("mesh_projection_receipt_digest")
        ):
            raise ValueError("adp_content_agents_readiness_mesh_count_mismatch")
        key = (
            f"{item.get('task_id')}|{item.get('replacement_slot')}|"
            f"{item.get('cad_agent_backend_id')}"
        )
        blockers = ["content_agents_paid_attempt_authority_missing"]
        preflight_path_value = preflights.get(key)
        preflight_record: dict[str, Any] | None = None
        local_preflight_record: dict[str, Any] | None = None
        static_preflight_record: dict[str, Any] | None = None
        if preflight_path_value is None:
            blockers.append("content_agents_static_config_preflight_missing")
            blockers.append("content_agents_local_docker_config_preflight_missing")
            blockers.append("content_agents_paid_model_access_preflight_missing")
        else:
            if isinstance(preflight_path_value, (str, Path)):
                preflight_path_values = [preflight_path_value]
            elif isinstance(preflight_path_value, Sequence):
                preflight_path_values = list(preflight_path_value)
            else:
                preflight_path_values = []
            if not preflight_path_values:
                blockers.append("content_agents_config_preflight_invalid")
            for candidate_path_value in preflight_path_values:
                if not isinstance(candidate_path_value, (str, Path)):
                    blockers.append("content_agents_config_preflight_invalid")
                    continue
                preflight_path = Path(candidate_path_value).expanduser().resolve()
                preflight = _read_json(preflight_path)
                base_preflight_valid = (
                    preflight_path.is_file()
                    and preflight.get("bundle_receipt_sha256") == _sha256(receipt_path)
                    and preflight.get("bundle_sha256") == bundle_record.get("sha256")
                    and preflight.get("receipt_digest")
                    == canonical_digest(preflight, digest_field="receipt_digest")
                )
                schema = preflight.get("schema_version")
                record = {
                    "path": str(preflight_path),
                    "sha256": _sha256(preflight_path),
                    "size_bytes": preflight_path.stat().st_size
                    if preflight_path.is_file()
                    else 0,
                    "receipt_digest": preflight.get("receipt_digest"),
                }
                if (
                    base_preflight_valid
                    and schema == "adp_content_agents_bundle_config_preflight.v2"
                    and preflight.get("status") == "passed"
                ):
                    preflight_record = record
                elif (
                    base_preflight_valid
                    and schema == "adp_content_agents_local_bundle_config_preflight.v1"
                    and preflight.get("status")
                    == "local_passed_paid_model_access_not_checked"
                    and preflight.get("docker_network_disabled") is True
                    and preflight.get("paid_model_access_required") is False
                    and preflight.get("provider_mutations_performed") == 0
                    and preflight.get("paid_resource_allocated") is False
                    and preflight.get("blockers")
                    == ["content_agents_paid_model_access_preflight_missing"]
                ):
                    local_preflight_record = record
                elif (
                    base_preflight_valid
                    and schema == "adp_content_agents_local_bundle_config_preflight.v1"
                    and preflight.get("status") == "blocked_local_docker_unavailable"
                    and preflight.get("all_required_dry_runs_executed") is False
                    and preflight.get("docker_executed") is False
                    and preflight.get("docker_network_disabled") is True
                    and preflight.get("paid_model_access_required") is False
                    and preflight.get("provider_mutations_performed") == 0
                    and preflight.get("paid_resource_allocated") is False
                    and isinstance(preflight.get("blockers"), list)
                    and preflight.get("blockers")
                ):
                    local_preflight_record = record
                    blockers.extend(str(blocker) for blocker in preflight["blockers"])
                elif (
                    base_preflight_valid
                    and schema == "adp_content_agents_static_bundle_config_preflight.v1"
                    and preflight.get("status")
                    == "static_passed_docker_and_paid_model_access_not_checked"
                    and preflight.get("docker_executed") is False
                    and preflight.get("paid_model_access_required") is False
                    and preflight.get("provider_mutations_performed") == 0
                    and preflight.get("paid_resource_allocated") is False
                    and preflight.get("blockers")
                    == [
                        "content_agents_local_docker_config_preflight_missing",
                        "content_agents_paid_model_access_preflight_missing",
                    ]
                ):
                    static_preflight_record = record
                else:
                    blockers.append("content_agents_config_preflight_invalid")
            if preflight_record is None:
                blockers.append("content_agents_paid_model_access_preflight_missing")
            if preflight_record is None and local_preflight_record is None:
                blockers.append("content_agents_local_docker_config_preflight_missing")
            if (
                preflight_record is None
                and local_preflight_record is None
                and static_preflight_record is None
            ):
                blockers.append("content_agents_static_config_preflight_missing")
        rows.append(
            {
                "task_id": item.get("task_id"),
                "replacement_slot": item.get("replacement_slot"),
                "asset_id": item.get("asset_id"),
                "cad_agent_backend_id": item.get("cad_agent_backend_id"),
                "bundle": {
                    "path": str(bundle_path),
                    "sha256": bundle_record.get("sha256"),
                    "size_bytes": bundle_path.stat().st_size,
                },
                "bundle_receipt": {
                    "path": str(receipt_path),
                    "sha256": receipt_record.get("sha256"),
                    "size_bytes": receipt_path.stat().st_size,
                },
                "config_preflight": preflight_record,
                "local_config_preflight": local_preflight_record,
                "static_config_preflight": static_preflight_record,
                "local_bundle_ready": True,
                "exact_entrypoint_rehearsed": True,
                "paid_model_access_required_for_execute": True,
                "paid_attempt_authority_required_for_execute": True,
                "execute_admitted": False,
                "provider_mutations_performed": 0,
                "blockers": sorted(set(blockers)),
            }
        )

    receipt: dict[str, Any] = {
        "schema_version": EXECUTION_READINESS_SCHEMA,
        "generated_at": generated_at or utc_now_iso(),
        "status": "blocked_before_paid_execution",
        "input_variant": "agent_cad_v1",
        "candidate_count": len(rows),
        "replacement_object_capacity": dict(capacity),
        "content_agents_bundle_matrix_digest": matrix["receipt_digest"],
        "content_agents_executed": False,
        "paid_resource_allocated": False,
        "provider_mutations_performed": 0,
        "private_or_gated_dataset_bytes_uploaded": False,
        "raw_interiorgs_bytes_uploaded": False,
        "items": sorted(
            rows,
            key=lambda row: (
                str(row["task_id"]),
                int(row["replacement_slot"]),
                str(row["cad_agent_backend_id"]),
            ),
        ),
        "claim_boundary": {
            "content_agents_bundles_built": True,
            "exact_entrypoint_rehearsed": True,
            "content_agents_executed": False,
            "simready_qualified": False,
            "native_simulator_import_qualified": False,
            "physical_equivalence": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(output_path, receipt)
    return receipt


def validate_content_agents_paid_attempt_authority(
    authority: Mapping[str, Any],
    *,
    prepared_bundle: Mapping[str, Any],
    bundle_receipt_sha256: str | None,
    config_preflight: Mapping[str, Any],
    config_preflight_receipt_sha256: str | None,
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    allowed_active_instance_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Bind one explicit paid Content Agents attempt to exact local receipts.

    Bundle construction, config preflight, and allocator dry-runs intentionally
    do not require this grant. It is an execution-only, single-use capability
    layered over the immutable bundle and local preflight receipts.
    """

    value = dict(authority)
    errors: list[str] = []
    structured_allowlist = "active_instance_allowlist" in value
    authority_allowlist = normalize_active_instance_allowlist(
        value.get("active_instance_allowlist", value.get("external_instance_allowlist"))
    )
    expected_allowlist = normalize_active_instance_allowlist(allowed_active_instance_ids)
    if value.get("schema_version") != PAID_ATTEMPT_AUTHORITY_SCHEMA:
        errors.append("schema_invalid")
    if value.get("authority_kind") != "explicit_user_direction_in_current_goal":
        errors.append("authority_kind_invalid")
    if not str(value.get("authority_reference") or "").strip():
        errors.append("authority_reference_invalid")
    if not str(value.get("authorized_by") or "").strip():
        errors.append("authorized_by_invalid")
    if not str(value.get("authorized_on") or "").strip():
        errors.append("authorized_on_invalid")
    if value.get("purpose") != "nvidia_content_agents_advisory_enrichment":
        errors.append("purpose_invalid")
    if value.get("provider") != "vast":
        errors.append("provider_invalid")
    if value.get("paid_compute_authorized") is not True:
        errors.append("paid_compute_not_authorized")
    if value.get("bundle_sha256") != prepared_bundle.get("bundle_sha256"):
        errors.append("bundle_sha256_mismatch")
    if value.get("bundle_receipt_sha256") != bundle_receipt_sha256:
        errors.append("bundle_receipt_sha256_mismatch")
    if value.get("config_preflight_receipt_sha256") != config_preflight_receipt_sha256:
        errors.append("config_preflight_receipt_sha256_mismatch")
    if value.get("config_preflight_receipt_digest") != config_preflight.get(
        "receipt_digest"
    ):
        errors.append("config_preflight_receipt_digest_mismatch")
    if value.get("content_agents_source_commit") != SOURCE_COMMIT:
        errors.append("content_agents_source_commit_mismatch")
    if value.get("content_agents_source_tree") != SOURCE_TREE:
        errors.append("content_agents_source_tree_mismatch")
    if value.get("container_image") != DEFAULT_IMAGE:
        errors.append("container_image_mismatch")
    if value.get("maximum_paid_attempts") != 1:
        errors.append("maximum_paid_attempts_invalid")
    if value.get("maximum_automatic_retries") != 0:
        errors.append("maximum_automatic_retries_invalid")
    if value.get("automatic_paid_retry_authorized") is not False:
        errors.append("automatic_paid_retry_authorized_invalid")
    if value.get("zero_retry") is not True:
        errors.append("zero_retry_invalid")
    if value.get("hard_attempt_spend_cap_usd") != hard_cap_usd:
        errors.append("hard_attempt_spend_cap_mismatch")
    if value.get("maximum_hourly_rate_usd") != max_hourly_rate_usd:
        errors.append("maximum_hourly_rate_mismatch")
    if value.get("maximum_single_resource_ttl_seconds") != hard_ttl_seconds:
        errors.append("maximum_single_resource_ttl_mismatch")
    if value.get("agent_output_is_simready_authority") is not False:
        errors.append("agent_output_authority_claim_invalid")
    if value.get("native_simulator_import_qualified") is not False:
        errors.append("native_simulator_import_claim_invalid")
    if authority_allowlist is None:
        errors.append(
            "active_instance_allowlist_invalid"
            if structured_allowlist
            else "external_instance_allowlist_invalid"
        )
    elif expected_allowlist is None:
        errors.append("allowed_active_instance_ids_invalid")
    elif flatten_active_instance_allowlist(
        authority_allowlist
    ) != flatten_active_instance_allowlist(expected_allowlist):
        errors.append(
            "active_instance_allowlist_mismatch"
            if structured_allowlist
            else "external_instance_allowlist_mismatch"
        )
    elif (metadata_error := active_instance_allowlist_metadata_error(
        value, allowlist=authority_allowlist
    )) is not None:
        errors.append(metadata_error)
    if value.get("authorization_digest") != canonical_digest(
        value, digest_field="authorization_digest"
    ):
        errors.append("authorization_digest_invalid")
    try:
        validate_bound_lane_prior_spend(value, lane="content_agents")
    except ValueError:
        errors.append("prior_spend_reconciliation_invalid")
    try:
        from .content_agents_preallocation_closeout import (
            validate_bound_prior_content_agents_preallocation_attempts,
        )

        validate_bound_prior_content_agents_preallocation_attempts(value)
    except ValueError:
        errors.append("prior_preallocation_attempts_invalid")
    if errors:
        raise ValueError(
            "adp_content_agents_paid_attempt_authority_invalid:"
            + ",".join(sorted(set(errors)))
        )
    return value


def consume_content_agents_paid_attempt_authority_once(
    authority: Mapping[str, Any], *, blueprint_commit: str
) -> dict[str, Any]:
    """Atomically consume a validated grant before object-store/provider mutation."""

    authorization_digest = str(authority.get("authorization_digest") or "")
    if not authorization_digest.startswith("sha256:") or len(authorization_digest) != 71:
        return {
            "status": "blocked",
            "blockers": ["adp_content_agents_paid_attempt_authority_identity_invalid"],
        }
    root = consumption_root()
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root_stat = root.stat()
        if (
            root.is_symlink()
            or root_stat.st_uid != os.getuid()
            or root_stat.st_mode & 0o077
        ):
            return {
                "status": "blocked",
                "blockers": ["adp_content_agents_authority_consumption_root_insecure"],
            }
        identity = authorization_digest.removeprefix("sha256:")
        destination = root / f"content-agents-{identity}.json"
        record = {
            "schema_version": "adp_content_agents_paid_attempt_consumption.v1",
            "authorization_digest": authorization_digest,
            "bundle_sha256": authority.get("bundle_sha256"),
            "config_preflight_receipt_digest": authority.get(
                "config_preflight_receipt_digest"
            ),
            "blueprint_commit": blueprint_commit,
            "consumed_at": utc_now_iso(),
            "maximum_provider_allocations": 1,
            "prior_preallocation_attempt_authority_digests": [
                row.get("attempt_authority_digest")
                for row in authority.get("prior_preallocation_attempts") or []
                if isinstance(row, Mapping)
            ],
            "preallocation_attempt_ordinal": authority.get(
                "preallocation_attempt_ordinal"
            ),
        }
        raw = (
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        temporary = root / f".content-agents-{identity}.{os.getpid()}.tmp"
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, destination)
            directory_descriptor = os.open(root, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            temporary.unlink(missing_ok=True)
    except FileExistsError:
        return {
            "status": "blocked",
            "blockers": ["adp_content_agents_paid_attempt_authority_consumed"],
        }
    except OSError:
        return {
            "status": "blocked",
            "blockers": ["adp_content_agents_authority_consumption_write_failed"],
        }
    return {
        "status": "consumed",
        "authorization_digest": authorization_digest,
        "consumption_record_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "record_location_disclosed": False,
    }


def _default_agent_cad_reference_images(
    agent_cad_output_manifest_path: Path | None,
) -> list[Path]:
    if agent_cad_output_manifest_path is None:
        raise ValueError("adp_content_agents_agent_cad_output_manifest_missing")
    try:
        output = validate_cad_agent_output(
            _read_json(agent_cad_output_manifest_path.expanduser().resolve()),
            verify_files=True,
        )
    except (OSError, SimReadyCadAgentContractError) as exc:
        raise ValueError("adp_content_agents_agent_cad_output_invalid") from exc
    references = ((output.get("request") or {}).get("inputs") or {}).get(
        "reference_images"
    )
    if not isinstance(references, list) or not references:
        raise ValueError("adp_content_agents_agent_cad_reference_missing")
    paths: list[Path] = []
    for record in references:
        if not isinstance(record, Mapping):
            raise ValueError("adp_content_agents_agent_cad_reference_missing")
        path = Path(str(record.get("path") or "")).expanduser().resolve()
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            raise ValueError("adp_content_agents_agent_cad_reference_identity_mismatch")
        paths.append(path)
    return paths


def _exact_file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ValueError("adp_content_agents_paired_target_input_not_host_resident")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _resolve_paired_target_registered_input(
    *,
    repo: Path,
    construction_bindings_path: Path | None,
    task_id: str | None,
    reference_sources: Sequence[Path],
    reference_rights_authority_path: Path | None,
) -> dict[str, Any]:
    """Admit one native-qualified registered asset without relabeling it as CAD.

    The construction binding already joins the registered replacement USD to its
    task freeze and native-import probe.  This adapter selects exactly one row,
    reopens the USD and reference bytes on the bundle-building host, and requires
    one human-issued rights receipt to name every disclosed reference digest.
    """

    if construction_bindings_path is None:
        raise ValueError("adp_content_agents_paired_target_bindings_missing")
    bindings_path = construction_bindings_path.expanduser().resolve()
    try:
        bindings = validate_paired_target_native_construction_bindings(
            _read_json(bindings_path)
        )
    except (
        OSError,
        json.JSONDecodeError,
        PairedTargetNativeConstructionBindingsError,
    ) as exc:
        raise ValueError("adp_content_agents_paired_target_bindings_invalid") from exc
    if (
        bindings_path.is_symlink()
        or not bindings_path.is_file()
        or bindings.get("construction_digest")
        != canonical_digest(bindings, digest_field="construction_digest")
    ):
        raise ValueError("adp_content_agents_paired_target_bindings_invalid")

    selected = [
        dict(row)
        for row in bindings.get("bindings") or []
        if isinstance(row, Mapping) and row.get("task_id") == str(task_id or "")
    ]
    if len(selected) != 1:
        raise ValueError("adp_content_agents_paired_target_task_selection_invalid")
    row = selected[0]
    evidence = row.get("evidence_receipts") or {}
    usd_record = evidence.get("registered_usd")
    if not _file_record_matches_current_bytes(usd_record):
        raise ValueError("adp_content_agents_paired_target_registered_usd_invalid")
    if row.get("replacement_asset_sha256") != usd_record.get("sha256"):
        raise ValueError("adp_content_agents_paired_target_registered_usd_mismatch")
    usd_source = Path(str(usd_record["path"])).expanduser().resolve()

    reference_records = [_exact_file_record(path) for path in reference_sources]
    if not reference_records or any(
        Path(record["path"]).read_bytes()[:8] != b"\x89PNG\r\n\x1a\n"
        for record in reference_records
    ):
        raise ValueError("adp_content_agents_paired_target_reference_invalid")
    if reference_rights_authority_path is None:
        raise ValueError("adp_content_agents_paired_target_reference_rights_missing")
    rights_path = reference_rights_authority_path.expanduser().resolve()
    rights_record = _exact_file_record(rights_path)
    rights = _read_json(rights_path)
    rights_status = str(rights.get("status") or rights.get("reviewer_status") or "")
    rights_scope = str(
        rights.get("use_ceiling") or rights.get("declared_use_scope") or ""
    ).strip()
    authorized_digests = rights.get("authorized_source_sha256")
    if (
        rights.get("schema_version") != RIGHTS_RECEIPT_SCHEMA
        or rights_status not in _APPROVED_REFERENCE_RIGHTS_STATES
        or rights.get("agent_accepted_terms") is True
        or not isinstance(authorized_digests, list)
        or any(record["sha256"] not in authorized_digests for record in reference_records)
    ):
        raise ValueError("adp_content_agents_paired_target_reference_rights_invalid")

    binding_record = _exact_file_record(bindings_path)
    return {
        "usd_source": usd_source,
        "config_sources": {
            f"{agent}_agent.yaml": repo
            / "docs/arm_decision_proof_v1/assets"
            / f"adp009d_content_agents_articulated_{agent}.vast.yaml"
            for agent in ("material", "texture", "physics")
        },
        "reference_image_sha256": reference_records[0]["sha256"],
        "reference_image_sha256s": [record["sha256"] for record in reference_records],
        "reference_image_authority": (
            "exact_rights_authority_bound_reference_bytes_for_registered_candidate_"
            "not_raw_capture_or_physical_truth"
        ),
        "variant": "paired_target_registered_v1",
        "paired_target_construction_bindings": binding_record,
        "paired_target_construction_digest": bindings["construction_digest"],
        "paired_target_scene_id": bindings["scene_id"],
        "paired_target_reference_rights_authority": rights_record,
        "paired_target_reference_rights_status": rights_status,
        "paired_target_reference_rights_scope": rights_scope or None,
        "paired_target_reference_images": reference_records,
        "task_id": row["task_id"],
        "asset_id": row["asset_id"],
        "task_freeze_digest": row["task_freeze_digest"],
        "registered_asset_receipt_digest": row[
            "registered_asset_receipt_digest"
        ],
        "replacement_asset_sha256": row["replacement_asset_sha256"],
        "native_import_probe_result_digest": row[
            "native_import_probe_result_digest"
        ],
        "native_simulator_import_qualified": True,
    }


def _resolve_input_variant(
    *,
    repo: Path,
    evidence_root: Path | None,
    reference_source: Path,
    reference_sources: Sequence[Path] | None = None,
    variant: str,
    agent_cad_output_manifest_path: Path | None = None,
    agent_mesh_projection_receipt_path: Path | None = None,
    paired_target_construction_bindings_path: Path | None = None,
    paired_target_task_id: str | None = None,
    reference_rights_authority_path: Path | None = None,
) -> dict[str, Any]:
    assets = repo / "docs" / "arm_decision_proof_v1" / "assets"
    if variant == "control_v1":
        if _sha256(reference_source) != REFERENCE_IMAGE_SHA256:
            raise ValueError("adp_content_agents_reference_image_identity_mismatch")
        return {
            "usd_source": assets / "adp009a_840313_canned_beverage_control.usda",
            "config_sources": {
                f"{agent}_agent.yaml": assets
                / f"adp009a_content_agents_{agent}.vast.yaml"
                for agent in ("material", "texture", "physics")
            },
            "reference_image_sha256": REFERENCE_IMAGE_SHA256,
            "reference_image_authority": "blueprint_cad_render_not_interiorgs_dataset_bytes",
            "variant": variant,
        }
    if variant == "articulated_v1":
        # The 840796 articulated candidate is scene-derived, so its bytes stay
        # in the evidence root under the recorded rights while the repo keeps
        # only the digest-bound manifest. The Content Agents pass may add
        # SimReady materials and physics priors; it may never re-derive the
        # articulation, so admission requires the statically admitted receipt.
        if evidence_root is None:
            raise ValueError("adp_content_agents_articulated_evidence_root_missing")
        evidence = evidence_root.expanduser().resolve()
        manifest = _verified_canonical_receipt(
            _under(
                repo / ARTICULATED_V1_MANIFEST_RELATIVE_PATH,
                repo,
                error="adp_content_agents_articulated_manifest_outside_repo",
            ),
            error="adp_content_agents_articulated_manifest_invalid",
        )
        authoring = manifest.get("authoring") or {}
        relative = manifest.get("evidence_relative_paths") or {}
        if (
            manifest.get("schema_version")
            != "second_scene_deterministic_simready_candidate.v1"
            or manifest.get("publisher_scene_id") != "840796"
            or authoring.get("status") != "simready_candidate_statically_admitted"
            or not str(authoring.get("task_joint_prim_path") or "")
            or not str(manifest.get("topology_validation_receipt_digest") or "")
            or not str(manifest.get("physics_validation_receipt_digest") or "")
        ):
            raise ValueError(
                "adp_content_agents_articulated_candidate_receipt_not_eligible"
            )
        usd_source = _under(
            evidence / str(relative.get("candidate_usd") or ""),
            evidence,
            error="adp_content_agents_articulated_usd_outside_evidence",
        )
        expected_reference = _under(
            evidence / str(relative.get("reference_image") or ""),
            evidence,
            error="adp_content_agents_articulated_reference_outside_evidence",
        )
        if (
            not usd_source.is_file()
            or _sha256(usd_source) != authoring.get("output_usd_sha256")
            or not expected_reference.is_file()
            or reference_source != expected_reference
            or _sha256(reference_source) != manifest.get("reference_image_sha256")
        ):
            raise ValueError(
                "adp_content_agents_articulated_source_identity_mismatch"
            )
        return {
            "usd_source": usd_source,
            "config_sources": {
                f"{agent}_agent.yaml": assets
                / f"adp009d_content_agents_articulated_{agent}.vast.yaml"
                for agent in ("material", "texture", "physics")
            },
            "reference_image_sha256": manifest["reference_image_sha256"],
            "reference_image_authority": (
                "blueprint_render_of_sage_derived_articulated_candidate_not_"
                "interiorgs_dataset_bytes"
            ),
            "variant": variant,
            "candidate_receipt_digest": manifest["receipt_digest"],
            "task_joint_prim_path": authoring["task_joint_prim_path"],
            "topology_validation_receipt_digest": manifest[
                "topology_validation_receipt_digest"
            ],
            "physics_validation_receipt_digest": manifest[
                "physics_validation_receipt_digest"
            ],
        }
    if variant == "paired_target_registered_v1":
        return _resolve_paired_target_registered_input(
            repo=repo,
            construction_bindings_path=paired_target_construction_bindings_path,
            task_id=paired_target_task_id,
            reference_sources=tuple(reference_sources or [reference_source]),
            reference_rights_authority_path=reference_rights_authority_path,
        )
    if variant == "agent_cad_v1":
        if agent_cad_output_manifest_path is None:
            raise ValueError("adp_content_agents_agent_cad_output_manifest_missing")
        if agent_mesh_projection_receipt_path is None:
            raise ValueError("adp_content_agents_agent_cad_projection_receipt_missing")
        output_path = agent_cad_output_manifest_path.expanduser().resolve()
        projection_path = agent_mesh_projection_receipt_path.expanduser().resolve()
        output_record = {
            "path": str(output_path),
            "size_bytes": output_path.stat().st_size if output_path.is_file() else 0,
            "sha256": _sha256(output_path) if output_path.is_file() else "",
        }
        projection_record = {
            "path": str(projection_path),
            "size_bytes": projection_path.stat().st_size
            if projection_path.is_file()
            else 0,
            "sha256": _sha256(projection_path) if projection_path.is_file() else "",
        }
        try:
            cad_output = validate_cad_agent_output(
                _read_json(output_path), verify_files=True
            )
        except (OSError, SimReadyCadAgentContractError) as exc:
            raise ValueError("adp_content_agents_agent_cad_output_invalid") from exc
        projection = _verified_canonical_receipt(
            projection_path,
            error="adp_content_agents_agent_cad_projection_receipt_invalid",
        )
        if (
            projection.get("schema_version") != "cad_agent_mesh_usd_projection.v1"
            or projection.get("status") != "mesh_working_copy_authored"
            or projection.get("content_agents_input_eligible") is not True
            or projection.get("canonical_simulator_asset") is not False
            or (projection.get("claim_boundary") or {}).get(
                "deterministic_format_conversion_only"
            )
            is not True
            or (projection.get("claim_boundary") or {}).get("collision_authority")
            is not False
            or (projection.get("claim_boundary") or {}).get("physics_authority")
            is not False
            or (cad_output.get("artifacts") or {}).get("step", {}).get("sha256")
            != (projection.get("step") or {}).get("sha256")
        ):
            raise ValueError(
                "adp_content_agents_agent_cad_projection_receipt_not_eligible"
            )
        usd_record = projection.get("output_usd") or {}
        usd_source = Path(str(usd_record.get("path") or "")).expanduser().resolve()
        reference_records = (
            ((cad_output.get("request") or {}).get("inputs") or {}).get(
                "reference_images"
            )
            or []
        )
        request_inputs = (cad_output.get("request") or {}).get("inputs") or {}
        reference_manifest_record = request_inputs.get("reference_manifest")
        reference_manifest_object_digest = request_inputs.get(
            "reference_manifest_object_digest"
        )
        expected_reference_records: list[Mapping[str, Any]] = []
        expected_reference_identities: set[tuple[str, str]] = set()
        for record in reference_records:
            if not isinstance(record, Mapping):
                continue
            record_path = Path(str(record.get("path") or "")).expanduser().resolve()
            record_sha = str(record.get("sha256") or "")
            if not record_path.is_file() or _sha256(record_path) != record_sha:
                continue
            expected_reference_records.append(record)
            expected_reference_identities.add((str(record_path), record_sha))
        provided_reference_identities = {
            (str(path.expanduser().resolve()), _sha256(path.expanduser().resolve()))
            for path in (reference_sources or [reference_source])
            if path.expanduser().resolve().is_file()
        }
        selected_reference_record = (
            expected_reference_records[0] if expected_reference_records else None
        )
        if (
            not usd_source.is_file()
            or _sha256(usd_source) != usd_record.get("sha256")
            or not isinstance(reference_records, list)
            or not expected_reference_records
            or provided_reference_identities != expected_reference_identities
        ):
            raise ValueError("adp_content_agents_agent_cad_source_identity_mismatch")
        if (
            not isinstance(reference_manifest_record, Mapping)
            or not _file_record_matches_current_bytes(reference_manifest_record)
            or not isinstance(reference_manifest_object_digest, str)
            or not reference_manifest_object_digest.startswith("sha256:")
        ):
            raise ValueError("adp_content_agents_agent_cad_reference_binding_invalid")
        mesh_prim_paths = projection.get("mesh_prim_paths")
        if (
            not isinstance(mesh_prim_paths, list)
            or not mesh_prim_paths
            or any(
                not isinstance(path, str) or not path.startswith("/Asset/links/")
                for path in mesh_prim_paths
            )
        ):
            raise ValueError("adp_content_agents_agent_cad_mesh_scope_invalid")
        backend = (cad_output.get("request") or {}).get("backend") or {}
        return {
            "usd_source": usd_source,
            "config_sources": {
                f"{agent}_agent.yaml": assets
                / f"adp009d_content_agents_articulated_{agent}.vast.yaml"
                for agent in ("material", "texture", "physics")
            },
            "reference_image_sha256": _sha256(reference_source),
            "reference_image_sha256s": [
                str(record["sha256"]) for record in expected_reference_records
            ],
            "reference_image_authority": (
                "rights_admitted_observed_reference_image_for_agent_authored_"
                "cad_candidate_not_raw_interiorgs_dataset_bytes"
            ),
            "variant": variant,
            "cad_agent_output_manifest": output_record,
            "cad_agent_output_receipt_digest": cad_output["receipt_digest"],
            "cad_agent_request_digest": cad_output["request_digest"],
            "cad_agent_backend_id": backend.get("backend_id"),
            "cad_agent_execution_mode": backend.get("execution_mode"),
            "cad_agent_reference_manifest": dict(reference_manifest_record),
            "cad_agent_reference_manifest_object_digest": (
                reference_manifest_object_digest
            ),
            "cad_agent_selected_reference_image": dict(selected_reference_record),
            "cad_agent_reference_images": [
                dict(record) for record in expected_reference_records
            ],
            "mesh_projection_receipt": projection_record,
            "mesh_projection_receipt_digest": projection["receipt_digest"],
            "mesh_packet_digest": projection["packet_digest"],
            "mesh_prim_paths": sorted(mesh_prim_paths),
            "default_material_path": projection["default_material_path"],
            "candidate_step_sha256": (projection.get("step") or {}).get("sha256"),
            "task_id": (cad_output.get("request") or {}).get("task_id"),
            "asset_id": (cad_output.get("request") or {}).get("asset_id"),
            "replacement_slot": (cad_output.get("request") or {}).get(
                "replacement_slot"
            ),
        }
    if variant != "match_v2":
        raise ValueError("adp_content_agents_input_variant_invalid")
    if evidence_root is None:
        raise ValueError("adp_content_agents_match_v2_evidence_root_missing")
    evidence = evidence_root.expanduser().resolve()
    control_path = _under(
        repo / MATCH_V2_RECEIPT_RELATIVE_PATH,
        repo,
        error="adp_content_agents_match_v2_control_receipt_outside_repo",
    )
    control = _verified_canonical_receipt(
        control_path, error="adp_content_agents_match_v2_control_receipt_invalid"
    )
    checks = control.get("checks") or {}
    if (
        control.get("control_id")
        != "adp009a-840313-canned-beverage-multiview-match-v2"
        or control.get("status") != "prepared_for_independent_validation"
        or checks.get("cad_inspection_passed") is not True
        or checks.get("target_dimensions_derived_not_caller_asserted") is not True
        or (control.get("visual_match_evidence") or {}).get(
            "projected_scale_and_pose_gate_passed"
        )
        is not True
    ):
        raise ValueError("adp_content_agents_match_v2_control_receipt_not_eligible")
    usd = control.get("usd") or {}
    usd_source = _under(
        repo / str(usd.get("relative_path") or ""),
        repo,
        error="adp_content_agents_match_v2_usd_outside_repo",
    )
    snapshot = (control.get("cad_evidence") or {}).get("snapshot") or {}
    expected_reference = _under(
        evidence / str(snapshot.get("relative_path") or ""),
        evidence,
        error="adp_content_agents_match_v2_reference_outside_evidence",
    )
    if (
        not usd_source.is_file()
        or _sha256(usd_source) != usd.get("sha256")
        or not expected_reference.is_file()
        or reference_source != expected_reference
        or _sha256(reference_source) != snapshot.get("sha256")
    ):
        raise ValueError("adp_content_agents_match_v2_source_identity_mismatch")
    replacement = _verified_canonical_receipt(
        _under(
            repo / MATCH_V2_REPLACEMENT_RECEIPT_RELATIVE_PATH,
            repo,
            error="adp_content_agents_match_v2_replacement_receipt_outside_repo",
        ),
        error="adp_content_agents_match_v2_replacement_receipt_invalid",
    )
    human_review = _verified_canonical_receipt(
        _under(
            repo / MATCH_V2_HUMAN_REVIEW_RELATIVE_PATH,
            repo,
            error="adp_content_agents_match_v2_human_review_outside_repo",
        ),
        error="adp_content_agents_match_v2_human_review_invalid",
    )
    if (
        replacement.get("status") != "composed_static_candidate"
        or (replacement.get("bindings") or {}).get("simready_control_receipt_digest")
        != control.get("receipt_digest")
        or human_review.get("status") != "human_accepted_for_native_validation"
        or human_review.get("technical_admission") is not False
        or (human_review.get("artifact_chain") or {}).get(
            "replacement_receipt_digest"
        )
        != replacement.get("receipt_digest")
    ):
        raise ValueError("adp_content_agents_match_v2_approval_chain_invalid")
    return {
        "usd_source": usd_source,
        "config_sources": {
            f"{agent}_agent.yaml": assets
            / f"adp009a_content_agents_{agent}.vast.yaml"
            for agent in ("material", "texture", "physics")
        },
        "reference_image_sha256": snapshot["sha256"],
        "reference_image_authority": (
            "blueprint_cad_snapshot_bound_to_human_approved_match_v2_not_"
            "interiorgs_dataset_bytes"
        ),
        "variant": variant,
        "control_receipt_digest": control["receipt_digest"],
        "replacement_receipt_digest": replacement["receipt_digest"],
        "human_review_receipt_digest": human_review["receipt_digest"],
        "replacement_receipt": replacement,
    }


def _content_agents_execution_route_binding(
    *, route_path: Path | None, variant: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Bind a paid bundle to one object-specific Codex-first route.

    A VAST bundle is only meaningful when the selected object retains the one
    released NVIDIA pipeline capability.  Codex-only review work must use the
    local route and is rejected here before a provider bundle is created.
    """

    if route_path is None:
        return None
    if variant.get("variant") != "agent_cad_v1":
        raise ValueError("adp_content_agents_execution_route_variant_invalid")
    path = route_path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError("adp_content_agents_execution_route_file_invalid")
    try:
        route = validate_content_agents_execution_route(_read_json(path))
        requires_nvidia, codex_capabilities, nvidia_capabilities = (
            nvidia_content_agents_required(
                route,
                replacement_slot=int(variant.get("replacement_slot")),
                task_id=str(variant.get("task_id") or ""),
                asset_id=str(variant.get("asset_id") or ""),
                source_binding_digest=str(
                    variant.get("cad_agent_output_receipt_digest") or ""
                ),
            )
        )
    except (ContentAgentsExecutionRouteError, TypeError, ValueError) as exc:
        raise ValueError("adp_content_agents_execution_route_binding_invalid") from exc
    if not requires_nvidia:
        raise ValueError(
            "adp_content_agents_vast_bundle_not_required_for_codex_local_route"
        )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "route_digest": route["route_digest"],
        "codex_local_capabilities": codex_capabilities,
        "nvidia_content_agents_capabilities": nvidia_capabilities,
    }


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_executable(path: Path, source: Path) -> None:
    shutil.copy2(source, path)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _materialize_articulated_content_agents_input(
    source: Path, destination: Path
) -> dict[str, Any]:
    """Normalize the articulated candidate for NVIDIA 0.5.2 without re-deriving it.

    The can control needs a grasp-curve extent fix and a render-purpose clear on
    one visual. The articulated candidate has neither; what it does need is
    every mesh readable at default purpose so the agents' bounds and dataset
    stages work, with the articulation left exactly as authored.
    """

    shutil.copy2(source, destination)
    stage = Usd.Stage.Open(str(destination))
    if stage is None or not stage.GetDefaultPrim().IsValid():
        raise ValueError("adp_content_agents_input_default_prim_invalid")
    cleared: list[str] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        if mesh.ComputePurpose() != UsdGeom.Tokens.default_:
            mesh.GetPurposeAttr().Clear()
            cleared.append(str(prim.GetPath()))
    stage.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(destination))
    if reopened is None:
        raise ValueError("adp_content_agents_input_reopen_failed")
    joints = [prim for prim in reopened.Traverse() if prim.IsA(UsdPhysics.Joint)]
    roots = [
        prim
        for prim in reopened.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    rigid_bodies = [
        prim for prim in reopened.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bounds = cache.ComputeWorldBound(reopened.GetDefaultPrim()).ComputeAlignedRange()
    if bounds.IsEmpty() or not joints or len(roots) != 1:
        raise ValueError("adp_content_agents_input_articulation_invalid")
    for prim in reopened.Traverse():
        if prim.IsA(UsdGeom.Mesh) and UsdGeom.Mesh(prim).ComputePurpose() != (
            UsdGeom.Tokens.default_
        ):
            raise ValueError("adp_content_agents_input_default_purpose_bbox_invalid")
    return {
        "source_input_usd_sha256": _sha256(source),
        "normalized_input_usd_sha256": _sha256(destination),
        "transformations": [
            "clear_non_default_mesh_purposes_for_nvidia_0_5_2_bbox",
        ],
        "cleared_purpose_prims": sorted(cleared),
        "default_purpose_bbox_nonempty": True,
        "articulation_preserved": True,
        "joint_count": len(joints),
        "rigid_body_count": len(rigid_bodies),
        "articulation_root_count": len(roots),
    }


def _materialize_paired_target_registered_input(
    source: Path, destination: Path
) -> dict[str, Any]:
    """Create a non-authoritative Content Agents working copy of registered USD."""

    shutil.copy2(source, destination)
    stage = Usd.Stage.Open(str(destination))
    if stage is None or not stage.GetDefaultPrim().IsValid():
        raise ValueError("adp_content_agents_input_default_prim_invalid")
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    if not meshes:
        raise ValueError("adp_content_agents_paired_target_mesh_missing")
    cleared: list[str] = []
    for prim in meshes:
        mesh = UsdGeom.Mesh(prim)
        if mesh.ComputePurpose() != UsdGeom.Tokens.default_:
            mesh.GetPurposeAttr().Clear()
            cleared.append(str(prim.GetPath()))
    default_path = str(stage.GetDefaultPrim().GetPath()).rstrip("/")
    material_path = f"{default_path}/Looks/content_agents_advisory"
    material = UsdShade.Material.Define(stage, material_path)
    for prim in meshes:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    stage.GetRootLayer().Save()

    reopened = Usd.Stage.Open(str(destination))
    if reopened is None or not reopened.GetDefaultPrim().IsValid():
        raise ValueError("adp_content_agents_input_reopen_failed")
    normalized_meshes = [
        prim for prim in reopened.Traverse() if prim.IsA(UsdGeom.Mesh)
    ]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bounds = cache.ComputeWorldBound(reopened.GetDefaultPrim()).ComputeAlignedRange()
    if bounds.IsEmpty() or any(
        UsdGeom.Mesh(prim).ComputePurpose() != UsdGeom.Tokens.default_
        for prim in normalized_meshes
    ):
        raise ValueError("adp_content_agents_input_default_purpose_bbox_invalid")
    joints = [prim for prim in reopened.Traverse() if prim.IsA(UsdPhysics.Joint)]
    rigid_bodies = [
        prim for prim in reopened.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    roots = [
        prim
        for prim in reopened.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if (joints and (len(roots) != 1 or len(rigid_bodies) < 2)) or (
        not joints and len(roots) > 1
    ):
        raise ValueError("adp_content_agents_input_articulation_invalid")
    return {
        "source_input_usd_sha256": _sha256(source),
        "normalized_input_usd_sha256": _sha256(destination),
        "transformations": [
            "copy_native_qualified_registered_usd_without_mutating_canonical_bytes",
            "clear_non_default_mesh_purposes_for_nvidia_0_5_2_bbox",
            "bind_advisory_working_copy_material_without_geometry_authority",
        ],
        "cleared_purpose_prims": sorted(cleared),
        "default_purpose_bbox_nonempty": True,
        "paired_target_registered_working_copy": True,
        "mesh_count": len(normalized_meshes),
        "mesh_prim_paths": sorted(str(prim.GetPath()) for prim in normalized_meshes),
        "default_material_path": material_path,
        "articulation_preserved": True,
        "joint_count": len(joints),
        "rigid_body_count": len(rigid_bodies),
        "articulation_root_count": len(roots),
    }


def _materialize_content_agents_input(
    source: Path, destination: Path, *, variant: str = "control_v1"
) -> dict[str, Any]:
    """Derive the exact NVIDIA-compatible USD without mutating canonical bytes."""

    if variant == "paired_target_registered_v1":
        return _materialize_paired_target_registered_input(source, destination)
    if variant == "agent_cad_v1":
        shutil.copy2(source, destination)
        stage = Usd.Stage.Open(str(destination))
        if stage is None or stage.GetDefaultPrim().GetPath() != "/Asset":
            raise ValueError("adp_content_agents_input_default_prim_invalid")
        meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
        joints = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]
        rigid_bodies = [
            prim
            for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        roots = [
            prim
            for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bounds = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
        if (
            not meshes
            or bounds.IsEmpty()
            or joints
            or rigid_bodies
            or roots
            or any(UsdGeom.Mesh(prim).ComputePurpose() != UsdGeom.Tokens.default_ for prim in meshes)
        ):
            raise ValueError("adp_content_agents_input_agent_cad_mesh_invalid")
        return {
            "source_input_usd_sha256": _sha256(source),
            "normalized_input_usd_sha256": _sha256(destination),
            "transformations": [
                "copy_agent_authored_cad_mesh_working_copy_without_geometry_generation",
            ],
            "default_purpose_bbox_nonempty": True,
            "agent_cad_mesh_working_copy": True,
            "mesh_count": len(meshes),
            "mesh_prim_paths": sorted(str(prim.GetPath()) for prim in meshes),
            "articulation_preserved": True,
            "joint_count": 0,
            "rigid_body_count": 0,
            "articulation_root_count": 0,
        }
    if variant == "articulated_v1":
        return _materialize_articulated_content_agents_input(source, destination)
    shutil.copy2(source, destination)
    stage = Usd.Stage.Open(str(destination))
    if stage is None or stage.GetDefaultPrim().GetPath() != "/canned_beverage":
        raise ValueError("adp_content_agents_input_default_prim_invalid")
    visual = UsdGeom.Mesh.Get(stage, "/canned_beverage/visuals/body")
    if not visual or visual.ComputePurpose() != UsdGeom.Tokens.render:
        raise ValueError("adp_content_agents_input_visual_purpose_invalid")
    visual.GetPurposeAttr().Clear()
    grasp = UsdGeom.BasisCurves.Get(stage, "/canned_beverage/grasp_identifier_01")
    computed_extent = UsdGeom.Boundable.ComputeExtentFromPlugins(
        grasp, Usd.TimeCode.Default()
    )
    if not computed_extent:
        raise ValueError("adp_content_agents_input_grasp_extent_unavailable")
    grasp.GetExtentAttr().Set(computed_extent)
    stage.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(destination))
    if reopened is None:
        raise ValueError("adp_content_agents_input_reopen_failed")
    normalized_visual = UsdGeom.Mesh.Get(reopened, "/canned_beverage/visuals/body")
    joints = [prim for prim in reopened.Traverse() if prim.IsA(UsdPhysics.Joint)]
    rigid_bodies = [
        prim for prim in reopened.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    roots = [
        prim
        for prim in reopened.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    bounds = cache.ComputeWorldBound(normalized_visual.GetPrim()).ComputeAlignedRange()
    if normalized_visual.ComputePurpose() != UsdGeom.Tokens.default_ or bounds.IsEmpty():
        raise ValueError("adp_content_agents_input_default_purpose_bbox_invalid")
    return {
        "source_input_usd_sha256": _sha256(source),
        "normalized_input_usd_sha256": _sha256(destination),
        "transformations": [
            "clear_visual_render_purpose_to_usd_default_for_nvidia_0_5_2_bbox",
            "recompute_grasp_identifier_extent_from_curve_width",
        ],
        "visual_prim": "/canned_beverage/visuals/body",
        "visual_purpose": "default",
        "default_purpose_bbox_nonempty": True,
        "articulation_preserved": not joints and not roots,
        "joint_count": len(joints),
        "rigid_body_count": len(rigid_bodies),
        "articulation_root_count": len(roots),
    }


def _derive_joint_agent_plan(
    *, input_variant: str, input_normalization: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive Joint Agent applicability from the normalized USD inventory."""

    joint_count = input_normalization.get("joint_count")
    rigid_body_count = input_normalization.get("rigid_body_count")
    root_count = input_normalization.get("articulation_root_count")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (joint_count, rigid_body_count, root_count)
    ):
        raise ValueError("adp_content_agents_joint_agent_inventory_invalid")
    if joint_count:
        if root_count != 1 or rigid_body_count < 2:
            raise ValueError("adp_content_agents_joint_agent_articulation_invalid")
        reason = "preexisting_articulation_preserved_by_enrichment_pass"
        single_rigid_body = False
    else:
        if input_variant == "agent_cad_v1":
            if root_count != 0 or rigid_body_count != 0:
                raise ValueError("adp_content_agents_joint_agent_mesh_input_invalid")
            reason = "agent_cad_mesh_working_copy_has_no_articulation_task"
            single_rigid_body = False
        elif input_variant == "paired_target_registered_v1":
            if root_count != 0 or rigid_body_count not in {0, 1}:
                raise ValueError("adp_content_agents_joint_agent_registered_input_invalid")
            reason = "paired_target_registered_candidate_has_no_articulation_task"
            single_rigid_body = rigid_body_count == 1
        elif root_count != 0 or rigid_body_count != 1:
            raise ValueError("adp_content_agents_joint_agent_rigid_input_invalid")
        else:
            reason = "single_rigid_body_has_no_articulation_task"
            single_rigid_body = True
    return {
        "planned": False,
        "executed_by_content_agents_bundle": False,
        "reason": reason,
        "input_variant": input_variant,
        "input_joint_count": joint_count,
        "input_rigid_body_count": rigid_body_count,
        "input_articulation_root_count": root_count,
        "joint_agent_inapplicable_single_rigid_body": single_rigid_body,
    }


def _validate_remote_configs(
    *, source: Path, config_sources: Mapping[str, Path]
) -> None:
    payloads = {
        name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for name, path in config_sources.items()
    }
    material_path = source / "apps/material_agent/data/materials/material_libs_default/materials.yaml"
    material = dict(payloads.get("material_agent.yaml") or {})
    texture = dict(payloads.get("texture_agent.yaml") or {})
    physics = dict(payloads.get("physics_agent.yaml") or {})
    texture_config = dict(texture.get("texture") or {})
    # Exactly one material spec per config; its key is scene-specific.
    texture_material_specs = dict(texture.get("material_textures") or {})
    texture_spec = (
        dict(next(iter(texture_material_specs.values())))
        if len(texture_material_specs) == 1
        else {}
    )
    physics_steps = dict(physics.get("steps") or {})
    material_steps = dict(material.get("steps") or {})
    material_predict = dict(material_steps.get("predict") or {})
    material_vlm = dict(material_predict.get("vlm") or {})
    material_llm = dict(material_predict.get("llm") or {})
    material_validation = dict(material_steps.get("validate_input") or {})
    identify_vlm = dict((physics_steps.get("identify_asset") or {}).get("vlm") or {})
    identify_enabled = (physics_steps.get("identify_asset") or {}).get("enabled")
    predict_vlm = dict((physics_steps.get("predict") or {}).get("vlm") or {})
    material_rendering_modes = set(
        ((material_steps.get("build_dataset_usd") or {}).get("renderer") or {})
        .get("rendering_modes", {})
    )
    physics_dataset = dict(physics_steps.get("build_dataset_usd") or {})
    physics_rendering_modes = set(
        (physics_dataset.get("renderer") or {}).get("rendering_modes", {})
    )
    # Target prims are scene-specific and bound by digests elsewhere; this
    # contract guards known paid-runtime failure modes. It therefore requires
    # the texture UV scope, declared targets, and the single material spec to
    # name the same non-empty absolute prims rather than one scene's paths.
    texture_targets = texture_config.get("uv_target_prim_paths")
    if len(texture_material_specs) != 1:
        raise ValueError("adp_content_agents_remote_config_contract_invalid")
    texture_targets_consistent = (
        isinstance(texture_targets, list)
        and bool(texture_targets)
        and all(
            isinstance(item, str) and item.startswith("/") for item in texture_targets
        )
        and texture.get("target_prims") == texture_targets
        and texture_spec.get("prim_paths") == texture_targets
        and isinstance(texture_spec.get("material_path"), str)
        and str(texture_spec.get("material_path")).startswith("/")
    )
    if (
        not material_path.is_file()
        or (material.get("materials") or {}).get("path")
        != "../content_agents_source/apps/material_agent/data/materials/"
        "material_libs_default/materials.yaml"
        or not texture_targets_consistent
        or texture_config.get("image_gen")
        != {"backend": "openai", "model": CONTENT_IMAGE_MODEL}
        or material_validation.get("on_failure") != "warn"
        or (material_steps.get("validate_output") or {}).get("on_failure") != "warn"
        or material_vlm.get("backend") != "openai"
        or material_vlm.get("model") != CONTENT_LLM_MODEL
        or material_vlm.get("reasoning_effort") != CONTENT_LLM_REASONING_EFFORT
        or material_llm.get("backend") != "openai"
        or material_llm.get("model") != CONTENT_LLM_MODEL
        or material_llm.get("reasoning_effort") != CONTENT_LLM_REASONING_EFFORT
        or identify_enabled is not False
        or identify_vlm.get("backend") != "openai"
        or identify_vlm.get("model") != CONTENT_LLM_MODEL
        or identify_vlm.get("reasoning_effort") != CONTENT_LLM_REASONING_EFFORT
        or predict_vlm.get("backend") != "openai"
        or predict_vlm.get("model") != CONTENT_LLM_MODEL
        or predict_vlm.get("reasoning_effort") != CONTENT_LLM_REASONING_EFFORT
        or material_rendering_modes != {"composition", "prim_only"}
        or physics_rendering_modes != {"composition", "prim_only"}
        or (physics_dataset.get("prim_filters") or {}).get("skip_invisible") is not True
    ):
        raise ValueError("adp_content_agents_remote_config_contract_invalid")


def _materialize_remote_configs(
    *,
    config_sources: Mapping[str, Path],
    destination: Path,
    variant: str,
    agent_mesh_prim_paths: Sequence[str] | None = None,
    agent_default_material_path: str | None = None,
    reference_image_relpaths: Sequence[str] | None = None,
) -> dict[str, str]:
    """Copy v1 configs or deterministically derive the approved v2 challenger."""

    mesh_paths = sorted(str(path) for path in (agent_mesh_prim_paths or ()))
    material_path = str(agent_default_material_path or "")
    reference_relpaths = list(reference_image_relpaths or ["../input/reference.png"])
    config_hashes: dict[str, str] = {}
    for name, path in config_sources.items():
        target = destination / name
        if variant in {"control_v1", "articulated_v1"}:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        elif variant in {"agent_cad_v1", "paired_target_registered_v1"}:
            if not mesh_paths or not material_path.startswith("/"):
                raise ValueError("adp_content_agents_candidate_config_scope_invalid")
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            project = payload["project"]
            agent_cad = variant == "agent_cad_v1"
            source_description = (
                "agent-authored CAD Mesh working copy"
                if agent_cad
                else "native-qualified registered replacement working copy"
            )
            project_name = (
                "adp_agent_cad_mesh_enrichment"
                if agent_cad
                else "adp_paired_target_registered_enrichment"
            )
            project["name"] = project_name
            project["session_id"] = project_name
            project["description"] = (
                "NVIDIA Content Agents advisory enrichment on an exact "
                f"{source_description}; output is not simulator, collision, "
                "physics, or physical-equivalence authority."
            )
            if name == "material_agent.yaml":
                dataset = payload["steps"]["build_dataset_usd"]
                dataset["prim_filters"]["paths"] = mesh_paths
                material_subject = (
                    "Classify visible materials on an agent-authored CAD candidate"
                    if agent_cad
                    else "Classify visible materials on the native-qualified "
                    "registered candidate"
                )
                payload["steps"]["build_dataset_prepare_dataset"]["prompts"][
                    "vlm_system"
                ] = material_subject + (
                    " using the provided reference image and renders. "
                    "Do not infer hidden, collision, physics, or physical truth. "
                    "Select only from the provided material library. "
                    "Available materials: {materials_list} "
                    "Respond as <reasoning>brief reasoning</reasoning>"
                    "<answer>{{\"material\": \"material name\"}}</answer>."
                )
                visible_subject = (
                    "Classify this visible CAD candidate surface. Treat it as "
                    if agent_cad
                    else "Classify this visible registered candidate surface. "
                    "Treat it as "
                )
                payload["steps"]["build_dataset_prepare_dataset"]["prompts"][
                    "vlm_user"
                ] = visible_subject + (
                    "generated candidate appearance, not observed truth."
                )
            elif name == "texture_agent.yaml":
                payload["texture"]["uv_target_prim_paths"] = mesh_paths
                payload["target_prims"] = mesh_paths
                # Keyed by the material's USD path, because that is the only
                # thing the texture agent's planner looks it up by. It resolves
                # `material_textures` against the material's alias paths and
                # then its name; a descriptive label matches neither, so the
                # material is skipped as `not_requested`, the plan contains zero
                # jobs, and the run is rejected -- after the GPU is already
                # rented. `material_path` inside the entry is only a guard the
                # planner uses to reject a name-keyed entry pointing elsewhere;
                # it is never what finds the entry.
                payload["material_textures"] = {
                    material_path: {
                        "prompt": (
                            "neutral realistic surface texture consistent with "
                            "the supplied observed reference image, no branding, "
                            "no text, no unobserved claims"
                        ),
                        "opacity": 1.0,
                        "material_path": material_path,
                        "prim_paths": mesh_paths,
                    }
                }
                payload["steps"]["render"]["focus_prim_paths"] = mesh_paths[:1]
            elif name == "physics_agent.yaml":
                payload["steps"]["build_dataset_usd"]["prim_filters"][
                    "paths"
                ] = mesh_paths
                payload["steps"]["apply_physics"]["collision_approx"] = "none"
                payload["steps"]["apply_physics"][
                    "mass_scale_policy"
                ] = "skip_mass"
            else:
                raise ValueError("adp_content_agents_config_sources_invalid")
        else:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            project = payload["project"]
            project["name"] = project["name"] + "_match_v2"
            project["session_id"] = project["session_id"] + "_match_v2"
            project["description"] = project["description"].replace(
                "CAD-derived can control", "human-approved multiview-matched v2 can"
            ).replace(
                "outputs remain authored priors",
                "human-approved v2 input; outputs remain authored priors",
            )
            if name == "material_agent.yaml":
                payload["steps"]["build_dataset_prepare_dataset"]["prompts"][
                    "vlm_user"
                ] = (
                    "Classify the rendered pale-mint beverage container appearance. "
                    "Do not assume aluminum, glass, plastic, or transparency from "
                    "shape or prompt alone."
                )
            elif name == "texture_agent.yaml":
                payload["material_textures"]["green_can"]["prompt"] = (
                    "pale mint green clean non-branded beverage container surface, "
                    "subtle vertical shading, no text or logo"
                )
        input_config = payload.get("input") or {}
        input_config["usd_path"] = "../input/source_asset.usda"
        if variant in {
            "agent_cad_v1",
            "paired_target_registered_v1",
        } or "reference_images" in input_config:
            input_config["reference_images"] = reference_relpaths
        payload["input"] = input_config
        target.write_text(
            yaml.safe_dump(payload, sort_keys=False, width=100), encoding="utf-8"
        )
        config_hashes[name] = _sha256(target)
    return config_hashes


def _deterministic_zip(source_root: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w") as archive:
        for path in sorted(source_root.rglob("*")):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(
                path.relative_to(source_root.parent).as_posix(),
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.create_system = 3
            info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
            archive.writestr(
                info,
                path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def build_content_agents_vast_bundle(
    *,
    repo_root: str | Path,
    content_agents_root: str | Path,
    job_dir: str | Path,
    reference_image_path: str | Path | None = None,
    reference_image_paths: Sequence[str | Path] | None = None,
    input_variant: str = "control_v1",
    evidence_root: str | Path | None = None,
    agent_cad_output_manifest_path: str | Path | None = None,
    agent_mesh_projection_receipt_path: str | Path | None = None,
    paired_target_construction_bindings_path: str | Path | None = None,
    paired_target_task_id: str | None = None,
    reference_rights_authority_path: str | Path | None = None,
    content_agents_execution_route_path: str | Path | None = None,
    generated_at: str | None = None,
    historical_replay_only: bool = False,
) -> dict[str, Any]:
    """Build one immutable bundle with explicit public-dataset byte accounting.

    The v1 control carries no dataset bytes. The approved v2 native probe carries
    the exact public CC-BY-NC SAGE collision companion, but never gated
    InteriorGS source bytes or Aura/InteriorGS appearance frames.
    """

    if historical_replay_only is not True and input_variant not in {
        "agent_cad_v1",
        "paired_target_registered_v1",
    }:
        raise ValueError("deterministic_cad_authoring_removed_use_agent_backend")
    repo = Path(repo_root).expanduser().resolve()
    source = Path(content_agents_root).expanduser().resolve()
    agent_output_path = (
        Path(agent_cad_output_manifest_path)
        if agent_cad_output_manifest_path is not None
        else None
    )
    if input_variant == "agent_cad_v1" and (
        reference_image_path is not None or reference_image_paths
    ):
        raise ValueError(
            "adp_content_agents_agent_cad_reference_must_come_from_manifest"
        )
    if (
        input_variant == "agent_cad_v1"
        and content_agents_execution_route_path is None
    ):
        raise ValueError("adp_content_agents_codex_first_route_missing")
    explicit_references = [
        Path(path).expanduser().resolve() for path in (reference_image_paths or ())
    ]
    if reference_image_path is not None:
        explicit_references.insert(
            0, Path(reference_image_path).expanduser().resolve()
        )
    if not explicit_references:
        if input_variant != "agent_cad_v1":
            raise ValueError("adp_content_agents_reference_image_missing")
        reference_sources = _default_agent_cad_reference_images(agent_output_path)
        reference_source = reference_sources[0]
    else:
        reference_sources = explicit_references
        reference_source = reference_sources[0]
    if input_variant == "paired_target_registered_v1" and any(
        path.is_symlink() or not path.is_file() for path in reference_sources
    ):
        raise ValueError("adp_content_agents_paired_target_input_not_host_resident")
    evidence = Path(evidence_root) if evidence_root is not None else None
    job = Path(job_dir).expanduser().resolve()
    if job.exists() and any(job.iterdir()):
        raise ValueError("adp_content_agents_bundle_job_dir_not_empty")
    head = _git(source, "rev-parse", "HEAD")
    tree = _git(source, "rev-parse", "HEAD^{tree}")
    dirty = bool(_git(source, "status", "--porcelain"))
    if head != SOURCE_COMMIT or tree != SOURCE_TREE or dirty:
        raise ValueError("adp_content_agents_source_identity_mismatch")
    if not reference_source.is_file() or reference_source.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("adp_content_agents_reference_image_identity_mismatch")
    variant = _resolve_input_variant(
        repo=repo,
        evidence_root=evidence,
        reference_source=reference_source,
        reference_sources=reference_sources,
        variant=input_variant,
        agent_cad_output_manifest_path=(
            agent_output_path.expanduser().resolve()
            if agent_output_path is not None
            else None
        ),
        agent_mesh_projection_receipt_path=(
            Path(agent_mesh_projection_receipt_path)
            if agent_mesh_projection_receipt_path is not None
            else None
        ),
        paired_target_construction_bindings_path=(
            Path(paired_target_construction_bindings_path)
            if paired_target_construction_bindings_path is not None
            else None
        ),
        paired_target_task_id=paired_target_task_id,
        reference_rights_authority_path=(
            Path(reference_rights_authority_path)
            if reference_rights_authority_path is not None
            else None
        ),
    )
    execution_route_binding = _content_agents_execution_route_binding(
        route_path=(
            Path(content_agents_execution_route_path)
            if content_agents_execution_route_path is not None
            else None
        ),
        variant=variant,
    )
    runtime = job / "provider_runtime"
    ensure_dir(runtime / "configs")
    ensure_dir(runtime / "input")

    source_zip = runtime / "content_agents_source.zip"
    subprocess.run(
        ["git", "-C", str(source), "archive", "--format=zip", f"--output={source_zip}", "HEAD"],
        check=True,
    )
    scripts = repo / "scripts"
    _write_executable(
        runtime / "run_adp_content_agents_provider_runtime.sh",
        scripts / "run_adp_content_agents_provider_runtime.sh",
    )
    shutil.copy2(
        scripts / "adp_content_agents_provider_runner.py",
        runtime / "adp_content_agents_provider_runner.py",
    )
    shutil.copy2(
        repo / "src/blueprint_pipeline/provider_archive.py",
        runtime / "provider_archive.py",
    )
    shutil.copy2(
        repo / "src/blueprint_pipeline/content_agents_model_compatibility.py",
        runtime / "content_agents_model_compatibility.py",
    )
    compatibility_plan = materialize_content_agents_model_compatibility_plan(
        model_ids=(CONTENT_LLM_MODEL, CONTENT_IMAGE_MODEL),
        destination=runtime / "content_agents_model_compatibility_plan.json",
    )
    native_probe: dict[str, Any] | None = None
    if variant["variant"] == "match_v2":
        native_probe = materialize_native_probe(
            evidence_root=evidence,
            destination=runtime / "native",
            replacement_receipt=variant["replacement_receipt"],
        )
        shutil.copy2(
            scripts / "run_ovrtx_preflight_worker.py",
            runtime / "native" / "run_ovrtx_preflight_worker.py",
        )
        shutil.copy2(
            scripts / "run_ovphysx_preflight_worker.py",
            runtime / "native" / "run_ovphysx_preflight_worker.py",
        )
    config_sources = {
        str(name): Path(path)
        for name, path in (variant.get("config_sources") or {}).items()
    }
    if set(config_sources) != {
        "material_agent.yaml",
        "texture_agent.yaml",
        "physics_agent.yaml",
    } or any(not path.is_file() for path in config_sources.values()):
        raise ValueError("adp_content_agents_config_sources_invalid")
    usd_source = Path(variant["usd_source"])
    runtime_usd_name = "source_asset.usda"
    input_normalization = _materialize_content_agents_input(
        usd_source,
        runtime / "input" / runtime_usd_name,
        variant=str(variant["variant"]),
    )
    reference_runtime_names = [
        "reference.png" if index == 0 else f"reference_{index + 1:04d}.png"
        for index in range(len(reference_sources))
    ]
    reference_runtime_relpaths = [
        f"../input/{name}" for name in reference_runtime_names
    ]
    config_hashes = _materialize_remote_configs(
        config_sources=config_sources,
        destination=runtime / "configs",
        variant=str(variant["variant"]),
        agent_mesh_prim_paths=(
            variant.get("mesh_prim_paths")
            or input_normalization.get("mesh_prim_paths")
        ),
        agent_default_material_path=(
            variant.get("default_material_path")
            or input_normalization.get("default_material_path")
        ),
        reference_image_relpaths=reference_runtime_relpaths,
    )
    runtime_configs = {
        name: runtime / "configs" / name for name in config_sources
    }
    _validate_remote_configs(source=source, config_sources=runtime_configs)
    joint_agent_plan = _derive_joint_agent_plan(
        input_variant=str(variant["variant"]),
        input_normalization=input_normalization,
    )
    reference_runtime_bindings: list[dict[str, Any]] = []
    for source_path, runtime_name in zip(
        reference_sources, reference_runtime_names, strict=True
    ):
        shutil.copy2(source_path, runtime / "input" / runtime_name)
        reference_runtime_bindings.append(
            {
                "relative_path": f"input/{runtime_name}",
                "sha256": _sha256(source_path),
            }
        )

    entrypoint = runtime / "run_adp_content_agents_provider_runtime.sh"
    runner = runtime / "adp_content_agents_provider_runner.py"
    blockers = provider_runtime_contract_blockers(
        provider_bundle_kind="adp_content_agents",
        entrypoint_text=entrypoint.read_text(encoding="utf-8"),
        runner_text=runner.read_text(encoding="utf-8"),
    )
    generated = generated_at or utc_now_iso()
    readiness = {
        "schema_version": "adp_content_agents_provider_bundle.v1",
        "generated_at": generated,
        "status": "ready" if not blockers else "blocked",
        "source_repository": "https://github.com/NVIDIA-Omniverse/usd-content-agents",
        "source_commit": head,
        "source_tree": tree,
        "source_version": SOURCE_VERSION,
        "container_image": DEFAULT_IMAGE,
        "container_platform": "linux/amd64",
        "source_archive_sha256": _sha256(source_zip),
        "model_parameter_compatibility": compatibility_plan,
        "input_usd_sha256": input_normalization["normalized_input_usd_sha256"],
        "input_usd_normalization": input_normalization,
        "input_variant": variant["variant"],
        "input_variant_bindings": {
            key: value
            for key, value in variant.items()
            if key.endswith("_receipt_digest")
            or key
            in {
                "cad_agent_output_manifest",
                "mesh_projection_receipt",
                "mesh_packet_digest",
                "candidate_step_sha256",
                "cad_agent_request_digest",
                "cad_agent_reference_manifest",
                "cad_agent_reference_manifest_object_digest",
                "cad_agent_selected_reference_image",
                "cad_agent_reference_images",
                "cad_agent_backend_id",
                "cad_agent_execution_mode",
                "task_id",
                "asset_id",
                "replacement_slot",
                "paired_target_construction_bindings",
                "paired_target_construction_digest",
                "paired_target_scene_id",
                "paired_target_reference_rights_authority",
                "paired_target_reference_rights_status",
                "paired_target_reference_rights_scope",
                "paired_target_reference_images",
                "task_freeze_digest",
                "registered_asset_receipt_digest",
                "replacement_asset_sha256",
                "native_import_probe_result_digest",
                "native_simulator_import_qualified",
            }
        },
        "content_agents_execution_route": execution_route_binding,
        "reference_image_sha256": variant["reference_image_sha256"],
        "reference_image_sha256s": variant.get(
            "reference_image_sha256s", [variant["reference_image_sha256"]]
        ),
        "runtime_reference_image_bindings": reference_runtime_bindings,
        "reference_image_authority": variant["reference_image_authority"],
        "runtime_entrypoint": "provider_runtime/run_adp_content_agents_provider_runtime.sh",
        "remote_config_contract_validated": True,
        "remote_config_sha256": config_hashes,
        "expected_output_filename": "adp_content_agents_vast_result.json",
        "material_agent_planned": True,
        "texture_agent_planned": True,
        "physics_agent_planned": True,
        "validation_agent_planned": True,
        "native_ovrtx_exact_camera_probe_planned": native_probe is not None,
        "native_ovphysx_drop_contact_settle_planned": native_probe is not None,
        "native_probe": native_probe,
        "gated_interiorgs_source_bytes_included": False,
        "public_sage_collision_bytes_included": native_probe is not None,
        "public_sage_collision_license": (
            "CC-BY-NC-4.0" if native_probe is not None else None
        ),
        "allowed_use_ceiling": (
            "internal_noncommercial_validation"
            if native_probe is not None
            or variant["variant"] == "paired_target_registered_v1"
            else "blueprint_owned_control"
        ),
        "runtime_input_binding": {
            "relative_path": f"input/{runtime_usd_name}",
            "sha256": input_normalization["normalized_input_usd_sha256"],
        },
        "joint_agent_plan": joint_agent_plan,
        "joint_agent_inapplicable_single_rigid_body": joint_agent_plan[
            "joint_agent_inapplicable_single_rigid_body"
        ],
        "execution_role": "optional_construction_enrichment",
        "failure_blocks_deterministic_asset_construction": False,
        "failure_blocks_native_simulator_qualification": False,
        "agent_output_is_simready_authority": False,
        "input_native_simulator_import_qualified": variant.get(
            "native_simulator_import_qualified"
        )
        is True,
        "canonical_simready_construction_unresolved": (
            variant["variant"] == "agent_cad_v1"
        ),
        "deterministic_usd_construction_remains_primary": variant["variant"]
        != "agent_cad_v1",
        "local_bundle_ready_for_remote_staging": not blockers,
        "provider_zero_required_after_return": True,
        "retry_cap": 0,
        "blockers": blockers,
        "raw_secret_values_recorded": False,
    }
    write_json(runtime / "adp_content_agents_provider_manifest.json", readiness)
    bundle_path = job / "adp_content_agents_provider_runtime_bundle.zip"
    _deterministic_zip(runtime, bundle_path)
    rehearsal = rehearse_provider_bundle_entrypoint(
        bundle_path=bundle_path,
        entrypoint_relative_path=(
            "provider_runtime/run_adp_content_agents_provider_runtime.sh"
        ),
        evidence_path=job / "adp_content_agents_exact_bundle_rehearsal.json",
    )
    receipt = {
        **readiness,
        "bundle_path": str(bundle_path),
        "bundle_sha256": _sha256(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
        "exact_bundle_entrypoint_rehearsal": rehearsal,
    }
    write_json(job / "adp_content_agents_bundle_receipt.json", receipt)
    return receipt


def _remaining_minutes(
    *, job: Path, hard_cap_usd: float, hard_ttl_seconds: int, max_hourly_rate_usd: float
) -> int:
    ledger = _read_json(job / "adp_content_agents_vast_session_budget.json")
    attempts = [row for row in ledger.get("attempts") or [] if isinstance(row, Mapping)]
    prior_seconds = sum(attempt_runtime_seconds(row) for row in attempts)
    prior_cost = sum(attempt_estimated_cost(row) for row in attempts)
    runtime_minutes = math.floor(max(0.0, hard_ttl_seconds - prior_seconds) / 60.0)
    spend_minutes = math.floor(
        max(0.0, hard_cap_usd - prior_cost) * 60.0 / max_hourly_rate_usd
    )
    return max(0, min(runtime_minutes, spend_minutes))


def _extract(path: Path, destination: Path) -> dict[str, Any]:
    blockers: list[str] = []
    if not path.is_file():
        return {"status": "blocked", "blockers": ["content_agents_provider_output_zip_missing"]}
    if destination.exists() and any(destination.iterdir()):
        return {
            "status": "blocked",
            "blockers": ["content_agents_provider_output_destination_not_empty"],
        }
    ensure_dir(destination)
    root = destination.resolve()
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if root not in target.parents and target != root:
                    blockers.append("content_agents_provider_output_zip_path_traversal")
            if not blockers:
                archive.extractall(destination)
    except (OSError, zipfile.BadZipFile):
        blockers.append("content_agents_provider_output_zip_invalid")
    result_path = destination / "adp_content_agents_vast_result.json"
    execution = _read_json(result_path)
    if not execution:
        blockers.append("content_agents_provider_result_missing")
    return {
        "status": "completed" if not blockers else "blocked",
        "result_path": str(result_path),
        "execution": execution,
        "blockers": sorted(set(blockers)),
    }


def _model_secret() -> str:
    """Resolve the forwarded model secret for a service first, a shell second.

    #488 fixed this in the config preflight and missed it here, so the paid
    preflight could prove model access while the run that followed could not
    find the same key: the units run with `ProtectHome=true` and home
    `/nonexistent`, and a developer home resolves to nothing there. The
    observed failure was `adp_content_agents_openai_secret_missing` raised
    after admission had already been granted.

    `_read_secret` honours `<NAME>_FILE`, then the configured secrets
    directory, and a developer home only when no directory is configured.
    """

    for name in _FORWARDED_SECRET_NAMES:
        value = str(os.getenv(name) or "").strip()
        if value:
            return value
    return str(_read_provider_secret("openai_api_key") or "")


@contextmanager
def _authority_environment():
    names = (
        *_VAST_MUTATION_ENV,
        _VAST_SINGLE_ATTEMPT_ENV,
        *_FORWARDED_SECRET_NAMES,
        "BLUEPRINT_VAST_FORWARD_SECRET_ENV_VARS",
    )
    previous = {name: os.environ.get(name) for name in names}
    secret = _model_secret()
    if not secret:
        raise ValueError("adp_content_agents_openai_secret_missing")
    try:
        for name in _VAST_MUTATION_ENV:
            os.environ[name] = "1"
        os.environ[_VAST_SINGLE_ATTEMPT_ENV] = "0"
        for name in _FORWARDED_SECRET_NAMES:
            os.environ[name] = secret
        os.environ["BLUEPRINT_VAST_FORWARD_SECRET_ENV_VARS"] = ",".join(
            _FORWARDED_SECRET_NAMES
        )
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_content_agents_vast(
    *,
    job_dir: str | Path,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    prepared_bundle: Mapping[str, Any],
    max_hourly_rate_usd: float = 1.0,
    hard_cap_usd: float = 3.0,
    hard_ttl_seconds: int = 7200,
    public_image: str = DEFAULT_IMAGE,
    allowed_active_instance_ids: Sequence[int] = (),
    machine_avoidlist_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one Content Agents attempt and always require provider-zero afterward."""

    job = Path(job_dir).expanduser().resolve()
    ensure_dir(job)
    bundle = dict(prepared_bundle)
    if public_image != DEFAULT_IMAGE:
        raise ValueError("adp_content_agents_container_image_not_frozen")
    bundle_path = Path(str(bundle.get("bundle_path") or "")).resolve()
    if (
        bundle.get("status") != "ready"
        or not bundle_path.is_file()
        or _sha256(bundle_path) != bundle.get("bundle_sha256")
        or provider_bundle_rehearsal_blockers(
            bundle.get("exact_bundle_entrypoint_rehearsal"),
            bundle_sha256=str(bundle.get("bundle_sha256") or ""),
            entrypoint_relative_path=(
                "provider_runtime/run_adp_content_agents_provider_runtime.sh"
            ),
        )
    ):
        raise ValueError("adp_content_agents_prepared_bundle_binding_invalid")
    if not execute:
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "dry_run_ready",
            "bundle": bundle,
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "blockers": [],
        }
        write_json(job / "adp_content_agents_vast_result.json", result)
        return result
    if paid_resource_admission_grant is None:
        raise ValueError("adp_content_agents_paid_resource_admission_grant_missing")

    remaining_minutes = _remaining_minutes(
        job=job,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        max_hourly_rate_usd=max_hourly_rate_usd,
    )
    if remaining_minutes < 45:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "blockers": ["adp_content_agents_budget_below_minimum_live_window"],
        }
    staging_dir = job / "object_store_staging"
    staging = stage_wam_provider_bundle_object_store(
        job_dir=staging_dir,
        bundle_path=str(bundle_path),
        key_prefix=DEFAULT_KEY_PREFIX,
        expiration_seconds=max(hard_ttl_seconds + 1800, 18_000),
    )
    if staging.get("status") != "completed":
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "blockers": staging.get("blockers") or ["content_agents_object_store_staging_blocked"],
        }
    provider_run = job / "vast_provider_run"
    output_zip = provider_run / "vast_provider_runtime_output.zip"
    watchdog_handoff, watchdog_handle = arm_independent_vast_watchdog(
        job_dir=job,
        max_live_minutes=remaining_minutes,
        generated_at=utc_now_iso(),
        allowed_active_instance_ids=allowed_active_instance_ids,
        pod_name_prefix=CONTENT_AGENTS_INSTANCE_LABEL_PREFIX,
    )
    if watchdog_handle is None:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "all_staged_objects_absent": cleanup.get("all_objects_absent"),
            "independent_watchdog": watchdog_handoff,
            "blockers": ["adp_content_agents_independent_watchdog_not_armed"],
        }
    adapter: dict[str, Any] = {}
    try:
        with _authority_environment():
            adapter = run_vast_provider_adapter(
                job_dir=provider_run,
                mode="live-startup-probe",
                allow_vast_api_call=True,
                allow_instance_launch=True,
                max_hourly_rate=max_hourly_rate_usd,
                target_spend_usd=hard_cap_usd,
                hard_cap_usd=hard_cap_usd,
                max_live_minutes=remaining_minutes,
                session_max_live_minutes=hard_ttl_seconds // 60,
                public_image=public_image,
                isaac_image=public_image,
                ngc_image_login_mode="never",
                provider_bundle=bundle_path,
                provider_bundle_url=(staging_dir / "provider_bundle_url.txt").read_text().strip(),
                provider_output_put_url=(staging_dir / "provider_output_put_url.txt").read_text().strip(),
                provider_output_get_url=(staging_dir / "provider_output_get_url.txt").read_text().strip(),
                provider_runtime_output_zip=output_zip,
                enable_isaac_smoke=False,
                enable_blueprint_bundle=True,
                provider_bundle_kind="adp_content_agents",
                vast_launch_mode="ssh_direct",
                allow_cold_isaac_image_pull=False,
                disk_gb=64,
                min_gpu_ram_mb=24_000,
                poll_interval_seconds=15,
                startup_timeout_seconds=remaining_minutes * 60,
                heartbeat_no_progress_seconds=1800,
                session_budget_ledger_path=job / "adp_content_agents_vast_session_budget.json",
                verify_staging_urls=True,
                require_known_supported_isaac_driver=False,
                preferred_gpu_keywords=("RTX 4090", "RTX A6000", "L40S", "A100"),
                prefer_isaac_rt=False,
                allowed_active_instance_ids=allowed_active_instance_ids,
                # The watchdog returns the unique prefix it armed. Deriving the
                # created label from that handle prevents a second literal from
                # silently moving the instance outside the watched name family.
                instance_label_prefix=watchdog_handle.pod_name_prefix,
                started_instance_id_path=watchdog_handle.started_instance_id_path,
                machine_avoidlist_path=machine_avoidlist_path,
                forward_hf_token=False,
                paid_resource_admission_grant=paid_resource_admission_grant,
                allowed_geolocation_country_codes=OPENAI_API_SUPPORTED_COUNTRY_CODES,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        adapter = {
            "status": "blocked",
            "blockers": [f"adp_content_agents_vast_adapter_failed:{redacted_failure_detail(exc)}"],
            "raw_secret_values_recorded": False,
        }
        # The adapter may never have been entered -- resolving a secret or a
        # staged URL raises before it. Record the absence of any allocation so
        # the run can close; the sealer declines whenever the evidence does not
        # support that claim.
        seal_unallocated_provider_teardown(
            provider_run, reason="adp_content_agents_vast_adapter_failed"
        )
    finally:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
    extracted = _extract(output_zip, job / "immutable_execution")
    execution = dict(extracted.get("execution") or {})
    teardown_path = provider_run / "vast_teardown_manifest.json"
    try:
        teardown = _read_json(teardown_path)
    except (OSError, json.JSONDecodeError):
        teardown = {}
    instance_ids = [
        int(value)
        for value in (
            teardown.get("vast_instance_ids")
            or adapter.get("vast_instance_ids")
            or []
        )
        if isinstance(value, int) and value > 0
    ]
    watchdog_close = close_independent_vast_watchdog(
        job_dir=job,
        handle=watchdog_handle,
        instance_ids=instance_ids,
        provider_teardown_completed=teardown.get("continuing_spend_from_this_run")
        is False,
        provider_allocation_impossible=(
            not instance_ids and adapter.get("provider_create_attempted") is not True
        ),
    )
    blockers = list(adapter.get("blockers") or []) + list(extracted.get("blockers") or [])
    if execution.get("status") != "completed":
        blockers.extend(execution.get("blockers") or ["content_agents_full_execution_not_completed"])
    if teardown.get("continuing_spend_from_this_run") is not False:
        blockers.append("content_agents_vast_provider_zero_not_proven")
    if cleanup.get("all_objects_absent") is not True:
        blockers.append("content_agents_object_store_provider_zero_not_proven")
    if watchdog_close.get("status") not in {
        "provider_terminal",
        "cancelled_no_allocation",
    }:
        blockers.append("adp_content_agents_independent_watchdog_not_closed")
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "completed" if not blockers else "blocked",
        "bundle_sha256": bundle["bundle_sha256"],
        "execution_result_path": extracted.get("result_path"),
        "adapter_result_path": str(provider_run / "vast_provider_adapter_result.json"),
        "teardown_manifest_path": str(teardown_path),
        "estimated_cost_usd": adapter.get("estimated_cost_usd"),
        "hard_cap_usd": hard_cap_usd,
        "hard_ttl_seconds": hard_ttl_seconds,
        "retry_cap": 0,
        "continuing_spend_from_this_run": teardown.get("continuing_spend_from_this_run"),
        "all_staged_objects_absent": cleanup.get("all_objects_absent"),
        "independent_watchdog": watchdog_close,
        "blockers": sorted(set(str(item) for item in blockers if str(item))),
        "raw_secret_values_recorded": False,
    }
    # Seal the two terminal artifacts every production launch profile asks
    # this result for. Without them the run ends
    # `allocator_terminal_artifact_missing:` whatever happened on the provider.
    result = seal_lane_terminal_artifacts(
        result,
        attempt_root=job,
        lane="adp_content_agents",
        binding={
            "bundle_sha256": bundle.get("bundle_sha256")
            if isinstance(bundle, Mapping)
            else None,
            "provider": "vast",
        },
    )
    write_json(job / "adp_content_agents_vast_result.json", result)
    return result


__all__ = [
    "DEFAULT_IMAGE",
    "EXECUTION_READINESS_SCHEMA",
    "PAID_ATTEMPT_AUTHORITY_SCHEMA",
    "PROBE_KIND",
    "REFERENCE_IMAGE_SHA256",
    "build_content_agents_vast_bundle",
    "consume_content_agents_paid_attempt_authority_once",
    "materialize_content_agents_execution_readiness",
    "run_content_agents_vast",
    "validate_content_agents_paid_attempt_authority",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the immutable ADP-009A Content Agents Vast bundle."
    )
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--content-agents-root", required=True)
    parser.add_argument(
        "--reference-image",
        action="append",
        help="Repeat for every exact rights-admitted reference image.",
    )
    parser.add_argument("--job-dir", required=True)
    parser.add_argument(
        "--input-variant",
        choices=(
            "control_v1",
            "match_v2",
            "articulated_v1",
            "agent_cad_v1",
            "paired_target_registered_v1",
        ),
        default="control_v1",
    )
    parser.add_argument("--evidence-root")
    parser.add_argument("--agent-cad-output-manifest")
    parser.add_argument("--agent-mesh-projection-receipt")
    parser.add_argument("--paired-target-construction-bindings")
    parser.add_argument("--paired-target-task-id")
    parser.add_argument("--reference-rights-authority")
    parser.add_argument("--content-agents-execution-route")
    parser.add_argument("--historical-replay-only", action="store_true")
    args = parser.parse_args(argv)
    receipt = build_content_agents_vast_bundle(
        repo_root=args.repo_root,
        content_agents_root=args.content_agents_root,
        reference_image_paths=args.reference_image,
        job_dir=args.job_dir,
        input_variant=args.input_variant,
        evidence_root=args.evidence_root,
        agent_cad_output_manifest_path=args.agent_cad_output_manifest,
        agent_mesh_projection_receipt_path=args.agent_mesh_projection_receipt,
        paired_target_construction_bindings_path=(
            args.paired_target_construction_bindings
        ),
        paired_target_task_id=args.paired_target_task_id,
        reference_rights_authority_path=args.reference_rights_authority,
        content_agents_execution_route_path=args.content_agents_execution_route,
        historical_replay_only=args.historical_replay_only,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt.get("status") == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
