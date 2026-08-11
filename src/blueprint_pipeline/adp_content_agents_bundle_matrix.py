"""Digest-bound 1--5-object Content Agents bundle matrix materialization.

The matrix is intentionally derived from sealed bundle receipts rather than a
caller-authored list of task/object facts.  It keeps two independent admitted
agent-CAD candidates for every replacement slot while proving that each new
bundle retained its object-specific Codex-first route before any provider work.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import utc_now_iso, write_json
from .content_agents_execution_route import (
    ContentAgentsExecutionRouteError,
    MAX_REPLACEMENT_OBJECTS,
    nvidia_content_agents_required,
    validate_content_agents_execution_route,
)
from .decision_evidence_contracts import canonical_digest
from .provider_bundle_rehearsal import provider_bundle_rehearsal_blockers
from .simready_cad_agent_contract import ADMITTED_BACKENDS


SCHEMA_VERSION = "adp_agent_cad_content_agents_bundle_matrix.v2"
_ENTRYPOINT = "provider_runtime/run_adp_content_agents_provider_runtime.sh"


class AgentCadContentAgentsBundleMatrixError(ValueError):
    """Fail-closed matrix construction/validation error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise AgentCadContentAgentsBundleMatrixError(
            "adp_content_agents_bundle_matrix_receipt_file_invalid"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise AgentCadContentAgentsBundleMatrixError(
            "adp_content_agents_bundle_matrix_receipt_file_invalid"
        ) from exc
    if not isinstance(value, Mapping):
        raise AgentCadContentAgentsBundleMatrixError(
            "adp_content_agents_bundle_matrix_receipt_file_invalid"
        )
    return dict(value)


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise AgentCadContentAgentsBundleMatrixError(
            "adp_content_agents_bundle_matrix_file_invalid"
        )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _text(value: Any) -> str:
    return str(value or "").strip()


def _digest(value: Any) -> str:
    text = _text(value)
    if (
        len(text) != 71
        or not text.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in text[7:])
    ):
        raise AgentCadContentAgentsBundleMatrixError(
            "adp_content_agents_bundle_matrix_digest_invalid"
        )
    return text


def _nonnegative_slot(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AgentCadContentAgentsBundleMatrixError(
            "adp_content_agents_bundle_matrix_identity_invalid"
        )
    if not 1 <= value <= MAX_REPLACEMENT_OBJECTS:
        raise AgentCadContentAgentsBundleMatrixError(
            "adp_content_agents_bundle_matrix_identity_invalid"
        )
    return value


def _normalized_bundle_item(bundle_receipt_path: str | Path) -> dict[str, Any]:
    receipt_path = Path(bundle_receipt_path).expanduser().resolve()
    receipt = _read_json(receipt_path)
    bundle_path = Path(_text(receipt.get("bundle_path"))).expanduser().resolve()
    bundle_record = _file_record(bundle_path)
    receipt_record = _file_record(receipt_path)
    if (
        receipt.get("schema_version") != "adp_content_agents_provider_bundle.v1"
        or receipt.get("status") != "ready"
        or receipt.get("input_variant") != "agent_cad_v1"
        or receipt.get("bundle_sha256") != bundle_record["sha256"]
        or receipt.get("bundle_size_bytes") != bundle_record["size_bytes"]
        or receipt.get("blockers") not in ([], None)
        or provider_bundle_rehearsal_blockers(
            receipt.get("exact_bundle_entrypoint_rehearsal"),
            bundle_sha256=bundle_record["sha256"],
            entrypoint_relative_path=_ENTRYPOINT,
        )
    ):
        raise AgentCadContentAgentsBundleMatrixError(
            "adp_content_agents_bundle_matrix_bundle_receipt_invalid"
        )
    bindings = receipt.get("input_variant_bindings")
    normalization = receipt.get("input_usd_normalization")
    route_binding = receipt.get("content_agents_execution_route")
    if (
        not isinstance(bindings, Mapping)
        or not isinstance(normalization, Mapping)
        or not isinstance(route_binding, Mapping)
    ):
        raise AgentCadContentAgentsBundleMatrixError(
            "adp_content_agents_bundle_matrix_bundle_binding_invalid"
        )
    slot = _nonnegative_slot(bindings.get("replacement_slot"))
    task_id = _text(bindings.get("task_id"))
    asset_id = _text(bindings.get("asset_id"))
    backend = _text(bindings.get("cad_agent_backend_id"))
    if not task_id or not asset_id or backend not in ADMITTED_BACKENDS:
        raise AgentCadContentAgentsBundleMatrixError(
            "adp_content_agents_bundle_matrix_identity_invalid"
        )
    scalar_binding_fields = (
        "cad_agent_output_receipt_digest",
        "cad_agent_request_digest",
        "cad_agent_reference_manifest_object_digest",
        "mesh_projection_receipt_digest",
        "mesh_packet_digest",
        "candidate_step_sha256",
    )
    normalized_bindings: dict[str, str] = {}
    for field in scalar_binding_fields:
        normalized_bindings[field] = _digest(bindings.get(field))
    mesh_count = normalization.get("mesh_count")
    if (
        isinstance(mesh_count, bool)
        or not isinstance(mesh_count, int)
        or mesh_count < 1
    ):
        raise AgentCadContentAgentsBundleMatrixError(
            "adp_content_agents_bundle_matrix_mesh_count_invalid"
        )
    route_path = Path(_text(route_binding.get("path"))).expanduser().resolve()
    route_record = _file_record(route_path)
    if (
        route_binding.get("sha256") != route_record["sha256"]
        or route_binding.get("size_bytes") != route_record["size_bytes"]
        or not _digest(route_binding.get("route_digest"))
    ):
        raise AgentCadContentAgentsBundleMatrixError(
            "adp_content_agents_bundle_matrix_route_binding_invalid"
        )
    try:
        route = validate_content_agents_execution_route(_read_json(route_path))
        requires_nvidia, codex_capabilities, nvidia_capabilities = (
            nvidia_content_agents_required(
                route,
                replacement_slot=slot,
                task_id=task_id,
                asset_id=asset_id,
                source_binding_digest=normalized_bindings[
                    "cad_agent_output_receipt_digest"
                ],
            )
        )
    except ContentAgentsExecutionRouteError as exc:
        raise AgentCadContentAgentsBundleMatrixError(
            "adp_content_agents_bundle_matrix_route_binding_invalid"
        ) from exc
    if (
        not requires_nvidia
        or route_binding.get("route_digest") != route["route_digest"]
        or route_binding.get("codex_local_capabilities") != codex_capabilities
        or route_binding.get("nvidia_content_agents_capabilities")
        != nvidia_capabilities
    ):
        raise AgentCadContentAgentsBundleMatrixError(
            "adp_content_agents_bundle_matrix_route_binding_invalid"
        )
    return {
        "replacement_slot": slot,
        "task_id": task_id,
        "asset_id": asset_id,
        "cad_agent_backend_id": backend,
        **normalized_bindings,
        "mesh_count": mesh_count,
        "bundle": bundle_record,
        "bundle_receipt": receipt_record,
        "content_agents_execution_route": {
            **route_record,
            "route_digest": route["route_digest"],
            "codex_local_capabilities": codex_capabilities,
            "nvidia_content_agents_capabilities": nvidia_capabilities,
        },
        "exact_bundle_entrypoint_rehearsal_status": "passed",
        "agent_output_is_simready_authority": False,
        "canonical_simready_construction_unresolved": True,
    }


def _normalized_items(receipt_paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    if not isinstance(receipt_paths, Sequence) or isinstance(receipt_paths, (str, bytes)):
        raise AgentCadContentAgentsBundleMatrixError(
            "adp_content_agents_bundle_matrix_receipt_paths_invalid"
        )
    if not receipt_paths or len(receipt_paths) > MAX_REPLACEMENT_OBJECTS * len(
        ADMITTED_BACKENDS
    ):
        raise AgentCadContentAgentsBundleMatrixError(
            "adp_content_agents_bundle_matrix_capacity_invalid"
        )
    items = [_normalized_bundle_item(path) for path in receipt_paths]
    seen: set[tuple[int, str]] = set()
    per_slot: dict[int, tuple[str, str, set[str]]] = {}
    for item in items:
        key = (item["replacement_slot"], item["cad_agent_backend_id"])
        if key in seen:
            raise AgentCadContentAgentsBundleMatrixError(
                "adp_content_agents_bundle_matrix_duplicate_candidate"
            )
        seen.add(key)
        prior = per_slot.setdefault(
            item["replacement_slot"],
            (item["task_id"], item["asset_id"], set()),
        )
        if prior[0] != item["task_id"] or prior[1] != item["asset_id"]:
            raise AgentCadContentAgentsBundleMatrixError(
                "adp_content_agents_bundle_matrix_slot_identity_mismatch"
            )
        prior[2].add(item["cad_agent_backend_id"])
    slots = sorted(per_slot)
    if slots != list(range(1, len(slots) + 1)) or any(
        row[2] != set(ADMITTED_BACKENDS) for row in per_slot.values()
    ):
        raise AgentCadContentAgentsBundleMatrixError(
            "adp_content_agents_bundle_matrix_backend_coverage_invalid"
        )
    return sorted(
        items,
        key=lambda row: (row["replacement_slot"], row["cad_agent_backend_id"]),
    )


def _expected_matrix(
    *, items: list[dict[str, Any]], generated_at: str
) -> dict[str, Any]:
    sealed_slots = len({item["replacement_slot"] for item in items})
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "local_bundles_ready_for_paid_resource_preflight",
        "input_variant": "agent_cad_v1",
        "candidate_count": len(items),
        "replacement_object_capacity": {
            "minimum": 1,
            "maximum": MAX_REPLACEMENT_OBJECTS,
            "sealed_slots": sealed_slots,
        },
        "items": items,
        "claim_boundary": {
            "content_agents_bundles_built": True,
            "exact_entrypoint_rehearsed": True,
            "content_agents_executed": False,
            "simready_qualified": False,
            "native_simulator_import_qualified": False,
            "physical_equivalence": False,
        },
    }


def validate_agent_cad_content_agents_bundle_matrix(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-open every referenced receipt and reject hand-authored matrix drift."""

    if not isinstance(value, Mapping):
        raise AgentCadContentAgentsBundleMatrixError(
            "adp_content_agents_bundle_matrix_not_mapping"
        )
    matrix = json.loads(json.dumps(value))
    if matrix.get("schema_version") != SCHEMA_VERSION:
        raise AgentCadContentAgentsBundleMatrixError(
            "adp_content_agents_bundle_matrix_schema_invalid"
        )
    if not _text(matrix.get("generated_at")):
        raise AgentCadContentAgentsBundleMatrixError(
            "adp_content_agents_bundle_matrix_generated_at_missing"
        )
    records = matrix.get("items")
    if not isinstance(records, list):
        raise AgentCadContentAgentsBundleMatrixError(
            "adp_content_agents_bundle_matrix_items_invalid"
        )
    items = _normalized_items(
        [
            str((record.get("bundle_receipt") or {}).get("path") or "")
            if isinstance(record, Mapping)
            else ""
            for record in records
        ]
    )
    expected = _expected_matrix(items=items, generated_at=_text(matrix["generated_at"]))
    for field, expected_value in expected.items():
        if matrix.get(field) != expected_value:
            raise AgentCadContentAgentsBundleMatrixError(
                "adp_content_agents_bundle_matrix_content_invalid"
            )
    if matrix.get("receipt_digest") != canonical_digest(
        matrix, digest_field="receipt_digest"
    ):
        raise AgentCadContentAgentsBundleMatrixError(
            "adp_content_agents_bundle_matrix_digest_invalid"
        )
    return matrix


def materialize_agent_cad_content_agents_bundle_matrix(
    *,
    bundle_receipt_paths: Sequence[str | Path],
    output_path: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Materialize a reusable, 1--5-object matrix from bundle receipt paths."""

    timestamp = _text(generated_at) or utc_now_iso()
    items = _normalized_items(bundle_receipt_paths)
    matrix = _expected_matrix(items=items, generated_at=timestamp)
    matrix["receipt_digest"] = canonical_digest(matrix, digest_field="receipt_digest")
    validated = validate_agent_cad_content_agents_bundle_matrix(matrix)
    write_json(output_path, validated)
    return validated


__all__ = [
    "AgentCadContentAgentsBundleMatrixError",
    "SCHEMA_VERSION",
    "materialize_agent_cad_content_agents_bundle_matrix",
    "validate_agent_cad_content_agents_bundle_matrix",
]
