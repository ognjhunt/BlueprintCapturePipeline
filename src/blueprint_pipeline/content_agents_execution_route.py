"""Codex-first routing for optional NVIDIA Content Agents work.

The released NVIDIA Content Agents source has API-backed model adapters, but a
large part of a replacement-asset workflow is local advisory work: validating
the input packet, reviewing a bounded material/physics scope, and reviewing a
bundle before a paid launch.  This contract makes that split explicit for one
to five independent replacement objects.

It deliberately does *not* claim that a Codex CLI session is an NVIDIA Content
Agents execution.  A route can use local Codex for advisory work without an API
key or upload; a released Content Agents material/texture/physics run remains a
separate, explicitly requested runtime capability.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .materializer_cli import Param, Step, run
from .openai_successor_models import OPENAI_REASONING_EFFORT, OPENAI_TEXT_MODEL


SCHEMA_VERSION = "content_agents_execution_route.v1"
MAX_REPLACEMENT_OBJECTS = 5

# These are bounded review/orchestration functions.  Their output is advisory
# and must still be accepted by the deterministic construction and evidence
# gates before it can influence a simulator candidate.
LOCAL_CODEX_CAPABILITIES = frozenset(
    {
        "input_packet_review",
        "appearance_scope_review",
        "material_group_proposal",
        "physics_scope_review",
        "configuration_review",
        "bundle_preflight_review",
        "output_receipt_review",
    }
)

# These are not interchangeable with an interactive Codex session.  The
# current pinned NVIDIA release executes them as one released-code pipeline
# with its own runtime/model transport and output contract.
NVIDIA_CONTENT_AGENTS_CAPABILITIES = frozenset(
    {
        "released_material_texture_physics_validation_pipeline",
    }
)


class ContentAgentsExecutionRouteError(ValueError):
    """Fail-closed invalid route with stable, sorted error identifiers."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted({str(code) for code in codes if str(code)}))
        super().__init__(";".join(self.codes))


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise ContentAgentsExecutionRouteError(
            ["content_agents_execution_route_not_json"]
        ) from exc
    if not isinstance(result, dict):
        raise ContentAgentsExecutionRouteError(
            ["content_agents_execution_route_not_mapping"]
        )
    return result


def _text(value: Any) -> str:
    return str(value or "").strip()


def _digest(value: Any) -> bool:
    text = _text(value)
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _positive_slot(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 1 <= value <= MAX_REPLACEMENT_OBJECTS else None


def _capabilities(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not value:
        return None
    normalized = sorted({_text(item) for item in value if _text(item)})
    if len(normalized) != len(value):
        return None
    if not set(normalized) <= (
        LOCAL_CODEX_CAPABILITIES | NVIDIA_CONTENT_AGENTS_CAPABILITIES
    ):
        return None
    return normalized


def _normalized_objects(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_REPLACEMENT_OBJECTS:
        raise ContentAgentsExecutionRouteError(
            ["content_agents_execution_route_object_capacity_invalid"]
        )
    objects: list[dict[str, Any]] = []
    seen_slots: set[int] = set()
    seen_task_ids: set[str] = set()
    seen_asset_ids: set[str] = set()
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise ContentAgentsExecutionRouteError(
                [f"content_agents_execution_route_object_invalid:{index}"]
            )
        slot = _positive_slot(row.get("replacement_slot"))
        task_id = _text(row.get("task_id"))
        asset_id = _text(row.get("asset_id"))
        source_binding_digest = _text(row.get("source_binding_digest"))
        capabilities = _capabilities(row.get("requested_capabilities"))
        if (
            slot is None
            or not task_id
            or not asset_id
            or not _digest(source_binding_digest)
            or capabilities is None
        ):
            raise ContentAgentsExecutionRouteError(
                [f"content_agents_execution_route_object_invalid:{index}"]
            )
        if (
            slot in seen_slots
            or task_id in seen_task_ids
            or asset_id in seen_asset_ids
        ):
            raise ContentAgentsExecutionRouteError(
                ["content_agents_execution_route_object_identity_duplicate"]
            )
        seen_slots.add(slot)
        seen_task_ids.add(task_id)
        seen_asset_ids.add(asset_id)
        codex_capabilities = sorted(
            set(capabilities) & LOCAL_CODEX_CAPABILITIES
        )
        nvidia_capabilities = sorted(
            set(capabilities) & NVIDIA_CONTENT_AGENTS_CAPABILITIES
        )
        objects.append(
            {
                "replacement_slot": slot,
                "task_id": task_id,
                "asset_id": asset_id,
                "source_binding_digest": source_binding_digest,
                "requested_capabilities": capabilities,
                "codex_local_capabilities": codex_capabilities,
                "nvidia_content_agents_capabilities": nvidia_capabilities,
            }
        )
    return sorted(objects, key=lambda row: row["replacement_slot"])


def _expected_route(
    *, objects: list[dict[str, Any]], generated_at: str | None
) -> dict[str, Any]:
    codex_capabilities = sorted(
        {
            capability
            for row in objects
            for capability in row["codex_local_capabilities"]
        }
    )
    nvidia_capabilities = sorted(
        {
            capability
            for row in objects
            for capability in row["nvidia_content_agents_capabilities"]
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "codex_local_only_ready"
            if not nvidia_capabilities
            else "hybrid_route_ready"
            if codex_capabilities
            else "nvidia_content_agents_only_ready"
        ),
        "generated_at": generated_at,
        "replacement_object_capacity": {
            "minimum": 1,
            "maximum": MAX_REPLACEMENT_OBJECTS,
            "sealed_slots": len(objects),
        },
        "objects": objects,
        "codex_local": {
            "required": bool(codex_capabilities),
            "capabilities": codex_capabilities,
            "transport": "codex_cli_host_oauth",
            "model": OPENAI_TEXT_MODEL,
            "reasoning_effort": OPENAI_REASONING_EFFORT,
            "api_key_forwarded": False,
            "dataset_upload_performed": False,
            "output_class": "advisory_candidate_only",
        },
        "nvidia_content_agents": {
            "required": bool(nvidia_capabilities),
            "capabilities": nvidia_capabilities,
            "runtime": "pinned_released_nvidia_usd_content_agents",
            "current_transport": "provider_configured_model_api",
            "api_key_forwarded": bool(nvidia_capabilities),
            "dataset_upload_performed": False,
            "output_class": "advisory_candidate_only",
        },
        "claim_boundary": {
            "codex_is_nvidia_content_agents_execution": False,
            "content_agents_executed": False,
            "simready_qualified": False,
            "native_simulator_import_qualified": False,
            "physical_equivalence": False,
        },
    }


def validate_content_agents_execution_route(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a digest-bound Codex/NVIDIA route."""

    route = _clone(value)
    errors: list[str] = []
    if route.get("schema_version") != SCHEMA_VERSION:
        errors.append("content_agents_execution_route_schema_invalid")
    try:
        objects = _normalized_objects(route.get("objects"))
    except ContentAgentsExecutionRouteError as exc:
        errors.extend(exc.codes)
        objects = []
    if objects:
        expected = _expected_route(
            objects=objects,
            generated_at=_text(route.get("generated_at")) or None,
        )
        for key, expected_value in expected.items():
            if key == "generated_at":
                if not _text(route.get(key)):
                    errors.append("content_agents_execution_route_generated_at_missing")
                continue
            if route.get(key) != expected_value:
                errors.append(f"content_agents_execution_route_{key}_invalid")
    supplied = _text(route.get("route_digest"))
    if supplied != canonical_digest(route, digest_field="route_digest"):
        errors.append("content_agents_execution_route_digest_invalid")
    if errors:
        raise ContentAgentsExecutionRouteError(errors)
    return route


def materialize_content_agents_execution_route(
    *,
    objects: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    generated_at: str,
) -> dict[str, Any]:
    """Seal a 1..5 object route without invoking Codex, an API, or a provider."""

    normalized = _normalized_objects(list(objects))
    if not _text(generated_at):
        raise ContentAgentsExecutionRouteError(
            ["content_agents_execution_route_generated_at_missing"]
        )
    route = _expected_route(objects=normalized, generated_at=_text(generated_at))
    route["route_digest"] = canonical_digest(route, digest_field="route_digest")
    validated = validate_content_agents_execution_route(route)
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(validated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return validated


def nvidia_content_agents_required(
    route: Mapping[str, Any],
    *,
    replacement_slot: int,
    task_id: str,
    asset_id: str,
    source_binding_digest: str,
) -> tuple[bool, list[str], list[str]]:
    """Return the selected object's residual NVIDIA-only capabilities.

    The exact object identity and source binding prevent a caller from carrying
    a Codex-only decision from one replacement into another object's paid run.
    """

    validated = validate_content_agents_execution_route(route)
    matches = [
        row
        for row in validated["objects"]
        if row["replacement_slot"] == replacement_slot
        and row["task_id"] == task_id
        and row["asset_id"] == asset_id
        and row["source_binding_digest"] == source_binding_digest
    ]
    if len(matches) != 1:
        raise ContentAgentsExecutionRouteError(
            ["content_agents_execution_route_object_binding_mismatch"]
        )
    row = matches[0]
    nvidia = list(row["nvidia_content_agents_capabilities"])
    return bool(nvidia), list(row["codex_local_capabilities"]), nvidia


STEPS = {
    "route": Step(
        "Seal the selected 1-5 object Codex/NVIDIA execution route.",
        materialize_content_agents_execution_route,
        {
            "objects": Param("--objects", required=True, json_file=True),
            "output_path": Param("--output", required=True),
            "generated_at": Param("--generated-at", required=True),
        },
    )
}


def main(argv: Sequence[str] | None = None) -> int:
    """Expose route sealing without invoking Codex, an API, or a provider."""

    return run(STEPS, argv, description=__doc__)


__all__ = [
    "ContentAgentsExecutionRouteError",
    "LOCAL_CODEX_CAPABILITIES",
    "MAX_REPLACEMENT_OBJECTS",
    "NVIDIA_CONTENT_AGENTS_CAPABILITIES",
    "SCHEMA_VERSION",
    "main",
    "materialize_content_agents_execution_route",
    "nvidia_content_agents_required",
    "validate_content_agents_execution_route",
]


if __name__ == "__main__":  # pragma: no cover - CLI seam
    raise SystemExit(main())
