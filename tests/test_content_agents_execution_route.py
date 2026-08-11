from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.content_agents_execution_route import (
    ContentAgentsExecutionRouteError,
    materialize_content_agents_execution_route,
    nvidia_content_agents_required,
    validate_content_agents_execution_route,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _digest(token: str) -> str:
    return "sha256:" + token * 64


def _object(
    slot: int,
    *,
    capabilities: list[str],
) -> dict[str, object]:
    letter = "abcdef"[slot - 1]
    return {
        "replacement_slot": slot,
        "task_id": f"task_{slot}",
        "asset_id": f"asset_{slot}",
        "source_binding_digest": _digest(letter),
        "requested_capabilities": capabilities,
    }


def test_codex_only_route_is_digest_bound_and_never_forwards_an_api_key(
    tmp_path: Path,
) -> None:
    route = materialize_content_agents_execution_route(
        objects=[
            _object(
                1,
                capabilities=["configuration_review", "material_group_proposal"],
            )
        ],
        output_path=tmp_path / "route.json",
        generated_at="2026-08-11T00:00:00Z",
    )

    assert route["status"] == "codex_local_only_ready"
    assert route["codex_local"]["api_key_forwarded"] is False
    assert route["nvidia_content_agents"]["required"] is False
    assert route["claim_boundary"]["codex_is_nvidia_content_agents_execution"] is False
    assert json.loads((tmp_path / "route.json").read_text()) == route
    assert nvidia_content_agents_required(
        route,
        replacement_slot=1,
        task_id="task_1",
        asset_id="asset_1",
        source_binding_digest=_digest("a"),
    ) == (False, ["configuration_review", "material_group_proposal"], [])


def test_hybrid_route_separates_residual_released_runtime_from_codex_work(
    tmp_path: Path,
) -> None:
    route = materialize_content_agents_execution_route(
        objects=[
            _object(
                1,
                capabilities=[
                    "input_packet_review",
                    "released_material_texture_physics_validation_pipeline",
                ],
            ),
            _object(2, capabilities=["output_receipt_review"]),
        ],
        output_path=tmp_path / "route.json",
        generated_at="2026-08-11T00:00:00Z",
    )

    assert route["status"] == "hybrid_route_ready"
    assert route["replacement_object_capacity"]["sealed_slots"] == 2
    assert route["nvidia_content_agents"] == {
        "required": True,
        "capabilities": [
            "released_material_texture_physics_validation_pipeline"
        ],
        "runtime": "pinned_released_nvidia_usd_content_agents",
        "current_transport": "provider_configured_model_api",
        "api_key_forwarded": True,
        "dataset_upload_performed": False,
        "output_class": "advisory_candidate_only",
    }
    assert nvidia_content_agents_required(
        route,
        replacement_slot=2,
        task_id="task_2",
        asset_id="asset_2",
        source_binding_digest=_digest("b"),
    ) == (False, ["output_receipt_review"], [])
    assert nvidia_content_agents_required(
        route,
        replacement_slot=1,
        task_id="task_1",
        asset_id="asset_1",
        source_binding_digest=_digest("a"),
    ) == (
        True,
        ["input_packet_review"],
        ["released_material_texture_physics_validation_pipeline"],
    )


def test_route_rejects_cross_object_reuse_and_tampering(tmp_path: Path) -> None:
    route = materialize_content_agents_execution_route(
        objects=[
            _object(1, capabilities=["input_packet_review"]),
            _object(2, capabilities=["output_receipt_review"]),
            _object(3, capabilities=["physics_scope_review"]),
            _object(4, capabilities=["appearance_scope_review"]),
            _object(5, capabilities=["bundle_preflight_review"]),
        ],
        output_path=tmp_path / "route.json",
        generated_at="2026-08-11T00:00:00Z",
    )
    assert route["replacement_object_capacity"]["sealed_slots"] == 5

    with pytest.raises(
        ContentAgentsExecutionRouteError,
        match="content_agents_execution_route_object_binding_mismatch",
    ):
        nvidia_content_agents_required(
            route,
            replacement_slot=1,
            task_id="task_1",
            asset_id="asset_1",
            source_binding_digest=_digest("b"),
        )

    tampered = json.loads(json.dumps(route))
    tampered["codex_local"]["api_key_forwarded"] = True
    tampered["route_digest"] = canonical_digest(
        tampered, digest_field="route_digest"
    )
    with pytest.raises(
        ContentAgentsExecutionRouteError,
        match="content_agents_execution_route_codex_local_invalid",
    ):
        validate_content_agents_execution_route(tampered)

    duplicate = json.loads(json.dumps(route))
    duplicate["objects"][1]["asset_id"] = duplicate["objects"][0]["asset_id"]
    duplicate["route_digest"] = canonical_digest(
        duplicate, digest_field="route_digest"
    )
    with pytest.raises(
        ContentAgentsExecutionRouteError,
        match="content_agents_execution_route_object_identity_duplicate",
    ):
        validate_content_agents_execution_route(duplicate)
