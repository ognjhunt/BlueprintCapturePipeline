from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.nvidia_siggraph_policy import (
    COMPONENT_POLICY,
    evaluate_component_activation,
    validate_post_conference_source_review,
    write_capability_registry,
)


def test_registry_preserves_all_defer_and_claim_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    payload = write_capability_registry(path)
    assert set(payload["components"]) == set(COMPONENT_POLICY)
    assert payload["agent_toolkit_required"] is False
    assert payload["raw_capture_authority_preserved"] is True
    assert payload["post_conference_refresh"]["required"] is True
    assert json.loads(path.read_text()) == payload


def test_deferred_components_cannot_slip_onto_critical_path() -> None:
    blender = evaluate_component_activation(
        "simready_blender",
        evidence={"explicit_opt_in": True, "critical_path": True},
        as_of_date="2026-07-21",
    )
    assert blender["status"] == "blocked"
    assert "simready_blender_critical_path_prohibited" in blender["blockers"]

    ovstage = evaluate_component_activation(
        "ovstage",
        evidence={"explicit_opt_in": True, "standalone_adoption": True},
        as_of_date="2026-07-21",
    )
    assert "ovstage_standalone_adoption_prohibited" in ovstage["blockers"]


def test_content_agents_require_buyer_need_human_approval_and_proposal_semantics() -> None:
    result = evaluate_component_activation(
        "content_agents",
        evidence={"explicit_opt_in": True},
        as_of_date="2026-07-21",
    )
    assert result["status"] == "blocked"
    assert "content_agents_human_approval_missing" in result["blockers"]
    assert "content_agents_raw_capture_upload_prohibited" in result["blockers"]
    assert "content_agents_physics_authority_prohibited" in result["blockers"]


def test_post_conference_refresh_becomes_mandatory_on_july_24() -> None:
    result = evaluate_component_activation(
        "ovrtx",
        evidence={"explicit_opt_in": True},
        as_of_date="2026-07-24",
    )
    assert "post_siggraph_source_version_license_refresh_required" in result["blockers"]


def test_post_conference_refresh_requires_every_component_and_unlocks_only_structurally() -> None:
    review = {
        "schema_version": "nvidia_siggraph_post_conference_source_review.v1",
        "reviewed_at": "2026-07-24T18:00:00Z",
        "reviewer_id": "source-reviewer-1",
        "components": [
            {
                "component": component,
                "source_url": f"https://example.invalid/{component}",
                "source_revision_or_package_version": "pinned-revision",
                "license_id": "reviewed-license",
                "license_compatible": True,
                "maturity": "reviewed",
                "decision": "retain-policy-boundary",
                "evidence_urls": [f"https://example.invalid/{component}/evidence"],
            }
            for component in COMPONENT_POLICY
        ],
    }
    validation = validate_post_conference_source_review(review, as_of_date="2026-07-24")
    assert validation["status"] == "completed"
    activation = evaluate_component_activation(
        "ovrtx",
        evidence={
            "explicit_opt_in": True,
            "post_conference_source_review": validation,
        },
        as_of_date="2026-07-24",
    )
    assert "post_siggraph_source_version_license_refresh_required" not in activation["blockers"]
