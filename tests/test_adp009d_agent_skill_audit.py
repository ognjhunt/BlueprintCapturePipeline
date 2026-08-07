from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.adp009d_agent_skill_audit import (
    AgentSkillAuditError,
    EXPECTED_REPOSITORIES,
    REQUIRED_GUIDANCE,
    load_agent_skill_audit,
    validate_agent_skill_audit,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPO_ROOT
    / "docs"
    / "arm_decision_proof_v1"
    / "manifests"
    / "adp009d_agent_skill_audit.v1.json"
)


def _audit() -> dict:
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


def _resign(value: dict) -> dict:
    value["audit_digest"] = canonical_digest(value, digest_field="audit_digest")
    return value


def test_checked_in_agent_skill_audit_is_exact_and_fail_closed() -> None:
    audit = load_agent_skill_audit(AUDIT_PATH)

    assert audit["audit_digest"] == canonical_digest(audit, digest_field="audit_digest")
    assert {row["source_id"] for row in audit["repositories"]} == set(
        EXPECTED_REPOSITORIES
    )
    assert {row["guidance_id"] for row in audit["guidance_application"]} == (
        REQUIRED_GUIDANCE
    )
    assert audit["implementation_base"]["commit"] == (
        "39ec8091e182e325f0ff0eb494f4dc9434c3bb27"
    )
    assert audit["implementation_base"]["provisional_until_pr_374_merge"] is False
    assert audit["runtime_compatibility_decision"][
        "latest_isaac_lab_develop_runtime_admitted"
    ] is False


def test_audit_separates_followed_guidance_from_pending_and_inapplicable() -> None:
    audit = load_agent_skill_audit(AUDIT_PATH)
    guidance = {row["guidance_id"]: row for row in audit["guidance_application"]}

    assert guidance["isaaclab-building-environments"]["actually_followed"] is True
    assert guidance["isaaclab-using-presets"]["actually_followed"] is False
    assert guidance["isaaclab-using-presets"]["application_status"] == "not_applicable"
    assert guidance["ovrtx-loading-usd"]["actually_followed"] is True
    assert (
        guidance["ovrtx-loading-usd"]["application_status"]
        == "followed_in_implementation"
    )
    runtime = audit["runtime_compatibility_decision"]
    assert runtime["ovrtx_selected"] is False
    assert runtime["sealed_aura_renderer_backend"] == (
        "AuraFusion360_official_native_2D_surfel_rasterizer"
    )
    assert runtime["native_aura_renderer_execution_status"] == (
        "materialized_unexecuted"
    )
    assert guidance["nvidia-omniverse-usd-performance-tuning"][
        "application_status"
    ] == "followed_in_implementation"
    assert guidance["nvidia-omniverse-usd-performance-tuning"][
        "actually_followed"
    ] is True


def test_audit_rejects_revision_rewrite_even_when_caller_resigns() -> None:
    audit = copy.deepcopy(_audit())
    next(row for row in audit["repositories"] if row["source_id"] == "ovrtx")[
        "revision"
    ] = "a" * 40

    with pytest.raises(AgentSkillAuditError) as exc_info:
        validate_agent_skill_audit(_resign(audit))

    assert "repository_ovrtx_revision_invalid" in exc_info.value.errors


def test_audit_rejects_invented_ovphysx_skill_path() -> None:
    audit = copy.deepcopy(_audit())
    physx = next(row for row in audit["repositories"] if row["source_id"] == "physx")
    physx["skill_path_audit"]["ovphysx_skill_path"] = "ovphysx/skills/SKILL.md"

    with pytest.raises(AgentSkillAuditError) as exc_info:
        validate_agent_skill_audit(_resign(audit))

    assert "physx_ovphysx_skill_path_invented" in exc_info.value.errors


def test_audit_rejects_false_actual_following_claim() -> None:
    audit = copy.deepcopy(_audit())
    ovrtx = next(
        row
        for row in audit["guidance_application"]
        if row["guidance_id"] == "ovrtx-warmup"
    )
    ovrtx["application_status"] = "pending_implementation"

    with pytest.raises(AgentSkillAuditError) as exc_info:
        validate_agent_skill_audit(_resign(audit))

    assert "guidance_ovrtx-warmup_actual_status_mismatch" in exc_info.value.errors


def test_audit_rejects_caller_asserted_digest() -> None:
    audit = _audit()
    audit["claim_ceiling"].append("caller asserted extra claim")

    with pytest.raises(AgentSkillAuditError) as exc_info:
        validate_agent_skill_audit(audit)

    assert "audit_digest_mismatch" in exc_info.value.errors
