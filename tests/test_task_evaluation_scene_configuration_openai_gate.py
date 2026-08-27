from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_scene_configuration_openai_gate as gate


def _environment() -> dict[str, str]:
    return {
        "BLUEPRINT_SCENE_CONFIGURATION_AUTHORITY_DIGEST": "sha256:" + "a" * 64,
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_COST_USD": "1.5",
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_REQUESTS": "32",
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_MAX_COST_USD": "0.4",
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_ARTIFIXER_VISUAL_REVIEW_MAX_COST_USD": "0.75",
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_CONTENT_AGENTS_MAX_COST_USD": "0.35",
        "OPENAI_ADMIN_API_KEY_FILE": "/private/admin-key",
        "OPENAI_PROJECT_ID": "proj_test",
        "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_FILE": "/private/key-semantic",
        "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_ID": "key_semantic",
        "BLUEPRINT_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_COST_SCOPE_ATTESTATION_FILE": (
            "/private/cost-scope-semantic.json"
        ),
        "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE": "/private/key-review",
        "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_ID": "key_review",
        "BLUEPRINT_OPENAI_ARTIFIXER_VISUAL_REVIEW_COST_SCOPE_ATTESTATION_FILE": (
            "/private/cost-scope-review.json"
        ),
        "OPENAI_CONTENT_AGENTS_API_KEY_FILE": "/private/key-content-agents",
        "OPENAI_CONTENT_AGENTS_API_KEY_ID": "key_content_agents",
        "BLUEPRINT_OPENAI_CONTENT_AGENTS_COST_SCOPE_ATTESTATION_FILE": (
            "/private/cost-scope-content-agents.json"
        ),
    }


def test_builds_exact_stage_gate_from_parent_authority(monkeypatch, tmp_path) -> None:
    captured = {}

    def build(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(gate, "build_openai_official_cost_run_gate", build)
    result = gate.scene_configuration_openai_stage_gate(
        environment=_environment(),
        stage="content_agents",
        run_id="configure-scene-content-agents",
        request_digest="sha256:" + "b" * 64,
        candidate_digest="sha256:" + "c" * 64,
        output_root=tmp_path,
    )

    assert result["max_cost_usd"] == 0.35
    assert result["authorization_receipt_digest"] == "sha256:" + "a" * 64
    assert result["admin_api_key_file"] == "/private/admin-key"
    assert result["api_key_id"] == "key_content_agents"
    assert result["require_zero_baseline"] is False
    # The gate reads the receipt the lane resolved for this stage, not the raw
    # operator path: an absent or pre-rename operator file is derived from the
    # provisioned key binding instead of stalling the run. Pin the binding the
    # receipt carries rather than where it happens to live.
    resolved = json.loads(
        Path(result["scope_attestation_path"]).read_text(encoding="utf-8")
    )
    assert resolved["paid_resource_class"] == (
        "task_evaluation_scene_configuration_content_agents"
    )
    assert resolved["api_key_id"] == "key_content_agents"
    assert resolved["project_id"] == "proj_test"
    assert resolved["exclusive_use"] is True


def test_each_stage_binds_its_own_exclusive_scope(monkeypatch, tmp_path) -> None:
    """One run holds three OpenAI stages; a shared key scope cannot pass.

    The operator attestation binds one exact ``paid_resource_class`` and the
    delta snapshot is meaningful only when no sibling stage shares that key.
    This pins that each stage resolves a distinct key id and attestation file,
    which is the property the shared single-scope environment silently broke.
    """

    monkeypatch.setattr(
        gate, "build_openai_official_cost_run_gate", lambda **kwargs: kwargs
    )
    environment = _environment()
    scopes = {}
    for stage in (
        "artifixer_semantic_teacher",
        "artifixer_visual_review",
        "content_agents",
    ):
        built = gate.scene_configuration_openai_stage_gate(
            environment=environment,
            stage=stage,
            run_id=f"configure-scene-{stage}",
            request_digest="sha256:" + "b" * 64,
            candidate_digest="sha256:" + "c" * 64,
            output_root=tmp_path / stage,
        )
        scopes[stage] = (built["api_key_id"], built["scope_attestation_path"])
    key_ids = [scope[0] for scope in scopes.values()]
    attestations = [scope[1] for scope in scopes.values()]
    assert len(set(key_ids)) == 3
    assert len(set(attestations)) == 3


def test_stage_scope_fails_closed_without_per_stage_names(tmp_path) -> None:
    environment = _environment()
    environment.pop("OPENAI_CONTENT_AGENTS_API_KEY_ID")
    # The retired shared names must not satisfy the per-stage scope.
    environment["OPENAI_API_KEY_ID"] = "key_shared"
    environment["BLUEPRINT_OPENAI_COST_SCOPE_ATTESTATION_FILE"] = (
        "/private/shared-cost-scope.json"
    )
    with pytest.raises(
        gate.TaskEvaluationSceneConfigurationOpenAIGateError,
        match="scene_configuration_openai_stage_scope_missing:content_agents",
    ):
        gate.scene_configuration_openai_stage_gate(
            environment=environment,
            stage="content_agents",
            run_id="configure-scene-content-agents",
            request_digest="sha256:" + "b" * 64,
            candidate_digest="sha256:" + "c" * 64,
            output_root=tmp_path,
        )


def test_stage_scope_refuses_unknown_stage() -> None:
    with pytest.raises(
        gate.TaskEvaluationSceneConfigurationOpenAIGateError,
        match="scene_configuration_openai_stage_unknown",
    ):
        gate.scene_configuration_openai_stage_scope(
            _environment(), stage="not_a_stage"
        )


def test_fails_closed_without_exact_parent_cap(tmp_path) -> None:
    environment = _environment()
    environment.pop(
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_MAX_COST_USD"
    )
    with pytest.raises(
        gate.TaskEvaluationSceneConfigurationOpenAIGateError,
        match="scene_configuration_openai_authority_invalid",
    ):
        gate.scene_configuration_openai_stage_gate(
            environment=environment,
            stage="artifixer_semantic_teacher",
            run_id="configure-scene-artifixer",
            request_digest="sha256:" + "b" * 64,
            candidate_digest="sha256:" + "c" * 64,
            output_root=tmp_path,
        )
