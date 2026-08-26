from __future__ import annotations

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
        "BLUEPRINT_OPENAI_COST_SCOPE_ATTESTATION_FILE": "/private/cost-scope.json",
        "OPENAI_ADMIN_API_KEY_FILE": "/private/admin-key",
        "OPENAI_PROJECT_ID": "proj_test",
        "OPENAI_API_KEY_ID": "key_test",
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
