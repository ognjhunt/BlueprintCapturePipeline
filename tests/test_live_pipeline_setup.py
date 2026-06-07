from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.live_pipeline_setup import build_live_pipeline_setup_manifest


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _capture_root(tmp_path: Path, *, with_webapp_ids: bool = False) -> Path:
    capture_root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    descriptor: dict[str, object] = {"scene_id": "scene-1", "capture_id": "capture-1"}
    if with_webapp_ids:
        descriptor.update(
            {
                "site_submission_id": "site-submission-1",
                "request_id": "request-1",
                "buyer_request_id": "buyer-request-1",
                "capture_job_id": "capture-job-1",
            }
        )
    _write_json(capture_root / "capture_descriptor.json", descriptor)
    _write_json(capture_root / "raw" / "manifest.json", {"scene_id": "scene-1"})
    return capture_root


def test_live_pipeline_setup_blocks_external_actions_without_gates(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    result = build_live_pipeline_setup_manifest(
        capture_root=capture_root,
        load_local_env=False,
        digitalocean_droplet_name="paperclip-prod-01",
        digitalocean_droplet_ip="206.81.11.69",
    )

    assert result["status"] == "local_ready_live_external_blocked"
    assert result["sections"]["local_deterministic_lane"]["ready"] is True
    assert "real_arena_execution:missing_env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION" in result["blockers"]
    assert "rollout_vision_labeling:missing_env_BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING" in result[
        "blockers"
    ]
    assert "delivery_upload:missing_env_BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD" in result["blockers"]
    assert "live_agents_operator:missing_openai_api_key" in result["blockers"]
    assert any(
        blocker.startswith("live_codex_operator:missing_env_BLUEPRINT_ALLOW_CODEX_CLI_HOST_OAUTH")
        or blocker.startswith("live_codex_operator:missing_openai_codex_sdk")
        for blocker in result["blockers"]
    )
    assert result["sections"]["digitalocean_control_plane"]["status"] == "configured_advisory"
    assert result["sections"]["digitalocean_control_plane"]["blockers"] == []
    assert not any(
        blocker.startswith("digitalocean_control_plane:") for blocker in result["blockers"]
    )
    assert (
        result["sections"]["digitalocean_control_plane"]["control_plane_boundary"][
            "simulator_execution_proven"
        ]
        is False
    )
    assert (capture_root / "pipeline" / "live_pipeline_setup" / "live_pipeline_setup_manifest.json").is_file()


def test_live_pipeline_setup_loads_env_without_exposing_values(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path, with_webapp_ids=True)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING=true",
                "BLUEPRINT_ROLLOUT_VISION_LABELING_COMMAND=python -c 'print(1)'",
                "GEMINI_API_KEY=secret-gemini",
                "OPENAI_API_KEY=secret-openai",
                "DIGITALOCEAN_ACCESS_TOKEN=secret-do",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = build_live_pipeline_setup_manifest(capture_root=capture_root)

    assert str(env_file.resolve()) in result["env_files"]["files"]
    assert result["secrets"]["GEMINI_API_KEY"]["present"] is True
    assert result["secrets"]["GEMINI_API_KEY"]["value_redacted"] is True
    assert "secret-gemini" not in json.dumps(result)
    assert result["sections"]["rollout_vision_labeling"]["status"] == "ready"
    assert result["sections"]["rollout_vision_labeling"]["gemini_env_present"] is True
    assert result["sections"]["webapp_upstream_truth"]["status"] == "ready"
    assert (
        result["sections"]["live_agents_operator"]["auth_boundary"]["chatgpt_pro_oauth"][
            "usable_by_repo_subprocess"
        ]
        is False
    )


def test_live_pipeline_setup_marks_live_sections_ready_when_explicitly_configured(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path, with_webapp_ids=True)
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_CODEX_SDK_OPERATORS", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    result = build_live_pipeline_setup_manifest(
        capture_root=capture_root,
        simulator_command="python -c 'print(1)'",
        vision_labeling_command="python -c 'print(1)'",
        delivery_command="python -c 'print(1)'",
        load_local_env=False,
    )

    assert result["sections"]["real_arena_execution"]["status"] == "ready"
    assert result["sections"]["rollout_vision_labeling"]["status"] == "ready"
    assert result["sections"]["delivery_upload"]["status"] == "ready"
    next_inputs = " ".join(result["next_inputs_needed"])
    assert "Provide a vision-labeling command" not in next_inputs
    assert "Provide a delivery command" not in next_inputs
    assert result["sections"]["live_agents_operator"]["status"] == "blocked"
    assert "missing_openai_agents_sdk" in result["sections"]["live_agents_operator"]["blockers"]
    assert result["sections"]["live_codex_operator"]["status"] == "blocked"
    assert (
        "missing_openai_codex_sdk_or_codex_cli_host_oauth"
        in result["sections"]["live_codex_operator"]["blockers"]
    )


def test_live_pipeline_setup_allows_codex_cli_host_oauth_when_gated(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path, with_webapp_ids=True)
    fake_codex = tmp_path / "codex"
    fake_codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_codex.chmod(0o755)
    monkeypatch.setattr(
        "blueprint_pipeline.live_pipeline_setup.codex_cli_path",
        lambda: str(fake_codex),
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_CODEX_SDK_OPERATORS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_CODEX_CLI_HOST_OAUTH", "true")

    result = build_live_pipeline_setup_manifest(
        capture_root=capture_root,
        load_local_env=False,
    )

    assert result["sections"]["live_codex_operator"]["status"] == "ready"
    assert result["sections"]["live_codex_operator"]["codex_cli_ready"] is True
    assert result["sections"]["live_codex_operator"]["codex_cli_host_oauth_allowed"] is True
    assert (
        result["sections"]["live_codex_operator"]["auth_boundary"]["codex_cli_host_oauth"][
            "usable_by_repo_subprocess"
        ]
        is True
    )
