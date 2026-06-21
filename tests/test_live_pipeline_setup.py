from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from blueprint_pipeline import live_pipeline_setup as lps
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
    monkeypatch.setattr("blueprint_pipeline.live_pipeline_setup._module_available", lambda _: False)
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


def test_live_pipeline_setup_accepts_owner_arena_results_without_overclaiming_webapp(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path, with_webapp_ids=False)
    arena_results = tmp_path / "arena-results"
    _write_json(
        arena_results / "rollout_manifest.json",
        {
            "episodes": [
                {
                    "episode_id": "episode-1",
                    "scenario_id": "scenario-1",
                    "status": "success",
                    "success": True,
                }
            ]
        },
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_CODEX_SDK_OPERATORS", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("blueprint_pipeline.live_pipeline_setup._module_available", lambda _: True)

    result = build_live_pipeline_setup_manifest(
        capture_root=capture_root,
        arena_results_dir=arena_results,
        vision_labeling_command="python -c 'print(1)'",
        delivery_command="python -c 'print(1)'",
        load_local_env=False,
    )

    assert result["status"] == "local_ready_live_external_blocked"
    assert result["sections"]["real_arena_execution"]["status"] == "ready_for_result_ingest"
    assert result["sections"]["real_arena_execution"]["ready"] is True
    assert result["sections"]["real_arena_execution"]["blockers"] == []
    assert result["sections"]["real_arena_execution"]["arena_results"]["ready"] is True
    assert result["sections"]["webapp_upstream_truth"]["status"] == "blocked"
    assert any(
        blocker.startswith("webapp_upstream_truth:missing_webapp_")
        for blocker in result["blockers"]
    )
    next_inputs = " ".join(result["next_inputs_needed"])
    assert "owner-system Arena simulator command" not in next_inputs
    assert result["sections"]["real_arena_execution"]["claim_boundary"]["simulator_execution_proven"] is False


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


def test_live_pipeline_setup_helper_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lps.shlex, "split", lambda _text: (_ for _ in ()).throw(ValueError("bad quote")))
    assert lps._first_executable("'unterminated") is None
    monkeypatch.setattr(lps.shlex, "split", lambda _text: [])
    assert lps._first_executable("present") is None
    monkeypatch.undo()
    assert lps._first_executable("FOO=bar python -c 'print(1)'") == "python"

    missing_dir = tmp_path / "missing-results"
    assert lps._arena_results_status(missing_dir)["blockers"] == ["arena_results_dir_missing"]
    empty_dir = tmp_path / "empty-results"
    empty_dir.mkdir()
    assert lps._arena_results_status(empty_dir)["blockers"] == ["arena_results_dir_has_no_json_artifacts"]
    assert lps._capture_upstream_truth(None)["blockers"] == ["capture_root_not_provided"]
    capture_root = _capture_root(tmp_path, with_webapp_ids=False)
    descriptor = json.loads((capture_root / "capture_descriptor.json").read_text(encoding="utf-8"))
    descriptor["site_submission_id"] = "site-submission-only"
    _write_json(capture_root / "capture_descriptor.json", descriptor)
    upstream = lps._capture_upstream_truth(capture_root)
    assert upstream["fields_present"]["request_id"] is True
    assert lps._package_audit_status(None)["blockers"] == ["package_dir_not_provided"]
    assert lps._overall_status({name: {"ready": True} for name in (
        "real_arena_execution",
        "rollout_vision_labeling",
        "delivery_upload",
        "live_agents_operator",
        "live_codex_operator",
        "webapp_upstream_truth",
    )}) == "ready_for_live_external_execution"
    blocked_sections = {
        name: {"ready": False}
        for name in (
            "real_arena_execution",
            "rollout_vision_labeling",
            "delivery_upload",
            "live_agents_operator",
            "live_codex_operator",
            "webapp_upstream_truth",
        )
    }
    blocked_sections["local_deterministic_lane"] = {"ready": False}
    assert lps._overall_status(blocked_sections) == "blocked"


def test_live_pipeline_setup_digitalocean_read_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DO_TOKEN", raising=False)
    missing = lps._digitalocean_read(
        allow_read=True,
        token_env="DO_TOKEN",
        droplet_name=None,
        droplet_ip=None,
        timeout_seconds=1,
    )
    assert missing["status"] == "blocked"
    assert "missing_env_DO_TOKEN" in missing["blockers"]
    assert "missing_droplet_name_or_ip" in missing["blockers"]

    monkeypatch.setenv("DO_TOKEN", "secret-token")
    monkeypatch.setattr(
        lps.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )
    failed = lps._digitalocean_read(
        allow_read=True,
        token_env="DO_TOKEN",
        droplet_name="worker",
        droplet_ip=None,
        timeout_seconds=1,
    )
    assert failed["blockers"] == ["digitalocean_api_read_failed:URLError"]

    class FakeResponse:
        def __init__(self, payload: object) -> None:
            self.payload = payload

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(self.payload).encode("utf-8")

    monkeypatch.setattr(
        lps.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(
            {
                "droplets": [
                    "not-a-droplet",
                    {
                        "id": 1,
                        "name": "worker",
                        "status": "active",
                        "region": {"slug": "nyc3"},
                        "memory": 4096,
                        "vcpus": 2,
                        "disk": 80,
                        "networks": {"v4": [{"type": "public", "ip_address": "203.0.113.10"}]},
                        "image": {"slug": "ubuntu-24-04"},
                    },
                ]
            }
        ),
    )
    no_match = lps._digitalocean_read(
        allow_read=True,
        token_env="DO_TOKEN",
        droplet_name="worker",
        droplet_ip="203.0.113.99",
        timeout_seconds=1,
    )
    assert no_match["status"] == "blocked"
    assert no_match["blockers"] == ["digitalocean_droplet_not_found"]
    matched = lps._digitalocean_read(
        allow_read=True,
        token_env="DO_TOKEN",
        droplet_name="worker",
        droplet_ip="203.0.113.10",
        timeout_seconds=1,
    )
    assert matched["status"] == "ready_control_plane"
    assert matched["matches"][0]["public_ipv4_present"] is True
    assert matched["matches"][0]["gpu_proof"] is False


def test_live_pipeline_setup_output_paths_and_missing_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lps.shutil, "which", lambda _name: None)
    output_path = tmp_path / "explicit" / "setup.json"
    explicit = build_live_pipeline_setup_manifest(
        load_local_env=False,
        output_path=output_path,
    )
    assert output_path.is_file()
    assert "local_deterministic_lane:missing_ffmpeg_for_clip_keyframe_paths" in explicit["blockers"]

    monkeypatch.chdir(tmp_path)
    default = build_live_pipeline_setup_manifest(load_local_env=False)
    assert (tmp_path / "live_pipeline_setup_manifest.json").is_file()
    assert default["capture_root"] is None


def test_live_pipeline_setup_main_success_and_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        lps,
        "build_live_pipeline_setup_manifest",
        lambda **_: {"status": "local_ready_live_external_blocked", "blockers": []},
    )
    output_path = tmp_path / "setup.json"
    assert lps.main(["--output-path", str(output_path), "--no-load-env-files"]) == 0
    assert f"manifest={output_path}" in capsys.readouterr().out

    capture_root = _capture_root(tmp_path)
    monkeypatch.setattr(
        lps,
        "build_live_pipeline_setup_manifest",
        lambda **_: {"status": "blocked", "blockers": ["missing"]},
    )
    assert lps.main(["--capture-root", str(capture_root), "--timeout-seconds", "1"]) == 1
    blocked_output = capsys.readouterr().out
    assert "status=blocked" in blocked_output
    assert "blockers=1" in blocked_output
