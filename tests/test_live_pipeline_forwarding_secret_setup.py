from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import live_pipeline_forwarding_secret_setup as setup


def test_live_pipeline_forwarding_env_setup_writes_secret_only_to_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / "secrets" / "forwarding.env"
    manifest_path = tmp_path / "forwarding-manifest.json"
    capture_root = tmp_path / "capture-root"
    monkeypatch.setattr(setup.secrets, "token_urlsafe", lambda _n: "generated-secret")

    summary = setup.create_live_pipeline_forwarding_env(
        env_file=env_file,
        forward_url="https://paperclip.tryblueprint.io/api/live-pipeline/job-requests",
        capture_root=capture_root,
        site_slug="first-gpu-walkthrough2",
        write_manifest=manifest_path,
        generated_at="2026-06-26T00:00:00+00:00",
    )

    env_text = env_file.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert summary["status"] == "created"
    assert summary["file_mode_octal"] == "0o600"
    assert "ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN" in summary["configured_keys"]
    assert "BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN" in summary["configured_keys"]
    assert "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON" in env_text
    assert "BLUEPRINT_LIVE_PIPELINE_STAGED_INPUTS_PATH" in env_text
    assert "generated-secret" in env_text
    assert "generated-secret" not in json.dumps(summary)
    assert "generated-secret" not in json.dumps(manifest)
    assert summary["raw_token_written_to_stdout"] is False
    assert summary["raw_token_written_to_manifest"] is False
    assert summary["raw_token_written_to_env_file"] is True


def test_live_pipeline_forwarding_env_setup_reuses_existing_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / "forwarding.env"
    monkeypatch.setattr(setup.secrets, "token_urlsafe", lambda _n: "first-secret")
    setup.create_live_pipeline_forwarding_env(env_file=env_file)

    monkeypatch.setattr(setup.secrets, "token_urlsafe", lambda _n: "second-secret")
    summary = setup.create_live_pipeline_forwarding_env(env_file=env_file)

    env_text = env_file.read_text(encoding="utf-8")
    assert summary["status"] == "already_present"
    assert "first-secret" in env_text
    assert "second-secret" not in env_text


def test_live_pipeline_forwarding_env_setup_validates_forward_url(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="job-requests"):
        setup.create_live_pipeline_forwarding_env(
            env_file=tmp_path / "forwarding.env",
            forward_url="https://paperclip.tryblueprint.io/health",
        )


def test_live_pipeline_forwarding_env_setup_preserves_absolute_capture_root_spelling(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target-capture-root"
    target.mkdir()
    capture_root = tmp_path / "remote-style-capture-root"
    capture_root.symlink_to(target, target_is_directory=True)
    env_file = tmp_path / "forwarding.env"

    summary = setup.create_live_pipeline_forwarding_env(
        env_file=env_file,
        capture_root=capture_root,
        site_slug="first-gpu-walkthrough-2",
    )

    values = setup.parse_env_file_values(env_file)
    assert summary["capture_root"] == str(capture_root)
    assert json.loads(values["ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON"]) == {
        "first-gpu-walkthrough-2": str(capture_root),
    }
    assert values["BLUEPRINT_LIVE_PIPELINE_STAGED_INPUTS_PATH"] == str(
        capture_root / "pipeline" / "live_pipeline_staged_inputs.json"
    )


def test_parse_env_file_values_handles_export_and_quotes(tmp_path: Path) -> None:
    env_file = tmp_path / "forwarding.env"
    env_file.write_text(
        "\n".join(
            [
                "export ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN='secret value'",
                'BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN="same token"',
                "bad-key=ignored",
            ]
        ),
        encoding="utf-8",
    )

    values = setup.parse_env_file_values(env_file)

    assert values["ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN"] == "secret value"
    assert values["BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN"] == "same token"
    assert "bad-key" not in values
