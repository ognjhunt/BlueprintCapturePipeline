from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import policy_endpoint_token_setup as token_setup
from blueprint_pipeline.g1_field_run_capture import _first_string, _webapp_route_prefill


def test_policy_endpoint_token_setup_creates_reuses_and_prints_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    token_file = tmp_path / "secrets" / "team-token.txt"
    manifest_path = tmp_path / "token-manifest.json"
    generated = "2026-06-20T00:00:00+00:00"

    monkeypatch.setattr(token_setup.secrets, "token_urlsafe", lambda _n: "first-token")
    created = token_setup.create_team_policy_endpoint_token(
        token_file=token_file,
        write_manifest=manifest_path,
        generated_at=generated,
    )

    assert created["status"] == "created"
    assert created["generated_at"] == generated
    assert created["file_mode_octal"] == "0o600"
    assert created["raw_token_written_to_artifacts"] is False
    assert token_file.read_text(encoding="utf-8") == "first-token\n"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["token_file"] == str(
        token_file.resolve()
    )

    monkeypatch.setattr(token_setup.secrets, "token_urlsafe", lambda _n: "second-token")
    reused = token_setup.create_team_policy_endpoint_token(token_file=token_file)
    assert reused["status"] == "already_present"
    assert token_file.read_text(encoding="utf-8") == "first-token\n"

    rc = token_setup.main(
        [
            "--token-file",
            str(token_file),
            "--force",
            "--write-manifest",
            str(tmp_path / "forced-manifest.json"),
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert printed["status"] == "created"
    assert printed["raw_token_written_to_stdout"] is False
    assert token_file.read_text(encoding="utf-8") == "second-token\n"


def test_g1_field_route_helpers_cover_empty_and_unaccepted_inputs() -> None:
    assert _first_string("", None, default="fallback") == "fallback"
    assert _webapp_route_prefill({"status": "blocked"}) == {}
