from __future__ import annotations

import os
from pathlib import Path

from blueprint_pipeline.safe_env import contract_test_env, load_env_files


def test_load_env_files_applies_local_and_alpha_without_printing_values(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("PIPELINE_SECRET", raising=False)
    monkeypatch.delenv("PIPELINE_MODE", raising=False)
    (tmp_path / ".env").write_text(
        "PIPELINE_SECRET=base-secret\nPIPELINE_MODE=base\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.local").write_text("PIPELINE_MODE=local\n", encoding="utf-8")
    (tmp_path / ".env.alpha.local").write_text(
        "PIPELINE_ALPHA_TOKEN='alpha-secret'\n",
        encoding="utf-8",
    )

    summary = load_env_files([tmp_path])

    assert os.environ["PIPELINE_SECRET"] == "base-secret"
    assert os.environ["PIPELINE_MODE"] == "local"
    assert os.environ["PIPELINE_ALPHA_TOKEN"] == "alpha-secret"
    assert str(tmp_path / ".env.alpha.local") in summary["files"]
    assert "PIPELINE_ALPHA_TOKEN" in summary["loaded_keys"]
    assert "alpha-secret" not in str(summary)


def test_load_env_files_keeps_exported_process_env_over_file_values(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN", "exported-token")
    (tmp_path / ".env.alpha.local").write_text(
        "PIPELINE_SYNC_TOKEN=file-token\n",
        encoding="utf-8",
    )

    summary = load_env_files([tmp_path])

    assert os.environ["PIPELINE_SYNC_TOKEN"] == "exported-token"
    assert "PIPELINE_SYNC_TOKEN" in summary["skipped_existing_keys"]
    assert "file-token" not in str(summary)


def test_load_env_files_skips_placeholder_values(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("PRIVACY_SAM3_URL", raising=False)
    (tmp_path / ".env.alpha.local").write_text(
        "PRIVACY_SAM3_URL=REPLACE_ME_PRIVACY_SAM3_URL\n",
        encoding="utf-8",
    )

    summary = load_env_files([tmp_path])

    assert "PRIVACY_SAM3_URL" not in os.environ
    assert "PRIVACY_SAM3_URL" in summary["skipped_placeholder_keys"]


def test_contract_test_env_removes_live_launch_keys(monkeypatch) -> None:
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://tryblueprint.io/api")
    monkeypatch.setenv("PRIVACY_SAM3_URL", "https://privacy.test/sam3")
    monkeypatch.setenv("REGULAR_TEST_VALUE", "kept")

    env = contract_test_env()

    assert "PIPELINE_SYNC_WEBAPP_URL" not in env
    assert "PRIVACY_SAM3_URL" not in env
    assert env["REGULAR_TEST_VALUE"] == "kept"
