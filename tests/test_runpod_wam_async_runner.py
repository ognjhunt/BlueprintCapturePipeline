from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline import runpod_wam_async_runner as runner
from blueprint_pipeline.runpod_provider_adapter import RUNPOD_API_GATE_ENV


def test_runpod_wam_direct_url_files_block_on_launch_gates_without_leaking_urls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "provider_bundle.zip"
    bundle.write_bytes(b"bundle")
    bundle_url_file = tmp_path / "provider_bundle_url.txt"
    output_url_file = tmp_path / "provider_output_put_url.txt"
    bundle_url_file.write_text(
        "https://spaces.example/bundle.zip?X-Amz-Signature=bundle-secret\n",
        encoding="utf-8",
    )
    output_url_file.write_text(
        "https://spaces.example/output.zip?X-Amz-Signature=output-secret\n",
        encoding="utf-8",
    )
    bundle_url_file.chmod(0o600)
    output_url_file.chmod(0o600)
    monkeypatch.delenv(RUNPOD_API_GATE_ENV, raising=False)
    monkeypatch.delenv(runner.RUNPOD_POD_LAUNCH_GATE_ENV, raising=False)
    monkeypatch.setattr(
        runner,
        "_read_runpod_api_key",
        lambda: (
            "runpod-secret-not-persisted",
            {"api_key_configured": True, "raw_secret_values_recorded": False},
        ),
    )

    manifest = runner.create_runpod_wam_async_run(
        job_dir=tmp_path / "job",
        bundle_path=bundle,
        provider_bundle_url_file=bundle_url_file,
        provider_output_put_url_file=output_url_file,
        skip_public_staging_verification=True,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert manifest["explicit_provider_urls_used"] is True
    assert "paid_runpod_launch_not_authorized_by_runner_flag" in manifest["blockers"]
    assert f"missing_env_{RUNPOD_API_GATE_ENV}" in manifest["blockers"]
    assert f"missing_env_{runner.RUNPOD_POD_LAUNCH_GATE_ENV}" in manifest["blockers"]
    assert manifest["provider_bundle_url_file"]["mode_is_0600"] is True
    assert manifest["provider_output_put_url_file"]["mode_is_0600"] is True
    persisted = (tmp_path / "job" / "runpod_wam_async_create_manifest.json").read_text(
        encoding="utf-8"
    )
    direct_manifest = (
        tmp_path / "job" / "runpod_wam_direct_provider_urls_manifest.json"
    ).read_text(encoding="utf-8")
    assert "bundle-secret" not in persisted
    assert "output-secret" not in persisted
    assert "runpod-secret-not-persisted" not in persisted
    assert "bundle-secret" not in direct_manifest
    assert "output-secret" not in direct_manifest
    parsed = json.loads(direct_manifest)
    assert parsed["provider_bundle_url_redacted"].endswith("?REDACTED_QUERY")
