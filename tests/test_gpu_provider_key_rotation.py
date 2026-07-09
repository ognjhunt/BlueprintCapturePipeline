from __future__ import annotations

import json
from datetime import datetime, timezone

from blueprint_pipeline.gpu_provider_key_rotation import (
    GPU_PROVIDER_KEY_DESCRIPTORS,
    build_gpu_provider_key_rotation_manifest,
    mark_gpu_provider_key_rotated,
)


NOW = datetime(2026, 7, 8, 12, 0, 0, tzinfo=timezone.utc)


def _write_provider_secret_files(secrets_dir):
    secrets_dir.mkdir(parents=True, exist_ok=True)
    for provider, descriptor in GPU_PROVIDER_KEY_DESCRIPTORS.items():
        (secrets_dir / descriptor.default_secret_filename).write_text(
            f"{provider}-secret-value-that-must-not-leak",
            encoding="utf-8",
        )


def test_rotation_manifest_blocks_missing_metadata_without_secret_leakage(tmp_path):
    secrets_dir = tmp_path / "secrets"
    _write_provider_secret_files(secrets_dir)
    ledger_path = secrets_dir / "rotation_ledger.json"

    manifest = build_gpu_provider_key_rotation_manifest(
        secrets_dir=secrets_dir,
        ledger_path=ledger_path,
        owner="security",
        now=NOW,
    )

    assert manifest["status"] == "blocked"
    assert manifest["secret_values_recorded"] is False
    for provider in GPU_PROVIDER_KEY_DESCRIPTORS:
        assert f"{provider}:rotation_metadata_missing" in manifest["blockers"]
        assert manifest["providers"][provider]["secret_file"]["present"] is True
        assert manifest["providers"][provider]["secret_file"]["path_redacted"] is True
        assert "path" not in manifest["providers"][provider]["secret_file"]
        assert manifest["providers"][provider]["default_secret_file_path_redacted"] is True
        assert manifest["providers"][provider]["secret_value_recorded"] is False

    serialized = json.dumps(manifest)
    assert str(secrets_dir) not in serialized
    assert "secret-value-that-must-not-leak" not in serialized


def test_mark_rotated_records_owner_and_manifest_passes_when_all_providers_fresh(tmp_path):
    secrets_dir = tmp_path / "secrets"
    _write_provider_secret_files(secrets_dir)
    ledger_path = secrets_dir / "rotation_ledger.json"

    for provider in GPU_PROVIDER_KEY_DESCRIPTORS:
        mark_gpu_provider_key_rotated(
            provider=provider,
            ledger_path=ledger_path,
            owner="platform-security",
            rotation_record_uri=f"secret-manager://blueprint/{provider}/versions/7",
            rotated_at=NOW,
            now=NOW,
        )

    manifest = build_gpu_provider_key_rotation_manifest(
        secrets_dir=secrets_dir,
        ledger_path=ledger_path,
        owner="platform-security",
        now=NOW,
    )

    assert manifest["status"] == "passed"
    assert manifest["blockers"] == []
    for provider, payload in manifest["providers"].items():
        assert payload["status"] == "passed"
        assert payload["rotation_owner"] == "platform-security"
        assert payload["rotation_record_uri"] == (
            f"secret-manager://blueprint/{provider}/versions/7"
        )
        assert payload["days_since_rotation"] == 0
        assert payload["inline_secret_values_recorded"] is False


def test_stale_rotation_metadata_blocks_provider(tmp_path):
    secrets_dir = tmp_path / "secrets"
    _write_provider_secret_files(secrets_dir)
    ledger_path = secrets_dir / "rotation_ledger.json"

    mark_gpu_provider_key_rotated(
        provider="runpod",
        ledger_path=ledger_path,
        owner="platform-security",
        rotation_record_uri="ticket://SEC-123",
        rotated_at=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        now=NOW,
    )

    manifest = build_gpu_provider_key_rotation_manifest(
        secrets_dir=secrets_dir,
        ledger_path=ledger_path,
        owner="platform-security",
        providers=["runpod"],
        max_age_days=30,
        now=NOW,
    )

    assert manifest["status"] == "blocked"
    assert manifest["providers"]["runpod"]["status"] == "blocked"
    assert "runpod:rotation_stale" in manifest["blockers"]
