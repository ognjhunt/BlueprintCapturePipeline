from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import native_g1_checkpoint_cache as cache
from blueprint_pipeline.native_g1_development_pair import PAIR_ORDER


def test_private_checkpoint_transfer_seals_urls_and_scrubs_after_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = tmp_path / "models"
    model_root.mkdir()
    for candidate in PAIR_ORDER:
        path = model_root / candidate / "model.safetensors"
        path.parent.mkdir()
        path.write_bytes(b"pinned-model")

    def verify(*, inventory_path: Path, candidate_id: str, output_dir: Path,
               verify_only: bool) -> dict:
        assert verify_only is True
        assert output_dir == model_root
        return {
            "status": "checkpoint_bytes_verified",
            "files": [{
                "relative_path": candidate_id + "/model.safetensors",
                "sha256": "sha256:" + "a" * 64,
                "size_bytes": len(b"pinned-model"),
            }],
        }

    monkeypatch.setattr(cache, "_fetcher", lambda: SimpleNamespace(materialize_candidate=verify))

    def stage(*, job_dir: Path, dependency_path: Path, expected_sha256: str,
              key_prefix: str, expiration_seconds: int, artifact_kind: str) -> dict:
        assert dependency_path.is_file()
        assert expected_sha256 == "sha256:" + "a" * 64
        assert artifact_kind == "g1_checkpoint"
        assert expiration_seconds == 18_000
        job_dir.mkdir()
        (job_dir / cache.RUNTIME_DEPENDENCY_URL_FILENAME).write_text(
            "https://private.example/model?signature=secret", encoding="utf-8"
        )
        return {"status": "completed", "remote_identity_verified": True,
                "cache_hit": True, "upload_performed": False}

    monkeypatch.setattr(cache, "stage_cached_runtime_dependency_object_store", stage)
    monkeypatch.setattr(cache, "close_cached_runtime_dependency_staging", lambda _: {})
    job = tmp_path / "transfer"
    result = cache.stage_g1_checkpoint_cache(
        cache_root=model_root, job_dir=job,
        key_prefix="blueprint/g1", expiration_seconds=18_000,
    )
    secret_path = job / "g1_checkpoint_transfer_urls.json"
    assert result["status"] == "completed"
    assert result["file_count"] == len(PAIR_ORDER)
    assert result["raw_signed_urls_recorded"] is False
    assert "signature=secret" not in json.dumps(result)
    assert secret_path.stat().st_mode & 0o777 == 0o600
    assert len(json.loads(secret_path.read_text())["files"]) == len(PAIR_ORDER)
    closeout = cache.close_g1_checkpoint_cache(job)
    assert closeout["signed_url_file_removed"] is True
    assert not secret_path.exists()
