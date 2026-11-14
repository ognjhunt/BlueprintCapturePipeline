from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

from blueprint_pipeline.openpi_checkpoint_inventory import (
    build_checkpoint_inventory,
    generation_manifest_sha256,
    legacy_object_manifest_sha256,
)


def _objects() -> list[dict[str, str]]:
    return [
        {
            "name": "checkpoints/polaris/test/assets/droid/norm_stats.json",
            "size": "2",
            "md5Hash": base64.b64encode(hashlib.md5(b"{}", usedforsecurity=False).digest()).decode(),
            "crc32c": "abc",
            "generation": "10",
            "metageneration": "1",
            "updated": "2026-01-01T00:00:00Z",
        }
    ]


def test_inventory_freezes_generations_and_matches_legacy_summary(tmp_path: Path) -> None:
    objects = _objects()
    cohort = tmp_path / "cohort.json"
    cohort.write_text(
        json.dumps(
            {
                "schema_version": "policy_ranking_warehouse_policy_cohort.v2",
                "openpi_revision": "rev",
                "primary_cohort": [
                    {
                        "policy_id": "test",
                        "checkpoint": "gs://openpi-assets/checkpoints/polaris/test",
                        "checkpoint_object_count": 1,
                        "checkpoint_size_bytes": 2,
                        "public_object_manifest_sha256": legacy_object_manifest_sha256(objects),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def opener(_url, timeout):
        assert timeout == 30
        return Response(json.dumps({"items": objects}).encode())

    result = build_checkpoint_inventory(cohort, opener=opener)
    assert result["status"] == "frozen"
    assert result["blockers"] == []
    assert result["entries"][0]["generation_manifest_sha256"] == generation_manifest_sha256(objects)
    assert result["claim_boundary"]["checkpoint_downloaded"] is False


def test_generation_changes_identity() -> None:
    before = _objects()
    after = [dict(before[0], generation="11")]
    assert generation_manifest_sha256(before) != generation_manifest_sha256(after)
