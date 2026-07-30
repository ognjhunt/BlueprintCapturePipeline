from __future__ import annotations

import pytest

from blueprint_pipeline.public_gcs_checkpoint_inventory import (
    build_public_gcs_checkpoint_inventory,
)


def _payload():
    return [
        {"url": "gs://bucket/checkpoint/", "type": "prefix"},
        {
            "url": "gs://bucket/checkpoint/empty",
            "type": "cloud_object",
            "metadata": {
                "name": "checkpoint/empty",
                "size": "0",
                "generation": "1",
                "metageneration": "1",
                "md5Hash": "zero",
                "crc32c": "zero",
                "updated": "2026-01-01T00:00:00Z",
            },
        },
        {
            "url": "gs://bucket/checkpoint/weights",
            "type": "cloud_object",
            "metadata": {
                "name": "checkpoint/weights",
                "size": "123",
                "generation": "42",
                "metageneration": "1",
                "md5Hash": "base64-md5",
                "crc32c": "base64-crc",
                "updated": "2026-01-02T00:00:00Z",
            },
        },
    ]


def test_inventory_freezes_nonempty_object_identities() -> None:
    result = build_public_gcs_checkpoint_inventory(
        source_uri="gs://bucket/checkpoint",
        payload=_payload(),
        observed_at="2026-07-30T12:00:00+00:00",
    )

    assert result["object_count"] == 1
    assert result["total_bytes"] == 123
    assert result["objects"][0]["generation"] == "42"
    assert result["objects"][0]["md5_base64"] == "base64-md5"
    assert len(result["object_inventory_sha256"]) == 64
    assert len(result["manifest_sha256"]) == 64
    assert result["raw_secret_values_recorded"] is False


def test_inventory_rejects_non_gcs_source() -> None:
    with pytest.raises(ValueError, match="uri_invalid"):
        build_public_gcs_checkpoint_inventory(
            source_uri="https://example.invalid/checkpoint",
            payload=_payload(),
            observed_at="2026-07-30T12:00:00+00:00",
        )
