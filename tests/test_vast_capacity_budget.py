from __future__ import annotations

from blueprint_pipeline.vast_capacity_budget import expected_transfer_bytes


def test_expected_transfer_bytes_rejects_bool_and_negative_values() -> None:
    download, upload, blockers = expected_transfer_bytes(
        {
            "expected_provider_download_bytes": True,
            "expected_provider_upload_bytes": -1,
        }
    )

    assert (download, upload) == (0, 0)
    assert blockers == [
        "vast_capacity_expected_provider_download_bytes_invalid",
        "vast_capacity_expected_provider_upload_bytes_invalid",
    ]


def test_expected_transfer_bytes_defaults_to_zero() -> None:
    assert expected_transfer_bytes({}) == (0, 0, [])
