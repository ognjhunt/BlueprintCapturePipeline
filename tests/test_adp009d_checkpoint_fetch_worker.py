from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import adp009d_checkpoint_fetch_worker as fetcher


def test_listing_follows_pagination_to_the_end(monkeypatch) -> None:
    """Stopping at the first page would silently fetch a partial model."""

    pages = [
        {"items": [{"name": "p/a", "size": "10"}], "nextPageToken": "t1"},
        {"items": [{"name": "p/b", "size": "20"}], "nextPageToken": "t2"},
        {"items": [{"name": "p/c", "size": "30"}]},
    ]
    seen: list[str] = []

    def _fake(url):
        seen.append(url)
        return pages[len(seen) - 1]

    monkeypatch.setattr(fetcher, "_request_json", _fake)
    objects = fetcher.list_gcs_objects("openpi-assets", "p/")

    assert [o["name"] for o in objects] == ["p/a", "p/b", "p/c"]
    assert len(seen) == 3
    assert "pageToken=t1" in seen[1]
    assert "pageToken=t2" in seen[2]


def test_no_authorization_header_is_ever_sent(monkeypatch) -> None:
    """Every frozen candidate is public; a token would mean a different artifact."""

    import inspect

    source = inspect.getsource(fetcher)
    # No request ever carries headers, which is the property that matters --
    # more robust than scanning prose for the word "Authorization".
    assert "add_header" not in source
    assert "headers=" not in source
    assert "HF_TOKEN" not in source
    assert "credentials_used" in source


def test_a_size_mismatch_fails_and_is_never_retried(monkeypatch, tmp_path) -> None:
    """Refetching what produced wrong bytes produces wrong bytes again."""

    monkeypatch.setattr(
        fetcher, "list_gcs_objects", lambda b, p: [{"name": "p/a", "size": "100"}]
    )
    calls: list[str] = []

    def _short(bucket, name, destination):
        calls.append(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"x" * 99)
        return 99

    monkeypatch.setattr(fetcher, "download_object", _short)

    with pytest.raises(RuntimeError) as excinfo:
        fetcher.fetch_gcs_prefix(uri="gs://openpi-assets/p", destination_root=tmp_path)
    assert "size_mismatch" in str(excinfo.value)
    assert calls == ["p/a"], "a wrong-size object must not be refetched"


def test_an_empty_prefix_is_a_failure_not_an_empty_success(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(fetcher, "list_gcs_objects", lambda b, p: [])

    with pytest.raises(RuntimeError) as excinfo:
        fetcher.fetch_gcs_prefix(uri="gs://openpi-assets/p", destination_root=tmp_path)
    assert "prefix_empty" in str(excinfo.value)


def test_a_successful_fetch_reports_exactly_what_landed(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        fetcher,
        "list_gcs_objects",
        lambda b, p: [
            {"name": "p/b", "size": "20"},
            {"name": "p/a", "size": "10"},
        ],
    )

    def _ok(bucket, name, destination):
        size = 20 if name.endswith("b") else 10
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"x" * size)
        return size

    monkeypatch.setattr(fetcher, "download_object", _ok)

    receipt = fetcher.fetch_gcs_prefix(
        uri="gs://openpi-assets/p", destination_root=tmp_path
    )

    assert receipt["status"] == "fetched"
    assert receipt["object_count"] == 2
    assert receipt["total_bytes"] == 30
    assert receipt["credentials_used"] is False
    # Reported in a stable order, so the receipt is comparable across runs.
    assert [row["name"] for row in receipt["objects"]] == ["p/a", "p/b"]
    assert json.dumps(receipt)


def test_the_bundle_ships_the_fetcher_and_provisioning_calls_it() -> None:
    """gcloud is not assumed to exist in an Isaac container."""

    from blueprint_pipeline.adp009d_native_microcheck_bundle import ENTRYPOINT
    from blueprint_pipeline.adp009d_policy_provisioning import (
        build_provisioning_script,
    )

    bundle_source = Path(
        __import__("blueprint_pipeline.adp009d_native_microcheck_bundle", fromlist=["x"]).__file__
    ).read_text(encoding="utf-8")
    assert "adp009d_checkpoint_fetch_worker.py" in bundle_source

    script = build_provisioning_script("pi05_droid")
    assert "adp009d_checkpoint_fetch_worker.py" in script
    assert "gcloud" not in script
    # The fetcher needs the paths the entrypoint owns.
    assert 'RUNTIME_DIR="$RUNTIME_DIR" OUT_DIR="$OUT_DIR"' in ENTRYPOINT
