"""Raw cache: the only network boundary. Exercised with a fake opener; no network."""

import hashlib
import json

import pytest

from tests.site_universe_fixture import FakeOpener
from tools.site_universe.fetch import (
    RAW_MANIFEST_NAME,
    FetchError,
    OfflineMiss,
    RawCache,
    SourceBlocked,
)

URL = "https://example.gov/files/state_combined_tx.zip"
BODY = b"PK fixture bytes"


def _cache(tmp_path, opener, **kwargs):
    stamps = iter(["2026-10-04T01:00:00Z", "2026-10-04T02:00:00Z", "2026-10-04T03:00:00Z"])
    return RawCache(tmp_path, opener=opener, clock=lambda: next(stamps), sleep=lambda s: None, **kwargs)


def test_download_records_sha256_and_time_and_is_reused(tmp_path):
    opener = FakeOpener({URL: BODY})
    cache = _cache(tmp_path, opener)
    entry = cache.get(URL, source_id="epa_frs")
    assert entry.sha256 == hashlib.sha256(BODY).hexdigest()
    assert entry.retrieved_at == "2026-10-04T01:00:00Z" and entry.acquisition == "network"
    assert entry.read_bytes() == BODY and entry.path.suffix == ".zip"
    manifest = json.loads((tmp_path / RAW_MANIFEST_NAME).read_text())
    assert manifest["entries"][URL]["sha256"] == entry.sha256
    assert manifest["entries"][URL]["retrieved_at"] == "2026-10-04T01:00:00Z"
    again = RawCache(tmp_path, opener=opener).get(URL, source_id="epa_frs")
    assert again.retrieved_at == entry.retrieved_at
    assert opener.requests == [URL]


def test_refresh_downloads_again(tmp_path):
    opener = FakeOpener({URL: BODY})
    _cache(tmp_path, opener).get(URL, source_id="epa_frs")
    opener.bodies[URL] = b"PK newer bytes"
    refreshed = _cache(tmp_path, opener, refresh=True)
    entry = refreshed.get(URL, source_id="epa_frs")
    assert entry.sha256 == hashlib.sha256(b"PK newer bytes").hexdigest()
    assert refreshed.get(URL, source_id="epa_frs").sha256 == entry.sha256
    assert opener.requests == [URL, URL]


def test_offline_miss_and_refused_urls(tmp_path):
    cache = RawCache(tmp_path, allow_network=False)
    with pytest.raises(OfflineMiss):
        cache.get(URL, source_id="epa_frs")
    online = _cache(tmp_path, FakeOpener({}))
    with pytest.raises(FetchError, match="https"):
        online.get("http://example.gov/plain.zip", source_id="epa_frs")
    with pytest.raises(FetchError, match="credentials"):
        online.get("https://user:secret@example.gov/a.zip", source_id="epa_frs")


def test_bot_protection_is_reported_not_bypassed(tmp_path):
    opener = FakeOpener({URL: BODY}, errors={URL: [403]})
    with pytest.raises(SourceBlocked):
        _cache(tmp_path, opener).get(URL, source_id="osha_ita")
    assert opener.requests == [URL]


def test_rate_limit_is_retried_politely(tmp_path):
    waits = []
    opener = FakeOpener({URL: BODY}, errors={URL: [429]})
    cache = RawCache(tmp_path, opener=opener, clock=lambda: "2026-10-04T01:00:00Z", sleep=waits.append)
    assert cache.get(URL, source_id="osm_overpass").sha256 == hashlib.sha256(BODY).hexdigest()
    assert waits == [30.0] and opener.requests == [URL, URL]
    always = FakeOpener({URL: BODY}, errors={URL: [503, 503, 503]})
    with pytest.raises(FetchError, match="503"):
        RawCache(tmp_path / "b", opener=always, sleep=lambda s: None).get(URL, source_id="osm_overpass")
    assert len(always.requests) == 3


def test_invalid_payload_is_not_cached(tmp_path):
    def refuse(path):
        raise FetchError("partial response")

    cache = _cache(tmp_path, FakeOpener({URL: BODY}))
    with pytest.raises(FetchError, match="partial"):
        cache.get(URL, source_id="osm_overpass", validate=refuse)
    assert cache.cached(URL) is None
    assert not list(tmp_path.glob("*.zip"))


def test_tampered_cache_file_fails_closed(tmp_path):
    cache = _cache(tmp_path, FakeOpener({URL: BODY}))
    entry = cache.get(URL, source_id="epa_frs")
    entry.path.write_bytes(b"changed")
    with pytest.raises(FetchError, match="SHA-256"):
        RawCache(tmp_path, allow_network=False).get(URL, source_id="epa_frs")
    with pytest.raises(FetchError):
        entry.read_bytes()


def test_manual_import_keeps_provenance(tmp_path):
    local = tmp_path / "downloaded.csv"
    local.write_bytes(b"id,establishment_name\n")
    cache = RawCache(tmp_path / "raw", allow_network=False)
    entry = cache.import_file("https://www.osha.gov/x.csv", local, source_id="osha_ita",
                              retrieved_at="2026-10-04T09:00:00Z")
    assert entry.acquisition == "manual_import" and entry.retrieved_at == "2026-10-04T09:00:00Z"
    assert entry.sha256 == hashlib.sha256(local.read_bytes()).hexdigest()
    reopened = RawCache(tmp_path / "raw", allow_network=False)
    assert reopened.get("https://www.osha.gov/x.csv", source_id="osha_ita").sha256 == entry.sha256
