"""Source registry and the fail-closed license check."""

import copy

import pytest

from tests.site_universe_fixture import seeded_cache
from tools.site_universe import build, sources
from tools.site_universe.sources import SourceRefused


def test_every_registered_source_passes_the_check_and_records_verification():
    registry = sources.registry()
    assert set(registry) == {"osha_ita", "epa_frs", "osm_overpass", "fsis_mpi"}
    for entry in registry.values():
        sources.check_source(entry)
        assert entry["verification"]["verified_at"] == "2026-10-04"
        assert entry["verification"]["evidence"]
        for item in entry["verification"]["evidence"]:
            assert item["url"].startswith("https://") and item["observed"]
        assert entry["license"]["url"].startswith("https://")
        assert entry["personal_data_policy"]
    osm = registry["osm_overpass"]
    assert (osm["license"]["id"], osm["attribution"], osm["share_alike"]) == (
        "ODbL-1.0", "© OpenStreetMap contributors", True,
    )
    assert registry["osha_ita"]["status"] == "manual_import_only"
    assert registry["fsis_mpi"]["status"] == "manual_import_only"
    assert registry["epa_frs"]["status"] == "enabled"


def _entry(**changes):
    entry = copy.deepcopy(sources.registry()["epa_frs"])
    for key, value in changes.items():
        if value is None:
            entry.pop(key, None)
        else:
            entry[key] = value
    return entry


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"license": None}, "no recorded license"),
        ({"license": {"id": "", "url": "https://example.org"}}, "no recorded license"),
        ({"license": {"id": "US-PD", "url": ""}}, "no recorded license"),
        ({"allowed_uses": None}, "no recorded allowed uses"),
        ({"allowed_uses": []}, "no recorded allowed uses"),
        ({"allowed_uses": ["internal_derivative_database"]}, "commercial_use"),
        ({"attribution_required": True, "attribution": None}, "attribution"),
        ({"attribution_required": None}, "attribution requirement"),
        ({"share_alike": None}, "share-alike"),
        ({"share_alike": "no"}, "share-alike"),
        ({"verification": {"evidence": []}}, "verification"),
        ({"status": "skipped", "skip_reason": "license unclear"}, "license unclear"),
        ({"status": "sometimes"}, "unknown status"),
        ({"download_url_template": None}, "without a download"),
    ],
)
def test_check_source_fails_closed(changes, message):
    with pytest.raises(SourceRefused, match=message):
        sources.check_source(_entry(**changes))


def test_build_refuses_an_unlicensed_source_before_reading_any_raw_input(tmp_path, monkeypatch):
    broken = sources.registry()
    broken["epa_frs"]["license"] = {}
    monkeypatch.setattr(sources, "registry", lambda: copy.deepcopy(broken))

    class NoTouch:
        def __getattr__(self, name):
            raise AssertionError("the raw cache must not be touched")

    with pytest.raises(SourceRefused, match="no recorded license"):
        build.build(["TX"], ["epa_frs"], out_dir=tmp_path / "out", cache=NoTouch())
    assert not (tmp_path / "out" / build.SITES_FILE).exists()


def test_build_refuses_unknown_sources(tmp_path):
    cache = seeded_cache(tmp_path / "raw")
    with pytest.raises(SourceRefused, match="unknown sources"):
        build.build(["TX"], ["epa_frs", "acme_scraper"], out_dir=tmp_path / "out", cache=cache)
