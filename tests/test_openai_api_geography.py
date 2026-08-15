"""The Vast geography allowlist must parse the formats Vast actually returns.

The 2026-08-15 launch `adp-ca-b-earth-be9ba935-web-20260815T232801Z` terminally
blocked with `no_vast_offer_at_or_below_max_hourly_rate` while the live market
held 50 eligible offers: the offer endpoint reports geolocation as display
strings ("Sweden, SE", sometimes with an empty region: ", CA") while the parser
accepted only underscore slugs ("california_us"), so every offer failed closed
and `geolocation_country_allowed_offer_count` was 0 of 100.
"""

from __future__ import annotations

from blueprint_pipeline.openai_api_geography import (
    OPENAI_API_SUPPORTED_COUNTRY_CODES,
    vast_country_policy_manifest,
    vast_geolocation_allowed,
    vast_geolocation_country_code,
)


def test_country_code_parses_underscore_slugs() -> None:
    assert vast_geolocation_country_code("california_us") == "us"
    assert vast_geolocation_country_code("hong_kong_hk") == "hk"
    assert vast_geolocation_country_code("se") == "se"


def test_country_code_parses_live_offer_display_strings() -> None:
    assert vast_geolocation_country_code("Sweden, SE") == "se"
    assert vast_geolocation_country_code("California, US") == "us"
    assert vast_geolocation_country_code("South Korea, KR") == "kr"
    assert vast_geolocation_country_code(", CA") == "ca"


def test_country_code_fails_closed_on_unprovable_values() -> None:
    assert vast_geolocation_country_code("Monaco") is None
    assert vast_geolocation_country_code("unknown") is None
    assert vast_geolocation_country_code("") is None
    assert vast_geolocation_country_code(None) is None
    assert vast_geolocation_country_code("u_s_a!") is None


def test_allowlist_admits_display_string_offers() -> None:
    assert vast_geolocation_allowed("Sweden, SE", OPENAI_API_SUPPORTED_COUNTRY_CODES)
    assert not vast_geolocation_allowed(
        "Hong Kong, HK", OPENAI_API_SUPPORTED_COUNTRY_CODES
    )
    assert not vast_geolocation_allowed("Monaco", OPENAI_API_SUPPORTED_COUNTRY_CODES)


def test_policy_manifest_counts_display_string_offers() -> None:
    offers = [
        {"geolocation": "Sweden, SE"},
        {"geolocation": "Hong Kong, HK"},
        {"geolocation": None},
    ]
    manifest = vast_country_policy_manifest(offers, OPENAI_API_SUPPORTED_COUNTRY_CODES)
    assert manifest["geolocation_country_allowed_offer_count"] == 1
