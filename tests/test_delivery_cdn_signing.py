"""Cloud CDN delivery signing (SCALE2-04) — flagged, fallback-safe.

See docs/BUYER_DELIVERY_CDN_DESIGN_2026-07-20.md. These tests pin the Cloud
CDN signed-URL scheme (HMAC-SHA1 over the full URL, base64url) and the
fail-quiet config gating that keeps direct GCS signed URLs as the fallback.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from pathlib import Path

import pytest

from blueprint_pipeline.arena_package_delivery_local import (
    CDN_BASE_URL_ENV,
    CDN_ENABLED_ENV,
    CDN_KEY_FILE_ENV,
    CDN_KEY_NAME_ENV,
    _cdn_delivery_config,
    _generate_cdn_signed_url,
)

KEY_BYTES = b"0123456789abcdef"
KEY_B64URL = base64.urlsafe_b64encode(KEY_BYTES).decode("utf-8")


def test_cdn_signed_url_matches_reference_hmac() -> None:
    url = _generate_cdn_signed_url(
        base_url="https://delivery.example.com",
        object_name="deliveries/run-1/archives/post_training_data_package.tar.gz",
        key_name="delivery-key-1",
        key_bytes=KEY_BYTES,
        ttl_seconds=900,
        now_epoch=1_752_000_000,
    )
    expected_unsigned = (
        "https://delivery.example.com/deliveries/run-1/archives/"
        "post_training_data_package.tar.gz?Expires=1752000900&KeyName=delivery-key-1"
    )
    expected_signature = base64.urlsafe_b64encode(
        hmac.new(KEY_BYTES, expected_unsigned.encode("utf-8"), hashlib.sha1).digest()
    ).decode("utf-8")
    assert url == f"{expected_unsigned}&Signature={expected_signature}"


def test_cdn_signed_url_is_deterministic_for_fixed_epoch() -> None:
    kwargs = dict(
        base_url="https://delivery.example.com/",
        object_name="/a/b.tar.gz",
        key_name="k",
        key_bytes=KEY_BYTES,
        ttl_seconds=60,
        now_epoch=1_000,
    )
    assert _generate_cdn_signed_url(**kwargs) == _generate_cdn_signed_url(**kwargs)
    assert "Expires=1060" in _generate_cdn_signed_url(**kwargs)


def _set_full_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    key_file = tmp_path / "cdn_url_signing_key"
    key_file.write_text(KEY_B64URL, encoding="utf-8")
    monkeypatch.setenv(CDN_ENABLED_ENV, "1")
    monkeypatch.setenv(CDN_BASE_URL_ENV, "https://delivery.example.com")
    monkeypatch.setenv(CDN_KEY_NAME_ENV, "delivery-key-1")
    monkeypatch.setenv(CDN_KEY_FILE_ENV, str(key_file))
    return key_file


def test_cdn_config_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CDN_ENABLED_ENV, raising=False)
    assert _cdn_delivery_config() is None


def test_cdn_config_resolves_when_complete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_full_config(monkeypatch, tmp_path)
    config = _cdn_delivery_config()
    assert config is not None
    assert config["base_url"] == "https://delivery.example.com"
    assert config["key_name"] == "delivery-key-1"
    assert config["key_bytes"] == KEY_BYTES


@pytest.mark.parametrize(
    "mutation",
    ["missing_key_file", "http_base_url", "missing_key_name", "empty_key"],
)
def test_cdn_config_fails_quietly_to_gcs_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    key_file = _set_full_config(monkeypatch, tmp_path)
    if mutation == "missing_key_file":
        key_file.unlink()
    elif mutation == "http_base_url":
        monkeypatch.setenv(CDN_BASE_URL_ENV, "http://delivery.example.com")
    elif mutation == "missing_key_name":
        monkeypatch.setenv(CDN_KEY_NAME_ENV, "")
    elif mutation == "empty_key":
        key_file.write_text("", encoding="utf-8")
    assert _cdn_delivery_config() is None
