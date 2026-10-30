"""World Labs completion webhooks only wake the Atlas provider poll (docs.atlas-beta.worldlabs.ai/webhooks).

Keys here are generated per test; nothing reaches World Labs.
"""
import base64
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI
from fastapi.testclient import TestClient

from blueprint_pipeline import atlas_webhook as webhook

ROOT = Path(__file__).resolve().parents[1]
SYSTEMD = ROOT / "deploy" / "systemd"
UNIT = "blueprint-website-atlas-provider-poll"


def _jwk(private):
    raw = private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return {"kty": "OKP", "crv": "Ed25519", "alg": "EdDSA", "use": "sig", "kid": "test",
            "x": base64.urlsafe_b64encode(raw).rstrip(b"=").decode()}


class Keys:
    """The published JWKS as World Labs answers it; counts every fetch."""

    def __init__(self, *private):
        self.private, self.fetches, self.down = list(private), 0, False

    def __call__(self):
        self.fetches += 1
        if self.down:
            raise OSError("jwks unreachable")
        return json.dumps({"keys": [{"kty": "RSA", "n": "ignored"}, *map(_jwk, self.private)]}).encode()


def _event(operation="op_01jexample"):
    return json.dumps({"id": "evt_op_01jexample", "type": "task.succeeded", "schemaVersion": "1",
                       "createdAt": "2026-10-30T18:10:00Z",
                       "data": {"accountId": "acct_example", "projectId": "project_example", "operationId": operation,
                                "taskName": "posedImages2Splats", "state": "SUCCEEDED"}}).encode()


def _headers(private, body, *, delivery="del_01jexample", stamp=None):
    stamp = str(int(time.time()) if stamp is None else stamp)
    signature = base64.b64encode(private.sign(f"{delivery}.{stamp}.".encode() + body)).decode()
    return {"webhook-id": delivery, "webhook-timestamp": stamp, "webhook-signature": f"v1a,{signature}"}


@pytest.fixture
def hook(tmp_path):
    key = Ed25519PrivateKey.generate()
    keys, clock = Keys(key), [1000.0]
    request = tmp_path / "atlas-provider-poll" / "requested"
    receiver = webhook.AtlasWebhookReceiver(poll_request=request, fetch_jwks=keys, monotonic=lambda: clock[0])
    app = FastAPI()
    webhook.register_atlas_webhook_route(app, receiver)
    return SimpleNamespace(key=key, keys=keys, clock=clock, receiver=receiver, request=request,
                           client=TestClient(app))


def _post(hook, body, headers):
    return hook.client.post(webhook.ROUTE, content=body, headers=headers)


def test_a_signed_fresh_delivery_requests_one_poll_and_logs_only_ids(hook, caplog):
    body = _event()
    with caplog.at_level(logging.INFO, logger=webhook.__name__):
        response = _post(hook, body, _headers(hook.key, body))
    assert response.status_code == 204 and hook.request.is_file() and hook.keys.fetches == 1
    logged = caplog.text
    assert "del_01jexample" in logged and "op_01jexample" in logged
    assert all(field not in logged for field in ("posedImages2Splats", "SUCCEEDED", "acct_example", "createdAt"))


def test_a_duplicate_delivery_is_acknowledged_without_a_second_poll(hook):
    body = _event()
    headers = _headers(hook.key, body)
    assert _post(hook, body, headers).status_code == 204
    hook.request.unlink()  # the poll consumed the first request
    assert _post(hook, body, headers).status_code == 204
    assert not hook.request.exists()
    other = _headers(hook.key, body, delivery="del_02jexample")
    assert _post(hook, body, other).status_code == 204 and hook.request.is_file()


@pytest.mark.parametrize("case, status", [
    ("other_key", 401), ("altered_body", 401), ("not_v1a", 401), ("stale", 400), ("future", 400),
    ("no_signature", 401), ("no_id", 400), ("unsafe_id", 400), ("no_timestamp", 400), ("unicode_timestamp", 400),
])
def test_an_unverified_or_stale_delivery_is_refused_and_requests_nothing(hook, case, status):
    body = _event()
    signer = Ed25519PrivateKey.generate() if case == "other_key" else hook.key
    stamp = {"stale": int(time.time()) - 301, "future": int(time.time()) + 301}.get(case)
    headers = _headers(signer, body, stamp=stamp)
    if case == "altered_body":
        body = _event("op_someone_else")
    elif case == "not_v1a":
        headers["webhook-signature"] = headers["webhook-signature"].replace("v1a,", "v1,")
    elif case.startswith("no_"):
        headers.pop({"no_signature": "webhook-signature", "no_id": "webhook-id",
                     "no_timestamp": "webhook-timestamp"}[case])
    elif case == "unsafe_id":
        headers["webhook-id"] = "del\nforged log line"
    elif case == "unicode_timestamp":
        headers["webhook-timestamp"] = ("²" * 3).encode("latin-1")  # str.isdigit() says yes
    assert _post(hook, body, headers).status_code == status
    assert not hook.request.exists()


def test_only_post_is_accepted(hook):
    assert hook.client.get(webhook.ROUTE).status_code == 405
    assert hook.client.put(webhook.ROUTE, content=b"{}").status_code == 405
    assert hook.receiver.receive("GET", {}, b"") == 405
    assert not hook.request.exists()


def test_an_oversized_body_is_refused_before_it_is_read_whole(hook):
    body = b" " * (webhook.MAX_BODY_BYTES + 1)
    assert _post(hook, body, _headers(hook.key, body)).status_code == 413

    def chunks():  # no Content-Length: the stream itself is cut off
        for _ in range(webhook.MAX_BODY_BYTES // 1024 + 2):
            yield b" " * 1024
    assert hook.client.post(webhook.ROUTE, content=chunks(), headers=_headers(hook.key, b"")).status_code == 413
    assert not hook.request.exists() and hook.keys.fetches == 0


def test_a_rotated_key_is_fetched_once_and_unsigned_traffic_cannot_drive_fetches(hook):
    body = _event()
    assert _post(hook, body, _headers(hook.key, body)).status_code == 204 and hook.keys.fetches == 1
    rotated = Ed25519PrivateKey.generate()
    hook.keys.private = [rotated]
    hook.clock[0] += 3600
    signed = _headers(rotated, body, delivery="del_rotated")
    assert _post(hook, body, signed).status_code == 204 and hook.keys.fetches == 2
    forged = _headers(Ed25519PrivateKey.generate(), body, delivery="del_forged")
    for _ in range(5):
        assert _post(hook, body, forged).status_code == 401
    assert hook.keys.fetches == 2  # one refetch per interval at most
    hook.clock[0] += webhook.JWKS_FETCH_INTERVAL_SECONDS
    assert _post(hook, body, forged).status_code == 401 and hook.keys.fetches == 3


def test_keys_are_cached_for_a_day_and_an_unreachable_jwks_asks_for_a_retry(hook):
    body = _event()
    for index in range(3):
        assert _post(hook, body, _headers(hook.key, body, delivery=f"del_{index}")).status_code == 204
    assert hook.keys.fetches == 1
    hook.clock[0] += webhook.JWKS_TTL_SECONDS + 1
    hook.keys.down = True
    hook.request.unlink()
    assert _post(hook, body, _headers(hook.key, body, delivery="del_late")).status_code == 503
    assert not hook.request.exists()


def test_the_request_rate_is_bounded(hook):
    body = _event()
    headers = _headers(Ed25519PrivateKey.generate(), body)
    statuses = [hook.receiver.receive("POST", headers, body) for _ in range(webhook.MAX_REQUESTS_PER_MINUTE + 3)]
    assert statuses.count(429) == 3 and set(statuses[:-3]) == {401}
    hook.clock[0] += 60
    assert hook.receiver.receive("POST", _headers(hook.key, body), body) == 204


def test_the_intake_serves_the_route_and_the_poll_units_consume_what_it_writes():
    from blueprint_pipeline.live_pipeline_intake_service import app
    routes = [route for route in app.routes if getattr(route, "path", None) == webhook.ROUTE]
    assert len(routes) == 1 and routes[0].methods == {"POST"}
    watched = (SYSTEMD / f"{UNIT}.path").read_text()
    service = (SYSTEMD / f"{UNIT}.service").read_text()
    assert f"PathExists={webhook.POLL_REQUEST_PATH}\n" in watched and f"Unit={UNIT}.service" in watched
    assert f"rm -f {webhook.POLL_REQUEST_PATH};" in service
    caddy = (ROOT / "deploy" / "caddy" / "Caddyfile").read_text()
    assert "handle /api/live-pipeline/* {\n\t\treverse_proxy 127.0.0.1:8765" in caddy


def test_the_poll_unit_is_the_preview_workers_read_only_sibling():
    preview = (SYSTEMD / "blueprint-website-atlas-preview.service").read_text()
    service = (SYSTEMD / f"{UNIT}.service").read_text()
    assert "-m blueprint_pipeline.website_atlas_preview poll --storage-root" in service
    assert "paid_resource_allocator" not in service and " tick " not in service
    same = [line for line in preview.splitlines() if line.startswith((
        "User=", "Group=", "UMask=", "EnvironmentFile=", "NoNewPrivileges=", "Private", "Protect", "Restrict",
        "CapabilityBoundingSet=", "AmbientCapabilities=", "SystemCallFilter=", "ReadWritePaths="))]
    assert len(same) > 15 and all(line in service.splitlines() for line in same)
    timer = (SYSTEMD / f"{UNIT}.timer").read_text()
    assert "OnUnitInactiveSec=60s" in timer and f"Unit={UNIT}.service" in timer
    installer = (ROOT / "scripts" / "install_live_pipeline_control_plane.sh").read_text()
    mover = (ROOT / "deploy" / "host" / "mount_work_volume.sh").read_text()
    for suffix in ("service", "timer", "path"):
        assert f"deploy/systemd/{UNIT}.{suffix}" in installer and f"  {UNIT}.{suffix}\n" in mover
    for trigger in ("timer", "path"):
        assert f"systemctl enable --now {UNIT}.{trigger}" in installer
