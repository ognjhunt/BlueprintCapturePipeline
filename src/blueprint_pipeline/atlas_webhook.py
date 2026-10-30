"""World Labs completion webhooks, used only to wake the Atlas provider poll.

https://docs.atlas-beta.worldlabs.ai/webhooks: a task that named a ``webhookUrl``
(``website_atlas_pose.new_request_body``) gets one Standard Webhooks POST when its
operation succeeds or fails. The event is thin and carries no results, so nothing
in it is acted on: a fresh, first-seen delivery signed by a World Labs key only
asks ``blueprint-website-atlas-provider-poll`` to run now (its path unit watches
``POLL_REQUEST_PATH``), and that poll reads every live operation through the
adapters, as it does every minute anyway. Verification needs no secret: World
Labs signs with Ed25519 and publishes its public keys.

Served by the intake service, which keeps answering through a control-plane
deploy (only timer and path units are quiesced) apart from its own restart once
the release moves; World Labs retries a delivery for 24 hours.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import re
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi import FastAPI, Request, Response

logger = logging.getLogger(__name__)

ROUTE = "/api/live-pipeline/atlas-webhook"
JWKS_URL = "https://api.atlas-beta.worldlabs.ai/.well-known/webhooks/jwks.json"
#: Watched by blueprint-website-atlas-provider-poll.path; the poll removes it as it starts.
POLL_REQUEST_PATH = Path("/var/lib/blueprint/pipeline-control-plane/atlas-provider-poll/requested")
MAX_BODY_BYTES = 16 * 1024
MAX_REQUESTS_PER_MINUTE = 120
TOLERANCE_SECONDS = 300
JWKS_TTL_SECONDS = 24 * 3600
#: At most one JWKS fetch per interval, so unsigned traffic cannot drive fetches.
JWKS_FETCH_INTERVAL_SECONDS = 60
SEEN_LIMIT = 4096
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_DIGITS = re.compile(r"[0-9]{1,12}")


def _fetch_jwks() -> bytes:
    from . import safe_outbound_http as boundary
    policy = boundary.pinned_api_policy(JWKS_URL, max_response_bytes=64 * 1024)
    response = boundary.request(JWKS_URL, headers={"Accept": "application/json"}, timeout_seconds=5, policy=policy)
    if response.status != 200:
        raise ValueError(f"atlas_webhook_jwks_http_{response.status}")
    return response.body


def _public_keys(document: bytes) -> list[Ed25519PublicKey]:
    keys = []
    for row in json.loads(document).get("keys", [])[:16]:
        if isinstance(row, dict) and row.get("kty") == "OKP" and row.get("crv") == "Ed25519":
            x = str(row.get("x") or "")
            keys.append(Ed25519PublicKey.from_public_bytes(base64.urlsafe_b64decode(x + "=" * (-len(x) % 4))))
    if not keys:
        raise ValueError("atlas_webhook_jwks_empty")
    return keys


def _signatures(header: str) -> list[bytes]:
    found = []
    for part in header.split()[:8]:
        if part.startswith("v1a,"):
            try:
                found.append(base64.b64decode(part[4:], validate=True))
            except ValueError:
                continue
    return found


def _verifies(keys: list[Ed25519PublicKey], signatures: list[bytes], signed: bytes) -> bool:
    for signature in signatures:
        for key in keys:
            try:
                key.verify(signature, signed)
                return True
            except (InvalidSignature, ValueError):
                continue
    return False


def _operation_id(body: bytes) -> str | None:
    try:
        value = json.loads(body)["data"]["operationId"]
    except (ValueError, TypeError, KeyError):
        return None
    return value if isinstance(value, str) and _ID.fullmatch(value) else None


class AtlasWebhookReceiver:
    """Verify, deduplicate and turn one delivery into a poll request; no other effect."""

    def __init__(self, *, poll_request: Path = POLL_REQUEST_PATH, fetch_jwks: Callable[[], bytes] = _fetch_jwks,
                 clock: Callable[[], float] = time.time, monotonic: Callable[[], float] = time.monotonic) -> None:
        self._poll_request, self._fetch, self._clock, self._monotonic = poll_request, fetch_jwks, clock, monotonic
        self._lock = threading.Lock()
        self._keys: list[Ed25519PublicKey] = []
        self._keys_at = self._fetched_at = -math.inf
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._recent: deque[float] = deque()

    def _load_keys(self, moment: float) -> bool:
        if moment - self._fetched_at < JWKS_FETCH_INTERVAL_SECONDS:
            return False
        self._fetched_at = moment
        self._keys, self._keys_at = _public_keys(self._fetch()), moment
        return True

    def _verified(self, signed: bytes, header: str) -> bool:
        signatures = _signatures(header)
        if not signatures:
            return False
        moment = self._monotonic()
        if moment - self._keys_at > JWKS_TTL_SECONDS and not self._load_keys(moment):
            raise ValueError("atlas_webhook_jwks_unavailable")
        # The header does not name its key: after a rotation, fetch the keys once more.
        return _verifies(self._keys, signatures, signed) or (
            self._load_keys(moment) and _verifies(self._keys, signatures, signed))

    def receive(self, method: str, headers: Mapping[str, str], body: bytes) -> int:
        """The HTTP status to answer; a 2xx only for a verified delivery whose poll request is recorded."""
        if method != "POST":
            return 405
        if len(body) > MAX_BODY_BYTES:
            return 413
        delivery, stamp, signature = (headers.get(name) or "" for name in
                                      ("webhook-id", "webhook-timestamp", "webhook-signature"))
        with self._lock:
            moment = self._monotonic()
            while self._recent and moment - self._recent[0] >= 60:
                self._recent.popleft()
            if len(self._recent) >= MAX_REQUESTS_PER_MINUTE:
                return 429
            self._recent.append(moment)
            if not _ID.fullmatch(delivery) or not _DIGITS.fullmatch(stamp):
                return 400
            if abs(self._clock() - int(stamp)) > TOLERANCE_SECONDS:
                return 400  # a captured delivery must not stay replayable
            try:
                verified = self._verified(f"{delivery}.{stamp}.".encode() + body, signature)
            except Exception as error:  # noqa: BLE001 - World Labs retries; the timer still polls
                logger.warning("atlas_webhook_keys_unavailable type=%s", type(error).__name__)
                return 503
            if not verified:
                logger.warning("atlas_webhook_signature_rejected id=%s", delivery)
                return 401
            if delivery in self._seen:
                return 204
            try:
                self._poll_request.parent.mkdir(parents=True, exist_ok=True)
                self._poll_request.touch()
            except OSError as error:
                logger.warning("atlas_webhook_poll_request_failed type=%s", type(error).__name__)
                return 503
            self._seen[delivery] = None
            while len(self._seen) > SEEN_LIMIT:
                self._seen.popitem(last=False)
        logger.info("atlas_webhook_poll_requested id=%s operation=%s", delivery, _operation_id(body))
        return 204


def register_atlas_webhook_route(app: FastAPI, receiver: AtlasWebhookReceiver | None = None) -> AtlasWebhookReceiver:
    """POST only (anything else is a 405); the body is read at most ``MAX_BODY_BYTES`` deep."""
    receiver = receiver or AtlasWebhookReceiver()

    @app.post(ROUTE, include_in_schema=False)
    async def atlas_webhook(request: Request) -> Response:
        declared = request.headers.get("content-length", "")
        if declared and (not _DIGITS.fullmatch(declared) or int(declared) > MAX_BODY_BYTES):
            return Response(status_code=413)
        body = b""
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_BODY_BYTES:
                return Response(status_code=413)
        return Response(status_code=await asyncio.to_thread(receiver.receive, "POST", request.headers, body))
    return receiver
