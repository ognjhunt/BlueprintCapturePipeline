"""Admitted World Labs Atlas images2PosedRGBD operation for website source registration.

``website_worldlabs.prepare_atlas_pose_inputs`` records the source frames as
pending and waits for an admitted producer. This is that producer. It uploads
those exact frames as project assets, submits one operation, saves the
operation id before polling, and keeps every returned file bound to the input
frame it came from. ``paid_resource_allocator.run_sponsored_website_atlas_pose``
is its only caller, which ``scripts/verify_paid_resource_allocator.py`` pins,
and the default geometry route never reaches it.

Development-only. The docs give depth in "world units" with no stated metric
scale, and the returned pixels are reframed onto the provider's grid with no
source-pixel transform. Nothing here is measured evidence, native geometry, a
world, a splat or a collider.
https://atlas-beta.worldlabs.ai/docs/images-to-posed-images
https://atlas-beta.worldlabs.ai/docs/api-reference
"""

from __future__ import annotations

import fcntl
import http.client
import ipaddress
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
import urllib.error
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Mapping, Sequence
from urllib.parse import urlencode, urlsplit

from .atlas_spend_settlement import settle_closed_atlas_pose_attempts
from .core.common import redacted_failure_text
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_bytes, _sha256_file
from .paid_resource_admission import PaidResourceAdmissionGrant, require_paid_resource_admission_grant

_API_ROOT = "/api/v2"
_POSE_TASK_PATH = "/tasks:images2PosedRGBD"
ATLAS_POSE_ENDPOINT = _API_ROOT + _POSE_TASK_PATH
ATLAS_API_BASE_URL_ENV = "WORLDLABS_ATLAS_API_BASE_URL"
ATLAS_API_KEY_ENV = "WORLDLABS_ATLAS_API_KEY"
ATLAS_PRICE_REVIEW_ENV = "BLUEPRINT_ATLAS_POSE_PRICE_REVIEW_JSON"
ATLAS_PROVIDER_RETENTION_ENV = "BLUEPRINT_ATLAS_PROVIDER_RETENTION"
ATLAS_WEBHOOK_URL_ENV = "BLUEPRINT_ATLAS_WEBHOOK_URL"
PRICE_REVIEW_SCHEMA = "website_atlas_pose_price_review.v1"
INPUTS_SCHEMA = "website_atlas_pose_inputs.v1"
SUBMISSION_SCHEMA = "website_atlas_pose_submission.v1"
RESULT_SCHEMA = "website_atlas_pose_result.v1"
RECEIPT_SCHEMA = "website_atlas_pose_operation_receipt.v1"
RESOURCE_CLASS = "provider_reconstruction_api"
#: Not Marble's ``world_labs``. Atlas has its own terms and billing (1,000
#: credits per dollar against Marble's 1,250), and a site's consent to the
#: Marble terms does not cover sending Atlas unedited source frames. The WebApp
#: refuses this identity until a sponsorship records Atlas's terms.
PROVIDER = "world_labs_atlas"
#: Where World Labs keeps uploaded frames and outputs, recorded by the owner
#: before the first upload: the account keeps neither, or the owner approved
#: keeping both in the project library.
PROVIDER_RETENTION_DECISIONS = ("account_does_not_retain", "project_library_retention_approved")
#: Pinned, and bound into the reviewed price, so a silent upstream default
#: change cannot change what is bought. Refinement is on because downstream
#: mask lifting compares views.
REQUEST_OPTIONS: Mapping[str, bool] = {"refinePoses": True, "base64Outputs": False}
#: The documented default grid in both orientations. Atlas scales a view to
#: cover the grid and then center-crops it.
TARGET_RESOLUTIONS: tuple[tuple[int, int], ...] = ((1280, 720), (720, 1280))
#: Below this share of a view kept after the crop, part of the site would be
#: hidden from Atlas. 16:9 phone video keeps over 99%; 4:3 would keep 75%.
MIN_VIEW_KEPT_FRACTION = 0.95
#: The beta rate card may change, so a price review goes stale.
PRICE_REVIEW_MAX_AGE_DAYS = 30
#: The WebApp's preparation-spend route refuses more than this per reservation.
MAX_REVIEWED_COST_USD = 1000.0
#: Configured input capacity of one images2PosedRGBD request: provider data, as
#: src/blueprint_pipeline/AGENTS.md requires. Atlas publishes no per-request limit,
#: so this is the owner's accuracy target of 2026-10-22, unverified until a real
#: request of this size succeeds; raise it when Atlas states or proves more.
MAX_POSE_INPUT_IMAGES = 64
API_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
#: A textured splats2Mesh GLB (4096 px atlas by default) routinely exceeds image outputs.
MAX_MESH_OUTPUT_BYTES = 1024 * 1024 * 1024
READY_DELAYS_SECONDS = (1.0, 2.0, 4.0, 8.0, 16.0)
#: The documented recovery for an uncertain or 402-refused submission is the
#: identical body under the same Idempotency-Key. It is bounded, and an
#: uncertain one is resent only soon after the first send, because the key's
#: lifetime is undocumented.
RESEND_LIMIT = 3
UNCERTAIN_RESEND_WINDOW_SECONDS = 3600
CREDIT_RESEND_INTERVAL_SECONDS = 300
# Terminal operation failures that a new submission may clear, by (google.rpc
# code, ErrorInfo reason). https://atlas-beta.worldlabs.ai/docs/errors (read
# 2026-10-24): THROTTLED means "submit a new task later with backoff" and an
# EXECUTION_TIMEOUT retry "is a new task submission"; code 4 alone can also mean
# an expired task. EXECUTION_FAILED says to investigate first, so it stays final.
RESUBMITTABLE_FAILURES = frozenset({(14, "TRANSIENT_UPSTREAM"), (8, "THROTTLED"), (4, "EXECUTION_TIMEOUT")})
#: World Labs' own execution failure may clear on a separate submission, but one that fails
#: again only spends: it gets a single resubmission (2026-10-30 capture-4b60431e atlasGenerate).
RESUBMITTABLE_ONCE_FAILURES = frozenset({(13, "EXECUTION_FAILED")})
# Each pose attempt is separately reserved; a failure above starts at most this many.
MAX_POSE_ATTEMPTS = 3
RESUBMIT_BACKOFF_SECONDS = 30
ADMISSION_MARGIN_SECONDS = 120
#: Operations are kept this long. Before then a 404 is not proof one is gone.
OPERATION_RETENTION_SECONDS = 30 * 86400
_UNCERTAIN_4XX = frozenset({408, 409, 429})
#: Reconciling an uncertain send from the operation list and each operation's
#: request trace (docs/operations, api-reference/get-operation-trace). Both stay
#: readable for the 30-day retention. A send is accepted within one request
#: timeout; the window margin also absorbs clock skew between us and Atlas.
RECONCILE_WINDOW_MARGIN_SECONDS = 900
#: "Nothing was bought" is concluded only this long after the last send, so an
#: operation accepted just before an answer was lost has reached the list.
RECONCILE_SETTLE_SECONDS = 900
#: The last day of retention is too close to call: an operation may expire mid-read.
RECONCILE_RETENTION_MARGIN_SECONDS = 86400
RECONCILE_PAGE_SIZE = 200
RECONCILE_MAX_PAGES = 50
#: A send proven to have bought nothing reopens for the documented recovery (the
#: identical bytes under the same key) at most this many times, then closes.
RECONCILE_REOPEN_LIMIT = 2
_EXR_MAGIC = b"\x76\x2f\x31\x01"
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_REVIEW_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_CONTENT_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
_OUTPUT_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".exr"}
_HELD_LOCKS = threading.local()

logger = logging.getLogger(__name__)

TransportResponse = tuple[int, bytes] | tuple[int, bytes, Mapping[str, str]]
Transport = Callable[[str, str, Mapping[str, str], "bytes | None"], TransportResponse]


class AtlasTransportError(RuntimeError):
    """No HTTP response arrived, so a mutation's outcome is unknown."""


class AtlasHttpError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        # Storage error bodies can echo signed query strings.
        super().__init__(f"website_atlas_api_{status}:{redacted_failure_text(detail)[:500]}")


class AtlasReconciliationPending(RuntimeError):
    """The operation list cannot show yet whether an uncertain send bought anything; read it again later."""


def request_options() -> dict[str, bool]:
    return dict(REQUEST_OPTIONS)


def _safe_transport(base_url: str) -> Transport:
    """Every Atlas call goes through the audited outbound boundary (FABLE-007).

    API calls are pinned to the API host and carry the key. Provider-issued
    upload and read URLs are pinned to their own host and carry no key.
    Redirects are refused either way.
    """
    from . import safe_outbound_http as boundary

    api_policy = boundary.pinned_api_policy(base_url, max_response_bytes=API_MAX_RESPONSE_BYTES)

    def send(method: str, url: str, headers: Mapping[str, str], data: bytes | None) -> TransportResponse:
        api = url.startswith(base_url + "/")
        policy = api_policy if api else boundary.presigned_transfer_policy(
            url, max_response_bytes=MAX_MESH_OUTPUT_BYTES)
        try:
            response = boundary.request(url, method=method, data=data, headers=headers,
                                        timeout_seconds=120 if api else 600, policy=policy)
        except urllib.error.HTTPError as exc:
            # Preserve only scheduling and rate-limit hints, never signed URLs or arbitrary headers.
            retry_after = exc.headers.get("Retry-After", "") if exc.headers else ""
            limits = {key.lower(): str(value)[:200] for key, value in (exc.headers.items() if exc.headers else ())
                      if key.lower().startswith(("ratelimit", "x-ratelimit"))}
            try:
                body = exc.read(4096)
            except (OSError, http.client.HTTPException, ValueError):
                body = b""  # The status is the answer; a lost error body changes nothing.
            finally:
                exc.close()
            return exc.code, body, {"retry-after": retry_after, **limits}
        except boundary.SafeOutboundHttpError:
            raise
        except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as exc:
            # Fixed text: http.client quotes header values, the API key among them.
            raise AtlasTransportError(f"website_atlas_transport_failed:{type(exc).__name__}") from None
        return response.status, response.body

    return send


def _https(url: Any) -> str:
    parts = urlsplit(str(url or ""))
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        raise ValueError("website_atlas_provider_url_invalid")
    return str(url)


def _safe_id(value: Any) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError("website_atlas_identifier_invalid")
    return value


def _json_object(status: int, payload: bytes) -> dict[str, Any]:
    if not 200 <= status < 300:
        raise AtlasHttpError(status, payload.decode("utf-8", "replace"))
    try:
        value = json.loads(payload or b"{}")
    except ValueError as exc:
        raise ValueError("website_atlas_api_response_invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("website_atlas_api_response_invalid")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def atlas_webhook_url() -> str | None:
    """The completion callback a new task request names, or None.

    World Labs takes https on 443 to a public DNS name, with no credentials or
    fragment (https://docs.atlas-beta.worldlabs.ai/webhooks). The callback only
    wakes the provider poll early, so an unusable value is flagged and left out.
    """
    raw = os.environ.get(ATLAS_WEBHOOK_URL_ENV, "").strip()
    if not raw:
        return None
    try:
        parts = urlsplit(raw)
        host = (parts.hostname or "").lower()
    except ValueError:
        # Malformed (e.g. an unclosed IPv6 bracket): the optional callback is
        # left out; the Atlas work itself still goes ahead.
        logger.warning("atlas_webhook_url_unusable; tasks are submitted without a callback")
        return None
    try:
        port = parts.port
    except ValueError:
        port = -1
    try:
        numeric = bool(ipaddress.ip_address(host))
    except ValueError:
        numeric = False
    if (parts.scheme != "https" or port not in (None, 443) or numeric or "." not in host
            or parts.username or parts.password or "#" in raw or any(c.isspace() for c in raw)):
        logger.warning("atlas_webhook_url_unusable; tasks are submitted without a callback")
        return None
    return raw


def new_request_body(request: Mapping[str, Any]) -> str:
    """The exact bytes of a task request that has never been sent.

    The configured callback rides only here, outside the request record, its
    digest and its idempotency key: a configuration change never re-identifies a
    purchase, and a request already on record is resent byte for byte.
    """
    url = atlas_webhook_url()
    return _canonical_json({**request, "webhookUrl": url} if url else dict(request))


def _without_urls(value: Any) -> Any:
    """Drop provider ``url`` fields: read URLs are short-lived credentials, asset IDs are the record."""
    if isinstance(value, Mapping):
        return {key: _without_urls(item) for key, item in value.items() if key != "url"}
    if isinstance(value, list):
        return [_without_urls(item) for item in value]
    return value


class AtlasClient:
    """The documented Atlas beta task API and nothing more. Tests inject ``transport``."""

    def __init__(self, *, base_url: str, api_key: str, transport: Transport | None = None,
                 source_checkpoint: Callable[[str], None] | None = None,
                 project_checkpoint: Callable[[AtlasClient], None] | None = None) -> None:
        self._base, self._key = base_url.rstrip("/"), api_key
        self._transport = transport or _safe_transport(self._base)
        self._source_checkpoint = source_checkpoint
        self._project_checkpoint = project_checkpoint
        self.retry_after: str | None = None
        self.rate_limit_headers: dict[str, str] = {}

    def _send_transport(self, method: str, url: str, headers: Mapping[str, str],
                        data: bytes | None) -> tuple[int, bytes]:
        # Reconciliation/output reads and teardown remain possible after a
        # source is revoked. Every upload and processing mutation reopens it.
        output_read = method == "POST" and url.startswith(self._base + "/assets/") and url.endswith(":createReadUrl")
        if method not in {"GET", "HEAD", "DELETE"} and not output_read:
            if self._project_checkpoint is not None:
                self._project_checkpoint(self)
                # A caller-supplied API header must not substitute a different
                # key after the project checkpoint queried this client's key.
                if url.startswith(self._base + "/") and any(
                    value != self._key for key, value in headers.items() if key.lower() == "wlt-api-key"
                ):
                    raise ValueError("owned_capture_atlas_project_key_changed")
            # Reopen source bytes/lifecycle after the metadata round trip, as
            # close to the actual upload or task mutation as possible.
            if self._source_checkpoint is not None:
                self._source_checkpoint("atlas_provider_disclosure")
        self.retry_after, self.rate_limit_headers = None, {}
        response = self._transport(method, url, headers, data)
        if len(response) == 3:
            self.retry_after = next((str(value) for key, value in response[2].items()
                                     if key.lower() == "retry-after"), None)
            self.rate_limit_headers = {key.lower(): str(value)[:200] for key, value in response[2].items()
                                       if key.lower().startswith(("ratelimit", "x-ratelimit"))}
        return response[0], response[1]

    @property
    def base_url(self) -> str:
        return self._base

    def send(self, method: str, path: str, body: Mapping[str, Any] | None = None, *,
             headers: Mapping[str, str] | None = None) -> tuple[int, bytes]:
        """One API request, returning the raw status and body."""
        data = None if body is None else _canonical_json(body).encode()
        return self.send_raw(method, path, data, headers=headers)

    def send_raw(self, method: str, path: str, data: bytes | None, *,
                 headers: Mapping[str, str] | None = None) -> tuple[int, bytes]:
        """Send exact bytes, so a resend under one Idempotency-Key is byte-identical."""
        sent = {"WLT-Api-Key": self._key, "Accept": "application/json", **(headers or {})}
        if data is not None:
            sent["Content-Type"] = "application/json"
        return self._send_transport(method, self._base + path, sent, data)

    def call(self, method: str, path: str, body: Mapping[str, Any] | None = None, *,
             headers: Mapping[str, str] | None = None) -> dict[str, Any]:
        return _json_object(*self.send(method, path, body, headers=headers))

    def send_bytes(self, url: Any, *, method: str, headers: Mapping[str, str], data: bytes) -> None:
        """Upload to a provider-issued URL. The API key never travels with it."""
        status, payload = self._send_transport(method, _https(url), dict(headers), data)
        if not 200 <= status < 300:
            raise AtlasHttpError(status, payload.decode("utf-8", "replace"))

    def fetch_bytes(self, url: Any, *, max_bytes: int | None = None) -> bytes:
        status, payload = self._send_transport("GET", _https(url), {}, None)
        if not 200 <= status < 300:
            raise AtlasHttpError(status, payload.decode("utf-8", "replace"))
        if len(payload) > (MAX_OUTPUT_BYTES if max_bytes is None else max_bytes):
            raise ValueError("website_atlas_output_too_large")
        return payload


def atlas_api_config() -> tuple[str, str]:
    """The documented ``$WLT_API_BASE_URL`` (ending ``/api/v2``) and project key.

    Neither is defaulted: the docs name no production host, and a key sent to
    a guessed host is a leaked key.
    """
    raw = (os.environ.get(ATLAS_API_BASE_URL_ENV) or os.environ.get("WLT_API_BASE_URL") or "").strip()
    if not raw:
        raise ValueError("website_atlas_api_base_url_missing")
    base = raw.rstrip("/")
    parts = urlsplit(base)
    host = (parts.hostname or "").lower()
    try:
        port = parts.port
    except ValueError:
        port = -1
    if (parts.scheme != "https" or parts.path != _API_ROOT or parts.query or parts.fragment
            or parts.username or parts.password or port not in (None, 443)
            or not (host == "worldlabs.ai" or host.endswith(".worldlabs.ai"))):
        raise ValueError("website_atlas_api_base_url_invalid")
    key = (os.environ.get(ATLAS_API_KEY_ENV) or os.environ.get("WLT_API_KEY") or "").strip()
    key_file = os.environ.get("WORLDLABS_ATLAS_API_KEY_FILE", "").strip()
    if not key and key_file:
        path = Path(key_file)
        if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
            raise ValueError("website_atlas_api_key_file_unsafe")
        key = path.read_text().strip()
    if not key:
        raise ValueError("website_atlas_api_key_missing")
    if any(character.isspace() or not character.isprintable() for character in key):
        # http.client would refuse it, quoting the key in its error.
        raise ValueError("website_atlas_api_key_invalid")
    return base, key


def atlas_provider_retention() -> str:
    """Uploading site frames needs the owner's recorded decision on provider retention."""
    value = os.environ.get(ATLAS_PROVIDER_RETENTION_ENV, "").strip()
    if value not in PROVIDER_RETENTION_DECISIONS:
        raise ValueError("website_atlas_provider_retention_decision_required")
    return value


def _kept_fraction(grid: Sequence[int], size: Sequence[int]) -> float:
    """Share of a view left after Atlas scales it to cover ``grid`` and center-crops it."""
    (width, height), (view_width, view_height) = grid, size
    scale = max(width / view_width, height / view_height)
    return (width * height) / (scale * view_width * scale * view_height)


def atlas_pose_target_resolution(pose_inputs: Mapping[str, Any]) -> list[int]:
    """The documented grid that keeps the most of every view, or a refusal."""
    from PIL import Image

    sizes = []
    # Square padding protects sources through the backend's native square
    # grid. Choose the final crop from original source aspects, so it removes
    # the padding rather than another part of the scene.
    for row in pose_inputs.get("full_view_sources", pose_inputs["frames"]):
        with Image.open(row["image_path"]) as image:
            sizes.append(image.size)
    best = max(TARGET_RESOLUTIONS, key=lambda grid: min(_kept_fraction(grid, size) for size in sizes))
    if min(_kept_fraction(best, size) for size in sizes) < MIN_VIEW_KEPT_FRACTION:
        raise ValueError("website_atlas_pose_view_crop_too_large")
    return list(best)


def _positive_finite(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _reviewed_recently(value: Any, today: date) -> bool:
    if not isinstance(value, str) or not _REVIEW_DATE.fullmatch(value):
        return False
    try:
        reviewed = date.fromisoformat(value)
    except ValueError:
        return False
    return today - timedelta(days=PRICE_REVIEW_MAX_AGE_DAYS) <= reviewed <= today


def atlas_pose_spend_line(*, planned_input_count: int, target_resolution: Sequence[int],
                          today: date | None = None) -> dict[str, Any]:
    """The reviewed ceiling for one operation, or a refusal.

    Atlas pricing is unpublished: the billing page (read 2026-10-19) points to
    the platform's rate card and says it may shift during the beta. So no
    ceiling exists until an owner records one read from that card, for these
    exact request options and grids and at most ``max_input_images`` frames.
    """
    raw = os.environ.get(ATLAS_PRICE_REVIEW_ENV, "").strip()
    if not raw:
        raise ValueError("website_atlas_reviewed_price_required")
    try:
        review = json.loads(raw)
    except ValueError as exc:
        raise ValueError("website_atlas_reviewed_price_invalid:json") from exc
    if not isinstance(review, dict):
        raise ValueError("website_atlas_reviewed_price_invalid:json")
    grids, options = review.get("target_resolutions"), review.get("request_options")
    today = today or datetime.now(timezone.utc).date()
    checks = (
        ("schema_version", review.get("schema_version") == PRICE_REVIEW_SCHEMA),
        ("endpoint", review.get("endpoint") == ATLAS_POSE_ENDPOINT),
        ("maximum_cost_usd", _positive_finite(review.get("maximum_cost_usd"))
         and review["maximum_cost_usd"] <= MAX_REVIEWED_COST_USD),
        ("max_input_images", type(review.get("max_input_images")) is int and review["max_input_images"] >= 2),
        ("target_resolutions", isinstance(grids, list) and bool(grids) and all(
            isinstance(grid, list) and all(type(side) is int for side in grid)
            and tuple(grid) in TARGET_RESOLUTIONS for grid in grids)),
        ("request_options", isinstance(options, dict) and all(type(item) is bool for item in options.values())),
        ("rate_card_reference", isinstance(review.get("rate_card_reference"), str)
         and bool(review["rate_card_reference"].strip())),
        ("reviewed_by", isinstance(review.get("reviewed_by"), str) and bool(review["reviewed_by"].strip())),
        ("reviewed_on", _reviewed_recently(review.get("reviewed_on"), today)),
    )
    for field, valid in checks:
        if not valid:
            raise ValueError(f"website_atlas_reviewed_price_invalid:{field}")
    if options != request_options() or list(target_resolution) not in grids:
        raise ValueError("website_atlas_reviewed_price_options_mismatch")
    if planned_input_count > review["max_input_images"]:
        raise ValueError("website_atlas_reviewed_price_input_count_exceeded")
    price = float(review["maximum_cost_usd"])
    # 1 and 1.0 are one price; the binding must not depend on how it was typed.
    return {"maximum_cost_usd": price, "review_digest": canonical_digest({**review, "maximum_cost_usd": price})}


def load_pending_atlas_pose_inputs(output_root: Path) -> dict[str, Any]:
    """The CPU handoff's pending record, with every frame's bytes re-verified."""
    path = output_root / "atlas_pose_inputs.json"
    if not path.is_file():
        raise ValueError("website_atlas_pose_inputs_missing")
    value = json.loads(path.read_text())
    frames = value.get("frames")
    if (value.get("schema_version") != INPUTS_SCHEMA or value.get("status") != "pending"
            or value.get("digest") != canonical_digest(value, digest_field="digest")
            or value.get("endpoint") != ATLAS_POSE_ENDPOINT or value.get("heldout_pixels_included") is not False
            or value.get("claim_ceiling") != "development_only"
            or value.get("metric_measurement_proven") is not False
            or not isinstance(frames, list) or not frames
            or len({row.get("frame_id") for row in frames}) != len(frames)):
        raise ValueError("website_atlas_pose_inputs_invalid")
    for row in frames:
        image = Path(row["image_path"])
        if image.suffix.lower() not in _CONTENT_TYPES:
            raise ValueError("website_atlas_pose_input_format_unsupported")
        if image.is_symlink() or not image.resolve().is_relative_to(output_root.resolve()):
            raise ValueError("website_atlas_pose_input_outside_capture")
        if not image.is_file() or _sha256_file(image) != row.get("image_digest"):
            raise ValueError("website_atlas_pose_input_changed")
    if value.get("full_view_sources") is not None:
        from .atlas_context_fov import full_view_sources
        full_view_sources(value, output_root)
    return value


def _frame_identities(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"frame_id": row.get("frame_id"), "image_digest": row.get("image_digest")} for row in frames]


def atlas_pose_request_binding(*, pose_inputs: Mapping[str, Any], spend: Mapping[str, Any],
                               task_context: Mapping[str, Any], target_resolution: Sequence[int],
                               provider_retention: str, attempt: int) -> dict[str, Any]:
    """Everything one reservation pays for: these frames, options, price and retention, on this attempt."""
    return {"endpoint": ATLAS_POSE_ENDPOINT, "pose_inputs_digest": pose_inputs["digest"],
            "source_binding": dict(pose_inputs["binding"]), "frames": _frame_identities(pose_inputs["frames"]),
            "request_options": request_options(), "target_resolution": list(target_resolution),
            "maximum_cost_usd": spend["maximum_cost_usd"], "price_review_digest": spend["review_digest"],
            "provider": PROVIDER, "provider_retention": provider_retention,
            "task_context_digest": task_context["context_digest"], "attempt": attempt}


def atlas_pose_submission_state(output_root: Path) -> dict[str, Any] | None:
    path = output_root / "atlas_pose" / "submission.json"
    return json.loads(path.read_text()) if path.is_file() else None


def needs_atlas_pose_reconciliation(state: Mapping[str, Any] | None, now: float) -> bool:
    """A send that may have bought an operation, past the documented resend recovery."""
    if not state:
        return False
    status = state.get("status")
    return (status in {"submitting", "rejected"} and bool(state.get("possibly_bought") or status == "submitting")
            and (state.get("resends", 0) >= RESEND_LIMIT or not state.get("possibly_bought")
                 or now - state.get("uncertain_since_epoch", 0) > UNCERTAIN_RESEND_WINDOW_SECONDS))


def refuse_closed_or_uncertain_attempt(state: Mapping[str, Any] | None, now: float) -> None:
    """Closed attempts, and uncertain ones past the documented resend, never buy another operation."""
    if not state:
        return
    status, resends = state.get("status"), state.get("resends", 0)
    if needs_atlas_pose_reconciliation(state, now):
        # A send may have bought an operation and its Idempotency-Key may have
        # lapsed: only reconcile_atlas_pose_submission can move it on.
        raise ValueError("website_atlas_pose_submission_requires_reconciliation")
    if status in {"submitting", "rejected"} and (state.get("possibly_bought") or status == "submitting"):
        return
    if status == "rejected":
        if not (state.get("provider_rejection") or {}).get("retryable") or resends >= RESEND_LIMIT:
            raise ValueError("website_atlas_pose_attempt_rejected")
        if now - state.get("last_attempt_at_epoch", 0) < CREDIT_RESEND_INTERVAL_SECONDS:
            raise ValueError("website_atlas_pose_resend_not_due")
    if status == "failed":
        raise ValueError("website_atlas_pose_attempt_failed")


def resubmittable_terminal_failure(state: Mapping[str, Any] | None, *, prior_resubmissions: int = 0) -> bool:
    """A finished operation whose documented failure a new, separate submission may clear."""
    failures = RESUBMITTABLE_FAILURES | (RESUBMITTABLE_ONCE_FAILURES if prior_resubmissions == 0 else frozenset())
    operation = (state or {}).get("terminal_operation") or {}
    error = operation.get("error") or {}
    return bool(state) and (state.get("status") == "failed" and operation.get("done") is True
            and operation.get("id") == state.get("operation_id")
            and any((error.get("code"), row.get("reason")) in failures
                    and row.get("domain") == "api.worldlabs.ai"
                    for row in error.get("details", []) if isinstance(row, dict)))


def _refuse_state_conflict(state: Mapping[str, Any] | None, binding: Mapping[str, Any]) -> None:
    if state is not None and (state.get("request_digest") != canonical_digest(binding)
                              or state.get("binding") != binding):
        raise ValueError("website_atlas_pose_already_bound_to_other_inputs")


def request_matches_trace(sent: Any, recorded: Any) -> bool:
    """Every field we sent, unchanged, inside the provider's recorded request.

    The trace projects a request through the task's current public schema, so
    it may carry defaults we never sent; numbers allow float wire rounding.
    """
    if isinstance(sent, Mapping):
        return isinstance(recorded, Mapping) and all(
            key in recorded and request_matches_trace(value, recorded[key]) for key, value in sent.items())
    if isinstance(sent, list):
        return (isinstance(recorded, list) and len(sent) == len(recorded)
                and all(request_matches_trace(a, b) for a, b in zip(sent, recorded)))
    if isinstance(sent, bool) or isinstance(recorded, bool):
        return sent is recorded
    if isinstance(sent, (int, float)) and isinstance(recorded, (int, float)):
        return (math.isfinite(sent) and math.isfinite(recorded)
                and math.isclose(sent, recorded, rel_tol=1e-6, abs_tol=1e-9))
    return sent == recorded


def _asset_ids(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "assetId" and isinstance(item, str):
                found.append(item)
            else:
                found.extend(_asset_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_asset_ids(item))
    return list(dict.fromkeys(found))


def _listed_operations(client: AtlasClient, *, created_after: int, created_before: int) -> tuple[list[str], bool]:
    """Every operation the key's project accepted in the window, and whether the listing ended."""
    ids: list[str] = []
    token = None
    for _ in range(RECONCILE_MAX_PAGES):
        query = {"createdAfter": created_after, "createdBefore": created_before, "pageSize": RECONCILE_PAGE_SIZE}
        page = client.call("GET", "/operations?" + urlencode({**query, **({"pageToken": token} if token else {})}))
        rows, token = page.get("operations"), page.get("nextPageToken")
        if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
            raise ValueError("website_atlas_operation_list_invalid")
        ids.extend(_safe_id(row.get("id")) for row in rows)
        if not token:
            return list(dict.fromkeys(ids)), True
        if not isinstance(token, str):
            raise ValueError("website_atlas_operation_list_invalid")
    return list(dict.fromkeys(ids)), False


def _project_asset(client: AtlasClient, asset_ids: Sequence[str]) -> str | None:
    """An asset of the request still readable by this key: the list is the same project's."""
    for asset_id in asset_ids:
        status, payload = client.send("GET", f"/assets/{_safe_id(asset_id)}")
        if status in {403, 404, 410}:
            continue
        asset = _json_object(status, payload)
        if asset.get("id") == asset_id and asset.get("state") != "DELETED":
            return asset_id
    return None


def reconcile_atlas_submission(client: AtlasClient, *, task_name: str, request: Mapping[str, Any],
                               owned_assets: Sequence[str], sent_at: Sequence[float], now: float) -> dict[str, Any]:
    """What the operation list says an uncertain submission bought.

    Either answer needs a complete listing of the window in which no trace is
    unreadable or carries an asset we uploaded without our request: an
    operation left unresolved could be another purchase of ours, and its
    outputs would then escape cleanup. ``bought`` names the operation whose
    recorded request carries every field we sent (the oldest, with any
    duplicates listed). ``not_bought`` is also read after the settle delay, by
    a key that still reads an asset of the request. ``lapsed`` means the
    retention has run out. Anything less raises AtlasReconciliationPending.
    """
    first, last = min(sent_at), max(sent_at)
    record: dict[str, Any] = {"at_epoch": now, "task": task_name}
    if now - first > OPERATION_RETENTION_SECONDS - RECONCILE_RETENTION_MARGIN_SECONDS:
        return {**record, "outcome": "lapsed"}
    window = [int(first) - RECONCILE_WINDOW_MARGIN_SECONDS, int(last) + RECONCILE_WINDOW_MARGIN_SECONDS + 1]
    listed, complete = _listed_operations(client, created_after=window[0], created_before=window[1])
    owned, matches, unresolved = set(owned_assets), [], []
    for operation_id in listed:
        status, payload = client.send("GET", f"/operations/{operation_id}:trace")
        if status in {404, 410}:
            unresolved.append(operation_id)
            continue
        trace = _json_object(status, payload)
        recorded = trace.get("request")
        if trace.get("operationId") != operation_id or not isinstance(recorded, Mapping):
            unresolved.append(operation_id)
        elif trace.get("taskName") == task_name and request_matches_trace(request, recorded):
            matches.append(operation_id)
        elif owned & set(_asset_ids(recorded)):
            # It carries a frame only this attempt uploaded, yet not our request:
            # never conclude that it is someone else's.
            unresolved.append(operation_id)
    if not complete or unresolved:
        raise AtlasReconciliationPending("website_atlas_reconciliation_pending:listing_unresolved")
    record.update(window=window, listed=len(listed))
    if matches:
        # Newest first, as listed: the oldest match is the original purchase.
        return {**record, "outcome": "bought", "operation_id": matches[-1],
                "duplicate_operation_ids": matches[:-1]}
    if now - last < RECONCILE_SETTLE_SECONDS:
        raise AtlasReconciliationPending("website_atlas_reconciliation_pending:settling")
    proof = _project_asset(client, _asset_ids(request))
    if proof is None:
        raise AtlasReconciliationPending("website_atlas_reconciliation_pending:project_unproven")
    return {**record, "outcome": "not_bought", "project_asset": proof}


def reconcile_atlas_pose_submission(output_root: Path, *, client: AtlasClient, now: float) -> dict[str, Any]:
    """Move an uncertain attempt on from the operation list; the caller holds the attempt lock.

    A bought operation is adopted and collected like any other. A send proven
    to have bought nothing reopens for the identical resend under the same key,
    which can buy at most that one operation, and closes after
    RECONCILE_REOPEN_LIMIT reopenings. Past retention the attempt fails, and a
    closed attempt can be archived.
    """
    path = output_root / "atlas_pose" / "submission.json"
    state = atlas_pose_submission_state(output_root)
    if not needs_atlas_pose_reconciliation(state, now):
        return state
    if not state.get("sends") or not isinstance(state.get("request"), Mapping):
        raise ValueError("website_atlas_pose_submission_requires_reconciliation")
    found = reconcile_atlas_submission(client, task_name=_POSE_TASK_PATH.split(":", 1)[1],
                                       request=state["request"], owned_assets=state.get("created_assets", []),
                                       sent_at=[row["at_epoch"] for row in state["sends"]], now=now)
    state.setdefault("reconciliations", []).append(found)
    if found["outcome"] == "bought":
        state.update(status="submitted", operation_id=found["operation_id"], api_base=client.base_url)
    elif found["outcome"] == "lapsed":
        state.update(status="failed", terminal_error={"reason": "reconciliation_window_lapsed"})
    else:
        reopen = state.get("reconciliation_reopens", 0) < RECONCILE_REOPEN_LIMIT
        state.update(status="rejected", possibly_bought=False, resends=0 if reopen else state.get("resends", 0),
                     provider_rejection={"status": None, "retryable": reopen, "reconciled_unbought": True},
                     reconciliation_reopens=state.get("reconciliation_reopens", 0) + reopen)
        state.pop("uncertain_since_epoch", None)
    _write_state(path, state)
    return state


def _attempt_number(output_root: Path) -> int:
    root = output_root / "atlas_pose_attempts"
    return sum(1 for path in root.iterdir() if path.is_dir()) if root.is_dir() else 0


def _write_durable(path: Path, data: bytes) -> None:
    """Publish ``data``, then fsync the file and its directory before any next step.

    Stricter than ``core.common.write_text``, whose directory fsync is best
    effort: intent written here is what stops a second paid submission.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_state(path: Path, value: Mapping[str, Any]) -> None:
    _write_durable(path, json.dumps(value, sort_keys=True, indent=2).encode())


@contextmanager
def _locked(output_root: Path) -> Iterator[None]:
    """One process at a time per capture; re-entrant within a thread."""
    path = output_root / "atlas_pose.lock"
    held = getattr(_HELD_LOCKS, "paths", None)
    if held is None:
        held = _HELD_LOCKS.paths = set()
    key = str(path.resolve())
    if key in held:
        yield
        return
    output_root.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("website_atlas_pose_operation_in_progress") from exc
        held.add(key)
        try:
            yield
        finally:
            held.discard(key)


def archive_closed_atlas_pose_attempt(output_root: Path) -> int:
    """Set an attempt aside so a new, separately reserved attempt can start.

    Only an attempt that provably bought nothing more qualifies: a failed or
    long-gone operation, a definitive refusal, uploads that never reached the
    task POST, or a reservation with no submission record. Any send that may
    have bought an operation keeps the attempt, since the next one would buy
    another under a new key. Returns the new attempt's number.
    """
    with _locked(output_root):
        root = output_root / "atlas_pose"
        state = atlas_pose_submission_state(output_root)
        status = (state or {}).get("status")
        closed = (status == "failed"
                  or (state is not None and not state.get("possibly_bought") and status in {"uploading", "rejected"})
                  or (state is None and root.is_dir()))
        if not closed:
            raise ValueError("website_atlas_pose_attempt_not_closed")
        number = _attempt_number(output_root)
        target = output_root / "atlas_pose_attempts" / f"{number:03d}"
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(root, target)
        result = output_root / "atlas_posed_rgbd_result.json"
        if result.is_file():
            os.replace(result, target / result.name)
    return number + 1


def _require_reported_size(asset: Mapping[str, Any], size: int, *, required: bool) -> None:
    """A bounded HTTP read can end early without an error; the provider's size is the check."""
    reported = asset.get("sizeBytes")
    if reported is None:
        if required:
            raise ValueError("website_atlas_output_size_unreported")
        return
    if isinstance(reported, bool) or not str(reported).isdigit() or int(reported) != size:
        raise ValueError("website_atlas_transfer_size_mismatch")


def _upload_frame(client: AtlasClient, frame: Mapping[str, Any], *, name: str,
                  state: dict[str, Any], path: Path, content_type: str | None = None) -> dict[str, Any]:
    """Single-object upload: create, upload URL, bytes, complete."""
    image = Path(frame["image_path"])
    data = image.read_bytes()
    # Hash the bytes actually sent, not the file as it was when admitted.
    if _sha256_bytes(data) != frame["image_digest"]:
        raise ValueError("website_atlas_pose_input_changed")
    content_type = content_type or _CONTENT_TYPES[image.suffix.lower()]
    asset_id = _safe_id(client.call("POST", "/assets", {"asset": {"displayName": name}}).get("id"))
    # Every asset that may come to hold site pixels is on record before the
    # bytes go, so a failed upload never leaves an untracked copy.
    state["created_assets"].append(asset_id)
    _write_state(path, state)
    grant = client.call("POST", f"/assets/{asset_id}:createUploadUrl",
                        {"contentType": content_type, "contentLength": len(data)})
    method, headers = str(grant.get("method") or "").upper(), grant.get("headers") or {}
    if (method not in {"PUT", "POST"} or not isinstance(headers, Mapping)
            or not all(isinstance(key, str) and isinstance(item, str) for key, item in headers.items())
            or any(key.lower() == "wlt-api-key" for key in headers)):
        raise ValueError("website_atlas_upload_grant_invalid")
    client.send_bytes(grant.get("uploadUrl"), method=method,
                      headers={"Content-Type": content_type, **headers}, data=data)
    asset = client.call("POST", f"/assets/{asset_id}:completeUpload", {})
    if asset.get("id") != asset_id:
        raise ValueError("website_atlas_asset_identity_mismatch")
    _require_reported_size(asset, len(data), required=False)
    return {"frame_id": frame["frame_id"], "image_digest": frame["image_digest"], "asset_id": asset_id,
            "content_type": content_type, "bytes": len(data), "ready": asset.get("state") == "READY"}


def _await_ready(client: AtlasClient, state: dict[str, Any], path: Path, sleep: Callable[[float], None]) -> None:
    """Re-check recorded assets together until READY; a frame is never uploaded twice."""
    pending = [row for row in state["uploads"] if not row["ready"]]
    for delay in (0.0, *READY_DELAYS_SECONDS):
        if not pending:
            return
        if delay:
            sleep(delay)
        for row in list(pending):
            asset = client.call("GET", f"/assets/{row['asset_id']}")
            if asset.get("id") != row["asset_id"] or asset.get("state") == "DELETED":
                raise ValueError("website_atlas_asset_not_ready")
            if asset.get("state") == "READY":
                _require_reported_size(asset, row["bytes"], required=False)
                row["ready"] = True
                pending.remove(row)
        _write_state(path, state)
    if pending:
        raise ValueError("website_atlas_asset_not_ready")


def _require_admission(admission: Mapping[str, Any], *, request_digest: str, binding: Mapping[str, Any],
                       now: float) -> None:
    if (not isinstance(admission, Mapping) or admission.get("allocation_binding_digest") != request_digest
            or admission.get("provider") != PROVIDER or admission.get("resource_class") != RESOURCE_CLASS):
        raise ValueError("website_atlas_pose_spend_binding_missing")
    if admission.get("external_disclosure_allowed") is not True:
        # Uploading source frames discloses site pixels to World Labs.
        raise ValueError("website_atlas_pose_disclosure_not_authorized")
    budget = admission.get("maximum_cost_usd")
    if not _positive_finite(budget) or budget < binding["maximum_cost_usd"]:
        raise ValueError("website_atlas_pose_budget_insufficient")
    expires = admission.get("expires_at_epoch")
    if not _positive_finite(expires) or expires <= now + ADMISSION_MARGIN_SECONDS:
        raise ValueError("website_atlas_pose_admission_expired")


def submit_atlas_pose_operation(*, pose_inputs: Mapping[str, Any], output_root: Path, client: AtlasClient,
                                binding: Mapping[str, Any], admission: Mapping[str, Any],
                                grant: PaidResourceAdmissionGrant | None, sleep: Callable[[float], None],
                                now: Callable[[], float]) -> dict[str, Any]:
    """Upload, then submit once. Every provider mutation follows durable intent."""
    request_digest = canonical_digest(binding)
    _require_admission(admission, request_digest=request_digest, binding=binding, now=now())
    require_paid_resource_admission_grant(grant, resource_class=RESOURCE_CLASS,
                                          allocation_binding_digest=request_digest,
                                          require_allocation_binding=True)
    if (pose_inputs.get("digest") != binding["pose_inputs_digest"]
            or _frame_identities(pose_inputs["frames"]) != binding["frames"]):
        raise ValueError("website_atlas_pose_inputs_not_bound")
    path = output_root / "atlas_pose" / "submission.json"
    with _locked(output_root):
        state = atlas_pose_submission_state(output_root)
        _refuse_state_conflict(state, binding)
        if state is None:
            state = {"schema_version": SUBMISSION_SCHEMA, "request_digest": request_digest,
                     "binding": dict(binding), "status": "uploading",
                     "idempotency_key": "bp-atlas-" + request_digest[7:39], "created_assets": [], "uploads": []}
            _write_state(path, state)
        if state["status"] in {"submitted", "completed"}:
            return state
        refuse_closed_or_uncertain_attempt(state, now())
        if state["status"] == "uploading":
            uploads = state["uploads"]
            if _frame_identities(uploads) != binding["frames"][:len(uploads)]:
                raise ValueError("website_atlas_pose_upload_binding_mismatch")
            for index in range(len(uploads), len(binding["frames"])):
                # Names identify the attempt, never the site or the capture.
                uploads.append(_upload_frame(client, pose_inputs["frames"][index],
                                             name=f"bp-{request_digest[7:19]}-{index:04d}", state=state, path=path))
                _write_state(path, state)
            _await_ready(client, state, path, sleep)
            request = {"frames": [{"imageAsset": {"assetId": row["asset_id"]}} for row in uploads],
                       "targetResolution": list(binding["target_resolution"]), **request_options()}
            # The exact bytes are kept, so every resend under the key is identical.
            state.update(status="submitting", resends=0, sends=[], possibly_bought=False, request=request,
                         request_body=new_request_body(request), submitting_at_epoch=now())
        else:
            # The documented recovery: the identical bytes under the same key.
            state.update(status="submitting", resends=state["resends"] + 1)
        # Uploads can be slow; the sponsorship must still cover the purchase.
        _require_admission(admission, request_digest=request_digest, binding=binding, now=now())
        sent_at, earlier_uncertain = now(), bool(state["possibly_bought"])
        state["last_attempt_at_epoch"] = sent_at
        state["sends"].append({"at_epoch": sent_at, "status": None})
        # Until an answer proves otherwise, this send may buy an operation.
        state["possibly_bought"] = True
        state.setdefault("uncertain_since_epoch", sent_at)
        _write_state(path, state)
        status, payload = client.send_raw("POST", _POSE_TASK_PATH, state["request_body"].encode(),
                                          headers={"Idempotency-Key": state["idempotency_key"]})
        state["sends"][-1].update(status=status,
                                  body=redacted_failure_text(payload.decode("utf-8", "replace"))[:65536])
        if 200 <= status < 300:
            # The paid answer is kept before it is judged.
            _write_state(path, state)
            operation = _without_urls(_json_object(status, payload))
            if not isinstance(operation.get("id"), str) or not _SAFE_ID.fullmatch(operation["id"]):
                raise ValueError("website_atlas_pose_operation_id_missing")
            state.update(status="submitted", operation_id=operation["id"], operation=operation,
                         api_base=client.base_url)
        elif 400 <= status < 500 and status not in _UNCERTAIN_4XX and not earlier_uncertain:
            # A definitive refusal of the only send so far bought nothing. A
            # 402 is resent later, as the billing docs describe; any other
            # refusal closes the attempt.
            state.update(status="rejected", provider_rejection={"status": status, "retryable": status == 402},
                         possibly_bought=False)
            state.pop("uncertain_since_epoch")
        _write_state(path, state)
        if state["status"] != "submitted":
            raise AtlasHttpError(status, payload.decode("utf-8", "replace"))
    return state


def _output_references(frames: Sequence[Mapping[str, Any]]) -> list[list[tuple[str, str]]]:
    """Every returned asset reference, checked before the first download."""
    rows = []
    for frame in frames:
        depth = frame["depth"]
        pairs = (("image", frame.get("imageAsset")), ("depth", depth.get("depthAsset")),
                 ("confidence", depth.get("confidenceAsset")), ("mask", frame.get("maskAsset")))
        rows.append([(role, _safe_id(reference.get("assetId") if isinstance(reference, Mapping) else None))
                     for role, reference in pairs if reference is not None])
    return rows


def _retain_outputs(client: AtlasClient, references: Sequence[Sequence[tuple[str, str]]],
                    output_root: Path) -> list[list[dict[str, Any]]]:
    """Copy every returned file locally, in submission order.

    Output URLs expire within a day, so the retained, fsynced bytes and their
    digests are the record. Fresh read URLs come from ``:createReadUrl`` and
    carry no key.
    """
    retained = []
    for index, pairs in enumerate(references):
        outputs = []
        for role, asset_id in pairs:
            url = client.call("POST", f"/assets/{asset_id}:createReadUrl").get("readUrl")
            data = client.fetch_bytes(url)
            asset = client.call("GET", f"/assets/{asset_id}")
            if asset.get("id") != asset_id:
                raise ValueError("website_atlas_asset_identity_mismatch")
            _require_reported_size(asset, len(data), required=True)
            if role == "depth" and not data.startswith(_EXR_MAGIC):
                raise ValueError("website_atlas_depth_not_exr")
            suffix = PurePosixPath(urlsplit(str(url)).path).suffix.lower()
            relative = PurePosixPath("atlas_pose", "outputs", f"{index:06d}",
                                     role + (suffix if suffix in _OUTPUT_SUFFIXES else ".bin"))
            _write_durable(output_root / relative, data)
            outputs.append({"role": role, "asset_id": asset_id, "path": relative.as_posix(),
                            "sha256": _sha256_bytes(data), "bytes": len(data)})
        retained.append(outputs)
    return retained


def collect_atlas_pose_operation(*, output_root: Path, client: AtlasClient, sleep: Callable[[float], None],
                                 now: Callable[[], float], max_polls: int,
                                 poll_interval_seconds: float) -> dict[str, Any]:
    """Poll the saved operation; on success retain outputs and write the result."""
    from .website_worldlabs import _atlas_posed_context

    path, result_path = output_root / "atlas_pose" / "submission.json", output_root / "atlas_posed_rgbd_result.json"
    with _locked(output_root):
        state = atlas_pose_submission_state(output_root)
        if not state or state.get("status") not in {"submitted", "completed"} or not state.get("operation_id"):
            raise ValueError("website_atlas_pose_operation_not_submitted")
        if state["status"] == "completed":
            result = json.loads(result_path.read_text()) if result_path.is_file() else {}
            if (result.get("digest") != state.get("result_digest")
                    or result.get("digest") != canonical_digest(result, digest_field="digest")):
                raise ValueError("website_atlas_pose_result_changed")
            return result
        if state.get("api_base") != client.base_url:
            # Another host or project cannot see this operation; a 404 there
            # would wrongly close an attempt that is still running and billed.
            raise ValueError("website_atlas_pose_api_base_changed")
        operation_id = state["operation_id"]
        for attempt in range(max_polls):
            status, payload = client.send("GET", f"/operations/{operation_id}")
            if status in {404, 410}:
                if now() - state["uncertain_since_epoch"] < OPERATION_RETENTION_SECONDS:
                    raise ValueError("website_atlas_pose_operation_not_found")
                # Past the documented retention, a missing operation is gone.
                state.update(status="failed", terminal_error={"status": status})
                _write_state(path, state)
                raise ValueError("website_atlas_pose_operation_gone")
            operation = _without_urls(_json_object(status, payload))
            if operation.get("id") != operation_id:
                raise ValueError("website_atlas_pose_operation_identity_mismatch")
            if operation.get("done") is True:
                break
            if attempt + 1 < max_polls:
                sleep(poll_interval_seconds)
        else:
            return {"status": "processing", "operation_id": operation_id,
                    "blockers": ["website_atlas_pose_operation_pending"]}
        if operation.get("error") is not None:
            state.update(status="failed", terminal_operation=operation)
            _write_state(path, state)
            raise ValueError("website_atlas_pose_operation_failed")
        try:
            canonical_digest(operation)
        except UnicodeError as exc:
            raise ValueError("website_atlas_pose_operation_unencodable") from exc
        _atlas_posed_context(operation=operation, expected_frame_count=len(state["uploads"]))
        frames, grid = operation["response"]["frames"], tuple(state["request"]["targetResolution"])
        if any((frame["camera"]["intrinsics"]["width"], frame["camera"]["intrinsics"]["height"]) != grid
               for frame in frames):
            raise ValueError("website_atlas_pose_grid_not_requested")
        retained = _retain_outputs(client, _output_references(frames), output_root)
        receipt = {"schema_version": RECEIPT_SCHEMA, "operation_id": operation_id,
                   "request_digest": state["request_digest"], "idempotency_key": state["idempotency_key"],
                   "binding": state["binding"], "request": state["request"],
                   # What was actually submitted, frame by frame, with the
                   # files each frame produced. Output count alone is not usage.
                   "input_usage": {"submitted_frame_count": len(state["uploads"]), "frames": [
                       {**{key: row[key] for key in ("frame_id", "image_digest", "asset_id", "content_type", "bytes")},
                        "outputs": outputs} for row, outputs in zip(state["uploads"], retained, strict=True)]},
                   "billing": {"status": "unreported",
                               "reason": "Atlas operations carry no usage or cost fields; the reservation "
                                         "stays fully charged until a reviewed settlement exists."}}
        result = {"schema_version": RESULT_SCHEMA, "endpoint": ATLAS_POSE_ENDPOINT,
                  "binding": dict(state["binding"]["source_binding"]), "operation": operation,
                  "operation_receipt": receipt, "claim_ceiling": "development_only",
                  "metric_measurement_proven": False, "physical_evidence": False,
                  "depth_unit": "provider_world_units", "scale_status": "unscaled_provider_world_units"}
        result["digest"] = canonical_digest(result, digest_field="digest")
        _write_state(result_path, result)
        state.update(status="completed", result_digest=result["digest"])
        _write_state(path, state)
    return result


def run_admitted_atlas_pose(*, output_root: Path, capture_id: str, task_context: Mapping[str, Any],
                            release_blockers: Callable[[], tuple[str, Sequence[str]]],
                            transport: Transport | None = None, sleep: Callable[[float], None] = time.sleep,
                            now: Callable[[], float] = time.time, max_polls: int = 40,
                            poll_interval_seconds: float = 15.0) -> dict[str, Any]:
    """The admitted operation behind ``paid_resource_allocator.run_sponsored_website_atlas_pose``.

    Uploads and the submission need, in this order: an admitted release, a
    retention decision, a reviewed price, then the WebApp-signed spend grant.
    The state check, the reservation and the submission share one lock.
    Polling and retaining an already-submitted operation are reads and need
    none of them. A documented resubmittable failure closes its attempt; a
    new, separately reserved attempt follows, at most ``MAX_POSE_ATTEMPTS``.
    """
    from .website_task_context import reserve_website_preparation_spend
    from .atlas_preparation_runtime import preparation_source_checkpoint
    from .atlas_preparation_context import validate_preparation_task_context
    from .owned_capture_atlas_project import owned_atlas_project_checkpoint

    if task_context.get("capture_id") != capture_id:
        raise ValueError("website_atlas_pose_task_capture_mismatch")
    validate_preparation_task_context(task_context)
    base_url, api_key = atlas_api_config()
    client = AtlasClient(base_url=base_url, api_key=api_key, transport=transport,
                         source_checkpoint=preparation_source_checkpoint(task_context),
                         project_checkpoint=owned_atlas_project_checkpoint(task_context))

    def resubmit_due() -> bool:
        # The archive itself refuses an attempt that could have bought more than its failed operation.
        return (resubmittable_terminal_failure(atlas_pose_submission_state(output_root),
                                               prior_resubmissions=_attempt_number(output_root))
                and _attempt_number(output_root) + 1 < MAX_POSE_ATTEMPTS)
    try:
        while True:
            # A closed attempt's hold is released before the next one reserves.
            settle_closed_atlas_pose_attempts(output_root, task_context=task_context)
            if resubmit_due():
                sleep(RESUBMIT_BACKOFF_SECONDS)
                archive_closed_atlas_pose_attempt(output_root)
            with _locked(output_root):
                state = atlas_pose_submission_state(output_root)
                if needs_atlas_pose_reconciliation(state, now()):
                    state = reconcile_atlas_pose_submission(output_root, client=client, now=now())
                refuse_closed_or_uncertain_attempt(state, now())
                if not (state and state.get("operation_id")):
                    pose_inputs = load_pending_atlas_pose_inputs(output_root)
                    retention = atlas_provider_retention()
                    commit, blockers = release_blockers()
                    if blockers:
                        raise ValueError("website_atlas_pose_release_not_admitted:" + ",".join(blockers))
                    grid = atlas_pose_target_resolution(pose_inputs)
                    spend = atlas_pose_spend_line(planned_input_count=len(pose_inputs["frames"]), target_resolution=grid)
                    binding = atlas_pose_request_binding(
                        pose_inputs=pose_inputs, spend=spend, task_context=task_context, target_resolution=grid,
                        provider_retention=retention, attempt=_attempt_number(output_root))
                    # Refused before reserving: the adapter would refuse it anyway,
                    # after the sponsorship had been charged for it.
                    _refuse_state_conflict(state, binding)
                    retained_path = output_root / "atlas_pose" / "controller_admission.json"
                    retained = json.loads(retained_path.read_text()) if retained_path.is_file() else None
                    admission, grant = reserve_website_preparation_spend(
                        task_context=task_context, binding_digest=canonical_digest(binding),
                        maximum_cost_usd=spend["maximum_cost_usd"], request_count=1,
                        resource_class=RESOURCE_CLASS, provider=PROVIDER, retained_admission=retained,
                        source_video_digest=pose_inputs["binding"]["source_video_digest"],
                        retain_admission=lambda value: _write_state(retained_path, {**value, "source_commit": commit}))
                    if admission.get("provider_retention", retention) != retention:
                        raise ValueError("website_atlas_provider_retention_admission_mismatch")
                    admission = {**admission, "source_commit": commit}
                    _write_state(retained_path, admission)
                    submit_atlas_pose_operation(pose_inputs=pose_inputs, output_root=output_root, client=client,
                                                binding=binding, admission=admission, grant=grant, sleep=sleep, now=now)
            try:
                result = collect_atlas_pose_operation(output_root=output_root, client=client, sleep=sleep, now=now,
                                                      max_polls=max_polls, poll_interval_seconds=poll_interval_seconds)
            except ValueError as exc:
                if str(exc) == "website_atlas_pose_operation_failed" and resubmit_due():
                    continue
                raise
            break
        if result.get("status") == "processing":
            return result
        receipt = result["operation_receipt"]
        return {"status": "completed", "operation_id": receipt["operation_id"],
                "result_path": str(output_root / "atlas_posed_rgbd_result.json"), "result_digest": result["digest"],
                "submitted_frame_count": receipt["input_usage"]["submitted_frame_count"]}
    finally:
        # Report every closed attempt, retained or new; a failed or refused hold stops counting.
        settle_closed_atlas_pose_attempts(output_root, task_context=task_context)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text()) if path.is_file() else {}
    return value if isinstance(value, dict) else {}


def atlas_pose_producer_status(output_root: Path) -> dict[str, Any]:
    """What the admitted producer has done for a result that is not complete.

    Uploads are provider mutations. Once a task POST was sent, actual input
    usage is unknown until the completed receipt says otherwise.
    """
    state = _read_json(output_root / "atlas_pose" / "submission.json")
    if not state:
        return {}
    return {"atlas_pose_status": state.get("status"),
            "provider_mutation_performed": bool(state.get("created_assets")),
            "actual_provider_input_count": None if state.get("sends") else 0}


def atlas_pose_result_completed(*, pose: Mapping[str, Any], output_root: Path) -> bool:
    """Whether the producer finished this result; a finished one that no longer matches is refused."""
    state = _read_json(output_root / "atlas_pose" / "submission.json")
    if state.get("status") != "completed":
        return False
    if state.get("result_digest") != pose.get("digest"):
        raise ValueError("website_atlas_pose_receipt_invalid")
    return True


def verify_atlas_pose_receipt(*, pose: Mapping[str, Any], frames: Sequence[Mapping[str, Any]],
                              output_root: Path) -> int:
    """Actual provider input usage, only for an admitted run whose receipt still binds these inputs and outputs."""
    def mapping(value: Any) -> Mapping[str, Any]:
        return value if isinstance(value, Mapping) else {}

    receipt = mapping(pose.get("operation_receipt"))
    binding, usage, request = (mapping(receipt.get(key)) for key in ("binding", "input_usage", "request"))
    rows, sent = usage.get("frames"), request.get("frames")
    identities = _frame_identities(frames)
    admission = _read_json(output_root / "atlas_pose" / "controller_admission.json")
    state = _read_json(output_root / "atlas_pose" / "submission.json")
    if (receipt.get("schema_version") != RECEIPT_SCHEMA
            or receipt.get("operation_id") != mapping(pose.get("operation")).get("id")
            or receipt.get("request_digest") != canonical_digest(dict(binding))
            or binding.get("source_binding") != pose.get("binding") or binding.get("frames") != identities
            or binding.get("request_options") != request_options() or binding.get("provider") != PROVIDER
            # Only an admitted run reports usage: the WebApp's grant and the
            # adapter's own record must both name this exact request.
            or admission.get("status") != "admitted" or admission.get("provider") != PROVIDER
            or admission.get("allocation_binding_digest") != receipt["request_digest"]
            or state.get("status") != "completed" or state.get("request_digest") != receipt["request_digest"]
            or state.get("operation_id") != receipt["operation_id"] or state.get("result_digest") != pose.get("digest")
            or not isinstance(rows, list) or not isinstance(sent, list)
            or not all(isinstance(item, Mapping) for item in [*rows, *sent])
            or usage.get("submitted_frame_count") != len(rows) or _frame_identities(rows) != identities
            or [mapping(item.get("imageAsset")).get("assetId") for item in sent] != [row.get("asset_id") for row in rows]
            or {key: item for key, item in request.items() if key not in {"frames", "targetResolution"}}
            != request_options()
            or request.get("targetResolution") != binding.get("target_resolution")):
        raise ValueError("website_atlas_pose_receipt_invalid")
    for row in rows:
        outputs = row.get("outputs")
        if (not isinstance(outputs, list) or not all(isinstance(output, Mapping) for output in outputs)
                or not {"image", "depth"} <= {output.get("role") for output in outputs}):
            raise ValueError("website_atlas_pose_receipt_invalid")
        for output in outputs:
            relative = PurePosixPath(str(output.get("path") or ""))
            path = output_root / relative
            if (relative.is_absolute() or ".." in relative.parts or path.is_symlink() or not path.is_file()
                    or not path.resolve().is_relative_to(output_root.resolve())
                    or _sha256_file(path) != output.get("sha256")):
                raise ValueError("website_atlas_retained_output_changed")
    return len(rows)
