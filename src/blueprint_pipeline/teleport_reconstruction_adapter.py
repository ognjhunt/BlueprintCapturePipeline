"""Fail-closed Teleport reconstruction provider lifecycle.

The adapter is deliberately not a launcher.  Provider mutation requires both a
validated reconstruction-provider execution request and the opaque grant issued
by :mod:`blueprint_pipeline.paid_resource_admission`.  The canonical
``paid_resource_allocator`` is the only operator-facing caller.

Teleport receives only the frozen candidate RGB archive.  ARKit poses,
intrinsics, depth, LiDAR, gravity, hidden views, and held-out pixels never enter
the provider request.  Downloaded native bytes are hashed before import and are
preserved unchanged.  A provider READY state proves only that provider output
was available; it does not prove metric scale, collision, Isaac compatibility,
task success, physical truth, or deployment readiness.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import http.client
import ipaddress
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import ssl
import stat
import tempfile
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import urlencode, urljoin, urlsplit
import zipfile

from .decision_evidence_contracts import canonical_digest, canonical_json
from .heldout_appearance_evaluation_v2 import evaluate_heldout_appearance_v2
from .paid_resource_admission import (
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
)
from .provider_splat_import import (
    ProviderSplatImportError,
    align_provider_reconstruction,
    import_provider_splat,
)
from .reconstruction_provider_contracts import (
    build_reconstruction_provider_deletion_receipt,
    build_reconstruction_provider_execution_receipt,
    require_reconstruction_provider_execution_authority,
)
from .sealed_camera_render import (
    render_splat_at_exact_cameras,
    transform_camera_into_provider_frame,
)


TELEPORT_RESOURCE_CLASS = "provider_reconstruction_api"
TELEPORT_PROVIDER_IDENTITY = "teleport"
TELEPORT_API_BASE = "https://teleport.varjo.com"
TELEPORT_AUTH_ENDPOINT = "https://signin.teleport.varjo.com/oauth2/token"
TELEPORT_SCOPE = "openid profile email"
TELEPORT_CREATE_CAPTURE_PATH = "/api/v1/captures"
TELEPORT_DELETE_CAPTURE_PATH_TEMPLATE = "/api/v1/captures/{eid}"
TELEPORT_TERMS_REVIEW_SCHEMA = "teleport_provider_terms_review.v1"
TELEPORT_READY_PACKET_SCHEMA = "teleport_ready_to_upload_packet.v1"
TELEPORT_PROGRESS_SCHEMA = "teleport_provider_progress.v1"
TELEPORT_COST_RECEIPT_SCHEMA = "teleport_provider_cost_receipt.v1"
TELEPORT_RUN_RECEIPT_SCHEMA = "teleport_provider_run_receipt.v1"
TELEPORT_SEALED_EVALUATION_SCHEMA = "teleport_sealed_evaluation_request.v1"

DEFAULT_CLIENT_ID_FILE = Path.home() / ".blueprint-secrets" / "teleport_client_id"
DEFAULT_CLIENT_SECRET_FILE = Path.home() / ".blueprint-secrets" / "teleport_client_secret"
CLIENT_ID_FILE_ENV = "TELEPORT_CLIENT_ID_FILE"
CLIENT_SECRET_FILE_ENV = "TELEPORT_CLIENT_SECRET_FILE"
PUBLIC_UPLOAD_AUTH_ENV = "TELEPORT_PUBLIC_DATA_UPLOAD_AUTHORIZED"
PUBLIC_SPEND_CAP_ENV = "TELEPORT_PUBLIC_DATA_SPEND_CAP_USD"

MAX_JSON_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_PLY_BYTES = 4_000_000_000
MIN_POLL_INTERVAL_SECONDS = 5.0
TOKEN_REFRESH_SKEW_SECONDS = 60.0
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_ETAG = re.compile(r"^[0-9A-Fa-f]{32,64}(?:-[1-9][0-9]*)?$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class TeleportAdapterError(RuntimeError):
    """Typed, secret-free lifecycle failure."""

    def __init__(self, codes: Sequence[str], *, result: Mapping[str, Any] | None = None) -> None:
        self.codes = tuple(sorted({str(code) for code in codes if str(code)}))
        self.result = dict(result) if result is not None else None
        super().__init__("; ".join(self.codes))


class TeleportTransportError(RuntimeError):
    """Transport failure with explicit ambiguity and retry semantics."""

    def __init__(
        self,
        code: str,
        *,
        retryable: bool = False,
        ambiguous_mutation: bool = False,
    ) -> None:
        self.code = code
        self.retryable = retryable
        self.ambiguous_mutation = ambiguous_mutation
        super().__init__(code)


@dataclass(frozen=True, repr=False)
class TeleportCredentials:
    client_id: str
    client_secret: str

    def __repr__(self) -> str:
        return "TeleportCredentials(client_id=<redacted>, client_secret=<redacted>)"


@dataclass(frozen=True, repr=False)
class TeleportToken:
    access_token: str
    expires_at_monotonic: float
    token_type: str = "bearer"

    def __repr__(self) -> str:
        return "TeleportToken(access_token=<redacted>, expires_at_monotonic=<redacted>)"


class TeleportTransport(Protocol):
    def authenticate(self, credentials: TeleportCredentials) -> Mapping[str, Any]: ...

    def api_json(
        self,
        method: str,
        path: str,
        *,
        access_token: str,
        payload: Mapping[str, Any] | None = None,
        query: Mapping[str, Any] | None = None,
    ) -> tuple[int, Mapping[str, str], Any]: ...

    def upload_part(
        self,
        url: str,
        *,
        source_path: Path,
        offset: int,
        length: int,
    ) -> tuple[int, Mapping[str, str]]: ...

    def download_bytes(self, url: str, *, maximum_bytes: int) -> bytes: ...

    def download_file(self, url: str, *, destination: Path, maximum_bytes: int) -> int: ...


class _BoundedReader:
    def __init__(self, stream: Any, remaining: int) -> None:
        self._stream = stream
        self._remaining = remaining

    def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        if size < 0 or size > self._remaining:
            size = self._remaining
        data = self._stream.read(size)
        self._remaining -= len(data)
        return data


class TeleportHttpTransport:
    """Small standard-library HTTPS transport with bounded streaming."""

    def __init__(
        self,
        *,
        api_base: str = TELEPORT_API_BASE,
        auth_endpoint: str = TELEPORT_AUTH_ENDPOINT,
        timeout_seconds: float = 60.0,
        chunk_bytes: int = 1024 * 1024,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.auth_endpoint = auth_endpoint
        self.timeout_seconds = timeout_seconds
        self.chunk_bytes = chunk_bytes
        self._ssl_context = ssl.create_default_context()

    def _connection(self, url: str) -> tuple[http.client.HTTPSConnection, str]:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise TeleportTransportError("teleport_https_url_invalid")
        hostname = parsed.hostname.rstrip(".").lower()
        if hostname == "localhost" or hostname.endswith(".localhost") or hostname.endswith(".local"):
            raise TeleportTransportError("teleport_https_url_host_not_public")
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise TeleportTransportError("teleport_https_url_host_not_public")
        connection = http.client.HTTPSConnection(
            parsed.hostname,
            parsed.port or 443,
            timeout=self.timeout_seconds,
            context=self._ssl_context,
        )
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        return connection, target

    def _small_request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        body: bytes | None = None,
        redirect_cap: int = 4,
    ) -> tuple[int, dict[str, str], bytes]:
        current = url
        current_method = method
        current_body = body
        current_headers = dict(headers or {})
        for _ in range(redirect_cap + 1):
            connection, target = self._connection(current)
            try:
                connection.request(
                    current_method,
                    target,
                    body=current_body,
                    headers=current_headers,
                )
                response = connection.getresponse()
                response_headers = {key.lower(): value for key, value in response.getheaders()}
                payload = response.read(MAX_JSON_BYTES + 1)
            except (OSError, http.client.HTTPException) as exc:
                raise TeleportTransportError(
                    "teleport_http_transport_failure",
                    retryable=True,
                    ambiguous_mutation=method in {"POST", "PUT", "DELETE"},
                ) from exc
            finally:
                connection.close()
            if len(payload) > MAX_JSON_BYTES:
                raise TeleportTransportError("teleport_http_response_oversized")
            if response.status in {301, 302, 303, 307, 308}:
                location = response_headers.get("location")
                if not location:
                    raise TeleportTransportError("teleport_http_redirect_location_missing")
                redirected = urljoin(current, location)
                previous_url = urlsplit(current)
                redirected_url = urlsplit(redirected)
                previous_origin = (
                    previous_url.scheme,
                    previous_url.hostname,
                    previous_url.port or 443,
                )
                redirected_origin = (
                    redirected_url.scheme,
                    redirected_url.hostname,
                    redirected_url.port or 443,
                )
                if redirected_origin != previous_origin:
                    current_headers = {
                        key: value
                        for key, value in current_headers.items()
                        if key.lower() != "authorization"
                    }
                current = redirected
                if response.status == 303:
                    current_method, current_body = "GET", None
                    current_headers = {
                        key: value
                        for key, value in current_headers.items()
                        if key.lower() not in {"content-length", "content-type"}
                    }
                continue
            return response.status, response_headers, payload
        raise TeleportTransportError("teleport_http_redirect_cap_exceeded")

    def authenticate(self, credentials: TeleportCredentials) -> Mapping[str, Any]:
        body = urlencode(
            {
                "grant_type": "client_credentials",
                "client_id": credentials.client_id,
                "client_secret": credentials.client_secret,
                "scope": TELEPORT_SCOPE,
            }
        ).encode("utf-8")
        status, _headers, raw = self._small_request(
            "POST",
            self.auth_endpoint,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            body=body,
        )
        if status != 200:
            raise TeleportTransportError("teleport_auth_http_status_invalid", retryable=status >= 500)
        return _decode_json(raw, "teleport_auth_response")

    def api_json(
        self,
        method: str,
        path: str,
        *,
        access_token: str,
        payload: Mapping[str, Any] | None = None,
        query: Mapping[str, Any] | None = None,
    ) -> tuple[int, Mapping[str, str], Any]:
        if not path.startswith("/") or ".." in PurePosixPath(path).parts:
            raise TeleportTransportError("teleport_api_path_invalid")
        url = self.api_base + path
        if query:
            url += "?" + urlencode({key: value for key, value in query.items() if value is not None})
        body = None if payload is None else canonical_json(dict(payload)).encode("utf-8")
        headers = {"Authorization": "Bearer " + access_token, "Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        status, response_headers, raw = self._small_request(method, url, headers=headers, body=body)
        if status == 204:
            return status, response_headers, None
        return status, response_headers, _decode_json(raw, "teleport_api_response")

    def upload_part(
        self,
        url: str,
        *,
        source_path: Path,
        offset: int,
        length: int,
    ) -> tuple[int, Mapping[str, str]]:
        if length <= 0:
            raise TeleportTransportError("teleport_upload_part_length_invalid")
        connection, target = self._connection(url)
        try:
            with source_path.open("rb") as stream:
                stream.seek(offset)
                bounded = _BoundedReader(stream, length)
                connection.request(
                    "PUT",
                    target,
                    body=bounded,
                    headers={"Content-Length": str(length)},
                    encode_chunked=False,
                )
                response = connection.getresponse()
                headers = {key.lower(): value for key, value in response.getheaders()}
                response.read(MAX_JSON_BYTES + 1)
                return response.status, headers
        except (OSError, http.client.HTTPException) as exc:
            raise TeleportTransportError("teleport_part_upload_transport_failure", retryable=True) from exc
        finally:
            connection.close()

    def download_bytes(self, url: str, *, maximum_bytes: int) -> bytes:
        temporary = Path(tempfile.mkstemp(prefix="teleport-download-")[1])
        try:
            self.download_file(url, destination=temporary, maximum_bytes=maximum_bytes)
            return temporary.read_bytes()
        finally:
            temporary.unlink(missing_ok=True)

    def download_file(self, url: str, *, destination: Path, maximum_bytes: int) -> int:
        current = url
        for _ in range(5):
            connection, target = self._connection(current)
            try:
                connection.request("GET", target, headers={"Accept": "application/octet-stream"})
                response = connection.getresponse()
                headers = {key.lower(): value for key, value in response.getheaders()}
                if response.status in {301, 302, 303, 307, 308}:
                    location = headers.get("location")
                    if not location:
                        raise TeleportTransportError("teleport_download_redirect_location_missing")
                    current = urljoin(current, location)
                    response.read(MAX_JSON_BYTES + 1)
                    continue
                if response.status != 200:
                    raise TeleportTransportError(
                        "teleport_download_http_status_invalid", retryable=response.status >= 500
                    )
                declared = headers.get("content-length")
                if declared is not None:
                    try:
                        declared_size = int(declared)
                    except ValueError as exc:
                        raise TeleportTransportError("teleport_download_content_length_invalid") from exc
                    if declared_size <= 0 or declared_size > maximum_bytes:
                        raise TeleportTransportError("teleport_download_size_out_of_bounds")
                total = 0
                with destination.open("wb") as stream:
                    while True:
                        chunk = response.read(self.chunk_bytes)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > maximum_bytes:
                            raise TeleportTransportError("teleport_download_size_out_of_bounds")
                        stream.write(chunk)
                if total <= 0 or (declared is not None and total != declared_size):
                    raise TeleportTransportError("teleport_download_incomplete")
                return total
            except (OSError, http.client.HTTPException) as exc:
                raise TeleportTransportError("teleport_download_transport_failure", retryable=True) from exc
            finally:
                connection.close()
        raise TeleportTransportError("teleport_download_redirect_cap_exceeded")


def _decode_json(raw: bytes, label: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TeleportTransportError(label + "_malformed_json") from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _digest_json(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = json.loads(canonical_json(dict(value)))
    result[field] = canonical_digest(result, digest_field=field)
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _secret_path(env: Mapping[str, str], variable: str, default: Path) -> Path:
    raw = str(env.get(variable) or "").strip()
    path = Path(raw).expanduser() if raw else default
    if path.is_symlink() or not path.is_file():
        raise TeleportAdapterError([f"{variable.lower()}_missing_or_symlink"])
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise TeleportAdapterError([f"{variable.lower()}_permissions_too_open"])
    return path


def load_teleport_credentials(
    env: Mapping[str, str] | None = None,
    *,
    client_id_file: Path | None = None,
    client_secret_file: Path | None = None,
) -> TeleportCredentials:
    """Load file-backed credentials without exposing values in artifacts or reprs."""

    source = os.environ if env is None else env
    id_path = client_id_file or _secret_path(source, CLIENT_ID_FILE_ENV, DEFAULT_CLIENT_ID_FILE)
    secret_path = client_secret_file or _secret_path(
        source, CLIENT_SECRET_FILE_ENV, DEFAULT_CLIENT_SECRET_FILE
    )
    for label, path in ((CLIENT_ID_FILE_ENV, id_path), (CLIENT_SECRET_FILE_ENV, secret_path)):
        if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise TeleportAdapterError([f"{label.lower()}_invalid"])
    client_id = id_path.read_text(encoding="utf-8").strip()
    client_secret = secret_path.read_text(encoding="utf-8").strip()
    if not client_id or not client_secret or "\n" in client_id or "\n" in client_secret:
        raise TeleportAdapterError(["teleport_credentials_invalid"])
    return TeleportCredentials(client_id=client_id, client_secret=client_secret)


def validate_teleport_terms_review(value: Mapping[str, Any]) -> dict[str, Any]:
    review = dict(value)
    errors: list[str] = []
    if review.get("schema_version") != TELEPORT_TERMS_REVIEW_SCHEMA:
        errors.append("teleport_terms_review_schema_invalid")
    for key in ("reviewed_at", "api_spec_version", "terms_version", "dpa_version"):
        if not str(review.get(key) or "").strip():
            errors.append(f"teleport_terms_review_{key}_missing")
    sources = review.get("official_sources")
    if not isinstance(sources, list) or len(sources) < 5:
        errors.append("teleport_terms_review_sources_incomplete")
    else:
        for row in sources:
            if not isinstance(row, Mapping) or not str(row.get("url") or "").startswith("https://"):
                errors.append("teleport_terms_review_source_invalid")
    for field, expected in (
        ("rgb_images_or_video_only", True),
        ("known_poses_ingest_documented", False),
        ("depth_lidar_ingest_documented", False),
        ("delete_success_has_durable_receipt", False),
        ("public_exact_per_capture_price_published", False),
        ("standard_terms_training_use_grant_present", True),
        ("standard_model_license_transferable", False),
    ):
        if review.get(field) is not expected:
            errors.append(f"teleport_terms_review_{field}_invalid")
    supplied = review.pop("teleport_provider_terms_review_digest", None)
    expected = canonical_digest(review, digest_field="teleport_provider_terms_review_digest")
    review["teleport_provider_terms_review_digest"] = expected
    if supplied is not None and supplied != expected:
        errors.append("teleport_terms_review_digest_mismatch")
    if errors:
        raise TeleportAdapterError(errors)
    return review


def validate_teleport_upload_packet(value: Mapping[str, Any], *, packet_root: Path) -> dict[str, Any]:
    packet = json.loads(canonical_json(dict(value)))
    errors: list[str] = []
    if packet.get("schema_version") not in {
        "teleport_t1_upload_packet.v1",
        TELEPORT_READY_PACKET_SCHEMA,
    }:
        errors.append("teleport_upload_packet_schema_invalid")
    for key in ("source_capture_digest", "frozen_split_digest", "candidate_dataset_digest"):
        if _DIGEST.fullmatch(str(packet.get(key) or "")) is None:
            errors.append(f"teleport_upload_packet_{key}_invalid")
    if packet.get("dataset_class") != "rights_cleared_public_dataset":
        errors.append("teleport_upload_packet_not_rights_cleared_public_dataset")
    if not str(packet.get("source_license") or "").strip():
        errors.append("teleport_upload_packet_source_license_missing")
    if packet.get("customer_or_confidential_data_included") is not False:
        errors.append("teleport_upload_packet_customer_or_confidential_data_not_false")
    upload = packet.get("upload_zip")
    if not isinstance(upload, Mapping):
        errors.append("teleport_upload_packet_zip_missing")
        upload = {}
    relative = PurePosixPath(str(upload.get("relative_path") or "").replace("\\", "/"))
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        errors.append("teleport_upload_packet_zip_path_invalid")
        zip_path = packet_root / "invalid"
    else:
        zip_path = (packet_root / Path(*relative.parts)).resolve()
        try:
            zip_path.relative_to(packet_root.resolve())
        except ValueError:
            errors.append("teleport_upload_packet_zip_path_escape")
    if zip_path.is_symlink() or not zip_path.is_file():
        errors.append("teleport_upload_packet_zip_missing")
    else:
        if _sha256_file(zip_path) != upload.get("digest"):
            errors.append("teleport_upload_packet_zip_digest_mismatch")
        if zip_path.stat().st_size != upload.get("size_bytes"):
            errors.append("teleport_upload_packet_zip_size_mismatch")
    mapping = packet.get("upload_name_to_observation_id")
    if not isinstance(mapping, Mapping) or not mapping:
        errors.append("teleport_upload_packet_image_mapping_missing")
    else:
        if len(mapping) != upload.get("image_count"):
            errors.append("teleport_upload_packet_image_count_mismatch")
        for name, observation_id in mapping.items():
            path = PurePosixPath(str(name))
            if (
                len(path.parts) != 1
                or path.suffix.lower() not in _IMAGE_SUFFIXES
                or not _SAFE_ID.fullmatch(str(observation_id or ""))
            ):
                errors.append("teleport_upload_packet_image_mapping_invalid")
    if zip_path.is_file() and isinstance(mapping, Mapping):
        try:
            with zipfile.ZipFile(zip_path) as archive:
                entries = archive.infolist()
                entry_names = [entry.filename for entry in entries]
                if (
                    len(entry_names) != len(set(entry_names))
                    or set(entry_names) != {str(name) for name in mapping}
                ):
                    errors.append("teleport_upload_packet_zip_entries_not_exact")
                for entry in entries:
                    if (
                        entry.is_dir()
                        or len(PurePosixPath(entry.filename).parts) != 1
                        or PurePosixPath(entry.filename).suffix.lower() not in _IMAGE_SUFFIXES
                        or entry.compress_type != zipfile.ZIP_STORED
                        or entry.flag_bits & 0x1
                        or entry.file_size <= 0
                    ):
                        errors.append("teleport_upload_packet_zip_entry_invalid")
                        break
        except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile):
            errors.append("teleport_upload_packet_zip_invalid")
    for key in ("hidden_images_included", "hidden_filenames_included"):
        if packet.get(key) is not False:
            errors.append(f"teleport_upload_packet_{key}_not_false")
    digest_field = (
        "teleport_t1_upload_packet_digest"
        if packet.get("schema_version") == "teleport_t1_upload_packet.v1"
        else "teleport_ready_to_upload_packet_digest"
    )
    supplied = packet.get(digest_field)
    if supplied != canonical_digest(packet, digest_field=digest_field):
        errors.append("teleport_upload_packet_digest_mismatch")
    if errors:
        raise TeleportAdapterError(errors)
    packet["_resolved_zip_path"] = str(zip_path)
    return packet


def _normalize_etag(raw: Any) -> str:
    value = str(raw or "")
    if value.startswith('"') and value.endswith('"') and len(value) > 2:
        value = value[1:-1]
    if not _ETAG.fullmatch(value):
        raise TeleportAdapterError(["teleport_upload_etag_invalid"])
    return value


def validate_uploaded_parts(
    parts: Sequence[Mapping[str, Any]], *, expected_count: int
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in parts:
        if not isinstance(row, Mapping) or isinstance(row.get("number"), bool):
            raise TeleportAdapterError(["teleport_uploaded_part_invalid"])
        normalized.append({"number": int(row["number"]), "etag": _normalize_etag(row.get("etag"))})
    numbers = [row["number"] for row in normalized]
    if numbers != list(range(1, expected_count + 1)) or len(set(numbers)) != len(numbers):
        raise TeleportAdapterError(["teleport_uploaded_part_numbers_not_exact"])
    return normalized


def normalize_teleport_cameras(value: Any) -> list[dict[str, Any]]:
    rows = value.get("cameras") if isinstance(value, Mapping) else value
    if not isinstance(rows, list) or not rows:
        raise TeleportAdapterError(["teleport_camera_metadata_rows_missing"])
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise TeleportAdapterError(["teleport_camera_metadata_row_invalid"])
        image_name = str(row.get("image_name") or row.get("filename") or "")
        if not image_name or PurePosixPath(image_name).name != image_name or image_name in seen:
            raise TeleportAdapterError(["teleport_camera_image_name_invalid_or_duplicate"])
        seen.add(image_name)
        position = row.get("position") or row.get("camera_center")
        if not isinstance(position, list) or len(position) != 3:
            raise TeleportAdapterError(["teleport_camera_explicit_center_missing"])
        try:
            center = [float(item) for item in position]
        except (TypeError, ValueError) as exc:
            raise TeleportAdapterError(["teleport_camera_center_invalid"]) from exc
        if not all(math.isfinite(item) for item in center):
            raise TeleportAdapterError(["teleport_camera_center_invalid"])
        result.append({"image_name": image_name, "position": center})
    return result


def _validate_metadata(value: Any, *, expected_sid: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TeleportAdapterError(["teleport_v2_metadata_not_object"])
    metadata = dict(value)
    errors: list[str] = []
    if metadata.get("sid") != expected_sid:
        errors.append("teleport_v2_metadata_sid_mismatch")
    if metadata.get("content_profile") != "ply":
        errors.append("teleport_v2_metadata_profile_not_full_ply")
    if metadata.get("coord_system") != "colmap":
        errors.append("teleport_v2_metadata_not_colmap")
    for key in ("model_url", "cameras_url"):
        parsed = urlsplit(str(metadata.get(key) or ""))
        if parsed.scheme != "https" or not parsed.hostname:
            errors.append(f"teleport_v2_metadata_{key}_invalid")
    if errors:
        raise TeleportAdapterError(errors)
    return metadata


class _Lifecycle:
    def __init__(
        self,
        *,
        transport: TeleportTransport,
        credentials: TeleportCredentials,
        retry_cap: int,
        poll_interval_seconds: float,
        monotonic: Callable[[], float],
        wall_time: Callable[[], str],
        sleep: Callable[[float], None],
        progress_path: Path,
    ) -> None:
        self.transport = transport
        self.credentials = credentials
        self.retry_cap = retry_cap
        self.poll_interval_seconds = max(poll_interval_seconds, MIN_POLL_INTERVAL_SECONDS)
        self.monotonic = monotonic
        self.wall_time = wall_time
        self.sleep = sleep
        self.progress_path = progress_path
        self.token: TeleportToken | None = None
        self.progress: list[dict[str, Any]] = []

    def event(self, phase: str, **fields: Any) -> None:
        row = {"sequence": len(self.progress) + 1, "timestamp": self.wall_time(), "phase": phase}
        row.update(fields)
        self.progress.append(row)
        artifact = _digest_json(
            {
                "schema_version": TELEPORT_PROGRESS_SCHEMA,
                "events": self.progress,
                "raw_secret_values_recorded": False,
            },
            "teleport_provider_progress_digest",
        )
        _write_json(self.progress_path, artifact)

    def authenticate(self) -> TeleportToken:
        response: Mapping[str, Any] | None = None
        for attempt in range(self.retry_cap + 1):
            try:
                response = self.transport.authenticate(self.credentials)
                break
            except TeleportTransportError as exc:
                if not exc.retryable or attempt >= self.retry_cap:
                    raise TeleportAdapterError([exc.code]) from exc
                self.sleep(min(self.poll_interval_seconds, 2.0**attempt))
        if not isinstance(response, Mapping):
            raise TeleportAdapterError(["teleport_auth_response_invalid"])
        access_token = response.get("access_token")
        expires_in = response.get("expires_in")
        token_type = str(response.get("token_type") or "").lower()
        if (
            not isinstance(access_token, str)
            or not access_token
            or isinstance(expires_in, bool)
            or not isinstance(expires_in, (int, float))
            or not math.isfinite(float(expires_in))
            or float(expires_in) <= TOKEN_REFRESH_SKEW_SECONDS
            or token_type != "bearer"
        ):
            raise TeleportAdapterError(["teleport_auth_response_invalid"])
        self.token = TeleportToken(
            access_token=access_token,
            expires_at_monotonic=self.monotonic() + float(expires_in),
            token_type=token_type,
        )
        self.event("authenticated", token_expires_in_seconds=int(float(expires_in)))
        return self.token

    def _access_token(self) -> str:
        if (
            self.token is None
            or self.monotonic() >= self.token.expires_at_monotonic - TOKEN_REFRESH_SKEW_SECONDS
        ):
            self.authenticate()
        assert self.token is not None
        return self.token.access_token

    def api(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        query: Mapping[str, Any] | None = None,
        allow_refresh: bool = True,
    ) -> tuple[int, Mapping[str, str], Any]:
        status, headers, value = self.transport.api_json(
            method,
            path,
            access_token=self._access_token(),
            payload=payload,
            query=query,
        )
        if status == 401 and allow_refresh:
            self.token = None
            self.event("auth_expired_refresh")
            return self.api(method, path, payload=payload, query=query, allow_refresh=False)
        return status, headers, value

    def list_captures(self, *, name: str | None = None) -> list[dict[str, Any]]:
        last_error: TeleportTransportError | None = None
        for attempt in range(self.retry_cap + 1):
            try:
                status, _headers, value = self.api(
                    "GET",
                    TELEPORT_CREATE_CAPTURE_PATH,
                    query={"limit": 100, "offset": 0, "name": name},
                )
                if status == 429 or status >= 500:
                    raise TeleportTransportError("teleport_capture_list_retryable", retryable=True)
                if status != 200 or not isinstance(value, list):
                    raise TeleportAdapterError(["teleport_capture_list_response_invalid"])
                rows = []
                for row in value:
                    if (
                        not isinstance(row, Mapping)
                        or _SAFE_ID.fullmatch(str(row.get("eid") or "")) is None
                    ):
                        raise TeleportAdapterError(["teleport_capture_list_row_invalid"])
                    rows.append(dict(row))
                return rows
            except TeleportTransportError as exc:
                last_error = exc
                if not exc.retryable or attempt >= self.retry_cap:
                    break
                self.sleep(min(self.poll_interval_seconds, 2.0**attempt))
        raise TeleportAdapterError([last_error.code if last_error else "teleport_capture_list_failed"])


def _capture_by_eid(
    lifecycle: _Lifecycle, *, eid: str, capture_name: str
) -> dict[str, Any] | None:
    if _SAFE_ID.fullmatch(eid) is None:
        raise TeleportAdapterError(["teleport_capture_eid_invalid"])
    matches = [
        row
        for row in lifecycle.list_captures(name=capture_name)
        if row.get("eid") == eid and row.get("name") == capture_name
    ]
    if len(matches) > 1:
        raise TeleportAdapterError(["teleport_capture_identity_not_unique"])
    return matches[0] if matches else None


def _create_or_reconcile_capture(
    lifecycle: _Lifecycle,
    *,
    capture_name: str,
    bytesize: int,
    training_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    existing = [row for row in lifecycle.list_captures(name=capture_name) if row.get("name") == capture_name]
    if len(existing) > 1:
        raise TeleportAdapterError(["teleport_immutable_job_identity_collision"])
    if len(existing) == 1:
        if _SAFE_ID.fullmatch(str(existing[0].get("eid") or "")) is None:
            raise TeleportAdapterError(["teleport_capture_eid_invalid"])
        lifecycle.event("capture_reconciled", eid=existing[0]["eid"])
        return dict(existing[0])
    payload = {
        "name": capture_name,
        "bytesize": bytesize,
        "input_data_format": "bulk-images",
        "training_parameters": dict(training_parameters),
    }
    status = 0
    response: Any = None
    for attempt in range(lifecycle.retry_cap + 1):
        try:
            status, _headers, response = lifecycle.api(
                "POST", TELEPORT_CREATE_CAPTURE_PATH, payload=payload
            )
        except TeleportTransportError as exc:
            if not exc.ambiguous_mutation:
                raise TeleportAdapterError([exc.code]) from exc
            reconciled = [
                row
                for row in lifecycle.list_captures(name=capture_name)
                if row.get("name") == capture_name
            ]
            if len(reconciled) != 1:
                raise TeleportAdapterError(
                    ["teleport_create_capture_ambiguous_unreconciled"]
                ) from exc
            lifecycle.event(
                "capture_create_ambiguity_reconciled", eid=reconciled[0]["eid"]
            )
            return dict(reconciled[0])
        if status == 429:
            reconciled = [
                row
                for row in lifecycle.list_captures(name=capture_name)
                if row.get("name") == capture_name
            ]
            if len(reconciled) == 1:
                lifecycle.event(
                    "capture_create_rate_limit_reconciled", eid=reconciled[0]["eid"]
                )
                return dict(reconciled[0])
            if len(reconciled) > 1:
                raise TeleportAdapterError(["teleport_immutable_job_identity_collision"])
            if attempt >= lifecycle.retry_cap:
                raise TeleportAdapterError(["teleport_create_capture_rate_limited"])
            lifecycle.sleep(min(lifecycle.poll_interval_seconds, 2.0**attempt))
            continue
        if status >= 500:
            reconciled = [
                row
                for row in lifecycle.list_captures(name=capture_name)
                if row.get("name") == capture_name
            ]
            if len(reconciled) == 1:
                lifecycle.event(
                    "capture_create_http_ambiguity_reconciled", eid=reconciled[0]["eid"]
                )
                return dict(reconciled[0])
            raise TeleportAdapterError(["teleport_create_capture_ambiguous_unreconciled"])
        break
    if status != 200 or not isinstance(response, Mapping):
        raise TeleportAdapterError(["teleport_create_capture_response_invalid"])
    try:
        eid = str(response["eid"])
        num_parts = int(response["num_parts"])
        chunk_size = int(response["chunk_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TeleportAdapterError(["teleport_create_capture_response_invalid"]) from exc
    if (
        _SAFE_ID.fullmatch(eid) is None
        or num_parts <= 0
        or chunk_size <= 0
        or num_parts != math.ceil(bytesize / chunk_size)
    ):
        raise TeleportAdapterError(["teleport_create_capture_partition_invalid"])
    lifecycle.event("capture_created", eid=eid, num_parts=num_parts, chunk_size=chunk_size)
    return {
        "eid": eid,
        "name": capture_name,
        "num_parts": num_parts,
        "chunk_size": chunk_size,
        "state": "CREATED",
    }


def _upload_parts(
    lifecycle: _Lifecycle,
    *,
    capture: Mapping[str, Any],
    source_path: Path,
    deadline: float,
) -> list[dict[str, Any]]:
    eid = str(capture["eid"])
    if _SAFE_ID.fullmatch(eid) is None:
        raise TeleportAdapterError(["teleport_capture_eid_invalid"])
    count = int(capture["num_parts"])
    chunk_size = int(capture["chunk_size"])
    total_size = source_path.stat().st_size
    parts: list[dict[str, Any]] = []
    for part_number in range(1, count + 1):
        if lifecycle.monotonic() >= deadline:
            raise TeleportAdapterError(["teleport_upload_ttl_expired"])
        offset = (part_number - 1) * chunk_size
        length = min(chunk_size, total_size - offset)
        if length <= 0:
            raise TeleportAdapterError(["teleport_upload_partition_invalid"])
        status = 0
        url_response: Any = None
        for attempt in range(lifecycle.retry_cap + 1):
            if lifecycle.monotonic() >= deadline:
                raise TeleportAdapterError(["teleport_upload_ttl_expired"])
            try:
                status, _headers, url_response = lifecycle.api(
                    "POST",
                    f"/api/v1/captures/{eid}/create-upload-url/{part_number}",
                    payload={"eid": eid, "bytesize": total_size},
                )
                if status == 429 or status >= 500:
                    raise TeleportTransportError(
                        "teleport_upload_url_retryable", retryable=True
                    )
                break
            except TeleportTransportError as exc:
                if not exc.retryable or attempt >= lifecycle.retry_cap:
                    raise TeleportAdapterError([exc.code]) from exc
                lifecycle.sleep(min(lifecycle.poll_interval_seconds, 2.0**attempt))
        if status != 200 or not isinstance(url_response, Mapping):
            raise TeleportAdapterError(["teleport_upload_url_response_invalid"])
        upload_url = str(url_response.get("upload_url") or "")
        if urlsplit(upload_url).scheme != "https":
            raise TeleportAdapterError(["teleport_upload_url_invalid"])
        upload_status = 0
        headers: Mapping[str, str] = {}
        for attempt in range(lifecycle.retry_cap + 1):
            if lifecycle.monotonic() >= deadline:
                raise TeleportAdapterError(["teleport_upload_ttl_expired"])
            try:
                upload_status, headers = lifecycle.transport.upload_part(
                    upload_url,
                    source_path=source_path,
                    offset=offset,
                    length=length,
                )
                if upload_status == 429 or upload_status >= 500:
                    raise TeleportTransportError("teleport_part_upload_retryable", retryable=True)
                break
            except TeleportTransportError as exc:
                if not exc.retryable or attempt >= lifecycle.retry_cap:
                    raise TeleportAdapterError([exc.code]) from exc
                lifecycle.sleep(min(lifecycle.poll_interval_seconds, 2.0**attempt))
        if upload_status < 200 or upload_status >= 300:
            raise TeleportAdapterError(["teleport_part_upload_http_status_invalid"])
        etag_values = [value for key, value in headers.items() if str(key).lower() == "etag"]
        if len(etag_values) != 1:
            raise TeleportAdapterError(["teleport_upload_etag_missing_or_ambiguous"])
        parts.append({"number": part_number, "etag": _normalize_etag(etag_values[0])})
        lifecycle.event("part_uploaded", part_number=part_number, size_bytes=length)
    return validate_uploaded_parts(parts, expected_count=count)


def _complete_upload_once(
    lifecycle: _Lifecycle,
    *,
    eid: str,
    capture_name: str,
    parts: Sequence[Mapping[str, Any]],
    deadline: float,
) -> dict[str, Any]:
    if lifecycle.monotonic() >= deadline:
        raise TeleportAdapterError(["teleport_upload_complete_ttl_expired"])
    capture = _capture_by_eid(lifecycle, eid=eid, capture_name=capture_name)
    if capture is None:
        raise TeleportAdapterError(["teleport_capture_missing_before_upload_complete"])
    if str(capture.get("state") or "") not in {"CREATED", "UPLOADING", "UPLOAD_PENDING"}:
        lifecycle.event("upload_completion_reconciled", state=capture.get("state"))
        return capture
    try:
        status, _headers, response = lifecycle.api(
            "POST",
            f"/api/v1/captures/{eid}/uploaded",
            payload={"eid": eid, "parts": list(parts)},
        )
    except TeleportTransportError as exc:
        if not exc.ambiguous_mutation:
            raise TeleportAdapterError([exc.code]) from exc
        reconciled = _capture_by_eid(lifecycle, eid=eid, capture_name=capture_name)
        if reconciled is None or str(reconciled.get("state") or "") in {
            "CREATED",
            "UPLOADING",
            "UPLOAD_PENDING",
        }:
            raise TeleportAdapterError(["teleport_upload_complete_ambiguous_unreconciled"]) from exc
        lifecycle.event("upload_complete_ambiguity_reconciled", state=reconciled.get("state"))
        return reconciled
    if status == 429 or status >= 500:
        reconciled = _capture_by_eid(lifecycle, eid=eid, capture_name=capture_name)
        if reconciled is not None and str(reconciled.get("state") or "") not in {
            "CREATED",
            "UPLOADING",
            "UPLOAD_PENDING",
        }:
            lifecycle.event(
                "upload_complete_http_ambiguity_reconciled",
                state=reconciled.get("state"),
            )
            return reconciled
        raise TeleportAdapterError(["teleport_upload_complete_ambiguous_unreconciled"])
    if status != 200 or not isinstance(response, Mapping) or not str(response.get("state") or ""):
        raise TeleportAdapterError(["teleport_upload_complete_response_invalid"])
    lifecycle.event("upload_completed_exactly_once", state=response.get("state"))
    return dict(response)


def _poll_ready(
    lifecycle: _Lifecycle, *, eid: str, capture_name: str, deadline: float
) -> dict[str, Any]:
    while lifecycle.monotonic() < deadline:
        capture = _capture_by_eid(lifecycle, eid=eid, capture_name=capture_name)
        if capture is None:
            raise TeleportAdapterError(["teleport_capture_disappeared_during_poll"])
        state = str(capture.get("state") or "")
        lifecycle.event(
            "poll",
            state=state,
            state_description=capture.get("state_description"),
            error_reason_slug=capture.get("error_reason_slug"),
        )
        if state == "READY":
            if _SAFE_ID.fullmatch(str(capture.get("sid") or "")) is None:
                raise TeleportAdapterError(["teleport_ready_capture_sid_missing"])
            return capture
        if state == "FAILED" or capture.get("error_reason") or capture.get("error_reason_slug"):
            raise TeleportAdapterError(["teleport_training_failed"])
        lifecycle.sleep(lifecycle.poll_interval_seconds)
    raise TeleportAdapterError(["teleport_poll_timeout"])


def _provider_output(
    *, artifact_id: str, artifact_kind: str, path: Path, relative_to: Path
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "artifact_kind": artifact_kind,
        "relative_path": path.relative_to(relative_to).as_posix(),
        "digest": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "download_complete": True,
        "hash_verified": True,
        "provider_native_bytes_preserved": True,
    }


def build_teleport_sealed_evaluation_runner(
    source: Mapping[str, Any],
) -> Callable[..., Mapping[str, Any]]:
    """Compile evaluator-owned hidden inputs into a post-provider callback.

    Merely building the callback does not open reference pixels.  The callback
    is invoked only after Teleport outputs are downloaded, imported, and
    aligned from candidate cameras.
    """

    config = json.loads(canonical_json(dict(source)))
    errors: list[str] = []
    if config.get("schema_version") != TELEPORT_SEALED_EVALUATION_SCHEMA:
        errors.append("teleport_sealed_evaluation_schema_invalid")
    for field, expected in (
        ("provider_received_hidden_views", False),
        ("provider_received_hidden_pixels", False),
        ("split_frozen_before_provider_execution", True),
        ("thresholds_frozen_before_evaluation", True),
    ):
        if config.get(field) is not expected:
            errors.append(f"teleport_sealed_evaluation_{field}_invalid")
    hidden = config.get("hidden_views")
    if not isinstance(hidden, list) or not hidden:
        errors.append("teleport_sealed_evaluation_hidden_views_missing")
    evaluator_root = Path(str(config.get("evaluator_root") or "")).expanduser().resolve()
    if evaluator_root.is_symlink() or not evaluator_root.is_dir():
        errors.append("teleport_sealed_evaluation_root_invalid")
    if errors:
        raise TeleportAdapterError(errors)

    def runner(**kwargs: Any) -> Mapping[str, Any]:
        output_root = Path(kwargs["output_root"])
        import_receipt = dict(kwargs["import_receipt"])
        alignment = dict(kwargs["alignment"])
        execution_request = dict(kwargs["execution_request"])
        packet = dict(kwargs["packet"])
        cameras = []
        pairs_by_id: dict[str, Mapping[str, Any]] = {}
        for row in hidden:
            if not isinstance(row, Mapping):
                raise TeleportAdapterError(["teleport_sealed_evaluation_hidden_view_invalid"])
            view_id = str(row.get("view_id") or "")
            if not _SAFE_ID.fullmatch(view_id) or view_id in pairs_by_id:
                raise TeleportAdapterError(["teleport_sealed_evaluation_view_id_invalid"])
            cameras.append(
                {
                    "camera_id": view_id,
                    "T_world_camera_provider_frame": transform_camera_into_provider_frame(
                        camera_to_world_candidate=row.get("T_world_camera_candidate_frame"),
                        alignment=alignment,
                    ),
                    "intrinsics": dict(row.get("intrinsics") or {}),
                }
            )
            pairs_by_id[view_id] = row
        splat_assets = [
            row
            for row in import_receipt.get("imported_assets", [])
            if row.get("artifact_kind") == "splat_ply"
        ]
        if len(splat_assets) != 1:
            raise TeleportAdapterError(["teleport_sealed_evaluation_imported_splat_missing"])
        splat = output_root / "imports" / str(splat_assets[0]["relative_path"])
        render_root = output_root / "sealed_heldout_evaluation" / "candidate_renders"
        render_manifest = render_splat_at_exact_cameras(
            splat_path=splat,
            cameras=cameras,
            output_dir=render_root,
            provider_splat_import_receipt_digest=import_receipt[
                "provider_splat_import_receipt_digest"
            ],
            alignment_digest=alignment["provider_reconstruction_alignment_digest"],
            camera_set_label="evaluator_owned_hidden_views",
        )
        renders = {row["camera_id"]: row for row in render_manifest["renders"]}
        pairs = []
        for view_id, row in pairs_by_id.items():
            render = renders.get(view_id)
            if render is None:
                raise TeleportAdapterError(["teleport_sealed_evaluation_render_missing"])
            pairs.append(
                {
                    "view_id": view_id,
                    "trajectory": row.get("trajectory"),
                    "split": "held_out",
                    "excluded_from_training": True,
                    "real_view_relative_path": row.get("real_view_relative_path"),
                    "real_view_digest": row.get("real_view_digest"),
                    "candidate_render_relative_path": render["relative_path"],
                    "candidate_render_digest": render["digest"],
                }
            )
        evaluation_request = {
            "schema_version": "heldout_appearance_evaluation_request.v2",
            "stable_run_identity": execution_request["stable_run_identity"],
            "source_capture_identity": execution_request["source_capture_identity"],
            "source_capture_digest": execution_request["source_capture_digest"],
            "reconstruction_dataset_digest": config["reconstruction_dataset_digest"],
            "frozen_split_digest": execution_request["train_heldout_split_digest"],
            "candidate_reconstruction_result_digest": alignment[
                "provider_reconstruction_alignment_digest"
            ],
            "evaluator_implementation_digest": config["evaluator_implementation_digest"],
            "source_commit_sha": execution_request["source_commit_sha"],
            "candidate_method_id": "teleport_modelv3",
            "candidate_provider_identity": TELEPORT_PROVIDER_IDENTITY,
            "evaluator_identity": config["evaluator_identity"],
            "evaluator_provider_identity": config["evaluator_provider_identity"],
            "candidate_root": str(render_root),
            "evaluator_root": str(evaluator_root),
            "coordinate_frame_declaration": {
                "provider_frame": "teleport_colmap",
                "alignment_digest": alignment["provider_reconstruction_alignment_digest"],
            },
            "authority_used": dict(config.get("authority_used") or {}),
            "split_frozen_before_training": True,
            "thresholds_frozen_before_evaluation": True,
            "candidate_had_hidden_access": False,
            "candidate_selected_heldout": False,
            "candidate_self_grading": False,
            "lpips_required": config.get("lpips_required", False),
            "lpips_model": config.get("lpips_model"),
            "thresholds": dict(config["thresholds"]),
            "pairs": pairs,
            "timestamp": str(config["timestamp"]),
            "provider_upload_packet_digest": packet.get(
                "teleport_t1_upload_packet_digest"
            )
            or packet.get("teleport_ready_to_upload_packet_digest"),
        }
        report = evaluate_heldout_appearance_v2(
            source_artifact=evaluation_request,
            output_root=output_root / "sealed_heldout_evaluation",
        )
        _write_json(
            output_root
            / "sealed_heldout_evaluation"
            / "visual_heldout_evaluation_report.v2.json",
            report,
        )
        return report

    return runner


def _default_sealed_evaluation_runner(**_kwargs: Any) -> Mapping[str, Any]:
    raise TeleportAdapterError(["teleport_sealed_heldout_evaluation_runner_missing"])


def _delete_and_verify(
    lifecycle: _Lifecycle,
    *,
    eid: str,
    capture_name: str,
    execution_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    delete_status: int | None = None
    delete_error: str | None = None
    for attempt in range(lifecycle.retry_cap + 1):
        try:
            delete_status, _headers, body = lifecycle.api(
                "DELETE", TELEPORT_DELETE_CAPTURE_PATH_TEMPLATE.format(eid=eid)
            )
            if delete_status == 204 and body is None:
                delete_error = None
                break
            delete_error = "teleport_delete_response_not_204_empty"
            retryable_status = delete_status == 429 or delete_status >= 500
            if not retryable_status or attempt >= lifecycle.retry_cap:
                break
        except (TeleportAdapterError, TeleportTransportError) as exc:
            delete_error = getattr(exc, "code", None) or str(exc)
            if not getattr(exc, "retryable", False) or attempt >= lifecycle.retry_cap:
                break
        lifecycle.sleep(min(lifecycle.poll_interval_seconds, 2.0**attempt))
    absent_checks = 0
    for _ in range(2):
        try:
            if _capture_by_eid(
                lifecycle, eid=eid, capture_name=capture_name
            ) is None:
                absent_checks += 1
            else:
                break
        except TeleportAdapterError:
            break
    verified = delete_status == 204 and delete_error is None and absent_checks == 2
    limitations = [
        "teleport_delete_returns_204_without_durable_deletion_receipt",
        "provider_backups_subprocessors_and_training_derivatives_not_observable_via_api",
        "provider_zero_not_proven",
    ]
    receipt = build_reconstruction_provider_deletion_receipt(
        {
            "provider_execution_receipt_digest": execution_receipt[
                "provider_execution_receipt_digest"
            ],
            "provider_identity": TELEPORT_PROVIDER_IDENTITY,
            "status": "verified_deleted" if verified else "failed",
            "provider_evidence": {
                "delete_http_status": delete_status,
                "delete_response_body_present": False if delete_status == 204 else None,
                "post_delete_absent_list_checks": absent_checks,
                "durable_provider_deletion_receipt_received": False,
                "failure_code": delete_error,
                "limitations": limitations,
            },
            "timestamp": lifecycle.wall_time(),
            "independently_verified": True,
            "provider_zero_proven": False,
            "proof_effect": "provider_deletion_evidence_only",
            "claim_ceiling": "none",
        },
        execution_receipt=execution_receipt,
    )
    lifecycle.event("deletion_verified" if verified else "deletion_failed", absent_checks=absent_checks)
    return receipt


def _cost_receipt(
    *,
    request: Mapping[str, Any],
    quoted_cost_usd: float,
    deletion_receipt: Mapping[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    maximum = float(request["max_cost_usd"])
    value = {
        "schema_version": TELEPORT_COST_RECEIPT_SCHEMA,
        "provider_identity": TELEPORT_PROVIDER_IDENTITY,
        "provider_execution_request_digest": request["provider_execution_request_digest"],
        "currency": "USD",
        "authorized_spend_cap_usd": maximum,
        "preapproved_provider_quote_usd": quoted_cost_usd,
        "provider_billing_api_available": False,
        "observed_final_charge_usd": None,
        "within_authorized_ceiling": quoted_cost_usd <= maximum,
        "reconciliation_status": "quote_bounded_final_provider_charge_unobservable",
        "provider_deletion_receipt_digest": deletion_receipt[
            "provider_deletion_receipt_digest"
        ],
        "limitations": [
            "public_pricing_is_image_count_based_without_exact_public_rate_table",
            "teleport_api_exposes_no_final_billing_receipt_endpoint",
            "operator_billing_statement_required_for_final_spend_reconciliation",
        ],
        "timestamp": timestamp,
        "proof_effect": "cost_bound_and_reconciliation_limitation_only",
        "claim_ceiling": "none",
    }
    return _digest_json(value, "teleport_provider_cost_receipt_digest")


def run_teleport_reconstruction(
    *,
    upload_packet_path: str | Path,
    execution_request: Mapping[str, Any],
    candidate_observations: Sequence[Mapping[str, Any]],
    output_root: str | Path,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    credentials: TeleportCredentials,
    transport: TeleportTransport | None = None,
    sealed_evaluation_runner: Callable[..., Mapping[str, Any]] = _default_sealed_evaluation_runner,
    maximum_ply_bytes: int = DEFAULT_MAX_PLY_BYTES,
    poll_interval_seconds: float = 30.0,
    monotonic: Callable[[], float] = time.monotonic,
    wall_time: Callable[[], str] = _utc_now,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Execute one authorized Teleport lifecycle and always attempt cleanup."""

    require_paid_resource_admission_grant(
        paid_resource_admission_grant, resource_class=TELEPORT_RESOURCE_CLASS
    )
    request = require_reconstruction_provider_execution_authority(
        execution_request, at_time=wall_time()
    )
    if request.get("provider_identity") != TELEPORT_PROVIDER_IDENTITY:
        raise TeleportAdapterError(["teleport_execution_request_provider_mismatch"])
    packet_path = Path(upload_packet_path).resolve()
    packet = validate_teleport_upload_packet(
        json.loads(packet_path.read_text(encoding="utf-8")), packet_root=packet_path.parent
    )
    for packet_field, request_field in (
        ("source_capture_digest", "source_capture_digest"),
        ("frozen_split_digest", "train_heldout_split_digest"),
        ("deterministic_configuration_digest", "deterministic_configuration_digest"),
    ):
        if packet.get(packet_field) != request.get(request_field):
            raise TeleportAdapterError([f"teleport_packet_{packet_field}_request_mismatch"])
    zip_digest = packet["upload_zip"]["digest"]
    if zip_digest not in request["immutable_input_digests"]:
        raise TeleportAdapterError(["teleport_upload_zip_not_authorized_input"])
    runtime_identity = request.get("provider_runtime_identity")
    quote = runtime_identity.get("provider_quote") if isinstance(runtime_identity, Mapping) else None
    quoted_cost = quote.get("quoted_cost_usd") if isinstance(quote, Mapping) else None
    if (
        isinstance(quoted_cost, bool)
        or not isinstance(quoted_cost, (int, float))
        or not math.isfinite(float(quoted_cost))
        or float(quoted_cost) <= 0
    ):
        raise TeleportAdapterError(["teleport_exact_provider_quote_missing"])
    quoted_cost = float(quoted_cost)
    if quoted_cost > float(request["max_cost_usd"]):
        raise TeleportAdapterError(["teleport_spend_ceiling_exceeded_before_mutation"])
    training = runtime_identity.get("training_parameters") if isinstance(runtime_identity, Mapping) else None
    if not isinstance(training, Mapping) or not isinstance(training.get("modelv3"), Mapping):
        raise TeleportAdapterError(["teleport_training_parameters_missing"])
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lifecycle = _Lifecycle(
        transport=transport or TeleportHttpTransport(),
        credentials=credentials,
        retry_cap=int(request["retry_cap"]),
        poll_interval_seconds=poll_interval_seconds,
        monotonic=monotonic,
        wall_time=wall_time,
        sleep=sleep,
        progress_path=output / "teleport_provider_progress.v1.json",
    )
    start = monotonic()
    deadline = start + int(request["ttl_seconds"])
    request_digest = request["provider_execution_request_digest"]
    capture_name = "bp-" + request_digest[7:31] + ".zip"
    source_path = Path(packet["_resolved_zip_path"])
    capture: dict[str, Any] | None = None
    downloaded_outputs: list[dict[str, Any]] = []
    execution_receipt: dict[str, Any] | None = None
    deletion_receipt: dict[str, Any] | None = None
    import_receipt: dict[str, Any] | None = None
    alignment: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    failure_codes: list[str] = []
    try:
        if lifecycle.monotonic() >= deadline:
            raise TeleportAdapterError(["teleport_create_ttl_expired"])
        capture = _create_or_reconcile_capture(
            lifecycle,
            capture_name=capture_name,
            bytesize=source_path.stat().st_size,
            training_parameters=training,
        )
        if "num_parts" not in capture or "chunk_size" not in capture:
            raise TeleportAdapterError(["teleport_reconciled_capture_upload_shape_unavailable"])
        parts = _upload_parts(
            lifecycle,
            capture=capture,
            source_path=source_path,
            deadline=deadline,
        )
        _complete_upload_once(
            lifecycle,
            eid=str(capture["eid"]),
            capture_name=capture_name,
            parts=parts,
            deadline=deadline,
        )
        ready = _poll_ready(
            lifecycle,
            eid=str(capture["eid"]),
            capture_name=capture_name,
            deadline=deadline,
        )
        sid = str(ready["sid"])
        metadata_status, _metadata_headers, metadata_value = lifecycle.api(
            "GET",
            f"/api/v2/captures/{sid}/metadata",
            query={"content_profile": "ply", "coord_system": "colmap"},
        )
        if metadata_status != 200:
            raise TeleportAdapterError(["teleport_v2_metadata_http_status_invalid"])
        metadata = _validate_metadata(metadata_value, expected_sid=sid)
        native = output / "provider_native"
        native.mkdir(exist_ok=True)
        metadata_path = native / "metadata_v2_colmap_ply.json"
        metadata_path.write_text(canonical_json(metadata) + "\n", encoding="utf-8")
        camera_bytes = lifecycle.transport.download_bytes(
            str(metadata["cameras_url"]), maximum_bytes=MAX_JSON_BYTES
        )
        camera_value = _decode_json(camera_bytes, "teleport_camera_metadata")
        provider_cameras = normalize_teleport_cameras(camera_value)
        cameras_path = native / "cameras_colmap.json"
        cameras_path.write_bytes(camera_bytes)
        ply_temporary = native / ".reconstruction.ply.partial"
        lifecycle.transport.download_file(
            str(metadata["model_url"]),
            destination=ply_temporary,
            maximum_bytes=maximum_ply_bytes,
        )
        ply_path = native / "reconstruction.ply"
        os.replace(ply_temporary, ply_path)
        downloaded_outputs = [
            _provider_output(
                artifact_id="teleport_metadata_v2_colmap_ply",
                artifact_kind="provider_metadata",
                path=metadata_path,
                relative_to=output,
            ),
            _provider_output(
                artifact_id="teleport_camera_metadata_colmap",
                artifact_kind="cameras_metadata",
                path=cameras_path,
                relative_to=output,
            ),
            _provider_output(
                artifact_id="teleport_full_ply",
                artifact_kind="splat_ply",
                path=ply_path,
                relative_to=output,
            ),
        ]
        lifecycle.event("provider_outputs_downloaded_and_hashed", output_count=3)
        execution_receipt = build_reconstruction_provider_execution_receipt(
            {
                "provider_execution_request_digest": request_digest,
                "source_capture_digest": request["source_capture_digest"],
                "train_heldout_split_digest": request["train_heldout_split_digest"],
                "provider_identity": TELEPORT_PROVIDER_IDENTITY,
                "provider_job_identity": str(capture["eid"]),
                "status": "succeeded_unqualified",
                "cost_usd": quoted_cost,
                "duration_seconds": max(0.0, monotonic() - start),
                "attempt_count": 1,
                "provider_runtime_identity": dict(runtime_identity),
                "downloaded_outputs": downloaded_outputs,
                "failure": None,
                "deletion_status": "pending",
                "warnings": ["provider_final_charge_unobservable_via_api"],
                "blockers": [],
                "timestamp": wall_time(),
                "provider_success_is_blueprint_qualification": False,
                "metric_scale_proven": False,
                "collision_geometry_validated": False,
                "isaac_compatibility_proven": False,
                "physical_success_proven": False,
                "deployment_readiness_proven": False,
                "proof_effect": "provider_output_derived_support_only",
                "claim_ceiling": "external_reconstruction_import",
            },
            request=request,
        )
        _write_json(output / "reconstruction_provider_execution_receipt.v1.json", execution_receipt)
        import_request = {
            "schema_version": "provider_splat_import_request.v1",
            "stable_run_identity": request["stable_run_identity"],
            "provider_identity": TELEPORT_PROVIDER_IDENTITY,
            "provider_job_identity": str(capture["eid"]),
            "provider_execution_receipt_digest": execution_receipt[
                "provider_execution_receipt_digest"
            ],
            "source_capture_digest": request["source_capture_digest"],
            "frozen_split_digest": request["train_heldout_split_digest"],
            "consumed_candidate_dataset_digest": packet["candidate_dataset_digest"],
            "source_commit_sha": request["source_commit_sha"],
            "asset_bindings": [
                {
                    "asset_id": "teleport_full_ply",
                    "artifact_kind": "splat_ply",
                    "relative_path": ply_path.relative_to(output).as_posix(),
                    "digest": _sha256_file(ply_path),
                },
                {
                    "asset_id": "teleport_cameras_colmap",
                    "artifact_kind": "cameras_metadata",
                    "relative_path": cameras_path.relative_to(output).as_posix(),
                    "digest": _sha256_file(cameras_path),
                },
            ],
            "provider_had_hidden_access": False,
            "hidden_heldout_pixels_included": False,
            "authority_used": dict(request["authority_used"]),
            "proof_effect": "provider_output_import_request_only",
            "claim_ceiling": "none",
            "timestamp": wall_time(),
        }
        import_receipt = import_provider_splat(
            source_artifact=import_request,
            artifact_root=output,
            output_root=output / "imports",
        )
        alignment = align_provider_reconstruction(
            import_receipt=import_receipt,
            provider_cameras=provider_cameras,
            candidate_observations=candidate_observations,
            image_name_to_observation_id=packet["upload_name_to_observation_id"],
            alignment_thresholds=runtime_identity.get("alignment_thresholds") or {},
            timestamp=wall_time(),
        )
        _write_json(output / "provider_reconstruction_alignment.v1.json", alignment)
        evaluation = dict(
            sealed_evaluation_runner(
                output_root=output,
                provider_native_ply=ply_path,
                import_receipt=import_receipt,
                alignment=alignment,
                execution_request=request,
                packet=packet,
            )
        )
        if evaluation.get("candidate_had_hidden_access") is not False or not _DIGEST.fullmatch(
            str(evaluation.get("visual_heldout_evaluation_report_digest") or "")
        ):
            raise TeleportAdapterError(["teleport_sealed_heldout_evaluation_invalid"])
        lifecycle.event("sealed_heldout_evaluation_completed", status=evaluation.get("status"))
    except TeleportAdapterError as exc:
        failure_codes.extend(exc.codes)
    except ProviderSplatImportError as exc:
        failure_codes.extend(exc.codes)
    except TeleportTransportError as exc:
        failure_codes.append(exc.code)
    except Exception as exc:  # preserve cleanup while returning a typed boundary
        failure_codes.append("teleport_unexpected_adapter_failure:" + type(exc).__name__)
    finally:
        if capture is not None:
            if execution_receipt is None:
                try:
                    execution_receipt = build_reconstruction_provider_execution_receipt(
                        {
                            "provider_execution_request_digest": request_digest,
                            "source_capture_digest": request["source_capture_digest"],
                            "train_heldout_split_digest": request["train_heldout_split_digest"],
                            "provider_identity": TELEPORT_PROVIDER_IDENTITY,
                            "provider_job_identity": str(capture["eid"]),
                            "status": "failed",
                            "cost_usd": quoted_cost,
                            "duration_seconds": min(
                                float(request["ttl_seconds"]), max(0.0, monotonic() - start)
                            ),
                            "attempt_count": 1,
                            "provider_runtime_identity": dict(runtime_identity),
                            "downloaded_outputs": downloaded_outputs,
                            "failure": {
                                "code": failure_codes[0] if failure_codes else "unknown",
                                "retryable": False,
                            },
                            "deletion_status": "pending",
                            "warnings": ["provider_final_charge_unobservable_via_api"],
                            "blockers": sorted(set(failure_codes or ["teleport_run_failed"])),
                            "timestamp": wall_time(),
                            "provider_success_is_blueprint_qualification": False,
                            "metric_scale_proven": False,
                            "collision_geometry_validated": False,
                            "isaac_compatibility_proven": False,
                            "physical_success_proven": False,
                            "deployment_readiness_proven": False,
                            "proof_effect": "provider_output_derived_support_only",
                            "claim_ceiling": "external_reconstruction_import",
                        },
                        request=request,
                    )
                    _write_json(
                        output / "reconstruction_provider_execution_receipt.v1.json",
                        execution_receipt,
                    )
                except Exception:
                    failure_codes.append("teleport_execution_receipt_emission_failed")
            if execution_receipt is not None:
                deletion_receipt = _delete_and_verify(
                    lifecycle,
                    eid=str(capture["eid"]),
                    capture_name=capture_name,
                    execution_receipt=execution_receipt,
                )
                _write_json(
                    output / "reconstruction_provider_deletion_receipt.v1.json",
                    deletion_receipt,
                )
                if deletion_receipt["status"] == "failed":
                    failure_codes.append("teleport_deletion_failed")
    if execution_receipt is None or deletion_receipt is None:
        raise TeleportAdapterError(failure_codes or ["teleport_receipts_incomplete"])
    cost = _cost_receipt(
        request=request,
        quoted_cost_usd=quoted_cost,
        deletion_receipt=deletion_receipt,
        timestamp=wall_time(),
    )
    _write_json(output / "teleport_provider_cost_receipt.v1.json", cost)
    run_receipt = _digest_json(
        {
            "schema_version": TELEPORT_RUN_RECEIPT_SCHEMA,
            "status": "succeeded_unqualified" if not failure_codes else "failed",
            "provider_identity": TELEPORT_PROVIDER_IDENTITY,
            "provider_execution_request_digest": request_digest,
            "provider_execution_receipt_digest": execution_receipt[
                "provider_execution_receipt_digest"
            ],
            "provider_deletion_receipt_digest": deletion_receipt[
                "provider_deletion_receipt_digest"
            ],
            "provider_cost_receipt_digest": cost["teleport_provider_cost_receipt_digest"],
            "provider_splat_import_receipt_digest": (
                import_receipt.get("provider_splat_import_receipt_digest")
                if import_receipt
                else None
            ),
            "provider_reconstruction_alignment_digest": (
                alignment.get("provider_reconstruction_alignment_digest") if alignment else None
            ),
            "visual_heldout_evaluation_report_digest": (
                evaluation.get("visual_heldout_evaluation_report_digest") if evaluation else None
            ),
            "failure_codes": sorted(set(failure_codes)),
            "provider_zero_proven": False,
            "teleport_consumed_rgb_only": True,
            "teleport_consumed_arkit_poses": False,
            "teleport_consumed_depth_or_lidar": False,
            "metric_scale_proven": False,
            "collision_geometry_validated": False,
            "isaac_compatibility_proven": False,
            "task_success_proven": False,
            "physical_truth_proven": False,
            "proof_effect": "provider_appearance_candidate_lifecycle_only",
            "claim_ceiling": "appearance_reconstruction_candidate",
            "timestamp": wall_time(),
        },
        "teleport_provider_run_receipt_digest",
    )
    _write_json(output / "teleport_provider_run_receipt.v1.json", run_receipt)
    if failure_codes:
        raise TeleportAdapterError(failure_codes, result=run_receipt)
    return run_receipt


__all__ = [
    "CLIENT_ID_FILE_ENV",
    "CLIENT_SECRET_FILE_ENV",
    "PUBLIC_SPEND_CAP_ENV",
    "PUBLIC_UPLOAD_AUTH_ENV",
    "TELEPORT_COST_RECEIPT_SCHEMA",
    "TELEPORT_PROVIDER_IDENTITY",
    "TELEPORT_RESOURCE_CLASS",
    "TELEPORT_TERMS_REVIEW_SCHEMA",
    "TeleportAdapterError",
    "TeleportCredentials",
    "TeleportHttpTransport",
    "TeleportToken",
    "TeleportTransportError",
    "load_teleport_credentials",
    "build_teleport_sealed_evaluation_runner",
    "normalize_teleport_cameras",
    "run_teleport_reconstruction",
    "validate_teleport_terms_review",
    "validate_teleport_upload_packet",
    "validate_uploaded_parts",
]
