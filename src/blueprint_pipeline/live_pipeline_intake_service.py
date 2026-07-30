"""Authenticated HTTP facade for Blueprint WebApp-to-Pipeline control traffic.

The service is a thin wrapper around ``build_live_pipeline_input_intake``. It
accepts a WebApp ``robot_eval_job_request.v1`` payload or queue envelope, accepts
job-specific policy packages, real robot POV evidence, deployment outcomes, and live closure evidence,
stages validated files into the configured control-plane paths, accepts
short-lived grants for immutable capture intake, executes only explicitly
authorized hermetic local methods, and optionally runs a configured trigger
command. It does not execute paid providers or promote proof claims.
"""

from __future__ import annotations

import argparse
import fcntl
import hmac
import json
import os
import re
import subprocess
import time
import uuid
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Mapping, Sequence

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .capture_upload_intake import (
    CAPTURE_MALWARE_SCANNER_ARGV_ENV,
    CAPTURE_UPLOAD_ALLOWED_HOSTS_ENV,
    CAPTURE_UPLOAD_STORE_ROOT_ENV,
    CaptureUploadTransferError,
    process_capture_upload_submission,
)
from .capture_qa_webapp_sync import build_capture_qa_webapp_publication
from .capture_lifecycle import (
    CaptureLifecycleError,
    apply_capture_lifecycle_action,
    inspect_capture_lifecycle,
    record_external_revocation_evidence,
    record_provider_deletion_evidence,
)
from .live_pipeline_control_plane import (
    CONTROL_PLANE_OUTPUT_PATH_ENV,
    WEBAPP_JOB_REQUEST_QUEUE_CONTRACT,
    WEBAPP_JOB_REQUEST_SCHEMA_VERSION,
)
from .live_pipeline_input_intake import (
    DECISION_EVIDENCE_QUEUE_CONTRACT,
    build_live_pipeline_input_intake,
    translate_decision_evidence_envelope_to_legacy_execution_request,
)
from .core.security_controls import json_shape_within_limits, strict_identifier
from .task_candidate_control_plane import (
    TaskCandidateControlPlaneError,
    process_task_candidate_decision_submission,
)
from .live_pipeline_reconstruction_testbed_routes import (
    register_reconstruction_testbed_routes,
)
from .task_evaluation_run_control_plane import (
    TaskEvaluationRunControlPlaneError,
    authorize_task_evaluation_run,
    execute_and_aggregate_task_evaluation_run,
    prepare_task_evaluation_run,
)
from .task_evaluation_run_state import (
    TaskEvaluationRunStateError,
    TaskEvaluationRunStateStore,
)
from .task_evaluation_method_catalog import (
    TaskEvaluationMethodCatalogError,
    load_task_evaluation_method_catalog,
)
from .task_evaluation_supervisor import (
    capture_supervisor_execution_options_from_env,
    run_capture_build_supervisor,
)
from .task_evaluation_run_webapp_sync import (
    TASK_EVALUATION_RUN_WEBAPP_SYNC_REQUIRED_ENV,
)


DEFAULT_MANIFEST_PATH = (
    "/var/lib/blueprint/pipeline-control-plane/live_pipeline_control_plane_manifest.json"
)
INTAKE_TOKEN_ENV = "BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN"
INTAKE_ALLOW_LEGACY_BEARER_ENV = "BLUEPRINT_LIVE_PIPELINE_INTAKE_ALLOW_LEGACY_BEARER"
INTAKE_ALLOW_LEGACY_WEBAPP_HMAC_ENV = (
    "BLUEPRINT_LIVE_PIPELINE_ALLOW_LEGACY_WEBAPP_HMAC_WITHOUT_CLIENT_ID"
)
INTAKE_MAX_CLOCK_SKEW_SECONDS_ENV = "BLUEPRINT_LIVE_PIPELINE_INTAKE_MAX_CLOCK_SKEW_SECONDS"
INTAKE_WORK_DIR_ENV = "BLUEPRINT_LIVE_PIPELINE_INTAKE_WORK_DIR"
INTAKE_TRIGGER_ENV = "BLUEPRINT_LIVE_PIPELINE_INTAKE_TRIGGER_COMMAND"
INTAKE_ALLOW_TRIGGER_ENV = "BLUEPRINT_ALLOW_LIVE_PIPELINE_INTAKE_TRIGGER"
INTAKE_OVERWRITE_ENV = "BLUEPRINT_LIVE_PIPELINE_INTAKE_OVERWRITE"
INTAKE_ALLOW_PER_REQUEST_CAPTURE_ROOT_ENV = "BLUEPRINT_LIVE_PIPELINE_ALLOW_PER_REQUEST_CAPTURE_ROOT"
INTAKE_CAPTURE_ROOT_BY_SITE_ENV = "BLUEPRINT_LIVE_PIPELINE_CAPTURE_ROOT_BY_SITE_JSON"
INTAKE_CLIENT_SECRETS_ENV = "BLUEPRINT_LIVE_PIPELINE_CLIENT_SECRETS_JSON"
INTAKE_CLIENT_ROOTS_ENV = "BLUEPRINT_LIVE_PIPELINE_CLIENT_ROOTS_JSON"
INTAKE_NONCE_STORE_DIR_ENV = "BLUEPRINT_LIVE_PIPELINE_NONCE_STORE_DIR"
INTAKE_TRIGGER_SYSTEMD_UNIT_ENV = "BLUEPRINT_LIVE_PIPELINE_TRIGGER_SYSTEMD_UNIT"
INTAKE_MAX_BODY_BYTES_ENV = "BLUEPRINT_LIVE_PIPELINE_MAX_BODY_BYTES"
INTAKE_MAX_JSON_DEPTH_ENV = "BLUEPRINT_LIVE_PIPELINE_MAX_JSON_DEPTH"
INTAKE_MAX_JSON_ITEMS_ENV = "BLUEPRINT_LIVE_PIPELINE_MAX_JSON_ITEMS"
INTAKE_RATE_LIMIT_PER_MINUTE_ENV = "BLUEPRINT_LIVE_PIPELINE_RATE_LIMIT_PER_MINUTE"
INTAKE_MAX_CONCURRENT_ENV = "BLUEPRINT_LIVE_PIPELINE_MAX_CONCURRENT"
INTAKE_MAX_QUEUE_FILES_ENV = "BLUEPRINT_LIVE_PIPELINE_MAX_QUEUE_FILES"
INTAKE_MAX_STORAGE_BYTES_ENV = "BLUEPRINT_LIVE_PIPELINE_MAX_STORAGE_BYTES"
INTAKE_SCHEMA_VERSION = "blueprint_live_pipeline_intake_service.v1"
CAPTURE_HANDOFF_SOURCE_KIND = "capture_pipeline_handoff"
DEFAULT_INTAKE_MAX_CLOCK_SKEW_SECONDS = 5 * 60
DEFAULT_INTAKE_MAX_BODY_BYTES = 2 * 1024 * 1024
DEFAULT_INTAKE_MAX_JSON_DEPTH = 32
DEFAULT_INTAKE_MAX_JSON_ITEMS = 100_000
DEFAULT_INTAKE_RATE_LIMIT_PER_MINUTE = 120
DEFAULT_INTAKE_MAX_CONCURRENT = 8
DEFAULT_INTAKE_MAX_QUEUE_FILES = 10_000
DEFAULT_INTAKE_MAX_STORAGE_BYTES = 20 * 1024 * 1024 * 1024
# Retained as an inert compatibility surface for older tests/importers. Replay
# authority is the shared filesystem store below, never this process-local map.
_INTAKE_NONCE_CACHE: Dict[str, float] = {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _truthy(value: Any) -> bool:
    return _string(value).lower() in {"1", "true", "yes", "on"}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _manifest_path() -> Path:
    return Path(os.getenv(CONTROL_PLANE_OUTPUT_PATH_ENV) or DEFAULT_MANIFEST_PATH).expanduser()


def _work_dir(manifest_path: Path) -> Path:
    configured = _string(os.getenv(INTAKE_WORK_DIR_ENV))
    if configured:
        return Path(configured).expanduser()
    return manifest_path.parent / "incoming_webapp_job_requests"


def _task_candidate_control_plane_root(manifest_path: Path) -> Path:
    return _work_dir(manifest_path).expanduser().resolve() / "task_candidate_control_plane"


def _task_evaluation_run_root(manifest_path: Path) -> Path:
    return _work_dir(manifest_path).expanduser().resolve() / "task_evaluation_runs"


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return stem[:120] or "webapp-job-request"


def _request_from_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("queue_contract") == WEBAPP_JOB_REQUEST_QUEUE_CONTRACT:
        return _mapping(payload.get("job_request"))
    if payload.get("queue_contract") == DECISION_EVIDENCE_QUEUE_CONTRACT:
        return _mapping(payload.get("decision_request"))
    if payload.get("schema_version") == WEBAPP_JOB_REQUEST_SCHEMA_VERSION:
        return payload
    return {}


def _candidate_path(payload: Mapping[str, Any], work_dir: Path) -> Path:
    request = _request_from_payload(payload)
    job_id = _string(request.get("job_id") or payload.get("job_id"))
    digest = sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[
        :12
    ]
    return work_dir / f"{_safe_stem(job_id or digest)}-{digest}.json"


def _capture_handoff_candidate_path(payload: Mapping[str, Any], work_dir: Path) -> Path:
    scene_id = _string(payload.get("scene_id") or payload.get("sceneId"))
    capture_id = _string(payload.get("capture_id") or payload.get("captureId"))
    digest = sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[
        :12
    ]
    stem = "-".join(part for part in (scene_id, capture_id, digest) if part)
    return work_dir / "capture_handoffs" / f"{_safe_stem(stem or digest)}.json"


def _closure_candidate_path(payload: Mapping[str, Any], work_dir: Path) -> Path:
    job_id = _string(
        payload.get("job_id")
        or payload.get("jobId")
        or payload.get("robot_eval_job_id")
        or payload.get("robotEvalJobId")
    )
    digest = sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[
        :12
    ]
    return work_dir / "live_closure_evidence" / f"{_safe_stem(job_id or digest)}-{digest}.json"


def _deployment_outcome_candidate_path(payload: Mapping[str, Any], work_dir: Path) -> Path:
    job_id = _string(
        payload.get("job_id")
        or payload.get("jobId")
        or payload.get("robot_eval_job_id")
        or payload.get("robotEvalJobId")
    )
    digest = sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[
        :12
    ]
    return work_dir / "deployment_outcomes" / f"{_safe_stem(job_id or digest)}-{digest}.json"


def _policy_package_candidate_path(payload: Mapping[str, Any], work_dir: Path) -> Path:
    job_id = _string(
        payload.get("job_id")
        or payload.get("jobId")
        or payload.get("robot_eval_job_id")
        or payload.get("robotEvalJobId")
    )
    digest = sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[
        :12
    ]
    return work_dir / "policy_packages" / f"{_safe_stem(job_id or digest)}-{digest}.json"


def _read_mapping_file(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _first_string(*values: Any) -> str:
    for value in values:
        text = _string(value)
        if text:
            return text
    return ""


def _list_from_payload(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _cards_from_file(path: Path) -> list[Mapping[str, Any]]:
    payload = read_json_any(path)
    if isinstance(payload, Mapping):
        cards = payload.get("cards")
        return [card for card in _list_from_payload(cards) if isinstance(card, Mapping)]
    return [card for card in _list_from_payload(payload) if isinstance(card, Mapping)]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_timestamp(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = _string(value)
    if not text:
        return None
    normalized = text.removesuffix("Z") + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _intake_max_clock_skew_seconds() -> float:
    configured = _string(os.getenv(INTAKE_MAX_CLOCK_SKEW_SECONDS_ENV))
    if not configured:
        return float(DEFAULT_INTAKE_MAX_CLOCK_SKEW_SECONDS)
    try:
        parsed = float(configured)
    except ValueError:
        return float(DEFAULT_INTAKE_MAX_CLOCK_SKEW_SECONDS)
    return parsed if parsed > 0 else float(DEFAULT_INTAKE_MAX_CLOCK_SKEW_SECONDS)


def _strip_sha256_prefix(value: str) -> str:
    return re.sub(r"^sha256=", "", _string(value), flags=re.IGNORECASE)


def _valid_intake_nonce(value: str) -> bool:
    nonce = _string(value)
    return bool(8 <= len(nonce) <= 160 and re.fullmatch(r"[A-Za-z0-9_.:-]+", nonce))


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(_string(os.getenv(name)))
    except ValueError:
        return default
    return value if value > 0 else default


def _client_secrets() -> Dict[str, str]:
    raw = _string(os.getenv(INTAKE_CLIENT_SECRETS_ENV))
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, Mapping):
        return {}
    return {
        strict_identifier(key, field="client_id", max_length=80): _string(value)
        for key, value in payload.items()
        if _string(value)
    }


def _nonce_store_dir() -> Path:
    configured = _string(os.getenv(INTAKE_NONCE_STORE_DIR_ENV))
    root = (
        Path(configured).expanduser().resolve()
        if configured
        else _manifest_path().expanduser().resolve().parent / "intake_nonce_store"
    )
    ensure_dir(root)
    root.chmod(0o700)
    return root


def _claim_intake_nonce(*, client_id: str, nonce: str, now: float, max_age_seconds: float) -> bool:
    """Atomically claim a scoped nonce across processes and restarts."""

    root = _nonce_store_dir()
    digest = sha256(f"{client_id}\0{nonce}".encode("utf-8")).hexdigest()
    path = root / f"{digest}.json"
    expires_at = now + max(max_age_seconds * 2, max_age_seconds + 60)
    payload = (
        json.dumps(
            {
                "schema_version": "blueprint_intake_nonce_claim.v1",
                "client_id_sha256": sha256(client_id.encode("utf-8")).hexdigest(),
                "nonce_sha256": sha256(nonce.encode("utf-8")).hexdigest(),
                "claimed_at_epoch": now,
                "expires_at_epoch": expires_at,
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    for _attempt in range(2):
        try:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            existing = _read_mapping_file(path)
            try:
                existing_expiry = float(existing.get("expires_at_epoch") or 0.0)
            except (TypeError, ValueError):
                existing_expiry = 0.0
            if existing_expiry > now:
                return False
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            continue
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory_descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return True
    return False


def _build_intake_signature(
    *, secret: str, timestamp: str, client_id: str, nonce: str, body: bytes
) -> str:
    canonical = f"{timestamp}.{client_id}.{nonce}.".encode("utf-8") + body
    return hmac.new(secret.encode("utf-8"), canonical, "sha256").hexdigest()


def _verify_intake_signature(
    *,
    secret: str,
    timestamp: str,
    client_id: str,
    nonce: str,
    signature: str,
    body: bytes,
    now: float | None = None,
) -> None:
    parsed_timestamp = _parse_timestamp(timestamp)
    if parsed_timestamp is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid intake signature timestamp",
        )
    max_skew = _intake_max_clock_skew_seconds()
    current = time.time() if now is None else now
    if abs(current - parsed_timestamp) > max_skew:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="intake signature timestamp outside replay window",
        )
    if not _valid_intake_nonce(nonce):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid intake signature nonce",
        )
    expected = _build_intake_signature(
        secret=secret,
        timestamp=_string(timestamp),
        client_id=client_id,
        nonce=_string(nonce),
        body=body,
    )
    provided = _strip_sha256_prefix(signature)
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid intake signature",
        )
    if not _claim_intake_nonce(
        client_id=client_id,
        nonce=_string(nonce),
        now=current,
        max_age_seconds=max_skew,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="replayed intake signature nonce",
        )


def _capture_upload_complete_freshness(capture_root: Path) -> Dict[str, Any]:
    path = capture_root / "raw" / "capture_upload_complete.json"
    if not path.is_file():
        return {"present": False}
    payload = _read_mapping_file(path)
    timestamp = None
    timestamp_source = None
    for key in (
        "capture_upload_completed_at",
        "captureUploadCompletedAt",
        "upload_completed_at",
        "uploadCompletedAt",
        "completed_at",
        "completedAt",
        "uploaded_at",
        "uploadedAt",
        "generated_at",
        "generatedAt",
        "timestamp",
    ):
        timestamp = _parse_timestamp(payload.get(key))
        if timestamp is not None:
            timestamp_source = key
            break
    if timestamp is None:
        timestamp = path.stat().st_mtime
        timestamp_source = "file_mtime"
    return {
        "present": True,
        "path": str(path),
        "sha256": _file_sha256(path),
        "timestamp": timestamp,
        "timestamp_source": timestamp_source,
    }


def _capture_root_ids(capture_root: Path) -> Dict[str, str]:
    descriptor = _read_mapping_file(capture_root / "capture_descriptor.json")
    upload_complete = _read_mapping_file(capture_root / "raw" / "capture_upload_complete.json")
    parts = list(capture_root.parts)
    scene_from_path = ""
    capture_from_path = capture_root.name
    if "scenes" in parts and "captures" in parts:
        scene_index = parts.index("scenes")
        capture_index = parts.index("captures")
        if scene_index + 1 < len(parts):
            scene_from_path = parts[scene_index + 1]
        if capture_index + 1 < len(parts):
            capture_from_path = parts[capture_index + 1]
    return {
        "scene_id": _first_string(
            descriptor.get("scene_id"),
            descriptor.get("sceneId"),
            upload_complete.get("scene_id"),
            upload_complete.get("sceneId"),
            scene_from_path,
        ),
        "capture_id": _first_string(
            descriptor.get("capture_id"),
            descriptor.get("captureId"),
            upload_complete.get("capture_id"),
            upload_complete.get("captureId"),
            capture_from_path,
        ),
    }


def _capture_root_map() -> Dict[str, Path]:
    raw = _string(os.getenv(INTAKE_CAPTURE_ROOT_BY_SITE_ENV))
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, Mapping):
        return {}
    roots: Dict[str, Path] = {}
    for key, value in payload.items():
        text_key = _string(key)
        text_value = _string(value)
        if text_key and text_value:
            roots[text_key] = Path(text_value).expanduser().resolve()
    return roots


def _client_root_map() -> Dict[str, Dict[str, Path]]:
    """Return authenticated-client -> server-owned site/root mappings."""

    raw = _string(os.getenv(INTAKE_CLIENT_ROOTS_ENV))
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, Mapping):
        return {}
    result: Dict[str, Dict[str, Path]] = {}
    for raw_client_id, raw_roots in payload.items():
        try:
            client_id = strict_identifier(
                raw_client_id,
                field="client_id",
                max_length=80,
            )
        except ValueError:
            continue
        if isinstance(raw_roots, str) and _string(raw_roots):
            result[client_id] = {"default": Path(raw_roots).expanduser().resolve()}
            continue
        if not isinstance(raw_roots, Mapping):
            continue
        roots: Dict[str, Path] = {}
        for key, value in raw_roots.items():
            root = _string(value)
            if _string(key) and root:
                roots[_string(key)] = Path(root).expanduser().resolve()
        if roots:
            result[client_id] = roots
    return result


def _request_scope_keys(payload: Mapping[str, Any]) -> list[str]:
    request = _request_from_payload(payload) or payload
    site_package = _mapping(request.get("site_package") or request.get("sitePackage"))
    source = _mapping(request.get("source"))
    selection = _mapping(source.get("selection_state") or source.get("selectionState"))
    values = [
        site_package.get("site_slug") or site_package.get("siteSlug"),
        site_package.get("site_submission_id") or site_package.get("siteSubmissionId"),
        site_package.get("capture_job_id") or site_package.get("captureJobId"),
        site_package.get("capture_id") or site_package.get("captureId"),
        payload.get("site_slug") or payload.get("siteSlug"),
        payload.get("site_submission_id") or payload.get("siteSubmissionId"),
        payload.get("capture_job_id") or payload.get("captureJobId"),
        payload.get("capture_id") or payload.get("captureId"),
        selection.get("site_slug") or selection.get("siteSlug"),
    ]
    return [text for value in values if (text := _string(value))]


def _server_capture_root_for_client(
    *,
    payload: Mapping[str, Any],
    client_id: str,
    manifest_capture_root: Path | None,
) -> Path | None:
    client_roots = _client_root_map()
    if client_roots:
        scoped = client_roots.get(client_id)
        if not scoped:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="authenticated client has no capture-root scope",
            )
        for key in _request_scope_keys(payload):
            if key in scoped:
                return scoped[key]
        if "default" in scoped:
            return scoped["default"]
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="request site is outside authenticated client scope",
        )
    return manifest_capture_root


def _bind_payload_to_server_capture_root(
    *,
    payload: Mapping[str, Any],
    client_id: str,
    manifest_capture_root: Path | None,
) -> Dict[str, Any]:
    """Replace every caller root with the authenticated server-side mapping."""

    bound = json.loads(json.dumps(dict(payload)))
    if bound.get("queue_contract") == DECISION_EVIDENCE_QUEUE_CONTRACT:
        root = _server_capture_root_for_client(
            payload=bound,
            client_id=client_id,
            manifest_capture_root=manifest_capture_root,
        )
        translated = translate_decision_evidence_envelope_to_legacy_execution_request(
            bound,
            expected_capture_root=root,
        )
        if translated is None:
            return bound
        return {
            "queue_contract": WEBAPP_JOB_REQUEST_QUEUE_CONTRACT,
            "source_kind": "decision_evidence_request_legacy_execution_adapter",
            "job_request": translated,
        }
    request = _request_from_payload(bound)
    if not request:
        return bound
    root = _server_capture_root_for_client(
        payload=bound,
        client_id=client_id,
        manifest_capture_root=manifest_capture_root,
    )
    if root is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="server capture-root mapping is not configured",
        )
    request_payload = dict(request)
    request_payload.pop("capture_root", None)
    request_payload.pop("captureRoot", None)
    site_package = dict(
        _mapping(request_payload.get("site_package") or request_payload.get("sitePackage"))
    )
    site_package.pop("captureRoot", None)
    site_package["capture_root"] = str(root)
    site_package["capture_root_source"] = "authenticated_server_mapping"
    request_payload.pop("sitePackage", None)
    request_payload["site_package"] = site_package
    request_payload["authenticated_client_scope"] = {
        "client_id": client_id,
        "server_capture_root_mapped": True,
        "caller_capture_root_authoritative": False,
    }
    if bound.get("queue_contract") == WEBAPP_JOB_REQUEST_QUEUE_CONTRACT:
        bound["job_request"] = request_payload
    else:
        bound = request_payload
    return bound


def _capture_root_from_handoff_payload(
    *,
    payload: Mapping[str, Any],
    client_id: str,
    manifest_capture_root: Path | None,
) -> Path | None:
    scoped = _server_capture_root_for_client(
        payload=payload,
        client_id=client_id,
        manifest_capture_root=manifest_capture_root,
    )
    if scoped is not None:
        return scoped
    roots = _capture_root_map()
    lookup_keys = [
        _first_string(payload.get("site_submission_id"), payload.get("siteSubmissionId")),
        _first_string(payload.get("buyer_request_id"), payload.get("buyerRequestId")),
        _first_string(payload.get("capture_job_id"), payload.get("captureJobId")),
        _first_string(payload.get("scene_id"), payload.get("sceneId")),
        _first_string(payload.get("capture_id"), payload.get("captureId")),
        _first_string(payload.get("site_slug"), payload.get("siteSlug")),
    ]
    for key in lookup_keys:
        if key and key in roots:
            return roots[key]
    return None


def _select_dataset_task(capture_root: Path) -> tuple[Dict[str, Any] | None, list[str]]:
    dataset_dir = capture_root / "pipeline" / "robot_eval_dataset"
    task_cards_path = dataset_dir / "task_cards.json"
    scenario_cards_path = dataset_dir / "scenario_cards.json"
    upload_freshness = _capture_upload_complete_freshness(capture_root)
    blockers: list[str] = []
    if not task_cards_path.is_file():
        blockers.append("robot_eval_task_cards_missing")
    if not scenario_cards_path.is_file():
        blockers.append("robot_eval_scenario_cards_missing")
    if blockers:
        return None, blockers
    task_cards = _cards_from_file(task_cards_path)
    scenario_cards = _cards_from_file(scenario_cards_path)
    if upload_freshness.get("present"):
        upload_timestamp = float(upload_freshness.get("timestamp") or 0.0)
        stale_paths = [
            path.name
            for path in (task_cards_path, scenario_cards_path)
            if path.stat().st_mtime + 0.001 < upload_timestamp
        ]
        if stale_paths:
            blockers.append("robot_eval_dataset_stale_for_capture_upload_complete")
    if not task_cards:
        blockers.append("robot_eval_task_cards_empty")
    if not scenario_cards:
        blockers.append("robot_eval_scenario_cards_empty")
    if blockers:
        return None, blockers
    for task in task_cards:
        task_id = _string(task.get("task_id") or task.get("taskId"))
        if not task_id:
            continue
        scenario = next(
            (
                card
                for card in scenario_cards
                if _string(card.get("task_id") or card.get("taskId")) == task_id
                and _string(card.get("scenario_id") or card.get("scenarioId"))
            ),
            None,
        )
        if scenario is None:
            continue
        return {
            "task_id": task_id,
            "scenario_id": _string(scenario.get("scenario_id") or scenario.get("scenarioId")),
            "task_cards_uri": str(task_cards_path),
            "scenario_cards_uri": str(scenario_cards_path),
            "task_cards_sha256": _file_sha256(task_cards_path),
            "scenario_cards_sha256": _file_sha256(scenario_cards_path),
            "capture_upload_complete": upload_freshness,
            "dataset_fresh_for_capture_upload_complete": not upload_freshness.get("present")
            or "robot_eval_dataset_stale_for_capture_upload_complete" not in blockers,
            "task_card_count": len(task_cards),
            "scenario_card_count": len(scenario_cards),
        }, []
    return None, ["robot_eval_no_task_scenario_pair"]


def _capture_handoff_requests_robot_eval(payload: Mapping[str, Any]) -> bool:
    requested_lanes = {
        _string(item)
        for item in _list_from_payload(
            payload.get("requested_lanes") or payload.get("requestedLanes")
        )
        if _string(item)
    }
    requested_outputs = {
        _string(item)
        for item in _list_from_payload(
            payload.get("requested_outputs") or payload.get("requestedOutputs")
        )
        if _string(item)
    }
    return (
        payload.get("robot_eval_dataset_requested") is True
        or payload.get("robotEvalDatasetRequested") is True
        or "robot_eval_dataset" in requested_lanes
        or "task_evaluation_run" in requested_lanes
        or "robot_eval_dataset" in requested_outputs
        or "task_evaluation_run" in requested_outputs
    )


def _capture_handoff_to_webapp_request(
    *,
    payload: Mapping[str, Any],
    capture_root: Path,
) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
    capture_ids = _capture_root_ids(capture_root)
    handoff_scene_id = _first_string(payload.get("scene_id"), payload.get("sceneId"))
    handoff_capture_id = _first_string(payload.get("capture_id"), payload.get("captureId"))
    site_submission_id = _first_string(
        payload.get("site_submission_id"),
        payload.get("siteSubmissionId"),
    )
    buyer_request_id = _first_string(payload.get("buyer_request_id"), payload.get("buyerRequestId"))
    capture_job_id = _first_string(payload.get("capture_job_id"), payload.get("captureJobId"))
    pipeline_handoff_uri = _first_string(
        payload.get("pipeline_handoff_uri"),
        payload.get("pipelineHandoffUri"),
    )
    capture_descriptor_uri = _first_string(
        payload.get("capture_descriptor_uri"),
        payload.get("captureDescriptorUri"),
    )
    blockers: list[str] = []
    if not _capture_handoff_requests_robot_eval(payload):
        blockers.append("capture_handoff_robot_eval_not_requested")
    if handoff_scene_id and capture_ids["scene_id"] and handoff_scene_id != capture_ids["scene_id"]:
        blockers.append("capture_handoff_scene_id_mismatch")
    if (
        handoff_capture_id
        and capture_ids["capture_id"]
        and handoff_capture_id != capture_ids["capture_id"]
    ):
        blockers.append("capture_handoff_capture_id_mismatch")
    for field, value in (
        ("site_submission_id", site_submission_id),
        ("buyer_request_id", buyer_request_id),
        ("capture_job_id", capture_job_id),
    ):
        if not value:
            blockers.append(f"capture_handoff_missing_{field}")
    dataset_selection, dataset_blockers = _select_dataset_task(capture_root)
    blockers.extend(dataset_blockers)
    if blockers:
        return None, {
            "status": "blocked",
            "ready": False,
            "scene_id": handoff_scene_id or capture_ids["scene_id"],
            "capture_id": handoff_capture_id or capture_ids["capture_id"],
            "blockers": blockers,
        }
    assert dataset_selection is not None
    identity_digest_material = {
        "handoff_payload": dict(payload),
        "dataset_selection": dataset_selection,
    }
    digest = sha256(
        json.dumps(identity_digest_material, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    scene_id = handoff_scene_id or capture_ids["scene_id"]
    capture_id = handoff_capture_id or capture_ids["capture_id"]
    job_id = _safe_stem(f"capture-handoff-{scene_id}-{capture_id}-{digest}")
    request = {
        "schema_version": WEBAPP_JOB_REQUEST_SCHEMA_VERSION,
        "job_id": job_id,
        "request_id": job_id,
        "buyer_request_id": buyer_request_id,
        "site_package": {
            "capture_root": str(capture_root),
            "scene_id": scene_id,
            "capture_id": capture_id,
            "site_submission_id": site_submission_id,
            "buyer_request_id": buyer_request_id,
            "capture_job_id": capture_job_id,
            "pipeline_prefix": str(capture_root / "pipeline"),
            "package_uri": str(
                capture_root
                / "pipeline"
                / "robot_eval_dataset"
                / "robot_eval_dataset_manifest.json"
            ),
            "pipeline_handoff_uri": pipeline_handoff_uri or None,
            "capture_descriptor_uri": capture_descriptor_uri or None,
        },
        "owner_system": {
            "name": "BlueprintCapturePipelineIntake",
            "request_id": job_id,
            "buyer_request_id": buyer_request_id,
            "site_submission_id": site_submission_id,
            "capture_job_id": capture_job_id,
            "capture_id": capture_id,
        },
        "source": {
            "system": "BlueprintCapture",
            "source_kind": CAPTURE_HANDOFF_SOURCE_KIND,
            "pipeline_handoff_uri": pipeline_handoff_uri or None,
            "capture_descriptor_uri": capture_descriptor_uri or None,
            "selection_state": {
                "source_kind": CAPTURE_HANDOFF_SOURCE_KIND,
                "scene_id": scene_id,
                "capture_id": capture_id,
                "site_submission_id": site_submission_id,
                "buyer_request_id": buyer_request_id,
                "capture_job_id": capture_job_id,
                "task_id": dataset_selection["task_id"],
                "scenario_id": dataset_selection["scenario_id"],
                "dataset_selection": dataset_selection,
            },
        },
        "source_kind": CAPTURE_HANDOFF_SOURCE_KIND,
        "requested_tasks": [
            {
                "task_id": dataset_selection["task_id"],
                "scenario_ids": [dataset_selection["scenario_id"]],
            }
        ],
        "robot_profile": {"robot_profile_id": "unitree_g1_humanoid"},
        "simulator_preference": {"framework": "mujoco"},
        "policy_package": {
            "policy_api_endpoint": {},
            "docker_container": {},
            "recorded_action_trace": {},
            "high_level_skill_trace": {
                "ordered_skill_sequence": ["walk_to_target"],
                "skill_taxonomy_version": "blueprint_default_test_policy.v1",
                "source_type": "capture_handoff_default_sim_only_policy",
                "confidence_coverage_note": (
                    "Capture handoff synthesized sim-only beta request; does not prove "
                    "robot-team policy execution."
                ),
            },
            "teleop_demo": {},
            "sim_controller_plugin": {},
        },
        "proof_boundary": {
            "capture_handoff_driven_request": True,
            "simulator_execution_proven": False,
            "robot_policy_execution_proven": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    envelope = {
        "queue_contract": WEBAPP_JOB_REQUEST_QUEUE_CONTRACT,
        "status": "queued_for_pipeline",
        "job_id": job_id,
        "source_kind": CAPTURE_HANDOFF_SOURCE_KIND,
        "capture_handoff": {
            "scene_id": scene_id,
            "capture_id": capture_id,
            "pipeline_handoff_uri": pipeline_handoff_uri or None,
            "capture_descriptor_uri": capture_descriptor_uri or None,
            "robot_eval_dataset_requested": True,
        },
        "job_request": request,
    }
    return envelope, {
        "status": "ready",
        "ready": True,
        "scene_id": scene_id,
        "capture_id": capture_id,
        "job_id": job_id,
        "dataset_selection": dataset_selection,
        "blockers": [],
    }


def _real_robot_pov_candidate_path(payload: Mapping[str, Any], work_dir: Path) -> Path:
    job_id = _string(
        payload.get("job_id")
        or payload.get("jobId")
        or payload.get("robot_eval_job_id")
        or payload.get("robotEvalJobId")
    )
    digest = sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[
        :12
    ]
    return work_dir / "real_robot_pov" / f"{_safe_stem(job_id or digest)}-{digest}.json"


def _redacted_intake_response(
    *,
    candidate_path: Path,
    intake: Mapping[str, Any],
    trigger: Mapping[str, Any],
) -> Dict[str, Any]:
    webapp = _mapping(intake.get("webapp_job_request"))
    staging = _mapping(intake.get("webapp_staging"))
    staged_inputs = _mapping(intake.get("staged_inputs"))
    return {
        "schema_version": INTAKE_SCHEMA_VERSION,
        "status": intake.get("status"),
        "accepted": intake.get("status") == "staged_for_control_plane",
        "generated_at": utc_now_iso(),
        "candidate": {
            "path": str(candidate_path),
            "sha256": webapp.get("sha256"),
        },
        "webapp_job_request": {
            "status": webapp.get("status"),
            "job_id": webapp.get("job_id"),
            "fields_present": webapp.get("fields_present"),
            "missing_fields": webapp.get("missing_fields"),
            "capture_root_matches_control_plane": webapp.get(
                "request_capture_root_matches_control_plane"
            ),
            "blockers": webapp.get("blockers", []),
        },
        "webapp_staging": {
            "status": staging.get("status"),
            "performed": bool(staging.get("performed")),
            "target_path": staging.get("target_path"),
            "blockers": staging.get("blockers", []),
        },
        "staged_inputs": {
            "status": staged_inputs.get("status"),
            "performed": bool(staged_inputs.get("performed")),
            "path": staged_inputs.get("path"),
            "blockers": staged_inputs.get("blockers", []),
        },
        "input_blockers": list(intake.get("input_blockers") or []),
        "trigger": dict(trigger),
        "proof_boundary": {
            "intake_performs_simulator_execution": False,
            "intake_sets_proof_booleans": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _redacted_policy_package_response(
    *,
    candidate_path: Path,
    intake: Mapping[str, Any],
    trigger: Mapping[str, Any],
) -> Dict[str, Any]:
    policy = _mapping(intake.get("policy_package"))
    staging = _mapping(intake.get("policy_package_staging"))
    staged_inputs = _mapping(intake.get("staged_inputs"))
    return {
        "schema_version": INTAKE_SCHEMA_VERSION,
        "status": intake.get("status"),
        "accepted": intake.get("status") == "staged_for_control_plane",
        "generated_at": utc_now_iso(),
        "candidate": {
            "path": str(candidate_path),
            "sha256": policy.get("sha256"),
        },
        "policy_package": {
            "status": policy.get("status"),
            "job_id": policy.get("job_id"),
            "selected_modalities": policy.get("selected_modalities"),
            "blockers": policy.get("blockers", []),
        },
        "policy_package_staging": {
            "status": staging.get("status"),
            "performed": bool(staging.get("performed")),
            "target_path": staging.get("target_path"),
            "blockers": staging.get("blockers", []),
        },
        "staged_inputs": {
            "status": staged_inputs.get("status"),
            "performed": bool(staged_inputs.get("performed")),
            "path": staged_inputs.get("path"),
            "blockers": staged_inputs.get("blockers", []),
        },
        "input_blockers": list(intake.get("input_blockers") or []),
        "trigger": dict(trigger),
        "proof_boundary": {
            "intake_performs_policy_execution": False,
            "intake_sets_proof_booleans": False,
            "robot_policy_execution_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _redacted_real_robot_pov_response(
    *,
    candidate_path: Path,
    intake: Mapping[str, Any],
    trigger: Mapping[str, Any],
) -> Dict[str, Any]:
    pov = _mapping(intake.get("real_robot_pov"))
    staging = _mapping(intake.get("real_robot_pov_staging"))
    staged_inputs = _mapping(intake.get("staged_inputs"))
    return {
        "schema_version": INTAKE_SCHEMA_VERSION,
        "status": intake.get("status"),
        "accepted": intake.get("status") == "staged_for_control_plane",
        "generated_at": utc_now_iso(),
        "candidate": {
            "path": str(candidate_path),
            "sha256": pov.get("sha256"),
        },
        "real_robot_pov": {
            "status": pov.get("status"),
            "job_id": pov.get("job_id"),
            "record_count": pov.get("record_count"),
            "record_ids": pov.get("record_ids"),
            "exact_key_record_count": pov.get("exact_key_record_count"),
            "camera_video_record_count": pov.get("camera_video_record_count"),
            "action_log_record_count": pov.get("action_log_record_count"),
            "timestamp_alignment_record_count": pov.get("timestamp_alignment_record_count"),
            "evidence_record_count": pov.get("evidence_record_count"),
            "missing_exact_key_record_ids": pov.get("missing_exact_key_record_ids"),
            "missing_camera_video_record_ids": pov.get("missing_camera_video_record_ids"),
            "missing_action_log_record_ids": pov.get("missing_action_log_record_ids"),
            "missing_timestamp_alignment_record_ids": pov.get(
                "missing_timestamp_alignment_record_ids"
            ),
            "missing_evidence_record_ids": pov.get("missing_evidence_record_ids"),
            "blockers": pov.get("blockers", []),
        },
        "real_robot_pov_staging": {
            "status": staging.get("status"),
            "performed": bool(staging.get("performed")),
            "target_path": staging.get("target_path"),
            "blockers": staging.get("blockers", []),
        },
        "staged_inputs": {
            "status": staged_inputs.get("status"),
            "performed": bool(staged_inputs.get("performed")),
            "path": staged_inputs.get("path"),
            "blockers": staged_inputs.get("blockers", []),
        },
        "input_blockers": list(intake.get("input_blockers") or []),
        "trigger": dict(trigger),
        "proof_boundary": {
            "intake_performs_robot_execution": False,
            "intake_sets_proof_booleans": False,
            "robot_pov_evidence_proven": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def stage_capture_handoff_for_control_plane(
    *,
    payload: Mapping[str, Any],
    capture_root: str | Path,
    manifest_path: str | Path,
    work_dir: str | Path | None = None,
    overwrite: bool = False,
    staged_inputs_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Convert a capture handoff into a control-plane inbox request.

    This is the non-HTTP form of ``/api/live-pipeline/capture-handoffs`` used by
    the Pub/Sub handoff listener. It stages input pointers only; it does not run
    simulator/provider work or promote proof booleans.
    """

    resolved_manifest_path = Path(manifest_path).expanduser().resolve()
    resolved_capture_root = Path(capture_root).expanduser().resolve()
    resolved_work_dir = (
        Path(work_dir).expanduser().resolve()
        if work_dir
        else _work_dir(resolved_manifest_path).resolve()
    )
    ensure_dir(resolved_work_dir)
    handoff_path = _capture_handoff_candidate_path(payload, resolved_work_dir)
    write_json(handoff_path, dict(payload))
    envelope, handoff_audit = _capture_handoff_to_webapp_request(
        payload=payload,
        capture_root=resolved_capture_root,
    )
    if envelope is None:
        return {
            "schema_version": INTAKE_SCHEMA_VERSION,
            "status": "blocked",
            "accepted": False,
            "generated_at": utc_now_iso(),
            "candidate": {"path": str(handoff_path)},
            "capture_handoff": handoff_audit,
            "input_blockers": [
                f"capture_handoff:{blocker}" for blocker in handoff_audit.get("blockers", [])
            ],
            "proof_boundary": {
                "capture_handoff_converted_to_job_request": False,
                "intake_performs_robot_execution": False,
                "intake_sets_proof_booleans": False,
                "simulator_execution_proven": False,
                "rank_fidelity_result_proven": False,
                "public_claim_upgrade_allowed": False,
            },
        }

    request_path = _candidate_path(envelope, resolved_work_dir)
    write_json(request_path, envelope)
    intake = build_live_pipeline_input_intake(
        manifest_path=resolved_manifest_path,
        webapp_job_request=request_path,
        stage_webapp_request=True,
        overwrite=overwrite,
        allow_request_capture_root=True,
        staged_inputs_path=staged_inputs_path,
    )
    response = _redacted_intake_response(
        candidate_path=request_path,
        intake=intake,
        trigger={
            "status": "not_run",
            "performed": False,
            "reason": "non_http_pubsub_staging_helper",
        },
    )
    response["capture_handoff"] = {
        **handoff_audit,
        "candidate_path": str(handoff_path),
        "webapp_job_request_candidate_path": str(request_path),
        "converted_to_job_request": True,
    }
    response["proof_boundary"] = {
        **_mapping(response.get("proof_boundary")),
        "capture_handoff_converted_to_job_request": True,
        "capture_handoff_endpoint_directly_runs_simulator": False,
        "pubsub_listener_directly_runs_control_plane": False,
    }
    return response


def _redacted_deployment_outcome_response(
    *,
    candidate_path: Path,
    intake: Mapping[str, Any],
    trigger: Mapping[str, Any],
) -> Dict[str, Any]:
    outcomes = _mapping(intake.get("deployment_outcomes"))
    staging = _mapping(intake.get("deployment_outcomes_staging"))
    staged_inputs = _mapping(intake.get("staged_inputs"))
    return {
        "schema_version": INTAKE_SCHEMA_VERSION,
        "status": intake.get("status"),
        "accepted": intake.get("status") == "staged_for_control_plane",
        "generated_at": utc_now_iso(),
        "candidate": {
            "path": str(candidate_path),
            "sha256": outcomes.get("sha256"),
        },
        "deployment_outcomes": {
            "status": outcomes.get("status"),
            "job_id": outcomes.get("job_id"),
            "record_count": outcomes.get("record_count"),
            "record_ids": outcomes.get("record_ids"),
            "owner_evidence_ready": bool(outcomes.get("owner_evidence_ready")),
            "owner_evidence_record_count": outcomes.get("owner_evidence_record_count"),
            "missing_owner_evidence_record_ids": outcomes.get("missing_owner_evidence_record_ids"),
            "blockers": outcomes.get("blockers", []),
        },
        "deployment_outcomes_staging": {
            "status": staging.get("status"),
            "performed": bool(staging.get("performed")),
            "target_path": staging.get("target_path"),
            "blockers": staging.get("blockers", []),
        },
        "staged_inputs": {
            "status": staged_inputs.get("status"),
            "performed": bool(staged_inputs.get("performed")),
            "path": staged_inputs.get("path"),
            "blockers": staged_inputs.get("blockers", []),
        },
        "input_blockers": list(intake.get("input_blockers") or []),
        "trigger": dict(trigger),
        "proof_boundary": {
            "intake_performs_simulator_execution": False,
            "intake_sets_proof_booleans": False,
            "real_world_outcome_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _redacted_closure_evidence_response(
    *,
    candidate_path: Path,
    intake: Mapping[str, Any],
    trigger: Mapping[str, Any],
) -> Dict[str, Any]:
    evidence = _mapping(intake.get("live_closure_evidence"))
    staging = _mapping(intake.get("live_closure_evidence_staging"))
    staged_inputs = _mapping(intake.get("staged_inputs"))
    return {
        "schema_version": INTAKE_SCHEMA_VERSION,
        "status": intake.get("status"),
        "accepted": intake.get("status") == "staged_for_control_plane",
        "generated_at": utc_now_iso(),
        "candidate": {
            "path": str(candidate_path),
            "sha256": evidence.get("sha256"),
        },
        "live_closure_evidence": {
            "status": evidence.get("status"),
            "job_id": evidence.get("job_id"),
            "sections": evidence.get("sections"),
            "blockers": evidence.get("blockers", []),
        },
        "live_closure_evidence_staging": {
            "status": staging.get("status"),
            "performed": bool(staging.get("performed")),
            "target_path": staging.get("target_path"),
            "blockers": staging.get("blockers", []),
        },
        "staged_inputs": {
            "status": staged_inputs.get("status"),
            "performed": bool(staged_inputs.get("performed")),
            "path": staged_inputs.get("path"),
            "blockers": staged_inputs.get("blockers", []),
        },
        "input_blockers": list(intake.get("input_blockers") or []),
        "trigger": dict(trigger),
        "proof_boundary": {
            "intake_performs_simulator_execution": False,
            "intake_sets_proof_booleans": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _trigger_control_plane() -> Dict[str, Any]:
    unit = _string(os.getenv(INTAKE_TRIGGER_SYSTEMD_UNIT_ENV))
    allowed = _truthy(os.getenv(INTAKE_ALLOW_TRIGGER_ENV))
    if not unit:
        return {
            "status": "not_configured",
            "performed": False,
            "allowed": allowed,
            "systemd_unit_configured": False,
        }
    if not allowed:
        return {
            "status": "blocked",
            "performed": False,
            "allowed": False,
            "systemd_unit_configured": True,
            "blockers": [f"missing_env_{INTAKE_ALLOW_TRIGGER_ENV}"],
        }
    if not re.fullmatch(r"[A-Za-z0-9@_.-]+\.service", unit):
        return {
            "status": "blocked",
            "performed": False,
            "allowed": True,
            "systemd_unit_configured": True,
            "blockers": ["intake_trigger_systemd_unit_invalid"],
        }
    command_argv = ["systemctl", "start", "--no-block", unit]
    completed = subprocess.run(
        command_argv,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return {
        "status": "triggered" if completed.returncode == 0 else "failed",
        "performed": completed.returncode == 0,
        "allowed": True,
        "systemd_unit_configured": True,
        "systemd_unit": unit,
        "command_argv_count": len(command_argv),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


async def _require_token(
    request: Request,
    authorization: str | None = Header(default=None),
    x_blueprint_intake_token: str | None = Header(default=None),
    x_blueprint_pipeline_timestamp: str | None = Header(default=None),
    x_blueprint_pipeline_signature: str | None = Header(default=None),
    x_blueprint_pipeline_nonce: str | None = Header(default=None),
    x_blueprint_pipeline_client_id: str | None = Header(default=None),
) -> str:
    shared_secret = _string(os.getenv(INTAKE_TOKEN_ENV))
    client_secrets = _client_secrets()
    if not shared_secret and not client_secrets:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="intake client secrets are not configured",
        )
    timestamp_header = _string(x_blueprint_pipeline_timestamp)
    signature_header = _string(x_blueprint_pipeline_signature)
    nonce_header = _string(x_blueprint_pipeline_nonce)
    if timestamp_header or signature_header or nonce_header:
        if not (timestamp_header and signature_header and nonce_header):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="intake signature requires timestamp, signature, and nonce headers",
            )
        if (
            not _string(x_blueprint_pipeline_client_id)
            and _truthy(os.getenv(INTAKE_ALLOW_LEGACY_WEBAPP_HMAC_ENV))
            and (request.method, request.url.path)
            in {
                ("GET", "/api/live-pipeline/intake-audit"),
                ("POST", "/api/live-pipeline/job-requests"),
            }
            and shared_secret
            and not client_secrets
        ):
            parsed_timestamp = _parse_timestamp(timestamp_header)
            now = time.time()
            if (
                parsed_timestamp is None
                or abs(now - parsed_timestamp) > _intake_max_clock_skew_seconds()
                or not _valid_intake_nonce(nonce_header)
            ):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="invalid legacy intake audit signature metadata",
                )
            body = await request.body()
            expected_legacy = hmac.new(
                shared_secret.encode("utf-8"),
                f"{timestamp_header}.{nonce_header}.".encode("utf-8") + body,
                "sha256",
            ).hexdigest()
            if not hmac.compare_digest(_strip_sha256_prefix(signature_header), expected_legacy):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="invalid legacy intake audit signature",
                )
            legacy_client_id = "legacy-webapp"
            if not _claim_intake_nonce(
                client_id=legacy_client_id,
                nonce=nonce_header,
                now=now,
                max_age_seconds=_intake_max_clock_skew_seconds(),
            ):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="replayed intake signature nonce",
                )
            request.state.intake_client_id = legacy_client_id
            return legacy_client_id
        try:
            client_id = strict_identifier(
                x_blueprint_pipeline_client_id,
                field="client_id",
                max_length=80,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="signed intake requires a valid client identity",
            ) from exc
        if client_secrets and client_id not in client_secrets:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="unknown intake client identity",
            )
        expected = client_secrets.get(client_id) or shared_secret
        if not expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="intake client has no configured signing secret",
            )
        _verify_intake_signature(
            secret=expected,
            timestamp=timestamp_header,
            client_id=client_id,
            nonce=nonce_header,
            signature=signature_header,
            body=await request.body(),
        )
        request.state.intake_client_id = client_id
        return client_id

    if not _truthy(os.getenv(INTAKE_ALLOW_LEGACY_BEARER_ENV)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="intake requires HMAC signature and nonce headers",
        )

    provided = _string(x_blueprint_intake_token)
    if not provided and authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer":
            provided = _string(token)
    if not provided or not shared_secret or not hmac.compare_digest(provided, shared_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid intake token",
        )
    request.state.intake_client_id = "legacy-bearer"
    return "legacy-bearer"


def _intake_storage_usage(root: Path) -> tuple[int, int]:
    file_count = 0
    size_bytes = 0
    if not root.exists():
        return 0, 0
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        file_count += 1
        try:
            size_bytes += path.stat().st_size
        except OSError:
            continue
    return file_count, size_bytes


def _admission_state_paths() -> tuple[Path, Path]:
    root = _work_dir(_manifest_path()).expanduser().resolve() / ".admission"
    ensure_dir(root)
    root.chmod(0o700)
    return root / "state.json", root / "state.lock"


def _claim_intake_admission(client_id: str) -> str:
    state_path, lock_path = _admission_state_paths()
    work_root = _work_dir(_manifest_path()).expanduser().resolve()
    file_count, storage_bytes = _intake_storage_usage(work_root)
    if file_count >= _positive_int_env(INTAKE_MAX_QUEUE_FILES_ENV, DEFAULT_INTAKE_MAX_QUEUE_FILES):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="intake queue file quota exceeded",
        )
    if storage_bytes >= _positive_int_env(
        INTAKE_MAX_STORAGE_BYTES_ENV, DEFAULT_INTAKE_MAX_STORAGE_BYTES
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="intake storage quota exceeded",
        )
    now = time.time()
    lease_id = f"lease-{uuid.uuid4().hex}"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state_payload = _read_mapping_file(state_path)
        rates = {
            str(key): [
                float(item)
                for item in value
                if isinstance(item, (int, float)) and float(item) > now - 60.0
            ]
            for key, value in _mapping(state_payload.get("rate_windows")).items()
            if isinstance(value, list)
        }
        active = {
            str(key): dict(value)
            for key, value in _mapping(state_payload.get("active_leases")).items()
            if isinstance(value, Mapping)
            and float(value.get("started_at_epoch") or 0.0) > now - 600.0
        }
        client_window = rates.setdefault(client_id, [])
        if len(client_window) >= _positive_int_env(
            INTAKE_RATE_LIMIT_PER_MINUTE_ENV,
            DEFAULT_INTAKE_RATE_LIMIT_PER_MINUTE,
        ):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="intake client rate limit exceeded",
            )
        if len(active) >= _positive_int_env(
            INTAKE_MAX_CONCURRENT_ENV,
            DEFAULT_INTAKE_MAX_CONCURRENT,
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="intake concurrency quota exceeded",
            )
        client_window.append(now)
        active[lease_id] = {
            "client_id_sha256": sha256(client_id.encode("utf-8")).hexdigest(),
            "started_at_epoch": now,
        }
        write_json(
            state_path,
            {
                "schema_version": "blueprint_live_intake_admission_state.v1",
                "updated_at": utc_now_iso(),
                "rate_windows": rates,
                "active_leases": active,
            },
        )
    return lease_id


def _release_intake_admission(lease_id: str) -> None:
    state_path, lock_path = _admission_state_paths()
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state_payload = _read_mapping_file(state_path)
        active = _mapping(state_payload.get("active_leases"))
        active.pop(lease_id, None)
        write_json(
            state_path,
            {
                **state_payload,
                "schema_version": "blueprint_live_intake_admission_state.v1",
                "updated_at": utc_now_iso(),
                "active_leases": active,
            },
        )


async def _require_admission(
    request: Request,
    client_id: str = Depends(_require_token),
) -> AsyncIterator[str]:
    body = await request.body()
    if len(body) > _positive_int_env(INTAKE_MAX_BODY_BYTES_ENV, DEFAULT_INTAKE_MAX_BODY_BYTES):
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="intake request body exceeds byte limit",
        )
    if body:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None and not json_shape_within_limits(
            parsed,
            max_depth=_positive_int_env(INTAKE_MAX_JSON_DEPTH_ENV, DEFAULT_INTAKE_MAX_JSON_DEPTH),
            max_items=_positive_int_env(INTAKE_MAX_JSON_ITEMS_ENV, DEFAULT_INTAKE_MAX_JSON_ITEMS),
        ):
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="intake JSON depth or item limit exceeded",
            )
    lease_id = _claim_intake_admission(client_id)
    try:
        yield client_id
    finally:
        _release_intake_admission(lease_id)


def create_app() -> FastAPI:
    app = FastAPI(title="Blueprint Live Pipeline Intake", version=INTAKE_SCHEMA_VERSION)

    @app.get("/health")
    def health() -> Dict[str, Any]:
        manifest_path = _manifest_path()
        return {
            "ok": True,
            "schema_version": INTAKE_SCHEMA_VERSION,
            "control_plane_ready": manifest_path.is_file(),
            "authentication_configured": bool(
                _string(os.getenv(INTAKE_TOKEN_ENV)) or _client_secrets()
            ),
            "signed_intake_required": not _truthy(os.getenv(INTAKE_ALLOW_LEGACY_BEARER_ENV)),
            "legacy_bearer_enabled": _truthy(os.getenv(INTAKE_ALLOW_LEGACY_BEARER_ENV)),
            "shared_nonce_store_enabled": True,
            "server_capture_root_mapping_enforced": True,
            "bounded_admission_enabled": True,
            "capture_upload_transfer_configured": bool(
                _string(os.getenv(CAPTURE_UPLOAD_STORE_ROOT_ENV))
                and _string(os.getenv(CAPTURE_UPLOAD_ALLOWED_HOSTS_ENV))
                and _string(os.getenv(CAPTURE_MALWARE_SCANNER_ARGV_ENV))
            ),
            "proof_boundary": {
                "authorized_hermetic_local_reconstruction_supported": True,
                "paid_or_live_provider_execution_supported": False,
                "simulator_execution_proven": False,
                "rank_fidelity_result_proven": False,
            },
        }

    @app.post(
        "/api/live-pipeline/capture-upload-intakes",
        dependencies=[Depends(_require_admission)],
    )
    async def intake_capture_upload(request: Request) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="expected JSON object")
        try:
            supervisor_options = capture_supervisor_execution_options_from_env()
        except ValueError as exc:
            raise HTTPException(
                status_code=503,
                detail="capture supervisor execution configuration is invalid",
            ) from exc
        store_root_text = _string(os.getenv(CAPTURE_UPLOAD_STORE_ROOT_ENV))
        if not store_root_text:
            raise HTTPException(
                status_code=503,
                detail="capture intake store is not configured",
            )
        store_root = Path(store_root_text).expanduser().resolve()
        try:
            receipt = await run_in_threadpool(
                process_capture_upload_submission,
                payload,
                store_root=store_root,
            )
        except CaptureUploadTransferError as exc:
            blockers = list(exc.blockers)
            if "capture_upload_idempotency_conflict" in blockers:
                response_status = 409
            elif any(
                blocker.endswith("not_configured")
                or blocker == "malware_scanner_configuration_invalid"
                for blocker in blockers
            ):
                response_status = 503
            elif "capture_transfer_download_failed" in blockers:
                response_status = 502
            else:
                response_status = 422
            return JSONResponse(
                status_code=response_status,
                content={
                    "schema_version": "capture_upload_intake_rejection.v1",
                    "status": "rejected",
                    "blockers": blockers,
                    "proof_boundary": {
                        "capture_qa_completed": False,
                        "task_success_established": False,
                        "physical_task_success_established": False,
                        "deployment_or_safety_approved": False,
                        "comparative_policy_ranking_verdict": "thesis_not_supported",
                    },
                },
            )
        artifact_root = (
            store_root / str(_mapping(receipt.get("artifact_reference")).get("uri") or "")
        ).resolve()
        if store_root != artifact_root and store_root not in artifact_root.parents:
            raise HTTPException(
                status_code=500,
                detail="capture intake artifact reference escaped configured store",
            )
        supervisor = await run_in_threadpool(
            run_capture_build_supervisor,
            capture_root=artifact_root / "capture_intake_envelope.json",
            **supervisor_options,
        )
        return {
            "schema_version": "capture_upload_processing_result.v1",
            "receipt": receipt,
            "capture_qa_publication": build_capture_qa_webapp_publication(
                capture_session_id=str(receipt["capture_session_id"]),
                report=_mapping(receipt.get("capture_qa_report")),
            ),
            "task_evaluation_supervisor": supervisor,
        }

    @app.post(
        "/api/live-pipeline/capture-upload-intakes/{capture_session_id}/{intake_id}/lifecycle",
        dependencies=[Depends(_require_admission)],
    )
    async def apply_capture_lifecycle(
        capture_session_id: str, intake_id: str, request: Request
    ) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="expected JSON object")
        if payload.get("schema_version") != "capture_lifecycle_submission.v1":
            raise HTTPException(status_code=422, detail="capture lifecycle schema version mismatch")
        store_root_text = _string(os.getenv(CAPTURE_UPLOAD_STORE_ROOT_ENV))
        if not store_root_text:
            raise HTTPException(status_code=503, detail="capture intake store is not configured")
        manifest_path = _manifest_path().resolve()
        actor_identity = _string(getattr(request.state, "intake_client_id", ""))
        try:
            return await run_in_threadpool(
                apply_capture_lifecycle_action,
                store_root=Path(store_root_text).expanduser().resolve(),
                work_root=_work_dir(manifest_path).expanduser().resolve(),
                capture_session_id=capture_session_id,
                intake_id=intake_id,
                capture_digest=str(payload.get("capture_digest") or ""),
                envelope_digest=str(payload.get("envelope_digest") or ""),
                action=str(payload.get("action") or ""),
                actor={
                    "role": "authenticated_pipeline_client",
                    "identity": actor_identity,
                },
                idempotency_key=str(payload.get("idempotency_key") or ""),
            )
        except CaptureLifecycleError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc

    @app.post(
        "/api/live-pipeline/capture-upload-intakes/{capture_session_id}/{intake_id}/provider-deletion-evidence",
        dependencies=[Depends(_require_admission)],
    )
    async def submit_provider_deletion_evidence(
        capture_session_id: str, intake_id: str, request: Request
    ) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="expected JSON object")
        if payload.get("schema_version") != "capture_provider_deletion_evidence_submission.v1":
            raise HTTPException(
                status_code=422, detail="provider deletion evidence schema version mismatch"
            )
        store_root_text = _string(os.getenv(CAPTURE_UPLOAD_STORE_ROOT_ENV))
        if not store_root_text:
            raise HTTPException(status_code=503, detail="capture intake store is not configured")
        try:
            return record_provider_deletion_evidence(
                store_root=Path(store_root_text).expanduser().resolve(),
                capture_session_id=capture_session_id,
                intake_id=intake_id,
                obligation_digest=str(payload.get("obligation_digest") or ""),
                deletion_receipt_digest=str(payload.get("deletion_receipt_digest") or ""),
                provider_identity=str(payload.get("provider_identity") or ""),
                deleted_at=str(payload.get("deleted_at") or ""),
                verification_method=str(payload.get("verification_method") or ""),
                idempotency_key=str(payload.get("idempotency_key") or ""),
            )
        except CaptureLifecycleError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc

    @app.get(
        "/api/live-pipeline/capture-upload-intakes/{capture_session_id}/{intake_id}/lifecycle",
        dependencies=[Depends(_require_admission)],
    )
    async def inspect_completed_capture_lifecycle(
        capture_session_id: str, intake_id: str
    ) -> Dict[str, Any]:
        store_root_text = _string(os.getenv(CAPTURE_UPLOAD_STORE_ROOT_ENV))
        if not store_root_text:
            raise HTTPException(status_code=503, detail="capture intake store is not configured")
        try:
            return inspect_capture_lifecycle(
                store_root=Path(store_root_text).expanduser().resolve(),
                capture_session_id=capture_session_id,
                intake_id=intake_id,
            )
        except CaptureLifecycleError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc

    @app.post(
        "/api/live-pipeline/capture-upload-intakes/{capture_session_id}/{intake_id}/external-revocation-evidence",
        dependencies=[Depends(_require_admission)],
    )
    async def submit_external_revocation_evidence(
        capture_session_id: str, intake_id: str, request: Request
    ) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="expected JSON object")
        if payload.get("schema_version") != "capture_external_revocation_evidence_submission.v1":
            raise HTTPException(
                status_code=422, detail="external revocation evidence schema version mismatch"
            )
        store_root_text = _string(os.getenv(CAPTURE_UPLOAD_STORE_ROOT_ENV))
        if not store_root_text:
            raise HTTPException(status_code=503, detail="capture intake store is not configured")
        try:
            return record_external_revocation_evidence(
                store_root=Path(store_root_text).expanduser().resolve(),
                capture_session_id=capture_session_id,
                intake_id=intake_id,
                action=str(payload.get("action") or ""),
                target_system=str(payload.get("target_system") or ""),
                receipt_digest=str(payload.get("receipt_digest") or ""),
                completed_at=str(payload.get("completed_at") or ""),
                verification_method=str(payload.get("verification_method") or ""),
                idempotency_key=str(payload.get("idempotency_key") or ""),
            )
        except CaptureLifecycleError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc

    register_reconstruction_testbed_routes(
        app,
        require_admission=_require_admission,
        manifest_path_provider=_manifest_path,
        work_dir_provider=_work_dir,
    )

    @app.post("/api/live-pipeline/job-requests", dependencies=[Depends(_require_admission)])
    async def intake_job_request(request: Request) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="expected JSON object")
        manifest_path = _manifest_path().resolve()
        if not manifest_path.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"control-plane manifest missing: {manifest_path}",
            )
        manifest_payload = read_json_any(manifest_path)
        if not isinstance(manifest_payload, Mapping):
            raise HTTPException(
                status_code=503,
                detail="control-plane manifest is not JSON object",
            )
        manifest_capture_root_text = _string(manifest_payload.get("capture_root"))
        manifest_capture_root = (
            Path(manifest_capture_root_text).resolve() if manifest_capture_root_text else None
        )
        payload = _bind_payload_to_server_capture_root(
            payload=payload,
            client_id=_string(getattr(request.state, "intake_client_id", "")),
            manifest_capture_root=manifest_capture_root,
        )
        work_dir = _work_dir(manifest_path).resolve()
        ensure_dir(work_dir)
        candidate_path = _candidate_path(payload, work_dir)
        write_json(candidate_path, dict(payload))
        intake = build_live_pipeline_input_intake(
            manifest_path=manifest_path,
            webapp_job_request=candidate_path,
            stage_webapp_request=True,
            overwrite=_truthy(os.getenv(INTAKE_OVERWRITE_ENV)),
            allow_request_capture_root=True,
        )
        trigger = (
            _trigger_control_plane()
            if intake.get("status") == "staged_for_control_plane"
            else {
                "status": "not_run",
                "performed": False,
                "reason": "intake_not_staged_for_control_plane",
            }
        )
        response = _redacted_intake_response(
            candidate_path=candidate_path,
            intake=intake,
            trigger=trigger,
        )
        if intake.get("input_blockers"):
            return JSONResponse(status_code=202, content=response)
        return response

    @app.post(
        "/api/live-pipeline/task-decisions",
        dependencies=[Depends(_require_admission)],
    )
    async def intake_task_decision(request: Request) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="expected JSON object")
        manifest_path = _manifest_path().resolve()
        if not manifest_path.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"control-plane manifest missing: {manifest_path}",
            )
        try:
            return process_task_candidate_decision_submission(
                state_root=_task_candidate_control_plane_root(manifest_path),
                submission=payload,
            )
        except TaskCandidateControlPlaneError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc

    @app.post(
        "/api/live-pipeline/task-evaluation-runs/plan",
        dependencies=[Depends(_require_admission)],
    )
    async def plan_task_evaluation_run(request: Request) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="expected JSON object")
        plan_schema = payload.get("schema_version")
        if plan_schema not in {
            "task_evaluation_run_plan_submission.v1",
            "task_evaluation_run_plan_submission.v2",
        }:
            raise HTTPException(status_code=422, detail="run plan schema version mismatch")
        manifest_path = _manifest_path().resolve()
        if not manifest_path.is_file():
            raise HTTPException(status_code=503, detail="control-plane manifest missing")
        catalog = None
        if plan_schema == "task_evaluation_run_plan_submission.v2":
            if "method_profiles" in payload or "qualifications" in payload:
                raise HTTPException(
                    status_code=422,
                    detail="v2 plan submission cannot select methods or qualifications",
                )
            try:
                catalog = load_task_evaluation_method_catalog()
            except TaskEvaluationMethodCatalogError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            methods = catalog["method_profiles"]
            qualifications = catalog["qualifications"]
        else:
            methods = payload.get("method_profiles")
            qualifications = payload.get("qualifications")
            if not isinstance(methods, list) or not isinstance(qualifications, list):
                raise HTTPException(
                    status_code=422,
                    detail="method profiles and qualifications must be lists",
                )
        try:
            return prepare_task_evaluation_run(
                state_root=_task_evaluation_run_root(manifest_path),
                run_id=str(payload.get("run_id") or ""),
                capture_session_id=str(payload.get("capture_session_id") or ""),
                intake_id=str(payload.get("intake_id") or ""),
                request_value=_mapping(payload.get("decision_evidence_request")),
                testbed_value=_mapping(payload.get("testbed")),
                method_values=[dict(row) for row in methods if isinstance(row, Mapping)],
                qualification_values=[
                    dict(row) for row in qualifications if isinstance(row, Mapping)
                ],
                idempotency_key=str(payload.get("idempotency_key") or ""),
                method_catalog_value=catalog,
            )
        except (TaskEvaluationRunControlPlaneError, TaskEvaluationRunStateError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post(
        "/api/live-pipeline/task-evaluation-runs/{run_id}/authorize",
        dependencies=[Depends(_require_admission)],
    )
    async def authorize_task_evaluation_run_execution(
        run_id: str, request: Request
    ) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="expected JSON object")
        if payload.get("schema_version") != "task_evaluation_run_authorization_submission.v1":
            raise HTTPException(status_code=422, detail="run authorization schema version mismatch")
        manifest_path = _manifest_path().resolve()
        references = payload.get("authorized_adapter_references")
        actor = payload.get("actor")
        if not isinstance(references, list) or not isinstance(actor, Mapping):
            raise HTTPException(status_code=422, detail="authorization references or actor invalid")
        try:
            return authorize_task_evaluation_run(
                state_root=_task_evaluation_run_root(manifest_path),
                run_id=run_id,
                plan_digest=str(payload.get("plan_digest") or ""),
                authorized_adapter_references=[str(row) for row in references],
                actor=dict(actor),
                idempotency_key=str(payload.get("idempotency_key") or ""),
            )
        except (TaskEvaluationRunControlPlaneError, TaskEvaluationRunStateError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post(
        "/api/live-pipeline/task-evaluation-runs/{run_id}/execute",
        dependencies=[Depends(_require_admission)],
    )
    async def execute_task_evaluation_run(run_id: str) -> Dict[str, Any]:
        manifest_path = _manifest_path().resolve()
        try:
            result = execute_and_aggregate_task_evaluation_run(
                state_root=_task_evaluation_run_root(manifest_path),
                run_id=run_id,
            )
        except (TaskEvaluationRunControlPlaneError, TaskEvaluationRunStateError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        webapp_sync = _mapping(result.get("webapp_sync"))
        if (
            _truthy(os.getenv(TASK_EVALUATION_RUN_WEBAPP_SYNC_REQUIRED_ENV))
            and webapp_sync.get("status") != "succeeded"
        ):
            raise HTTPException(
                status_code=502,
                detail=(
                    "task_evaluation_run_webapp_sync_required:"
                    f"{webapp_sync.get('reason') or webapp_sync.get('status')}"
                ),
            )
        return result

    @app.get(
        "/api/live-pipeline/task-evaluation-runs/{run_id}",
        dependencies=[Depends(_require_admission)],
    )
    async def inspect_task_evaluation_run(run_id: str) -> Dict[str, Any]:
        manifest_path = _manifest_path().resolve()
        try:
            state = TaskEvaluationRunStateStore(_task_evaluation_run_root(manifest_path)).inspect(
                run_id
            )
        except TaskEvaluationRunStateError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "schema_version": "task_evaluation_run_inspection.v1",
            **state,
        }

    @app.post("/api/live-pipeline/capture-handoffs", dependencies=[Depends(_require_admission)])
    async def intake_capture_handoff(request: Request) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="expected JSON object")
        manifest_path = _manifest_path().resolve()
        if not manifest_path.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"control-plane manifest missing: {manifest_path}",
            )
        manifest_payload = read_json_any(manifest_path)
        if not isinstance(manifest_payload, Mapping):
            raise HTTPException(status_code=503, detail="control-plane manifest is not JSON object")
        manifest_capture_root_text = _string(manifest_payload.get("capture_root"))
        manifest_capture_root = (
            Path(manifest_capture_root_text).resolve() if manifest_capture_root_text else None
        )
        capture_root = _capture_root_from_handoff_payload(
            payload=payload,
            client_id=_string(getattr(request.state, "intake_client_id", "")),
            manifest_capture_root=manifest_capture_root,
        )
        if capture_root is None:
            raise HTTPException(status_code=503, detail="control-plane capture_root missing")
        if not capture_root.is_dir():
            raise HTTPException(
                status_code=503,
                detail=f"capture_root missing for handoff: {capture_root}",
            )
        work_dir = _work_dir(manifest_path).resolve()
        ensure_dir(work_dir)
        handoff_path = _capture_handoff_candidate_path(payload, work_dir)
        write_json(handoff_path, dict(payload))
        envelope, handoff_audit = _capture_handoff_to_webapp_request(
            payload=payload,
            capture_root=capture_root,
        )
        if envelope is None:
            return JSONResponse(
                status_code=202,
                content={
                    "schema_version": INTAKE_SCHEMA_VERSION,
                    "status": "blocked",
                    "accepted": False,
                    "generated_at": utc_now_iso(),
                    "candidate": {"path": str(handoff_path)},
                    "capture_handoff": handoff_audit,
                    "input_blockers": [
                        f"capture_handoff:{blocker}"
                        for blocker in handoff_audit.get("blockers", [])
                    ],
                    "trigger": {
                        "status": "not_run",
                        "performed": False,
                        "reason": "capture_handoff_not_ready",
                    },
                    "proof_boundary": {
                        "capture_handoff_converted_to_job_request": False,
                        "intake_performs_robot_execution": False,
                        "intake_sets_proof_booleans": False,
                        "simulator_execution_proven": False,
                        "rank_fidelity_result_proven": False,
                        "public_claim_upgrade_allowed": False,
                    },
                },
            )
        request_path = _candidate_path(envelope, work_dir)
        write_json(request_path, envelope)
        intake = build_live_pipeline_input_intake(
            manifest_path=manifest_path,
            webapp_job_request=request_path,
            stage_webapp_request=True,
            overwrite=_truthy(os.getenv(INTAKE_OVERWRITE_ENV)),
            allow_request_capture_root=True,
        )
        trigger = (
            _trigger_control_plane()
            if intake.get("status") == "staged_for_control_plane"
            else {
                "status": "not_run",
                "performed": False,
                "reason": "intake_not_staged_for_control_plane",
            }
        )
        response = _redacted_intake_response(
            candidate_path=request_path,
            intake=intake,
            trigger=trigger,
        )
        response["capture_handoff"] = {
            **handoff_audit,
            "candidate_path": str(handoff_path),
            "webapp_job_request_candidate_path": str(request_path),
            "converted_to_job_request": True,
        }
        response["proof_boundary"] = {
            **_mapping(response.get("proof_boundary")),
            "capture_handoff_converted_to_job_request": True,
            "capture_handoff_endpoint_directly_runs_simulator": False,
        }
        if intake.get("input_blockers"):
            return JSONResponse(status_code=202, content=response)
        return response

    @app.post(
        "/api/live-pipeline/policy-packages",
        dependencies=[Depends(_require_admission)],
    )
    async def intake_policy_package(request: Request) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="expected JSON object")
        manifest_path = _manifest_path().resolve()
        if not manifest_path.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"control-plane manifest missing: {manifest_path}",
            )
        work_dir = _work_dir(manifest_path).resolve()
        ensure_dir(work_dir)
        candidate_path = _policy_package_candidate_path(payload, work_dir)
        write_json(candidate_path, dict(payload))
        intake = build_live_pipeline_input_intake(
            manifest_path=manifest_path,
            policy_package=candidate_path,
            stage_policy_package=True,
            overwrite=_truthy(os.getenv(INTAKE_OVERWRITE_ENV)),
        )
        trigger = (
            _trigger_control_plane()
            if intake.get("status") == "staged_for_control_plane"
            else {
                "status": "not_run",
                "performed": False,
                "reason": "intake_not_staged_for_control_plane",
            }
        )
        response = _redacted_policy_package_response(
            candidate_path=candidate_path,
            intake=intake,
            trigger=trigger,
        )
        if intake.get("input_blockers"):
            return JSONResponse(status_code=202, content=response)
        return response

    @app.post(
        "/api/live-pipeline/real-robot-pov",
        dependencies=[Depends(_require_admission)],
    )
    async def intake_real_robot_pov(request: Request) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="expected JSON object")
        manifest_path = _manifest_path().resolve()
        if not manifest_path.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"control-plane manifest missing: {manifest_path}",
            )
        work_dir = _work_dir(manifest_path).resolve()
        ensure_dir(work_dir)
        candidate_path = _real_robot_pov_candidate_path(payload, work_dir)
        write_json(candidate_path, dict(payload))
        intake = build_live_pipeline_input_intake(
            manifest_path=manifest_path,
            real_robot_pov=candidate_path,
            stage_real_robot_pov=True,
            overwrite=_truthy(os.getenv(INTAKE_OVERWRITE_ENV)),
        )
        trigger = (
            _trigger_control_plane()
            if intake.get("status") == "staged_for_control_plane"
            else {
                "status": "not_run",
                "performed": False,
                "reason": "intake_not_staged_for_control_plane",
            }
        )
        response = _redacted_real_robot_pov_response(
            candidate_path=candidate_path,
            intake=intake,
            trigger=trigger,
        )
        if intake.get("input_blockers"):
            return JSONResponse(status_code=202, content=response)
        return response

    @app.post(
        "/api/live-pipeline/deployment-outcomes",
        dependencies=[Depends(_require_admission)],
    )
    async def intake_deployment_outcomes(request: Request) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="expected JSON object")
        manifest_path = _manifest_path().resolve()
        if not manifest_path.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"control-plane manifest missing: {manifest_path}",
            )
        work_dir = _work_dir(manifest_path).resolve()
        ensure_dir(work_dir)
        candidate_path = _deployment_outcome_candidate_path(payload, work_dir)
        write_json(candidate_path, dict(payload))
        intake = build_live_pipeline_input_intake(
            manifest_path=manifest_path,
            deployment_outcomes=candidate_path,
            stage_deployment_outcomes=True,
            overwrite=_truthy(os.getenv(INTAKE_OVERWRITE_ENV)),
        )
        trigger = (
            _trigger_control_plane()
            if intake.get("status") == "staged_for_control_plane"
            else {
                "status": "not_run",
                "performed": False,
                "reason": "intake_not_staged_for_control_plane",
            }
        )
        response = _redacted_deployment_outcome_response(
            candidate_path=candidate_path,
            intake=intake,
            trigger=trigger,
        )
        if intake.get("input_blockers"):
            return JSONResponse(status_code=202, content=response)
        return response

    @app.post(
        "/api/live-pipeline/live-closure-evidence",
        dependencies=[Depends(_require_admission)],
    )
    async def intake_live_closure_evidence(request: Request) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="expected JSON object")
        manifest_path = _manifest_path().resolve()
        if not manifest_path.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"control-plane manifest missing: {manifest_path}",
            )
        work_dir = _work_dir(manifest_path).resolve()
        ensure_dir(work_dir)
        candidate_path = _closure_candidate_path(payload, work_dir)
        write_json(candidate_path, dict(payload))
        intake = build_live_pipeline_input_intake(
            manifest_path=manifest_path,
            live_closure_evidence=candidate_path,
            stage_live_closure_evidence=True,
            overwrite=_truthy(os.getenv(INTAKE_OVERWRITE_ENV)),
        )
        trigger = (
            _trigger_control_plane()
            if intake.get("status") == "staged_for_control_plane"
            else {
                "status": "not_run",
                "performed": False,
                "reason": "intake_not_staged_for_control_plane",
            }
        )
        response = _redacted_closure_evidence_response(
            candidate_path=candidate_path,
            intake=intake,
            trigger=trigger,
        )
        if intake.get("input_blockers"):
            return JSONResponse(status_code=202, content=response)
        return response

    @app.get("/api/live-pipeline/intake-audit", dependencies=[Depends(_require_token)])
    def latest_intake_audit() -> Dict[str, Any]:
        manifest_path = _manifest_path().resolve()
        audit_path = manifest_path.parent / "live_pipeline_input_intake_audit.json"
        if not audit_path.is_file():
            raise HTTPException(status_code=404, detail="intake audit not found")
        payload = read_json_any(audit_path)
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=500, detail="intake audit is not a JSON object")
        return {
            "schema_version": INTAKE_SCHEMA_VERSION,
            "audit_path": str(audit_path),
            "status": payload.get("status"),
            "input_blockers": list(payload.get("input_blockers") or []),
            "webapp_job_request": _mapping(payload.get("webapp_job_request")),
            "webapp_staging": _mapping(payload.get("webapp_staging")),
            "policy_package": _mapping(payload.get("policy_package")),
            "policy_package_staging": _mapping(payload.get("policy_package_staging")),
            "real_robot_pov": _mapping(payload.get("real_robot_pov")),
            "real_robot_pov_staging": _mapping(payload.get("real_robot_pov_staging")),
            "deployment_outcomes": _mapping(payload.get("deployment_outcomes")),
            "deployment_outcomes_staging": _mapping(payload.get("deployment_outcomes_staging")),
            "live_closure_evidence": _mapping(payload.get("live_closure_evidence")),
            "live_closure_evidence_staging": _mapping(payload.get("live_closure_evidence_staging")),
            "staged_inputs": _mapping(payload.get("staged_inputs")),
            "proof_boundary": payload.get("proof_boundary"),
        }

    return app


app = create_app()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the live Pipeline intake HTTP service.")
    parser.add_argument(
        "--host", default=os.getenv("BLUEPRINT_LIVE_PIPELINE_INTAKE_HOST", "127.0.0.1")
    )
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8765")))
    args = parser.parse_args(argv)
    import uvicorn

    uvicorn.run(
        "blueprint_pipeline.live_pipeline_intake_service:app", host=args.host, port=args.port
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
