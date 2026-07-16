"""Consent-revocation takedown propagation across already-delivered artifacts.

The build-time consent gate blocks NEW exports, but rights are authoritative
continuously: a revocation that lands after delivery must recall what already
shipped. This module provides the recall path:

- ``propagate_consent_takedown(capture_root)`` enumerates every downstream
  artifact whose lineage traces to the capture — including derived-of-derived
  chains that carry no capture ids of their own — and emits a fail-closed
  ``takedown_manifest.v1`` with a tombstone per artifact plus the explicit
  webapp revocation verdict.
- ``evaluate_delivery_time_takedown_gate`` is the serve/sync-time check: it
  re-reads consent live (not just the manifest), so a revocation blocks
  delivery even before propagation has run.
- ``sync_webapp_consent_revocation`` pushes the revoked VERDICT (never mere
  absence) over the same signed channel as webapp sync, and never claims
  execution it did not perform.

Enumeration is intentionally over-inclusive: for a takedown, listing an extra
artifact is safe while missing one is the legal risk.
"""

from __future__ import annotations

import json
import shutil
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from .common import utc_now_iso, write_json
from .consent_normalization import resolve_consent_signals
from .webapp_sync import _int_env, _pipeline_sync_headers, _string_env

TAKEDOWN_MANIFEST_SCHEMA_VERSION = "takedown_manifest.v1"
WEBAPP_REVOCATION_SIGNAL_SCHEMA_VERSION = "webapp_consent_revocation_signal.v1"
DELIVERY_GATE_SCHEMA_VERSION = "consent_takedown_delivery_gate.v1"
RECALL_EXECUTION_SCHEMA_VERSION = "takedown_recall_execution.v1"
RECALL_MARKER_SCHEMA_VERSION = "artifact_recall_marker.v1"
TAKEDOWN_DRILL_SCHEMA_VERSION = "takedown_recall_drill.v1"

# All takedown outputs (manifest, per-artifact recall markers, quarantine, the
# executed-recall audit record) live under this prefix so re-running enumeration
# never treats a takedown output as a new derived artifact to recall.
CONSENT_TAKEDOWN_DIR_PREFIX = "pipeline/consent_takedown/"
TAKEDOWN_MANIFEST_RELATIVE_PATH = "pipeline/consent_takedown/takedown_manifest.json"
RECALL_EXECUTION_RECORD_RELATIVE_PATH = (
    "pipeline/consent_takedown/recall_execution_record.json"
)
QUARANTINE_RELATIVE_DIR = "pipeline/consent_takedown/quarantine"
# Sidecar recall marker suffix. Markers are excluded from enumeration everywhere.
RECALL_MARKER_SUFFIX = ".recall.json"

# Recall modes for the pipeline-owned recall of DERIVED deliverables.
RECALL_MODE_QUARANTINE = "quarantine"
RECALL_MODE_DELETE = "delete"
RECALL_MODE_MARK = "mark"
_RECALL_MODES = frozenset({RECALL_MODE_QUARANTINE, RECALL_MODE_DELETE, RECALL_MODE_MARK})

# Per-target recall outcomes that count as a terminal recalled/quarantined state.
_TERMINAL_RECALLED_OUTCOMES = frozenset(
    {"quarantined", "deleted", "recalled_marked", "recalled_absent"}
)

# Consent source priority mirrors post_training_data_package._consent_source_payload
# so build-time and takedown-time reads can never disagree about the source of truth.
_CONSENT_SOURCE_RELATIVES = (
    "raw/rights_consent.json",
    "rights_consent.json",
    "raw/manifest.json",
    "capture_descriptor.json",
)

_REQUIRED_TAKEDOWN_ACTIONS = (
    "block_new_package_exports",
    "refuse_delivery_and_hosting_of_tombstoned_artifacts",
    "disable_signed_delivery_access",
    "remove_hosted_review_assets",
    "remove_or_expire_hosted_sessions",
    "sync_webapp_revocation_verdict",
    "stop_downstream_training_or_finetuning_use",
    "notify_buyer_and_owner",
)

# File suffixes that can carry lineage references worth parsing.
_PARSEABLE_SUFFIXES = {".json"}

# Payload keys that identify the capture directly.
_CAPTURE_ID_KEYS = ("capture_id", "captureId", "source_capture_id", "sourceCaptureId")
_SCENE_ID_KEYS = ("scene_id", "sceneId")


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha256_file(path: Path) -> str | None:
    try:
        digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _identity_from_path(capture_root: Path) -> tuple[str, str]:
    parts = capture_root.parts
    scene_id = ""
    capture_id = ""
    if "scenes" in parts:
        index = parts.index("scenes")
        if index + 1 < len(parts):
            scene_id = parts[index + 1]
    if "captures" in parts:
        index = parts.index("captures")
        if index + 1 < len(parts):
            capture_id = parts[index + 1]
    return scene_id, capture_id


def read_consent_state(capture_root: Path) -> Dict[str, Any]:
    """Read the authoritative consent state for one capture root.

    Fail-closed contract: ``state`` is ``"active"`` only when a consent source
    exists AND it carries an explicitly active consent status AND nothing in it
    indicates revocation. Unknown, wrong-typed, or contradictory statuses are
    ``"unknown"`` (blocked); missing sources are ``"unknown"``.
    """
    root = Path(capture_root).expanduser()
    for relative in _CONSENT_SOURCE_RELATIVES:
        payload = _read_json(root / relative)
        if not payload:
            continue
        signals = resolve_consent_signals(payload)
        # raw/manifest.json and capture_descriptor.json exist for every capture;
        # if they carry no consent fields they are not a consent source. A
        # dedicated rights_consent.json without consent fields IS a consent
        # source — one that cannot verify consent, so it stays "unknown".
        if not signals["has_consent_fields"] and relative in (
            "raw/manifest.json",
            "capture_descriptor.json",
        ):
            continue
        return {
            "state": signals["state"],
            "consent_revoked": signals["consent_revoked"],
            "consent_status": signals["consent_status"],
            "consent_revoked_at": signals["consent_revoked_at"],
            "malformed_consent_fields": signals["malformed_fields"],
            "source_path": str(root / relative),
        }
    return {
        "state": "unknown",
        "consent_revoked": False,
        "consent_status": None,
        "consent_revoked_at": None,
        "malformed_consent_fields": [],
        "source_path": None,
    }


def _iter_files(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not path.name.endswith(RECALL_MARKER_SUFFIX)
    )


def _reference_strings(value: Any, out: List[str]) -> None:
    if isinstance(value, str):
        text = value.strip()
        if text:
            out.append(text)
    elif isinstance(value, Mapping):
        for item in value.values():
            _reference_strings(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reference_strings(item, out)


def _resolve_reference(text: str, *, base_dir: Path, roots: Sequence[Path]) -> Path | None:
    candidate = text
    if candidate.startswith("file://"):
        candidate = candidate[7:]
    elif "://" in candidate:
        return None
    try:
        path = Path(candidate).expanduser()
    except (OSError, ValueError):
        return None
    if path.is_absolute():
        return path if path.is_file() else None
    for base in (base_dir, *roots):
        resolved = base / path
        if resolved.is_file():
            return resolved
    return None


def _payload_names_capture(payload: Mapping[str, Any], *, capture_id: str, scene_id: str) -> bool:
    if capture_id and any(
        _string(payload.get(key)) == capture_id for key in _CAPTURE_ID_KEYS
    ):
        return True
    if scene_id and capture_id:
        # A scene id alone is not lineage; require the capture id somewhere in
        # the serialized payload (covers nested structures).
        serialized = json.dumps(payload)
        return capture_id in serialized
    return False


def enumerate_derived_artifacts(
    *,
    capture_root: Path,
    capture_id: str,
    scene_id: str,
    additional_artifact_roots: Sequence[Path] = (),
) -> List[Dict[str, Any]]:
    """Enumerate every artifact whose lineage traces to the capture.

    Everything under ``capture_root`` (except ``raw/``, the capturer's raw truth)
    traces directly (depth 1). Files under additional roots are walked as a
    reference graph to a fixpoint, so derived-of-derived chains that carry no
    capture ids of their own are still caught: an artifact is derived if it
    names the capture, references a derived artifact, or is referenced by one.
    """
    root = Path(capture_root).expanduser()
    roots = [Path(item).expanduser() for item in additional_artifact_roots]

    entries: Dict[Path, Dict[str, Any]] = {}
    for path in _iter_files(root):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == "raw":
            continue
        # Never re-enumerate our own takedown outputs (manifest, recall markers,
        # quarantined bytes, executed-recall record) as fresh recall targets.
        if relative.as_posix().startswith(CONSENT_TAKEDOWN_DIR_PREFIX):
            continue
        entries[path] = {
            "root": root,
            "relative": relative.as_posix(),
            "depth": 1,
            "via": [],
        }

    # Reference-graph fixpoint over the additional roots.
    candidates: Dict[Path, Dict[str, Any]] = {}
    for extra_root in roots:
        for path in _iter_files(extra_root):
            payload = (
                _read_json(path) if path.suffix.lower() in _PARSEABLE_SUFFIXES else {}
            )
            refs: List[str] = []
            _reference_strings(payload, refs)
            resolved_refs = [
                resolved
                for resolved in (
                    _resolve_reference(text, base_dir=path.parent, roots=roots)
                    for text in refs
                )
                if resolved is not None
            ]
            candidates[path] = {
                "root": extra_root,
                "relative": path.relative_to(extra_root).as_posix(),
                "payload": payload,
                "references": resolved_refs,
                "names_capture": bool(payload)
                and _payload_names_capture(
                    payload, capture_id=capture_id, scene_id=scene_id
                ),
            }

    for path, info in candidates.items():
        if info["names_capture"] or (
            capture_id and f"captures/{capture_id}" in path.as_posix()
        ):
            entries.setdefault(
                path,
                {
                    "root": info["root"],
                    "relative": info["relative"],
                    "depth": 1,
                    "via": [],
                },
            )

    changed = True
    while changed:
        changed = False
        for path, info in candidates.items():
            references = info["references"]
            if path not in entries:
                derived_parents = [ref for ref in references if ref in entries]
                if derived_parents:
                    parent_depth = min(entries[ref]["depth"] for ref in derived_parents)
                    entries[path] = {
                        "root": info["root"],
                        "relative": info["relative"],
                        "depth": parent_depth + 1,
                        "via": sorted(str(ref) for ref in derived_parents),
                    }
                    changed = True
            if path in entries:
                # A derived manifest's referenced payload files (clips, videos)
                # are part of the derived set even when they are opaque bytes.
                # They inherit the manifest's depth: a manifest and the payload
                # it describes are one artifact, not another derivation hop.
                for ref in references:
                    if ref not in entries and any(
                        ref.is_relative_to(extra_root) for extra_root in roots
                    ):
                        ref_root = next(
                            extra_root
                            for extra_root in roots
                            if ref.is_relative_to(extra_root)
                        )
                        entries[ref] = {
                            "root": ref_root,
                            "relative": ref.relative_to(ref_root).as_posix(),
                            "depth": entries[path]["depth"],
                            "via": [str(path)],
                        }
                        changed = True

    artifacts: List[Dict[str, Any]] = []
    for path in sorted(entries):
        info = entries[path]
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = None
        artifacts.append(
            {
                "path": str(path),
                "relative_path": info["relative"],
                "size_bytes": size_bytes,
                "sha256": _sha256_file(path),
                "lineage": {
                    "traces_to_capture": True,
                    "depth": info["depth"],
                    "via": info["via"],
                },
            }
        )
    return artifacts


def _external_surfaces(capture_root: Path) -> Dict[str, Any]:
    root = Path(capture_root).expanduser()
    sync_result = _read_json(root / "pipeline" / "webapp_sync_result.json")
    syncs = _mapping(sync_result.get("syncs"))
    webapp_response_ids: Dict[str, Any] = {}
    upstream_ids: Dict[str, Any] = {}
    for payload in ([sync_result] + [value for value in syncs.values()]):
        payload = _mapping(payload)
        for key, value in _mapping(payload.get("webapp_response_ids")).items():
            webapp_response_ids.setdefault(key, value)
        attachment = _mapping(payload.get("attachment_payload"))
        for key in ("site_submission_id", "request_id", "buyer_request_id", "capture_job_id"):
            value = _string(attachment.get(key))
            if value:
                upstream_ids.setdefault(key, value)
    site_world_spec = _read_json(root / "pipeline" / "site_world_spec.json")
    hosted_manifest = _read_json(
        root / "pipeline" / "hosted_session_runtime_manifest.json"
    )
    return {
        "webapp": {
            "sync_result_present": bool(sync_result),
            "sync_status": _string(sync_result.get("status")) or None,
            "webapp_response_ids": webapp_response_ids,
            "upstream_ids": upstream_ids,
        },
        "hosted_sessions": {
            "site_world_spec_present": bool(site_world_spec),
            "canonical_package_uri": _string(site_world_spec.get("canonical_package_uri"))
            or None,
            "canonical_package_version": _string(
                site_world_spec.get("canonical_package_version")
            )
            or None,
            "hosted_session_runtime_manifest_present": bool(hosted_manifest),
        },
    }


def _webapp_revocation_signal(
    *,
    scene_id: str,
    capture_id: str,
    consent: Mapping[str, Any],
    external_surfaces: Mapping[str, Any],
    artifact_count: int,
    generated_at: str,
) -> Dict[str, Any]:
    webapp = _mapping(_mapping(external_surfaces).get("webapp"))
    return {
        "schema_version": WEBAPP_REVOCATION_SIGNAL_SCHEMA_VERSION,
        "generated_at": generated_at,
        "verdict": "revoked",
        "verdict_is_explicit_not_absence": True,
        "scene_id": scene_id,
        "capture_id": capture_id,
        "consent_revoked_at": consent.get("consent_revoked_at"),
        "consent_state": consent.get("state"),
        "required_webapp_state": "blocked_consent_revoked_takedown_required",
        "required_actions": [
            "mark_webapp_rights_privacy_blocking",
            "revoke_buyer_and_reviewer_access",
            "hide_package_delivery_affordances",
            "hide_training_export_affordances",
            "acknowledge_revocation_verdict",
        ],
        "webapp_response_ids": _mapping(webapp.get("webapp_response_ids")),
        "upstream_ids": _mapping(webapp.get("upstream_ids")),
        "artifact_count": artifact_count,
    }


def propagate_consent_takedown(
    *,
    capture_root: Path,
    additional_artifact_roots: Sequence[Path] = (),
    output_path: Path | None = None,
    write: bool = True,
) -> Dict[str, Any]:
    """Build (and persist) the fail-closed takedown manifest for one capture.

    ``takedown_open`` is True when consent is revoked OR when the consent state
    cannot be verified — an unverifiable rights state must never keep serving.
    """
    root = Path(capture_root).expanduser()
    generated_at = utc_now_iso()
    scene_id, capture_id = _identity_from_path(root)
    raw_manifest = _read_json(root / "raw" / "manifest.json")
    scene_id = _string(raw_manifest.get("scene_id")) or scene_id
    capture_id = _string(raw_manifest.get("capture_id")) or capture_id

    consent = read_consent_state(root)
    consent_state = consent["state"]
    takedown_open = consent_state != "active"
    if consent_state == "revoked":
        status = "takedown_open"
    elif consent_state == "unknown":
        status = "takedown_open_consent_state_unknown"
    else:
        status = "not_required"

    artifacts = enumerate_derived_artifacts(
        capture_root=root,
        capture_id=capture_id,
        scene_id=scene_id,
        additional_artifact_roots=additional_artifact_roots,
    )
    tombstone_template = {
        "status": "takedown_required",
        "reason": "consent_revoked"
        if consent_state == "revoked"
        else "consent_state_unverifiable",
        "scene_id": scene_id,
        "capture_id": capture_id,
        "consent_revoked_at": consent.get("consent_revoked_at"),
        "tombstoned_at": generated_at,
        "serve_allowed": False,
        "training_use_allowed": False,
        "hosting_allowed": False,
    }
    for entry in artifacts:
        entry["tombstone"] = dict(tombstone_template) if takedown_open else {}

    external_surfaces = _external_surfaces(root)

    blockers: List[str] = []
    if consent_state == "unknown":
        blockers.append("consent_state_unverifiable")
    if takedown_open:
        blockers.append("webapp_revocation_sync_not_executed")
        blockers.append("hosted_session_takedown_not_executed")

    manifest: Dict[str, Any] = {
        "schema_version": TAKEDOWN_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": scene_id,
        "capture_id": capture_id,
        "capture_root": str(root),
        "status": status,
        "takedown_open": takedown_open,
        "consent": consent,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "additional_artifact_roots": [str(Path(item).expanduser()) for item in additional_artifact_roots],
        "external_surfaces": external_surfaces,
        "webapp_revocation_signal": (
            _webapp_revocation_signal(
                scene_id=scene_id,
                capture_id=capture_id,
                consent=consent,
                external_surfaces=external_surfaces,
                artifact_count=len(artifacts),
                generated_at=generated_at,
            )
            if takedown_open
            else {}
        ),
        "webapp_revocation_sync": {
            "executed": False,
            "status": "queued_unexecuted_webapp_revocation_sync"
            if takedown_open
            else "not_required",
        },
        "required_actions": list(_REQUIRED_TAKEDOWN_ACTIONS) if takedown_open else [],
        "blockers": blockers,
        "claim_boundary": {
            "manifest_is_local_enumeration_not_downstream_execution_proof": True,
            "enumeration_over_inclusive_by_design": True,
            "raw_capture_truth_excluded_from_derived_enumeration": True,
            "webapp_revocation_sync_executed": False,
            "hosted_session_takedown_executed": False,
            "takedown_manifest_is_not_legal_advice": True,
        },
    }

    if write:
        path = (
            Path(output_path).expanduser()
            if output_path
            else root / TAKEDOWN_MANIFEST_RELATIVE_PATH
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, manifest)
        manifest["takedown_manifest_path"] = str(path)
    return manifest


def evaluate_delivery_time_takedown_gate(
    *,
    capture_root: Path,
    artifact_path: Path | None = None,
    surface: str = "unspecified",
) -> Dict[str, Any]:
    """Fail-closed serve/sync-time gate for one capture's artifacts.

    Consent is re-read live so a revocation blocks delivery even before
    propagation has run; a persisted open takedown stays authoritative until
    explicitly re-propagated, even if the consent source flips back.
    """
    root = Path(capture_root).expanduser()
    consent = read_consent_state(root)
    manifest = _read_json(root / TAKEDOWN_MANIFEST_RELATIVE_PATH)
    manifest_open = bool(manifest.get("takedown_open"))

    blockers: List[str] = []
    if consent["state"] == "revoked":
        blockers.append("consent_revoked_takedown_required")
    if manifest_open:
        blockers.append("open_takedown_manifest_present")
    if consent["state"] == "unknown" and not manifest_open:
        status = "blocked_consent_state_unverifiable"
        blockers.append("consent_state_unverifiable")
    elif blockers:
        status = "blocked_open_consent_takedown"
    else:
        status = "allowed"

    return {
        "schema_version": DELIVERY_GATE_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": status,
        "serve_allowed": status == "allowed",
        "surface": surface,
        "capture_root": str(root),
        "artifact_path": str(Path(artifact_path).expanduser()) if artifact_path else None,
        "consent": consent,
        "takedown_manifest_present": bool(manifest),
        "takedown_manifest_open": manifest_open,
        "blockers": blockers,
    }


def sync_webapp_consent_revocation(
    *,
    takedown_manifest: Mapping[str, Any],
    timeout_seconds: int | None = None,
) -> Dict[str, Any]:
    """Push the explicit revoked verdict to the webapp over the signed channel.

    Never claims execution it did not perform: an unconfigured or failed sync
    returns a queued/failed status with blockers, not silence.
    """
    signal = _mapping(takedown_manifest.get("webapp_revocation_signal"))
    if not signal:
        return {
            "status": "not_required",
            "executed": False,
            "blockers": [],
        }
    sync_url = _string_env("PIPELINE_SYNC_WEBAPP_URL")
    sync_token = _string_env("PIPELINE_SYNC_TOKEN")
    if not sync_url or not sync_token:
        return {
            "status": "queued_unexecuted_webapp_revocation_sync",
            "executed": False,
            "blockers": ["webapp_revocation_sync_not_configured"],
        }
    parsed_sync_url = urllib_parse.urlsplit(sync_url)
    if parsed_sync_url.scheme not in {"http", "https"} or not parsed_sync_url.netloc:
        return {
            "status": "failed_webapp_revocation_sync",
            "executed": False,
            "blockers": ["webapp_revocation_sync_url_invalid"],
        }
    body = json.dumps(dict(signal), separators=(",", ":")).encode("utf-8")
    request = urllib_request.Request(
        sync_url,
        data=body,
        headers=_pipeline_sync_headers(sync_token, body),
        method="POST",
    )
    timeout = timeout_seconds or max(1, _int_env("PIPELINE_SYNC_TIMEOUT_SECONDS", 10))
    try:
        with urllib_request.urlopen(  # nosec B310 -- HTTP(S) URL validated above.
            request, timeout=timeout
        ) as response:
            raw = response.read().decode("utf-8")
    except (urllib_error.URLError, TimeoutError, ValueError, OSError) as exc:
        return {
            "status": "failed_webapp_revocation_sync",
            "executed": False,
            "reason": f"{exc.__class__.__name__}:{exc}",
            "blockers": ["webapp_revocation_sync_failed"],
        }
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        parsed = {}
    return {
        "status": "executed",
        "executed": True,
        "response": parsed if isinstance(parsed, dict) else {},
        "blockers": [],
    }


# ---------------------------------------------------------------------------
# Recall EXECUTION: enumeration is not enough — a revoked capture's downstream
# derived artifacts must be actively recalled (quarantined/deleted) and the
# buyer-entitlement revocation handed off to the webapp. Raw capture truth is
# authoritative and is never modified: only DERIVED deliverables are recalled.
# ---------------------------------------------------------------------------


def _quarantine_key(src: Path) -> str:
    """Collision-free quarantine filename that preserves the original name."""
    digest = sha256(str(src).encode("utf-8")).hexdigest()[:16]
    return f"{digest}__{src.name}"


def _move_artifact_to_quarantine(src: Path, dst: Path) -> None:
    """Move a derived deliverable out of its delivery location into quarantine.

    Isolated for testability: the takedown drill patches this to prove a target
    that cannot be recalled surfaces as an explicit blocked state, never a false
    success.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def _delete_artifact(path: Path) -> None:
    """Delete a derived deliverable. Allowed: derived outputs are re-derivable;
    only the raw capture bundle is authoritative and must never be deleted."""
    path.unlink()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _recall_one_artifact(
    artifact: Mapping[str, Any],
    *,
    recall_mode: str,
    quarantine_root: Path,
    raw_root: Path,
    reason: str,
    scene_id: str,
    capture_id: str,
    consent_revoked_at: Any,
    recalled_at: str,
) -> Dict[str, Any]:
    """Execute the recall for a single enumerated artifact and write its marker.

    Fail-closed: a raw-truth path or a failed filesystem recall yields an
    explicit ``blocked`` outcome with ``needs_operator`` — never a silent pass.
    """
    src = Path(str(artifact.get("path")))
    relative_path = _string(artifact.get("relative_path")) or src.name
    quarantine_path: str | None = None
    detail: str | None = None

    if _is_within(src, raw_root):
        # Defensive: enumeration already excludes raw/, but never let a recall
        # touch the authoritative capture bundle even if one slipped through.
        outcome = "blocked"
        detail = "raw_capture_truth_protected"
    elif recall_mode == RECALL_MODE_MARK:
        outcome = "recalled_marked"
    elif not src.exists():
        outcome = "recalled_absent"
    elif recall_mode == RECALL_MODE_DELETE:
        try:
            _delete_artifact(src)
            outcome = "deleted"
        except OSError as exc:
            outcome = "blocked"
            detail = f"delete_failed:{exc.__class__.__name__}:{exc}"
    else:  # RECALL_MODE_QUARANTINE
        dst = quarantine_root / _quarantine_key(src)
        try:
            _move_artifact_to_quarantine(src, dst)
            outcome = "quarantined"
            quarantine_path = str(dst)
        except OSError as exc:
            outcome = "blocked"
            detail = f"quarantine_failed:{exc.__class__.__name__}:{exc}"

    terminal = outcome in _TERMINAL_RECALLED_OUTCOMES
    needs_operator = outcome == "blocked"

    marker = {
        "schema_version": RECALL_MARKER_SCHEMA_VERSION,
        "status": outcome,
        "reason": reason,
        "detail": detail,
        "recall_mode": recall_mode,
        "scene_id": scene_id,
        "capture_id": capture_id,
        "relative_path": relative_path,
        "original_path": str(src),
        "original_sha256": artifact.get("sha256"),
        "original_size_bytes": artifact.get("size_bytes"),
        "quarantine_path": quarantine_path,
        "consent_revoked_at": consent_revoked_at,
        "recalled_at": recalled_at,
        "serve_allowed": False,
        "training_use_allowed": False,
        "hosting_allowed": False,
        "terminal": terminal,
        "needs_operator": needs_operator,
        "raw_capture_truth_preserved": True,
    }
    marker_path = Path(str(src) + RECALL_MARKER_SUFFIX)
    try:
        write_json(marker_path, marker)
        marker_written = True
    except OSError as exc:
        # If we cannot even record the recall, that is itself a blocked state.
        marker_written = False
        outcome = "blocked"
        terminal = False
        needs_operator = True
        detail = (detail + "; " if detail else "") + (
            f"marker_write_failed:{exc.__class__.__name__}:{exc}"
        )

    return {
        "relative_path": relative_path,
        "path": str(src),
        "outcome": outcome,
        "detail": detail,
        "terminal": terminal,
        "blocked": outcome == "blocked",
        "needs_operator": needs_operator,
        "quarantine_path": quarantine_path,
        "marker_path": str(marker_path) if marker_written else None,
        "recalled_at": recalled_at,
        "lineage": dict(_mapping(artifact.get("lineage"))),
    }


def _webapp_handoff_record(
    manifest: Mapping[str, Any],
    *,
    sync_webapp: bool,
    timeout_seconds: int | None,
) -> Dict[str, Any]:
    """Emit the authoritative buyer-entitlement revocation handoff.

    Entitlement/access revocation is WEBAPP-owned. The pipeline produces the
    signed revoked-verdict payload the webapp consumes and, only when the signed
    channel is configured and ``sync_webapp`` is requested, attempts the push.
    It never fabricates a webapp call: an unexecuted handoff stays explicitly
    pending, not silently successful.
    """
    signal = _mapping(manifest.get("webapp_revocation_signal"))
    if not signal:
        return {
            "handoff_owner": "webapp",
            "signal": {},
            "executed": False,
            "status": "not_required",
            "blockers": [],
        }
    if not sync_webapp:
        return {
            "handoff_owner": "webapp",
            "signal": dict(signal),
            "executed": False,
            "status": "handoff_emitted_pending_webapp_execution",
            "blockers": ["webapp_entitlement_revocation_pending_webapp_execution"],
        }
    sync_result = sync_webapp_consent_revocation(
        takedown_manifest=manifest, timeout_seconds=timeout_seconds
    )
    return {
        "handoff_owner": "webapp",
        "signal": dict(signal),
        "executed": bool(sync_result.get("executed")),
        "status": sync_result.get("status"),
        "response": sync_result.get("response", {}),
        "reason": sync_result.get("reason"),
        "blockers": list(sync_result.get("blockers") or []),
    }


def execute_consent_takedown(
    *,
    capture_root: Path,
    manifest: Mapping[str, Any] | None = None,
    additional_artifact_roots: Sequence[Path] = (),
    recall_mode: str = RECALL_MODE_QUARANTINE,
    quarantine_root: Path | None = None,
    sync_webapp: bool = False,
    timeout_seconds: int | None = None,
    output_path: Path | None = None,
    write: bool = True,
) -> Dict[str, Any]:
    """Execute the recall an enumerated takedown only *describes*.

    For every enumerated derived artifact this recalls the deliverable
    (quarantine by default; delete or mark are also supported), writes a
    per-artifact recall marker, emits the webapp buyer-entitlement revocation
    handoff, and records an executed-recall audit record with per-target
    outcomes. Fail-closed: any target that cannot be recalled surfaces as an
    explicit ``blocked`` outcome and drives the overall status to
    ``blocked_needs_operator`` — success is never claimed for it.

    Capture-truth boundary: only DERIVED deliverables are recalled. The raw
    capture bundle (``raw/``) is authoritative, excluded from enumeration, and
    never quarantined, deleted, or rewritten by this path.
    """
    if recall_mode not in _RECALL_MODES:
        raise ValueError(f"unknown recall_mode: {recall_mode!r}")

    root = Path(capture_root).expanduser()
    if manifest is None:
        manifest = propagate_consent_takedown(
            capture_root=root,
            additional_artifact_roots=additional_artifact_roots,
            write=write,
        )

    executed_at = utc_now_iso()
    scene_id = _string(manifest.get("scene_id"))
    capture_id = _string(manifest.get("capture_id"))
    consent = _mapping(manifest.get("consent"))
    takedown_open = bool(manifest.get("takedown_open"))
    reason = (
        "consent_revoked"
        if consent.get("state") == "revoked"
        else "consent_state_unverifiable"
    )
    raw_root = root / "raw"
    q_root = (
        Path(quarantine_root).expanduser()
        if quarantine_root is not None
        else root / QUARANTINE_RELATIVE_DIR
    )

    webapp_handoff = _webapp_handoff_record(
        manifest, sync_webapp=sync_webapp, timeout_seconds=timeout_seconds
    )

    if not takedown_open:
        record: Dict[str, Any] = {
            "schema_version": RECALL_EXECUTION_SCHEMA_VERSION,
            "generated_at": executed_at,
            "executed_at": executed_at,
            "scene_id": scene_id,
            "capture_id": capture_id,
            "capture_root": str(root),
            "status": "not_required",
            "executed": False,
            "recall_mode": recall_mode,
            "takedown_open": False,
            "consent": consent,
            "target_count": 0,
            "targets": [],
            "outcome_counts": {},
            "webapp_handoff": webapp_handoff,
            "webapp_entitlement_revocation_executed": bool(
                webapp_handoff.get("executed")
            ),
            "blockers": [],
            "capture_truth_boundary": _capture_truth_boundary(),
            "claim_boundary": _recall_claim_boundary(executed=False, handoff=webapp_handoff),
        }
        if write:
            record = _write_recall_record(record, root=root, output_path=output_path)
        return record

    targets: List[Dict[str, Any]] = []
    for artifact in manifest.get("artifacts") or []:
        targets.append(
            _recall_one_artifact(
                artifact,
                recall_mode=recall_mode,
                quarantine_root=q_root,
                raw_root=raw_root,
                reason=reason,
                scene_id=scene_id,
                capture_id=capture_id,
                consent_revoked_at=consent.get("consent_revoked_at"),
                recalled_at=executed_at,
            )
        )

    outcome_counts: Dict[str, int] = {}
    for target in targets:
        outcome_counts[target["outcome"]] = outcome_counts.get(target["outcome"], 0) + 1
    blocked = [target for target in targets if target["blocked"]]

    blockers: List[str] = [
        f"recall_blocked:{target['relative_path']}" for target in blocked
    ]
    status = "blocked_needs_operator" if blocked else "executed"
    executed = status == "executed"

    record = {
        "schema_version": RECALL_EXECUTION_SCHEMA_VERSION,
        "generated_at": executed_at,
        "executed_at": executed_at,
        "scene_id": scene_id,
        "capture_id": capture_id,
        "capture_root": str(root),
        "status": status,
        "executed": executed,
        "recall_mode": recall_mode,
        "takedown_open": True,
        "consent": consent,
        "target_count": len(targets),
        "terminal_recalled_count": sum(
            1 for target in targets if target["terminal"]
        ),
        "blocked_count": len(blocked),
        "targets": targets,
        "outcome_counts": outcome_counts,
        "quarantine_root": str(q_root),
        "webapp_handoff": webapp_handoff,
        "webapp_entitlement_revocation_executed": bool(webapp_handoff.get("executed")),
        "blockers": blockers,
        "capture_truth_boundary": _capture_truth_boundary(),
        "claim_boundary": _recall_claim_boundary(executed=executed, handoff=webapp_handoff),
    }
    if write:
        record = _write_recall_record(record, root=root, output_path=output_path)
    return record


def _capture_truth_boundary() -> Dict[str, Any]:
    return {
        "raw_capture_bundle_never_modified": True,
        "raw_excluded_from_recall_targets": True,
        "only_derived_deliverables_recalled": True,
        "derived_deliverable_deletion_allowed": True,
        "raw_is_authoritative_provenance": True,
    }


def _recall_claim_boundary(*, executed: bool, handoff: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "pipeline_owned_recall_executed": executed,
        "webapp_entitlement_revocation_is_webapp_owned": True,
        "webapp_revocation_handoff_executed": bool(handoff.get("executed")),
        "blocked_targets_surface_as_blocked_not_success": True,
        "recall_record_is_execution_proof_not_legal_advice": True,
    }


def _write_recall_record(
    record: Dict[str, Any], *, root: Path, output_path: Path | None
) -> Dict[str, Any]:
    path = (
        Path(output_path).expanduser()
        if output_path
        else root / RECALL_EXECUTION_RECORD_RELATIVE_PATH
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, record)
    record = dict(record)
    record["recall_execution_record_path"] = str(path)
    return record


def run_takedown_drill(
    *,
    capture_root: Path,
    additional_artifact_roots: Sequence[Path] = (),
    recall_mode: str = RECALL_MODE_QUARANTINE,
    sync_webapp: bool = False,
    write: bool = True,
) -> Dict[str, Any]:
    """The takedown DRILL: run a takedown end-to-end and prove every enumerated
    target reached a terminal recalled/quarantined state OR an explicit blocked
    state — the exercise the R049 finding says was missing.

    Returns a drill report. ``coverage_complete`` is True only when every
    enumerated artifact has a corresponding target that reached a recognized
    terminal-or-blocked state (no target silently left accessible). ``status`` is
    ``passed`` when the recall fully executed, ``blocked`` when the recall was
    coverage-complete but at least one target needs an operator, ``not_required``
    when consent is intact, and ``failed`` if any enumerated target was left
    unresolved.
    """
    root = Path(capture_root).expanduser()
    manifest = propagate_consent_takedown(
        capture_root=root,
        additional_artifact_roots=additional_artifact_roots,
        write=write,
    )
    execution = execute_consent_takedown(
        capture_root=root,
        manifest=manifest,
        additional_artifact_roots=additional_artifact_roots,
        recall_mode=recall_mode,
        sync_webapp=sync_webapp,
        write=write,
    )

    if not manifest.get("takedown_open"):
        return {
            "schema_version": TAKEDOWN_DRILL_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "capture_root": str(root),
            "status": "not_required",
            "coverage_complete": True,
            "all_recalled": True,
            "enumerated_target_count": len(manifest.get("artifacts") or []),
            "terminal_recalled_count": 0,
            "blocked_count": 0,
            "unresolved_targets": [],
            "manifest_status": manifest.get("status"),
            "execution_status": execution.get("status"),
            "execution": execution,
        }

    enumerated_paths = [str(a.get("path")) for a in manifest.get("artifacts") or []]
    target_by_path: Dict[str, Dict[str, Any]] = {
        str(target.get("path")): target for target in execution.get("targets") or []
    }
    unresolved: List[str] = []
    for path in enumerated_paths:
        target = target_by_path.get(path)
        # A target is "resolved" only if it reached a terminal recalled state or
        # an explicit blocked state. Anything else means the enumerated artifact
        # was left silently accessible — exactly the R049 failure.
        if target is None or not (target.get("terminal") or target.get("blocked")):
            unresolved.append(path)

    coverage_complete = not unresolved
    blocked_count = int(execution.get("blocked_count") or 0)
    all_recalled = coverage_complete and blocked_count == 0

    if not coverage_complete:
        status = "failed"
    elif blocked_count:
        status = "blocked"
    else:
        status = "passed"

    return {
        "schema_version": TAKEDOWN_DRILL_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "capture_root": str(root),
        "status": status,
        "coverage_complete": coverage_complete,
        "all_recalled": all_recalled,
        "enumerated_target_count": len(enumerated_paths),
        "terminal_recalled_count": int(execution.get("terminal_recalled_count") or 0),
        "blocked_count": blocked_count,
        "unresolved_targets": unresolved,
        "manifest_status": manifest.get("status"),
        "execution_status": execution.get("status"),
        "webapp_entitlement_revocation_executed": bool(
            execution.get("webapp_entitlement_revocation_executed")
        ),
        "execution": execution,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    mode = "propagate"
    if args and args[0] in {"--propagate", "--execute", "--drill"}:
        mode = args.pop(0)[2:]
    if not args:
        print(
            "usage: python -m blueprint_pipeline.consent_takedown "
            "[--propagate|--execute|--drill] <capture_root> "
            "[additional_artifact_root ...]",
            file=sys.stderr,
        )
        return 2
    capture_root = Path(args[0])
    extra = [Path(item) for item in args[1:]]

    if mode == "drill":
        report = run_takedown_drill(
            capture_root=capture_root, additional_artifact_roots=extra
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] in {"passed", "not_required"} else 1
    if mode == "execute":
        record = execute_consent_takedown(
            capture_root=capture_root, additional_artifact_roots=extra
        )
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0 if record["status"] in {"executed", "not_required"} else 1

    manifest = propagate_consent_takedown(
        capture_root=capture_root,
        additional_artifact_roots=extra,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "not_required" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
