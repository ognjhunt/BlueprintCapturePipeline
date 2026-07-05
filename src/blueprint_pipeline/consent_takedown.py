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
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

from .common import utc_now_iso, write_json
from .webapp_sync import _int_env, _pipeline_sync_headers, _string_env

TAKEDOWN_MANIFEST_SCHEMA_VERSION = "takedown_manifest.v1"
WEBAPP_REVOCATION_SIGNAL_SCHEMA_VERSION = "webapp_consent_revocation_signal.v1"
DELIVERY_GATE_SCHEMA_VERSION = "consent_takedown_delivery_gate.v1"

TAKEDOWN_MANIFEST_RELATIVE_PATH = "pipeline/consent_takedown/takedown_manifest.json"

CONSENT_REVOKED_STATUSES = frozenset({"revoked", "withdrawn", "rescinded"})

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


def _explicit_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "revoked", "withdrawn", "rescinded"}:
            return True
        if normalized in {"0", "false", "no", "n", "active", "documented"}:
            return False
    return None


def _explicit_true(*values: Any) -> bool:
    return any(_explicit_bool(value) is True for value in values)


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
    exists AND it does not indicate revocation. Missing sources are ``"unknown"``.
    """
    root = Path(capture_root).expanduser()
    for relative in _CONSENT_SOURCE_RELATIVES:
        payload = _read_json(root / relative)
        if not payload:
            continue
        nested = _mapping(
            payload.get("capture_rights")
            or payload.get("rights_consent")
            or payload.get("rights")
        )
        source = nested or payload
        consent_status = _string(
            source.get("consent_status") or source.get("consentStatus")
        )
        consent_revoked = (
            consent_status.lower() in CONSENT_REVOKED_STATUSES
            or _explicit_true(source.get("consent_revoked"))
            or _explicit_true(source.get("consentRevoked"))
            or bool(source.get("consent_revoked_at") or source.get("consentRevokedAt"))
        )
        # raw/manifest.json and capture_descriptor.json exist for every capture;
        # if they carry no consent fields they are not a consent source.
        if not consent_status and not consent_revoked and relative in (
            "raw/manifest.json",
            "capture_descriptor.json",
        ):
            continue
        return {
            "state": "revoked" if consent_revoked else "active",
            "consent_revoked": consent_revoked,
            "consent_status": consent_status or None,
            "consent_revoked_at": _string(
                source.get("consent_revoked_at") or source.get("consentRevokedAt")
            )
            or None,
            "source_path": str(root / relative),
        }
    return {
        "state": "unknown",
        "consent_revoked": False,
        "consent_status": None,
        "consent_revoked_at": None,
        "source_path": None,
    }


def _iter_files(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


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
        if relative.as_posix() == TAKEDOWN_MANIFEST_RELATIVE_PATH:
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
    body = json.dumps(dict(signal), separators=(",", ":")).encode("utf-8")
    request = urllib_request.Request(
        sync_url,
        data=body,
        headers=_pipeline_sync_headers(sync_token, body),
        method="POST",
    )
    timeout = timeout_seconds or max(1, _int_env("PIPELINE_SYNC_TIMEOUT_SECONDS", 10))
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
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


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(
            "usage: python -m blueprint_pipeline.consent_takedown <capture_root> "
            "[additional_artifact_root ...]",
            file=sys.stderr,
        )
        return 2
    manifest = propagate_consent_takedown(
        capture_root=Path(args[0]),
        additional_artifact_roots=[Path(item) for item in args[1:]],
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "not_required" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
