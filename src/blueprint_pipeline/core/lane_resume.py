"""Per-lane completion ledger so production dispatch retries skip completed lanes.

The production dispatch path (``functions/storage_trigger.py`` ->
``run_capture_pipeline``) never goes through ``run_e2e``, so the run_e2e stage
ledger cannot protect Cloud Tasks x Cloud Run Job retries from re-running lanes
that already completed for the same capture input. This module persists a small
per-lane marker under the capture's canonical pipeline prefix
(``scenes/<scene>/captures/<cap>/pipeline/lane_ledger/<lane>.json``) keyed on
the same capture input fingerprint run_e2e uses (capture descriptor + raw
manifest sha256). A lane is skipped only when its marker exists, the
fingerprint matches, and every output path recorded at completion time still
exists. Lane semantics, ordering, and fault isolation are unchanged: markers
are written only after a lane completes successfully, and failed lanes never
write markers.

Kill switch: ``BLUEPRINT_LANE_RESUME_DISABLED=1`` disables both marker reads
and marker writes.
"""

from __future__ import annotations

import json
import logging
import os
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .common import utc_now_iso, write_json
from .stage_outcome import OutcomeKind, StageOutcome

logger = logging.getLogger(__name__)

LANE_LEDGER_DIRNAME = "lane_ledger"
LANE_LEDGER_SCHEMA_VERSION = "capture_lane_ledger.v1"
LANE_RESUME_DISABLED_ENV = "BLUEPRINT_LANE_RESUME_DISABLED"
# Shared with run_e2e: identical derivation and schema id so both resume layers
# agree on what "same capture input" means.
CAPTURE_INPUT_FINGERPRINT_SCHEMA_VERSION = "run_e2e_capture_input_fingerprint.v1"
LANE_LEDGER_FINGERPRINT_SCHEMA_VERSION = "capture_lane_input_fingerprint.v1"
# Descriptor metadata keys the pipeline itself writes back into
# capture_descriptor.json mid-run (qualification.py canonical-package /
# worldlabs writes). The lane-ledger fingerprint must exclude them: they change
# between the first run and a Cloud Tasks/Cloud Run retry, and hashing them
# would make every marker mismatch exactly in the retry scenario this ledger
# exists for. Genuine input changes (raw manifest, requested outputs, any other
# descriptor field) still change the fingerprint.
PIPELINE_WRITTEN_DESCRIPTOR_METADATA_KEYS = frozenset(
    {
        "canonical_site_package_uri",
        "provider_adapter_inputs",
        "worldlabs_request_manifest_uri",
    }
)

__all__ = [
    "CAPTURE_INPUT_FINGERPRINT_SCHEMA_VERSION",
    "LANE_LEDGER_DIRNAME",
    "LANE_LEDGER_FINGERPRINT_SCHEMA_VERSION",
    "LANE_LEDGER_SCHEMA_VERSION",
    "LANE_RESUME_DISABLED_ENV",
    "PIPELINE_WRITTEN_DESCRIPTOR_METADATA_KEYS",
    "capture_input_fingerprint",
    "lane_ledger_dir",
    "lane_ledger_input_fingerprint",
    "lane_marker_path",
    "lane_resume_disabled",
    "read_completed_lane_result",
    "record_lane_completion",
]


def lane_resume_disabled() -> bool:
    return str(os.getenv(LANE_RESUME_DISABLED_ENV) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def lane_ledger_dir(capture_root: Path) -> Path:
    return capture_root / "pipeline" / LANE_LEDGER_DIRNAME


def lane_marker_path(capture_root: Path, lane: str) -> Path:
    return lane_ledger_dir(capture_root) / f"{lane}.json"


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_input_fingerprint_source(
    *,
    capture_root: Path,
    role: str,
    path: Path,
) -> dict[str, Any]:
    relative_path = path.relative_to(capture_root) if path.is_relative_to(capture_root) else path
    exists = path.is_file()
    return {
        "role": role,
        "relative_path": str(relative_path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": _sha_file(path) if exists else None,
    }


def capture_input_fingerprint(
    *,
    capture_root: Path,
    descriptor_path: Path,
    raw_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Capture input fingerprint (descriptor + raw manifest sha256).

    This is the run_e2e derivation, factored out so the production dispatch
    path and run_e2e resume share one definition of "same input".
    """

    resolved_raw_root = raw_root if raw_root is not None else capture_root / "raw"
    sources = [
        _capture_input_fingerprint_source(
            capture_root=capture_root,
            role="capture_descriptor",
            path=descriptor_path,
        ),
        _capture_input_fingerprint_source(
            capture_root=capture_root,
            role="raw_manifest",
            path=resolved_raw_root / "manifest.json",
        ),
    ]
    payload = {
        "schema_version": CAPTURE_INPUT_FINGERPRINT_SCHEMA_VERSION,
        "sources": sources,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "fingerprint_sha256": sha256(encoded).hexdigest()}


def _normalized_descriptor_source(
    *,
    capture_root: Path,
    descriptor_path: Path,
) -> dict[str, Any]:
    """Descriptor fingerprint source with pipeline-written metadata excluded.

    Falls back to the raw file-hash source when the descriptor is not valid
    JSON (nothing to normalize in that case).
    """

    try:
        payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = None
    if not isinstance(payload, Mapping):
        return _capture_input_fingerprint_source(
            capture_root=capture_root,
            role="capture_descriptor",
            path=descriptor_path,
        )
    normalized = dict(payload)
    metadata = normalized.get("metadata")
    if isinstance(metadata, Mapping):
        normalized["metadata"] = {
            key: value
            for key, value in metadata.items()
            if key not in PIPELINE_WRITTEN_DESCRIPTOR_METADATA_KEYS
        }
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
    relative_path = (
        descriptor_path.relative_to(capture_root)
        if descriptor_path.is_relative_to(capture_root)
        else descriptor_path
    )
    return {
        "role": "capture_descriptor_normalized",
        "relative_path": str(relative_path),
        "exists": True,
        "excluded_metadata_keys": sorted(PIPELINE_WRITTEN_DESCRIPTOR_METADATA_KEYS),
        "sha256": sha256(encoded.encode("utf-8")).hexdigest(),
    }


def lane_ledger_input_fingerprint(
    *,
    capture_root: Path,
    descriptor_path: Path,
    raw_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Retry-stable capture input fingerprint for the lane ledger.

    Same shape as :func:`capture_input_fingerprint` but the descriptor source
    is normalized to exclude the metadata keys qualification writes back into
    the descriptor mid-run, so markers recorded during the first run still
    match when a Cloud Tasks/Cloud Run retry recomputes the fingerprint from
    the mutated descriptor.
    """

    resolved_raw_root = raw_root if raw_root is not None else capture_root / "raw"
    sources = [
        _normalized_descriptor_source(
            capture_root=capture_root,
            descriptor_path=descriptor_path,
        ),
        _capture_input_fingerprint_source(
            capture_root=capture_root,
            role="raw_manifest",
            path=resolved_raw_root / "manifest.json",
        ),
    ]
    payload = {
        "schema_version": LANE_LEDGER_FINGERPRINT_SCHEMA_VERSION,
        "sources": sources,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "fingerprint_sha256": sha256(encoded).hexdigest()}


def _lane_output_paths(lane_result: Mapping[str, Any]) -> list[str]:
    """Local artifact paths recorded in a lane result (``*_path`` keys).

    Only existing local files are recorded; ``gs://`` URIs and non-file values
    are ignored. These are re-checked on resume so a marker whose canonical
    outputs were removed forces a re-run.
    """

    outputs: list[str] = []
    for key, value in lane_result.items():
        if not str(key).endswith("_path") or not isinstance(value, str):
            continue
        text = value.strip()
        if not text or text.startswith("gs://"):
            continue
        try:
            if Path(text).is_file():
                outputs.append(text)
        except OSError:
            continue
    return outputs


def record_lane_completion(
    *,
    capture_root: Path,
    lane: str,
    fingerprint: Mapping[str, Any],
    lane_result: Mapping[str, Any],
) -> None:
    """Persist a completion marker for a successfully completed lane.

    Marker write failures are logged, never raised: a completed lane must not
    fail because its resume marker could not be written.
    """

    try:
        safe_result = json.loads(json.dumps(dict(lane_result), default=str))
    except (TypeError, ValueError):
        logger.warning(
            "lane_resume.marker_result_not_serializable lane=%s capture_root=%s",
            lane,
            capture_root,
        )
        return
    marker = {
        "schema_version": LANE_LEDGER_SCHEMA_VERSION,
        "lane": lane,
        "completed_at": utc_now_iso(),
        "capture_input_fingerprint": dict(fingerprint),
        "output_paths": _lane_output_paths(lane_result),
        "outcome": StageOutcome(
            kind=OutcomeKind.PRODUCED,
            artifact=safe_result,
        ).to_mapping(),
        "lane_result": safe_result,
    }
    try:
        write_json(lane_marker_path(capture_root, lane), marker)
    except OSError:
        logger.warning(
            "lane_resume.marker_write_failed lane=%s capture_root=%s",
            lane,
            capture_root,
            exc_info=True,
        )


def read_completed_lane_result(
    *,
    capture_root: Path,
    lane: str,
    fingerprint: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return the stored lane result when the lane may be safely skipped.

    Returns ``None`` (forcing a fresh run) when the marker is missing or
    unreadable, its schema or lane does not match, the capture input
    fingerprint changed, or any recorded output path no longer exists.
    """

    path = lane_marker_path(capture_root, lane)
    if not path.is_file():
        return None
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(marker, Mapping):
        return None
    if marker.get("schema_version") != LANE_LEDGER_SCHEMA_VERSION:
        return None
    if marker.get("lane") != lane:
        return None
    if marker.get("capture_input_fingerprint") != dict(fingerprint):
        return None
    lane_result = marker.get("lane_result")
    if not isinstance(lane_result, Mapping):
        return None
    outcome = marker.get("outcome")
    if outcome is not None:
        if not isinstance(outcome, Mapping):
            return None
        if outcome.get("schema_version") != "stage_outcome.v1":
            return None
        if outcome.get("kind") != OutcomeKind.PRODUCED.value:
            return None
        if outcome.get("artifact") != lane_result:
            return None
    outputs = marker.get("output_paths")
    if not isinstance(outputs, list):
        return None
    for output in outputs:
        try:
            if not Path(str(output)).is_file():
                return None
        except OSError:
            return None
    return dict(lane_result)
