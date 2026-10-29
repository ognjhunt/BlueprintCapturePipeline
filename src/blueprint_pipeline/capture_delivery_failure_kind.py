"""Secret-free staging taxonomy: only proven transport failures permit replay."""
from __future__ import annotations

import errno
from urllib.error import HTTPError

from google.api_core.exceptions import GoogleAPICallError

from .common import PipelineError
from .task_evaluation_scene_retirement_access import SceneRetirementAccessError


def staging_failure_reason(error: BaseException) -> str | None:
    """Inspect typed causes through wrappers; never parse remote error text."""
    from .capture_original_owner_observer import CaptureOwnerObservationError
    seen = set()
    current: BaseException | None = error
    transient = False
    invalid = False
    for _ in range(16):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        # google.api_core HTTP exceptions expose an integer code; arbitrary
        # text or an unknown exception's incidental attributes are not proof.
        code = current.code if isinstance(current, (GoogleAPICallError, HTTPError)) else None
        if isinstance(code, int) and not isinstance(code, bool):
            if 400 <= code < 500 and code not in {408, 429}:
                return "capture_staging_access_denied" if code in {401, 403} else "capture_staging_source_invalid"
            transient = transient or code in {408, 429} or 500 <= code <= 599
        if isinstance(current, PermissionError) or isinstance(current, OSError) and current.errno in {errno.EACCES, errno.EPERM}:
            return "capture_staging_access_denied"
        transient = transient or isinstance(current, (TimeoutError, ConnectionError))
        if isinstance(current, CaptureOwnerObservationError):
            reason = str(current)
            if reason == "capture_owner_response_denied":
                return "capture_staging_access_denied"
            transient = transient or reason == "capture_owner_response_unavailable"
        elif isinstance(current, (PipelineError, SceneRetirementAccessError)):
            reason = str(current)
            if reason.startswith("capture_membership_") and reason != "capture_membership_source_unavailable":
                invalid = True
            if reason in {"capture_staging_content_mismatch", "capture_staging_destination_alias",
                          "capture_staging_manifest_alias"}:
                invalid = True
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    if invalid:
        return "capture_staging_source_invalid"
    if transient:
        return "capture_staging_transport_unavailable"
    return None
