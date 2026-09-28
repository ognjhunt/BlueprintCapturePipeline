"""Which captures belong to the website self-capture lane.

A site that films its own workcell from its website capture link arrives in one
of two shapes:

- a browser upload, which says so in ``capture_source == "browser_self_capture"``
  and carries no ARKit evidence;
- an iPhone Raw Contract V3.2 bundle recorded with the Blueprint app or App Clip.
  It keeps ``capture_source == "iphone"`` — iPhone treatment is derived from
  ``capture_profile_id`` and must not be lost — and instead carries a
  ``site_self_capture`` block that Blueprint-WebApp writes at upload completion.

The block is honoured only in its exact server-authored shape and only when it
names the same submission as ``site_submission_id``. A device-written manifest
may not carry it (the WebApp refuses one that does). The same rule is
``isServerAuthoredSiteSelfCapture`` in BlueprintCapture ``cloud/extract-frames``.

Downstream stages read the lane from ``metadata.capture_entry_source``; this
module is the one place that decides it.
"""

from __future__ import annotations

from typing import Any, Mapping

BROWSER_SELF_CAPTURE = "browser_self_capture"
SITE_SELF_CAPTURE = "site_self_capture"
WEBSITE_ENTRY_SOURCES = frozenset({BROWSER_SELF_CAPTURE, SITE_SELF_CAPTURE})

SITE_SELF_CAPTURE_SCHEMA_VERSION = "site_self_capture.v1"


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def is_server_authored_site_self_capture(manifest: Mapping[str, Any] | None) -> bool:
    if not isinstance(manifest, Mapping):
        return False
    origin = manifest.get("site_self_capture")
    if not isinstance(origin, Mapping):
        return False
    request_id = _text(origin.get("request_id"))
    return (
        _text(origin.get("schema_version")) == SITE_SELF_CAPTURE_SCHEMA_VERSION
        and _text(origin.get("authored_by")) == "blueprint_webapp"
        and origin.get("site_filmed_itself") is True
        and origin.get("capture_job_exists") is False
        and bool(request_id)
        and request_id == _text(manifest.get("site_submission_id"))
    )


def website_entry_source(manifest: Mapping[str, Any] | None) -> str | None:
    """The website lane a manifest belongs to, or None for a device capture."""
    if not isinstance(manifest, Mapping):
        return None
    if manifest.get("capture_source") == BROWSER_SELF_CAPTURE:
        return BROWSER_SELF_CAPTURE
    if is_server_authored_site_self_capture(manifest):
        return SITE_SELF_CAPTURE
    return None


def capture_entry_source(manifest: Mapping[str, Any] | None) -> str:
    """What ``metadata.capture_entry_source`` records for this manifest."""
    lane = website_entry_source(manifest)
    if lane:
        return lane
    if not isinstance(manifest, Mapping):
        return ""
    return str(manifest.get("capture_source") or "")


def is_website_entry_source(value: Any) -> bool:
    return isinstance(value, str) and value in WEBSITE_ENTRY_SOURCES


def is_website_capture_manifest(manifest: Mapping[str, Any] | None) -> bool:
    return website_entry_source(manifest) is not None
