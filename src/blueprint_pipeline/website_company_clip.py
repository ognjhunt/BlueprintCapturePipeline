"""A silent, browser-seekable copy of a website capture's video for approved deployment companies.

Owner decision 2026-11-12: the job, its video included, is shared with approved deployment
companies as the service itself, under the recording statement the site accepts at submission
(``2026-11-12.v4``). Until now no person could watch a site's video: the raw upload sat in
storage and the Pipeline kept its own copies on the host. This makes one "company clip" per
website capture and publishes it exactly as the scene preview is published
(``website_scene_preview``):

- made from the clip website preparation reads (``website_source_video``:
  ``pipeline/source_video/normalization.json`` names it), re-encoded every time, so the raw
  upload is never what is served, even when it was already within the normalization bounds.
  A capture handed off before normalization existed (no record, a derived folder already there)
  has its clip made, the same way, from the raw upload its ``raw/manifest.json`` names
  (``video_uri``); a new capture waits for normalization. The manifest
  records which (``source``: ``normalized`` or ``raw_upload``). A record that exists but cannot
  be used never falls back to the raw upload;
- video stream only: no audio stream at all (sites are told people heard in a clip must know
  it is shared; a company does not need sound), no subtitle or data stream, and no metadata
  (``-map_metadata -1``: a phone's location tags do not travel), and nothing inside the video
  either: captions are not re-embedded (``-a53cc 0``), every SEI unit is dropped, and every frame
  is decoded after writing to prove none carries side data;
- H.264 in MP4 with ``+faststart`` so a browser can seek, at most ``MAX_LONG_EDGE`` px on the
  long side and exactly ``MAX_FPS`` frames a second at most, at a bitrate that keeps a 60 to 120
  second clip well under ``MAX_CLIP_BYTES``, the file bounded while encoding. A clip that cannot
  be brought under the cap is recorded as refused for that source and never retried for it; a
  build that runs past ``ENCODE_TIMEOUT_SECONDS`` is tried again a window later (flagged, never
  failing anything);
- the bytes go to the private artifact store under a name for the capture (withdrawal deletion
  finds them by that name, also after workspace retirement); the WebApp is told the clip's
  identity through the signed ``company-clip`` control operation and reads the bytes, by range,
  through the signed ``/api/live-pipeline/website-company-clips`` route.

Rights are a hard stop. Nothing is made unless the capture is unwithdrawn, the delivery-time
takedown gate allows it, the WebApp's current task context grants consent, and the recording
statement version covers sharing or the capture is Blueprint's own, as
``sharing_statement.sharing_basis`` decides (``config/blueprint_owned_captures.v1.json``, its row's
video digest prefix matched against the digest of the upload the clip is made from). A
capture under an older statement gets nothing, and nothing about it is sent anywhere.

A basis that ends takes the clip back (host copy and stored versions) and closes the route:
Blueprint's list is re-read on every pass and every route read, and a published clip's live
rights at least once every ``LIVE_STATEMENT_RECHECK_SECONDS``, each answer recorded. A WebApp
refusal is remembered: nothing is uploaded again until the live rights differ or a window passes.

Which statement: the capture's files keep the one accepted at submission
(``pipeline/website_task_context.json``, else ``raw/manifest.json``, ``capture_rights.
statement_version``), a snapshot that never changes. A site that accepts the sharing statement
later, on its job page, changes only the WebApp's request record, and nothing re-sends a context
for a video already handed off. So the statement is decided by the live, signed ``task-context``
read (the WebApp projects ``capture_rights`` from the request's current consent attestation on
every read): at once when the snapshot already covers sharing, and otherwise at most once every
``LIVE_STATEMENT_RECHECK_SECONDS`` per capture (``statement_check.json``), so an older job that
agrees later gets its clip within that window and a capture that never agrees costs one read
per window.

It runs in the Atlas preview timer's pass, after the scene preview pass
(``website_atlas_preview.company_clip_tick``): local ffmpeg only, no model, provider or paid
resource. It stops between captures for a hold, a quiesce or a new release, and keeps to
``TICK_BUDGET_SECONDS``. It never touches a job: every failure is a returned status, retried on the next pass.
The Pipeline runs no privacy screen on a website video itself (faces are blurred only in the
frames sent to World Labs), so consent, withdrawal and the takedown gate are the screen.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import math
import os
import re
import stat
import struct
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .permission_recheck import recheck_requested_since
from .sharing_statement import covers_sharing, owned_captures, sharing_basis
from .website_atlas_preview_store import capture_name_matches_request, capture_withdrawn

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA = "website_company_clip_manifest.v1"
CLIP_SCHEMA = "website_company_clip.v1"
ACCEPTANCE_SCHEMA = "website_company_clip_acceptance.v1"
DIRECTORY = ("pipeline", "company_clip")
MANIFEST_FILE = "company_clip.json"
CLIP_FILE = "company_clip.mp4"
#: Withdrawal deletion holds this lock, so no pass publishes between its listing and deletion.
LOCK_FILE = "company_clip.lock"
ARTIFACT_KIND = "website-company-clip"
FORMAT = "mp4_h264_faststart_video_only"
CONTENT_TYPE = "video/mp4"
#: Beside the manifest: the last live read of the capture's rights (``_record_live``), on success and on failure.
STATEMENT_CHECK_FILE = "statement_check.json"
#: The live rights are read at most once a window per capture, and at least once a window for a published clip.
LIVE_STATEMENT_RECHECK_SECONDS = 6 * 3600
#: A clip shared on a live grant stays shared while the last grant is younger than this, so a WebApp that does
#: not answer one re-read does not take every clip back; a definite "no" takes it back at once.
LIVE_GRANT_LIFETIME_SECONDS = 2 * LIVE_STATEMENT_RECHECK_SECONDS
MAX_CLIP_BYTES = 64 * 1024 * 1024
MAX_LONG_EDGE = 1280
MAX_FPS = 30
#: 3.5 Mbit/s at 1280 px: a 120 s clip is at most ~50 MiB; longer clips get a lower ceiling.
TARGET_VIDEO_KBPS = 3500
#: Below this the picture is not worth sending: such a clip (over ~30 minutes) is refused.
MIN_VIDEO_KBPS = 250
#: The share of the cap the bitrate aims at, leaving room for the container and rate overshoot.
_SIZE_HEADROOM = 0.85
ENCODER = {"codec": "libx264", "profile": "high", "preset": "veryfast", "crf": 23, "pix_fmt": "yuv420p",
           "threads": 2, "scale_flags": "lanczos"}
#: v2: closed captions and every SEI unit stripped, frames decoded and proven free of side data, the frame
#: rate capped at exactly ``MAX_FPS``, the output size bounded while encoding. A v1 clip is rebuilt.
BUILDER_REVISION = "website_company_clip_builder.v2"
OVER_CAP = "website_company_clip_over_size_cap"
ENCODE_TIMED_OUT = "website_company_clip_encode_timed_out"
#: One clip's whole build (both encodes and the checks). A 30-minute source, the longest the bitrate floor
#: allows, encodes in about 12 minutes at the measured 0.4x real time on the host's two threads.
ENCODE_TIMEOUT_SECONDS = 900
#: One pass of the clip tick starts no new capture after this. The pass runs last in the Atlas preview unit
#: (``TimeoutStartSec=2h``) after up to an hour of continued Atlas steps that hold paid work, so it is kept to
#: at most this plus one build (900 + 900 s, about 30 minutes): the unit still ends well inside its 2 hours.
TICK_BUDGET_SECONDS = 900
#: The highest frame rate a clip may carry, as the contract states it.
MAX_CLIP_FPS = MAX_FPS + 0.0001
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
#: The only tags a clip may carry: what the MP4 muxer and encoder write themselves. Anything else
#: (a phone's location, make, model, title, comment, creation time, QuickTime keys) came from the
#: source and refuses the clip; an allow-list, so a tag nobody thought of is refused too.
_FORMAT_TAGS = {"major_brand": re.compile(r"[a-z0-9 ]{4}"), "minor_version": re.compile(r"[0-9]{1,10}"),
                "compatible_brands": re.compile(r"(?:[a-z0-9 ]{4}){1,16}"), "encoder": re.compile(r"Lavf[0-9.]+")}
_STREAM_TAGS = {"language": re.compile(r"und"), "handler_name": re.compile(r"VideoHandler"),
                "vendor_id": re.compile(r"\[0\]\[0\]\[0\]\[0\]"), "encoder": re.compile(r"Lavc[0-9.]+ libx264")}
#: A WebApp refusal that means the site's rights no longer allow the clip.
_RIGHTS_REFUSALS = re.compile(r"(source_revoked|consent_revoked|consent_expired|website_capture_withdrawn"
                              r"|capture_revoked_or_expired|website_company_clip_not_permitted)\b")
_NOT_PERMITTED = "website_company_clip_not_permitted"
_RIGHTS_REFUSED = "website_company_clip_rights_refused"
#: Beside the lock: a raw upload's digest, kept before any manifest exists (keyed by path, size and mtime).
RAW_DIGEST_FILE = "raw_upload_digest.json"
#: An after-encode proof that refuses a clip is recorded for that source and builder revision and tried
#: again only after ``LIVE_STATEMENT_RECHECK_SECONDS``, as a timeout is: no full encode every pass.
_OUTPUT_REFUSAL = re.compile(r"website_company_clip_output_[a-z_]+")


class CompanyClipUnavailable(ValueError):
    """No archived company clip this caller may read; never says why."""


class CompanyClipBusy(RuntimeError):
    """Another pass (or withdrawal deletion) holds this capture's clip lock."""


class RangeNotSatisfiable(ValueError):
    """A byte range that starts at or past the end of the clip; ``size`` is the clip's length."""

    def __init__(self, size: int):
        super().__init__("website_company_clip_range_not_satisfiable")
        self.size = int(size)


def clip_filename(capture_id: str, request_id: str) -> str:
    """The archive name: the walkthrough's own, or a continuation scoped to its job."""
    if not capture_name_matches_request(request_id, capture_id):
        raise CompanyClipUnavailable("website_company_clip_unavailable")
    return f"{capture_id}.mp4" if capture_id.startswith("walkthrough-") else f"{request_id}-{capture_id}.mp4"


def _read(path: Path) -> dict[str, Any]:
    try:
        value = {} if path.is_symlink() else json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _identity(capture_root: Path) -> dict[str, str]:
    root = Path(capture_root)
    scene_id, capture_id = root.parent.parent.name, root.name
    request_id = scene_id.removeprefix("site-")
    if (root.parent.name != "captures" or not scene_id.startswith("site-")
            or not capture_name_matches_request(request_id, capture_id)):
        raise ValueError("website_company_clip_capture_identity_invalid")
    return {"request_id": request_id, "scene_id": scene_id, "capture_id": capture_id}


def _sha256_file(path: Path) -> str:
    from .website_source_video import _sha256
    return _sha256(path)


#: The raw upload a capture's ``raw/manifest.json`` names: ``scenes/<site>/captures/<capture>/raw/<file>``.
RAW_VIDEO_SUFFIXES = frozenset({".mov", ".mp4", ".m4v"})
_RAW_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}")


def raw_upload_source(capture_root: Path) -> dict[str, Any] | None:
    """The raw upload ``raw/manifest.json`` names (``video_uri``), a regular file in this capture's ``raw/``.

    Only for a capture handed off before normalization existed: ``website_source_video``'s own test, a
    derived folder (``DERIVED_ARTIFACT_DIRS``) and no record. A new capture waits for normalization.
    Cheap: no read of the video. Its digest and whether it is a readable video are decided under the lock.
    """
    from .website_source_video import DERIVED_ARTIFACT_DIRS
    root = Path(capture_root)
    pipeline = root / "pipeline"
    if pipeline.is_symlink() or not any((pipeline / name).is_dir() and not (pipeline / name).is_symlink()
                                        for name in DERIVED_ARTIFACT_DIRS):
        return None
    named = _read(root / "raw" / "manifest.json").get("video_uri")
    prefix = f"scenes/{root.parent.parent.name}/captures/{root.name}/raw/"
    name = named[len(prefix):] if isinstance(named, str) and named.startswith(prefix) else ""
    if not _RAW_NAME.fullmatch(name) or Path(name).suffix.lower() not in RAW_VIDEO_SUFFIXES:
        return None
    path = root / "raw" / name
    try:
        info = path.lstat()
        inside = not (root / "raw").is_symlink() and path.resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError):
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0 or not inside:
        return None
    return {"path": path, "relative": f"raw/{name}", "source": "raw_upload", "kind": "raw_upload",
            "size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def _raw_upload_digest(source: Mapping[str, Any], manifest: Mapping[str, Any], directory: Path) -> str | None:
    """The raw upload's digest, or None when it is not a readable video.

    Read once: a pass reuses the digest the manifest kept, or before any manifest exists the one kept in
    ``RAW_DIGEST_FILE`` beside the lock, for the same file, size and modification time, so no pass reads a
    large upload in full again.
    """
    from .website_source_video import probe_video

    def same(kept: Any) -> bool:
        return (isinstance(kept, Mapping) and all(kept.get(key) == source[key]
                                                  for key in ("relative", "size_bytes", "mtime_ns"))
                and bool(_DIGEST.fullmatch(str(kept.get("digest") or ""))))
    for kept in ((manifest.get("source_video") if manifest.get("source") == "raw_upload" else None),
                 _read(directory / RAW_DIGEST_FILE)):
        if same(kept):
            return kept["digest"]
    try:
        probe_video(Path(source["path"]))
        digest = _sha256_file(Path(source["path"]))
    except (subprocess.SubprocessError, OSError, ValueError):
        return None
    write_json(directory / RAW_DIGEST_FILE, {key: source[key] for key in ("relative", "size_bytes", "mtime_ns")}
               | {"digest": digest})
    return digest


def clip_source(capture_root: Path) -> dict[str, Any] | None:
    """The clip website preparation reads, as its normalization record names it; without a record, for a
    capture handed off before normalization existed, the raw upload the capture's manifest names; or None.

    A record that exists but cannot be used is None: it never falls back to the raw upload. A raw upload's
    ``digest`` is not set here (see ``_raw_upload_digest``).
    """
    from .website_source_video import RECORD_FILE, normalization_root, read_record
    root = Path(capture_root)
    if not os.path.lexists(normalization_root(root) / RECORD_FILE):
        return raw_upload_source(root)
    try:
        record = read_record(root)
    except (OSError, ValueError):
        return None
    if not record:
        return None
    chosen = record.get("normalized") if record.get("normalization_ran") is True else record.get("original")
    relative = record.get("preparation_video")
    if not isinstance(chosen, Mapping) or not isinstance(relative, str) or chosen.get("path") != relative:
        return None
    path = Path(relative) if Path(relative).is_absolute() else root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        return None
    if not _DIGEST.fullmatch(str(chosen.get("sha256") or "")) or path.stat().st_size != chosen.get("size_bytes"):
        return None
    return {"path": path, "relative": relative, "digest": chosen["sha256"], "source": "normalized",
            "kind": "normalized" if record.get("normalization_ran") is True else "within_bounds_upload",
            "duration_seconds": chosen.get("duration_seconds"), "fps": chosen.get("fps"),
            "width": chosen.get("width"), "height": chosen.get("height")}


def clip_revision(source: Mapping[str, Any], *, max_bytes: int = MAX_CLIP_BYTES) -> str:
    return canonical_digest({"builder": BUILDER_REVISION, "format": FORMAT, "max_bytes": int(max_bytes),
                             "max_long_edge": MAX_LONG_EDGE, "max_fps": MAX_FPS, "source_digest": source["digest"]})


#: Read at call time, so a test can point it at a temporary list.
OWNED_CAPTURES_LIST: Path | None = None


def recorded_video_sha256(capture_root: Path) -> str | None:
    """The uploaded video's digest as normalization recorded it (``original.sha256``)."""
    from .website_source_video import read_record
    try:
        record = read_record(Path(capture_root)) or {}
    except (OSError, ValueError):
        return None
    digest = (record.get("original") or {}).get("sha256")
    return digest if isinstance(digest, str) and _DIGEST.fullmatch(digest) else None


def upload_sha256(capture_root: Path, source: Mapping[str, Any]) -> str | None:
    """The digest of the upload the clip is made from, which Blueprint's own list names by prefix: the
    original normalization recorded, or for a capture with no record the raw upload itself."""
    return source.get("digest") if source.get("source") == "raw_upload" else recorded_video_sha256(capture_root)


def owned_basis_holds(capture_id: str, video_sha256: str | None) -> bool:
    """The capture is on Blueprint's own list and its row's prefix names this upload. A local read, made
    every pass and on every route read; an unreadable list holds for nothing."""
    return sharing_basis({"capture_id": capture_id}, video_sha256=video_sha256,
                         owned_list=OWNED_CAPTURES_LIST) is not None


def _on_owned_list(capture_id: str) -> bool:
    return capture_id in owned_captures(OWNED_CAPTURES_LIST)[0]


def recorded_statement_version(capture_root: Path) -> str | None:
    """The statement version the capture's files kept at submission (a snapshot)."""
    root = Path(capture_root)
    for path in (root / "pipeline" / "website_task_context.json", root / "raw" / "manifest.json"):
        rights = _read(path).get("capture_rights")
        version = rights.get("statement_version") if isinstance(rights, Mapping) else None
        if isinstance(version, str):
            return version
    return None


def _live_rights(identity: Mapping[str, str]) -> Mapping[str, Any]:
    """The WebApp's current ``capture_rights``, projected from the request's current consent."""
    from .website_task_context import load_current_website_task_context
    context = load_current_website_task_context(request_id=identity["request_id"], scene_id=identity["scene_id"],
                                                capture_id=identity["capture_id"], purpose="scene_preparation")
    rights = context.get("capture_rights")
    return rights if isinstance(rights, Mapping) else {}


def _record_live(root: Path, identity: Mapping[str, str], check_rights: Callable[..., Mapping[str, Any]],
                 now: float) -> dict[str, Any]:
    """Read the live rights once and record the answer, a failure included.

    ``outcome``: ``covers_sharing`` (consent granted, sharing statement), ``not_sharing`` (consent granted,
    an older statement), ``rights_refused`` (consent not granted, or the WebApp said revoked: definite) or
    ``unanswered`` (no answer: the clip's state does not change on it). ``granted_at_epoch`` is the last
    time the answer covered sharing; a definite answer that does not clears it.
    """
    previous = _read(root.joinpath(*DIRECTORY) / STATEMENT_CHECK_FILE)
    try:
        rights = check_rights(identity)
    except Exception as error:  # noqa: BLE001 - classified: a definite refusal, or no answer
        rights, failure = None, error
    if rights is None:
        outcome = "rights_refused" if _RIGHTS_REFUSALS.search(str(failure)) else "unanswered"
        version, fingerprint, reason = None, None, _reason(failure)
    else:
        version = rights.get("statement_version")
        version = version if isinstance(version, str) else None
        granted = rights.get("consent_revoked") is False and rights.get("consent_status") == "granted"
        outcome = ("rights_refused" if not granted else "covers_sharing" if covers_sharing(version)
                   else "not_sharing")
        fingerprint = canonical_digest({"statement_version": version, "consent_status": rights.get("consent_status"),
                                        "consent_revoked": rights.get("consent_revoked")})
        reason = {"rights_refused": _RIGHTS_REFUSED, "not_sharing": _NOT_PERMITTED}.get(outcome)
    kept = previous.get("granted_at_epoch") if outcome == "unanswered" else None
    value = {"checked_at_epoch": int(now), "outcome": outcome, "reason": reason, "statement_version": version,
             "rights_fingerprint": fingerprint,
             "granted_at_epoch": int(now) if outcome == "covers_sharing" else kept if isinstance(kept, int) else None}
    write_json(root.joinpath(*DIRECTORY) / STATEMENT_CHECK_FILE, value)
    return value


def _live_due(check: Mapping[str, Any], now: float, root: Path) -> bool:
    """A live read is due: none yet, the window has passed, or the website asked for a recheck since
    (``permission_recheck``: a job's permission changed)."""
    checked = check.get("checked_at_epoch")
    return (not isinstance(checked, int) or not 0 <= now - checked < LIVE_STATEMENT_RECHECK_SECONDS
            or recheck_requested_since(root, checked))


def live_grant_holds(check: Mapping[str, Any], now: float | None = None) -> bool:
    """The last live read granted sharing and no definite answer has said otherwise since; with ``now``,
    also that the grant is younger than ``LIVE_GRANT_LIFETIME_SECONDS``."""
    granted = check.get("granted_at_epoch")
    if check.get("outcome") not in ("covers_sharing", "unanswered") or not isinstance(granted, int):
        return False
    return now is None or 0 <= now - granted < LIVE_GRANT_LIFETIME_SECONDS


def _may_be_permitted(root: Path, identity: Mapping[str, str], now: float) -> bool:
    """No network: the snapshot covers sharing, the capture is on Blueprint's own list, the last live read
    granted sharing, or a live read is due. Otherwise a capture gets no directory, file or request."""
    if covers_sharing(recorded_statement_version(root)) or _on_owned_list(identity["capture_id"]):
        return True
    check = _read(root.joinpath(*DIRECTORY) / STATEMENT_CHECK_FILE)
    return _live_due(check, now, root) or live_grant_holds(check, now)


def _decided_basis(check: Mapping[str, Any], owned: bool) -> dict[str, Any] | None:
    """The rule that lets the clip be shared now: a live grant, else Blueprint's own capture, else None."""
    if check.get("outcome") == "covers_sharing":
        return {"permission_basis": "site_sharing_statement", "statement_version": check.get("statement_version")}
    if owned and check.get("outcome") == "not_sharing":  # consent definitely granted, Blueprint's own video
        return {"permission_basis": "blueprint_owned_capture", "statement_version": check.get("statement_version")}
    return None


def delivery_gate_allows(capture_root: Path) -> bool:
    """The serve-time takedown gate (consent re-read live, no open takedown) and no withdrawal."""
    from .consent_takedown import evaluate_delivery_time_takedown_gate
    root = Path(capture_root)
    if capture_withdrawn(root):
        return False
    gate = evaluate_delivery_time_takedown_gate(capture_root=root, surface="website_company_clip")
    return gate.get("status") == "allowed"


def target_dimensions(width: int, height: int) -> tuple[int, int]:
    """Long edge at most ``MAX_LONG_EDGE``, same aspect, even sides; never larger than the source."""
    def even(side: float) -> int:
        return max(2, int(round(side / 2)) * 2)
    long_edge = max(width, height)
    scale = min(1.0, MAX_LONG_EDGE / long_edge)
    return even(width * scale), even(height * scale)


def video_kbps(duration_seconds: Any, *, max_bytes: int = MAX_CLIP_BYTES) -> int | None:
    """The bitrate ceiling for a clip this long, or None when no worthwhile rate fits the cap."""
    if not isinstance(duration_seconds, (int, float)) or not math.isfinite(duration_seconds) or duration_seconds <= 0:
        return TARGET_VIDEO_KBPS
    fitting = int(max_bytes * 8 * _SIZE_HEADROOM / duration_seconds / 1000)
    rate = min(TARGET_VIDEO_KBPS, fitting)
    return rate if rate >= MIN_VIDEO_KBPS else None


def probe_streams(path: Path) -> dict[str, Any]:
    """Every stream's type and codec, the video's size and rate, and the container's duration."""
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,width,height,avg_frame_rate:stream_tags:format=duration,format_name"
         ":format_tags", "-of", "json",
         str(path)], check=True, capture_output=True, text=True, timeout=600)
    value = json.loads(completed.stdout or "{}")
    streams = [row for row in value.get("streams") or [] if isinstance(row, dict)]
    duration = (value.get("format") or {}).get("duration")
    try:
        seconds = round(float(duration), 3)
    except (TypeError, ValueError):
        seconds = None
    return {"streams": streams, "duration_seconds": seconds if seconds is not None and math.isfinite(seconds)
            and 0 <= seconds <= 86_400 else None, "format_name": (value.get("format") or {}).get("format_name"),
            "format_tags": (value.get("format") or {}).get("tags") or {}}


def metadata_clean(probe: Mapping[str, Any]) -> bool:
    """Every format and stream tag is one the muxer or encoder wrote, with the value it writes."""
    def clean(tags: Any, allowed: Mapping[str, re.Pattern[str]]) -> bool:
        return isinstance(tags, Mapping) and all(
            key in allowed and isinstance(value, str) and allowed[key].fullmatch(value) for key, value in tags.items())
    return clean(probe.get("format_tags") or {}, _FORMAT_TAGS) and all(
        clean(row.get("tags") or {}, _STREAM_TAGS) for row in probe.get("streams") or [])


def moov_before_mdat(path: Path) -> bool:
    """``+faststart``: the movie header comes before the media data, so a browser can seek at once."""
    order = []
    with Path(path).open("rb") as stream:
        size_total, position = os.fstat(stream.fileno()).st_size, 0
        while position + 8 <= size_total and len(order) < 64:
            stream.seek(position)
            size, kind = struct.unpack(">I4s", stream.read(8))
            if size == 1:
                size = struct.unpack(">Q", stream.read(8))[0]
            elif size == 0:
                size = size_total - position
            if size < 8:
                return False
            order.append(kind)
            position += size
    return b"moov" in order and b"mdat" in order and order.index(b"moov") < order.index(b"mdat")


def _encode(source: Path, target: Path, *, width: int, height: int, fps_cap: bool, kbps: int, max_bytes: int,
            timeout: float) -> None:
    filters = [f"scale={width}:{height}:flags={ENCODER['scale_flags']}"] + ([f"fps={MAX_FPS}"] if fps_cap else [])
    # Video stream 0 only: -an/-sn/-dn drop every other stream; no tag or chapter of the source is copied.
    # -a53cc 0: libx264 would re-embed the source's closed captions (on by default). The bitstream filter
    # drops every SEI unit (type 6), x264's own settings string included, so a decoded frame carries no side
    # data at all and ``frames_clean`` can demand exactly that. -fs stops the file just past the cap, so a
    # partial file never grows beyond it; such a file is then over the cap and refused.
    command = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-threads", str(ENCODER["threads"]), "-i", str(source),
               "-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1", "-map_metadata:s:v", "-1",
               "-map_chapters", "-1",
               "-filter_threads", "1", "-vf", ",".join(filters),
               "-c:v", ENCODER["codec"], "-profile:v", ENCODER["profile"], "-preset", ENCODER["preset"],
               "-crf", str(ENCODER["crf"]), "-maxrate", f"{kbps}k", "-bufsize", f"{2 * kbps}k", "-a53cc", "0",
               "-pix_fmt", ENCODER["pix_fmt"], "-threads", str(ENCODER["threads"]),
               "-bsf:v", "filter_units=remove_types=6", "-fs", str(int(max_bytes) + 1),
               "-movflags", "+faststart", "-f", "mp4", str(target)]
    subprocess.run(command, check=True, capture_output=True, timeout=max(1.0, timeout))


def frames_clean(path: Path, *, timeout: float = ENCODE_TIMEOUT_SECONDS) -> bool:
    """Every frame decoded, and not one carries side data (captions, SEI user data, anything else)."""
    completed = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
                                "-show_entries", "frame=side_data_list", "-of", "json", str(path)],
                               check=True, capture_output=True, text=True, timeout=max(1.0, timeout))
    frames = json.loads(completed.stdout or "{}").get("frames")
    return isinstance(frames, list) and bool(frames) and not any(
        not isinstance(frame, dict) or frame.get("side_data_list") for frame in frames)


def sei_units_present(path: Path, *, timeout: float = ENCODE_TIMEOUT_SECONDS) -> bool:
    """Any SEI unit (NAL type 6, where captions and user data travel) left in the clip's H.264 stream.

    The stream is copied out in Annex B form and split at its start codes; emulation prevention means a
    start code never occurs inside a unit, so this reads unit headers only, never chance bytes in the
    picture data. The encode drops every SEI unit (``filter_units``), so the answer must be no.
    """
    completed = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0", "-c", "copy",
                                "-bsf:v", "h264_mp4toannexb", "-f", "h264", "-"],
                               check=True, capture_output=True, timeout=max(1.0, timeout))
    stream, position = completed.stdout, 0
    if not stream:
        return True  # nothing to read is not a proof
    while (found := stream.find(b"\x00\x00\x01", position)) >= 0:
        if found + 3 < len(stream) and stream[found + 3] & 0x1F == 6:
            return True
        position = found + 3
    return False


def build_clip(source: Path, output: Path, *, max_bytes: int = MAX_CLIP_BYTES,
               timeout_seconds: float = ENCODE_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Encode ``source`` (only read) to ``output``, proven video-only, caption-free, seekable and under the cap.

    One lower-rate retry when the first encode lands over the cap; then ``OVER_CAP``. The whole build has
    ``timeout_seconds``; past it, ``ENCODE_TIMED_OUT``.
    """
    from .website_source_video import _rate, probe_video
    deadline = time.monotonic() + timeout_seconds
    try:
        probe = probe_video(source)
    except (subprocess.SubprocessError, OSError, ValueError):
        raise ValueError("website_company_clip_source_unreadable") from None
    kbps = video_kbps(probe.get("duration_seconds"), max_bytes=max_bytes)
    if kbps is None:
        raise ValueError(OVER_CAP)
    width, height = target_dimensions(probe["width"], probe["height"])
    partial = output.with_name(output.name + ".partial")
    try:
        for attempt in range(2):
            partial.unlink(missing_ok=True)
            try:
                _encode(source, partial, width=width, height=height, fps_cap=probe["fps"] > MAX_FPS, kbps=kbps,
                        max_bytes=max_bytes, timeout=deadline - time.monotonic())
            except subprocess.TimeoutExpired:
                raise ValueError(ENCODE_TIMED_OUT) from None
            except subprocess.SubprocessError:
                raise ValueError("website_company_clip_encode_failed") from None
            size = partial.stat().st_size
            if size <= max_bytes:
                break
            kbps = int(kbps * max_bytes * _SIZE_HEADROOM / size)
            if attempt or kbps < MIN_VIDEO_KBPS:
                raise ValueError(OVER_CAP)
        from .capture_audio_minimization import probe_audio_stream_count
        if probe_audio_stream_count(partial) != 0:  # the packet tool's own proof: ffprobe counts no audio
            raise ValueError("website_company_clip_output_audio_not_proven_absent")
        streams = probe_streams(partial)
        video = [row for row in streams["streams"] if row.get("codec_type") == "video"]
        if (len(streams["streams"]) != 1 or len(video) != 1 or video[0].get("codec_name") != "h264"
                or max(video[0].get("width") or 0, video[0].get("height") or 0) > MAX_LONG_EDGE):
            raise ValueError("website_company_clip_output_not_video_only_h264")
        if not metadata_clean(streams):
            raise ValueError("website_company_clip_output_metadata_not_clean")
        if not moov_before_mdat(partial):
            raise ValueError("website_company_clip_output_not_faststart")
        fps = _rate(video[0].get("avg_frame_rate"))
        if fps is None or fps > MAX_CLIP_FPS:
            raise ValueError("website_company_clip_output_frame_rate_over_cap")
        source_seconds, seconds = probe.get("duration_seconds"), streams["duration_seconds"]
        if isinstance(source_seconds, (int, float)) and source_seconds > 0 and (
                seconds is None or seconds < source_seconds - max(1.0, 0.02 * source_seconds)):
            raise ValueError("website_company_clip_output_truncated")
        # Two separate proofs, each on its own: every frame decoded carries no side data, and no SEI unit is
        # left in the stream.
        try:
            clean = frames_clean(partial, timeout=deadline - time.monotonic())
            sei = clean and sei_units_present(partial, timeout=deadline - time.monotonic())
        except subprocess.TimeoutExpired:
            raise ValueError(ENCODE_TIMED_OUT) from None
        except (subprocess.SubprocessError, ValueError):
            clean, sei = False, True
        if not clean:
            raise ValueError("website_company_clip_output_frame_side_data")
        if sei:
            raise ValueError("website_company_clip_output_sei_units_left")
        os.replace(partial, output)
    finally:
        partial.unlink(missing_ok=True)
    return {"format": FORMAT, "content_type": CONTENT_TYPE, "digest": _sha256_file(output), "size_bytes": size,
            "width": int(video[0]["width"]), "height": int(video[0]["height"]),
            "fps": round(fps, 3), "duration_seconds": streams["duration_seconds"],
            "has_audio": False, "video_kbps_ceiling": kbps, "max_bytes": int(max_bytes)}


def clip_record(*, identity: Mapping[str, str], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """What the WebApp may store: the clip's identity, never a path, object key or person."""
    built, basis = manifest["clip"], manifest["permission"]
    value = {"schema_version": CLIP_SCHEMA, "capture_id": identity["capture_id"],
             "clip_revision": manifest["clip_revision"], "state": "ready", "format": built["format"],
             "content_type": built["content_type"], "digest": built["digest"], "size_bytes": built["size_bytes"],
             "width": built["width"], "height": built["height"], "fps": built["fps"],
             "duration_seconds": built["duration_seconds"], "has_audio": False,
             "permission_basis": basis["permission_basis"], "statement_version": basis["statement_version"],
             "updated_at_epoch": int(manifest["built_at_epoch"])}
    value["clip_digest"] = cross_runtime_canonical_digest(value)
    return value


@contextmanager
def clip_lock(capture_root: Path) -> Iterator[None]:
    directory = Path(capture_root).joinpath(*DIRECTORY)
    if not directory.parent.is_dir() or directory.parent.is_symlink():
        raise ValueError("website_company_clip_capture_changed")  # a retired capture is never re-created
    directory.mkdir(mode=0o750, exist_ok=True)
    descriptor = os.open(directory / LOCK_FILE, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(descriptor, "r+") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError("website_company_clip_lock_unsafe")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CompanyClipBusy("website_company_clip_busy") from None
        yield


def _archive(path: Path, *, identity: Mapping[str, str]) -> dict[str, Any]:
    """The private artifact store only, never the legacy fallback, with its bytes proven."""
    from .task_evaluation_configured_scene_object_store import (
        publish_configured_scene_artifact, verify_dedicated_artifact_store)
    verify_dedicated_artifact_store()
    filename = clip_filename(identity["capture_id"], identity["request_id"])
    reference = publish_configured_scene_artifact(path=path, artifact_kind=ARTIFACT_KIND, filename=filename)
    return {"uri": reference["uri"], "digest": reference["digest"], "size_bytes": reference["size_bytes"],
            "filename": filename}


def _retract(uri: str) -> None:
    from .website_scene_preview import _retract as retract_object
    retract_object(uri)


def _deliver(record: Mapping[str, Any], *, identity: Mapping[str, str]) -> None:
    from .website_task_context import website_webapp_request
    acceptance = website_webapp_request(capture_id=identity["capture_id"], operation="company-clip", payload={
        "request_id": identity["request_id"], "scene_id": identity["scene_id"], "clip": dict(record)})
    if (acceptance.get("schema_version") != ACCEPTANCE_SCHEMA or acceptance.get("accepted") is not True
            or acceptance.get("clip_revision") != record["clip_revision"]
            or acceptance.get("clip_digest") != record["clip_digest"]):
        raise ValueError("website_company_clip_acceptance_invalid")


def _reason(error: BaseException) -> str:
    text = str(error)
    return text if re.fullmatch(r"[a-z][a-z0-9_.:-]{0,159}", text) else type(error).__name__


def _hidden(archive: Mapping[str, Any]) -> bool:
    return archive.get("retracted") is True or archive.get("retraction") == "pending"


def _take_back(manifest: dict[str, Any], manifest_path: Path, retract: Callable[[str], None]) -> bool:
    """Record the retraction before deleting, so a failed deletion is retried, never forgotten."""
    from .website_scene_preview import _finish_retraction
    archive = manifest.get("archive")
    if not isinstance(archive, Mapping) or not isinstance(archive.get("uri"), str) or archive.get("retracted"):
        return True
    manifest["archive"] = {**archive, "retraction": "pending"}
    write_json(manifest_path, manifest)
    return _finish_retraction(manifest, manifest_path, retract)


def _retraction_pending(manifest: Mapping[str, Any]) -> bool:
    archive = manifest.get("archive")
    return isinstance(archive, Mapping) and archive.get("retraction") == "pending" and isinstance(archive.get("uri"), str)


def _current(manifest: Mapping[str, Any], revision: str, clip: Path) -> bool:
    built, archive = manifest.get("clip") or {}, manifest.get("archive") or {}
    return (manifest.get("schema_version") == MANIFEST_SCHEMA and manifest.get("clip_revision") == revision
            and not clip.is_symlink() and clip.is_file() and clip.stat().st_size == built.get("size_bytes")
            and archive.get("digest") == built.get("digest") and not _hidden(archive)
            and (manifest.get("delivered") or {}).get("clip_digest") == manifest.get("clip_digest"))


def _take_back_if_archived(root: Path, manifest_path: Path, retract: Callable[[str], None]) -> None:
    """A capture that may no longer be shared loses any archived copy (only one with a manifest)."""
    if not manifest_path.is_file():
        return
    with clip_lock(root):
        manifest = _read(manifest_path)
        if manifest:
            _take_back(manifest, manifest_path, retract)


def publish_company_clip(capture_root: Path, *, now: Callable[[], float] = time.time,
                         check_rights: Callable[[Mapping[str, str]], Mapping[str, Any]] = _live_rights,
                         archive: Callable[..., dict[str, Any]] = _archive,
                         retract: Callable[[str], None] = _retract,
                         deliver: Callable[..., None] = _deliver,
                         max_bytes: int = MAX_CLIP_BYTES) -> dict[str, Any]:
    """One idempotent pass for one capture: make, archive and deliver what is not yet current, and take
    back what may no longer be shared.

    Returns ``{"status": ...}``: ``current``, ``published``, ``not_permitted`` (no basis to share; any
    archived copy is taken back), ``no_source`` (no usable source video yet), ``over_size_cap``,
    ``encode_timed_out`` (retried after ``LIVE_STATEMENT_RECHECK_SECONDS``), ``withdrawn``, ``refused``
    (rights, the takedown gate or the WebApp, with a reason), ``busy`` or ``failed`` (retried by the next
    pass). It never raises and never touches the capture's job.
    """
    try:
        return _publish(Path(capture_root), now, check_rights, archive, retract, deliver, max_bytes)
    except CompanyClipBusy:
        return {"status": "busy"}
    except Exception as error:  # noqa: BLE001 - one capture never stops a pass; retried next time
        logger.warning("website_company_clip_failed capture=%s type=%s", Path(capture_root).name,
                       type(error).__name__)
        return {"status": "failed", "reason": _reason(error)}


def _refused_live(check: Mapping[str, Any], manifest: dict[str, Any], manifest_path: Path,
                  retract: Callable[[str], None]) -> dict[str, Any] | None:
    """The answer to a live read that stops sharing: a definite refusal or no basis takes any archived copy
    back; no answer refuses this pass only. None when the answer allows nothing to be said yet."""
    if check["outcome"] == "rights_refused":
        if manifest:
            _take_back(manifest, manifest_path, retract)
        return {"status": "refused", "reason": check.get("reason") or _RIGHTS_REFUSED}
    if check["outcome"] == "unanswered":
        return {"status": "refused", "reason": check.get("reason") or "website_company_clip_rights_unanswered"}
    return None


def _publish(root, now, check_rights, archive, retract, deliver, max_bytes):
    identity = _identity(root)
    directory = root.joinpath(*DIRECTORY)
    manifest_path, clip_path = directory / MANIFEST_FILE, directory / CLIP_FILE
    if _retraction_pending(_read(manifest_path)):
        with clip_lock(root):
            manifest = _read(manifest_path)
            if _retraction_pending(manifest) and not _take_back(manifest, manifest_path, retract):
                return {"status": "failed", "reason": "website_company_clip_retraction_pending"}
    if capture_withdrawn(root):
        return {"status": "withdrawn"}
    if not delivery_gate_allows(root):
        _take_back_if_archived(root, manifest_path, retract)
        return {"status": "refused", "reason": "website_company_clip_takedown_gate_closed"}
    # Read-only first: a capture that may not be shared gets no directory, file or request
    # until its live statement is due a recheck; any copy it still has is taken back.
    if not _may_be_permitted(root, identity, now()):
        _take_back_if_archived(root, manifest_path, retract)
        return {"status": "not_permitted"}
    if clip_source(root) is None:
        return {"status": "no_source"}
    with clip_lock(root):
        source = clip_source(root)
        if source is None:
            return {"status": "no_source"}
        manifest = _read(manifest_path)
        if source["source"] == "raw_upload":
            # Decided before any rights read: a missing or broken upload is no source, never a failure.
            source["digest"] = _raw_upload_digest(source, manifest, directory)
            if source["digest"] is None:
                return {"status": "no_source"}
        revision = clip_revision(source, max_bytes=max_bytes)
        upload = upload_sha256(root, source)
        owned = owned_basis_holds(identity["capture_id"], upload)
        check = _read(directory / STATEMENT_CHECK_FILE)
        if _current(manifest, revision, clip_path):
            # A published clip keeps its basis or goes: Blueprint's list every pass (a local read), the live
            # rights at least once a window. A definite "no" takes the clip back at once.
            if _live_due(check, now(), root):
                check = _record_live(root, identity, check_rights, now())
                if check["outcome"] == "rights_refused":
                    return _refused_live(check, manifest, manifest_path, retract)
            if owned or live_grant_holds(check, now()):
                return {"status": "current", "clip_revision": revision}
            _take_back(manifest, manifest_path, retract)
            return {"status": "not_permitted"}
        if manifest.get("clip_revision") == revision and manifest.get("refused") == OVER_CAP:
            return {"status": "over_size_cap"}
        refused, refused_at = manifest.get("refused"), manifest.get("refused_at_epoch")
        if (manifest.get("clip_revision") == revision and isinstance(refused, str) and isinstance(refused_at, int)
                and (refused == ENCODE_TIMED_OUT or _OUTPUT_REFUSAL.fullmatch(refused))
                and 0 <= now() - refused_at < LIVE_STATEMENT_RECHECK_SECONDS):
            return ({"status": "encode_timed_out"} if refused == ENCODE_TIMED_OUT
                    else {"status": "failed", "reason": refused})
        # A definite "no" from this window's live read stands until the window passes: no request every pass.
        if not owned and not _live_due(check, now(), root) and check.get("outcome") in ("not_sharing", "rights_refused"):
            if manifest:
                _take_back(manifest, manifest_path, retract)
            return ({"status": "not_permitted"} if check["outcome"] == "not_sharing"
                    else {"status": "refused", "reason": check.get("reason") or _RIGHTS_REFUSED})
        # A build or a delivery: always on a fresh live answer, recorded whatever it says.
        check = _record_live(root, identity, check_rights, now())
        stopped = _refused_live(check, manifest, manifest_path, retract)
        if stopped is not None:
            return stopped
        basis = _decided_basis(check, owned)
        if basis is None:
            if manifest:
                _take_back(manifest, manifest_path, retract)
            return {"status": "not_permitted"}
        # The WebApp refused this clip: nothing is archived again until the live rights differ from those at the
        # refusal or a window passes. A recheck only makes the live read due; unchanged rights keep the refusal.
        memory = manifest.get("refused_by_webapp")
        if (isinstance(memory, Mapping) and memory.get("clip_revision") == revision
                and memory.get("rights_fingerprint") == check.get("rights_fingerprint")
                and isinstance(memory.get("at_epoch"), int)
                and 0 <= now() - memory["at_epoch"] < LIVE_STATEMENT_RECHECK_SECONDS):
            return {"status": "refused", "reason": str(memory.get("reason"))}
        built = manifest.get("clip") or {}
        if (manifest.get("clip_revision") != revision or clip_path.is_symlink() or not clip_path.is_file()
                or _sha256_file(clip_path) != built.get("digest")):
            if _sha256_file(Path(source["path"])) != source["digest"]:
                return {"status": "failed", "reason": "website_company_clip_source_changed"}
            if manifest.get("archive") and not _take_back(manifest, manifest_path, retract):
                return {"status": "failed", "reason": "website_company_clip_retraction_pending"}
            base = {"schema_version": MANIFEST_SCHEMA, **identity, "clip_revision": revision,
                    "source": source["source"],
                    "source_video": {key: source[key] for key in ("kind", "digest", "relative", "size_bytes",
                                                                  "mtime_ns") if key in source},
                    "upload_sha256": upload, "permission": basis, "built_at_epoch": int(now()), "archive": None,
                    "delivered": None}
            try:
                built = build_clip(Path(source["path"]), clip_path, max_bytes=max_bytes)
            except ValueError as error:
                code = str(error)
                if code not in (OVER_CAP, ENCODE_TIMED_OUT) and not _OUTPUT_REFUSAL.fullmatch(code):
                    raise
                clip_path.unlink(missing_ok=True)
                write_json(manifest_path, {**base, "clip": None, "refused": code, "refused_at_epoch": int(now())})
                logger.warning("%s capture=%s", code, identity["capture_id"])
                return ({"status": "over_size_cap"} if code == OVER_CAP else {"status": "encode_timed_out"}
                        if code == ENCODE_TIMED_OUT else {"status": "failed", "reason": code})
            manifest = {**base, "clip": {**built, "path": CLIP_FILE}}
        manifest["permission"], manifest["upload_sha256"] = basis, upload
        record = clip_record(identity=identity, manifest=manifest)
        manifest["clip_digest"] = record["clip_digest"]
        write_json(manifest_path, manifest)
        if capture_withdrawn(root):
            return {"status": "withdrawn"}
        stored = manifest.get("archive") or {}
        if stored.get("digest") != built["digest"] or _hidden(stored):
            manifest["archive"] = archive(clip_path, identity=identity)
            if manifest["archive"]["digest"] != built["digest"]:
                raise ValueError("website_company_clip_archive_mismatch")
            write_json(manifest_path, manifest)
        # A withdrawal that landed during the upload takes the copy back at once.
        if capture_withdrawn(root):
            _take_back(manifest, manifest_path, retract)
            return {"status": "withdrawn"}
        try:
            deliver(record, identity=identity)
        except Exception as error:  # noqa: BLE001 - classified below; the archive stands only with rights
            reason = _reason(error)
            if _RIGHTS_REFUSALS.search(str(error)):
                _take_back(manifest, manifest_path, retract)
                manifest["refused_by_webapp"] = {"clip_revision": revision, "reason": reason, "at_epoch": int(now()),
                                                 "rights_fingerprint": check.get("rights_fingerprint")}
                write_json(manifest_path, manifest)
                return {"status": "refused", "reason": reason}
            manifest["last_delivery_error"] = reason
            write_json(manifest_path, manifest)
            return {"status": "failed", "reason": reason}
        manifest["delivered"] = {"clip_digest": record["clip_digest"], "at_epoch": int(now())}
        manifest.pop("last_delivery_error", None)
        manifest.pop("refused_by_webapp", None)
        write_json(manifest_path, manifest)
        return {"status": "published", "clip_revision": revision, "clip_digest": record["clip_digest"]}


def capture_roots(storage_root: Path) -> list[Path]:
    """Every website capture whose video normalization has recorded its outcome, or that has a raw manifest
    (a capture handed off before normalization existed)."""
    found = set()
    for pattern, depth in (("*/scenes/site-*/captures/*/pipeline/source_video/normalization.json", 2),
                           ("*/scenes/site-*/captures/*/raw/manifest.json", 1)):
        for path in Path(storage_root).glob(pattern):
            capture = path.parents[depth]
            if capture_name_matches_request(capture.parent.parent.name.removeprefix("site-"), capture.name):
                found.add(capture)
    return sorted(found)


def tick(storage_root: Path, *, stop_reason: Callable[[], str | None] | None = None,
         budget_seconds: float = TICK_BUDGET_SECONDS, clock: Callable[[], float] = time.monotonic,
         **options: Any) -> dict[str, int]:
    """One pass over every capture; counts of each outcome.

    Between captures it stops for a hold, a quiesce or a new release (``stop_reason``, the Atlas preview
    tick's own) and starts no capture after ``budget_seconds``; the captures left are counted ``deferred``.
    The next pass starts from the first capture again; those already current cost one cheap check each.
    """
    counts: dict[str, int] = {}
    started = clock()
    captures = capture_roots(storage_root)
    for index, capture in enumerate(captures):
        reason = stop_reason() if stop_reason is not None else None
        if reason or clock() - started >= budget_seconds:
            counts["deferred"] = len(captures) - index
            counts["stopped_" + (reason or "time_budget")] = 1
            break
        status = publish_company_clip(capture, **options)["status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """One ``bytes=`` range as inclusive (first, last), None for the whole file.

    Several ranges, another unit or a malformed value are ignored (the whole file, as RFC 9110
    allows); a range starting at or past the end is ``RangeNotSatisfiable``.
    """
    # At most 19 digits a side (any real offset); a longer number is not a range this route answers.
    match = re.fullmatch(r"\s*bytes\s*=\s*(\d{0,19})\s*-\s*(\d{0,19})\s*", header or "")
    if match is None or size <= 0 or not (match.group(1) or match.group(2)):
        return None
    first, last = match.groups()
    if not first:
        length = int(last)
        if length == 0:
            raise RangeNotSatisfiable(size)
        return max(0, size - length), size - 1
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if last and int(last) < start:
        return None
    if start >= size:
        raise RangeNotSatisfiable(size)
    return start, end


#: The route's clock (the pass takes ``now`` from its caller); a test can set it.
def _wall_clock() -> float:
    return time.time()


def _basis_still_holds(root: Path, capture_id: str, manifest: Mapping[str, Any]) -> bool:
    basis = (manifest.get("permission") or {}).get("permission_basis")
    if basis == "blueprint_owned_capture":
        return owned_basis_holds(capture_id, manifest.get("upload_sha256"))
    if basis == "site_sharing_statement":
        return live_grant_holds(_read(root.joinpath(*DIRECTORY) / STATEMENT_CHECK_FILE), _wall_clock())
    return False


def open_company_clip(*, request_id: str, capture_id: str, digest: str, storage_root: Path | None,
                      range_header: str | None = None, client: Any = None,
                      bucket: str | None = None) -> tuple[Any, int, tuple[int, int] | None]:
    """The archived clip body, its full size and the range served, for a read the WebApp authorized.

    While the workspace exists the capture must be unwithdrawn, the takedown gate must allow it,
    its manifest must name this digest as archived and not taken back, and its basis must still hold:
    Blueprint's own list names it, or the last live read granted sharing and nothing has said no since.
    After retirement only the object named for this capture and digest exists to read (withdrawal
    deletion deletes it). With no storage root configured nothing can be checked, so nothing is served.
    """
    if (storage_root is None or not capture_name_matches_request(request_id, capture_id)
            or not _DIGEST.fullmatch(str(digest))):
        raise CompanyClipUnavailable("website_company_clip_unavailable")
    roots = [root for root in Path(storage_root).glob(f"*/scenes/site-{request_id}/captures/{capture_id}")
             if root.is_dir()]
    if len(roots) > 1:
        raise CompanyClipUnavailable("website_company_clip_unavailable")
    if roots:
        manifest = _read(roots[0].joinpath(*DIRECTORY) / MANIFEST_FILE)
        archive = manifest.get("archive") or {}
        if (not delivery_gate_allows(roots[0]) or manifest.get("schema_version") != MANIFEST_SCHEMA
                or archive.get("digest") != digest or _hidden(archive)
                or not _basis_still_holds(roots[0], capture_id, manifest)):
            raise CompanyClipUnavailable("website_company_clip_unavailable")
    from .task_evaluation_configured_scene_object_keys import LARGE_ARTIFACT_KEY_PREFIX
    from .task_evaluation_configured_scene_object_store import _artifact_object_store_client, _object_missing
    if client is None or bucket is None:
        client, bucket = _artifact_object_store_client()
    key = f"{LARGE_ARTIFACT_KEY_PREFIX}/{ARTIFACT_KIND}/sha256/{digest[7:]}/{clip_filename(capture_id, request_id)}"
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except Exception as error:  # noqa: BLE001 - provider exception shapes vary
        if _object_missing(error):
            raise CompanyClipUnavailable("website_company_clip_unavailable") from None
        raise
    size = int(head["ContentLength"])
    if (head.get("Metadata") or {}).get("sha256") != digest[7:] or not 0 < size <= MAX_CLIP_BYTES:
        raise CompanyClipUnavailable("website_company_clip_unavailable")
    served = parse_range(range_header, size)
    extra = {"Range": f"bytes={served[0]}-{served[1]}"} if served else {}
    return client.get_object(Bucket=bucket, Key=key, **extra)["Body"], size, served


RECHECK_SCHEMA = "website_company_recheck.v1"


def request_company_clip_recheck(*, request_id: str, capture_id: str, storage_root: Path | None,
                                 now: float | None = None) -> str:
    """The website says a job's permission changed: mark this capture's stored permission check and any
    remembered refusal stale (``permission_recheck``) so the next pass reads the live rights again.

    ``"accepted"``, ``"too_soon"`` (asked within a minute for this capture) or ``"unknown"`` (no such
    capture for this request, or nothing can be checked). Nothing else happens here: no encode, upload or
    deletion. It never raises and says nothing more about the capture.
    """
    from .permission_recheck import request_recheck
    try:
        if (storage_root is None or not isinstance(request_id, str) or not isinstance(capture_id, str)
                or not capture_name_matches_request(request_id, capture_id)):
            return "unknown"
        roots = [root for root in Path(storage_root).glob(f"*/scenes/site-{request_id}/captures/{capture_id}")
                 if root.is_dir() and not root.is_symlink()]
        if len(roots) != 1:
            return "unknown"
        outcome = request_recheck(roots[0], now=now)
    except Exception:  # noqa: BLE001 - a recheck request never fails loudly; the pass's own window is the fallback
        return "unknown"
    return "unknown" if outcome == "unavailable" else outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("publish", help="make and publish one capture's company clip if it is not current")
    one.add_argument("--capture-root", type=Path, required=True)
    every = commands.add_parser("tick", help="make and publish every capture's company clip that is not current")
    every.add_argument("--storage-root", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    result = publish_company_clip(args.capture_root) if args.command == "publish" else tick(args.storage_root)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
