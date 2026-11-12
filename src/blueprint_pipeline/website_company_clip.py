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
  A capture handed off before normalization existed has no such record: its clip is made, the
  same way, from the raw upload its ``raw/manifest.json`` names (``video_uri``). The manifest
  records which (``source``: ``normalized`` or ``raw_upload``). A record that exists but cannot
  be used never falls back to the raw upload;
- video stream only: no audio stream at all (sites are told people heard in a clip must know
  it is shared; a company does not need sound), no subtitle or data stream, and no metadata
  (``-map_metadata -1``: a phone's location tags do not travel);
- H.264 in MP4 with ``+faststart`` so a browser can seek, at most ``MAX_LONG_EDGE`` px on the
  long side and ``MAX_FPS`` frames a second, at a bitrate that keeps a 60 to 120 second clip
  well under ``MAX_CLIP_BYTES``. A clip that cannot be brought under the cap is recorded as
  refused for that source and never retried for it (flagged, never failing anything);
- the bytes go to the private artifact store under a name for the capture (withdrawal deletion
  finds them by that name, also after workspace retirement); the WebApp is told the clip's
  identity through the signed ``company-clip`` control operation and reads the bytes, by range,
  through the signed ``/api/live-pipeline/website-company-clips`` route.

Rights are a hard stop. Nothing is made unless the capture is unwithdrawn, the delivery-time
takedown gate allows it, the WebApp's current task context grants consent, and the recording
statement version covers sharing or the capture is Blueprint's own, as
``sharing_statement.sharing_basis`` decides (``config/blueprint_owned_captures.v1.json``, its row's
video digest prefix matched against the upload's recorded digest). A
capture under an older statement gets nothing, and nothing about it is sent anywhere.

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
resource. It never touches a job: every failure is a returned status, retried on the next pass.
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
from .sharing_statement import sharing_basis
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
#: Beside the manifest: when the live statement was last read for a capture whose snapshot does not cover sharing.
STATEMENT_CHECK_FILE = "statement_check.json"
LIVE_STATEMENT_RECHECK_SECONDS = 6 * 3600
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
BUILDER_REVISION = "website_company_clip_builder.v1"
OVER_CAP = "website_company_clip_over_size_cap"
_ENCODE_TIMEOUT_SECONDS = 3600
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

    Cheap: no read of the video. Its digest and whether it is a readable video are decided under the lock.
    A capture with no ``pipeline/`` directory was never handed off (or was retired): it has no source.
    """
    root = Path(capture_root)
    if (root / "pipeline").is_symlink() or not (root / "pipeline").is_dir():
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


def _raw_upload_digest(source: Mapping[str, Any], manifest: Mapping[str, Any]) -> str | None:
    """The raw upload's digest, or None when it is not a readable video.

    Read once: a pass reuses the digest the manifest kept for the same file, size and modification time,
    so a current clip costs no read of a large upload every pass.
    """
    from .website_source_video import probe_video
    kept = manifest.get("source_video")
    if (manifest.get("source") == "raw_upload" and isinstance(kept, Mapping)
            and all(kept.get(key) == source[key] for key in ("relative", "size_bytes", "mtime_ns"))
            and _DIGEST.fullmatch(str(kept.get("digest") or ""))):
        return kept["digest"]
    try:
        probe_video(Path(source["path"]))
        return _sha256_file(Path(source["path"]))
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def clip_source(capture_root: Path) -> dict[str, Any] | None:
    """The clip website preparation reads, as its normalization record names it; without a record, the raw
    upload the capture's manifest names; or None.

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


def _basis(root: Path, identity: Mapping[str, str], rights: Mapping[str, Any]) -> dict[str, Any] | None:
    found = sharing_basis({"capture_id": identity["capture_id"], "capture_rights": dict(rights)},
                          video_sha256=recorded_video_sha256(root), owned_list=OWNED_CAPTURES_LIST)
    if found is None:
        return None
    version = rights.get("statement_version")
    return {"permission_basis": found["permission_basis"], "statement_version": version if isinstance(version, str)
            else None}


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


def permission_from_rights(root: Path, rights: Mapping[str, Any], identity: Mapping[str, str]) -> dict[str, Any]:
    """Which rule lets this capture's video be shared, from current rights; a typed refusal otherwise."""
    if rights.get("consent_revoked") is not False or rights.get("consent_status") != "granted":
        raise ValueError("website_company_clip_rights_refused")
    basis = _basis(root, identity, rights)
    if basis is None:
        raise ValueError(_NOT_PERMITTED)
    return basis


def _may_be_permitted(root: Path, identity: Mapping[str, str], now: float) -> bool:
    """No network: the snapshot covers sharing, the capture is Blueprint's own, or a live recheck is due."""
    if _basis(root, identity, {"statement_version": recorded_statement_version(root)}) is not None:
        return True
    checked = _read(root.joinpath(*DIRECTORY) / STATEMENT_CHECK_FILE).get("checked_at_epoch")
    return not isinstance(checked, int) or not 0 <= now - checked < LIVE_STATEMENT_RECHECK_SECONDS


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


def _encode(source: Path, target: Path, *, width: int, height: int, fps_cap: bool, kbps: int) -> None:
    filters = [f"scale={width}:{height}:flags={ENCODER['scale_flags']}"] + ([f"fps={MAX_FPS}"] if fps_cap else [])
    # Video stream 0 only: -an/-sn/-dn drop every other stream; no tag or chapter of the source is copied.
    command = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-threads", str(ENCODER["threads"]), "-i", str(source),
               "-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1", "-map_metadata:s:v", "-1",
               "-map_chapters", "-1",
               "-filter_threads", "1", "-vf", ",".join(filters),
               "-c:v", ENCODER["codec"], "-profile:v", ENCODER["profile"], "-preset", ENCODER["preset"],
               "-crf", str(ENCODER["crf"]), "-maxrate", f"{kbps}k", "-bufsize", f"{2 * kbps}k",
               "-pix_fmt", ENCODER["pix_fmt"], "-threads", str(ENCODER["threads"]),
               "-movflags", "+faststart", "-f", "mp4", str(target)]
    subprocess.run(command, check=True, capture_output=True, timeout=_ENCODE_TIMEOUT_SECONDS)


def build_clip(source: Path, output: Path, *, max_bytes: int = MAX_CLIP_BYTES) -> dict[str, Any]:
    """Encode ``source`` (only read) to ``output``, proven video-only, seekable and under the cap.

    One lower-rate retry when the first encode lands over the cap; then ``OVER_CAP``.
    """
    from .website_source_video import probe_video
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
                _encode(source, partial, width=width, height=height, fps_cap=probe["fps"] > MAX_FPS + 0.01, kbps=kbps)
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
        from .website_source_video import _rate
        fps = _rate(video[0].get("avg_frame_rate"))
        os.replace(partial, output)
    finally:
        partial.unlink(missing_ok=True)
    return {"format": FORMAT, "content_type": CONTENT_TYPE, "digest": _sha256_file(output), "size_bytes": size,
            "width": int(video[0]["width"]), "height": int(video[0]["height"]),
            "fps": round(fps, 3) if fps else None, "duration_seconds": streams["duration_seconds"],
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
    """One idempotent pass for one capture: make, archive and deliver what is not yet current.

    Returns ``{"status": ...}``: ``current``, ``published``, ``not_permitted`` (older statement),
    ``no_source`` (no normalization record and no readable raw upload named by the capture's manifest,
    or a record that cannot be used), ``over_size_cap``, ``withdrawn``, ``refused`` (rights or
    the takedown gate, with a reason), ``busy`` or ``failed`` (retried by the next pass). It
    never raises and never touches the capture's job.
    """
    try:
        return _publish(Path(capture_root), now, check_rights, archive, retract, deliver, max_bytes)
    except CompanyClipBusy:
        return {"status": "busy"}
    except Exception as error:  # noqa: BLE001 - one capture never stops a pass; retried next time
        logger.warning("website_company_clip_failed capture=%s type=%s", Path(capture_root).name,
                       type(error).__name__)
        return {"status": "failed", "reason": _reason(error)}


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
    # until its live statement is due a recheck.
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
            source["digest"] = _raw_upload_digest(source, manifest)
            if source["digest"] is None:
                return {"status": "no_source"}
        revision = clip_revision(source, max_bytes=max_bytes)
        if _current(manifest, revision, clip_path):
            return {"status": "current", "clip_revision": revision}
        if manifest.get("clip_revision") == revision and manifest.get("refused") == OVER_CAP:
            return {"status": "over_size_cap"}
        try:
            basis = permission_from_rights(root, check_rights(identity), identity)
        except Exception as error:  # noqa: BLE001 - any doubt about rights is a refusal
            # Recorded either way: a capture its files do not permit is read live once a window, not every pass.
            write_json(directory / STATEMENT_CHECK_FILE, {"checked_at_epoch": int(now()), "outcome": _reason(error)})
            if str(error) != _NOT_PERMITTED:
                return {"status": "refused", "reason": _reason(error)}
            if manifest:
                _take_back(manifest, manifest_path, retract)
            return {"status": "not_permitted"}
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
                    "permission": basis, "built_at_epoch": int(now()), "archive": None, "delivered": None}
            try:
                built = build_clip(Path(source["path"]), clip_path, max_bytes=max_bytes)
            except ValueError as error:
                if str(error) != OVER_CAP:
                    raise
                clip_path.unlink(missing_ok=True)
                write_json(manifest_path, {**base, "clip": None, "refused": OVER_CAP})
                logger.warning("website_company_clip_over_size_cap capture=%s", identity["capture_id"])
                return {"status": "over_size_cap"}
            manifest = {**base, "clip": {**built, "path": CLIP_FILE}}
        manifest["permission"] = basis
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
                return {"status": "refused", "reason": reason}
            manifest["last_delivery_error"] = reason
            write_json(manifest_path, manifest)
            return {"status": "failed", "reason": reason}
        manifest["delivered"] = {"clip_digest": record["clip_digest"], "at_epoch": int(now())}
        manifest.pop("last_delivery_error", None)
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


def tick(storage_root: Path, **options: Any) -> dict[str, int]:
    """One pass over every capture; counts of each outcome."""
    counts: dict[str, int] = {}
    for capture in capture_roots(storage_root):
        status = publish_company_clip(capture, **options)["status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """One ``bytes=`` range as inclusive (first, last), None for the whole file.

    Several ranges, another unit or a malformed value are ignored (the whole file, as RFC 9110
    allows); a range starting at or past the end is ``RangeNotSatisfiable``.
    """
    match = re.fullmatch(r"\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*", header or "")
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


def open_company_clip(*, request_id: str, capture_id: str, digest: str, storage_root: Path | None,
                      range_header: str | None = None, client: Any = None,
                      bucket: str | None = None) -> tuple[Any, int, tuple[int, int] | None]:
    """The archived clip body, its full size and the range served, for a read the WebApp authorized.

    While the workspace exists the capture must be unwithdrawn, the takedown gate must allow it
    and its manifest must name this digest as archived; after retirement only the object named
    for this capture and digest exists to read (withdrawal deletion deletes it).
    """
    if not capture_name_matches_request(request_id, capture_id) or not _DIGEST.fullmatch(str(digest)):
        raise CompanyClipUnavailable("website_company_clip_unavailable")
    if storage_root is not None:
        roots = [root for root in Path(storage_root).glob(f"*/scenes/site-{request_id}/captures/{capture_id}")
                 if root.is_dir()]
        if len(roots) > 1:
            raise CompanyClipUnavailable("website_company_clip_unavailable")
        if roots:
            manifest = _read(roots[0].joinpath(*DIRECTORY) / MANIFEST_FILE)
            archive = manifest.get("archive") or {}
            if (not delivery_gate_allows(roots[0]) or manifest.get("schema_version") != MANIFEST_SCHEMA
                    or archive.get("digest") != digest or _hidden(archive)):
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
