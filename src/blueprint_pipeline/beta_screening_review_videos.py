"""Keep a beta screening run's episode review videos so the job's owner can watch them.

Beta journey: L5 candidate screening, once a run's provider closeout is proven and
its result joined (``beta_policy_screening_launch.reconcile_candidate``), before
the coordinator records the summary and settles the attempt. Each episode renders
review videos of its cameras (``episode_visual_evidence``:
``media/<episode_id>/<camera>.mp4``, ``failed-<camera>.mp4`` for an episode that
stopped after its first observation, ``episode.mp4`` for one camera), and
``adp009d_policy_episode`` renders the pre-start readiness check beside it as
``<episode_id>--prestart-readiness``. The worker's ``media_root`` is
``cell_runs/NN/episodes`` and its episode id ``<run_id>--<cell_id>--<candidate_id>``.

A streamed run's archive never lands on the host. After teardown it is promoted
to the private artifact store with a full readback and indexed
(``provider_output_promotion``), only its JSON needed set is ingested
(``policy_canary_output_members``), and the staged copy is deleted. A download
run keeps its ZIP in the provider run directory.

``retain`` copies exactly these members, nothing else (no frame, NumPy array,
splat or mesh), for a planned cell of this candidate under this attempt's run id:

    cell_runs/<NN>/episodes/media/<run_id>--<cell_id>--<candidate_id>[--prestart-readiness]/[failed-]<camera>.mp4

``<camera>`` is a camera id (``[a-z][a-z0-9_]{0,63}``: ``external``, ``wrist``,
``overview``, ``exact_policy_input``, ``episode``). Bounds, per attempt:
``MAX_VIDEOS`` (ten planned cells, ``DEFAULT_EPISODES``, times two media
directories of at most four videos) and ``MAX_TOTAL_BYTES`` (1 GiB, both stored
and transferred), each video at most ``MAX_VIDEO_BYTES``. The 2026-11-05 run of
capture-447ae4f4 had 60 such videos, 77,308,353 bytes in all and 4,729,053 at
most: the bounds leave about 13 times that. Run videos are kept before set-up
checks when a bound is reached, and what was left out is counted. A promoted
member is read by one range request from the durable archive and must match its
sealed index row (size, CRC-32 and SHA-256); a local member must match its ZIP
entry (size and CRC-32). Each video's SHA-256 is recorded, with its duration
when the MP4 header states one within a day (``mvhd``). Every write first checks
that ``FLOOR_BYTES`` stays free after it. A published entry is checked against
the website's own rules (``publication_entry_valid``) and left out, never the
result refused, when it would break them.

Withdrawal and retirement race. A pass holds ``LOCK_NAME`` beside the attempt
directories, which withdrawal deletion also takes, for its whole copy. Before
each video and before the manifest it re-checks that the capture root is the
same directory and not withdrawn; a capture withdrawn mid-copy loses what this
pass copied. Directories are created one level at a time under an existing
``pipeline/``, so a retired or deleted capture is never re-created.

Where: ``<capture_root>/pipeline/beta_screening_review_videos/<attempt_id>/``,
beside the capture's Atlas preview job, with ``manifest.json``. The videos are
renders of the customer's site, so they live and die with the capture's own
workspace: ``retire-scene-workspace`` removes them from the host with the scene
directory (they remain only inside its private retirement archive), and withdrawal
deletion (``website_capture_withdrawal_deletion``) deletes the scene directory
and that archive. The screening work root is not one of withdrawal deletion's
run roots, which is why the copies are not kept beside the attempt. A withdrawn
capture is never copied (``resolve_capture_root``) and never served (``open_video``).

Money: this runs after provider teardown, so no paid compute is alive. It reads
only our own object store, within ``MAX_TOTAL_BYTES``. It never raises: a
failure is returned as ``not_retained`` or ``partial`` with typed blockers for
the ledger summary, and never changes a run's verdict, settlement or cost.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import re
import shutil
import stat
import struct
import time
import uuid
import zipfile
from collections.abc import Iterable, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

SCHEMA = "beta_screening_review_videos.v1"
MANIFEST_SCHEMA = "beta_screening_review_videos_manifest.v1"
MANIFEST_NAME = "manifest.json"
DIRNAME = "beta_screening_review_videos"
MAX_VIDEOS = 80
MAX_VIDEO_BYTES = 64 * 1024**2
MAX_TOTAL_BYTES = 1024**3
MAX_MANIFEST_BYTES = 1024**2
#: The website's bound on a published duration (a day); a header stating more is not trusted.
MAX_DURATION_SECONDS = 86_400
#: Taken by every retention pass and by withdrawal deletion (``website_capture_withdrawal_deletion``).
LOCK_NAME = "retention.lock"
LOCK_WAIT_SECONDS = 120
BLOCK_BYTES = 8 * 1024**2
DEADLINE_SECONDS = 900
PRESIGN_SECONDS = 3600
LOCAL_ARCHIVE = Path("vast_provider_run") / "vast_provider_runtime_output.zip"
PHASES = ("run", "setup_check")
OUTCOMES = ("success", "failure", "void")
CHANGED = "beta_screening_review_videos_capture_withdrawn_or_changed"
NO_SPACE = "beta_screening_review_videos_disk_space_insufficient"
BUDGET = "beta_screening_review_videos_transfer_budget_reached"
UNSAFE = "beta_screening_review_videos_destination_unsafe"
_SETUP_SUFFIX = "--prestart-readiness"
_NAME = re.compile(r"(?:failed-)?([a-z][a-z0-9_]{0,63})\.mp4")
_CAMERA = re.compile(r"[a-z][a-z0-9_]{0,63}")
_CELL_RUN = re.compile(r"[0-9]{2}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_CODE = re.compile(r"[a-z][a-z0-9_:.=-]{0,199}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")
_VIDEO_FIELDS = ("cell_id", "phase", "camera", "sha256", "size_bytes", "duration_seconds")
_PUBLISHED_VIDEO_FIELDS = ("sha256", "camera", "phase", "size_bytes", "duration_seconds")
#: Review videos of this run left out by a bound; other MP4 names are only counted.
_BOUNDS = frozenset({"over_video_count_cap", "over_total_byte_cap", "over_video_byte_cap"})


def _blocker(exc: BaseException, fallback: str) -> str:
    code = str(exc)
    return code if _CODE.fullmatch(code) else f"{fallback}:{type(exc).__name__}"[:200]


def not_retained(*codes: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA, "status": "not_retained", "videos": [],
            "blockers": sorted({code for code in codes if code})}


def select_review_videos(members: Iterable[Mapping[str, Any]], *, run_id: str, candidate_id: str,
                         cell_ids: list[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The review MP4 rows to copy, in keep order, and the counts of those left out.

    ``members`` are index rows (``path``, ``kind``, ``size``, ``compressed_size``).
    """
    expected: dict[str, tuple[int, str, str]] = {}
    for position, cell_id in enumerate(cell_ids):
        episode_id = f"{run_id}--{cell_id}--{candidate_id}"
        expected[episode_id] = (position, cell_id, "run")
        expected[episode_id + _SETUP_SUFFIX] = (position, cell_id, "setup_check")
    rows, omitted, seen = [], {}, set()

    def omit(reason: str) -> None:
        omitted[reason] = omitted.get(reason, 0) + 1

    for member in members:
        path = str(member.get("path") or "")
        if member.get("kind") != "file" or not path.lower().endswith(".mp4"):
            continue
        parts = path.split("/")
        name = _NAME.fullmatch(parts[-1]) if len(parts) == 6 else None
        if (name is None or parts[0] != "cell_runs" or not _CELL_RUN.fullmatch(parts[1])
                or parts[2:4] != ["episodes", "media"] or parts[4] not in expected):
            omit("unexpected_member_name")
            continue
        position, cell_id, phase = expected[parts[4]]
        relative = f"videos/{parts[1]}-{phase}-{parts[5]}"
        if relative in seen:
            omit("duplicate_member")
            continue
        size, compressed = member.get("size"), member.get("compressed_size")
        if type(size) is not int or type(compressed) is not int or not 0 < size <= MAX_VIDEO_BYTES \
                or not 0 < compressed <= MAX_VIDEO_BYTES:
            omit("over_video_byte_cap")
            continue
        seen.add(relative)
        rows.append({"member": member, "cell_id": cell_id, "episode_id": parts[4][:-len(_SETUP_SUFFIX)]
                     if phase == "setup_check" else parts[4], "phase": phase, "camera": name.group(1),
                     "relative_path": relative, "order": (PHASES.index(phase), position, parts[5])})
    rows.sort(key=lambda row: row["order"])
    kept, stored, read = [], 0, 0
    for row in rows:
        if len(kept) >= MAX_VIDEOS:
            omit("over_video_count_cap")
        elif (stored + row["member"]["size"] > MAX_TOTAL_BYTES
              or read + row["member"]["compressed_size"] > MAX_TOTAL_BYTES):
            omit("over_total_byte_cap")
        else:
            kept.append(row)
            stored += row["member"]["size"]
            read += row["member"]["compressed_size"]
    return kept, omitted


def _box(stream, start: int, end: int, kind: bytes) -> tuple[int, int] | None:
    position = start
    for _ in range(256):
        if position + 8 > end:
            return None
        stream.seek(position)
        header = stream.read(16)
        if len(header) < 8:
            return None
        size, name = struct.unpack(">I4s", header[:8])
        offset = 8
        if size == 1:
            if len(header) < 16:
                return None
            size, offset = struct.unpack(">Q", header[8:16])[0], 16
        elif size == 0:
            size = end - position
        if size < offset or position + size > end:
            return None
        if name == kind:
            return position + offset, position + size
        position += size
    return None


def mp4_duration_seconds(path: Path) -> float | None:
    """The movie header's duration, or None when the file does not state one plainly."""
    try:
        with path.open("rb") as stream:
            moov = _box(stream, 0, os.fstat(stream.fileno()).st_size, b"moov")
            mvhd = _box(stream, *moov, b"mvhd") if moov else None
            if mvhd is None:
                return None
            stream.seek(mvhd[0])
            head = stream.read(32)
            if head[:1] == b"\x00" and len(head) >= 20:
                timescale, duration = struct.unpack(">II", head[12:20])
                unknown = duration == 0xFFFFFFFF
            elif head[:1] == b"\x01" and len(head) >= 32:
                timescale, duration = struct.unpack(">IQ", head[20:32])
                unknown = duration == 0xFFFFFFFFFFFFFFFF
            else:
                return None
            if not timescale or unknown:
                return None
            seconds = duration / timescale
            return round(seconds, 3) if math.isfinite(seconds) and 0 <= seconds <= MAX_DURATION_SECONDS else None
    except (OSError, struct.error, TypeError, ValueError):
        return None


def _write_manifest(directory: Path, manifest: dict[str, Any]) -> Path:
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    path = directory / MANIFEST_NAME
    temporary = directory / f".{MANIFEST_NAME}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o440)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_manifest(path: Path) -> dict[str, Any] | None:
    """A sealed manifest, or None when it is missing, a link, oversized or altered."""
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    if (not isinstance(value, dict) or value.get("schema_version") != MANIFEST_SCHEMA
            or not isinstance(value.get("videos"), list)
            or value.get("manifest_digest") != canonical_digest(value, digest_field="manifest_digest")):
        return None
    return value


def summary_reference(manifest: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """What the ledger summary keeps: the manifest's path and digest and one row per video."""
    videos = [{key: video[key] for key in _VIDEO_FIELDS} for video in manifest["videos"]]
    return {"schema_version": SCHEMA, "status": manifest["status"], "manifest_path": str(path),
            "manifest_digest": manifest["manifest_digest"], "video_count": len(videos),
            "total_bytes": sum(video["size_bytes"] for video in videos), "videos": videos,
            "blockers": list(manifest["blockers"])}


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _identity(path: Path) -> tuple[int, int] | None:
    try:
        status = os.stat(path, follow_symlinks=False)
    except OSError:
        return None
    return (status.st_dev, status.st_ino) if stat.S_ISDIR(status.st_mode) else None


class _Capture:
    """The capture root as first seen; each write re-checks it is still that directory and not withdrawn."""

    def __init__(self, root: Path, directory: Path):
        self.root, self.directory, self.identity = root, directory, _identity(root)

    def same(self) -> bool:
        return self.identity is not None and _identity(self.root) == self.identity

    def current(self) -> bool:
        from .website_atlas_preview_store import capture_withdrawn

        return self.same() and not capture_withdrawn(self.root)


def _make_directory(path: Path) -> None:
    """One level only: a parent that is gone (a retired or deleted capture) is never re-created."""
    try:
        path.mkdir(mode=0o750)
    except FileExistsError:
        pass
    if path.is_symlink() or not path.is_dir():
        raise ValueError(UNSAFE)


@contextmanager
def _retention_lock(base: Path):
    """Exclusive ``LOCK_NAME``, which withdrawal deletion also takes, so neither runs inside the other."""
    descriptor = os.open(base / LOCK_NAME, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o640)
    with os.fdopen(descriptor, "rb+") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError(UNSAFE)
        deadline = time.monotonic() + LOCK_WAIT_SECONDS
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ValueError("beta_screening_review_videos_capture_busy") from None
                time.sleep(1)
        yield


def _room(path: Path, disk_usage):
    """Before every write: ``FLOOR_BYTES`` must stay free after it."""
    from .provider_output_range_ingestion import FLOOR_BYTES

    def check(needed: int) -> None:
        if (disk_usage or shutil.disk_usage)(path).free < FLOOR_BYTES + needed:
            raise ValueError(NO_SPACE)
    return check


def _copy(feed, target: Path, *, size: int, expected: Mapping[str, Any], room) -> dict[str, Any]:
    """Stream one member into a fresh partial beside its target, then publish it only if it matches."""
    from .provider_output_range_ingestion import _PartialWriter

    partial = target.with_name("." + target.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        with partial.open("xb") as sink:  # its directory must exist already; nothing is re-created
            writer = _PartialWriter(sink, 0, size, room)
            feed(writer.write)
            record = writer.record()
        if record["size_bytes"] != size or any(record[key] != value for key, value in expected.items()):
            raise ValueError("provider_output_member_digest_mismatch")
        os.chmod(partial, 0o440)
        os.link(partial, target)  # never replaces
    finally:
        partial.unlink(missing_ok=True)
    return record


def _promoted(index: Mapping[str, Any], opener):
    """The durable archive's pinned reader, and how to stream one indexed member from it."""
    from .provider_output_member_index import MemberInflater, inflate_step_bytes
    from .provider_output_range_ingestion import CasArchiveSource
    from .task_evaluation_configured_scene_object_store import presign_configured_scene_artifact

    reference = index["archive"]["durable_reference"]
    source = CasArchiveSource(reference, opener=opener, block_bytes=BLOCK_BYTES, deadline_seconds=DEADLINE_SECONDS,
                              presign=lambda: presign_configured_scene_artifact(
                                  reference=reference, expiration_seconds=PRESIGN_SECONDS))
    source.durable_reference()
    reader = source.open(maximum_archive_bytes=index["archive"]["size"])
    if reader.identity["size_bytes"] != index["archive"]["size"]:
        raise ValueError("beta_screening_review_videos_archive_size_mismatch")

    def feed_for(row):
        member = row["member"]

        def feed(write):
            if reader.transferred_bytes + member["compressed_size"] > MAX_TOTAL_BYTES:
                raise ValueError(BUDGET)
            inflater = MemberInflater(member["method"], member["size"], write,
                                      step_bytes=inflate_step_bytes(BLOCK_BYTES))
            if member["compressed_size"]:
                reader.stream_to(inflater.feed, start=member["data_offset"],
                                 end=member["data_offset"] + member["compressed_size"])
            inflater.finish()
        return feed, {"sha256": member["sha256"], "crc32": member["crc32"]}
    return reader, feed_for


def _local(archive: zipfile.ZipFile):
    def feed_for(row):
        info = row["member"]["info"]

        def feed(write):
            with archive.open(info) as stream:  # zipfile also checks the CRC-32 at the end
                for chunk in iter(lambda: stream.read(1024**2), b""):
                    write(chunk)
        return feed, {"crc32": info.CRC}
    return feed_for


def _copy_all(rows, capture: _Capture, feed_for, room) -> tuple[list, list]:
    from .provider_output_range_transport import ProviderOutputTransportError

    copied, blockers = [], []
    for row in rows:
        if not capture.current():
            blockers.append(CHANGED)
            break
        target = capture.directory / row["relative_path"]
        size, sha256 = row["member"]["size"], row["member"].get("sha256")
        if _kept(target, size, sha256):
            copied.append({**row, "sha256": sha256 or _sha256_file(target)})
            continue
        target.unlink(missing_ok=True)
        feed, expected = feed_for(row)
        try:
            record = _copy(feed, target, size=size, expected=expected, room=room)
        except ProviderOutputTransportError as exc:
            blockers.append(_blocker(exc, "beta_screening_review_videos_transport_failed"))
            break  # A pinned reader that failed once fails every later read.
        except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:  # RuntimeError: an encrypted entry
            blockers.append(_blocker(exc, "beta_screening_review_videos_member_invalid"))
            if blockers[-1] in (NO_SPACE, BUDGET):
                break
            continue
        copied.append({**row, "sha256": record["sha256"]})
    return copied, blockers


def _tidy(directory: Path) -> None:
    """Drop partial copies, then the directories a pass left empty."""
    videos = directory / "videos"
    for partial in videos.glob(".*.partial") if videos.is_dir() and not videos.is_symlink() else ():
        partial.unlink(missing_ok=True)
    for path in (videos, directory):
        try:
            path.rmdir()
        except OSError:
            break  # not empty, or not there


def _kept(target: Path, size: int, sha256: str | None) -> bool:
    """A video an earlier, interrupted pass already copied and verified."""
    if target.is_symlink() or not target.is_file() or target.stat().st_size != size:
        return False
    return sha256 is None or _sha256_file(target) == sha256


def _source(lane_root: Path) -> tuple[str, Any]:
    """The promoted archive's sealed index, else the local download ZIP, else nothing."""
    from .arena_provider_output_streaming import STAGING_DIRNAME, _sealed_index
    from .provider_output_promotion_records import load_promotion_receipt, staging_manifest_sha256

    staging = lane_root / STAGING_DIRNAME
    if staging.is_dir() and not staging.is_symlink():
        receipt = load_promotion_receipt(staging, staging_manifest_sha256=staging_manifest_sha256(staging))
        index = _sealed_index(lane_root, receipt) if receipt and receipt.get("status") == "promoted" else None
        if index is not None:
            return "promoted_archive", index
    archive = lane_root / LOCAL_ARCHIVE
    if archive.is_file() and not archive.is_symlink():
        return "local_archive", archive
    return "none", None


def retain(*, plan: Mapping[str, Any], candidate: Mapping[str, Any], prepared: Mapping[str, Any],
           adapter: Mapping[str, Any], attempt_dir: Path, capture_root: Path | None = None,
           disk_usage=None, opener=None) -> dict[str, Any]:
    """Copy this attempt's review videos beside its capture; the summary reference. Never raises."""
    try:
        return _retain(plan, candidate, prepared, adapter, Path(attempt_dir).resolve(), capture_root,
                       disk_usage, opener)
    except Exception as exc:  # noqa: BLE001 - keeping videos never decides a run
        return not_retained(_blocker(exc, "beta_screening_review_videos_failed"))


def _retain(plan, candidate, prepared, adapter, attempt_dir, capture_root, disk_usage, opener):
    from .provider_output_range_ingestion import FLOOR_BYTES

    if capture_root is None:
        from .team_scene_capture_root import resolve_capture_root
        capture_root = resolve_capture_root(plan["source"])  # refuses a withdrawn capture
    capture_root = Path(capture_root)
    lane_root = Path(str(adapter.get("attempt_root") or ""))
    if (not lane_root.is_absolute() or lane_root.is_symlink() or not lane_root.is_dir()
            or not lane_root.resolve().is_relative_to(attempt_dir)):
        return not_retained("beta_screening_review_videos_lane_root_unbound")
    pipeline = capture_root / "pipeline"
    base = pipeline / DIRNAME
    capture = _Capture(capture_root, base / attempt_dir.name)
    existing = read_manifest(capture.directory / MANIFEST_NAME)
    if existing is not None and existing.get("attempt_id") == attempt_dir.name:
        return summary_reference(existing, capture.directory / MANIFEST_NAME)
    if not capture.current() or pipeline.is_symlink() or not pipeline.is_dir():
        return not_retained(CHANGED)
    kind, value = _source(lane_root)
    if kind == "none":
        return not_retained("beta_screening_review_videos_source_unavailable")
    binding = {"run_id": str(prepared["run_id"]), "candidate_id": str(candidate["candidate_id"]),
               "cell_ids": [str(cell["cell_id"]) for cell in candidate["scenario_cells"]]}
    with ExitStack() as stack:
        archive = stack.enter_context(zipfile.ZipFile(value)) if kind == "local_archive" else None
        members = (value["members"] if archive is None else
                   [{"path": info.filename, "kind": "directory" if info.is_dir() else "file", "info": info,
                     "size": info.file_size, "compressed_size": info.compress_size} for info in archive.infolist()])
        rows, omitted = select_review_videos(members, **binding)
        if not rows:
            return not_retained("beta_screening_review_videos_none_found")
        if (disk_usage or shutil.disk_usage)(capture_root).free < sum(row["member"]["size"] for row in rows) + FLOOR_BYTES:
            return not_retained(NO_SPACE)
        _make_directory(base)
        stack.enter_context(_retention_lock(base))
        if not capture.current():
            return not_retained(CHANGED)
        _make_directory(capture.directory)
        _make_directory(capture.directory / "videos")
        stack.callback(_tidy, capture.directory)
        reader, feed_for = _promoted(value, opener) if archive is None else (None, _local(archive))
        copied, blockers = _copy_all(rows, capture, feed_for, _room(capture_root, disk_usage))
        transfer = ({"transferred_bytes": reader.transferred_bytes, "http_request_count": reader.request_count}
                    if reader is not None else {"transferred_bytes": 0, "http_request_count": 0})
        if CHANGED in blockers or not capture.current():
            # Withdrawn (or replaced) while copying: what this pass copied of a withdrawn capture goes again.
            if capture.same():
                shutil.rmtree(capture.directory, ignore_errors=True)
            return not_retained(CHANGED)
        if not copied:
            return not_retained(*blockers, "beta_screening_review_videos_none_copied")
        if set(omitted) & _BOUNDS:
            blockers.append("beta_screening_review_videos_bound_reached")
        videos = [{"cell_id": row["cell_id"], "episode_id": row["episode_id"], "phase": row["phase"],
                   "camera": row["camera"], "size_bytes": row["member"]["size"], "sha256": row["sha256"],
                   "duration_seconds": mp4_duration_seconds(capture.directory / row["relative_path"]),
                   "relative_path": row["relative_path"], "source_member": row["member"]["path"]} for row in copied]
        from .common import utc_now_iso

        manifest = {
            "schema_version": MANIFEST_SCHEMA, "status": "partial" if blockers else "retained",
            "site_id": plan["site_id"], "capture_id": plan["source"]["capture_id"], "campaign_id": plan["campaign_id"],
            "plan_digest": plan["plan_digest"], "candidate_id": binding["candidate_id"], "attempt_id": attempt_dir.name,
            "run_id": binding["run_id"], "source": {"kind": kind, **({"archive_sha256": value["archive"]["sha256"],
                                                                     "member_index_digest": value["index_digest"]}
                                                                    if archive is None else {})},
            "limits": {"max_videos": MAX_VIDEOS, "max_video_bytes": MAX_VIDEO_BYTES, "max_total_bytes": MAX_TOTAL_BYTES},
            "videos": videos, "omitted": dict(sorted(omitted.items())), **transfer,
            "blockers": sorted(set(blockers)), "retained_at": utc_now_iso(), "claims_success": False,
        }
        if not capture.current():
            if capture.same():
                shutil.rmtree(capture.directory, ignore_errors=True)
            return not_retained(CHANGED)
        path = _write_manifest(capture.directory, manifest)
        return summary_reference(manifest, path)


def _duration_ok(value: Any) -> bool:
    return value is None or (type(value) in (int, float) and math.isfinite(value)
                             and 0 <= value <= MAX_DURATION_SECONDS)


def valid_reference(value: Any) -> dict[str, Any] | None:
    """A summary reference fit for the ledger, else None."""
    if not isinstance(value, Mapping) or value.get("schema_version") != SCHEMA:
        return None
    if value.get("status") == "not_retained":
        blockers = [code for code in value.get("blockers") or [] if isinstance(code, str) and _CODE.fullmatch(code)]
        return not_retained(*blockers[:16]) if blockers else None
    videos = value.get("videos")
    if (value.get("status") not in {"retained", "partial"} or not _DIGEST.fullmatch(str(value.get("manifest_digest")))
            or not isinstance(videos, list) or not 0 < len(videos) <= MAX_VIDEOS
            or any(not isinstance(row, Mapping) or set(row) != set(_VIDEO_FIELDS)
                   or not _DIGEST.fullmatch(str(row["sha256"])) or row["phase"] not in PHASES
                   or not isinstance(row["cell_id"], str) or not _CAMERA.fullmatch(str(row["camera"]))
                   or type(row["size_bytes"]) is not int or not 0 < row["size_bytes"] <= MAX_VIDEO_BYTES
                   or not _duration_ok(row["duration_seconds"])
                   for row in videos)):
        return None
    return json.loads(json.dumps(dict(value)))


def _published_video_ok(video: Any) -> bool:
    return (isinstance(video, Mapping) and set(video) == set(_PUBLISHED_VIDEO_FIELDS)
            and isinstance(video["sha256"], str) and _DIGEST.fullmatch(video["sha256"]) is not None
            and isinstance(video["camera"], str) and _CAMERA.fullmatch(video["camera"]) is not None
            and video["phase"] in PHASES and type(video["size_bytes"]) is int
            and 0 < video["size_bytes"] <= MAX_VIDEO_BYTES and _duration_ok(video["duration_seconds"]))


def publication_entry_valid(entry: Any, row: Mapping[str, Any]) -> bool:
    """The website's rules for one published ``review_videos`` entry (WebApp ``betaPolicyScreeningContract``).

    One practice run per planned episode, restating the candidate's own counts, for a
    completed candidate, with 1 to ``MAX_VIDEOS`` well-formed videos.
    """
    try:
        runs = entry["practice_runs"]
        if (set(entry) != {"candidate_id", "manifest_digest", "practice_runs"} or row.get("status") != "completed"
                or entry["candidate_id"] != row.get("candidate_id") or not isinstance(entry["candidate_id"], str)
                or not _IDENTIFIER.fullmatch(entry["candidate_id"]) or not isinstance(entry["manifest_digest"], str)
                or not _DIGEST.fullmatch(entry["manifest_digest"]) or not isinstance(runs, list)
                or not 1 <= len(runs) <= 100 or len(runs) != row.get("episodes_planned")
                or any(not isinstance(run, Mapping) or set(run) != {"cell_id", "outcome", "videos"}
                       or not isinstance(run["cell_id"], str) or not _IDENTIFIER.fullmatch(run["cell_id"])
                       or run["outcome"] not in OUTCOMES or not isinstance(run["videos"], list) for run in runs)):
            return False
        videos = [video for run in runs for video in run["videos"]]
        counts = [sum(run["outcome"] == outcome for run in runs) for outcome in OUTCOMES]
        return (len({run["cell_id"] for run in runs}) == len(runs) and 1 <= len(videos) <= MAX_VIDEOS
                and counts == [row.get("episodes_succeeded"), row.get("episodes_failed"), row.get("episodes_void")]
                and all(_published_video_ok(video) for video in videos))
    except (KeyError, TypeError):
        return False


def publication_review_videos_valid(entries: Any, candidates: Any) -> bool:
    """A whole publication's ``review_videos``: 1 to 256 entries, one per completed candidate at most."""
    rows = {row.get("candidate_id"): row for row in candidates if isinstance(row, Mapping)} \
        if isinstance(candidates, list) else {}
    return (isinstance(entries, list) and 1 <= len(entries) <= 256
            and all(isinstance(entry, Mapping) and entry.get("candidate_id") in rows for entry in entries)
            and len({entry["candidate_id"] for entry in entries}) == len(entries)
            and all(publication_entry_valid(entry, rows[entry["candidate_id"]]) for entry in entries))


def publication_entry(row: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any] | None:
    """A published candidate row's practice runs: outcome and videos, no codes or paths.

    None when nothing was kept, or when the entry would break the website's rules:
    the videos are left out rather than ever refusing the result itself.
    """
    reference = summary.get("review_videos")
    if not isinstance(reference, Mapping) or reference.get("status") not in {"retained", "partial"}:
        return None
    try:
        runs = []
        for outcome in summary.get("episode_outcomes") or []:
            videos = [{"sha256": video["sha256"], "camera": video["camera"], "phase": video["phase"],
                       "size_bytes": video["size_bytes"],
                       "duration_seconds": video["duration_seconds"] if _duration_ok(video["duration_seconds"]) else None}
                      for video in reference["videos"] if video["cell_id"] == outcome["cell_id"]]
            runs.append({"cell_id": outcome["cell_id"], "outcome": outcome["outcome"], "videos": videos})
        entry = {"candidate_id": row["candidate_id"], "manifest_digest": reference["manifest_digest"],
                 "practice_runs": runs}
    except (KeyError, TypeError):
        return None
    return entry if publication_entry_valid(entry, row) else None


def open_video(*, request_id: str, capture_id: str, digest: str, storage_root: Path | None,
               manifest_digest: str | None = None) -> tuple[Any, int]:
    """One retained video's open file and size for a WebApp view the WebApp already authorized.

    Callers name a capture and a digest (and the published manifest's digest),
    never a path. A capture that is withdrawn, missing or ambiguous, or a digest
    no sealed manifest of that capture names, is ``PreviewUnavailable``.
    """
    from .website_atlas_preview_store import PreviewUnavailable, capture_name_matches_request, capture_withdrawn

    if (storage_root is None or not capture_name_matches_request(request_id, capture_id)
            or not _DIGEST.fullmatch(str(digest))
            or (manifest_digest is not None and not _DIGEST.fullmatch(str(manifest_digest)))):
        raise PreviewUnavailable("beta_screening_review_video_unavailable")
    roots = [root for root in Path(storage_root).glob(f"*/scenes/site-{request_id}/captures/{capture_id}")
             if root.is_dir() and not root.is_symlink()]
    if len(roots) != 1 or capture_withdrawn(roots[0]):
        raise PreviewUnavailable("beta_screening_review_video_unavailable")
    base = roots[0] / "pipeline" / DIRNAME
    for attempt in sorted(base.iterdir()) if base.is_dir() and not base.is_symlink() else ():
        manifest = read_manifest(attempt / MANIFEST_NAME) if attempt.is_dir() and not attempt.is_symlink() else None
        if (manifest is None or manifest.get("attempt_id") != attempt.name or manifest.get("capture_id") != capture_id
                or manifest.get("site_id") != f"site-{request_id}"
                or manifest_digest is not None and manifest["manifest_digest"] != manifest_digest):
            continue
        for video in manifest["videos"]:
            relative = str(video.get("relative_path") or "")
            if video.get("sha256") != digest or not re.fullmatch(r"videos/[0-9]{2}-[a-z_]+-[a-z0-9_-]+\.mp4", relative):
                continue
            path = attempt / relative
            if path.is_symlink() or not path.is_file() or path.stat().st_size != video.get("size_bytes"):
                continue
            return path.open("rb"), int(video["size_bytes"])
    raise PreviewUnavailable("beta_screening_review_video_unavailable")


__all__ = [
    "DIRNAME", "LOCK_NAME", "MANIFEST_NAME", "MAX_TOTAL_BYTES", "MAX_VIDEOS", "MAX_VIDEO_BYTES", "SCHEMA",
    "mp4_duration_seconds", "not_retained", "open_video", "publication_entry", "publication_entry_valid",
    "publication_review_videos_valid", "read_manifest", "retain", "select_review_videos", "summary_reference",
    "valid_reference",
]
