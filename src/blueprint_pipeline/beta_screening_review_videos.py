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
when the MP4 header states it (``mvhd``). Writing needs ``FLOOR_BYTES`` of free
space left after the copy.

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

import json
import os
import re
import struct
import uuid
import zipfile
from collections.abc import Iterable, Mapping
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
MAX_ATTEMPTS_READ = 64
BLOCK_BYTES = 8 * 1024**2
DEADLINE_SECONDS = 900
PRESIGN_SECONDS = 3600
LOCAL_ARCHIVE = Path("vast_provider_run") / "vast_provider_runtime_output.zip"
PHASES = ("run", "setup_check")
_SETUP_SUFFIX = "--prestart-readiness"
_NAME = re.compile(r"(?:failed-)?([a-z][a-z0-9_]{0,63})\.mp4")
_CAMERA = re.compile(r"[a-z][a-z0-9_]{0,63}")
_CELL_RUN = re.compile(r"[0-9]{2}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_CODE = re.compile(r"[a-z][a-z0-9_:.=-]{0,199}")
_VIDEO_FIELDS = ("cell_id", "phase", "camera", "sha256", "size_bytes", "duration_seconds")
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
            return None if not timescale or unknown else round(duration / timescale, 3)
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


def _partial(target: Path) -> Path:
    """A copy in progress, beside its target (``_prepare_partial`` creates only that directory)."""
    return target.with_name("." + target.name + ".partial")


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


def _copy_promoted(rows, directory: Path, index: Mapping[str, Any], *, opener=None) -> tuple[list, dict, list]:
    from .provider_output_member_index import inflate_step_bytes
    from .provider_output_range_ingestion import CasArchiveSource, _materialize_member
    from .provider_output_range_transport import ProviderOutputTransportError
    from .task_evaluation_configured_scene_object_store import presign_configured_scene_artifact

    reference = index["archive"]["durable_reference"]
    source = CasArchiveSource(reference, opener=opener, block_bytes=BLOCK_BYTES, deadline_seconds=DEADLINE_SECONDS,
                              presign=lambda: presign_configured_scene_artifact(
                                  reference=reference, expiration_seconds=PRESIGN_SECONDS))
    source.durable_reference()
    reader = source.open(maximum_archive_bytes=index["archive"]["size"])
    if reader.identity["size_bytes"] != index["archive"]["size"]:
        raise ValueError("beta_screening_review_videos_archive_size_mismatch")
    copied, blockers = [], []
    for row in rows:
        member = row["member"]
        target = directory / row["relative_path"]
        if not _kept(target, member["size"], member["sha256"]):
            target.unlink(missing_ok=True)
            if reader.transferred_bytes + member["compressed_size"] > MAX_TOTAL_BYTES:
                blockers.append("beta_screening_review_videos_transfer_budget_reached")
                break
            try:
                _materialize_member(reader, member, target, _partial(target),
                                    step_bytes=inflate_step_bytes(BLOCK_BYTES))
            except ProviderOutputTransportError as exc:
                blockers.append(_blocker(exc, "beta_screening_review_videos_transport_failed"))
                break  # A pinned reader that failed once fails every later read.
            except ValueError as exc:
                blockers.append(_blocker(exc, "beta_screening_review_videos_member_invalid"))
                continue
        copied.append({**row, "sha256": member["sha256"]})
    transfer = {"transferred_bytes": reader.transferred_bytes, "http_request_count": reader.request_count}
    return copied, transfer, blockers


def _copy_local(rows, directory: Path, archive: zipfile.ZipFile, capture_root: Path,
                disk_usage) -> tuple[list, dict, list]:
    from .provider_output_range_ingestion import FLOOR_BYTES, _extract_member

    copied, blockers = [], []
    for row in rows:
        info = row["member"]["info"]
        target = directory / row["relative_path"]
        if _kept(target, info.file_size, None):
            copied.append({**row, "sha256": _sha256_file(target)})
            continue
        target.unlink(missing_ok=True)
        try:
            record = _extract_member(archive, info, target, _partial(target), root=capture_root,
                                     reserve=FLOOR_BYTES, disk_usage_provider=disk_usage)
        except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:  # RuntimeError: encrypted entry
            blockers.append(_blocker(exc, "beta_screening_review_videos_member_invalid"))
            continue
        os.chmod(target, 0o440)
        copied.append({**row, "sha256": record["sha256"]})
    return copied, {"transferred_bytes": 0, "http_request_count": 0}, blockers


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
    import shutil

    from .provider_output_range_ingestion import FLOOR_BYTES

    if capture_root is None:
        from .team_scene_packet_worker import resolve_capture_root
        capture_root = resolve_capture_root(plan["source"])  # refuses a withdrawn capture
    capture_root = Path(capture_root)
    lane_root = Path(str(adapter.get("attempt_root") or ""))
    if (not lane_root.is_absolute() or lane_root.is_symlink() or not lane_root.is_dir()
            or not lane_root.resolve().is_relative_to(attempt_dir)):
        return not_retained("beta_screening_review_videos_lane_root_unbound")
    directory = capture_root / "pipeline" / DIRNAME / attempt_dir.name
    existing = read_manifest(directory / MANIFEST_NAME)
    if existing is not None and existing.get("attempt_id") == attempt_dir.name:
        return summary_reference(existing, directory / MANIFEST_NAME)
    kind, value = _source(lane_root)
    if kind == "none":
        return not_retained("beta_screening_review_videos_source_unavailable")
    binding = {"run_id": str(prepared["run_id"]), "candidate_id": str(candidate["candidate_id"]),
               "cell_ids": [str(cell["cell_id"]) for cell in candidate["scenario_cells"]]}
    archive = zipfile.ZipFile(value) if kind == "local_archive" else None
    try:
        members = (value["members"] if archive is None else
                   [{"path": info.filename, "kind": "directory" if info.is_dir() else "file", "info": info,
                     "size": info.file_size, "compressed_size": info.compress_size} for info in archive.infolist()])
        rows, omitted = select_review_videos(members, **binding)
        if not rows:
            return not_retained("beta_screening_review_videos_none_found")
        needed = sum(row["member"]["size"] for row in rows)
        if (not capture_root.is_dir() or any(part.is_symlink() for part in (
                capture_root, capture_root / "pipeline", directory.parent, directory, directory / "videos"))):
            return not_retained("beta_screening_review_videos_destination_unsafe")
        if (disk_usage or shutil.disk_usage)(capture_root).free < needed + FLOOR_BYTES:
            return not_retained("beta_screening_review_videos_disk_space_insufficient")
        try:
            copied, transfer, blockers = (_copy_promoted(rows, directory, value, opener=opener) if archive is None
                                          else _copy_local(rows, directory, archive, capture_root, disk_usage))
        finally:
            _tidy(directory)
    finally:
        if archive is not None:
            archive.close()
    if not copied:
        return not_retained(*blockers, "beta_screening_review_videos_none_copied")
    if set(omitted) & _BOUNDS:
        blockers.append("beta_screening_review_videos_bound_reached")
    videos = [{"cell_id": row["cell_id"], "episode_id": row["episode_id"], "phase": row["phase"],
               "camera": row["camera"], "size_bytes": row["member"]["size"], "sha256": row["sha256"],
               "duration_seconds": mp4_duration_seconds(directory / row["relative_path"]),
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
    path = _write_manifest(directory, manifest)
    return summary_reference(manifest, path)


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
                   for row in videos)):
        return None
    return json.loads(json.dumps(dict(value)))


def publication_entry(candidate_id: str, summary: Mapping[str, Any]) -> dict[str, Any] | None:
    """The practice runs the website shows for one completed candidate: outcome and videos, no codes or paths."""
    reference = summary.get("review_videos")
    if not isinstance(reference, Mapping) or reference.get("status") not in {"retained", "partial"}:
        return None
    runs = []
    for outcome in summary.get("episode_outcomes") or []:
        videos = [{"sha256": row["sha256"], "camera": row["camera"], "phase": row["phase"],
                   "size_bytes": row["size_bytes"], "duration_seconds": row["duration_seconds"]}
                  for row in reference["videos"] if row["cell_id"] == outcome["cell_id"]]
        runs.append({"cell_id": outcome["cell_id"], "outcome": outcome["outcome"], "videos": videos})
    return {"candidate_id": candidate_id, "manifest_digest": reference["manifest_digest"], "practice_runs": runs}


def open_video(*, request_id: str, capture_id: str, digest: str, storage_root: Path | None) -> tuple[Any, int]:
    """One retained video's open file and size for a WebApp view the WebApp already authorized.

    Callers name a capture and a digest, never a path. A capture that is
    withdrawn, missing or ambiguous, or a digest no sealed manifest of that
    capture names, is ``PreviewUnavailable``.
    """
    from .website_atlas_preview_store import PreviewUnavailable, capture_name_matches_request, capture_withdrawn

    if (storage_root is None or not capture_name_matches_request(request_id, capture_id)
            or not _DIGEST.fullmatch(str(digest))):
        raise PreviewUnavailable("beta_screening_review_video_unavailable")
    roots = [root for root in Path(storage_root).glob(f"*/scenes/site-{request_id}/captures/{capture_id}")
             if root.is_dir() and not root.is_symlink()]
    if len(roots) != 1 or capture_withdrawn(roots[0]):
        raise PreviewUnavailable("beta_screening_review_video_unavailable")
    base = roots[0] / "pipeline" / DIRNAME
    attempts = sorted(base.iterdir())[:MAX_ATTEMPTS_READ] if base.is_dir() and not base.is_symlink() else []
    for attempt in attempts:
        manifest = read_manifest(attempt / MANIFEST_NAME) if not attempt.is_symlink() else None
        if (manifest is None or manifest.get("attempt_id") != attempt.name or manifest.get("capture_id") != capture_id
                or manifest.get("site_id") != f"site-{request_id}"):
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
    "DIRNAME", "MANIFEST_NAME", "MAX_TOTAL_BYTES", "MAX_VIDEOS", "MAX_VIDEO_BYTES", "SCHEMA",
    "mp4_duration_seconds", "not_retained", "open_video", "publication_entry", "read_manifest", "retain",
    "select_review_videos", "summary_reference", "valid_reference",
]
