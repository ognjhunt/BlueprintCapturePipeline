"""Hosted SAM 3.1 for task masks; original frame identity survives CPU encoding."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError
from urllib.request import Request
import zlib

import numpy as np
from PIL import Image

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file
from .paid_resource_admission import PaidResourceAdmissionGrant, require_paid_resource_admission_grant
from .semantic_teacher_image_edit_worker import _open_no_redirect
from .website_track_mask_codec import (canonical_track_digest, load_track_json, plain_runs, raster_runs,
                                       write_track_json)

MODEL = "sam-3.1"
ENDPOINT = "https://api.meta.ai/v1/responses"
# https://dev.meta.ai/docs/pricing-rate-limits#sam-pricing (2026-09-19).
PRICE_PER_FRAME_USD = 0.0002
PROFILE = {"provider": "meta_model_api", "model": MODEL, "mask_encoding": "one_bit",
           "parser": "meta-sam-parser==0.0.5", "price_per_frame_usd": PRICE_PER_FRAME_USD,
           "minimum_reserved_frames": 50, "single_frame_transport": "input_image", "price_per_image_usd": 0.0025}
_HOST_NODE = Path("/var/lib/blueprint/task-evaluation-inputs/system-runtime-prerequisites/"
                  "splat-render-v1/node/bin/node")


def _sam_parser_node() -> Path | None:
    if _HOST_NODE.is_file() and os.access(_HOST_NODE, os.X_OK):
        return _HOST_NODE
    found = shutil.which("node")
    return Path(found) if found else None


#: The parser decodes every mask of every frame. Its work grows with the response, so its
#: time limit does too: 300 s per MiB of provider text, at least 300 s and at most an hour.
#: capture-5b067498 (2026-11-06): a 1.8 MB response for 437 frames at 3840x2160 timed out at
#: a fixed 300 s on a loaded host, and the same response parsed in a later pass.
PARSER_SECONDS_PER_MIB = 300
PARSER_TIMEOUT_BOUNDS_S = (300, 3600)


def parser_timeout_seconds(text: str) -> int:
    size = len(text.encode("utf-8")) / (1024 * 1024)
    return int(min(PARSER_TIMEOUT_BOUNDS_S[1], max(PARSER_TIMEOUT_BOUNDS_S[0], math.ceil(PARSER_SECONDS_PER_MIB * size))))


#: HTTP statuses the provider answers when it refused a request without processing it.
#: A refusal it answered is a definite outcome, unlike a timeout or a dropped connection.
#: 429 (rate limited) and 503 (unavailable) refuse before any work; the job retries them later.
REFUSAL_STATUSES = frozenset({400, 401, 403, 404, 409, 413, 415, 422, 429, 503})
#: For an inline request only (nothing stored at the provider), 408 (request timeout) and 425 (too
#: early) also refuse before any work, and any other answered error (a 500, 502, 504...) may have
#: been processed: it is counted like an unanswered dispatch (``count_unanswered_dispatches``),
#: and its one resend gets its own reservation. A request naming an uploaded file keeps the rules
#: above exactly, since resending it would name a file its run has already deleted.
INLINE_REFUSAL_STATUSES = REFUSAL_STATUSES | {408, 425}


def retire_refused_dispatches(root: Path, pending: Sequence[int], *, inline: bool = False) -> int:
    """Archive pending prompts whose last dispatch the provider refused; return this binding's refusal count.

    The provider answered each with a refusal status (``failure-<i>.json``, written only from its
    HTTP response), so nothing was processed and the prompt is dispatched again, under a new
    admission digest so the website reserves it afresh and no one-dispatch grant is replayed.
    A 404 also retires the uploaded video it named (the provider no longer holds that file), so
    the next dispatch uploads it again. A prompt whose dispatch got no answer at all is counted as
    spent and sent once more (``count_unanswered_dispatches``). Any other answer, such as a 500,
    may have been processed: it is counted the same way.
    """
    archive = Path(root) / "refused"
    count = sum(1 for path in archive.glob("attempt-*") if path.is_dir()) if archive.is_dir() else 0
    for index in pending:
        intent, failure = Path(root) / f"intent-{index}.json", Path(root) / f"failure-{index}.json"
        if not intent.is_file() or not failure.is_file() or intent.is_symlink() or failure.is_symlink():
            continue
        try:
            status = json.loads(failure.read_text()).get("http_status")
        except ValueError:
            continue
        if isinstance(status, bool) or status not in (INLINE_REFUSAL_STATUSES if inline else REFUSAL_STATUSES):
            continue
        count += 1
        target = archive / f"attempt-{count}"
        target.mkdir(parents=True)
        os.replace(intent, target / intent.name)
        os.replace(failure, target / failure.name)
        if status == 404:
            for name in ("uploaded-video.json", "video-upload-intent.json", "video-cleanup.json"):
                if (Path(root) / name).is_file():
                    os.replace(Path(root) / name, target / name)
    return count


#: Owner decision 2026-11-07 ("Yes, resend once"), for every job: a SAM request a crashed
#: attempt left with no answer may be sent once more. The first stays counted as spent; the
#: resend has its own reservation inside the job's existing caps; both are recorded; a resend
#: that also gets no answer stops for review.
UNANSWERED_ARCHIVE = "unanswered"
COUNTED_AS_SPENT = "counted-as-spent.json"
COUNTED_AS_SPENT_SCHEMA = "meta_sam31_unanswered_dispatch.v1"
_ALLOCATION_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def _counted_as_spent(root: Path) -> dict[int, tuple[Path, dict[str, Any]]]:
    archive = Path(root) / UNANSWERED_ARCHIVE
    counted: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(archive.glob(f"attempt-*/{COUNTED_AS_SPENT}")) if archive.is_dir() else ():
        record = json.loads(path.read_text())
        index = record.get("prompt_index")
        if (path.is_symlink() or record.get("schema_version") != COUNTED_AS_SPENT_SCHEMA
                or type(index) is not int or index in counted):
            raise ValueError("meta_sam_unanswered_record_invalid")
        counted[index] = (path.parent, record)
    return counted


def count_unanswered_dispatches(root: Path, pending: Sequence[int], *, binding_digest: str,
                                limit: int | None = None, inline: bool = False) -> dict[int, dict[str, Any]]:
    """Count each pending prompt's unanswered dispatch as spent, once per prompt; return the resend marks.

    An intent with no response and no recorded answer is what a crashed attempt leaves (or a
    timeout, or a dropped connection): the provider may have processed and billed it, and its
    response is lost. Its reservation, named in the intent, stays held at its maximum: nothing
    here releases or settles it. One durable record says so, then the intent moves beside it
    under ``unanswered/attempt-N``, and the prompt is dispatched once more under a new
    admission digest that names the intent it resends. Each mark returned (prompt index ->
    ``resend_of``, ``resend_index``) goes into that admission and into the resend's own intent.
    A prompt already counted is never counted again: its resend, if it too gets no answer,
    still requires reconciliation. An intent that names no reservation of this binding is
    left for reconciliation, as is one beyond ``limit`` resends (a caller's own cap).
    """
    root = Path(root)
    counted = _counted_as_spent(root)
    archive = root / UNANSWERED_ARCHIVE
    attempts = sum(1 for path in archive.glob("attempt-*") if path.is_dir()) if archive.is_dir() else 0
    for index in pending:
        intent, failure = root / f"intent-{index}.json", root / f"failure-{index}.json"
        if not intent.is_file() or intent.is_symlink() or failure.is_symlink():
            continue
        answered_status = None
        if failure.exists() and not inline:
            continue  # a request naming an uploaded file: any answered error waits, as it always has
        if failure.exists():
            # An answered error the provider may have processed (a 500, 502, 504...) is counted like
            # no answer. A refusal that did no work was retired before this (``retire_refused_dispatches``).
            try:
                answered_status = json.loads(failure.read_text()).get("http_status")
            except ValueError:
                continue
            if isinstance(answered_status, bool) or not isinstance(answered_status, int) \
                    or answered_status in INLINE_REFUSAL_STATUSES:
                continue
        raw = intent.read_bytes()
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        if index in counted:
            target, record = counted[index]
            if record.get("intent_sha256") == digest and not (target / intent.name).exists():
                os.replace(intent, target / intent.name)  # finish an archive a crash interrupted
                if failure.is_file() and not (target / failure.name).exists():
                    os.replace(failure, target / failure.name)
            continue
        try:
            value = json.loads(raw)
        except ValueError:
            continue
        if type(value) is not dict:
            continue
        allocation = value.get("allocation_binding_digest")
        if (value.get("binding_digest") != binding_digest or "resend_of" in value or type(allocation) is not str
                or not _ALLOCATION_DIGEST.fullmatch(allocation) or (limit is not None and len(counted) >= limit)):
            continue
        attempts += 1
        target = archive / f"attempt-{attempts}"
        target.mkdir(parents=True)
        record = {"schema_version": COUNTED_AS_SPENT_SCHEMA, "status": "counted_as_spent", "outcome": "unknown",
                  **({"answered_http_status": answered_status} if answered_status is not None else {}),
                  "prompt_index": index, "binding_digest": binding_digest, "intent_sha256": digest,
                  "allocation_binding_digest": allocation, "clip_digest": value.get("clip_digest"),
                  "reservation": "held_at_maximum_never_released_or_settled_as_zero",
                  "resends_allowed": 1, "recorded_at": datetime.now(timezone.utc).isoformat()}
        write_json(target / COUNTED_AS_SPENT, record)
        os.replace(intent, target / intent.name)
        if failure.is_file():
            os.replace(failure, target / failure.name)
        counted[index] = (target, record)
    return {index: {"resend_of": record["intent_sha256"], "resend_index": 1}
            for index, (_, record) in counted.items()}


def _parse_js_rasters(text: str, accept: Any) -> bool:
    node = _sam_parser_node()
    if node is None:
        return False
    runner = Path(__file__).with_name("_sam_parser_js") / "runner.mjs"
    if not runner.is_file():
        raise ValueError("meta_sam_js_parser_bundle_missing")
    with tempfile.TemporaryDirectory(prefix="sam31-parse-") as temporary:
        root = Path(temporary)
        (root / "input.json").write_text(json.dumps({"text": text}))
        result = subprocess.run(
            [str(node), str(runner), str(root / "input.json"), str(root / "rasters.jsonl")],
            capture_output=True, timeout=parser_timeout_seconds(text), env={"PATH": "/usr/bin:/bin"}, check=False)
        if result.stderr.strip() == b"sam31_js_parser_output_malformed":
            raise ValueError("meta_sam_output_malformed")
        if result.returncode or result.stdout or not (root / "rasters.jsonl").is_file():
            raise ValueError("meta_sam_js_parser_failed")
        count = 0
        ended = False
        with (root / "rasters.jsonl").open() as stream:
            begin = json.loads(stream.readline())
            if begin != {"kind": "begin", "schema": "sam31_rasters.v1"}:
                raise ValueError("meta_sam_js_parser_output_invalid")
            for line in stream:
                row = json.loads(line)
                if row.get("kind") == "end":
                    if row.get("mask_count") != count or ended:
                        raise ValueError("meta_sam_js_parser_output_invalid")
                    ended = True
                    continue
                if ended or row.get("kind") != "mask":
                    raise ValueError("meta_sam_js_parser_output_invalid")
                bounds = row.get("bounds") or {}
                box = [bounds.get(key) for key in ("left", "top", "right", "bottom")]
                try:
                    if row.get("raster_encoding") != "deflate-raw-base64":
                        raise ValueError("encoding_invalid")
                    raster = zlib.decompress(base64.b64decode(row["raster"], validate=True), -15)
                    if len(raster) != row["width"] * row["height"]:
                        raise ValueError("raster_size_invalid")
                except (KeyError, TypeError, ValueError, zlib.error) as error:
                    raise ValueError("meta_sam_js_parser_output_invalid") from error
                accept(row["object_id"], row["frame_index"], box,
                       row["width"], row["height"], raster)
                count += 1
        if not ended:
            raise ValueError("meta_sam_js_parser_output_invalid")
    return True


def meta_api_key() -> str:
    value = os.getenv("META_MODEL_API_KEY", "").strip()
    if not value:
        from .gpu_render_providers import _read_secret
        value = (_read_secret("meta_model_api_key") or "").strip()
    if not value or any(c.isspace() for c in value):
        raise ValueError("meta_model_api_key_missing_or_invalid")
    return value


def request_binding(*, frame_registry: Sequence[Mapping[str, Any]],
                    frame_artifacts: Sequence[Mapping[str, Any]], prompts: Sequence[Mapping[str, Any]],
                    video_artifact: Mapping[str, Any] | None = None) -> dict[str, Any]:
    binding = {"profile": PROFILE, "frame_registry": list(frame_registry),
            "frame_artifacts": [{k: row[k] for k in ("source_frame_id", "sha256")} for row in frame_artifacts],
            "prompts": list(prompts)}
    if video_artifact is not None:
        binding["continuous_video"] = {key: video_artifact[key] for key in ("sha256", "source_video_digest", "encoding")}
        binding["video_transport"] = "meta_files_v1"
    return binding


def _upload_video(*, clip: Path, root: Path, token: str, opener: Any) -> str:
    receipt_path = root / "uploaded-video.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("clip_digest") != _sha256_file(clip):
            raise ValueError("meta_sam_uploaded_video_changed")
        return receipt["file_id"]
    # Uploads do not purchase segmentation, but an uncertain upload must not
    # silently accumulate copies of private footage on the provider. Its file id
    # is unknown, so that copy could never be deleted: unlike a SAM request, an
    # unanswered upload is never resent (privacy, not money, stops it).
    intent = root / "video-upload-intent.json"
    if intent.exists():
        raise ValueError("meta_sam_video_upload_requires_reconciliation")
    boundary = "blueprint-" + _sha256_file(clip)[7:39]
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nuser_data\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="walkthrough.mp4"\r\n'
            'Content-Type: video/mp4\r\n\r\n').encode() + clip.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
    with intent.open("x") as file:
        json.dump({"clip_digest": _sha256_file(clip)}, file)
    request = Request("https://api.meta.ai/v1/files", data=body, method="POST", headers={
        "Authorization": "Bearer " + token, "Content-Type": "multipart/form-data; boundary=" + boundary})
    with opener(request, timeout=120) as response:
        value = json.loads(response.read(100_000))
    file_id = value.get("id")
    if not isinstance(file_id, str) or not re.fullmatch(r"file-[A-Za-z0-9_-]{1,200}", file_id):
        raise ValueError("meta_sam_uploaded_file_id_invalid")
    write_json(receipt_path, {"file_id": file_id, "clip_digest": _sha256_file(clip)})
    return file_id


_CONTINUOUS_CRF_LADDER = (18, 23, 28)
_CONTINUOUS_VIDEO_MAX_BYTES = 32 * 1024**2
# Target this share of the bound so rate-control overshoot still fits.
_CONTINUOUS_VIDEO_FILL = 0.9
# Receipts written by earlier encodes stay reusable as-is.
_LEGACY_CONTINUOUS_ENCODINGS = frozenset({"upright_h264_crf18_veryfast_threads2_all_source_frames_v2"} | {
    f"upright_h264_crf{crf}_veryfast_threads2_source_timebase_all_source_frames_v3" for crf in (18, 23, 28)})


def _probe_video(path: Path) -> dict[str, Any]:
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "stream=width,height,time_base:frame=best_effort_timestamp_time", "-of", "json", str(path)],
        check=True, timeout=600, capture_output=True)
    return json.loads(result.stdout)


def prepare_continuous_video(*, source: Path, source_digest: str, root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep all source frames for tracking, independently of geometry sampling."""
    if _sha256_file(source) != source_digest:
        raise ValueError("meta_sam_source_video_changed")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "continuous-upright.mp4"
    receipt_path = root / "continuous-video.json"
    encodings = {crf: f"upright_h264_crf{crf}_maxrate_fit_veryfast_threads2_source_timebase_all_source_frames_v4"
                 for crf in _CONTINUOUS_CRF_LADDER}
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        video = receipt.get("video") or {}
        if (receipt.get("digest") != canonical_digest(receipt, digest_field="digest")
                or video.get("source_video_digest") != source_digest
                or video.get("encoding") not in {*encodings.values(), *_LEGACY_CONTINUOUS_ENCODINGS}
                or video.get("path") != str(destination) or not destination.is_file()
                or _sha256_file(destination) != video.get("sha256")):
            raise ValueError("meta_sam_prepared_video_receipt_invalid")
        return receipt["registry"], video
    original = _probe_video(source)
    if not 2 <= len(original["frames"]) <= 15000:
        raise ValueError("meta_sam_frame_count_invalid")
    # Phone clips are variable-frame-rate on a fine source timebase (1/600 for
    # iPhone MOV). Encoding on x264's default 1/framerate timebase shifts frames
    # by up to a frame interval, which the per-frame mapping check rejects, so
    # keep the demuxer timebase and the source's track timescale.
    numerator, _, denominator = str((original.get("streams") or [{}])[0].get("time_base", "")).partition("/")
    if numerator != "1" or not denominator.isdigit() or not 0 < int(denominator) <= 1_000_000:
        raise ValueError("meta_sam_source_time_base_invalid")
    # Cap the bitrate so the first encode fits the upload bound; an uncapped
    # CRF 18 pass of a 30-second phone clip came out at 50 MB and was discarded,
    # doubling the encode time. The cap follows from the clip's own duration.
    stamps = [float(frame["best_effort_timestamp_time"]) for frame in original["frames"]]
    duration = (stamps[-1] - stamps[0]) * len(stamps) / (len(stamps) - 1)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("meta_sam_frame_count_invalid")
    maxrate = str(int(_CONTINUOUS_VIDEO_FILL * _CONTINUOUS_VIDEO_MAX_BYTES * 8 / duration))
    # Full-resolution, every-frame tracking input; use the CPU preset rather
    # than x264's slower default. Allow for the shared service's CPU quota/load;
    # the former two-minute wall limit expired even for a 13-second walkthrough.
    # Never publish a timed-out partial as ready. A long or detailed walkthrough
    # can exceed the upload bound at CRF 18 (a 30-second 1080p HEVC phone clip
    # re-encodes to ~50 MB), so step down a fixed quality ladder; resolution and
    # frame identity never change, and the chosen rung is recorded in `encoding`.
    for crf in _CONTINUOUS_CRF_LADDER:
        fd, temporary = tempfile.mkstemp(prefix="continuous-", suffix=".mp4", dir=root)
        os.close(fd)
        partial = Path(temporary)
        try:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-map", "0:v:0", "-an",
                "-fps_mode", "passthrough", "-enc_time_base", "-1", "-c:v", "libx264", "-preset", "veryfast", "-threads", "2",
                "-crf", str(crf), "-maxrate", maxrate, "-bufsize", maxrate,
                "-pix_fmt", "yuv420p", "-video_track_timescale", denominator,
                "-movflags", "+faststart", str(partial)],
                check=True, timeout=600, capture_output=True)
            encoded = _probe_video(partial)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        if len(encoded["frames"]) != len(original["frames"]):
            partial.unlink(missing_ok=True)
            raise ValueError("meta_sam_encoded_frame_mapping_invalid")
        width, height = encoded["streams"][0]["width"], encoded["streams"][0]["height"]
        if min(width, height) <= 0 or max(width, height) > 4096:
            partial.unlink(missing_ok=True)
            raise ValueError("meta_sam_inline_video_limits_exceeded")
        if partial.stat().st_size <= _CONTINUOUS_VIDEO_MAX_BYTES:
            break
        partial.unlink(missing_ok=True)
    else:
        raise ValueError("meta_sam_inline_video_limits_exceeded")
    source_start = float(original["frames"][0]["best_effort_timestamp_time"])
    encoded_start = float(encoded["frames"][0]["best_effort_timestamp_time"])
    rows = []
    for index, (before, after) in enumerate(zip(original["frames"], encoded["frames"], strict=True)):
        pts = float(before["best_effort_timestamp_time"])
        if abs((pts - source_start) - (float(after["best_effort_timestamp_time"]) - encoded_start)) > 0.002:
            raise ValueError("meta_sam_encoded_frame_mapping_invalid")
        rows.append({"source_frame_id": f"decoded-{index:09d}", "model_frame_index": index,
                     "decoded_pts_seconds": pts, "width": width, "height": height,
                     "retained_video_digest": source_digest})
    partial.replace(destination)
    video = {"path": str(destination), "sha256": _sha256_file(destination),
             "source_video_digest": source_digest, "encoding": encodings[crf]}
    receipt = {"registry": rows, "video": video}
    receipt["digest"] = canonical_digest(receipt, digest_field="digest")
    temporary_receipt = receipt_path.with_suffix(".tmp")
    write_json(temporary_receipt, receipt)
    temporary_receipt.replace(receipt_path)
    return rows, video


def parse_tracks(response: Mapping[str, Any], *, prompt: Mapping[str, Any],
                 registry: Sequence[Mapping[str, Any]],
                 backend_out: list[str] | None = None) -> list[dict[str, Any]]:
    """Tracks whose observations hold each mask's whole-frame runs as a compact ``RunLengths``.

    ``runs`` reads, iterates, compares and encodes like the list of ``{"start", "length"}``
    dicts, without holding one Python dict per run (a 4K video's large surfaces reach millions).
    """
    from meta_sam_parser import (CompletedOutcome, InvalidSegmentationMaskError,
                                 SegmentationMaskIdentity, SegmentationMaskRecord,
                                 decode_mask_to_raster)
    from meta_sam_parser._segmentation import _SegmentationParser

    if response.get("status") != "completed":
        raise ValueError("meta_sam_response_incomplete")
    text = "".join(part["text"] for item in response.get("output", []) if item.get("type") == "message"
                   for part in item.get("content", []) if part.get("type") == "output_text")
    sizes = {(int(row["width"]), int(row["height"])) for row in registry}
    if len(sizes) != 1:
        raise ValueError("meta_sam_frame_dimensions_inconsistent")
    width, height = next(iter(sizes))
    # The official parser retains box-local bounds, but not the source dimensions.
    if any((int(w), int(h)) != (width, height) for w, h in re.findall(r";w=(\d+);h=(\d+)\|>", text)):
        raise ValueError("meta_sam_source_dimensions_mismatch")
    observations: dict[str, dict[int, dict[str, Any]]] = {}

    def accept_mask(object_id: str, index: int, box: Sequence[int],
                    mask_width: int, mask_height: int, decoded: bytes) -> None:
        if not 0 <= index < len(registry):
            raise ValueError("meta_sam_frame_index_invalid")
        if any(not math.isfinite(v) or int(v) != v for v in box):
            raise ValueError("meta_sam_mask_bounds_invalid")
        left, top, right, bottom = map(int, box)
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("meta_sam_mask_bounds_invalid")
        if not (0 < mask_width * mask_height <= width * height):
            raise ValueError("meta_sam_mask_dimensions_invalid")
        raster = np.frombuffer(decoded, dtype=np.uint8).reshape(mask_height, mask_width)
        if raster.shape != (bottom - top, right - left):
            raster = np.asarray(Image.fromarray(raster).resize((right - left, bottom - top), Image.Resampling.NEAREST))
        # The box's runs in the whole frame, as if pasted into it; no frame-sized array per mask.
        observations.setdefault(object_id, {})[index] = {
            "source_frame_id": registry[index]["source_frame_id"], "width": width, "height": height,
            "runs": raster_runs(raster, left=left, top=top, width=width),
        }

    class _SingleDecodeParser(_SegmentationParser):
        """The pinned Meta parser's grammar and checks, retaining each verified raster once."""

        def __init__(self) -> None:
            super().__init__("video")

        def _accept_mask(self, object_id: str, frame: Any, mask: Any, raw: str, bounds: Any) -> None:
            area = mask.width * mask.height
            if mask.width <= 0 or mask.height <= 0 or area > (1 << 53) - 1:
                self._diagnose("invalid_mask_size", "Mask dimensions must be positive.", raw)
                return
            try:
                decoded = decode_mask_to_raster(mask)
            except InvalidSegmentationMaskError as error:
                self._diagnose("invalid_mask_payload", str(error), raw)
                return
            identity = SegmentationMaskIdentity(
                media=self._media, frame_index=None if frame is None else frame.frame_index,
                object_id=object_id)
            revision = self._revisions.get(identity, 0) + 1
            self._revisions[identity] = revision
            record = SegmentationMaskRecord(
                order=len(self._records), object_id=object_id, frame=frame,
                identity=identity, revision=revision, mask=mask, bounds=bounds)
            self._add_record(record)
            accept_mask(object_id, frame.frame_index if frame else -1,
                        [bounds.left, bounds.top, bounds.right, bounds.bottom],
                        mask.width, mask.height, decoded)

    used_js = _parse_js_rasters(text, accept_mask)
    if not used_js:
        parser = _SingleDecodeParser()
        parser.push(text, emit=False)
        parsed = parser.finish(CompletedOutcome()).result
        if parsed.diagnostics or any(row.kind == "text" and row.text.strip() for row in parsed.records):
            raise ValueError("meta_sam_output_malformed")
    if backend_out is not None:
        backend_out.append("@meta-sam/parser@0.0.12" if used_js else "meta-sam-parser==0.0.5")
    # No invented confidence; IDs are never stitched across disappearance/occlusion.
    return [{"track_id": f"meta-sam31-{prompt['prompt_id']}-{oid}", "label": prompt["output_label"],
             "label_source": "model_inferred", "observations": [rows[i] for i in sorted(rows)]}
            for oid, rows in observations.items()]


#: The largest clip any request carries (an inline clip is further bound by the request ceiling).
CLIP_MAX_BYTES = 32 * 1024**2


def encode_clip(*, registry: Sequence[Mapping[str, Any]], artifacts: Sequence[Mapping[str, Any]], root: Path,
                rung: Mapping[str, Any] | None = None) -> Path:
    """The frames as one clip, losslessly (crf 0, yuv444p), or at an ``INLINE_CLIP_LADDER`` rung."""
    if not 1 <= len(registry) <= 15000 or len(artifacts) != len(registry):
        raise ValueError("meta_sam_frame_count_invalid")
    sizes = {(int(row["width"]), int(row["height"])) for row in registry}
    if len(sizes) != 1:
        raise ValueError("meta_sam_frame_dimensions_inconsistent")
    width, height = next(iter(sizes))
    if min(width, height) <= 0 or max(width, height) > 4096:
        raise ValueError("meta_sam_frame_dimensions_invalid")
    for index, (row, artifact) in enumerate(zip(registry, artifacts)):
        path = Path(artifact["path"])
        if row["source_frame_id"] != artifact["source_frame_id"] or _sha256_file(path) != artifact["sha256"]:
            raise ValueError("meta_sam_source_frame_changed")
        with Image.open(path) as image:
            if image.size != (width, height):
                raise ValueError("meta_sam_source_dimensions_mismatch")
            image.convert("RGB").save(root / f"frame-{index:06d}.png")
    clip = root / "retained-frames.mp4"
    # One encoded frame per registry row, with no resizing, interpolation or frame dropping.
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "1", "-i", str(root / "frame-%06d.png"),
                    "-frames:v", str(len(registry)), "-c:v", "libx264",
                    *(["-threads", str(rung["threads"])] if rung else []),
                    "-crf", str(rung["crf"] if rung else 0), "-pix_fmt", rung["pix_fmt"] if rung else "yuv444p",
                    "-movflags", "+faststart", str(clip)], check=True, timeout=120, capture_output=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height,nb_read_frames", "-of", "json", str(clip)],
                           check=True, timeout=60, capture_output=True)
    stream = json.loads(probe.stdout)["streams"][0]
    if (stream["width"], stream["height"], int(stream["nb_read_frames"])) != (width, height, len(registry)):
        raise ValueError("meta_sam_encoded_frame_mapping_invalid")
    if rung is None and clip.stat().st_size > CLIP_MAX_BYTES:
        raise ValueError("meta_sam_inline_clip_too_large")
    return clip


#: A multi-frame request sent inline (the clip as a base64 data URL in the request body) stays under
#: this many bytes, about half the size at which the provider refused one with 413 (capture-39ecdc97,
#: 2026-11-11: 17,324,561 bytes refused). Nothing is stored at the provider by an inline request.
INLINE_REQUEST_CEILING_BYTES = 8_000_000
#: The inline clip's encodings, tried in order until the request fits. The source is a phone video
#: that is already compressed. Measured on that job's 16 frames at 1920x1080 (SSIM against the
#: lossless clip; lossless was 12,993,229 bytes): crf 22 2,592,735 bytes, SSIM 0.9941; crf 26
#: 1,394,354, 0.9922; crf 30 780,206, 0.9898; crf 34 472,799, 0.9866. Two encoder threads keep the
#: output identical from run to run on one encoder build.
INLINE_CLIP_LADDER: tuple[dict[str, Any], ...] = tuple(
    {"encoding": f"h264_crf{crf}_yuv444p_threads2_v1", "crf": crf, "pix_fmt": "yuv444p", "threads": 2}
    for crf in (22, 26, 30, 34))
#: The lowest quality an inline clip is sent at: the last rung, SSIM 0.9866 against the lossless clip on
#: capture-39ecdc97's 16 frames. A clip that does not fit even here is not sent (``INLINE_CLIP_TOO_LARGE``).
INLINE_CLIP_QUALITY_FLOOR = INLINE_CLIP_LADDER[-1]
INLINE_CLIP_TOO_LARGE = "meta_sam_inline_clip_exceeds_request_ceiling"
#: How a clip encoded before the ladder existed was made (always lossless).
LEGACY_LOSSLESS_ENCODING = "h264_crf0_yuv444p_lossless"
CLIP_ENCODING_FILE = "clip-encoding.json"
#: Answers bought for an earlier clip of this binding, set aside when the clip changed.
SUPERSEDED_CLIP_ARCHIVE = "superseded-clip"


def _inline_body_bytes(clip: Path, prompts: Sequence[Mapping[str, Any]]) -> int:
    """The largest request body these prompts send with this clip inline."""
    encoded = 4 * math.ceil(clip.stat().st_size / 3)
    longest = max((str(prompt["text"]) for prompt in prompts), key=lambda text: len(json.dumps(text)), default="")
    shell = {"model": MODEL, "stream": False, "metadata": {"mask_encoding": "one_bit"}, "input": [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": longest},
                                                        {"type": "input_video", "video_url": "data:video/mp4;base64,"}]}]}
    return len(json.dumps(shell).encode()) + encoded


def encode_inline_clip(*, registry: Sequence[Mapping[str, Any]], artifacts: Sequence[Mapping[str, Any]],
                       root: Path, prompts: Sequence[Mapping[str, Any]]) -> tuple[Path, dict[str, Any]]:
    """The first ladder rung whose whole request fits under ``INLINE_REQUEST_CEILING_BYTES``, and its record.

    Recorded in ``clip-encoding.json`` and in each answer bought with it. If even the last rung does
    not fit, the step ends with ``INLINE_CLIP_TOO_LARGE``: sending it could never succeed.
    """
    for rung in INLINE_CLIP_LADDER:
        clip = encode_clip(registry=registry, artifacts=artifacts, root=root, rung=rung)
        body = _inline_body_bytes(clip, prompts)
        if body <= INLINE_REQUEST_CEILING_BYTES and clip.stat().st_size <= CLIP_MAX_BYTES:
            record = {"encoding": rung["encoding"], "clip_digest": _sha256_file(clip),
                      "clip_bytes": clip.stat().st_size, "largest_request_bytes": body,
                      "ceiling_bytes": INLINE_REQUEST_CEILING_BYTES}
            write_json(root / CLIP_ENCODING_FILE, record)
            return clip, record
    raise ValueError(INLINE_CLIP_TOO_LARGE)


def _retained_inline_clip(root: Path, *, answered_digests: set[str],
                          prompts: Sequence[Mapping[str, Any]]) -> tuple[Path, dict[str, Any]] | None:
    """This binding's retained clip, when its digest is the one its record or its answers name and its
    request fits under the ceiling: reused as it is, never encoded again (another encoder build or
    thread count would give other bytes)."""
    clip = root / "retained-frames.mp4"
    if not clip.is_file() or clip.is_symlink():
        return None
    digest = _sha256_file(clip)
    try:
        record = json.loads((root / CLIP_ENCODING_FILE).read_text())
    except (OSError, ValueError):
        record = None
    if isinstance(record, dict) and record.get("clip_digest") == digest:
        verified = record
    elif answered_digests == {digest}:
        verified = {"encoding": LEGACY_LOSSLESS_ENCODING, "clip_digest": digest}
    else:
        return None
    body = _inline_body_bytes(clip, prompts)
    if body > INLINE_REQUEST_CEILING_BYTES or clip.stat().st_size > CLIP_MAX_BYTES:
        return None
    return clip, {**verified, "clip_bytes": clip.stat().st_size, "largest_request_bytes": body,
                  "ceiling_bytes": INLINE_REQUEST_CEILING_BYTES}


def _clip_encoding_name(root: Path, clip_digest: str, current: Mapping[str, Any] | None) -> str:
    if current:
        return str(current["encoding"])
    try:
        record = json.loads((root / CLIP_ENCODING_FILE).read_text())
    except (OSError, ValueError):
        record = {}
    return str(record["encoding"]) if record.get("clip_digest") == clip_digest else LEGACY_LOSSLESS_ENCODING


SET_ASIDE_RECORD = "superseded.json"


def _recorded_rebuys(root: Path) -> list[dict[str, Any]]:
    """Each re-buy this binding recorded, oldest first (a record is written before any answer moves)."""
    archive = root / SUPERSEDED_CLIP_ARCHIVE
    rows = []
    for path in archive.glob(f"attempt-*/{SET_ASIDE_RECORD}") if archive.is_dir() else ():
        record = json.loads(path.read_text())
        rows.append((int(path.parent.name.removeprefix("attempt-")), {**record, "directory": str(path.parent)}))
    return [record for _, record in sorted(rows, key=lambda row: row[0])]


def _move_set_aside(root: Path, record: Mapping[str, Any]) -> None:
    target = Path(record["directory"])
    earlier = set(record["earlier_clip_digests"])
    for index in record["prompt_indices"]:
        response = root / f"response-{index}.json"
        if response.is_file() and json.loads(response.read_text()).get("clip_digest") not in earlier:
            continue  # bought again with the new clip since: it stays
        for name in (f"response-{index}.json", f"parsed-response-{index}.json", f"intent-{index}.json"):
            if (root / name).is_file() and not (target / name).exists():
                os.replace(root / name, target / name)
    if (root / "tracks.json").is_file() and not (target / "tracks.json").exists():
        os.replace(root / "tracks.json", target / "tracks.json")


def _finish_interrupted_set_asides(root: Path) -> None:
    """A run stopped between the record and its last move left answers in place: move them now."""
    for record in _recorded_rebuys(root):
        _move_set_aside(root, record)


def _set_aside_answers_for_another_clip(root: Path, answered: Sequence[int], *, clip_digest: str) -> int:
    """Set aside answers bought with an earlier clip (with their intents and parsed tracks); their count.

    The record goes first, fsynced, so every later run derives the re-buy from it, then each file
    moves. A run stopped anywhere in between is finished by the next (``_finish_interrupted_set_asides``).
    """
    archive = root / SUPERSEDED_CLIP_ARCHIVE
    count = sum(1 for path in archive.glob("attempt-*") if path.is_dir()) if archive.is_dir() else 0
    target = archive / f"attempt-{count + 1}"
    target.mkdir(parents=True)
    old = {json.loads((root / f"response-{index}.json").read_text()).get("clip_digest") for index in answered}
    record = {"reason": "clip_changed", "earlier_clip_digests": sorted(map(str, old)), "clip_digest": clip_digest,
              "prompt_indices": list(answered)}
    temporary = target / (SET_ASIDE_RECORD + ".tmp")
    with temporary.open("w") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target / SET_ASIDE_RECORD)
    _move_set_aside(root, {**record, "directory": str(target)})
    return count + 1


def run_meta_sam31(*, frame_registry: Sequence[Mapping[str, Any]], frame_artifacts: Sequence[Mapping[str, Any]],
                   prompts: Sequence[Mapping[str, Any]], output_root: Path, admission: Mapping[str, Any],
                   admission_grant: PaidResourceAdmissionGrant | None = None,
                   task_context: Mapping[str, Any] | None = None,
                   video_artifact: Mapping[str, Any] | None = None,
                   opener: Any = _open_no_redirect, source_video_digest: str | None = None,
                   require_retained_video: bool = False, maximum_cost_usd: float | None = None,
                   compact_runs: bool = False) -> dict[str, Any]:
    """Hosted SAM 3.1 masks for each prompt over the frames, bought once per binding, replayed without a
    second purchase.

    ``maximum_cost_usd`` is a caller's own ceiling on all this binding's dispatches. A resend
    of an unanswered dispatch is reserved only while it, the prompts and every refused
    dispatch's held reservation still fit under it; otherwise the unanswered one stays for
    reconciliation.

    Decoded tracks are hashed and written chunk by chunk from their compact runs (the bytes
    ``write_json`` and ``canonical_digest`` give for the list form). The returned tracks carry
    those compact ``RunLengths`` when ``compact_runs``; otherwise each observation's ``runs`` is
    the plain list of dicts, as before.
    """
    binding = request_binding(frame_registry=frame_registry, frame_artifacts=frame_artifacts, prompts=prompts,
                              video_artifact=video_artifact)
    if source_video_digest is not None:
        binding["atlas_source_video_digest"] = source_video_digest
    digest = canonical_digest(binding)
    root = output_root / digest[7:]
    pending = [i for i in range(len(prompts)) if not (root / f"response-{i}.json").is_file()]
    retained = bool(prompts) and not pending
    inline = video_artifact is None and len(frame_registry) > 1
    # Finish a set-aside a stopped run began, before anything else reads which prompts are answered.
    if inline and not require_retained_video:
        _finish_interrupted_set_asides(root)
        pending = [i for i in range(len(prompts)) if not (root / f"response-{i}.json").is_file()]
        retained = bool(prompts) and not pending
    # The newer answer rules (``INLINE_REFUSAL_STATUSES``) cover every request that names no upload.
    sent_inline = video_artifact is None
    refused = (retire_refused_dispatches(root, pending, inline=sent_inline)
               if pending and not require_retained_video else 0)
    image_request = len(frame_registry) == 1
    unit_cost = 0.0025 if image_request else max(50, len(frame_registry)) * PRICE_PER_FRAME_USD
    room = (None if maximum_cost_usd is None
            else max(0, math.floor(maximum_cost_usd / unit_cost + 1e-9) - len(prompts) - refused))
    # The resend marks of every pending prompt whose first dispatch got no answer: that one is
    # counted as spent (its reservation stays held), and this dispatch is its single resend.
    resends = ({index: mark for index, mark in count_unanswered_dispatches(
                    root, pending, binding_digest=digest, limit=room, inline=sent_inline).items() if index in pending}
               if pending and not require_retained_video else {})
    if require_retained_video:
        required = [root / name for name in ("binding.json", "tracks.json")]
        required.extend(root / f"{prefix}-{index}.json" for prefix in ("response", "parsed-response")
                        for index in range(len(prompts)))
        if (video_artifact is None or not retained or root.is_symlink()
                or any(path.is_symlink() or not path.is_file() for path in required)
                or json.loads((root / "binding.json").read_text()) != binding):
            raise ValueError("meta_sam_retained_video_required")
    # The original grant authorizes one dispatch of the whole prompt set. A
    # stopped worker may have retained only a prefix of its provider responses;
    # continuing that prefix needs a distinct, bounded grant for the missing
    # prompts rather than redispatching the already reserved parent binding.
    # A resend is never the original grant: the binding names each intent it resends, so the
    # website reserves it afresh beside the held first reservation, never "already_reserved".
    def admission_for(pending: Sequence[int]) -> str:
        # Every re-buy this binding recorded (``superseded-clip``) is part of the digest on every later
        # run, so a re-buy whose reservation was refused, or whose worker died, is never reserved
        # under the original binding again ("already_reserved"), nor bought twice under one hold.
        rebuys = _recorded_rebuys(root) if inline else []
        if len(pending) == len(prompts) and not refused and not resends and not rebuys:
            return digest
        return canonical_digest({
            "kind": "meta_sam31_pending_prompts_v1", "request_binding_digest": digest,
            "pending_prompt_indices": list(pending), **({"refused_dispatches": refused} if refused else {}),
            **({"unanswered_resends": [{"prompt_index": index, **resends[index]} for index in sorted(resends)]}
               if resends else {}),
            **({"clip_rebuy": len(rebuys), "clip_digest": rebuys[-1]["clip_digest"]} if rebuys else {})})

    admission_digest = admission_for(pending)
    if not prompts or len({p["prompt_id"] for p in prompts}) != len(prompts):
        raise ValueError("meta_sam_prompts_invalid")
    for prompt in prompts:
        if not str(prompt.get("text", "")).strip() or len(prompt["text"]) > 200:
            raise ValueError("meta_sam_prompt_invalid")
    token = meta_api_key() if not retained else ""
    if not require_retained_video:
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "binding.json", binding)
    clip_encoding, clip_digest = None, None
    answered_digests = {str(json.loads((root / f"response-{index}.json").read_text()).get("clip_digest"))
                        for index in range(len(prompts)) if index not in pending} if inline else set()
    if inline and retained and len(answered_digests) == 1:
        # Every prompt has its answer for one clip: nothing is encoded, sent or set aside.
        clip, clip_digest = root / "retained-frames.mp4", next(iter(answered_digests))
    elif inline:
        # Sent inline: the retained clip when it is verified and fits; otherwise the first rung of
        # ``INLINE_CLIP_LADDER`` whose whole request fits, retained for the next run.
        kept = _retained_inline_clip(root, answered_digests=answered_digests, prompts=prompts)
        clip, clip_encoding = kept or encode_inline_clip(registry=frame_registry, artifacts=frame_artifacts,
                                                         root=root, prompts=prompts)
    elif video_artifact is None:
        clip = encode_clip(registry=frame_registry, artifacts=frame_artifacts, root=root)
    else:
        clip = Path(video_artifact["path"])
        if frame_artifacts or image_request or _sha256_file(clip) != video_artifact["sha256"]:
            raise ValueError("meta_sam_continuous_video_invalid")
        probe = _probe_video(clip)
        stream = probe["streams"][0]
        if (len(probe["frames"]) != len(frame_registry) or not 2 <= len(frame_registry) <= 15000
                or clip.stat().st_size > 32 * 1024**2 or max(stream["width"], stream["height"]) > 4096
                or any((row["width"], row["height"]) != (stream["width"], stream["height"]) for row in frame_registry)):
            raise ValueError("meta_sam_encoded_frame_mapping_invalid")
    clip_digest = clip_digest or _sha256_file(clip)
    answered = [index for index in range(len(prompts)) if index not in pending]
    if (video_artifact is None and not image_request and answered and not require_retained_video
            and any(json.loads((root / f"response-{index}.json").read_text()).get("clip_digest") != clip_digest
                    for index in answered)):
        # The clip changed since those answers were bought (a new encoding, an encoder update): they
        # describe other pixels, so every prompt is bought again with this clip, under its own
        # reservation, never the earlier one's. The answers set aside stay retained and are recorded.
        if admission_grant is not None:
            # A caller's own grant names the request it was given for, not this new one.
            raise ValueError("meta_sam_caller_grant_covers_another_clip")
        _set_aside_answers_for_another_clip(root, answered, clip_digest=clip_digest)
        pending, retained = list(range(len(prompts))), False
        token = token or meta_api_key()
        admission_digest = admission_for(pending)
    if any((root / f"intent-{i}.json").exists() for i in pending):
        raise ValueError("meta_sam_submission_requires_reconciliation")
    for index in range(len(prompts)):
        if index in pending:
            continue
        receipt = json.loads((root / f"response-{index}.json").read_text())
        if receipt.get("binding_digest") != digest or receipt.get("clip_digest") != clip_digest:
            raise ValueError("meta_sam_retained_response_changed")
    if not retained:
        if admission_grant is None and task_context is not None:
            from .website_task_context import reserve_website_sam_spend
            admission, admission_grant = reserve_website_sam_spend(task_context=task_context,
                binding_digest=admission_digest, maximum_cost_usd=unit_cost * len(pending),
                request_count=len(pending), source_video_digest=source_video_digest)
        require_paid_resource_admission_grant(admission_grant, resource_class="evaluator_api",
                                              allocation_binding_digest=admission_digest, require_allocation_binding=True)
        budget = admission.get("maximum_cost_usd")
        if (admission.get("allocation_binding_digest") != admission_digest or admission.get("external_disclosure_allowed") is not True
                or isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget)
                or budget < unit_cost * len(pending)):
            raise ValueError("meta_sam_authorization_missing")
    def dispatch(request: Request, **kwargs: Any) -> Any:
        require_paid_resource_admission_grant(admission_grant, resource_class="evaluator_api",
            allocation_binding_digest=admission_digest, require_allocation_binding=True)
        return opener(request, **kwargs)

    media = None
    if not retained:
        media = ({"type": "input_video", "file_id": _upload_video(clip=clip, root=root, token=token, opener=dispatch)}
                 if video_artifact else
                 {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode((root / "frame-000000.png").read_bytes()).decode()}
                 if image_request else {"type": "input_video", "video_url": "data:video/mp4;base64," + base64.b64encode(clip.read_bytes()).decode()})
    try:
        tracks, receipts = [], []
        for index, prompt in enumerate(prompts):
            result_path, intent_path = root / f"response-{index}.json", root / f"intent-{index}.json"
            if result_path.is_file():
                receipt = json.loads(result_path.read_text())
                if receipt["binding_digest"] != digest or receipt["clip_digest"] != clip_digest:
                    raise ValueError("meta_sam_retained_response_changed")
                response = receipt["response"]
            else:
                if retained:
                    raise ValueError("meta_sam_retained_response_missing")
                if intent_path.exists():
                    raise ValueError("meta_sam_submission_requires_reconciliation")
                payload = {"model": MODEL, "stream": False, "metadata": {"mask_encoding": "one_bit"}, "input": [
                    {"type": "message", "role": "user", "content": [
                        {"type": "input_text", "text": prompt["text"]},
                        media]}]}
                request = Request(ENDPOINT, data=json.dumps(payload).encode(), method="POST",
                                  headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
                require_paid_resource_admission_grant(admission_grant, resource_class="evaluator_api",
                    allocation_binding_digest=admission_digest, require_allocation_binding=True)
                with intent_path.open("x") as file:
                    json.dump({"binding_digest": digest, "clip_digest": _sha256_file(clip),
                               "allocation_binding_digest": admission_digest, **resends.get(index, {})}, file)
                    file.flush()
                    os.fsync(file.fileno())
                try:
                    with dispatch(request, timeout=300) as result:
                        raw = result.read(64 * 1024**2 + 1)
                    if len(raw) > 64 * 1024**2:
                        raise ValueError("meta_sam_response_too_large")
                    response = json.loads(raw)
                except HTTPError as exc:
                    # Never include the request, key or a provider-echoed body in logs.
                    write_json(root / f"failure-{index}.json", {"http_status": exc.code, "binding_digest": digest})
                    raise ValueError(f"meta_sam_http_{exc.code}") from None
                receipt = {"binding_digest": digest, "clip_digest": _sha256_file(clip),
                           "allocation_binding_digest": admission_digest, "response": response,
                           "reserved_cost_usd": unit_cost, **resends.get(index, {}),
                           **({"clip_encoding": clip_encoding["encoding"]} if clip_encoding else {})}
                temporary = result_path.with_suffix(".tmp")
                write_json(temporary, receipt)
                os.replace(temporary, result_path)
            processed = (response.get("usage") or {}).get("video_frames_processed")
            if not image_request and (isinstance(processed, bool) or not isinstance(processed, int) or processed < 0):
                raise ValueError("meta_sam_usage_missing")
            if not image_request and processed > max(50, len(frame_registry)):
                raise ValueError("meta_sam_usage_exceeds_reservation")
            parsed_path = root / f"parsed-response-{index}.json"
            parsed_binding = {"response_digest": _sha256_file(result_path), "request_digest": digest,
                              "parser_revision": 1, "profile": PROFILE}
            if parsed_path.is_file():
                parsed = load_track_json(parsed_path)
                if (parsed.get("binding") != parsed_binding
                        or parsed.get("digest") != canonical_track_digest(parsed, digest_field="digest")):
                    raise ValueError("meta_sam_retained_tracks_changed")
                decoded = parsed["tracks"]
            else:
                if require_retained_video:
                    raise ValueError("meta_sam_retained_tracks_changed")
                backend: list[str] = []
                decoded = parse_tracks(response, prompt=prompt, registry=frame_registry, backend_out=backend)
                parsed = {"binding": parsed_binding, "tracks": decoded, "parser_runtime": backend[0]}
                parsed["digest"] = canonical_track_digest(parsed, digest_field="digest")
                temporary = parsed_path.with_suffix(".tmp")
                write_track_json(temporary, parsed)
                os.replace(temporary, parsed_path)
            tracks.extend(decoded)
            receipts.append({"path": str(result_path), "sha256": _sha256_file(result_path)})
        result = {"schema_version": "website_meta_sam31_tracks.v1", "status": "completed", "binding_digest": digest,
                  "profile": PROFILE, "tracks": tracks, "responses": receipts, "claim_ceiling": "development_only"}
        if inline:
            # The rung the answers were bought with, in the receipt every caller keeps.
            result["clip_encoding"] = _clip_encoding_name(root, clip_digest, clip_encoding)
        if require_retained_video:
            if load_track_json(root / "tracks.json") != result:
                raise ValueError("meta_sam_retained_tracks_changed")
        else:
            write_track_json(root / "tracks.json", result)
        return result if compact_runs else plain_runs(result)
    finally:
        if video_artifact and not retained:
            try:
                request = Request("https://api.meta.ai/v1/files/" + media["file_id"], method="DELETE",
                                  headers={"Authorization": "Bearer " + token})
                with opener(request, timeout=30) as response:
                    value = json.loads(response.read(100_000))
                write_json(root / "video-cleanup.json", {"file_id": media["file_id"], "deleted": value.get("deleted") is True})
            except Exception:
                write_json(root / "video-cleanup.json", {"file_id": media["file_id"], "deleted": False})
