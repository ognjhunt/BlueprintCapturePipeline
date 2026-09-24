"""Hosted SAM 3.1 for task masks; original frame identity survives CPU encoding."""
from __future__ import annotations

import base64
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
            capture_output=True, timeout=300, env={"PATH": "/usr/bin:/bin"}, check=False)
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
    # silently accumulate copies of private footage on the provider.
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
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[top:bottom, left:right] = raster
        edges = np.diff(np.pad(mask.reshape(-1).astype(np.int8), (1, 1)))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        observations.setdefault(object_id, {})[index] = {
            "source_frame_id": registry[index]["source_frame_id"], "width": width, "height": height,
            "runs": [{"start": int(start), "length": int(end - start)} for start, end in zip(starts, ends)],
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


def encode_clip(*, registry: Sequence[Mapping[str, Any]], artifacts: Sequence[Mapping[str, Any]], root: Path) -> Path:
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
                    "-frames:v", str(len(registry)), "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p",
                    "-movflags", "+faststart", str(clip)], check=True, timeout=120, capture_output=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height,nb_read_frames", "-of", "json", str(clip)],
                           check=True, timeout=60, capture_output=True)
    stream = json.loads(probe.stdout)["streams"][0]
    if (stream["width"], stream["height"], int(stream["nb_read_frames"])) != (width, height, len(registry)):
        raise ValueError("meta_sam_encoded_frame_mapping_invalid")
    if clip.stat().st_size > 32 * 1024**2:
        raise ValueError("meta_sam_inline_clip_too_large")
    return clip


def run_meta_sam31(*, frame_registry: Sequence[Mapping[str, Any]], frame_artifacts: Sequence[Mapping[str, Any]],
                   prompts: Sequence[Mapping[str, Any]], output_root: Path, admission: Mapping[str, Any],
                   admission_grant: PaidResourceAdmissionGrant | None = None,
                   task_context: Mapping[str, Any] | None = None,
                   video_artifact: Mapping[str, Any] | None = None,
                   opener: Any = _open_no_redirect) -> dict[str, Any]:
    binding = request_binding(frame_registry=frame_registry, frame_artifacts=frame_artifacts, prompts=prompts,
                              video_artifact=video_artifact)
    digest = canonical_digest(binding)
    root = output_root / digest[7:]
    pending = [i for i in range(len(prompts)) if not (root / f"response-{i}.json").is_file()]
    retained = bool(prompts) and not pending
    # The original grant authorizes one dispatch of the whole prompt set. A
    # stopped worker may have retained only a prefix of its provider responses;
    # continuing that prefix needs a distinct, bounded grant for the missing
    # prompts rather than redispatching the already reserved parent binding.
    admission_digest = (digest if len(pending) == len(prompts) else canonical_digest({
        "kind": "meta_sam31_pending_prompts_v1", "request_binding_digest": digest,
        "pending_prompt_indices": pending}))
    image_request = len(frame_registry) == 1
    unit_cost = 0.0025 if image_request else max(50, len(frame_registry)) * PRICE_PER_FRAME_USD
    if not prompts or len({p["prompt_id"] for p in prompts}) != len(prompts):
        raise ValueError("meta_sam_prompts_invalid")
    for prompt in prompts:
        if not str(prompt.get("text", "")).strip() or len(prompt["text"]) > 200:
            raise ValueError("meta_sam_prompt_invalid")
    token = meta_api_key() if not retained else ""
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "binding.json", binding)
    if video_artifact is None:
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
    if any((root / f"intent-{i}.json").exists() for i in pending):
        raise ValueError("meta_sam_submission_requires_reconciliation")
    clip_digest = _sha256_file(clip)
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
                request_count=len(pending))
        require_paid_resource_admission_grant(admission_grant, resource_class="evaluator_api",
                                              allocation_binding_digest=admission_digest, require_allocation_binding=True)
        budget = admission.get("maximum_cost_usd")
        if (admission.get("allocation_binding_digest") != admission_digest or admission.get("external_disclosure_allowed") is not True
                or isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget)
                or budget < unit_cost * len(pending)):
            raise ValueError("meta_sam_authorization_missing")
    media = None
    if not retained:
        media = ({"type": "input_video", "file_id": _upload_video(clip=clip, root=root, token=token, opener=opener)}
                 if video_artifact else
                 {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode((root / "frame-000000.png").read_bytes()).decode()}
                 if image_request else {"type": "input_video", "video_url": "data:video/mp4;base64," + base64.b64encode(clip.read_bytes()).decode()})
    tracks, receipts = [], []
    for index, prompt in enumerate(prompts):
        result_path, intent_path = root / f"response-{index}.json", root / f"intent-{index}.json"
        if result_path.is_file():
            receipt = json.loads(result_path.read_text())
            if receipt["binding_digest"] != digest or receipt["clip_digest"] != _sha256_file(clip):
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
            with intent_path.open("x") as file:
                json.dump({"binding_digest": digest, "clip_digest": _sha256_file(clip),
                           "allocation_binding_digest": admission_digest}, file)
                file.flush()
                os.fsync(file.fileno())
            try:
                with opener(request, timeout=300) as result:
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
                       "reserved_cost_usd": unit_cost}
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
            parsed = json.loads(parsed_path.read_text())
            if (parsed.get("binding") != parsed_binding
                    or parsed.get("digest") != canonical_digest(parsed, digest_field="digest")):
                raise ValueError("meta_sam_retained_tracks_changed")
            decoded = parsed["tracks"]
        else:
            backend: list[str] = []
            decoded = parse_tracks(response, prompt=prompt, registry=frame_registry, backend_out=backend)
            parsed = {"binding": parsed_binding, "tracks": decoded, "parser_runtime": backend[0]}
            parsed["digest"] = canonical_digest(parsed, digest_field="digest")
            temporary = parsed_path.with_suffix(".tmp")
            write_json(temporary, parsed)
            os.replace(temporary, parsed_path)
        tracks.extend(decoded)
        receipts.append({"path": str(result_path), "sha256": _sha256_file(result_path)})
    result = {"schema_version": "website_meta_sam31_tracks.v1", "status": "completed", "binding_digest": digest,
              "profile": PROFILE, "tracks": tracks, "responses": receipts, "claim_ceiling": "development_only"}
    write_json(root / "tracks.json", result)
    if video_artifact and not retained:
        try:
            request = Request("https://api.meta.ai/v1/files/" + media["file_id"], method="DELETE",
                              headers={"Authorization": "Bearer " + token})
            with opener(request, timeout=30) as response:
                value = json.loads(response.read(100_000))
            write_json(root / "video-cleanup.json", {"file_id": media["file_id"], "deleted": value.get("deleted") is True})
        except Exception:
            write_json(root / "video-cleanup.json", {"file_id": media["file_id"], "deleted": False})
    return result
