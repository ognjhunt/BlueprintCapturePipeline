"""Hosted SAM 3.1 for task masks; original frame identity survives CPU encoding."""
from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError
from urllib.request import Request

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


def meta_api_key() -> str:
    value = os.getenv("META_MODEL_API_KEY", "").strip()
    if not value:
        from .gpu_render_providers import _read_secret
        value = (_read_secret("meta_model_api_key") or "").strip()
    if not value or any(c.isspace() for c in value):
        raise ValueError("meta_model_api_key_missing_or_invalid")
    return value


def request_binding(*, frame_registry: Sequence[Mapping[str, Any]],
                    frame_artifacts: Sequence[Mapping[str, Any]], prompts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"profile": PROFILE, "frame_registry": list(frame_registry),
            "frame_artifacts": [{k: row[k] for k in ("source_frame_id", "sha256")} for row in frame_artifacts],
            "prompts": list(prompts)}


def parse_tracks(response: Mapping[str, Any], *, prompt: Mapping[str, Any],
                 registry: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, video_segmentation_format

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
    parser = video_segmentation_format().create_parser()
    parser.push(text)
    parsed = parser.finish(CompletedOutcome()).result
    if parsed.diagnostics or any(row.kind == "text" and row.text.strip() for row in parsed.records):
        raise ValueError("meta_sam_output_malformed")
    observations: dict[str, dict[int, dict[str, Any]]] = {}
    for record in parsed.records:
        if record.kind != "mask":
            continue
        index = record.frame.frame_index if record.frame else -1
        if not 0 <= index < len(registry):
            raise ValueError("meta_sam_frame_index_invalid")
        box = [record.bounds.left, record.bounds.top, record.bounds.right, record.bounds.bottom]
        if any(not math.isfinite(v) or int(v) != v for v in box):
            raise ValueError("meta_sam_mask_bounds_invalid")
        left, top, right, bottom = map(int, box)
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("meta_sam_mask_bounds_invalid")
        if not (0 < record.mask.width * record.mask.height <= width * height):
            raise ValueError("meta_sam_mask_dimensions_invalid")
        raster = np.frombuffer(decode_mask_to_raster(record.mask), dtype=np.uint8).reshape(record.mask.height, record.mask.width)
        if raster.shape != (bottom - top, right - left):
            raster = np.asarray(Image.fromarray(raster).resize((right - left, bottom - top), Image.Resampling.NEAREST))
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[top:bottom, left:right] = raster
        edges = np.diff(np.pad(mask.reshape(-1).astype(np.int8), (1, 1)))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        observations.setdefault(record.object_id, {})[index] = {
            "source_frame_id": registry[index]["source_frame_id"], "width": width, "height": height,
            "runs": [{"start": int(start), "length": int(end - start)} for start, end in zip(starts, ends)],
        }
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
                   opener: Any = _open_no_redirect) -> dict[str, Any]:
    binding = request_binding(frame_registry=frame_registry, frame_artifacts=frame_artifacts, prompts=prompts)
    digest = canonical_digest(binding)
    require_paid_resource_admission_grant(admission_grant, resource_class="evaluator_api",
                                          allocation_binding_digest=digest, require_allocation_binding=True)
    budget = admission.get("maximum_cost_usd")
    image_request = len(frame_registry) == 1
    unit_cost = 0.0025 if image_request else max(50, len(frame_registry)) * PRICE_PER_FRAME_USD
    if (admission.get("allocation_binding_digest") != digest or admission.get("external_disclosure_allowed") is not True
            or isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget)
            or budget < unit_cost * len(prompts)):
        raise ValueError("meta_sam_authorization_missing")
    if not prompts or len({p["prompt_id"] for p in prompts}) != len(prompts):
        raise ValueError("meta_sam_prompts_invalid")
    for prompt in prompts:
        if not str(prompt.get("text", "")).strip() or len(prompt["text"]) > 200:
            raise ValueError("meta_sam_prompt_invalid")
    token = meta_api_key()
    root = output_root / digest[7:]
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "binding.json", binding)
    clip = encode_clip(registry=frame_registry, artifacts=frame_artifacts, root=root)
    media = ({"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode((root / "frame-000000.png").read_bytes()).decode()}
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
            if intent_path.exists():
                raise ValueError("meta_sam_submission_requires_reconciliation")
            payload = {"model": MODEL, "stream": False, "metadata": {"mask_encoding": "one_bit"}, "input": [
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": prompt["text"]},
                    media]}]}
            request = Request(ENDPOINT, data=json.dumps(payload).encode(), method="POST",
                              headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
            with intent_path.open("x") as file:
                json.dump({"binding_digest": digest, "clip_digest": _sha256_file(clip)}, file)
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
            receipt = {"binding_digest": digest, "clip_digest": _sha256_file(clip), "response": response,
                       "reserved_cost_usd": unit_cost}
            temporary = result_path.with_suffix(".tmp")
            write_json(temporary, receipt)
            os.replace(temporary, result_path)
        processed = (response.get("usage") or {}).get("video_frames_processed")
        if not image_request and (isinstance(processed, bool) or not isinstance(processed, int) or processed < 0):
            raise ValueError("meta_sam_usage_missing")
        if not image_request and processed > max(50, len(frame_registry)):
            raise ValueError("meta_sam_usage_exceeds_reservation")
        tracks.extend(parse_tracks(response, prompt=prompt, registry=frame_registry))
        receipts.append({"path": str(result_path), "sha256": _sha256_file(result_path)})
    result = {"schema_version": "website_meta_sam31_tracks.v1", "status": "completed", "binding_digest": digest,
              "profile": PROFILE, "tracks": tracks, "responses": receipts, "claim_ceiling": "development_only"}
    write_json(root / "tracks.json", result)
    return result
