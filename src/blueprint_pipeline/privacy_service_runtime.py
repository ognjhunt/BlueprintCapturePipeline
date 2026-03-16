"""Concrete runtime helpers for GPU-backed privacy runner services.

The HTTP service accepts JSON requests with a shared envelope:

- ``input_video_uri`` or ``input_video_path``: source walkthrough video
- ``output_json_uri`` or ``output_json_path``: where the result manifest should be written

SAM3 requests additionally accept:

- ``masks_prefix_uri`` or ``masks_dir_path``: destination for per-frame person masks
- ``prompt``: text prompt, defaults to ``person``
- ``sam3_weights_path``: optional local path, ``gs://`` URI, or HTTPS URL to the checkpoint

VIP requests additionally accept:

- ``masks_prefix_uri`` or ``masks_dir_path``: source masks from SAM3
- ``output_video_uri`` or ``output_video_path``: cleaned walkthrough destination
- ``arkit_depth_prefix_uri`` / ``arkit_confidence_prefix_uri``: optional ARKit depth bundles
- ``preferred_depth_source``: ``arkit`` or ``depth_anything``
- ``vip_model_path``: reserved for proprietary VIP model drops
- ``depth_anything_model_path``: local path, ``gs://`` URI, or HTTPS URL to the Depth Anything weights

DeepPrivacy2 requests additionally accept:

- ``output_video_uri`` or ``output_video_path``: anonymized walkthrough destination
- ``deepprivacy2_model_path``: optional local path, ``gs://`` URI, or HTTPS URL used as ``TORCH_HOME``
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
from urllib import request as urllib_request

from .common import (
    ensure_dir,
    ensure_local_uri_path,
    is_gs_uri,
    parse_gs_uri,
    parse_bool,
    resolve_gs_uri_to_path,
)


def _gcs_root() -> Path:
    return Path(str(os.getenv("GCS_ROOT") or "/mnt/gcs").strip() or "/mnt/gcs")


def _json_path(target_root: Path, filename: str) -> Path:
    ensure_dir(target_root)
    return target_root / filename


def _string(value: Any) -> str:
    return str(value or "").strip()


def _to_jsonable(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(dict(payload)))


def _storage_client() -> Any:
    from google.cloud import storage as gcs_storage  # type: ignore[import-untyped]

    return gcs_storage.Client()


def _download_http_to_path(url: str, destination: Path) -> Path:
    ensure_dir(destination.parent)
    req = urllib_request.Request(url, headers={"User-Agent": "BlueprintCapturePipeline/1.0"})
    with urllib_request.urlopen(req, timeout=1800) as response:
        destination.write_bytes(response.read())
    return destination


def _materialize_input_file(
    *,
    uri: str,
    local_hint: str,
    working_path: Path,
) -> Path:
    hint_path = Path(local_hint) if local_hint else None
    if hint_path and hint_path.is_file():
        return hint_path
    if not uri:
        raise FileNotFoundError("missing_input_video")
    if is_gs_uri(uri):
        return ensure_local_uri_path(uri, gcs_root=_gcs_root(), scratch_dir=working_path.parent)
    if uri.startswith(("https://", "http://")):
        return _download_http_to_path(uri, working_path)
    local_path = Path(uri)
    if local_path.is_file():
        return local_path
    raise FileNotFoundError(f"input_file_not_found:{uri}")


def _materialize_prefix(
    *,
    uri: str,
    local_hint: str,
    working_dir: Path,
) -> Path:
    hint_path = Path(local_hint) if local_hint else None
    if hint_path and hint_path.is_dir():
        return hint_path
    if not uri:
        raise FileNotFoundError("missing_input_prefix")
    if not is_gs_uri(uri):
        local_path = Path(uri)
        if local_path.is_dir():
            return local_path
        raise FileNotFoundError(f"input_prefix_not_found:{uri}")

    mounted = resolve_gs_uri_to_path(uri, _gcs_root())
    if mounted.is_dir():
        return mounted

    parsed = parse_gs_uri(uri.rstrip("/"))
    prefix = parsed.key.rstrip("/") + "/"
    ensure_dir(working_dir)
    bucket = _storage_client().bucket(parsed.bucket)
    downloaded = False
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith("/"):
            continue
        relative = blob.name[len(prefix) :] if blob.name.startswith(prefix) else Path(blob.name).name
        target_path = working_dir / relative
        ensure_dir(target_path.parent)
        blob.download_to_filename(str(target_path))
        downloaded = True
    if not downloaded:
        raise FileNotFoundError(f"prefix_not_found:{uri}")
    return working_dir


def _materialize_model_path(value: str, *, working_dir: Path, default_name: str) -> str:
    raw = _string(value)
    if not raw:
        return ""
    path = Path(raw)
    if path.exists():
        return str(path)
    if is_gs_uri(raw):
        return str(ensure_local_uri_path(raw, gcs_root=_gcs_root(), scratch_dir=working_dir))
    if raw.startswith(("https://", "http://")):
        target = working_dir / default_name
        return str(_download_http_to_path(raw, target))
    return raw


def _copy_file_to_uri(source: Path, destination: str) -> None:
    if not destination:
        return
    if is_gs_uri(destination):
        mounted = resolve_gs_uri_to_path(destination, _gcs_root())
        try:
            ensure_dir(mounted.parent)
            shutil.copyfile(source, mounted)
            return
        except Exception:
            parsed = parse_gs_uri(destination)
            _storage_client().bucket(parsed.bucket).blob(parsed.key).upload_from_filename(str(source))
            return
    target = Path(destination)
    ensure_dir(target.parent)
    shutil.copyfile(source, target)


def _copy_directory_to_uri(source_dir: Path, destination: str) -> list[str]:
    materialized: list[str] = []
    if not destination:
        return materialized
    files = sorted(path for path in source_dir.rglob("*") if path.is_file())
    if is_gs_uri(destination):
        mounted = resolve_gs_uri_to_path(destination.rstrip("/"), _gcs_root())
        parsed = parse_gs_uri(destination.rstrip("/"))
        use_mount = False
        try:
            ensure_dir(mounted)
            use_mount = True
        except Exception:
            use_mount = False
        bucket = None if use_mount else _storage_client().bucket(parsed.bucket)
        prefix = parsed.key.rstrip("/")
        for file_path in files:
            rel = file_path.relative_to(source_dir).as_posix()
            object_uri = f"gs://{parsed.bucket}/{prefix}/{rel}"
            if use_mount:
                target = mounted / rel
                ensure_dir(target.parent)
                shutil.copyfile(file_path, target)
            else:
                bucket.blob(f"{prefix}/{rel}").upload_from_filename(str(file_path))
            materialized.append(object_uri)
        return materialized

    target_root = Path(destination)
    for file_path in files:
        rel = file_path.relative_to(source_dir)
        target = target_root / rel
        ensure_dir(target.parent)
        shutil.copyfile(file_path, target)
        materialized.append(str(target))
    return materialized


def _write_payload_json(payload: Mapping[str, Any], output_json_path: Path, output_json_uri: str) -> None:
    ensure_dir(output_json_path.parent)
    output_json_path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    if output_json_uri:
        _copy_file_to_uri(output_json_path, output_json_uri)


def _frames_with_suffix(directory: Optional[Path], suffixes: Iterable[str]) -> list[Path]:
    if directory is None or not directory.is_dir():
        return []
    wanted = {suffix.lower() for suffix in suffixes}
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in wanted)


def _load_depth_anything_runtime(model_path: str) -> tuple[Optional[Any], list[str]]:
    from .geometry_da3 import _infer_depth_with_runtime, _load_da3_runtime

    os.environ.setdefault("DA3_MODEL_NAME", str(os.getenv("DA3_MODEL_NAME") or "da3metric-large"))
    if model_path:
        os.environ["DA3_MODEL_PATH"] = model_path
    runtime, warnings = _load_da3_runtime("depth_anything")
    if runtime is None:
        return None, warnings
    runtime._blueprint_infer_depth = _infer_depth_with_runtime  # type: ignore[attr-defined]
    return runtime, warnings


def _ffprobe_duration(video_path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return 0.0
    try:
        return float((proc.stdout or "0").strip() or "0")
    except ValueError:
        return 0.0


def _run_sam3_backend(
    *,
    input_video: Path,
    masks_dir: Path,
    prompt: str,
    stage_name: str,
    weights_path: str,
) -> Dict[str, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np
        from PIL import Image
        from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore[import-not-found]
        from sam3.model_builder import build_sam3_image_model  # type: ignore[import-not-found]
    except Exception as exc:
        return {
            "status": "failed",
            "reason": f"sam3_runtime_unavailable:{exc.__class__.__name__}",
            "detail": str(exc),
        }

    frame_stride = max(1, int(os.getenv("PRIVACY_SAM3_FRAME_STRIDE") or "1"))
    score_threshold = float(os.getenv("PRIVACY_SAM3_SCORE_THRESHOLD") or "0.2")
    ensure_dir(masks_dir)
    for old_mask in masks_dir.glob("*.png"):
        old_mask.unlink()

    try:
        model = build_sam3_image_model(
            checkpoint_path=weights_path or None,
            load_from_HF=not bool(weights_path),
        )
        processor = Sam3Processor(model)
    except Exception as exc:
        return {
            "status": "failed",
            "reason": f"sam3_model_load_failed:{exc.__class__.__name__}",
            "detail": str(exc),
        }

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        return {"status": "failed", "reason": f"video_open_failed:{input_video}"}

    mask_paths: list[str] = []
    people_count = 0
    processed_frames = 0
    positive_frames = 0
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % frame_stride != 0:
                frame_index += 1
                continue
            processed_frames += 1
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            try:
                state = processor.set_image(image)
                output = processor.set_text_prompt(state=state, prompt=prompt or "person")
            except Exception as exc:
                return {
                    "status": "failed",
                    "reason": f"sam3_inference_failed:{exc.__class__.__name__}",
                    "detail": str(exc),
                }

            raw_masks = output.get("masks") if isinstance(output, Mapping) else None
            raw_scores = output.get("scores") if isinstance(output, Mapping) else None
            if raw_masks is None:
                frame_index += 1
                continue

            scores: list[float]
            if raw_scores is None:
                scores = []
            elif hasattr(raw_scores, "detach"):
                scores = [float(value) for value in raw_scores.detach().cpu().reshape(-1).tolist()]
            else:
                scores = [float(value) for value in getattr(raw_scores, "reshape", lambda *_a, **_k: raw_scores)(-1)]

            if hasattr(raw_masks, "detach"):
                masks_array = raw_masks.detach().cpu().numpy()
            else:
                masks_array = np.asarray(raw_masks)
            while masks_array.ndim > 3:
                masks_array = masks_array[0]
            if masks_array.ndim == 2:
                masks_array = masks_array[None, ...]
            if masks_array.ndim != 3:
                frame_index += 1
                continue

            kept = []
            for mask_idx, mask in enumerate(masks_array):
                score = scores[mask_idx] if mask_idx < len(scores) else 1.0
                if score < score_threshold:
                    continue
                kept.append(mask.astype("float32"))
            if not kept:
                frame_index += 1
                continue
            positive_frames += 1
            people_count = max(people_count, len(kept))
            combined = np.zeros_like(kept[0], dtype="uint8")
            for mask in kept:
                combined = np.maximum(combined, (mask > 0).astype("uint8") * 255)
            mask_path = masks_dir / f"frame_{frame_index:06d}.png"
            cv2.imwrite(str(mask_path), combined)
            mask_paths.append(str(mask_path))
            frame_index += 1
    finally:
        cap.release()

    return {
        "status": "succeeded",
        "runner_kind": "sam3",
        "prompt": prompt or "person",
        "stage_name": stage_name or None,
        "people_detected": bool(people_count),
        "people_count": int(people_count),
        "mask_paths": mask_paths,
        "frames_scanned": processed_frames,
        "frames_with_people": positive_frames,
        "frame_stride": frame_stride,
    }


def _read_depth_frame(path: Path) -> Any:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    if path.suffix.lower() == ".npy":
        depth = np.load(path).astype("float32")
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"depth_frame_missing:{path}")
        depth = depth.astype("float32")
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth


def _inpaint_with_depth(
    *,
    frame: Any,
    mask: Any,
    depth_map: Optional[Any],
    confidence_map: Optional[Any],
) -> Any:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    mask_u8 = (mask > 0).astype("uint8") * 255
    if not mask_u8.any():
        return frame
    radius = 3
    if depth_map is not None:
        masked_depth = depth_map[mask_u8 > 0]
        if masked_depth.size:
            span = float(masked_depth.max() - masked_depth.min())
            radius = max(3, min(9, int(round(3.0 + span * 2.0))))
    dilation = radius + 2
    if confidence_map is not None and confidence_map.size:
        mean_confidence = float(confidence_map[mask_u8 > 0].mean()) if (mask_u8 > 0).any() else 1.0
        if mean_confidence < 0.4:
            dilation = max(dilation, 9)
    kernel = np.ones((dilation, dilation), dtype="uint8")
    expanded = cv2.dilate(mask_u8, kernel, iterations=1)
    repaired = cv2.inpaint(frame, expanded, float(radius), cv2.INPAINT_TELEA)
    output = frame.copy()
    output[expanded > 0] = repaired[expanded > 0]
    return output


def _merge_audio_if_possible(*, source_video: Path, video_only_path: Path, output_video: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        shutil.copyfile(video_only_path, output_video)
        return
    proc = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_only_path),
            "-i",
            str(source_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-shortest",
            str(output_video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not output_video.is_file():
        shutil.copyfile(video_only_path, output_video)


def _run_vip_backend(
    *,
    input_video: Path,
    masks_dir: Path,
    output_video: Path,
    preferred_depth_source: str,
    arkit_depth_dir: Optional[Path],
    arkit_confidence_dir: Optional[Path],
    vip_model_path: str,
    depth_anything_model_path: str,
) -> Dict[str, Any]:
    del vip_model_path
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np
    except Exception as exc:
        return {
            "status": "failed",
            "reason": f"vip_runtime_unavailable:{exc.__class__.__name__}",
            "detail": str(exc),
        }

    depth_source = "arkit"
    arkit_depth_frames = _frames_with_suffix(arkit_depth_dir, {".png", ".jpg", ".jpeg", ".npy", ".tif", ".tiff"})
    arkit_confidence_frames = _frames_with_suffix(
        arkit_confidence_dir, {".png", ".jpg", ".jpeg", ".npy", ".tif", ".tiff"}
    )
    if preferred_depth_source == "arkit" and arkit_depth_frames:
        depth_runtime = None
        depth_warnings: list[str] = []
    else:
        depth_source = "depth_anything"
        depth_runtime, depth_warnings = _load_depth_anything_runtime(depth_anything_model_path)
        if depth_runtime is None:
            return {
                "status": "failed",
                "reason": "depth_anything_runtime_unavailable",
                "warnings": depth_warnings,
            }

    ensure_dir(output_video.parent)
    masks = {path.stem: path for path in _frames_with_suffix(masks_dir, {".png"})}
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        return {"status": "failed", "reason": f"video_open_failed:{input_video}"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or 720
    temp_video = output_video.with_suffix(".video_only.mp4")
    writer = cv2.VideoWriter(
        str(temp_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        return {"status": "failed", "reason": f"video_writer_failed:{temp_video}"}

    masks_used = 0
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            mask_path = masks.get(f"frame_{frame_index:06d}")
            if mask_path is None:
                writer.write(frame)
                frame_index += 1
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or not mask.any():
                writer.write(frame)
                frame_index += 1
                continue
            depth_map = None
            confidence_map = None
            if depth_source == "arkit" and arkit_depth_frames:
                depth_path = arkit_depth_frames[min(frame_index, len(arkit_depth_frames) - 1)]
                depth_map = _read_depth_frame(depth_path)
                if arkit_confidence_frames:
                    confidence_path = arkit_confidence_frames[min(frame_index, len(arkit_confidence_frames) - 1)]
                    confidence_map = _read_depth_frame(confidence_path)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                infer = getattr(depth_runtime, "_blueprint_infer_depth", None)
                depth_map = infer(depth_runtime, rgb) if callable(infer) else None
                if depth_map is None:
                    return {
                        "status": "failed",
                        "reason": "depth_anything_inference_failed",
                        "warnings": depth_warnings,
                    }
            processed = _inpaint_with_depth(
                frame=frame,
                mask=mask,
                depth_map=depth_map,
                confidence_map=confidence_map,
            )
            writer.write(processed)
            masks_used += 1
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    _merge_audio_if_possible(source_video=input_video, video_only_path=temp_video, output_video=output_video)
    if temp_video.exists():
        temp_video.unlink()
    return {
        "status": "succeeded",
        "runner_kind": "vip",
        "backend": "depth_guided_inpaint",
        "output_video": str(output_video),
        "depth_source": depth_source,
        "frames_processed": frame_index,
        "masks_used": masks_used,
        "warnings": depth_warnings,
    }


def _deepprivacy2_repo_dir() -> Path:
    return Path(str(os.getenv("DEEPPRIVACY2_REPO_DIR") or "/opt/deepprivacy2").strip() or "/opt/deepprivacy2")


def _run_deepprivacy2_backend(
    *,
    input_video: Path,
    output_video: Path,
    deepprivacy2_model_path: str,
) -> Dict[str, Any]:
    repo_dir = _deepprivacy2_repo_dir()
    anonymize_script = repo_dir / "anonymize.py"
    config_path = repo_dir / "configs" / "anonymizers" / "face.py"
    if not anonymize_script.is_file() or not config_path.is_file():
        return {
            "status": "failed",
            "reason": f"deepprivacy2_repo_missing:{repo_dir}",
        }

    ensure_dir(output_video.parent)
    env = os.environ.copy()
    if deepprivacy2_model_path:
        env["TORCH_HOME"] = deepprivacy2_model_path
        env["DEEPPRIVACY2_MODEL_PATH"] = deepprivacy2_model_path
    command = [
        sys.executable,
        str(anonymize_script),
        str(config_path),
        "-i",
        str(input_video),
        "-o",
        str(output_video),
        "--track",
    ]
    proc = subprocess.run(
        command,
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0 or not output_video.is_file():
        return {
            "status": "failed",
            "reason": f"deepprivacy2_command_failed:{proc.returncode}",
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
    duration = _ffprobe_duration(output_video)
    segment_end = round(duration, 3) if duration > 0 else None
    face_segments = [f"0.0-{segment_end}" if segment_end is not None else "0.0-end"]
    return {
        "status": "succeeded",
        "runner_kind": "deepprivacy2",
        "output_video": str(output_video),
        "face_anonymized_segments": face_segments,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def execute_privacy_service_request(kind: str, body: Mapping[str, Any]) -> Dict[str, Any]:
    runner_kind = _string(kind).lower()
    if runner_kind not in {"sam3", "vip", "deepprivacy2"}:
        return {"status": "failed", "reason": f"unsupported_runner_kind:{runner_kind}"}

    with tempfile.TemporaryDirectory(prefix=f"privacy-{runner_kind}-") as temp_dir:
        workdir = Path(temp_dir)
        inputs_dir = workdir / "inputs"
        outputs_dir = workdir / "outputs"
        ensure_dir(inputs_dir)
        ensure_dir(outputs_dir)

        try:
            input_video = _materialize_input_file(
                uri=_string(body.get("input_video_uri")),
                local_hint=_string(body.get("input_video_path")),
                working_path=inputs_dir / "input_video.mov",
            )
        except Exception as exc:
            return {"status": "failed", "reason": str(exc)}

        output_json_path = outputs_dir / "result.json"
        output_json_uri = _string(body.get("output_json_uri"))

        if runner_kind == "sam3":
            try:
                masks_dir = _json_path(outputs_dir, "masks")
            except Exception:
                masks_dir = outputs_dir / "masks"
            weights_path = _materialize_model_path(
                _string(body.get("sam3_weights_path") or os.getenv("SAM3_WEIGHTS_PATH")),
                working_dir=inputs_dir,
                default_name="sam3.pt",
            )
            result = _run_sam3_backend(
                input_video=input_video,
                masks_dir=masks_dir,
                prompt=_string(body.get("prompt")) or "person",
                stage_name=_string(body.get("stage_name")),
                weights_path=weights_path,
            )
            if _string(result.get("status")).lower() == "succeeded":
                mask_paths = _copy_directory_to_uri(masks_dir, _string(body.get("masks_prefix_uri")))
                if not mask_paths and _string(body.get("masks_dir_path")):
                    mask_paths = _copy_directory_to_uri(masks_dir, _string(body.get("masks_dir_path")))
                result["mask_paths"] = mask_paths
            _write_payload_json(result, output_json_path, output_json_uri)
            return _to_jsonable(
                {
                    **result,
                    "output_json_uri": output_json_uri or None,
                    "output_json_path": str(output_json_path),
                }
            )

        if runner_kind == "vip":
            try:
                masks_dir = _materialize_prefix(
                    uri=_string(body.get("masks_prefix_uri")),
                    local_hint=_string(body.get("masks_dir_path")),
                    working_dir=inputs_dir / "masks",
                )
            except Exception as exc:
                result = {"status": "failed", "reason": str(exc)}
                _write_payload_json(result, output_json_path, output_json_uri)
                return _to_jsonable(result)

            arkit_depth_dir: Optional[Path] = None
            arkit_confidence_dir: Optional[Path] = None
            if _string(body.get("arkit_depth_prefix_uri")) or _string(body.get("arkit_depth_dir_path")):
                try:
                    arkit_depth_dir = _materialize_prefix(
                        uri=_string(body.get("arkit_depth_prefix_uri")),
                        local_hint=_string(body.get("arkit_depth_dir_path")),
                        working_dir=inputs_dir / "arkit_depth",
                    )
                except Exception:
                    arkit_depth_dir = None
            if _string(body.get("arkit_confidence_prefix_uri")) or _string(body.get("arkit_confidence_dir_path")):
                try:
                    arkit_confidence_dir = _materialize_prefix(
                        uri=_string(body.get("arkit_confidence_prefix_uri")),
                        local_hint=_string(body.get("arkit_confidence_dir_path")),
                        working_dir=inputs_dir / "arkit_confidence",
                    )
                except Exception:
                    arkit_confidence_dir = None
            depth_anything_model_path = _materialize_model_path(
                _string(body.get("depth_anything_model_path") or os.getenv("DEPTH_ANYTHING_MODEL_PATH")),
                working_dir=inputs_dir,
                default_name="depth_anything_model",
            )
            vip_model_path = _materialize_model_path(
                _string(body.get("vip_model_path") or os.getenv("VIP_MODEL_PATH")),
                working_dir=inputs_dir,
                default_name="vip_model",
            )
            local_output_video = outputs_dir / "vip_output.mov"
            result = _run_vip_backend(
                input_video=input_video,
                masks_dir=masks_dir,
                output_video=local_output_video,
                preferred_depth_source=_string(body.get("preferred_depth_source")) or "depth_anything",
                arkit_depth_dir=arkit_depth_dir,
                arkit_confidence_dir=arkit_confidence_dir,
                vip_model_path=vip_model_path,
                depth_anything_model_path=depth_anything_model_path,
            )
            if _string(result.get("status")).lower() == "succeeded":
                output_video_uri = _string(body.get("output_video_uri"))
                if output_video_uri:
                    _copy_file_to_uri(local_output_video, output_video_uri)
                    result["output_video_uri"] = output_video_uri
                elif _string(body.get("output_video_path")):
                    _copy_file_to_uri(local_output_video, _string(body.get("output_video_path")))
                    result["output_video_uri"] = None
                result["output_video"] = str(local_output_video)
            _write_payload_json(result, output_json_path, output_json_uri)
            return _to_jsonable(
                {
                    **result,
                    "output_json_uri": output_json_uri or None,
                    "output_json_path": str(output_json_path),
                }
            )

        deepprivacy2_model_path = _materialize_model_path(
            _string(body.get("deepprivacy2_model_path") or os.getenv("DEEPPRIVACY2_MODEL_PATH")),
            working_dir=inputs_dir,
            default_name="deepprivacy2_model",
        )
        local_output_video = outputs_dir / "deepprivacy2_output.mov"
        result = _run_deepprivacy2_backend(
            input_video=input_video,
            output_video=local_output_video,
            deepprivacy2_model_path=deepprivacy2_model_path,
        )
        if _string(result.get("status")).lower() == "succeeded":
            output_video_uri = _string(body.get("output_video_uri"))
            if output_video_uri:
                _copy_file_to_uri(local_output_video, output_video_uri)
                result["output_video_uri"] = output_video_uri
            elif _string(body.get("output_video_path")):
                _copy_file_to_uri(local_output_video, _string(body.get("output_video_path")))
                result["output_video_uri"] = None
            result["output_video"] = str(local_output_video)
        _write_payload_json(result, output_json_path, output_json_uri)
        return _to_jsonable(
            {
                **result,
                "output_json_uri": output_json_uri or None,
                "output_json_path": str(output_json_path),
            }
        )


def privacy_service_enabled() -> bool:
    return parse_bool(os.getenv("PRIVACY_PIPELINE_ENABLED"), default=True)
