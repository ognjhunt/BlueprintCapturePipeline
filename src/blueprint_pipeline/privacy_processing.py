"""Privacy-preserving post-processing for capture walkthrough video."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib import parse as urllib_parse
from urllib import error as urllib_error
from urllib import request as urllib_request

from .common import (
    ensure_dir,
    ensure_local_uri_path,
    infer_storage_root_from_scene_path,
    parse_bool,
    utc_now_iso,
    write_json,
)
from .launch_proof_policy import production_forces_true
from .site_taxonomy import resolve_site_type


# Redaction target classes for the SAM3 detection prompt (audit finding R010).
# ``person`` is always targeted for every site type. Industrial sites
# (warehouses / factories / cold storage / stockrooms; see
# ``blueprint_pipeline.site_taxonomy``) also expose worker badges/ID cards,
# monitors/screens showing proprietary data, whiteboards/signage, and vehicle
# license plates, so the detection prompt is expanded to target those classes in
# addition to people. Non-industrial captures preserve the person-focused prompt.
PERSON_REDACTION_CLASS = "person"
INDUSTRIAL_SENSITIVE_REDACTION_CLASSES: tuple[str, ...] = (
    "text",
    "screen",
    "monitor",
    "badge",
    "id card",
    "vehicle license plate",
    "signage",
    "whiteboard",
)


def _redaction_classes_for_site(site_type: Optional[str]) -> tuple[list[str], bool]:
    """Resolve the ordered SAM3 target classes for ``site_type``.

    Returns ``(classes, is_industrial)``. ``person`` is always first; industrial
    site types additionally target the industrial-sensitive classes (badges,
    screens/monitors, license plates, signage/whiteboards). Non-industrial (and
    unrecognized) site types keep the historical person-only class set.
    """

    resolution = resolve_site_type(site_type)
    classes = [PERSON_REDACTION_CLASS]
    if resolution.is_industrial:
        for cls in INDUSTRIAL_SENSITIVE_REDACTION_CLASSES:
            if cls not in classes:
                classes.append(cls)
    return classes, resolution.is_industrial


def _detection_prompt(classes: Sequence[str]) -> str:
    """Compose the open-vocabulary SAM3 detection prompt from target classes."""

    ordered = [str(cls).strip() for cls in classes if str(cls).strip()]
    if not ordered:
        ordered = [PERSON_REDACTION_CLASS]
    return ". ".join(ordered)


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _env_flag(name: str, *, default: bool = False) -> bool:
    return parse_bool(os.getenv(name), default=default)


def _timeout_env(name: str, *, default: int) -> int:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _render_command(template: str, substitutions: Mapping[str, object]) -> list[str]:
    rendered = str(template or "").strip()
    for key, value in substitutions.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return shlex.split(rendered)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _run_json_command(
    *,
    command_template: str,
    substitutions: Mapping[str, object],
    timeout_seconds: int,
) -> Dict[str, Any]:
    command = _render_command(command_template, substitutions)
    if not command:
        return {"status": "failed", "reason": "empty_command"}
    output_json = Path(str(substitutions["OUTPUT_JSON"]))
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    payload = _load_json(output_json)
    if proc.returncode != 0:
        return {
            "status": "failed",
            "reason": f"command_failed:{proc.returncode}",
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            **payload,
        }
    if payload:
        return payload
    return {
        "status": "succeeded",
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def _http_runner_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = str(os.getenv("PRIVACY_RUNNER_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _runner_url_invalid_reason(url: str) -> str | None:
    text = str(url or "").strip()
    if not text:
        return "runner_url_missing"
    parsed = urllib_parse.urlsplit(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "runner_url_invalid_or_placeholder"
    if "REPLACE_ME" in text or "replace_me" in text.lower():
        return "runner_url_invalid_or_placeholder"
    return None


def _run_http_json(
    *,
    url: str,
    body: Mapping[str, object],
    timeout_seconds: int,
) -> Dict[str, Any]:
    invalid_reason = _runner_url_invalid_reason(url)
    if invalid_reason:
        return {"status": "failed", "reason": invalid_reason}
    request = urllib_request.Request(
        url,
        data=json.dumps(dict(body)).encode("utf-8"),
        headers=_http_runner_headers(),
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return {
            "status": "failed",
            "reason": f"http_error:{exc.code}",
            "detail": detail[-4000:],
        }
    except urllib_error.URLError as exc:
        return {
            "status": "failed",
            "reason": f"http_unreachable:{exc.reason}",
        }

    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {
            "status": "failed",
            "reason": "http_invalid_json",
            "detail": raw[-4000:],
        }
    if isinstance(payload, dict):
        return payload
    return {"status": "failed", "reason": "http_non_object_json"}


def _copy_or_remux_video(source: Path, destination: Path) -> Dict[str, Any]:
    ensure_dir(destination.parent)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        proc = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-c",
                "copy",
                str(destination),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and destination.is_file():
            return {"status": "succeeded", "mode": "remux"}
    shutil.copyfile(source, destination)
    return {"status": "succeeded", "mode": "copy"}


def _local_full_frame_redaction_enabled() -> bool:
    return _env_flag("PRIVACY_LOCAL_FULL_FRAME_REDACTION_ENABLED", default=False) or _env_flag(
        "BLUEPRINT_PRIVACY_LOCAL_FULL_FRAME_REDACTION",
        default=False,
    )


def _run_local_full_frame_redaction(source: Path, destination: Path) -> Dict[str, Any]:
    """Create an explicitly local, full-frame deidentified walkthrough."""

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {"status": "failed", "reason": "ffmpeg_not_found"}
    ensure_dir(destination.parent)
    width = max(24, int(os.getenv("PRIVACY_LOCAL_REDACTION_PIXEL_WIDTH") or "96"))
    blur = max(2, int(os.getenv("PRIVACY_LOCAL_REDACTION_BLUR") or "16"))
    vf = (
        f"scale={width}:-2,"
        f"boxblur={blur}:1,"
        "scale=trunc(iw*8/2)*2:trunc(ih*8/2)*2:flags=neighbor,"
        "format=yuv420p"
    )
    proc = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            vf,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "32",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not destination.is_file():
        return {
            "status": "failed",
            "reason": f"ffmpeg_redaction_failed:{proc.returncode}",
            "stderr": proc.stderr[-4000:],
        }
    return {
        "status": "succeeded",
        "mode": "full_frame_pixelation_blur",
        "audio_removed": True,
        "pixel_width": width,
        "blur_radius": blur,
        "output_video": str(destination),
    }


def _sam3_command_template() -> str:
    return str(os.getenv("PRIVACY_SAM3_COMMAND") or os.getenv("SAM3_COMMAND") or "").strip()


def _sam3_runner_url() -> str:
    return str(os.getenv("PRIVACY_SAM3_URL") or "").strip()


def _vip_command_template() -> str:
    return str(os.getenv("PRIVACY_VIP_COMMAND") or os.getenv("VIP_COMMAND") or "").strip()


def _vip_runner_url() -> str:
    return str(os.getenv("PRIVACY_VIP_URL") or "").strip()


def _depth_anything_command_template() -> str:
    return str(os.getenv("PRIVACY_DEPTH_ANYTHING_COMMAND") or os.getenv("DEPTH_ANYTHING_COMMAND") or "").strip()


def _depth_anything_runner_url() -> str:
    return str(os.getenv("PRIVACY_DEPTH_ANYTHING_URL") or _vip_runner_url()).strip()


def _deepprivacy_command_template() -> str:
    return str(
        os.getenv("PRIVACY_DEEPPRIVACY2_COMMAND") or os.getenv("DEEPPRIVACY2_COMMAND") or ""
    ).strip()


def _deepprivacy_runner_url() -> str:
    return str(os.getenv("PRIVACY_DEEPPRIVACY2_URL") or "").strip()


def _run_sam3(
    *,
    input_video: Path,
    input_video_uri: str,
    output_json: Path,
    output_json_uri: str,
    masks_dir: Path,
    masks_prefix_uri: str,
    stage_name: str,
    detection_classes: Sequence[str] = (PERSON_REDACTION_CLASS,),
) -> Dict[str, Any]:
    timeout_seconds = _timeout_env("PRIVACY_SAM3_TIMEOUT_SECONDS", default=3600)
    ensure_dir(masks_dir)
    prompt_classes = [str(cls).strip() for cls in detection_classes if str(cls).strip()] or [
        PERSON_REDACTION_CLASS
    ]
    prompt = _detection_prompt(prompt_classes)
    runner_url = _sam3_runner_url()
    if runner_url:
        payload = _run_http_json(
            url=runner_url,
            body={
                "input_video_uri": input_video_uri,
                "input_video_path": str(input_video),
                "output_json_uri": output_json_uri,
                "output_json_path": str(output_json),
                "masks_prefix_uri": masks_prefix_uri,
                "masks_dir_path": str(masks_dir),
                "prompt": prompt,
                "prompt_classes": prompt_classes,
                "stage_name": stage_name,
                "sam3_weights_path": str(os.getenv("SAM3_WEIGHTS_PATH") or ""),
            },
            timeout_seconds=timeout_seconds,
        )
    else:
        template = _sam3_command_template()
        if not template:
            return {"status": "failed", "reason": "sam3_runner_not_configured"}
        payload = _run_json_command(
            command_template=template,
            substitutions={
                "INPUT_VIDEO": input_video,
                "INPUT_VIDEO_URI": input_video_uri,
                "OUTPUT_JSON": output_json,
                "OUTPUT_JSON_URI": output_json_uri,
                "MASKS_DIR": masks_dir,
                "MASKS_PREFIX_URI": masks_prefix_uri,
                "PROMPT": prompt,
                "PROMPT_CLASSES": ",".join(prompt_classes),
                "STAGE_NAME": stage_name,
                "SAM3_WEIGHTS_PATH": str(os.getenv("SAM3_WEIGHTS_PATH") or ""),
            },
            timeout_seconds=timeout_seconds,
        )
    payload.setdefault("detection_classes", list(prompt_classes))
    payload.setdefault("detection_prompt", prompt)
    people_detected = bool(payload.get("people_detected"))
    people_count = int(payload.get("people_count") or 0) if payload.get("people_count") is not None else 0
    if not people_detected and people_count > 0:
        people_detected = True
    payload["people_detected"] = people_detected
    payload["people_count"] = people_count
    payload["mask_paths"] = _string_list(payload.get("mask_paths"))
    return payload


def _run_vip(
    *,
    input_video: Path,
    input_video_uri: str,
    masks_dir: Path,
    masks_prefix_uri: str,
    output_video: Path,
    output_video_uri: str,
    output_json: Path,
    output_json_uri: str,
    arkit_depth_prefix_uri: Optional[str],
    arkit_confidence_prefix_uri: Optional[str],
    depth_manifest_uri: Optional[str],
    confidence_manifest_uri: Optional[str],
) -> Dict[str, Any]:
    timeout_seconds = _timeout_env("PRIVACY_VIP_TIMEOUT_SECONDS", default=7200)
    ensure_dir(output_video.parent)
    preferred_depth_source = "arkit" if arkit_depth_prefix_uri else "depth_anything"
    runner_url = _vip_runner_url()
    if runner_url:
        payload = _run_http_json(
            url=runner_url,
            body={
                "input_video_uri": input_video_uri,
                "input_video_path": str(input_video),
                "masks_prefix_uri": masks_prefix_uri,
                "masks_dir_path": str(masks_dir),
                "output_video_uri": output_video_uri,
                "output_video_path": str(output_video),
                "output_json_uri": output_json_uri,
                "output_json_path": str(output_json),
                "arkit_depth_prefix_uri": arkit_depth_prefix_uri,
                "arkit_confidence_prefix_uri": arkit_confidence_prefix_uri,
                "depth_manifest_uri": depth_manifest_uri,
                "confidence_manifest_uri": confidence_manifest_uri,
                "preferred_depth_source": preferred_depth_source,
                "vip_model_path": str(os.getenv("VIP_MODEL_PATH") or ""),
                "depth_anything_model_path": str(os.getenv("DEPTH_ANYTHING_MODEL_PATH") or ""),
            },
            timeout_seconds=timeout_seconds,
        )
    else:
        template = _vip_command_template()
        if not template:
            return {"status": "failed", "reason": "vip_runner_not_configured"}
        payload = _run_json_command(
            command_template=template,
            substitutions={
                "INPUT_VIDEO": input_video,
                "INPUT_VIDEO_URI": input_video_uri,
                "MASKS_DIR": masks_dir,
                "MASKS_PREFIX_URI": masks_prefix_uri,
                "OUTPUT_VIDEO": output_video,
                "OUTPUT_VIDEO_URI": output_video_uri,
                "OUTPUT_JSON": output_json,
                "OUTPUT_JSON_URI": output_json_uri,
                "ARKIT_DEPTH_PREFIX_URI": arkit_depth_prefix_uri or "",
                "ARKIT_CONFIDENCE_PREFIX_URI": arkit_confidence_prefix_uri or "",
                "DEPTH_MANIFEST_URI": depth_manifest_uri or "",
                "CONFIDENCE_MANIFEST_URI": confidence_manifest_uri or "",
                "DEPTH_SOURCE": preferred_depth_source,
                "VIP_MODEL_PATH": str(os.getenv("VIP_MODEL_PATH") or ""),
                "DEPTH_ANYTHING_MODEL_PATH": str(os.getenv("DEPTH_ANYTHING_MODEL_PATH") or ""),
            },
            timeout_seconds=timeout_seconds,
        )
    if str(payload.get("status") or "").strip().lower() == "succeeded" and output_video.is_file():
        payload["output_video"] = str(output_video)
        payload.setdefault("depth_source", preferred_depth_source)
        return payload
    if str(payload.get("status") or "").strip().lower() == "succeeded":
        payload["output_video"] = str(output_video)
        payload.setdefault("output_video_uri", output_video_uri)
        payload.setdefault("depth_source", preferred_depth_source)
        return payload
    if output_video.is_file():
        payload["output_video"] = str(output_video)
        payload["status"] = "succeeded"
        payload.setdefault("depth_source", preferred_depth_source)
        return payload
    return {
        "status": "failed",
        "reason": str(payload.get("reason") or "vip_output_missing"),
        **payload,
    }


def _depth_conditioning_from_arkit(
    *,
    raw_video_uri: str,
    arkit_depth_prefix_uri: Optional[str],
    arkit_confidence_prefix_uri: Optional[str],
) -> Dict[str, Any]:
    return {
        "status": "available",
        "source": "arkit",
        "provider": "arkit",
        "model_name": None,
        "source_video_uri": raw_video_uri or None,
        "depth_prefix_uri": arkit_depth_prefix_uri,
        "confidence_prefix_uri": arkit_confidence_prefix_uri,
        "depth_manifest_uri": None,
        "confidence_manifest_uri": None,
        "depth_manifest_path": None,
        "confidence_manifest_path": None,
        "frame_count": None,
    }


def _run_depth_anything(
    *,
    input_video: Path,
    input_video_uri: str,
    depth_dir: Path,
    depth_prefix_uri: str,
    confidence_dir: Path,
    confidence_prefix_uri: str,
    depth_manifest_path: Path,
    depth_manifest_uri: str,
    confidence_manifest_path: Path,
    confidence_manifest_uri: str,
) -> Dict[str, Any]:
    timeout_seconds = _timeout_env("PRIVACY_DEPTH_ANYTHING_TIMEOUT_SECONDS", default=7200)
    ensure_dir(depth_dir)
    ensure_dir(confidence_dir)
    runner_url = _depth_anything_runner_url()
    if runner_url:
        payload = _run_http_json(
            url=runner_url,
            body={
                "input_video_uri": input_video_uri,
                "input_video_path": str(input_video),
                "depth_generation_only": True,
                "depth_output_prefix_uri": depth_prefix_uri,
                "depth_output_dir_path": str(depth_dir),
                "confidence_output_prefix_uri": confidence_prefix_uri,
                "confidence_output_dir_path": str(confidence_dir),
                "output_depth_manifest_uri": depth_manifest_uri,
                "output_depth_manifest_path": str(depth_manifest_path),
                "output_confidence_manifest_uri": confidence_manifest_uri,
                "output_confidence_manifest_path": str(confidence_manifest_path),
                "depth_anything_model_path": str(os.getenv("DEPTH_ANYTHING_MODEL_PATH") or ""),
            },
            timeout_seconds=timeout_seconds,
        )
    else:
        template = _depth_anything_command_template()
        if not template:
            return {"status": "failed", "reason": "depth_anything_runner_not_configured"}
        payload = _run_json_command(
            command_template=template,
            substitutions={
                "INPUT_VIDEO": input_video,
                "INPUT_VIDEO_URI": input_video_uri,
                "DEPTH_DIR": depth_dir,
                "DEPTH_PREFIX_URI": depth_prefix_uri,
                "CONFIDENCE_DIR": confidence_dir,
                "CONFIDENCE_PREFIX_URI": confidence_prefix_uri,
                "DEPTH_MANIFEST": depth_manifest_path,
                "DEPTH_MANIFEST_URI": depth_manifest_uri,
                "CONFIDENCE_MANIFEST": confidence_manifest_path,
                "CONFIDENCE_MANIFEST_URI": confidence_manifest_uri,
                "DEPTH_ANYTHING_MODEL_PATH": str(os.getenv("DEPTH_ANYTHING_MODEL_PATH") or ""),
            },
            timeout_seconds=timeout_seconds,
        )
    if str(payload.get("status") or "").strip().lower() != "succeeded":
        return {
            "status": "failed",
            "reason": str(payload.get("reason") or "depth_anything_failed"),
            **payload,
        }

    _ensure_remote_output_local(
        capture_root=input_video.parent.parent,
        output_uri=str(payload.get("depth_manifest_uri") or depth_manifest_uri),
        destination=depth_manifest_path,
    )
    _ensure_remote_output_local(
        capture_root=input_video.parent.parent,
        output_uri=str(payload.get("confidence_manifest_uri") or confidence_manifest_uri),
        destination=confidence_manifest_path,
    )
    result = {
        **payload,
        "status": "succeeded",
        "source": "depth_anything",
        "provider": str(payload.get("provider") or "depth_anything_3"),
        "model_name": str(payload.get("model_name") or os.getenv("DA3_MODEL_NAME") or "da3metric-large"),
        "source_video_uri": input_video_uri or None,
        "depth_prefix_uri": str(payload.get("depth_prefix_uri") or depth_prefix_uri),
        "confidence_prefix_uri": str(payload.get("confidence_prefix_uri") or confidence_prefix_uri),
        "depth_manifest_uri": str(payload.get("depth_manifest_uri") or depth_manifest_uri),
        "confidence_manifest_uri": str(payload.get("confidence_manifest_uri") or confidence_manifest_uri),
        "depth_manifest_path": str(depth_manifest_path),
        "confidence_manifest_path": str(confidence_manifest_path),
        "frame_count": int(payload.get("frame_count") or 0),
    }
    return result


def _run_deepprivacy2(
    *,
    input_video: Path,
    input_video_uri: str,
    output_video: Path,
    output_video_uri: str,
    output_json: Path,
    output_json_uri: str,
) -> Dict[str, Any]:
    timeout_seconds = _timeout_env("PRIVACY_DEEPPRIVACY2_TIMEOUT_SECONDS", default=7200)
    ensure_dir(output_video.parent)
    runner_url = _deepprivacy_runner_url()
    if runner_url:
        payload = _run_http_json(
            url=runner_url,
            body={
                "input_video_uri": input_video_uri,
                "input_video_path": str(input_video),
                "output_video_uri": output_video_uri,
                "output_video_path": str(output_video),
                "output_json_uri": output_json_uri,
                "output_json_path": str(output_json),
                "deepprivacy2_model_path": str(os.getenv("DEEPPRIVACY2_MODEL_PATH") or ""),
            },
            timeout_seconds=timeout_seconds,
        )
    else:
        template = _deepprivacy_command_template()
        if not template:
            return {"status": "failed", "reason": "deepprivacy2_runner_not_configured"}
        payload = _run_json_command(
            command_template=template,
            substitutions={
                "INPUT_VIDEO": input_video,
                "INPUT_VIDEO_URI": input_video_uri,
                "OUTPUT_VIDEO": output_video,
                "OUTPUT_VIDEO_URI": output_video_uri,
                "OUTPUT_JSON": output_json,
                "OUTPUT_JSON_URI": output_json_uri,
                "DEEPPRIVACY2_MODEL_PATH": str(os.getenv("DEEPPRIVACY2_MODEL_PATH") or ""),
            },
            timeout_seconds=timeout_seconds,
        )
    if str(payload.get("status") or "").strip().lower() == "succeeded" and output_video.is_file():
        payload["output_video"] = str(output_video)
        payload["face_anonymized_segments"] = _string_list(payload.get("face_anonymized_segments"))
        return payload
    if str(payload.get("status") or "").strip().lower() == "succeeded":
        payload["output_video"] = str(output_video)
        payload.setdefault("output_video_uri", output_video_uri)
        payload["face_anonymized_segments"] = _string_list(payload.get("face_anonymized_segments"))
        return payload
    if output_video.is_file():
        payload["output_video"] = str(output_video)
        payload["status"] = "succeeded"
        payload["face_anonymized_segments"] = _string_list(payload.get("face_anonymized_segments"))
        return payload
    return {
        "status": "failed",
        "reason": str(payload.get("reason") or "deepprivacy2_output_missing"),
        **payload,
    }


def _verification_report(
    *,
    initial_detection: Mapping[str, Any],
    vip_verification: Optional[Mapping[str, Any]],
    fallback_verification: Optional[Mapping[str, Any]],
    result_status: str,
    result_mode: str,
) -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "initial_detection": dict(initial_detection),
        "vip_verification": dict(vip_verification or {}),
        "fallback_verification": dict(fallback_verification or {}),
        "status": result_status,
        "mode": result_mode,
    }


def _ensure_remote_output_local(
    *,
    capture_root: Path,
    output_uri: str,
    destination: Path,
) -> None:
    if destination.is_file() or not output_uri:
        return
    storage_root = infer_storage_root_from_scene_path(capture_root)
    local_path = ensure_local_uri_path(
        output_uri,
        gcs_root=storage_root,
        scratch_dir=destination.parent,
    )
    if local_path != destination:
        ensure_dir(destination.parent)
        shutil.copyfile(local_path, destination)


def run_privacy_postprocess(
    *,
    bucket: str,
    scene_id: str,
    capture_id: str,
    capture_root: Path,
    pipeline_dir: Path,
    raw_video_path: Optional[Path],
    site_type: Optional[str] = None,
) -> Dict[str, Any]:
    # R010: the detection class-set is site-type-aware. Industrial sites expand
    # the SAM3 target classes beyond ``person`` (badges, screens/monitors, license
    # plates, signage/whiteboards); non-industrial captures keep person-only.
    redaction_classes, is_industrial_site = _redaction_classes_for_site(site_type)
    industrial_sensitive_required = (
        list(INDUSTRIAL_SENSITIVE_REDACTION_CLASSES) if is_industrial_site else []
    )
    privacy_root = capture_root / "privacy"
    masks_root = privacy_root / "masks"
    final_video_path = privacy_root / "final_walkthrough.mov"
    vip_video_path = privacy_root / "intermediate_vip_walkthrough.mov"
    deepprivacy_video_path = privacy_root / "intermediate_deepprivacy2_walkthrough.mov"
    manifest_path = pipeline_dir / "privacy_processing_manifest.json"
    verification_path = pipeline_dir / "privacy_verification_report.json"

    ensure_dir(privacy_root)
    ensure_dir(masks_root)

    enabled = production_forces_true("PRIVACY_PIPELINE_ENABLED", default=False)
    fail_closed = _env_flag("PRIVACY_FAIL_CLOSED", default=True)
    privacy_prefix = f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/privacy"
    raw_prefix = f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/raw"
    manifest_uri = f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/pipeline/privacy_processing_manifest.json"
    verification_uri = f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/pipeline/privacy_verification_report.json"
    final_video_uri = f"{privacy_prefix}/final_walkthrough.mov"
    vip_video_uri = f"{privacy_prefix}/intermediate_vip_walkthrough.mov"
    deepprivacy_video_uri = f"{privacy_prefix}/intermediate_deepprivacy2_walkthrough.mov"
    privacy_depth_root = pipeline_dir / "privacy_depth"
    depth_dir = privacy_depth_root / "depth"
    confidence_dir = privacy_depth_root / "confidence"
    depth_manifest_path = privacy_depth_root / "depth_manifest.json"
    confidence_manifest_path = privacy_depth_root / "confidence_manifest.json"
    depth_prefix_uri = f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/pipeline/privacy_depth/depth"
    confidence_prefix_uri = f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/pipeline/privacy_depth/confidence"
    depth_manifest_uri = f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/pipeline/privacy_depth/depth_manifest.json"
    confidence_manifest_uri = f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/pipeline/privacy_depth/confidence_manifest.json"
    raw_video_uri = f"{raw_prefix}/{raw_video_path.name}" if raw_video_path else ""
    raw_arkit_root = capture_root / "raw" / "arkit"
    arkit_depth_prefix_uri = (
        f"{raw_prefix}/arkit/depth"
        if (raw_arkit_root / "depth").is_dir()
        else None
    )
    arkit_confidence_prefix_uri = (
        f"{raw_prefix}/arkit/confidence"
        if (raw_arkit_root / "confidence").is_dir()
        else None
    )

    payload: Dict[str, Any] = {
        "schema_version": "v1",
        "scene_id": scene_id,
        "capture_id": capture_id,
        "generated_at": utc_now_iso(),
        "enabled": enabled,
        "raw_retained": True,
        "fail_closed": fail_closed,
        "site_type": site_type,
        "is_industrial_site": is_industrial_site,
        "redaction_classes": list(redaction_classes),
        "industrial_sensitive_classes": industrial_sensitive_required,
        "industrial_sensitive_classes_handled": False,
        "status": "not_run",
        "mode": "none",
        "fallback_used": False,
        "people_detected": 0,
        "people_removed": 0,
        "face_anonymized_segments": [],
        "depth_source": None,
        "depth_conditioning": None,
        "raw_video_path": str(raw_video_path) if raw_video_path else None,
        "privacy_processed_video_uri": None,
        "world_model_video_uri": None,
        "privacy_manifest_uri": manifest_uri,
        "privacy_verification_report_uri": verification_uri,
        "steps": [],
    }

    if raw_video_path is None or not raw_video_path.is_file():
        payload["status"] = "failed_closed"
        payload["reason"] = "raw_video_missing"
        write_json(manifest_path, payload)
        write_json(
            verification_path,
            _verification_report(
                initial_detection={},
                vip_verification=None,
                fallback_verification=None,
                result_status=payload["status"],
                result_mode=payload["mode"],
            ),
        )
        return payload

    if not enabled:
        payload["reason"] = "privacy_pipeline_disabled"
        write_json(manifest_path, payload)
        write_json(
            verification_path,
            _verification_report(
                initial_detection={},
                vip_verification=None,
                fallback_verification=None,
                result_status=payload["status"],
                result_mode=payload["mode"],
            ),
        )
        return payload

    if _local_full_frame_redaction_enabled():
        redaction_result = _run_local_full_frame_redaction(raw_video_path, final_video_path)
        payload["steps"].append({"name": "local_full_frame_redaction", "result": dict(redaction_result)})
        payload["mode"] = "full_frame_redaction"
        payload["fallback_used"] = True
        payload["people_detection_performed"] = False
        payload["people_detector"] = None
        payload["privacy_method"] = "local_full_frame_pixelation_blur"
        payload["production_review_required"] = True
        payload["local_repo_proof_only"] = True
        payload["limitations"] = [
            "Full-frame pixelation/blur removes inspectable visual identity detail but is not model-based person segmentation.",
            "This artifact proves privacy-safe final-walkthrough selection for local pipeline rehearsal only.",
            "Production customer delivery still requires configured privacy runners or human review acceptance.",
        ]
        payload["proof_boundary"] = {
            "privacy_safe_final_walkthrough_selected": False,
            "local_full_frame_redaction_executed": False,
            "model_based_person_removal_proven": False,
            "live_privacy_service_proven": False,
            "production_review_required": True,
            "raw_video_bypass_used": False,
            "public_claim_upgrade_allowed": False,
        }
        if str(redaction_result.get("status") or "").strip().lower() != "succeeded":
            payload["status"] = "failed_closed"
            payload["reason"] = str(redaction_result.get("reason") or "local_full_frame_redaction_failed")
            write_json(manifest_path, payload)
            report = _verification_report(
                initial_detection={
                    "status": "not_run",
                    "reason": "local_full_frame_redaction_does_not_run_person_detector",
                },
                vip_verification=None,
                fallback_verification=redaction_result,
                result_status=payload["status"],
                result_mode=payload["mode"],
            )
            report["local_full_frame_redaction"] = dict(redaction_result)
            report["proof_boundary"] = dict(payload["proof_boundary"])
            write_json(verification_path, report)
            return payload

        payload["status"] = "full_frame_redacted_local_proof"
        payload["privacy_processed_video_uri"] = final_video_uri
        payload["world_model_video_uri"] = final_video_uri
        payload["proof_boundary"] = {
            **dict(payload["proof_boundary"]),
            "privacy_safe_final_walkthrough_selected": True,
            "local_full_frame_redaction_executed": True,
        }
        write_json(manifest_path, payload)
        report = _verification_report(
            initial_detection={
                "status": "not_run",
                "reason": "local_full_frame_redaction_does_not_run_person_detector",
            },
            vip_verification={"status": "not_run", "reason": "local_full_frame_redaction_used"},
            fallback_verification=redaction_result,
            result_status=payload["status"],
            result_mode=payload["mode"],
        )
        report["local_full_frame_redaction"] = dict(redaction_result)
        report["proof_boundary"] = dict(payload["proof_boundary"])
        write_json(verification_path, report)
        return payload

    initial_detection = _run_sam3(
        input_video=raw_video_path,
        input_video_uri=raw_video_uri,
        output_json=pipeline_dir / "privacy_sam3_detection.json",
        output_json_uri=f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/pipeline/privacy_sam3_detection.json",
        masks_dir=masks_root / "sam3_initial",
        masks_prefix_uri=f"{privacy_prefix}/masks/sam3_initial",
        stage_name="initial_detection",
        detection_classes=redaction_classes,
    )
    payload["steps"].append({"name": "sam3_initial_detection", "result": dict(initial_detection)})
    if str(initial_detection.get("status") or "").strip().lower() != "succeeded":
        payload["status"] = "failed_closed"
        payload["reason"] = str(initial_detection.get("reason") or "sam3_initial_detection_failed")
        write_json(manifest_path, payload)
        write_json(
            verification_path,
            _verification_report(
                initial_detection=initial_detection,
                vip_verification=None,
                fallback_verification=None,
                result_status=payload["status"],
                result_mode=payload["mode"],
            ),
        )
        return payload

    payload["people_detected"] = int(initial_detection.get("people_count") or 0)
    # The initial detection ran against the site-type-aware class set, so for
    # industrial sites the industrial-sensitive classes were actually targeted.
    payload["industrial_sensitive_classes_handled"] = is_industrial_site
    if arkit_depth_prefix_uri:
        depth_conditioning = _depth_conditioning_from_arkit(
            raw_video_uri=raw_video_uri,
            arkit_depth_prefix_uri=arkit_depth_prefix_uri,
            arkit_confidence_prefix_uri=arkit_confidence_prefix_uri,
        )
        payload["steps"].append({"name": "depth_conditioning", "result": dict(depth_conditioning)})
    else:
        depth_conditioning = _run_depth_anything(
            input_video=raw_video_path,
            input_video_uri=raw_video_uri,
            depth_dir=depth_dir,
            depth_prefix_uri=depth_prefix_uri,
            confidence_dir=confidence_dir,
            confidence_prefix_uri=confidence_prefix_uri,
            depth_manifest_path=depth_manifest_path,
            depth_manifest_uri=depth_manifest_uri,
            confidence_manifest_path=confidence_manifest_path,
            confidence_manifest_uri=confidence_manifest_uri,
        )
        payload["steps"].append({"name": "depth_conditioning", "result": dict(depth_conditioning)})
        if str(depth_conditioning.get("status") or "").strip().lower() != "succeeded":
            payload["status"] = "failed_closed"
            payload["reason"] = str(depth_conditioning.get("reason") or "depth_conditioning_failed")
            payload["depth_conditioning"] = depth_conditioning
            write_json(manifest_path, payload)
            write_json(
                verification_path,
                _verification_report(
                    initial_detection=initial_detection,
                    vip_verification=None,
                    fallback_verification=None,
                    result_status=payload["status"],
                    result_mode=payload["mode"],
                ),
            )
            return payload

    payload["depth_source"] = str(depth_conditioning.get("source") or "").strip() or None
    payload["depth_conditioning"] = depth_conditioning
    if not bool(initial_detection.get("people_detected")):
        passthrough = _copy_or_remux_video(raw_video_path, final_video_path)
        payload["steps"].append({"name": "passthrough_copy", "result": dict(passthrough)})
        payload["status"] = "no_people_detected"
        payload["mode"] = "none"
        payload["privacy_processed_video_uri"] = final_video_uri
        payload["world_model_video_uri"] = final_video_uri
        write_json(manifest_path, payload)
        write_json(
            verification_path,
            _verification_report(
                initial_detection=initial_detection,
                vip_verification={"status": "not_needed"},
                fallback_verification=None,
                result_status=payload["status"],
                result_mode=payload["mode"],
            ),
        )
        return payload

    vip_result = _run_vip(
        input_video=raw_video_path,
        input_video_uri=raw_video_uri,
        masks_dir=masks_root / "sam3_initial",
        masks_prefix_uri=f"{privacy_prefix}/masks/sam3_initial",
        output_video=vip_video_path,
        output_video_uri=vip_video_uri,
        output_json=pipeline_dir / "privacy_vip_result.json",
        output_json_uri=f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/pipeline/privacy_vip_result.json",
        arkit_depth_prefix_uri=arkit_depth_prefix_uri,
        arkit_confidence_prefix_uri=arkit_confidence_prefix_uri,
        depth_manifest_uri=(
            str(depth_conditioning.get("depth_manifest_uri") or "").strip() or None
            if str(depth_conditioning.get("source") or "").strip() == "depth_anything"
            else None
        ),
        confidence_manifest_uri=(
            str(depth_conditioning.get("confidence_manifest_uri") or "").strip() or None
            if str(depth_conditioning.get("source") or "").strip() == "depth_anything"
            else None
        ),
    )
    payload["steps"].append({"name": "vip_inpainting", "result": dict(vip_result)})
    if str(vip_result.get("status") or "").strip().lower() != "succeeded":
        payload["status"] = "failed_closed"
        payload["mode"] = "removal"
        payload["reason"] = str(vip_result.get("reason") or "vip_failed")
        write_json(manifest_path, payload)
        write_json(
            verification_path,
            _verification_report(
                initial_detection=initial_detection,
                vip_verification=vip_result,
                fallback_verification=None,
                result_status=payload["status"],
                result_mode=payload["mode"],
            ),
        )
        return payload
    _ensure_remote_output_local(
        capture_root=capture_root,
        output_uri=str(vip_result.get("output_video_uri") or vip_video_uri),
        destination=vip_video_path,
    )

    vip_verification = _run_sam3(
        input_video=vip_video_path,
        input_video_uri=vip_video_uri,
        output_json=pipeline_dir / "privacy_vip_verification.json",
        output_json_uri=f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/pipeline/privacy_vip_verification.json",
        masks_dir=masks_root / "sam3_vip_verify",
        masks_prefix_uri=f"{privacy_prefix}/masks/sam3_vip_verify",
        stage_name="vip_verification",
        detection_classes=redaction_classes,
    )
    payload["steps"].append({"name": "sam3_vip_verification", "result": dict(vip_verification)})
    if str(vip_verification.get("status") or "").strip().lower() != "succeeded":
        payload["status"] = "failed_closed"
        payload["mode"] = "removal"
        payload["reason"] = str(vip_verification.get("reason") or "vip_verification_failed")
        write_json(manifest_path, payload)
        write_json(
            verification_path,
            _verification_report(
                initial_detection=initial_detection,
                vip_verification=vip_verification,
                fallback_verification=None,
                result_status=payload["status"],
                result_mode=payload["mode"],
            ),
        )
        return payload

    if not bool(vip_verification.get("people_detected")):
        shutil.copyfile(vip_video_path, final_video_path)
        payload["status"] = "person_removed"
        payload["mode"] = "removal"
        payload["people_removed"] = payload["people_detected"]
        payload["depth_source"] = (
            str(vip_result.get("depth_source") or "").strip()
            or str((payload.get("depth_conditioning") or {}).get("source") or "").strip()
            or None
        )
        payload["privacy_processed_video_uri"] = final_video_uri
        payload["world_model_video_uri"] = final_video_uri
        write_json(manifest_path, payload)
        write_json(
            verification_path,
            _verification_report(
                initial_detection=initial_detection,
                vip_verification=vip_verification,
                fallback_verification=None,
                result_status=payload["status"],
                result_mode=payload["mode"],
            ),
        )
        return payload

    deepprivacy_result = _run_deepprivacy2(
        input_video=vip_video_path,
        input_video_uri=vip_video_uri,
        output_video=deepprivacy_video_path,
        output_video_uri=deepprivacy_video_uri,
        output_json=pipeline_dir / "privacy_deepprivacy2_result.json",
        output_json_uri=f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/pipeline/privacy_deepprivacy2_result.json",
    )
    payload["steps"].append({"name": "deepprivacy2_fallback", "result": dict(deepprivacy_result)})
    if str(deepprivacy_result.get("status") or "").strip().lower() != "succeeded":
        payload["status"] = "failed_closed"
        payload["mode"] = "anonymized_fallback"
        payload["reason"] = str(deepprivacy_result.get("reason") or "deepprivacy2_failed")
        write_json(manifest_path, payload)
        write_json(
            verification_path,
            _verification_report(
                initial_detection=initial_detection,
                vip_verification=vip_verification,
                fallback_verification=deepprivacy_result,
                result_status=payload["status"],
                result_mode=payload["mode"],
            ),
        )
        return payload
    _ensure_remote_output_local(
        capture_root=capture_root,
        output_uri=str(deepprivacy_result.get("output_video_uri") or deepprivacy_video_uri),
        destination=deepprivacy_video_path,
    )

    fallback_verification = _run_sam3(
        input_video=deepprivacy_video_path,
        input_video_uri=deepprivacy_video_uri,
        output_json=pipeline_dir / "privacy_deepprivacy2_verification.json",
        output_json_uri=f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/pipeline/privacy_deepprivacy2_verification.json",
        masks_dir=masks_root / "sam3_deepprivacy_verify",
        masks_prefix_uri=f"{privacy_prefix}/masks/sam3_deepprivacy_verify",
        stage_name="deepprivacy2_verification",
        detection_classes=redaction_classes,
    )
    payload["steps"].append({"name": "sam3_deepprivacy2_verification", "result": dict(fallback_verification)})
    if str(fallback_verification.get("status") or "").strip().lower() != "succeeded":
        payload["status"] = "failed_closed"
        payload["mode"] = "anonymized_fallback"
        payload["reason"] = str(fallback_verification.get("reason") or "deepprivacy2_verification_failed")
        write_json(manifest_path, payload)
        write_json(
            verification_path,
            _verification_report(
                initial_detection=initial_detection,
                vip_verification=vip_verification,
                fallback_verification=fallback_verification,
                result_status=payload["status"],
                result_mode=payload["mode"],
            ),
        )
        return payload

    if not _string_list(deepprivacy_result.get("face_anonymized_segments")):
        payload["status"] = "failed_closed"
        payload["mode"] = "anonymized_fallback"
        payload["reason"] = "deepprivacy2_face_segments_missing"
        write_json(manifest_path, payload)
        write_json(
            verification_path,
            _verification_report(
                initial_detection=initial_detection,
                vip_verification=vip_verification,
                fallback_verification=fallback_verification,
                result_status=payload["status"],
                result_mode=payload["mode"],
            ),
        )
        return payload

    shutil.copyfile(deepprivacy_video_path, final_video_path)
    payload["status"] = "face_anonymized_fallback"
    payload["mode"] = "anonymized_fallback"
    payload["fallback_used"] = True
    payload["people_removed"] = max(
        0,
        payload["people_detected"] - int(fallback_verification.get("people_count") or 0),
    )
    payload["face_anonymized_segments"] = _string_list(deepprivacy_result.get("face_anonymized_segments"))
    payload["depth_source"] = (
        str(vip_result.get("depth_source") or "").strip()
        or str((payload.get("depth_conditioning") or {}).get("source") or "").strip()
        or None
    )
    payload["privacy_processed_video_uri"] = final_video_uri
    payload["world_model_video_uri"] = final_video_uri
    write_json(manifest_path, payload)
    write_json(
        verification_path,
        _verification_report(
            initial_detection=initial_detection,
            vip_verification=vip_verification,
            fallback_verification=fallback_verification,
            result_status=payload["status"],
            result_mode=payload["mode"],
        ),
    )
    return payload
