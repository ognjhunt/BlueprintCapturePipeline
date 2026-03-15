"""Privacy-preserving post-processing for capture walkthrough video."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .common import ensure_dir, parse_bool, utc_now_iso, write_json


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


def _sam3_command_template() -> str:
    return str(os.getenv("PRIVACY_SAM3_COMMAND") or os.getenv("SAM3_COMMAND") or "").strip()


def _vip_command_template() -> str:
    return str(os.getenv("VIP_COMMAND") or "").strip()


def _deepprivacy_command_template() -> str:
    return str(os.getenv("DEEPPRIVACY2_COMMAND") or "").strip()


def _run_sam3(
    *,
    input_video: Path,
    output_json: Path,
    masks_dir: Path,
    stage_name: str,
) -> Dict[str, Any]:
    template = _sam3_command_template()
    if not template:
        return {"status": "failed", "reason": "sam3_command_not_configured"}
    ensure_dir(masks_dir)
    payload = _run_json_command(
        command_template=template,
        substitutions={
            "INPUT_VIDEO": input_video,
            "OUTPUT_JSON": output_json,
            "MASKS_DIR": masks_dir,
            "PROMPT": "person",
            "STAGE_NAME": stage_name,
            "SAM3_WEIGHTS_PATH": str(os.getenv("SAM3_WEIGHTS_PATH") or ""),
        },
        timeout_seconds=_timeout_env("PRIVACY_SAM3_TIMEOUT_SECONDS", default=3600),
    )
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
    masks_dir: Path,
    output_video: Path,
    output_json: Path,
) -> Dict[str, Any]:
    template = _vip_command_template()
    if not template:
        return {"status": "failed", "reason": "vip_command_not_configured"}
    ensure_dir(output_video.parent)
    payload = _run_json_command(
        command_template=template,
        substitutions={
            "INPUT_VIDEO": input_video,
            "MASKS_DIR": masks_dir,
            "OUTPUT_VIDEO": output_video,
            "OUTPUT_JSON": output_json,
            "VIP_MODEL_PATH": str(os.getenv("VIP_MODEL_PATH") or ""),
        },
        timeout_seconds=_timeout_env("PRIVACY_VIP_TIMEOUT_SECONDS", default=7200),
    )
    if str(payload.get("status") or "").strip().lower() == "succeeded" and output_video.is_file():
        payload["output_video"] = str(output_video)
        return payload
    if output_video.is_file():
        payload["output_video"] = str(output_video)
        payload["status"] = "succeeded"
        return payload
    return {
        "status": "failed",
        "reason": str(payload.get("reason") or "vip_output_missing"),
        **payload,
    }


def _run_deepprivacy2(
    *,
    input_video: Path,
    output_video: Path,
    output_json: Path,
) -> Dict[str, Any]:
    template = _deepprivacy_command_template()
    if not template:
        return {"status": "failed", "reason": "deepprivacy2_command_not_configured"}
    ensure_dir(output_video.parent)
    payload = _run_json_command(
        command_template=template,
        substitutions={
            "INPUT_VIDEO": input_video,
            "OUTPUT_VIDEO": output_video,
            "OUTPUT_JSON": output_json,
            "DEEPPRIVACY2_MODEL_PATH": str(os.getenv("DEEPPRIVACY2_MODEL_PATH") or ""),
        },
        timeout_seconds=_timeout_env("PRIVACY_DEEPPRIVACY2_TIMEOUT_SECONDS", default=7200),
    )
    if str(payload.get("status") or "").strip().lower() == "succeeded" and output_video.is_file():
        payload["output_video"] = str(output_video)
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


def run_privacy_postprocess(
    *,
    bucket: str,
    scene_id: str,
    capture_id: str,
    capture_root: Path,
    pipeline_dir: Path,
    raw_video_path: Optional[Path],
) -> Dict[str, Any]:
    privacy_root = capture_root / "privacy"
    masks_root = privacy_root / "masks"
    final_video_path = privacy_root / "final_walkthrough.mov"
    vip_video_path = privacy_root / "intermediate_vip_walkthrough.mov"
    deepprivacy_video_path = privacy_root / "intermediate_deepprivacy2_walkthrough.mov"
    manifest_path = pipeline_dir / "privacy_processing_manifest.json"
    verification_path = pipeline_dir / "privacy_verification_report.json"

    ensure_dir(privacy_root)
    ensure_dir(masks_root)

    enabled = _env_flag("PRIVACY_PIPELINE_ENABLED", default=False)
    fail_closed = _env_flag("PRIVACY_FAIL_CLOSED", default=True)
    privacy_prefix = f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/privacy"
    manifest_uri = f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/pipeline/privacy_processing_manifest.json"
    verification_uri = f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/pipeline/privacy_verification_report.json"
    final_video_uri = f"{privacy_prefix}/final_walkthrough.mov"

    payload: Dict[str, Any] = {
        "schema_version": "v1",
        "scene_id": scene_id,
        "capture_id": capture_id,
        "generated_at": utc_now_iso(),
        "enabled": enabled,
        "raw_retained": True,
        "fail_closed": fail_closed,
        "status": "not_run",
        "mode": "none",
        "fallback_used": False,
        "people_detected": 0,
        "people_removed": 0,
        "face_anonymized_segments": [],
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

    initial_detection = _run_sam3(
        input_video=raw_video_path,
        output_json=pipeline_dir / "privacy_sam3_detection.json",
        masks_dir=masks_root / "sam3_initial",
        stage_name="initial_detection",
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
        masks_dir=masks_root / "sam3_initial",
        output_video=vip_video_path,
        output_json=pipeline_dir / "privacy_vip_result.json",
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

    vip_verification = _run_sam3(
        input_video=vip_video_path,
        output_json=pipeline_dir / "privacy_vip_verification.json",
        masks_dir=masks_root / "sam3_vip_verify",
        stage_name="vip_verification",
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
        output_video=deepprivacy_video_path,
        output_json=pipeline_dir / "privacy_deepprivacy2_result.json",
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

    fallback_verification = _run_sam3(
        input_video=deepprivacy_video_path,
        output_json=pipeline_dir / "privacy_deepprivacy2_verification.json",
        masks_dir=masks_root / "sam3_deepprivacy_verify",
        stage_name="deepprivacy2_verification",
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
    payload["people_removed"] = max(0, payload["people_detected"] - int(fallback_verification.get("people_count") or 0))
    payload["face_anonymized_segments"] = _string_list(deepprivacy_result.get("face_anonymized_segments"))
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
