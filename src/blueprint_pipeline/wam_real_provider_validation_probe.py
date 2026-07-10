"""Run a fail-closed WAM perception harness real-provider proof probe."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageStat

from .common import ensure_dir, utc_now_iso, write_json
from .wam_derived_observation_harness import (
    BACKEND_RESULT_SCHEMA_VERSION,
    run_wam_derived_observation_harness_step,
)


PROOF_MANIFEST_SCHEMA_VERSION = "wam_real_provider_validation_proof_manifest.v1"
DEFAULT_JOB_PREFIX = "wam_real_provider_validation_probe"
SAM3_WEIGHTS_ENV = "SAM3_WEIGHTS_PATH"
ALT_SAM3_WEIGHTS_ENV = "BLUEPRINT_SAM3_WEIGHTS_PATH"
SAM3_AUTODOWNLOAD_ENV = "BLUEPRINT_WAM_ALLOW_SAM3_ULTRALYTICS_AUTODOWNLOAD"
SAM3_MODEL_ENV = "BLUEPRINT_WAM_SAM3_MODEL"
DEFAULT_SAM3_MODEL_REF = "sam3.pt"
SAM3_PROVIDER_KIND_ENV = "BLUEPRINT_WAM_SAM3_PROVIDER_KIND"
SAM3_TRANSFORMERS_ENV = "BLUEPRINT_WAM_ALLOW_SAM3_TRANSFORMERS_PROVIDER"
SAM3_HF_MODEL_ENV = "BLUEPRINT_WAM_SAM3_HF_MODEL_ID"
SAM3_HF_REVISION_ENV = "BLUEPRINT_WAM_SAM3_HF_MODEL_REVISION"
DEFAULT_SAM3_HF_MODEL_ID = "facebook/sam3"
DEFAULT_SAM3_HF_MODEL_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
DEPTH_COMMAND_ENV = "BLUEPRINT_WAM_DEPTH_PROVIDER_COMMAND"
AUTO_DEPTH_ENV = "BLUEPRINT_ALLOW_WAM_AUTO_DEPTH_PROVIDER"
DEPTH_PROVIDER_KIND_ENV = "BLUEPRINT_WAM_DEPTH_PROVIDER_KIND"
DEPTH_MODEL_ENV = "BLUEPRINT_WAM_DEPTH_MODEL_ID"
DEPTH_MODEL_REVISION_ENV = "BLUEPRINT_WAM_DEPTH_MODEL_REVISION"
DEFAULT_DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEFAULT_DEPTH_MODEL_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
AUTO_DA3_ENV = "BLUEPRINT_ALLOW_WAM_AUTO_DA3_PROVIDER"
DA3_MODEL_ENV = "BLUEPRINT_WAM_DA3_MODEL_ID"
DA3_MODEL_REVISION_ENV = "BLUEPRINT_WAM_DA3_MODEL_REVISION"
DA3_PROCESS_RES_ENV = "BLUEPRINT_WAM_DA3_PROCESS_RES"
DEFAULT_DA3_MODEL_ID = "depth-anything/DA3-BASE"
DEFAULT_DA3_MODEL_REVISION = "f4a6c9b3c95e41c82048423d3493a81ec3fa810e"
SAM3_CONFIDENCE_ENV = "BLUEPRINT_WAM_SAM3_CONFIDENCE"
SAM3_DEVICE_ENV = "BLUEPRINT_WAM_SAM3_DEVICE"
DA3_DEVICE_ENV = "BLUEPRINT_WAM_DA3_DEVICE"
TORCH_DEVICE_ENV = "BLUEPRINT_WAM_TORCH_DEVICE"
POSE_COMMAND_ENV = "BLUEPRINT_WAM_POSE_PROVIDER_COMMAND"
POSE_MODEL_ENV = "BLUEPRINT_WAM_POSE_MODEL_PATH"
AUTO_POSE_ENV = "BLUEPRINT_ALLOW_WAM_AUTO_POSE_PROVIDER"
REQUIRE_POSE_ENV = "BLUEPRINT_WAM_REQUIRE_POSE_PROVIDER"
DEFAULT_POSE_MODEL_PATH = "yolo11n-pose.pt"
HF_TOKEN_ENVS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")
HF_TOKEN_FILE_ENVS = ("HF_TOKEN_FILE", "HUGGINGFACE_HUB_TOKEN_FILE", "HUGGING_FACE_HUB_TOKEN_FILE")
REAL_VALIDATION_FLAG_KEYS = (
    "capture_backed",
    "capture_truth",
    "real_labeled_validation",
    "accepted_real_world_anchor",
    "physical_robot_evidence",
    "operator_attested",
)
VALIDATION_LABEL_KEYS = (
    "expected_object_id",
    "object_id",
    "expected_target_visible",
    "target_visible",
    "expected_contact",
    "contact_expected",
    "actual_success",
    "real_success",
    "capture_success",
    "plain_video_success",
    "generated_video_success",
)
VALIDATION_SOURCE_KEYS = (
    "source_capture_path",
    "source_capture_bundle_path",
    "source_artifact_path",
    "source_manifest_path",
    "source_video_path",
    "source_frame_path",
    "source_label_path",
    "evidence_path",
    "operator_attestation_path",
    "capture_id",
    "capture_bundle_id",
)
VALIDATION_TARGET_KEYS = (
    "target_prompt",
    "target_prompts",
    "target_object_prompt",
    "target_object_prompts",
)
VALIDATION_FRAME_ID_KEYS = (
    "frame_id",
    "source_frame_id",
    "generated_frame_id",
    "validation_frame_id",
)
VALIDATION_FRAME_PATH_KEYS = (
    "source_generated_frame_path",
    "generated_frame_path",
    "source_frame_path",
    "frame_path",
    "validation_frame_path",
)
VALIDATION_PROVENANCE_KEYS = (
    "reviewer_id",
    "reviewer",
    "reviewed_by",
    "review_decision",
    "review_status",
    "source_label_path",
    "operator_attestation_path",
    "label_provenance",
)
PROVIDER_ONLY_SOURCE_MARKERS = (
    "wam_perception_backend",
    "provider_result",
    "sam3_semantic_predictor",
    "depth_provider",
    "pose_provider",
    "generated_pixels",
    "model_output",
)
PLACEHOLDER_VALUES = {
    "",
    "none",
    "null",
    "n/a",
    "na",
    "todo",
    "tbd",
    "unknown",
    "<reviewer>",
    "<reviewer-id>",
    "<source>",
    "<path>",
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _subprocess_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _load_json_value(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _validation_rows_from_value(value: Any) -> list[dict[str, Any]]:
    rows: list[Any] = []
    if isinstance(value, list):
        rows = value
    elif isinstance(value, Mapping):
        row_container_seen = False
        for key in ("rows", "labels", "validation_rows", "anchors", "steps"):
            row_container_seen = row_container_seen or key in value
            candidate = value.get(key)
            if isinstance(candidate, list):
                rows = candidate
                break
        if not row_container_seen and value:
            rows = [value]
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _row_has_truthy_key(row: Mapping[str, Any], keys: Sequence[str]) -> bool:
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool) and value:
            return True
        if _truthy(value):
            return True
    return False


def _row_has_label(row: Mapping[str, Any]) -> bool:
    return any(_string(row.get(key)) for key in VALIDATION_LABEL_KEYS)


def _row_has_source_ref(row: Mapping[str, Any]) -> bool:
    return bool(_valid_ref_strings(row, VALIDATION_SOURCE_KEYS))


def _is_placeholder(value: Any) -> bool:
    text = _string(value)
    return text.lower() in PLACEHOLDER_VALUES or (text.startswith("<") and text.endswith(">"))


def _strings_from_value(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        values: list[str] = []
        for item in value.values():
            values.extend(_strings_from_value(item))
        return values
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = []
        for item in value:
            values.extend(_strings_from_value(item))
        return values
    text = _string(value)
    return [text] if text else []


def _valid_ref_strings(row: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    refs: list[str] = []
    for key in keys:
        for value in _strings_from_value(row.get(key)):
            if not _is_placeholder(value):
                refs.append(value)
    return refs


def _target_strings(row: Mapping[str, Any]) -> list[str]:
    return _valid_ref_strings(row, VALIDATION_TARGET_KEYS)


def _row_has_empty_target_field(row: Mapping[str, Any]) -> bool:
    for key in VALIDATION_TARGET_KEYS:
        if key in row and not _target_strings({key: row.get(key)}):
            return True
    return False


def _row_has_label_provenance(row: Mapping[str, Any]) -> bool:
    return bool(_valid_ref_strings(row, VALIDATION_PROVENANCE_KEYS))


def _row_has_provider_only_source(row: Mapping[str, Any]) -> bool:
    if _truthy(row.get("provider_output_only")) or _truthy(row.get("model_output_only")):
        return True
    refs = _valid_ref_strings(row, VALIDATION_SOURCE_KEYS)
    if not refs:
        return False
    provider_refs = [
        ref
        for ref in refs
        if any(marker in ref.lower() for marker in PROVIDER_ONLY_SOURCE_MARKERS)
    ]
    return bool(provider_refs) and len(provider_refs) == len(refs)


def _frame_ref_strings(row: Mapping[str, Any]) -> list[str]:
    return [
        *_valid_ref_strings(row, VALIDATION_FRAME_ID_KEYS),
        *_valid_ref_strings(row, VALIDATION_FRAME_PATH_KEYS),
    ]


def _expected_frame_tokens(expected_frame_path: Path | None) -> set[str]:
    if expected_frame_path is None:
        return set()
    path = expected_frame_path.expanduser()
    tokens = {str(path), path.name, path.stem}
    try:
        resolved = path.resolve()
        tokens.update({str(resolved), resolved.name, resolved.stem})
    except OSError:
        pass
    return {token for token in tokens if token}


def _row_matches_expected_frame(
    row: Mapping[str, Any],
    expected_frame_path: Path | None,
) -> bool:
    refs = _frame_ref_strings(row)
    if not refs or expected_frame_path is None:
        return True
    expected = _expected_frame_tokens(expected_frame_path)
    expected_path = str(expected_frame_path)
    for ref in refs:
        ref_path = Path(ref).expanduser()
        ref_tokens = {ref, ref_path.name, ref_path.stem}
        try:
            resolved = ref_path.resolve()
            ref_tokens.update({str(resolved), resolved.name, resolved.stem})
        except OSError:
            pass
        if expected.intersection(token for token in ref_tokens if token):
            return True
        if ref.endswith(expected_path) or expected_path.endswith(ref):
            return True
    return False


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _env_present(name: str) -> bool:
    return bool(os.environ.get(name))


def _redacted_env_presence(names: Sequence[str]) -> dict[str, bool]:
    return {name: _env_present(name) for name in names}


def _redacted_file_env_presence(names: Sequence[str]) -> dict[str, bool]:
    presence: dict[str, bool] = {}
    for name in names:
        raw = _string(os.environ.get(name))
        presence[name] = bool(raw and Path(raw).expanduser().is_file())
    return presence


def _hf_token_value() -> str | None:
    for name in HF_TOKEN_ENVS:
        token = _string(os.environ.get(name))
        if token:
            return token
    for name in HF_TOKEN_FILE_ENVS:
        raw = _string(os.environ.get(name))
        if not raw:
            continue
        path = Path(raw).expanduser()
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if token:
            return token
    return None


def _truthy(value: Any) -> bool:
    return _string(value).lower() in {"1", "true", "yes", "y", "on", "allow", "enabled"}


def _float_env(name: str, default: float) -> float:
    try:
        return float(_string(os.environ.get(name)) or default)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(_string(os.environ.get(name)) or default)
    except ValueError:
        return default


def _select_torch_device(env_name: str) -> str:
    requested = _string(os.environ.get(env_name)) or _string(os.environ.get(TORCH_DEVICE_ENV))
    if requested:
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        return "cpu"
    return "cpu"


def _resolve_sam3_weights() -> Path | None:
    for name in (SAM3_WEIGHTS_ENV, ALT_SAM3_WEIGHTS_ENV):
        raw = _string(os.environ.get(name))
        if raw:
            return Path(raw).expanduser()
    local = Path("sam3.pt")
    if local.is_file():
        return local.resolve()
    return None


def _resolve_sam3_model_ref() -> tuple[str | None, Path | None, str]:
    weights = _resolve_sam3_weights()
    if weights is not None:
        return str(weights), weights, "weights_path"
    if _truthy(os.environ.get(SAM3_AUTODOWNLOAD_ENV)):
        model_ref = _string(os.environ.get(SAM3_MODEL_ENV)) or DEFAULT_SAM3_MODEL_REF
        return model_ref, None, "ultralytics_autodownload"
    return None, None, "missing"


def _target_prompts(request: Mapping[str, Any]) -> list[str]:
    grounding = _mapping(request.get("eval_ready_task_grounding"))
    task = _mapping(grounding.get("task"))
    selected = _mapping(grounding.get("selected_task_target"))
    prompts = [
        *_sequence(task.get("target_prompts_for_object_index_backends")),
        selected.get("label"),
        selected.get("source_prompt"),
        selected.get("object_id"),
        _mapping(request.get("source_policy_action")).get("task_prompt"),
    ]
    cleaned = []
    for prompt in prompts:
        text = _string(prompt)
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned or ["target object"]


def _bbox_from_result(result: Any) -> tuple[list[float] | None, float | None]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return None, None
    try:
        xyxy = boxes.xyxy
        conf = getattr(boxes, "conf", None)
        if hasattr(xyxy, "detach"):
            xyxy = xyxy.detach().cpu().numpy()
        if hasattr(conf, "detach"):
            conf = conf.detach().cpu().numpy()
        if len(xyxy) < 1:
            return None, None
        bbox = [round(float(item), 3) for item in xyxy[0].tolist()]
        confidence = round(float(conf[0]), 4) if conf is not None and len(conf) else None
        return bbox, confidence
    except Exception:
        return None, None


def _numeric_rows(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return _sequence(value)


def _numeric_scalar(value: Any, default: float) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _run_transformers_sam3_provider(
    request: Mapping[str, Any],
    job_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model_id = _string(os.environ.get(SAM3_HF_MODEL_ENV)) or DEFAULT_SAM3_HF_MODEL_ID
    model_revision = (
        _string(os.environ.get(SAM3_HF_REVISION_ENV)) or DEFAULT_SAM3_HF_MODEL_REVISION
    )
    confidence_threshold = max(0.0, min(1.0, _float_env(SAM3_CONFIDENCE_ENV, 0.05)))
    device_name = _select_torch_device(SAM3_DEVICE_ENV)
    status = {
        "provider": "sam3",
        "kind": "transformers_sam3",
        "ran": False,
        "device": device_name,
        "device_env": SAM3_DEVICE_ENV,
        "confidence_threshold": confidence_threshold,
        "confidence_env": SAM3_CONFIDENCE_ENV,
        "provider_kind_env": SAM3_PROVIDER_KIND_ENV,
        "provider_kind": "transformers",
        "transformers_env": SAM3_TRANSFORMERS_ENV,
        "transformers_provider_enabled": _truthy(os.environ.get(SAM3_TRANSFORMERS_ENV)),
        "model_id_env": SAM3_HF_MODEL_ENV,
        "model_id": model_id,
        "model_revision": model_revision,
        "model_remote_code_trusted": False,
        "hf_token_present": bool(_hf_token_value()),
        "module_transformers_available": _module_available("transformers"),
        "runtime_package": None,
        "runtime_class": None,
        "model_family": "sam3",
        "blockers": [],
    }
    if not status["transformers_provider_enabled"]:
        status["blockers"].append("sam3_transformers_provider_not_enabled")
    if not status["module_transformers_available"]:
        status["blockers"].append("sam3_transformers_package_missing")
    if (
        model_id != DEFAULT_SAM3_HF_MODEL_ID
        or model_revision != DEFAULT_SAM3_HF_MODEL_REVISION
    ):
        status["blockers"].append("sam3_model_revision_not_approved")
    source_frame = Path(_string(request.get("source_generated_frame_path"))).expanduser()
    if not source_frame.is_file():
        status["blockers"].append("source_generated_frame_missing_for_sam3")
    if status["blockers"]:
        return [], status

    try:
        import torch
        from transformers import Sam3Model, Sam3Processor

        status["runtime_package"] = "transformers"
        status["runtime_class"] = "Sam3Model/Sam3Processor"
        image = Image.open(source_frame).convert("RGB")
        token = _hf_token_value()
        load_kwargs = {
            "revision": model_revision,
            "trust_remote_code": False,
        }
        if token:
            load_kwargs["token"] = token
        model = Sam3Model.from_pretrained(model_id, **load_kwargs).to(device_name)
        processor = Sam3Processor.from_pretrained(model_id, **load_kwargs)
        objects: list[dict[str, Any]] = []
        prompts = _target_prompts(request)[:3]
        for prompt_index, prompt in enumerate(prompts):
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(device_name)
            with torch.no_grad():
                outputs = model(**inputs)
            original_sizes = inputs.get("original_sizes") if isinstance(inputs, Mapping) else None
            target_sizes = original_sizes.tolist() if hasattr(original_sizes, "tolist") else [[image.height, image.width]]
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=confidence_threshold,
                mask_threshold=confidence_threshold,
                target_sizes=target_sizes,
            )
            result = _mapping(results[0] if results else {})
            boxes = _numeric_rows(result.get("boxes"))
            scores = _numeric_rows(result.get("scores"))
            masks = _numeric_rows(result.get("masks"))
            for box_index, box in enumerate(boxes):
                if not isinstance(box, Sequence) or len(box) < 4:
                    continue
                confidence = _numeric_scalar(
                    scores[box_index] if box_index < len(scores) else None,
                    0.5,
                )
                mask_path = None
                if box_index < len(masks):
                    try:
                        import numpy as np

                        mask_image = Image.new("L", image.size)
                        mask_rows = masks[box_index]
                        if hasattr(mask_rows, "detach"):
                            mask_rows = mask_rows.detach().cpu().numpy()
                        mask_rows = np.asarray(mask_rows)
                        mask_image = Image.fromarray(
                            (255 * (mask_rows > 0)).astype("uint8")
                        )
                        mask_path = job_dir / f"sam3_transformers_mask_{prompt_index:02d}_{box_index:04d}.png"
                        mask_image.save(mask_path)
                    except Exception:
                        mask_path = None
                object_id = f"sam3_transformers_target_{prompt_index:02d}_{box_index:04d}"
                objects.append(
                    {
                        "object_id": object_id,
                        "track_id": object_id,
                        "label": prompt,
                        "bbox": [round(float(item), 3) for item in list(box)[:4]],
                        "confidence": round(confidence, 4),
                        "mask_path": str(mask_path) if mask_path else None,
                        "source": "sam3_transformers_from_generated_pixels",
                        "source_prompt": prompt,
                    }
                )
    except Exception as exc:
        status["blockers"].append(f"sam3_provider_run_failed:{type(exc).__name__}")
        status["error"] = str(exc)[:500]
        return [], status

    status["ran"] = True
    status["object_count"] = len(objects)
    if not objects:
        status["blockers"].append("sam3_completed_without_target_objects")
    return objects, status


def _run_sam3_provider(
    request: Mapping[str, Any],
    job_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested_kind = _string(os.environ.get(SAM3_PROVIDER_KIND_ENV)).lower()
    if requested_kind in {"transformers", "hf", "huggingface"} or (
        _truthy(os.environ.get(SAM3_TRANSFORMERS_ENV)) and _resolve_sam3_weights() is None
    ):
        return _run_transformers_sam3_provider(request, job_dir)

    model_ref, weights, model_ref_source = _resolve_sam3_model_ref()
    confidence_threshold = max(0.0, min(1.0, _float_env(SAM3_CONFIDENCE_ENV, 0.05)))
    device_name = _select_torch_device(SAM3_DEVICE_ENV)
    auto_download_enabled = _truthy(os.environ.get(SAM3_AUTODOWNLOAD_ENV))
    status = {
        "provider": "sam3",
        "kind": "sam3_semantic_segmentation",
        "ran": False,
        "device": device_name,
        "device_env": SAM3_DEVICE_ENV,
        "confidence_threshold": confidence_threshold,
        "confidence_env": SAM3_CONFIDENCE_ENV,
        "weights_path_present": bool(weights),
        "weights_path": str(weights) if weights else None,
        "weights_file_exists": bool(weights and weights.is_file()),
        "model_ref": model_ref,
        "model_ref_source": model_ref_source,
        "model_env": SAM3_MODEL_ENV,
        "provider_kind_env": SAM3_PROVIDER_KIND_ENV,
        "provider_kind": "ultralytics",
        "autodownload_env": SAM3_AUTODOWNLOAD_ENV,
        "autodownload_enabled": auto_download_enabled,
        "hf_token_present": bool(_hf_token_value()),
        "module_sam3_available": _module_available("sam3"),
        "module_ultralytics_available": _module_available("ultralytics"),
        "runtime_package": None,
        "runtime_class": None,
        "model_family": "sam3",
        "blockers": [],
    }
    if model_ref is None:
        status["blockers"].append("sam3_weights_path_missing")
    elif weights is not None and not weights.is_file():
        status["blockers"].append("sam3_weights_file_missing")
    if not status["module_ultralytics_available"] and not status["module_sam3_available"]:
        status["blockers"].append("sam3_runtime_package_missing")
    source_frame = Path(_string(request.get("source_generated_frame_path"))).expanduser()
    if not source_frame.is_file():
        status["blockers"].append("source_generated_frame_missing_for_sam3")
    if status["blockers"]:
        return [], status

    try:
        from ultralytics.models.sam import SAM3SemanticPredictor

        status["runtime_package"] = "ultralytics.models.sam"
        status["runtime_class"] = "SAM3SemanticPredictor"
        overrides = {
            "conf": confidence_threshold,
            "task": "segment",
            "mode": "predict",
            "model": str(model_ref),
            "device": device_name,
            "half": False,
            "verbose": False,
            "save": False,
        }
        predictor = SAM3SemanticPredictor(overrides=overrides)
        predictor.set_image(str(source_frame))
        results = predictor(text=_target_prompts(request)[:3])
    except Exception as exc:
        status["blockers"].append(f"sam3_provider_run_failed:{type(exc).__name__}")
        status["error"] = str(exc)[:500]
        return [], status

    objects: list[dict[str, Any]] = []
    for index, result in enumerate(_sequence(results)):
        bbox, confidence = _bbox_from_result(result)
        if not bbox:
            continue
        prompt = _target_prompts(request)[0]
        objects.append(
            {
                "object_id": f"sam3_target_{index:04d}",
                "track_id": f"sam3_target_{index:04d}",
                "label": prompt,
                "bbox": bbox,
                "confidence": confidence if confidence is not None else 0.5,
                "source": "sam3_semantic_predictor_from_generated_pixels",
                "source_prompt": prompt,
            }
        )
    status["ran"] = True
    status["object_count"] = len(objects)
    if not objects:
        status["blockers"].append("sam3_completed_without_target_objects")
    return objects, status


def _provider_command_status(env_name: str, provider: str) -> dict[str, Any]:
    command = _string(os.environ.get(env_name))
    return {
        "provider": provider,
        "ran": False,
        "env_name": env_name,
        "command_configured": bool(command),
        "blockers": [] if command else [f"{provider}_provider_command_not_configured"],
    }


def _run_command_provider(
    *,
    env_name: str,
    provider: str,
    request_path: Path,
    job_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    status = _provider_command_status(env_name, provider)
    command = _string(os.environ.get(env_name))
    if not command:
        return {}, status
    args = shlex.split(command)
    if not args:
        status["blockers"] = [f"{provider}_provider_command_empty"]
        return {}, status
    output_path = job_dir / f"{provider}_provider_result.json"
    env = os.environ.copy()
    env.update(
        {
            "BLUEPRINT_WAM_PROVIDER_INPUT": str(request_path),
            "BLUEPRINT_WAM_PROVIDER_OUTPUT": str(output_path),
            "BLUEPRINT_WAM_PROVIDER_JOB_DIR": str(job_dir),
        }
    )
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        status["blockers"] = [f"{provider}_provider_command_timed_out"]
        (job_dir / f"{provider}_provider.stdout.log").write_text(
            _subprocess_text(exc.stdout),
            encoding="utf-8",
        )
        (job_dir / f"{provider}_provider.stderr.log").write_text(
            _subprocess_text(exc.stderr),
            encoding="utf-8",
        )
        return {}, status
    (job_dir / f"{provider}_provider.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (job_dir / f"{provider}_provider.stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    payload = _load_json(output_path)
    if completed.returncode != 0 or not payload:
        status["blockers"] = [f"{provider}_provider_command_failed"]
        status["returncode"] = completed.returncode
        return {}, status
    status["ran"] = True
    status["blockers"] = []
    status["result_path"] = str(output_path)
    return payload, status


def _source_frame_path(request: Mapping[str, Any], provider: str) -> tuple[Path | None, list[str]]:
    source_frame = Path(_string(request.get("source_generated_frame_path"))).expanduser()
    if not source_frame.is_file():
        return None, [f"source_generated_frame_missing_for_{provider}"]
    return source_frame, []


def _run_transformers_depth_provider(
    request: Mapping[str, Any],
    job_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_id = _string(os.environ.get(DEPTH_MODEL_ENV)) or DEFAULT_DEPTH_MODEL_ID
    model_revision = (
        _string(os.environ.get(DEPTH_MODEL_REVISION_ENV)) or DEFAULT_DEPTH_MODEL_REVISION
    )
    status: dict[str, Any] = {
        "provider": "depth",
        "kind": "transformers_depth_anything_v2",
        "ran": False,
        "model_id": model_id,
        "model_revision": model_revision,
        "model_remote_code_trusted": False,
        "auto_provider_env": AUTO_DEPTH_ENV,
        "auto_provider_enabled": _truthy(os.environ.get(AUTO_DEPTH_ENV)),
        "transformers_available": _module_available("transformers"),
        "blockers": [],
    }
    if not status["auto_provider_enabled"]:
        status["blockers"].append("depth_provider_command_not_configured")
        return {}, status
    if not status["transformers_available"]:
        status["blockers"].append("transformers_depth_provider_package_missing")
        return {}, status
    if model_id != DEFAULT_DEPTH_MODEL_ID or model_revision != DEFAULT_DEPTH_MODEL_REVISION:
        status["blockers"].append("depth_model_revision_not_approved")
        return {}, status
    source_frame, blockers = _source_frame_path(request, "depth")
    if blockers:
        status["blockers"].extend(blockers)
        return {}, status
    try:
        from transformers import pipeline

        pipe = pipeline(
            "depth-estimation",
            model=model_id,
            revision=model_revision,
            trust_remote_code=False,
        )
        output = pipe(Image.open(source_frame).convert("RGB"))
        depth_image = output.get("depth")
        if depth_image is None:
            status["blockers"].append("depth_provider_returned_no_depth_image")
            return {}, status
        if not isinstance(depth_image, Image.Image):
            depth_image = Image.fromarray(depth_image)
        depth_gray = depth_image.convert("L")
        depth_path = job_dir / "depth_provider_depth_map.png"
        depth_gray.save(depth_path)
        extrema = depth_gray.getextrema()
        mean = float(ImageStat.Stat(depth_gray).mean[0])
        relative_depth = round(mean / 255.0, 6)
    except Exception as exc:
        status["blockers"].append(f"depth_provider_run_failed:{type(exc).__name__}")
        status["error"] = str(exc)[:500]
        return {}, status
    status.update(
        {
            "ran": True,
            "blockers": [],
            "depth_map_path": str(depth_path),
            "depth_min_max": [float(extrema[0]), float(extrema[1])],
        }
    )
    payload = {
        "depth_estimates": [
            {
                "label": "generated_frame_depth_anything_v2",
                "object_id": "generated_frame",
                "relative_depth": relative_depth,
                "metric_depth": None,
                "confidence": 0.62,
                "calibration_source": "not_available_monocular_depth",
                "metric_depth_source": "not_metric_depth_monocular_model",
                "source": "Depth Anything V2 via transformers from generated pixels",
                "depth_map_ref": str(depth_path),
            }
        ]
    }
    write_json(job_dir / "depth_provider_result.json", payload)
    return payload, status


def _run_da3_depth_provider(
    request: Mapping[str, Any],
    job_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_id = _string(os.environ.get(DA3_MODEL_ENV)) or DEFAULT_DA3_MODEL_ID
    model_revision = _string(os.environ.get(DA3_MODEL_REVISION_ENV)) or DEFAULT_DA3_MODEL_REVISION
    process_res = max(32, _int_env(DA3_PROCESS_RES_ENV, 504))
    device_name = _select_torch_device(DA3_DEVICE_ENV)
    status: dict[str, Any] = {
        "provider": "depth",
        "kind": "depth_anything_3",
        "ran": False,
        "model_id": model_id,
        "model_revision": model_revision,
        "model_remote_code_trusted": False,
        "device": device_name,
        "device_env": DA3_DEVICE_ENV,
        "process_res": process_res,
        "process_res_env": DA3_PROCESS_RES_ENV,
        "auto_provider_env": AUTO_DA3_ENV,
        "auto_provider_enabled": _truthy(os.environ.get(AUTO_DA3_ENV)),
        "depth_provider_kind_env": DEPTH_PROVIDER_KIND_ENV,
        "depth_provider_kind": _string(os.environ.get(DEPTH_PROVIDER_KIND_ENV)) or "da3",
        "depth_anything_3_available": _module_available("depth_anything_3"),
        "blockers": [],
    }
    if not status["auto_provider_enabled"]:
        status["blockers"].append("da3_depth_provider_not_enabled")
        return {}, status
    if not status["depth_anything_3_available"]:
        status["blockers"].append("da3_depth_provider_package_missing")
        return {}, status
    if model_id != DEFAULT_DA3_MODEL_ID or model_revision != DEFAULT_DA3_MODEL_REVISION:
        status["blockers"].append("da3_model_revision_not_approved")
        return {}, status
    source_frame, blockers = _source_frame_path(request, "depth")
    if blockers:
        status["blockers"].extend(blockers)
        return {}, status
    try:
        import numpy as np
        import torch
        from depth_anything_3.api import DepthAnything3
        from huggingface_hub import snapshot_download

        device = torch.device(device_name)
        model_path = snapshot_download(
            repo_id=model_id,
            revision=model_revision,
        )
        model = DepthAnything3.from_pretrained(model_path).to(device=device)
        prediction = model.inference([str(source_frame)], process_res=process_res)
        depth = np.asarray(prediction.depth[0], dtype=np.float32)
        finite = depth[np.isfinite(depth)]
        if finite.size == 0:
            status["blockers"].append("da3_depth_provider_returned_no_finite_depth")
            return {}, status
        depth_min = float(finite.min())
        depth_max = float(finite.max())
        if depth_max > depth_min:
            normalized = (depth - depth_min) / (depth_max - depth_min)
        else:
            normalized = np.zeros_like(depth, dtype=np.float32)
        depth_path = job_dir / "da3_depth_provider_depth_map.png"
        Image.fromarray((normalized * 255.0).clip(0, 255).astype("uint8")).save(depth_path)
        relative_depth = round(float(np.nanmean(normalized)), 6)
    except Exception as exc:
        status["blockers"].append(f"da3_depth_provider_run_failed:{type(exc).__name__}")
        status["error"] = str(exc)[:500]
        return {}, status
    status.update(
        {
            "ran": True,
            "blockers": [],
            "depth_map_path": str(depth_path),
            "depth_min_max": [depth_min, depth_max],
            "device": str(device),
        }
    )
    payload = {
        "depth_estimates": [
            {
                "label": "generated_frame_depth_anything_3",
                "object_id": "generated_frame",
                "relative_depth": relative_depth,
                "metric_depth": None,
                "confidence": 0.68,
                "calibration_source": "not_available_for_sim_only_probe",
                "metric_depth_source": "not_sensor_depth_generated_pixel_da3",
                "source": "Depth Anything 3 via depth_anything_3.api from generated pixels",
                "depth_map_ref": str(depth_path),
            }
        ]
    }
    write_json(job_dir / "da3_depth_provider_result.json", payload)
    return payload, status


def _tensor_to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return _sequence(value)


def _run_pose_model_provider(
    request: Mapping[str, Any],
    job_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_model = _string(os.environ.get(POSE_MODEL_ENV)) or DEFAULT_POSE_MODEL_PATH
    model_path = Path(raw_model).expanduser()
    status: dict[str, Any] = {
        "provider": "pose",
        "kind": "ultralytics_yolo_pose",
        "ran": False,
        "auto_provider_env": AUTO_POSE_ENV,
        "auto_provider_enabled": _truthy(os.environ.get(AUTO_POSE_ENV)),
        "model_path": str(model_path),
        "model_path_configured": bool(_string(os.environ.get(POSE_MODEL_ENV))),
        "module_ultralytics_available": _module_available("ultralytics"),
        "blockers": [],
    }
    if not status["auto_provider_enabled"]:
        status["blockers"].append("pose_provider_command_not_configured")
        return {}, status
    if not status["module_ultralytics_available"]:
        status["blockers"].append("pose_runtime_package_missing")
        return {}, status
    source_frame, blockers = _source_frame_path(request, "pose")
    if blockers:
        status["blockers"].extend(blockers)
        return {}, status
    try:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        results = model(str(source_frame), verbose=False, imgsz=320)
    except Exception as exc:
        status["blockers"].append(f"pose_provider_run_failed:{type(exc).__name__}")
        status["error"] = str(exc)[:500]
        return {}, status
    pose_rows: list[dict[str, Any]] = []
    for index, result in enumerate(_sequence(results)):
        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)
        box_rows = _tensor_to_list(getattr(boxes, "xyxy", None))
        confidence_rows = _tensor_to_list(getattr(boxes, "conf", None))
        keypoint_rows = _tensor_to_list(getattr(keypoints, "xy", None))
        keypoint_conf_rows = _tensor_to_list(getattr(keypoints, "conf", None))
        for det_index, bbox in enumerate(box_rows):
            keypoints_xy = keypoint_rows[det_index] if det_index < len(keypoint_rows) else []
            keypoints_conf = (
                keypoint_conf_rows[det_index] if det_index < len(keypoint_conf_rows) else []
            )
            pose_rows.append(
                {
                    "object_id": f"yolo_pose_{index:04d}_{det_index:04d}",
                    "pose_2d": {
                        "bbox_xyxy": [round(float(item), 3) for item in bbox],
                        "keypoints_xy": keypoints_xy,
                        "keypoint_confidence": keypoints_conf,
                    },
                    "pose_3d": None,
                    "confidence": round(float(confidence_rows[det_index]), 4)
                    if det_index < len(confidence_rows)
                    else 0.5,
                    "source": "Ultralytics YOLO pose from generated pixels",
                    "calibration_source": None,
                }
            )
    status.update({"ran": True, "pose_count": len(pose_rows), "blockers": []})
    if not pose_rows:
        status["blockers"].append("pose_provider_completed_without_pose_detections")
    payload = {"pose_estimates": pose_rows}
    write_json(job_dir / "pose_provider_result.json", payload)
    return payload, status


def _run_pose_model_status() -> dict[str, Any]:
    model_path = Path(_string(os.environ.get(POSE_MODEL_ENV))).expanduser()
    status = {
        "provider": "pose",
        "ran": False,
        "model_env_name": POSE_MODEL_ENV,
        "model_path_configured": bool(_string(os.environ.get(POSE_MODEL_ENV))),
        "model_path": str(model_path) if _string(os.environ.get(POSE_MODEL_ENV)) else None,
        "module_ultralytics_available": _module_available("ultralytics"),
        "blockers": [],
    }
    if not status["model_path_configured"]:
        status["blockers"].append("pose_model_path_not_configured")
    elif not model_path.is_file():
        status["blockers"].append("pose_model_file_missing")
    if not status["module_ultralytics_available"]:
        status["blockers"].append("pose_runtime_package_missing")
    return status


def run_external_backend_from_env() -> int:
    request_path = Path(os.environ.get("BLUEPRINT_WAM_PERCEPTION_BACKEND_INPUT", "")).expanduser()
    output_path = Path(os.environ.get("BLUEPRINT_WAM_PERCEPTION_BACKEND_OUTPUT", "")).expanduser()
    job_dir = Path(os.environ.get("BLUEPRINT_WAM_PERCEPTION_BACKEND_JOB_DIR", ".")).expanduser()
    ensure_dir(job_dir)
    request = _load_json(request_path)
    blockers: list[str] = []

    objects, sam3_status = _run_sam3_provider(request, job_dir)
    if _env_present(DEPTH_COMMAND_ENV):
        depth_payload, depth_status = _run_command_provider(
            env_name=DEPTH_COMMAND_ENV,
            provider="depth",
            request_path=request_path,
            job_dir=job_dir,
        )
    elif _string(os.environ.get(DEPTH_PROVIDER_KIND_ENV)).lower() in {
        "da3",
        "depth_anything_3",
        "depth-anything-3",
    } or _truthy(os.environ.get(AUTO_DA3_ENV)):
        depth_payload, depth_status = _run_da3_depth_provider(request, job_dir)
    else:
        depth_payload, depth_status = _run_transformers_depth_provider(request, job_dir)
    pose_required = _truthy(os.environ.get(REQUIRE_POSE_ENV))
    if _env_present(POSE_COMMAND_ENV):
        pose_payload, pose_command_status = _run_command_provider(
            env_name=POSE_COMMAND_ENV,
            provider="pose",
            request_path=request_path,
            job_dir=job_dir,
        )
        pose_model_status = {}
    elif _truthy(os.environ.get(AUTO_POSE_ENV)):
        pose_payload, pose_command_status = _run_pose_model_provider(request, job_dir)
        pose_model_status = {}
    elif not pose_required:
        pose_payload = {}
        pose_command_status = {
            "provider": "pose",
            "ran": False,
            "required": False,
            "status": "not_requested",
            "env_name": POSE_COMMAND_ENV,
            "command_configured": False,
            "blockers": [],
        }
        pose_model_status = {}
    else:
        pose_payload, pose_command_status = {}, _provider_command_status(
            POSE_COMMAND_ENV, "pose"
        )
        pose_model_status = _run_pose_model_status()

    provider_statuses = [sam3_status, depth_status, pose_command_status]
    if pose_model_status:
        provider_statuses.append(pose_model_status)
    for status in provider_statuses:
        blockers.extend(_sequence(status.get("blockers")))

    depth_rows = _sequence(depth_payload.get("depth_estimates"))
    pose_rows = _sequence(pose_payload.get("pose_estimates"))
    real_ran = bool(
        sam3_status.get("ran") or depth_status.get("ran") or pose_command_status.get("ran")
    )
    if not real_ran:
        blockers.append("no_real_sam3_depth_or_pose_provider_ran")

    payload = {
        "schema_version": BACKEND_RESULT_SCHEMA_VERSION,
        "status": "completed" if real_ran and not blockers else "partial" if real_ran else "blocked",
        "backend": {
            "kind": "real_provider_probe",
            "status": "completed"
            if real_ran and not blockers
            else "partial"
            if real_ran
            else "blocked",
            "real_sam_or_depth_model_ran": real_ran,
            "blockers": sorted(set(str(item) for item in blockers if item)),
            "provider_statuses": provider_statuses,
        },
        "objects": objects,
        "depth_estimates": depth_rows,
        "pose_estimates": pose_rows,
        "contact_likelihood": None,
        "claim_boundary": {
            "harness_outputs_are_derived_from_generated_pixels": True,
            "sam3_masks_are_not_physical_truth": True,
            "estimated_depth_is_not_sensor_depth": True,
            "pose_estimates_are_not_physical_pose_truth": True,
            "non_ranking_operational_claim_proven": False,
        },
    }
    write_json(output_path, payload)
    return 0


def _discover_generated_frame() -> Path | None:
    patterns = (
        "robot_eval_jobs/**/robot_policy_wam_closed_loop/**/*frame*.jpg",
        "robot_eval_jobs/**/robot_policy_wam_closed_loop/**/*frame*.png",
        "robot_eval_jobs/**/generated_rollout_frame_review/frames/*.jpg",
        "robot_eval_jobs/**/generated_rollout_frame_review/frames/*.png",
        "robot_eval_jobs/**/mujoco_frames/**/*.png",
    )
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(Path(".").glob(pattern))
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        return None
    return sorted(existing, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def _write_synthetic_probe_frame(output_dir: Path) -> Path:
    frame_path = output_dir / "synthetic_generated_probe_frame.jpg"
    image = Image.new("RGB", (640, 480), (38, 42, 48))
    draw = ImageDraw.Draw(image)
    draw.rectangle((250, 185, 390, 285), fill=(194, 70, 54), outline=(255, 255, 255), width=4)
    draw.rectangle((290, 300, 350, 345), fill=(90, 120, 150))
    image.save(frame_path)
    return frame_path


def _default_output_dir() -> Path:
    stamp = utc_now_iso().replace(":", "").replace("-", "").split(".")[0]
    return Path("robot_eval_jobs") / f"{DEFAULT_JOB_PREFIX}_{stamp}Z"


def _validation_status(
    path: Path | None,
    *,
    expected_frame_path: Path | None = None,
    target_prompts: Sequence[str] | None = None,
) -> dict[str, Any]:
    if path is None:
        return {
            "status": "not_requested",
            "optional_validation_requested": False,
            "required_for_sim_only": False,
            "diagnostic_issues": [],
        }
    if not path.is_file():
        return {
            "status": "diagnostic_issues",
            "optional_validation_requested": True,
            "required_for_sim_only": False,
            "path": str(path),
            "diagnostic_issues": ["validation_set_file_missing"],
        }
    rows = _validation_rows_from_value(_load_json_value(path))
    supplied_target_prompts = [
        prompt for prompt in (_string(item) for item in _sequence(target_prompts)) if prompt
    ]
    capture_backed_rows = [
        row for row in rows if _row_has_truthy_key(row, REAL_VALIDATION_FLAG_KEYS)
    ]
    real_labeled_rows = [
        row for row in capture_backed_rows if _row_has_label(row)
    ]
    sourced_real_labeled_rows = [
        row for row in real_labeled_rows if _row_has_source_ref(row)
    ]
    row_results: list[dict[str, Any]] = []
    accepted_contract_rows = 0
    frame_matched_rows = 0
    target_prompt_rows = 0
    provenance_rows = 0
    provider_only_rows = 0
    empty_target_rows = 0
    for index, row in enumerate(rows):
        row_issues: list[str] = []
        has_capture_backing = _row_has_truthy_key(row, REAL_VALIDATION_FLAG_KEYS)
        has_label = _row_has_label(row)
        has_source = _row_has_source_ref(row)
        has_target = bool(_target_strings(row) or supplied_target_prompts)
        has_provenance = _row_has_label_provenance(row)
        frame_matches = _row_matches_expected_frame(row, expected_frame_path)
        provider_only_source = _row_has_provider_only_source(row)
        empty_target = _row_has_empty_target_field(row)
        target_prompt_rows += int(has_target and not empty_target)
        provenance_rows += int(has_provenance)
        frame_matched_rows += int(frame_matches)
        provider_only_rows += int(provider_only_source)
        empty_target_rows += int(empty_target)
        if not has_capture_backing:
            row_issues.append("row_not_capture_backed_or_real_anchor")
        if not has_label:
            row_issues.append("row_validation_label_missing")
        if not has_source:
            row_issues.append("row_source_reference_missing")
        if provider_only_source:
            row_issues.append("row_source_is_provider_only_output")
        if empty_target:
            row_issues.append("row_target_prompt_empty")
        elif not has_target:
            row_issues.append("row_target_prompt_missing")
        if not has_provenance:
            row_issues.append("row_reviewer_or_label_provenance_missing")
        if not frame_matches:
            row_issues.append("row_frame_id_or_path_mismatch")
        if not row_issues:
            accepted_contract_rows += 1
        row_results.append(
            {
                "row_index": index,
                "step_index": row.get("step_index"),
                "frame_refs": _frame_ref_strings(row),
                "target_prompts": _target_strings(row) or supplied_target_prompts,
                "accepted_for_probe_validation": not row_issues,
                "diagnostic_issues": row_issues,
            }
        )
    diagnostic_issues: list[str] = []
    if not rows:
        diagnostic_issues.append("validation_set_rows_missing")
    row_level_issues = sorted(
        {
            str(issue)
            for result in row_results
            for issue in _sequence(result.get("diagnostic_issues"))
            if issue
        }
    )
    if not capture_backed_rows:
        diagnostic_issues.append("capture_backed_validation_rows_missing")
    if not real_labeled_rows:
        diagnostic_issues.append("real_labeled_validation_rows_missing")
    if real_labeled_rows and not sourced_real_labeled_rows:
        diagnostic_issues.append("real_labeled_validation_source_missing")
    if rows and not target_prompt_rows:
        diagnostic_issues.append("validation_target_prompt_missing")
    if empty_target_rows:
        diagnostic_issues.append("validation_target_prompt_empty")
    if real_labeled_rows and not provenance_rows:
        diagnostic_issues.append("real_labeled_validation_provenance_missing")
    if provider_only_rows:
        diagnostic_issues.append("provider_only_validation_source_not_accepted")
    if rows and not frame_matched_rows:
        diagnostic_issues.append("validation_frame_id_or_path_mismatch")
    if rows and not accepted_contract_rows:
        diagnostic_issues.append("validation_rows_do_not_satisfy_probe_contract")
    diagnostic_issues.extend(row_level_issues)
    return {
        "status": "available" if not diagnostic_issues else "diagnostic_issues",
        "optional_validation_requested": True,
        "required_for_sim_only": False,
        "path": str(path),
        "row_count": len(rows),
        "capture_backed_row_count": len(capture_backed_rows),
        "real_labeled_row_count": len(real_labeled_rows),
        "sourced_real_labeled_row_count": len(sourced_real_labeled_rows),
        "target_prompt_row_count": target_prompt_rows,
        "frame_matched_row_count": frame_matched_rows,
        "provenance_row_count": provenance_rows,
        "provider_only_row_count": provider_only_rows,
        "accepted_contract_row_count": accepted_contract_rows,
        "row_results": row_results,
        "optional_validation_row_contract": {
            "accepts_capture_backed_or_real_anchor": True,
            "must_include_validation_label": True,
            "must_include_source_reference": True,
            "must_include_target_prompt_or_cli_target_prompt": True,
            "must_match_probe_frame_when_frame_id_or_path_is_supplied": True,
            "must_include_reviewer_or_label_provenance": True,
            "provider_only_outputs_are_not_validation_sources": True,
            "accepted_capture_truth_flags": list(REAL_VALIDATION_FLAG_KEYS),
            "accepted_label_fields": list(VALIDATION_LABEL_KEYS),
            "accepted_source_fields": list(VALIDATION_SOURCE_KEYS),
            "accepted_target_fields": list(VALIDATION_TARGET_KEYS),
            "accepted_frame_id_fields": list(VALIDATION_FRAME_ID_KEYS),
            "accepted_frame_path_fields": list(VALIDATION_FRAME_PATH_KEYS),
            "accepted_provenance_fields": list(VALIDATION_PROVENANCE_KEYS),
        },
        "diagnostic_issues": sorted(set(diagnostic_issues)),
    }


def _provider_readiness_snapshot() -> dict[str, Any]:
    sam3_weights = _resolve_sam3_weights()
    sam3_model_ref, _, sam3_model_ref_source = _resolve_sam3_model_ref()
    pose_model = Path(_string(os.environ.get(POSE_MODEL_ENV))).expanduser()
    return {
        "sam3": {
            "sam3_module_available": _module_available("sam3"),
            "ultralytics_available": _module_available("ultralytics"),
            "transformers_available": _module_available("transformers"),
            "weights_env_presence": _redacted_env_presence([SAM3_WEIGHTS_ENV, ALT_SAM3_WEIGHTS_ENV]),
            "weights_path_present": bool(sam3_weights),
            "weights_file_exists": bool(sam3_weights and sam3_weights.is_file()),
            "model_env": SAM3_MODEL_ENV,
            "model_ref": sam3_model_ref,
            "model_ref_source": sam3_model_ref_source,
            "provider_kind_env": SAM3_PROVIDER_KIND_ENV,
            "provider_kind": _string(os.environ.get(SAM3_PROVIDER_KIND_ENV)) or None,
            "autodownload_env": SAM3_AUTODOWNLOAD_ENV,
            "autodownload_enabled": _truthy(os.environ.get(SAM3_AUTODOWNLOAD_ENV)),
            "transformers_env": SAM3_TRANSFORMERS_ENV,
            "transformers_provider_enabled": _truthy(os.environ.get(SAM3_TRANSFORMERS_ENV)),
            "hf_model_env": SAM3_HF_MODEL_ENV,
            "hf_model_id": _string(os.environ.get(SAM3_HF_MODEL_ENV)) or DEFAULT_SAM3_HF_MODEL_ID,
            "hf_model_revision_env": SAM3_HF_REVISION_ENV,
            "hf_model_revision": (
                _string(os.environ.get(SAM3_HF_REVISION_ENV))
                or DEFAULT_SAM3_HF_MODEL_REVISION
            ),
            "remote_code_trusted": False,
            "hf_token_env_presence": _redacted_env_presence(HF_TOKEN_ENVS),
            "hf_token_file_env_presence": _redacted_file_env_presence(HF_TOKEN_FILE_ENVS),
        },
        "depth": {
            "depth_anything_available": _module_available("depth_anything")
            or _module_available("depth_anything_v2"),
            "depth_anything_3_available": _module_available("depth_anything_3"),
            "depth_command_configured": _env_present(DEPTH_COMMAND_ENV),
            "depth_provider_kind": _string(os.environ.get(DEPTH_PROVIDER_KIND_ENV)) or None,
            "da3_auto_provider_enabled": _truthy(os.environ.get(AUTO_DA3_ENV)),
            "depth_model_revision": (
                _string(os.environ.get(DEPTH_MODEL_REVISION_ENV))
                or DEFAULT_DEPTH_MODEL_REVISION
            ),
            "da3_model_revision": (
                _string(os.environ.get(DA3_MODEL_REVISION_ENV))
                or DEFAULT_DA3_MODEL_REVISION
            ),
            "remote_code_trusted": False,
        },
        "pose": {
            "pose_command_configured": _env_present(POSE_COMMAND_ENV),
            "pose_model_configured": _env_present(POSE_MODEL_ENV),
            "pose_model_file_exists": bool(_env_present(POSE_MODEL_ENV) and pose_model.is_file()),
            "ultralytics_available": _module_available("ultralytics"),
        },
    }


def _provider_completed(provider_statuses: Sequence[Any], provider: str) -> bool:
    for status_value in provider_statuses:
        status = _mapping(status_value)
        if status.get("provider") != provider:
            continue
        return bool(status.get("ran")) and not bool(_sequence(status.get("blockers")))
    return False


def run_probe(
    *,
    output_dir: Path,
    generated_frame_path: Path | None,
    validation_set_path: Path | None,
    policy_id: str,
    policy_observation_schema: Mapping[str, Any],
    target_prompts: Sequence[str] | None = None,
    require_pose: bool = True,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    selected_frame = generated_frame_path if generated_frame_path and generated_frame_path.is_file() else None
    frame_source = "explicit_generated_frame_path" if selected_frame else None
    if selected_frame is None:
        selected_frame = _discover_generated_frame()
        frame_source = "discovered_existing_generated_or_sim_frame" if selected_frame else None
    synthetic_frame_used = False
    if selected_frame is None:
        selected_frame = _write_synthetic_probe_frame(output_dir)
        frame_source = "synthetic_probe_frame"
        synthetic_frame_used = True

    cleaned_target_prompts = [
        prompt for prompt in (_string(item) for item in _sequence(target_prompts)) if prompt
    ]
    validation_snapshot = _validation_status(
        validation_set_path,
        expected_frame_path=selected_frame,
        target_prompts=cleaned_target_prompts,
    )
    backend_command = [
        sys.executable,
        "-m",
        "blueprint_pipeline.wam_real_provider_validation_probe",
        "backend",
    ]
    harness_dir = output_dir / "wam_derived_observation_harness"
    previous_pose_required = os.environ.get(REQUIRE_POSE_ENV)
    os.environ[REQUIRE_POSE_ENV] = "true" if require_pose else "false"
    try:
        harness_result = run_wam_derived_observation_harness_step(
            output_dir=harness_dir,
            step_index=1,
            source_generated_frame_path=selected_frame,
            source_wam_rollout_id=output_dir.name,
            transition_id="real_provider_validation_probe_step_0001",
            source_policy_action={
                "action_type": "perception_validation_probe",
                "task_prompt": cleaned_target_prompts[0] if cleaned_target_prompts else None,
            },
            current_policy_observation={
                "schema_version": "blueprint_policy_observation.v1",
                "camera_frame_path": str(selected_frame),
                "visual_observation": {
                    "camera_frame_path": str(selected_frame),
                    "wam_generated_observation": True,
                },
            },
            eval_ready_task_grounding={
                "schema_version": "eval_ready_task_grounding.v1",
                "status": "probe_prompt_only" if cleaned_target_prompts else "not_supplied",
                "task": {
                    "task_id": "real_provider_validation_probe",
                    "target_prompts_for_object_index_backends": cleaned_target_prompts,
                },
                "selected_task_target": {
                    "object_id": "sam3_prompt_target",
                    "label": cleaned_target_prompts[0],
                    "source_prompt": cleaned_target_prompts[0],
                    "source": "probe_cli_target_prompt",
                }
                if cleaned_target_prompts
                else {},
            },
            backend_kind="real_provider_probe",
            backend_command=backend_command,
            allow_external_backend=True,
            backend_timeout_seconds=600,
            policy_id=policy_id,
            declared_policy_observation_schema=policy_observation_schema,
            validation_set_path=validation_set_path,
        )
    finally:
        if previous_pose_required is None:
            os.environ.pop(REQUIRE_POSE_ENV, None)
        else:
            os.environ[REQUIRE_POSE_ENV] = previous_pose_required
    backend = _mapping(harness_result["step_record"].get("harness_backend"))
    validation_report = _mapping(harness_result.get("validation_report"))
    blockers = list(_sequence(backend.get("blockers")))
    if synthetic_frame_used:
        blockers.append("synthetic_probe_frame_used_no_wam_frame_supplied_or_discovered")
    if not cleaned_target_prompts:
        blockers.append("target_prompt_not_supplied")
    optional_validation_diagnostic_issues = [
        *[
            str(issue)
            for issue in _sequence(validation_snapshot.get("diagnostic_issues"))
            if issue
        ],
        *[
            str(issue)
            for issue in (
                _sequence(validation_report.get("diagnostic_issues"))
                or _sequence(validation_report.get("blockers"))
            )
            if issue
        ],
    ]

    provider_statuses = _sequence(backend.get("provider_statuses"))
    sam3_completed = _provider_completed(provider_statuses, "sam3")
    depth_completed = _provider_completed(provider_statuses, "depth")
    pose_completed = _provider_completed(provider_statuses, "pose")
    provider_pair_completed = sam3_completed and depth_completed
    provider_triplet_completed = sam3_completed and depth_completed and pose_completed
    provider_requirement_completed = (
        provider_triplet_completed if require_pose else provider_pair_completed
    )
    validation_contract_available = validation_snapshot.get("status") == "available"
    validation_requested = validation_snapshot.get("status") != "not_requested"
    backend_completed = backend.get("status") == "completed"
    manifest = {
        "schema_version": PROOF_MANIFEST_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "completed" if not blockers and backend_completed else "blocked",
        "proof_scope": {
            "requested": (
                "real_sam3_depth_pose_providers_with_optional_labeled_validation_data"
                if require_pose
                else "real_sam3_depth_providers_with_optional_pose_and_labeled_validation_data"
            ),
            "real_pose_provider_required": require_pose,
            "real_sam3_provider_completed": sam3_completed,
            "real_depth_provider_completed": depth_completed,
            "real_pose_provider_completed": pose_completed,
            "real_sam3_depth_provider_pair_completed": provider_pair_completed,
            "real_sam3_depth_pose_provider_triplet_completed": provider_triplet_completed,
            "real_sam3_depth_pose_proof_complete": provider_triplet_completed,
            "real_provider_requirement_completed": provider_requirement_completed,
            "real_sam3_depth_proof_complete": provider_pair_completed,
            "optional_labeled_validation_requested": validation_requested,
            "optional_labeled_validation_completed": bool(
                validation_requested
                and validation_report.get("status") == "completed"
                and validation_contract_available
                and backend_completed
            ),
            "non_ranking_operational_claim_proven": False,
            "generated_world_rank_fidelity_result_proven": False,
        },
        "selected_generated_frame_path": str(selected_frame),
        "target_prompts": cleaned_target_prompts,
        "frame_source": frame_source,
        "synthetic_probe_frame_used": synthetic_frame_used,
        "provider_readiness": _provider_readiness_snapshot(),
        "provider_statuses": provider_statuses,
        "validation_set": validation_snapshot,
        "harness_backend": backend,
        "harness_artifact_paths": harness_result.get("artifact_paths"),
        "validation_report": validation_report,
        "optional_validation_diagnostic_issues": sorted(
            set(optional_validation_diagnostic_issues)
        ),
        "false_success_reduction_metrics": harness_result.get("false_success_reduction_metrics"),
        "blockers": sorted(set(str(item) for item in blockers if item)),
        "claim_boundary": {
            "harness_outputs_are_derived_observations_not_real_sensors": True,
            "generated_frames_are_not_capture_truth": True,
            "sam3_masks_are_not_physical_truth": True,
            "inferred_depth_is_not_sensor_depth": True,
            "contact_likelihood_is_not_physical_contact_proof": True,
            "generated_rollout_or_harness_outputs_do_not_prove_accepted_anchor_success": True,
        },
    }
    manifest_path = output_dir / "wam_real_provider_validation_proof_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _schema_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.policy_schema == "rgb_only":
        return {"rgb_only": True, "fields": ["camera_frame_path", "visual_observation"]}
    return {
        "modalities": ["rgb", "depth", "mask", "pose", "state"],
        "fields": [
            "camera_frame_path",
            "visual_observation",
            "objects",
            "depth_estimates",
            "pose_estimates",
            "robot_state",
            "contact_likelihood",
            "uncertainty",
            "consistency_checks",
        ],
        "supports_depth": True,
        "supports_masks": True,
        "supports_state": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="run the harness proof probe")
    run.add_argument("--output-dir", type=Path, default=None)
    run.add_argument("--generated-frame", type=Path, default=None)
    run.add_argument("--validation-set", type=Path, default=None)
    run.add_argument("--policy-id", default="wam_real_provider_probe_policy")
    run.add_argument("--policy-schema", choices=["rgb_only", "rgbd_mask_pose"], default="rgbd_mask_pose")
    run.add_argument(
        "--target-prompt",
        action="append",
        default=[],
        help="Target object/concept prompt to pass to SAM3; can be supplied multiple times.",
    )
    run.add_argument(
        "--no-require-pose",
        action="store_true",
        help="Require real SAM3 and depth providers but treat pose as optional diagnostics.",
    )
    sub.add_parser("backend", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "backend":
        return run_external_backend_from_env()
    if args.command not in {None, "run"}:
        parser.error(f"unsupported command: {args.command}")
    output_dir = args.output_dir or _default_output_dir()
    manifest = run_probe(
        output_dir=output_dir,
        generated_frame_path=args.generated_frame,
        validation_set_path=args.validation_set,
        policy_id=args.policy_id,
        policy_observation_schema=_schema_from_args(args),
        target_prompts=args.target_prompt,
        require_pose=not args.no_require_pose,
    )
    print(json.dumps({"status": manifest["status"], "manifest_path": manifest["manifest_path"]}))
    return 0 if manifest["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
