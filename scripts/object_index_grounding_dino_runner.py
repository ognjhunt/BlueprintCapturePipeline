#!/usr/bin/env python3
"""Grounding-DINO-style adapter for object-index stage.

Uses task-specific prompts. Falls back to YOLO-World when a native Grounding DINO
runtime is not installed on the VM.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping


def _read_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _keyframes(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    value = payload.get("keyframes")
    return [dict(item) for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _prompts(payload: Mapping[str, Any]) -> List[str]:
    bank = payload.get("prompt_bank")
    if isinstance(bank, Mapping):
        prompts = bank.get("task_specific")
        if isinstance(prompts, list):
            out = [str(item).strip() for item in prompts if str(item).strip()]
            if out:
                return out
        prompts = bank.get("all")
        if isinstance(prompts, list):
            return [str(item).strip() for item in prompts if str(item).strip()]
    return []


def _run_with_ultralytics(payload: Mapping[str, Any], prompts: List[str], keyframes: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        from ultralytics import YOLOWorld
    except Exception as exc:
        return {"detections": [], "backend_status": "skipped", "reason": f"ultralytics_missing:{exc}"}

    model_name = str(payload.get("grounding_dino_fallback_model") or "yolov8s-worldv2.pt")
    conf = float(payload.get("grounding_dino_conf") or 0.2)
    try:
        model = YOLOWorld(model_name)
        model.set_classes(prompts)
    except Exception as exc:
        return {"detections": [], "backend_status": "failed", "reason": f"fallback_model_init_failed:{exc}"}

    sources = [str(item.get("image_path") or "") for item in keyframes if str(item.get("image_path") or "").strip()]
    if not sources:
        return {"detections": [], "backend_status": "skipped", "reason": "missing_keyframe_sources"}
    try:
        results = model.predict(source=sources, conf=conf, verbose=False)
    except Exception as exc:
        return {"detections": [], "backend_status": "failed", "reason": f"fallback_predict_failed:{exc}"}
    detections: List[Dict[str, Any]] = []
    for keyframe, result in zip(keyframes, results):
        names = result.names if isinstance(getattr(result, "names", None), Mapping) else {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy_values = boxes.xyxy.tolist() if getattr(boxes, "xyxy", None) is not None else []
        conf_values = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else []
        cls_values = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else []
        for xyxy, score, cls_id in zip(xyxy_values, conf_values, cls_values):
            label = str(names.get(int(cls_id), prompts[int(cls_id)] if int(cls_id) < len(prompts) else "object"))
            detections.append(
                {
                    "frame_index": int(keyframe.get("frame_index", 0)),
                    "label": label,
                    "score": float(score),
                    "bbox_xyxy": [float(x) for x in xyxy[:4]],
                    "source_prompt": label,
                }
            )
    return {
        "detections": detections,
        "backend_status": "ok",
        "backend_mode": "yolo_world_fallback",
        "model_name": model_name,
    }


def _run_grounding(payload: Mapping[str, Any]) -> Dict[str, Any]:
    prompts = _prompts(payload)
    keyframes = _keyframes(payload)
    if not prompts or not keyframes:
        return {"detections": [], "backend_status": "skipped", "reason": "missing_prompts_or_keyframes"}
    return _run_with_ultralytics(payload, prompts, keyframes)


def main(argv: List[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) != 2:
        print("usage: object_index_grounding_dino_runner.py <input_json> <output_json>", file=sys.stderr)
        return 2
    input_path = Path(args[0])
    output_path = Path(args[1])
    payload = _read_payload(input_path)
    result = _run_grounding(payload)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
