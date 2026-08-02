"""Local TorchVision proposals for the rendered-scene task-target pipeline.

This backend performs object detection on already-rendered RGB observations. It
emits proposals only: the rendered-scene orchestrator still binds them to 3D and
the deterministic qualification gate decides whether any target is authorized.

The model checkpoint must be supplied explicitly and match the pinned digest.
The analyzer never downloads weights or uploads scene images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


MODEL_ID = "torchvision/fasterrcnn_resnet50_fpn_v2_coco"
MODEL_IMPLEMENTATION_VERSION = "1"
MODEL_WEIGHT_SHA256 = "dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf"

DetectionBackend = Callable[[Sequence[Path]], Sequence[Sequence[Mapping[str, Any]]]]

_FRANKA_TASKS: dict[str, tuple[str, tuple[str, ...]]] = {
    "apple": ("franka_small_object_pick", ("grasp", "pick", "place")),
    "banana": ("franka_small_object_pick", ("grasp", "pick", "place")),
    "bottle": ("franka_small_object_pick", ("grasp", "pick", "place")),
    "bowl": ("franka_small_object_pick", ("grasp", "pick", "place")),
    "cup": ("franka_small_object_pick", ("grasp", "pick", "place")),
    "dining table": ("franka_work_surface_inspection", ("inspect_surface", "wipe_surface")),
    "fork": ("franka_utensil_pick", ("grasp", "pick", "place")),
    "knife": ("franka_utensil_pick", ("grasp", "pick", "place")),
    "microwave": ("franka_appliance_interaction", ("inspect", "open_door", "press_control")),
    "orange": ("franka_small_object_pick", ("grasp", "pick", "place")),
    "oven": ("franka_appliance_interaction", ("inspect", "open_door", "press_control")),
    "refrigerator": ("franka_appliance_interaction", ("inspect", "open_door")),
    "scissors": ("franka_tool_pick", ("grasp", "pick", "place")),
    "sink": ("franka_sink_inspection", ("inspect", "reach", "wipe_surface")),
    "spoon": ("franka_utensil_pick", ("grasp", "pick", "place")),
    "toaster": ("franka_appliance_interaction", ("inspect", "press_control")),
    "toothbrush": ("franka_small_object_pick", ("grasp", "pick", "place")),
}

_G1_TASKS: dict[str, tuple[str, tuple[str, ...]]] = {
    "backpack": ("g1_object_retrieval", ("approach", "grasp", "carry")),
    "bottle": ("g1_object_retrieval", ("approach", "grasp", "carry")),
    "chair": ("g1_obstacle_aware_navigation", ("approach", "navigate_around")),
    "dining table": ("g1_work_surface_inspection", ("approach", "inspect_surface")),
    "handbag": ("g1_object_retrieval", ("approach", "grasp", "carry")),
    "microwave": ("g1_appliance_interaction", ("approach", "inspect", "open_door")),
    "oven": ("g1_appliance_interaction", ("approach", "inspect", "open_door")),
    "refrigerator": ("g1_appliance_interaction", ("approach", "inspect", "open_door")),
    "sink": ("g1_sink_inspection", ("approach", "inspect")),
    "suitcase": ("g1_object_retrieval", ("approach", "grasp", "carry")),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_analyzer_contract() -> dict[str, Any]:
    """Return the portable contract to bind into analyzer provenance."""

    contract = {
        "schema_version": "rendered_scene_analyzer_contract.v1",
        "analyzer_id": "blueprint_local_torchvision_coco_detector",
        "implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "model_id": MODEL_ID,
        "model_weight_digest": "sha256:" + MODEL_WEIGHT_SHA256,
        "task_catalog_version": "franka-g1-coco-v1",
        "analyzer_request_schema": "rendered_scene_task_analyzer_request.v1",
        "analyzer_run_schema": "rendered_scene_task_analyzer_run.v1",
        "uploads_scene_data": False,
        "downloads_weights_during_analysis": False,
        "candidate_may_self_authorize": False,
    }
    contract["analyzer_contract_digest"] = canonical_digest(
        contract, digest_field="analyzer_contract_digest"
    )
    return contract


def _abstention(request_digest: str, blocker: str) -> dict[str, Any]:
    return {
        "status": "abstained",
        "analyzer_request_digest": request_digest,
        "candidate_may_self_authorize": False,
        "proposals": [],
        "blockers": [blocker],
    }


def _task_catalog(robot_id: str) -> Mapping[str, tuple[str, tuple[str, ...]]] | None:
    normalized = robot_id.strip().lower().replace("-", "_")
    if normalized in {"franka", "franka_panda", "panda"}:
        return _FRANKA_TASKS
    if normalized in {"g1", "unitree_g1"}:
        return _G1_TASKS
    return None


def _requested_labels(
    task_context: Mapping[str, Any],
    catalog: Mapping[str, tuple[str, tuple[str, ...]]],
) -> set[str]:
    raw = task_context.get("allowed_object_labels")
    if raw is None:
        raw = task_context.get("desired_object_labels")
    if not isinstance(raw, list) or not raw:
        return set(catalog)
    return {str(item).strip().lower() for item in raw if str(item).strip().lower() in catalog}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _build_proposals(
    *,
    detections: Sequence[Sequence[Mapping[str, Any]]],
    view_ids: Sequence[str],
    catalog: Mapping[str, tuple[str, tuple[str, ...]]],
    allowed_labels: set[str],
    score_threshold: float,
    maximum_proposals: int,
) -> list[dict[str, Any]]:
    best_by_label: dict[str, dict[str, Any]] = {}
    for view_id, rows in zip(view_ids, detections, strict=True):
        for row in rows:
            label = str(row.get("label") or "").strip().lower()
            score = row.get("score")
            bbox = row.get("bbox_xyxy_pixels")
            if (
                label not in allowed_labels
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or float(score) < score_threshold
                or not isinstance(bbox, (list, tuple))
                or len(bbox) != 4
            ):
                continue
            numeric_bbox = [float(item) for item in bbox]
            candidate = {
                "label": label,
                "score": float(score),
                "bbox": numeric_bbox,
                "view_id": view_id,
            }
            previous = best_by_label.get(label)
            if previous is None or (candidate["score"], candidate["view_id"], candidate["bbox"]) > (
                previous["score"],
                previous["view_id"],
                previous["bbox"],
            ):
                best_by_label[label] = candidate

    ranked = sorted(
        best_by_label.values(),
        key=lambda row: (-row["score"], row["label"], row["view_id"]),
    )[:maximum_proposals]
    proposals: list[dict[str, Any]] = []
    for index, row in enumerate(ranked, start=1):
        task_family, affordances = catalog[row["label"]]
        proposals.append(
            {
                "proposal_id": f"{_slug(row['view_id'])}-{_slug(row['label'])}-{index:03d}",
                "object_label": row["label"],
                "task_family": task_family,
                "affordances": list(affordances),
                "visual_confidence": round(row["score"], 9),
                "bbox_xyxy_pixels": [round(item, 3) for item in row["bbox"]],
                "binding_view_id": row["view_id"],
                "supporting_view_ids": [row["view_id"]],
            }
        )
    return proposals


def _torchvision_detector(weights_path: Path) -> DetectionBackend:
    import torch
    from PIL import Image
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    transform = weights.transforms()
    categories = weights.meta["categories"]

    def detect(paths: Sequence[Path]) -> Sequence[Sequence[Mapping[str, Any]]]:
        images = [transform(Image.open(path).convert("RGB")) for path in paths]
        with torch.inference_mode():
            results = model(images)
        normalized: list[list[dict[str, Any]]] = []
        for result in results:
            rows: list[dict[str, Any]] = []
            for score, label, bbox in zip(
                result["scores"], result["labels"], result["boxes"], strict=True
            ):
                rows.append(
                    {
                        "label": categories[int(label)],
                        "score": float(score),
                        "bbox_xyxy_pixels": [float(item) for item in bbox],
                    }
                )
            normalized.append(rows)
        return normalized

    return detect


def analyze_payload(
    payload: Mapping[str, Any],
    *,
    weights_path: Path,
    score_threshold: float = 0.15,
    maximum_proposals: int = 12,
    detector: DetectionBackend | None = None,
) -> dict[str, Any]:
    request = payload.get("analyzer_request")
    runtime = payload.get("runtime_inputs")
    if not isinstance(request, Mapping) or not isinstance(runtime, Mapping):
        return _abstention("", "torchvision_analyzer_payload_invalid")
    request_digest = str(request.get("analyzer_request_digest") or "")
    catalog = _task_catalog(str(request.get("robot_id") or ""))
    if catalog is None:
        return _abstention(request_digest, "torchvision_analyzer_robot_unsupported")
    if not weights_path.is_file():
        return _abstention(request_digest, "torchvision_analyzer_weights_missing")
    if _sha256(weights_path) != MODEL_WEIGHT_SHA256:
        return _abstention(request_digest, "torchvision_analyzer_weights_digest_mismatch")

    declared_views = request.get("rendered_views")
    runtime_views = runtime.get("rendered_views")
    if not isinstance(declared_views, list) or not isinstance(runtime_views, list):
        return _abstention(request_digest, "torchvision_analyzer_views_invalid")
    declared_by_id = {
        str(row.get("view_id") or ""): row for row in declared_views if isinstance(row, Mapping)
    }
    paths: list[Path] = []
    view_ids: list[str] = []
    for row in runtime_views:
        if not isinstance(row, Mapping):
            return _abstention(request_digest, "torchvision_analyzer_runtime_view_invalid")
        view_id = str(row.get("view_id") or "")
        declared = declared_by_id.get(view_id)
        path = Path(str(row.get("rgb_path") or ""))
        if declared is None or not path.is_file():
            return _abstention(request_digest, "torchvision_analyzer_view_binding_invalid")
        if "sha256:" + _sha256(path) != declared.get("rgb_digest"):
            return _abstention(request_digest, "torchvision_analyzer_rgb_digest_mismatch")
        paths.append(path)
        view_ids.append(view_id)
    if not paths or len(paths) != len(declared_by_id):
        return _abstention(request_digest, "torchvision_analyzer_view_set_incomplete")

    task_context = request.get("task_context")
    if not isinstance(task_context, Mapping):
        task_context = {}
    allowed_labels = _requested_labels(task_context, catalog)
    if not allowed_labels:
        return _abstention(request_digest, "torchvision_analyzer_no_allowed_task_labels")
    try:
        inference = detector or _torchvision_detector(weights_path)
        detections = inference(paths)
        proposals = _build_proposals(
            detections=detections,
            view_ids=view_ids,
            catalog=catalog,
            allowed_labels=allowed_labels,
            score_threshold=score_threshold,
            maximum_proposals=maximum_proposals,
        )
    except Exception:  # noqa: BLE001 - model/runtime failures must abstain safely
        return _abstention(request_digest, "torchvision_analyzer_inference_failed")
    if not proposals:
        return _abstention(
            request_digest,
            "torchvision_analyzer_no_taskable_detections_above_threshold",
        )
    return {
        "status": "completed",
        "analyzer_request_digest": request_digest,
        "candidate_may_self_authorize": False,
        "proposals": proposals,
        "blockers": [],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", help="Pinned local TorchVision checkpoint")
    parser.add_argument("--print-contract", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=0.15)
    parser.add_argument("--maximum-proposals", type=int, default=12)
    args = parser.parse_args(argv)
    if args.print_contract:
        print(
            json.dumps(
                build_analyzer_contract(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return 0
    if not args.weights:
        parser.error("--weights is required unless --print-contract is used")
    if not 0.0 <= args.score_threshold <= 1.0 or args.maximum_proposals <= 0:
        parser.error("analyzer limits are invalid")
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, Mapping):
        payload = {}
    result = analyze_payload(
        payload,
        weights_path=Path(args.weights).resolve(),
        score_threshold=args.score_threshold,
        maximum_proposals=args.maximum_proposals,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MODEL_ID",
    "MODEL_IMPLEMENTATION_VERSION",
    "MODEL_WEIGHT_SHA256",
    "analyze_payload",
    "build_analyzer_contract",
]
