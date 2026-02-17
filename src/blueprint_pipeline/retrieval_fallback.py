"""Retrieval fallback for required articulated swap candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .blueprintpipeline_runner import BlueprintPipelineRunner
from .common import StageError, utc_now_iso


@dataclass(frozen=True)
class FallbackResult:
    object_id: str
    resolved: bool
    reason: str
    asset_dir: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "resolved": self.resolved,
            "reason": self.reason,
            "asset_dir": self.asset_dir,
        }


def _candidate_to_adapter_object(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    object_id = str(candidate["object_id"])
    asset_dir = str(candidate.get("asset_dir") or f"obj_{object_id}")
    label = str(candidate.get("label") or "object")
    obb = candidate.get("obb") if isinstance(candidate.get("obb"), Mapping) else {}
    center = obb.get("center") if isinstance(obb.get("center"), list) else [0.0, 0.0, 0.0]

    obj: Dict[str, Any] = {
        "id": asset_dir,
        "name": label,
        "category": label,
        "description": f"Retrieval fallback asset for {label}",
        "sim_role": str(candidate.get("sim_role") or "articulated_furniture"),
        "asset_strategy": "retrieved",
        "transform": {
            "position": {
                "x": float(center[0]) if len(center) > 0 else 0.0,
                "y": float(center[1]) if len(center) > 1 else 0.0,
                "z": float(center[2]) if len(center) > 2 else 0.0,
            },
            "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
        },
        "dimensions_est": dict(candidate.get("dimensions_est") or {}),
        "physics_hints": dict(candidate.get("physics_hints") or {}),
        "articulation": {
            "required": True,
            "backend_hint": "retrieval_primary",
            "requirement_source": "fallback",
            "candidate": True,
        },
        "source": {
            "capture_object_id": object_id,
            "source_pipeline": "capture-nurec-swap",
        },
    }

    # Include reference image for image-conditioned retrieval
    ref_crop = candidate.get("reference_crop")
    if ref_crop:
        obj["reference_image"] = str(ref_crop)

    return obj


def _iter_text_files(asset_dir: Path) -> Iterable[Path]:
    for path in sorted(asset_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".usd", ".usda", ".urdf", ".xml", ".txt", ".json"}:
            yield path


def _detect_articulation(asset_dir: Path) -> bool:
    patterns = [
        "physicsrevolutejoint",
        "physicsprismaticjoint",
        "articulationroot",
        "def physicsrevolutejoint",
        "def physicsprismaticjoint",
        "<joint",
        'type="revolute"',
        'type="prismatic"',
    ]
    for path in _iter_text_files(asset_dir):
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if any(pattern in text for pattern in patterns):
            return True
    return False


def run_retrieval_fallback(
    *,
    runner: BlueprintPipelineRunner,
    storage_root: Path,
    scene_id: str,
    assets_prefix: str,
    room_type: str,
    failed_candidates: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Run retrieval-only fallback and validate articulation outputs."""

    if not failed_candidates:
        return {
            "schema_version": "v1",
            "scene_id": scene_id,
            "policy": "catalog_first",
            "generated_at": utc_now_iso(),
            "results": [],
            "resolved_ids": [],
            "unresolved_ids": [],
        }

    adapter_objects = [_candidate_to_adapter_object(candidate) for candidate in failed_candidates]
    runner.materialize_text_assets(
        scene_id=scene_id,
        assets_prefix=assets_prefix,
        objects=adapter_objects,
        room_type=room_type,
        generation_enabled=False,
        retrieval_enabled=True,
        retrieval_mode="ann_primary",
        generation_provider_chain="sam3d,hunyuan3d",
    )

    results: List[FallbackResult] = []
    resolved_ids: List[str] = []
    unresolved_ids: List[str] = []

    for candidate in failed_candidates:
        object_id = str(candidate["object_id"])
        asset_dir_name = str(candidate.get("asset_dir") or f"obj_{object_id}")
        asset_dir = storage_root / assets_prefix / asset_dir_name

        if not asset_dir.is_dir():
            results.append(
                FallbackResult(
                    object_id=object_id,
                    resolved=False,
                    reason="missing_asset_dir",
                    asset_dir=f"{assets_prefix}/{asset_dir_name}",
                )
            )
            unresolved_ids.append(object_id)
            continue

        articulated = _detect_articulation(asset_dir)
        if articulated:
            resolved_ids.append(object_id)
            results.append(
                FallbackResult(
                    object_id=object_id,
                    resolved=True,
                    reason="retrieval_articulation_detected",
                    asset_dir=f"{assets_prefix}/{asset_dir_name}",
                )
            )
        else:
            unresolved_ids.append(object_id)
            results.append(
                FallbackResult(
                    object_id=object_id,
                    resolved=False,
                    reason="retrieved_asset_not_articulated",
                    asset_dir=f"{assets_prefix}/{asset_dir_name}",
                )
            )

    payload = {
        "schema_version": "v1",
        "scene_id": scene_id,
        "policy": "catalog_first",
        "generated_at": utc_now_iso(),
        "results": [item.to_dict() for item in results],
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
    }
    return payload


def enforce_hard_fail_if_unresolved(payload: Mapping[str, Any]) -> None:
    unresolved_ids = payload.get("unresolved_ids") if isinstance(payload.get("unresolved_ids"), list) else []
    if unresolved_ids:
        joined = ", ".join(str(item) for item in unresolved_ids)
        raise StageError("retrieval_fallback", f"unresolved required articulation IDs: {joined}")
