"""SAM3D-first asset materialization and NuRec scene-shell helpers."""

from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .blueprintpipeline_runner import BlueprintPipelineRunner
from .common import StageError, ensure_dir, has_nonempty_file, utc_now_iso, write_json, write_text
from .reference_image_utils import cleanup_crop_with_vlm, find_best_reference_image


@dataclass(frozen=True)
class MaterializedAsset:
    object_id: str
    asset_dir: str
    status: str
    source_kind: str
    model_path: str
    mesh_glb_path: str
    metadata_path: str
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "asset_dir": self.asset_dir,
            "status": self.status,
            "source_kind": self.source_kind,
            "model_path": self.model_path,
            "mesh_glb_path": self.mesh_glb_path,
            "metadata_path": self.metadata_path,
            "reason": self.reason,
        }


def _write_reference_model_usd(path: Path, reference_rel_path: str) -> None:
    ref = reference_rel_path.replace("\\", "/")
    payload = f'''#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root" (
    prepend references = @{ref}@
)
{{
}}
'''
    write_text(path, payload)


def _candidate_to_adapter_object(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    object_id = str(candidate["object_id"])
    asset_dir = str(candidate.get("asset_dir") or f"obj_{object_id}")
    label = str(candidate.get("label") or "object")
    obb = candidate.get("obb") if isinstance(candidate.get("obb"), Mapping) else {}
    center = obb.get("center") if isinstance(obb.get("center"), list) else [0.0, 0.0, 0.0]
    quat = (
        obb.get("orientationQuaternion")
        if isinstance(obb.get("orientationQuaternion"), list)
        else [1.0, 0.0, 0.0, 0.0]
    )

    role = str(candidate.get("sim_role") or "manipulable_object")
    articulation = candidate.get("articulation") if isinstance(candidate.get("articulation"), Mapping) else {}

    obj: Dict[str, Any] = {
        "id": asset_dir,
        "name": label,
        "category": label,
        "description": f"Swappable asset for {label}",
        "sim_role": role,
        "asset_strategy": "generated",
        "transform": {
            "position": {
                "x": float(center[0]) if len(center) > 0 else 0.0,
                "y": float(center[1]) if len(center) > 1 else 0.0,
                "z": float(center[2]) if len(center) > 2 else 0.0,
            },
            "rotation_quaternion": {
                "w": float(quat[0]) if len(quat) > 0 else 1.0,
                "x": float(quat[1]) if len(quat) > 1 else 0.0,
                "y": float(quat[2]) if len(quat) > 2 else 0.0,
                "z": float(quat[3]) if len(quat) > 3 else 0.0,
            },
            "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
        },
        "dimensions_est": dict(candidate.get("dimensions_est") or {}),
        "physics_hints": dict(candidate.get("physics_hints") or {}),
        "articulation": {
            "required": bool(articulation.get("required", False)),
            "backend_hint": "particulate_first" if bool(articulation.get("required", False)) else "none",
            "requirement_source": str(articulation.get("requirement_source") or "policy"),
            "candidate": True,
        },
        "source": {
            "capture_object_id": object_id,
            "source_pipeline": "capture-nurec-swap",
        },
    }

    # Include reference image for image-conditioned generation
    ref_crop = candidate.get("reference_crop")
    if ref_crop:
        obj["reference_image"] = str(ref_crop)
    all_crops = candidate.get("all_crops")
    if isinstance(all_crops, list) and all_crops:
        obj["reference_images"] = [str(c) for c in all_crops if c]

    return obj


def _discover_glb_file(asset_dir: Path) -> Optional[Path]:
    preferred = [
        asset_dir / "mesh.glb",
        asset_dir / "model.glb",
        asset_dir / "part.glb",
    ]
    for path in preferred:
        if path.is_file():
            return path
    glbs = sorted(asset_dir.rglob("*.glb"))
    return glbs[0] if glbs else None


def _ensure_mesh_glb(asset_dir: Path) -> Path:
    mesh_glb = asset_dir / "mesh.glb"
    if mesh_glb.is_file() and mesh_glb.stat().st_size > 0:
        return mesh_glb

    discovered = _discover_glb_file(asset_dir)
    if discovered is None:
        raise StageError("sam3d", f"No GLB file found for asset at {asset_dir}")
    if discovered != mesh_glb:
        shutil.copy2(discovered, mesh_glb)
    return mesh_glb


def _build_materialized_records(
    *,
    storage_root: Path,
    assets_prefix: str,
    candidates: Iterable[Mapping[str, Any]],
    provenance_assets: List[Mapping[str, Any]],
) -> List[MaterializedAsset]:
    provenance_by_asset_dir: Dict[str, Mapping[str, Any]] = {}
    for item in provenance_assets:
        path = str(item.get("path") or "")
        if path:
            # path is assets_prefix/<asset_dir>/model.usd
            parts = path.split("/")
            if len(parts) >= 2:
                provenance_by_asset_dir[parts[-2]] = item

    out: List[MaterializedAsset] = []
    for candidate in candidates:
        object_id = str(candidate["object_id"])
        asset_dir_name = str(candidate.get("asset_dir") or f"obj_{object_id}")
        asset_dir = storage_root / assets_prefix / asset_dir_name
        model_path = asset_dir / "model.usd"
        metadata_path = asset_dir / "metadata.json"

        reason = ""
        source_kind = "unknown"
        status = "success"

        if not has_nonempty_file(model_path):
            status = "failed"
            reason = "missing_model_usd"
            mesh_glb = asset_dir / "mesh.glb"
        else:
            try:
                mesh_glb = _ensure_mesh_glb(asset_dir)
            except StageError as exc:
                status = "failed"
                reason = str(exc)
                mesh_glb = asset_dir / "mesh.glb"

        if not has_nonempty_file(metadata_path):
            if status == "success":
                status = "failed"
                reason = "missing_metadata"
        prov = provenance_by_asset_dir.get(asset_dir_name)
        if prov is not None:
            source_kind = str(prov.get("materialization") or prov.get("source") or "unknown")

        out.append(
            MaterializedAsset(
                object_id=object_id,
                asset_dir=f"{assets_prefix}/{asset_dir_name}",
                status=status,
                source_kind=source_kind,
                model_path=f"{assets_prefix}/{asset_dir_name}/model.usd",
                mesh_glb_path=f"{assets_prefix}/{asset_dir_name}/mesh.glb",
                metadata_path=f"{assets_prefix}/{asset_dir_name}/metadata.json",
                reason=reason,
            )
        )

    return out


def _stage_d_materialization_mode() -> str:
    mode = (os.getenv("STAGE_D_MATERIALIZATION_MODE") or "image_conditioned").strip().lower()
    if mode in {"image_conditioned", "adapter"}:
        return mode
    return "image_conditioned"


def _is_truthy(value: str, *, default: bool) -> bool:
    text = value.strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _env_any(*names: str) -> str:
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def _normalize_generation_provider(raw: str) -> str:
    value = (raw or "").strip().lower().replace("-", "_")
    if value in {"ttt", "tttlrm", "ttt_lrm"}:
        return "ttt_lrm"
    return value


def _parse_generation_provider_chain(raw_chain: str) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for token in (raw_chain or "").split(","):
        normalized = _normalize_generation_provider(token)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)

    return ordered or ["image_to_3d", "proxy_box"]


def _provider_image_to_3d_command(provider: str, *, generic_command: str) -> str:
    if provider == "sam3d":
        return _env_any(
            "STAGE_D_SAM3D_IMAGE_TO_3D_COMMAND",
            "SAM3D_IMAGE_TO_3D_COMMAND",
        ) or generic_command
    if provider == "hunyuan3d":
        return _env_any(
            "STAGE_D_HUNYUAN3D_IMAGE_TO_3D_COMMAND",
            "HUNYUAN3D_IMAGE_TO_3D_COMMAND",
        ) or generic_command
    if provider == "ttt_lrm":
        return _env_any(
            "STAGE_D_TTTLRM_IMAGE_TO_3D_COMMAND",
            "STAGE_D_TTT_LRM_IMAGE_TO_3D_COMMAND",
            "TTTLRM_IMAGE_TO_3D_COMMAND",
            "TTT_LRM_IMAGE_TO_3D_COMMAND",
        ) or generic_command
    if provider == "image_to_3d":
        return generic_command
    return ""


def _coerce_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_mesh_extents(candidate: Mapping[str, Any]) -> tuple[float, float, float]:
    obb = candidate.get("obb") if isinstance(candidate.get("obb"), Mapping) else {}
    extents = obb.get("extents") if isinstance(obb.get("extents"), list) else []
    if len(extents) >= 3:
        ex = max(0.02, min(8.0, _coerce_float(extents[0], default=0.35)))
        ey = max(0.02, min(8.0, _coerce_float(extents[1], default=0.35)))
        ez = max(0.02, min(8.0, _coerce_float(extents[2], default=0.35)))
        return ex, ey, ez

    dims = candidate.get("dimensions_est") if isinstance(candidate.get("dimensions_est"), Mapping) else {}
    width = max(0.02, min(8.0, _coerce_float(dims.get("width"), default=0.35)))
    height = max(0.02, min(8.0, _coerce_float(dims.get("height"), default=0.35)))
    depth = max(0.02, min(8.0, _coerce_float(dims.get("depth"), default=0.35)))
    return width, height, depth


def _write_proxy_mesh_glb(candidate: Mapping[str, Any], glb_path: Path) -> None:
    try:
        import trimesh  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency-dependent
        raise StageError("sam3d", f"trimesh is required for proxy mesh generation: {exc}") from exc

    extents = _candidate_mesh_extents(candidate)
    mesh = trimesh.creation.box(extents=extents)
    ensure_dir(glb_path.parent)
    mesh.export(glb_path)


def _run_image_to_3d_command(
    *,
    command_template: str,
    reference_image: Path,
    reference_images: List[Path],
    output_glb: Path,
    output_dir: Path,
    scene_id: str,
    object_id: str,
    asset_dir_name: str,
    room_type: str,
    timeout_seconds: int,
) -> tuple[bool, str, Dict[str, Any]]:
    substitutions = {
        "REFERENCE_IMAGE": str(reference_image),
        "INPUT_IMAGE": str(reference_image),
        "REFERENCE_IMAGES": ",".join(str(path) for path in reference_images),
        "INPUT_IMAGES": ",".join(str(path) for path in reference_images),
        "REFERENCE_IMAGES_JSON": json.dumps([str(path) for path in reference_images]),
        "NUM_REFERENCE_IMAGES": str(len(reference_images)),
        "OUTPUT_GLB": str(output_glb),
        "OUTPUT_DIR": str(output_dir),
        "ASSET_DIR": str(output_dir),
        "ASSET_ID": asset_dir_name,
        "OBJECT_ID": object_id,
        "SCENE_ID": scene_id,
        "ROOM_TYPE": room_type,
    }
    for idx, path in enumerate(reference_images[:8], 1):
        substitutions[f"REFERENCE_IMAGE_{idx}"] = str(path)
        substitutions[f"INPUT_IMAGE_{idx}"] = str(path)
    rendered = command_template
    for key, value in substitutions.items():
        rendered = rendered.replace("{" + key + "}", value)

    try:
        command = shlex.split(rendered)
    except ValueError as exc:
        return False, f"invalid STAGE_D_IMAGE_TO_3D_COMMAND: {exc}", {"rendered_command": rendered}

    if not command:
        return False, "empty STAGE_D_IMAGE_TO_3D_COMMAND", {"rendered_command": rendered}

    try:
        proc = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return (
            False,
            f"image-to-3d command timed out after {timeout_seconds}s",
            {"command": command, "timeout_seconds": timeout_seconds},
        )
    except Exception as exc:  # pragma: no cover - subprocess edge
        return False, f"failed to execute image-to-3d command: {exc}", {"command": command}

    invocation = {
        "command": command,
        "return_code": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "timeout_seconds": timeout_seconds,
    }
    if proc.returncode != 0:
        return False, f"image-to-3d command failed with code {proc.returncode}", invocation
    if not has_nonempty_file(output_glb):
        return False, f"image-to-3d command completed but missing output GLB at {output_glb}", invocation
    return True, "ok", invocation


def _candidate_reference_image_paths(
    candidate: Mapping[str, Any],
    *,
    assets_root: Path,
    max_candidates: int,
) -> List[Path]:
    seen: set[str] = set()
    out: List[Path] = []

    def _push(value: Any) -> None:
        text = str(value or "").strip()
        if not text:
            return
        path = Path(text)
        if path.is_file():
            resolved = str(path.resolve())
            if resolved not in seen:
                seen.add(resolved)
                out.append(Path(resolved))

    _push(candidate.get("reference_crop"))
    raw_all_crops = candidate.get("all_crops")
    if isinstance(raw_all_crops, list):
        for crop in raw_all_crops:
            _push(crop)
            if len(out) >= max_candidates:
                return out

    raw_refs = candidate.get("reference_images")
    if isinstance(raw_refs, list):
        for crop in raw_refs:
            _push(crop)
            if len(out) >= max_candidates:
                return out

    asset_dir_name = str(candidate.get("asset_dir") or "")
    if asset_dir_name:
        for ext in ("png", "jpg", "jpeg"):
            _push(assets_root / asset_dir_name / f"reference.{ext}")
            if len(out) >= max_candidates:
                return out

    fallback = find_best_reference_image(dict(candidate), storage_root=assets_root)
    if fallback:
        _push(fallback)

    return out[:max_candidates]


def _score_reference_image(path: Path) -> Dict[str, Any]:
    try:
        from PIL import Image, ImageFilter, ImageStat  # type: ignore
    except Exception:
        return {
            "path": str(path),
            "score": 0.0,
            "sharpness_raw": 0.0,
            "coverage": 0.0,
            "mask_quality": 0.0,
        }

    try:
        with Image.open(path) as img:
            image = img.convert("RGBA")
            width, height = image.size
            area = max(1, width * height)

            # Sharpness proxy from edge variance on grayscale.
            gray = image.convert("L")
            edges = gray.filter(ImageFilter.FIND_EDGES)
            sharpness_raw = float(ImageStat.Stat(edges).var[0] or 0.0)

            alpha = image.getchannel("A")
            alpha_data = alpha.tobytes()
            nonzero = sum(1 for value in alpha_data if value > 16)
            coverage = float(nonzero) / float(area)

            if nonzero > 0:
                x0, y0, x1, y1 = alpha.getbbox() or (0, 0, width, height)
                bbox_w = max(1, x1 - x0)
                bbox_h = max(1, y1 - y0)
                bbox_area = max(1, bbox_w * bbox_h)
                fill_ratio = min(1.0, float(nonzero) / float(bbox_area))
                touches_border = int(x0 <= 0 or y0 <= 0 or x1 >= width or y1 >= height)
                border_penalty = 0.35 if touches_border else 0.0
                mask_quality = max(0.0, min(1.0, fill_ratio - border_penalty))
            else:
                mask_quality = 0.0

            return {
                "path": str(path),
                "score": 0.0,
                "sharpness_raw": max(0.0, sharpness_raw),
                "coverage": max(0.0, min(1.0, coverage)),
                "mask_quality": max(0.0, min(1.0, mask_quality)),
            }
    except Exception:
        return {
            "path": str(path),
            "score": 0.0,
            "sharpness_raw": 0.0,
            "coverage": 0.0,
            "mask_quality": 0.0,
        }


def _rank_reference_images(paths: List[Path], *, top_k: int) -> tuple[List[Path], List[Dict[str, Any]]]:
    if not paths:
        return [], []

    rows = [_score_reference_image(path) for path in paths]
    max_sharp = max((float(row.get("sharpness_raw") or 0.0) for row in rows), default=0.0)
    sharp_norm_div = max(1e-9, max_sharp)

    for row in rows:
        sharp_norm = math.sqrt(max(0.0, float(row.get("sharpness_raw") or 0.0)) / sharp_norm_div)
        coverage = max(0.0, min(1.0, float(row.get("coverage") or 0.0)))
        mask_quality = max(0.0, min(1.0, float(row.get("mask_quality") or 0.0)))
        score = (0.5 * sharp_norm) + (0.3 * coverage) + (0.2 * mask_quality)
        row["sharpness_norm"] = sharp_norm
        row["score"] = score

    rows.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
    selected_paths = [Path(str(row["path"])) for row in rows[: max(1, top_k)]]
    return selected_paths, rows


def _materialize_with_adapter(
    *,
    runner: BlueprintPipelineRunner,
    storage_root: Path,
    scene_id: str,
    assets_prefix: str,
    room_type: str,
    swap_candidates: List[Mapping[str, Any]],
    generation_provider_chain: str,
) -> Dict[str, Any]:
    adapter_objects = [_candidate_to_adapter_object(candidate) for candidate in swap_candidates]

    for candidate in swap_candidates:
        ref_crop = candidate.get("reference_crop")
        if ref_crop:
            ref_src = Path(str(ref_crop))
            if ref_src.is_file():
                object_id = str(candidate["object_id"])
                asset_dir_name = str(candidate.get("asset_dir") or f"obj_{object_id}")
                dest_dir = storage_root / assets_prefix / asset_dir_name
                ensure_dir(dest_dir)
                shutil.copy2(ref_src, dest_dir / "reference.png")

    return runner.materialize_text_assets(
        scene_id=scene_id,
        assets_prefix=assets_prefix,
        objects=adapter_objects,
        room_type=room_type,
        generation_enabled=True,
        retrieval_enabled=False,
        retrieval_mode="ann_shadow",
        generation_provider_chain=generation_provider_chain,
    )


def _materialize_with_image_conditioned_pipeline(
    *,
    runner: BlueprintPipelineRunner,
    storage_root: Path,
    scene_id: str,
    assets_prefix: str,
    room_type: str,
    swap_candidates: List[Mapping[str, Any]],
    generation_provider_chain: str,
) -> Dict[str, Any]:
    assets_root = storage_root / assets_prefix
    ensure_dir(assets_root)

    cleanup_provider = (
        os.getenv("STAGE_D_IMAGE_CLEANUP_PROVIDER")
        or os.getenv("CROP_CLEANUP_PROVIDER")
        or "skip"
    ).strip().lower()
    image_to_3d_command = (
        os.getenv("STAGE_D_IMAGE_TO_3D_COMMAND")
        or os.getenv("IMAGE_TO_3D_COMMAND")
        or ""
    ).strip()
    provider_order = _parse_generation_provider_chain(generation_provider_chain)
    top_k_references = _int_env("STAGE_D_IMAGE_TO_3D_TOPK", 3)
    max_reference_candidates = _int_env("STAGE_D_REFERENCE_MAX_CROPS", 12)
    cleanup_top_k = _int_env("STAGE_D_CLEANUP_TOPK", 1)
    timeout_seconds = _int_env("STAGE_D_IMAGE_TO_3D_TIMEOUT_SECONDS", 900)
    allow_proxy_fallback = _is_truthy(
        os.getenv("STAGE_D_ALLOW_PROXY_FALLBACK", "true"),
        default=True,
    )

    provenance_assets: List[Dict[str, Any]] = []
    method_counts: Dict[str, int] = {
        "articulated_retrieval": 0,
        "image_to_3d": 0,
        "proxy_box": 0,
        "failed": 0,
    }
    articulated_candidates = []
    non_articulated_candidates = []
    for candidate in swap_candidates:
        articulation = candidate.get("articulation") if isinstance(candidate.get("articulation"), Mapping) else {}
        if bool(articulation.get("required", False)):
            articulated_candidates.append(candidate)
        else:
            non_articulated_candidates.append(candidate)

    articulated_error = ""
    if articulated_candidates:
        adapter_objects = [_candidate_to_adapter_object(candidate) for candidate in articulated_candidates]
        try:
            runner.materialize_text_assets(
                scene_id=scene_id,
                assets_prefix=assets_prefix,
                objects=adapter_objects,
                room_type=room_type,
                generation_enabled=False,
                retrieval_enabled=True,
                retrieval_mode="ann_primary",
                generation_provider_chain=generation_provider_chain,
            )
        except Exception as exc:  # pragma: no cover - adapter/runtime dependent
            articulated_error = str(exc)
            method_counts["failed"] += len(articulated_candidates)

    for candidate in articulated_candidates:
        object_id = str(candidate["object_id"])
        asset_dir_name = str(candidate.get("asset_dir") or f"obj_{object_id}")
        asset_dir = assets_root / asset_dir_name
        ensure_dir(asset_dir)

        reference_image_text = find_best_reference_image(dict(candidate), storage_root=assets_root)
        original_reference_path = Path(reference_image_text) if reference_image_text else None
        if original_reference_path and original_reference_path.is_file():
            reference_copy = asset_dir / "reference.png"
            if original_reference_path != reference_copy:
                shutil.copy2(original_reference_path, reference_copy)

        source_kind = "articulated_retrieval"
        mesh_glb_path = asset_dir / "mesh.glb"

        if not has_nonempty_file(mesh_glb_path):
            discovered = _discover_glb_file(asset_dir)
            if discovered is not None and discovered != mesh_glb_path:
                shutil.copy2(discovered, mesh_glb_path)

        if not has_nonempty_file(mesh_glb_path):
            if not allow_proxy_fallback:
                raise StageError(
                    "sam3d",
                    "articulated retrieval produced no mesh.glb and STAGE_D_ALLOW_PROXY_FALLBACK=false",
                )
            _write_proxy_mesh_glb(candidate, mesh_glb_path)
            source_kind = "articulated_retrieval_proxy_box"
            method_counts["proxy_box"] += 1
        else:
            method_counts["articulated_retrieval"] += 1

        model_path = asset_dir / "model.usd"
        if not has_nonempty_file(model_path):
            _write_reference_model_usd(model_path, "./mesh.glb")

        metadata_path = asset_dir / "metadata.json"
        metadata_payload: Dict[str, Any] = {
            "schema_version": "v1",
            "scene_id": scene_id,
            "object_id": object_id,
            "asset_dir": f"{assets_prefix}/{asset_dir_name}",
            "source_kind": source_kind,
            "router_branch": "articulated_required",
            "articulation_required": True,
            "status": "success",
            "room_type": room_type,
            "generation_provider_chain": generation_provider_chain,
            "reference_image": str(asset_dir / "reference.png") if (asset_dir / "reference.png").is_file() else "",
            "mesh_glb_path": f"{assets_prefix}/{asset_dir_name}/mesh.glb",
        }
        if articulated_error:
            metadata_payload["articulated_retrieval_error"] = articulated_error
        write_json(metadata_path, metadata_payload)

        provenance_assets.append(
            {
                "object_id": asset_dir_name,
                "path": f"{assets_prefix}/{asset_dir_name}/model.usd",
                "materialization": source_kind,
            }
        )

    for candidate in non_articulated_candidates:
        object_id = str(candidate["object_id"])
        asset_dir_name = str(candidate.get("asset_dir") or f"obj_{object_id}")
        asset_dir = assets_root / asset_dir_name
        ensure_dir(asset_dir)

        raw_reference_paths = _candidate_reference_image_paths(
            candidate,
            assets_root=assets_root,
            max_candidates=max(1, max_reference_candidates),
        )
        ranked_reference_paths, ranked_reference_metrics = _rank_reference_images(
            raw_reference_paths,
            top_k=max(1, top_k_references),
        )
        working_reference_paths: List[Path] = []
        for idx, path in enumerate(ranked_reference_paths):
            ref_name = "reference.png" if idx == 0 else f"reference_{idx + 1}.png"
            copied = asset_dir / ref_name
            if path != copied:
                shutil.copy2(path, copied)
            else:
                copied = path

            working = copied
            if cleanup_provider != "skip" and idx < max(0, cleanup_top_k):
                cleaned_name = "reference_clean.png" if idx == 0 else f"reference_{idx + 1}_clean.png"
                cleaned_path = cleanup_crop_with_vlm(
                    working,
                    asset_dir / cleaned_name,
                    provider=cleanup_provider,
                )
                if cleaned_path is not None and Path(cleaned_path).is_file():
                    working = Path(cleaned_path)
            working_reference_paths.append(working)

        original_reference_path = raw_reference_paths[0] if raw_reference_paths else None
        working_reference_path: Optional[Path] = (
            working_reference_paths[0] if working_reference_paths else None
        )

        mesh_glb_path = asset_dir / "mesh.glb"
        source_kind = ""
        command_error = ""
        image_to_3d_invocation: Dict[str, Any] = {}
        provider_attempts: List[Dict[str, Any]] = []
        selected_provider = ""
        tried_image_to_3d = False

        if working_reference_path and working_reference_path.is_file():
            for provider in provider_order:
                if provider == "proxy_box":
                    continue

                provider_command = _provider_image_to_3d_command(
                    provider,
                    generic_command=image_to_3d_command,
                )
                if not provider_command:
                    provider_attempts.append(
                        {
                            "provider": provider,
                            "status": "skipped_unconfigured",
                            "detail": "no command template configured",
                        }
                    )
                    continue

                tried_image_to_3d = True
                rendered_command = provider_command.replace("{PROVIDER}", provider)
                ok, detail, invocation = _run_image_to_3d_command(
                    command_template=rendered_command,
                    reference_image=working_reference_path,
                    reference_images=working_reference_paths or [working_reference_path],
                    output_glb=mesh_glb_path,
                    output_dir=asset_dir,
                    scene_id=scene_id,
                    object_id=object_id,
                    asset_dir_name=asset_dir_name,
                    room_type=room_type,
                    timeout_seconds=timeout_seconds,
                )
                invocation = dict(invocation)
                invocation["provider"] = provider
                image_to_3d_invocation = invocation
                provider_attempts.append(
                    {
                        "provider": provider,
                        "status": "success" if ok else "failed",
                        "detail": detail,
                        "return_code": invocation.get("return_code"),
                    }
                )
                if ok:
                    selected_provider = provider
                    source_kind = (
                        "image_to_3d"
                        if provider == "image_to_3d"
                        else f"image_to_3d_{provider}"
                    )
                    method_counts["image_to_3d"] += 1
                    provider_metric = f"image_to_3d_{provider}"
                    method_counts[provider_metric] = int(method_counts.get(provider_metric, 0)) + 1
                    command_error = ""
                    break

                command_error = f"{provider}: {detail}"
                method_counts["failed"] += 1

        if not has_nonempty_file(mesh_glb_path):
            if not allow_proxy_fallback:
                error_detail = (
                    "image-conditioned generation produced no mesh.glb and "
                    "STAGE_D_ALLOW_PROXY_FALLBACK=false"
                )
                if command_error:
                    error_detail = f"{error_detail}; last_error={command_error}"
                raise StageError(
                    "sam3d",
                    error_detail,
                )
            _write_proxy_mesh_glb(candidate, mesh_glb_path)
            source_kind = source_kind or "image_conditioned_proxy_box"
            method_counts["proxy_box"] += 1

        model_path = asset_dir / "model.usd"
        _write_reference_model_usd(model_path, "./mesh.glb")

        metadata_path = asset_dir / "metadata.json"
        metadata_payload = {
            "schema_version": "v1",
            "scene_id": scene_id,
            "object_id": object_id,
            "asset_dir": f"{assets_prefix}/{asset_dir_name}",
            "source_kind": source_kind or "image_conditioned_proxy_box",
            "router_branch": "non_articulated",
            "articulation_required": False,
            "status": "success",
            "room_type": room_type,
            "generation_provider_chain": generation_provider_chain,
            "image_to_3d_provider_chain": provider_order,
            "image_to_3d_selected_provider": selected_provider,
            "image_to_3d_provider_attempts": provider_attempts,
            "cleanup_provider": cleanup_provider,
            "reference_image_original": str(original_reference_path) if original_reference_path else "",
            "reference_image": str(working_reference_path) if working_reference_path else "",
            "reference_images_ranked": ranked_reference_metrics,
            "reference_images_selected": [str(path) for path in working_reference_paths],
            "image_to_3d_reference_count": len(working_reference_paths),
            "image_to_3d_attempted": tried_image_to_3d,
            "mesh_glb_path": f"{assets_prefix}/{asset_dir_name}/mesh.glb",
        }
        if image_to_3d_invocation:
            metadata_payload["image_to_3d_invocation"] = image_to_3d_invocation
        if command_error:
            metadata_payload["image_to_3d_error"] = command_error
        write_json(metadata_path, metadata_payload)

        provenance_assets.append(
            {
                "object_id": asset_dir_name,
                "path": f"{assets_prefix}/{asset_dir_name}/model.usd",
                "materialization": source_kind or "image_conditioned_proxy_box",
            }
        )

    return {
        "provenance_assets": provenance_assets,
        "retrieval_report": {
            "mode": "image_conditioned",
            "method_counts": method_counts,
        },
    }


def materialize_candidate_assets(
    *,
    runner: BlueprintPipelineRunner,
    storage_root: Path,
    scene_id: str,
    assets_prefix: str,
    room_type: str,
    swap_candidates: List[Mapping[str, Any]],
    generation_provider_chain: str = "sam3d,hunyuan3d",
) -> Dict[str, Any]:
    """Materialize swappable objects with SAM3D-first policy."""
    mode = _stage_d_materialization_mode()
    if mode == "adapter":
        result = _materialize_with_adapter(
            runner=runner,
            storage_root=storage_root,
            scene_id=scene_id,
            assets_prefix=assets_prefix,
            room_type=room_type,
            swap_candidates=swap_candidates,
            generation_provider_chain=generation_provider_chain,
        )
    else:
        result = _materialize_with_image_conditioned_pipeline(
            runner=runner,
            storage_root=storage_root,
            scene_id=scene_id,
            assets_prefix=assets_prefix,
            room_type=room_type,
            swap_candidates=swap_candidates,
            generation_provider_chain=generation_provider_chain,
        )

    records = _build_materialized_records(
        storage_root=storage_root,
        assets_prefix=assets_prefix,
        candidates=swap_candidates,
        provenance_assets=[
            dict(item) for item in result.get("provenance_assets", []) if isinstance(item, Mapping)
        ],
    )

    return {
        "schema_version": "v1",
        "scene_id": scene_id,
        "policy": "sam3d_first",
        "generated_at": utc_now_iso(),
        "records": [record.to_dict() for record in records],
        "retrieval_report": result.get("retrieval_report", {}),
    }


def _ply_to_glb(ply_path: Path, glb_path: Path) -> None:
    try:
        import trimesh  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency-dependent
        raise StageError("sam3d", f"trimesh is required for PLY->GLB conversion: {exc}") from exc

    mesh = trimesh.load_mesh(str(ply_path))
    if mesh is None:
        raise StageError("sam3d", f"Failed to load mesh from {ply_path}")
    ensure_dir(glb_path.parent)
    mesh.export(glb_path)


def _obb_mask_keep_faces(mesh, swap_candidates: List[Mapping[str, Any]]):
    try:
        import numpy as np  # type: ignore
    except Exception:
        return None

    if not hasattr(mesh, "faces") or not hasattr(mesh, "triangles_center"):
        return None

    centers = mesh.triangles_center
    if centers is None:
        return None

    keep = np.ones(len(centers), dtype=bool)
    for candidate in swap_candidates:
        obb = candidate.get("obb") if isinstance(candidate.get("obb"), Mapping) else {}
        center = obb.get("center") if isinstance(obb.get("center"), list) else [0.0, 0.0, 0.0]
        extents = obb.get("extents") if isinstance(obb.get("extents"), list) else [0.0, 0.0, 0.0]
        axes = obb.get("axes") if isinstance(obb.get("axes"), list) else None
        if len(center) < 3 or len(extents) < 3:
            continue
        center_vec = np.array(center[:3], dtype=float)
        half = np.array(extents[:3], dtype=float) * 0.5
        if axes and len(axes) >= 3:
            basis = np.array([axis[:3] for axis in axes[:3]], dtype=float)
            if basis.shape != (3, 3):
                continue
        else:
            basis = np.eye(3)

        local = (centers - center_vec) @ basis.T
        inside = (
            (abs(local[:, 0]) <= half[0])
            & (abs(local[:, 1]) <= half[1])
            & (abs(local[:, 2]) <= half[2])
        )
        keep &= ~inside

    return keep


def _prune_scene_shell_mesh(glb_path: Path, swap_candidates: List[Mapping[str, Any]]) -> Dict[str, Any]:
    if not swap_candidates:
        return {"enabled": False, "faces_removed": 0}
    try:
        import trimesh  # type: ignore
    except Exception:
        return {"enabled": False, "reason": "trimesh_unavailable", "faces_removed": 0}

    mesh = trimesh.load_mesh(str(glb_path))
    keep = _obb_mask_keep_faces(mesh, swap_candidates)
    if keep is None:
        return {"enabled": False, "reason": "mesh_not_prunable", "faces_removed": 0}

    removed = int((~keep).sum())
    if removed <= 0:
        return {"enabled": True, "faces_removed": 0}

    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    mesh.export(glb_path)
    return {"enabled": True, "faces_removed": removed}


def _mesh_face_count(mesh: Any) -> int:
    if hasattr(mesh, "faces"):
        try:
            return int(len(mesh.faces))
        except Exception:
            return 0
    return 0


def _simplify_scene_shell_mesh(glb_path: Path, max_faces: int) -> Dict[str, Any]:
    if max_faces <= 0:
        return {"enabled": False, "reason": "invalid_budget", "before_faces": 0, "after_faces": 0}
    try:
        import trimesh  # type: ignore
    except Exception:
        return {
            "enabled": False,
            "reason": "trimesh_unavailable",
            "before_faces": 0,
            "after_faces": 0,
            "budget_faces": max_faces,
        }

    mesh = trimesh.load_mesh(str(glb_path), process=False)
    if mesh is None:
        return {
            "enabled": False,
            "reason": "mesh_load_failed",
            "before_faces": 0,
            "after_faces": 0,
            "budget_faces": max_faces,
        }
    if hasattr(trimesh, "Scene") and isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        if not geometries:
            return {
                "enabled": False,
                "reason": "empty_scene_mesh",
                "before_faces": 0,
                "after_faces": 0,
                "budget_faces": max_faces,
            }
        mesh = trimesh.util.concatenate(geometries)

    before_faces = _mesh_face_count(mesh)
    if before_faces <= 0:
        return {
            "enabled": False,
            "reason": "no_faces",
            "before_faces": before_faces,
            "after_faces": before_faces,
            "budget_faces": max_faces,
        }
    if before_faces <= max_faces:
        return {
            "enabled": True,
            "method": "none",
            "before_faces": before_faces,
            "after_faces": before_faces,
            "budget_faces": max_faces,
        }

    reduction_fraction = max(0.01, min(0.99, 1.0 - (float(max_faces) / float(before_faces))))
    target_faces = max(4, min(before_faces - 1, int(round(before_faces * (1.0 - reduction_fraction)))))
    simplified = None
    method = "none"

    if hasattr(mesh, "simplify_quadric_decimation"):
        fn = mesh.simplify_quadric_decimation
        for call in (
            lambda: fn(percent=reduction_fraction),
            lambda: fn(face_count=target_faces),
            lambda: fn(reduction_fraction),
        ):
            try:
                simplified = call()
                method = "quadric_decimation"
                if simplified is not None:
                    break
            except Exception:
                continue

    if simplified is None and hasattr(mesh, "simplify_quadratic_decimation"):
        fn = mesh.simplify_quadratic_decimation
        for call in (
            lambda: fn(percent=reduction_fraction),
            lambda: fn(face_count=target_faces),
            lambda: fn(target_faces),
        ):
            try:
                simplified = call()
                method = "quadratic_decimation"
                if simplified is not None:
                    break
            except Exception:
                continue

    if simplified is None:
        try:
            import numpy as np  # type: ignore

            keep_idx = np.linspace(0, before_faces - 1, num=target_faces, dtype=int)
            simplified = mesh.submesh([keep_idx], append=True, repair=False)
            method = "face_subsample"
        except Exception:
            simplified = None

    if simplified is None:
        return {
            "enabled": False,
            "reason": "simplification_failed",
            "before_faces": before_faces,
            "after_faces": before_faces,
            "budget_faces": max_faces,
            "reduction_fraction": round(reduction_fraction, 6),
        }

    after_faces = _mesh_face_count(simplified)
    if after_faces <= 0:
        return {
            "enabled": False,
            "reason": "simplification_no_faces",
            "before_faces": before_faces,
            "after_faces": after_faces,
            "budget_faces": max_faces,
            "reduction_fraction": round(reduction_fraction, 6),
        }

    if after_faces > max_faces:
        try:
            import numpy as np  # type: ignore

            keep_idx = np.linspace(0, after_faces - 1, num=max_faces, dtype=int)
            simplified = simplified.submesh([keep_idx], append=True, repair=False)
            method = f"{method}_plus_subsample"
            after_faces = _mesh_face_count(simplified)
        except Exception:
            pass

    simplified.export(glb_path)
    return {
        "enabled": True,
        "method": method,
        "before_faces": before_faces,
        "after_faces": after_faces,
        "budget_faces": max_faces,
        "reduction_fraction": round(reduction_fraction, 6),
    }


def materialize_scene_shell_assets(
    *,
    storage_root: Path,
    assets_prefix: str,
    nurec_outputs: Mapping[str, Any],
    swap_candidates: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Materialize NuRec visual + collision shell assets."""

    artifacts = nurec_outputs.get("artifacts") if isinstance(nurec_outputs.get("artifacts"), Mapping) else {}
    visual_uri = str(artifacts.get("visual_usdz") or "").strip()
    visual_mesh_uri = str(artifacts.get("visual_mesh_glb") or "").strip()
    mesh_uri = str(artifacts.get("collision_mesh_ply") or "").strip()
    inpainted_mesh_active = False

    # Prefer inpainted visual mesh from Inpaint360GS scene cleaning (Stage 9.5)
    inpainted_glb_uri = str(artifacts.get("inpainted_visual_mesh_glb") or "").strip()
    if inpainted_glb_uri:
        try:
            from .common import resolve_gs_uri_to_path as _resolve
            inpainted_src = _resolve(inpainted_glb_uri, storage_root)
            if has_nonempty_file(inpainted_src):
                visual_mesh_uri = inpainted_glb_uri
                inpainted_mesh_active = True
                print(f"[sam3d_assets] Using inpainted visual mesh: {inpainted_src}", flush=True)
        except Exception:
            pass  # Fall through to original visual mesh
    if not mesh_uri or (not visual_uri and not visual_mesh_uri):
        raise StageError(
            "sam3d",
            "nurec_outputs missing collision_mesh_ply and at least one visual artifact "
            "(visual_mesh_glb or visual_usdz)",
        )

    from .common import resolve_gs_uri_to_path

    visual_src = resolve_gs_uri_to_path(visual_uri, storage_root) if visual_uri else None
    visual_mesh_src = resolve_gs_uri_to_path(visual_mesh_uri, storage_root) if visual_mesh_uri else None
    mesh_src = resolve_gs_uri_to_path(mesh_uri, storage_root)
    if visual_mesh_src is not None and not has_nonempty_file(visual_mesh_src):
        visual_mesh_src = None
    if visual_src is not None and not has_nonempty_file(visual_src):
        raise StageError("sam3d", f"missing NuRec visual USDZ at {visual_src}")
    if visual_mesh_src is None and visual_src is None:
        raise StageError("sam3d", "no valid visual source artifact found in nurec outputs")
    if not has_nonempty_file(mesh_src):
        raise StageError("sam3d", f"missing NuRec mesh PLY at {mesh_src}")

    visual_dir = storage_root / assets_prefix / "obj_nurec_visual"
    shell_dir = storage_root / assets_prefix / "obj_scene_shell"
    ensure_dir(visual_dir)
    ensure_dir(shell_dir)

    visual_glb = visual_dir / "model.glb"
    visual_usdz = visual_dir / "model.usdz"
    if visual_mesh_src is not None and has_nonempty_file(visual_mesh_src):
        shutil.copy2(visual_mesh_src, visual_glb)
    if visual_src is not None and has_nonempty_file(visual_src):
        shutil.copy2(visual_src, visual_usdz)

    visual_primary = (os.getenv("NUREC_VISUAL_PRIMARY") or "usdz").strip().lower()
    if visual_primary not in {"usdz", "mesh", "auto"}:
        visual_primary = "usdz"

    has_mesh = has_nonempty_file(visual_glb)
    has_volume = has_nonempty_file(visual_usdz)
    if not has_mesh and not has_volume:
        raise StageError("sam3d", "no valid visual source artifact found after materialization")

    selected_visual_source = "nurec_export_volume"
    if inpainted_mesh_active and has_mesh:
        # Cleaned mesh must be primary when available (USDZ remains as fallback).
        _write_reference_model_usd(visual_dir / "model.usd", "model.glb")
        selected_visual_source = "inpaint360gs_cleaned_mesh"
    elif visual_primary == "mesh":
        if has_mesh:
            _write_reference_model_usd(visual_dir / "model.usd", "model.glb")
            selected_visual_source = "nurec_visual_mesh"
        else:
            _write_reference_model_usd(visual_dir / "model.usd", "model.usdz")
    elif visual_primary == "auto":
        if has_mesh:
            _write_reference_model_usd(visual_dir / "model.usd", "model.glb")
            selected_visual_source = "nurec_visual_mesh"
        else:
            _write_reference_model_usd(visual_dir / "model.usd", "model.usdz")
    else:
        if has_volume:
            _write_reference_model_usd(visual_dir / "model.usd", "model.usdz")
            selected_visual_source = "nurec_export_volume"
        else:
            _write_reference_model_usd(visual_dir / "model.usd", "model.glb")
            selected_visual_source = "nurec_visual_mesh"

    shell_glb = shell_dir / "mesh.glb"
    _ply_to_glb(mesh_src, shell_glb)
    prune_report = _prune_scene_shell_mesh(shell_glb, swap_candidates)
    simplify_report = _simplify_scene_shell_mesh(
        shell_glb,
        _int_env("QUALITY_MAX_COLLISION_FACES", 500000),
    )
    _write_reference_model_usd(shell_dir / "model.usd", "mesh.glb")

    shell_metadata = {
        "schema_version": "v1",
        "asset_id": "obj_scene_shell",
        "source": "nurec_nvblox_mesh",
        "source_uri": mesh_uri,
        "pruning": prune_report,
        "simplification": simplify_report,
        "generated_at": utc_now_iso(),
    }
    write_json(shell_dir / "metadata.json", shell_metadata)

    visual_metadata = {
        "schema_version": "v1",
        "asset_id": "obj_nurec_visual",
        "source": selected_visual_source,
        "source_uri": (
            inpainted_glb_uri
            if selected_visual_source == "inpaint360gs_cleaned_mesh"
            else (visual_mesh_uri if selected_visual_source == "nurec_visual_mesh" else visual_uri)
        ),
        "fallback_visual_usdz_uri": visual_uri,
        "primary_visual_preference": visual_primary,
        "available_visual_assets": {
            "usdz": has_volume,
            "mesh_glb": has_mesh,
        },
        "inpainted_visual_mesh": bool(inpainted_mesh_active),
        "generated_at": utc_now_iso(),
    }
    write_json(visual_dir / "metadata.json", visual_metadata)

    return {
        "visual_asset": f"{assets_prefix}/obj_nurec_visual/model.usd",
        "shell_asset": f"{assets_prefix}/obj_scene_shell/model.usd",
        "shell_mesh": f"{assets_prefix}/obj_scene_shell/mesh.glb",
        "pruning": prune_report,
        "simplification": simplify_report,
    }


def write_swap_execution_report(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, payload)


def write_swap_quality_report(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, payload)


def write_completion_marker(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, payload)


def write_failure_marker(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, payload)
