"""SAM3D-first asset materialization and NuRec scene-shell helpers."""

from __future__ import annotations

import json
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
        "OUTPUT_GLB": str(output_glb),
        "OUTPUT_DIR": str(output_dir),
        "ASSET_DIR": str(output_dir),
        "ASSET_ID": asset_dir_name,
        "OBJECT_ID": object_id,
        "SCENE_ID": scene_id,
        "ROOM_TYPE": room_type,
    }
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
        or "qwen_image_edit"
    ).strip().lower()
    image_to_3d_command = (
        os.getenv("STAGE_D_IMAGE_TO_3D_COMMAND")
        or os.getenv("IMAGE_TO_3D_COMMAND")
        or ""
    ).strip()
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

        reference_image_text = find_best_reference_image(dict(candidate), storage_root=assets_root)
        original_reference_path = Path(reference_image_text) if reference_image_text else None
        working_reference_path: Optional[Path] = None

        if original_reference_path and original_reference_path.is_file():
            reference_copy = asset_dir / "reference.png"
            if original_reference_path != reference_copy:
                shutil.copy2(original_reference_path, reference_copy)
            else:
                reference_copy = original_reference_path

            working_reference_path = reference_copy
            if cleanup_provider != "skip":
                cleaned_path = cleanup_crop_with_vlm(
                    working_reference_path,
                    asset_dir / "reference_clean.png",
                    provider=cleanup_provider,
                )
                if cleaned_path is not None and Path(cleaned_path).is_file():
                    working_reference_path = Path(cleaned_path)

        mesh_glb_path = asset_dir / "mesh.glb"
        source_kind = ""
        command_error = ""
        image_to_3d_invocation: Dict[str, Any] = {}
        tried_image_to_3d = False

        if image_to_3d_command and working_reference_path and working_reference_path.is_file():
            tried_image_to_3d = True
            ok, detail, invocation = _run_image_to_3d_command(
                command_template=image_to_3d_command,
                reference_image=working_reference_path,
                output_glb=mesh_glb_path,
                output_dir=asset_dir,
                scene_id=scene_id,
                object_id=object_id,
                asset_dir_name=asset_dir_name,
                room_type=room_type,
                timeout_seconds=timeout_seconds,
            )
            image_to_3d_invocation = invocation
            if ok:
                source_kind = "image_to_3d"
                method_counts["image_to_3d"] += 1
            else:
                command_error = detail
                method_counts["failed"] += 1

        if not has_nonempty_file(mesh_glb_path):
            if not allow_proxy_fallback:
                raise StageError(
                    "sam3d",
                    (
                        "image-conditioned generation produced no mesh.glb and "
                        "STAGE_D_ALLOW_PROXY_FALLBACK=false"
                    ),
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
            "cleanup_provider": cleanup_provider,
            "reference_image_original": str(original_reference_path) if original_reference_path else "",
            "reference_image": str(working_reference_path) if working_reference_path else "",
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
    mesh_uri = str(artifacts.get("collision_mesh_ply") or "").strip()
    if not visual_uri or not mesh_uri:
        raise StageError("sam3d", "nurec_outputs missing visual_usdz or collision_mesh_ply")

    from .common import resolve_gs_uri_to_path

    visual_src = resolve_gs_uri_to_path(visual_uri, storage_root)
    mesh_src = resolve_gs_uri_to_path(mesh_uri, storage_root)
    if not has_nonempty_file(visual_src):
        raise StageError("sam3d", f"missing NuRec visual USDZ at {visual_src}")
    if not has_nonempty_file(mesh_src):
        raise StageError("sam3d", f"missing NuRec mesh PLY at {mesh_src}")

    visual_dir = storage_root / assets_prefix / "obj_nurec_visual"
    shell_dir = storage_root / assets_prefix / "obj_scene_shell"
    ensure_dir(visual_dir)
    ensure_dir(shell_dir)

    visual_usdz = visual_dir / "model.usdz"
    shutil.copy2(visual_src, visual_usdz)
    _write_reference_model_usd(visual_dir / "model.usd", "model.usdz")

    shell_glb = shell_dir / "mesh.glb"
    _ply_to_glb(mesh_src, shell_glb)
    prune_report = _prune_scene_shell_mesh(shell_glb, swap_candidates)
    _write_reference_model_usd(shell_dir / "model.usd", "mesh.glb")

    shell_metadata = {
        "schema_version": "v1",
        "asset_id": "obj_scene_shell",
        "source": "nurec_nvblox_mesh",
        "source_uri": mesh_uri,
        "pruning": prune_report,
        "generated_at": utc_now_iso(),
    }
    write_json(shell_dir / "metadata.json", shell_metadata)

    visual_metadata = {
        "schema_version": "v1",
        "asset_id": "obj_nurec_visual",
        "source": "nurec_export",
        "source_uri": visual_uri,
        "generated_at": utc_now_iso(),
    }
    write_json(visual_dir / "metadata.json", visual_metadata)

    return {
        "visual_asset": f"{assets_prefix}/obj_nurec_visual/model.usd",
        "shell_asset": f"{assets_prefix}/obj_scene_shell/model.usd",
        "shell_mesh": f"{assets_prefix}/obj_scene_shell/mesh.glb",
        "pruning": prune_report,
    }


def write_swap_execution_report(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, payload)


def write_swap_quality_report(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, payload)


def write_completion_marker(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, payload)


def write_failure_marker(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, payload)
