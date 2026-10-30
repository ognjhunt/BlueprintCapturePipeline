"""Atlas refined splats as the final appearance, behind the admitted task runner.

ADP-040 / day 28. The local depth-fused draft still drives gap detection and
gap review. When the reviewed policy prices a refined splat, the views that
draft fused are posted once to posedImages2Splats (real frames whole; partly
admitted generated views only where their review admitted them), and its SPZ
replaces the draft for meshing, export and scene handoff. Refinement is generated appearance: it never proves metric
scale, source-geometry preservation or physics.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file

MANIFEST = "atlas_refined_splat_manifest.json"
LEGACY_MANIFEST = "atlas_depth_splat_manifest.json"


def splat_manifest_path(root: Path) -> Path:
    native = root / MANIFEST
    return native if native.exists() else root / LEGACY_MANIFEST


def load_splat_manifest(root: Path) -> dict:
    path = splat_manifest_path(root)
    if path.is_symlink():
        raise ValueError("atlas_splat_manifest_symlink")
    value = json.loads(path.read_text())
    if value.get("digest") != canonical_digest(value, digest_field="digest"):
        raise ValueError("atlas_splat_manifest_changed")
    return value


def _png(values, path: Path) -> Path:
    from PIL import Image

    Image.fromarray(values.astype(np.uint8)).save(path)
    return path


def _admission(frame) -> tuple[bool, object]:
    """Whether a view enters the splat, and its alpha mask (None: every pixel).

    Real frames go in whole: low Atlas depth confidence does not make their RGB
    wrong. A review that admitted only part of a view (generated gap views,
    recovered geometry) recorded that in ``keep_mask``; only those pixels enter.
    """
    keep = frame.get("keep_mask")
    if keep is None or keep.all():
        return True, None
    return bool(keep.any()), keep


def prepare_splat_request(*, frames, root: Path):
    """Post every fused view with its camera, linear depth, Atlas depth confidence and review mask.

    One grid is posted unchanged; mixed grids (portrait source beside landscape
    generated views) use the ray-preserving photograph tiler.
    """
    from PIL import Image
    from .atlas_depth_splat import write_exr_depth
    from .website_atlas_tasks import validate_task_request

    root.mkdir(parents=True, exist_ok=True)
    # Recovered views can repeat a generated view's ID; the position is the identity here.
    keyed = [{**frame, "frame_id": f"{index:04d}"} for index, frame in enumerate(frames)]
    views = []
    if len({(f["intrinsics"]["width"], f["intrinsics"]["height"]) for f in frames}) == 1:
        for frame in keyed:
            k = frame["intrinsics"]
            if (frame["rgb"].shape != (k["height"], k["width"], 3)
                    or frame["depth"].shape != (k["height"], k["width"])):
                raise ValueError("atlas_splat_frame_grid_changed")
            admitted, keep = _admission(frame)
            if not admitted:
                continue  # The review admitted none of this view's pixels.
            key = frame["frame_id"]
            rgb = root / f"{key}.png"
            Image.fromarray(frame["rgb"]).save(rgb)
            confidence = np.clip(np.round(np.nan_to_num(frame["confidence"]) * 255), 0, 255)
            views.append((key, rgb, write_exr_depth(root / f"{key}.exr", frame["depth"]),
                          None if keep is None else _png(np.where(keep, 255, 0), root / f"{key}-mask.png"),
                          _png(confidence, root / f"{key}-confidence.png"),
                          {"intrinsics": frame["intrinsics"], "extrinsics": frame["extrinsics"]}))
    else:
        from .atlas_scene_chisel import prepare_source_photographs, project_source_depth

        by_id = {}
        for frame in keyed:
            admitted, keep = _admission(frame)
            if admitted:
                # Project depth over the admitted pixels only (all of a real frame).
                support = np.ones(frame["depth"].shape, np.float32) if keep is None else keep.astype(np.float32)
                by_id[frame["frame_id"]] = ({**frame, "confidence": support}, keep is not None)
        for photo in prepare_source_photographs(frames=[f for f, _ in by_id.values()], root=root / "photographs"):
            key = photo["frame_id"]
            source, masked = by_id[photo["source_frame_id"]]
            depth = project_source_depth([source], photo["camera"])
            valid = np.isfinite(depth) & (depth > 0)
            if masked and not valid.any():
                continue
            views.append((key, Path(photo["image_path"]), write_exr_depth(root / f"{key}.exr", depth),
                          _png(np.where(valid, 255, 0), root / f"{key}-mask.png") if masked and not valid.all() else None,
                          None, photo["camera"]))
    if len(views) < 2:
        raise ValueError("atlas_splat_frames_required")
    contexts, uploads, provenance = [], {}, []
    for i, (key, rgb, depth, mask, confidence, camera) in enumerate(views):
        context = {"imageAsset": {"assetId": f"upload:frame-{i}-rgb"}, "camera": camera,
                   "depth": {"depthAsset": {"assetId": f"upload:frame-{i}-depth"}}}
        row = {"frame_id": frames[int(key[:4])]["frame_id"], "camera": camera}
        for role, path in (("rgb", rgb), ("depth", depth), ("mask", mask), ("confidence", confidence)):
            if path is None:
                continue
            uploads[f"frame-{i}-{role}"] = {"path": str(Path(path).resolve()), "sha256": _sha256_file(Path(path))}
            row[f"{role}_sha256"] = uploads[f"frame-{i}-{role}"]["sha256"]
        if mask is not None:
            context["maskAsset"] = {"assetId": f"upload:frame-{i}-mask"}
        if confidence is not None:
            context["depth"]["confidenceAsset"] = {"assetId": f"upload:frame-{i}-confidence"}
        contexts.append(context)
        provenance.append(row)
    request = {"posedFrames": contexts}
    validate_task_request("posedImages2Splats", request)
    return request, uploads, provenance


def run_native_splat(*, frames, result, draft, output_root: Path, policy, task_context,
                     task_runner, max_polls=2, checkpoint=None):
    """Resume one refined-splat operation; publish its original SPZ and lossless decoded PLY."""
    from .atlas_depth_splat import decode_spz, splats2mesh_task_spec
    from .atlas_pricing import splat_cost
    from .gaussian_splat_decode import write_standard_3dgs_ply
    from .atlas_result_bundle import _completed_task

    request, uploads, provenance = prepare_splat_request(frames=frames, root=output_root / "inputs")
    source = result["binding"]["source_video_digest"]
    stage = output_root / "native"
    binding = {"pose_result_digest": result["digest"], "source_video_digest": source,
               "draft_splat_manifest_digest": draft["digest"], "request_digest": canonical_digest(request),
               "uploads": uploads, "frames": provenance,
               "generated_result_digests": list(draft.get("generated_result_digests") or []),
               "visual_review_digest": draft.get("visual_review_digest")}
    record = output_root / "input_binding.json"
    if record.exists() and json.loads(record.read_text()) != binding:
        raise ValueError("atlas_native_splat_inputs_changed")
    write_json(record, binding)
    if (stage / "result.json").is_file():
        state = _completed_task(stage, source, "splat")
        if (state.get("binding", {}).get("request_template") != request
                or state["binding"].get("uploads") != uploads
                or state["binding"].get("task_context_digest") != task_context["context_digest"]):
            raise ValueError("atlas_native_splat_receipt_binding_changed")
    else:
        quote = {"schema_version": "website_atlas_task_price_review.v1", "task": "posedImages2Splats",
                 "request_digest": canonical_digest(request), "maximum_cost_usd": splat_cost(policy),
                 **{key: policy[key] for key in ("rate_card_reference", "reviewed_by", "reviewed_on")}}
        if checkpoint:
            checkpoint("refined_splat", frame_count=len(provenance))
        state = task_runner(task="posedImages2Splats", request=request, uploads=uploads,
            source_video_digest=source, output_root=stage, task_context=task_context,
            price_review=quote, max_polls=max_polls)
    if state["status"] != "completed":
        return {"status": state["status"], "stage": "refined_splat", "operation_id": state.get("operation_id")}
    if (output_root / MANIFEST).is_file():
        manifest = load_splat_manifest(output_root)
        if manifest["native_result_digest"] != state["digest"] or manifest["input_binding_digest"] != canonical_digest(binding):
            raise ValueError("atlas_native_splat_result_changed")
        for row in manifest["outputs"]:
            path = output_root / row["path"]
            if path.is_symlink() or not path.resolve().is_relative_to(output_root.resolve()) or _sha256_file(path) != row["sha256"]:
                raise ValueError("atlas_native_splat_output_changed")
        return {"status": "completed", "splat_manifest": manifest, "splat_root": str(output_root)}
    # Atlas may publish several counts; the largest is the full refinement.
    selected = max((r for r in state["outputs"] if r["role"].startswith("refinedSplats-")),
                   key=lambda r: int(r["role"].split("-")[-1]))
    original = stage / selected["path"]
    if (original.is_symlink() or not original.resolve().is_relative_to(stage.resolve())
            or _sha256_file(original) != selected["sha256"]):
        raise ValueError("atlas_native_splat_output_changed")
    spz, ply = output_root / "atlas_refined_splat.spz", output_root / "atlas_refined_splat.ply"
    shutil.copyfile(original, spz)
    # Atlas publishes SPZ v2 with degree-0 colour, which decodes exactly in
    # Python (bit-identical to splat-transform on a 2M-Gaussian Atlas splat).
    # Any other layout fails closed here with the paid SPZ retained.
    splat = decode_spz(spz.read_bytes())
    parameters = [splat.xyz, splat.opacity, splat.f_dc, splat.scales, splat.quats]
    if splat.count != int(selected["role"].split("-")[-1]) or not all(np.isfinite(v).all() for v in parameters):
        raise ValueError("atlas_native_splat_decoded_count_invalid")
    write_standard_3dgs_ply(splat, ply)
    manifest = {"schema_version": "atlas_refined_splat.v1", **binding,
        "input_binding_digest": canonical_digest(binding), "native_result_digest": state["digest"],
        "pose_operation_id": result["operation"]["id"], "operation_id": state["operation"]["id"],
        "provider": "world_labs_atlas", "task": "posedImages2Splats", "gaussian_count": splat.count,
        # Mesh registration measures support in the draft's fusion grid.
        "voxel_size": draft["voxel_size"],
        "source_geometry_preservation_proven": False, "splat_kind": "atlas_refined",
        "photometric_optimization": True, "generated_content": True,
        "world_frame": "atlas_world_rub_y_up", "units": result.get("depth_unit"),
        "scale_status": "unscaled_provider_world_units", "metric_measurement_proven": False,
        "simulation_qualified": False, "physical_evidence": False,
        "claim_scope": "development_only", "billing": state["billing"],
        "decoder": {"name": "atlas_depth_splat.decode_spz", "format": "spz_v2_sh0"},
        "outputs": [{"role": role, "path": path.name, "sha256": _sha256_file(path), "bytes": path.stat().st_size}
                    for role, path in (("splat_spz", spz), ("splat_ply", ply))]}
    manifest["digest"] = canonical_digest(manifest, digest_field="digest")
    write_json(output_root / "splats2mesh_task_spec.json",
               splats2mesh_task_spec(frames=frames, spz=spz, source_video_digest=source))
    write_json(output_root / MANIFEST, manifest)
    return {"status": "completed", "splat_manifest": manifest, "splat_root": str(output_root)}
