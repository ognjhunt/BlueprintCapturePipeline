"""Native Atlas refined appearance, behind the existing admitted task runner.

ADP-040 / day 28. A depth-fused preview may guide coverage, but newly purchased
final appearance comes from posedImages2Splats. It is never metric/physics proof.
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


def prepare_splat_request(*, frames, root: Path, prompt=""):
    """Keep all RGB/depth/camera tuples; mixed grids use the existing ray-preserving tiler."""
    from PIL import Image
    from .atlas_depth_splat import write_exr_depth
    from .website_atlas_tasks import validate_task_request

    root.mkdir(parents=True, exist_ok=True)
    ids = [f["frame_id"] for f in frames]
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("atlas_splat_distinct_frames_required")
    grids = {(f["intrinsics"]["width"], f["intrinsics"]["height"]) for f in frames}
    if len(grids) == 1:
        contexts, uploads, provenance = [], {}, []
        for i, frame in enumerate(frames):
            k = frame["intrinsics"]
            if (frame["rgb"].shape != (k["height"], k["width"], 3)
                    or frame["depth"].shape != (k["height"], k["width"])):
                raise ValueError("atlas_splat_frame_grid_changed")
            rgb, depth = root / f"{i:04d}.png", root / f"{i:04d}.exr"
            Image.fromarray(frame["rgb"]).save(rgb)
            write_exr_depth(depth, frame["depth"])
            camera = {"intrinsics": frame["intrinsics"], "extrinsics": frame["extrinsics"]}
            for role, path in (("rgb", rgb), ("depth", depth)):
                uploads[f"frame-{i}-{role}"] = {"path": str(path.resolve()), "sha256": _sha256_file(path)}
            contexts.append({"imageAsset": {"assetId": f"upload:frame-{i}-rgb"}, "camera": camera,
                             "depth": {"depthAsset": {"assetId": f"upload:frame-{i}-depth"}}})
            provenance.append({"source_frame_id": frame["frame_id"], "camera": camera,
                               "image_sha256": uploads[f"frame-{i}-rgb"]["sha256"],
                               "depth_sha256": uploads[f"frame-{i}-depth"]["sha256"]})
    else:
        from .atlas_scene_generate import prepare_rgbd_context
        contexts, uploads, provenance = prepare_rgbd_context(frames=frames, root=root)
    request = {"posedFrames": contexts}
    if prompt.strip():
        request.update(prompt=prompt.strip(), enhancePrompt=False)
    validate_task_request("posedImages2Splats", request)
    return request, uploads, provenance


def run_native_splat(*, frames, result, output_root: Path, policy, task_context,
                     task_runner, max_polls=2, generated_result_digests=(),
                     visual_review_digest=None, checkpoint=None, prompt="", decoder_root=None):
    """Resume one native operation; publish its original SPZ and lossless decoded PLY."""
    from .atlas_depth_splat import splats2mesh_task_spec
    from .atlas_pricing import splat_cost
    from .gaussian_splat_decode import convert_to_standard_ply, read_standard_3dgs_ply, find_splat_transform_cli
    from .atlas_result_bundle import _completed_task

    request, uploads, provenance = prepare_splat_request(frames=frames, root=output_root / "inputs", prompt=prompt)
    source = result["binding"]["source_video_digest"]
    stage = output_root / "native"
    binding = {"pose_result_digest": result["digest"], "source_video_digest": source,
               "request_digest": canonical_digest(request), "uploads": uploads,
               "frames": provenance, "generated_result_digests": list(generated_result_digests),
               "visual_review_digest": visual_review_digest}
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
        if find_splat_transform_cli(decoder_root) is None or shutil.which("node") is None:
            raise ValueError("atlas_native_splat_decoder_required_before_purchase")
        quote = {"schema_version": "website_atlas_task_price_review.v1", "task": "posedImages2Splats",
                 "request_digest": canonical_digest(request), "maximum_cost_usd": splat_cost(policy),
                 **{key: policy[key] for key in ("rate_card_reference", "reviewed_by", "reviewed_on")}}
        if checkpoint:
            checkpoint("refined_splat")
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
    outputs = state["outputs"]
    selected = max((r for r in outputs if r["role"].startswith("refinedSplats-")),
                   key=lambda r: int(r["role"].split("-")[-1]))
    original = stage / selected["path"]
    if (original.is_symlink() or not original.resolve().is_relative_to(stage.resolve())
            or _sha256_file(original) != selected["sha256"]):
        raise ValueError("atlas_native_splat_output_changed")
    spz, ply = output_root / "atlas_refined_splat.spz", output_root / "atlas_refined_splat.ply"
    shutil.copyfile(original, spz)
    decoded = convert_to_standard_ply(spz, ply, repo_root=decoder_root)
    if decoded["status"] != "completed":
        raise ValueError("atlas_native_splat_decode_unavailable:" + ",".join(decoded["blockers"]))
    splat = read_standard_3dgs_ply(ply)
    parameters = [splat.xyz, splat.opacity, splat.f_dc, splat.scales, splat.quats]
    if splat.sh_rest is not None:
        parameters.append(splat.sh_rest)
    if splat.count != int(selected["role"].split("-")[-1]) or not all(np.isfinite(v).all() for v in parameters):
        raise ValueError("atlas_native_splat_decoded_count_invalid")
    # A numerical registration tolerance, not a measurement or a fusion voxel.
    scale = float(np.median(np.exp(splat.scales.astype(np.float64))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("atlas_native_splat_scale_invalid")
    manifest = {"schema_version": "atlas_refined_splat.v1", **binding,
        "input_binding_digest": canonical_digest(binding), "native_result_digest": state["digest"],
        "pose_operation_id": result["operation"]["id"], "operation_id": state["operation"]["id"],
        "provider": "world_labs_atlas", "task": "posedImages2Splats", "gaussian_count": splat.count,
        "source_geometry_preservation_proven": False,
        "splat_kind": "atlas_refined", "photometric_optimization": True, "generated_content": True,
        "world_frame": "atlas_world_rub_y_up", "units": result.get("depth_unit"),
        "scale_status": "unscaled_provider_world_units", "registration_tolerance_world_units": scale,
        "metric_measurement_proven": False, "simulation_qualified": False, "physical_evidence": False,
        "claim_scope": "development_only", "billing": state["billing"], "decoder": decoded["decoder"],
        "outputs": [{"role": role, "path": path.name, "sha256": _sha256_file(path), "bytes": path.stat().st_size}
                    for role, path in (("splat_spz", spz), ("splat_ply", ply))]}
    manifest["digest"] = canonical_digest(manifest, digest_field="digest")
    write_json(output_root / "splats2mesh_task_spec.json",
               splats2mesh_task_spec(frames=frames, spz=spz, source_video_digest=source))
    write_json(output_root / MANIFEST, manifest)
    return {"status": "completed", "splat_manifest": manifest, "splat_root": str(output_root)}
