"""Atlas refined splats as the final appearance, behind the admitted task runner.

ADP-040 / day 28. The local depth-fused draft still drives gap detection and
gap review. When the reviewed policy prices a refined splat, the views that
draft fused are posted once to posedImages2Splats (real frames whole; partly
admitted generated views only where their review admitted them), and its SPZ
replaces the draft for meshing, export and scene handoff. Refinement is generated appearance: it never proves metric
scale, source-geometry preservation or physics.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file

MANIFEST = "atlas_refined_splat_manifest.json"
LEGACY_MANIFEST = "atlas_depth_splat_manifest.json"
UPGRADE = "gap_preserving_upgrade.json"
# Each request form has its own root beside the draft, tried in this order.
# "-merged" is only ever an operator upgrade: the full form again, once its own
# root had posted duplicate views.
FORMS = (("reviewed_masks_and_confidence", ""), ("reviewed_masks_and_confidence", "-merged"),
         ("reviewed_masks", "-masks"), ("documented_minimal", "-minimal"))


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


def _same_camera(a, b, tolerance=1e-6) -> bool:
    """The same grid and axes, with intrinsics, position and rotation within a tiny tolerance."""
    from .atlas_depth_splat import _rotation_matrix

    def vector(camera):
        k, e = camera["intrinsics"], camera["extrinsics"]
        return np.concatenate([[float(k[key]) for key in ("fx", "fy", "cx", "cy")],
                               np.asarray(e["position"], float), _rotation_matrix(e["quaternion"]).ravel()])
    grids = [(c["intrinsics"]["width"], c["intrinsics"]["height"], c["extrinsics"].get("coordinateSystem", "rub"))
             for c in (a, b)]
    return grids[0] == grids[1] and bool(np.allclose(vector(a), vector(b), rtol=0, atol=tolerance))


def merge_duplicate_views(frames):
    """One posed view per camera and image: (base position, merged view, duplicate positions).

    capture-447ae4f4 (2026-10-30) posted 4 of 22 views twice: recovered views
    repeating the generation anchors and the generated gap views, with their own
    depth and masks. A whole duplicate makes the view whole, otherwise the
    admissions unite; depth and confidence start from the whole (or first) view
    and take each partial duplicate's values where its review admitted them.
    """
    groups = []
    for index, frame in enumerate(frames):
        rgb = np.ascontiguousarray(frame["rgb"])
        key = (rgb.shape, hashlib.sha256(rgb.tobytes()).digest())
        group = next((g for g in groups if g[0] == key and _same_camera(frames[g[1][0]], frame)), None)
        if group is None:
            groups.append((key, [index]))
        else:
            group[1].append(index)
    merged = []
    for _, members in groups:
        keeps = {i: _admission(frames[i])[1] for i in members}
        base = next((i for i in members if keeps[i] is None), members[0])
        view = frames[base]
        if len(members) > 1:
            depth, confidence = np.array(view["depth"], copy=True), np.array(view["confidence"], copy=True)
            for i in members:
                if i != base and keeps[i] is not None:
                    depth[keeps[i]] = frames[i]["depth"][keeps[i]]
                    confidence[keeps[i]] = frames[i]["confidence"][keeps[i]]
            whole = any(keep is None for keep in keeps.values())
            view = {**view, "depth": depth, "confidence": confidence,
                    "keep_mask": None if whole else np.logical_or.reduce(list(keeps.values()))}
        merged.append((base, view, [i for i in members if i != base]))
    return merged


def prepare_splat_request(*, frames, root: Path, form="reviewed_masks_and_confidence"):
    """Post each fused view once with its camera, linear depth, Atlas depth confidence and review mask.

    Duplicate views merge first. One grid is posted unchanged; mixed grids
    (portrait source beside landscape generated views) use the ray-preserving
    photograph tiler. ``reviewed_masks`` leaves out the confidence assets;
    ``documented_minimal`` sends the provider guide's form: whole views only,
    image + depth + camera.
    """
    from PIL import Image
    from .atlas_depth_splat import write_exr_depth
    from .website_atlas_tasks import validate_task_request

    root.mkdir(parents=True, exist_ok=True)
    minimal = form == "documented_minimal"
    merged = merge_duplicate_views(frames)
    # Recovered views can repeat a generated view's ID; the position is the identity here.
    keyed = [{**view, "frame_id": f"{base:04d}"} for base, view, _ in merged]
    duplicates = {f"{base:04d}": rest for base, _, rest in merged if rest}
    views = []
    if len({(f["intrinsics"]["width"], f["intrinsics"]["height"]) for f in frames}) == 1:
        for frame in frames:
            k = frame["intrinsics"]
            if (frame["rgb"].shape != (k["height"], k["width"], 3)
                    or frame["depth"].shape != (k["height"], k["width"])):
                raise ValueError("atlas_splat_frame_grid_changed")
        for frame in keyed:
            admitted, keep = _admission(frame)
            if not admitted or (minimal and keep is not None):
                continue  # The review admitted none (or, in the minimal form, not all) of this view.
            key = frame["frame_id"]
            rgb = root / f"{key}.png"
            Image.fromarray(frame["rgb"]).save(rgb)
            confidence = np.clip(np.round(np.nan_to_num(frame["confidence"]) * 255), 0, 255)
            views.append((key, rgb, write_exr_depth(root / f"{key}.exr", frame["depth"]),
                          None if keep is None else _png(np.where(keep, 255, 0), root / f"{key}-mask.png"),
                          _png(confidence, root / f"{key}-confidence.png")
                          if form == "reviewed_masks_and_confidence" else None,
                          {"intrinsics": frame["intrinsics"], "extrinsics": frame["extrinsics"]}))
    else:
        from .atlas_scene_chisel import prepare_source_photographs, project_source_depth

        by_id = {}
        for frame in keyed:
            admitted, keep = _admission(frame)
            if admitted and not (minimal and keep is not None):
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
        if key[:4] in duplicates:
            row["merged_duplicates"] = [{"position": j, "frame_id": frames[j]["frame_id"]} for j in duplicates[key[:4]]]
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


def _provider_failed_and_exhausted(root: Path) -> bool:
    """Every attempt under root ended without a purchase, and the provider failed one."""
    states = []
    for path in root.rglob("atlas_pose/submission.json"):
        try:
            states.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            return False
    return (bool(states) and all(state.get("status") in ("failed", "rejected") for state in states)
            and any(state.get("status") == "failed" and state.get("terminal_operation") for state in states))


def _form_roots(output_root: Path):
    return [(form, output_root.with_name(output_root.name + suffix)) for form, suffix in FORMS]


def _bought(root: Path) -> bool:
    return any((root / "native").rglob("atlas_pose/submission.json"))


def _started(root: Path) -> bool:
    return (root / "input_binding.json").is_file() or _bought(root)


def _posted_duplicates(root: Path) -> bool:
    """The root's retained request predates merging and posted one camera and image twice."""
    path = root / "input_binding.json"
    rows = (json.loads(path.read_text()).get("frames") or []) if path.is_file() else []
    return any(a.get("rgb_sha256") and a["rgb_sha256"] == b.get("rgb_sha256") and _same_camera(a["camera"], b["camera"])
               for i, a in enumerate(rows) for b in rows[i + 1:])


def _untried_gap_roots(roots) -> list:
    """Gap-preserving forms never sent: the full form (merged, if its root posted duplicates), then masks."""
    (_, full), (_, merged), (_, masks), _ = roots
    untried = [full] if not _bought(full) else [merged] if not _bought(merged) and _posted_duplicates(full) else []
    return untried + ([masks] if not _bought(masks) else [])


def _upgrade_roots(minimal: Path) -> set:
    """Roots an operator's explicit upgrade of this exact minimal splat may start."""
    path = minimal / UPGRADE
    if path.is_symlink() or not path.is_file() or not (minimal / MANIFEST).is_file():
        return set()
    value = json.loads(path.read_text())
    return (set(value.get("roots") or [])
            if value.get("splat_manifest_digest") == load_splat_manifest(minimal)["digest"] else set())


def request_gap_preserving_upgrade(root: Path, *, splat_manifest_digest: str, at_epoch: int) -> dict:
    """Record an operator's request to send the gap-preserving forms a minimal splat left untried.

    Never automatic. The next run sends each named root at most once; a success
    supersedes the minimal splat, and if every one fails it is adopted again.
    """
    if root.is_symlink() or not root.name.endswith("-minimal") or not (root / MANIFEST).is_file():
        raise ValueError("atlas_refined_splat_upgrade_invalid")
    manifest = load_splat_manifest(root)
    if manifest["digest"] != splat_manifest_digest or manifest.get("request_form") != "documented_minimal":
        raise ValueError("atlas_refined_splat_upgrade_invalid")
    roots = [path.name for path in _untried_gap_roots(_form_roots(root.with_name(root.name.removesuffix("-minimal"))))]
    if not roots:
        raise ValueError("atlas_refined_splat_upgrade_unavailable")
    value = {"schema_version": "atlas_refined_splat_upgrade.v1", "splat_manifest_digest": splat_manifest_digest,
             "roots": roots, "requested_at_epoch": at_epoch}
    write_json(root / UPGRADE, value)
    return value


def _refined_marker(form, root: Path, manifest, frames, roots) -> dict:
    """The adopted form; a minimal splat also names the admitted partial views it left out."""
    marker = {"request_form": form, "splat_root": str(root), "splat_manifest_digest": manifest["digest"]}
    left = sorted({frames[base]["frame_id"] for base, view, _ in merge_duplicate_views(frames)
                   if _admission(view)[0] and _admission(view)[1] is not None}) if form == "documented_minimal" else []
    if left:
        marker.update(left_out_admitted_view_ids=left, gap_preserving_form_untried=bool(_untried_gap_roots(roots)))
    return marker


def run_native_splat(*, frames, result, draft, output_root: Path, policy, task_context,
                     task_runner, max_polls=2, checkpoint=None):
    """Resume one refined-splat operation; publish its original SPZ and lossless decoded PLY.

    Forms run in order, each from its own root: reviewed masks and confidence,
    masks only (``-masks``), the guide's documented minimal form (``-minimal``).
    World Labs failed the full form with EXECUTION_FAILED (capture-4b60431e and
    capture-447ae4f4, 2026-10-30). A form is left only once every attempt of it
    was provider-failed or rejected; a completed splat in any root is adopted,
    and no earlier form is newly bought behind a later live or completed one
    unless an operator requested that upgrade (``request_gap_preserving_upgrade``).
    """
    roots = _form_roots(output_root)
    upgrade = _upgrade_roots(roots[-1][1])
    for index, (form, root) in enumerate(roots):
        if not (root / "native/result.json").is_file():
            if _provider_failed_and_exhausted(root / "native"):
                continue
            later = any(_started(r) and not _provider_failed_and_exhausted(r / "native") for _, r in roots[index + 1:])
            if not _started(root) and root.name not in upgrade and (later or index == 1):
                continue
        try:
            value = _run_form(form=form, root=root, frames=frames, result=result, draft=draft, policy=policy,
                              task_context=task_context, task_runner=task_runner, max_polls=max_polls,
                              checkpoint=checkpoint)
        except ValueError:
            # The provider failed this form's last attempt; the next form follows.
            if not _provider_failed_and_exhausted(root / "native"):
                raise
            continue
        if value["status"] == "completed":
            value["refined_splat"] = _refined_marker(form, root, value["splat_manifest"], frames, roots)
        return value
    raise ValueError("atlas_task_operation_failed")


def _run_form(*, form, root: Path, frames, result, draft, policy, task_context, task_runner, max_polls,
              checkpoint):
    """One form's root. A bought root is evaluated by what it sent, never re-derived.

    Its request may predate duplicate merging, and its retained uploads stay intact.
    """
    from .atlas_depth_splat import decode_spz, splats2mesh_task_spec
    from .atlas_pricing import splat_cost
    from .gaussian_splat_decode import write_standard_3dgs_ply
    from .atlas_result_bundle import _completed_task

    source = result["binding"]["source_video_digest"]
    stage, record = root / "native", root / "input_binding.json"
    identity = {"pose_result_digest": result["digest"], "source_video_digest": source,
                "draft_splat_manifest_digest": draft["digest"],
                "generated_result_digests": list(draft.get("generated_result_digests") or []),
                "visual_review_digest": draft.get("visual_review_digest")}
    state = _completed_task(stage, source, "splat") if (stage / "result.json").is_file() else None
    submission = stage / "atlas_pose/submission.json"
    sent = state["binding"] if state else (json.loads(submission.read_text()).get("binding")
                                           if submission.is_file() else None)
    if sent and record.is_file():
        binding = json.loads(record.read_text())
        request, uploads = sent["request_template"], sent["uploads"]
        if any(binding.get(key) != value for key, value in identity.items()):
            raise ValueError("atlas_native_splat_inputs_changed")
        if canonical_digest(request) != binding.get("request_digest") or uploads != binding.get("uploads"):
            raise ValueError("atlas_native_splat_receipt_binding_changed")
    else:
        request, uploads, provenance = prepare_splat_request(frames=frames, root=root / "inputs", form=form)
        binding = {**identity, "request_digest": canonical_digest(request), "request_form": form,
                   "uploads": uploads, "frames": provenance}
        # A root that sent nothing follows the current request (one written before merging, say).
        if record.exists() and json.loads(record.read_text()) != binding and _bought(root):
            raise ValueError("atlas_native_splat_inputs_changed")
        write_json(record, binding)
    if state is not None:
        if (state["binding"].get("request_template") != request
                or state["binding"].get("uploads") != uploads
                or state["binding"].get("task_context_digest") != task_context["context_digest"]):
            raise ValueError("atlas_native_splat_receipt_binding_changed")
    else:
        quote = {"schema_version": "website_atlas_task_price_review.v1", "task": "posedImages2Splats",
                 "request_digest": canonical_digest(request), "maximum_cost_usd": splat_cost(policy),
                 **{key: policy[key] for key in ("rate_card_reference", "reviewed_by", "reviewed_on")}}
        if checkpoint:
            checkpoint("refined_splat", frame_count=len(binding["frames"]))
        state = task_runner(task="posedImages2Splats", request=request, uploads=uploads,
            source_video_digest=source, output_root=stage, task_context=task_context,
            price_review=quote, max_polls=max_polls)
    if state["status"] != "completed":
        return {"status": state["status"], "stage": "refined_splat", "operation_id": state.get("operation_id")}
    if (root / MANIFEST).is_file():
        manifest = load_splat_manifest(root)
        if manifest["native_result_digest"] != state["digest"] or manifest["input_binding_digest"] != canonical_digest(binding):
            raise ValueError("atlas_native_splat_result_changed")
        for row in manifest["outputs"]:
            path = root / row["path"]
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or _sha256_file(path) != row["sha256"]:
                raise ValueError("atlas_native_splat_output_changed")
        return {"status": "completed", "splat_manifest": manifest, "splat_root": str(root)}
    # Atlas may publish several counts; the largest is the full refinement.
    selected = max((r for r in state["outputs"] if r["role"].startswith("refinedSplats-")),
                   key=lambda r: int(r["role"].split("-")[-1]))
    original = stage / selected["path"]
    if (original.is_symlink() or not original.resolve().is_relative_to(stage.resolve())
            or _sha256_file(original) != selected["sha256"]):
        raise ValueError("atlas_native_splat_output_changed")
    spz, ply = root / "atlas_refined_splat.spz", root / "atlas_refined_splat.ply"
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
    write_json(root / "splats2mesh_task_spec.json",
               splats2mesh_task_spec(frames=frames, spz=spz, source_video_digest=source))
    write_json(root / MANIFEST, manifest)
    return {"status": "completed", "splat_manifest": manifest, "splat_root": str(root)}
