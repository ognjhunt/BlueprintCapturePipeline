"""CPU preparation and verified GPU-return continuation of source-calibrated views."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .decision_evidence_contracts import canonical_digest, canonical_json
from .sealed_camera_render import _standard_ply_vertex_count

SCHEMA = "public_scene_inpainting_prepared_inputs.v1"
FILENAME = SCHEMA + ".json"
ROLES = ("images", "target_support", "scene_without_target")


def _artifact(path: Path) -> dict[str, Any]:
    from .public_scene_inpainting_inputs import _sha256
    return {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def prepare_public_scene_inpainting_inputs(**kwargs) -> dict[str, Any]:
    from .public_scene_inpainting_inputs import (
        _prepare_public_scene_inpainting_context, SEALED_SOURCE_ADAPTERS,
        PublicSceneInpaintingInputError,
    )
    context = _prepare_public_scene_inpainting_context(**kwargs)
    if context["source_adapter"] not in SEALED_SOURCE_ADAPTERS:
        raise PublicSceneInpaintingInputError(["edit_input_gpu_preparation_requires_sealed_source"])
    paths, identity = context["paths"], context["source_identity"]
    layers = {}
    for role, field, count, purpose in (
        ("images", "standard_ply", context["scene_gaussian_count"], "complete source appearance"),
        ("target_support", "target_ply", context["target_count"], "candidate target OBB Gaussian support"),
        ("scene_without_target", "background_ply", context["scene_gaussian_count"] - context["target_count"],
         "source appearance without candidate target OBB Gaussians"),
    ):
        layers[role] = {**_artifact(Path(paths[field])), "retained_gaussian_count": count,
            "camera_set_label": f"{identity['task_id']}:removal_input:{role}",
            "purpose": f"{identity['task_id']} calibrated removal analysis: {purpose}",
            "provider_splat_import_receipt_digest": identity["conversion_receipt_digest"],
            "alignment_digest": identity["registered_frame_receipt_digest"]}
    output = Path(paths["output"])
    path = output / FILENAME
    prepared = {"schema_version": SCHEMA, "status": "prepared_for_exact_render",
        "preparation_path": str(path), "repository": context["repository"],
        "request_digest": context["request"]["request_digest"],
        "request_file": _artifact(Path(paths["request_file"])),
        "camera_file": _artifact(Path(paths["camera_file"])),
        "cameras": context["sealed_cameras"], "layers": layers,
        "render_options": dict(context["request"]["rendering"]),
        "context": context, "rendered": False, "candidate_policy_queried": False,
        "preparation_digest": ""}
    prepared["preparation_digest"] = canonical_digest(prepared, digest_field="preparation_digest")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(prepared) + "\n")
    return prepared


def validate_prepared_inputs(preparation_path: str | Path) -> dict[str, Any]:
    from .public_scene_inpainting_inputs import (
        PublicSceneInpaintingInputError, _git_identity, _read_object, _require_under,
        _verified_dual_task_scene_source, build_public_scene_inpainting_input_request, _camera_rows,
    )
    def require(condition, code):
        if not condition:
            raise PublicSceneInpaintingInputError(["edit_input_preparation_" + code])
    path = Path(preparation_path).expanduser()
    prepared = _read_object(path, code="edit_input_preparation_invalid")
    require(prepared.get("schema_version") == SCHEMA
        and prepared.get("status") == "prepared_for_exact_render"
        and prepared.get("preparation_digest") == canonical_digest(prepared, digest_field="preparation_digest"),
        "digest_invalid")
    context = prepared["context"]
    paths = context["paths"]
    repo, data, output = (Path(paths[key]) for key in ("repo", "data", "output"))
    require(path == output / FILENAME and prepared.get("preparation_path") == str(path), "path_mismatch")
    _require_under(output, (data,), code="edit_input_output_outside_data_root")
    require(prepared["repository"] == context["repository"] == _git_identity(repo), "execution_commit_mismatch")
    for row in [prepared["request_file"], prepared["camera_file"], *prepared["layers"].values()]:
        source = _require_under(Path(row["path"]), (repo, data), code="edit_input_preparation_artifact_outside_roots")
        require(_artifact(source) == {key: row[key] for key in ("path", "sha256", "size_bytes")}, "artifact_changed")
    request = build_public_scene_inpainting_input_request(
        _read_object(Path(paths["request_file"]), code="edit_input_request_invalid"))
    require(request == context["request"] and request["request_digest"] == prepared["request_digest"], "request_changed")
    require(prepared["render_options"] == request["rendering"], "render_options_changed")
    source = _verified_dual_task_scene_source(scene=request["scene"], repo=repo, data=data)
    require({key: value for key, value in source.items() if key not in {"standard_ply", "corners", "source_artifacts"}}
            == context["source_identity"] and source["source_artifacts"] == context["observed_sources"]
            and np.array_equal(source["corners"], np.asarray(context["corners"])), "source_changed")
    require(prepared["cameras"] == context["sealed_cameras"]
            == json.loads(Path(prepared["camera_file"]["path"]).read_text()), "cameras_changed")
    cameras = _camera_rows(request, source["corners"].mean(axis=0))
    require(context["cameras"] == cameras and prepared["cameras"] == [
        {"camera_id": row["camera_id"], "T_world_camera_provider_frame": row["T_world_camera_opencv"],
         "intrinsics": row["intrinsics"]} for row in cameras], "camera_request_binding_changed")
    require(set(prepared["layers"]) == set(ROLES), "layers_invalid")
    for role, field in zip(ROLES, ("standard_ply", "target_ply", "background_ply"), strict=True):
        row = prepared["layers"][role]
        require(row["path"] == paths[field]
                and _standard_ply_vertex_count(Path(row["path"])) == row["retained_gaussian_count"], "layer_count_changed")
    require(prepared["layers"]["images"]["retained_gaussian_count"] == context["scene_gaussian_count"]
            == source["gaussian_count"]
            and prepared["layers"]["target_support"]["retained_gaussian_count"] == context["target_count"]
            and prepared["layers"]["scene_without_target"]["retained_gaussian_count"]
            == context["scene_gaussian_count"] - context["target_count"], "counts_changed")
    return prepared


def finalize_public_scene_inpainting_inputs(*, preparation_path: str | Path,
        returned_group_path: str | Path) -> dict[str, Any]:
    from .public_scene_inpainting_inputs import PublicSceneInpaintingInputError, _sha256
    from .public_scene_inpainting_finalize import finish_prepared_inputs
    from .source_calibration_render_return import verify_source_calibration_return

    prepared = validate_prepared_inputs(preparation_path)
    returned_path = Path(returned_group_path).expanduser()
    groups = verify_source_calibration_return(prepared, returned_path)
    if set(groups) != set(ROLES):
        raise PublicSceneInpaintingInputError(["edit_input_returned_render_groups_incomplete"])
    context = prepared["context"]
    output = Path(context["paths"]["output"])
    receipt_name = "public_scene_interiorgs_edit_input_receipt.v2.json"
    retained = context["paths"]["retained_receipt"]
    if (output / receipt_name).exists() or (retained and Path(retained).exists()):
        raise PublicSceneInpaintingInputError(["edit_input_receipt_output_exists"])
    manifests = {}
    for role in ROLES:
        group = groups[role]
        manifests[role] = group["manifest"]
        manifest_source = Path(group["manifest_path"])
        manifest_bytes = manifest_source.read_bytes()
        if json.loads(manifest_bytes) != manifests[role]:
            raise PublicSceneInpaintingInputError(["edit_input_returned_manifest_changed"])
        sources = [(manifest_source, output / role / "sealed_camera_render_manifest.v1.json",
                    "sha256:" + hashlib.sha256(manifest_bytes).hexdigest())]
        frame_digests = {row["camera_id"]: row["digest"] for row in manifests[role]["renders"]}
        sources.extend((Path(group["root"]) / "frames" / f"{camera['camera_id']}.png",
                        output / role / "frames" / f"{camera['camera_id']}.png",
                        frame_digests[camera["camera_id"]]) for camera in prepared["cameras"])
        for source, target, expected_digest in sources:
            if (any(p.is_symlink() for p in (target, *target.parents))
                    or (target.exists() and _sha256(target) != expected_digest)):
                raise PublicSceneInpaintingInputError(["edit_input_returned_artifact_conflict"])
            if _sha256(source) != expected_digest:
                raise PublicSceneInpaintingInputError(["edit_input_returned_artifact_changed"])
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            if _sha256(target) != expected_digest:
                raise PublicSceneInpaintingInputError(["edit_input_returned_artifact_copy_mismatch"])
    commands = {role: {"command": ["sealed-camera-render", manifests[role]["sealed_camera_render_manifest_digest"]]}
                for role in ROLES}
    return finish_prepared_inputs(context, sealed_render_manifests=manifests,
        rgb_run=commands["images"], support_run=commands["target_support"],
        background_run=commands["scene_without_target"], render_frame_subdir="frames",
        render_execution_evidence={"preparation_digest": prepared["preparation_digest"],
                                   "returned_group": _artifact(returned_path)})
