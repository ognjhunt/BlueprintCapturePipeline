"""Portable, digest-bound downloads of an admitted Atlas pose run (ADP-040).

This exports observations, never manufactures a world, splat or qualified mesh.
A depth-fused draft splat and an Atlas splats2Mesh result travel only with their
digest-bound manifests and stay unscaled and development_only.
Provider read URLs often have no suffix; identify the retained bytes without
renaming or rewriting the authoritative run's files.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file
from .website_atlas_pose import load_pending_atlas_pose_inputs, verify_atlas_pose_receipt


def _replace_bundle_member(source: Path, destination: Path, *, duplicate: Path | None = None) -> None:
    """Publish complete members and keep other hard-linked snapshots unchanged."""
    with tempfile.TemporaryDirectory(prefix=".atlas-member-", dir=destination.parent) as folder:
        temporary = Path(folder) / "member"
        if duplicate is not None:
            try:
                os.link(duplicate, temporary)
            except OSError:
                shutil.copyfile(source, temporary)
        else:
            shutil.copyfile(source, temporary)
        if destination.exists() and (temporary.stat().st_mode & 0o777) != (destination.stat().st_mode & 0o777):
            if temporary.stat().st_nlink > 1:
                temporary.unlink()
                shutil.copyfile(source, temporary)
            shutil.copymode(destination, temporary)
        os.replace(temporary, destination)


def _suffix(data: bytes) -> str:
    if data.startswith(b"\x76\x2f\x31\x01"):
        return ".exr"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8"):
        return ".jpg"
    if data.startswith(b"glTF"):
        return ".glb"
    raise ValueError("atlas_bundle_output_format_unknown")


def _completed_task(root: Path, source_video_digest: str, label: str) -> dict:
    value = json.loads((root / "result.json").read_text())
    if (value.get("status") != "completed" or value.get("digest") != canonical_digest(value, digest_field="digest")
            or value["binding"]["source_video_digest"] != source_video_digest):
        raise ValueError(f"atlas_bundle_{label}_result_invalid")
    state = json.loads((root / "atlas_pose/submission.json").read_text())
    if state.get("status") != "completed" or state.get("result_digest") != value["digest"]:
        raise ValueError(f"atlas_bundle_{label}_receipt_invalid")
    return value


def retained_mesh_for_splat(*, mesh_root: Path, splat_root: Path,
                            source_video_digest: str, task_context_digest: str) -> dict:
    """Adopt the exact completed mesh, including a verified mesh-only reduction.

    Attempt paths are deliberately not rewritten: the original upload, operation
    and accounting receipts continue to identify the paid request.
    """
    spec = json.loads((splat_root / "splats2mesh_task_spec.json").read_text())
    from .atlas_native_splat import load_splat_manifest
    splat = load_splat_manifest(splat_root)
    mesh = _completed_task(mesh_root, source_video_digest, "mesh")
    state = json.loads((mesh_root / "atlas_pose/submission.json").read_text())
    binding = mesh["binding"]
    if (splat.get("digest") != canonical_digest(splat, digest_field="digest")
            or spec.get("source_video_digest") != source_video_digest
            or mesh.get("task") != "splats2Mesh" or binding.get("task") != "splats2Mesh"
            or binding.get("task_context_digest") != task_context_digest
            or binding.get("request_template") != spec.get("request")
            or state.get("binding") != binding
            or mesh.get("request_digest") != canonical_digest(binding)
            or state.get("request_digest") != mesh.get("request_digest")
            or state.get("operation_id") != mesh.get("operation", {}).get("id")):
        raise ValueError("atlas_retained_mesh_binding_changed")
    source = spec["uploads"]["splat"]
    upload = binding.get("uploads", {}).get("splat", {})
    source_path, upload_path = Path(source["path"]), Path(upload.get("path", ""))
    if (source_path.is_symlink() or not source_path.is_file()
            or _sha256_file(source_path) != source["sha256"]
            or upload_path.is_symlink() or not upload_path.is_file()
            or _sha256_file(upload_path) != upload.get("sha256")):
        raise ValueError("atlas_retained_mesh_input_changed")
    if upload["sha256"] != source["sha256"]:
        receipt_path = upload_path.parent / "mesh_input.json"
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise ValueError("atlas_retained_mesh_reduction_missing")
        reduction = json.loads(receipt_path.read_text())
        if (reduction.get("schema_version") != "atlas_splats2mesh_input.v1"
                or reduction.get("source_sha256") != source["sha256"]
                or reduction.get("source_gaussian_count") != splat.get("gaussian_count")
                or reduction.get("path") != str(upload_path)
                or reduction.get("sha256") != upload["sha256"]):
            raise ValueError("atlas_retained_mesh_reduction_changed")
    for row in mesh["outputs"]:
        path = mesh_root / row["path"]
        if (path.is_symlink() or not path.resolve().is_relative_to(mesh_root.resolve())
                or not path.is_file() or path.stat().st_size != row["bytes"]
                or _sha256_file(path) != row["sha256"]):
            raise ValueError("atlas_retained_mesh_output_changed")
    registration_path = mesh_root / "registered/atlas_mesh_registration.json"
    if registration_path.is_symlink() or not registration_path.is_file():
        raise ValueError("atlas_retained_mesh_registration_missing")
    registration = json.loads(registration_path.read_text())
    if (registration.get("digest") != canonical_digest(registration, digest_field="digest")
            or registration.get("splat_manifest_digest") != splat["digest"]
            or registration.get("source_glb_sha256") not in {row["sha256"] for row in mesh["outputs"]}):
        raise ValueError("atlas_retained_mesh_registration_changed")
    for row in registration["outputs"]:
        path = mesh_root / "registered" / row["path"]
        if (path.is_symlink() or not path.resolve().is_relative_to((mesh_root / "registered").resolve())
                or not path.is_file() or _sha256_file(path) != row["sha256"]):
            raise ValueError("atlas_retained_mesh_output_changed")
    return mesh


def find_retained_mesh_root(*, output_root: Path, splat_root: Path,
                            source_video_digest: str, task_context_digest: str) -> Path | None:
    """Find a same-job export without relying on its historical directory name.

    This is discovery only. The caller must validate it with
    ``retained_mesh_for_splat`` before adoption; matching corrupt state must not
    silently cause a second purchase.
    """
    from .atlas_native_splat import load_splat_manifest
    splat = load_splat_manifest(splat_root)
    for path in sorted(output_root.glob('*/registered/atlas_mesh_registration.json')):
        root = path.parent.parent
        if root.is_symlink() or path.is_symlink() or not (root / 'result.json').is_file():
            continue
        registration = json.loads(path.read_text())
        if registration.get('splat_manifest_digest') != splat['digest']:
            continue
        mesh = json.loads((root / 'result.json').read_text())
        binding = mesh.get('binding', {})
        if (mesh.get('status') == 'completed' and mesh.get('task') == 'splats2Mesh'
                and binding.get('source_video_digest') == source_video_digest
                and binding.get('task_context_digest') == task_context_digest):
            return root
    return None


def export_pose_bundle(*, pose_root: Path, output_root: Path, generated_roots=(), splat_root: Path | None = None,
                       mesh_root: Path | None = None, analysis_paths=()) -> dict:
    pose_root = pose_root.resolve()
    result = json.loads((pose_root / "atlas_posed_rgbd_result.json").read_text())
    inputs = load_pending_atlas_pose_inputs(pose_root)
    count = verify_atlas_pose_receipt(pose=result, frames=inputs["frames"], output_root=pose_root)
    receipt = result["operation_receipt"]
    manifest = {"schema_version": "atlas_download_bundle.v1", "source_video_digest": result["binding"]["source_video_digest"],
                "operation_id": result["operation"]["id"], "pose_result_digest": result["digest"],
                "provider_input_count": count, "pose_refinement_accepted": result["operation"]["response"].get("poseRefinementAccepted"),
                "claim_ceiling": "development_only", "metric_measurement_proven": False,
                "scale_status": "unscaled_provider_world_units", "physical_evidence": False,
                "splat_available": False, "mesh_available": False, "simulation_qualified": False,
                "billing": receipt["billing"], "assets": []}
    if inputs.get("generated_input_content") is True:
        manifest["generated_pose_inputs"] = True
    source_digest = result["binding"]["source_video_digest"]
    generated = [(root, _completed_task(root, source_digest, "generated")) for root in generated_roots]
    mesh = _completed_task(mesh_root, source_digest, "mesh") if mesh_root is not None else None
    registration = None
    if mesh is not None:
        # The raw provider GLB is in another frame and closes unseen regions;
        # it ships only beside its registration to the observed splat.
        record = mesh_root / "registered" / "atlas_mesh_registration.json"
        registration = json.loads(record.read_text()) if record.is_file() else None
        if (registration is None or registration.get("digest") != canonical_digest(registration, digest_field="digest")
                or registration["source_glb_sha256"] not in {row["sha256"] for row in mesh["outputs"]}):
            raise ValueError("atlas_bundle_mesh_unregistered")
    splat = None
    if splat_root is not None:
        from .atlas_native_splat import load_splat_manifest
        splat = load_splat_manifest(splat_root)
        if (splat.get("digest") != canonical_digest(splat, digest_field="digest")
                or splat["pose_result_digest"] != result["digest"]):
            raise ValueError("atlas_bundle_splat_manifest_invalid")
    manifest["generated_result_digests"] = [value["digest"] for _, value in generated]
    parts = {"pose": result["digest"], "generated": manifest["generated_result_digests"]}
    if splat is not None:
        parts["splat"] = manifest["splat_manifest_digest"] = splat["digest"]
    if mesh is not None:
        parts["mesh"] = manifest["mesh_result_digest"] = mesh["digest"]
        parts["mesh_registration"] = manifest["mesh_registration_digest"] = registration["digest"]
    if analysis_paths:
        parts["analysis"] = [_sha256_file(p) for p in analysis_paths]
    identity = canonical_digest(parts)
    scene = bool(generated) or splat is not None or mesh is not None
    target = output_root / (("atlas-scene-" + identity[7:23]) if scene else ("atlas-pose-" + result["digest"][7:23]))
    if target.is_symlink():
        raise ValueError("atlas_bundle_output_symlink")
    target.mkdir(parents=True, exist_ok=True)
    binary_members: dict[tuple[str, int], Path] = {}

    def copy(source, relative):
        path = target / relative
        if path.is_symlink():
            raise ValueError("atlas_bundle_output_symlink")
        path.parent.mkdir(parents=True, exist_ok=True)
        digest, size = _sha256_file(source), source.stat().st_size
        key = digest, size
        duplicate = binary_members.get(key) if relative.suffix.lower() != ".json" else None
        if duplicate is not None and (duplicate.is_symlink() or not duplicate.is_file()
                or duplicate.stat().st_size != size or _sha256_file(duplicate) != digest):
            duplicate = None
        _replace_bundle_member(source, path, duplicate=duplicate)
        if path.stat().st_size != size or _sha256_file(path) != digest:
            raise ValueError("atlas_bundle_copy_changed")
        if relative.suffix.lower() != ".json":
            binary_members[key] = path
        manifest["assets"].append({"path": str(relative), "sha256": digest, "bytes": size})

    def copy_task(root, value, folder, label):
        for row in value["outputs"]:
            source = root / row["path"]
            if (not source.resolve().is_relative_to(root.resolve()) or source.is_symlink()
                    or source.stat().st_size != row["bytes"] or _sha256_file(source) != row["sha256"]):
                raise ValueError(f"atlas_bundle_{label}_output_changed")
            copy(source, folder / Path(row["path"]).name)
        copy(root / "result.json", folder / "result.json")

    for index, (root, value) in enumerate(generated):
        copy_task(root, value, Path("generated") / str(index), "generated")
    if mesh is not None:
        copy_task(mesh_root, mesh, Path("mesh"), "mesh")
        for row in registration["outputs"]:
            source = mesh_root / "registered" / row["path"]
            if source.is_symlink() or _sha256_file(source) != row["sha256"]:
                raise ValueError("atlas_bundle_mesh_output_changed")
            copy(source, Path("mesh/registered") / row["path"])
        copy(mesh_root / "registered" / "atlas_mesh_registration.json", Path("mesh/registered/atlas_mesh_registration.json"))
        manifest.update(mesh_available=True, mesh_frame=registration["glb_frame"],
                        supported_mesh_path="mesh/registered/" + registration["outputs"][0]["path"])
    if splat is not None:
        for row in splat["outputs"]:
            source = splat_root / row["path"]
            if (source.is_symlink() or not source.resolve().is_relative_to(splat_root.resolve())
                    or _sha256_file(source) != row["sha256"]):
                raise ValueError("atlas_bundle_splat_output_changed")
            copy(source, Path("splat") / row["path"])
        from .atlas_native_splat import splat_manifest_path
        manifest_path = splat_manifest_path(splat_root)
        copy(manifest_path, Path("splat") / manifest_path.name)
        if splat.get("task") == "posedImages2Splats":
            native = _completed_task(splat_root / "native", source_digest, "splat")
            if native["digest"] != splat.get("native_result_digest"):
                raise ValueError("atlas_bundle_native_splat_receipt_changed")
            copy_task(splat_root / "native", native, Path("evidence/refined_splat"), "splat")
            copy(splat_root / "native/atlas_pose/submission.json", Path("evidence/refined_splat_submission.json"))
            copy(splat_root / "input_binding.json", Path("evidence/refined_splat_inputs.json"))
            manifest["splat_native_result_digest"] = native["digest"]
        manifest.update(splat_available=True, splat_kind=splat["splat_kind"])

    for index, row in enumerate(receipt["input_usage"]["frames"]):
        for asset in row["outputs"]:
            source = pose_root / asset["path"]
            copy(source, Path("views") / f'{index:03d}-{asset["role"]}{_suffix(source.read_bytes()[:16])}')
        camera = result["operation"]["response"]["frames"][index]["camera"]
        camera_path = target / "views" / f"{index:03d}-camera.json"
        camera_path.write_text(json.dumps(camera, indent=2) + "\n")
        manifest["assets"].append({"path": str(camera_path.relative_to(target)), "sha256": _sha256_file(camera_path),
                                   "bytes": camera_path.stat().st_size})
    # Source frames stay with the job (owner decision D5): the views above are
    # the provider's own posed images, so the download carries no input copies.
    copy(pose_root / "atlas_posed_rgbd_result.json", Path("evidence/atlas_posed_rgbd_result.json"))
    if (pose_root / "atlas_view_selection.json").exists():
        copy(pose_root / "atlas_view_selection.json", Path("evidence/atlas_view_selection.json"))
    for index, source in enumerate(analysis_paths):
        copy(source, Path("analysis") / f"{index:03d}-{source.name}")
    preserved_views = []
    by_digest = {row["sha256"]: row["path"] for row in manifest["assets"]}
    for source in analysis_paths:
        if source.name != "preservation.json":
            continue
        value = json.loads(source.read_text())
        if (value.get("schema_version") != "atlas_scene_preservation.v1"
                or value.get("digest") != canonical_digest(value, digest_field="digest")
                or value.get("known_pixels_changed") != 0 or value.get("source_geometry_modified") is not False
                or value.get("native_outputs_modified") is not False
                or value.get("native_result_digest") not in manifest["generated_result_digests"]):
            raise ValueError("atlas_bundle_preservation_receipt_invalid")
        try:
            roles = {row["role"]: by_digest[row["sha256"]] for row in value["assets"]}
            preserved_views.append({"frame_id": value["frame_id"],
                "image_path": roles["composite_rgb"], "known_mask_path": roles["known_pixels"],
                "generated_geometry_mask_path": roles["generated_geometry_mask"],
                "context_image_path": by_digest[value["context_image_digest"]],
                "native_result_digest": value["native_result_digest"], "native_frame_index": value["native_frame_index"]})
        except KeyError:
            raise ValueError("atlas_bundle_preservation_asset_missing") from None
    if preserved_views:
        manifest.update(scene_views=preserved_views, approved_context_pixels_preserved=True,
                        generated_geometry_region="unobserved_only")
    if splat is not None and splat.get("generated_content"):
        manifest["generated_background_splat"] = True
        manifest["visual_review_digest"] = splat.get("visual_review_digest")
    manifest["digest"] = canonical_digest(manifest, digest_field="digest")
    if (target / "manifest.json").is_symlink():
        raise ValueError("atlas_bundle_output_symlink")
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = target.with_suffix(".zip")
    if archive.is_symlink():
        raise ValueError("atlas_bundle_output_symlink")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for relative in sorted([row["path"] for row in manifest["assets"]] + ["manifest.json"]):
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            zipped.writestr(info, (target / relative).read_bytes())
    return {"directory": str(target), "zip_path": str(archive), "zip_sha256": _sha256_file(archive),
            "manifest_digest": manifest["digest"], "operation_id": manifest["operation_id"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    print(json.dumps(export_pose_bundle(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
