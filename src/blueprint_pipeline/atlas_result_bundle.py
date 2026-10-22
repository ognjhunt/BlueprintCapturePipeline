"""Portable, digest-bound downloads of an admitted Atlas pose run (ADP-040).

This exports observations, never manufactures a world, splat or qualified mesh.
Provider read URLs often have no suffix; identify the retained bytes without
renaming or rewriting the authoritative run's files.
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file
from .website_atlas_pose import load_pending_atlas_pose_inputs, verify_atlas_pose_receipt, _safe_id


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


def export_pose_bundle(*, pose_root: Path, output_root: Path, generated_roots=()) -> dict:
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
    generated = []
    for root in generated_roots:
        value = json.loads((root / "result.json").read_text())
        if (value.get("status") != "completed" or value.get("digest") != canonical_digest(value, digest_field="digest")
                or value["binding"]["source_video_digest"] != result["binding"]["source_video_digest"]):
            raise ValueError("atlas_bundle_generated_result_invalid")
        state = json.loads((root / "atlas_pose/submission.json").read_text())
        if state.get("status") != "completed" or state.get("result_digest") != value["digest"]:
            raise ValueError("atlas_bundle_generated_receipt_invalid")
        generated.append((root, value))
    manifest["generated_result_digests"] = [value["digest"] for _, value in generated]
    identity = canonical_digest({"pose": result["digest"], "generated": manifest["generated_result_digests"]})
    target = output_root / (("atlas-scene-" + identity[7:23]) if generated else ("atlas-pose-" + result["digest"][7:23]))
    if target.is_symlink():
        raise ValueError("atlas_bundle_output_symlink")
    target.mkdir(parents=True, exist_ok=True)

    def copy(source, relative):
        path = target / relative
        if path.is_symlink():
            raise ValueError("atlas_bundle_output_symlink")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)
        manifest["assets"].append({"path": str(relative), "sha256": _sha256_file(path), "bytes": path.stat().st_size})

    for index, (root, value) in enumerate(generated):
        for row in value["outputs"]:
            source = root / row["path"]
            if (not source.resolve().is_relative_to(root.resolve()) or source.is_symlink()
                    or source.stat().st_size != row["bytes"] or _sha256_file(source) != row["sha256"]):
                raise ValueError("atlas_bundle_generated_output_changed")
            copy(source, Path("generated") / str(index) / Path(row["path"]).name)
        copy(root / "result.json", Path("generated") / str(index) / "result.json")

    for index, row in enumerate(receipt["input_usage"]["frames"]):
        for asset in row["outputs"]:
            source = pose_root / asset["path"]
            copy(source, Path("views") / f'{index:03d}-{asset["role"]}{_suffix(source.read_bytes()[:16])}')
        camera = result["operation"]["response"]["frames"][index]["camera"]
        camera_path = target / "views" / f"{index:03d}-camera.json"
        camera_path.write_text(json.dumps(camera, indent=2) + "\n")
        manifest["assets"].append({"path": str(camera_path.relative_to(target)), "sha256": _sha256_file(camera_path),
                                   "bytes": camera_path.stat().st_size})
        copy(Path(inputs["frames"][index]["image_path"]), Path("inputs") / (_safe_id(row["frame_id"]) + ".png"))
    copy(pose_root / "atlas_posed_rgbd_result.json", Path("evidence/atlas_posed_rgbd_result.json"))
    if (pose_root / "atlas_view_selection.json").exists():
        copy(pose_root / "atlas_view_selection.json", Path("evidence/atlas_view_selection.json"))
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
