"""A retained complete background reaches the real scene reader without a provider."""
import json
from pathlib import Path
import shutil
import zipfile

import pytest

from blueprint_pipeline.atlas_retained_handoff import retain_background
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.website_atlas_scene_source import atlas_scene_source
from tests.test_website_atlas_scene_source import job_fixture, digest, write, SHA


def completed(tmp_path):
    job = job_fixture(tmp_path)
    root = job / "automation"
    result = json.loads((job / "result.json").read_text())
    portable = root / "downloads/scene"
    pairs = [(root / "background_splat", portable / "splat"),
             (root / "background_mesh/registered", portable / "mesh/registered")]
    for source, target in pairs:
        shutil.copytree(source, target)
    pose = root / "fallback_pose-v2/atlas_posed_rgbd_result.json"
    (portable / "evidence").mkdir()
    shutil.copyfile(pose, portable / "evidence/atlas_posed_rgbd_result.json")
    posed = json.loads(pose.read_text())
    manifest = digest({"schema_version": "atlas_download_bundle.v1", "source_video_digest": SHA,
        "pose_result_digest": posed["digest"], "splat_available": True, "mesh_available": True,
        "splat_manifest_digest": result["automation"]["background"]["splat_manifest_digest"],
        "mesh_registration_digest": json.loads((portable / "mesh/registered/atlas_mesh_registration.json").read_text())["digest"],
        "assets": [{"path": str(path.relative_to(portable)), "sha256": _sha256_file(path)}
                   for path in sorted(portable.rglob("*")) if path.is_file()]})
    write(portable / "manifest.json", manifest)
    archive = portable.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w") as zipped:
        for path in portable.rglob("*"):
            if path.is_file():
                zipped.write(path, str(path.relative_to(portable)))
    result["automation"]["background"]["download"] = {
        "directory": str(portable), "manifest_digest": manifest["digest"],
        "zip_path": str(archive), "zip_sha256": _sha256_file(archive)}
    write(root / "result.json", digest(result["automation"]))
    write(root / "background/prepared_background.json", digest({
        "binding": {"source_video_digest": SHA, "task_context_digest": canonical_digest({"context_digest": "context"})}}))
    return job, root, result


def test_completed_background_adoption_is_idempotent_and_readable_by_object_preparation(tmp_path):
    job, root, result = completed(tmp_path)
    original = (root / "result.json").read_bytes()
    options = dict(automation_root=root, job_dir=job, source_video_digest=SHA, task_context={"context_digest": "context"})
    adopted = retain_background(**options)
    assert retain_background(**options) == adopted
    assert (root / "result.json").read_bytes() == original
    assert adopted["retained_result_digest"] == json.loads(original)["digest"]
    result["automation"] = adopted
    write(job / "result.json", digest(result))
    write(job / "preview_job.json", {"state": "ready", "result_digest": result["digest"]})
    ready = atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
    assert ready["status"] == "ready" and ready["alignment"] == "anchored"
    # The scene is a levelled copy of the adopted background, never the baseline.
    levelling = ready["base_scene"]["levelling"]
    for key in ("provider_splat_path", "provider_collision_mesh_path"):
        assert Path(levelling[key]).is_relative_to(job / "automation/retained")


@pytest.mark.parametrize("changed", ["context", "asset", "archive"])
def test_changed_retained_inputs_refuse_before_adoption(tmp_path, changed):
    job, root, result = completed(tmp_path)
    if changed == "asset":
        (root / "downloads/scene/splat/atlas_depth_splat.ply").write_bytes(b"changed")
    if changed == "archive":
        (root / "downloads/scene.zip").write_bytes(b"changed")
    with pytest.raises(ValueError, match="atlas_retained_handoff_"):
        retain_background(automation_root=root, job_dir=job, source_video_digest=SHA,
                          task_context={"context_digest": "wrong" if changed == "context" else "context"})
    assert not (job / "automation/retained").exists()
