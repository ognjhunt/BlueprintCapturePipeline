"""Build the synthetic records of ``tests/fixtures/replay_set_v1`` from the existing test builders.

Every record here is synthetic: it is made by the same builders the replay tools' own unit tests use
(``tests/test_task_evaluation_downstream_replay.website_job`` for a rigid pick-and-place job,
``tests/test_website_navigation_intake`` for a navigation job, ``tests/test_atlas_preview_replay`` for an
Atlas preview job, ``tests/atlas_level_room_fixture`` and ``tests/test_atlas_mesh_registration`` for a
reconstruction stage's splat and mesh). Frames are seeded noise images and the video is a placeholder:
none of it comes from a host, a capture, a clip or a retained job.

The records name absolute paths (a launch receipt names its result by path, and that path is under the
receipt's digest), so each record is built at ``<materialize_root>/<record_id>/host`` and the replay set
runner (``python -m blueprint_pipeline.replay_set``) puts it back at exactly that path before replaying it.

Rebuild (rewrites the records and the manifest; review the diff)::

    PYTHONPATH=src python -m tests.replay_set_v1_build --out tests/fixtures/replay_set_v1
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from collections.abc import Callable
from pathlib import Path
from typing import Any

from blueprint_pipeline.replay_set import (
    CLOCK_FIELD,
    DEFAULT_MATERIALIZE_ROOT,
    GZIP_SUFFIX,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA,
    _locked,
)

DOWNSTREAM = "task_evaluation_downstream_replay"
ATLAS_PREVIEW = "atlas_preview_replay"
STAGE = "atlas_reconstruction_stage_replay"
PROVENANCE = ("synthetic: built by tests/replay_set_v1_build.py from the replay tools' unit-test builders; "
              "no real capture, clip or job record is in it")
# JSON, PLY and GLB files larger than this are committed gzipped (deterministically) and decompressed
# when the record is put back.
GZIP_ABOVE_BYTES = 32 * 1024
GZIP_SUFFIXES = (".json", ".ply", ".glb")


def _rigid(host_parent: Path, *, trace: bool, drop: tuple[str, ...] = ()) -> dict[str, Any]:
    from tests.test_task_evaluation_downstream_replay import website_job

    runtime = None
    if not trace:
        # A scene without a retained probe trace needs the probe run; its compile reads the runtime sources.
        from tests.test_native_task_arena_bundle import _runtime_source_packet

        runtime = _runtime_source_packet(host_parent / "host" / "runtime-sources")
    job = website_job(host_parent, trace=trace, runtime_source=runtime)
    if runtime is not None:
        # The packet's receipt and archive are what the compile reads; the scratch git checkouts are not.
        shutil.rmtree(runtime.parents[1] / "runtime-source-repos", ignore_errors=True)
    for relative in drop:
        (job["host"] / relative).unlink()
    host = job["host"]
    return {"launch_run": job["run"].relative_to(host).as_posix(),
            "intent_dir": job["intent_dir"].relative_to(host).as_posix(),
            "capture_dir": job["capture"].relative_to(host).as_posix(),
            "registry": job["registry"].relative_to(host).as_posix(),
            "mirror": job["mirror"].relative_to(host).as_posix(),
            "webapp_snapshots": job["webapp"].relative_to(host).as_posix()}


def _navigation(host_parent: Path, *, submission_id: str) -> dict[str, Any]:
    """A navigation job: its intent, prepared by the real progression with no launch (kept as filmed)."""
    from tests.test_website_navigation_intake import _advance, _request, _staged

    host = host_parent / "host"
    host.mkdir(parents=True)
    staged, intent = _staged(host, _request(submission_id=submission_id))
    progress = _advance(host, staged, intent)
    if progress.get("status") != "completed":
        raise RuntimeError(f"navigation progression did not complete: {progress.get('status')}")
    shutil.rmtree(host / "factory-output", ignore_errors=True)
    return {"launch_run": "launch-runs/none", "intent_dir": staged["directory"].relative_to(host).as_posix(),
            "empty_dirs": ["launch-runs/none"]}


def _atlas_capture(host_parent: Path, *, request: str) -> SimpleNamespace:
    """An Atlas preview job as intake enqueues it: placeholder video, task context, allowance, survey.

    As ``tests/test_atlas_preview_replay.site`` builds it; the price review is dated the build day,
    which the replay clock keeps current.
    """
    from blueprint_pipeline import website_atlas_preview as preview
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file

    host = host_parent / "host"
    capture = host / f"storage/scenes/site-{request}/captures/walkthrough-{request}"
    (capture / "raw").mkdir(parents=True)
    (capture / "raw/walkthrough.mov").write_bytes(b"synthetic placeholder, not a video")
    (capture / "pipeline").mkdir()
    context = {"schema_version": "website_site_task_context.v1", "request_id": request, "scene_id": f"site-{request}",
               "capture_id": f"walkthrough-{request}", "purpose": "scene_preparation", "confirmed": False,
               "confirmed_at": None, "description": "Move the blue box to the tray"}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    (capture / "pipeline/website_task_context.json").write_text(json.dumps(context))
    routing = os.environ.get(preview.ROUTING_ENV)
    os.environ[preview.ROUTING_ENV] = "consented"
    try:
        job = preview.enqueue_preview(
            descriptor=SimpleNamespace(site_submission_id=request, metadata={}, capture_id=f"walkthrough-{request}",
                                       scene_id=f"site-{request}"),
            capture_root=capture, pipeline_dir=capture / "pipeline",
            source_video=lambda: capture / "raw/walkthrough.mov")
    finally:
        if routing is None:
            os.environ.pop(preview.ROUTING_ENV, None)
        else:
            os.environ[preview.ROUTING_ENV] = routing
    if not job or job["state"] != "processing":
        raise RuntimeError("the synthetic Atlas preview job was not enqueued")
    source = _sha256_file(capture / "raw/walkthrough.mov")
    today = datetime.now(timezone.utc).date().isoformat()
    review = {"rate_card_reference": "synthetic replay fixture, not a rate card", "reviewed_by": "replay set builder",
              "reviewed_on": today}
    policy = {"model": "gpt-6.1-sol", "maximum_review_cost_usd": .5, "pose_input_cost_usd": .01,
              "rgbd_view_cost_usd": .05, **review}
    configuration = {"schema_version": "website_atlas_configuration.v1", "source_video_digest": source,
                     "task_context_digest": context["context_digest"], "dispatch_authorized": False,
                     "authority": {"maximum_cost_usd": 5.0, "provider_retention": "account_does_not_retain"},
                     "policy": policy, "reserved_or_settled_cost_usd": 1.0, "remaining_cost_usd": 4.0}
    atlas = capture / "pipeline/atlas"
    (atlas / "configuration.json").write_text(json.dumps(configuration))
    return SimpleNamespace(host=host, capture=capture, atlas=atlas, job=atlas / preview.JOB_FILE, source=source,
                           review=review)


def _survey(atlas: Path, *, source: str, frames: int) -> dict[str, Any]:
    """A survey of ``frames`` seeded noise images: no two share a feature, so none overlaps another."""
    import numpy as np
    from PIL import Image

    from blueprint_pipeline.atlas_video_inputs import SURVEY_SCHEMA
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file

    root = atlas / "survey"
    (root / "source").mkdir(parents=True)
    rows = []
    for index in range(frames):
        path = root / "source" / f"frame-{index:02d}.png"
        pixels = np.random.default_rng(index).integers(0, 256, (48, 64), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        rows.append({"frame_id": f"f{index}", "timestamp_seconds": 0.5 * index, "source_image_path": str(path),
                     "source_image_digest": _sha256_file(path), "sharpness": 500.0, "clipped_fraction": 0.0,
                     "width": 64, "height": 48})
    survey = {"schema_version": SURVEY_SCHEMA, "capture_id": atlas.parents[1].name, "source_video_digest": source,
              "sample_count": frames, "sampling_policy": {}, "display_rotation_degrees": 0.0, "frames": rows,
              "heldout_frame_count": 0, "heldout_pixels_included": False, "claim_ceiling": "development_only"}
    survey["digest"] = canonical_digest(survey, digest_field="digest")
    (root / "atlas_video_survey.json").write_text(json.dumps(survey))
    return survey


def _coverage_gap(host_parent: Path) -> dict[str, Any]:
    """A reviewed selection whose task region no filmed view connects to the task object."""
    from blueprint_pipeline.atlas_video_inputs import REVIEW_SCHEMA

    site = _atlas_capture(host_parent, request="req-coverage")
    survey = _survey(site.atlas, source=site.source, frames=3)
    review = {"schema_version": REVIEW_SCHEMA, "survey_digest": survey["digest"], "moving_people_excluded": True,
              "same_scene_state": True, "reviewed_by": "replay set builder", "scene_state": "one static state",
              "static_intervals_seconds": [[0.0, 2.0]], "anchor_frame_id": "f0",
              "coverage_requirements": [
                  {"role": "task_object", "label": "blue box", "required_frame_ids": ["f0"]},
                  {"role": "task_region", "label": "tray", "required_frame_ids": ["f1"]},
                  {"role": "surrounding_scene", "label": "far wall", "required_frame_ids": ["f2"]}],
              "selection_budget": {"maximum_pose_cost_usd": 1.0, "input_cost_usd": .01, **site.review}}
    (site.atlas / "selection_binding.json").write_text(json.dumps({
        "survey_relative_path": "survey/atlas_video_survey.json", "review": review,
        "planner_receipt": {"model": "gpt-6.1-sol", "proposal": {"status": "proposed"}}}))
    return {"job": site.job.relative_to(site.host).as_posix()}


def _waiting_refusal(host_parent: Path) -> dict[str, Any]:
    """A job whose mesh registration refused on its retained inputs, as the controller records it."""
    from blueprint_pipeline import website_atlas_preview as preview
    from blueprint_pipeline.deterministic_refusal import DeterministicRefusal

    site = _atlas_capture(host_parent, request="req-refused")
    job = json.loads(site.job.read_text())
    refusal = DeterministicRefusal("atlas_mesh_not_registered_to_splat", step="atlas_mesh_registration", inputs={
        "source_glb_sha256": "sha256:" + "a" * 64, "splat_manifest_digest": "sha256:" + "b" * 64})
    failed = preview._failed_step({**job, "stage": "mesh"}, refusal, int(time.time()))
    if failed["state"] != "needs_input" or "deterministic_refusal" not in failed or not failed.get("retry_at_epoch"):
        raise RuntimeError("the controller did not record the deterministic refusal as a wait")
    site.job.write_text(json.dumps(failed))
    return {"job": site.job.relative_to(site.host).as_posix()}


def _small_world(host_parent: Path, *, terrain: bool, sh_rest: bool) -> dict[str, Any]:
    """Rolling open ground, or a floor and two walls, tilted 10 degrees and written as an Atlas splat and mesh.

    The ground follows ``tests/atlas_level_room_fixture.terrain_parts`` on a coarser grid: no level surface.
    """
    import numpy as np
    import trimesh

    from blueprint_pipeline.atlas_depth_splat import PLY_PROPERTIES
    from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
    from tests.atlas_level_room_fixture import _grid, provider_units, tilt

    world = host_parent / "host" / "world"
    world.mkdir(parents=True)
    rng = np.random.default_rng(0)
    if terrain:
        points, triangles = _grid((-4.0, 0.0, -4.0), (0.2, 0.0, 0.0), (0.0, 0.0, 0.2), 41, 41)
        x, z = points[:, 0], points[:, 2]
        points[:, 1] = 1.5 * (0.8 * np.sin(0.9 * x) * np.cos(0.7 * z) + 0.48 * np.sin(2.1 * x + 1.0)
                              + 0.32 * np.cos(1.7 * z + 0.4)) + rng.normal(0.0, 0.05, len(x))
        parts = [(points, triangles)]
        standing = [(0.0, 1.8, 0.0), (1.0, 1.9, 0.5), (-1.0, 1.7, -0.5)]
    else:
        parts = [_grid((-2.0, 0.0, -1.5), u, v, nu, nv) for u, v, nu, nv in (
            ((0.2, 0.0, 0.0), (0.0, 0.0, 0.2), 21, 16), ((0.2, 0.0, 0.0), (0.0, 0.2, 0.0), 21, 8),
            ((0.0, 0.0, 0.2), (0.0, 0.2, 0.0), 16, 8))]
        standing = [(0.0, 1.2, 0.0), (0.5, 1.35, 0.3), (-0.5, 1.5, -0.2)]
    vertices, faces, start = [], [], 0
    for points, triangles in parts:
        vertices.append(points)
        faces.append(triangles + start)
        start += len(points)
    rotation, mpu = tilt(10.0), 0.9
    units = provider_units(np.concatenate(vertices) + rng.normal(0.0, 0.003, (start, 3)), rotation=rotation,
                           meters_per_unit=mpu)
    trimesh.Trimesh(vertices=units, faces=np.concatenate(faces), process=False).export(world / "mesh.glb")
    up = rotation @ [0.0, 1.0, 0.0]
    quats = np.tile([1.0 + up[2], -up[1], up[0], 0.0], (start, 1))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    write_standard_3dgs_ply(SplatData(
        count=start, xyz=units.astype(np.float32), opacity=np.zeros(start, np.float32),
        f_dc=np.zeros((start, 3), np.float32), scales=np.tile(np.log([0.02, 0.02, 0.002]), (start, 1)).astype(
            np.float32), quats=quats.astype(np.float32), properties=PLY_PROPERTIES,
        sh_rest=np.zeros((start, 9), np.float32) if sh_rest else None), world / "splat.ply")
    cameras = provider_units(standing, rotation=rotation, meters_per_unit=mpu)
    (world / "cameras.json").write_text(json.dumps({"meters_per_unit": mpu, "camera_centres": cameras.tolist()}))
    return {"splat": "world/splat.ply", "mesh": "world/mesh.glb", "cameras": "world/cameras.json"}


def _unregistered_mesh(host_parent: Path) -> dict[str, Any]:
    """A mesh in no declared frame of its splat: offset along x, as ``tests/test_atlas_mesh_registration`` makes it."""
    from tests.test_atlas_mesh_registration import exported_glb, grid, splat_root

    root = host_parent / "host" / "registration"
    (root / "splat").mkdir(parents=True)
    splat_root(root / "splat", grid(-2.0, extent=0.1)[0])
    exported_glb(root / "mesh.glb", grid(-2.0, extent=0.1, x0=0.16), sign=(1.0, 1.0, 1.0))
    return {"splat_root": "registration/splat", "mesh": "registration/mesh.glb"}


RECORDS: list[dict[str, Any]] = [
    {"record_id": "rigid-pick-place-1080p-short", "job_kind": "rigid_pick_and_place",
     "input_variety": {"video_resolution": "1080p", "clip_length": "short"},
     "description": "A completed rigid pick-and-place preparation with its robot-free probe trace retained.",
     "build": lambda root: _rigid(root, trace=True),
     "until": "screening_plan",
     "expected_stop": {"kind": "completed", "stage": "screening_plan", "code": "until_screening_plan"}},
    {"record_id": "rigid-pick-place-4k-long", "job_kind": "rigid_pick_and_place",
     "input_variety": {"video_resolution": "4k", "clip_length": "long"},
     "description": "A completed rigid pick-and-place preparation with no probe trace: the probe must run, "
                    "so the chain stops where the probe lane would buy its GPU run.",
     "build": lambda root: _rigid(root, trace=False),
     "until": "screening_plan",
     "expected_stop": {"kind": "would_buy", "stage": "probe_enqueue",
                       "code": "paid_resource_allocator:gpu-canary:rigid-robot-free-probe"}},
    {"record_id": "rigid-pick-place-1080p-long-closure-missing", "job_kind": "rigid_pick_and_place",
     "input_variety": {"video_resolution": "1080p", "clip_length": "long"},
     "description": "A rigid pick-and-place launch whose post-teardown provider-zero receipt was not retained: "
                    "the join names the file to pull.",
     "build": lambda root: _rigid(root, trace=True, drop=(
         "launch-runs/scene-website-qualifying/post_teardown_provider_zero_receipt.json",)),
     "until": "screening_plan",
     "expected_stop": {"kind": "missing_input", "stage": "completion",
                       "code": "{host}/launch-runs/scene-website-qualifying/post_teardown_provider_zero_receipt.json"}},
    {"record_id": "navigation-4k-short", "job_kind": "navigation",
     "input_variety": {"video_resolution": "4k", "clip_length": "short"},
     "description": "A navigation job prepared with its scene kept as filmed (no launch). The post-completion "
                    "chain is the launched preparation's; its join does not adopt this job, by name.",
     "build": lambda root: _navigation(root, submission_id="walkthrough-nav-4k-short"),
     "until": "screening_plan",
     "expected_stop": {"kind": "refusal", "stage": "completion", "code": "preparation_completion_not_adopted"}},
    {"record_id": "navigation-1080p-long", "job_kind": "navigation",
     "input_variety": {"video_resolution": "1080p", "clip_length": "long"},
     "description": "A second navigation job, as above, under its own submission.",
     "build": lambda root: _navigation(root, submission_id="walkthrough-nav-1080p-long"),
     "until": "screening_plan",
     "expected_stop": {"kind": "refusal", "stage": "completion", "code": "preparation_completion_not_adopted"}},
    # Between "upload accepted" and "scene ready to configure" (2026-11-10 reconstruction changes).
    {"record_id": "atlas-coverage-gap-site-ask", "job_kind": "website_scene_preparation",
     "input_variety": {"video_resolution": "not_applicable", "clip_length": "not_applicable"},
     "description": "An Atlas preview job whose reviewed task region no filmed view connects to the task object. "
                    "Frame selection refuses before any upload and the job ends at once as needs_input with "
                    "the coverage code the website turns into its ask for a new video: no timed retry.",
     "tool": ATLAS_PREVIEW, "build": _coverage_gap, "until": "first_stop",
     "expected_stop": {"kind": "refusal", "stage": "queued", "code": "atlas_task_and_scene_coverage_gap",
                       "final_state": "needs_input"}},
    {"record_id": "atlas-deterministic-refusal-waits", "job_kind": "website_scene_preparation",
     "input_variety": {"video_resolution": "not_applicable", "clip_length": "not_applicable"},
     "description": "An Atlas preview job whose mesh registration refused on its retained inputs, recorded as a "
                    "deterministic refusal under this release's revision. The controller does not run the "
                    "step again: the job keeps waiting on its key, and nothing is bought.",
     "tool": ATLAS_PREVIEW, "build": _waiting_refusal, "until": "first_stop",
     "expected_stop": {"kind": "refusal", "stage": "mesh", "code": "atlas_mesh_not_registered_to_splat",
                       "final_state": "needs_input"}},
    {"record_id": "levelling-provider-up-estimate", "job_kind": "website_scene_preparation",
     "input_variety": {"video_resolution": "not_applicable", "clip_length": "not_applicable"},
     "description": "Rolling open ground with no level surface: levelling continues on the provider's +Y, "
                    "unverified, and names the estimate.",
     "tool": STAGE, "build": lambda root: _small_world(root, terrain=True, sh_rest=False), "until": "levelling",
     "expected_stop": {"kind": "estimate", "stage": "levelling", "code": "scene_estimate:level"}},
    {"record_id": "levelling-view-dependent-splat", "job_kind": "website_scene_preparation",
     "input_variety": {"video_resolution": "not_applicable", "clip_length": "not_applicable"},
     "description": "A floor and two walls whose splat carries view-dependent colour: levelled on the observed "
                    "floor with its base colour only, and the floor filled under the filmed path, both named.",
     "tool": STAGE, "build": lambda root: _small_world(root, terrain=False, sh_rest=True), "until": "levelling",
     "expected_stop": {"kind": "estimate", "stage": "levelling",
                       "code": "scene_estimate:floor,scene_estimate:splat_view_dependence_dropped"}},
    {"record_id": "mesh-registration-closest-frame-estimate", "job_kind": "website_scene_preparation",
     "input_variety": {"video_resolution": "not_applicable", "clip_length": "not_applicable"},
     "description": "A mesh that registers to no declared frame of its splat: the frame of smallest median "
                    "residual is taken, only faces near the splat are kept, and the collider is named an estimate.",
     "tool": STAGE, "build": _unregistered_mesh, "until": "mesh_registration",
     "expected_stop": {"kind": "estimate", "stage": "mesh_registration",
                       "code": "scene_estimate:collider_registration"}},
]


def _commit_tree(source: Path, target: Path) -> list[str]:
    """Copy a built host tree into the fixture, gzipping large JSON; return the committed relative paths."""
    written = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        data = path.read_bytes()
        if path.suffix in GZIP_SUFFIXES and len(data) > GZIP_ABOVE_BYTES:
            relative = relative.with_name(relative.name + GZIP_SUFFIX)
            data = gzip.compress(data, mtime=0)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        destination.chmod(path.stat().st_mode & 0o777)
        written.append(relative.as_posix())
    return written


def build(out: Path, *, materialize_root: Path = Path(DEFAULT_MATERIALIZE_ROOT)) -> dict[str, Any]:
    with _locked(materialize_root):  # a set may be replaying at these paths right now
        return _build(out, materialize_root=materialize_root)


def _build(out: Path, *, materialize_root: Path) -> dict[str, Any]:
    records_dir = out / "records"
    shutil.rmtree(records_dir, ignore_errors=True)
    entries = []
    for spec in RECORDS:
        record_id = spec["record_id"]
        parent = materialize_root / record_id
        shutil.rmtree(parent, ignore_errors=True)
        parent.mkdir(parents=True)
        inputs: Callable[[Path], dict[str, Any]] = spec["build"]
        built = inputs(parent)
        empty_dirs = built.pop("empty_dirs", [])
        files = _commit_tree(parent / "host", records_dir / record_id)
        shutil.rmtree(parent, ignore_errors=True)
        entries.append({
            "record_id": record_id, "synthetic": True, "provenance": PROVENANCE, "job_kind": spec["job_kind"],
            "description": spec["description"],
            "input_variety": {**spec["input_variety"], "read_by_replayed_code": False},
            "empty_dirs": empty_dirs, "file_count": len(files),
            "tools": [{"tool": spec.get("tool", DOWNSTREAM), "until": spec["until"], "inputs": built,
                       "expected_stop": spec["expected_stop"]}]})
    # The builders stamp consent and expiry from the wall clock, so the set is replayed at this instant.
    manifest = {"schema_version": MANIFEST_SCHEMA, "set_id": out.name, "synthetic": True,
                "materialize_root": str(materialize_root), CLOCK_FIELD: int(time.time()),
                "built_by": "PYTHONPATH=src python -m tests.replay_set_v1_build --out " + out.as_posix(),
                "records": entries}
    (out / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = build(args.out)
    print(f"built {len(manifest['records'])} synthetic records in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
