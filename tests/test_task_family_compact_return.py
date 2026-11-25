"""A family preparation leaves with its source tracks compact, and every hop it takes writes the plain lists' bytes.

``prepare_task_family`` wrote its record with the codec and then returned it as plain lists:
every run of every target's whole-video source track became a dict of about 256 bytes. On the
synthetic 60 s 1080p job (``tests.meta_sam31_synthetic``, 2.4 M runs) the family step grew peak
memory by 1.48 GiB. It returns the record compact; the revision binding, the handoff record,
the stage's manifest and its checks, and the run's stage snapshot read and write it with the
codec, to the bytes and digests the plain lists give.
"""
from __future__ import annotations

import copy
import json
import tracemalloc
from pathlib import Path

import numpy as np

from blueprint_pipeline import website_preparation_revision as revisions
from blueprint_pipeline import website_task_target_choice as choices
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.run_e2e import _json_safe_stage_result_snapshot
from blueprint_pipeline.task_family_preparation import prepare_task_family
from blueprint_pipeline.website_track_mask_codec import RunLengths, canonical_track_digest, plain_runs
from tests.test_meta_sam31_memory import written
from tests.test_website_preparation_revision import preparation_request, trusted_issuer  # noqa: F401


def _runs(count: int, offset: int = 0) -> RunLengths:
    return RunLengths(np.arange(count, dtype=np.int64) * 7 + offset, np.full(count, 3, dtype=np.int64))


def _family_inputs(tmp_path: Path, *, runs: int) -> dict:
    """A family job's inputs whose three targets each hold one whole-video track of ``runs`` compact runs."""
    geometry = {"schema_version": "website_source_frames.v1", "frames": [], "binding": {"source_video_digest": None}}
    geometry["digest"] = canonical_digest(geometry, digest_field="digest")
    targets = [{"target_id": f"target-{index}", "semantic_label": "shelf",
                "source_track": {"track_id": f"track-{index}", "observations": [
                    {"source_frame_id": "decoded-000000000", "width": 1920, "height": 1080,
                     "runs": _runs(runs, index)}]},
                "track": {"track_id": f"track-{index}", "observations": []}} for index in range(3)]
    masks = {"schema_version": "website_task_masks.v1", "targets": targets, "source_geometry_digest": geometry["digest"]}
    masks["digest"] = canonical_track_digest(masks, digest_field="digest")
    scene = tmp_path / "scene"
    scene.mkdir(parents=True)
    (scene / "splat.ply").write_bytes(b"synthetic splat stand-in")
    (scene / "collision.glb").write_bytes(b"synthetic collision stand-in")
    base_scene = {"splat_path": str(scene / "splat.ply"), "splat_digest": _sha256_file(scene / "splat.ply"),
                  "splat_binding_id": "synthetic-splat", "collision_mesh_path": str(scene / "collision.glb"),
                  "collision_mesh_digest": _sha256_file(scene / "collision.glb"), "up_axis": "Z",
                  "meters_per_unit": 1.0}
    context = {"capture_id": "synthetic-capture", "scene_id": "synthetic-scene", "task_family": "button_press",
               "description": "Press the button on the end shelf.",
               "capture_rights": {"derived_scene_generation_allowed": True}}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    return {"task_context": context, "task_masks": masks, "removal_manifest": {"entries": []},
            "source_geometry": geometry, "base_scene": base_scene}


def test_the_family_preparation_returns_its_tracks_compact_and_writes_the_plain_bytes(tmp_path):
    inputs = _family_inputs(tmp_path, runs=500)
    preparation = prepare_task_family(**inputs, output_root=tmp_path / "family")
    tracks = [row["runs"] for target in preparation["task_evaluation_inputs"]["source_targets"]
              for row in target["source_track"]["observations"]]
    assert tracks and all(type(runs) is RunLengths for runs in tracks)
    plain = plain_runs(preparation)
    assert (tmp_path / "family" / "preparation.json").read_text() == written(plain)
    assert preparation["digest"] == canonical_digest(plain, digest_field="digest")


def test_the_family_preparation_holds_no_dict_per_run(tmp_path):
    """Main returned ``plain_runs`` of the record: about 256 traced bytes a run."""
    runs = 400_000
    inputs = _family_inputs(tmp_path, runs=runs)
    tracemalloc.start()
    try:
        preparation = prepare_task_family(**inputs, output_root=tmp_path / "family")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert preparation["status"] == "needs_input"
    assert peak < 16 * 3 * runs + (16 << 20), peak


def test_a_corrected_preparation_is_bound_with_its_tracks_compact(tmp_path, monkeypatch):
    monkeypatch.setattr(choices, "correction_binding_digest", lambda _: "sha256:" + "a" * 64)
    preparation = {"intake_request": copy.deepcopy(preparation_request()), "status": "intake_ready",
                   "task_evaluation_inputs": {"source_targets": [{"runs": _runs(300)}]}}
    preparation["digest"] = canonical_track_digest(preparation, digest_field="digest")
    bound = revisions.bind_preparation(preparation, output_root=revisions.workspace(tmp_path), capture_root=tmp_path)
    assert type(bound["task_evaluation_inputs"]["source_targets"][0]["runs"]) is RunLengths
    plain = plain_runs(bound)
    assert bound["digest"] == canonical_digest(plain, digest_field="digest")
    assert (revisions.workspace(tmp_path) / "preparation.json").read_text() == written(plain)


def test_the_run_snapshot_keeps_compact_runs_as_the_plain_lists_they_equal():
    compact = {"status": "completed", "website_scene_preparation": {"source_targets": [{"runs": _runs(40)}]}}
    assert _json_safe_stage_result_snapshot(compact) == _json_safe_stage_result_snapshot(plain_runs(compact))
    assert json.dumps(_json_safe_stage_result_snapshot(compact)).count('"start"') == 40
    # A job's whole tracks are past the snapshot's bound, as their plain lists were.
    assert _json_safe_stage_result_snapshot({"runs": _runs(50_000)}) is None
