"""Authored surfaces retain image checks without claiming captured-site proof."""
import json

import numpy as np
import pytest

from blueprint_pipeline.common import sha256_file
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_packet import materialize_native_task_arena_packet
from blueprint_pipeline.native_task_arena_construction_worker import _camera_snapshot
from blueprint_pipeline.native_task_camera_observability import (
    NativeTaskCameraObservabilityError,
    validate_native_task_policy_start_camera_observability,
)
from blueprint_pipeline.native_task_nurec_render_setup import camera_site_appearance_required
from tests.test_native_task_arena_packet import _request
from tests.test_native_task_arena_construction_worker import _FakeCameraData, _FakeEnv


@pytest.mark.parametrize("authored,geometry", [(True, True), (False, True), (True, False)])
def test_packet_seals_only_explicit_authored_geometry_scope(tmp_path, authored, geometry):
    evidence = tmp_path / "evidence"
    request = _request(evidence, articulated=False)
    request["configured_task_template_adapter"] = {"source_documents": {"documents": {
        "support_plane": {
            "authority": "authored_development_surface" if authored else "registered_estimated_capture_and_reconstruction",
            "physical_scale_measured": False,
        },
        "definition": {"physical_world_truth_claimed": False},
    }}}
    if geometry:
        row = next(row for row in request["assets"] if row["semantic_role"] == "scene_appearance")
        path = evidence / "scene_appearance/development.usda"
        path.write_text('''#usda 1.0
(defaultPrim = "Root"
metersPerUnit = 1
upAxis = "Z")
def Xform "Root" {
    def Cube "Surface" {
        double size = 10
    }
}
''')
        row["filename"] = path.name
        row["source"].update(relative_path="scene_appearance/development.usda",
                             size_bytes=path.stat().st_size, sha256="sha256:" + sha256_file(path))
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    output = tmp_path / "packet"
    receipt = materialize_native_task_arena_packet(request=request, evidence_root=evidence, output_dir=output)
    plan = json.loads((output / "native_task_arena_scene_plan.v1.json").read_text())
    assert receipt["arena_scene_plan_digest"] == canonical_digest(plan, digest_field="plan_digest")
    assert camera_site_appearance_required(plan) is not (authored and geometry)
    if authored and geometry:
        assert plan["claim_boundary"]["captured_scene_evaluation_allowed"] is False
    else:
        assert "camera_scene_scope" not in plan


def test_development_scope_needs_both_explicit_claim_ceiling_and_geometry():
    plan = {"camera_scene_scope": "authored_development_surface"}
    assert camera_site_appearance_required(plan)
    plan["appearance_frame_alignment"] = {"representation": "usd_geometry"}
    assert camera_site_appearance_required(plan)
    plan["claim_boundary"] = {"captured_scene_evaluation_allowed": False}
    assert not camera_site_appearance_required(plan)


@pytest.mark.parametrize("blank,missing_object", [(False, False), (True, False), (False, True)])
def test_plain_surface_keeps_real_frame_and_target_requirements(tmp_path, blank, missing_object):
    rgb = np.full((64, 64, 3), 220, dtype=np.uint8)
    rgb[20:44, 20:44] = np.random.default_rng(7).integers(20, 140, (24, 24, 3), dtype=np.uint8)
    semantic = np.zeros((64, 64), dtype=np.int32)
    if not missing_object:
        semantic[20:44, 20:44] = 7
    if blank:
        rgb[:] = 0
    cameras = {}
    for role in ("external", "wrist"):
        camera = type("Camera", (), {})()
        camera.data = _FakeCameraData(rgb=rgb, semantic=semantic, labels={"7": {"class": "task_object"}})
        cameras[role] = camera
    snapshot = _camera_snapshot(env=_FakeEnv(cameras), camera_scene_names={k: k for k in cameras},
                                output_root=tmp_path, snapshot_id="reset", site_appearance_render_expected=False)
    construction = {"camera_snapshots": [snapshot]}
    for row in snapshot["cameras"]:
        assert row["observability"]["site_appearance_claimed"] is False
    if blank or missing_object:
        with pytest.raises(NativeTaskCameraObservabilityError):
            validate_native_task_policy_start_camera_observability(construction, site_appearance_render_expected=False)
    else:
        assert validate_native_task_policy_start_camera_observability(
            construction, site_appearance_render_expected=False)["passed"]
    # Default captured-site admission must still refuse these same frames.
    with pytest.raises(NativeTaskCameraObservabilityError):
        validate_native_task_policy_start_camera_observability(construction)
