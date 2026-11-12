"""Room particles inside the placed task object or its marker are removed from the cameras' view.

The numbers are capture-447ae4f4's: the tray's collision box at its start pose, the marker
disc on the 0.8039 m collider top, and table particles about 2.5 cm under that plane.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline import scene_appearance_task_clearance as clearance

PLANE = 0.8038877905903447
TASK_SPEC = {
    "start_pose_world": [-0.4326211349137302, -0.3030958365897039, 0.8142599161339986, 0.0, 0.0, 0.0, 1.0],
    "subject_collision_bounds_scoring_frame_m": {"minimum": [-0.15668, -0.26056, -0.010372],
                                                  "maximum": [0.15668, 0.26056, 0.034556]},
    "visible_target_marker": {"non_colliding": True, "radius_m": 0.3900576605858041,
                              "schema_version": "native_task_target_marker.v1", "shape": "flat_green_disc",
                              "surface_position_world_m": [-0.22765085, 0.09984718, PLANE]},
}
UNDER_TRAY = [-0.43, -0.30, PLANE - 0.025]          # table particle reaching through the tray's plane
INSIDE_TRAY = [-0.43, -0.30, PLANE + 0.02]          # a particle inside the tray's own volume
UNDER_MARKER = [-0.10, 0.20, PLANE - 0.02]          # table particle under the marker
IN_FRONT = [-0.43, -0.30, PLANE + 0.30]             # a real thing 30 cm above the tray: in front of it
DEEP_TABLE = [-0.43, -0.30, PLANE - 0.05]           # table body further down: not inside
FAR = [1.5, 1.5, PLANE - 0.02]                      # the rest of the room


def _plan(points, sigma=None, opacity=None, volumes=None):
    points = np.asarray(points, float)
    sigma = np.full(len(points), 0.016) if sigma is None else np.asarray(sigma, float)
    opacity = np.full(len(points), 0.9) if opacity is None else np.asarray(opacity, float)
    return clearance.clearance_plan(positions_world=points, sigma_world=sigma, opacities=opacity,
                                    volumes=clearance.task_volumes(TASK_SPEC) if volumes is None else volumes)


def _with_filler(points, total=1000):
    """The given points, then the room far away, so the field is large enough for the share bound."""
    return np.asarray([*points, *([FAR] * (total - len(points)))], float)


def test_table_particles_reaching_through_are_pressed_down_keeping_their_place_in_the_table() -> None:
    plan = _plan(_with_filler([UNDER_TRAY, INSIDE_TRAY, UNDER_MARKER, IN_FRONT, DEEP_TABLE]))
    moved = plan["positions_world"]
    # Pressed: their extent now stays 1 mm under the plane; x and y unchanged.
    for index in (0, 1, 2):
        assert moved[index, 2] + clearance.SIGMA_REACH * 0.016 == pytest.approx(PLANE - clearance.CLEARANCE_M)
    assert moved[:3, :2].tolist() == np.asarray([UNDER_TRAY, INSIDE_TRAY, UNDER_MARKER])[:, :2].tolist()
    # A real thing above the tray and the table deeper down are untouched.
    assert moved[3].tolist() == IN_FRONT and moved[4].tolist() == DEEP_TABLE
    receipt = plan["receipt"]
    assert receipt["status"] == "pressed" and receipt["pressed_count"] == 3 and receipt["flags"] == []
    assert receipt["colour_and_opacity_changed"] is False and receipt["method"] == "pressed_down_keeping_colour"


def test_the_mark_is_its_own_outline_and_a_thin_band_only() -> None:
    edge = [-0.22765 + 0.395, 0.09985, PLANE - 0.02]    # 5 mm outside the mark's 0.39 m outline
    standing = [-0.10, 0.20, PLANE + 0.02]             # something real 2 cm up, standing on the mark
    plan = _plan(_with_filler([edge, standing]), sigma=[0.016, 0.001] + [0.016] * 998)
    assert plan["positions_world"][0].tolist() == edge and plan["positions_world"][1].tolist() == standing
    assert plan["receipt"]["status"] == "nothing_reaches"


def test_a_particle_that_would_move_too_far_is_left_and_counted() -> None:
    wide = [-0.43, -0.30, PLANE - 0.01]
    plan = _plan(_with_filler([wide]), sigma=[0.05] + [0.016] * 999)   # its extent reaches 14 cm up
    assert plan["positions_world"][0].tolist() == wide and plan["receipt"]["left_too_far_count"] == 1


def test_a_press_over_the_bound_is_flagged_not_done() -> None:
    plan = _plan(_with_filler([UNDER_TRAY] * 20))
    assert plan["receipt"]["status"] == "too_large_not_pressed"
    assert plan["receipt"]["flags"] == [clearance.FLAG_TOO_LARGE] and not plan["pressed"].any()


def test_bad_inputs_are_refused() -> None:
    with pytest.raises(clearance.SceneAppearanceClearanceError, match="volumes_missing"):
        _plan([FAR], volumes=[])
    with pytest.raises(clearance.SceneAppearanceClearanceError, match="positions_not_finite"):
        _plan([[np.nan, 0, 0]])
    with pytest.raises(clearance.SceneAppearanceClearanceError, match="values_out_of_range"):
        _plan([FAR], opacity=[1.5])
    assert clearance.task_volumes({}) == []


# --- The compiled packet: derived field, receipt, and the readers that consume it ---------------


def _particlefield(path: Path, points: np.ndarray, opacity: float = 0.9) -> Path:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    prim = stage.DefinePrim("/World/Gaussians", "ParticleField3DGaussianSplat")
    n = len(points)
    prim.CreateAttribute("positions", Sdf.ValueTypeNames.Point3fArray).Set(
        Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in points]))
    prim.CreateAttribute("scales", Sdf.ValueTypeNames.Float3Array).Set(Vt.Vec3fArray([Gf.Vec3f(0.01)] * n))
    prim.CreateAttribute("orientations", Sdf.ValueTypeNames.QuatfArray).Set(Vt.QuatfArray([Gf.Quatf(1)] * n))
    prim.CreateAttribute("opacities", Sdf.ValueTypeNames.FloatArray).Set(Vt.FloatArray([opacity] * n))
    stage.Save()
    return path


def _native_appearance(path: Path) -> dict:
    from blueprint_pipeline.task_evaluation_native_arena_episode_compiler import _appearance_render_backend, _sha256_and_size

    digest = _sha256_and_size(path)[0]
    return {"status": "nurec_reused_cached_official_particlefield", "path": str(path),
            "representation": "particlefield_3d_gaussian_splat",
            "source_configured_appearance_digest": "sha256:" + "1" * 64, "particlefield_digest": digest,
            "representation_conversion_performed": True, "exact_learned_arrays_preserved": True,
            "gaussian_field_quality": {"status": "qualified"},
            "appearance_render_backend": _appearance_render_backend(
                kind="nvidia_3dgrut_direct_nurec_transcode", source_digest="sha256:" + "1" * 64,
                particlefield_digest=digest, authoring_receipt_digest="sha256:" + "a" * 64,
                upstream_converter={"module": "threedgrut.export.scripts.transcode", "source_identity_verified": True,
                                    "source_revision": "a37ef721012dea0f29c0fcfff2d525023b4e854a"})}


def test_the_packet_carries_the_derived_field_its_receipt_and_a_backend_bound_to_it(tmp_path) -> None:
    from pxr import Usd

    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.native_task_arena_policy_canary_worker import appearance_render_backend_from_plan
    from blueprint_pipeline.task_evaluation_native_arena_episode_compiler import (
        _sha256_and_size, apply_task_volume_clearance, packet_appearance_variant,
    )

    source = _particlefield(tmp_path / "source.usdc", _with_filler([UNDER_TRAY, INSIDE_TRAY, IN_FRONT]))
    before = _sha256_and_size(source)
    appearance = apply_task_volume_clearance(_native_appearance(source), task_spec=TASK_SPEC, output_root=tmp_path)
    assert _sha256_and_size(source) == before                     # the source file is never changed
    derived = Path(appearance["path"])
    assert derived.parent == tmp_path / "native-appearance-clearance" and derived != source
    digest = _sha256_and_size(derived)[0]
    receipt = appearance["appearance_task_clearance"]
    assert receipt["status"] == "pressed" and receipt["pressed_count"] == 2
    assert receipt["source_particlefield_digest"] == before[0] and receipt["derived_particlefield_digest"] == digest
    stage = Usd.Stage.Open(str(derived))
    field = stage.GetPrimAtPath("/World/Gaussians")
    opacities = list(field.GetAttribute("opacities").Get())
    assert opacities == [pytest.approx(0.9)] * len(opacities)            # colour and opacity kept
    pressed = [list(point) for point in field.GetAttribute("positions").Get()][:3]
    assert pressed[0][2] < UNDER_TRAY[2] and pressed[1][2] < INSIDE_TRAY[2] and pressed[2] == pytest.approx(IN_FRONT)
    # The variant says what happened; the backend names the derived field and its digest still checks.
    variant = packet_appearance_variant(appearance)
    assert variant["exact_learned_arrays_preserved"] is False and variant["task_clearance"] == receipt
    backend = variant["render_backend"]
    assert backend["particlefield_digest"] == digest
    assert backend["backend_digest"] == canonical_digest(backend, digest_field="backend_digest")
    # The policy worker reads the packet's scene appearance by the derived file's digest and accepts it.
    plan = {"objects": [{"name": "scene_appearance", "semantic_role": "scene_appearance", "sha256": digest[7:]}],
            "appearance_frame_alignment": {"status": "aligned", "representation": "particlefield_3d_gaussian_splat"}}
    sealed = appearance_render_backend_from_plan(plan, packet_request={"appearance_variant": variant})
    assert sealed["derived_asset_digest"] == digest


def test_with_nothing_inside_the_packet_keeps_the_exact_field(tmp_path) -> None:
    from blueprint_pipeline.task_evaluation_native_arena_episode_compiler import (
        apply_task_volume_clearance, packet_appearance_variant,
    )

    source = _particlefield(tmp_path / "source.usdc", _with_filler([IN_FRONT]))
    original = _native_appearance(source)
    appearance = apply_task_volume_clearance(original, task_spec=TASK_SPEC, output_root=tmp_path)
    assert appearance["path"] == str(source) and appearance["appearance_render_backend"] == original["appearance_render_backend"]
    variant = packet_appearance_variant(appearance)
    assert variant["exact_learned_arrays_preserved"] is True
    assert variant["task_clearance"]["status"] == "nothing_reaches"
    assert not (tmp_path / "native-appearance-clearance").exists()
    # An appearance compiled before this rule (no receipt) reads exactly as it did.
    assert "task_clearance" not in packet_appearance_variant(original)
    assert packet_appearance_variant(original)["exact_learned_arrays_preserved"] is True


def _packet_request(variant, appearance_sha):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.native_task_arena_packet import REQUEST_SCHEMA_VERSION

    request = {"schema_version": REQUEST_SCHEMA_VERSION, "appearance_variant": variant,
               "assets": [{"semantic_role": "scene_appearance", "source": {"sha256": appearance_sha}}],
               "request_digest": ""}
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    return request


def test_a_pressed_field_says_it_was_mutated_and_passes_the_packet_gate_only_with_its_receipt(tmp_path) -> None:
    from blueprint_pipeline.native_task_arena_packet import NativeTaskArenaPacketError, _clone_request
    from blueprint_pipeline.task_evaluation_native_arena_episode_compiler import (
        apply_task_volume_clearance, packet_appearance_variant,
    )

    room = np.random.default_rng(3).uniform([-3, -3, 0], [3, 3, 2.5], (1999, 3))
    room = room[np.abs(room[:, 2] - PLANE) > 0.2]
    source = _particlefield(tmp_path / "source.usdc", np.vstack([[UNDER_TRAY], room]))
    appearance = apply_task_volume_clearance(_native_appearance(source), task_spec=TASK_SPEC, output_root=tmp_path)
    variant = packet_appearance_variant(appearance)
    quality, clearance = variant["gaussian_field_quality"], variant["task_clearance"]
    assert quality["learned_tensors_mutated"] is True
    assert quality["mutation"]["derived_particlefield_digest"] == clearance["derived_particlefield_digest"]
    derived = clearance["derived_particlefield_digest"]
    assert _clone_request(_packet_request(variant, derived))["appearance_variant"] == variant
    # A mutated field with no clearance receipt, or a receipt naming another file, is refused as today.
    for broken, sha in (({k: v for k, v in variant.items() if k != "task_clearance"}, derived),
                        (variant, "sha256:" + "0" * 64),
                        ({**variant, "task_clearance": {**clearance, "pressed_count": 2}}, derived)):
        with pytest.raises(NativeTaskArenaPacketError, match="particlefield_quality_missing_or_invalid"):
            _clone_request(_packet_request(broken, sha))


def test_an_unmoved_field_keeps_its_quality_exactly(tmp_path) -> None:
    from blueprint_pipeline.task_evaluation_native_arena_episode_compiler import apply_task_volume_clearance

    source = _particlefield(tmp_path / "source.usdc", _with_filler([IN_FRONT]))
    original = _native_appearance(source)
    appearance = apply_task_volume_clearance(original, task_spec=TASK_SPEC, output_root=tmp_path)
    assert appearance["gaussian_field_quality"] == original["gaussian_field_quality"]
