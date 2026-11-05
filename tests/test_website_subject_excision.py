"""A geometry-only job cuts its filmed subject out of the collider and ends at the robot fit check.

Step 2 of the spec-feasibility tier (owner, 2026-11-03). The scene keeps the subject
as filmed, and the surface under it was never seen; the cut fills that surface at the
height of the surface around it, so support lookups and robot checks meet the counter,
never the subject. The handoff then writes the capability verdict and the robot fit
table and stops: nothing is registered or enqueued.
"""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_subject_excision.py
#   src/blueprint_pipeline/website_task_preparation.py
#   src/blueprint_pipeline/website_scene_handoff.py
#   src/blueprint_pipeline/website_scene_runtime_inputs.py
#   src/blueprint_pipeline/website_task_context_contracts.py

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from blueprint_pipeline import website_native_background as background
from blueprint_pipeline import website_scene_handoff as handoff
from blueprint_pipeline import website_subject_excision as excision
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.website_support_geometry import same_support_surface, support_under
from blueprint_pipeline.website_task_context_contracts import GEOMETRY_ONLY_BLOCKER, retains_geometry_only_fit

_FRAMES = {"Z": (2, 1), "Y": (1, 1), "-Y": (1, -1)}
COUNTER = 0.72
#: The filmed object on the counter, upright (along, across, height), metres.
OBJECT = ((0.50, 0.50, COUNTER), (0.70, 0.62, 0.90))
#: Its video estimate: the bottom 2 cm into the counter (capture-447ae4f4 was 4 cm),
#: the top 1 cm short and each side 1 cm in.
ESTIMATE = ((0.51, 0.51, 0.70), (0.69, 0.61, 0.89))


def _place(points, up_axis):
    """Upright (along, across, height) points in a scene whose up is ``up_axis``."""
    up, sign = _FRAMES[up_axis]
    points = np.asarray(points, dtype=float)
    plan = [axis for axis in range(3) if axis != up]
    placed = np.empty_like(points)
    placed[..., plan[0]], placed[..., plan[1]], placed[..., up] = points[..., 0], points[..., 1], sign * points[..., 2]
    return placed


def _upright(points, up_axis):
    up, sign = _FRAMES[up_axis]
    points = np.asarray(points, dtype=float)
    plan = [axis for axis in range(3) if axis != up]
    return np.stack([points[..., plan[0]], points[..., plan[1]], sign * points[..., up]], axis=-1)


def _box(low, high, up_axis):
    corners = _place(np.array([low, high]), up_axis)
    return corners.min(axis=0).tolist(), corners.max(axis=0).tolist()


def _scan(tmp_path: Path, up_axis: str, *, counter: bool = True) -> dict:
    """A counter scanned with the object on it: the object hides the counter under it."""
    pieces = []
    if counter:
        steps = np.linspace(0.0, 1.2, 25)
        vertices = np.array([[a, b, COUNTER] for a in steps for b in steps])
        faces = []
        for i in range(24):
            for j in range(24):
                k = i * 25 + j
                faces.extend([[k, k + 25, k + 1], [k + 1, k + 25, k + 26]])
        faces = np.array(faces)
        centres = vertices[faces].mean(axis=1)
        (a0, b0, _), (a1, b1, _) = OBJECT
        hidden = (centres[:, 0] > a0) & (centres[:, 0] < a1) & (centres[:, 1] > b0) & (centres[:, 1] < b1)
        pieces.append(trimesh.Trimesh(vertices=vertices, faces=faces[~hidden], process=False))
    body = trimesh.creation.box(bounds=np.array(OBJECT))
    pieces.append(trimesh.Trimesh(vertices=body.vertices, faces=body.faces[body.face_normals[:, 2] > -0.5],
                                  process=False))
    mesh = trimesh.util.concatenate(pieces)
    path = tmp_path / f"scan-{up_axis.replace('-', 'minus')}.glb"
    trimesh.Trimesh(vertices=_place(mesh.vertices, up_axis), faces=mesh.faces, process=False).export(path)
    return {"collision_mesh_path": str(path), "collision_mesh_digest": _sha256_file(path),
            "collision_binding_id": "atlas-mesh-1", "up_axis": up_axis}


def _excise(tmp_path, base_scene, up_axis):
    up, sign = _FRAMES[up_axis]
    low, high = _box(*ESTIMATE, up_axis)
    return excision.excise_subject(base_scene=base_scene,
        collider=trimesh.load(base_scene["collision_mesh_path"], force="mesh", process=False),
        subject_min=low, subject_max=high, up=up, up_sign=sign, meters_per_unit=1.0, placement_residual_m=None,
        output_root=tmp_path / "out")


@pytest.mark.parametrize("up_axis", ["Z", "Y", "-Y"])
def test_the_subject_is_cut_out_and_the_surface_it_hid_is_filled_level_with_the_counter(tmp_path, up_axis):
    up, sign = _FRAMES[up_axis]
    base = _scan(tmp_path, up_axis)
    raw = trimesh.load(base["collision_mesh_path"], force="mesh", process=False)
    at_subject, elsewhere = _box(*ESTIMATE, up_axis), _box((0.1, 0.8, COUNTER), (0.3, 0.95, COUNTER + 0.1), up_axis)
    # As filmed, nothing under the subject carries it: the object hid the counter.
    assert support_under(raw, *at_subject, up=up, meters_per_unit=1.0, up_sign=sign) is None

    out = _excise(tmp_path, base, up_axis)
    assert out["status"] == "excised", out["record"]
    record = out["record"]
    assert sign * record["support_height_runtime_units"] == pytest.approx(COUNTER, abs=1e-6)
    assert record["fill"]["basis"] == "surrounding_surface_height" and record["fill"]["observed"] is False
    cut = out["collider"]
    # Nothing of the object is left, although its estimate sat 2 cm in the counter and fell short of it.
    upright = _upright(cut.triangles.reshape(-1, 3), up_axis)
    footprint = ((upright[:, 0] > 0.48) & (upright[:, 0] < 0.72) & (upright[:, 1] > 0.48) & (upright[:, 1] < 0.64))
    assert footprint.any() and upright[footprint, 2].max() == pytest.approx(COUNTER, abs=1e-6)
    # The lookups meet the counter under the subject, as one surface with the counter around it.
    under = support_under(cut, *at_subject, up=up, meters_per_unit=1.0, up_sign=sign)
    assert under is not None and sign * under["top_runtime_units"] == pytest.approx(COUNTER, abs=1e-6)
    beside = support_under(cut, *elsewhere, up=up, meters_per_unit=1.0, up_sign=sign)
    assert beside is not None and same_support_surface(under, beside)

    sealed = out["base_scene"]
    assert sealed["collision_mesh_path"] != base["collision_mesh_path"]
    assert sealed["collision_mesh_digest"] == _sha256_file(Path(sealed["collision_mesh_path"]))
    assert sealed["collision_binding_id"] == "website-collider-" + sealed["collision_mesh_digest"][7:39]
    assert sealed["subject_excision"]["source_collision_mesh_digest"] == base["collision_mesh_digest"]
    assert _sha256_file(Path(base["collision_mesh_path"])) == base["collision_mesh_digest"]
    receipt = json.loads(Path(record["receipt_path"]).read_text())
    assert receipt["digest"] == record["receipt_digest"] and receipt["subject_modelled"] is False


def test_a_retry_reuses_the_bound_cut(tmp_path, monkeypatch):
    base = _scan(tmp_path, "Z")
    first = _excise(tmp_path, base, "Z")
    monkeypatch.setattr(background, "exclude_box_interior",
                        lambda *a, **k: pytest.fail("A bound cut is reused, never redone"))
    again = _excise(tmp_path, base, "Z")
    assert again["status"] == "excised" and again["record"] == first["record"]
    assert again["base_scene"] == first["base_scene"]


def test_nothing_is_cut_without_a_surface_around_the_subject(tmp_path):
    base = _scan(tmp_path, "Z", counter=False)
    out = _excise(tmp_path, base, "Z")
    assert out == {"status": "unchanged", "record": {"schema_version": excision.REVISION, "status": "unchanged",
                                                     "reason": "surrounding_support_not_found"}}
    assert not (tmp_path / "out" / "collider").exists()


def test_the_surface_height_is_the_highest_band_the_window_reaches():
    # A counter top at 0.72 m and its lip at 0.66 m both lie within a 7.2 cm search
    # window; the subject rests on the top (capture-447ae4f4: 0.62-0.69 under a 0.72 top).
    def level(a, b, height):
        return [[a, b, height], [a + .01, b, height], [a, b + .01, height]]

    ring = [(0.45, 0.55), (0.75, 0.55), (0.60, 0.45), (0.60, 0.67)]
    triangles = np.array([level(a, b, height) for height in (COUNTER, 0.66) for a, b in ring])
    found = excision.surrounding_support_height(triangles, low=[0.5, 0.5, 0.0], high=[0.7, 0.62, 1.0], up=2,
                                                up_sign=1, bottom_height=0.68, window=0.072, meters_per_unit=1.0)
    assert found == (pytest.approx(COUNTER), 4)


def test_the_fit_check_runs_only_on_a_cut_rigid_pick_and_place_preparation(tmp_path, monkeypatch):
    def fail(**_):
        pytest.fail("The fit check never runs here")

    monkeypatch.setattr("blueprint_pipeline.website_scene_runtime_inputs.normalize_runtime_collision", fail)
    ready = {"blockers": [GEOMETRY_ONLY_BLOCKER], "intake_request": {"task": {"strategy": "pick_and_place"}},
             "subject": {"geometry_origin": "retained_in_captured_reconstruction"},
             "subject_excision": {"status": "excised"}}
    for change, reason in ((
            {"blockers": ["support_surface_not_found_under_subject", GEOMETRY_ONLY_BLOCKER]}, "preparation_blocked"),
            ({"intake_request": {"task": {"strategy": "push"}}}, "task_strategy_not_modelled"),
            ({"subject_excision": {"status": "unchanged", "reason": "surrounding_support_not_found"}},
             "subject_not_excised")):
        assert handoff.geometry_only_fit_check(preparation={**ready, **change}, base_scene={},
                                               native_root=tmp_path) == {"status": "not_run", "reason": reason}


def _cup_in_the_scan(args):
    """The compile fixture's collider as a geometry-only scan holds it: the cup still on the table.

    One frame's surface, as a scan has one (the fixture overlays both frames' grids).
    """
    from blueprint_pipeline.website_task_masks import estimate_target_bounds
    from tests.test_website_task_preparation import RUNTIME_ROTATION, RUNTIME_SCALE, RUNTIME_TRANSLATION, _grid

    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = RUNTIME_SCALE * RUNTIME_ROTATION, RUNTIME_TRANSLATION
    cup = estimate_target_bounds(args["task_masks"]["targets"][0]["track"], args["source_geometry"]["frames"],
                                 source_to_target=matrix)
    (x0, _, z0), (x1, y1, z1) = cup["minimum"], cup["maximum"]
    body = trimesh.creation.box(bounds=[[x0, 1.75, z0], [x1, y1, z1]])
    body = trimesh.Trimesh(vertices=body.vertices, faces=body.faces[body.face_normals[:, 1] > -0.5], process=False)
    points, faces = _grid({"frames": args["source_geometry"]["frames"][:1]})
    table = trimesh.Trimesh(vertices=(points @ RUNTIME_ROTATION.T) * RUNTIME_SCALE + RUNTIME_TRANSLATION,
                            faces=faces, process=False)
    path = Path(args["base_scene"]["collision_mesh_path"])
    trimesh.util.concatenate([table, body]).export(path)
    args["base_scene"]["collision_mesh_digest"] = _sha256_file(path)


def _hand_off(tmp_path, monkeypatch, args, **metadata):
    """The handoff of a capture whose geometry-only Atlas job finished with the subject in its scene."""
    (tmp_path / "pipeline").mkdir(exist_ok=True)
    context = args["task_context"]
    geometry = {**args["source_geometry"], "geometry_backend": "world_labs_atlas"}
    geometry["digest"] = canonical_digest(geometry, digest_field="digest")
    masks = {**args["task_masks"], "source_geometry_digest": geometry["digest"]}
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    monkeypatch.setattr("blueprint_pipeline.website_atlas_scene_source.atlas_scene_source", lambda **_: {
        "status": "ready", "base_scene": {**args["base_scene"], "provider": "world_labs_atlas"},
        "source_geometry": geometry, "alignment": "anchored", "source_pose_digest": "sha256:" + "1" * 64,
        "background_pose_digest": "sha256:" + "2" * 64, "subject_origin": "captured",
        "atlas_route": "geometry_only", "subject_in_reconstruction": True})
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry", lambda **_: masks)
    for name in ("register_website_preparation", "binding_root"):
        monkeypatch.setattr(f"blueprint_pipeline.website_scene_dispatch.{name}",
                            lambda **_: pytest.fail("A geometry-only scene is never registered"))
    monkeypatch.setattr("blueprint_pipeline.website_task_context.enqueue_website_prepared_scene",
                        lambda **_: pytest.fail("A geometry-only scene is never enqueued"))
    removal = tmp_path / "pipeline" / "removal.json"
    write_json(removal, args["removal_manifest"])
    return handoff.prepare_website_scene_handoff(descriptor={"capture_id": context["capture_id"],
        "scene_id": context["scene_id"], "metadata": {"site_task_context": context, **metadata}},
        clean_plate={"privacy_verified": True, "status": "noop", "task_masks": args["task_masks"],
            "source_geometry": None, "removal_manifest_path": str(removal)},
        capture_root=tmp_path, now=args["now"])


def test_a_geometry_only_scene_ends_at_the_fit_check_and_is_never_registered(tmp_path, monkeypatch):
    from tests.test_website_task_preparation import _arguments

    args = _arguments(tmp_path)
    _cup_in_the_scan(args)
    context = args["task_context"]
    result = _hand_off(tmp_path, monkeypatch, args, website_scene_execution_authority=args["spend"])

    assert result["atlas_route"] == "geometry_only" and result["status"] == "needs_input", result
    assert result["blockers"] == [GEOMETRY_ONLY_BLOCKER] and result["simulator_ready"] is False
    assert not {"source_registration", "website_intake_outbox", "runtime_inputs"} & set(result)
    fit = result["fit_check"]
    assert fit["status"] == "completed", fit
    verdict = json.loads(Path(fit["capability_verdict"]["path"]).read_text())
    assert verdict["verdict_digest"] == fit["capability_verdict"]["digest"]
    assert verdict["collision_sha256"] == fit["collision"]["digest"]
    table = json.loads(Path(fit["robot_fit_table"]["path"]).read_text())
    assert table["table_digest"] == fit["robot_fit_table"]["digest"] and sum(table["counts"].values()) > 0

    preparation = json.loads(Path(result["preparation_path"]).read_text())
    assert preparation["status"] == "needs_input" and preparation["blockers"] == [GEOMETRY_ONLY_BLOCKER]
    assert preparation["subject"]["geometry_origin"] == "retained_in_captured_reconstruction"
    assert preparation["subject_excision"]["status"] == "excised"
    assert preparation["recipe_plan"]["observed_appearance_object_removal"] == "not_performed_geometry_only"
    assert preparation["recipe_plan"]["collision_object_excision"] == "subject_box_excised_from_captured_reconstruction"
    # The cut collider is the one compile bound, and the fit check read it.
    sealed = json.loads(Path(result["base_scene_path"]).read_text())
    assert preparation["binding"]["collision_mesh_digest"] == sealed["collision_mesh_digest"]
    assert sealed["subject_excision"]["source_collision_mesh_digest"] == args["base_scene"]["collision_mesh_digest"]
    assert preparation["support"] is not None
    on_disk = json.loads((tmp_path / "pipeline/website_scene_preparation/handoff.json").read_text())
    assert on_disk == result and retains_geometry_only_fit(on_disk, context)


def test_a_family_task_on_the_geometry_only_route_is_recorded_not_assessed(tmp_path, monkeypatch):
    from tests.test_website_task_preparation import _arguments

    args = _arguments(tmp_path)
    monkeypatch.setattr("blueprint_pipeline.task_family_scoring.canonical_task_family", lambda _: "surface_wipe")
    monkeypatch.setattr(handoff, "compile_website_scene_preparation",
                        lambda **_: pytest.fail("The fit check models one rigid subject"))
    result = _hand_off(tmp_path, monkeypatch, args)
    assert result.get("fit_check") == {"status": "not_run", "reason": "task_family_not_modelled"}, result
    assert result["status"] == "needs_input" and result["blockers"] == [GEOMETRY_ONLY_BLOCKER]
    assert retains_geometry_only_fit(result, args["task_context"])


def _fit_handoff(context):
    value = {"schema_version": "website_scene_handoff.v1", "status": "needs_input",
             "scene_id": context["scene_id"], "capture_id": context["capture_id"],
             "blockers": [GEOMETRY_ONLY_BLOCKER], "atlas_route": "geometry_only",
             "simulator_ready": False, "visual_reconstruction_ready": True,
             "fit_check": {"status": "completed"}}
    value["digest"] = canonical_digest(value, digest_field="digest")
    return value


@pytest.mark.parametrize("fault", [None, "digest", "capture", "route", "status", "fit_check", "blocker",
                                   "source_registration", "website_intake_outbox", "runtime_inputs", "simulator"])
def test_only_an_unregistered_geometry_only_fit_check_is_a_completed_stage(fault):
    context = {"scene_id": "site-req1", "capture_id": "walkthrough-req1"}
    value = _fit_handoff(context)
    changes = {"capture": {"capture_id": "walkthrough-other"}, "route": {"atlas_route": "full"},
               "status": {"status": "intake_ready"}, "fit_check": {"fit_check": {"status": "failed"}},
               "blocker": {"blockers": ["support_surface_not_found_under_subject"]},
               "source_registration": {"source_registration": {"path": "x"}},
               "website_intake_outbox": {"website_intake_outbox": {"state": "forward_pending"}},
               "runtime_inputs": {"runtime_inputs": {"status": "awaiting_inputs"}},
               "simulator": {"simulator_ready": True}}
    if fault in changes:
        value.update(changes[fault])
        value["digest"] = canonical_digest(value, digest_field="digest")
    elif fault == "digest":
        value["digest"] = "sha256:" + "9" * 64
    assert retains_geometry_only_fit(value, context) is (fault is None)
