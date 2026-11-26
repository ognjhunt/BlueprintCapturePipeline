"""Several objects per job, stage 3: each free further object authored on the CPU, placed where it was filmed and
carried into the composed scene and the company's package as ``/World/Object_02`` onward, beside the scored object.

The fixture is ``tests.test_website_task_preparation``'s captured cup on its table, filmed with more objects beside it
(raised patches in the frames, absent from the collider, as objects taken out before reconstruction are). One-object
jobs are pinned byte for byte elsewhere (``tests/test_website_preparation_request.py``, ``tests/test_test_job_matrix``).
Offline: no provider, no network, nothing paid.
"""
# Covers: src/blueprint_pipeline/website_further_objects.py
# Covers: src/blueprint_pipeline/website_preparation_request.py
# Covers: src/blueprint_pipeline/website_task_preparation.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import website_build_plan as build_plan
from blueprint_pipeline import website_further_objects as further
from blueprint_pipeline import website_library_catalog as library_catalog
from blueprint_pipeline import website_object_library as library
from blueprint_pipeline import website_parametric_object as parametric
from blueprint_pipeline import website_task_preparation as preparation
from blueprint_pipeline.atlas_scene_estimates import preparation_record_flags
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from tests import test_website_task_preparation as fx

pytest.importorskip("pxr")

#: Where each object stands in the frames: the first image column of its four (rows 20 to 24, on the table).
COLUMNS = {"cup-1": 21, "carton-1": 3, "mug-1": 11, "static-1": 30, "artisan-1": 34}
ENTRY_ID = "gso:Test_Mug_Box@1"
CREDIT = {"entry_id": ENTRY_ID, "license": "CC-BY-4.0", "format": "obj", "library": "google_scanned_objects",
          "title": "Test Mug Box", "author": "Google", "library_title": "Google Scanned Objects",
          "item_url": "https://fuel.gazebosim.org/1.0/GoogleResearch/models/Test_Mug_Box/1/Test_Mug_Box.zip",
          "license_name": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/"}


def _box_obj(x: float, y: float, z: float) -> bytes:
    corners = [(sx * x / 2, sy * y / 2, sz * z) for sz in (0, 1) for sy in (-1, 1) for sx in (-1, 1)]
    quads = [(1, 3, 4, 2), (5, 6, 8, 7), (1, 2, 6, 5), (3, 7, 8, 4), (1, 5, 7, 3), (2, 4, 8, 6)]
    return ("\n".join([*(f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in corners),
                       *("f " + " ".join(str(i) for i in quad) for quad in quads)]) + "\n").encode()


def _track(column: int) -> dict:
    runs = [{"start": row * fx.WIDTH + column, "length": 4, "probability": 0.9} for row in range(20, 25)]
    return {"track_id": f"t{column}", "label": "object", "observations": [
        {"source_frame_id": "frame-0", "height": fx.HEIGHT, "width": fx.WIDTH, "runs": runs}]}


def _frame(root: Path, index: int) -> dict:
    """``fx._frame`` with every object of ``COLUMNS`` raised above the table in the depth."""
    depth = fx._depth()
    yy, xx = np.indices((5, 4))
    for column in COLUMNS.values():
        depth[20:25, column:column + 4] = 1.57 + 0.005 * (xx + yy)
    geometry = root / f"frame-{index}.npz"
    np.savez(geometry, depth_m=depth, valid_mask=np.ones_like(depth, dtype=bool))
    image = root / f"frame-{index}.png"
    Image.fromarray(np.full((fx.HEIGHT, fx.WIDTH, 3), 120, dtype=np.uint8)).save(image)
    pose = np.eye(4)
    pose[0, 3] = 0.05 * index
    return {"frame_id": f"frame-{index}", "timestamp_seconds": index * 0.5, "image_path": str(image),
            "image_digest": _sha256_file(image), "geometry_path": str(geometry), "geometry_digest": _sha256_file(geometry),
            "intrinsics": [[fx.FOCAL, 0, fx.WIDTH / 2], [0, fx.FOCAL, fx.HEIGHT / 2], [0, 0, 1]],
            "world_from_camera": pose.tolist(), "width": fx.WIDTH, "height": fx.HEIGHT}


def _entry(target: str, label: str, effect: str = "manipulated") -> dict:
    return {"target_id": target, "semantic_label": label, "task_effect": effect, "disposition": "remove",
            "rigidity": "rigid", "rigidity_reason": "hard plastic"}


LABELS = {"cup-1": "cup", "carton-1": "carton", "mug-1": "mug box", "static-1": "crate", "artisan-1": "toy robot",
          "overlap-1": "saucer"}


def _job(root: Path, names: list[str]) -> dict:
    """The compile's arguments for the cup and the further objects ``names``, each removed before reconstruction (as
    stage 5 will remove every built object); ``overlap-1`` is filmed where the cup is."""
    root.mkdir(parents=True, exist_ok=True)
    frames = [_frame(root, index) for index in range(2)]
    geometry = {"schema_version": "website_source_geometry.v1", "frames": frames, "unit": "estimated_meters",
                "scale_status": "model_estimated", "metric_measurement_proven": False}
    geometry["digest"] = canonical_digest(geometry, digest_field="digest")
    masks = fx._masks(geometry)
    for name in names:
        masks["targets"].append({"target_id": name, "task_effect": "manipulated", "disposition": "remove",
                                 "semantic_label": LABELS[name],
                                 "track": _track(COLUMNS.get(name, COLUMNS["cup-1"])),
                                 "estimated_visible_bounds": fx._bounds([0, 0.4, 1.45], [0.1, 0.5, 1.55])})
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    removal = {"schema_version": "clean_plate_removal_manifest.v1",
               "entries": [*fx.REMOVAL["entries"], *(_entry(name, LABELS[name]) for name in names)]}
    return {"task_context": fx._task_context(), "task_masks": masks, "removal_manifest": removal,
            "source_geometry": geometry, "base_scene": fx._base_scene(root, geometry), "output_root": root / "out",
            "spend": fx.SPEND, "now": fx.NOW}


def _plan_row(target: str, *, sourcing: str = "parametric", shape: str | None = "carton") -> dict:
    return {"target_id": target, "name": LABELS[target], "role": "moved_object", "variety": None, "rigidity": "rigid",
            "size_m": None, "mass_kg": None, "sourcing": sourcing, "parametric_shape": shape,
            "reason": f"The job moves the {LABELS[target]}.", "editor_object_ids": ["e-" + target]}


def _seal_pick(library_root: Path, row: dict, context: dict, size=(0.16, 0.034, 0.22)) -> dict:
    """The mug box's library pick as the library step seals one: a box model, its credit and its fetched file."""
    from blueprint_pipeline.website_extra_objects import object_root
    root = object_root(library_root, row["target_id"])
    (root / "files").mkdir(parents=True, exist_ok=True)
    mesh = root / "files" / "model.obj"
    mesh.write_bytes(_box_obj(*size))
    subject = {"schema_version": library.SUBJECT_SCHEMA, "target_id": row["target_id"], "route": "extra",
               "reading_digest": library.reading_digest(row, context), "attribution": dict(CREDIT),
               "files": [{"role": "mesh", "path": str(mesh), "sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
                          "size_bytes": mesh.stat().st_size, "suffix": ".obj"}],
               "pick": {"source": "model", "reason": "The same kind of object."}}
    subject["digest"] = canonical_digest(subject, digest_field="digest")
    (root / library.SUBJECT_RECORD).write_text(json.dumps(subject))
    return subject


@pytest.fixture(autouse=True)
def _catalog(monkeypatch):
    """The deployed catalog allows the test model under the license it was chosen under (rights, read now)."""
    monkeypatch.setattr(library_catalog, "catalog", lambda *_, **__: [{"id": ENTRY_ID, "license": "CC-BY-4.0"}])


def _inputs(arguments: dict, rows: list[dict], *, size=(0.16, 0.034, 0.22)) -> list[dict]:
    """``further_inputs`` from a sealed build plan: the cup first (the scored object), then ``rows``."""
    context, entries = arguments["task_context"], arguments["removal_manifest"]["entries"]
    editor = [{"target_id": "e-" + row["target_id"], "label": row["semantic_label"], "disposition": "remove"}
              for row in entries]
    inputs = build_plan.plan_inputs(task_context=context, targets=entries, editor_objects=editor)
    plan = build_plan.validate_plan({"objects": [_plan_row("cup-1", sourcing="generate", shape=None), *rows]},
                                    inputs=inputs, task_context=context)
    assert "dropped" not in plan, plan
    library_root = Path(arguments["output_root"]).parent / "object_library"
    for row in plan["objects"]:
        if row["target_id"] == "mug-1":
            _seal_pick(library_root, row, context, size)
    return further.further_inputs(plan, entries=entries, editor_objects=editor, task_context=context,
                                  library_root=library_root)


def _compile(tmp_path: Path, names: list[str], rows: list[dict], **options) -> dict:
    arguments = _job(tmp_path, names)
    return preparation.compile_website_scene_preparation(**arguments, further_objects=_inputs(arguments, rows, **options))


# --- one id per object -------------------------------------------------------------------------------------------

def test_each_object_has_its_own_replacement_id_and_the_scored_objects_is_unchanged():
    from blueprint_pipeline.website_preparation_request import compose_back_record, replacement_id
    context = {"context_digest": "sha256:" + "ab" * 32}
    scored = "website-subject-" + context["context_digest"][7:27]
    assert replacement_id(context) == replacement_id(context, 1) == scored
    assert [replacement_id(context, rank) for rank in (2, 3, 8)] == [scored + "-02", scored + "-03", scored + "-08"]
    record = dict(task_context=context, subject_min=[0, 0, 0], subject_max=[2, 2, 2], base_scene={"up_axis": "Y"},
                  meters_per_unit=2.0, snap=0.0, registration_path=Path("registration.json"))
    assert compose_back_record(**record) == compose_back_record(**record, rank=1)
    assert compose_back_record(**record)["replacement_asset_id"] == scored
    assert compose_back_record(**record, rank=2) == {**compose_back_record(**record), "replacement_asset_id": scored + "-02"}


# --- two free further objects, composed ---------------------------------------------------------------------------

def test_two_free_further_objects_are_authored_placed_where_filmed_and_composed_beside_the_scored_one(tmp_path):
    value = _compile(tmp_path / "job", ["carton-1", "mug-1"], [_plan_row("carton-1"), _plan_row("mug-1")])
    assert value["status"] == "intake_ready", value["blockers"]
    record = value["further_objects"]
    assert record["digest"] == canonical_digest(record, digest_field="digest")
    assert record["scored_object"] == "/World/Object" and record["left_in_scan"] == [] and record["flags"] == []
    carton, mug = record["objects"]
    scored = value["compose_back"]["replacement_asset_id"]
    assert scored == "website-subject-" + value["binding"]["task_context_digest"][7:27]
    assert [(row["rank"], row["target_id"], row["prim_path"], row["runtime_name"], row["source"]) for row in (carton, mug)
            ] == [(2, "carton-1", "/World/Object_02", "job_object_02", "parametric"),
                  (3, "mug-1", "/World/Object_03", "job_object_03", "library")]
    ids = [scored, carton["replacement_asset_id"], mug["replacement_asset_id"]]
    assert ids == [scored, scored + "-02", scored + "-03"] and len(set(ids)) == 3
    assert [row["runtime_asset_id"] for row in (carton, mug)] == [scored.replace("-", "_") + "_02",
                                                                   scored.replace("-", "_") + "_03"]
    # The library model was used (its file sealed, fetched, allowed now), with its credit; the carton is a shape.
    assert mug["library_design"]["entry_id"] == ENTRY_ID and "parametric_design" not in mug
    assert carton["parametric_design"]["shape"] == "carton" and "library_design" not in carton
    out = tmp_path / "job" / "out"
    for row in (carton, mug):
        asset = out / row["asset"]["relative_path"]
        assert asset.is_file() and "sha256:" + hashlib.sha256(asset.read_bytes()).hexdigest() == row["asset"]["sha256"]
        assert row["static_check"] == "passed" and row["asset_receipt"]["authoring_cost_usd"] == 0.0
        assert further.static_check(asset, dimensions_m=[b - a for a, b in zip(row["bounds_world_m"]["minimum"],
                                                                                row["bounds_world_m"]["maximum"])],
                                    tolerance=preparation.DIMENSION_RELATIVE_ERROR) is None
    # Each at its own filmed pose: its box from its own masks through the scored object's registration, its bottom on
    # the table the scored object rests on, the same surface (one support height), apart from the cup and each other.
    subject = value["subject"]
    table = subject["aabb_min_xyz"][2]
    for row in (carton, mug):
        low, high = row["bounds_world_m"]["minimum"], row["bounds_world_m"]["maximum"]
        assert low[2] == pytest.approx(table, abs=1e-6) and row["support"]["basis"].startswith("support_under")
        assert row["pose_world"] == {"position_world_m": [(a + b) / 2 for a, b in zip(low, high)],
                                     "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
        assert high[0] < subject["aabb_min_xyz"][0]  # filmed left of the cup, so placed left of it
    assert carton["bounds_world_m"]["maximum"][0] < mug["bounds_world_m"]["minimum"][0]
    # The intake request names both, ranked; composition read them from it.
    request = value["intake_request"]
    assert [row["rank"] for row in request["task"]["additional_objects"]] == [2, 3]
    assert [row["subject"]["aabb_min_xyz"] for row in request["task"]["additional_objects"]] == [
        carton["bounds_world_m"]["minimum"], mug["bounds_world_m"]["minimum"]]
    assert {row["subject"]["geometry_origin"] for row in request["task"]["additional_objects"]} == {
        "removed_before_reconstruction"}
    # One compose-back record per object on the composed removal manifest, by target; the scored one's as before.
    composed = json.loads((out / "removal_manifest.composed.json").read_text())
    backs = {row["target_id"]: row.get("compose_back") for row in composed["entries"]}
    assert backs["cup-1"] == value["compose_back"]
    assert backs["carton-1"] == carton["compose_back"] and backs["mug-1"] == mug["compose_back"]
    assert [backs[name]["replacement_asset_id"] for name in ("cup-1", "carton-1", "mug-1")] == ids
    # The preparation on disk is the record returned; nothing was spent or asked.
    assert json.loads((out / "preparation.json").read_text()) == value
    assert value["provider_mutation_performed"] is False
    assert preparation_record_flags(value) == preparation_record_flags({**value, "further_objects": None})


def test_a_library_model_whose_proportions_miss_falls_back_to_its_shape_and_says_why(tmp_path):
    value = _compile(tmp_path / "job", ["mug-1"], [_plan_row("mug-1")], size=(0.6, 0.4, 0.3))
    assert value["status"] == "intake_ready", value["blockers"]
    (mug,) = value["further_objects"]["objects"]
    assert mug["source"] == "parametric" and mug["parametric_design"]["shape"] == "carton"
    assert mug["library_not_used"].startswith("library_aspect_mismatch")


def test_without_further_objects_a_several_object_job_is_handled_as_before(tmp_path):
    arguments = _job(tmp_path / "job", ["carton-1"])
    with pytest.raises(ValueError, match="website_single_manipulated_subject_required"):
        preparation.compile_website_scene_preparation(**arguments)
    # Only an object that came through as a further object composition may take relaxes it.
    entries = arguments["removal_manifest"]["entries"]
    assert further.scored_entries(entries, ()) == [row for row in entries if row["task_effect"] == "manipulated"]
    assert [row["target_id"] for row in further.scored_entries(entries, [{"rank": 2, "target_id": "carton-1"}])
            ] == ["cup-1"]
    for bad in ([{"rank": 1, "target_id": "carton-1"}], [{"rank": 3, "target_id": "carton-1"},
                                                         {"rank": 2, "target_id": "x"}],
                [{"rank": 2, "target_id": "carton-1"}, {"rank": 3, "target_id": "carton-1"}]):
        assert len(further.scored_entries(entries, bad)) == 2


# --- left in the scan, flagged, never refused ---------------------------------------------------------------------

def test_a_further_object_that_needs_artisan_fails_its_static_check_or_overlaps_stays_in_the_scan(tmp_path, monkeypatch):
    original = parametric.author_asset

    def without_colliders(chosen, *, physics_bounds, output_root):
        receipt = original(chosen, physics_bounds=physics_bounds, output_root=output_root)
        if Path(output_root).name == "rank-04":  # the crate: its file loses its colliders after authoring
            from pxr import Sdf
            path = Path(output_root) / receipt["asset"]["path"]
            layer = Sdf.Layer.FindOrOpen(str(path))
            del layer.GetPrimAtPath("/Asset/Geometry").nameChildren["Collision"]
            layer.Save()
            data = path.read_bytes()
            receipt = {**receipt, "asset": {**receipt["asset"], "sha256": hashlib.sha256(data).hexdigest(),
                                            "size_bytes": len(data)}}
        return receipt

    monkeypatch.setattr(parametric, "author_asset", without_colliders)
    value = _compile(tmp_path / "job", ["carton-1", "artisan-1", "static-1", "overlap-1"],
                     [_plan_row("carton-1"), _plan_row("artisan-1", sourcing="generate", shape=None),
                      _plan_row("static-1"), _plan_row("overlap-1")])
    # The scored object's job goes on, and so does the carton.
    assert value["status"] == "intake_ready", value["blockers"]
    record = value["further_objects"]
    assert [row["target_id"] for row in record["objects"]] == ["carton-1"]
    assert record["objects"][0]["prim_path"] == "/World/Object_02"
    assert {row["target_id"]: row["reason"] for row in record["left_in_scan"]} == {
        "artisan-1": "further_object_needs_artisan:website_extra_object_no_free_source",
        "static-1": "further_object_static_check_failed:no_collider",
        "overlap-1": "further_object_overlaps:scored_object_start"}
    assert [row["rank"] for row in record["left_in_scan"]] == [3, 4, 5]
    assert record["flags"] == [further.LEFT_IN_SCAN]
    assert further.LEFT_IN_SCAN in preparation_record_flags(value)
    # Only what was composed is in the request and the composed manifest.
    assert [row["rank"] for row in value["intake_request"]["task"]["additional_objects"]] == [2]
    composed = json.loads((tmp_path / "job" / "out" / "removal_manifest.composed.json").read_text())
    assert {row["target_id"] for row in composed["entries"] if "compose_back" in row and
            row["compose_back"].get("replacement_asset_id")} == {"cup-1", "carton-1"}


def test_further_objects_the_request_would_not_take_are_left_in_the_scan_and_the_job_goes_on(tmp_path):
    arguments = _job(tmp_path / "job", ["carton-1"])
    common = dict(entries=arguments["removal_manifest"]["entries"], targets={}, frames=[], matrix=np.eye(4),
                  runtime_to_sim=np.eye(4), residual=None, collider=object(), up=2, up_sign=1, mpu=1.0,
                  scored=([0, 0, 0], [1, 1, 1]), destination=None, tolerance=0.25, output_root=tmp_path / "out")
    row = {"rank": 2, "target_id": "carton-1", "source": "parametric", "parametric_shape": "carton"}
    articulated = further.place([row], strategy="articulated_open_close", origin="removed_before_reconstruction",
                                **common)
    assert articulated["objects"] == articulated["requested"] == []
    assert articulated["left_in_scan"] == [{"rank": 2, "target_id": "carton-1",
                                            "reason": "scene_intake_task_additional_objects_strategy_invalid"}]
    kept = further.place([row], strategy="pick_and_place", origin="retained_in_captured_reconstruction", **common)
    assert kept["left_in_scan"][0]["reason"] == "scene_intake_task_additional_object_subject_invalid"
    unregistered = further.place([row], strategy="pick_and_place", origin="removed_before_reconstruction",
                                 **{**common, "collider": None})
    assert unregistered["left_in_scan"][0]["reason"] == further.NO_SCENE
    assert not (tmp_path / "out").exists()  # nothing authored for an object that is not composed
    # The request leaving them out (its own check) leaves them in the scan too, flagged, at compose.
    fields, backs = further.compose({"strategy": "pick_and_place"}, {"objects": [
        {"rank": 2, "target_id": "carton-1"}], "left_in_scan": []}, task_context={}, base_scene={}, mpu=1.0,
        registration_path=Path("r.json"), output_root=tmp_path)
    assert backs == {} and fields["further_objects"]["left_in_scan"] == [
        {"rank": 2, "target_id": "carton-1", "reason": further.NOT_TAKEN}]
    assert further.compose({}, {"objects": [], "left_in_scan": []}, task_context={}, base_scene={}, mpu=1.0,
                           registration_path=Path("r.json"), output_root=tmp_path) == ({}, {})



def test_an_unexpected_fault_in_one_further_object_leaves_that_object_in_the_scan_alone(tmp_path, monkeypatch):
    original = further.static_check

    def faulty(path, **options):
        if "rank-02" in str(path):
            raise RuntimeError("a fault of this object's own")
        return original(path, **options)
    monkeypatch.setattr(further, "static_check", faulty)
    value = _compile(tmp_path / "job", ["carton-1", "mug-1"], [_plan_row("carton-1"), _plan_row("mug-1")])
    assert value["status"] == "intake_ready", value["blockers"]
    record = value["further_objects"]
    assert record["left_in_scan"] == [{"rank": 2, "target_id": "carton-1", "reason": "further_object_failed:RuntimeError"}]
    # The next object takes the first free prim: prims follow what was composed, ids follow the rank.
    (mug,) = record["objects"]
    assert (mug["prim_path"], mug["runtime_name"]) == ("/World/Object_02", "job_object_02")
    assert mug["replacement_asset_id"].endswith("-03")
