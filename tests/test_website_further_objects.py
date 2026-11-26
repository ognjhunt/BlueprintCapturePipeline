"""Several objects per job, stages 3 and 4: each free further object authored on the CPU, placed where it was filmed
(seated on its support by its own file's lowest point) and carried into the composed scene and the company's package
as ``/World/Object_02`` onward, beside the scored object; what the package's README and plan note say of each; and a
further object whose library model cannot be credited left out alone.

The fixture is ``tests.test_website_task_preparation``'s captured cup on its table, filmed with more objects beside it
(raised patches in the frames, absent from the collider, as objects taken out before reconstruction are). One-object
jobs are pinned byte for byte elsewhere (``tests/test_website_preparation_request.py``, ``tests/test_test_job_matrix``).
Offline: no provider, no network, nothing paid.
"""
# Covers: src/blueprint_pipeline/website_further_objects.py
# Covers: src/blueprint_pipeline/website_preparation_request.py
# Covers: src/blueprint_pipeline/website_task_preparation.py
# Covers: src/blueprint_pipeline/website_company_readme.py
# Covers: src/blueprint_pipeline/website_build_plan_notes.py
# Covers: src/blueprint_pipeline/website_library_notes.py
# Covers: src/blueprint_pipeline/website_half_preparation.py
# Covers: src/blueprint_pipeline/website_packet_facts.py
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


def _seal_pick(library_root: Path, row: dict, context: dict, size=(0.16, 0.034, 0.22), *, credit: dict = CREDIT,
               model: bytes | None = None) -> dict:
    """The mug box's library pick as the library step seals one: a box model (or ``model``), its credit and its
    fetched file."""
    from blueprint_pipeline.website_extra_objects import object_root
    root = object_root(library_root, row["target_id"])
    (root / "files").mkdir(parents=True, exist_ok=True)
    mesh = root / "files" / "model.obj"
    mesh.write_bytes(_box_obj(*size) if model is None else model)
    subject = {"schema_version": library.SUBJECT_SCHEMA, "target_id": row["target_id"], "route": "extra",
               "reading_digest": library.reading_digest(row, context), "attribution": dict(credit),
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


def _inputs(arguments: dict, rows: list[dict], *, size=(0.16, 0.034, 0.22),
            picks: dict | None = None) -> list[dict]:
    """``further_inputs`` from a sealed build plan: the cup first (the scored object), then ``rows``; the mug box's
    library pick sealed, and each of ``picks`` (a target's credit and model file)."""
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
        elif row["target_id"] in (picks or {}):
            credit, model = picks[row["target_id"]]
            _seal_pick(library_root, row, context, credit=credit, model=model)
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
        assert row["pose_world"]["orientation_xyzw"] == [0.0, 0.0, 0.0, 1.0]
        assert row["pose_world"]["position_world_m"] == pytest.approx([(a + b) / 2 for a, b in zip(low, high)],
                                                                      abs=1e-6)
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



def _seated_bottom(out: Path, row: dict) -> tuple[float, float]:
    """Where the further object's file, placed at its pose, has its lowest point, and how tall the file is: read from
    the file itself, never from the record's own bounds."""
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(out / row["asset"]["relative_path"]))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide])
    box = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    return row["pose_world"]["position_world_m"][2] + box.GetMin()[2], box.GetMax()[2] - box.GetMin()[2]


@pytest.mark.parametrize("factor", [1.1, 0.9], ids=["taller", "shorter"])
def test_a_library_model_taller_or_shorter_than_the_estimate_sits_exactly_on_its_support(tmp_path, factor):
    """Codex review of #3755: a library model used as it is, within the size tolerance but not the estimate's height,
    was placed by the estimated box's middle, so a taller one sank into the table and a shorter one floated above it.
    It is seated by its own file's lowest point now, and its compose-back record names the same place."""
    value = _compile(tmp_path / "job", ["mug-1"], [_plan_row("mug-1")], size=(0.16, 0.034, 0.22 * factor))
    assert value["status"] == "intake_ready", value["blockers"]
    (mug,) = value["further_objects"]["objects"]
    assert mug["source"] == "library" and mug["library_design"]["sizing"]["status"] == "as_is"
    table = value["subject"]["aabb_min_xyz"][2]
    bottom, height = _seated_bottom(tmp_path / "job" / "out", mug)
    assert bottom == pytest.approx(table, abs=further.FLOAT_SLACK_M)  # on the table: neither sunk nor floating
    estimated = mug["estimated_bounds_world_m"]
    seen = estimated["maximum"][2] - estimated["minimum"][2]
    assert height == pytest.approx(0.22 * factor, abs=1e-6) and abs(height - seen) > 0.01  # not the estimate's height
    assert mug["bounds_world_m"]["minimum"][2] == pytest.approx(table, abs=1e-9)
    assert mug["bounds_world_m"]["maximum"][2] - mug["bounds_world_m"]["minimum"][2] == pytest.approx(height, abs=1e-6)
    assert mug["support"]["seated_on"] == "authored_asset_bottom"
    # The footprint's middle is where the video shows it, as before.
    assert mug["pose_world"]["position_world_m"][:2] == pytest.approx(
        [(a + b) / 2 for a, b in zip(estimated["minimum"][:2], estimated["maximum"][:2])], abs=1e-9)
    # The compose-back record and the intake request say the same place and box.
    to_sim = np.asarray(value["coordinate_frame"]["runtime_to_simulator"], dtype=float)
    back = to_sim @ [*mug["compose_back"]["pose_world"]["position"], 1.0]
    assert back[:3].tolist() == pytest.approx(mug["pose_world"]["position_world_m"], abs=1e-9)
    (requested,) = value["intake_request"]["task"]["additional_objects"]
    assert requested["subject"]["aabb_min_xyz"] == mug["bounds_world_m"]["minimum"]
    assert requested["subject"]["aabb_max_xyz"] == mug["bounds_world_m"]["maximum"]


def test_a_seated_object_keeps_its_middle_and_puts_its_own_lowest_point_on_the_support():
    """``seat`` alone: an asset whose origin is not its middle is seated by its own lowest point, in both up axes."""
    estimated = ([1.0, 2.0, 0.5], [1.2, 2.4, 0.7])
    asset = {"minimum": [-0.1, -0.2, -0.03], "maximum": [0.1, 0.2, 0.19]}  # 22 cm tall, origin 3 cm above its bottom
    for to_sim in (np.diag([2.0, 2.0, 2.0, 1.0]),
                   np.array([[2.0, 0, 0, 0], [0, 0, -2.0, 0], [0, 2.0, 0, 0], [0, 0, 0, 1.0]])):
        seated = further.seat(asset, estimated=estimated, runtime_to_sim=to_sim)
        assert seated["position"] == pytest.approx([1.1, 2.2, 0.53])
        assert seated["bounds"][0][2] == pytest.approx(0.5) and seated["bounds"][1][2] == pytest.approx(0.72)
        centre = [(a + b) / 2 for a, b in zip(*seated["runtime"])]
        assert (to_sim @ [*centre, 1.0])[:3].tolist() == pytest.approx(seated["position"])

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
    # The company's page says so in plain words, claiming no reason the flag does not carry.
    from blueprint_pipeline.website_packet_facts import ESTIMATE_SENTENCES
    assert ESTIMATE_SENTENCES[further.LEFT_IN_SCAN] == (
        "One or more of the job's other objects were not built as objects of their own: they are left as the scan "
        "shows them, and the success rule does not score them.")
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

# --- the scene half and the company's package ---------------------------------------------------------------------

#: Where the further objects stand in the scene half's fixture room: on the task object's support, beside it.
ROOM_POSES = {"job_object_02": [2.7, -6.76, 0.75], "job_object_03": [2.97, -6.45, 0.75]}


def _composed(tmp_path: Path) -> tuple[Path, dict]:
    """A capture whose preparation composed the carton and the mug box, and that preparation's bytes."""
    capture = tmp_path / "capture"
    arguments = _job(tmp_path / "job", ["carton-1", "mug-1"])
    arguments["output_root"] = capture / "pipeline" / "website_scene_preparation"
    value = preparation.compile_website_scene_preparation(
        **arguments, further_objects=_inputs(arguments, [_plan_row("carton-1"), _plan_row("mug-1")]))
    assert [row["prim_path"] for row in value["further_objects"]["objects"]] == ["/World/Object_02", "/World/Object_03"]
    return capture, value


def _revision(capture: Path) -> dict:
    """A website revision's source: the staged preparation, named by its bytes (``website_half_preparation``)."""
    data = (capture / "pipeline" / "website_scene_preparation" / "preparation.json").read_bytes()
    return {"source": {"manifest": {"uri": "s3://scenes/x/website/preparation.json",
                                    "digest": "sha256:" + hashlib.sha256(data).hexdigest()}}}


def _in_room(found: dict) -> dict:
    """The further objects as a half reads them, set beside the fixture room's task object: the fixture room is
    another scene than the compile fixture's, so only their poses are moved, never their files or ids."""
    return {**found, "objects": [{**row, "pose_world": {"position_world_m": ROOM_POSES[row["runtime_name"]],
                                                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}}
                                 for row in found["objects"]]}


def _half(tmp_path: Path, further_objects: dict | None, **flags) -> tuple[dict, dict, dict]:
    from blueprint_pipeline.team_scene_packet import materialize_team_scene
    from tests import test_team_scene_packet as packet_tests
    (tmp_path / "half").mkdir(parents=True)
    scene = packet_tests._scene(tmp_path / "half")
    inputs = {key: value for key, value in scene.items() if key not in ("profile", "store", "revision_reference")}
    compiled = materialize_team_scene(**inputs, output_root=tmp_path / "half" / "team-scene", package_only=True,
                                      **({"further_objects": further_objects} if further_objects else {}), **flags)
    record = json.loads(Path(compiled["path"]).read_text())
    folder = Path(compiled["path"]).parent
    plans = [folder / name / plan for name, plan in (("native-task-packet-three-view", "native_task_arena_scene_plan.v1.json"),
                                                   ("native-task-packet", "native_task_arena_scene_plan.v1.json"),
                                                   ("scene-package", "scene_package_plan.v1.json"))]
    plan = json.loads(next(path for path in plans if path.is_file()).read_text())
    return compiled, record, plan


def _with_rule(plan: dict, record: dict) -> dict:
    """The plan with the success contract its half sealed in its record (as the package reads it)."""
    return {**plan, "task_spec": {**plan["task_spec"], "task_success_contract": record["task_success_contract"]}}


def test_a_website_revision_names_its_further_objects_by_its_preparations_bytes(tmp_path):
    capture, value = _composed(tmp_path)
    found = further.for_revision(capture, _revision(capture))
    assert found["record_digest"] == value["further_objects"]["digest"] and further.LEFT_OUT not in found
    assert [Path(row["asset_path"]).is_file() for row in found["objects"]] == [True, True]
    # Not a website revision, another preparation's bytes, or nothing composed: none, so the half is as before.
    assert further.for_revision(capture, {"source": {"manifest": {"uri": "s3://x/manifest.json"}}}) is None
    assert further.for_revision(capture, {"source": {"manifest": {"uri": "s3://x/website/preparation.json",
                                                                  "digest": "sha256:" + "0" * 64}}}) is None
    from blueprint_pipeline.team_scene_packet import further_objects_for
    assert further_objects_for(capture, {}) == {}
    # A file changed since it was composed is left out, said why; the other object still comes.
    changed = Path(found["objects"][0]["asset_path"])
    changed.write_bytes(changed.read_bytes() + b"\n")
    again = further.for_revision(capture, _revision(capture))
    assert [row["runtime_name"] for row in again["objects"]] == ["job_object_03"]
    assert again[further.LEFT_OUT] == {"reason": "further_object_asset_changed", "prims": ["/World/Object_02"]}


def test_the_scene_half_places_each_further_object_after_the_task_row_under_its_own_name(tmp_path):
    capture, _value = _composed(tmp_path)
    found = _in_room(further.for_revision(capture, _revision(capture)))
    compiled, record, plan = _half(tmp_path, found)
    assert record.get("robot_station_absent") is None  # the arm's packet took them
    rows = [(row["name"], row["semantic_role"], row["prim_path"]) for row in plan["objects"]]
    assert rows[2:] == [("task_object", "task_object", "{ENV_REGEX_NS}/task_object"),
                        ("job_object_02", "job_object", "{ENV_REGEX_NS}/job_object_02"),
                        ("job_object_03", "job_object", "{ENV_REGEX_NS}/job_object_03")]
    by_name = {row["name"]: row for row in plan["objects"]}
    assert [by_name[name]["asset_id"] for name in ("job_object_02", "job_object_03")] == [
        row["runtime_asset_id"] for row in found["objects"]]
    assert {by_name[name]["object_type"] for name in ROOM_POSES} == {"RIGID"}
    assert by_name["task_object"]["task_subject"] is True
    assert not any(by_name[name]["task_subject"] for name in ROOM_POSES)
    assert {name: by_name[name]["pose_world"]["position_world_m"] for name in ROOM_POSES} == ROOM_POSES
    assert record["further_objects"] == {"record_digest": found["record_digest"],
                                         "composed": ["job_object_02", "job_object_03"]}
    from blueprint_pipeline.scene_package_stage import neutral_rule, placements
    assert [(row["prim"], row["role"], row["name"]) for row in placements(plan)][2:] == [
        ("Object", "task_object", "object"), ("Object_02", "job_object", "scene/object_02"),
        ("Object_03", "job_object", "scene/object_03")]
    rule = neutral_rule(_with_rule(plan, record), packet=Path(compiled["path"]).parent / "native-task-packet")
    assert rule["scored_object"] == "/World/Object" and rule["not_scored"] == ["/World/Object_02", "/World/Object_03"]
    # The rule still scores the top-ranked object: its start is the task object's.
    assert rule["object_start"]["position_m"] == by_name["task_object"]["pose_world"]["position_world_m"]


def test_a_half_without_further_objects_is_as_before(tmp_path):
    _compiled, record, plan = _half(tmp_path, None)
    assert "further_objects" not in record
    assert [row["semantic_role"] for row in plan["objects"]] == ["scene_collision", "scene_appearance", "task_object"]
    from blueprint_pipeline.scene_package_stage import neutral_rule
    rule = neutral_rule(_with_rule(plan, record))
    assert "scored_object" not in rule and "not_scored" not in rule


def test_a_packet_writer_that_refuses_the_extra_rows_has_them_dropped_flagged_and_the_half_goes_on(tmp_path):
    capture, _value = _composed(tmp_path)
    found = further.for_revision(capture, _revision(capture))  # their filmed poses lie outside the fixture room's look
    _compiled, record, plan = _half(tmp_path, found)
    assert [row["semantic_role"] for row in plan["objects"]] == ["scene_collision", "scene_appearance", "task_object"]
    left_out = record["further_objects"][further.LEFT_OUT]
    assert record["further_objects"]["composed"] == []
    assert left_out["runtime_names"] == ["job_object_02", "job_object_03"]
    assert "job_object" in left_out["reason"]
    # A refusal that names no further row is raised as before.
    with pytest.raises(ValueError, match="other"):
        further.refused_rows_dropped(lambda _: (_ for _ in ()).throw(ValueError("other")),
                                     {"assets": [{"semantic_role": "job_object", "name": "job_object_02"}]},
                                     output_dir=tmp_path / "none")


def test_the_runtime_contract_takes_further_rows_only_by_their_own_distinct_names():
    from blueprint_pipeline.native_task_runtime_contract import _asset_rows
    row = {"filename": "a.usda", "sha256": "sha256:" + "1" * 64, "object_type": "RIGID",
           "pose_world": {"position_world_m": [0, 0, 0], "orientation_xyzw": [0, 0, 0, 1]},
           "reset_state": {"root_pose_world": {}, "joint_positions": {}}}
    base = [{**row, "semantic_role": "scene_collision"}, {**row, "semantic_role": "task_object", "asset_id": "obj"}]
    for names, refused in ((["job_object_02", "job_object_03"], False), (["job_object_02", "job_object_02"], True),
                           (["task_object"], True)):
        errors: list[str] = []
        rows, _ = _asset_rows([*base, *({**row, "semantic_role": "job_object", "name": name, "asset_id": name}
                                        for name in names)], subject_asset_id="obj", errors=errors)
        assert ("native_task_runtime_asset_role_invalid:job_object" in errors) is refused
        if not refused:
            assert [(item["runtime_name"], item["task_subject"]) for item in rows][-2:] == [
                ("job_object_02", False), ("job_object_03", False)]


def test_the_robot_absent_writer_keeps_every_row_of_a_repeated_role(tmp_path):
    from blueprint_pipeline import team_scene_robot_absent as absent
    pose = {"position_world_m": [0, 0, 0], "orientation_xyzw": [0, 0, 0, 1]}
    source = {"root": "evidence", "relative_path": "x", "size_bytes": 1, "sha256": "sha256:" + "1" * 64}
    assets = [{"semantic_role": "scene_collision", "filename": "collision.usda", "pose_world": pose, "source": source},
              {"semantic_role": "task_object", "filename": "first.usda", "pose_world": pose, "source": source},
              {"semantic_role": "task_object", "filename": "second.usda", "pose_world": pose, "source": source},
              {"semantic_role": "job_object", "name": "job_object_02", "filename": "job_object_02.usda",
               "pose_world": pose, "source": source, "object_type": "RIGID"}]
    rows = absent._rows(assets, task_spec={"task_kind": "rigid_pick_place"})
    # Before, the second task_object row replaced the first in the composition's by-role read.
    assert [row["filename"] for row in rows] == ["collision.usda", "first.usda", "second.usda", "job_object_02.usda"]
    assert rows[3]["name"] == "job_object_02" and rows[3]["object_type"] == "RIGID"


def test_a_half_without_the_arm_keeps_both_further_objects(tmp_path, monkeypatch):
    from blueprint_pipeline import team_embodiment_stance as stance
    from blueprint_pipeline.team_embodiment import TeamEmbodimentError

    def no_stance(**_):
        raise TeamEmbodimentError(["team_embodiment_task_motion_no_clear_stance"])
    monkeypatch.setattr(stance, "ground_stance", no_stance)
    capture, _value = _composed(tmp_path)
    found = further.for_revision(capture, _revision(capture))
    _compiled, record, plan = _half(tmp_path, found)
    assert record["robot_station_absent"]["reason"] == "team_embodiment_task_motion_no_clear_stance"
    assert [(row["name"], row["semantic_role"], row["task_subject"]) for row in plan["objects"]][2:] == [
        ("task_object", "task_object", True), ("job_object_02", "job_object", False),
        ("job_object_03", "job_object", False)]
    assert record["further_objects"]["composed"] == ["job_object_02", "job_object_03"]


def _decision(targets: list[str]) -> dict:
    """The Atlas job's sealed decision with its build plan: the cup first (the object it takes out of the scan to
    build), then ``targets``, as their plan rows (``_plan_row``) name them."""
    from blueprint_pipeline.website_build_plan_notes import DECISION_SCHEMA
    sourcing = {"cup-1": "generate", "mug-1": "library", "artisan-1": "generate"}
    rows = [_plan_row(name, sourcing=sourcing.get(name, "parametric"),
                      shape=None if sourcing.get(name) == "generate" else "carton") for name in ["cup-1", *targets]]
    decided = {"schema_version": DECISION_SCHEMA, "applied": False, "build_plan": {"plan": {"objects": rows}},
               "built": {"target_id": "cup-1", "editor_object_ids": ["e-cup-1"]}}
    decided["digest"] = canonical_digest(decided, digest_field="digest")
    return decided


def _published(tmp_path: Path, further_objects: dict, record: dict | None,
               decided: dict | None = None, extra: dict | None = None) -> tuple[dict, Path]:
    """The company's package of a half compiled with ``further_objects``, read from a capture whose preparation holds
    ``record`` as its ``further_objects`` (as the package reads its notes and credits, ``website_half_preparation``),
    or no such record (and ``extra``), and whose Atlas job sealed ``decided``; the publication's result and where the
    package was unpacked."""
    import zipfile

    from tests.test_website_company_simulation import Job, World, _package
    compiled, _record, _plan = _half(tmp_path, further_objects)
    job = Job(tmp_path / "download")
    job.preparation.write_text(json.dumps({**({"further_objects": record} if record is not None else {}),
                                           **(extra or {})}))
    if decided is not None:
        (job.capture / "pipeline" / "atlas").mkdir(parents=True, exist_ok=True)
        (job.capture / "pipeline" / "atlas" / "job_object_rule.json").write_text(json.dumps(decided))
    status = next((job.queue / "status").glob("*.json"))
    status.write_text(json.dumps({**json.loads(status.read_text()), "team_scene_inputs_path": compiled["path"]}))
    result = World(job).publish()
    unpacked = tmp_path / "unpacked"
    if result["status"] == "published":
        with zipfile.ZipFile(_package(job)) as archive:
            archive.extractall(unpacked)
    return {**result, "job": job}, unpacked


def _uncredited(unpacked: Path, design: dict) -> list[str]:
    """Every shipped file holding the library model's geometry (a mesh of its topology, read independently of the
    package's own credit) that its credit under Library models does not name. Rights: always empty."""
    from blueprint_pipeline.website_library_notes import mesh_topologies
    readme = " ".join((unpacked / "README.txt").read_text().split())
    credit = readme.split("Library models", 1)[1].split("These files stay", 1)[0] if "Library models" in readme else ""
    topology = (int(design["model_topology"][0]), str(design["model_topology"][1]))
    holding = sorted(path.relative_to(unpacked).as_posix() for path in unpacked.rglob("*")
                     if path.suffix in (".usd", ".usda", ".usdc", ".usdz") and topology in mesh_topologies(path))
    assert holding, "the model's geometry is in the package"
    return [name for name in holding if name not in credit or "Test Mug Box by Google" not in credit]


def test_the_package_places_both_further_objects_scores_only_the_scored_one_and_credits_the_library_model(tmp_path):
    from pxr import Usd, UsdGeom
    capture, value = _composed(tmp_path)
    result, unpacked = _published(tmp_path, _in_room(further.for_revision(capture, _revision(capture))),
                                  value["further_objects"])
    assert result["status"] == "published", result
    names = sorted(path.relative_to(unpacked).as_posix() for path in unpacked.rglob("*") if path.is_file())
    assert {"scene/object_02.usda", "scene/object_03.usda", "success_rule.json"} <= set(names), names
    stage = Usd.Stage.Open(str(unpacked / "scene.usda"))
    for prim, name in (("/World/Object_02", "job_object_02"), ("/World/Object_03", "job_object_03")):
        placed = UsdGeom.Xformable(stage.GetPrimAtPath(prim)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        assert list(placed.ExtractTranslation()) == pytest.approx(ROOM_POSES[name])
    rule = json.loads((unpacked / "success_rule.json").read_text())
    assert rule["scored_object"] == "/World/Object" and rule["not_scored"] == ["/World/Object_02", "/World/Object_03"]
    readme = " ".join((unpacked / "README.txt").read_text().split())
    assert ("success_rule.json scores only the job's object, /World/Object. /World/Object_02 and /World/Object_03 are "
            "also the job's objects, placed where the video shows them; the rule does not score them.") in readme
    assert "scene/object_02.usda (/World/Object_02): another of the job's objects, where the video shows it." in readme
    # The library model's geometry ships with its credit, naming the files that hold it, and only those.
    credit = readme.split("Library models", 1)[1]
    assert "Test Mug Box by Google" in credit and "scene/object_03.usda" in credit
    scored = next(name for name in names if name.startswith("object."))
    assert "scene/object_02.usda" not in credit and scored not in credit
    mug = next(row for row in value["further_objects"]["objects"] if row["source"] == "library")
    assert _uncredited(unpacked, mug["library_design"]) == []
    variations = json.loads((unpacked / "variations.json").read_text())
    assert {row["part"] for row in variations["objects"]} >= {"/World/Object_02", "/World/Object_03"}


def test_a_library_objects_credit_follows_it_to_its_final_prim_when_an_earlier_one_is_left_out(tmp_path):
    """Codex review of #3755: a half that leaves an earlier further object out renumbers the ones after it, so the
    library model recorded as /World/Object_03 ships as /World/Object_02. Its credit follows it by its identity (its
    runtime asset id and its file's bytes), never by the prim it was recorded at."""
    capture, value = _composed(tmp_path)
    carton = next(row for row in value["further_objects"]["objects"] if row["source"] == "parametric")
    mug = next(row for row in value["further_objects"]["objects"] if row["source"] == "library")
    assert (carton["prim_path"], mug["prim_path"]) == ("/World/Object_02", "/World/Object_03")
    changed = capture / "pipeline" / "website_scene_preparation" / carton["asset"]["relative_path"]
    changed.write_bytes(changed.read_bytes() + b"\n")  # the carton's file changed since it was composed: left out
    found = further.for_revision(capture, _revision(capture))
    assert [row["runtime_name"] for row in found["objects"]] == ["job_object_03"]
    result, unpacked = _published(tmp_path, _in_room(found), value["further_objects"])
    assert result["status"] == "published", result
    assert not (unpacked / "scene" / "object_03.usda").exists()
    rule = json.loads((unpacked / "success_rule.json").read_text())
    assert rule["not_scored"] == ["/World/Object_02"]
    readme = " ".join((unpacked / "README.txt").read_text().split())
    credit = readme.split("Library models", 1)[1]
    assert "- scene/object_02.usda and variations.usda: Test Mug Box by Google" in credit
    assert _uncredited(unpacked, mug["library_design"]) == []


def test_a_further_part_its_preparation_does_not_account_for_never_leaves(tmp_path):
    """Rights: a shipped further part that no record of its preparation names by its identity could hold a library
    model's geometry with no credit, so the package may not leave with it; the room floor ships instead."""
    from blueprint_pipeline import website_company_simulation as sim
    from blueprint_pipeline import website_library_notes as notes
    capture, value = _composed(tmp_path)
    found = _in_room(further.for_revision(capture, _revision(capture)))
    renamed = {**value["further_objects"], "objects": [{**row, "runtime_asset_id": row["runtime_asset_id"] + "_x"}
                                                       if row["source"] == "library" else row
                                                       for row in value["further_objects"]["objects"]]}
    for index, record in enumerate((None, renamed)):
        result, _unpacked = _published(tmp_path / str(index), found, record)
        source = sim.simulation_source(result["job"].capture, queue_root=result["job"].queue)
        assert notes.refusal(source) == notes.FURTHER_UNACCOUNTED
        with pytest.raises(sim.ShapeRefused, match=notes.FURTHER_UNACCOUNTED):
            sim.build_package(source, tmp_path / str(index) / "package.zip", literals=[])


# --- S4: what the README says of each further object ----------------------------------------------------------------

def _readme(unpacked: Path) -> str:
    return " ".join((unpacked / "README.txt").read_text().split())


def test_the_readme_names_each_further_object_how_it_was_built_and_each_left_in_the_scan_with_why(tmp_path):
    """Two further objects composed and one left in the scan: the README names each part the package holds by its
    prim and how it was built, says its place is the video's estimate, says the rule scores only the job's object, and
    its plan note says what became of each further object; the library model's geometry ships with its credit."""
    capture, value = _composed(tmp_path)
    record = {**value["further_objects"], "left_in_scan": [
        {"rank": 4, "target_id": "static-1", "reason": "further_object_static_check_failed:no_collider"}]}
    result, unpacked = _published(tmp_path, _in_room(further.for_revision(capture, _revision(capture))), record,
                                  _decision(["carton-1", "mug-1", "static-1"]))
    assert result["status"] == "published", result
    readme = _readme(unpacked)
    for sentence in (
            # The plan, and what became of each of its further objects.
            "What this simulation needs, as Blueprint's model read the job: cup, carton, mug box and crate.",
            "The plan takes this out of the scan to build as its own object: cup.",
            "Also built as their own objects, where the video shows them: carton and mug box.",
            "Left in the scan as filmed, not built on its own: crate, because its file did not pass Blueprint's "
            "static check.",
            # The parts the package holds, by their prims, how each was built, and its place an estimate.
            "The package also holds /World/Object_02 (carton): a plain carton shape Blueprint made for it, sized from "
            "the video, with none of the real object's detail. It stands where the video shows it; that position is "
            "an estimate from the video, not a measurement.",
            "The package also holds /World/Object_03 (mug box): a model of the same kind of object from a public "
            "library, picked and sized from the video and credited under Library models; it is not the site's own "
            "object. It stands where the video shows it; that position is an estimate from the video, not a "
            "measurement.",
            # The rule scores only the top-ranked object.
            "success_rule.json scores only the job's object, /World/Object. /World/Object_02 and /World/Object_03 are "
            "also the job's objects, placed where the video shows them; the rule does not score them."):
        assert sentence in readme, sentence
    assert "keeps these in the scan" not in readme and "exactly" not in readme
    # The plan's words ship only as plain object words; the library model ships with its credit naming its files.
    credit = readme.split("Library models", 1)[1]
    assert "- scene/object_03.usda and variations.usda: Test Mug Box by Google" in credit
    mug = next(row for row in value["further_objects"]["objects"] if row["source"] == "library")
    assert _uncredited(unpacked, mug["library_design"]) == []
    # The further objects' record names the package's revision: the same package without it is another revision.
    from blueprint_pipeline import website_company_simulation as sim
    source = sim.simulation_source(result["job"].capture, queue_root=result["job"].queue)
    assert source["further_objects"]["left_in_scan"] == [
        {"rank": 4, "target_id": "static-1", "name": "crate", "reason": "further_object_static_check_failed:no_collider"}]
    assert sim.simulation_revision(source) != sim.simulation_revision({**source, "further_objects": None})


def test_a_further_object_built_but_left_out_of_the_half_is_said_to_be_not_in_the_package(tmp_path):
    capture, value = _composed(tmp_path)
    carton = next(row for row in value["further_objects"]["objects"] if row["source"] == "parametric")
    changed = capture / "pipeline" / "website_scene_preparation" / carton["asset"]["relative_path"]
    changed.write_bytes(changed.read_bytes() + b"\n")  # changed since it was composed: the half leaves it out
    result, unpacked = _published(tmp_path, _in_room(further.for_revision(capture, _revision(capture))),
                                  value["further_objects"], _decision(["carton-1", "mug-1"]))
    assert result["status"] == "published", result
    readme = _readme(unpacked)
    assert "Built as its own object but not in this package: carton." in readme
    assert "The package also holds /World/Object_02 (mug box): a model of the same kind of object" in readme
    assert "/World/Object_03" not in readme


def test_the_plan_note_says_what_became_of_each_further_object_and_nothing_without_them():
    from blueprint_pipeline.website_build_plan_notes import LEFT_FOR_ANOTHER_REASON, left_words, plan_notes
    decided = _decision(["carton-1", "mug-1", "static-1", "artisan-1"])
    before = plan_notes(decided)
    assert before[2] == "The plan keeps these in the scan as filmed, not built on their own: carton, mug box, crate and " \
                        "an object the job moves."  # the stale note, still true without a further-objects record
    assert plan_notes(decided, None, {"composed": [], "left_in_scan": []}) == before
    notes = plan_notes(decided, None, {"composed": [{"target_id": "mug-1"}], "left_in_scan": [
        {"target_id": "artisan-1", "reason": "further_object_needs_artisan:website_extra_object_no_free_source"},
        {"target_id": "static-1", "reason": "further_object_overlaps:scored_object_destination"}]})
    assert notes[2:] == [
        "Also built as its own object, where the video shows it: mug box.",
        "Left in the scan as filmed, not built on its own: an object the job moves, because it needs a model "
        "Blueprint would generate, and none was made for it.",
        "Left in the scan as filmed, not built on its own: crate, because where the video shows it, it would overlap "
        "where the job's object goes.",
        "The plan keeps these in the scan as filmed, not built on their own: carton."]
    # A reason it has no words for is never guessed at.
    assert left_words("further_object_something_new:x") == left_words("further_object_overlaps:elsewhere") \
        == LEFT_FOR_ANOTHER_REASON


# --- S4: a further object whose library model cannot be credited is left out alone ---------------------------------

CARTON_ID = "gso:Test_Carton_Box@1"
CARTON_CREDIT = {**CREDIT, "entry_id": CARTON_ID, "title": "Test Carton Box",
                 "item_url": "https://fuel.gazebosim.org/1.0/GoogleResearch/models/Test_Carton_Box/1/Test_Carton_Box.zip"}


def _fanned_box_obj(x: float, y: float, z: float) -> bytes:
    """A box whose top is four triangles about its middle: another mesh (another topology) than ``_box_obj``'s."""
    corners = [(sx * x / 2, sy * y / 2, sz * z) for sz in (0, 1) for sy in (-1, 1) for sx in (-1, 1)]
    faces = [(1, 3, 4, 2), (1, 2, 6, 5), (3, 7, 8, 4), (1, 5, 7, 3), (2, 4, 8, 6),
             (5, 6, 9), (6, 8, 9), (8, 7, 9), (7, 5, 9)]
    return ("\n".join([*(f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in [*corners, (0.0, 0.0, z)]),
                       *("f " + " ".join(str(i) for i in face) for face in faces)]) + "\n").encode()


def _two_library_models(tmp_path: Path, monkeypatch) -> tuple[Path, dict]:
    """A capture whose preparation composed the carton and the mug box, each from its own library model."""
    monkeypatch.setattr(library_catalog, "catalog", lambda *_, **__: [{"id": ENTRY_ID, "license": "CC-BY-4.0"},
                                                                      {"id": CARTON_ID, "license": "CC-BY-4.0"}])
    capture = tmp_path / "capture"
    arguments = _job(tmp_path / "job", ["carton-1", "mug-1"])
    arguments["output_root"] = capture / "pipeline" / "website_scene_preparation"
    value = preparation.compile_website_scene_preparation(**arguments, further_objects=_inputs(
        arguments, [_plan_row("carton-1"), _plan_row("mug-1")],
        picks={"carton-1": (CARTON_CREDIT, _fanned_box_obj(0.16, 0.034, 0.22))}))
    assert [(row["prim_path"], row["source"], row["library_design"]["entry_id"])
            for row in value["further_objects"]["objects"]] == [("/World/Object_02", "library", CARTON_ID),
                                                                ("/World/Object_03", "library", ENTRY_ID)]
    return capture, value


def _holding(unpacked: Path, design: dict) -> list[str]:
    """Every shipped USD file with a mesh of the model's own topology, read independently of the package's credit."""
    from blueprint_pipeline.website_library_notes import mesh_topologies
    topology = (int(design["model_topology"][0]), str(design["model_topology"][1]))
    return sorted(path.relative_to(unpacked).as_posix() for path in unpacked.rglob("*")
                  if path.suffix in (".usd", ".usda", ".usdc", ".usdz") and topology in mesh_topologies(path))


def test_a_further_objects_refused_credit_leaves_out_that_object_alone_and_the_rest_ship_credited(tmp_path, monkeypatch):
    """Rights: the carton's library model is no longer allowed by the deployed catalog. Before, the whole package went
    to the room floor; now only the carton is left out (its geometry never ships), said in the README and on the
    package's record, and the job's object and the mug box, credited, ship."""
    from blueprint_pipeline import website_company_simulation as sim
    from blueprint_pipeline import website_library_notes as notes
    capture, value = _two_library_models(tmp_path, monkeypatch)
    monkeypatch.setattr(library_catalog, "catalog", lambda *_, **__: [{"id": ENTRY_ID, "license": "CC-BY-4.0"}])
    found = _in_room(further.for_revision(capture, _revision(capture)))
    result, unpacked = _published(tmp_path, found, value["further_objects"], _decision(["carton-1", "mug-1"]))
    assert result["status"] == "published", result
    carton, mug = value["further_objects"]["objects"]
    # Only the carton is left out: the mug box takes the first further prim, and the rule names only it.
    assert (unpacked / "scene" / "object_02.usda").is_file() and not (unpacked / "scene" / "object_03.usda").exists()
    rule = json.loads((unpacked / "success_rule.json").read_text())
    assert rule["scored_object"] == "/World/Object" and rule["not_scored"] == ["/World/Object_02"]
    readme = _readme(unpacked)
    assert ("Not in this package: carton. It was built from a library model whose license Blueprint can no longer "
            "confirm, so its geometry is left out.") in readme
    assert "The package also holds /World/Object_02 (mug box): a model of the same kind of object" in readme
    credit = readme.split("Library models", 1)[1]
    assert "- scene/object_02.usda and variations.usda: Test Mug Box by Google" in credit
    assert "Test Carton Box" not in readme
    # Independently of the package's own credit: the mug box's geometry ships named in its credit, the carton's not at
    # all.
    assert _uncredited(unpacked, mug["library_design"]) == []
    assert _holding(unpacked, carton["library_design"]) == []
    # Flagged on the package's record, by the carton's identity and the credit's own refusal.
    manifest = json.loads((result["job"].capture / "pipeline" / "company_simulation" / "company_simulation.json")
                          .read_text())
    (withheld,) = manifest["source"]["further_withheld"]
    assert (withheld["target_id"], withheld["reason"], withheld["runtime_asset_id"]) == (
        "carton-1", notes.LICENSE_REFUSED, carton["runtime_asset_id"])
    # Library geometry still never ships without its credit: were the carton's part placed all the same, the package
    # would not leave with it.
    source = sim.simulation_source(result["job"].capture, queue_root=result["job"].queue)
    assert notes.refusal(source) is None
    monkeypatch.setattr(notes, "without_withheld", lambda plan, _source: plan)
    unfiltered = sim.simulation_source(result["job"].capture, queue_root=result["job"].queue)
    assert len(notes.further_prims(unfiltered)) == 2
    assert notes.refusal(unfiltered) == notes.LICENSE_REFUSED
    with pytest.raises(sim.ShapeRefused, match=notes.LICENSE_REFUSED):
        sim.build_package(unfiltered, tmp_path / "unfiltered.zip", literals=[])


def test_a_further_objects_unreadable_credit_leaves_out_that_object_alone(tmp_path, monkeypatch):
    from blueprint_pipeline import website_library_notes as notes
    capture, value = _two_library_models(tmp_path, monkeypatch)
    record = json.loads(json.dumps(value["further_objects"]))
    del record["objects"][1]["library_design"]["attribution"]["author"]  # the mug box's credit cannot be read
    result, unpacked = _published(tmp_path, _in_room(further.for_revision(capture, _revision(capture))), record,
                                  _decision(["carton-1", "mug-1"]))
    assert result["status"] == "published", result
    readme = _readme(unpacked)
    assert ("Not in this package: mug box. It was built from a library model whose credit could not be read, so its "
            "geometry is left out.") in readme
    assert "- scene/object_02.usda and variations.usda: Test Carton Box by Google" in readme
    assert _holding(unpacked, value["further_objects"]["objects"][1]["library_design"]) == []
    assert notes.further_credits({"further_objects": record}, capture)["further_withheld"][0]["reason"] == \
        notes.CREDIT_UNREADABLE


def test_the_job_objects_own_refused_credit_still_ships_the_room_floor(tmp_path):
    """The scored object's credit is as before: when it cannot be read, the package may not leave with the object."""
    from blueprint_pipeline import website_library_notes as notes
    refused = notes.merged({"library_refusal": notes.CREDIT_UNREADABLE},
                           {"further_withheld": [{"runtime_asset_id": "x", "asset_sha256": "y", "reason": "z"}]})
    assert refused == {"library_refusal": notes.CREDIT_UNREADABLE}
    assert notes.refusal({"files": {"Object": {"role": "task_object"}}, **refused}) == notes.CREDIT_UNREADABLE


# --- S4: each further object varied on its own evidence ------------------------------------------------------------

def test_each_further_object_is_varied_on_its_own_size_and_mass_evidence_never_the_job_objects(tmp_path):
    """A further object's variation ranges rest on its own evidence, as the job's object's do: its plan row's mass
    quoted from the site's words, else the mass prior from its own size in the video; for size, the video's scale band.
    The site's answer of the item's weight is the job's object's alone."""
    capture, value = _composed(tmp_path)
    decided = _decision(["carton-1", "mug-1"])
    plan = decided["build_plan"]["plan"]["objects"]
    plan[1] = {**plan[1], "mass_kg": {"low": 0.2, "high": 0.4, "basis": "site_words"}}  # the carton's, as the site said
    decided["digest"] = canonical_digest(decided, digest_field="digest")
    answer = {"authoring_inputs": {"configuration": {"construction_constraints": {
        "operator_task_details": {"item_weight": "2 kg"}}}, "metric_envelope": value["authoring_inputs"]["metric_envelope"]}}
    result, unpacked = _published(tmp_path, _in_room(further.for_revision(capture, _revision(capture))),
                                  value["further_objects"], decided, answer)
    assert result["status"] == "published", result
    by_part = {row["part"]: row["ranges"] for row in json.loads((unpacked / "variations.json").read_text())["objects"]}
    assert by_part["/World/Object"]["mass_kg"]["from"] == "site_answer_item_weight"  # the job's object's own answer
    carton, mug = by_part["/World/Object_02"], by_part["/World/Object_03"]
    assert {key: carton["mass_kg"][key] for key in ("low", "high", "basis", "from")} == {
        "low": 0.2, "high": 0.4, "basis": "site_answer", "from": "build_plan_site_words"}
    prior = next(row for row in value["further_objects"]["objects"] if row["source"] == "library")[
        "asset_receipt"]["physics_bounds"]["mass_kg"]
    assert [mug["mass_kg"][key] for key in ("low", "high", "basis", "from")] == [
        round(prior[0], 6), round(prior[1], 6), "video_estimate", "video_size_prior"]
    for ranges in (carton, mug):
        assert (ranges["scale"]["basis"], ranges["scale"]["from"]) == ("video_estimate", "video_scale_band")
        assert [ranges["scale"]["low"], ranges["scale"]["high"]] == [0.75, round(1 / 0.75, 6)]
        assert ranges["static_friction"]["basis"] == "default"
    readme = _readme(unpacked)
    assert "/World/Object_02: size (video_estimate), mass (site_answer) and friction (default)." in readme
    assert "/World/Object_03: size (video_estimate), mass (video_estimate) and friction (default)." in readme
    # Matched by identity, never by prim: a part another object's evidence does not name gets default bands.
    from blueprint_pipeline import website_company_simulation as sim
    from blueprint_pipeline import website_object_variations as variations
    source = sim.simulation_source(result["job"].capture, queue_root=result["job"].queue)
    assert variations._told(source, "Object_02")["mass_kg"]["from"] == "build_plan_site_words"
    rows = source["variation_evidence"][variations.FURTHER_EVIDENCE]
    renamed = {**source, "variation_evidence": {**source["variation_evidence"], variations.FURTHER_EVIDENCE: [
        {**row, "asset_sha256": "sha256:" + "0" * 64} for row in rows]}}
    assert variations._told(renamed, "Object_02") == {}
