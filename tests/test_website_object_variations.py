"""Simulated variations of the job's object in the company's package: anchored ranges, deterministic samples, a layer
that opens in ``pxr``, and words that never call them measurements.

Hermetic: the package tests' capture and compiled half (``tests/test_website_company_simulation``), with an object
file written here that holds a rigid body, a mass, a physics material, a colour and a roughness. Nothing is paid for.
"""
# Covers: src/blueprint_pipeline/website_object_variations.py
# Covers: src/blueprint_pipeline/website_job_estimates.py
# Covers: src/blueprint_pipeline/website_library_notes.py
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import outbound_file_scan as scan
from blueprint_pipeline import website_company_simulation as sim
from blueprint_pipeline import website_object_variations as variations
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_build_plan_notes import DECISION_SCHEMA

pxr = pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils  # noqa: E402

from tests.test_website_company_simulation import (  # noqa: E402
    ACCOUNT, CAPTURE, OBJECT_POSE, REQUEST, TASK, Job, World, _package, _write)

SIDE = 0.2  # the object's cube, metres: its largest side
MASS, STATIC, DYNAMIC, COLOUR, ROUGHNESS = 1.1, 0.55, 0.475, (0.5, 0.3, 0.1), 0.6
QUOTE = "the tray is about twenty centimetres across"
ANSWER = "about 11 lb"


def _object(path: Path, *, joint: bool = False, two_materials: bool = False) -> Path:
    """An object file as the build writes one: a rigid body with its mass, a box that sits on z = 0, a physics
    material, a display colour and a preview surface's roughness."""
    path.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    body = UsdGeom.Xform.Define(stage, "/Asset/Body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body).CreateMassAttr(MASS)
    box = UsdGeom.Cube.Define(stage, "/Asset/Body/Box")
    box.CreateSizeAttr(SIDE)
    box.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, SIDE / 2))
    box.CreateDisplayColorAttr([Gf.Vec3f(*COLOUR)])
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    for name in ("PhysicsMaterial", *(("OtherMaterial",) if two_materials else ())):
        material = UsdPhysics.MaterialAPI.Apply(UsdGeom.Scope.Define(stage, f"/Asset/{name}").GetPrim())
        material.CreateStaticFrictionAttr(STATIC if name == "PhysicsMaterial" else 0.9)
        material.CreateDynamicFrictionAttr(DYNAMIC)
    UsdShade.Material.Define(stage, "/Asset/Look")
    surface = UsdShade.Shader.Define(stage, "/Asset/Look/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(ROUGHNESS)
    if joint:
        UsdPhysics.RevoluteJoint.Define(stage, "/Asset/Hinge")
    stage.GetRootLayer().Save()
    return path


def _job(tmp_path: Path, **kwargs) -> Job:
    job = Job(tmp_path)
    job.object.unlink()
    job.object = _object(job.assets / "replacement.usda", **kwargs)
    job.seal()
    return job


def _preparation(**changes) -> dict:
    """What a preparation records of its object: the site's weight answer, the video's sizes and its mass prior."""
    value = {"authoring_inputs": {
        "configuration": {"source_object_identity": "tray",
                          "construction_constraints": {"operator_task_details": {"item_weight": ANSWER}}},
        "metric_envelope": {"maximum_dimension_relative_error": 0.25}, "dimension_authority": "estimated"},
        "physics": {"basis": "estimated", "bounds": {"mass_kg": [0.4, 3.0], "static_friction": [0.2, 0.9],
                                                     "dynamic_friction": [0.15, 0.8]},
                    "priors": {"static_friction": [0.2, 0.9], "dynamic_friction": [0.15, 0.8]}},
        "weight_estimate": {"status": "estimated", "basis": "physics_from_size_and_material", "range_kg": [1.0, 2.0]}}
    for path, item in changes.items():
        node = value
        *parents, last = path.split(".")
        for name in parents:
            node = node[name]
        if item is None:
            node.pop(last)
        else:
            node[last] = item
    return value


def _decide(capture: Path, **row) -> None:
    """The job's sealed build plan, with its row for the object the preparation authored."""
    plan_row = {"target_id": "tray", "name": "tray", "role": "moved_object", "variety": None, "rigidity": "rigid",
                "size_m": None, "mass_kg": None, "sourcing": "parametric", "parametric_shape": "tray", "reason": "r",
                "editor_object_ids": [], **row}
    decided = {"schema_version": DECISION_SCHEMA, "applied": False,
               "build_plan": {"plan": {"objects": [plan_row]}}, "built": None}
    decided["digest"] = canonical_digest(decided, digest_field="digest")
    _write(capture / "pipeline" / "atlas" / "job_object_rule.json", decided)


def _built(job: Job, tmp_path: Path, name: str = "package.zip") -> tuple[dict, Path]:
    source = sim.simulation_source(job.capture, queue_root=job.queue)
    built = sim.build_package(source, tmp_path / name, literals=sim.secret_literals(job.capture, source))
    unpacked = tmp_path / (name + ".unpacked")
    with zipfile.ZipFile(tmp_path / name) as archive:
        archive.extractall(unpacked)
    return built, unpacked


def _sidecar(unpacked: Path) -> dict:
    return json.loads((unpacked / variations.SIDECAR_FILE).read_text())


# --- the layer, the sidecar and the README -------------------------------------------------------------------------

def test_each_sample_composes_its_values_in_pxr_and_keeps_the_object_on_its_surface(tmp_path):
    job = _job(tmp_path / "job")
    built, unpacked = _built(job, tmp_path)
    assert {variations.LAYER_FILE, variations.SIDECAR_FILE} <= set(built["files"])
    # The layer opens on its own, and everything it uses is inside the package.
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(unpacked / variations.LAYER_FILE))
    assert unresolved == []
    for item in [*(layer.realPath for layer in layers), *assets]:
        assert Path(item.split("[", 1)[0]).resolve().is_relative_to(unpacked.resolve())
    stage = Usd.Stage.Open(str(unpacked / variations.LAYER_FILE))
    assert UsdGeom.GetStageUpAxis(stage) == "Z" and UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    part = stage.GetPrimAtPath("/World/Object")
    chosen = part.GetVariantSets().GetVariantSet(variations.VARIANT_SET)
    assert chosen.GetVariantSelection() == variations.NOMINAL
    row = _sidecar(unpacked)["objects"][0]
    assert row["part"] == "/World/Object" and row["varied"] is True
    assert set(row["ranges"]) == set(variations.QUANTITIES) and row["not_varied"] == {}
    assert sorted(chosen.GetVariantNames()) == sorted([variations.NOMINAL, *(s["variant"] for s in row["samples"])])
    assert len(row["samples"]) == variations.SAMPLES

    body, box = stage.GetPrimAtPath("/World/Object/Asset/Body"), stage.GetPrimAtPath("/World/Object/Asset/Body/Box")
    material = stage.GetPrimAtPath("/World/Object/Asset/PhysicsMaterial")
    surface = stage.GetPrimAtPath("/World/Object/Asset/Look/Surface")

    def bound():
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        return cache.ComputeWorldBound(stage.GetPrimAtPath("/World/Object/Asset")).ComputeAlignedRange()

    def read():
        return (body.GetAttribute("physics:mass").Get(), material.GetAttribute("physics:staticFriction").Get(),
                material.GetAttribute("physics:dynamicFriction").Get(), tuple(box.GetAttribute(
                    "primvars:displayColor").Get()[0]), surface.GetAttribute("inputs:roughness").Get(), bound())
    nominal = read()
    # Nominal is the object exactly as scene.usda has it.
    plain = Usd.Stage.Open(str(unpacked / "scene.usda"))
    assert plain.GetPrimAtPath("/World/Object/Asset/Body").GetAttribute("physics:mass").Get() == nominal[0]
    assert nominal[0] == pytest.approx(MASS) and nominal[1] == pytest.approx(STATIC)
    assert nominal[5].GetMin()[2] == pytest.approx(OBJECT_POSE["position_world_m"][2])
    stage.SetEditTarget(stage.GetSessionLayer())
    for sample in row["samples"]:
        chosen.SetVariantSelection(sample["variant"])
        values = sample["values"]
        mass, static, dynamic, colour, roughness, box_range = read()
        assert mass == pytest.approx(values["mass_kg"]) and static == pytest.approx(values["static_friction"])
        assert dynamic == pytest.approx(values["dynamic_friction"])
        assert colour == pytest.approx(tuple(min(1.0, c * values["display_color_factor"]) for c in COLOUR), abs=1e-5)
        assert roughness == pytest.approx(ROUGHNESS + values["roughness_offset"], abs=1e-5)
        # The size scales about the bottom centre: the bottom and the centre of the footprint stay where they were.
        assert box_range.GetSize()[2] == pytest.approx(SIDE * values["scale"], rel=1e-5)
        assert box_range.GetMin()[2] == pytest.approx(nominal[5].GetMin()[2], abs=1e-6)
        for axis in (0, 1):
            assert (box_range.GetMin()[axis] + box_range.GetMax()[axis]) == pytest.approx(
                nominal[5].GetMin()[axis] + nominal[5].GetMax()[axis], abs=1e-6)
        # Every value is inside its range.
        for key, value in values.items():
            assert row["ranges"][key]["low"] <= value <= row["ranges"][key]["high"], key
        assert dynamic <= max(static, row["ranges"]["dynamic_friction"]["low"]) + 1e-9
    # The shipped files are unchanged: nothing was written back.
    assert (unpacked / variations.LAYER_FILE).read_bytes() == zipfile.ZipFile(tmp_path / "package.zip").read(
        variations.LAYER_FILE)


def test_the_files_pass_the_packages_scans_and_carry_neither_the_sites_words_nor_a_digest(tmp_path):
    job = _job(tmp_path / "job")
    _decide(job.capture, size_m={"low": 0.18, "high": 0.22, "basis": "site_words", "quote": QUOTE})
    job.preparation.write_text(json.dumps(_preparation()))
    world = World(job)
    assert world.publish()["status"] == "published"
    with zipfile.ZipFile(_package(job)) as archive:
        archive.extractall(tmp_path / "unpacked")
        texts = {name: archive.read(name) for name in (variations.LAYER_FILE, variations.SIDECAR_FILE)}
    literals = scan.literal_forms([(REQUEST, "a job id"), (CAPTURE, "a job id"), (TASK, "a job id"),
                                   (ACCOUNT, "an account id")])
    assert scan.scan_tree(tmp_path / "unpacked", literals) == []
    for text in texts.values():
        for value in (QUOTE.encode(), ANSWER.encode(), b"11 lb", b"sha256", b"tray", job.capture.as_posix().encode()):
            assert value not in text


def test_the_readme_says_plainly_they_are_simulated_variations_not_measurements(tmp_path):
    job = _job(tmp_path / "job")
    job.preparation.write_text(json.dumps(_preparation()))
    _, unpacked = _built(job, tmp_path)
    readme = (unpacked / "README.txt").read_text()
    flat = " ".join(readme.split())
    assert f"\n{variations.HEADING}\n" in readme and readme.index(variations.HEADING) < readme.index("\nTerms\n")
    assert ("These are simulated variations for testing in your own tools, not measurements of the site's "
            "objects.") in flat
    assert "variations.usda: scene.usda with one variant set, simulated_variation" in flat
    assert "sample_01 to sample_08" in flat and "variations.json:" in flat
    assert ("/World/Object: size (video_estimate), mass (site_answer), friction (default), colour (default) and "
            "roughness (default).") in flat
    assert variations.RULE_LINE in flat and variations.SIZE_LINE in flat
    assert "Library models" not in readme  # no library model, so no word of one
    sidecar = _sidecar(unpacked)
    assert sidecar["what_these_are"] == variations.NOT_MEASUREMENTS
    assert set(sidecar["bases"]) == {variations.SITE_ANSWER, variations.VIDEO_ESTIMATE, variations.DEFAULT}
    # Nothing in either file claims a measurement.
    for text in (json.dumps(sidecar), (unpacked / variations.LAYER_FILE).read_text()):
        assert "measured" not in text and text.count("measurement") == text.count("not measurements")


# --- determinism ----------------------------------------------------------------------------------------------------

def test_the_same_inputs_give_the_same_samples_and_bytes_and_another_object_others(tmp_path):
    job = _job(tmp_path / "job")
    first, unpacked = _built(job, tmp_path, "one.zip")
    again, again_unpacked = _built(job, tmp_path, "two.zip")
    assert first["digest"] == again["digest"]
    assert (unpacked / variations.LAYER_FILE).read_bytes() == (again_unpacked / variations.LAYER_FILE).read_bytes()
    before = _sidecar(unpacked)["objects"][0]["samples"]
    # Another object file (a recompiled half): new draws, inside the same ranges.
    stage = Usd.Stage.Open(str(job.object))
    stage.GetPrimAtPath("/Asset/Look").SetCustomDataByKey("revision", 2)
    stage.GetRootLayer().Save()
    job.seal()
    _, changed = _built(job, tmp_path, "three.zip")
    after = _sidecar(changed)["objects"][0]
    assert after["samples"] != before
    assert all(after["ranges"][key]["low"] <= value <= after["ranges"][key]["high"]
               for sample in after["samples"] for key, value in sample["values"].items())


# --- every range rests on its evidence ------------------------------------------------------------------------------

@pytest.mark.parametrize(("changes", "plan", "expected"), [
    ({}, {}, {"basis": "site_answer", "from": "site_answer_item_weight",
              "range": [11 * 0.45359237 * 0.9, 11 * 0.45359237 * 1.1]}),
    ({"authoring_inputs.configuration.construction_constraints": None},
     {"mass_kg": {"low": 2.0, "high": 3.0, "basis": "site_words", "quote": QUOTE}},
     {"basis": "site_answer", "from": "build_plan_site_words", "range": [2.0, 3.0]}),
    ({"authoring_inputs.configuration.construction_constraints": None},
     {"mass_kg": {"low": 1.5, "high": 4.0, "basis": "estimated", "from": "its size"}},
     {"basis": "video_estimate", "from": "build_plan_estimate", "range": [1.5, 4.0]}),
    ({"authoring_inputs.configuration.construction_constraints": None}, {},
     {"basis": "video_estimate", "from": "video_size_and_material", "range": [1.0, 2.0]}),
    ({"authoring_inputs.configuration.construction_constraints": None,
      "weight_estimate": {"status": "estimated", "basis": "published_exact_model", "range_kg": [5.0, 6.0]}}, {},
     {"basis": "video_estimate", "from": "video_size_prior", "range": [0.4, 3.0]}),
    # A designed object's physics is the design's, not the video's: its mass varies around the file's value.
    ({"authoring_inputs.configuration.construction_constraints": None, "weight_estimate": None,
      "physics.basis": "synthetic_design"}, {}, None),
])
def test_the_mass_range_is_the_first_evidence_that_gives_one(tmp_path, changes, plan, expected):
    capture = tmp_path / "capture"
    _decide(capture, **plan)
    found = variations.evidence(_preparation(**changes), capture).get("variation_evidence", {}).get("Object", {})
    if expected is None:
        assert "mass_kg" not in found
    else:
        assert found["mass_kg"]["basis"] == expected["basis"] and found["mass_kg"]["from"] == expected["from"]
        assert found["mass_kg"]["range"] == pytest.approx(expected["range"])
    assert QUOTE not in json.dumps(found) and ANSWER not in json.dumps(found)


@pytest.mark.parametrize(("changes", "plan", "expected"), [
    ({}, {"size_m": {"low": 0.18, "high": 0.22, "basis": "site_words", "quote": QUOTE}},
     ("site_answer", "build_plan_site_words", [0.9, 1.1])),
    ({}, {"size_m": {"low": 0.1, "high": 0.3, "basis": "estimated", "from": "the frames"}},
     ("video_estimate", "build_plan_estimate", [0.5, 1.5])),
    ({}, {}, ("video_estimate", "video_scale_band", [0.75, 1 / 0.75])),
    ({"authoring_inputs.dimension_authority": "published_exact_model"}, {}, ("default", "scene_usda_size", [0.9, 1.1])),
    ({"authoring_inputs.metric_envelope": None}, {}, ("default", "scene_usda_size", [0.9, 1.1])),
])
def test_the_size_range_rests_on_the_site_then_the_video_then_a_default_band(tmp_path, changes, plan, expected):
    job = _job(tmp_path / "job")
    _decide(job.capture, **plan)
    job.preparation.write_text(json.dumps(_preparation(**changes)))
    _, unpacked = _built(job, tmp_path)
    scale = _sidecar(unpacked)["objects"][0]["ranges"]["scale"]
    assert (scale["basis"], scale["from"]) == expected[:2]
    assert [scale["low"], scale["high"]] == pytest.approx(expected[2], abs=1e-6)
    assert scale["nominal_largest_side_m"] == pytest.approx(SIDE)


def test_with_no_evidence_every_range_is_a_narrow_band_around_the_files_value(tmp_path):
    job = _job(tmp_path / "job")
    _, unpacked = _built(job, tmp_path)
    ranges = _sidecar(unpacked)["objects"][0]["ranges"]
    assert {row["basis"] for row in ranges.values()} == {variations.DEFAULT}
    for key, nominal in (("mass_kg", MASS), ("static_friction", STATIC), ("dynamic_friction", DYNAMIC)):
        low, high = variations.DEFAULT_BANDS[key]
        assert [ranges[key]["low"], ranges[key]["high"]] == pytest.approx([nominal * low, nominal * high], abs=1e-6)
        assert ranges[key]["nominal"] == pytest.approx(nominal)
    assert [ranges["scale"]["low"], ranges["scale"]["high"]] == list(variations.DEFAULT_BANDS["scale"])


def test_what_the_file_does_not_hold_is_not_varied_and_says_why(tmp_path):
    job = _job(tmp_path / "job", joint=True, two_materials=True)
    _, unpacked = _built(job, tmp_path)
    row = _sidecar(unpacked)["objects"][0]
    assert row["not_varied"] == {"scale": "has_joints", "mass_kg": "has_joints", "static_friction": "materials_differ"}
    assert set(row["ranges"]) == {"dynamic_friction", "display_color_factor", "roughness_offset"}
    assert row["ranges"]["dynamic_friction"]["prims"] == ["/World/Object/Asset/PhysicsMaterial",
                                                          "/World/Object/Asset/OtherMaterial"]
    readme = " ".join((unpacked / "README.txt").read_text().split())
    assert variations.SIZE_LINE not in readme
    assert "/World/Object: dynamic friction (default), colour (default) and roughness (default)." in readme


# --- no object, no rights: no variations ----------------------------------------------------------------------------

def test_a_room_only_package_gets_no_variations(tmp_path, monkeypatch):
    from tests.test_website_package_floor import Floor
    floor = Floor(tmp_path, monkeypatch)
    floor.started(reason="spend_cap_reached")
    world = World(floor.job)
    assert world.publish()["status"] == "published"
    with zipfile.ZipFile(_package(floor.job)) as archive:
        names, readme = archive.namelist(), archive.read("README.txt").decode()
    assert variations.LAYER_FILE not in names and variations.SIDECAR_FILE not in names
    assert variations.HEADING not in readme and "simulated variation" not in readme


def test_a_package_without_an_object_part_writes_nothing(tmp_path):
    source = {"files": {"Room": {"role": "scene_appearance"}}, "axis": "Z"}
    assert variations.write(source, tmp_path, {"Room": "scene/room.usdc"}) == {}
    assert variations.write({**source, "room": {"plan_digest": None}}, tmp_path, {}) == {}
    assert list(tmp_path.iterdir()) == [] and variations.readme_lines({}) == []


LIBRARY_ROW = {"title": "Tote", "author": "A", "library": "L", "link": "https://example.org/tote", "changes": "x",
               "file": None}


@pytest.mark.parametrize(("license", "url", "stop"), [
    ("CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/", None),
    ("CC0 1.0", "https://creativecommons.org/publicdomain/zero/1.0/", None),
    ("CC BY-ND 4.0", "https://creativecommons.org/licenses/by-nd/4.0/", variations.LICENSE_STOP),
    ("CC BY 4.0", "https://creativecommons.org/licenses/by-nd/4.0/", variations.LICENSE_STOP),
])
def test_a_library_model_is_varied_only_under_a_license_that_allows_adaptations(tmp_path, license, url, stop):
    source = {"files": {"Object": {"role": "task_object"}}, "axis": "Z",
              "library_models": [{**LIBRARY_ROW, "license": license, "license_url": url}]}
    assert variations.rights_stop(source) == stop
    if stop is not None:
        summary = variations.write(source, tmp_path, {"Object": "object.usdz"})
        assert summary["variations"]["objects"] == [{"part": "/World/Object", "varied": False, "reason": stop}]
        assert list(tmp_path.iterdir()) == []  # nothing written, and pxr never opened
        readme = " ".join(variations.readme_lines(summary))
        assert "no simulated variations of the job's object" in readme and "does not allow changes" in readme
    assert variations.rights_stop({**source, "library_refusal": "any"}) == variations.CREDIT_STOP


def test_a_library_model_s_variants_are_said_to_change_it(tmp_path):
    lines = variations.readme_lines({"variations": {"library": True, "rule": True, "objects": [
        {"part": "/World/Object", "varied": True,
         "ranges": {"scale": {"basis": "default"}, "static_friction": {"basis": "default"},
                    "dynamic_friction": {"basis": "default"}}}]}})
    flat = " ".join(" ".join(lines).split())
    assert ("The variants change the size and friction of the library model credited under Library models; its own "
            "file is unchanged.") in flat


def test_a_fault_in_the_layer_ships_the_package_without_it(tmp_path, monkeypatch):
    job = _job(tmp_path / "job")
    monkeypatch.setattr(variations, "_check", lambda *args: "variant_value_not_composed")
    built, unpacked = _built(job, tmp_path)
    assert variations.LAYER_FILE not in built["files"] and variations.SIDECAR_FILE not in built["files"]
    assert variations.HEADING not in (unpacked / "README.txt").read_text()
