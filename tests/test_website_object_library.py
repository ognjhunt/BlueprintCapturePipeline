"""The job's object from a licensed library first (owner direction 2026-11-23): a license gate that admits only the
allow-list, a model's pick that plain code checks, a fallback that never invents a match, the library before the
parametric shape before the paid builder, and the model's credit in the company's package.

Offline: a tiny fixture catalog and a small fixture mesh; every fetch goes to an injected fetcher, no Gemini key is
set (``tests/conftest.py``) and nothing is paid for.
"""
# Covers: src/blueprint_pipeline/website_object_library.py
# Covers: src/blueprint_pipeline/website_library_asset.py
# Covers: src/blueprint_pipeline/website_library_notes.py
# Covers: src/blueprint_pipeline/website_supplied_mesh.py
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import website_library_asset as asset
from blueprint_pipeline import website_library_catalog as library_catalog
from blueprint_pipeline import website_object_library as library
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_supplied_mesh import LIBRARY_SUBJECT, supplied_mesh_configuration

pytest.importorskip("pxr")

CC_BY = "Creative Commons Attribution 4.0 International"
API = "https://fuel.gazebosim.org/1.0/GoogleResearch/models"
ROW = {"target_id": "tote", "name": "blue storage tote", "role": "moved_object", "variety": "one tote of parts",
       "rigidity": "rigid", "size_m": {"low": 0.5, "high": 0.7, "basis": "estimated", "from": "the analysis"},
       "mass_kg": None, "sourcing": "generate", "parametric_shape": None, "reason": "The job moves the tote.",
       "editor_object_ids": ["centerpiece"]}


def _box_obj(x: float, y: float, z: float) -> bytes:
    """A closed box, its base on the origin's plane, faces as quads (an OBJ's polygons are fanned on reading)."""
    corners = [(sx * x / 2, sy * y / 2, sz * z) for sz in (0, 1) for sy in (-1, 1) for sx in (-1, 1)]
    quads = [(1, 3, 4, 2), (5, 6, 8, 7), (1, 2, 6, 5), (3, 7, 8, 4), (1, 5, 7, 3), (2, 4, 8, 6)]
    lines = ["mtllib model.mtl", *(f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in corners), "vt 0 0", "usemtl m",
             *("f " + " ".join(f"{i}/1" for i in quad) for quad in quads)]
    return ("\n".join(lines) + "\n").encode()


def _jpeg() -> bytes:
    from PIL import Image
    out = io.BytesIO()
    Image.new("RGB", (640, 460), (90, 120, 200)).save(out, format="JPEG")
    return out.getvalue()


MESH, THUMB = _box_obj(0.6, 0.4, 0.3), _jpeg()


def _record(name: str, title: str, *, rights: int = 0, size=(0.6, 0.4, 0.3), categories=()) -> dict:
    return {"name": name, "version": 1, "title": title, "categories": list(categories), "rights": rights,
            "mesh": {"sha256": hashlib.sha256(MESH).hexdigest(), "size_bytes": len(MESH), "faces": 12},
            "thumbnail": {"sha256": hashlib.sha256(THUMB).hexdigest(), "size_bytes": len(THUMB)},
            "size_m": list(size)}


def _rights(fuel: str | None, metadata: str | None) -> dict:
    return {"authors": ["Google"], "copyright": "Copyright 2020 Google LLC.",
            "license": {"fuel_license_name": fuel, "fuel_license_url": "http://creativecommons.org/licenses/by/4.0/"
                        if fuel == CC_BY else None, "metadata_license": metadata}}


def _index(tmp_path: Path, records: list[dict] | None = None) -> dict[str, Path]:
    index = {"schema_version": "object_library_gso_index.v1", "library": "google_scanned_objects",
             "library_title": "Google Scanned Objects", "publisher": "Google",
             "mesh_url": API + "/{name}/{version}/files/meshes/model.obj",
             "thumbnail_url": API + "/{name}/{version}/files/thumbnails/0.jpg",
             "item_url": API + "/{name}/{version}/{name}.zip",
             "rights": [_rights(CC_BY, CC_BY), _rights("Creative Commons Attribution-NonCommercial 4.0 International",
                                                       CC_BY), _rights(CC_BY, None)],
             "records": records if records is not None else [
                 _record("Sterilite_Storage_Tote_Blue", "Sterilite Storage Tote, Blue\nStackable."),
                 # A better word match under a license off the list, and one whose metadata states none: never shown.
                 _record("Blue_Storage_Tote_NC", "Blue Storage Tote of Parts", rights=1),
                 _record("Storage_Tote_Unstated", "Storage Tote of Parts", rights=2),
                 _record("Tiny_Storage_Tote", "Storage Tote, Doll Size", size=(0.06, 0.04, 0.03)),
                 _record("Ceramic_Mug", "Ceramic Mug", size=(0.13, 0.1, 0.11), categories=["Bottles and Cans and Cups"])]}
    path = tmp_path / "object_library_gso.v1.json"
    path.write_text(json.dumps(index))
    return deploy({"google_scanned_objects": path})


def deploy(indexes: dict[str, Path]) -> dict[str, Path]:
    """Make ``indexes`` the deployed catalog for the rest of the test, as a release ships its own: compile and the
    company package look a sealed model up in it (``website_library_catalog.rights_now``). A later index of the same
    library replaces the earlier one, as a deploy would; ``tests/conftest.py`` restores the release's after the test."""
    current = library_catalog.INDEXES
    base = {} if current is library_catalog.SHIPPED_INDEXES else dict(current)
    library_catalog.INDEXES = {**base, **indexes}
    return indexes


def _entries(tmp_path: Path) -> list[dict]:
    return library.catalog(_index(tmp_path))


class Fetcher:
    """The injected network: the fixture's files by URL, and each request kept."""

    def __init__(self, *, fail: bool = False):
        self.fail, self.urls = fail, []

    def __call__(self, url, target, *, sha256, max_bytes):
        self.urls.append(url)
        if self.fail:
            raise OSError("no network")
        Path(target).write_bytes(THUMB if url.endswith(".jpg") else MESH)


TASK = {"schema_version": "website_site_task_context.v1", "request_id": "itest", "scene_id": "site-itest",
        "capture_id": "walkthrough-itest", "confirmed": True, "confirmed_at": "2026-11-16",
        "description": "Move the blue tote of parts from the bench to the shelf."}
TASK["context_digest"] = canonical_digest(TASK, digest_field="context_digest")


def _asking(answer):
    asked = []

    def ask(**kwargs):
        asked.append(kwargs)
        if isinstance(answer, BaseException):
            raise answer
        return {"answer": answer}
    return ask, asked


# --- rights: the license gate --------------------------------------------------------------------------------

def test_only_an_allow_listed_license_named_exactly_by_every_statement_is_admitted(tmp_path):
    assert library.license_of([("a", CC_BY), ("b", "https://creativecommons.org/licenses/by/4.0/")]) == ("CC-BY-4.0", None)
    assert library.license_of([("a", "CC0 1.0 Universal")]) == ("CC0-1.0", None)
    for statements, why in (([], "license_missing"), ([("a", CC_BY), ("b", None)], "license_not_allowed:b"),
                            ([("a", "Creative Commons Attribution-NonCommercial 4.0 International")],
                             "license_not_allowed:a"),
                            ([("a", "CC BY-SA 4.0")], "license_not_allowed:a"), ([("a", "cc by 4")], "license_not_allowed:a"),
                            ([("a", CC_BY), ("b", "CC0-1.0")], "licenses_disagree")):
        assert library.license_of(statements) == (None, why)
    entries = {entry["title"]: entry for entry in _entries(tmp_path)}
    assert entries["Blue Storage Tote of Parts"]["license_refused"] == "license_not_allowed:Gazebo Fuel's license"
    assert entries["Storage Tote of Parts"]["license_refused"] == "license_not_allowed:the model's own metadata"
    assert entries["Sterilite Storage Tote, Blue"]["license"] == "CC-BY-4.0"
    # The refused entries name the object better and are still never retrieved, shown or chosen.
    found = library.candidates(ROW, list(entries.values()))
    assert [row["entry"]["title"] for row in found] == ["Sterilite Storage Tote, Blue"]


def test_retrieval_drops_another_size_and_never_counts_a_colour(tmp_path):
    entries = _entries(tmp_path)
    found = library.candidates({**ROW, "name": "blue tote"}, entries)
    assert [row["entry"]["title"] for row in found] == ["Sterilite Storage Tote, Blue"]  # the doll's tote is too small
    assert library.candidates({**ROW, "name": "blue", "variety": None}, entries) == []
    assert library.tokens("SM_BoxCardboard_A01 boxes, 12 pieces") == ["box", "cardboard", "box"]


# --- the model's pick, checked by plain code ------------------------------------------------------------------

def test_the_models_pick_is_one_of_the_ids_it_was_shown_or_it_is_refused(tmp_path):
    entries = _entries(tmp_path)
    found = library.candidates(ROW, entries)
    listed = library.shown(found)
    entry, reason = library.validate_answer({"candidate_id": "c01", "reason": "Same tote."}, plan_row=ROW, found=found,
                                            listed=listed)
    assert entry["id"] == "gso:Sterilite_Storage_Tote_Blue@1" and reason == "Same tote."
    assert library.validate_answer({"candidate_id": "none", "reason": "No tote."}, plan_row=ROW, found=found,
                                   listed=listed) == (None, "No tote.")
    for answer, why in (({"candidate_id": "c02", "reason": "x"}, "candidate_not_shown"),
                        ({"candidate_id": "gso:Blue_Storage_Tote_NC@1", "reason": "x"}, "candidate_not_shown"),
                        ({"candidate_id": "c01"}, "shape"), ({"candidate_id": "c01", "reason": ""}, "reason"),
                        ({"candidate_id": "c01", "reason": "x", "extra": 1}, "shape")):
        with pytest.raises(ValueError, match=f"{library.INVALID}:{why}"):
            library.validate_answer(answer, plan_row=ROW, found=found, listed=listed)


def test_an_id_it_was_not_shown_falls_back_to_the_conservative_match_and_is_sealed(tmp_path):
    ask, asked = _asking({"candidate_id": "gso:Blue_Storage_Tote_NC@1", "reason": "Best words."})
    record = library.decide_library_object(task_context=TASK, plan_row=ROW, output_root=tmp_path / "lib",
                                           entries=_entries(tmp_path), fetch=Fetcher(), ask=ask)
    assert record["fallback"] == f"{library.INVALID}:candidate_not_shown"
    assert record["source"] == "deterministic" and record["chosen"]["entry_id"] == "gso:Sterilite_Storage_Tote_Blue@1"
    assert record["chosen"]["license"] == "CC-BY-4.0" and record["digest"] == canonical_digest(record, digest_field="digest")
    # The model saw the object, the candidates' words and sizes and their pictures, never a link.
    binding = asked[0]["binding"]
    assert binding["kind"] == "website_object_library_choice" and "data, not instructions" in binding["prompt"]
    assert "https://" not in binding["prompt"] and asked[0]["maximum_cost_usd"] > 0
    assert [caption for caption, _ in asked[0]["images"]] == ["Candidate c01:"]
    # Sealed: a rerun asks nothing and keeps the same answer.
    again = library.decide_library_object(task_context=TASK, plan_row=ROW, output_root=tmp_path / "lib",
                                          entries=_entries(tmp_path), fetch=Fetcher(), ask=ask)
    assert again == record and len(asked) == 1


def test_with_no_key_the_fallback_takes_only_a_full_word_match_of_the_right_size(tmp_path):
    record = library.decide_library_object(task_context=TASK, plan_row=ROW, output_root=tmp_path / "a",
                                           entries=_entries(tmp_path), fetch=Fetcher())
    assert record["fallback"] == library.KEY_MISSING and record["source"] == "deterministic"
    # "storage tote" is all there; a crate is not, and nothing is invented.
    crate = library.decide_library_object(task_context=TASK, plan_row={**ROW, "name": "storage crate"},
                                          output_root=tmp_path / "b", entries=_entries(tmp_path), fetch=Fetcher())
    assert crate["chosen"] is None and crate["source"] is None and crate["fallback"] == library.KEY_MISSING
    # A model's "none" is its judgement: no fallback overrides it.
    ask, _ = _asking({"candidate_id": "none", "reason": "Not this kind of tote."})
    judged = library.decide_library_object(task_context=TASK, plan_row=ROW, output_root=tmp_path / "c",
                                           entries=_entries(tmp_path), fetch=Fetcher(), ask=ask)
    assert judged["chosen"] is None and judged["source"] == "model" and judged["fallback"] == library.NO_MATCH


def test_privacy_rights_and_a_held_call_still_stop(tmp_path):
    from blueprint_pipeline.website_build_choice import IN_PROGRESS
    ask, _ = _asking(ValueError(IN_PROGRESS))
    with pytest.raises(ValueError, match=IN_PROGRESS):
        library.decide_library_object(task_context=TASK, plan_row=ROW, output_root=tmp_path, entries=_entries(tmp_path),
                                      fetch=Fetcher(), ask=ask)
    assert not (tmp_path / library.RECORD).exists()


# --- fetching: allow-listed, pinned, cached -------------------------------------------------------------------

def test_a_file_is_fetched_only_from_an_allowed_host_checked_and_cached_by_content(tmp_path):
    row = {"url": API + "/x/1/files/meshes/model.obj", "sha256": hashlib.sha256(MESH).hexdigest(),
           "size_bytes": len(MESH), "suffix": ".obj"}
    fetch = Fetcher()
    path = asset.fetch_pinned(row, tmp_path, fetch=fetch)
    assert path.read_bytes() == MESH and path.name == row["sha256"] + ".obj"
    assert asset.fetch_pinned(row, tmp_path, fetch=fetch) == path and len(fetch.urls) == 1  # cached
    with pytest.raises(ValueError, match="library_host_not_allowed"):
        asset.fetch_pinned({**row, "url": "https://example.org/model.obj"}, tmp_path / "b", fetch=fetch)
    with pytest.raises(ValueError, match="library_host_not_allowed"):
        asset.fetch_pinned({**row, "url": row["url"].replace("https", "http")}, tmp_path / "b", fetch=fetch)
    with pytest.raises(ValueError, match="library_file_too_large"):
        asset.fetch_pinned({**row, "size_bytes": asset.MAX_FILE_BYTES + 1}, tmp_path / "b", fetch=fetch)
    with pytest.raises(ValueError, match="library_file_digest_mismatch"):
        asset.fetch_pinned({**row, "sha256": "0" * 64}, tmp_path / "c", fetch=fetch)
    assert not list((tmp_path / "c").glob("*"))


# --- the handoff: the plan's object, sealed --------------------------------------------------------------------

def _sealed_plan(capture: Path, row=ROW) -> None:
    from blueprint_pipeline.website_build_plan_notes import DECISION_SCHEMA
    decided = {"schema_version": DECISION_SCHEMA, "applied": False, "build_plan": {"plan": {"objects": [row]}},
               "built": {"target_id": row["target_id"], "name": row["name"], "sourcing": row["sourcing"],
                         "editor_object_ids": row["editor_object_ids"]}}
    decided["digest"] = canonical_digest(decided, digest_field="digest")
    (capture / "pipeline" / "atlas").mkdir(parents=True, exist_ok=True)
    (capture / "pipeline" / "atlas" / "job_object_rule.json").write_text(json.dumps(decided))


def test_the_handoff_names_the_library_model_with_its_files_and_credit_once(tmp_path):
    _sealed_plan(tmp_path)
    ask, asked = _asking({"candidate_id": "c01", "reason": "The same stackable tote."})
    fetch = Fetcher()
    options = dict(capture_root=tmp_path, task_context=TASK, output_root=tmp_path / "library",
                   entries=_entries(tmp_path), fetch=fetch, ask=ask)
    removal = library.attach_library_subject({"entries": []}, **options)
    subject = removal[LIBRARY_SUBJECT]
    assert subject["target_id"] == "tote" and "not_used" not in subject
    assert subject["attribution"]["title"] == "Sterilite Storage Tote, Blue" and subject["attribution"]["author"] == "Google"
    assert subject["attribution"]["item_url"] == API + "/Sterilite_Storage_Tote_Blue/1/Sterilite_Storage_Tote_Blue.zip"
    assert subject["pick"]["source"] == "model" and subject["pick"]["reason"] == "The same stackable tote."
    assert Path(subject["files"][0]["path"]).read_bytes() == MESH
    assert library.attach_library_subject({"entries": []}, **options) == removal and len(asked) == 1
    # No built object (today's one-object route): unchanged.
    assert library.attach_library_subject({"entries": []}, **{**options, "capture_root": tmp_path / "none"}) == {
        "entries": []}


@pytest.mark.parametrize("case, reason", [("download", library.DOWNLOAD_FAILED), ("opens", library.NOT_RIGID),
                                          ("nothing", library.NO_CANDIDATE)])
def test_a_skipped_library_says_why_and_stops_nothing(tmp_path, case, reason):
    row = {**ROW, "rigidity": "articulated"} if case == "opens" else {**ROW, "name": "forklift", "variety": None} if case == "nothing" else ROW
    _sealed_plan(tmp_path, row)
    ask, _ = _asking({"candidate_id": "c01", "reason": "Same."})
    removal = library.attach_library_subject({"entries": []}, capture_root=tmp_path, task_context=TASK,
                                             output_root=tmp_path / "library", entries=_entries(tmp_path),
                                             fetch=Fetcher(fail=case == "download"), ask=ask)
    assert removal[LIBRARY_SUBJECT]["not_used"].startswith(reason) and "files" not in removal[LIBRARY_SUBJECT]


# --- compile: library, then the parametric shape, then the paid builder ----------------------------------------

def _configuration(tolerance=0.25):
    return {"authoring_backend": "artisan_v1", "source_observation_kind": "website_capture_frames",
            "metric_envelope": {"minimum_xyz_m": [0, 0, 0], "maximum_xyz_m": [1, 1, 1],
                                "maximum_dimension_relative_error": tolerance}}


def _subject(tmp_path: Path) -> dict:
    _sealed_plan(tmp_path)
    ask, _ = _asking({"candidate_id": "c01", "reason": "Same."})
    return library.attach_library_subject({"entries": []}, capture_root=tmp_path, task_context=TASK,
                                          output_root=tmp_path / "library", entries=_entries(tmp_path),
                                          fetch=Fetcher(), ask=ask)[LIBRARY_SUBJECT]


@pytest.mark.parametrize("estimate, status, scale, turns", [((0.6, 0.4, 0.3), "as_is", 1.0, 0),
                                                            ((0.42, 0.62, 0.29), "as_is", 1.0, 1),
                                                            ((0.9, 0.6, 0.45), "scaled_to_video_estimate", 1.5, 0)])
def test_the_library_model_comes_first_at_its_size_or_scaled_to_the_videos_estimate(tmp_path, estimate, status, scale,
                                                                                    turns):
    from blueprint_pipeline.website_synthetic_subject import AUTHORING_METHOD
    made = supplied_mesh_configuration(_configuration(), synthetic=None, synthetic_authoring=None,
                                       parametric={"target_id": "tote", "shape": "tote"}, subject_id="tote",
                                       sim_min=[1, 1, 0], sim_max=[1 + estimate[0], 1 + estimate[1], estimate[2]],
                                       library=_subject(tmp_path), library_root=tmp_path / "model")
    assert "authoring_backend" not in made and "parametric_design" not in made
    assert made["authoring_method"] == AUTHORING_METHOD and made["authoring_source"] == "library"
    assert made["authoring_cost_usd"] == 0.0 and made["appearance_inputs"] == "library_model"
    design = made["library_design"]
    assert (design["sizing"]["status"], design["sizing"]["quarter_turns_about_up"]) == (status, turns)
    assert design["sizing"]["scale"] == pytest.approx(scale) and design["physical_measurement"] is False
    assert made["geometry_support"] == "library_model_" + status
    assert design["attribution"]["license"] == "CC-BY-4.0" and design["converted_from"] == "OBJ"
    # The model file is the model as converted and scaled about its own origin, in metres with Z up.
    from pxr import Usd, UsdGeom
    model = Usd.Stage.Open(design["model_file"]["path"])
    assert UsdGeom.GetStageUpAxis(model) == "Z" and UsdGeom.GetStageMetersPerUnit(model) == 1.0
    points = UsdGeom.Mesh(model.GetPrimAtPath("/LibraryModel/Mesh")).GetPointsAttr().Get()
    assert max(p[2] for p in points) == pytest.approx(0.3 * scale, abs=1e-5)
    # Placed where the object stood, turned as it was fitted.
    candidate = asset.write_candidate(design, center_xyz_m=[2.0, 3.0, 0.5], path=tmp_path / "candidate.usda")
    assert candidate["source"] == "library" and candidate["authoring_cost_usd"] == 0.0
    stage = Usd.Stage.Open(str(tmp_path / "candidate.usda"))
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(
        stage.GetPrimAtPath("/Root/SourceObjectCandidate")).GetRange()
    expected = design["sizing"]["dimensions_m"]
    assert list(box.GetSize()) == pytest.approx(expected, abs=1e-5)
    assert list(box.GetMidpoint()) == pytest.approx([2.0, 3.0, 0.5], abs=1e-5)


def test_a_model_of_other_proportions_leaves_the_parametric_shape_then_the_paid_builder(tmp_path):
    subject = _subject(tmp_path)
    options = dict(synthetic=None, synthetic_authoring=None, subject_id="tote", sim_min=[0, 0, 0],
                   sim_max=[1.2, 0.15, 0.3], library=subject, library_root=tmp_path / "model")
    # A long flat object is no tote: the library is refused with its ratios, and the plan's carton is used.
    made = supplied_mesh_configuration(_configuration(), parametric={"target_id": "tote", "shape": "carton"}, **options)
    assert made["library_not_used"].startswith(asset.ASPECT_MISMATCH) and "library_design" not in made
    assert made["authoring_source"] == "parametric" and made["parametric_design"]["shape"] == "carton"
    # No shape either: today's builder, with both reasons.
    kept = supplied_mesh_configuration(_configuration(), parametric={"target_id": "tote", "shape": "cylinder"}, **options)
    assert kept["authoring_backend"] == "artisan_v1" and kept["library_not_used"].startswith(asset.ASPECT_MISMATCH)
    assert kept["parametric_not_used"] == "website_parametric_object_outside_estimated_bounds"
    # The handoff's own reason travels on; an opening object keeps its builder; another object changes nothing.
    skipped = supplied_mesh_configuration(_configuration(), parametric=None,
                                          **{**options, "library": {"target_id": "tote", "not_used": "x_reason"}})
    assert skipped == {**_configuration(), "library_not_used": "x_reason"}
    opening = supplied_mesh_configuration({**_configuration(), "mechanism": {}}, parametric=None, **options)
    assert opening["library_not_used"] == "website_object_library_object_opens" and "authoring_backend" in opening
    assert supplied_mesh_configuration(_configuration(), parametric=None, **{**options, "subject_id": "carton"}) == \
        _configuration()
    # A file changed since it was fetched is never used: its pinned bytes are fetched again first (stress round 2),
    # and with the network down it leaves the next source.
    Path(subject["files"][0]["path"]).write_bytes(_box_obj(0.6, 0.4, 0.31))
    changed = supplied_mesh_configuration(_configuration(), parametric=None, library_fetch=Fetcher(fail=True),
                                          **{**options, "sim_max": [0.6, 0.4, 0.3]})
    assert changed["library_not_used"] == "library_file_changed" and changed["authoring_backend"] == "artisan_v1"


# --- construction and the disclosure: supplied geometry, no provider -------------------------------------------

def _library_stage(tmp_path: Path):
    from blueprint_pipeline import task_evaluation_scene_configuration_submission_records as records
    from blueprint_pipeline.website_synthetic_subject import AUTHORING_ADAPTER
    configuration = records.stage_three_configuration(scene_id="site-1",
        replacement_identity={"id": "website-subject-tote", "version": "v1"}, source_instance_id="tote",
        authoring_target="tote", source_min=[0.7, 1.8, 0.0], source_max=[1.3, 2.2, 0.3], dimension_tolerance=0.25,
        physics_bounds={"mass_kg_bounds": [1, 2], "static_friction_bounds": [.5, .8],
                        "dynamic_friction_bounds": [.4, .5], "restitution_bounds": [.1, .3]})
    configuration.update(source_observation_kind="website_capture_frames")
    made = supplied_mesh_configuration(configuration, synthetic=None, synthetic_authoring=None, parametric=None,
                                       subject_id="tote", sim_min=[0.7, 1.8, 0.0], sim_max=[1.3, 2.2, 0.3],
                                       library=_subject(tmp_path), library_root=tmp_path / "model")
    stage = {**records.stage_sequence()[2], "adapter": {"id": AUTHORING_ADAPTER, "version": "v1"},
             "execution_class": "no_spend"}
    return made, stage


def test_construction_authors_the_library_model_as_supplied_geometry_and_calls_no_model(tmp_path):
    from blueprint_pipeline.task_evaluation_completed_scene_adapters import execute_provided_mesh_authoring
    configuration, stage = _library_stage(tmp_path)
    candidate = tmp_path / "source_object_candidate.library.usda"
    asset.write_candidate(configuration["library_design"], center_xyz_m=[1.0, 2.0, 0.15], path=candidate)
    data = candidate.read_bytes()
    dependency = {"output_artifacts": [{"role": "source_object_candidate_mesh", "path": str(candidate),
                                        "digest": "sha256:" + hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}]}
    (tmp_path / "stage-3.json").write_text(json.dumps(configuration))
    result = execute_provided_mesh_authoring(envelope={}, stage=stage, configuration=configuration,
        configuration_path=tmp_path / "stage-3.json", dependency_results=(dependency,), output_root=tmp_path / "out")
    receipt = json.loads((tmp_path / "out" / "replacement_authoring_receipt.v1.json").read_text())
    assert result["status"] == "completed" and receipt["model_called"] is False
    assert receipt["source_candidate_claim"] == "library_model_candidate_geometry"
    completion = receipt["candidate_physics_completion"]
    assert completion["metric_envelope_validation"]["status"] == "within_preregistered_metric_envelope"
    assert completion["collision_dimensions_m"] == pytest.approx([0.6, 0.4, 0.3], abs=0.002)


def test_the_disclosure_records_a_supplied_mesh_with_no_provider_and_names_the_pick(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_configuration_builtin_adapters as builtin
    from blueprint_pipeline import website_native_inputs as native
    from blueprint_pipeline.task_evaluation_scene_configuration_disclosure import render_inputs_disclosure_is_coherent
    from blueprint_pipeline.website_native_background import PREFIX
    from blueprint_pipeline.website_synthetic_subject import AUTHORING_ADAPTER
    configuration, _ = _library_stage(tmp_path)
    candidate = tmp_path / "candidate.usda"
    asset.write_candidate(configuration["library_design"], center_xyz_m=[1.0, 2.0, 0.15], path=candidate)
    data = candidate.read_bytes()
    row = {"contract_path": PREFIX + ".candidate", "materialized_path": str(candidate), "size_bytes": len(data),
           "digest": "sha256:" + hashlib.sha256(data).hexdigest(), "full_byte_service_account_readback_passed": True}
    third = {"stage_id": "stage-3", "adapter": {"id": AUTHORING_ADAPTER, "version": "v1"}, "execution_class": "no_spend"}
    envelope = {"recipe": {"stage_sequence": [{"stage_id": "stage-1"}, {}, third]}, "materialized_references": [row],
                "request": {"run_id": "run-1"}}
    # No provider request: the preflight accepts the library form and only it.
    assert native.preflight_website_authoring_request(envelope=envelope, configurations={"stage-3": configuration}) is None
    for value in ({key: v for key, v in configuration.items() if key != "library_design"},
                  {**configuration, "authoring_backend": "artisan_v1"}):
        with pytest.raises(ValueError, match="designed_subject_authoring_invalid"):
            native.preflight_website_authoring_request(envelope=envelope, configurations={"stage-3": value})
    monkeypatch.setattr(builtin, "_materialized_reference", lambda envelope, contract_path: (None, tmp_path / "rt.json"))
    monkeypatch.setattr(native, "_runtime", lambda path: {"appearance": {"digest": "sha256:" + "a" * 64}})
    monkeypatch.setattr(native, "_configurations", lambda envelope: {"stage-1": {}, "stage-3": configuration})
    monkeypatch.setattr(native, "validate_website_native_inputs", lambda **kwargs: {"captured_frame_count": 3})
    record = native.materialize_website_inputs(envelope=envelope, stage_one_configuration={}, output_root=tmp_path)
    assert record["supplied_mesh_authoring"] is True and record["captured_frame_derivatives_in_provider_packet"] is False
    assert record["library_model"] == {"entry_id": "gso:Sterilite_Storage_Tote_Blue@1",
                                       "library": "google_scanned_objects", "license": "CC-BY-4.0",
                                       "pick": configuration["library_design"]["pick"], "sizing": "as_is", "scale": 1.0,
                                       "provider_called": False, "authoring_cost_usd": 0.0}
    assert record["library_model"]["pick"]["source"] == "model"
    assert render_inputs_disclosure_is_coherent(record)


# --- the company's package: the model's own file, its credit and its license --------------------------------

#: The model's own file, the job's object part, which is built from it, and the layer of its simulated variations;
#: Google's own copyright notice, kept.
CREDIT = ("- scene/files/library_model.usda, object.usdz and variations.usda: Sterilite Storage Tote, Blue by Google (Copyright 2020 "
          "Google LLC.), from Google Scanned Objects "
          f"({API}/Sterilite_Storage_Tote_Blue/1/Sterilite_Storage_Tote_Blue.zip), under CC BY 4.0 "
          "(https://creativecommons.org/licenses/by/4.0/). Blueprint converted it from OBJ and scaled it to fit "
          "this job.")
CARVE_OUT = ("These files stay under their own license. Blueprint does not own them, and the terms Blueprint shares "
             "this package under do not restrict them.")


@pytest.mark.parametrize("with_library", [True, False])
def test_the_package_credits_a_library_model_in_its_own_file_and_says_nothing_without_one(tmp_path, with_library):
    from blueprint_pipeline import outbound_file_scan as scan
    from blueprint_pipeline import website_company_simulation as sim
    from tests.test_website_company_simulation import Job
    job = Job(tmp_path / "job")
    before = sim.simulation_source(job.capture, queue_root=job.queue)
    if with_library:
        made = supplied_mesh_configuration(_configuration(), synthetic=None, synthetic_authoring=None, parametric=None,
                                           subject_id="tote", sim_min=[0, 0, 0], sim_max=[0.9, 0.6, 0.45],
                                           library=_subject(job.capture / "lib"),
                                           library_root=job.capture / "pipeline/website_scene_preparation/model")
        preparation = job.capture / "pipeline" / "website_scene_preparation" / "preparation.json"
        preparation.write_text(json.dumps({"authoring_inputs": {"configuration": made}}))
    source = sim.simulation_source(job.capture, queue_root=job.queue)
    assert bool(source.get("library_models")) is with_library
    assert (sim.simulation_revision(source) == sim.simulation_revision(before)) is (not with_library)
    sim.build_package(source, tmp_path / "package.zip", literals=[])
    with zipfile.ZipFile(tmp_path / "package.zip") as archive:
        names = archive.namelist()
        readme = archive.read("README.txt").decode()
        archive.extractall(tmp_path / "unpacked")
    flat = " ".join(readme.split())
    assert ("Library models" in readme) is with_library and (CREDIT in flat) is with_library
    assert (CARVE_OUT in flat) is with_library
    assert ("scene/files/library_model.usda" in names) is with_library
    # The model is not one of the parts scene.usda places; the room's texture is still a file a part uses.
    assert "library_model.usda (/World/" not in readme and "- scene/files/: files the parts above use." in readme
    if with_library:
        # The section sits before the terms, its links whole on their lines; the terms are unchanged.
        assert readme.index("Library models") < readme.index("\nTerms\n")
        assert "Do not pass it on and do not use it" in readme
        from pxr import Usd, UsdGeom
        model = Usd.Stage.Open(str(tmp_path / "unpacked/scene/files/library_model.usda"))
        assert UsdGeom.Mesh(model.GetPrimAtPath("/LibraryModel/Mesh")).GetPointsAttr().Get()
    # The links the credit gives pass the outbound scan; any other link still does not.
    assert scan.scan_tree(tmp_path / "unpacked", []) == []
    assert scan.scan_file(_text(tmp_path, "see https://fuel.gazebosim.org/1.0/Other/models/x"), "a.txt", []).kind == \
        scan.WEB_LINK
    assert scan.scan_file(_text(tmp_path, f"see {API}/x/1/x.zip?token=abcdefghijkl"), "b.txt", []).kind == scan.KEY


def _text(tmp_path: Path, text: str) -> Path:
    path = tmp_path / f"t{abs(hash(text))}.txt"
    path.write_text(text)
    return path


def _library_topologies(path: Path) -> list[tuple]:
    """Independently of the package's own reader: every mesh of a USD file or usdz, composed, as its topology."""
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(path))
    return [(tuple(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get() or ()),
             tuple(UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get() or ()))
            for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()) if prim.IsA(UsdGeom.Mesh)]


def _job_with_a_library_object(tmp_path: Path):
    """A job whose object part is construction's no-spend authoring of a library model, with a copy of the model in
    the room's collision file, and the preparation that names the model; returns the job and the model's topology."""
    from pxr import Usd, UsdGeom
    from blueprint_pipeline.task_evaluation_completed_scene_adapters import execute_provided_mesh_authoring
    from tests.test_website_company_simulation import Job
    job = Job(tmp_path / "job")
    configuration, stage = _library_stage(tmp_path)
    design = configuration["library_design"]
    candidate = tmp_path / "source_object_candidate.library.usda"
    asset.write_candidate(design, center_xyz_m=[1.0, 2.0, 0.15], path=candidate)
    data = candidate.read_bytes()
    dependency = {"output_artifacts": [{"role": "source_object_candidate_mesh", "path": str(candidate),
                                        "digest": "sha256:" + hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}]}
    (tmp_path / "stage-3.json").write_text(json.dumps(configuration))
    execute_provided_mesh_authoring(envelope={}, stage=stage, configuration=configuration,
                                    configuration_path=tmp_path / "stage-3.json", dependency_results=(dependency,),
                                    output_root=tmp_path / "authored")
    job.object.write_bytes((tmp_path / "authored" / "provided_mesh_candidate.usdz").read_bytes())
    model_stage = Usd.Stage.Open(design["model_file"]["path"])
    model = UsdGeom.Mesh(model_stage.GetPrimAtPath("/LibraryModel/Mesh"))
    collision = Usd.Stage.Open(str(job.collision))
    copy = UsdGeom.Mesh.Define(collision, "/Collision/ModelCopy")
    for name in ("Points", "FaceVertexCounts", "FaceVertexIndices"):
        getattr(copy, f"Create{name}Attr")(getattr(model, f"Get{name}Attr")().Get())
    collision.GetRootLayer().Save()
    job.seal()
    made = supplied_mesh_configuration(_configuration(), synthetic=None, synthetic_authoring=None, parametric=None,
                                       subject_id="tote", sim_min=[0.7, 1.8, 0.0], sim_max=[1.3, 2.2, 0.3],
                                       library=_subject(job.capture / "lib"),
                                       library_root=job.capture / "pipeline/website_scene_preparation/model")
    preparation = job.capture / "pipeline/website_scene_preparation/preparation.json"
    preparation.write_text(json.dumps({"authoring_inputs": {"configuration": made}}))
    job.seal_record(bound_to=preparation)  # the half was compiled from this preparation's revision
    return job, made, set(_library_topologies(Path(design["model_file"]["path"])))


#: Every file a company receives holding the model's geometry, by the state of the model's own file at packaging.
# ``variations.usda`` composes the object with its simulated variations (``website_object_variations``): named too.
HOLDING = {"kept": ["object.usdz", "scene/files/library_model.usda", "scene/room_collision.usdc", "variations.usda"],
           "deleted": ["object.usdz", "scene/room_collision.usdc", "variations.usda"],
           "changed": ["object.usdz", "scene/room_collision.usdc", "variations.usda"]}


@pytest.mark.parametrize("model_file", ["kept", "deleted", "changed"])
def test_every_shipped_file_holding_the_models_geometry_is_named_in_its_credit(tmp_path, model_file):
    """The object part built from the model and a collision copy of it, as well as the model's own file: each file a
    company receives that holds the model's geometry is named in the model's credit, under the settled carve-out.
    The credit never depends on the model's own file: gone or changed since it was written, that file is not
    shipped, and the credit still names every file that is."""
    from blueprint_pipeline import website_company_simulation as sim
    job, made, reference = _job_with_a_library_object(tmp_path)
    model_path = Path(made["library_design"]["model_file"]["path"])
    if model_file == "deleted":
        model_path.unlink()
    elif model_file == "changed":
        model_path.write_text(model_path.read_text() + "\n")
    sim.build_package(sim.simulation_source(job.capture, queue_root=job.queue), tmp_path / "package.zip", literals=[])
    with zipfile.ZipFile(tmp_path / "package.zip") as archive:
        readme = " ".join(archive.read("README.txt").decode().split())
        archive.extractall(tmp_path / "unpacked")
    bullet = next(line for line in readme.split(" - ") if "by Google (Copyright 2020 Google LLC.)" in line)
    named = bullet.split(": ", 1)[0].replace(" and ", ", ").split(", ")
    unpacked = tmp_path / "unpacked"
    holding = sorted(path.relative_to(unpacked).as_posix() for path in unpacked.rglob("*")
                     if path.suffix in (".usd", ".usda", ".usdc", ".usdz") and path.name != "scene.usda"
                     and reference & set(_library_topologies(path)))
    assert holding == HOLDING[model_file]
    missing = [name for name in holding if name not in named]
    assert not missing, f"shipped files with the model's geometry the credit does not name: {missing}"
    assert sorted(named) == holding and "scene/room.usdc" not in named  # and no file that is not shipped
    assert (unpacked / "scene/files/library_model.usda").is_file() is (model_file == "kept")
    assert ("These files stay under their own license. Blueprint does not own them, and the terms Blueprint shares "
            "this package under do not restrict them.") in readme


@pytest.mark.parametrize("break_credit", [lambda design: design.pop("attribution"),
                                          lambda design: design["attribution"].pop("author"),
                                          lambda design: design["attribution"].update(title=" "),
                                          lambda design: design.update(attribution="Google")])
def test_an_object_from_a_library_model_whose_credit_cannot_be_read_never_leaves(tmp_path, break_credit):
    """Rights is a hard stop: without its credit, the object built from a library model may not leave in a package
    (refused, never shipped bare); a package without the object is not refused, so the half gives way to the room
    floor (``refused_half``), as a half whose own scene cannot be staged does."""
    from blueprint_pipeline import website_company_simulation as sim
    from blueprint_pipeline import website_library_notes as notes
    from tests.test_website_company_simulation import World, _manifest
    job, made, _ = _job_with_a_library_object(tmp_path)
    break_credit(made["library_design"])
    preparation = job.capture / "pipeline/website_scene_preparation/preparation.json"
    preparation.write_text(json.dumps({"authoring_inputs": {"configuration": made}}))
    job.seal_record(bound_to=preparation)  # its own preparation reads; only the credit in it does not
    source = sim.simulation_source(job.capture, queue_root=job.queue)
    assert source["library_refusal"] == notes.CREDIT_UNREADABLE and not source.get("library_models")
    with pytest.raises(sim.PackageRefused) as refused:
        sim.build_package(source, tmp_path / "package.zip", literals=[])
    assert refused.value.reason == notes.CREDIT_UNREADABLE and isinstance(refused.value, sim.ShapeRefused)
    world = World(job)
    result = world.publish()
    assert result["status"] == "refused" and result.get("reason") == notes.CREDIT_UNREADABLE and not world.delivered
    assert _manifest(job)["refused_half"]["reason"] == notes.CREDIT_UNREADABLE  # its floor ships in its place
    room = {**source, "files": {prim: row for prim, row in source["files"].items() if row["role"] != "task_object"}}
    assert notes.refusal(room) is None


def _index_files() -> list[Path]:
    """Every file a library reads: each index, and each part an index lists."""
    files = []
    for path in library_catalog.SHIPPED_INDEXES.values():
        files.append(path)
        files += [path.parent / part["file"] for part in json.loads(path.read_text()).get("parts") or []]
    return files


def test_both_library_indexes_ship_in_the_wheel_and_the_sdist():
    """An installed package carries both indexes and every part of them: matched by the package data and by
    MANIFEST.in, so a library is never silently empty."""
    import fnmatch
    import tomllib
    root = Path(__file__).resolve().parents[1]
    package = root / "src" / "blueprint_pipeline"
    data = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]
    manifest = [line.split(None, 1)[1].strip() for line in (root / "MANIFEST.in").read_text().splitlines()
                if line.startswith("include ")]
    files = _index_files()
    assert len(files) > len(library_catalog.SHIPPED_INDEXES)  # the SimReady index is in parts
    for path in files:
        assert path.is_file(), path
        relative = path.relative_to(package).as_posix()
        assert any(fnmatch.fnmatch(relative, pattern) for pattern in data["blueprint_pipeline"]), relative
        assert any(fnmatch.fnmatch(path.relative_to(root).as_posix(), pattern) for pattern in manifest), relative


def test_every_library_index_file_fits_the_release_source_leaf_budget():
    """The release source manifest refuses any tracked file over its leaf budget; every object library file,
    listed or not, stays under it (read from the manifest builder, never copied)."""
    from scripts.release_source_manifest import MAX_LEAF_BYTES
    found = sorted((library.DATA).glob("object_library_*"))
    assert set(_index_files()) <= set(found)
    for path in found:
        assert path.stat().st_size <= MAX_LEAF_BYTES, (path.name, path.stat().st_size)


@pytest.mark.parametrize("damage, reason", [("delete", library.PART_MISSING), ("change", library.PART_MISMATCH),
                                            ("truncate", library.PART_MISMATCH), ("count", library.PART_MISMATCH)])
def test_a_missing_or_changed_index_part_makes_that_library_unavailable_never_partial(tmp_path, monkeypatch, damage,
                                                                                     reason):
    """A library is read whole or not at all: a part missing, changed or not the count its index names leaves that
    library out, said why where the choice records its catalog, while the other library still answers."""
    import shutil
    from scripts import build_simready_object_index as builder
    (tmp_path / "gso").mkdir()
    indexes = _index(tmp_path / "gso")
    simready = tmp_path / "simready" / "object_library_simready.v1.json"
    simready.parent.mkdir()
    index = {"schema_version": "object_library_simready_index.v1", "library": "nvidia_simready",
             "library_title": "NVIDIA SimReady",
             "item_url": "https://huggingface.co/datasets/{repo}/resolve/{commit}/{asset}/x.json",
             "file_url": "https://huggingface.co/datasets/{repo}/resolve/{commit}/{asset}/{path}",
             "sources": [{"repo": "nvidia/simready-assets", "commit": "6" * 40}],
             "rights": [{"package_license": "CC-BY-4.0", "asset_license": "CC-BY 4.0", "author": "NVIDIA"}],
             "records": [{"asset": f"storage_tote_{i:03d}", "root": "simready_usd/r.usd", "size_m": [0.6, 0.4, 0.3],
                          "bom": {"sha256": "a" * 64, "size_bytes": 10}, "source": 0, "rights": 0}
                         for i in range(40)], "unreadable": []}
    parts = builder.write_index(index, simready, max_part_bytes=2000)
    assert len(parts) > 2 and all(part["size_bytes"] <= 2000 for part in parts)
    both = {**indexes, "nvidia_simready": simready}
    assert len(library.catalog(both)) == 5 + 40  # whole, in the index's order
    assert [entry["title"] for entry in library.catalog(both)[5:]] == [row["asset"] for row in index["records"]]
    damaged = tmp_path / "damaged"
    shutil.copytree(simready.parent, damaged)
    target = damaged / parts[1]["file"]
    if damage == "delete":
        target.unlink()
    elif damage == "change":
        target.write_text(target.read_text().replace("storage_tote", "storage_tank", 1))
    elif damage == "truncate":
        target.write_text(target.read_text()[:-40])
    else:
        listed = json.loads((damaged / simready.name).read_text())
        listed["parts"][1]["record_count"] += 1
        (damaged / simready.name).write_text(json.dumps(listed))
    partial = {**indexes, "nvidia_simready": damaged / simready.name}
    entries = library.catalog(partial)
    assert {entry["library"] for entry in entries} == {"google_scanned_objects"}  # none of SimReady, not some
    status = {row["library"]: row for row in library.catalog_status(partial)}
    assert status["nvidia_simready"]["status"] == "unavailable"
    assert status["nvidia_simready"]["reason"] == f"{reason}:{parts[1]['file']}"
    assert status["google_scanned_objects"]["status"] == "read"
    # The job's choice records it beside its other reasons: the catalog it read, and why a library offered nothing.
    monkeypatch.setattr(library_catalog, "INDEXES", partial)
    record = library.decide_library_object(task_context=TASK, plan_row={**ROW, "name": "forklift", "variety": None},
                                           output_root=tmp_path / "choice", fetch=Fetcher())
    assert {row["library"]: row["status"] for row in record["catalog_libraries"]} == {
        "google_scanned_objects": "read", "nvidia_simready": "unavailable"}
    assert record["fallback"] == f"{library.NO_CANDIDATE}:unavailable:nvidia_simready={reason}:{parts[1]['file']}"
