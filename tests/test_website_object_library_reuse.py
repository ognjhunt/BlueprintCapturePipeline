"""A library model reused, rebuilt and shipped (the $0 stress run of round 2, 2026-11-24): rights read as they stand
now, a passing failure never traded for the paid builder, never a partial model, and the README saying what the
object is.

Offline, at $0: the library tests' fixture catalogs (Google Scanned Objects, and a SimReady package written with
``pxr``), each deployed as the release's catalog for its test (``tests.test_website_object_library.deploy``), injected
fetchers and model answers, no Gemini key, nothing paid for. The handoff's own ``attach_library_subject``, compile's
``supplied_mesh_configuration`` and the company package's real builder are used throughout.
"""
# Covers: src/blueprint_pipeline/website_object_library.py
# Covers: src/blueprint_pipeline/website_library_catalog.py
# Covers: src/blueprint_pipeline/website_library_asset.py
# Covers: src/blueprint_pipeline/website_library_notes.py
# Covers: src/blueprint_pipeline/website_supplied_mesh.py
# Covers: src/blueprint_pipeline/website_half_preparation.py
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("pxr")

from blueprint_pipeline import website_company_simulation as sim  # noqa: E402
from blueprint_pipeline import website_half_preparation as half  # noqa: E402
from blueprint_pipeline import website_library_asset as asset  # noqa: E402
from blueprint_pipeline import website_library_catalog as library_catalog  # noqa: E402
from blueprint_pipeline import website_library_notes as notes  # noqa: E402
from blueprint_pipeline import website_object_library as library  # noqa: E402
from blueprint_pipeline.website_build_plan import apply_to_removal  # noqa: E402
from blueprint_pipeline.website_supplied_mesh import LIBRARY_SUBJECT, PARAMETRIC_SUBJECT, \
    supplied_mesh_configuration  # noqa: E402
from tests import test_website_object_library_simready as simready  # noqa: E402
from tests.test_website_company_simulation import Job  # noqa: E402
from tests.test_website_object_library import (  # noqa: E402
    TASK, Fetcher, _asking, _configuration, _index, _job_with_a_library_object, _record)
from tests.test_website_object_library_single_object import ENTRY, REMOVAL, ROW, _sealed_plan  # noqa: E402

PREPARATION = ("pipeline", "website_scene_preparation", "preparation.json")


def _attach(capture: Path, removal, *, entries, fetch, answer=None):
    ask, asked = _asking(answer or {"candidate_id": "c01", "reason": "The same kind of object."})
    out = library.attach_library_subject(dict(removal), capture_root=capture, task_context=TASK,
                                         output_root=capture / "handoff" / "object_library", entries=entries,
                                         fetch=fetch, ask=ask)
    return out, asked


def _compile(removal, *, subject_id, extents, root, fetch=None):
    return supplied_mesh_configuration(_configuration(), synthetic=None, synthetic_authoring=None,
                                       parametric=removal.get(PARAMETRIC_SUBJECT), subject_id=subject_id,
                                       sim_min=[0, 0, 0], sim_max=list(extents), library=removal.get(LIBRARY_SUBJECT),
                                       library_root=root, library_fetch=fetch or Fetcher(fail=True))


def _readme(job, tmp_path: Path, name="package.zip") -> str:
    sim.build_package(sim.simulation_source(job.capture, queue_root=job.queue), tmp_path / name, literals=[])
    with zipfile.ZipFile(tmp_path / name) as archive:
        return archive.read("README.txt").decode()


# --- the README says what each source is ------------------------------------------------------------------------

def test_a_library_object_is_said_to_be_a_stand_in_of_the_same_kind_not_the_filmed_object(tmp_path):
    """A library model got only its credit, which reads as the object's maker and source; nothing said it is a model
    of the same kind of object, picked and sized from the video, and not the site's own. Now a plain note does."""
    job, _made, _ = _job_with_a_library_object(tmp_path)
    readme = " ".join(_readme(job, tmp_path).split())
    assert half.LIBRARY_NOTE in readme and "Library models" in readme
    assert half.LIBRARY_NOTE == ("The job's object is a model of the same kind of object from a public library, picked "
                                 "and sized from the video: it is not the site's own object.")
    assert half.GENERATED_NOTE not in readme and "plain tote shape" not in readme


@pytest.mark.parametrize("route, expected", [
    ("parametric", "The job's object is a plain tote shape Blueprint made for it, sized from the video"),
    ("artisan", "The job's object is a 3D model Blueprint generated from frames of the video")])
def test_the_one_object_routes_readme_names_what_the_object_is(tmp_path, route, expected):
    """Control, end to end on the one-object route: no library match, then the plan's tote shape (fits) or ARTISAN
    (the plan names no shape); the half's own preparation carries the configuration the route made."""
    capture = tmp_path / "capture"
    row = {**ROW, "name": "storage crate", "sourcing": "parametric" if route == "parametric" else "generate",
           "parametric_shape": "tote" if route == "parametric" else None}
    _sealed_plan(capture, [row])
    removal = apply_to_removal(REMOVAL, capture)
    removal, _ = _attach(capture, removal, entries=library.catalog(_index(tmp_path)), fetch=Fetcher(),
                         answer={"candidate_id": "none", "reason": "No crate among them."})
    made = _compile(removal, subject_id="tote", extents=(0.6, 0.4, 0.3), root=tmp_path / "model")
    assert made.get("authoring_source") == ("parametric" if route == "parametric" else None)
    job = Job(tmp_path / "job")
    path = job.capture.joinpath(*PREPARATION)
    path.write_text(json.dumps({"authoring_inputs": {"configuration": made}}))
    job.seal_record(bound_to=path)
    readme = " ".join(_readme(job, tmp_path).split())
    assert expected in readme and "Library models" not in readme and half.LIBRARY_NOTE not in readme


# --- a download failing on the way is tried again, never traded for the paid builder ------------------------------

def _box_route(tmp_path: Path):
    """A one-object job whose object is a cardboard box with a SimReady model of it."""
    indexes, fetch = simready._fixture(tmp_path)
    capture = tmp_path / "capture"
    entry = {**ENTRY, "target_id": "box", "semantic_label": "cardboard box"}
    _sealed_plan(capture, [{**simready.ROW, "sourcing": "generate", "parametric_shape": None}])
    return capture, {**REMOVAL, "entries": [entry]}, library.catalog(indexes), fetch


class _FailsOnce:
    """The network drops on the package's payload layer, once: the bill and the root are already in the cache."""

    def __init__(self, inner):
        self.inner, self.failed = inner, False

    def __call__(self, url, target, *, sha256, max_bytes):
        if url.endswith("geometries.usda") and not self.failed:
            self.failed = True
            Path(target).write_bytes(b"#usda 1.0\n(partial")  # what a dropped transfer leaves
            raise OSError("Connection reset by peer")
        return self.inner(url, target, sha256=sha256, max_bytes=max_bytes)


def test_a_download_that_fails_midway_is_tried_again_before_the_paid_builder(tmp_path):
    """A reset on the second of three files sealed ``not_used: download_failed``, so every rerun skipped the free
    model and the job paid ARTISAN's $12 for it. Now a failure on the way is a wait: it is raised as an outage, seals
    nothing, and the job's own retry, with the network back, builds the free model."""
    from blueprint_pipeline.website_preparation_retry import OUTAGE, classify
    capture, removal, entries, fetch = _box_route(tmp_path)
    with pytest.raises(ValueError, match=library.DOWNLOAD_FAILED) as waited:
        _attach(capture, removal, entries=entries, fetch=_FailsOnce(fetch))
    assert classify(str(waited.value)) == OUTAGE  # the handoff waits and retries it, as for any outage
    assert not (capture / "handoff" / "object_library" / library.SUBJECT_RECORD).exists()
    again, asked = _attach(capture, removal, entries=entries, fetch=fetch)  # the network is back
    assert not asked  # the pick was sealed; only the download is tried again
    made = _compile(again, subject_id="box", extents=(0.6, 0.4, 0.3), root=tmp_path / "model")
    assert made.get("authoring_source") == "library" and "authoring_backend" not in made


def test_a_download_that_can_never_succeed_leaves_the_next_source_recorded(tmp_path):
    """Control: bytes that never match their pinned sha256 are no outage; the job goes on with the next source and
    says why, as before."""
    capture, removal, entries, fetch = _box_route(tmp_path)

    def tampered(url, target, *, sha256, max_bytes):
        fetch(url, target, sha256=sha256, max_bytes=max_bytes)
        if url.endswith("geometries.usda"):
            Path(target).write_bytes(Path(target).read_bytes() + b"\n")
    out, _ = _attach(capture, removal, entries=entries, fetch=tampered)
    assert out[LIBRARY_SUBJECT]["not_used"] == f"{library.DOWNLOAD_FAILED}:library_file_digest_mismatch"


def test_a_cached_library_file_changed_before_compile_is_fetched_again_not_paid_for(tmp_path):
    """The cache is keyed by the pinned sha256, but compile only checked it: a cached file damaged between the
    handoff and compile sent the object to the paid builder. Now compile fetches the pinned bytes again first."""
    capture = tmp_path / "capture"
    _sealed_plan(capture)
    removal, _ = _attach(capture, REMOVAL, entries=library.catalog(_index(tmp_path)), fetch=Fetcher())
    mesh = Path(next(row["path"] for row in removal[LIBRARY_SUBJECT]["files"] if row["role"] == "mesh"))
    pinned = mesh.read_bytes()
    mesh.write_bytes(pinned + b"\n# damaged\n")
    fetch = Fetcher()
    made = _compile(removal, subject_id="tote", extents=(0.6, 0.4, 0.3), root=tmp_path / "model", fetch=fetch)
    assert made.get("authoring_source") == "library" and mesh.read_bytes() == pinned
    assert fetch.urls == [removal[LIBRARY_SUBJECT]["attribution"]["files"][0]["url"]]
    mesh.unlink()  # gone altogether: fetched again the same way
    again = _compile(removal, subject_id="tote", extents=(0.6, 0.4, 0.3), root=tmp_path / "model2", fetch=Fetcher())
    assert again.get("authoring_source") == "library" and mesh.read_bytes() == pinned


def test_a_model_file_changed_after_authoring_is_not_shipped_and_the_credit_still_names_the_object(tmp_path):
    """Control: the model's own file changed after the object was authored from it; the package leaves it out and the
    credit still names the object part built from it."""
    job, made, _ = _job_with_a_library_object(tmp_path)
    model = Path(made["library_design"]["model_file"]["path"])
    model.write_text(model.read_text() + "\n")
    readme = " ".join(_readme(job, tmp_path).split())
    assert "object.usdz" in readme.split("Library models", 1)[1] and "library_model.usda" not in readme


# --- rights read as they stand now ------------------------------------------------------------------------------

V2_REFUSES = [_record("Sterilite_Storage_Tote_Blue", "Sterilite Storage Tote, Blue\nStackable.", rights=1)]


def test_a_model_whose_license_a_new_index_no_longer_allows_is_not_used_or_shipped(tmp_path):
    """Rights is a hard stop. The job sealed its choice under index v1 (CC BY 4.0); a new index (a deploy) says the
    same model is CC BY-NC. The sealed choice and subject were reused as they stood and the package credited the model
    under CC BY 4.0. Now the handoff looks the model up in the catalog it is given, and it is not used."""
    capture = tmp_path / "capture"
    _sealed_plan(capture)
    for name in ("v1", "v2"):
        (tmp_path / name).mkdir()
    v1 = library.catalog(_index(tmp_path / "v1"))
    first, _ = _attach(capture, REMOVAL, entries=v1, fetch=Fetcher())
    assert first[LIBRARY_SUBJECT]["attribution"]["license"] == "CC-BY-4.0"
    v2 = library.catalog(_index(tmp_path / "v2", V2_REFUSES))
    assert v2[0]["license"] is None  # the new index refuses it
    again, asked = _attach(capture, REMOVAL, entries=v2, fetch=Fetcher())
    subject = again[LIBRARY_SUBJECT]
    assert subject["not_used"].startswith(library_catalog.REFUSED_NOW) and "attribution" not in subject
    assert "files" not in subject and not asked
    made = _compile(again, subject_id="tote", extents=(0.6, 0.4, 0.3), root=tmp_path / "model")
    assert made.get("authoring_source") != "library" and made["library_not_used"] == subject["not_used"]
    # The sealed records are left as they were: what the job chose, and when, stays on record.
    sealed = json.loads((capture / "handoff/object_library" / library.SUBJECT_RECORD).read_text())
    assert sealed["attribution"]["license"] == "CC-BY-4.0"


def test_compile_reads_the_deployed_catalog_whatever_the_handoff_sealed(tmp_path):
    """The handoff ran under v1 and sealed the model; v2 is deployed before compile runs on the same manifest:
    compile looks the model up in the deployed catalog and builds the next source, said why."""
    capture = tmp_path / "capture"
    _sealed_plan(capture)
    (tmp_path / "v1").mkdir()
    (tmp_path / "v2").mkdir()
    removal, _ = _attach(capture, REMOVAL, entries=library.catalog(_index(tmp_path / "v1")), fetch=Fetcher())
    assert "attribution" in removal[LIBRARY_SUBJECT]
    _index(tmp_path / "v2", V2_REFUSES)  # the deploy
    made = _compile(removal, subject_id="tote", extents=(0.6, 0.4, 0.3), root=tmp_path / "model")
    assert made["library_not_used"].startswith(library_catalog.REFUSED_NOW) and "library_design" not in made
    (tmp_path / "v3").mkdir()
    _index(tmp_path / "v3", [])  # gone from the catalog altogether: not used either
    gone = _compile(removal, subject_id="tote", extents=(0.6, 0.4, 0.3), root=tmp_path / "model")
    assert gone["library_not_used"] == library_catalog.NOT_IN_CATALOG


def test_a_package_built_after_the_index_refused_the_model_ships_the_room_floor(tmp_path):
    """The half was compiled under v1 from the library model; v2 refuses it before the package is built. The package
    may not leave with that object (rights), and the room floor ships instead, flagged."""
    job, _made, _ = _job_with_a_library_object(tmp_path)
    assert not sim.simulation_source(job.capture, queue_root=job.queue).get("library_refusal")
    (tmp_path / "v2").mkdir()
    _index(tmp_path / "v2", V2_REFUSES)  # the deploy
    source = sim.simulation_source(job.capture, queue_root=job.queue)
    assert source["library_refusal"] == notes.LICENSE_REFUSED and not source.get("library_models")
    with pytest.raises(sim.ShapeRefused) as refused:
        sim.build_package(source, tmp_path / "package.zip", literals=[])
    assert refused.value.reason == notes.LICENSE_REFUSED
    room = {**source, "files": {prim: row for prim, row in source["files"].items() if row["role"] != "task_object"}}
    assert notes.refusal(room) is None  # the room floor is not refused


def test_a_dataset_refused_after_a_job_chose_from_it_never_ships(tmp_path, monkeypatch):
    """The same for a whole dataset left out later (``DATASETS_REFUSED``, how M-XR was refused on 2026-11-23): a job
    that sealed a model of it before kept it."""
    indexes, fetch = simready._fixture(tmp_path)
    monkeypatch.setattr(library_catalog, "DATASETS_REFUSED", {})
    before = library.catalog(indexes)
    capture, removal, _entries, _ = _box_route(tmp_path / "route")
    first, _ = _attach(capture, removal, entries=before, fetch=fetch)
    assert "attribution" in first[LIBRARY_SUBJECT]
    monkeypatch.setattr(library_catalog, "DATASETS_REFUSED", {simready.REPO: "attribution_link_not_carried"})
    after = library.catalog(indexes)
    assert after[0]["license"] is None
    again, _ = _attach(capture, removal, entries=after, fetch=fetch)
    assert again[LIBRARY_SUBJECT]["not_used"] == f"{library_catalog.REFUSED_NOW}:attribution_link_not_carried"
    made = _compile(again, subject_id="box", extents=(0.6, 0.4, 0.3), root=tmp_path / "model")
    assert made.get("authoring_source") != "library", "a model of a dataset now refused is still authored"
    # Compile reads the refusal from the deployed catalog too, whatever the manifest says.
    rebuilt = _compile(first, subject_id="box", extents=(0.6, 0.4, 0.3), root=tmp_path / "model2")
    assert rebuilt["library_not_used"] == f"{library_catalog.REFUSED_NOW}:attribution_link_not_carried"


def test_a_missing_index_part_leaves_the_next_source_and_says_why(tmp_path, monkeypatch):
    """Control: one part of the SimReady index is missing; that library offers nothing, the record says why, and
    the job goes to the parametric shape."""
    indexes, fetch = simready._fixture(tmp_path)
    index_path = indexes["nvidia_simready"]
    index = json.loads(index_path.read_text())
    records = index.pop("records")
    part = tmp_path / "object_library_simready.v1.part-01.json"
    data = json.dumps({"records": records}).encode()
    index["parts"] = [{"file": part.name, "sha256": hashlib.sha256(data).hexdigest(), "record_count": 1}]
    index_path.write_text(json.dumps(index))  # the part is listed, never written
    monkeypatch.setattr(library_catalog, "INDEXES", indexes)
    library._catalog.cache_clear()
    capture = tmp_path / "capture"
    entry = {**ENTRY, "target_id": "box", "semantic_label": "cardboard box"}
    _sealed_plan(capture, [simready.ROW])
    removal = apply_to_removal({**REMOVAL, "entries": [entry]}, capture)
    ask, asked = _asking({"candidate_id": "c01", "reason": "x"})
    out = library.attach_library_subject(removal, capture_root=capture, task_context=TASK,
                                         output_root=capture / "handoff" / "object_library", fetch=fetch, ask=ask)
    library._catalog.cache_clear()
    assert out[LIBRARY_SUBJECT]["not_used"].startswith(library.NO_CANDIDATE + ":unavailable:nvidia_simready=")
    made = _compile(out, subject_id="box", extents=(0.6, 0.4, 0.3), root=tmp_path / "model")
    assert made["authoring_source"] == "parametric" and not asked


# --- never a partial model --------------------------------------------------------------------------------------

def _package_with_a_lid_the_bill_omits(tmp_path: Path):
    """A SimReady package whose root composes two payloads (the box and its lid) while its bill lists only the box's
    layer: the lid's layer is never fetched."""
    from pxr import Usd, UsdGeom
    files = simready._package(tmp_path)
    folder = tmp_path / "package-source"
    lid = Usd.Stage.CreateNew(str(folder / "simready_usd/payloads/lid.usda"))
    UsdGeom.SetStageUpAxis(lid, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(lid, 0.01)
    lid.SetDefaultPrim(UsdGeom.Xform.Define(lid, "/RootNode").GetPrim())
    mesh = UsdGeom.Mesh.Define(lid, "/RootNode/Lid")
    mesh.CreatePointsAttr([(x, y, z) for z in (-20, 20) for y in (30, 60) for x in (-30, 30)])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3])
    lid.GetRootLayer().Save()
    root = Usd.Stage.Open(str(folder / simready.ROOT))
    root.GetPrimAtPath("/RootNode").GetPayloads().AddPayload("./payloads/lid.usda")
    root.GetRootLayer().Save()
    files[simready.ROOT] = (folder / simready.ROOT).read_bytes()
    bom = simready._bom(files)  # the lid's layer is not listed
    served = {simready.BASE + path: data for path, data in files.items()}
    served[simready.BASE + simready.BOM] = bom
    chosen = {"format": "simready_usd", "package": {"base_url": simready.BASE, "root": simready.ROOT},
              "files": [{"role": "bom", "url": simready.BASE + simready.BOM, "sha256": hashlib.sha256(bom).hexdigest(),
                         "size_bytes": len(bom), "suffix": ".json"}]}
    return chosen, simready.Fetcher(served)


def test_a_package_layer_the_bill_omits_is_never_built_into_a_partial_model(tmp_path):
    """``read_usd`` composed what it could: the model was the box alone, scaled to the video's estimate and shipped
    as the object. Now a stage with a composition error is refused, and the next source is used."""
    chosen, fetch = _package_with_a_lid_the_bill_omits(tmp_path)
    rows = asset.fetch_files(chosen, tmp_path / "cache", fetch=fetch)
    root = Path(next(row["path"] for row in rows if row["role"] == "root"))
    with pytest.raises(ValueError, match=asset.COMPOSITION_INCOMPLETE):
        asset.read_usd(root)
    subject = {"target_id": "box", "attribution": {**chosen, "entry_id": "simready:x"}, "files": rows}
    design, why = asset.library_design(subject, estimated_m=[0.6, 0.4, 0.6], tolerance=0.25,
                                       output_root=tmp_path / "model", fetch=fetch)
    assert design is None and why == asset.COMPOSITION_INCOMPLETE


def test_a_layer_that_was_not_fetched_and_verified_is_never_composed(tmp_path):
    """A layer the stage finds on disk beside the fetched ones (left there by anything else) is no fetched, pinned
    file: the model is refused."""
    indexes, fetch = simready._fixture(tmp_path)
    chosen = library._attribution(library.catalog(indexes)[0])
    rows = asset.fetch_files(chosen, tmp_path / "cache", fetch=fetch)
    root = Path(next(row["path"] for row in rows if row["role"] == "root"))
    points, _ = asset.read_usd(root, fetched=[Path(row["path"]) for row in rows])  # every layer fetched: read
    assert len(points)
    payload = next(row for row in rows if row.get("relative_path") == simready.PAYLOAD)
    with pytest.raises(ValueError, match=asset.LAYER_NOT_FETCHED):
        asset.read_usd(root, fetched=[Path(row["path"]) for row in rows if row is not payload])


# --- a choice is sealed for one reading of the object -----------------------------------------------------------

def test_a_choice_sealed_for_one_reading_of_the_object_is_not_reused_for_another(tmp_path):
    """The choice and the subject were sealed by the object's ``target_id`` alone: prepared again on a plan that reads
    the same target as another object (a ``crate`` of another size), the tote model was reused without asking. Now
    both are sealed for the plan's row and the task context, and another reading picks again."""
    capture = tmp_path / "capture"
    _sealed_plan(capture)
    entries = library.catalog(_index(tmp_path))
    first, asked_first = _attach(capture, REMOVAL, entries=entries, fetch=Fetcher())
    assert first[LIBRARY_SUBJECT]["attribution"]["title"] == "Sterilite Storage Tote, Blue" and len(asked_first) == 1
    # The same reading again: nothing is asked.
    same, asked_same = _attach(capture, REMOVAL, entries=entries, fetch=Fetcher())
    assert same == first and not asked_same
    _sealed_plan(capture, [{**ROW, "name": "wooden crate", "size_m": {"low": 0.55, "high": 0.65, "basis": "estimated",
                                                                       "from": "the analysis"}}])
    again, _ = _attach(capture, REMOVAL, entries=entries, fetch=Fetcher(),
                       answer={"candidate_id": "none", "reason": "No crate among them."})
    # Picked again for the crate: no entry of the catalog is one, so no tote model stands in for it.
    assert "attribution" not in again[LIBRARY_SUBJECT]
    assert again[LIBRARY_SUBJECT]["not_used"] == library.NO_CANDIDATE
    # Back to the tote reading: the model is asked again, and answers by its retained receipt in production.
    _sealed_plan(capture, [{**ROW, "name": "stackable blue storage tote"}])
    tote, asked = _attach(capture, REMOVAL, entries=entries, fetch=Fetcher())
    assert asked and tote[LIBRARY_SUBJECT]["attribution"]["title"] == "Sterilite Storage Tote, Blue"
    # Another task context (the site's own correction) is another reading too.
    _sealed_plan(capture)
    corrected = {**TASK, "description": "Move the blue tote of parts to the top shelf."}
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    corrected["context_digest"] = canonical_digest(corrected, digest_field="context_digest")
    assert library.reading_digest(ROW, corrected) != library.reading_digest(ROW, TASK)


def test_a_job_compiled_before_the_route_keeps_its_builder_whatever_it_reads_later(tmp_path):
    """Control: a one-object job compiled before this route existed keeps today's builder, sealed; a later reading
    of its object never moves it onto a library model mid-flight."""
    capture = tmp_path / "capture"
    _sealed_plan(capture)
    folder = capture / "handoff"
    folder.mkdir(parents=True)
    (folder / "preparation.json").write_text("{}")
    entries = library.catalog(_index(tmp_path))
    out, asked = _attach(capture, REMOVAL, entries=entries, fetch=Fetcher())
    assert LIBRARY_SUBJECT not in out and not asked
    _sealed_plan(capture, [{**ROW, "name": "stackable blue tote"}])
    again, asked = _attach(capture, REMOVAL, entries=entries, fetch=Fetcher())
    assert LIBRARY_SUBJECT not in again and not asked
