"""The job's object from a library model, a parametric shape or the paid builder, in the company package.

From the $0 stress round of 2026-11-24 (each test reproduced a gap, now closed): library geometry never ships without
its credit; every note comes from the preparation the shipped half was built from (``website_half_preparation``); an
unreadable credit stops only the object; the README says what the object is. Offline, no provider, no network,
nothing paid for: the repository's own fixtures (``tests/test_website_object_library``,
``tests/test_website_company_simulation``). A half's record is sealed as a compile seals it, bound to its preparation.
"""
# Covers: src/blueprint_pipeline/website_half_preparation.py
# Covers: src/blueprint_pipeline/website_library_notes.py
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("pxr")

from blueprint_pipeline import website_company_simulation as sim  # noqa: E402
from blueprint_pipeline import website_library_notes as notes  # noqa: E402
from blueprint_pipeline.website_supplied_mesh import supplied_mesh_configuration  # noqa: E402
from tests.test_website_company_simulation import NOW, Job, World, _manifest  # noqa: E402
from tests.test_website_object_library import (  # noqa: E402
    _configuration, _job_with_a_library_object, _library_topologies, _subject,
)

PREPARATION = ("pipeline", "website_scene_preparation", "preparation.json")


def _readme_and_names(zip_path: Path) -> tuple[str, list[str], Path]:
    with zipfile.ZipFile(zip_path) as archive:
        readme = archive.read("README.txt").decode()
        names = archive.namelist()
        out = zip_path.with_suffix(".unpacked")
        archive.extractall(out)
    return readme, names, out


def _holding_model(unpacked: Path, reference: set) -> list[str]:
    return sorted(p.relative_to(unpacked).as_posix() for p in unpacked.rglob("*")
                  if p.suffix in (".usd", ".usda", ".usdc", ".usdz") and p.name != "scene.usda"
                  and reference & set(_library_topologies(p)))


# --- rights: library geometry and its credit ---------------------------------------------------------------

@pytest.mark.parametrize("damage", ["corrupt_json", "missing", "not_a_dict"])
def test_library_geometry_never_ships_without_its_credit_when_the_preparation_record_is_unreadable(tmp_path, damage):
    """Rights is a hard stop: an object built from a CC BY model must not ship without its credit. The credit is read
    only from the half's own preparation record; when that record cannot be read the package refuses the object (as
    an unreadable credit does), never ships the model's geometry bare."""
    job, _made, reference = _job_with_a_library_object(tmp_path)
    path = job.capture.joinpath(*PREPARATION)
    if damage == "corrupt_json":
        path.write_text('{"authoring_inputs": {"configuration": ')
    elif damage == "missing":
        path.unlink()
    else:
        path.write_text("[]")
    source = sim.simulation_source(job.capture, queue_root=job.queue)
    try:
        sim.build_package(source, tmp_path / "package.zip", literals=[])
    except sim.PackageRefused:
        return  # the right outcome: nothing leaves
    readme, _names, unpacked = _readme_and_names(tmp_path / "package.zip")
    holding = _holding_model(unpacked, reference)
    assert not holding or notes.HEADING in readme, (
        f"library geometry shipped in {holding} with no 'Library models' credit ({damage} preparation record)")


def test_the_credit_follows_the_half_that_ships_not_the_newest_preparation(tmp_path):
    """The published half was built from the library model (revision A; its record names A's bytes). A retry
    re-prepared the job and the handoff now names revision B, built from a parametric shape, whose half is not
    compiled yet. The package still ships half A, so it still credits the model."""
    job, _made, reference = _job_with_a_library_object(tmp_path)
    folder = job.capture / "pipeline" / "website_scene_preparation"
    newer = supplied_mesh_configuration(_configuration(), synthetic=None, synthetic_authoring=None,
                                        parametric={"target_id": "tote", "shape": "tote"}, subject_id="tote",
                                        sim_min=[0.7, 1.8, 0.0], sim_max=[1.3, 2.2, 0.3], library=None,
                                        library_root=None)
    (folder / "revision-b").mkdir()
    (folder / "revision-b" / "preparation.json").write_text(json.dumps({"authoring_inputs": {"configuration": newer}}))
    (folder / "handoff.json").write_text(json.dumps({"preparation_path": str(folder / "revision-b" / "preparation.json")}))
    source = sim.simulation_source(job.capture, queue_root=job.queue)
    sim.build_package(source, tmp_path / "package.zip", literals=[])
    readme, _names, unpacked = _readme_and_names(tmp_path / "package.zip")
    holding = _holding_model(unpacked, reference)
    assert holding, "fixture: the shipped object holds the model's geometry"
    assert notes.HEADING in readme, f"half A's library geometry ships in {holding} with no credit"


def test_an_unreadable_credit_still_ships_the_room_without_the_object(tmp_path, monkeypatch):
    """The credit is sealed in the half's own preparation, so it never becomes readable. Rights stops the object; the
    doctrine says everything else keeps going, and ``website_library_notes.refusal`` says a package without the
    object is not refused. Two passes a window apart end with the job's room floor published (the object left out,
    flagged with why), not with the job refused for good."""
    from tests.test_website_package_floor import Floor
    job, made, _ = _job_with_a_library_object(tmp_path)
    made["library_design"]["attribution"].pop("author")
    job.capture.joinpath(*PREPARATION).write_text(json.dumps({"authoring_inputs": {"configuration": made}}))
    job.seal_record(bound_to=job.capture.joinpath(*PREPARATION))
    Floor(tmp_path / "floor", monkeypatch, job=job, half=True)  # the room floor every routed job's handoff writes
    world = World(job)
    first = world.publish()
    from blueprint_pipeline import company_share_permission as permission
    later = sim.publish_company_simulation(
        job.capture, now=lambda: NOW + permission.LIVE_STATEMENT_RECHECK_SECONDS + 1, check_rights=world.check_rights,
        archive=world.archive, retract=world.retract, deliver=world.deliver, queue_root=job.queue)
    outcomes = [(first.get("status"), first.get("reason")), (later.get("status"), later.get("reason"))]
    assert later.get("status") == "published", f"the job never ships anything: {outcomes}"
    manifest = _manifest(job)
    assert manifest["source_kind"] == "floor" and manifest["floor_reason"] == "object_not_built:" + notes.CREDIT_UNREADABLE


# --- wording: what the README says the object is --------------------------------------------------------------

def _package_with(tmp_path: Path, configuration: dict) -> str:
    job = Job(tmp_path / "job")
    job.capture.joinpath(*PREPARATION).parent.mkdir(parents=True, exist_ok=True)
    job.capture.joinpath(*PREPARATION).write_text(json.dumps({"authoring_inputs": {"configuration": configuration}}))
    sim.build_package(sim.simulation_source(job.capture, queue_root=job.queue), tmp_path / "package.zip", literals=[])
    readme, _names, _out = _readme_and_names(tmp_path / "package.zip")
    return " ".join(readme.split())


def test_a_library_model_is_named_as_such(tmp_path):
    """Control: the library route does say what the object is (its credit)."""
    made = supplied_mesh_configuration(_configuration(), synthetic=None, synthetic_authoring=None, parametric=None,
                                       subject_id="tote", sim_min=[0, 0, 0], sim_max=[0.9, 0.6, 0.45],
                                       library=_subject(tmp_path / "lib"), library_root=tmp_path / "model")
    readme = _package_with(tmp_path, made)
    assert notes.HEADING in readme and "Sterilite Storage Tote" in readme


def test_the_shipped_halfs_mass_and_friction_come_from_its_own_preparation(tmp_path):
    """Same root cause as the credit: every disclosure is read from the preparation the shipped half was built from,
    never the one the handoff names now. Half A ships; the handoff names re-preparation B (another object estimate).
    Half A's success_rule.json carries A's mass and friction ranges, never B's."""
    job = Job(tmp_path / "job")
    folder = job.capture / "pipeline" / "website_scene_preparation"
    folder.mkdir(parents=True, exist_ok=True)
    a = {"physics": {"bounds": {"mass_kg": [0.2, 0.4], "static_friction": [0.5, 0.8]}}}
    b = {"physics": {"bounds": {"mass_kg": [9.0, 12.0], "static_friction": [0.1, 0.2]}}}
    (folder / "preparation.json").write_text(json.dumps(a))  # the preparation half A was compiled from
    job.seal_record(bound_to=folder / "preparation.json")
    (folder / "revision-b").mkdir()
    (folder / "revision-b" / "preparation.json").write_text(json.dumps(b))
    (folder / "handoff.json").write_text(json.dumps({"preparation_path": str(folder / "revision-b" / "preparation.json")}))
    sim.build_package(sim.simulation_source(job.capture, queue_root=job.queue), tmp_path / "p.zip", literals=[])
    with zipfile.ZipFile(tmp_path / "p.zip") as archive:
        rule = json.loads(archive.read("success_rule.json"))
    shipped = json.dumps(rule.get("object_estimates_from_video"))
    assert shipped != "null", "fixture: the rule carries the estimates"
    assert "9.0" not in shipped, f"half A ships preparation B's mass range: {shipped}"
