"""The build plan's object names never carry a person's or a place's name into the company's README.

Brought from the $0 stress test of 2026-11-24 (``tests/stress/test_stress_package_words.py`` on the stress branch):
the README leaves the site's own words out because only the WebApp can redact them (``ROOM_NOT_RUN``'s note in
``website_company_simulation``), but the build plan's object names and varieties are a model's short phrases from
those same words, and went into the README verbatim (``website_build_plan_notes.plan_notes``). A job that says "move
Dr. Ana Patel's sample tray to the fridge" shipped her name to every company. Offline, nothing paid for.
"""
# Covers: src/blueprint_pipeline/website_object_words.py
# Covers: src/blueprint_pipeline/website_build_plan_notes.py
from __future__ import annotations

import zipfile

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_build_plan_notes import DECISION_SCHEMA, KEPT_AT_CAP, plan_notes
from blueprint_pipeline.website_job_spend_cap import SPENDING_CAP_REACHED
from blueprint_pipeline.website_object_words import plain_phrase

ROW = {"variety": None, "rigidity": "rigid", "size_m": None, "mass_kg": None, "reason": "r", "editor_object_ids": []}
PEOPLE = ("Ana Patel", "Patel", "Okafor", "Jane", "Doe", "Boston", "Smith", "Rose")


def _decided(objects, built="tray", plan_digest=None):
    build_plan = {"plan": {"objects": objects}, **({"digest": plan_digest} if plan_digest else {})}
    decided = {"schema_version": DECISION_SCHEMA, "applied": False, "build_plan": build_plan,
               "built": {"target_id": built, "editor_object_ids": ["e"]}}
    decided["digest"] = canonical_digest(decided, digest_field="digest")
    return decided


@pytest.mark.parametrize("phrase, plain", [
    ("Dr. Ana Patel's sample tray", "sample tray"),
    ("one tray of Ms. Okafor's samples", "one tray of samples"),
    ("the Smith family's tray", "the tray"),
    ("blue tote", "blue tote"), ("one tote of parts", "one tote of parts"), ("open bin", "open bin"),
    ("96-well plate", "96-well plate"), ("about 20 vials", "about 20 vials"), ("500 ml bottles", "500 ml bottles"),
    ("Sample tray", "sample tray"), ("cups, plates and bowls", "cups, plates and bowls"),
    # A name the closed list does not hold, a capital inside the phrase, an initial, a phone or street number.
    ("ana patel's tray", None), ("Ana Patel tray", None), ("Boston freezer", None), ("Rose tray", None),
    ("a tray of Jane Doe samples", None), ("tray for m. l.", None), ("tray 555 1234", None),
    ("12 main street", None), ("Jane's", None), ("", None), (None, None)])
def test_only_plain_object_words_ship(phrase, plain):
    assert plain_phrase(phrase) == plain


def test_a_name_that_is_not_plain_words_is_called_by_its_shape_or_its_role():
    objects = [{**ROW, "target_id": "tray", "name": "Ana Patel tray", "role": "moved_object", "sourcing": "parametric",
                "parametric_shape": "tray"},
               {**ROW, "target_id": "fridge", "name": "Dr Okafor fridge", "role": "destination", "sourcing": "library",
                "parametric_shape": None, "variety": "Okafor lab"},
               {**ROW, "target_id": "not_in_video", "name": "Jane Doe", "role": "moved_object", "sourcing": "generate",
                "parametric_shape": None}]
    notes = plan_notes(_decided(objects))
    assert notes == [
        "What this simulation needs, as Blueprint's model read the job: tray, where an object goes and an object the "
        "job moves.",
        "The plan takes this out of the scan to build as its own object: tray.",
        "Not in the video, so not in this simulation: an object the job moves."]
    assert not any(name in note for note in notes for name in PEOPLE)


def test_an_object_kept_at_the_paid_work_limit_is_named_in_plain_words_too():
    objects = [{**ROW, "target_id": "tray", "name": "sample tray", "role": "moved_object", "sourcing": "parametric",
                "parametric_shape": "tray"},
               {**ROW, "target_id": "box", "name": "Ms. Okafor's cooler box", "role": "moved_object",
                "sourcing": "generate", "parametric_shape": None}]
    decided = _decided(objects, plan_digest="sha256:" + "c" * 64)
    budget = {"build_plan_digest": "sha256:" + "c" * 64,
              "objects": [{"target_id": "box", "name": "Ms. Okafor's cooler box", "status": KEPT_AT_CAP,
                           "reason": SPENDING_CAP_REACHED}]}
    notes = plan_notes(decided, budget)
    assert "Kept as filmed: cooler box, the job reached its limit on paid work." in notes
    assert not any("Okafor" in note for note in notes)


def test_the_build_plans_object_names_never_carry_a_persons_name_into_the_readme(tmp_path):
    """The stress test as it was written: the plan's name and variety for the job's one object name two people."""
    pytest.importorskip("pxr")
    from tests.test_website_company_simulation import Job, World, _package, _write
    job = Job(tmp_path / "job")
    plan = [{**ROW, "target_id": "tray", "name": "Dr. Ana Patel's sample tray", "role": "moved_object",
             "sourcing": "parametric", "variety": "one tray of Ms. Okafor's samples"}]
    _write(job.capture / "pipeline" / "atlas" / "job_object_rule.json", _decided(plan))
    published = World(job).publish()
    assert published["status"] == "published", published
    with zipfile.ZipFile(_package(job)) as archive:
        readme = " ".join(archive.read("README.txt").decode().split())
        rule = archive.read("success_rule.json").decode()
        stage = archive.read("scene.usda").decode()
    leaked = [name for name in ("Ana Patel", "Okafor") if name in readme or name in rule or name in stage]
    assert not leaked, f"people's names from the site's words ship in the package: {leaked}"
    # The company still reads what the object is.
    assert ("What this simulation needs, as Blueprint's model read the job: sample tray (one tray of samples)."
            in readme)
