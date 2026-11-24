"""The company package reads a half's notes from the preparation the half was built from, found by its bytes.

A website revision's source manifest is the preparation its submission staged; both compiles seal its digest in the
half's record, and the package finds that preparation under the capture wherever a later run left it (a correction's
revision folder, a superseded Atlas result's copy), never the one the handoff names now. A half whose preparation
cannot be read ships without its object (rights: a library model's geometry never ships without its credit).
"""
# Covers: src/blueprint_pipeline/website_half_preparation.py
# Covers: src/blueprint_pipeline/team_scene_packet.py
# Covers: src/blueprint_pipeline/scene_packet.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pxr")

from blueprint_pipeline import website_company_simulation as sim  # noqa: E402
from blueprint_pipeline import website_half_preparation as half  # noqa: E402
from blueprint_pipeline import website_library_notes as notes  # noqa: E402
from blueprint_pipeline.atlas_automation_plan import GENERATED_ESTIMATES_BASIS  # noqa: E402
from blueprint_pipeline.decision_evidence_contracts import canonical_digest  # noqa: E402
from tests.test_website_company_simulation import Job, _write  # noqa: E402

A = {"physics": {"bounds": {"mass_kg": [0.2, 0.4]}}, "subject_extent": {"status": "from_its_own_points"}}
B = {"physics": {"bounds": {"mass_kg": [9.0, 12.0]}}}
WEBSITE = {"uri": "s3://bucket/website-team/run/" + half.STAGED_NAME, "digest": "sha256:" + "a" * 64,
           "size_bytes": 10}


def _sealed_handoff(job: Job, **fields) -> None:
    value = {"status": "intake_ready", **fields}
    value["digest"] = canonical_digest(value, digest_field="digest")
    _write(job.capture / "pipeline/website_scene_preparation/handoff.json", value)


def _source(job: Job) -> dict:
    return sim.simulation_source(job.capture, queue_root=job.queue)


def test_only_a_website_revision_binds_its_half_to_a_preparation():
    assert half.binding({"source": {"manifest": WEBSITE}}) == {half.FIELD: WEBSITE["digest"]}
    assert half.binding({"source": {"manifest": {**WEBSITE, "uri": "s3://bucket/configured-scene/object-1.json"}}}) == {}
    assert half.binding({"source": {"manifest": {**WEBSITE, "digest": "sha256:short"}}}) == {}
    assert half.binding({}) == {}


@pytest.mark.parametrize("where", ["revisions/" + "1" * 64, "superseded_atlas_results/0123456789abcdef"])
def test_a_bound_half_reads_its_own_preparation_wherever_a_later_run_left_it(tmp_path, where):
    job = Job(tmp_path / "job")
    folder = job.capture / "pipeline/website_scene_preparation"
    own = _write(folder / where / "preparation.json", A)
    _write(folder / "preparation.json", B)  # the job prepared again since, at the default path
    _sealed_handoff(job, preparation_path=str(folder / "preparation.json"))
    job.seal_record(bound_to=own)
    source = _source(job)
    assert source["rule"]["object_estimates_from_video"] == {"mass_kg": [0.2, 0.4]}
    assert source.get("size_from_its_points") is True and not source.get("library_refusal")


def test_a_bound_half_whose_preparation_is_gone_ships_without_its_object(tmp_path):
    job = Job(tmp_path / "job")
    own = _write(job.capture / "pipeline/website_scene_preparation/preparation.json", A)
    job.seal_record(bound_to=own)
    _write(own, B)  # rewritten in place by a later run: those bytes are gone
    source = _source(job)
    assert source["library_refusal"] == notes.CREDIT_UNREADABLE
    assert "object_estimates_from_video" not in source["rule"] and not source.get("size_from_its_points")
    assert notes.refusal(source) == notes.CREDIT_UNREADABLE


def test_the_base_scene_notes_come_from_beside_the_half_s_own_preparation(tmp_path):
    job = Job(tmp_path / "job")
    folder = job.capture / "pipeline/website_scene_preparation"
    corrected = folder / "revisions" / ("2" * 64)
    own = _write(corrected / "preparation.json", A)
    _write(corrected / "base_scene.json", {"content_provenance": {"basis": GENERATED_ESTIMATES_BASIS,
                                                                  "generated_geometry_in": half.COLLISION_ONLY}})
    _write(folder / "base_scene.json", {})  # the handoff's base scene now: no generated geometry
    _sealed_handoff(job, base_scene_path=str(folder / "base_scene.json"))
    assert sim.generated_patch(job.capture) is None
    job.seal_record(bound_to=own)
    assert _source(job)["generated_patch"] == half.COLLISION_ONLY


@pytest.mark.parametrize("value", ['{"authoring_inputs": []}', '{"authoring_inputs": {"configuration": "x"}}', "[]",
                                   "{"])
def test_an_unbound_half_s_unreadable_preparation_stops_its_object(tmp_path, value):
    job = Job(tmp_path / "job")
    job.preparation.write_text(value)
    assert _source(job)["library_refusal"] == notes.CREDIT_UNREADABLE


def test_a_preparation_with_no_authoring_inputs_reads_as_no_library_model(tmp_path):
    """A task family's preparation records no stage-3 configuration (``task_family_preparation``); library sourcing
    always records its model there, so such a preparation used none."""
    job = Job(tmp_path / "job")
    _write(job.preparation, {"runtime_adapter": "task_evaluation_task_family_native_adapter"})
    source = _source(job)
    assert not source.get("library_refusal") and not source.get("library_models")


def _with_website_manifest(scene: dict) -> dict:
    revision = {**scene["revision"], "source": {**scene["revision"]["source"], "manifest": dict(WEBSITE)}}
    revision["revision_digest"] = canonical_digest(revision, digest_field="revision_digest")
    return {**scene, "revision": revision}


@pytest.mark.parametrize("website", [True, False])
def test_the_compile_seals_a_website_revision_s_preparation_digest_in_the_half_s_record(tmp_path, website):
    from tests.test_team_scene_packet import _compile, _scene
    scene = _scene(tmp_path)
    scene = _with_website_manifest(scene) if website else scene
    compiled, _request = _compile(tmp_path, scene)
    record = json.loads(Path(compiled["path"]).read_text())
    assert record.get(half.FIELD) == (WEBSITE["digest"] if website else None)
    assert record["inputs_digest"] == canonical_digest(record, digest_field="inputs_digest")


def test_the_assumption_marker_is_the_compile_s_own():
    from blueprint_pipeline import website_owner_success_assumption as assumption
    assert (half.ASSUMPTION_FIELD, half.ASSUMED) == (assumption.FIELD, assumption.ASSUMED)


def _assumed(job: Job) -> None:
    from blueprint_pipeline import website_owner_success_assumption as assumption
    job.seal_record(**{assumption.FIELD: {"status": assumption.ASSUMED, "thresholds_digest": "sha256:" + "b" * 64}})


def test_an_assumed_half_s_rule_names_each_of_its_numbers_assumed_and_says_so(tmp_path):
    """Owner direction 2026-11-07: inferences are disclosed as assumptions. Every number of the shipped criteria is
    marked ``assumed`` by its place in them, and the README says the site has not confirmed them."""
    import zipfile
    job = Job(tmp_path / "job")
    _assumed(job)
    source = _source(job)
    assert source["rule"]["criteria_basis"] == {
        "destination_containment.position_bounds_world_m.minimum": "assumed",
        "destination_containment.position_bounds_world_m.maximum": "assumed",
        "settling.position_tolerance_m": "assumed", "motion.minimum_lift_m": "assumed"}
    sim.build_package(source, tmp_path / "p.zip", literals=[])
    with zipfile.ZipFile(tmp_path / "p.zip") as archive:
        readme = " ".join(archive.read("README.txt").decode().split())
    assert half.THRESHOLDS_ASSUMED in readme and "criteria_basis says" in readme
    job.seal_record(owner_success_assumption=None)  # a half compiled on thresholds the site confirmed
    assert "criteria_basis" not in _source(job)["rule"]


def test_an_assumed_half_keeps_the_preparation_s_own_basis_for_its_numbers(tmp_path):
    job = Job(tmp_path / "job")
    _write(job.preparation, {"success_rule_basis": {"applied": True, "fields": {
        "maximum_episode_seconds": {"value": 60, "basis": "site_words"}}}})
    _assumed(job)
    source = _source(job)
    assert source["rule"]["criteria_basis"] == {"maximum_episode_seconds": "site_stated"}
    assert source["thresholds_assumed"] is True
