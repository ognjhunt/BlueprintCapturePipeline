"""Is every sentence of the company package true for its job? README.txt and success_rule.json.

From the $0 stress round of 2026-11-24 (each test reproduced a gap, now closed). Offline, no provider, nothing paid
for: the company builder's own test job (``tests/test_website_company_simulation.Job``).
"""
# Covers: src/blueprint_pipeline/website_company_simulation.py
# Covers: src/blueprint_pipeline/website_company_readme.py
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("pxr")

from blueprint_pipeline import website_company_readme as company_readme  # noqa: E402
from blueprint_pipeline import website_company_simulation as sim  # noqa: E402
from blueprint_pipeline import website_owner_success_assumption as assumption  # noqa: E402
from blueprint_pipeline.decision_evidence_contracts import canonical_digest  # noqa: E402
from tests.test_website_company_simulation import Job  # noqa: E402


def _seal_record(job: Job, **fields) -> None:
    """The half's own record, sealed, with ``fields`` added (the plan names its digest)."""
    plan = json.loads((job.packet / sim.PLAN_FILE).read_text())
    path = job.scene / "team_scene_inputs.v1.json"
    record = {**json.loads(path.read_text()), "plan_digest": plan["plan_digest"], **fields, "inputs_digest": ""}
    record["inputs_digest"] = canonical_digest(record, digest_field="inputs_digest")
    path.write_text(json.dumps(record))


def _built(job: Job, tmp_path: Path) -> tuple[str, dict]:
    sim.build_package(sim.simulation_source(job.capture, queue_root=job.queue), tmp_path / "p.zip", literals=[])
    with zipfile.ZipFile(tmp_path / "p.zip") as archive:
        readme = archive.read("README.txt").decode()
        rule = json.loads(archive.read("success_rule.json")) if "success_rule.json" in archive.namelist() else {}
    return " ".join(readme.split()), rule


def test_thresholds_the_site_never_confirmed_are_said_to_be_assumed(tmp_path):
    """A site that answered no questions: its half was compiled on Blueprint's standard thresholds, marked
    ``owner_success_assumption`` (owner direction 2026-11-07: "inferences are disclosed as assumptions"). The README
    calls them "the rule for what counts as the job done"; it and success_rule.json say they are Blueprint's
    standard thresholds, not the site's."""
    job = Job(tmp_path / "job")
    _seal_record(job, **{assumption.FIELD: {"status": assumption.ASSUMED, "thresholds_digest": "sha256:" + "a" * 64}})
    readme, rule = _built(job, tmp_path)
    assert "the rule for what counts as the job done" in readme  # what it says now
    disclosed = "assum" in readme.split("How it was made", 1)[1].lower() and "assum" in json.dumps(rule).lower()
    assert disclosed, "assumed thresholds ship as the site's rule, with nothing saying they are assumed"
    assert set(rule["criteria_basis"].values()) == {"assumed"}


def test_the_readme_explains_every_top_level_field_of_success_rule_json(tmp_path):
    """success_rule.json carries ``criteria_fields_left_out`` (how many fields Blueprint removed) on every package;
    the README names it, and what was removed and why, beside ``criteria_basis``."""
    job = Job(tmp_path / "job")
    readme, rule = _built(job, tmp_path)
    unexplained = [key for key in rule if key not in readme and key not in ("criteria", "units", "up_axis",
                                                                         # described in words: "start position", "size", "the run's timing"
                                                                         "object_start", "object_size_m", "timing")]
    assert not unexplained, f"fields of success_rule.json the README never names: {unexplained}"


def test_a_room_package_without_a_rule_never_points_at_a_success_rule_json_it_does_not_ship():
    """``readme``'s note for a part whose file carries its own root offset says "see part_notes in success_rule.json"
    only when that file ships; ``_build_once`` writes part_notes only into a rule that exists. A room package with
    no rule (nothing found, an unknown kind) ships no success_rule.json, so the note never names it. Latent: today's
    room parts are written with an identity root, so a real job needs a part with its own offset."""
    source = {"room": {"success_rule_defined": False, "success_rule_assumed": False, "not_found": True},
              "axis": "Z", "files": {"Room": {"role": "scene_appearance", "placement": {"visible": True}},
                                     "RoomCollision": {"role": "scene_collision", "placement": {"visible": False}}}}
    placed = {"Room": "scene/room.usdc", "RoomCollision": "scene/room_collision.usdc"}
    units = {prim: ("Z", 1.0) for prim in placed}
    offsets = {"Room": {"translation_m": [0.0, 0.0, 1.5], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}}
    names = ["README.txt", "scene.usda", *placed.values()]  # what ships: no success_rule.json
    readme = company_readme.readme(source, placed, names, units, offsets)
    assert "success_rule.json" not in readme, "README points at success_rule.json, which this package does not ship"
