"""Is every sentence of the company package true for its job? README.txt and success_rule.json.

From the $0 stress round of 2026-11-24 (each test reproduced a gap, now closed). Offline, no provider, nothing paid
for: the company builder's own test job (``tests/test_website_company_simulation.Job``).
"""
# Covers: src/blueprint_pipeline/website_company_simulation.py
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("pxr")

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
