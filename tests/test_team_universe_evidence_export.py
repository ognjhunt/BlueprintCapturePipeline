"""Synthetic producer-to-daily-agent projection with every rank/audit/retained-source binding rechecked."""
import json

import pytest

from tests.team_universe_fixture import TODAY, pipeline, screen_answers, weights
from tests.test_team_universe_qualification import PRODUCT, audit_raw, decision
from tools.daily_research import site_screen as ss
from tools.daily_research import team_universe as te
from tools.team_universe import evidence_export as ee
from tools.team_universe import rank
from tools.team_universe import universe as tu


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "VOLATILE_ROOTS", ())
    monkeypatch.delenv("BLUEPRINT_DAILY_RESEARCH_WORKER_ENABLED", raising=False)
    monkeypatch.setattr(ss, "code_state", lambda root=None: {"commit": "0" * 40, "dirty": False, "source": "git"})
    workspace, _, _, records = pipeline(tmp_path, {1: screen_answers(1, robot_forms="fixed_arm", robot_forms_quote=PRODUCT)})
    states, _ = workspace.states()
    teams, _ = tu.team_list(tu.stage_records(workspace, states, "discover").values())
    record = records[1]
    pages = tu._evidence(workspace.path("screen", "evidence", record["site_key"]).read_bytes())
    audited = audit_raw(workspace, teams, {record["site_key"]: record}, [decision(teams, record, pages)])
    rank.rank(workspace, weights(), audit_raw=audited, today=TODAY)
    return workspace, audited


def test_export_projects_current_reviewed_physical_evidence_and_omits_contacts(source):
    workspace, audited = source
    raw = ee.build(workspace, audited, today=TODAY)
    value = te.load(raw, today=TODAY)
    assert len(value["teams"]) == 1 and value["teams"][0]["status"] == "capability_prospect"
    assert value["manifest"]["audit_sha256"] == te.sha(audited)
    assert value["manifest"]["ranked_sha256"] == te.sha((workspace.root / tu.RANKED_NAME).read_bytes())
    assert b"partnerships@" not in raw and b"contact_email" not in raw and b"source_claimed_" not in raw
    assert value["teams"][0]["qualification"]["evaluation_compatibility"] == "not_verified"


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(schema_version="blueprint.team-rank.ranked.v1"),
    lambda d: d.update(assessed_on="2026-10-04"),
    lambda d: d.update(scope_sha256="0" * 64),
    lambda d: d["audit"].update(sha256="0" * 64),
    lambda d: d["teams"][0].update(status="beta_candidate"),
    lambda d: d["teams"][0]["qualification"].update(offering="robot_integrator_task"),
])
def test_stale_or_mutated_rank_never_exports(source, mutation):
    workspace, audited = source
    path = workspace.root / tu.RANKED_NAME
    value = json.loads(path.read_bytes())
    mutation(value)
    path.write_text(json.dumps(value))
    with pytest.raises(tu.TeamError):
        ee.build(workspace, audited, today=TODAY)


def test_changed_retained_page_or_raw_audit_binding_cannot_be_exported(source):
    workspace, audited = source
    value = json.loads(audited)
    value["teams"][0]["capability_evidence"][0]["page_sha256"] = "0" * 64
    with pytest.raises(tu.TeamError):
        ee.build(workspace, json.dumps(value).encode(), today=TODAY)
