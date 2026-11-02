"""D-20: with no intake ranking, screening starts with three strong learned DROID policies."""
from __future__ import annotations

import json

import pytest

from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog
from blueprint_pipeline.beta_policy_screening_default_priority import (
    DefaultPriorityError, default_screening_priority,
)
from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan
from tests.test_beta_policy_screening_plan import NOW, admitted, scene


def test_default_order_is_three_learned_droid_policies_in_the_catalog():
    order = default_screening_priority()
    assert order == ["pi05_droid", "cosmos3_edge_policy_droid", "groot_n17_droid"]
    by_id = {row["candidate_id"]: row for row in load_default_catalog()["candidates"]}
    assert all(by_id[candidate]["policy"]["kind"] != "scripted" for candidate in order)


def test_plan_without_an_intake_ranking_funds_the_default_three_first():
    catalog = load_default_catalog()
    by_id = {row["candidate_id"]: row for row in catalog["candidates"]}
    ids = ["cosmos3_edge_policy_droid", "groot_n17_droid", "molmoact2_droid", "pi05_droid"]
    bindings, quotes = {}, {}
    for candidate_id in ids:
        bindings[candidate_id], quotes[candidate_id] = admitted(by_id[candidate_id], maximum="4.00")
    plan = build_screening_plan(catalog, scene(), campaign_id="campaign1",
                                capabilities={"candidate_bindings": bindings}, cost_quotes=quotes, now=NOW,
                                candidate_ids=ids, priority=default_screening_priority())
    selected = [row["candidate_id"] for row in plan["candidates"] if row["status"] == "selected"]
    assert selected == ["pi05_droid", "cosmos3_edge_policy_droid", "groot_n17_droid"]


@pytest.mark.parametrize("document", [
    {"schema_version": "beta_screening_default_priority.v1", "priority": []},
    {"schema_version": "beta_screening_default_priority.v1", "priority": [{"candidate_id": "pi05_droid"}] * 2},
    {"schema_version": "other", "priority": [{"candidate_id": "pi05_droid", "basis": "x"}]},
])
def test_malformed_default_order_is_refused(tmp_path, document):
    path = tmp_path / "priority.json"
    path.write_text(json.dumps(document))
    with pytest.raises(DefaultPriorityError):
        default_screening_priority(path)
