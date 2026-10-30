"""Every planning route retains a receipt; exact registrations remain independent."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import beta_policy_screening_launch as launch
from blueprint_pipeline.beta_policy_candidate_catalog import catalog_candidate, load_default_catalog, seal_catalog
from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan
from blueprint_pipeline.beta_policy_screening_g1 import CANDIDATES as G1_CANDIDATES
from blueprint_pipeline.beta_policy_screening_team import catalog_for_profile
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.deployment_embodiment_support import support_catalog
from blueprint_pipeline.team_embodiment import validate_embodiment_manifest
from tests.test_beta_policy_screening_plan import NOW, scene
from tests.test_beta_policy_screening_team import team_plan


ORIGINAL_IDS = {
    "pi05_droid", "groot_n17_droid", "cosmos3_nano_policy_droid", "molmoact2_droid",
    "flux3_action_droid", "cosmos3_edge_policy_droid", "dreamlabs_dream1", "dreamzero_droid",
    "openwam_alpha_franka", "physical_intelligence_pi_api", "dream1_i2rt_yam_bimanual",
    "dream1_agilex_piper_bimanual", "dream1_universal_robots_ur5e_bimanual",
    "dream1_franka_research_3_bimanual", "dream1_kinova_gen3_bimanual",
    "dream1_ufactory_xarm7_bimanual", "almond_axol_pi_api", "almond_axol_team_controller",
    "trlc_dk1_lerobot_controller", "extend_ur5e_team_controller", "jacobi_fanuc_m710ic_45m",
    "graymatter_fanuc_crx_10ia_l", "unchained_yaskawa_gp12_luna", "unchained_yaskawa_gp20_luna",
    "standard_bots_ro1_controller", "standard_bots_core_controller",
}


def test_packaged_routes_cover_every_recorded_pair_once_without_changing_enabled_identities():
    catalog = load_default_catalog()
    expected = {(body["embodiment_id"], route["route_id"])
                for body in support_catalog()["embodiments"] for route in body["policy_routes"]}
    public_baselines = {row["candidate_id"] for row in catalog["candidates"]
                        if (row.get("deployment") or {}).get("relationship") == "public_controller_integration"
                        and not row["deployment"]["companies"]}
    assert len(public_baselines) == 13
    pairs = Counter((row["deployment"]["configuration_id"], row["deployment"]["policy_route_id"])
                    for row in catalog["candidates"] if (row.get("deployment") or {}).get("policy_route_id")
                    and row["candidate_id"] not in public_baselines)
    assert len(catalog["candidates"]) == 101 and len(expected) == 74
    assert pairs == Counter({pair: 1 for pair in expected})
    assert ORIGINAL_IDS <= {row["candidate_id"] for row in catalog["candidates"]}
    assert {row["candidate_id"]: row["configuration_digest"] for row in catalog["candidates"]
            if row["candidate_id"] in {"pi05_droid", "groot_n17_droid", "cosmos3_edge_policy_droid"}} == {
        "pi05_droid": "sha256:667f2ad06a2e65e82b8876736f93ddbc11e3b875d0c92a0946ae84a2573cd445",
        "groot_n17_droid": "sha256:bee000dea18063838a80c1f4e362b3e6b360c5c32c1a5824c3e77b86e0e58692",
        "cosmos3_edge_policy_droid": "sha256:2ef6b265d6a79187db6079230e6f9a9ac0b7477eeeb7fb1e561bf222518c4be7",
    }
    added = [row for row in catalog["candidates"]
             if row["candidate_id"] not in ORIGINAL_IDS | set(G1_CANDIDATES) | public_baselines]
    assert len(added) == 58
    assert all(row["deployment"]["relationship"] == "planning_candidate" for row in added)
    assert all(row["interface"] is None and row["runtime"] is None for row in added)


def test_all_74_route_reasons_survive_real_planning_queue_and_publication(tmp_path):
    catalog = load_default_catalog()
    plan = build_screening_plan(catalog, scene(), campaign_id="all-routes", now=NOW)
    coordinator.enqueue_screening_campaign(plan, queue_root=tmp_path)
    [publication] = coordinator.pending_publications(queue_root=tmp_path)
    published = {row["candidate_id"]: row for row in publication["candidates"]}
    assert len(published) == len(plan["candidates"]) == 101
    for candidate in catalog["candidates"]:
        row = published[candidate["candidate_id"]]
        assert row["status"] == "unavailable"
        assert set(candidate["eligibility"]["blockers"]) <= set(row["blockers"])
        assert "runtime_candidate_binding_missing" in row["blockers"]
        if not candidate["supported_task_kinds"]:
            assert "task_kind_support_unverified" in row["blockers"]
            assert "task_kind_not_supported_by_candidate" not in row["blockers"]
    assert plan["maximum_planned_cost_usd"] == "0.00"
    assert publication["budget"]["committed_upper_bound_usd"] == "0"


def test_legacy_deployment_metadata_remains_valid_without_a_policy_route():
    catalog = load_default_catalog()
    candidate = deepcopy(catalog_candidate(catalog, "extend_ur5e_team_controller"))
    candidate["deployment"].pop("policy_route_id")
    assert "policy_route_id" not in seal_catalog({**catalog, "candidates": [candidate]})["candidates"][0]["deployment"]


def _registered_pair(tmp_path, monkeypatch, *, body="universal_robots_ur5e", route="pi05_custom_checkpoint"):
    # A tiny synthetic model tests metadata/registration binding only. This is
    # not evidence that the named production robot has run in a simulator.
    base = load_default_catalog()
    _, _, registrations = team_plan(tmp_path, monkeypatch, count=2)
    record = registrations["team_policy_00"]["robot_policy"]["manifest"]
    manifest = launch._read(Path(record["path"]))
    manifest["embodiment_id"] = body
    manifest.pop("embodiment_digest")
    manifest_path = launch._write(tmp_path / "named-fixture.manifest.json", validate_embodiment_manifest(manifest))
    for settings in registrations.values():
        registration = settings["robot_policy"]
        registration["manifest"] = launch._record(manifest_path)
        registration["policy_spec"]["embodiment_id"] = body
        registration["deployment"] = {"configuration_id": body, "policy_route_id": route,
            "companies": [], "relationship": "planning_candidate", "evidence_url": "https://operator.example/registration"}
    return base, registrations


def _rename_registration(registrations, old, new):
    settings = registrations.pop(old)
    registration = settings["robot_policy"]
    registration["checkpoint_id"] = new
    registration["policy_spec"]["policy_id"] = new
    rights = registration["rights_binding"]
    rights.update(checkpoint_id=new, candidate_id=new)
    rights["rights_digest"] = canonical_digest(rights, digest_field="rights_digest")
    registrations[new] = settings


def test_distinct_registered_checkpoints_on_one_body_and_route_are_not_collapsed(tmp_path, monkeypatch):
    base, registrations = _registered_pair(tmp_path, monkeypatch)
    catalog = catalog_for_profile({"screening_candidates": registrations}, base)
    assert len(catalog["candidates"]) == 103
    first, second = [catalog_candidate(catalog, key) for key in registrations]
    assert first["deployment"] == second["deployment"]
    assert first["policy"]["digest"] != second["policy"]["digest"]
    assert first["configuration_digest"] != second["configuration_digest"]
    assert all(row["eligibility"]["enabled"] for row in (first, second))
    # Independently registered candidates do not activate the planning template.
    assert not catalog_candidate(catalog, "universal_robots_ur5e__pi05_custom_checkpoint")["eligibility"]["enabled"]


@pytest.mark.parametrize("mutation,reason", [
    ("wrong_route", "beta_screening_policy_route_binding_mismatch"),
    ("wrong_body", "beta_screening_company_embodiment_binding_mismatch"),
    ("malformed_deployment", "beta_screening_policy_route_binding_mismatch"),
    ("missing_named_route", "beta_screening_policy_route_binding_mismatch"),
])
def test_bad_named_route_registration_retains_its_reason_and_valid_sibling(tmp_path, monkeypatch, mutation, reason):
    body = "operator-custom-arm" if mutation == "wrong_body" else "universal_robots_ur5e"
    base, registrations = _registered_pair(tmp_path, monkeypatch, body=body)
    key = "universal_robots_ur5e__pi05_custom_checkpoint"
    _rename_registration(registrations, "team_policy_00", key)
    registration = registrations[key]["robot_policy"]
    if mutation == "wrong_route":
        registration["policy_route_id"] = "groot_custom_checkpoint"
    elif mutation == "malformed_deployment":
        registration["deployment"] = "invalid"
    elif mutation == "missing_named_route":
        registration.pop("deployment")
    catalog = catalog_for_profile({"screening_candidates": registrations}, base)
    failed = catalog_candidate(catalog, key)
    assert not failed["eligibility"]["enabled"] and failed["eligibility"]["blockers"] == [reason]
    assert catalog_candidate(catalog, "team_policy_01")["eligibility"]["enabled"]


def test_generic_https_cannot_silently_activate_a_dream_route(tmp_path, monkeypatch):
    base, registrations = _registered_pair(tmp_path, monkeypatch, body="i2rt_yam_bimanual", route="customer_controller")
    key = "dream1_i2rt_yam_bimanual"
    _rename_registration(registrations, "team_policy_00", key)
    registrations[key]["robot_policy"]["deployment"]["policy_route_id"] = "dreamlabs_dream1"
    catalog = catalog_for_profile({"screening_candidates": registrations}, base)
    assert catalog_candidate(catalog, key)["eligibility"]["blockers"] == ["beta_screening_policy_route_interface_mismatch"]
    assert catalog_candidate(catalog, "team_policy_01")["eligibility"]["enabled"]


def test_custom_body_and_route_are_not_excluded_by_the_packaged_picker(tmp_path, monkeypatch):
    base, registrations = _registered_pair(tmp_path, monkeypatch, body="operator-custom-arm", route="operator-new-policy")
    registrations["team_policy_01"]["robot_policy"].pop("deployment")
    catalog = catalog_for_profile({"screening_candidates": registrations}, base)
    registered = catalog_candidate(catalog, "team_policy_00")
    assert registered["eligibility"]["enabled"]
    assert registered["deployment"]["configuration_id"] == "operator-custom-arm"
    assert registered["deployment"]["policy_route_id"] == "operator-new-policy"
    # Sharing a body does not manufacture a route association for another row.
    assert "deployment" not in catalog_candidate(catalog, "team_policy_01")
    plan = build_screening_plan(catalog, scene(), campaign_id="custom-routes", now=NOW)
    planned = next(row for row in plan["candidates"] if row["candidate_id"] == "team_policy_00")
    assert planned["status"] == "unavailable" and "runtime_candidate_binding_missing" in planned["blockers"]
    assert len(plan["candidates"]) == 103
