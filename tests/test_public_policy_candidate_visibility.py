"""Known local routes stay visible before installation; all fixtures are development-only."""
from copy import deepcopy
import json

import pytest

from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog
from blueprint_pipeline.beta_policy_screening_coordinator import enqueue_screening_campaign, pending_publications
from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan
from blueprint_pipeline.beta_policy_screening_team import catalog_for_profile
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.scripted_robot_baseline_runtime import candidate_id
from blueprint_pipeline.scripted_robot_runtime_dispatch import runtime_for_manifest
from tests.test_beta_policy_screening_plan import scene, NOW
from tests.test_operator_door_public_policy_registration import contract, installer, PROFILE_ROOT, write
from tests.test_operator_door_scripted_registration import batch as batch  # noqa: F401


PUBLIC = ("openwam_alpha_piper_bimanual", "molmoact2_bimanual_yam", "openwam_alpha_franka")
SCRIPTED = {candidate_id(body) for body in contract.SCRIPTED_BODIES}
KNOWN = SCRIPTED | set(PUBLIC)


def rows(value):
    result = {row["candidate_id"]: row for row in value["candidates"]}
    assert len(result) == len(value["candidates"])
    return result


def publication(catalog, tmp_path):
    # No capabilities or quotes: neither catalog data nor installation can grant a run.
    plan = build_screening_plan(catalog, scene(), campaign_id="development-only-visibility", now=NOW)
    assert plan["maximum_planned_cost_usd"] == "0.00"
    enqueue_screening_campaign(plan, queue_root=tmp_path / "queue")
    result, = pending_publications(queue_root=tmp_path / "queue")
    assert all(row["status"] == "unavailable" and row["success_rate"] is None
               and row["episodes_scored"] == 0 for row in result["candidates"])
    return rows(result)


def test_exact_fixed_installer_candidates_exist_without_registration_or_private_access():
    from blueprint_pipeline import openwam_piper_runtime as piper, molmoact2_yam_runtime as yam

    catalog = rows(load_default_catalog())
    assert KNOWN <= catalog.keys()
    assert {key for key in catalog if key.startswith("blueprint_scripted_")} == SCRIPTED
    for key in KNOWN:
        row = catalog[key]
        assert row["interface"] is None and row["runtime"] is None
        assert row["eligibility"]["enabled"] is False
        assert row["eligibility"]["blockers"] == ["prepared_robot_policy_registration_missing"]
        assert row["eligibility"]["requires_credentials"] == []
        assert row["supported_task_kinds"] == ["rigid_pick_place"]
    for module, filename in ((piper, "openwam_piper_registration_profile.json"),
                             (yam, "molmoact2_yam_registration_profile.json")):
        row = catalog[module.CANDIDATE_ID]
        profile = json.loads((PROFILE_ROOT / filename).read_text())
        assert row["deployment"] == profile["deployment"]
        assert row["policy"] == {"kind": "checkpoint", "source": module.MODEL_SOURCE,
            "revision": module.MODEL_REVISION, "digest": canonical_digest(module.checkpoint_identity()),
            "digest_kind": "pinned_runtime_file_inventory"}
        assert row["eligibility"]["license_id"] == "apache-2.0"
        assert row["deployment"]["companies"] == []
    for body in contract.SCRIPTED_BODIES:
        row = catalog[candidate_id(body)]
        assert row["deployment"]["configuration_id"] == body
        assert row["deployment"]["companies"] == []
        assert row["policy"]["kind"] == "scripted"
        assert row["policy"]["digest_kind"] == "source_code_inventory"
        assert row["policy"]["digest"] == canonical_digest(runtime_for_manifest({"embodiment_id": body}).source_identity())
        assert row["eligibility"]["license_id"] == "MIT"
    assert catalog["dream1_i2rt_yam_bimanual"]["deployment"]["companies"] == ["Dream Labs"]
    assert catalog["dream1_i2rt_yam_bimanual"]["eligibility"]["enabled"] is False


def test_unregistered_candidates_keep_individual_reasons_through_plan_and_publication(tmp_path):
    published = publication(catalog_for_profile({}), tmp_path)
    for key in KNOWN:
        assert "prepared_robot_policy_registration_missing" in published[key]["blockers"]
        assert "runtime_candidate_binding_missing" in published[key]["blockers"]
    for key in SCRIPTED:
        assert published[key]["execution_evidence_kind"] == "scripted_simulator_state_baseline"
        assert published[key]["scripted_controller_invoked"] is False
        assert published[key]["candidate_policy_queried"] is False
        assert published[key]["scripted_rollout_count"] == published[key]["learned_policy_rollout_count"] == 0


@pytest.mark.parametrize("kind", ["piper", "yam", "franka", "scripted"])
def test_successful_binding_replaces_same_row_and_invalid_binding_remains_visible(tmp_path, monkeypatch, kind):
    from tests.test_openwam_piper_runtime import prepared_registration as piper
    from tests.test_molmoact2_yam_runtime import prepared_registration as yam
    from tests.test_openwam_franka_runtime import prepared_registration as franka
    from tests.test_scripted_robot_baseline_bundle import registered_world

    base = load_default_catalog()
    if kind == "scripted":
        _, profile, registration, key = registered_world(tmp_path, monkeypatch)
    else:
        _, profile, registration = {"piper": piper, "yam": yam, "franka": franka}[kind](tmp_path, monkeypatch)
        key, = profile["screening_candidates"]
    original = rows(base)
    valid = rows(catalog_for_profile(profile, base))
    assert valid.keys() == original.keys()
    assert valid[key]["eligibility"]["enabled"] is True
    assert valid[key]["eligibility"]["blockers"] == []
    assert valid[key]["configuration_digest"] != original[key]["configuration_digest"]
    assert {k: v for k, v in valid.items() if k != key} == {k: v for k, v in original.items() if k != key}
    original_rights = deepcopy(registration["rights_binding"])
    registration["rights_binding"]["license_to_run_granted"] = False
    registration["rights_binding"]["rights_digest"] = canonical_digest(
        registration["rights_binding"], digest_field="rights_digest")
    denied = rows(catalog_for_profile(profile, base))
    assert denied.keys() == original.keys()
    assert denied[key]["eligibility"]["enabled"] is False
    assert denied[key]["eligibility"]["blockers"] == ["team_candidate_rights_scope_insufficient"]
    registration["rights_binding"] = original_rights
    # A broken model is a specific candidate failure, not disappearance or a second row.
    registration["model"]["sha256"] = "sha256:" + "0" * 64
    invalid_catalog = catalog_for_profile(profile, base)
    invalid = rows(invalid_catalog)
    assert invalid.keys() == original.keys()
    assert invalid[key]["eligibility"]["enabled"] is False
    assert invalid[key]["eligibility"]["blockers"] == ["beta_screening_input_bytes_changed"]
    assert "beta_screening_input_bytes_changed" in publication(invalid_catalog, tmp_path)[key]["blockers"]


@pytest.mark.parametrize("failed_count", [3, len(contract.SCRIPTED_BODIES)])
def test_partial_or_failed_first_scripted_install_preserves_every_candidate(batch, tmp_path, failed_count):
    # Exercise the real privileged verifier/CAS against private temporary development records.
    for body in contract.SCRIPTED_BODIES[:failed_count]:
        key = candidate_id(body)
        del batch.proposal["registrations"][key]
        row = next(row for row in batch.proposal["results"] if row["candidate_id"] == key)
        row.update(status="unavailable", reason="tool_profile_unavailable")
    write(batch.destination / "proposal.json", batch.proposal)
    receipt = installer.apply(batch.request, **batch.options)
    assert receipt["registered_count"] == len(contract.SCRIPTED_BODIES) - failed_count
    assert len(receipt["results"]) == len(contract.SCRIPTED_BODIES)
    assert all(row["reason"] == "tool_profile_unavailable" for row in receipt["results"][:failed_count])
    registry = json.loads(batch.options["registry_path"].read_text())
    catalog = catalog_for_profile(registry["derived_profiles"]["defaults"])
    actual = rows(catalog)
    assert actual.keys() == rows(load_default_catalog()).keys()
    for index, body in enumerate(contract.SCRIPTED_BODIES):
        row = actual[candidate_id(body)]
        # This installer fixture deliberately contains non-USD bytes. Its installed
        # entries remain visible with that precise failure, never claimed runnable.
        assert row["eligibility"]["enabled"] is False
        assert row["eligibility"]["blockers"] == (
            ["prepared_robot_policy_registration_missing"] if index < failed_count
            else ["team_embodiment_model_not_usd"])
    # The task view truthfully states registration is missing; the exact tool failure stays
    # in the install receipt, without inventing a model binding or task outcome for it.
    published = publication(catalog, tmp_path)
    for index, body in enumerate(contract.SCRIPTED_BODIES):
        reason = "prepared_robot_policy_registration_missing" if index < failed_count else "team_embodiment_model_not_usd"
        assert reason in published[candidate_id(body)]["blockers"]


def test_rejected_public_install_keeps_missing_registration_reason(tmp_path):
    from tests.test_operator_door_public_policy_registration import retained_fixture

    retained = retained_fixture(tmp_path)
    changed = deepcopy(retained.request)
    changed["expected_registry_digest"] = "sha256:" + "0" * 64
    changed["request_digest"] = contract.digest(changed, "request_digest")
    write(retained.destination / "request.json", changed)
    with pytest.raises(ValueError, match="registry_compare_failed"):
        installer.apply(changed, **retained.options)
    registry = json.loads(retained.options["registry_path"].read_text())
    assert registry == retained.registry
    published = publication(catalog_for_profile(registry["derived_profiles"]["defaults"]), tmp_path)
    assert "prepared_robot_policy_registration_missing" in published[PUBLIC[0]]["blockers"]
