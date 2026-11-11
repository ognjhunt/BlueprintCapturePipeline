"""A policy we run ourselves enters screening as one registry entry and runs over the standard contract.

Plan 2.1 (``docs/launch-readiness/2026-11-11-evaluation-service-plan.md``). Hermetic: the real
planner, capabilities, launch preparation, bundle, allocator dispatch (provider replaced), coordinator,
settlement and publication; no GPU, no provider, no spend.
"""
from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import zipfile

import re

import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import beta_policy_screening_launch as launch
from blueprint_pipeline.beta_policy_effective_catalog import catalog_for_profile
from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog
from blueprint_pipeline.beta_policy_screening_capabilities import build_planning_inputs
from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.session_served_policy import ENTRY_SCHEMA, TOKEN_ENV
from blueprint_pipeline.session_served_policy_catalog import catalog_for_profile as served_catalog
from blueprint_pipeline.team_candidate_quick10_registry import profile_for_capture
from tests.test_beta_policy_screening_coordinator import Backend, native_result, publication
from tests.test_beta_policy_screening_launch import _plan

CANDIDATE = "pi05_droid_contract"
ENTRY = {"schema_version": ENTRY_SCHEMA, "server": "openpi_droid_contract_server.v1",
         "native_candidate_id": "pi05_droid", "display_name": "π0.5 DROID (standard connection)"}


def served_plan(tmp_path, monkeypatch, entry=ENTRY):
    base_plan, _facts = _plan(tmp_path, monkeypatch, scene_half="team_scene")
    profile = profile_for_capture(str(tmp_path / "capture"))
    profile["screening_candidates"] = {CANDIDATE: {"session_served_policy": dict(entry)}}
    catalog = catalog_for_profile(profile)
    planning = build_planning_inputs(catalog=catalog, profile=profile, capture_root=tmp_path / "capture",
                                     source=base_plan["source"], inputs_dir=tmp_path / "served-inputs")
    plan = build_screening_plan(catalog, planning["scene"], capabilities=planning["capabilities"],
                                cost_quotes=planning["cost_quotes"], campaign_id="served-campaign",
                                candidate_ids=[CANDIDATE])
    return plan, planning, catalog


def test_registry_entry_is_a_catalog_row_with_the_native_identity_over_the_contract():
    base = load_default_catalog()
    catalog = served_catalog({"screening_candidates": {CANDIDATE: {"session_served_policy": ENTRY}}}, base)
    row = next(item for item in catalog["candidates"] if item["candidate_id"] == CANDIDATE)
    native = next(item for item in base["candidates"] if item["candidate_id"] == "pi05_droid")
    assert row["policy"] == native["policy"] and row["eligibility"]["license_id"] == native["eligibility"]["license_id"]
    assert row["interface"] == {**native["interface"], "policy_interface_id": "openpi_droid_https.v1"}
    assert row["runtime"] == native["runtime"] and row["eligibility"]["requires_credentials"] == []
    # Every packaged row is unchanged; only the registry entry is new.
    assert [item for item in catalog["candidates"] if item["candidate_id"] != CANDIDATE] == base["candidates"]
    assert served_catalog({}, base) == base


@pytest.mark.parametrize("change", ["native_id", "unknown_server", "port", "takes_packaged_id"])
def test_an_entry_that_cannot_be_derived_is_listed_with_its_reason_and_never_enabled(change):
    base = load_default_catalog()
    entry, key = dict(ENTRY), CANDIDATE
    if change == "native_id":
        entry["native_candidate_id"] = "groot_n17_droid"
    elif change == "unknown_server":
        entry["server"] = "some_other_server.v9"
    elif change == "port":
        entry["port"] = 8000
    else:
        key = "molmoact2_droid"
    catalog = served_catalog({"screening_candidates": {key: {"session_served_policy": entry}}}, base)
    rows = {item["candidate_id"]: item for item in catalog["candidates"]}
    if change == "takes_packaged_id":
        assert rows[key] == next(item for item in base["candidates"] if item["candidate_id"] == key)
    else:
        assert rows[key]["eligibility"]["enabled"] is False and rows[key]["eligibility"]["blockers"]


def test_served_candidate_is_planned_with_the_gpu_cap_only_and_bundled_without_operator_secret(tmp_path, monkeypatch):
    from blueprint_pipeline.policy_canary_session_bundle_identity import (
        run_kind_entrypoint_blockers, run_kind_required_entries, validate_policy_canary_provider_manifest,
    )
    from blueprint_pipeline.vast_provider_adapter import _blueprint_bundle_preflight

    plan, planning, _ = served_plan(tmp_path, monkeypatch)
    candidate = next(row for row in plan["candidates"] if row["candidate_id"] == CANDIDATE)
    assert candidate["status"] == "selected", candidate.get("blockers")
    descriptor = candidate["execution_spec"]
    # No hosted fields, no credential: a native candidate's money and nothing else.
    assert not {"credential_file", "hosted_inference_maximum_usd", "provider_hard_cap_usd"} & set(descriptor)
    assert candidate["max_cost_usd"] == "4.00"
    assert planning["capabilities"]["credentials_available"] == []
    spec = launch._read(Path(descriptor["policy_execution_spec"]["path"]))
    native = launch._read(Path(planning["capabilities"]["candidate_bindings"]["pi05_droid"]["execution_spec"]
                               ["policy_execution_spec"]["path"]))
    assert spec["session_policy_server"]["native_policy_spec"] == native["policy_spec"]
    assert spec["checkpoint_digest"] == native["checkpoint_digest"]
    assert spec["policy_spec"]["endpoint_url"] == "http://127.0.0.1:8765/v1/infer"
    assert descriptor["checkpoint_inventory"] == planning["capabilities"]["candidate_bindings"]["pi05_droid"][
        "execution_spec"]["checkpoint_inventory"]
    prepared = launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=tmp_path / "attempt")
    assert "--beta-screening-credential-file" not in prepared["argv"]
    assert "--adp-max-spend-usd" in prepared["argv"]
    assert prepared["argv"][prepared["argv"].index("--adp-max-spend-usd") + 1] == "4.00"
    receipt = launch._read(Path(prepared["bundle_receipt"]["path"]))
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
        manifest = validate_policy_canary_provider_manifest(json.loads(archive.read(
            "provider_runtime/adp_arena_provider_manifest.json")))
        assert manifest["team_policy_interface_id"] == "openpi_droid_https.v1"
        assert "team_credential_env" not in manifest and manifest["secret_material_bundled"] is False
        assert manifest["session_policy_server"]["token_origin"] == "generated_on_session"
        assert run_kind_required_entries(manifest) <= names
        entrypoint = archive.read("provider_runtime/run_adp_arena_provider_runtime.sh").decode()
        assert run_kind_entrypoint_blockers(manifest, entrypoint) == []
        assert f'export {TOKEN_ENV}="$SESSION_POLICY_DIR/token"' in entrypoint
        assert f"adp009d_policy_provisioning.{CANDIDATE}.sh" in entrypoint
        assert "--terminate-ready-server" in entrypoint
        script = archive.read(f"provider_runtime/adp009d_policy_provisioning.{CANDIDATE}.sh").decode()
        # The native venv, source and checkpoint; the contract server instead of the websocket one.
        assert "/opt/adp009d-policy-venv/pi05_droid" in script and "/opt/adp009d-checkpoints/pi05_droid" in script
        assert "blueprint_pipeline.session_policy_server_start" in script
        assert "adp009d_policy_server_worker.py" not in script
    preflight = _blueprint_bundle_preflight(job_dir=tmp_path / "transport-preflight", generated_at="fixed",
        enable_blueprint_bundle=True, enable_isaac_smoke=True,
        provider_bundle_kind="native_task_arena_policy_canary_session", bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://storage.example.com/bundle.zip",
        provider_output_put_url="https://storage.example.com/output.zip")
    assert preflight["status"] == "passed", preflight.get("blockers")
    # The canonical allocator dispatches it on the ordinary lane, with no private transport.
    from blueprint_pipeline import paid_resource_allocator as allocator
    from blueprint_pipeline import policy_canary_allocator_lane as lane
    from blueprint_pipeline import team_candidate_quick10_vast as transport
    observed = []
    monkeypatch.setattr(allocator, "_control_plane_checkout_blockers",
                        lambda: ([], {"orchestrator_source_commit": descriptor["implementation_commit"]}))
    monkeypatch.setattr(lane, "_launch_environment_blockers", lambda *args: [])
    monkeypatch.delenv(lane.REQUIRE_INTEGRATION_CANARY_ENV, raising=False)
    monkeypatch.setattr(transport, "run_arena_native_control_vast",
                        lambda **kwargs: pytest.fail("a session-served policy has no private credential transport"))
    monkeypatch.setattr(lane, "run_native_task_arena_policy_canary_session_vast",
                        lambda **kwargs: observed.append(kwargs) or {"status": "dry_run_ready",
                                                                     "provider_mutations_performed": 0})
    assert allocator.main([arg for arg in prepared["argv"] if arg != "--execute"]) == 0
    assert observed and observed[0]["execute"] is False


def test_served_candidate_settles_and_publishes_through_the_real_coordinator(tmp_path, monkeypatch):
    plan, _, catalog = served_plan(tmp_path, monkeypatch)
    plan["candidates"] = [row for row in plan["candidates"] if row["candidate_id"] == CANDIDATE]
    plan["maximum_planned_cost_usd"] = plan["candidates"][0]["max_cost_usd"]
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    queue = tmp_path / "queue"
    coordinator.enqueue_screening_campaign(plan, queue_root=queue)
    result = native_result(plan, plan["candidates"][0])
    backend = Backend({"status": "completed", "provider_allocation_performed": True,
                       "provider_zero_verified": True, "result": result})
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=queue, executor=backend)
    published = publication(queue, plan["campaign_id"])
    row = published["candidates"][0]
    assert backend.launches == 1 and published["status"] == "completed"
    assert row["candidate_id"] == CANDIDATE and row["episodes_scored"] == 10 and row["episodes_succeeded"] == 7
    # Settled at the GPU cap, exactly like a native candidate: no API fee is added.
    assert Decimal(published["budget"]["spent_upper_bound_usd"]) == Decimal("4")


def test_existing_native_candidates_are_unchanged_by_a_registry_entry(tmp_path, monkeypatch):
    """With the entry, every native row plans, binds and quotes exactly as without it; only its own row is new."""
    plan, facts = _plan(tmp_path, monkeypatch, scene_half="team_scene")
    profile = profile_for_capture(str(tmp_path / "capture"))
    profile["screening_candidates"] = {CANDIDATE: {"session_served_policy": dict(ENTRY)}}
    catalog = catalog_for_profile(profile)
    served = build_planning_inputs(catalog=catalog, profile=profile, capture_root=tmp_path / "capture",
                                   source=plan["source"], inputs_dir=tmp_path / "screening-inputs")
    before, after = facts["capabilities"]["candidate_bindings"], served["capabilities"]["candidate_bindings"]
    assert set(after) == set(before) | {CANDIDATE}
    for key in before:
        assert after[key] == before[key]
        assert {k: v for k, v in served["cost_quotes"][key].items() if k not in {"valid_until", "quote_digest"}} == {
            k: v for k, v in facts["cost_quotes"][key].items() if k not in {"valid_until", "quote_digest"}}
    rows = {row["candidate_id"]: row for row in catalog["candidates"]}
    assert all(rows[row["candidate_id"]] == row for row in load_default_catalog()["candidates"])
    # Without an entry the effective catalog is the packaged one, so existing campaign ids are unchanged.
    assert catalog_for_profile(profile_for_capture(str(tmp_path / "capture")))["catalog_digest"] == plan["catalog_digest"]


def _planned_with(tmp_path, monkeypatch, candidates):
    base_plan, facts = _plan(tmp_path, monkeypatch, scene_half="team_scene")
    profile = profile_for_capture(str(tmp_path / "capture"))
    profile["screening_candidates"] = candidates
    catalog = catalog_for_profile(profile)
    planning = build_planning_inputs(catalog=catalog, profile=profile, capture_root=tmp_path / "capture",
                                     source=base_plan["source"], inputs_dir=tmp_path / "again")
    return facts, planning


@pytest.mark.parametrize("entry", ["x", [1], 5, {"schema_version": "blueprint.session_served_policy.v1"},
                                   {**ENTRY, "native_candidate_id": ["x"]}, {**ENTRY, "server": "nope"},
                                   {**ENTRY, "startup_timeout_seconds": True}])
def test_a_malformed_entry_blocks_only_its_own_candidate(tmp_path, monkeypatch, entry):
    """Review of aefc7d09a, finding 1: one bad registry value stopped planning for every candidate."""
    facts, planning = _planned_with(tmp_path, monkeypatch, {"x_contract": {"session_served_policy": entry}})
    capabilities = planning["capabilities"]
    assert "x_contract" not in capabilities["candidate_bindings"]
    assert set(facts["capabilities"]["candidate_bindings"]) <= set(capabilities["candidate_bindings"])
    blockers = capabilities["candidate_blockers"].get("x_contract")
    assert blockers is None or all(re.fullmatch(r"[a-z][a-z0-9_]{1,127}", code) for code in blockers)


@pytest.mark.parametrize("value", ["x", [1]])
def test_a_malformed_override_row_blocks_only_its_own_candidate(tmp_path, monkeypatch, value):
    facts, planning = _planned_with(tmp_path, monkeypatch, {"x_contract": value})
    assert set(facts["capabilities"]["candidate_bindings"]) <= set(planning["capabilities"]["candidate_bindings"])


def test_an_entry_never_shadows_a_packaged_row(tmp_path, monkeypatch):
    """Finding 2: an entry under ``pi05_droid`` switched the native row off."""
    facts, planning = _planned_with(tmp_path, monkeypatch, {"pi05_droid": {"session_served_policy": {
        **ENTRY, "native_candidate_id": "pi0_droid_jointpos_polaris"}}})
    before = facts["capabilities"]["candidate_bindings"]["pi05_droid"]
    after = planning["capabilities"]["candidate_bindings"]["pi05_droid"]
    assert "pi05_droid" not in planning["capabilities"]["candidate_blockers"]
    assert after["configuration_digest"] == before["configuration_digest"]
    assert (after["execution_spec"]["policy_execution_spec"]["sha256"]
            == before["execution_spec"]["policy_execution_spec"]["sha256"])
    assert after["runtime_digest"] == before["runtime_digest"]
    catalog = catalog_for_profile({"screening_candidates": {"pi05_droid": {"session_served_policy": ENTRY}}})
    native = next(row for row in catalog["candidates"] if row["candidate_id"] == "pi05_droid")
    assert native["interface"]["policy_interface_id"] == "openpi_droid.v1"


@pytest.mark.parametrize(("arguments", "sha256"), [
    ({}, "0156e2d69833721220b7555ff34e826a7cc30fe4e49501ef7c54f9724788c6da"),
    ({"local_candidates": ("pi05_droid",)}, "635bf7b4e68c2085ff0c1b85cb50bd26f427d63ba34ee4edfbfe28f72be4f7c8"),
    ({"run_kind": "blueprint_beta_screening"}, "f5eaf8425965014248c64db99649dcb49b9b7ed4710fe175b6fc6936f3894156"),
    ({"openwam_local": True}, "136fef5a8ce54026217b489978fe901d8204678182e71c56c78dbb597dc052f7"),
    ({"family_runtime_catalog_staged": True}, "ea5ef002e1d97a6591e8af7971f62eefb38d9a5e5dbdad166f0c428ec1d1ba1d"),
    ({"local_candidates": ("groot_n17_droid",), "run_kind": "blueprint_beta_screening"},
     "89775d1f51d10fe79373a50dc96a246bbd0df48f2d7b9fcd9cbe8c0f4b2c4e85"),
])
def test_a_native_candidate_s_entrypoint_is_byte_identical_to_before_the_standard_connection(arguments, sha256):
    """Finding 5: the empty session setup left one blank line. Digests are c7a4803a8's own entrypoints."""
    import hashlib

    from blueprint_pipeline.native_task_arena_policy_canary_bundle import _entrypoint

    assert hashlib.sha256(_entrypoint(**arguments).encode()).hexdigest() == sha256


def _crashed_result(value, candidate, tmp_path, *, in_flight, later):
    """A run whose policy server died serving its first try: each try's evidence as the worker writes it."""
    from blueprint_pipeline.policy_canary_worker_evidence import _write_episode_failure_gap

    result = native_result(value, candidate)
    episodes = []
    for index, row in enumerate(result["episodes"]):
        failure, progress = (in_flight, {"first_observation_retained": True, "candidate_policy_query_attempted": True,
                                         "policy_inference_failed": True}) if index == 0 else (later, {})
        path = _write_episode_failure_gap(output_root=tmp_path / "episodes-out", run_id="run", failure=failure,
            context={"cell_id": row["cell_id"], "seed": row["seed"], "candidate_id": candidate["candidate_id"]},
            progress=progress)
        evidence = json.loads(path.read_text())
        episodes.append({**row, **{key: item for key, item in evidence.items() if key != "cell_id"}})
    return {**result, "episodes": episodes}


@pytest.mark.parametrize("path", ["session_served", "native"])
def test_a_server_that_dies_on_one_try_ends_its_row_as_on_the_native_path(tmp_path, monkeypatch, path):
    """Review finding 8. On the native path a policy process that crashes during a try leaves: that try a
    scored failure (the policy's), each later try void (nothing answers the readiness check), the row
    completed with one scored try, published, a finished pair that is never run or paid for again. The
    session-served row ends the same way, and its published row also says why the nine were not counted."""
    from blueprint_pipeline import native_task_arena_policy_canary_session as session
    from blueprint_pipeline import beta_policy_screening_tested_pairs as tested
    from tests.test_beta_policy_screening_coordinator import plan

    value = plan()
    candidate = value["candidates"][0]
    if path == "session_served":
        failures = (ValueError("droid_https_policy_response_connection_lost:RemoteDisconnected"),
                    ValueError("session_policy_server_unavailable:ConnectionRefusedError"))
    else:  # the native websocket client's own errors: a dropped connection, then nothing listening
        failures = (RuntimeError("no close frame received or sent"), ConnectionRefusedError("Connect call failed"))
    result = _crashed_result(value, candidate, tmp_path, in_flight=failures[0], later=failures[1])
    # The session-level validator is exercised elsewhere; here every row is the worker's own evidence.
    monkeypatch.setattr(session, "validate_session_result", lambda payload, **kwargs: payload)
    queue = tmp_path / "queue"
    coordinator.enqueue_screening_campaign(value, queue_root=queue)
    backend = Backend({"status": "completed", "provider_allocation_performed": True,
                       "provider_zero_verified": True, "result": result})
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=queue, executor=backend)
    row = publication(queue, value["campaign_id"])["candidates"][0]
    assert (row["status"], row["episodes_scored"], row["episodes_failed"], row["episodes_void"]) == ("completed", 1, 1, 9)
    ledger = json.loads((queue / coordinator.LEDGER_FILENAME).read_text())
    state = ledger["campaigns"][value["campaign_id"]]["candidates"][candidate["candidate_id"]]
    reasons = {outcome["reason"] for outcome in state["summary"]["episode_outcomes"][1:]}
    if path == "session_served":
        assert row["setup_quality_flags"] == ["session_policy_server_lost"]
        assert reasons == {"policy_server_unavailable"}
    else:
        assert "setup_quality_flags" not in row and reasons == {"simulator_failed_before_policy"}
    # A finished result: never re-run, so a crashed server never leads to a paid re-run.
    assert tested.finished(state)
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=queue, executor=backend)
    assert backend.launches == 1
