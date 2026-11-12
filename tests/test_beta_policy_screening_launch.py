"""Real beta bundles and crash-safe canonical allocator starts, without spend."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import zipfile

import pytest

from blueprint_pipeline import beta_policy_screening_launch as launch
from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog
from blueprint_pipeline.beta_policy_screening_capabilities import build_planning_inputs
from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_session import (
    BETA_RUN_KIND, PolicyCanarySessionError, validate_runtime_input_manifest, validate_session_authority,
)
from blueprint_pipeline.policy_canary_session_bundle_identity import validate_policy_canary_provider_manifest
from blueprint_pipeline.team_candidate_quick10_registry import profile_for_capture
from tests.test_team_candidate_quick10_launch import _world


def _plan(tmp_path, monkeypatch, *, catalog_override=None, scene_half="frozen_pair"):
    world = _world(tmp_path, monkeypatch, derived=scene_half == "team_scene", scene_half=scene_half)
    profile = profile_for_capture(str(world["capture"]))
    source = {"capture_id": "capture1", "task_id": profile["task_id"], "scene_intent_id": "intent1",
              "scene_intent_digest": "sha256:" + "a" * 64, "task_context_digest": "sha256:" + "b" * 64,
              "scene_revision_digest": world["probe_receipt"]["configured_scene_revision_digest"]}
    catalog = catalog_override or load_default_catalog()
    facts = build_planning_inputs(catalog=catalog, profile=profile, capture_root=world["capture"],
                                 source=source, inputs_dir=tmp_path / "screening-inputs")
    plan = build_screening_plan(catalog, facts["scene"], capabilities=facts["capabilities"],
                                cost_quotes=facts["cost_quotes"], campaign_id="campaign1")
    return plan, facts


@pytest.mark.parametrize("scene_half", ["frozen_pair", "team_scene"])
def test_real_single_candidate_robolab_bundle_passes_both_pre_create_gates(tmp_path, monkeypatch, scene_half):
    from blueprint_pipeline.vast_provider_adapter import _blueprint_bundle_preflight

    plan, facts = _plan(tmp_path, monkeypatch, scene_half=scene_half)
    selected = [row for row in plan["candidates"] if row["status"] == "selected"]
    assert {row["candidate_id"] for row in selected} == {"pi05_droid", "groot_n17_droid", "molmoact2_droid"}
    assert plan["maximum_planned_cost_usd"] == "12.00"
    candidate = next(row for row in selected if row["candidate_id"] == "pi05_droid")
    prepared = launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=tmp_path / "attempt")
    authority = validate_session_authority(launch._read(Path(prepared["authority"]["path"])))
    assert authority["run_kind"] == BETA_RUN_KIND
    assert authority["candidate_ids"] == ["pi05_droid"] and authority["learned_policy_rollout_count"] == 10
    assert authority["beta_screening_admission"]["plan_digest"] == plan["plan_digest"]
    receipt = launch._read(Path(prepared["bundle_receipt"]["path"]))
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        manifest = validate_policy_canary_provider_manifest(json.loads(archive.read(
            "provider_runtime/adp_arena_provider_manifest.json")))
        assert manifest["evaluation_backend"] == "robolab.v1"
        assert "provider_runtime/robolab/eval/episode.py" in archive.namelist()
        assert "provider_runtime/adp009d_policy_provisioning.groot_n17_droid.sh" not in archive.namelist()
    preflight = _blueprint_bundle_preflight(job_dir=tmp_path / "transport-preflight", generated_at="fixed",
        enable_blueprint_bundle=True, enable_isaac_smoke=True,
        provider_bundle_kind="native_task_arena_policy_canary_session", bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://storage.example.com/bundle.zip",
        provider_output_put_url="https://storage.example.com/output.zip")
    assert preflight["status"] == "passed", preflight.get("blockers")
    assert launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=tmp_path / "attempt") == prepared
    assert not (tmp_path / "attempt" / "allocator_launch_intent.json").exists()
    # A deploy switched while the row waited for capacity: the staged bundle holds the old
    # release's code, so reusing it is refused (free) and the row is planned at the new commit.
    monkeypatch.setattr(launch, "current_commit", lambda: "e" * 40)
    with pytest.raises(ValueError, match="beta_screening_implementation_commit_changed"):
        launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=tmp_path / "attempt")
    assert launch._read(tmp_path / "attempt" / "prepared.json") == prepared
    assert not (tmp_path / "attempt" / "allocator_launch_intent.json").exists()


def test_openpi_polaris_variant_is_selected_and_bundled_like_pi05(tmp_path, monkeypatch):
    from blueprint_pipeline.adp009d_policy_provisioning import CHECKPOINT_INVENTORY_STAGED_NAME
    from blueprint_pipeline.native_task_arena_policy_bundle import OPENPI_CHECKPOINT_INVENTORY_PATH
    from blueprint_pipeline.vast_provider_adapter import _blueprint_bundle_preflight

    variant = "pi0_fast_droid_jointpos_polaris"
    world = _world(tmp_path, monkeypatch, derived=False, scene_half="frozen_pair")
    profile = profile_for_capture(str(world["capture"]))
    source = {"capture_id": "capture1", "task_id": profile["task_id"], "scene_intent_id": "intent1",
              "scene_intent_digest": "sha256:" + "a" * 64, "task_context_digest": "sha256:" + "b" * 64,
              "scene_revision_digest": world["probe_receipt"]["configured_scene_revision_digest"]}
    catalog = load_default_catalog()
    facts = build_planning_inputs(catalog=catalog, profile=profile, capture_root=world["capture"],
                                 source=source, inputs_dir=tmp_path / "screening-inputs")
    plan = build_screening_plan(catalog, facts["scene"], capabilities=facts["capabilities"],
                                cost_quotes=facts["cost_quotes"], campaign_id="campaign1", candidate_ids=[variant])
    [candidate] = plan["candidates"]
    assert candidate["status"] == "selected" and candidate["blockers"] == []
    spec = launch._read(Path(candidate["execution_spec"]["policy_execution_spec"]["path"]))
    assert spec["policy_interface_id"] == "openpi_droid.v1" and spec["policy_spec"]["policy_id"] == variant
    assert spec["policy_spec"]["action_chunk_rows"] == 10 and spec["open_loop_horizon"] == 8
    prepared = launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=tmp_path / "attempt")
    authority = validate_session_authority(launch._read(Path(prepared["authority"]["path"])))
    assert authority["run_kind"] == BETA_RUN_KIND and authority["candidate_ids"] == [variant]
    receipt = launch._read(Path(prepared["bundle_receipt"]["path"]))
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        manifest = validate_policy_canary_provider_manifest(json.loads(archive.read(
            "provider_runtime/adp_arena_provider_manifest.json")))
        assert manifest["candidate_ids"] == [variant] and manifest["evaluation_backend"] == "robolab.v1"
        provisioning = archive.read(f"provider_runtime/adp009d_policy_provisioning.{variant}.sh").decode()
        assert f'"gs://openpi-assets/checkpoints/polaris/{variant}"' in provisioning
        assert "pi05_droid" not in provisioning
        assert (archive.read("provider_runtime/" + CHECKPOINT_INVENTORY_STAGED_NAME)
                == OPENPI_CHECKPOINT_INVENTORY_PATH.read_bytes())
    preflight = _blueprint_bundle_preflight(job_dir=tmp_path / "transport-preflight", generated_at="fixed",
        enable_blueprint_bundle=True, enable_isaac_smoke=True,
        provider_bundle_kind="native_task_arena_policy_canary_session", bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://storage.example.com/bundle.zip",
        provider_output_put_url="https://storage.example.com/output.zip")
    assert preflight["status"] == "passed", preflight.get("blockers")
    assert not (tmp_path / "attempt" / "allocator_launch_intent.json").exists()


def test_launch_intent_crash_is_not_retried_or_released(tmp_path, monkeypatch):
    root = tmp_path / "attempt"
    root.mkdir()
    monkeypatch.setattr(launch, "_validate_prepared", lambda *args: None)
    prepared = {"prepared_digest": "sha256:" + "a" * 64, "argv": ["gpu-canary"]}
    # A legacy intent has no durable job fence and remains honestly ambiguous.
    launch._write(root / "allocator_launch_intent.json", {"prepared_digest": prepared["prepared_digest"]})
    kwargs = dict(plan={"source": {}, "site_id": "site-request1"}, candidate={}, attempt_dir=root, prepared=prepared)
    result = launch.launch_candidate(**kwargs)
    assert result["status"] == "reconciliation_required"
    assert result["no_allocation_proven"] is False
    assert "provider_allocation_performed" not in result


def test_new_pre_spawn_crash_recovers_only_after_durable_no_create_fence(tmp_path, monkeypatch):
    from blueprint_pipeline import allocator_invocation as jobs
    from blueprint_pipeline import beta_policy_screening_worker as service
    from blueprint_pipeline import beta_policy_screening_control_transport as control_transport
    from tests.test_beta_policy_screening_control_contracts import binding, response
    import time

    monkeypatch.setattr(launch, "_validate_prepared", lambda *args: None)
    monkeypatch.setattr(launch, "_check_quote", lambda *args: None)
    monkeypatch.setattr(service, "require_current_source", lambda *args, **kwargs: {})
    monkeypatch.setattr(control_transport, "request_control", lambda *, binding, **kwargs:
                        response(scope=binding, now=int(time.time()), deadline=int(time.time()) + 3600))
    authority = launch._write(tmp_path / "authority.json", {"authority_digest": "sha256:" + "c" * 64,
                                                          "resource_name": "canonical-owned-resource"})
    bundle = launch._write(tmp_path / "bundle.json", {"bundle_sha256": "sha256:" + "d" * 64})
    prepared = {"prepared_digest": "sha256:" + "a" * 64, "argv": ["gpu-canary"],
                "authority": launch._record(authority), "bundle_receipt": launch._record(bundle)}
    calls = []

    def crash(path, **kwargs):
        calls.append(path)
        raise RuntimeError("crash before or after spawn acknowledgment")

    monkeypatch.setattr(jobs, "start_invocation", crash)
    kwargs = dict(plan={"campaign_id": "campaign-one", "source": binding()["source"], "site_id": "site-request1",
                        "plan_digest": "sha256:" + "e" * 64},
                  candidate={"candidate_id": "candidate", "configuration_digest": "sha256:" + "f" * 64},
                  attempt_dir=tmp_path, prepared=prepared)
    with pytest.raises(RuntimeError):
        launch.launch_candidate(**kwargs)
    result = launch.launch_candidate(**kwargs)
    assert result["retryable_unstarted_refusal"] is True
    assert result["no_allocation_proven"] is True
    assert result["provider_allocation_performed"] is False
    assert len(calls) == 1
    assert jobs.run_invocation(calls[0], allocator=lambda args: pytest.fail("late duplicate create")) == 2


def test_closed_allocator_refusal_is_the_only_free_release(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "_validate_prepared", lambda *args: None)
    launch._write(tmp_path / "allocator_result.json", {"status": "blocked", "provider_mutations_performed": 0,
                                                       "provider_create_attempted": False})
    result = launch.reconcile_candidate(plan={}, candidate={}, attempt_dir=tmp_path, prepared={})
    assert result["no_allocation_proven"] and result["provider_allocation_performed"] is False


def _refused_create(root: Path, *, watchdog: dict, http_status: int = 410) -> None:
    """The allocator result a stale-offer create leaves: Vast refused it and nothing exists."""
    run = root / "allocator" / "vast_provider_run"
    run.mkdir(parents=True)
    launch._write(run / "vast_provider_adapter_result.json", {
        "status": "failed", "provider_create_attempted": True, "vast_instance_ids": [],
        "vast_side_effects_may_have_occurred": False, "continuing_spend_from_this_run": False,
        "create_failure_diagnosis": {"http_status_code": http_status, "definite_create_refusal": True,
            "create_inventory_verified": True, "create_inventory_http_status_code": 200,
            "create_produced_no_instance": True, "matching_attempt_instance_ids": [],
            "attempted_labels": ["blueprint-native-task-policy-canary-beta-1"]},
        "provider_attempt_classification": {"classification": "pre_execution_provider_null",
                                            "scientific_attempt_consumed": False}})
    launch._write(run / "vast_teardown_manifest.json",
                  {"status": "completed", "vast_instance_ids": [], "continuing_spend_from_this_run": False})
    launch._write(root / "allocator_result.json", {
        "status": "blocked", "blockers": ["vast_instance_create_failed"],
        "scientific_attempt_started": False, "candidate_policy_queried": False,
        "continuing_spend_from_this_run": False, "all_staged_objects_absent": True,
        "adapter_result_path": str(run / "vast_provider_adapter_result.json"),
        "teardown_manifest_path": str(run / "vast_teardown_manifest.json"),
        "provider_closeout": {"provider_zero_confirmed": True,
                              "adapter_result": launch._record(run / "vast_provider_adapter_result.json"),
                              "teardown_manifest": launch._record(run / "vast_teardown_manifest.json")},
        "independent_watchdog_close": watchdog})


def test_definite_create_refusal_without_instance_is_released_not_left_unresolved(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "_validate_prepared", lambda *args: None)
    _refused_create(tmp_path, watchdog={"status": "cancelled_no_allocation", "provider_mutations_performed": 0})
    result = launch.reconcile_candidate(plan={}, candidate={}, attempt_dir=tmp_path, prepared={})
    assert result == {"status": "blocked", "provider_allocation_performed": False, "provider_zero_verified": True,
                      "no_allocation_proven": True, "blockers": ["vast_instance_create_failed"]}


@pytest.mark.parametrize("fault", ["watchdog_retained", "ambiguous_400", "receipt_changed"])
def test_create_refusal_without_complete_absence_proof_stays_reserved(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(launch, "_validate_prepared", lambda *args: None)
    watchdog = ({"status": "retained_until_hard_ttl", "provider_mutations_performed": 0,
                 "reason": "provider_allocation_identity_ambiguous", "watchdog_retention_liveness_confirmed": True,
                 "watchdog_armed_before_allocation": True, "instance_ids": []}
                if fault == "watchdog_retained" else {"status": "cancelled_no_allocation", "provider_mutations_performed": 0})
    _refused_create(tmp_path, watchdog=watchdog, http_status=400 if fault == "ambiguous_400" else 410)
    if fault == "receipt_changed":
        teardown = tmp_path / "allocator" / "vast_provider_run" / "vast_teardown_manifest.json"
        teardown.write_text(teardown.read_text() + " ")
    result = launch.reconcile_candidate(plan={}, candidate={}, attempt_dir=tmp_path, prepared={})
    assert result["status"] == "reconciliation_required" and result["no_allocation_proven"] is False


def test_current_source_refusal_never_spawns_and_stays_proven_unallocated(tmp_path, monkeypatch):
    from blueprint_pipeline import team_candidate_quick10_launch as native
    from blueprint_pipeline import beta_policy_screening_worker as service

    monkeypatch.setattr(launch, "_validate_prepared", lambda *args: None)
    monkeypatch.setattr(launch, "_check_quote", lambda *args: None)
    def refuse(*args, **kwargs):
        raise ValueError("current consent withdrawn")
    monkeypatch.setattr(service, "require_current_source", refuse)
    monkeypatch.setattr(native, "_start_allocator", lambda *args, **kwargs: pytest.fail("must not spawn"))
    kwargs = dict(plan={"source": {}, "site_id": "site-request1"}, candidate={}, attempt_dir=tmp_path,
                  prepared={"prepared_digest": "sha256:" + "a" * 64})
    first = launch.launch_candidate(**kwargs)
    assert first["status"] == "blocked" and first["no_allocation_proven"]
    assert first["provider_allocation_performed"] is False
    assert launch.launch_candidate(**kwargs) == first
    assert launch.reconcile_candidate(**kwargs) == first


def test_the_money_check_before_spawn_counts_the_queue_that_owns_the_launch(tmp_path, monkeypatch):
    # A custom queue's $2 reservation must not be checked against the (empty) default queue.
    from blueprint_pipeline import team_candidate_quick10_launch as native
    from blueprint_pipeline import beta_policy_screening_job_ceiling as job_ceiling
    from blueprint_pipeline import beta_policy_screening_worker as service

    monkeypatch.setattr(launch, "_validate_prepared", lambda *args: None)
    monkeypatch.setattr(launch, "_check_quote", lambda *args: None)
    monkeypatch.setattr(service, "require_current_source", lambda *args, **kwargs: {})
    seen = []

    def check(plan, *, context, queue_root=None, **kwargs):
        seen.append(queue_root)
        raise ValueError(job_ceiling.EXCEEDED)
    monkeypatch.setattr(job_ceiling, "require_current_job_ceiling", check)
    monkeypatch.setattr(native, "_start_allocator", lambda *args, **kwargs: pytest.fail("must not spawn"))
    queue = tmp_path / "custom-queue"
    result = launch.launch_candidate(plan={"source": {}, "site_id": "site-request1"}, candidate={},
                                     attempt_dir=tmp_path / "attempt", prepared={"prepared_digest": "sha256:" + "a" * 64},
                                     queue_root=queue)
    assert seen == [queue]
    assert result["status"] == "blocked" and result["provider_allocation_performed"] is False


@pytest.mark.parametrize("field,value", [("control_hz", 30), ("image_height", 225),
    ("robot_profile_id", "another_robot"), ("camera_roles", ["external"]),
    ("normalization_digest", "sha256:" + "d" * 64)])
def test_changed_catalog_interface_does_not_admit_the_same_runtime(tmp_path, monkeypatch, field, value):
    from blueprint_pipeline.beta_policy_candidate_catalog import CATALOG_ENV, seal_catalog
    catalog = load_default_catalog()
    next(row for row in catalog["candidates"] if row["candidate_id"] == "pi05_droid")["interface"][field] = value
    catalog = seal_catalog(catalog)
    override = tmp_path / "operator-catalog.json"
    override.write_text(json.dumps(catalog))
    monkeypatch.setenv(CATALOG_ENV, str(override))
    plan, facts = _plan(tmp_path, monkeypatch, catalog_override=catalog)
    assert "pi05_droid" not in facts["capabilities"]["candidate_bindings"]
    candidate = next(row for row in plan["candidates"] if row["candidate_id"] == "pi05_droid")
    assert candidate["status"] == "unavailable"


def test_derived_profiles_accept_operator_screening_config_but_never_scene_authored_limits(tmp_path):
    from blueprint_pipeline import team_candidate_quick10_registry as registry
    defaults = {"allowed_founder_approvers": [], "max_hourly_rate_usd": 0.8, "open_loop_horizon": 8,
                "screening_candidates": {"custom_droid": {"operator_spec": "sealed.json"}},
                "screening_hard_ttl_seconds": 7200, "screening_candidate_cap_usd": "2.00"}
    settings = {"derived_profiles": {"root": str(tmp_path), "defaults": defaults}}
    assert registry._derived_settings(settings)["defaults"] == defaults
    assert not set(defaults).intersection(registry.DERIVED_FIELDS)
    settings["derived_profiles"]["defaults"]["scene_revision_digest"] = "sha256:" + "a" * 64
    with pytest.raises(registry.TeamCandidateRegistryError, match="derived_profile_settings_invalid"):
        registry._derived_settings(settings)


@pytest.mark.parametrize("interface,candidate_id", [("openpi_droid.v1", "local_alias"), ("groot_n17_droid.v1", "local_alias")])
def test_interface_without_its_actual_delivery_path_is_not_a_capability(interface, candidate_id):
    from blueprint_pipeline.beta_policy_screening_capabilities import _interface_matches
    spec = {"policy_interface_id": interface, "candidate_id": candidate_id}
    assert _interface_matches({}, spec) is False


def test_beta_runtime_requires_its_admission_and_keeps_adp_pair_rule(tmp_path):
    from tests.test_native_task_arena_policy_canary_session import _authority
    authority, inputs = _authority(tmp_path)
    single = deepcopy(inputs)
    single["candidate_ids"] = ["pi05_droid"]
    single["runtime_inputs_digest"] = canonical_digest(single, digest_field="runtime_inputs_digest")
    with pytest.raises(PolicyCanarySessionError, match="candidate_pair_invalid"):
        validate_runtime_input_manifest(single)
    single["run_kind"] = BETA_RUN_KIND
    single["runtime_inputs_digest"] = canonical_digest(single, digest_field="runtime_inputs_digest")
    with pytest.raises(PolicyCanarySessionError, match="beta_admission_binding_invalid"):
        validate_runtime_input_manifest(single)


def test_preparation_refuses_changed_packet_before_allocator(tmp_path, monkeypatch):
    plan, _ = _plan(tmp_path, monkeypatch)
    candidate = next(row for row in plan["candidates"] if row["status"] == "selected")
    scene = launch._read(Path(plan["scene_inputs"]["path"]))
    Path(scene["base_native_packet"]["path"]).write_text("changed\n")
    with pytest.raises(ValueError, match="input_bytes_changed"):
        launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=tmp_path / "attempt")
    assert not (tmp_path / "attempt" / "allocator_launch_intent.json").exists()


def _preflight_kwargs(tmp_path, monkeypatch):
    """The launch shape of test_new_pre_spawn_crash_recovers_only_after_durable_no_create_fence."""
    from blueprint_pipeline import beta_policy_screening_worker as service
    from tests.test_beta_policy_screening_control_contracts import binding

    monkeypatch.setattr(launch, "_validate_prepared", lambda *args: None)
    monkeypatch.setattr(launch, "_check_quote", lambda *args: None)
    monkeypatch.setattr(service, "require_current_source", lambda *args, **kwargs: {})
    authority = launch._write(tmp_path / "authority.json", {"authority_digest": "sha256:" + "c" * 64,
                                                          "resource_name": "canonical-owned-resource"})
    bundle = launch._write(tmp_path / "bundle.json", {"bundle_sha256": "sha256:" + "d" * 64})
    prepared = {"prepared_digest": "sha256:" + "a" * 64, "argv": ["gpu-canary"],
                "authority": launch._record(authority), "bundle_receipt": launch._record(bundle)}
    return dict(plan={"campaign_id": "campaign-one", "source": binding()["source"], "site_id": "site-request1",
                      "plan_digest": "sha256:" + "e" * 64},
                candidate={"candidate_id": "candidate", "configuration_digest": "sha256:" + "f" * 64},
                attempt_dir=tmp_path, prepared=prepared)


@pytest.mark.parametrize("failure", ["http_429", "http_503", "timeout", "unreachable", "incomplete_read"])
def test_control_read_without_answer_before_spawn_is_a_retryable_refusal(tmp_path, monkeypatch, failure):
    import http.client
    import time
    import urllib.error
    from blueprint_pipeline import allocator_invocation as jobs
    from blueprint_pipeline import beta_policy_screening_control_transport as control_transport
    from tests.test_beta_policy_screening_control_contracts import response

    kwargs = _preflight_kwargs(tmp_path, monkeypatch)
    error = {"http_429": ValueError("beta_screening_execution_control_http_429"),
             "http_503": ValueError("beta_screening_execution_control_http_503"),
             "timeout": TimeoutError("The read operation timed out"),
             "unreachable": urllib.error.URLError(ConnectionRefusedError(61, "refused")),
             "incomplete_read": http.client.IncompleteRead(b"{")}[failure]

    def unavailable(**_):
        raise error

    monkeypatch.setattr(control_transport, "request_control", unavailable)
    monkeypatch.setattr(jobs, "start_invocation", lambda *a, **k: pytest.fail("spawned without a current answer"))
    deferred = launch.launch_candidate(**kwargs)
    assert deferred["status"] == "deferred" and deferred["retryable_control_refusal"] is True
    assert deferred["no_allocation_proven"] is True and deferred["provider_allocation_performed"] is False
    # No launch slot is claimed, so the same attempt reads again and can start.
    assert not (tmp_path / "allocator_launch_intent.json").exists()
    assert not (tmp_path / "invocation").exists()
    spawned = []
    monkeypatch.setattr(control_transport, "request_control", lambda *, binding, **_:
                        response(scope=binding, now=int(time.time()), deadline=int(time.time()) + 3600))
    monkeypatch.setattr(jobs, "start_invocation", lambda path, **_: spawned.append(path) or 0)
    monkeypatch.setattr("blueprint_pipeline.team_candidate_quick10_launch._process_start_ticks", lambda pid: "1")
    assert launch.launch_candidate(**kwargs)["status"] == "running" and len(spawned) == 1


@pytest.mark.parametrize("answer", ["stopped", "http_409", "http_403"])
def test_definite_control_answer_before_spawn_still_blocks(tmp_path, monkeypatch, answer):
    import time
    from blueprint_pipeline import beta_policy_screening_control_transport as control_transport
    from tests.test_beta_policy_screening_control_contracts import response

    kwargs = _preflight_kwargs(tmp_path, monkeypatch)

    def definite(*, binding, **_):
        if answer == "stopped":
            return response(scope=binding, now=int(time.time()), deadline=int(time.time()) + 3600, stopped=True)
        raise ValueError(f"beta_screening_execution_control_{answer}")

    monkeypatch.setattr(control_transport, "request_control", definite)
    blocked = launch.launch_candidate(**kwargs)
    assert blocked["status"] == "blocked" and "retryable_control_refusal" not in blocked
    assert launch._read(tmp_path / "allocator_launch_intent.json")["status"] == "refused_before_spawn"
    # The refusal names its typed reason in the ledger's public code form, not the bare class.
    from blueprint_pipeline.beta_policy_screening_coordinator import _BLOCKER
    reason = ("beta_screening_current_source_revoked_before_spawn" if answer == "stopped"
              else f"beta_screening_execution_control_{answer}")
    assert blocked["blockers"] == ["beta_screening_prelaunch_refused:" + reason]
    assert _BLOCKER.fullmatch(blocked["blockers"][0])


@pytest.mark.parametrize(("exc", "expected"), [
    (ValueError("g1_beta_prepared_execution_changed"), "g1_beta_prepared_execution_changed"),
    (ValueError("cannot read /var/lib/blueprint/site/x.json"), "valueerror"),
    (KeyError("scene_inputs"), "keyerror"),
    (FileNotFoundError(2, "No such file"), "filenotfounderror"),
])
def test_a_refusal_before_spawn_keeps_only_a_typed_reason(exc, expected):
    from blueprint_pipeline.beta_policy_screening_coordinator import _BLOCKER

    code = launch.prelaunch_refusal_code(exc)
    assert code == "beta_screening_prelaunch_refused:" + expected
    assert _BLOCKER.fullmatch(code)


def _live_invocation(tmp_path, monkeypatch, *, read):
    """A spawned allocator whose identity is proven; ``read`` answers the control."""
    import time
    from blueprint_pipeline import allocator_invocation as jobs
    from blueprint_pipeline import beta_policy_screening_worker as worker
    from blueprint_pipeline import team_candidate_quick10_launch as native

    monkeypatch.setattr(launch, "_validate_prepared", lambda *args: None)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    prepared = {"prepared_digest": "sha256:" + "c" * 64}
    value = {"site_id": "site-one", "source": {"capture_id": "capture"}, **prepared,
             "initial_observed_at_epoch": time.time(), "execution_deadline_epoch": None}
    path = jobs.retain_invocation(attempt_dir=attempt, binding={**prepared, "provider": "vast"},
                                  argv=["gpu-canary"], control=value)
    (path.parent / "started.json").write_text(json.dumps(
        {"pid": 4242, "start_ticks": "t", "manifest_digest": jobs.manifest(path)["manifest_digest"]}))
    monkeypatch.setattr(native, "_process_start_ticks", lambda pid: "t")
    monkeypatch.setattr(worker, "require_current_source", read)
    kills = []
    monkeypatch.setattr(jobs.os, "kill", lambda pid, signum: kills.append(pid))
    reconcile = dict(plan={}, candidate={}, attempt_dir=attempt, prepared=prepared)
    return path, value, kills, reconcile


def _hold(path):
    """The live allocator holds its job lock, so recovery sees it running, never unstarted."""
    import fcntl
    import os
    fd = os.open(path.parent / "invocation.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _beat(path, *, pid=4242, age=0.0):
    import time
    (path.parent / "monitor_heartbeat.json").write_text(json.dumps({"pid": pid, "beat_at_epoch": time.time() - age}))


def _revoked(*args, **kwargs):
    raise ValueError("beta_screening_current_capture_consent_required")


def test_timer_never_sends_a_second_stop_after_the_monitor_sent_one(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_control as control

    import os
    path, value, kills, reconcile = _live_invocation(tmp_path, monkeypatch, read=_revoked)
    lock = _hold(path)
    monitor = control.CurrentSourceMonitor({"control": value, "root": str(path.parent)},
                                           stop=lambda: kills.append("monitor"))
    monkeypatch.setattr(monitor.done, "wait", lambda interval: False)
    monitor._run()
    assert kills == ["monitor"]
    try:
        assert launch.reconcile_candidate(**reconcile)["status"] == "running"
        assert launch.reconcile_candidate(**reconcile)["status"] == "running"
    finally:
        os.close(lock)
    assert kills == ["monitor"]


def test_timer_signals_only_an_allocator_whose_monitor_is_dead_and_only_once(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_control as control

    import os
    path, value, kills, reconcile = _live_invocation(tmp_path, monkeypatch, read=_revoked)
    lock = _hold(path)
    try:
        assert control.observe(control=value, root=path.parent)["new_processing_authorized"] is False
        _beat(path)  # Its own monitor is alive: it sends the one stop at its next poll.
        launch.reconcile_candidate(**reconcile)
        assert kills == [] and not (path.parent / "stop_signalled.json").exists()
        _beat(path, age=control.MONITOR_STALE_SECONDS + 1)  # Dead monitor: the timer sends it.
        launch.reconcile_candidate(**reconcile)
        assert kills == [4242]
        assert launch._read(path.parent / "stop_signalled.json")["sender"] == "screening_reconcile"
        launch.reconcile_candidate(**reconcile)
        assert kills == [4242]
    finally:
        os.close(lock)


def test_timer_reuses_a_live_monitors_fresh_read(tmp_path, monkeypatch):
    import time
    from blueprint_pipeline import beta_policy_screening_control as control

    import os
    reads = []
    path, value, kills, reconcile = _live_invocation(tmp_path, monkeypatch,
                                                     read=lambda *a, **k: reads.append(1) or {})
    lock = _hold(path)
    try:
        control.observe(control=value, root=path.parent)  # The monitor's own successful read.
        _beat(path)
        assert launch.reconcile_candidate(**reconcile)["status"] == "running"
        assert len(reads) == 1
        # Its last success is no longer fresh (reads failing or slow): the timer reads itself.
        later = time.time() + 3 * control.POLL_SECONDS
        monkeypatch.setattr(launch.time, "time", lambda: later)
        _beat(path)
        assert launch.reconcile_candidate(**reconcile)["status"] == "running"
        assert len(reads) == 2 and kills == []
    finally:
        os.close(lock)


def test_finished_result_is_never_observed_again_or_voided(tmp_path, monkeypatch):
    """Past the grace, with the control unreachable, a completed paid result stays completed."""
    import time
    from blueprint_pipeline import beta_policy_screening_closeout as closeout
    from blueprint_pipeline import policy_canary_provider_null_closeout as null_closeout
    from blueprint_pipeline import team_candidate_quick10_closeout as native_binding

    def unreachable(*args, **kwargs):
        pytest.fail("a finished allocator's result was observed again")

    path, value, kills, reconcile = _live_invocation(tmp_path, monkeypatch, read=unreachable)
    # The allocator finished long ago; its monitor's last success is past the 600 s grace.
    old = json.loads(path.read_text())
    assert old["control"]["initial_observed_at_epoch"] > time.time() - 60
    monkeypatch.setattr(launch.time, "time", lambda: old["control"]["initial_observed_at_epoch"] + 3600)
    attempt = reconcile["attempt_dir"]
    native_result = launch._write(attempt / "native_result.json", {"status": "completed"})
    launch._write(attempt / "allocator_result.json", {
        "status": "completed", "provider_mutations_performed": 1, "vast_instance_ids": [991],
        "provider_create_attempted": True, "native_control_result_path": str(native_result)})
    monkeypatch.setattr(null_closeout, "proven_provider_null_closeout", lambda *a, **k: None)
    monkeypatch.setattr(closeout, "collect_scoped_closeout", lambda **kwargs: {"teardown": "verified"})
    monkeypatch.setattr(closeout, "scoped_closeout_verified", lambda *args, **kwargs: True)
    monkeypatch.setattr(closeout, "join_scoped_session_closeout", lambda **kwargs: {
        "beta_screening_admission": {"plan_digest": "sha256:" + "9" * 64}, "blockers": []})
    monkeypatch.setattr(native_binding, "validate_native_result_binding", lambda *a, **k: None)
    monkeypatch.setattr(launch, "validate_session_result", lambda value: value)
    result = launch.reconcile_candidate(**{**reconcile, "plan": {"plan_digest": "sha256:" + "9" * 64}})
    assert result["status"] == "completed" and "applicability" not in result
    assert not (path.parent / "stop_requested.json").exists() and kills == []


@pytest.mark.parametrize("stop,retryable", [("outage_grace", True), ("revoked", False)])
def test_stop_before_any_create_blocks_only_on_a_definite_answer(tmp_path, monkeypatch, stop, retryable):
    """The monitor's SIGTERM interrupted the adapter before its create: nothing was spent."""
    from blueprint_pipeline import beta_policy_screening_control as control

    def unreachable(*args, **kwargs):
        raise OSError("control unreachable")

    path, value, kills, reconcile = _live_invocation(
        tmp_path, monkeypatch, read=_revoked if stop == "revoked" else unreachable)
    moment = value["initial_observed_at_epoch"] + (control.MONITOR_GRACE_SECONDS if stop == "outage_grace" else 1)
    assert control.observe(control=value, root=path.parent, now=moment)["new_processing_authorized"] is False
    launch._write(reconcile["attempt_dir"] / "allocator_result.json", {
        "status": "blocked", "provider_mutations_performed": 0, "provider_create_attempted": False,
        "blockers": ["vast_probe_interrupted_before_completion"]})
    result = launch.reconcile_candidate(**reconcile)
    assert result["status"] == "blocked" and result["no_allocation_proven"] is True
    assert result.get("retryable_unstarted_refusal", False) is retryable
    assert (path.parent / "no_create_fence.json").exists() is retryable
