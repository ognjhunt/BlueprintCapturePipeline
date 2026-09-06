"""Persistent owner pair reaches real handoff/profile/native-spec producers."""
from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import task_evaluation_scene_policy_binding as binding
from blueprint_pipeline import task_evaluation_policy_canary_scene_setup as native_setup
from blueprint_pipeline import task_evaluation_policy_canary_preparation_dispatch as dispatch
from blueprint_pipeline.adp009d_policy_candidate_admission import EXPECTED_CANDIDATES
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests import test_task_evaluation_policy_canary_handoff as rehearsal
from tests.test_task_evaluation_scene_intake import request
from tests.test_task_evaluation_policy_canary_preparation_dispatch import _profile_and_request
from blueprint_pipeline.task_evaluation_launch_dispatcher import dispatch_launch_request, validate_launch_request


def _seal(value, field):
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _owner_scene(tmp_path, monkeypatch, *, wrong_pair=False, cap=20,
                 interpretation=False, allow_openai=True):
    state = rehearsal._prepared(tmp_path)
    now = time.time()
    owner = request()
    owner["task"]["task_id"] = rehearsal.TASK_ID
    if interpretation:
        owner["task"]["episode_interpretation"] = True
    pair = [{"id": key, "artifact_digest": EXPECTED_CANDIDATES[key]["checkpoint_inventory_digest"]}
            for key in ("pi05_droid", "groot_n17_droid")]
    if wrong_pair:
        pair[0]["artifact_digest"] = "sha256:" + "0" * 64
    owner["execution"].update(policy_candidates=pair, expires_at_epoch=now + 3600,
        max_total_spend_usd=cap, max_paid_attempts=8,
        allowed_providers=["vast", "openai"] if allow_openai else ["vast"])
    owner["consent"]["accepted_at_epoch"] = now - 1
    root = tmp_path / "scene-intents"
    staged = intake.stage_scene_intent(value=owner, queue_root=root,
        authenticated_client="webapp", trusted_clients={"webapp"}, now=now)
    monkeypatch.setenv(intake.ROOT_ENV, str(root))
    monkeypatch.setenv(intake.CLIENTS_ENV, "webapp")
    monkeypatch.delenv("BLUEPRINT_TASK_EVALUATION_CONTROLS_AUTOPROVISION_CONFIG", raising=False)
    profile_path = tmp_path / "launch-runs" / rehearsal.SOURCE_LAUNCH_ID / "launch_profile.json"
    profile = json.loads(profile_path.read_text())
    initial = intake.reserve_scene_attempt(queue_root=root, intent_id=staged["intent_id"],
        attempt_id="initial-configuration", source_commit=rehearsal.CONFIGURATION_COMMIT,
        runtime_digest="sha256:" + "d" * 64, input_digest="sha256:" + "e" * 64,
        provider="vast", maximum_spend_usd=min(1, cap))
    profile.update(scene_intent_digest=staged["intent_digest"], scene_attempt_id=initial["attempt_id"],
        scene_attempt_binding={"schema_version": "task_evaluation_scene_attempt_binding.v1",
            **{key: initial[key] for key in ("intent_id", "intent_digest", "attempt_id", "source_commit",
                                          "runtime_digest", "input_digest")}})
    rehearsal._write(profile_path, _seal(profile, "profile_digest"))
    launch_path = profile_path.parent / "launch_request.json"
    launch = json.loads(launch_path.read_text())
    launch["launch_profile_digest"] = profile["profile_digest"]
    rehearsal._write(launch_path, _seal(launch, "request_digest"))
    return state, root / staged["intent_id"], pair


def _run(tmp_path, state):
    webapp = rehearsal._WebApp()
    profile_calls = []
    result = rehearsal._advance(tmp_path, state=state, webapp=webapp,
        publisher=rehearsal._Publisher(), profile_calls=profile_calls)
    profile = json.loads(profile_calls[0]["profile_path"].read_text())
    return result, profile, webapp


def test_owner_checkpoint_pair_reaches_real_profile_and_native_specs(tmp_path, monkeypatch):
    state, owner_dir, pair = _owner_scene(tmp_path, monkeypatch)
    result, profile, webapp = _run(tmp_path, state)
    assert result["status"] == "canary_launch_submitted"
    assert profile["scene_policy_candidates"] == pair
    plan = profile["internal_policy_canary_execution_plan"]
    bound = plan["scene_policy_binding"]
    assert bound["attempt_id"] == profile["scene_attempt_id"] != "initial-configuration"
    assert binding.profile_binding_blockers(profile) == []
    assert dispatch.policy_canary_execution_plan_blockers(profile) == []
    attempts = list((owner_dir / "attempts").glob("*.json"))
    assert len(attempts) == 2
    reservation = intake._read(owner_dir / "attempts" / (bound["attempt_id"] + ".json"), "attempt_digest")
    assert reservation["maximum_spend_usd"] == profile["allocator"]["max_spend_usd"] == 4
    assert (reservation["runtime_digest"], reservation["input_digest"]) == binding.policy_attempt_identity(plan, pair)
    assert webapp.calls[0]["selection"]["episode_interpretation"] == {"enabled": False}

    parameters = json.loads((state / "policy-canary-inputs" / "presubmission_parameters.json").read_text())
    accepted = inspect.signature(native_setup.materialize_scene839873_policy_canary_setup).parameters
    native = {key: value for key, value in parameters.items() if key in accepted}
    native.update(activation_digest="sha256:" + "a" * 64, capture_session_id="fixture-capture",
        intake_id="fixture-intake", output_dir=str(tmp_path / "actual-native-specs"),
        scene_policy_binding=bound)
    emitted = native_setup.materialize_scene839873_policy_canary_setup(**native)
    specs = [json.loads(Path(emitted["records"][key]["path"]).read_text())
             for key in ("pi05_execution_spec", "groot_execution_spec")]
    binding.validate_execution_specs(specs, bound)
    assert {row["candidate_id"]: row["checkpoint_digest"] for row in specs} == binding.candidate_map(pair)
    assert emitted["scene_policy_binding"] == bound
    assert emitted["scene_attempt_binding"] == profile["scene_attempt_binding"]
    assert binding.execution_setup_binding_blockers(emitted, specs) == []
    wrong_spec = json.loads(json.dumps(specs))
    wrong_spec[0]["checkpoint_digest"] = "sha256:" + "0" * 64
    assert binding.execution_setup_binding_blockers(emitted, wrong_spec) == ["scene_policy_execution_checkpoint_mismatch"]
    missing_owner_field = {key: value for key, value in emitted.items() if key != "scene_intent_digest"}
    assert binding.execution_setup_binding_blockers(missing_owner_field, specs) == ["scene_policy_execution_owner_fields_missing"]
    repeated_candidate = {**emitted, "candidate_ids": [*emitted["candidate_ids"], emitted["candidate_ids"][0]]}
    assert binding.execution_setup_binding_blockers(repeated_candidate, specs) == ["scene_policy_execution_owner_binding_mismatch"]
    # The actual post-activation template consumer must preserve the same pair.
    activation_path = rehearsal._write(tmp_path / "activation-result.json", {
        "schema_version": "task_evaluation_launch_activation_result.v1",
        "status": "policy_campaign_queue_materialized_no_execution",
        "policy_campaign_activation_digest": "sha256:" + "a" * 64})
    public = profile["internal_policy_canary_setup"]
    envelope = _seal({"schema_version": "task_evaluation_policy_canary_dispatch_envelope.v1",
        "run_kind": "internal_policy_canary", "claim_ceiling": "diagnostic_policy_execution",
        "source_commit": rehearsal.COMMIT, "capture_session_id": "fixture-capture",
        "intake_id": "fixture-intake", "request_digest": "sha256:" + "9" * 64,
        "task_success_contract": public["task_success_contract"],
        "task_success_contract_digest": public["task_success_contract_digest"],
        "activation_result": {"path": str(activation_path), "size_bytes": activation_path.stat().st_size,
                              "sha256": rehearsal._digest(activation_path)}}, "envelope_digest")
    template_path = state / "policy-canary-presubmission" / "task_evaluation_policy_canary_execution_setup_template.v1.json"
    post_activation = native_setup.materialize_scene839873_policy_canary_setup_from_template(
        template_path=template_path, activation_envelope=envelope, output_dir=tmp_path / "post-activation")
    assert post_activation["scene_policy_binding"] == bound
    template = json.loads(template_path.read_text())
    del template["scene_policy_binding"]
    changed_template = rehearsal._write(tmp_path / "missing-binding-template.json", _seal(template, "template_digest"))
    with pytest.raises(ValueError, match="execution_template_binding_missing"):
        native_setup.materialize_scene839873_policy_canary_setup_from_template(
            template_path=changed_template, activation_envelope=envelope, output_dir=tmp_path / "must-not-materialize")
    # Restart adopts the same handoff/attempt; it cannot reserve another run.
    again = rehearsal._advance(tmp_path, state=state, webapp=webapp,
        publisher=rehearsal._Publisher(), profile_calls=[])
    assert again == result
    assert len(list((owner_dir / "attempts").glob("*.json"))) == 2


def test_wrong_owner_checkpoint_refuses_before_policy_publication(tmp_path, monkeypatch):
    state, owner_dir, _ = _owner_scene(tmp_path, monkeypatch, wrong_pair=True)
    calls = []
    webapp = rehearsal._WebApp()
    with pytest.raises(ValueError, match="checkpoint_mismatch"):
        rehearsal._advance(tmp_path, state=state, webapp=webapp,
            publisher=rehearsal._Publisher(), profile_calls=calls)
    assert calls == [] and webapp.calls == []
    assert len(list((owner_dir / "attempts").glob("*.json"))) == 1


def test_policy_attempt_does_not_borrow_controls_budget(tmp_path, monkeypatch):
    state, owner_dir, _ = _owner_scene(tmp_path, monkeypatch, cap=4)
    with pytest.raises(ValueError, match="spend_cap_exhausted"):
        _run(tmp_path, state)
    assert len(list((owner_dir / "attempts").glob("*.json"))) == 1


def test_interpretation_requires_explicit_request_and_separate_hold(tmp_path, monkeypatch):
    state, owner_dir, _ = _owner_scene(tmp_path, monkeypatch, interpretation=True)
    _, _, webapp = _run(tmp_path, state)
    assert webapp.calls[0]["selection"]["episode_interpretation"]["enabled"] is True
    rows = [intake._read(path, "attempt_digest") for path in (owner_dir / "attempts").glob("*.json")]
    assert len(rows) == 3
    assert [(row["provider"], row["maximum_spend_usd"]) for row in rows if row["provider"] == "openai"] == [("openai", 1.5)]


def test_interpretation_cannot_add_an_unapproved_provider(tmp_path, monkeypatch):
    state, _, _ = _owner_scene(tmp_path, monkeypatch, interpretation=True, allow_openai=False)
    webapp = rehearsal._WebApp()
    with pytest.raises(ValueError, match="interpretation_provider_not_authorized"):
        rehearsal._advance(tmp_path, state=state, webapp=webapp,
            publisher=rehearsal._Publisher(), profile_calls=[])
    assert webapp.calls == []


@pytest.mark.parametrize("field", ["scene_policy_candidates", "scene_attempt_id", "scene_intent_digest"])
def test_profile_cannot_drop_owner_pair_or_attempt(tmp_path, monkeypatch, field):
    state, _, _ = _owner_scene(tmp_path, monkeypatch)
    _, profile, _ = _run(tmp_path, state)
    del profile[field]
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    assert binding.profile_binding_blockers(profile)


def test_corrupt_runtime_or_checkpoint_is_refused_after_resealing(tmp_path, monkeypatch):
    state, _, _ = _owner_scene(tmp_path, monkeypatch)
    _, profile, _ = _run(tmp_path, state)
    plan = profile["internal_policy_canary_execution_plan"]
    plan["preparation_template"]["execution_adapter"]["runtime_source_bundle"]["digest"] = "sha256:" + "0" * 64
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    assert "scene_policy_plan_attempt_identity_mismatch" in binding.profile_binding_blockers(profile)


def test_revocation_after_profile_publication_blocks_queued_preparation(tmp_path, monkeypatch):
    state, owner_dir, _ = _owner_scene(tmp_path, monkeypatch)
    _, profile, _ = _run(tmp_path, state)
    rehearsal._write(owner_dir / "revoked.json", {})
    request = {"run_kind": "internal_policy_canary", "launch_id": "new-policy-request",
        "run_id": "new-policy-request", "request_digest": "sha256:" + "9" * 64}
    receipt = dispatch.maybe_dispatch_policy_canary_preparation(request=request, profile=profile,
        blockers=[], state_root=tmp_path / "queued-launches", preparation_queue_root=tmp_path / "queue")
    assert receipt["status"] == "blocked"
    assert "scene_policy_owner_revoked" in receipt["blockers"]
    assert not (tmp_path / "queue").exists()


def test_emitted_private_binding_schemas_match_the_producers(tmp_path, monkeypatch):
    import jsonschema
    from referencing import Registry, Resource
    state, _, _ = _owner_scene(tmp_path, monkeypatch)
    _, profile, _ = _run(tmp_path, state)
    schema_root = Path(__file__).resolve().parents[1] / "docs" / "schemas"
    names = ["task_evaluation_scene_policy_binding.v1", "task_evaluation_policy_canary_execution_plan.v1",
             "task_evaluation_policy_canary_profile_materialization_input.v1", "rigid_task_success_contract.v1"]
    schemas = {name: json.loads((schema_root / (name + ".schema.json")).read_text()) for name in names}
    registry = Registry().with_resources((value["$id"], Resource.from_contents(value)) for value in schemas.values())
    plan = profile["internal_policy_canary_execution_plan"]
    wrapper = json.loads((state / "policy-canary-presubmission" /
        "task_evaluation_policy_canary_profile_materialization_input.v1.json").read_text())
    for name, value in zip(names, (plan["scene_policy_binding"], plan, wrapper)):
        schema = schemas[name]
        validator = jsonschema.Draft202012Validator(schema, registry=registry)
        validator.validate(value)


def test_crash_after_reservation_before_handoff_checkpoint_reuses_same_attempt(tmp_path, monkeypatch):
    state, owner_dir, _ = _owner_scene(tmp_path, monkeypatch)
    original = rehearsal.handoff._seal_state
    def interrupt(path, value):
        if value["status"] == "canary_presubmitted":
            raise OSError("crash before checkpoint")
        return original(path, value)
    monkeypatch.setattr(rehearsal.handoff, "_seal_state", interrupt)
    with pytest.raises(OSError, match="crash before checkpoint"):
        _run(tmp_path, state)
    assert len(list((owner_dir / "attempts").glob("*.json"))) == 2
    monkeypatch.setattr(rehearsal.handoff, "_seal_state", original)
    monkeypatch.setattr(native_setup, "utc_now_iso", lambda: "2099-01-01T00:00:00Z")
    _, profile, _ = _run(tmp_path, state)
    assert len(list((owner_dir / "attempts").glob("*.json"))) == 2
    assert profile["internal_policy_canary_setup"]["episode_presets"][0]["estimate"]["as_of"] != "2099-01-01T00:00:00Z"


def test_real_dispatch_queues_same_pair_and_owner_digest_without_allocator(tmp_path, monkeypatch):
    state, _, pair = _owner_scene(tmp_path, monkeypatch)
    _, profile, _ = _run(tmp_path, state)
    template_root = tmp_path / "request-template"
    template_root.mkdir()
    _, selected = _profile_and_request(template_root)
    public = profile["internal_policy_canary_setup"]
    plan = profile["internal_policy_canary_execution_plan"]
    matrix = public["episode_presets"][0]["matrix"]
    selected.update({key: profile[key] for key in ("source_bundle", "evaluation_run_spec", "source_commit")})
    selected.update(launch_profile_id=profile["profile_id"], launch_profile_digest=profile["profile_digest"],
        source_launch_id=public["source_launch_id"], offering_digest=public["offering_digest"],
        setup_digest=public["setup_digest"], scene_revision_digest=public["scene_revision_digest"],
        task_success_contract=public["task_success_contract"], task_success_contract_digest=public["task_success_contract_digest"])
    selected["episode_plan"].update(variation_matrix_digest=matrix["matrix_digest"],
        resolved_cells=matrix["cells"], resolved_seeds=[row["seed"] for row in matrix["cells"]])
    selected["authorization"]["actor"]["id"] = "u1"
    selected["authorization"]["spend"].update(max_spend_usd=plan["resource_authority"]["hard_cap_usd"],
        hard_ttl_seconds=plan["resource_authority"]["hard_ttl_seconds"])
    selected.pop("episode_interpretation_authority")
    selected.pop("episode_interpretation_source_rights_admission")
    _seal(selected, "request_digest")
    assert validate_launch_request(selected) == []
    profiles = tmp_path / "actual-dispatch-profiles"
    rehearsal._write(profiles / (profile["profile_id"] + ".json"), profile)
    selected_path = rehearsal._write(tmp_path / "selected-policy-request.json", selected)
    queue = tmp_path / "actual-preparation-queue"
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_QUEUE_ROOT", str(queue))
    receipt = dispatch_launch_request(request_path=selected_path, profile_dir=profiles,
        state_root=tmp_path / "actual-dispatch", execute=True,
        allocator_runner=lambda argv: pytest.fail("No provider allocator may run during preparation"))
    assert receipt["status"] == "queued_for_no_spend_preparation", receipt.get("blockers")
    assert receipt["allocator_invoked"] is False and receipt["provider_mutation_attempted"] is False
    preparation = json.loads(next((queue / "pending").glob("*.json")).read_text())["request"]
    assert preparation["scene_intent_digest"] == profile["scene_intent_digest"]
    assert set(preparation["policy_run_configuration"]["candidate_ids"]) == set(binding.candidate_map(pair))


def test_late_interpretation_selection_cannot_bypass_owner_request(tmp_path, monkeypatch):
    state, _, _ = _owner_scene(tmp_path, monkeypatch)
    _, profile, _ = _run(tmp_path, state)
    request = {"run_kind": "internal_policy_canary", "launch_id": "new-policy-request",
        "run_id": "new-policy-request", "request_digest": "sha256:" + "9" * 64,
        "episode_interpretation_authority": {"maximum_cost_usd": 1.5, "interpreter": {"provider_id": "openai"}}}
    receipt = dispatch.maybe_dispatch_policy_canary_preparation(request=request, profile=profile,
        blockers=[], state_root=tmp_path / "queued-launches", preparation_queue_root=tmp_path / "queue")
    assert receipt["status"] == "blocked"
    assert "scene_policy_interpretation_not_requested" in receipt["blockers"]
    assert not (tmp_path / "queue").exists()
