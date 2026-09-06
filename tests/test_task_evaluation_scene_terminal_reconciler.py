"""Terminal downstream receipts reconcile into the persistent owner status.

Spec E: ``_advance_intent`` recognised an already-completed projection but never
emitted completion from downstream controls/policy/publication receipts; its
successful tail only reached ``awaiting_execution``. These tests exercise the
terminal owner-result join against the REAL retained receipt shapes (the policy
canary result projection, the authenticated Website readback, the launch
reconciler's post-teardown provider-zero closure, the owner ``scene_policy``
binding and reserved attempt) and assert:

* a terminal completed-unqualified diagnostic updates the owner status to
  ``completed`` with a result reference whose digest matches the projection,
* the authenticated Website readback matches the same intent/result digest,
* a blocked result preserves failed children and stays ``blocked``,
* absent resource closure, an absent authenticated readback, a stale receipt and
  a changed release all remain explicit and never fabricate completion,
* duplicate ticks and a worker restart are idempotent.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import task_evaluation_scene_policy_binding as scene_policy
from blueprint_pipeline import task_evaluation_scene_terminal_reconciler as reconciler
from blueprint_pipeline.decision_evidence_contracts import (
    canonical_digest,
    cross_runtime_canonical_digest,
)
from tests.test_task_evaluation_scene_intake import request as owner_request
from tests.test_task_evaluation_policy_canary_setup import _setup as public_setup


COMMIT = "d" * 40
OTHER_COMMIT = "e" * 40
RUNTIME_DIGEST = "sha256:" + "f" * 64
INPUT_DIGEST = "sha256:" + "1" * 64
RUN_ID = "scene-1-policy-canary-abc123"
REQUEST_DIGEST = "sha256:" + "2" * 64
CONFIG_DIGEST = "sha256:" + "3" * 64


def _write(path: Path, value: dict, field: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if field:
        value = dict(value)
        value[field] = canonical_digest(value, digest_field=field)
    path.write_text(json.dumps(value))
    return path


def _artifact(character: str, artifact_id: str) -> dict:
    return {"digest": "sha256:" + character * 64, "size_bytes": 10, "artifact_id": artifact_id}


def _blocked_projection() -> dict:
    setup = public_setup()
    result = {
        "schema_version": "task_evaluation_policy_canary_result_projection.v1",
        "run_id": RUN_ID,
        "request_digest": REQUEST_DIGEST,
        "configuration_digest": CONFIG_DIGEST,
        "result_delivery_digest": "sha256:" + "9" * 64,
        "task_success_contract": setup["task_success_contract"],
        "task_success_contract_digest": setup["task_success_contract_digest"],
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "scene_controls_status": "configured_controls_pending",
        "result_status": "blocked",
        "warning": "Controls pending — results are unqualified.",
        "counts": {
            "policy_count": 2, "episodes_per_policy": 10, "learned_policy_rollout_count": 20,
            "completed_learned_policy_rollout_count": 0,
            "diagnostic_control_rollout_count": 20, "completed_diagnostic_control_rollout_count": 0,
        },
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "candidate_results": [
            {"candidate_id": candidate, "episodes_completed": 0, "interpretable_episode_count": 0,
             "actions_delivered_episode_count": 0, "metrics": {}, "failure_counts": {"pre_provider_blocked": 10}}
            for candidate in ("pi05_droid", "groot_n17_droid")
        ],
        "episodes": [
            {
                "episode_id": "episode-1", "candidate_id": "pi05_droid", "cell_id": "cell-1", "seed": 1,
                "terminal_state": "blocked", "candidate_policy_queried": False, "actions_reached_robot": False,
                "arm_moved": False, "policy_outcome_interpretable": False, "failure_taxonomy": "RuntimeError",
                "evidence": {
                    "checkpoint_digest": "sha256:" + "a" * 64, "runtime_identity_digest": "sha256:" + "b" * 64,
                    "reset_state_digest": "sha256:" + "c" * 64, "reset_state": None, "frame_manifest": None,
                    "review_video": None, "policy_query_receipt": None, "action_sequence": None,
                    "action_delivery_readback": None, "state_trace": None, "contact_force_trace": None,
                    "task_object_trajectory": None, "score_receipt": None,
                    "evidence_gaps": ["before_first_observation"],
                    "typed_media_gap": "provider_runtime_failed_before_first_observation",
                },
            }
        ],
        "comparison": {"matched_cell_count": 0, "winner_declared": False, "official_ranking_contribution": False},
        "report": {
            "result_digest": "sha256:" + "3" * 64,
            "permanent_result_path": "/internal/task-evaluation-runs/" + RUN_ID,
            "machine_readable_report": _artifact("4", "full-report"),
            "evidence_manifest": _artifact("5", "evidence-manifest"),
        },
        "closure": {
            "billing": _artifact("6", "billing"), "teardown": _artifact("7", "teardown"),
            "provider_zero": {**_artifact("8", "provider-zero"), "provider_zero_verified": True},
        },
        "notification_delivery": {
            "terminal_state": "blocked", "status": "pending", "attempts": 0,
            "provider": "website_terminal_handler", "message_id": None, "delivered_at": None,
            "run_result_digest": "sha256:" + "3" * 64,
        },
        "blockers": ["provider_capacity_unavailable"],
        "projection_digest": "",
    }
    result["projection_digest"] = cross_runtime_canonical_digest(result, digest_field="projection_digest")
    return result


def _completed_projection() -> dict:
    setup = public_setup()
    roles = ("reset_state", "frame_manifest", "review_video", "policy_query_receipt", "action_sequence",
             "action_delivery_readback", "state_trace", "contact_force_trace", "task_object_trajectory",
             "score_receipt")
    episodes = []
    for candidate in ("pi05_droid", "groot_n17_droid"):
        for cell in range(10):
            evidence = {
                "checkpoint_digest": "sha256:" + "1" * 64, "runtime_identity_digest": "sha256:" + "2" * 64,
                "reset_state_digest": "sha256:" + "3" * 64, "evidence_gaps": [],
            }
            for role in roles:
                evidence[role] = _artifact("a", f"{candidate}-{cell}-{role}")
            episodes.append({
                "episode_id": f"{RUN_ID}--cell{cell}--{candidate}", "candidate_id": candidate,
                "cell_id": f"cell{cell}", "seed": cell, "terminal_state": "completed",
                "candidate_policy_queried": True, "actions_reached_robot": True, "arm_moved": True,
                "policy_outcome_interpretable": True, "failure_taxonomy": None, "interpretation": None,
                "evidence": evidence,
            })
    result = {
        "schema_version": "task_evaluation_policy_canary_result_projection.v1",
        "run_id": RUN_ID, "request_digest": REQUEST_DIGEST, "configuration_digest": CONFIG_DIGEST,
        "result_delivery_digest": "sha256:" + "9" * 64,
        "task_success_contract": setup["task_success_contract"],
        "task_success_contract_digest": setup["task_success_contract_digest"],
        "run_kind": "internal_policy_canary", "claim_ceiling": "diagnostic_policy_execution",
        "scene_controls_status": "configured_controls_pending", "result_status": "completed_unqualified",
        "warning": "Controls pending — results are unqualified.",
        "counts": {
            "policy_count": 2, "episodes_per_policy": 10, "learned_policy_rollout_count": 20,
            "completed_learned_policy_rollout_count": 20,
            "diagnostic_control_rollout_count": 20, "completed_diagnostic_control_rollout_count": 0,
        },
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "candidate_results": [
            {"candidate_id": candidate, "episodes_completed": 10, "interpretable_episode_count": 10,
             "actions_delivered_episode_count": 10, "metrics": {}, "failure_counts": {}}
            for candidate in ("pi05_droid", "groot_n17_droid")
        ],
        "episodes": episodes,
        "comparison": {"matched_cell_count": 10, "winner_declared": False, "official_ranking_contribution": False},
        "report": {
            "result_digest": "sha256:" + "3" * 64,
            "permanent_result_path": "/internal/task-evaluation-runs/" + RUN_ID,
            "machine_readable_report": _artifact("4", "full-report"),
            "evidence_manifest": _artifact("5", "evidence-manifest"),
        },
        "closure": {
            "billing": _artifact("6", "billing"), "teardown": _artifact("7", "teardown"),
            "provider_zero": {**_artifact("8", "provider-zero"), "provider_zero_verified": True},
        },
        "notification_delivery": {
            "terminal_state": "completed", "status": "pending", "attempts": 0,
            "provider": "website_terminal_handler", "message_id": None, "delivered_at": None,
            "run_result_digest": "sha256:" + "3" * 64,
        },
        "blockers": [], "projection_digest": "",
    }
    result["projection_digest"] = cross_runtime_canonical_digest(result, digest_field="projection_digest")
    return result


def _readback(projection: dict, *, status: str = "succeeded") -> dict:
    return {
        "schema_version": "task_evaluation_policy_canary_webapp_sync_result.v1",
        "capture_session_id": "capture-1", "intake_id": "intake-1",
        "run_id": projection["run_id"], "request_digest": projection["request_digest"],
        "configuration_digest": projection["configuration_digest"],
        "result_status": projection["result_status"],
        "result_delivery_digest": projection["result_delivery_digest"],
        "policy_canary_projection_digest": projection["projection_digest"],
        "status": status, "attempts": 1,
        "notification_delivery": {
            "terminal_state": "completed" if projection["result_status"] == "completed_unqualified" else "blocked",
            "status": "accepted", "attempts": 1, "provider": "resend", "message_id": "message-1",
            "delivered_at": None, "run_result_digest": projection["projection_digest"],
        },
    }


def _closure(*, status: str = "provider_zero_confirmed") -> dict:
    """The REAL terminal resource-closure evidence of a policy-canary run: the
    dispatcher's sealed Vast post-teardown provider-zero receipt
    (``post_teardown_global_provider_zero.json``). The launch reconciler never
    produces a ``task_evaluation_post_teardown_provider_zero.v1`` closure for a
    canary launch (its launch receipt is ``execute_requested: False``), so that
    schema is not this run's closure and is not accepted here."""
    confirmed = status == "provider_zero_confirmed"
    value = {
        "schema_version": "task_evaluation_policy_canary_vast_provider_zero.v1",
        "status": status, "provider": "vast", "inventory_scope": "global_billable_resources",
        "api_confirmed": confirmed, "live_instance_count": 0 if confirmed else 1,
        "provider_zero_verified": confirmed,
        "global_gpu_guard_snapshot": {"path": "/var/lib/blueprint/pipeline-control-plane/gpu_spend_guard/latest.json",
                                      "size_bytes": 8470, "sha256": "sha256:" + "b" * 64},
        "blockers": [] if confirmed else ["vast_live_instances_remaining"],
        "raw_provider_response_recorded": False, "receipt_digest": "",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    return value


def _file_record(path: Path) -> dict:
    import hashlib
    return {"path": str(path), "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size}


def _binder(projection: dict, readback: dict | None, *, projection_path: Path, sync_path: Path | None,
            closure_path: Path | None) -> dict:
    """The dispatcher's sealed ``dispatch_receipt.json`` (task_evaluation_policy_canary_dispatch.v1):
    the one producer record binding the projection, the persisted Website sync and
    the provider-zero closure to THIS run by file digest."""
    value = {
        "schema_version": "task_evaluation_policy_canary_dispatch.v1", "status": projection["result_status"],
        "run_id": projection["run_id"], "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution", "authority_digest": "sha256:" + "1" * 64,
        "bundle_sha256": "sha256:" + "2" * 64,
        "terminal_result": {"path": "/run/policy_canary_terminal_result.json", "sha256": "sha256:" + "3" * 64,
                            "size_bytes": 10},
        "result_delivery_digest": projection["result_delivery_digest"],
        "policy_canary_projection_digest": projection["projection_digest"],
        "policy_canary_result_projection": _file_record(projection_path),
        **({"policy_canary_webapp_sync": _file_record(sync_path)} if sync_path is not None else {}),
        "notification_delivery": (readback or {}).get("notification_delivery"),
        "official_billing": {"path": "/run/official_billing_reconciliation.json", "sha256": "sha256:" + "6" * 64,
                             "size_bytes": 10, "official_billing_sealed": True},
        "teardown": {"path": "/run/teardown.json", "sha256": "sha256:" + "7" * 64, "size_bytes": 10,
                     "teardown_completed": True},
        **({"provider_zero": {**_file_record(closure_path), "provider_zero_verified": True}}
           if closure_path is not None else {}),
        "allocator_invoked": False, "automatic_retry_performed": False, "scene_promotion_performed": False,
        "official_ranking_performed": False, "retry_cap": 0, "receipt_digest": "",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    return value


def _publication(projection: dict) -> dict:
    value = {
        "schema_version": "task_evaluation_scene_terminal_result_publication.v1",
        "run_id": projection["run_id"],
        "uri": "s3://blueprint-task-evaluation-artifacts/control-plane-evidence/sha256/" + "a" * 64 + "/run.tar",
        "digest": projection["projection_digest"], "archive_digest": "sha256:" + "a" * 64, "size_bytes": 4096,
        "provider_allocated": False, "publication_digest": "",
    }
    value["publication_digest"] = canonical_digest(value, digest_field="publication_digest")
    return value


def _env(tmp_path: Path, *, commit: str = COMMIT, source_cost: float = 2.0, policy_cost: float = 1.0):
    """Real owner intent + reserved scene/policy attempts + scene_policy binding."""
    now = time.time()
    intake_root = tmp_path / "intents"
    owner = owner_request()
    owner["execution"].update(max_total_spend_usd=8, max_paid_attempts=4,
                              allowed_providers=["vast"], expires_at_epoch=now + 3600)
    owner["consent"]["accepted_at_epoch"] = now - 1
    receipt = intake.stage_scene_intent(value=owner, queue_root=intake_root,
                                         authenticated_client="blueprint-webapp",
                                         trusted_clients={"blueprint-webapp"}, now=now)
    intent_id = receipt["intent_id"]
    directory = intake_root / intent_id
    intent = json.loads((directory / "intent.json").read_text())
    # The scene-configuration attempt and the policy-canary attempt both sit on
    # the owner intent; the terminal policy result binds to the policy attempt.
    intake.reserve_scene_attempt(queue_root=intake_root, intent_id=intent_id, attempt_id="source-1",
                                 source_commit=commit, runtime_digest=RUNTIME_DIGEST, input_digest="sha256:" + "5" * 64,
                                 provider="vast", maximum_spend_usd=source_cost, now=now)
    policy_attempt_id = "policy-" + INPUT_DIGEST.removeprefix("sha256:")[:48]
    intake.reserve_scene_attempt(queue_root=intake_root, intent_id=intent_id, attempt_id=policy_attempt_id,
                                 source_commit=commit, runtime_digest=RUNTIME_DIGEST, input_digest=INPUT_DIGEST,
                                 provider="vast", maximum_spend_usd=policy_cost, now=now)
    binding = scene_policy.seal_binding(
        scene_intent_digest=intent["intent_digest"], attempt_id=policy_attempt_id,
        policy_candidates=owner["execution"]["policy_candidates"],
        runtime_digest=RUNTIME_DIGEST, input_digest=INPUT_DIGEST)
    config = {"intent_root": str(intake_root), "terminal_result_root": str(tmp_path / "terminal")}
    release = {"source_commit": commit, "runtime_digest": RUNTIME_DIGEST}
    return dict(intent=intent, intent_id=intent_id, directory=directory, config=config, release=release,
                binding=binding, now=now, output=tmp_path / "out")


def _profile(binding: dict, *, commit: str = COMMIT) -> dict:
    profile = {
        "schema_version": "task_evaluation_launch_profile.v1", "profile_id": "policy-canary-profile",
        "source_commit": commit,
        "internal_policy_canary_execution_plan": {"scene_policy_binding": binding},
        "profile_digest": "",
    }
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    return profile


def _launch_request(profile_digest: str, *, commit: str = COMMIT) -> dict:
    return {
        "schema_version": "task_evaluation_launch_request.v1", "run_id": RUN_ID, "source_commit": commit,
        "request_digest": REQUEST_DIGEST, "launch_profile_digest": profile_digest,
    }


def _receipts(env, *, projection=None, readback=None, include_readback=True, include_closure=True,
              include_publication=True, include_binder=True, closure_status="provider_zero_confirmed",
              profile_binding=None, commit=COMMIT):
    """Write the full owner-scoped terminal receipt set the reconciler joins."""
    projection = projection if projection is not None else _completed_projection()
    root = Path(env["config"]["terminal_result_root"]) / env["intent_id"]
    profile = _profile(profile_binding if profile_binding is not None else env["binding"], commit=commit)
    _write(root / "launch_profile.json", profile)
    _write(root / "launch_request.json", _launch_request(profile["profile_digest"], commit=commit))
    projection_path = _write(root / "policy_canary_result_projection.json", projection)
    if include_readback and readback is None:
        readback = _readback(projection)
    elif not include_readback:
        readback = None
    sync_path = _write(root / "policy_canary_webapp_sync.json", readback) if include_readback else None
    closure_path = (_write(root / "provider_zero_closure.json", _closure(status=closure_status))
                    if include_closure else None)
    if include_binder:
        _write(root / "policy_canary_dispatch_receipt.json",
               _binder(projection, readback, projection_path=projection_path, sync_path=sync_path,
                       closure_path=closure_path))
    if include_publication:
        _write(root / "terminal_result_publication.json", _publication(projection))
    return projection


# --------------------------------------------------------------------- replay success

def test_terminal_completed_result_updates_owner_status_to_completed(tmp_path):
    env = _env(tmp_path)
    projection = _receipts(env)
    result = reconciler.reconcile_terminal_owner_result(
        intent=env["intent"], config=env["config"], release=env["release"], now=env["now"], output=env["output"])
    assert result is not None and result["terminal"] is True
    assert result["status"] == "completed", result
    assert result["result_reference"]["digest"] == projection["projection_digest"]
    assert result["result_reference"]["uri"].startswith("s3://")
    assert result["result_reference"]["size_bytes"] > 0
    # The claim ceiling stays a development-only diagnostic; no ranking upgrade.
    join = json.loads(Path(result["state"]["terminal_join"]["path"]).read_text())
    assert join["claim_ceiling"] == "diagnostic_policy_execution"
    assert join["result_status"] == "completed_unqualified"
    assert join["provider_mutation_performed"] is False


def test_completed_status_reads_back_through_authenticated_website_projection(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_progression_state import advance
    env = _env(tmp_path)
    projection = _receipts(env)
    result = reconciler.reconcile_terminal_owner_result(
        intent=env["intent"], config=env["config"], release=env["release"], now=env["now"], output=env["output"])
    # Persist the terminal owner status exactly as the progression worker does.
    advance(env["directory"], env["intent"], None, status=result["status"], phase=result["phase"],
            state=result["state"], blockers=result.get("blockers", ()),
            result_reference=result["result_reference"], now=env["now"])
    status = intake.scene_intent_status(queue_root=env["config"]["intent_root"], intent_id=env["intent_id"],
                                        now=env["now"])
    assert status["status"] == "completed"
    # The authenticated Website readback authenticates (status is digest-sealed
    # exactly as the intake seals it) and matches the same intent/result digest.
    assert status["status_digest"] == cross_runtime_canonical_digest(status, digest_field="status_digest")
    assert status["intent_digest"] == env["intent"]["intent_digest"]
    assert status["result_reference"]["digest"] == projection["projection_digest"]
    readback = json.loads(
        (Path(env["config"]["terminal_result_root"]) / env["intent_id"] / "policy_canary_webapp_sync.json").read_text())
    assert readback["policy_canary_projection_digest"] == projection["projection_digest"]
    assert readback["notification_delivery"]["run_result_digest"] == projection["projection_digest"]


# --------------------------------------------------------------------- failed child / blocked

def test_blocked_result_stays_blocked_and_preserves_failed_children(tmp_path):
    env = _env(tmp_path)
    _receipts(env, projection=_blocked_projection())
    result = reconciler.reconcile_terminal_owner_result(
        intent=env["intent"], config=env["config"], release=env["release"], now=env["now"], output=env["output"])
    assert result["terminal"] is True and result["status"] == "blocked", result
    assert result["result_reference"] is None
    assert "provider_capacity_unavailable" in result["blockers"]
    children = result["state"]["terminal_failed_children"]
    assert children and children[0]["typed_media_gap"] == "provider_runtime_failed_before_first_observation"
    assert children[0]["failure_taxonomy"] == "RuntimeError"


# --------------------------------------------------------------------- stale receipt / changed release

def test_receipt_for_a_different_input_digest_is_not_adopted(tmp_path):
    env = _env(tmp_path)
    stale = scene_policy.seal_binding(
        scene_intent_digest=env["intent"]["intent_digest"], attempt_id="policy-" + ("9" * 48),
        policy_candidates=env["intent"]["request"]["execution"]["policy_candidates"],
        runtime_digest=RUNTIME_DIGEST, input_digest="sha256:" + "0" * 64)
    _receipts(env, profile_binding=stale)
    result = reconciler.reconcile_terminal_owner_result(
        intent=env["intent"], config=env["config"], release=env["release"], now=env["now"], output=env["output"])
    assert result is None


def test_historical_attempt_reconciles_after_a_later_deploy(tmp_path):
    # A8: a legitimately-authorized historical attempt (execution identity all at
    # COMMIT: attempt, launch profile and launch request) must still close out
    # read-only after a later deploy moved the CURRENT release to a new commit.
    # Terminal reconciliation resolves the run's OWN immutable execution identity;
    # it is never gated on the current release commit.
    env = _env(tmp_path)  # attempts + receipts all at COMMIT
    _receipts(env, commit=COMMIT)
    deployed = dict(env["release"], source_commit=OTHER_COMMIT)  # a later deploy moved the release
    result = reconciler.reconcile_terminal_owner_result(
        intent=env["intent"], config=env["config"], release=deployed, now=env["now"], output=env["output"])
    assert result is not None and result["terminal"] is True and result["status"] == "completed"


def test_profile_from_a_different_commit_than_the_attempt_is_not_adopted(tmp_path):
    # A7: the launch profile + request name a DIFFERENT commit than the reserved
    # attempt. Even though profile<->request are internally consistent, the two
    # pairs must be JOINED to the attempt's execution commit; a profile from
    # another commit is never joined to this owner attempt (a wrong completion).
    env = _env(tmp_path)  # attempt reserved at COMMIT
    _receipts(env, commit=OTHER_COMMIT)  # launch profile + request re-sealed at OTHER_COMMIT
    result = reconciler.reconcile_terminal_owner_result(
        intent=env["intent"], config=env["config"], release=env["release"], now=env["now"], output=env["output"])
    assert result is None


def test_receipt_bound_to_a_different_intent_is_ignored(tmp_path):
    env = _env(tmp_path)
    foreign = scene_policy.seal_binding(
        scene_intent_digest="sha256:" + "4" * 64, attempt_id="policy-" + ("1" * 48),
        policy_candidates=env["intent"]["request"]["execution"]["policy_candidates"],
        runtime_digest=RUNTIME_DIGEST, input_digest=INPUT_DIGEST)
    _receipts(env, profile_binding=foreign)
    result = reconciler.reconcile_terminal_owner_result(
        intent=env["intent"], config=env["config"], release=env["release"], now=env["now"], output=env["output"])
    assert result is None


# --------------------------------------------------------------------- absent closure / readback

def test_absent_resource_closure_stays_explicit_and_never_completes(tmp_path):
    env = _env(tmp_path)
    _receipts(env, include_closure=False)
    result = reconciler.reconcile_terminal_owner_result(
        intent=env["intent"], config=env["config"], release=env["release"], now=env["now"], output=env["output"])
    assert result["terminal"] is False and result["status"] != "completed"
    assert "terminal_resource_closure_pending" in result["blockers"]


def test_ambiguous_or_unconfirmed_closure_stays_explicit(tmp_path):
    env = _env(tmp_path)
    _receipts(env, closure_status="provider_zero_pending")
    result = reconciler.reconcile_terminal_owner_result(
        intent=env["intent"], config=env["config"], release=env["release"], now=env["now"], output=env["output"])
    assert result["terminal"] is False and result["status"] != "completed"
    assert "terminal_resource_closure_pending" in result["blockers"]


def test_absent_authenticated_readback_stays_explicit(tmp_path):
    env = _env(tmp_path)
    _receipts(env, include_readback=False)
    result = reconciler.reconcile_terminal_owner_result(
        intent=env["intent"], config=env["config"], release=env["release"], now=env["now"], output=env["output"])
    assert result["terminal"] is False and result["status"] != "completed"
    assert "terminal_website_readback_pending" in result["blockers"]


def test_readback_digest_mismatch_never_completes(tmp_path):
    env = _env(tmp_path)
    projection = _completed_projection()
    _receipts(env, projection=projection)
    root = Path(env["config"]["terminal_result_root"]) / env["intent_id"]
    tampered = _readback(projection)
    tampered["policy_canary_projection_digest"] = "sha256:" + "0" * 64
    _write(root / "policy_canary_webapp_sync.json", tampered)
    result = reconciler.reconcile_terminal_owner_result(
        intent=env["intent"], config=env["config"], release=env["release"], now=env["now"], output=env["output"])
    assert result["status"] != "completed"


# --------------------------------------------------------------------- publication integrity / notification

def _seal_pub(value: dict) -> dict:
    value = {k: v for k, v in value.items() if k != "publication_digest"}
    value["publication_digest"] = canonical_digest(value, digest_field="publication_digest")
    return value


def test_publication_wrong_schema_run_id_or_unsealed_stays_publication_pending(tmp_path):
    # A7: a completed-unqualified result completes only with a durable, sealed
    # publication for THIS run. A wrong schema, an unrelated run_id, an omitted
    # provider_allocated flag, an unsealed record, or a tampered byte each leaves
    # the result publication pending, never silently completed.
    cases = {
        "wrong_schema": lambda p: _seal_pub({**p, "schema_version": "something_else.v1"}),
        "wrong_run_id": lambda p: _seal_pub({**p, "run_id": "run-not-this-one"}),
        "provider_allocated_missing": lambda p: _seal_pub({k: v for k, v in p.items() if k != "provider_allocated"}),
        "unsealed": lambda p: {k: v for k, v in p.items() if k != "publication_digest"},
        "tampered_after_seal": lambda p: {**p, "size_bytes": 999999},
    }
    for name, mutate in cases.items():
        sub = tmp_path / name
        sub.mkdir()
        env = _env(sub)
        projection = _completed_projection()
        _receipts(env, projection=projection)
        root = Path(env["config"]["terminal_result_root"]) / env["intent_id"]
        _write(root / "terminal_result_publication.json", mutate(_publication(projection)))
        result = reconciler.reconcile_terminal_owner_result(
            intent=env["intent"], config=env["config"], release=env["release"], now=env["now"], output=env["output"])
        assert result["status"] != "completed", name
        assert "terminal_result_publication_pending" in result["blockers"], name


def test_failed_notification_after_durable_readback_still_completes(tmp_path):
    # A8: the durable Website readback (status succeeded, digests bound) gates
    # terminal completion. A FAILED push notification after a successful durable
    # readback must NOT strand the run; notification delivery is reported
    # separately in the terminal state, never a completion gate.
    env = _env(tmp_path)
    projection = _completed_projection()
    readback = _readback(projection)
    readback["notification_delivery"]["status"] = "failed"
    # The dispatcher's sync result carried the failed notification; its sealed
    # receipt records exactly that readback (binder-bound), as in production.
    _receipts(env, projection=projection, readback=readback)
    result = reconciler.reconcile_terminal_owner_result(
        intent=env["intent"], config=env["config"], release=env["release"], now=env["now"], output=env["output"])
    assert result["terminal"] is True and result["status"] == "completed"
    assert result["state"]["terminal_notification_delivery"]["status"] == "failed"


# --------------------------------------------------------------------- idempotency

def test_duplicate_tick_is_byte_identical(tmp_path):
    env = _env(tmp_path)
    _receipts(env)
    first = reconciler.reconcile_terminal_owner_result(
        intent=env["intent"], config=env["config"], release=env["release"], now=env["now"], output=env["output"])
    second = reconciler.reconcile_terminal_owner_result(
        intent=env["intent"], config=env["config"], release=env["release"], now=env["now"] + 5, output=env["output"])
    assert first == second
    join_path = Path(first["state"]["terminal_join"]["path"])
    assert json.loads(join_path.read_text())["provider_mutation_performed"] is False


def test_no_projection_yet_returns_none(tmp_path):
    env = _env(tmp_path)
    # Only the bridge exists; the terminal policy result has not been retained.
    root = Path(env["config"]["terminal_result_root"]) / env["intent_id"]
    profile = _profile(env["binding"])
    _write(root / "launch_profile.json", profile)
    _write(root / "launch_request.json", _launch_request(profile["profile_digest"]))
    result = reconciler.reconcile_terminal_owner_result(
        intent=env["intent"], config=env["config"], release=env["release"], now=env["now"], output=env["output"])
    assert result is None


def test_unconfigured_terminal_root_returns_none(tmp_path):
    env = _env(tmp_path)
    result = reconciler.reconcile_terminal_owner_result(
        intent=env["intent"], config={"intent_root": env["config"]["intent_root"]},
        release=env["release"], now=env["now"], output=env["output"])
    assert result is None


# --------------------------------------------------------------------- _advance_intent hook

def test_advance_intent_completes_owner_status_from_terminal_receipts(tmp_path):
    """The progression tail joins terminal receipts once activation exists.

    A progression already at ``awaiting_execution`` with an activation record is
    advanced to ``completed`` on the next tick when the terminal receipts are
    retained -- without re-running the factory chain.
    """
    from blueprint_pipeline import task_evaluation_scene_progression as engine
    from blueprint_pipeline import task_evaluation_scene_progression_state as state
    env = _env(tmp_path)
    projection = _receipts(env)
    # A completed activation, exactly as the normal tail leaves it.
    state.advance(env["directory"], env["intent"], None, status="awaiting_execution",
                  phase="scene_configuration", state={"activation": {"provider_allocation_performed": False},
                                                       "attempt_id": "source-1"}, now=env["now"])
    config = dict(env["config"], factory_output_root=str(tmp_path / "factory-output"))
    progress = engine._advance_intent(env["directory"], env["intent"], config, env["release"],
                                      resolver=None, publisher=None, submitter=None, status_reader=None,
                                      activation_provisioner=None, now=env["now"])
    assert progress["status"] == "completed", progress
    status = intake.scene_intent_status(queue_root=env["config"]["intent_root"], intent_id=env["intent_id"],
                                        now=env["now"])
    assert status["status"] == "completed"
    assert status["result_reference"]["digest"] == projection["projection_digest"]


def test_advance_intent_without_activation_does_not_run_terminal_join(tmp_path):
    from blueprint_pipeline import task_evaluation_scene_progression as engine
    from blueprint_pipeline import task_evaluation_scene_progression_state as state
    env = _env(tmp_path)
    _receipts(env)
    # No activation recorded yet -> the tail must not short-circuit to a join.
    state.advance(env["directory"], env["intent"], None, status="awaiting_execution",
                  phase="publication_ready", state={"attempt_id": "source-1"}, now=env["now"])
    config = dict(env["config"], factory_output_root=str(tmp_path / "factory-output"),
                  submission_enabled=False, activation_enabled=False, machinery_path="/nonexistent",
                  release_binding_path="/nonexistent", public_source_binding_root="/nonexistent",
                  trusted_clients=["blueprint-webapp"], supported_source_kinds=["capture_bundle"])
    progress = engine._advance_intent(env["directory"], env["intent"], config, env["release"],
                                      resolver=None, publisher=None, submitter=None, status_reader=None,
                                      activation_provisioner=None, now=env["now"])
    assert progress["status"] != "completed"


def test_advance_intent_completion_is_idempotent_across_restart(tmp_path):
    """A completed owner status is byte-identical on the next worker tick."""
    from blueprint_pipeline import task_evaluation_scene_progression as engine
    from blueprint_pipeline import task_evaluation_scene_progression_state as state
    env = _env(tmp_path)
    _receipts(env)
    state.advance(env["directory"], env["intent"], None, status="awaiting_execution",
                  phase="scene_configuration", state={"activation": {"provider_allocation_performed": False},
                                                       "attempt_id": "source-1"}, now=env["now"])
    config = dict(env["config"], factory_output_root=str(tmp_path / "factory-output"))
    first = engine._advance_intent(env["directory"], env["intent"], config, env["release"],
                                   resolver=None, publisher=None, submitter=None, status_reader=None,
                                   activation_provisioner=None, now=env["now"])
    assert first["status"] == "completed"
    # A restart re-reads the retained events and receipts and changes nothing.
    second = engine._advance_intent(env["directory"], env["intent"], config, env["release"],
                                    resolver=None, publisher=None, submitter=None, status_reader=None,
                                    activation_provisioner=None, now=env["now"] + 30)
    assert second == first


def test_advance_intent_closes_out_completed_run_after_authority_expiry(tmp_path):
    # A8: a run authorized when it executed must still close out read-only after
    # its authority window lapses. The terminal reconciliation hook runs BEFORE
    # the expiry gate, so an expired-but-completed run reconciles to completed
    # rather than being stranded as blocked-authority.
    from blueprint_pipeline import task_evaluation_scene_progression as engine
    from blueprint_pipeline import task_evaluation_scene_progression_state as state
    env = _env(tmp_path)
    _receipts(env)
    state.advance(env["directory"], env["intent"], None, status="awaiting_execution",
                  phase="scene_configuration", state={"activation": {"provider_allocation_performed": False},
                                                      "attempt_id": "source-1"}, now=env["now"])
    config = dict(env["config"], factory_output_root=str(tmp_path / "factory-output"))
    expired = env["intent"]["request"]["execution"]["expires_at_epoch"] + 1  # authority window lapsed
    progress = engine._advance_intent(env["directory"], env["intent"], config, env["release"],
                                      resolver=None, publisher=None, submitter=None, status_reader=None,
                                      activation_provisioner=None, now=expired)
    assert progress["status"] == "completed", progress


def test_advance_intent_expiry_without_terminal_receipts_still_blocks(tmp_path):
    # The hook-move must not swallow the authority gate: an expired intent with no
    # owner-bound terminal result (reconciler returns None) still falls through to
    # blocked-authority.
    from blueprint_pipeline import task_evaluation_scene_progression as engine
    from blueprint_pipeline import task_evaluation_scene_progression_state as state
    env = _env(tmp_path)  # no _receipts(): nothing to reconcile
    state.advance(env["directory"], env["intent"], None, status="awaiting_execution",
                  phase="scene_configuration", state={"activation": {"provider_allocation_performed": False},
                                                      "attempt_id": "source-1"}, now=env["now"])
    config = dict(env["config"], factory_output_root=str(tmp_path / "factory-output"))
    expired = env["intent"]["request"]["execution"]["expires_at_epoch"] + 1
    progress = engine._advance_intent(env["directory"], env["intent"], config, env["release"],
                                      resolver=None, publisher=None, submitter=None, status_reader=None,
                                      activation_provisioner=None, now=expired)
    assert progress["status"] == "blocked"
    assert "scene_intake_authority_expired" in progress["blockers"]
