"""A website job's proposal_only thresholds compile as the owner's only through the owner's sealed confirmation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import task_evaluation_scene_rigid_robot_free_readiness as rigid
from blueprint_pipeline import team_scene_packet_worker as worker
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.team_scene_packet import TeamScenePacketError, materialize_team_scene, read_team_scene_inputs
from tests.test_task_evaluation_owner_success_confirmation import (
    OWNER,
    confirm,
    proposal_only_case,
    publish,
    rethreshold,
)
from tests.test_task_evaluation_rigid_relocation_native_adapter import SUCCESS
from tests.test_team_scene_packet import _scene, _sha

REFUSED = "team_scene_owner_contract_invalid:configured_owner_success_contract_authority_missing"


def _job(tmp_path: Path, monkeypatch) -> tuple[Any, dict[str, Any]]:
    """The published proposal_only job, with the bundle, DROID profile and runtime the team compile reads."""

    base = _scene(tmp_path)
    configured, references = proposal_only_case(tmp_path / "owner-case")
    bundle = Path(base["configured_scene_bundle_path"])
    configured["configured_scene_bundle"] = {"uri": "s3://scenes/configured-scene.zip",
                                             "digest": _sha(bundle.read_bytes()), "size_bytes": bundle.stat().st_size}
    configured["revision_digest"] = canonical_digest(configured, digest_field="revision_digest")
    job = publish(tmp_path, monkeypatch, configured=configured, references=references)
    inputs = {key: base[key] for key in ("configured_scene_bundle_path", "droid_profile_path",
                                         "droid_profile_digest", "runtime_source_receipt_path")}
    return job, inputs


def _compile(job: Any, inputs: dict[str, Any], root: Path, *, revision: dict[str, Any] | None = None,
             scope: bool = True) -> dict[str, Any]:
    revision = revision or job.revision
    documents = rigid.fetch_revision_documents(revision=revision, output_root=root / "work",
                                               fetcher=lambda reference: job.store[reference["uri"]])
    return materialize_team_scene(
        revision=revision, documents=documents, output_root=root / "team-scene", **inputs,
        owner_success_confirmation_scope={"intent": job.intent, "directory": job.directory} if scope else None)


def test_a_confirmed_proposal_compiles_the_owner_contract_as_the_owners(tmp_path, monkeypatch) -> None:
    from blueprint_pipeline.task_evaluation_policy_canary_scene_setup import _require_strict_owner_success_contract

    job, inputs = _job(tmp_path, monkeypatch)
    applied = confirm(job)
    compiled = _compile(job, inputs, tmp_path / "confirmed")
    assert compiled["owner_success_confirmation"] == applied["confirmation_digest"]
    half = read_team_scene_inputs(compiled["path"])
    contract = half["task_success_contract"]
    assert contract["provenance"]["author_source"] == "task_owner"
    assert contract["provenance"]["confirmation_status"] == "confirmed"
    assert contract["provenance"]["confirmed_by_team_id"] == OWNER["user_id"]
    assert contract["provenance"]["author_id"] == "owner-success-confirmation:" + applied["confirmation_digest"]
    # The thresholds are the ones the job was built with; the proposal is kept beside the confirmation.
    packet = Path(compiled["path"]).parent / "native-task-packet"
    task_spec = json.loads((packet / "native_task_arena_packet_request.v1.json").read_text())["task_spec"]
    success = json.loads(Path(job.references[SUCCESS]["materialized_path"]).read_text())
    assert task_spec["configured_success_criteria"] == {key: value for key, value in success.items()
                                                        if key not in {"schema_version", "status"}}
    assert contract["criteria"]["temporal_invariants"]["maximum_task_contact_force_n"] == (
        success["maximum_task_contact_force_n"])
    authority = task_spec["configured_owner_authority"]
    assert authority["confirmation_status"] == "confirmed"
    assert authority["proposal_authority"]["confirmation_status"] == "proposal_only"
    # Screening reads this sealed contract and admits it as a confirmed, strict owner contract.
    plan = json.loads((packet / "native_task_arena_scene_plan.v1.json").read_text())
    assert plan["task_spec"]["task_success_contract"] == contract
    _require_strict_owner_success_contract(task_spec=plan["task_spec"], contract=contract)


def test_without_a_confirmation_it_refuses_as_before(tmp_path, monkeypatch) -> None:
    job, inputs = _job(tmp_path, monkeypatch)
    for name, scope in (("unscoped", False), ("scoped", True)):
        with pytest.raises(TeamScenePacketError, match=f"\\A{REFUSED}\\Z"):
            _compile(job, inputs, tmp_path / name, scope=scope)


def test_a_confirmation_of_other_thresholds_is_refused(tmp_path, monkeypatch) -> None:
    job, inputs = _job(tmp_path, monkeypatch)
    confirm(job)
    # The same intent republished with one threshold changed: the owner confirmed the old one, not this.
    configured, references = rethreshold(job.revision, job.references, tmp_path / "owner-case",
                                         maximum_task_contact_force_n=50.0)
    job.store.update({row["uri"]: Path(row["materialized_path"]).read_bytes() for row in references.values()})
    with pytest.raises(TeamScenePacketError, match=f"\\A{REFUSED}\\Z"):
        _compile(job, inputs, tmp_path / "changed", revision=configured)
    # Another owner's intent holds nothing for this job either.
    other = {**job.intent, "request": {**job.intent["request"], "owner": {"user_id": "u2", "organization_id": "o2"}}}
    documents = rigid.fetch_revision_documents(revision=job.revision, output_root=tmp_path / "other" / "work",
                                               fetcher=lambda reference: job.store[reference["uri"]])
    with pytest.raises(TeamScenePacketError, match=f"\\A{REFUSED}\\Z"):
        materialize_team_scene(revision=job.revision, documents=documents, output_root=tmp_path / "other" / "half",
                               **inputs, owner_success_confirmation_scope={"intent": other,
                                                                           "directory": job.directory})


def _request(job: Any, **changes: Any) -> dict[str, Any]:
    value = {"schema_version": worker.REQUEST_SCHEMA, "intent_id": job.intent["intent_id"],
             "intent_digest": job.intent["intent_digest"], "capture_id": "walkthrough-1",
             "task_id": "website-" + "0" * 20, "source_launch_root": str(job.run),
             "launch_receipt_digest": "sha256:" + "1" * 64, "scene_revision_digest": job.revision["revision_digest"],
             "robot_free_readiness_status": "refused", **changes}
    return worker.validated_request(worker._seal(value, "request_digest"))


def test_the_worker_reads_only_the_requests_own_intent(tmp_path, monkeypatch) -> None:
    from blueprint_pipeline import task_evaluation_scene_intake as intake

    staged = publish(tmp_path, monkeypatch)
    request = _request(staged)
    # Bind the progression to this request, as the completion join records it.
    progression = json.loads((staged.directory / "progression.json").read_text())
    progression["state"]["scene_preparation_completion"]["team_scene_request"]["request_digest"] = (
        request["request_digest"])
    (staged.directory / "progression.json").write_text(json.dumps(intake._seal(progression, "progression_digest")))
    scope, reason = worker.owner_success_confirmation_scope(request)
    assert reason is None and scope == {"intent": staged.intent, "directory": staged.directory}
    # Another request (here: another launch) for the same intent is not the one its progression queued.
    assert worker.owner_success_confirmation_scope(_request(staged, source_launch_root=str(tmp_path))) == (
        None, "intent_binding_mismatch")
    assert worker.owner_success_confirmation_scope(_request(staged, intent_digest="sha256:" + "e" * 64)) == (
        None, "intent_binding_mismatch")
    monkeypatch.delenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT")
    assert worker.owner_success_confirmation_scope(request) == (None, "scene_intake_root_not_configured")
    assert worker.owner_success_confirmation_scope(request, intent_root=tmp_path / "nowhere") == (
        None, "intent_unreadable")


def test_an_owner_contract_refusal_returns_to_pending_once_per_new_confirmation(tmp_path, monkeypatch) -> None:
    from blueprint_pipeline import task_evaluation_scene_intake as intake

    job = publish(tmp_path, monkeypatch)
    # No known running release: only the owner's confirmation can return this refusal.
    monkeypatch.setattr(worker, "current_release_commit", lambda: None)
    request = _request(job)
    progression = json.loads((job.directory / "progression.json").read_text())
    progression["state"]["scene_preparation_completion"]["team_scene_request"]["request_digest"] = (
        request["request_digest"])
    (job.directory / "progression.json").write_text(json.dumps(intake._seal(progression, "progression_digest")))
    queue = worker._queue(tmp_path / "queue")
    key = worker.request_key(request)
    (queue / "processed" / f"{key}.json").write_text(json.dumps(request))
    blocked = {"schema_version": worker.STATUS_SCHEMA, "request_key": key, "status": "blocked",
               "blockers": [REFUSED]}
    (queue / "status" / f"{key}.json").write_text(json.dumps(blocked))
    # Nothing confirmed yet: a deterministic refusal stays blocked.
    assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == []
    applied = confirm(job)
    requeued = worker.requeue_transient_blocked(queue_root=queue)["requeued"]
    assert requeued == [{"request_key": key, "reason": "owner_success_confirmation",
                         "confirmation_digests": [applied["confirmation_digest"]]}]
    assert (queue / "pending" / f"{key}.json").is_file()
    status = json.loads((queue / "status" / f"{key}.json").read_text())
    assert status[worker.OWNER_CONFIRMATION_REQUEUE_FIELD] == [applied["confirmation_digest"]]
    # Refused again with the same confirmation: it is never retried for that confirmation again.
    (queue / "pending" / f"{key}.json").rename(queue / "processed" / f"{key}.json")
    assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == []
    # A newly deployed release never returns an owner-contract refusal: only a new confirmation does.
    monkeypatch.setattr(worker, "current_release_commit", lambda: "a" * 40)
    assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == []
    # Without a known release, any other deterministic refusal never moves.
    monkeypatch.setattr(worker, "current_release_commit", lambda: None)
    (queue / "status" / f"{key}.json").write_text(json.dumps({**blocked, "blockers": ["team_scene_task_invalid"]}))
    assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == []
