"""The owner's confirmation of a website job's built thresholds: preview, then apply with the preview's digest."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from blueprint_pipeline import task_evaluation_owner_success_confirmation as confirmation
from blueprint_pipeline import task_evaluation_owner_success_confirmation_door as door
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_configured_controls_deferred_inputs import _document_name
from blueprint_pipeline.task_evaluation_scene_intent_contracts import ROOT_ENV
from tests.test_task_evaluation_rigid_relocation_native_adapter import (
    DEFINITION,
    EXECUTION,
    SOURCE_OBJECT,
    STATIC,
    SUCCESS,
    SUPPORT,
    _case,
    _rewrite,
)
from tests.test_task_evaluation_scene_intake import request, stage

OWNER = request()["owner"]
EVIDENCE = "owner accepted the default thresholds in session X, 2026-11-03"


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def proposal_only_case(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """A website pick-and-place job as published: owner contract required, Blueprint's thresholds proposal_only.

    Built with the website's own record builders, as ``website_native_submission`` builds it.
    """

    from blueprint_pipeline.task_evaluation_scene_configuration_submission_records import pick_and_place_task_records
    from blueprint_pipeline.task_evaluation_surface_target import derive_surface_target, surface_execution_limits

    root.mkdir(parents=True)
    _launch, configured, references, docs = _case(root)
    start = docs[DEFINITION]["start_center_xyz_m"]
    source, support = docs[SOURCE_OBJECT], docs[SUPPORT]
    target = derive_surface_target(destination={"kind": "green_region", "relation": "on", "visible_label": "green spot",
        "position_world_m": [start[0] + 0.3, start[1], support["top_z_m"]], "orientation_xyzw": [0, 0, 0, 1],
        "radius_m": 0.15}, support=support, source_min=source["aabb_min_xyz_m"], source_max=source["aabb_max_xyz_m"],
        support_instance_id="support")
    success = surface_execution_limits(success={"control_frequency_hz": 20, "maximum_episode_seconds": 12,
        "minimum_lift_m": 0.05, "pregrasp_clearance_m": 0.1, "minimum_planar_displacement_m": 0.1,
        "maximum_final_planar_target_error_m": 0.05, "maximum_retries": 0, "maximum_regrasps": 0}, support=support)
    definition, criteria, execution = pick_and_place_task_records(
        task_identity=configured["task_template"]["identity"], object_identity=configured["replacement"]["identity"],
        start_center=start, target_center=[start[0] + 0.3, start[1], start[2]], source_min=source["aabb_min_xyz_m"],
        source_max=source["aabb_max_xyz_m"], grasp_axis=2, grasp_sign=1.0, jaw_axis=1, success=success,
        resolved_seed=839873104)
    definition["surface_target"] = definition["success"]["surface_target"] = criteria["surface_target"] = target
    definition["owner_success_contract_authority"] = {
        "confirmation_status": "proposal_only", "accepted_by": OWNER["user_id"],
        "authority_reference": "scene-intent:sha256:" + "f" * 64,
        "basis": "fixed_development_control_not_translated_owner_target", "scorer_translation_verified": False,
        "owner_success_criteria": {"basis": "owner_stated_target", "status": "not_supplied", "targets": None}}
    static = copy.deepcopy(docs[STATIC])
    center = static["observed_structure"]["center_of_mass_m"]
    static["observed_structure"]["collision_bounds_body_frame_m"] = {
        "minimum": [source["aabb_min_xyz_m"][i] - start[i] + center[i] for i in range(3)],
        "maximum": [source["aabb_max_xyz_m"][i] - start[i] + center[i] for i in range(3)]}
    static["result_digest"] = canonical_digest(static, digest_field="result_digest")
    for key, document in ((DEFINITION, definition), (SUCCESS, criteria), (EXECUTION, execution), (STATIC, static)):
        _rewrite(tmp_path=root, configured=configured, references=references, contract_path=key, document=document)
    configured["task_template"]["surface_target"] = target
    configured["revision_digest"] = canonical_digest(configured, digest_field="revision_digest")
    return configured, references


def rethreshold(configured: dict[str, Any], references: dict[str, Any], root: Path,
                **values: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """The same job republished with other thresholds (its template and success document agree)."""

    configured, references = copy.deepcopy(configured), copy.deepcopy(references)
    for key in (DEFINITION, SUCCESS):
        document = json.loads(Path(references[key]["materialized_path"]).read_text())
        (document["success"] if key == DEFINITION else document).update(values)
        _rewrite(tmp_path=root, configured=configured, references=references, contract_path=key, document=document)
    return configured, references


def publish(tmp_path: Path, monkeypatch, *, configured: dict[str, Any] | None = None,
            references: dict[str, Any] | None = None, request_digest: str = "sha256:" + "a" * 64) -> SimpleNamespace:
    """A staged intent whose progression completed with a published revision, as the host retains it."""

    from blueprint_pipeline import task_evaluation_configured_controls_progression_worker as source

    if configured is None:
        configured, references = proposal_only_case(tmp_path / "case")
    assert references is not None
    intents = tmp_path / "intents"
    intents.mkdir()
    record = stage(intents)
    directory = intents / record["intent_id"]
    intent = intake._read(directory / "intent.json", "intent_digest")
    store = {row["uri"]: Path(row["materialized_path"]).read_bytes() for row in references.values()}
    run = tmp_path / "launch"
    documents = run / "robot-free-readiness" / "documents"
    documents.mkdir(parents=True)
    for contract_path, row in references.items():
        (documents / _document_name(contract_path)).write_bytes(store[row["uri"]])
    payload = json.dumps(configured, sort_keys=True).encode()
    revision_path = run / "configured_scene_revision.v1.json"
    revision_path.write_bytes(payload)
    receipt = run / "launch_receipt.json"
    receipt.write_text("{}")
    progression = intake._seal({
        "schema_version": "task_evaluation_scene_progression.v1", "intent_id": intent["intent_id"],
        "intent_digest": intent["intent_digest"], "status": "completed", "phase": "scene_prepared",
        "result_reference": {"uri": "s3://scenes/revision.json", "digest": _sha(payload), "size_bytes": len(payload)},
        "state": {"scene_preparation_completion": {
            "launch_receipt": {"path": str(receipt), "sha256": _sha(receipt.read_bytes()), "size_bytes": 2},
            "team_scene_request": {"status": "enqueued", "request_key": "0" * 32, "request_digest": request_digest}}},
        "blockers": []}, "progression_digest")
    (directory / "progression.json").write_text(json.dumps(progression))
    monkeypatch.setattr(source, "_validate_source", lambda root: ({
        "configured_scene_revision_path": str(revision_path),
        "configured_scene_revision_digest": configured["revision_digest"]}, {}, {}) if Path(root) == run else (
        _ for _ in ()).throw(RuntimeError("configured_controls_worker_launch_receipt_invalid")))
    monkeypatch.setenv(ROOT_ENV, str(intents))
    return SimpleNamespace(intents=intents, directory=directory, intent=intent, revision=configured,
                           references=references, store=store, run=run, documents=documents)


def body(job: SimpleNamespace, **changes: Any) -> dict[str, Any]:
    value = {"kind": "confirm-owner-success-contract", "intent_id": job.intent["intent_id"],
             "owner_user_id": OWNER["user_id"], "owner_organization_id": OWNER["organization_id"],
             "approved_by": OWNER["user_id"], "approval_evidence": EVIDENCE, "apply": False, "apply_ack": "",
             "ack": "confirm-owner-success-contract"}
    value.update(changes)
    return value


def confirm(job: SimpleNamespace, request_id: str = "req-apply") -> dict[str, Any]:
    preview = door.run(body(job), door_caller="cloud-scene-runs", door_request_id="req-preview")
    return door.run(body(job, apply=True, apply_ack=preview["apply_ack"]), door_caller="cloud-scene-runs",
                    door_request_id=request_id)


def compiled_task_spec(job: SimpleNamespace, revision: dict[str, Any] | None = None) -> dict[str, Any]:
    """The task the team packet compiles from the published revision (its rigid branch)."""

    from blueprint_pipeline import task_evaluation_configured_controls_deferred_inputs as inputs
    from blueprint_pipeline import task_evaluation_native_arena_episode_compiler as compiler
    from blueprint_pipeline import task_evaluation_scene_rigid_robot_free_readiness as rigid
    from blueprint_pipeline.task_evaluation_rigid_relocation_native_adapter import adapt_rigid_relocation_task_template

    revision = revision or job.revision
    documents = rigid.fetch_revision_documents(revision=revision, output_root=job.run / "compile",
                                               fetcher=lambda reference: job.store[reference["uri"]])
    adapter = adapt_rigid_relocation_task_template(
        configured_revision=revision, materialized_references=inputs._materialized_references(
            revision=revision, documents=documents))
    task_spec = compiler._runtime_subject_task_spec(adapter["native_task_definition"]["task_spec"])
    task_spec["success_criteria"] = adapter["native_success_criteria"].get("criteria")
    return task_spec


def test_preview_writes_nothing_and_apply_with_its_digest_records_one_confirmation(tmp_path, monkeypatch):
    job = publish(tmp_path, monkeypatch)
    success = json.loads((job.documents / _document_name(SUCCESS)).read_text())
    preview = door.run(body(job), door_caller="cloud-scene-runs", door_request_id="req-1")
    assert preview["door_outcome"] == "planned" and preview["existing_confirmation_digest"] is None
    shown = preview["preview"]
    # The thresholds exactly as built, the revision and document they come from, and the proposal they carry.
    assert shown["configured_success_criteria"] == {key: value for key, value in success.items()
                                                    if key not in {"schema_version", "status"}}
    assert shown["configured_scene_revision_digest"] == job.revision["revision_digest"]
    assert shown["success_criteria_document"] == job.revision["task_template"]["success_criteria"]
    assert shown["proposal_authority"]["confirmation_status"] == "proposal_only"
    assert shown["owner"] == OWNER and shown["approval_evidence"] == EVIDENCE
    assert not (job.directory / confirmation.DIRECTORY).exists()
    with pytest.raises(ValueError, match="owner_success_confirmation_apply_ack_mismatch"):
        door.run(body(job, apply=True, apply_ack="sha256:" + "0" * 64), door_caller="cloud", door_request_id="r")
    assert not (job.directory / confirmation.DIRECTORY).exists()
    applied = door.run(body(job, apply=True, apply_ack=preview["apply_ack"]), door_caller="cloud-scene-runs",
                       door_request_id="req-3")
    assert applied["door_outcome"] == "confirmed" and applied["mode"] == "applied"
    [path] = (job.directory / confirmation.DIRECTORY).glob("*.json")
    row = json.loads(path.read_text())
    assert path.name == row["confirmation_digest"].removeprefix("sha256:") + ".json"
    assert row["schema_version"] == "task_evaluation_owner_success_contract_confirmation.v1"
    assert row["intent_digest"] == job.intent["intent_digest"] and row["owner"] == OWNER
    assert row["approved_by"] == OWNER["user_id"] and row["approval_evidence"] == EVIDENCE
    assert row["configured_success_criteria"] == shown["configured_success_criteria"]
    assert row["thresholds_changed"] is False and row["provider_mutation_performed"] is False
    assert row["confirmed_at"].endswith("Z")
    assert "channel=operator_door" in row["authorization_reference"]
    assert "door_request_id=req-3" in row["authorization_reference"]
    assert not path.stat().st_mode & 0o222  # sealed read-only
    # The retained documents are only read.
    assert json.loads((job.documents / _document_name(SUCCESS)).read_text()) == success


def test_reapplying_is_idempotent_and_never_overwrites(tmp_path, monkeypatch):
    job = publish(tmp_path, monkeypatch)
    first = confirm(job, "req-a")
    [path] = (job.directory / confirmation.DIRECTORY).glob("*.json")
    before = path.read_bytes()
    again = confirm(job, "req-b")
    assert again["door_outcome"] == "already_confirmed"
    assert again["confirmation_digest"] == first["confirmation_digest"]
    assert list((job.directory / confirmation.DIRECTORY).glob("*.json")) == [path] and path.read_bytes() == before
    preview = door.run(body(job), door_caller="cloud", door_request_id="req-c")
    assert preview["existing_confirmation_digest"] == first["confirmation_digest"]
    # A different approval of the same thresholds is already covered too: nothing more is written.
    other = door.run(body(job, approval_evidence="second approval"), door_caller="cloud", door_request_id="p")
    repeat = door.run(body(job, approval_evidence="second approval", apply=True, apply_ack=other["apply_ack"]),
                      door_caller="cloud", door_request_id="req-d")
    assert repeat["door_outcome"] == "already_confirmed" and path.read_bytes() == before


@pytest.mark.parametrize("changes", [
    {"owner_user_id": "someone-else", "approved_by": "someone-else"},
    {"owner_organization_id": "other-org"},
    {"approved_by": "someone-else"},
])
def test_an_approval_that_is_not_the_owners_is_refused(tmp_path, monkeypatch, changes):
    job = publish(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="owner_success_confirmation_owner_or_revocation_mismatch"):
        door.run(body(job, **changes), door_caller="cloud", door_request_id="r")
    assert not (job.directory / confirmation.DIRECTORY).exists()


def test_a_revoked_intent_cannot_be_confirmed_and_loses_its_confirmation(tmp_path, monkeypatch):
    job = publish(tmp_path, monkeypatch)
    confirm(job)
    task_spec = compiled_task_spec(job)
    scope = {"intent": job.intent, "directory": job.directory}
    assert confirmation.confirmed_owner_authority(task_spec=task_spec, revision=job.revision, **scope) is not None
    (job.directory / "revoked.json").write_text("{}")
    assert confirmation.confirmed_owner_authority(task_spec=task_spec, revision=job.revision, **scope) is None
    with pytest.raises(ValueError, match="owner_or_revocation_mismatch"):
        door.run(body(job), door_caller="cloud", door_request_id="r")


def test_the_confirmation_binds_the_exact_revision_thresholds_and_proposal(tmp_path, monkeypatch):
    job = publish(tmp_path, monkeypatch)
    applied = confirm(job)
    task_spec = compiled_task_spec(job)
    scope = {"intent": job.intent, "directory": job.directory}
    authority = confirmation.confirmed_owner_authority(task_spec=task_spec, revision=job.revision, **scope)
    assert authority == {
        "author_source": "task_owner", "confirmation_status": "confirmed", "accepted_by": OWNER["user_id"],
        "authority_reference": "owner-success-confirmation:" + applied["confirmation_digest"],
        "basis": "owner_confirmed_built_thresholds_via_operator_door",
        "confirmation_digest": applied["confirmation_digest"],
        "proposal_authority": task_spec["configured_owner_authority"]}

    def found(spec: dict[str, Any], revision: dict[str, Any] = job.revision, **values: Any) -> bool:
        return confirmation.confirmed_owner_authority(task_spec=spec, revision=revision,
                                                      **{**scope, **values}) is not None

    # A changed threshold (the same revision cannot carry one, but the binding never trusts that).
    changed = copy.deepcopy(task_spec)
    changed["configured_success_criteria"]["maximum_task_contact_force_n"] += 1.0
    assert not found(changed)
    # A different published revision: another revision digest, or another success document.
    assert not found(task_spec, {**job.revision, "revision_digest": "sha256:" + "b" * 64})
    moved = copy.deepcopy(job.revision)
    moved["task_template"]["success_criteria"]["digest"] = "sha256:" + "c" * 64
    assert not found(task_spec, moved)
    # Another proposal, another intent of the same owner, or an authority that is not a proposal.
    reproposed = copy.deepcopy(task_spec)
    reproposed["configured_owner_authority"]["basis"] = "another_proposal"
    assert not found(reproposed)
    assert not found(task_spec, intent={**job.intent, "intent_digest": "sha256:" + "d" * 64})
    confirmed = copy.deepcopy(task_spec)
    confirmed["configured_owner_authority"]["confirmation_status"] = "confirmed"
    assert not found(confirmed)
    # A record whose bytes changed is no confirmation at all.
    [path] = (job.directory / confirmation.DIRECTORY).glob("*.json")
    row = json.loads(path.read_text())
    path.chmod(0o644)
    path.write_text(json.dumps({**row, "configured_success_criteria": changed["configured_success_criteria"]}))
    assert not found(task_spec)


def test_a_changed_published_job_is_refused_before_any_write(tmp_path, monkeypatch):
    job = publish(tmp_path, monkeypatch)
    success = job.documents / _document_name(SUCCESS)
    original = success.read_bytes()
    success.write_bytes(original.replace(b"maximum_task_contact_force_n", b"maximum_task_contact_force_N"))
    with pytest.raises(ValueError, match="success_document_unavailable"):
        door.run(body(job), door_caller="cloud", door_request_id="r")
    success.write_bytes(original)
    revision = job.run / "configured_scene_revision.v1.json"
    revision.write_bytes(revision.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="revision_not_the_published_result"):
        door.run(body(job), door_caller="cloud", door_request_id="r")
    assert not (job.directory / confirmation.DIRECTORY).exists()


def test_a_preview_taken_before_the_thresholds_changed_cannot_be_applied(tmp_path, monkeypatch):
    job = publish(tmp_path, monkeypatch)
    preview = door.run(body(job), door_caller="cloud", door_request_id="p")
    # The job is republished with another threshold: the owner never saw these.
    configured, references = rethreshold(job.revision, job.references, tmp_path / "case",
                                         maximum_task_contact_force_n=50.0)
    payload = json.dumps(configured, sort_keys=True).encode()
    (job.run / "configured_scene_revision.v1.json").write_bytes(payload)
    for key in (DEFINITION, SUCCESS):
        (job.documents / _document_name(key)).write_bytes(Path(references[key]["materialized_path"]).read_bytes())
    progression = json.loads((job.directory / "progression.json").read_text())
    progression["result_reference"] = {**progression["result_reference"], "digest": _sha(payload),
                                       "size_bytes": len(payload)}
    (job.directory / "progression.json").write_text(json.dumps(intake._seal(progression, "progression_digest")))
    from blueprint_pipeline import task_evaluation_configured_controls_progression_worker as source
    monkeypatch.setattr(source, "_validate_source", lambda root: ({
        "configured_scene_revision_path": str(job.run / "configured_scene_revision.v1.json"),
        "configured_scene_revision_digest": configured["revision_digest"]}, {}, {}))
    with pytest.raises(ValueError, match="apply_ack_mismatch"):
        door.run(body(job, apply=True, apply_ack=preview["apply_ack"]), door_caller="cloud", door_request_id="a")
    assert not (job.directory / confirmation.DIRECTORY).exists()


def test_main_reports_a_refusal_as_a_door_outcome(tmp_path, monkeypatch, capsys):
    job = publish(tmp_path, monkeypatch)
    arguments = ["--request-json", json.dumps(body(job, approved_by="someone-else")), "--door-caller", "cloud",
                 "--door-request-id", "r"]
    assert door.main(arguments) == 2
    assert json.loads(capsys.readouterr().out) == {
        "door_outcome": "refused", "code": "owner_success_confirmation_owner_or_revocation_mismatch"}
    assert door.main(["--request-json", json.dumps(body(job)), "--door-caller", "cloud",
                      "--door-request-id", "r"]) == 0
    assert json.loads(capsys.readouterr().out)["door_outcome"] == "planned"
