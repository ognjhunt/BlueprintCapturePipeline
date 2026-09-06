"""Drive persistent scene intent through real no-allocation preparation services."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Callable

from . import task_evaluation_scene_intake as intake
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .task_evaluation_public_scene_attempt_factory import materialize_public_scene_attempt, record
from .task_evaluation_scene_configuration_submission_inputs import read, checked_file
from .task_evaluation_scene_progression_state import (
    advance, atomic_json, intent_lock, load_progression, require, safe_path,
)
from .task_evaluation_scene_progression_transport import submit_preparation, read_preparation_status

CONFIG_SCHEMA = "task_evaluation_scene_progression_config.v1"
CONFIG_ENV = "BLUEPRINT_TASK_EVALUATION_SCENE_PROGRESSION_CONFIG"


@dataclass(frozen=True)
class SourceResolution:
    status: str
    binding_path: Path | None = None
    machinery_path: Path | None = None
    materializer: Callable[..., dict] | None = None
    blockers: tuple[str, ...] = ()
    analysis_reference: dict | None = None


def _reference(ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256", "size_bytes"}, "reference_invalid")
    return checked_file(safe_path(ref["path"]), ref)


def _put(path, value):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    if path.exists():
        require(read(path) == value, "immutable_record_conflict")
    else:
        intake.write_exclusive(path, value)
    return path


def _source(intent, config, release, resolver):
    if intent["request"]["source"]["kind"] == "public_scene":
        path = safe_path(Path(config["public_source_binding_root"]) / (intent["request"]["source"]["binding_id"] + ".json"))
        if not path.is_file():
            return SourceResolution("awaiting_source", blockers=("public_source_binding_missing",))
        return SourceResolution("resolved", path, Path(config["machinery_path"]), materialize_public_scene_attempt)
    if resolver is None:
        try:
            from .task_evaluation_scene_source_resolver import resolve_scene_source
        except ImportError:
            return SourceResolution("awaiting_source", blockers=("scene_source_resolver_unavailable",))
        resolver = resolve_scene_source
    result = resolver(intent=intent, config=config, release=release)
    require(isinstance(result, SourceResolution), "source_resolution_invalid")
    return result


def _queue(preparation, queue_root):
    from .task_evaluation_launch_preparation_queue import QUEUE_STATES, ENVELOPE_SCHEMA_VERSION
    from .task_evaluation_launch_preparation_contract import launch_preparation_request_digest
    queue = safe_path(queue_root)
    require(queue.is_dir(), "preparation_queue_missing")
    matches = [(state, path) for state in QUEUE_STATES
               for path in (queue / state).glob(preparation["preparation_id"] + "-*.json")]
    require(len(matches) <= 1, "preparation_identity_ambiguous")
    if not matches:
        return {"status": "not_found"}
    state, path = matches[0]
    envelope = read(path, digest_field="envelope_digest")
    actual = envelope["request"]
    require(envelope.get("schema_version") == ENVELOPE_SCHEMA_VERSION
            and cross_runtime_canonical_digest(actual) == cross_runtime_canonical_digest(preparation)
            and envelope.get("request_digest") == launch_preparation_request_digest(actual),
            "preparation_envelope_mismatch")
    observed = {"status": state, "envelope": record(path), "request": actual,
                "request_digest": envelope["request_digest"], "result_filename": path.name}
    result_path = queue / "results" / path.name
    if result_path.exists():
        result = read(result_path, digest_field="result_digest")
        require(result.get("preparation_id") == preparation["preparation_id"]
                and result.get("source_commit") == preparation["expected_production_commit"],
                "preparation_result_mismatch")
        observed.update(result=result, result_reference=record(result_path))
    if state == "awaiting_source_preparation":
        from .task_evaluation_sam31_preparation_queue import load_progress
        progress = load_progress(queue, path.name, envelope["request_digest"])
        require(progress is not None, "source_progress_missing")
        observed["source_progress"] = progress
    return observed


def _publish(factory, output, config, publisher):
    if publisher is None:
        from .task_evaluation_scene_configuration_submission_publication import publish_scene_configuration_submission
        publisher = publish_scene_configuration_submission
    manifest_path = _reference(factory["submission_manifest"])
    path = output / "publication.json"
    result = publisher(manifest_path=manifest_path, receipt_path=path,
        expected_source_commit=factory["source_commit"], service_account=config.get("service_account", "blueprint"),
        lock_root=Path(config["publication_lock_root"]))
    require(result.get("status") == "published_and_read_back"
            and result.get("source_commit") == factory["source_commit"]
            and result.get("raw_source_uploaded") is False and result.get("provider_allocated") is False
            and result.get("manifest_sha256") == factory["submission_manifest"]["sha256"], "publication_receipt_invalid")
    _put(path, result)
    return record(path)


def _submission(*, request_path, output, config, observed, submitter, status_reader, now, intent_reference=None):
    preparation = read(request_path)
    digest = cross_runtime_canonical_digest(preparation)
    receipt_path = output / "submission.json"
    if receipt_path.exists():
        value = intake._read(receipt_path, "receipt_digest")
        require(value.get("request_digest") == digest and value.get("preparation_id") == preparation["preparation_id"],
                "submission_receipt_mismatch")
        return record(receipt_path)
    calls_root = output / "submission-attempts"
    calls = sorted(calls_root.glob("*.json"))
    for path in calls:
        row = intake._read(path, "receipt_digest")
        require(row.get("request_digest") == digest, "submission_attempt_mismatch")
    if calls:
        if observed["status"] != "not_found":
            value = {"status": "reconciled_from_pipeline_envelope", "envelope": observed["envelope"]}
            _put(receipt_path, intake._seal({"schema_version": "task_evaluation_scene_submission.v1",
                "request_digest": digest, "preparation_id": preparation["preparation_id"],
                "observed_at_epoch": now, **value}, "receipt_digest"))
            return record(receipt_path)
        local = config.get("submission_transport") == "local_owned_queue"
        status = ({"status": "not_found", "authoritative": True,
                   "request_digest": digest, "preparation_id": preparation["preparation_id"]}
                  if local else (status_reader or read_preparation_status)(request_path=request_path, config=config))
        require(status.get("authoritative") is True and status.get("request_digest") == digest
                and status.get("preparation_id") == preparation["preparation_id"], "submission_status_binding_invalid")
        status_path = output / "submission-status" / (str(len(calls)) + ".json")
        if not status_path.exists():
            _put(status_path, intake._seal(status, "receipt_digest"))
        if status["status"] != "not_found":
            return None  # Forwarding is known, but no local envelope exists yet.
        require(len(calls) < config.get("maximum_http_submission_attempts", 2), "submission_retry_cap_exhausted")
    elif observed["status"] != "not_found":
        raise ValueError("scene_progression_unowned_preparation_already_exists")
    # Persist uncertainty BEFORE POST. A crash cannot erase a transport attempt
    # and silently lead to another idempotency key or unchecked resubmission.
    call = intake._seal({"schema_version": "task_evaluation_scene_submission_attempt.v1",
        "request_digest": digest, "preparation_id": preparation["preparation_id"],
        "request": record(request_path), "sequence": len(calls) + 1,
        "observed_at_epoch": now, "status": "sending"}, "receipt_digest")
    _put(calls_root / f"{len(calls) + 1:03d}.json", call)
    local = config.get("submission_transport") == "local_owned_queue"
    if local:
        from .task_evaluation_scene_progression_transport import submit_owned_preparation
        result = submit_owned_preparation(request_path=request_path, config=config, intent_reference=intent_reference)
    else:
        result = (submitter or submit_preparation)(request_path=request_path, config=config)
    require(result.get("schema_version") == ("task_evaluation_owned_preparation_submission.v1" if local
                else "task_evaluation_launch_preparation_web_submission_receipt.v1")
            and result.get("status") in {"submitted", "replayed"}
            and result.get("request_digest" if local else "webapp_request_digest") == digest
            and result.get("preparation_id") == preparation["preparation_id"]
            and result.get("paid_execution_requested_by_this_tool") is False, "submission_response_invalid")
    value = intake._seal({"schema_version": "task_evaluation_scene_submission.v1",
        "status": "submitted", "request_digest": digest, "preparation_id": preparation["preparation_id"],
        "webapp_evidence": result, "observed_at_epoch": now}, "receipt_digest")
    _put(receipt_path, value)
    return record(receipt_path)


def _link(*, intent, attempt, observed, directory, config, now):
    from .task_evaluation_controls_autoprovision import build_preparation_link
    request = observed["request"]
    digest = observed["request_digest"]
    for key in ("preparation_id", "team_namespace"):
        require(intake._identifier(request[key]), "preparation_identifier_too_long")
    paid = {}
    if config.get("activation_enabled", True):
        main = intake.reserve_scene_attempt(queue_root=config["intent_root"], intent_id=intent["intent_id"],
            attempt_id="scene-configuration-" + digest[7:31], source_commit=attempt["source_commit"],
            runtime_digest=request["execution_adapter"]["runtime_source_bundle"]["digest"],
            input_digest=digest, provider="vast", maximum_spend_usd=request["spend"]["hard_cap_usd"], now=now)
        paid["scene_configuration_attempt"] = record(directory / "attempts" / (main["attempt_id"] + ".json"))
    link = build_preparation_link(intent_id=intent["intent_id"], intent_digest=intent["intent_digest"],
        preparation_id=request["preparation_id"], request_digest=digest, expected_production_commit=attempt["source_commit"],
        team_namespace=request["team_namespace"], scene_id=request["scene"]["identity"]["id"],
        task_id=request["task"]["identity"]["id"], result_filename=observed["result_filename"],
        **paid)
    path = _put(directory / "preparations" / (digest[7:] + ".json"), link)
    atomic_json(directory / "preparation-link.json", link)
    return record(path)


def _activation(*, intent, link, config, output, now, provisioner):
    from .task_evaluation_scene_configuration_activation_automation import provision_scene_configuration_activation_intent
    from .project_spend_reconciliation import validate_project_spend_reconciliation
    path = output / "activation_provisioning.json"
    if path.exists():
        value = intake._read(path, "receipt_digest")
        require(value.get("link_digest") == link["link_digest"], "activation_link_changed")
        return record(path)
    if provisioner is None:
        from .task_evaluation_scene_spend import refresh_configured_scene_project_spend
        refresh_configured_scene_project_spend()
    inputs_path = output / "activation_inputs.json"
    if inputs_path.exists():
        inputs = intake._read(inputs_path, "receipt_digest")
    else:
        current = read(config["project_spend_current_path"], digest_field="receipt_digest")
        require(current.get("schema_version") == "task_evaluation_project_spend_current.v1"
                and intake._number(current.get("observed_at_epoch"))
                and 0 <= now - current["observed_at_epoch"] <= 900, "project_spend_stale")
        source = checked_file(current["path"], {"sha256": current["digest"], "size_bytes": Path(current["path"]).stat().st_size})
        validate_project_spend_reconciliation(source)
        snapshot = _put(output / "project_spend_snapshot.json", read(source))
        inputs = intake._seal({"spend": record(snapshot), "issued_at_epoch": now,
                               "link_digest": link["link_digest"]}, "receipt_digest")
        _put(inputs_path, inputs)
    require(inputs["link_digest"] == link["link_digest"], "activation_inputs_changed")
    owner = intent["request"]["owner"]["user_id"]
    seconds = min(86400, int(intent["request"]["execution"]["expires_at_epoch"] - inputs["issued_at_epoch"]))
    require(seconds >= 300, "activation_authority_window_too_short")
    main = intake._read(_reference(link["scene_configuration_attempt"]), "attempt_digest")
    result = (provisioner or provision_scene_configuration_activation_intent)(
        expected_production_commit=link["expected_production_commit"], team_namespace=link["team_namespace"],
        scene_id=link["scene_id"], task_id=link["task_id"], authorization_reference="scene-intent:" + intent["intent_digest"],
        authorized_by=owner, profile_revision="scene-" + intent["intent_id"][-16:], valid_for_seconds=seconds,
        project_spend_reconciliation_path=_reference(inputs["spend"]),
        rights_scope=intent["request"]["consent"]["rights_reference"], maximum_hard_cap_usd=main["maximum_spend_usd"],
        release_reference="scene-intent:" + intent["intent_digest"], intent_root=config["activation_intent_root"],
        materialization_root=output / "activation-inputs", release_window_valid_for_seconds=seconds,
        service_group=config.get("service_group"))
    require(result.get("expected_production_commit") == link["expected_production_commit"]
            and result.get("provider_mutation_performed") is False, "activation_producer_invalid")
    _put(path, intake._seal({"link_digest": link["link_digest"], "activation_intent": result,
                          "provider_allocation_performed": False}, "receipt_digest"))
    return record(path)


def _clear_attempt(state):
    for key in ("attempt_id", "attempt_commit", "attempt", "factory", "publication", "submission",
                "preparation_state", "preparation_link", "preparation_result", "activation", "failure"):
        state.pop(key, None)


def _release_successor(*, directory, intent, state, config, release, now):
    from .task_evaluation_scene_progression_recovery import reconcile_ownership
    previous_id = state["attempt_id"]
    previous_path = _reference(state["attempt"])
    previous = intake._read(previous_path, "attempt_digest")
    old_output = Path(config["factory_output_root"]) / intent["intent_id"] / previous_id
    calls = list((old_output / "submission-attempts").glob("*.json"))
    lineage = {"attempt": record(previous_path),
               "new_source_commit": release["source_commit"]}
    if previous.get("schema_version") == "task_evaluation_scene_preparation_attempt.v1":
        require(config.get("activation_enabled") is False and not state.get("activation")
                and previous.get("maximum_spend_usd") == 0
                and previous.get("paid_authority_granted") is False,
                "preparation_release_authority_conflict")
        # Once any paid reservation exists, use the execution recovery path;
        # administrative preparation alone must never explain away a live run.
        require(not list((directory / "attempts").glob("*.json")), "preparation_release_paid_reservation_exists")
        lineage["basis"] = "preparation_only_no_execution_authority_issued"
    elif not calls:
        lineage["basis"] = "no_submission_attempt_performed"
    else:
        factory = read(_reference(state["factory"]), digest_field="factory_digest")
        old_request = read(_reference(factory["submission_request"]))
        observed = _queue(old_request, config["preparation_queue_root"])
        if observed["status"] not in {"blocked", "completed", "materialized"}:
            return False
        transition_path = old_output / "release-transition.json"
        if not transition_path.exists():
            value = {"schema_version": "task_evaluation_scene_release_transition.v1",
                "attempt_digest": previous["attempt_digest"], "observed_at_epoch": now,
                "parent_envelope": observed["envelope"], "parent_state": observed["status"],
                "provider_allocation_performed": False}
            value["failure_digest"] = canonical_digest(value, digest_field="failure_digest")
            _put(transition_path, value)
        lineage["reconciliation"] = reconcile_ownership(attempt=previous, failure_path=transition_path,
            config=config, output_root=old_output / "release-reconciliation", now=now)
        lineage["basis"] = "terminal_preparation_and_reconciled_global_ownership"
    state.setdefault("release_predecessors", []).append(lineage)
    _clear_attempt(state)
    return True


def _recover(*, directory, intent, state, attempt, link, config, release, machinery, output, now):
    from .task_evaluation_scene_progression_recovery import retain_failure, reconcile_ownership
    failure = retain_failure(attempt=attempt, link=link, child_queue_root=config["child_queue_root"],
        output_root=output / "recovery", now=now)
    if failure is None:
        return False
    state["failure"] = record(failure)
    if intent["request"]["execution"]["max_retries"] == 0:
        return False
    evidence = reconcile_ownership(attempt=attempt, failure_path=failure, config=config,
        output_root=output / "recovery/reconciliations", now=now)
    successor_id = "source-" + canonical_digest({"prior_attempt_digest": attempt["attempt_digest"],
        "source_commit": release["source_commit"], "intent_digest": intent["intent_digest"]})[7:31]
    successor = intake.reserve_scene_attempt(queue_root=config["intent_root"], intent_id=intent["intent_id"],
        attempt_id=successor_id, source_commit=release["source_commit"], runtime_digest=release["runtime_digest"],
        input_digest=state["binding_digest"], provider=attempt["provider"],
        maximum_spend_usd=machinery["maximum_preparation_spend_usd"], now=now,
        recovery_from_attempt_id=attempt["attempt_id"], recovery_evidence=evidence)
    state.setdefault("recovery_predecessors", []).append({"attempt": state["attempt"], "evidence": evidence})
    _clear_attempt(state)
    state.update(attempt_id=successor_id, attempt_commit=release["source_commit"],
                 attempt=record(directory / "attempts" / (successor["attempt_id"] + ".json")))
    return True


def _advance_intent(directory, intent, config, release, *, resolver, publisher, submitter, status_reader,
                    activation_provisioner, now):
    progress = load_progression(directory, intent)
    state = deepcopy(progress.get("state", {})) if progress else {}
    def emit(status, phase, blockers=(), result_reference=None):
        nonlocal progress
        progress = advance(directory, intent, progress, status=status, phase=phase, state=deepcopy(state),
                           blockers=blockers, result_reference=result_reference, now=now)
        return progress
    if progress and progress["status"] == "completed":
        return progress
    # Spec E: once activation has been issued, join any retained downstream
    # terminal receipts (policy result, authenticated Website readback,
    # provider-zero closure) back into the persistent owner status. This is a
    # READ-ONLY closeout of already-authorized execution and runs BEFORE the
    # expiry/revocation/pause gates below (A8): those gate NEW execution, not the
    # read-only join of a run that was authorized when it executed -- otherwise a
    # completed run whose authority window later lapsed could never close out. It
    # never launches, retries, or reruns completed GPU work; when there is no
    # owner-bound terminal result yet it returns None and control falls through to
    # the authority gates unchanged.
    if config.get("terminal_result_root") and state.get("activation"):
        from .task_evaluation_scene_terminal_reconciler import reconcile_terminal_owner_result
        terminal = reconcile_terminal_owner_result(intent=intent, config=config, release=release, now=now,
            output=safe_path(Path(config["factory_output_root"]) / intent["intent_id"] / "terminal-reconciliation"))
        if terminal is not None:
            state.update(terminal.get("state", {}))
            return emit(terminal["status"], terminal["phase"], terminal.get("blockers", ()),
                        terminal.get("result_reference"))
    if (directory / "revoked.json").exists() or now >= intent["request"]["execution"]["expires_at_epoch"]:
        return emit("blocked", "authority", ["scene_intake_authority_revoked" if (directory / "revoked.json").exists()
                                              else "scene_intake_authority_expired"])
    if intent["intent_id"] in config.get("paused_intent_ids", []):
        return emit("awaiting_execution", "paused", ["scene_intent_paused"])
    if config.get("supported_source_kinds") is not None and intent["request"]["source"]["kind"] not in config["supported_source_kinds"]:
        return emit("needs_input", "source", ["source_kind_not_supported_by_progression"])
    resolution = _source(intent, config, release, resolver)
    if resolution.analysis_reference is not None:
        _reference(resolution.analysis_reference)
        state["source_analysis"] = resolution.analysis_reference
    if resolution.status != "resolved":
        require(resolution.status in {"awaiting_source", "needs_input", "blocked"}, "source_status_invalid")
        return emit(resolution.status, "source", resolution.blockers)
    require(resolution.binding_path is not None and resolution.machinery_path is not None and callable(resolution.materializer),
            "source_resolution_incomplete")
    binding = read(resolution.binding_path, digest_field="binding_digest")
    machinery = read(resolution.machinery_path, digest_field="machinery_digest")
    require(binding.get("binding_id") == intent["request"]["source"]["binding_id"]
            and binding.get("source_content_digest") == intent["request"]["source"]["content_digest"],
            "source_binding_mismatch")
    if state.get("binding_digest") is not None:
        require(state["binding_digest"] == binding["binding_digest"], "source_binding_changed")
    state["binding_digest"] = binding["binding_digest"]
    active_id = state.get("attempt_id")
    if active_id and state.get("attempt_commit") != release["source_commit"]:
        # Do not replace an old in-flight attempt merely because a deploy moved.
        # A terminal preparation and fresh ownership/zero reconciliation must
        # precede an administrative successor.
        if not _release_successor(directory=directory, intent=intent, state=state,
                                  config=config, release=release, now=now):
            return emit("running", "previous_release", ["previous_release_attempt_not_terminal"])
        active_id = None
    if not active_id:
        identity = {"intent_digest": intent["intent_digest"], "binding_digest": binding["binding_digest"],
                    "source_commit": release["source_commit"], "runtime_digest": release["runtime_digest"]}
        active_id = "source-" + canonical_digest(identity)[7:31]
        state.update(attempt_id=active_id, attempt_commit=release["source_commit"])
    preparation_only = (config.get("activation_enabled") is False
                        and machinery.get("schema_version") == "task_evaluation_completed_scene_machinery.v1")
    attempt_args = {"source_commit": release["source_commit"], "runtime_digest": release["runtime_digest"],
        "input_digest": binding["binding_digest"], "provider": machinery.get("provider", "vast"),
        "maximum_spend_usd": machinery["maximum_preparation_spend_usd"]}
    attempt_path = directory / "attempts" / (active_id + ".json")
    if preparation_only:
        from .task_evaluation_scene_preparation_attempts import create_preparation_attempt, preparation_attempt_path
        attempt = create_preparation_attempt(directory=directory, attempt_id=active_id, now=now,
            **{key: attempt_args[key] for key in ("source_commit", "runtime_digest", "input_digest")})
        attempt_path = preparation_attempt_path(directory, active_id)
    elif attempt_path.exists():
        attempt = intake._read(attempt_path, "attempt_digest")
        require(attempt.get("intent_digest") == intent["intent_digest"]
                and all(attempt.get(key) == value for key, value in attempt_args.items()), "attempt_binding_changed")
    else:
        attempt = intake.reserve_scene_attempt(queue_root=config["intent_root"], intent_id=intent["intent_id"],
            attempt_id=active_id, **attempt_args, now=now)
    output = safe_path(Path(config["factory_output_root"]) / intent["intent_id"] / active_id)
    output.mkdir(parents=True, exist_ok=True, mode=0o750)
    state["attempt"] = record(attempt_path)
    if not state.get("factory"):
        emit("preparing", "factory")
    # Mutable service pointers are snapshotted once per immutable attempt.
    binding_path = _put(output / "source_binding.json", binding)
    machinery_path = _put(output / "machinery.json", machinery)
    release_path = _put(output / "release_binding.json", release)
    factory_path = output / "factory.json"
    if state.get("factory"):
        factory = read(_reference(state["factory"]), digest_field="factory_digest")
    else:
        from .task_evaluation_scene_preparation_attempts import preparation_storage
        with preparation_storage(config, binding, output):
            factory = resolution.materializer(intent_path=directory / "intent.json", source_binding_path=binding_path,
                machinery_path=machinery_path, release_binding_path=release_path, output_root=output / "materialized",
                attempt_id=active_id)
        if factory.get("status") in {"needs_input", "awaiting_source", "blocked"}:
            return emit(factory["status"], "factory", factory.get("blockers", []))
        require(factory.get("status") == "publication_ready"
                and factory.get("source_commit") == attempt["source_commit"]
                and factory.get("intent_digest") == intent["intent_digest"]
                and factory.get("attempt_digest") == attempt["attempt_digest"]
                and factory.get("factory_digest") == canonical_digest(factory, digest_field="factory_digest")
                and factory.get("provider_mutation_performed") is False, "factory_receipt_invalid")
        _put(factory_path, factory)
        state["factory"] = record(factory_path)
    request_path = _reference(factory["submission_request"])
    preparation = read(request_path)
    require(preparation.get("scene_intent_digest") == intent["intent_digest"]
            and intake._identifier(preparation.get("preparation_id")), "preparation_intent_or_identifier_invalid")
    if not config.get("submission_enabled", False):
        return emit("awaiting_execution", "publication_ready", ["scene_submission_paused"])
    if not state.get("publication"):
        emit("preparing", "publication")
        state["publication"] = _publish(factory, output, config, publisher)
    else:
        _reference(state["publication"])
    observed = _queue(preparation, config["preparation_queue_root"])
    if not state.get("submission"):
        emit("preparing", "submission")
    submitted = _submission(request_path=request_path, output=output, config=config, observed=observed,
                            submitter=submitter, status_reader=status_reader, now=now,
                            intent_reference=record(directory / "intent.json"))
    if submitted is None:
        return emit("preparing", "submission_reconciliation", ["preparation_forwarding_pending"])
    state["submission"] = submitted
    observed = _queue(preparation, config["preparation_queue_root"])
    if observed["status"] == "not_found":
        return emit("preparing", "submission_reconciliation", ["pipeline_preparation_receipt_pending"])
    state["preparation_state"] = observed["status"]
    if not state.get("preparation_link"):
        state["preparation_link"] = _link(intent=intent, attempt=attempt, observed=observed,
                                          directory=directory, config=config, now=now)
    link = read(_reference(state["preparation_link"]), digest_field="link_digest")
    if observed.get("result_reference"):
        state["preparation_result"] = observed["result_reference"]
    if observed["status"] in {"blocked", "awaiting_source_preparation"}:
        if preparation_only:
            return emit("blocked", "source_preparation", observed.get("result", {}).get("blockers") or ["preparation_failed"])
        if _recover(directory=directory, intent=intent, state=state, attempt=attempt, link=link,
                    config=config, release=release, machinery=machinery, output=output, now=now):
            return emit("preparing", "recovery_reserved")
        if observed["status"] == "blocked" or state.get("failure"):
            return emit("blocked", "source_preparation", ["preparation_failed"])
    if observed["status"] == "awaiting_source_preparation":
        phase = observed["source_progress"].get("advancement", {}).get("phase", "source_preparation")
        return emit("running", phase if intake._identifier(phase) else "source_preparation")
    if observed["status"] in {"pending", "processing"}:
        return emit("running", "source_preparation")
    result = observed.get("result", {})
    if result.get("status") == "queued_for_production_scene_configuration":
        if config.get("activation_enabled", True) is False:
            return emit("awaiting_execution", "construction_prepared")
        if not state.get("activation"):
            emit("preparing", "activation")
        state["activation"] = _activation(intent=intent, link=link, config=config, output=output,
                                           now=now, provisioner=activation_provisioner)
        return emit("awaiting_execution", "scene_configuration")
    return emit("awaiting_execution", "preparation_complete")


def process_scene_intents(*, config_path, source_resolver=None, publisher=None, submitter=None,
                          status_reader=None, activation_provisioner=None, now=None):
    config_path = safe_path(config_path)
    require(config_path.stat().st_mode & 0o002 == 0, "config_world_writable")
    config = read(config_path, digest_field="config_digest")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config_schema_invalid")
    root = safe_path(config["intent_root"])
    require(root.is_dir(), "intent_root_missing")
    require(type(config.get("maximum_intents_per_pass", 16)) is int
            and 1 <= config.get("maximum_intents_per_pass", 16) <= 64, "pass_bound_invalid")
    require(type(config.get("maximum_http_submission_attempts", 2)) is int
            and 1 <= config.get("maximum_http_submission_attempts", 2) <= 3, "http_retry_bound_invalid")
    require(type(config.get("submission_enabled", False)) is bool, "submission_mode_invalid")
    require(type(config.get("activation_enabled", True)) is bool, "activation_mode_invalid")
    require(config.get("submission_transport", "webapp") in {"webapp", "local_owned_queue"},
            "submission_transport_invalid")
    from .public_scene_host_input_intake import _verified_checkout_head
    from .task_evaluation_scene_release_binding import resolve_release_binding
    release = resolve_release_binding(config, running_commit=_verified_checkout_head())
    moment = time.time() if now is None else now
    directories = sorted(path for path in root.glob("scene-*") if path.is_dir())
    require(len(directories) <= 10000, "intent_inventory_bound_exceeded")
    cursor_path = root / "progression-cursor.json"
    cursor = intake._read(cursor_path, "cursor_digest")["last_intent_id"] if cursor_path.exists() else ""
    ordered = [p for p in directories if p.name > cursor] + [p for p in directories if p.name <= cursor]
    chosen = ordered[:config.get("maximum_intents_per_pass", 16)]
    if chosen:
        atomic_json(cursor_path, intake._seal({"last_intent_id": chosen[-1].name}, "cursor_digest"))
    rows = []
    for directory in chosen:
        try:
            with intent_lock(directory) as acquired:
                if not acquired:
                    rows.append({"intent_id": directory.name, "status": "writer_active"})
                    continue
                intent = intake._read(directory / "intent.json", "intent_digest")
                require(intent.get("intent_id") == directory.name
                        and intent.get("authenticated_issuer") in config["trusted_clients"], "intent_issuer_invalid")
                intake.validate_request(intent["request"], now=intent["accepted_at_epoch"])
                try:
                    progress = _advance_intent(directory, intent, config, release, resolver=source_resolver,
                        publisher=publisher, submitter=submitter, status_reader=status_reader,
                        activation_provisioner=activation_provisioner, now=moment)
                except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
                    progress = load_progression(directory, intent)
                    code = str(exc).split(":", 1)[0] if isinstance(exc, ValueError) else "scene_progression_dependency_unavailable"
                    if not code or not all(c.islower() or c.isdigit() or c == "_" for c in code):
                        code = "scene_progression_dependency_unavailable"
                    progress = advance(directory, intent, progress, status="blocked", phase="preparation",
                        state=progress.get("state", {}) if progress else {}, blockers=[code], now=moment)
                rows.append({"intent_id": directory.name, "status": progress["status"], "phase": progress["phase"],
                             "blockers": progress.get("blockers", []),
                             "progression_digest": progress["progression_digest"]})
        except (OSError, ValueError, KeyError, TypeError):
            rows.append({"intent_id": directory.name, "status": "blocked", "blockers": ["scene_progression_state_invalid"]})
    return intake._seal({"schema_version": "task_evaluation_scene_progression_run.v1",
        "status": "processed" if rows else "idle", "source_commit": release["source_commit"], "results": rows,
        "provider_allocation_performed": False}, "run_digest")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.getenv(CONFIG_ENV), required=not os.getenv(CONFIG_ENV))
    args = parser.parse_args(argv)
    config = read(safe_path(args.config), digest_field="config_digest")
    if config.get("preparation_worker") is not None:
        from .task_evaluation_scene_preparation_service import run_preparation_service
        result = run_preparation_service(config_path=args.config)
    else:
        result = process_scene_intents(config_path=args.config)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
