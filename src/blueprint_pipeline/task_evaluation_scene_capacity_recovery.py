"""Observe closed pre-allocation capacity refusals and admit bounded controller recovery.

Historical bundle receipts prove what failed; a removed historical ZIP never
becomes executable input. Every successor still builds and admits a new bundle.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
import shutil

from . import task_evaluation_scene_intake as intake
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_public_scene_attempt_factory import record
from .task_evaluation_scene_configuration_submission_inputs import checked_file, read
from .task_evaluation_scene_progression_state import require, safe_path

KIND = "preallocation_capacity"
BLOCKER = "scene_configuration_provider_output_disk_capacity_insufficient"
CREDIT_KIND = "preallocation_credit"
#: A scene-configuration launch whose provider machine was created and then produced
#: nothing. The disk-capacity path above is strictly a $0 no-instance refusal, so this
#: is a different class and gets its own evidence: the instance really did exist.
#: Scene 840938, 2026-09-16: two consecutive machines (137888, 108924) accepted the
#: create call and never started a container, and with no launch-level recovery the
#: hands-off run parked at scene_configuration_failed each time.
DEAD_MACHINE_PROOF_BLOCKERS = frozenset({
    "vast_heartbeat_instance_exited",
    "vast_heartbeat_container_missing",
    "vast_heartbeat_no_log_progress_timeout",
    "vast_probe_failed",
})
#: Consequences of producing no output. They may accompany the proof above, and on
#: their own they mean the work failed rather than the machine dying.
DEAD_MACHINE_CONSEQUENCE_BLOCKERS = frozenset({
    "scene_configuration_configured_revision_not_published",
    "scene_configuration_provider_envelope_mismatch",
    "scene_configuration_provider_not_completed",
    "scene_configuration_provider_output_zip_invalid",
    "scene_configuration_provider_run_id_mismatch",
    "scene_configuration_provider_source_commit_mismatch",
    "scene_configuration_provider_source_envelope_mismatch",
    "task_evaluation_artifact_role_missing:provider_runtime_evidence",
})


def credit_launch_failure(result) -> bool:
    """A funding refusal before any allocation or API preparation executed."""
    if not isinstance(result, Mapping):
        return False
    blockers = set(result.get("blockers") or [])
    return (result.get("schema_version") == "task_evaluation_scene_configuration_vast_result.v1"
            and result.get("status") == "blocked"
            and "provider_credit_insufficient" in blockers
            and blockers <= DEAD_MACHINE_CONSEQUENCE_BLOCKERS | {"provider_credit_insufficient"}
            and result.get("api_pretraining") is None
            and result.get("continuing_spend_from_this_run") is False
            and not result.get("provider_runtime_output_zip_path")
            and type(result.get("provider_mutations_performed")) is int
            and result["provider_mutations_performed"] == 0
            and not result.get("instance_id") and not result.get("vast_instance_ids")
            and result.get("allocation_created") is not True)


def dead_machine_launch_failure(result) -> bool:
    """Did this launch rent a machine that then produced nothing at all?

    Every blocker must be a recognised dead-machine proof or one of its
    consequences, and at least one must actually prove the machine died. Nothing
    may still be spending. An execution failure on a live machine keeps every
    other terminal path it had.
    """
    if not isinstance(result, Mapping):
        return False
    blockers = [str(b) for b in (result.get("blockers") or [])]
    allowed = DEAD_MACHINE_PROOF_BLOCKERS | DEAD_MACHINE_CONSEQUENCE_BLOCKERS
    return (result.get("schema_version") == "task_evaluation_scene_configuration_vast_result.v1"
            and result.get("status") == "blocked"
            and bool(blockers)
            and all(b in allowed for b in blockers)
            and any(b in DEAD_MACHINE_PROOF_BLOCKERS for b in blockers)
            and result.get("continuing_spend_from_this_run") is False)
RESULT_RELATIVE = "allocator/scene-configuration-job/task_evaluation_scene_configuration_vast_result.v1.json"
MAX_LAUNCHES = 4096


def _values(refs):
    result = {}
    fields = {"profile": "profile_digest", "request": "request_digest", "launch": "receipt_digest",
              "result": "result_digest", "zero": "provider_zero_receipt_digest",
              "bundle": "receipt_digest", "authority": "authority_digest", "link": "link_digest", "factory": "factory_digest"}
    require(set(refs) == set(fields) | {"configuration_attempt", "preparation"}, "capacity_records_invalid")
    for name, ref in refs.items():
        path = checked_file(safe_path(ref["path"]), ref)
        result[name] = (intake._read(path, "attempt_digest") if name == "configuration_attempt"
                        else read(path, digest_field=fields.get(name)))
    return result


def validate_source(refs, *, prior_attempt, kind="preallocation_capacity"):
    """Reopen exact producer/request/owner bytes, including the failed disk phase."""
    from .task_evaluation_scene_configuration_provider_artifacts import (
        _provider_output_disk_requirements, _provider_transfer_byte_budget,
    )
    values = _values(refs)
    from .task_evaluation_launch_dispatcher import validate_launch_profile_structure, verify_profile_immutable_inputs
    require(not validate_launch_profile_structure(values["profile"]), "capacity_profile_contract_invalid")
    require(not verify_profile_immutable_inputs(values["profile"]), "capacity_profile_immutable_inputs_invalid")
    profile, request, launch, result, zero, bundle, authority, link, main, preparation = (
        values[name] for name in ("profile", "request", "launch", "result", "zero", "bundle", "authority",
                                 "link", "configuration_attempt", "preparation"))
    factory = values["factory"]
    require(factory.get("attempt_digest") == prior_attempt["attempt_digest"]
            and factory.get("intent_digest") == prior_attempt["intent_digest"]
            and factory.get("source_commit") == prior_attempt["source_commit"]
            and factory.get("submission_request") == refs["preparation"], "capacity_prior_source_mismatch")
    scope, owner = profile.get("task_evaluation_run") or {}, profile.get("scene_attempt_binding") or {}
    free_preparation = (prior_attempt.get("schema_version") == "task_evaluation_scene_preparation_attempt.v1"
                        and prior_attempt.get("provider") == "control_plane"
                        and prior_attempt.get("maximum_spend_usd") == 0
                        and prior_attempt.get("paid_authority_granted") is False)
    require(main.get("provider") == "vast" and (prior_attempt.get("provider") == "vast" or free_preparation)
            and link.get("intent_digest") == prior_attempt["intent_digest"] == main.get("intent_digest")
            == profile.get("scene_intent_digest") == preparation.get("scene_intent_digest")
            and link.get("intent_id") == prior_attempt["intent_id"] == main.get("intent_id")
            and link.get("request_digest") == canonical_digest(preparation) == main.get("input_digest")
            and link.get("scene_configuration_attempt") == refs["configuration_attempt"]
            and all(owner.get(key) == main.get(key) for key in (
                "intent_id", "intent_digest", "attempt_id", "source_commit", "runtime_digest", "input_digest"))
            and scope.get("run_mode") == "scene_configuration"
            and all(scope.get(key) == link.get(key) for key in ("team_namespace", "scene_id", "task_id"))
            and scope.get("configuration_run_id") == preparation.get("run_id") == result.get("run_id") == bundle.get("run_id")
            and profile.get("source_commit") == request.get("source_commit") == result.get("source_commit")
            == bundle.get("source_commit") == authority.get("source_commit") == prior_attempt["source_commit"]
            and request.get("launch_profile_id") == profile.get("profile_id")
            and request.get("launch_profile_digest") == launch.get("launch_profile_digest")
            == zero.get("launch_profile_digest") == profile["profile_digest"]
            and request.get("request_digest") == launch.get("request_digest") == zero.get("request_digest")
            and request.get("launch_id") == launch.get("launch_id") == zero.get("launch_id")
            == Path(refs["profile"]["path"]).parent.name
            and launch.get("status") == "blocked"
            and zero.get("status") == "provider_zero_confirmed" and zero.get("provider_zero_verified") is True
            and zero.get("continuing_spend_from_this_run") is False and not zero.get("blockers")
            and zero.get("receipt_digest") == launch["receipt_digest"], "capacity_launch_binding_invalid")
    inputs = {row["name"]: row for row in profile["immutable_inputs"]}
    for name, source_name in (("bundle", "source_bundle_manifest"), ("authority", "scene_configuration_attempt_authority")):
        require(inputs[source_name]["path"] == refs[name]["path"]
                and inputs[source_name]["digest"] == refs[name]["sha256"], "capacity_immutable_input_changed")
    require(authority.get("bundle_receipt") == refs["bundle"]
            and result.get("bundle_sha256") == bundle.get("bundle_sha256") == authority.get("bundle_sha256")
            and result.get("authority_digest") == authority.get("authority_digest")
            and result.get("schema_version") == "task_evaluation_scene_configuration_vast_result.v1"
            and result.get("status") == "blocked"
            and (result.get("blockers") == [BLOCKER] if kind == KIND
                 else credit_launch_failure(result) if kind == CREDIT_KIND
                 else kind == "provider_dead_machine" and dead_machine_launch_failure(result))
            and type(result.get("provider_mutations_performed")) is int and result["provider_mutations_performed"] == 0
            and type(result.get("retry_cap")) is int and result["retry_cap"] == 0
            and result.get("continuing_spend_from_this_run") is False
            and (kind != "preallocation_capacity" or (
                not result.get("instance_id") and not result.get("vast_instance_ids")
                and result.get("allocation_created") is not True)),
            "capacity_not_zero_provider_failure")
    if kind == CREDIT_KIND:
        from .task_evaluation_retained_controls_evidence import _file
        ref = launch["terminal_evidence"]["artifacts"]["teardown_manifest_path"]
        require(ref.get("exists") is True
                and _file(safe_path(ref["path"]))["digest"] == ref["digest"], "credit_teardown_changed")
        teardown = read(ref["path"])
        require(teardown.get("schema_version") == "vast_teardown_manifest.v1"
                and teardown.get("status") == "not_required_prelaunch_inventory_guard_blocked"
                and teardown.get("vast_instance_ids") == []
                and teardown.get("continuing_spend_from_this_run") is False, "credit_teardown_invalid")
        return values
    if kind != "preallocation_capacity":
        # A dead machine never reached the disk phase, so there is no shortfall
        # measurement to reopen. Every identity and binding assertion above still ran.
        return values
    _download, upload = _provider_transfer_byte_budget(bundle)
    requirements = _provider_output_disk_requirements(upload)
    disk = result.get("provider_output_disk_capacity") or {}
    require(result.get("expected_provider_upload_bytes") == upload
            and result.get("provider_output_disk_requirements") == requirements
            and disk.get("schema_version") == "scene_configuration_provider_output_disk_capacity.v1"
            and disk.get("phase") == "before_allocation_and_staging" and disk.get("status") == "blocked"
            and disk.get("blockers") == [BLOCKER]
            and disk.get("measurement_path") == str(Path(refs["result"]["path"]).parent / "vast_provider_run")
            and disk.get("required_free_bytes") == requirements["required_free_bytes_before_download"]
            and type(disk.get("observed_free_bytes")) is int
            and 0 <= disk["observed_free_bytes"] < disk["required_free_bytes"], "capacity_phase_or_requirement_invalid")
    return values


def observe_failure(*, attempt, link_path, preparation_path, factory_path, config):
    """Find only this preparation's configuration launch, never another intent's failure."""
    link, preparation = read(link_path, digest_field="link_digest"), read(preparation_path)
    main = intake._read(checked_file(link["scene_configuration_attempt"]["path"], link["scene_configuration_attempt"]), "attempt_digest")
    root = safe_path(config["launch_execution_root"])
    matches = []
    for index, directory in enumerate(root.iterdir()):
        require(index < MAX_LAUNCHES, "capacity_launch_scan_bound_exceeded")
        if directory.is_symlink() or not directory.is_dir() or not (directory / "launch_profile.json").is_file():
            continue
        profile = read(directory / "launch_profile.json")
        if ((profile.get("task_evaluation_run") or {}).get("configuration_run_id") != preparation["run_id"]
                or (profile.get("scene_attempt_binding") or {}).get("attempt_id")
                != main["attempt_id"]):
            continue
        result_path = directory / RESULT_RELATIVE
        if not result_path.is_file():
            continue
        result = read(result_path, digest_field="result_digest")
        if result.get("status") != "blocked":
            continue
        # Surface other terminal failures, but do not turn them into capacity retries.
        kind = None
        if result.get("blockers") == [BLOCKER]:
            kind = "preallocation_capacity"
        elif credit_launch_failure(result):
            kind = CREDIT_KIND
        elif dead_machine_launch_failure(result):
            kind = "provider_dead_machine"
        if kind is None:
            return {"recoverable": False, "blockers": result.get("blockers") or ["configuration_launch_failed"],
                    "result": record(result_path)}
        inputs = {row["name"]: row for row in profile["immutable_inputs"]}
        refs = {name: record(directory / filename) for name, filename in (
            ("profile", "launch_profile.json"), ("request", "launch_request.json"),
            ("launch", "launch_receipt.json"), ("zero", "post_teardown_provider_zero_receipt.json"),
            ("result", RESULT_RELATIVE))}
        refs.update(bundle=record(Path(inputs["source_bundle_manifest"]["path"])),
                    authority=record(Path(inputs["scene_configuration_attempt_authority"]["path"])),
                    link=record(link_path), preparation=record(preparation_path), factory=record(factory_path),
                    configuration_attempt=link["scene_configuration_attempt"])
        values = validate_source(refs, prior_attempt=attempt, kind=kind)
        matches.append({"recoverable": True, "kind": kind,
                        "blockers": [str(b) for b in (result.get("blockers") or [])],
                        "records": refs, "result": refs["result"], "values": values})
    require(len(matches) <= 1, "capacity_launch_identity_ambiguous")
    return matches[0] if matches else None


def capacity_admission(observation, config, now):
    """Measure current headroom for the whole CPU chain, next bundle, and output."""
    from .control_plane_capacity_controller import whole_chain_admission
    from .control_plane_disk_budget import DEFAULT_RESERVATION_ROOT, ROLE_FOOTPRINT_BYTES
    bundle, result = observation["values"]["bundle"], observation["values"]["result"]
    overhead = bundle["bundle_size_bytes"]
    require(type(overhead) is int and overhead > 0, "capacity_bundle_size_invalid")
    reservation_root = (config.get("preparation_worker") or {}).get("disk_reservation_root", DEFAULT_RESERVATION_ROOT)
    chain = whole_chain_admission(config["factory_output_root"], reservation_root=reservation_root, now=now)
    measurement = chain["measurement"]
    require(measurement.get("status") == "measured", "capacity_measurement_unavailable")
    # The full declared chain is larger than the semantic CPU role, but retain
    # both terms explicitly. Exact future bundle/capsule gates still run normally.
    cpu_required = measurement["floor_bytes"] + measurement["reserved_bytes"] + max(
        chain["required_workspace_bytes"], ROLE_FOOTPRINT_BYTES["semantic_pretraining"])
    output_required = result["provider_output_disk_requirements"]["required_free_bytes_before_download"]
    output_path = safe_path(config["launch_execution_root"])
    output_free = shutil.disk_usage(output_path).free
    passed = measurement["free_bytes"] >= cpu_required + overhead and output_free >= output_required + overhead
    credit = None
    if observation.get("kind") == CREDIT_KIND:
        from .provider_credit_admission import credit_admission, observe_vast_credit, RESERVE_ENV
        import os
        credit = credit_admission(observe_vast_credit(),
            required_usd=observation["values"]["authority"]["provider_compute_spend_cap_usd"],
            reserve_usd=float(os.getenv(RESERVE_ENV, "1")))
        passed = passed and credit["status"] == "admitted"
    value = {"schema_version": "task_evaluation_preallocation_capacity_admission.v1",
             "status": "admitted" if passed else "waiting_for_capacity", "observed_at_epoch": now,
             "cpu_path": str(config["factory_output_root"]), "output_path": str(output_path),
             "reservation_root": str(reservation_root), "next_bundle_overhead_bytes": overhead,
             "cpu_required_free_bytes": cpu_required + overhead, "cpu_observed_free_bytes": measurement["free_bytes"],
             "output_required_free_bytes": output_required + overhead, "output_observed_free_bytes": output_free,
             "whole_chain_admission": chain, "provider_mutation_performed": False,
             "historical_bundle_payload_required": False, "successor_requires_new_sealed_bundle": True}
    if credit is not None:
        value["provider_credit_admission"] = credit
    value["capacity_digest"] = canonical_digest(value, digest_field="capacity_digest")
    return value


def retain_failure(*, observation, attempt, output_root, admission):
    require(admission["status"] == "admitted", "capacity_not_recovered")
    output = safe_path(output_root)
    output.mkdir(parents=True, exist_ok=True, mode=0o750)
    result = observation["values"]["result"]
    occurred = datetime.fromisoformat(result["generated_at"].replace("Z", "+00:00")).timestamp()
    value = {"schema_version": "task_evaluation_scene_attempt_failure.v1", "status": "failed",
             "attempt_digest": attempt["attempt_digest"], "failure_kind": KIND, "observed_at_epoch": occurred,
             "producer_result": observation["result"], "construction_records": observation["records"],
             "capacity_admission": admission}
    if observation.get("kind") == CREDIT_KIND:
        value["configuration_failure_kind"] = CREDIT_KIND
    value["failure_digest"] = canonical_digest(value, digest_field="failure_digest")
    path = output / (value["failure_digest"][7:] + ".json")
    if not path.exists():
        intake.write_exclusive(path, value)
    return path


def validate_capacity_failure(failure, producer, prior_attempt, now):
    kind = failure.get("configuration_failure_kind", KIND)
    values = validate_source(failure.get("construction_records") or {}, prior_attempt=prior_attempt, kind=kind)
    require(values["result"] == producer and failure["producer_result"] == failure["construction_records"]["result"],
            "capacity_producer_reference_changed")
    prior = failure.get("capacity_admission") or {}
    require(prior.get("capacity_digest") == canonical_digest(prior, digest_field="capacity_digest"),
            "capacity_admission_digest_invalid")
    require(prior.get("status") == "admitted", "capacity_not_recovered")
    require(type(prior.get("observed_at_epoch")) in (int, float)
            and 0 <= now - prior["observed_at_epoch"] <= 300, "capacity_admission_stale")
    refs = failure["construction_records"]
    require(prior["cpu_path"] == str(Path(refs["factory"]["path"]).parents[2])
            and prior["output_path"] == str(Path(refs["profile"]["path"]).parents[1]), "capacity_measurement_scope_changed")
    current = capacity_admission({"values": values, "kind": kind}, {"factory_output_root": prior["cpu_path"],
        "launch_execution_root": prior["output_path"],
        "preparation_worker": {"disk_reservation_root": prior["reservation_root"]}}, now)
    require(current["status"] == "admitted", "capacity_not_recovered")
