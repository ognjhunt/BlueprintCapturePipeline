"""Adopt a closed source prefix or complete SAM preparation into one successor.

This module never writes a queue, invokes an executor, allocates a provider or
changes historical evidence. It emits a separately sealed provenance record.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import time

from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_evaluation_scene_configuration_submission_inputs import read, require, sha
from .task_evaluation_scene_configuration_sam31_plan import PHASES, PROFILE_SCHEMA, validate_sam31_preparation_plan
from .task_evaluation_sam31_prefix_evidence import (
    camera_science, source_science, task_science, validate_current_rights,
    validate_render, validate_tracking,
)

SCHEMA = "task_evaluation_sam31_completed_prefix_adoption.v1"
PREFIX_LENGTHS = {"calibrated_views": 3, "sam31_tracking": 5, "segment_cutout": len(PHASES)}
DEFAULT_QUEUE = Path("/var/lib/blueprint/pipeline-control-plane/sam31-preparation-executions")
DEFAULT_PARENT_QUEUE = Path("/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-preparations")
DEFAULT_EXECUTION = Path("/var/lib/blueprint/task-evaluation-inputs/sam31-preparations")


def record(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            "sam31_adoption_file_invalid")
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def _ref(row, roots):
    require(isinstance(row, dict) and set(row) == {"path", "sha256", "size_bytes"}, "sam31_adoption_reference_invalid")
    path = Path(row["path"])
    require(any(path.resolve().is_relative_to(root.resolve()) for root in roots)
            and record(path) == row, "sam31_adoption_reference_changed")
    return path


def _profile(path, commit):
    profile = read(path, digest_field="profile_digest")
    require(profile.get("schema_version") == PROFILE_SCHEMA and profile.get("source_commit") == commit,
            "sam31_adoption_source_profile_invalid")
    return profile


def _zero(path, *, at):
    value = read(path)
    require(value.get("provider") == "vast" and value.get("status") == "observed"
            and value.get("api_confirmed") is True and value.get("name_prefix") == ""
            and value.get("live_resource_count") == 0 and value.get("resources") == []
            and value.get("http") == 200 and isinstance(value.get("observed_at_epoch"), (int, float))
            and 0 <= at - value["observed_at_epoch"] <= 900,
            "sam31_adoption_fresh_global_zero_required")


def _seed(plan, profile, roots):
    inputs = {**plan["host_inputs"], **profile["artifact_references"]}
    inherited = None
    if profile.get("completed_prefix_adoption") is not None:
        path = _ref(profile["completed_prefix_adoption"], roots)
        inherited = validate_completed_prefix_adoption(path,
            expected_source_commit=plan["source_commit"], approved_roots=roots,
            current_plan=plan,
            current_provider_profile_path=profile["artifact_references"]["sam31_provider_profile"]["path"])
        for name, ref in inherited["artifacts"].items():
            require(name not in inputs or inputs[name] == ref, "sam31_adoption_inherited_input_conflict")
            inputs[name] = ref
    return inputs, inherited


def _render_artifacts(artifacts):
    # Administrative conversion rebinding must not rewrite the original renderer input.
    request = read(artifacts["calibrated_view_request"]["path"])
    original = record(request["scene"]["standard_splat_path"])
    require(all(original[k] == artifacts["standard_splat"][k] for k in ("sha256", "size_bytes")),
            "sam31_adoption_render_source_changed")
    return {**artifacts, "standard_splat": original}


def _phase_chain(value, roots):
    old_commit = value["original_execution_commit"]
    plan_path = _ref(value["source_plan"], roots)
    plan = validate_sam31_preparation_plan(read(plan_path, digest_field="plan_digest"),
                                         source_commit=old_commit, approved_roots=roots)
    profile_path = _ref(value["source_profile"], roots)
    require(sha(profile_path) == plan["server_profile_sha256"], "sam31_adoption_source_profile_changed")
    profile = _profile(profile_path, old_commit)
    inputs, inherited = _seed(plan, profile, roots)
    rows = value["phase_records"]
    start = inherited["phase_count"] if inherited else 0
    require(start < PREFIX_LENGTHS[value["through_phase"]], "sam31_adoption_prefix_not_extended")
    expected = list(PHASES[start:PREFIX_LENGTHS[value["through_phase"]]])
    require([row.get("phase") for row in rows] == expected, "sam31_adoption_prefix_incomplete")
    artifacts = dict(inherited["artifacts"]) if inherited else {}
    outcomes = dict(inherited["outcomes"]) if inherited else {}
    for row in rows:
        phase = row["phase"]
        job = read(_ref(row["job"], roots), digest_field="job_digest")
        result = read(_ref(row["result"], roots), digest_field="result_digest")
        receipt = read(_ref(row["execution_receipt"], roots), digest_field="receipt_digest")
        identities = {k: {key: ref[key] for key in ("sha256", "size_bytes")} for k, ref in inputs.items()}
        key = {"parent_request_digest": value["original_parent_request_digest"],
               "plan_digest": value["source_plan"]["sha256"], "phase": phase,
               "inputs_digest": canonical_digest(identities)}
        child_id = "sam31-" + canonical_digest(key).removeprefix("sha256:")
        require(job.get("schema_version") == "task_evaluation_sam31_preparation_execution_job.v1"
                and job.get("child_id") == child_id and job.get("expected_source_commit") == old_commit
                and job.get("inputs") == inputs and all(job.get(k) == v for k, v in key.items())
                and job.get("plan_ref") == value["source_plan"], "sam31_adoption_job_changed")
        require(result.get("schema_version") == "task_evaluation_sam31_preparation_execution_result.v1"
                and result.get("source_commit") == old_commit and result.get("child_id") == child_id
                and result.get("job_digest") == job["job_digest"] and result.get("status") == "completed"
                and all(result.get(k) == key[k] for k in ("phase", "parent_request_digest", "plan_digest")),
                "sam31_adoption_terminal_result_invalid")
        require(receipt.get("schema_version") == "task_evaluation_sam31_phase_execution_receipt.v1"
                and receipt.get("source_commit") == old_commit and receipt.get("job_digest") == job["job_digest"]
                and receipt.get("phase") == phase and receipt.get("outcome", {}).get("status") == "completed"
                and receipt["outcome"].get("artifacts") == result.get("artifacts")
                and bool(result.get("artifacts")), "sam31_adoption_execution_receipt_invalid")
        for name, ref in inputs.items():
            _ref(ref, roots)
        for name, ref in result["artifacts"].items():
            _ref(ref, roots)
            require(name not in inputs or inputs[name] == ref, "sam31_adoption_artifact_conflict")
            inputs[name] = artifacts[name] = ref
        if phase == "standard_splat_conversion":
            inputs["standard_splat_conversion"] = artifacts["standard_splat_conversion_receipt"]
            artifacts["standard_splat_conversion"] = inputs["standard_splat_conversion"]
        outcomes[phase] = receipt["outcome"]
    tracking_origin = (inherited["tracking_origin"] if inherited else
                       {"profile": profile, "commit": old_commit})
    return plan, profile, artifacts, outcomes, tracking_origin


def validate_completed_prefix_adoption(path, *, expected_source_commit, approved_roots,
                                      current_plan=None, current_provider_profile_path=None):
    roots = tuple(Path(root) for root in approved_roots)
    value = deepcopy(path) if isinstance(path, dict) else read(path, digest_field="adoption_digest")
    require(value.get("adoption_digest") == canonical_digest(value, digest_field="adoption_digest"), "sam31_adoption_digest_invalid")
    require(value.get("schema_version") == SCHEMA and value.get("status") == "verified_completed_prefix"
            and value.get("source_commit") == expected_source_commit and value.get("through_phase") in PREFIX_LENGTHS
            and value.get("historical_receipts_modified") is False and value.get("paid_execution_performed") is False
            and value.get("candidate_policy_queried") is False, "sam31_adoption_contract_invalid")
    _zero(_ref(value["provider_zero_at_adoption"], roots), at=value["created_at_epoch"])
    old_plan, old_profile, artifacts, outcomes, tracking_origin = _phase_chain(value, roots)
    old_task, old_source, old_science = source_science(old_plan["host_inputs"], value["original_execution_commit"])
    current_host = value["current_host_inputs"]
    require(set(current_host) == set(old_plan["host_inputs"]), "sam31_adoption_current_inputs_invalid")
    for row in current_host.values():
        _ref(row, roots)
    task, source, science = source_science(current_host, expected_source_commit)
    require(task.get("expected_production_commit") == expected_source_commit
            and old_task.get("expected_production_commit") == value["original_execution_commit"]
            and task_science(task) == task_science(old_task) and science == old_science,
            "sam31_adoption_task_or_source_changed")
    require(current_host["interiorgs_terms"]["sha256"] == old_plan["host_inputs"]["interiorgs_terms"]["sha256"],
            "sam31_adoption_terms_changed")
    current_conversion = validate_current_rights(task, source, current_host, expected_source_commit, roots)
    require(all(current_conversion["standard"][k] == artifacts["standard_splat"][k] for k in ("sha256", "size_bytes")),
            "sam31_adoption_standard_source_changed")
    if current_plan is not None:
        require(current_plan["host_inputs"] == current_host and current_plan["source_commit"] == expected_source_commit
                and all(current_plan[k] == old_plan[k] for k in
                        ("task_identity", "scene_identity", "publisher_scene_id", "rendering", "mask_policy", "claim_boundary"))
                and camera_science(current_plan["camera_policy"]) == camera_science(old_plan["camera_policy"]),
                "sam31_adoption_plan_science_changed")
    from .public_scene_inpainting_inputs import _git_identity
    current_repo = Path(value["current_release_root"])
    require(_git_identity(current_repo)["commit"] == expected_source_commit, "sam31_adoption_current_release_changed")
    tracking_phase = "sam31_tracking" if PREFIX_LENGTHS[value["through_phase"]] >= 5 else "calibrated_views"
    release = validate_render(outcomes["calibrated_views"], _render_artifacts(artifacts), old_plan, current_repo, tracking_phase)
    require(release == value["retained_release_pin"], "sam31_adoption_retained_release_changed")
    provider_path = _ref(value["current_sam31_provider_profile"], roots)
    if current_provider_profile_path is not None:
        require(record(current_provider_profile_path) == value["current_sam31_provider_profile"], "sam31_adoption_current_model_changed")
    if PREFIX_LENGTHS[value["through_phase"]] >= 5:
        tracking = validate_tracking(outcomes["sam31_tracking"], artifacts, tracking_origin["profile"], provider_path,
                                     tracking_origin["commit"], _ref(value["sam31_billing_source"], roots))
        require(tracking == value["tracking_identity"], "sam31_adoption_tracking_identity_changed")
    if value["through_phase"] == "segment_cutout":
        from .task_evaluation_sam31_preparation_paid_stages import validate_retained_paid_stage
        validate_retained_paid_stage(outcomes["contribution_sweep"], stage_id="contribution_sweep")
        for relative in (
            "scripts/adp_gaussian_excision_provider_runner.py",
            "src/blueprint_pipeline/public_scene_gaussian_excision_audit.py",
            "src/blueprint_pipeline/public_scene_calibrated_object_masks.py",
            "src/blueprint_pipeline/public_scene_segment_contribution_cutout.py",
            "src/blueprint_pipeline/task_evaluation_sam31_preparation_review_stages.py",
            "src/blueprint_pipeline/task_evaluation_sam31_preparation_profile.py",
        ):
            before = Path(old_profile["repo_root"]) / relative
            require(before.is_file() and sha(before) == sha(current_repo / relative),
                    "sam31_adoption_producer_code_changed:" + relative)
    # A canonical parent envelope must still join the old immutable plan and child chain.
    from .task_evaluation_launch_preparation_contract import validate_launch_preparation_request, launch_preparation_request_digest
    envelope = read(_ref(value["original_parent_envelope"], roots), digest_field="envelope_digest")
    parent = validate_launch_preparation_request(envelope["request"])
    require(launch_preparation_request_digest(parent) == envelope["request_digest"] == value["original_parent_request_digest"]
            and parent["expected_production_commit"] == value["original_execution_commit"]
            and parent["scene"]["identity"] == old_plan["scene_identity"]
            and parent["task"]["identity"] == old_plan["task_identity"], "sam31_adoption_parent_changed")
    require(any(row.get("source", {}).get("digest") == value["source_plan"]["sha256"]
                and row["source"].get("size_bytes") == value["source_plan"]["size_bytes"]
                for row in parent["runtime"]["mounts"]), "sam31_adoption_parent_plan_unbound")
    for row in value["phase_records"]:
        require(read(row["job"]["path"])["parent_preparation_id"] == parent["preparation_id"],
                "sam31_adoption_parent_identity_changed")
    # Future contribution disclosure is bound to the successor conversion
    # receipt. Historical render/SAM receipts continue to point at the old one.
    rebindings = {name: {"original": artifacts[name], "successor": current_conversion[key]}
                  for name, key in (("standard_splat", "standard"),
                                    ("standard_splat_conversion_receipt", "conversion"),
                                    ("standard_splat_conversion", "conversion"))}
    require(value.get("administrative_rebindings") == rebindings, "sam31_adoption_conversion_rebinding_changed")
    current_artifacts = {**artifacts, **{name: row["successor"] for name, row in rebindings.items()}}
    return {"record": deepcopy(value), "artifacts": current_artifacts,
            "phase_count": PREFIX_LENGTHS[value["through_phase"]], "outcomes": outcomes,
            "tracking_origin": tracking_origin}


def publish_adoption_release_binding(adoption_path, *, binding_root=None):
    """Publish the existing retention schema after a validated adoption exists."""
    from .task_evaluation_release_retention import (
        DEFAULT_EVIDENCE_BINDING_ROOT, EVIDENCE_BINDING_SCHEMA_VERSION, _write_exclusive,
    )
    value = read(adoption_path, digest_field="adoption_digest")
    require(value.get("schema_version") == SCHEMA and value.get("status") == "verified_completed_prefix",
            "sam31_adoption_retention_binding_invalid")
    root = Path(binding_root) if binding_root is not None else DEFAULT_EVIDENCE_BINDING_ROOT
    require(root.is_absolute() and root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)),
            "sam31_adoption_retention_root_invalid")
    profile = read(value["source_profile"]["path"], digest_field="profile_digest")
    retained_release = value["retained_release_pin"]
    if profile.get("completed_prefix_adoption") is not None:
        publish_adoption_release_binding(profile["completed_prefix_adoption"]["path"], binding_root=root)
        from .public_scene_inpainting_inputs import _git_identity
        source_repo = Path(profile["repo_root"])
        identity = _git_identity(source_repo)
        require(identity["commit"] == value["original_execution_commit"], "sam31_adoption_original_release_changed")
        retained_release = {"path": str(source_repo), "source_commit": identity["commit"], "tree": identity["tree"]}
    binding = {"schema_version": EVIDENCE_BINDING_SCHEMA_VERSION, "status": "required",
               "source_commit": value["original_execution_commit"],
               "reason": "Completed SAM prefix replay requires its original immutable renderer release",
               "evidence": record(adoption_path), "retained_release": retained_release}
    target = root / ("sam31-prefix-" + value["adoption_digest"].removeprefix("sha256:") + ".json")
    if target.exists():
        require(read(target) == binding, "sam31_adoption_retention_binding_conflict")
    else:
        _write_exclusive(target, binding)
    return record(target)


def materialize_completed_prefix_adoption(*, source_plan_path, source_profile_path, parent_request_digest,
    through_phase, current_host_inputs, current_provider_profile_path, current_repo_root,
    expected_source_commit, provider_zero_path, output_path, approved_roots,
    queue_root=DEFAULT_QUEUE, parent_queue_root=DEFAULT_PARENT_QUEUE, execution_root=DEFAULT_EXECUTION, now_epoch=None,
    sam31_billing_source_path=None, release_binding_root=None):
    source_plan_path, source_profile_path, current_provider_profile_path, current_repo_root, provider_zero_path = (
        Path(path) for path in (source_plan_path, source_profile_path, current_provider_profile_path,
                               current_repo_root, provider_zero_path))
    require(through_phase in PREFIX_LENGTHS, "sam31_adoption_prefix_invalid")
    at = time.time() if now_epoch is None else now_epoch
    _zero(provider_zero_path, at=at)
    old_plan = read(source_plan_path, digest_field="plan_digest")
    old_profile = _profile(source_profile_path, old_plan["source_commit"])
    queue = Path(queue_root)
    rows = []
    roots = tuple(Path(root) for root in approved_roots)
    old_inputs, inherited = _seed(old_plan, old_profile, roots)
    start = inherited["phase_count"] if inherited else 0
    require(start < PREFIX_LENGTHS[through_phase], "sam31_adoption_prefix_not_extended")
    for phase in PHASES[start:PREFIX_LENGTHS[through_phase]]:
        # Derive the one exact child; unrelated historical failures cannot
        # poison adoption and no queue lookup may enqueue or overwrite work.
        identities = {name: {key: ref[key] for key in ("sha256", "size_bytes")}
                      for name, ref in old_inputs.items()}
        identity = {"parent_request_digest": parent_request_digest, "plan_digest": sha(source_plan_path),
                    "phase": phase, "inputs_digest": canonical_digest(identities)}
        child_id = "sam31-" + canonical_digest(identity).removeprefix("sha256:")
        result_path = queue / "results" / (child_id + ".json")
        result = read(result_path, digest_field="result_digest")
        require(result.get("status") == "completed", "sam31_adoption_prefix_not_terminal")
        job_path = queue / "completed" / (result["child_id"] + ".json")
        job = read(job_path, digest_field="job_digest")
        phase_receipt = Path(execution_root) / parent_request_digest.removeprefix("sha256:") / result["child_id"] / "phase_execution_receipt.v1.json"
        rows.append({"phase": phase, "job": record(job_path), "result": record(result_path), "execution_receipt": record(phase_receipt)})
        old_inputs.update(result["artifacts"])
        if phase == "standard_splat_conversion":
            old_inputs["standard_splat_conversion"] = old_inputs["standard_splat_conversion_receipt"]
    from .task_evaluation_sam31_preparation_execution import _parent
    _, state, parent_path = _parent(job, Path(parent_queue_root))
    require(state in {"blocked", "completed", "materialized"}, "sam31_adoption_parent_not_terminal")
    value = {"schema_version": SCHEMA, "status": "verified_completed_prefix", "created_at_epoch": at,
             "source_commit": expected_source_commit, "current_release_root": str(current_repo_root),
             "original_execution_commit": old_plan["source_commit"], "original_parent_request_digest": parent_request_digest,
             "original_parent_envelope": record(parent_path), "source_plan": record(source_plan_path),
             "source_profile": record(source_profile_path), "through_phase": through_phase, "phase_records": rows,
             "current_host_inputs": current_host_inputs, "current_sam31_provider_profile": record(current_provider_profile_path),
             "provider_zero_at_adoption": record(provider_zero_path), "historical_receipts_modified": False,
             "paid_execution_performed": False, "candidate_policy_queried": False}
    roots = tuple(Path(root) for root in approved_roots)
    _, _, artifacts, outcomes, tracking_origin = _phase_chain(value, roots)
    tracking_phase = "sam31_tracking" if PREFIX_LENGTHS[through_phase] >= 5 else "calibrated_views"
    value["retained_release_pin"] = validate_render(outcomes["calibrated_views"], _render_artifacts(artifacts), old_plan, Path(current_repo_root), tracking_phase)
    if PREFIX_LENGTHS[through_phase] >= 5:
        require(sam31_billing_source_path is not None, "sam31_adoption_official_billing_required")
        value["sam31_billing_source"] = record(sam31_billing_source_path)
        value["tracking_identity"] = validate_tracking(outcomes["sam31_tracking"], artifacts, tracking_origin["profile"],
                                                       current_provider_profile_path, tracking_origin["commit"], sam31_billing_source_path)
    task, source, _ = source_science(current_host_inputs, expected_source_commit)
    converted = validate_current_rights(task, source, current_host_inputs, expected_source_commit, roots)
    value["administrative_rebindings"] = {
        name: {"original": artifacts[name], "successor": converted[key]} for name, key in
        (("standard_splat", "standard"), ("standard_splat_conversion_receipt", "conversion"),
         ("standard_splat_conversion", "conversion"))}
    value["adoption_digest"] = canonical_digest(value, digest_field="adoption_digest")
    # Complete validation before publishing any success-looking record.
    validate_completed_prefix_adoption(value, expected_source_commit=expected_source_commit, approved_roots=roots)
    if output_path is not None:
        output = Path(output_path)
        require(not output.exists() and not output.is_symlink(), "sam31_adoption_output_exists")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(canonical_json(value) + "\n")
        publish_adoption_release_binding(output, binding_root=release_binding_root)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-plan", "source-profile", "parent-request-digest", "current-task-request", "current-installation-receipt",
                 "current-publisher-intake", "current-source-preparation-receipt", "current-interiorgs-terms",
                 "current-sam31-provider-profile", "current-repo-root", "source-commit", "provider-zero"):
        parser.add_argument("--" + name, required=True,
                            type=str if name in {"parent-request-digest", "source-commit"} else Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--through-phase", choices=tuple(PREFIX_LENGTHS), required=True)
    parser.add_argument("--approved-root", action="append", type=Path, required=True)
    parser.add_argument("--sam31-billing-source", type=Path)
    parser.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--parent-queue-root", type=Path, default=DEFAULT_PARENT_QUEUE)
    parser.add_argument("--execution-root", type=Path, default=DEFAULT_EXECUTION)
    args = parser.parse_args(argv)
    require(args.check_only != bool(args.output), "sam31_adoption_choose_output_or_check_only")
    host = {name: record(getattr(args, "current_" + name)) for name in
            ("task_request", "installation_receipt", "publisher_intake", "source_preparation_receipt", "interiorgs_terms")}
    result = materialize_completed_prefix_adoption(source_plan_path=args.source_plan, source_profile_path=args.source_profile,
        parent_request_digest=args.parent_request_digest, through_phase=args.through_phase, current_host_inputs=host,
        current_provider_profile_path=args.current_sam31_provider_profile, current_repo_root=args.current_repo_root,
        expected_source_commit=args.source_commit, provider_zero_path=args.provider_zero, output_path=args.output,
        approved_roots=args.approved_root, queue_root=args.queue_root, parent_queue_root=args.parent_queue_root,
        execution_root=args.execution_root, sam31_billing_source_path=args.sam31_billing_source)
    print(canonical_json({"status": result["status"], "adoption_digest": result["adoption_digest"], "output": str(args.output) if args.output is not None else None,
                          "paid_execution_performed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
