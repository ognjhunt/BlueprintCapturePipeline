"""Stage only unscored cells of the isolated diagnostic; never allocate a GPU.

Preserve original session bytes, frozen cells and credential. Closeout can
replace only an explicitly recorded reset refusal before any policy query.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from blueprint_pipeline import internal_team_evaluation_diagnostic as diagnostic
from blueprint_pipeline import team_candidate_quick10_launch as launch
from blueprint_pipeline.controlled_policy_configuration import canonical_request_digest
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_session import validate_session_result
from blueprint_pipeline.policy_canary_media_integrity import require_completed_episode_media
from blueprint_pipeline.policy_canary_provider_null_closeout import provider_allocation_fields
from blueprint_pipeline.policy_canary_retained_billing import adapter_instance_ids
from blueprint_pipeline.task_evaluation_scene_execution_authority import proves_no_provider_allocation
from blueprint_pipeline.team_candidate_episode_recovery import recovery_arguments
from blueprint_pipeline.team_candidate_quick10_bundle import assert_secret_absent
from blueprint_pipeline.team_candidate_quick10_closeout import join_team_session_closeout, validate_native_result_binding
from blueprint_pipeline.team_candidate_quick10_registry import REGISTRY_ENV, offer_eligibility
from blueprint_pipeline.vast_official_billing_extractor import extract_vast_official_instance_charge


def verify_media(native: Path, rows: list[dict]) -> dict:
    """Compare retained PNG pixels with losslessly restored policy requests."""
    import numpy as np
    from PIL import Image
    from blueprint_pipeline.droid_https_policy_client import IMAGE_KEYS
    from blueprint_pipeline.policy_request_evidence import restore_request, validate_request_evidence

    counts = {"episodes": 0, "policy_request_frame_pairs": 0, "review_videos": 0}
    for index, row in enumerate(rows):
        if row["status"] != "completed":
            continue
        cell = native.parent / "cell_runs" / f"{index:02d}"
        child = diagnostic._read(cell / native.name)
        child_row = child["episodes"][0]
        if ({key: value for key, value in child_row.items() if key != "evidence_artifacts"}
                != {key: value for key, value in row.items() if key != "evidence_artifacts"}):
            raise ValueError("diagnostic_retained_child_episode_mismatch")
        episode = row["episode"]
        require_completed_episode_media(cell, episode)
        frames, requests = episode["candidate_exact_policy_input_frames"], episode["policy_request_artifacts"]
        if len(frames) != len(requests) or len(frames) != episode["policy_queries"]:
            raise ValueError("diagnostic_retained_query_frame_count_mismatch")
        media = (cell / "episodes").resolve()
        for frame, request in zip(frames, requests, strict=True):
            path = media / frame["relative_path"]
            path.resolve().relative_to(media)
            if path.is_symlink() or "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() != frame["png_sha256"]:
                raise ValueError("diagnostic_retained_frame_changed")
            pixels = np.array(Image.open(path).convert("RGB"))
            receipt = validate_request_evidence(diagnostic._read(media / request["relative_path"]))
            observation = restore_request(receipt["request"])
            if (not np.array_equal(pixels, np.concatenate([observation[key] for key in IMAGE_KEYS], axis=1))
                    or "sha256:" + hashlib.sha256(pixels.tobytes()).hexdigest() != frame["raw_rgb_sha256"]):
                raise ValueError("diagnostic_retained_policy_pixels_mismatch")
            counts["policy_request_frame_pairs"] += 1
        counts["episodes"] += 1
        counts["review_videos"] += sum("video" in artifact["role"] for artifact in episode["media_artifacts"])
    return counts


def remaining_groups(cells: list[dict], retained: list[dict]) -> tuple[list[list[dict]], list[str]]:
    """Never rerun a scored cell. Put proven pre-policy refusals first."""
    by_id = {cell["cell_id"]: cell for cell in cells}
    completed, refused = set(), []
    observed = set()
    for row in retained:
        cell_id = row["cell_id"]
        if cell_id not in by_id or cell_id in observed:
            raise ValueError("diagnostic_retained_cell_not_unique_or_frozen")
        observed.add(cell_id)
        if any(row.get(key) != by_id[cell_id].get(key) for key in ("seed", "cell_spec_digest", "resolved_scenario_digest")):
            raise ValueError("diagnostic_retained_cell_identity_mismatch")
        if row["status"] == "completed":
            if (row.get("candidate_policy_queried") is not True or row.get("actions_reached_robot") is not True
                    or row.get("scoring_authority") != "deterministic_simulator_state"
                    or row["episode"]["score"].get("status") != "scored"
                    or type(row["episode"]["score"].get("task_succeeded")) is not bool):
                raise ValueError("diagnostic_retained_episode_not_scored")
            completed.add(cell_id)
        else:
            episode = row.get("episode") or {}
            if (row.get("status") != "blocked" or row.get("episode_failure_stage") != "before_first_observation"
                    or row.get("failure_message") != "scientific_reset_scenario_application_mismatch"
                    or any(row.get(key) is not False for key in
                           ("candidate_policy_query_attempted", "candidate_policy_queried", "actions_reached_robot"))
                    or episode.get("candidate_policy_query_attempted") is not False
                    or episode.get("policy_response_status") != "not_attempted"
                    or (episode.get("score") or {}).get("status") == "scored"):
                raise ValueError("diagnostic_refusal_not_proven_before_policy")
            refused.append(cell_id)
    pending = [by_id[key] for key in refused] + [cell for cell in cells
        if cell["cell_id"] not in completed and cell["cell_id"] not in refused]
    return [pending[index:index + 25] for index in range(0, len(pending), 25)], refused


def stage(prior: Path, root: Path, billing_path: Path, commit: str) -> dict:
    if launch._source_commit() != commit or root.exists():
        raise ValueError("diagnostic_resume_source_or_destination_invalid")
    old = diagnostic._read(prior / "diagnostic.json")
    job = Path(old["job_dir"])
    terminal = diagnostic._read(job / launch.TERMINAL_FILENAME)
    request = diagnostic._read(job / launch.REQUEST_FILENAME)
    admission = diagnostic._read(job / "team_candidate_admission.json")
    if (terminal.get("status") != "blocked" or terminal.get("provider_zero_verified") is not True
            or request.get("internal_diagnostic") is not True or request.get("customer_recommendation_permitted") is not False
            or admission.get("candidate_id") != diagnostic.CANDIDATE_ID):
        raise ValueError("diagnostic_resume_prior_scope_invalid")
    zero = launch._default_provider_zero()
    if zero.get("provider_zero_verified") is not True:
        raise ValueError("diagnostic_resume_provider_zero_unproven")
    intent = diagnostic._read(job / launch.INTENT_FILENAME)
    originals = [launch._current_shard(job, index) for index in range(intent["shard_count"])]
    registry = diagnostic._read(Path(old["registry_path"]))
    profile = registry["profiles"][0]
    os.environ[REGISTRY_ENV] = old["registry_path"]
    facts, blockers = offer_eligibility(profile)
    if facts is None or blockers:
        raise ValueError(";".join(blockers))
    cells = [cell for shard in originals for cell in shard["cells"]]
    if len(cells) != 200 or len({cell["cell_id"] for cell in cells}) != 200 or cells != launch._session_cells(
            purpose="scored", facts=facts, profile=profile):
        raise ValueError("diagnostic_resume_original_cells_changed")
    retained, adopted, media, absence, ids = [], [], [], [], set()
    records = []
    for role, path in [("prior_terminal", job / launch.TERMINAL_FILENAME),
                       ("original_request", job / launch.REQUEST_FILENAME),
                       ("original_admission", job / "team_candidate_admission.json")]:
        records.append({"role": role, **diagnostic._record(path)})
    for shard in originals:
        source = Path(shard["root"])
        if launch._shard_state(shard) != "returned":
            raise ValueError("diagnostic_resume_prior_allocator_not_returned")
        records.append({"role": "original_shard", **diagnostic._record(source / launch.SHARD_FILENAME)})
        adapter = diagnostic._read(source / "allocator_result.json")
        native = Path(str(adapter.get("native_control_result_path") or ""))
        if native.is_file():
            inner = diagnostic._read(native)
            validate_native_result_binding(inner, expected_digest=adapter["native_control_result_digest"])
            validate_session_result(join_team_session_closeout(inner=inner, adapter=adapter,
                provider_zero=zero, adapter_path=source / "allocator_result.json"))
            retained.extend(inner["episodes"])
            media.append({"native": diagnostic._record(native), **verify_media(native, inner["episodes"])})
            ids.update(adapter_instance_ids(adapter, result_path=source / "allocator_result.json",
                read_json=lambda path, code: diagnostic._read(path), error_factory=ValueError))
            adopted.append(shard)
            records.append({"role": "retained_native_session", **diagnostic._record(native)})
        else:
            nested = diagnostic._read(Path(adapter["adapter_result_path"]))
            fields = provider_allocation_fields(adapter=nested, watchdog=adapter["independent_watchdog_close"],
                teardown=diagnostic._read(Path(adapter["teardown_manifest_path"])),
                cleanup=diagnostic._read(Path(adapter["object_store_cleanup_path"])))
            if not proves_no_provider_allocation({**adapter, **fields}):
                raise ValueError("diagnostic_resume_unallocated_shard_unproven")
            absence.append({"original": diagnostic._record(source / "allocator_result.json"), "proven_fields": fields})
    groups, refused = remaining_groups(cells, retained)
    billing = diagnostic._read(billing_path)
    charges = billing["charges"]
    for charge in charges:
        if charge != extract_vast_official_instance_charge(
                provider_billing_source_receipt_path=Path(charge["provider_billing_source_receipt"]["path"]),
                instance_id=charge["provider_instance_id"], launch_label=charge["launch_label"]):
            raise ValueError("diagnostic_resume_official_charge_changed")
    charged_ids = {charge["provider_instance_id"] for charge in charges}
    settled = round(sum(charge["official_charge_usd"] for charge in charges), 6)
    caps = [2.6 if len(group) == 25 else .8 for group in groups]
    approval_path = Path(old["owner_approval"]["path"])
    if (diagnostic._record(approval_path) != old["owner_approval"] or len(charged_ids) != len(charges)
            or not ids.issubset(charged_ids) or settled != billing["official_total_usd"]
            or settled + sum(caps) > 20 or not refused):
        raise ValueError("diagnostic_resume_cumulative_budget_or_recovery_invalid")
    token = (prior / "policy-bearer").read_text().strip()
    credential = canonical_digest({"credential_sha256": hashlib.sha256(token.encode()).hexdigest()})
    if admission != diagnostic.internal_admission(facts=facts, approval=diagnostic._read(approval_path),
            policy_url=old["policy_url"], credential_digest=credential):
        raise ValueError("diagnostic_resume_original_admission_changed")
    root.mkdir(mode=0o700)
    newjob = root / "evaluation"
    newjob.mkdir(mode=0o700)
    (root / "scene").symlink_to(prior / "scene", target_is_directory=True)
    # Only the quote ceiling changes; scene/cell/policy/scoring identity is frozen.
    profile["max_hourly_rate_usd"] = .9
    registry["registry_digest"] = canonical_digest(registry, digest_field="registry_digest")
    registry_path = diagnostic._write(root / "registry.json", registry)
    os.environ[REGISTRY_ENV] = str(registry_path)
    diagnostic._write(newjob / launch.REQUEST_FILENAME, request)
    diagnostic._write(newjob / "team_candidate_admission.json", admission)
    private = newjob / ".team-credential"
    private.mkdir(mode=0o700)
    for path in (root / "policy-bearer", private / "bearer"):
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
            stream.write(token + "\n")
    for index, shard in enumerate(adopted):
        target = newjob / "shards" / f"{index:02d}" / "stage-00"
        (target / "bundle").mkdir(parents=True, mode=0o700)
        source = Path(shard["root"])
        for relative in ("allocator_result.json", "bundle/team_candidate_quick10_session_bundle_receipt.v1.json"):
            shutil.copy2(source / relative, target / relative)
        diagnostic._write(target / launch.SHARD_FILENAME, {**shard, "index": index, "root": str(target),
            "restage": 0, "original_shard": diagnostic._record(source / launch.SHARD_FILENAME)})
    probe = job / "robot_free_probe_retention.json"
    if probe.is_file():
        shutil.copy2(probe, newjob / probe.name)
    run_id = "internal-team-diag-recovery-" + commit[:12]
    for index, (group, cap) in enumerate(zip(groups, caps, strict=True), start=len(adopted)):
        shard = launch._stage_shard(job_dir=newjob, index=index, cells=group, admission=admission,
            facts=facts, profile=profile, cap_usd=cap, base_run_id=run_id, commit=commit)
        assert_secret_absent(shard["bundle_path"], token)
        print(json.dumps({"staged_shard": index, "cells": len(group), "cap_usd": cap}), flush=True)
    plan = {"schema_version": "team_candidate_pre_policy_reset_recovery_plan.v1",
        "canonical_request_digest": canonical_request_digest(request), "admission_digest": admission["admission_digest"],
        "source_records": records, "expected_cells": cells, "recovery_cell_ids": refused, "plan_digest": ""}
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    recovery_arguments(plan, request_digest=canonical_request_digest(request),
        admission_digest=admission["admission_digest"], episodes=200)
    diagnostic._write(newjob / launch.INTENT_FILENAME, {**intent, "source_commit": commit, "run_id": run_id,
        "shard_count": len(adopted) + len(groups), "pre_policy_reset_recovery": plan})
    diagnostic._write(newjob / "initial_provider_zero.json", zero)
    count = sum(item["episodes"] for item in media)
    proof = {"adopted_scored_episodes": count, "remaining_scored_episodes": 200 - count,
        "retained_media": media, "unallocated_originals": absence, "settled_billing": diagnostic._record(billing_path),
        "settled_cost_usd": settled, "future_allocation_caps_usd": caps,
        "maximum_total_cost_usd": round(settled + sum(caps), 6), "original_evidence_edited": False,
        "rerun_completed_episodes": 0, "provider_mutations": 0}
    proof_path = diagnostic._write(root / "recovery_adoption.json", proof)
    result = {**old, "source_commit": commit, "job_dir": str(newjob), "registry_path": str(registry_path),
        "recovery_adoption": diagnostic._record(proof_path), "adopted_scored_episodes": count,
        "remaining_scored_episodes": 200 - count, "cap_usd": 20.}
    diagnostic._write(root / "diagnostic.json", result)
    return {"status": "recovery_staged", "root": str(root), **proof}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-root", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--settled-billing", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.prior_root, args.root, args.settled_billing, args.source_commit)), flush=True)
