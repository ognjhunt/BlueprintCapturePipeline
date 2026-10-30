"""Bind native G1 to the existing beta launch intent and scoped reconciliation."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from .decision_evidence_contracts import canonical_digest
from .native_g1_beta_session import build_session_authority, PROBE_KIND


def reopen_candidate(plan, candidate):
    from .beta_policy_screening_launch import _bound_path, _read
    from .beta_policy_screening_g1 import candidate_execution_spec, validate_execution_spec
    execution = candidate["execution_spec"]
    native = validate_execution_spec(_read(_bound_path(execution["policy_execution_spec"])), candidate_id=candidate["candidate_id"])
    scene = _read(_bound_path(plan["scene_inputs"]))
    base = _bound_path(scene["base_native_packet"]).parent
    actual = candidate_execution_spec(candidate_id=candidate["candidate_id"],
        registration={key: record for key, record in native["native_g1"].items() if key != "model_terms_review"},
        scene_plan=_read(base / "native_task_arena_scene_plan.v1.json"), base_packet=scene["base_native_packet"],
        source=plan["source"], contract=native["task_success_contract"], site_id=plan["site_id"])
    if (actual != native or native["runtime_identity_digest"] != execution["runtime_digest"]
            or native["task_success_contract_digest"] != execution["task_success_contract_digest"]
            or execution["configuration_digest"] != candidate["configuration_digest"]
            or canonical_digest(execution) != candidate["admission_digest"]):
        raise ValueError("g1_beta_prepared_execution_changed")
    return native


def prepare_candidate(*, plan, candidate, attempt_dir):
    from .beta_policy_screening_launch import (_identity, _check_quote, _read, _stage, _write,
        _record, _validate_prepared)
    from .native_g1_beta_bundle import build_bundle, load_verified_bundle
    root = Path(attempt_dir).resolve()
    identity = _identity(plan, candidate)
    _check_quote(plan, candidate)
    if (root / "prepared.json").exists():
        prepared = _read(root / "prepared.json")
        _validate_prepared(plan, candidate, root, prepared)
        return prepared
    execution = candidate["execution_spec"]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True, check=True).stdout.strip()
    if commit != execution["implementation_commit"]:
        raise ValueError("g1_beta_implementation_commit_changed")
    native = reopen_candidate(plan, candidate)
    authority = build_session_authority(plan=plan, candidate=candidate, spec=native)
    authority_path = _stage(root / "runtime_inputs/session_authority.json", authority)
    bundle_record = root / "bundle_receipt.json"
    if bundle_record.exists():
        bundle = load_verified_bundle(bundle_record, expected_implementation_commit=commit)
    else:
        completed = root / "bundle/native_g1_beta_provider_bundle.v1.json"
        if completed.exists():
            bundle = load_verified_bundle(completed, expected_implementation_commit=commit)
        else:
            if (root / "bundle").exists():
                # Retain interrupted unpaid staging; a new directory cannot
                # overwrite bytes that another preparation may be inspecting.
                import uuid
                (root / "bundle").rename(root / ("bundle.interrupted-" + uuid.uuid4().hex))
            bundle = build_bundle(job_dir=root / "bundle", authority=authority, spec=native, implementation_commit=commit)
        _stage(bundle_record, bundle)
    argv = ["gpu-canary", "--provider", "vast", "--probe-kind", PROBE_KIND,
        "--g1-beta-session-authority", str(authority_path), "--g1-beta-session-bundle-receipt", str(bundle_record),
        "--adp-job-dir", str(root / "allocator"), "--adp-max-hourly-rate-usd", str(execution["max_hourly_rate_usd"]),
        "--adp-max-spend-usd", str(authority["hard_cap_usd"]), "--adp-hard-ttl-seconds", str(authority["hard_ttl_seconds"]),
        "--admission-out", str(root / "paid_admission.json"), "--adapter-output", str(root / "allocator_result.json"), "--execute"]
    if execution.get("machine_avoidlist_path"):
        argv += ["--adp-machine-avoidlist", str(execution["machine_avoidlist_path"])]
    prepared = {"schema_version": "beta_policy_screening_prepared.v1", **identity, "run_id": authority["run_id"],
        "attempt_dir": str(root), "argv": argv, "authority": _record(authority_path), "bundle_receipt": _record(bundle_record)}
    prepared["prepared_digest"] = canonical_digest(prepared, digest_field="prepared_digest")
    _write(root / "prepared.json", prepared)
    return prepared


def join_result(*, plan, candidate, adapter, prepared, receipt, adapter_path):
    from .beta_policy_screening_closeout import scoped_closeout_verified, SCOPE
    from .beta_policy_screening_launch import _read, _bound_path
    from .policy_canary_staged_object_absence import closeout_staged_objects
    from .native_g1_beta_output import verify_output
    from .beta_policy_screening_g1 import summarize_result
    if (not scoped_closeout_verified(receipt, adapter=adapter, prepared=prepared, adapter_path=adapter_path)
            or closeout_staged_objects(adapter)["blockers"]):
        raise ValueError("g1_beta_result_scoped_closeout_unverified")
    watchdog = adapter.get("independent_watchdog_close") or {}
    if (watchdog.get("status") != "provider_terminal" or watchdog.get("provider_absence_confirmed") is not True
            or watchdog.get("instance_ids") != [receipt["instance_id"]]
            or type(watchdog.get("watchdog_process_exit_code")) is not int):
        raise ValueError("g1_beta_result_watchdog_not_terminal")
    bundle = _read(_bound_path(prepared["bundle_receipt"]))
    path = Path(str(adapter.get("native_control_result_path") or ""))
    if not path.is_absolute() or not path.resolve().is_relative_to(Path(prepared["attempt_dir"])):
        raise ValueError("g1_beta_result_foreign_path")
    native = verify_output(output_dir=path.parent, bundle=bundle)
    if native["source_result_digest"] != adapter.get("native_control_result_digest"):
        raise ValueError("g1_beta_result_adapter_digest_changed")
    closed = {"provider_zero_confirmed": True, "staged_objects_deleted": True, "watchdog_cancelled": True,
        "scope": SCOPE, "global_provider_zero_claimed": False, "provider_absence_receipt_digest": receipt["receipt_digest"],
        "watchdog_close_digest": canonical_digest(watchdog), "authority_digest": bundle["authority_digest"]}
    closed["receipt_digest"] = canonical_digest(closed, digest_field="receipt_digest")
    joined = json.loads(json.dumps(native))
    joined["session_closeout"] = closed
    joined["result_digest"] = canonical_digest(joined, digest_field="result_digest")
    summarize_result(joined, plan=plan, candidate=candidate)
    return joined
