"""Explicitly authorized new four-arm retry; cumulative ledger never reset."""

import argparse
from decimal import Decimal
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import time

from blueprint_pipeline.agent_execution.contracts import AgentExecutionError
from blueprint_pipeline.agent_execution.journal import AgentJournal
from experiments.provider_eval_recovery.harness import Ledger, digest, exclusive, read_json, write_once
from experiments.provider_eval_recovery.live_http import credential_presence
from . import soft_pilot as base
from .hosted import INSTRUCTIONS

PROTOCOL = "hosted_agent_research_v2_retry1"
APPROVAL = {"assistant": "Sentinel_f8a3006241c88191bcb27ec7e2121aba",
    "proposal": "retry the four-mode Chef pilot; $2 soft total hosted target including failed attempt; original $10 ledger retained; not a hard cap",
    "user": "Sentinel_c4c056bcbf408191850ec0dd9e86f6c5", "answer": "yes"}
TARGET, STOP = Decimal("2.00"), Decimal("1.90")
MODEL_HOLD, CONTAINER_HOLD = Decimal("0.26"), Decimal("0.09")
BASE_HOLD = 4 * (MODEL_HOLD + CONTAINER_HOLD) + base.REVIEW_HOLD
PRIOR_SESSION = "sess_0efe74876b1df856006abdb27f210881979445fa9da044cf87"
SELECTIVE = """Use selective inspection to conserve cumulative context and cost. Start
with the case and index metadata; choose relevant source IDs and narrow ranges.
Use literal find/offset/range reads or hosted Python for relevant sections. Return
compact extracts, source IDs and findings rather than dumping the whole corpus or
printing full files repeatedly. All retained text remains accessible at every
offset; inspect more when the evidence requires it. Keep the same source verification,
task criteria and explicit unknowns. No fixed prefix limits or inaccessible evidence.
""" + INSTRUCTIONS


def prior_state(root, owner, prior_sha, *, clock=time.time):
    old = base.validate_receipt(root, owner, prior_sha, readonly_previous=True)
    monitor = base.SoftMonitor(root, old, clock=clock, notify=lambda _: None)
    if not (monitor.path / "stop.json").is_file():
        raise AgentExecutionError("original_pilot_must_remain_stopped")
    report = monitor.report()
    sessions = report["sessions"]
    journal = AgentJournal(monitor.path / "agent_journal")
    frozen = {monitor.task_id(mode) for mode in base.MODES}
    if (len(sessions) != 1 or sessions[0]["session_id"] != PRIOR_SESSION
            or sessions[0]["cleanup"] != "approved_deleted_api_absence_observed"
            or any(t["task_id"] not in frozen for t in journal.tasks(active_only=False, limit=10))
            or journal.unsettled_operations(sessions[0]["task_id"])
            or not isinstance(sessions[0]["usage"], dict)):
        raise AgentExecutionError("exact_prior_deleted_trace_and_offline_cleanup_adoption_required")
    proof = journal.event("hosted_cleanup_" + sessions[0]["task_id"])
    # Old unused allocations stay in the $10 ledger; they are not billed usage.
    # This total soft target includes prior observed model high-water plus its
    # retained cleanup buffer, without calling the old bill settled or zero.
    prior_soft = (Decimal(sessions[0]["model_high_water_estimate_usd"])
                  + max(base.CONTAINER_HOLD, Decimal(sessions[0]["container_elapsed_estimate_usd"])))
    return {"prior_approval_sha256": prior_sha, "prior_cleanup_proof_sha256": digest(proof),
            "prior_deleted_session_id": PRIOR_SESSION, "prior_hosted_soft_exposure_usd": str(prior_soft),
            "prior_soft_basis": "observed_max_cachewrite_model_plus_retained_cleanup_buffer; unused allocations remain reserved in aggregate",
            "old_baseline_reserved_usd": old["baseline_reserved_usd"]}


def receipt_path(root):
    return Path(root) / "protocols" / PROTOCOL / "soft_pilot_approval.json"


def prepare_receipt(root, owner, prior_sha, *, clock=time.time):
    root = Path(root).resolve()
    public, access_sha, scope_sha = base.scope_inputs(root, owner)
    previous = prior_state(root, owner, prior_sha, clock=clock)
    path = receipt_path(root)
    if path.exists():
        saved = read_json(path)
        return validate_receipt(root, owner, digest(saved), clock=clock), digest(saved)
    commit, code_sha = base.code_identity()
    with exclusive(root):
        ledger = Ledger(root / "live_journal.jsonl", "10.00")
        if ledger.exposure < Decimal("4.765920"):
            raise AgentExecutionError("prior_4_765920_checkpoint_not_preserved")
        if any(s not in {"completed", "not_accepted"} for s in ledger.states.values()):
            raise AgentExecutionError("prior_inflight_or_uncertain_dispatch_blocks_retry")
        if any(e.get("protocol") == PROTOCOL for e in ledger.events):
            raise AgentExecutionError("retry_already_used_no_new_baseline")
        if ledger.exposure + BASE_HOLD + base.SEARCH_HOLD > 10:
            raise AgentExecutionError("original_aggregate_reservation_ceiling_exhausted")
        projected = Decimal(previous["prior_hosted_soft_exposure_usd"]) + BASE_HOLD + base.SEARCH_HOLD
        if projected >= STOP:
            raise AgentExecutionError("cumulative_hosted_soft_target_has_no_retry_headroom")
        receipt = {"schema": "hosted_one_case_soft_retry.v1", "approval": APPROVAL, "protocol": PROTOCOL,
            "case_id": public["cases"][0]["id"], "case_sha256": digest(public["cases"][0]),
            "public_inputs_sha256": base.DECLARED_ORIGINAL_SHA256, "arms": list(base.MODES),
            "model": base.MODEL, "project": base.PROJECT, "execution_owner": base.OWNER, "catalog": base.CATALOG,
            "root": str(root), "source_commit": commit, "code_sha256": code_sha,
            "access_sha256": access_sha, "original_scope_sha256": scope_sha, **previous,
            "baseline_events": len(ledger.events), "baseline_head": ledger.previous,
            "baseline_reserved_usd": str(ledger.exposure), "soft_target_usd": str(TARGET),
            "stop_threshold_usd": str(STOP), "model_hold_usd_per_arm": str(MODEL_HOLD),
            "container_hold_usd_per_arm": str(CONTAINER_HOLD), "retry_base_hold_usd": str(BASE_HOLD),
            "hard_cap": False, "no_new_credentials_or_access_grants": True,
            "per_arm_seconds": base.ARM_SECONDS, "max_function_calls": 48, "poll_seconds": base.POLL_SECONDS,
            "permanent_deletion_authorized": False, "container_expiry_guaranteed": False,
            "sdk_versions": {"openai": base.SDK_VERSION, "openai-agents": base.AGENTS_SDK_VERSION},
            "selective_instructions_sha256": digest(SELECTIVE), "max_input_tokens_target": 50000,
            "max_output_tokens_target": 2048, "initial_projected_soft_total_usd": str(projected),
            "created_at": clock(), "expires_at": clock() + 3600}
        write_once(path, receipt)
    return receipt, digest(receipt)


def validate_receipt(root, owner, expected_sha, *, clock=time.time):
    root = Path(root).resolve()
    public, access_sha, scope_sha = base.scope_inputs(root, owner)
    saved = read_json(receipt_path(root))
    if digest(saved) != expected_sha:
        raise AgentExecutionError("exact_retry_receipt_hash_required")
    previous = prior_state(root, owner, saved.get("prior_approval_sha256"), clock=clock)
    commit, code_sha = base.code_identity()
    required = {"schema": "hosted_one_case_soft_retry.v1", "approval": APPROVAL, "protocol": PROTOCOL,
        "case_id": "BP-EVAL-01", "case_sha256": digest(public["cases"][0]), "arms": list(base.MODES),
        "public_inputs_sha256": base.DECLARED_ORIGINAL_SHA256, "model": base.MODEL, "project": base.PROJECT,
        "execution_owner": base.OWNER, "catalog": base.CATALOG, "root": str(root),
        "source_commit": commit, "code_sha256": code_sha, "access_sha256": access_sha,
        "original_scope_sha256": scope_sha, **previous, "soft_target_usd": str(TARGET),
        "stop_threshold_usd": str(STOP), "model_hold_usd_per_arm": str(MODEL_HOLD),
        "container_hold_usd_per_arm": str(CONTAINER_HOLD), "retry_base_hold_usd": str(BASE_HOLD),
        "hard_cap": False, "no_new_credentials_or_access_grants": True, "per_arm_seconds": base.ARM_SECONDS,
        "max_function_calls": 48, "poll_seconds": base.POLL_SECONDS, "permanent_deletion_authorized": False,
        "container_expiry_guaranteed": False, "selective_instructions_sha256": digest(SELECTIVE),
        "max_input_tokens_target": 50000, "max_output_tokens_target": 2048,
        "sdk_versions": {"openai": base.SDK_VERSION, "openai-agents": base.AGENTS_SDK_VERSION},
        "initial_projected_soft_total_usd": str(Decimal(previous["prior_hosted_soft_exposure_usd"]) + BASE_HOLD + base.SEARCH_HOLD)}
    if any(saved.get(k) != v for k, v in required.items()):
        raise AgentExecutionError("exact_reviewed_retry_scope_required")
    ledger = Ledger(root / "live_journal.jsonl", "10.00")
    n = saved.get("baseline_events")
    if type(n) is not int or not 1 <= n <= len(ledger.events):
        raise AgentExecutionError("retry_original_journal_prefix_changed")
    states = {e["attempt_id"]: e["kind"] for e in ledger.events[:n]}
    held = sum((Decimal(e["amount_usd"]) for e in ledger.events[:n] if e["kind"] == "reserved"
                and states[e["attempt_id"]] != "not_accepted"), Decimal(0))
    if (ledger.events[n - 1]["sha256"] != saved["baseline_head"] or str(held) != saved["baseline_reserved_usd"]
            or held < Decimal("4.765920") or any(e.get("protocol") == PROTOCOL for e in ledger.events[:n])
            or any(s not in {"completed", "not_accepted"} for s in states.values())
            or any(e.get("soft_receipt") != expected_sha for e in ledger.events
                   if e.get("protocol") == PROTOCOL and e.get("role") == "soft_pilot_allowance")
            or type(saved.get("created_at")) not in {int, float}
            or saved.get("expires_at") != saved["created_at"] + 3600):
        raise AgentExecutionError("retry_original_journal_prefix_changed")
    return saved


class RetryMonitor(base.SoftMonitor):
    target, stop_threshold = TARGET, STOP
    model_hold, container_hold, base_hold = MODEL_HOLD, CONTAINER_HOLD, BASE_HOLD
    protocol = PROTOCOL
    task_options = {"protocol": PROTOCOL, "instructions": SELECTIVE, "max_input_tokens": 50000, "max_output_tokens": 2048}

    def __init__(self, root, receipt, **kwargs):
        super().__init__(root, receipt, **kwargs)
        self.path = self.root / "protocols" / PROTOCOL / "soft_pilot"
        self.evidence_root = self.root / "protocols" / PROTOCOL / "evidence"

    def task_id(self, mode):
        return "hosted_retry1_01_" + mode + "_" + self.sha[:16]

    def search_accounting_events(self, ledger):
        return ledger.events[self.receipt["baseline_events"]:]

    def prior_soft_exposure(self):
        # Revalidate the old deletion/usage anchor before any new paid dispatch.
        previous = prior_state(self.root, base.OWNER, self.receipt["prior_approval_sha256"], clock=self.clock)
        if any(self.receipt.get(k) != v for k, v in previous.items()):
            self.stop("prior_hosted_evidence_changed")
            raise AgentExecutionError("prior_hosted_evidence_changed")
        return Decimal(previous["prior_hosted_soft_exposure_usd"])

    def guard(self, *, preparing_creation=None):
        if digest(self.receipt) != self.sha:
            self.stop("retry_in_memory_scope_changed")
            raise AgentExecutionError("retry_in_memory_scope_changed")
        validate_receipt(self.root, base.OWNER, self.sha, clock=self.clock)
        return super().guard(preparing_creation=preparing_creation)


def run_retry(root, receipt, *, clock=time.time, sleep=time.sleep, notify=print, factory=base.make_runtime):
    if validate_receipt(root, base.OWNER, digest(receipt), clock=clock) != receipt:
        raise AgentExecutionError("exact_retry_receipt_required")
    monitor = RetryMonitor(root, receipt, clock=clock, notify=notify)
    monitor.reserve()
    # Share the original owner's lock too; independent cohorts cannot overlap.
    old_path = Path(root).resolve() / "protocols" / base.PROTOCOL / "soft_pilot"
    with exclusive(old_path), exclusive(monitor.path):
        return base._run_owned(root, receipt, monitor, sleep=sleep, notify=notify, factory=factory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-root", required=True, type=Path)
    parser.add_argument("--execution-owner-task-id", required=True)
    parser.add_argument("--prior-approval-sha256")
    parser.add_argument("--retry-receipt-sha256")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-retry", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--status", action="store_true")
    args = parser.parse_args()
    root = args.aggregate_root.resolve()
    try:
        if args.prepare_retry:
            receipt, sha = prepare_receipt(root, args.execution_owner_task_id, args.prior_approval_sha256)
            print(json.dumps({"receipt": receipt, "retry_receipt_sha256": sha, "new_paid_calls": 0}, indent=2))
            return
        receipt = validate_receipt(root, args.execution_owner_task_id, args.retry_receipt_sha256)
        if args.status:
            print(json.dumps(RetryMonitor(root, receipt).report(), indent=2))
            return
        if version("openai") != base.SDK_VERSION or version("openai-agents") != base.AGENTS_SDK_VERSION:
            raise AgentExecutionError("install_reviewed_sdk_pair_openai_3.22.1_openai_agents_0.22.3")
        if not all(credential_presence().values()):
            raise AgentExecutionError("existing_bindings_not_present_no_credential_creation")
        result = run_retry(root, receipt)
        if result["budget"]["stopped"] or len(result["outcomes"]) != 4:
            raise SystemExit(2)
    except (AgentExecutionError, ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        reason = str(exc) if isinstance(exc, AgentExecutionError) else "local_retry_scope_or_journal_validation_failed"
        print(json.dumps({"status": "blocked_or_stopped", "reason": reason}))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
