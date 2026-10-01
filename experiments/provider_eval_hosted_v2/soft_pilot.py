"""Explicit one-case soft-budget admission and sole-owner hosted pilot CLI."""

import argparse
from decimal import Decimal
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess
import time

from blueprint_pipeline.agent_execution.contracts import AgentAdmission, AgentExecutionError, AgentTask, digest as task_digest
from blueprint_pipeline.agent_execution.journal import AgentJournal, TERMINAL_STATES
from blueprint_pipeline.agent_execution.openai_transport import OpenAIAgentsHTTP
from blueprint_pipeline.agent_execution.operations import AgentOperations
from blueprint_pipeline.paid_resource_admission import (PAID_LANE_ADMISSION_SCHEMA_VERSION,
    require_paid_resource_admission, require_paid_resource_admission_grant)
from experiments.provider_eval_recovery.harness import Ledger, digest, exclusive, read_json, write_once
from experiments.provider_eval_recovery.live_http import MODEL, PROJECT, credential_presence, existing_key
from experiments.provider_eval_recovery.live_runner import OWNER, CATALOG
from experiments.provider_eval_recovery.public_inputs import DECLARED_ORIGINAL_SHA256, load_public
from .evidence import Evidence, MODES, PROTOCOL
from .hosted import HostedRuntime, prepare_task
from .routes import ExistingSearchRoute, public_fetch
from .evidence import ResearchTools

HERE = Path(__file__).resolve().parent
PUBLIC = HERE.parent / "provider_eval_recovery/real_public/inputs.parent-message.json"
SDK_VERSION = "3.22.1"
AGENTS_SDK_VERSION = "0.22.3"
APPROVAL = {"assistant": "Sentinel_51d1baf8e5ec8191ab1a53906ca66bf0",
    "proposal": "one real case across all four search modes; $1 soft incremental target; hosted costs not hard-capped",
    "user": "Sentinel_6c0c10271a64819181ead8587cfb6cb5", "answer": "yes"}
TARGET, STOP = Decimal("1.00"), Decimal("0.80")
MODEL_HOLD, CONTAINER_HOLD = Decimal("0.055"), Decimal("0.09")
REVIEW_HOLD = Decimal("0.05548")
BASE_HOLD = 4 * (MODEL_HOLD + CONTAINER_HOLD) + REVIEW_HOLD
SEARCH_HOLD = Decimal("0.0735")
ARM_SECONDS, POLL_SECONDS = 300, 5
USAGE_GRACE_SECONDS = 30
PREVIOUS_SOURCE_IDENTITY = ("a3b583a93446d444a20b67279f71bf416802d28c",
    "8c704a8582aea0a7a1ed51a8399a5ab76a0b326cded647919f98252e2552f84a")


def code_identity():
    repo = HERE.parents[1]
    files = sorted(p for p in HERE.glob("*.py") if not p.name.startswith("test_"))
    files += sorted((repo / "src/blueprint_pipeline/agent_execution").glob("*.py"))
    files += [repo / "src/blueprint_pipeline/paid_resource_admission.py"]
    for folder in ("provider_eval_adaptive_v1", "provider_eval_recovery"):
        files += sorted(p for p in (HERE.parent / folder).glob("*.py") if not p.name.startswith("test_"))
    names = [str(p.relative_to(repo)) for p in files]
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", "--", *names], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", *names], check=True)
    return commit, digest({name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in zip(names, files)})


def scope_inputs(root, owner):
    if owner != OWNER:
        raise AgentExecutionError("sole_fresh_execution_owner_required")
    access, original = read_json(root / "live_access.json"), read_json(root / "live_scope.json")
    public, _, provenance = load_public(PUBLIC)
    if (access.get("status") != "existing_access_configured" or access.get("execution_owner_task_id") != OWNER
            or access.get("catalog_version") != CATALOG or access.get("openai_project") != PROJECT
            or access.get("verified_model") != MODEL or access.get("journal_root") != str(root)
            or access.get("openai_model_http_status") != 200
            or not (access.get("parallel_x_api_key_supported") is True
                    or (access.get("parallel_x_api_key_supported") == "unverified"
                        and access.get("parallel_header_pilot_probe_authorized") is True))
            or access.get("key_presence") != dict.fromkeys(("parallel", "perplexity", "openai"), True)
            or set(access.get("allowed_hosts", [])) != {"api.parallel.ai", "api.perplexity.ai", "api.openai.com"}
            or original.get("budget_usd") != "10.00" or original.get("journal_root") != str(root)
            or original.get("model") != MODEL or original.get("public_inputs_sha256") != DECLARED_ORIGINAL_SHA256
            or not provenance["byte_parity_with_original_public"]):
        raise AgentExecutionError("existing_frozen_scope_access_receipt_required")
    return public, digest(access), digest(original)


def prepare_receipt(root, owner, *, clock=time.time):
    root = Path(root).resolve()
    public, access_hash, original_hash = scope_inputs(root, owner)
    commit, code_hash = code_identity()
    path = root / "protocols" / PROTOCOL / "soft_pilot_approval.json"
    if path.exists():
        receipt = read_json(path)
        return validate_receipt(root, owner, digest(receipt)), digest(receipt)
    with exclusive(root):
        journal_path = root / "live_journal.jsonl"
        if not journal_path.exists() or not journal_path.stat().st_size:
            raise AgentExecutionError("existing_prior_journal_required_no_zero_baseline")
        ledger = Ledger(root / "live_journal.jsonl", "10.00")
        if ledger.exposure < Decimal("3.934105"):
            raise AgentExecutionError("prior_reconciled_hold_checkpoint_not_preserved")
        if any(e.get("protocol") == PROTOCOL for e in ledger.events):
            raise AgentExecutionError("used_hosted_state_requires_original_soft_receipt_no_reset")
        if any(s not in {"completed", "not_accepted"} for s in ledger.states.values()):
            raise AgentExecutionError("prior_inflight_or_uncertain_dispatch_blocks_pilot")
        if ledger.exposure + BASE_HOLD + SEARCH_HOLD > 10:
            raise AgentExecutionError("original_aggregate_reservation_ceiling_exhausted")
        receipt = {"schema": "hosted_one_case_soft_pilot.v1", "approval": APPROVAL, "protocol": PROTOCOL,
            "case_id": public["cases"][0]["id"], "case_sha256": digest(public["cases"][0]),
            "public_inputs_sha256": DECLARED_ORIGINAL_SHA256, "arms": list(MODES), "model": MODEL, "project": PROJECT,
            "execution_owner": OWNER, "catalog": CATALOG, "root": str(root), "source_commit": commit,
            "code_sha256": code_hash, "access_sha256": access_hash, "original_scope_sha256": original_hash,
            "baseline_events": len(ledger.events), "baseline_head": ledger.previous,
            "baseline_reserved_usd": str(ledger.exposure), "soft_target_usd": str(TARGET), "stop_threshold_usd": str(STOP),
            "hard_cap": False, "no_new_credentials_or_access_grants": True,
            "per_arm_seconds": ARM_SECONDS, "max_function_calls": 48, "poll_seconds": POLL_SECONDS,
            "retained_container_hold_usd_per_arm": str(CONTAINER_HOLD),
            "container_expiry_guaranteed": False, "permanent_deletion_authorized": False,
            "sdk_versions": {"openai": SDK_VERSION, "openai-agents": AGENTS_SDK_VERSION},
            "created_at": clock(), "expires_at": clock() + 3600}
        write_once(path, receipt)
    return receipt, digest(receipt)


def validate_receipt(root, owner, expected_hash, *, readonly_previous=False):
    root = Path(root).resolve()
    public, access_hash, original_hash = scope_inputs(root, owner)
    receipt = read_json(root / "protocols" / PROTOCOL / "soft_pilot_approval.json")
    current_identity = code_identity()
    # This exact previous receipt may only be observed/reconciled. Execute still
    # requires the current code identity; no scope/approval is migrated or reset.
    commit, code_hash = PREVIOUS_SOURCE_IDENTITY if readonly_previous else current_identity
    required = {"schema": "hosted_one_case_soft_pilot.v1", "approval": APPROVAL, "protocol": PROTOCOL,
        "case_id": "BP-EVAL-01", "case_sha256": digest(public["cases"][0]), "arms": list(MODES),
        "public_inputs_sha256": DECLARED_ORIGINAL_SHA256, "model": MODEL, "project": PROJECT, "execution_owner": OWNER,
        "catalog": CATALOG, "root": str(root), "source_commit": commit, "code_sha256": code_hash,
        "access_sha256": access_hash, "original_scope_sha256": original_hash, "soft_target_usd": str(TARGET),
        "stop_threshold_usd": str(STOP), "hard_cap": False, "no_new_credentials_or_access_grants": True,
        "per_arm_seconds": ARM_SECONDS, "max_function_calls": 48, "poll_seconds": POLL_SECONDS,
        "retained_container_hold_usd_per_arm": str(CONTAINER_HOLD), "container_expiry_guaranteed": False,
        "permanent_deletion_authorized": False}
    required["sdk_versions"] = {"openai": SDK_VERSION, "openai-agents": AGENTS_SDK_VERSION}
    if digest(receipt) != expected_hash or any(receipt.get(k) != v for k, v in required.items()):
        raise AgentExecutionError("exact_reviewed_soft_pilot_receipt_required")
    ledger = Ledger(root / "live_journal.jsonl", "10.00")
    n = receipt.get("baseline_events")
    if type(n) is not int or not 0 <= n <= len(ledger.events):
        raise AgentExecutionError("original_journal_prefix_changed")
    head = ledger.events[n - 1]["sha256"] if n else "0" * 64
    states = {e["attempt_id"]: e["kind"] for e in ledger.events[:n]}
    held = sum((Decimal(e["amount_usd"]) for e in ledger.events[:n] if e["kind"] == "reserved"
                and states[e["attempt_id"]] != "not_accepted"), Decimal(0))
    if (head != receipt["baseline_head"] or str(held) != receipt["baseline_reserved_usd"]
            or any(e.get("protocol") == PROTOCOL for e in ledger.events[:n])
            or any(e.get("soft_receipt") != expected_hash for e in ledger.events
                   if e.get("protocol") == PROTOCOL and e.get("role") == "soft_pilot_allowance")
            or any(s not in {"completed", "not_accepted"} for s in states.values())
            or type(receipt.get("created_at")) not in {int, float}
            or receipt.get("expires_at") != receipt["created_at"] + 3600):
        raise AgentExecutionError("original_journal_prefix_changed")
    return receipt


class SoftMonitor:
    target, stop_threshold = TARGET, STOP
    model_hold, container_hold = MODEL_HOLD, CONTAINER_HOLD
    review_hold, base_hold, search_hold = REVIEW_HOLD, BASE_HOLD, SEARCH_HOLD
    protocol = PROTOCOL
    task_options = {}

    def __init__(self, root, receipt, *, clock=time.time, notify=print, sleep=time.sleep):
        self.root, self.receipt = Path(root).resolve(), receipt
        self.path = self.root / "protocols" / PROTOCOL / "soft_pilot"
        self.clock, self.notify = clock, notify
        self.sleep = sleep
        self.sha = digest(receipt)

    def task_id(self, mode):
        return "hosted_pilot_01_" + mode + "_" + self.sha[:16]

    def search_accounting_events(self, ledger):
        return ledger.events

    def prior_soft_exposure(self):
        return Decimal(0)

    def reserve(self):
        key = digest({"soft_receipt": self.sha, "kind": "model_container_review_allowance"})
        proof = {"soft_receipt": self.sha, "amount_usd": str(self.base_hold), "paid_call": False,
                 "model_usd": str(4 * self.model_hold), "container_carry_usd": str(4 * self.container_hold),
                 "independent_review_allowance_usd": str(self.review_hold)}
        with exclusive(self.root):
            ledger = Ledger(self.root / "live_journal.jsonl", "10.00")
            if key not in ledger.states:
                write_once(self.path / "allowance.json", proof)
                ledger.append("reserved", key, amount_usd=str(self.base_hold), protocol=self.protocol,
                    role="soft_pilot_allowance", soft_receipt=self.sha, dispatch_kind="accounting_reserve_only")
                ledger.append("completed", key, raw_sha256=digest(proof))
            elif ledger.states[key] != "completed":
                raise AgentExecutionError("pilot_allowance_reconciliation_required")

    def observations(self):
        result = []
        journal = AgentJournal(self.path / "agent_journal")
        for path in self.path.glob("*/observations/*.json"):
            row = read_json(path)
            if digest(row) != path.stem or journal.event("hosted_observe_" + path.stem) != row:
                self.stop("hosted_observation_integrity_failure")
                continue
            result.append(row)
        return result

    def stop(self, reason):
        path = self.path / "stop.json"
        if not path.exists():
            write_once(path, {"reason": reason, "at": self.clock(), "soft_receipt": self.sha})

    def report(self, *, preparing_creation=None):
        ledger = Ledger(self.root / "live_journal.jsonl", "10.00")
        sessions, overage = [], Decimal(0)
        journal = AgentJournal(self.path / "agent_journal")
        tasks = {t["task_id"]: t for t in journal.tasks(active_only=False, limit=10)}
        for mode in MODES:
            task_id = self.task_id(mode)
            start = self.path / task_id / "started.json"
            anchor = journal.event("hosted_creation_" + task_id)
            known = tasks.get(task_id, {})
            if anchor is None and not start.exists() and known.get("state", "queued") == "queued":
                continue
            if (anchor is None and not start.exists() and known.get("state") == "creating"
                    and task_id == preparing_creation and known.get("session_id") is None):
                # Only this exact task-bound first POST may install its proof.
                continue
            if (anchor is None or not start.exists() or read_json(start) != anchor
                    or anchor.get("task_id") != task_id or anchor.get("soft_receipt") != self.sha
                    or anchor.get("task_digest") != known.get("task_digest")):
                self.stop("durable_creation_proof_missing_or_changed")
            # A lost sidecar cannot hide a known task or reset its lifetime.
            started = anchor["at"] if anchor is not None else self.receipt["created_at"]
            observed = [o for o in self.observations() if o["task_id"] == task_id]
            latest = max(observed, key=lambda o: o["at"]) if observed else {}
            costs = [Decimal(o["model_estimate_usd"]) for o in observed if o["model_estimate_usd"] is not None]
            model_cost = max([self.model_hold, *costs])
            from .reconcile import retained_cleanup
            cleanup = retained_cleanup(journal, self.path, task_id, self.sha, known.get("task_digest"), started)
            until = cleanup["deleted_at"] if cleanup else self.clock()
            carry = Decimal(max(1, math.ceil(max(0, until - started) / 1200))) * Decimal("0.03")
            overage += max(Decimal(0), model_cost - self.model_hold) + max(Decimal(0), carry - self.container_hold)
            sessions.append({"mode": mode, "task_id": task_id, "session_id": known.get("session_id") or latest.get("session_id"),
                "environment_id": latest.get("environment_id"), "usage": latest.get("usage"),
                "model_high_water_estimate_usd": str(model_cost), "container_elapsed_estimate_usd": str(carry),
                "cleanup": "approved_deleted_api_absence_observed" if cleanup else "retained_pending_exact_session_deletion_approval",
                "cleanup_receipt_sha256": cleanup["raw_sha256"] if cleanup else None,
                "container_age_estimate_stopped_at_delete_ack": bool(cleanup),
                "ongoing_container_cost_unreconciled": True})
        topups = sum((ledger.reservations[e["attempt_id"]] for e in ledger.events if e["kind"] == "reserved"
            and e.get("soft_receipt") == self.sha and e.get("role") == "soft_pilot_upward_hold"), Decimal(0))
        shortfall = max(Decimal(0), overage - topups)
        if shortfall:
            proof = {"soft_receipt": self.sha, "upward_total_usd": str(overage), "amount_usd": str(shortfall)}
            key = digest(proof)
            write_once(self.path / "upward_observations" / (key + ".json"), proof)
            if ledger.exposure + shortfall <= 10:
                with exclusive(self.root):
                    ledger = Ledger(self.root / "live_journal.jsonl", "10.00")
                    if key not in ledger.states:
                        ledger.append("reserved", key, amount_usd=str(shortfall), protocol=self.protocol,
                            role="soft_pilot_upward_hold", soft_receipt=self.sha, dispatch_kind="accounting_reserve_only")
                        ledger.append("completed", key, raw_sha256=digest(proof))
                shortfall = Decimal(0)
            else:
                self.stop("observed_overrun_beyond_original_aggregate_ceiling")
        # Full remaining search opportunity is included, never treated as zero.
        actual_searches = sum((ledger.reservations[e["attempt_id"]] for e in self.search_accounting_events(ledger) if e["kind"] == "reserved"
            and e.get("protocol") == PROTOCOL and e.get("role") == PROTOCOL + ":search"), Decimal(0))
        incremental = ledger.exposure - Decimal(self.receipt["baseline_reserved_usd"]) + shortfall
        prior_soft = self.prior_soft_exposure()
        projected = prior_soft + incremental + max(Decimal(0), self.search_hold - actual_searches)
        if projected >= self.stop_threshold:
            self.stop("soft_target_approaching_stop_threshold")
        return {"soft_target_usd": str(self.target), "stop_threshold_usd": str(self.stop_threshold), "hard_cap": False,
            "aggregate_reserved_usd": str(ledger.exposure), "incremental_held_or_observed_usd": str(incremental),
            "projected_with_remaining_search_opportunity_usd": str(projected), "unreserved_observed_overrun_usd": str(shortfall),
            "prior_hosted_soft_exposure_usd": str(prior_soft),
            "sessions": sessions, "stopped": (self.path / "stop.json").exists(),
            "stop": read_json(self.path / "stop.json") if (self.path / "stop.json").exists() else None,
            "final_billing_cache_writes_tax_and_container_lifetime_unreconciled": True}

    def guard(self, *, preparing_creation=None):
        report = self.report(preparing_creation=preparing_creation)
        allowance = digest({"soft_receipt": self.sha, "kind": "model_container_review_allowance"})
        if Ledger(self.root / "live_journal.jsonl", "10.00").states.get(allowance) != "completed":
            raise AgentExecutionError("soft_pilot_allowance_not_durably_reserved")
        if self.clock() >= self.receipt["expires_at"]:
            self.stop("soft_pilot_approval_expired")
        if (self.path / "stop.json").exists():
            raise AgentExecutionError("soft_pilot_stopped_no_further_paid_work")
        return report

    def before(self, method, path, body):
        if method != "POST":
            return
        if path.endswith("/events") and all(e.get("type") == "agent.session.input.cancel" for e in body.get("events", [])):
            return
        if path == "/agents/sessions":
            task_id = body["metadata"]["blueprint_task_id"]
            if task_id not in {self.task_id(m) for m in MODES}:
                raise AgentExecutionError("only_four_frozen_pilot_tasks_authorized")
            started = self.path / task_id / "started.json"
            if started.exists():
                raise AgentExecutionError("durable_creation_intent_already_exists_no_retry")
            journal = AgentJournal(self.path / "agent_journal")
            task = journal.task(task_id)
            if body["metadata"].get("blueprint_task_digest") != task["task_digest"]:
                raise AgentExecutionError("durable_creation_task_binding_mismatch")
            self.guard(preparing_creation=task_id)
            intent = {"at": self.clock(), "task_id": task_id, "soft_receipt": self.sha, "task_digest": task["task_digest"]}
            journal.record_event("hosted_creation_" + task_id, intent)
            write_once(started, intent)
        self.guard()
        if path != "/agents/sessions":
            self.require_fresh_usage()
        # Use the same canonical paid allocator chokepoint for managed model
        # creation/resumption as for native provider searches. The status is
        # derived from the approved soft receipt and live guard, never a flag.
        binding = task_digest(self.receipt)
        grant = require_paid_resource_admission({"schema_version": PAID_LANE_ADMISSION_SCHEMA_VERSION,
            "resource_class": "openai_api_candidate", "status": "admitted", "blockers": [],
            "allocation_binding_digest": binding}, resource_class="openai_api_candidate",
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
        require_paid_resource_admission_grant(grant, resource_class="openai_api_candidate",
            allocation_binding_digest=binding, require_allocation_binding=True)

    def observe(self, task_id, session, *, creation=False):
        usage = session.get("usage")
        known = isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("input_tokens", "output_tokens"))
        prior = sorted((o for o in self.observations() if o["task_id"] == task_id), key=lambda o: o["at"])
        valid = [o for o in prior if o["model_estimate_usd"] is not None]
        if (usage is not None and not known) or (known and valid and any(
                usage[k] < max(o["usage"][k] for o in valid) for k in ("input_tokens", "output_tokens"))):
            self.stop("hosted_usage_malformed_or_nonmonotonic")
            raise AgentExecutionError("hosted_usage_malformed_or_nonmonotonic")
        if prior and self.clock() < prior[-1]["at"]:
            self.stop("hosted_usage_observation_clock_regressed")
            raise AgentExecutionError("hosted_usage_observation_clock_regressed")
        # Aggregate counts do not identify each call's context tier/cache writes:
        # use maximum long-context cache-write/output rates for monitoring.
        cost = (usage["input_tokens"] * Decimal("5") + usage["output_tokens"] * Decimal("15")) / 1000000 if known else None
        observation = {"at": self.clock(), "task_id": task_id, "session_id": session.get("id"),
            "environment_id": (session.get("environment") or {}).get("id"), "usage": usage,
            "model_estimate_usd": str(cost) if cost is not None else None, "usage_best_effort": True}
        AgentJournal(self.path / "agent_journal").record_event("hosted_observe_" + digest(observation), observation)
        write_once(self.path / task_id / "observations" / (digest(observation) + ".json"), observation)
        self.report()
        return known

    def usage_deadline(self, task_id):
        """Unknown reporting has a durable start; restarts cannot renew it."""
        rows = sorted((o for o in self.observations() if o["task_id"] == task_id), key=lambda o: o["at"])
        last_known = max((i for i, o in enumerate(rows) if o["model_estimate_usd"] is not None), default=-1)
        unknown = rows[last_known + 1:]
        if not unknown:
            return self.clock() + USAGE_GRACE_SECONDS
        journal = AgentJournal(self.path / "agent_journal")
        state = journal.task(task_id)
        active = [s for s in journal.lineage_tasks(task_id) if s["state"] not in TERMINAL_STATES]
        task_deadline = max((s["task"]["deadline"] for s in active), default=state["task"]["deadline"])
        return min(unknown[0]["at"] + USAGE_GRACE_SECONDS, task_deadline, self.receipt["expires_at"])

    def require_fresh_usage(self):
        journal = AgentJournal(self.path / "agent_journal")
        for state in journal.tasks(active_only=True, limit=10):
            if not state.get("session_id"):
                continue
            owner = journal.session_owner(state["task_id"])["task_id"]
            rows = [o for o in self.observations() if o["task_id"] == owner]
            latest = max(rows, key=lambda o: o["at"]) if rows else {}
            if (latest.get("model_estimate_usd") is None or self.clock() - latest["at"] > USAGE_GRACE_SECONDS):
                raise AgentExecutionError("fresh_hosted_usage_required_before_paid_admission")


class MonitoredRuntime(HostedRuntime):
    def __init__(self, *, monitor, **kwargs):
        self.monitor = monitor
        super().__init__(**kwargs)

    def prepare_event(self, task, path, body):
        if hasattr(self.transport, "prepare"):
            self.transport.prepare(path, body, task=task)

    def _request(self, method, path, **kwargs):
        self.monitor.before(method, path, kwargs.get("body") or {})
        result = super()._request(method, path, **kwargs)
        if ((method == "POST" and path == "/agents/sessions")
                or (method == "GET" and path.startswith("/agents/sessions/") and path.count("/") == 3)):
            task_id = (result.get("metadata") or {}).get("blueprint_task_id")
            if task_id != self.monitor.task_id(self.evidence.mode):
                raise AgentExecutionError("observed_session_pilot_task_mismatch")
            known = self.monitor.observe(task_id, result, creation=method == "POST")
            if method == "GET" and not known:
                deadline = self.monitor.usage_deadline(task_id)
                while not known and self.monitor.clock() < deadline:
                    # No tools, message events, create or inference during this
                    # reporting wait. Existing model/container holds stay counted.
                    self.monitor.guard()
                    self.monitor.sleep(min(POLL_SECONDS, deadline - self.monitor.clock()))
                    result = super()._request(method, path, **kwargs)
                    if self._validate_session(AgentTask.model_validate(self.journal.task(task_id)["task"]), result) != path.rsplit("/", 1)[-1]:
                        raise AgentExecutionError("agents_api_session_identity_changed")
                    known = self.monitor.observe(task_id, result)
                if not known:
                    self.monitor.stop("hosted_usage_reporting_grace_expired")
                    raise AgentExecutionError("hosted_usage_reporting_grace_expired")
        return result


def make_runtime(root, receipt, monitor, mode, *, transport=None, runtime_class=MonitoredRuntime):
    public = load_public(PUBLIC)[0]
    evidence = Evidence(getattr(monitor, "evidence_root", root / "protocols"), public["cases"][0], mode)
    evidence.reuse(root)
    journal = AgentJournal(monitor.path / "agent_journal")
    if transport is None:
        from .event_transport import DurableEvents, sdk_client
        api_key = existing_key("openai")
        transport = DurableEvents(fallback=OpenAIAgentsHTTP(api_key=api_key, project_id=PROJECT),
            journal=journal, receipt_root=monitor.path / "future_events",
            client_factory=lambda: sdk_client(api_key, PROJECT), clock=monitor.clock)
    def grant(context):
        monitor.guard()
        monitor.require_fresh_usage()
        return require_paid_resource_admission({"schema_version": PAID_LANE_ADMISSION_SCHEMA_VERSION,
            "resource_class": "evaluator_api", "status": "admitted", "blockers": [],
            "allocation_binding_digest": context.authority_digest}, resource_class="evaluator_api",
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
    tools = ResearchTools(evidence, search=ExistingSearchRoute(root, authorize=grant),
        fetch=lambda request, context: public_fetch(request, context, authorize=lambda _: monitor.guard()))
    def authorize(*_):
        monitor.guard()
        monitor.require_fresh_usage()
    ops = AgentOperations(journal, tools.tools(), authorize=authorize, clock=monitor.clock)
    runtime = runtime_class(evidence=evidence, common_prompt=public["common_prompt"], monitor=monitor,
        transport=transport,
        project_id=PROJECT, journal=journal, operations=ops, validate_admission=lambda _: monitor.guard(), clock=monitor.clock)
    task_id = monitor.task_id(mode)
    try:
        task = AgentTask.model_validate(journal.task(task_id)["task"])
    except AgentExecutionError as exc:
        if str(exc) != "agent_task_missing":
            raise
        inputs = [{"role": "user", "content": [{"type": "input_text", "text":
            "Research this public case using the complete evidence files and registered tools: " + json.dumps(evidence.case, sort_keys=True)}]}]
        admission = AgentAdmission(authority_digest=task_digest(receipt), authority_reference=str(monitor.path.parent / "soft_pilot_approval.json"),
            project_id=PROJECT, runtime="openai_agents_api", disclosure_scope="frozen_public_case1_same_provider_arm",
            allowed_input_digests=(task_digest(inputs), task_digest(runtime.files())), allowed_tool_ids=tuple(t.tool_id for t in tools.tools()),
            session_retention="until_deleted", trace_retention="provider_default", region="us",
            budget_policy="project_guard_accepted_uncertainty", inference_budget_usd=float(monitor.target),
            project_guard_receipt_digest=task_digest(receipt), expires_at=receipt["expires_at"])
        task = prepare_task(runtime, admission, task_id=task_id, source_commit=receipt["source_commit"],
                            deadline=min(monitor.clock() + ARM_SECONDS, receipt["expires_at"]), **monitor.task_options)
    return runtime, task


def run_pilot(root, receipt, *, clock=time.time, sleep=time.sleep, notify=print, factory=make_runtime):
    monitor = SoftMonitor(root, receipt, clock=clock, notify=notify, sleep=sleep)
    monitor.reserve()
    with exclusive(monitor.path):
        return _run_owned(root, receipt, monitor, sleep=sleep, notify=notify, factory=factory)


def _run_owned(root, receipt, monitor, *, sleep, notify, factory):
    outcomes = []
    for mode in MODES:
        runtime, task, state = None, None, None
        try:
            monitor.guard()
            runtime, task = factory(Path(root).resolve(), receipt, monitor, mode)
            state = runtime.start(task)
            if state["state"] == "creation_unresolved":
                monitor.stop("hosted_creation_outcome_uncertain")
                monitor.guard()
            while state["state"] not in TERMINAL_STATES:
                state = runtime.step(task.task_id)
                notify(json.dumps({"mode": mode, "state": state["state"], "budget": monitor.report()}))
                if state["state"] in {"creation_unresolved", "reconciling", "cancelling"}:
                    monitor.stop("unsettled_hosted_or_external_operation")
                if (state["state"] not in TERMINAL_STATES
                        and (state.get("cancel_requested") or monitor.clock() >= task.deadline)):
                    monitor.stop("arm_deadline_or_cancellation_unsettled")
                monitor.guard()
                if state["state"] not in TERMINAL_STATES:
                    sleep(POLL_SECONDS)
        except Exception:
            monitor.stop("pilot_interrupted_no_additional_paid_dispatch")
            # Durable cancellation; absence of a reply never proves cancellation.
            try:
                if runtime is None or task is None:
                    raise AgentExecutionError("no_owned_hosted_task_to_cancel")
                runtime.cancel(task.task_id)
                for _ in range(3):
                    state = runtime.step(task.task_id)
                    if state["state"] in TERMINAL_STATES:
                        break
                    sleep(POLL_SECONDS)
            except Exception:
                pass
            if runtime is not None and task is not None:
                outcomes.append({"mode": mode, "state": runtime.journal.task(task.task_id)})
            break
        outcomes.append({"mode": mode, "state": state})
        if (state["state"] != "completed" or state.get("cancel_requested")
                or runtime.journal.unsettled_operations(task.task_id)):
            monitor.stop("prior_arm_not_successfully_settled")
            break
        if monitor.report()["stopped"]:
            break
    result = {"protocol": monitor.protocol, "case": "BP-EVAL-01", "outcomes": outcomes, "budget": monitor.report(),
              "no_session_deleted": True, "no_further_cases_authorized": True}
    write_once(monitor.path / "reports" / (digest(result) + ".json"), result)
    notify(json.dumps(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-root", type=Path, required=True)
    parser.add_argument("--execution-owner-task-id", required=True)
    parser.add_argument("--soft-receipt-sha256")
    parser.add_argument("--cleanup-receipt-sha256")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-soft-pilot", action="store_true")
    action.add_argument("--check-access", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--reconcile-session", metavar="EXACT_SESSION_ID")
    action.add_argument("--reconcile-cleanup", type=Path, metavar="EXACT_CLEANUP_RECEIPT")
    args = parser.parse_args()
    root = args.aggregate_root.resolve()
    try:
        scope_inputs(root, args.execution_owner_task_id)
        if args.prepare_soft_pilot:
            receipt, sha = prepare_receipt(root, args.execution_owner_task_id)
            print(json.dumps({"receipt": receipt, "soft_receipt_sha256": sha, "new_paid_calls": 0}, indent=2))
            return
        if args.status:
            receipt = validate_receipt(root, args.execution_owner_task_id, args.soft_receipt_sha256)
            print(json.dumps(SoftMonitor(root, receipt).report(), indent=2))
            return
        if args.reconcile_session:
            from .reconcile import reconcile_session
            receipt = validate_receipt(root, args.execution_owner_task_id, args.soft_receipt_sha256,
                                       readonly_previous=True)
            result = reconcile_session(root, receipt, args.reconcile_session,
                transport=OpenAIAgentsHTTP(api_key=existing_key("openai"), project_id=PROJECT))
            print(json.dumps(result, indent=2))
            return
        if args.reconcile_cleanup:
            from .reconcile import reconcile_cleanup
            receipt = validate_receipt(root, args.execution_owner_task_id, args.soft_receipt_sha256,
                                       readonly_previous=True)
            print(json.dumps(reconcile_cleanup(root, receipt, args.reconcile_cleanup,
                args.cleanup_receipt_sha256), indent=2))
            return
        if version("openai") != SDK_VERSION or version("openai-agents") != AGENTS_SDK_VERSION:
            raise AgentExecutionError("install_reviewed_sdk_pair_openai_3.22.1_openai_agents_0.22.3")
        if not all(credential_presence().values()):
            raise AgentExecutionError("existing_bindings_not_present_no_credential_creation")
        if args.check_access:
            from openai import OpenAI
            client = OpenAI(api_key=existing_key("openai"), project=PROJECT, max_retries=0, timeout=20,
                            default_headers={"OpenAI-Beta": "agents=v1"})
            try:
                client.beta.agents.sessions.list(limit=1)
            except Exception as exc:
                print(json.dumps({"agents_read_access": False, "http_status": getattr(exc, "status_code", None),
                                  "hosted_write_inference_verified": False, "paid_calls": 0}))
                raise SystemExit(2) from None
            finally:
                client.close()
            print(json.dumps({"agents_read_access": True, "hosted_write_inference_verified": False, "paid_calls": 0}))
            return
        receipt = validate_receipt(root, args.execution_owner_task_id, args.soft_receipt_sha256)
        result = run_pilot(root, receipt)
        if result["budget"]["stopped"] or len(result["outcomes"]) != 4:
            raise SystemExit(2)
    except (AgentExecutionError, ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        reason = str(exc) if isinstance(exc, AgentExecutionError) else "local_receipt_scope_or_journal_validation_failed"
        print(json.dumps({"status": "blocked_or_stopped", "reason": reason}))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
