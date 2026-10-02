"""One bounded fresh-session canary using the exact installed research package.

inspect: GETs and private plan file only. stage: private canary control/inputs.
execute: one durable create, existing research/QA/publication. reconcile: never
creates. No scheduler, normal run edits, outreach or permanent deletion.
"""
import argparse
import base64
import copy
import hashlib
import importlib.util
import json
import os
import re
import signal
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from tools.daily_research import discovery, render, search
from tools.daily_research.adaptive_runtime import verify_process_watchdog
from tools.daily_research.consumer import Consumer, qa_deadline
from tools.daily_research.firestore import (
    Bridge,
    FencedProvider,
    FirestoreLedger,
    control_configuration,
)
from tools.daily_research.runner import (
    TERMINAL,
    Provider,
    Refusal,
    Runner,
    canonical,
    configuration,
    crm_snapshot,
    digest,
    due_date,
    instant,
    load_knowledge_bundle,
    preflight,
    read_json,
    save_bytes,
    validate_output,
)

spec = importlib.util.spec_from_file_location("reviewed_oct2_control", Path(__file__).with_name("research-oct2-control.py"))
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)
TEST = "perplexity-fast-20261001"
ROOT = "blueprintDailyResearch/sites-first/canaries/" + TEST
DAY = "2026-10-01"
SCOPE = "one-time-fresh-research-agent-qa-canonical-publication-no-outreach"
EXPIRES = "2026-10-02T10:00:00+00:00"
ORIGIN_DAY = "2026-10-01"
BASELINE = None
BASELINE_AUTHORITY = "Sentinel_c2046c5f146c81918921eba1ed7f6caa"
BASELINE_SCOPE = "baseline-research-agent-qa-canonical-publication-with-retries-no-outreach"


def select_attempt(number, day):
    """Host-selected retry identity; never rename or clear a previous attempt."""
    global TEST, ROOT, DAY, BASELINE  # One CLI context per process.
    if type(number) is not int or number < 1 or not isinstance(day, str):
        raise Refusal("baseline_attempt_identity_invalid")
    try:
        if datetime.fromisoformat(day).date().isoformat() != day:
            raise ValueError
    except ValueError:
        raise Refusal("baseline_attempt_identity_invalid") from None
    TEST = f"baseline-20261002-attempt-{number:04d}"
    ROOT = "blueprintDailyResearch/sites-first/canaries/" + TEST
    DAY = day
    BASELINE = {"baseline_id": "baseline-20261002", "attempt_number": number,
                "root": "blueprintDailyResearch/sites-first/baselines/baseline-20261002",
                "authority_reference": BASELINE_AUTHORITY, "soft_total_usd": 25}


def admission(value, now=None):
    now = now or datetime.now(timezone.utc)
    scope = BASELINE_SCOPE if BASELINE else SCOPE
    if (not isinstance(value, dict) or set(value) != {"schema_version", "test_id", "authority_reference", "ceiling_usd", "scope"}
            or value["schema_version"] != "blueprint.perplexity-canary-admission.v1"
            or value["test_id"] != TEST or type(value["ceiling_usd"]) is not int or value["ceiling_usd"] != 25
            or value["scope"] != scope or not isinstance(value["authority_reference"], str)
            or not value["authority_reference"].strip() or value["authority_reference"].startswith("PENDING")
            or (BASELINE and value["authority_reference"] != BASELINE_AUTHORITY)
            or (not BASELINE and now >= instant(EXPIRES)) or due_date(now, DAY) != DAY):
        raise Refusal("canary_one_time_admission_invalid_or_expired")
    return copy.deepcopy(value)


def driver(package, destination):
    text = Path(__file__).with_suffix(".mjs").read_text()
    text = text.replace("__RESEARCH_PACKAGE_URL__", Path(package).resolve().as_uri() + "/")
    text = text.replace("__CANARY_CONTEXT__", canonical({"test_id": TEST, "day": DAY, "baseline": BASELINE}))
    path = Path(destination) / "canary-bridge.mjs"
    path.write_text(text)
    path.chmod(0o600)
    return path


def repair_package_receipt(package, archive, expected_sha256, expected_source):
    """Verify an isolated reviewed overlay; never modify the installed package."""
    root, archive = Path(package).resolve(), Path(archive)
    if (not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
            or not isinstance(expected_source, str) or not re.fullmatch(r"[0-9a-f]{40}", expected_source)
            or hashlib.sha256(archive.read_bytes()).hexdigest() != expected_sha256):
        raise Refusal("repair_archive_binding_invalid")
    manifest = read_json(root / "manifest.json")
    with tarfile.open(archive) as bundle:
        if bundle.extractfile("manifest.json").read() != (root / "manifest.json").read_bytes():
            raise Refusal("repair_manifest_binding_invalid")
    if manifest.get("source_commit") != expected_source or not isinstance(manifest.get("files"), dict):
        raise Refusal("repair_source_binding_invalid")
    for name, sha256 in manifest["files"].items():
        original = root / name
        path = original.resolve()
        if (not path.is_relative_to(root) or original.is_symlink() or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != sha256):
            raise Refusal("repair_file_binding_invalid")
    if Path(render.__file__).resolve() != root / "tools/daily_research/render.py":
        raise Refusal("repair_imported_package_mismatch")
    return {"source_commit": expected_source, "archive_sha256": expected_sha256,
            "manifest_digest": digest(manifest), "files_verified": len(manifest["files"])}


def diagnose_saved_output(ledger, cfg, *, now=None):
    """Exact dated run only; no provider calls, store writes or private text dumps."""
    from tools.daily_research import recovery
    row = ledger.get(DAY)
    if not row or row.get("artifact_downloaded") is not True:
        raise Refusal("diagnostic_artifact_missing")
    raw = ledger.read_bytes(DAY + "-artifact.json")
    if hashlib.sha256(raw).hexdigest() != row.get("raw_output_digest"):
        raise Refusal("diagnostic_artifact_digest_mismatch")
    output = json.loads(raw)
    checked = now or datetime.now(timezone.utc)
    _, known = crm_snapshot(cfg["crm_snapshot"], checked)
    context, policy = row.get("knowledge_context"), row.get("refresh_policy")
    if digest(context) != row.get("knowledge_context_digest") or digest(policy) != row.get("refresh_policy_digest"):
        raise Refusal("diagnostic_context_binding_invalid")
    options = {"contract_version": row.get("research_contract_version"), "knowledge_context": context,
               "refresh_policy": policy, "observed_at": checked}
    def check(value):
        try:
            candidates, duplicates = validate_output(value, DAY, known, **options)
            discovery.validate_coverage(value.get("coverage"), len(value["candidates"]))
            return {"valid": True, "exact_deduped_candidate_count": len(candidates), "duplicate_count": len(duplicates)}
        except Refusal as exc:
            return {"valid": False, "error": str(exc)}
        except (ValueError, TypeError, KeyError):
            return {"valid": False, "error": "output_schema_invalid"}
    original_validation = check(output)
    try:
        derived, quarantine = recovery.quarantine_null_operator_deltas(output)
        replay = check(derived)
    except ValueError:
        quarantine, replay = [], {"valid": False, "error": "output_recovery_no_matching_proposal"}
    payload_input = row.get("create_payload", {}).get("input", "")
    return {"raw_output_sha256": row["raw_output_digest"], "session_id": row["session_id"], "turn_id": row["turn_id"],
            "original_validation": original_validation, "quarantined_derivation_validation": replay,
            "invalid_fields": [field for proposal in quarantine for field in proposal["invalid_fields"]],
            "candidate_count": len(output.get("candidates", [])), "finding_count": len(output.get("findings", [])),
            "knowledge_delta_count": len(output.get("proposed_knowledge_deltas", [])),
            "blocker_count": len(output.get("blockers", [])), "coverage_digest": digest(output.get("coverage")),
            "coverage": output.get("coverage"),
            "root_tool_trace": [{"call_id": key, "name": call.get("request", {}).get("name"),
                                 "query": call.get("request", {}).get("arguments", {}).get("query"),
                                 "url": call.get("request", {}).get("arguments", {}).get("url"),
                                 "state": call.get("state"), "result_digest": call.get("result_digest")}
                                for key, call in list(row.get("application_tool_calls", {}).items())[:100]
                                if call.get("phase") == "research"],
            "root_tool_trace_complete": len(row.get("application_tool_calls", {})) <= 100,
            "prompt_constraints": {"legacy_three_candidate_limit": "Find up to THREE" in payload_input,
                                   "legacy_two_search_limit": "at most two searches and two page opens" in payload_input,
                                   "ten_opportunity_target": "Target at least 10 NEW" in payload_input,
                                   "no_count_stopping_rule": "no prospect-count stopping rule" in payload_input,
                                   "defined_scope_required": "Define this run's concrete task/industry/region hypotheses" in payload_input},
            "application_tool_usage": row.get("application_tool_usage"),
            "input_sha256": hashlib.sha256(payload_input.encode()).hexdigest(),
            "knowledge_context_attached": canonical(canonical(context)) in payload_input,
            "knowledge_context_digest": row.get("knowledge_context_digest"),
            "crm_snapshot_digest": digest(row.get("crm_snapshot")),
            "crm_snapshot_complete": row.get("crm_snapshot", {}).get("complete") is True,
            "crm_snapshot_attached_to_research": any(value in payload_input for value in (
                canonical(row.get("crm_snapshot")), canonical(canonical(row.get("crm_snapshot"))))),
            "crm_instruction_note_present": ("No CRM is supplied to the sandbox" in payload_input
                or "not supplied to this research sandbox" in payload_input),
            "newness_verified": False, "provider_mutations": 0, "store_writes": 0}


def authorize_recovered_qa(bridge, receipt, *, clock=lambda: datetime.now(timezone.utc)):
    """Pin one bounded QA-only continuation under the existing shared allowance.

    No inference here. Root dates/deadlines, original spend proof and create guard
    are preserved. Repeating this request never extends the ten-minute window.
    """
    ledger = FirestoreLedger(bridge)
    with ledger.lock():
        row = ledger.get(DAY)
        required = {"authority_reference", "scope", "baseline_id", "soft_total_usd", "session_id", "root_turn_id",
                    "raw_output_sha256", "packet_digest", "model_observation_digest"}
        if (BASELINE is None or not row or row.get("state") != "awaiting_review" or row.get("qa")
                or not isinstance(receipt, dict) or set(receipt) != required
                or row.get("canary") != bridge.call("control").get("canary")):
            raise Refusal("recovered_qa_state_not_admitted")
        previous = row.get("qa_continuation")
        if previous:
            if previous["request"] != receipt:
                raise Refusal("recovered_qa_already_bound")
            return row
        row["qa_continuation"] = {"schema_version": "blueprint.recovered-research-qa.v1", "request": receipt,
                                   "started_at": clock().isoformat(), "duration_seconds": 600,
                                   "model_observation": copy.deepcopy(row.get("canary_model_estimate"))}
        qa_deadline(row, {})  # Validate every session/artifact/scope/cost binding before the write.
        ledger.put(row)
        return row


class CanaryBridge(Bridge):
    def call(self, op, **fields):
        if op == "put":
            row = fields["row"]
            if BASELINE:
                # Core date keys remain intact. External publication uses the
                # stable private attempt ID, so a retry cannot impersonate a
                # previous attempt or the normal dated research publication.
                for name, delivery in row.get("delivery", {}).items():
                    if delivery.get("key") == row["run_key"] + ":" + name:
                        delivery["key"] = "blueprint-research-canary:" + TEST + ":" + name
                        if name == "notion":
                            delivery["payload"]["summary"] = ("Blueprint baseline attempt " + TEST + "\n"
                                + delivery["payload"]["summary"])
                            delivery["payload_json"] = canonical(delivery["payload"])
                            delivery["payload_digest"] = digest(delivery["payload"])
            if row.get("state") == "creating" and not row.get("canary"):
                binding = super().call("control")["canary"]
                body = row["create_payload"]
                # Mutate the core's SAME body object before Store and actual POST.
                # Node-only rewriting would fail to change the provider payload.
                baseline_note = (f"This is baseline attempt {BASELINE['attempt_number']} under one shared $25 soft "
                    "TOTAL testing allowance across the baseline and all retries, including model/search/hosted compute. "
                    "Measure normal scope-complete research and QA; missing cost counts are unknown, not zero or a stop. "
                    "The recurring daily target is separate. " if BASELINE else "")
                body["input"] = (baseline_note + "This is the one-time Blueprint Perplexity canary; explicitly label the resulting "
                    "brief as a test. Prioritize publicly verified named decision contacts relevant to each "
                    "site/task opportunity, then role/team channels; generic contact channels are last. "
                    "Do not invent contact details or add fields outside the strict output schema. " + body["input"])
                metadata = body["metadata"]
                metadata.pop("payload_digest", None)
                metadata.update(purpose="blueprint_research_perplexity_canary", test_id=TEST,
                                canary_root=ROOT, admission_digest=binding["admission_digest"])
                if BASELINE:
                    metadata.update(baseline_id=BASELINE["baseline_id"],
                                    baseline_attempt=str(BASELINE["attempt_number"]),
                                    budget_authority_reference=BASELINE["authority_reference"],
                                    blueprint_run_id="blueprint-research-canary:" + TEST)
                metadata["payload_digest"] = digest(body)
                row["metadata"] = metadata
                row["canary"] = copy.deepcopy(binding)
        return super().call(op, **fields)


def inspect(bridge, approval, receipt, api, cache, now=None):
    approved = admission(approval, now or datetime.now(timezone.utc))
    origin = bridge.call("origin")
    control = origin["control"]
    row = origin["oct1_row"]
    if (control.get("enabled") is not False or control.get("config", {}).get("enabled") is not False
            or control.get("source_commit") != migration.SOURCE or origin["summary"].get("unfinished")
            or origin["summary"].get("cleanup_required") or origin.get("active_qa")
            or not row or row.get("cleanup_required") is not False or not row.get("cleanup_receipt")):
        raise Refusal("canary_daily_guard_unreconciled_or_changed")
    raw = base64.b64decode(bridge.call("origin_file", name=ORIGIN_DAY + "-artifact.json"), validate=True)
    migration.verify_failed(row, raw)
    inputs = {"knowledge.json": base64.b64decode(bridge.call("origin_file", name="knowledge.json"), validate=True),
              "refresh-policy.json": base64.b64decode(bridge.call("origin_file", name="refresh-policy.json"), validate=True)}
    snapshot = bridge.call("read_crm")
    # The reader timestamps its snapshot after the network request. Compare it
    # with a time sampled after that read, never the earlier admission time.
    validated_at = now or datetime.now(timezone.utc)
    inputs["crm.json"] = (canonical(snapshot) + "\n").encode()
    for name, value in inputs.items():
        save_bytes(cache / name, value)
    cfg = {**control_configuration(control), "crm_snapshot": str(cache / "crm.json"),
           "knowledge_snapshot": str(cache / "knowledge.json"), "knowledge_refresh_policy": str(cache / "refresh-policy.json")}
    configuration(cfg)
    _, known = crm_snapshot(cfg["crm_snapshot"], validated_at)
    load_knowledge_bundle(cfg, validated_at)
    checked = preflight(api, migration.INSTRUCTIONS, search.PROFILE)
    candidate = copy.deepcopy(control)
    candidate["workflow"]["enabled"] = False
    candidate["config"].update(first_date=DAY,
        scheduler_authority_reference="one-time-explicit-canary:" + approved["authority_reference"])
    candidate["canary"] = {"test_id": TEST, "root": ROOT, "admission": approved,
        "admission_digest": digest(approved), "origin_control": copy.deepcopy(control),
        "origin_row_digest": digest(row), "origin_raw_sha256": migration.RAW,
        "origin_row_blob": origin["oct1_row_blob"],
        "expires_at": None if BASELINE else EXPIRES, "package": receipt, "research_seconds": 1200,
        "total_seconds": 1800, "usage_observation_policy": "best-effort-post-run-v1", "hard_total_cap": False,
        "input_digests": {name: hashlib.sha256(value).hexdigest() for name, value in inputs.items()},
        "tool_definitions_digest": digest(checked["session_agent_override"]["tools"])}
    if BASELINE:
        candidate["canary"]["baseline"] = copy.deepcopy(BASELINE)
        candidate["canary"]["run_date"] = DAY
        # Read-only sequential/cleanup and aggregate-budget evidence, before
        # staging any intent. Stage rechecks atomically; this is not a lease.
        bridge.call("baseline_check")
    if (candidate["config"].get("max_runtime_seconds") != 1800
            or candidate["config"].get("qa_reserved_seconds") != 600
            or candidate["config"].get("search_provider") != search.PROFILE
            or candidate["config"].get("discovery_profile") != "adaptive-sites-v1"
            or candidate["config"].get("soft_target_usd") != 5
            or candidate["config"].get("recurring_budget_authority_reference") != migration.BUDGET_AUTHORITY):
        raise Refusal("canary_production_profile_not_migrated")
    candidate["enabled"] = candidate["config"]["enabled"] = candidate["workflow"]["enabled"] = True
    configuration(control_configuration(candidate))
    from tools.daily_research.consumer import workflow
    workflow(candidate)
    candidate["enabled"] = candidate["config"]["enabled"] = candidate["workflow"]["enabled"] = False
    result = {"schema_version": "blueprint.perplexity-canary-plan.v1", "package": receipt,
              "candidate": candidate, "inputs": {name: base64.b64encode(value).decode("ascii") for name, value in inputs.items()},
              "known_crm_identity_keys": len(known), "crm_values_digest": digest(snapshot["values"]),
              "provider_calls_read_only": True, "provider_mutations": 0, "firestore_writes": 0}
    result["plan_digest"] = digest(result)
    if BASELINE:
        # The retained plan remains immutable; budget telemetry is part of it.
        result["baseline_budget"] = bridge.call("baseline_status")
        result.pop("plan_digest")
        result["plan_digest"] = digest(result)
    return result


def stage(bridge, plan, receipt):
    expected = copy.deepcopy(plan)
    pinned = expected.pop("plan_digest", None)
    if pinned != digest(expected) or plan.get("schema_version") != "blueprint.perplexity-canary-plan.v1" or plan.get("package") != receipt:
        raise Refusal("canary_plan_binding_invalid")
    candidate = plan["candidate"]
    admission(candidate["canary"]["admission"])
    if candidate["canary"]["admission_digest"] != digest(candidate["canary"]["admission"]):
        raise Refusal("canary_admission_binding_invalid")
    bridge.call("stage", value=candidate)
    ledger = FirestoreLedger(bridge)
    with ledger.lock():
        bridge.call("guard")
        current = bridge.call("control")
        if ledger.get(DAY):
            raise Refusal("canary_existing_intent_reconcile_only")
        if current["canary"] != candidate["canary"]:
            raise Refusal("canary_admission_already_bound")
        for name, encoded in plan["inputs"].items():
            value = base64.b64decode(encoded, validate=True)
            if hashlib.sha256(value).hexdigest() != candidate["canary"]["input_digests"][name]:
                raise Refusal("canary_input_binding_invalid")
            ledger.write_bytes(name, value)
        candidate = copy.deepcopy(candidate)
        candidate["enabled"] = candidate["config"]["enabled"] = candidate["workflow"]["enabled"] = True
        bridge.call("configure", value=candidate)
        if migration.normalized(bridge.call("control")) != candidate:
            raise Refusal("canary_control_readback_failed")
    return {"state": "canary_staged", "test_id": TEST, "root": ROOT, "control_digest": digest(candidate),
            "provider_mutations": 0, "normal_control_changed": False}


def spend(api, row):
    # Public usage is best-effort, nullable even after completion, and may
    # change. Observe it without converting missing counts into zero or a cap.
    try:
        turns = api.listing("turns", row["session_id"])
    except Exception:  # noqa: BLE001 - telemetry failure is not proof of spend
        return {"known": False, "estimate_usd": None, "reported_estimate_usd": None,
                "usage_state": "unavailable", "hard_total_cap": False}
    return discovery.model_cost_observation(turns)


class CanaryProvider(FencedProvider):
    stopped = staticmethod(lambda: False)
    clock = staticmethod(lambda: datetime.now(timezone.utc))

    def safe(self, row=None):
        self.ledger.bridge.call("guard")
        control = self.ledger.bridge.call("control")
        binding = control["canary"]
        if not row:
            admission(binding["admission"], self.clock())
            return
        if row.get("canary") != binding:
            raise Refusal("canary_admission_binding_invalid")
        estimate = spend(self, row)
        discovery.preserve_estimate(row, "canary_model_estimate", estimate)

    def create(self, payload):
        self.safe()
        self.ledger.bridge.call("create_check", day=DAY, metadata=payload["metadata"])
        self.ledger.bridge.call("guard")
        self.ledger.bridge.call("assert_lease")
        control = self.ledger.bridge.call("control")
        row = self.ledger.get(DAY)
        admission(control["canary"]["admission"], self.clock())
        expired = not row or (self.clock() - instant(row["started_at"])).total_seconds() >= row["research_runtime_seconds"]
        if self.stopped() or control.get("enabled") is not True or expired:
            raise Refusal("canary_stopped_or_disabled_before_create")
        return Provider.create(self, payload)

    def tool_admit(self, row, phase):
        self.safe(row)
        return super().tool_admit(row, phase)

    def qa_input(self, session_id, event, key, day, request_digest, deadline_ms):
        self.safe(self.ledger.get(day))
        self.ledger.bridge.call("qa_check", day=day, request_digest=request_digest, deadline_ms=deadline_ms)
        self.ledger.bridge.call("guard")
        self.ledger.bridge.call("assert_lease")
        control = self.ledger.bridge.call("control")
        if (self.stopped() or control.get("enabled") is not True or control.get("workflow", {}).get("enabled") is not True
                or self.clock().timestamp() * 1000 >= deadline_ms):
            raise Refusal("canary_stopped_disabled_or_expired_before_qa")
        self.api.sessions.events.create(session_id, events=[event], idempotency_key=key)


def run(bridge, cache, *, execute=False, api_factory=CanaryProvider, recovery_only=False,
        stopped=lambda: False, clock=lambda: datetime.now(timezone.utc), sleep=time.sleep):
    ledger = FirestoreLedger(bridge)
    cfg = render.configured(bridge, cache, allow_create=not recovery_only)
    control = bridge.call("control")
    if not control or control.get("canary", {}).get("test_id") != TEST:
        raise Refusal("canary_not_staged")
    existing = ledger.get(DAY)
    if recovery_only:
        if not existing or not existing.get("qa_continuation") or existing["state"] not in {"awaiting_review", "reviewed", "completed"}:
            raise Refusal("recovered_qa_intent_required")
        qa_deadline(existing, cfg)
    if not existing:
        if not execute:
            return {"state": "no_canary_intent", "provider_mutations": 0}
        admission(control["canary"]["admission"], clock())
        bridge.call("guard")
        if due_date(clock(), cfg["first_date"]) != DAY:
            raise Refusal("canary_date_scope_invalid")
    api = api_factory(ledger, os.environ.get("OPENAI_API_KEY", ""))
    api.stopped = stopped
    api.clock = clock
    runner = Runner(ledger, cfg, api, clock=clock)
    runner.stop_requested = stopped
    consumer = Consumer(ledger, cfg, api, clock=clock, stopped=stopped)
    consumer.active_day = DAY
    try:
        if existing:
            # Recovery follows the existing test identity even after due_date
            # advances. Terminal/QA rows must not look up a different date.
            with ledger.lock():
                current = ledger.get(DAY)
                result = current if current["state"] in TERMINAL else runner.observe(current)
        else:
            result = runner.start_or_resume(allow_create=execute)
        while True:
            row = ledger.get(DAY)
            if not row:
                return result
            reason = "canary_interrupted" if stopped() else None
            terminal = row["state"] in {"failed", "cancelled", "completed", "creation_unresolved"}
            if row.get("session_id"):
                try:
                    if not terminal:
                        bridge.call("guard")
                    estimate = spend(api, row)
                    with ledger.lock():
                        fresh = ledger.get(DAY)
                        discovery.preserve_estimate(fresh, "canary_model_estimate", estimate)
                        ledger.put(fresh)
                        row = fresh
                except Exception:  # noqa: BLE001 - no upstream secrets
                    reason = "canary_guard_or_state_unavailable"
            if terminal:
                return {**summary(row), **({"observer_error": reason} if reason else {})}
            total_exhausted = (clock() >= qa_deadline(row, cfg)) if recovery_only else (
                (clock() - instant(row["started_at"])).total_seconds() >= row["total_runtime_seconds"])
            if total_exhausted:
                reason = "canary_total_observation_deadline"
            if row["state"] in {"running", "collecting", "cancel_pending", "creating"}:
                if reason:
                    runner.cancel_current(DAY, reason)
                result = runner.start_or_resume(allow_create=False)
            elif row["state"] in {"awaiting_review", "reviewed"}:
                if reason and row.get("qa", {}).get("state") != "validated":
                    if row.get("qa"):
                        with ledger.lock():
                            fresh = ledger.get(DAY)
                            consumer.cancel(fresh, reason)
                        # Observe the existing QA even after the deadline so a
                        # confirmed cancellation can become terminal. A stopped
                        # Consumer never starts input/tools or publication.
                        consumer.stopped = lambda: True
                        result = consumer.step()
                        return {**summary(ledger.get(DAY)), "observer_error": reason,
                                "observer_state": result["state"]}
                    return {**summary(ledger.get(DAY)), "observer_error": reason}
                # Validated terminal QA permits only canonical publication/GET
                # recovery after paid-work deadline. No new inference here.
                result = consumer.step()
                if result["state"] in {"qa_blocked", "qa_cancel_pending", "workflow_disabled", "publication_pending"}:
                    return {**summary(ledger.get(DAY)), "observer_state": result["state"]}
            else:
                raise Refusal("canary_state_invalid")
            if reason and row["state"] != "reviewed":
                return {**summary(ledger.get(DAY)), "observer_error": reason}
            sleep(3)
    finally:
        api.client.close()


def summary(row):
    result = render.status_summary(row)
    result.update(test_id=TEST, blueprint_run_id="blueprint-research-canary:" + TEST, root=ROOT,
                  row_digest=digest(row), environment_id=row.get("environment_id"),
                  root_turn_status=row.get("turn_status"), artifact_downloaded=row.get("artifact_downloaded"),
                  raw_output_sha256=row.get("raw_output_digest"), hard_total_cap_verified=False,
                  canary_model_estimate=row.get("canary_model_estimate"), normal_control_changed=False,
                  qa_turn_id=row.get("qa", {}).get("turn_id"), qa_state=row.get("qa", {}).get("state"),
                  qa_turn_status=row.get("qa", {}).get("turn_status"),
                  qa_artifact_sha256=row.get("qa", {}).get("artifact_digest"),
                  delivery={name: {"state": value.get("state"), "receipt": value.get("receipt")}
                            for name, value in row.get("delivery", {}).items()},
                  admission_digest=row.get("canary", {}).get("admission_digest"))
    if BASELINE:
        result["baseline"] = copy.deepcopy(BASELINE)
    return result


def record_cleanup(bridge, cache, receipt, *, api_factory=FencedProvider):
    """Record an independently approved/completed deletion; never delete here."""
    ledger = FirestoreLedger(bridge)
    cfg = render.configured(bridge, cache)
    api = api_factory(ledger, os.environ.get("OPENAI_API_KEY", ""))
    try:
        return summary(Runner(ledger, cfg, api).record_cleanup(DAY, receipt))
    finally:
        api.client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "stage", "execute", "reconcile", "status", "export", "record-cleanup", "abandon-unstarted", "recover-output", "diagnose-output", "reprice", "authorize-recovered-qa", "resume-qa", "export-recovered"])
    parser.add_argument("--package", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--approval")
    parser.add_argument("--plan")
    parser.add_argument("--output")
    parser.add_argument("--receipt")
    parser.add_argument("--repair-package", help="Isolated new reviewed package; offline/read-only provider repairs only")
    parser.add_argument("--repair-archive")
    parser.add_argument("--repair-source")
    parser.add_argument("--repair-sha256")
    parser.add_argument("--attempt", type=int, help="Sequential attempt under the approved shared baseline allowance")
    parser.add_argument("--date", help="Chicago due date for this private attempt; normal dated history is unchanged")
    args = parser.parse_args()
    if args.attempt is not None:
        select_attempt(args.attempt, args.date)
    elif args.date is not None:
        raise Refusal("baseline_attempt_identity_invalid")
    repair_command = args.command in {"recover-output", "diagnose-output", "reprice", "authorize-recovered-qa", "resume-qa", "export-recovered"}
    repair_arguments = (args.repair_package, args.repair_archive, args.repair_source, args.repair_sha256)
    if repair_command != all(repair_arguments) or not repair_command and any(repair_arguments):
        raise Refusal("repair_package_required_or_command_not_admitted")
    receipt = migration.package_receipt(args.package, args.archive, verify_import=not repair_command)
    repair_receipt = None
    if repair_command:
        if Path(args.repair_package).resolve() == Path(args.package).resolve():
            raise Refusal("repair_must_preserve_installed_package")
        repair_receipt = repair_package_receipt(args.repair_package, args.repair_archive, args.repair_sha256, args.repair_source)
    if args.command in {"execute", "reconcile", "resume-qa"}:
        verify_process_watchdog()
    stop = {"requested": False}
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.update(requested=True))
    with tempfile.TemporaryDirectory(prefix="blueprint-perplexity-canary-") as temporary:
        cache = Path(temporary)
        bridge = CanaryBridge(script=driver(args.repair_package if repair_command else args.package, cache))
        api = None
        try:
            if args.command == "inspect":
                if not args.approval or not args.output:
                    raise Refusal("canary_required_argument_missing")
                api = Provider(os.environ.get("OPENAI_API_KEY", ""))
                result = inspect(bridge, read_json(args.approval), receipt, api, cache)
                migration.write_private(args.output, result)
                result = {k: v for k, v in result.items() if k not in {"candidate", "inputs", "package"}}
                result["state"] = "canary_read_only_admission_checked"
            elif args.command == "stage":
                if not args.plan:
                    raise Refusal("canary_required_argument_missing")
                result = stage(bridge, read_json(args.plan), receipt)
            elif args.command == "status":
                row = FirestoreLedger(bridge).get(DAY)
                result = summary(row) if row else {"state": "no_canary_intent", "provider_mutations": 0}
            elif args.command == "export":
                if not args.output:
                    raise Refusal("canary_required_argument_missing")
                result = render.export_snapshot(bridge, DAY, args.output)
            elif args.command == "record-cleanup":
                if not args.receipt:
                    raise Refusal("canary_required_argument_missing")
                result = record_cleanup(bridge, cache, read_json(args.receipt))
            elif args.command == "recover-output":
                if not args.receipt:
                    raise Refusal("canary_required_argument_missing")
                ledger = FirestoreLedger(bridge)
                row = Runner(ledger, render.configured(bridge, cache, allow_create=False), None).recover_output(DAY, read_json(args.receipt))
                result = {**summary(row), "provider_mutations": 0, "publication_writes": 0,
                          "quarantined_proposal_count": len(row["packet"]["output_recovery"]["quarantined_proposals"])}
            elif args.command == "diagnose-output":
                result = diagnose_saved_output(FirestoreLedger(bridge), render.configured(bridge, cache, allow_create=False))
            elif args.command == "reprice":
                ledger = FirestoreLedger(bridge)
                with ledger.lock():
                    row = ledger.get(DAY)
                    if not row or row.get("turn_status") not in {"completed", "failed", "cancelled"}:
                        raise Refusal("reprice_requires_retained_terminal_turn")
                    api = Provider(os.environ.get("OPENAI_API_KEY", ""))
                    discovery.preserve_estimate(row, "canary_model_estimate", spend(api, row))
                    ledger.put(row)
                    result = {**summary(row), "provider_mutations": 0, "publication_writes": 0}
            elif args.command == "authorize-recovered-qa":
                if not args.receipt:
                    raise Refusal("canary_required_argument_missing")
                result = {**summary(authorize_recovered_qa(bridge, read_json(args.receipt))),
                          "provider_mutations": 0, "publication_writes": 0}
            elif args.command == "resume-qa":
                # This explicitly paid command is distinct from diagnosis,
                # repricing and packet recovery, and has no root-create path.
                result = run(bridge, cache, recovery_only=True, stopped=lambda: stop["requested"])
            elif args.command == "export-recovered":
                if not args.output or not FirestoreLedger(bridge).get(DAY).get("output_recovery"):
                    raise Refusal("recovered_export_requires_output_and_receipt")
                result = render.export_snapshot(bridge, DAY, args.output)
            elif args.command == "abandon-unstarted":
                if not BASELINE:
                    raise Refusal("canary_baseline_not_selected")
                result = bridge.call("abandon_unstarted")
            else:
                result = run(bridge, cache, execute=args.command == "execute", stopped=lambda: stop["requested"])
            if BASELINE:
                result["baseline_budget"] = bridge.call("baseline_status")
            if repair_receipt:
                result["repair_package"] = repair_receipt
            print(canonical(result), flush=True)
        finally:
            if api:
                api.client.close()
            bridge.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - fixed errors only
        print(canonical({"state": "blocked", "error": str(error) if isinstance(error, Refusal) else "canary_runtime_unavailable"}), flush=True)
        raise SystemExit(1) from None
