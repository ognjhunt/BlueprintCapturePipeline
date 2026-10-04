"""Optional host-admitted Exa expansion; no default transport or billing claims.

The caller holds the existing daily ledger lock/fence. Allocation and tool_schema
come from trusted host evidence, never model arguments. Transport must implement
start(request) and read(original_run_id), without automatic POST retries.
The injected transport must bound each request to the original run deadline.
"""
import copy
import hashlib
import json
import re
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

START = "blueprint_start_exa_expansion"
READ = "blueprint_read_exa_expansion"
PROFILE = "exa-guarded-v1"
ALLOCATION = "blueprint.research-expansion-allocation.v1"
LIMIT_MICROS = 5_000_000
# Exa documents a $1 minimum for Ultra; advertise exactly the range the host enforces.
ULTRA_MIN_MICROS = 1_000_000
TERMINAL = {"completed", "failed", "cancelled"}
MAX_BYTES = 5_000_000


class ExpansionError(ValueError):
    """Stable, secret-free binding/integrity errors."""


def tools():
    return [
        {"type": "function", "name": START, "defer_loading": False,
         "description": "Optionally expand a useful discovery family once within this daily run. Host verifies existing authentication, native cap and remaining all-in allocation; unverified admission skips without a provider start.",
         "parameters": {"type": "object", "additionalProperties": False,
                        "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 24000},
                                       "max_cost_micros": {"type": "integer", "minimum": ULTRA_MIN_MICROS, "maximum": LIMIT_MICROS}},
                        "required": ["query", "max_cost_micros"]}},
        {"type": "function", "name": READ, "defer_loading": False,
         "description": "Read only this daily run's acknowledged original Exa expansion, retaining full results and sources. Never starts a replacement or extends the original deadline.",
         "parameters": {"type": "object", "additionalProperties": False,
                        "properties": {}, "required": []}},
    ]


def instructions():
    return (
        "After first-pass research, choose whether one targeted Exa list expansion would fill a useful gap "
        "before final output and source QA. US sites only; consider manufacturing, laundry, food production, "
        "packing, machine tending, material handling, warehouses and other relevant physical tasks. "
        "Use one exact query grounded in retained findings, without enrichments or outreach. "
        "The requested max_cost_micros is an integer from 1000000 ($1, Exa Ultra's documented minimum) to 5000000 ($5); "
        "a smaller cap is refused, never raised automatically. "
        "It shares the existing all-in $5 research allocation and is not extra authority. "
        "Host admission requires verified authentication, actual native cap support and remaining all-in headroom. "
        "If any is unavailable or unknown, retain the actionable skip and continue ordinary research. "
        "Once attempted, never repeat a native start, change the query, use previousRunId or create another job. "
        "Observe pending work only with the original-ID read tool within the original deadline. "
        "Merge returned raw site/task evidence into the same output before final QA and deduplication; "
        "provider matches remain discovery candidates with unsupported ownership, interest, consent and robot fit unknown."
    )


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timezone_required")
    return parsed


def _skip(reason):
    return {"ok": False, "state": "skipped", "reason": reason,
            "action": "Continue ordinary research; retain this expansion gap. No Exa start was made."}


def below_ultra_minimum(name, args):
    """A start whose cap is below Exa Ultra's minimum needs no context, key or catalog call."""
    return (name == START and isinstance(args, dict) and type(args.get("max_cost_micros")) is int
            and 0 < args["max_cost_micros"] < ULTRA_MIN_MICROS)


def _deadline(row):
    seconds = row.get("research_runtime_seconds")
    if type(seconds) is not int or not 0 < seconds <= 1800:
        raise ExpansionError("expansion_original_deadline_invalid")
    return _time(row["started_at"]) + timedelta(seconds=seconds)


def _cap_supported(schema, request):
    if not isinstance(schema, dict):
        return False
    try:
        properties = schema.get("properties", {})
        effort = properties.get("effort", {})
        budget = properties.get("budget", {})
        cap = budget.get("properties", {}).get("maxCostDollars", {})
        dollars, query = request["budget"]["maxCostDollars"], request["query"]
        compatible = (cap.get("type") == "number" and budget.get("type") == "object"
                and request.get("effort") == "ultra" and effort.get("type") == "string"
                and isinstance(effort.get("enum"), list) and "ultra" in effort["enum"]
                and properties.get("query", {}).get("type") == "string"
                and not set(schema.get("required", [])) - {"query", "effort", "budget"}
                and not set(budget.get("required", [])) - {"maxCostDollars"})
        # Ultra's documented minimum applies even when tools/list omits it.
        return (compatible and ULTRA_MIN_MICROS / 1_000_000 <= dollars <= LIMIT_MICROS / 1_000_000
                and dollars >= cap.get("minimum", 1) and dollars <= cap.get("maximum", 5)
                and ("exclusiveMinimum" not in cap or dollars > cap["exclusiveMinimum"])
                and ("exclusiveMaximum" not in cap or dollars < cap["exclusiveMaximum"])
                and ("enum" not in cap or dollars in cap["enum"])
                and properties["query"].get("minLength", 1) <= len(query) <= properties["query"].get("maxLength", 24000))
    except (AttributeError, KeyError, TypeError):
        return False


def _allocation(snapshot, row, cap, now, deadline):
    return _allocation_problem(snapshot, row, cap, now, deadline) is None


def _allocation_problem(snapshot, row, cap, now, deadline):
    """None when admitted; "cap_exceeds_remaining" when only the requested cap is too
    large for a verified snapshot; "unverified" for every other gap."""
    try:
        fields = ("limit_micros", "committed_micros", "reserved_micros", "remaining_micros")
        if (not isinstance(snapshot, dict) or snapshot.get("schema_version") != ALLOCATION
                or snapshot.get("all_in_verified") is not True or snapshot.get("usage_unknown") is not False
                or snapshot.get("run_key") != row["run_key"]
                or snapshot.get("authority_reference") != row.get("recurring_budget_authority_reference")
                or not isinstance(snapshot.get("evidence_reference"), str) or not snapshot["evidence_reference"].strip()
                or any(type(snapshot.get(key)) is not int or snapshot[key] < 0 for key in fields)
                or snapshot["limit_micros"] != LIMIT_MICROS or row.get("soft_target_usd") != 5
                or snapshot["remaining_micros"] != LIMIT_MICROS - snapshot["committed_micros"] - snapshot["reserved_micros"]
                or not _time(snapshot["checked_at"]) <= now < _time(snapshot["valid_until"]) <= deadline):
            return "unverified"
    except (AttributeError, KeyError, TypeError, ValueError):
        return "unverified"
    return "cap_exceeds_remaining" if cap > snapshot["remaining_micros"] else None


def allocation_diagnostic(row, snapshot=None, *, now=None):
    """Read-only, allowlisted explanation; never manufactures cost evidence.

    A token count, soft target or communications reservation cannot produce an
    all-in research allocation. The original admission predicate remains the
    source of truth. Unknown/malformed evidence yields no numeric headroom.
    """
    now = now or datetime.now(timezone.utc)
    result = {"schema_version": "blueprint.research-expansion-allocation-status.v1",
              "verification": "unknown", "remaining_micros": None,
              "reasons": [], "claim_created": False, "provider_started": False}
    if not isinstance(row, dict):
        result["reasons"] = ["research_run_context_missing"]
        return result
    result["model_usage_present"] = isinstance(row.get("usage"), dict)
    result["application_tool_billing_verified"] = row.get("application_tool_usage", {}).get("provider_billing_verified") is True if isinstance(row.get("application_tool_usage"), dict) else False
    if not isinstance(snapshot, dict):
        result["reasons"] = ["company_all_in_allocation_missing"]
        result["required_evidence"] = ["model_and_agent_hosting_billing", "research_tool_billing",
                                       "remaining_phase_reservations", "current_run_authority_binding"]
        return result
    reasons = result["reasons"]
    if snapshot.get("schema_version") != ALLOCATION:
        reasons.append("allocation_schema_invalid")
    if snapshot.get("all_in_verified") is not True or snapshot.get("usage_unknown") is not False:
        reasons.append("all_in_usage_unverified")
    if snapshot.get("run_key") != row.get("run_key"):
        reasons.append("allocation_run_mismatch")
    if snapshot.get("authority_reference") != row.get("recurring_budget_authority_reference"):
        reasons.append("allocation_authority_mismatch")
    if not isinstance(snapshot.get("evidence_reference"), str) or not snapshot["evidence_reference"].strip():
        reasons.append("allocation_evidence_reference_missing")
    fields = ("limit_micros", "committed_micros", "reserved_micros", "remaining_micros")
    integers = all(type(snapshot.get(key)) is int and snapshot[key] >= 0 for key in fields)
    if not integers:
        reasons.append("allocation_amounts_invalid")
    elif snapshot["remaining_micros"] != LIMIT_MICROS - snapshot["committed_micros"] - snapshot["reserved_micros"]:
        reasons.append("allocation_arithmetic_invalid")
    if snapshot.get("limit_micros") != LIMIT_MICROS or row.get("soft_target_usd") != 5:
        reasons.append("allocation_existing_limit_mismatch")
    try:
        deadline = _deadline(row)
        if not _time(snapshot["checked_at"]) <= now < _time(snapshot["valid_until"]) <= deadline:
            reasons.append("allocation_not_current_within_original_deadline")
        valid = _allocation(snapshot, row, 0, now, deadline)
    except (AttributeError, KeyError, TypeError, ValueError):
        reasons.append("allocation_time_or_run_context_invalid")
        valid = False
    if valid:
        result.update(verification="verified_retained_snapshot", remaining_micros=snapshot["remaining_micros"])
    elif not reasons:
        reasons.append("allocation_admission_unverified")
    return result


def _receipt(ledger, claim, label, raw):
    if len(raw) > MAX_BYTES:
        raise ExpansionError("expansion_receipt_too_large_not_truncated")
    filename = f"{claim['date']}-exa-{claim['intent_sha256']}-{label}.json"
    try:
        existing = ledger.read_bytes(filename)
    except FileNotFoundError:
        ledger.write_bytes(filename, raw)
    else:
        if existing != raw:
            raise ExpansionError("expansion_receipt_conflict")
    return {"file": filename, "sha256": _hash(raw), "bytes": len(raw)}


def _read_receipt(ledger, ref):
    raw = ledger.read_bytes(ref["file"])
    if len(raw) != ref["bytes"] or _hash(raw) != ref["sha256"]:
        raise ExpansionError("expansion_receipt_digest_mismatch")
    return json.loads(raw)


def _project(claim, result=None):
    value = {"ok": True, "state": claim["state"], "run_id": claim.get("run_id"),
             "intent_sha256": claim["intent_sha256"], "reserved_micros": claim["cap_micros"],
             "replay_permitted": False, "billing_verified": False}
    if result is not None:
        value["provider_record"] = result
    if not claim.get("run_id"):
        value.update(ok=False, reason="expansion_submission_unresolved",
                     action="Preserve this claim and reservation; reconcile the original submission without replay.")
    return value


def execute(name, args, row, ledger, *, transport=None, allocation=None,
            tool_schema=None, phase="research", now=None, admit=None,
            unavailable_reason="expansion_authenticated_transport_missing", allocation_status=None):
    """Single start per daily run; original-ID reads retain full JSON provenance."""
    now = now or datetime.now(timezone.utc)
    started = time.monotonic()

    def current_time():
        return now + timedelta(seconds=time.monotonic() - started)

    if name not in {START, READ} or not isinstance(args, dict):
        raise ExpansionError("expansion_arguments_invalid")
    stored = ledger.get(row["date"])
    if not stored or any(stored.get(k) != row.get(k) for k in ("run_key", "session_id", "turn_id", "started_at")):
        raise ExpansionError("expansion_daily_binding_changed")
    claim = stored.get("exa_expansion")
    if claim:
        row["exa_expansion"] = copy.deepcopy(claim)
        if (claim["run_key"] != row["run_key"] or claim["date"] != row["date"]
                or any(claim["intent"].get(k) != row.get(k) for k in ("run_key", "session_id", "turn_id"))
                or _hash(_bytes(claim["intent"])) != claim["intent_sha256"]
                or claim.get("intent_json") != _bytes(claim["intent"]).decode()):
            raise ExpansionError("expansion_intent_binding_changed")
    if name == READ:
        if args:
            raise ExpansionError("expansion_read_original_id_only")
        if not claim:
            return _skip("expansion_original_run_missing")
    else:
        if (set(args) != {"query", "max_cost_micros"} or not isinstance(args.get("query"), str)
                or not args["query"].strip() or len(args["query"].encode()) > 24000
                or type(args.get("max_cost_micros")) is not int or not 0 < args["max_cost_micros"] <= LIMIT_MICROS):
            raise ExpansionError("expansion_start_arguments_invalid")
        if claim:
            original = claim["intent"]["request"]
            expected = {"query": args["query"], "budget": {"maxCostDollars": args["max_cost_micros"] / 1_000_000}}
            # Old immutable starts omitted effort. They remain observation-only;
            # never retrofit Ultra or replay their already consumed claim.
            if "effort" in original:
                expected["effort"] = "ultra"
            if original != expected:
                raise ExpansionError("expansion_already_claimed_different_request")
    if claim and claim.get("terminal_receipt"):
        record = _read_receipt(ledger, claim["terminal_receipt"])
        if (record.get("intent_sha256") != claim["intent_sha256"]
                or record.get("run_id") != claim.get("run_id")
                or record.get("provider_record", {}).get("id") != claim.get("run_id")):
            raise ExpansionError("expansion_terminal_binding_changed")
        return _project(claim, record["provider_record"])
    # Recover an ACK file saved before a row-pointer persistence failure.
    if claim and not claim.get("run_id"):
        filename = f"{claim['date']}-exa-{claim['intent_sha256']}-start.json"
        try:
            raw = ledger.read_bytes(filename)
        except FileNotFoundError:
            raw_filename = f"{claim['date']}-exa-{claim['intent_sha256']}-start-http.json"
            try:
                raw_ack = ledger.read_bytes(raw_filename)
                from tools.daily_research.exa_transport import reconcile_start_ack
                result = reconcile_start_ack(json.loads(raw_ack), claim["intent"]["request"])
            except (FileNotFoundError, ValueError):
                return _project(claim)
            raw_ref = {"file": raw_filename, "sha256": _hash(raw_ack), "bytes": len(raw_ack)}
            record = {"intent_sha256": claim["intent_sha256"], "run_id": result["id"],
                      "provider_record": result, "source_raw_ack": raw_ref}
            raw = _bytes(record)
            _receipt(ledger, claim, "start", raw)
        record = json.loads(raw)
        result = record.get("provider_record", {})
        if (record.get("intent_sha256") != claim["intent_sha256"] or not _valid_id(result.get("id"))
                or record.get("run_id") != result["id"]):
            raise ExpansionError("expansion_ack_binding_changed")
        source = record.get("source_raw_ack")
        if source is not None:
            from tools.daily_research.exa_transport import reconcile_start_ack
            if (not isinstance(source, dict)
                    or source.get("file") != f"{claim['date']}-exa-{claim['intent_sha256']}-start-http.json"
                    or reconcile_start_ack(_read_receipt(ledger, source), claim["intent"]["request"]) != result):
                raise ExpansionError("expansion_ack_source_binding_changed")
            refs = row.setdefault("exa_transport_receipts", [])
            if source not in refs:
                refs.append(source)
        claim.update(run_id=result["id"], state=result.get("status", "accepted"),
                     start_receipt={"file": filename, "sha256": _hash(raw), "bytes": len(raw)})
        if claim["state"] in TERMINAL:
            claim["terminal_receipt"] = claim["start_receipt"]
        row["exa_expansion"] = claim
        ledger.put(row)
        if claim.get("terminal_receipt"):
            return _project(claim, copy.deepcopy(result))
    if name == START and claim:
        return _project(claim)  # Another call ID cannot create another run.
    deadline = _deadline(row)
    if now >= deadline:
        return _skip("expansion_original_deadline_exhausted") if not claim else _project(claim)
    if name == START and args["max_cost_micros"] < ULTRA_MIN_MICROS:
        # Pure argument check: needs no credential or allocation and consumes no claim.
        return {**_skip("expansion_cap_below_ultra_minimum"),
                "action": "No Exa start was made and no claim was consumed. If the expansion still serves a useful gap, call once "
                          f"more with max_cost_micros from {ULTRA_MIN_MICROS} to {LIMIT_MICROS} within the shared "
                          "$5 research allocation; otherwise continue ordinary research."}
    if transport is None or not callable(getattr(transport, "start", None)) or not callable(getattr(transport, "read", None)):
        outcome = _skip(unavailable_reason) if not claim else _project(claim)
        if allocation_status is not None:
            outcome["allocation_status"] = allocation_diagnostic(row, allocation, now=now)
        return outcome
    if name == START:
        if phase != "research" or row.get("state") != "running" or row.get("qa") or row.get("raw_output_digest"):
            return _skip("expansion_before_final_qa_only")
        request = {"query": args["query"], "effort": "ultra",
                   "budget": {"maxCostDollars": args["max_cost_micros"] / 1_000_000}}
        if not _cap_supported(tool_schema, request):
            return _skip("expansion_supported_native_cap_unverified")
        problem = _allocation_problem(allocation, row, args["max_cost_micros"], now, deadline)
        if problem == "cap_exceeds_remaining":
            remaining = allocation["remaining_micros"]
            fits = remaining >= ULTRA_MIN_MICROS
            return {**_skip("expansion_cap_exceeds_remaining_allocation"), "remaining_micros": remaining,
                    "action": "No Exa start was made and no claim was consumed. " + (
                        f"Call once more with max_cost_micros from {ULTRA_MIN_MICROS} to {min(remaining, LIMIT_MICROS)} "
                        "if that still serves the gap; otherwise continue ordinary research." if fits else
                        "No Ultra cap fits the remaining verified allocation; continue ordinary research.")}
        if problem:
            return _skip("expansion_remaining_all_in_allocation_unverified")
        intent = {"run_key": row["run_key"], "session_id": row["session_id"], "turn_id": row["turn_id"],
                  "deadline": deadline.isoformat(), "authority_reference": row["recurring_budget_authority_reference"],
                  "request": request,
                  "allocation": copy.deepcopy(allocation), "tool_schema_sha256": _hash(_bytes(tool_schema))}
        claim = {"date": row["date"], "run_key": row["run_key"], "intent": intent,
                 "intent_json": _bytes(intent).decode(),
                 "intent_sha256": _hash(_bytes(intent)), "cap_micros": args["max_cost_micros"],
                 "state": "submission_unresolved", "attempted": True, "run_id": None}
        if row["date"] != now.astimezone(ZoneInfo("America/Chicago")).date().isoformat():
            return _skip("expansion_actual_daily_date_required")
        if admit:
            admit(row)
        row["exa_expansion"] = claim
        ledger.put(row)  # Caller holds daily lock/fence; claim permanently precedes POST.
        if admit:
            admit(row)
        if current_time() >= deadline or not _allocation(allocation, row, args["max_cost_micros"], current_time(), deadline):
            return {**_project(claim), "reason": "expansion_admission_expired_before_submission",
                    "action": "No provider call was made; retain the consumed claim and allocation for host reconciliation."}
        try:
            result = transport.start(copy.deepcopy(intent["request"]))
        except Exception:  # noqa: BLE001 - an uncertain POST is never repeated
            return _project(claim)
    else:
        if not claim.get("run_id"):
            return _project(claim)
        if admit:
            admit(row)
        if current_time() >= deadline:
            return _project(claim)
        try:
            result = transport.read(claim["run_id"])
        except Exception:  # noqa: BLE001 - safe retry is an original-ID read only
            return {**_project(claim), "ok": False, "reason": "expansion_original_read_unavailable"}
    if not isinstance(result, dict) or not _valid_id(result.get("id")):
        return _project(claim)
    if claim.get("run_id") and result["id"] != claim["run_id"]:
        raise ExpansionError("expansion_provider_id_changed")
    claim.update(run_id=result["id"], state=result.get("status") if isinstance(result.get("status"), str) else "accepted")
    record = {"intent_sha256": claim["intent_sha256"], "run_id": result["id"],
              "observed_at": now.isoformat(), "provider_record": result}
    label = "start" if name == START else "read-" + _hash(_bytes(record))
    ref = _receipt(ledger, claim, label, _bytes(record))
    claim["start_receipt" if name == START else "last_receipt"] = ref
    if claim["state"] in TERMINAL:
        claim["terminal_receipt"] = ref
    row["exa_expansion"] = claim
    ledger.put(row)
    return _project(claim, copy.deepcopy(result))


def _valid_id(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}", value) is not None
