"""Parallel FindAll list-building tools for the existing leased daily research caller.

Owner decision 2026-10-04: FindAll is enabled under the same owner-set per-run paid
expansion allowance as Exa (allocation.py). A create is admitted only by
``allocation.problem(..., source="findall")`` against the row's frozen grant and the
live control fences (brake, source_commit, valid_until), and then by the canonical
``parallel_findall`` grant for that exact request. Its durable claim reserves the
request's whole ``maximum_cost_usd``; an uncertain create keeps that reservation.
A run still active when research ends or reaches its original deadline is
cancelled through the provider's cancel API (``settle``), which then retains one
free result snapshot of every run it cancels or finds terminal.

A status or result read alarms only its provider GET, for at most
``FINDALL_READ_BOUND_SECONDS``; every store call runs outside that alarm, so an
expiry can never interrupt the store bridge mid-call. Creates keep the 15 s bound.

The registry, instructions and binding checks are standard library only. The
canonical FindAll closure (``blueprint_pipeline.parallel_findall*``) is imported only
when a handler is built or used, so importing this module never loads a credential
or calls a provider.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from tools.daily_research import allocation, search
from tools.daily_research.runner import MAX_ADAPTIVE_RUNTIME_SECONDS, Refusal, digest, identifier

CREATE = "blueprint_findall_create"
STATUS = "blueprint_findall_status"
RESULT = "blueprint_findall_result"
NAMES = frozenset({CREATE, STATUS, RESULT})
PROFILE = "parallel-findall-v1"
API_KEY_ENV = "PARALLEL_API_KEY"  # parallel_findall.API_KEY_ENV; only its presence is checked here
SUBMISSIONS_FIELD = allocation.FINDALL_FIELD  # parallel_findall_owner.SUBMISSIONS_FIELD
READS_FIELD = "parallel_findall_reads"
SETTLEMENTS_FIELD = "parallel_findall_settlements"
OWNER_FIELDS = (SUBMISSIONS_FIELD, READS_FIELD, SETTLEMENTS_FIELD)
# The portable release projects the canonical closure beside this module
# (standalone.RUNTIME_PREFIX): the WebApp installer admits only tools/daily_research/.
RUNTIME_DIRECTORY = "pipeline_runtime"
CREATE_FIELDS = frozenset({"objective", "entity_type", "generator", "match_limit", "match_conditions",
                           "maximum_cost_usd"})
CAP_PROBLEMS = frozenset({"paid_expansion_cap_exceeds_remaining", "paid_expansion_cap_exceeds_per_start_maximum"})
CLIENT_TIMEOUT_SECONDS = 10  # Per socket operation; inside the 15 s create bound.
TOOL_BOUND_SECONDS = 15  # search.respond's application-tool bound, which creates keep.
# A status/result GET returns every evaluated candidate (no pagination) and grows with
# the run; its alarm covers only that GET and encoding, never a store call.
FINDALL_READ_BOUND_SECONDS = 120
FIRST_MATCH_LIMIT = 50  # The recommended first match_limit per request.
MAX_CANCEL_ATTEMPTS = 3
MAX_SETTLEMENT_PASSES = 5
SETTLED = frozenset({"terminal", "provider_id_unknown", "cancel_attempts_exhausted", "settlement_unconfirmed"})
CENTRAL = ZoneInfo("America/Chicago")
STAY = "No FindAll run was started and no claim was consumed. "


def tools():
    """Definitions for the combined daily registry, frozen into each session intent."""
    properties = {
        "objective": {"type": "string"}, "entity_type": {"type": "string"},
        "generator": {"type": "string", "enum": ["preview", "base", "core", "pro"]},
        "match_limit": {"type": "integer", "minimum": 5, "maximum": 1000},
        "match_conditions": {"type": "array", "items": {"type": "object",
            "additionalProperties": False, "properties": {
                "name": {"type": "string"}, "description": {"type": "string"}},
            "required": ["name", "description"]}},
        "maximum_cost_usd": {"type": "string"},
    }
    definitions = [{"type": "function", "name": CREATE, "defer_loading": False,
        "description": "Start one asynchronous Parallel FindAll enumeration early in research, before deep individual "
                       "verification and QA. Choose objective, entity type, positive evidence conditions, generator and "
                       f"match_limit; start with a match_limit of about {FIRST_MATCH_LIMIT}. The result lists every "
                       "evaluated candidate, including non-matches, so it is larger than match_limit suggests. "
                       "maximum_cost_usd is a USD string with at most two decimals that covers the "
                       "generator's price for match_limit; the host reserves all of it from this run's owner-set "
                       "paid expansion allowance, shared with Exa, and admits at most half of that allowance per "
                       "start. A refusal names max_start_micros and remaining_micros and starts nothing. Matches "
                       "are discovery hypotheses. No automatic retry follows an uncertain start.",
        "parameters": {"type": "object", "additionalProperties": False,
                       "properties": properties, "required": list(properties)}}]
    for name, description in (
        (STATUS, ("Read the status of a FindAll run that this daily run started, by its findall_id. Poll it "
                  "until the run is no longer active before reading its result. Each check uses one of this "
                  "run's tool calls, so keep researching between checks and check about every one to two minutes.")),
        (RESULT, ("Read the result of a FindAll run that this daily run started, once it is no longer active: a "
                  "raw discovery snapshot preserving every evaluated candidate (matches and non-matches), "
                  "citations, basis, reasoning, status and unknown fields. Matches are not verified or qualified "
                  "leads. Read each run's result once. A large snapshot is retained whole and returned as lossless "
                  "24,000-character JSON fragments; page through it with the returned receipt sha256 and "
                  "next_page, which never reads a new snapshot. Concatenate pages in order to recover the exact "
                  "snapshot. A result too large for the host to retain is refused before anything is stored; then "
                  "start a narrower or smaller request.")),
    ):
        definitions.append({"type": "function", "name": name, "defer_loading": False,
            "description": description, "parameters": {"type": "object",
                "additionalProperties": False, "properties": {"findall_id": {"type": "string"},
                    # The host binds receipt_sha256 to an owned receipt; no pattern keyword
                    # that a session echo could normalize away.
                    **({"receipt_sha256": {"type": "string"},
                        "page": {"type": "integer", "minimum": 0}} if name == RESULT else {})},
                "required": ["findall_id"]}})
    return definitions


def runtime():
    """The canonical stdlib FindAll closure: the source checkout first, else the release projection."""
    try:
        import blueprint_pipeline  # noqa: F401 - the canonical package wherever it is importable
    except ModuleNotFoundError as exc:
        projected = Path(__file__).resolve().with_name(RUNTIME_DIRECTORY)
        if exc.name != "blueprint_pipeline" or not (projected / "blueprint_pipeline" / "__init__.py").is_file():
            raise
        if str(projected) not in sys.path:
            sys.path.append(str(projected))  # Appended: it never shadows another module.
    from blueprint_pipeline import (
        paid_resource_admission,
        parallel_findall,
        parallel_findall_admission,
        parallel_findall_execution,
        parallel_findall_owner,
    )
    return SimpleNamespace(admission=paid_resource_admission, api=parallel_findall,
                           issuer=parallel_findall_admission, execution=parallel_findall_execution,
                           owner=parallel_findall_owner)


def instructions():
    rt = runtime()
    execution = rt.execution
    prices = "; ".join(f"{generator} ${fixed} per run plus ${per_match} per match"
                       for generator, (fixed, per_match) in execution._RATES.items())
    # The same listed estimate prepare_submission computes for a base request of FIRST_MATCH_LIMIT.
    base_fixed, base_per_match = execution._RATES["base"]
    first = Decimal(base_fixed) + Decimal(base_per_match) * FIRST_MATCH_LIMIT
    retention_mib = rt.owner.MAX_SNAPSHOT_BYTES // (1024 * 1024)
    return (" Parallel FindAll application tools (blueprint_findall_create, blueprint_findall_status and "
            "blueprint_findall_result) support asynchronous list building. For broad operating-site discovery, "
            "prefer usable operator location lists and other sourced bulk inputs first; use FindAll early "
            "when it adds useful coverage after defining the scope from existing company context, before spending most "
            "of the run on individual Perplexity searches or deep verification. Preliminary searches are "
            "useful only when needed to clarify that scope; a fully researched seed prospect is not required. "
            "Enumerate named physical sites/operators within the admitted geography, not generic company "
            "headquarters. Match conditions at this stage establish sourced category/location plausibility; "
            "do not require verified manual workflow, robot fit, hiring, contacts or buying interest to retain "
            "a raw possibility. Keep missing facts explicitly unknown for later verification. Choose the objective, "
            "entity type, simple positive match conditions, generator and match_limit yourself; preview allows 5 to "
            f"10 matches. Start with a match_limit of about {FIRST_MATCH_LIMIT} per request, and raise it only when "
            "a completed result shows that a segment needs more. "
            f"Versioned prices ({execution.PRICING_VERSION}): {prices}, times match_limit. "
            "maximum_cost_usd is a USD string with at most two decimals and at least that estimate; the host "
            "reserves all of it until the run's actual cost is known, so a larger value only uses more allowance. "
            f"At a $10 combined allowance, one base request of {FIRST_MATCH_LIMIT} matches has a listed estimate "
            f"of ${first} (${base_fixed} per run plus {FIRST_MATCH_LIMIT} matches at ${base_per_match}) and "
            f"reserves ${first}, leaving ${Decimal('10.00') - first} for further distinct requests or an Exa "
            "start; these are requested limits, not promised or qualified matches. "
            "Use distinct useful segments and deduplicate sites; do not spend two requests on the same list. "
            "The general admission rule, current prices and actual remaining allowance decide every request; "
            "the example is not extra authority or a required provider quota. "
            "FindAll draws on this run's separate host-reserved paid expansion allowance, shared with Exa and frozen "
            "from the owner's direction at run start; the research soft target is unchanged and neither is extra "
            "authority. The host admits at most half of the owner-set allowance per start and only within what "
            "remains; a refusal names max_start_micros and remaining_micros and starts nothing. Creates are refused "
            "after research output, in QA or repair, after the original deadline, under the owner's brake or when "
            "the owner's direction omits FindAll; retain that actionable skip and continue ordinary research. "
            "Retain each returned findall_id. Poll blueprint_findall_status until the run is no longer active "
            "before reading its result, within the original deadline; when research ends the host cancels any "
            "run still active and retains one result snapshot of it. Each status check uses one of this run's tool "
            "calls, so keep researching between checks and check about every one to two minutes. "
            "An uncertain start must never be restarted. "
            "A result lists every candidate the run evaluated, including non-matches, so it is much larger than "
            "match_limit suggests. Read each completed run's result once; every read retains a new immutable "
            "snapshot. A result larger than the tool output ceiling is retained whole as one immutable file and "
            "returned in 24,000-character json_fragment pages, never truncated. Page through it with its "
            "receipt_sha256 and page; these pages slice the same snapshot without a new provider read. "
            "Concatenate json_fragment values in page order for the complete raw JSON. A result above the host's "
            f"{retention_mib} MiB retention limit is refused as findall_snapshot_too_large before anything is "
            "stored; its findall_id stays recorded, so start a narrower or smaller request instead of reading it "
            "again. Preserve all raw candidates, provider status, basis, reasoning, citations and "
            "unknowns for independent qualification. Provider matches are discovery only, never verified or "
            "CRM-ready leads.")


def installed_profile(api):
    handler = getattr(api, "findall_application_tools", None)
    if handler is None:
        return None
    if not isinstance(handler, FindAllApplicationTools):
        raise Refusal("findall_tool_owner_dependencies_required")
    return PROFILE


def check_binding(row):
    """The optional registry is frozen with the durable session intent."""
    expected = digest(tools())
    if (row.get("findall_profile") != PROFILE
            or row.get("metadata", {}).get("findall_tools_digest") != expected
            or row.get("preflight", {}).get("findall_tools_digest") != expected):
        raise Refusal("findall_tool_registry_binding_changed")


def stable_errors():
    """Secret-free FindAll error classes whose codes reach the agent unchanged."""
    try:
        rt = runtime()
    except ImportError:
        return ()
    return (rt.api.FindAllError, rt.admission.PaidResourceAdmissionBlocked)


def research_deadline(row):
    """The original research deadline (the grant's valid_until never exceeds it), or None."""
    seconds = row.get("research_runtime_seconds") if isinstance(row, dict) else None
    try:
        started = datetime.fromisoformat(row["started_at"])
    except (KeyError, TypeError, ValueError):
        return None
    if started.tzinfo is None or type(seconds) is not int or not 0 < seconds <= MAX_ADAPTIVE_RUNTIME_SECONDS:
        return None
    return started + timedelta(seconds=seconds)


def settlement_due(row, now):
    """True once research ended or reached its original deadline with a run not yet settled."""
    entries = row.get(SUBMISSIONS_FIELD) if isinstance(row, dict) else None
    if not isinstance(entries, dict) or not entries:
        return False
    deadline = research_deadline(row)
    if row.get("state") == "running" and deadline is not None and now < deadline:
        return False
    records = row.get(SETTLEMENTS_FIELD) or {}
    return any((records.get(key) or {}).get("state") not in SETTLED for key in entries)


def status(row):
    """Read-only summary of this row's FindAll claims for status output; never manufactures cost."""
    entries = row.get(SUBMISSIONS_FIELD) or {}
    records = row.get(SETTLEMENTS_FIELD) or {}
    claims = [claim for claim in allocation.claims(row) if claim["source"] == "findall"]
    amounts = [claim["reserved_micros"] for claim in claims]
    return {"claims": len(entries),
            "reserved_micros": sum(amounts) if all(type(a) is int for a in amounts) else None,
            "states": sorted(str(entry.get("state")) for entry in entries.values() if isinstance(entry, dict)),
            "settlements": sorted(str((records.get(key) or {}).get("state", "unsettled")) for key in entries),
            "billing_verified": False}


def receipt_refs(row):
    """Every immutable FindAll file this row binds: created runs, owned reads and settlement reads.

    Each snapshot is one file; its pages are slices of that file, never separate files.
    """
    refs = [{"file": entry.get("receipt_file"), "sha256": entry.get("receipt_sha256")}
            for entry in (row.get(SUBMISSIONS_FIELD) or {}).values()
            if isinstance(entry, dict) and entry.get("receipt_file")]
    refs += [{key: read.get(key) for key in ("file", "sha256", "bytes")}
             for read in (row.get(READS_FIELD) or {}).values() if isinstance(read, dict)]
    refs += [dict(ref) for record in (row.get(SETTLEMENTS_FIELD) or {}).values() if isinstance(record, dict)
             for ref in record.get("receipts", []) if isinstance(ref, dict)]
    return refs


def validate_snapshot_exports(row, read_bytes):
    """Every snapshot receipt binds its whole file: exact bytes, page layout and complete JSON."""
    receipts = [entry["receipt"] for entry in (row.get(SUBMISSIONS_FIELD) or {}).values()
                if isinstance(entry, dict) and entry.get("receipt") is not None]
    receipts += [read for read in (row.get(READS_FIELD) or {}).values() if isinstance(read, dict)]
    receipts += [ref for record in (row.get(SETTLEMENTS_FIELD) or {}).values() if isinstance(record, dict)
                 for ref in record.get("receipts", []) if isinstance(ref, dict)]
    for receipt in receipts:
        runtime().owner.validate_snapshot(receipt, read_bytes)


def tool_bound(handler, name, seconds):
    """search.respond's wall-time bound for one FindAll call, given the phase's remaining seconds.

    A create, or a call without a handler, keeps the 15 s application-tool alarm around
    the whole call. A status or result read arms no alarm here: its handler alarms only
    the provider GET, for min(FINDALL_READ_BOUND_SECONDS, seconds), and stores outside it.
    """
    if name != CREATE and isinstance(handler, FindAllApplicationTools):
        return handler.read_window(seconds)
    return search.bounded_request(min(TOOL_BOUND_SECONDS, seconds))


def _failure_code(exc, stable):
    """A fixed, secret-free code: FindAll, tool-watchdog and store refusals already carry one."""
    return str(exc) if isinstance(exc, (search.ToolFailure, Refusal, *stable)) else "findall_result_observation_failed"


def unavailable(name):
    """A pinned session whose process has no handler continues ordinary research."""
    return {"ok": False, "state": "skipped", "reason": "findall_runtime_unavailable", "provider": "parallel_findall",
            "claim_created": False, "provider_started": False,
            "action": (STAY if name == CREATE else "No FindAll read was made. ") + (
                "This worker has no FindAll binding. Retain the list-building gap and continue ordinary research; "
                "a run already started keeps its claim and reservation.")}


def _operation_key(operation_id):
    return hashlib.sha256(operation_id.encode()).hexdigest()


def restore(row, ledger):
    """After a failed write, mirror exactly what the store holds; keep the possibly committed copy otherwise."""
    try:
        stored = ledger.get(row["date"])
    except Exception:  # noqa: BLE001 - an unreadable store keeps the conservative in-memory claims
        return
    if isinstance(stored, dict):
        for field in OWNER_FIELDS:
            if field in stored:
                row[field] = copy.deepcopy(stored[field])
            else:
                row.pop(field, None)


class _MirroringLedger:
    """Keep the caller's pending row from overwriting newly durable claims."""
    def __init__(self, ledger, row):
        self.ledger, self.row = ledger, row

    def __getattr__(self, name):
        return getattr(self.ledger, name)

    def put(self, value):
        if value.get("date") != self.row.get("date"):
            raise Refusal("findall_tool_owner_row_changed")
        # Mirror before the store attempt. A failure must keep the caller from
        # replacing a possibly committed claim with its older in-memory row.
        for field in (SUBMISSIONS_FIELD, READS_FIELD):
            if field in value:
                self.row[field] = copy.deepcopy(value[field])
        try:
            return self.ledger.put(value)
        except Exception:
            # A refused write committed nothing; a lost reply may have committed.
            # Either way the caller now mirrors the store's actual claims.
            restore(self.row, self.ledger)
            raise


class FindAllApplicationTools:
    """The leased daily caller's FindAll tools; every create is admitted by the shared allocation.

    The caller already holds the ledger lock or fenced lease; ``assert_current_lease``
    asserts it freshly, returns literal True and never acquires another lock.
    ``control`` returns live company control (brake, current direction, source_commit).
    This class never reads a credential: ``client`` arrives preconfigured.
    """
    names = NAMES

    def __init__(self, *, ledger, client, control, assert_current_lease, clock=None, stopped=None):
        self.rt = runtime()
        if (not isinstance(client, self.rt.execution.AdmittedFindAllClient)
                or any(not callable(fn) for fn in (control, assert_current_lease))
                or any(fn is not None and not callable(fn) for fn in (clock, stopped))):
            raise self.rt.api.FindAllError("findall_tool_owner_dependencies_required")
        self.ledger, self.client, self.control = ledger, client, control
        self.assert_current_lease = assert_current_lease
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.stopped = stopped or (lambda: False)
        self._read_until = None  # Monotonic end of search.respond's current read window.

    @contextmanager
    def read_window(self, seconds):
        """search.respond's bound for one status/result call: no alarm, only the phase's remaining time."""
        if seconds <= 0:
            raise search.ToolFailure("research_tool_absolute_deadline")
        self._read_until = time.monotonic() + seconds
        try:
            yield
        finally:
            self._read_until = None

    def provider_read(self, read, findall_id):
        """One free provider GET and its encoding, alarmed for min(FINDALL_READ_BOUND_SECONDS, remaining).

        Nothing is stored inside this alarm, so its expiry (research_tool_absolute_deadline)
        always comes before any store write and never interrupts a store call. A snapshot
        above the retention limit is refused here (FindAllSnapshotTooLarge), before any write.
        """
        seconds = FINDALL_READ_BOUND_SECONDS
        if self._read_until is not None:
            seconds = min(seconds, self._read_until - time.monotonic())
        with search.bounded_request(seconds):
            return self.rt.owner.encode_snapshot(read(findall_id))

    def assert_fresh_caller(self, row):
        """Refuse stale journal/call rows before the caller can overwrite them."""
        if self.assert_current_lease() is not True:
            raise Refusal("findall_owner_current_lease_required")
        stored = self.ledger.get(row["date"])
        if (not isinstance(stored, dict) or stored.get("run_key") != row.get("run_key")
                or any(digest(stored.get(field, {})) != digest(row.get(field, {}))
                       for field in ("application_tool_calls", *OWNER_FIELDS,
                                     "exa_expansion", "exa_transport_receipts"))):
            raise Refusal("findall_tool_stale_owner_row")

    def create_problem(self, owner, prepared, phase):
        """None while this exact create is admitted now; otherwise a named refusal."""
        control = self.control()
        now = self.clock()
        if not isinstance(owner, dict) or owner.get("findall_profile") != PROFILE:
            return "findall_existing_owner_record_required"
        if (phase != "research" or owner.get("state") != "running" or owner.get("qa")
                or owner.get("raw_output_digest")):
            return "findall_before_final_qa_only"
        if self.stopped():
            return "findall_stopped"
        deadline = research_deadline(owner)
        if deadline is None or now >= deadline:
            return "findall_original_deadline_exhausted"
        if owner.get("date") != now.astimezone(CENTRAL).date().isoformat():
            return "findall_actual_daily_date_required"
        cap = allocation.findall_micros(prepared.get("maximum_cost_usd") if isinstance(prepared, dict) else None)
        grant = owner.get("paid_expansion_grant")
        code = allocation.problem(grant, allocation.claims(owner), cap, now, control=control,
                                  source="findall")
        if code is None and grant.get("run_key") != owner.get("run_key"):
            code = "paid_expansion_grant_invalid"
        return code

    def authority(self, current, scope, phase):
        """The owner journal's action-time check, rerun under the held lease before the one POST."""
        if self.assert_current_lease() is not True or self.stopped():
            return False
        if not isinstance(current, dict) or current.get("findall_profile") != PROFILE:
            return False
        if isinstance(scope, dict) and scope.get("operation") in {"status", "result"}:
            return True  # Free reads of an owned run; they spend nothing.
        return self.create_problem(current, scope, phase) is None

    def settlement_reason(self, row, now):
        """Cancellation is due at row end, or on fresh loss of paid authority."""
        entries = row.get(SUBMISSIONS_FIELD) or {}
        records = row.get(SETTLEMENTS_FIELD) or {}
        if not any((records.get(key) or {}).get("state") not in SETTLED for key in entries):
            return None
        if settlement_due(row, now):
            return "original_research_deadline" if row.get("state") == "running" else "research_ended"
        if self.stopped():
            return "findall_stopped"
        try:
            control = self.control()
            now = self.clock()  # The control read may cross an expiry boundary.
            grant = row.get("paid_expansion_grant")
            code = allocation.standing(grant, control, now, source="findall")
            if code:
                return code
            found = allocation.claims(row)
            limit = allocation.effective_limit(grant, control)
            room = allocation.headroom(grant, found, limit)
            if room["reserved_micros"] is None:
                return "paid_expansion_claims_unverified"
            if (room["reserved_micros"] > limit or any(
                    claim["source"] == "findall" and claim["reserved_micros"] > allocation.per_start_max(limit)
                    for claim in found)):
                return "paid_expansion_live_limit_lowered"
        except Exception:  # noqa: BLE001 - unknown live authority conservatively cancels and retains reservations
            return "paid_expansion_authority_unverified"
        return None

    def admit(self, day, prepared, phase):
        """The exact-request broker: allocation blockers decide the canonical grant."""
        owner = self.ledger.get(day)
        code = self.create_problem(owner, prepared, phase)
        return self.rt.issuer.admit_exact_request(prepared, blockers=[code] if code else [])

    def room(self, owner, code):
        """Headroom named in a refusal; nothing can start while a standing fence refuses."""
        grant = owner.get("paid_expansion_grant") if isinstance(owner, dict) else None
        if allocation.granted(grant):
            return {"remaining_micros": None, "max_start_micros": None}
        try:
            control = self.control()
        except Exception:  # noqa: BLE001 - unknown live control names no headroom
            return {"remaining_micros": None, "max_start_micros": None}
        found = allocation.headroom(grant, allocation.claims(owner), allocation.effective_limit(grant, control))
        return {"remaining_micros": found["remaining_micros"],
                "max_start_micros": found["max_start_micros"] if code in CAP_PROBLEMS or code is None else 0}

    def skip(self, owner, code, guidance, **extra):
        return {"ok": False, "state": "skipped", "reason": code, "provider": "parallel_findall",
                "claim_created": False, "provider_started": False, **self.room(owner, code), **extra,
                "action": STAY + guidance}

    def execute(self, action, *, row, phase):
        FindAllError = self.rt.api.FindAllError
        if self.assert_current_lease() is not True:
            raise FindAllError("findall_owner_current_lease_required")
        name = action.get("name")
        args = action.get("arguments")
        if phase not in {"research", "qa", "repair"}:
            raise FindAllError("findall_tool_phase_invalid")
        tid = row.get("turn_id") if phase == "research" else (
            row.get("validation_repairs", [{}])[-1].get("turn_id") if phase == "repair"
            else row.get("qa", {}).get("turn_id"))
        if (name not in NAMES or action.get("type") != "function_call"
                or not tid or action.get("turn_id") != tid or not isinstance(args, dict)):
            raise FindAllError("findall_tool_action_binding_invalid")
        cid = identifier(action.get("call_id"))
        binding = {k: action.get(k) for k in ("turn_id", "call_id", "name", "arguments")}
        prior = row.get("application_tool_calls", {}).get(cid)
        owner = self.ledger.get(row["date"])
        stored = owner.get("application_tool_calls", {}).get(cid) if isinstance(owner, dict) else None
        if (not isinstance(prior, dict) or prior.get("attempted") is not True
                or prior.get("request_digest") != digest(binding) or not isinstance(stored, dict)
                or stored.get("attempted") is not True or stored.get("request_digest") != digest(binding)
                or owner.get("run_key") != row.get("run_key")):
            raise FindAllError("findall_tool_durable_call_claim_required")
        ledger = _MirroringLedger(self.ledger, row)
        if name == CREATE:
            return self._create(args, row, owner, phase, cid, ledger)
        return self._read(name, args, row, owner, phase, binding, cid, ledger)

    def _create(self, args, row, owner, phase, cid, ledger):
        rt = self.rt
        if set(args) != CREATE_FIELDS:
            raise rt.api.FindAllError("findall_tool_arguments_invalid")
        cost = args["maximum_cost_usd"]
        cap = allocation.findall_micros(cost)
        if cap is None:
            raise rt.api.FindAllError("findall_maximum_cost_usd_invalid")
        spec = {key: value for key, value in args.items() if key != "maximum_cost_usd"}
        operation_id = row["run_key"] + ":findall:" + cid
        claimed = (owner.get(SUBMISSIONS_FIELD) or {}).get(_operation_key(operation_id))
        if isinstance(claimed, dict):
            # This call already consumed its one claim: report it, never another create.
            return {"ok": claimed.get("findall_id") is not None, "state": claimed.get("state"),
                    "provider": "parallel_findall", "findall_id": claimed.get("findall_id"),
                    "reserved_micros": allocation.findall_micros(claimed.get("prepared", {}).get("maximum_cost_usd")),
                    "replay_permitted": False, "billing_verified": False}
        try:
            prepared = rt.execution.prepare_submission(spec, operation_id=operation_id, maximum_cost_usd=cost)
        except rt.api.FindAllError as exc:
            estimate = {}
            try:
                estimate["estimated_maximum_cost_usd"] = rt.execution.prepare_submission(
                    spec, operation_id=operation_id, maximum_cost_usd="1000000")["estimated_maximum_cost_usd"]
            except rt.api.FindAllError:
                pass
            return self.skip(owner, str(exc), "Correct the request or continue ordinary research.", **estimate)
        code = self.create_problem(owner, prepared, phase)
        if code in CAP_PROBLEMS:
            return self.skip(owner, code, "Use maximum_cost_usd no larger than max_start_micros (USD micros) with a "
                             "generator and match_limit whose estimate fits, or continue ordinary research.",
                             estimated_maximum_cost_usd=prepared["estimated_maximum_cost_usd"])
        if code:
            return self.skip(owner, code, "FindAll is not admitted now; retain this list-building gap and continue "
                             "ordinary research.")
        try:
            grant = self.admit(row["date"], prepared, phase)
        except rt.admission.PaidResourceAdmissionBlocked as exc:
            return self.skip(owner, exc.blockers[0] if exc.blockers else "paid_resource_admission_blocked",
                             "FindAll is not admitted now; continue ordinary research.")
        try:
            rt.owner.create_under_owner_lease(
                self.client, spec, ledger=ledger, day=row["date"], operation_id=operation_id,
                maximum_cost_usd=cost, paid_resource_admission_grant=grant,
                current_authority=lambda current, scope: self.authority(current, scope, phase),
                assert_current_lease=self.assert_current_lease)
        except rt.execution.FindAllSubmissionUnresolved as exc:
            return {"ok": False, "state": "submission_unresolved", "reason": "findall_submission_unresolved",
                    "provider": "parallel_findall", "findall_id": exc.findall_id, "reserved_micros": cap,
                    "replay_permitted": False, "billing_verified": False,
                    "action": "Preserve this claim and its whole reservation; never start a replacement. " + (
                        "Poll status or result with this findall_id within the original deadline."
                        if exc.findall_id else "No findall_id is known; it stays an unknown cost.")}
        shown = self.rt.owner.snapshot_page(
            ledger.get(row["date"])[SUBMISSIONS_FIELD][_operation_key(operation_id)]["receipt"],
            ledger.read_bytes)
        if "snapshot" in shown:
            shown["run"] = shown.pop("snapshot")
        return {"ok": True, "state": "created", "provider": "parallel_findall", "evidence_scope": "discovery_only",
                "findall_id": args.get("findall_id") or ledger.get(row["date"])[SUBMISSIONS_FIELD][
                    _operation_key(operation_id)]["findall_id"],
                "reserved_micros": cap, "replay_permitted": False, "billing_verified": False, **shown}

    def _read(self, name, args, row, owner, phase, binding, cid, ledger):
        FindAllError = self.rt.api.FindAllError
        cursor = name == RESULT and set(args) == {"findall_id", "receipt_sha256", "page"}
        if set(args) != {"findall_id"} and not cursor:
            raise FindAllError("findall_tool_arguments_invalid")
        slots = owner.get(SUBMISSIONS_FIELD, {}) if isinstance(owner, dict) else {}
        if not any(entry.get("findall_id") == args["findall_id"] for entry in slots.values()):
            raise FindAllError("findall_tool_run_not_owned")
        read_scope = {"operation": "status" if name == STATUS else "result",
                      "findall_id": args["findall_id"], "call_binding_digest": digest(binding)}
        if not self.authority(owner, read_scope, phase):
            raise FindAllError("findall_tool_current_authority_required")
        if cursor:
            matches = [read for read in (owner.get(READS_FIELD) or {}).values()
                       if read.get("operation") == "result" and read.get("findall_id") == args["findall_id"]
                       and read.get("sha256") == args["receipt_sha256"]]
            matches += [entry["receipt"] for entry in slots.values()
                        if entry.get("findall_id") == args["findall_id"] and isinstance(entry.get("receipt"), dict)
                        and entry["receipt"].get("sha256") == args["receipt_sha256"]]
            if not matches:
                raise FindAllError("findall_snapshot_cursor_not_owned")
            receipt = matches[0]
            # One store read of the same immutable file; no provider read.
            shown = self.rt.owner.snapshot_page(receipt, ledger.read_bytes, args["page"])
        else:
            try:
                raw, facts = self.provider_read(self.client.status if name == STATUS else self.client.result,
                                                args["findall_id"])
            except self.rt.owner.FindAllSnapshotTooLarge as exc:
                return {"ok": False, "state": "snapshot_too_large", "reason": str(exc), "provider": "parallel_findall",
                        "findall_id": args["findall_id"], "snapshot_bytes": exc.snapshot_bytes,
                        "maximum_snapshot_bytes": self.rt.owner.MAX_SNAPSHOT_BYTES, "snapshot_retained": False,
                        "action": "This FindAll snapshot is larger than the host can retain, so nothing was stored. "
                                  "The run, its findall_id and its whole reservation stay recorded, so a later "
                                  "recovery can still read it; reading it again returns this same refusal. Run a "
                                  "narrower or smaller request (tighter match conditions or a lower match_limit) "
                                  "within the remaining allowance, or continue ordinary research."}
            # Outside the GET alarm: one write and one readback of one immutable file.
            filename = row["date"] + "-tool-findall-read-" + digest(binding) + ".json"
            receipt = self.rt.owner.store_snapshot(ledger, filename, raw, facts)
            owner.setdefault(READS_FIELD, {})[cid] = {
                **read_scope, **receipt, "checked_at": self.clock().isoformat(),
            }
            ledger.put(owner)
            shown = self.rt.owner.page_view(receipt, raw)  # The read-back-checked bytes; no second read.
        return {"ok": True, "provider": "parallel_findall", "evidence_scope": "discovery_only", **shown}

    def _observe(self, row, findall_id, record):
        """One free status GET retained as an immutable receipt; None when unavailable."""
        FindAllError = self.rt.api.FindAllError
        try:
            snapshot = self.client.status(findall_id)
        except FindAllError as exc:
            record["observation_error"] = str(exc)
            return None
        raw, facts = self.rt.owner.encode_snapshot(snapshot)
        name = f"{row['date']}-tool-findall-settle-{facts['sha256']}.json"
        receipt = self.rt.owner.store_snapshot(self.ledger, name, raw, facts)
        if receipt not in record["receipts"]:
            record["receipts"].append(receipt)
        status = snapshot.get("status") if isinstance(snapshot, dict) else None
        active = status.get("is_active") if isinstance(status, dict) else None
        record.update(provider_status=status.get("status") if isinstance(status, dict) else None,
                      is_active=active if type(active) is bool else None, observed_at=self.clock().isoformat())
        record.pop("observation_error", None)
        return active if type(active) is bool else None  # Unknown activity is treated as still active.

    def _observe_result(self, row, findall_id, record):
        """ONE free, bounded result read once settlement cancels a run or finds it terminal.

        It runs after the cancel and its durable outcome, so it never blocks the cancel
        path. The snapshot, with the matches already paid for, is retained as one
        immutable file. A failure records result_observation_error and settlement
        continues; the claim keeps its whole reservation either way.
        """
        try:
            raw, facts = self.provider_read(self.client.result, findall_id)
            receipt = self.rt.owner.store_snapshot(
                self.ledger, f"{row['date']}-tool-findall-settle-result-{facts['sha256']}.json", raw, facts)
        except Exception as exc:  # noqa: BLE001 - a stable code only; settlement continues
            record["result_observation_error"] = _failure_code(exc, stable_errors())
            if isinstance(exc, self.rt.owner.FindAllSnapshotTooLarge):
                record["result_snapshot_bytes"] = exc.snapshot_bytes
            return
        if receipt not in record["receipts"]:
            record["receipts"].append(receipt)
        record.update(result_receipt=receipt, result_observed_at=self.clock().isoformat())

    def settle(self, row, *, reason):
        """Cancel each run of this row that may still be active once research ended or its deadline passed.

        The claims keep their whole reservations (allocation.claims); only provider
        truth is recorded. A run whose findall_id is unknown cannot be cancelled and
        stays an unknown cost. A cancel is claimed durably before its POST and tried
        at most MAX_CANCEL_ATTEMPTS times; cancellation is not a refund. Once a run is
        cancelled or terminal, one free result read retains its paid-for matches.
        """
        FindAllError = self.rt.api.FindAllError
        if self.assert_current_lease() is not True:
            raise FindAllError("findall_owner_current_lease_required")
        entries = row.get(SUBMISSIONS_FIELD) or {}
        records = row.setdefault(SETTLEMENTS_FIELD, {})
        for key in sorted(entries):
            entry = entries[key]
            record = copy.deepcopy(records.get(key) or {"operation_id": entry.get("operation_id"), "findall_id": None,
                                                        "state": "unsettled", "cancel_attempts": 0, "passes": 0,
                                                        "receipts": []})
            if record.get("state") in SETTLED:
                continue
            findall_id = entry.get("findall_id")
            if not findall_id:
                record.update(state="provider_id_unknown", reservation_held=True, observed_at=self.clock().isoformat())
                records[key] = record
                self.ledger.put(row)
                continue
            if record.get("findall_id") not in (None, findall_id):
                raise FindAllError("findall_settlement_binding_changed")
            record.update(findall_id=findall_id, passes=record["passes"] + 1, reason=reason)
            active = self._observe(row, findall_id, record)
            if active is not False and record["state"] != "cancel_accepted" and record["cancel_attempts"] < MAX_CANCEL_ATTEMPTS:
                record.update(state="cancel_unresolved", cancel_attempts=record["cancel_attempts"] + 1,
                              cancel_requested_at=self.clock().isoformat())
                records[key] = copy.deepcopy(record)
                self.ledger.put(row)  # The cancel claim is durable before its one POST.
                try:
                    self.client.cancel(findall_id)
                    record["state"] = "cancel_accepted"
                    record.pop("cancel_error", None)
                except FindAllError as exc:
                    record.update(state="cancel_outcome_unknown", cancel_error=str(exc))
                active = self._observe(row, findall_id, record)
            ended = active is False or record["state"] == "cancel_accepted"
            if active is False:
                record["state"] = "terminal"
            elif record["state"] != "cancel_accepted" and record["cancel_attempts"] >= MAX_CANCEL_ATTEMPTS:
                record["state"] = "cancel_attempts_exhausted"
            elif record["passes"] >= MAX_SETTLEMENT_PASSES:
                record["state"] = "settlement_unconfirmed"
            records[key] = record
            self.ledger.put(row)  # The cancel outcome is durable before any result read.
            if ended and "result_receipt" not in record and "result_observation_error" not in record:
                self._observe_result(row, findall_id, record)
                self.ledger.put(row)


def owner_handler(provider):
    """The production handler for the fenced daily provider, or None (no FindAll tools are advertised).

    Built only when the worker environment holds PARALLEL_API_KEY; presence is the
    only check here. The value goes straight to the admitted client, which keeps it
    private, and is never logged, persisted or sent anywhere but the pinned Parallel host.
    """
    if API_KEY_ENV not in os.environ:
        return None
    try:
        rt = runtime()
        client = rt.execution.AdmittedFindAllClient(os.environ[API_KEY_ENV], timeout_seconds=CLIENT_TIMEOUT_SECONDS)
    except (ImportError, ValueError):  # FindAllError is a ValueError; never echo the binding
        return None
    ledger = provider.ledger

    def assert_lease():
        ledger.bridge.call("assert_lease")
        return True

    return FindAllApplicationTools(
        ledger=ledger, client=client, control=lambda: ledger.bridge.call("control"), assert_current_lease=assert_lease,
        clock=lambda: getattr(provider, "clock", lambda: datetime.now(timezone.utc))(),
        stopped=lambda: getattr(provider, "stopped", lambda: False)())


def runtime_status(api=None):
    """Local-process metadata only; never prove production binding by inference."""
    adapter = getattr(api, "findall_application_tools", None)
    return {"schema_version": "blueprint.findall-runtime-status.v1",
            "observation_scope": "local_process", "module_imported": True,
            "callable_handler_installed": isinstance(adapter, FindAllApplicationTools),
            "tool_names": sorted(NAMES), "credential_binding_name": API_KEY_ENV,
            "credential_binding_present": API_KEY_ENV in os.environ,
            "credential_value_inspected": False, "production_binding_verified": False,
            "admission": "allocation.problem(source=findall) + exact parallel_findall grant",
            "allocation_sources": list(allocation.SOURCES),
            "fresh_exact_admission_required": True, "paid_operations_authorized": False}


if __name__ == "__main__":
    print(json.dumps(runtime_status(), sort_keys=True))
