"""FindAll application handler for an existing leased daily-research caller.

The owner supplies the preconfigured client and existing admission broker.
Importing or inspecting this module never loads a credential or calls a provider.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import datetime, timezone

from blueprint_pipeline.parallel_findall import FindAllError
from blueprint_pipeline.parallel_findall_execution import (
    AdmittedFindAllClient,
    prepare_submission,
)
from blueprint_pipeline.parallel_findall_owner import (
    SUBMISSIONS_FIELD,
    create_under_owner_lease,
)

CREATE = "blueprint_findall_create"
STATUS = "blueprint_findall_status"
RESULT = "blueprint_findall_result"
NAMES = frozenset({CREATE, STATUS, RESULT})
PROFILE = "parallel-findall-v1"
READS_FIELD = "parallel_findall_reads"


def tools():
    """Definitions for the release owner's combined tool registry."""
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
        "description": "Start one admitted asynchronous Parallel FindAll discovery. Choose objective and evidence conditions. The owner independently checks current shared spending and disclosure authority. Provider matches remain discovery hypotheses. No automatic retry follows an uncertain start.",
        "parameters": {"type": "object", "additionalProperties": False,
                       "properties": properties, "required": list(properties)}}]
    for name, description in (
        (STATUS, "Read status of a FindAll run retained in this daily owner's journal."),
        (RESULT, "Read the complete current raw discovery snapshot, preserving candidates, citations, basis, reasoning, status and unknown fields. Matches are not verified or qualified leads."),
    ):
        definitions.append({"type": "function", "name": name, "defer_loading": False,
            "description": description, "parameters": {"type": "object",
                "additionalProperties": False, "properties": {"findall_id": {"type": "string"}},
                "required": ["findall_id"]}})
    return definitions


def installed_profile(api):
    handler = getattr(api, "findall_application_tools", None)
    if handler is None:
        return None
    if not isinstance(handler, FindAllApplicationTools):
        from tools.daily_research.runner import Refusal
        raise Refusal("findall_tool_owner_dependencies_required")
    return PROFILE


def check_binding(row):
    """The optional registry is frozen with the durable session intent."""
    from tools.daily_research.runner import Refusal, digest
    profile = row.get("findall_profile")
    expected = digest(tools())
    if (profile != PROFILE
            or row.get("metadata", {}).get("findall_tools_digest") != expected
            or row.get("preflight", {}).get("findall_tools_digest") != expected):
        raise Refusal("findall_tool_registry_binding_changed")


def instructions():
    return (" Parallel FindAll application tools support asynchronous list discovery. Choose objective, "
            "positive evidence conditions and generator yourself when useful within current authority. "
            "Availability creates no spending or disclosure authority: each create requires fresh exact "
            "admission under the shared remaining allowance. Retain the returned findall_id and poll "
            "status/result as needed; an uncertain start must never be restarted automatically. Preserve "
            "all raw candidates, provider status, basis, reasoning, citations and unknowns for independent "
            "qualification. Provider matches are discovery only, never verified or CRM-ready leads.")


class _MirroringLedger:
    """Keep the caller's pending row from overwriting newly durable claims."""
    def __init__(self, ledger, row):
        self.ledger, self.row = ledger, row

    def __getattr__(self, name):
        return getattr(self.ledger, name)

    def put(self, value):
        if value.get("date") != self.row.get("date"):
            raise FindAllError("findall_tool_owner_row_changed")
        # Mirror before the store attempt. A failure must keep the caller from
        # replacing a possibly committed claim with its older in-memory row.
        for field in (SUBMISSIONS_FIELD, READS_FIELD):
            if field in value:
                self.row[field] = copy.deepcopy(value[field])
        return self.ledger.put(value)


class FindAllApplicationTools:
    """Injected handler; caller must already hold and freshly assert its lease.

    grant_provider consumes the prepared exact request through the existing
    approved admission broker. This class never issues a grant or reads a key.
    current_authority must check current stop/scope/expiry/disclosure/pricing
    and shared remaining allowance, not just a historical approval reference.
    """
    names = NAMES

    def __init__(self, *, ledger, client, grant_provider, current_authority,
                 assert_current_lease):
        if (not isinstance(client, AdmittedFindAllClient)
                or any(not callable(fn) for fn in (
                    grant_provider, current_authority, assert_current_lease))):
            raise FindAllError("findall_tool_owner_dependencies_required")
        self.ledger, self.client = ledger, client
        self.grant_provider, self.current_authority = grant_provider, current_authority
        self.assert_current_lease = assert_current_lease

    def assert_fresh_caller(self, row):
        """Refuse stale journal/call rows before the caller can overwrite them."""
        from tools.daily_research.runner import Refusal, digest
        if self.assert_current_lease() is not True:
            raise Refusal("findall_owner_current_lease_required")
        stored = self.ledger.get(row["date"])
        if (not isinstance(stored, dict) or stored.get("run_key") != row.get("run_key")
                or any(digest(stored.get(field, {})) != digest(row.get(field, {}))
                       for field in ("application_tool_calls", SUBMISSIONS_FIELD, READS_FIELD,
                                     "exa_expansion", "exa_transport_receipts"))):
            raise Refusal("findall_tool_stale_owner_row")

    def execute(self, action, *, row, phase):
        from tools.daily_research.runner import canonical, digest, identifier

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

        def authority(owner, prepared):
            return (self.assert_current_lease() is True
                    and self.current_authority(owner, prepared) is True)

        if name == CREATE:
            if set(args) != {"objective", "entity_type", "generator", "match_limit",
                             "match_conditions", "maximum_cost_usd"}:
                raise FindAllError("findall_tool_arguments_invalid")
            spec = {k: v for k, v in args.items() if k != "maximum_cost_usd"}
            operation_id = row["run_key"] + ":findall:" + cid
            prepared = prepare_submission(spec, operation_id=operation_id,
                                          maximum_cost_usd=args["maximum_cost_usd"])
            # Current action-time authority is checked before requesting a grant
            # and again inside the existing durable owner journal before POST.
            if not authority(self.ledger.get(row["date"]), prepared):
                raise FindAllError("findall_tool_current_authority_required")
            grant = self.grant_provider(copy.deepcopy(prepared))
            run = create_under_owner_lease(
                self.client, spec, ledger=ledger, day=row["date"], operation_id=operation_id,
                maximum_cost_usd=args["maximum_cost_usd"], paid_resource_admission_grant=grant,
                current_authority=authority, assert_current_lease=self.assert_current_lease,
            )
            return {"provider": "parallel_findall", "evidence_scope": "discovery_only",
                    "run": run}

        if set(args) != {"findall_id"}:
            raise FindAllError("findall_tool_arguments_invalid")
        owner = self.ledger.get(row["date"])
        slots = owner.get(SUBMISSIONS_FIELD, {}) if isinstance(owner, dict) else {}
        if not any(entry.get("findall_id") == args["findall_id"] for entry in slots.values()):
            raise FindAllError("findall_tool_run_not_owned")
        read_scope = {"operation": "status" if name == STATUS else "result",
                      "findall_id": args["findall_id"], "call_binding_digest": digest(binding)}
        if not authority(owner, read_scope):
            raise FindAllError("findall_tool_current_authority_required")
        snapshot = (self.client.status(args["findall_id"]) if name == STATUS
                    else self.client.result(args["findall_id"]))
        raw = (canonical(snapshot) + "\n").encode()
        filename = row["date"] + "-tool-findall-read-" + digest(binding) + ".json"
        ledger.write_bytes(filename, raw)
        if ledger.read_bytes(filename) != raw:
            raise FindAllError("findall_tool_snapshot_readback_failed")
        owner.setdefault(READS_FIELD, {})[cid] = {
            **read_scope, "file": filename, "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw), "checked_at": datetime.now(timezone.utc).isoformat(),
        }
        ledger.put(owner)
        return {"provider": "parallel_findall", "evidence_scope": "discovery_only",
                "snapshot": snapshot, "receipt": copy.deepcopy(owner[READS_FIELD][cid])}


def runtime_status(api=None):
    """Local-process metadata only; never prove production binding by inference."""
    adapter = getattr(api, "findall_application_tools", None)
    return {"schema_version": "blueprint.findall-runtime-status.v1",
            "observation_scope": "local_process", "module_imported": True,
            "callable_handler_installed": isinstance(adapter, FindAllApplicationTools),
            "tool_names": sorted(NAMES), "credential_binding_name": "PARALLEL_API_KEY",
            "credential_binding_present": "PARALLEL_API_KEY" in os.environ,
            "credential_value_inspected": False, "production_binding_verified": False,
            "fresh_exact_admission_required": True, "paid_operations_authorized": False}


if __name__ == "__main__":
    print(json.dumps(runtime_status(), sort_keys=True))
