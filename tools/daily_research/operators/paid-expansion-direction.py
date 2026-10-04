"""Owner command for the per-run paid expansion allowance (Exa now; FindAll later).

show: control and audit reads only. set/disable are dry runs unless --apply.
Apply writes only the content-addressed, create-only direction object (set) and
control.paid_expansion through the fenced compare-and-swap bridge op, then reads
both back. The amount is data: $10 to $20 or $30 is one command, with no code
change, redeploy or package. No provider, model, session, CRM or send.
"""
import argparse
import base64
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from tools.daily_research import allocation
from tools.daily_research.firestore import Bridge
from tools.daily_research.runner import Refusal, canonical

SCHEMA = "blueprint.research-paid-expansion-operation.v1"
TERM = timedelta(days=90)  # Default direction lifetime; --expires-at may set up to 366 days.
# A busy worker releases the lease for about 3 s between passes and a lost lease expires
# after 180 s, so a fast poll for 200 s always wins it without displacing the worker.
LEASE_WAIT_SECONDS, LEASE_POLL_SECONDS = 200.0, 0.25


def object_sha(uri):
    sha = uri.removeprefix(f"gs://{allocation.BUCKET}/{allocation.OBJECT_PREFIX}").removesuffix("/direction.json")
    if not allocation.SHA.fullmatch(sha) or allocation.uri(sha) != uri:
        raise Refusal("paid_expansion_object_uri_invalid")
    return sha


class BridgeObjects:
    """Default object adapter: the existing worker identity through the bridge, create-only."""

    def __init__(self, bridge):
        self.bridge = bridge

    def create(self, uri, raw):
        self.bridge.call("paid_expansion_object_put", sha256=object_sha(uri), bytes=base64.b64encode(raw).decode("ascii"))

    def read(self, uri):
        return base64.b64decode(self.bridge.call("paid_expansion_object_get", sha256=object_sha(uri))["bytes"], validate=True)


@contextmanager
def lease(bridge, sleep=time.sleep, monotonic=time.monotonic, wait=LEASE_WAIT_SECONDS):
    """The existing fenced lease, held only for the swap; a worker holding it is waited for."""
    deadline = monotonic() + wait
    while True:
        try:
            bridge.call("acquire")
            break
        except Refusal as exc:
            if str(exc) != "runner_overlap" or monotonic() >= deadline:
                raise
            sleep(LEASE_POLL_SECONDS)
    try:
        yield
    finally:
        bridge.call("release")


def utc(value):
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def amount(value):
    """Owner input such as 20 or 20.00, normalized to two decimals within the $1–$100 code ceiling."""
    text = value + ".00" if isinstance(value, str) and value.isdigit() else value
    if allocation.micros(text) is None:
        raise Refusal("paid_expansion_limit_invalid")
    return text


def described(entry, enabled):
    direction = entry["direction"]
    limit = allocation.micros(direction["per_run_limit_usd"])
    per_start = allocation.per_start_max(limit)
    return {"state": "enabled" if enabled else "disabled", "version": entry["version"], "sha256": entry["sha256"],
            "uri": entry["uri"], "per_run_limit_usd": direction["per_run_limit_usd"], "limit_micros": limit,
            "per_start_max_micros": per_start,
            **{key: direction[key] for key in ("sources", "effective_from", "expires_at", "approval_reference",
                                               "approved_by", "issued_at", "reason", "supersedes")}}


def chain_problem(audit, entry):
    """None when audit records 1..n are verified directions, each superseding the last, ending at current."""
    if not isinstance(audit, list):
        return "paid_expansion_audit_chain_invalid"
    previous = None
    for version, record in enumerate(audit, 1):
        bound = {key: record.get(key) for key in ("sha256", "version", "uri", "direction")} if isinstance(record, dict) else None
        if (allocation.entry_problem(bound) or bound["version"] != version
                or bound["direction"]["supersedes"] != previous):
            return "paid_expansion_audit_chain_invalid"
        previous = bound["sha256"]
    if previous != (entry or {}).get("sha256"):
        return "paid_expansion_audit_chain_invalid"
    return None


def show(bridge, objects=None):
    """Current amount, version and the verified audit chain; reads only."""
    control = bridge.call("control") or {}
    paid = control.get("paid_expansion")
    entry = paid.get("current") if isinstance(paid, dict) else None
    audit = bridge.call("paid_expansion_audit")
    problem = chain_problem(audit, entry)
    result = {"schema_version": SCHEMA, "command": "show", "state": "unset",
              "audit_chain_verified": problem is None, "audit_chain_problem": problem,
              "audit_chain": [{key: record.get(key) for key in ("version", "sha256", "recorded_at")}
                              | {key: (record.get("direction") or {}).get(key) for key in (
                                  "per_run_limit_usd", "supersedes", "issued_at", "expires_at", "approval_reference",
                                  "approved_by", "reason")} for record in audit if isinstance(record, dict)],
              "source_commit": control.get("source_commit"), "firestore_writes": 0, "object_writes": 0,
              "provider_calls": 0}
    if entry is not None:
        code = allocation.entry_problem(entry)
        if code:
            return {**result, "state": "unverified", "current_problem": code}
        result.update(described(entry, paid.get("enabled") is True))
        if objects is not None:
            try:
                result["object_verified"] = objects.read(entry["uri"]) == canonical(entry["direction"]).encode()
            except (Refusal, OSError, KeyError, ValueError):
                result["object_verified"] = False
    return result


def parent_of(control, audit):
    """The record a new direction supersedes: current control, else the audit head (a
    rollback can drop control's copy). Only its address and version must be usable, so an
    owner can replace a direction this package cannot verify."""
    paid = control.get("paid_expansion") if isinstance(control, dict) else None
    prior = paid.get("current") if isinstance(paid, dict) else None
    if prior is None and audit:
        prior = max((record for record in audit if isinstance(record, dict)),
                    key=lambda record: record.get("version") if type(record.get("version")) is int else 0, default=None)
    if prior is not None and (not isinstance(prior, dict) or not isinstance(prior.get("sha256"), str)
                              or not allocation.SHA.fullmatch(prior["sha256"])
                              or type(prior.get("version")) is not int or prior["version"] < 1):
        raise Refusal("paid_expansion_current_unverified")
    return prior


def next_entry(control, *, per_run_usd, approval_reference, approved_by, reason, now, expires_at=None, expect=None,
               audit=None):
    """The next owner direction: version+1, superseding the current one (or the audit head)."""
    paid = control.get("paid_expansion") if isinstance(control, dict) else None
    current = paid.get("current") if isinstance(paid, dict) else None
    observed = current.get("sha256") if isinstance(current, dict) else None
    if expect is not None and expect != (observed or "none"):
        raise Refusal("paid_expansion_direction_conflict")
    prior = parent_of(control, audit)
    issued = utc(now)
    direction = {"schema_version": allocation.DIRECTION, "version": prior["version"] + 1 if prior else 1,
                 "supersedes": prior["sha256"] if prior else None, "per_run_limit_usd": amount(per_run_usd),
                 "sources": list(allocation.SOURCES),
                 "scope": dict(allocation.SCOPE), "effective_from": issued,
                 "expires_at": utc(expires_at) if expires_at else utc(now + TERM),
                 "approval_reference": approval_reference, "approved_by": approved_by, "issued_at": issued, "reason": reason}
    code = allocation.direction_problem(direction)
    if code:
        raise Refusal(code)
    sha = allocation.digest(direction)
    return observed, {"sha256": sha, "version": direction["version"], "uri": allocation.uri(sha), "direction": direction}


def set_direction(bridge, objects, *, apply=False, during_active_run=False, sleep=time.sleep,
                  monotonic=time.monotonic, now=None, **direction):
    control = bridge.call("control")
    observed, entry = next_entry(control, now=now or datetime.now(timezone.utc),
                                 audit=bridge.call("paid_expansion_audit"), **direction)
    current = ((control or {}).get("paid_expansion") or {}).get("current")
    result = {"schema_version": SCHEMA, "command": "set", "expected_sha256": observed, "next": described(entry, True),
              "direction": entry["direction"], "apply": apply, "provider_calls": 0,
              "current_problem": allocation.entry_problem(current) if current is not None else None}
    if not apply:
        return {**result, "state": "planned", "firestore_writes": 0, "object_writes": 0}
    if not during_active_run and (bridge.call("summary").get("unfinished") or bridge.call("active_qa")):
        # The new amount applies from the next run anyway; avoid contending for an active worker's lease.
        raise Refusal("paid_expansion_run_active_apply_after_run")
    raw = canonical(entry["direction"]).encode()
    objects.create(entry["uri"], raw)
    if objects.read(entry["uri"]) != raw:
        raise Refusal("paid_expansion_object_readback_failed")
    with lease(bridge, sleep, monotonic):
        receipt = bridge.call("paid_expansion_set", expected_sha256=observed, value={"enabled": True, "current": entry})
    if (bridge.call("control").get("paid_expansion") != {"enabled": True, "current": entry}
            or objects.read(entry["uri"]) != raw):
        raise Refusal("paid_expansion_readback_failed")
    return {**result, "state": "applied", "receipt": receipt, "readback_verified": True,
            "object": {"uri": entry["uri"], "sha256": entry["sha256"], "bytes": len(raw)},
            "firestore_writes": 2, "object_writes": 1}


def disable(bridge, *, apply=False, sleep=time.sleep, monotonic=time.monotonic):
    """Emergency brake: stops new paid starts now, including in an active run; no object-storage dependency."""
    paid = (bridge.call("control") or {}).get("paid_expansion")
    result = {"schema_version": SCHEMA, "command": "disable", "apply": apply, "provider_calls": 0, "object_writes": 0}
    if not isinstance(paid, dict) or paid.get("enabled") is not True:
        return {**result, "state": "already_disabled", "firestore_writes": 0}
    entry = paid.get("current")
    result.update(expected_sha256=(entry or {}).get("sha256"), version=(entry or {}).get("version"))
    if not apply:
        return {**result, "state": "planned", "firestore_writes": 0}
    with lease(bridge, sleep, monotonic):
        receipt = bridge.call("paid_expansion_set", expected_sha256=result["expected_sha256"],
                              value={"enabled": False, "current": entry})
    if bridge.call("control").get("paid_expansion") != {"enabled": False, "current": entry}:
        raise Refusal("paid_expansion_readback_failed")
    return {**result, "state": "disabled", "receipt": receipt, "readback_verified": True, "firestore_writes": 1}


def main(argv=None, *, bridge_factory=Bridge, objects_factory=BridgeObjects, clock=lambda: datetime.now(timezone.utc),
         sleep=time.sleep, monotonic=time.monotonic):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("show")
    setter = commands.add_parser("set")
    setter.add_argument("--per-run-usd", required=True, help="Combined per-run limit, $1.00-$100.00, e.g. 20.00")
    setter.add_argument("--approval-reference", required=True)
    setter.add_argument("--approved-by", default="owner")
    setter.add_argument("--reason", default="Owner per-run paid expansion allowance")
    setter.add_argument("--expires-at", type=datetime.fromisoformat, help="UTC ISO time; default 90 days")
    setter.add_argument("--expect-current", help="Current direction sha256 from show, or none")
    setter.add_argument("--during-active-run", action="store_true")
    setter.add_argument("--apply", action="store_true")
    brake = commands.add_parser("disable")
    brake.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    bridge = bridge_factory()
    try:
        if args.command == "show":
            result = show(bridge, objects_factory(bridge))
        elif args.command == "set":
            if args.expires_at is not None and args.expires_at.tzinfo is None:
                raise Refusal("paid_expansion_time_invalid")
            result = set_direction(bridge, objects_factory(bridge), apply=args.apply, during_active_run=args.during_active_run,
                                   sleep=sleep, monotonic=monotonic, now=clock(), per_run_usd=args.per_run_usd,
                                   approval_reference=args.approval_reference, approved_by=args.approved_by,
                                   reason=args.reason, expires_at=args.expires_at, expect=args.expect_current)
        else:
            result = disable(bridge, apply=args.apply, sleep=sleep, monotonic=monotonic)
        print(canonical(result))
        return result
    finally:
        bridge.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - stable codes; never expose provider/account exceptions
        print(canonical({"state": "blocked", "error": str(error) if isinstance(error, Refusal) else "paid_expansion_operation_unavailable"}))
        raise SystemExit(1) from None
