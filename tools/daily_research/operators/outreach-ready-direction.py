"""Owner command for the outreach-ready hypothesis direction (ADP-010 partner discovery).

show only reads. set and disable are dry runs unless --apply. set --apply writes only the
content-addressed, create-only direction object and control.outreach_ready, pinned by object
generation and SHA-256, through the fenced compare-and-swap bridge op, then reads both back.
disable is the brake: it keeps the pin with enabled=false. Without an enabled direction every
run is in shadow mode. A direction lets daily QA label outreach-ready hypotheses for drafting
only; it never authorizes a send. No provider, model, session, CRM write or send.
"""
import argparse
import base64
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from tools.daily_research import outreach_ready, verification
from tools.daily_research.firestore import Bridge
from tools.daily_research.runner import Refusal, canonical

SCHEMA = "blueprint.outreach-ready-operation.v1"
TERM = timedelta(days=30)  # Default direction lifetime; --expires-at may set up to 366 days.
# A busy worker releases the lease for about 3 s between passes and a lost lease expires
# after 180 s, so a fast poll for 200 s always wins it without displacing the worker.
LEASE_WAIT_SECONDS, LEASE_POLL_SECONDS = 200.0, 0.25


class BridgeObjects:
    """Default object adapter: the existing worker identity through the bridge, create-only and generation-pinned."""

    def __init__(self, bridge):
        self.bridge = bridge

    def create(self, sha256, raw):
        return self.bridge.call("outreach_ready_object_put", sha256=sha256,
                                bytes=base64.b64encode(raw).decode("ascii"))["generation"]

    def read(self, sha256, generation):
        value = self.bridge.call("outreach_ready_object_get", sha256=sha256, generation=generation)
        return base64.b64decode(value["bytes"], validate=True)


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


def path_list(value):
    """Owner input such as "daily_qa" or "daily_qa,site_screen": known, nonempty and unrepeated, in canonical order."""
    names = [part.strip() for part in value.split(",")] if isinstance(value, str) else []
    if not names or any(name not in outreach_ready.PATHS for name in names) or len(set(names)) != len(names):
        raise Refusal("outreach_ready_paths_invalid")
    return sorted(names)


def row_limit(value):
    if type(value) is not int or not 1 <= value <= outreach_ready.MAX_ROWS_PER_BATCH:
        raise Refusal("outreach_ready_rows_invalid")
    return value


def active_run(bridge):
    return bool(bridge.call("summary").get("unfinished") or bridge.call("active_qa"))


def described(entry, enabled):
    direction = entry["direction"]
    return {"state": "enabled" if enabled else "disabled", "version": entry["version"], "sha256": entry["sha256"],
            "generation": entry.get("generation"), "uri": entry["uri"], **direction["scope"],
            **{key: direction[key] for key in ("rule_version", "effective_from", "expires_at", "approval_reference",
                                               "approved_by", "issued_at", "reason", "supersedes")}}


def next_run(value, now):
    """What the next run would freeze from this control value: shadow (a screen-only direction names its
    code but freezes nothing), refused with a code, or enabled."""
    frozen = outreach_ready.assess(value, {"run_key": outreach_ready.BINDING["run_key_prefix"] + "next"}, now)
    if frozen is None:
        return {"state": "shadow"}
    if frozen.get("code") == outreach_ready.NOT_DIRECTED:
        return {"state": "shadow", "code": frozen["code"]}
    return {"state": frozen["state"], "code": frozen.get("code"), "paths": frozen.get("paths"),
            "max_rows_per_batch": frozen.get("max_rows_per_batch")}


def show(bridge, objects=None, *, now):
    """Current pin, its verified object and what the next run would freeze; reads only."""
    control = bridge.call("control") or {}
    pin = control.get("outreach_ready")
    entry = pin.get("current") if isinstance(pin, dict) else None
    result = {"schema_version": SCHEMA, "command": "show", "state": "unset", "firestore_writes": 0, "object_writes": 0,
              "provider_calls": 0, "sends_authorized": False, "next_run": next_run(pin, now)}
    if pin is None:
        return result
    code = outreach_ready.entry_problem(entry)
    if code:
        return {**result, "state": "unverified", "current_problem": code}
    result.update(described(entry, pin.get("enabled") is True))
    if objects is not None:
        try:
            result["object_verified"] = objects.read(entry["sha256"], entry["generation"]) == canonical(entry["direction"]).encode()
        except (Refusal, OSError, KeyError, ValueError):
            result["object_verified"] = False
    return result


def next_entry(control, *, paths, max_rows_per_batch, approval_reference, approved_by, reason, now, expires_at=None,
               expect=None):
    """The next owner direction: version+1, superseding control's current one."""
    pin = control.get("outreach_ready") if isinstance(control, dict) else None
    current = pin.get("current") if isinstance(pin, dict) else None
    observed = current.get("sha256") if isinstance(current, dict) else None
    if expect is not None and expect != (observed or "none"):
        raise Refusal("outreach_ready_direction_conflict")
    if current is not None and (not isinstance(current, dict) or not isinstance(observed, str)
                                or not outreach_ready.SHA.fullmatch(observed)
                                or type(current.get("version")) is not int or current["version"] < 1):
        raise Refusal("outreach_ready_current_unverified")
    issued = utc(now)
    direction = {"schema_version": outreach_ready.DIRECTION, "version": current["version"] + 1 if current else 1,
                 "supersedes": observed, "rule_version": verification.OUTREACH_RULE_VERSION,
                 "scope": {"paths": path_list(paths), "label": outreach_ready.LABEL,
                           "max_rows_per_batch": row_limit(max_rows_per_batch), "sends_authorized": False},
                 "binding": dict(outreach_ready.BINDING), "effective_from": issued,
                 "expires_at": utc(expires_at) if expires_at else utc(now + TERM),
                 "approval_reference": approval_reference, "approved_by": approved_by, "issued_at": issued, "reason": reason}
    code = outreach_ready.direction_problem(direction)
    if code:
        raise Refusal(code)
    sha = outreach_ready.digest(direction)
    return observed, {"sha256": sha, "version": direction["version"], "uri": outreach_ready.uri(sha), "direction": direction}


def set_direction(bridge, objects, *, apply=False, during_active_run=False, sleep=time.sleep,
                  monotonic=time.monotonic, now=None, **direction):
    now = now or datetime.now(timezone.utc)
    control = bridge.call("control")
    observed, entry = next_entry(control, now=now, **direction)
    current = ((control or {}).get("outreach_ready") or {}).get("current")
    result = {"schema_version": SCHEMA, "command": "set", "expected_sha256": observed,
              "next": described({**entry, "generation": None}, True), "direction": entry["direction"], "apply": apply,
              "provider_calls": 0, "sends_authorized": False,
              "current_problem": outreach_ready.entry_problem(current) if current is not None else None}
    if not apply:
        return {**result, "state": "planned", "firestore_writes": 0, "object_writes": 0}
    if not during_active_run and active_run(bridge):
        # A running row keeps what it froze; avoid contending for an active worker's lease.
        raise Refusal("outreach_ready_run_active_apply_after_run")
    raw = canonical(entry["direction"]).encode()
    generation = objects.create(entry["sha256"], raw)
    if objects.read(entry["sha256"], generation) != raw:
        raise Refusal("outreach_ready_object_readback_failed")
    pinned = {**entry, "generation": generation}
    with lease(bridge, sleep, monotonic):
        receipt = bridge.call("outreach_ready_set", expected_sha256=observed, value={"enabled": True, "current": pinned})
    if ((bridge.call("control") or {}).get("outreach_ready") != {"enabled": True, "current": pinned}
            or objects.read(entry["sha256"], generation) != raw):
        raise Refusal("outreach_ready_readback_failed")
    return {**result, "state": "applied", "next": described(pinned, True), "receipt": receipt, "readback_verified": True,
            "object": {"uri": entry["uri"], "sha256": entry["sha256"], "generation": generation, "bytes": len(raw)},
            "firestore_writes": 1, "object_writes": 1}


def disable(bridge, *, apply=False, sleep=time.sleep, monotonic=time.monotonic):
    """The brake: stops new admissions at once, including in an active run; no object-storage dependency."""
    pin = (bridge.call("control") or {}).get("outreach_ready")
    result = {"schema_version": SCHEMA, "command": "disable", "apply": apply, "provider_calls": 0, "object_writes": 0,
              "sends_authorized": False}
    if not isinstance(pin, dict) or pin.get("enabled") is not True:
        return {**result, "state": "already_disabled" if pin is not None else "unset", "firestore_writes": 0}
    entry = pin.get("current")
    result.update(expected_sha256=(entry or {}).get("sha256"), version=(entry or {}).get("version"))
    if not apply:
        return {**result, "state": "planned", "firestore_writes": 0}
    with lease(bridge, sleep, monotonic):
        receipt = bridge.call("outreach_ready_set", expected_sha256=result["expected_sha256"],
                              value={"enabled": False, "current": entry})
    if (bridge.call("control") or {}).get("outreach_ready") != {"enabled": False, "current": entry}:
        raise Refusal("outreach_ready_readback_failed")
    return {**result, "state": "disabled", "receipt": receipt, "readback_verified": True, "firestore_writes": 1}


def main(argv=None, *, bridge_factory=Bridge, objects_factory=BridgeObjects, clock=lambda: datetime.now(timezone.utc),
         sleep=time.sleep, monotonic=time.monotonic):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("show")
    setter = commands.add_parser("set")
    setter.add_argument("--approval-reference", required=True, help="The owner decision record")
    setter.add_argument("--paths", default="daily_qa", help="Comma-separated paths: " + ", ".join(outreach_ready.PATHS))
    setter.add_argument("--max-rows-per-batch", type=int, default=outreach_ready.MAX_ROWS_PER_BATCH,
                        help=f"1-{outreach_ready.MAX_ROWS_PER_BATCH} hypotheses per batch (default {outreach_ready.MAX_ROWS_PER_BATCH})")
    setter.add_argument("--approved-by", default="owner")
    setter.add_argument("--reason", default="Owner outreach-ready hypothesis direction")
    setter.add_argument("--expires-at", type=datetime.fromisoformat, help="UTC ISO time; default 30 days")
    setter.add_argument("--expect-current", help="Current direction sha256 from show, or none")
    setter.add_argument("--during-active-run", action="store_true")
    setter.add_argument("--apply", action="store_true")
    brake = commands.add_parser("disable")
    brake.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    bridge = bridge_factory()
    try:
        if args.command == "show":
            result = show(bridge, objects_factory(bridge), now=clock())
        elif args.command == "set":
            if args.expires_at is not None and args.expires_at.tzinfo is None:
                raise Refusal("outreach_ready_time_invalid")
            result = set_direction(bridge, objects_factory(bridge), apply=args.apply, during_active_run=args.during_active_run,
                                   sleep=sleep, monotonic=monotonic, now=clock(), paths=args.paths,
                                   max_rows_per_batch=args.max_rows_per_batch, approval_reference=args.approval_reference,
                                   approved_by=args.approved_by, reason=args.reason, expires_at=args.expires_at,
                                   expect=args.expect_current)
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
        print(canonical({"state": "blocked", "error": str(error) if isinstance(error, Refusal) else "outreach_ready_operation_unavailable"}))
        raise SystemExit(1) from None
