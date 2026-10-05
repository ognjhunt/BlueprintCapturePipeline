"""Owner command for the site universe backlog slice of the daily research run.

show and funnel only read. publish, pin and disable are dry runs unless --apply.
publish writes only the content-addressed, create-only export object; pin and
disable write only control.site_universe through the fenced compare-and-swap bridge
op, then read it back. pin first runs the runtime loader and a dry-run selection.
Output shows counts and ids, never site names. No provider, model, session, CRM
write or send.
"""
import argparse
import base64
import hashlib
import json
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from tools.daily_research import site_universe
from tools.daily_research.firestore import Bridge, FirestoreLedger
from tools.daily_research.runner import CENTRAL, Refusal, canonical, due_date

SCHEMA = "blueprint.site-universe-operation.v1"
# A busy worker releases the lease for about 3 s between passes and a lost lease expires
# after 180 s, so a fast poll for 200 s always wins it without displacing the worker.
LEASE_WAIT_SECONDS, LEASE_POLL_SECONDS = 200.0, 0.25


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


def research_seconds(control):
    """The research window a new row would pin from current control."""
    config = control.get("config") if isinstance(control, dict) else None
    config = config if isinstance(config, dict) else {}
    total = config.get("max_runtime_seconds", 180)
    reserved = config.get("qa_reserved_seconds", 0) if config.get("discovery_profile") == "adaptive-sites-v1" else 0
    return total - reserved if type(total) is int and type(reserved) is int else None


def next_run_date(bridge, now):
    """The date the next create would use: today's due date, or the next one once it has a row."""
    day = date.fromisoformat(due_date(now, "2000-01-01"))
    return (day + timedelta(days=1) if bridge.call("get", day=day.isoformat()) else day).isoformat()


def crm_values(bridge):
    """The canonical CRM snapshot rows the run would pre-filter by, or None when unreadable."""
    try:
        snapshot = json.loads(FirestoreLedger(bridge).read_bytes("crm.json"))
    except (FileNotFoundError, ValueError):
        return None
    values = snapshot.get("values") if isinstance(snapshot, dict) else None
    return values if isinstance(values, list) else None


def read_object(bridge, sha256, generation, size=None):
    value = bridge.call("site_universe_object_get", sha256=sha256, generation=generation, size=size)
    return base64.b64decode(value["data"], validate=True)


def pin_ids(value):
    return {key: value.get(key) for key in ("enabled", "uri", "generation", "sha256", "bytes", "snapshot_id",
                                            "rank_config_sha256", "slice_size", "reoffer_after_days",
                                            "approval_reference")} if isinstance(value, dict) else None


def export_ids(export):
    manifest = export["manifest"]
    return {"sha256": export["sha256"], "bytes": export["bytes"], "snapshot_id": manifest["snapshot_id"],
            "rows_sha256": manifest["rows_sha256"], "rank_config_sha256": manifest["rank_config_sha256"],
            "rank_config_version": manifest["rank_config_version"], "counts": manifest["counts"],
            "lead_capability_counts": manifest["lead_capability_counts"],
            "seed_capabilities": manifest["selection_policy"]["seed_capabilities"],
            "previous_snapshot_id": manifest["previous_snapshot_id"], "new_sites": manifest["new_sites"],
            "distribution": manifest["distribution"], "approval_reference": manifest["approval_reference"]}


def dry_run(bridge, export, pin, now):
    """The selection the next run would make, from current history and the canonical CRM; ids only."""
    values = crm_values(bridge)
    day = next_run_date(bridge, now)
    sites, selection = site_universe.select(export, history=bridge.call("rows"), crm_values=values or [], run_date=day,
                                            slice_size=pin["slice_size"], reoffer_after_days=pin["reoffer_after_days"])
    return {"run_date": day, "crm_snapshot_included": values is not None, "selection": selection,
            "site_ids": [site["site_id"] for site in sites]}


def show(bridge, *, now):
    """Current pin, its verified export and the next run's dry-run selection; reads only."""
    control = bridge.call("control") or {}
    value = control.get("site_universe")
    result = {"schema_version": SCHEMA, "command": "show", "firestore_writes": 0, "object_writes": 0,
              "provider_calls": 0, "pin": pin_ids(value), "research_seconds": research_seconds(control), "ready": False}
    if value is None:
        return {**result, "state": "unset"}
    if isinstance(value, dict) and value.get("enabled") is False:
        return {**result, "state": "disabled"}
    try:
        found = site_universe.pin(value, result["research_seconds"])
        export = site_universe.load_export(read_object(bridge, found["sha256"], found["generation"], found["bytes"]), found)
        check = dry_run(bridge, export, found, now)
    except (site_universe.SiteUniverseError, Refusal) as exc:
        if isinstance(exc, Refusal) and not site_universe.CODE.fullmatch(str(exc)):
            raise
        return {**result, "state": "enabled_unusable", "code": str(exc)}
    return {**result, "state": "enabled", "export": export_ids(export), "dry_run": check, "ready": bool(check["site_ids"])}


def publish(bridge, path, *, apply=False):
    """Validate a local export with the runtime loader, then write it create-only and read it back."""
    with open(path, "rb") as handle:
        raw = handle.read(site_universe.MAX_OBJECT_BYTES + 1)
    export = site_universe.load_export(raw)
    sha = hashlib.sha256(raw).hexdigest()
    result = {"schema_version": SCHEMA, "command": "publish", "apply": apply, "provider_calls": 0, "firestore_writes": 0,
              "uri": site_universe.object_uri(sha), "export": export_ids(export)}
    if not apply:
        return {**result, "state": "planned", "object_writes": 0}
    stored = bridge.call("site_universe_object_put", sha256=sha, bytes=base64.b64encode(raw).decode("ascii"))
    if read_object(bridge, sha, stored["generation"], len(raw)) != raw:
        raise Refusal("site_universe_object_readback_failed")
    return {**result, "state": "published", "generation": stored["generation"], "readback_verified": True,
            "object_writes": 1}


def pin(bridge, *, sha256, generation, approval_reference, slice_size=site_universe.DEFAULT_SLICE,
        reoffer_after_days=site_universe.DEFAULT_REOFFER_DAYS, expect=None, apply=False, during_active_run=False,
        now=None, sleep=time.sleep, monotonic=time.monotonic):
    """Pin one published export after the runtime loader and a dry-run selection accept it."""
    now = now or datetime.now(timezone.utc)
    control = bridge.call("control") or {}
    current = control.get("site_universe")
    observed = current.get("sha256") if isinstance(current, dict) else None
    if expect is not None and expect != (observed or "none"):
        raise Refusal("site_universe_control_conflict")
    if not isinstance(sha256, str) or not site_universe.SHA.fullmatch(sha256):
        raise Refusal("site_universe_pin_invalid")
    raw = read_object(bridge, sha256, generation)
    export = site_universe.load_export(raw)
    value = {"enabled": True, "uri": site_universe.object_uri(sha256), "generation": generation, "sha256": sha256,
             "bytes": len(raw), "snapshot_id": export["manifest"]["snapshot_id"],
             "rank_config_sha256": export["manifest"]["rank_config_sha256"], "slice_size": slice_size,
             "reoffer_after_days": reoffer_after_days, "approval_reference": approval_reference}
    found = site_universe.pin(value, research_seconds(control))
    site_universe.load_export(raw, found)
    check = dry_run(bridge, export, found, now)
    if not check["site_ids"]:
        raise Refusal("site_universe_slice_empty")
    result = {"schema_version": SCHEMA, "command": "pin", "apply": apply, "provider_calls": 0, "object_writes": 0,
              "expected_sha256": observed, "pin": value, "export": export_ids(export), "dry_run": check}
    if not apply:
        return {**result, "state": "planned", "firestore_writes": 0}
    if not during_active_run and (bridge.call("summary").get("unfinished") or bridge.call("active_qa")):
        # A pin applies from the next create anyway; avoid contending for an active worker's lease.
        raise Refusal("site_universe_run_active_apply_after_run")
    with lease(bridge, sleep, monotonic):
        receipt = bridge.call("site_universe_set", expected_sha256=observed, value=value)
    if (bridge.call("control") or {}).get("site_universe") != value:
        raise Refusal("site_universe_readback_failed")
    return {**result, "state": "pinned", "receipt": receipt, "readback_verified": True, "firestore_writes": 1}


def disable(bridge, *, apply=False, sleep=time.sleep, monotonic=time.monotonic):
    """Turn the slice off from the next create; the current pin is kept with enabled=false."""
    value = (bridge.call("control") or {}).get("site_universe")
    result = {"schema_version": SCHEMA, "command": "disable", "apply": apply, "provider_calls": 0, "object_writes": 0}
    if not isinstance(value, dict) or value.get("enabled") is False:
        return {**result, "state": "already_disabled" if value is not None else "unset", "firestore_writes": 0}
    result.update(expected_sha256=value.get("sha256"), generation=value.get("generation"))
    if not apply:
        return {**result, "state": "planned", "firestore_writes": 0}
    disabled = {**value, "enabled": False}
    with lease(bridge, sleep, monotonic):
        receipt = bridge.call("site_universe_set", expected_sha256=value.get("sha256"), value=disabled)
    if (bridge.call("control") or {}).get("site_universe") != disabled:
        raise Refusal("site_universe_readback_failed")
    return {**result, "state": "disabled", "receipt": receipt, "readback_verified": True, "firestore_writes": 1}


def _add(total, value):
    for key, item in value.items():
        if isinstance(item, dict):
            _add(total.setdefault(key, {}), item)
        elif type(item) is int:
            total[key] = total.get(key, 0) + item


def funnel(bridge, *, days=7, now=None):
    """Per-run site universe states and funnels for the last ``days`` run dates, with totals; counts only."""
    if type(days) is not int or not 1 <= days <= 366:
        raise Refusal("site_universe_funnel_days_invalid")
    today = (now or datetime.now(timezone.utc)).astimezone(CENTRAL).date()
    start = today - timedelta(days=days - 1)
    runs, totals = [], {}
    for row in bridge.call("rows"):
        if not start <= date.fromisoformat(row["date"]) <= today:
            continue
        state = site_universe.status(row)
        runs.append({"date": row["date"], "run_state": row.get("state"), "site_universe": state or {"state": "off"}})
        if state and isinstance(state.get("funnel"), dict):
            _add(totals, {key: state["funnel"][key] for key in ("selection", "agent", "qa", "run")
                          if isinstance(state["funnel"].get(key), dict)})
    return {"schema_version": SCHEMA, "command": "funnel", "from": start.isoformat(), "to": today.isoformat(),
            "runs": runs, "attached_runs": sum(run["site_universe"].get("state") == "attached" for run in runs),
            "totals": totals, "not_measured": {"contacted": None, "replied": None, "conversations": None,
                                                "per_site_cost": "not_measured"},
            "firestore_writes": 0, "object_writes": 0, "provider_calls": 0}


def main(argv=None, *, bridge_factory=Bridge, clock=lambda: datetime.now(timezone.utc), sleep=time.sleep,
         monotonic=time.monotonic):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("show")
    publisher = commands.add_parser("publish")
    publisher.add_argument("--file", required=True, type=Path, help="Local backlog.v1.json.gz from the export command")
    publisher.add_argument("--apply", action="store_true")
    pinner = commands.add_parser("pin")
    pinner.add_argument("--sha256", required=True, help="Published export SHA-256 (publish prints it)")
    pinner.add_argument("--generation", required=True, help="Published object generation (publish prints it)")
    pinner.add_argument("--approval-reference", required=True)
    pinner.add_argument("--slice-size", type=int, default=site_universe.DEFAULT_SLICE,
                        help=f"{site_universe.MIN_SLICE}-{site_universe.MAX_SLICE}, at most research seconds // 90")
    pinner.add_argument("--reoffer-after-days", type=int, default=site_universe.DEFAULT_REOFFER_DAYS)
    pinner.add_argument("--expect-current", help="Current pin sha256 from show, or none")
    pinner.add_argument("--during-active-run", action="store_true")
    pinner.add_argument("--apply", action="store_true")
    brake = commands.add_parser("disable")
    brake.add_argument("--apply", action="store_true")
    report = commands.add_parser("funnel")
    report.add_argument("--days", type=int, default=7)
    args = parser.parse_args(argv)
    bridge = bridge_factory()
    try:
        if args.command == "show":
            result = show(bridge, now=clock())
        elif args.command == "publish":
            result = publish(bridge, args.file, apply=args.apply)
        elif args.command == "pin":
            result = pin(bridge, sha256=args.sha256, generation=args.generation, approval_reference=args.approval_reference,
                         slice_size=args.slice_size, reoffer_after_days=args.reoffer_after_days, expect=args.expect_current,
                         apply=args.apply, during_active_run=args.during_active_run, now=clock(), sleep=sleep,
                         monotonic=monotonic)
        elif args.command == "disable":
            result = disable(bridge, apply=args.apply, sleep=sleep, monotonic=monotonic)
        else:
            result = funnel(bridge, days=args.days, now=clock())
        print(canonical(result))
        return result
    finally:
        bridge.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - stable codes; never expose provider/account exceptions
        code = str(error) if isinstance(error, (Refusal, site_universe.SiteUniverseError)) else "site_universe_operation_unavailable"
        print(canonical({"state": "blocked", "error": code}))
        raise SystemExit(1) from None
