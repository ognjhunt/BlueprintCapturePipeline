"""Counts-only private team-evidence publication/pin. Writes require --apply; no provider/CRM/send.

Publish validates and writes the content-addressed private object create-only. Pin validates that
exact generation, checks research/QA/repair/publication idle before and under the existing fenced
lease, then CAS-swaps only control.team_universe. Old run evidence never changes.
"""
import argparse
import base64
from datetime import datetime, timezone
from pathlib import Path

from tools.daily_research import team_universe as tu
from tools.daily_research.firestore import Bridge
from tools.daily_research.runner import Refusal, canonical
from tools.daily_research.screen_admission import idle, lease

SCHEMA = "blueprint.team-evidence-operation.v1"


def ids(value):
    m = value["manifest"]
    return {key: m[key] for key in ("ranked_sha256", "audit_sha256", "scope_sha256", "assessed_on", "counts")}


def read(bridge, digest, generation):
    found = bridge.call("team_universe_object_get", sha256=digest, generation=generation)
    return base64.b64decode(found["data"], validate=True)


def publish(bridge, path, *, apply=False, today=None):
    with open(path, "rb") as handle:
        raw = handle.read(tu.MAX_BYTES + 1)
    value = tu.load(raw, today=today)
    digest = tu.sha(raw)
    result = {"schema_version": SCHEMA, "command": "publish", "apply": apply, "uri": tu.object_uri(digest),
              "sha256": digest, "bytes": len(raw), "export": ids(value), "provider_calls": 0, "firestore_writes": 0}
    if not apply:
        return {**result, "state": "planned", "object_writes": 0}
    stored = bridge.call("team_universe_object_put", sha256=digest, bytes=base64.b64encode(raw).decode())
    if read(bridge, digest, stored["generation"]) != raw:
        raise Refusal("team_universe_object_readback_failed")
    return {**result, "state": "published", "generation": stored["generation"], "readback_verified": True, "object_writes": 1}


def pin(bridge, *, sha256, generation, approval_reference, expect=None, apply=False, today=None):
    today = today or datetime.now(timezone.utc).date()
    control = bridge.call("control") or {}
    current = control.get("team_universe")
    observed = current.get("sha256") if isinstance(current, dict) else None
    if expect is not None and expect != (observed or "none"):
        raise Refusal("team_universe_pin_conflict")
    raw = read(bridge, sha256, generation)
    value = tu.load(raw, today=today)
    found = {"schema_version": tu.PIN, "enabled": True, "version": (current or {}).get("version", 0) + 1,
             "sha256": sha256, "uri": tu.object_uri(sha256), "generation": generation, "bytes": len(raw),
             **{key: value["manifest"][key] for key in ("ranked_sha256", "audit_sha256", "scope_sha256", "assessed_on")},
             "approval_reference": approval_reference}
    tu.load(raw, expected=tu.pin(found), today=today)
    # Exercise the exact bounded runtime freeze, not merely the object schema.
    class Snapshot:
        def team_universe_snapshot(self):
            return {"pin": found, "data": base64.b64encode(raw).decode()}
    record, frozen = tu.attach(Snapshot(), today)
    if frozen is None:
        raise Refusal(record["code"])
    result = {"schema_version": SCHEMA, "command": "pin", "apply": apply, "pin": found, "export": ids(value),
              "input_sha256": record["sha256"], "input_bytes": len(frozen), "provider_calls": 0, "object_writes": 0}
    if not apply:
        return {**result, "state": "planned", "firestore_writes": 0}
    if not idle(bridge):
        raise Refusal("team_universe_active_research_qa_repair_or_publication")
    with lease(bridge):
        if not idle(bridge):
            raise Refusal("team_universe_active_research_qa_repair_or_publication")
        receipt = bridge.call("team_universe_set", expected_sha256=observed, value=found)
    if (bridge.call("control") or {}).get("team_universe") != found:
        raise Refusal("team_universe_pin_readback_failed")
    return {**result, "state": "pinned", "readback_verified": True, "receipt": receipt, "firestore_writes": 1}


def show(bridge, *, today=None):
    class Snapshot:
        def team_universe_snapshot(self):
            return bridge.call("team_universe_snapshot")
    record, raw = tu.attach(Snapshot(), today or datetime.now(timezone.utc).date())
    result = {"schema_version": SCHEMA, "command": "show", "input": record, "provider_calls": 0,
              "firestore_writes": 0, "object_writes": 0}
    if raw:
        import json
        result["export"] = ids(json.loads(raw))
    return result


def main(argv=None, *, bridge_factory=Bridge, clock=lambda: datetime.now(timezone.utc)):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("show")
    pub = commands.add_parser("publish")
    pub.add_argument("--file", required=True, type=Path)
    pub.add_argument("--apply", action="store_true")
    setter = commands.add_parser("pin")
    setter.add_argument("--sha256", required=True)
    setter.add_argument("--generation", required=True)
    setter.add_argument("--approval-reference", required=True)
    setter.add_argument("--expect-current")
    setter.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    bridge = bridge_factory()
    try:
        if args.command == "show":
            result = show(bridge, today=clock().date())
        elif args.command == "publish":
            result = publish(bridge, args.file, apply=args.apply, today=clock().date())
        else:
            result = pin(bridge, sha256=args.sha256, generation=args.generation, approval_reference=args.approval_reference,
                         expect=args.expect_current, apply=args.apply, today=clock().date())
        print(canonical(result))
        return result
    finally:
        bridge.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - counts/stable codes only, no private row content
        code = str(error) if isinstance(error, (Refusal, tu.TeamEvidenceError)) else "team_universe_operation_unavailable"
        print(canonical({"state": "blocked", "error": code}))
        raise SystemExit(1) from None
