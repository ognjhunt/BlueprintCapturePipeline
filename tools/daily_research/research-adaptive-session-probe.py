"""Exact-session read-only admission projection; no inference or Firestore writes."""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_SOURCE = "b9aa5d0d129ecf02082a766e12657a94097e803a"
STATUS = "64a32eddce89a61b5b5d3984ac722eab9573a2a91a5995f0231eb0d84140d16f"
SESSION = "sess_03449d612c6384f3006abe4b46ab20819aa8405df2844574d9"
sys.path.insert(0, str(Path.cwd()))
from tools.daily_research.consumer import Consumer
from tools.daily_research.firestore import Bridge, FirestoreLedger
from tools.daily_research.runner import (
    Provider,
    Refusal,
    canonical,
    digest,
    read_json,
)


def enum(value, allowed):
    return value if value in allowed else "unknown"


def main():
    bridge = api = None
    receipt = {"schema_version": "blueprint.adaptive-session-probe.v1", "observed_at": datetime.now(timezone.utc).isoformat(),
               "provider_mutations": 0, "firestore_writes": 0, "errors": [], "complete": False}
    try:
        if read_json("manifest.json").get("source_commit") != EXPECTED_SOURCE:
            raise Refusal("installed_source_mismatch")
        bridge = Bridge()
        row = FirestoreLedger(bridge).get("2026-10-01")
        control = bridge.call("control")
        if control.get("source_commit") != EXPECTED_SOURCE or digest(row) != STATUS or row.get("session_id") != SESSION:
            raise Refusal("retained_release_or_intent_mismatch")
        api = Provider(os.environ.get("OPENAI_API_KEY", ""))
        # Enforce GET-only transport even if the installed read wrappers drift.
        def guard(request):
            if request.method != "GET":
                raise Refusal("provider_read_only_required")
        api.client._client.event_hooks["request"].append(guard)
        session = api.get("session", SESSION)
        Consumer.check_session(row, session)
        env = api.get("environment", row["environment_id"])
        turns = api.listing("turns", SESSION)
        root = [t for t in turns if t.get("subagent_id") is None]
        if any(t.get("session_id") not in (None, SESSION) for t in turns):
            raise Refusal("exact_session_turn_binding_mismatch")
        tier = enum(session.get("agent", {}).get("service_tier"), {"auto", "default", "flex", "priority", "fast", "ultrafast"})
        size = enum(session.get("environment", {}).get("container_size"), {"small", "medium", "large", "xlarge"})
        required = session.get("required_actions")
        receipt.update(session_id=SESSION, environment_id=row["environment_id"],
                       original_turn_id=row["turn_id"], original_failed_intent_unchanged=True,
                       session_status=enum(session.get("status"), {"idle", "in_progress", "requires_action", "failed"}),
                       environment_status=enum(env.get("status"), {"pending", "connected", "disconnected", "expired", "failed"}),
                       environment_id_matches=env.get("id") == row["environment_id"],
                       service_tier=tier, reported_container_size=size,
                       required_action_count=len(required) if isinstance(required, list) else (0 if not required else "unknown"),
                       turn_count=len(turns), root_turns=[{"id": t["id"], "status":enum(t.get("status"), {"pending", "in_progress", "completed", "failed", "cancelled"})} for t in root],
                       subagent_turn_count=len(turns)-len(root), cleanup_required=row.get("cleanup_required"),
                       qa_present=bool(row.get("qa")), publication_present=bool(row.get("delivery")))
        receipt["same_session_followup_admissible"] = (receipt["session_status"] == "idle" and not required
            and receipt["environment_status"] == "connected" and receipt["environment_id_matches"]
            and tier == "default" and size == "small" and len(turns) == len(root) == 1
            and root[0].get("id") == row["turn_id"] and root[0].get("status") == "completed")
        receipt["complete"] = True
    except Exception as exc:  # noqa: BLE001 - never expose upstream bodies or credentials
        status = getattr(exc, "status_code", None)
        receipt["errors"].append(str(exc) if isinstance(exc, Refusal) else
                                 "exact_session_read_http_" + str(status) if type(status) is int else "exact_session_read_unavailable")
    finally:
        if api:
            api.client.close()
        if bridge:
            bridge.close()
    print(canonical(receipt))
    return 0 if receipt["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
