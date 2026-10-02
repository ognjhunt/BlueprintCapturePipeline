"""Optional Render store. Existing Firebase Admin is used through a private pipe."""
import base64
import json
import os
import re
import select
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from tools.daily_research.runner import (
    AGENT,
    MODEL,
    PROJECT,
    TEMPLATE,
    Provider,
    Refusal,
    canonical,
)


class Bridge:
    def __init__(self, node="node", script=None):
        self.closed = False
        self.broken = False
        self.process = subprocess.Popen(
            [node, str(script or Path(__file__).with_name("firestore_bridge.mjs"))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1,
            env={k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "FIREBASE_SERVICE_ACCOUNT_JSON", "NOTION_API_TOKEN", "NOTION_API_KEY"}},
        )

    def call(self, op, **fields):
        if self.broken or self.closed:
            raise Refusal("firestore_bridge_unavailable")
        try:
            self.process.stdin.write(canonical({"op": op, **fields}) + "\n")
            self.process.stdin.flush()
            if not select.select([self.process.stdout], [], [], 35)[0]:
                self.broken = True
                self.close()
                raise Refusal("firestore_bridge_deadline")
            result = json.loads(self.process.stdout.readline())
            if not isinstance(result, dict) or type(result.get("ok")) is not bool:
                raise ValueError("invalid protocol frame")
            if result.get("ok") is not True:
                code = result.get("error", "firestore_bridge_unavailable")
                raise Refusal(code if isinstance(code, str) and code.replace("_", "").isalnum() else "firestore_bridge_unavailable")
            return result["value"]
        except (OSError, ValueError, KeyError):
            self.broken = True
            self.close()
            raise Refusal("firestore_bridge_unavailable") from None

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)


class FirestoreLedger:
    def __init__(self, bridge):
        self.bridge = bridge

    @contextmanager
    def lock(self):
        self.bridge.call("acquire")
        try:
            yield
        finally:
            self.bridge.call("release")

    def rows(self):
        return self.bridge.call("rows")

    def get(self, day):
        return self.bridge.call("get", day=day)

    def put(self, row):
        from tools.daily_research import search
        if row.get("search_provider") == search.PROFILE and len(canonical(row).encode()) > search.MAX_RECORD:
            raise Refusal("research_tool_record_resource_ceiling")
        self.bridge.call("put", row=row)

    def write_bytes(self, name, value):
        self.bridge.call("file_put", name=name, bytes=base64.b64encode(value).decode("ascii"))

    def write_json(self, name, value):
        self.write_bytes(name, (canonical(value) + "\n").encode())

    def read_bytes(self, name):
        try:
            return base64.b64decode(self.bridge.call("file_get", name=name), validate=True)
        except Refusal as exc:
            if str(exc) == "firestore_file_missing":
                raise FileNotFoundError(name) from None
            raise


class FencedProvider(Provider):
    def __init__(self, ledger, api_key):
        super().__init__(api_key)
        self.ledger = ledger

    def create(self, payload):
        self.ledger.bridge.call("create_check", day=payload["metadata"]["run_key"].split(":", 1)[1], metadata=payload["metadata"])
        return super().create(payload)

    def cancel(self, session_id, run_key):
        self.ledger.bridge.call("assert_lease")
        return super().cancel(session_id, run_key)

    def tool_admit(self, row, phase):
        super().tool_admit(row, phase)
        self.ledger.bridge.call("assert_lease")
        control = self.ledger.bridge.call("control")
        if (control.get("enabled") is not True or control.get("config", {}).get("search_provider") != row.get("search_provider")
                or phase == "qa" and control.get("workflow", {}).get("enabled") is not True):
            raise Refusal("research_tool_disabled_or_profile_changed")
        if (
                control.get("config", {}).get("recurring_budget_authority_reference") != row["recurring_budget_authority_reference"]
                or control.get("config", {}).get("soft_target_usd") != row["soft_target_usd"]):
            raise Refusal("research_tool_budget_authority_changed")

    def qa_input(self, session_id, event, key, day, request_digest, deadline_ms):
        self.ledger.bridge.call("qa_check", day=day, request_digest=request_digest, deadline_ms=deadline_ms)
        if datetime.now(timezone.utc).timestamp() * 1000 >= deadline_ms:
            raise Refusal("agent_qa_total_runtime_exhausted")
        self.api.sessions.events.create(session_id, events=[event], idempotency_key=key)

    def repair_input(self, session_id, event, key, day, request_digest, deadline_ms, attempt):
        # One durable claim per attempt, under the enabled control and pinned budget.
        self.ledger.bridge.call("repair_check", day=day, attempt=attempt, request_digest=request_digest, deadline_ms=deadline_ms)
        if datetime.now(timezone.utc).timestamp() * 1000 >= deadline_ms:
            raise Refusal("research_repair_window_expired")
        self.api.sessions.events.create(session_id, events=[event], idempotency_key=key)


def control_configuration(value):
    if (not isinstance(value, dict) or value.get("schema_version") != "blueprint.research-control.v1"
            or value.get("project_id") != PROJECT or value.get("agent_id") != AGENT
            or value.get("model") != MODEL or value.get("template_id") != TEMPLATE
            or type(value.get("enabled")) is not bool):
        raise Refusal("firestore_control_binding_invalid")
    config = value.get("config")
    if not isinstance(config, dict):
        raise Refusal("firestore_control_config_missing")
    if value["enabled"] and not re.fullmatch(r"[a-f0-9]{64}", str(config.get("expected_agent_instructions_sha256", ""))):
        raise Refusal("agent_instructions_pin_required")
    return {**config, "enabled": value["enabled"]}
