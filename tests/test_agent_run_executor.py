from __future__ import annotations

from pathlib import Path
import hashlib
import hmac
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

from blueprint_pipeline import agent_run_executor as executor


def _row(tmp_path: Path) -> dict[str, Any]:
    episode_specs = tmp_path / "pipeline" / "simulation_automation" / "episode_specs.json"
    episode_specs.parent.mkdir(parents=True, exist_ok=True)
    episode_specs.write_text(json.dumps({"episode_count": 5}), encoding="utf-8")
    admission = {
        "schema_version": executor.ADMISSION_SCHEMA_VERSION,
        "source_request_id": "request-1",
        "decision_request": {"request_id": "request-1"},
        "canonical_execution_request": {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": "canonical-job-1",
            "customer": {"id": "team-1"},
            "site_package": {"capture_id": "capture-1", "capture_root": str(tmp_path)},
            "requested_tasks": [{"task_id": "pick_place", "scenario_ids": ["nominal"]}],
            "robot_profile": {"robot_profile_id": "arm-1"},
            "policy_package": {"policy_api_endpoint": {"endpoint_url": "https://policy.example"}},
            "execution_authorization": {
                "task_id": "pick_place",
                "scenario_id": "nominal",
                "episodes": 5,
                "max_cost_usd": 10,
            },
        },
        "binding": {
            "team_id": "team-1",
            "checkpoint_id": "checkpoint-1",
            "scene_request_id": "scene-request-1",
            "capture_id": "capture-1",
            "task_family": "pick_place",
        },
        "proof_boundary": {"provider_spend_authorized": False},
    }
    canonical_json = json.dumps(admission, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {
        "run_id": "run-1",
        "reservation_id": "reservation-1",
        "team_id": "team-1",
        "task_family": "pick_place",
        "quoted_episodes": 5,
        "quoted_usd": 10,
        "dispatch": None,
        "checkpoint": {"checkpoint_id": "checkpoint-1"},
        "scene": {"request_id": "scene-request-1"},
        "execution_admission": admission,
        "execution_admission_canonical_json": canonical_json,
        "execution_admission_digest": "sha256:" + hashlib.sha256(canonical_json.encode()).hexdigest(),
    }


class FakeClient:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows
        self.claims: list[tuple[str, str]] = []
        self.blocked: list[str] = []
        self.completed: list[dict[str, Any]] = []

    def list_runs(self, _limit: int) -> list[dict[str, Any]]:
        return self.rows

    def claim(self, run_id: str, pipeline_run_id: str, _admission_digest: str) -> dict[str, Any]:
        self.claims.append((run_id, pipeline_run_id))
        return {"pipeline_run_id": pipeline_run_id}

    def get_run(self, run_id: str) -> dict[str, Any]:
        return {"run_id": run_id, "dispatch": None}

    def report_blocked(self, _row: Any, _pipeline_run_id: str, reason: str) -> None:
        self.blocked.append(reason)

    def report_completed(self, _row: Any, _pipeline_run_id: str, receipt: dict[str, Any]) -> None:
        self.completed.append(receipt)


def test_missing_admission_refuses_claim_and_execution(tmp_path: Path) -> None:
    row = _row(tmp_path)
    row.pop("execution_admission")
    client = FakeClient([row])
    summary = executor.poll_once(client=client, capture_root=tmp_path)
    assert summary["examined"] == 1
    assert summary["claimed"] == 0
    assert summary["blocked"] == 1
    assert not client.claims


def test_digest_bound_episode_receipt_closes_claimed_run(tmp_path: Path) -> None:
    row = _row(tmp_path)
    receipt = {
        "status": "completed",
        "episodes_run": 5,
        "episodes_succeeded": 3,
        "median_cycle_seconds": 4.2,
        "artifact_uri": "gs://blueprint/results/run-1/receipt.json",
        "observed_cost_usd": 8.0,
    }
    client = FakeClient([row])

    summary = executor.poll_once(
        client=client,
        capture_root=tmp_path,
        terminal_reader=lambda **_kwargs: receipt,
    )

    assert summary["claimed"] == 1
    assert summary["staged"] == 1
    assert summary["completed"] == 1
    assert client.claims[0][0] == "run-1"
    assert client.completed == [receipt]


def test_pending_terminal_artifacts_keep_claimed_run_open(tmp_path: Path) -> None:
    client = FakeClient([_row(tmp_path)])
    summary = executor.poll_once(
        client=client,
        capture_root=tmp_path,
        terminal_reader=lambda **_kwargs: {"status": "pending"},
    )
    assert summary["pending"] == 1
    assert client.blocked == []


def test_frozen_utf8_admission_bytes_are_digest_authority(tmp_path: Path) -> None:
    row = _row(tmp_path)
    row["execution_admission"]["operator_name"] = "José 🤖"
    row["execution_admission"]["integer_quote"] = 1
    frozen = json.dumps(
        row["execution_admission"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    row["execution_admission_canonical_json"] = frozen
    row["execution_admission_digest"] = "sha256:" + hashlib.sha256(frozen.encode()).hexdigest()
    assert executor.validate_queue_run(row) == []


def test_restart_reconciles_committed_claim_before_staging(tmp_path: Path) -> None:
    row = _row(tmp_path)
    client = FakeClient([])
    owner = "agent-attempt-restart"
    client.get_run = lambda _run_id: {
        "run_id": "run-1",
        "dispatch": {"pipeline_run_id": owner},
    }
    journal_dir = tmp_path / "journal"
    executor._write_json_atomic(
        journal_dir / f"run-1--{owner}.json",
        {
            "schema_version": "blueprint.agent_run_executor_journal.v1",
            "state": "claim_intent",
            "run_id": "run-1",
            "pipeline_run_id": owner,
            "canonical_job_id": "canonical-job-1",
            "execution_admission_digest": row["execution_admission_digest"],
            "row": row,
        },
    )
    summary = executor.poll_once(
        client=client,
        capture_root=tmp_path,
        journal_dir=journal_dir,
        terminal_reader=lambda **_kwargs: {"status": "pending"},
    )
    assert summary["staged"] == 1
    assert summary["pending"] == 1
    assert client.claims == []


def test_signed_local_webapp_queue_closes_digest_bound_fixture(tmp_path: Path) -> None:
    row = _row(tmp_path)
    received: list[tuple[str, dict[str, Any]]] = []
    token = "pipeline-test-token"

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _body(self) -> tuple[bytes, dict[str, Any]]:
            raw = self.rfile.read(int(self.headers.get("content-length", "0")))
            return raw, json.loads(raw or b"{}")

        def _authorized(self, body: bytes) -> bool:
            timestamp = self.headers["X-Blueprint-Pipeline-Timestamp"]
            expected = hmac.new(
                token.encode(), timestamp.encode() + b"." + body, hashlib.sha256
            ).hexdigest()
            return self.headers["X-Blueprint-Pipeline-Signature"] == f"sha256={expected}"

        def do_GET(self) -> None:  # noqa: N802
            assert self._authorized(b"{}")
            self._reply({"runs": [row], "count": 1})

        def do_POST(self) -> None:  # noqa: N802
            raw, payload = self._body()
            assert self._authorized(raw)
            received.append((self.path, payload))
            if self.path.endswith("/started"):
                self._reply({"pipeline_run_id": payload["pipeline_run_id"]})
            else:
                self._reply({"ok": True})

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        receipt = {
            "status": "completed",
            "episodes_run": 5,
            "episodes_succeeded": 4,
            "artifact_uri": "gs://blueprint/results/canonical-job-1/receipt.json",
            "observed_cost_usd": 8.0,
        }
        client = executor.AgentRunWebAppClient(
            base_url=f"http://127.0.0.1:{server.server_port}", token=token
        )
        summary = executor.poll_once(
            client=client,
            capture_root=tmp_path,
            terminal_reader=lambda **_kwargs: receipt,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert summary["claimed"] == 1
    assert summary["staged"] == 1
    assert summary["completed"] == 1
    assert [path for path, _ in received] == [
        "/api/internal/pipeline/agent-runs/run-1/started",
        "/api/internal/pipeline/agent-run-results",
        "/api/internal/pipeline/agent-run-settlements",
    ]
    assert all(
        payload["execution_admission_digest"] == row["execution_admission_digest"]
        for _, payload in received
    )
