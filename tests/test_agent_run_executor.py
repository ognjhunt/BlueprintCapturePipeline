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
    admission = {
        "schema_version": executor.ADMISSION_SCHEMA_VERSION,
        "source_request_id": "request-1",
        "decision_request": {"request_id": "request-1"},
        "canonical_execution_request": {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": "canonical-job-1",
            "customer": {"id": "team-1"},
            "site_package": {"capture_id": "capture-1", "capture_root": str(tmp_path)},
            "requested_tasks": [{"task_id": "pick_place"}],
            "robot_profile": {"robot_profile_id": "arm-1"},
            "policy_package": {"policy_api_endpoint": {"endpoint_url": "https://policy.example"}},
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
        "execution_admission_digest": executor._digest(admission),
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

    def report_blocked(self, _row: Any, _pipeline_run_id: str, reason: str) -> None:
        self.blocked.append(reason)

    def report_completed(self, _row: Any, _pipeline_run_id: str, receipt: dict[str, Any]) -> None:
        self.completed.append(receipt)


def test_missing_admission_refuses_claim_and_execution(tmp_path: Path) -> None:
    row = _row(tmp_path)
    row.pop("execution_admission")
    client = FakeClient([row])
    called = False

    def must_not_execute(**_kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    summary = executor.poll_once(client=client, capture_root=tmp_path, executor=must_not_execute)
    assert summary == {"examined": 1, "claimed": 0, "blocked": 1, "completed": 0}
    assert not client.claims
    assert called is False


def test_digest_bound_episode_receipt_closes_claimed_run(tmp_path: Path) -> None:
    row = _row(tmp_path)
    pipeline_run_id = executor._pipeline_run_id(row)
    receipt = {
        "schema_version": "blueprint.agent_run_episode_receipt.v1",
        "pipeline_run_id": pipeline_run_id,
        "execution_admission_digest": row["execution_admission_digest"],
        "episodes_run": 5,
        "episodes_succeeded": 3,
        "median_cycle_seconds": 4.2,
        "artifact_uri": "gs://blueprint/results/run-1/receipt.json",
    }
    receipt["receipt_sha256"] = executor._digest(receipt)
    client = FakeClient([row])

    summary = executor.poll_once(
        client=client,
        capture_root=tmp_path,
        executor=lambda **_kwargs: {"agent_run_episode_receipt": receipt},
    )

    assert summary == {"examined": 1, "claimed": 1, "blocked": 0, "completed": 1}
    assert client.claims == [("run-1", pipeline_run_id)]
    assert client.completed == [receipt]


def test_unbound_executor_result_releases_as_no_episode(tmp_path: Path) -> None:
    client = FakeClient([_row(tmp_path)])
    summary = executor.poll_once(
        client=client,
        capture_root=tmp_path,
        executor=lambda **_kwargs: {"status": "completed"},
    )
    assert summary["blocked"] == 1
    assert client.blocked == ["canonical_executor_reported_no_episodes"]


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
        pipeline_run_id = executor._pipeline_run_id(row)
        receipt = {
            "schema_version": "blueprint.agent_run_episode_receipt.v1",
            "pipeline_run_id": pipeline_run_id,
            "execution_admission_digest": row["execution_admission_digest"],
            "episodes_run": 5,
            "episodes_succeeded": 4,
            "artifact_uri": "gs://blueprint/results/canonical-job-1/receipt.json",
        }
        receipt["receipt_sha256"] = executor._digest(receipt)
        client = executor.AgentRunWebAppClient(
            base_url=f"http://127.0.0.1:{server.server_port}", token=token
        )
        summary = executor.poll_once(
            client=client,
            capture_root=tmp_path,
            executor=lambda **_kwargs: {"agent_run_episode_receipt": receipt},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert summary == {"examined": 1, "claimed": 1, "blocked": 0, "completed": 1}
    assert [path for path, _ in received] == [
        "/api/internal/pipeline/agent-runs/run-1/started",
        "/api/internal/pipeline/agent-run-results",
        "/api/internal/pipeline/agent-run-settlements",
    ]
    assert all(
        payload["execution_admission_digest"] == row["execution_admission_digest"]
        for _, payload in received
    )
