from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from blueprint_pipeline import provider_worker_policy_command_adapter as adapter
from blueprint_pipeline import provider_worker_session_runner as session_runner

import pytest

pytestmark = pytest.mark.slow


class _WorkerServer:
    def __init__(
        self,
        *,
        ready_payload: Mapping[str, Any] | None = None,
        infer_payload: Mapping[str, Any] | None = None,
        require_token: str | None = None,
    ) -> None:
        self.ready_payload = dict(
            ready_payload
            or {"status": "ready", "ready_for_inference": True, "model_ready": True}
        )
        self.infer_payload = dict(
            infer_payload
            or {
                "policy_id": "unitree_provider_worker_policy",
                "model_ran": True,
                "unitree_policy_action_command_ran": True,
                "action": {"action_type": "waypoint", "target_waypoint": [0.5, 0.0]},
            }
        )
        self.require_token = require_token
        self.ready_count = 0
        self.infer_count = 0
        self.shutdown_count = 0
        self.last_infer_request: dict[str, Any] = {}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:
                return

            def _authorized(self) -> bool:
                if not owner.require_token:
                    return True
                return self.headers.get("Authorization") == f"Bearer {owner.require_token}"

            def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
                body = json.dumps(dict(payload)).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if not self._authorized():
                    self._send_json(403, {"status": "blocked", "error": "forbidden"})
                    return
                if urlsplit(self.path).path == "/readyz":
                    owner.ready_count += 1
                    self._send_json(200, owner.ready_payload)
                    return
                self._send_json(404, {"status": "missing"})

            def do_POST(self) -> None:  # noqa: N802
                if not self._authorized():
                    self._send_json(403, {"status": "blocked", "error": "forbidden"})
                    return
                if urlsplit(self.path).path == "/infer":
                    owner.infer_count += 1
                    length = int(self.headers.get("Content-Length") or "0")
                    raw = self.rfile.read(length).decode("utf-8")
                    owner.last_infer_request = json.loads(raw or "{}")
                    self._send_json(200, owner.infer_payload)
                    return
                if urlsplit(self.path).path == "/shutdown":
                    owner.shutdown_count += 1
                    self._send_json(
                        200,
                        {
                            "status": "shutdown_requested",
                            "shutdown_acknowledged": True,
                            "secret_token": "do-not-write",
                        },
                    )
                    return
                self._send_json(404, {"status": "missing"})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_WorkerServer":
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address
        return f"http://{host}:{port}"


def test_provider_worker_policy_command_adapter_calls_ready_then_infer(
    tmp_path: Path,
) -> None:
    token_file = tmp_path / "token.txt"
    token_file.write_text("secret-token\n", encoding="utf-8")
    with _WorkerServer(
        ready_payload={
            "status": "ready",
            "ready_for_inference": True,
            "model_ready": True,
            "secret_token": "do-not-write",
        },
        infer_payload={
            "policy_id": "unitree_provider_worker_policy",
            "model_ran": True,
            "unitree_policy_action_command_ran": True,
            "secret_token": "do-not-write",
            "action": {"action_type": "waypoint", "target_waypoint": [0.5, 0.0]},
        },
        require_token="secret-token",
    ) as server:
        response, exit_code = adapter.run_provider_worker_policy_command(
            payload={"observation": {"task_id": "contact_or_push_light_object"}},
            worker_url=f"{server.base_url}/infer?token=must-not-persist",
            ready_url=f"{server.base_url}/readyz?token=must-not-persist",
            auth_token_file=str(token_file),
            timeout_seconds=2,
            ready_timeout_seconds=2,
            ready_poll_seconds=0.01,
        )

    assert exit_code == 0
    assert response["status"] == "completed"
    assert response["provider_worker_policy_command_ran"] is True
    assert response["provider_worker_readyz_checked"] is True
    assert response["provider_worker_infer_completed"] is True
    assert response["unitree_policy_action_command_ran"] is True
    assert response["action"]["action_type"] == "waypoint"
    assert response["worker_response_redacted"]["secret_token"] == "<redacted>"
    assert "must-not-persist" not in json.dumps(response)
    assert "secret-token" not in json.dumps(response)
    assert server.ready_count == 1
    assert server.infer_count == 1
    assert server.last_infer_request["observation"]["task_id"] == "contact_or_push_light_object"


def test_provider_worker_policy_command_adapter_blocks_when_not_ready() -> None:
    with _WorkerServer(
        ready_payload={"status": "loading", "ready_for_inference": False}
    ) as server:
        response, exit_code = adapter.run_provider_worker_policy_command(
            payload={"observation": {"task_id": "contact_or_push_light_object"}},
            worker_url=f"{server.base_url}/infer",
            ready_url=f"{server.base_url}/readyz",
            timeout_seconds=2,
            ready_timeout_seconds=0.05,
            ready_poll_seconds=0.01,
        )

    assert exit_code == 2
    assert response["status"] == "blocked"
    assert "provider_policy_worker_not_ready" in response["blockers"]
    assert server.ready_count >= 1
    assert server.infer_count == 0


def test_provider_worker_policy_command_adapter_blocks_missing_action() -> None:
    with _WorkerServer(infer_payload={"policy_id": "missing_action"}) as server:
        response, exit_code = adapter.run_provider_worker_policy_command(
            payload={"observation": {"task_id": "contact_or_push_light_object"}},
            worker_url=f"{server.base_url}/infer",
            ready_url=f"{server.base_url}/readyz",
            timeout_seconds=2,
            ready_timeout_seconds=2,
            ready_poll_seconds=0.01,
        )

    assert exit_code == 1
    assert response["status"] == "blocked"
    assert "provider_policy_worker_response_missing_action" in response["blockers"]
    assert server.infer_count == 1


def test_provider_worker_policy_command_adapter_main_reads_and_writes_env(
    tmp_path: Path, monkeypatch,
) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(
        json.dumps({"observation": {"task_id": "contact_or_push_light_object"}}),
        encoding="utf-8",
    )
    with _WorkerServer() as server:
        monkeypatch.setenv("BLUEPRINT_POLICY_ACTION_INPUT", str(input_path))
        monkeypatch.setenv("BLUEPRINT_POLICY_ACTION_OUTPUT", str(output_path))
        monkeypatch.setenv("BLUEPRINT_PROVIDER_POLICY_WORKER_URL", f"{server.base_url}/infer")
        monkeypatch.setenv(
            "BLUEPRINT_PROVIDER_POLICY_WORKER_READY_URL",
            f"{server.base_url}/readyz",
        )

        assert adapter.main(["--ready-timeout-seconds", "2"]) == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["policy_id"] == "unitree_provider_worker_policy"
    assert payload["action"]["action_type"] == "waypoint"


def test_provider_worker_session_runner_reuses_ready_worker_and_shutdowns_once(
    tmp_path: Path,
) -> None:
    with _WorkerServer() as server:
        manifest = session_runner.run_provider_worker_session(
            observations=[
                {"observation": {"task_id": "task-1"}},
                {"observation": {"task_id": "task-2"}},
                {"observation": {"task_id": "task-3"}},
            ],
            output_dir=tmp_path / "session",
            worker_url=f"{server.base_url}/infer",
            ready_url=f"{server.base_url}/readyz",
            shutdown_url=f"{server.base_url}/shutdown",
            ready_timeout_seconds=2,
            ready_poll_seconds=0.01,
            request_shutdown=True,
            generated_at="now",
        )

    assert manifest["status"] == "completed"
    assert manifest["provider_worker_session_reused"] is True
    assert manifest["readyz_checked_before_batch"] is True
    assert manifest["requested_infer_count"] == 3
    assert manifest["attempted_infer_count"] == 3
    assert manifest["completed_infer_count"] == 3
    assert manifest["shutdown_acknowledged"] is True
    assert manifest["provider_shutdown_proven"] is False
    assert "do-not-write" not in json.dumps(manifest)
    assert server.ready_count == 1
    assert server.infer_count == 3
    assert server.shutdown_count == 1
    output_rows = [
        json.loads(line)
        for line in Path(manifest["outputs_jsonl"]).read_text(encoding="utf-8").splitlines()
    ]
    assert len(output_rows) == 3
    assert all(row["status"] == "completed" for row in output_rows)
