from __future__ import annotations

from pathlib import Path
import hashlib
import hmac
import json
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

import pytest

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
            "capture_root": str(tmp_path),
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
        "funding": {"payer": "blueprint", "customer_price_usd": 0, "cap_usd": 10,
                    "max_attempts": 1, "expires_at_iso": "2100-01-01T00:00:00Z",
                    "approved_by": "fixture-operator", "approval_digest": "sha256:" + "a" * 64},
    }
    canonical_json = json.dumps(admission, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {
        "run_id": "run-1",
        "reservation_id": "reservation-1",
        "team_id": "team-1",
        "task_family": "pick_place",
        "quoted_episodes": 5,
        "quoted_usd": 0,
        "evaluation_purpose": "pilot",
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

    def list_runs(self, _limit: int, *, capture_id: str | None = None) -> list[dict[str, Any]]:
        if capture_id:
            return [
                row for row in self.rows
                if row["execution_admission"]["binding"]["capture_id"] == capture_id
            ]
        return self.rows

    def claim(self, run_id: str, pipeline_run_id: str, _admission_digest: str) -> dict[str, Any]:
        self.claims.append((run_id, pipeline_run_id))
        return {"pipeline_run_id": pipeline_run_id}

    def get_run(self, run_id: str) -> dict[str, Any]:
        return {"run_id": run_id, "state": "requested", "money_resolved": False, "dispatch": None}

    def report_blocked(self, _row: Any, _pipeline_run_id: str, reason: str) -> None:
        self.blocked.append(reason)

    def report_completed(self, _row: Any, _pipeline_run_id: str, receipt: dict[str, Any]) -> None:
        self.completed.append(receipt)


@pytest.mark.parametrize("purpose,price", [(None, 0), ("private", 0), ("private", 99), ("pilot", 1), ("pilot", False), ("pilot", "0"), ("unknown", 0)])
def test_free_beta_refuses_paid_or_unclassified_queue_rows(tmp_path: Path, purpose: Any, price: Any) -> None:
    row = _row(tmp_path)
    row["evaluation_purpose"] = purpose
    row["quoted_usd"] = price
    client = FakeClient([row])
    summary = executor.poll_once(client=client, capture_root=tmp_path)
    assert summary["blocked"] == 1
    assert summary["claimed"] == 0
    assert not client.claims
    assert not client.completed


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


def test_missing_provider_cost_does_not_block_quoted_episode_settlement(tmp_path: Path) -> None:
    row = _row(tmp_path)
    # Historical paid outcomes still settle even though new paid claims refuse.
    row["quoted_usd"] = 10
    received: list[tuple[str, dict[str, Any]]] = []

    class RecordingClient(FakeClient):
        def _json(self, path: str, *, method: str = "GET", payload: dict[str, Any] | None = None):
            received.append((path, payload or {}))
            return {"ok": True}

    client = RecordingClient([])
    executor.AgentRunWebAppClient.report_completed(
        client,
        row,
        "attempt-1",
        {"episodes_run": 4, "episodes_succeeded": 3, "artifact_uri": "gs://result"},
    )
    settlement = received[-1][1]
    assert settlement["episodes_run"] == 4
    assert settlement["rate_usd"] == 2.0
    assert "observed_cost_usd" not in settlement


def test_pending_terminal_artifacts_keep_claimed_run_open(tmp_path: Path) -> None:
    client = FakeClient([_row(tmp_path)])
    summary = executor.poll_once(
        client=client,
        capture_root=tmp_path,
        terminal_reader=lambda **_kwargs: {"status": "pending"},
    )
    assert summary["pending"] == 1
    assert client.blocked == []


def test_episode_spec_mismatch_claims_and_releases_full_hold(tmp_path: Path) -> None:
    row = _row(tmp_path)
    (tmp_path / "pipeline" / "simulation_automation" / "episode_specs.json").write_text(
        json.dumps({"episode_count": 6}), encoding="utf-8"
    )
    client = FakeClient([row])
    summary = executor.poll_once(client=client, capture_root=tmp_path)
    assert summary["claimed"] == 1
    assert summary["blocked"] == 1
    assert client.blocked == ["agent_execution_episode_spec_count_mismatch"]


def test_unapproved_staged_policy_claims_and_releases_without_execution(tmp_path: Path) -> None:
    row = _row(tmp_path)
    staged = tmp_path / "pipeline" / "robot_eval_inputs" / "canonical-job-1" / "policy_package.json"
    staged.parent.mkdir(parents=True)
    staged.write_text(json.dumps({"job_id": "canonical-job-1", "policy_package": {}}))
    client = FakeClient([row])
    summary = executor.poll_once(client=client, capture_root=tmp_path)
    assert summary["claimed"] == 1
    assert summary["blocked"] == 1
    assert client.blocked == ["agent_execution_unapproved_staged_policy_package"]


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
        "state": "requested",
        "money_resolved": False,
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
    assert client.claims == [("run-1", owner)]


def test_restart_preserves_preflight_release_disposition_after_claim(tmp_path: Path) -> None:
    row = _row(tmp_path)
    owner = "agent-attempt-blocked"
    client = FakeClient([])
    client.get_run = lambda _run_id: {
        "run_id": "run-1", "state": "requested", "money_resolved": False,
        "dispatch": {"pipeline_run_id": owner},
    }
    journal_dir = tmp_path / "journal"
    executor._write_json_atomic(journal_dir / f"run-1--{owner}.json", {
        "schema_version": "blueprint.agent_run_executor_journal.v1",
        "state": "claim_intent", "run_id": "run-1", "pipeline_run_id": owner,
        "canonical_job_id": "canonical-job-1",
        "execution_admission_digest": row["execution_admission_digest"],
        "disposition": "block_before_execution",
        "preflight_blockers": ["agent_execution_episode_spec_count_mismatch"],
        "row": row,
    })
    summary = executor.poll_once(client=client, capture_root=tmp_path, journal_dir=journal_dir)
    assert summary["blocked"] == 1
    assert client.blocked == ["agent_execution_episode_spec_count_mismatch"]
    assert not (tmp_path / "pipeline" / "robot_eval_job_requests" / "inbox" / "canonical-job-1.json").exists()


def test_restart_does_not_stage_money_resolved_run(tmp_path: Path) -> None:
    row = _row(tmp_path)
    owner = "agent-attempt-resolved"
    client = FakeClient([])
    client.get_run = lambda _run_id: {
        "run_id": "run-1", "state": "completed", "money_resolved": True,
        "dispatch": {"pipeline_run_id": owner},
    }
    journal_dir = tmp_path / "journal"
    executor._write_json_atomic(journal_dir / f"run-1--{owner}.json", {
        "schema_version": "blueprint.agent_run_executor_journal.v1",
        "state": "claim_intent", "run_id": "run-1", "pipeline_run_id": owner,
        "canonical_job_id": "canonical-job-1",
        "execution_admission_digest": row["execution_admission_digest"],
        "disposition": "stage_for_execution", "preflight_blockers": [], "row": row,
    })
    summary = executor.poll_once(client=client, capture_root=tmp_path, journal_dir=journal_dir)
    assert summary["blocked"] == 1
    assert client.claims == []


def test_restart_records_same_owner_lease_409_without_staging(tmp_path: Path) -> None:
    row = _row(tmp_path)
    owner = "agent-attempt-expired"
    client = FakeClient([])
    client.get_run = lambda _run_id: {
        "run_id": "run-1", "state": "requested", "money_resolved": False,
        "dispatch": {"pipeline_run_id": owner},
    }
    client.claim = lambda *_args: (_ for _ in ()).throw(
        urllib.error.HTTPError("http://local", 409, "expired", {}, None)
    )
    journal_dir = tmp_path / "journal"
    path = journal_dir / f"run-1--{owner}.json"
    executor._write_json_atomic(path, {
        "schema_version": "blueprint.agent_run_executor_journal.v1",
        "state": "claim_intent", "run_id": "run-1", "pipeline_run_id": owner,
        "canonical_job_id": "canonical-job-1",
        "execution_admission_digest": row["execution_admission_digest"],
        "disposition": "stage_for_execution", "preflight_blockers": [], "row": row,
    })
    summary = executor.poll_once(client=client, capture_root=tmp_path, journal_dir=journal_dir)
    assert summary["blocked"] == 1
    assert executor._load_json(path)["state"] == "claim_rejected"


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
    assert received[-1][1]["rate_usd"] == 0.0


def _partition_row(partition: Path, scene_id: str, capture_id: str, run_id: str) -> dict[str, Any]:
    capture_root = partition / "scenes" / scene_id / "captures" / capture_id
    row = _row(capture_root)
    canonical = row["execution_admission"]["canonical_execution_request"]
    canonical["job_id"] = f"canonical-{run_id}"
    canonical["site_package"]["capture_id"] = capture_id
    row["execution_admission"]["binding"]["capture_id"] = capture_id
    row["run_id"] = run_id
    canonical_json = json.dumps(row["execution_admission"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    row["execution_admission_canonical_json"] = canonical_json
    row["execution_admission_digest"] = "sha256:" + hashlib.sha256(canonical_json.encode()).hexdigest()
    return row


def test_partition_scope_serves_every_admitted_capture_without_reconfiguration(tmp_path: Path) -> None:
    partition = tmp_path / "captures"
    first = _partition_row(partition, "site-a", "walkthrough-a", "run-a")
    second = _partition_row(partition, "site-b", "walkthrough-b", "run-b")
    outside_root = tmp_path / "elsewhere" / "scenes" / "site-c" / "captures" / "walkthrough-c"
    outside = _partition_row(tmp_path / "elsewhere", "site-c", "walkthrough-c", "run-c")
    assert outside_root.is_dir()
    mismatched = _partition_row(partition, "site-d", "walkthrough-d", "run-d")
    mismatched["execution_admission"]["binding"]["capture_id"] = "walkthrough-other"
    client = FakeClient([first, second, outside, mismatched])
    seen: list[Path] = []

    def reader(*, job_dir: Path, **_kwargs: Any) -> dict[str, Any]:
        seen.append(job_dir)
        return {"status": "completed", "episodes_run": 5, "episodes_succeeded": 4}

    summary = executor.poll_once(
        client=client, capture_partition_root=partition, inbox_dir=tmp_path / "inbox",
        journal_dir=tmp_path / "journal", terminal_reader=reader)

    assert summary["examined"] == 4 and summary["claimed"] == 2 and summary["completed"] == 2
    assert summary["blocked"] == 2
    assert {claim[0] for claim in client.claims} == {"run-a", "run-b"}
    assert sorted(path.parents[2].name for path in seen) == ["walkthrough-a", "walkthrough-b"]
    journals = [json.loads(path.read_text()) for path in sorted((tmp_path / "journal").glob("*.json"))]
    assert {journal["capture_root"] for journal in journals} == {
        str((partition / "scenes/site-a/captures/walkthrough-a").resolve()),
        str((partition / "scenes/site-b/captures/walkthrough-b").resolve())}


def test_capture_scope_must_be_exactly_one_of_single_or_partition(tmp_path: Path) -> None:
    client = FakeClient([])
    with pytest.raises(ValueError, match="agent_execution_capture_scope_required"):
        executor.poll_once(client=client)
    with pytest.raises(ValueError, match="agent_execution_capture_scope_required"):
        executor.poll_once(client=client, capture_root=tmp_path, capture_partition_root=tmp_path)
    with pytest.raises(ValueError, match="requires_shared_inbox_and_journal"):
        executor.poll_once(client=client, capture_partition_root=tmp_path)


def test_installed_dispatcher_and_control_plane_share_the_canonical_inbox() -> None:
    repo = Path(__file__).resolve().parents[1]
    dispatcher = (repo / "deploy/systemd/blueprint-agent-run-dispatcher.service").read_text()
    control_plane = (repo / "deploy/systemd/blueprint-pipeline-control-plane.service").read_text()
    timer = (repo / "deploy/systemd/blueprint-pipeline-control-plane.timer").read_text()
    env_example = (repo / "deploy/systemd/pipeline-control-plane.env.example").read_text()
    installer = (repo / "scripts/install_live_pipeline_control_plane.sh").read_text()
    assert "EnvironmentFile=/etc/blueprint/pipeline-control-plane.env" in dispatcher
    assert 'test "$${BLUEPRINT_AGENT_RUN_DISPATCH_ENABLED:-false}" = true' in dispatcher
    assert '--inbox-dir "$${BLUEPRINT_ROBOT_EVAL_JOB_REQUEST_INBOX}"' in dispatcher
    assert "BLUEPRINT_AGENT_RUN_CAPTURE_PARTITION_ROOT" in dispatcher
    assert "BLUEPRINT_AGENT_RUN_CAPTURE_PARTITION_ROOT=" in env_example
    assert "blueprint_pipeline.live_pipeline_control_plane" in control_plane
    assert "OnUnitActiveSec=5min" in timer
    assert (
        "BLUEPRINT_ROBOT_EVAL_JOB_REQUEST_INBOX=/var/lib/blueprint/pipeline-control-plane/robot-eval-job-requests"
        in env_example
    )
    assert "blueprint-agent-run-dispatcher.timer" in installer
    assert "blueprint-pipeline-control-plane.timer" in installer


@pytest.mark.parametrize("partition", ["", "capture partition"])
def test_dispatcher_service_passes_one_capture_scope(tmp_path: Path, partition: str) -> None:
    import os
    import shlex
    import subprocess
    service = Path(__file__).resolve().parents[1] / "deploy/systemd/blueprint-agent-run-dispatcher.service"
    line = next(line for line in service.read_text().splitlines() if line.startswith("ExecStart="))
    command = shlex.split(line.removeprefix("ExecStart="))[2].replace("$$", "$")
    printer = tmp_path / "print-argv"
    printer.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    printer.chmod(0o755)
    env = {**os.environ, "BLUEPRINT_PIPELINE_REPO": "/archived/check-out",
           "BLUEPRINT_PIPELINE_PYTHON": "/archived/python",
           "BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_REPO": str(tmp_path),
           "BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_PYTHON": str(printer), "BLUEPRINT_WEBAPP_URL": "https://example.com",
           "BLUEPRINT_AGENT_RUN_CAPTURE_PARTITION_ROOT": partition,
           "BLUEPRINT_AGENT_RUN_CAPTURE_ROOT": "single capture", "BLUEPRINT_AGENT_RUN_CAPTURE_ID": "capture-1",
           "BLUEPRINT_PIPELINE_SYNC_TOKEN_FILE": "token file", "BLUEPRINT_ROBOT_EVAL_JOB_REQUEST_INBOX": "inbox",
           "BLUEPRINT_AGENT_RUN_JOURNAL_DIR": "journal"}
    args = subprocess.run(["/bin/bash", "-c", command], env=env, check=True,
                          text=True, capture_output=True).stdout.splitlines()
    expected_scope = ["--capture-partition-root", partition] if partition else ["--capture-root", "single capture", "--capture-id", "capture-1"]
    assert args == ["-m", "blueprint_pipeline.agent_run_executor", "--webapp-url", "https://example.com",
                    *expected_scope, "--token-file", "token file", "--inbox-dir", "inbox", "--journal-dir", "journal"]


def test_controlled_request_without_registry_never_stages_in_legacy_inbox(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline import controlled_native_queue as native
    from blueprint_pipeline import adp_task_evaluation_abstention as abstention

    monkeypatch.delenv(native.REGISTRY_ENV, raising=False)
    monkeypatch.setattr(abstention, "collect_vast_provider_zero_receipt", lambda: {"provider_zero": True})
    row = _row(tmp_path)
    canonical = row["execution_admission"]["canonical_execution_request"]
    canonical["policy_package"]["policy_api_endpoint"]["execution_profile"] = "controlled_observation_v1"
    inbox = tmp_path / "inbox"
    staged = executor._stage_canonical_request(row, inbox)
    assert staged == inbox / "controlled-native" / "canonical-job-1.json"
    assert not (inbox / "canonical-job-1.json").exists()
    job = tmp_path / "native-job"
    native.execute_staged_controlled_request(request=canonical, job_dir=job)
    terminal = native.read_native_terminal(job_dir=job, expected_job_id=canonical["job_id"],
        expected_canonical_request_digest=executor._digest(canonical))
    assert terminal["status"] == "blocked"
    assert terminal["blockers"] == ["controlled_native_task_profile_required"]
    assert not (job / "controlled_native_execution_intent.json").exists()
    assert not (job / "native_allocator_result.json").exists()


@pytest.mark.parametrize("legacy_count", [None, 3])
def test_native_admission_uses_frozen_profile_instead_of_mutable_legacy_episodes(tmp_path: Path, monkeypatch, legacy_count) -> None:
    from blueprint_pipeline import controlled_native_queue as native
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    row = _row(tmp_path)
    request = row["execution_admission"]["canonical_execution_request"]
    request["policy_package"]["policy_api_endpoint"]["execution_profile"] = "controlled_observation_v1"
    request["execution_authorization"]["episodes"] = row["quoted_episodes"] = 1
    frozen = json.dumps(row["execution_admission"], sort_keys=True, separators=(",", ":"))
    row["execution_admission_canonical_json"] = frozen
    row["execution_admission_digest"] = "sha256:" + hashlib.sha256(frozen.encode()).hexdigest()
    specs = tmp_path / "pipeline/simulation_automation/episode_specs.json"
    if legacy_count is None:
        specs.unlink()
    else:
        specs.write_text(json.dumps({"episode_count": legacy_count}))
    registry = {"schema_version": "blueprint.controlled_native_registry.v1", "profiles": [{
        "capture_root": str(tmp_path), "task_id": "pick_place", "scenario_id": "nominal",
        "allowed_team_ids": ["team-1"], "allowed_checkpoint_ids": ["arm-1"]}]}
    registry["registry_digest"] = canonical_digest(registry, digest_field="registry_digest")
    path = tmp_path / "native-registry.json"
    path.write_text(json.dumps(registry))
    path.chmod(0o600)
    monkeypatch.setenv(native.REGISTRY_ENV, str(path))
    assert executor.validate_queue_run(row, capture_root=tmp_path) == []
    registry["profiles"][0]["allowed_checkpoint_ids"] = []
    registry["registry_digest"] = canonical_digest(registry, digest_field="registry_digest")
    path.write_text(json.dumps(registry))
    assert executor.validate_queue_run(row, capture_root=tmp_path) == ["agent_execution_native_profile_missing"]
    client = FakeClient([row])
    summary = executor.poll_once(client=client, capture_root=tmp_path)
    assert summary["claimed"] == 1 and summary["blocked"] == 1 and summary["staged"] == 0
    assert client.blocked == ["agent_execution_native_profile_missing"]


@pytest.mark.parametrize("state,transport_failure", [(None, False), ("claim_intent", False),
    ("staged", False), ("native_execution_in_progress", False), (None, True)])
def test_native_credential_request_uses_confirmed_journal_owner_on_restart(
    tmp_path: Path, monkeypatch, state, transport_failure, capsys, caplog,
) -> None:
    from blueprint_pipeline import controlled_native_queue as native
    from blueprint_pipeline import adp_task_evaluation_abstention as abstention
    from blueprint_pipeline import controlled_policy_session as session
    from blueprint_pipeline.safe_outbound_http import SafeHttpResponse

    row = _row(tmp_path)
    canonical = row["execution_admission"]["canonical_execution_request"]
    reference = "policy-credential-00000000-0000-0000-0000-000000000001"
    canonical["policy_package"]["policy_api_endpoint"].update(
        execution_profile="controlled_observation_v1", endpoint_url="https://policy.example/action",
        credential_ref=reference, credential_kind="bearer")
    canonical["execution_authorization"]["episodes"] = row["quoted_episodes"] = 1
    frozen = json.dumps(row["execution_admission"], sort_keys=True, separators=(",", ":"))
    row["execution_admission_canonical_json"] = frozen
    row["execution_admission_digest"] = "sha256:" + hashlib.sha256(frozen.encode()).hexdigest()
    key = tmp_path / "synthetic-sync-key"
    token = "synthetic-offline-worker-key-" + "x" * 32
    secret = "synthetic-private-bearer-do-not-log"
    key.write_text(token)
    key.chmod(0o600)
    monkeypatch.setenv("BLUEPRINT_PIPELINE_SYNC_TOKEN_FILE", str(key))
    monkeypatch.setenv("BLUEPRINT_WEBAPP_URL", "https://webapp.example")
    config = tmp_path / "synthetic-configuration.json"
    config.write_text(json.dumps({"contract": {}, "allowed_origins": ["https://policy.example"]}))
    monkeypatch.setattr(native, "configured_profile", lambda _request: {
        "configuration_path": str(config), "allowed_modalities": ["policy_api_endpoint"],
        "hard_cap_usd": 1, "task_id": "pick_place", "scenario_id": "nominal",
        "packet_dir": str(tmp_path), "runtime_source_packet_receipt": str(tmp_path / "unused.json")})
    monkeypatch.setattr(native, "validate_native_configuration", lambda value: value)
    monkeypatch.setattr(session, "customer_hosted_client", lambda **_kwargs: None)
    monkeypatch.setattr(abstention, "collect_vast_provider_zero_receipt", lambda: {"provider_zero": True})

    class OwnerClient(FakeClient):
        def report_blocked(self, _row, pipeline_run_id, reason):
            self.blocked.append((pipeline_run_id, reason))

    client = OwnerClient([row] if state is None else [])
    owner = "agent-attempt-" + "a" * 32
    journals = tmp_path / "pipeline/agent_run_executor/journal"
    if state is not None:
        journal = {"state": state, "run_id": row["run_id"], "pipeline_run_id": owner,
            "canonical_job_id": canonical["job_id"], "execution_admission_digest": row["execution_admission_digest"],
            "disposition": "stage_for_execution", "row": row}
        executor._write_json_atomic(journals / "restart.json", journal)

    received = []

    def signed_credential_request(url, *, data, headers, **_kwargs):
        nonlocal owner
        if state is None:
            owner = client.claims[-1][1]
        body = json.loads(data)
        assert body == {"action": "access", "job_id": canonical["job_id"],
            "pipeline_run_id": owner, "canonical_request_digest": executor._digest(canonical)}
        assert body["canonical_request_digest"] != row["execution_admission_digest"]
        expected = hmac.new(token.encode(), headers["X-Blueprint-Pipeline-Timestamp"].encode() + b"." + data,
            hashlib.sha256).hexdigest()
        assert headers["X-Blueprint-Pipeline-Signature"] == "sha256=" + expected
        received.append(body)
        if transport_failure:
            raise urllib.error.URLError(secret)
        response = {**body, "ok": True, "credential_ref": reference, "kind": "bearer",
            "credential": {"job_id": canonical["job_id"], "endpoint_url": "https://policy.example/action",
                "bearer_token": secret}}
        return SafeHttpResponse(status=200, body=json.dumps(response).encode(), url=url, final_url=url)

    def stop_before_paid_admission(**kwargs):
        assert kwargs["policy_credential"]["bearer_token"] == secret
        raise ValueError("synthetic_stop_before_paid_admission")

    monkeypatch.setattr(executor, "safe_request", signed_credential_request)
    monkeypatch.setattr(native, "build_controlled_native_policy_bundle", stop_before_paid_admission)
    summary = executor.poll_once(client=client, capture_root=tmp_path)
    assert summary["blocked"] == 1 and len(received) == 1
    reason = ("checkpoint_policy_credential_delivery_unavailable" if transport_failure
        else "synthetic_stop_before_paid_admission")
    assert client.blocked == [(owner, reason)]
    assert all(claimed_owner == owner for _, claimed_owner in client.claims)
    assert executor._digest(canonical) == received[0]["canonical_request_digest"]
    job_dir = tmp_path / "pipeline/robot_eval_jobs" / canonical["job_id"]
    assert not (job_dir / "native_authority.json").exists()
    assert not (job_dir / "native_allocator.log").exists()
    assert secret not in "".join(path.read_text() for path in job_dir.glob("*.json"))
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err + caplog.text


@pytest.mark.parametrize("accepted_owner", [None, "another-owner"])
def test_missing_or_mismatched_start_owner_never_enters_native_execution(
    tmp_path: Path, monkeypatch, accepted_owner,
) -> None:
    from blueprint_pipeline import controlled_native_queue as native

    row = _row(tmp_path)
    canonical = row["execution_admission"]["canonical_execution_request"]
    canonical["policy_package"]["policy_api_endpoint"]["execution_profile"] = "controlled_observation_v1"
    canonical["execution_authorization"]["episodes"] = row["quoted_episodes"] = 1
    frozen = json.dumps(row["execution_admission"], sort_keys=True, separators=(",", ":"))
    row["execution_admission_canonical_json"] = frozen
    row["execution_admission_digest"] = "sha256:" + hashlib.sha256(frozen.encode()).hexdigest()
    monkeypatch.setattr(native, "configured_profile", lambda _request: {})
    monkeypatch.setattr(native, "execute_staged_controlled_request",
        lambda **_kwargs: pytest.fail("unconfirmed claim entered native execution"))
    client = FakeClient([row])
    monkeypatch.setattr(client, "claim", lambda *_args: {"pipeline_run_id": accepted_owner})
    summary = executor.poll_once(client=client, capture_root=tmp_path)
    assert summary["blocked"] == 1 and summary["claimed"] == summary["staged"] == 0
    assert not list((tmp_path / "pipeline/robot_eval_job_requests").glob("**/*.json"))


@pytest.mark.parametrize("mode", ["paid", "expired", "cancelled", "unavailable"])
@pytest.mark.parametrize("state", ["staged", "native_execution_in_progress"])
def test_restart_never_launches_after_authority_ends_but_still_reconciles(tmp_path, monkeypatch, mode, state):
    from blueprint_pipeline import controlled_native_queue as native
    row = _row(tmp_path)
    if mode == "paid":
        row["quoted_usd"] = 99
    if mode == "expired":
        row["execution_admission"]["funding"]["expires_at_iso"] = "2020-01-01T00:00:00Z"
    journal_dir = tmp_path / "journals"
    journal_dir.mkdir()
    (journal_dir / "run-1.json").write_text(json.dumps({
        "state": state, "run_id": "run-1", "pipeline_run_id": "attempt-1", "row": row,
        "canonical_job_id": "canonical-job-1", "capture_root": str(tmp_path),
    }))
    client = FakeClient([])
    def current(_run_id):
        if mode == "unavailable":
            raise OSError("transport unavailable")
        return {"state": "requested", "cancellation_requested": mode == "cancelled"}
    client.get_run = current
    monkeypatch.setattr(native, "routes_controlled_request", lambda _request: True)
    def forbidden(**_kwargs):
        pytest.fail("stale authority must not execute native work")
    monkeypatch.setattr(native, "execute_staged_controlled_request", forbidden)
    receipt = {"status": "completed", "episodes_run": 1, "episodes_succeeded": 0, "observed_cost_usd": 2}
    summary = executor.poll_once(client=client, capture_root=tmp_path, journal_dir=journal_dir,
        terminal_reader=lambda **_kwargs: receipt)
    assert summary["completed"] == 1
    assert client.completed == [receipt]
