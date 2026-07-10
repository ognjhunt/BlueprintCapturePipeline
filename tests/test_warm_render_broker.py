from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from blueprint_pipeline.warm_render_broker import (
    ClaimConflict,
    DurableWarmRenderQueue,
    IdempotencyConflict,
    create_warm_render_broker_app,
)


def test_durable_queue_uses_server_ids_and_idempotent_commits(tmp_path: Path) -> None:
    queue = DurableWarmRenderQueue(tmp_path / "broker" / "queue.sqlite3")
    first = queue.submit(
        scenario={"scenario_id": "open_fridge"},
        idempotency_key="buyer-job-1",
        client_request_label="../../must-never-be-a-path",
        session_nonce="session-1",
    )
    duplicate = queue.submit(
        scenario={"scenario_id": "open_fridge"},
        idempotency_key="buyer-job-1",
        client_request_label="../../must-never-be-a-path",
        session_nonce="session-1",
    )

    canonical_id = first["canonical_job_id"]
    assert canonical_id.startswith("wrj_")
    assert duplicate["canonical_job_id"] == canonical_id
    assert duplicate["idempotent_replay"] is True
    assert not (tmp_path / "must-never-be-a-path").exists()

    with pytest.raises(IdempotencyConflict):
        queue.submit(
            scenario={"scenario_id": "different"},
            idempotency_key="buyer-job-1",
            client_request_label="../../must-never-be-a-path",
            session_nonce="session-1",
        )

    claimed = queue.claim(worker_id="worker-1", lease_seconds=60)
    assert claimed is not None
    assert claimed["canonical_job_id"] == canonical_id
    assert claimed["client_request_label"] == "../../must-never-be-a-path"
    committed = queue.publish_result(
        canonical_job_id=canonical_id,
        claim_token=claimed["claim_token"],
        result={"status": "completed"},
    )
    replay = queue.publish_result(
        canonical_job_id=canonical_id,
        claim_token=claimed["claim_token"],
        result={"status": "completed"},
    )
    assert committed["idempotent_replay"] is False
    assert replay["idempotent_replay"] is True
    result = queue.get_result(
        canonical_job_id=canonical_id,
        session_nonce="session-1",
    )
    assert result is not None
    assert result["result"]["canonical_job_id"] == canonical_id

    with pytest.raises((ValueError, ClaimConflict)):
        queue.get_result(
            canonical_job_id="../../escaped.json",
            session_nonce="session-1",
        )


def test_concurrent_producers_and_claimers_lose_no_jobs(tmp_path: Path) -> None:
    queue = DurableWarmRenderQueue(tmp_path / "queue.sqlite3")

    def submit(index: int) -> str:
        return str(
            queue.submit(
                scenario={"index": index},
                idempotency_key=f"producer-job-{index}",
                client_request_label=f"client-label-{index}",
            )["canonical_job_id"]
        )

    with ThreadPoolExecutor(max_workers=12) as executor:
        submitted = list(executor.map(submit, range(64)))
    assert len(set(submitted)) == 64
    assert queue.counts() == {"queued": 64, "leased": 0, "completed": 0}

    def claim(index: int) -> str | None:
        claimed = queue.claim(worker_id=f"worker-{index}", lease_seconds=60)
        return str(claimed["canonical_job_id"]) if claimed is not None else None

    with ThreadPoolExecutor(max_workers=12) as executor:
        claimed_ids = list(executor.map(claim, range(64)))
    assert None not in claimed_ids
    assert set(claimed_ids) == set(submitted)
    assert queue.counts() == {"queued": 0, "leased": 64, "completed": 0}


def test_restart_reclaims_only_expired_lease(tmp_path: Path) -> None:
    now = {"value": 100.0}
    database = tmp_path / "queue.sqlite3"
    first_process = DurableWarmRenderQueue(database, clock=lambda: now["value"])
    submitted = first_process.submit(
        scenario={"scenario_id": "sink_faucet"},
        idempotency_key="restart-job",
    )
    first_claim = first_process.claim(worker_id="worker-before-restart", lease_seconds=10)
    assert first_claim is not None

    restarted = DurableWarmRenderQueue(database, clock=lambda: now["value"])
    assert restarted.claim(worker_id="worker-too-early", lease_seconds=10) is None
    now["value"] = 111.0
    recovered = restarted.claim(worker_id="worker-after-restart", lease_seconds=10)
    assert recovered is not None
    assert recovered["canonical_job_id"] == submitted["canonical_job_id"]
    assert recovered["attempt_count"] == 2

    with pytest.raises(ClaimConflict):
        restarted.publish_result(
            canonical_job_id=submitted["canonical_job_id"],
            claim_token=first_claim["claim_token"],
            result={"status": "stale"},
        )


def test_broker_http_contract_requires_auth_and_returns_server_id(tmp_path: Path) -> None:
    token = "t" * 64
    app = create_warm_render_broker_app(
        database_path=tmp_path / "queue.sqlite3",
        auth_token=token,
    )
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {token}"}

    assert client.post("/v1/warm-render/jobs", json={}).status_code == 401
    submitted = client.post(
        "/v1/warm-render/jobs",
        headers=headers,
        json={
            "scenario": {"scenario_id": "stovetop_knob"},
            "idempotency_key": "http-job-1",
            "client_request_label": "../../../crafted",
            "session_nonce": "http-session",
        },
    )
    assert submitted.status_code == 200
    canonical_id = submitted.json()["canonical_job_id"]
    assert canonical_id.startswith("wrj_")

    claimed = client.post(
        "/v1/warm-render/jobs/claim",
        headers=headers,
        json={"worker_id": "worker-http", "lease_seconds": 60},
    )
    assert claimed.status_code == 200
    assert claimed.json()["canonical_job_id"] == canonical_id
    committed = client.put(
        f"/v1/warm-render/jobs/{canonical_id}/result",
        headers=headers,
        json={
            "claim_token": claimed.json()["claim_token"],
            "result": {"status": "completed"},
        },
    )
    assert committed.status_code == 200
    fetched = client.get(
        f"/v1/warm-render/jobs/{canonical_id}/result",
        headers={**headers, "X-Warm-Session-Nonce": "http-session"},
    )
    assert fetched.status_code == 200
    assert fetched.json()["result"]["status"] == "completed"
    assert client.get(
        "/v1/warm-render/jobs/../../escaped/result",
        headers=headers,
    ).status_code in {404, 422}
    health = client.get("/healthz").json()
    assert "queue.sqlite3" not in str(health)
    assert "database" not in str(health).lower()
