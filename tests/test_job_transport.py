"""Job transport (C8): envelope, shadow publish, Cloud Tasks dispatch, retries.

Managed queues carry delivery only; Blueprint keeps job identity, idempotent
claims, terminal commits, spend authority, and independent watchdogs.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from blueprint_pipeline.job_transport_envelope import (
    JOB_ENVELOPE_SCHEMA_VERSION,
    JobEnvelopeCredentialError,
    build_job_envelope,
    validate_job_envelope,
)
from blueprint_pipeline.job_transport_shadow import (
    InMemoryEnvelopePublisher,
    compare_shadow_parity,
    record_shadow_delivery,
    shadow_publish_job_envelope,
)
from blueprint_pipeline.cloud_tasks_dispatch import (
    CLOUD_TASKS_ALLOWED_LANES,
    dispatch_job_envelope_task,
)
from blueprint_pipeline.transport_retry_policy import (
    MutationRetryForbidden,
    TransportRetryConfigError,
    bounded_read_retry,
    mutation_single_attempt,
    optional_circuit_breaker,
    reconcile_then_retry_mutation,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _request() -> dict:
    return {
        "schema_version": "robot_eval_job_request.v1",
        "job_id": "job-123",
        "operation": "evaluation_run",
        "site_package": {"capture_root": "/captures/x"},
    }


def _envelope() -> dict:
    return build_job_envelope(
        job_request=_request(),
        job_id="job-123",
        source_lane="fixture",
        created_at="2026-08-02T00:00:00+00:00",
    )


# ---------------------------------------------------------------- envelope


def test_envelope_identity_is_deterministic_and_content_bound() -> None:
    first = _envelope()
    second = build_job_envelope(
        job_request=_request(),
        job_id="job-123",
        source_lane="fixture",
        created_at="2026-08-02T09:00:00+00:00",
    )
    assert first["schema_version"] == JOB_ENVELOPE_SCHEMA_VERSION
    assert first["envelope_id"] == second["envelope_id"]
    assert first["payload_sha256"] == second["payload_sha256"]

    changed = dict(_request())
    changed["operation"] = "other"
    third = build_job_envelope(
        job_request=changed,
        job_id="job-123",
        source_lane="fixture",
        created_at="2026-08-02T00:00:00+00:00",
    )
    assert third["envelope_id"] != first["envelope_id"]
    assert first["execution_authority"] == "filesystem"
    assert validate_job_envelope(first) == []


def test_envelope_refuses_credential_shaped_content() -> None:
    tainted = _request()
    tainted["provider"] = {"api_key": "sk-should-never-ride-the-queue"}
    with pytest.raises(JobEnvelopeCredentialError):
        build_job_envelope(
            job_request=tainted,
            job_id="job-123",
            source_lane="fixture",
            created_at="2026-08-02T00:00:00+00:00",
        )


def test_envelope_validation_fails_closed() -> None:
    envelope = _envelope()
    envelope.pop("payload_sha256")
    envelope["schema_version"] = "blueprint.job_envelope.v999"
    blockers = validate_job_envelope(envelope)
    assert "job_envelope_schema_version_invalid" in blockers
    assert "job_envelope_field_missing:payload_sha256" in blockers


# ---------------------------------------------------------------- shadow


def test_shadow_publish_records_evidence_and_never_executes(tmp_path: Path) -> None:
    publisher = InMemoryEnvelopePublisher()
    record = shadow_publish_job_envelope(
        envelope=_envelope(),
        publisher=publisher,
        evidence_dir=tmp_path,
    )
    assert record["status"] == "published"
    assert record["execution_authority"] == "filesystem"
    assert len(publisher.published) == 1
    ledger = (tmp_path / "shadow_publish_ledger.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in ledger.splitlines()]
    assert rows[0]["envelope_id"] == _envelope()["envelope_id"]
    assert rows[0]["status"] == "published"


def test_shadow_publish_failure_is_contained(tmp_path: Path) -> None:
    class ExplodingPublisher:
        def publish(self, envelope):  # noqa: ANN001
            raise RuntimeError("broker down")

    record = shadow_publish_job_envelope(
        envelope=_envelope(),
        publisher=ExplodingPublisher(),
        evidence_dir=tmp_path,
    )
    assert record["status"] == "publish_failed"
    assert "RuntimeError" in record["error"]
    rows = (tmp_path / "shadow_publish_ledger.jsonl").read_text(encoding="utf-8")
    assert "publish_failed" in rows


def test_shadow_parity_reports_missing_and_duplicate_deliveries(tmp_path: Path) -> None:
    publisher = InMemoryEnvelopePublisher()
    envelope = _envelope()
    shadow_publish_job_envelope(envelope=envelope, publisher=publisher, evidence_dir=tmp_path)
    shadow_publish_job_envelope(envelope=envelope, publisher=publisher, evidence_dir=tmp_path)
    parity = compare_shadow_parity(tmp_path)
    assert parity["published_unique"] == 1
    assert parity["delivered_unique"] == 0
    assert parity["missing_delivery"] == [envelope["envelope_id"]]
    assert parity["status"] == "delivery_gap"

    record_shadow_delivery(
        envelope_id=envelope["envelope_id"], evidence_dir=tmp_path, consumer="fixture"
    )
    record_shadow_delivery(
        envelope_id=envelope["envelope_id"], evidence_dir=tmp_path, consumer="fixture"
    )
    parity = compare_shadow_parity(tmp_path)
    assert parity["delivered_unique"] == 1
    assert parity["duplicate_deliveries"] == [envelope["envelope_id"]]
    assert parity["missing_delivery"] == []
    assert parity["status"] == "parity_with_duplicates"


# ---------------------------------------------------------------- cloud tasks


class _FakeTasksClient:
    def __init__(self, *, raise_already_exists: bool = False) -> None:
        self.raise_already_exists = raise_already_exists
        self.created: list = []

    def queue_path(self, project: str, location: str, queue: str) -> str:
        return f"projects/{project}/locations/{location}/queues/{queue}"

    def create_task(self, request):  # noqa: ANN001
        if self.raise_already_exists:
            class AlreadyExists(Exception):
                pass

            raise AlreadyExists("task exists")
        self.created.append(request)

        class _Response:
            name = request["task"]["name"]

        return _Response()


def _tasks_env() -> dict[str, str]:
    return {
        "BLUEPRINT_JOB_TRANSPORT_TASKS_QUEUE": "pipeline-queue",
        "BLUEPRINT_JOB_TRANSPORT_TASKS_LOCATION": "us-central1",
        "BLUEPRINT_JOB_TRANSPORT_TASKS_URL": "https://dispatch.example/run",
        "GOOGLE_CLOUD_PROJECT": "blueprint-8c1ca",
    }


def test_cloud_tasks_task_name_derives_from_envelope_for_dedup() -> None:
    client = _FakeTasksClient()
    envelope = _envelope()
    result = dispatch_job_envelope_task(envelope=envelope, env=_tasks_env(), client=client)
    assert result["status"] == "dispatched"
    (request,) = client.created
    assert request["task"]["name"].endswith(f"/tasks/{envelope['envelope_id']}")
    assert request["task"]["http_request"]["url"] == "https://dispatch.example/run"
    body = json.loads(request["task"]["http_request"]["body"].decode("utf-8"))
    assert body["envelope_id"] == envelope["envelope_id"]


def test_cloud_tasks_duplicate_is_idempotent_success() -> None:
    result = dispatch_job_envelope_task(
        envelope=_envelope(),
        env=_tasks_env(),
        client=_FakeTasksClient(raise_already_exists=True),
    )
    assert result["status"] == "deduplicated"


def test_cloud_tasks_lane_allowlist_fails_closed() -> None:
    assert CLOUD_TASKS_ALLOWED_LANES == frozenset({"fixture"})
    envelope = build_job_envelope(
        job_request=_request(),
        job_id="job-123",
        source_lane="paid_gpu",
        created_at="2026-08-02T00:00:00+00:00",
    )
    result = dispatch_job_envelope_task(
        envelope=envelope, env=_tasks_env(), client=_FakeTasksClient()
    )
    assert result["status"] == "blocked_lane_not_allowlisted"


def test_cloud_tasks_missing_config_is_unavailable_not_partial() -> None:
    result = dispatch_job_envelope_task(
        envelope=_envelope(), env={}, client=_FakeTasksClient()
    )
    assert result["status"] == "unavailable"
    assert "cloud_tasks_queue_missing" in result["blockers"]
    assert "cloud_tasks_location_missing" in result["blockers"]
    assert "cloud_tasks_url_missing" in result["blockers"]


# ---------------------------------------------------------------- retry policy


def test_bounded_read_retry_retries_allowlisted_then_succeeds() -> None:
    attempts: list[int] = []
    evidence: list[dict] = []
    sleeps: list[float] = []

    @bounded_read_retry(
        operation="vast_list_instances",
        exception_allowlist=(ConnectionError,),
        max_attempts=4,
        max_delay_seconds=30.0,
        evidence_hook=evidence.append,
        sleep=sleeps.append,
    )
    def flaky() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("transient")
        return "ok"

    assert flaky() == "ok"
    assert len(attempts) == 3
    assert len(evidence) == 2
    assert all(row["operation"] == "vast_list_instances" for row in evidence)
    assert len(sleeps) == 2
    assert all(delay >= 0.0 for delay in sleeps)


def test_bounded_read_retry_does_not_retry_unlisted_exceptions() -> None:
    attempts: list[int] = []

    @bounded_read_retry(
        operation="read",
        exception_allowlist=(ConnectionError,),
        max_attempts=4,
        max_delay_seconds=5.0,
        evidence_hook=lambda row: None,
        sleep=lambda _s: None,
    )
    def broken() -> None:
        attempts.append(1)
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        broken()
    assert len(attempts) == 1


def test_bounded_read_retry_reraises_original_after_exhaustion() -> None:
    @bounded_read_retry(
        operation="read",
        exception_allowlist=(ConnectionError,),
        max_attempts=2,
        max_delay_seconds=5.0,
        evidence_hook=lambda row: None,
        sleep=lambda _s: None,
    )
    def always_down() -> None:
        raise ConnectionError("still down")

    with pytest.raises(ConnectionError):
        always_down()


def test_mutation_single_attempt_never_retries() -> None:
    attempts: list[int] = []

    def create() -> None:
        attempts.append(1)
        raise ConnectionError("ambiguous after send")

    with pytest.raises(ConnectionError):
        mutation_single_attempt(operation="create_instance", evidence_hook=lambda row: None)(
            create
        )()
    assert len(attempts) == 1


def test_reconcile_then_retry_only_retries_when_absence_is_proven() -> None:
    mutations: list[int] = []

    def mutate() -> dict:
        mutations.append(1)
        if len(mutations) == 1:
            raise ConnectionError("response lost")
        return {"status": "created", "instance_id": "i-2"}

    found = reconcile_then_retry_mutation(
        operation="create_instance",
        mutate=lambda: (_ for _ in ()).throw(ConnectionError("response lost")),
        reconcile=lambda: {"exists": True, "instance_id": "i-1"},
        evidence_hook=lambda row: None,
    )
    assert found["status"] == "reconciled_existing"
    assert found["instance_id"] == "i-1"

    created = reconcile_then_retry_mutation(
        operation="create_instance",
        mutate=mutate,
        reconcile=lambda: {"exists": False},
        evidence_hook=lambda row: None,
    )
    assert created["status"] == "created"
    assert len(mutations) == 2


def test_reconcile_unproven_absence_refuses_retry() -> None:
    with pytest.raises(MutationRetryForbidden):
        reconcile_then_retry_mutation(
            operation="create_instance",
            mutate=lambda: (_ for _ in ()).throw(ConnectionError("lost")),
            reconcile=lambda: {"exists": None},
            evidence_hook=lambda row: None,
        )


def test_optional_circuit_breaker_requires_pybreaker() -> None:
    with pytest.raises(TransportRetryConfigError, match="pybreaker_not_installed"):
        optional_circuit_breaker(name="vast")


# ---------------------------------------------------------------- wiring


def test_orchestrator_shadow_hook_is_gated_and_contained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from blueprint_pipeline import robot_eval_job_orchestrator as orchestrator

    disabled = orchestrator._maybe_shadow_publish_job_transport(
        request=_request(), job_id="job-123", queue_root=tmp_path
    )
    assert disabled["status"] == "disabled"

    monkeypatch.setenv("BLUEPRINT_JOB_TRANSPORT_SHADOW", "1")
    published = orchestrator._maybe_shadow_publish_job_transport(
        request=_request(), job_id="job-123", queue_root=tmp_path
    )
    assert published["status"] == "published"
    assert (tmp_path / ".transport_shadow" / "shadow_publish_ledger.jsonl").is_file()

    tainted = _request()
    tainted["auth_token"] = "never"
    contained = orchestrator._maybe_shadow_publish_job_transport(
        request=tainted, job_id="job-123", queue_root=tmp_path
    )
    assert contained["status"] == "shadow_failed"


def test_watchdogs_remain_transport_independent() -> None:
    watchdog_paths = (
        REPO_ROOT / "src/blueprint_pipeline/paid_lane_guard.py",
        REPO_ROOT / "scripts/gpu_spend_guard.py",
        REPO_ROOT / "src/blueprint_pipeline/isaac_particlefield_render_job.py",
    )
    forbidden_roots = {"job_transport_shadow", "job_transport_envelope", "cloud_tasks_dispatch"}
    for path in watchdog_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert "pubsub" not in module, f"{path.name} imports pubsub"
                assert not (
                    set(module.split(".")) & forbidden_roots
                ), f"{path.name} imports job transport"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert "pubsub" not in alias.name, f"{path.name} imports pubsub"
