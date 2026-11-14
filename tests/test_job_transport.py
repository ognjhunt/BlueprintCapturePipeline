"""Job transport (C8): envelope, shadow publish, Cloud Tasks dispatch, retries.

Managed queues carry delivery only; Blueprint keeps job identity, idempotent
claims, terminal commits, spend authority, and independent watchdogs.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

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


# ---------------------------------------------------------------- envelope


# ---------------------------------------------------------------- shadow


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
