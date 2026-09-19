"""Bounded WebApp agent-run consumer using the canonical Evaluation Run path.

The worker is inert until invoked. It accepts only immutable, digest-bound
``blueprint.agent_execution_admission.v1`` envelopes and never manufactures a
task, checkpoint, capture, rights grant, budget, or testbed binding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from .robot_eval_evaluation_run_adapter import (
    execute_robot_eval_request_as_evaluation_run,
)
from .safe_outbound_http import (
    loopback_service_policy,
    pinned_api_policy,
    request as safe_request,
)
from .task_evaluation_launch_webapp_sync import load_pipeline_sync_token
from .webapp_sync import _pipeline_sync_headers

ADMISSION_SCHEMA_VERSION = "blueprint.agent_execution_admission.v1"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def validate_queue_run(row: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    admission = _mapping(row.get("execution_admission"))
    binding = _mapping(admission.get("binding"))
    request = _mapping(admission.get("decision_request"))
    canonical = _mapping(admission.get("canonical_execution_request"))
    if admission.get("schema_version") != ADMISSION_SCHEMA_VERSION:
        blockers.append("agent_execution_admission_schema_invalid")
    if row.get("execution_admission_digest") != _digest(admission):
        blockers.append("agent_execution_admission_digest_mismatch")
    exact = {
        "team_id": row.get("team_id"),
        "checkpoint_id": _mapping(row.get("checkpoint")).get("checkpoint_id"),
        "scene_request_id": _mapping(row.get("scene")).get("request_id"),
        "task_family": row.get("task_family"),
    }
    for field, expected in exact.items():
        if not expected or binding.get(field) != expected:
            blockers.append(f"agent_execution_binding_mismatch:{field}")
    if request.get("request_id") != admission.get("source_request_id"):
        blockers.append("agent_execution_source_request_mismatch")
    if canonical.get("schema_version") != "robot_eval_job_request.v1":
        blockers.append("agent_execution_canonical_request_missing")
    if _mapping(canonical.get("customer")).get("id") != row.get("team_id"):
        blockers.append("agent_execution_canonical_team_mismatch")
    if _mapping(canonical.get("site_package")).get("capture_id") != binding.get("capture_id"):
        blockers.append("agent_execution_canonical_capture_mismatch")
    proof = _mapping(admission.get("proof_boundary"))
    if proof.get("provider_spend_authorized") is not False:
        blockers.append("agent_execution_webapp_spend_boundary_invalid")
    if row.get("dispatch") is not None:
        blockers.append("agent_execution_run_already_claimed")
    return blockers


class AgentRunWebAppClient:
    def __init__(self, *, base_url: str, token: str, timeout_seconds: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        parsed = urlsplit(self.base_url)
        self.policy = (
            loopback_service_policy(max_response_bytes=2_000_000)
            if parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            else pinned_api_policy(self.base_url, max_response_bytes=2_000_000)
        )

    def _json(
        self, path: str, *, method: str = "GET", payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        body = _canonical_json(payload if payload is not None else {})
        response = safe_request(
            self.base_url + path,
            method=method,
            data=body if method != "GET" else None,
            headers=_pipeline_sync_headers(self.token, body),
            timeout_seconds=self.timeout_seconds,
            policy=self.policy,
            max_response_bytes=2_000_000,
        )
        decoded = json.loads(response.body.decode("utf-8")) if response.body else {}
        if not isinstance(decoded, Mapping):
            raise ValueError("agent_execution_webapp_response_not_object")
        return dict(decoded)

    def list_runs(self, limit: int) -> list[dict[str, Any]]:
        payload = self._json(f"/api/internal/pipeline/agent-runs?limit={max(1, min(limit, 200))}")
        rows = payload.get("runs")
        if not isinstance(rows, list):
            raise ValueError("agent_execution_queue_response_invalid")
        return [dict(row) for row in rows if isinstance(row, Mapping)]

    def claim(self, run_id: str, pipeline_run_id: str, admission_digest: str) -> dict[str, Any]:
        return self._json(
            f"/api/internal/pipeline/agent-runs/{run_id}/started",
            method="POST",
            payload={
                "pipeline_run_id": pipeline_run_id,
                "execution_admission_digest": admission_digest,
            },
        )

    def report_blocked(self, row: Mapping[str, Any], pipeline_run_id: str, reason: str) -> None:
        self._json(
            "/api/internal/pipeline/agent-run-settlements",
            method="POST",
            payload={
                "team_id": row["team_id"],
                "reservation_id": row["reservation_id"],
                "run_id": pipeline_run_id,
                "pipeline_run_id": pipeline_run_id,
                "execution_admission_digest": row["execution_admission_digest"],
                "episodes_run": 0,
                "rate_usd": 0,
                "blocked_before_any_episode": True,
                "reason": reason[:400],
            },
        )

    def report_completed(
        self,
        row: Mapping[str, Any],
        pipeline_run_id: str,
        receipt: Mapping[str, Any],
    ) -> None:
        episodes = int(receipt["episodes_run"])
        self._json(
            "/api/internal/pipeline/agent-run-results",
            method="POST",
            payload={
                "reservation_id": row["reservation_id"],
                "pipeline_run_id": pipeline_run_id,
                "execution_admission_digest": row["execution_admission_digest"],
                "episodes_run": episodes,
                "episodes_succeeded": int(receipt["episodes_succeeded"]),
                "median_cycle_seconds": receipt.get("median_cycle_seconds"),
                "cycle_seconds_p10": receipt.get("cycle_seconds_p10"),
                "cycle_seconds_p90": receipt.get("cycle_seconds_p90"),
                "note": receipt.get("note"),
                "artifact_uri": receipt.get("artifact_uri"),
            },
        )
        rate = float(row["quoted_usd"]) / max(1, int(row["quoted_episodes"]))
        self._json(
            "/api/internal/pipeline/agent-run-settlements",
            method="POST",
            payload={
                "team_id": row["team_id"],
                "reservation_id": row["reservation_id"],
                "run_id": pipeline_run_id,
                "pipeline_run_id": pipeline_run_id,
                "execution_admission_digest": row["execution_admission_digest"],
                "episodes_run": episodes,
                "rate_usd": rate,
                "reason": "Digest-bound canonical episode receipt",
            },
        )


def _pipeline_run_id(row: Mapping[str, Any]) -> str:
    canonical = _mapping(
        _mapping(row.get("execution_admission")).get("canonical_execution_request")
    )
    job_id = str(canonical.get("job_id") or "").strip()
    if not job_id:
        raise ValueError("agent_execution_canonical_job_id_missing")
    return job_id


def execute_one(
    *,
    row: Mapping[str, Any],
    capture_root: Path,
    executor: Callable[..., Mapping[str, Any]] = execute_robot_eval_request_as_evaluation_run,
) -> Mapping[str, Any]:
    admission = _mapping(row.get("execution_admission"))
    translated = _mapping(admission.get("canonical_execution_request"))
    required = ("customer", "site_package", "requested_tasks", "robot_profile", "policy_package")
    missing = [field for field in required if not translated.get(field)]
    if missing:
        raise ValueError("agent_execution_canonical_fields_missing:" + ",".join(missing))
    declared_root = str(_mapping(translated.get("site_package")).get("capture_root") or "")
    if not declared_root or Path(declared_root).resolve() != capture_root.resolve():
        raise ValueError("agent_execution_capture_root_mismatch")
    return executor(
        capture_root=capture_root,
        job_request=translated,
        job_id=_pipeline_run_id(row),
        provisioner="fixture_local",
        simulator="fixture",
        allow_wam_provider=False,
        allow_gpu_provisioning=False,
        allow_simulator_execution=False,
        allow_training=False,
        allow_policy_execution=False,
        allow_delivery_upload=False,
    )


def terminal_episode_receipt(
    row: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any] | None:
    receipt = _mapping(result.get("agent_run_episode_receipt"))
    if receipt.get("schema_version") != "blueprint.agent_run_episode_receipt.v1":
        return None
    if receipt.get("pipeline_run_id") != _pipeline_run_id(row):
        return None
    if receipt.get("execution_admission_digest") != row.get("execution_admission_digest"):
        return None
    episodes = receipt.get("episodes_run")
    successes = receipt.get("episodes_succeeded")
    if (
        not isinstance(episodes, int)
        or isinstance(episodes, bool)
        or episodes <= 0
        or not isinstance(successes, int)
        or isinstance(successes, bool)
        or successes < 0
        or successes > episodes
        or not receipt.get("artifact_uri")
        or receipt.get("receipt_sha256")
        != _digest({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    ):
        return None
    return receipt


def poll_once(
    *,
    client: AgentRunWebAppClient,
    capture_root: Path,
    limit: int = 10,
    executor: Callable[..., Mapping[str, Any]] = execute_robot_eval_request_as_evaluation_run,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"examined": 0, "claimed": 0, "blocked": 0, "completed": 0}
    for row in client.list_runs(limit):
        summary["examined"] += 1
        blockers = validate_queue_run(row)
        if blockers:
            summary["blocked"] += 1
            continue
        pipeline_run_id = _pipeline_run_id(row)
        claim = client.claim(
            str(row["run_id"]),
            pipeline_run_id,
            str(row["execution_admission_digest"]),
        )
        if claim.get("pipeline_run_id") != pipeline_run_id:
            summary["blocked"] += 1
            continue
        summary["claimed"] += 1
        try:
            result = execute_one(row=row, capture_root=capture_root, executor=executor)
        except Exception as exc:  # terminal no-episode closeout is safe and retry-idempotent
            client.report_blocked(row, pipeline_run_id, f"executor_failed:{type(exc).__name__}")
            summary["blocked"] += 1
            continue
        # Canonical execution may produce a structured blocked result. Billing
        # and measured-result publication require an explicit episode receipt;
        # a status string alone can never fabricate episodes or success.
        receipt = terminal_episode_receipt(row, result)
        if receipt is None:
            client.report_blocked(row, pipeline_run_id, "canonical_executor_reported_no_episodes")
            summary["blocked"] += 1
            continue
        client.report_completed(row, pipeline_run_id, receipt)
        summary["completed"] += 1
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--webapp-url", required=True)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--token-file", required=True, type=Path)
    args = parser.parse_args()
    token = load_pipeline_sync_token(token_file_path=args.token_file, require_file=True)
    client = AgentRunWebAppClient(base_url=args.webapp_url, token=token)
    print(
        json.dumps(
            poll_once(client=client, capture_root=args.capture_root, limit=args.limit),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
