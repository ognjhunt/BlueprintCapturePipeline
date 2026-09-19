"""Bounded WebApp agent-run consumer using the canonical Evaluation Run path.

The worker is inert until invoked. It accepts only immutable, digest-bound
``blueprint.agent_execution_admission.v1`` envelopes and never manufactures a
task, checkpoint, capture, rights grant, budget, or testbed binding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping
import urllib.error
from urllib.parse import quote, urlsplit
from uuid import uuid4

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


def validate_queue_run(row: Mapping[str, Any], *, capture_root: Path | None = None) -> list[str]:
    blockers: list[str] = []
    admission = _mapping(row.get("execution_admission"))
    binding = _mapping(admission.get("binding"))
    request = _mapping(admission.get("decision_request"))
    canonical = _mapping(admission.get("canonical_execution_request"))
    authorization = _mapping(canonical.get("execution_authorization"))
    if admission.get("schema_version") != ADMISSION_SCHEMA_VERSION:
        blockers.append("agent_execution_admission_schema_invalid")
    frozen = row.get("execution_admission_canonical_json")
    if not isinstance(frozen, str):
        blockers.append("agent_execution_admission_canonical_json_missing")
    else:
        try:
            parsed_frozen = json.loads(frozen)
        except json.JSONDecodeError:
            parsed_frozen = None
            blockers.append("agent_execution_admission_canonical_json_invalid")
        if parsed_frozen != admission:
            blockers.append("agent_execution_admission_canonical_json_mismatch")
    frozen_digest = (
        "sha256:" + hashlib.sha256(frozen.encode("utf-8")).hexdigest()
        if isinstance(frozen, str)
        else None
    )
    if row.get("execution_admission_digest") != frozen_digest:
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
    if capture_root is not None:
        configured_root = str(capture_root.resolve())
        request_root = str(_mapping(canonical.get("site_package")).get("capture_root") or "")
        if request_root != configured_root or canonical.get("capture_root") != configured_root:
            blockers.append("agent_execution_capture_root_partition_mismatch")
        episode_specs_path = capture_root / "pipeline" / "simulation_automation" / "episode_specs.json"
        if not episode_specs_path.is_file():
            blockers.append("agent_execution_episode_specs_missing")
        else:
            try:
                episode_specs = _load_json(episode_specs_path)
            except (OSError, ValueError):
                blockers.append("agent_execution_episode_specs_invalid")
            else:
                if episode_specs.get("episode_count") != row.get("quoted_episodes"):
                    blockers.append("agent_execution_episode_spec_count_mismatch")
        job_id = str(canonical.get("job_id") or "")
        if Path(job_id).name != job_id or job_id in {"", ".", ".."}:
            blockers.append("agent_execution_canonical_job_id_unsafe")
        else:
            staged_policy = (
                capture_root
                / "pipeline"
                / "robot_eval_inputs"
                / job_id
                / "policy_package.json"
            )
            if staged_policy.exists():
                blockers.append("agent_execution_unapproved_staged_policy_package")
    if authorization.get("episodes") != row.get("quoted_episodes"):
        blockers.append("agent_execution_episode_quote_mismatch")
    if authorization.get("max_cost_usd") != row.get("quoted_usd"):
        blockers.append("agent_execution_cost_quote_mismatch")
    tasks = canonical.get("requested_tasks")
    if not isinstance(tasks, list) or len(tasks) != 1:
        blockers.append("agent_execution_task_scope_unbounded")
    else:
        task = _mapping(tasks[0])
        scenarios = task.get("scenario_ids")
        if (
            authorization.get("task_id") != task.get("task_id")
            or not isinstance(scenarios, list)
            or len(scenarios) != 1
            or authorization.get("scenario_id") != scenarios[0]
        ):
            blockers.append("agent_execution_scenario_scope_mismatch")
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

    def list_runs(self, limit: int, *, capture_id: str | None = None) -> list[dict[str, Any]]:
        path = f"/api/internal/pipeline/agent-runs?limit={max(1, min(limit, 200))}"
        if capture_id:
            path += f"&capture_id={quote(capture_id, safe='')}"
        payload = self._json(path)
        rows = payload.get("runs")
        if not isinstance(rows, list):
            raise ValueError("agent_execution_queue_response_invalid")
        return [dict(row) for row in rows if isinstance(row, Mapping)]

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._json(f"/api/internal/pipeline/agent-runs/{run_id}")

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
        quoted_episodes = int(row["quoted_episodes"])
        quoted_usd = float(row["quoted_usd"])
        if episodes <= 0 or episodes > quoted_episodes or quoted_episodes <= 0:
            raise ValueError("agent_execution_terminal_episode_count_invalid")
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
        rate = quoted_usd / quoted_episodes
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
                "reason": "Purchased per-episode quote applied to observed canonical episodes",
            },
        )


def _canonical_job_id(row: Mapping[str, Any]) -> str:
    canonical = _mapping(
        _mapping(row.get("execution_admission")).get("canonical_execution_request")
    )
    job_id = str(canonical.get("job_id") or "").strip()
    if not job_id:
        raise ValueError("agent_execution_canonical_job_id_missing")
    return job_id


def _write_json_atomic(path: Path, value: Mapping[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(value)
    if exclusive and path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        os.write(descriptor, encoded)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        if exclusive:
            os.link(temporary_name, path)
            os.unlink(temporary_name)
        else:
            os.replace(temporary_name, path)
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("agent_execution_journal_not_object")
    return dict(value)


def _stage_canonical_request(row: Mapping[str, Any], inbox_dir: Path) -> Path:
    admission = _mapping(row.get("execution_admission"))
    canonical = _mapping(admission.get("canonical_execution_request"))
    required = ("customer", "site_package", "requested_tasks", "robot_profile", "policy_package")
    missing = [field for field in required if not canonical.get(field)]
    if missing:
        raise ValueError("agent_execution_canonical_fields_missing:" + ",".join(missing))
    job_id = _canonical_job_id(row)
    if Path(job_id).name != job_id or job_id in {".", ".."}:
        raise ValueError("agent_execution_canonical_job_id_unsafe")
    capture_root = Path(str(canonical.get("capture_root") or "")).resolve()
    staged_policy = capture_root / "pipeline" / "robot_eval_inputs" / job_id / "policy_package.json"
    if staged_policy.exists():
        raise ValueError("agent_execution_unapproved_staged_policy_package")
    path = inbox_dir / f"{job_id}.json"
    if path.exists():
        if _digest(_load_json(path)) != _digest(canonical):
            raise ValueError("agent_execution_inbox_identity_conflict")
        return path
    _write_json_atomic(path, canonical, exclusive=True)
    return path


def _default_terminal_reader(
    *, job_dir: Path, expected_job_id: str, expected_canonical_request_digest: str
) -> Mapping[str, Any]:
    from .robot_eval_terminal_artifact_adapter import read_terminal_robot_eval_artifacts

    observed = read_terminal_robot_eval_artifacts(job_dir)
    if observed.get("status") == "blocked":
        blockers = list(observed.get("blockers") or [])
        manifest_status = str(observed.get("job_manifest_status") or "")
        if (
            all(str(item).endswith("_missing") for item in blockers)
            or manifest_status not in {"completed", "failed", "blocked", "cancelled"}
            or set(map(str, blockers)).issubset(
                {"actual_gpu_time_not_observed", "actual_cost_usd_not_observed"}
            )
        ):
            return {"status": "pending", "blockers": blockers}
        return observed
    blockers: list[str] = []
    if observed.get("job_id") != expected_job_id:
        blockers.append("terminal_canonical_job_id_mismatch")
    if observed.get("canonical_job_request_digest") != expected_canonical_request_digest:
        blockers.append("terminal_canonical_job_request_mismatch")
    if blockers:
        return {"status": "blocked", "blockers": blockers}
    episode = _mapping(observed.get("episode_result"))
    return {
        "status": "completed",
        "episodes_run": episode.get("episodes_run"),
        "episodes_succeeded": episode.get("episodes_succeeded"),
        "median_cycle_seconds": None,
        "cycle_seconds_p10": None,
        "cycle_seconds_p90": None,
        "note": "Observed canonical terminal robot-eval artifacts",
        "artifact_uri": episode.get("metrics_path"),
        "observed_cost_usd": _mapping(observed.get("cost")).get("observed_cost_usd"),
    }


def poll_once(
    *,
    client: AgentRunWebAppClient,
    capture_root: Path,
    capture_id: str | None = None,
    limit: int = 10,
    inbox_dir: Path | None = None,
    journal_dir: Path | None = None,
    terminal_reader: Callable[..., Mapping[str, Any]] = _default_terminal_reader,
) -> dict[str, Any]:
    pipeline_root = capture_root / "pipeline"
    inbox = inbox_dir or pipeline_root / "robot_eval_job_requests" / "inbox"
    journals = journal_dir or pipeline_root / "agent_run_executor" / "journal"
    jobs_root = pipeline_root / "robot_eval_jobs"
    journals.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "examined": 0,
        "claimed": 0,
        "staged": 0,
        "pending": 0,
        "blocked": 0,
        "completed": 0,
    }
    # Reconcile intents first. A crash may happen after the server commits the
    # claim but before this process records the response. The durable unique
    # attempt id is therefore looked up and reused; a second owner is never
    # created for that run.
    for journal_path in sorted(journals.glob("*.json")):
        journal = _load_json(journal_path)
        if journal.get("state") != "claim_intent":
            continue
        row = _mapping(journal.get("row"))
        run_id = str(journal.get("run_id") or "")
        pipeline_run_id = str(journal.get("pipeline_run_id") or "")
        try:
            observed = client.get_run(run_id)
        except (OSError, ValueError, urllib.error.URLError):
            summary["pending"] += 1
            continue
        observed_state = str(observed.get("state") or "")
        if observed.get("money_resolved") is True or observed_state != "requested":
            journal["state"] = "claim_rejected"
            journal["rejection"] = "server_run_not_executable"
            _write_json_atomic(journal_path, journal)
            summary["blocked"] += 1
            continue
        dispatch = _mapping(observed.get("dispatch"))
        observed_owner = str(dispatch.get("pipeline_run_id") or "")
        if observed_owner and observed_owner != pipeline_run_id:
            journal["state"] = "claim_rejected"
            journal["rejection"] = "server_dispatch_owner_conflict"
            _write_json_atomic(journal_path, journal)
            summary["blocked"] += 1
            continue
        try:
            claim = client.claim(
                run_id,
                pipeline_run_id,
                str(journal["execution_admission_digest"]),
            )
        except urllib.error.HTTPError as exc:
            if exc.code == 409:
                journal["state"] = "claim_rejected"
                journal["rejection"] = "server_claim_no_longer_valid"
                _write_json_atomic(journal_path, journal)
                summary["blocked"] += 1
            else:
                summary["pending"] += 1
            continue
        except (OSError, ValueError, urllib.error.URLError):
            summary["pending"] += 1
            continue
        observed_owner = str(claim.get("pipeline_run_id") or "")
        if observed_owner != pipeline_run_id:
            journal["state"] = "claim_rejected"
            _write_json_atomic(journal_path, journal)
            summary["blocked"] += 1
            continue
        if journal.get("disposition") == "block_before_execution":
            try:
                client.report_blocked(
                    row,
                    pipeline_run_id,
                    ";".join(str(item) for item in journal.get("preflight_blockers", [])),
                )
            except (OSError, ValueError, urllib.error.URLError):
                summary["pending"] += 1
                continue
            journal["state"] = "reported_blocked"
            _write_json_atomic(journal_path, journal)
            summary["claimed"] += 1
            summary["blocked"] += 1
            continue
        try:
            staged_path = _stage_canonical_request(row, inbox)
            journal["state"] = "staged"
            journal["staged_request_path"] = str(staged_path)
            _write_json_atomic(journal_path, journal)
            summary["claimed"] += 1
            summary["staged"] += 1
        except Exception as exc:
            journal["state"] = "claim_intent"
            journal["last_staging_error"] = f"staging_failed:{type(exc).__name__}"
            _write_json_atomic(journal_path, journal)
            summary["pending"] += 1
    runtime_preflight_blockers = {
        "agent_execution_episode_specs_missing",
        "agent_execution_episode_specs_invalid",
        "agent_execution_episode_spec_count_mismatch",
        "agent_execution_unapproved_staged_policy_package",
    }
    for row in client.list_runs(limit, capture_id=capture_id):
        summary["examined"] += 1
        blockers = validate_queue_run(row, capture_root=capture_root)
        if capture_id and _mapping(_mapping(row.get("execution_admission")).get("binding")).get(
            "capture_id"
        ) != capture_id:
            summary["blocked"] += 1
            continue
        contract_blockers = [item for item in blockers if item not in runtime_preflight_blockers]
        preflight_blockers = [item for item in blockers if item in runtime_preflight_blockers]
        if contract_blockers:
            summary["blocked"] += 1
            continue
        pipeline_run_id = f"agent-attempt-{uuid4().hex}"
        journal_path = journals / f"{row['run_id']}--{pipeline_run_id}.json"
        journal = {
            "schema_version": "blueprint.agent_run_executor_journal.v1",
            "state": "claim_intent",
            "run_id": row["run_id"],
            "pipeline_run_id": pipeline_run_id,
            "canonical_job_id": _canonical_job_id(row),
            "execution_admission_digest": row["execution_admission_digest"],
            "disposition": (
                "block_before_execution" if preflight_blockers else "stage_for_execution"
            ),
            "preflight_blockers": preflight_blockers,
            "row": dict(row),
        }
        _write_json_atomic(journal_path, journal, exclusive=True)
        claim = client.claim(
            str(row["run_id"]),
            pipeline_run_id,
            str(row["execution_admission_digest"]),
        )
        if claim.get("pipeline_run_id") != pipeline_run_id:
            journal["state"] = "claim_rejected"
            _write_json_atomic(journal_path, journal)
            summary["blocked"] += 1
            continue
        summary["claimed"] += 1
        if preflight_blockers:
            reason = ";".join(preflight_blockers)
            client.report_blocked(row, pipeline_run_id, reason)
            journal["state"] = "reported_blocked"
            journal["blocker"] = reason
            _write_json_atomic(journal_path, journal)
            summary["blocked"] += 1
            continue
        try:
            staged_path = _stage_canonical_request(row, inbox)
            journal["state"] = "staged"
            journal["staged_request_path"] = str(staged_path)
            _write_json_atomic(journal_path, journal)
            summary["staged"] += 1
        except Exception as exc:
            journal["state"] = "claim_intent"
            journal["last_staging_error"] = f"staging_failed:{type(exc).__name__}"
            _write_json_atomic(journal_path, journal)
            summary["pending"] += 1
    for journal_path in sorted(journals.glob("*.json")):
        journal = _load_json(journal_path)
        if journal.get("state") != "staged":
            continue
        row = _mapping(journal.get("row"))
        terminal = terminal_reader(
            job_dir=jobs_root / str(journal["canonical_job_id"]),
            expected_job_id=str(journal["canonical_job_id"]),
            expected_canonical_request_digest=_digest(
                _mapping(_mapping(row.get("execution_admission")).get("canonical_execution_request"))
            ),
        )
        status = terminal.get("status")
        if status in {"pending", "not_found"}:
            summary["pending"] += 1
            continue
        if status == "blocked":
            client.report_blocked(
                row,
                str(journal["pipeline_run_id"]),
                ";".join(str(item) for item in terminal.get("blockers", [])),
            )
            journal["state"] = "reported_blocked"
            summary["blocked"] += 1
        elif status == "completed":
            client.report_completed(row, str(journal["pipeline_run_id"]), terminal)
            journal["state"] = "reported_completed"
            summary["completed"] += 1
        else:
            summary["pending"] += 1
            continue
        _write_json_atomic(journal_path, journal)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--webapp-url", required=True)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--inbox-dir", type=Path)
    parser.add_argument("--journal-dir", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=0)
    args = parser.parse_args()
    token = load_pipeline_sync_token(token_file_path=args.token_file, require_file=True)
    client = AgentRunWebAppClient(base_url=args.webapp_url, token=token)
    while True:
        print(json.dumps(poll_once(
            client=client,
            capture_root=args.capture_root,
            capture_id=args.capture_id,
            limit=args.limit,
            inbox_dir=args.inbox_dir,
            journal_dir=args.journal_dir,
        ), sort_keys=True), flush=True)
        if args.poll_seconds <= 0:
            break
        time.sleep(max(1.0, args.poll_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
