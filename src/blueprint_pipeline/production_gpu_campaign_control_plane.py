"""Durable provider-neutral control plane for one GPU episode campaign.

This is the sole customer/campaign lifecycle authority.  It performs no cloud
provider mutation and accepts no provider credentials.  A separate autoscaler
may satisfy warm-capacity demand; workers report attempt and artifact evidence
back through this interface.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence


SCHEMA_VERSION = "production_gpu_campaign_control_plane.v1"
CAMPAIGN_SPEC_SCHEMA_VERSION = "production_gpu_campaign_spec.v1"
ATTEMPT_SCHEMA_VERSION = "production_gpu_episode_attempt.v1"
CUSTOMER_STATUS_SCHEMA_VERSION = "production_gpu_customer_status.v1"
ARTIFACT_SCHEMA_VERSION = "production_gpu_resumable_artifact.v1"
CAMPAIGN_TOKEN_FILE_ENV = "BLUEPRINT_PRODUCTION_GPU_CAMPAIGN_TOKEN_FILE"
_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}\Z")
_IMAGE = re.compile(r"\A[^\s@]+@sha256:[0-9a-f]{64}\Z")
_SHA = re.compile(r"\A[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")

ATTEMPT_KINDS = ("smoke", "episode")
ATTEMPT_STATES = (
    "planned",
    "waiting_for_worker",
    "running",
    "collecting",
    "validating",
    "passed",
    "failed",
    "timed_out",
    "cancelled",
)
TERMINAL_ATTEMPT_STATES = frozenset({"passed", "failed", "timed_out", "cancelled"})
CAMPAIGN_STATES = (
    "accepted",
    "queued",
    "running_smoke",
    "smoke_blocked",
    "running_episodes",
    "collecting",
    "validating",
    "teardown_pending",
    "completed",
    "failed",
    "cancelled",
)
TERMINAL_CAMPAIGN_STATES = frozenset({"smoke_blocked", "completed", "failed", "cancelled"})
ALLOWED_ATTEMPT_TRANSITIONS = {
    "planned": {"waiting_for_worker", "running", "cancelled"},
    "waiting_for_worker": {"running", "timed_out", "cancelled"},
    "running": {"collecting", "failed", "timed_out", "cancelled"},
    "collecting": {"validating", "failed", "timed_out"},
    "validating": TERMINAL_ATTEMPT_STATES,
}
REQUIRED_ARTIFACTS = (
    "worker_log",
    "simulator_log",
    "policy_log",
    "renderer_log",
    "action_trace",
    "evaluator_result",
    "frame_manifest",
    "review_video",
    "attempt_manifest",
)


class CampaignControlPlaneError(RuntimeError):
    pass


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _spec_digest(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key not in {"spec_digest", "status"}}
    return "sha256:" + hashlib.sha256(_canonical(payload).encode()).hexdigest()


def _identifier(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not _ID.fullmatch(text):
        raise ValueError(f"{field}_invalid")
    return text


def _safe_relative(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact_relative_path_invalid")
    return text


def build_campaign_spec(
    *,
    campaign_id: str,
    source_sha: str,
    release_candidate_fingerprint: str,
    worker_image_ref: str,
    scenario_id: str,
    task_id: str,
    policy_revision: str,
    model_asset_revisions: Mapping[str, str],
    smoke_seed: int = 1000,
    episode_seeds: Sequence[int] = (1001, 1002, 1003),
    smoke_timeout_seconds: int = 300,
    episode_timeout_seconds: int = 900,
    queue_timeout_seconds: int = 300,
    width: int = 640,
    height: int = 480,
) -> dict[str, Any]:
    blockers: list[str] = []
    try:
        campaign = _identifier(campaign_id, "campaign_id")
    except ValueError as exc:
        campaign = str(campaign_id or "")
        blockers.append(str(exc))
    if not _SHA.fullmatch(source_sha):
        blockers.append("source_sha_invalid")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", release_candidate_fingerprint):
        blockers.append("release_candidate_fingerprint_invalid")
    if not _IMAGE.fullmatch(worker_image_ref):
        blockers.append("worker_image_ref_must_be_digest_pinned")
    if not str(scenario_id).strip():
        blockers.append("scenario_id_missing")
    if not str(task_id).strip():
        blockers.append("task_id_missing")
    if not str(policy_revision).strip():
        blockers.append("policy_revision_missing")
    models = {str(k): str(v) for k, v in sorted(model_asset_revisions.items())}
    if not models or any(not value for value in models.values()):
        blockers.append("model_asset_revisions_missing")
    seeds = [int(value) for value in episode_seeds]
    if len(seeds) != 3 or len(set(seeds + [int(smoke_seed)])) != 4:
        blockers.append("one_smoke_and_three_unique_episode_seeds_required")
    if not 30 <= int(smoke_timeout_seconds) <= 300:
        blockers.append("smoke_timeout_out_of_range")
    if not 60 <= int(episode_timeout_seconds) <= 900:
        blockers.append("episode_timeout_out_of_range")
    if not 30 <= int(queue_timeout_seconds) <= 1800:
        blockers.append("queue_timeout_out_of_range")
    if int(width) < 640 or int(height) < 480:
        blockers.append("review_resolution_below_640x480")
    payload = {
        "schema_version": CAMPAIGN_SPEC_SCHEMA_VERSION,
        "campaign_id": campaign,
        "source_sha": source_sha,
        "release_candidate_fingerprint": release_candidate_fingerprint,
        "worker_image_ref": worker_image_ref,
        "scenario_id": str(scenario_id),
        "task_id": str(task_id),
        "policy_revision": str(policy_revision),
        "model_asset_revisions": models,
        "attempts": [{"attempt_id": "smoke", "kind": "smoke", "seed": int(smoke_seed)}]
        + [
            {"attempt_id": f"episode-{index}", "kind": "episode", "seed": seed}
            for index, seed in enumerate(seeds, start=1)
        ],
        "runtime": {
            "dynamic_episode_termination": True,
            "stop_immediately_on_declared_completion": True,
            "smoke_timeout_seconds": int(smoke_timeout_seconds),
            "episode_timeout_seconds": int(episode_timeout_seconds),
            "queue_timeout_seconds": int(queue_timeout_seconds),
            "review_width": int(width),
            "review_height": int(height),
            "fixed_frame_count": None,
        },
        "smoke_gate": {
            "full_episodes_may_start_only_after_smoke_passed": True,
            "requires_scene_loaded": True,
            "requires_simulator_steps": True,
            "requires_policy_actions": True,
            "requires_nonempty_learned_action_trace": True,
        },
        "blockers": blockers,
    }
    payload["spec_digest"] = _spec_digest(payload)
    payload["status"] = "valid" if not blockers else "blocked"
    return payload


class ProductionGpuCampaignControlPlane:
    def __init__(
        self,
        database_path: str | Path,
        artifact_root: str | Path,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript("""
            CREATE TABLE IF NOT EXISTS gpu_campaigns(
              campaign_id TEXT PRIMARY KEY, spec_json TEXT NOT NULL, spec_digest TEXT NOT NULL,
              state TEXT NOT NULL, blocker TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS gpu_attempts(
              campaign_id TEXT NOT NULL, attempt_id TEXT NOT NULL, kind TEXT NOT NULL, seed INTEGER NOT NULL,
              state TEXT NOT NULL, terminal_reason TEXT, semantic_task_success INTEGER,
              simulator_steps INTEGER, policy_actions INTEGER, started_at REAL, ended_at REAL,
              PRIMARY KEY(campaign_id, attempt_id));
            CREATE TABLE IF NOT EXISTS gpu_artifacts(
              campaign_id TEXT NOT NULL, attempt_id TEXT NOT NULL, artifact_id TEXT NOT NULL,
              relative_path TEXT NOT NULL, byte_size INTEGER NOT NULL, received_size INTEGER NOT NULL,
              expected_sha256 TEXT, sha256 TEXT, state TEXT NOT NULL, updated_at REAL NOT NULL,
              PRIMARY KEY(campaign_id, attempt_id, artifact_id));
            CREATE TABLE IF NOT EXISTS gpu_campaign_events(
              sequence INTEGER PRIMARY KEY AUTOINCREMENT, campaign_id TEXT NOT NULL, attempt_id TEXT,
              event TEXT NOT NULL, detail_json TEXT NOT NULL, created_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS gpu_campaign_finalization(
              campaign_id TEXT PRIMARY KEY, provider_result_json TEXT NOT NULL,
              teardown_proof_json TEXT NOT NULL, provider_result_sha256 TEXT NOT NULL,
              teardown_proof_sha256 TEXT NOT NULL, verified_at REAL NOT NULL);
            """)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def create_campaign(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(spec)
        if (
            payload.get("schema_version") != CAMPAIGN_SPEC_SCHEMA_VERSION
            or payload.get("status") != "valid"
        ):
            raise CampaignControlPlaneError("valid_campaign_spec_required")
        if payload.get("spec_digest") != _spec_digest(payload):
            raise CampaignControlPlaneError("campaign_spec_digest_mismatch")
        campaign = _identifier(payload.get("campaign_id"), "campaign_id")
        now = float(self._clock())
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO gpu_campaigns VALUES(?,?,?,'accepted',NULL,?,?)",
                (campaign, _canonical(payload), payload["spec_digest"], now, now),
            )
            for attempt in payload["attempts"]:
                connection.execute(
                    "INSERT INTO gpu_attempts(campaign_id,attempt_id,kind,seed,state) VALUES(?,?,?,?,?)",
                    (campaign, attempt["attempt_id"], attempt["kind"], attempt["seed"], "planned"),
                )
            self._event(
                connection,
                campaign,
                None,
                "campaign_created",
                {"spec_digest": payload["spec_digest"]},
            )
        return self.snapshot(campaign)

    def _event(
        self,
        connection: sqlite3.Connection,
        campaign: str,
        attempt: str | None,
        event: str,
        detail: Mapping[str, Any],
    ) -> None:
        connection.execute(
            "INSERT INTO gpu_campaign_events(campaign_id,attempt_id,event,detail_json,created_at) VALUES(?,?,?,?,?)",
            (campaign, attempt, event, _canonical(dict(detail)), float(self._clock())),
        )

    def transition_attempt(
        self,
        campaign_id: str,
        attempt_id: str,
        next_state: str,
        *,
        terminal_reason: str | None = None,
        semantic_task_success: bool | None = None,
        simulator_steps: int | None = None,
        policy_actions: int | None = None,
    ) -> dict[str, Any]:
        campaign = _identifier(campaign_id, "campaign_id")
        attempt = _identifier(attempt_id, "attempt_id")
        if next_state not in ATTEMPT_STATES:
            raise ValueError("attempt_state_invalid")
        now = float(self._clock())
        deadline_blocker: str | None = None
        with self._transaction() as connection:
            deadline_reasons = self._reconcile_deadlines_locked(connection, campaign, now)
            row = connection.execute(
                "SELECT * FROM gpu_attempts WHERE campaign_id=? AND attempt_id=?",
                (campaign, attempt),
            ).fetchone()
            if row is None:
                raise CampaignControlPlaneError("attempt_unknown")
            if attempt in deadline_reasons:
                deadline_blocker = f"attempt_deadline_enforced:{deadline_reasons[attempt]}"
            else:
                self._transition_attempt_locked(
                    connection,
                    campaign=campaign,
                    attempt=attempt,
                    row=row,
                    next_state=next_state,
                    terminal_reason=terminal_reason,
                    semantic_task_success=semantic_task_success,
                    simulator_steps=simulator_steps,
                    policy_actions=policy_actions,
                    now=now,
                )
        if deadline_blocker is not None:
            raise CampaignControlPlaneError(deadline_blocker)
        return self.snapshot(campaign)

    def _transition_attempt_locked(
        self,
        connection: sqlite3.Connection,
        *,
        campaign: str,
        attempt: str,
        row: sqlite3.Row,
        next_state: str,
        terminal_reason: str | None,
        semantic_task_success: bool | None,
        simulator_steps: int | None,
        policy_actions: int | None,
        now: float,
    ) -> None:
        current = str(row["state"])
        if next_state not in ALLOWED_ATTEMPT_TRANSITIONS.get(current, set()):
            raise CampaignControlPlaneError(f"attempt_transition_invalid:{current}->{next_state}")
        if row["kind"] == "episode":
            smoke = connection.execute(
                "SELECT state FROM gpu_attempts WHERE campaign_id=? AND kind='smoke'",
                (campaign,),
            ).fetchone()
            if next_state in {"running", "collecting", "validating", "passed"} and (
                smoke is None or smoke[0] != "passed"
            ):
                raise CampaignControlPlaneError("full_episode_blocked_until_smoke_passes")
        if next_state == "passed":
            missing = self._missing_artifacts(connection, campaign, attempt)
            if missing:
                raise CampaignControlPlaneError("attempt_artifacts_incomplete:" + ",".join(missing))
            minimum_activity = 3 if row["kind"] == "smoke" else 1
            if (
                int(simulator_steps if simulator_steps is not None else row["simulator_steps"] or 0)
                < minimum_activity
            ):
                raise CampaignControlPlaneError(
                    "smoke_simulator_steps_below_three"
                    if row["kind"] == "smoke"
                    else "simulator_steps_required_for_pass"
                )
            if (
                int(policy_actions if policy_actions is not None else row["policy_actions"] or 0)
                < minimum_activity
            ):
                raise CampaignControlPlaneError(
                    "smoke_policy_actions_below_three"
                    if row["kind"] == "smoke"
                    else "policy_actions_required_for_pass"
                )
        if next_state in TERMINAL_ATTEMPT_STATES and not str(terminal_reason or "").strip():
            raise CampaignControlPlaneError("terminal_reason_required")
        started = (
            now if next_state == "running" and row["started_at"] is None else row["started_at"]
        )
        ended = now if next_state in TERMINAL_ATTEMPT_STATES else None
        connection.execute(
            """UPDATE gpu_attempts SET state=?,terminal_reason=?,semantic_task_success=?,
                simulator_steps=COALESCE(?,simulator_steps),policy_actions=COALESCE(?,policy_actions),
                started_at=?,ended_at=COALESCE(?,ended_at) WHERE campaign_id=? AND attempt_id=?""",
            (
                next_state,
                terminal_reason,
                None if semantic_task_success is None else int(semantic_task_success),
                simulator_steps,
                policy_actions,
                started,
                ended,
                campaign,
                attempt,
            ),
        )
        if row["kind"] == "smoke" and next_state in {
            "failed",
            "timed_out",
            "cancelled",
        }:
            connection.execute(
                "UPDATE gpu_attempts SET state='cancelled',terminal_reason=?,ended_at=? "
                "WHERE campaign_id=? AND kind='episode' "
                "AND state IN ('planned','waiting_for_worker')",
                (terminal_reason, now, campaign),
            )
        self._derive_campaign_state(connection, campaign)
        self._event(
            connection,
            campaign,
            attempt,
            "attempt_transition",
            {"from": current, "to": next_state},
        )

    def _derive_campaign_state(self, connection: sqlite3.Connection, campaign: str) -> None:
        rows = connection.execute(
            "SELECT kind,state FROM gpu_attempts WHERE campaign_id=?", (campaign,)
        ).fetchall()
        smoke = next(str(row["state"]) for row in rows if row["kind"] == "smoke")
        episodes = [str(row["state"]) for row in rows if row["kind"] == "episode"]
        if smoke in {"failed", "timed_out", "cancelled"}:
            state = "smoke_blocked"
        elif smoke != "passed":
            state = "running_smoke" if smoke != "planned" else "queued"
        elif all(value in TERMINAL_ATTEMPT_STATES for value in episodes):
            if all(value == "passed" for value in episodes):
                finalized = connection.execute(
                    "SELECT 1 FROM gpu_campaign_finalization WHERE campaign_id=?",
                    (campaign,),
                ).fetchone()
                state = "completed" if finalized else "teardown_pending"
            else:
                state = "failed"
        elif any(value == "validating" for value in episodes):
            state = "validating"
        elif any(value == "collecting" for value in episodes):
            state = "collecting"
        elif any(value == "running" for value in episodes):
            state = "running_episodes"
        else:
            state = "running_episodes"
        connection.execute(
            "UPDATE gpu_campaigns SET state=?,updated_at=? WHERE campaign_id=?",
            (state, float(self._clock()), campaign),
        )

    def finalize_campaign(
        self,
        campaign_id: str,
        *,
        provider_result: Mapping[str, Any],
        teardown_proof: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Complete a passed campaign only after provider absence is proven."""

        campaign = _identifier(campaign_id, "campaign_id")
        provider_payload = dict(provider_result)
        teardown_payload = dict(teardown_proof)
        with self._connect() as connection:
            campaign_row = connection.execute(
                "SELECT spec_json FROM gpu_campaigns WHERE campaign_id=?",
                (campaign,),
            ).fetchone()
        if campaign_row is None:
            raise CampaignControlPlaneError("campaign_unknown")
        spec = json.loads(campaign_row["spec_json"])
        allocation_id = str(provider_payload.get("allocation_id") or "").strip()
        if provider_payload.get("schema_version") != "production_gpu_provider_result.v1":
            raise CampaignControlPlaneError("provider_result_schema_invalid")
        if provider_payload.get("status") != "provider_terminal":
            raise CampaignControlPlaneError("provider_result_not_terminal")
        if provider_payload.get("campaign_id") != campaign:
            raise CampaignControlPlaneError("provider_result_campaign_mismatch")
        if provider_payload.get("source_sha") != spec["source_sha"]:
            raise CampaignControlPlaneError("provider_result_source_mismatch")
        if provider_payload.get("worker_image_ref") != spec["worker_image_ref"]:
            raise CampaignControlPlaneError("provider_result_image_mismatch")
        if not allocation_id:
            raise CampaignControlPlaneError("provider_result_allocation_id_missing")
        if provider_payload.get("raw_secret_values_recorded") is not False:
            raise CampaignControlPlaneError("provider_result_secret_boundary_unverified")
        if teardown_payload.get("schema_version") != "production_gpu_teardown_proof.v1":
            raise CampaignControlPlaneError("teardown_proof_schema_invalid")
        if teardown_payload.get("status") != "PASS":
            raise CampaignControlPlaneError("teardown_proof_status_invalid")
        if teardown_payload.get("campaign_id") != campaign:
            raise CampaignControlPlaneError("teardown_proof_campaign_mismatch")
        if teardown_payload.get("allocation_id") != allocation_id:
            raise CampaignControlPlaneError("teardown_proof_allocation_mismatch")
        if teardown_payload.get("raw_secret_values_recorded") is not False:
            raise CampaignControlPlaneError("teardown_proof_secret_boundary_unverified")
        if teardown_payload.get("provider_absence_confirmed") is not True:
            raise CampaignControlPlaneError("provider_absence_proof_required")
        if teardown_payload.get("billing_stopped") is not True:
            raise CampaignControlPlaneError("billing_stop_proof_required")
        final_inventory = teardown_payload.get("final_inventory")
        final_inventory = final_inventory if isinstance(final_inventory, Mapping) else {}
        if not (
            final_inventory.get("api_confirmed") is True
            and final_inventory.get("live_resource_count") == 0
        ):
            raise CampaignControlPlaneError("provider_final_inventory_not_zero")
        now = float(self._clock())
        with self._transaction() as connection:
            states = connection.execute(
                "SELECT state FROM gpu_attempts WHERE campaign_id=?",
                (campaign,),
            ).fetchall()
            if len(states) != 4 or any(row[0] != "passed" for row in states):
                raise CampaignControlPlaneError("all_attempts_must_pass_before_finalization")
            provider_json = _canonical(provider_payload)
            teardown_json = _canonical(teardown_payload)
            existing = connection.execute(
                "SELECT provider_result_json,teardown_proof_json "
                "FROM gpu_campaign_finalization WHERE campaign_id=?",
                (campaign,),
            ).fetchone()
            inserted = existing is None
            if existing is not None:
                if existing[0] != provider_json or existing[1] != teardown_json:
                    raise CampaignControlPlaneError("campaign_finalization_evidence_conflict")
            else:
                connection.execute(
                    "INSERT INTO gpu_campaign_finalization VALUES(?,?,?,?,?,?)",
                    (
                        campaign,
                        provider_json,
                        teardown_json,
                        hashlib.sha256(provider_json.encode()).hexdigest(),
                        hashlib.sha256(teardown_json.encode()).hexdigest(),
                        now,
                    ),
                )
            self._derive_campaign_state(connection, campaign)
            if inserted:
                self._event(
                    connection,
                    campaign,
                    None,
                    "campaign_provider_teardown_verified",
                    {"provider_absence_confirmed": True},
                )
        return self.snapshot(campaign)

    def begin_artifact(
        self,
        campaign_id: str,
        attempt_id: str,
        artifact_id: str,
        *,
        relative_path: str,
        total_size: int,
        expected_sha256: str | None = None,
    ) -> dict[str, Any]:
        campaign, attempt, artifact = (
            _identifier(campaign_id, "campaign_id"),
            _identifier(attempt_id, "attempt_id"),
            _identifier(artifact_id, "artifact_id"),
        )
        relative = _safe_relative(relative_path)
        if int(total_size) <= 0:
            raise ValueError("artifact_total_size_invalid")
        if expected_sha256 and not _SHA256.fullmatch(expected_sha256):
            raise ValueError("artifact_sha256_invalid")
        path = self.artifact_root / campaign / attempt / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(path.suffix + ".part")
        received = part.stat().st_size if part.is_file() else 0
        with self._transaction() as connection:
            connection.execute(
                """INSERT INTO gpu_artifacts VALUES(?,?,?,?,?,?,?,NULL,'receiving',?)
                ON CONFLICT(campaign_id,attempt_id,artifact_id) DO UPDATE SET
                relative_path=excluded.relative_path,byte_size=excluded.byte_size,
                expected_sha256=excluded.expected_sha256,updated_at=excluded.updated_at""",
                (
                    campaign,
                    attempt,
                    artifact,
                    relative,
                    int(total_size),
                    received,
                    expected_sha256,
                    float(self._clock()),
                ),
            )
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_id": artifact,
            "next_offset": received,
        }

    def append_artifact_chunk(
        self,
        campaign_id: str,
        attempt_id: str,
        artifact_id: str,
        *,
        offset: int,
        data: bytes,
        chunk_sha256: str,
    ) -> dict[str, Any]:
        campaign, attempt, artifact = (
            _identifier(campaign_id, "campaign_id"),
            _identifier(attempt_id, "attempt_id"),
            _identifier(artifact_id, "artifact_id"),
        )
        if hashlib.sha256(data).hexdigest() != chunk_sha256:
            raise CampaignControlPlaneError("artifact_chunk_hash_mismatch")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM gpu_artifacts WHERE campaign_id=? AND attempt_id=? AND artifact_id=?",
                (campaign, attempt, artifact),
            ).fetchone()
            if row is None:
                raise CampaignControlPlaneError("artifact_unknown")
            path = self.artifact_root / campaign / attempt / row["relative_path"]
            part = path.with_suffix(path.suffix + ".part")
            current = part.stat().st_size if part.is_file() else 0
            if int(offset) != current or current != int(row["received_size"]):
                raise CampaignControlPlaneError(f"artifact_offset_conflict:expected_{current}")
            if current + len(data) > int(row["byte_size"]):
                raise CampaignControlPlaneError("artifact_exceeds_declared_size")
            with part.open("ab") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            received = current + len(data)
            state, digest = "receiving", None
            if received == int(row["byte_size"]):
                digest = hashlib.sha256(part.read_bytes()).hexdigest()
                if row["expected_sha256"] and digest != row["expected_sha256"]:
                    # Keep the file and durable offset consistent when the
                    # transaction rolls back after a corrupt final chunk.
                    with part.open("r+b") as handle:
                        handle.truncate(current)
                    raise CampaignControlPlaneError("artifact_final_hash_mismatch")
                os.replace(part, path)
                state = "complete"
            connection.execute(
                "UPDATE gpu_artifacts SET received_size=?,sha256=?,state=?,updated_at=? WHERE campaign_id=? AND attempt_id=? AND artifact_id=?",
                (received, digest, state, float(self._clock()), campaign, attempt, artifact),
            )
            self._event(
                connection,
                campaign,
                attempt,
                "artifact_progress",
                {"artifact_id": artifact, "received_size": received, "state": state},
            )
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_id": artifact,
            "state": state,
            "received_size": received,
            "sha256": digest,
        }

    @staticmethod
    def _missing_artifacts(
        connection: sqlite3.Connection, campaign: str, attempt: str
    ) -> list[str]:
        complete = {
            row[0]
            for row in connection.execute(
                "SELECT artifact_id FROM gpu_artifacts WHERE campaign_id=? AND attempt_id=? AND state='complete'",
                (campaign, attempt),
            ).fetchall()
        }
        return [name for name in REQUIRED_ARTIFACTS if name not in complete]

    def snapshot(self, campaign_id: str) -> dict[str, Any]:
        campaign = _identifier(campaign_id, "campaign_id")
        with self._connect() as connection:
            root = connection.execute(
                "SELECT * FROM gpu_campaigns WHERE campaign_id=?", (campaign,)
            ).fetchone()
            if root is None:
                raise CampaignControlPlaneError("campaign_unknown")
            attempts = []
            for row in connection.execute(
                "SELECT * FROM gpu_attempts WHERE campaign_id=? ORDER BY seed", (campaign,)
            ):
                item = dict(row)
                item["semantic_task_success"] = (
                    None
                    if row["semantic_task_success"] is None
                    else bool(row["semantic_task_success"])
                )
                item["missing_required_artifacts"] = self._missing_artifacts(
                    connection, campaign, row["attempt_id"]
                )
                attempts.append(item)
            events = int(
                connection.execute(
                    "SELECT COUNT(*) FROM gpu_campaign_events WHERE campaign_id=?", (campaign,)
                ).fetchone()[0]
            )
            finalization = connection.execute(
                "SELECT provider_result_sha256,teardown_proof_sha256,verified_at "
                "FROM gpu_campaign_finalization WHERE campaign_id=?",
                (campaign,),
            ).fetchone()
            spec = json.loads(root["spec_json"])
        return {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": campaign,
            "state": root["state"],
            "blocker": root["blocker"],
            "created_at_epoch": root["created_at"],
            "queue_deadline_epoch": root["created_at"]
            + int(spec["runtime"]["queue_timeout_seconds"]),
            "spec_digest": root["spec_digest"],
            "release_fingerprint": spec["release_candidate_fingerprint"],
            "worker_image_ref": spec["worker_image_ref"],
            "terminal": root["state"] in TERMINAL_CAMPAIGN_STATES,
            "attempts": attempts,
            "finalization": (
                {
                    "provider_result_sha256": finalization[0],
                    "teardown_proof_sha256": finalization[1],
                    "verified_at_epoch": finalization[2],
                    "provider_absence_confirmed": True,
                }
                if finalization is not None
                else None
            ),
            "event_count": events,
            "provider_calls_performed": 0,
            "provider_credentials_accepted": False,
        }

    def customer_status(self, campaign_id: str) -> dict[str, Any]:
        self.reconcile_deadlines(campaign_id)
        snapshot = self.snapshot(campaign_id)
        state = snapshot["state"]
        projection = {
            "accepted": ("accepted", "Evaluation accepted"),
            "queued": ("queued", "Waiting for a warm worker"),
            "running_smoke": ("starting", "Validating the simulator and policy"),
            "running_episodes": ("running", "Running evaluation episodes"),
            "collecting": ("processing", "Retrieving episode evidence"),
            "validating": ("processing", "Validating videos and results"),
            "teardown_pending": ("processing", "Verifying provider teardown"),
            "completed": ("completed", "Evaluation complete"),
            "smoke_blocked": ("failed", "Startup validation failed"),
            "failed": ("failed", "Evaluation did not complete"),
            "cancelled": ("cancelled", "Evaluation cancelled"),
        }[state]
        completed = sum(row["state"] in TERMINAL_ATTEMPT_STATES for row in snapshot["attempts"])
        semantic_success: bool | None = None
        if snapshot["terminal"]:
            semantic_success = any(
                row["semantic_task_success"] is True
                for row in snapshot["attempts"]
                if row["kind"] == "episode"
            )
        return {
            "schema_version": CUSTOMER_STATUS_SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "status": projection[0],
            "message": projection[1],
            "terminal": snapshot["terminal"],
            "attempts_completed": completed,
            "attempts_total": len(snapshot["attempts"]),
            "semantic_task_success": semantic_success,
            "provider_internal_state_exposed": False,
            "estimated_completion_time": None,
            "queue_deadline_epoch": snapshot["queue_deadline_epoch"],
        }

    def reconcile_deadlines(self, campaign_id: str) -> dict[str, Any]:
        """Enforce queue, smoke, and episode deadlines from durable timestamps."""

        campaign = _identifier(campaign_id, "campaign_id")
        now = float(self._clock())
        with self._transaction() as connection:
            self._reconcile_deadlines_locked(connection, campaign, now)
        return self.snapshot(campaign)

    def _reconcile_deadlines_locked(
        self, connection: sqlite3.Connection, campaign: str, now: float
    ) -> dict[str, str]:
        """Apply all elapsed deadlines inside the caller's write transaction."""

        row = connection.execute(
            "SELECT * FROM gpu_campaigns WHERE campaign_id=?", (campaign,)
        ).fetchone()
        if row is None:
            raise CampaignControlPlaneError("campaign_unknown")
        spec = json.loads(row["spec_json"])
        enforced: dict[str, str] = {}
        deadline = float(row["created_at"]) + int(spec["runtime"]["queue_timeout_seconds"])
        smoke_state = connection.execute(
            "SELECT attempt_id,state FROM gpu_attempts WHERE campaign_id=? AND kind='smoke'",
            (campaign,),
        ).fetchone()
        if (
            row["state"] in {"accepted", "queued"}
            and smoke_state is not None
            and smoke_state["state"] in {"planned", "waiting_for_worker"}
            and now >= deadline
        ):
            connection.execute(
                "UPDATE gpu_campaigns SET state='failed',blocker='queue_timeout',updated_at=? WHERE campaign_id=?",
                (now, campaign),
            )
            connection.execute(
                "UPDATE gpu_attempts SET state='cancelled',terminal_reason='queue_timeout',ended_at=? "
                "WHERE campaign_id=? AND state IN ('planned','waiting_for_worker')",
                (now, campaign),
            )
            enforced[str(smoke_state["attempt_id"])] = "queue_timeout"
            self._event(
                connection,
                campaign,
                None,
                "campaign_queue_timeout",
                {"queue_deadline_epoch": deadline},
            )
        active = connection.execute(
            "SELECT * FROM gpu_attempts WHERE campaign_id=? "
            "AND state IN ('running','collecting','validating')",
            (campaign,),
        ).fetchall()
        for active_attempt in active:
            kind = str(active_attempt["kind"])
            timeout_field = (
                "smoke_timeout_seconds" if kind == "smoke" else "episode_timeout_seconds"
            )
            attempt_deadline = float(active_attempt["started_at"]) + int(
                spec["runtime"][timeout_field]
            )
            if now < attempt_deadline:
                continue
            reason = "smoke_timeout" if kind == "smoke" else "episode_timeout"
            cursor = connection.execute(
                "UPDATE gpu_attempts SET state='timed_out',terminal_reason=?,ended_at=? "
                "WHERE campaign_id=? AND attempt_id=? "
                "AND state IN ('running','collecting','validating')",
                (reason, now, campaign, active_attempt["attempt_id"]),
            )
            if cursor.rowcount != 1:
                continue
            enforced[str(active_attempt["attempt_id"])] = reason
            self._event(
                connection,
                campaign,
                str(active_attempt["attempt_id"]),
                "attempt_deadline_exceeded",
                {"deadline_epoch": attempt_deadline, "terminal_reason": reason},
            )
            if kind == "smoke":
                connection.execute(
                    "UPDATE gpu_attempts SET state='cancelled',"
                    "terminal_reason='smoke_timeout',ended_at=? "
                    "WHERE campaign_id=? AND kind='episode' "
                    "AND state IN ('planned','waiting_for_worker')",
                    (now, campaign),
                )
                connection.execute(
                    "UPDATE gpu_campaigns SET blocker='smoke_timeout' WHERE campaign_id=?",
                    (campaign,),
                )
        if active:
            self._derive_campaign_state(connection, campaign)
        return enforced


def _read_api_token(explicit: str | None) -> str:
    if explicit is not None:
        token = explicit.strip()
    else:
        raw = os.getenv(CAMPAIGN_TOKEN_FILE_ENV, "").strip()
        if not raw:
            raise RuntimeError("production_gpu_campaign_token_file_required")
        path = Path(raw).expanduser().resolve()
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("production_gpu_campaign_token_file_invalid")
        if path.stat().st_mode & 0o077:
            raise RuntimeError("production_gpu_campaign_token_file_permissions_too_open")
        token = path.read_text(encoding="utf-8").strip()
    if len(token.encode()) < 32:
        raise RuntimeError("production_gpu_campaign_token_too_short")
    return token


def create_production_gpu_campaign_app(
    *,
    database_path: str | Path,
    artifact_root: str | Path,
    auth_token: str | None = None,
    clock: Callable[[], float] = time.time,
) -> Any:
    """Create the private campaign API; no route can mutate a provider."""

    from fastapi import Depends, FastAPI, Header, HTTPException, status

    control = ProductionGpuCampaignControlPlane(database_path, artifact_root, clock=clock)
    expected = _read_api_token(auth_token)

    def authorize(authorization: str | None = Header(default=None)) -> None:
        supplied = (
            authorization[7:] if authorization and authorization.startswith("Bearer ") else ""
        )
        if not supplied or not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    app = FastAPI(title="Blueprint Production GPU Campaign Control Plane", version=SCHEMA_VERSION)
    app.state.control = control

    @app.get("/healthz")
    def health() -> dict[str, Any]:
        return {"status": "ok", "schema_version": SCHEMA_VERSION, "provider_calls_performed": 0}

    @app.post("/v1/campaigns", dependencies=[Depends(authorize)])
    def create(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return control.create_campaign(payload)
        except (ValueError, CampaignControlPlaneError, sqlite3.IntegrityError) as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT) from exc

    @app.get("/v1/campaigns/{campaign_id}", dependencies=[Depends(authorize)])
    def snapshot(campaign_id: str) -> dict[str, Any]:
        try:
            return control.snapshot(campaign_id)
        except (ValueError, CampaignControlPlaneError) as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc

    @app.get("/v1/campaigns/{campaign_id}/status", dependencies=[Depends(authorize)])
    def customer_status(campaign_id: str) -> dict[str, Any]:
        try:
            return control.customer_status(campaign_id)
        except (ValueError, CampaignControlPlaneError) as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc

    @app.post(
        "/v1/campaigns/{campaign_id}/attempts/{attempt_id}/transition",
        dependencies=[Depends(authorize)],
    )
    def transition(campaign_id: str, attempt_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return control.transition_attempt(
                campaign_id,
                attempt_id,
                str(payload.get("state") or ""),
                terminal_reason=payload.get("terminal_reason"),
                semantic_task_success=payload.get("semantic_task_success"),
                simulator_steps=payload.get("simulator_steps"),
                policy_actions=payload.get("policy_actions"),
            )
        except (TypeError, ValueError, CampaignControlPlaneError) as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT) from exc

    @app.post(
        "/v1/campaigns/{campaign_id}/attempts/{attempt_id}/artifacts/{artifact_id}/begin",
        dependencies=[Depends(authorize)],
    )
    def begin_artifact(
        campaign_id: str, attempt_id: str, artifact_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            return control.begin_artifact(
                campaign_id,
                attempt_id,
                artifact_id,
                relative_path=str(payload.get("relative_path") or ""),
                total_size=int(payload.get("total_size") or 0),
                expected_sha256=payload.get("expected_sha256"),
            )
        except (TypeError, ValueError, CampaignControlPlaneError) as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT) from exc

    @app.post(
        "/v1/campaigns/{campaign_id}/attempts/{attempt_id}/artifacts/{artifact_id}/chunk",
        dependencies=[Depends(authorize)],
    )
    def artifact_chunk(
        campaign_id: str, attempt_id: str, artifact_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            data = base64.b64decode(str(payload.get("data_base64") or ""), validate=True)
            return control.append_artifact_chunk(
                campaign_id,
                attempt_id,
                artifact_id,
                offset=int(payload.get("offset") or 0),
                data=data,
                chunk_sha256=str(payload.get("chunk_sha256") or ""),
            )
        except (TypeError, ValueError, CampaignControlPlaneError) as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT) from exc

    @app.post(
        "/v1/campaigns/{campaign_id}/finalize",
        dependencies=[Depends(authorize)],
    )
    def finalize(campaign_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return control.finalize_campaign(
                campaign_id,
                provider_result=dict(payload.get("provider_result") or {}),
                teardown_proof=dict(payload.get("teardown_proof") or {}),
            )
        except (
            TypeError,
            ValueError,
            CampaignControlPlaneError,
            sqlite3.IntegrityError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT) from exc

    return app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        raise SystemExit("campaign control plane must bind loopback behind private ingress")
    import uvicorn

    uvicorn.run(
        create_production_gpu_campaign_app(
            database_path=args.database, artifact_root=args.artifact_root
        ),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
