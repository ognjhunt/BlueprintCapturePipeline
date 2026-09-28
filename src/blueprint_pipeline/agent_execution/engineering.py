"""Bound an unresolved saved-stage defect to the existing engineering lane.

The model diagnoses; a private controller policy grants the engineering scope.
Only machine replay evidence and fixed acceptance requirements leave this
boundary. Creating an engineering issue does not authorize a paid resubmission.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import re
import time
import subprocess
import os
import selectors
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..common import write_json
from ..decision_evidence_contracts import cross_runtime_canonical_digest
from .contracts import AgentExecutionError, IDENTIFIER
from .webapp_delivery import post_admission

ENDPOINT = "/api/internal/pipeline/agent-execution/engineering"


class EngineeringPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["blueprint_agent_engineering_policy.v1"]
    enabled: bool
    policy_id: str = Field(pattern=IDENTIFIER)
    repository: Literal["ognjhunt/BlueprintCapturePipeline"]
    run_id_prefixes: tuple[str, ...] = Field(min_length=1, max_length=20)
    allowed_paths: tuple[str, ...] = Field(min_length=1, max_length=30)
    required_test_paths: tuple[str, ...] = Field(min_length=1, max_length=20)
    maximum_handoffs: int = Field(ge=1, le=20)
    maximum_changed_files: int = Field(ge=1, le=30)
    maximum_patch_bytes: int = Field(ge=1, le=200_000)
    worker_budget_reference: str = Field(min_length=1, max_length=300)
    maximum_worker_timeout_seconds: int = Field(ge=60, le=3600)
    accepted_by: str = Field(min_length=1, max_length=192)
    expires_at: int = Field(gt=0)

    @model_validator(mode="after")
    def bounded_paths(self):
        for name in (*self.allowed_paths, *self.required_test_paths):
            path = Path(name)
            if (not re.fullmatch(r"[A-Za-z0-9_./-]+", name) or path.is_absolute() or path.as_posix() != name
                    or any(part in {"..", ".git"} or part.startswith(".env") for part in path.parts)
                    or len(path.parts) < 2):
                raise ValueError("engineering_path_not_bounded")
        if any(not name.startswith("tests/") or not name.endswith(".py") for name in self.required_test_paths):
            raise ValueError("engineering_test_path_invalid")
        if any(not prefix or len(prefix) < 6 or not re.fullmatch(r"[A-Za-z0-9._:-]+", prefix)
               for prefix in self.run_id_prefixes):
            raise ValueError("engineering_run_scope_invalid")
        return self

    @property
    def policy_digest(self):
        return cross_runtime_canonical_digest(self.model_dump(mode="json"))


def engineering_policy_status(service, record):
    from .production import _read_private
    try:
        policy = EngineeringPolicy.model_validate_json(_read_private(Path(service.config.engineering_policy_file)))
        return {"policy_digest": policy.policy_digest, "enabled": bool(policy.enabled and record.enabled
            and time.time() < policy.expires_at and record.task.run_id.startswith(policy.run_id_prefixes))}
    except (AgentExecutionError, OSError, TypeError, ValueError):
        return {"policy_digest": None, "enabled": False}


def queue_engineering_handoff(service, record):
    """One immutable handoff after an admitted technical diagnosis and replay."""
    from .production import _read_private
    path = service.config.engineering_policy_file
    if not path or record.task.capability != "runtime_failure_recovery" or not record.stage_replays:
        return None
    policy = EngineeringPolicy.model_validate_json(_read_private(Path(path)))
    if (not policy.enabled or time.time() >= policy.expires_at
            or not record.task.run_id.startswith(policy.run_id_prefixes)
            or not record.enabled or "blueprint-webapp" not in record.owner_client_ids):
        return None
    try:
        state = service.journal.task(record.task.task_id)
    except AgentExecutionError as exc:
        if str(exc) == "agent_task_missing":
            return None
        raise
    result = state["result"]
    if (state["state"] != "completed" or state["cancel_requested"] or not result
            or result["output"].get("disposition") not in {"investigate", "abstain"}):
        return None
    candidates = []
    for operation in service.journal.task_operations(record.task.task_id):
        call, outcome = operation["request"], operation["outcome"] or {}
        if call["tool_id"] != "replay_retained_stage" or outcome.get("success") is not True:
            continue
        replay = outcome.get("output", {})
        # Missing capacity, source views, authority and a deliberately excluded
        # paid boundary are not permission to patch the program's predicates.
        # Canonical replay uses `refused` only after the stage handler starts.
        # `job_refused` is parent/job admission and cannot authorize code repair.
        if (replay.get("status") != "refused"
                or replay.get("external_boundary_reached") is not False
                or replay.get("blocker_code") in {None, "replay_process_did_not_write_report", "replay_wall_time_exhausted"}):
            continue
        binding = next((item for item in record.stage_replays if item.replay_id == call["arguments"].get("replay_id")), None)
        if binding and replay.get("job_sha256") == binding.job_sha256 and replay.get("source_commit") == record.task.source_commit:
            candidates.append((binding, replay))
    if len(candidates) != 1:
        return None
    binding, replay = candidates[0]
    packet = {"schema_version": "blueprint_agent_engineering_handoff.v1", "program": "arm-decision-proof-v1",
        "task_id": record.task.task_id, "task_digest": record.task.task_digest, "run_id": record.task.run_id,
        "source_commit": record.task.source_commit, "diagnosis_result_digest": result["result_digest"],
        "child_id": binding.child_id, "job_sha256": binding.job_sha256,
        "replay_report_digest": replay["report_digest"], "replay_status": replay["status"],
        "blocker_code": replay["blocker_code"], "policy": policy.model_dump(mode="json"),
        "policy_digest": policy.policy_digest, "paid_resubmission_authorized": False,
        "scientific_acceptance_granted": False, "independent_review_required": True}
    identity = cross_runtime_canonical_digest({key: packet[key] for key in (
        "task_digest", "replay_report_digest", "diagnosis_result_digest")})
    packet["handoff_id"] = "repair-" + identity[7:]
    packet["handoff_digest"] = cross_runtime_canonical_digest(packet)
    root = service.journal.root / "engineering-handoffs"
    with service.journal.own_task("engineering-policy:" + policy.policy_digest):
        path = root / "pending" / (packet["handoff_id"] + ".json")
        if path.exists():
            if json.loads(_read_private(path)) != packet:
                raise AgentExecutionError("agent_engineering_handoff_conflict")
            return packet
        reservation_id = "engineering_reserved_" + packet["handoff_id"]
        if service.journal.event(reservation_id) is None:
            with service.journal._connect() as connection:
                rows = connection.execute("SELECT payload_json FROM events WHERE event_id LIKE ?", ("engineering_reserved_%",)).fetchall()
            count = sum(json.loads(row[0]).get("policy_digest") == policy.policy_digest for row in rows)
            if count >= policy.maximum_handoffs:
                raise AgentExecutionError("agent_engineering_handoff_limit_reached")
            service.journal.record_event(reservation_id, {"policy_digest": policy.policy_digest,
                "handoff_id": packet["handoff_id"], "handoff_digest": packet["handoff_digest"]})
        write_json(path, packet)
        path.chmod(0o640)
    return packet


def flush_engineering_handoffs(service, *, post=post_admission):
    """Keep the exact packet until the Website acknowledges durable storage."""
    from .production import _read_private
    if not service.config.webapp_admission_url or not service.config.webapp_sync_token_file:
        return None
    root = service.journal.root / "engineering-handoffs"
    parts = urlsplit(service.config.webapp_admission_url)
    endpoint = urlunsplit((parts.scheme, parts.netloc, ENDPOINT, "", ""))
    for path in sorted((root / "pending").glob("*.json"), key=lambda item: (item.stat().st_mtime_ns, item.name)):
        receipt_path = root / "receipts" / path.name
        if receipt_path.exists():
            continue
        try:
            with service.journal.own_task("engineering-delivery:" + path.stem):
                if receipt_path.exists():
                    continue
                packet = json.loads(_read_private(path))
                if packet["handoff_digest"] != cross_runtime_canonical_digest({k: v for k, v in packet.items() if k != "handoff_digest"}):
                    raise AgentExecutionError("agent_engineering_packet_changed")
                policy = engineering_policy_status(service, service.record(packet["task_id"]))
                if not policy["enabled"] or policy["policy_digest"] != packet["policy_digest"]:
                    write_json(receipt_path, {"handoff_id": packet["handoff_id"], "status": "revoked",
                                             "stored": False, "engineering_complete": False})
                    continue
                token = _read_private(Path(service.config.webapp_sync_token_file), secret=True, limit=16_000).decode().strip()
                response = post(packet, endpoint=endpoint, token=token, expected_path=ENDPOINT)
                if (response.get("schema_version") != "blueprint_agent_engineering_admission_receipt.v1"
                        or response.get("handoff_digest") != packet["handoff_digest"]
                        or response.get("handoff_id") != packet["handoff_id"] or response.get("stored") is not True):
                    raise AgentExecutionError("agent_engineering_delivery_readback_invalid")
                write_json(receipt_path, {**response, "observed_at": time.time(), "engineering_complete": False})
                return response
        except (AgentExecutionError, ValueError, OSError, KeyError):
            try:
                newest = max(item.stat().st_mtime_ns for item in (root / "pending").glob("*.json"))
                os.utime(path, ns=(path.stat().st_atime_ns, max(time.time_ns(), newest + 1)), follow_symlinks=False)
            except OSError:
                pass
            return {"handoff_id": path.stem, "status": "delivery_pending", "engineering_complete": False}
    return None


def bind_paperclip_issue(service, *, task_id, issue_id, post=post_admission):
    """Trusted controller operation; never exposed as a model function tool."""
    from .production import _read_private
    import uuid
    if str(uuid.UUID(issue_id)) != issue_id:
        raise AgentExecutionError("agent_paperclip_issue_identifier_invalid")
    record = service.authorize_client(task_id, "blueprint-webapp")
    path = "/api/internal/pipeline/agent-execution/paperclip-bindings"
    if not service.config.webapp_admission_url or not service.config.webapp_sync_token_file:
        raise AgentExecutionError("agent_webapp_admission_not_configured")
    parts = urlsplit(service.config.webapp_admission_url)
    payload = {"task_id": task_id, "task_digest": record.task.task_digest, "issue_id": issue_id}
    token = _read_private(Path(service.config.webapp_sync_token_file), secret=True, limit=16_000).decode().strip()
    response = post(payload, endpoint=urlunsplit((parts.scheme, parts.netloc, path, "", "")), token=token, expected_path=path)
    if (response.get("schema_version") != "blueprint_paperclip_issue_binding.v1" or response.get("stored") is not True
            or any(response.get("binding", {}).get(key) != value for key, value in payload.items())):
        raise AgentExecutionError("agent_paperclip_binding_readback_invalid")
    service.journal.record_event("paperclip_bound_" + issue_id, response)
    return response


def verify_engineering_candidate(packet, *, repository_root, candidate_commit):
    """Inspect immutable Git objects; do not run candidate code or accept a release."""
    if (not re.fullmatch(r"[0-9a-f]{40}", candidate_commit)
            or packet.get("handoff_digest") != cross_runtime_canonical_digest({k: v for k, v in packet.items() if k != "handoff_digest"})):
        raise AgentExecutionError("agent_engineering_candidate_identity_invalid")
    policy = EngineeringPolicy.model_validate(packet["policy"])
    if policy.policy_digest != packet["policy_digest"]:
        raise AgentExecutionError("agent_engineering_candidate_policy_changed")
    base = packet["source_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", base):
        raise AgentExecutionError("agent_engineering_base_invalid")
    root = Path(repository_root).resolve(strict=True)
    def git(*args, limit=64000):
        process = subprocess.Popen(["git", "-c", f"safe.directory={root}", "-C", str(root), *args],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        output = bytearray()
        try:
            with selectors.DefaultSelector() as selected:
                selected.register(process.stdout, selectors.EVENT_READ)
                deadline = time.monotonic() + 30
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selected.select(remaining):
                        raise AgentExecutionError("agent_engineering_git_observation_timeout")
                    chunk = os.read(process.stdout.fileno(), min(8192, limit + 1 - len(output)))
                    if not chunk:
                        break
                    output.extend(chunk)
                    if len(output) > limit:
                        raise AgentExecutionError("agent_engineering_git_output_exceeds_scope")
            if process.wait(timeout=max(0.1, deadline - time.monotonic())):
                raise AgentExecutionError("agent_engineering_git_observation_failed")
            return bytes(output)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()
    remote = git("remote", "get-url", "origin").decode().strip()
    if remote not in {f"https://github.com/{policy.repository}", f"https://github.com/{policy.repository}.git", f"git@github.com:{policy.repository}.git"}:
        raise AgentExecutionError("agent_engineering_repository_mismatch")
    git("merge-base", "--is-ancestor", base, candidate_commit)
    changed = [name for name in git("diff", "--name-only", "-z", base, candidate_commit).decode().split("\0") if name]
    patch_bytes = len(git("diff", "--binary", base, candidate_commit, limit=policy.maximum_patch_bytes))
    if (not changed or not set(changed) <= set(policy.allowed_paths)
            or len(changed) > policy.maximum_changed_files or patch_bytes > policy.maximum_patch_bytes):
        raise AgentExecutionError("agent_engineering_patch_outside_scope")
    for name in changed:
        entry = git("ls-tree", candidate_commit, "--", name)
        if entry and entry.split(b" ", 1)[0] not in {b"100644", b"100755"}:
            raise AgentExecutionError("agent_engineering_patch_link_or_submodule_refused")
        if name in policy.required_test_paths and git("ls-tree", base, "--", name):
            raise AgentExecutionError("agent_engineering_required_baseline_test_changed")
    receipt = {"schema_version": "blueprint_agent_engineering_candidate_scope.v1", "status": "scope_verified",
        "handoff_id": packet["handoff_id"], "handoff_digest": packet["handoff_digest"], "policy_digest": policy.policy_digest,
        "base_commit": base, "candidate_commit": candidate_commit, "changed_paths": changed, "patch_bytes": patch_bytes,
        "required_test_paths": list(policy.required_test_paths), "tests_passed_inferred": False,
        "independent_review_required": True, "production_promotion_authorized": False}
    receipt["receipt_digest"] = cross_runtime_canonical_digest(receipt)
    return receipt


def main(argv=None):
    from .production import configured_service
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    bind = commands.add_parser("bind-issue")
    bind.add_argument("--task-id", required=True)
    bind.add_argument("--issue-id", required=True)
    verify = commands.add_parser("verify-candidate")
    verify.add_argument("--handoff-id", required=True)
    verify.add_argument("--candidate-commit", required=True)
    args = parser.parse_args(argv)
    service = configured_service()
    if args.action == "bind-issue":
        result = bind_paperclip_issue(service, task_id=args.task_id, issue_id=args.issue_id)
    else:
        from .production import _read_private
        if not re.fullmatch(r"repair-[a-f0-9]{64}", args.handoff_id):
            raise AgentExecutionError("agent_engineering_handoff_identifier_invalid")
        packet = json.loads(_read_private(service.journal.root / "engineering-handoffs" / "pending" / (args.handoff_id + ".json")))
        result = verify_engineering_candidate(packet, repository_root=Path(__file__).resolve().parents[3], candidate_commit=args.candidate_commit)
        write_json(service.journal.root / "engineering-handoffs" / "candidate-checks" / (args.candidate_commit + ".json"), result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
