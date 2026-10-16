"""Close one team's scored evaluation and seal the terminal the agent-run executor reports.

Beta journey step: L5 team policy evaluation. A run is one GPU session per
shard of its 200 cells. The terminal
receipt (``team_candidate_quick10_terminal.json`` in the run's job directory,
read by the executor's default terminal reader) is written only after:

- every shard's allocator result proves nothing was allocated (a refusal), or
- every shard's session result is joined to its teardown, staged-object
  absence and a fresh authenticated global provider-zero, and the closed
  results together validate for the admitted candidate
  (``team_candidate_terminal``).

While provider-zero is unproven nothing is written, so the run stays pending
and its hold is not released; the next poll retries. A failure is sealed as a
blocked terminal only once provider-zero is observed. The scoring statement in
a completed terminal is the one the WebApp admits: simulator-scored, the
team-candidate Quick-10 runtime, every binding present. It is a bounded
simulation estimate, never physical performance or safety, and never Arm
Decision Proof evidence.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .common import utc_now_iso
from .controlled_policy_configuration import canonical_request_digest
from .decision_evidence_contracts import canonical_digest
from .policy_canary_retained_billing import adapter_instance_ids
from .policy_canary_staged_object_absence import closeout_staged_objects
from .task_evaluation_scene_execution_authority import proves_no_provider_allocation
from .team_candidate_quick10_admission import runtime_binding, validate_admission
from .team_candidate_quick10_launch import (
    FAILURE_FILENAME,
    INTENT_FILENAME,
    SHARD_FILENAME,
    TERMINAL_FILENAME,
    TERMINAL_SCHEMA,
    TeamCandidateLaunchError,
)
from .team_candidate_quick10_result import team_candidate_terminal

ZERO_SCHEMA = "task_evaluation_policy_canary_vast_provider_zero.v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TeamCandidateLaunchError("team_candidate_record_invalid")
    return value


def _replace(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _seal(job_dir: Path, request: Mapping[str, Any], body: Mapping[str, Any]) -> dict[str, Any]:
    terminal = {"schema_version": TERMINAL_SCHEMA, "job_id": request["job_id"],
                "canonical_request_digest": canonical_request_digest(request), **body,
                "sealed_at_iso": utc_now_iso(), "terminal_digest": ""}
    terminal["terminal_digest"] = canonical_digest(terminal, digest_field="terminal_digest")
    path = job_dir / TERMINAL_FILENAME
    with path.open("x", encoding="utf-8") as stream:
        path.chmod(0o600)
        stream.write(json.dumps(terminal, sort_keys=True, allow_nan=False))
        stream.flush()
        os.fsync(stream.fileno())
    return terminal


def _zero_verified(value: Mapping[str, Any]) -> bool:
    return (value.get("schema_version") == ZERO_SCHEMA and value.get("provider_zero_verified") is True
            and value.get("live_instance_count") == 0 and value.get("blockers") == [])


def validate_native_result_binding(inner: Mapping[str, Any], *, expected_digest: str) -> None:
    """Authenticate raw content before a closeout join derives a new sealed result."""
    if (inner.get("result_digest") != expected_digest
            or inner.get("result_digest") != canonical_digest(inner, digest_field="result_digest")):
        raise TeamCandidateLaunchError("team_candidate_provider_result_binding_mismatch")


def join_team_session_closeout(*, inner: Mapping[str, Any], adapter: Mapping[str, Any],
                               provider_zero: Mapping[str, Any], adapter_path: Path | None = None) -> dict[str, Any]:
    """The session result with its provider closeout bound in (one allocation, every planned episode)."""

    value = json.loads(json.dumps(dict(inner), allow_nan=False))
    instance_ids = adapter_instance_ids(adapter, result_path=adapter_path, read_json=lambda path, code: _read(path),
                                        error_factory=TeamCandidateLaunchError)
    closeout = adapter.get("provider_closeout")
    staged = closeout_staged_objects(adapter)
    teardown = (isinstance(closeout, Mapping) and closeout.get("provider_zero_confirmed") is True
                and closeout.get("warm_session_retained") is False and staged["absent"]
                and adapter.get("continuing_spend_from_this_run") is False)
    zero = _zero_verified(provider_zero)
    episodes = value.get("episodes")
    # Every planned episode returned a row. A missing row is an incomplete
    # session; a failed row is an outcome team_candidate_terminal scores or voids.
    returned = (isinstance(episodes, list) and len(episodes) == value.get("learned_policy_rollout_count")
                and all(isinstance(row, Mapping) for row in episodes))
    closeout_blockers = [] if returned else ["team_candidate_episodes_incomplete"]
    closeout_blockers += [] if len(instance_ids) == 1 else ["policy_canary_provider_allocation_count_invalid"]
    closeout_blockers += [] if teardown else ["policy_canary_teardown_incomplete"]
    closeout_blockers += [] if zero else ["policy_canary_global_provider_zero_unproven"]
    closeout_blockers = sorted(set(closeout_blockers + list(staged["blockers"])))
    value["provider_allocations_observed"] = len(instance_ids)
    value["session_closeout"] = {"status": "closed" if teardown and zero else "blocked",
                                 "provider_allocations_observed": len(instance_ids),
                                 "teardown_completed": teardown, "provider_zero_confirmed": zero,
                                 "blockers": closeout_blockers}
    # The session validator's terms: completed_unqualified only when every
    # episode completed; a closed session with failed episodes is "blocked".
    value["status"] = ("completed_unqualified" if not closeout_blockers
                       and all(row.get("status") == "completed" for row in episodes) else "blocked")
    blockers = [str(item) for item in value.get("blockers") or [] if str(item)]
    value["blockers"] = sorted(set(blockers + closeout_blockers))
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    return value


def _retain_probe(job_dir: Path, request: Mapping[str, Any], trace: Mapping[str, Any]) -> None:
    """Score and retain the scene's robot-free probe recorded in this session; never fail the run."""

    from .team_candidate_quick10_registry import profile_for_request, scene_facts
    from .team_robot_free_probe import retain_recorded_probe

    path = job_dir / "robot_free_probe_retention.json"
    if path.exists():
        return
    try:
        profile = profile_for_request(request)
        outcome = retain_recorded_probe(profile=profile, facts=scene_facts(profile), trace=trace)
    except Exception as exc:  # noqa: BLE001 - the probe is evidence for the scene, not this run's result
        outcome = {"status": "retention_failed", "reason": f"{type(exc).__name__}:{str(exc)[:300]}"}
    _replace(path, {**outcome, "trace_digest": trace.get("trace_digest")})


def _shard_roots(job_dir: Path, shard_count: int) -> list[Path]:
    roots = []
    for index in range(shard_count):
        stages = sorted((job_dir / "shards" / f"{index:02d}").glob(f"stage-*/{SHARD_FILENAME}"))
        if not stages:
            raise TeamCandidateLaunchError("team_evaluation_shard_missing")
        roots.append(stages[-1].parent)
    return roots


def seal_team_candidate_terminal(*, request: Mapping[str, Any], job_dir: Path,
                                 provider_zero: Callable[[], Mapping[str, Any]]) -> dict[str, Any] | None:
    """Seal the terminal from every shard's allocator result, or return None while provider-zero is unproven."""

    if (job_dir / TERMINAL_FILENAME).exists():
        return _read(job_dir / TERMINAL_FILENAME)
    intent = _read(job_dir / INTENT_FILENAME)
    if intent.get("job_id") != request["job_id"] or intent.get("canonical_request_digest") != canonical_request_digest(
            request):
        raise TeamCandidateLaunchError("team_candidate_terminal_request_binding_mismatch")
    admission = validate_admission(_read(job_dir / "team_candidate_admission.json"))
    if admission["admission_digest"] != intent.get("admission_digest"):
        raise TeamCandidateLaunchError("team_candidate_terminal_admission_mismatch")
    roots = _shard_roots(job_dir, int(intent["shard_count"]))
    adapters = [(root, _read(root / "allocator_result.json")) for root in roots]
    if all(proves_no_provider_allocation(adapter) for _, adapter in adapters):
        return _seal(job_dir, request, {"status": "blocked", "provider_allocation_performed": False,
                                        "blockers": sorted({str(item) for _, adapter in adapters
                                                            for item in adapter.get("blockers") or []})
                                        or ["team_candidate_provider_not_allocated"]})
    zero = dict(provider_zero())
    _replace(job_dir / "post_teardown_global_provider_zero.json", zero)
    if not _zero_verified(zero):
        return None
    joined_results = []
    for root, adapter in adapters:
        if proves_no_provider_allocation(adapter):
            raise TeamCandidateLaunchError("team_evaluation_shard_not_allocated:" + root.parent.name)
        bundle = _read(root / "bundle" / "team_candidate_quick10_session_bundle_receipt.v1.json")
        if adapter.get("bundle_sha256") != bundle.get("bundle_sha256"):
            raise TeamCandidateLaunchError("team_candidate_allocator_result_bundle_mismatch")
        native_path = Path(str(adapter.get("native_control_result_path") or ""))
        if not native_path.is_file():
            raise TeamCandidateLaunchError("team_candidate_provider_result_missing")
        inner = _read(native_path)
        validate_native_result_binding(inner, expected_digest=adapter.get("native_control_result_digest"))
        if isinstance(inner.get("robot_free_probe_trace"), Mapping):
            _retain_probe(job_dir, request, inner["robot_free_probe_trace"])
        joined = join_team_session_closeout(inner=inner, adapter=adapter, provider_zero=zero,
                                            adapter_path=root / "allocator_result.json")
        _replace(root / "team_candidate_session_result.json", joined)
        if joined["session_closeout"]["blockers"]:
            raise TeamCandidateLaunchError(";".join(joined["session_closeout"]["blockers"]))
        joined_results.append(joined)
    binding = runtime_binding(admission)
    if binding.get("robot_free_probe_receipt_digest") is None:
        # The scene's probe was recorded and passed in this evaluation's first cell: bind that receipt.
        from .team_candidate_quick10_registry import offer_eligibility, profile_for_request

        facts, _ = offer_eligibility(profile_for_request(request))
        if facts is None or facts.get("probe") is None:
            raise TeamCandidateLaunchError("team_scene_robot_free_probe_not_passed")
        binding = {**binding, "robot_free_probe_receipt_digest": facts["probe"]["receipt_digest"]}
    recovery_kwargs = {}
    if intent.get("pre_policy_reset_recovery") is not None:
        from .team_candidate_episode_recovery import recovery_arguments
        recovery_kwargs = recovery_arguments(intent["pre_policy_reset_recovery"],
            request_digest=canonical_request_digest(request), admission_digest=admission["admission_digest"],
            episodes=int(admission["episodes"]))
    receipt = team_candidate_terminal(session_results=joined_results, runtime_binding=binding,
                                      purpose=str(admission["evaluation_purpose"]), episodes=int(admission["episodes"]),
                                      **recovery_kwargs)
    return _seal(job_dir, request, {**receipt, "provider_zero_verified": True,
                                    "admission_digest": admission["admission_digest"]})


def seal_blocked_terminal(*, request: Mapping[str, Any], job_dir: Path, error: Exception,
                          provider_zero: Callable[[], Mapping[str, Any]]) -> None:
    """Record the failure; release the run only once provider-zero is observed."""

    failure = job_dir / FAILURE_FILENAME
    if not failure.exists():
        _replace(failure, {"schema_version": "blueprint.team_candidate_quick10_failure.v1",
                           "job_id": request.get("job_id"), "error_class": type(error).__name__,
                           "blockers": [str(error)[:300] or type(error).__name__], "observed_at_iso": utc_now_iso()})
    if (job_dir / TERMINAL_FILENAME).exists() or allocator_running(job_dir):
        return
    if not (job_dir / INTENT_FILENAME).exists():
        # Refused before any launch was recorded: nothing could have been allocated.
        _seal(job_dir, request, {"status": "blocked", "blockers": _read(failure)["blockers"],
                                 "provider_zero_not_applicable": True})
        return
    try:
        zero = dict(provider_zero())
    except Exception:  # noqa: BLE001 - unproven provider zero keeps the run pending
        return
    if not _zero_verified(zero):
        return
    _replace(job_dir / "failed_provider_zero.json", zero)
    _seal(job_dir, request, {"status": "blocked", "blockers": _read(failure)["blockers"],
                             "provider_zero_verified": True})


def allocator_running(job_dir: Path) -> bool:
    """True while any shard's recorded allocator process for this job is still alive."""

    for path in job_dir.glob("shards/*/stage-*/allocator_process.json"):
        if (path.parent / "allocator_result.json").exists():
            continue
        try:
            os.kill(int(_read(path)["pid"]), 0)
        except (OSError, KeyError, TypeError, ValueError):
            continue
        return True
    return False


def read_team_candidate_terminal(*, job_dir: Path, expected_job_id: str,
                                 expected_canonical_request_digest: str) -> dict[str, Any] | None:
    """The sealed terminal (pending while a team run is in flight); None for a non-team job."""

    path = job_dir / TERMINAL_FILENAME
    if not path.exists():
        in_flight = (job_dir / INTENT_FILENAME).exists() or (job_dir / FAILURE_FILENAME).exists()
        return {"status": "pending", "blockers": []} if in_flight else None
    row = _read(path)
    if (row.get("schema_version") != TERMINAL_SCHEMA
            or row.get("terminal_digest") != canonical_digest(row, digest_field="terminal_digest")
            or row.get("job_id") != expected_job_id
            or row.get("canonical_request_digest") != expected_canonical_request_digest):
        raise ValueError("team_candidate_terminal_binding_mismatch")
    return row


__all__ = [
    "validate_native_result_binding",
    "allocator_running",
    "join_team_session_closeout",
    "read_team_candidate_terminal",
    "seal_blocked_terminal",
    "seal_team_candidate_terminal",
]
