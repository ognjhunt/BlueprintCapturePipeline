"""The scene packet queue: claim a website job's request, compile its packet, write its status, retry by itself.

A run (``--max-messages 1``) holds the queue lock, recovers a request a killed run left in
``processing/`` (its attempt removed unless the derived row names it), returns due blocks to
``pending/``, then claims requests: the capture is resolved (a withdrawn one is refused before
anything compiles), a readable half of the same revision is reused, else ``scene_packet``
compiles a new attempt. Every request ends in a written status before it moves to ``processed/``.

Retries: transient blocks after 300, 1800, 7200 s; other refusals on ``job_retry_policy``'s
schedule, never a privacy, rights, money or owner-contract one; an owner-contract refusal once per
new site confirmation; every block once per new release; cold storage once its run is back; an
unscorable half once per release. ``--recover`` runs the site-confirmation exchange, recovery and
requeue, and claims nothing. Nothing here spends money.
"""

from __future__ import annotations

import argparse
import builtins
import fcntl
import json
import os
import re
import secrets
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .job_retry_policy import REFUSAL_RETRY_DELAYS_SECONDS, hard_stop_kind, refusal_retry_delay
from .scene_packet import OWNER_CONTRACT_INVALID
from .team_scene_capture_root import (
    QUEUE_ROOT_ENV,
    QUEUE_STATES,
    TEMPORARY_PREFIX,
    TEMPORARY_SUFFIX,
    WEBSITE_TASK_PREFIX,
    TeamScenePacketRequestRefused,
    _read_json,
    capture_key,
    resolve_capture_root,
)

# The request and status contracts the download, the confirmation exchange and withdrawal deletion read.
REQUEST_SCHEMA = "blueprint.team_scene_packet_request.v1"
STATUS_SCHEMA = "blueprint.team_scene_packet_status.v1"
RUN_SCHEMA = "blueprint.team_scene_packet_queue_run.v1"
REQUEST_FIELDS = frozenset({"schema_version", "intent_id", "intent_digest", "capture_id", "task_id",
                            "source_launch_root", "launch_receipt_digest", "scene_revision_digest",
                            "robot_free_readiness_status", "request_digest"})
SCENARIO_ID = "nominal"
PACKET_INPUT_FIELDS = ("droid_profile_path", "droid_profile_digest", "runtime_source_receipt_path", "output_root")
CONFIGURED_BUNDLE_FILENAME = "configured_scene_bundle.v1.zip"
TRANSIENT_BLOCKERS = frozenset({"team_scene_packet_interrupted", "configured_controls_deferred_fetch_failed"})
AUTOMATIC_RETRY_BACKOFF_SECONDS = (300, 1800, 7200)
# Status fields (the old worker's names, so a status either stage wrote reads the same).
AUTOMATIC_RETRY = "automatic_retry"
REFUSAL_RETRY = "refusal_retry"
RELEASE_COMMIT = "release_commit"
RELEASE_REQUEUED = "release_requeued"
CONFIRMATION_REQUEUED = "owner_success_confirmation_requeued"
UNSCORABLE_REQUEUED = "unscorable_half_release_requeued"
CARRIED_FIELDS = (AUTOMATIC_RETRY, REFUSAL_RETRY, RELEASE_REQUEUED, CONFIRMATION_REQUEUED, UNSCORABLE_REQUEUED)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_ATTEMPT = re.compile(r"[0-9a-f]{16}-[0-9a-f]{8}")
_CAPTURE_KEY = re.compile(r"[0-9a-f]{32}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
Fetcher = Callable[[Mapping[str, Any]], bytes]


def _temporary(directory: Path, value: Mapping[str, Any]) -> Path:
    # Never named *.json, so no queue glob sees a half-written file.
    with tempfile.NamedTemporaryFile("w", dir=directory, prefix=TEMPORARY_PREFIX, suffix=TEMPORARY_SUFFIX,
                                     delete=False, encoding="utf-8") as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(value, indent=1, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    return temporary


def _write(path: Path, value: Mapping[str, Any]) -> None:
    os.replace(_temporary(path.parent, value), path)


def _queue(raw: str | Path) -> Path:
    """The queue root with its state directories, refused when relative or reached through a link."""

    root = Path(raw)
    if not root.is_absolute() or any(part.is_symlink() for part in (root, *root.parents)):
        raise TeamScenePacketRequestRefused("team_scene_packet_queue_root_unsafe")
    for state in QUEUE_STATES:
        (root / state).mkdir(parents=True, exist_ok=True, mode=0o750)
    return root


@contextmanager
def _locked(root: Path, *, wait: bool = True) -> Iterator[bool]:
    """Hold the queue lock; with ``wait=False`` yield False at once when another run holds it."""

    descriptor = os.open(root / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def running_release() -> str | None:
    """The release commit this module runs from (the checkout's HEAD, else its directory name), or None."""

    try:
        from .task_evaluation_release_identity import running_release_commit

        commit = running_release_commit(__file__)
        if not _COMMIT.fullmatch(commit or ""):
            commit = Path(__file__).resolve().parents[2].name
    except (OSError, ValueError, IndexError):
        return None
    return commit if _COMMIT.fullmatch(commit) else None


def validated_request(value: Any) -> dict[str, Any]:
    """The sealed request with exactly its fields, a website task and an absolute launch root, or refuse."""

    if (not isinstance(value, Mapping) or set(value) != REQUEST_FIELDS
            or value.get("schema_version") != REQUEST_SCHEMA
            or value.get("request_digest") != canonical_digest(dict(value), digest_field="request_digest")
            or not isinstance(value.get("capture_id"), str) or not _ID.fullmatch(value["capture_id"])
            or not str(value.get("task_id") or "").startswith(WEBSITE_TASK_PREFIX)
            or not isinstance(value.get("scene_revision_digest"), str)
            or not _DIGEST.fullmatch(value["scene_revision_digest"])
            or not Path(str(value.get("source_launch_root") or "")).is_absolute()):
        raise TeamScenePacketRequestRefused("team_scene_request_invalid")
    return dict(value)


def request_key(request: Mapping[str, Any]) -> str:
    """One request per capture and published revision."""

    return canonical_digest({"capture_id": request["capture_id"],
                             "scene_revision_digest": request["scene_revision_digest"]})[7:39]


def owner_scope(request: Mapping[str, Any], *, intent_root: str | Path | None = None
                ) -> tuple[dict[str, Any] | None, str | None]:
    """The request's own intent, where the site's confirmation is kept, or None and why.

    The intent must be exactly the one the request names, and its progression must have
    queued exactly this request.
    """

    from . import task_evaluation_scene_intake as intake
    from .task_evaluation_scene_intent_contracts import ROOT_ENV

    raw = str(intent_root or os.environ.get(ROOT_ENV, "")).strip()
    if not raw:
        return None, "scene_intake_root_not_configured"
    root = Path(raw)
    directory = root / str(request["intent_id"])
    if (not root.is_absolute() or not intake._identifier(request["intent_id"])
            or any(part.is_symlink() for part in (directory, *directory.parents))):
        return None, "intent_directory_unsafe"
    try:
        intent = intake._read(directory / "intent.json", "intent_digest")
        progression = intake._read(directory / "progression.json", "progression_digest")
    except (OSError, ValueError):
        return None, "intent_unreadable"
    queued = ((progression.get("state") or {}).get("scene_preparation_completion") or {}).get("team_scene_request")
    if (intent.get("intent_id") != request["intent_id"] or intent.get("intent_digest") != request["intent_digest"]
            or progression.get("intent_digest") != request["intent_digest"]
            or not isinstance(queued, Mapping) or queued.get("request_digest") != request["request_digest"]):
        return None, "intent_binding_mismatch"
    return {"intent": intent, "directory": directory}, None


def packet_inputs(registry: Mapping[str, Any]) -> dict[str, Any] | None:
    """The operator's fixed compile inputs (``derived_profiles.packet_inputs``): exactly four fields, or None."""

    from .team_candidate_registry_reader import TeamCandidateRegistryError, _operator_file

    settings = registry.get("derived_profiles")
    block = settings.get("packet_inputs") if isinstance(settings, Mapping) else None
    if block is None:
        return None
    if not isinstance(block, Mapping) or set(block) != set(PACKET_INPUT_FIELDS):
        raise TeamCandidateRegistryError("team_scene_packet_inputs_invalid")
    output = Path(str(block.get("output_root") or ""))
    if (not _DIGEST.fullmatch(str(block.get("droid_profile_digest") or "")) or not output.is_absolute()
            or output.is_symlink()):
        raise TeamCandidateRegistryError("team_scene_packet_inputs_invalid")
    files = {field: _operator_file(Path(str(block[field])), code="team_scene_packet_inputs_not_operator_owned")
             for field in ("droid_profile_path", "runtime_source_receipt_path")}
    return {**files, "droid_profile_digest": str(block["droid_profile_digest"]), "output_root": output}


def configured_scene_bundle(result: Mapping[str, Any]) -> Path:
    """The configured bundle published beside the launch's revision."""

    revision_path = Path(str(result.get("configured_scene_revision_path") or ""))
    bundle = revision_path.parent / CONFIGURED_BUNDLE_FILENAME
    if not revision_path.is_absolute() or bundle.is_symlink() or not bundle.is_file():
        raise TeamScenePacketRequestRefused("team_scene_configured_bundle_unavailable")
    return bundle


def unscorable(path: Any) -> str | None:
    """Why the half at ``path`` can never be scored from its own spawn, or None (also when it does not read)."""

    from .rigid_support_pad import half_support_hold
    from .rigid_support_seating import start_spawn_mismatch
    from .scene_packet import half_plan

    try:
        plan = half_plan(str(path)) if path else None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return None if plan is None else (start_spawn_mismatch(plan) or half_support_hold(path))


def _reusable_half(row: Mapping[str, Any] | None, revision: str) -> str | None:
    """The capture's half at ``revision`` when its row names one that still reads and can be scored."""

    from .scene_packet import read_half

    path = str((row or {}).get("team_scene_inputs_path") or "")
    if not path or (row or {}).get("scene_revision_digest") != revision:
        return None
    try:
        read_half(path)
    except (OSError, ValueError, KeyError):
        return None
    return None if unscorable(path) is not None else path


def _discard_unreferenced(attempt_root: Any, capture_root: Any) -> None:
    """Remove an attempt directory unless the capture's derived row names it (an unknown row keeps it)."""

    from .team_candidate_quick10_registry import profile_for_capture

    attempt = Path(str(attempt_root or ""))
    if (not attempt.is_absolute() or not _ATTEMPT.fullmatch(attempt.name)
            or not _CAPTURE_KEY.fullmatch(attempt.parent.name)
            or any(part.is_symlink() for part in (attempt, *attempt.parents)) or not attempt.is_dir()):
        return
    try:
        row = profile_for_capture(str(capture_root)) if capture_root else None
    except (OSError, ValueError):
        return
    named = str((row or {}).get("team_scene_inputs_path") or "")
    if not (named and Path(named).is_relative_to(attempt)):
        shutil.rmtree(attempt, ignore_errors=True)


def _compile(root: Path, request: Mapping[str, Any], status: Mapping[str, Any], *, capture: Path,
             inputs: Mapping[str, Any], scope: Mapping[str, Any] | None, keep_assumed: bool,
             fetcher: Fetcher | None) -> dict[str, Any]:
    """Compile a new attempt directory for the request, record its row and queue the check; the status fields."""

    from .scene_packet import prepare
    from .task_evaluation_configured_controls_source_evidence import _validate_source

    run = Path(request["source_launch_root"])
    if run.is_symlink() or not run.is_dir():
        from .scene_run_cold_storage import COLD_STORAGE, in_cold_storage

        raise TeamScenePacketRequestRefused(COLD_STORAGE if in_cold_storage(run)
                                            else "team_scene_source_launch_unavailable")
    result, _receipt, _zero = _validate_source(run)
    if result.get("configured_scene_revision_digest") != request["scene_revision_digest"]:
        raise TeamScenePacketRequestRefused("team_scene_source_launch_revision_changed")
    bundle = configured_scene_bundle(result)
    parent = Path(inputs["output_root"]) / capture_key(capture)
    attempt = parent / f"{request['scene_revision_digest'][7:23]}-{secrets.token_hex(4)}"
    # Named before it exists, so an interrupted attempt is found and removed on the next run.
    _write(root / "status" / f"{status['request_key']}.json", {**status, "status": "processing",
                                                               "attempt_root": str(attempt)})
    parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    attempt.mkdir(mode=0o750)
    try:
        prepared = prepare(run_root=run, capture_root=capture, scenario_id=SCENARIO_ID, output_root=attempt,
                           fetcher=fetcher, configured_scene_bundle_path=bundle,
                           droid_profile_path=inputs["droid_profile_path"],
                           droid_profile_digest=inputs["droid_profile_digest"],
                           runtime_source_receipt_path=inputs["runtime_source_receipt_path"],
                           owner_scope=scope, assume_unconfirmed_thresholds=True,
                           keep_assumed_thresholds=keep_assumed)
        written = dict(prepared.get("registry") or {})
        if written.get("status") != "written":
            raise TeamScenePacketRequestRefused("team_scene_derived_profile_" + str(written.get("status")
                                                                                     or "not_written"))
    except Exception:
        # Nothing names this attempt yet (the derived row is the last write), so it goes.
        shutil.rmtree(attempt, ignore_errors=True)
        raise
    return {"outcome": "compiled", "attempt_root": str(attempt), "task_id": prepared["task_id"],
            "team_scene_inputs_path": prepared["path"], "team_scene_inputs_digest": prepared["inputs_digest"],
            "derived_profile": {key: written[key] for key in ("path", "profile_digest")},
            **({"owner_success_confirmation": {"status": "applied",
                                               "confirmation_digest": prepared["owner_success_confirmation"]}}
               if prepared.get("owner_success_confirmation") else {}),
            **({"owner_success_assumption": dict(prepared["owner_success_assumption"])}
               if prepared.get("owner_success_assumption") else {})}


def _blocker(exc: BaseException) -> str:
    text = str(exc).strip()
    if isinstance(exc, (ValueError, RuntimeError)) and text:
        return text[:200]
    return f"team_scene_packet_failed:{type(exc).__name__}"


def process_request(root: Path, claimed: Path, *, fetcher: Fetcher | None) -> dict[str, Any]:
    """One claimed request to its status (never raised)."""

    from . import website_owner_success_assumption as assumption
    from .team_candidate_quick10_registry import profile_for_capture
    from .team_candidate_registry_reader import _derived_settings, load_registry

    prior = _read_json(root / "status" / claimed.name) or {}
    status: dict[str, Any] = {"schema_version": STATUS_SCHEMA, "request_key": claimed.stem, "scenario_id": SCENARIO_ID,
                              **{field: json.loads(json.dumps(prior[field])) for field in CARRIED_FIELDS
                                 if field in prior}}

    def skipped(reason: str) -> dict[str, Any]:
        return {**status, "status": "skipped", "reason": reason, "blockers": []}

    try:
        request = validated_request(_read_json(claimed))
        if request_key(request) != claimed.stem:
            raise TeamScenePacketRequestRefused("team_scene_request_invalid")
        status.update({key: request[key] for key in ("request_digest", "capture_id", "scene_revision_digest")})
        registry = load_registry()
        if registry is None:
            return skipped("team_candidate_registry_not_configured")
        if _derived_settings(registry) is None:
            return skipped("derived_profiles_not_configured")
        inputs = packet_inputs(registry)
        if inputs is None:
            return skipped("team_scene_packet_inputs_not_configured")
        # Privacy: a withdrawn capture is refused here, before anything compiles.
        capture = resolve_capture_root(request)
        status["capture_root"] = str(capture)
        if any(isinstance(row, Mapping) and row.get("capture_root") == str(capture) for row in registry["profiles"]):
            return skipped("team_candidate_operator_row_present")
        row = profile_for_capture(capture)
        same_revision = (row or {}).get("scene_revision_digest") == request["scene_revision_digest"]
        assumed = assumption.half_assumption((row or {}).get("team_scene_inputs_path")) if same_revision else None
        existing = _reusable_half(row, request["scene_revision_digest"])
        if existing is not None:
            status.update(outcome="already_compiled", team_scene_inputs_path=existing)
            if assumed is not None:
                status[assumption.FIELD] = assumed
        else:
            scope, unbound = owner_scope(request)
            if unbound is not None:
                status["owner_success_confirmation"] = {"status": "intent_unbound", "reason": unbound}
            status.update(_compile(root, request, status, capture=capture, inputs=inputs, scope=scope,
                                   keep_assumed=assumed is not None, fetcher=fetcher))
        if isinstance(status.get(assumption.FIELD), Mapping):
            # The site's answers an earlier status kept for this same assumed contract stay with it.
            status[assumption.FIELD] = assumption.carried(status[assumption.FIELD], prior.get(assumption.FIELD))
    except Exception as exc:  # noqa: BLE001 - every request ends in a typed status; the queue keeps moving
        blocker = _blocker(exc)
        blocked = {**status, "status": "blocked", "blockers": [blocker]}
        if blocker.startswith("team_scene_packet_failed:"):
            from .bounded_error import error_description

            blocked["error"] = error_description(exc)
        return blocked
    return {**status, "status": "completed", "blockers": []}


def transient_cause(status: Mapping[str, Any]) -> str | None:
    """The single blocker of a blocked status that a later run can clear by itself, or None."""

    blockers = status.get("blockers")
    if status.get("status") != "blocked" or not isinstance(blockers, list) or len(blockers) != 1:
        return None
    blocker = str(blockers[0])
    if blocker in TRANSIENT_BLOCKERS:
        return blocker
    prefix = "team_scene_packet_failed:"
    kind = getattr(builtins, blocker[len(prefix):], None) if blocker.startswith(prefix) else None
    return blocker if isinstance(kind, type) and issubclass(kind, OSError) else None


def refusal_retry_cause(status: Mapping[str, Any]) -> str | None:
    """The blocker the shared refusal schedule retries, or None.

    None for a transient block (its own backoff), an owner-contract refusal (the site's
    confirmation retries it) and any blocker whose words name privacy, rights or money.
    """

    blockers = status.get("blockers")
    if (status.get("status") != "blocked" or not isinstance(blockers, list) or not blockers
            or transient_cause(status) is not None):
        return None
    for blocker in map(str, blockers):
        if blocker.startswith(OWNER_CONTRACT_INVALID) or hard_stop_kind(blocker) is not None:
            return None
    return str(blockers[0])


def _count(record: Any, limit: int) -> int:
    """Retries already used; an unreadable record counts as spent, so the bound always holds."""

    if record is None:
        return 0
    attempts = record.get("attempts") if isinstance(record, Mapping) else None
    return attempts if isinstance(attempts, int) and not isinstance(attempts, bool) and attempts >= 0 else limit


def _transient_delay(attempts: int) -> int | None:
    return AUTOMATIC_RETRY_BACKOFF_SECONDS[attempts] if attempts < len(AUTOMATIC_RETRY_BACKOFF_SECONDS) else None


# (status field, cause, delay for retries done, limit)
_SCHEDULES = ((AUTOMATIC_RETRY, transient_cause, _transient_delay, len(AUTOMATIC_RETRY_BACKOFF_SECONDS)),
              (REFUSAL_RETRY, refusal_retry_cause, refusal_retry_delay, len(REFUSAL_RETRY_DELAYS_SECONDS)))


def _scheduled(status: Mapping[str, Any], now: datetime) -> dict[str, Any]:
    """The status with when its block may run again, or that its timed retries are spent."""

    value = dict(status)
    for field, cause_of, delay_for, limit in _SCHEDULES:
        cause = cause_of(value)
        if cause is None:
            continue
        prior = value.get(field)
        record = {key: item for key, item in (dict(prior) if isinstance(prior, Mapping) else {}).items()
                  if key not in {"not_before", "exhausted"}}
        attempts = _count(prior, limit)
        record.update(attempts=attempts, limit=limit, reason=cause)
        delay = delay_for(attempts)
        if delay is None:
            record["exhausted"] = True
        else:
            record["not_before"] = (now + timedelta(seconds=delay)).isoformat(timespec="seconds")
        value[field] = record
    return value


def _due(record: Mapping[str, Any], now: datetime) -> bool:
    try:
        not_before = datetime.fromisoformat(str(record["not_before"]))
    except (KeyError, ValueError):
        return True
    return not_before.tzinfo is None or not_before <= now


def _listed(status: Mapping[str, Any], field: str) -> list[Any]:
    value = status.get(field)
    return list(value) if isinstance(value, list) else []


def _confirmations_untried(path: Path, status: Mapping[str, Any]) -> list[str]:
    """The site's sealed confirmations of an owner-contract refusal's own intent that no run has tried."""

    blockers = status.get("blockers")
    if (status.get("status") != "blocked" or not isinstance(blockers, list) or len(blockers) != 1
            or not str(blockers[0]).startswith(OWNER_CONTRACT_INVALID)):
        return []
    from .task_evaluation_owner_success_confirmation import confirmations

    try:
        scope, _unbound = owner_scope(validated_request(_read_json(path)))
        rows = [] if scope is None else confirmations(scope["directory"], scope["intent"])
    except (OSError, ValueError, KeyError, TypeError):
        return []
    tried = _listed(status, CONFIRMATION_REQUEUED)
    return [row["confirmation_digest"] for row in rows if row["confirmation_digest"] not in tried]


def _run_restored(path: Path, status: Mapping[str, Any]) -> bool:
    """A request refused only because its scene run was in cold storage, whose run is on disk again."""

    from .scene_run_cold_storage import COLD_STORAGE

    if status.get("status") != "blocked" or status.get("blockers") != [COLD_STORAGE]:
        return False
    run = Path(str((_read_json(path) or {}).get("source_launch_root") or ""))
    return run.is_absolute() and run.is_dir() and not run.is_symlink()


def _site_answer_noted(path: Path, status: Mapping[str, Any]) -> dict[str, Any] | None:
    """An assumed half's status with the site's answer recorded beside the assumption, or None when nothing is new."""

    from .website_owner_success_assumption import noted_status, status_assumed
    from .website_owner_success_confirmation import STATUS_FIELD

    if not status_assumed(status):
        return None
    try:
        return noted_status(status, validated_request(_read_json(path)), exchange=status.get(STATUS_FIELD))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _requeue_reason(path: Path, status: dict[str, Any], release: str | None, now: datetime
                    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Whether a processed request goes back to pending now: (the status to write first, the run's row)."""

    key = path.stem
    blocked = status.get("status") == "blocked" and bool(status.get("blockers"))
    confirmed = [] if transient_cause(status) else _confirmations_untried(path, status)
    if confirmed:
        return ({**status, CONFIRMATION_REQUEUED: [*_listed(status, CONFIRMATION_REQUEUED), *confirmed]},
                {"request_key": key, "reason": "owner_success_confirmation", "confirmation_digests": confirmed})
    if (release is not None and blocked and status.get(RELEASE_COMMIT) != release
            and release not in _listed(status, RELEASE_REQUEUED)):
        return ({**status, RELEASE_REQUEUED: [*_listed(status, RELEASE_REQUEUED), release]},
                {"request_key": key, "reason": "new_release", "release_commit": release,
                 "blocked_release_commit": status.get(RELEASE_COMMIT)})
    if (release is not None and status.get("status") == "completed" and status.get(RELEASE_COMMIT) != release
            and release not in _listed(status, UNSCORABLE_REQUEUED)):
        detail = unscorable(status.get("team_scene_inputs_path"))
        if detail is not None:
            return ({**status, UNSCORABLE_REQUEUED: [*_listed(status, UNSCORABLE_REQUEUED), release]},
                    {"request_key": key, "reason": "unscorable_half", "detail": detail, "release_commit": release})
    if _run_restored(path, status):
        return dict(status), {"request_key": key, "reason": "scene_run_restored"}
    for field, cause_of, delay_for, limit in _SCHEDULES:
        cause = cause_of(status)
        if cause is None:
            continue
        record = status.get(field)
        if field == REFUSAL_RETRY and record is None:
            return None  # blocked before timed retries existed: its clock starts at the next finish
        done = _count(record, limit)
        if delay_for(done) is None or not _due(record if isinstance(record, Mapping) else {}, now):
            return None
        updated = {key: item for key, item in (dict(record) if isinstance(record, Mapping) else {}).items()
                   if key != "not_before"}
        updated.update(attempts=done + 1, limit=limit, reason=cause, requeued_at=now.isoformat(timespec="seconds"))
        return ({**status, field: updated},
                {"request_key": key, "reason": "refusal_retry" if field == REFUSAL_RETRY else cause,
                 "attempt": done + 1})
    return None


def requeue_due(root: Path, now: datetime, release: str | None) -> list[dict[str, Any]]:
    """Return each due request to pending, recording why in its status first (the lock is held)."""

    requeued = []
    for path in sorted((root / "processed").glob("*.json")):
        target = root / "pending" / path.name
        if target.exists():
            continue
        status = _read_json(root / "status" / path.name) or {}
        noted = _site_answer_noted(path, status)
        if noted is not None:
            # Recorded only: the half and its contract stand.
            _write(root / "status" / path.name, noted)
            status = noted
        decided = _requeue_reason(path, status, release, now)
        if decided is None:
            continue
        written, row = decided
        # Counted before the move, so an interruption between the two never adds an uncounted retry.
        _write(root / "status" / path.name, written)
        os.replace(path, target)
        requeued.append(row)
    return requeued


def _finish(root: Path, claimed: Path, status: Mapping[str, Any], *, now: datetime,
            release: str | None) -> dict[str, Any]:
    """Write the final status (with its retry schedule and release), then move the request to processed."""

    value = {key: item for key, item in _scheduled(status, now).items() if key != RELEASE_COMMIT}
    if release is not None:
        value[RELEASE_COMMIT] = release
    _write(root / "status" / claimed.name, value)
    os.replace(claimed, root / "processed" / claimed.name)
    return value


def recover_interrupted(root: Path, claimed: Path, *, now: datetime, release: str | None) -> dict[str, Any]:
    """A request a killed run left in processing: its attempt removed unless named, blocked as interrupted."""

    prior = _read_json(root / "status" / claimed.name) or {}
    _discard_unreferenced(prior.get("attempt_root"), prior.get("capture_root"))
    kept = {key: prior[key] for key in ("request_digest", "capture_id", "scene_revision_digest", "capture_root",
                                        *CARRIED_FIELDS) if key in prior}
    return _finish(root, claimed, {"schema_version": STATUS_SCHEMA, "request_key": claimed.stem, **kept,
                                   "scenario_id": SCENARIO_ID, "status": "blocked",
                                   "blockers": ["team_scene_packet_interrupted"]}, now=now, release=release)


def process_queue(*, queue_root: str | Path, max_messages: int = 1, fetcher: Fetcher | None = None
                  ) -> dict[str, Any]:
    """Recover, requeue what is due, then take up to ``max_messages`` pending requests to a final status."""

    if not isinstance(max_messages, int) or isinstance(max_messages, bool) or not 1 <= max_messages <= 16:
        raise TeamScenePacketRequestRefused("team_scene_packet_max_messages_invalid")
    root = _queue(queue_root)
    release = running_release()
    results = []
    with _locked(root):
        # The lock is held for the whole run, so anything still in processing/ was interrupted.
        now = _now()
        recovered = [recover_interrupted(root, path, now=now, release=release)
                     for path in sorted((root / "processing").glob("*.json"))]
        requeued = requeue_due(root, now, release)
        for pending in sorted((root / "pending").glob("*.json"))[:max_messages]:
            claimed = root / "processing" / pending.name
            try:
                os.replace(pending, claimed)
            except FileNotFoundError:
                continue
            status = process_request(root, claimed, fetcher=fetcher)
            results.append(_finish(root, claimed, status, now=_now(), release=release))
    return {"schema_version": RUN_SCHEMA, "status": "processed" if results or recovered else "idle",
            "processed_count": len(results), "results": results, "recovered": recovered,
            "automatic_requeue": requeued, "provider_mutation_performed": False, "paid_execution_requested": False}


def _record_exchanges(root: Path, outcomes: Mapping[str, Mapping[str, Any]]) -> None:
    """Keep each request's latest website exchange in its status while its thresholds are still unconfirmed."""

    from .website_owner_success_confirmation import STATUS_FIELD, _unconfirmed

    for key, outcome in outcomes.items():
        path = root / "status" / f"{key}.json"
        status = _read_json(path)
        if (status is None or not _unconfirmed(status) or not (root / "processed" / f"{key}.json").exists()
                or status.get(STATUS_FIELD) == dict(outcome)):
            continue
        _write(path, {**status, STATUS_FIELD: dict(outcome)})


def recover(*, queue_root: str | Path, website_transport: Callable[..., Any] | None = None) -> dict[str, Any]:
    """The timer's pass: the site-confirmation exchange, then recovery and requeue; never claims a request.

    The exchange runs without the lock (it reads statuses; a confirmation is written under the
    intake's own lock) and never stops the pass. When a run holds the queue, nothing else is done:
    that run recovers and requeues the same way.
    """

    from .website_owner_success_confirmation import exchange_owner_decisions

    root = _queue(queue_root)
    release = running_release()
    error = None
    try:
        exchanged = exchange_owner_decisions(root, transport=website_transport)
    except Exception as exc:  # noqa: BLE001 - a pass never stops on the website exchange
        from .bounded_error import error_description

        exchanged, error = {}, error_description(exc)
    with _locked(root, wait=False) as locked:
        if not locked:
            return {"schema_version": RUN_SCHEMA, "status": "busy", "requeued": [], "recovered": [],
                    "website_owner_decisions": exchanged, "provider_mutation_performed": False,
                    "paid_execution_requested": False}
        now = _now()
        recovered = [recover_interrupted(root, path, now=now, release=release)["request_key"]
                     for path in sorted((root / "processing").glob("*.json"))]
        _record_exchanges(root, exchanged)
        requeued = requeue_due(root, now, release)
    return {"schema_version": RUN_SCHEMA, "status": "requeued" if requeued else "idle", "requeued": requeued,
            "recovered": recovered, "website_owner_decisions": exchanged,
            **({"website_owner_decisions_error": error} if error else {}),
            "provider_mutation_performed": False, "paid_execution_requested": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queue-root", default=os.environ.get(QUEUE_ROOT_ENV, ""))
    parser.add_argument("--max-messages", type=int, default=1)
    parser.add_argument("--recover", action="store_true",
                        help="run the site-confirmation exchange, recovery and requeue; claim nothing")
    args = parser.parse_args(argv)
    if not args.queue_root:
        print(json.dumps({"schema_version": RUN_SCHEMA, "status": "blocked",
                          "blockers": ["team_scene_packet_queue_root_not_configured"]}, sort_keys=True))
        return 2
    result = (recover(queue_root=args.queue_root) if args.recover
              else process_queue(queue_root=args.queue_root, max_messages=args.max_messages))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
