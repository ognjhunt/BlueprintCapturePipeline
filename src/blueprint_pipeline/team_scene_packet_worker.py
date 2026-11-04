"""Compile a prepared website scene's team half without a person, and republish its job's offer.

Beta journey step: L5 team policy evaluation. A robot team can run on a website
scene only once the scene's own half of the evaluation is compiled
(``team_scene_packet``), recorded where the team registry reads it, and the
job's offer is republished with it. That took an operator running
``python -m blueprint_pipeline.team_scene_packet`` with eight arguments for
every scene, so no website scene reached a team without a person.

Scene preparation now enqueues one small sealed request here once it has
tried to retain the scene's robot-free readiness receipt, unless that receipt
is a decided failure no team can be admitted past
(``task_evaluation_scene_preparation_completion``; that join never compiles).
A receipt that could not be retained does not hold the compile, which never
reads it; the team gate still refuses every team until one passes. This
no-spend worker, woken by its path unit, does the rest on the CPU:

- the capture is resolved under the agent-run partition, the root the executor
  admits, and must be the website capture whose retained task context the
  scene's task was derived from; a withdrawn capture is refused;
- an operator row for the capture always wins, as does a derived row holding
  the operator's frozen internal pair: nothing is compiled or written;
- a derived row that already holds this capture's scene half at this revision
  is reused and only the offer is republished;
- otherwise the launch's published revision is compiled with the configured
  scene bundle the scene configuration retained beside that revision, and with
  what only the operator fixes, in the registry's ``derived_profiles.packet_inputs``
  block: the DROID profile (path and digest), the runtime-source receipt, and
  the root compiled halves are written under. A registry without that block
  keeps this off, by name; an invalid block refuses. Every team body the
  derived defaults register for screening is compiled with the scene too, so
  each gets its own station (``registered_embodiments``), chosen with its own
  arm where its registered robot model is intact (``registered_embodiment_models``);
- a website job's ``proposal_only`` success thresholds compile as the owner's
  only through the owner's sealed confirmation under the request's own intent
  (``owner_success_confirmation_scope``: the intent must be the request's, and
  its progression must have queued this request); otherwise the owner contract
  refuses as before. The owner confirms either through the operator door or on
  the website job page: the timer's pass sends each such blocked job's
  thresholds to the WebApp and records the owner's website confirmation with
  the door's own writer (``website_owner_success_confirmation``).
- a compiled rigid half no ranked stance carries from asks the site owner, on
  the same timer pass, first whether to measure the scene's movable furniture
  (chairs, stools) when it plausibly blocks the carry, with the most that would
  cost (``robot_access_measurement``), then whether to move the items that would
  give the Franka room (``robot_access_furniture``). Each approval, sealed under
  the request's intent, returns the completed request to pending once. A
  measurement approval is the one spend this worker makes: the analysis call and
  SAM masks for just those items, reserved through the job's preparation-spend
  path within the approved figure, kept as a sidecar of the intent's current
  preparation. A move approval rebuilds the half without those items on a
  derived collider. A decline holds the request by name in its status
  (``website_robot_access``). No decision is no change.

The scenario is ``nominal``: the website journey never names one (the WebApp
copies the offer's scenario into the run request it prepares), and a team
evaluation runs the scene's nominal cell and its hidden variations
(``team_candidate_quick10_registry.NOMINAL_CELL``), not a capture's episode specs.

Every attempt compiles into a new directory of its own. The derived row,
written by rename, is the only pointer to it, so a reader never sees a partial
half, and a failed or interrupted attempt's directory is removed unless a row
names it. Every request ends in a status file: ``completed``, ``skipped`` with
a reason, or ``blocked`` with typed blockers. A block a later run can clear by
itself (an interrupted or timed-out run, a failed offer post, a failed fetch,
an operating-system error) goes back to pending automatically, at most
``len(AUTOMATIC_RETRY_BACKOFF_SECONDS)`` times and each only after its wait;
the status keeps the attempt count and cause under ``automatic_retry``. This
worker's own pass and the screening timer's pass both do it
(``requeue_transient_blocked``). An owner-contract refusal goes back once per
new sealed owner confirmation of its intent, on the same passes. Every status
records the release that wrote it (``release_commit``, the commit of the
release checkout this module runs from), so any other deterministic refusal
goes back once per newly deployed release, on the same passes: a deploy that
fixes its cause re-drives it, and the status keeps the releases already tried
(``release_requeued``), so a refusal never loops. Without a known running
release nothing is requeued on that basis; ``--requeue-unfinished`` retries the
rest once their cause is fixed. A completed rigid half whose own plan spawns its
task object farther from the task's start than the scorer's reset tolerance
(``rigid_support_seating.start_spawn_mismatch``; capture-447ae4f4: 14.7 mm), or
whose support record holds it (the CPU rest check refused it, or it was compiled
before the support was modelled: ``rigid_support_pad.half_support_hold``), can
never be scored, so each newly deployed release recompiles it once, on the same
passes and the CPU, and never reuses it (``unscorable_half_release_requeued``). Nothing
here allocates a provider, starts a run or spends, except the owner-approved furniture
measurement above: the offer publishes facts, never authority, and every rights,
consent and spend gate after it is unchanged.
"""

from __future__ import annotations

import argparse
import builtins
import fcntl
import hashlib
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

REQUEST_SCHEMA = "blueprint.team_scene_packet_request.v1"
STATUS_SCHEMA = "blueprint.team_scene_packet_status.v1"
RUN_SCHEMA = "blueprint.team_scene_packet_queue_run.v1"
QUEUE_ROOT_ENV = "BLUEPRINT_TEAM_SCENE_PACKET_QUEUE_ROOT"
# The root the agent-run executor admits captures under (agent_run_executor._row_capture_root).
CAPTURE_PARTITION_ENV = "BLUEPRINT_AGENT_RUN_CAPTURE_PARTITION_ROOT"
SCENARIO_ID = "nominal"
# A website scene's task is "website-" plus its task context digest (website_task_preparation).
WEBSITE_TASK_PREFIX = "website-"
PACKET_INPUT_FIELDS = ("droid_profile_path", "droid_profile_digest", "runtime_source_receipt_path", "output_root")
# Published beside the revision (task_evaluation_scene_configuration_publication), where
# configured-controls autostart also reads it; materialize_team_scene re-checks its digest.
CONFIGURED_BUNDLE_FILENAME = "configured_scene_bundle.v1.zip"
REQUEST_FIELDS = frozenset({"schema_version", "intent_id", "intent_digest", "capture_id", "task_id",
                            "source_launch_root", "launch_receipt_digest", "scene_revision_digest",
                            "robot_free_readiness_status", "request_digest"})
QUEUE_STATES = ("pending", "processing", "processed", "status")
# A request whose scene preparation retained a decided failed probe receipt: never compiled, only read by the
# screening pass to hold the website job by name.
HOLDS_DIRNAME = "holds"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_ATTEMPT = re.compile(r"[0-9a-f]{16}-[0-9a-f]{8}")
_CAPTURE_KEY = re.compile(r"[0-9a-f]{32}")
Fetcher = Callable[[Mapping[str, Any]], bytes]
# Blocks a later run can clear by itself: a run killed mid-compile (a unit timeout included), a WebApp outage on
# the offer post, a failed object-store fetch. An operating-system error (``team_scene_packet_failed:<OSError>``)
# is one too. Every other blocker is a deterministic refusal and stays blocked.
TRANSIENT_BLOCKERS = frozenset({"team_scene_packet_interrupted", "team_scene_offer_publication_failed",
                                "configured_controls_deferred_fetch_failed"})
# The wait before each automatic retry, in order; once all are used the request stays blocked.
AUTOMATIC_RETRY_BACKOFF_SECONDS = (300, 1800, 7200)
# An owner-contract refusal a later run can clear only once the owner confirms the job's thresholds
# (``task_evaluation_owner_success_confirmation``): each new sealed confirmation returns it to pending once,
# and the status keeps the confirmations already tried, so a refusal never loops.
OWNER_CONTRACT_BLOCKER_PREFIX = "team_scene_owner_contract_invalid:"
OWNER_CONFIRMATION_REQUEUE_FIELD = "owner_success_confirmation_requeued"
# Any other deterministic refusal a new release may have fixed: each status names the release that wrote it,
# each newly deployed release returns the refusal to pending once, and the status keeps the releases tried.
RELEASE_COMMIT_FIELD = "release_commit"
RELEASE_REQUEUE_FIELD = "release_requeued"
# A completed rigid half whose own plan spawns the task object farther from the task's start than the scorer's
# reset tolerance can never be scored (capture-447ae4f4: 14.7 mm). Each newly deployed release recompiles it once,
# on the CPU, and the status keeps the releases already tried, so a half that stays unscorable never loops.
UNSCORABLE_HALF_REQUEUE_FIELD = "unscorable_half_release_requeued"
# A completed half that proposes moving named furniture out of the Franka's way is compiled again once per
# sealed approval of that exact proposal by the site owner (``robot_access_furniture``).
ROBOT_ACCESS_REQUEUE_FIELD = "robot_access_confirmation_requeued"
_COMMIT = re.compile(r"[0-9a-f]{40}")


class TeamScenePacketRequestRefused(ValueError):
    """This request cannot become a team scene half; the message names why."""


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    sealed = {**value, field: ""}
    sealed[field] = canonical_digest(sealed, digest_field=field)
    return sealed


def _read_json(path: Path) -> dict[str, Any] | None:
    if path.is_symlink():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _temporary(directory: Path, value: Mapping[str, Any]) -> Path:
    # Never named *.json, so no queue glob can see a half-written file.
    with tempfile.NamedTemporaryFile("w", dir=directory, prefix=".team-scene-", suffix=".tmp",
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


def _write_replace(path: Path, value: Mapping[str, Any]) -> None:
    os.replace(_temporary(path.parent, value), path)


def _write_new(path: Path, value: Mapping[str, Any], *, staging: Path) -> bool:
    """Write ``path`` only if absent: staged outside its directory, then linked in whole."""

    temporary = _temporary(staging, value)
    try:
        os.link(temporary, path)
    except FileExistsError:
        return False
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _queue(queue_root: str | Path) -> Path:
    root = Path(queue_root)
    if not root.is_absolute() or any(part.is_symlink() for part in (root, *root.parents)):
        raise TeamScenePacketRequestRefused("team_scene_packet_queue_root_unsafe")
    for state in QUEUE_STATES:
        (root / state).mkdir(parents=True, exist_ok=True, mode=0o750)
    return root


@contextmanager
def _locked(root: Path, *, wait: bool = True) -> Iterator[bool]:
    """Hold the queue lock; with ``wait=False`` yield False at once when a run already holds it."""

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


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def transient_cause(status: Mapping[str, Any]) -> str | None:
    """The one blocker of a blocked status that a later run can clear by itself, or None."""

    blockers = status.get("blockers")
    if status.get("status") != "blocked" or not isinstance(blockers, list) or len(blockers) != 1:
        return None
    blocker = str(blockers[0])
    if blocker in TRANSIENT_BLOCKERS:
        return blocker
    prefix = "team_scene_packet_failed:"
    kind = getattr(builtins, blocker[len(prefix):], None) if blocker.startswith(prefix) else None
    return blocker if isinstance(kind, type) and issubclass(kind, OSError) else None


def _retry_attempts(retry: Mapping[str, Any]) -> int:
    attempts = retry.get("attempts", 0)
    # An unreadable count is treated as spent, so the bound always holds.
    valid = isinstance(attempts, int) and not isinstance(attempts, bool) and attempts >= 0
    return attempts if valid else len(AUTOMATIC_RETRY_BACKOFF_SECONDS)


def _with_retry(status: Mapping[str, Any], *, now: datetime) -> dict[str, Any]:
    """Name when a transient block may run again, or that its automatic retries are spent."""

    cause = transient_cause(status)
    if cause is None:
        return dict(status)
    retry = {key: value for key, value in dict(status.get("automatic_retry") or {}).items()
             if key not in {"not_before", "exhausted"}}
    attempts = _retry_attempts(retry)
    retry.update(attempts=attempts, limit=len(AUTOMATIC_RETRY_BACKOFF_SECONDS), reason=cause)
    if attempts < len(AUTOMATIC_RETRY_BACKOFF_SECONDS):
        retry["not_before"] = (now + timedelta(seconds=AUTOMATIC_RETRY_BACKOFF_SECONDS[attempts])).isoformat(
            timespec="seconds")
    else:
        retry["exhausted"] = True
    return {**status, "automatic_retry": retry}


def _due(retry: Mapping[str, Any], now: datetime) -> bool:
    try:
        not_before = datetime.fromisoformat(str(retry["not_before"]))
    except (KeyError, ValueError):
        return True  # A block recorded before automatic retries existed is due at once.
    return not_before.tzinfo is None or not_before <= now


def _owner_confirmations_untried(path: Path, status: Mapping[str, Any]) -> list[str]:
    """The owner's sealed confirmations of an owner-contract refusal's own intent not yet tried, else none.

    Any valid confirmation of the intent's owner counts here; whether it binds this exact task is decided
    only by the compile (``team_scene_packet._owner_confirmed``).
    """

    blockers = status.get("blockers")
    if (status.get("status") != "blocked" or not isinstance(blockers, list) or len(blockers) != 1
            or not str(blockers[0]).startswith(OWNER_CONTRACT_BLOCKER_PREFIX)):
        return []
    from .task_evaluation_owner_success_confirmation import confirmations

    try:
        request = validated_request(_read_json(path))
        scope, _unbound = owner_success_confirmation_scope(request)
        rows = [] if scope is None else confirmations(scope["directory"], scope["intent"])
    except (OSError, ValueError, KeyError, TypeError):
        return []
    tried = status.get(OWNER_CONFIRMATION_REQUEUE_FIELD)
    tried = tried if isinstance(tried, list) else []
    return [row["confirmation_digest"] for row in rows if row["confirmation_digest"] not in tried]


def current_release_commit() -> str | None:
    """The release this module runs from, or None when it cannot be told.

    Units ``cd -P`` into ``BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_REPO``, a detached release worktree named by
    its commit, and import this module from there; the worker's own pass and the screening timer's pass both do,
    so each reads the same identity from this file: the checkout's exact ``HEAD``
    (``task_evaluation_release_identity``), else the release directory's own commit name.
    """

    try:
        from .task_evaluation_release_identity import running_release_commit

        commit = running_release_commit(__file__)
        if not _COMMIT.fullmatch(commit or ""):
            commit = Path(__file__).resolve().parents[2].name
    except (OSError, ValueError, IndexError):
        return None
    return commit if _COMMIT.fullmatch(commit) else None


def _release(release: Any) -> str | None:
    return release if isinstance(release, str) and _COMMIT.fullmatch(release) else None


def _new_release_untried(status: Mapping[str, Any], release: str | None) -> bool:
    """A blocked request (any cause but an owner-contract refusal) written by another release, or by none
    recorded, that this known release has not yet returned to pending.

    A new release may fix any cause, so it gets one try at once, without waiting out a transient block's
    backoff or after its automatic retries are spent: capture-447ae4f4's compile, killed three times for
    host memory on 2026-11-03, otherwise waited two hours for its last retry after the memory fix shipped.
    The transient backoff still governs retries under the same release.
    """

    blockers = status.get("blockers")
    if (release is None or status.get("status") != "blocked" or not isinstance(blockers, list) or not blockers
            or any(str(blocker).startswith(OWNER_CONTRACT_BLOCKER_PREFIX) for blocker in blockers)
            or status.get(RELEASE_COMMIT_FIELD) == release):
        return False
    tried = status.get(RELEASE_REQUEUE_FIELD)
    return release not in (tried if isinstance(tried, list) else [])


def final_block(status: Mapping[str, Any], *, release: str | None) -> bool:
    """A blocked status that no pass under ``release`` returns to pending, so the scene's team half stays uncompiled.

    Not final: a transient block with automatic retries left, an owner-contract refusal (the owner's sealed
    confirmation returns it), one a newly deployed release has not yet tried, or any block when the running
    release cannot be told. The screening pass names a final one to the website job
    (``beta_policy_screening_holds``).
    """

    blockers = status.get("blockers")
    if (status.get("status") != "blocked" or not isinstance(blockers, list) or not blockers
            or any(str(blocker).startswith(OWNER_CONTRACT_BLOCKER_PREFIX) for blocker in blockers)
            or _release(release) is None or _new_release_untried(status, release)):
        return False
    retry = status.get("automatic_retry")
    return (transient_cause(status) is None
            or _retry_attempts(retry if isinstance(retry, Mapping) else {}) >= len(AUTOMATIC_RETRY_BACKOFF_SECONDS))


def unscorable_half(path: Any) -> str | None:
    """Why the compiled half at ``path`` can never be scored from its own spawn, or None (also when unreadable).

    Its object spawns off its start (``rigid_support_seating.start_spawn_mismatch``), or its support
    record holds it: the CPU rest check refused it, or a rigid half has no record because a release
    before the support was modelled compiled it (``rigid_support_pad.half_support_hold``).
    """

    from .rigid_support_pad import half_support_hold
    from .rigid_support_seating import start_spawn_mismatch
    from .team_scene_packet import read_team_scene_inputs

    try:
        if not path:
            return None
        inputs = read_team_scene_inputs(str(path))
        plan = _read_json(Path(inputs["base_native_packet"]["path"]).parent / "native_task_arena_scene_plan.v1.json")
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if plan is None:
        return None
    return start_spawn_mismatch(plan) or half_support_hold(path)


def _unscorable_half_untried(status: Mapping[str, Any], release: str | None) -> str | None:
    """A completed half this known release has not yet recompiled, that no episode of can be scored."""

    tried = status.get(UNSCORABLE_HALF_REQUEUE_FIELD)
    if (release is None or status.get("status") != "completed" or status.get(RELEASE_COMMIT_FIELD) == release
            or release in (tried if isinstance(tried, list) else [])):
        return None
    return unscorable_half(status.get("team_scene_inputs_path"))


def _robot_access_approvals_untried(path: Path, status: Mapping[str, Any]) -> list[str]:
    """The site owner's sealed approvals of a completed half's own robot-access proposal not yet compiled."""

    if status.get("status") != "completed":
        return []
    from .robot_access_furniture import untried_confirmations

    try:
        return untried_confirmations(status, validated_request(_read_json(path)))
    except (OSError, ValueError, KeyError, TypeError):
        return []


def _requeue_due(root: Path, now: datetime, release: str | None = None) -> list[dict[str, Any]]:
    """Return each due transient block, each owner-contract refusal the owner has since confirmed, and each
    other deterministic refusal once per newly deployed ``release``, to pending, recording the attempt first
    (the lock is held)."""

    release = _release(release)
    requeued: list[dict[str, Any]] = []
    for path in sorted((root / "processed").glob("*.json")):
        status = _read_json(root / "status" / path.name) or {}
        cause = transient_cause(status)
        target = root / "pending" / path.name
        confirmed = [] if cause is not None or target.exists() else _owner_confirmations_untried(path, status)
        if confirmed:
            tried = status.get(OWNER_CONFIRMATION_REQUEUE_FIELD)
            tried = [*(tried if isinstance(tried, list) else []), *confirmed]
            # Recorded before the move, so each confirmation is tried at most once.
            _write_replace(root / "status" / path.name, {**status, OWNER_CONFIRMATION_REQUEUE_FIELD: tried})
            os.replace(path, target)
            requeued.append({"request_key": path.stem, "reason": "owner_success_confirmation",
                             "confirmation_digests": confirmed})
            continue
        if not target.exists() and _new_release_untried(status, release):
            tried = status.get(RELEASE_REQUEUE_FIELD)
            # Recorded before the move, so each release is tried at most once.
            _write_replace(root / "status" / path.name, {
                **status, RELEASE_REQUEUE_FIELD: [*(tried if isinstance(tried, list) else []), release]})
            os.replace(path, target)
            requeued.append({"request_key": path.stem, "reason": "new_release", "release_commit": release,
                             "blocked_release_commit": status.get(RELEASE_COMMIT_FIELD)})
            continue
        approvals = [] if target.exists() else _robot_access_approvals_untried(path, status)
        if approvals:
            tried = status.get(ROBOT_ACCESS_REQUEUE_FIELD)
            # Recorded before the move, so each approval is compiled at most once.
            _write_replace(root / "status" / path.name, {
                **status, ROBOT_ACCESS_REQUEUE_FIELD: [*(tried if isinstance(tried, list) else []), *approvals]})
            os.replace(path, target)
            requeued.append({"request_key": path.stem, "reason": "robot_access_confirmation",
                             "confirmation_digests": approvals})
            continue
        unscorable = None if target.exists() else _unscorable_half_untried(status, release)
        if unscorable is not None:
            tried = status.get(UNSCORABLE_HALF_REQUEUE_FIELD)
            # Recorded before the move, so each release recompiles it at most once.
            _write_replace(root / "status" / path.name, {
                **status, UNSCORABLE_HALF_REQUEUE_FIELD: [*(tried if isinstance(tried, list) else []), release]})
            os.replace(path, target)
            requeued.append({"request_key": path.stem, "reason": "unscorable_half", "detail": unscorable,
                             "release_commit": release})
            continue
        retry = dict(status.get("automatic_retry") or {})
        attempts = _retry_attempts(retry)
        if (cause is None or target.exists() or attempts >= len(AUTOMATIC_RETRY_BACKOFF_SECONDS)
                or not _due(retry, now)):
            continue
        retry = {key: value for key, value in retry.items() if key != "not_before"}
        retry.update(attempts=attempts + 1, limit=len(AUTOMATIC_RETRY_BACKOFF_SECONDS), reason=cause,
                     requeued_at=now.isoformat(timespec="seconds"))
        # Counted before the move, so an interruption between the two never adds an uncounted attempt.
        _write_replace(root / "status" / path.name, {**status, "automatic_retry": retry})
        os.replace(path, target)
        requeued.append({"request_key": path.stem, "attempt": attempts + 1, "reason": cause})
    return requeued


def request_key(request: Mapping[str, Any]) -> str:
    """One request per capture and published revision."""

    return canonical_digest({"capture_id": request["capture_id"],
                             "scene_revision_digest": request["scene_revision_digest"]})[7:39]


def capture_key(capture_root: Path) -> str:
    """The name the team registry files this capture's derived row under."""

    return hashlib.sha256(str(capture_root).encode()).hexdigest()[:32]


def validated_request(value: Any) -> dict[str, Any]:
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


def enqueue_team_scene_request(*, intent: Mapping[str, Any], run_root: str | Path,
                               receipt: Mapping[str, Any], result: Mapping[str, Any],
                               readiness: Mapping[str, Any], queue_root: str | Path | None = None
                               ) -> dict[str, Any]:
    """Queue the one request that compiles a completed website scene's team half.

    Called by scene preparation completion once it has tried to retain the
    robot-free readiness receipt. It writes one small sealed file and nothing else: no
    compile, no process, no provider. Only an I/O failure raises (the caller
    retries); everything else is a typed status the completion records.
    """

    raw = str(queue_root or os.environ.get(QUEUE_ROOT_ENV, "")).strip()
    if not raw:
        return {"status": "not_configured"}
    task_id = str(intent["request"]["task"]["task_id"])
    if not task_id.startswith(WEBSITE_TASK_PREFIX):
        return {"status": "not_a_website_scene"}
    # A receipt that could not be retained (a refused score) does not hold the compile: the compile never
    # reads it, and the probe lane that can still decide it is queued only from the compiled half.
    failed = readiness.get("status") == "failed"
    try:
        from .website_preparation_revision_identity import capture_id
        request = validated_request(_seal({
            "schema_version": REQUEST_SCHEMA, "intent_id": str(intent["intent_id"]),
            "intent_digest": str(intent["intent_digest"]), "capture_id": str(capture_id(intent["request"])),
            "task_id": task_id, "source_launch_root": str(Path(run_root).resolve()),
            "launch_receipt_digest": str(receipt.get("receipt_digest") or ""),
            "scene_revision_digest": str(result.get("configured_scene_revision_digest") or ""),
            "robot_free_readiness_status": str(readiness.get("status") or "")}, "request_digest"))
        root = _queue(raw)
    except TeamScenePacketRequestRefused as exc:
        return {"status": "not_enqueued", "reason": "robot_free_probe_failed" if failed else str(exc)}
    key = request_key(request)
    if failed:
        # A decided failure is sealed for this revision: no team can ever be admitted to it, so nothing is
        # compiled. The request is kept under holds/ instead, where the screening pass reads it to tell the
        # website job its evaluation is held, by name (``beta_policy_screening_holds``).
        (root / HOLDS_DIRNAME).mkdir(exist_ok=True, mode=0o750)
        _write_new(root / HOLDS_DIRNAME / f"{key}.json", request, staging=root)
        return {"status": "not_enqueued", "reason": "robot_free_probe_failed", "hold_request_key": key}
    for state in ("pending", "processing", "processed"):
        if (root / state / f"{key}.json").exists():
            return {"status": "already_enqueued", "request_key": key, "state": state}
    # Staged in the queue root, so the watched pending/ only ever sees the whole request appear.
    written = _write_new(root / "pending" / f"{key}.json", request, staging=root)
    return {"status": "enqueued" if written else "already_enqueued", "request_key": key,
            "request_digest": request["request_digest"]}


def packet_inputs(registry: Mapping[str, Any]) -> dict[str, Any] | None:
    """The operator's fixed compile inputs under ``derived_profiles.packet_inputs``; None when absent.

    Exactly the four fields, fail-closed like the derived ``defaults``: the
    DROID profile and runtime-source receipt are operator-owned regular files,
    the profile digest is a sha256 (``materialize_team_scene`` re-checks it
    against the bytes), and the output root is absolute.
    """

    from .team_candidate_quick10_registry import TeamCandidateRegistryError, _operator_file

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


def registered_embodiments(registry: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """The team bodies registered for screening, once each, so the scene also compiles their own stations.

    Each is a ``screening_candidates.<id>.robot_policy.manifest`` under the derived defaults, the rows every
    derived scene's screening reads. One whose bytes changed or that does not validate is named and skipped.
    """

    from .beta_policy_screening_launch import _bound_path, _read
    from .team_candidate_quick10_registry import _derived_settings
    from .team_embodiment import validate_embodiment_manifest

    configured = ((_derived_settings(registry) or {}).get("defaults") or {}).get("screening_candidates") or {}
    manifests: dict[str, dict[str, Any]] = {}
    unreadable: list[str] = []
    for key, settings in sorted(configured.items()):
        registration = settings.get("robot_policy") if isinstance(settings, Mapping) else None
        if not isinstance(registration, Mapping) or not isinstance(registration.get("manifest"), Mapping):
            continue
        try:
            manifest = validate_embodiment_manifest(_read(_bound_path(registration["manifest"])))
        except (OSError, ValueError, KeyError, TypeError):
            unreadable.append(str(key))
            continue
        manifests[manifest["embodiment_digest"]] = manifest
    return [manifests[digest] for digest in sorted(manifests)], unreadable


def registered_embodiment_models(registry: Mapping[str, Any]) -> dict[str, Path]:
    """Each registered body's own robot model, by embodiment digest, so its station is chosen with its own arm.

    A model is kept only when its bytes are still the registration's bound record and that record is the
    manifest's declared model; any other is left out, and that body's station keeps its declared reach.
    """

    from .beta_policy_screening_launch import _bound_path, _read
    from .team_candidate_quick10_registry import _derived_settings
    from .team_embodiment import validate_embodiment_manifest

    configured = ((_derived_settings(registry) or {}).get("defaults") or {}).get("screening_candidates") or {}
    models: dict[str, Path] = {}
    for _key, settings in sorted(configured.items()):
        registration = settings.get("robot_policy") if isinstance(settings, Mapping) else None
        if not isinstance(registration, Mapping) or not all(
                isinstance(registration.get(field), Mapping) for field in ("manifest", "model")):
            continue
        try:
            manifest = validate_embodiment_manifest(_read(_bound_path(registration["manifest"])))
            if registration["model"].get("sha256") != manifest["model"]["sha256"]:
                continue
            models.setdefault(manifest["embodiment_digest"], _bound_path(registration["model"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return models


def resolve_capture_root(request: Mapping[str, Any], *, partition: str | Path | None = None) -> Path:
    """The request's website capture inside the executor's partition, bound to the scene's task."""

    from .website_atlas_preview_store import capture_withdrawn

    raw = str(partition or os.environ.get(CAPTURE_PARTITION_ENV, "")).strip()
    if not raw:
        raise TeamScenePacketRequestRefused("team_scene_capture_partition_not_configured")
    root = Path(raw)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise TeamScenePacketRequestRefused("team_scene_capture_partition_invalid")
    capture_id = str(request["capture_id"])
    matches = sorted(path for path in (root / "scenes").glob(f"*/captures/{capture_id}")
                     if path.is_dir() and not path.is_symlink())
    if len(matches) != 1:
        raise TeamScenePacketRequestRefused(
            "team_scene_capture_root_" + ("missing" if not matches else "ambiguous"))
    capture = matches[0].resolve()
    scene_id = capture.parent.parent.name
    descriptor = _read_json(capture / "capture_descriptor.json") or {}
    if (not capture.is_relative_to(root.resolve()) or descriptor.get("capture_id") != capture_id
            or descriptor.get("scene_id") != scene_id):
        raise TeamScenePacketRequestRefused("team_scene_capture_root_identity_mismatch")
    if capture_withdrawn(capture):
        raise TeamScenePacketRequestRefused("team_scene_capture_withdrawn")
    context = _read_json(capture / "pipeline" / "website_task_context.json") or {}
    digest = str(context.get("context_digest") or "")
    if (context.get("capture_id") != capture_id or context.get("scene_id") != scene_id
            or digest != canonical_digest(context, digest_field="context_digest")
            or request["task_id"] != WEBSITE_TASK_PREFIX + digest[7:27]):
        raise TeamScenePacketRequestRefused("team_scene_website_task_context_mismatch")
    return capture


def owner_success_confirmation_scope(request: Mapping[str, Any], *,
                                    intent_root: str | Path | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """The request's own sealed intent, where an owner's success confirmation is kept, or None and why.

    The request names its intent (id and digest, sealed in its own digest). The
    intent is read from the scene intake root and must be that exact intent, and
    the intent's progression must have queued this exact request; only then can
    a ``proposal_only`` owner authority look for the owner's confirmation there
    (``task_evaluation_owner_success_confirmation``). An unbound request compiles
    as before, so the owner contract still refuses without a confirmation.
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


def configured_scene_bundle(result: Mapping[str, Any]) -> Path:
    """The configured bundle retained beside the launch's published revision."""

    revision_path = Path(str(result.get("configured_scene_revision_path") or ""))
    bundle = revision_path.parent / CONFIGURED_BUNDLE_FILENAME
    if not revision_path.is_absolute() or bundle.is_symlink() or not bundle.is_file():
        raise TeamScenePacketRequestRefused("team_scene_configured_bundle_unavailable")
    return bundle


def _current_half(profile: Mapping[str, Any] | None, revision: str) -> str | None:
    """This capture's compiled scene half at ``revision``, when its row names one that still reads."""

    from .team_scene_packet import read_team_scene_inputs

    path = str((profile or {}).get("team_scene_inputs_path") or "")
    if not path or (profile or {}).get("scene_revision_digest") != revision:
        return None
    try:
        read_team_scene_inputs(path)
    except (OSError, ValueError, KeyError):
        return None
    # A half no episode can be scored from is compiled again, not reused.
    return None if unscorable_half(path) is not None else path


def _discard_unreferenced(attempt_root: Any, capture_root: Any) -> None:
    """Remove a failed attempt's own directory unless a derived row names it (unknown keeps it)."""

    from .team_candidate_quick10_registry import profile_for_capture

    attempt = Path(str(attempt_root or ""))
    if (not attempt.is_absolute() or not _ATTEMPT.fullmatch(attempt.name)
            or not _CAPTURE_KEY.fullmatch(attempt.parent.name)
            or any(part.is_symlink() for part in (attempt, *attempt.parents)) or not attempt.is_dir()):
        return
    try:
        profile = profile_for_capture(str(capture_root)) if capture_root else None
    except (OSError, ValueError):
        return
    named = str((profile or {}).get("team_scene_inputs_path") or "")
    if not (named and Path(named).is_relative_to(attempt)):
        shutil.rmtree(attempt, ignore_errors=True)


def _compile(root: Path, request: Mapping[str, Any], *, status: Mapping[str, Any], capture: Path,
             inputs: Mapping[str, Any], fetcher: Fetcher | None,
             embodiments: Sequence[Mapping[str, Any]] = (),
             models: Mapping[str, Path] | None = None,
             owner_scope: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from .task_evaluation_configured_controls_progression_worker import _validate_source
    from .team_scene_packet import prepare_team_scene

    run = Path(request["source_launch_root"])
    if run.is_symlink() or not run.is_dir():
        raise TeamScenePacketRequestRefused("team_scene_source_launch_unavailable")
    result, _receipt, _zero = _validate_source(run)
    if result.get("configured_scene_revision_digest") != request["scene_revision_digest"]:
        raise TeamScenePacketRequestRefused("team_scene_source_launch_revision_changed")
    bundle = configured_scene_bundle(result)
    parent = Path(inputs["output_root"]) / capture_key(capture)
    attempt = parent / f"{request['scene_revision_digest'][7:23]}-{secrets.token_hex(4)}"
    # Named before it exists, so an interrupted attempt is found and removed on the next run.
    _write_replace(root / "status" / f"{status['request_key']}.json",
                   {**status, "status": "processing", "attempt_root": str(attempt)})
    parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    attempt.mkdir(mode=0o750)
    try:
        prepared = prepare_team_scene(
            run_root=run, capture_root=capture, scenario_id=SCENARIO_ID, output_root=attempt, fetcher=fetcher,
            configured_scene_bundle_path=bundle, droid_profile_path=inputs["droid_profile_path"],
            droid_profile_digest=inputs["droid_profile_digest"],
            runtime_source_receipt_path=inputs["runtime_source_receipt_path"],
            **({"embodiment_manifests": list(embodiments)} if embodiments else {}),
            **({"embodiment_models": dict(models)} if embodiments and models else {}),
            # The request's own intent is also where the site owner's approval to move furniture is kept.
            **({"owner_success_confirmation_scope": owner_scope, "robot_access_scope": owner_scope}
               if owner_scope is not None else {}))
        written = dict(prepared.get("registry") or {})
        if written.get("status") != "written":
            raise TeamScenePacketRequestRefused(
                "team_scene_derived_profile_" + str(written.get("status") or "not_written"))
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
            **({"robot_access_removal": dict(prepared["robot_access_removal"])}
               if prepared.get("robot_access_removal") else {})}


def _measure_furniture(request: Mapping[str, Any], approval: Mapping[str, Any]) -> dict[str, Any]:
    """Run the owner-approved movable furniture measurement (``robot_access_measurement``); its outcome, never raised.

    The only spend this worker makes, and only with the site owner's sealed approval of exactly this
    measurement: the analysis call and the SAM masks reserve through the job's preparation-spend path,
    within the approved figure. A failure is recorded; the half is compiled again either way and asks
    nothing new for the same approval.
    """

    from .robot_access_measurement import run_measurement

    try:
        scope, unbound = owner_success_confirmation_scope(request)
        if scope is None:
            return {"status": "intent_unbound", "reason": unbound}
        sidecar = run_measurement(scope, approval, revision_digest=request["scene_revision_digest"])
    except Exception as exc:  # noqa: BLE001 - a measurement that fails is recorded; the compile goes on
        from .bounded_error import error_description

        return {"status": "failed", "confirmation_digest": approval.get("confirmation_digest"),
                "error": error_description(exc)}
    return {"status": "measured", "confirmation_digest": approval.get("confirmation_digest"),
            "sidecar_digest": sidecar["sidecar_digest"],
            "reserved_cost_usd": round(float(sidecar["analysis"]["reserved_cost_usd"])
                                       + float(sidecar["segmentation"]["reserved_cost_usd"]), 4)}


def _blocker(exc: BaseException) -> str:
    text = str(exc).strip()
    if isinstance(exc, (ValueError, RuntimeError)) and text:
        return text[:200]
    return f"team_scene_packet_failed:{type(exc).__name__}"


def _process(root: Path, claimed: Path, *, fetcher: Fetcher | None,
             offer_transport: Callable[..., Any] | None) -> dict[str, Any]:
    from . import team_candidate_quick10_registry as registry_module
    from .agent_execution_offer import publish_agent_execution_offer

    status: dict[str, Any] = {"schema_version": STATUS_SCHEMA, "request_key": claimed.stem,
                              "scenario_id": SCENARIO_ID, "provider_mutation_performed": False,
                              "paid_execution_requested": False}
    # An automatically retried request keeps its attempt count, and the owner confirmations and releases
    # already tried, whatever this run ends in.
    prior = _read_json(root / "status" / claimed.name) or {}
    retry = prior.get("automatic_retry")
    if isinstance(retry, Mapping):
        status["automatic_retry"] = dict(retry)
    for field in (OWNER_CONFIRMATION_REQUEUE_FIELD, RELEASE_REQUEUE_FIELD, UNSCORABLE_HALF_REQUEUE_FIELD,
                  ROBOT_ACCESS_REQUEUE_FIELD):
        if isinstance(prior.get(field), list):
            status[field] = list(prior[field])

    def skipped(reason: str) -> dict[str, Any]:
        return {**status, "status": "skipped", "reason": reason, "blockers": []}

    try:
        request = validated_request(_read_json(claimed))
        if request_key(request) != claimed.stem:
            raise TeamScenePacketRequestRefused("team_scene_request_invalid")
        status.update({key: request[key] for key in ("request_digest", "capture_id", "scene_revision_digest")})
        registry = registry_module.load_registry()
        if registry is None:
            return skipped("team_candidate_registry_not_configured")
        if registry_module._derived_settings(registry) is None:
            return skipped("derived_profiles_not_configured")
        inputs = packet_inputs(registry)
        if inputs is None:
            return skipped("team_scene_packet_inputs_not_configured")
        capture = resolve_capture_root(request)
        status["capture_root"] = str(capture)
        if any(isinstance(row, Mapping) and row.get("capture_root") == str(capture) for row in registry["profiles"]):
            # The operator's row is what every reader uses; nothing is compiled beside it.
            return skipped("team_candidate_operator_row_present")
        current = registry_module.profile_for_capture(capture)
        if (current or {}).get("pair_runtime_inputs_path"):
            # A frozen internal pair is the operator's choice of scene half, never replaced here.
            return skipped("team_candidate_frozen_pair_present")
        existing = _current_half(current, request["scene_revision_digest"])
        if existing is not None:
            from .robot_access_furniture import MEASURE, approved_proposal

            approval = approved_proposal(existing, request)
            if approval is not None:
                if approval["question"] == MEASURE:
                    # The site owner approved measuring the furniture this half asks about: measured once,
                    # within that approval and the job's caps, before the half is compiled again.
                    status["robot_access_measurement"] = _measure_furniture(request, approval)
                # The site owner answered this half's question: it is compiled again with the answer.
                existing = None
        if existing is not None:
            status.update(outcome="already_compiled", team_scene_inputs_path=existing)
        else:
            embodiments, unreadable = registered_embodiments(registry)
            if unreadable:
                status["embodiment_manifests_unreadable"] = unreadable
            owner_scope, unbound = owner_success_confirmation_scope(request)
            if unbound is not None:
                # Recorded, never a blocker: an unbound request compiles exactly as before.
                status["owner_success_confirmation"] = {"status": "intent_unbound", "reason": unbound}
            status.update(_compile(root, request, status=status, capture=capture, inputs=inputs, fetcher=fetcher,
                                   embodiments=embodiments,
                                   models=registered_embodiment_models(registry) if embodiments else None,
                                   owner_scope=owner_scope))
    except Exception as exc:  # noqa: BLE001 - every request ends in a typed status; the queue keeps moving
        blocker = _blocker(exc)
        if not blocker.startswith("team_scene_packet_failed:"):
            return {**status, "status": "blocked", "blockers": [blocker]}
        from .bounded_error import error_description

        # The type alone hides a full disk or a missing input; keep a redacted, trimmed message.
        return {**status, "status": "blocked", "blockers": [blocker], "error": error_description(exc)}
    published = publish_agent_execution_offer(capture, transport=offer_transport)
    status["offer"] = {"status": published.get("status"), "reason": published.get("reason"),
                       "team_evaluation_listed": bool((published.get("offer") or {}).get("team_evaluation")),
                       "team_candidate_offer_refused": published.get("team_candidate_offer_refused")}
    if published.get("status") == "failed":
        # The half is recorded; a requeue reuses it and only publishes again.
        return {**status, "status": "blocked", "blockers": ["team_scene_offer_publication_failed"]}
    return {**status, "status": "completed", "blockers": []}


def _finish(root: Path, claimed: Path, status: Mapping[str, Any], *, now: datetime,
            release: str | None = None) -> dict[str, Any]:
    status = {key: value for key, value in _with_retry(status, now=now).items() if key != RELEASE_COMMIT_FIELD}
    if _release(release) is not None:
        # The release that wrote this status; a later release returns a deterministic refusal once.
        status[RELEASE_COMMIT_FIELD] = release
    _write_replace(root / "status" / claimed.name, status)
    os.replace(claimed, root / "processed" / claimed.name)
    return status


def _recover_interrupted(root: Path, claimed: Path, *, now: datetime, release: str | None = None
                         ) -> dict[str, Any]:
    prior = _read_json(root / "status" / claimed.name) or {}
    _discard_unreferenced(prior.get("attempt_root"), prior.get("capture_root"))
    kept = {key: prior[key] for key in ("request_digest", "capture_id", "scene_revision_digest", "capture_root",
                                        "automatic_retry", OWNER_CONFIRMATION_REQUEUE_FIELD,
                                        RELEASE_REQUEUE_FIELD, UNSCORABLE_HALF_REQUEUE_FIELD,
                                        ROBOT_ACCESS_REQUEUE_FIELD) if key in prior}
    return _finish(root, claimed, {"schema_version": STATUS_SCHEMA, "request_key": claimed.stem, **kept,
                                   "scenario_id": SCENARIO_ID, "status": "blocked",
                                   "blockers": ["team_scene_packet_interrupted"],
                                   "provider_mutation_performed": False, "paid_execution_requested": False},
                   now=now, release=release)


def process_team_scene_queue(*, queue_root: str | Path, max_messages: int = 2, fetcher: Fetcher | None = None,
                             offer_transport: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Advance up to ``max_messages`` pending requests to a terminal status, on the CPU."""

    if not isinstance(max_messages, int) or isinstance(max_messages, bool) or not 1 <= max_messages <= 16:
        raise TeamScenePacketRequestRefused("team_scene_packet_max_messages_invalid")
    root = _queue(queue_root)
    results: list[dict[str, Any]] = []
    release = current_release_commit()
    with _locked(root):
        # The lock is held for the whole run, so anything still processing was interrupted.
        now = _utc_now()
        recovered = [_recover_interrupted(root, path, now=now, release=release)
                     for path in sorted((root / "processing").glob("*.json"))]
        retried = _requeue_due(root, now, release)
        for pending in sorted((root / "pending").glob("*.json"))[:max_messages]:
            claimed = root / "processing" / pending.name
            try:
                os.replace(pending, claimed)
            except FileNotFoundError:
                continue
            results.append(_finish(root, claimed, _process(root, claimed, fetcher=fetcher,
                                                           offer_transport=offer_transport), now=_utc_now(),
                                   release=release))
    return {"schema_version": RUN_SCHEMA, "status": "processed" if results or recovered else "idle",
            "processed_count": len(results), "results": results, "recovered": recovered,
            "automatic_requeue": retried, "provider_mutation_performed": False, "paid_execution_requested": False}


def _record_website_owner_decisions(root: Path, outcomes: Mapping[str, Mapping[str, Any]]) -> None:
    """Keep each owner-contract-blocked request's latest website exchange in its status (the lock is held).

    Written only when it changed, and only while the request is still blocked
    on the owner contract, so a request requeued or recompiled meanwhile keeps
    its own status.
    """

    from .website_owner_success_confirmation import STATUS_FIELD, _owner_contract_blocked

    for key, outcome in outcomes.items():
        path = root / "status" / f"{key}.json"
        status = _read_json(path)
        if (status is None or not _owner_contract_blocked(status) or not (root / "processed" / f"{key}.json").exists()
                or status.get(STATUS_FIELD) == dict(outcome)):
            continue
        _write_replace(path, {**status, STATUS_FIELD: dict(outcome)})


def _record_website_robot_access(root: Path, outcomes: Mapping[str, Mapping[str, Any]]) -> None:
    """Keep each completed half's latest robot-access exchange in its status (the lock is held).

    Written only when it changed, and only while the request is still completed and not pending, so a
    request requeued or recompiled meanwhile keeps its own status.
    """

    from .robot_access_furniture import STATUS_FIELD

    for key, outcome in outcomes.items():
        path = root / "status" / f"{key}.json"
        status = _read_json(path)
        if (status is None or status.get("status") != "completed" or (root / "pending" / f"{key}.json").exists()
                or not (root / "processed" / f"{key}.json").exists() or status.get(STATUS_FIELD) == dict(outcome)):
            continue
        _write_replace(path, {**status, STATUS_FIELD: dict(outcome)})


def requeue_transient_blocked(*, queue_root: str | Path,
                              website_transport: Callable[..., Any] | None = None,
                              robot_access_transport: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Recover an interrupted run and return each due transient block to pending, for a timer's pass.

    The worker runs only when something is pending, so a request its unit's
    timeout left in ``processing/``, a block waiting out its backoff, an
    owner-contract refusal the owner has since confirmed, or a deterministic
    refusal a newly deployed release has not yet tried, needs a pass from
    outside. This pass reads the release it runs from itself
    (``current_release_commit``). When a worker run holds the queue this does
    nothing: that run recovers and requeues the same way.

    First, before taking the queue lock, each request blocked only on an
    unconfirmed owner contract asks the WebApp for the owner's decision on its
    thresholds (``website_owner_success_confirmation``); a website confirmation
    is written under the intent, so the requeue below returns that request to
    pending in this same pass. Each request is asked again only when its kept
    check time comes (``website_owner_success_confirmation.BACKOFF``). Nothing in
    that exchange stops this pass: any error is recorded and the requeue below
    still runs. ``website_transport`` replaces the signed POST (tests).

    The same pass sends each completed half's proposal to move named furniture out
    of the Franka's way to the WebApp (``robot_access_furniture``): an approval is
    sealed under the intent, so the requeue below compiles that half again; a
    decline holds it by name in its status. ``robot_access_transport`` replaces
    that POST (tests).
    """

    from .robot_access_furniture import exchange_robot_access_decisions
    from .website_owner_success_confirmation import exchange_owner_decisions

    root = _queue(queue_root)
    release = current_release_commit()
    errors: dict[str, str] = {}
    exchanges: dict[str, dict[str, Any]] = {}
    for name, exchange, transport in (
            ("website_owner_decisions", exchange_owner_decisions, website_transport),
            ("website_robot_access", exchange_robot_access_decisions, robot_access_transport)):
        try:
            exchanges[name] = exchange(root, transport=transport)
        except Exception as exc:  # noqa: BLE001 - a pass never stops on a website exchange
            from .bounded_error import error_description

            exchanges[name], errors[name] = {}, error_description(exc)
    website, robot_access = exchanges["website_owner_decisions"], exchanges["website_robot_access"]
    with _locked(root, wait=False) as held:
        if not held:
            return {"schema_version": RUN_SCHEMA, "status": "busy", "requeued": [], "recovered": [],
                    "website_owner_decisions": website, "website_robot_access": robot_access,
                    "provider_mutation_performed": False, "paid_execution_requested": False}
        now = _utc_now()
        recovered = [_recover_interrupted(root, path, now=now, release=release)["request_key"]
                     for path in sorted((root / "processing").glob("*.json"))]
        for name, record, outcomes in (("website_owner_decisions", _record_website_owner_decisions, website),
                                       ("website_robot_access", _record_website_robot_access, robot_access)):
            try:
                record(root, outcomes)
            except Exception as exc:  # noqa: BLE001 - a status that cannot be written is retried next pass
                from .bounded_error import error_description

                errors.setdefault(name, error_description(exc))
        requeued = _requeue_due(root, now, release)
    return {"schema_version": RUN_SCHEMA, "status": "requeued" if requeued else "idle", "requeued": requeued,
            "recovered": recovered, "website_owner_decisions": website, "website_robot_access": robot_access,
            **{f"{name}_error": error for name, error in errors.items()},
            "provider_mutation_performed": False, "paid_execution_requested": False}


def requeue_unfinished(*, queue_root: str | Path) -> dict[str, Any]:
    """Return every processed request whose status is not completed to pending, for one more run."""

    root = _queue(queue_root)
    requeued: list[str] = []
    with _locked(root):
        for path in sorted((root / "processed").glob("*.json")):
            target = root / "pending" / path.name
            if (_read_json(root / "status" / path.name) or {}).get("status") == "completed" or target.exists():
                continue
            os.replace(path, target)
            requeued.append(path.stem)
    return {"schema_version": RUN_SCHEMA, "status": "requeued", "requeued": requeued,
            "provider_mutation_performed": False, "paid_execution_requested": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queue-root", default=os.environ.get(QUEUE_ROOT_ENV, ""))
    parser.add_argument("--max-messages", type=int, default=2)
    parser.add_argument("--requeue-unfinished", action="store_true",
                        help="return skipped and blocked requests to pending once their cause is fixed")
    args = parser.parse_args(argv)
    if not args.queue_root:
        print(json.dumps({"schema_version": RUN_SCHEMA, "status": "blocked",
                          "blockers": ["team_scene_packet_queue_root_not_configured"]}, sort_keys=True))
        return 2
    result = (requeue_unfinished(queue_root=args.queue_root) if args.requeue_unfinished
              else process_team_scene_queue(queue_root=args.queue_root, max_messages=args.max_messages))
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "CAPTURE_PARTITION_ENV",
    "HOLDS_DIRNAME",
    "PACKET_INPUT_FIELDS",
    "QUEUE_ROOT_ENV",
    "REQUEST_SCHEMA",
    "AUTOMATIC_RETRY_BACKOFF_SECONDS",
    "OWNER_CONFIRMATION_REQUEUE_FIELD",
    "OWNER_CONTRACT_BLOCKER_PREFIX",
    "RELEASE_COMMIT_FIELD",
    "RELEASE_REQUEUE_FIELD",
    "SCENARIO_ID",
    "STATUS_SCHEMA",
    "TRANSIENT_BLOCKERS",
    "TeamScenePacketRequestRefused",
    "capture_key",
    "configured_scene_bundle",
    "current_release_commit",
    "enqueue_team_scene_request",
    "final_block",
    "owner_success_confirmation_scope",
    "packet_inputs",
    "process_team_scene_queue",
    "registered_embodiment_models",
    "registered_embodiments",
    "request_key",
    "requeue_transient_blocked",
    "requeue_unfinished",
    "resolve_capture_root",
    "transient_cause",
    "validated_request",
]


if __name__ == "__main__":
    raise SystemExit(main())
