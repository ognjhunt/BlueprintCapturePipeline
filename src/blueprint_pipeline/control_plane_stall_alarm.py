"""Page when a live job stops moving and nothing else would say so.

On 2026-11-01 one website job sat for hours, several times, and nothing paged:
a launch lease stayed in ``task-evaluation-launches/processing/`` while the
launch reconciler timer had been inactive since 2026-09-30, a scene progression
row kept one blocker code for hours, and a deployed trigger was left inactive
with no owner hold. The capacity controller already pages a person through
``BLUEPRINT_OPERATOR_ALERT_WEBHOOK_URL``; this module adds three checks to its
tick so those stops page on the same route:

``scene_progression_stalled``
    A scene intent's ``status``, ``phase`` and ``blockers`` have not changed for
    longer than its threshold. A row that names blockers waits on the pipeline
    and pages after ``progression`` minutes (default 90). A row without blockers
    in ``awaiting_execution`` or ``running`` has been handed to a downstream
    worker that may legitimately hold it for a whole run (scene configuration
    launches carry a 7.5 h hard TTL), so it pages only after ``handoff`` minutes
    (default 8 h); a dead worker underneath is caught sooner by the next two
    checks. Completed rows, ``needs_input`` rows, expired or revoked execution
    windows and intended holds (pauses, privacy, rights and money approvals)
    never page.
``launch_lease_stuck``
    A request in ``processing/`` that no live dispatcher owns
    (``unowned_processing_names``) is older than the bound the launch reconciler
    applies to it, plus a grace for the reconciler to close it.
``trigger_inactive_unheld``
    A ``.timer`` or ``.path`` the deploy tool always arms is loaded but has been
    inactive longer than ``trigger`` minutes (default 60, above a deploy's
    trigger pause) with no operator-door hold record.
``launch_activation_stuck``
    A row in ``task-evaluation-launch-activations/processing/`` that no process
    holds a lock on, claimed longer ago than the activation role's TTL. The
    activation worker holds an exclusive ``flock`` on the row from its claim to
    its terminal move; the kernel lists every held lock in ``/proc/locks``, so
    an unlisted row has no live worker. 2026-10-31: activation ``6e9f0cac`` sat
    in ``processing/`` for two days, held 14 GB of staging as queue-referenced,
    and nothing paged.

Four more stores were found unwatched on 2026-11-10: a job stuck in any of them
was invisible. Each has its own check, with the same route and repeat:

``atlas_preview_stalled``
    A website Atlas preview job (``pipeline/atlas/preview_job.json``) has made
    no progress for ``atlas`` minutes (default 90) and is not in a scheduled
    wait whose next attempt is still in the future: a wait the worker scheduled
    (an outage's backoff, the refusal schedule, a killed step's retry, a
    ``needs_input`` recheck) pages only once its attempt is overdue by as long,
    and a step in flight that long pages. ``ready``, ``failed`` and
    ``withdrawn`` jobs, ``needs_input`` jobs that wait for a person, and any
    wait on privacy, rights, consent or money never page.
``team_scene_packet_stalled``
    A team scene packet request has waited in ``pending/`` or ``processing/``,
    or stayed ``blocked``, for ``team_scene`` minutes (default 8 h, past the
    worker's own retries of 5 min, 30 min and 2 h). A block whose next timed
    retry (``automatic_retry`` or ``refusal_retry``) is still in the future is
    quiet; one overdue by the threshold pages. A block that waits for the site
    owner's confirmation, or names a withdrawn capture, never pages.
``rigid_probe_stalled``
    A rigid robot-free probe entry is still in ``pending/`` ``rigid_probe``
    minutes (default 8 h: two funded attempts of at most 1 h each) after it was
    queued or its latest attempt last wrote a record, or a retired staging
    refusal's next timed retry is overdue by as long. An entry whose last
    attempt the WebApp refused the grant for (a money stop the owner lifts)
    never pages.
``screening_campaign_stalled`` and ``screening_publication_stalled``
    A screening campaign has rows still to run and its ledger entry has not
    changed for ``screening`` minutes (default 8 h, twice a run's longest hard
    TTL), or its latest revision has stayed unpublished for ``publication``
    minutes (default 90). Waitlisted rows (the site allowance or job ceiling)
    never page. A ``blocked`` row is not a row still to run, so one waiting for
    the coordinator's timed re-plan is quiet; an overdue re-plan is not seen
    (the ledger records no due time for it).

A store whose own trigger the operator door holds is paused on purpose and is
not read. A request, probe or campaign pages only while its capture still has a
scene intent with an open execution window (``open_window_captures``); after
that it cannot run again and is history.

Every check only reads: it lists the named directories with limits, reads
bounded JSON, takes the dispatcher lock exactly as the reconciler does to learn
ownership, reads ``/proc/locks`` (never locking an activation row itself), and
runs one ``systemctl show``. It never moves a queue entry,
writes a progression, calls a provider or starts or stops a unit. A check that
cannot run reports ``stall_check_incomplete`` (warn) and the tick goes on.
"""

from __future__ import annotations

import ast
import json
import os
import re
import stat
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .control_plane_disk_budget import ROLE_TTL_SECONDS
from .job_retry_policy import MONEY, PRIVACY_RIGHTS, hard_stop_kind, refusal_retry_delay
from .task_evaluation_launch_dispatcher_ownership import DISPATCHER_LOCK_NAME, unowned_processing_names

SCHEMA_VERSION = "control_plane_stall_check.v1"
PAGE_CODES = frozenset({"scene_progression_stalled", "launch_lease_stuck", "trigger_inactive_unheld",
                        "launch_activation_stuck", "atlas_preview_stalled", "team_scene_packet_stalled",
                        "rigid_probe_stalled", "screening_campaign_stalled", "screening_publication_stalled",
                        # Not a stall: a scene's ceiling passed a whole multiple of its grant (gates nothing).
                        "scene_budget_doubled",
                        # Not a stall: a preparation stopped on Blueprint's own records, paged once
                        # (preparation_fault_alert).
                        "website_preparation_blueprint_fault"})
INCOMPLETE_CODE = "stall_check_incomplete"

DISABLE_ENV = "BLUEPRINT_STALL_ALARM"
PROGRESSION_MINUTES_ENV = "BLUEPRINT_STALL_PROGRESSION_MINUTES"
HANDOFF_MINUTES_ENV = "BLUEPRINT_STALL_HANDOFF_MINUTES"
LAUNCH_GRACE_MINUTES_ENV = "BLUEPRINT_STALL_LAUNCH_GRACE_MINUTES"
TRIGGER_MINUTES_ENV = "BLUEPRINT_STALL_TRIGGER_MINUTES"
# The same names the launch reconciler unit uses for the same roots.
SCENE_INTENT_ROOT_ENV = "BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT"
LAUNCH_QUEUE_ROOT_ENV = "BLUEPRINT_TASK_EVALUATION_LAUNCH_QUEUE_ROOT"
LAUNCH_STATE_ROOT_ENV = "BLUEPRINT_TASK_EVALUATION_LAUNCH_STATE_ROOT"
# The activation worker unit's own name for its queue.
ACTIVATION_QUEUE_ROOT_ENV = "BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_QUEUE_ROOT"
# Each job store by the name its own worker unit gives it.
ATLAS_STORAGE_ROOT_ENV = "BLUEPRINT_WEBSITE_ATLAS_PREVIEW_STORAGE_ROOT"
TEAM_SCENE_QUEUE_ROOT_ENV = "BLUEPRINT_TEAM_SCENE_PACKET_QUEUE_ROOT"
RIGID_PROBE_QUEUE_ROOT_ENV = "BLUEPRINT_RIGID_ROBOT_FREE_PROBE_QUEUE"
SCREENING_QUEUE_ROOT_ENV = "BLUEPRINT_BETA_POLICY_SCREENING_QUEUE_ROOT"
ATLAS_MINUTES_ENV = "BLUEPRINT_STALL_ATLAS_MINUTES"
TEAM_SCENE_MINUTES_ENV = "BLUEPRINT_STALL_TEAM_SCENE_MINUTES"
RIGID_PROBE_MINUTES_ENV = "BLUEPRINT_STALL_RIGID_PROBE_MINUTES"
SCREENING_MINUTES_ENV = "BLUEPRINT_STALL_SCREENING_MINUTES"
PUBLICATION_MINUTES_ENV = "BLUEPRINT_STALL_PUBLICATION_MINUTES"

DEFAULT_PROGRESSION_MINUTES = 90
DEFAULT_HANDOFF_MINUTES = 8 * 60
DEFAULT_LAUNCH_GRACE_MINUTES = 30
DEFAULT_TRIGGER_MINUTES = 60
# The job stores take one of the two existing waits. A job that waits on the pipeline itself
# gets the progression wait (90 min): an Atlas job is stepped every minute and a deploy pauses
# its timer for 30-40 minutes, and a publication is offered again on every screening pass. A
# request handed to a worker that may hold it for whole runs gets the handoff wait (8 h): a
# team scene request waits out its worker's own retries (5 min, 30 min, 2 h) behind compiles of
# up to an hour, a rigid probe runs two funded attempts of at most an hour each, and a
# screening run holds its row for up to its hard TTL of at most four hours.
DEFAULT_ATLAS_MINUTES = DEFAULT_PROGRESSION_MINUTES
DEFAULT_TEAM_SCENE_MINUTES = DEFAULT_HANDOFF_MINUTES
DEFAULT_RIGID_PROBE_MINUTES = DEFAULT_HANDOFF_MINUTES
DEFAULT_SCREENING_MINUTES = DEFAULT_HANDOFF_MINUTES
DEFAULT_PUBLICATION_MINUTES = DEFAULT_PROGRESSION_MINUTES

# The launch reconciler's own bounds (``task_evaluation_launch_reconciler``):
# a lease whose allocator never started is stale after PRE_RUN_STALE_SECONDS,
# a started one after its ``hard_ttl_seconds`` or the CLI fallback. Mirrored,
# not imported, so the capacity tick does not load the dispatcher stack; a
# test keeps them equal.
LAUNCH_PRE_RUN_STALE_SECONDS = 900
LAUNCH_FALLBACK_STALE_SECONDS = 14_400

# Statuses ``task_evaluation_scene_progression_state.advance`` can write.
TERMINAL_STATUSES = frozenset({"completed"})
WAITING_ON_PERSON_STATUSES = frozenset({"needs_input"})
HANDOFF_STATUSES = frozenset({"awaiting_execution", "running"})
# ``paused`` comes from the progression config's paused_intent_ids;
# ``construction_prepared`` only when the config sets activation_enabled false.
OPERATOR_HOLD_PHASES = frozenset({"paused", "construction_prepared"})
# Intended holds, matched on whole ``_``-separated words of a blocker code.
_HOLD_WORDS = {
    "paused": "operator_pause",
    "revoked": "rights",
    "rights": "rights",
    "license": "rights",
    "licence": "rights",
    "privacy": "privacy",
    "consent": "privacy",
    "approval": "money_approval",
}
_HOLD_CODES = {
    "scene_intake_authority_expired": "execution_window",
    # The intent's sealed execution budget is used up; only an owner budget
    # extension continues the job (the WebApp's "Owner approval needed").
    "scene_intake_spend_cap_exhausted": "money_approval",
    "scene_intake_attempt_cap_exhausted": "money_approval",
}

# The four job stores below name a withdrawn capture in their own codes
# (``team_scene_capture_withdrawn``); only their checks read these. ``_HOLD_WORDS`` is unchanged.
_STORE_HOLD_WORDS = {"withdrawn": "privacy", "withdrawal": "privacy"}

# What each store's owner writes, mirrored (not imported) so the capacity tick loads none of
# the workers, their provider adapters or the paid launch path; a test keeps each equal.
ATLAS_JOB_GLOB = "*/scenes/site-*/captures/*/pipeline/atlas/preview_job.json"
ATLAS_JOB_SCHEMA = "website_atlas_preview_job.v1"
ATLAS_TRIGGER = "blueprint-website-atlas-preview.timer"
# ``processing`` and ``needs_input`` jobs are stepped again; ready, failed and withdrawn are final.
ATLAS_WAITING_STATES = frozenset({"processing", "needs_input"})
ATLAS_STEP_IN_FLIGHT = "step_in_flight"
TEAM_SCENE_WAITING_STATES = ("pending", "processing")
TEAM_SCENE_OWNER_CONTRACT_PREFIX = "team_scene_owner_contract_invalid:"
TEAM_SCENE_TRIGGER = "blueprint-team-scene-packet.path"
# A blocked status's timed retries: the transient backoff and the shared refusal schedule. Each names
# ``not_before`` until the worker returns the request to pending, or ``exhausted`` once spent.
TEAM_SCENE_RETRY_FIELDS = ("automatic_retry", "refusal_retry")
RIGID_PROBE_INGEST_FILENAME = "rigid_robot_free_probe_ingest.json"
RIGID_PROBE_GRANT_REFUSED_STATE = "website_grant_refused"
RIGID_PROBE_TRIGGER = "blueprint-rigid-robot-free-probe-runner.timer"
# A retired entry whose staging refusal the runner sets aside again on the shared refusal schedule.
RIGID_PROBE_TERMINAL_FILENAME = "rigid_robot_free_probe_terminal.json"
RIGID_PROBE_TIMED_STATES = frozenset({"staging_refused"})
RIGID_PROBE_SUPERSEDED_DIRNAME = "superseded_terminals"
RIGID_PROBE_REFUSAL_RETRY_TRIGGER = "refusal_retry"
MAX_SUPERSESSIONS = 64
SCREENING_LEDGER_FILENAME = "screening_ledger.v1.json"
SCREENING_LEDGER_SCHEMA = "blueprint.beta_policy_screening_ledger.v1"
# Rows the screening queue still has to run or settle; ``waitlisted`` waits for money instead.
SCREENING_LIVE_STATUSES = frozenset({"pending", "preparing", "launching", "running", "reconciliation_required"})
_SCREENING_STATUS_ORDER = ("reconciliation_required", "running", "launching", "preparing", "pending")
SCREENING_WAITLISTED_STATUS = "waitlisted"
SCREENING_TRIGGER = "blueprint-beta-policy-screening.timer"
MAX_ATLAS_JOBS = 512
MAX_QUEUE_ENTRIES = 512
# A status is kept for every request ever finished; the newest are read.
MAX_STATUS_ENTRIES = 2048
MAX_CAMPAIGNS = 512
MAX_LEDGER_BYTES = 4 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{16}\Z")

MAX_INTENT_DIRECTORY_ENTRIES = 10_000
MAX_INTENTS = 256
MAX_EVENTS_PER_INTENT = 64
MAX_EVENT_READS = 512
MAX_LEASES = 256
MAX_ACTIVATIONS = 256
PROC_LOCKS = Path("/proc/locks")
MAX_PROC_LOCKS_BYTES = 4 * 1024 * 1024
# A claimed activation row older than this has outlived any worker: the unit's
# start timeout is shorter, and the disk ledger uses the same bound.
ACTIVATION_STALE_SECONDS = ROLE_TTL_SECONDS["launch_activation"]
_LOCK_KEY = re.compile(r"([0-9a-f]+):([0-9a-f]+):([0-9]+)\Z")
MAX_TARGETS = 10
MAX_JSON_BYTES = 1024 * 1024
MAX_DEPLOY_TOOL_BYTES = 8 * 1024 * 1024
DEADLINE_SECONDS = 90.0
SYSTEMCTL_TIMEOUT_SECONDS = 15.0
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@-]{0,255}\Z")
_UNIT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@:-]{0,250}\.(?:timer|path)\Z")

# Names the deploy tool declares; read as data from its source, never imported.
_DEPLOY_TRIGGER_LISTS = (
    "DEFAULT_ALWAYS_ARM_PATH_UNITS",
    "DEFAULT_ALWAYS_ARM_AUTHORITY_GATED_PATH_UNITS",
    "DEFAULT_ALWAYS_ARM_TIMER_UNITS",
)
_DEPLOY_OPTIONAL_LISTS = ("DEPLOY_EXEMPT_TRIGGER_UNITS",)
# Re-armed only when a deploy is not told to preserve their state, so their
# being off can be the owner's choice; left out of the trigger check.
_DEPLOY_PRESERVABLE_LIST = "CONFIGURED_CONTROLS_AUTOMATION_UNITS"
_DEPLOY_HOLDS_DIR = "DEFAULT_DOOR_HOLDS_DIR"


class StallCheckError(RuntimeError):
    """A check could not observe what it needs; the reason is a fixed code."""


@dataclass(frozen=True)
class StallConfig:
    scene_intent_root: Path | None
    launch_queue_root: Path | None
    launch_state_root: Path | None
    deploy_tool: Path | None
    progression_seconds: float = DEFAULT_PROGRESSION_MINUTES * 60.0
    handoff_seconds: float = DEFAULT_HANDOFF_MINUTES * 60.0
    launch_grace_seconds: float = DEFAULT_LAUNCH_GRACE_MINUTES * 60.0
    trigger_seconds: float = DEFAULT_TRIGGER_MINUTES * 60.0
    config_errors: tuple[str, ...] = ()
    activation_queue_root: Path | None = None
    atlas_storage_root: Path | None = None
    team_scene_queue_root: Path | None = None
    rigid_probe_queue_root: Path | None = None
    screening_queue_root: Path | None = None
    atlas_seconds: float = DEFAULT_ATLAS_MINUTES * 60.0
    team_scene_seconds: float = DEFAULT_TEAM_SCENE_MINUTES * 60.0
    rigid_probe_seconds: float = DEFAULT_RIGID_PROBE_MINUTES * 60.0
    screening_seconds: float = DEFAULT_SCREENING_MINUTES * 60.0
    publication_seconds: float = DEFAULT_PUBLICATION_MINUTES * 60.0


def default_deploy_tool() -> Path:
    """The deploy tool of the release this module runs from."""
    return Path(__file__).resolve().parents[2] / "scripts" / "deploy_control_plane_commit.py"


def config_from_environment(environ: Mapping[str, str] | None = None) -> StallConfig | None:
    """``None`` when the owner turned the alarm off with ``BLUEPRINT_STALL_ALARM=0``."""

    env = os.environ if environ is None else environ
    if str(env.get(DISABLE_ENV) or "").strip().lower() in {"0", "false", "off", "no"}:
        return None
    errors: list[str] = []

    def minutes(name: str, default: int) -> float:
        raw = str(env.get(name) or "").strip()
        if not raw:
            return default * 60.0
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value <= 0:
            errors.append(f"threshold_invalid:{name}")
            return default * 60.0
        return value * 60.0

    def root(name: str) -> Path | None:
        raw = str(env.get(name) or "").strip()
        return Path(raw) if raw.startswith("/") else None

    return StallConfig(
        scene_intent_root=root(SCENE_INTENT_ROOT_ENV),
        launch_queue_root=root(LAUNCH_QUEUE_ROOT_ENV),
        launch_state_root=root(LAUNCH_STATE_ROOT_ENV),
        deploy_tool=default_deploy_tool(),
        progression_seconds=minutes(PROGRESSION_MINUTES_ENV, DEFAULT_PROGRESSION_MINUTES),
        handoff_seconds=minutes(HANDOFF_MINUTES_ENV, DEFAULT_HANDOFF_MINUTES),
        launch_grace_seconds=minutes(LAUNCH_GRACE_MINUTES_ENV, DEFAULT_LAUNCH_GRACE_MINUTES),
        trigger_seconds=minutes(TRIGGER_MINUTES_ENV, DEFAULT_TRIGGER_MINUTES),
        activation_queue_root=root(ACTIVATION_QUEUE_ROOT_ENV),
        atlas_storage_root=root(ATLAS_STORAGE_ROOT_ENV),
        team_scene_queue_root=root(TEAM_SCENE_QUEUE_ROOT_ENV),
        rigid_probe_queue_root=root(RIGID_PROBE_QUEUE_ROOT_ENV),
        screening_queue_root=root(SCREENING_QUEUE_ROOT_ENV),
        atlas_seconds=minutes(ATLAS_MINUTES_ENV, DEFAULT_ATLAS_MINUTES),
        team_scene_seconds=minutes(TEAM_SCENE_MINUTES_ENV, DEFAULT_TEAM_SCENE_MINUTES),
        rigid_probe_seconds=minutes(RIGID_PROBE_MINUTES_ENV, DEFAULT_RIGID_PROBE_MINUTES),
        screening_seconds=minutes(SCREENING_MINUTES_ENV, DEFAULT_SCREENING_MINUTES),
        publication_seconds=minutes(PUBLICATION_MINUTES_ENV, DEFAULT_PUBLICATION_MINUTES),
        # Last: every threshold above may add to it.
        config_errors=tuple(errors),
    )


# --------------------------------------------------------------------------- #
# Bounded reads
# --------------------------------------------------------------------------- #


def _read_json(path: Path, limit: int = MAX_JSON_BYTES) -> dict[str, Any]:
    """One regular file, at most ``limit`` bytes, never through a symlink."""

    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("stall_input_unsafe")
        raw = os.read(descriptor, limit + 1)
    finally:
        os.close(descriptor)
    if len(raw) > limit:
        raise ValueError("stall_input_too_large")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("stall_input_not_object")
    return value


def _number(value: Any) -> float | None:
    if type(value) in (int, float) and value == value and value not in (float("inf"), float("-inf")):
        return float(value)
    return None


def _epoch(value: Any) -> float | None:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc).timestamp() if parsed.tzinfo is not None else None


class _Budget:
    def __init__(self, *, deadline: float, events: int, clock: Callable[[], float]) -> None:
        self.deadline = deadline
        self.events = events
        self.clock = clock

    def expired(self) -> bool:
        return self.clock() >= self.deadline


def _incomplete(reason: str, **detail: Any) -> dict[str, Any]:
    return {"code": INCOMPLETE_CODE, "reason": reason, **detail}


def _page(code: str, offenders: Sequence[Mapping[str, Any]], **extra: Any) -> dict[str, Any]:
    """One alert per code: the oldest offender, the count and the first targets."""

    ordered = sorted(offenders, key=lambda row: (-float(row["age_seconds"]), str(row["target"])))
    oldest = ordered[0]
    return {
        "code": code,
        "count": len(ordered),
        "targets": [str(row["target"]) for row in ordered[:MAX_TARGETS]],
        **{key: value for key, value in oldest.items() if value is not None},
        **extra,
    }


# --------------------------------------------------------------------------- #
# 1. Scene progression rows that stopped changing
# --------------------------------------------------------------------------- #


def _signature(row: Mapping[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    blockers = row.get("blockers") if isinstance(row.get("blockers"), list) else []
    return (str(row.get("status") or ""), str(row.get("phase") or ""),
            tuple(sorted(str(code) for code in blockers)))


def hold_reason(status: str, phase: str, blockers: Iterable[str]) -> str | None:
    """Why a row is an intended stop that must not page, or ``None``."""

    if status in TERMINAL_STATUSES:
        return "terminal"
    if status in WAITING_ON_PERSON_STATUSES:
        return "needs_input"
    if phase in OPERATOR_HOLD_PHASES:
        return "operator_pause"
    for code in blockers:
        text = str(code)
        if text in _HOLD_CODES:
            return _HOLD_CODES[text]
        for word in re.split(r"[_:]", text.lower()):
            if word in _HOLD_WORDS:
                return _HOLD_WORDS[word]
    return None


def _execution_expired(directory: Path, intent: Mapping[str, Any], *, now: float) -> bool | None:
    """Whether the effective execution window (with extensions) has passed; ``None`` if unknown."""

    from .task_evaluation_scene_execution_window_evidence import effective_execution_expiry

    try:
        return now >= effective_execution_expiry(directory, intent)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _unchanged_since(directory: Path, progression: Mapping[str, Any], *, budget: _Budget) -> tuple[float, bool]:
    """When the current status, phase and blockers began; ``exact`` is False for a lower bound."""

    signature = _signature(progression)
    since = float(progression["updated_at_epoch"])
    sequence = int(progression["event_sequence"])
    events = directory / "progression-events"
    reads = 0
    while True:
        sequence -= 1
        if sequence < 1:
            return since, True
        if reads >= MAX_EVENTS_PER_INTENT or budget.events <= 0 or budget.expired():
            return since, False
        reads += 1
        budget.events -= 1
        try:
            event = _read_json(events / f"{sequence:06d}.json")
        except (OSError, ValueError):
            return since, False
        observed = _number(event.get("observed_at_epoch"))
        if event.get("sequence") != sequence or observed is None:
            return since, False
        if _signature(event) != signature:
            return since, True
        since = observed


def _remembered_since(prior: Any, signature: list[Any], since: float, exact: bool) -> tuple[float, bool]:
    """Carry an earlier start of this same signature across ticks.

    The backwards event walk is bounded, so a row whose state churns every
    minute (a capacity wait rewrites ``next_check_epoch`` each tick) shows only
    about 64 minutes of history. A walk that found where the signature began is
    exact and wins; a lower bound is replaced by an earlier remembered start of
    the same signature, so a long stall still reaches its threshold.
    """

    if exact or not isinstance(prior, Mapping) or prior.get("signature") != signature:
        return since, exact
    remembered = _number(prior.get("since"))
    if remembered is None or remembered >= since:
        return since, exact
    return remembered, prior.get("exact") is True


def _scene_intent_entries(root: Path) -> tuple[list[tuple[float, str]], bool]:
    """Every ``scene-*`` intent directory as (mtime, name), newest first, and whether the listing was cut.

    Newest first: a row that is still changing or newly stalled is never the
    one a truncated walk drops; months-old revoked rows are. Raises ``OSError``.
    """

    entries: list[tuple[float, str]] = []
    truncated = False
    with os.scandir(root) as iterator:
        for index, entry in enumerate(iterator):
            if index >= MAX_INTENT_DIRECTORY_ENTRIES:
                truncated = True
                break
            if entry.name.startswith("scene-") and entry.is_dir(follow_symlinks=False):
                try:
                    entries.append((entry.stat(follow_symlinks=False).st_mtime, entry.name))
                except OSError:
                    continue
    entries.sort(key=lambda item: (-item[0], item[1]))
    return entries, truncated


def check_scene_progressions(
    config: StallConfig, *, now: float, budget: _Budget,
    remembered: Mapping[str, Any] | None = None, memory_out: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``remembered`` is the previous tick's ``memory_out``; watched rows are written to ``memory_out``."""

    remembered = remembered if isinstance(remembered, Mapping) else {}
    root = config.scene_intent_root
    section: dict[str, Any] = {"status": "not_configured"}
    if root is None:
        return section, [_incomplete("scene_intent_root_unconfigured")]
    alerts: list[dict[str, Any]] = []
    try:
        entries, truncated = _scene_intent_entries(root)
    except FileNotFoundError:
        return {"status": "absent"}, []
    except OSError:
        return {"status": "unreadable"}, [_incomplete("scene_intent_root_unreadable")]
    if len(entries) > MAX_INTENTS:
        truncated = True
    counts: dict[str, int] = {}
    stalled: list[dict[str, Any]] = []
    unreadable = 0
    examined = 0
    for _mtime, name in entries[:MAX_INTENTS]:
        if budget.expired():
            truncated = True
            break
        examined += 1
        directory = root / name
        try:
            progression = _read_json(directory / "progression.json")
        except FileNotFoundError:
            progression = None
        except (OSError, ValueError):
            unreadable += 1
            continue
        if progression is not None and (
            progression.get("schema_version") != "task_evaluation_scene_progression.v1"
            or progression.get("intent_id") != name
            or type(progression.get("event_sequence")) is not int
            or _number(progression.get("updated_at_epoch")) is None
        ):
            unreadable += 1
            continue
        status, phase, blockers = _signature(progression or {"status": "not_started", "phase": "intake"})
        reason = hold_reason(status, phase, blockers)
        if reason is None and (directory / "revoked.json").exists():
            reason = "rights"
        if reason is not None:
            counts[reason] = counts.get(reason, 0) + 1
            continue
        try:
            intent = _read_json(directory / "intent.json")
        except (OSError, ValueError):
            intent = None
        if intent is not None and _execution_expired(directory, intent, now=now):
            counts["execution_window"] = counts.get("execution_window", 0) + 1
            continue
        if progression is None:
            accepted = _number((intent or {}).get("accepted_at_epoch"))
            if accepted is None:
                unreadable += 1
                continue
            since, exact = accepted, True
        else:
            since, exact = _unchanged_since(directory, progression, budget=budget)
            signature = [status, phase, list(blockers)]
            since, exact = _remembered_since(remembered.get(name), signature, since, exact)
            if memory_out is not None:
                memory_out[name] = {"signature": signature, "since": since, "exact": exact}
        threshold = (config.handoff_seconds if not blockers and status in HANDOFF_STATUSES
                     else config.progression_seconds)
        age = now - since
        counts["watched"] = counts.get("watched", 0) + 1
        if age < threshold:
            continue
        request = (intent or {}).get("request")
        label = request.get("submission_id") if isinstance(request, Mapping) else None
        stalled.append({
            "target": name,
            "label": label[:120] if isinstance(label, str) and _IDENTIFIER.match(label) else None,
            "status": status,
            "phase": phase,
            "blockers": list(blockers[:4]),
            "age_seconds": round(age),
            "age_is_lower_bound": not exact,
            "threshold_seconds": round(threshold),
        })
    section = {"status": "observed", "examined": examined, "inventory": len(entries),
               "truncated": truncated, "unreadable": unreadable, "stalled": len(stalled),
               "dispositions": dict(sorted(counts.items()))}
    if stalled:
        alerts.append(_page("scene_progression_stalled", stalled))
    if truncated:
        alerts.append(_incomplete("scene_intent_inventory_truncated", count=len(entries)))
    if unreadable:
        alerts.append(_incomplete("scene_progression_unreadable", count=unreadable))
    return section, alerts


# --------------------------------------------------------------------------- #
# 2. Launch leases no live dispatcher owns, past the reconciler's bound
# --------------------------------------------------------------------------- #


def check_launch_leases(config: StallConfig, *, now: float, budget: _Budget) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    queue, state = config.launch_queue_root, config.launch_state_root
    if queue is None or state is None:
        return {"status": "not_configured"}, [_incomplete("launch_roots_unconfigured")]
    processing = queue / "processing"
    try:
        with os.scandir(processing) as iterator:
            in_flight = sum(1 for entry in iterator if entry.name.endswith(".json"))
    except FileNotFoundError:
        return {"status": "absent"}, []
    except OSError:
        return {"status": "unreadable"}, [_incomplete("launch_queue_unreadable")]
    if not in_flight:
        return {"status": "observed", "processing": 0, "unowned": 0, "stuck": 0}, []
    if not (queue / DISPATCHER_LOCK_NAME).exists():
        # Until a dispatcher that takes the lock has run, the reconciler treats
        # every lease as owned; so does this check.
        return {"status": "ownership_unknown", "processing": in_flight}, []
    unowned = sorted(unowned_processing_names(queue))
    stuck: list[dict[str, Any]] = []
    for name in unowned[:MAX_LEASES]:
        if budget.expired():
            break
        request_path = processing / name
        try:
            modified = request_path.lstat().st_mtime
        except FileNotFoundError:
            continue  # closed since the listing
        except OSError:
            modified = None
        try:
            request = _read_json(request_path)
        except (OSError, ValueError):
            request = {}
        launch_id = request.get("launch_id")
        if not isinstance(launch_id, str) or not _IDENTIFIER.match(launch_id) or ".." in launch_id:
            launch_id = Path(name).stem
        try:
            started = _read_json(state / launch_id / "launch_started.json")
        except (OSError, ValueError):
            started = {}
        started_at = _epoch(started.get("started_at"))
        ttl = started.get("hard_ttl_seconds")
        bound = float(ttl) if type(ttl) is int and ttl > 0 else float(LAUNCH_FALLBACK_STALE_SECONDS)
        if not started:
            # The allocator was never invoked: the reconciler's pre-run bound.
            bound = min(bound, float(LAUNCH_PRE_RUN_STALE_SECONDS))
        since = started_at if started_at is not None else modified
        if since is None:
            continue
        age = now - since
        if age > bound + config.launch_grace_seconds:
            stuck.append({"target": launch_id, "phase": "run" if started else "pre_run",
                          "age_seconds": round(age), "bound_seconds": round(bound)})
    section = {"status": "observed", "processing": in_flight, "unowned": len(unowned), "stuck": len(stuck),
               "truncated": len(unowned) > MAX_LEASES}
    return section, ([_page("launch_lease_stuck", stuck)] if stuck else [])


# --------------------------------------------------------------------------- #
# 3. Activation queue rows no live worker holds, past the activation TTL
# --------------------------------------------------------------------------- #


def read_lock_holders(path: Path = PROC_LOCKS) -> dict[tuple[int, int, int], list[int]]:
    """Every file lock the kernel lists, as ``(major, minor, inode) -> pids``.

    Reading ``/proc/locks`` takes no lock and changes nothing. A waiter line
    (``->``) counts too: someone still wants the row.
    """

    with path.open("rb") as stream:
        raw = stream.read(MAX_PROC_LOCKS_BYTES + 1)
    if len(raw) > MAX_PROC_LOCKS_BYTES:
        raise StallCheckError("proc_locks_oversized")
    holders: dict[tuple[int, int, int], list[int]] = {}
    for line in raw.decode("ascii", "replace").splitlines():
        fields = [field for field in line.split() if field != "->"]
        for index, field in enumerate(fields[1:], start=1):
            match = _LOCK_KEY.match(field)
            if match is None:
                continue
            try:
                pid = int(fields[index - 1])
            except ValueError:
                pid = -1
            key = (int(match.group(1), 16), int(match.group(2), 16), int(match.group(3)))
            holders.setdefault(key, []).append(pid)
            break
    return holders


def _activation_target(path: Path) -> str:
    try:
        request = _read_json(path).get("request")
    except (OSError, ValueError):
        request = None
    activation_id = request.get("activation_id") if isinstance(request, Mapping) else None
    if isinstance(activation_id, str) and _IDENTIFIER.match(activation_id) and ".." not in activation_id:
        return activation_id
    return path.stem[:200]


def check_launch_activations(
    config: StallConfig,
    *,
    now: float,
    budget: _Budget,
    lock_reader: Callable[[], Mapping[tuple[int, int, int], Sequence[int]]] = read_lock_holders,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    queue = config.activation_queue_root
    if queue is None:
        return {"status": "not_configured"}, []
    processing = queue / "processing"
    try:
        with os.scandir(processing) as iterator:
            names = sorted(entry.name for entry in iterator if entry.name.endswith(".json"))
    except FileNotFoundError:
        return {"status": "absent"}, []
    except OSError:
        return {"status": "unreadable"}, [_incomplete("activation_queue_unreadable")]
    if not names:
        return {"status": "observed", "processing": 0, "unowned": 0, "stuck": 0}, []
    try:
        holders = lock_reader()
    except (OSError, ValueError, StallCheckError):
        return {"status": "ownership_unknown", "processing": len(names)}, [_incomplete("lock_table_unreadable")]
    bound = float(ACTIVATION_STALE_SECONDS)
    unowned = 0
    stuck: list[dict[str, Any]] = []
    for name in names[:MAX_ACTIVATIONS]:
        if budget.expired():
            break
        path = processing / name
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue  # reached a terminal state since the listing
        except OSError:
            continue
        if not stat.S_ISREG(info.st_mode):
            continue
        if (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino) in holders:
            continue
        unowned += 1
        # The claim renames the row into processing/, which sets its ctime;
        # its mtime is when the envelope was written, possibly much earlier.
        age = now - info.st_ctime
        if age > bound:
            stuck.append({"target": _activation_target(path), "age_seconds": round(age),
                          "bound_seconds": round(bound)})
    section = {"status": "observed", "processing": len(names), "unowned": unowned, "stuck": len(stuck),
               "truncated": len(names) > MAX_ACTIVATIONS}
    return section, ([_page("launch_activation_stuck", stuck)] if stuck else [])


# --------------------------------------------------------------------------- #
# 4. Always-armed triggers left inactive without an owner hold
# --------------------------------------------------------------------------- #


def read_deploy_trigger_contract(path: Path) -> tuple[tuple[str, ...], Path]:
    """The deploy tool's always-armed ``.timer``/``.path`` units and its door holds dir."""

    try:
        if path.stat().st_size > MAX_DEPLOY_TOOL_BYTES:
            raise StallCheckError("deploy_tool_too_large")
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        raise StallCheckError("deploy_tool_unreadable") from exc
    wanted = {*_DEPLOY_TRIGGER_LISTS, *_DEPLOY_OPTIONAL_LISTS, _DEPLOY_PRESERVABLE_LIST, _DEPLOY_HOLDS_DIR}
    values: dict[str, Any] = {}
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        for target in targets:
            if isinstance(target, ast.Name) and target.id in wanted and getattr(node, "value", None) is not None:
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except ValueError as exc:
                    raise StallCheckError(f"deploy_tool_list_not_literal:{target.id}") from exc

    def units(name: str, *, required: bool) -> set[str]:
        value = values.get(name)
        if value is None and not required:
            return set()
        if not isinstance(value, tuple) or not all(isinstance(item, str) for item in value):
            raise StallCheckError(f"deploy_tool_list_missing:{name}")
        return set(value)

    armed: set[str] = set()
    for name in _DEPLOY_TRIGGER_LISTS:
        armed |= units(name, required=True)
    for name in _DEPLOY_OPTIONAL_LISTS:
        armed |= units(name, required=False)
    armed -= units(_DEPLOY_PRESERVABLE_LIST, required=False)
    holds = values.get(_DEPLOY_HOLDS_DIR)
    if not isinstance(holds, str) or not holds.startswith("/"):
        raise StallCheckError(f"deploy_tool_list_missing:{_DEPLOY_HOLDS_DIR}")
    return tuple(sorted(unit for unit in armed if _UNIT.match(unit))), Path(holds)


def read_unit_states(units: Sequence[str], *, runner: Callable[..., Any] = subprocess.run) -> dict[str, dict[str, str]]:
    """One bounded ``systemctl show``; only the properties this check needs."""

    if not units:
        return {}
    completed = runner(
        ["systemctl", "show", "--no-pager",
         "-p", "Id,LoadState,ActiveState,SubState,InactiveEnterTimestampMonotonic", "--", *units],
        capture_output=True, text=True, check=False, timeout=SYSTEMCTL_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise StallCheckError("systemctl_show_failed")
    states: dict[str, dict[str, str]] = {}
    current: dict[str, str] = {}
    for line in [*str(completed.stdout or "").splitlines(), ""]:
        if not line.strip():
            if current.get("Id"):
                states[current["Id"]] = current
            current = {}
            continue
        key, _, value = line.partition("=")
        current[key] = value
    return states


def trigger_held(holds: Path | None, unit: str) -> bool:
    """Whether the operator door holds ``unit``; a hold that cannot be ruled out is not paged as a stall."""

    if holds is None:
        return False
    try:
        os.lstat(holds / f"{unit}.json")
    except FileNotFoundError:
        return False
    except OSError:
        pass
    return True


def check_triggers(
    config: StallConfig,
    *,
    unit_reader: Callable[[Sequence[str]], Mapping[str, Mapping[str, str]]] = read_unit_states,
    monotonic: Callable[[], float] = time.monotonic,
    contract: tuple[tuple[str, ...], Path] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``contract`` is ``read_deploy_trigger_contract`` already read this tick; read here when absent."""

    if config.deploy_tool is None:
        return {"status": "not_configured"}, [_incomplete("deploy_tool_unconfigured")]
    try:
        units, holds = contract if contract is not None else read_deploy_trigger_contract(config.deploy_tool)
        states = unit_reader(units)
    except (StallCheckError, OSError, subprocess.SubprocessError) as exc:
        reason = str(exc) if isinstance(exc, StallCheckError) else "systemctl_show_failed"
        return {"status": "unavailable", "reason": reason}, [_incomplete(reason)]
    held: list[str] = []
    inactive: list[dict[str, Any]] = []
    loaded = 0
    for unit in units:
        row = states.get(unit) or {}
        if row.get("LoadState") != "loaded":
            continue  # not deployed on this host
        loaded += 1
        if row.get("ActiveState") in {"active", "activating", "reloading"}:
            continue
        if trigger_held(holds, unit):
            held.append(unit)
            continue
        try:
            entered = int(row.get("InactiveEnterTimestampMonotonic") or 0) / 1_000_000
        except ValueError:
            entered = 0.0
        age = monotonic() - entered  # 0: inactive since boot
        if age >= config.trigger_seconds:
            inactive.append({"target": unit, "status": row.get("ActiveState") or "unknown",
                             "age_seconds": round(age)})
    section = {"status": "observed", "watched": len(units), "loaded": loaded,
               "held": sorted(held), "inactive_unheld": len(inactive)}
    return section, ([_page("trigger_inactive_unheld", inactive)] if inactive else [])


# --------------------------------------------------------------------------- #
# 5-8. Job stores nothing else watches: Atlas previews, team scene packets,
#      rigid robot-free probes, screening campaigns and their publications
# --------------------------------------------------------------------------- #


def store_hold(codes: Iterable[Any]) -> str | None:
    """Why a store row naming ``codes`` is an intended stop: ``hold_reason``'s words, or a withdrawn capture."""

    texts = [str(code) for code in codes]
    reason = hold_reason("", "", texts)
    if reason is not None:
        return reason
    for text in texts:
        for word in re.split(r"[_:]", text.lower()):
            if word in _STORE_HOLD_WORDS:
                return _STORE_HOLD_WORDS[word]
    return None


def _store(root: Path | None, trigger: str, holds: Path | None, reason: str
           ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """The answer for a store that is not read (unset, absent, unreadable or held), else ``None``."""

    if root is None:
        return {"status": "not_configured"}, []
    try:
        with os.scandir(root) as iterator:
            next(iterator, None)
    except FileNotFoundError:
        return {"status": "absent"}, []
    except OSError:
        return {"status": "unreadable"}, [_incomplete(reason)]
    if trigger_held(holds, trigger):
        # The operator paused this store's worker through the door: an intended stop.
        return {"status": "held", "trigger": trigger}, []
    return None


def _json_names(directory: Path) -> list[str] | None:
    """The ``*.json`` names in one queue directory, newest first (none when it is absent).

    ``None`` when it cannot be listed. Newest first, so a truncated walk drops months-old
    finished records, never a request that just arrived or just changed.
    """

    rows: list[tuple[float, str]] = []
    try:
        with os.scandir(directory) as iterator:
            for entry in iterator:
                if not entry.name.endswith(".json"):
                    continue
                try:
                    rows.append((entry.stat(follow_symlinks=False).st_mtime, entry.name))
                except OSError:
                    rows.append((0.0, entry.name))
    except FileNotFoundError:
        return []
    except OSError:
        return None
    return [name for _mtime, name in sorted(rows, key=lambda row: (-row[0], row[1]))]


def _label(value: Any) -> str | None:
    return value[:120] if isinstance(value, str) and _IDENTIFIER.match(value) and ".." not in value else None


def _codes(value: Any, limit: int = 4) -> list[str]:
    return [str(code)[:200] for code in value if isinstance(code, str)][:limit] if isinstance(value, list) else []


def open_window_captures(config: StallConfig, *, now: float, budget: _Budget) -> set[str] | None:
    """The captures with a scene intent whose execution window is still open, or ``None`` when that is unknown.

    A team scene request, a rigid probe or a screening campaign of a capture whose every intent is
    revoked or past its window (extensions included) cannot be funded or run again: its row is
    history, not a stall, by the rule ``check_scene_progressions`` applies to an intent's own row.
    Without it a refusal from months ago would page every hour for good. Every listed intent is
    read, not only the newest ``MAX_INTENTS``: a capture left out would count as closed. A listing
    that was cut, or a scan the deadline ended, is unknown (``None``), and unknown never hides a
    row. An intent whose window cannot be read counts as open.
    """

    from .website_preparation_revision_identity import capture_id

    root = config.scene_intent_root
    if root is None:
        return None
    try:
        entries, truncated = _scene_intent_entries(root)
    except OSError:
        return None
    if truncated:
        return None
    captures: set[str] = set()
    for _mtime, name in entries:
        if budget.expired():
            return None
        directory = root / name
        try:
            if (directory / "revoked.json").exists():
                continue
            intent = _read_json(directory / "intent.json")
            if _execution_expired(directory, intent, now=now) is True:
                continue
            request = intent["request"]
            try:
                captures.add(str(capture_id(request)))
            except (ValueError, KeyError, TypeError, AttributeError):
                # A request whose revision marker does not verify still names its submission.
                captures.add(str(request["submission_id"]))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue  # an intent that does not read names no capture
    return captures


def _window_closed(windows: Callable[[], set[str] | None] | None, capture: str | None) -> bool:
    """Whether ``capture`` is known to have no open execution window; unknown is never closed."""

    if windows is None or capture is None:
        return False
    captures = windows()
    return captures is not None and capture not in captures


def _atlas_hold(code: Any) -> str | None:
    """Why a recorded Atlas refusal is an intended stop: ``store_hold``'s words, or the worker's own reading.

    The worker sends a refusal ``job_retry_policy.hard_stop_kind`` reads as money to ``needs_input`` and
    one it reads as privacy or rights to ``failed``; the alarm reads the same code the same way.
    """

    if not isinstance(code, str) or not code:
        return None
    hold = store_hold([code])
    if hold is not None:
        return hold
    kind = hard_stop_kind(code)
    return "money_approval" if kind == MONEY else "privacy_rights" if kind == PRIVACY_RIGHTS else None


def check_atlas_previews(config: StallConfig, *, now: float, budget: _Budget, holds: Path | None = None
                         ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A job with no progress for too long that is not in a scheduled wait still in the future.

    What the worker (``website_atlas_preview``) writes, read here:

    - ``updated_at_epoch``: every step that returns (``processing`` at a new or the same stage, or a
      new state). This is progress. ``enqueued_at_epoch`` stands in before the first step.
    - ``next_attempt_at_epoch`` and ``retry_at_epoch``: the job's next attempt is the later of the
      two. An outage sets the first by capped backoff (``transient_failures``,
      ``last_transient_reason``), an unclassified refusal or a killed step by the six timed retries
      (``refusal_retries``, ``last_refusal_reason``); a ``needs_input`` recheck sets the second.
    - ``step_in_flight.since_epoch``: a step that started and has not returned. The next tick after
      a kill turns it into a timed retry, so an old one means no tick came.

    A ``processing`` job pages when its step has been in flight for the threshold, or when the
    threshold has passed since the later of its last progress and its next attempt time (a wait
    still in the future is quiet; an attempt overdue by the threshold means the worker is not
    taking it). A ``needs_input`` job is quiet, except a recheck the worker scheduled for a refusal
    that is no hold and that is overdue by the threshold. A refusal that names privacy, rights,
    consent or money never pages. ``ready``, ``failed`` and ``withdrawn`` are final.
    """

    root = config.atlas_storage_root
    skipped = _store(root, ATLAS_TRIGGER, holds, "atlas_preview_root_unreadable")
    if skipped is not None:
        return skipped
    found: list[tuple[float, Path]] = []
    try:
        for path in root.glob(ATLAS_JOB_GLOB):
            try:
                found.append((path.lstat().st_mtime, path))
            except OSError:
                found.append((0.0, path))
    except OSError:
        return {"status": "unreadable"}, [_incomplete("atlas_preview_root_unreadable")]
    # Newest first: a live job's file was written within its 7-day window; retained finished ones age out.
    paths = [path for _mtime, path in sorted(found, key=lambda row: (-row[0], str(row[1])))]
    counts: dict[str, int] = {}
    stalled: list[dict[str, Any]] = []
    unreadable = examined = 0
    truncated = len(paths) > MAX_ATLAS_JOBS
    threshold = config.atlas_seconds
    for path in paths[:MAX_ATLAS_JOBS]:
        if budget.expired():
            truncated = True
            break
        try:
            job = _read_json(path)
        except FileNotFoundError:
            continue  # retired since the listing
        except (OSError, ValueError):
            unreadable += 1
            continue
        examined += 1
        state = job.get("state")
        if job.get("schema_version") != ATLAS_JOB_SCHEMA or not isinstance(state, str):
            unreadable += 1
            continue
        if state not in ATLAS_WAITING_STATES:
            counts[state[:40]] = counts.get(state[:40], 0) + 1  # ready, failed, withdrawn: final
            continue
        progress = _number(job.get("updated_at_epoch"))
        progress = progress if progress is not None else _number(job.get("enqueued_at_epoch"))
        if progress is None:
            unreadable += 1
            continue
        due = max(_number(job.get("next_attempt_at_epoch")) or 0.0, _number(job.get("retry_at_epoch")) or 0.0)
        outages, refusals = job.get("transient_failures"), job.get("refusal_retries")
        outages = outages if type(outages) is int and outages > 0 else None
        refusals = refusals if type(refusals) is int and refusals > 0 else None
        # The refusal behind the current wait: an outage streak, the refusal schedule, or the state's reason.
        cause = (job.get("last_transient_reason") if outages else job.get("last_refusal_reason") if refusals
                 else job.get("reason"))
        hold = _atlas_hold(cause)
        if state == "needs_input":
            if hold is not None:
                counts[hold] = counts.get(hold, 0) + 1
            elif _number(job.get("retry_at_epoch")) is None:
                counts["needs_input"] = counts.get("needs_input", 0) + 1  # waits for a person, not the clock
            elif now - due < threshold:
                counts["scheduled_wait"] = counts.get("scheduled_wait", 0) + 1
            else:
                stalled.append(_atlas_row(job, path, threshold, "recheck_overdue", now - due, cause))
            continue
        flight = job.get(ATLAS_STEP_IN_FLIGHT)
        started = _number(flight.get("since_epoch")) if isinstance(flight, Mapping) else None
        if started is not None:
            if now - started < threshold:
                counts["step_running"] = counts.get("step_running", 0) + 1
            else:
                stalled.append(_atlas_row(job, path, threshold, "step_unreturned", now - started, None))
            continue
        if due > now:
            counts["scheduled_wait"] = counts.get("scheduled_wait", 0) + 1
            continue
        age = now - max(progress, due)
        if age < threshold:
            counts["watched"] = counts.get("watched", 0) + 1
        elif hold is not None and (outages or refusals):
            counts[hold] = counts.get(hold, 0) + 1
        else:
            stalled.append({**_atlas_row(job, path, threshold, "retry_overdue" if due > progress else "silent", age,
                                         cause if (outages or refusals) else None),
                            "transient_failures": outages, "refusal_retries": refusals})
    section = {"status": "observed", "examined": examined, "inventory": len(paths), "truncated": truncated,
               "unreadable": unreadable, "stalled": len(stalled), "dispositions": dict(sorted(counts.items()))}
    alerts = [_page("atlas_preview_stalled", stalled)] if stalled else []
    if truncated:
        alerts.append(_incomplete("atlas_preview_inventory_truncated", count=len(paths)))
    if unreadable:
        alerts.append(_incomplete("atlas_preview_job_unreadable", count=unreadable))
    return section, alerts


def _atlas_row(job: Mapping[str, Any], path: Path, threshold: float, status: str, age: float,
               cause: Any) -> dict[str, Any]:
    # ``phase`` is the job's stage: the door's summary keeps that key of a stall page.
    return {"target": _label(job.get("capture_id")) or path.parents[2].name[:120],
            "phase": job["stage"][:60] if isinstance(job.get("stage"), str) else None,
            "status": status, "age_seconds": round(age), "threshold_seconds": round(threshold),
            "blockers": _codes([cause]) or None}


def check_team_scene_packets(
    config: StallConfig, *, now: float, budget: _Budget, holds: Path | None = None,
    windows: Callable[[], set[str] | None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A request waiting in ``pending/`` or ``processing/``, or left ``blocked``, for too long.

    A request's age in a queue directory is its ctime there (every move into one is a rename or a
    link); a blocked status's is the status file's last write. A blocked request back in a queue
    directory is counted there, not as blocked. ``windows`` answers ``open_window_captures``.
    """

    root = config.team_scene_queue_root
    skipped = _store(root, TEAM_SCENE_TRIGGER, holds, "team_scene_queue_unreadable")
    if skipped is not None:
        return skipped
    listed = {state: _json_names(root / state) for state in (*TEAM_SCENE_WAITING_STATES, "status")}
    if any(names is None for names in listed.values()):
        return {"status": "unreadable"}, [_incomplete("team_scene_queue_unreadable")]
    counts: dict[str, int] = {}
    stalled: list[dict[str, Any]] = []
    unreadable = 0
    truncated = (any(len(listed[state]) > MAX_QUEUE_ENTRIES for state in TEAM_SCENE_WAITING_STATES)
                 or len(listed["status"]) > MAX_STATUS_ENTRIES)
    waiting: set[str] = set()
    for state in TEAM_SCENE_WAITING_STATES:
        waiting.update(listed[state])
        for name in listed[state][:MAX_QUEUE_ENTRIES]:
            if budget.expired():
                truncated = True
                break
            try:
                age = now - (root / state / name).lstat().st_ctime
            except FileNotFoundError:
                continue  # claimed or finished since the listing
            except OSError:
                unreadable += 1
                continue
            if age < config.team_scene_seconds:
                counts[state] = counts.get(state, 0) + 1
                continue
            try:
                label = _label(_read_json(root / state / name).get("capture_id"))
            except (OSError, ValueError):
                label = None
            if _window_closed(windows, label):
                counts["execution_window"] = counts.get("execution_window", 0) + 1
                continue
            counts[state] = counts.get(state, 0) + 1
            stalled.append({"target": name[:-5][:120], "label": label, "status": state, "age_seconds": round(age),
                            "threshold_seconds": round(config.team_scene_seconds)})
    for name in [name for name in listed["status"][:MAX_STATUS_ENTRIES] if name not in waiting]:
        if budget.expired():
            truncated = True
            break
        path = root / "status" / name
        try:
            status = _read_json(path)
            age = now - path.lstat().st_mtime
        except FileNotFoundError:
            continue
        except (OSError, ValueError):
            unreadable += 1
            continue
        if status.get("status") != "blocked":
            continue  # completed or skipped: finished
        blockers = _codes(status.get("blockers"), limit=16)
        hold = ("owner_confirmation" if any(code.startswith(TEAM_SCENE_OWNER_CONTRACT_PREFIX) for code in blockers)
                else store_hold(blockers))
        if hold is not None:
            counts[hold] = counts.get(hold, 0) + 1
            continue
        due = _team_scene_retry_due(status)
        if due is not None and due > now:
            counts["scheduled_wait"] = counts.get("scheduled_wait", 0) + 1
            continue
        if due is not None:
            # Since its timed retry fell due, if that is later: the worker has not returned it to pending.
            age = min(age, now - due)
        label = _label(status.get("capture_id"))
        if label is None and age >= config.team_scene_seconds:
            try:  # a status written before the request was read: the retained request names the capture
                label = _label(_read_json(root / "processed" / name).get("capture_id"))
            except (OSError, ValueError):
                label = None
        if age >= config.team_scene_seconds and _window_closed(windows, label):
            counts["execution_window"] = counts.get("execution_window", 0) + 1
            continue
        counts["blocked"] = counts.get("blocked", 0) + 1
        if age >= config.team_scene_seconds:
            stalled.append({"target": name[:-5][:120], "label": label,
                            "status": "blocked" if due is None else "retry_overdue",
                            "blockers": blockers[:4], "age_seconds": round(age),
                            "threshold_seconds": round(config.team_scene_seconds)})
    section = {"status": "observed", "truncated": truncated, "unreadable": unreadable, "stalled": len(stalled),
               "dispositions": dict(sorted(counts.items()))}
    alerts = [_page("team_scene_packet_stalled", stalled)] if stalled else []
    if truncated:
        alerts.append(_incomplete("team_scene_queue_truncated"))
    if unreadable:
        alerts.append(_incomplete("team_scene_request_unreadable", count=unreadable))
    return section, alerts


def _team_scene_retry_due(status: Mapping[str, Any]) -> float | None:
    """When a blocked status's next timed retry falls due (``not_before``), or ``None`` without one.

    The worker writes ``not_before`` on ``automatic_retry`` (its transient backoff) or on
    ``refusal_retry`` (the shared refusal schedule) and removes it when it returns the request to
    pending; a spent schedule says ``exhausted`` instead. A record whose time does not read names no wait.
    """

    times = []
    for field in TEAM_SCENE_RETRY_FIELDS:
        record = status.get(field)
        if isinstance(record, Mapping) and record.get("exhausted") is not True:
            times.append(_epoch(record.get("not_before")))
    known = [value for value in times if value is not None]
    return max(known) if known else None


def _rigid_probe_last_attempt(root: Path, name: str, entry: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
    """The runner's own record of the entry's latest attempt (its ``state`` and blockers), and when the
    attempt's directory last changed (0 without one): the runner writes each of its records there."""

    digest = str(entry.get("probe_request_digest") or "").removeprefix("sha256:")[:16]
    if not _HEX.match(digest):
        return {}, 0.0
    attempts = root / "jobs" / f"{name[:-5]}-{digest}" / "attempts"
    try:
        with os.scandir(attempts) as iterator:
            numbers = sorted(item.name for item in iterator if re.fullmatch(r"[0-9][0-9]", item.name))
        if not numbers:
            return {}, 0.0
        changed = (attempts / numbers[-1]).lstat().st_mtime
    except OSError:
        return {}, 0.0  # no attempt staged
    try:
        return _read_json(attempts / numbers[-1] / RIGID_PROBE_INGEST_FILENAME), changed
    except (OSError, ValueError):
        return {}, changed  # an attempt still running, or one whose fate is unknown: nothing recorded yet


def _rigid_probe_timed_retry(root: Path, name: str, entry: Mapping[str, Any]) -> tuple[float, str] | None:
    """When the runner sets a retired entry's staging refusal aside again, and that refusal; else ``None``.

    The runner's own rule (``_refusal_retry_due``): a ``staging_refused`` terminal whose recorded
    reason ``hard_stop_kind`` reads as no hard stop, with a step of the shared schedule left, is due
    ``refusal_retry_delay(n)`` after it was sealed, where ``n`` counts the job's earlier
    ``refusal_retry`` set-asides (its supersession markers). A marker that does not read counts as
    spent there, so here it ends the wait too. Raises ``OSError`` or ``ValueError`` when the terminal
    exists but does not read.
    """

    digest = str(entry.get("probe_request_digest") or "").removeprefix("sha256:")[:16]
    if not _HEX.match(digest):
        return None
    job = root / "jobs" / f"{name[:-5]}-{digest}"
    try:
        terminal = _read_json(job / RIGID_PROBE_TERMINAL_FILENAME)
    except FileNotFoundError:
        return None
    detail = terminal.get("detail")
    reason = detail.get("reason") if isinstance(detail, Mapping) else None
    sealed = _epoch(terminal.get("sealed_at_iso"))
    if (terminal.get("status") not in RIGID_PROBE_TIMED_STATES or not isinstance(reason, str) or not reason
            or hard_stop_kind(reason) is not None or sealed is None):
        return None
    done = 0
    try:
        with os.scandir(job / RIGID_PROBE_SUPERSEDED_DIRNAME) as iterator:
            markers = sorted(item.name for item in iterator if item.name.endswith(".supersession.json"))
    except FileNotFoundError:
        markers = []
    if len(markers) > MAX_SUPERSESSIONS:
        return None
    for marker in markers:
        try:
            done += _read_json(job / RIGID_PROBE_SUPERSEDED_DIRNAME / marker).get("trigger") == (
                RIGID_PROBE_REFUSAL_RETRY_TRIGGER)
        except (OSError, ValueError):
            return None
    delay = refusal_retry_delay(done)
    return None if delay is None else (sealed + delay, reason[:200])


def check_rigid_probes(
    config: StallConfig, *, now: float, budget: _Budget, holds: Path | None = None,
    windows: Callable[[], set[str] | None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """An entry in ``pending/`` whose latest attempt has recorded nothing new for too long, or one in
    ``done/`` whose timed staging retry is overdue by as long.

    Every pass retires a decided or unprobeable entry to ``done/``. An entry whose capture's grant
    ran out stays pending with nothing recorded (the runner's precheck defers it before any
    attempt), which is why ``windows`` (``open_window_captures``) is asked. A retired staging
    refusal waiting for its next timed retry (``_rigid_probe_timed_retry``) is quiet until it is due.
    """

    root = config.rigid_probe_queue_root
    skipped = _store(root, RIGID_PROBE_TRIGGER, holds, "rigid_probe_queue_unreadable")
    if skipped is not None:
        return skipped
    names = _json_names(root / "pending")
    if names is None:
        return {"status": "unreadable"}, [_incomplete("rigid_probe_queue_unreadable")]
    counts: dict[str, int] = {}
    stalled: list[dict[str, Any]] = []
    unreadable = 0
    truncated = len(names) > MAX_QUEUE_ENTRIES
    for name in names[:MAX_QUEUE_ENTRIES]:
        if budget.expired():
            truncated = True
            break
        path = root / "pending" / name
        try:
            # A requeue renames the entry back from done/, which sets its ctime.
            queued = path.lstat().st_ctime
            entry = _read_json(path)
        except FileNotFoundError:
            continue  # retired since the listing
        except (OSError, ValueError):
            unreadable += 1
            continue
        last, changed = _rigid_probe_last_attempt(root, name, entry)
        # Since the entry was queued or its latest attempt last wrote a record, whichever is later:
        # a run started after a long money stop is not a stall.
        age = now - max(queued, changed)
        state = last.get("state") if isinstance(last.get("state"), str) else None
        blockers = _codes(last.get("allocator_blockers"), limit=16)
        # The WebApp refused the capture's grant (cap, ceiling, terms, expiry): the owner lifts it.
        hold = "money_approval" if state == RIGID_PROBE_GRANT_REFUSED_STATE else store_hold(blockers)
        if hold is not None:
            counts[hold] = counts.get(hold, 0) + 1
            continue
        capture = Path(str(entry.get("capture_root") or "")).name
        if age >= config.rigid_probe_seconds and _window_closed(windows, _label(capture)):
            counts["execution_window"] = counts.get("execution_window", 0) + 1
            continue
        counts["watched"] = counts.get("watched", 0) + 1
        if age >= config.rigid_probe_seconds:
            stalled.append({"target": name[:-5][:120], "label": _label(capture),
                            # No record: its allocator is still running, or its fate is unknown.
                            "status": (state or ("attempt_unsettled" if changed else "no_attempt_recorded"))[:60],
                            "blockers": blockers[:4], "age_seconds": round(age),
                            "threshold_seconds": round(config.rigid_probe_seconds)})
    # A retired entry whose staging refusal waits for its next timed retry: quiet until it falls due,
    # a page once it is overdue by the threshold and the runner has not returned it to pending.
    retired = _json_names(root / "done")
    if retired is None:
        unreadable += 1
        retired = []
    truncated = truncated or len(retired) > MAX_STATUS_ENTRIES
    pending = set(names)
    for name in [name for name in retired[:MAX_STATUS_ENTRIES] if name not in pending]:
        if budget.expired():
            truncated = True
            break
        try:
            entry = _read_json(root / "done" / name)
            timed = _rigid_probe_timed_retry(root, name, entry)
        except FileNotFoundError:
            continue  # returned to pending since the listing
        except (OSError, ValueError):
            unreadable += 1
            continue
        if timed is None:
            continue  # decided, or final: nothing retries it by the clock
        due, reason = timed
        hold = store_hold([reason])
        capture = _label(Path(str(entry.get("capture_root") or "")).name)
        if hold is not None:
            counts[hold] = counts.get(hold, 0) + 1
        elif due > now:
            counts["scheduled_wait"] = counts.get("scheduled_wait", 0) + 1
        elif now - due < config.rigid_probe_seconds:
            counts["watched"] = counts.get("watched", 0) + 1
        elif _window_closed(windows, capture):
            counts["execution_window"] = counts.get("execution_window", 0) + 1
        else:
            counts["watched"] = counts.get("watched", 0) + 1
            stalled.append({"target": name[:-5][:120], "label": capture, "status": "retry_overdue",
                            "blockers": [reason], "age_seconds": round(now - due),
                            "threshold_seconds": round(config.rigid_probe_seconds)})
    section = {"status": "observed", "pending": len(names), "truncated": truncated, "unreadable": unreadable,
               "stalled": len(stalled), "dispositions": dict(sorted(counts.items()))}
    alerts = [_page("rigid_probe_stalled", stalled)] if stalled else []
    if truncated:
        alerts.append(_incomplete("rigid_probe_queue_truncated", count=len(names)))
    if unreadable:
        alerts.append(_incomplete("rigid_probe_entry_unreadable", count=unreadable))
    return section, alerts


def check_screening(
    config: StallConfig, *, now: float, budget: _Budget, holds: Path | None = None,
    remembered: Mapping[str, Any] | None = None, memory_out: dict[str, Any] | None = None,
    windows: Callable[[], set[str] | None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A campaign with rows still to run and no ledger change, or a revision left unpublished, for too long.

    ``updated_at`` moves with every change to a campaign's ledger entry. The ledger does not say
    since when a revision has been unpublished (a restart resets ``published_revision`` on
    purpose), so ``memory_out`` carries, per campaign, when this alarm first saw it unpublished;
    ``remembered`` is the previous tick's, and one tick that sees it published forgets it.
    ``windows`` answers ``open_window_captures``; a campaign's capture is named by its retained plan.
    """

    remembered = remembered if isinstance(remembered, Mapping) else {}
    root = config.screening_queue_root
    skipped = _store(root, SCREENING_TRIGGER, holds, "screening_queue_unreadable")
    if skipped is not None:
        return skipped
    try:
        ledger = _read_json(root / SCREENING_LEDGER_FILENAME, MAX_LEDGER_BYTES)
    except FileNotFoundError:
        return {"status": "absent"}, []
    except (OSError, ValueError):
        return {"status": "unreadable"}, [_incomplete("screening_ledger_unreadable")]
    campaigns = ledger.get("campaigns")
    if ledger.get("schema_version") != SCREENING_LEDGER_SCHEMA or not isinstance(campaigns, dict):
        return {"status": "unreadable"}, [_incomplete("screening_ledger_unreadable")]
    counts: dict[str, int] = {}
    stalled: list[dict[str, Any]] = []
    unpublished: list[dict[str, Any]] = []
    unreadable = pending_publications = 0
    truncated = len(campaigns) > MAX_CAMPAIGNS

    plan_captures: dict[str, str | None] = {}

    def window_closed(campaign_id: str) -> bool:
        if campaign_id not in plan_captures:
            try:
                plan = _read_json(root / "plans" / f"{campaign_id}.json", MAX_LEDGER_BYTES)
                plan_captures[campaign_id] = _label(plan.get("capture_id"))
            except (OSError, ValueError):
                plan_captures[campaign_id] = None  # unknown: the campaign stays watched
        return _window_closed(windows, plan_captures[campaign_id])

    for campaign_id, campaign in sorted(campaigns.items())[:MAX_CAMPAIGNS]:
        if budget.expired():
            truncated = True
            break
        rows = campaign.get("candidates") if isinstance(campaign, dict) else None
        if _label(campaign_id) is None or not isinstance(rows, dict) or not all(
                isinstance(row, dict) and isinstance(row.get("status"), str) for row in rows.values()):
            unreadable += 1
            continue
        live = [row for row in rows.values() if row["status"] in SCREENING_LIVE_STATUSES]
        row_holds = [store_hold(_codes(row.get("blockers"), limit=16)) for row in live]
        every_hold = [store_hold(_codes(row.get("blockers"), limit=16)) for row in rows.values()]
        updated = _epoch(campaign.get("updated_at"))
        if live and all(hold is not None for hold in row_holds):
            disposition = row_holds[0]
        elif live and updated is None:
            unreadable += 1
            disposition = None
        elif live and now - updated >= config.screening_seconds and window_closed(campaign_id):
            disposition = "execution_window"
        elif live:
            disposition = "watched"
            if now - updated >= config.screening_seconds:
                waits = [row for row, hold in zip(live, row_holds) if hold is None]
                stalled.append({
                    # The row that most needs a person names the campaign's state.
                    "target": campaign_id[:120], "status": next(
                        status for status in _SCREENING_STATUS_ORDER if any(row["status"] == status for row in waits)),
                    "unfinished_rows": len(waits),
                    "blockers": sorted({code for row in waits for code in _codes(row.get("blockers"))})[:4],
                    "age_seconds": round(now - updated), "threshold_seconds": round(config.screening_seconds)})
        elif any(row["status"] == SCREENING_WAITLISTED_STATUS for row in rows.values()):
            # Only the site's allowance or the job's ceiling keeps these rows from running.
            disposition = "money_approval"
        else:
            disposition = "finished"
        if disposition is not None:
            counts[disposition] = counts.get(disposition, 0) + 1
        if campaign.get("revision") == campaign.get("published_revision"):
            continue
        pending_publications += 1
        if any(hold in {"privacy", "rights"} for hold in every_hold):
            continue  # the website ended a withdrawn or revoked job itself; nothing is owed to it
        since = _number(remembered.get(campaign_id))
        since = now if since is None or since > now else since
        if memory_out is not None:
            memory_out[campaign_id] = since
        if now - since >= config.publication_seconds and not window_closed(campaign_id):
            unpublished.append({"target": campaign_id[:120], "age_seconds": round(now - since),
                                "age_is_lower_bound": True,
                                "threshold_seconds": round(config.publication_seconds)})
    section = {"status": "observed", "campaigns": len(campaigns), "truncated": truncated, "unreadable": unreadable,
               "stalled": len(stalled), "publications_pending": pending_publications,
               "publications_stalled": len(unpublished), "dispositions": dict(sorted(counts.items()))}
    alerts = [_page("screening_campaign_stalled", stalled)] if stalled else []
    if unpublished:
        alerts.append(_page("screening_publication_stalled", unpublished))
    if truncated:
        alerts.append(_incomplete("screening_ledger_truncated", count=len(campaigns)))
    if unreadable:
        alerts.append(_incomplete("screening_campaign_unreadable", count=unreadable))
    return section, alerts


# --------------------------------------------------------------------------- #
# One tick
# --------------------------------------------------------------------------- #


def _check_scene_budgets(config: StallConfig, *, now: float, remembered: Any,
                         memory_out: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from .scene_budget_alert import check_scene_budgets
    root = config.scene_intent_root
    if root is None:
        return {"status": "not_configured"}, []
    try:
        entries, _truncated = _scene_intent_entries(root)
    except FileNotFoundError:
        return {"status": "absent"}, []
    return check_scene_budgets(root, [name for _mtime, name in entries[:MAX_INTENTS]], now=now,
                               remembered=remembered, memory_out=memory_out)


def _check_preparation_faults(config: StallConfig, *, now: float, remembered: Any,
                              memory_out: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # Preparation ledgers sit beside the Atlas jobs, under the same capture roots.
    from .preparation_fault_alert import check_preparation_faults
    return check_preparation_faults(config.atlas_storage_root, now=now, remembered=remembered, memory_out=memory_out)


def observe_stalls(
    config: StallConfig,
    *,
    now: float | None = None,
    previous: Mapping[str, Any] | None = None,
    unit_reader: Callable[[Sequence[str]], Mapping[str, Mapping[str, str]]] = read_unit_states,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Every check; ``alerts`` are capacity-report alert rows (severity added there).

    ``previous`` is the last tick's result; its ``memory`` keeps each watched
    row's unchanged-since time across the bounded event walk.
    """

    observed = time.time() if now is None else float(now)
    prior_memory = (previous or {}).get("memory") if isinstance(previous, Mapping) else None
    prior_memory = prior_memory if isinstance(prior_memory, Mapping) else {}
    remembered = prior_memory.get("scene_progression")
    progression_memory: dict[str, Any] = {}
    publication_memory: dict[str, Any] = {}
    scene_budget_memory: dict[str, Any] = {}
    preparation_fault_memory: dict[str, Any] = {}
    budget = _Budget(deadline=monotonic() + DEADLINE_SECONDS, events=MAX_EVENT_READS, clock=monotonic)
    alerts: list[dict[str, Any]] = [_incomplete(error) for error in config.config_errors]
    # The deploy tool's lists and the door's holds directory, read once for every check that asks
    # whether a trigger is held. Unreadable: the trigger check reports it, and no store counts as held.
    try:
        contract = None if config.deploy_tool is None else read_deploy_trigger_contract(config.deploy_tool)
    except StallCheckError:
        contract = None
    holds = None if contract is None else contract[1]
    open_windows: list[set[str] | None] = []

    def windows() -> set[str] | None:
        # Asked only once a store row is past its threshold, and read once for the tick.
        if not open_windows:
            open_windows.append(open_window_captures(config, now=observed, budget=budget))
        return open_windows[0]

    checks: dict[str, Any] = {}
    for name, run in (
        ("scene_progression", lambda: check_scene_progressions(
            config, now=observed, budget=budget, remembered=remembered, memory_out=progression_memory)),
        ("launch_leases", lambda: check_launch_leases(config, now=observed, budget=budget)),
        ("launch_activations", lambda: check_launch_activations(config, now=observed, budget=budget)),
        ("triggers", lambda: check_triggers(config, unit_reader=unit_reader, monotonic=monotonic,
                                            contract=contract)),
        ("atlas_preview", lambda: check_atlas_previews(config, now=observed, budget=budget, holds=holds)),
        ("team_scene_packets", lambda: check_team_scene_packets(
            config, now=observed, budget=budget, holds=holds, windows=windows)),
        ("rigid_probes", lambda: check_rigid_probes(
            config, now=observed, budget=budget, holds=holds, windows=windows)),
        ("screening", lambda: check_screening(
            config, now=observed, budget=budget, holds=holds, windows=windows,
            remembered=prior_memory.get("screening_publication"), memory_out=publication_memory)),
        ("scene_budget", lambda: _check_scene_budgets(
            config, now=observed, remembered=prior_memory.get("scene_budget"), memory_out=scene_budget_memory)),
        ("preparation_faults", lambda: _check_preparation_faults(
            config, now=observed, remembered=prior_memory.get("preparation_fault"),
            memory_out=preparation_fault_memory)),
    ):
        try:
            section, rows = run()
        except Exception as exc:  # noqa: BLE001 - one broken check must not hide the others
            section, rows = {"status": "failed", "reason": type(exc).__name__}, [
                _incomplete(f"{name}_check_failed")]
        checks[name] = section
        alerts.extend(rows)
    if open_windows and open_windows[0] is None and config.scene_intent_root is not None:
        # A store row asked and the intent store could not answer in full: the row stays watched.
        alerts.append(_incomplete("execution_window_scan_incomplete"))
    return {
        "schema_version": SCHEMA_VERSION,
        "observed_at_epoch": observed,
        "thresholds_seconds": {
            "progression": config.progression_seconds,
            "handoff": config.handoff_seconds,
            "launch_grace": config.launch_grace_seconds,
            "trigger": config.trigger_seconds,
            "atlas": config.atlas_seconds,
            "team_scene": config.team_scene_seconds,
            "rigid_probe": config.rigid_probe_seconds,
            "screening": config.screening_seconds,
            "publication": config.publication_seconds,
        },
        "checks": checks,
        "memory": {"scene_progression": progression_memory, "screening_publication": publication_memory,
                   "scene_budget": scene_budget_memory, "preparation_fault": preparation_fault_memory},
        "alerts": alerts,
    }


def describe(alert: Mapping[str, Any]) -> str | None:
    """One human line for a stall page, or ``None`` for any other alert."""

    code = alert.get("code")
    if code not in PAGE_CODES:
        return None
    seconds = float(alert.get("age_seconds") or 0)
    age = (f"{seconds / 86400:.1f}d" if seconds >= 2 * 86400 else f"{seconds / 3600:.1f}h"
           if seconds >= 2 * 3600 else f"{round(seconds / 60)}m")
    if alert.get("age_is_lower_bound"):
        age = "over " + age
    target = str(alert.get("target") or "")
    more = int(alert.get("count") or 1) - 1
    tail = f" (+{more} more)" if more > 0 else ""
    if code == "scene_progression_stalled":
        name = str(alert.get("label") or target[:22])
        blockers = alert.get("blockers") or []
        why = f" [{blockers[0]}]" if blockers else ""
        return (f"scene progression unchanged {age} at {alert.get('status')}/{alert.get('phase')}{why}: "
                f"{name}{tail}")
    if code == "scene_budget_doubled":
        from .scene_budget_alert import describe as describe_budget
        return describe_budget(alert)
    if code == "website_preparation_blueprint_fault":
        from .preparation_fault_alert import describe as describe_fault
        return describe_fault(alert)
    if code == "launch_lease_stuck":
        return f"launch lease in processing/ {age} with no dispatcher: {target}{tail}"
    if code == "launch_activation_stuck":
        return f"launch activation in processing/ {age} with no live worker: {target}{tail}"
    blockers = alert.get("blockers") or []
    why = f" [{blockers[0]}]" if blockers else ""
    name = str(alert.get("label") or target[:40])
    if code == "atlas_preview_stalled":
        status = alert.get("status")
        what = (f"step started {age} ago never returned" if status == "step_unreturned"
                else f"scheduled retry overdue {age}{why}" if status == "retry_overdue"
                else f"input recheck overdue {age}{why}" if status == "recheck_overdue"
                else f"processing with no finished step {age}")
        return f"Atlas preview job {what} at stage {alert.get('phase') or 'unknown'}: {name}{tail}"
    if code == "team_scene_packet_stalled":
        status = "blocked, timed retry overdue" if alert.get("status") == "retry_overdue" else alert.get("status")
        return f"team scene packet request {status} {age}{why}: {name}{tail}"
    if code == "rigid_probe_stalled":
        if alert.get("status") == "retry_overdue":
            return f"rigid robot-free probe timed staging retry overdue {age}{why}: {name}{tail}"
        return f"rigid robot-free probe pending {age} ({alert.get('status')}){why}: {name}{tail}"
    if code == "screening_campaign_stalled":
        return (f"screening campaign without ledger progress {age}, {alert.get('unfinished_rows')} unfinished "
                f"({alert.get('status')}){why}: {name}{tail}")
    if code == "screening_publication_stalled":
        return f"screening publication undelivered {age}: {name}{tail}"
    return f"{target} {alert.get('status') or 'inactive'} {age} with no owner hold{tail}"


__all__ = [
    "INCOMPLETE_CODE",
    "PAGE_CODES",
    "SCHEMA_VERSION",
    "StallCheckError",
    "StallConfig",
    "check_atlas_previews",
    "check_launch_activations",
    "check_launch_leases",
    "check_rigid_probes",
    "check_scene_progressions",
    "check_screening",
    "check_team_scene_packets",
    "check_triggers",
    "config_from_environment",
    "default_deploy_tool",
    "describe",
    "hold_reason",
    "observe_stalls",
    "open_window_captures",
    "read_deploy_trigger_contract",
    "read_lock_holders",
    "read_unit_states",
    "store_hold",
    "trigger_held",
]
