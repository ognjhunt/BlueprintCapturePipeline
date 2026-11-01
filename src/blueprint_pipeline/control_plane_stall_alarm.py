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

Every check only reads: it lists the named directories with limits, reads
bounded JSON, takes the dispatcher lock exactly as the reconciler does to learn
ownership, and runs one ``systemctl show``. It never moves a queue entry,
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

from .task_evaluation_launch_dispatcher_ownership import DISPATCHER_LOCK_NAME, unowned_processing_names

SCHEMA_VERSION = "control_plane_stall_check.v1"
PAGE_CODES = frozenset({"scene_progression_stalled", "launch_lease_stuck", "trigger_inactive_unheld"})
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

DEFAULT_PROGRESSION_MINUTES = 90
DEFAULT_HANDOFF_MINUTES = 8 * 60
DEFAULT_LAUNCH_GRACE_MINUTES = 30
DEFAULT_TRIGGER_MINUTES = 60

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

MAX_INTENT_DIRECTORY_ENTRIES = 10_000
MAX_INTENTS = 256
MAX_EVENTS_PER_INTENT = 64
MAX_EVENT_READS = 512
MAX_LEASES = 256
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


def check_scene_progressions(config: StallConfig, *, now: float, budget: _Budget) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = config.scene_intent_root
    section: dict[str, Any] = {"status": "not_configured"}
    if root is None:
        return section, [_incomplete("scene_intent_root_unconfigured")]
    alerts: list[dict[str, Any]] = []
    entries: list[tuple[float, str]] = []
    truncated = False
    try:
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
    except FileNotFoundError:
        return {"status": "absent"}, []
    except OSError:
        return {"status": "unreadable"}, [_incomplete("scene_intent_root_unreadable")]
    # Newest first: a row that is still changing or newly stalled is never the
    # one a truncated walk drops; months-old revoked rows are.
    entries.sort(key=lambda item: (-item[0], item[1]))
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
# 3. Always-armed triggers left inactive without an owner hold
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


def check_triggers(
    config: StallConfig,
    *,
    unit_reader: Callable[[Sequence[str]], Mapping[str, Mapping[str, str]]] = read_unit_states,
    monotonic: Callable[[], float] = time.monotonic,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if config.deploy_tool is None:
        return {"status": "not_configured"}, [_incomplete("deploy_tool_unconfigured")]
    try:
        units, holds = read_deploy_trigger_contract(config.deploy_tool)
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
        try:
            os.lstat(holds / f"{unit}.json")
            held.append(unit)
            continue
        except FileNotFoundError:
            pass
        except OSError:
            # A hold that cannot be ruled out is not paged as a stall.
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
# One tick
# --------------------------------------------------------------------------- #


def observe_stalls(
    config: StallConfig,
    *,
    now: float | None = None,
    unit_reader: Callable[[Sequence[str]], Mapping[str, Mapping[str, str]]] = read_unit_states,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """All three checks; ``alerts`` are capacity-report alert rows (severity added there)."""

    observed = time.time() if now is None else float(now)
    budget = _Budget(deadline=monotonic() + DEADLINE_SECONDS, events=MAX_EVENT_READS, clock=monotonic)
    alerts: list[dict[str, Any]] = [_incomplete(error) for error in config.config_errors]
    checks: dict[str, Any] = {}
    for name, run in (
        ("scene_progression", lambda: check_scene_progressions(config, now=observed, budget=budget)),
        ("launch_leases", lambda: check_launch_leases(config, now=observed, budget=budget)),
        ("triggers", lambda: check_triggers(config, unit_reader=unit_reader, monotonic=monotonic)),
    ):
        try:
            section, rows = run()
        except Exception as exc:  # noqa: BLE001 - one broken check must not hide the others
            section, rows = {"status": "failed", "reason": type(exc).__name__}, [
                _incomplete(f"{name}_check_failed")]
        checks[name] = section
        alerts.extend(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "observed_at_epoch": observed,
        "thresholds_seconds": {
            "progression": config.progression_seconds,
            "handoff": config.handoff_seconds,
            "launch_grace": config.launch_grace_seconds,
            "trigger": config.trigger_seconds,
        },
        "checks": checks,
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
    if code == "launch_lease_stuck":
        return f"launch lease in processing/ {age} with no dispatcher: {target}{tail}"
    return f"{target} {alert.get('status') or 'inactive'} {age} with no owner hold{tail}"


__all__ = [
    "INCOMPLETE_CODE",
    "PAGE_CODES",
    "SCHEMA_VERSION",
    "StallCheckError",
    "StallConfig",
    "check_launch_leases",
    "check_scene_progressions",
    "check_triggers",
    "config_from_environment",
    "default_deploy_tool",
    "describe",
    "hold_reason",
    "observe_stalls",
    "read_deploy_trigger_contract",
    "read_unit_states",
]
