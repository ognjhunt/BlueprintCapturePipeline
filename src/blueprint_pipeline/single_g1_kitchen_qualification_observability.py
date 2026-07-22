"""Attempt-bound, non-secret observability for the persistent qualification lane."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping


STARTUP_DIAGNOSTICS_SCHEMA_VERSION = (
    "single_g1_kitchen_qualification_startup_diagnostics.v1"
)
STARTUP_STALL_SECONDS = 900


def dispatch_diagnostics_shell() -> str:
    """Return fixed shell/Python that binds a dispatch record to one attempt."""

    return r'''if [ "$1" = episode ] || [ "$1" = bootstrap ]; then
    python3 - /workspace/closed_loop_out/qualification_startup_diagnostics.json "$pid" "$ATTEMPT_SEQUENCE" "$ATTEMPT_NONCE_SHA256" "$EXPECTED_LAUNCH_SESSION_ID" <<'PY' || printf 'qualification_startup_diagnostics_write_failed\n' >&2
import json, os, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": "single_g1_kitchen_qualification_startup_diagnostics.v1",
    "launch_session_id": sys.argv[5],
    "attempt_sequence": int(sys.argv[3]),
    "attempt_nonce_sha256": sys.argv[4],
    "root_pid": int(sys.argv[2]),
    "startup_phase": "episode_process_dispatched",
    "startup_health": "progressing",
    "dispatched_at_epoch": int(time.time()),
    "raw_secret_values_recorded": False,
}
temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.chmod(temporary, 0o600)
os.replace(temporary, path)
PY
  fi'''


def status_diagnostics_shell(*, stall_seconds: int = STARTUP_STALL_SECONDS) -> str:
    """Return fixed status inspection that never records argv, environment, or secrets."""

    source = r'''diagnostics=""
    if [ "$COMPONENT" = episode ] || [ "$COMPONENT" = bootstrap ]; then
      diagnostics=$(python3 - /workspace/closed_loop_out/qualification_startup_diagnostics.json /workspace/bootstrap.json "$(log_path "$COMPONENT")" "$STATE/$COMPONENT.pid" "$state" __STALL_SECONDS__ "$attempt_sequence" "$attempt_nonce_sha256" <<'PY'
import json, os, pathlib, re, sys, time

diagnostic_path = pathlib.Path(sys.argv[1])
bootstrap_path = pathlib.Path(sys.argv[2])
log_path = pathlib.Path(sys.argv[3])
pid_path = pathlib.Path(sys.argv[4])
process_state = sys.argv[5]
stall_seconds = int(sys.argv[6])
expected_attempt_sequence = sys.argv[7]
expected_attempt_nonce_sha256 = sys.argv[8]
now = int(time.time())

def safe_regular(path):
    return path.is_file() and not path.is_symlink()

diagnostic = {}
diagnostic_record_valid = False
if safe_regular(diagnostic_path):
    try:
        value = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            diagnostic = value
            diagnostic_record_valid = True
    except Exception:
        diagnostic = {}

phase = "progress_marker_missing"
phase_epoch = int(diagnostic.get("dispatched_at_epoch") or now)
phase_source = "dispatch_record"
if safe_regular(bootstrap_path):
    try:
        phase_epoch = int(bootstrap_path.stat().st_mtime)
        bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
        candidate = str(bootstrap.get("phase") or "")
        if re.fullmatch(r"[a-z0-9_.-]{1,96}", candidate):
            phase = candidate
            phase_source = "bootstrap_json"
    except Exception:
        phase = "progress_marker_invalid"
        phase_source = "bootstrap_json_invalid"

diagnostic_binding = "not_applicable"
if expected_attempt_sequence:
    if (
        diagnostic_record_valid
        and str(diagnostic.get("attempt_sequence")) == expected_attempt_sequence
        and diagnostic.get("attempt_nonce_sha256") == expected_attempt_nonce_sha256
    ):
        diagnostic_binding = "valid"
    else:
        diagnostic_binding = "missing_or_mismatch"

phase_age = max(0, now - phase_epoch)
terminal_phases = {
    "runner_done",
    "runner_timeout",
}
if process_state == "stopped":
    startup_health = "stopped"
elif diagnostic_binding == "missing_or_mismatch":
    startup_health = "stalled"
elif phase in terminal_phases:
    startup_health = "ready_or_terminal"
elif phase_age >= stall_seconds:
    startup_health = "stalled"
else:
    startup_health = "progressing"

log_bytes = 0
log_age = -1
if safe_regular(log_path):
    log_stat = log_path.stat()
    log_bytes = int(log_stat.st_size)
    log_age = max(0, now - int(log_stat.st_mtime))

root_pid = 0
root_pid_state = "missing"
root_pid_elapsed = -1
if safe_regular(pid_path):
    try:
        raw_pid = pid_path.read_text(encoding="ascii", errors="ignore").strip()
    except OSError:
        raw_pid = ""
    if raw_pid.isdigit():
        root_pid = int(raw_pid)
        stat_path = pathlib.Path(f"/proc/{root_pid}/stat")
        if safe_regular(stat_path):
            try:
                fields = stat_path.read_text(
                    encoding="ascii", errors="replace"
                ).split()
                if len(fields) >= 22:
                    uptime = float(pathlib.Path("/proc/uptime").read_text().split()[0])
                    ticks = int(os.sysconf("SC_CLK_TCK"))
                    root_pid_state = fields[2]
                    root_pid_elapsed = max(
                        0, int(uptime - (int(fields[21]) / ticks))
                    )
            except (OSError, ValueError, IndexError):
                root_pid_state = "missing"
                root_pid_elapsed = -1

diagnostic.update({
    "schema_version": "single_g1_kitchen_qualification_startup_diagnostics.v1",
    "startup_phase": phase,
    "startup_phase_source": phase_source,
    "startup_health": startup_health,
    "startup_phase_age_seconds": phase_age,
    "startup_stall_threshold_seconds": stall_seconds,
    "diagnostic_attempt_binding": diagnostic_binding,
    "remote_process_state": process_state,
    "root_pid": root_pid or diagnostic.get("root_pid"),
    "root_pid_state": root_pid_state,
    "root_pid_elapsed_seconds": root_pid_elapsed,
    "log_size_bytes": log_bytes,
    "log_age_seconds": log_age,
    "observed_at_epoch": now,
    "raw_secret_values_recorded": False,
})
temporary = diagnostic_path.with_name(
    "." + diagnostic_path.name + "." + str(os.getpid()) + ".tmp"
)
if diagnostic_path.parent.is_dir() and not diagnostic_path.parent.is_symlink():
    temporary.write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, diagnostic_path)
print(
    "startup_phase=" + phase,
    "startup_health=" + startup_health,
    "phase_age_seconds=" + str(phase_age),
    "log_bytes=" + str(log_bytes),
    "log_age_seconds=" + str(log_age),
    "root_pid=" + str(root_pid),
    "root_pid_state=" + root_pid_state,
    "root_pid_elapsed_seconds=" + str(root_pid_elapsed),
    "diagnostic_binding=" + diagnostic_binding,
)
PY
)
    fi'''
    return source.replace("__STALL_SECONDS__", str(int(stall_seconds)))


def parse_startup_diagnostics(
    stdout: str, *, observed_at: str
) -> tuple[dict[str, Any] | None, list[str]]:
    """Normalize fixed-control stdout without treating process life as model proof."""

    match = re.search(
        r"\bstartup_phase=([a-z0-9_.-]+)\s+"
        r"startup_health=(progressing|ready_or_terminal|stalled|stopped)\s+"
        r"phase_age_seconds=(\d+)\s+log_bytes=(\d+)\s+"
        r"log_age_seconds=(-?\d+)\s+root_pid=(\d+)\s+"
        r"root_pid_state=([A-Za-z]+|missing)\s+"
        r"root_pid_elapsed_seconds=(-?\d+)\s+"
        r"diagnostic_binding=(valid|missing_or_mismatch|not_applicable)\b",
        stdout,
    )
    if not match:
        return None, []
    diagnostics = {
        "schema_version": STARTUP_DIAGNOSTICS_SCHEMA_VERSION,
        "startup_phase": match.group(1),
        "startup_health": match.group(2),
        "startup_phase_age_seconds": int(match.group(3)),
        "log_size_bytes": int(match.group(4)),
        "log_age_seconds": int(match.group(5)),
        "root_pid": int(match.group(6)),
        "root_pid_state": match.group(7),
        "root_pid_elapsed_seconds": int(match.group(8)),
        "diagnostic_attempt_binding": match.group(9),
        "observed_at": observed_at,
        "raw_secret_values_recorded": False,
    }
    blockers: list[str] = []
    if diagnostics["startup_health"] == "stalled":
        blockers.append("qualification_episode_startup_progress_stalled")
    if diagnostics["diagnostic_attempt_binding"] == "missing_or_mismatch":
        blockers.append("qualification_episode_startup_diagnostics_binding_invalid")
    if (
        diagnostics["startup_health"] == "stopped"
        and diagnostics["diagnostic_attempt_binding"] == "valid"
        and diagnostics["startup_phase"] not in {"runner_done", "runner_timeout"}
    ):
        blockers.append("qualification_episode_process_stopped_before_terminal_phase")
    return diagnostics, blockers


def collected_isaac_frames_directory(collected_root: Path) -> Path | None:
    candidates = [
        collected_root / "closed_loop_out" / "isaac_task_state" / "frames",
        collected_root / "closed_loop_out" / "episode_001" / "isaac_task_state" / "frames",
    ]
    observed = [path for path in candidates if path.is_dir() and not path.is_symlink()]
    if len(observed) > 1:
        raise ValueError("qualification_collected_isaac_frames_directory_ambiguous")
    return observed[0] if observed else None


def relative_collected_artifact_paths(collected_root: Path) -> dict[str, Any]:
    frames = collected_isaac_frames_directory(collected_root)

    def relative_files(pattern: str) -> list[str]:
        if frames is None:
            return []
        return [
            str(path.relative_to(collected_root))
            for path in sorted(frames.glob(pattern))
            if path.is_file() and not path.is_symlink()
        ]

    def relative_file(path: Path) -> str | None:
        return (
            str(path.relative_to(collected_root))
            if path.is_file() and not path.is_symlink()
            else None
        )

    episode_dir = collected_root / "closed_loop_out" / "episode_001"
    log_names = (
        "closed_loop_out/qualification_episode.log",
        "groot_server.log",
        "gear_sonic_controller.log",
        "gear_sonic_isaac_dds_bridge.log",
        "isaac_task_executor.log",
        "closed_loop_stdout.log",
        "closed_loop_stderr.log",
    )
    return {
        "overview_frames": relative_files("overview_*.png"),
        "robot_pov_frames": relative_files("robot_pov_*.png"),
        "final_review_video": relative_file(episode_dir / "final_review.mp4"),
        "overview_review_video": relative_file(episode_dir / "isaac_overview_review.mp4"),
        "robot_pov_review_video": relative_file(episode_dir / "isaac_robot_pov_review.mp4"),
        "wam_prediction_review_video": relative_file(episode_dir / "wam_prediction_review.mp4"),
        "final_review_validation": relative_file(episode_dir / "final_review_validation.json"),
        "runner_result": relative_file(collected_root / "isaac_runtime_result.json"),
        "qualification_attempt": relative_file(
            collected_root / "closed_loop_out" / "qualification_attempt.json"
        ),
        "startup_diagnostics": relative_file(
            collected_root
            / "closed_loop_out"
            / "qualification_startup_diagnostics.json"
        ),
        "logs": [
            name
            for name in log_names
            if (collected_root / name).is_file()
            and not (collected_root / name).is_symlink()
        ],
    }


def absolute_collected_artifact_paths(
    collected_root: Path, relative_paths: Mapping[str, Any]
) -> dict[str, Any]:
    absolute: dict[str, Any] = {}
    for key, value in relative_paths.items():
        if isinstance(value, list):
            absolute[key] = [str(collected_root / str(item)) for item in value]
        elif value:
            absolute[key] = str(collected_root / str(value))
        else:
            absolute[key] = None
    return absolute
