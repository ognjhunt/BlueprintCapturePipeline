"""What a failed beta screening run predicts: run it again, hold its siblings, or neither.

Beta journey: a prepared site's candidate screening. A candidate that was
allocated and returned no score used to be one thing, a harness failure: it
spent one of the candidate's harness retries, held every other candidate of the
campaign unspent, and waited for a deploy (2026-11-05: a cuDNN wheel download
that hit its time limit held four candidates; 2026-11-09: a rented machine that
could not reach ``archive.ubuntu.com`` held one). The run already returns typed
codes that say more. This module reads only those codes, never free text, and
sorts a failure into one of three scopes.

1. **Transient infrastructure** (``TRANSIENT_PROVISIONING_BLOCKERS``). The
   rented machine, its network or its package mirror failed before our harness
   could say anything about the policy. The candidate is planned again at the
   same commit under the existing machine-fault bound
   (``MACHINE_FAULT_RETRY_LIMIT``, one counter for both), inside its own cap and
   the site limit. It spends no harness retry and holds no sibling. Evidence:

   - ``rented_machine``: the Vast adapter itself judged the machine at fault
     and put it on the avoid list (``_machine_avoidlist_reason``: the bundle
     download timed out on the machine, the instance exited before the bundle
     reported an end, its asset sandbox was unavailable, or it never reached
     the start-up heartbeat);
   - ``bundle_download``: the machine could not fetch our bundle
     (``provider_remote_blocker:download_failed:<curl code>``);
   - ``package_mirror``: the entrypoint could not install the media toolchain
     from the distribution's archive (``policy_canary_media_toolchain_unavailable``;
     on the G1 lane every cell is void with ``g1_beta_media_toolchain_unavailable``);
   - ``provider_log_silent``: the bundle started, no end marker and no output
     came back, and the run's own log record says the wait ended because the
     provider log stopped or could not be read (``break_reason`` in
     ``vast_startup_probe_manifest.json``). A run that used its whole time
     (``max_wait_deadline_reached``) is not this class.

2. **Candidate-specific** (``CANDIDATE_SPECIFIC_BLOCKERS``). The simulator
   opened and the scene passed its gate; the candidate's own policy could not
   be provisioned or loaded (its client dependency, its checkpoint). It is a
   harness failure of that candidate alone: it keeps its harness retry rule
   (planned again after a deploy) and holds no sibling.

3. **Shared stage** (every other harness failure). The simulator could not
   start, the scene could not load, a common import failed, or the run returned
   nothing usable with no stage named. Every candidate on the same runtime
   backend would fail the same way, so the pending ones on that backend are
   held unspent until the next deploy, as before. A candidate on another
   backend is not held: its bundle, entrypoint and simulator session differ.

What the returned codes cannot tell apart stays in the wider scope. A
candidate's policy provisioning (``uv``, a source clone, a wheel or checkpoint
download) reports its stage (``policy_canary_policy_provisioning_failed``,
written by the beta entrypoint), not whether the network or our script failed:
it is candidate-specific, not transient. The G1 lane's result does not separate
the candidate's checkpoint from the runtime every G1 candidate shares, so a G1
failure other than the media toolchain stays a shared-stage failure.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .policy_canary_cell_setup_failure import BLOCKER_PREFIX as SETUP_FAILURE_PREFIX

#: "session" keeps the website's launch wording ("Could not start in our simulator. This is ours to fix.").
TRANSIENT_PROVISIONING_PREFIX = "beta_screening_session_provisioning_transient"
TRANSIENT_CLASSES = ("rented_machine", "bundle_download", "package_mirror", "provider_log_silent")
TRANSIENT_PROVISIONING_BLOCKERS = frozenset(f"{TRANSIENT_PROVISIONING_PREFIX}:{name}" for name in TRANSIENT_CLASSES)
#: The entrypoints' own typed refusal when the distribution's archive did not deliver the media tools.
MEDIA_TOOLCHAIN_CODES = frozenset({"policy_canary_media_toolchain_unavailable",
                                   "g1_beta_media_toolchain_unavailable"})
#: A G1 run closes with every cell void and this reason when its entrypoint refused before any cell.
TRANSIENT_VOID_REASONS = frozenset({"g1_beta_media_toolchain_unavailable"})
BUNDLE_DOWNLOAD_PREFIX = "provider_remote_blocker:download_failed:"
#: The bundle never reported an end, and no output archive reached the host.
_NO_OUTPUT_CODES = frozenset({"provider_bundle_completion_marker_missing",
                              "provider_runtime_output_zip_not_received_locally"})
#: Why the adapter stopped waiting on the provider log when the log itself went dark.
LOG_SILENT_BREAK_REASONS = frozenset({"no_log_progress_timeout", "log_transport_unavailable"})
STARTUP_PROBE_MANIFEST = "vast_startup_probe_manifest.json"
_MAX_MANIFEST_BYTES = 32 * 1024 * 1024

#: The beta entrypoint's fallback code when a candidate's own policy provisioning script failed.
POLICY_PROVISIONING_STAGE_CODE = "policy_canary_policy_provisioning_failed"
#: The row's code for it. "session" keeps the website's launch wording.
POLICY_PROVISIONING_BLOCKER = "beta_screening_session_policy_provisioning_failed"
CANDIDATE_SPECIFIC_BLOCKERS = frozenset({f"{SETUP_FAILURE_PREFIX}:policy_load", POLICY_PROVISIONING_BLOCKER})


def _codes(value: Any) -> list[str]:
    return [code for code in value or [] if isinstance(code, str)] if isinstance(value, (list, tuple)) else []


def provider_log_break_reason(adapter: Mapping[str, Any], *, attempt_dir: str | Path) -> str | None:
    """Why the adapter stopped waiting on this run's provider log, from the run's own record; else ``None``.

    The record sits beside the raw adapter result the allocator result names.
    Only a regular file inside the attempt is read; anything else is unknown.
    """
    try:
        record = ((adapter.get("provider_closeout") or {}).get("adapter_result") or {}).get("path")
        root = Path(attempt_dir).resolve()
        path = Path(str(record or "")).parent / STARTUP_PROBE_MANIFEST
        if (not record or not path.is_absolute() or ".." in path.parts or path.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(root) or path.stat().st_size > _MAX_MANIFEST_BYTES):
            return None
        log = json.loads(path.read_text(encoding="utf-8")).get("container_log_result")
        reason = log.get("break_reason") if isinstance(log, Mapping) else None
        return reason if isinstance(reason, str) and reason else None
    except (OSError, ValueError, AttributeError, TypeError):
        return None


def transient_provisioning_blocker(blockers: Iterable[Any], *, log_break_reason: str | None = None) -> str | None:
    """The transient-infrastructure code a closed, unscored run's own blockers stand for, else ``None``.

    ``blockers`` are the allocator result's typed codes (the adapter's and the
    entrypoint's). A run whose entrypoint named its policy provisioning stage is
    never transient: the stage is known and the cause is not.
    """
    codes = _codes(list(blockers or []))
    observed = set(codes)
    if observed & MEDIA_TOOLCHAIN_CODES:
        return f"{TRANSIENT_PROVISIONING_PREFIX}:package_mirror"
    if POLICY_PROVISIONING_STAGE_CODE in observed:
        return None
    if any(code.startswith(BUNDLE_DOWNLOAD_PREFIX) for code in codes):
        return f"{TRANSIENT_PROVISIONING_PREFIX}:bundle_download"
    # The adapter's own evidence-bounded judgement; the same call that put the machine on the avoid list.
    from .vast_provider_adapter import _machine_avoidlist_reason

    if _machine_avoidlist_reason(codes) is not None:
        return f"{TRANSIENT_PROVISIONING_PREFIX}:rented_machine"
    if _NO_OUTPUT_CODES <= observed and log_break_reason in LOG_SILENT_BREAK_REASONS:
        return f"{TRANSIENT_PROVISIONING_PREFIX}:provider_log_silent"
    return None


def unscored_run_blockers(adapter: Mapping[str, Any], *, attempt_dir: str | Path) -> list[str] | None:
    """The scoped row blockers of a closed run that returned no usable result, else ``None``.

    ``None`` leaves the caller's own harness code: nothing typed narrows the failure.
    """
    codes = _codes(adapter.get("blockers"))
    transient = transient_provisioning_blocker(
        codes, log_break_reason=provider_log_break_reason(adapter, attempt_dir=attempt_dir))
    if transient is not None:
        return [transient]
    return [POLICY_PROVISIONING_BLOCKER] if POLICY_PROVISIONING_STAGE_CODE in codes else None


def transient_void_run(row: Mapping[str, Any]) -> bool:
    """A closed run whose every episode is void for a transient provisioning reason (the G1 lane's shape)."""
    summary = row.get("summary") or {}
    outcomes = summary.get("episode_outcomes")
    return (row.get("status") == "completed" and not summary.get("episodes_scored")
            and isinstance(outcomes, list) and bool(outcomes)
            and all(isinstance(outcome, Mapping) and outcome.get("outcome") == "void"
                    and outcome.get("reason") in TRANSIENT_VOID_REASONS for outcome in outcomes))


def transient_provisioning_failed(row: Mapping[str, Any]) -> bool:
    """A spent row that failed on transient infrastructure: blocked only by such a code, or void for one."""
    blockers = set(_codes(row.get("blockers")))
    return (row.get("status") == "blocked" and bool(blockers)
            and blockers <= TRANSIENT_PROVISIONING_BLOCKERS) or transient_void_run(row)


def candidate_specific_failure(row: Mapping[str, Any]) -> bool:
    """A blocked row whose every code names the candidate's own policy provisioning or load."""
    blockers = set(_codes(row.get("blockers")))
    return row.get("status") == "blocked" and bool(blockers) and blockers <= CANDIDATE_SPECIFIC_BLOCKERS


__all__ = [
    "CANDIDATE_SPECIFIC_BLOCKERS", "LOG_SILENT_BREAK_REASONS", "MEDIA_TOOLCHAIN_CODES", "POLICY_PROVISIONING_BLOCKER",
    "POLICY_PROVISIONING_STAGE_CODE", "TRANSIENT_PROVISIONING_BLOCKERS", "TRANSIENT_PROVISIONING_PREFIX",
    "TRANSIENT_VOID_REASONS", "candidate_specific_failure", "provider_log_break_reason",
    "transient_provisioning_blocker", "transient_provisioning_failed", "transient_void_run", "unscored_run_blockers",
]
