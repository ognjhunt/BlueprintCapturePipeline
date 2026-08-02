"""Fail-closed contracts for the Postshot Windows worker.

The paid worker is an adapter.  These helpers deliberately contain no AWS or
object-store calls so the safety policy can be replayed with a fake worker and
with collected ``worker_pulse.v2`` artifacts before any provider mutation.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


PULSE_SCHEMA_VERSION = "worker_pulse.v2"
ATTEMPT_LEDGER_SCHEMA_VERSION = "postshot_attempt_ledger.v1"
DELETION_RECEIPT_SCHEMA_VERSION = "postshot_deletion_receipt.v1"
TEARDOWN_PROOF_SCHEMA_VERSION = "postshot_worker_teardown_proof.v2"
EXTERNAL_WATCHDOG_SCHEMA_VERSION = "postshot_external_watchdog.v2"

PULSE_INTERVAL_SECONDS = 120
STALE_PULSE_TERMINATION_SECONDS = 5 * 60
NO_PROGRESS_PULSE_LIMIT = 3
STARTUP_GRACE_SECONDS = 10 * 60

PHASE_LIMITS_SECONDS: dict[str, int] = {
    "windows_boot": 10 * 60,
    "nvidia_driver": 15 * 60,
    "msi_download_install": 10 * 60,
    "dataset_license_retrieval": 15 * 60,
    "cli_activation_canary": 10 * 60,
    "tiny_training_canary": 20 * 60,
    # The worker idles here until the watcher validates the canary artifacts
    # and grants approval; P1 must not start without it (master goal phase D).
    "awaiting_canary_approval": 15 * 60,
    "P1": 150 * 60,
    "P2": 150 * 60,
    "whole_instance": 300 * 60,
}

# Worker pulses are serialized by PowerShell's ConvertTo-Json, which is not
# canonical_json.  Their digest is computed over the exact compact JSON line
# with the trailing pulse_digest member absent, flagged by this encoding tag
# (the same tag the worker template emits and validate_pulse accepts).
WORKER_PULSE_LINE_ENCODING = "sha256:json_utf8_noncanonical"
_PULSE_LINE_DIGEST_SUFFIX_RE = re.compile(r',"pulse_digest":"sha256:([0-9a-f]{64})"\}\s*$')


def verify_worker_pulse_line(line: str) -> list[str]:
    """Integrity-check one worker-emitted pulse line at the raw-byte level.

    The worker appends ``pulse_digest`` as the final member of the compact
    JSON object, where the digest is sha256 over the exact line bytes without
    that member.  Removing the suffix therefore reproduces the digest input
    without needing to reproduce PowerShell's serialization.
    """

    text = line.strip()
    if not text:
        return ["pulse_line_empty"]
    match = _PULSE_LINE_DIGEST_SUFFIX_RE.search(text)
    if match is None:
        return ["pulse_line_digest_member_missing"]
    expected = match.group(1)
    payload = text[: match.start()] + "}"
    observed = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if observed != expected:
        return ["pulse_line_digest_mismatch"]
    return []


def derive_phase_started_epoch(
    pulses: Sequence[Mapping[str, Any]], *, launched_epoch: float
) -> float:
    """Anchor phase timeouts at the current phase's start, not at launch.

    Prefers the worker-reported ``phase_started_at_utc``; otherwise walks the
    contiguous run of latest-phase pulses.  Falls back to launch time so a
    missing field can only make the timeout stricter, never looser.
    """

    if not pulses:
        return launched_epoch
    latest = pulses[-1]
    reported = parse_timestamp(latest.get("phase_started_at_utc"))
    phase = str(latest.get("phase", ""))
    earliest_observed: float | None = None
    for pulse in reversed(pulses):
        if str(pulse.get("phase", "")) != phase:
            break
        observed = parse_timestamp(pulse.get("observed_at_utc"))
        if observed is not None:
            earliest_observed = observed
    candidates = [value for value in (reported, earliest_observed) if value is not None]
    if not candidates:
        return launched_epoch
    anchor = min(candidates)
    return max(anchor, launched_epoch) if anchor < launched_epoch else anchor

TRAINING_PHASES = {"tiny_training_canary", "P1", "P2"}
POSTSHOT_PROCESS_REQUIRED_PHASES = TRAINING_PHASES

_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
_BEARER_RE = re.compile(r"(?i)(\bBearer\s+)[^\s,;]+")
_KEY_VALUE_SECRET_RE = re.compile(
    r"(?i)(\b(?:password|passwd|secret|token|authorization|api[_-]?key|access[_-]?key)\b\s*[:=]\s*)[^\s,;]+"
)
_AWS_ACCESS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def utc_now_iso(epoch: float | None = None) -> str:
    """Return a stable UTC timestamp suitable for an evidence artifact."""

    value = datetime.fromtimestamp(epoch, tz=timezone.utc) if epoch is not None else datetime.now(timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def parse_timestamp(value: Any) -> float | None:
    """Parse an ISO timestamp without accepting local-time ambiguity."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.timestamp()


def sanitize_text(value: Any, secrets: Iterable[str] = ()) -> str:
    """Redact credentials, bearer values, access keys, and all URLs.

    Logs and provider metadata are hostile input.  URL removal is intentional:
    a presigned URL is a credential even when its path is otherwise harmless.
    """

    text = _CONTROL_RE.sub("", str(value))
    for secret in sorted({str(item) for item in secrets if str(item)}, key=len, reverse=True):
        text = text.replace(secret, "[REDACTED_SECRET]")
    text = _URL_RE.sub("[REDACTED_URL]", text)
    text = _BEARER_RE.sub(r"\1[REDACTED_SECRET]", text)
    text = _KEY_VALUE_SECRET_RE.sub(r"\1[REDACTED_SECRET]", text)
    return _AWS_ACCESS_KEY_RE.sub("[REDACTED_ACCESS_KEY]", text)


def sanitize_path(value: Any, secrets: Iterable[str] = ()) -> str:
    """Keep only a benign basename for a path supplied by a provider/worker."""

    basename = Path(str(value)).name.replace("\\", "_")
    return _SAFE_NAME_RE.sub("_", sanitize_text(basename, secrets)).strip(".") or "unknown"


def assert_secret_free(value: Any, secrets: Iterable[str] = ()) -> None:
    """Raise if a serializable artifact still contains a supplied secret/URL."""

    rendered = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    if _URL_RE.search(rendered):
        raise ValueError("artifact_contains_url")
    for secret in secrets:
        if str(secret) and str(secret) in rendered:
            raise ValueError("artifact_contains_secret")
    if _AWS_ACCESS_KEY_RE.search(rendered):
        raise ValueError("artifact_contains_access_key")


def redact_command(arguments: Sequence[Any], secrets: Iterable[str] = ()) -> list[str]:
    """Redact command arguments without changing argument boundaries."""

    return [sanitize_text(argument, secrets) for argument in arguments]


def build_postshot_train_args(
    *,
    login_email: str,
    login_password: str,
    dataset: str,
    profile: str,
    output_project: str,
    output_splat: str,
    max_image_size: int = 0,
    train_steps_limit_ksteps: int | None = None,
    max_num_splats_ksplats: int | None = None,
) -> list[str]:
    """Build the corrected Postshot invocation.

    Postshot's login flags are global options.  Keeping this as a pure helper
    prevents a later PowerShell edit from moving them after ``train`` again.
    ``max_image_size=0`` means full resolution (the frozen P1/P2 setting); the
    canary passes bounded values instead.  Steps and splats are in kilo-units,
    matching ``-s,--train-steps-limit`` and ``--max-num-splats``.
    """

    arguments = [
        "--login",
        login_email,
        "--password",
        login_password,
        "train",
        "--import",
        dataset,
        "--profile",
        profile,
        "--no-recenter-points",
        "--max-image-size",
        str(int(max_image_size)),
    ]
    if train_steps_limit_ksteps is not None:
        arguments += ["--train-steps-limit", str(int(train_steps_limit_ksteps))]
    if max_num_splats_ksplats is not None:
        arguments += ["--max-num-splats", str(int(max_num_splats_ksplats))]
    arguments += [
        "--output",
        output_project,
        "--export-splat",
        output_splat,
    ]
    return arguments


# Frozen tiny-canary bounds (phase D): candidate-only images, minimum supported
# step limit (1 kStep), bounded image size and splat count, PSHT + PLY outputs.
TINY_CANARY_SPEC: dict[str, Any] = {
    "arm_id": "C0_canary_splat3",
    "profile": "Splat3",
    "max_image_size": 256,
    "train_steps_limit_ksteps": 1,
    "max_num_splats_ksplats": 100,
    "hidden_images_included": False,
    "phase": "tiny_training_canary",
    "required_outputs": ["C0_canary_splat3.psht", "C0_canary_splat3.ply"],
    "required_consecutive_credible_pulses": 3,
}


def decode_worker_text(raw: bytes) -> str:
    """Decode worker-produced text that may be UTF-16 (PowerShell ``*>``)."""

    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def evaluate_canary_gate(
    *,
    pulses: Sequence[Mapping[str, Any]],
    canary_outputs: Mapping[str, int],
    secrets_found: bool,
    required_consecutive: int = 3,
) -> dict[str, Any]:
    """Decide whether the same-instance tiny canary earned P1/P2 approval.

    Requires ``required_consecutive`` consecutive valid pulses observed since
    the canary phase began (canary + approval-wait phases), at least one pulse
    inside the canary phase showing a live Postshot PID with GPU or log/output
    evidence, both bounded outputs present and non-empty, and zero secret
    leaks anywhere in the pulse series.
    """

    reasons: list[str] = []
    canary_phases = {"tiny_training_canary", "awaiting_canary_approval"}
    window = [p for p in pulses if str(p.get("phase", "")) in canary_phases]
    consecutive = 0
    best_streak = 0
    for pulse in window:
        progress = pulse.get("progress", {}) if isinstance(pulse.get("progress"), Mapping) else {}
        credible = bool(
            progress.get("credible_training_progress")
            or progress.get("log_progress")
            or progress.get("output_progress")
            or (progress.get("postshot_process_alive") and progress.get("gpu_active"))
        )
        consecutive = consecutive + 1 if credible else 0
        best_streak = max(best_streak, consecutive)
    if best_streak < required_consecutive:
        reasons.append(
            f"insufficient_consecutive_credible_pulses:{best_streak}<{required_consecutive}"
        )
    trained = [
        p
        for p in window
        if str(p.get("phase")) == "tiny_training_canary"
        and isinstance(p.get("progress"), Mapping)
        and p["progress"].get("postshot_process_alive")
        and (
            p["progress"].get("gpu_active")
            or p["progress"].get("log_progress")
            or p["progress"].get("output_progress")
        )
    ]
    if not trained:
        reasons.append("no_pulse_shows_live_postshot_process_with_gpu_or_growth")
    for name in ("psht", "ply"):
        if int(canary_outputs.get(name, 0) or 0) <= 0:
            reasons.append(f"canary_output_missing_or_empty:{name}")
    if secrets_found:
        reasons.append("secret_leak_in_pulse_series")
    return {
        "schema_version": "postshot_canary_gate.v1",
        "passed": not reasons,
        "reasons": sorted(reasons),
        "credible_pulse_streak": best_streak,
        "required_consecutive": required_consecutive,
        "canary_pulse_count": len(window),
    }


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def parse_nvidia_smi_csv(raw: str) -> dict[str, Any]:
    """Parse a query-gpu CSV row while treating malformed output as unknown."""

    result: dict[str, Any] = {
        "name": None,
        "driver_version": None,
        "utilization_percent": None,
        "memory_used_mib": None,
        "memory_total_mib": None,
        "temperature_c": None,
        "power_w": None,
        "parse_error": None,
    }
    try:
        rows = list(csv.reader(io.StringIO(raw.strip())))
        if not rows or not rows[0]:
            result["parse_error"] = "empty_output"
            return result
        row = [item.strip() for item in rows[0]]
        if len(row) < 7:
            result["parse_error"] = "field_count_too_small"
            return result

        def numeric(item: str) -> float | None:
            match = re.search(r"-?\d+(?:\.\d+)?", item)
            return float(match.group(0)) if match else None

        result.update(
            {
                "name": sanitize_text(row[0]),
                "driver_version": sanitize_text(row[1]),
                "utilization_percent": numeric(row[2]),
                "memory_used_mib": numeric(row[3]),
                "memory_total_mib": numeric(row[4]),
                "temperature_c": numeric(row[5]),
                "power_w": numeric(row[6]),
            }
        )
        return result
    except (csv.Error, TypeError, ValueError) as exc:
        result["parse_error"] = type(exc).__name__
        return result


def output_growth(previous: Mapping[str, Any] | None, current: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Annotate output sizes with growth relative to the prior pulse."""

    old_sizes = {
        str(item.get("path")): int(item.get("bytes", 0) or 0)
        for item in (previous or {}).get("outputs", [])
        if isinstance(item, Mapping)
    }
    result: list[dict[str, Any]] = []
    for item in current:
        path = sanitize_path(item.get("path", "unknown"))
        size = max(0, int(item.get("bytes", 0) or 0))
        result.append(
            {
                "path": path,
                "bytes": size,
                "growth_bytes": size - old_sizes.get(path, 0),
                "digest": item.get("digest") if isinstance(item.get("digest"), str) else None,
            }
        )
    return result


def _phase_progress(previous: Mapping[str, Any] | None, phase: str) -> bool:
    return bool(previous and str(previous.get("phase", "")) != phase)


def _gpu_active(gpu: Mapping[str, Any]) -> bool:
    util = gpu.get("utilization_percent")
    memory = gpu.get("memory_used_mib")
    try:
        return float(util or 0) > 0 or float(memory or 0) > 0
    except (TypeError, ValueError):
        return False


def _has_growth(items: Sequence[Mapping[str, Any]]) -> bool:
    return any(int(item.get("growth_bytes", 0) or 0) > 0 for item in items)


def build_worker_pulse(
    *,
    run_id: str,
    attempt: int,
    arm: str,
    phase: str,
    sequence: int,
    observed_at: str,
    instance: Mapping[str, Any],
    postshot_process: Mapping[str, Any],
    postshot_log: Mapping[str, Any],
    gpu: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]],
    disk_free_bytes: int | None,
    last_credible_progress_at: str | None,
    live_cost_estimate_usd: float,
    incremental_cap_usd: float,
    ttl_deadline: str,
    next_automatic_kill_deadline: str,
    result_upload_state: str,
    credential_object_deletion_state: str,
    startup_grace_until: str | None,
    previous_pulse: Mapping[str, Any] | None = None,
    secrets: Iterable[str] = (),
) -> dict[str, Any]:
    """Create the curated, hashable pulse used by both watchdogs."""

    safe_log_tail = sanitize_text(postshot_log.get("tail", ""), secrets)[-4000:]
    safe_gpu = {
        "name": sanitize_text(gpu.get("name", ""), secrets) or None,
        "driver_version": sanitize_text(gpu.get("driver_version", ""), secrets) or None,
        "utilization_percent": gpu.get("utilization_percent"),
        "memory_used_mib": gpu.get("memory_used_mib"),
        "memory_total_mib": gpu.get("memory_total_mib"),
        "temperature_c": gpu.get("temperature_c"),
        "power_w": gpu.get("power_w"),
    }
    safe_outputs = output_growth(previous_pulse, outputs)
    log_bytes = max(0, int(postshot_log.get("byte_count", 0) or 0))
    previous_log_bytes = max(0, int((previous_pulse or {}).get("postshot_log", {}).get("byte_count", 0) or 0))
    log_growth = log_bytes - previous_log_bytes
    phase_advanced = _phase_progress(previous_pulse, phase)
    in_grace = bool(parse_timestamp(startup_grace_until) and parse_timestamp(observed_at) is not None and parse_timestamp(observed_at) < parse_timestamp(startup_grace_until))
    log_progress = log_growth > 0
    output_progress = _has_growth(safe_outputs)
    gpu_active = _gpu_active(safe_gpu)
    credible = not in_grace and (log_progress or output_progress or phase_advanced or (bool(postshot_process.get("alive")) and gpu_active))
    pulse: dict[str, Any] = {
        "schema_version": PULSE_SCHEMA_VERSION,
        "run_id": sanitize_text(run_id, secrets),
        "attempt": int(attempt),
        "arm": sanitize_text(arm, secrets),
        "phase": sanitize_text(phase, secrets),
        "phase_started_at_utc": (previous_pulse or {}).get("phase_started_at_utc") if previous_pulse and previous_pulse.get("phase") == phase else observed_at,
        "sequence": int(sequence),
        "observed_at_utc": observed_at,
        "instance": {
            "id": sanitize_text(instance.get("id", ""), secrets),
            "type": sanitize_text(instance.get("type", ""), secrets),
            "state": sanitize_text(instance.get("state", ""), secrets),
        },
        "postshot_process": {
            "pid": postshot_process.get("pid"),
            "start_time_utc": postshot_process.get("start_time_utc"),
            "alive": bool(postshot_process.get("alive")),
            "exit_code": postshot_process.get("exit_code"),
        },
        "postshot_log": {
            "tail": safe_log_tail,
            "byte_count": log_bytes,
            "digest": postshot_log.get("digest") if isinstance(postshot_log.get("digest"), str) else None,
            "growth_bytes": log_growth,
        },
        "gpu": safe_gpu,
        "outputs": safe_outputs,
        "disk_free_bytes": disk_free_bytes,
        "last_credible_progress_at_utc": last_credible_progress_at,
        "live_cost_estimate_usd": round(float(live_cost_estimate_usd), 6),
        "incremental_cap_usd": round(float(incremental_cap_usd), 6),
        "ttl_deadline_utc": ttl_deadline,
        "next_automatic_kill_deadline_utc": next_automatic_kill_deadline,
        "startup_grace_until_utc": startup_grace_until,
        "result_upload_state": sanitize_text(result_upload_state, secrets),
        "credential_object_deletion_state": sanitize_text(credential_object_deletion_state, secrets),
        "progress": {
            "os_alive": instance.get("state") == "running",
            "postshot_process_alive": bool(postshot_process.get("alive")),
            "gpu_active": gpu_active,
            "log_progress": log_progress,
            "output_progress": output_progress,
            "phase_progress": phase_advanced,
            "startup_grace_active": in_grace,
            "credible_training_progress": credible,
        },
    }
    pulse["pulse_digest_encoding"] = "canonical_json_v1"
    pulse["pulse_digest"] = canonical_digest(pulse, digest_field="pulse_digest")
    assert_secret_free(pulse, secrets)
    return pulse


def validate_pulse(pulse: Mapping[str, Any], previous: Mapping[str, Any] | None = None) -> list[str]:
    """Return stable validation errors for a pulse received by the watcher."""

    errors: list[str] = []
    if pulse.get("schema_version") != PULSE_SCHEMA_VERSION:
        errors.append("schema_version")
    if not str(pulse.get("run_id", "")):
        errors.append("run_id")
    try:
        sequence = int(pulse.get("sequence", -1))
        previous_sequence = int(previous.get("sequence", -1)) if previous is not None else None
    except (TypeError, ValueError):
        sequence = -1
        previous_sequence = None
        errors.append("sequence")
    if sequence < 0:
        errors.append("sequence")
    if previous_sequence is not None and sequence <= previous_sequence:
        errors.append("sequence_not_monotonic")
    digest_encoding = pulse.get("pulse_digest_encoding", "canonical_json_v1")
    if digest_encoding == "canonical_json_v1":
        expected = canonical_digest(pulse, digest_field="pulse_digest")
        if pulse.get("pulse_digest") != expected:
            errors.append("pulse_digest")
    elif digest_encoding == "sha256:json_utf8_noncanonical":
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(pulse.get("pulse_digest", ""))):
            errors.append("pulse_digest")
    else:
        errors.append("pulse_digest_encoding")
    for field in ("observed_at_utc", "ttl_deadline_utc", "next_automatic_kill_deadline_utc"):
        if parse_timestamp(pulse.get(field)) is None:
            errors.append(field)
    progress = pulse.get("progress")
    if not isinstance(progress, Mapping):
        errors.append("progress")
    return sorted(set(errors))


@dataclass(frozen=True)
class WatchDecision:
    action: str
    reason: str
    credible_progress: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "credible_progress": self.credible_progress,
        }


def evaluate_pulses(
    pulses: Sequence[Mapping[str, Any]],
    *,
    now_epoch: float,
    phase_started_epoch: float,
    launched_epoch: float,
    live_cost_estimate_usd: float,
    incremental_cap_usd: float,
) -> WatchDecision:
    """Apply the fail-closed paid-run policy to a collected pulse series."""

    if not pulses:
        if now_epoch - launched_epoch > STALE_PULSE_TERMINATION_SECONDS:
            return WatchDecision("terminate", "pulse_stale_gt_300s")
        return WatchDecision("continue", "startup_pulse_grace")
    latest = pulses[-1]
    observed_epoch = parse_timestamp(latest.get("observed_at_utc"))
    if observed_epoch is None or now_epoch - observed_epoch > STALE_PULSE_TERMINATION_SECONDS:
        return WatchDecision("terminate", "pulse_stale_gt_300s")
    if live_cost_estimate_usd >= incremental_cap_usd:
        return WatchDecision("terminate", "incremental_spend_cap_reached")
    ttl_epoch = parse_timestamp(latest.get("ttl_deadline_utc"))
    if ttl_epoch is not None and now_epoch >= ttl_epoch:
        return WatchDecision("terminate", "whole_instance_ttl_expired")
    phase = str(latest.get("phase", ""))
    phase_limit = PHASE_LIMITS_SECONDS.get(phase)
    pulse_phase_started = parse_timestamp(latest.get("phase_started_at_utc"))
    effective_phase_started = pulse_phase_started if pulse_phase_started is not None else phase_started_epoch
    if phase_limit is not None and now_epoch >= effective_phase_started + phase_limit:
        return WatchDecision("abort", f"phase_timeout:{phase}")
    process_alive = bool(latest.get("progress", {}).get("postshot_process_alive"))
    if phase in POSTSHOT_PROCESS_REQUIRED_PHASES and not process_alive:
        return WatchDecision("abort", "postshot_process_dead", False)
    grace_until = parse_timestamp(latest.get("startup_grace_until_utc"))
    in_grace = grace_until is not None and now_epoch < grace_until
    credible = bool(latest.get("progress", {}).get("credible_training_progress"))
    no_progress_count = 0
    for pulse in reversed(pulses):
        if bool(pulse.get("progress", {}).get("startup_grace_active")):
            break
        if bool(pulse.get("progress", {}).get("credible_training_progress")):
            break
        no_progress_count += 1
    if not in_grace and no_progress_count >= NO_PROGRESS_PULSE_LIMIT:
        return WatchDecision("abort", "no_credible_progress_3_pulses", False)
    return WatchDecision("continue", "credible_progress" if credible else "startup_or_grace", credible)


def evaluate_missing_pulse(*, last_pulse_epoch: float | None, now_epoch: float) -> WatchDecision:
    if last_pulse_epoch is None or now_epoch - last_pulse_epoch > STALE_PULSE_TERMINATION_SECONDS:
        return WatchDecision("terminate", "pulse_stale_gt_300s")
    return WatchDecision("continue", "pulse_fresh")


def build_live_cost_estimate(
    *,
    as_of_utc: str,
    instance_usd: float,
    ebs_usd: float,
    transfer_usd: float,
    object_storage_usd: float,
    license_increment_usd: float,
) -> dict[str, Any]:
    """Keep an upper-bound live estimate distinct from delayed billing."""

    parts = {
        "instance_usd": float(instance_usd),
        "ebs_usd": float(ebs_usd),
        "transfer_usd": float(transfer_usd),
        "object_storage_usd": float(object_storage_usd),
        "license_increment_usd": float(license_increment_usd),
    }
    total = sum(parts.values())
    return {
        "schema_version": "postshot_live_cost_estimate.v1",
        "kind": "live_estimate",
        "as_of_utc": as_of_utc,
        "reconciled": False,
        **{key: round(value, 6) for key, value in parts.items()},
        "total_usd": round(total, 6),
    }


def build_reconciled_cost(*, source: str, reconciled_at_utc: str, total_usd: float, line_items: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "postshot_reconciled_cost.v1",
        "kind": "reconciled_billing",
        "source": sanitize_text(source),
        "reconciled_at_utc": reconciled_at_utc,
        "reconciled": True,
        "total_usd": round(float(total_usd), 6),
        "line_items": {sanitize_text(k): value for k, value in line_items.items()},
    }


def build_provider_zero_proof(
    *,
    run_id: str,
    region: str,
    instances: Sequence[Mapping[str, Any]],
    volumes: Sequence[Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Any]],
    images: Sequence[Mapping[str, Any]],
    elastic_ips: Sequence[Mapping[str, Any]],
    security_groups: Sequence[Mapping[str, Any]] = (),
    checked_at_utc: str | None = None,
) -> dict[str, Any]:
    """Calculate provider-zero only from exact run-owned inventory."""

    blockers: list[str] = []
    active_states = {"pending", "running", "stopping", "stopped", "shutting-down"}
    active_instances = [x for x in instances if str(x.get("state", "")) in active_states]
    attached_volumes = [x for x in volumes if x.get("state") not in {"deleted", "missing"}]
    if active_instances:
        blockers.append("run_owned_instance_not_terminated")
    if attached_volumes:
        blockers.append("run_owned_volume_present")
    if snapshots:
        blockers.append("run_owned_snapshot_present")
    if images:
        blockers.append("run_owned_ami_present")
    if elastic_ips:
        blockers.append("run_owned_elastic_ip_present")
    return {
        "schema_version": TEARDOWN_PROOF_SCHEMA_VERSION,
        "run_id": sanitize_text(run_id),
        "region": sanitize_text(region),
        "checked_at_utc": checked_at_utc or utc_now_iso(),
        "inventory_scope": "exact_run_tag_account_region",
        "provider_zero": not blockers,
        "blockers": blockers,
        "run_owned_instances": list(instances),
        "run_owned_volumes": list(volumes),
        "run_owned_snapshots": list(snapshots),
        "run_owned_images": list(images),
        "run_owned_elastic_ips": list(elastic_ips),
        "non_billable_security_groups": list(security_groups),
        "security_group_is_not_provider_zero_evidence": True,
    }


def build_deletion_receipt(
    *,
    run_id: str,
    checked_at_utc: str,
    objects: Sequence[Mapping[str, Any]],
    secrets: Iterable[str] = (),
) -> dict[str, Any]:
    """Return an object-level deletion/absence receipt without URL values."""

    safe_objects = []
    for item in objects:
        key = sanitize_text(item.get("key", ""), secrets)
        safe_objects.append(
            {
                "key": key,
                "delete_requested": bool(item.get("delete_requested")),
                "absent_verified": bool(item.get("absent_verified")),
                "object_kind": sanitize_text(item.get("object_kind", "unknown"), secrets),
            }
        )
    receipt = {
        "schema_version": DELETION_RECEIPT_SCHEMA_VERSION,
        "run_id": sanitize_text(run_id, secrets),
        "checked_at_utc": checked_at_utc,
        "objects": safe_objects,
        "all_absent_verified": bool(safe_objects) and all(item["absent_verified"] for item in safe_objects),
        "raw_secret_values_recorded": False,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    assert_secret_free(receipt, secrets)
    return receipt


def build_external_watchdog_record(
    *,
    run_id: str,
    instance_id: str,
    pid: int,
    started_at_utc: str,
    ttl_deadline_utc: str,
    log_path: str,
    command_digest: str,
) -> dict[str, Any]:
    record = {
        "schema_version": EXTERNAL_WATCHDOG_SCHEMA_VERSION,
        "run_id": sanitize_text(run_id),
        "instance_id": sanitize_text(instance_id),
        "pid": int(pid),
        "started_at_utc": started_at_utc,
        "ttl_deadline_utc": ttl_deadline_utc,
        "log_path": sanitize_path(log_path),
        "command_digest": command_digest,
        "status": "armed",
        "independent_process": True,
        "raw_secret_values_recorded": False,
    }
    record["record_digest"] = canonical_digest(record, digest_field="record_digest")
    return record


def build_attempt_ledger(
    *,
    attempts: Sequence[Mapping[str, Any]],
    historical_bakeoff_budget_usd: float,
    historical_postshot_spend_estimate_usd: float,
    generated_at_utc: str,
) -> dict[str, Any]:
    """Build the append-only historical ledger without filling missing facts."""

    ledger = {
        "schema_version": ATTEMPT_LEDGER_SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc,
        "append_only": True,
        "historical_bakeoff_budget_usd": float(historical_bakeoff_budget_usd),
        "historical_postshot_spend_estimate_usd": float(historical_postshot_spend_estimate_usd),
        "historical_spend_reconciliation_status": "required_not_observed",
        "attempts": [dict(item) for item in attempts],
        "claim_boundary": "attempt evidence is not reconstruction success; accepted output requires independent evaluation",
    }
    ledger["ledger_digest"] = canonical_digest(ledger, digest_field="ledger_digest")
    assert_secret_free(ledger)
    return ledger
