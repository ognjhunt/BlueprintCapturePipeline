"""Operator alerting for live pipeline control-plane manifests.

The live control plane intentionally exits zero when a pass is externally
blocked so timers keep running. This module is the separate operator-signal
surface: it reads the latest manifest, sends a bounded webhook notification
when the pass is blocked, and can fail closed when production alerting is not
configured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from math import isfinite
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
from urllib.parse import urlsplit

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .task_evaluation_release_identity import running_release_commit


LIVE_PIPELINE_MANIFEST_ALERT_SCHEMA_VERSION = "blueprint_live_pipeline_manifest_alert.v1"
OPERATOR_ALERT_WEBHOOK_URL_ENV = "BLUEPRINT_OPERATOR_ALERT_WEBHOOK_URL"
OPERATOR_ALERT_REQUIRE_WEBHOOK_ENV = "BLUEPRINT_OPERATOR_ALERT_REQUIRE_WEBHOOK"
DEFAULT_MANIFEST_PATH = (
    "/var/lib/blueprint/pipeline-control-plane/live_pipeline_control_plane_manifest.json"
)
SPEND_ADMISSION_LOCK_SCHEMA_VERSION = "blueprint.paid_spend_admission_lock.v1"
# The control-plane timer runs every five minutes. An unchanged blocked pass is
# re-sent hourly, as capacity pages are; a changed status or blocker set is sent
# at once.
ALERT_REPEAT_SECONDS = 60 * 60


def _string(value: Any) -> str:
    return str(value or "").strip()


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any, *, limit: int = 12) -> list[str]:
    if not isinstance(value, list) or limit <= 0:
        return []
    result: list[str] = []
    for item in value:
        text = _string(item)
        if text:
            result.append(text[:200])
        if len(result) >= limit:
            break
    return result


def _env_truthy(name: str) -> bool:
    return _string(os.getenv(name)).lower() in {"1", "true", "yes", "on"}


def _read_manifest(path: Path) -> Dict[str, Any]:
    payload = read_json_any(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected control-plane manifest JSON object at {path}")
    return dict(payload)


def _all_manifest_blockers(manifest: Mapping[str, Any]) -> list[str]:
    # Live runs project setup blockers at the top level rather than embedding
    # the setup manifest. Preserve those reasons in the operator notification.
    setup = _mapping(manifest.get("setup"))
    external_input_packet = _mapping(manifest.get("external_input_packet"))
    blockers: list[str] = []
    for value in (manifest.get("blockers"), manifest.get("setup_blockers"),
                  setup.get("blockers"), external_input_packet.get("blockers")):
        if isinstance(value, list):
            blockers.extend(_string_list(value, limit=len(value)))
    return blockers


def _manifest_blockers(manifest: Mapping[str, Any]) -> list[str]:
    return _all_manifest_blockers(manifest)[:12]


def _alert_required(manifest: Mapping[str, Any]) -> bool:
    if _mapping(manifest.get("page_event")).get("required") is True:
        return True
    status = _string(manifest.get("status")).lower()
    if "blocked" in status:
        return True
    return bool(_manifest_blockers(manifest))


def _message_text(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    blockers: Sequence[str],
    blocker_count: int,
    fingerprint: str,
    source_version: str,
) -> str:
    status = _string(manifest.get("status")) or "unknown"
    job_id = _string(manifest.get("job_id"))
    capture_root = _string(manifest.get("capture_root"))
    page_event = _mapping(manifest.get("page_event"))
    blocker_text = (
        ", ".join(blockers)
        if blockers
        else (
            "threshold crossing requires operator notification"
            if page_event.get("required") is True
            else "status contains blocked"
        )
    )
    if manifest.get("schema_version") == SPEND_ADMISSION_LOCK_SCHEMA_VERSION:
        effective_spend = manifest.get("effective_spend_usd")
        hard_stop = manifest.get("hard_stop_usd")
        if status == "override_open":
            headline = (
                "Blueprint paid spend override is active after a hard-stop crossing: "
                f"status={status}, effective_spend_usd={effective_spend}, "
                f"hard_stop_usd={hard_stop}."
            )
        else:
            headline = (
                "Blueprint paid spend admission is locked: "
                f"status={status}, effective_spend_usd={effective_spend}, "
                f"hard_stop_usd={hard_stop}."
            )
    else:
        headline = f"Blueprint live pipeline control plane is blocked: status={status}."
    parts = [headline, f"manifest={manifest_path}", f"source_version={source_version}",
             f"fingerprint={fingerprint}", f"blocker_count={blocker_count}"]
    if job_id:
        parts.append(f"job_id={job_id}")
    if capture_root:
        parts.append(f"capture_root={capture_root}")
    parts.append(f"blockers={blocker_text}")
    # Slack's readable message may be bounded; structured metadata and the
    # retained audit below keep the complete blocker list and evidence identity.
    return " ".join(parts)[:40000]


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def _validated_webhook_url(value: str) -> str:
    url = _string(value)
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError("operator webhook URL is malformed") from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or port not in {None, 443}
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise RuntimeError("operator webhook URL must use a credential-free HTTPS origin")
    return url


def _post_webhook(url: str, payload: Mapping[str, Any], *, timeout_seconds: float) -> None:
    if not isfinite(timeout_seconds) or not 0.1 <= timeout_seconds <= 30.0:
        raise RuntimeError("operator webhook timeout must be between 0.1 and 30 seconds")
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        _validated_webhook_url(url),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(_RejectRedirects)
    with opener.open(request, timeout=timeout_seconds) as response:
        status = int(getattr(response, "status", 0) or 0)
        if status < 200 or status >= 300:
            raise RuntimeError(f"webhook returned HTTP {status}")


def _alert_fingerprint(manifest: Mapping[str, Any], blockers: Sequence[str], source_version: str) -> str:
    identity = {
        "schema_version": _string(manifest.get("schema_version")),
        "status": _string(manifest.get("status")),
        "job_id": _string(manifest.get("job_id")),
        "capture_root": _string(manifest.get("capture_root")),
        "blockers": list(blockers),
        "source_version": source_version,
    }
    return "sha256:" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


def _last_sent_epoch(output_path: Path, fingerprint: str) -> float | None:
    """When this exact alert was last delivered, from the previous pass's audit."""

    try:
        previous = read_json_any(output_path)
    except (OSError, ValueError):
        return None
    if not isinstance(previous, Mapping) or previous.get("alert_fingerprint") != fingerprint:
        return None
    sent = previous.get("last_sent_at_epoch")
    return float(sent) if isinstance(sent, (int, float)) and not isinstance(sent, bool) else None


def build_live_pipeline_manifest_alert(
    *,
    manifest_path: Path,
    output_path: Path | None = None,
    webhook_url: str | None = None,
    require_webhook: bool | None = None,
    dry_run: bool = False,
    timeout_seconds: float = 10.0,
    now: float | None = None,
) -> Dict[str, Any]:
    observed = time.time() if now is None else float(now)
    resolved_manifest_path = Path(manifest_path).expanduser().resolve()
    resolved_output_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else resolved_manifest_path.parent / "live_pipeline_manifest_alert.json"
    )
    manifest = _read_manifest(resolved_manifest_path)
    all_blockers = _all_manifest_blockers(manifest)
    blockers = all_blockers
    alert_required = _alert_required(manifest)
    module_path = Path(__file__).resolve()
    try:
        source_hash = "sha256:" + hashlib.sha256(module_path.read_bytes()).hexdigest()
    except OSError:
        source_hash = "unknown"
    source_version = running_release_commit(module_path) or source_hash
    fingerprint = _alert_fingerprint(manifest, all_blockers, source_version)
    # Spend-lock and threshold-crossing pages keep paging on every pass.
    repeat_suppressible = (
        manifest.get("schema_version") != SPEND_ADMISSION_LOCK_SCHEMA_VERSION
        and _mapping(manifest.get("page_event")).get("required") is not True
    )
    last_sent = _last_sent_epoch(resolved_output_path, fingerprint) if repeat_suppressible else None
    resolved_webhook_url = _string(webhook_url or os.getenv(OPERATOR_ALERT_WEBHOOK_URL_ENV))
    webhook_required = (
        bool(require_webhook)
        if require_webhook is not None
        else _env_truthy(OPERATOR_ALERT_REQUIRE_WEBHOOK_ENV)
    )
    message_text = _message_text(
        manifest_path=resolved_manifest_path,
        manifest=manifest,
        blockers=blockers,
        blocker_count=len(all_blockers),
        fingerprint=fingerprint,
        source_version=source_version,
    )
    report = {
        "workflow": "paid_spend_admission" if manifest.get("schema_version") == SPEND_ADMISSION_LOCK_SCHEMA_VERSION
        else "live_pipeline_control_plane",
        "run_id": _string(manifest.get("job_id")) or str(resolved_manifest_path),
        "fingerprint": fingerprint, "source_version": source_version,
        "severity": "critical" if manifest.get("schema_version") == SPEND_ADMISSION_LOCK_SCHEMA_VERSION else "warning",
        "error": _string(manifest.get("status")) or "unknown", "blockers": all_blockers,
        "evidence_ref": str(resolved_manifest_path),
    }
    webhook_payload = {"text": message_text, "metadata": {
        "event_type": "blueprint_ops_report", "event_payload": report,
    }}

    notification_status = "not_required"
    notification_error = ""
    attempted = False
    if (alert_required and resolved_webhook_url and not dry_run and last_sent is not None
            and 0 <= observed - last_sent < ALERT_REPEAT_SECONDS):
        notification_status = "suppressed_unchanged"
    elif alert_required and resolved_webhook_url and not dry_run:
        attempted = True
        try:
            _post_webhook(
                resolved_webhook_url,
                webhook_payload,
                timeout_seconds=timeout_seconds,
            )
            notification_status = "sent"
            last_sent = observed
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            notification_status = "failed"
            notification_error = f"{type(exc).__name__}: {exc}"[:500]
    elif alert_required and resolved_webhook_url and dry_run:
        notification_status = "dry_run"
    elif alert_required and not resolved_webhook_url:
        notification_status = (
            "blocked_missing_required_webhook"
            if webhook_required
            else "skipped_webhook_not_configured"
        )

    audit = {
        "schema_version": LIVE_PIPELINE_MANIFEST_ALERT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "manifest_path": str(resolved_manifest_path),
        "manifest_status": _string(manifest.get("status")) or "unknown",
        "alert_required": alert_required,
        "blockers": blockers,
        "blocker_count": len(all_blockers),
        "alert_fingerprint": fingerprint,
        "source_version": source_version,
        "imported_notifier_path": str(module_path),
        "notifier_source_sha256": source_hash,
        "webhook_payload": webhook_payload,
        "last_sent_at_epoch": last_sent,
        "webhook_configured": bool(resolved_webhook_url),
        "webhook_required": webhook_required,
        "notification_attempted": attempted,
        "notification_status": notification_status,
        "notification_error": notification_error,
        "message_text": message_text,
        "proof_boundary": {
            "alert_reads_manifest_only": True,
            "alert_performs_pipeline_work": False,
            "alert_includes_webhook_secret": False,
        },
    }
    ensure_dir(resolved_output_path.parent)
    audit["output_path"] = str(resolved_output_path)
    write_json(resolved_output_path, audit)
    return audit


def _exit_code(audit: Mapping[str, Any]) -> int:
    if not audit.get("alert_required"):
        return 0
    status = _string(audit.get("notification_status"))
    if status in {"sent", "dry_run", "suppressed_unchanged"}:
        return 0
    return 2 if status == "blocked_missing_required_webhook" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send operator alert for blocked live control-plane manifests.")
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-path")
    parser.add_argument("--webhook-url")
    parser.add_argument("--require-webhook", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)

    audit = build_live_pipeline_manifest_alert(
        manifest_path=Path(args.manifest_path),
        output_path=Path(args.output_path) if args.output_path else None,
        webhook_url=args.webhook_url,
        require_webhook=True if args.require_webhook else None,
        dry_run=args.dry_run,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"[live-pipeline-manifest-alert] audit={audit['output_path']}")
    print(f"[live-pipeline-manifest-alert] status={audit['notification_status']}")
    print(f"[live-pipeline-manifest-alert] alert_required={audit['alert_required']}")
    if audit["notification_error"]:
        print(f"[live-pipeline-manifest-alert] error={audit['notification_error']}")
    return _exit_code(audit)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
