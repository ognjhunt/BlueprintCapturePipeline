"""Operator alerting for live pipeline control-plane manifests.

The live control plane intentionally exits zero when a pass is externally
blocked so timers keep running. This module is the separate operator-signal
surface: it reads the latest manifest, sends a bounded webhook notification
when the pass is blocked, and can fail closed when production alerting is not
configured.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from math import isfinite
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
from urllib.parse import urlsplit

from .common import ensure_dir, read_json_any, utc_now_iso, write_json


LIVE_PIPELINE_MANIFEST_ALERT_SCHEMA_VERSION = "blueprint_live_pipeline_manifest_alert.v1"
OPERATOR_ALERT_WEBHOOK_URL_ENV = "BLUEPRINT_OPERATOR_ALERT_WEBHOOK_URL"
OPERATOR_ALERT_REQUIRE_WEBHOOK_ENV = "BLUEPRINT_OPERATOR_ALERT_REQUIRE_WEBHOOK"
DEFAULT_MANIFEST_PATH = (
    "/var/lib/blueprint/pipeline-control-plane/live_pipeline_control_plane_manifest.json"
)


def _string(value: Any) -> str:
    return str(value or "").strip()


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any, *, limit: int = 12) -> list[str]:
    if not isinstance(value, list):
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


def _manifest_blockers(manifest: Mapping[str, Any]) -> list[str]:
    blockers = _string_list(manifest.get("blockers"))
    setup = _mapping(manifest.get("setup"))
    blockers.extend(_string_list(setup.get("blockers"), limit=12 - len(blockers)))
    if len(blockers) < 12:
        external_input_packet = _mapping(manifest.get("external_input_packet"))
        blockers.extend(
            _string_list(
                external_input_packet.get("blockers"),
                limit=12 - len(blockers),
            )
        )
    return blockers[:12]


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
) -> str:
    status = _string(manifest.get("status")) or "unknown"
    job_id = _string(manifest.get("job_id"))
    capture_root = _string(manifest.get("capture_root"))
    page_event = _mapping(manifest.get("page_event"))
    blocker_text = (
        ", ".join(blockers[:5])
        if blockers
        else (
            "threshold crossing requires operator notification"
            if page_event.get("required") is True
            else "status contains blocked"
        )
    )
    if manifest.get("schema_version") == "blueprint.paid_spend_admission_lock.v1":
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
    parts = [headline, f"manifest={manifest_path}"]
    if job_id:
        parts.append(f"job_id={job_id}")
    if capture_root:
        parts.append(f"capture_root={capture_root}")
    parts.append(f"blockers={blocker_text}")
    return " ".join(parts)[:3000]


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


def build_live_pipeline_manifest_alert(
    *,
    manifest_path: Path,
    output_path: Path | None = None,
    webhook_url: str | None = None,
    require_webhook: bool | None = None,
    dry_run: bool = False,
    timeout_seconds: float = 10.0,
) -> Dict[str, Any]:
    resolved_manifest_path = Path(manifest_path).expanduser().resolve()
    resolved_output_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else resolved_manifest_path.parent / "live_pipeline_manifest_alert.json"
    )
    manifest = _read_manifest(resolved_manifest_path)
    blockers = _manifest_blockers(manifest)
    alert_required = _alert_required(manifest)
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
    )

    notification_status = "not_required"
    notification_error = ""
    attempted = False
    if alert_required and resolved_webhook_url and not dry_run:
        attempted = True
        try:
            _post_webhook(
                resolved_webhook_url,
                {"text": message_text},
                timeout_seconds=timeout_seconds,
            )
            notification_status = "sent"
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
    if status in {"sent", "dry_run"}:
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
