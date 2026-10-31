"""Operator-door entry for an owner-approved, time-only scene intent renewal.

Run by ``deploy/operator-door/door-extend-scene-execution-window.sh`` as the
service user with the active release. It reads the capture's current rights
from the WebApp, records the extension through
``extend_scene_execution_window_by_operator`` (the only write), and then sends
the sealed extension to the WebApp over the existing signed Pipeline sync
transport so the WebApp's own readers honour the same window. A failed notice
is retried a bounded number of times and then reported (exit 1); rerunning the
same request re-sends it, because an equal expiry is "already covered".

Nothing here changes a spend cap, an attempt count, a provider, the task or
the rights; it starts no compute and calls no provider.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit

NOTICE_SCHEMA = "task_evaluation_scene_execution_window_extension_notice.v1"
_EXPIRES_AT = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
NOTICE_ATTEMPTS = 3


def parse_expires_at(value: Any) -> int:
    """A whole-second UTC instant written as ``YYYY-MM-DDTHH:MM:SSZ``."""
    if not isinstance(value, str) or _EXPIRES_AT.fullmatch(value) is None:
        raise ValueError("scene_execution_window_expiry_invalid")
    try:
        moment = dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        raise ValueError("scene_execution_window_expiry_invalid") from None
    return int(moment.timestamp())


def extension_notice(extension: Mapping[str, Any]) -> dict[str, Any]:
    """The fields the WebApp stores; the sealed record stays on the Pipeline host."""
    return {"schema_version": NOTICE_SCHEMA, "intent_id": extension["intent_id"],
            "intent_digest": extension["intent_digest"], "extension_digest": extension["extension_digest"],
            "original_expires_at_epoch": extension["original_expires_at_epoch"],
            "expires_at_epoch": extension["expires_at_epoch"],
            "authorization_reference": extension["authorization_reference"],
            "authorized_via": extension["authorized_via"], "issued_at_epoch": extension["issued_at_epoch"]}


def send_extension_notice(notice: Mapping[str, Any]) -> dict[str, Any]:
    """POST the notice with the existing signed Pipeline -> WebApp transport."""
    from .safe_outbound_http import pinned_api_policy, request
    from .task_evaluation_launch_webapp_sync import load_pipeline_sync_token
    from .webapp_sync import _pipeline_sync_headers, validated_https_sync_url

    configured = os.getenv("PIPELINE_SYNC_WEBAPP_URL", "").strip()
    if not configured:
        raise ValueError("scene_execution_window_webapp_url_missing")
    parsed = urlsplit(validated_https_sync_url(configured))
    origin = f"{parsed.scheme}://{parsed.netloc}"
    body = json.dumps(dict(notice), separators=(",", ":"), allow_nan=False).encode()
    url = (f"{origin}/api/internal/pipeline/task-evaluation-scene-intents/"
           f"{quote(str(notice['intent_id']), safe='')}/execution-window-extensions")
    try:
        response = request(url, method="POST", data=body,
            headers=_pipeline_sync_headers(load_pipeline_sync_token(), body), timeout_seconds=15,
            policy=pinned_api_policy(origin, max_response_bytes=20_000), max_response_bytes=20_000)
    except HTTPError as exc:
        code = None
        try:
            failure = json.loads(exc.read(4096))
            code = failure.get("code") if isinstance(failure, dict) else None
        except (ValueError, OSError):
            pass
        finally:
            exc.close()
        suffix = f":{code}" if isinstance(code, str) and re.fullmatch(r"[a-z0-9_]{1,80}", code) else ""
        raise ValueError(f"scene_execution_window_webapp_http_{exc.code}{suffix}") from None
    value = json.loads(response.body)
    if (not isinstance(value, dict) or value.get("status") not in {"recorded", "replayed"}
            or value.get("intent_id") != notice["intent_id"]
            or value.get("extension_digest") != notice["extension_digest"]
            or value.get("expires_at_epoch") != notice["expires_at_epoch"]):
        raise ValueError("scene_execution_window_webapp_ack_invalid")
    return value


def deliver_notice(notice: Mapping[str, Any], *, transport: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
                   sleep: Callable[[float], None] = time.sleep, attempts: int = NOTICE_ATTEMPTS) -> dict[str, Any]:
    """Bounded retries; the last failure is returned, never swallowed."""
    errors = []
    for attempt in range(attempts):
        try:
            return {"status": "acknowledged", "attempts": attempt + 1,
                    "receipt": (transport or send_extension_notice)(notice)}
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(str(exc)[:200] or type(exc).__name__)
            # A typed refusal (4xx) will not change on retry; transport failures may.
            if re.fullmatch(r"scene_execution_window_webapp_http_4[0-9]{2}(:.*)?", errors[-1]):
                break
            if attempt + 1 < attempts:
                sleep(2 ** (attempt + 1))
    return {"status": "failed", "attempts": len(errors), "errors": errors}


def current_capture_rights(*, intent_root: Path, request: Mapping[str, Any]) -> dict[str, Any]:
    """Read the capture's rights now, through the retained website source registration.

    Only a website scene has a WebApp task context. Anything else has no
    capture-rights reading here, so it stays unknown and is refused.
    """
    from .decision_evidence_contracts import cross_runtime_canonical_digest
    from .task_evaluation_scene_configuration_submission_inputs import checked_file, read
    from .website_scene_source_registry_paths import _index_path, binding_root
    from .website_task_context import load_current_website_task_context

    path = _index_path(binding_root({"intent_root": str(intent_root)}), request)
    if not path.is_file() or path.is_symlink():
        raise ValueError("scene_execution_window_capture_rights_unavailable")
    registration = read(path, digest_field="registration_digest")
    reference = (registration.get("references") or {}).get("task_context")
    if (registration.get("request_digest") != cross_runtime_canonical_digest(request)
            or not isinstance(reference, dict)):
        raise ValueError("scene_execution_window_capture_rights_unavailable")
    context = read(checked_file(reference["path"], reference), digest_field="context_digest")
    current = load_current_website_task_context(request_id=context["request_id"], scene_id=context["scene_id"],
        capture_id=context["capture_id"], purpose=context.get("purpose"))
    rights = current.get("capture_rights")
    if not isinstance(rights, dict):
        raise ValueError("scene_execution_window_capture_rights_unavailable")
    return {"capture_rights": rights, "request_id": context["request_id"], "capture_id": context["capture_id"]}


def run(request: Mapping[str, Any], *, door_caller: str, door_request_id: str, now: float | None = None,
        rights_reader: Callable[..., dict[str, Any]] = current_capture_rights,
        transport: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
        sleep: Callable[[float], None] = time.sleep) -> tuple[int, dict[str, Any]]:
    from .task_evaluation_scene_execution_window import extend_scene_execution_window_by_operator
    from .task_evaluation_scene_intent_contracts import CLIENTS_ENV, ROOT_ENV, _identifier, _read

    if (not isinstance(request, Mapping) or request.get("kind") != "extend-scene-execution-window"
            or not _identifier(request.get("intent_id"))):
        raise ValueError("scene_execution_window_extension_request_invalid")
    root_text = os.getenv(ROOT_ENV, "").strip()
    if not root_text:
        raise ValueError("scene_execution_window_intake_root_missing")
    intent_root = Path(root_text)
    trusted = {item.strip() for item in os.getenv(CLIENTS_ENV, "blueprint-webapp").split(",") if item.strip()}
    directory = intent_root / request["intent_id"]
    if directory.is_symlink():
        raise ValueError("scene_execution_window_store_unsafe")
    intent = _read(directory / "intent.json", "intent_digest")
    observed = rights_reader(intent_root=intent_root, request=intent["request"])
    result = extend_scene_execution_window_by_operator(queue_root=intent_root, intent_id=request["intent_id"],
        intent_digest=request["intent_digest"], expires_at_epoch=parse_expires_at(request["expires_at"]),
        authorization_reference=request["authorization_reference"], ack=request["ack"],
        door_caller=door_caller, door_request_id=door_request_id, trusted_clients=trusted,
        capture_rights=observed["capture_rights"], apply=request["apply"], now=now)
    status = result["status"]
    outcome = {"execution_window_extension_planned": "planned", "execution_window_extended": "extended",
               "execution_window_already_covers_request": "already_covered"}[status]
    report: dict[str, Any] = {"schema_version": "task_evaluation_scene_execution_window_door_result.v1",
        "door_outcome": outcome, "intent_id": request["intent_id"], "capture_id": observed.get("capture_id"),
        "result": result, "spend_caps_changed": False, "provider_mutation_performed": False}
    extension = result.get("extension") if status != "execution_window_already_covers_request" \
        else result.get("covering_extension")
    if status == "execution_window_extension_planned" or extension is None:
        # A plan writes nothing and tells the WebApp nothing; a window covered by
        # another issuer's record has nothing for this door to deliver.
        return 0, report
    report["webapp_notice"] = deliver_notice(extension_notice(extension), transport=transport, sleep=sleep)
    if report["webapp_notice"]["status"] != "acknowledged":
        report["door_outcome"] = "webapp_notice_failed"
        return 1, report
    return 0, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--door-caller", required=True)
    parser.add_argument("--door-request-id", required=True)
    args = parser.parse_args(argv)
    try:
        request = json.loads(args.request_json)
        code, report = run(request, door_caller=args.door_caller, door_request_id=args.door_request_id)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        text = str(exc)
        reason = text if re.fullmatch(r"[a-z0-9_:.-]{1,160}", text) else type(exc).__name__
        report, code = {"schema_version": "task_evaluation_scene_execution_window_door_result.v1",
                        "door_outcome": "refused", "code": reason, "provider_mutation_performed": False}, 2
    print(json.dumps(report, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
