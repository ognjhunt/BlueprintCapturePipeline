"""Automatically screen prepared website scenes and deliver diagnostic results.

This worker consumes the existing scene-packet queue after compilation. The
coordinator retains the site's budget across revisions and worker restarts;
only the canonical allocator bridge can execute a selected candidate.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit

from .decision_evidence_contracts import canonical_digest


def require_current_source(source: Mapping[str, Any], *, site_id: str) -> dict[str, Any]:
    """Re-read current website task and consent before every paid attempt."""
    from .team_scene_packet_worker import _read_json, resolve_capture_root
    from .website_task_context import load_current_website_task_context

    if not re.fullmatch(r"site-[A-Za-z0-9][A-Za-z0-9._:-]{0,191}", site_id):
        raise ValueError("beta_screening_site_identity_invalid")
    capture = resolve_capture_root(source)
    retained = _read_json(capture / "pipeline" / "website_task_context.json") or {}
    if (retained.get("scene_id") != site_id or retained.get("request_id") != site_id[5:]
            or retained.get("context_digest") != source.get("task_context_digest")
            or source.get("task_id") != "website-" + str(source.get("task_context_digest"))[7:27]):
        raise ValueError("beta_screening_source_context_mismatch")
    current = load_current_website_task_context(request_id=site_id[5:], scene_id=site_id,
        capture_id=str(source["capture_id"]), purpose=retained.get("purpose"))
    rights = current.get("capture_rights") or {}
    if (rights.get("consent_status") != "granted" or rights.get("consent_revoked") is not False
            or "robot_evaluation" not in rights.get("consent_scope", [])
            or rights.get("derived_scene_generation_allowed") is not True):
        raise ValueError("beta_screening_current_capture_consent_required")
    if current.get("context_digest") != source.get("task_context_digest"):
        raise ValueError("beta_screening_current_task_context_changed")
    return current


def enqueue_prepared_scenes(*, queue_root: str | Path, packet_queue_root: str | Path,
                           max_sources: int = 32) -> dict[str, Any]:
    """Adopt completed packet requests; blocked sources remain visible and retryable."""
    from .beta_policy_candidate_catalog import load_default_catalog
    from .beta_policy_screening_capabilities import build_planning_inputs
    from .beta_policy_screening_coordinator import _lock, _read, _root, _write, enqueue_screening_campaign
    from .beta_policy_screening_plan import build_screening_plan
    from .task_evaluation_configured_controls_progression_worker import _validate_source
    from .team_candidate_quick10_registry import profile_for_capture
    from .team_scene_packet_worker import _read_json, request_key, resolve_capture_root, validated_request

    root = _root(queue_root)
    packet_root = Path(packet_queue_root)
    if (not packet_root.is_absolute() or any(p.is_symlink() for p in (packet_root, *packet_root.parents))
            or not 1 <= max_sources <= 100):
        raise ValueError("beta_screening_source_queue_invalid")
    catalog = load_default_catalog()
    outcomes = []
    with _lock(root, ".source.lock"):
        states = root / "sources"
        states.mkdir(exist_ok=True, mode=0o700)
        paths = sorted((packet_root / "processed").glob("*.json"))
        cursor_path = root / "source_cursor.json"
        last = _read(cursor_path).get("last", "") if cursor_path.exists() else ""
        paths = [p for p in paths if p.name > last] + [p for p in paths if p.name <= last]
        for path in paths[:max_sources]:
            outcome: dict[str, Any] = {"request_key": path.stem, "status": "blocked"}
            try:
                request = validated_request(_read_json(path))
                if request_key(request) != path.stem:
                    raise ValueError("beta_screening_source_request_key_invalid")
                packet_status = _read_json(packet_root / "status" / path.name) or {}
                if packet_status.get("status") not in {"completed", "skipped"}:
                    raise ValueError("beta_screening_scene_packet_not_ready")
                capture = resolve_capture_root(request)
                retained = _read_json(capture / "pipeline" / "website_task_context.json") or {}
                source = {"scene_intent_id": request["intent_id"], "scene_intent_digest": request["intent_digest"],
                          "capture_id": request["capture_id"], "task_id": request["task_id"],
                          "task_context_digest": retained["context_digest"],
                          "scene_revision_digest": request["scene_revision_digest"]}
                campaign_id = "screening-" + canonical_digest({"source": source,
                    "catalog_digest": catalog["catalog_digest"], "protocol": "beta_policy_screening.v1"})[7:39]
                existing = root / "plans" / f"{campaign_id}.json"
                if existing.exists():
                    outcome.update(enqueue_screening_campaign(_read(existing), queue_root=root))
                else:
                    context = require_current_source(source, site_id=retained["scene_id"])
                    result, receipt, _zero = _validate_source(Path(request["source_launch_root"]))
                    if (result.get("configured_scene_revision_digest") != source["scene_revision_digest"]
                            or receipt.get("receipt_digest") != request["launch_receipt_digest"]):
                        raise ValueError("beta_screening_completed_scene_binding_changed")
                    profile = profile_for_capture(capture)
                    if not profile or profile.get("scene_id") != context["scene_id"]:
                        raise ValueError("beta_screening_scene_profile_unavailable")
                    # Reconcile packets prepared before the probe lane was
                    # installed as well as newly compiled scenes. Queueing is
                    # unpaid and idempotent; planning still requires a pass.
                    from .rigid_robot_free_probe_runner import enqueue_rigid_robot_free_probe

                    probe = enqueue_rigid_robot_free_probe(capture)
                    outcome["robot_free_probe_status"] = probe["status"]
                    inputs = build_planning_inputs(catalog=catalog, profile=profile, capture_root=capture,
                        source=source, inputs_dir=root / "inputs" / campaign_id)
                    plan = build_screening_plan(catalog, inputs["scene"], campaign_id=campaign_id,
                        capabilities=inputs["capabilities"], cost_quotes=inputs["cost_quotes"])
                    outcome.update(enqueue_screening_campaign(plan, queue_root=root))
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                # No raw remote response, URL, credentials or customer content in operational logs.
                code = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else ""
                outcome["blockers"] = [code if re.fullmatch(r"[a-z][a-z0-9_:,-]{0,199}", code)
                                       else "beta_screening_source_unavailable"]
            _write(states / path.name, outcome)
            _write(cursor_path, {"last": path.name})
            outcomes.append(outcome)
    from .beta_policy_screening_family_sources import enqueue_registered_family_scenes
    family = enqueue_registered_family_scenes(queue_root=root, max_sources=max_sources)
    return {"sources": outcomes + family["sources"]}


def publish_screening(publication: Mapping[str, Any]) -> dict[str, Any]:
    """Send the immutable publication using the existing signed Pipeline transport."""
    from .safe_outbound_http import pinned_api_policy, request
    from .task_evaluation_launch_webapp_sync import load_pipeline_sync_token
    from .webapp_sync import _pipeline_sync_headers, validated_https_sync_url

    configured = os.getenv("PIPELINE_SYNC_WEBAPP_URL", "").strip()
    if not configured:
        raise ValueError("beta_screening_webapp_url_missing")
    parsed = urlsplit(validated_https_sync_url(configured))
    origin = f"{parsed.scheme}://{parsed.netloc}"
    body = json.dumps(publication, separators=(",", ":"), allow_nan=False).encode()
    try:
        response = request(f"{origin}/api/internal/pipeline/beta-policy-screening-runs", method="POST",
            data=body, headers=_pipeline_sync_headers(load_pipeline_sync_token(), body), timeout_seconds=15,
            policy=pinned_api_policy(origin, max_response_bytes=100_000), max_response_bytes=100_000)
    except HTTPError as exc:
        code = exc.code
        exc.close()
        raise ValueError(f"beta_screening_publication_http_{code}") from None
    value = json.loads(response.body)
    if (not isinstance(value, dict) or value.get("status") not in {"created", "updated", "replayed"}
            or value.get("run_id") != publication["campaign_id"]
            or value.get("revision") != publication["revision"]
            or value.get("publication_digest") != publication["publication_digest"]):
        raise ValueError("beta_screening_publication_ack_invalid")
    return value


def deliver_publications(*, queue_root: str | Path, transport=None,
                         refresh_startable: bool = False) -> dict[str, Any]:
    from .beta_policy_screening_coordinator import acknowledge_publication, pending_publications
    sent = failed = 0
    for publication in pending_publications(queue_root=queue_root, refresh_startable=refresh_startable):
        try:
            (transport or publish_screening)(publication)
            acknowledged = acknowledge_publication(publication["campaign_id"], publication["publication_digest"],
                queue_root=queue_root)
            sent += int(acknowledged)
        except (OSError, ValueError, RuntimeError):
            failed += 1
    return {"acknowledged": sent, "pending": failed}


def main(argv: Sequence[str] | None = None) -> int:
    from .beta_policy_screening_coordinator import QUEUE_ROOT_ENV, _root, process_screening_queue
    from .team_scene_packet_worker import QUEUE_ROOT_ENV as PACKET_QUEUE_ENV
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", default=os.getenv(QUEUE_ROOT_ENV))
    parser.add_argument("--packet-queue-root", default=os.getenv(PACKET_QUEUE_ENV))
    parser.add_argument("--max-campaigns", type=int, default=2)
    parser.add_argument("--max-launches", type=int, default=1)
    parser.add_argument("--max-sources", type=int, default=32)
    args = parser.parse_args(argv)
    root = _root(args.queue_root)
    sources = enqueue_prepared_scenes(queue_root=root, packet_queue_root=args.packet_queue_root,
        max_sources=args.max_sources) if args.packet_queue_root else {"status": "source_queue_not_configured"}
    # Publish queued rows before a potentially long allocator attempt.
    before = deliver_publications(queue_root=root, refresh_startable=True)
    execution = process_screening_queue(queue_root=root, max_campaigns=args.max_campaigns,
        max_launches=args.max_launches)
    print(json.dumps({"sources": sources, "execution": execution, "publication_before": before,
                      "publication_after": deliver_publications(queue_root=root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
