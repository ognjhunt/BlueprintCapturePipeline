"""Automatically evaluate prepared website scenes and deliver each job's result.

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


def implementation_commit() -> str:
    """The deployed commit, as planning binds it (``build_planning_inputs``)."""
    import subprocess

    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                          capture_output=True, text=True, check=True).stdout.strip()


def enqueue_prepared_scenes(*, queue_root: str | Path, packet_queue_root: str | Path,
                           max_sources: int = 32) -> dict[str, Any]:
    """Adopt completed packet requests; blocked sources remain visible and retryable."""
    from .beta_policy_candidate_catalog import load_default_catalog
    from .beta_policy_screening_capabilities import build_planning_inputs
    from .beta_policy_screening_coordinator import (
        _lock, _read, _root, _write, enqueue_screening_campaign, latest_campaign, replan_scope, supersede_campaign,
    )
    from .beta_policy_screening_holds import RELEASE, packet_hold, probe_hold, settle_hold
    from .beta_policy_screening_job_ceiling import plan_within_job_ceiling
    from .beta_policy_screening_team import catalog_for_profile
    from .rigid_robot_free_probe_runner import _running_release, source_commit
    from .task_evaluation_configured_controls_progression_worker import _validate_source
    from .team_candidate_quick10_registry import profile_for_capture
    from .team_scene_packet_worker import (
        _read_json,
        current_release_commit,
        request_key,
        requeue_transient_blocked,
        resolve_capture_root,
        validated_request,
    )
    from .website_candidate_ranking import retained_priority
    from .website_candidate_selection import retain_campaign_catalog

    root = _root(queue_root)
    packet_root = Path(packet_queue_root)
    if (not packet_root.is_absolute() or any(p.is_symlink() for p in (packet_root, *packet_root.parents))
            or not 1 <= max_sources <= 100):
        raise ValueError("beta_screening_source_queue_invalid")
    try:
        # The packet worker runs only when something is pending: this timer's pass returns a timed-out
        # compile or a failed offer post to it, bounded and after a wait; a deterministic refusal goes back once
        # per newly deployed release (the release this pass runs from), so a deploy that fixes it re-drives it.
        packet_requeue = requeue_transient_blocked(queue_root=packet_root)
    except (OSError, ValueError) as exc:
        from .bounded_error import error_description

        packet_requeue = {"status": "failed", "error": error_description(exc)}
    base_catalog = load_default_catalog()
    # The releases the two lanes retry their refusals under: an evaluation is held only once nothing will.
    packet_release = current_release_commit()
    probe_release = _running_release({"source_commit": source_commit})
    outcomes = []
    # Only the newest request of each scene intent speaks for its hold: an older revision never flips it.
    newest = newest_intent_requests(packet_root)
    with _lock(root, ".source.lock"):
        states = root / "sources"
        states.mkdir(exist_ok=True, mode=0o700)
        paths = sorted((packet_root / "processed").glob("*.json"))
        cursor_path = root / "source_cursor.json"
        last = _read(cursor_path).get("last", "") if cursor_path.exists() else ""
        paths = [p for p in paths if p.name > last] + [p for p in paths if p.name <= last]
        for path in paths[:max_sources]:
            outcome: dict[str, Any] = {"request_key": path.stem, "status": "blocked"}
            request: dict[str, Any] | None = None
            # A hold, RELEASE on positive progress, or None: nothing decided, so a hold is kept.
            hold: tuple[str, str | None] | str | None = None
            try:
                request = validated_request(_read_json(path))
                if request_key(request) != path.stem:
                    raise ValueError("beta_screening_source_request_key_invalid")
                packet_status = _read_json(packet_root / "status" / path.name) or {}
                if packet_status.get("status") not in {"completed", "skipped"}:
                    # Say why the scene half is not ready, and whether it is still being retried.
                    outcome["scene_packet"] = {key: packet_status[key] for key in
                                               ("status", "blockers", "automatic_retry") if key in packet_status}
                    hold = packet_hold(packet_status, release=packet_release)
                    raise ValueError("beta_screening_scene_packet_not_ready")
                capture = resolve_capture_root(request)
                retained = _read_json(capture / "pipeline" / "website_task_context.json") or {}
                source = {"scene_intent_id": request["intent_id"], "scene_intent_digest": request["intent_digest"],
                          "capture_id": request["capture_id"], "task_id": request["task_id"],
                          "task_context_digest": retained["context_digest"],
                          "scene_revision_digest": request["scene_revision_digest"]}
                profile = profile_for_capture(capture)
                if not profile:
                    raise ValueError("beta_screening_scene_profile_unavailable")
                catalog = catalog_for_profile(profile, base_catalog)
                campaign_id = "screening-" + canonical_digest({"source": source,
                    "catalog_digest": catalog["catalog_digest"], "protocol": "beta_policy_screening.v1"})[7:39]
                existing = root / "plans" / f"{campaign_id}.json"
                replan: list[str] | None = None
                superseded = None
                if existing.exists():
                    # A deploy since planning leaves unprepared candidates bound to
                    # a commit preparation refuses: plan them again at this one.
                    current = latest_campaign(campaign_id, queue_root=root)
                    commit = implementation_commit()
                    replan = replan_scope(current, implementation_commit=commit, queue_root=root)
                    if replan is None:
                        outcome.update(enqueue_screening_campaign(_read(root / "plans" / f"{current}.json"),
                                                                  queue_root=root))
                    else:
                        superseded = current
                        campaign_id = "screening-" + canonical_digest({"source": source,
                            "catalog_digest": catalog["catalog_digest"], "protocol": "beta_policy_screening.v1",
                            "supersedes": current, "implementation_commit": commit})[7:39]
                if not existing.exists() or replan is not None:
                    context = require_current_source(source, site_id=retained["scene_id"])
                    result, receipt, _zero = _validate_source(Path(request["source_launch_root"]))
                    if (result.get("configured_scene_revision_digest") != source["scene_revision_digest"]
                            or receipt.get("receipt_digest") != request["launch_receipt_digest"]):
                        raise ValueError("beta_screening_completed_scene_binding_changed")
                    if not profile or profile.get("scene_id") != context["scene_id"]:
                        raise ValueError("beta_screening_scene_profile_unavailable")
                    # Reconcile packets prepared before the probe lane was
                    # installed as well as newly compiled scenes. Queueing is
                    # unpaid and idempotent; planning still requires a pass.
                    from .rigid_robot_free_probe_runner import enqueue_rigid_robot_free_probe

                    probe = enqueue_rigid_robot_free_probe(capture)
                    outcome["robot_free_probe_status"] = probe["status"]
                    # The intake ranking orders the plan; the site budget decides how far it runs.
                    priority = retained_priority(capture / "pipeline", task_context_digest=source["task_context_digest"])
                    outcome["candidate_order"] = "intake_ranking" if priority else "beta_default_priority"
                    if not priority:
                        from .beta_policy_screening_default_priority import default_screening_priority

                        priority = default_screening_priority()
                    inputs = build_planning_inputs(catalog=catalog, profile=profile, capture_root=capture,
                        source=source, inputs_dir=root / "inputs" / campaign_id, priority=priority)
                    # The job's one dollar ceiling sizes the plan; unreadable, it holds and retries.
                    plan = plan_within_job_ceiling(catalog, inputs["scene"], campaign_id=campaign_id,
                        queue_root=root, context=context, capture_root=capture,
                        capabilities=inputs["capabilities"], cost_quotes=inputs["cost_quotes"],
                        priority=inputs.get("candidate_priority"), candidate_ids=replan, superseding=superseded)
                    retain_campaign_catalog(queue_root=root, campaign_id=campaign_id, catalog=catalog)
                    if superseded is None:
                        outcome.update(enqueue_screening_campaign(plan, queue_root=root))
                    else:
                        outcome.update(supersede_campaign(superseded, plan, implementation_commit=commit,
                                                          queue_root=root))
                # A campaign is planned (now or before): the evaluation started.
                hold = RELEASE
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                # No raw remote response, URL, credentials or customer content in operational logs.
                code = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else ""
                typed = re.fullmatch(r"[a-z][a-z0-9_:,-]{0,199}", code) is not None
                outcome["blockers"] = [code if typed else "beta_screening_source_unavailable"]
                if not typed:
                    from .bounded_error import error_description

                    # The generic code covers a full disk, a missing retained
                    # file and an unreadable record alike; keep which it was.
                    outcome["error"] = error_description(exc)
                if hold is None and request is not None and code.startswith("team_evaluation_robot_free_probe_"):
                    try:
                        # A probe refusal read as not final (queued, waiting, due a new release's try) is progress.
                        hold = probe_hold(code, resolve_capture_root(request), release=probe_release) or RELEASE
                    except (OSError, ValueError, KeyError, TypeError):
                        hold = None  # unreadable now: the hold is kept and the next pass decides again
            if request is not None and newest.get(request["intent_id"]) == ("processed", path.name):
                # The website job hears by name when its evaluation cannot start, and again when it moves.
                settled = settle_hold(root, request, hold)
                if settled is not None:
                    outcome["evaluation_hold"] = settled
            _write(states / path.name, outcome)
            _write(cursor_path, {"last": path.name})
            outcomes.append(outcome)
        # A team half a new release (or a due retry) returned to its queue is moving again: release its hold.
        for state in ("pending", "processing"):
            for path in sorted((packet_root / state).glob("*.json")):
                try:
                    request = validated_request(_read_json(path))
                except ValueError:
                    continue
                if newest.get(request["intent_id"]) == (state, path.name):
                    settle_hold(root, request, RELEASE)
    outcomes.extend(_scene_preparation_holds(root, packet_root, max_sources=max_sources, newest=newest))
    from .beta_policy_screening_family_sources import enqueue_registered_family_scenes
    family = enqueue_registered_family_scenes(queue_root=root, max_sources=max_sources)
    return {"sources": outcomes + family["sources"], "team_scene_packet_requeue": packet_requeue}


_REQUEST_STATES = ("pending", "processing", "processed")


def newest_intent_requests(packet_root: Path) -> dict[str, tuple[str, str]]:
    """Each scene intent's newest request (its latest scene revision): ``(where, file name)``, by when it was queued.

    A request is written once and moved by rename, so its modification time is when it was queued. The newest
    of a team half's request and a held request of the same intent speaks for the intent's hold.
    """
    from .team_scene_packet_worker import HOLDS_DIRNAME, _read_json

    newest: dict[str, tuple[int, str, str, str]] = {}
    for where in (*_REQUEST_STATES, HOLDS_DIRNAME):
        for path in (packet_root / where).glob("*.json"):
            try:
                if path.is_symlink():
                    continue
                stamp = path.stat().st_mtime_ns
            except OSError:
                continue
            intent = str((_read_json(path) or {}).get("intent_id") or "")
            if intent and (intent not in newest or (stamp, path.name) > newest[intent][:2]):
                newest[intent] = (stamp, path.name, where, path.name)
    return {intent: (row[2], row[3]) for intent, row in newest.items()}


def _scene_preparation_holds(root: Path, packet_root: Path, *, max_sources: int,
                             newest: Mapping[str, tuple[str, str]]) -> list[dict[str, Any]]:
    """Hold each website job whose scene preparation retained a decided failed probe receipt.

    Its team half is never compiled (``team_scene_packet_worker.enqueue_team_scene_request``), so no source
    ever reaches the pass above. The receipt is re-read, never trusted from the request. The entries rotate
    under their own cursor, so none waits behind the first ``max_sources``. An entry is set aside under
    ``holds/settled/`` once the website acknowledged its hold, or under ``holds/superseded/`` once a newer
    request of its intent exists; nothing is deleted.
    """
    from .beta_policy_screening_coordinator import _read, _write
    from .beta_policy_screening_holds import scene_preparation_probe_failed, settle_hold
    from .team_scene_packet_worker import HOLDS_DIRNAME, _read_json, request_key, validated_request

    directory = packet_root / HOLDS_DIRNAME
    cursor_path = root / "hold_request_cursor.json"
    last = _read(cursor_path).get("last", "") if cursor_path.exists() else ""
    paths = sorted(directory.glob("*.json"))
    paths = [p for p in paths if p.name > last] + [p for p in paths if p.name <= last]

    def set_aside(path: Path, where: str) -> None:
        (directory / where).mkdir(exist_ok=True, mode=0o750)
        os.replace(path, directory / where / path.name)

    outcomes = []
    for path in paths[:max_sources]:
        outcome: dict[str, Any] = {"request_key": path.stem, "status": "held"}
        try:
            request = validated_request(_read_json(path))
            if request_key(request) != path.stem or not scene_preparation_probe_failed(request):
                raise ValueError("beta_screening_hold_request_invalid")
            if newest.get(request["intent_id"]) != (HOLDS_DIRNAME, path.name):
                # A newer scene revision of the intent speaks for its hold.
                set_aside(path, "superseded")
                outcome["status"] = "superseded"
            else:
                held = settle_hold(root, request, ("robot_free_probe_failed", "scene_preparation_receipt_failed"))
                outcome["evaluation_hold"] = held
                if held is not None and held.get("acknowledged") is True:
                    set_aside(path, "settled")  # the website keeps it; nothing more to send
        except (OSError, ValueError, KeyError, TypeError) as exc:
            code = str(exc) if isinstance(exc, ValueError) else ""
            outcome.update(status="blocked", blockers=[code if re.fullmatch(r"[a-z][a-z0-9_:,-]{0,199}", code)
                                                       else "beta_screening_hold_request_unavailable"])
        _write(cursor_path, {"last": path.name})
        outcomes.append(outcome)
    return outcomes


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
    failure: dict[str, Any] = {}
    for publication in pending_publications(queue_root=queue_root, refresh_startable=refresh_startable):
        try:
            (transport or publish_screening)(publication)
            acknowledged = acknowledge_publication(publication["campaign_id"], publication["publication_digest"],
                queue_root=queue_root)
            sent += int(acknowledged)
        except (OSError, ValueError, RuntimeError) as exc:
            failed += 1
            from .bounded_error import error_description, rate_limited

            # Name the refusal for the worker status (a 429 or an outage is
            # "unavailable") and keep the failure's type and redacted message.
            http = re.fullmatch(r"beta_screening_publication_http_([0-9]{3})", str(exc))
            status = int(http.group(1)) if http else 0
            failure = {"last_failure_category": (
                "auth" if status in (401, 403) else "source" if status == 409
                else "ack" if str(exc) == "beta_screening_publication_ack_invalid"
                else "unavailable" if rate_limited(exc) or status >= 500 or isinstance(exc, OSError)
                else "unknown"), "last_error": error_description(exc)}
    return {"acknowledged": sent, "pending": failed, **failure}


def main(argv: Sequence[str] | None = None) -> int:
    from .beta_policy_screening_coordinator import QUEUE_ROOT_ENV, _root, process_screening_queue
    from .beta_policy_screening_status import write_worker_status
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
    write_worker_status(root, publication_before=before)
    execution = process_screening_queue(queue_root=root, max_campaigns=args.max_campaigns,
        max_launches=args.max_launches)
    after = deliver_publications(queue_root=root)
    from .beta_policy_screening_holds import deliver_holds

    # A held evaluation (no campaign could start) reaches the website by name, like a campaign.
    holds = deliver_holds(queue_root=root)
    from .website_candidate_selection import deliver_selections, retain_selections
    try:
        # After a terminal result is published, choose from it; local and unpaid.
        selections = retain_selections(queue_root=root)
        selections["delivery"] = deliver_selections(queue_root=root)
    except (OSError, ValueError, KeyError) as exc:
        code = str(exc) if isinstance(exc, ValueError) else ""
        selections = {"status": "failed", "blocker": code if re.fullmatch(r"[a-z][a-z0-9_:,-]{0,199}", code)
                      else "website_candidate_selection_failed"}
    status = write_worker_status(root, publication_before=before, publication_after=after, execution=execution)
    print(json.dumps({"sources": sources, "execution": execution, "publication_before": before,
                      "publication_after": after, "holds": holds, "selections": selections,
                      "worker_status": status},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
