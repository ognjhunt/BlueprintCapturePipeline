"""Keep the scene stage moving and tell each website job, by name, when it is held.

Beta journey: a prepared site's scene packet, after its scene is published. These duties
used to run inside the early trial's timer (``beta_policy_screening_worker``); the owner
dropped Blueprint's own early trial (2026-11-14), so they run here, in their own unit
(``blueprint-scene-stage-maintenance.{service,timer}``), and only they run here:

- the scene-packet stage's recovery and requeue, and the site's success-rule
  confirmation exchange (``recover_scene_packets``);
- the "held" notices the site sees for the scene stage (packet blocked, robot-free probe
  failed / not completed / unavailable, cold storage, a stalled source, scene-preparation
  holds), kept in the same store with the same schema and hold ids, and delivered to the
  WebApp's holds route exactly as before (``scene_stage_holds``);
- bringing an open job's scene run back from cold storage (``_restore_source_run``);
- queueing the robot-free probe of a packet compiled before the probe lane existed
  (``rigid_robot_free_probe_runner.enqueue_rigid_robot_free_probe``).

Nothing here plans, reserves, launches or publishes an evaluation campaign, and nothing
spends: the probe is only queued (its own lane decides and pays under its own caps), and a
restore is unpaid. Where the trial's pass planned a campaign once a completed source's probe
gate passed, this pass records that the scene stage moves on (the job's hold is released).
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import re
from typing import Any

from .decision_evidence_contracts import canonical_digest

#: The outcome a source's state keeps once its probe gate admitted the scene: nothing more to decide for
#: this request (a decided passed receipt is never replaced), so later passes read no remote state for it.
PROBE_PASSED = "probe_passed"


def require_current_source(source: Mapping[str, Any], *, site_id: str) -> dict[str, Any]:
    """Re-read current website task and consent before every paid attempt."""
    from .team_scene_packet_worker import _read_json, resolve_capture_root
    from .website_task_context import load_current_website_task_context

    if not re.fullmatch(r"site-[A-Za-z0-9][A-Za-z0-9._:-]{0,191}", site_id):
        raise ValueError("beta_screening_site_identity_invalid")
    from .website_task_context_lineage import accepts_bound_digest

    capture = resolve_capture_root(source)
    retained = _read_json(capture / "pipeline" / "website_task_context.json") or {}
    # The source binds its own context, or the unconfirmed parent of the retained confirmed successor:
    # confirming a brief is consent state, not task content (website_task_context_lineage).
    if (retained.get("scene_id") != site_id or retained.get("request_id") != site_id[5:]
            or not accepts_bound_digest(retained, source.get("task_context_digest"))
            or source.get("task_id") != "website-" + str(source.get("task_context_digest"))[7:27]):
        raise ValueError("beta_screening_source_context_mismatch")
    current = load_current_website_task_context(request_id=site_id[5:], scene_id=site_id,
        capture_id=str(source["capture_id"]), purpose=retained.get("purpose"))
    rights = current.get("capture_rights") or {}
    if (rights.get("consent_status") != "granted" or rights.get("consent_revoked") is not False
            or "robot_evaluation" not in rights.get("consent_scope", [])
            or rights.get("derived_scene_generation_allowed") is not True):
        raise ValueError("beta_screening_current_capture_consent_required")
    if not accepts_bound_digest(current, source.get("task_context_digest")):
        raise ValueError("beta_screening_current_task_context_changed")
    return current


def _restore_source_run(request: Mapping[str, Any], *, context: Mapping[str, Any] | None = None,
                        packet_queue_root: str | Path | None = None) -> dict[str, Any]:
    """Bring an open job's scene run back from cold storage (``scene_run_restore``); never raises.

    Privacy: the job's withdrawal and its source (consent, rights) are read again before anything is fetched
    and again under the deletion's lock before anything is placed.
    """
    from .scene_run_restore import restore_open_job_run
    from .screening_site_withdrawal import site_withdrawal_state
    from .team_scene_packet_worker import _read_json, resolve_capture_root

    try:
        retained = _read_json(resolve_capture_root(request) / "pipeline" / "website_task_context.json") or {}
        source = {"scene_intent_id": request["intent_id"], "scene_intent_digest": request["intent_digest"],
                  "capture_id": request["capture_id"], "task_id": request["task_id"],
                  "task_context_digest": retained["context_digest"],
                  "scene_revision_digest": request["scene_revision_digest"]}
        if context is None:
            context = require_current_source(source, site_id=retained["scene_id"])

        def still_current() -> bool:
            try:
                require_current_source(source, site_id=retained["scene_id"])
            except Exception:  # noqa: BLE001 - not current now (revoked, ended, unreadable): nothing lands
                return False
            return site_withdrawal_state(str(retained["scene_id"])) in {"not_withdrawn", "not_configured"}

        return restore_open_job_run(Path(request["source_launch_root"]), window_open=not job_window_closed(context),
                                    still_current=still_current, packet_queue_root=packet_queue_root)
    except Exception as exc:  # noqa: BLE001 - an unread source restores nothing now; the next pass asks again
        return {"status": "source_unavailable", "error_type": type(exc).__name__}


def job_window_closed(context: Mapping[str, Any]) -> bool:
    """Whether the current source says its job's window has ended (its execution deadline or its
    consent's expiry). Unknown or unreadable is not closed: the launch preflight still decides."""
    import time

    from .beta_policy_screening_control import _deadline

    try:
        deadline = _deadline(context)
    except ValueError:
        return False
    return deadline is not None and deadline <= time.time()


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
    from .scene_stage_holds import scene_preparation_probe_failed, settle_hold
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


def recover_scene_packets(packet_root: Path) -> dict[str, Any]:
    """The scene-packet stage's recovery, requeue and site confirmation exchange; the one call to re-point.

    The packet worker runs only when something is pending: this pass returns a timed-out compile or a
    failed offer post to it, bounded and after a wait; a deterministic refusal goes back once per newly
    deployed release (the release this pass runs from), so a deploy that fixes it re-drives it. The same
    call asks the WebApp for each site's decision on its success rule (``website_owner_success_confirmation``).
    """
    from .team_scene_packet_worker import requeue_transient_blocked

    return requeue_transient_blocked(queue_root=packet_root)


def probe_gate(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Admit the scene's robot-free probe receipt as the trial's planning did, or raise its typed refusal.

    The same reads and the same gate call ``beta_policy_screening_capabilities.build_planning_inputs``
    made before planning (``robot_free_probe_gate.require_robot_free_probe``), so a refusal carries the
    same ``team_evaluation_robot_free_probe_<reason>`` code the job's hold was decided from.
    """
    from .robot_free_probe_gate import (
        NAVIGATION_TASK_KIND,
        require_robot_free_probe,
        retained_probe_receipt,
        task_success_contract_binding,
    )
    from .team_candidate_quick10_registry import scene_facts

    facts = scene_facts(profile)
    inputs = facts["inputs"]
    task = facts["scene"]["task_spec"]
    navigation = task.get("task_kind") == NAVIGATION_TASK_KIND
    # A navigation record keeps its probe receipt in its own directory, beside the packet (its scene facts).
    receipt = retained_probe_receipt(Path(facts["robot_free_probe_root"] if navigation
                                          else profile["robot_free_probe_root"]))
    return require_robot_free_probe(receipt, scene_revision_digest=inputs["scene_revision_digest"],
        task_kind=task["task_kind"], task_spec_digest=canonical_digest(task),
        task_success_contract_digest=task_success_contract_binding(task["task_kind"], inputs["task_success_contract"]))


def scene_stage_pass(*, queue_root: str | Path, packet_queue_root: str | Path,
                     max_sources: int = 32) -> dict[str, Any]:
    """Recover the packet queue, then decide and record one hold per scene intent; plans nothing.

    ``queue_root`` is where the holds, the source states and their cursors have always been kept (the
    screening queue root); ``packet_queue_root`` is the team scene packet queue.
    """
    from .beta_policy_screening_coordinator import _lock, _read, _root, _write
    from .rigid_robot_free_probe_runner import _running_release, enqueue_rigid_robot_free_probe, source_commit
    from .scene_run_cold_storage import COLD_STORAGE, cold_storage_hold, in_cold_storage, require_on_disk
    from .scene_stage_holds import (
        RELEASE, WINDOW_CLOSED, blocked_since, packet_hold, probe_hold, settle_hold, stalled_source_hold,
    )
    from .task_evaluation_configured_controls_source_evidence import _validate_source
    from .team_candidate_quick10_registry import profile_for_capture
    from .team_scene_packet_worker import (
        _read_json,
        current_release_commit,
        request_key,
        resolve_capture_root,
        validated_request,
    )

    root = _root(queue_root)
    packet_root = Path(packet_queue_root)
    if (not packet_root.is_absolute() or any(p.is_symlink() for p in (packet_root, *packet_root.parents))
            or not 1 <= max_sources <= 100):
        raise ValueError("beta_screening_source_queue_invalid")
    try:
        packet_requeue = recover_scene_packets(packet_root)
    except (OSError, ValueError) as exc:
        from .bounded_error import error_description

        packet_requeue = {"status": "failed", "error": error_description(exc)}
    # The releases the two lanes retry their refusals under: a job is held only once nothing will.
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
            # Since when this source has been refused the same way (kept in its state file only).
            refused_since: str | None = None
            previous = _read_json(states / path.name) or {}
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
                    if hold is None and packet_status.get("blockers") == [COLD_STORAGE]:
                        # The team half cannot be compiled again while its scene run is in cold storage: the job
                        # is told at once, by name, and an open job's run is brought back by itself.
                        outcome["scene_run_restore"] = _restore_source_run(request, packet_queue_root=packet_root)
                        hold = cold_storage_hold(request["source_launch_root"])
                    raise ValueError("beta_screening_scene_packet_not_ready")
                if previous.get("scene_stage") == PROBE_PASSED:
                    # This request's probe gate already admitted its scene; a passed receipt is never replaced.
                    outcome.update(status="scene_stage_ready", scene_stage=PROBE_PASSED)
                    hold = RELEASE
                else:
                    capture = resolve_capture_root(request)
                    retained = _read_json(capture / "pipeline" / "website_task_context.json") or {}
                    source = {"scene_intent_id": request["intent_id"], "scene_intent_digest": request["intent_digest"],
                              "capture_id": request["capture_id"], "task_id": request["task_id"],
                              "task_context_digest": retained["context_digest"],
                              "scene_revision_digest": request["scene_revision_digest"]}
                    profile = profile_for_capture(capture)
                    if not profile:
                        raise ValueError("beta_screening_scene_profile_unavailable")
                    context = require_current_source(source, site_id=retained["scene_id"])
                    # A scene run moved to cold storage is its own typed state, never a catch-all refusal; an
                    # open job's run is brought back by itself first (``scene_run_restore``).
                    kept = None
                    if in_cold_storage(Path(request["source_launch_root"])):
                        from .scene_run_restore import clean_stale_staging
                        clean_stale_staging(Path(request["source_launch_root"]))  # open or not: no crashed leftovers
                        if job_window_closed(context):
                            raise ValueError(WINDOW_CLOSED)
                        # What its pointer kept is enough to bind the source; only without it is the run restored.
                        from .task_evaluation_configured_controls_source_evidence import validate_retained_source
                        try:
                            kept = validate_retained_source(Path(request["source_launch_root"]))
                            outcome["scene_run_restore"] = {"status": "not_needed_read_from_pointer"}
                        except Exception:  # noqa: BLE001 - not kept, or not intact: bring the run back
                            outcome["scene_run_restore"] = _restore_source_run(request, context=context,
                                                                                packet_queue_root=packet_root)
                    if kept is None:
                        require_on_disk(Path(request["source_launch_root"]))
                    result, receipt, _zero = kept or _validate_source(Path(request["source_launch_root"]))
                    if (result.get("configured_scene_revision_digest") != source["scene_revision_digest"]
                            or receipt.get("receipt_digest") != request["launch_receipt_digest"]):
                        raise ValueError("beta_screening_completed_scene_binding_changed")
                    if profile.get("scene_id") != context["scene_id"]:
                        raise ValueError("beta_screening_scene_profile_unavailable")
                    # Reconcile packets prepared before the probe lane was installed as well as newly
                    # compiled scenes. Queueing is unpaid and idempotent; the probe lane decides and pays.
                    probe = enqueue_rigid_robot_free_probe(capture)
                    outcome["robot_free_probe_status"] = probe["status"]
                    probe_gate(profile)
                    # The scene's probe admitted it: the scene stage moves on, so its hold is released.
                    outcome.update(status="scene_stage_ready", scene_stage=PROBE_PASSED)
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
                # A refusal that keeps coming back the same way is told to the job as held, once it has
                # lasted long enough; the source is still read on every pass.
                refused_since = blocked_since(previous, outcome["blockers"])
                if hold is None and code == COLD_STORAGE:
                    hold = cold_storage_hold(request["source_launch_root"] if request is not None else None)
                if hold is None:
                    hold = stalled_source_hold(outcome["blockers"][0], refused_since)
            if request is not None and newest.get(request["intent_id"]) == ("processed", path.name):
                # The website job hears by name when its scene stage is held, and again when it moves.
                settled = settle_hold(root, request, hold)
                if settled is not None:
                    outcome["evaluation_hold"] = settled
            _write(states / path.name, outcome if refused_since is None
                   else {**outcome, "blocked_since": refused_since})
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
    return {"sources": outcomes, "team_scene_packet_requeue": packet_requeue}


def main(argv: Sequence[str] | None = None) -> int:
    from .beta_policy_screening_coordinator import QUEUE_ROOT_ENV, _root
    from .scene_stage_holds import deliver_holds
    from .team_scene_packet_worker import QUEUE_ROOT_ENV as PACKET_QUEUE_ENV

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", default=os.getenv(QUEUE_ROOT_ENV))
    parser.add_argument("--packet-queue-root", default=os.getenv(PACKET_QUEUE_ENV))
    parser.add_argument("--max-sources", type=int, default=32)
    args = parser.parse_args(argv)
    root = _root(args.queue_root)
    sources = scene_stage_pass(queue_root=root, packet_queue_root=args.packet_queue_root,
        max_sources=args.max_sources) if args.packet_queue_root else {"status": "source_queue_not_configured"}
    # A held job (its scene stage cannot move) reaches the website by name.
    holds = deliver_holds(queue_root=root)
    print(json.dumps({"sources": sources, "holds": holds}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
