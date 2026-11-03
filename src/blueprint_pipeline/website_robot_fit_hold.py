"""Run a website job held for robot fit again when the robots it was held against change.

Beta journey step: L3 website scene preparation. A job the robot fit precheck held before any spend
(``website_task_context_contracts.retains_robot_fit_hold``) is a completed stage: a redelivery never
reruns it, so it is never burnt through Pub/Sub's delivery limit, and the site reads it as waiting
(``awaiting_inputs``), not failing. What held it is the screening catalog and the operator's team
registry at the time, recorded beside the hold (``recheck_inputs``). When either changes, a candidate
enabled or a team body registered, the existing drain tick replays the job's own handoff bytes, kept in
the job ledger at completion as a correction of its target is (``website_task_target_choice``). The
website preparation stage then runs again from its retained outputs (its stage input binding includes
the changed inputs, ``preparation_binding_digest``), so the precheck reads the new catalog: the job goes
on to registration, or holds again against the new inputs. No new queue, publisher, credential or grant;
a held job whose inputs did not change is never rerun, and each change is replayed a bounded number of
times.
"""
from __future__ import annotations

import base64
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest

SCHEMA = "website_robot_fit_hold_replay.v1"
#: Job ledger field holding the exact handoff bytes a recheck replays.
FIELD = "robot_fit_hold_replay"
#: Attempt-history mark of a run a recheck started (the inputs it reran against).
RECHECK_MARK = "robot_fit_recheck"
STATE_FILE = "robot_fit_recheck.json"
#: Replays of one change of inputs, at most, before it waits for the next change.
MAX_RECHECKS = 3
_RECORD_FIELDS = frozenset({"schema_version", "payload_base64", "source_payload_sha256", "producer_delivery_key",
                            "recheck_inputs"})


def team_registry_state() -> tuple[dict[str, Any] | None, str | None]:
    """The operator's team registry and what it is as a recheck input: its digest, None, or ``unreadable``."""
    from .team_candidate_quick10_registry import load_registry

    try:
        registry = load_registry()
    except (OSError, ValueError, KeyError, TypeError):
        return None, "unreadable"
    return registry, None if registry is None else str(registry.get("registry_digest"))


def recheck_inputs() -> dict[str, Any]:
    """What a robot fit hold was decided against: the screening catalog and the team registry, as of now."""
    from .beta_policy_candidate_catalog import load_default_catalog

    try:
        catalog = load_default_catalog()["catalog_digest"]
    except (OSError, ValueError, KeyError, TypeError):
        catalog = "unreadable"
    return {"catalog_digest": catalog, "team_registry_digest": team_registry_state()[1]}


def _handoff(pipeline_root: Path) -> dict[str, Any] | None:
    from .website_preparation_outcome import _load

    path = Path(pipeline_root) / "website_scene_preparation" / "handoff.json"
    try:
        return _load(path, Path(pipeline_root).parent.absolute(), "digest") if path.exists() else None
    except (OSError, ValueError):
        return None


def held_inputs(pipeline_root: Path) -> dict[str, Any] | None:
    """The inputs the capture's retained robot fit hold was decided against, or None when it holds none."""
    from .website_task_context_contracts import retains_robot_fit_hold

    root = Path(pipeline_root).parent
    handoff = _handoff(pipeline_root)
    if handoff is None or not retains_robot_fit_hold(handoff, {"scene_id": root.parent.parent.name,
                                                               "capture_id": root.name}):
        return None
    inputs = handoff["robot_fit_precheck"].get("recheck_inputs")
    return dict(inputs) if isinstance(inputs, Mapping) else {}


def pending_recheck(pipeline_root: Path) -> dict[str, Any] | None:
    """The current inputs, when the retained hold was decided against others; else None."""
    held = held_inputs(pipeline_root)
    if held is None:
        return None
    current = recheck_inputs()
    return None if held == current else current


def preparation_binding_digest(pipeline_root: Path, correction_digest: str | None) -> str | None:
    """The website preparation stage's input binding: the site's corrections and, while a robot fit hold
    was decided against other inputs than now, those inputs. Unchanged (the corrections') otherwise, so
    no other stage reruns."""
    pending = pending_recheck(pipeline_root)
    if pending is None:
        return correction_digest
    return canonical_digest({"corrections_binding": correction_digest, "robot_fit_recheck_inputs": pending})


def replay_record(listener, *, payload, payload_digest, producer_delivery_key, capture_root: Path) -> dict | None:
    """The exact handoff bytes a recheck replays, kept while the run's handoff holds for robot fit. Never raises."""
    from .website_assessment_resume import _payload_bytes

    try:
        held = held_inputs(Path(capture_root) / "pipeline")
        if producer_delivery_key is None or held is None:
            return None
        raw = _payload_bytes(listener, payload, payload_digest)
    except Exception:  # noqa: BLE001 - without the bytes the hold stands and is still shown
        return None
    return {"schema_version": SCHEMA, "payload_base64": base64.b64encode(raw).decode(),
            "source_payload_sha256": payload_digest, "producer_delivery_key": producer_delivery_key,
            "recheck_inputs": held}


def claim_matches(ledger, expected, *, payload_sha256, producer_delivery_key) -> bool:
    """Inside the claim lock: still the completed, held ledger the drain read, for the same delivery."""
    record = expected.get("record")
    return (type(expected.get("revision")) is int and ledger.get("revision") == expected["revision"]
            and ledger.get("status") == "completed" and type(record) is dict and ledger.get(FIELD) == record
            and record.get("source_payload_sha256") == payload_sha256
            and record.get("producer_delivery_key") == producer_delivery_key)


def _state(pipeline: Path) -> dict[str, Any]:
    from .website_preparation_status import _read

    try:
        return _read(pipeline / "website_scene_preparation" / STATE_FILE)
    except (OSError, ValueError):
        return {}


def _due(pipeline: Path, digest: str, now: float) -> bool:
    value = _state(pipeline)
    if value.get("inputs_digest") != digest:
        return True
    return int(value.get("replays") or 0) < MAX_RECHECKS and float(value.get("next_at_epoch") or 0) <= now


def _note(pipeline: Path, digest: str, now: float) -> None:
    previous = _state(pipeline)
    replays = int(previous.get("replays") or 0) + 1 if previous.get("inputs_digest") == digest else 1
    # Durable before the run, so a crash mid-run cannot turn into an unbounded loop.
    write_json(pipeline / "website_scene_preparation" / STATE_FILE, {
        "schema_version": "website_robot_fit_recheck_state.v1", "inputs_digest": digest, "replays": replays,
        "next_at_epoch": now + min(1800, 60 * 2 ** min(replays - 1, 5))})


def reconcile_robot_fit_holds(listener, *, storage_root: Path, process_args, limit=1,
                              now: float | None = None) -> dict[str, int]:
    """Rerun each held job whose catalog or team registry changed since it was held; at most ``limit`` per tick."""
    from .task_evaluation_scene_retirement_generations import capture_birth_source_projection
    from .website_assessment_resume import MAX_BYTES, _payload_bytes, _require
    from .website_preparation_status import DELIVERY_FILE, _read, _root, read_preparation_status, validate_selectors

    current = time.time() if now is None else now
    counts = {"rechecked": 0, "held": 0, "waiting": 0}
    inputs = None
    roots = sorted(path.parent for path in Path(storage_root).glob(
        f"*/scenes/site-*/captures/*/{listener.JOB_LEDGER_FILENAME}")
        if path.parent.name.startswith(("walkthrough-", "supplement-")))
    for root in roots:
        if counts["rechecked"] >= max(0, min(limit, 50)):
            break
        try:
            _root(root)
            _require(not (root / listener.JOB_LEDGER_FILENAME).is_symlink())
            before = listener._read_job_ledger(root)
            record = before.get(FIELD)
            if before.get("status") != "completed" or type(record) is not dict:
                continue
            pipeline = root / "pipeline"
            held = held_inputs(pipeline)
            if held is None:
                continue
            inputs = recheck_inputs() if inputs is None else inputs
            if held == inputs:
                counts["held"] += 1
                continue
            digest = canonical_digest({"recheck_inputs": inputs})
            if (set(record) != _RECORD_FIELDS or record.get("schema_version") != SCHEMA
                    or record.get("recheck_inputs") != held or not _due(pipeline, digest, current)):
                counts["waiting"] += 1
                continue
            _require(type(record["payload_base64"]) is str and len(record["payload_base64"]) <= 4 * ((MAX_BYTES + 2) // 3))
            raw = _payload_bytes(listener, base64.b64decode(record["payload_base64"], validate=True),
                                 record["source_payload_sha256"])
            handoff = listener.parse_handoff_payload(raw)
            _require(root == listener._handoff_capture_root(handoff, storage_root=storage_root)
                     and before.get("source_payload_sha256") == record["source_payload_sha256"]
                     and before.get("producer_delivery_key") == record["producer_delivery_key"])
            _require(handoff.source_finalize is not None and handoff.source_membership_selector is not None)
            # The status reader re-checks the active birth and membership, fresh owner, local consent,
            # current context and unchanged ledger; a withdrawn or ended job is never rerun.
            selectors = validate_selectors((_read(root / DELIVERY_FILE) or {}).get("selectors"))
            status = read_preparation_status(capture_root=root, selectors=selectors)
            _require(status["revision"] == before["revision"] and status["state"] != "authority_ended")
            birth = capture_birth_source_projection(root, expected_purpose="scene_preparation")
            _require(birth is not None and birth["source_membership_selector"] == handoff.source_membership_selector)
            _require(listener._read_job_ledger(root) == before)
            _note(pipeline, digest, current)
            counts["rechecked"] += 1
            listener.process_handoff_payload(raw, storage_root=storage_root,
                payload_digest=record["source_payload_sha256"],
                expected_robot_fit_recheck={"revision": before["revision"], "record": record, "inputs": inputs},
                **({"expected_preparation_purpose": "scene_preparation"}
                   if birth.get("capture_observation_purpose") == "scene_preparation" else {}), **process_args)
        except Exception:  # noqa: BLE001 - one capture never stops the others; an owed recheck stays owed
            listener.logger.debug("pubsub_handoff.robot_fit_recheck_pending")
    return counts


__all__ = ["FIELD", "RECHECK_MARK", "SCHEMA", "claim_matches", "held_inputs", "pending_recheck",
           "preparation_binding_digest", "recheck_inputs", "reconcile_robot_fit_holds", "replay_record",
           "team_registry_state"]
