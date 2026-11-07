"""Run a held website job's preparation again, once, when the fact it was held on arrives.

Beta journey step: L3 website scene preparation. A held job is a completed stage: a redelivery never
reruns it. These holds wait on a fact the job's own preparation cannot produce, so nothing would ever
re-run them:

- **A brief confirmed after preparation.** The website refused the job's intake because the site had not
  confirmed its brief (``CONSENT_HOLDS``). Once the website's current task context is the retained one's
  confirmed successor (``website_task_context_lineage.confirmation_successor``: only the confirmation
  changed), the job runs again on its retained outputs. Confirming a brief is consent state, not task
  content, so every retained, paid artifact stays bound and nothing is bought again.
- **An intake receipt bound after the handoff.** The website forwards a navigation intake
  asynchronously, so the handoff's own receipt lookup can find none and the job holds on
  ``pipeline:navigation_source_identity``. Once the screening pass binds the receipt beside the record
  (``website_navigation_intake``), the job runs again and is handed off.
- **A Blueprint integration gap.** A hold naming any missing ``pipeline:*`` piece waits on Blueprint
  itself, which a new release can fix. Each release (``current_release_commit``) re-evaluates the job
  once, keyed by that commit and the pipeline pieces the hold names; the drain tick's limit bounds it.

Modelled on the robot fit recheck (``website_robot_fit_hold``): the job ledger keeps the run's exact
handoff bytes while it holds this way (``replay_record``); the existing drain tick replays them once per
arrived fact (``reconcile_held_handoffs``, its state written before the run, so a crash never loops); the
website preparation stage's input binding names that fact (``preparation_binding``), so only that stage
runs again from its retained outputs. No new queue, publisher, credential or grant. The confirmation is
always the website's own reading; nothing here sets or assumes it.
"""
from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest

SCHEMA = "website_held_handoff_replay.v1"
#: Job ledger field holding the exact handoff bytes a replay runs again.
FIELD = "held_handoff_replay"
#: Attempt-history mark of a run a replay started (the fact it ran again for).
MARK = "held_handoff_replay"
STATE_FILE = "held_handoff_replay.json"
#: Holds whose reason is the site's unconfirmed brief (the website's refusal of the job's intake).
CONSENT_HOLDS = frozenset({"website_task_context_not_confirmed"})
SOURCE_IDENTITY_MISSING = "pipeline:navigation_source_identity"
_RECORD_FIELDS = frozenset({"schema_version", "payload_base64", "source_payload_sha256", "producer_delivery_key"})


def _retained_context(pipeline_root: Path) -> dict[str, Any] | None:
    import json

    try:
        value = json.loads((Path(pipeline_root) / "website_task_context.json").read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _handoff(pipeline_root: Path) -> dict[str, Any] | None:
    path = Path(pipeline_root) / "website_scene_preparation" / "handoff.json"
    try:
        from .website_preparation_outcome import _load

        return _load(path, Path(pipeline_root).parent.absolute(), "digest") if path.exists() else None
    except (OSError, ValueError):
        return None


def _pipeline_missing(handoff: dict[str, Any]) -> list[str]:
    """The Blueprint integration pieces a hold names as missing (``pipeline:*``)."""
    gap = handoff.get("native_task_preparation_gap") or {}
    return sorted({item for item in gap.get("missing") or () if isinstance(item, str) and item.startswith("pipeline:")})


def held_kind(pipeline_root: Path) -> str | None:
    """How the retained handoff waits on a later fact: ``consent``, ``intake_receipt``, ``integration_gap`` or None.

    ``integration_gap``: the hold names a missing Blueprint piece (``pipeline:*``), which a new release
    can build, so the job is re-evaluated once per release.
    """
    from .website_task_context_contracts import holds_native_task_preparation

    root = Path(pipeline_root)
    handoff, context = _handoff(root), _retained_context(root)
    if handoff is None or context is None or not holds_native_task_preparation(handoff, context):
        return None
    blockers = handoff.get("blockers") or []
    if len(blockers) == 1 and blockers[0] in CONSENT_HOLDS and context.get("confirmed") is False:
        return "consent"
    gap = handoff.get("native_task_preparation_gap") or {}
    inputs = handoff.get("navigation_scene_inputs") or {}
    if (SOURCE_IDENTITY_MISSING in (gap.get("missing") or []) and inputs.get("status") == "written"
            and (handoff.get("navigation_intake") or {}).get("status") == "forward_pending"
            and isinstance(inputs.get("path"), str)):
        return "intake_receipt"
    return "integration_gap" if _pipeline_missing(handoff) else None


def arrived_fact(pipeline_root: Path, *, load_current_context=None) -> dict[str, Any] | None:
    """The fact the held job now has and did not when it was held; None while it has none.

    ``confirmed_successor``: the website's current task context, one signed read, is the retained
    context's confirmed successor. ``intake_receipt``: the record's intake receipt is bound and reads.
    ``release``: the running release (``current_release_commit``) with the pipeline pieces the hold names,
    for a hold on a Blueprint integration gap; each release re-evaluates it once.
    """
    root = Path(pipeline_root)
    kind = held_kind(root)
    fact = None
    if kind == "consent":
        from .website_task_context_lineage import confirmation_successor

        retained = _retained_context(root) or {}
        if load_current_context is None:
            from .website_task_context import load_current_website_task_context as load_current_context
        current = load_current_context(request_id=retained["request_id"], scene_id=retained["scene_id"],
                                       capture_id=retained["capture_id"], purpose=retained.get("purpose"))
        lineage = confirmation_successor(retained, current)
        if lineage is not None:
            fact = {"kind": "confirmed_successor", **{key: lineage[key] for key in (
                "confirmation_parent_context_digest", "context_digest", "confirmed_at")}}
    elif kind == "intake_receipt":
        from .website_navigation_intake import read_navigation_intake_receipt, receipt_exists

        path = (_handoff(root) or {})["navigation_scene_inputs"]["path"]
        receipt = read_navigation_intake_receipt(path) if receipt_exists(path) else None
        if receipt is not None:
            fact = {"kind": "intake_receipt", "receipt_digest": receipt["receipt_digest"]}
    if fact is None and kind in {"intake_receipt", "integration_gap"}:
        from .team_scene_packet_worker import current_release_commit

        release, missing = current_release_commit(), _pipeline_missing(_handoff(root) or {})
        if release and missing:
            fact = {"kind": "release", "release_commit": release, "missing_pipeline": missing}
    if fact is None:
        return None
    fact["fact_digest"] = canonical_digest(fact)
    return fact


def replay_record(listener, *, payload, payload_digest, producer_delivery_key, capture_root: Path) -> dict | None:
    """The exact handoff bytes a replay runs again, kept while the run's handoff waits this way. Never raises."""
    from .website_assessment_resume import _payload_bytes

    try:
        if producer_delivery_key is None or held_kind(Path(capture_root) / "pipeline") is None:
            return None
        raw = _payload_bytes(listener, payload, payload_digest)
    except Exception:  # noqa: BLE001 - without the bytes the hold stands and is still shown
        return None
    return {"schema_version": SCHEMA, "payload_base64": base64.b64encode(raw).decode(),
            "source_payload_sha256": payload_digest, "producer_delivery_key": producer_delivery_key}


def claim_matches(ledger, expected, *, payload_sha256, producer_delivery_key) -> bool:
    """Inside the claim lock: still the held, completed ledger the drain read, for the same delivery."""
    record = expected.get("record")
    return (type(expected.get("revision")) is int and ledger.get("revision") == expected["revision"]
            and ledger.get("status") == "completed"
            and type(record) is dict and ledger.get(FIELD) == record
            and record.get("source_payload_sha256") == payload_sha256
            and record.get("producer_delivery_key") == producer_delivery_key)


def _state(pipeline: Path) -> dict[str, Any]:
    import json

    try:
        value = json.loads((pipeline / "website_scene_preparation" / STATE_FILE).read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def preparation_binding(pipeline_root: Path, binding: str | None) -> str | None:
    """The website preparation stage's input binding, naming a replay's fact while the hold it reruns stands.

    Local only: the fact the drain recorded before the replay, while the retained handoff is still the
    one it was recorded against. Unchanged otherwise, so no other stage reruns.
    """
    pipeline = Path(pipeline_root)
    state = _state(pipeline)
    handoff = _handoff(pipeline)
    if not state.get("fact_digest") or handoff is None or state.get("handoff_digest") != handoff.get("digest"):
        return binding
    return canonical_digest({"binding": binding, "held_handoff_replay": state["fact_digest"]})


def reconcile_held_handoffs(listener, *, storage_root: Path, process_args, limit=1,
                            now: float | None = None, load_current_context=None) -> dict[str, int]:
    """Run each held job again whose awaited fact arrived; once per fact, at most ``limit`` per tick."""
    from .task_evaluation_scene_retirement_generations import capture_birth_source_projection
    from .website_assessment_resume import MAX_BYTES, _payload_bytes, _require
    from .website_preparation_status import DELIVERY_FILE, _read as read_json, _root, read_preparation_status, \
        validate_selectors

    current = time.time() if now is None else now
    counts = {"replayed": 0, "held": 0, "already_replayed": 0}
    roots = sorted(path.parent for path in Path(storage_root).glob(
        f"*/scenes/site-*/captures/*/{listener.JOB_LEDGER_FILENAME}")
        if path.parent.name.startswith(("walkthrough-", "supplement-")))
    for root in roots:
        if counts["replayed"] >= max(0, min(limit, 50)):
            break
        try:
            _root(root)
            _require(not (root / listener.JOB_LEDGER_FILENAME).is_symlink())
            before = listener._read_job_ledger(root)
            record = before.get(FIELD)
            if type(record) is not dict or before.get("status") != "completed":
                continue
            pipeline = root / "pipeline"
            fact = arrived_fact(pipeline, load_current_context=load_current_context)
            if fact is None:
                counts["held"] += 1
                continue
            spent = _state(pipeline)
            # Once per fact: a replay that held again waits for the next. A fact is spent only once a run
            # claimed the job (its ledger revision moved); a replay that returned or stopped before its claim
            # (a transient owner read, a crash) leaves the fact eligible.
            if (spent.get("fact_digest") == fact["fact_digest"]
                    and spent.get("ledger_revision") != before.get("revision")):
                counts["already_replayed"] += 1
                continue
            _require(set(record) == _RECORD_FIELDS and record.get("schema_version") == SCHEMA)
            _require(type(record["payload_base64"]) is str and len(record["payload_base64"]) <= 4 * ((MAX_BYTES + 2) // 3))
            raw = _payload_bytes(listener, base64.b64decode(record["payload_base64"], validate=True),
                                 record["source_payload_sha256"])
            handoff = listener.parse_handoff_payload(raw)
            _require(root == listener._handoff_capture_root(handoff, storage_root=storage_root)
                     and before.get("source_payload_sha256") == record["source_payload_sha256"]
                     and before.get("producer_delivery_key") == record["producer_delivery_key"])
            _require(handoff.source_finalize is not None and handoff.source_membership_selector is not None)
            # The status reader re-checks the active birth and membership, fresh owner, local consent,
            # current context (a confirmed successor included) and unchanged ledger; a withdrawn or ended
            # job is never run again.
            selectors = validate_selectors((read_json(root / DELIVERY_FILE) or {}).get("selectors"))
            status = read_preparation_status(capture_root=root, selectors=selectors)
            _require(status["revision"] == before["revision"] and status["state"] != "authority_ended")
            birth = capture_birth_source_projection(root, expected_purpose="scene_preparation")
            _require(birth is not None and birth["source_membership_selector"] == handoff.source_membership_selector)
            _require(listener._read_job_ledger(root) == before)
            # Durable before the run, so a crash mid-run cannot turn into a second replay of the same fact.
            write_json(pipeline / "website_scene_preparation" / STATE_FILE, {
                "schema_version": "website_held_handoff_replay_state.v1", "fact": fact,
                "fact_digest": fact["fact_digest"], "handoff_digest": (_handoff(pipeline) or {}).get("digest"),
                "ledger_revision": before.get("revision"), "replayed_at_epoch": current})
            listener.process_handoff_payload(raw, storage_root=storage_root,
                payload_digest=record["source_payload_sha256"],
                expected_held_replay={"revision": before["revision"], "record": record, "fact": fact},
                **({"expected_preparation_purpose": "scene_preparation"}
                   if birth.get("capture_observation_purpose") == "scene_preparation" else {}), **process_args)
            counts["replayed"] += 1
        except Exception:  # noqa: BLE001 - one capture never stops the others
            listener.logger.debug("pubsub_handoff.held_handoff_replay_pending")
    return counts


__all__ = ["CONSENT_HOLDS", "FIELD", "MARK", "SCHEMA", "arrived_fact", "claim_matches", "held_kind",
           "preparation_binding", "reconcile_held_handoffs", "replay_record"]
