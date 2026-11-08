"""Run a website job's completed preparation again, once, when a fact it was held on (or lacks) arrives.

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
  A job whose re-derived scene half's intake the website refused as a changed request
  (``idempotency_conflict``, before the Pipeline could re-issue it) lacks ``pipeline:navigation_intake_reissue``
  (``_reissue_refused``); its next replay sends the intake re-issued.

Modelled on the robot fit recheck (``website_robot_fit_hold``): the job ledger keeps the run's exact
handoff bytes while it holds this way (``replay_record``); the existing drain tick replays them once per
arrived fact (``reconcile_held_handoffs``, its state written before the run, so a crash never loops); the
website preparation stage's input binding names that fact (``preparation_binding``), so only that stage
runs again from its retained outputs. No new queue, publisher, credential or grant. The confirmation is
always the website's own reading; nothing here sets or assumes it.

A job held before the ledger kept its bytes is replayed by the operator instead
(``resume_held_preparation``, the door's ``--resume-preparation``), on the original payload the door
verified, through the same claim, once per fact.

One more fact makes any completed preparation stale, held or handed off:

- **An Atlas result published after the preparation** (``superseded_atlas_result``). The handoff's base
  scene names the Atlas result it was built from (``base_scene.atlas_result_digest``); once the job's
  Atlas lane is final again (``ready``) on another sealed result, such as a splat rebuild
  (``atlas_splat_rebuild``), the preparation runs once more on it, keyed by the new result's digest. Every
  run reads the current result, so the levelled scene, the base scene, the navigation scene half and the
  rest of the handoff are derived from it again, written beside the old ones (``website_atlas_scene_source``,
  ``website_scene_handoff.retain_superseded_records``). It adds no spend: the stage runs as any run
  does, under the job's existing admission and allowances (a navigation job's run is CPU only). While
  the Atlas job is processing (a rebuild in flight) or when its result is the one the base scene names
  (a stopped rebuild hands the parent back), there is no fact. The new scene half's intake goes out in
  the same run, re-issued under its own submission (``website_navigation_intake.reissue_parent``).

  Only the operator replays it, and the drain keeps no bytes for it. Every republish of a ready job is
  an operator reopening it (``atlas_preparation_job_control.retry_preparation_job``, the door's
  ``selected-handoff --atlas-retry``), and the door's status then shows the completed preparation's stale
  base scene (``native_preparation.atlas_result_superseded``). A handed-off job's ledger keeps no bytes
  (``replay_record``), and keeping them for every completed job would retain each payload indefinitely
  for an event only an operator causes. A held job the drain replays for its own fact reads the current
  result like any run.
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
#: What a navigation job lacked when the website refused its re-derived scene half's intake as a changed
#: request (``idempotency_conflict``): a re-issued intake (``_reissue_refused``).
REISSUE_MISSING = "pipeline:navigation_intake_reissue"
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


def _reissue_refused(handoff: dict[str, Any]) -> bool:
    """Whether the website refused a re-derived scene half's intake as a changed request of the capture's.

    A release before the re-issued intake (``website_preparation_revision_identity.reissue``) sent a
    re-derived record's intake under the capture's own submission, which the website already held for an
    earlier record, and the job held by the website's ``idempotency_conflict``. That is a Blueprint
    integration gap: the Pipeline could not re-issue the intake.
    """
    gap = handoff.get("native_task_preparation_gap") or {}
    inputs = handoff.get("navigation_scene_inputs") or {}
    if (gap.get("code") != "idempotency_conflict"
            or (gap.get("navigation_intake") or {}).get("reason") != "idempotency_conflict"
            or inputs.get("status") != "written" or not isinstance(inputs.get("path"), str)):
        return False
    from .website_navigation_intake import OUTBOX_FILENAME

    own = Path(inputs["path"]).parent
    return any(path.parent != own and path.is_file() and not path.is_symlink()
               for path in own.parent.glob(f"*/{OUTBOX_FILENAME}"))


def _pipeline_missing(handoff: dict[str, Any]) -> list[str]:
    """The Blueprint integration pieces a hold names as missing (``pipeline:*``)."""
    gap = handoff.get("native_task_preparation_gap") or {}
    missing = {item for item in gap.get("missing") or () if isinstance(item, str) and item.startswith("pipeline:")}
    if _reissue_refused(handoff):
        missing.add(REISSUE_MISSING)
    return sorted(missing)


def held_kind(pipeline_root: Path) -> str | None:
    """How the retained handoff waits on a later fact: ``consent``, ``intake_receipt``, ``integration_gap`` or None.

    ``integration_gap``: the hold names a missing Blueprint piece (``pipeline:*``), which a new release
    can build, so the job is re-evaluated once per release.
    """
    from .website_task_context_contracts import holds_native_task_preparation
    from .website_task_context_lineage import binding_contexts

    root = Path(pipeline_root)
    handoff, context = _handoff(root), _retained_context(root)
    # A hold decided before the brief was confirmed stays bound to the retained context's verified parent
    # until a run rewrites it (website_task_context_lineage.binding_contexts).
    if handoff is None or context is None or not any(
            holds_native_task_preparation(handoff, bound) for bound in binding_contexts(root, context)):
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


def superseded_atlas_result(pipeline_root: Path) -> dict[str, Any] | None:
    """The Atlas result that replaced the one the retained handoff's base scene was built from; None otherwise.

    ``atlas_result_superseded``: the handoff names its base scene, that base scene names the sealed Atlas
    result it came from, and the job's Atlas lane is final (``ready``, no recheck pending) on another sealed
    result. Held or handed off alike. Read-only and never raises: anything unreadable is no fact.
    """
    import json
    import re

    from .website_atlas_preview_store import JOB_FILE, read_job

    root = Path(pipeline_root)
    digest = re.compile(r"sha256:[0-9a-f]{64}")
    handoff = _handoff(root)
    named = (handoff or {}).get("base_scene_path")
    if not isinstance(named, str):
        return None
    try:
        path, preparation = Path(named).absolute(), (root / "website_scene_preparation").absolute()
        if (".." in path.parts or not path.is_relative_to(preparation) or not path.is_file()
                or any(item.is_symlink() for item in (path, *path.parents) if item.is_relative_to(preparation))):
            return None
        built = json.loads(path.read_text()).get("atlas_result_digest")
        job = read_job(root / "atlas" / JOB_FILE)
        result = json.loads((root / "atlas" / "result.json").read_text())
        current = result.get("digest")
    except (OSError, ValueError, AttributeError):
        return None
    if (not isinstance(built, str) or not digest.fullmatch(built)
            or job.get("state") != "ready" or job.get("retry_at_epoch")
            or not isinstance(current, str) or not digest.fullmatch(current) or current == built
            or current != job.get("result_digest") or current != canonical_digest(result, digest_field="digest")):
        return None
    fact = {"kind": "atlas_result_superseded", "result_digest": current, "superseded_result_digest": built}
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


#: An operator's reviewed replay of a held ledger that keeps no bytes of its own (it predates them).
OPERATOR_SCHEMA = "website_held_handoff_operator_replay.v1"


def claim_matches(ledger, expected, *, payload_sha256, producer_delivery_key) -> bool:
    """Inside the claim lock: still the held, completed ledger the drain (or operator) read, for the same delivery.

    The operator form names the exact reviewed ledger (its digest), the selected original payload the door
    verified against the producer delivery, and the fact; the drain form names the ledger's own kept bytes.
    """
    if expected.get("review_schema") == OPERATOR_SCHEMA:
        import re

        digest = expected.get("ledger_digest")
        return (set(expected) == {"review_schema", "ledger_digest", "selected_payload_sha256", "fact"}
                and ledger.get("status") == "completed"
                and type(digest) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is not None
                and canonical_digest(dict(ledger)) == digest
                and type(payload_sha256) is str and payload_sha256 == expected["selected_payload_sha256"]
                and producer_delivery_key is not None and producer_delivery_key == ledger.get("producer_delivery_key")
                and isinstance(expected.get("fact"), dict) and type(expected["fact"].get("fact_digest")) is str)
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


def resume_held_preparation(listener, *, capture_root: Path, storage_root: Path, handoff_payload: bytes,
                            ledger_digest: str, ledger_revision: int, provider: str) -> dict[str, Any]:
    """An operator's replay of a held, completed ledger that keeps no replay bytes (it predates the hook).

    ``handoff_payload`` is the original payload the door verified against the producer delivery; the claim
    re-checks the exact reviewed ledger. Eligible once the job's Atlas result replaced the one its base scene
    was built from (``superseded_atlas_result``, held or handed off), or while the retained handoff holds on
    a fact that has arrived (a confirmed successor of its context, a bound intake receipt, or a release for
    its integration gap), and once per fact. Everything else refuses by name; nothing is bought.
    """
    import hashlib

    pipeline = Path(capture_root) / "pipeline"
    # A republished Atlas result first: its run reads the held job's own fact again too.
    fact = superseded_atlas_result(pipeline) or arrived_fact(pipeline)
    if fact is None:
        raise ValueError("website_held_handoff_resume_not_eligible")
    spent = _state(pipeline)
    # As in the drain: a fact is spent once a run claimed the job (its ledger revision moved).
    if spent.get("fact_digest") == fact["fact_digest"] and spent.get("ledger_revision") != ledger_revision:
        raise ValueError("website_held_handoff_resume_already_replayed")
    # Durable before the run, so the same fact never runs twice (the drain reads it too).
    write_json(pipeline / "website_scene_preparation" / STATE_FILE, {
        "schema_version": "website_held_handoff_replay_state.v1", "fact": fact,
        "fact_digest": fact["fact_digest"], "handoff_digest": (_handoff(pipeline) or {}).get("digest"),
        "ledger_revision": ledger_revision, "replayed_at_epoch": time.time(), "operator": True})
    return listener.process_handoff_payload(handoff_payload, storage_root=Path(storage_root), provider=provider,
        expected_preparation_purpose="scene_preparation",
        expected_held_replay={"review_schema": OPERATOR_SCHEMA, "ledger_digest": ledger_digest,
                              "selected_payload_sha256": hashlib.sha256(handoff_payload).hexdigest(), "fact": fact})


__all__ = ["CONSENT_HOLDS", "FIELD", "MARK", "OPERATOR_SCHEMA", "SCHEMA", "arrived_fact", "claim_matches",
           "held_kind", "preparation_binding", "reconcile_held_handoffs", "replay_record", "resume_held_preparation",
           "superseded_atlas_result"]
