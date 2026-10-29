"""ADP-040/day 28: resume website scene preparation once its Atlas job ends.

Scene preparation reads the scene the capture's own Atlas job reconstructs
(``website_atlas_scene_source``). That job runs on its own worker timer and can
outlast Pub/Sub's delivery limit: the capture would be dead-lettered, and the
silence sweep would end its preparation while the scene was still being built.
A run that stopped only on that wait keeps its exact handoff bytes in the
existing job ledger, as the assessment wait does, and the existing drain tick
replays them once the Atlas job is final. No new queue, publisher, credentials
or job; the replay re-validates the same current source and authority.
"""
from __future__ import annotations

import base64
from pathlib import Path

from .website_assessment_resume import MAX_BYTES, PENDING_CODE, SCHEMA as ASSESSMENT_SCHEMA, _payload_bytes, _require

#: What the capture orchestrator raises while the capture's Atlas job is still processing.
WAIT_ERROR = "website_scene_preparation: website_atlas_job_processing"
SCHEMA = "website_reconstruction_resume.v1"
FIELD = "reconstruction_resume"
_RECORD_FIELDS = frozenset({"schema_version", "payload_base64", "source_payload_sha256",
                            "producer_delivery_key", "error"})
_CHAIN_LIMIT = 16


def atlas_wait(exc: BaseException) -> bool:
    """Whether the Atlas wait alone stopped this run, anywhere in its exception chain."""
    seen: set[int] = set()
    current: BaseException | None = exc
    for _ in range(_CHAIN_LIMIT):
        if current is None or id(current) in seen:
            return False
        seen.add(id(current))
        try:
            if str(current) == WAIT_ERROR:
                return True
        except Exception:  # noqa: BLE001 - an unprintable error is no typed wait
            return False
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return False


def _explicit_wait(row) -> bool:
    if type(row) is not dict:
        return False
    if row.get("stage") == "website_assessment_preparation" and row.get("error") == PENDING_CODE:
        record = row.get("assessment_resume")
        return type(record) is dict and record.get("schema_version") == ASSESSMENT_SCHEMA
    record = row.get(FIELD)
    return type(record) is dict and record.get("schema_version") == SCHEMA and row.get("error") == record.get("error")


def safe_to_arm(attempt_count, previous_history, recovered_expired_lease, *, correction: bool = False) -> bool:
    """Only explicit waits came before: an unknown provider outcome never becomes fresh retry capacity.

    The site's correction of the object the robot moves starts a new run of the
    same job (``correction``, marked in its history row): the rows before it are
    settled by that run's own claim, and only the explicit waits since count.
    """
    if recovered_expired_lease or len(previous_history) != attempt_count - 1:
        return False
    if correction:
        return True
    from .website_task_target_choice import CORRECTION_MARK
    marks = [index for index, row in enumerate(previous_history) if type(row) is dict and row.get(CORRECTION_MARK)]
    since = previous_history[marks[-1]:] if marks else previous_history
    return all(_explicit_wait(row) for row in since)


def resume_record(listener, *, exc, payload, payload_digest, producer_delivery_key, capture_root: Path,
                  attempt_count, previous_history, recovered_expired_lease, correction: bool = False) -> dict | None:
    """The record that lets the drain replay this exact handoff once the Atlas job is final, or None.

    Runs inside the listener's failure path, so it never raises: the original
    failure is recorded either way.
    """
    from .website_atlas_preview_store import JOB_FILE, read_job
    try:
        if producer_delivery_key is None or not atlas_wait(exc) or not safe_to_arm(
                attempt_count, previous_history, recovered_expired_lease, correction=correction):
            return None
        # The orchestrator's word is checked against the job it waited on.
        if read_job(Path(capture_root) / "pipeline" / "atlas" / JOB_FILE).get("state") != "processing":
            return None
        raw = _payload_bytes(listener, payload, payload_digest)
    except Exception:  # noqa: BLE001 - an unarmed wait falls back to Pub/Sub's own retries
        return None
    return {"schema_version": SCHEMA, "payload_base64": base64.b64encode(raw).decode(),
            "source_payload_sha256": payload_digest, "producer_delivery_key": producer_delivery_key,
            "error": str(exc)}


def armed(ledger) -> bool:
    """Whether this ledger's current attempt waits only on its capture's Atlas job."""
    record = ledger.get(FIELD)
    return (ledger.get("status") == "failed_retryable" and type(record) is dict
            and record.get("schema_version") == SCHEMA and record.get("error") == ledger.get("last_error"))


def claim_matches(ledger, expected, *, payload_sha256, producer_delivery_key) -> bool:
    """Inside the claim lock: the ledger still waits on exactly the record the drain replays."""
    record = expected.get("record")
    return (type(expected.get("revision")) is int and ledger.get("revision") == expected["revision"]
            and armed(ledger) and type(record) is dict and ledger.get(FIELD) == record
            and record.get("source_payload_sha256") == payload_sha256
            and record.get("producer_delivery_key") == producer_delivery_key)


def _final(job) -> bool:
    """The worker's own notion of done: terminal and not waiting to re-check authority."""
    from .website_atlas_preview_store import TERMINAL
    return job.get("state") in TERMINAL and not job.get("retry_at_epoch")


def reconcile_waiting_reconstructions(listener, *, storage_root: Path, process_args, limit=1):
    """Replay each waiting capture's own handoff once its Atlas job is final; at most ``limit`` per tick."""
    from .task_evaluation_scene_retirement_generations import capture_birth_source_projection
    from .website_atlas_preview_store import JOB_FILE, read_job
    from .website_preparation_status import _read, _root, read_preparation_status
    counts = {"resumed": 0, "pending": 0}
    # Include same-job source revisions; admission below still rechecks each birth.
    jobs = sorted(path for path in Path(storage_root).glob(f"*/scenes/site-*/captures/*/pipeline/atlas/{JOB_FILE}")
                  if path.parents[2].name.startswith(("walkthrough-", "supplement-")))
    for job_path in jobs:
        if counts["resumed"] >= max(0, min(limit, 50)):
            break
        root = job_path.parents[2]
        try:
            _root(root)
            _require(not (root / listener.JOB_LEDGER_FILENAME).is_symlink())
            before = listener._read_job_ledger(root)
            if not armed(before):
                continue
            if not _final(read_job(job_path)):
                counts["pending"] += 1
                continue
            record = before[FIELD]
            _require(set(record) == _RECORD_FIELDS and type(record["payload_base64"]) is str
                     and len(record["payload_base64"]) <= 4 * ((MAX_BYTES + 2) // 3))
            raw = _payload_bytes(listener, base64.b64decode(record["payload_base64"], validate=True),
                                 record["source_payload_sha256"])
            handoff = listener.parse_handoff_payload(raw)
            _require(root == listener._handoff_capture_root(handoff, storage_root=storage_root)
                     and before.get("source_payload_sha256") == record["source_payload_sha256"]
                     and before.get("producer_delivery_key") == record["producer_delivery_key"])
            _require(handoff.source_finalize is not None and handoff.source_membership_selector is not None)
            request_id = handoff.scene_id.removeprefix("site-")
            context = _read(root / "pipeline" / "website_task_context.json")
            # The actual status reader re-checks the active birth and membership,
            # fresh owner, local consent, current context and unchanged ledger.
            status = read_preparation_status(capture_root=root, selectors={"request_id": request_id,
                "scene_id": handoff.scene_id, "capture_id": handoff.capture_id,
                "completion_marker_generation": handoff.source_finalize["generation"],
                "producer_delivery_key": record["producer_delivery_key"],
                "source_payload_sha256": record["source_payload_sha256"],
                "task_context_digest": context.get("context_digest")})
            _require(status["state"] == "failed_retryable" and status["revision"] == before["revision"])
            birth = capture_birth_source_projection(root, expected_purpose="scene_preparation")
            _require(birth is not None and birth["source_membership_selector"] == handoff.source_membership_selector)
            _require(listener._read_job_ledger(root) == before)
            # The same lease claim consumes the record; its failure row keeps the bytes.
            result = listener.process_handoff_payload(raw, storage_root=storage_root,
                payload_digest=record["source_payload_sha256"],
                expected_reconstruction_resume={"revision": before["revision"], "record": record},
                **({"expected_preparation_purpose": "scene_preparation"}
                   if birth.get("capture_observation_purpose") == "scene_preparation" else {}), **process_args)
            if result.get("status") in {"processed", "retryable_blocked"}:
                counts["resumed"] += 1
        except Exception:  # noqa: BLE001 - one unreadable capture never stops the others
            listener.logger.debug("pubsub_handoff.reconstruction_resume_pending")
    return counts
