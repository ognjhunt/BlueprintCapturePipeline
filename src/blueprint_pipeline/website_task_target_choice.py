"""Ask the site which object the robot moves, then resume the same job.

When the segmenter found instances of a manipulated task object but no spatial
anchor picks one, preparation used to fail the same way on every redelivery
until Pub/Sub gave up, and nobody asked the site. Instead the attempt retains
one typed question listing the instances it already found (opaque ids and
position labels derived from their masks; never pixels, because these masks
come from unscreened source frames) and parks the job in the existing ledger
with its exact handoff bytes. Nothing is spent while it waits.

The site answers on its job page. The existing drain tick reads that answer
through the existing signed WebApp control transport, refuses one bound to any
other question or candidate set, and replays the same handoff: retained
provider results are reused and the chosen instance is selected. "None of
these" ends the wait; the site sends an updated walkthrough of the same job.
No new queue, service, grant or approval step.
"""
from __future__ import annotations

import base64
import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

QUESTION_FILE = "website_task_target_choice.json"
ANSWER_FILE = "website_task_target_choice_answer.json"
POLL_FILE = "website_task_target_choice_poll.json"
QUESTION_SCHEMA = "website_task_target_choice_question.v1"
ANSWER_SCHEMA = "website_task_target_choice_answer.v1"
RESUME_SCHEMA = "website_task_target_choice_resume.v1"
#: The job ledger field that holds the parked handoff while the site chooses.
FIELD = "target_choice_resume"
BLOCKER = "website_task_target_choice_required"
OPERATION = "task-target-choice"
MAX_CANDIDATES = 8
_TARGET_ID = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
_CANDIDATE_ID = re.compile(r"c[0-9a-f]{16}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
# The whole failure is the choice, possibly under stage prefixes; never one of several blockers.
_ONLY_CHOICE = re.compile(rf"(?:[a-z_]+: )*{BLOCKER}:({_TARGET_ID})")
_RECORD_FIELDS = frozenset({"schema_version", "payload_base64", "source_payload_sha256",
                            "producer_delivery_key", "error", "question_digest"})
_SELECTORS = ("request_id", "scene_id", "capture_id", "completion_marker_generation",
              "producer_delivery_key", "source_payload_sha256", "task_context_digest")
_CHAIN_LIMIT = 16


class TaskTargetAmbiguous(ValueError):
    """The unchanged ambiguity refusal, carrying the instances the segmenter found."""

    def __init__(self, target_id: str, *, candidates: Sequence[Mapping[str, Any]] = (),
                 frames: Sequence[Mapping[str, Any]] | None = None) -> None:
        super().__init__(f"task_target_track_ambiguous:{target_id}")
        self.target_id = target_id
        self.candidates = [track for track in candidates
                           if isinstance(track, Mapping) and track.get("track_id") is not None]
        self.frames = list(frames or [])


def _chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    for _ in range(_CHAIN_LIMIT):
        if current is None or id(current) in seen:
            return
        seen.add(id(current))
        yield current
        current = current.__cause__ if current.__cause__ is not None else current.__context__


def _candidate_rows(tracks: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    rows = []
    for track in tracks:
        identity = {"track_id": str(track["track_id"]),
                    "observations_digest": canonical_digest({"observations": list(track.get("observations") or [])})}
        rows.append({**identity, "candidate_id": "c" + canonical_digest(identity)[7:23]})
    return sorted(rows, key=lambda row: row["track_id"])


def candidate_set_digest(target_id: str, tracks: Sequence[Mapping[str, Any]]) -> str:
    """Binds an answer to these exact instances of this target, mask for mask."""
    return canonical_digest({"target_id": target_id, "candidates": [
        {"track_id": row["track_id"], "observations_digest": row["observations_digest"]}
        for row in _candidate_rows(tracks)]})


def chosen_track(target_id: str, tracks: Sequence[Mapping[str, Any]],
                 choice: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """The site's chosen instance, only when the answer names exactly this candidate set."""
    if not isinstance(choice, Mapping) or choice.get("target_id") != target_id or not tracks:
        return None
    if choice.get("candidate_set_digest") != candidate_set_digest(target_id, tracks):
        return None
    by_candidate = {row["candidate_id"]: row["track_id"] for row in _candidate_rows(tracks)}
    track_id = by_candidate.get(choice.get("candidate_id"))
    return next((track for track in tracks if str(track["track_id"]) == track_id), None)


def _run_length(observation: Mapping[str, Any]) -> int:
    return sum(int(run.get("length") or 0) for run in observation.get("runs") or [])


def _centroid(observation: Mapping[str, Any]) -> tuple[float, float, float]:
    import numpy as np

    from .website_task_masks import decode_track_mask
    mask = decode_track_mask(observation)
    ys, xs = np.nonzero(mask)
    return float(xs.mean() / mask.shape[1]), float(ys.mean() / mask.shape[0]), float(len(xs) / mask.size)


def _where(x: float, y: float) -> str:
    side = "left side" if x < 1 / 3 else "middle" if x < 2 / 3 else "right side"
    height = "upper part" if y < 1 / 3 else "lower part" if y > 2 / 3 else "center"
    return f"{side}, {height} of the view"


def _semantic_label(target: Mapping[str, Any]) -> str:
    text = re.sub(r"[^A-Za-z0-9 ,.'()/-]+", " ", str(target.get("semantic_label") or ""))
    text = re.sub(r"\s+", " ", text).strip()[:60].strip()
    return (text[:1].upper() + text[1:]) if text else "Object"


def _candidates(semantic: str, tracks: Sequence[Mapping[str, Any]],
                frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Position labels from the masks the segmenter returned; no pixels leave the Pipeline."""
    times = {}
    for frame in frames:
        try:
            times[str(frame["frame_id"])] = max(0, round(float(frame["timestamp_seconds"]) * 1000))
        except (KeyError, TypeError, ValueError):
            continue
    rows = {row["track_id"]: row for row in _candidate_rows(tracks)}
    visible: dict[str, dict[str, Mapping[str, Any]]] = {}
    for track in tracks:
        for observation in track.get("observations") or []:
            if _run_length(observation) > 0:
                visible.setdefault(str(observation.get("source_frame_id")), {})[str(track["track_id"])] = observation
    if not visible:
        return []
    # One shared view where most instances are visible, so "left" means the same thing for each.
    reference = min(visible, key=lambda frame: (-len(visible[frame]),
                                                -sum(_run_length(row) for row in visible[frame].values()), frame))
    placed = []
    for track in tracks:
        track_id = str(track["track_id"])
        observation = visible[reference].get(track_id)
        frame = reference
        if observation is None:
            seen = [(frame_id, rows_by_track[track_id]) for frame_id, rows_by_track in visible.items()
                    if track_id in rows_by_track]
            if not seen:
                continue
            frame, observation = max(seen, key=lambda item: (_run_length(item[1]), item[0]))
        x, y, area = _centroid(observation)
        placed.append({"track_id": track_id, "x": x, "y": y, "area": area, "frame": frame,
                       "shared": frame == reference})
    # The most visible instances when there are more than can be asked about at once.
    shown = sorted(placed, key=lambda row: (-row["area"], row["track_id"]))[:MAX_CANDIDATES]
    shown.sort(key=lambda row: (not row["shared"], row["x"], row["y"], row["track_id"]))
    result = []
    for index, row in enumerate(shown, start=1):
        where = _where(row["x"], row["y"]) if row["shared"] else f"seen elsewhere in the video, {_where(row['x'], row['y'])}"
        label = (f"The {semantic.lower()} Blueprint found, {where}" if len(shown) == 1
                 else f"{semantic} {index}: {where}")
        result.append({"candidate_id": rows[row["track_id"]]["candidate_id"], "label": label[:120],
                       "reference_time_ms": times.get(row["frame"])})
    return result


def build_question(*, target: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]],
                   frames: Sequence[Mapping[str, Any]], scene_id: str, capture_id: str,
                   source_geometry: Mapping[str, Any]) -> dict[str, Any] | None:
    target_id = str(target["target_id"])
    if (re.fullmatch(_TARGET_ID, target_id) is None or target.get("task_effect") != "manipulated"
            or target.get("target_class") == "person" or not candidates):
        return None
    semantic = _semantic_label(target)
    shown = _candidates(semantic, candidates, frames)
    if not shown:
        return None
    question = {"schema_version": QUESTION_SCHEMA, "kind": "task_target_choice",
                "scene_id": scene_id, "capture_id": capture_id, "target_id": target_id,
                "semantic_label": semantic,
                "source_video_digest": (source_geometry.get("binding") or {}).get("source_video_digest"),
                "source_geometry_digest": source_geometry.get("digest"),
                "candidate_set_digest": candidate_set_digest(target_id, candidates), "candidates": shown}
    question["question_digest"] = canonical_digest(question, digest_field="question_digest")
    return question


def retain_question(pipeline_root: Path, *, exc: BaseException, plan: Mapping[str, Any],
                    source_geometry: Mapping[str, Any], scene_id: str, capture_id: str) -> str | None:
    """The typed blocker for an ambiguous manipulated target whose question is now retained, or None."""
    from .core.handoff_lease import guarded_handoff_write_json
    ambiguous = next((item for item in _chain(exc) if isinstance(item, TaskTargetAmbiguous)), None)
    if ambiguous is None:
        return None
    target = next((row for row in plan.get("targets") or []
                   if isinstance(row, Mapping) and row.get("target_id") == ambiguous.target_id), None)
    try:
        question = None if target is None else build_question(target=target, candidates=ambiguous.candidates,
            frames=ambiguous.frames, scene_id=scene_id, capture_id=capture_id, source_geometry=source_geometry)
    except Exception:  # noqa: BLE001 - an unreadable candidate keeps the original refusal
        question = None
    if question is None:
        return None
    guarded_handoff_write_json(Path(pipeline_root) / QUESTION_FILE, question)
    return f"{BLOCKER}:{question['target_id']}"


def _read(path: Path) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if type(value) is dict else None


def read_question(pipeline_root: Path) -> dict[str, Any] | None:
    try:
        question = _read(Path(pipeline_root) / QUESTION_FILE)
    except (OSError, ValueError):
        return None
    if (question is None or question.get("schema_version") != QUESTION_SCHEMA
            or question.get("question_digest") != canonical_digest(question, digest_field="question_digest")
            or question.get("capture_id") != Path(pipeline_root).parent.name
            or type(question.get("candidates")) is not list or not question["candidates"]):
        return None
    return question


def validate_answer(value: Any, *, question: Mapping[str, Any],
                    selectors: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Refuse an answer bound to another question, candidate set, source or digest."""
    fields = {"schema_version", *_SELECTORS, "question_digest", "target_id", "candidate_set_digest",
              "choice", "answered_at", "answer_digest"}
    choice = value.get("choice") if isinstance(value, Mapping) else None
    shown = {row.get("candidate_id") for row in question["candidates"]}
    if (type(value) is not dict or set(value) != fields or value["schema_version"] != ANSWER_SCHEMA
            or value["question_digest"] != question["question_digest"]
            or value["target_id"] != question["target_id"]
            or value["candidate_set_digest"] != question["candidate_set_digest"]
            or value["capture_id"] != question["capture_id"] or value["scene_id"] != question["scene_id"]
            or (selectors is not None and any(value[key] != selectors[key] for key in _SELECTORS))
            or type(choice) is not dict
            or not (choice == {"kind": "none_of_these"}
                    or (set(choice) == {"kind", "candidate_id"} and choice["kind"] == "candidate"
                        and choice["candidate_id"] in shown))
            or type(value["answered_at"]) is not str or len(value["answered_at"]) > 40
            or type(value["answer_digest"]) is not str or _DIGEST.fullmatch(value["answer_digest"]) is None
            or value["answer_digest"] != cross_runtime_canonical_digest(value, digest_field="answer_digest")):
        raise ValueError("website_task_target_choice_answer_invalid")
    return dict(value)


def read_answer(pipeline_root: Path, question: Mapping[str, Any]) -> dict[str, Any] | None:
    """The retained answer to exactly this question; an answer to any other question is not one."""
    try:
        value = _read(Path(pipeline_root) / ANSWER_FILE)
        return None if value is None else validate_answer(value, question=question)
    except (OSError, ValueError):
        return None


def load_target_choices(pipeline_root: Path) -> dict[str, dict[str, Any]]:
    """{target_id: choice} for the task masks: only a candidate the site chose for the current question."""
    question = read_question(pipeline_root)
    answer = None if question is None else read_answer(pipeline_root, question)
    if answer is None or answer["choice"]["kind"] != "candidate":
        return {}
    return {question["target_id"]: {"target_id": question["target_id"],
            "candidate_set_digest": question["candidate_set_digest"],
            "candidate_id": answer["choice"]["candidate_id"], "question_digest": question["question_digest"]}}


def required_choice(exc: BaseException) -> str | None:
    """The target this run stopped on for the site's choice alone, anywhere in its chain."""
    for item in _chain(exc):
        try:
            match = _ONLY_CHOICE.fullmatch(str(item))
        except Exception:  # noqa: BLE001 - an unprintable error is no typed question
            return None
        if match:
            return match.group(1)
    return None


def resume_record(listener, *, exc, payload, payload_digest, producer_delivery_key,
                  capture_root: Path) -> dict | None:
    """The record that parks this exact handoff until the site answers, or None.

    Runs inside the listener's failure path, so it never raises: the original
    failure is recorded either way.
    """
    from .website_assessment_resume import _payload_bytes
    try:
        target_id = required_choice(exc)
        question = read_question(Path(capture_root) / "pipeline")
        if (producer_delivery_key is None or target_id is None or question is None
                or question["target_id"] != target_id):
            return None
        raw = _payload_bytes(listener, payload, payload_digest)
    except Exception:  # noqa: BLE001 - an unparked failure falls back to Pub/Sub's own retries
        return None
    return {"schema_version": RESUME_SCHEMA, "payload_base64": base64.b64encode(raw).decode(),
            "source_payload_sha256": payload_digest, "producer_delivery_key": producer_delivery_key,
            "error": str(exc), "question_digest": question["question_digest"]}


def armed(ledger: Mapping[str, Any]) -> bool:
    """Whether this ledger's current attempt waits only on the site's choice."""
    record = ledger.get(FIELD)
    return (ledger.get("status") == "failed_retryable" and type(record) is dict
            and record.get("schema_version") == RESUME_SCHEMA and set(record) == _RECORD_FIELDS
            and record.get("error") == ledger.get("last_error"))


def claim_matches(ledger, expected, *, payload_sha256, producer_delivery_key) -> bool:
    """Inside the claim lock: the ledger still waits on exactly the record the drain replays."""
    record = expected.get("record")
    return (type(expected.get("revision")) is int and ledger.get("revision") == expected["revision"]
            and armed(ledger) and type(record) is dict and ledger.get(FIELD) == record
            and record.get("source_payload_sha256") == payload_sha256
            and record.get("producer_delivery_key") == producer_delivery_key)


def public_question(capture_root: Path, ledger: Mapping[str, Any]) -> dict[str, Any] | None:
    """The question this ledger is parked on, for as long as it is parked.

    It stays after the site answered: the WebApp shows its own stored answer
    for it (resuming, or the same-job walkthrough update for "none of these").
    """
    if not armed(ledger):
        return None
    question = read_question(Path(capture_root) / "pipeline")
    if (question is None or question["question_digest"] != ledger[FIELD]["question_digest"]
            or question.get("scene_id") != ledger.get("scene_id")):
        return None
    return {"kind": "task_target_choice", **{key: question[key] for key in (
        "question_digest", "target_id", "semantic_label", "candidate_set_digest")},
        "candidates": [{key: row[key] for key in ("candidate_id", "label", "reference_time_ms")}
                       for row in question["candidates"]]}


def _poll_due(pipeline: Path, question_digest: str, now: float) -> bool:
    try:
        value = _read(pipeline / POLL_FILE) or {}
    except (OSError, ValueError):
        return True
    return value.get("question_digest") != question_digest or float(value.get("next_poll_at_epoch") or 0) <= now


def _defer(pipeline: Path, question_digest: str, now: float) -> None:
    try:
        value = _read(pipeline / POLL_FILE) or {}
    except (OSError, ValueError):
        value = {}
    failures = int(value.get("failures") or 0) + 1 if value.get("question_digest") == question_digest else 1
    write_json(pipeline / POLL_FILE, {"schema_version": "website_task_target_choice_poll.v1",
               "question_digest": question_digest, "failures": failures,
               "next_poll_at_epoch": now + min(1800, 60 * 2 ** min(failures - 1, 5))})


def fetch_answer(root: Path, ledger: Mapping[str, Any], question: Mapping[str, Any], *,
                 now: float) -> dict[str, Any] | None:
    """Ask the WebApp for the site's answer; retain it only when it answers this exact question."""
    from .website_preparation_status import DELIVERY_FILE, validate_selectors
    from .website_task_context import website_webapp_request
    pipeline = root / "pipeline"
    delivery = _read(root / DELIVERY_FILE) or {}
    selectors = validate_selectors(delivery.get("selectors"))
    if (selectors["capture_id"] != root.name or selectors["scene_id"] != question["scene_id"]
            or selectors["producer_delivery_key"] != ledger.get("producer_delivery_key")
            or selectors["source_payload_sha256"] != ledger.get("source_payload_sha256")):
        raise ValueError("website_task_target_choice_selectors_changed")
    try:
        answer = validate_answer(website_webapp_request(capture_id=selectors["capture_id"], operation=OPERATION,
            payload={**selectors, "question_digest": question["question_digest"]}),
            question=question, selectors=selectors)
    except Exception:  # noqa: BLE001 - unanswered, unreachable or stale: ask again later
        _defer(pipeline, question["question_digest"], now)
        return None
    # Only reached while no answer to this question is retained; one to an older question is replaced.
    write_json(pipeline / ANSWER_FILE, answer)
    return answer


def reconcile_waiting_target_choices(listener, *, storage_root: Path, process_args, limit=1,
                                     poll_limit=4, now: float | None = None) -> dict[str, int]:
    """Resume each parked capture the site has answered; at most ``limit`` replays per tick."""
    from .task_evaluation_scene_retirement_generations import capture_birth_source_projection
    from .website_assessment_resume import MAX_BYTES, _payload_bytes, _require
    from .website_preparation_status import _root, read_preparation_status
    current = time.time() if now is None else now
    counts = {"resumed": 0, "waiting": 0, "declined": 0}
    polled = 0
    roots = sorted(path.parent for path in Path(storage_root).glob(
        f"*/scenes/site-*/captures/*/{listener.JOB_LEDGER_FILENAME}")
        if path.parent.name.startswith(("walkthrough-", "supplement-")))
    for root in roots:
        if counts["resumed"] >= max(0, min(limit, 50)):
            break
        try:
            _root(root)
            _require(not (root / listener.JOB_LEDGER_FILENAME).is_symlink())
            before = listener._read_job_ledger(root)
            if not armed(before):
                continue
            record, pipeline = before[FIELD], root / "pipeline"
            question = read_question(pipeline)
            _require(question is not None and question["question_digest"] == record["question_digest"])
            answer = read_answer(pipeline, question)
            if answer is None:
                if polled >= max(0, poll_limit) or not _poll_due(pipeline, question["question_digest"], current):
                    counts["waiting"] += 1
                    continue
                polled += 1
                answer = fetch_answer(root, before, question, now=current)
                if answer is None:
                    counts["waiting"] += 1
                    continue
            if answer["choice"]["kind"] != "candidate":
                # The site will send an updated walkthrough of the same job instead.
                counts["declined"] += 1
                continue
            _require(len(record["payload_base64"]) <= 4 * ((MAX_BYTES + 2) // 3))
            raw = _payload_bytes(listener, base64.b64decode(record["payload_base64"], validate=True),
                                 record["source_payload_sha256"])
            handoff = listener.parse_handoff_payload(raw)
            _require(root == listener._handoff_capture_root(handoff, storage_root=storage_root)
                     and before.get("source_payload_sha256") == record["source_payload_sha256"]
                     and before.get("producer_delivery_key") == record["producer_delivery_key"])
            _require(handoff.source_finalize is not None and handoff.source_membership_selector is not None)
            context = _read(pipeline / "website_task_context.json") or {}
            # The status reader re-checks the active birth and membership, fresh
            # owner, local consent, current context and unchanged ledger.
            status = read_preparation_status(capture_root=root, selectors={
                "request_id": handoff.scene_id.removeprefix("site-"), "scene_id": handoff.scene_id,
                "capture_id": handoff.capture_id,
                "completion_marker_generation": handoff.source_finalize["generation"],
                "producer_delivery_key": record["producer_delivery_key"],
                "source_payload_sha256": record["source_payload_sha256"],
                "task_context_digest": context.get("context_digest")})
            _require(status["state"] == "awaiting_inputs" and status["revision"] == before["revision"])
            birth = capture_birth_source_projection(root, expected_purpose="scene_preparation")
            _require(birth is not None and birth["source_membership_selector"] == handoff.source_membership_selector)
            _require(listener._read_job_ledger(root) == before)
            # The same lease claim consumes the record; its failure row keeps the bytes.
            result = listener.process_handoff_payload(raw, storage_root=storage_root,
                payload_digest=record["source_payload_sha256"],
                expected_target_choice_resume={"revision": before["revision"], "record": record},
                **({"expected_preparation_purpose": "scene_preparation"}
                   if birth.get("capture_observation_purpose") == "scene_preparation" else {}), **process_args)
            if result.get("status") in {"processed", "retryable_blocked", "awaiting_site_target_choice"}:
                counts["resumed"] += 1
        except Exception:  # noqa: BLE001 - one unreadable capture never stops the others
            listener.logger.debug("pubsub_handoff.target_choice_resume_pending")
    return counts
