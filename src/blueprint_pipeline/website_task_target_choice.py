"""Guess which object the robot moves, keep going, and let the site correct it.

Owner direction 2026-10-28: only privacy, rights and money stop a job. When the
segmenter found several instances of the object the robot moves and no spatial
anchor picks one, preparation makes the best supported guess (rule in
``best_supported_guess``), records it, and continues the same job. The guess is
published as one optional question on the job page: "We picked <label>. Is that
the object the robot should move?", with the other instances it already found.
Candidates are opaque ids and position labels derived from their masks; no
pixels leave the Pipeline, because these masks come from unscreened frames.

The existing drain tick reads the site's answer through the existing signed
WebApp control transport and refuses one bound to any other question, candidate
set or source. Confirming the guess records it; nothing reruns. Another
candidate reruns the same job from its retained stage outputs with that target,
inside the existing grant and budget, and the drain keeps owning that rerun
until it finishes or reaches a wait another drain owns. "None of these" retires
the guessed result at an honest evidence limit; further recording is optional. Choices are kept per target, so two
ambiguous objects never overwrite each other. No new queue, service or gate.
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

QUESTION_DIR = "website_task_target_choices"
INDEX_FILE = "current.json"
REPLAY_STATE_FILE = "replay.json"
QUESTION_SCHEMA = "website_task_target_choice_question.v1"
INDEX_SCHEMA = "website_task_target_choice_index.v1"
ANSWER_SCHEMA = "website_task_target_choice_answer.v1"
REPLAY_SCHEMA = "website_task_target_choice_replay.v1"
#: Job ledger field holding the exact handoff bytes a correction replays.
REPLAY_FIELD = "target_choice_replay"
#: Attempt-history mark of a run the site's correction started (its corrections digest).
CORRECTION_MARK = "target_choice_correction"
OPERATION = "task-target-choice"
MAX_CANDIDATES = 8
MAX_CORRECTION_REPLAYS = 6
CORRECTABLE_STATUSES = frozenset({"completed", "failed_retryable", "retryable_blocked"})
_TARGET_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REPLAY_FIELDS = frozenset({"schema_version", "payload_base64", "source_payload_sha256", "producer_delivery_key"})
_SELECTORS = ("request_id", "scene_id", "capture_id", "completion_marker_generation",
              "producer_delivery_key", "source_payload_sha256", "task_context_digest")


class TaskTargetAmbiguous(ValueError):
    """The unchanged ambiguity refusal, carrying the instances the segmenter found."""

    def __init__(self, target_id: str, *, candidates: Sequence[Mapping[str, Any]] = (),
                 frames: Sequence[Mapping[str, Any]] | None = None,
                 scores: Mapping[str, float] | None = None) -> None:
        super().__init__(f"task_target_track_ambiguous:{target_id}")
        self.target_id = target_id
        self.candidates = [track for track in candidates
                           if isinstance(track, Mapping) and track.get("track_id") is not None]
        self.frames = list(frames or [])
        self.scores = dict(scores or {})
        # The concept that produced these candidates, when it is not the target's first prompt.
        self.segmentation_prompt: str | None = None


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


def _area(observation: Mapping[str, Any]) -> float:
    try:
        pixels = int(observation.get("width") or 0) * int(observation.get("height") or 0)
    except (TypeError, ValueError):
        return 0.0
    return _run_length(observation) / pixels if pixels > 0 else 0.0


def best_supported_guess(*, target_id: str, candidates: Sequence[Mapping[str, Any]],
                         anchor_scores: Mapping[str, float]) -> dict[str, Any]:
    """The instance preparation continues with when no anchor decides.

    Only evidence the candidates already carry is used, in this order:
    1. the highest mean overlap with the video analysis's own anchor boxes
       (the selector's evidence, here merely not decisive);
    2. the most source frames in which the instance is visible (a non-empty mask);
    3. the largest mean visible mask area, as a fraction of the frame;
    4. the smallest candidate id (a digest of the track and its masks), so the
       same candidates always give the same guess on every replay.
    """
    if not candidates:
        raise TaskTargetAmbiguous(target_id)
    ids = {row["track_id"]: row["candidate_id"] for row in _candidate_rows(candidates)}

    def rank(track: Mapping[str, Any]) -> tuple[float, int, float, str]:
        visible = [row for row in track.get("observations") or [] if _run_length(row) > 0]
        area = sum(_area(row) for row in visible) / len(visible) if visible else 0.0
        return (-round(float(anchor_scores.get(str(track["track_id"]), 0.0)), 9), -len(visible),
                -round(area, 9), ids[str(track["track_id"])])
    return {**min(candidates, key=rank), "selection_basis": "best_supported_guess"}


def _centroid(observation: Mapping[str, Any]) -> tuple[float, float, float]:
    import numpy as np

    from .website_track_mask_codec import decode_track_mask
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


def _candidates(semantic: str, tracks: Sequence[Mapping[str, Any]], frames: Sequence[Mapping[str, Any]],
                keep: str) -> list[dict[str, Any]]:
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
    # The most visible instances when there are more than can be asked about at once; the guess is always shown.
    shown = sorted(placed, key=lambda row: (row["track_id"] != keep, -row["area"], row["track_id"]))[:MAX_CANDIDATES]
    if not any(row["track_id"] == keep for row in shown):
        return []
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
                   frames: Sequence[Mapping[str, Any]], guessed_track_id: str, scene_id: str, capture_id: str,
                   source_geometry: Mapping[str, Any]) -> dict[str, Any] | None:
    target_id = str(target["target_id"])
    if (_TARGET_ID.fullmatch(target_id) is None or target.get("task_effect") != "manipulated"
            or target.get("target_class") == "person" or not candidates):
        return None
    shown = _candidates(_semantic_label(target), candidates, frames, str(guessed_track_id))
    guessed = next((row["candidate_id"] for row in _candidate_rows(candidates)
                    if row["track_id"] == str(guessed_track_id)), None)
    if not shown or guessed is None:
        return None
    question = {"schema_version": QUESTION_SCHEMA, "kind": "task_target_choice",
                "scene_id": scene_id, "capture_id": capture_id, "target_id": target_id,
                "semantic_label": _semantic_label(target),
                "source_video_digest": (source_geometry.get("binding") or {}).get("source_video_digest"),
                "source_geometry_digest": source_geometry.get("digest"),
                "candidate_set_digest": candidate_set_digest(target_id, candidates),
                "guessed_candidate_id": guessed, "candidates": shown}
    question["question_digest"] = canonical_digest(question, digest_field="question_digest")
    return question


def _path(pipeline_root: Path, target_id: str, kind: str) -> Path:
    if _TARGET_ID.fullmatch(str(target_id)) is None:
        raise ValueError("website_task_target_choice_target_invalid")
    return Path(pipeline_root) / QUESTION_DIR / f"{target_id}.{kind}.json"


def _read(path: Path) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if type(value) is dict else None


def retain_questions(pipeline_root: Path, *, guesses: Sequence[Mapping[str, Any]],
                     source_geometry: Mapping[str, Any], scene_id: str, capture_id: str) -> list[str]:
    """Record the questions this task-mask run's guesses raise; the index names only these.

    Called after a completed task-mask run, so a target settled since (by the
    site's choice, or by new evidence) no longer has a current question.
    """
    from .core.handoff_lease import guarded_handoff_write_json
    questions: dict[str, dict[str, Any]] = {}
    for guess in guesses:
        try:
            question = build_question(target=guess["target"], candidates=guess["candidates"],
                frames=guess["frames"], guessed_track_id=guess["guessed_track_id"], scene_id=scene_id,
                capture_id=capture_id, source_geometry=source_geometry)
        except Exception:  # noqa: BLE001 - an unreadable candidate set is a guess without a question
            question = None
        if question is not None:
            questions[question["target_id"]] = question
    for target_id, question in questions.items():
        guarded_handoff_write_json(_path(pipeline_root, target_id, "question"), question)
    index = {"schema_version": INDEX_SCHEMA,
             "questions": {target_id: questions[target_id]["question_digest"] for target_id in sorted(questions)}}
    guarded_handoff_write_json(Path(pipeline_root) / QUESTION_DIR / INDEX_FILE, index)
    return list(index["questions"].values())


def read_question(pipeline_root: Path, target_id: str) -> dict[str, Any] | None:
    try:
        question = _read(_path(pipeline_root, target_id, "question"))
    except (OSError, ValueError):
        return None
    shown = {row.get("candidate_id") for row in (question or {}).get("candidates") or [] if isinstance(row, Mapping)}
    if (question is None or question.get("schema_version") != QUESTION_SCHEMA
            or question.get("question_digest") != canonical_digest(question, digest_field="question_digest")
            or question.get("target_id") != target_id
            or question.get("capture_id") != Path(pipeline_root).parent.name
            or question.get("guessed_candidate_id") not in shown):
        return None
    return question


def current_questions(pipeline_root: Path) -> list[dict[str, Any]]:
    """Questions raised by the latest completed task-mask run, in target order."""
    try:
        index = _read(Path(pipeline_root) / QUESTION_DIR / INDEX_FILE) or {}
    except (OSError, ValueError):
        return []
    rows = index.get("questions") if index.get("schema_version") == INDEX_SCHEMA else None
    result = []
    for target_id, digest in sorted(rows.items()) if isinstance(rows, dict) else []:
        question = read_question(pipeline_root, target_id) if type(target_id) is str else None
        if question is not None and question["question_digest"] == digest:
            result.append(question)
    return result


def _all_questions(pipeline_root: Path) -> list[dict[str, Any]]:
    folder = Path(pipeline_root) / QUESTION_DIR
    names = sorted(path.name.removesuffix(".question.json") for path in folder.glob("*.question.json")) \
        if folder.is_dir() and not folder.is_symlink() else []
    return [question for name in names if (question := read_question(pipeline_root, name)) is not None]


def validate_answer(value: Any, *, question: Mapping[str, Any],
                    selectors: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Refuse an answer bound to another question, candidate set, guess, source or digest."""
    fields = {"schema_version", *_SELECTORS, "question_digest", "target_id", "candidate_set_digest",
              "guessed_candidate_id", "choice", "answered_at", "answer_digest"}
    choice = value.get("choice") if isinstance(value, Mapping) else None
    shown = {row.get("candidate_id") for row in question["candidates"]}
    guessed = question["guessed_candidate_id"]
    if (type(value) is not dict or set(value) != fields or value["schema_version"] != ANSWER_SCHEMA
            or value["question_digest"] != question["question_digest"]
            or value["target_id"] != question["target_id"]
            or value["candidate_set_digest"] != question["candidate_set_digest"]
            or value["guessed_candidate_id"] != guessed
            or value["capture_id"] != question["capture_id"] or value["scene_id"] != question["scene_id"]
            or (selectors is not None and any(value[key] != selectors[key] for key in _SELECTORS))
            or type(choice) is not dict
            or not (choice == {"kind": "none_of_these"}
                    or choice == {"kind": "confirm_guess", "candidate_id": guessed}
                    or (set(choice) == {"kind", "candidate_id"} and choice["kind"] == "other_candidate"
                        and choice["candidate_id"] in shown and choice["candidate_id"] != guessed))
            or type(value["answered_at"]) is not str or len(value["answered_at"]) > 40
            or type(value["answer_digest"]) is not str or _DIGEST.fullmatch(value["answer_digest"]) is None
            or value["answer_digest"] != cross_runtime_canonical_digest(value, digest_field="answer_digest")):
        raise ValueError("website_task_target_choice_answer_invalid")
    return dict(value)


def read_answer(pipeline_root: Path, question: Mapping[str, Any]) -> dict[str, Any] | None:
    """The retained answer to exactly this question; an answer to any other question is not one."""
    try:
        value = _read(_path(pipeline_root, question["target_id"], "answer"))
        answer = None if value is None else validate_answer(value, question=question)
        context = _read(pipeline_root / 'website_task_context.json')
        if answer and context and context.get('context_digest') != answer['task_context_digest']:
            return None
        return answer
    except (OSError, ValueError):
        return None


def load_target_choices(pipeline_root: Path) -> dict[str, dict[str, Any]]:
    """{target_id: choice} for the task masks: every target whose site answer names one of its instances."""
    choices = {}
    for question in _all_questions(pipeline_root):
        answer = read_answer(pipeline_root, question)
        if answer is not None and answer["choice"]["kind"] in {"confirm_guess", "other_candidate"}:
            choices[question["target_id"]] = {"target_id": question["target_id"],
                "candidate_set_digest": question["candidate_set_digest"],
                "candidate_id": answer["choice"]["candidate_id"], "question_digest": question["question_digest"],
                "kind": answer["choice"]["kind"], "task_context_digest": answer['task_context_digest']}
    return choices


def rejected_questions(pipeline_root: Path) -> list[dict[str, Any]]:
    """Explicit owner rejections are source facts, never another guess."""
    rows = []
    for question in _all_questions(pipeline_root):
        answer = read_answer(pipeline_root, question)
        if answer is not None and answer['choice']['kind'] == 'none_of_these':
            rows.append({'question': question, 'answer': answer})
    return rows


def corrections(pipeline_root: Path) -> list[dict[str, str]]:
    """An explicit change or rejection invalidates the guessed preparation."""
    values = [{"target_id": choice["target_id"], "question_digest": choice["question_digest"],
               "candidate_id": choice["candidate_id"]}
              for target_id, choice in sorted(load_target_choices(pipeline_root).items())
              if choice["candidate_id"] != (read_question(pipeline_root, target_id) or {}).get("guessed_candidate_id")]
    values.extend({'target_id': row['question']['target_id'], 'question_digest': row['question']['question_digest'],
                   'kind': 'none_of_these'} for row in rejected_questions(pipeline_root))
    return sorted(values, key=lambda row: row['target_id'])


def correction_binding_digest(pipeline_root: Path) -> str | None:
    """Input binding of the website preparation stage, so a correction reruns it and a confirmation does not."""
    fixes = corrections(pipeline_root)
    return canonical_digest({"corrections": fixes}) if fixes else None


def public_question(capture_root: Path) -> dict[str, Any] | None:
    """The one question a site sees: the first current guess it has not settled.

    A "none of these" answer keeps the optional correction available. It never
    requires a replacement recording or leaves the rejected guess current.
    """
    pipeline = Path(capture_root) / "pipeline"
    for question in current_questions(pipeline):
        answer = read_answer(pipeline, question)
        if answer is None or answer["choice"]["kind"] == "none_of_these":
            return {"kind": "task_target_choice", **{key: question[key] for key in (
                "question_digest", "target_id", "semantic_label", "candidate_set_digest", "guessed_candidate_id")},
                "candidates": [{key: row[key] for key in ("candidate_id", "label", "reference_time_ms")}
                               for row in question["candidates"]]}
    return None


def replay_record(listener, *, payload, payload_digest, producer_delivery_key, capture_root: Path) -> dict | None:
    """The exact handoff bytes a correction replays, kept while a guess or a correction exists. Never raises."""
    from .website_assessment_resume import _payload_bytes
    try:
        pipeline = Path(capture_root) / "pipeline"
        if producer_delivery_key is None or not (current_questions(pipeline) or corrections(pipeline)):
            return None
        raw = _payload_bytes(listener, payload, payload_digest)
    except Exception:  # noqa: BLE001 - without the bytes the guess stands and is still shown
        return None
    return {"schema_version": REPLAY_SCHEMA, "payload_base64": base64.b64encode(raw).decode(),
            "source_payload_sha256": payload_digest, "producer_delivery_key": producer_delivery_key}


def _reconstruction_armed(ledger: Mapping[str, Any]) -> bool:
    from .website_reconstruction_resume import armed
    return armed(ledger) or isinstance(ledger.get("assessment_resume"), Mapping)


def correction_pending(ledger: Mapping[str, Any], pipeline_root: Path) -> str | None:
    """The corrections digest the drain still owes a run for, or None.

    Owed while the latest task-mask run still guessed a target the site
    corrected, or while the correction's own run stopped short and nothing else
    (Pub/Sub, or the reconstruction drain's armed wait) owns its retry.
    """
    fixes = corrections(pipeline_root)
    if not fixes:
        # Reaffirming the original guess normally does no work. After a
        # negative answer stopped preparation, it must release that evidence
        # limit. Keep the ordinary no-correction workspace/submission identity
        # so retained completed native work is reused rather than repurchased.
        handoff = _read(pipeline_root / 'website_scene_preparation/handoff.json') or {}
        rejection = handoff.get('selection_rejection') or {}
        restored = []
        for row in rejection.get('rejections') or []:
            if not isinstance(row, Mapping):
                return None
            question = row.get('question') or {}
            answer = read_answer(pipeline_root, question) if question.get('target_id') else None
            if (answer and answer['choice']['kind'] == 'confirm_guess'
                    and row.get('answer', {}).get('choice', {}).get('kind') == 'none_of_these'
                    and answer['task_context_digest'] == rejection.get('task_context_digest')):
                restored.append({'target_id': answer['target_id'], 'question_digest': answer['question_digest'],
                                 'candidate_id': answer['choice']['candidate_id']})
        if (restored and len(restored) == len(rejection['rejections'])
                and handoff.get('digest') == canonical_digest(handoff, digest_field='digest')
                and rejection.get('digest') == canonical_digest(rejection, digest_field='digest')):
            return canonical_digest({'restored_rejected_guesses': sorted(restored, key=lambda row: row['target_id'])})
        return None
    digest = canonical_digest({"corrections": fixes})
    history = ledger.get("attempt_history")
    last = history[-1] if type(history) is list and history else None
    handoff = _read(pipeline_root / 'website_scene_preparation' / 'handoff.json') or {}
    rejection = handoff.get('selection_rejection') or {}
    if (ledger.get('status') == 'completed' and isinstance(last, Mapping)
            and last.get(CORRECTION_MARK) == digest and rejection.get('corrections_digest') == digest
            and handoff.get('digest') == canonical_digest(handoff, digest_field='digest')
            and rejection.get('digest') == canonical_digest(rejection, digest_field='digest')):
        return None  # A truthful evidence limit is not a blind retry loop.
    current = {question["question_digest"] for question in current_questions(pipeline_root)}
    if any(fix["question_digest"] in current for fix in fixes):
        return digest
    history = ledger.get("attempt_history")
    last = history[-1] if type(history) is list and history else None
    if (ledger.get("status") in {"failed_retryable", "retryable_blocked"} and isinstance(last, Mapping)
            and last.get(CORRECTION_MARK) == digest and not _reconstruction_armed(ledger)):
        return digest
    return None


def claim_matches(ledger, expected, *, payload_sha256, producer_delivery_key) -> bool:
    """Inside the claim lock: the ledger is still the one the drain read, and correctable."""
    return (type(expected.get("revision")) is int and ledger.get("revision") == expected["revision"]
            and type(expected.get("corrections_digest")) is str
            and ledger.get("status") in CORRECTABLE_STATUSES and not _reconstruction_armed(ledger)
            and ledger.get("source_payload_sha256") == payload_sha256
            and ledger.get("producer_delivery_key") == producer_delivery_key)


def _poll_due(pipeline: Path, target_id: str, question_digest: str, now: float) -> bool:
    try:
        value = _read(_path(pipeline, target_id, "poll")) or {}
    except (OSError, ValueError):
        return True
    return value.get("question_digest") != question_digest or float(value.get("next_poll_at_epoch") or 0) <= now


def _backoff(previous: Mapping[str, Any], key: str, value: str, now: float) -> dict[str, Any]:
    failures = int(previous.get("failures") or 0) + 1 if previous.get(key) == value else 1
    return {key: value, "failures": failures, "next_poll_at_epoch": now + min(1800, 60 * 2 ** min(failures - 1, 5))}


def fetch_answer(root: Path, ledger: Mapping[str, Any], question: Mapping[str, Any], *,
                 now: float) -> dict[str, Any] | None:
    """Ask the WebApp for the site's answer; retain it only when it answers this exact question."""
    from .website_preparation_status import DELIVERY_FILE, validate_selectors
    from .website_task_context import website_webapp_request
    pipeline = root / "pipeline"
    poll = _path(pipeline, question["target_id"], "poll")
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
        try:
            previous = _read(poll) or {}
        except (OSError, ValueError):
            previous = {}
        write_json(poll, {"schema_version": "website_task_target_choice_poll.v1",
                          **_backoff(previous, "question_digest", question["question_digest"], now)})
        return None
    # Negative answers can be clarified later against the same current question.
    # Retain both observations; unchanged reads never create a new replay.
    previous = read_answer(pipeline, question)
    for value in (previous, answer):
        if value is not None:
            path = pipeline / QUESTION_DIR / 'history' / (value['answer_digest'][7:] + '.json')
            if path.exists() and _read(path) != value:
                raise ValueError('website_task_target_choice_history_changed')
            if not path.exists():
                write_json(path, value)
    write_json(_path(pipeline, question["target_id"], "answer"), answer)
    write_json(poll, {'schema_version': 'website_task_target_choice_poll.v1',
                      'question_digest': question['question_digest'], 'next_poll_at_epoch': now + 60})
    return answer


def _replay_due(pipeline: Path, digest: str, now: float) -> bool:
    try:
        value = _read(pipeline / QUESTION_DIR / REPLAY_STATE_FILE) or {}
    except (OSError, ValueError):
        value = {}
    if value.get("corrections_digest") != digest:
        return True
    return int(value.get("failures") or 0) < MAX_CORRECTION_REPLAYS and float(value.get("next_poll_at_epoch") or 0) <= now


def _note_replay(pipeline: Path, digest: str, now: float) -> None:
    path = pipeline / QUESTION_DIR / REPLAY_STATE_FILE
    try:
        previous = _read(path) or {}
    except (OSError, ValueError):
        previous = {}
    # Durable before the run, so a crash mid-run cannot turn into an unbounded loop.
    write_json(path, {"schema_version": "website_task_target_choice_replay_state.v1",
                      **_backoff(previous, "corrections_digest", digest, now)})


def reconcile_target_choices(listener, *, storage_root: Path, process_args, limit=1, poll_limit=4,
                             now: float | None = None) -> dict[str, int]:
    """Read the site's answers and rerun the same job for a correction; at most ``limit`` reruns per tick."""
    from .task_evaluation_scene_retirement_generations import capture_birth_source_projection
    from .website_assessment_resume import MAX_BYTES, _payload_bytes, _require
    from .website_preparation_status import (
        DELIVERY_FILE,
        _root,
        read_preparation_status,
        validate_selectors,
    )
    current = time.time() if now is None else now
    counts = {"replayed": 0, "confirmed": 0, "corrected": 0, "declined": 0, "waiting": 0}
    polled = 0
    roots = sorted(path.parent for path in Path(storage_root).glob(
        f"*/scenes/site-*/captures/*/{listener.JOB_LEDGER_FILENAME}")
        if path.parent.name.startswith(("walkthrough-", "supplement-")))
    for root in roots:
        if counts["replayed"] >= max(0, min(limit, 50)):
            break
        try:
            pipeline = root / "pipeline"
            if not (pipeline / QUESTION_DIR).is_dir():
                continue
            _root(root)
            _require(not (root / listener.JOB_LEDGER_FILENAME).is_symlink())
            before = listener._read_job_ledger(root)
            if before.get("status") == listener.TERMINAL_AUTHORITY_STATUS:
                continue
            for question in current_questions(pipeline):
                retained = read_answer(pipeline, question)
                if retained is not None and retained['choice']['kind'] != 'none_of_these':
                    continue
                if polled >= max(0, poll_limit) or not _poll_due(pipeline, question["target_id"],
                                                                 question["question_digest"], current):
                    counts["waiting"] += 1
                    continue
                polled += 1
                answer = fetch_answer(root, before, question, now=current)
                kind = None if answer is None else answer["choice"]["kind"]
                counts[{"confirm_guess": "confirmed", "other_candidate": "corrected",
                        "none_of_these": "declined"}.get(kind, "waiting")] += 1
            digest = correction_pending(before, pipeline)
            record = before.get(REPLAY_FIELD)
            if (digest is None or before.get("status") not in CORRECTABLE_STATUSES or _reconstruction_armed(before)
                    or type(record) is not dict or set(record) != _REPLAY_FIELDS
                    or record.get("schema_version") != REPLAY_SCHEMA or not _replay_due(pipeline, digest, current)):
                continue
            _require(type(record["payload_base64"]) is str and len(record["payload_base64"]) <= 4 * ((MAX_BYTES + 2) // 3))
            raw = _payload_bytes(listener, base64.b64decode(record["payload_base64"], validate=True),
                                 record["source_payload_sha256"])
            handoff = listener.parse_handoff_payload(raw)
            _require(root == listener._handoff_capture_root(handoff, storage_root=storage_root)
                     and before.get("source_payload_sha256") == record["source_payload_sha256"]
                     and before.get("producer_delivery_key") == record["producer_delivery_key"])
            _require(handoff.source_finalize is not None and handoff.source_membership_selector is not None)
            # The status reader re-checks the active birth and membership, fresh
            # owner, local consent, current context and unchanged ledger.
            selectors = validate_selectors((_read(root / DELIVERY_FILE) or {}).get("selectors"))
            status = read_preparation_status(capture_root=root, selectors=selectors)
            _require(status["revision"] == before["revision"] and status["state"] != "authority_ended")
            birth = capture_birth_source_projection(root, expected_purpose="scene_preparation")
            _require(birth is not None and birth["source_membership_selector"] == handoff.source_membership_selector)
            _require(listener._read_job_ledger(root) == before)
            _note_replay(pipeline, digest, current)
            counts["replayed"] += 1
            listener.process_handoff_payload(raw, storage_root=storage_root,
                payload_digest=record["source_payload_sha256"],
                expected_target_choice_correction={"revision": before["revision"], "corrections_digest": digest},
                **({"expected_preparation_purpose": "scene_preparation"}
                   if birth.get("capture_observation_purpose") == "scene_preparation" else {}), **process_args)
        except Exception:  # noqa: BLE001 - one capture never stops the others; an owed rerun stays owed
            listener.logger.debug("pubsub_handoff.target_choice_correction_pending")
    return counts
