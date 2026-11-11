"""How far each learned policy got in a beta screening run: its ``progress`` (publication v4).

Beta journey: a finished run's counts reach the site as a bounded estimate. A run that scored no
success said only that the simulation "could not rank" the policies, although each episode recorded
what happened. This module reads that record with the existing grader
(``policy_episode_graded_report.progress_episode`` and ``summarize_progress``): every stage comes from
the deterministic score report's own criteria and the episode's sealed state trace, so a grade can
never disagree with the score. There is no second grader.

The episodes are the ones the validated session result already embeds. An episode is graded only
when its score is bound by ``deterministic_score_digest`` and its trace by ``state_trace_digest``,
the summary counted it as scored, and its report is on the pick-and-place ladder. Anything else is
not graded (``episodes_graded`` says how many were): a grading error on one episode never fails the
summary or the publication. Scripted and control rows carry no progress, and a candidate with nothing
graded carries no field at all.

``progress`` informs: it names no winner, changes no count and holds nothing.

Offline, read-only: ``python -m blueprint_pipeline.beta_policy_screening_progress --session-result
<path> [--session-result <path> ...] [--output <path>]`` prints the per-candidate table for retained
results as JSON, including a run that did not complete every episode (the episodes with a bound
score are graded; the rest are counted by why they were skipped).

The summary and the publication read only what the sealed score carries: a score sealed before it
recorded ``behaviour`` has no closest approach, so its candidate's ladder starts at ``contact`` and
carries no ``closest_approach_m``. Only the offline driver, and only with ``--recompute-behaviour``,
measures that distance from the recorded trace with the scorer's own ``_behaviour``, and it labels
each such row ``closest_approach_source: recomputed_from_recorded_trace``.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .policy_episode_graded_report import (
    APPROACHED_DISTANCE_M, PROGRESS_GRADER_VERSION, PROGRESS_STAGE_IDS, progress_episode, summarize_progress,
)

PROGRESS_FIELD = "progress"
REPORT_SCHEMA = "beta_policy_screening_progress_report.v1"
SCORED_OUTCOMES = frozenset({"success", "failure"})


class EpisodeNotGraded(ValueError):
    """Why one episode was left out of a candidate's progress (a typed code)."""


def _bound_inputs(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], list[Any]]:
    """The episode's score and state samples, each checked against the digest the session result binds."""
    if row.get("status") != "completed":
        raise EpisodeNotGraded("episode_not_completed")
    episode = row.get("episode")
    score = episode.get("score") if isinstance(episode, Mapping) else None
    if not isinstance(score, Mapping):
        raise EpisodeNotGraded("score_missing")
    if row.get("deterministic_score_digest") != canonical_digest({"value": score}):
        raise EpisodeNotGraded("score_digest_mismatch")
    trace = episode.get("state_trace")
    samples = trace.get("task_state_samples") if isinstance(trace, Mapping) else None
    if not isinstance(samples, list) or not samples:
        raise EpisodeNotGraded("state_trace_missing")
    if row.get("state_trace_digest") != canonical_digest({"value": trace}):
        raise EpisodeNotGraded("state_trace_digest_mismatch")
    return score, samples


def _recomputed_approach(row: Mapping[str, Any], samples: list[Any]) -> float | None:
    """The closest approach the scorer's own ``_behaviour`` measures on the recorded trace (offline driver only)."""
    from .adp_rigid_task_scoring import _behaviour
    from .policy_episode_graded_report import closest_approach_m

    task_spec = row["episode"].get("task_spec")
    if not isinstance(task_spec, Mapping):
        return None
    return closest_approach_m({"behaviour": _behaviour(task_spec, samples)})


def grade_rows(rows: Iterable[Any], *, scored_cells: set[str] | None = None,
               recompute_behaviour: bool = False) -> tuple[dict[str, Any] | None, dict[str, int]]:
    """A candidate's ``progress`` from its episode rows, and how many were skipped and why.

    ``scored_cells`` (the cells the summary counted as success or failure) keeps
    ``episodes_graded <= episodes_scored``; ``None`` reads each row's own outcome.

    ``recompute_behaviour`` is for the offline driver only, never the summary or the publication: a score
    sealed before it carried ``behaviour`` gets its closest approach from the recorded trace with the
    scorer's own function, and each such episode is counted under ``behaviour_recomputed``.
    """
    from .team_policy_episode_outcome import episode_outcome

    graded, skipped, recomputed = [], {}, 0
    for row in rows:
        try:
            if not isinstance(row, Mapping):
                raise EpisodeNotGraded("episode_row_invalid")
            counted = (row.get("cell_id") in scored_cells if scored_cells is not None
                       else episode_outcome(row)[0] in SCORED_OUTCOMES)
            if not counted:
                raise EpisodeNotGraded("episode_not_scored")
            score, samples = _bound_inputs(row)
            item = progress_episode(score=score, samples=samples)
            if item is None:
                raise EpisodeNotGraded("not_on_pick_and_place_ladder")
            if recompute_behaviour and item["approach_distance_m"] is None and "behaviour" not in score:
                item["approach_distance_m"] = _recomputed_approach(row, samples)
                if item["approach_distance_m"] is not None:
                    recomputed += 1
            graded.append(item)
        except EpisodeNotGraded as exc:
            skipped[str(exc)] = skipped.get(str(exc), 0) + 1
        except Exception:  # noqa: BLE001 - one episode's grading error never fails the summary.
            skipped["grading_error"] = skipped.get("grading_error", 0) + 1
    counts = {**skipped, **({"behaviour_recomputed": recomputed} if recomputed else {})}
    try:
        return summarize_progress(graded), dict(sorted(counts.items()))
    except Exception:  # noqa: BLE001 - the same rule for the candidate's totals.
        return None, {"grading_error": len(graded), **counts}


def summary_progress(result: Mapping[str, Any], *, candidate: Mapping[str, Any],
                     outcomes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The summary's ``progress`` field for one learned-policy candidate, or nothing."""
    from .beta_policy_screening_known_good_control import is_control

    if is_control(candidate):
        return {}
    scored = {row["cell_id"] for row in outcomes if row.get("outcome") in SCORED_OUTCOMES}
    rows = [row for row in result.get("episodes") or []
            if isinstance(row, Mapping) and row.get("candidate_id") == candidate.get("candidate_id")]
    progress, _ = grade_rows(rows, scored_cells=scored)
    return {PROGRESS_FIELD: progress} if progress is not None else {}


def progress_valid(value: Any, *, episodes_scored: Any) -> bool:
    """The wire rules both repositories enforce: grader version, closed ordered stages, the inequalities."""
    keys = {"grader_version", "episodes_graded", "stages", "mean_score"}
    if not isinstance(value, Mapping) or set(value) - {"closest_approach_m"} != keys:
        return False
    graded, stages, mean = value["episodes_graded"], value["stages"], value["mean_score"]
    if (value["grader_version"] != PROGRESS_GRADER_VERSION or type(graded) is not int
            or type(episodes_scored) is not int or not 1 <= graded <= episodes_scored
            or not isinstance(stages, list) or isinstance(mean, bool) or not isinstance(mean, (int, float))
            or not 0 <= mean <= 1 or round(mean, 4) != mean):
        return False
    ids = [row.get("stage") if isinstance(row, Mapping) else None for row in stages]
    if ids not in (list(PROGRESS_STAGE_IDS), list(PROGRESS_STAGE_IDS[1:])):
        return False
    if ("closest_approach_m" in value) != (ids[0] == "approached") or not _approach_valid(value.get("closest_approach_m")):
        return False
    counts = [row.get("episodes") for row in stages]
    if any(set(row) != {"stage", "episodes"} or type(row["episodes"]) is not int for row in stages):
        return False
    return all(0 <= count <= graded for count in counts) and all(a >= b for a, b in zip(counts, counts[1:]))


def _approach_valid(value: Any) -> bool:
    """Absent, or ``{median, min}`` in metres, 4 decimals, ``0 <= min <= median``."""
    if value is None:
        return True
    if not isinstance(value, Mapping) or set(value) != {"median", "min"}:
        return False
    numbers = [value["min"], value["median"]]
    return (all(not isinstance(item, bool) and isinstance(item, (int, float)) and 0 <= item < 1e6
                and round(item, 4) == item for item in numbers) and numbers[0] <= numbers[1])


def publication_progress(configured: Mapping[str, Any], row: Mapping[str, Any],
                         summary: Mapping[str, Any]) -> dict[str, Any]:
    """The candidate row's ``progress``: a completed learned-policy row's valid summary field, else nothing."""
    from .beta_policy_screening_known_good_control import is_control
    from .policy_execution_evidence import SCRIPTED

    value = summary.get(PROGRESS_FIELD)
    if (value is None or row.get("status") != "completed" or is_control(configured)
            or SCRIPTED in (configured.get("execution_evidence_kind"), summary.get("execution_evidence_kind"))
            or not progress_valid(value, episodes_scored=summary.get("episodes_scored"))):
        return {}
    return {PROGRESS_FIELD: json.loads(json.dumps(value))}


# ------------------------------------------------------------- offline driver


def result_report(value: Mapping[str, Any], *, path: str, recompute_behaviour: bool = False) -> dict[str, Any]:
    """One retained session result's per-candidate table. Read-only; partial runs are graded as far as they go.

    With ``recompute_behaviour`` a candidate row whose closest approach came from the recorded trace rather
    than the sealed score says so (``closest_approach_source``); without it, only the sealed score is read.
    """
    from .native_task_arena_policy_canary_session import validate_session_result
    from .policy_execution_evidence import SCRIPTED, candidate_execution_kinds

    try:
        validate_session_result(value, allow_legacy_missing_task_success_contract=True)
        validation = None
    except ValueError as exc:
        validation = str(exc) if str(exc).replace("_", "").isalnum() else "session_result_invalid"
    try:
        kinds = candidate_execution_kinds(value)
    except ValueError:
        kinds = {}
    rows = [row for row in value.get("episodes") or [] if isinstance(row, Mapping)]
    candidates = list(dict.fromkeys([*(value.get("candidate_ids") or []), *(row.get("candidate_id") for row in rows)]))
    table = []
    for candidate_id in candidates:
        own = [row for row in rows if row.get("candidate_id") == candidate_id]
        kind = kinds.get(candidate_id, "learned_policy")
        progress, skipped = ((None, {"scripted_row": len(own)}) if kind == SCRIPTED
                             else grade_rows(own, recompute_behaviour=recompute_behaviour))
        source = (None if not progress or "closest_approach_m" not in progress
                  else "recomputed_from_recorded_trace" if skipped.get("behaviour_recomputed") == progress["episodes_graded"]
                  else "partly_recomputed_from_recorded_trace" if skipped.get("behaviour_recomputed")
                  else "sealed_score")
        table.append({"candidate_id": candidate_id, "execution_evidence_kind": kind, "episodes": len(own),
                      "closest_approach_source": source,
                      "episodes_completed": sum(row.get("status") == "completed" for row in own),
                      "episodes_graded": progress["episodes_graded"] if progress else 0,
                      "episodes_skipped": skipped, "progress": progress})
    return {"path": path, "run_id": value.get("run_id"), "status": value.get("status"),
            "result_digest_valid": value.get("result_digest") == canonical_digest(value, digest_field="result_digest"),
            "session_result_valid": validation is None, "session_result_error": validation, "candidates": table}


def report(paths: Sequence[str | Path], *, recompute_behaviour: bool = False) -> dict[str, Any]:
    results = []
    for path in paths:
        source = Path(path).expanduser()
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("beta_screening_progress_session_result_invalid")
        results.append(result_report(value, path=str(path), recompute_behaviour=recompute_behaviour))
    return {"schema_version": REPORT_SCHEMA, "grader_version": PROGRESS_GRADER_VERSION,
            "approached_distance_m": APPROACHED_DISTANCE_M, "behaviour_recompute_requested": recompute_behaviour,
            "results": results}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session-result", action="append", required=True,
                        help="A retained native session result (JSON); repeat for several.")
    parser.add_argument("--output", help="Write the table here instead of stdout (refuses an existing file).")
    parser.add_argument("--recompute-behaviour", action="store_true",
                        help="For a score sealed without 'behaviour', measure the closest approach from the recorded "
                             "trace with the scorer's own function; such rows are labelled as recomputed.")
    args = parser.parse_args(argv)
    text = json.dumps(report(args.session_result, recompute_behaviour=args.recompute_behaviour),
                      indent=2, sort_keys=True) + "\n"
    if args.output:
        out = Path(args.output)
        if out.exists():
            raise SystemExit("beta_screening_progress_output_exists")
        out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["PROGRESS_FIELD", "REPORT_SCHEMA", "EpisodeNotGraded", "grade_rows", "main", "progress_valid",
           "publication_progress", "report", "result_report", "summary_progress"]
