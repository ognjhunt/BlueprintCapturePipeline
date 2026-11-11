"""Does a policy behave the same through the standard contract as on its native path?

Plan 2.1's proof (``docs/launch-readiness/2026-11-11-evaluation-service-plan.md``):
π0.5 served over ``openpi_droid_https.v1`` on the session's machine gives the
same result as π0.5 on the native OpenPI path, on the same job and the same
tries. Given two finished session results of the same policy on the same scene
revision and cells, one native and one through the contract, this compares
each try:

- the outcome (``team_policy_episode_outcome.episode_outcome``: success,
  failure or void, with its reason);
- the deepest stage reached and the closest approach, from the existing grader
  (``policy_episode_graded_report.progress_episode``), read only from a score
  and state trace bound by their digests;
- the commanded action trace. Both paths seed each cell's simulator from the
  plan and load the same checkpoint with the same loader; neither infers at
  readiness, so the policy's random state advances identically. The traces are
  then expected to match exactly; a GPU's floating-point nondeterminism can
  still separate them, which is why outcome agreement is reported on its own.

The verdict is plain: ``same`` (every try agrees, actions included), ``same_
outcomes`` (outcome, stage and approach agree on every try, some action trace
differs), ``differs`` (with each difference listed), or ``not_comparable``
(different scene, contract, cells or policy). It reads retained files only:
nothing is launched and nothing is spent.

``python -m blueprint_pipeline.policy_path_parity --native <result.json> --contract <result.json>
--native-candidate pi05_droid --contract-candidate pi05_droid_contract [--output report.json]``
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

from .decision_evidence_contracts import canonical_digest

REPORT_SCHEMA = "blueprint.policy_path_parity_report.v1"
APPROACH_TOLERANCE_M = 1e-6
_CELL_FIELDS = ("seed", "cell_spec_digest", "resolved_scenario_digest")


def _deepest(row: Mapping[str, Any]) -> tuple[str | None, float | None, str | None]:
    """The deepest stage in ladder order and the closest approach, or why the try was not graded."""

    from .beta_policy_screening_progress import EpisodeNotGraded, _bound_inputs
    from .policy_episode_graded_report import APPROACHED_DISTANCE_M, PROGRESS_STAGE_IDS, progress_episode

    try:
        score, samples = _bound_inputs(row)
        item = progress_episode(score=score, samples=samples)
    except EpisodeNotGraded as exc:
        return None, None, str(exc)
    except Exception:  # noqa: BLE001 - a grading error is reported, never raised
        return None, None, "grading_error"
    if item is None:
        return None, None, "not_on_pick_and_place_ladder"
    reached = {stage["id"] for stage in item["stages"] if stage["condition_met"]}
    approach = item["approach_distance_m"]
    if approach is not None and (approach <= APPROACHED_DISTANCE_M or "contact" in reached):
        reached.add("approached")
    deepest = None
    for stage in PROGRESS_STAGE_IDS:  # Ladder order: a stage counts only when every one before it was met.
        if stage not in reached:
            break
        deepest = stage
    return deepest, approach, None


def _numbers(value: Any) -> list[float] | None:
    out: list[float] = []

    def walk(item: Any) -> bool:
        if isinstance(item, Mapping):
            return all(walk(item[key]) for key in sorted(item))
        if isinstance(item, (list, tuple)):
            return all(walk(child) for child in item)
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return True
        out.append(float(item))
        return True

    walk(value)
    return out


def _actions(native: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    left = native.get("returned_action_sequence_digest")
    right = contract.get("returned_action_sequence_digest")
    if left is None or right is None:
        return {"compared": False, "reason": "action_trace_digest_missing"}
    if left == right:
        return {"compared": True, "identical": True}
    result: dict[str, Any] = {"compared": True, "identical": False}
    sequences = [(row.get("episode") or {}).get("commanded_actions") for row in (native, contract)]
    if all(isinstance(value, list) for value in sequences):
        a, b = (_numbers(value) for value in sequences)
        if len(sequences[0]) != len(sequences[1]) or len(a) != len(b):
            result["length"] = [len(sequences[0]), len(sequences[1])]
        else:
            result["max_abs_difference"] = max((abs(x - y) for x, y in zip(a, b, strict=True)), default=0.0)
    return result


def _try(cell_id: str, native: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    from .team_policy_episode_outcome import episode_outcome

    sides = {}
    for name, row in (("native", native), ("contract", contract)):
        outcome, reason = episode_outcome(row)
        deepest, approach, ungraded = _deepest(row)
        sides[name] = {"outcome": outcome, "reason": reason, "deepest_stage": deepest,
                       "closest_approach_m": approach, **({"not_graded": ungraded} if ungraded else {})}
    differences = [field for field in ("outcome", "reason", "deepest_stage")
                   if sides["native"][field] != sides["contract"][field]]
    a, b = sides["native"]["closest_approach_m"], sides["contract"]["closest_approach_m"]
    if (a is None) != (b is None) or (a is not None and b is not None and abs(a - b) > APPROACH_TOLERANCE_M):
        differences.append("closest_approach_m")
    actions = _actions(native, contract)
    if actions.get("compared") and not actions["identical"]:
        differences.append("action_trace")
    return {"cell_id": cell_id, "seed": native.get("seed"), **sides, "actions": actions, "differences": differences}


def _rows(result: Mapping[str, Any], candidate: str) -> dict[str, Mapping[str, Any]]:
    rows = {}
    for row in result.get("episodes") or []:
        if isinstance(row, Mapping) and row.get("candidate_id") == candidate and isinstance(row.get("cell_id"), str):
            if row["cell_id"] in rows:
                raise ValueError("policy_path_parity_cell_repeated:" + row["cell_id"])
            rows[row["cell_id"]] = row
    return rows


def compare_runs(*, native: Mapping[str, Any], contract: Mapping[str, Any], native_candidate: str,
                 contract_candidate: str) -> dict[str, Any]:
    """The per-try comparison and its verdict."""

    reasons = []
    for field in ("scene_revision_digest", "task_success_contract_digest"):
        if native.get(field) is None or native.get(field) != contract.get(field):
            reasons.append(field + "_differs")
    left, right = _rows(native, native_candidate), _rows(contract, contract_candidate)
    if not left or not right:
        reasons.append("episodes_missing")
    if set(left) != set(right):
        reasons.append("cells_differ")
    for cell in sorted(set(left) & set(right)):
        if any(left[cell].get(field) != right[cell].get(field) for field in _CELL_FIELDS):
            reasons.append("cell_seed_or_scenario_differs:" + cell)
        if left[cell].get("checkpoint_digest") != right[cell].get("checkpoint_digest"):
            reasons.append("checkpoint_differs:" + cell)
    tries = [_try(cell, left[cell], right[cell]) for cell in sorted(set(left) & set(right))]
    if reasons:
        verdict = "not_comparable"
    elif all(not row["differences"] for row in tries):
        verdict = "same"
    elif all(row["differences"] == ["action_trace"] for row in tries if row["differences"]):
        verdict = "same_outcomes"
    else:
        verdict = "differs"
    summary = {name: {outcome: sum(row[name]["outcome"] == outcome for row in tries)
                      for outcome in ("success", "failure", "void")} for name in ("native", "contract")}
    report = {"schema_version": REPORT_SCHEMA, "verdict": verdict, "not_comparable_reasons": sorted(set(reasons)),
              "native_candidate": native_candidate, "contract_candidate": contract_candidate,
              "scene_revision_digest": native.get("scene_revision_digest"),
              "task_success_contract_digest": native.get("task_success_contract_digest"),
              "tries": tries, "outcome_counts": summary,
              "tries_differing": [row["cell_id"] for row in tries if row["differences"]],
              "approach_tolerance_m": APPROACH_TOLERANCE_M, "report_digest": ""}
    report["report_digest"] = canonical_digest(report, digest_field="report_digest")
    return report


def plain_verdict(report: Mapping[str, Any]) -> str:
    verdict = report["verdict"]
    if verdict == "same":
        return f"Same: all {len(report['tries'])} tries agree, action traces included."
    if verdict == "same_outcomes":
        return (f"Same outcomes: every try's outcome, deepest stage and closest approach agree; "
                f"action traces differ on {', '.join(report['tries_differing'])}.")
    if verdict == "not_comparable":
        return "Not comparable: " + ", ".join(report["not_comparable_reasons"]) + "."
    parts = [f"{row['cell_id']} ({', '.join(row['differences'])})" for row in report["tries"] if row["differences"]]
    return "Differs: " + "; ".join(parts) + "."


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--native", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--native-candidate", required=True)
    parser.add_argument("--contract-candidate", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = compare_runs(native=json.loads(Path(args.native).read_text(encoding="utf-8")),
                          contract=json.loads(Path(args.contract).read_text(encoding="utf-8")),
                          native_candidate=args.native_candidate, contract_candidate=args.contract_candidate)
    report["plain_verdict"] = plain_verdict(report)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["REPORT_SCHEMA", "compare_runs", "plain_verdict"]
