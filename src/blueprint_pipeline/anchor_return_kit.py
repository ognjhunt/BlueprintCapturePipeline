"""Issue pre-populated templates for returning real-world trial outcomes.

Calibration joins an accepted real-world anchor to a Blueprint prediction on
four exact keys -- ``scenario_eval_run_id``, ``policy_id``, ``task_id``,
``scenario_variation_instance_id`` (``robot_eval_calibration``). A returned row
that does not reproduce all four is rejected as ``unmatched_actual_row``.

The failure mode that creates is expensive and late: an operator schedules
physical trials, runs a robot for days, and only discovers at ingest that the
rows cannot join -- because a task id was transcribed differently, a variation
instance was renamed, or the run id was never recorded at the bench at all. The
trials are real and the robot time is spent, but the outcomes are unusable.

The fix is to stop asking operators to reconstruct join keys after the fact.
This module issues, for every prediction, a row that already carries all four
keys, leaving only the outcome fields to fill in. It then validates a returned
file against the issued one *before* ingest, so a mismatch is caught while the
robot is still available rather than after the campaign is over.

Issuing a kit is not a measurement, and a returned kit is not an accepted
anchor: acceptance remains the calibration path's decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import read_json_any, utc_now_iso, write_json
from .robot_eval_calibration import ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS


KIT_SCHEMA_VERSION = "anchor_return_kit.v1"
RETURN_VALIDATION_SCHEMA_VERSION = "anchor_return_validation.v1"

# Filled in at the bench. Everything else is pre-populated and must not change.
OUTCOME_FIELDS = (
    "trial_index",
    "observed_success",
    "observed_at",
    "operator_id",
    "failure_mode",
    "notes",
)
REQUIRED_OUTCOME_FIELDS = ("trial_index", "observed_success", "observed_at", "operator_id")

CSV_COLUMNS = tuple(ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS) + OUTCOME_FIELDS


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _join_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(_string(row.get(key)) for key in ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS)


def build_anchor_return_kit(
    *,
    kit_id: str,
    predictions: Sequence[Mapping[str, Any]],
    trials_per_condition: int = 1,
    instructions_uri: str = "",
) -> dict[str, Any]:
    """Issue one return row per prediction per planned trial.

    Every row arrives with its four join keys already filled in, so the bench
    never has to reconstruct them. Rows are emitted even when a prediction is
    incomplete -- with the offending keys reported -- because discovering a
    missing ``scenario_variation_instance_id`` now is the entire point.
    """

    blockers: list[str] = []
    if not _string(kit_id):
        blockers.append("anchor_kit_id_missing")
    if trials_per_condition < 1:
        blockers.append("anchor_kit_trials_per_condition_invalid")
        trials_per_condition = 1

    issued: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for index, raw in enumerate(predictions):
        prediction = _mapping(raw)
        keys = _join_key(prediction)
        missing = [
            name
            for name, value in zip(ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS, keys)
            if not value
        ]
        if missing:
            blockers.append(
                f"prediction_missing_join_keys:{index}:{','.join(missing)}"
            )
            continue
        if keys in seen:
            # Two predictions sharing all four keys would make returned rows
            # ambiguous at join time.
            blockers.append(f"duplicate_prediction_join_key:{'/'.join(keys)}")
            continue
        seen.add(keys)
        for trial_index in range(1, trials_per_condition + 1):
            row = dict(zip(ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS, keys))
            row.update(
                {
                    "trial_index": trial_index,
                    "observed_success": None,
                    "observed_at": None,
                    "operator_id": None,
                    "failure_mode": None,
                    "notes": None,
                    "predicted_success_rate": prediction.get("predicted_success_rate"),
                }
            )
            issued.append(row)

    if not issued and not blockers:
        blockers.append("anchor_kit_has_no_rows")

    kit_core = {
        "kit_id": _string(kit_id),
        "join_keys": list(ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS),
        "rows": [
            {key: row[key] for key in ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS}
            | {"trial_index": row["trial_index"]}
            for row in issued
        ],
    }
    return {
        "schema_version": KIT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "issued" if not blockers else "blocked",
        "kit_id": _string(kit_id) or None,
        "join_keys": list(ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS),
        "outcome_fields": list(OUTCOME_FIELDS),
        "required_outcome_fields": list(REQUIRED_OUTCOME_FIELDS),
        "trials_per_condition": trials_per_condition,
        "row_count": len(issued),
        "rows": issued,
        # Binds the returned file to the exact set of rows that was issued.
        "kit_sha256": canonical_sha256(kit_core),
        "instructions_uri": _string(instructions_uri) or None,
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "issuing_a_kit_is_not_a_measurement": True,
            "returned_rows_are_not_accepted_anchors_until_calibration_accepts_them": True,
            "kit_does_not_authorize_any_public_claim": True,
        },
    }


def render_kit_csv(kit: Mapping[str, Any]) -> str:
    """The bench-facing artifact: join keys pre-filled, outcomes blank."""

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for row in _rows(kit.get("rows")):
        writer.writerow(
            {
                column: ("" if row.get(column) is None else row.get(column))
                for column in CSV_COLUMNS
            }
        )
    return buffer.getvalue()


def parse_returned_csv(text: str) -> list[dict[str, Any]]:
    return [dict(row) for row in csv.DictReader(io.StringIO(text))]


def _coerce_success(value: Any) -> bool | None:
    text = _string(value).lower()
    if text in {"1", "true", "yes", "y", "success", "pass"}:
        return True
    if text in {"0", "false", "no", "n", "failure", "fail"}:
        return False
    if isinstance(value, bool):
        return value
    return None


def validate_returned_anchors(
    *,
    kit: Mapping[str, Any],
    returned_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Check a returned file against the issued kit before ingest.

    Reports unmatched, missing and duplicate rows explicitly. This is the check
    that used to happen only at calibration ingest, by which point the robot
    time had already been spent.
    """

    blockers: list[str] = []
    if kit.get("schema_version") != KIT_SCHEMA_VERSION:
        blockers.append("anchor_kit_schema_missing_or_unsupported")
    if kit.get("status") != "issued":
        blockers.append("anchor_kit_not_issued")

    expected: dict[tuple[str, ...], set[int]] = {}
    for row in _rows(kit.get("rows")):
        expected.setdefault(_join_key(row), set()).add(int(row.get("trial_index") or 0))

    matched: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, ...], int]] = set()

    for index, raw in enumerate(returned_rows):
        row = _mapping(raw)
        keys = _join_key(row)
        try:
            trial_index = int(_string(row.get("trial_index")) or 0)
        except ValueError:
            trial_index = 0
        identity = {"join_key": list(keys), "trial_index": trial_index, "row_index": index}

        if keys not in expected or trial_index not in expected[keys]:
            # This is the row that would have been silently rejected at ingest.
            unmatched.append(identity)
            continue
        if (keys, trial_index) in seen:
            duplicates.append(identity)
            continue
        seen.add((keys, trial_index))

        missing = [
            field
            for field in REQUIRED_OUTCOME_FIELDS
            if field != "trial_index" and not _string(row.get(field))
        ]
        success = _coerce_success(row.get("observed_success"))
        if success is None:
            missing.append("observed_success")
        if missing:
            incomplete.append({**identity, "missing_fields": sorted(set(missing))})
            continue
        matched.append(
            {
                **dict(zip(ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS, keys)),
                "trial_index": trial_index,
                "observed_success": success,
                "observed_at": _string(row.get("observed_at")),
                "operator_id": _string(row.get("operator_id")),
                "failure_mode": _string(row.get("failure_mode")) or None,
            }
        )

    expected_pairs = {(keys, trial) for keys, trials in expected.items() for trial in trials}
    missing_rows = [
        {"join_key": list(keys), "trial_index": trial}
        for keys, trial in sorted(expected_pairs - seen)
    ]

    if unmatched:
        blockers.append("returned_rows_do_not_join_to_issued_kit")
    if duplicates:
        blockers.append("returned_rows_contain_duplicate_trials")
    if incomplete:
        blockers.append("returned_rows_missing_required_outcome_fields")

    return {
        "schema_version": RETURN_VALIDATION_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "kit_id": kit.get("kit_id"),
        "kit_sha256": kit.get("kit_sha256"),
        "status": "ready_for_ingest" if not blockers else "blocked",
        "issued_row_count": len(expected_pairs),
        "returned_row_count": len(returned_rows),
        "matched_rows": matched,
        "matched_row_count": len(matched),
        "unmatched_rows": unmatched,
        "duplicate_rows": duplicates,
        "incomplete_rows": incomplete,
        "missing_rows": missing_rows,
        "coverage_fraction": (
            round(len(matched) / len(expected_pairs), 6) if expected_pairs else None
        ),
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "validation_checks_joinability_not_outcome_truth": True,
            "ready_for_ingest_is_not_accepted_anchor_status": True,
        },
    }


def _command_issue(args: argparse.Namespace) -> int:
    payload = _mapping(read_json_any(Path(args.input)))
    kit = build_anchor_return_kit(
        kit_id=_string(payload.get("kit_id")),
        predictions=_rows(payload.get("predictions")),
        trials_per_condition=int(payload.get("trials_per_condition") or 1),
        instructions_uri=_string(payload.get("instructions_uri")),
    )
    write_json(Path(args.output), kit)
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        Path(args.csv).write_text(render_kit_csv(kit), encoding="utf-8")
    print(json.dumps({"path": args.output, "status": kit["status"], "rows": kit["row_count"]}, sort_keys=True))
    return 0 if kit["status"] == "issued" else 1


def _command_validate(args: argparse.Namespace) -> int:
    kit = _mapping(read_json_any(Path(args.kit)))
    text = Path(args.returned).read_text(encoding="utf-8")
    rows = parse_returned_csv(text) if args.returned.endswith(".csv") else _rows(json.loads(text))
    report = validate_returned_anchors(kit=kit, returned_rows=rows)
    write_json(Path(args.output), report)
    print(json.dumps({"path": args.output, "status": report["status"]}, sort_keys=True))
    return 0 if report["status"] == "ready_for_ingest" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Issue and validate real-world anchor return kits"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    issue = sub.add_parser("issue", help="issue a pre-populated return kit")
    issue.add_argument("--input", required=True)
    issue.add_argument("--output", required=True)
    issue.add_argument("--csv", default=None)
    issue.set_defaults(func=_command_issue)

    validate = sub.add_parser("validate", help="check a returned file before ingest")
    validate.add_argument("--kit", required=True)
    validate.add_argument("--returned", required=True)
    validate.add_argument("--output", required=True)
    validate.set_defaults(func=_command_validate)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
