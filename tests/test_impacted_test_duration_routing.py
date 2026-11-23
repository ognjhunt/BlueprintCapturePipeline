"""The impacted-test selector routes by estimated duration as well as by file count."""

from __future__ import annotations

import json
import re
from pathlib import Path

from blueprint_pipeline import impacted_test_selection as selection


ROOT = Path(__file__).resolve().parents[1]


def _repository(tmp_path: Path, durations: dict[str, float]) -> Path:
    (tmp_path / "tests").mkdir()
    for path in durations:
        (tmp_path / path).write_text("def test_one():\n    pass\n", encoding="utf-8")
    baseline = tmp_path / selection.DURATION_BASELINE
    baseline.parent.mkdir(parents=True)
    baseline.write_text(
        json.dumps(
            {
                "default_seconds_per_test": 0.25,
                "files": [
                    {"path": path, "duration_seconds": seconds, "test_count": 1}
                    for path, seconds in durations.items()
                ],
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_a_selection_estimated_over_the_bounded_budget_goes_to_the_wide_lane(tmp_path: Path) -> None:
    slow = "tests/test_slow_room.py"
    root = _repository(tmp_path, {slow: selection.BOUNDED_LANE_BUDGET_SECONDS + 1.0})

    plan = selection.build_plan(root, [slow])

    assert plan["requires_wide_impacted"] is True
    assert plan["requires_full_suite"] is False
    assert plan["wide_impacted_tests"] == [slow]
    assert slow not in plan["selected_tests"]
    assert any(
        reason.startswith("impacted_estimated_seconds_exceed_budget:") for reason in plan["reasons"]
    )


def test_a_selection_estimated_within_the_bounded_budget_stays_in_it(tmp_path: Path) -> None:
    fast = "tests/test_fast_room.py"
    root = _repository(tmp_path, {fast: 1.0})

    plan = selection.build_plan(root, [fast])

    assert plan["requires_wide_impacted"] is False
    assert fast in plan["selected_tests"]


def test_the_budget_matches_the_bounded_lane_timeout_in_ci() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    timeouts = re.findall(
        r"impacted_test_selection[^\n]*\n(?:[^\n]*\\\n)*?[^\n]*--timeout-seconds (\d+)", workflow
    )
    assert timeouts == [str(selection.BOUNDED_LANE_BUDGET_SECONDS)]
