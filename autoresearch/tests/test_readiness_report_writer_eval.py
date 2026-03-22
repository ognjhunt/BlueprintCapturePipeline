from __future__ import annotations

from autoresearch.test_support import require_target


def test_readiness_report_contains_required_sections() -> None:
    context = require_target("readiness_report_writer")
    expectations = context.expectations("not_ready_yet")
    report = context.text_output("not_ready_yet")
    for section in expectations["required_sections"]:
        assert section in report


def test_pre_screen_report_keeps_required_caveat() -> None:
    context = require_target("readiness_report_writer")
    expectations = context.expectations("pre_screen")
    report = context.text_output("pre_screen")
    for phrase in expectations["required_phrases"]:
        assert phrase in report
