from __future__ import annotations

from scripts.validate_pubsub_handoff_infra import has_run_e2e_result_binding


def test_run_e2e_result_binding_accepts_formatter_multiline_conditional() -> None:
    source = """
result = (
    run_e2e(**run_kwargs)
    if run_e2e_enabled
    else {"status": "skipped"}
)
"""

    assert has_run_e2e_result_binding(source) is True


def test_run_e2e_result_binding_rejects_unbound_or_different_arguments() -> None:
    assert has_run_e2e_result_binding("run_e2e(**run_kwargs)") is False
    assert has_run_e2e_result_binding("result = run_e2e(capture_root=root)") is False
