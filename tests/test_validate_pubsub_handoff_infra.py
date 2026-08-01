from __future__ import annotations

from scripts.validate_pubsub_handoff_infra import (
    has_project_runtime_dependency,
    has_run_e2e_result_binding,
)


def test_pubsub_must_be_a_direct_production_dependency() -> None:
    optional_only = """
[project]
dependencies = ["google-cloud-storage>=2.10.0"]

[project.optional-dependencies]
cloud = ["google-cloud-pubsub>=2.21.0"]
"""
    direct = """
[project]
dependencies = ["google-cloud-pubsub>=2.21.0"]
"""

    assert has_project_runtime_dependency(optional_only, "google-cloud-pubsub") is False
    assert has_project_runtime_dependency(direct, "google-cloud-pubsub") is True


def test_runtime_dependency_parser_fails_closed_for_invalid_toml() -> None:
    assert has_project_runtime_dependency("[project", "google-cloud-pubsub") is False


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
