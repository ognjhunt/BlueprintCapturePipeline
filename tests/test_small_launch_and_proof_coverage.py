from __future__ import annotations

import logging
from pathlib import Path

import pytest

from blueprint_pipeline import proof_contracts
from blueprint_pipeline.core import logging_utils


def test_logging_utils_sanitizes_sensitive_and_structured_fields(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    assert logging_utils._redacted_marker("") == "<redacted:sensitive>"
    assert logging_utils._sanitize_value("api_key", "secret") == "<redacted:api-key>"
    assert logging_utils._sanitize_value("path", tmp_path / "artifact.json") == str(
        tmp_path / "artifact.json"
    )
    assert (
        logging_utils._sanitize_value("url", "https://x.test/file?X-Goog-Signature=abc")
        == "<redacted:signed-url>"
    )
    assert logging_utils._sanitize_value("nested", {"token": "secret"}) == {
        "token": "<redacted:token>"
    }
    assert logging_utils._sanitize_value("items", [{"password": "secret"}]) == [
        {"password": "<redacted:password>"}
    ]
    assert logging_utils._sanitize_value("count", 3) == 3
    assert logging_utils._format_fields({}) == ""

    logger = logging.getLogger("blueprint_pipeline.test_logging_utils")
    with caplog.at_level(logging.INFO, logger=logger.name):
        logging_utils.log_event(
            logger,
            logging.INFO,
            "coverage.event\r\nforged-event",
            message="coverage message\nforged-message",
            api_key="secret",
            artifact_path=tmp_path / "artifact.json",
            detail="safe detail\nforged-field\r",
            **{"not-an-identifier": "kept"},
        )

    record = caplog.records[-1]
    assert record.blueprint_event == "coverage.event\\r\\nforged-event"
    assert "\n" not in record.getMessage()
    assert "coverage message\\nforged-message" in record.getMessage()
    assert record.blueprint_fields["api_key"] == "<redacted:api-key>"
    assert record.blueprint_fields["detail"] == "safe detail\\nforged-field\\r"
    assert "detail='safe detail\\\\nforged-field\\\\r'" in record.getMessage()
    assert record.artifact_path == str(tmp_path / "artifact.json")
    assert not hasattr(record, "not-an-identifier")


def test_proof_contracts_missing_hosted_inputs() -> None:
    assert proof_contracts._string_list("alpha") == ["alpha"]
    assert proof_contracts._string_list(123) == []

    package = proof_contracts.build_site_package_manifest(
        scene_id="scene-1",
        capture_id="capture-1",
        site_submission_id=None,
        opportunity_id=None,
        evaluation_prep_manifest={"canonical_package_status": "registration_blocked"},
        site_world_spec={},
        site_world_registration={},
        site_world_health={"launchable": False},
        launchable_export_bundle={"status": "missing"},
        site_identity={},
        adjacent_systems=[],
        rights_review={"status": "blocked"},
    )
    assert package["status"] == "blocked"
    assert "site_world_spec_missing" in package["blockers"]

    hosted = proof_contracts.build_hosted_review_readiness(
        scene_id="scene-1",
        capture_id="capture-1",
        site_submission_id=None,
        opportunity_id=None,
        site_identity={"site_id": "site-1"},
        adjacent_systems=[],
        preview_manifest_uri=None,
        worldlabs_launch_url=None,
        runtime_demo_manifest_uri=None,
        demo_readiness_state="blocked",
        demo_blockers=["demo_missing"],
        site_world_health={"launchable": False},
        launchable_export_bundle={"status": "missing"},
    )
    assert hosted["status"] == "blocked"
    assert "preview_manifest_missing" in hosted["blockers"]
    assert "runtime_demo_manifest_missing" in hosted["blockers"]

    ready_pack = proof_contracts.build_proof_pack_manifest(
        scene_id="scene-1",
        capture_id="capture-1",
        site_submission_id="site-submission-1",
        opportunity_id="opp-1",
        site_package_manifest={"status": "ready", "site_labeling": {"site_scope": "exact_site"}},
        rights_review={"status": "cleared"},
        hosted_review_readiness={"status": "ready"},
    )
    assert ready_pack["status"] == "ready"
    assert ready_pack["proof_pack_ready"] is True
    blocked_pack = proof_contracts.build_proof_pack_manifest(
        scene_id="scene-1",
        capture_id="capture-1",
        site_submission_id=None,
        opportunity_id=None,
        site_package_manifest={"status": "blocked"},
        rights_review={"status": "blocked"},
        hosted_review_readiness={"status": "blocked"},
    )
    assert blocked_pack["status"] == "blocked"
    assert blocked_pack["blockers"] == ["site_package_not_ready", "rights_review:blocked"]
