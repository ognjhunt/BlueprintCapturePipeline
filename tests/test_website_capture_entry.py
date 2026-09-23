"""The website self-capture lane accepts an app-filmed iPhone bundle only by its server marker."""

from __future__ import annotations

import pytest

from blueprint_pipeline.intake_packet_fields import normalize_intake_packet
from blueprint_pipeline.materialization import _has_minimum_intake
from blueprint_pipeline.website_capture_entry import (
    BROWSER_SELF_CAPTURE,
    SITE_SELF_CAPTURE,
    capture_entry_source,
    is_server_authored_site_self_capture,
    is_website_capture_manifest,
    is_website_entry_source,
    website_entry_source,
)

ORIGIN = {
    "schema_version": "site_self_capture.v1",
    "authored_by": "blueprint_webapp",
    "site_filmed_itself": True,
    "capture_job_exists": False,
    "request_id": "req-1",
    "client": "ios_app_clip",
}


def _app_manifest(**origin_overrides):
    return {
        "scene_id": "site-req-1",
        "capture_id": "walkthrough-req-1",
        "capture_source": "iphone",
        "capture_profile_id": "iphone_arkit_lidar",
        "site_submission_id": "req-1",
        "site_self_capture": {**ORIGIN, **origin_overrides},
    }


def test_server_marker_routes_an_iphone_bundle_to_the_website_lane():
    manifest = _app_manifest()
    assert is_server_authored_site_self_capture(manifest)
    assert website_entry_source(manifest) == SITE_SELF_CAPTURE
    assert capture_entry_source(manifest) == SITE_SELF_CAPTURE
    assert is_website_capture_manifest(manifest)
    # iPhone treatment is keyed off the unchanged source and profile.
    assert manifest["capture_source"] == "iphone"


def test_browser_uploads_keep_their_existing_entry_source():
    manifest = {"capture_source": BROWSER_SELF_CAPTURE, "site_submission_id": "req-1"}
    assert capture_entry_source(manifest) == BROWSER_SELF_CAPTURE
    assert is_website_capture_manifest(manifest)
    assert is_website_entry_source(BROWSER_SELF_CAPTURE)
    assert is_website_entry_source(SITE_SELF_CAPTURE)


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": "site_self_capture.v2"},
        {"authored_by": "ios_app_clip"},
        {"site_filmed_itself": "true"},
        {"site_filmed_itself": 1},
        {"capture_job_exists": True},
        {"capture_job_exists": 0},
        {"request_id": "req-2"},
        {"request_id": ""},
    ],
)
def test_forged_or_partial_markers_stay_in_the_device_lane(overrides):
    manifest = _app_manifest(**overrides)
    assert not is_server_authored_site_self_capture(manifest)
    assert website_entry_source(manifest) is None
    assert capture_entry_source(manifest) == "iphone"
    assert not is_website_capture_manifest(manifest)


def test_marker_must_name_the_manifest_submission():
    manifest = _app_manifest()
    manifest.pop("site_submission_id")
    assert not is_website_capture_manifest(manifest)
    manifest = _app_manifest()
    manifest["site_self_capture"] = "site_self_capture.v1"
    assert not is_website_capture_manifest(manifest)
    assert capture_entry_source(None) == ""
    assert not is_website_entry_source("iphone")
    assert not is_website_entry_source(None)


def test_contract_snake_case_intake_is_read_like_camel_case():
    snake = {
        "schema_version": "v1",
        "workflow_name": "Tote transfer",
        "task_steps": ["Tote transfer"],
        "owner": "site_operator",
        "privacy_security_limits": ["No faces"],
    }
    normalized = normalize_intake_packet(snake)
    assert normalized["workflowName"] == "Tote transfer"
    assert normalized["taskSteps"] == ["Tote transfer"]
    assert normalized["privacySecurityLimits"] == ["No faces"]
    assert _has_minimum_intake(normalized)
    # The original keys are preserved.
    assert normalized["workflow_name"] == "Tote transfer"


def test_unconfirmed_site_intake_stays_honestly_incomplete():
    unconfirmed = {"schema_version": "v1", "workflow_name": None, "task_steps": []}
    normalized = normalize_intake_packet(unconfirmed)
    assert not _has_minimum_intake(normalized)
    assert "zone" not in normalized and "owner" not in normalized


def test_camel_case_values_are_kept_when_both_spellings_exist():
    both = {"workflowName": "Camel", "workflow_name": "Snake", "taskSteps": [], "task_steps": ["Step"]}
    normalized = normalize_intake_packet(both)
    assert normalized["workflowName"] == "Camel"
    assert normalized["taskSteps"] == ["Step"]
    assert normalize_intake_packet(None) == {}
