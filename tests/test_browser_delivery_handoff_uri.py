"""Actual synthetic Capture producer envelopes at Pipeline's admission boundary."""

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.common import PipelineError
from blueprint_pipeline.pubsub_handoff_listener import parse_handoff_payload


def producer_message():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "capture-browser-handoff-e858.json").read_text())
    assert fixture["synthetic"] is True
    return copy.deepcopy(fixture["message"])


def test_actual_browser_delivery_uri_is_bound_to_selected_membership():
    message = producer_message()
    handoff = parse_handoff_payload(message)
    assert handoff.pipeline_handoff_uri == message["pipeline_handoff_uri"]
    assert handoff.source_membership_selector == message["source_membership_selector"]


@pytest.mark.parametrize("mutation", [
    "bucket", "scene", "capture", "marker_generation", "wrong_delivery",
    "missing_selector", "wrong_membership", "traversal", "suffix", "null_source",
])
def test_selected_uri_cannot_broaden_capture_authority(mutation):
    message = producer_message()
    if mutation == "bucket":
        message["pipeline_handoff_uri"] = message["pipeline_handoff_uri"].replace("test-bucket", "other-bucket")
    elif mutation == "scene":
        message["pipeline_handoff_uri"] = message["pipeline_handoff_uri"].replace("site-r1", "site-other")
    elif mutation == "capture":
        message["pipeline_handoff_uri"] = message["pipeline_handoff_uri"].replace("walkthrough-r1", "walkthrough-other")
    elif mutation == "marker_generation":
        message["source_finalize"]["generation"] = "18446744073709551613"
    elif mutation == "wrong_delivery":
        parts = message["pipeline_handoff_uri"].split("/")
        parts[-2] = "0" * 64
        message["pipeline_handoff_uri"] = "/".join(parts)
    elif mutation == "missing_selector":
        message.pop("source_membership_selector")
    elif mutation == "wrong_membership":
        message["source_membership_selector"]["object_name"] += ".other"
    elif mutation == "traversal":
        message["pipeline_handoff_uri"] = message["pipeline_handoff_uri"].replace("/deliveries/", "/../deliveries/")
    elif mutation == "suffix":
        message["pipeline_handoff_uri"] += "?generation=1"
    elif mutation == "null_source":
        message["source_finalize"] = None
    with pytest.raises(PipelineError):
        parse_handoff_payload(message)


def test_canonical_handoff_remains_compatible():
    message = producer_message()
    message["pipeline_handoff_uri"] = f"gs://{message['bucket']}/scenes/{message['scene_id']}/captures/{message['capture_id']}/pipeline_handoff.json"
    message.pop("source_finalize")
    message.pop("source_membership_selector")
    handoff = parse_handoff_payload(message)
    assert handoff.source_finalize is None
    assert handoff.source_membership_selector is None
