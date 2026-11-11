"""The upload hand-off contract between the capture repository's extractFrames and this Pipeline.

The capture repository is not in this CI, so its side is a recorded fixture
(``tests/fixtures/capture_bridge_contract_v1.json``) naming the source file and
commit it was read from. This test pins that record against what the Pipeline
itself declares and accepts: the Terraform and deploy topic names, the
messages the listener parses, and the dispatch refusal of website captures.
A change on either side that breaks one of these fails here, and the fixture
is re-recorded on purpose (docs/CAPTURE_BRIDGE_CONTRACT.md).
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

from blueprint_pipeline.pubsub_handoff_listener import parse_handoff_payload

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests" / "fixtures" / "capture_bridge_contract_v1.json").read_text())
TERRAFORM = (ROOT / "deploy" / "terraform" / "main.tf").read_text()
DEPLOY = (ROOT / "deploy" / "scripts" / "deploy.sh").read_text()


def _storage_trigger():
    spec = importlib.util.spec_from_file_location("storage_trigger_contract", ROOT / "functions" / "storage_trigger.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resource_block(kind: str, name: str) -> str:
    start = TERRAFORM.index(f'resource "{kind}" "{name}"')
    end = TERRAFORM.find('\nresource "', start + 1)
    return TERRAFORM[start:end if end > 0 else None]


def _topic(name: str) -> str:
    return re.search(r'name\s*=\s*"([^"]+)"', _resource_block("google_pubsub_topic", name)).group(1)


def test_the_fixture_names_its_source() -> None:
    capture = FIXTURE["recorded_from"]["capture"]
    assert capture["file"] == "cloud/extract-frames/src/index.ts"
    assert re.fullmatch(r"[0-9a-f]{40}", capture["commit"])
    assert re.fullmatch(r"[0-9a-f]{40}", FIXTURE["recorded_from"]["webapp"]["commit"])


def test_topic_names_agree_across_extract_frames_terraform_and_deploy() -> None:
    frames = FIXTURE["extract_frames"]
    # extractFrames publishes its full message to the swap dispatch's topic by default ...
    assert _topic("pipeline_trigger") == frames["pipeline_handoff_topic_default"]
    assert "google_pubsub_topic.pipeline_trigger.id" in _resource_block(
        "google_cloudfunctions2_function", "swap_dispatch_worker")
    assert f'SWAP_TOPIC="${{SWAP_TOPIC:-{frames["pipeline_handoff_topic_default"]}}}"' in DEPLOY
    # ... while the listener's subscription reads the bridge topic, which only this Pipeline's
    # storage trigger publishes to in the repository defaults (see the contract document).
    assert _topic("capture_bridge_handoff") == "blueprint-capture-bridge-handoff"
    assert "google_pubsub_topic.capture_bridge_handoff.id" in _resource_block(
        "google_pubsub_subscription", "pipeline_handoff_listener")
    assert 'HANDOFF_TOPIC="${HANDOFF_TOPIC:-blueprint-capture-bridge-handoff}"' in DEPLOY
    # The large-video topic exists with no subscription or consumer anywhere in this repository.
    large = _topic("large_video_ingest")
    assert large == frames["large_video_ingest_topic_default"]
    blocks = re.split(r'\n(?=(?:resource|output|variable|data|module|locals)\b)', TERRAFORM)
    referencing = sorted({block.split("{", 1)[0].strip() for block in blocks
                          if "google_pubsub_topic.large_video_ingest." in block})
    # A publisher grant and an output; no subscription, no trigger.
    assert referencing == ['output "large_video_ingest_topic"',
                           'resource "google_pubsub_topic_iam_member" "capture_extract_frames_large_video_ingest_publisher"']
    consumers = [path for path in (ROOT / "src").rglob("*.py") if large in path.read_text(encoding="utf-8")]
    assert consumers == []


def test_the_listener_accepts_extract_frames_full_message_with_its_selected_delivery() -> None:
    message = FIXTURE["browser_delivery_message"]
    handoff = parse_handoff_payload(json.dumps(message))
    assert handoff.source_finalize == message["source_finalize"]
    assert handoff.source_membership_selector == message["source_membership_selector"]
    assert handoff.pipeline_handoff_uri.endswith(
        message["source_membership_selector"]["object_name"].rsplit("/", 1)[0].rsplit("/", 1)[-1]
        + "/pipeline_handoff.json")


def test_the_storage_trigger_short_message_parses_but_carries_no_selected_delivery() -> None:
    trigger = _storage_trigger()
    payload = trigger._build_handoff_payload(
        bucket="blueprint-capture-test-bucket", scene_id="site-r1", capture_id="walkthrough-r1",
        source_finalize=FIXTURE["browser_delivery_message"]["source_finalize"])
    handoff = parse_handoff_payload(json.dumps(payload))
    assert handoff.source_finalize is not None
    # Staging a website delivery needs the selector, which only extractFrames' message carries.
    assert handoff.source_membership_selector is None
    assert handoff.pipeline_handoff_uri.endswith("/walkthrough-r1/pipeline_handoff.json")


def test_extract_frames_full_message_never_starts_the_capture_app_path() -> None:
    trigger = _storage_trigger()
    assert trigger._website_capture_named(FIXTURE["browser_delivery_message"])
    assert trigger._execute_pipeline_payload(dict(FIXTURE["browser_delivery_message"])) == "skipped:website_capture"


def test_a_video_the_webapp_accepts_but_extract_frames_blocks_leaves_a_marker() -> None:
    frames, webapp = FIXTURE["extract_frames"], FIXTURE["webapp"]
    # The WebApp's cap may exceed the ingest limit. Such an upload gets no hand-off at all, so
    # the job's only trace is the block marker, which the WebApp's no-grant sweep reads.
    if webapp["self_capture_max_upload_bytes_default"] > frames["max_inline_video_bytes_default"]:
        assert frames["large_video_block_publishes_pipeline_handoff"] is False
        assert frames["large_video_block_object"] == (
            "scenes/{scene_id}/captures/{capture_id}/large_video_ingest_blocked.json")
        assert frames["large_video_block_schema"] == "extract_frames_large_video_block.v1"
        assert frames["large_video_block_code"] == "blocked_large_video_requires_segmented_ingest"
