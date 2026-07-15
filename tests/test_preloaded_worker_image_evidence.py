import json
import subprocess
import sys
from pathlib import Path

from blueprint_pipeline.gpu_campaign_state_machine import CampaignConfig


def test_g4_self_test_converts_to_campaign_preload_evidence(tmp_path: Path):
    host = tmp_path / "host.json"
    health = tmp_path / "health.json"
    output = tmp_path / "preload.json"
    host.write_text(
        json.dumps(
            {
                "schema_version": "blueprint_g4_host_self_test.v1",
                "status": "passed",
                "worker_source_sha": "5" * 40,
                "preloaded_worker_image_digest": "sha256:" + "7" * 64,
                "image_present_before_allocation": True,
                "local_digest_inspect_passed": True,
                "cold_pull_required_during_campaign": False,
            }
        )
    )
    health.write_text(
        json.dumps(
            {
                "schema_version": "groot_oscar_closed_loop_image_healthcheck.v1",
                "status": "passed",
                "blockers": [],
                "worker_image_digest": "sha256:" + "7" * 64,
                "runtime_metadata": {"source_commit": "5" * 40},
            }
        )
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_preloaded_worker_image_evidence.py",
            "--host-self-test",
            str(host),
            "--runtime-health",
            str(health),
            "--allocation-key",
            "blueprint-g4",
            "--host-image-id",
            "g4-host-1",
            "--output",
            str(output),
        ],
        check=True,
    )
    evidence = json.loads(output.read_text())
    cfg = CampaignConfig(
        campaign_id="campaign-1",
        allocation_key="blueprint-g4",
        source_sha="5" * 40,
        image_digest="sha256:" + "7" * 64,
        hourly_rate_usd=4.5,
        max_provider_seconds=60,
        spend_authorization_usd=20,
        prior_exposure_usd=0,
        image_total_compressed_bytes=47_101_357_226,
        image_largest_layer_bytes=14_083_497_680,
        image_residency_evidence=evidence,
    )
    assert cfg.validate() == []


def test_converter_rejects_runtime_health_from_different_digest(tmp_path: Path):
    host = tmp_path / "host.json"
    health = tmp_path / "health.json"
    host.write_text(
        json.dumps(
            {
                "schema_version": "blueprint_g4_host_self_test.v1",
                "status": "passed",
                "worker_source_sha": "5" * 40,
                "preloaded_worker_image_digest": "sha256:" + "7" * 64,
            }
        )
    )
    health.write_text(
        json.dumps(
            {
                "schema_version": "groot_oscar_closed_loop_image_healthcheck.v1",
                "status": "passed",
                "blockers": [],
                "worker_image_digest": "sha256:" + "8" * 64,
                "runtime_metadata": {"source_commit": "5" * 40},
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_preloaded_worker_image_evidence.py",
            "--host-self-test",
            str(host),
            "--runtime-health",
            str(health),
            "--allocation-key",
            "blueprint-g4",
            "--host-image-id",
            "g4-host-1",
            "--output",
            str(tmp_path / "out.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "runtime health digest does not match" in result.stderr


def test_converter_rejects_failed_runtime_health(tmp_path: Path):
    host = tmp_path / "host.json"
    health = tmp_path / "health.json"
    host.write_text(
        json.dumps({"schema_version": "blueprint_g4_host_self_test.v1", "status": "passed"})
    )
    health.write_text(
        json.dumps(
            {
                "schema_version": "groot_oscar_closed_loop_image_healthcheck.v1",
                "status": "blocked",
                "blockers": ["cuda_missing"],
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_preloaded_worker_image_evidence.py",
            "--host-self-test",
            str(host),
            "--runtime-health",
            str(health),
            "--allocation-key",
            "blueprint-g4",
            "--host-image-id",
            "g4-host-1",
            "--output",
            str(tmp_path / "out.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "runtime health preflight did not pass" in result.stderr
