import json
import subprocess
import sys
from pathlib import Path


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
    assert evidence["schema_version"] == "preloaded_worker_image.v1"
    assert evidence["source_sha"] == "5" * 40
    assert evidence["image_digest"] == "sha256:" + "7" * 64
    assert evidence["allocation_key"] == "blueprint-g4"
    assert evidence["image_present_before_allocation"] is True
    assert evidence["local_digest_inspect_passed"] is True
    assert evidence["runtime_health_preflight_passed"] is True
    assert evidence["cold_pull_required_during_campaign"] is False
    assert len(evidence["host_self_test_sha256"]) == 64
    assert len(evidence["runtime_health_sha256"]) == 64


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
