from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.provider_worker_contract import (
    INFER_PATH,
    READYZ_PATH,
    SHUTDOWN_PATH,
)
from blueprint_pipeline.provider_worker_endpoint_manifest import (
    PROVIDER_WORKER_ENDPOINT_MANIFEST_SCHEMA_VERSION,
    build_provider_worker_endpoint_manifest,
    main as endpoint_manifest_main,
    write_provider_worker_endpoint_manifest,
)


def test_direct_worker_endpoint_manifest_redacts_urls_and_exports_consumer_contract() -> None:
    manifest = build_provider_worker_endpoint_manifest(
        provider="vast",
        mode="live-startup-probe",
        job_id="job-1",
        worker_url="https://user:token@example.test:8443/infer?token=secret",
        ready_url="https://example.test:8443/readyz?token=secret",
        shutdown_url="https://example.test:8443/shutdown?token=secret",
    )

    assert manifest["schema_version"] == PROVIDER_WORKER_ENDPOINT_MANIFEST_SCHEMA_VERSION
    assert manifest["status"] == "endpoint_ready_for_policy_commands"
    assert manifest["blockers"] == []
    assert manifest["direct_policy_infer_from_local_loop_allowed"] is True
    assert manifest["known_endpoint"]["worker_url_present"] is True
    assert manifest["known_endpoint"]["worker_url_redacted"] == "https://example.test:8443/infer"
    serialized = json.dumps(manifest)
    assert "token=secret" not in serialized
    assert "user:token" not in serialized
    assert manifest["http_contract"]["readyz"]["path"] == READYZ_PATH
    assert manifest["http_contract"]["infer"]["path"] == INFER_PATH
    assert manifest["http_contract"]["shutdown"]["path"] == SHUTDOWN_PATH
    assert (
        manifest["consumer_env_contract"]["worker_url_env"]
        == "BLUEPRINT_PROVIDER_POLICY_WORKER_URL"
    )
    assert (
        manifest["consumer_env_contract"]["policy_command_adapter"]
        == "blueprint-provider-worker-policy-command-adapter"
    )
    assert manifest["claim_boundary"]["endpoint_manifest_is_not_worker_ready_proof"] is True
    assert manifest["claim_boundary"]["shutdown_response_is_not_provider_teardown_or_cost_proof"] is True


def test_runpod_serverless_manifest_is_job_submission_not_direct_infer() -> None:
    manifest = build_provider_worker_endpoint_manifest(
        provider="runpod",
        mode="serverless-run",
        job_id="job-2",
        serverless_endpoint_id="rp-endpoint",
    )

    assert manifest["status"] == "endpoint_discovery_pending_provider_runtime"
    assert manifest["worker_invocation_grain"] == "evaluation_job_provider_submission"
    assert manifest["direct_http_worker_endpoint_expected"] is False
    assert manifest["direct_policy_infer_from_local_loop_allowed"] is False
    assert manifest["known_endpoint"]["serverless_endpoint_id_present"] is True
    assert manifest["known_endpoint"]["serverless_endpoint_id_redacted"] == "<configured>"
    assert manifest["blockers"] == ["provider_worker_endpoint_not_discovered_yet"]


def test_write_and_cli_create_endpoint_manifest(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    manifest = write_provider_worker_endpoint_manifest(
        output_dir=tmp_path,
        provider="runpod",
        mode="on-demand-pod",
        job_id="job-3",
    )

    path = tmp_path / "provider_worker_endpoint_manifest.json"
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == manifest

    output_dir = tmp_path / "cli"
    assert (
        endpoint_manifest_main(
            [
                "--output-dir",
                str(output_dir),
                "--provider",
                "vast",
                "--mode",
                "dry-run",
                "--job-id",
                "job-4",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert '"provider": "vast"' in captured.out
    assert (output_dir / "provider_worker_endpoint_manifest.json").is_file()
