from __future__ import annotations

import json
import sys
from pathlib import Path

from blueprint_pipeline.provider_worker_contract import (
    PROVIDER_WORKER_CONTRACT_SCHEMA_VERSION,
    build_provider_worker_contract,
    classify_policy_worker_command,
    main,
    write_provider_worker_contract,
)

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def test_provider_worker_contract_blocks_one_shot_vast_policy_launcher(monkeypatch) -> None:
    monkeypatch.delenv("BLUEPRINT_ALLOW_PROVIDER_LAUNCH_PER_POLICY_INFERENCE", raising=False)
    command = (
        f"{sys.executable} -m "
        "blueprint_pipeline.unitree_groot_n17_sonic_vast_policy_command"
    )

    classification = classify_policy_worker_command(command)

    assert classification["status"] == "blocked"
    assert classification["invocation_kind"] == "one_shot_provider_launcher"
    assert classification["repeated_policy_loop_allowed"] is False
    assert classification["provider_instance_launch_per_inference"] is True
    assert (
        "one_shot_provider_launcher_not_allowed_for_repeated_policy_loop"
        in classification["blockers"]
    )
    assert classification["command_value_redacted"] == "<configured>"


def test_provider_worker_contract_allows_one_shot_vast_policy_launcher_with_explicit_opt_in(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_PROVIDER_LAUNCH_PER_POLICY_INFERENCE", "true")
    command = (
        f"{sys.executable} -m "
        "blueprint_pipeline.unitree_groot_n17_sonic_vast_policy_command"
    )

    classification = classify_policy_worker_command(command)

    assert classification["status"] == "compatible"
    assert classification["invocation_kind"] == "one_shot_provider_launcher"
    assert classification["repeated_policy_loop_allowed"] is True
    assert classification["provider_instance_launch_per_inference"] is True
    assert classification["blockers"] == []
    assert (
        "provider_instance_launch_per_policy_inference_explicitly_allowed"
        in classification["warnings"]
    )


def test_provider_worker_contract_allows_http_worker_endpoint() -> None:
    classification = classify_policy_worker_command("https://worker.example.test/infer")

    assert classification["status"] == "compatible"
    assert classification["invocation_kind"] == "http_worker_endpoint"
    assert classification["repeated_policy_loop_allowed"] is True
    assert classification["provider_instance_launch_per_inference"] is False


def test_provider_worker_contract_classifies_http_worker_adapter_command() -> None:
    classification = classify_policy_worker_command(
        "blueprint-provider-worker-policy-command-adapter"
    )

    assert classification["status"] == "compatible"
    assert classification["invocation_kind"] == "persistent_backend_client_command"
    assert classification["repeated_policy_loop_allowed"] is True
    assert classification["provider_instance_launch_per_inference"] is False


def test_write_provider_worker_contract_redacts_command_and_keeps_claim_boundary(
    tmp_path: Path,
) -> None:
    contract = write_provider_worker_contract(
        output_dir=tmp_path,
        generated_at="now",
        provider="vast",
        worker_role="unitree_policy_action_worker",
        policy_command="blueprint-run-vast-provider-adapter --allow-paid-vast-launch",
    )
    loaded = json.loads(
        (tmp_path / "provider_worker_contract.json").read_text(encoding="utf-8")
    )

    assert loaded == contract
    assert contract["schema_version"] == PROVIDER_WORKER_CONTRACT_SCHEMA_VERSION
    assert contract["http_contract"]["canonical"]["ready"]["path"] == "/readyz"
    assert contract["http_contract"]["canonical"]["infer"]["path"] == "/infer"
    assert (
        contract["policy_command_classification"]["command_value_redacted"]
        == "<configured>"
    )
    assert contract["claim_boundary"]["contract_artifact_is_not_provider_execution_proof"] is True
    assert "allow-paid" not in json.dumps(contract)


def test_provider_worker_contract_cli_uses_policy_command_env(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("UNIT_TEST_POLICY_WORKER_COMMAND", "http://127.0.0.1:8765/infer")

    assert main(
        [
            "--output-dir",
            str(tmp_path),
            "--provider",
            "runpod",
            "--policy-command-env",
            "UNIT_TEST_POLICY_WORKER_COMMAND",
        ]
    ) == 0

    contract = json.loads(
        (tmp_path / "provider_worker_contract.json").read_text(encoding="utf-8")
    )
    assert contract["provider"] == "runpod"
    assert (
        contract["policy_command_classification"]["invocation_kind"]
        == "http_worker_endpoint"
    )


def test_build_provider_worker_contract_records_provider_portability() -> None:
    contract = build_provider_worker_contract(
        generated_at="now",
        provider="provider_neutral",
        policy_command="blueprint-unitree-unifolm-vla-server-bridge",
    )

    assert "vast" in contract["provider_portability"]["same_contract_for"]
    assert (
        contract["policy_command_classification"]["invocation_kind"]
        == "persistent_backend_client_command"
    )
