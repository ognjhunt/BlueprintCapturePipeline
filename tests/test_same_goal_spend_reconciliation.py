from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_attempt_authority import bind_lane_prior_spend
from blueprint_pipeline.same_goal_spend_reconciliation import (
    SUPPORTED_LANES,
    materialize_same_goal_spend_reconciliation,
)
from blueprint_pipeline.semantic_teacher_image_edit_paid_authority import (
    _validate_prior_spend_reconciliation,
)
from blueprint_pipeline.task_evaluation_launch_dispatcher import TaskEvaluationLaunchError
from blueprint_pipeline.task_evaluation_live_profile import (
    expand_prior_spend_immutable_inputs,
)


def _write(path: Path, value: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _digest_bound(value: dict[str, object], field: str = "receipt_digest") -> dict[str, object]:
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _fixture(root: Path, *, instance_id: int = 47593142, amount: float = 0.025) -> dict[str, Path]:
    authority_digest = "sha256:" + "a" * 64
    bundle_sha256 = "sha256:" + "b" * 64
    result = _digest_bound(
        {
            "schema_version": "fixture_terminal_result.v1",
            "status": "completed",
            "launch_id": "fixture-attempt-1",
            "estimated_cost_usd": 0.015933,
            "continuing_spend_from_this_run": False,
            "bundle_sha256": bundle_sha256,
            "authorization_consumption": {"authorization_digest": authority_digest},
        }
    )
    teardown = {
        "schema_version": "vast_teardown_manifest.v1",
        "status": "completed",
        "vast_instance_ids": [instance_id],
        "continuing_spend_from_this_run": False,
    }
    zero = _digest_bound(
        {
            "schema_version": "fixture_provider_zero.v1",
            "status": "provider_zero_confirmed",
            "provider_zero_verified": True,
            "continuing_spend_from_this_run": False,
        }
    )
    result_path = _write(root / "launch" / "allocator" / "result.json", result)
    teardown_path = _write(root / "teardown.json", teardown)
    zero_path = _write(root / "zero.json", zero)
    billing_path = _write(
        root / "billing.json",
        {
            "results": [
                {"source": f"instance-{instance_id}", "amount": amount},
                {"source": "instance-999", "amount": 0.5},
            ]
        },
    )
    import hashlib

    billing_sha = "sha256:" + hashlib.sha256(billing_path.read_bytes()).hexdigest()
    billing_source = _digest_bound(
        {
            "schema_version": "blueprint.provider_billing_source_receipt.v1",
            "status": "reconciled",
            "sources": [
                {
                    "provider": "vast",
                    "retained_path": str(billing_path.resolve()),
                    "response_digest": billing_sha,
                    "response_size_bytes": billing_path.stat().st_size,
                }
            ],
        }
    )
    billing_source_path = _write(root / "billing-source.json", billing_source)
    return {
        "result": result_path,
        "teardown": teardown_path,
        "zero": zero_path,
        "billing": billing_path,
        "billing_source": billing_source_path,
    }


def _materialize(root: Path, lane: str, fixture: dict[str, Path]) -> tuple[Path, dict[str, object]]:
    output = root / "same-goal-spend.json"
    value = materialize_same_goal_spend_reconciliation(
        lane=lane,
        terminal_result_paths=[fixture["result"]],
        teardown_manifest_paths=[fixture["teardown"]],
        provider_zero_paths=[fixture["zero"]],
        official_billing_response_paths=[fixture["billing"]],
        provider_billing_source_receipt_paths=[fixture["billing_source"]],
        output_path=output,
    )
    return output, value


@pytest.mark.parametrize("lane", sorted(SUPPORTED_LANES))
def test_materializer_produces_each_issuer_lane_ledger(tmp_path: Path, lane: str) -> None:
    fixture = _fixture(tmp_path / lane)
    output, value = _materialize(tmp_path / lane, lane, fixture)

    assert value["total_cost_usd"] == 0.025
    assert value["entry_count"] == 1
    assert value["provider_mutation_performed"] is False
    assert output.stat().st_mode & 0o777 == 0o440
    binding = bind_lane_prior_spend(
        prior_result_paths=[fixture["result"]],
        reconciliation_path=output,
        lane=lane,
    )
    assert binding["actual_total_usd"] == 0.025
    assert binding["prior_terminal_attempts"][0]["estimated_cost_usd"] == 0.015933
    reopened, record = _validate_prior_spend_reconciliation(
        output,
        expected_total_cost_usd=0.025,
    )
    assert reopened["receipt_digest"] == record["receipt_digest"]


def test_materializer_accepts_semantic_teacher_terminal_shapes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "semantic")
    result = json.loads(fixture["result"].read_text(encoding="utf-8"))
    result["schema_version"] = "semantic_teacher_image_edit_vast_execution.v1"
    result["cost_usd"] = result.pop("estimated_cost_usd")
    result.pop("receipt_digest")
    # The real semantic terminal also binds its separate provider-zero receipt.
    # That field is not the terminal's self-digest and must not be selected as one.
    result["provider_zero_digest"] = "sha256:" + "c" * 64
    result["execution_digest"] = canonical_digest(result, digest_field="execution_digest")
    fixture["result"].write_text(json.dumps(result), encoding="utf-8")
    teardown = json.loads(fixture["teardown"].read_text(encoding="utf-8"))
    teardown["status"] = "PASS"
    teardown["instance_id"] = str(teardown.pop("vast_instance_ids")[0])
    fixture["teardown"].write_text(json.dumps(teardown), encoding="utf-8")
    zero = json.loads(fixture["zero"].read_text(encoding="utf-8"))
    zero["provider_zero_api_confirmed"] = zero.pop("provider_zero_verified")
    zero.pop("receipt_digest")
    zero["provider_zero_digest"] = canonical_digest(
        zero, digest_field="provider_zero_digest"
    )
    fixture["zero"].write_text(json.dumps(zero), encoding="utf-8")

    output, value = _materialize(
        tmp_path / "semantic",
        "semantic_teacher_image_edit_gpu_canary",
        fixture,
    )

    assert value["total_cost_usd"] == 0.025
    assert next(
        binding
        for binding in value["entries"][0]["bindings"]
        if binding["kind"] == "provider_zero"
    )["json_path"] == ["provider_zero_api_confirmed"]
    binding = bind_lane_prior_spend(
        prior_result_paths=[fixture["result"]],
        reconciliation_path=output,
        lane="semantic_teacher_image_edit_gpu_canary",
    )
    assert binding["actual_total_usd"] == 0.025
    assert binding["prior_terminal_attempts"][0]["estimated_cost_usd"] == 0.015933
    reopened, record = _validate_prior_spend_reconciliation(
        output,
        expected_total_cost_usd=0.025,
    )
    assert reopened["receipt_digest"] == record["receipt_digest"]


def test_live_profile_expands_every_nested_prior_spend_receipt(tmp_path: Path) -> None:
    """The dispatcher user must preflight the same files the allocator reopens."""

    fixture = _fixture(tmp_path / "fixture")
    output, _ = _materialize(tmp_path / "fixture", "gaussian_excision", fixture)
    binding = bind_lane_prior_spend(
        prior_result_paths=[fixture["result"]],
        reconciliation_path=output,
        lane="gaussian_excision",
    )
    authority = _write(
        tmp_path / "authority.json",
        {
            "prior_terminal_attempts": binding["prior_terminal_attempts"],
            "prior_spend_reconciliation": binding["reconciliation"],
            "prior_actual_provider_spend_usd": binding["actual_total_usd"],
        },
    )
    inputs = expand_prior_spend_immutable_inputs(
        [
            {
                "name": "paid_attempt_authority",
                "path": str(authority),
                "digest": "sha256:" + "0" * 64,
            }
        ]
    )

    observed = {Path(row["path"]) for row in inputs}
    assert observed == {
        authority.resolve(),
        output.resolve(),
        *(path.resolve() for path in fixture.values()),
    }
    assert len(inputs) == 7


def test_live_profile_rejects_changed_nested_prior_spend_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    output, _ = _materialize(tmp_path / "fixture", "gaussian_excision", fixture)
    binding = bind_lane_prior_spend(
        prior_result_paths=[fixture["result"]],
        reconciliation_path=output,
        lane="gaussian_excision",
    )
    authority = _write(
        tmp_path / "authority.json",
        {
            "prior_terminal_attempts": binding["prior_terminal_attempts"],
            "prior_spend_reconciliation": binding["reconciliation"],
            "prior_actual_provider_spend_usd": binding["actual_total_usd"],
        },
    )
    fixture["zero"].write_text("{}", encoding="utf-8")

    with pytest.raises(
        TaskEvaluationLaunchError,
        match="live_profile_prior_spend_dependency_invalid:paid_attempt_authority",
    ):
        expand_prior_spend_immutable_inputs(
            [
                {
                    "name": "paid_attempt_authority",
                    "path": str(authority),
                    "digest": "sha256:" + "0" * 64,
                }
            ]
        )


def test_live_profile_expands_legacy_issuer_nested_receipts(tmp_path: Path) -> None:
    """Older lane validators can keep their schema without hiding dependencies."""

    terminal = _write(tmp_path / "legacy-terminal.json", {"status": "completed"})
    terminal_record = {
        "path": str(terminal.resolve()),
        "size_bytes": terminal.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(terminal.read_bytes()).hexdigest(),
    }
    reconciliation = _write(
        tmp_path / "legacy-reconciliation.json",
        {
            "schema_version": "legacy_lane_reconciliation.v1",
            "entries": [
                {
                    "source_receipts": [
                        {"role": "terminal", "record": terminal_record}
                    ]
                }
            ],
        },
    )
    reconciliation_record = {
        "path": str(reconciliation.resolve()),
        "size_bytes": reconciliation.stat().st_size,
        "sha256": "sha256:"
        + hashlib.sha256(reconciliation.read_bytes()).hexdigest(),
    }
    authority = _write(
        tmp_path / "legacy-authority.json",
        {"prior_spend_reconciliation": reconciliation_record},
    )

    inputs = expand_prior_spend_immutable_inputs(
        [
            {
                "name": "legacy_paid_attempt_authority",
                "path": str(authority),
                "digest": "sha256:" + "0" * 64,
            }
        ]
    )

    assert {Path(row["path"]) for row in inputs} == {
        authority.resolve(),
        reconciliation.resolve(),
        terminal.resolve(),
    }


def test_cli_derives_cost_and_digests_without_handwritten_ledger(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    output = tmp_path / "ledger.json"
    command = [
        sys.executable,
        "scripts/materialize_same_goal_spend_reconciliation.py",
        "--lane",
        "retained_scene_render",
        "--terminal-result",
        str(fixture["result"]),
        "--teardown-manifest",
        str(fixture["teardown"]),
        "--provider-zero",
        str(fixture["zero"]),
        "--official-billing-response",
        str(fixture["billing"]),
        "--provider-billing-source-receipt",
        str(fixture["billing_source"]),
        "--output",
        str(output),
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path.cwd() / "src")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    summary = json.loads(completed.stdout)
    assert summary["status"] == "materialized"
    assert summary["total_cost_usd"] == 0.025
    assert output.is_file()


def test_materializer_prefers_provider_zero_schema_digest(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    zero = json.loads(fixture["zero"].read_text(encoding="utf-8"))
    zero["receipt_digest"] = "sha256:" + "0" * 64
    zero["provider_zero_receipt_digest"] = canonical_digest(
        zero, digest_field="provider_zero_receipt_digest"
    )
    fixture["zero"].write_text(json.dumps(zero), encoding="utf-8")

    _, value = _materialize(tmp_path / "fixture", "retained_scene_render", fixture)

    provider_zero = next(
        source
        for source in value["entries"][0]["source_receipts"]
        if source["role"] == "provider_zero"
    )
    assert provider_zero["digest_field"] == "provider_zero_receipt_digest"
    assert (
        provider_zero["record"]["receipt_digest"]
        == zero["provider_zero_receipt_digest"]
    )


def test_materializer_binds_legacy_simready_result_digest_by_exact_bytes(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "fixture")
    result = json.loads(fixture["result"].read_text(encoding="utf-8"))
    result["schema_version"] = "adp009b_simready_isaac_vast_run.v1"
    result["result_digest"] = "sha256:" + "1" * 64
    result.pop("receipt_digest")
    fixture["result"].write_text(json.dumps(result), encoding="utf-8")

    output, value = _materialize(tmp_path / "fixture", "simready_isaac", fixture)

    terminal = next(
        source
        for source in value["entries"][0]["source_receipts"]
        if source["role"] == "terminal_result"
    )
    assert terminal["digest_field"] is None
    assert terminal["legacy_digest_gap"] == (
        "exact_source_bytes_sha256_bound_no_canonical_digest"
    )
    assert terminal["legacy_present_digest_field"] == "result_digest"
    assert bind_lane_prior_spend(
        prior_result_paths=[fixture["result"]],
        reconciliation_path=output,
        lane="simready_isaac",
    )["actual_total_usd"] == 0.025


def test_materializer_rejects_noncanonical_result_digest_for_modern_schema(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "fixture")
    result = json.loads(fixture["result"].read_text(encoding="utf-8"))
    result["result_digest"] = "sha256:" + "1" * 64
    result.pop("receipt_digest")
    fixture["result"].write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(
        ValueError, match="same_goal_spend_source_digest_invalid:terminal_result"
    ):
        _materialize(tmp_path / "fixture", "simready_isaac", fixture)


def test_materializer_refuses_billing_not_bound_by_source_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    billing = json.loads(fixture["billing"].read_text(encoding="utf-8"))
    billing["results"][0]["amount"] = 0.5
    fixture["billing"].write_text(json.dumps(billing), encoding="utf-8")

    with pytest.raises(ValueError, match="billing_source_unbound"):
        _materialize(tmp_path / "fixture", "retained_scene_render", fixture)


def test_materializer_refuses_teardown_without_explicit_instance_id(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    teardown = json.loads(fixture["teardown"].read_text(encoding="utf-8"))
    teardown.pop("vast_instance_ids")
    fixture["teardown"].write_text(json.dumps(teardown), encoding="utf-8")

    with pytest.raises(ValueError, match="teardown_instance_ids_invalid"):
        _materialize(tmp_path / "fixture", "retained_scene_render", fixture)


def test_materializer_refuses_overwrite(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    output, _ = _materialize(tmp_path / "fixture", "retained_scene_render", fixture)
    original = output.read_bytes()

    with pytest.raises(ValueError, match="output_exists"):
        materialize_same_goal_spend_reconciliation(
            lane="retained_scene_render",
            terminal_result_paths=[fixture["result"]],
            teardown_manifest_paths=[fixture["teardown"]],
            provider_zero_paths=[fixture["zero"]],
            official_billing_response_paths=[fixture["billing"]],
            provider_billing_source_receipt_paths=[fixture["billing_source"]],
            output_path=output,
        )
    assert output.read_bytes() == original
