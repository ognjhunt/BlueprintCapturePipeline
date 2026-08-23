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


def _content_fixture(
    root: Path,
    *,
    instance_id: int = 47940042,
    amount: float = 0.366,
) -> dict[str, Path]:
    fixture = _fixture(root, instance_id=instance_id, amount=amount)
    result = json.loads(fixture["result"].read_text(encoding="utf-8"))
    authority_digest = result.pop("authorization_consumption")["authorization_digest"]
    result["schema_version"] = "adp_content_agents_vast_run.v1"
    result.pop("receipt_digest")
    fixture["result"].write_text(json.dumps(result), encoding="utf-8")
    allocation_binding = {
        "program_id": "arm-decision-proof-v1",
        "probe_kind": "adp-usd-content-agents",
        "orchestrator_source_commit": "6bcc65db104f5022ae2ad40ae606d242b846f990",
        "paid_attempt_authority_digest": authority_digest,
        "bundle_sha256": result["bundle_sha256"],
    }
    fixture["admission"] = _write(
        fixture["result"].with_name("admission.json"),
        {
            "schema_version": "paid_lane_admission.v1",
            "status": "admitted",
            "resource_class": "vast_provider_adapter",
            "blockers": [],
            "provider_mutations_performed": 0,
            "program_id": "arm-decision-proof-v1",
            "probe_kind": "adp-usd-content-agents",
            "authority": "explicit_content_agents_paid_attempt_authority_bound",
            "paid_attempt_authority_required_for_execute": True,
            "allocation_binding": allocation_binding,
            "allocation_binding_digest": canonical_digest(allocation_binding),
            "control_plane_identity": {
                "orchestrator_source_commit": allocation_binding[
                    "orchestrator_source_commit"
                ],
            },
        },
    )
    return fixture


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


def _zero_charge_absence_fixture(root: Path) -> dict[str, Path]:
    fixture = _fixture(root, instance_id=48344658, amount=0.001)
    result = json.loads(fixture["result"].read_text(encoding="utf-8"))
    result["generated_at"] = "2026-08-21T23:03:10+00:00"
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    fixture["result"].write_text(json.dumps(result), encoding="utf-8")
    teardown = json.loads(fixture["teardown"].read_text(encoding="utf-8"))
    teardown["generated_at"] = "2026-08-21T23:03:38+00:00"
    fixture["teardown"].write_text(json.dumps(teardown), encoding="utf-8")
    billing = {"results": [{"source": "instance-48342135", "amount": 0.275}]}
    fixture["billing"].write_text(json.dumps(billing), encoding="utf-8")
    billing_sha = "sha256:" + hashlib.sha256(
        fixture["billing"].read_bytes()
    ).hexdigest()
    post_source = {
        "schema_version": "blueprint.provider_billing_source_receipt.v1",
        "status": "reconciled",
        "generated_at": "2026-08-21T23:13:49+00:00",
        "provider_totals_usd": {"vast": 301.242},
        "sources": [
            {
                "provider": "vast",
                "retained_path": str(fixture["billing"].resolve()),
                "response_digest": billing_sha,
                "response_size_bytes": fixture["billing"].stat().st_size,
            }
        ],
    }
    fixture["billing_source"].write_text(
        json.dumps(_digest_bound(post_source)), encoding="utf-8"
    )
    fixture["pre_billing_source"] = _write(
        root / "pre-billing-source.json",
        _digest_bound(
            {
                "schema_version": "blueprint.provider_billing_source_receipt.v1",
                "status": "reconciled",
                "generated_at": "2026-08-21T22:50:59+00:00",
                "provider_totals_usd": {"vast": 301.242},
                "sources": [],
            }
        ),
    )
    return fixture


def test_native_arena_reconciles_provider_confirmed_zero_charge_absence(
    tmp_path: Path,
) -> None:
    fixture = _zero_charge_absence_fixture(tmp_path / "zero-charge")
    output = tmp_path / "zero-charge" / "reconciliation.json"

    value = materialize_same_goal_spend_reconciliation(
        lane="native_task_arena",
        terminal_result_paths=[fixture["result"]],
        teardown_manifest_paths=[fixture["teardown"]],
        provider_zero_paths=[fixture["zero"]],
        official_billing_response_paths=[fixture["billing"]],
        provider_billing_source_receipt_paths=[fixture["billing_source"]],
        pre_attempt_provider_billing_source_receipt_paths=[
            fixture["pre_billing_source"]
        ],
        output_path=output,
    )

    assert value["total_cost_usd"] == 0.0
    assert value["entries"][0]["provider_instance_id"] == 48344658
    assert value["entries"][0]["evidence_kind"] == (
        "official_billing_zero_charge_absence_after_grace"
    )
    assert bind_lane_prior_spend(
        prior_result_paths=[fixture["result"]],
        reconciliation_path=output,
        lane="native_task_arena",
    )["actual_total_usd"] == 0.0


def test_native_arena_reconciles_a_proven_no_allocation_attempt(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "no-allocation")
    result = json.loads(fixture["result"].read_text(encoding="utf-8"))
    result["estimated_cost_usd"] = 0.0
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    fixture["result"].write_text(json.dumps(result), encoding="utf-8")
    teardown = json.loads(fixture["teardown"].read_text(encoding="utf-8"))
    teardown["status"] = "not_required_blueprint_bundle_preflight_blocked"
    teardown["vast_instance_ids"] = []
    fixture["teardown"].write_text(json.dumps(teardown), encoding="utf-8")
    zero = json.loads(fixture["zero"].read_text(encoding="utf-8"))
    zero["inventory_scope"] = "no_provider_allocation"
    zero["receipt_digest"] = canonical_digest(zero, digest_field="receipt_digest")
    fixture["zero"].write_text(json.dumps(zero), encoding="utf-8")

    output, value = _materialize(
        tmp_path / "no-allocation", "native_task_arena", fixture
    )

    assert value["total_cost_usd"] == 0.0
    assert value["entries"][0]["provider_instance_id"] is None
    assert value["entries"][0]["evidence_kind"] == "provider_zero_no_allocation"
    binding = bind_lane_prior_spend(
        prior_result_paths=[fixture["result"]],
        reconciliation_path=output,
        lane="native_task_arena",
    )
    assert binding["actual_total_usd"] == 0.0
    assert binding["prior_terminal_attempts"][0]["provider_instance_id"] is None


@pytest.mark.parametrize("mutation", ["total_changed", "grace_too_short"])
def test_native_arena_refuses_ambiguous_zero_charge_absence(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _zero_charge_absence_fixture(tmp_path / mutation)
    if mutation == "total_changed":
        post = json.loads(fixture["billing_source"].read_text(encoding="utf-8"))
        post["provider_totals_usd"]["vast"] = 301.243
        post["receipt_digest"] = canonical_digest(post, digest_field="receipt_digest")
        fixture["billing_source"].write_text(json.dumps(post), encoding="utf-8")
    elif mutation == "grace_too_short":
        post = json.loads(fixture["billing_source"].read_text(encoding="utf-8"))
        post["generated_at"] = "2026-08-21T23:12:00+00:00"
        post["receipt_digest"] = canonical_digest(post, digest_field="receipt_digest")
        fixture["billing_source"].write_text(json.dumps(post), encoding="utf-8")
    with pytest.raises(ValueError):
        materialize_same_goal_spend_reconciliation(
            lane="native_task_arena",
            terminal_result_paths=[fixture["result"]],
            teardown_manifest_paths=[fixture["teardown"]],
            provider_zero_paths=[fixture["zero"]],
            official_billing_response_paths=[fixture["billing"]],
            provider_billing_source_receipt_paths=[fixture["billing_source"]],
            pre_attempt_provider_billing_source_receipt_paths=[
                fixture["pre_billing_source"]
            ],
            output_path=tmp_path / mutation / "blocked.json",
        )


def test_native_arena_prefers_posted_charge_over_zero_charge_absence(
    tmp_path: Path,
) -> None:
    fixture = _zero_charge_absence_fixture(tmp_path / "posted")
    billing = json.loads(fixture["billing"].read_text(encoding="utf-8"))
    billing["results"].append({"source": "instance-48344658", "amount": 0.001})
    fixture["billing"].write_text(json.dumps(billing), encoding="utf-8")
    post = json.loads(fixture["billing_source"].read_text(encoding="utf-8"))
    post["sources"][0]["response_digest"] = "sha256:" + hashlib.sha256(
        fixture["billing"].read_bytes()
    ).hexdigest()
    post["sources"][0]["response_size_bytes"] = fixture["billing"].stat().st_size
    post["receipt_digest"] = canonical_digest(post, digest_field="receipt_digest")
    fixture["billing_source"].write_text(json.dumps(post), encoding="utf-8")

    value = materialize_same_goal_spend_reconciliation(
        lane="native_task_arena",
        terminal_result_paths=[fixture["result"]],
        teardown_manifest_paths=[fixture["teardown"]],
        provider_zero_paths=[fixture["zero"]],
        official_billing_response_paths=[fixture["billing"]],
        provider_billing_source_receipt_paths=[fixture["billing_source"]],
        pre_attempt_provider_billing_source_receipt_paths=[
            fixture["pre_billing_source"]
        ],
        output_path=tmp_path / "posted" / "reconciliation.json",
    )

    assert value["total_cost_usd"] == 0.001
    assert value["entries"][0]["evidence_kind"] == "fully_bound_official_billing"


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


def test_content_agents_binds_authority_from_exact_sibling_admission(
    tmp_path: Path,
) -> None:
    fixture = _content_fixture(tmp_path / "content")

    output, value = _materialize(
        tmp_path / "content", "content_agents", fixture
    )

    entry = value["entries"][0]
    admission = next(
        source
        for source in entry["source_receipts"]
        if source["role"] == "admission"
    )
    assert Path(admission["record"]["path"]) == fixture["admission"].resolve()
    assert admission["allocation_binding_digest"] == canonical_digest(
        json.loads(fixture["admission"].read_text(encoding="utf-8"))[
            "allocation_binding"
        ]
    )
    assert entry["authority_digest"] == "sha256:" + "a" * 64
    assert entry["orchestrator_source_commit"] == (
        "6bcc65db104f5022ae2ad40ae606d242b846f990"
    )
    binding_sources = {
        binding["kind"]: binding["source_role"] for binding in entry["bindings"]
    }
    assert binding_sources["authority_digest"] == "admission"
    assert binding_sources["bundle_sha256"] == "terminal_result"
    assert bind_lane_prior_spend(
        prior_result_paths=[fixture["result"]],
        reconciliation_path=output,
        lane="content_agents",
    )["actual_total_usd"] == 0.366


def test_content_agents_ten_entry_reconciliation_accepts_latest_official_charge(
    tmp_path: Path,
) -> None:
    fixtures = [
        _content_fixture(
            tmp_path / f"attempt-{index}",
            instance_id=(47940042 if index == 9 else 47000000 + index),
            amount=(0.366 if index == 9 else 0.01),
        )
        for index in range(10)
    ]
    for index, fixture in enumerate(fixtures):
        result = json.loads(fixture["result"].read_text(encoding="utf-8"))
        result["launch_id"] = f"content-attempt-{index}"
        fixture["result"].write_text(json.dumps(result), encoding="utf-8")

    value = materialize_same_goal_spend_reconciliation(
        lane="content_agents",
        terminal_result_paths=[fixture["result"] for fixture in fixtures],
        teardown_manifest_paths=[fixture["teardown"] for fixture in fixtures],
        provider_zero_paths=[fixture["zero"] for fixture in fixtures],
        official_billing_response_paths=[fixture["billing"] for fixture in fixtures],
        provider_billing_source_receipt_paths=[
            fixture["billing_source"] for fixture in fixtures
        ],
        output_path=tmp_path / "content-ten-entry.json",
    )

    assert value["entry_count"] == 10
    assert value["entries"][-1]["provider_instance_id"] == 47940042
    assert value["entries"][-1]["cost_usd"] == 0.366
    assert value["total_cost_usd"] == pytest.approx(0.456)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "content_agents_allocator_admission_missing"),
        ("tampered_binding", "content_agents_allocator_admission_invalid"),
        ("blocked_status", "content_agents_allocator_admission_invalid"),
        ("bundle_mismatch", "content_agents_allocator_admission_invalid"),
        ("wrong_result_path", "content_agents_allocator_admission_path_invalid"),
    ],
)
def test_content_agents_refuses_unbound_allocator_admission(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    fixture = _content_fixture(tmp_path / mutation)
    if mutation == "missing":
        fixture["admission"].unlink()
    elif mutation == "tampered_binding":
        admission = json.loads(fixture["admission"].read_text(encoding="utf-8"))
        admission["allocation_binding"]["paid_attempt_authority_digest"] = (
            "sha256:" + "c" * 64
        )
        fixture["admission"].write_text(json.dumps(admission), encoding="utf-8")
    elif mutation == "blocked_status":
        admission = json.loads(fixture["admission"].read_text(encoding="utf-8"))
        admission["status"] = "blocked"
        admission["blockers"] = ["fixture_blocker"]
        fixture["admission"].write_text(json.dumps(admission), encoding="utf-8")
    elif mutation == "bundle_mismatch":
        admission = json.loads(fixture["admission"].read_text(encoding="utf-8"))
        admission["allocation_binding"]["bundle_sha256"] = "sha256:" + "c" * 64
        admission["allocation_binding_digest"] = canonical_digest(
            admission["allocation_binding"]
        )
        fixture["admission"].write_text(json.dumps(admission), encoding="utf-8")
    elif mutation == "wrong_result_path":
        wrong = _write(
            fixture["result"].parent.parent / "terminal.json",
            json.loads(fixture["result"].read_text(encoding="utf-8")),
        )
        fixture["result"] = wrong

    with pytest.raises(ValueError, match=message):
        _materialize(tmp_path / mutation, "content_agents", fixture)


def test_semantic_teacher_prior_spend_accepts_json_float_serialization_tail(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "semantic-float-tail", amount=0.013)
    output, value = _materialize(
        tmp_path / "semantic-float-tail",
        "semantic_teacher_image_edit_gpu_canary",
        fixture,
    )
    value["total_cost_usd"] = 0.013000000000000001
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    output.chmod(0o640)
    output.write_text(json.dumps(value), encoding="utf-8")

    reopened, record = _validate_prior_spend_reconciliation(
        output,
        expected_total_cost_usd=0.013,
    )

    assert reopened["total_cost_usd"] == 0.013000000000000001
    assert record["receipt_digest"] == value["receipt_digest"]


def test_semantic_teacher_prior_spend_rejects_one_nanodollar_mismatch(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "semantic-real-mismatch", amount=0.013)
    output, _ = _materialize(
        tmp_path / "semantic-real-mismatch",
        "semantic_teacher_image_edit_gpu_canary",
        fixture,
    )

    with pytest.raises(
        ValueError,
        match="semantic_teacher_prior_spend_reconciliation_invalid",
    ):
        _validate_prior_spend_reconciliation(
            output,
            expected_total_cost_usd=0.013000001,
        )


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
