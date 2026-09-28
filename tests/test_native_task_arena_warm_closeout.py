from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_warm_closeout import (
    materialize_expired_warm_closeout,
    materialize_failed_watchdog_recovery_closeout,
)
from blueprint_pipeline.native_task_arena_paid_authority import (
    materialize_native_task_arena_provider_zero,
    validate_terminal_spend_chain,
)


CLOSEOUT_SCRIPT = "scripts/close_native_task_arena_expired_warm_session.py"


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    authority = {
        "schema_version": "native_task_arena_paid_attempt_authority.v1",
        "bundle_sha256": "sha256:" + "a" * 64,
        "hard_attempt_spend_cap_usd": 2.0,
        "maximum_single_resource_ttl_seconds": 7200,
        "aggregate_goal_spend_before_attempt_usd": 0.0,
        "aggregate_goal_spend_cap_usd": 10.0,
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    authority_path = _write(tmp_path / "authority.json", authority)
    watchdog_path = _write(
        tmp_path / "watchdog.json",
        {
            "schema_version": "groot_oscar_runpod_canary_watchdog.v1",
            "provider": "vast",
            "status": "provider_terminal",
            "completed_at": "2026-08-22T06:52:00+00:00",
            "provider_absence_confirmed": True,
            "final_inventory": {"api_confirmed": True, "live_resource_count": 0},
            "recorded_vast_instance_teardown": {
                "instance_id": 42,
                "provider_absence_confirmed": True,
            },
            "terminations": [{"instance_id": 42, "status": "absent"}],
        },
    )
    guard_path = _write(
        tmp_path / "guard.json",
        {
            "schema_version": "gpu_spend_guard.v1",
            "generated_at": "2026-08-22T06:53:00+00:00",
            "status": "passed",
            "blockers": [],
            "provider_zero_verified": True,
            "live_instance_count": 0,
            "inventory_results": [
                {"provider": "vast", "status": "succeeded", "row_count": 0}
            ],
        },
    )
    adapter_path = _write(
        tmp_path / "adapter.json",
        {
            "schema_version": "vast_provider_adapter_result.v1",
            "status": "completed",
            "vast_instance_ids": [42],
            "retained_owned": True,
            "continuing_spend_from_this_run": True,
        },
    )
    teardown_path = _write(
        tmp_path / "teardown.json",
        {
            "schema_version": "vast_teardown_manifest.v1",
            "status": "retained_owned",
            "vast_instance_ids": [42],
            "continuing_spend_from_this_run": True,
        },
    )
    cleanup_path = _write(
        tmp_path / "cleanup.json",
        {"all_objects_absent": True, "signed_url_files_removed": True},
    )
    result_path = _write(
        tmp_path / "result.json",
        {
            "schema_version": "native_task_arena_vast_run.v1",
            "status": "blocked",
            "authorization_consumption": {
                "status": "consumed",
                "authorization_digest": authority["authorization_digest"],
            },
            "bundle_sha256": "sha256:" + "a" * 64,
            "hard_cap_usd": 2.0,
            "hard_ttl_seconds": 7200,
            "retry_cap": 0,
            "estimated_cost_usd": 0.2,
            "all_staged_objects_absent": True,
            "continuing_spend_from_this_run": True,
            "warm_session": {
                "status": "ready",
                "instance_id": 42,
                "continuing_spend": True,
                "watchdog_pid": 100,
                "watchdog_deadline_epoch": 2_000.0,
                "watchdog_out_dir": str(watchdog_path.parent),
                "watchdog_pod_name_prefix": "blueprint-original-",
            },
            "adapter_result_path": str(adapter_path),
            "teardown_manifest_path": str(teardown_path),
            "watchdog_receipt_path": str(watchdog_path),
            "object_store_cleanup_path": str(cleanup_path),
        },
    )
    return authority_path, result_path, guard_path


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _recovery_fixture(tmp_path: Path) -> dict[str, object]:
    authority, retained, _guard = _fixture(tmp_path)
    retained_value = json.loads(retained.read_text())
    watchdog_path = Path(retained_value["watchdog_receipt_path"])
    _write(
        watchdog_path,
        {
            "schema_version": "groot_oscar_runpod_canary_watchdog.v1",
            "provider": "vast",
            "status": "armed",
            "pid": 100,
            "deadline_epoch": 2_000.0,
            "independent_process": True,
            "provider_mutations_performed": 0,
            "raw_secret_values_recorded": False,
        },
    )
    call_input = (
        "VastRenderProvider; p=VastRenderProvider(); "
        'p.terminate(\\"42\\")'
    )
    inspect_input = (
        "VastRenderProvider; p=VastRenderProvider(); "
        'p.inspect(\\"42\\"); p.billable_inventory(name_prefix=\\"\\")'
    )
    session_rows = [
        {
            "timestamp": "2026-08-22T06:52:00Z",
            "type": "response_item",
            "payload": {"type": "custom_tool_call", "input": call_input},
        },
        {
            "timestamp": "2026-08-22T06:52:02Z",
            "type": "event_msg",
            "payload": {
                "item": {
                    "type": "CommandExecution",
                    "exit_code": 0,
                    "stdout": '{"http": 200, "status": "stopped"}\n',
                    "stderr": "",
                }
            },
        },
        {
            "timestamp": "2026-08-22T06:52:07Z",
            "type": "response_item",
            "payload": {"type": "custom_tool_call", "input": inspect_input},
        },
        {
            "timestamp": "2026-08-22T06:52:09Z",
            "type": "event_msg",
            "payload": {
                "item": {
                    "type": "CommandExecution",
                    "exit_code": 0,
                    "stdout": json.dumps(
                        {
                            "inspect": {
                                "status": "absent",
                                "provider": "vast",
                                "instance_id": "42",
                                "api_confirmed": True,
                                "provider_absence_confirmed": True,
                                "raw_provider_response_recorded": False,
                            },
                            "inventory": {
                                "status": "observed",
                                "provider": "vast",
                                "name_prefix": "",
                                "api_confirmed": True,
                                "live_resource_count": 0,
                                "resources": [],
                                "raw_provider_response_recorded": False,
                            },
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    "stderr": "",
                }
            },
        },
    ]
    session_excerpt = tmp_path / "termination-session.jsonl"
    session_excerpt.write_text(
        "".join(json.dumps(row) + "\n" for row in session_rows), encoding="utf-8"
    )
    absence_paths: list[Path] = []
    for index, observed_at in enumerate(
        ("2026-08-22T06:52:12+00:00", "2026-08-22T06:52:14+00:00"),
        start=1,
    ):
        absence = {
            "schema_version": "vast_exact_instance_absence_observation.v1",
            "observed_at": observed_at,
            "provider": "vast",
            "instance_id": 42,
            "inspect_result": {
                "status": "absent",
                "provider": "vast",
                "instance_id": "42",
                "api_confirmed": True,
                "provider_absence_confirmed": True,
                "raw_provider_response_recorded": False,
            },
            "raw_secret_values_recorded": False,
            "receipt_digest": "",
        }
        absence["receipt_digest"] = canonical_digest(
            absence, digest_field="receipt_digest"
        )
        absence_paths.append(_write(tmp_path / f"absence-{index}.json", absence))
    zero = {
        "schema_version": "adp_paid_provider_zero.v1",
        "provider": "vast",
        "observed_at_utc": "2026-08-22T06:52:20+00:00",
        "api_command": [
            "blueprint_pipeline.gpu_render_providers.VastRenderProvider.billable_inventory",
            "name_prefix=",
        ],
        "api_confirmed": True,
        "global_live_resource_count": 0,
        "provider_zero": True,
        "inventory": [],
        "stderr_present": False,
        "raw_secret_values_recorded": False,
        "provider_zero_digest": "",
    }
    zero["provider_zero_digest"] = canonical_digest(
        zero, digest_field="provider_zero_digest"
    )
    zero_path = _write(tmp_path / "provider-zero.json", zero)
    billing_path = _write(
        tmp_path / "billing.json",
        {"results": [{"source": "instance-42", "amount": 0.344}]},
    )
    billing_source_path = _write(
        tmp_path / "billing-source.json",
        {
            "status": "reconciled",
            "sources": [
                {
                    "provider": "vast",
                    "retained_path": str(billing_path),
                    "response_digest": _file_sha256(billing_path),
                    "response_size_bytes": billing_path.stat().st_size,
                }
            ],
        },
    )
    return {
        "authority": authority,
        "retained": retained,
        "session_excerpt": session_excerpt,
        "absence_paths": absence_paths,
        "zero": zero_path,
        "billing": billing_path,
        "billing_source": billing_source_path,
    }


def test_materializes_terminal_derivatives_without_provider_mutation(tmp_path: Path) -> None:
    authority, retained, guard = _fixture(tmp_path)
    output = tmp_path / "closeout"

    receipt = materialize_expired_warm_closeout(
        authority_path=authority,
        retained_result_path=retained,
        provider_zero_guard_path=guard,
        output_dir=output,
    )

    result = json.loads((output / "adp_arena_vast_result.json").read_text())
    teardown = json.loads((output / "vast_teardown_manifest.json").read_text())
    adapter = json.loads((output / "vast_provider_adapter_result.json").read_text())
    assert receipt["status"] == "completed"
    assert receipt["provider_mutation_performed"] is False
    assert result["status"] == "blocked"
    assert result["continuing_spend_from_this_run"] is False
    assert result["warm_session"]["continuing_spend"] is False
    assert teardown["status"] == "completed"
    assert teardown["provider_instance_absent"] is True
    assert adapter["retained_owned"] is False
    assert adapter["continuing_spend_from_this_run"] is False
    provider_zero = materialize_native_task_arena_provider_zero(
        authority_path=authority,
        result_path=output / "adp_arena_vast_result.json",
        output_path=tmp_path / "provider_zero.json",
    )
    assert provider_zero["provider_zero_confirmed"] is True
    assert provider_zero["continuing_spend_from_this_run"] is False


def test_refuses_nonterminal_watchdog(tmp_path: Path) -> None:
    authority, retained, guard = _fixture(tmp_path)
    result = json.loads(retained.read_text())
    watchdog = Path(result["watchdog_receipt_path"])
    value = json.loads(watchdog.read_text())
    value["status"] = "armed"
    watchdog.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="expired_warm_closeout_invalid"):
        materialize_expired_warm_closeout(
            authority_path=authority,
            retained_result_path=retained,
            provider_zero_guard_path=guard,
            output_dir=tmp_path / "closeout",
        )


def test_materializes_after_exact_watchdog_supersession(tmp_path: Path) -> None:
    authority, retained, guard = _fixture(tmp_path)
    retained_value = json.loads(retained.read_text())
    original_watchdog = Path(retained_value["watchdog_receipt_path"])
    _write(
        original_watchdog,
        {
            "schema_version": "groot_oscar_runpod_canary_watchdog.v1",
            "status": "armed",
            "provider": "vast",
            "pid": 100,
            "deadline_epoch": 2_000.0,
            "watchdog_out_dir": str(original_watchdog.parent),
            "pod_name_prefix": "blueprint-original-",
        },
    )
    successor_dir = tmp_path / "successor" / "independent_vast_watchdog"
    successor_dir.mkdir(parents=True)
    (successor_dir / "started_vast_instance_id.txt").write_text(
        "42\n", encoding="utf-8"
    )
    successor_watchdog = _write(
        successor_dir / "groot_oscar_runpod_canary_watchdog.json",
        {
            "schema_version": "groot_oscar_runpod_canary_watchdog.v1",
            "status": "provider_terminal",
            "provider": "vast",
            "pid": 200,
            "deadline_epoch": 3_000.0,
            "watchdog_out_dir": str(successor_dir),
            "completed_at": "2026-08-22T06:52:00+00:00",
            "provider_absence_confirmed": True,
            "final_inventory": {"api_confirmed": True, "live_resource_count": 0},
            "recorded_vast_instance_teardown": {
                "instance_id": 42,
                "provider_absence_confirmed": True,
            },
            "terminations": [{"instance_id": 42, "status": "absent"}],
        },
    )
    inspection = {
        "api_confirmed": True,
        "instance_id": "42",
        "actual_status": "running",
    }
    supersession = _write(
        tmp_path / "successor" / "vast_independent_watchdog_supersession.json",
        {
            "schema_version": "vast_independent_watchdog_supersession.v1",
            "status": "superseded",
            "instance_id": 42,
            "predecessor_watchdog_pid": 100,
            "predecessor_watchdog_deadline_epoch": 2_000.0,
            "predecessor_watchdog_retired": True,
            "successor_watchdog_pid": 200,
            "successor_watchdog_deadline_epoch": 3_000.0,
            "successor_watchdog_out_dir": str(successor_dir),
            "provider_inspect_before": inspection,
            "provider_inspect_successor_armed": inspection,
            "provider_inspect_after_transfer": inspection,
            "provider_instance_running_after_transfer": True,
        },
    )

    receipt = materialize_expired_warm_closeout(
        authority_path=authority,
        retained_result_path=retained,
        provider_zero_guard_path=guard,
        watchdog_supersession_path=supersession,
        successor_watchdog_path=successor_watchdog,
        output_dir=tmp_path / "closeout",
    )

    assert receipt["status"] == "completed"
    assert receipt["watchdog_supersession"]["path"] == str(supersession)


def test_refuses_supersession_bound_to_different_instance(tmp_path: Path) -> None:
    authority, retained, guard = _fixture(tmp_path)
    successor_dir = tmp_path / "successor" / "independent_vast_watchdog"
    successor_dir.mkdir(parents=True)
    (successor_dir / "started_vast_instance_id.txt").write_text(
        "42\n", encoding="utf-8"
    )
    successor = _write(
        successor_dir / "groot_oscar_runpod_canary_watchdog.json",
        {"schema_version": "groot_oscar_runpod_canary_watchdog.v1"},
    )
    supersession = _write(
        tmp_path / "supersession.json",
        {
            "schema_version": "vast_independent_watchdog_supersession.v1",
            "status": "superseded",
            "instance_id": 99,
        },
    )

    with pytest.raises(ValueError, match="expired_warm_closeout_invalid"):
        materialize_expired_warm_closeout(
            authority_path=authority,
            retained_result_path=retained,
            provider_zero_guard_path=guard,
            watchdog_supersession_path=supersession,
            successor_watchdog_path=successor,
            output_dir=tmp_path / "closeout",
        )


def test_materializes_failed_watchdog_recovery_without_rewriting_watchdog(
    tmp_path: Path,
) -> None:
    fixture = _recovery_fixture(tmp_path)
    retained = Path(fixture["retained"])
    retained_value = json.loads(retained.read_text())
    watchdog_path = Path(retained_value["watchdog_receipt_path"])
    original_watchdog_bytes = watchdog_path.read_bytes()
    output = tmp_path / "recovery-closeout"

    receipt = materialize_failed_watchdog_recovery_closeout(
        authority_path=fixture["authority"],
        retained_result_path=retained,
        termination_session_excerpt_path=fixture["session_excerpt"],
        exact_absence_observation_paths=fixture["absence_paths"],
        provider_zero_path=fixture["zero"],
        official_billing_response_path=fixture["billing"],
        provider_billing_source_receipt_path=fixture["billing_source"],
        output_dir=output,
    )

    terminal = json.loads((output / "adp_arena_vast_result.json").read_text())
    teardown = json.loads((output / "vast_teardown_manifest.json").read_text())
    recovery = json.loads(
        (output / "native_task_arena_failed_watchdog_recovery.v1.json").read_text()
    )
    arena_zero_path = output / "native_task_arena_provider_zero.v1.json"
    arena_zero = json.loads(arena_zero_path.read_text())
    assert receipt["status"] == "completed"
    assert receipt["official_billing_amount_usd"] == 0.344
    assert recovery["watchdog_performed_teardown"] is False
    assert recovery["provider_mutation_performed_by_materializer"] is False
    assert terminal["status"] == "blocked"
    assert terminal["continuing_spend_from_this_run"] is False
    assert teardown["status"] == "completed"
    assert teardown["provider_instance_absent"] is True
    assert arena_zero["status"] == "completed_recovered_provider_zero"
    assert arena_zero["provider_zero_confirmed"] is True
    assert arena_zero["continuing_spend_from_this_run"] is False
    chain = validate_terminal_spend_chain(
        authority_path=fixture["authority"],
        result_path=output / "adp_arena_vast_result.json",
        provider_zero_path=arena_zero_path,
    )
    assert chain["attempt_cost_usd"] == 0.2
    assert watchdog_path.read_bytes() == original_watchdog_bytes


def test_failed_watchdog_recovery_refuses_forged_termination_response(
    tmp_path: Path,
) -> None:
    fixture = _recovery_fixture(tmp_path)
    session_excerpt = Path(fixture["session_excerpt"])
    rows = [json.loads(line) for line in session_excerpt.read_text().splitlines()]
    rows[1]["payload"]["item"]["stdout"] = (
        '{"http": 200, "status": "stopped", "instance_id": 99}\n'
    )
    session_excerpt.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(
        ValueError, match="failed_watchdog_recovery_session_excerpt_invalid"
    ):
        materialize_failed_watchdog_recovery_closeout(
            authority_path=fixture["authority"],
            retained_result_path=fixture["retained"],
            termination_session_excerpt_path=session_excerpt,
            exact_absence_observation_paths=fixture["absence_paths"],
            provider_zero_path=fixture["zero"],
            official_billing_response_path=fixture["billing"],
            provider_billing_source_receipt_path=fixture["billing_source"],
            output_dir=tmp_path / "recovery-closeout",
        )
