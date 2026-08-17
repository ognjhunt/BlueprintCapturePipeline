from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import blueprint_pipeline.content_agents_preallocation_closeout as closeout
from blueprint_pipeline.common import write_json
from blueprint_pipeline.content_agents_preallocation_closeout import (
    bind_prior_content_agents_preallocation_attempts,
    main,
    materialize_content_agents_preallocation_provider_zero,
    validate_bound_prior_content_agents_preallocation_attempts,
    validate_content_agents_preallocation_provider_zero,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


LAUNCH_ID = (
    "adp-content-agents-840920-task-a-127ed7c8-r1-api-"
    "20260817T104117Z-a674781e"
)


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)
    return path


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    spend_root = tmp_path / "spend"
    monkeypatch.setenv("BLUEPRINT_SPEND_AUTHORITY_ROOT", str(spend_root))
    authority = {
        "schema_version": "adp_content_agents_paid_attempt_authority.v1",
        "bundle_sha256": "sha256:" + "b" * 64,
        "prior_preallocation_attempts": [],
        "preallocation_attempt_ordinal": 1,
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    watchdog = {
        "schema_version": "vast_independent_watchdog_handoff.v1",
        "generated_at": "2026-08-17T10:41:19+00:00",
        "status": "blocked",
        "independent_process": False,
        "watchdog_armed_before_allocation": False,
        "pod_name_prefix": "blueprint-adp-content-agents-",
        "provider_mutations_performed": 0,
        "blockers": ["independent_vast_watchdog_not_armed"],
        "raw_secret_values_recorded": False,
    }
    result = {
        "schema_version": "adp_content_agents_vast_run.v1",
        "generated_at": "2026-08-17T10:41:19+00:00",
        "status": "blocked",
        "provider_mutations_performed": 0,
        "all_staged_objects_absent": True,
        "independent_watchdog": watchdog,
        "blockers": ["adp_content_agents_independent_watchdog_not_armed"],
    }
    cleanup = {
        "schema_version": "wam_provider_object_store_cleanup.v1",
        "status": "completed",
        "all_objects_absent": True,
        "signed_url_files_removed": True,
    }
    guard = {
        "schema_version": "gpu_spend_guard.v1",
        "status": "passed",
        "generated_at": "2026-08-17T10:45:00+00:00",
        "reap_mode": True,
        "provider_zero_verified": True,
        "live_instance_count": 0,
        "total_burn_per_hour_usd": 0.0,
        "reap_candidate_ids": [],
        "reap_results": [],
        "inventory_results": [
            {
                "provider": provider,
                "required": True,
                "status": "succeeded",
                "row_count": 0,
            }
            for provider in ("runpod", "vast", "digitalocean")
        ],
        "provider_zero": {
            "status": "verified",
            "required_provider_ids": ["runpod", "vast", "digitalocean"],
            "global_live_instance_count": 0,
            "global_total_burn_per_hour_usd": 0.0,
        },
    }
    request_digest = "sha256:" + "r" * 64
    sync = {
        "schema_version": "task_evaluation_launch_webapp_sync_result.v1",
        "status": "succeeded",
        "launch_id": LAUNCH_ID,
        "run_id": LAUNCH_ID,
        "request_digest": request_digest,
        "receipt_digest": "sha256:" + "c" * 64,
        "response": {
            "schema_version": "task_evaluation_launch_web_sync_receipt.v1",
            "status": "blocked",
            "launch_id": LAUNCH_ID,
            "run_id": LAUNCH_ID,
            "request_digest": request_digest,
            "receipt_digest": "sha256:" + "c" * 64,
        },
        "attempt_number": 1,
        "attempted_at": "2026-08-17T10:43:00+00:00",
        "sync_result_digest": "",
    }
    sync["sync_result_digest"] = canonical_digest(
        sync, digest_field="sync_result_digest"
    )
    run = tmp_path / "state" / LAUNCH_ID
    consumption = {
        "schema_version": "adp_content_agents_paid_attempt_consumption.v1",
        "authorization_digest": authority["authorization_digest"],
        "bundle_sha256": authority["bundle_sha256"],
        "config_preflight_receipt_digest": "sha256:" + "p" * 64,
        "blueprint_commit": "127ed7c8ae0d07da091641b0089da86305a85bc3",
        "consumed_at": "2026-08-17T10:41:17+00:00",
        "maximum_provider_allocations": 1,
    }
    _write(
        spend_root
        / "consumed"
        / f"content-agents-{authority['authorization_digest'][7:]}.json",
        consumption,
    )
    return {
        "attempt_authority": _write(run / "authority.json", authority),
        "allocator_result": _write(
            run
            / "allocator"
            / "content-agents-job"
            / "adp_content_agents_vast_result.json",
            result,
        ),
        "watchdog_handoff": _write(
            run / "allocator" / "content-agents-job" / "watchdog_handoff.json",
            watchdog,
        ),
        "object_store_cleanup": _write(
            run / "allocator" / "content-agents-job" / "cleanup.json", cleanup
        ),
        "fresh_global_guard": _write(tmp_path / "gpu_spend_guard" / "latest.json", guard),
        "webapp_sync": _write(run / "webapp_sync_succeeded.json", sync),
    }


def _materialize(
    paths: dict[str, Path],
    output: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    observed_at: str = "2026-08-17T10:45:30+00:00",
) -> dict:
    monkeypatch.setattr(closeout, "utc_now_iso", lambda: observed_at)
    return materialize_content_agents_preallocation_provider_zero(
        attempt_authority_path=paths["attempt_authority"],
        allocator_result_path=paths["allocator_result"],
        watchdog_handoff_path=paths["watchdog_handoff"],
        object_store_cleanup_path=paths["object_store_cleanup"],
        fresh_global_guard_path=paths["fresh_global_guard"],
        webapp_sync_path=paths["webapp_sync"],
        output_path=output,
    )


def test_exact_production_shaped_consumed_attempt_is_zero_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "content-agents-preallocation-zero.json"
    value = _materialize(paths, output, monkeypatch)
    assert value["launch_id"] == LAUNCH_ID
    assert value["provider_allocations_performed"] == 0
    assert value["official_cost_usd"] == 0.0
    assert value["continuing_spend_from_attempt"] is False
    assert value["content_agents_completed"] is False
    assert validate_content_agents_preallocation_provider_zero(output) == value


@pytest.mark.parametrize(
    ("role", "field", "replacement"),
    [
        ("allocator_result", "provider_mutations_performed", 1),
        ("allocator_result", "instance_id", 47999999),
        ("object_store_cleanup", "all_objects_absent", False),
        ("fresh_global_guard", "live_instance_count", 1),
    ],
)
def test_closeout_rejects_allocated_nonzero_or_unclean_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    field: str,
    replacement: object,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    value = json.loads(paths[role].read_text(encoding="utf-8"))
    value[field] = replacement
    write_json(paths[role], value)
    with pytest.raises(ValueError):
        _materialize(paths, tmp_path / "refused.json", monkeypatch)


def test_closeout_reopens_sources_and_rejects_later_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "zero.json"
    _materialize(paths, output, monkeypatch)
    result = json.loads(paths["allocator_result"].read_text(encoding="utf-8"))
    result["all_staged_objects_absent"] = False
    write_json(paths["allocator_result"], result)
    with pytest.raises(ValueError, match="unbound"):
        validate_content_agents_preallocation_provider_zero(output)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema_version", "wrong_webapp_receipt.v1"),
        ("status", "completed"),
    ],
)
def test_closeout_rejects_nonexact_webapp_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: str,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    sync = json.loads(paths["webapp_sync"].read_text(encoding="utf-8"))
    sync["response"][field] = replacement
    sync["sync_result_digest"] = canonical_digest(
        sync, digest_field="sync_result_digest"
    )
    write_json(paths["webapp_sync"], sync)
    with pytest.raises(ValueError, match="webapp_sync_invalid"):
        _materialize(paths, tmp_path / "refused.json", monkeypatch)


@pytest.mark.parametrize(
    "role",
    ["allocator_result", "watchdog_handoff", "object_store_cleanup", "webapp_sync"],
)
def test_closeout_rejects_evidence_outside_the_exact_launch_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    outside = tmp_path / "unrelated-launch" / paths[role].name
    outside.parent.mkdir()
    paths[role].replace(outside)
    paths[role] = outside
    with pytest.raises(ValueError, match="launch_binding_invalid"):
        _materialize(paths, tmp_path / "refused.json", monkeypatch)


@pytest.mark.parametrize(
    ("guard_time", "observed_at"),
    [
        ("2026-08-17T10:40:00+00:00", "2026-08-17T10:45:30+00:00"),
        ("2026-08-17T10:45:00+00:00", "2026-08-17T10:50:01+00:00"),
    ],
    ids=["guard-before-terminal-evidence", "guard-stale-at-materialization"],
)
def test_closeout_rejects_guard_before_attempt_or_stale_at_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    guard_time: str,
    observed_at: str,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    guard = json.loads(paths["fresh_global_guard"].read_text(encoding="utf-8"))
    guard["generated_at"] = guard_time
    write_json(paths["fresh_global_guard"], guard)
    with pytest.raises(
        ValueError, match="content_agents_preallocation_guard_not_fresh_after_attempt"
    ):
        _materialize(
            paths, tmp_path / "refused.json", monkeypatch, observed_at=observed_at
        )


def test_successor_authority_binds_the_complete_zero_cost_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "zero.json"
    zero = _materialize(paths, output, monkeypatch)
    entries = bind_prior_content_agents_preallocation_attempts([output])
    successor = {
        "prior_preallocation_attempts": entries,
        "preallocation_attempt_ordinal": 2,
    }
    assert validate_bound_prior_content_agents_preallocation_attempts(successor) == entries
    assert entries[0]["attempt_authority_digest"] == zero["attempt_authority_digest"]
    assert entries[0]["official_cost_usd"] == 0.0
    omitted = copy.deepcopy(successor)
    omitted["prior_preallocation_attempts"] = []
    with pytest.raises(ValueError):
        validate_bound_prior_content_agents_preallocation_attempts(omitted)


def test_cli_materializes_without_provider_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "zero.json"
    argv = []
    for flag, role in (
        ("--attempt-authority", "attempt_authority"),
        ("--allocator-result", "allocator_result"),
        ("--watchdog-handoff", "watchdog_handoff"),
        ("--object-store-cleanup", "object_store_cleanup"),
        ("--fresh-global-guard", "fresh_global_guard"),
        ("--webapp-sync", "webapp_sync"),
    ):
        argv.extend([flag, str(paths[role])])
    argv.extend(["--output", str(output)])
    monkeypatch.setattr(
        closeout, "utc_now_iso", lambda: "2026-08-17T10:45:30+00:00"
    )
    assert main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["provider_allocations_performed"] == 0
    assert printed["official_cost_usd"] == 0.0
