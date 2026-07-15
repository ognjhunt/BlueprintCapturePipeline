"""Unified paid-lane guard: pre-spend chokepoint + crash-safe orphan reaper.

Every paid GPU lane (render, WAM async, robot-eval launcher) must route through
``require_pre_spend_preflight`` and leave a ``pending_teardown.v1`` record that a
standalone ``reap_orphans`` pass can clean up even when the launching process died.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from blueprint_pipeline import paid_lane_guard as guard
from blueprint_pipeline.paid_lane_guard import (
    PENDING_TEARDOWN_DIR_ENV,
    PENDING_TEARDOWN_SCHEMA_VERSION,
    PreSpendPreflightBlocked,
    bind_pending_teardown_instance,
    close_pending_teardown,
    image_contract_from_ref,
    load_pending_teardowns,
    open_pending_teardown,
    provider_state_from_inspect,
    reap_orphans,
    require_pre_spend_preflight,
)
from blueprint_pipeline.provider_reliability_manifest import build_teardown_proof


def _good_preflight_kwargs(**overrides) -> dict:
    kwargs = {
        "lane": "isaac_particlefield_render",
        "provider": "runpod",
        "credential_present": True,
        "capacity_evidence": {"available": True, "detail": "api_key_present"},
        "image_contract": {
            "image_ref": "nijelhunt/blueprint-capture-pipeline:sam3-ready-v7",
            "pinned": True,
        },
        "runtime_contract": {
            "startup_marker": "container_bash_started",
            "progress_marker": "bootstrap.json",
            "startup_timeout_seconds": 900,
            "no_progress_timeout_seconds": 900,
        },
        "spend_gate_open": True,
    }
    kwargs.update(overrides)
    return kwargs


class FakeProviderClient:
    """Scripted inspect/terminate provider used by reaper tests."""

    def __init__(self, inspect_results, terminate_result=None):
        self.inspect_results = list(inspect_results)
        self.terminate_result = terminate_result or {"status": "terminated", "http": 200}
        self.inspect_calls: list[str] = []
        self.terminate_calls: list[str] = []

    def inspect(self, instance_id: str) -> dict:
        self.inspect_calls.append(instance_id)
        if len(self.inspect_results) > 1:
            return self.inspect_results.pop(0)
        return self.inspect_results[0]

    def terminate(self, instance_id: str) -> dict:
        self.terminate_calls.append(instance_id)
        return self.terminate_result


class FakeRunPodVolumeClient:
    def _key(self) -> str:
        return "test-runpod-key"


def _observed(desired_status: str) -> dict:
    return {"status": "observed", "http": 200, "desiredStatus": desired_status}


def _gone() -> dict:
    return {"status": "unavailable", "http": 404}


# ---------------------------------------------------------------------------
# require_pre_spend_preflight — one fail-closed chokepoint for all lanes.
# ---------------------------------------------------------------------------


class TestRequirePreSpendPreflight:
    def test_passing_evidence_returns_preflight_with_lane(self) -> None:
        preflight = require_pre_spend_preflight(**_good_preflight_kwargs())
        assert preflight["status"] == "PASS"
        assert preflight["spend_allowed"] is True
        assert preflight["lane"] == "isaac_particlefield_render"

    def test_failing_evidence_raises_fail_closed(self) -> None:
        with pytest.raises(PreSpendPreflightBlocked) as exc_info:
            require_pre_spend_preflight(
                **_good_preflight_kwargs(capacity_evidence=None)
            )
        preflight = exc_info.value.preflight
        assert preflight["status"] == "FAIL"
        assert preflight["spend_allowed"] is False
        assert any(b.startswith("capacity_unavailable") for b in preflight["blockers"])

    def test_closed_spend_gate_raises_for_every_lane_identically(self) -> None:
        blockers_by_lane = {}
        for lane in ("isaac_particlefield_render", "runpod_wam_async", "robot_eval_provider_launcher"):
            with pytest.raises(PreSpendPreflightBlocked) as exc_info:
                require_pre_spend_preflight(
                    **_good_preflight_kwargs(lane=lane, spend_gate_open=False)
                )
            blockers_by_lane[lane] = exc_info.value.preflight["blockers"]
        assert len({tuple(b) for b in blockers_by_lane.values()}) == 1

    def test_missing_lane_fails_closed(self) -> None:
        with pytest.raises(PreSpendPreflightBlocked) as exc_info:
            require_pre_spend_preflight(**_good_preflight_kwargs(lane=""))
        assert any(
            "pre_spend_chokepoint_lane_missing" in b
            for b in exc_info.value.preflight["blockers"]
        )

    def test_record_dir_persists_preflight_artifact(self, tmp_path: Path) -> None:
        preflight = require_pre_spend_preflight(
            **_good_preflight_kwargs(), record_dir=tmp_path
        )
        recorded = json.loads(
            (tmp_path / "pre_spend_preflight.json").read_text(encoding="utf-8")
        )
        assert recorded["status"] == "PASS"
        assert recorded["lane"] == preflight["lane"]

    def test_failing_preflight_is_still_persisted(self, tmp_path: Path) -> None:
        with pytest.raises(PreSpendPreflightBlocked):
            require_pre_spend_preflight(
                **_good_preflight_kwargs(spend_gate_open=False), record_dir=tmp_path
            )
        recorded = json.loads(
            (tmp_path / "pre_spend_preflight.json").read_text(encoding="utf-8")
        )
        assert recorded["status"] == "FAIL"


class TestImageContractFromRef:
    def test_versioned_tag_is_pinned(self) -> None:
        contract = image_contract_from_ref("repo/img:v7")
        assert contract["pinned"] is True
        assert contract["image_ref"] == "repo/img:v7"

    def test_digest_is_pinned_with_digest_recorded(self) -> None:
        contract = image_contract_from_ref("repo/img@sha256:abc123")
        assert contract["pinned"] is True
        assert contract["digest"] == "sha256:abc123"

    def test_latest_tag_is_not_pinned(self) -> None:
        assert image_contract_from_ref("repo/img:latest")["pinned"] is False

    def test_empty_ref_is_not_pinned(self) -> None:
        contract = image_contract_from_ref("")
        assert contract["pinned"] is False
        assert contract["image_ref"] is None


# ---------------------------------------------------------------------------
# pending_teardown.v1 records.
# ---------------------------------------------------------------------------


class TestPendingTeardownRecords:
    def test_open_writes_v1_record_before_spend(self, tmp_path: Path) -> None:
        record = open_pending_teardown(
            provider="runpod",
            lane="runpod_wam_async",
            run_id="job-1",
            registry_dir=tmp_path,
        )
        stored = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
        assert stored["schema_version"] == PENDING_TEARDOWN_SCHEMA_VERSION
        assert stored["provider"] == "runpod"
        assert stored["lane"] == "runpod_wam_async"
        assert stored["run_id"] == "job-1"
        assert stored["instance_id"] is None
        assert stored["status"] == "open"
        assert stored["started_at"]
        assert stored["started_at_epoch"] > 0
        assert stored["max_age_seconds"] > 0

    def test_registry_dir_env_fallback(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv(PENDING_TEARDOWN_DIR_ENV, str(tmp_path / "registry"))
        record = open_pending_teardown(
            provider="runpod", lane="render", run_id="job-env"
        )
        assert Path(record["path"]).parent == tmp_path / "registry"

    def test_bind_instance_id(self, tmp_path: Path) -> None:
        record = open_pending_teardown(
            provider="runpod", lane="render", run_id="job-2", registry_dir=tmp_path
        )
        updated = bind_pending_teardown_instance(record["path"], "pod-abc")
        assert updated["instance_id"] == "pod-abc"
        stored = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
        assert stored["instance_id"] == "pod-abc"
        assert stored["status"] == "open"

    def test_close_refuses_unproven_teardown(self, tmp_path: Path) -> None:
        record = open_pending_teardown(
            provider="runpod",
            lane="render",
            run_id="job-3",
            instance_id="pod-x",
            registry_dir=tmp_path,
        )
        unproven = build_teardown_proof(
            provider="runpod",
            allocation_id="pod-x",
            terminate_requested=True,
            provider_terminal_status=None,
        )
        result = close_pending_teardown(record["path"], unproven)
        assert result["status"] == "open"
        assert result["close_refused_reason"] == "teardown_proof_not_passed"
        stored = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
        assert stored["status"] == "open"

    def test_close_accepts_proven_teardown(self, tmp_path: Path) -> None:
        record = open_pending_teardown(
            provider="runpod",
            lane="render",
            run_id="job-4",
            instance_id="pod-y",
            registry_dir=tmp_path,
        )
        proven = build_teardown_proof(
            provider="runpod",
            allocation_id="pod-y",
            terminate_requested=True,
            provider_terminal_status="terminated",
            verified_at="2026-07-04T12:00:00Z",
            status_source="provider_api",
        )
        result = close_pending_teardown(record["path"], proven)
        assert result["status"] == "closed"
        open_records = load_pending_teardowns(registry_dir=tmp_path)
        assert open_records == []
        all_records = load_pending_teardowns(registry_dir=tmp_path, include_closed=True)
        assert len(all_records) == 1
        assert all_records[0]["teardown_proof"]["status"] == "PASS"


# ---------------------------------------------------------------------------
# provider_state_from_inspect — API evidence classification for the reaper.
# ---------------------------------------------------------------------------


class TestProviderStateFromInspect:
    def test_http_404_is_api_confirmed_not_found(self) -> None:
        state = provider_state_from_inspect(_gone())
        assert state["provider_status"] == "not_found"
        assert state["api_confirmed"] is True

    def test_observed_pod_reports_desired_status(self) -> None:
        state = provider_state_from_inspect(_observed("EXITED"))
        assert state["provider_status"] == "exited"
        assert state["api_confirmed"] is True

    def test_probe_failure_is_not_api_confirmed(self) -> None:
        state = provider_state_from_inspect({"status": "blocked", "blockers": ["x"]})
        assert state["api_confirmed"] is False
        assert state["provider_status"] == ""


# ---------------------------------------------------------------------------
# reap_orphans — crash-safe teardown independent of the launching process.
# ---------------------------------------------------------------------------


def _aged_record(tmp_path: Path, *, run_id: str = "crashed-run", instance_id: str = "pod-orphan",
                 age_seconds: float = 10_000, max_age_seconds: int = 3600) -> dict:
    record = open_pending_teardown(
        provider="runpod",
        lane="runpod_wam_async",
        run_id=run_id,
        instance_id=instance_id,
        max_age_seconds=max_age_seconds,
        registry_dir=tmp_path,
    )
    path = Path(record["path"])
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["started_at_epoch"] = time.time() - age_seconds
    path.write_text(json.dumps(stored), encoding="utf-8")
    return stored | {"path": str(path)}


def _aged_network_volume_record(
    tmp_path: Path,
    *,
    run_id: str,
    instance_id: str = "",
) -> dict:
    record = open_pending_teardown(
        provider="runpod",
        lane="groot_oscar_model_volume",
        run_id=run_id,
        instance_id=instance_id,
        resource_kind="network_volume",
        resource_name="blueprint-model-cache-test",
        provider_location="US-WA-1",
        max_age_seconds=1,
        registry_dir=tmp_path,
    )
    path = Path(record["path"])
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["started_at_epoch"] = time.time() - 10_000
    path.write_text(json.dumps(stored), encoding="utf-8")
    return stored | {"path": str(path)}


class TestReapOrphans:
    def test_bound_network_volume_is_deleted_and_404_closes_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _aged_network_volume_record(
            tmp_path, run_id="bound-volume", instance_id="volume-123"
        )
        calls = []
        responses = iter([(200, {"id": "volume-123"}), (204, {}), (404, {})])

        def fake_call(method, path, body, **kwargs):
            calls.append((method, path, body, kwargs))
            return next(responses)

        monkeypatch.setattr(
            "blueprint_pipeline.gpu_render_providers._runpod_call", fake_call
        )
        report = reap_orphans(
            registry_dir=tmp_path,
            provider_clients={"runpod": FakeRunPodVolumeClient()},
        )
        assert [call[:2] for call in calls] == [
            ("GET", "/networkvolumes/volume-123"),
            ("DELETE", "/networkvolumes/volume-123"),
            ("GET", "/networkvolumes/volume-123"),
        ]
        assert report["records"][0]["outcome"] == "network_volume_deleted_and_verified"
        assert report["reaped_count"] == 1
        assert load_pending_teardowns(registry_dir=tmp_path) == []

    def test_lost_create_network_volume_is_recovered_by_exact_name_and_location(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _aged_network_volume_record(tmp_path, run_id="lost-create-volume")
        responses = iter(
            [
                (
                    200,
                    [
                        {
                            "id": "volume-recovered",
                            "name": "blueprint-model-cache-test",
                            "dataCenterId": "US-WA-1",
                        }
                    ],
                ),
                (200, {"id": "volume-recovered"}),
                (204, {}),
                (404, {}),
            ]
        )
        monkeypatch.setattr(
            "blueprint_pipeline.gpu_render_providers._runpod_call",
            lambda *args, **kwargs: next(responses),
        )
        report = reap_orphans(
            registry_dir=tmp_path,
            provider_clients={"runpod": FakeRunPodVolumeClient()},
        )
        assert report["records"][0]["instance_id"] == "volume-recovered"
        assert report["records"][0]["outcome"] == "network_volume_deleted_and_verified"
        assert load_pending_teardowns(registry_dir=tmp_path) == []

    def test_verified_zero_network_volume_matches_cancels_unbound_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _aged_network_volume_record(tmp_path, run_id="no-volume-created")
        monkeypatch.setattr(
            "blueprint_pipeline.gpu_render_providers._runpod_call",
            lambda *args, **kwargs: (200, []),
        )
        report = reap_orphans(
            registry_dir=tmp_path,
            provider_clients={"runpod": FakeRunPodVolumeClient()},
        )
        assert report["records"][0]["outcome"] == "network_volume_absence_verified"
        assert report["open_billing_risk_count"] == 0
        assert load_pending_teardowns(registry_dir=tmp_path) == []

    def test_multiple_network_volume_matches_remain_open(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _aged_network_volume_record(tmp_path, run_id="ambiguous-volume")
        rows = [
            {
                "id": volume_id,
                "name": "blueprint-model-cache-test",
                "dataCenterId": "US-WA-1",
            }
            for volume_id in ("volume-a", "volume-b")
        ]
        monkeypatch.setattr(
            "blueprint_pipeline.gpu_render_providers._runpod_call",
            lambda *args, **kwargs: (200, rows),
        )
        report = reap_orphans(
            registry_dir=tmp_path,
            provider_clients={"runpod": FakeRunPodVolumeClient()},
        )
        assert report["records"][0]["outcome"] == "network_volume_identity_unresolved"
        assert report["open_billing_risk_count"] == 1
        assert load_pending_teardowns(registry_dir=tmp_path) != []

    def test_failed_network_volume_delete_stays_open(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _aged_network_volume_record(
            tmp_path, run_id="delete-failed", instance_id="volume-stuck"
        )
        responses = iter([(200, {"id": "volume-stuck"}), (500, {}), (200, {})])
        monkeypatch.setattr(
            "blueprint_pipeline.gpu_render_providers._runpod_call",
            lambda *args, **kwargs: next(responses),
        )
        report = reap_orphans(
            registry_dir=tmp_path,
            provider_clients={"runpod": FakeRunPodVolumeClient()},
        )
        assert report["records"][0]["outcome"] == "network_volume_teardown_unverified"
        assert report["open_billing_risk_count"] == 1
        assert load_pending_teardowns(registry_dir=tmp_path) != []

    def test_crashed_launch_is_reaped_and_proven_terminal(self, tmp_path: Path) -> None:
        _aged_record(tmp_path)
        client = FakeProviderClient(
            inspect_results=[_observed("EXITED"), _gone()],
        )
        report = reap_orphans(
            registry_dir=tmp_path, provider_clients={"runpod": client}
        )
        assert client.terminate_calls == ["pod-orphan"]
        outcomes = {r["run_id"]: r for r in report["records"]}
        entry = outcomes["crashed-run"]
        assert entry["outcome"] == "reaped_terminal_proven"
        assert entry["teardown_proof"]["status"] == "PASS"
        assert entry["teardown_proof"]["provider_terminal_status"] == "not_found"
        assert report["reaped_count"] == 1
        assert report["open_billing_risk_count"] == 0
        # The record is closed so a second sweep does nothing.
        assert load_pending_teardowns(registry_dir=tmp_path) == []

    def test_not_due_records_are_left_alone(self, tmp_path: Path) -> None:
        open_pending_teardown(
            provider="runpod",
            lane="render",
            run_id="fresh-run",
            instance_id="pod-live",
            max_age_seconds=7200,
            registry_dir=tmp_path,
        )
        client = FakeProviderClient(inspect_results=[_gone()])
        report = reap_orphans(
            registry_dir=tmp_path, provider_clients={"runpod": client}
        )
        assert client.terminate_calls == []
        assert report["records"][0]["outcome"] == "not_due"
        assert load_pending_teardowns(registry_dir=tmp_path) != []

    def test_unverified_terminate_leaves_open_billing_risk(self, tmp_path: Path) -> None:
        _aged_record(tmp_path, run_id="stuck-run", instance_id="pod-stuck")
        client = FakeProviderClient(
            inspect_results=[_observed("RUNNING"), _observed("RUNNING")],
        )
        report = reap_orphans(
            registry_dir=tmp_path, provider_clients={"runpod": client}
        )
        entry = report["records"][0]
        assert entry["outcome"] == "terminate_not_proven"
        assert entry["open_billing_risk"] is True
        assert entry["teardown_proof"]["status"] == "FAIL"
        assert report["open_billing_risk_count"] == 1
        # Fail-closed: the record stays open for the next sweep.
        assert load_pending_teardowns(registry_dir=tmp_path) != []

    def test_exited_but_present_after_terminate_is_open_billing_risk(
        self, tmp_path: Path
    ) -> None:
        _aged_record(tmp_path, run_id="exited-run", instance_id="pod-exited")
        client = FakeProviderClient(
            inspect_results=[_observed("EXITED"), _observed("EXITED")],
        )
        report = reap_orphans(
            registry_dir=tmp_path, provider_clients={"runpod": client}
        )
        entry = report["records"][0]
        assert entry["open_billing_risk"] is True
        assert entry["teardown_proof"]["open_billing_risk"] is True
        assert any(
            "runpod_stopped_volume_may_continue_billing" in b
            for b in entry["teardown_proof"]["blockers"]
        )

    def test_missing_instance_id_is_unresolvable_open_risk(self, tmp_path: Path) -> None:
        _aged_record(tmp_path, run_id="no-id-run", instance_id="")
        client = FakeProviderClient(inspect_results=[_gone()])
        report = reap_orphans(
            registry_dir=tmp_path, provider_clients={"runpod": client}
        )
        entry = report["records"][0]
        assert entry["outcome"] == "unresolvable_instance_id_missing"
        assert entry["open_billing_risk"] is True
        assert client.inspect_calls == []
        assert client.terminate_calls == []

    def test_dry_run_terminates_nothing(self, tmp_path: Path) -> None:
        _aged_record(tmp_path, run_id="dry-run", instance_id="pod-dry")
        client = FakeProviderClient(inspect_results=[_observed("RUNNING")])
        report = reap_orphans(
            registry_dir=tmp_path,
            provider_clients={"runpod": client},
            dry_run=True,
        )
        entry = report["records"][0]
        assert entry["outcome"] == "would_terminate"
        assert client.terminate_calls == []
        assert load_pending_teardowns(registry_dir=tmp_path) != []

    def test_max_age_override_makes_fresh_record_due(self, tmp_path: Path) -> None:
        open_pending_teardown(
            provider="runpod",
            lane="render",
            run_id="fresh-but-forced",
            instance_id="pod-forced",
            max_age_seconds=999_999,
            registry_dir=tmp_path,
        )
        client = FakeProviderClient(inspect_results=[_gone(), _gone()])
        report = reap_orphans(
            registry_dir=tmp_path,
            provider_clients={"runpod": client},
            max_age_override_seconds=0,
        )
        assert report["records"][0]["outcome"] == "reaped_terminal_proven"


class TestReapOrphansCli:
    def test_cli_empty_registry_exits_zero(self, tmp_path: Path, capsys) -> None:
        rc = guard.main(["reap-orphans", "--registry-dir", str(tmp_path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "orphan_reap_report" in out

    def test_cli_open_billing_risk_exits_nonzero(self, tmp_path: Path) -> None:
        _aged_record(tmp_path, run_id="cli-risk", instance_id="")
        rc = guard.main(["reap-orphans", "--registry-dir", str(tmp_path)])
        assert rc == 1
