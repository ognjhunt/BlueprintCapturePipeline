from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from blueprint_pipeline import groot_oscar_runpod_volume_watchdog as volume_watchdog
from blueprint_pipeline.groot_oscar_runpod_volume_watchdog import (
    _extract_id,
    _matching_resources,
    _watchdog_process_running,
    watchdog,
)


def test_watchdog_inventory_failure_is_not_treated_as_zero(monkeypatch) -> None:
    monkeypatch.setattr(
        volume_watchdog,
        "_runpod_call",
        lambda method, path, body, **kwargs: (503, {}),
    )
    pods, volumes, verified = _matching_resources(
        key="secret",
        pod_prefix=volume_watchdog.POD_NAME_PREFIX,
        volume_prefix=volume_watchdog.VOLUME_NAME_PREFIX,
    )
    assert pods == []
    assert volumes == []
    assert verified is False


def test_watchdog_global_inventory_counts_unrelated_names(monkeypatch) -> None:
    def fake_call(method, path, body, **kwargs):
        del method, body, kwargs
        if path == "/pods":
            return 200, [{"id": "other-pod", "name": "unrelated-worker"}]
        return 200, [{"id": "other-volume", "name": "unrelated-cache"}]

    monkeypatch.setattr(volume_watchdog, "_runpod_call", fake_call)
    pods, volumes, verified = _matching_resources(
        key="secret",
        pod_prefix=None,
        volume_prefix=None,
    )
    assert verified is True
    assert pods == ["other-pod"]
    assert volumes == ["other-volume"]


def test_watchdog_rejects_provider_ids_that_can_escape_urls() -> None:
    assert _extract_id({"id": "safe-pod_123"}) == "safe-pod_123"
    assert _extract_id({"id": "../../pods/other"}) == ""
    assert _extract_id({"id": True}) == ""


def test_watchdog_emits_nonce_bound_armed_handoff(tmp_path: Path) -> None:
    state = tmp_path / "watchdog_state.json"
    state.write_text(
        json.dumps(
            {
                "deadline_epoch": time.time() + 120,
                "pod_name_prefix": "blueprint-storage-only-no-pod-test",
                "volume_name": "blueprint-groot-oscar-models-test",
                "watchdog_nonce": "nonce-for-test",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "watchdog_handoff.json").write_text(
        json.dumps({"status": "cancelled_before_provider_allocation"}),
        encoding="utf-8",
    )
    assert watchdog(state_path=state) == 0
    armed = json.loads((tmp_path / "watchdog_armed.json").read_text())
    assert armed["status"] == "armed"
    assert armed["pid"] == os.getpid()
    assert armed["watchdog_nonce"] == "nonce-for-test"


def test_ready_handoff_still_deletes_volume_at_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = tmp_path / "watchdog_state.json"
    state.write_text(
        json.dumps(
            {
                "deadline_epoch": time.time() + 0.02,
                "pod_name_prefix": "blueprint-storage-only-no-pod-test",
                "volume_name": "blueprint-groot-oscar-models-test",
                "watchdog_nonce": "nonce-for-ready-test",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "watchdog_handoff.json").write_text(
        json.dumps({"status": "volume_ready_watchdog_retained"}),
        encoding="utf-8",
    )

    class Provider:
        @staticmethod
        def _key() -> str:
            return "runpod-test-key"

    inventories = iter([([], ["volume-1"], True), ([], [], True), ([], [], True)])
    deleted: list[str] = []
    monkeypatch.setattr(volume_watchdog, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(
        volume_watchdog,
        "_matching_resources",
        lambda **_kwargs: next(inventories),
    )
    monkeypatch.setattr(
        volume_watchdog,
        "_delete_volume",
        lambda **kwargs: deleted.append(kwargs["volume_id"])
        or {"provider_absence_confirmed": True},
    )
    monkeypatch.setattr(volume_watchdog.time, "sleep", lambda _seconds: None)

    assert watchdog(state_path=state) == 0
    assert deleted == ["volume-1"]
    result = json.loads((tmp_path / "watchdog_result.json").read_text())
    assert result["status"] == "provider_terminal"


def test_stale_watchdog_does_not_mutate_after_retention_rotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = tmp_path / "watchdog_state.json"
    state.write_text(
        json.dumps(
            {
                "deadline_epoch": time.time() - 1,
                "pod_name_prefix": "blueprint-storage-only-no-pod-old",
                "volume_name": "blueprint-groot-oscar-models-old",
                "watchdog_nonce": "old-nonce",
                "provider_lane_handoff": {
                    "lease_path": str(tmp_path / "provider.lease.json"),
                    "binding": {
                        "provider": "runpod",
                        "lane": "groot_oscar_model_volume",
                        "volume_id": "volume-1",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        volume_watchdog,
        "claim_transferred_paid_provider_lane_teardown",
        lambda **_kwargs: {"status": "ownership_transferred"},
    )
    monkeypatch.setattr(
        volume_watchdog,
        "get_render_provider",
        lambda _name: (_ for _ in ()).throw(
            AssertionError("provider inventory reached from stale watchdog")
        ),
    )

    assert watchdog(state_path=state) == 0
    result = json.loads((tmp_path / "watchdog_result.json").read_text())
    assert result["status"] == "ownership_transferred_no_mutation"
    assert result["provider_mutations_performed"] == 0


def test_dead_watchdog_process_is_not_running() -> None:
    class Process:
        @staticmethod
        def poll() -> int:
            return 1

    assert _watchdog_process_running(Process()) is False
