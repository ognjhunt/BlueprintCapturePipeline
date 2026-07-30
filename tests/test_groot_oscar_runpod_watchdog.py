import json
import os

import pytest

from blueprint_pipeline import groot_oscar_runpod_watchdog as watchdog_module
from blueprint_pipeline.groot_oscar_runpod_watchdog import (
    run_watchdog,
    terminate_canary_resources,
)
from blueprint_pipeline.production_gpu_campaign_budget import ProductionGpuCampaignBudget


class _Provider:
    def __init__(self) -> None:
        self.ids = ["pod-1", "pod-2"]

    def billable_inventory(self, *, name_prefix: str) -> dict:
        assert name_prefix == "blueprint-groot-oscar-canary-attempt-"
        return {
            "api_confirmed": True,
            "live_resource_count": len(self.ids),
            "resources": [{"instance_id": item} for item in self.ids],
        }

    def terminate(self, instance_id: str) -> dict:
        self.ids.remove(instance_id)
        return {"status": "terminated"}


def test_vast_watchdog_reaps_only_active_label_prefix_matches_and_proves_absence(
    monkeypatch,
) -> None:
    prefix = "blueprint-groot-oscar-canary-single-episode-"
    rows = [
        {
            "id": 101,
            "label": prefix + "target",
            "actual_status": "running",
            "gpu_name": "RTX 6000 Ada",
            "dph_total": 0.5,
        },
        {
            "id": 202,
            "label": "unrelated-vast-instance",
            "actual_status": "running",
        },
        {
            "id": 303,
            "label": prefix + "already-done",
            "actual_status": "destroyed",
        },
    ]
    api_calls: list[tuple[str, str]] = []

    def fake_api_json(**kwargs):
        assert kwargs["api_key"] == "vast-secret"
        api_calls.append((kwargs["method"], kwargs["path"]))
        return 200, {"instances": list(rows)}

    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._api_json", fake_api_json)

    class VastProvider:
        name = "vast"

        def _key(self):
            return "vast-secret"

        def terminate(self, instance_id: str) -> dict:
            assert instance_id == "101"
            rows[0]["actual_status"] = "destroyed"
            return {"status": "stopped", "http": 204}

    result = terminate_canary_resources(
        provider=VastProvider(),
        provider_name="vast",
        pod_name_prefix=prefix,
        armed={"status": "armed", "provider": "vast"},
    )

    assert result["status"] == "provider_terminal"
    assert result["provider_absence_confirmed"] is True
    assert result["initial_inventory"]["live_resource_count"] == 1
    assert result["initial_inventory"]["resources"][0]["instance_id"] == "101"
    assert result["final_inventory"]["live_resource_count"] == 0
    assert result["terminations"] == [{"instance_id": "101", "status": "stopped", "http": 204}]
    assert api_calls == [("GET", "/instances/"), ("GET", "/instances/")]
    assert "vast-secret" not in json.dumps(result)


def test_vast_watchdog_deletes_recorded_terminal_instance_and_proves_exact_absence(
    tmp_path, monkeypatch
) -> None:
    prefix = "blueprint-groot-oscar-canary-single-episode-"
    instance_id = "45121866"
    (tmp_path / "started_vast_instance_id.txt").write_text(instance_id, encoding="utf-8")
    monkeypatch.setattr(
        "blueprint_pipeline.vast_provider_adapter._api_json",
        lambda **_kwargs: (
            200,
            {
                "instances": [
                    {
                        "id": int(instance_id),
                        "label": prefix + "target",
                        "actual_status": "exited",
                    }
                ]
            },
        ),
    )

    class VastProvider:
        name = "vast"

        def __init__(self) -> None:
            self.delete_calls = 0
            self.inspect_calls = 0

        def _key(self):
            return "vast-secret"

        def terminate(self, observed_id: str) -> dict:
            assert observed_id == instance_id
            self.delete_calls += 1
            if self.delete_calls == 1:
                return {"status": "stopped", "http": 204}
            return {
                "status": "stopped",
                "http": 404,
                "already_gone": True,
            }

        def inspect(self, observed_id: str) -> dict:
            assert observed_id == instance_id
            self.inspect_calls += 1
            if self.inspect_calls == 1:
                return {
                    "status": "observed",
                    "provider": "vast",
                    "http": 200,
                    "instance_id": instance_id,
                    "api_confirmed": True,
                    "provider_absence_confirmed": False,
                    "actual_status": "exited",
                    "name": prefix + "target",
                }
            return {
                "status": "absent",
                "provider": "vast",
                "http": 404,
                "instance_id": instance_id,
                "api_confirmed": True,
                "provider_absence_confirmed": True,
            }

    provider = VastProvider()
    result = terminate_canary_resources(
        provider=provider,
        provider_name="vast",
        pod_name_prefix=prefix,
        armed={
            "status": "armed",
            "provider": "vast",
            "pod_name_prefix": prefix,
            "watchdog_out_dir": str(tmp_path),
        },
    )

    # The name-scoped list intentionally excludes the exited row, but the
    # attempt-local ownership file still forces exact-id DELETE and GET proof.
    assert result["initial_inventory"]["live_resource_count"] == 0
    assert result["final_inventory"]["live_resource_count"] == 0
    assert provider.delete_calls == 2
    assert provider.inspect_calls == 2
    assert result["provider_absence_confirmed"] is True
    assert result["recorded_vast_instance"] == {
        "status": "recorded",
        "required": True,
        "path": str(tmp_path / "started_vast_instance_id.txt"),
        "instance_id": instance_id,
        "scope_confirmed": True,
        "pod_name_prefix": prefix,
    }
    assert result["recorded_vast_instance_teardown"]["provider_absence_confirmed"] is True
    assert [row["instance_id"] for row in result["terminations"]] == [
        instance_id,
        instance_id,
    ]


def test_vast_watchdog_refuses_absence_when_recorded_id_survives_repeated_delete(
    tmp_path, monkeypatch
) -> None:
    prefix = "blueprint-groot-oscar-canary-single-episode-"
    instance_id = "45121866"
    (tmp_path / "started_vast_instance_id.txt").write_text(instance_id, encoding="utf-8")
    monkeypatch.setattr(
        watchdog_module,
        "_vast_billable_inventory",
        lambda **_kwargs: {
            "status": "observed",
            "provider": "vast",
            "api_confirmed": True,
            "live_resource_count": 0,
            "resources": [],
        },
    )

    class VastProvider:
        name = "vast"

        def __init__(self) -> None:
            self.delete_calls = 0

        def terminate(self, observed_id: str) -> dict:
            assert observed_id == instance_id
            self.delete_calls += 1
            return {"status": "stopped", "http": 204}

        def inspect(self, observed_id: str) -> dict:
            assert observed_id == instance_id
            return {
                "status": "observed",
                "provider": "vast",
                "http": 200,
                "instance_id": instance_id,
                "api_confirmed": True,
                "provider_absence_confirmed": False,
                "actual_status": "exited",
                "name": prefix + "target",
            }

    provider = VastProvider()
    result = terminate_canary_resources(
        provider=provider,
        provider_name="vast",
        pod_name_prefix=prefix,
        armed={
            "status": "armed",
            "provider": "vast",
            "pod_name_prefix": prefix,
            "watchdog_out_dir": str(tmp_path),
        },
    )

    assert provider.delete_calls == 2
    assert result["final_inventory"]["live_resource_count"] == 0
    assert result["provider_absence_confirmed"] is False
    assert result["status"] == "teardown_unverified"
    assert result["recorded_vast_instance_teardown"]["status"] == ("teardown_unverified")


def test_vast_watchdog_still_deletes_recorded_id_when_initial_inventory_raises(
    tmp_path, monkeypatch
) -> None:
    prefix = "blueprint-groot-oscar-canary-single-episode-"
    instance_id = "45121866"
    (tmp_path / "started_vast_instance_id.txt").write_text(instance_id, encoding="utf-8")
    inventory_calls = 0

    def inventory(**_kwargs):
        nonlocal inventory_calls
        inventory_calls += 1
        if inventory_calls == 1:
            raise TimeoutError("secret provider response")
        return {
            "status": "observed",
            "provider": "vast",
            "api_confirmed": True,
            "live_resource_count": 0,
            "resources": [],
        }

    monkeypatch.setattr(watchdog_module, "_vast_billable_inventory", inventory)

    class VastProvider:
        name = "vast"

        def __init__(self) -> None:
            self.deleted: list[str] = []

        def terminate(self, observed_id: str) -> dict:
            self.deleted.append(observed_id)
            return {"status": "stopped", "http": 204}

        def inspect(self, observed_id: str) -> dict:
            return {
                "status": "absent",
                "provider": "vast",
                "http": 404,
                "instance_id": observed_id,
                "api_confirmed": True,
                "provider_absence_confirmed": True,
            }

    provider = VastProvider()
    result = terminate_canary_resources(
        provider=provider,
        provider_name="vast",
        pod_name_prefix=prefix,
        armed={
            "status": "armed",
            "provider": "vast",
            "pod_name_prefix": prefix,
            "watchdog_out_dir": str(tmp_path),
        },
    )

    assert provider.deleted == [instance_id]
    assert result["initial_inventory"]["api_confirmed"] is False
    assert result["initial_inventory"]["error_type"] == "TimeoutError"
    assert result["provider_absence_confirmed"] is True
    assert "secret provider response" not in json.dumps(result)


def test_vast_watchdog_started_id_file_requires_exact_armed_prefix(tmp_path, monkeypatch) -> None:
    prefix = "blueprint-groot-oscar-canary-single-episode-"
    (tmp_path / "started_vast_instance_id.txt").write_text("45121866", encoding="utf-8")
    monkeypatch.setattr(
        watchdog_module,
        "_vast_billable_inventory",
        lambda **_kwargs: {
            "status": "observed",
            "provider": "vast",
            "api_confirmed": True,
            "live_resource_count": 0,
            "resources": [],
        },
    )

    class VastProvider:
        name = "vast"

        def terminate(self, _instance_id: str) -> dict:
            raise AssertionError("a differently scoped id must not be destroyed")

    result = terminate_canary_resources(
        provider=VastProvider(),
        provider_name="vast",
        pod_name_prefix=prefix,
        armed={
            "status": "armed",
            "provider": "vast",
            "pod_name_prefix": prefix + "different",
            "watchdog_out_dir": str(tmp_path),
        },
    )

    assert result["provider_absence_confirmed"] is False
    assert result["recorded_vast_instance"]["blockers"] == [
        "vast_started_instance_id_scope_mismatch"
    ]
    assert result["terminations"] == []


def test_vast_watchdog_fails_closed_on_active_instance_without_label(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "blueprint_pipeline.vast_provider_adapter._api_json",
        lambda **_kwargs: (
            200,
            {"instances": [{"id": 404, "actual_status": "running"}]},
        ),
    )

    class VastProvider:
        name = "vast"

        def _key(self):
            return "vast-secret"

        def terminate(self, _instance_id: str) -> dict:
            raise AssertionError("an unlabeled instance must never be destroyed")

    result = terminate_canary_resources(
        provider=VastProvider(),
        provider_name="vast",
        pod_name_prefix="blueprint-groot-oscar-canary-single-episode-",
        armed={"status": "armed", "provider": "vast"},
    )

    assert result["status"] == "teardown_unverified"
    assert result["provider_absence_confirmed"] is False
    assert result["initial_inventory"]["api_confirmed"] is False
    assert result["initial_inventory"]["blockers"] == ["vast_active_instance_label_missing"]
    assert result["terminations"] == []


def test_run_watchdog_selects_vast_provider_without_changing_default_contract(
    tmp_path, monkeypatch
) -> None:
    selected: list[str] = []

    class VastProvider:
        name = "vast"

    def provider_factory(name: str):
        selected.append(name)
        return VastProvider()

    monkeypatch.setattr(
        watchdog_module,
        "_vast_billable_inventory",
        lambda **_kwargs: {
            "status": "observed",
            "provider": "vast",
            "api_confirmed": True,
            "live_resource_count": 0,
            "resources": [],
        },
    )
    result = run_watchdog(
        out_dir=tmp_path,
        pod_name_prefix="blueprint-groot-oscar-canary-single-episode-",
        deadline_epoch=10_000_000_000.0,
        provider_name="vast",
        provider_factory=provider_factory,
        clock=lambda: 10_000_000_000.0,
        sleeper=lambda _seconds: None,
    )

    assert selected == ["vast"]
    assert result["provider"] == "vast"
    assert result["status"] == "provider_terminal"
    persisted = json.loads(
        (tmp_path / "groot_oscar_runpod_canary_watchdog.json").read_text(encoding="utf-8")
    )
    assert persisted == result


def test_independent_watchdog_never_mutates_provider_before_hard_deadline(
    tmp_path, monkeypatch
) -> None:
    now = {"value": 100.0}
    deadline = 200.0
    events: list[tuple[str, float]] = []
    monkeypatch.setattr(watchdog_module.time, "time", lambda: now["value"])

    class Provider:
        name = "runpod"

        def billable_inventory(self, *, name_prefix: str) -> dict:
            events.append(("inventory", now["value"]))
            assert name_prefix == "blueprint-groot-oscar-canary-attempt-"
            assert now["value"] >= deadline
            return {
                "api_confirmed": True,
                "live_resource_count": 0,
                "resources": [],
            }

    def provider_factory(_name: str) -> Provider:
        events.append(("provider_factory", now["value"]))
        assert now["value"] >= deadline
        return Provider()

    def sleeper(seconds: float) -> None:
        events.append(("sleep", now["value"]))
        assert now["value"] < deadline
        now["value"] += seconds

    result = run_watchdog(
        out_dir=tmp_path,
        pod_name_prefix="blueprint-groot-oscar-canary-attempt-",
        deadline_epoch=deadline,
        provider_factory=provider_factory,
        clock=lambda: now["value"],
        sleeper=sleeper,
    )

    assert result["status"] == "provider_terminal"
    assert result["provider_mutation_trigger"] == "hard_deadline_only"
    assert result["pre_deadline_provider_mutation_allowed"] is False
    assert events[0] == ("sleep", 100.0)
    assert all(
        observed_at >= deadline
        for event, observed_at in events
        if event in {"provider_factory", "inventory"}
    )


def test_owner_teardown_cancel_runs_zero_verification_before_deadline(
    tmp_path, monkeypatch
) -> None:
    now = 100.0
    deadline = 200.0
    prefix = "blueprint-groot-oscar-canary-attempt-"
    monkeypatch.setattr(watchdog_module.time, "time", lambda: now)
    cancel = {
        "schema_version": watchdog_module.OWNER_TEARDOWN_CANCEL_SCHEMA_VERSION,
        "requested_by": "qualification_owner_teardown",
        "provider": "runpod",
        "instance_id": "pod-terminated",
        "pod_name_prefix": prefix,
        "provider_absence_confirmed": True,
        "provider_absence_evidence": ("provider_api_exact_id_prefix_and_global_inventory"),
    }
    cancel_path = tmp_path / watchdog_module.OWNER_TEARDOWN_CANCEL_NAME
    cancel_path.write_text(json.dumps(cancel), encoding="utf-8")
    cancel_path.chmod(0o600)
    events: list[str] = []

    class Provider:
        name = "runpod"

        def billable_inventory(self, *, name_prefix: str) -> dict:
            events.append(f"inventory:{name_prefix}")
            return {
                "api_confirmed": True,
                "live_resource_count": 0,
                "resources": [],
            }

    result = run_watchdog(
        out_dir=tmp_path,
        pod_name_prefix=prefix,
        deadline_epoch=deadline,
        provider_factory=lambda _name: Provider(),
        clock=lambda: now,
        sleeper=lambda _seconds: pytest.fail("valid cancellation must not sleep"),
    )

    assert result["status"] == "provider_terminal"
    assert result["owner_teardown_cancel_requested"] is True
    assert result["owner_teardown_cancel_request_valid"] is True
    assert result["provider_mutation_trigger"] == (
        "owner_teardown_cancel_request_after_provider_zero"
    )
    assert events == [
        f"inventory:{prefix}",
        "inventory:",
        f"inventory:{prefix}",
        "inventory:",
    ]


def test_owner_teardown_cancel_requires_global_provider_zero(tmp_path, monkeypatch) -> None:
    now = {"value": 100.0}
    deadline = 200.0
    prefix = "blueprint-groot-oscar-canary-attempt-"
    monkeypatch.setattr(watchdog_module.time, "time", lambda: now["value"])
    cancel_path = tmp_path / watchdog_module.OWNER_TEARDOWN_CANCEL_NAME
    cancel_path.write_text(
        json.dumps(
            {
                "schema_version": (watchdog_module.OWNER_TEARDOWN_CANCEL_SCHEMA_VERSION),
                "requested_by": "qualification_owner_teardown",
                "provider": "runpod",
                "instance_id": "pod-terminated",
                "pod_name_prefix": prefix,
                "provider_absence_confirmed": True,
                "provider_absence_evidence": ("provider_api_exact_id_prefix_and_global_inventory"),
            }
        ),
        encoding="utf-8",
    )
    cancel_path.chmod(0o600)
    observed: list[tuple[float, str]] = []

    class Provider:
        name = "runpod"

        def billable_inventory(self, *, name_prefix: str) -> dict:
            observed.append((now["value"], name_prefix))
            count = 1 if name_prefix == "" else 0
            return {
                "api_confirmed": True,
                "live_resource_count": count,
                "resources": ([{"instance_id": "unrelated-live"}] if count else []),
            }

    result = run_watchdog(
        out_dir=tmp_path,
        pod_name_prefix=prefix,
        deadline_epoch=deadline,
        provider_factory=lambda _name: Provider(),
        clock=lambda: now["value"],
        sleeper=lambda seconds: now.__setitem__("value", now["value"] + seconds),
    )

    assert result["owner_teardown_cancel_requested"] is False
    assert any(name_prefix == "" for _, name_prefix in observed)
    assert all(observed_at < deadline for observed_at, name_prefix in observed if name_prefix == "")


def test_vast_owner_cancel_requires_exact_recorded_id_absence(tmp_path, monkeypatch) -> None:
    now = 100.0
    deadline = 200.0
    prefix = "blueprint-groot-oscar-canary-qualification-attempt-"
    instance_id = "45483300"
    monkeypatch.setattr(watchdog_module.time, "time", lambda: now)
    monkeypatch.setattr(
        watchdog_module,
        "_vast_billable_inventory",
        lambda **_kwargs: {
            "status": "observed",
            "provider": "vast",
            "api_confirmed": True,
            "live_resource_count": 0,
            "resources": [],
        },
    )
    (tmp_path / "started_vast_instance_id.txt").write_text(instance_id)
    cancel_path = tmp_path / watchdog_module.OWNER_TEARDOWN_CANCEL_NAME
    cancel_path.write_text(
        json.dumps(
            {
                "schema_version": (watchdog_module.OWNER_TEARDOWN_CANCEL_SCHEMA_VERSION),
                "requested_by": "qualification_owner_teardown",
                "provider": "vast",
                "instance_id": instance_id,
                "pod_name_prefix": prefix,
                "provider_absence_confirmed": True,
                "provider_absence_evidence": ("provider_api_exact_id_prefix_and_global_inventory"),
            }
        ),
        encoding="utf-8",
    )
    cancel_path.chmod(0o600)

    class Provider:
        name = "vast"

        def inspect(self, observed_id: str) -> dict:
            assert observed_id == instance_id
            return {
                "status": "absent",
                "provider": "vast",
                "http": 404,
                "instance_id": instance_id,
                "api_confirmed": True,
                "provider_absence_confirmed": True,
            }

    result = run_watchdog(
        out_dir=tmp_path,
        pod_name_prefix=prefix,
        deadline_epoch=deadline,
        provider_name="vast",
        provider_factory=lambda _name: Provider(),
        clock=lambda: now,
        sleeper=lambda _seconds: pytest.fail("exact zero must cancel immediately"),
    )

    assert result["status"] == "provider_terminal"
    assert result["provider_mutations_performed"] == 0
    assert result["recorded_vast_instance_teardown"]["provider_absence_confirmed"] is True
    assert len(result["recorded_vast_instance_teardown"]["inspect_attempts"]) == 2


def test_vast_owner_cancel_without_recorded_id_accepts_repeated_global_zero(
    tmp_path, monkeypatch
) -> None:
    now = 100.0
    deadline = 200.0
    prefix = "blueprint-groot-oscar-canary-openpi-ranking-"
    monkeypatch.setattr(watchdog_module.time, "time", lambda: now)
    inventory_prefixes: list[str] = []

    def zero_inventory(*, provider, name_prefix: str) -> dict:
        del provider
        inventory_prefixes.append(name_prefix)
        return {
            "status": "observed",
            "provider": "vast",
            "api_confirmed": True,
            "live_resource_count": 0,
            "resources": [],
        }

    monkeypatch.setattr(watchdog_module, "_vast_billable_inventory", zero_inventory)
    cancel_path = tmp_path / watchdog_module.OWNER_TEARDOWN_CANCEL_NAME
    cancel_path.write_text(
        json.dumps(
            {
                "schema_version": (watchdog_module.OWNER_TEARDOWN_CANCEL_SCHEMA_VERSION),
                "requested_by": "qualification_owner_teardown",
                "provider": "vast",
                "instance_id": prefix + "no-instance-created",
                "pod_name_prefix": prefix,
                "provider_absence_confirmed": True,
                "provider_absence_evidence": ("provider_api_exact_id_prefix_and_global_inventory"),
            }
        ),
        encoding="utf-8",
    )
    cancel_path.chmod(0o600)

    class Provider:
        name = "vast"

        def inspect(self, _instance_id: str) -> dict:
            pytest.fail("no synthetic exact-id inspection is allowed")

        def terminate(self, _instance_id: str) -> dict:
            pytest.fail("zero-inventory cancellation must not mutate the provider")

    result = run_watchdog(
        out_dir=tmp_path,
        pod_name_prefix=prefix,
        deadline_epoch=deadline,
        provider_name="vast",
        provider_factory=lambda _name: Provider(),
        clock=lambda: now,
        sleeper=lambda _seconds: pytest.fail("double zero must cancel immediately"),
    )

    assert result["status"] == "provider_terminal"
    assert result["provider_absence_confirmed"] is True
    assert result["provider_mutations_performed"] == 0
    assert result["owner_teardown_cancel_request_valid"] is True
    assert result["recorded_vast_instance"]["status"] == "not_recorded"
    assert inventory_prefixes == [prefix, "", prefix, ""]


def test_vast_owner_cancel_does_not_hide_recorded_contract(tmp_path, monkeypatch) -> None:
    now = {"value": 100.0}
    deadline = 200.0
    prefix = "blueprint-groot-oscar-canary-qualification-attempt-"
    instance_id = "45483300"
    monkeypatch.setattr(watchdog_module.time, "time", lambda: now["value"])
    monkeypatch.setattr(
        watchdog_module,
        "_vast_billable_inventory",
        lambda **_kwargs: {
            "status": "observed",
            "provider": "vast",
            "api_confirmed": True,
            "live_resource_count": 0,
            "resources": [],
        },
    )
    (tmp_path / "started_vast_instance_id.txt").write_text(instance_id)
    cancel_path = tmp_path / watchdog_module.OWNER_TEARDOWN_CANCEL_NAME
    cancel_path.write_text(
        json.dumps(
            {
                "schema_version": (watchdog_module.OWNER_TEARDOWN_CANCEL_SCHEMA_VERSION),
                "requested_by": "qualification_owner_teardown",
                "provider": "vast",
                "instance_id": instance_id,
                "pod_name_prefix": prefix,
                "provider_absence_confirmed": True,
                "provider_absence_evidence": ("provider_api_exact_id_prefix_and_global_inventory"),
            }
        ),
        encoding="utf-8",
    )
    cancel_path.chmod(0o600)
    terminate_times: list[float] = []

    class Provider:
        name = "vast"

        def inspect(self, observed_id: str) -> dict:
            assert observed_id == instance_id
            return {
                "status": "observed",
                "provider": "vast",
                "http": 200,
                "instance_id": instance_id,
                "api_confirmed": True,
                "provider_absence_confirmed": False,
                "actual_status": "exited",
            }

        def terminate(self, observed_id: str) -> dict:
            assert observed_id == instance_id
            terminate_times.append(now["value"])
            return {"status": "stopped", "http": 204}

    provider = Provider()
    result = run_watchdog(
        out_dir=tmp_path,
        pod_name_prefix=prefix,
        deadline_epoch=deadline,
        provider_name="vast",
        provider_factory=lambda _name: provider,
        clock=lambda: now["value"],
        sleeper=lambda seconds: now.__setitem__("value", now["value"] + seconds),
    )

    assert result["owner_teardown_cancel_requested"] is False
    assert terminate_times
    assert all(observed_at >= deadline for observed_at in terminate_times)


def test_owner_teardown_cancel_never_terminates_live_resource_before_deadline(
    tmp_path, monkeypatch
) -> None:
    now = {"value": 100.0}
    deadline = 200.0
    prefix = "blueprint-groot-oscar-canary-attempt-"
    monkeypatch.setattr(watchdog_module.time, "time", lambda: now["value"])
    cancel_path = tmp_path / watchdog_module.OWNER_TEARDOWN_CANCEL_NAME
    cancel_path.write_text(
        json.dumps(
            {
                "schema_version": (watchdog_module.OWNER_TEARDOWN_CANCEL_SCHEMA_VERSION),
                "requested_by": "qualification_owner_teardown",
                "provider": "runpod",
                "instance_id": "pod-active",
                "pod_name_prefix": prefix,
                "provider_absence_confirmed": True,
                "provider_absence_evidence": ("provider_api_exact_id_prefix_and_global_inventory"),
            }
        ),
        encoding="utf-8",
    )
    cancel_path.chmod(0o600)
    terminate_times: list[float] = []

    class Provider:
        name = "runpod"

        def __init__(self) -> None:
            self.ids = ["pod-active"]

        def billable_inventory(self, *, name_prefix: str) -> dict:
            assert name_prefix == prefix
            return {
                "api_confirmed": True,
                "live_resource_count": len(self.ids),
                "resources": [{"instance_id": item} for item in self.ids],
            }

        def terminate(self, instance_id: str) -> dict:
            terminate_times.append(now["value"])
            assert now["value"] >= deadline
            self.ids.remove(instance_id)
            return {"status": "terminated"}

    provider = Provider()
    result = run_watchdog(
        out_dir=tmp_path,
        pod_name_prefix=prefix,
        deadline_epoch=deadline,
        provider_factory=lambda _name: provider,
        clock=lambda: now["value"],
        sleeper=lambda seconds: now.__setitem__("value", now["value"] + seconds),
    )

    assert result["status"] == "provider_terminal"
    assert result["owner_teardown_cancel_requested"] is False
    assert terminate_times == [deadline]


def test_watchdog_reaps_every_name_bound_resource_and_proves_absence() -> None:
    result = terminate_canary_resources(
        provider=_Provider(),
        pod_name_prefix="blueprint-groot-oscar-canary-attempt-",
        armed={"status": "armed"},
    )
    assert result["status"] == "provider_terminal"
    assert result["provider_absence_confirmed"] is True
    assert [row["instance_id"] for row in result["terminations"]] == [
        "pod-1",
        "pod-2",
    ]


def test_watchdog_inventory_error_returns_secret_safe_unverified_evidence() -> None:
    class Provider:
        def billable_inventory(self, *, name_prefix: str):
            del name_prefix
            raise TimeoutError("secret provider response")

    result = terminate_canary_resources(
        provider=Provider(),
        pod_name_prefix="blueprint-groot-oscar-canary-attempt-",
        armed={"status": "armed"},
    )
    assert result["status"] == "teardown_unverified"
    assert result["provider_absence_confirmed"] is False
    assert result["teardown_error_type"] == "TimeoutError"
    assert "secret provider response" not in json.dumps(result)


def test_watchdog_persists_provider_factory_error(tmp_path) -> None:
    def fail_provider(_name: str):
        raise TimeoutError("secret provider initialization")

    result = run_watchdog(
        out_dir=tmp_path,
        pod_name_prefix="blueprint-groot-oscar-canary-attempt-",
        deadline_epoch=10_000_000_000.0,
        provider_factory=fail_provider,
        clock=lambda: 10_000_000_000.0,
        sleeper=lambda _seconds: None,
    )
    persisted = json.loads(
        (tmp_path / "groot_oscar_runpod_canary_watchdog.json").read_text(encoding="utf-8")
    )
    assert result == persisted
    assert persisted["status"] == "teardown_unverified"
    assert persisted["teardown_error_type"] == "TimeoutError"
    assert "secret provider initialization" not in json.dumps(persisted)


def test_watchdog_closes_pod_record_and_returns_lane_owner(tmp_path, monkeypatch) -> None:
    pending_path = tmp_path / "pending.json"
    pending_path.write_text(
        json.dumps(
            {
                "status": "open",
                "provider": "runpod",
                "lane": "groot_oscar_gpu_canary",
                "resource_kind": "compute_instance",
                "resource_name": "blueprint-groot-oscar-canary-attempt-pod",
            }
        ),
        encoding="utf-8",
    )
    receipt_path = tmp_path / "provider_lane_handoff_receipt.json"
    ledger_path = tmp_path / "campaign-budget.json"
    ledger = ProductionGpuCampaignBudget(
        ledger_path,
        initial_spent_usd=11.57,
        initial_used_gpu_seconds=11_619,
        combined_gpu_wall_cap_seconds=16_800,
    )
    reservation = ledger.reserve(
        reservation_id="watchdog-budget-test",
        gpu_seconds=100,
        max_hourly_rate_usd=1.99,
    )
    receipt = {
        "lease_path": str(tmp_path / "lane.lease.json"),
        "owner_pid": 222,
        "pod_pending_teardown_record": str(pending_path),
        "pod_id": "pod-1",
        "pod_name_prefix": "blueprint-groot-oscar-canary-attempt-",
        "campaign_budget": {
            "status": "reserved",
            "ledger_path": str(ledger_path),
            "reservation_id": "watchdog-budget-test",
            "reserved_at_epoch": 9_999_999_900.0,
            "reservation": reservation,
            "identity": {
                "initial_spent_usd": 11.57,
                "initial_used_gpu_seconds": 11_619,
                "total_spend_cap_usd": 20.0,
                "combined_gpu_wall_cap_seconds": 16_800,
            },
        },
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    os.chmod(receipt_path, 0o600)

    monkeypatch.setattr(
        "blueprint_pipeline.paid_lane_guard.close_pending_teardown",
        lambda path, evidence: {
            "status": "closed",
            "path": path,
            "evidence": evidence,
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.paid_provider_lane_lease.restore_paid_provider_lane_lease_to_retained_watchdog",
        lambda observed: {
            "status": "restored",
            "restored": observed == receipt,
        },
    )

    class EmptyProvider:
        def billable_inventory(self, *, name_prefix: str) -> dict:
            assert name_prefix == receipt["pod_name_prefix"]
            return {
                "api_confirmed": True,
                "live_resource_count": 0,
                "resources": [],
            }

    result = run_watchdog(
        out_dir=tmp_path,
        pod_name_prefix=receipt["pod_name_prefix"],
        deadline_epoch=10_000_000_000.0,
        provider_factory=lambda _name: EmptyProvider(),
        clock=lambda: 10_000_000_000.0,
        sleeper=lambda _seconds: None,
    )

    assert result["status"] == "provider_terminal"
    assert result["control_plane_terminal"] is True
    assert result["pod_pending_teardown_close"]["status"] == "closed"
    assert result["provider_lane_owner_return"]["status"] == "restored"
    assert result["campaign_budget_settlement"]["status"] == "settled"
    assert result["campaign_budget_settlement"]["charged_gpu_seconds"] == 100

    retried = run_watchdog(
        out_dir=tmp_path,
        pod_name_prefix=receipt["pod_name_prefix"],
        deadline_epoch=10_000_000_000.0,
        provider_factory=lambda _name: EmptyProvider(),
        clock=lambda: 10_000_000_000.0,
        sleeper=lambda _seconds: None,
    )
    assert retried["campaign_budget_settlement"]["status"] == "settled"


@pytest.mark.parametrize(
    ("campaign_kind", "lane", "prefix", "pod_id"),
    (
        (
            "openpi_policy_ranking",
            "openpi_policy_ranking_gpu_canary",
            "blueprint-groot-oscar-canary-openpi-ranking-",
            "pod-openpi",
        ),
        (
            "nvidia_warehouse_native_camera",
            "nvidia_warehouse_native_camera_gpu_canary",
            "blueprint-native-warehouse-camera-",
            "pod-camera",
        ),
    ),
)
def test_watchdog_closes_guarded_compute_lane_and_settles_budget(
    tmp_path, monkeypatch, campaign_kind, lane, prefix, pod_id
) -> None:
    pending_path = tmp_path / "openpi-pending.json"
    pending_path.write_text(
        json.dumps(
            {
                "status": "open",
                "provider": "runpod",
                "lane": lane,
                "resource_kind": "compute_instance",
                "resource_name": prefix + "test",
                "instance_id": pod_id,
            }
        ),
        encoding="utf-8",
    )
    lease_path = tmp_path / "openpi-lane.lease.json"
    lease_path.write_text(
        json.dumps(
            {
                "provider": "runpod",
                "lane": lane,
                "owner_pid": os.getpid(),
                "retained_teardown_owner_pid": os.getpid(),
            }
        ),
        encoding="utf-8",
    )
    ledger_path = tmp_path / "openpi-campaign-budget.json"
    ledger = ProductionGpuCampaignBudget(
        ledger_path,
        initial_spent_usd=0.0,
        initial_used_gpu_seconds=0,
        combined_gpu_wall_cap_seconds=1_000,
    )
    reservation = ledger.reserve(
        reservation_id="openpi-watchdog-test",
        gpu_seconds=100,
        max_hourly_rate_usd=0.44,
    )
    receipt = {
        "lease_path": str(lease_path),
        "owner_pid": os.getpid(),
        "provider_lane_release_mode": "watchdog_direct_compute",
        "pod_pending_teardown_record": str(pending_path),
        "pod_id": pod_id,
        "pod_name_prefix": prefix,
        "campaign_kind": campaign_kind,
        "paid_lane": lane,
        "campaign_budget": {
            "status": "reserved",
            "ledger_path": str(ledger_path),
            "reservation_id": "openpi-watchdog-test",
            "reserved_at_epoch": 9_999_999_900.0,
            "reservation": reservation,
            "identity": {
                "initial_spent_usd": 0.0,
                "initial_used_gpu_seconds": 0,
                "total_spend_cap_usd": 20.0,
                "combined_gpu_wall_cap_seconds": 1_000,
            },
        },
    }
    receipt_path = tmp_path / "provider_lane_handoff_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    os.chmod(receipt_path, 0o600)
    monkeypatch.setattr(watchdog_module, "load_pending_teardowns", lambda: [])
    monkeypatch.setattr(
        "blueprint_pipeline.paid_lane_guard.close_pending_teardown",
        lambda path, evidence: {"status": "closed", "path": path, "evidence": evidence},
    )

    class EmptyProvider:
        def billable_inventory(self, *, name_prefix: str) -> dict:
            assert name_prefix == prefix
            return {
                "api_confirmed": True,
                "live_resource_count": 0,
                "resources": [],
            }

    result = run_watchdog(
        out_dir=tmp_path,
        pod_name_prefix=prefix,
        deadline_epoch=10_000_000_000.0,
        provider_factory=lambda _name: EmptyProvider(),
        clock=lambda: 10_000_000_000.0,
        sleeper=lambda _seconds: None,
    )

    assert result["status"] == "provider_terminal"
    assert result["control_plane_terminal"] is True
    assert result["provider_lane_terminal_release"]["status"] == "released"
    assert not lease_path.exists()
    assert result["campaign_budget_settlement"]["status"] == "settled"
    assert result["campaign_budget_settlement"]["charged_gpu_seconds"] == 100


def test_watchdog_accepts_persistent_carrier_runner_pending_lane(tmp_path, monkeypatch) -> None:
    prefix = "blueprint-groot-oscar-canary-persistent-"
    pending_path = tmp_path / "persistent-pending.json"
    pending_path.write_text(
        json.dumps(
            {
                "status": "open",
                "provider": "runpod",
                "lane": "runpod_wam_async",
                "resource_kind": "compute_instance",
                "resource_name": prefix + "pod",
                "instance_id": "pod-persistent-1",
            }
        ),
        encoding="utf-8",
    )
    receipt = {
        "status": "accepted",
        "campaign_kind": "persistent_policy_wam_loop",
        "pod_name_prefix": prefix,
        "pod_pending_teardown_record": str(pending_path),
        "pod_id": "pod-persistent-1",
    }
    receipt_path = tmp_path / "provider_lane_handoff_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o600)
    monkeypatch.setattr(
        "blueprint_pipeline.paid_lane_guard.close_pending_teardown",
        lambda *_args, **_kwargs: {"status": "closed"},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.paid_provider_lane_lease.restore_paid_provider_lane_lease_to_retained_watchdog",
        lambda _receipt: {"status": "restored", "restored": True},
    )

    class EmptyProvider:
        def billable_inventory(self, *, name_prefix: str) -> dict:
            assert name_prefix == prefix
            return {
                "api_confirmed": True,
                "live_resource_count": 0,
                "resources": [],
            }

    result = run_watchdog(
        out_dir=tmp_path,
        pod_name_prefix=prefix,
        deadline_epoch=10_000_000_000.0,
        provider_factory=lambda _name: EmptyProvider(),
        clock=lambda: 10_000_000_000.0,
        sleeper=lambda _seconds: None,
    )

    assert result["status"] == "provider_terminal"
    assert result["control_plane_terminal"] is True
    assert result["pod_pending_teardown_close"]["status"] == "closed"


def test_unverified_teardown_retains_open_campaign_reservation(tmp_path) -> None:
    watchdog_dir = tmp_path / "watchdog"
    watchdog_dir.mkdir()
    ledger_path = tmp_path / "campaign-budget.json"
    ledger = ProductionGpuCampaignBudget(
        ledger_path,
        initial_spent_usd=11.57,
        initial_used_gpu_seconds=11_619,
        combined_gpu_wall_cap_seconds=16_800,
    )
    reservation = ledger.reserve(
        reservation_id="unverified-watchdog",
        gpu_seconds=100,
        max_hourly_rate_usd=1.99,
    )
    receipt_path = watchdog_dir / "provider_lane_handoff_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "pod_name_prefix": "blueprint-groot-oscar-canary-unverified-",
                "campaign_budget": {
                    "status": "reserved",
                    "ledger_path": str(ledger_path),
                    "reservation_id": "unverified-watchdog",
                    "reserved_at_epoch": 9_999_999_900.0,
                    "reservation": reservation,
                    "identity": {
                        "initial_spent_usd": 11.57,
                        "initial_used_gpu_seconds": 11_619,
                        "total_spend_cap_usd": 20.0,
                        "combined_gpu_wall_cap_seconds": 16_800,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    os.chmod(receipt_path, 0o600)

    class UnverifiedProvider:
        def billable_inventory(self, *, name_prefix: str):
            del name_prefix
            raise TimeoutError

    result = run_watchdog(
        out_dir=watchdog_dir,
        pod_name_prefix="blueprint-groot-oscar-canary-unverified-",
        deadline_epoch=10_000_000_000.0,
        provider_factory=lambda _name: UnverifiedProvider(),
        clock=lambda: 10_000_000_000.0,
        sleeper=lambda _seconds: None,
    )
    assert result["status"] == "teardown_unverified"
    assert result["control_plane_terminal"] is False
    assert "campaign_budget_settlement" not in result
    snapshot = ledger.snapshot()
    assert snapshot["open_reservation_count"] == 1
    assert snapshot["reservations"][0]["status"] == "open"


def test_elapsed_beyond_reservation_retains_open_budget_breach(tmp_path, monkeypatch) -> None:
    pending_path = tmp_path / "pending.json"
    pending_path.write_text(
        json.dumps(
            {
                "status": "open",
                "provider": "runpod",
                "lane": "groot_oscar_gpu_canary",
                "resource_kind": "compute_instance",
                "resource_name": "blueprint-groot-oscar-canary-overrun-pod",
                "instance_id": "pod-1",
            }
        ),
        encoding="utf-8",
    )
    ledger_path = tmp_path / "campaign-budget.json"
    ledger = ProductionGpuCampaignBudget(
        ledger_path,
        initial_spent_usd=11.57,
        initial_used_gpu_seconds=11_619,
        combined_gpu_wall_cap_seconds=16_800,
    )
    reservation = ledger.reserve(
        reservation_id="watchdog-overrun",
        gpu_seconds=10,
        max_hourly_rate_usd=1.99,
    )
    receipt = {
        "pod_name_prefix": "blueprint-groot-oscar-canary-overrun-",
        "pod_pending_teardown_record": str(pending_path),
        "pod_id": "pod-1",
        "campaign_budget": {
            "status": "reserved",
            "ledger_path": str(ledger_path),
            "reservation_id": "watchdog-overrun",
            "reserved_at_epoch": 9_999_999_989.0,
            "reservation": reservation,
            "identity": {
                "initial_spent_usd": 11.57,
                "initial_used_gpu_seconds": 11_619,
                "total_spend_cap_usd": 20.0,
                "combined_gpu_wall_cap_seconds": 16_800,
            },
        },
    }
    receipt_path = tmp_path / "provider_lane_handoff_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    os.chmod(receipt_path, 0o600)
    monkeypatch.setattr(
        "blueprint_pipeline.paid_lane_guard.close_pending_teardown",
        lambda *_args, **_kwargs: {"status": "closed"},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.paid_provider_lane_lease.restore_paid_provider_lane_lease_to_retained_watchdog",
        lambda _receipt: {"status": "restored"},
    )

    class EmptyProvider:
        def billable_inventory(self, *, name_prefix):
            del name_prefix
            return {"api_confirmed": True, "live_resource_count": 0, "resources": []}

    result = run_watchdog(
        out_dir=tmp_path,
        pod_name_prefix=receipt["pod_name_prefix"],
        deadline_epoch=10_000_000_000.0,
        provider_factory=lambda _name: EmptyProvider(),
        clock=lambda: 10_000_000_000.0,
        sleeper=lambda _seconds: None,
    )
    assert result["status"] == "provider_terminal_budget_reservation_exceeded"
    assert result["campaign_budget_settlement"] == {
        "status": "retained_open_budget_breach",
        "elapsed_gpu_seconds": 11,
        "reserved_gpu_seconds": 10,
    }
    snapshot = ledger.snapshot()
    assert snapshot["open_reservation_count"] == 1
    assert snapshot["reservations"][0]["status"] == "open"
