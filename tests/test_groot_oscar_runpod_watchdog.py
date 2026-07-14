import json

from blueprint_pipeline.groot_oscar_runpod_watchdog import (
    run_watchdog,
    terminate_canary_resources,
)


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
        (tmp_path / "groot_oscar_runpod_canary_watchdog.json").read_text(
            encoding="utf-8"
        )
    )
    assert result == persisted
    assert persisted["status"] == "teardown_unverified"
    assert persisted["teardown_error_type"] == "TimeoutError"
    assert "secret provider initialization" not in json.dumps(persisted)
