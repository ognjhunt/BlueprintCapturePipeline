from blueprint_pipeline.groot_oscar_runpod_watchdog import terminate_canary_resources


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
