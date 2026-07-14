import pytest

from blueprint_pipeline.gpu_campaign_provider_adapters import (
    AwsCampaignAdapter,
    GcpCampaignAdapter,
)


@pytest.mark.parametrize(
    "adapter_type,provider", [(GcpCampaignAdapter, "gcp"), (AwsCampaignAdapter, "aws")]
)
def test_cloud_adapters_satisfy_same_provider_neutral_interface(adapter_type, provider):
    adapter = adapter_type(
        inventory_operation=lambda key: [],
        allocate_operation=lambda config: {"allocation_id": "vm-1"},
        stage_operation=lambda allocation, stage, deadline, config: {
            "status": "passed",
            "stage": stage,
        },
        retrieve_operation=lambda allocation, config: {"status": "retrieved"},
        terminate_operation=lambda allocation: {"status": "delete_requested"},
        inspect_operation=lambda allocation: {"http": 404, "absent": True},
    )
    assert adapter.provider_name == provider
    assert adapter.inventory("key") == []
    assert adapter.allocate({})["allocation_id"] == "vm-1"
    assert adapter.run_stage("vm-1", "smoke", deadline_seconds=3, config={})["status"] == "passed"
    assert adapter.retrieve("vm-1", {})["status"] == "retrieved"
    assert adapter.terminate("vm-1")["status"] == "delete_requested"
    assert adapter.inspect("vm-1")["absent"] is True
