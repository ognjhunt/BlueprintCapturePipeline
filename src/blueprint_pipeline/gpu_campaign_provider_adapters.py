"""GCP and AWS adapters for the provider-neutral GPU campaign seam.

Cloud SDK details stay behind injected operations. This keeps unit/contract
tests hermetic while production wiring can use gcloud, boto3, or a broker
without changing the campaign state machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


Inventory = Callable[[str], Sequence[Mapping[str, Any]]]
Allocate = Callable[[Mapping[str, Any]], Mapping[str, Any]]
RunStage = Callable[[str, str, int, Mapping[str, Any]], Mapping[str, Any]]
Retrieve = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]
Terminate = Callable[[str], Mapping[str, Any]]
Inspect = Callable[[str], Mapping[str, Any]]


@dataclass
class _OperationsAdapter:
    inventory_operation: Inventory
    allocate_operation: Allocate
    stage_operation: RunStage
    retrieve_operation: Retrieve
    terminate_operation: Terminate
    inspect_operation: Inspect
    provider_name: str = "external"

    def inventory(self, allocation_key: str) -> Sequence[Mapping[str, Any]]:
        return self.inventory_operation(allocation_key)

    def allocate(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.allocate_operation(config)

    def run_stage(
        self,
        allocation_id: str,
        stage: str,
        *,
        deadline_seconds: int,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return self.stage_operation(allocation_id, stage, deadline_seconds, config)

    def retrieve(self, allocation_id: str, config: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.retrieve_operation(allocation_id, config)

    def terminate(self, allocation_id: str) -> Mapping[str, Any]:
        return self.terminate_operation(allocation_id)

    def inspect(self, allocation_id: str) -> Mapping[str, Any]:
        return self.inspect_operation(allocation_id)


class GcpCampaignAdapter(_OperationsAdapter):
    """Adapter role for Google Compute Engine operations."""

    provider_name = "gcp"

    def __init__(self, **operations: Any) -> None:
        super().__init__(provider_name="gcp", **operations)


class AwsCampaignAdapter(_OperationsAdapter):
    """Adapter role for EC2 operations."""

    provider_name = "aws"

    def __init__(self, **operations: Any) -> None:
        super().__init__(provider_name="aws", **operations)
