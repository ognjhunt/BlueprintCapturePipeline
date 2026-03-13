from __future__ import annotations

from blueprint_contracts.runtime_layer_contract import classify_region as shared_classify_region
from blueprint_pipeline.runtime_layer_grounding import classify_region


def test_pipeline_runtime_layer_wrapper_points_at_shared_contract() -> None:
    assert classify_region is shared_classify_region
