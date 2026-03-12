from blueprint_pipeline.runtime_layer_grounding import classify_region


def test_classify_region_respects_thresholds_and_task_critical_override() -> None:
    assert classify_region(grounding_level="observed", confidence=0.9, task_critical=False, provenance_present=True) == "locked"
    assert classify_region(grounding_level="reconstructed", confidence=0.72, task_critical=False, provenance_present=True) == "uncertain"
    assert classify_region(grounding_level="reconstructed", confidence=0.72, task_critical=True, provenance_present=True) == "locked"
    assert classify_region(grounding_level="generated", confidence=0.99, task_critical=False, provenance_present=True) == "editable"
    assert classify_region(grounding_level="", confidence=None, task_critical=False, provenance_present=False) == "editable"
