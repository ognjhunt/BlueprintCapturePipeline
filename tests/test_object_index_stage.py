from __future__ import annotations

from blueprint_pipeline.object_index_stage import _existing_index_is_reusable


def test_existing_index_is_not_reused_when_empty_and_runtime_was_missing() -> None:
    reusable = _existing_index_is_reusable(
        loaded=[],
        report={
            "status": "built",
            "object_count": 0,
            "empty_index_cause": "runtime_missing",
            "runtime_preflight": {
                "backends": {
                    "yolo_world": {
                        "support_level": "required",
                        "status": "runtime_missing",
                    }
                }
            },
        },
    )

    assert reusable is False


def test_existing_index_is_reused_when_zero_objects_were_a_real_result() -> None:
    reusable = _existing_index_is_reusable(
        loaded=[],
        report={
            "status": "built",
            "object_count": 0,
            "empty_index_cause": "zero_detections",
            "runtime_preflight": {
                "backends": {
                    "yolo_world": {
                        "support_level": "required",
                        "status": "configured",
                    }
                }
            },
        },
    )

    assert reusable is True
