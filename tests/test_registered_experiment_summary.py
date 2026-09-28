"""Native registered action receipts reach the public path-free GC summary."""

import json

from blueprint_pipeline.control_plane_storage_gc_reasons import build_storage_gc_summary


def test_registered_action_summary_keeps_measured_logical_and_allocated_bytes():
    phase = dict(
        enabled=True,
        outcomes=[
            dict(
                action_id="private-id",
                intent_id="private-intent",
                decision="retired",
                reason="evidence_preserved",
                receipt={"path": "/PRIVATE/secret"},
                removed_logical_bytes=7,
                removed_allocated_bytes=4096,
            )
        ],
    )
    summary = build_storage_gc_summary({"status": "applied", "registered_experiments": phase})
    entry = summary["phases"]["registered_experiments"]
    assert entry["enabled"] is True
    assert entry["candidate_bytes"] is None
    assert entry["removed_or_offloaded_bytes"] == 4096
    assert entry["removed_logical_bytes"] == 7 and entry["removed_allocated_bytes"] == 4096
    assert entry["outcomes"] == [
        {"decision": "retired", "reason": "evidence_preserved", "count": 1}
    ]
    assert "private" not in json.dumps(entry) and "PRIVATE" not in json.dumps(entry)


def test_registered_action_summary_never_sizes_kept_unknown_or_corrupt_bytes():
    summary = build_storage_gc_summary(
        {
            "status": "applied",
            "registered_experiments": dict(
                enabled=True,
                outcomes=[
                    dict(
                        decision="kept",
                        reason="current_reference",
                        removed_logical_bytes=0,
                        removed_allocated_bytes=0,
                    ),
                    dict(
                        decision="kept",
                        reason="/PRIVATE/error",
                        removed_logical_bytes=True,
                        removed_allocated_bytes=-1,
                    ),
                ],
            ),
        }
    )
    entry = summary["phases"]["registered_experiments"]
    assert entry["candidate_bytes"] is None
    assert entry["removed_logical_bytes"] is entry["removed_allocated_bytes"] is None
    assert entry["retained_by_reason"] == {
        "current_reference": {"count": 1, "bytes": None},
        "unrecognized_reason": {"count": 1, "bytes": None},
    }
    assert "/PRIVATE" not in json.dumps(entry)
