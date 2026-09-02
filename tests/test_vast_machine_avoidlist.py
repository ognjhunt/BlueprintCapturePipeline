from __future__ import annotations

import json
from pathlib import Path

import pytest

import blueprint_pipeline.vast_provider_adapter as vpa
from blueprint_pipeline.provider_machine_avoidlist import avoidlist_machine_ids


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_published_machines_shape_excludes_known_bad_machine(tmp_path: Path) -> None:
    """The exact Scene 839873 inherited shape must exclude machine 144209."""

    avoidlist = _write(
        tmp_path / "adp_arena_vast_machine_avoidlist.json",
        {
            "schema_version": "vast_machine_avoidlist.v1",
            "generated_at": "2026-08-25T16:14:45.535750+00:00",
            "machines": [
                {
                    "machine_id": 144209,
                    "reason": (
                        "vast_startup_control_plane_did_not_reach_onstart_heartbeat"
                    ),
                    "recorded_at": "2026-08-25T16:14:45.535750+00:00",
                    "evidence": [
                        "adp-arena-policy-diagnostic-840920-task-a-groot-"
                        "fec521b4ef3c67ec25f73e154d4a8db7681fda55-20260825T151624Z-codex",
                        "adp-arena-policy-diagnostic-840920-task-a-groot-"
                        "fec521b4ef3c67ec25f73e154d4a8db7681fda55-20260825T154150Z-codex",
                        "adp-arena-policy-diagnostic-840920-task-a-groot-"
                        "428a6a5fc7d2330217a2c8ae1988042738680cd9-20260825T155442Z-codex",
                    ],
                }
            ],
        },
    )

    excluded = avoidlist_machine_ids(avoidlist)

    assert excluded == {144209}
    selected = vpa._select_offer(
        [
            {
                "id": 1,
                "machine_id": 144209,
                "gpu_name": "RTX A6000",
                "dph_total": 0.10,
            },
            {
                "id": 2,
                "machine_id": 24997,
                "gpu_name": "RTX A6000",
                "dph_total": 0.20,
            },
        ],
        max_hourly_rate=0.50,
        excluded_machine_ids=excluded,
    )
    assert selected is not None
    assert selected["machine_id"] == 24997


def test_current_machine_ids_and_entries_shapes_remain_supported(tmp_path: Path) -> None:
    avoidlist = _write(
        tmp_path / "vast_machine_avoidlist.json",
        {
            "schema_version": "vast_machine_avoidlist.v1",
            "machine_ids": ["5"],
            "entries": [{"machine_id": "6"}],
        },
    )

    assert avoidlist_machine_ids(avoidlist) == {5, 6}


@pytest.mark.parametrize(
    "value",
    [
        {"machines": {"machine_id": 144209}},
        {"machines": [{"reason": "missing machine id"}]},
        {"machine_ids": [144209, "not-an-id"]},
        {"schema_version": "vast_machine_avoidlist.v2", "machine_ids": [144209]},
    ],
)
def test_invalid_avoidlist_shape_fails_closed(tmp_path: Path, value: object) -> None:
    avoidlist = _write(tmp_path / "vast_machine_avoidlist.json", value)

    with pytest.raises(ValueError, match="vast_machine_avoidlist_invalid"):
        avoidlist_machine_ids(avoidlist)


def test_unparseable_avoidlist_fails_closed(tmp_path: Path) -> None:
    avoidlist = tmp_path / "vast_machine_avoidlist.json"
    avoidlist.write_text("{bad json", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="vast_machine_avoidlist_invalid:blocked_parse_failed",
    ):
        avoidlist_machine_ids(avoidlist)
