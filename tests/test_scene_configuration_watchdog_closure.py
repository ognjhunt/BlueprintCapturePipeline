from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_scene_configuration_vast as scene_vast


def test_scene_lane_routes_adapter_closure_through_bound_helper() -> None:
    source = inspect.getsource(scene_vast.run_scene_configuration_vast)

    assert "watchdog_close = _close_watchdog_after_adapter(" in source


def test_rejected_create_uses_double_inventory_zero_closer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def close_without_allocation(**kwargs):  # type: ignore[no-untyped-def]
        calls.append("double_inventory_zero")
        assert kwargs == {"job_dir": tmp_path, "handle": "watchdog"}
        return {
            "status": "provider_terminal",
            "provider_absence_confirmed": True,
        }

    monkeypatch.setattr(
        scene_vast,
        "close_independent_vast_watchdog_without_allocation",
        close_without_allocation,
    )
    monkeypatch.setattr(
        scene_vast,
        "close_independent_vast_watchdog",
        lambda **_kwargs: pytest.fail("ambiguous closer must not retain this watchdog"),
    )

    result = scene_vast._close_watchdog_after_adapter(
        job_dir=tmp_path,
        handle="watchdog",
        adapter={
            "provider_create_attempted": True,
            "vast_side_effects_may_have_occurred": False,
        },
        teardown={"continuing_spend_from_this_run": False},
        instance_ids=[],
    )

    assert result["status"] == "provider_terminal"
    assert result["provider_absence_confirmed"] is True
    assert calls == ["double_inventory_zero"]


@pytest.mark.parametrize(
    "adapter,teardown",
    [
        (
            {
                "provider_create_attempted": True,
                "vast_side_effects_may_have_occurred": True,
            },
            {"continuing_spend_from_this_run": False},
        ),
        (
            {
                "provider_create_attempted": True,
                "vast_side_effects_may_have_occurred": False,
            },
            {"continuing_spend_from_this_run": True},
        ),
    ],
)
def test_ambiguous_or_unclosed_create_keeps_existing_watchdog_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adapter: dict[str, bool],
    teardown: dict[str, bool],
) -> None:
    captured: dict[str, object] = {}

    def close(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {"status": "retained_until_hard_ttl"}

    monkeypatch.setattr(scene_vast, "close_independent_vast_watchdog", close)
    monkeypatch.setattr(
        scene_vast,
        "close_independent_vast_watchdog_without_allocation",
        lambda **_kwargs: pytest.fail("provider zero is not proven"),
    )

    result = scene_vast._close_watchdog_after_adapter(
        job_dir=tmp_path,
        handle="watchdog",
        adapter=adapter,
        teardown=teardown,
        instance_ids=[],
    )

    assert result["status"] == "retained_until_hard_ttl"
    assert captured["provider_allocation_impossible"] is False
    assert captured["provider_teardown_completed"] is (
        teardown["continuing_spend_from_this_run"] is False
    )
