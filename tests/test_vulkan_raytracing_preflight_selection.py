from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_vulkan_raytracing_preflight.py"
)
_spec = importlib.util.spec_from_file_location("_vk_preflight", _SCRIPT)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
evaluate_device_selection = _module.evaluate_device_selection


def _device(index: int, capable: bool) -> dict:
    missing = (
        []
        if capable
        else [
            "VK_KHR_acceleration_structure",
            "VK_KHR_deferred_host_operations",
            "VK_KHR_ray_tracing_pipeline",
        ]
    )
    return {
        "device_index": index,
        "extension_count": 252 if capable else 114,
        "required_raytracing_extensions_present": capable,
        "missing_required_extensions": missing,
    }


def test_mixed_host_without_a_pinned_device_fails_closed() -> None:
    """The exact host configuration a live OVRTX run passed on, then rendered black.

    One ray-tracing GPU plus one 114-extension software rasterizer.  Which one
    the renderer binds decides whether frames contain pixels, and nothing pins it.
    """

    rows = [_device(0, True), _device(1, False)]

    blockers, incapable = evaluate_device_selection(rows, None)

    assert blockers == ["vulkan_raytracing_device_selection_ambiguous"]
    assert incapable == [1]


def test_pinning_the_capable_device_clears_the_ambiguity() -> None:
    rows = [_device(0, True), _device(1, False)]

    blockers, incapable = evaluate_device_selection(rows, 0)

    assert blockers == []
    assert incapable == [1]


def test_pinning_the_incapable_device_is_blocked_by_name() -> None:
    rows = [_device(0, True), _device(1, False)]

    blockers, _ = evaluate_device_selection(rows, 1)

    assert blockers == ["vulkan_raytracing_selected_device_incapable"]


def test_out_of_range_selection_never_silently_passes() -> None:
    rows = [_device(0, True), _device(1, False)]

    for index in (2, -1, 99):
        blockers, _ = evaluate_device_selection(rows, index)
        assert blockers == ["vulkan_raytracing_selected_device_out_of_range"]


def test_uniform_hosts_are_left_to_the_existing_checks() -> None:
    """All-capable and none-capable hosts are unambiguous; do not double-report."""

    all_capable = [_device(0, True), _device(1, True)]
    assert evaluate_device_selection(all_capable, None) == ([], [])

    # None capable is already reported as vulkan_raytracing_extensions_missing.
    none_capable = [_device(0, False), _device(1, False)]
    blockers, incapable = evaluate_device_selection(none_capable, None)
    assert blockers == []
    assert incapable == [0, 1]

    # A single capable device needs no pin.
    assert evaluate_device_selection([_device(0, True)], None) == ([], [])

    assert evaluate_device_selection([], None) == ([], [])


def test_preflight_reports_the_capability_split_for_review() -> None:
    """The receipt must carry enough to re-derive the verdict without the host."""

    source = _SCRIPT.read_text(encoding="utf-8")
    assert '"raytracing_capable_device_count"' in source
    assert '"raytracing_incapable_device_indices"' in source
    assert '"selected_device_index"' in source
    # The rule must stay callable without Vulkan so it can be tested off-GPU.
    rule = source[source.index("def evaluate_device_selection(") :]
    rule = rule[: rule.index("def probe(")]
    assert "ctypes" not in rule
    assert "library." not in rule


@pytest.mark.parametrize("selected", [None, 0, 1])
def test_rule_never_raises_on_any_selection(selected: int | None) -> None:
    rows = [_device(0, False), _device(1, True), _device(2, False)]

    blockers, incapable = evaluate_device_selection(rows, selected)

    assert isinstance(blockers, list)
    assert incapable == [0, 2]
