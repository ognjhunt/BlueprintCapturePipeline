"""The Joint OVRTX readiness budget must clear observed warm-up, not graze it.

On 2026-08-17 run `adp-joint-agent-840920-task-a-99741f78-r1` sealed
`joint_agent_local_ovrtx_renderer_not_ready` while its own service log recorded
"OVRTX renderer warmed up - GPU is ready" seconds *after* the poll budget
expired.  The GPU was fine; the runner killed a working renderer and reported a
false negative -- the same shape `087125e0` fixed for the daemon probe, one step
later in the same script.

The budget was 180 polls x 5s = 900s against a warm-up observed at ~900s. A
budget that equals the thing it waits for fails half the time, so this pins a
margin rather than a specific number.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_adp_joint_agent_provider_runtime.sh"
)

#: Longest OVRTX warm-up actually observed on a paid instance, in seconds.
OBSERVED_WARMUP_SECONDS = 900

#: The budget must exceed observed warm-up by at least this factor. Warm-up is
#: GPU- and scene-dependent, so "a bit more than the worst we saw" is not
#: enough; a slower card must not reproduce the same false negative.
REQUIRED_MARGIN = 1.5


def _source() -> str:
    return RUNNER.read_text(encoding="utf-8")


def _readiness_budget_seconds(source: str) -> int:
    """Poll count x sleep interval for the renderer readiness loop."""

    polls = re.search(
        r'OVRTX_RENDERER_READY_POLLS="\$\{OVRTX_RENDERER_READY_POLLS:-(\d+)\}"',
        source,
    )
    assert polls, "renderer readiness poll count not found"
    loop = re.search(
        r'for _ in \$\(seq 1 "\$\{OVRTX_RENDERER_READY_POLLS\}"\); do(.*?)done',
        source,
        re.S,
    )
    assert loop, "renderer readiness loop not found"
    sleep = re.search(r"sleep (\d+)", loop.group(1))
    assert sleep, "renderer readiness sleep interval not found"
    return int(polls.group(1)) * int(sleep.group(1))


def test_runner_exists() -> None:
    assert RUNNER.is_file()


def test_readiness_budget_clears_observed_warmup_with_margin() -> None:
    budget = _readiness_budget_seconds(_source())
    required = OBSERVED_WARMUP_SECONDS * REQUIRED_MARGIN
    assert budget >= required, (
        f"renderer readiness budget is {budget}s; observed warm-up reached "
        f"{OBSERVED_WARMUP_SECONDS}s, so the budget must be at least "
        f"{required:.0f}s or a slow GPU reproduces the false negative that "
        "killed a working renderer on 2026-08-17"
    )


def test_the_old_grazing_budget_would_fail_this_test() -> None:
    """Guards the guard: the pre-fix 900s budget must not satisfy the rule."""

    assert 180 * 5 < OBSERVED_WARMUP_SECONDS * REQUIRED_MARGIN


def test_readiness_still_gates_on_gpu_initialized_not_health_alone() -> None:
    """`/health` returns 200 while warm-up runs; only gpu_initialized is ready."""

    source = _source()
    assert '"gpu_initialized":true' in source
    assert "joint_agent_local_ovrtx_renderer_not_ready" in source


def test_budget_is_overridable_without_editing_the_script() -> None:
    """A slower GPU should be answerable by configuration, not a code change."""

    assert "${OVRTX_RENDERER_READY_POLLS:-" in _source()


@pytest.mark.parametrize("polls,expected", [(180, 900), (300, 1500)])
def test_budget_arithmetic(polls: int, expected: int) -> None:
    assert polls * 5 == expected
