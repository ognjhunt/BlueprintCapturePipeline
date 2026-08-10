from __future__ import annotations

import pytest

from blueprint_pipeline.provider_runtime_bundle_contract import (
    PROVIDER_RUNTIME_BUNDLE_KINDS,
    provider_runtime_contract_blockers,
)

ARTICULATED_KIND = "adp009d_articulated_arena"
LANE_RUNNER_TEXT = (
    "adp009d_native_microcheck.json ARENA_REVISION ISAAC_LAB_REVISION "
    "provider_zero_required_after_return candidate_policy_queried"
)
LANE_ENTRYPOINT_TEXT = (
    "adp009d_worker_failed_without_runtime_result adp009d_native_microcheck.json"
)


def test_the_articulated_payload_has_its_own_kind() -> None:
    """Its required entries genuinely differ from the rigid lane's.

    The rigid kind demands assets/sage_collision_overlay.usda, produced by an
    inspector that exists to identify one scene - default prim, mesh inventory,
    two named collider prims. A different scene cannot make one. Exempting the
    rigid kind from its own overlay would weaken a gate reaching paid hardware;
    a separate kind states what this payload actually ships instead.
    """

    assert ARTICULATED_KIND in PROVIDER_RUNTIME_BUNDLE_KINDS


def test_a_compliant_articulated_bundle_has_no_contract_blockers() -> None:
    blockers = provider_runtime_contract_blockers(
        provider_bundle_kind=ARTICULATED_KIND,
        entrypoint_text=LANE_ENTRYPOINT_TEXT,
        runner_text=LANE_RUNNER_TEXT,
    )

    assert blockers == []


def test_a_runner_missing_the_lane_tokens_is_still_refused() -> None:
    """A new kind must not become a way around the runner contract."""

    blockers = provider_runtime_contract_blockers(
        provider_bundle_kind=ARTICULATED_KIND,
        entrypoint_text=LANE_ENTRYPOINT_TEXT,
        runner_text="print('hello')",
    )

    assert any("runner_missing" in b for b in blockers)


def test_an_entrypoint_that_cannot_report_a_dead_worker_is_refused() -> None:
    """The fallback result is why a crashed run still says something."""

    blockers = provider_runtime_contract_blockers(
        provider_bundle_kind=ARTICULATED_KIND,
        entrypoint_text="echo hi",
        runner_text=LANE_RUNNER_TEXT,
    )

    assert blockers != []


def test_the_rigid_kind_is_untouched() -> None:
    """It is already qualified; adding a kind must not alter it."""

    assert (
        provider_runtime_contract_blockers(
            provider_bundle_kind="adp009d_isaac",
            entrypoint_text=LANE_ENTRYPOINT_TEXT,
            runner_text=LANE_RUNNER_TEXT,
        )
        == []
    )


def test_an_unknown_kind_still_raises() -> None:
    with pytest.raises(ValueError):
        provider_runtime_contract_blockers(
            provider_bundle_kind="not_a_kind",
            entrypoint_text="",
            runner_text="",
        )
