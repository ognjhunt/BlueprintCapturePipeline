"""The ArtiFixer stage must be given time for the repair pass it is funded for.

PR #1361 raised the semantic-teacher cost allowance to cover one first pass
plus one bounded repair pass, but left the time allowance sized for one pass.
Scene 839873 run ...dfd77804-r2-web-20260829T223435Z executed the repair round
for the first time and was then killed mid-retrain:

    TimeoutExpired: '.../artifixer3d_observed_object_removal/run'
    timed out after 7200 seconds

Measured from that lane's own markers, one ArtiFixer execute pass ran
16:41:56Z -> 17:42:57Z = 3_661s. Two passes plus the repair edit and two
independent reviews need roughly 8_400s, so the stage has to be sized for two
passes -- and it has to stay inside the compute cap that already exists.
"""

from __future__ import annotations

from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import (
    BOOTSTRAP_TRANSFER_AND_NO_SPEND_RESERVE_SECONDS,
    GPU_STAGE_TIMEOUT_SECONDS,
    MAX_HOURLY_RATE_USD,
    MAX_PROVIDER_COMPUTE_SPEND_USD,
    REQUIRED_PARENT_TTL_SECONDS,
)

# Measured on scene 839873, run ...b0b7908c: a single ArtiFixer execute pass.
MEASURED_SINGLE_PASS_SECONDS = 3_661
# Repair edit plus two independent eight-frame reviews, measured generously.
MEASURED_REPAIR_AND_REVIEW_SECONDS = 1_100
# Instance creation through container start, measured on the same lane.
MEASURED_BOOTSTRAP_SECONDS = 360


def test_artifixer_stage_is_sized_for_two_passes() -> None:
    required = 2 * MEASURED_SINGLE_PASS_SECONDS + MEASURED_REPAIR_AND_REVIEW_SECONDS
    assert GPU_STAGE_TIMEOUT_SECONDS["artifixer3d_observed_object_removal"] >= required


def test_parent_ttl_still_fits_the_existing_compute_cap() -> None:
    """Sizing for two passes must not quietly raise what a run may spend."""

    worst_case = MAX_HOURLY_RATE_USD * REQUIRED_PARENT_TTL_SECONDS / 3_600
    assert worst_case <= MAX_PROVIDER_COMPUTE_SPEND_USD + 1e-9
    # The preparation contract refuses when compute cap < hourly * TTL.
    assert not (
        MAX_PROVIDER_COMPUTE_SPEND_USD + 1e-9
        < MAX_HOURLY_RATE_USD * REQUIRED_PARENT_TTL_SECONDS / 3_600
    )


def test_bootstrap_reserve_still_far_exceeds_observed_bootstrap() -> None:
    """Time was taken from an over-provisioned reserve, not from safety."""

    assert BOOTSTRAP_TRANSFER_AND_NO_SPEND_RESERVE_SECONDS >= (
        10 * MEASURED_BOOTSTRAP_SECONDS
    )


def test_ttl_is_the_sum_of_its_declared_parts() -> None:
    from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import (
        OUTPUT_AND_CLOSURE_RESERVE_SECONDS,
        SERIAL_GPU_STAGE_TIMEOUT_SECONDS,
    )

    assert SERIAL_GPU_STAGE_TIMEOUT_SECONDS == sum(GPU_STAGE_TIMEOUT_SECONDS.values())
    assert REQUIRED_PARENT_TTL_SECONDS == (
        SERIAL_GPU_STAGE_TIMEOUT_SECONDS
        + BOOTSTRAP_TRANSFER_AND_NO_SPEND_RESERVE_SECONDS
        + OUTPUT_AND_CLOSURE_RESERVE_SECONDS
    )
