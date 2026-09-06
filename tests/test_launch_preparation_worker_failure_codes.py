"""A worker failure row carries the lane's own blocker code, never free text."""

from __future__ import annotations

from blueprint_pipeline import task_evaluation_launch_preparation_worker as worker
from blueprint_pipeline.public_scene_inpainting_inputs import PublicSceneInpaintingInputError
from blueprint_pipeline.task_evaluation_scene_configuration_submission_inputs import (
    SceneConfigurationSubmissionError,
)


def _raised(exc: BaseException) -> BaseException:
    try:
        raise exc
    except BaseException as caught:  # noqa: BLE001 - the exception is the subject
        return caught


def test_a_lane_code_message_is_carried_on_the_failure_row() -> None:
    """2026-09-06: two parent replays reported only ``PublicSceneInpaintingInputError`` and
    ``SceneConfigurationSubmissionError``; the codes those lanes raised were dropped."""

    assert worker.worker_failure_blocker(
        _raised(SceneConfigurationSubmissionError("scene_configuration_submission_catalog_missing"))
    ) == (
        "launch_preparation_worker_failed:SceneConfigurationSubmissionError"
        ":scene_configuration_submission_catalog_missing"
    )
    # PublicSceneInpaintingInputError joins a set of codes with ";" — the whole bounded set is carried.
    assert worker.worker_failure_blocker(
        _raised(PublicSceneInpaintingInputError(["public_scene_inpainting_frame_set_invalid:calibrated_views/12",
                                                 "public_scene_inpainting_mask_source_missing"]))
    ) == (
        "launch_preparation_worker_failed:PublicSceneInpaintingInputError"
        ":public_scene_inpainting_frame_set_invalid:calibrated_views/12;public_scene_inpainting_mask_source_missing"
    )


def test_free_text_and_unbounded_messages_are_never_carried() -> None:
    assert worker.worker_failure_blocker(_raised(RuntimeError("disk full at /var/lib/blueprint"))) == (
        "launch_preparation_worker_failed:RuntimeError"
    )
    assert worker.worker_failure_blocker(_raised(OSError(28, "No space left on device"))) == (
        "launch_preparation_worker_failed:OSError"
    )
    assert worker.worker_failure_blocker(_raised(ValueError("x" * 400))) == (
        "launch_preparation_worker_failed:ValueError"
    )
    assert worker.worker_failure_blocker(_raised(ValueError(""))) == "launch_preparation_worker_failed:ValueError"
