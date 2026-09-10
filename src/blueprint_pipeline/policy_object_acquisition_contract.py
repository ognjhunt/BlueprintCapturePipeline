"""Object-acquisition conditions and independent episode assessments.

This module neither changes a frozen matrix nor admits a GPU launch. Search
may relax initial object visibility only; renderer, calibration, reset, and
sensor freshness remain mandatory. Acquisition is separate from task success.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import Field, StrictBool, StrictInt, model_validator

from .decision_evidence_contracts import canonical_digest
from .policy_observation_information import Digest, Identifier, InformationContract

Visibility = Literal["visible", "partially_occluded", "initially_out_of_view"]
_VISIBILITY_TARGET = "ObservationManager.object_acquisition.initial_visibility"


class ObjectAcquisitionProtocol(InformationContract):
    mode: Literal["baseline_visible", "visual_search"] = "baseline_visible"
    initial_visibility: Visibility = "visible"
    minimum_target_pixels: Annotated[StrictInt, Field(ge=1)] = 8
    maximum_search_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    camera_calibration_digests: dict[Identifier, Digest]

    @model_validator(mode="after")
    def explicit_search(self) -> ObjectAcquisitionProtocol:
        if not self.camera_calibration_digests:
            raise ValueError("acquisition_policy_cameras_required")
        if self.mode == "baseline_visible" and (
            self.initial_visibility != "visible" or self.maximum_search_seconds is not None
        ):
            raise ValueError("baseline_requires_initial_visibility")
        if self.mode == "visual_search" and self.maximum_search_seconds is None:
            raise ValueError("visual_search_time_budget_required")
        return self


class AcquisitionCameraSample(InformationContract):
    frame_digest: Digest
    calibration_digest: Digest
    renderer_frame: Annotated[StrictInt, Field(ge=0)]
    rendered: StrictBool
    fresh: StrictBool
    target_pixels: Annotated[StrictInt, Field(ge=0)]
    target_visibility: Literal[
        "visible", "partially_occluded", "initially_out_of_view", "fully_occluded"
    ]

    @model_validator(mode="after")
    def visibility_matches_pixels(self) -> AcquisitionCameraSample:
        if (self.target_visibility in {"initially_out_of_view", "fully_occluded"}) != (
            self.target_pixels == 0
        ):
            raise ValueError("acquisition_visibility_pixel_count_inconsistent")
        return self


class AcquisitionSample(InformationContract):
    """Independent, frame-bound visibility readback, never a policy self-grade."""

    episode_id: Identifier
    target_object_id: Identifier
    episode_elapsed_s: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    episode_started_at_sim_time_s: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    observation_sim_time_s: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    physics_step: Annotated[StrictInt, Field(ge=0)]
    reset_digest: Digest
    visibility_source: Literal[
        "deterministic_simulator_segmentation",
        "deterministic_simulator_segmentation_and_bbox_occlusion",
    ]
    cameras: dict[Identifier, AcquisitionCameraSample]

    @model_validator(mode="after")
    def episode_time_origin(self) -> AcquisitionSample:
        if (
            abs(
                self.observation_sim_time_s
                - self.episode_started_at_sim_time_s
                - self.episode_elapsed_s
            )
            > 1e-9
        ):
            raise ValueError("acquisition_elapsed_time_origin_mismatch")
        return self


def assess_object_acquisition(
    protocol: ObjectAcquisitionProtocol,
    samples: Sequence[AcquisitionSample],
    *,
    reset_digest: str,
    episode_id: str,
    target_object_id: str,
    episode_started_at_sim_time_s: float,
    observation_series_complete: bool = False,
) -> dict[str, Any]:
    """Assess sampled acquisition timing while rejecting invalid camera input.

    Initially out of view is valid only in an explicit search protocol. Absence
    of acquisition is an outcome; absence of a valid observation is an error.
    This staged assessor does not replace any current runtime admission gate.
    """

    if not samples or samples[0].episode_elapsed_s != 0:
        raise ValueError("acquisition_initial_observation_required")
    previous = None
    first_acquired = None
    acquisition_lower_bound = None
    initial_visibility = None
    for sample in samples:
        if sample.episode_id != episode_id or sample.target_object_id != target_object_id:
            raise ValueError("acquisition_episode_or_target_mismatch")
        if sample.reset_digest != reset_digest:
            raise ValueError("acquisition_reset_binding_mismatch")
        if sample.episode_started_at_sim_time_s != episode_started_at_sim_time_s:
            raise ValueError("acquisition_episode_time_origin_mismatch")
        if set(sample.cameras) != set(protocol.camera_calibration_digests):
            raise ValueError("acquisition_policy_camera_set_mismatch")
        if previous and (
            sample.episode_elapsed_s <= previous.episode_elapsed_s
            or sample.physics_step <= previous.physics_step
        ):
            raise ValueError("acquisition_observation_time_not_increasing")
        for name, camera in sample.cameras.items():
            if not camera.rendered or not camera.fresh:
                raise ValueError("acquisition_camera_invalid_or_stale")
            if camera.calibration_digest != protocol.camera_calibration_digests[name]:
                raise ValueError("acquisition_camera_calibration_mismatch")
            if previous and camera.renderer_frame <= previous.cameras[name].renderer_frame:
                raise ValueError("acquisition_renderer_frame_not_increasing")
        visible = any(
            c.target_pixels >= protocol.minimum_target_pixels for c in sample.cameras.values()
        )
        if previous is None:
            if protocol.initial_visibility == "visible" and all(
                c.target_pixels >= protocol.minimum_target_pixels for c in sample.cameras.values()
            ):
                initial_visibility = "visible"
            elif all(
                c.target_visibility == "initially_out_of_view" for c in sample.cameras.values()
            ):
                initial_visibility = "initially_out_of_view"
            elif any(c.target_visibility == "partially_occluded" for c in sample.cameras.values()):
                initial_visibility = "partially_occluded"
            else:
                raise ValueError("acquisition_initial_visibility_protocol_mismatch")
            if initial_visibility != protocol.initial_visibility:
                raise ValueError("acquisition_initial_visibility_protocol_mismatch")
        if visible and first_acquired is None:
            first_acquired = sample.episode_elapsed_s
            acquisition_lower_bound = previous.episode_elapsed_s if previous else 0.0
        previous = sample
    deadline = protocol.maximum_search_seconds
    within_budget = first_acquired is not None and (deadline is None or first_acquired <= deadline)
    result = {
        "schema_version": "policy_object_acquisition_assessment.v1",
        "protocol_digest": canonical_digest(protocol.model_dump(mode="json")),
        "reset_digest": reset_digest,
        "episode_id": episode_id,
        "target_object_id": target_object_id,
        "episode_started_at_sim_time_s": episode_started_at_sim_time_s,
        "observations_digest": canonical_digest(
            {"samples": [s.model_dump(mode="json") for s in samples]}
        ),
        "initial_visibility": initial_visibility,
        "acquisition_status": "acquired"
        if within_budget
        else "budget_exceeded"
        if deadline is not None and samples[-1].episode_elapsed_s >= deadline
        else "not_acquired_in_policy_inputs"
        if observation_series_complete
        else "pending",
        "observation_series_complete": observation_series_complete,
        "first_observed_acquisition_seconds": first_acquired,
        "previous_observation_seconds": acquisition_lower_bound,
        "last_observation_seconds": samples[-1].episode_elapsed_s,
        "acquisition_is_task_success": False,
        "physical_outcome_proven": False,
    }
    result["assessment_digest"] = canonical_digest(result)
    return result
