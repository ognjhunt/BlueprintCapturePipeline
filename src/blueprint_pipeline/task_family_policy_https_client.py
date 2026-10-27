"""Authenticated policy transport for qualified native family observations/actions.

The endpoint identity pins the native schemas and action bounds. No task state,
reward or evaluator output crosses this interface.
"""
from __future__ import annotations

import base64
import io
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from .droid_https_policy_client import DroidHttpsTeamPolicyClient

INTERFACE_ID = "task_family_policy_https.v1"


@dataclass(frozen=True)
class TaskFamilyPolicySpec:
    policy_id: str
    endpoint_url: str
    metadata_url: str
    allowed_origin: str
    model_identity_digest: str
    embodiment_id: str
    controller: str
    action_schema_id: str
    observation_schema_id: str
    action_bounds: list[list[float]]
    action_chunk_rows: int = 1
    open_loop_horizon: int = 1
    action_space: str = "native_schema"
    policy_interface_id: str = INTERFACE_ID
    camera_id: str | None = None
    image_height: int | None = None
    image_width: int | None = None

    def validate(self) -> None:
        for url in (self.endpoint_url, self.metadata_url):
            parsed = urlsplit(url)
            if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                    or parsed.fragment or f"{parsed.scheme}://{parsed.netloc}" != self.allowed_origin):
                raise ValueError("task_family_policy_endpoint_not_approved")
        for value in (self.policy_id, self.embodiment_id, self.controller,
                      self.action_schema_id, self.observation_schema_id):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
                raise ValueError("task_family_policy_schema_identity_invalid")
        if any(value is not None for value in (self.camera_id, self.image_height, self.image_width)):
            if (not isinstance(self.camera_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.camera_id)
                    or type(self.image_height) is not int or not 1 <= self.image_height <= 16384
                    or type(self.image_width) is not int or not 1 <= self.image_width <= 16384):
                raise ValueError("task_family_policy_image_contract_invalid")
        if (self.policy_interface_id != INTERFACE_ID or self.action_space != "native_schema"
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", self.model_identity_digest)):
            raise ValueError("task_family_policy_identity_invalid")
        if (not isinstance(self.action_bounds, list) or not self.action_bounds
                or any(not isinstance(row, list) or len(row) != 2
                       or any(type(value) not in (int, float) for value in row) for row in self.action_bounds)):
            raise ValueError("task_family_policy_action_contract_invalid")
        bounds = np.asarray(self.action_bounds, dtype=float)
        if (bounds.ndim != 2 or bounds.shape[0] < 1 or bounds.shape[1] != 2
                or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1])
                or type(self.action_chunk_rows) is not int or not 1 <= self.action_chunk_rows <= 64
                or type(self.open_loop_horizon) is not int
                or not 1 <= self.open_loop_horizon <= self.action_chunk_rows):
            raise ValueError("task_family_policy_action_contract_invalid")

    def expected_metadata(self) -> dict[str, Any]:
        self.validate()
        return {"schema_version": "blueprint.task_family_policy_metadata.v1",
                **{key: getattr(self, key) for key in (
                    "policy_id", "policy_interface_id", "model_identity_digest", "embodiment_id",
                    "controller", "action_schema_id", "observation_schema_id", "action_bounds",
                    "action_space", "action_chunk_rows", "open_loop_horizon")},
                **({key: getattr(self, key) for key in ("camera_id", "image_height", "image_width")}
                   if self.camera_id is not None else {})}


class TaskFamilyPolicyHttpsClient(DroidHttpsTeamPolicyClient):
    """The existing bounded, identity checked HTTPS transport with native schemas."""

    interface_id = INTERFACE_ID
    request_schema = "blueprint.task_family_policy_request.v1"
    transport_name = "task_family_policy_https_json_bearer"

    def __init__(self, *, prompt: str, **kwargs: Any) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("task_family_policy_instruction_missing")
        self.prompt = prompt
        super().__init__(**kwargs)

    def _encode(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        from PIL import Image

        if set(observation) != {"rgb", "proprioception"}:
            raise ValueError("task_family_policy_observation_fields_invalid")
        pixels = np.asarray(observation["rgb"])
        proprio = np.asarray(observation["proprioception"], dtype=float)
        if (pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3
                or proprio.ndim != 1 or not np.isfinite(proprio).all()):
            raise ValueError("task_family_policy_observation_invalid")
        if self._spec.image_height is not None and pixels.shape != (self._spec.image_height, self._spec.image_width, 3):
            raise ValueError("task_family_policy_image_shape_mismatch")
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, format="PNG")
        return {"rgb": {"encoding": "png_base64", "height": int(pixels.shape[0]),
                        "width": int(pixels.shape[1]),
                        "data": base64.b64encode(stream.getvalue()).decode("ascii")},
                "proprioception": proprio.tolist(), "prompt": self.prompt}


__all__ = ["INTERFACE_ID", "TaskFamilyPolicySpec", "TaskFamilyPolicyHttpsClient"]
