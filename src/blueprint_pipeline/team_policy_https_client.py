"""A robot team's own policy for its own robot, reached over authenticated HTTPS (``blueprint_policy_https.v1``).

Beta journey step: L5 team policy evaluation. The DROID interface fixes the
robot (a Franka with a Robotiq gripper) and the observation (two 224x224
views). A team bringing its own arm (``team_embodiment``) needs a contract that
fixes neither, so the request carries exactly what the team declared:

    POST endpoint_url
    {"schema_version": "blueprint.policy_https_request.v1", "request_id": "...",
     "policy_interface_id": "blueprint_policy_https.v1",
     "observation": {"images": {"<camera name>": {"encoding": "png_base64", "height": H, "width": W,
                                                  "data": "..."}},
                     "state": {"joint_position": [one per declared arm joint, radians or metres],
                               "gripper_position": [closure, 0 open .. 1 closed] or []},
                     "prompt": "<the site job's instruction>"}}
    -> {"actions": [[arm joint targets..., gripper closure], ...]}   (1..action_chunk_rows rows)

and the metadata endpoint answers the team's declared identity, now bound to
the declared robot by ``embodiment_digest``. Transport, bounds, identity
checks and evidence are the DROID client's, unchanged
(``droid_https_policy_client``); the simulator executes and scores every
action, and nothing else the service returns is read as an outcome.
"""

from __future__ import annotations

import base64
import io
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from .droid_https_policy_client import DroidHttpsTeamPolicyClient

INTERFACE_ID = "blueprint_policy_https.v1"
REQUEST_SCHEMA = "blueprint.policy_https_request.v1"
METADATA_SCHEMA = "blueprint.policy_https_metadata.v1"
TRANSPORT = "blueprint_policy_https_json_bearer"
METADATA_KEYS = frozenset({"schema_version", "policy_id", "policy_interface_id", "action_space",
                           "action_chunk_rows", "model_identity_digest", "embodiment_digest"})
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class TeamPolicySpec:
    policy_id: str
    endpoint_url: str
    metadata_url: str
    allowed_origin: str
    model_identity_digest: str
    embodiment_digest: str
    action_chunk_rows: int
    open_loop_horizon: int
    camera_names: list[str] = field(default_factory=list)
    action_space: str = "joint_position"
    policy_interface_id: str = INTERFACE_ID

    def validate(self) -> None:
        for url in (self.endpoint_url, self.metadata_url):
            parsed = urlsplit(url)
            if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                    or parsed.fragment or f"{parsed.scheme}://{parsed.netloc}" != self.allowed_origin):
                raise ValueError("team_policy_https_endpoint_not_approved")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.policy_id):
            raise ValueError("team_policy_https_policy_id_invalid")
        if self.policy_interface_id != INTERFACE_ID or self.action_space != "joint_position":
            raise ValueError("team_policy_https_interface_mismatch")
        if not _DIGEST.fullmatch(self.model_identity_digest) or not _DIGEST.fullmatch(self.embodiment_digest):
            raise ValueError("team_policy_https_identity_invalid")
        if (type(self.action_chunk_rows) is not int or not 1 <= self.action_chunk_rows <= 64
                or not 1 <= self.open_loop_horizon <= self.action_chunk_rows or not self.camera_names):
            raise ValueError("team_policy_https_chunk_contract_invalid")

    def expected_metadata(self) -> dict[str, Any]:
        self.validate()
        return {"schema_version": METADATA_SCHEMA, "policy_id": self.policy_id,
                "policy_interface_id": self.policy_interface_id, "action_space": self.action_space,
                "action_chunk_rows": self.action_chunk_rows, "model_identity_digest": self.model_identity_digest,
                "embodiment_digest": self.embodiment_digest}


def encode_team_observation(observation: Mapping[str, Any], *, camera_names: list[str]) -> dict[str, Any]:
    """The team's declared views as lossless PNG, its joint and gripper state, and the prompt."""

    import numpy as np
    from PIL import Image

    images = observation.get("images") if isinstance(observation.get("images"), Mapping) else {}
    state = observation.get("state") if isinstance(observation.get("state"), Mapping) else {}
    if (set(observation) != {"images", "state", "prompt"} or sorted(images) != sorted(camera_names)
            or set(state) != {"joint_position", "gripper_position"}):
        raise ValueError("team_policy_https_observation_keys_invalid")
    wire: dict[str, Any] = {"images": {}, "state": {}}
    for name in camera_names:
        image = np.asarray(images[name])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("team_policy_https_observation_image_invalid")
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        wire["images"][name] = {"encoding": "png_base64", "height": int(image.shape[0]),
                                "width": int(image.shape[1]), "data": base64.b64encode(buffer.getvalue()).decode()}
    for key in ("joint_position", "gripper_position"):
        values = [float(value) for value in np.asarray(state[key], dtype=float).reshape(-1)]
        if (key == "joint_position" and not values) or not all(math.isfinite(value) for value in values):
            raise ValueError("team_policy_https_observation_state_invalid")
        wire["state"][key] = values
    if not isinstance(observation["prompt"], str) or not observation["prompt"].strip():
        raise ValueError("team_policy_https_observation_prompt_invalid")
    wire["prompt"] = observation["prompt"]
    return wire


class TeamPolicyHttpsClient(DroidHttpsTeamPolicyClient):
    """The team's service for its own robot, with per-operation declared-identity checks."""

    interface_id = INTERFACE_ID
    request_schema = REQUEST_SCHEMA
    transport_name = TRANSPORT

    def _encode(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        return encode_team_observation(observation, camera_names=list(self._spec.camera_names))


__all__ = ["INTERFACE_ID", "METADATA_KEYS", "METADATA_SCHEMA", "TeamPolicyHttpsClient", "TeamPolicySpec",
           "encode_team_observation"]
