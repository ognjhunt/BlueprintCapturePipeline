"""ADP-050/day 28: bridge an existing simulator to controlled policy inputs.

Bindings and action translation belong to trusted robot/task configuration.
In particular, the adapter never guesses joint order or gripper conventions.
"""
from __future__ import annotations

import io
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from .company_policy_container_contract_v2 import validate_company_policy_container_contract_v2
from .controlled_policy_actions import validate_action_response


class ControlledSimulatorAdapter:
    """Wrap the existing Isaac EpisodeEnvironment without exporting its assets."""
    def __init__(self, *, native_environment: Any, contract: Mapping[str, Any],
                 camera_bindings: Mapping[str, str], state_bindings: Mapping[str, str],
                 state_units: Mapping[str, str], control_frequency_hz: float,
                 prompt: str,
                 translate_action: Callable[[Sequence[float], Any, Mapping[str, Any]], Sequence[float]],
                 terminal: Callable[[], bool], stop_controller: Callable[[], None],
                 outcome_recorder: Any = None):
        self.native = native_environment
        self.contract = validate_company_policy_container_contract_v2(contract)
        observation = self.contract["observation_schema"]
        if (dict(state_units) != {row["name"]: row["unit"] for row in observation["state_fields"]}
                or isinstance(control_frequency_hz, bool)
                or control_frequency_hz != observation["control_frequency_hz"]):
            raise ValueError("controlled_simulator_units_or_control_rate_mismatch")
        if (set(camera_bindings) != {row["name"] for row in observation["cameras"]}
                or set(state_bindings) != {row["name"] for row in observation["state_fields"]}
                or any(not isinstance(value, str) or not value for value in [*camera_bindings.values(), *state_bindings.values()])):
            raise ValueError("controlled_simulator_observation_bindings_invalid")
        self.camera_bindings, self.state_bindings = dict(camera_bindings), dict(state_bindings)
        self.prompt, self.translate_action = prompt, translate_action
        self.terminal, self.stop_controller = terminal, stop_controller
        self.outcome_recorder = outcome_recorder

    def read_policy_inputs(self) -> Mapping[str, Any]:
        inputs = self.native.read_policy_inputs()
        cameras = {}
        for camera in self.contract["observation_schema"]["cameras"]:
            frame = np.asarray(inputs[self.camera_bindings[camera["name"]]])
            if frame.dtype != np.uint8 or list(frame.shape) != [camera["height"], camera["width"], 3]:
                raise ValueError("controlled_simulator_camera_interface_mismatch")
            output = io.BytesIO()
            Image.fromarray(frame).save(output, format="PNG")
            cameras[camera["name"]] = output.getvalue()
        state = {}
        for field in self.contract["observation_schema"]["state_fields"]:
            raw = np.asarray(inputs[self.state_bindings[field["name"]]])
            # Existing arm adapters expose gripper_position as a scalar. The
            # explicit width-one field converts it to its declared vector.
            if raw.shape == () and field["shape"] == [1]:
                raw = raw.reshape(1)
            if (raw.dtype.kind not in "iuf" or list(raw.shape) != field["shape"]
                    or not np.isfinite(raw).all()):
                raise ValueError("controlled_simulator_state_interface_mismatch")
            state[field["name"]] = raw.tolist()
        return {"camera_pngs": cameras, "robot_state": state, "prompt": self.prompt}

    def apply_action_chunk(self, actions: list[list[float]], *, action_schema: Mapping[str, Any]) -> Mapping[str, Any]:
        if dict(action_schema) != self.contract["action_schema"]:
            raise ValueError("controlled_simulator_action_contract_mismatch")
        rows = validate_action_response({"actions": actions}, action_schema=dict(action_schema))["actions"]
        executed = 0
        for row in rows:
            if self.is_terminal():
                break
            # The configured translator can reuse the measured DROID/Isaac
            # velocity/position and gripper bridge. No inferred conversion.
            native_action = self.translate_action(row, self.native, action_schema)
            if not isinstance(native_action, (list, tuple, np.ndarray)) or not np.isfinite(np.asarray(native_action, dtype=float)).all():
                raise ValueError("controlled_simulator_native_action_invalid")
            self.native.step(native_action)
            if self.outcome_recorder is not None:
                self.outcome_recorder.capture(motor_step=True)
            executed += 1
        return {"executed_motor_steps": executed, "independent_outcome_required": True}

    def is_terminal(self) -> bool:
        value = self.terminal()
        if not isinstance(value, bool):
            raise ValueError("controlled_simulator_terminal_state_invalid")
        return value

    def stop(self) -> None:
        self.stop_controller()
