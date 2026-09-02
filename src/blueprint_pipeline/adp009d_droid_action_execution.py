"""Execute DROID policy action chunks in the ADP-009D Isaac environment.

The compatibility DROID action path uses seven joint-velocity dimensions and
one absolute gripper command.  Arena's available DROID embodiment accepts
absolute joint-position targets instead, so each bounds-validated velocity row
is converted to a bounded position increment from the *currently observed*
joints.  Treating such a raw row as seven positions made 448/480 actions hit
joint limits in a paid run and was a harness fault, not a policy result.  The
current frozen pi05 jointpos config and GR00T adapter instead return absolute
joint positions; those use the same validated direct-position execution but
retain distinct candidate-specific source representations in their receipts.

The environment's ``sim.dt = 1/120`` with ``decimation = 8`` means one
``env.step()`` advances 1/15 s, exactly DROID's 15 Hz control rate.  So one
policy action row is still one environment step, with no resampling.

What does *not* line up for free is the gripper.  DROID encodes it as a scalar
in [0, 1] where above 0.5 means closed; Arena's eighth action dimension has its
own convention, and guessing it would silently invert every grasp.  The
convention is therefore a required, measured input rather than a default: see
``GripperConvention`` and the probe contract it documents.

This module is pure arithmetic so it can be tested without a GPU.  It never
queries a policy and never steps a simulator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

try:  # flat provider-bundle layout
    from droid_policy_bridge import (
        DROID_MAX_JOINT_DELTA_RAD,
        DROID_SOURCE_REVISION,
        OPENPI_SOURCE_REVISION,
        droid_action_to_mujoco_targets,
    )
except ModuleNotFoundError:  # repository package
    from .droid_policy_bridge import (
        DROID_MAX_JOINT_DELTA_RAD,
        DROID_SOURCE_REVISION,
        OPENPI_SOURCE_REVISION,
        droid_action_to_mujoco_targets,
    )

ACTION_EXECUTION_SCHEMA_VERSION = "adp009d_droid_action_execution.v2"
ACTION_SPACE_JOINT_VELOCITY = "joint_velocity"
ACTION_SPACE_JOINT_POSITION = "joint_position"
SOURCE_DROID_VELOCITY = "droid_joint_velocity_plus_absolute_gripper"
SOURCE_PI05_POSITION = (
    "pi05_droid_jointpos_polaris_absolute_joint_position_plus_absolute_gripper"
)
SOURCE_GROOT_POSITION = "groot_decoded_absolute_joint_position_plus_absolute_gripper"

_CANDIDATE_SOURCE_ACTION_SPACES = {
    ("pi05_droid", ACTION_SPACE_JOINT_VELOCITY): SOURCE_DROID_VELOCITY,
    ("pi05_droid", ACTION_SPACE_JOINT_POSITION): SOURCE_PI05_POSITION,
    ("groot_n17_droid", ACTION_SPACE_JOINT_POSITION): SOURCE_GROOT_POSITION,
}

# DROID's published control contract.
DROID_CONTROL_HZ = 15
DROID_OPEN_LOOP_HORIZON = 8
DROID_ACTION_WIDTH = 8
ARM_JOINT_COUNT = 7

# The ADP-009D environment's own timing, from the runtime configuration.
ISAAC_SIM_DT_SECONDS = 1.0 / 120.0
ISAAC_DECIMATION = 8
ISAAC_ACTION_DIM = 8

BLOCKER_CHUNK_SHAPE = "droid_action_chunk_shape_invalid"
BLOCKER_CHUNK_NONFINITE = "droid_action_chunk_nonfinite"
BLOCKER_HORIZON_UNAVAILABLE = "droid_action_chunk_shorter_than_open_loop_horizon"
BLOCKER_CONTROL_RATE_MISMATCH = "isaac_step_rate_does_not_match_droid_control_hz"
BLOCKER_GRIPPER_CONVENTION_UNMEASURED = "isaac_gripper_convention_unmeasured"
BLOCKER_JOINT_VELOCITY_BOUNDS = "candidate_action_joint_velocity_bounds_invalid"
BLOCKER_JOINT_POSITION_BOUNDS = "candidate_action_joint_position_bounds_invalid"
BLOCKER_GRIPPER_BOUNDS = "candidate_action_gripper_bounds_invalid"
# Declared-channel generalization (company-supplied policy contracts).  The
# per-channel envelope that the DROID gripper hardcodes above becomes data:
# each declared channel carries its own command interval, raw accepted
# envelope, and executed semantics, and the validator polices the raw envelope
# per column exactly the way it polices the gripper today.
BLOCKER_CHANNEL_BOUNDS = "candidate_action_channel_bounds_invalid"
BLOCKER_CHANNEL_CONTRACT_INVALID = "candidate_action_channel_contract_invalid"
BLOCKER_CHANNEL_WIDTH = "candidate_action_channel_contract_width_mismatch"

# The released pi05 DROID checkpoint emits normalized joint-velocity commands.
# The bridge maps each inclusive [-1, 1] value to at most this candidate-space
# delta. Values outside that raw interval are not evidence merely because the
# compatibility bridge could clip them.
DROID_NORMALIZED_JOINT_VELOCITY_BOUNDS = (-1.0, 1.0)
DROID_GRIPPER_BOUNDS = (0.0, 1.0)
# The raw response envelope for the gripper channel is wider than its command
# interval.  The released checkpoints regress this scalar and overshoot it
# slightly -- the 20260825T125800Z live pi05 run returned 1.0253 on all 15
# rows -- and the native DROID adapter in ``droid_policy_bridge`` clips the
# scalar to [0, 1] and binarizes at 0.5, so 1.0253 and 1.0 command the
# identical grasp.  Refusing the overshoot made this harness stricter than the
# runtime it mirrors: the same class of harness fault as treating velocity
# rows as positions above.  The envelope still fails closed on wrong-unit or
# wrong-channel decodes (radians, meters, logits), which land far outside a
# quarter of the interval.
DROID_GRIPPER_RAW_ACCEPTED_BOUNDS = (-0.25, 1.25)

# The exact GR00T-N1.7-DROID processor first clips normalized actions to
# [-1, 1], unnormalizes with the checkpoint's q01/q99 statistics, then converts
# its relative joint action to an absolute target by adding the observed state.
# That publisher-defined conversion can legitimately cross a robot's hard
# position interval when the observed joint is already near a stop.  The
# official DROID client passes the decoded target to RobotEnv, and Blueprint's
# frozen candidate contract requires the target Franka limits to be applied and
# every clip recorded by the adapter.
#
# These are the per-joint extrema over rows 0..7 (the executable open-loop
# prefix) and rows 0..39 (the retained full response) from the exact checkpoint
# ``statistics.json`` at revision 05e7cc97e40dbd33b0890c35cc0214fcb0547ab5
# (publisher git blob 03e76c7666bafe2e31fcc2320ee5ffcdddc6d675).  They
# preserve a finite, checkpoint-derived raw envelope: a wrong-unit target still
# fails closed instead of becoming evidence merely because the native command
# can saturate it.
GROOT_N17_EXECUTED_RELATIVE_JOINT_LOWER_RAD = (
    -0.2842104376852512,
    -0.40980345545336605,
    -0.28288332268595695,
    -0.45300650000572207,
    -0.44081393226981164,
    -0.39843987703323364,
    -0.49863362465053795,
)
GROOT_N17_EXECUTED_RELATIVE_JOINT_UPPER_RAD = (
    0.2789923369884495,
    0.5074403650313619,
    0.2830187319219114,
    0.46241843521595005,
    0.4362581911683084,
    0.49101936221122755,
    0.5042999897152186,
)
GROOT_N17_FULL_RELATIVE_JOINT_LOWER_RAD = (
    -0.6787592408061027,
    -0.9014112558960915,
    -0.6620200968161225,
    -1.0085058695077895,
    -1.1221638709306716,
    -0.918896906375885,
    -1.2273675373196602,
)
GROOT_N17_FULL_RELATIVE_JOINT_UPPER_RAD = (
    0.6822606457769874,
    1.24374953299761,
    0.6420625650882736,
    1.1286851704120637,
    1.102525160312653,
    1.0609106874465946,
    1.2582319760322576,
)
GROOT_N17_RAW_ENVELOPE_PROVENANCE = {
    "checkpoint_id": "nvidia/GR00T-N1.7-DROID",
    "checkpoint_revision": "05e7cc97e40dbd33b0890c35cc0214fcb0547ab5",
    "statistics_sha256": (
        "127832f7df25cda15da4ba6be81737f96b65673d0f892f9fc1bce1bc062fa858"
    ),
    "statistics_publisher_git_blob": "03e76c7666bafe2e31fcc2320ee5ffcdddc6d675",
    "normalization": "q01_q99_relative_action_clipped_to_normalized_minus1_plus1",
}


class DroidActionExecutionError(ValueError):
    """Fail-closed DROID action execution contract errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(e) for e in errors if str(e)}))
        super().__init__(";".join(self.errors))


def _source_action_space(*, action_space: str, candidate_id: str | None) -> str:
    """Name the source representation without changing legacy utility defaults.

    Before the pi0.5 joint-position runtime existed, the only joint-position
    caller was GR00T, so the public arithmetic helpers implicitly labeled every
    such row as GR00T-decoded.  Keep that default for compatibility-only callers
    that do not carry an episode candidate, while the episode path supplies its
    frozen candidate id and therefore receives exact provenance.
    """

    if candidate_id is None:
        if action_space == ACTION_SPACE_JOINT_VELOCITY:
            return SOURCE_DROID_VELOCITY
        if action_space == ACTION_SPACE_JOINT_POSITION:
            return SOURCE_GROOT_POSITION
        raise DroidActionExecutionError(
            [f"droid_action_space_unsupported:{action_space}"]
        )
    try:
        return _CANDIDATE_SOURCE_ACTION_SPACES[(str(candidate_id), action_space)]
    except KeyError as exc:
        raise DroidActionExecutionError(
            [
                "candidate_action_space_unsupported:"
                f"candidate_id={candidate_id}:action_space={action_space}"
            ]
        ) from exc


@dataclass(frozen=True)
class GripperConvention:
    """How Arena's eighth action dimension encodes open and closed.

    Both values must come from a probe that commanded each and observed the
    resulting finger joint travel.  There is no default: an inverted convention
    would turn every commanded grasp into a release, and the resulting eval
    would look like a policy failure rather than a harness bug.
    """

    closed_command: float
    open_command: float
    measured_by_probe: bool = False

    def command_for(self, droid_gripper: float) -> float:
        # DROID: scalar in [0, 1], above 0.5 means closed.
        return self.closed_command if float(droid_gripper) > 0.5 else self.open_command


def isaac_steps_per_droid_action(
    *,
    sim_dt_seconds: float = ISAAC_SIM_DT_SECONDS,
    decimation: int = ISAAC_DECIMATION,
    control_hz: int = DROID_CONTROL_HZ,
) -> int:
    """Environment steps per policy action, refusing any non-integer ratio.

    A fractional ratio would mean the policy's actions and the simulator's
    timeline drift apart, so it fails closed rather than rounding.
    """

    step_seconds = float(sim_dt_seconds) * int(decimation)
    action_seconds = 1.0 / float(control_hz)
    ratio = action_seconds / step_seconds
    nearest = round(ratio)
    if nearest < 1 or abs(ratio - nearest) > 1e-9:
        raise DroidActionExecutionError(
            [f"{BLOCKER_CONTROL_RATE_MISMATCH}:ratio={ratio!r}"]
        )
    return int(nearest)


def validate_action_chunk(chunk: Any, *, horizon: int = DROID_OPEN_LOOP_HORIZON) -> Any:
    """Validate a policy's action chunk before any of it reaches the simulator."""

    import numpy as np

    values = np.asarray(chunk, dtype=float)
    errors: list[str] = []
    if values.ndim != 2 or values.shape[1] != DROID_ACTION_WIDTH:
        raise DroidActionExecutionError(
            [f"{BLOCKER_CHUNK_SHAPE}:{tuple(values.shape)}"]
        )
    if not np.isfinite(values).all():
        errors.append(BLOCKER_CHUNK_NONFINITE)
    if values.shape[0] < int(horizon):
        errors.append(f"{BLOCKER_HORIZON_UNAVAILABLE}:{values.shape[0]}<{horizon}")
    if errors:
        raise DroidActionExecutionError(errors)
    return values


def _validate_declared_channel_bounds(
    chunk: Any,
    *,
    action_space: str,
    channel_contracts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate a chunk against *declared* per-channel envelope contracts.

    This is the generalization of the hardcoded DROID gripper envelope: a
    company-supplied policy declares, per channel, the command interval its
    runtime executes, the wider raw envelope its server may legitimately
    return, and the executed semantics explaining the gap.  Refusal applies
    the raw envelope; command-interval overshoot is *reported* per channel,
    never policed -- refusing it made this harness stricter than the runtime
    it mirrors (the 20260825T125800Z pi05 gripper lesson, now as data).

    The declared path deliberately does not touch the DROID-specific arm and
    gripper logic: the frozen-candidate contract stays code, the company
    contract stays data, and neither can silently borrow the other's bounds.
    """

    import numpy as np

    contracts = list(channel_contracts)
    if not contracts or any(
        not isinstance(contract, Mapping) for contract in contracts
    ):
        raise DroidActionExecutionError(
            [f"{BLOCKER_CHANNEL_CONTRACT_INVALID}:contracts_not_mappings"]
        )
    values = np.asarray(chunk, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1:
        raise DroidActionExecutionError(
            [f"{BLOCKER_CHUNK_SHAPE}:{tuple(values.shape)}"]
        )
    if values.shape[1] != len(contracts):
        # A declared contract for the wrong width would validate columns
        # against another channel's envelope -- silently, and plausibly.
        raise DroidActionExecutionError(
            [
                f"{BLOCKER_CHANNEL_WIDTH}:declared={len(contracts)}:"
                f"chunk={int(values.shape[1])}"
            ]
        )
    if not np.isfinite(values).all():
        raise DroidActionExecutionError([BLOCKER_CHUNK_NONFINITE])

    errors: list[str] = []
    applied: list[dict[str, Any]] = []
    for index, contract in enumerate(contracts):
        name = str(contract.get("name") or "")
        kind = str(contract.get("kind") or "")
        executed = str(contract.get("executed_semantics") or "")
        try:
            command_lower, command_upper = (
                float(bound) for bound in contract["command_interval"]
            )
            raw_lower, raw_upper = (
                float(bound) for bound in contract["raw_accepted_bounds"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DroidActionExecutionError(
                [f"{BLOCKER_CHANNEL_CONTRACT_INVALID}:{name or index}"]
            ) from exc
        # Trust nothing about the contract shape even though the admission
        # validator normally produced it: this function is also reachable with
        # hand-built dicts, and executing against a self-contradictory
        # envelope would be a silent harness fault.
        if (
            not name
            or not kind
            or not executed
            or not np.isfinite([command_lower, command_upper, raw_lower, raw_upper]).all()
            or not raw_lower <= command_lower < command_upper <= raw_upper
        ):
            raise DroidActionExecutionError(
                [f"{BLOCKER_CHANNEL_CONTRACT_INVALID}:{name or index}"]
            )
        column = values[:, index]
        outside_raw = np.argwhere((column < raw_lower) | (column > raw_upper))
        if outside_raw.size:
            row_index = int(outside_raw[0, 0])
            errors.append(
                f"{BLOCKER_CHANNEL_BOUNDS}:{name}:count={len(outside_raw)}:"
                f"first_row={row_index}:value={column[row_index]!r}:"
                f"bounds=[{raw_lower},{raw_upper}]"
            )
        outside_command = np.argwhere(
            (column < command_lower) | (column > command_upper)
        )
        overshoot = float(
            np.max(
                np.clip(
                    np.maximum(column - command_upper, command_lower - column),
                    0.0,
                    None,
                )
            )
        )
        applied.append(
            {
                "name": name,
                "kind": kind,
                "command_interval": [command_lower, command_upper],
                "raw_accepted_bounds": [raw_lower, raw_upper],
                "executed_semantics": executed,
                "rows_outside_command_interval": int(len(outside_command)),
                "max_command_interval_overshoot": overshoot,
            }
        )
    if errors:
        raise DroidActionExecutionError(errors)
    return {
        "action_space": action_space,
        "validated_rows": int(values.shape[0]),
        "channel_contracts_applied": applied,
        "raw_candidate_clipping_permitted": False,
    }


def validate_candidate_action_bounds(
    chunk: Any,
    *,
    action_space: str,
    joint_limits: Sequence[Sequence[float]] | None = None,
    channel_contracts: Sequence[Mapping[str, Any]] | None = None,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    """Strictly validate every supplied candidate row before adaptation.

    This pure validator does not decide which rows are executable. The episode
    path supplies the open-loop prefix for fail-closed execution validation and
    separately supplies the full response for a retained, nonexecuting tail
    audit. Any caller that supplies the full chunk still receives the original
    strict all-row behavior.

    With ``channel_contracts=None`` (the frozen ADP candidates) behavior is
    exactly the historical DROID contract below.  With declared per-channel
    contracts (company-supplied policies) validation routes through the
    generalized envelope path instead; the two never mix.
    """

    import numpy as np

    if channel_contracts is not None:
        return _validate_declared_channel_bounds(
            chunk, action_space=action_space, channel_contracts=channel_contracts
        )

    values = validate_action_chunk(chunk, horizon=1)
    errors: list[str] = []

    gripper_lower, gripper_upper = DROID_GRIPPER_BOUNDS
    raw_gripper_lower, raw_gripper_upper = DROID_GRIPPER_RAW_ACCEPTED_BOUNDS
    gripper_values = values[:, ARM_JOINT_COUNT]
    # Refusal applies the raw response envelope; the tighter command interval
    # is executed via the native clip-then-threshold semantics and is reported,
    # not policed.  See DROID_GRIPPER_RAW_ACCEPTED_BOUNDS.
    invalid_gripper = np.argwhere(
        (gripper_values < raw_gripper_lower) | (gripper_values > raw_gripper_upper)
    )
    if invalid_gripper.size:
        row_index = int(invalid_gripper[0, 0])
        errors.append(
            f"{BLOCKER_GRIPPER_BOUNDS}:count={len(invalid_gripper)}:"
            f"first_row={row_index}:value={float(gripper_values[row_index])!r}:"
            f"bounds=[{raw_gripper_lower},{raw_gripper_upper}]"
        )
    gripper_outside_command_interval = np.argwhere(
        (gripper_values < gripper_lower) | (gripper_values > gripper_upper)
    )
    gripper_command_interval_overshoot = float(
        np.max(
            np.clip(
                np.maximum(
                    gripper_values - gripper_upper, gripper_lower - gripper_values
                ),
                0.0,
                None,
            )
        )
    )

    if action_space == ACTION_SPACE_JOINT_VELOCITY:
        arm_lower, arm_upper = DROID_NORMALIZED_JOINT_VELOCITY_BOUNDS
        invalid_arm = np.argwhere(
            (values[:, :ARM_JOINT_COUNT] < arm_lower)
            | (values[:, :ARM_JOINT_COUNT] > arm_upper)
        )
        if invalid_arm.size:
            row_index, dimension_index = (
                int(invalid_arm[0, 0]),
                int(invalid_arm[0, 1]),
            )
            errors.append(
                f"{BLOCKER_JOINT_VELOCITY_BOUNDS}:count={len(invalid_arm)}:"
                f"first_row={row_index}:first_dimension={dimension_index}:"
                f"value={float(values[row_index, dimension_index])!r}:"
                f"bounds=[{arm_lower},{arm_upper}]"
            )
        arm_contract: dict[str, Any] = {
            "kind": "normalized_joint_velocity",
            "inclusive_bounds": [arm_lower, arm_upper],
            "maximum_mapped_joint_delta_rad": DROID_MAX_JOINT_DELTA_RAD,
        }
    elif action_space == ACTION_SPACE_JOINT_POSITION:
        limits = np.asarray(joint_limits, dtype=float)
        if (
            limits.shape != (ARM_JOINT_COUNT, 2)
            or not np.isfinite(limits).all()
            or np.any(limits[:, 0] > limits[:, 1])
        ):
            raise DroidActionExecutionError(["isaac_joint_limits_invalid"])
        command_lower = limits[:, 0]
        command_upper = limits[:, 1]
        if candidate_id == "groot_n17_droid":
            if values.shape[0] <= DROID_OPEN_LOOP_HORIZON:
                relative_lower = np.asarray(
                    GROOT_N17_EXECUTED_RELATIVE_JOINT_LOWER_RAD, dtype=float
                )
                relative_upper = np.asarray(
                    GROOT_N17_EXECUTED_RELATIVE_JOINT_UPPER_RAD, dtype=float
                )
                envelope_scope = "checkpoint_q01_q99_executable_rows_0_through_7"
            else:
                relative_lower = np.asarray(
                    GROOT_N17_FULL_RELATIVE_JOINT_LOWER_RAD, dtype=float
                )
                relative_upper = np.asarray(
                    GROOT_N17_FULL_RELATIVE_JOINT_UPPER_RAD, dtype=float
                )
                envelope_scope = "checkpoint_q01_q99_full_rows_0_through_39"
            raw_lower = command_lower + relative_lower
            raw_upper = command_upper + relative_upper
        else:
            raw_lower = command_lower
            raw_upper = command_upper
            envelope_scope = "native_joint_limits"
        arm_values = values[:, :ARM_JOINT_COUNT]
        invalid_arm = np.argwhere(
            (arm_values < raw_lower[None, :]) | (arm_values > raw_upper[None, :])
        )
        if invalid_arm.size:
            row_index, dimension_index = (
                int(invalid_arm[0, 0]),
                int(invalid_arm[0, 1]),
            )
            errors.append(
                f"{BLOCKER_JOINT_POSITION_BOUNDS}:count={len(invalid_arm)}:"
                f"first_row={row_index}:first_dimension={dimension_index}:"
                f"value={float(values[row_index, dimension_index])!r}:"
                f"bounds=[{float(raw_lower[dimension_index])!r},"
                f"{float(raw_upper[dimension_index])!r}]"
            )
        arm_outside_command_interval = np.argwhere(
            (arm_values < command_lower[None, :])
            | (arm_values > command_upper[None, :])
        )
        arm_command_interval_overshoot = float(
            np.max(
                np.clip(
                    np.maximum(
                        arm_values - command_upper[None, :],
                        command_lower[None, :] - arm_values,
                    ),
                    0.0,
                    None,
                )
            )
        )
        arm_contract = {
            "kind": "absolute_joint_position_rad",
            # Preserve the established receipt field while separately naming
            # the command interval versus GR00T's wider raw accepted envelope.
            "inclusive_bounds_by_joint": limits.tolist(),
            "command_interval_by_joint": limits.tolist(),
            "raw_accepted_bounds_by_joint": np.stack(
                (raw_lower, raw_upper), axis=1
            ).tolist(),
            "raw_envelope_scope": envelope_scope,
            "raw_envelope_provenance": (
                dict(GROOT_N17_RAW_ENVELOPE_PROVENANCE)
                if candidate_id == "groot_n17_droid"
                else None
            ),
            "executed_semantics": (
                "clip_to_native_joint_limits_and_record_each_saturation"
                if candidate_id == "groot_n17_droid"
                else "direct_within_native_joint_limits"
            ),
            "rows_outside_command_interval": int(
                len(arm_outside_command_interval)
            ),
            "max_command_interval_overshoot_rad": arm_command_interval_overshoot,
        }
    else:
        raise DroidActionExecutionError(
            [f"droid_action_space_unsupported:{action_space}"]
        )

    if errors:
        raise DroidActionExecutionError(errors)
    return {
        "action_space": action_space,
        "validated_rows": int(values.shape[0]),
        "arm_contract": arm_contract,
        "gripper_contract": {
            "kind": "absolute_gripper_scalar",
            "command_interval": [gripper_lower, gripper_upper],
            "raw_accepted_bounds": [raw_gripper_lower, raw_gripper_upper],
            "executed_semantics": "clip_to_command_interval_then_threshold_at_0.5",
            "native_reference": (
                "droid_policy_bridge.droid_action_to_mujoco_targets"
            ),
            "rows_outside_command_interval": int(
                len(gripper_outside_command_interval)
            ),
            "max_command_interval_overshoot": gripper_command_interval_overshoot,
        },
        "raw_candidate_clipping_permitted": candidate_id == "groot_n17_droid",
    }


def droid_row_to_isaac_action(
    row: Sequence[float],
    *,
    current_joint_position: Sequence[float],
    joint_limits: Sequence[Sequence[float]],
    gripper: GripperConvention,
    action_space: str = ACTION_SPACE_JOINT_VELOCITY,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    """Convert one candidate row into an Arena absolute-position target.

    Compatibility OpenPI configs may expose the DROID velocity action.  The
    frozen pi05 jointpos config returns absolute positions, and GR00T's
    processor decodes its configured relative representation back to raw
    *absolute* joint positions before returning from ``get_action``. Treating
    either absolute representation as velocity would be another silent
    action-space harness fault.
    """

    import numpy as np

    if not gripper.measured_by_probe:
        raise DroidActionExecutionError([BLOCKER_GRIPPER_CONVENTION_UNMEASURED])

    values = np.asarray(row, dtype=float)
    if values.shape != (DROID_ACTION_WIDTH,) or not np.isfinite(values).all():
        raise DroidActionExecutionError(
            [f"{BLOCKER_CHUNK_SHAPE}:{tuple(values.shape)}"]
        )
    limits = np.asarray(joint_limits, dtype=float)
    if limits.shape != (ARM_JOINT_COUNT, 2) or not np.isfinite(limits).all():
        raise DroidActionExecutionError(["isaac_joint_limits_invalid"])
    validate_candidate_action_bounds(
        values[None, :],
        action_space=action_space,
        joint_limits=limits,
        candidate_id=candidate_id,
    )

    if action_space == ACTION_SPACE_JOINT_VELOCITY:
        try:
            mapped = droid_action_to_mujoco_targets(
                values,
                current_joint_position=current_joint_position,
                joint_limits=limits,
            )
        except ValueError as exc:
            raise DroidActionExecutionError([f"droid_velocity_mapping_invalid:{exc}"]) from exc
        target = np.asarray(mapped["joint_position_target_rad"], dtype=float)
        clipped_source = list(mapped["clipped_action"])
        velocity_command = [float(v) for v in values[:ARM_JOINT_COUNT]]
        joint_limit_clamped = bool(mapped["joint_limit_clamped"])
        source_action_space = _source_action_space(
            action_space=action_space, candidate_id=candidate_id
        )
        position_adapter = "observed_joint_plus_validated_normalized_velocity_delta"
        adapter_max_delta: float | None = DROID_MAX_JOINT_DELTA_RAD
    elif action_space == ACTION_SPACE_JOINT_POSITION:
        current = np.asarray(current_joint_position, dtype=float)
        if current.shape != (ARM_JOINT_COUNT,) or not np.isfinite(current).all():
            raise DroidActionExecutionError(["isaac_current_joint_position_invalid"])
        raw_target = values[:ARM_JOINT_COUNT]
        if candidate_id == "groot_n17_droid":
            target = np.clip(raw_target, limits[:, 0], limits[:, 1])
            clipped_source = [*target.tolist(), float(values[ARM_JOINT_COUNT])]
            position_adapter = (
                "groot_decoded_absolute_joint_position_with_native_limit_saturation"
            )
        else:
            target = raw_target.copy()
            clipped_source = [float(value) for value in values]
            position_adapter = "decoded_absolute_joint_position_direct_within_limits"
        velocity_command = []
        joint_limit_clamped = bool(np.any(np.abs(target - raw_target) > 1e-12))
        source_action_space = _source_action_space(
            action_space=action_space, candidate_id=candidate_id
        )
        adapter_max_delta = None
    else:
        raise DroidActionExecutionError(
            [f"droid_action_space_unsupported:{action_space}"]
        )
    action = np.zeros(ISAAC_ACTION_DIM, dtype=float)
    action[:ARM_JOINT_COUNT] = target
    action[ARM_JOINT_COUNT] = gripper.command_for(values[ARM_JOINT_COUNT])
    return {
        "isaac_action": [float(v) for v in action],
        "joint_position_target_rad": [float(v) for v in target],
        "joint_velocity_command_rad_s": velocity_command,
        "source_arm_command": [float(v) for v in values[:ARM_JOINT_COUNT]],
        "clipped_droid_action": clipped_source,
        "joint_limit_clamped": joint_limit_clamped,
        "droid_gripper_scalar": float(values[ARM_JOINT_COUNT]),
        "gripper_closed": bool(float(values[ARM_JOINT_COUNT]) > 0.5),
        "source_action_space": source_action_space,
        "position_adapter": position_adapter,
        "position_adapter_max_joint_delta_rad": adapter_max_delta,
        "raw_candidate_bounds_validated": True,
    }


def plan_chunk_execution(
    chunk: Any,
    *,
    horizon: int = DROID_OPEN_LOOP_HORIZON,
    action_space: str = ACTION_SPACE_JOINT_VELOCITY,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    """Validate a chunk and retain the exact raw rows selected for execution.

    Only the first ``horizon`` rows are executed; DROID's open-loop horizon is
    shorter than the chunk a policy returns, and executing the tail would run
    the arm on predictions the policy expected to have superseded.
    """

    values = validate_action_chunk(chunk, horizon=horizon)
    if action_space == ACTION_SPACE_JOINT_VELOCITY:
        source_action_space = _source_action_space(
            action_space=action_space, candidate_id=candidate_id
        )
        position_adapter = "observed_joint_plus_validated_normalized_velocity_delta"
        adapter_max_delta: float | None = DROID_MAX_JOINT_DELTA_RAD
    elif action_space == ACTION_SPACE_JOINT_POSITION:
        source_action_space = _source_action_space(
            action_space=action_space, candidate_id=candidate_id
        )
        position_adapter = (
            "groot_decoded_absolute_joint_position_with_native_limit_saturation"
            if candidate_id == "groot_n17_droid"
            else "decoded_absolute_joint_position_direct_within_limits"
        )
        adapter_max_delta = None
    else:
        raise DroidActionExecutionError(
            [f"droid_action_space_unsupported:{action_space}"]
        )
    steps_per_action = isaac_steps_per_droid_action()
    rows = [
        {"droid_action": [float(value) for value in values[index]]}
        for index in range(int(horizon))
    ]
    returned_chunk = [
        [float(value) for value in values[index]]
        for index in range(int(values.shape[0]))
    ]
    return {
        "schema_version": ACTION_EXECUTION_SCHEMA_VERSION,
        "chunk_shape": [int(values.shape[0]), int(values.shape[1])],
        "executed_rows": int(horizon),
        "discarded_rows": int(values.shape[0]) - int(horizon),
        "isaac_steps_per_action": steps_per_action,
        "control_hz": DROID_CONTROL_HZ,
        "environment_step_seconds": ISAAC_SIM_DT_SECONDS * ISAAC_DECIMATION,
        "actions": rows,
        # Retain every row the model returned, including the deliberately
        # unexecuted tail. The open-loop horizon is an execution decision, not
        # permission to discard model output from the scientific receipt.
        "returned_chunk": returned_chunk,
        "source_action_space": source_action_space,
        "position_adapter": position_adapter,
        "position_adapter_max_joint_delta_rad": adapter_max_delta,
        "droid_source_revision": DROID_SOURCE_REVISION,
        "openpi_source_revision": OPENPI_SOURCE_REVISION,
        "candidate_policy_queried": True,
    }


def build_gripper_convention_probe_request() -> dict[str, Any]:
    """Describe the probe that must measure the gripper convention.

    Recorded rather than executed here: the measurement needs a simulator, and
    the point of this contract is that the convention is never assumed.
    """

    return {
        "schema_version": ACTION_EXECUTION_SCHEMA_VERSION,
        "purpose": "measure_isaac_eighth_action_dimension_gripper_convention",
        "method": (
            "command each candidate value on action dimension 7 with the arm "
            "held at the canonical pose, step until the finger joints settle, "
            "and record finger joint travel for each"
        ),
        "candidate_commands": [0.0, 1.0],
        "observed_joint_names": [
            "finger_joint",
            "left_inner_finger_joint",
            "right_inner_finger_joint",
        ],
        "decision_rule": (
            "closed_command is whichever value reduces finger separation; "
            "an ambiguous or zero-travel result fails closed rather than "
            "defaulting, because an inverted convention turns every commanded "
            "grasp into a release"
        ),
    }


__all__ = [
    "ACTION_EXECUTION_SCHEMA_VERSION",
    "ACTION_SPACE_JOINT_POSITION",
    "ACTION_SPACE_JOINT_VELOCITY",
    "ARM_JOINT_COUNT",
    "DROID_ACTION_WIDTH",
    "DROID_CONTROL_HZ",
    "DROID_OPEN_LOOP_HORIZON",
    "DroidActionExecutionError",
    "GripperConvention",
    "SOURCE_GROOT_POSITION",
    "SOURCE_PI05_POSITION",
    "build_gripper_convention_probe_request",
    "droid_row_to_isaac_action",
    "isaac_steps_per_droid_action",
    "plan_chunk_execution",
    "validate_action_chunk",
]
