"""Turn contact sensors and poses into the booleans the scorer demands.

The task-neutral scorer requires six booleans on every sample. Five of them are
statements about the physical world - is the gripper touching the handle, did
the robot hit something, did the appliance stay where it was put - and none can
be inferred from joint angles. Until this existed the articulated scoring path
had never been given a real sample: every control run would have been refused,
whatever else worked.

Each predicate is built from one named source and refuses when that source is
absent. The refusal matters more than the reading: a contact predicate that
quietly returns False when its sensor is missing does not report "we could not
look", it asserts "the robot did not collide". That is a safety claim, and this
program must never make one by omission.

Thresholds are in newtons on measured contact force. A sensor reports small
non-zero forces constantly from resting contact and solver noise, so a bare
"any force" test is always true and would fail every episode.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence


ARTICULATED_SCENE_OBSERVATIONS_SCHEMA_VERSION = "articulated_scene_observations.v1"
# Above resting-contact and solver noise, below anything a 7-axis arm does on
# purpose. A gripper closing on a handle registers tens of newtons.
DEFAULT_CONTACT_FORCE_THRESHOLD_N = 1.0
# A cabinet that has moved this far has been dragged, not opened.
DEFAULT_BASE_DISPLACEMENT_TOLERANCE_M = 0.02
# Far enough from the handle that the door is unambiguously unheld.
DEFAULT_RETREAT_DISTANCE_M = 0.25


class ArticulatedSceneObservationError(ValueError):
    """Stable, sorted observation-source failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _require(value: Any, *, error: str) -> Any:
    if value is None:
        raise ArticulatedSceneObservationError([error])
    return value


def max_contact_force_n(
    forces: Any, *, body_indices: Sequence[int] | None = None
) -> float:
    """Largest contact-force magnitude over the selected bodies.

    Accepts the (envs, bodies, 3) tensor a ContactSensor reports. Reduced here
    rather than in the caller so every predicate applies the same reduction and
    a threshold means the same thing in each.
    """

    if forces is None:
        raise ArticulatedSceneObservationError(
            ["articulated_scene_observation_contact_forces_missing"]
        )
    # A ContactSensor reports (envs, bodies, 3); tests and some call sites pass
    # a plain nested list, which has no .shape. Unwrap by inspecting depth so
    # both behave identically - a shape-only check silently mis-indexes lists
    # and reduces over the wrong axis.
    def _depth(value: Any) -> int:
        shape = getattr(value, "shape", None)
        if shape is not None:
            return len(shape)
        count = 0
        probe = value
        while isinstance(probe, (list, tuple)) and probe:
            count += 1
            probe = probe[0]
        return count

    rows = forces[0] if _depth(forces) == 3 else forces
    selected = range(len(rows)) if body_indices is None else body_indices
    largest = 0.0
    for index in selected:
        if index < 0 or index >= len(rows):
            raise ArticulatedSceneObservationError(
                [f"articulated_scene_observation_body_index_out_of_range:{index}"]
            )
        vector = rows[index]
        magnitude = float(sum(float(component) ** 2 for component in vector) ** 0.5)
        largest = max(largest, magnitude)
    return largest


def resolve_contact_sensor_rows(
    *,
    sensor_body_names: Sequence[str],
    finger_body_names: Sequence[str],
) -> dict[str, Any]:
    """Row indices into a ContactSensor's arrays, from its own body list.

    A ContactSensor matches bodies by prim-path regex and publishes the subset
    it found as ``body_names``. Those rows are not the articulation's body
    indices, and rt29 spent a launch discovering that: index 9 and index 14,
    both valid on the robot, both out of range on the sensor.

    Two index spaces that happen to both be integers is a bug waiting to be
    written, so the only supported way to get rows is from the array's own
    names.
    """

    names = [str(value) for value in sensor_body_names]
    if not names:
        raise ArticulatedSceneObservationError(
            ["articulated_scene_observation_sensor_body_names_empty"]
        )
    fingers = {str(value) for value in finger_body_names}
    finger_rows = [index for index, name in enumerate(names) if name in fingers]
    if not finger_rows:
        raise ArticulatedSceneObservationError(
            [
                "articulated_scene_observation_finger_rows_absent:"
                + ",".join(sorted(fingers))
                + ":observed=" + ",".join(names)
            ]
        )
    return {
        "finger_rows": finger_rows,
        "non_finger_rows": [
            index for index in range(len(names)) if index not in set(finger_rows)
        ],
        "sensor_body_names": names,
    }


def build_scene_observations(
    *,
    read_task_contact_forces: Callable[[], Any],
    read_robot_contact_forces: Callable[[], Any],
    read_scene_contact_forces: Callable[[], Any],
    read_task_object_base_position_m: Callable[[], Sequence[float]],
    authored_task_object_base_position_m: Sequence[float],
    read_end_effector_position_m: Callable[[], Sequence[float]],
    read_handle_position_m: Callable[[], Sequence[float]],
    finger_body_indices: Sequence[int] | None = None,
    non_finger_body_indices: Sequence[int] | None = None,
    read_pinch_position_m: Callable[[], Sequence[float]] | None = None,
    pinch_on_task_corridor: Callable[[Sequence[float]], bool] | None = None,
    read_gripper_net_forces: Callable[[], Any] | None = None,
    contact_force_threshold_n: float = DEFAULT_CONTACT_FORCE_THRESHOLD_N,
    base_displacement_tolerance_m: float = DEFAULT_BASE_DISPLACEMENT_TOLERANCE_M,
    retreat_distance_m: float = DEFAULT_RETREAT_DISTANCE_M,
) -> dict[str, Callable[[], bool]]:
    """The five observed predicates, each bound to one named source."""

    threshold = float(contact_force_threshold_n)
    authored = [float(value) for value in authored_task_object_base_position_m]
    if len(authored) != 3:
        raise ArticulatedSceneObservationError(
            ["articulated_scene_observation_authored_base_position_invalid"]
        )

    def _distance(left: Sequence[float], right: Sequence[float]) -> float:
        pair = list(left), list(right)
        if len(pair[0]) < 3 or len(pair[1]) < 3:
            raise ArticulatedSceneObservationError(
                ["articulated_scene_observation_position_invalid"]
            )
        return float(
            sum((float(pair[0][i]) - float(pair[1][i])) ** 2 for i in range(3)) ** 0.5
        )

    def _finger_net_force() -> float:
        # The gripper subtree's own sensor, its own row space. rt60 sampled
        # the ARM sensor with finger-sensor indices - the rt29 crossover
        # resurrected - and read 0.00N through an entire 50-degree pull.
        if read_gripper_net_forces is None:
            return 0.0
        forces = _require(
            read_gripper_net_forces(),
            error="articulated_scene_observation_gripper_contact_unavailable",
        )
        return max_contact_force_n(forces)

    def _task_contact_attribution() -> str:
        """Which evidence path says the fingers are on the task, if any.

        rt56 drove the door to 50.6 degrees with the filtered finger-vs-twin
        matrix reading zero on all 123 samples. If that matrix is dead, the
        honest substitute is force on the finger bodies WHILE the pinch sits
        on the handle arc - geometry the plan already guarantees. First-class
        evidence wins when present; the fallback names itself; neither path
        blesses a shove landed away from the handle.
        """

        filtered = _require(
            read_task_contact_forces(),
            error="articulated_scene_observation_task_contact_unavailable",
        )
        if max_contact_force_n(filtered, body_indices=finger_body_indices) > threshold:
            return "filtered_matrix"
        if read_pinch_position_m is not None and pinch_on_task_corridor is not None:
            if _finger_net_force() > threshold and bool(
                pinch_on_task_corridor(read_pinch_position_m())
            ):
                return "finger_net_force_on_task_corridor"
        return "none"

    def task_contact_active() -> bool:
        return _task_contact_attribution() != "none"

    def robot_collision_failure() -> bool:
        # Deliberately excludes the fingers: a gripper touching the handle is
        # the task succeeding, not the robot colliding.
        forces = _require(
            read_robot_contact_forces(),
            error="articulated_scene_observation_robot_contact_unavailable",
        )
        return (
            max_contact_force_n(forces, body_indices=non_finger_body_indices)
            > threshold
        )

    def scene_collision_failure() -> bool:
        forces = _require(
            read_scene_contact_forces(),
            error="articulated_scene_observation_scene_contact_unavailable",
        )
        scene_force = max_contact_force_n(forces)
        if _task_contact_attribution() == "finger_net_force_on_task_corridor":
            # The residual channel cannot subtract what the dead filtered
            # matrix never reported. Forces attributed to the task by the
            # fallback are removed here instead - up to the finger-net
            # magnitude, never below zero, so a genuine simultaneous scene
            # strike still shows through.
            scene_force = max(0.0, scene_force - _finger_net_force())
        return scene_force > threshold

    def containment_violation() -> bool:
        # The appliance is meant to stay where it was placed. If the base has
        # moved, the arm dragged the whole unit instead of swinging the door,
        # and the joint angle alone would still read like success.
        observed = _require(
            read_task_object_base_position_m(),
            error="articulated_scene_observation_base_position_unavailable",
        )
        return _distance(observed, authored) > float(base_displacement_tolerance_m)

    def retreat_completed() -> bool:
        end_effector = _require(
            read_end_effector_position_m(),
            error="articulated_scene_observation_end_effector_unavailable",
        )
        handle = _require(
            read_handle_position_m(),
            error="articulated_scene_observation_handle_position_unavailable",
        )
        return _distance(end_effector, handle) > float(retreat_distance_m)

    def contact_diagnostics() -> dict[str, float]:
        # The raw magnitudes behind the three contact booleans. rt56 scored
        # 43 scene-collision samples and zero task contacts while the door
        # tracked the plan to 50.6 degrees; booleans alone cannot say whether
        # the filtered matrix read zero or a threshold was wrong. Bounded on
        # purpose: three floats per sample, not per-body arrays.
        return {
            "task_contact_attribution": _task_contact_attribution(),
            "finger_filtered_force_n": max_contact_force_n(
                read_task_contact_forces()
            ),
            "robot_net_force_n": max_contact_force_n(
                read_robot_contact_forces()
            ),
            "residual_scene_force_n": max_contact_force_n(
                read_scene_contact_forces()
            ),
        }

    return {
        "read_task_contact_active": task_contact_active,
        "read_robot_collision_failure": robot_collision_failure,
        "read_scene_collision_failure": scene_collision_failure,
        "read_containment_violation": containment_violation,
        "read_retreat_completed": retreat_completed,
        "read_contact_diagnostics": contact_diagnostics,
    }


__all__ = [
    "ARTICULATED_SCENE_OBSERVATIONS_SCHEMA_VERSION",
    "DEFAULT_BASE_DISPLACEMENT_TOLERANCE_M",
    "DEFAULT_CONTACT_FORCE_THRESHOLD_N",
    "DEFAULT_RETREAT_DISTANCE_M",
    "ArticulatedSceneObservationError",
    "build_scene_observations",
    "max_contact_force_n",
]
