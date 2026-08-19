"""One definition per runtime tunable, or a merged fix ships nowhere.

A tunable a runtime reads is only as good as its single definition. When the
same number is also written down somewhere else -- as a dict literal, as a
function default, or as a same-name constant in a second module -- raising it in
one place leaves every other copy behind, and nothing fails. The run looks
entirely normal while executing the old value.

That is not hypothetical here. PR #786 raised the Franka servo bounds from
0.03/0.20 to 0.10/1.00, merged, and deployed. r17, r19 and r20 then reported
*exactly* 10.717 rad of summed arm travel, because the packet restated the pair
as literals (#789) and the construction worker omitted them at the call site and
inherited the servo's own defaults (#793). Three paid runs replayed identical
commands while a merged, deployed fix sat inert.

#789 fixed the packet builder and #793 fixed the call site and moved the shared
bounds into ``native_task_construction_plan``. This module pins what remains of
that sweep: the copies neither PR removed, plus the divergences that are
deliberate and must not be "tidied" into agreement later.
"""

from __future__ import annotations

import inspect
import pathlib
import re

import pytest

from blueprint_pipeline.native_articulated_control_plan import (
    GRIPPER_DWELL_MAXIMUM_STEPS,
    GRIPPER_DWELL_MINIMUM_STEPS,
    MOTION_MAXIMUM_STEPS,
)
from blueprint_pipeline.native_task_construction_plan import (
    MAX_JOINT_DELTA_RAD,
    MAX_JOINT_SETPOINT_LEAD_RAD,
)


SOURCE_ROOT = (
    pathlib.Path(__file__).resolve().parent.parent / "src" / "blueprint_pipeline"
)

#: Every module allowed to bind each shared tunable name, and why.
#:
#: Three generations of articulated control plan coexist in this tree and they
#: reuse each other's constant NAMES with different numbers. That is the same
#: hazard as a bare literal, only better disguised: a same-name binding reads as
#: single-sourced, so an importer reaching for the wrong module gets a stale
#: value with no signal.
#:
#:   native_task_construction_plan      the shared Panda command bounds (#793)
#:   native_articulated_control_plan    v1 controls (legacy adapter)
#:   paired_target_native_arena_request v2 packet (paired-target, graph)
#:   adp009d_control_episode            ADP-009D DROID episode runtime
#:
#: Entries are allowlisted rather than unified where the numbers are genuinely
#: different -- different generation, budget, or robot -- and unifying them
#: would break the owner. The allowlist is EXACT, so a new copy still fails.
TUNABLE_DEFINING_MODULES = {
    # The Panda command bounds. #793 made native_task_construction_plan their
    # single home; both control-plan compilers import from there, so
    # construction and controls cannot sit on different bounds.
    "MAX_JOINT_DELTA_RAD": {
        "native_task_construction_plan.py",
        # DROID @ 15 Hz, not a Panda @ 20 Hz -- a different robot's limit, in
        # the frozen policy-ranking program.
        "policy_ranking_droid_kinematics.py",
    },
    "MAX_JOINT_SETPOINT_LEAD_RAD": {"native_task_construction_plan.py"},
    # Per-phase step budgets. Each generation sizes these to its OWN action
    # budget and the sizes are structurally incompatible -- see
    # test_step_budgets_differ_by_control_plan_generation_on_purpose.
    "MOTION_MINIMUM_STEPS": {
        "native_articulated_control_plan.py",
        "paired_target_native_arena_request.py",
    },
    "MOTION_MAXIMUM_STEPS": {
        "native_articulated_control_plan.py",
        "paired_target_native_arena_request.py",
    },
    "GRIPPER_DWELL_MINIMUM_STEPS": {
        "native_articulated_control_plan.py",
        "paired_target_native_arena_request.py",
        # ADP-009D DROID episode: 30/120 at a different control rate.
        "adp009d_control_episode.py",
    },
    "GRIPPER_DWELL_MAXIMUM_STEPS": {
        "native_articulated_control_plan.py",
        "paired_target_native_arena_request.py",
        "adp009d_control_episode.py",
    },
    # Tolerances that AGREE across generations today. Listed anyway: agreement
    # is not single-sourcing. The point is that a value which happens to match
    # is still owned by exactly one module per generation, so a future tune of
    # one cannot silently retune the other.
    "ARRIVAL_TOLERANCE_M": {
        "native_articulated_control_plan.py",
        "paired_target_native_arena_request.py",
    },
    "ARRIVAL_STABILITY_STEPS": {
        "native_articulated_control_plan.py",
        "paired_target_native_arena_request.py",
    },
    "ZERO_ACTION_STEPS": {
        "native_articulated_control_plan.py",
        "adp009d_control_episode.py",
    },
}


def _module_sources() -> list[tuple[str, str]]:
    return [
        (path.name, path.read_text(encoding="utf-8"))
        for path in sorted(SOURCE_ROOT.glob("*.py"))
    ]


@pytest.mark.parametrize("constant", sorted(TUNABLE_DEFINING_MODULES))
def test_shared_tunable_names_are_bound_only_where_allowlisted(
    constant: str,
) -> None:
    """A same-name shadow is worse than a literal: it reads as single-sourced.

    ``adp009d_control_episode`` bound MAX_JOINT_SETPOINT_LEAD_RAD to 0.20,
    exported it, and read it nowhere -- a dead binding of a live name, which
    would have handed the pre-#786 throttle to anyone who imported it from
    there. It is deleted, not allowlisted.

    Growing this allowlist is not the fix for a new failure here. Ask first
    whether the new binding is a genuinely different number -- different
    generation, budget, or robot -- or just another copy.
    """

    definition = re.compile(rf"^{constant}\s*=", re.MULTILINE)
    defining = {
        name for name, source in _module_sources() if definition.search(source)
    }

    assert defining == TUNABLE_DEFINING_MODULES[constant], (
        f"{constant} must be defined only in "
        f"{sorted(TUNABLE_DEFINING_MODULES[constant])}, found in "
        f"{sorted(defining)}. Import it instead of redefining it -- a second "
        "binding of the same name silently wins for whoever imports it."
    )


def test_no_module_restates_a_command_bound_as_a_literal() -> None:
    """The #789 grep, widened from one module to the whole package.

    A literal is invisible to the constant it duplicates: raising the constant
    leaves it untouched and nothing fails.
    """

    offenders: list[str] = []
    for name, source in _module_sources():
        for field in ("max_joint_delta_rad", "max_joint_setpoint_lead_rad"):
            for pattern in (
                # "max_joint_delta_rad": 0.03      -- a packet/plan dict literal
                rf'"{field}":\s*[0-9]',
                # max_joint_delta_rad: float = 0.03 -- an invisible default
                rf"{field}\s*:\s*float\s*=\s*[0-9]",
            ):
                if re.search(pattern, source):
                    offenders.append(f"{name}: {field}")

    assert offenders == [], (
        "command bounds restated as literals: "
        + ", ".join(sorted(offenders))
        + ". Import MAX_JOINT_DELTA_RAD / MAX_JOINT_SETPOINT_LEAD_RAD from "
        "native_task_construction_plan so raising them ships everywhere."
    )


def test_the_pose_servo_refuses_to_supply_its_own_bounds() -> None:
    """No default, because a default is invisible at the call site.

    #793 fixed the construction worker's omitted keywords. This closes the door
    rather than the instance: with no default there is nothing to inherit, so
    the next caller cannot reintroduce the same silence. The original defect was
    invisible to every behavioural test precisely because the omitted values
    equalled these defaults.
    """

    from blueprint_pipeline.native_franka_pose_servo import (
        NativeFrankaDifferentialIkServo,
    )

    signature = inspect.signature(
        NativeFrankaDifferentialIkServo.action_for_grasp_target
    )
    for field in ("max_joint_delta_rad", "max_joint_setpoint_lead_rad"):
        parameter = signature.parameters[field]
        assert parameter.default is inspect.Parameter.empty, (
            f"{field} must stay required on the shared servo. A default lets a "
            "caller drive the arm under bounds it never named, which is how a "
            "raised constant stayed inert across three paid runs."
        )


def test_step_budgets_differ_by_control_plan_generation_on_purpose() -> None:
    """DELIBERATE divergence -- do not "fix" this by importing the constants.

    ``native_articulated_control_plan`` is the v1 compatibility adapter:
    ``materialize_native_task_control_plan`` routes to it only when the task
    spec is NOT ``adp_task_spec.v2``. Everything
    ``paired_target_native_arena_request`` emits is v2, so the live lane runs
    ``materialize_native_graph_articulated_control_plan``, which reads its step
    budgets out of the packet's interaction affordance and never consults the
    v1 constants at all.

    They are also structurally incompatible. v1's fixed 12 phases cost
    10*35 + 2*20 = 390 steps, +40 settle = 430 against its 450 budget; the same
    phases at v2's 64/5/12 cost 664, +40 = 704, which the v1 compiler refuses
    outright. Unifying them breaks whichever generation loses.

    And importing MOTION_MAXIMUM_STEPS into the request would cut the live
    motion budget 64 -> 35 while the arm in this lane already fails by running
    OUT of steps -- a silent regression dressed as a cleanup.

    The command bounds are the exception that proves the rule: those describe
    the Panda hardware, identical across generations, which is why they are
    shared (#789, #793) and these are not.
    """

    from blueprint_pipeline import paired_target_native_arena_request as v2

    plan_source = (SOURCE_ROOT / "native_task_control_plan.py").read_text(
        encoding="utf-8"
    )
    request_source = (
        SOURCE_ROOT / "paired_target_native_arena_request.py"
    ).read_text(encoding="utf-8")

    # Each generation names its own budgets; neither imports the other's.
    assert (MOTION_MAXIMUM_STEPS, v2.MOTION_MAXIMUM_STEPS) == (35, 64)
    assert (GRIPPER_DWELL_MINIMUM_STEPS, v2.GRIPPER_DWELL_MINIMUM_STEPS) == (8, 5)
    assert (GRIPPER_DWELL_MAXIMUM_STEPS, v2.GRIPPER_DWELL_MAXIMUM_STEPS) == (20, 12)

    # The three that agree today are still separately owned.
    assert v2.MOTION_MINIMUM_STEPS == 1
    assert v2.ARRIVAL_TOLERANCE_M == 0.02
    assert v2.ARRIVAL_STABILITY_STEPS == 2

    # ...and none of the six may go back to being a bare literal in the packet.
    for field in (
        "motion_minimum_steps",
        "motion_maximum_steps",
        "gripper_dwell_minimum_steps",
        "gripper_dwell_maximum_steps",
        "arrival_tolerance_m",
        "arrival_stability_steps",
    ):
        assert (
            re.search(rf'"{field}":\s*[0-9]', request_source) is None
        ), f"{field} is restated as a literal in the v2 packet builder"

    # Nothing may be imported from the v1 adapter into the v2 packet builder.
    assert (
        re.search(
            r"from \.native_articulated_control_plan import", request_source
        )
        is None
    ), (
        "the v2 packet builder must not import the v1 adapter's constants. "
        "Importing its step budgets would cut the live motion budget from 64 "
        "to 35 in a lane that already fails by running OUT of steps."
    )

    # ...and the live v2 branch reads the packet, so the v1 constants cannot
    # reach it. If this stops being true, the divergence stops being safe.
    for field in (
        "motion_minimum_steps",
        "motion_maximum_steps",
        "gripper_dwell_minimum_steps",
        "gripper_dwell_maximum_steps",
    ):
        assert f'"{field}"' in plan_source, (
            f"the v2 graph branch must keep reading {field} from the packet; "
            "if it starts using the v1 constant instead, the live motion "
            "budget silently drops from 64 to 35"
        )
    assert (
        'if (scene_plan.get("task_spec") or {}).get("schema_version") '
        '== "adp_task_spec.v2":' in plan_source
    ), (
        "the v1/v2 split is what makes the step-budget divergence safe; if the "
        "dispatcher stops routing on schema_version, re-derive which budget wins"
    )


def test_the_command_bounds_are_still_the_raised_physical_values() -> None:
    """0.10 rad/step at 20 Hz is 2 rad/s, inside a Panda's ~2.6 rad/s limit."""

    assert MAX_JOINT_DELTA_RAD == 0.10
    assert MAX_JOINT_SETPOINT_LEAD_RAD == 1.00


def test_the_arrival_orientation_tolerance_is_written_once() -> None:
    """It was the same literal twice, ~100 lines apart, once per task kind.

    Both copies reach the runtime, so tuning one and missing the other would
    qualify the gate at one tolerance and execute at another, with nothing in
    either module to say so.
    """

    from blueprint_pipeline.paired_target_native_arena_request import (
        ARRIVAL_ORIENTATION_TOLERANCE_RAD,
    )

    source = (
        SOURCE_ROOT / "paired_target_native_arena_request.py"
    ).read_text(encoding="utf-8")

    assert ARRIVAL_ORIENTATION_TOLERANCE_RAD == 0.08
    assert (
        re.search(r'"arrival_orientation_tolerance_rad":\s*[0-9]', source) is None
    ), (
        "arrival_orientation_tolerance_rad is restated as a literal; both task "
        "kinds emit it, so a second copy can drift from the first unnoticed"
    )
    assert (
        source.count(
            '"arrival_orientation_tolerance_rad": '
            "ARRIVAL_ORIENTATION_TOLERANCE_RAD"
        )
        == 2
    ), "both the articulated and the rigid affordance must emit the constant"
