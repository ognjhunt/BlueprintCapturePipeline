"""Which paid lanes rent a GPU with no process-independent hard-TTL backstop.

The in-process spend cap lives in the provider adapter. If that process dies
-- a deploy, an OOM, a unit restart, all documented conditions on the control
plane host -- nothing in this repository stops the meter; the instance bills
to the provider's own limits. The independent watchdog is the only backstop
that survives its parent, which is why every live profile publishes
`watchdog_required: True`.

Nothing verifies that claim. `watchdog_required` is asserted in many places as
a profile *field* and in none as a lane *behaviour*, so a lane can publish it
and arm nothing. This test does not fix that -- rewiring a paid lane's arming
and teardown is a change per lane, and a mis-wired watchdog reaps a healthy
run, which is worse than the gap. It pins the set instead, so the gap is
visible, bounded, and cannot grow without a test failing.
"""

from __future__ import annotations

import ast
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "blueprint_pipeline"

#: Lanes whose watchdog is armed by the shared allocator rather than the lane
#: module, so absence of the call inside the module is not a gap.
ARMED_BY_THE_ALLOCATOR = frozenset({"semantic_teacher_image_edit_vast"})

#: Provider-launching lanes with no independent watchdog anywhere today. Each
#: rents a GPU whose only spend ceiling dies with the adapter process. Listed
#: rather than summarised so that closing one is a visible, reviewable edit.
LANES_WITHOUT_AN_INDEPENDENT_WATCHDOG = frozenset(
    {
        "adp_aura_author_smoke_vast",
        "adp_inpaint360_interiorgs_vast",
        "openvla_policy_provider_smoke",
        "public_scene_simready_isaac_vast",
        "robot_eval_provider_launcher",
        "simpler_public_vast",
        "unitree_groot_n17_sonic_vast_image_canary",
        "unitree_groot_n17_sonic_vast_persistent_session",
        "unitree_groot_n17_sonic_vast_policy_command",
        "vast_authorized_probe_runner",
        "vast_provider_adapter",
    }
)


def _names_used(module: Path) -> set[str]:
    tree = ast.parse(module.read_text(encoding="utf-8"))
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.alias):
            used.add((node.asname or node.name).rsplit(".", 1)[-1])
    return used


def _provider_launching_lanes() -> dict[str, set[str]]:
    lanes: dict[str, set[str]] = {}
    for module in sorted(_PACKAGE.glob("*.py")):
        used = _names_used(module)
        if {"run_vast_provider_adapter", "launch"} & used:
            lanes[module.stem] = used
    return lanes


def test_the_set_of_lanes_without_an_independent_watchdog_has_not_grown() -> None:
    unwatched = {
        stem
        for stem, used in _provider_launching_lanes().items()
        if "run_vast_provider_adapter" in used
        and "arm_independent_vast_watchdog" not in used
        and stem not in ARMED_BY_THE_ALLOCATOR
    }

    assert unwatched == set(LANES_WITHOUT_AN_INDEPENDENT_WATCHDOG), {
        "newly_unwatched": sorted(unwatched - LANES_WITHOUT_AN_INDEPENDENT_WATCHDOG),
        "now_watched": sorted(LANES_WITHOUT_AN_INDEPENDENT_WATCHDOG - unwatched),
    }


def test_every_lane_that_arms_one_binds_it_to_the_names_it_creates() -> None:
    """An armed watchdog that watches another lane's prefix reaps nothing.

    The joint-agent lane armed on the GR00T name family while labelling its own
    instances differently, so its name-scoped sweep matched an empty set and
    still reported provider zero. Any lane that both arms and labels must pass
    the same value to each.
    """

    offenders: list[str] = []
    for module in sorted(_PACKAGE.glob("*.py")):
        text = module.read_text(encoding="utf-8")
        if "arm_independent_vast_watchdog" not in text:
            continue
        if "instance_label_prefix=" not in text:
            continue
        tree = ast.parse(text)
        armed: set[str] = set()
        labelled: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.keyword) or node.arg is None:
                continue
            rendered = ast.unparse(node.value)
            if node.arg == "pod_name_prefix":
                armed.add(rendered)
            elif node.arg == "instance_label_prefix":
                labelled.add(rendered)
        # A label derived from the handle is bound by construction, however it
        # is spelled, so only a label that shares no expression with the armed
        # prefix and does not read it off the handle counts as drifted.
        if any("pod_name_prefix" in item for item in labelled):
            continue
        if armed and labelled and not (armed & labelled):
            offenders.append(f"{module.stem}: armed={sorted(armed)} labelled={sorted(labelled)}")

    #: Known drifted lanes, listed so the set cannot grow silently.
    #: `adp_aura_interiorgs_vast` is a retired appearance method with no launch
    #: profile, so its drift cannot reach a provider; it is recorded rather
    #: than repaired.
    known = {"adp_aura_interiorgs_vast"}

    assert {item.split(":", 1)[0] for item in offenders} == known, offenders
