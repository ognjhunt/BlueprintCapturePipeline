"""Robot bodies a policy could run on, and why most cannot run yet.

An embodiment profile is a complete robot setup: arm(s), tool, camera rig,
controller and simulator runtime. Its declared observation and action schemas
are what a policy must match. Only a profile whose asset, controller, camera rig
and reset are integrated and have executed an integration canary is
``runnable``. Others are listed so routing can say which body would unlock which
policy and what work remains. Listing a robot here makes no claim about its
capability or about who deploys it.

Adding a robot: declare its interface and ``missing`` work here. Promoting it
needs a verified asset, controller adapter, camera rig, reset pose and an
integration canary receipt; editing ``status`` alone must not make it run.

One narrow, explicit exception can be declared (``DevelopmentRunAllowance``):
a development-only run of one named task profile may proceed while named items
are still ``missing``, each stood in for by a named basis and the owner decision
behind it, and its result is recorded ``development_only``. It never makes a
body ``runnable``, never removes an item from ``missing`` and covers no other
task profile. The registry's output calls such an item ``STOOD_IN``, never
satisfied.

The G1 declares one, ``G1_NAVIGATION_DEVELOPMENT_RUN``: general, for every site
and scene, and for ``goal_region_navigation`` only
(docs/G1_NAVIGATION_FROM_COMPILED_SPEC.md). Owner decisions of 2026-11-07:

- ``reviewed_checkpoint_terms``: each navigation checkpoint's own human
  development rights review (``approved_for_development_simulation``,
  docs/arm_decision_proof_v1/manifests/g1_841757_<candidate>_development_rights_review.v1.json)
  stands in for it. Decision 2, verbatim: "yes - make sure this is the case for
  all future runs/scenes". A checkpoint without its own pinned review keeps the
  item missing. Every G1 run re-validates that review
  (``native_g1_beta_authority.validate_beta_authority``, embedded and
  digest-pinned in the execution spec, the session authority and the worker).
- ``verified_site_episode``: a development run stands in for it. Decision 1,
  verbatim: "Dev check, then counts". The run proceeds as a development-only
  check, and no result of it claims more than ``development_only``. Nothing
  here writes a verified site episode or removes the item from the G1's
  ``missing``: promoting the first run that passes every check to G1's first
  verified site episode is future work, and needs that run's receipt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Any
from typing import Literal

from .droid_policy_canary_embodiment import (
    DROID_ARENA_SOURCE_PATH,
    DROID_ARENA_SOURCE_REVISION,
    DROID_EMBODIMENT_ID,
)
from .task_evaluation_g1_catalog import G1_EMBODIMENT_ID
from .task_evaluation_policy_candidate_registry import (
    ABSOLUTE_JOINT_POSITION_ACTION,
    BIMANUAL_EEF_POSE_ACTION,
    DREAM1_NATIVE_JOINT_ACTION,
    HEAD_AND_WRIST_CAMERAS_OBSERVATION,
    TWO_CAMERA_OBSERVATION,
)
from .task_evaluation_rigid_relocation_native_adapter import NATIVE_PHYSICS_FREQUENCY_HZ

EmbodimentStatus = Literal["runnable", "not_integrated", "unavailable"]
Mobility = Literal["fixed", "mobile_base", "humanoid"]

FIXED_ARM_TASK_PROFILES = ("rigid_pick_place", "articulated_open_close")
_INTEGRATION_WORK = (
    "verified_robot_asset", "controller_adapter", "camera_rig", "reset_pose",
    "integration_canary_receipt",
)
_DEPLOYMENT_CATALOG_PATH = (
    Path(__file__).parent / "_catalog_data" / "deployment_embodiment_support.v1.json"
)


def deployment_embodiment_support_catalog() -> dict[str, Any]:
    """Read the planning catalog without adding runtime or admission claims.

    Return a fresh object so a response consumer cannot mutate the canonical
    metadata or change subsequent routing calls. Catalog validation and its
    response digest belong to the deployment-support boundary.
    """

    value = json.loads(_DEPLOYMENT_CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("deployment_embodiment_support_catalog_not_object")
    return value


#: The only task profiles a development-only run allowance may name: G1 navigation, by the owner's
#: decisions of 2026-11-07 (``G1_NAVIGATION_DEVELOPMENT_RUN``).
DEVELOPMENT_RUN_TASK_PROFILES = frozenset({"goal_region_navigation"})

#: What the registry's output says of an item an allowance stands in for. It is never "satisfied": the
#: body still misses the item, and a run that proceeds on it is recorded ``development_only``.
STOOD_IN = "stood in by the development-run allowance (development_only)"

#: The item a stand-in may cover only per checkpoint, from that checkpoint's own human review (rights).
CHECKPOINT_TERMS_ITEM = "reviewed_checkpoint_terms"


@dataclass(frozen=True)
class StandIn:
    """One still-``missing`` item a development-only run may proceed without, what stands in, and who decided.

    ``owner_decision`` quotes the decision verbatim, with its date and number. ``checkpoint_review``, when set,
    reads a candidate checkpoint's own human review (a mapping, or ``None`` when it has none that approves);
    the item is then stood in for only on a candidate whose review it returns.
    """

    item: str
    basis: str
    owner_decision: str
    checkpoint_review: Callable[[str], Mapping[str, Any] | None] | None = None


@dataclass(frozen=True)
class DevelopmentRunAllowance:
    """What a development-only run of one task profile may proceed without, and on what basis.

    Each ``stood_in_by`` entry names a still-``missing`` item and what stands in for it on such a run. The
    body stays as listed (its ``status`` and ``missing`` unchanged); only a run of ``task_profile_id`` whose
    result is recorded ``claim_ceiling`` may proceed without those items.
    """

    task_profile_id: str
    stood_in_by: tuple[StandIn, ...]
    claim_ceiling: str = "development_only"

    def for_candidate(self, candidate_id: str | None) -> DevelopmentRunAllowance | None:
        """The stand-ins that hold for one candidate checkpoint, or ``None`` when none does.

        A per-checkpoint stand-in holds only for a candidate whose own review its reader returns; without a
        candidate it holds for none.
        """
        held = tuple(row for row in self.stood_in_by if row.checkpoint_review is None
                     or (candidate_id is not None and row.checkpoint_review(candidate_id) is not None))
        return replace(self, stood_in_by=held) if held else None

    def record(self, candidate_id: str | None = None) -> dict[str, Any]:
        """The allowance as the registry's output names it: each item ``STOOD_IN``, with its basis and decision."""
        stood_in = {}
        for row in self.stood_in_by:
            entry = {"status": STOOD_IN, "basis": row.basis, "owner_decision": row.owner_decision}
            if row.checkpoint_review is not None and candidate_id is not None:
                entry["checkpoint_review"] = dict(row.checkpoint_review(candidate_id) or {})
            stood_in[row.item] = entry
        return {"task_profile_id": self.task_profile_id, "claim_ceiling": self.claim_ceiling, "stood_in": stood_in}


@dataclass(frozen=True)
class EmbodimentProfile:
    embodiment_id: str
    robot: str
    tool: str | None
    arm_count: int
    mobility: Mobility
    observation_schema_ids: tuple[str, ...]
    action_schema_ids: tuple[str, ...]
    controller: str | None
    simulator_runtime_id: str | None
    sensor_reference: str | None
    task_profile_ids: tuple[str, ...]
    status: EmbodimentStatus
    missing: tuple[str, ...] = ()
    reason: str | None = None
    #: The catalog interfaces' ``robot_profile_id`` values that run on this body.
    policy_robot_profile_ids: tuple[str, ...] = ()
    #: Narrow exceptions for a development-only run of one task profile (``DevelopmentRunAllowance``).
    development_run_allowances: tuple[DevelopmentRunAllowance, ...] = ()

    @property
    def available(self) -> bool:
        return self.status == "runnable"

    def development_run_missing(self, task_profile_id: str, *, candidate_id: str | None = None
                                ) -> tuple[tuple[str, ...], DevelopmentRunAllowance | None]:
        """What a development-only run of ``task_profile_id`` by ``candidate_id`` on this body still lacks.

        Returns those items and the allowance used, narrowed to the stand-ins that hold for that candidate
        (``DevelopmentRunAllowance.for_candidate``). Without an allowance for that profile, or with none of
        its stand-ins holding, it is everything ``missing``, as for any run.
        """
        allowance = next((row for row in self.development_run_allowances
                          if row.task_profile_id == task_profile_id and task_profile_id in self.task_profile_ids), None)
        allowance = allowance.for_candidate(candidate_id) if allowance is not None else None
        if allowance is None:
            return self.missing, None
        stood_in = {row.item for row in allowance.stood_in_by}
        return tuple(item for item in self.missing if item not in stood_in), allowance

    def meets(self, requirements: Mapping[str, Any]) -> bool:
        """Whether this body can do what a compiled task requires: chosen by need, never by task kind."""
        if requirements.get("locomote") and self.mobility == "fixed":
            return False
        return not (requirements.get("manipulate") and self.arm_count < 1)


def _arm(embodiment_id: str, robot: str, arm_count: int = 1, **interface) -> EmbodimentProfile:
    """A fixed arm we can name but have not integrated into the simulator."""

    return EmbodimentProfile(
        embodiment_id=embodiment_id, robot=robot, tool=interface.get("tool"), arm_count=arm_count,
        mobility="fixed",
        observation_schema_ids=interface.get("observation_schema_ids", ()),
        action_schema_ids=interface.get("action_schema_ids", ()),
        controller=None, simulator_runtime_id=None, sensor_reference=None,
        task_profile_ids=FIXED_ARM_TASK_PROFILES, status="not_integrated",
        missing=interface.get("missing", _INTEGRATION_WORK),
        reason=f"{robot} is not integrated: its simulator asset, controller adapter, camera rig and reset pose are not built yet.",
    )


def _catalog_arm(row: dict[str, Any]) -> EmbodimentProfile:
    """List an evidenced asset option while leaving every runtime gap explicit."""

    # Only the publisher's actual paired Dream demonstrations declare its
    # interface here. Neither a robot asset nor a customer-supplied route is an
    # inferred policy match. These remain non-runnable even when schemas match.
    dream_demonstrated = row["arm_count"] == 2 and any(
        route["route_id"] == "dreamlabs_dream1"
        and route["evidence"] == "publisher_demonstrated"
        for route in row["policy_routes"]
    )
    actions = (BIMANUAL_EEF_POSE_ACTION,) if dream_demonstrated else ()
    if row["embodiment_id"] == "i2rt_yam_bimanual":
        actions = (DREAM1_NATIVE_JOINT_ACTION, BIMANUAL_EEF_POSE_ACTION)
    missing = [*_INTEGRATION_WORK, *row["required_artifacts"]]
    if dream_demonstrated:
        missing.append("eef_pose_ik_adapter")
    return _arm(
        row["embodiment_id"], row["display_name"], arm_count=row["arm_count"],
        tool=row["tool_requirement"],
        observation_schema_ids=(HEAD_AND_WRIST_CAMERAS_OBSERVATION,) if dream_demonstrated else (),
        action_schema_ids=actions,
        missing=tuple(dict.fromkeys(missing)),
    )


#: A G1 navigation checkpoint's candidate id ends so: the rule planning (``beta_policy_screening_g1``) and
#: registration (``native_g1_site_registration``) use to run only these on a navigation scene.
G1_NAVIGATION_CHECKPOINT_SUFFIX = "_vision_navi"


def g1_navigation_checkpoint_development_rights_review(candidate_id: str) -> dict[str, Any] | None:
    """A G1 navigation checkpoint's own pinned human development rights review, when it approves development simulation.

    Read from docs/arm_decision_proof_v1/manifests/g1_841757_<candidate>_development_rights_review.v1.json
    through the reader every G1 run's authority uses (``native_g1_beta_authority.reviewed_model_terms``: the
    file's sha256 is pinned per candidate). ``None`` for a manipulation checkpoint (owner decision 2 names the
    navigation checkpoints' reviews only), a candidate with no pinned review, a changed file, or a review that
    does not approve this checkpoint's terms for development simulation.
    """
    if not candidate_id.endswith(G1_NAVIGATION_CHECKPOINT_SUFFIX):
        return None
    from .native_g1_beta_authority import REVIEW_DIGESTS, reviewed_model_terms

    try:
        terms = reviewed_model_terms(candidate_id)
    except ValueError:
        return None
    receipt = terms["receipt"]
    if (receipt.get("candidate_id") != candidate_id
            or receipt.get("status") != "approved_for_development_simulation"
            or receipt.get("checkpoint_terms_reviewed") is not True
            or receipt.get("rights_review_digest") != REVIEW_DIGESTS.get(candidate_id)):
        return None
    return {"artifact_sha256": terms["artifact_sha256"], "rights_review_digest": receipt["rights_review_digest"],
            "status": receipt["status"]}


#: G1 navigation on every site and scene, development-only (owner decisions 2026-11-07; module docstring).
#: Nothing names a site. It never writes a verified site episode: the G1 row's ``missing`` keeps both items.
G1_NAVIGATION_DEVELOPMENT_RUN = DevelopmentRunAllowance(
    task_profile_id="goal_region_navigation",
    stood_in_by=(
        StandIn(
            item="verified_site_episode",
            basis="development_only_run_proceeds_as_the_development_check",
            owner_decision='owner decision 2026-11-07, decision 1: "Dev check, then counts"',
        ),
        StandIn(
            item=CHECKPOINT_TERMS_ITEM,
            basis="each_navigation_checkpoints_own_human_development_rights_review_approved_for_development_simulation",
            owner_decision='owner decision 2026-11-07, decision 2: "yes - make sure this is the case for all '
                           'future runs/scenes"',
            checkpoint_review=g1_navigation_checkpoint_development_rights_review,
        ),
    ),
)


EMBODIMENT_REGISTRY: tuple[EmbodimentProfile, ...] = (
    EmbodimentProfile(
        embodiment_id=DROID_EMBODIMENT_ID,
        robot="Franka Emika Panda", tool="Robotiq 2F-85", arm_count=1, mobility="fixed",
        observation_schema_ids=(TWO_CAMERA_OBSERVATION,),
        action_schema_ids=(ABSOLUTE_JOINT_POSITION_ACTION,),
        controller=f"absolute joint position at 15 Hz over {NATIVE_PHYSICS_FREQUENCY_HZ} Hz physics",
        simulator_runtime_id="isaac_native_arena_v1",
        sensor_reference=f"{DROID_ARENA_SOURCE_PATH}@{DROID_ARENA_SOURCE_REVISION}",
        task_profile_ids=FIXED_ARM_TASK_PROFILES, status="runnable",
        policy_robot_profile_ids=("droid_franka_panda_robotiq_2f85_v1",),
    ),
    EmbodimentProfile(
        embodiment_id=G1_EMBODIMENT_ID,
        robot="Unitree G1", tool="Dex3 hand", arm_count=2, mobility="humanoid",
        observation_schema_ids=("humanoidarena_head_rgb_state64_v1",),
        action_schema_ids=("humanoidarena_semantic_v3",),
        controller="SONIC whole-body controller", simulator_runtime_id="isaac_native_arena_g1_v1",
        sensor_reference="configs/g1_humanoidarena_checkpoint_inventory.v1.json",
        task_profile_ids=("rigid_pick_place", "goal_region_navigation"), status="unavailable",
        missing=("verified_site_episode", "reviewed_checkpoint_terms"),
        reason="No production policy-canary profile has executed a verified G1 episode on a site scene, and its checkpoint terms are pending review.",
        policy_robot_profile_ids=("unitree_g1_dex3_sonic_v1",),
        development_run_allowances=(G1_NAVIGATION_DEVELOPMENT_RUN,),
    ),
    EmbodimentProfile(
        embodiment_id="boston_dynamics_atlas",
        robot="Boston Dynamics Atlas", tool=None, arm_count=2, mobility="humanoid",
        observation_schema_ids=(), action_schema_ids=(), controller=None,
        simulator_runtime_id=None, sensor_reference=None, task_profile_ids=(), status="unavailable",
        missing=("accessible_runtime", "simulator_embodiment", "policy_checkpoint_or_api", "delivery_channel"),
        reason="No Atlas runtime, simulator embodiment, policy checkpoint or delivery channel is accessible to Blueprint.",
    ),
    *(_catalog_arm(row) for row in deployment_embodiment_support_catalog()["embodiments"]),
    # Compatibility rows remain distinct from the exact variants above.
    _arm("universal_robots_ur10e", "Universal Robots UR10e"),
    _arm("fanuc_crx_10ia", "FANUC CRX-10iA"),
    _arm("abb_gofa_crb15000", "ABB GoFa CRB 15000"),
    _arm("kinova_gen3", "Kinova Gen3"),
    _arm("ufactory_xarm7", "UFACTORY xArm 7"),
)

_BY_ID = {row.embodiment_id: row for row in EMBODIMENT_REGISTRY}


def embodiment(embodiment_id: str) -> EmbodimentProfile | None:
    return _BY_ID.get(embodiment_id)


def embodiment_registry_violations() -> list[str]:
    """Why the registry would overstate which robots can run; empty when it does not."""

    violations = []
    if len({row.embodiment_id for row in EMBODIMENT_REGISTRY}) != len(EMBODIMENT_REGISTRY):
        violations.append("embodiment_id_duplicated")
    for row in EMBODIMENT_REGISTRY:
        if row.available:
            if row.missing or not (row.observation_schema_ids and row.action_schema_ids
                                   and row.controller and row.simulator_runtime_id):
                violations.append(f"runnable_embodiment_incomplete:{row.embodiment_id}")
        elif not row.reason or not row.missing:
            violations.append(f"unavailable_embodiment_without_reason:{row.embodiment_id}")
        for allowance in row.development_run_allowances:
            # Narrow and explicit: one listed profile, development-only, only items the body still misses, each
            # with a basis and its owner decision; checkpoint terms only per checkpoint, from its own review.
            items = [stand.item for stand in allowance.stood_in_by]
            if (row.available or allowance.task_profile_id not in row.task_profile_ids
                    or allowance.task_profile_id not in DEVELOPMENT_RUN_TASK_PROFILES
                    or allowance.claim_ceiling != "development_only" or not allowance.stood_in_by
                    or len(set(items)) != len(items)
                    or any(stand.item not in row.missing or not stand.basis or not stand.owner_decision
                           or (stand.item == CHECKPOINT_TERMS_ITEM and stand.checkpoint_review is None)
                           for stand in allowance.stood_in_by)):
                violations.append(f"development_run_allowance_invalid:{row.embodiment_id}")
    return violations


__all__ = [
    "CHECKPOINT_TERMS_ITEM", "DevelopmentRunAllowance", "EMBODIMENT_REGISTRY", "EmbodimentProfile",
    "G1_NAVIGATION_CHECKPOINT_SUFFIX", "G1_NAVIGATION_DEVELOPMENT_RUN", "STOOD_IN", "StandIn",
    "deployment_embodiment_support_catalog",
    "embodiment", "embodiment_registry_violations", "g1_navigation_checkpoint_development_rights_review",
]
