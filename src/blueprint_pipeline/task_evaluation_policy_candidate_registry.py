"""Which policies a task evaluation run may compare, and why the rest may not.

Every run still compares exactly two frozen candidates, sealed before any
outcome exists. What this registry adds is the menu they are chosen from:
each policy we know about, pinned to an exact checkpoint, with its license
standing and its integration status stated rather than implied. A policy is
selectable only when all three hold:

- the canary runtime can actually serve it (a runtime binding exists and the
  scene setup lists it);
- its license allows a paid, hosted evaluation service;
- its checkpoint is pinned to bytes we verified, not only to a hub listing.

Everything else is published to the Website as ``unavailable`` with the exact
reason, so a robot team sees what is coming and why it is not offered yet,
and nothing becomes runnable by editing a label.

Backlog: ADP-050 (integrate candidates behind one normalized contract). The
per-run rule is unchanged; this widens only the set a pair is drawn from.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .adp009d_policy_candidate_admission import EXPECTED_CANDIDATES

REGISTRY_SCHEMA_VERSION = "task_evaluation_policy_candidate_registry.v1"

Status = Literal["runnable", "integration_pending", "license_review_pending"]
CommercialUse = Literal["allowed", "allowed_with_conditions", "under_review", "not_allowed"]
DigestKind = Literal["downloaded_snapshot_inventory", "hub_tree_listing"]

TWO_CAMERA_OBSERVATION = "droid_two_camera_robot_state_v1"
THREE_CAMERA_OBSERVATION = "droid_three_camera_robot_state_v1"
ABSOLUTE_JOINT_POSITION_ACTION = "droid_absolute_joint_position_v1"


class PolicyCandidateRegistryError(ValueError):
    """A requested candidate pair cannot be run."""

    def __init__(self, blockers: Sequence[str]):
        self.blockers = tuple(blockers)
        super().__init__(";".join(self.blockers))


@dataclass(frozen=True)
class PolicyCandidate:
    candidate_id: str
    display_name: str
    family: str
    checkpoint_repository: str
    checkpoint_revision: str
    checkpoint_digest: str
    checkpoint_digest_kind: DigestKind
    checkpoint_total_bytes: int
    license_id: str
    commercial_use: CommercialUse
    license_note: str
    status: Status
    adapter_id: str
    observation_schema_id: str
    action_schema_id: str
    minimum_gpu_memory_gb: int
    blockers: tuple[str, ...] = ()
    reason: str | None = None

    @property
    def runnable(self) -> bool:
        return self.status == "runnable"

    @property
    def checkpoint_uri(self) -> str:
        if self.checkpoint_repository.startswith("https://huggingface.co/"):
            return f"{self.checkpoint_repository}/tree/{self.checkpoint_revision}"
        return self.checkpoint_repository


def _pinned(candidate_id: str) -> dict[str, object]:
    pin = EXPECTED_CANDIDATES[candidate_id]
    return {
        "checkpoint_repository": str(pin["checkpoint_repository"]),
        "checkpoint_revision": str(pin["checkpoint_revision"]),
        "checkpoint_digest": str(pin["checkpoint_inventory_digest"]),
        "checkpoint_digest_kind": "downloaded_snapshot_inventory",
        "checkpoint_total_bytes": int(pin["checkpoint_total_bytes"]),
    }


# Hub pins for policies not yet served: the revision and a digest over the
# hub's file listing at that revision (path, size, LFS sha256 or git blob id).
# Admission replaces each with a digest over the downloaded bytes.
REGISTRY: tuple[PolicyCandidate, ...] = (
    PolicyCandidate(
        candidate_id="pi05_droid",
        display_name="π0.5 DROID",
        family="Physical Intelligence openpi",
        **_pinned("pi05_droid"),  # type: ignore[arg-type]
        license_id="apache-2.0-gemma-terms",
        commercial_use="allowed_with_conditions",
        license_note="Apache-2.0 code; PaliGemma-derived weights under the Gemma Terms of Use.",
        status="runnable",
        adapter_id="openpi_droid_to_official_arena_droid_abs_joint_v2",
        observation_schema_id=TWO_CAMERA_OBSERVATION,
        action_schema_id=ABSOLUTE_JOINT_POSITION_ACTION,
        minimum_gpu_memory_gb=12,
    ),
    PolicyCandidate(
        candidate_id="groot_n17_droid",
        display_name="GR00T N1.7 DROID",
        family="NVIDIA Isaac GR00T",
        **_pinned("groot_n17_droid"),  # type: ignore[arg-type]
        license_id="nvidia-open-model-license",
        commercial_use="allowed",
        license_note="NVIDIA Open Model License; Apache-2.0 code.",
        status="runnable",
        adapter_id="groot_n17_droid_to_official_arena_droid_abs_joint_v2",
        observation_schema_id=TWO_CAMERA_OBSERVATION,
        action_schema_id=ABSOLUTE_JOINT_POSITION_ACTION,
        minimum_gpu_memory_gb=24,
    ),
    PolicyCandidate(
        candidate_id="cosmos3_nano_policy_droid",
        display_name="Cosmos 3 Nano Policy DROID",
        family="NVIDIA Cosmos 3",
        checkpoint_repository="https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID",
        checkpoint_revision="805c0d6d46196ecdc789213a6de72303c383895b",
        checkpoint_digest="sha256:efcf09c97b0382aba67ccf4c3311671b999511fc545b8e553cce60eb4a8710d4",
        checkpoint_digest_kind="hub_tree_listing",
        checkpoint_total_bytes=32_937_432_088,
        license_id="openmdw-1.1",
        commercial_use="allowed",
        license_note="OpenMDW-1.1 per the model card: ready for commercial and non-commercial use.",
        status="integration_pending",
        adapter_id="cosmos3_nano_droid_openpi_websocket_v0",
        observation_schema_id=THREE_CAMERA_OBSERVATION,
        action_schema_id=ABSOLUTE_JOINT_POSITION_ACTION,
        minimum_gpu_memory_gb=40,
        blockers=(
            "canary_runtime_serving_path_missing",
            "second_exterior_camera_not_in_policy_views",
            "absolute_joint_position_plus_gripper_mapping_missing",
            "session_gpu_memory_below_co_resident_requirement",
            "checkpoint_bytes_not_downloaded_and_verified",
            "integration_canary_receipt_missing",
        ),
        reason=(
            "Coming soon. It is being connected to our simulator and checked on a "
            "reference task before it can run."
        ),
    ),
    PolicyCandidate(
        candidate_id="molmoact2_droid",
        display_name="MolmoAct 2 DROID",
        family="Ai2 MolmoAct",
        checkpoint_repository="https://huggingface.co/allenai/MolmoAct2-DROID",
        checkpoint_revision="d8c1abd8a27d8e859455bbe514df2bcc617db0fb",
        checkpoint_digest="sha256:c33efc98a197940e29aeaa8bc1caaa63d20bdc9e085a73e9c8368099c6bc8fb2",
        checkpoint_digest_kind="hub_tree_listing",
        checkpoint_total_bytes=21_781_603_004,
        license_id="apache-2.0-research-intent",
        commercial_use="under_review",
        license_note=(
            "Weights stated as Apache-2.0 but 'intended for research and educational use'; "
            "the Molmo2 base card notes non-commercial training data. Needs legal review."
        ),
        status="license_review_pending",
        adapter_id="molmoact2_droid_fastapi_v0",
        observation_schema_id=TWO_CAMERA_OBSERVATION,
        action_schema_id=ABSOLUTE_JOINT_POSITION_ACTION,
        minimum_gpu_memory_gb=16,
        blockers=(
            "license_review_pending",
            "canary_runtime_serving_path_missing",
            "checkpoint_bytes_not_downloaded_and_verified",
            "integration_canary_receipt_missing",
        ),
        reason="Not offered yet. Its license is being reviewed for use in a paid service.",
    ),
    PolicyCandidate(
        candidate_id="flux3_action_droid",
        display_name="FLUX 3 Action DROID",
        family="Black Forest Labs FLUX 3",
        checkpoint_repository="https://huggingface.co/black-forest-labs/flux-3-action-droid",
        checkpoint_revision="3d0887bdc7acee1686b19afac267125d519ff4f1",
        checkpoint_digest="sha256:855bea9ad3c809ac45e42b19b3acba5f826f2e3f56886495f56f9420d41450f1",
        checkpoint_digest_kind="hub_tree_listing",
        checkpoint_total_bytes=63_003_003_839,
        license_id="flux-kommunity-1.0",
        commercial_use="not_allowed",
        license_note=(
            "FLUX Kommunity License v1.0: non-commercial by default; outputs-only commercial "
            "use for qualifying users under US$5M revenue; no use to improve other models. "
            "Needs a written license or legal sign-off."
        ),
        status="license_review_pending",
        adapter_id="flux3_action_droid_openpi_websocket_v0",
        observation_schema_id=THREE_CAMERA_OBSERVATION,
        action_schema_id=ABSOLUTE_JOINT_POSITION_ACTION,
        minimum_gpu_memory_gb=32,
        blockers=(
            "license_review_pending",
            "canary_runtime_serving_path_missing",
            "second_exterior_camera_not_in_policy_views",
            "checkpoint_bytes_not_downloaded_and_verified",
            "integration_canary_receipt_missing",
        ),
        reason="Not offered yet. Its license limits commercial use and is being reviewed.",
    ),
)

_BY_ID = {candidate.candidate_id: candidate for candidate in REGISTRY}


def candidate(candidate_id: str) -> PolicyCandidate | None:
    return _BY_ID.get(candidate_id)


def runnable_candidate_ids() -> tuple[str, ...]:
    return tuple(item.candidate_id for item in REGISTRY if item.runnable)


def unavailable_candidates() -> tuple[PolicyCandidate, ...]:
    return tuple(item for item in REGISTRY if not item.runnable)


def validate_selected_pair(candidate_ids: Sequence[str]) -> tuple[str, str]:
    """Two distinct runnable candidates, in registry order.

    The pair is an unordered choice. Registry order is the one canonical order,
    so the same two policies always make the same run whichever was picked
    first.
    """

    ids = tuple(str(item or "") for item in candidate_ids)
    if len(ids) != 2 or ids[0] == ids[1]:
        raise PolicyCandidateRegistryError(["policy_candidate_pair_must_be_two_distinct"])
    blockers = []
    for candidate_id in ids:
        row = _BY_ID.get(candidate_id)
        if row is None:
            blockers.append(f"policy_candidate_unknown:{candidate_id}")
        elif not row.runnable:
            blockers.append(f"policy_candidate_not_runnable:{candidate_id}:{row.status}")
    if blockers:
        raise PolicyCandidateRegistryError(blockers)
    order = [row.candidate_id for row in REGISTRY]
    return tuple(sorted(ids, key=order.index))  # type: ignore[return-value]


def registry_violations(canary_runtime_candidate_ids: Sequence[str]) -> list[str]:
    """Why the registry would overstate what can run; empty when it does not.

    A runnable entry needs the canary runtime to serve it, a license that allows
    a paid hosted service, a checkpoint verified from downloaded bytes, and no
    open blocker. Anything not runnable must say why.
    """

    runtime = set(canary_runtime_candidate_ids)
    violations = []
    for item in REGISTRY:
        if item.runnable:
            if item.candidate_id not in runtime:
                violations.append(f"runnable_without_canary_runtime:{item.candidate_id}")
            if item.commercial_use not in {"allowed", "allowed_with_conditions"}:
                violations.append(f"runnable_without_commercial_license:{item.candidate_id}")
            if item.checkpoint_digest_kind != "downloaded_snapshot_inventory":
                violations.append(f"runnable_without_verified_checkpoint:{item.candidate_id}")
            if item.blockers:
                violations.append(f"runnable_with_open_blockers:{item.candidate_id}")
        else:
            if not item.reason or not item.blockers:
                violations.append(f"unavailable_without_reason:{item.candidate_id}")
            if item.commercial_use in {"under_review", "not_allowed"} and item.status != "license_review_pending":
                violations.append(f"license_blocked_but_not_marked:{item.candidate_id}")
    if len(runnable_candidate_ids()) < 2:
        violations.append("fewer_than_two_runnable_candidates")
    return violations


__all__ = [
    "REGISTRY",
    "REGISTRY_SCHEMA_VERSION",
    "PolicyCandidate",
    "PolicyCandidateRegistryError",
    "THREE_CAMERA_OBSERVATION",
    "TWO_CAMERA_OBSERVATION",
    "candidate",
    "registry_violations",
    "runnable_candidate_ids",
    "unavailable_candidates",
    "validate_selected_pair",
]
