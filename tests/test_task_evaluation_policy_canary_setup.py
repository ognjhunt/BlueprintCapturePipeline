from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_policy_canary_setup import (
    TaskEvaluationPolicyCanarySetupError,
    validate_policy_canary_setup,
)
from scripts.attach_internal_policy_canary_setup import (
    attach_internal_policy_canary_setup,
)


def _setup() -> dict[str, object]:
    families = [
        "canonical_anchor",
        "canonical_anchor",
        "placement_approach",
        "placement_approach",
        "illumination",
        "camera_sensor",
        "bounded_physics",
        "admitted_object_material_cousin",
        "pairwise_stress",
        "held_out_composition",
    ]
    cells = []
    for index, family in enumerate(families):
        scenario = {"family": family, "ordinal": index}
        cells.append(
            {
                "cell_id": f"quick-{index}",
                "family": family,
                "seed": 1000 + index,
                "partition": (
                    "canonical"
                    if family == "canonical_anchor"
                    else "held_out"
                    if family == "held_out_composition"
                    else "stress"
                ),
                "label": f"{family} {index}",
                "cell_digest": canonical_digest(scenario),
            }
        )
    candidates = []
    for index, candidate_id in enumerate(("pi05_droid", "groot_n17_droid")):
        candidates.append(
            {
                "candidate_id": candidate_id,
                "display_name": candidate_id,
                "checkpoint": {
                    "uri": f"s3://checkpoints/{candidate_id}",
                    "digest": f"sha256:{index + 1:064x}",
                },
                "adapter_id": f"{candidate_id}_adapter",
                "license_id": "admitted_license",
                "compatibility": {
                    "robot_preset_ids": ["franka_panda_robotiq_2f85_v1"],
                    "embodiment_ids": ["franka_robotiq"],
                    "observation_schema_ids": ["droid_observation_v1"],
                    "action_schema_ids": ["droid_joint_position_v1"],
                    "simulator_runtime_ids": ["isaac_native_arena_v1"],
                    "task_family_ids": ["rigid_relocation"],
                },
                "readiness": {
                    "status": "verified_runnable",
                    "receipt": {
                        "uri": f"s3://readiness/{candidate_id}.json",
                        "digest": f"sha256:{index + 5:064x}",
                    },
                    "reason": None,
                },
            }
        )
    value: dict[str, object] = {
        "schema_version": "task_evaluation_policy_canary_setup.v1",
        "source_launch_id": "scene-839873-policy-canary",
        "offering_digest": "sha256:" + "a" * 64,
        "scene_revision_digest": "sha256:" + "b" * 64,
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "registry_digest": "sha256:" + "c" * 64,
        "robot_presets": [
            {
                "robot_preset_id": "franka_panda_robotiq_2f85_v1",
                "display_name": "Franka Panda + Robotiq 2F-85",
                "embodiment_id": "franka_robotiq",
                "task_family_id": "rigid_relocation",
                "simulator_runtime_id": "isaac_native_arena_v1",
                "runtime_image": {
                    "uri": "registry.example/isaac@sha256:" + "3" * 64,
                    "digest": "sha256:" + "3" * 64,
                },
                "observation_schema": {
                    "schema_id": "droid_observation_v1",
                    "cameras": ["external", "wrist"],
                    "modalities": ["rgb", "joint_position"],
                },
                "action_schema": {
                    "schema_id": "droid_joint_position_v1",
                    "space": "joint_position",
                    "control_hz": 15,
                },
                "readiness": {
                    "status": "verified_runnable",
                    "receipt": {
                        "uri": "s3://readiness/franka.json",
                        "digest": "sha256:" + "8" * 64,
                    },
                    "reason": None,
                },
                "policy_candidates": candidates,
            }
        ],
        "episode_presets": [
            {
                "preset_id": "quick_10",
                "label": "Quick",
                "episodes_per_policy": 10,
                "availability": "enabled",
                "recommended": True,
                "matrix": {
                    "matrix_digest": canonical_digest({"ordered_cells": cells}),
                    "resolver_id": "quick_10_deterministic",
                    "resolver_version": "v1",
                    "deterministic": True,
                    "cells": cells,
                    "expected_family_counts": {
                        "canonical_anchor": 2,
                        "placement_approach": 2,
                        "illumination": 1,
                        "camera_sensor": 1,
                        "bounded_physics": 1,
                        "admitted_object_material_cousin": 1,
                        "pairwise_stress": 1,
                        "held_out_composition": 1,
                    },
                    "coverage_gaps": [],
                },
                "estimate": {
                    "duration_minutes": {"minimum": 20, "maximum": 60},
                    "maximum_authorized_cost_usd": 5,
                    "hard_ttl_seconds": 9000,
                    "basis_digest": "sha256:" + "9" * 64,
                    "as_of": "2026-08-31T12:00:00Z",
                },
            },
            {
                "preset_id": "standard_100",
                "label": "Standard",
                "episodes_per_policy": 100,
                "availability": "coming_later",
                "recommended": False,
                "matrix": {"matrix_digest": "sha256:" + "d" * 64, "resolver_id": "standard", "resolver_version": "v1", "deterministic": True, "cells": [], "expected_family_counts": {"canonical_anchor": 0, "placement_approach": 0, "illumination": 0, "camera_sensor": 0, "bounded_physics": 0, "admitted_object_material_cousin": 0, "pairwise_stress": 0, "held_out_composition": 0}, "coverage_gaps": []},
                "estimate": {"duration_minutes": {"minimum": 0, "maximum": 0}, "maximum_authorized_cost_usd": 5, "hard_ttl_seconds": 9000, "basis_digest": "sha256:" + "d" * 64, "as_of": "2026-08-31T12:00:00Z"},
            },
            {
                "preset_id": "deep_500",
                "label": "Deep",
                "episodes_per_policy": 500,
                "availability": "coming_later",
                "recommended": False,
                "matrix": {"matrix_digest": "sha256:" + "e" * 64, "resolver_id": "deep", "resolver_version": "v1", "deterministic": True, "cells": [], "expected_family_counts": {"canonical_anchor": 0, "placement_approach": 0, "illumination": 0, "camera_sensor": 0, "bounded_physics": 0, "admitted_object_material_cousin": 0, "pairwise_stress": 0, "held_out_composition": 0}, "coverage_gaps": []},
                "estimate": {"duration_minutes": {"minimum": 0, "maximum": 0}, "maximum_authorized_cost_usd": 5, "hard_ttl_seconds": 9000, "basis_digest": "sha256:" + "e" * 64, "as_of": "2026-08-31T12:00:00Z"},
            },
        ],
        "diagnostics": {
            "zero_action": "nonblocking",
            "deterministic_scripted_positive": "nonblocking",
        },
        "setup_digest": "",
    }
    value["setup_digest"] = canonical_digest(value, digest_field="setup_digest")
    return value


def test_setup_and_profile_attachment_are_digest_bound() -> None:
    setup = _setup()
    assert validate_policy_canary_setup(setup) == setup
    profile = {
        "profile_id": setup["source_launch_id"],
        "profile_digest": "sha256:" + "f" * 64,
    }
    attached = attach_internal_policy_canary_setup(
        profile=profile,
        setup=setup,
        profile_validator=lambda value: [],
    )
    assert attached["internal_policy_canary_setup"] == setup
    assert attached["profile_digest"] == canonical_digest(
        attached, digest_field="profile_digest"
    )


def test_setup_rejects_unrunnable_second_policy() -> None:
    setup = _setup()
    mutated = copy.deepcopy(setup)
    readiness = mutated["robot_presets"][0]["policy_candidates"][1]["readiness"]
    readiness["status"] = "unavailable"
    readiness["receipt"] = None
    readiness["reason"] = "checkpoint_not_installed"
    mutated["setup_digest"] = canonical_digest(mutated, digest_field="setup_digest")
    with pytest.raises(
        TaskEvaluationPolicyCanarySetupError,
        match="policy_canary_setup_runnable_pair_invalid",
    ):
        validate_policy_canary_setup(mutated)
