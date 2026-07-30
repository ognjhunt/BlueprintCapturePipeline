"""Frozen GPU profiles for specialized public WAM replay arms."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .ctrl_world_provider_bundle import (
    CTRL_WORLD_CHECKPOINT_REPOSITORY,
    CTRL_WORLD_CHECKPOINT_REVISION,
    CTRL_WORLD_PUBLIC_IMAGE,
    CTRL_WORLD_SOURCE_REVISION,
)
from .oscar_official_release import (
    OFFICIAL_OSCAR_HF_REPO,
    OFFICIAL_OSCAR_HF_REVISION,
    OFFICIAL_OSCAR_SOURCE_COMMIT,
    OFFICIAL_OSCAR_WAM_IMAGE_REF,
)


def build_oscar_public_replay_profile(profile_type: Callable[..., Any]) -> Any:
    """Build the frozen exact-public OSCAR replay profile."""

    return profile_type(
        experiment_id="policy_ranking_cosmos3_edge_closed_loop_20260729",
        admission_schema="policy_ranking_oscar_public_replay_gpu_admission.v2",
        authorization_schema="policy_ranking_cosmos3_edge_closed_loop_compute_authorization.v1",
        preflight_schema="policy_ranking_oscar_public_replay_vast_preflight.v2",
        receipt_schema="oscar_public_replay_bundle_receipt.v2",
        authorization_ids_by_allocation_index={
            7: "policy-ranking-cosmos3-edge-closed-loop-20260729-allocation-7",
        },
        cost_authorization_binding_sha256=(
            "4b7e126dd677b3a79317a7d51738428951efc1019189a766251e1ed39bb98400"
        ),
        expected_bundle_sha256=("d6447c8432eb9d484c64f61244eaec40739f9115d778390f6b1fef18c9564752"),
        expected_bundle_size_bytes=1_135_718,
        expected_embedded_input_hashes={
            "runtime_manifest_file_sha256": (
                "2cc85cf88be3318fc1692fd2872aeb802ee3de24f993f80ced0d2b5b348205f2"
            ),
            "rollout_manifest_file_sha256": (
                "2b50a0abf072ba691e68f612ab95d8d5e3856688c90be0e81741785d4929d823"
            ),
            "first_frame_sha256": (
                "2efae31ef115800a18f04302b668241a5891d36f3ad29734b22a46c791307a35"
            ),
            "skeleton_video_sha256": (
                "4ad9dd1c6cf2acd2bd62fd51a4e5c0744ee0a077b8ea06b7a0a35af24418d08c"
            ),
            "runner_sha256": ("4b1367be37fc49b319b602c5cedbd40c66c5fe5d16afcf10697cccc8f0fa60e4"),
            "entrypoint_sha256": (
                "194f4809624cc82e1f3a88f64e12304451708986dad5880695af0518a0efb19c"
            ),
        },
        qualification_canary_request_count=1,
        scientific_matrix_request_count=0,
        total_initial_generation_request_count=1,
        request_budget_amendment_sha256=None,
        max_compute_cap_usd=5.0,
        max_hourly_rate_usd=2.05,
        target_spend_usd=5.0,
        hard_ttl_seconds=7_200,
        oscar_replay_bundle=True,
        provider_bundle_kind="wam",
        bundle_schema="wam_provider_runtime_manifest.v1",
        checkpoint_repository=OFFICIAL_OSCAR_HF_REPO,
        checkpoint_revision=OFFICIAL_OSCAR_HF_REVISION,
        public_image=OFFICIAL_OSCAR_WAM_IMAGE_REF,
        min_gpu_ram_mb=80_000,
        cosmos_revision=None,
        cosmos_framework_revision=OFFICIAL_OSCAR_SOURCE_COMMIT,
        vllm_omni_revision=None,
        allowed_providers=("vast",),
        compatible_gpu_keywords=("RTX PRO 6000", "H100"),
    )


def build_ctrl_world_replay_profile(profile_type: Callable[..., Any]) -> Any:
    """Build the frozen exact-public Ctrl-World replay profile."""

    return profile_type(
        experiment_id="policy_ranking_cosmos3_edge_closed_loop_20260729",
        admission_schema="policy_ranking_ctrl_world_replay_gpu_admission.v1",
        authorization_schema="policy_ranking_cosmos3_edge_closed_loop_compute_authorization.v1",
        preflight_schema="policy_ranking_ctrl_world_replay_vast_preflight.v1",
        receipt_schema="ctrl_world_replay_bundle_receipt.v1",
        authorization_ids_by_allocation_index={
            index: f"policy-ranking-cosmos3-edge-closed-loop-20260729-allocation-{index}"
            for index in range(8, 13)
        },
        cost_authorization_binding_sha256=(
            "4b7e126dd677b3a79317a7d51738428951efc1019189a766251e1ed39bb98400"
        ),
        expected_bundle_sha256=("9d1133f481c5ba75386d8dfa35b1a70d4eca8f382242422cf263c9ed5df36070"),
        expected_bundle_size_bytes=2_579_230,
        expected_embedded_input_hashes={
            "runtime_manifest_file_sha256": (
                "9efb0badcc347b76b445a352623d125dbbd5bc74d8dbb1e644a097840adbbe50"
            ),
            "rollout_manifest_file_sha256": (
                "ed5084a914688b047a4ab798b12cd4f8b9fd49833759f9f504c564daa3c36f22"
            ),
            "canary_manifest_sha256": (
                "418ab0b7e6a69ec13010f32569dac469cb4b68159f72214f8bc550bcffd643ff"
            ),
            "annotation_sha256": (
                "cc7bf532144fa40492eae9e44e919290ea45fc57283304f0c5d58689785e700c"
            ),
            "view_manifest_sha256": (
                "2de9a23f04338bf391c056c835215ed3cdacb0450aa84b0079705dc074dcc8e9"
            ),
            "source_manifest_sha256": (
                "8379bb479232e2a5638c9b209f7f7436e5366137609afcf4f6462e177b531255"
            ),
            "runner_sha256": ("f79b63c23dd5e0ae78e8962ffa764b5aaa5d55fe02e37e564d7397a20c532655"),
            "entrypoint_sha256": (
                "5e4c8dbf17c23af4367d96fa8edc2d032c65130e338b9119122722a112e2620f"
            ),
        },
        qualification_canary_request_count=1,
        scientific_matrix_request_count=0,
        total_initial_generation_request_count=1,
        request_budget_amendment_sha256=None,
        max_compute_cap_usd=5.0,
        max_hourly_rate_usd=2.05,
        target_spend_usd=3.0,
        hard_ttl_seconds=4_800,
        ctrl_world_replay_bundle=True,
        provider_bundle_kind="wam",
        bundle_schema="wam_provider_runtime_manifest.v1",
        checkpoint_repository=CTRL_WORLD_CHECKPOINT_REPOSITORY,
        checkpoint_revision=CTRL_WORLD_CHECKPOINT_REVISION,
        public_image=CTRL_WORLD_PUBLIC_IMAGE,
        min_gpu_ram_mb=80_000,
        cosmos_revision=None,
        cosmos_framework_revision=CTRL_WORLD_SOURCE_REVISION,
        vllm_omni_revision=None,
        allowed_providers=("vast",),
        compatible_gpu_keywords=("RTX PRO 6000", "H100"),
    )


__all__ = ["build_ctrl_world_replay_profile", "build_oscar_public_replay_profile"]
