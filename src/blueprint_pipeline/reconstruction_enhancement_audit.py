"""Deterministic qualification posture for generated reconstruction enhancers."""

from __future__ import annotations

from typing import Any

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "reconstruction_enhancement_method_audit.v1"

_METHODS: dict[str, dict[str, Any]] = {
    "artifixer": {
        "display_name": "NVIDIA ArtiFixer",
        "source_repository": "https://github.com/nv-tlabs/artifixer",
        "source_commit": "a392c4dfe17459ef9952407accdb9fcdcdddba98",
        "source_tree": "f9283bfe5e3a6cc160fd418f4e66412746a19a07",
        "source_license": "Apache-2.0",
        "source_license_sha256": (
            "sha256:b3b341839dbbbbbe32a8664d9aac72a78270a8986d40f7065082b37bbce4b301"
        ),
        "model_repository": "https://huggingface.co/nvidia/ArtiFixer",
        "model_revision": "f96352ad72c84a628d5844b6543e94ae8c4479b3",
        "model_license": "NVIDIA License; research_and_development_only",
        "model_license_url": (
            "https://developer.download.nvidia.com/licenses/"
            "NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf"
        ),
        "model_license_sha256": (
            "sha256:4ff203c3f7997c7fed287a463d733f794934a79cfabb2936008fca0bcc8ad3d6"
        ),
        "release_checkpoints": {
            "artifixer_1_3b_v1": {
                "filename": "artifixer-1.3b.pt",
                "size_bytes": 6_715_346_651,
                "sha256": (
                    "sha256:23e909fb4232c6a74a1c59eaf0ebfd419dd188e601aa0ab0145b9aaea821e059"
                ),
                "base_model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "base_model_revision": "0fad780a534b6463e45facd96134c9f345acfa5b",
                "base_model_license": "Apache-2.0",
            },
            "artifixer_14b_v1": {
                "filename": "artifixer-14b.pt",
                "size_bytes": 67_644_337_412,
                "sha256": (
                    "sha256:c1a6d31fb849211d4c682a28b40980549cd8f807ee309e7bc0141a336ffcd16b"
                ),
                "base_model_id": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                "base_model_revision": "38ec498cb3208fb688890f8cc7e94ede2cbd7f68",
                "base_model_license": "Apache-2.0",
            },
        },
        "official_cuda12_base_image": {
            "reference": "nvcr.io/nvidia/pytorch:25.01-py3",
            "manifest_list_digest": (
                "sha256:96990c82825613c3bdeebb66675c7c91b0123f64a5895623316dc5b824e0d7a9"
            ),
            "linux_amd64_digest": (
                "sha256:0981807f1a51a156563e28b59dc2e7a9b5c1c7d85d1169d4965c5fd91fa38bcb"
            ),
        },
        "modes": [
            "artifixer_direct_sequence_generation",
            "artifixer3d_fresh_3dgrut_distillation",
            "artifixer3d_plus_postprocess_over_distilled_renders",
        ],
        "runtime": "Linux;CUDA12_or_13;PyTorch;Diffusers;3DGRUT;FlashAttention",
        "status": "rejected_pending_model_license_and_runtime_qualification",
        "blockers": [
            "commercial_model_use_not_qualified",
            "checkpoint_digest_not_pinned_in_worker",
            "base_model_digest_not_pinned_in_worker",
            "worker_image_not_built_or_smoke_tested",
            "real_heldout_baseline_comparison_not_executed",
        ],
    },
    "difix3d": {
        "display_name": "NVIDIA Difix3D+",
        "source_repository": "https://github.com/nv-tlabs/Difix3D",
        "source_commit": "c76edc595586e16732c91ddee82f3a6d83a8a9cc",
        "source_license": "NVIDIA License; noncommercial_research_or_evaluation_only",
        "model_repository": "https://huggingface.co/nvidia/difix_ref",
        "model_license": "NVIDIA License plus Stability AI Community License",
        "modes": ["single_frame_postprocess", "iterative_3d_distillation"],
        "runtime": "Linux;CUDA;PyTorch;diffusers-0.25.1;transformers-4.38.0;gsplat_or_nerfstudio",
        "status": "rejected_noncommercial_default",
        "blockers": [
            "source_and_model_license_noncommercial",
            "commercial_license_receipt_missing",
            "trust_remote_code_not_permitted_in_pinned_worker",
            "checkpoint_digest_not_pinned_in_worker",
            "temporal_consistency_not_established_for_single_frame_mode",
            "real_heldout_baseline_comparison_not_executed",
        ],
    },
    "fixer": {
        "display_name": "NVIDIA Fixer V2 (Cosmos)",
        "source_repository": "https://github.com/nv-tlabs/Fixer",
        "source_commit": "b39dfcaf4eeec90dc943b057ff368c16252c6c6e",
        "source_license": "Apache-2.0",
        "model_repository": "https://huggingface.co/nvidia/Fixer",
        "model_license": "NVIDIA Open Model License Agreement; commercial_use_permitted",
        "modes": ["single_frame_nontemporal", "temporally_conditioned_sequence"],
        "runtime": "Linux;CUDA;PyTorch;Cosmos-Predict-0.6B",
        "status": "rejected_pending_checkpoint_runtime_qualification",
        "blockers": [
            "dependency_license_receipt_missing",
            "checkpoint_digest_not_pinned_in_worker",
            "cosmos_base_model_digest_not_pinned_in_worker",
            "worker_image_not_built_or_smoke_tested",
            "real_heldout_baseline_comparison_not_executed",
        ],
    },
    "harmonizer": {
        "display_name": "NVIDIA DiffusionHarmonizer",
        "source_repository": "https://github.com/NVIDIA/harmonizer",
        "source_commit": "d9a817c8376f82000721a52f9d740ef5c24f47bd",
        "source_license": "repository_and_dependency_review_required",
        "model_repository": "https://huggingface.co/nvidia/Harmonizer",
        "model_license": "NVIDIA Open Model License Agreement; commercial_use_permitted",
        "modes": ["single_frame_nontemporal", "temporally_conditioned_sequence", "offline_distillation"],
        "runtime": "Linux;CUDA;PyTorch;Cosmos-Predict2-0.6B;576x1024;Ampere_or_newer",
        "status": "rejected_pending_checkpoint_runtime_qualification",
        "blockers": [
            "source_and_dependency_license_receipt_missing",
            "checkpoint_digest_not_pinned_in_worker",
            "cosmos_base_model_digest_not_pinned_in_worker",
            "worker_image_not_built_or_smoke_tested",
            "real_heldout_baseline_comparison_not_executed",
        ],
    },
}


def enhancement_method_audit(method_id: str) -> dict[str, Any]:
    """Return the frozen, proof-bounded decision for one enhancement method."""

    if method_id not in _METHODS:
        raise ValueError(f"unknown_reconstruction_enhancement_method:{method_id}")
    value = {
        "schema_version": SCHEMA_VERSION,
        "method_id": method_id,
        **_METHODS[method_id],
        "candidate_registered": True,
        "baseline_reconstruction_required": True,
        "frozen_real_heldout_views_required": True,
        "hidden_heldout_available_to_candidate": False,
        "independent_evaluator_required": True,
        "unenhanced_baseline_preserved": True,
        "generated_output_provenance_required": True,
        "statistical_and_operational_improvement_required": True,
        "generated_pixels_are_captured_evidence": False,
        "metric_or_collision_proof_effect": False,
        "physical_or_deployment_proof_effect": False,
        "evaluation_evidence_use_permitted": False,
        "policy_input_use_permitted": False,
        "offline_reconstruction_modification_permitted": False,
        "presentation_enhancement_after_inputs_sealed_only": True,
        "proof_effect": "deterministic_enhancement_rejection_only",
        "claim_ceiling": "generated_visual_support",
        "legal_next_actions": [
            "obtain_required_license_receipts",
            "pin_source_models_dependencies_and_worker_image",
            "run_hermetic_worker_smoke",
            "compare_with_unenhanced_baseline_on_frozen_real_heldout_views",
            "preserve_evidence_and_stop",
        ],
    }
    value["enhancement_method_audit_digest"] = canonical_digest(
        value, digest_field="enhancement_method_audit_digest"
    )
    return value


def enhancement_method_audits() -> list[dict[str, Any]]:
    return [enhancement_method_audit(method_id) for method_id in sorted(_METHODS)]


__all__ = ["SCHEMA_VERSION", "enhancement_method_audit", "enhancement_method_audits"]
