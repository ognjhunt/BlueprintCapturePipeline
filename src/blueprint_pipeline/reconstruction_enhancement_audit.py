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
        "source_license": "Apache-2.0",
        "model_repository": "https://huggingface.co/nvidia/ArtiFixer",
        "model_license": "NVIDIA License; research_and_development_only",
        "modes": ["single_sequence_generation", "iterative_3d_distillation"],
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
    "harmonizer": {
        "display_name": "NVIDIA DiffusionHarmonizer",
        "source_repository": "https://github.com/NVIDIA/harmonizer",
        "source_commit": "d9a817c8376f82000721a52f9d740ef5c24f47bd",
        "source_license": "repository_and_dependency_review_required",
        "model_repository": "https://huggingface.co/nvidia/Harmonizer",
        "model_license": "NVIDIA Open Model License Agreement",
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
