"""Pinned cuRoboV2 candidate-generator adapter for the native GPU worker."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from .task_evaluation_collision_aware_candidate_generation import (
    CandidateGeneratorContext,
    CommandRunner,
    JsonProcessCandidateGenerator,
)


# v0.8.0 is the first tagged cuRoboV2 release.  Unlike v0.7.8 it supports the
# native worker's Python 3.12 runtime and is released under Apache-2.0.
CUROBO_BACKEND_IDENTITY: dict[str, Any] = {
    "backend_id": "curobo_v2_motion_generation",
    "package_name": "nvidia-curobo",
    "package_version": "0.8.0",
    "source_url": "https://github.com/NVlabs/curobo",
    "source_revision": "4ea77366ca48ee453e7df139e39fa6532af49f3b",
    "source_tree": "00eef6854824ced813a0a8550c4185c908eef968",
    "source_tag": "v0.8.0",
    "license_expression": "Apache-2.0",
    "license_url": "https://github.com/NVlabs/curobo/blob/v0.8.0/LICENSE",
    "license_sha256": "sha256:f39b6dc8b687586fe57a7d6dd6df4d241575e9dea459dbc4c2c080452397b45c",
    "runtime_kind": "isaac_gpu_python",
    "api_generation": "v2",
}


class CuroboCandidateGenerator(JsonProcessCandidateGenerator):
    """Generate collision-aware motion candidates on the owned GPU runtime."""

    def __init__(
        self,
        *,
        context: CandidateGeneratorContext,
        command: Sequence[str] | None = None,
        runner: CommandRunner | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if runner is not None:
            kwargs["runner"] = runner
        super().__init__(
            context=context,
            backend_identity=CUROBO_BACKEND_IDENTITY,
            command=command
            or (
                sys.executable,
                "-m",
                "blueprint_pipeline.task_evaluation_curobo_candidate_service",
            ),
            require_cuda=True,
            environment={
                "BLUEPRINT_CUROBO_SOURCE_REVISION": CUROBO_BACKEND_IDENTITY[
                    "source_revision"
                ]
            },
            **kwargs,
        )


def curobo_gpu_runtime_capability_contract() -> dict[str, Any]:
    """Provisioning requirements; this is not an availability assertion."""

    return {
        "schema_version": "task_evaluation_candidate_generator_capability.v1",
        "backend_identity": dict(CUROBO_BACKEND_IDENTITY),
        "required_capabilities": {
            "operating_system": "linux",
            "nvidia_gpu": True,
            "cuda_runtime": True,
            "torch_cuda": True,
            "minimum_cuda_compute_capability": "7.0",
            "minimum_gpu_memory_bytes": 4 * 1024**3,
            "python_requirement": ">=3.10",
            "isaac_sim_integration": "supported_separate_python_process",
            "sealed_robot_configuration_required": True,
            "sealed_world_configuration_required": True,
            "sealed_task_trajectory_required": True,
            "sealed_analytic_inventory_required": True,
        },
        "provisioning": {
            "install_mode": "exact_source_checkout_editable_no_build_isolation",
            "source_revision": CUROBO_BACKEND_IDENTITY["source_revision"],
            "python_distribution": "nvidia-curobo==0.8.0",
            "runtime_probe_required_before_generation": True,
            "kernel_warmup_required": True,
            "process_isolation_from_native_isaac_execution": True,
        },
        "claim_boundary": {
            "emits_collision_aware_candidate_evidence": True,
            "native_orientation_execution_unresolved": True,
            "native_collision_and_contact_readback_unresolved": True,
            "native_camera_observability_unresolved": True,
            "native_task_execution_unresolved": True,
        },
    }


__all__ = [
    "CUROBO_BACKEND_IDENTITY",
    "CuroboCandidateGenerator",
    "curobo_gpu_runtime_capability_contract",
]
