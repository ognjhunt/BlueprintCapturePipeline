"""Pinned cuRoboV2 candidate-generator adapter for the native GPU worker."""

from __future__ import annotations

import sys
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_json
from .gpu_render_providers import enroll_vast_ssh_host_key
from .native_task_arena_warm_vast import _run_pinned_ssh
from .task_evaluation_collision_aware_candidate_generation import (
    CandidateGeneratorContext,
    CollisionAwareCandidateGenerationError,
    CommandRunner,
    JsonProcessCandidateGenerator,
    build_candidate_generation_request,
    build_native_candidate_inventory,
    validate_runtime_probe,
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


class RemoteCuroboCandidateGenerator:
    """Run the exact cuRobo service on an already-owned warm Vast worker.

    The transport allocates nothing.  It uses the existing production SSH
    identity and attempt-local pinned host key, streams a canonical request to
    a digest-addressed remote checkpoint, executes the bundled service, and
    streams the exact result back.  Runtime provisioning must supply both the
    Blueprint service package root and pinned cuRobo source; absence is a typed
    refusal that the composite controller may record before CPU fallback.
    """

    def __init__(
        self,
        *,
        context: CandidateGeneratorContext,
        warm_session: Mapping[str, Any],
        local_transport_root: str | Path,
        remote_python_package_root: str,
        identity_file: str | Path | None = None,
    ) -> None:
        remote_root = PurePosixPath(remote_python_package_root)
        if (
            not re.fullmatch(r"/workspace/[A-Za-z0-9_./-]+", remote_python_package_root)
            or not remote_root.is_absolute()
            or ".." in remote_root.parts
            or remote_root.parts[:2] != ("/", "workspace")
        ):
            raise CollisionAwareCandidateGenerationError(
                "curobo_remote_runtime_root_invalid"
            )
        self._context = context
        self._session = json.loads(json.dumps(dict(warm_session), allow_nan=False))
        self._root = Path(local_transport_root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._remote_python_package_root = remote_python_package_root.rstrip("/")
        self._identity_file = identity_file
        enrollment = enroll_vast_ssh_host_key(
            self._session,
            attempt_dir=self._root / "ssh-trust",
            timeout_seconds=15.0,
        )
        if enrollment.get("status") != "enrolled":
            raise CollisionAwareCandidateGenerationError(
                "curobo_remote_ssh_unavailable"
            )
        self._known_hosts = str(enrollment["known_hosts_file"])

    def _ssh(
        self,
        argv: list[str],
        *,
        stdin: bytes | None = None,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        result = _run_pinned_ssh(
            session=self._session,
            known_hosts_file=self._known_hosts,
            identity_file=self._identity_file,
            remote_argv=argv,
            stdin=stdin,
            timeout_seconds=timeout_seconds,
        )
        if result.get("status") != "completed":
            raise CollisionAwareCandidateGenerationError(
                "curobo_remote_process_failed"
            )
        return result

    def _invoke(self, request: Mapping[str, Any] | None) -> dict[str, Any]:
        key = (
            "probe"
            if request is None
            else str(request["request_digest"])[7:]
        )
        remote_root = f"/workspace/blueprint-curobo-candidates/{key}"
        request_path = f"{remote_root}/request.json"
        result_path = f"{remote_root}/result.json"
        if request is not None:
            payload = (canonical_json(dict(request)) + "\n").encode("utf-8")
            upload_code = (
                "import os,sys,pathlib,tempfile;"
                "p=pathlib.Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);"
                "d=sys.stdin.buffer.read();t=p.with_suffix('.tmp');"
                "t.write_bytes(d);os.chmod(t,0o600);os.replace(t,p)"
            )
            self._ssh(
                ["/isaac-sim/python.sh", "-c", upload_code, request_path],
                stdin=payload,
                timeout_seconds=30.0,
            )
        command = [
            "env",
            f"PYTHONPATH={self._remote_python_package_root}",
            (
                "BLUEPRINT_CUROBO_SOURCE_REVISION="
                + CUROBO_BACKEND_IDENTITY["source_revision"]
            ),
            "BLUEPRINT_SOURCE_COMMIT=" + self._context.expected_production_commit,
            "/isaac-sim/python.sh",
            "-m",
            "blueprint_pipeline.task_evaluation_curobo_candidate_service",
        ]
        if request is None:
            command.append("--probe")
        else:
            command.extend(("--request-json", request_path))
        command.extend(("--result-json", result_path))
        self._ssh(
            command,
            timeout_seconds=max(30.0, self._context.maximum_runtime_seconds + 30.0),
        )
        downloaded = self._ssh(
            ["cat", "--", result_path], timeout_seconds=30.0
        )
        truncation = downloaded.get("stdout_truncation") or {}
        if truncation.get("truncated") is True:
            raise CollisionAwareCandidateGenerationError(
                "curobo_remote_result_too_large"
            )
        try:
            value = json.loads(str(downloaded.get("stdout") or ""))
        except json.JSONDecodeError as exc:
            raise CollisionAwareCandidateGenerationError(
                "curobo_remote_result_invalid"
            ) from exc
        if not isinstance(value, Mapping):
            raise CollisionAwareCandidateGenerationError(
                "curobo_remote_result_invalid"
            )
        return dict(value)

    def generate(
        self,
        *,
        source_native_feedback: Mapping[str, Any] | None,
        prior_history: Sequence[Mapping[str, Any]],
        round_index: int,
        maximum_candidates: int,
    ) -> Mapping[str, Any]:
        probe = validate_runtime_probe(
            self._invoke(None), expected_backend_identity=CUROBO_BACKEND_IDENTITY
        )
        if (
            probe.get("cuda_available") is not True
            or int(probe.get("cuda_device_count") or 0) < 1
        ):
            raise CollisionAwareCandidateGenerationError(
                "candidate_generator_cuda_unavailable"
            )
        request = build_candidate_generation_request(
            context=self._context,
            backend_identity=CUROBO_BACKEND_IDENTITY,
            source_native_feedback=source_native_feedback,
            prior_history=prior_history,
            round_index=round_index,
            maximum_candidates=maximum_candidates,
        )
        return build_native_candidate_inventory(
            result=self._invoke(request),
            request=request,
            backend_identity=CUROBO_BACKEND_IDENTITY,
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
    "RemoteCuroboCandidateGenerator",
    "curobo_gpu_runtime_capability_contract",
]
