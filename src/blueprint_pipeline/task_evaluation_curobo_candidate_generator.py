"""Pinned cuRoboV2 candidate-generator adapter for the native GPU worker."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
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


def _enroll_warm_host_key(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    # Control-plane-only transport dependency: keep the provider runtime's
    # import closure free of spend/provider modules.  The bundled cuRobo service
    # imports this file only for the immutable backend identity above.
    from .gpu_render_providers import enroll_vast_ssh_host_key

    return enroll_vast_ssh_host_key(*args, **kwargs)


def _run_warm_ssh(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    from .native_task_arena_warm_vast import _run_pinned_ssh

    return _run_pinned_ssh(*args, **kwargs)


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


def _last_meaningful_line(value: Any) -> str:
    """The last non-empty redacted stderr line, bounded for a predicate."""

    lines = [line.strip() for line in str(value or "").splitlines() if line.strip()]
    if not lines:
        return ""
    return re.sub(r"[^A-Za-z0-9_.:/ +-]", "_", lines[-1])[:160]


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
        remote_python_package_root: str | None = None,
        identity_file: str | Path | None = None,
    ) -> None:
        remote_work_dir = str(warm_session.get("remote_work_dir") or "")
        if remote_work_dir not in {"/workspace", "/tmp/blueprint_vast_work"}:
            raise CollisionAwareCandidateGenerationError(
                "curobo_remote_work_dir_invalid"
            )
        expected_package_root = (
            remote_work_dir + "/adp_arena_provider_bundle/provider_runtime"
        )
        package_root = remote_python_package_root or expected_package_root
        if package_root != expected_package_root:
            raise CollisionAwareCandidateGenerationError(
                "curobo_remote_runtime_root_mismatch"
            )
        remote_root = PurePosixPath(package_root)
        if (
            not re.fullmatch(
                r"/(workspace|tmp/blueprint_vast_work)/[A-Za-z0-9_./-]+",
                package_root,
            )
            or not remote_root.is_absolute()
            or ".." in remote_root.parts
            or not (
                remote_root.parts[:2] == ("/", "workspace")
                or remote_root.parts[:3] == ("/", "tmp", "blueprint_vast_work")
            )
        ):
            raise CollisionAwareCandidateGenerationError(
                "curobo_remote_runtime_root_invalid"
            )
        self._context = context
        self._session = json.loads(json.dumps(dict(warm_session), allow_nan=False))
        self._remote_work_dir = remote_work_dir
        self._root = Path(local_transport_root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._remote_python_package_root = package_root.rstrip("/")
        self._identity_file = identity_file
        enrollment = _enroll_warm_host_key(
            self._session,
            attempt_dir=self._root / "ssh-trust",
            timeout_seconds=15.0,
        )
        if enrollment.get("status") != "enrolled":
            raise CollisionAwareCandidateGenerationError(
                "curobo_remote_ssh_unavailable"
            )
        self._known_hosts = str(enrollment["known_hosts_file"])
        self._provision_runtime()

    def _ssh(
        self,
        argv: list[str],
        *,
        stdin: bytes | None = None,
        timeout_seconds: float,
        maximum_timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if maximum_timeout_seconds is not None:
            extra["maximum_timeout_seconds"] = maximum_timeout_seconds
        result = _run_warm_ssh(
            session=self._session,
            known_hosts_file=self._known_hosts,
            identity_file=self._identity_file,
            remote_argv=argv,
            stdin=stdin,
            timeout_seconds=timeout_seconds,
            **extra,
        )
        if result.get("status") != "completed":
            # The transport already reports why -- a timeout, or an exit code --
            # and redacts the output before returning it. Collapsing that into
            # one generic word costs a GPU allocation to rediscover, so the
            # predicate carries the transport blocker and the exit status, and
            # the redacted transcript is retained next to the run's evidence.
            self._retain_ssh_failure(result)
            raise CollisionAwareCandidateGenerationError(
                ":".join(
                    [
                        "curobo_remote_process_failed",
                        *(str(value) for value in (result.get("blockers") or [])),
                        *(
                            [f"exit_{result['returncode']}"]
                            if result.get("returncode") is not None
                            else []
                        ),
                        *(
                            [_last_meaningful_line(result.get("stderr"))]
                            if _last_meaningful_line(result.get("stderr"))
                            else []
                        ),
                    ]
                )
            )
        return result

    def _retain_ssh_failure(self, result: Mapping[str, Any]) -> None:
        """Keep the redacted remote transcript; a torn-down GPU cannot be asked again."""

        directory = self._root / "warm-ssh-failures"
        try:
            directory.mkdir(mode=0o750, parents=True, exist_ok=True)
            path = directory / f"failure-{len(list(directory.glob('failure-*.json'))):03d}.json"
            path.write_text(
                json.dumps(dict(result), indent=1, sort_keys=True, default=str),
                encoding="utf-8",
            )
        except OSError:
            # Evidence retention must never mask the failure it documents.
            return

    def _provision_runtime(self) -> None:
        source_root = (
            self._remote_work_dir + "/blueprint-curobo-v080/"
            + CUROBO_BACKEND_IDENTITY["source_revision"]
        )
        script = f"""set -euo pipefail
mkdir -p {self._remote_work_dir}/blueprint-curobo-v080
exec 9>{self._remote_work_dir}/blueprint-curobo-v080/provision.lock
flock 9
test -f {self._remote_python_package_root}/blueprint_pipeline/task_evaluation_curobo_candidate_service.py || exit 81
root={source_root}
if [ ! -d "$root/.git" ]; then
  stage="$(mktemp -d {self._remote_work_dir}/blueprint-curobo-v080/stage.XXXXXXXX)"
  git clone --filter=blob:none --no-checkout {CUROBO_BACKEND_IDENTITY['source_url']} "$stage"
  git -C "$stage" fetch --depth 1 origin {CUROBO_BACKEND_IDENTITY['source_revision']}
  git -C "$stage" checkout --detach {CUROBO_BACKEND_IDENTITY['source_revision']}
  test "$(git -C "$stage" rev-parse HEAD)" = {CUROBO_BACKEND_IDENTITY['source_revision']}
  test "$(git -C "$stage" rev-parse 'HEAD^{{tree}}')" = {CUROBO_BACKEND_IDENTITY['source_tree']}
  test "sha256:$(sha256sum "$stage/LICENSE" | cut -d' ' -f1)" = {CUROBO_BACKEND_IDENTITY['license_sha256']}
  mv "$stage" "$root"
fi
test "$(git -C "$root" rev-parse HEAD)" = {CUROBO_BACKEND_IDENTITY['source_revision']}
test "$(git -C "$root" rev-parse 'HEAD^{{tree}}')" = {CUROBO_BACKEND_IDENTITY['source_tree']}
test "sha256:$(sha256sum "$root/LICENSE" | cut -d' ' -f1)" = {CUROBO_BACKEND_IDENTITY['license_sha256']}
/isaac-sim/python.sh -m pip install -e "$root" --no-deps --no-build-isolation
PYTHONPATH="$root" /isaac-sim/python.sh - <<'PY'
import importlib.metadata
import curobo
assert importlib.metadata.version("nvidia-curobo") == "0.8.0"
assert str(curobo.__version__).lstrip("v") == "0.8.0"
PY
printf '%s\n' BLUEPRINT_CUROBO_RUNTIME_READY
"""
        # Cloning curobo and building its CUDA extensions does not finish in a
        # five minute probe budget on a cold container.
        result = self._ssh(
            ["/bin/bash", "-c", script],
            timeout_seconds=1500.0,
            maximum_timeout_seconds=1800.0,
        )
        if "BLUEPRINT_CUROBO_RUNTIME_READY" not in str(result.get("stdout") or ""):
            raise CollisionAwareCandidateGenerationError(
                "curobo_remote_runtime_unavailable"
            )
        self._remote_curobo_source_root = source_root

    def _invoke(self, request: Mapping[str, Any] | None) -> dict[str, Any]:
        key = (
            "probe"
            if request is None
            else str(request["request_digest"])[7:]
        )
        remote_root = f"{self._remote_work_dir}/blueprint-curobo-candidates/{key}"
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
            (
                "PYTHONPATH="
                + self._remote_python_package_root
                + ":"
                + self._remote_curobo_source_root
            ),
            (
                "BLUEPRINT_CUROBO_SOURCE_REVISION="
                + CUROBO_BACKEND_IDENTITY["source_revision"]
            ),
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

    def _upload_input(self, reference: Mapping[str, Any]) -> dict[str, Any]:
        local_path = Path(str(reference["path"]))
        try:
            payload = local_path.read_bytes()
        except OSError as exc:
            raise CollisionAwareCandidateGenerationError(
                "curobo_remote_input_unavailable"
            ) from exc
        digest = str(reference["digest"])
        if len(payload) != reference["size_bytes"]:
            raise CollisionAwareCandidateGenerationError(
                "curobo_remote_input_invalid"
            )
        suffix = local_path.suffix if re.fullmatch(r"\.[A-Za-z0-9]{1,8}", local_path.suffix) else ".bin"
        remote_path = (
            f"{self._remote_work_dir}/blueprint-curobo-inputs/"
            f"{digest[7:]}{suffix}"
        )
        upload_code = (
            "import hashlib,os,sys,pathlib;"
            "p=pathlib.Path(sys.argv[1]);e=sys.argv[2];d=sys.stdin.buffer.read();"
            "a='sha256:'+hashlib.sha256(d).hexdigest();"
            "(_ for _ in ()).throw(RuntimeError('digest')) if a!=e else None;"
            "p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');"
            "t.write_bytes(d);os.chmod(t,0o600);os.replace(t,p)"
        )
        self._ssh(
            [
                "/isaac-sim/python.sh",
                "-c",
                upload_code,
                remote_path,
                digest,
            ],
            stdin=payload,
            timeout_seconds=max(30.0, min(300.0, len(payload) / 1_000_000 + 30.0)),
        )
        return {**dict(reference), "path": remote_path}

    def _stage_remote_request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        remote = json.loads(json.dumps(dict(request), allow_nan=False))
        for role in (
            "robot_configuration",
            "world_configuration",
            "task_trajectory",
            "analytic_candidate_inventory",
        ):
            source_reference = dict(remote[role])
            remote_attachments = []
            replacements: dict[str, str] = {}
            for attachment in source_reference.get("attachments") or []:
                uploaded = self._upload_input(attachment)
                replacements[str(attachment["path"])] = str(uploaded["path"])
                remote_attachments.append(uploaded)
            if replacements:
                source_document = json.loads(
                    Path(str(source_reference["path"])).read_text(encoding="utf-8")
                )

                def replace(value: Any) -> Any:
                    if isinstance(value, str):
                        return replacements.get(value, value)
                    if isinstance(value, list):
                        return [replace(row) for row in value]
                    if isinstance(value, Mapping):
                        return {str(key): replace(child) for key, child in value.items()}
                    return value

                transported = (canonical_json(replace(source_document)) + "\n").encode()
                transported_path = self._root / f"transport-{role}.json"
                transported_path.write_bytes(transported)
                import hashlib

                source_reference = {
                    **source_reference,
                    "path": str(transported_path),
                    "size_bytes": len(transported),
                    "digest": "sha256:" + hashlib.sha256(transported).hexdigest(),
                    "source_digest": source_reference["digest"],
                    "attachments": remote_attachments,
                }
            remote[role] = self._upload_input(source_reference)
            if remote_attachments:
                remote[role]["attachments"] = remote_attachments
        remote["request_digest"] = canonical_digest(
            remote, digest_field="request_digest"
        )
        return remote

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
        remote_request = self._stage_remote_request(request)
        return build_native_candidate_inventory(
            result=self._invoke(remote_request),
            request=remote_request,
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
