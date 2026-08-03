"""Concrete worker bindings for canonical Postshot and Splatfacto arms.

This module runs on an already-admitted worker. It does not allocate paid
compute, acquire licenses, or infer authorization. It only translates one
digest-bound arm and one immutable COLMAP dataset into the runner receipt
consumed by :mod:`canonical_3dgs_pipeline`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .canonical_3dgs_admission import require_canonical_3dgs_worker_admission
from .canonical_3dgs_pipeline import (
    PLAN_SCHEMA,
    POSTSHOT_METHOD,
    SPLATFACTO_METHOD,
    Canonical3DGSPipelineError,
    canonical_3dgs_worker_package_digest,
    verify_canonical_3dgs_plan_inputs,
)
from .canonical_3dgs_transport import validate_canonical_3dgs_transport_receipt
from .decision_evidence_contracts import canonical_digest, canonical_json
from .postshot_worker_contracts import (
    assert_secret_free,
    build_postshot_train_args,
    redact_command,
    sanitize_text,
)


CommandExecutor = Callable[[Sequence[str], Path, Path], int]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise Canonical3DGSPipelineError(["worker_immutable_artifact_conflict"])
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != payload:
                raise Canonical3DGSPipelineError(
                    ["worker_immutable_artifact_conflict"]
                )
    finally:
        temporary.unlink(missing_ok=True)


def _execute(arguments: Sequence[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as stream:
        deadline = float(os.environ.get("BLUEPRINT_CANONICAL_3DGS_DEADLINE_EPOCH", "0"))
        timeout = max(0.001, deadline - time.time()) if deadline > 0 else None
        try:
            completed = subprocess.run(
                list(arguments),
                cwd=cwd,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            stream.write(b"canonical_3dgs_worker_hard_ttl_expired\n")
            return 124
    return int(completed.returncode)


def _require_arm(arm: Mapping[str, Any], *, arm_id: str, method_id: str) -> None:
    if arm.get("arm_id") != arm_id or arm.get("method_id") != method_id:
        raise Canonical3DGSPipelineError([f"worker_arm_contract_invalid:{arm_id}"])


def _artifact(kind: str, path: Path, root: Path) -> dict[str, str]:
    resolved = path.resolve()
    if root.resolve() not in resolved.parents or not resolved.is_file() or resolved.is_symlink():
        raise Canonical3DGSPipelineError([f"worker_output_invalid:{kind}"])
    return {"kind": kind, "relative_path": resolved.relative_to(root.resolve()).as_posix()}


def _sanitize_postshot_log(path: Path, secrets: Sequence[str]) -> None:
    """Remove credentials and URL-shaped bearer material before collection."""

    if not path.is_file():
        return
    temporary = path.with_name(f".{path.name}.sanitized")
    try:
        with path.open("r", encoding="utf-8", errors="replace") as source, temporary.open(
            "w", encoding="utf-8"
        ) as destination:
            for line in source:
                destination.write(sanitize_text(line, secrets))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_postshot_arm(
    arm: Mapping[str, Any],
    dataset_root: Path,
    output_root: Path,
    *,
    environment: Mapping[str, str] | None = None,
    executor: CommandExecutor = _execute,
) -> dict[str, Any]:
    """Run the full-resolution Postshot Splat3 primary arm on Windows."""

    _require_arm(arm, arm_id="postshot-primary", method_id=POSTSHOT_METHOD)
    env = dict(os.environ if environment is None else environment)
    email = str(env.get("POSTSHOT_LOGIN_EMAIL") or "")
    password = str(env.get("POSTSHOT_LOGIN_PASSWORD") or "")
    if not email or not password:
        raise Canonical3DGSPipelineError(["postshot_runtime_credentials_missing"])
    executable = str(
        env.get("POSTSHOT_CLI_PATH")
        or r"C:\Program Files\Jawset Postshot\bin\postshot-cli.exe"
    )
    run_root = output_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    project = run_root / "postshot-primary.psht"
    splat = run_root / "postshot-primary.ply"
    log = run_root / "training.log"
    arguments = build_postshot_train_args(
        login_email=email,
        login_password=password,
        dataset=str(dataset_root.resolve()),
        profile="Splat3",
        output_project=str(project),
        output_splat=str(splat),
        max_image_size=0,
    )
    actual_argv = [executable, *arguments]
    exit_code = executor(actual_argv, run_root, log)
    _sanitize_postshot_log(log, (email, password))
    redacted = redact_command(actual_argv, (email, password))
    artifacts = [_artifact("training_log", log, run_root)]
    if project.is_file():
        artifacts.append(_artifact("postshot_project", project, run_root))
    if splat.is_file():
        artifacts.append(_artifact("standard_3dgs_ply", splat, run_root))
    receipt = {
        "exit_code": exit_code,
        "argv": redacted,
        "runtime_identity": {
            "platform": "windows",
            "product": "Jawset Postshot CLI",
            "profile": "Splat3",
            "full_resolution": True,
        },
        "artifacts": artifacts,
        "timestamp": _now(),
    }
    assert_secret_free(receipt, (email, password))
    return receipt


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def validate_trainer_runtime_binding(
    arm_id: str,
    admission: Mapping[str, Any],
    environment: Mapping[str, str],
) -> dict[str, str]:
    if arm_id == "postshot-primary":
        executable = Path(
            environment.get("POSTSHOT_CLI_PATH")
            or r"C:\Program Files\Jawset Postshot\bin\postshot-cli.exe"
        )
        if not executable.is_file() or executable.is_symlink():
            raise Canonical3DGSPipelineError(["postshot_cli_binary_missing"])
        digest = _sha256_file(executable)
        product = "Jawset Postshot CLI"
    else:
        versions = {
            "nerfstudio": _package_version("nerfstudio"),
            "gsplat": _package_version("gsplat"),
        }
        if versions != {"nerfstudio": "1.1.5", "gsplat": "1.4.0"}:
            raise Canonical3DGSPipelineError(["splatfacto_runtime_version_mismatch"])
        digest = canonical_digest(
            {
                "product": "Nerfstudio Splatfacto",
                "nerfstudio_version": versions["nerfstudio"],
                "gsplat_version": versions["gsplat"],
            }
        )
        product = "Nerfstudio Splatfacto"
    if digest != admission.get("trainer_runtime_digest"):
        raise Canonical3DGSPipelineError(["trainer_runtime_digest_mismatch"])
    return {
        "trainer_product": product,
        "trainer_runtime_digest": digest,
        "trainer_runtime_version": str(admission["trainer_runtime_version"]),
    }


def validate_worker_image_binding(
    admission: Mapping[str, Any], environment: Mapping[str, str]
) -> str:
    observed = str(environment.get("BLUEPRINT_WORKER_IMAGE_DIGEST") or "")
    expected = str(admission.get("worker_image_digest") or "")
    if not observed or observed != expected:
        raise Canonical3DGSPipelineError(["worker_image_digest_mismatch"])
    return observed


def run_splatfacto_arm(
    arm: Mapping[str, Any],
    dataset_root: Path,
    output_root: Path,
    *,
    environment: Mapping[str, str] | None = None,
    executor: CommandExecutor = _execute,
    runtime_versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run pinned Nerfstudio Splatfacto and export a standard 3DGS PLY."""

    _require_arm(
        arm,
        arm_id="splatfacto-comparison",
        method_id=SPLATFACTO_METHOD,
    )
    env = dict(os.environ if environment is None else environment)
    train_binary = str(env.get("NS_TRAIN_PATH") or "ns-train")
    export_binary = str(env.get("NS_EXPORT_PATH") or "ns-export")
    versions = dict(
        runtime_versions
        or {
            "nerfstudio": _package_version("nerfstudio"),
            "gsplat": _package_version("gsplat"),
        }
    )
    if versions != {"nerfstudio": "1.1.5", "gsplat": "1.4.0"}:
        raise Canonical3DGSPipelineError(["splatfacto_runtime_version_mismatch"])
    run_root = output_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    log = run_root / "training.log"
    train_argv = [
        train_binary,
        "splatfacto",
        "--vis",
        "tensorboard",
        "--max-num-iterations",
        "30000",
        "--machine.seed",
        "42",
        "--pipeline.model.cull_alpha_thresh=0.005",
        # In Nerfstudio 1.1.5, DefaultStrategy stops both growth and pruning at
        # stop_split_at. `continue_cull_post_densification` was added after this
        # pinned release and must not be passed to the 1.1.5 CLI.
        "--pipeline.model.stop_split_at",
        "15000",
        "--data",
        str(dataset_root.resolve()),
        "colmap",
        "--colmap-path",
        "sparse/0",
        "--downscale-factor",
        "1",
        "--orientation-method",
        "none",
        "--center-method",
        "none",
        "--auto-scale-poses",
        "False",
        "--assume-colmap-world-coordinate-convention",
        "False",
        "--eval-mode",
        "all",
    ]
    train_code = executor(train_argv, run_root, log)
    export_argv: list[str] = []
    final_code = train_code
    config: Path | None = None
    splat: Path | None = None
    if train_code == 0:
        configs = sorted(path for path in run_root.rglob("config.yml") if path.is_file())
        if len(configs) != 1:
            final_code = 70
        else:
            config = configs[0]
            export_root = run_root / "export"
            export_root.mkdir(parents=True, exist_ok=True)
            export_argv = [
                export_binary,
                "gaussian-splat",
                "--load-config",
                str(config),
                "--output-dir",
                str(export_root),
            ]
            final_code = executor(export_argv, run_root, log)
            if final_code == 0:
                splats = sorted(path for path in export_root.rglob("*.ply") if path.is_file())
                if len(splats) != 1:
                    final_code = 71
                else:
                    splat = splats[0]
    artifacts = [_artifact("training_log", log, run_root)]
    if config is not None:
        artifacts.append(_artifact("nerfstudio_config", config, run_root))
    if splat is not None:
        artifacts.append(_artifact("standard_3dgs_ply", splat, run_root))
    return {
        "exit_code": final_code,
        "argv": [train_argv, export_argv],
        "runtime_identity": {
            "platform": "linux",
            "product": "Nerfstudio Splatfacto",
            "nerfstudio_version": versions["nerfstudio"],
            "gsplat_version": versions["gsplat"],
        },
        "artifacts": artifacts,
        "timestamp": _now(),
    }


def build_runner(
    arm_id: str,
    *,
    environment: Mapping[str, str] | None = None,
    executor: CommandExecutor = _execute,
) -> Callable[[Mapping[str, Any], Path, Path], Mapping[str, Any]]:
    if arm_id == "postshot-primary":
        return lambda arm, dataset, output: run_postshot_arm(
            arm,
            dataset,
            output,
            environment=environment,
            executor=executor,
        )
    if arm_id == "splatfacto-comparison":
        return lambda arm, dataset, output: run_splatfacto_arm(
            arm,
            dataset,
            output,
            environment=environment,
            executor=executor,
        )
    raise Canonical3DGSPipelineError([f"worker_arm_unknown:{arm_id}"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("postshot-primary", "splatfacto-comparison"))
    parser.add_argument("--plan", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--transport-receipt", required=True)
    parser.add_argument("--admission", required=True)
    arguments = parser.parse_args(argv)
    plan = json.loads(Path(arguments.plan).read_text(encoding="utf-8"))
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("canonical_3dgs_execution_plan_digest")
        != canonical_digest(plan, digest_field="canonical_3dgs_execution_plan_digest")
    ):
        raise Canonical3DGSPipelineError(["execution_plan_invalid"])
    verify_canonical_3dgs_plan_inputs(plan=plan, dataset_root=arguments.dataset_root)
    transport = validate_canonical_3dgs_transport_receipt(
        json.loads(Path(arguments.transport_receipt).read_text(encoding="utf-8"))
    )
    admission = require_canonical_3dgs_worker_admission(
        json.loads(Path(arguments.admission).read_text(encoding="utf-8")),
        arm_id=arguments.arm,
        plan_digest=plan["canonical_3dgs_execution_plan_digest"],
        dataset_digest=plan["colmap_training_dataset_digest"],
        transport_bundle_digest=transport["transport_bundle_digest"],
        worker_package_digest=plan["worker_python_package_digest"],
    )
    trainer_runtime = validate_trainer_runtime_binding(
        arguments.arm, admission, os.environ
    )
    worker_image_digest = validate_worker_image_binding(admission, os.environ)
    expires = datetime.fromisoformat(str(admission["expires_at"]).replace("Z", "+00:00"))
    os.environ["BLUEPRINT_CANONICAL_3DGS_DEADLINE_EPOCH"] = str(expires.timestamp())
    matches = [row for row in plan.get("arms") or [] if row.get("arm_id") == arguments.arm]
    if len(matches) != 1:
        raise Canonical3DGSPipelineError([f"worker_arm_missing:{arguments.arm}"])
    receipt = dict(build_runner(arguments.arm)(
        matches[0],
        Path(arguments.dataset_root),
        Path(arguments.output_root),
    ))
    runtime_identity = dict(receipt.get("runtime_identity") or {})
    runtime_identity["worker_python_package_digest"] = (
        canonical_3dgs_worker_package_digest()
    )
    runtime_identity["source_commit_sha_bound_by_plan"] = plan["source_commit_sha"]
    runtime_identity.update(trainer_runtime)
    runtime_identity["worker_image_digest"] = worker_image_digest
    receipt["runtime_identity"] = runtime_identity
    receipt["canonical_3dgs_execution_plan_digest"] = plan[
        "canonical_3dgs_execution_plan_digest"
    ]
    receipt["transport_bundle_digest"] = transport["transport_bundle_digest"]
    receipt["transport_receipt_digest"] = transport["receipt_digest"]
    receipt["canonical_3dgs_worker_admission_digest"] = admission[
        "canonical_3dgs_worker_admission_digest"
    ]
    receipt["allocation_binding_digest"] = admission["allocation_binding_digest"]
    receipt["provider_zero_required_after_execution"] = True
    output_root = Path(arguments.output_root).expanduser().resolve()
    transport_snapshot = output_root / "canonical_3dgs_transport_receipt.json"
    admission_snapshot = output_root / "canonical_3dgs_worker_admission.json"
    _write_immutable_json(transport_snapshot, transport)
    _write_immutable_json(admission_snapshot, admission)
    receipt["transport_receipt_relative_path"] = transport_snapshot.relative_to(
        output_root
    ).as_posix()
    receipt["worker_admission_relative_path"] = admission_snapshot.relative_to(
        output_root
    ).as_posix()
    receipt["canonical_3dgs_worker_receipt_digest"] = canonical_digest(
        receipt, digest_field="canonical_3dgs_worker_receipt_digest"
    )
    receipt_path = Path(arguments.receipt)
    _write_immutable_json(receipt_path, receipt)
    return 0 if receipt["exit_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_runner",
    "run_postshot_arm",
    "run_splatfacto_arm",
    "validate_trainer_runtime_binding",
    "validate_worker_image_binding",
]
