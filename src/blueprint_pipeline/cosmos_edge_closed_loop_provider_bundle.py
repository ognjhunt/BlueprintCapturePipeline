"""Build the first identity-bound Cosmos Edge policy canary provider bundle."""

from __future__ import annotations

import shutil
import stat
import textwrap
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .common import ensure_dir, utc_now_iso, write_json
from .policy_ranking_successor_cosmos import VLLM_IMAGE, VLLM_IMAGE_DIGEST
from .policy_ranking_thesis import canonical_sha256, file_sha256


BUNDLE_SCHEMA = "cosmos_edge_closed_loop_provider_bundle.v1"
RECEIPT_SCHEMA = "cosmos_edge_closed_loop_bundle_receipt.v1"
EXPERIMENT_ID = "policy_ranking_cosmos3_edge_closed_loop_20260729"
FRAMEWORK_URL = "https://github.com/NVIDIA/cosmos-framework.git"
FRAMEWORK_REVISION = "2f603cb114ff8b335e116060444d0b6caee3a85e"
MODEL_ID = "nvidia/Cosmos3-Edge-Policy-DROID"
MODEL_REVISION = "3ea407af3e156c0af3b4bb6edd85842cc9a58777"
PUBLIC_IMAGE = f"{VLLM_IMAGE}@{VLLM_IMAGE_DIGEST}"


REMOTE_RUNNER = textwrap.dedent(
    r"""#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

FRAMEWORK_URL = "https://github.com/NVIDIA/cosmos-framework.git"
FRAMEWORK_REVISION = "2f603cb114ff8b335e116060444d0b6caee3a85e"
MODEL_ID = "nvidia/Cosmos3-Edge-Policy-DROID"
MODEL_REVISION = "3ea407af3e156c0af3b4bb6edd85842cc9a58777"
REGISTERED_NEXT_WAM = "OSCAR-2B"


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(argv: list[str], *, cwd: Path | None = None, env: dict | None = None, timeout: int = 3600) -> None:
    subprocess.run(argv, cwd=cwd, env=env, check=True, timeout=timeout)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    started = time.time()
    bundle = Path(os.environ["BLUEPRINT_WAM_PROVIDER_BUNDLE_DIR"]).resolve()
    runtime = bundle / "provider_runtime"
    output = Path(os.environ["BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "wam_runtime_result.json"
    process = None
    blockers: list[str] = []
    canary = {}
    try:
        gpu_probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        work = Path(
            os.environ.get("BLUEPRINT_EDGE_POLICY_WORK_DIR", "/workspace/blueprint_edge_runtime_work")
        ).resolve()
        legacy_work = output / "runtime_work"
        if legacy_work.exists() and not work.exists():
            legacy_work.rename(work)
        work.mkdir(parents=True, exist_ok=True)
        source = work / "cosmos-framework"
        if not source.exists():
            run(["git", "clone", "--filter=blob:none", FRAMEWORK_URL, str(source)], timeout=900)
        run(["git", "fetch", "origin", FRAMEWORK_REVISION, "--depth=1"], cwd=source, timeout=300)
        run(["git", "checkout", "--detach", FRAMEWORK_REVISION], cwd=source, timeout=120)
        resolved = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=source, check=True,
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        if resolved != FRAMEWORK_REVISION:
            raise RuntimeError("cosmos_framework_revision_mismatch")
        uv = work / "uv"
        if not uv.exists():
            run([sys.executable, "-m", "pip", "install", "--target", str(uv), "uv==0.8.17"], timeout=300)
        uv_bin = uv / "bin/uv"
        if uv_bin is None or not uv_bin.is_file():
            raise RuntimeError("uv_bootstrap_missing")
        run(
            [str(uv_bin), "sync", "--all-extras", "--group=cu128-train", "--group=policy-server"],
            cwd=source, timeout=2400,
        )
        python = source / ".venv/bin/python"
        if os.environ.get("BLUEPRINT_EDGE_POLICY_VENV_REEXEC") != "1":
            reexec_env = os.environ.copy()
            reexec_env["BLUEPRINT_EDGE_POLICY_VENV_REEXEC"] = "1"
            reexec_env["PYTHONPATH"] = str(runtime) + os.pathsep + str(source)
            os.execve(str(python), [str(python), __file__], reexec_env)
        checkpoint = work / "policy_snapshot"
        download_code = (
            "from huggingface_hub import snapshot_download;"
            "snapshot_download(repo_id='" + MODEL_ID + "', revision='" + MODEL_REVISION
            + "', local_dir=r'" + str(checkpoint) + "')"
        )
        run([str(python), "-c", download_code], cwd=source, timeout=1800)
        port = free_port()
        server_env = os.environ.copy()
        server_env["PYTHONPATH"] = str(runtime) + os.pathsep + str(source)
        server_log = (output / "policy_server.log").open("w", encoding="utf-8")
        server_started = time.time()
        process = subprocess.Popen(
            [
                str(python), "-m", "blueprint_pipeline.cosmos_edge_droid_policy_server",
                "--checkpoint-path", str(checkpoint),
                "--snapshot-manifest", str(runtime / "policy_snapshot_manifest.json"),
                "--host", "127.0.0.1", "--port", str(port),
                "--output-dir", str(output / "policy_server"),
            ],
            cwd=source, env=server_env, stdout=server_log, stderr=subprocess.STDOUT, text=True,
        )
        startup = output / "policy_server/policy_server_startup.json"
        deadline = time.time() + 2400
        while time.time() < deadline and not startup.is_file():
            if process.poll() is not None:
                raise RuntimeError("policy_server_exited_before_startup")
            time.sleep(5)
        if not startup.is_file():
            raise RuntimeError("policy_server_startup_timeout")
        policy_server_load_seconds = time.time() - server_started
        startup_payload = json.loads(startup.read_text(encoding="utf-8"))
        if startup_payload.get("nvidia_guardrails_enabled") is not False:
            raise RuntimeError("policy_server_action_only_guardrail_mode_not_proven")

        sys.path.insert(0, str(runtime))
        from blueprint_pipeline.cosmos_edge_droid_policy_runtime import (
            CosmosEdgeDroidPolicyClient, CosmosEdgeDroidPolicySpec,
        )
        manifest = json.loads((runtime / "policy_snapshot_manifest.json").read_text())
        spec = CosmosEdgeDroidPolicySpec(snapshot_manifest_sha256=manifest["manifest_sha256"])
        client_deadline = time.time() + 120
        client_error = None
        while time.time() < client_deadline:
            try:
                client = CosmosEdgeDroidPolicyClient(spec=spec, host="127.0.0.1", port=port)
                break
            except (ConnectionError, OSError, TimeoutError) as exc:
                client_error = exc
                if process.poll() is not None:
                    raise RuntimeError("policy_server_exited_before_client_ready") from exc
                time.sleep(2)
        else:
            raise RuntimeError("policy_server_client_readiness_timeout") from client_error
        input_payload = json.loads((runtime / "policy_canary/input.json").read_text())
        observation = {
            key: np.asarray(Image.open(runtime / value).convert("RGB"), dtype=np.uint8)
            for key, value in input_payload["views"].items()
        }
        observation.update({
            "observation/joint_position": np.asarray(input_payload["joint_position"], dtype=np.float64),
            "observation/gripper_position": np.asarray(input_payload["gripper_position"], dtype=np.float64),
            "prompt": input_payload["prompt"],
        })
        request_started = time.time()
        response = client.infer(observation)
        latency = time.time() - request_started
        gpu_memory_after_inference = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        canary = {
            "status": "passed",
            "native_action": response["native_action"].tolist(),
            "wam_prefix_action": response["action"].tolist(),
            "executed_action": response["executed_action"].tolist(),
            "commanded_next_joint_position": response["commanded_next_joint_position"].tolist(),
            "commanded_next_gripper_position": response["commanded_next_gripper_position"].tolist(),
            "policy_request_receipt": response["policy_request_receipt"],
            "policy_endpoint_evidence": client.evidence_summary(),
            "policy_server_load_seconds": policy_server_load_seconds,
            "nvidia_guardrails_enabled": False,
            "blueprint_action_and_abstention_gates_remain_enabled": True,
            "latency_seconds": latency,
            "gpu_memory_after_inference_mb": gpu_memory_after_inference,
        }
        write(output / "policy_structured_canary.json", canary)
        result = {
            "schema_version": "cosmos_edge_closed_loop_runtime.v1",
            "status": "completed",
            "runtime": "policy_structured_canary",
            "gpu_probe": gpu_probe,
            "cosmos_framework_revision": resolved,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "native_action_shape": [32, 8],
            "wam_prefix_action_shape": [16, 8],
            "executed_prefix_steps": 8,
            "commanded_state_advance_proven": True,
            "policy_server_load_seconds": policy_server_load_seconds,
            "nvidia_guardrails_enabled": False,
            "blueprint_action_and_abstention_gates_remain_enabled": True,
            "policy_inference_latency_seconds": latency,
            "gpu_memory_after_inference_mb": gpu_memory_after_inference,
            "structured_policy_canary_passed": True,
            "registered_next_wam": REGISTERED_NEXT_WAM,
            "action_conditioned_video_rollout_generated": False,
            "learned_wam_model_ran": False,
            "duration_seconds": time.time() - started,
            "blockers": [],
            "claim_boundary": "Policy transport and action contract only; no WAM, ranking, physical, or transfer credit.",
            "raw_credentials_written_to_artifacts": False,
        }
        write(result_path, result)
        return 0
    except Exception as exc:
        blockers.append(f"{type(exc).__name__}:{str(exc)[:500]}")
        write(result_path, {
            "schema_version": "cosmos_edge_closed_loop_runtime.v1",
            "status": "blocked",
            "runtime": "policy_structured_canary",
            "registered_next_wam": REGISTERED_NEXT_WAM,
            "action_conditioned_video_rollout_generated": False,
            "learned_wam_model_ran": False,
            "duration_seconds": time.time() - started,
            "blockers": blockers,
            "raw_credentials_written_to_artifacts": False,
        })
        return 2
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
"""
)


REMOTE_ENTRYPOINT = textwrap.dedent(
    r"""#!/usr/bin/env bash
set -uo pipefail
OUTPUT_DIR="${BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR:-runtime_output}"
mkdir -p "$OUTPUT_DIR"

write_missing_result() {
  local runner_rc="${1:-999}"
  python - "$OUTPUT_DIR/wam_runtime_result.json" "$runner_rc" <<'PY'
import json, sys
from pathlib import Path
path=Path(sys.argv[1]); rc=int(sys.argv[2])
if not path.is_file():
    path.write_text(json.dumps({
      "schema_version":"cosmos_edge_closed_loop_runtime.v1",
      "status":"blocked",
      "blockers":["wam_runner_process_exited_without_runtime_result","blocked_wam_process_exited_without_result"],
      "runner_returncode":rc,
      "action_conditioned_video_rollout_generated":False,
      "raw_credentials_written_to_artifacts":False,
    },indent=2,sort_keys=True)+"\n")
PY
}

PYTHON_BIN="${BLUEPRINT_WAM_PROVIDER_PYTHON:-python3}"
"$PYTHON_BIN" "$(dirname "$0")/wam_provider_runtime_runner.py"
runner_rc=$?
write_missing_result "$runner_rc"
exit "$runner_rc"
"""
)


def build_cosmos_edge_policy_canary_bundle(
    *,
    output_dir: str | Path,
    policy_snapshot_manifest_path: str | Path,
    view_first_frames: Mapping[str, str | Path],
    joint_position: Any,
    gripper_position: Any,
    prompt: str,
    oscar_fixture_first_frame: str | Path,
    oscar_fixture_skeleton: str | Path,
    source_first_frame_sha256_by_view: Mapping[str, str] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic provider bundle for one diagnostic policy canary."""

    out = Path(output_dir).expanduser().resolve()
    ensure_dir(out)
    root = out / "cosmos_edge_policy_canary_bundle"
    if root.exists():
        shutil.rmtree(root)
    runtime = root / "provider_runtime"
    canary_dir = runtime / "policy_canary"
    package_dir = runtime / "blueprint_pipeline"
    core_package_dir = package_dir / "core"
    oscar_dir = runtime / "oscar_input"
    for path in (canary_dir, package_dir, core_package_dir, oscar_dir):
        ensure_dir(path)
    source_root = Path(__file__).resolve().parent
    for name in (
        "__init__.py",
        "common.py",
        "policy_ranking_thesis.py",
        "droid_policy_bridge.py",
        "cosmos_edge_droid_policy_runtime.py",
        "cosmos_edge_droid_policy_server.py",
    ):
        shutil.copy2(source_root / name, package_dir / name)
    for name in ("__init__.py", "common.py"):
        shutil.copy2(source_root / "core" / name, core_package_dir / name)
    snapshot_manifest = Path(policy_snapshot_manifest_path).expanduser().resolve()
    shutil.copy2(snapshot_manifest, runtime / "policy_snapshot_manifest.json")
    required_views = {
        "observation/wrist_image_left",
        "observation/exterior_image_1_left",
        "observation/exterior_image_2_left",
    }
    if set(view_first_frames) != required_views:
        raise ValueError("policy_canary_required_views_mismatch")
    view_entries: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    for index, key in enumerate(sorted(view_first_frames)):
        source = Path(view_first_frames[key]).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"policy_canary_view_first_frame_missing:{key}")
        relative = f"policy_canary/view_{index}.png"
        with Image.open(source) as image:
            image.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS).save(
                runtime / relative
            )
        view_entries[key] = relative
        source_hashes[key] = (
            str(source_first_frame_sha256_by_view[key])
            if source_first_frame_sha256_by_view is not None
            else file_sha256(source)
        )
    if source_first_frame_sha256_by_view is not None:
        if set(source_first_frame_sha256_by_view) != required_views:
            raise ValueError("policy_canary_source_hash_views_mismatch")
        if any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in source_hashes.values()
        ):
            raise ValueError("policy_canary_source_hash_invalid")
    joints = np.asarray(joint_position, dtype=np.float64)
    gripper = np.asarray(gripper_position, dtype=np.float64)
    if joints.shape != (7,) or not np.isfinite(joints).all():
        raise ValueError("policy_canary_joint_position_invalid")
    if gripper.shape != (1,) or not np.isfinite(gripper).all():
        raise ValueError("policy_canary_gripper_position_invalid")
    if not prompt.strip():
        raise ValueError("policy_canary_prompt_missing")
    canary = {
        "schema_version": "cosmos_edge_policy_structured_canary_input.v1",
        "views": view_entries,
        "source_first_frame_sha256_by_view": source_hashes,
        "joint_position": joints.tolist(),
        "gripper_position": gripper.tolist(),
        "prompt": prompt,
        "policy_identity_hidden_from_policy": False,
        "physical_outcome_fields_included": False,
        "diagnostic_only": True,
    }
    write_json(canary_dir / "input.json", canary)
    shutil.copy2(oscar_fixture_first_frame, oscar_dir / "first_frame.png")
    shutil.copy2(oscar_fixture_skeleton, oscar_dir / "blueprint_proxy_skeleton_conditioning.mp4")
    (runtime / "wam_provider_runtime_runner.py").write_text(REMOTE_RUNNER, encoding="utf-8")
    (runtime / "run_wam_provider_runtime.sh").write_text(REMOTE_ENTRYPOINT, encoding="utf-8")
    (runtime / "run_wam_provider_runtime.sh").chmod(
        (runtime / "run_wam_provider_runtime.sh").stat().st_mode | stat.S_IXUSR
    )
    rollout = {
        "schema_version": "cosmos_edge_policy_canary_rollout_input.v1",
        "stage": "policy_structured_canary",
        "provider_called": False,
        "action_conditioned_video_requested": False,
    }
    write_json(runtime / "wam_rollout_input_manifest.json", rollout)
    manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "generated_at": generated_at or utc_now_iso(),
        "status": "ready_for_paid_admission",
        "public_image": PUBLIC_IMAGE,
        "cosmos_framework": {"url": FRAMEWORK_URL, "revision": FRAMEWORK_REVISION},
        "policy": {"model_id": MODEL_ID, "revision": MODEL_REVISION},
        "native_action_shape": [32, 8],
        "wam_prefix_action_shape": [16, 8],
        "qualification_canary_request_count": 1,
        "scientific_matrix_request_count": 0,
        "total_initial_generation_request_count": 1,
        "registered_next_wam": "OSCAR-2B",
        "nvidia_guardrails_enabled": False,
        "guardrail_mode_scope": "action_only_policy_endpoint",
        "blueprint_action_and_abstention_gates_remain_enabled": True,
        "action_conditioned_video_rollout_generated": False,
        "physical_outcomes_in_bundle": False,
        "raw_credentials_in_bundle": False,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    write_json(runtime / "wam_provider_runtime_manifest.json", manifest)
    bundle_path = out / "cosmos_edge_policy_canary_provider_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root).as_posix())
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "bundle_path": str(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
        "bundle_sha256": file_sha256(bundle_path),
        "runtime_manifest_sha256": manifest["manifest_sha256"],
        "canary_input_sha256": canonical_sha256(canary),
        "runner_sha256": file_sha256(runtime / "wam_provider_runtime_runner.py"),
        "entrypoint_sha256": file_sha256(runtime / "run_wam_provider_runtime.sh"),
        "paid_execution_admitted": False,
        "provider_called": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    write_json(out / "cosmos_edge_policy_canary_bundle_receipt.json", receipt)
    write_json(out / "oscar_wam_provider_bundle_manifest.json", manifest)
    return receipt


__all__ = ["build_cosmos_edge_policy_canary_bundle"]
