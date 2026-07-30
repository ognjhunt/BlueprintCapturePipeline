"""Build a deterministic provider bundle for a reduced public Ctrl-World replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .policy_ranking_successor_cosmos import canonical_sha256


EXPERIMENT_ID = "policy_ranking_cosmos3_edge_closed_loop_20260729"
CTRL_WORLD_SOURCE_REPOSITORY = "https://github.com/Robert-gyj/Ctrl-World"
CTRL_WORLD_SOURCE_REVISION = "99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d"
CTRL_WORLD_CHECKPOINT_REPOSITORY = "yjguo/Ctrl-World"
CTRL_WORLD_CHECKPOINT_REVISION = "8cf814693f411962dc866a2ddb5b785afd17a93a"
SVD_REPOSITORY = "stabilityai/stable-video-diffusion-img2vid"
SVD_REVISION = "9cf024d5bfa8f56622af86c884f26a52f6676f2e"
CLIP_REPOSITORY = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
CTRL_WORLD_PUBLIC_IMAGE = (
    "docker.io/pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime@sha256:"
    "c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2"
)
CTRL_WORLD_TORCH_VERSION = "2.7.1+cu128"
CTRL_WORLD_PROVIDER_BUNDLE_SCHEMA_VERSION = "ctrl_world_provider_bundle_manifest.v1"
CTRL_WORLD_RECEIPT_SCHEMA_VERSION = "ctrl_world_replay_bundle_receipt.v1"
DEFAULT_BUNDLE_FILENAME = "ctrl_world_replay_provider_runtime_bundle.zip"

SOURCE_FILES = (
    "LICENSE.txt",
    "readme.md",
    "requirements.txt",
    "config.py",
    "models/__init__.py",
    "models/ctrl_world.py",
    "models/pipeline_ctrl_world.py",
    "models/pipeline_stable_video_diffusion.py",
    "models/unet_spatio_temporal_condition.py",
    "models/utils.py",
    "scripts/rollout_replay_traj.py",
    "dataset_meta_info/droid/stat.json",
    "dataset_example/droid_subset/annotation/val/899.json",
    "dataset_example/droid_subset/videos/val/899/0.mp4",
    "dataset_example/droid_subset/videos/val/899/1.mp4",
    "dataset_example/droid_subset/videos/val/899/2.mp4",
)

PYTHON_DEPENDENCIES = (
    "accelerate==1.8.1",
    "annotated-types==0.8.0",
    "asttokens==3.0.2",
    "boto3==1.43.59",
    "botocore==1.43.59",
    "certifi==2026.7.22",
    "charset-normalizer==3.4.9",
    "click==8.4.2",
    "contourpy==1.3.3",
    "cycler==0.12.1",
    "decorator==5.3.1",
    "decord==0.6.0",
    "diffusers==0.34.0",
    "docker-pycreds==0.4.0",
    "einops==0.8.1",
    "executing==2.2.1",
    "filelock==3.32.2",
    "fonttools==4.63.0",
    "fsspec==2026.7.0",
    "gitdb==4.0.12",
    "gitpython==3.1.57",
    "hf-xet==1.5.2",
    "huggingface-hub==0.34.4",
    "idna==3.18",
    "imageio-ffmpeg==0.6.0",
    "importlib-metadata==9.0.0",
    "ipython==9.15.0",
    "ipython-pygments-lexers==1.1.1",
    "jedi==0.20.0",
    "jinja2==3.1.6",
    "jmespath==1.1.0",
    "kiwisolver==1.5.0",
    "markdown-it-py==4.2.0",
    "markupsafe==3.0.3",
    "matplotlib==3.11.1",
    "matplotlib-inline==0.2.2",
    "mdurl==0.1.2",
    "mediapy==1.2.4",
    "mpmath==1.3.0",
    "networkx==3.6.1",
    "numpy==1.26.4",
    "nvidia-ml-py==13.610.43",
    "opencv-python-headless==4.10.0.84",
    "packaging==26.2",
    "pandas==2.2.3",
    "parso==0.8.7",
    "pexpect==4.9.0",
    "pillow==11.1.0",
    "platformdirs==4.11.0",
    "prettytable==3.18.0",
    "prompt-toolkit==3.0.53",
    "protobuf==5.29.4",
    "psutil==7.2.2",
    "ptyprocess==0.7.0",
    "pure-eval==0.2.3",
    "pydantic==2.13.4",
    "pydantic-core==2.46.4",
    "pyecharts==2.1.0",
    "pygments==2.20.0",
    "pyparsing==3.3.2",
    "python-dateutil==2.9.0.post0",
    "pytz==2026.3.post1",
    "pyyaml==6.0.3",
    "regex==2026.7.19",
    "requests==2.34.2",
    "rich==13.9.4",
    "s3transfer==0.19.2",
    "safetensors==0.8.0",
    "scipy==1.15.3",
    "sentencepiece==0.2.0",
    "sentry-sdk==2.66.1",
    "setproctitle==1.3.7",
    "setuptools==83.0.0",
    "simplejson==4.1.1",
    "six==1.17.0",
    "smmap==5.0.3",
    "stack-data==0.6.3",
    "swanlab==0.6.13",
    "sympy==1.14.0",
    "tokenizers==0.21.4",
    "tqdm==4.70.0",
    "traitlets==5.15.1",
    "transformers==4.48.1",
    "typing-extensions==4.16.0",
    "typing-inspection==0.4.2",
    "tzdata==2026.3",
    "urllib3==2.7.0",
    "wandb==0.19.11",
    "wcwidth==0.8.2",
    "wrapt==2.3.0",
    "zipp==4.1.0",
)

MODEL_FREEZE = (
    {
        "name": "ctrl_world",
        "repository": CTRL_WORLD_CHECKPOINT_REPOSITORY,
        "revision": CTRL_WORLD_CHECKPOINT_REVISION,
        "allow_patterns": ["checkpoint-10000.pt"],
        "required_blobs": [
            {
                "relative_path": "checkpoint-10000.pt",
                "size_bytes": 9_281_040_326,
                "sha256": "ed17de48180d4e6f89fd33c53e9fb7a0196189c1a67d44c2c486a279a80ea8a8",
            }
        ],
    },
    {
        "name": "stable_video_diffusion",
        "repository": SVD_REPOSITORY,
        "revision": SVD_REVISION,
        "allow_patterns": [
            "model_index.json",
            "feature_extractor/preprocessor_config.json",
            "image_encoder/config.json",
            "image_encoder/model.safetensors",
            "scheduler/scheduler_config.json",
            "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ],
        "required_blobs": [
            {
                "relative_path": "image_encoder/model.safetensors",
                "size_bytes": 2_528_371_296,
                "sha256": "ed1e5af7b4042ca30ec29999a4a5cfcac90b7fb610fd05ace834f2dcbb763eab",
            },
            {
                "relative_path": "unet/diffusion_pytorch_model.safetensors",
                "size_bytes": 6_098_682_464,
                "sha256": "98c5e6b99df6bef015b2681c0f8ab9d4c807b733be46c067d6c9966101698f58",
            },
            {
                "relative_path": "vae/diffusion_pytorch_model.safetensors",
                "size_bytes": 391_017_740,
                "sha256": "9975042d7bee021bd53a72b1af14c8627d624f6547ec9abe661b68b962b88c49",
            },
        ],
    },
    {
        "name": "clip",
        "repository": CLIP_REPOSITORY,
        "revision": CLIP_REVISION,
        "allow_patterns": [
            "config.json",
            "merges.txt",
            "preprocessor_config.json",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "vocab.json",
        ],
        "required_blobs": [
            {
                "relative_path": "pytorch_model.bin",
                "size_bytes": 605_247_071,
                "sha256": "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f",
            }
        ],
    },
)

REMOTE_ENTRYPOINT = """#!/usr/bin/env bash
set -uo pipefail
write_missing_result() {
  local rc="$1"
  local output_dir="${BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR:-runtime_output}"
  mkdir -p "$output_dir"
  python - "$output_dir/wam_runtime_result.json" "$rc" <<'PY'
import json
import pathlib
import sys
result_path = pathlib.Path(sys.argv[1])
rc = int(sys.argv[2])
payload = {
    "schema_version": "ctrl_world_replay_runtime_result.v1",
    "status": "blocked",
    "model_name": "Ctrl-World",
    "action_conditioned_video_rollout_generated": False,
    "blockers": [
        "wam_runner_process_exited_without_runtime_result",
        "blocked_wam_process_exited_without_result",
    ],
    "runner_exit_code": rc,
    "raw_credentials_written_to_artifacts": False,
    "secret_hashes_written_to_artifacts": False,
}
result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}
PYTHON_BIN="${BLUEPRINT_WAM_PROVIDER_PYTHON:-python}"
mkdir -p "${BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR:-runtime_output}"
RUNNER_LOG="${BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR:-runtime_output}/wam_provider_runner.log"
"$PYTHON_BIN" "$(dirname "$0")/wam_provider_runtime_runner.py" 2>&1 | tee "$RUNNER_LOG"
rc=${PIPESTATUS[0]}
if [ ! -f "${BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR:-runtime_output}/wam_runtime_result.json" ]; then
  write_missing_result "$rc"
fi
exit "$rc"
"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit(source_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _source_status(source_root: Path) -> str:
    completed = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=source_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unreadable"


def _write_deterministic_zip(*, source_root: Path, bundle_path: Path) -> list[str]:
    entries: list[str] = []
    with zipfile.ZipFile(
        bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(source_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source_root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 7, 30, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if path.stat().st_mode & stat.S_IXUSR else 0o644) << 16
            archive.writestr(
                info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9
            )
            entries.append(relative)
    return entries


def build_ctrl_world_provider_bundle(
    *,
    job_dir: str | Path,
    ctrl_world_source_dir: str | Path,
    bundle_filename: str = DEFAULT_BUNDLE_FILENAME,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    resolved_job_dir = Path(job_dir).expanduser().resolve()
    source_root = Path(ctrl_world_source_dir).expanduser().resolve()
    ensure_dir(resolved_job_dir)
    bundle_root = resolved_job_dir / "ctrl_world_provider_bundle"
    runtime_dir = bundle_root / "provider_runtime"
    packaged_source = runtime_dir / "ctrl_world_source"
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    ensure_dir(packaged_source)
    blockers: list[str] = []
    observed_commit = _source_commit(source_root)
    observed_status = _source_status(source_root)
    if observed_commit != CTRL_WORLD_SOURCE_REVISION:
        blockers.append("ctrl_world_source_revision_mismatch")
    if observed_status:
        blockers.append("ctrl_world_source_worktree_not_clean")
    source_manifest: list[dict[str, Any]] = []
    for relative in SOURCE_FILES:
        source = source_root / relative
        destination = packaged_source / relative
        if not source.is_file():
            blockers.append(f"ctrl_world_source_file_missing:{relative}")
            continue
        ensure_dir(destination.parent)
        shutil.copy2(source, destination)
        source_manifest.append(
            {
                "relative_path": relative,
                "size_bytes": source.stat().st_size,
                "sha256": _sha256_file(source),
            }
        )

    input_root = runtime_dir / "ctrl_world_replay"
    ensure_dir(input_root)
    annotation_source = source_root / "dataset_example/droid_subset/annotation/val/899.json"
    shutil.copy2(annotation_source, input_root / "annotation.json")
    view_manifest: list[dict[str, Any]] = []
    for index in range(3):
        source = source_root / f"dataset_example/droid_subset/videos/val/899/{index}.mp4"
        destination = input_root / f"view_{index}.mp4"
        shutil.copy2(source, destination)
        view_manifest.append(
            {
                "view_index": index,
                "relative_path": f"provider_runtime/ctrl_world_replay/view_{index}.mp4",
                "size_bytes": source.stat().st_size,
                "sha256": _sha256_file(source),
            }
        )
    canary_manifest = {
        "schema_version": "ctrl_world_public_replay_canary.v1",
        "arm_id": "ctrl_world_public_replay_reduced_canary",
        "trajectory_id": "899",
        "start_index": 8,
        "interaction_count": 1,
        "public_default_interaction_count": 12,
        "views": view_manifest,
        "annotation_sha256": _sha256_file(annotation_source),
        "task_instruction_source": "released_annotation_texts_0",
        "action_source": "released_recorded_DROID_cartesian_state_trajectory",
        "physical_future_rgb_used_as_generation_condition": False,
        "physical_future_rgb_decoded_for_public_comparison": True,
        "physical_outcome_labels_accessed": False,
        "closed_loop_policy_evaluation": False,
        "claim_ceiling": "label_free_open_loop_replay_technical_canary",
    }
    canary_manifest["manifest_sha256"] = canonical_sha256(canary_manifest)
    write_json(input_root / "canary_manifest.json", canary_manifest)

    runner_source = Path(__file__).with_name("ctrl_world_provider_runtime_runner.py")
    runner_destination = runtime_dir / "wam_provider_runtime_runner.py"
    shutil.copy2(runner_source, runner_destination)
    runner_destination.chmod(runner_destination.stat().st_mode | stat.S_IXUSR)
    entrypoint = runtime_dir / "run_wam_provider_runtime.sh"
    entrypoint.write_text(REMOTE_ENTRYPOINT, encoding="utf-8")
    entrypoint.chmod(entrypoint.stat().st_mode | stat.S_IXUSR)
    retained_control_source = Path(__file__).with_name(
        "policy_ranking_successor_retained_remote.py"
    )
    if retained_control_source.is_file():
        shutil.copy2(retained_control_source, runtime_dir / "successor_retained_control.py")
    else:
        blockers.append("successor_retained_control_source_missing")

    runtime_manifest = {
        "schema_version": "wam_provider_runtime_manifest.v1",
        "runtime": "ctrl_world_public_replay_runtime",
        "experiment_id": EXPERIMENT_ID,
        "model_candidate": "ctrl_world",
        "model_name": "Ctrl-World",
        "public_image": CTRL_WORLD_PUBLIC_IMAGE,
        "ctrl_world_source_repository": CTRL_WORLD_SOURCE_REPOSITORY,
        "ctrl_world_source_revision": CTRL_WORLD_SOURCE_REVISION,
        "checkpoint_repository": CTRL_WORLD_CHECKPOINT_REPOSITORY,
        "checkpoint_revision": CTRL_WORLD_CHECKPOINT_REVISION,
        "torch_version": CTRL_WORLD_TORCH_VERSION,
        "python_dependencies": list(PYTHON_DEPENDENCIES),
        "models": list(MODEL_FREEZE),
        "source_files": source_manifest,
        "canary_settings": {
            "trajectory_id": "899",
            "start_index": 8,
            "interaction_count": 1,
            "timeout_seconds": 3600,
            "public_script": "scripts/rollout_replay_traj.py",
            "public_script_sha256": _sha256_file(source_root / "scripts/rollout_replay_traj.py"),
        },
        "qualification_canary_request_count": 1,
        "scientific_matrix_request_count": 0,
        "total_initial_generation_request_count": 1,
        "truth_boundary": {
            "exact_public_script_bytes_packaged": True,
            "blueprint_wrapper_reduces_scope_only": True,
            "generated_only_outputs_required": True,
            "physical_comparison_pixels_forbidden_from_provider_output": True,
            "technical_canary_not_ranking_or_thesis_evidence": True,
        },
    }
    write_json(runtime_dir / "wam_provider_runtime_manifest.json", runtime_manifest)
    rollout_manifest = {
        "schema_version": "ctrl_world_public_replay_rollout_input.v1",
        "experiment_id": EXPERIMENT_ID,
        "arm_id": "ctrl_world_public_replay_reduced_canary",
        "canary_manifest_path": "provider_runtime/ctrl_world_replay/canary_manifest.json",
        "physical_outcome_labels_accessed": False,
        "physical_future_rgb_provided_to_model": False,
        "candidate_policy_requeried": False,
        "closed_loop": False,
    }
    write_json(runtime_dir / "wam_rollout_input_manifest.json", rollout_manifest)

    bundle_path = resolved_job_dir / bundle_filename
    zip_entries: list[str] = []
    if not blockers:
        zip_entries = _write_deterministic_zip(source_root=bundle_root, bundle_path=bundle_path)
        with zipfile.ZipFile(bundle_path) as archive:
            if archive.testzip() is not None:
                blockers.append("ctrl_world_provider_bundle_zip_invalid")

    embedded_hashes = {
        "runtime_manifest_file_sha256": _sha256_file(
            runtime_dir / "wam_provider_runtime_manifest.json"
        ),
        "rollout_manifest_file_sha256": _sha256_file(
            runtime_dir / "wam_rollout_input_manifest.json"
        ),
        "canary_manifest_sha256": canonical_sha256(
            {key: value for key, value in canary_manifest.items() if key != "manifest_sha256"}
        ),
        "annotation_sha256": _sha256_file(input_root / "annotation.json"),
        "view_manifest_sha256": canonical_sha256(view_manifest),
        "source_manifest_sha256": canonical_sha256(source_manifest),
        "runner_sha256": _sha256_file(runner_destination),
        "entrypoint_sha256": _sha256_file(entrypoint),
    }
    bundle_sha256 = _sha256_file(bundle_path) if bundle_path.is_file() else None
    bundle_size_bytes = bundle_path.stat().st_size if bundle_path.is_file() else 0
    receipt = {
        "schema_version": CTRL_WORLD_RECEIPT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "bundle_sha256": bundle_sha256,
        "bundle_size_bytes": bundle_size_bytes,
        **embedded_hashes,
    }
    write_json(resolved_job_dir / "ctrl_world_replay_bundle_receipt.json", receipt)
    manifest = {
        "schema_version": CTRL_WORLD_PROVIDER_BUNDLE_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "completed" if not blockers else "blocked",
        "experiment_id": EXPERIMENT_ID,
        "provider_bundle_kind": "wam",
        "bundle_path": str(bundle_path),
        "bundle_present": bundle_path.is_file(),
        "bundle_sha256": bundle_sha256,
        "bundle_size_bytes": bundle_size_bytes,
        "local_bundle_ready_for_remote_staging": not blockers,
        "zip_entry_count": len(zip_entries),
        "zip_entries": zip_entries,
        "source_revision": observed_commit,
        "source_worktree_clean": not observed_status,
        "embedded_hashes": embedded_hashes,
        "receipt_path": str(resolved_job_dir / "ctrl_world_replay_bundle_receipt.json"),
        "blockers": blockers,
        "compatibility_readiness_filename_only": "oscar_wam_provider_bundle_manifest.json",
        "attribution": "Ctrl-World_not_OSCAR_not_Cosmos",
        "truth_boundary": {
            "bundle_build_is_not_model_execution": True,
            "public_replay_is_not_closed_loop_policy_evaluation": True,
            "generated_or_replayed_output_is_not_physical_success": True,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(resolved_job_dir / "ctrl_world_provider_bundle_manifest.json", manifest)
    write_json(resolved_job_dir / "oscar_wam_provider_bundle_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--ctrl-world-source-dir", required=True)
    parser.add_argument("--bundle-filename", default=DEFAULT_BUNDLE_FILENAME)
    args = parser.parse_args(argv)
    result = build_ctrl_world_provider_bundle(
        job_dir=args.job_dir,
        ctrl_world_source_dir=args.ctrl_world_source_dir,
        bundle_filename=args.bundle_filename,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
