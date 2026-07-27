"""Build the immutable Cosmos3 successor provider bundle from public DROID data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq
from PIL import Image

from .common import write_json
from .policy_ranking_successor_cosmos import (
    CHECKPOINT_REVISION,
    EXPERIMENT_ID,
    VLLM_IMAGE,
    VLLM_IMAGE_DIGEST,
    build_action_controls,
    canonical_sha256,
    convert_droid_states_to_action_stream,
    droid_action_stream,
    validate_smoke_inventory_manifest,
)
from .policy_ranking_successor_gpu_admission import BUNDLE_SCHEMA, PUBLIC_IMAGE


SAMPLE_FILE_HASHES = {
    "data/chunk-000/file-000.parquet": "56e3defd9e75a101a7b812ad7ae263dde8ec5699b6b47db51ba6954f81be2593",
    "meta/info.json": "64e39f3dbedbd5ffade567007093d92d1827fcdb68c5d9d573ff3e80eef23cf6",
    "meta/episodes/chunk-000/file-000.parquet": "23f8e342bec24cf1af5fc4c5f58d8b51097ce09c2aaaafe6d467328af6dc016e",
    "meta/tasks.parquet": "ee2c7ec4f086cf2025b9a1d169bf6ba1857fed2579604c8859b49741f5eb29d6",
    "videos/observation.image.wrist_image_left/chunk-000/file-000.mp4": "e9d641240f9efc344924c31715a44be1f66207f667dd63c292bbd943dfa816d0",
    "videos/observation.image.exterior_image_1_left/chunk-000/file-000.mp4": "6aac49551ff9b9fc7b8e9899df4fc050f5cd31164ed4ee3c014f332bcef3b9f1",
    "videos/observation.image.exterior_image_2_left/chunk-000/file-000.mp4": "34f6e5dae8591809324f1df9bf166c6e304bd2cbdad643595a991f641cbb4fd7",
}
EXPECTED_INITIAL_OBSERVATION_SHA256 = (
    "8843f0fc9c68914dfb62222c961db19b37a5f155e602ff4a545eea1dcf42636d"
)
SHUFFLE_SEED = 20260727

RUN_SCRIPT = """#!/usr/bin/env bash
set -u
RUNTIME_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT_DIR=${BLUEPRINT_VAST_PROVIDER_OUTPUT_DIR:-"$RUNTIME_DIR/../runtime_output"}
mkdir -p "$OUTPUT_DIR"
export BLUEPRINT_VAST_PROVIDER_OUTPUT_DIR="$OUTPUT_DIR"
python "$RUNTIME_DIR/wam_provider_runtime_runner.py"
runner_rc=$?
write_missing_result() {
  python - "$OUTPUT_DIR/wam_runtime_result.json" "$runner_rc" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "schema_version": "policy_ranking_successor_cosmos_runtime.v1",
    "status": "blocked",
    "failure_class": "infrastructure_failure",
    "blockers": ["wam_runner_process_exited_without_runtime_result"],
    "runner_exit_code": int(sys.argv[2]),
    "blocked_wam_process_exited_without_result": True,
    "action_conditioned_video_rollout_generated": False,
    "evaluator_eligible": False,
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
}
if [ ! -f "$OUTPUT_DIR/wam_runtime_result.json" ]; then
  write_missing_result
fi
exit "$runner_rc"
"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sample(sample_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in SAMPLE_FILE_HASHES.items():
        path = sample_root / relative
        if not path.is_file():
            raise ValueError(f"public_droid_sample_file_missing:{relative}")
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(f"public_droid_sample_hash_mismatch:{relative}")
        observed[relative] = actual
    return observed


def _extract_first_frame(video: Path, output: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("ffmpeg_missing")
    completed = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(output),
        ],
        check=False,
        capture_output=True,
        timeout=120,
    )
    if completed.returncode != 0 or not output.is_file():
        raise ValueError("droid_first_frame_decode_failed")


def _compose_initial_observation(sample_root: Path, output: Path) -> str:
    keys = (
        "observation.image.wrist_image_left",
        "observation.image.exterior_image_1_left",
        "observation.image.exterior_image_2_left",
    )
    with tempfile.TemporaryDirectory(prefix="cosmos3-droid-frames-") as temporary:
        frames: list[Image.Image] = []
        for index, key in enumerate(keys):
            frame_path = Path(temporary) / f"{index}.png"
            video = sample_root / "videos" / key / "chunk-000" / "file-000.mp4"
            _extract_first_frame(video, frame_path)
            frames.append(Image.open(frame_path).convert("RGB"))
        if any(frame.size != (640, 360) for frame in frames):
            raise ValueError("droid_decoded_camera_resolution_invalid")
        composed = Image.new("RGB", (640, 540))
        composed.paste(frames[0], (0, 0))
        composed.paste(frames[1].resize((320, 180), Image.Resampling.BILINEAR), (0, 360))
        composed.paste(frames[2].resize((320, 180), Image.Resampling.BILINEAR), (320, 360))
        composed.save(output, format="PNG", optimize=False)
        for frame in frames:
            frame.close()
    actual = _sha256_file(output)
    if actual != EXPECTED_INITIAL_OBSERVATION_SHA256:
        raise ValueError("droid_initial_observation_hash_mismatch")
    return actual


def _build_action_streams(sample_root: Path) -> dict[str, Any]:
    table = pq.read_table(sample_root / "data/chunk-000/file-000.parquet")
    rows = table.slice(0, 17).to_pylist()
    states = [row["observation.state.cartesian_position"] for row in rows]
    source_gripper = [row["action.gripper_position"] for row in rows[:16]]
    recorded = convert_droid_states_to_action_stream(
        states,
        source_gripper,
        source_gripper_action_flipped=True,
    )
    swapped_actions = [
        [0.001, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0 if i < 8 else 0.0] for i in range(16)
    ]
    swapped = droid_action_stream(swapped_actions)
    conditions = build_action_controls(recorded, swapped, shuffle_seed=SHUFFLE_SEED)
    return {
        "schema_version": "policy_ranking_successor_action_streams.v1",
        "experiment_id": EXPERIMENT_ID,
        "conditions": conditions,
        "condition_order": list(conditions),
        "source_gripper_action_flipped": True,
        "source_codebase_version": "v3.0",
        "policy_identity_present": False,
    }


def _zip_write(
    archive: zipfile.ZipFile, name: str, data: bytes, *, executable: bool = False
) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = ((0o755 if executable else 0o644) & 0xFFFF) << 16
    archive.writestr(info, data)


def build_successor_cosmos_bundle(
    *,
    sample_root: str | Path,
    smoke_inventory_path: str | Path,
    output_bundle: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(sample_root).expanduser().resolve()
    inventory_path = Path(smoke_inventory_path).expanduser().resolve()
    output = Path(output_bundle).expanduser().resolve()
    source_hashes = _verify_sample(root)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    validate_smoke_inventory_manifest(inventory)
    action_streams = _build_action_streams(root)
    conditions = action_streams["conditions"]
    for condition, expected in inventory["action_hashes"].items():
        if conditions[condition]["action_sha256"] != expected:
            raise ValueError(f"frozen_action_hash_mismatch:{condition}")
    runtime_source = (
        Path(__file__).with_name("policy_ranking_successor_cosmos_provider_runtime.py")
    ).read_bytes()
    with tempfile.TemporaryDirectory(prefix="cosmos3-successor-bundle-") as temporary:
        initial = Path(temporary) / "initial_observation.png"
        initial_hash = _compose_initial_observation(root, initial)
        if inventory["initial_observation_sha256"] != initial_hash:
            raise ValueError("frozen_initial_observation_hash_mismatch")
        action_payload = (
            json.dumps(action_streams, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
            + b"\n"
        )
        inventory_payload = (
            json.dumps(inventory, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
        )
        input_manifest = {
            "schema_version": "policy_ranking_successor_wam_rollout_input.v1",
            "experiment_id": EXPERIMENT_ID,
            "initial_observation_sha256": initial_hash,
            "action_streams_sha256": canonical_sha256(action_streams),
            "smoke_inventory_sha256": canonical_sha256(inventory),
            "request_count": 10,
            "scientific_matrix_request_count": 10,
            "policy_identity_present": False,
            "independent_outcomes_accessed": False,
            "public_calibration_data_only": True,
        }
        runtime_manifest = {
            "schema_version": BUNDLE_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "checkpoint": "nvidia/Cosmos3-Nano",
            "checkpoint_revision": CHECKPOINT_REVISION,
            "public_image": PUBLIC_IMAGE,
            "generic_preview_path_used": False,
            "forward_dynamics_endpoint": "/v1/videos/sync",
            "pipeline_class": "Cosmos3OmniDiffusersPipeline",
            "vllm_omni_source_revision": "9c1b7504b178afcf541867c1a2d30db48c69cda8",
            "raw_action_dim": 10,
            "action_space": "midtrain",
            "qualification_canary_request_count": 2,
            "scientific_matrix_request_count": 10,
            "total_initial_generation_request_count": 12,
            "request_budget_amendment_sha256": (
                "e67226e16318a073e9190915554dc37b1d378fc155c6eb6bec7ecc79fb27786a"
            ),
            "precision": "bf16",
            "trust_remote_code": False,
            "post_training_or_lora": False,
            "source_file_hashes": source_hashes,
            "initial_observation_sha256": initial_hash,
            "claims": {
                "implementation": True,
                "runtime": False,
                "generated_media": False,
                "wam_causal_validity": False,
                "evaluator_validity": False,
                "ranking_fidelity": False,
                "physical_performance": False,
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w") as archive:
            _zip_write(
                archive,
                "provider_runtime/wam_provider_runtime_runner.py",
                runtime_source,
                executable=True,
            )
            _zip_write(
                archive,
                "provider_runtime/run_wam_provider_runtime.sh",
                RUN_SCRIPT.encode("utf-8"),
                executable=True,
            )
            _zip_write(
                archive,
                "provider_runtime/wam_provider_runtime_manifest.json",
                json.dumps(runtime_manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n",
            )
            _zip_write(
                archive,
                "provider_runtime/wam_rollout_input_manifest.json",
                json.dumps(input_manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n",
            )
            _zip_write(
                archive,
                "provider_runtime/cosmos3_input/initial_observation.png",
                initial.read_bytes(),
            )
            _zip_write(
                archive,
                "provider_runtime/cosmos3_input/smoke_request_inventory.json",
                inventory_payload,
            )
            _zip_write(
                archive,
                "provider_runtime/cosmos3_input/action_streams.json",
                action_payload,
            )
    receipt = {
        "schema_version": "policy_ranking_successor_cosmos_bundle_receipt.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "built",
        "bundle_path": str(output),
        "bundle_sha256": _sha256_file(output),
        "bundle_size_bytes": output.stat().st_size,
        "public_image": f"{VLLM_IMAGE}@{VLLM_IMAGE_DIGEST}",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "initial_observation_sha256": EXPECTED_INITIAL_OBSERVATION_SHA256,
        "smoke_inventory_sha256": canonical_sha256(inventory),
        "action_streams_sha256": canonical_sha256(action_streams),
        "source_file_hashes": source_hashes,
        "provider_mutations_performed": 0,
        "paid_resources_used": False,
        "claims": {
            "implementation": True,
            "runtime": False,
            "generated_media": False,
            "wam_causal_validity": False,
            "evaluator_validity": False,
            "ranking_fidelity": False,
            "physical_performance": False,
        },
    }
    if receipt_path:
        write_json(Path(receipt_path), receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", required=True)
    parser.add_argument("--smoke-inventory", required=True)
    parser.add_argument("--output-bundle", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    result = build_successor_cosmos_bundle(
        sample_root=args.sample_root,
        smoke_inventory_path=args.smoke_inventory,
        output_bundle=args.output_bundle,
        receipt_path=args.receipt,
    )
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
