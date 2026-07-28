"""Build immutable Cosmos3-Nano Reasoner evaluator diagnostic bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import write_json
from .policy_ranking_evaluator_diagnostic import (
    COSMOS_MODEL,
    COSMOS_REVISION,
    PAIR_OUTPUT_SCHEMA,
    PAIR_PROMPT,
    diagnostic_protocol,
)
from .policy_ranking_roboarena_calibration import canonical_sha256, file_sha256


SCHEMA_VERSION = "policy_ranking_cosmos_reasoner_bundle.v1"
RECEIPT_SCHEMA_VERSION = "policy_ranking_cosmos_reasoner_bundle_receipt.v1"
RUN_PURPOSE = "structured_output_canary_only"
PUBLIC_IMAGE = (
    "docker.io/vllm/vllm-omni:cosmos3@"
    "sha256:6d2630c7d637b699557573f2c3fee8df5d4d0cd718977aa22549ed6a6ef30587"
)
NATIVE_REASONER_ARCHITECTURE = "Cosmos3ForConditionalGeneration"
MODEL_CONFIG_SHA256 = "c32f2468a54542c21946bc8eab6172b911dcec9a7193a94c023ea2d4073bcda6"

RUN_SCRIPT = """#!/usr/bin/env bash
set -u
RUNTIME_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT_DIR=${BLUEPRINT_EVALUATOR_PROVIDER_OUTPUT_DIR:-"$RUNTIME_DIR/../runtime_output"}
mkdir -p "$OUTPUT_DIR"
export BLUEPRINT_EVALUATOR_PROVIDER_OUTPUT_DIR="$OUTPUT_DIR"
export BLUEPRINT_EVALUATOR_PROVIDER_BUNDLE_DIR=${BLUEPRINT_EVALUATOR_PROVIDER_BUNDLE_DIR:-"$RUNTIME_DIR/.."}
export BLUEPRINT_EVALUATOR_INPUT=${BLUEPRINT_EVALUATOR_INPUT:-"$RUNTIME_DIR/evaluator_input_manifest.json"}
PYTHON_BIN=${BLUEPRINT_EVALUATOR_PROVIDER_PYTHON:-python3}
"$PYTHON_BIN" "$RUNTIME_DIR/evaluator_provider_runtime_runner.py"
runner_rc=$?
write_missing_result() {
  "$PYTHON_BIN" - "$OUTPUT_DIR/evaluator_runtime_result.json" "$runner_rc" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "schema_version": "policy_ranking_cosmos_reasoner_runtime.v1",
    "status": "blocked",
    "blockers": ["evaluator_runner_process_exited_without_runtime_result"],
    "runner_exit_code": int(sys.argv[2]),
    "blocked_evaluator_process_exited_without_result": True,
    "result_count": 0,
    "error_count": 1,
    "model": "nvidia/Cosmos3-Nano",
    "claim_class": "post_unseal_diagnostic_only",
    "action_conditioned_video_rollout_generated": False,
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
}
if [ ! -f "$OUTPUT_DIR/evaluator_runtime_result.json" ]; then
  write_missing_result
fi
exit "$runner_rc"
"""

PROVIDER_RUNTIME_RUNNER_SHA256 = hashlib.sha256(
    Path(__file__)
    .with_name("policy_ranking_evaluator_diagnostic_cosmos_provider_runtime.py")
    .read_bytes()
).hexdigest()
RUN_SCRIPT_SHA256 = hashlib.sha256(RUN_SCRIPT.encode()).hexdigest()


def _read_mapping(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected_json_object:{path}")
    return dict(value)


def _zip_write(
    archive: zipfile.ZipFile,
    name: str,
    data: bytes,
    *,
    executable: bool = False,
    stored: bool = False,
) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    info.external_attr = ((0o755 if executable else 0o644) & 0xFFFF) << 16
    archive.writestr(info, data)


def _manifest_rows(native: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if (
        native.get("status") != "passed"
        or native.get("video_count") != 441
        or native.get("all_physical_right_half_pixels_excluded") is not True
    ):
        raise ValueError("native_video_manifest_not_ready_441")
    rows = native.get("receipts")
    if not isinstance(rows, list) or len(rows) != 441:
        raise ValueError("native_video_rows_invalid")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("native_video_row_invalid")
        request_id = str(row.get("request_id") or "")
        path = Path(str(row.get("output_path") or "")).expanduser().resolve()
        expected = str(row.get("output_sha256") or "")
        if not request_id or not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"native_video_row_hash_invalid:{request_id}")
        indexed[request_id] = {**dict(row), "resolved_path": str(path), "sha256": expected}
    if len(indexed) != 441:
        raise ValueError("native_video_request_ids_not_unique")
    return indexed


def build_cosmos_reasoner_bundle(
    *,
    inventory_path: str | Path,
    native_video_manifest_path: str | Path,
    output_bundle: str | Path,
    receipt_path: str | Path,
    source_commit: str,
    offset: int = 0,
    count: int = 1,
) -> dict[str, Any]:
    inventory = _read_mapping(inventory_path)
    native = _read_mapping(native_video_manifest_path)
    protocol = diagnostic_protocol()
    if (
        inventory.get("status") != "ready"
        or inventory.get("pair_count") != 441
        or inventory.get("protocol_sha256") != protocol["protocol_sha256"]
    ):
        raise ValueError("pair_inventory_invalid")
    if offset < 0 or count <= 0 or offset + count > 441:
        raise ValueError("bundle_pair_range_invalid")
    source = str(source_commit).strip().lower()
    if len(source) != 40 or any(char not in "0123456789abcdef" for char in source):
        raise ValueError("source_commit_invalid")
    rows = _manifest_rows(native)
    pairs = inventory["pairs"][offset : offset + count]
    runtime_pairs: list[dict[str, Any]] = []
    selected_videos: dict[str, Mapping[str, Any]] = {}
    for pair in pairs:
        runtime_pair = {
            "pair_id": str(pair["pair_id"]),
            "task_instruction": str(pair["task_instruction"]),
            "prompt": PAIR_PROMPT,
        }
        for side in ("episode_a", "episode_b"):
            request_id = str(pair[side]["source_request_id"])
            row = rows[request_id]
            selected_videos[request_id] = row
            runtime_pair[f"{side}_video"] = f"provider_runtime/videos/{request_id}.mp4"
            runtime_pair[f"{side}_video_sha256"] = row["sha256"]
        runtime_pairs.append(runtime_pair)
    input_manifest: dict[str, Any] = {
        "schema_version": "policy_ranking_cosmos_reasoner_input.v1",
        "arm_id": "cosmos3_nano_reasoner",
        "claim_class": "post_unseal_diagnostic_only",
        "run_purpose": RUN_PURPOSE,
        "offset": offset,
        "pair_count": count,
        "pairs": runtime_pairs,
        "uniform_video_frames_per_episode": 32,
        "max_output_tokens": 4096,
        "output_schema": PAIR_OUTPUT_SCHEMA,
        "output_schema_sha256": canonical_sha256(PAIR_OUTPUT_SCHEMA),
        "policy_identity_present": False,
        "physical_outcome_present": False,
        "physical_ground_truth_pixels_present": False,
    }
    input_manifest["manifest_sha256"] = canonical_sha256(input_manifest)
    runtime_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model": COSMOS_MODEL,
        "model_revision": COSMOS_REVISION,
        "surface": "reasoner_only_vllm",
        "public_image": PUBLIC_IMAGE,
        "precision": "bf16",
        "reasoner_architecture_mode": "native_frozen_model_config",
        "native_reasoner_architecture": NATIVE_REASONER_ARCHITECTURE,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "reasoner_architecture_override": None,
        "provider_runtime_runner_sha256": PROVIDER_RUNTIME_RUNNER_SHA256,
        "provider_runtime_entrypoint_sha256": RUN_SCRIPT_SHA256,
        "media_io_kwargs": {"video": {"num_frames": 32}},
        "max_model_len": 131072,
        "tensor_parallel_size": 1,
        "source_commit": source,
        "claim_class": "post_unseal_diagnostic_only",
        "run_purpose": RUN_PURPOSE,
        "cannot_be_sole_judge_of_native_cosmos_generated_rollouts": True,
        "generated_video_or_policy_endpoint_invoked": False,
    }
    runtime_manifest["manifest_sha256"] = canonical_sha256(runtime_manifest)
    runner = (
        Path(__file__)
        .with_name("policy_ranking_evaluator_diagnostic_cosmos_provider_runtime.py")
        .read_bytes()
    )
    output = Path(output_bundle).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
        _zip_write(
            archive,
            "provider_runtime/evaluator_provider_runtime_runner.py",
            runner,
            executable=True,
        )
        _zip_write(
            archive,
            "provider_runtime/run_evaluator_provider_runtime.sh",
            RUN_SCRIPT.encode(),
            executable=True,
        )
        _zip_write(
            archive,
            "provider_runtime/evaluator_provider_runtime_manifest.json",
            (json.dumps(runtime_manifest, indent=2, sort_keys=True) + "\n").encode(),
        )
        _zip_write(
            archive,
            "provider_runtime/evaluator_input_manifest.json",
            (json.dumps(input_manifest, indent=2, sort_keys=True) + "\n").encode(),
        )
        for request_id in sorted(selected_videos):
            row = selected_videos[request_id]
            _zip_write(
                archive,
                f"provider_runtime/videos/{request_id}.mp4",
                Path(str(row["resolved_path"])).read_bytes(),
                stored=True,
            )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "built",
        "bundle_filename": output.name,
        "bundle_sha256": file_sha256(output),
        "bundle_size_bytes": output.stat().st_size,
        "source_commit": source,
        "model": COSMOS_MODEL,
        "model_revision": COSMOS_REVISION,
        "native_reasoner_architecture": NATIVE_REASONER_ARCHITECTURE,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "provider_runtime_runner_sha256": PROVIDER_RUNTIME_RUNNER_SHA256,
        "provider_runtime_entrypoint_sha256": RUN_SCRIPT_SHA256,
        "public_image": PUBLIC_IMAGE,
        "offset": offset,
        "pair_count": count,
        "unique_video_count": len(selected_videos),
        "input_manifest_sha256": input_manifest["manifest_sha256"],
        "provider_mutations_performed": 0,
        "paid_resources_used": False,
        "claim_class": "post_unseal_diagnostic_only",
        "run_purpose": RUN_PURPOSE,
        "policy_identity_in_bundle": False,
        "physical_outcome_in_bundle": False,
        "physical_ground_truth_pixels_in_bundle": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    write_json(Path(receipt_path), receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--native-video-manifest", required=True)
    parser.add_argument("--output-bundle", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args(argv)
    result = build_cosmos_reasoner_bundle(
        inventory_path=args.inventory,
        native_video_manifest_path=args.native_video_manifest,
        output_bundle=args.output_bundle,
        receipt_path=args.receipt,
        source_commit=args.source_commit,
        offset=args.offset,
        count=args.count,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "receipt_sha256"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
