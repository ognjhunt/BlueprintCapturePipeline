"""Build the immutable native-Cosmos Phase-B real-trace causal canary bundle."""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from .common import write_json
from .policy_ranking_successor_cosmos import (
    CHECKPOINT_REVISION,
    PHASE_B_REQUIRED_CONDITIONS,
    VLLM_IMAGE,
    VLLM_IMAGE_DIGEST,
    canonical_sha256,
    validate_droid_action_stream,
    validate_smoke_inventory_manifest,
)
from .policy_ranking_successor_cosmos_bundle import RUN_SCRIPT, _zip_write
from .policy_ranking_successor_gpu_admission import BUNDLE_SCHEMA, PUBLIC_IMAGE
from .policy_ranking_thesis import file_sha256


EXPERIMENT_ID = "policy_ranking_roboarena_disjoint_reasoner_successor_20260728"
OFFICIAL_POSITIVE_CONTROL_ASSET_SHA256 = {
    "first_frame": "78b7288846b05c2265f4aa9a31b7aa905d75e0154c2126adcf8472432a81053c",
    "action_chunks": "ba8408f727f9c77d4450b239069f81e9cdd9d099bf05da7f33f5bfb4cb2d55cd",
    "reference_output": "9b01266b6cd27478514133b00ada9c33db3f9444167f09942c11f880c629c8c0",
}
POSITIVE_CONTROL_FROZEN_GATES = {
    # Metrics use grayscale pixel values in [0, 255].  These thresholds are
    # conservative relative to the frozen NVIDIA reference (about 2.73 mean
    # adjacent-frame difference and 10.7-13.3 first-to-last per chunk).
    "chunk_temporal_absolute_difference_mean_minimum": 1.0,
    "chunk_first_to_last_absolute_difference_mean_minimum": 3.0,
    "minimum_dynamic_chunks": 3,
}


def _positive_control_manifest(
    *, first_frame: Path, action_chunks: Path, reference_output: Path
) -> dict[str, Any]:
    assets = {
        "first_frame": first_frame,
        "action_chunks": action_chunks,
        "reference_output": reference_output,
    }
    observed = {name: file_sha256(path) for name, path in assets.items()}
    if observed != OFFICIAL_POSITIVE_CONTROL_ASSET_SHA256:
        raise ValueError("official_positive_control_asset_sha256_mismatch")
    action_spec = json.loads(action_chunks.read_text(encoding="utf-8"))
    expected_metadata = {
        "prompt": "Pickup items in the supermarket",
        "fps": 10,
        "action_chunk_size": 16,
        "domain_name": "agibotworld",
        "image_size": 480,
        "view_point": "concat_view",
        "num_chunks": 4,
    }
    if any(action_spec.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("official_positive_control_action_metadata_mismatch")
    chunks = action_spec.get("action_chunks")
    if (
        not isinstance(chunks, list)
        or len(chunks) != 4
        or any(not isinstance(chunk, list) or len(chunk) != 16 for chunk in chunks)
        or any(
            not isinstance(row, list) or len(row) != 29
            for chunk in chunks
            for row in chunk
        )
    ):
        raise ValueError("official_positive_control_action_shape_mismatch")
    with Image.open(first_frame) as source:
        if source.size != (640, 720):
            raise ValueError("official_positive_control_first_frame_geometry_mismatch")
    manifest: dict[str, Any] = {
        "schema_version": "policy_ranking_cosmos_official_positive_control.v1",
        "source": {
            "repository": "https://github.com/NVIDIA/cosmos-framework",
            "revision": "09f23119ea92c707207bba55565e7a09d16896a2",
            "example": "AgiBotWorld forward-dynamics four-chunk example",
        },
        "asset_sha256": observed,
        "request_count": 4,
        "published_request_contract": expected_metadata,
        "request_deviations": [
            "guardrails=false because the frozen public robotics server starts with --no-guardrails"
        ],
        "frozen_gates": POSITIVE_CONTROL_FROZEN_GATES,
        "decision_rule": {
            "pass": "admit the already-frozen DROID diagnostic matrix",
            "fail": "submit zero DROID requests and tear down the model server",
        },
        "claim_boundary": (
            "A pass validates only the pinned deployment on NVIDIA's AgiBotWorld "
            "pathway; it does not qualify DROID, ranking, or physical success."
        ),
        "paid_execution_admitted": False,
        "provider_called": False,
        "outcome_labels_accessed": False,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def _runtime_sources() -> tuple[bytes, bytes]:
    package = Path(__file__).resolve().parent
    return (
        (package / "policy_ranking_successor_cosmos_provider_runtime.py").read_bytes(),
        (package / "policy_ranking_successor_retained_remote.py").read_bytes(),
    )


def build_phase_b_cosmos_canary_bundle(
    *,
    replay_canary_path: str | Path,
    output_bundle: str | Path,
    receipt_path: str | Path,
    task_instruction: str,
    positive_control_first_frame_path: str | Path | None = None,
    positive_control_action_chunks_path: str | Path | None = None,
    positive_control_reference_output_path: str | Path | None = None,
) -> dict[str, Any]:
    task_instruction = str(task_instruction).strip()
    if not task_instruction:
        raise ValueError("task_specific_instruction_required")
    if task_instruction == "A robot manipulates an object.":
        raise ValueError("generic_robot_manipulation_prompt_forbidden")
    canary_path = Path(replay_canary_path).resolve()
    canary = json.loads(canary_path.read_text(encoding="utf-8"))
    recorded = str(canary.get("manifest_sha256") or "")
    if recorded != canonical_sha256(
        {key: value for key, value in canary.items() if key != "manifest_sha256"}
    ):
        raise ValueError("replay_canary_manifest_sha256_mismatch")
    canary_task = str(canary.get("task_instruction") or "").strip()
    if canary_task and canary_task != task_instruction:
        raise ValueError("task_instruction_does_not_match_replay_canary")
    access = canary.get("access_contract")
    label_seal = canary.get("label_seal")
    label_blind = (
        isinstance(access, dict)
        and access.get("selection_used_outcomes") is False
        and access.get("outcome_fields_parsed_for_task_prompt") is False
        and access.get("physical_future_pixels_in_provider_input") is False
        and access.get("policy_identity_in_provider_payload") is False
    ) or (
        isinstance(label_seal, dict) and label_seal.get("outcome_labels_accessed") is False
    )
    if not label_blind:
        raise ValueError("replay_canary_label_blind_contract_missing")
    controls = canary.get("controls")
    if not isinstance(controls, dict):
        raise ValueError("replay_canary_controls_missing")
    conditions = {
        condition: validate_droid_action_stream(controls.get(condition))
        for condition in PHASE_B_REQUIRED_CONDITIONS
    }
    observation_source = Path(canary["initial_observation"]["path"]).resolve()
    if file_sha256(observation_source) != canary["initial_observation"]["sha256"]:
        raise ValueError("replay_canary_observation_sha256_mismatch")
    positive_control_values = (
        positive_control_first_frame_path,
        positive_control_action_chunks_path,
        positive_control_reference_output_path,
    )
    if any(value is not None for value in positive_control_values) and not all(
        value is not None for value in positive_control_values
    ):
        raise ValueError("official_positive_control_assets_must_be_all_or_none")
    positive_control_assets: dict[str, Path] | None = None
    positive_control_manifest: dict[str, Any] | None = None
    if all(value is not None for value in positive_control_values):
        positive_control_assets = {
            "first_frame": Path(str(positive_control_first_frame_path)).resolve(),
            "action_chunks": Path(str(positive_control_action_chunks_path)).resolve(),
            "reference_output": Path(str(positive_control_reference_output_path)).resolve(),
        }
        positive_control_manifest = _positive_control_manifest(**positive_control_assets)

    output = Path(output_bundle).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phase-b-cosmos-canary-") as temporary:
        padded = Path(temporary) / "initial_observation.png"
        with Image.open(observation_source) as source:
            rgb = source.convert("RGB")
            if rgb.size != (640, 540):
                raise ValueError("replay_canary_observation_geometry_invalid")
            target = Image.new("RGB", (640, 544))
            target.paste(rgb, (0, 0))
            edge = rgb.crop((0, 539, 640, 540)).resize((640, 4), Image.Resampling.NEAREST)
            target.paste(edge, (0, 540))
            target.save(padded, format="PNG", optimize=False)
            edge.close()
            target.close()
        observation_sha = file_sha256(padded)
        action_hashes = {
            condition: payload["action_sha256"] for condition, payload in conditions.items()
        }
        rows: list[dict[str, Any]] = []
        for condition in PHASE_B_REQUIRED_CONDITIONS:
            for seed in (0, 1):
                request_material = {
                    "initial_observation_sha256": observation_sha,
                    "task_instruction": task_instruction,
                    "action_sha256": action_hashes[condition],
                    "seed": seed,
                    "checkpoint_revision": CHECKPOINT_REVISION,
                }
                rows.append(
                    {
                        "request_id": canonical_sha256(request_material),
                        "condition": condition,
                        "seed": seed,
                        "action_sha256": action_hashes[condition],
                    }
                )
        inventory_rows = [
            {
                "request_id": row["request_id"],
                "condition": row["condition"],
                "seed": row["seed"],
                "action_sha256": row["action_sha256"],
                "observation_sha256": observation_sha,
            }
            for row in rows
        ]
        inventory = {
            "schema_version": "policy_ranking_successor_smoke_request_inventory.v1",
            "experiment_id": EXPERIMENT_ID,
            "initial_observation_sha256": observation_sha,
            "task_instruction": task_instruction,
            "action_hashes": action_hashes,
            "required_conditions": list(PHASE_B_REQUIRED_CONDITIONS),
            "requests": rows,
            "inventory_sha256": canonical_sha256(inventory_rows),
            "policy_identity_in_provider_payload": False,
            "outcome_labels_accessed": False,
        }
        validate_smoke_inventory_manifest(inventory)
        action_streams = {
            "schema_version": "policy_ranking_phase_b_native_cosmos_action_streams.v1",
            "experiment_id": EXPERIMENT_ID,
            "conditions": conditions,
            "condition_order": list(conditions),
            "policy_identity_present": False,
            "source_replay_canary_manifest_sha256": recorded,
        }
        runtime_source, retained_source = _runtime_sources()
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
            "vision_conditioning_mode": "first_pixel_frame_only",
            "additional_starter_video_frames_condition_model": False,
            "vision_conditioning_source": (
                "pinned_vllm_omni__prepare_latents_action_video_and_upstream_test"
            ),
            "raw_action_dim": 10,
            "action_space": "midtrain",
            "positive_control_request_count": 4 if positive_control_manifest else 0,
            "qualification_canary_request_count": 2,
            "scientific_matrix_request_count": 12,
            "total_initial_generation_request_count": 18 if positive_control_manifest else 14,
            "request_budget_amendment_sha256": (
                positive_control_manifest["manifest_sha256"]
                if positive_control_manifest
                else None
            ),
            "precision": "bf16",
            "trust_remote_code": False,
            "post_training_or_lora": False,
            "source_replay_canary_file_sha256": file_sha256(canary_path),
            "source_replay_canary_manifest_sha256": recorded,
            "initial_observation_sha256": observation_sha,
            "claims": {
                "implementation": True,
                "runtime": False,
                "generated_media": False,
                "wam_causal_validity": False,
                "ranking_fidelity": False,
                "physical_performance": False,
            },
        }
        input_manifest = {
            "schema_version": "policy_ranking_phase_b_native_cosmos_input.v1",
            "experiment_id": EXPERIMENT_ID,
            "initial_observation_sha256": observation_sha,
            "action_streams_sha256": canonical_sha256(action_streams),
            "smoke_inventory_sha256": canonical_sha256(inventory),
            "request_count": 12,
            "real_recorded_trace": True,
            "real_policy_swapped_trace": True,
            "valid_identity_rot6d_no_motion": True,
            "outcome_labels_accessed": False,
            "task_specific_instruction_used": True,
            "task_instruction_sha256": canonical_sha256(task_instruction),
            "vision_conditioning_mode": "first_pixel_frame_only",
        }
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
                RUN_SCRIPT.replace(
                    "set -u\n",
                    f'set -u\nexport BLUEPRINT_COSMOS_EXPERIMENT_ID="{EXPERIMENT_ID}"\n',
                    1,
                ).encode("utf-8"),
                executable=True,
            )
            _zip_write(
                archive,
                "provider_runtime/successor_retained_control.py",
                retained_source,
                executable=True,
            )
            for name, value in (
                ("wam_provider_runtime_manifest.json", runtime_manifest),
                ("wam_rollout_input_manifest.json", input_manifest),
                ("cosmos3_input/smoke_request_inventory.json", inventory),
                ("cosmos3_input/action_streams.json", action_streams),
            ):
                _zip_write(
                    archive,
                    f"provider_runtime/{name}",
                    json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n",
                )
            _zip_write(
                archive,
                "provider_runtime/cosmos3_input/initial_observation.png",
                padded.read_bytes(),
            )
            if positive_control_manifest and positive_control_assets:
                _zip_write(
                    archive,
                    "provider_runtime/cosmos3_positive_control/manifest.json",
                    json.dumps(positive_control_manifest, indent=2, sort_keys=True).encode("utf-8")
                    + b"\n",
                )
                for archive_name, source_name in (
                    ("first_frame.png", "first_frame"),
                    ("action_chunks.json", "action_chunks"),
                    ("reference_output.mp4", "reference_output"),
                ):
                    _zip_write(
                        archive,
                        f"provider_runtime/cosmos3_positive_control/{archive_name}",
                        positive_control_assets[source_name].read_bytes(),
                    )
    receipt: dict[str, Any] = {
        "schema_version": (
            "policy_ranking_phase_b_native_cosmos_positive_control_bundle_receipt.v1"
            if positive_control_manifest
            else "policy_ranking_phase_b_native_cosmos_bundle_receipt.v1"
        ),
        "experiment_id": EXPERIMENT_ID,
        "status": "built",
        "bundle_path": str(output),
        "bundle_sha256": file_sha256(output),
        "bundle_size_bytes": output.stat().st_size,
        "public_image": f"{VLLM_IMAGE}@{VLLM_IMAGE_DIGEST}",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "source_replay_canary_manifest_sha256": recorded,
        "initial_observation_sha256": observation_sha,
        "smoke_inventory_sha256": canonical_sha256(inventory),
        "action_streams_sha256": canonical_sha256(action_streams),
        "positive_control_included": positive_control_manifest is not None,
        "positive_control_manifest_sha256": (
            positive_control_manifest["manifest_sha256"]
            if positive_control_manifest
            else None
        ),
        "provider_mutations_performed": 0,
        "paid_resources_used": False,
    }
    receipt["manifest_sha256"] = canonical_sha256(receipt)
    write_json(Path(receipt_path), receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-canary", required=True)
    parser.add_argument("--output-bundle", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--task-instruction", required=True)
    parser.add_argument("--positive-control-first-frame")
    parser.add_argument("--positive-control-action-chunks")
    parser.add_argument("--positive-control-reference-output")
    args = parser.parse_args(argv)
    result = build_phase_b_cosmos_canary_bundle(
        replay_canary_path=args.replay_canary,
        output_bundle=args.output_bundle,
        receipt_path=args.receipt,
        task_instruction=args.task_instruction,
        positive_control_first_frame_path=args.positive_control_first_frame,
        positive_control_action_chunks_path=args.positive_control_action_chunks,
        positive_control_reference_output_path=args.positive_control_reference_output,
    )
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
