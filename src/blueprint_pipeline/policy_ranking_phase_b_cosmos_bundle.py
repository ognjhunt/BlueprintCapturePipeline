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
    ALLOWED_CONDITIONS,
    CHECKPOINT_REVISION,
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


def _runtime_sources() -> tuple[bytes, bytes]:
    package = Path(__file__).resolve().parent
    return (
        (package / "policy_ranking_successor_cosmos_provider_runtime.py").read_bytes(),
        (package / "policy_ranking_successor_retained_remote.py").read_bytes(),
    )


def build_phase_b_cosmos_canary_bundle(
    *, replay_canary_path: str | Path, output_bundle: str | Path, receipt_path: str | Path
) -> dict[str, Any]:
    canary_path = Path(replay_canary_path).resolve()
    canary = json.loads(canary_path.read_text(encoding="utf-8"))
    recorded = str(canary.get("manifest_sha256") or "")
    if recorded != canonical_sha256(
        {key: value for key, value in canary.items() if key != "manifest_sha256"}
    ):
        raise ValueError("replay_canary_manifest_sha256_mismatch")
    controls = canary.get("controls")
    if not isinstance(controls, dict):
        raise ValueError("replay_canary_controls_missing")
    conditions = {
        condition: validate_droid_action_stream(controls.get(condition))
        for condition in ALLOWED_CONDITIONS
    }
    observation_source = Path(canary["initial_observation"]["path"]).resolve()
    if file_sha256(observation_source) != canary["initial_observation"]["sha256"]:
        raise ValueError("replay_canary_observation_sha256_mismatch")

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
        task_instruction = "A robot manipulates an object."
        rows: list[dict[str, Any]] = []
        for condition in ALLOWED_CONDITIONS:
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
            "raw_action_dim": 10,
            "action_space": "midtrain",
            "qualification_canary_request_count": 2,
            "scientific_matrix_request_count": 10,
            "total_initial_generation_request_count": 12,
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
            "request_count": 10,
            "real_recorded_trace": True,
            "real_policy_swapped_trace": True,
            "valid_identity_rot6d_no_motion": True,
            "outcome_labels_accessed": False,
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
    receipt: dict[str, Any] = {
        "schema_version": "policy_ranking_phase_b_native_cosmos_bundle_receipt.v1",
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
    args = parser.parse_args(argv)
    result = build_phase_b_cosmos_canary_bundle(
        replay_canary_path=args.replay_canary,
        output_bundle=args.output_bundle,
        receipt_path=args.receipt,
    )
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
