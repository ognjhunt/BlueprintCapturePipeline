"""Build the immutable official-DROID Cosmos3 reference canary bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Sequence

from .common import write_json
from .policy_ranking_successor_cosmos import (
    CHECKPOINT_REVISION,
    VLLM_IMAGE,
    VLLM_IMAGE_DIGEST,
    canonical_sha256,
)
from .policy_ranking_successor_cosmos_bundle import RUN_SCRIPT, _sha256_file, _zip_write
from .policy_ranking_successor_gpu_admission import BUNDLE_SCHEMA, PUBLIC_IMAGE


EXPERIMENT_ID = "policy_ranking_roboarena_droid_reference_confirmation_20260729"
RECEIPT_SCHEMA = "policy_ranking_cosmos3_droid_reference_bundle_receipt.v1"
REFERENCE_SCHEMA = "policy_ranking_cosmos3_official_droid_reference_canary.v1"


def _load_reference(reference_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    manifest_path = reference_dir / "canary_manifest.json"
    action_path = reference_dir / "action_streams.json"
    observation_path = reference_dir / "initial_observation.png"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = str(manifest.get("manifest_sha256") or "")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if not recorded or recorded != canonical_sha256(payload):
        raise ValueError("official_droid_reference_manifest_sha256_mismatch")
    if manifest.get("schema_version") != REFERENCE_SCHEMA:
        raise ValueError("official_droid_reference_manifest_schema_invalid")
    actions = json.loads(action_path.read_text(encoding="utf-8"))
    provider_inputs = manifest.get("provider_inputs") or {}
    if _sha256_file(observation_path) != provider_inputs.get("initial_observation_sha256"):
        raise ValueError("official_droid_reference_initial_sha256_mismatch")
    if canonical_sha256(actions) != provider_inputs.get("action_streams_sha256"):
        raise ValueError("official_droid_reference_actions_sha256_mismatch")
    if ((manifest.get("runtime") or {}).get("paid_execution_admitted")) is not False:
        raise ValueError("official_droid_reference_must_be_unpaid_at_bundle_build")
    if ((manifest.get("runtime") or {}).get("provider_called")) is not False:
        raise ValueError("official_droid_reference_provider_already_called")
    return manifest, actions, observation_path


def _runtime_sources() -> tuple[bytes, bytes]:
    package = Path(__file__).resolve().parent
    return (
        (package / "policy_ranking_successor_cosmos_provider_runtime.py").read_bytes(),
        (package / "policy_ranking_successor_retained_remote.py").read_bytes(),
    )


def build_droid_reference_bundle(
    *,
    reference_dir: str | Path,
    output_bundle: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    reference_root = Path(reference_dir).expanduser().resolve()
    manifest, actions, observation = _load_reference(reference_root)
    runtime_source, retained_source = _runtime_sources()
    runtime_source_sha256 = hashlib.sha256(runtime_source).hexdigest()
    output_argument = Path(output_bundle).expanduser()
    output = output_argument.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt_bundle_path = (
        output_argument.as_posix() if not output_argument.is_absolute() else output.name
    )
    runtime_manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "checkpoint": "nvidia/Cosmos3-Nano",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "public_image": PUBLIC_IMAGE,
        "forward_dynamics_endpoint": "/v1/videos",
        "published_request_geometry": "640x540",
        "pipeline_class": "Cosmos3OmniDiffusersPipeline",
        "request_count_maximum": 2,
        "recorded_request_count_maximum": 1,
        "no_motion_request_count_maximum": 1,
        "qualification_canary_request_count": 2,
        "scientific_matrix_request_count": 0,
        "total_initial_generation_request_count": 2,
        "reference_manifest_sha256": manifest["manifest_sha256"],
        "provider_runtime_runner_sha256": runtime_source_sha256,
        "claims": {
            "implementation": True,
            "runtime": False,
            "wam_causal_validity": False,
            "ranking_fidelity": False,
            "captured_site_transfer": False,
            "physical_performance": False,
        },
    }
    input_manifest = {
        "schema_version": "policy_ranking_cosmos3_droid_reference_input.v1",
        "experiment_id": EXPERIMENT_ID,
        "reference_manifest_sha256": manifest["manifest_sha256"],
        "initial_observation_sha256": _sha256_file(observation),
        "action_streams_sha256": canonical_sha256(actions),
        "physical_future_pixels_in_provider_input": False,
        "policy_ranking_labels_accessed": False,
        "request_count_maximum": 2,
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
        for name, payload in (
            ("wam_provider_runtime_manifest.json", runtime_manifest),
            ("wam_rollout_input_manifest.json", input_manifest),
            ("cosmos3_droid_reference/canary_manifest.json", manifest),
            ("cosmos3_droid_reference/action_streams.json", actions),
        ):
            _zip_write(
                archive,
                f"provider_runtime/{name}",
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
                + b"\n",
            )
        _zip_write(
            archive,
            "provider_runtime/cosmos3_droid_reference/initial_observation.png",
            observation.read_bytes(),
        )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "built",
        "bundle_path": receipt_bundle_path,
        "bundle_sha256": _sha256_file(output),
        "bundle_size_bytes": output.stat().st_size,
        "public_image": f"{VLLM_IMAGE}@{VLLM_IMAGE_DIGEST}",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "reference_manifest_sha256": manifest["manifest_sha256"],
        "initial_observation_sha256": input_manifest["initial_observation_sha256"],
        "action_streams_sha256": input_manifest["action_streams_sha256"],
        "provider_runtime_runner_sha256": runtime_source_sha256,
        "provider_mutations_performed": 0,
        "paid_resources_used": False,
    }
    receipt["manifest_sha256"] = canonical_sha256(receipt)
    write_json(Path(receipt_path), receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output-bundle", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    result = build_droid_reference_bundle(
        reference_dir=args.reference_dir,
        output_bundle=args.output_bundle,
        receipt_path=args.receipt,
    )
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
