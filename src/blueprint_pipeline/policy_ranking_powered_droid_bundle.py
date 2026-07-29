"""Build the immutable 17-session native-Cosmos powered confirmation bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .common import write_json
from .policy_ranking_successor_cosmos import (
    CHECKPOINT_REVISION,
    PHASE_B_REQUIRED_CONDITIONS,
    VLLM_IMAGE,
    VLLM_IMAGE_DIGEST,
    canonical_sha256,
    validate_droid_action_stream,
)
from .policy_ranking_successor_cosmos_bundle import RUN_SCRIPT, _zip_write
from .policy_ranking_successor_gpu_admission import BUNDLE_SCHEMA, PUBLIC_IMAGE
from .policy_ranking_thesis import file_sha256


EXPERIMENT_ID = "policy_ranking_roboarena_powered_droid_confirmation_20260729"
PROVIDER_PACKET_SCHEMA = "policy_ranking_powered_droid_provider_packet.v1"
RECEIPT_SCHEMA = "policy_ranking_powered_droid_bundle_receipt.v1"


def _canonical_artifact(path: Path, *, digest_field: str = "manifest_sha256") -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"artifact_not_object:{path.name}")
    payload = dict(value)
    recorded = str(payload.get(digest_field) or "")
    if recorded != canonical_sha256(
        {key: item for key, item in payload.items() if key != digest_field}
    ):
        raise ValueError(f"artifact_digest_mismatch:{path.name}")
    return payload


def _runtime_sources() -> tuple[bytes, bytes]:
    package = Path(__file__).resolve().parent
    return (
        (package / "policy_ranking_successor_cosmos_provider_runtime.py").read_bytes(),
        (package / "policy_ranking_successor_retained_remote.py").read_bytes(),
    )


def _padded_observation(source: Path, output: Path) -> str:
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        if rgb.size != (640, 540):
            raise ValueError("powered_observation_geometry_invalid")
        target = Image.new("RGB", (640, 544))
        target.paste(rgb, (0, 0))
        edge = rgb.crop((0, 539, 640, 540)).resize((640, 4), Image.Resampling.NEAREST)
        target.paste(edge, (0, 540))
        target.save(output, format="PNG", optimize=False)
        edge.close()
        target.close()
    return file_sha256(output)


def build_powered_droid_bundle(
    *,
    replay_packet_path: str | Path,
    official_canary_dir: str | Path,
    output_bundle: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Build a deterministic provider bundle without contacting a provider."""

    source_packet_path = Path(replay_packet_path).resolve()
    source_packet = _canonical_artifact(source_packet_path)
    if source_packet.get("schema_version") != "policy_ranking_phase_b_powered_replay_packet.v1":
        raise ValueError("powered_replay_packet_schema_invalid")
    if source_packet.get("status") != "passed":
        raise ValueError("powered_replay_packet_not_passed")
    if source_packet.get("session_count") != 17 or source_packet.get("window_count") != 51:
        raise ValueError("powered_replay_packet_power_invalid")
    if source_packet.get("scientific_request_count") != 612:
        raise ValueError("powered_replay_packet_request_count_invalid")
    label_seal = source_packet.get("label_seal")
    if not isinstance(label_seal, Mapping) or any(label_seal.values()):
        raise ValueError("powered_replay_packet_label_seal_invalid")

    canary_root = Path(official_canary_dir).resolve()
    canary_manifest_path = canary_root / "canary_manifest.json"
    canary_manifest = _canonical_artifact(canary_manifest_path)
    canary_image = canary_root / "initial_observation.png"
    canary_actions_path = canary_root / "action_streams.json"
    canary_actions = json.loads(canary_actions_path.read_text(encoding="utf-8"))
    provider_inputs = canary_manifest.get("provider_inputs") or {}
    if file_sha256(canary_image) != provider_inputs.get("initial_observation_sha256"):
        raise ValueError("official_canary_image_digest_mismatch")
    if canonical_sha256(canary_actions) != provider_inputs.get("action_streams_sha256"):
        raise ValueError("official_canary_actions_digest_mismatch")

    output = Path(output_bundle).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    provider_rows: list[dict[str, Any]] = []
    image_archive_rows: list[tuple[str, bytes]] = []
    with tempfile.TemporaryDirectory(prefix="powered-droid-bundle-") as temporary:
        temp_root = Path(temporary)
        rows = source_packet.get("rows")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            raise ValueError("powered_replay_rows_invalid")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("powered_replay_row_invalid")
            session_id = str(row.get("session_id_internal_only") or "")
            window_index = int(row.get("window_index", -1))
            observation = row.get("initial_observation")
            controls = row.get("controls")
            if not session_id or window_index < 0 or not isinstance(observation, Mapping):
                raise ValueError("powered_replay_row_identity_invalid")
            if not isinstance(controls, Mapping) or set(controls) != set(
                PHASE_B_REQUIRED_CONDITIONS
            ):
                raise ValueError("powered_replay_controls_invalid")
            validated_controls = {
                condition: validate_droid_action_stream(controls[condition])
                for condition in PHASE_B_REQUIRED_CONDITIONS
            }
            source_image = Path(str(observation.get("path") or "")).resolve()
            if file_sha256(source_image) != observation.get("sha256"):
                raise ValueError("powered_replay_observation_digest_mismatch")
            padded = temp_root / f"{session_id}-{window_index:02d}.png"
            padded_sha = _padded_observation(source_image, padded)
            relative = f"images/{session_id}/window_{window_index:02d}.png"
            image_archive_rows.append((relative, padded.read_bytes()))
            provider_rows.append(
                {
                    "session_id_internal_only": session_id,
                    "window_index": window_index,
                    "initial_observation_relative_path": relative,
                    "initial_observation_sha256": padded_sha,
                    "controls": validated_controls,
                }
            )

        if len(provider_rows) != 51:
            raise ValueError("powered_provider_window_count_invalid")
        provider_packet: dict[str, Any] = {
            "schema_version": PROVIDER_PACKET_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "source_replay_packet_manifest_sha256": source_packet["manifest_sha256"],
            "session_count": 17,
            "window_count": 51,
            "conditions": list(PHASE_B_REQUIRED_CONDITIONS),
            "seeds": [0, 1],
            "structured_canary_request_count": 1,
            "scientific_request_count": 612,
            "rows": provider_rows,
            "provider_payload_contract": {
                "policy_identity_in_model_request": False,
                "physical_future_pixels_in_model_request": False,
                "outcome_labels_accessed": False,
                "task_metadata_accessed": False,
                "prompt": "single ASCII space",
            },
        }
        provider_packet["manifest_sha256"] = canonical_sha256(provider_packet)
        image_manifest = [
            {
                "relative_path": row["initial_observation_relative_path"],
                "sha256": row["initial_observation_sha256"],
            }
            for row in provider_rows
        ]
        runtime_source, retained_source = _runtime_sources()
        runtime_runner_sha256 = hashlib.sha256(runtime_source).hexdigest()
        runtime_manifest = {
            "schema_version": BUNDLE_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "checkpoint": "nvidia/Cosmos3-Nano",
            "checkpoint_revision": CHECKPOINT_REVISION,
            "public_image": PUBLIC_IMAGE,
            "pipeline_class": "Cosmos3OmniDiffusersPipeline",
            "vllm_omni_source_revision": "1c6e7313394923000215a3299f4f79ede3873ecc",
            "vision_conditioning_mode": "matched_first_pixel_frame_per_window",
            "qualification_canary_request_count": 1,
            "scientific_matrix_request_count": 612,
            "total_initial_generation_request_count": 613,
            "precision": "bf16",
            "trust_remote_code": False,
            "post_training_or_lora": False,
            "provider_packet_sha256": provider_packet["manifest_sha256"],
            "provider_runtime_runner_sha256": runtime_runner_sha256,
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
            "schema_version": "policy_ranking_powered_droid_input_manifest.v1",
            "experiment_id": EXPERIMENT_ID,
            "provider_packet_sha256": provider_packet["manifest_sha256"],
            "image_manifest_sha256": canonical_sha256(image_manifest),
            "official_canary_manifest_sha256": canary_manifest["manifest_sha256"],
            "request_count": 613,
            "outcome_labels_accessed": False,
            "physical_future_pixels_in_provider_input": False,
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
            for archive_name, value in (
                ("wam_provider_runtime_manifest.json", runtime_manifest),
                ("wam_rollout_input_manifest.json", input_manifest),
                ("cosmos3_powered_droid/packet.json", provider_packet),
            ):
                _zip_write(
                    archive,
                    f"provider_runtime/{archive_name}",
                    json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n",
                )
            for relative, data in image_archive_rows:
                _zip_write(
                    archive,
                    f"provider_runtime/cosmos3_powered_droid/{relative}",
                    data,
                )
            for name, source in (
                ("canary_manifest.json", canary_manifest_path),
                ("initial_observation.png", canary_image),
                ("action_streams.json", canary_actions_path),
            ):
                _zip_write(
                    archive,
                    f"provider_runtime/cosmos3_powered_droid/official_canary/{name}",
                    source.read_bytes(),
                )

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "built",
        "bundle_path": str(output),
        "bundle_sha256": file_sha256(output),
        "bundle_size_bytes": output.stat().st_size,
        "public_image": f"{VLLM_IMAGE}@{VLLM_IMAGE_DIGEST}",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "source_replay_packet_manifest_sha256": source_packet["manifest_sha256"],
        "provider_packet_sha256": provider_packet["manifest_sha256"],
        "image_manifest_sha256": canonical_sha256(image_manifest),
        "official_canary_manifest_sha256": canary_manifest["manifest_sha256"],
        "provider_runtime_runner_sha256": runtime_runner_sha256,
        "provider_mutations_performed": 0,
        "paid_resources_used": False,
    }
    receipt["manifest_sha256"] = canonical_sha256(receipt)
    write_json(Path(receipt_path), receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-packet", required=True)
    parser.add_argument("--official-canary-dir", required=True)
    parser.add_argument("--output-bundle", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    result = build_powered_droid_bundle(
        replay_packet_path=args.replay_packet,
        official_canary_dir=args.official_canary_dir,
        output_bundle=args.output_bundle,
        receipt_path=args.receipt,
    )
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
