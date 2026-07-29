from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from pathlib import Path

from PIL import Image

from blueprint_pipeline.policy_ranking_powered_droid_bundle import (
    build_powered_droid_bundle,
)
from blueprint_pipeline.policy_ranking_successor_cosmos import (
    canonical_sha256,
    droid_action_stream,
)
from blueprint_pipeline.policy_ranking_successor_gpu_admission import (
    POWERED_DROID_PROFILE,
    inspect_successor_bundle,
)
from blueprint_pipeline.policy_ranking_thesis import file_sha256


def _stream(offset: float) -> dict:
    return droid_action_stream(
        [
            [
                offset + index / 100_000,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                float(index >= 8),
            ]
            for index in range(16)
        ]
    )


def test_powered_bundle_binds_all_images_actions_and_official_canary(tmp_path: Path) -> None:
    controls = {
        name: _stream(index / 100.0)
        for index, name in enumerate(
            ("recorded", "zero", "shuffled", "reversed", "policy_swapped", "shifted")
        )
    }
    rows = []
    for session_index in range(17):
        for window_index in range(3):
            image = tmp_path / f"observation-{session_index}-{window_index}.png"
            Image.new("RGB", (640, 540), color=(session_index, window_index, 20)).save(image)
            rows.append(
                {
                    "session_id_internal_only": f"session-{session_index:02d}",
                    "window_index": window_index,
                    "initial_observation": {
                        "path": str(image),
                        "sha256": file_sha256(image),
                    },
                    "controls": controls,
                }
            )
    packet = {
        "schema_version": "policy_ranking_phase_b_powered_replay_packet.v1",
        "status": "passed",
        "session_count": 17,
        "window_count": 51,
        "scientific_request_count": 612,
        "rows": rows,
        "label_seal": {
            "metadata_yaml_opened": False,
            "outcome_labels_accessed": False,
            "task_instruction_accessed": False,
            "physical_future_pixels_used_as_provider_input": False,
        },
    }
    packet["manifest_sha256"] = canonical_sha256(packet)
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")

    canary = tmp_path / "canary"
    canary.mkdir()
    canary_image = canary / "initial_observation.png"
    Image.new("RGB", (640, 544), color=(1, 2, 3)).save(canary_image)
    canary_actions = {"recorded": _stream(0.01), "no_motion": _stream(0.0)}
    (canary / "action_streams.json").write_text(json.dumps(canary_actions), encoding="utf-8")
    canary_manifest = {
        "schema_version": "policy_ranking_cosmos3_official_droid_reference_canary.v2",
        "provider_inputs": {
            "initial_observation_sha256": file_sha256(canary_image),
            "action_streams_sha256": canonical_sha256(canary_actions),
        },
    }
    canary_manifest["manifest_sha256"] = canonical_sha256(canary_manifest)
    (canary / "canary_manifest.json").write_text(json.dumps(canary_manifest), encoding="utf-8")

    bundle = tmp_path / "bundle.zip"
    receipt_path = tmp_path / "receipt.json"
    receipt = build_powered_droid_bundle(
        replay_packet_path=packet_path,
        official_canary_dir=canary,
        output_bundle=bundle,
        receipt_path=receipt_path,
    )

    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        assert sum("/images/" in name for name in names) == 51
        provider_packet = json.loads(
            archive.read("provider_runtime/cosmos3_powered_droid/packet.json")
        )
        assert provider_packet["scientific_request_count"] == 612
        assert (
            provider_packet["provider_payload_contract"]["policy_identity_in_model_request"]
            is False
        )
    profile = replace(
        POWERED_DROID_PROFILE,
        expected_bundle_sha256=receipt["bundle_sha256"],
        expected_bundle_size_bytes=receipt["bundle_size_bytes"],
        expected_embedded_input_hashes={
            key: receipt[key]
            for key in (
                "provider_packet_sha256",
                "image_manifest_sha256",
                "official_canary_manifest_sha256",
                "provider_runtime_runner_sha256",
            )
        },
    )
    inspection = inspect_successor_bundle(bundle, receipt=receipt, profile=profile)

    assert inspection["status"] == "passed"
    assert inspection["blockers"] == []
