import json
import zipfile

from blueprint_pipeline.policy_ranking_evaluator_diagnostic import diagnostic_protocol
from blueprint_pipeline.policy_ranking_evaluator_diagnostic_cosmos_bundle import (
    RUN_PURPOSE,
    build_cosmos_reasoner_bundle,
)
from blueprint_pipeline.policy_ranking_roboarena_calibration import file_sha256


COMMIT = "a" * 40


def test_reasoner_bundle_defaults_to_one_structured_output_canary(tmp_path):
    videos = []
    receipts = []
    for index in range(441):
        path = tmp_path / f"video-{index}.mp4"
        path.write_bytes(f"video-{index}".encode())
        request_id = f"request-{index:03d}"
        videos.append(request_id)
        receipts.append(
            {
                "request_id": request_id,
                "output_path": str(path),
                "output_sha256": file_sha256(path),
            }
        )
    protocol = diagnostic_protocol()
    inventory = {
        "status": "ready",
        "pair_count": 441,
        "protocol_sha256": protocol["protocol_sha256"],
        "pairs": [
            {
                "pair_id": f"pair-{index:03d}",
                "task_instruction": "move the object",
                "episode_a": {"source_request_id": videos[index]},
                "episode_b": {"source_request_id": videos[(index + 1) % 441]},
            }
            for index in range(441)
        ],
    }
    native = {
        "status": "passed",
        "video_count": 441,
        "all_physical_right_half_pixels_excluded": True,
        "receipts": receipts,
    }
    inventory_path = tmp_path / "inventory.json"
    native_path = tmp_path / "native.json"
    bundle_path = tmp_path / "bundle.zip"
    receipt_path = tmp_path / "receipt.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    native_path.write_text(json.dumps(native), encoding="utf-8")

    receipt = build_cosmos_reasoner_bundle(
        inventory_path=inventory_path,
        native_video_manifest_path=native_path,
        output_bundle=bundle_path,
        receipt_path=receipt_path,
        source_commit=COMMIT,
    )

    assert receipt["pair_count"] == 1
    assert receipt["unique_video_count"] == 2
    assert receipt["run_purpose"] == RUN_PURPOSE
    with zipfile.ZipFile(bundle_path) as archive:
        inputs = json.loads(
            archive.read("provider_runtime/evaluator_input_manifest.json")
        )
    assert inputs["pair_count"] == 1
    assert inputs["run_purpose"] == RUN_PURPOSE
    assert len(inputs["pairs"]) == 1
