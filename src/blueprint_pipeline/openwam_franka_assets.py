"""Pinned public single-FR3 model inventory; full weights have not been loaded here."""

from pathlib import Path
from typing import Any
from .openwam_piper_assets import _verify_checkpoint_inventory
from .openwam_franka_policy import MODEL_ID, MODEL_REVISION, WEIGHT_SHA256

EXPECTED_FILES = {
    "README.md": (1384, "456ba724d1f54e86ea92a9349042d5b4548af59cce39845038fe6883e8812461"),
    "checkpoint_step_9860.safetensors": (
        24813767464,
        "76e53e963525d9eebdb70e5285908c6497082270de66f9227874844e96bc43bb",
    ),
    "config.yaml": (3407, "704222ea8504bb9332b6492a83054dcdeaa9f7b810836eec6dc9436bb971adc2"),
    "normalization_stats.npy": (
        1368,
        "d54b7b411f545de021a8ffd04f35637db6588d36bbcbd4463755271b91859d9a",
    ),
    "tokenizer/google/umt5-xxl/special_tokens_map.json": (
        6623,
        "7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d",
    ),
    "tokenizer/google/umt5-xxl/spiece.model": (
        4548313,
        "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458",
    ),
    "tokenizer/google/umt5-xxl/tokenizer.json": (
        16837417,
        "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b",
    ),
    "tokenizer/google/umt5-xxl/tokenizer_config.json": (
        61728,
        "ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05",
    ),
}


def verify_checkpoint_directory(
    directory: str | Path, *, deadline_epoch: float | None = None
) -> dict[str, Any]:
    _verify_checkpoint_inventory(directory, EXPECTED_FILES, deadline_epoch=deadline_epoch)
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "weight_sha256": WEIGHT_SHA256,
        "checkpoint_bytes_verified": True,
        "files_verified": len(EXPECTED_FILES),
        "license_basis": "pinned_model_card_apache_2.0",
        "inference_performed": False,
    }
