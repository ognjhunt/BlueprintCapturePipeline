"""Static admission contract for the G1 development provider bundle."""

from __future__ import annotations

import json
from typing import Any
from zipfile import ZipFile

from .decision_evidence_contracts import canonical_digest
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE

REQUIRED_ENTRIES = {
    "provider_runtime/run_adp_arena_provider_runtime.sh",
    "provider_runtime/native_g1_provider_manifest.json",
    "provider_runtime/inputs/native_g1_development_campaign.v1.json",
    "provider_runtime/inputs/book_handoff.json",
    "provider_runtime/inputs/navigation_authority.json",
    "provider_runtime/inputs/scene_packets/manipulation/native_task_arena_packet_receipt.v1.json",
    "provider_runtime/inputs/scene_packets/movement/native_task_arena_packet_receipt.v1.json",
    "provider_runtime/publisher-source/native_g1_publisher_source_stage.v1.json",
    "provider_runtime/publisher-source/source/.git/config",
    "provider_runtime/native_task_runtime_sources/native_task_runtime_source_packet.v1.json",
    "provider_runtime/blueprint_pipeline/__init__.py",
    "provider_runtime/blueprint_pipeline/native_g1_provider_runtime.py",
    "provider_runtime/scripts/fetch_g1_humanoidarena_checkpoint.py",
    "provider_runtime/scripts/fetch_g1_sonic_assets.py",
    "configs/g1_humanoidarena_checkpoint_inventory.v1.json",
    "configs/g1_sonic_default_asset_inventory.v1.json",
    "configs/g1_humanoidarena_lerobot_pi_py312_linux_x86_64.requirements.txt",
} | {
    "provider_runtime/inputs/rights/" + candidate + ".json"
    for candidate in (
        "humanoidarena_dp_g1_dex3_sonic",
        "humanoidarena_pi05_g1_dex3_sonic",
        "humanoidarena_dp_g1_dex3_sonic_vision_navi",
        "humanoidarena_pi05_g1_dex3_sonic_vision_navi",
    )
}


def validate_manifest(archive: ZipFile) -> dict[str, Any]:
    manifest = json.loads(archive.read("provider_runtime/native_g1_provider_manifest.json"))
    campaign = json.loads(
        archive.read("provider_runtime/inputs/native_g1_development_campaign.v1.json")
    )
    source = json.loads(
        archive.read(
            "provider_runtime/publisher-source/native_g1_publisher_source_stage.v1.json"
        )
    )
    runtime_source = json.loads(
        archive.read(
            "provider_runtime/native_task_runtime_sources/native_task_runtime_source_packet.v1.json"
        )
    )
    expected_candidates = [
        "humanoidarena_dp_g1_dex3_sonic",
        "humanoidarena_pi05_g1_dex3_sonic",
        "humanoidarena_dp_g1_dex3_sonic_vision_navi",
        "humanoidarena_pi05_g1_dex3_sonic_vision_navi",
    ]
    if (
        manifest.get("schema_version") != "native_g1_provider_bundle.v1"
        or manifest.get("status") != "ready"
        or manifest.get("provider_bundle_kind") != "native_g1_development_campaign"
        or manifest.get("manifest_digest")
        != canonical_digest(manifest, digest_field="manifest_digest")
        or manifest.get("container_image") != NATIVE_TASK_ARENA_IMAGE
        or manifest.get("expected_output_filename")
        != "native_g1_provider_campaign_result.v1.json"
        or manifest.get("runtime_entrypoint")
        != "provider_runtime/run_adp_arena_provider_runtime.sh"
        or manifest.get("candidate_ids") != expected_candidates
        or manifest.get("campaign_plan_digest") != campaign.get("plan_digest")
        or manifest.get("publisher_source_receipt_digest") != source.get("receipt_digest")
        or (manifest.get("runtime_source_packet") or {}).get("packet_sha256")
        != runtime_source.get("packet_sha256")
        or (manifest.get("runtime_source_packet") or {}).get("embedded_in_provider_bundle")
        is not False
        or manifest.get("claim_ceiling") != "development_only"
        or manifest.get("provider_zero_required_after_return") is not True
    ):
        raise ValueError("native_g1_manifest_binding_invalid")
    return manifest
