"""Seal the exact Aura/SAGE/SimReady internal hybrid replacement control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


REQUEST_SCHEMA_VERSION = "adp009b_hybrid_replacement_seal_request.v1"
RECEIPT_SCHEMA_VERSION = "adp009b_hybrid_replacement_seal.v1"


class HybridReplacementSealError(ValueError):
    """The requested hybrid seal is incomplete, mismatched, or caller-asserted."""


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HybridReplacementSealError(f"json_object_required:{path.name}")
    return value


def _under(root: Path, relative: str) -> Path:
    root = root.expanduser().resolve()
    path = (root / relative).expanduser().resolve()
    if root not in path.parents:
        raise HybridReplacementSealError(f"path_outside_approved_root:{path}")
    return path


def _explicit_under(root: Path, path: Path) -> Path:
    root = root.expanduser().resolve()
    if not path.is_absolute():
        path = root / path
    path = path.expanduser().resolve()
    if path != root and root not in path.parents:
        raise HybridReplacementSealError(f"path_outside_approved_root:{path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _verify_file(path: Path, record: Mapping[str, Any], *, error: str) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise HybridReplacementSealError(error)


def _verify_canonical(
    value: Mapping[str, Any], *, field: str, error: str
) -> None:
    if value.get(field) != canonical_digest(value, digest_field=field):
        raise HybridReplacementSealError(error)


def materialize_hybrid_replacement_seal(
    *, request_path: Path, repo_root: Path, data_root: Path, output_path: Path
) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    request_path = _explicit_under(repo_root, request_path)
    output_path = _explicit_under(repo_root, output_path)
    request = _read(request_path)
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise HybridReplacementSealError("request_schema_invalid")
    if {"status", "admitted", "qualified", "complete"}.intersection(request):
        raise HybridReplacementSealError("caller_asserted_seal_forbidden")

    aura = _read(_under(repo_root, str(request["aura_human_review_receipt_path"])))
    replacement = _read(_under(repo_root, str(request["replacement_receipt_path"])))
    static = _read(_under(repo_root, str(request["static_validation_receipt_path"])))
    component_manifest = _read(
        _under(repo_root, str(request["exact_simready_component_manifest_path"]))
    )
    component_receipt = _read(
        _under(repo_root, str(request["exact_simready_component_receipt_path"]))
    )
    _verify_canonical(
        aura, field="receipt_digest", error="aura_human_review_digest_mismatch"
    )
    _verify_canonical(
        replacement, field="receipt_digest", error="replacement_receipt_digest_mismatch"
    )
    _verify_canonical(
        static, field="receipt_digest", error="static_validation_digest_mismatch"
    )
    _verify_canonical(
        component_manifest,
        field="manifest_digest",
        error="exact_simready_manifest_digest_mismatch",
    )
    _verify_canonical(
        component_receipt,
        field="receipt_digest",
        error="exact_simready_receipt_digest_mismatch",
    )

    if (
        aura.get("schema_version") != "adp009b_aura_human_visual_review.v1"
        or aura.get("status")
        != "human_accepted_visual_candidate_for_internal_hybrid_control"
        or (aura.get("observed_quality") or {}).get("human_visual_acceptance") is not True
        or aura.get("technical_admission") is not False
    ):
        raise HybridReplacementSealError("aura_human_review_not_accepted")
    composition = replacement.get("composition") or {}
    if (
        replacement.get("schema_version") != "adp009b_simready_replacement_receipt.v1"
        or replacement.get("status") != "composed_static_candidate"
        or composition.get("source_target_collider_active") is not False
        or composition.get("support_collider_active") is not True
        or composition.get("replacement_rigid_body_active") is not True
        or composition.get("replacement_collision_api_present") is not True
        or composition.get("unresolved_dependency_count") != 0
    ):
        raise HybridReplacementSealError("replacement_composition_invalid")
    if (
        static.get("status") != "statically_validated"
        or (static.get("checks") or {}).get("simready_foundation_profile_passed") is not True
        or (static.get("checks") or {}).get("usd_readback_passed") is not True
    ):
        raise HybridReplacementSealError("static_validation_not_passed")
    checks = component_receipt.get("checks") or {}
    observed = component_manifest.get("observed_evidence") or {}
    probes = observed.get("probe_summaries")
    if (
        component_receipt.get("role") != "exact_simready_object"
        or component_receipt.get("status") != "admitted"
        or component_receipt.get("component_manifest_digest")
        != component_manifest.get("manifest_digest")
        or not all(
            checks.get(key) is True
            for key in (
                "cad_materialization_receipt_digest_verified",
                "simready_foundation_profile_passed",
                "isaac_dynamic_probes_passed",
                "teardown_and_object_cleanup_verified",
                "usd_bytes_verified",
            )
        )
        or not isinstance(probes, list)
        or {row.get("probe") for row in probes if isinstance(row, Mapping)}
        != {"drop", "slide", "tip", "gripper"}
        or any(row.get("passed") is not True for row in probes)
        or observed.get("provider_zero_verified") is not True
        or observed.get("object_store_zero_verified") is not True
    ):
        raise HybridReplacementSealError("exact_simready_dynamic_admission_invalid")

    aura_scene = aura.get("scene") or {}
    replacement_scene = replacement.get("scene") or {}
    component_scene = component_manifest.get("scene_mapping") or {}
    if (
        aura_scene.get("publisher_scene_id") != "840313"
        or aura_scene.get("target_instance_id") != "ins160"
        or replacement_scene.get("publisher_scene_id") != "840313"
        or str(replacement_scene.get("target_instance_id")) != "160"
        or component_scene.get("publisher_scene_id") != "840313"
        or str(component_scene.get("interiorgs_instance_id")) != "160"
    ):
        raise HybridReplacementSealError("scene_or_target_identity_mismatch")
    if (
        (replacement.get("bindings") or {}).get("aura_execution_receipt_digest")
        != (aura.get("bindings") or {}).get("aura_execution_receipt_digest")
        or (replacement.get("bindings") or {}).get("exact_simready_asset", {}).get(
            "sha256"
        )
        != (static.get("usd") or {}).get("sha256")
    ):
        raise HybridReplacementSealError("appearance_or_asset_join_mismatch")

    _verify_file(
        _under(data_root, str(composition["relative_path"])),
        composition,
        error="composition_bytes_changed",
    )
    _verify_file(
        _under(repo_root, str((static.get("usd") or {})["relative_path"])),
        static["usd"],
        error="exact_simready_asset_bytes_changed",
    )

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009B",
        "status": "sealed_internal_hybrid_replacement_control",
        "scene": {
            "publisher_scene_id": "840313",
            "target_instance_id": "160",
            "target_semantic_label": replacement_scene["target_semantic_label"],
            "source_target_collider_active": False,
            "support_collider_active": True,
        },
        "bindings": {
            "aura_human_review_receipt_digest": aura["receipt_digest"],
            "aura_execution_receipt_digest": (aura["bindings"])[
                "aura_execution_receipt_digest"
            ],
            "replacement_receipt_digest": replacement["receipt_digest"],
            "static_validation_receipt_digest": static["receipt_digest"],
            "exact_simready_component_manifest_digest": component_manifest[
                "manifest_digest"
            ],
            "exact_simready_component_receipt_digest": component_receipt[
                "receipt_digest"
            ],
            "composition_sha256": composition["sha256"],
            "exact_simready_asset_sha256": static["usd"]["sha256"],
            "native_visual_review_receipt_digest": (aura["bindings"])[
                "native_visual_review_receipt_digest"
            ],
            "isaac_runtime_result_digest": observed["runtime_result_digest"],
        },
        "observed_gates": {
            "project_owner_visual_acceptance": True,
            "frozen_camera_count": (aura["observed_quality"])["view_count"],
            "static_simready_profile_passed": True,
            "source_collider_deactivated": True,
            "support_collider_preserved": True,
            "four_native_isaac_probes_passed": True,
            "provider_and_object_store_zero": True,
        },
        "adp009b_complete": False,
        "technical_inpainting_admitted": False,
        "physical_evidence": False,
        "claim_ceiling": "internal_noncommercial_hybrid_replacement_control_only",
        "claim_boundaries": {
            "aura_background_is_project_owner_accepted_visual_candidate": True,
            "native_object_layer_composite_is_not_full_scene_native_render": True,
            "hidden_background_truth_available": False,
            "simulation_is_not_physical_truth": True,
            "digital_twin": False,
        },
        "remaining_blockers": [
            "infusion_primary_adapter_checkpoint_license_missing",
            "full_native_hybrid_background_renderer_missing",
            "controlled_background_truth_missing",
        ],
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = materialize_hybrid_replacement_seal(
        request_path=args.request,
        repo_root=args.repo_root,
        data_root=args.data_root,
        output_path=args.output,
    )
    print(
        json.dumps(
            {"status": receipt["status"], "receipt_digest": receipt["receipt_digest"]}
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
