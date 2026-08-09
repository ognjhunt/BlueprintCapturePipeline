"""Canonical zero-retry Vast execution for the bounded ADP-009A Content Agents case."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import yaml
from pxr import Usd, UsdGeom, UsdPhysics

from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .paid_resource_admission import PaidResourceAdmissionGrant
from .public_scene_simready_native import materialize_native_probe
from .provider_runtime_bundle_contract import provider_runtime_contract_blockers
from .provider_bundle_rehearsal import (
    provider_bundle_rehearsal_blockers,
    rehearse_provider_bundle_entrypoint,
)
from .vast_provider_adapter import run_vast_provider_adapter
from .vast_session_budget_contract import attempt_estimated_cost, attempt_runtime_seconds
from .wam_provider_object_store import (
    cleanup_staged_wam_provider_objects,
    stage_wam_provider_bundle_object_store,
)


PROBE_KIND = "adp-usd-content-agents"
RESULT_SCHEMA_VERSION = "adp_content_agents_vast_run.v1"
SOURCE_COMMIT = "36dbf3f274f8e256637230a05a085853f65cc175"
SOURCE_TREE = "d36ddaed4c3ea44ab81c9f8178ab40d2eb0f8fe3"
SOURCE_VERSION = "0.5.2"
DEFAULT_IMAGE = (
    "docker.io/nvidia/cuda@"
    "sha256:cff3a0d82d2c2b47bab252d67fa9b34a20ef4c50781d98501b5c7367ea9afd10"
)
REFERENCE_IMAGE_SHA256 = (
    "sha256:80954198df572d782e095d8670e0d4e8ceea530c8fe53c8476a487d1aebe137f"
)
MATCH_V2_RECEIPT_RELATIVE_PATH = (
    "docs/arm_decision_proof_v1/manifests/"
    "adp009a_840313_canned_beverage_match_v2_receipt.v1.json"
)
MATCH_V2_REPLACEMENT_RECEIPT_RELATIVE_PATH = (
    "docs/arm_decision_proof_v1/manifests/"
    "adp009b_simready_replacement_match_v2_receipt.v1.json"
)
MATCH_V2_HUMAN_REVIEW_RELATIVE_PATH = (
    "docs/arm_decision_proof_v1/manifests/"
    "adp009b_simready_match_v2_human_review_receipt.v1.json"
)
ARTICULATED_V1_MANIFEST_RELATIVE_PATH = (
    "docs/arm_decision_proof_v1/manifests/"
    "second_scene_840796_deterministic_simready_candidate.v1.json"
)
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/content-agents"
_VAST_MUTATION_ENV = (
    "BLUEPRINT_ALLOW_VAST_API_CALLS",
    "BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH",
)
_VAST_SINGLE_ATTEMPT_ENV = "BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS"
_FORWARDED_SECRET_NAMES = (
    "OPENAI_API_KEY",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _under(path: Path, root: Path, *, error: str) -> Path:
    resolved = path.expanduser().resolve()
    root = root.expanduser().resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(error)
    return resolved


def _verified_canonical_receipt(path: Path, *, error: str) -> dict[str, Any]:
    receipt = _read_json(path)
    supplied = receipt.get("receipt_digest")
    if not receipt or supplied != canonical_digest(receipt, digest_field="receipt_digest"):
        raise ValueError(error)
    return receipt


def _resolve_input_variant(
    *,
    repo: Path,
    evidence_root: Path | None,
    reference_source: Path,
    variant: str,
) -> dict[str, Any]:
    assets = repo / "docs" / "arm_decision_proof_v1" / "assets"
    if variant == "control_v1":
        if _sha256(reference_source) != REFERENCE_IMAGE_SHA256:
            raise ValueError("adp_content_agents_reference_image_identity_mismatch")
        return {
            "usd_source": assets / "adp009a_840313_canned_beverage_control.usda",
            "config_prefix": "adp009a_content_agents_",
            "reference_image_sha256": REFERENCE_IMAGE_SHA256,
            "reference_image_authority": "blueprint_cad_render_not_interiorgs_dataset_bytes",
            "variant": variant,
        }
    if variant == "articulated_v1":
        # The 840796 articulated candidate is scene-derived, so its bytes stay
        # in the evidence root under the recorded rights while the repo keeps
        # only the digest-bound manifest. The Content Agents pass may add
        # SimReady materials and physics priors; it may never re-derive the
        # articulation, so admission requires the statically admitted receipt.
        if evidence_root is None:
            raise ValueError("adp_content_agents_articulated_evidence_root_missing")
        evidence = evidence_root.expanduser().resolve()
        manifest = _verified_canonical_receipt(
            _under(
                repo / ARTICULATED_V1_MANIFEST_RELATIVE_PATH,
                repo,
                error="adp_content_agents_articulated_manifest_outside_repo",
            ),
            error="adp_content_agents_articulated_manifest_invalid",
        )
        authoring = manifest.get("authoring") or {}
        relative = manifest.get("evidence_relative_paths") or {}
        if (
            manifest.get("schema_version")
            != "second_scene_deterministic_simready_candidate.v1"
            or manifest.get("publisher_scene_id") != "840796"
            or authoring.get("status") != "simready_candidate_statically_admitted"
            or not str(authoring.get("task_joint_prim_path") or "")
            or not str(manifest.get("topology_validation_receipt_digest") or "")
            or not str(manifest.get("physics_validation_receipt_digest") or "")
        ):
            raise ValueError(
                "adp_content_agents_articulated_candidate_receipt_not_eligible"
            )
        usd_source = _under(
            evidence / str(relative.get("candidate_usd") or ""),
            evidence,
            error="adp_content_agents_articulated_usd_outside_evidence",
        )
        expected_reference = _under(
            evidence / str(relative.get("reference_image") or ""),
            evidence,
            error="adp_content_agents_articulated_reference_outside_evidence",
        )
        if (
            not usd_source.is_file()
            or _sha256(usd_source) != authoring.get("output_usd_sha256")
            or not expected_reference.is_file()
            or reference_source != expected_reference
            or _sha256(reference_source) != manifest.get("reference_image_sha256")
        ):
            raise ValueError(
                "adp_content_agents_articulated_source_identity_mismatch"
            )
        return {
            "usd_source": usd_source,
            "config_prefix": "adp009d_content_agents_articulated_",
            "reference_image_sha256": manifest["reference_image_sha256"],
            "reference_image_authority": (
                "blueprint_render_of_sage_derived_articulated_candidate_not_"
                "interiorgs_dataset_bytes"
            ),
            "variant": variant,
            "candidate_receipt_digest": manifest["receipt_digest"],
            "task_joint_prim_path": authoring["task_joint_prim_path"],
            "topology_validation_receipt_digest": manifest[
                "topology_validation_receipt_digest"
            ],
            "physics_validation_receipt_digest": manifest[
                "physics_validation_receipt_digest"
            ],
        }
    if variant != "match_v2":
        raise ValueError("adp_content_agents_input_variant_invalid")
    if evidence_root is None:
        raise ValueError("adp_content_agents_match_v2_evidence_root_missing")
    evidence = evidence_root.expanduser().resolve()
    control_path = _under(
        repo / MATCH_V2_RECEIPT_RELATIVE_PATH,
        repo,
        error="adp_content_agents_match_v2_control_receipt_outside_repo",
    )
    control = _verified_canonical_receipt(
        control_path, error="adp_content_agents_match_v2_control_receipt_invalid"
    )
    checks = control.get("checks") or {}
    if (
        control.get("control_id")
        != "adp009a-840313-canned-beverage-multiview-match-v2"
        or control.get("status") != "prepared_for_independent_validation"
        or checks.get("cad_inspection_passed") is not True
        or checks.get("target_dimensions_derived_not_caller_asserted") is not True
        or (control.get("visual_match_evidence") or {}).get(
            "projected_scale_and_pose_gate_passed"
        )
        is not True
    ):
        raise ValueError("adp_content_agents_match_v2_control_receipt_not_eligible")
    usd = control.get("usd") or {}
    usd_source = _under(
        repo / str(usd.get("relative_path") or ""),
        repo,
        error="adp_content_agents_match_v2_usd_outside_repo",
    )
    snapshot = (control.get("cad_evidence") or {}).get("snapshot") or {}
    expected_reference = _under(
        evidence / str(snapshot.get("relative_path") or ""),
        evidence,
        error="adp_content_agents_match_v2_reference_outside_evidence",
    )
    if (
        not usd_source.is_file()
        or _sha256(usd_source) != usd.get("sha256")
        or not expected_reference.is_file()
        or reference_source != expected_reference
        or _sha256(reference_source) != snapshot.get("sha256")
    ):
        raise ValueError("adp_content_agents_match_v2_source_identity_mismatch")
    replacement = _verified_canonical_receipt(
        _under(
            repo / MATCH_V2_REPLACEMENT_RECEIPT_RELATIVE_PATH,
            repo,
            error="adp_content_agents_match_v2_replacement_receipt_outside_repo",
        ),
        error="adp_content_agents_match_v2_replacement_receipt_invalid",
    )
    human_review = _verified_canonical_receipt(
        _under(
            repo / MATCH_V2_HUMAN_REVIEW_RELATIVE_PATH,
            repo,
            error="adp_content_agents_match_v2_human_review_outside_repo",
        ),
        error="adp_content_agents_match_v2_human_review_invalid",
    )
    if (
        replacement.get("status") != "composed_static_candidate"
        or (replacement.get("bindings") or {}).get("simready_control_receipt_digest")
        != control.get("receipt_digest")
        or human_review.get("status") != "human_accepted_for_native_validation"
        or human_review.get("technical_admission") is not False
        or (human_review.get("artifact_chain") or {}).get(
            "replacement_receipt_digest"
        )
        != replacement.get("receipt_digest")
    ):
        raise ValueError("adp_content_agents_match_v2_approval_chain_invalid")
    return {
        "usd_source": usd_source,
        "config_prefix": "adp009a_content_agents_match_v2_",
        "reference_image_sha256": snapshot["sha256"],
        "reference_image_authority": (
            "blueprint_cad_snapshot_bound_to_human_approved_match_v2_not_"
            "interiorgs_dataset_bytes"
        ),
        "variant": variant,
        "control_receipt_digest": control["receipt_digest"],
        "replacement_receipt_digest": replacement["receipt_digest"],
        "human_review_receipt_digest": human_review["receipt_digest"],
        "replacement_receipt": replacement,
    }


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_executable(path: Path, source: Path) -> None:
    shutil.copy2(source, path)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _materialize_articulated_content_agents_input(
    source: Path, destination: Path
) -> dict[str, Any]:
    """Normalize the articulated candidate for NVIDIA 0.5.2 without re-deriving it.

    The can control needs a grasp-curve extent fix and a render-purpose clear on
    one visual. The articulated candidate has neither; what it does need is
    every mesh readable at default purpose so the agents' bounds and dataset
    stages work, with the articulation left exactly as authored.
    """

    shutil.copy2(source, destination)
    stage = Usd.Stage.Open(str(destination))
    if stage is None or not stage.GetDefaultPrim().IsValid():
        raise ValueError("adp_content_agents_input_default_prim_invalid")
    cleared: list[str] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        if mesh.ComputePurpose() != UsdGeom.Tokens.default_:
            mesh.GetPurposeAttr().Clear()
            cleared.append(str(prim.GetPath()))
    stage.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(destination))
    if reopened is None:
        raise ValueError("adp_content_agents_input_reopen_failed")
    joints = [prim for prim in reopened.Traverse() if prim.IsA(UsdPhysics.Joint)]
    roots = [
        prim
        for prim in reopened.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    rigid_bodies = [
        prim for prim in reopened.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bounds = cache.ComputeWorldBound(reopened.GetDefaultPrim()).ComputeAlignedRange()
    if bounds.IsEmpty() or not joints or len(roots) != 1:
        raise ValueError("adp_content_agents_input_articulation_invalid")
    for prim in reopened.Traverse():
        if prim.IsA(UsdGeom.Mesh) and UsdGeom.Mesh(prim).ComputePurpose() != (
            UsdGeom.Tokens.default_
        ):
            raise ValueError("adp_content_agents_input_default_purpose_bbox_invalid")
    return {
        "source_input_usd_sha256": _sha256(source),
        "normalized_input_usd_sha256": _sha256(destination),
        "transformations": [
            "clear_non_default_mesh_purposes_for_nvidia_0_5_2_bbox",
        ],
        "cleared_purpose_prims": sorted(cleared),
        "default_purpose_bbox_nonempty": True,
        "articulation_preserved": True,
        "joint_count": len(joints),
        "rigid_body_count": len(rigid_bodies),
        "articulation_root_count": len(roots),
    }


def _materialize_content_agents_input(
    source: Path, destination: Path, *, variant: str = "control_v1"
) -> dict[str, Any]:
    """Derive the exact NVIDIA-compatible USD without mutating canonical bytes."""

    if variant == "articulated_v1":
        return _materialize_articulated_content_agents_input(source, destination)
    shutil.copy2(source, destination)
    stage = Usd.Stage.Open(str(destination))
    if stage is None or stage.GetDefaultPrim().GetPath() != "/canned_beverage":
        raise ValueError("adp_content_agents_input_default_prim_invalid")
    visual = UsdGeom.Mesh.Get(stage, "/canned_beverage/visuals/body")
    if not visual or visual.ComputePurpose() != UsdGeom.Tokens.render:
        raise ValueError("adp_content_agents_input_visual_purpose_invalid")
    visual.GetPurposeAttr().Clear()
    grasp = UsdGeom.BasisCurves.Get(stage, "/canned_beverage/grasp_identifier_01")
    computed_extent = UsdGeom.Boundable.ComputeExtentFromPlugins(
        grasp, Usd.TimeCode.Default()
    )
    if not computed_extent:
        raise ValueError("adp_content_agents_input_grasp_extent_unavailable")
    grasp.GetExtentAttr().Set(computed_extent)
    stage.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(destination))
    if reopened is None:
        raise ValueError("adp_content_agents_input_reopen_failed")
    normalized_visual = UsdGeom.Mesh.Get(reopened, "/canned_beverage/visuals/body")
    joints = [prim for prim in reopened.Traverse() if prim.IsA(UsdPhysics.Joint)]
    rigid_bodies = [
        prim for prim in reopened.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    roots = [
        prim
        for prim in reopened.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    bounds = cache.ComputeWorldBound(normalized_visual.GetPrim()).ComputeAlignedRange()
    if normalized_visual.ComputePurpose() != UsdGeom.Tokens.default_ or bounds.IsEmpty():
        raise ValueError("adp_content_agents_input_default_purpose_bbox_invalid")
    return {
        "source_input_usd_sha256": _sha256(source),
        "normalized_input_usd_sha256": _sha256(destination),
        "transformations": [
            "clear_visual_render_purpose_to_usd_default_for_nvidia_0_5_2_bbox",
            "recompute_grasp_identifier_extent_from_curve_width",
        ],
        "visual_prim": "/canned_beverage/visuals/body",
        "visual_purpose": "default",
        "default_purpose_bbox_nonempty": True,
        "articulation_preserved": not joints and not roots,
        "joint_count": len(joints),
        "rigid_body_count": len(rigid_bodies),
        "articulation_root_count": len(roots),
    }


def _derive_joint_agent_plan(
    *, input_variant: str, input_normalization: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive Joint Agent applicability from the normalized USD inventory."""

    joint_count = input_normalization.get("joint_count")
    rigid_body_count = input_normalization.get("rigid_body_count")
    root_count = input_normalization.get("articulation_root_count")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (joint_count, rigid_body_count, root_count)
    ):
        raise ValueError("adp_content_agents_joint_agent_inventory_invalid")
    if joint_count:
        if root_count != 1 or rigid_body_count < 2:
            raise ValueError("adp_content_agents_joint_agent_articulation_invalid")
        reason = "preexisting_articulation_preserved_by_enrichment_pass"
        single_rigid_body = False
    else:
        if root_count != 0 or rigid_body_count != 1:
            raise ValueError("adp_content_agents_joint_agent_rigid_input_invalid")
        reason = "single_rigid_body_has_no_articulation_task"
        single_rigid_body = True
    return {
        "planned": False,
        "executed_by_content_agents_bundle": False,
        "reason": reason,
        "input_variant": input_variant,
        "input_joint_count": joint_count,
        "input_rigid_body_count": rigid_body_count,
        "input_articulation_root_count": root_count,
        "joint_agent_inapplicable_single_rigid_body": single_rigid_body,
    }


def _validate_remote_configs(
    *, source: Path, config_sources: Mapping[str, Path]
) -> None:
    payloads = {
        name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for name, path in config_sources.items()
    }
    material_path = (
        source
        / "apps/material_agent/data/materials/material_libs_default/materials.yaml"
    )
    material = dict(payloads.get("material_agent.yaml") or {})
    texture = dict(payloads.get("texture_agent.yaml") or {})
    physics = dict(payloads.get("physics_agent.yaml") or {})
    texture_config = dict(texture.get("texture") or {})
    # Exactly one material spec per config; its key is scene-specific.
    texture_material_specs = dict(texture.get("material_textures") or {})
    texture_spec = (
        dict(next(iter(texture_material_specs.values())))
        if len(texture_material_specs) == 1
        else {}
    )
    physics_steps = dict(physics.get("steps") or {})
    material_steps = dict(material.get("steps") or {})
    material_predict = dict(material_steps.get("predict") or {})
    material_vlm = dict(material_predict.get("vlm") or {})
    material_llm = dict(material_predict.get("llm") or {})
    material_validation = dict(material_steps.get("validate_input") or {})
    identify_vlm = dict((physics_steps.get("identify_asset") or {}).get("vlm") or {})
    identify_enabled = (physics_steps.get("identify_asset") or {}).get("enabled")
    predict_vlm = dict((physics_steps.get("predict") or {}).get("vlm") or {})
    material_rendering_modes = set(
        ((material_steps.get("build_dataset_usd") or {}).get("renderer") or {})
        .get("rendering_modes", {})
    )
    physics_dataset = dict(physics_steps.get("build_dataset_usd") or {})
    physics_rendering_modes = set(
        (physics_dataset.get("renderer") or {}).get("rendering_modes", {})
    )
    # Target prims are scene-specific and bound by digests elsewhere; this
    # contract guards known paid-runtime failure modes. It therefore requires
    # the texture UV scope, declared targets, and the single material spec to
    # name the same non-empty absolute prims rather than one scene's paths.
    texture_targets = texture_config.get("uv_target_prim_paths")
    if len(texture_material_specs) != 1:
        raise ValueError("adp_content_agents_remote_config_contract_invalid")
    texture_targets_consistent = (
        isinstance(texture_targets, list)
        and bool(texture_targets)
        and all(
            isinstance(item, str) and item.startswith("/") for item in texture_targets
        )
        and texture.get("target_prims") == texture_targets
        and texture_spec.get("prim_paths") == texture_targets
        and isinstance(texture_spec.get("material_path"), str)
        and str(texture_spec.get("material_path")).startswith("/")
    )
    if (
        not material_path.is_file()
        or (material.get("materials") or {}).get("path")
        != "../content_agents_source/apps/material_agent/data/materials/"
        "material_libs_default/materials.yaml"
        or not texture_targets_consistent
        or texture_config.get("image_gen")
        != {"backend": "openai", "model": "gpt-image-1"}
        or material_validation.get("on_failure") != "warn"
        or (material_steps.get("validate_output") or {}).get("on_failure") != "warn"
        or material_vlm.get("backend") != "openai"
        or material_vlm.get("model") != "gpt-4.1"
        or material_llm.get("backend") != "openai"
        or material_llm.get("model") != "gpt-4.1"
        or identify_enabled is not False
        or identify_vlm.get("backend") != "openai"
        or identify_vlm.get("model") != "gpt-4.1"
        or predict_vlm.get("backend") != "openai"
        or predict_vlm.get("model") != "gpt-4.1"
        or material_rendering_modes != {"composition", "prim_only"}
        or physics_rendering_modes != {"composition", "prim_only"}
        or (physics_dataset.get("prim_filters") or {}).get("skip_invisible") is not True
    ):
        raise ValueError("adp_content_agents_remote_config_contract_invalid")


def _materialize_remote_configs(
    *,
    config_sources: Mapping[str, Path],
    destination: Path,
    variant: str,
) -> dict[str, str]:
    """Copy v1 configs or deterministically derive the approved v2 challenger."""

    config_hashes: dict[str, str] = {}
    for name, path in config_sources.items():
        target = destination / name
        if variant in {"control_v1", "articulated_v1"}:
            # The articulated configs are authored per-variant and checked in
            # already; copying them keeps the shipped bytes identical to the
            # reviewed source instead of deriving a second representation.
            shutil.copy2(path, target)
        else:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            project = payload["project"]
            project["name"] = project["name"] + "_match_v2"
            project["session_id"] = project["session_id"] + "_match_v2"
            project["description"] = project["description"].replace(
                "CAD-derived can control", "human-approved multiview-matched v2 can"
            ).replace(
                "outputs remain authored priors",
                "human-approved v2 input; outputs remain authored priors",
            )
            if name == "material_agent.yaml":
                payload["steps"]["build_dataset_prepare_dataset"]["prompts"][
                    "vlm_user"
                ] = (
                    "Classify the rendered pale-mint beverage container appearance. "
                    "Do not assume aluminum, glass, plastic, or transparency from "
                    "shape or prompt alone."
                )
            elif name == "texture_agent.yaml":
                payload["material_textures"]["green_can"]["prompt"] = (
                    "pale mint green clean non-branded beverage container surface, "
                    "subtle vertical shading, no text or logo"
                )
            target.write_text(
                yaml.safe_dump(payload, sort_keys=False, width=100), encoding="utf-8"
            )
        config_hashes[name] = _sha256(target)
    return config_hashes


def _deterministic_zip(source_root: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w") as archive:
        for path in sorted(source_root.rglob("*")):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(
                path.relative_to(source_root.parent).as_posix(),
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.create_system = 3
            info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
            archive.writestr(
                info,
                path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def build_content_agents_vast_bundle(
    *,
    repo_root: str | Path,
    content_agents_root: str | Path,
    reference_image_path: str | Path,
    job_dir: str | Path,
    input_variant: str = "control_v1",
    evidence_root: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build one immutable bundle with explicit public-dataset byte accounting.

    The v1 control carries no dataset bytes. The approved v2 native probe carries
    the exact public CC-BY-NC SAGE collision companion, but never gated
    InteriorGS source bytes or Aura/InteriorGS appearance frames.
    """

    repo = Path(repo_root).expanduser().resolve()
    source = Path(content_agents_root).expanduser().resolve()
    reference_source = Path(reference_image_path).expanduser().resolve()
    evidence = Path(evidence_root) if evidence_root is not None else None
    job = Path(job_dir).expanduser().resolve()
    if job.exists() and any(job.iterdir()):
        raise ValueError("adp_content_agents_bundle_job_dir_not_empty")
    runtime = job / "provider_runtime"
    ensure_dir(runtime / "configs")
    ensure_dir(runtime / "input")
    head = _git(source, "rev-parse", "HEAD")
    tree = _git(source, "rev-parse", "HEAD^{tree}")
    dirty = bool(_git(source, "status", "--porcelain"))
    if head != SOURCE_COMMIT or tree != SOURCE_TREE or dirty:
        raise ValueError("adp_content_agents_source_identity_mismatch")
    if not reference_source.is_file() or reference_source.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("adp_content_agents_reference_image_identity_mismatch")
    variant = _resolve_input_variant(
        repo=repo,
        evidence_root=evidence,
        reference_source=reference_source,
        variant=input_variant,
    )

    source_zip = runtime / "content_agents_source.zip"
    subprocess.run(
        ["git", "-C", str(source), "archive", "--format=zip", f"--output={source_zip}", "HEAD"],
        check=True,
    )
    scripts = repo / "scripts"
    _write_executable(
        runtime / "run_adp_content_agents_provider_runtime.sh",
        scripts / "run_adp_content_agents_provider_runtime.sh",
    )
    shutil.copy2(
        scripts / "adp_content_agents_provider_runner.py",
        runtime / "adp_content_agents_provider_runner.py",
    )
    shutil.copy2(
        repo / "src/blueprint_pipeline/provider_archive.py",
        runtime / "provider_archive.py",
    )
    native_probe: dict[str, Any] | None = None
    if variant["variant"] == "match_v2":
        native_probe = materialize_native_probe(
            evidence_root=evidence,
            destination=runtime / "native",
            replacement_receipt=variant["replacement_receipt"],
        )
        shutil.copy2(
            scripts / "run_ovrtx_preflight_worker.py",
            runtime / "native" / "run_ovrtx_preflight_worker.py",
        )
        shutil.copy2(
            scripts / "run_ovphysx_preflight_worker.py",
            runtime / "native" / "run_ovphysx_preflight_worker.py",
        )
    assets = repo / "docs" / "arm_decision_proof_v1" / "assets"
    config_prefix = str(variant["config_prefix"])
    if config_prefix == "adp009d_content_agents_articulated_":
        config_sources = {
            f"{agent}_agent.yaml": assets / f"{config_prefix}{agent}.vast.yaml"
            for agent in ("material", "texture", "physics")
        }
    else:
        config_sources = {
            "material_agent.yaml": assets / "adp009a_content_agents_material.vast.yaml",
            "texture_agent.yaml": assets / "adp009a_content_agents_texture.vast.yaml",
            "physics_agent.yaml": assets / "adp009a_content_agents_physics.vast.yaml",
        }
    config_hashes = _materialize_remote_configs(
        config_sources=config_sources,
        destination=runtime / "configs",
        variant=str(variant["variant"]),
    )
    runtime_configs = {
        name: runtime / "configs" / name for name in config_sources
    }
    _validate_remote_configs(source=source, config_sources=runtime_configs)
    usd_source = Path(variant["usd_source"])
    # Each variant's configs reference their own input filenames, so the
    # runtime input names follow the variant rather than the 840313 control.
    if variant["variant"] == "articulated_v1":
        runtime_usd_name = "adp009d_840796_articulated_refrigerator_candidate.usda"
        reference_name = (
            "adp009d_840796_articulated_refrigerator_candidate_reference.png"
        )
    else:
        runtime_usd_name = "adp009a_840313_canned_beverage_control.usda"
        reference_name = "adp009a_840313_canned_beverage_control_reference.png"
    input_normalization = _materialize_content_agents_input(
        usd_source,
        runtime / "input" / runtime_usd_name,
        variant=str(variant["variant"]),
    )
    joint_agent_plan = _derive_joint_agent_plan(
        input_variant=str(variant["variant"]),
        input_normalization=input_normalization,
    )
    shutil.copy2(reference_source, runtime / "input" / reference_name)

    entrypoint = runtime / "run_adp_content_agents_provider_runtime.sh"
    runner = runtime / "adp_content_agents_provider_runner.py"
    blockers = provider_runtime_contract_blockers(
        provider_bundle_kind="adp_content_agents",
        entrypoint_text=entrypoint.read_text(encoding="utf-8"),
        runner_text=runner.read_text(encoding="utf-8"),
    )
    generated = generated_at or utc_now_iso()
    readiness = {
        "schema_version": "adp_content_agents_provider_bundle.v1",
        "generated_at": generated,
        "status": "ready" if not blockers else "blocked",
        "source_repository": "https://github.com/NVIDIA-Omniverse/usd-content-agents",
        "source_commit": head,
        "source_tree": tree,
        "source_version": SOURCE_VERSION,
        "container_image": DEFAULT_IMAGE,
        "container_platform": "linux/amd64",
        "source_archive_sha256": _sha256(source_zip),
        "input_usd_sha256": input_normalization["normalized_input_usd_sha256"],
        "input_usd_normalization": input_normalization,
        "input_variant": variant["variant"],
        "input_variant_bindings": {
            key: value
            for key, value in variant.items()
            if key.endswith("_receipt_digest")
        },
        "reference_image_sha256": variant["reference_image_sha256"],
        "reference_image_authority": variant["reference_image_authority"],
        "runtime_entrypoint": "provider_runtime/run_adp_content_agents_provider_runtime.sh",
        "remote_config_contract_validated": True,
        "remote_config_sha256": config_hashes,
        "expected_output_filename": "adp_content_agents_vast_result.json",
        "material_agent_planned": True,
        "texture_agent_planned": True,
        "physics_agent_planned": True,
        "validation_agent_planned": True,
        "native_ovrtx_exact_camera_probe_planned": native_probe is not None,
        "native_ovphysx_drop_contact_settle_planned": native_probe is not None,
        "native_probe": native_probe,
        "gated_interiorgs_source_bytes_included": False,
        "public_sage_collision_bytes_included": native_probe is not None,
        "public_sage_collision_license": (
            "CC-BY-NC-4.0" if native_probe is not None else None
        ),
        "allowed_use_ceiling": (
            "internal_noncommercial_validation"
            if native_probe is not None
            else "blueprint_owned_control"
        ),
        "runtime_input_binding": {
            "relative_path": f"input/{runtime_usd_name}",
            "sha256": input_normalization["normalized_input_usd_sha256"],
        },
        "joint_agent_plan": joint_agent_plan,
        "joint_agent_inapplicable_single_rigid_body": joint_agent_plan[
            "joint_agent_inapplicable_single_rigid_body"
        ],
        "execution_role": "optional_construction_enrichment",
        "failure_blocks_deterministic_asset_construction": False,
        "failure_blocks_native_simulator_qualification": False,
        "agent_output_is_simready_authority": False,
        "deterministic_usd_construction_remains_primary": True,
        "local_bundle_ready_for_remote_staging": not blockers,
        "provider_zero_required_after_return": True,
        "retry_cap": 0,
        "blockers": blockers,
        "raw_secret_values_recorded": False,
    }
    write_json(runtime / "adp_content_agents_provider_manifest.json", readiness)
    bundle_path = job / "adp_content_agents_provider_runtime_bundle.zip"
    _deterministic_zip(runtime, bundle_path)
    rehearsal = rehearse_provider_bundle_entrypoint(
        bundle_path=bundle_path,
        entrypoint_relative_path=(
            "provider_runtime/run_adp_content_agents_provider_runtime.sh"
        ),
        evidence_path=job / "adp_content_agents_exact_bundle_rehearsal.json",
    )
    receipt = {
        **readiness,
        "bundle_path": str(bundle_path),
        "bundle_sha256": _sha256(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
        "exact_bundle_entrypoint_rehearsal": rehearsal,
    }
    write_json(job / "adp_content_agents_bundle_receipt.json", receipt)
    return receipt


def _remaining_minutes(
    *, job: Path, hard_cap_usd: float, hard_ttl_seconds: int, max_hourly_rate_usd: float
) -> int:
    ledger = _read_json(job / "adp_content_agents_vast_session_budget.json")
    attempts = [row for row in ledger.get("attempts") or [] if isinstance(row, Mapping)]
    prior_seconds = sum(attempt_runtime_seconds(row) for row in attempts)
    prior_cost = sum(attempt_estimated_cost(row) for row in attempts)
    runtime_minutes = math.floor(max(0.0, hard_ttl_seconds - prior_seconds) / 60.0)
    spend_minutes = math.floor(
        max(0.0, hard_cap_usd - prior_cost) * 60.0 / max_hourly_rate_usd
    )
    return max(0, min(runtime_minutes, spend_minutes))


def _extract(path: Path, destination: Path) -> dict[str, Any]:
    blockers: list[str] = []
    if not path.is_file():
        return {"status": "blocked", "blockers": ["content_agents_provider_output_zip_missing"]}
    if destination.exists() and any(destination.iterdir()):
        return {
            "status": "blocked",
            "blockers": ["content_agents_provider_output_destination_not_empty"],
        }
    ensure_dir(destination)
    root = destination.resolve()
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if root not in target.parents and target != root:
                    blockers.append("content_agents_provider_output_zip_path_traversal")
            if not blockers:
                archive.extractall(destination)
    except (OSError, zipfile.BadZipFile):
        blockers.append("content_agents_provider_output_zip_invalid")
    result_path = destination / "adp_content_agents_vast_result.json"
    execution = _read_json(result_path)
    if not execution:
        blockers.append("content_agents_provider_result_missing")
    return {
        "status": "completed" if not blockers else "blocked",
        "result_path": str(result_path),
        "execution": execution,
        "blockers": sorted(set(blockers)),
    }


def _model_secret() -> str:
    for name in _FORWARDED_SECRET_NAMES:
        value = str(os.getenv(name) or "").strip()
        if value:
            return value
    path = Path("~/.blueprint-secrets/openai_api_key").expanduser()
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


@contextmanager
def _authority_environment():
    names = (
        *_VAST_MUTATION_ENV,
        _VAST_SINGLE_ATTEMPT_ENV,
        *_FORWARDED_SECRET_NAMES,
        "BLUEPRINT_VAST_FORWARD_SECRET_ENV_VARS",
    )
    previous = {name: os.environ.get(name) for name in names}
    secret = _model_secret()
    if not secret:
        raise ValueError("adp_content_agents_openai_secret_missing")
    try:
        for name in _VAST_MUTATION_ENV:
            os.environ[name] = "1"
        os.environ[_VAST_SINGLE_ATTEMPT_ENV] = "0"
        for name in _FORWARDED_SECRET_NAMES:
            os.environ[name] = secret
        os.environ["BLUEPRINT_VAST_FORWARD_SECRET_ENV_VARS"] = ",".join(
            _FORWARDED_SECRET_NAMES
        )
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_content_agents_vast(
    *,
    job_dir: str | Path,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    prepared_bundle: Mapping[str, Any],
    max_hourly_rate_usd: float = 1.0,
    hard_cap_usd: float = 3.0,
    hard_ttl_seconds: int = 7200,
    public_image: str = DEFAULT_IMAGE,
) -> dict[str, Any]:
    """Run one Content Agents attempt and always require provider-zero afterward."""

    job = Path(job_dir).expanduser().resolve()
    ensure_dir(job)
    bundle = dict(prepared_bundle)
    if public_image != DEFAULT_IMAGE:
        raise ValueError("adp_content_agents_container_image_not_frozen")
    bundle_path = Path(str(bundle.get("bundle_path") or "")).resolve()
    if (
        bundle.get("status") != "ready"
        or not bundle_path.is_file()
        or _sha256(bundle_path) != bundle.get("bundle_sha256")
        or provider_bundle_rehearsal_blockers(
            bundle.get("exact_bundle_entrypoint_rehearsal"),
            bundle_sha256=str(bundle.get("bundle_sha256") or ""),
            entrypoint_relative_path=(
                "provider_runtime/run_adp_content_agents_provider_runtime.sh"
            ),
        )
    ):
        raise ValueError("adp_content_agents_prepared_bundle_binding_invalid")
    if not execute:
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "dry_run_ready",
            "bundle": bundle,
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "blockers": [],
        }
        write_json(job / "adp_content_agents_vast_result.json", result)
        return result
    if paid_resource_admission_grant is None:
        raise ValueError("adp_content_agents_paid_resource_admission_grant_missing")

    remaining_minutes = _remaining_minutes(
        job=job,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        max_hourly_rate_usd=max_hourly_rate_usd,
    )
    if remaining_minutes < 45:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "blockers": ["adp_content_agents_budget_below_minimum_live_window"],
        }
    staging_dir = job / "object_store_staging"
    staging = stage_wam_provider_bundle_object_store(
        job_dir=staging_dir,
        bundle_path=str(bundle_path),
        key_prefix=DEFAULT_KEY_PREFIX,
        expiration_seconds=max(hard_ttl_seconds + 1800, 18_000),
    )
    if staging.get("status") != "completed":
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "blockers": staging.get("blockers") or ["content_agents_object_store_staging_blocked"],
        }
    provider_run = job / "vast_provider_run"
    output_zip = provider_run / "vast_provider_runtime_output.zip"
    adapter: dict[str, Any] = {}
    try:
        with _authority_environment():
            adapter = run_vast_provider_adapter(
                job_dir=provider_run,
                mode="live-startup-probe",
                allow_vast_api_call=True,
                allow_instance_launch=True,
                max_hourly_rate=max_hourly_rate_usd,
                target_spend_usd=hard_cap_usd,
                hard_cap_usd=hard_cap_usd,
                max_live_minutes=remaining_minutes,
                session_max_live_minutes=hard_ttl_seconds // 60,
                public_image=public_image,
                isaac_image=public_image,
                ngc_image_login_mode="never",
                provider_bundle=bundle_path,
                provider_bundle_url=(staging_dir / "provider_bundle_url.txt").read_text().strip(),
                provider_output_put_url=(staging_dir / "provider_output_put_url.txt").read_text().strip(),
                provider_output_get_url=(staging_dir / "provider_output_get_url.txt").read_text().strip(),
                provider_runtime_output_zip=output_zip,
                enable_isaac_smoke=False,
                enable_blueprint_bundle=True,
                provider_bundle_kind="adp_content_agents",
                vast_launch_mode="ssh_direct",
                allow_cold_isaac_image_pull=False,
                disk_gb=64,
                min_gpu_ram_mb=24_000,
                poll_interval_seconds=15,
                startup_timeout_seconds=remaining_minutes * 60,
                heartbeat_no_progress_seconds=1800,
                session_budget_ledger_path=job / "adp_content_agents_vast_session_budget.json",
                verify_staging_urls=True,
                require_known_supported_isaac_driver=False,
                preferred_gpu_keywords=("RTX 4090", "RTX A6000", "L40S", "A100"),
                prefer_isaac_rt=False,
                instance_label_prefix="blueprint-adp-content-agents-",
                forward_hf_token=False,
                paid_resource_admission_grant=paid_resource_admission_grant,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        adapter = {
            "status": "blocked",
            "blockers": [f"adp_content_agents_vast_adapter_failed:{type(exc).__name__}"],
            "raw_secret_values_recorded": False,
        }
    finally:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
    extracted = _extract(output_zip, job / "immutable_execution")
    execution = dict(extracted.get("execution") or {})
    teardown = _read_json(provider_run / "vast_teardown_manifest.json")
    blockers = list(adapter.get("blockers") or []) + list(extracted.get("blockers") or [])
    if execution.get("status") != "completed":
        blockers.extend(execution.get("blockers") or ["content_agents_full_execution_not_completed"])
    if teardown.get("continuing_spend_from_this_run") is not False:
        blockers.append("content_agents_vast_provider_zero_not_proven")
    if cleanup.get("all_objects_absent") is not True:
        blockers.append("content_agents_object_store_provider_zero_not_proven")
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "completed" if not blockers else "blocked",
        "bundle_sha256": bundle["bundle_sha256"],
        "execution_result_path": extracted.get("result_path"),
        "adapter_result_path": str(provider_run / "vast_provider_adapter_result.json"),
        "teardown_manifest_path": str(provider_run / "vast_teardown_manifest.json"),
        "estimated_cost_usd": adapter.get("estimated_cost_usd"),
        "hard_cap_usd": hard_cap_usd,
        "hard_ttl_seconds": hard_ttl_seconds,
        "retry_cap": 0,
        "continuing_spend_from_this_run": teardown.get("continuing_spend_from_this_run"),
        "all_staged_objects_absent": cleanup.get("all_objects_absent"),
        "blockers": sorted(set(str(item) for item in blockers if str(item))),
        "raw_secret_values_recorded": False,
    }
    write_json(job / "adp_content_agents_vast_result.json", result)
    return result


__all__ = [
    "DEFAULT_IMAGE",
    "PROBE_KIND",
    "REFERENCE_IMAGE_SHA256",
    "build_content_agents_vast_bundle",
    "run_content_agents_vast",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the immutable ADP-009A Content Agents Vast bundle."
    )
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--content-agents-root", required=True)
    parser.add_argument("--reference-image", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument(
        "--input-variant", choices=("control_v1", "match_v2"), default="control_v1"
    )
    parser.add_argument("--evidence-root")
    args = parser.parse_args(argv)
    receipt = build_content_agents_vast_bundle(
        repo_root=args.repo_root,
        content_agents_root=args.content_agents_root,
        reference_image_path=args.reference_image,
        job_dir=args.job_dir,
        input_variant=args.input_variant,
        evidence_root=args.evidence_root,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt.get("status") == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
