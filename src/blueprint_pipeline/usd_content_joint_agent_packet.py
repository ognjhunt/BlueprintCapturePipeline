"""Build a fail-closed NVIDIA USD Content Agents Joint Agent packet.

The packet binds an exact released-code checkout and an exact locally derived
USD source asset.  It deliberately stops before renderer/model disclosure when
that disclosure or paid execution lacks authority.  A dry-run configuration is
still useful evidence: it makes component splitting, model identities, retry
policy, topology adapter, and expected outputs reviewable without uploading
scene-derived bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import yaml

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "usd_content_joint_agent_packet.v1"
JOINT_AGENT_IDENTITY = {
    "repository": "https://github.com/NVIDIA-Omniverse/usd-content-agents",
    "tag": "v0.5.2",
    "version": "0.5.2",
    "commit": "36dbf3f274f8e256637230a05a085853f65cc175",
    "license": "Apache-2.0",
    "files": {
        "VERSION.md": "sha256:26e0ab83aa1ff6bd0ae3882787a92663aa19db128e1241ffc99375fb9ca0c121",
        "LICENSE": "sha256:ef19b519dfed322eb3eb05a77c764c0caf9a7fad1189d0f0a30aedca95f0dc76",
        "apps/joint_agent/README.md": "sha256:c8c9aae025a60bfeba8aabad188a87262f6d5d7eff91fb7f80fd544d24bed79b",
        "apps/joint_agent/configs/byoa_joint_rigger.yaml": (
            "sha256:272500444f68509a36d2120669ab91387e08ce91da3af23908fa41c0cb035e35"
        ),
    },
}


class JointAgentPacketError(ValueError):
    """Stable, sorted construction-packet failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def inspect_joint_agent_checkout(
    checkout: str | Path,
    *,
    expected_identity: Mapping[str, Any] = JOINT_AGENT_IDENTITY,
) -> dict[str, Any]:
    """Read the exact git and file identity of a Joint Agent checkout."""

    root = Path(checkout).expanduser().resolve()
    errors: list[str] = []
    if not root.is_dir():
        raise JointAgentPacketError(["joint_agent_checkout_missing"])
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise JointAgentPacketError(["joint_agent_checkout_git_identity_missing"]) from exc
    if commit != expected_identity.get("commit"):
        errors.append("joint_agent_checkout_commit_mismatch")
    files: dict[str, Any] = {}
    expected_files = expected_identity.get("files")
    if not isinstance(expected_files, Mapping):
        raise JointAgentPacketError(["joint_agent_expected_file_lock_invalid"])
    for relative, expected_digest in sorted(expected_files.items()):
        path = root / str(relative)
        if not path.is_file():
            errors.append(f"joint_agent_release_file_missing:{relative}")
            continue
        observed_digest = _sha256(path)
        files[str(relative)] = {
            "sha256": observed_digest,
            "size_bytes": path.stat().st_size,
        }
        if observed_digest != expected_digest:
            errors.append(f"joint_agent_release_file_digest_mismatch:{relative}")
    version_path = root / "VERSION.md"
    version = version_path.read_text(encoding="utf-8").strip() if version_path.is_file() else ""
    if version != expected_identity.get("version"):
        errors.append("joint_agent_release_version_mismatch")
    if errors:
        raise JointAgentPacketError(errors)
    return {
        "repository": expected_identity["repository"],
        "tag": expected_identity["tag"],
        "version": version,
        "commit": commit,
        "license": expected_identity["license"],
        "checkout_path": str(root),
        "files": files,
    }


def _load_source_receipt(path: Path, asset_path: Path) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JointAgentPacketError(["articulated_source_receipt_invalid"]) from exc
    if not isinstance(receipt, Mapping):
        raise JointAgentPacketError(["articulated_source_receipt_invalid"])
    errors: list[str] = []
    if receipt.get("schema_version") != "articulated_source_asset.v1":
        errors.append("articulated_source_receipt_schema_invalid")
    expected_receipt_digest = canonical_digest(receipt, digest_field="receipt_digest")
    if receipt.get("receipt_digest") != expected_receipt_digest:
        errors.append("articulated_source_receipt_digest_invalid")
    output = receipt.get("output_asset")
    if not isinstance(output, Mapping):
        errors.append("articulated_source_output_identity_missing")
    else:
        if output.get("sha256") != _sha256(asset_path):
            errors.append("articulated_source_asset_digest_mismatch")
        if output.get("size_bytes") != asset_path.stat().st_size:
            errors.append("articulated_source_asset_size_mismatch")
    components = receipt.get("connected_component_count")
    if isinstance(components, bool) or not isinstance(components, int) or components < 1:
        errors.append("articulated_source_component_count_invalid")
    if errors:
        raise JointAgentPacketError(errors)
    return dict(receipt)


def build_joint_agent_packet(
    *,
    source_asset_path: str | Path,
    source_receipt_path: str | Path,
    joint_agent_checkout: str | Path,
    output_dir: str | Path,
    external_disclosure_authorized: bool = False,
    paid_execution_authorized: bool = False,
    expected_identity: Mapping[str, Any] = JOINT_AGENT_IDENTITY,
) -> dict[str, Any]:
    """Materialize an exact config and admission receipt without remote execution."""

    source = Path(source_asset_path).expanduser().resolve()
    source_receipt = Path(source_receipt_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not source.is_file():
        raise JointAgentPacketError(["articulated_source_asset_missing"])
    if destination.exists() and any(destination.iterdir()):
        raise JointAgentPacketError(["joint_agent_packet_output_not_empty"])
    destination.mkdir(parents=True, exist_ok=True)
    receipt = _load_source_receipt(source_receipt, source)
    release = inspect_joint_agent_checkout(
        joint_agent_checkout, expected_identity=expected_identity
    )
    component_count = int(receipt["connected_component_count"])
    working_dir = destination / "joint_agent_work"
    config: dict[str, Any] = {
        "project": {
            "name": "blueprint_adp_second_scene_articulation",
            "description": (
                "Infer candidate articulation topology for one exact SAGE-derived asset."
            ),
            "working_dir": str(working_dir),
        },
        "input": {"usd_path": str(source)},
        "steps": {
            "optimize_usd": {
                "enabled": True,
                "optimization_config": {
                    "backend": "local",
                    "scene_optimizer_settings": {
                        "enable_deinstance": False,
                        "enable_split_meshes": True,
                        "enable_deduplicate": False,
                    },
                },
            },
            "identify_asset": {"enabled": True, "renderer": {"backend": "remote"}},
            "analyze_structure": {
                "enabled": True,
                "strategy": "auto",
                "use_prompt_library": False,
                "llm": {"backend": "nim", "model": "google/gemma-4-31b-it"},
            },
            "build_dataset_usd": {
                "enabled": True,
                "renderer": {"backend": "remote"},
            },
            "build_dataset_prepare_dataset": {
                "enabled": True,
                "prompt_profile": "prop_articulation",
            },
            "predict": {
                "enabled": True,
                "completion_retries": 0,
                "vlm": {"backend": "nim", "model": "google/gemma-4-31b-it"},
            },
            "consistency_pass": {
                "enabled": True,
                "output_key": "classification",
                "harmonize_fields": ["axis_hint"],
                "harmonize_motion_profiles": True,
            },
            "infer_articulation_candidates": {
                "enabled": True,
                "output_key": "classification",
                "candidate_joint_types": ["revolute", "prismatic"],
                "adjudication": {
                    "enabled": True,
                    "reconcile_topology": True,
                    "model_key": "vlm",
                    "min_confidence": "high",
                    "max_tokens": 8192,
                    "max_images": component_count,
                    "require_source_images": True,
                },
                "vlm": {"backend": "nim", "model": "google/gemma-4-31b-it"},
            },
            "apply_joint_rigger": {
                "enabled": False,
                "adapter": "owned_core",
                "articulation_candidates_path": str(
                    working_dir
                    / "articulation_candidates"
                    / "articulation_candidates.json"
                ),
                "output_usd_path": str(working_dir / "joint_rigger" / "rigged.usdz"),
                "apply_masses": False,
                "apply_collision": False,
            },
            "author_physics_schemas": {"enabled": False},
        },
        "advanced": {"keep_temp_files": True, "log_level": "INFO"},
    }
    config_path = destination / "joint_agent.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    blockers: list[str] = []
    if not external_disclosure_authorized:
        blockers.append("external_scene_derived_byte_disclosure_authority_missing")
    if not paid_execution_authorized:
        blockers.append("fresh_paid_joint_agent_execution_authority_missing")
    packet: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked_before_remote_execution" if blockers else "ready_for_dry_run_only",
        "released_code": release,
        "source_asset": {
            "path": str(source),
            "sha256": _sha256(source),
            "size_bytes": source.stat().st_size,
            "source_receipt_path": str(source_receipt),
            "source_receipt_digest": receipt["receipt_digest"],
            "connected_component_count": component_count,
        },
        "config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
            "completion_retries": 0,
            "split_meshes": True,
            "predicted_dataset_prim_upper_bound": component_count,
            "topology_reconciliation_max_images": component_count,
            "adapter": "owned_core",
            "apply_joint_rigger_initially_enabled": False,
        },
        "execution_admission": {
            "external_disclosure_authorized": external_disclosure_authorized,
            "paid_execution_authorized": paid_execution_authorized,
            "blockers": blockers,
            "remote_execution_performed": False,
            "automatic_paid_retry_allowed": False,
        },
        "required_post_execution_evidence": [
            "exact_prediction_row_for_every_dataset_prim",
            "stage2_candidate_document",
            "human_or_authorized_review_before_rigger_enable",
            "owned_core_joint_graph_readback",
            "gate3a_static_receipt",
            "gate3b_static_receipt",
            "independent_native_dynamic_articulation_receipt",
        ],
        "claim_boundary": {
            "config_materialized": True,
            "released_code_identity_verified": True,
            "component_count_is_not_joint_count": True,
            "joint_topology_inferred": False,
            "simready_qualified": False,
            "physical_equivalence_proven": False,
        },
        "packet_digest": "",
    }
    packet["packet_digest"] = canonical_digest(packet, digest_field="packet_digest")
    packet_path = destination / "joint_agent_packet.json"
    packet_path.write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return packet


def _resolved_under(path: str | Path, approved_roots: Sequence[str | Path]) -> Path:
    resolved = Path(path).expanduser().resolve()
    roots = [Path(root).expanduser().resolve() for root in approved_roots]
    if not roots or not any(resolved == root or root in resolved.parents for root in roots):
        raise JointAgentPacketError([f"path_outside_approved_roots:{resolved}"])
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-asset", required=True)
    parser.add_argument("--source-receipt", required=True)
    parser.add_argument("--joint-agent-checkout", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--approved-root", action="append", required=True)
    args = parser.parse_args(argv)
    packet = build_joint_agent_packet(
        source_asset_path=_resolved_under(args.source_asset, args.approved_root),
        source_receipt_path=_resolved_under(args.source_receipt, args.approved_root),
        joint_agent_checkout=_resolved_under(args.joint_agent_checkout, args.approved_root),
        output_dir=_resolved_under(args.output_dir, args.approved_root),
    )
    print(json.dumps(packet, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "JOINT_AGENT_IDENTITY",
    "JointAgentPacketError",
    "SCHEMA_VERSION",
    "build_joint_agent_packet",
    "inspect_joint_agent_checkout",
]
