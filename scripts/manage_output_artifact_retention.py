#!/usr/bin/env python3
"""Inventory, select, and optionally prune local ``output/`` artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "output_artifact_retention_manifest.v1"
DEFAULT_OUTPUT_ROOT = "output"
DEFAULT_MANIFEST_PATH = "output/output_artifact_retention_manifest.json"
EXECUTE_ACK = "delete-output-artifacts"

RETENTION_CLASSES: dict[str, dict[str, Any]] = {
    "canonical_launch_evidence": {
        "delete_after_days": 365,
        "prunable": False,
        "description": "Current launch, CI, paid-gate, and operator handoff evidence.",
    },
    "external_asset_cache": {
        "delete_after_days": None,
        "prunable": False,
        "description": "Reusable local assets such as MuJoCo menagerie or collected USD/mesh inputs.",
    },
    "provider_runtime_or_paid_run": {
        "delete_after_days": 30,
        "prunable": True,
        "description": "Provider/runtime run output, bundles, object-store staging, and paid render runs.",
    },
    "local_preflight_or_dry_run": {
        "delete_after_days": 14,
        "prunable": True,
        "description": "Local dry-render, preflight, bootstrap, and no-spend smoke artifacts.",
    },
    "ci_or_capacity_artifact": {
        "delete_after_days": 90,
        "prunable": True,
        "description": "CI evidence, capacity reports, and generated release support files.",
    },
    "uncategorized_output": {
        "delete_after_days": 30,
        "prunable": True,
        "description": "Output artifact without a known current handoff or cache role.",
    },
}

CANONICAL_ARTIFACT_NAMES: dict[str, str] = {
    "launch_readiness_packet.json": "launch_readiness_packet_json",
    "launch_readiness_packet.md": "launch_readiness_packet_markdown",
    "paid_marketplace_launch_gate.json": "paid_marketplace_launch_gate_json",
    "paid_marketplace_launch_gate.md": "paid_marketplace_launch_gate_markdown",
    "pipeline_main_ci_evidence.json": "pipeline_main_ci_evidence",
    "pipeline_full_test_lane_ci_evidence.json": "pipeline_full_test_lane_ci_evidence",
    "pipeline_sim_only_local_gate_ci_evidence.json": "pipeline_sim_only_local_gate_ci_evidence",
    "webapp_main_ci_evidence.json": "webapp_main_ci_evidence",
}

CANONICAL_ARTIFACT_PREFIXES: tuple[tuple[str, str], ...] = (
    ("launch_audit_live_pipeline_setup_", "launch_audit_live_pipeline_setup"),
    (
        "launch_audit_unitree_groot_sonic_provider_readiness_",
        "launch_audit_unitree_groot_sonic_provider_readiness",
    ),
    ("sim_only_beta_local_gate_report", "sim_only_beta_local_gate_report"),
    ("sim_only_beta_release_gate_report", "sim_only_beta_release_gate_report"),
    ("sim_only_beta_deployment_parity_proof", "sim_only_beta_deployment_parity_proof"),
)


@dataclass(frozen=True)
class ArtifactEntry:
    path: Path
    relative_path: str
    kind: str
    size_bytes: int
    mtime_epoch: float
    age_days: float
    retention_class: str


def _now() -> float:
    return time.time()


def _size_bytes(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file() or item.is_symlink():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def _latest_mtime(path: Path) -> float:
    try:
        latest = path.stat().st_mtime
    except OSError:
        return 0.0
    if not path.is_dir():
        return latest
    for item in path.rglob("*"):
        try:
            latest = max(latest, item.stat().st_mtime)
        except OSError:
            continue
    return latest


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _canonical_key(path: Path) -> str | None:
    name = path.name
    if name in CANONICAL_ARTIFACT_NAMES:
        return CANONICAL_ARTIFACT_NAMES[name]
    for prefix, key in CANONICAL_ARTIFACT_PREFIXES:
        if name.startswith(prefix) and path.suffix in {".json", ".md"}:
            return key
    return None


def _classify(path: Path) -> str:
    text = path.as_posix().lower()
    name = path.name.lower()
    if _canonical_key(path):
        return "canonical_launch_evidence"
    if "external_assets" in text or "mujoco_menagerie" in text or "collected_" in text:
        return "external_asset_cache"
    if any(token in text for token in ("object_store_real_run", "runpod", "vast", "digitalocean", "_paid", "provider_reliability")):
        return "provider_runtime_or_paid_run"
    if any(token in text for token in ("dry_render", "preflight", "bootstrap", "no_spend", "local_no_spend", "canary")):
        return "local_preflight_or_dry_run"
    if any(token in text for token in ("ci_evidence", "beta_capacity", "deployment", "readiness_packet")):
        return "ci_or_capacity_artifact"
    if name in {".ds_store"}:
        return "local_preflight_or_dry_run"
    return "uncategorized_output"


def _entry(path: Path, root: Path, now: float) -> ArtifactEntry:
    mtime = _latest_mtime(path)
    return ArtifactEntry(
        path=path,
        relative_path=_relative(path, root),
        kind="directory" if path.is_dir() else "file",
        size_bytes=_size_bytes(path),
        mtime_epoch=mtime,
        age_days=max(0.0, (now - mtime) / 86400.0),
        retention_class=_classify(path),
    )


def _top_level_entries(root: Path, now: float) -> list[ArtifactEntry]:
    if not root.exists():
        return []
    return [_entry(path, root, now) for path in sorted(root.iterdir(), key=lambda item: item.name)]


def _canonical_candidates(root: Path, now: float) -> list[ArtifactEntry]:
    if not root.exists():
        return []
    return [
        _entry(path, root, now)
        for path in root.rglob("*")
        if path.is_file() and _canonical_key(path) is not None
    ]


def select_canonical_artifacts(entries: Iterable[ArtifactEntry]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key: dict[str, list[ArtifactEntry]] = {}
    for entry in entries:
        key = _canonical_key(entry.path)
        if key:
            by_key.setdefault(key, []).append(entry)
    canonical: list[dict[str, Any]] = []
    superseded: list[dict[str, Any]] = []
    for key, values in sorted(by_key.items()):
        ranked = sorted(values, key=lambda item: (item.mtime_epoch, item.relative_path), reverse=True)
        winner = ranked[0]
        canonical.append(_entry_payload(winner, canonical_key=key, canonical=True))
        superseded.extend(
            _entry_payload(item, canonical_key=key, canonical=False)
            for item in ranked[1:]
        )
    return canonical, superseded


def _entry_payload(entry: ArtifactEntry, *, canonical_key: str | None = None, canonical: bool | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": entry.relative_path,
        "kind": entry.kind,
        "size_bytes": entry.size_bytes,
        "mtime_epoch": entry.mtime_epoch,
        "age_days": round(entry.age_days, 3),
        "retention_class": entry.retention_class,
    }
    if canonical_key is not None:
        payload["canonical_key"] = canonical_key
    if canonical is not None:
        payload["canonical"] = canonical
    return payload


def _prune_candidates(entries: Iterable[ArtifactEntry], canonical_artifacts: list[dict[str, Any]]) -> list[ArtifactEntry]:
    canonical_top_level = {
        str(item["path"]).split("/", 1)[0]
        for item in canonical_artifacts
    }
    candidates: list[ArtifactEntry] = []
    for entry in entries:
        policy = RETENTION_CLASSES[entry.retention_class]
        delete_after = policy.get("delete_after_days")
        if not policy.get("prunable") or delete_after is None:
            continue
        if entry.relative_path.split("/", 1)[0] in canonical_top_level:
            continue
        if entry.age_days >= float(delete_after):
            candidates.append(entry)
    return candidates


def build_manifest(
    *,
    output_root: Path,
    dry_run: bool,
    now: float | None = None,
) -> tuple[dict[str, Any], list[ArtifactEntry]]:
    resolved_now = now if now is not None else _now()
    top_level = _top_level_entries(output_root, resolved_now)
    canonical, superseded = select_canonical_artifacts(_canonical_candidates(output_root, resolved_now))
    candidates = _prune_candidates(top_level, canonical)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_epoch": resolved_now,
        "output_root": str(output_root),
        "dry_run": dry_run,
        "status": "dry_run" if dry_run else "execute_requested",
        "retention_classes": RETENTION_CLASSES,
        "total_top_level_entry_count": len(top_level),
        "total_top_level_size_bytes": sum(entry.size_bytes for entry in top_level),
        "canonical_artifacts": canonical,
        "superseded_canonical_artifacts": superseded,
        "prune_candidates": [_entry_payload(entry) for entry in candidates],
        "top_level_entries": [_entry_payload(entry) for entry in top_level],
        "claim_boundary": {
            "dry_run_default": True,
            "deletion_requires_execute_and_ack": True,
            "canonical_selection_is_local_snapshot_not_live_proof": True,
            "raw_capture_truth_legal_hold_not_automated_by_this_script": True,
        },
    }
    return manifest, candidates


def execute_prune(candidates: Iterable[ArtifactEntry]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for entry in candidates:
        if not entry.path.exists():
            actions.append({"path": entry.relative_path, "status": "already_missing"})
            continue
        if entry.path.is_dir():
            shutil.rmtree(entry.path)
        else:
            entry.path.unlink()
        actions.append({"path": entry.relative_path, "status": "deleted"})
    return actions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-delete-output-artifacts")
    args = parser.parse_args(argv)

    output_root = Path(args.output_root).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve()
    dry_run = not args.execute
    if args.execute and args.acknowledge_delete_output_artifacts != EXECUTE_ACK:
        raise SystemExit(
            f"--execute requires --acknowledge-delete-output-artifacts {EXECUTE_ACK!r}"
        )
    manifest, candidates = build_manifest(output_root=output_root, dry_run=dry_run)
    if args.execute:
        manifest["prune_actions"] = execute_prune(candidates)
        manifest["status"] = "completed"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[output-retention] status={manifest['status']}")
    print(f"[output-retention] manifest={manifest_path}")
    print(f"[output-retention] prune_candidates={len(candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
