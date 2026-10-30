"""Refined-splat request forms, their roots and an operator's upgrade request.

Job control imports this to reopen a finished job, so it reads retained
records only and imports no provider, model or reconstruction code.
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import write_json
from .decision_evidence_contracts import canonical_digest

MANIFEST = "atlas_refined_splat_manifest.json"
LEGACY_MANIFEST = "atlas_depth_splat_manifest.json"
UPGRADE = "gap_preserving_upgrade.json"
REVIEW_RECORD = "whole_view_review.json"
WHOLE, MINIMAL = "documented_reviewed_whole", "documented_minimal"
# Each request form has its own root beside the draft, in this order. The first
# two are earlier releases' forms with masks and depth confidence, which World
# Labs does not document: adopted or polled, never bought again.
FORMS = (("reviewed_masks_and_confidence", ""), ("reviewed_masks", "-masks"), (WHOLE, "-whole"), (MINIMAL, "-minimal"))
LEGACY_FORMS = frozenset({"reviewed_masks_and_confidence", "reviewed_masks"})


def splat_manifest_path(root: Path) -> Path:
    native = root / MANIFEST
    return native if native.exists() else root / LEGACY_MANIFEST


def load_splat_manifest(root: Path) -> dict:
    path = splat_manifest_path(root)
    if path.is_symlink():
        raise ValueError("atlas_splat_manifest_symlink")
    value = json.loads(path.read_text())
    if value.get("digest") != canonical_digest(value, digest_field="digest"):
        raise ValueError("atlas_splat_manifest_changed")
    return value


def form_roots(output_root: Path):
    return [(form, output_root.with_name(output_root.name + suffix)) for form, suffix in FORMS]


def bought(root: Path) -> bool:
    return any((root / "native").rglob("atlas_pose/submission.json"))


def untried_gap_roots(roots) -> list:
    """The reviewed-whole form, if it was never sent and its whole-view review never ran."""
    whole = dict(roots)[WHOLE]
    return [] if bought(whole) or (whole / REVIEW_RECORD).is_file() else [whole]


def request_gap_preserving_upgrade(root: Path, *, splat_manifest_digest: str, at_epoch: int) -> dict:
    """Record an operator's request to try the reviewed-whole form a minimal splat left untried.

    Never automatic. The next run reviews the generated views if needed and sends
    the form at most once; a success supersedes the minimal splat, and if nothing
    is admitted or the provider fails it, the minimal splat is adopted again.
    """
    if root.is_symlink() or not root.name.endswith("-minimal") or not (root / MANIFEST).is_file():
        raise ValueError("atlas_refined_splat_upgrade_invalid")
    manifest = load_splat_manifest(root)
    if manifest["digest"] != splat_manifest_digest or manifest.get("request_form") != MINIMAL:
        raise ValueError("atlas_refined_splat_upgrade_invalid")
    roots = [path.name for path in untried_gap_roots(form_roots(root.with_name(root.name.removesuffix("-minimal"))))]
    if not roots:
        raise ValueError("atlas_refined_splat_upgrade_unavailable")
    value = {"schema_version": "atlas_refined_splat_upgrade.v1", "splat_manifest_digest": splat_manifest_digest,
             "roots": roots, "requested_at_epoch": at_epoch}
    write_json(root / UPGRADE, value)
    return value
