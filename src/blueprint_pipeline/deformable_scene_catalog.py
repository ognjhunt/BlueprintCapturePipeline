"""Digest-bound, outcome-blind catalog for public deformable-transfer scenes.

The catalog is deliberately a semantic/topology filter.  It inventories every
InteriorGS scene present in the declared roots, joins the exact canonical SAGE
collision file when available, and identifies same-publisher-room compatible
movable/destination pairs.  It cannot qualify material behavior, openness,
registration, robot reachability, or policy outcomes; those remain later gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

from .decision_evidence_contracts import canonical_digest
from .scene_placement.interiorgs_index import (
    inventory_deformable_transfer_candidates,
    load_interiorgs_labels,
    load_interiorgs_structure,
    point_in_polygon,
)

SCHEMA_VERSION = "adp_deformable_scene_catalog.v1"
_SCENE_DIR = re.compile(r"(?:^|_)([0-9]{6})$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _scene_sources(roots: Sequence[str | Path]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for root_value in roots:
        root = Path(root_value).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"interiorgs_catalog_root_missing:{root}")
        for scene_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            match = _SCENE_DIR.search(scene_dir.name)
            labels = scene_dir / "labels.json"
            structure = scene_dir / "structure.json"
            if match is None or not labels.is_file() or not structure.is_file():
                continue
            scene_id = match.group(1)
            if scene_id in sources:
                raise ValueError(f"interiorgs_scene_id_ambiguous:{scene_id}")
            sources[scene_id] = scene_dir
    return sources


def _collision_source(scene_id: str, roots: Sequence[str | Path]) -> Path | None:
    matches: list[Path] = []
    for root_value in roots:
        root = Path(root_value).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"sage_catalog_root_missing:{root}")
        candidates = (
            root / "Collision_Mesh" / scene_id / f"{scene_id}_collision.usd",
            root / scene_id / f"{scene_id}_collision.usd",
        )
        matches.extend(path for path in candidates if path.is_file())
    unique = sorted({path.resolve() for path in matches})
    if len(unique) > 1:
        raise ValueError(f"sage_scene_id_ambiguous:{scene_id}")
    return unique[0] if unique else None


def _room_index_by_object(scene_dir: Path) -> tuple[dict[str, int | None], list[Any]]:
    structure = load_interiorgs_structure(scene_dir / "structure.json")
    objects = load_interiorgs_labels(scene_dir / "labels.json")
    room_indices = {
        item.id: next(
            (
                index
                for index, polygon in enumerate(structure.rooms)
                if point_in_polygon((item.centroid[0], item.centroid[1]), polygon)
            ),
            None,
        )
        for item in objects
    }
    return room_indices, objects


def build_deformable_scene_catalog(
    *,
    interiorgs_roots: Sequence[str | Path],
    sage_roots: Sequence[str | Path],
    previously_used_scene_ids: Sequence[str] = (),
    expected_scene_count: int | None = None,
) -> dict[str, Any]:
    """Inventory every declared InteriorGS topology before any target outcome."""

    sources = _scene_sources(interiorgs_roots)
    if expected_scene_count is not None and len(sources) != int(expected_scene_count):
        raise ValueError(
            f"known_scene_count_mismatch:expected={expected_scene_count}:observed={len(sources)}"
        )
    previously_used = {str(value) for value in previously_used_scene_ids}
    scenes: list[dict[str, Any]] = []
    for scene_id, scene_dir in sorted(sources.items(), key=lambda row: int(row[0])):
        room_indices, objects = _room_index_by_object(scene_dir)
        inventory = inventory_deformable_transfer_candidates(objects)
        same_room_pairs = [
            {
                **pair,
                "publisher_room_index": room_indices[pair["movable_ins_id"]],
            }
            for pair in inventory["compatible_pairs"]
            if room_indices.get(pair["movable_ins_id"]) is not None
            and room_indices.get(pair["movable_ins_id"])
            == room_indices.get(pair["destination_ins_id"])
        ]
        appearance = _artifact(scene_dir / "3dgs_compressed.ply")
        collision_path = _collision_source(scene_id, sage_roots)
        collision = _artifact(collision_path) if collision_path else None
        movable_ranks = {
            int(row["task_family_rank"])
            for row in inventory["movable_deformable_candidates"]
        }
        compatible_destinations = [
            row
            for row in inventory["destination_receptacle_candidates"]
            if movable_ranks.intersection(row["compatible_task_family_ranks"])
        ]
        if scene_id in previously_used:
            status = "rejected"
            rejection_reasons = ["previously_used_scene"]
        elif not inventory["movable_deformable_candidates"]:
            status = "rejected"
            rejection_reasons = ["no_admitted_movable_semantic"]
        elif not compatible_destinations:
            status = "rejected"
            rejection_reasons = ["no_compatible_destination_semantic"]
        elif not same_room_pairs:
            status = "rejected"
            rejection_reasons = ["no_same_publisher_room_compatible_pair"]
        elif appearance is None or collision is None:
            status = "rejected"
            rejection_reasons = ["exact_appearance_collision_pair_unavailable"]
        else:
            status = "semantic_shortlist"
            rejection_reasons = []
        scenes.append(
            {
                "scene_id": scene_id,
                "interiorgs_directory": scene_dir.name,
                "status": status,
                "rejection_reasons": rejection_reasons,
                "source_files": {
                    "labels": _artifact(scene_dir / "labels.json"),
                    "structure": _artifact(scene_dir / "structure.json"),
                    "appearance": appearance,
                    "collision": collision,
                },
                "inventory": inventory,
                "same_publisher_room_compatible_pairs": same_room_pairs,
                "same_publisher_room_compatible_pair_count": len(same_room_pairs),
            }
        )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "purpose": "outcome_blind_complete_known_topology_scene_selection",
        "learned_policy_outcomes_inspected": False,
        "known_scene_count": len(scenes),
        "previously_used_scene_ids": sorted(previously_used),
        "scenes": scenes,
        "semantic_shortlist_scene_ids": [
            row["scene_id"] for row in scenes if row["status"] == "semantic_shortlist"
        ],
        "selection_authority": "publisher_labels_structure_and_exact_local_source_identity",
        "claim_boundary": {
            "material_class_qualified": False,
            "destination_open_interior_qualified": False,
            "appearance_collision_registration_qualified": False,
            "robot_reachability_qualified": False,
            "scene_selected": False,
            "policy_outcomes_used": False,
        },
        "catalog_digest": "",
    }
    result["catalog_digest"] = canonical_digest(result, digest_field="catalog_digest")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interiorgs-root", action="append", required=True)
    parser.add_argument("--sage-root", action="append", required=True)
    parser.add_argument("--previously-used-scene-id", action="append", default=[])
    parser.add_argument("--expected-scene-count", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = build_deformable_scene_catalog(
        interiorgs_roots=args.interiorgs_root,
        sage_roots=args.sage_root,
        previously_used_scene_ids=args.previously_used_scene_id,
        expected_scene_count=args.expected_scene_count,
    )
    output = Path(args.out).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(output), "catalog_digest": result["catalog_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
