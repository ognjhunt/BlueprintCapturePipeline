"""Frozen held-out splits for Post-Training Data Packages.

Blueprint sells captured clips as training data. Nothing in the package told a
buyer which clips to keep out of training, and nothing stopped their evaluation
set from overlapping the clips they had just been sold. A model evaluated on its
own training data reports a number that means nothing, and the buyer has no way
to know -- the overlap is invisible from inside the package.

This is a defect in the shipped product rather than a research nicety. The fix is
to ship the split *with* the data, frozen and digest-pinned, so:

* a held-out cut is carved from the same capture, so it is in-distribution
  rather than a different site with different lighting and layout;
* assignment is deterministic from ``(split_id, clip_id)``, so it is
  reproducible, independent of clip ordering, and cannot be quietly reshuffled
  until a favourable split appears;
* the partitions are provably disjoint, and the package is checked against the
  split so a training payload containing a held-out clip fails closed; and
* the buyer can verify all of it from the digests alone.

A split is a data-partitioning contract. It is not a benchmark, it does not make
any evaluation run on it decision-grade, and it says nothing about whether a
model trained on the training partition is any good.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import read_json_any, utc_now_iso, write_json


HOLDOUT_SPLIT_SCHEMA_VERSION = "post_training_holdout_split.v1"
PACKAGE_CHECK_SCHEMA_VERSION = "post_training_holdout_package_check.v1"

TRAIN_PARTITION = "train"
HOLDOUT_PARTITION = "holdout"
PARTITIONS = (TRAIN_PARTITION, HOLDOUT_PARTITION)

DEFAULT_HOLDOUT_FRACTION = 0.15
# Below this the held-out cut is too small to support any useful evaluation, so
# shipping it would imply a rigour the package cannot deliver.
MIN_HOLDOUT_CLIPS = 3
MIN_TRAIN_CLIPS = 1


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assignment_score(split_id: str, clip_id: str) -> float:
    """Deterministic [0, 1) score for one clip under one split.

    Hash-based rather than shuffle-based so the assignment depends only on the
    split id and the clip id -- not on how the clips happened to be ordered, and
    not on a random seed someone could re-roll until the split flattered them.
    """

    digest = hashlib.sha256(f"{split_id}\x00{clip_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _group_key(row: Mapping[str, Any], stratify_by: Sequence[str]) -> tuple[str, ...]:
    return tuple(_string(row.get(field)) for field in stratify_by)


def build_holdout_split(
    *,
    split_id: str,
    clips: Sequence[Mapping[str, Any]],
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    stratify_by: Sequence[str] = (),
    scene_id: str = "",
    capture_id: str = "",
) -> dict[str, Any]:
    """Carve a frozen, digest-pinned held-out cut from one capture's clips.

    ``stratify_by`` names clip fields (task, room, lighting condition) whose
    distribution should be preserved across partitions; without it a small
    capture can put every instance of a task on one side of the split and make
    the held-out cut unrepresentative.
    """

    blockers: list[str] = []
    if not _string(split_id):
        blockers.append("holdout_split_id_missing")
    if not 0.0 < holdout_fraction < 1.0:
        blockers.append("holdout_fraction_out_of_range")
        holdout_fraction = DEFAULT_HOLDOUT_FRACTION

    seen: set[str] = set()
    ordered: list[dict[str, Any]] = []
    for index, raw in enumerate(clips):
        row = _mapping(raw)
        clip_id = _string(row.get("clip_id")) or _string(row.get("id")) or _string(
            row.get("clip_path")
        )
        if not clip_id:
            blockers.append(f"clip_id_missing:{index}")
            continue
        if clip_id in seen:
            # A duplicate id could otherwise land in both partitions.
            blockers.append(f"duplicate_clip_id:{clip_id}")
            continue
        seen.add(clip_id)
        ordered.append({**row, "clip_id": clip_id})

    if not ordered:
        blockers.append("holdout_split_has_no_clips")

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in ordered:
        groups.setdefault(_group_key(row, stratify_by), []).append(row)

    train: list[str] = []
    holdout: list[str] = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda item: item["clip_id"])
        # Rank within the stratum by the deterministic score, then take the
        # lowest-scoring share into the holdout. Ranking (rather than
        # thresholding) keeps the requested fraction exact per stratum.
        ranked = sorted(members, key=lambda item: (assignment_score(split_id, item["clip_id"]), item["clip_id"]))
        take = int(round(len(ranked) * holdout_fraction))
        if len(ranked) > 1:
            take = max(1, min(take, len(ranked) - 1))
        else:
            take = 0
        holdout.extend(row["clip_id"] for row in ranked[:take])
        train.extend(row["clip_id"] for row in ranked[take:])

    train_sorted = sorted(train)
    holdout_sorted = sorted(holdout)
    overlap = sorted(set(train_sorted) & set(holdout_sorted))
    if overlap:
        blockers.append("holdout_and_train_partitions_overlap")
    if len(holdout_sorted) < MIN_HOLDOUT_CLIPS:
        blockers.append(
            f"holdout_partition_below_minimum:{len(holdout_sorted)}<{MIN_HOLDOUT_CLIPS}"
        )
    if len(train_sorted) < MIN_TRAIN_CLIPS:
        blockers.append("train_partition_empty")

    core = {
        "split_id": _string(split_id),
        "scene_id": _string(scene_id),
        "capture_id": _string(capture_id),
        "holdout_fraction": round(float(holdout_fraction), 6),
        "stratify_by": list(stratify_by),
        "train_clip_ids": train_sorted,
        "holdout_clip_ids": holdout_sorted,
    }
    return {
        "schema_version": HOLDOUT_SPLIT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "frozen" if not blockers else "blocked",
        **core,
        "assignment_method": "sha256(split_id || clip_id) ranked within stratum",
        "train_clip_count": len(train_sorted),
        "holdout_clip_count": len(holdout_sorted),
        "total_clip_count": len(ordered),
        "partitions_disjoint": not overlap,
        "overlapping_clip_ids": overlap,
        "split_sha256": canonical_sha256(core),
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "split_is_a_data_partition_not_a_benchmark": True,
            "holdout_evaluation_is_not_decision_grade_rank_fidelity": True,
            "split_does_not_certify_model_quality": True,
            "holdout_is_in_distribution_from_the_same_capture": True,
        },
    }


def check_package_against_split(
    *,
    split: Mapping[str, Any],
    training_clip_ids: Sequence[str],
    delivered_holdout_clip_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fail closed when a training payload contains held-out clips.

    This is the check that makes the split real rather than advisory: without
    it, a package could ship a split document and still include every held-out
    clip in the training payload.
    """

    blockers: list[str] = []
    if split.get("schema_version") != HOLDOUT_SPLIT_SCHEMA_VERSION:
        blockers.append("holdout_split_schema_missing_or_unsupported")
    if split.get("status") != "frozen":
        blockers.append("holdout_split_not_frozen")

    holdout = {_string(item) for item in split.get("holdout_clip_ids") or [] if _string(item)}
    declared_train = {
        _string(item) for item in split.get("train_clip_ids") or [] if _string(item)
    }
    delivered_train = {_string(item) for item in training_clip_ids if _string(item)}

    leaked = sorted(delivered_train & holdout)
    if leaked:
        blockers.append("training_payload_contains_holdout_clips")

    undeclared = sorted(delivered_train - declared_train - holdout)
    if undeclared:
        # Clips outside the frozen split were never assigned a partition, so
        # their status is unknown rather than safe.
        blockers.append("training_payload_contains_clips_outside_the_frozen_split")

    delivered_holdout = {
        _string(item) for item in (delivered_holdout_clip_ids or []) if _string(item)
    }
    missing_holdout = sorted(holdout - delivered_holdout) if delivered_holdout else []
    if delivered_holdout and missing_holdout:
        blockers.append("delivered_holdout_cut_incomplete")

    return {
        "schema_version": PACKAGE_CHECK_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "clean" if not blockers else "blocked",
        "split_id": split.get("split_id"),
        "split_sha256": split.get("split_sha256"),
        "training_clip_count": len(delivered_train),
        "holdout_clip_count": len(holdout),
        "leaked_clip_ids": leaked,
        "leaked_clip_count": len(leaked),
        "undeclared_clip_ids": undeclared,
        "missing_holdout_clip_ids": missing_holdout,
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "check_verifies_partition_hygiene_not_data_quality": True,
            "clean_does_not_mean_the_holdout_is_a_sufficient_evaluation_set": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build and verify frozen held-out splits for training packages"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="freeze a held-out split")
    build.add_argument("--input", required=True)
    build.add_argument("--output", required=True)
    build.set_defaults(kind="build")

    check = sub.add_parser("check", help="verify a package against its split")
    check.add_argument("--split", required=True)
    check.add_argument("--input", required=True)
    check.add_argument("--output", required=True)
    check.set_defaults(kind="check")

    args = parser.parse_args(argv)
    if args.kind == "build":
        payload = _mapping(read_json_any(Path(args.input)))
        result = build_holdout_split(
            split_id=_string(payload.get("split_id")),
            clips=_rows(payload.get("clips")),
            holdout_fraction=float(payload.get("holdout_fraction") or DEFAULT_HOLDOUT_FRACTION),
            stratify_by=[_string(item) for item in payload.get("stratify_by") or []],
            scene_id=_string(payload.get("scene_id")),
            capture_id=_string(payload.get("capture_id")),
        )
        ok = result["status"] == "frozen"
    else:
        payload = _mapping(read_json_any(Path(args.input)))
        result = check_package_against_split(
            split=_mapping(read_json_any(Path(args.split))),
            training_clip_ids=[_string(item) for item in payload.get("training_clip_ids") or []],
            delivered_holdout_clip_ids=[
                _string(item) for item in payload.get("delivered_holdout_clip_ids") or []
            ]
            or None,
        )
        ok = result["status"] == "clean"

    write_json(Path(args.output), result)
    print(json.dumps({"path": args.output, "status": result["status"]}, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
