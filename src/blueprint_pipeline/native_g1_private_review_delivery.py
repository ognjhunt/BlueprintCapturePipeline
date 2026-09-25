"""Register verified G1 review media for authenticated Pipeline artifact reads.

This is a private post-provider-zero handoff. It creates no public publication
record and does not upgrade a simulated result into physical evidence.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .core.security_controls import strict_identifier
from .decision_evidence_contracts import canonical_digest, canonical_json
from .native_g1_paid_campaign import verify_g1_paid_output
from .native_g1_private_review import project_g1_private_review
from .native_g1_provider_bundle import load_verified_g1_provider_bundle
from .task_evaluation_result_delivery import (
    REGISTRY_SCHEMA_VERSION,
    _artifact_id,
    _sha256,
    resolve_task_evaluation_result_artifact,
)


def _read(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("g1_review_delivery_input_invalid")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("g1_review_delivery_input_invalid")
    return value


def _relative(value: str) -> Path:
    pure = PurePosixPath(value)
    if (
        not value or pure.is_absolute() or "\\" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("g1_review_delivery_artifact_path_invalid")
    return Path(*pure.parts)


def _media(review: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    if (
        review.get("status") != "verified_private_development_review"
        or review.get("claim_ceiling") != "development_only"
        or review.get("public_redistribution_authorized") is not False
        or review.get("physical_outcome_claimed") is not False
        or not isinstance(review.get("episodes"), list)
        or len(review["episodes"]) != 4
    ):
        raise ValueError("g1_review_delivery_unverified_review")
    records: list[tuple[str, Mapping[str, Any]]] = []
    for episode in review["episodes"]:
        if not isinstance(episode, Mapping):
            raise ValueError("g1_review_delivery_episode_invalid")
        videos = episode.get("review_videos")
        if not isinstance(videos, Mapping) or set(videos) != {"head", "overview"}:
            raise ValueError("g1_review_delivery_episode_invalid")
        for role, row in (
            ("g1_frame_manifest", episode.get("frame_manifest")),
            ("g1_head_video", videos["head"]),
            ("g1_overview_video", videos["overview"]),
        ):
            if not isinstance(row, Mapping):
                raise ValueError("g1_review_delivery_episode_invalid")
            records.append((role, row))
    return records


def _stage_review_artifacts(
    *, review: Mapping[str, Any], source_root: Path, result_root: Path, run_id: str
) -> dict[str, Any]:
    run = strict_identifier(run_id, field="run_id", max_length=192)
    root = Path(result_root)
    source = Path(source_root)
    if (
        not root.is_absolute() or root.is_symlink() or not root.is_dir()
        or not source.is_absolute() or source.is_symlink() or not source.is_dir()
    ):
        raise ValueError("g1_review_delivery_root_invalid")
    target = root / f"{run}-activation"
    if target.exists() or target.is_symlink():
        raise ValueError("g1_review_delivery_run_already_registered")
    stage = Path(tempfile.mkdtemp(prefix=f".g1-{run}-", dir=root))
    try:
        os.chmod(stage, 0o750)
        evidence = stage / "evidence"
        evidence.mkdir(mode=0o750)
        records: list[dict[str, Any]] = []
        paths: set[str] = set()
        for role, row in _media(review):
            relative_text = str(row.get("relative_path") or "")
            relative = _relative(relative_text)
            if relative_text in paths:
                raise ValueError("g1_review_delivery_duplicate_artifact")
            paths.add(relative_text)
            src = source / relative
            if any(part.is_symlink() for part in (src, *src.parents) if part != Path("/")):
                raise ValueError("g1_review_delivery_artifact_symlink_forbidden")
            if not src.is_file():
                raise ValueError("g1_review_delivery_artifact_missing")
            digest, size = row.get("sha256"), row.get("size_bytes")
            if (
                not isinstance(digest, str) or not digest.startswith("sha256:")
                or len(digest) != 71 or type(size) is not int or size <= 0
                or src.stat().st_size != size or _sha256(src) != digest
            ):
                raise ValueError("g1_review_delivery_artifact_identity_invalid")
            destination = evidence / relative
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
            try:
                os.link(src, destination)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                shutil.copyfile(src, destination)
            if destination.stat().st_size != size or _sha256(destination) != digest:
                raise ValueError("g1_review_delivery_staged_identity_invalid")
            records.append({
                "artifact_id": _artifact_id(role, relative_text, digest),
                "role": role,
                "relative_path": relative_text,
                "sha256": digest,
                "size_bytes": size,
                "content_type": "application/json" if role == "g1_frame_manifest" else "video/mp4",
                "evidence_root": str(target / "evidence"),
            })
        registry = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "run_id": run,
            "delivery_digest": review["review_digest"],
            "artifacts": records,
            "registry_digest": "",
        }
        registry["registry_digest"] = canonical_digest(registry, digest_field="registry_digest")
        registry_path = stage / "artifacts/result_delivery/artifact_registry.json"
        registry_path.parent.mkdir(parents=True, mode=0o750)
        registry_path.write_text(canonical_json(registry) + "\n", encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise ValueError("g1_review_delivery_run_already_registered")
        os.rename(stage, target)
        for record in records:
            resolve_task_evaluation_result_artifact(
                run_root=target, run_id=run, artifact_id=record["artifact_id"]
            )
        return {
            "schema_version": "native_g1_private_review_delivery.v1",
            "status": "registered_private_development_review",
            "run_id": run,
            "review_digest": review["review_digest"],
            "registry_digest": registry["registry_digest"],
            "artifact_count": len(records),
            "run_root": str(target),
            "public_redistribution_authorized": False,
        }
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def materialize_g1_private_review_delivery(
    *, adapter_result_path: Path, bundle_receipt_path: Path,
    retained_review_path: Path, result_root: Path, run_id: str,
) -> dict[str, Any]:
    adapter = _read(adapter_result_path)
    bundle_receipt = _read(bundle_receipt_path)
    bundle = load_verified_g1_provider_bundle(
        bundle_receipt_path,
        expected_implementation_commit=bundle_receipt["implementation_commit"],
    )
    verification = verify_g1_paid_output(adapter, bundle)
    review = project_g1_private_review(verification=verification, bundle=bundle)
    if _read(retained_review_path) != review:
        raise ValueError("g1_review_delivery_retained_review_changed")
    attempt_root = Path(str(adapter.get("attempt_root") or ""))
    if not attempt_root.is_absolute() or attempt_root.is_symlink():
        raise ValueError("g1_review_delivery_attempt_root_invalid")
    return _stage_review_artifacts(
        review=review, source_root=attempt_root / "immutable_execution",
        result_root=result_root, run_id=run_id,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-result", type=Path, required=True)
    parser.add_argument("--bundle-receipt", type=Path, required=True)
    parser.add_argument("--retained-review", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    result = materialize_g1_private_review_delivery(
        adapter_result_path=args.adapter_result,
        bundle_receipt_path=args.bundle_receipt,
        retained_review_path=args.retained_review,
        result_root=args.result_root,
        run_id=args.run_id,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
