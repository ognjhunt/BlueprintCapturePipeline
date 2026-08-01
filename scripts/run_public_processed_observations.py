#!/usr/bin/env python3
"""Replay a public processed RGB-D/pose dataset through bounded Blueprint lanes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint_pipeline.processed_observation_dataset import (  # noqa: E402
    REQUEST_SCHEMA_VERSION,
    build_processed_observation_dataset_request,
    compile_bound_processed_observation_dataset,
)
from blueprint_pipeline.reconstruction_colmap_dataset import (  # noqa: E402
    export_colmap_training_dataset,
)


SCHEMA_VERSION = "public_processed_observation_replay.v1"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class PublicProcessedObservationReplayError(RuntimeError):
    """Stable fail-closed operator replay error."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise PublicProcessedObservationReplayError(
                "public_processed_observation_summary_conflict"
            )


def _source_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PublicProcessedObservationReplayError(
            "source_commit_unavailable"
        ) from exc
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PublicProcessedObservationReplayError("source_commit_invalid")
    if dirty:
        raise PublicProcessedObservationReplayError("source_checkout_not_clean")
    return commit


def _load_bound_artifact(
    *, output_root: Path, reference: Mapping[str, Any]
) -> dict[str, Any]:
    relative_path = str(reference.get("relative_path") or "")
    path = (output_root / relative_path).resolve()
    if path != output_root and output_root not in path.parents:
        raise PublicProcessedObservationReplayError("artifact_reference_path_escape")
    if path.is_symlink() or not path.is_file():
        raise PublicProcessedObservationReplayError("artifact_reference_missing")
    if _sha256_file(path) != reference.get("digest"):
        raise PublicProcessedObservationReplayError(
            "artifact_reference_digest_mismatch"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicProcessedObservationReplayError(
            "artifact_reference_json_invalid"
        ) from exc
    if not isinstance(value, Mapping):
        raise PublicProcessedObservationReplayError("artifact_reference_not_object")
    return dict(value)


def run_public_processed_observation_replay(
    *,
    dataset_id: str,
    scene_id: str,
    source_bundle: str | Path,
    source_bundle_sha256: str,
    source_bundle_uri: str,
    license_id: str,
    dataset_root: str | Path,
    long_transformations_relative_path: str,
    declared_heldout_ids_relative_path: str,
    independent_transformations_relative_path: str,
    output_root: str | Path,
    operator_identity: str,
    source_commit: str,
    timestamp: str,
) -> dict[str, Any]:
    bundle = Path(source_bundle).expanduser()
    dataset = Path(dataset_root).expanduser()
    output = Path(output_root).expanduser()
    if any(path.is_symlink() for path in (bundle, dataset, output)):
        raise PublicProcessedObservationReplayError("replay_path_symlink_forbidden")
    bundle = bundle.resolve()
    dataset = dataset.resolve()
    output = output.resolve()
    if not bundle.is_file() or not dataset.is_dir():
        raise PublicProcessedObservationReplayError("replay_source_missing")
    if not _SHA256.fullmatch(source_bundle_sha256):
        raise PublicProcessedObservationReplayError("source_bundle_digest_invalid")
    if _sha256_file(bundle) != source_bundle_sha256:
        raise PublicProcessedObservationReplayError("source_bundle_digest_mismatch")
    if not operator_identity.strip():
        raise PublicProcessedObservationReplayError("operator_identity_missing")
    request = build_processed_observation_dataset_request(
        {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "scene_id": scene_id,
            "source_bundle_digest": source_bundle_sha256,
            "source_bundle_size_bytes": bundle.stat().st_size,
            "source_bundle_uri": source_bundle_uri,
            "license_id": license_id,
            "long_transformations_relative_path": long_transformations_relative_path,
            "declared_heldout_ids_relative_path": declared_heldout_ids_relative_path,
            "independent_transformations_relative_path": independent_transformations_relative_path,
            "source_commit_sha": source_commit,
            "authority_used": {
                "local_processing_allowed": True,
                "external_provider_upload_allowed": False,
                "privacy_scope": "restricted_local_only",
                "operator_identity": operator_identity,
                "license_id": license_id,
            },
            "coordinate_frame_declaration": {
                "source": "dataset_transformations_json",
                "camera_convention": "camera_to_world",
                "world_up": "not_independently_verified",
                "metric_scale": "dataset_declared_not_independently_verified",
            },
            "timestamp": timestamp,
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    compiled = compile_bound_processed_observation_dataset(
        source_artifact=request,
        source_bundle=bundle,
        dataset_root=dataset,
        output_root=output,
    )
    schema = json.loads(
        (
            REPO_ROOT
            / "docs"
            / "schemas"
            / "processed_observation_dataset.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(compiled, schema)
    artifacts = {
        name: _load_bound_artifact(output_root=output, reference=reference)
        for name, reference in compiled["artifact_references"].items()
    }
    for value in artifacts.values():
        jsonschema.validate(value, schema)
    split = artifacts["frozen_split_manifest"]
    candidate = artifacts["candidate_dataset_manifest"]
    observations = artifacts["candidate_camera_observation_manifest"]
    colmap = export_colmap_training_dataset(
        source_artifact={
            "schema_version": "colmap_training_dataset_export_request.v1",
            "stable_run_identity": compiled["stable_run_identity"],
            "source_capture_digest": compiled["source_capture_digest"],
            "reconstruction_dataset_digest": compiled["dataset_manifest_digest"],
            "frozen_split_digest": split["split_digest"],
            "source_commit_sha": source_commit,
            "camera_observation_manifest": observations,
            "candidate_dataset_manifest": candidate,
            "coordinate_frame_declaration": compiled[
                "coordinate_frame_declaration"
            ],
            "units": compiled["units"],
            "metric_scale_status": compiled["metric_scale_status"],
            "authority_used": compiled["authority_used"],
            "timestamp": timestamp,
        },
        artifact_root=output / compiled["relative_path"],
        output_root=output / "colmap",
    )
    flags = compiled["claim_flags"]
    expected_false = (
        "raw_capture_authority",
        "decoded_video_timing",
        "metric_scale_verified",
        "collision_geometry",
        "physics",
        "physical_task_success",
        "deployment_readiness",
        "safety_certification",
    )
    if (
        flags.get("processed_captured_observation") is not True
        or any(flags.get(key) is not False for key in expected_false)
        or flags.get("comparative_policy_ranking_verdict")
        != "thesis_not_supported"
        or candidate.get("heldout_pixels_included") is not False
        or artifacts["hidden_heldout_evaluator_manifest"].get(
            "candidate_method_access_allowed"
        )
        is not False
        or colmap.get("hidden_heldout_pixels_included") is not False
    ):
        raise PublicProcessedObservationReplayError("replay_claim_boundary_upgraded")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "scene_id": scene_id,
        "source_commit_sha": source_commit,
        "source_bundle": {
            "source_uri": source_bundle_uri,
            "digest": source_bundle_sha256,
            "size_bytes": bundle.stat().st_size,
            "license_id": license_id,
        },
        "compile_request_digest": request[
            "processed_observation_dataset_compile_request_digest"
        ],
        "processed_dataset_digest": compiled["dataset_manifest_digest"],
        "frozen_split_digest": split["split_digest"],
        "candidate_dataset_digest": candidate["candidate_dataset_digest"],
        "hidden_heldout_digest": artifacts[
            "hidden_heldout_evaluator_manifest"
        ]["hidden_heldout_digest"],
        "camera_observation_digest": observations["camera_observation_digest"],
        "colmap_training_dataset_digest": colmap[
            "colmap_training_dataset_digest"
        ],
        "counts": dict(compiled["stream_metadata"]),
        "colmap_candidate_image_count": colmap["image_count"],
        "claim_ceiling": compiled["claim_ceiling"],
        "claim_flags": dict(flags),
        "raw_capture_gate_passed": False,
        "customer_upload_gate_passed": False,
        "production_security_gate_passed": False,
        "external_provider_upload_performed": False,
        "cost_usd": 0.0,
        "timestamp": timestamp,
    }
    _write_immutable(output / "public_processed_observation_replay.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--source-bundle-sha256", required=True)
    parser.add_argument("--source-bundle-uri", required=True)
    parser.add_argument("--license-id", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--long-transformations", required=True)
    parser.add_argument("--declared-heldout-ids", required=True)
    parser.add_argument("--independent-transformations", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--operator-identity", required=True)
    parser.add_argument("--timestamp", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_public_processed_observation_replay(
        dataset_id=args.dataset_id,
        scene_id=args.scene_id,
        source_bundle=args.source_bundle,
        source_bundle_sha256=args.source_bundle_sha256,
        source_bundle_uri=args.source_bundle_uri,
        license_id=args.license_id,
        dataset_root=args.dataset_root,
        long_transformations_relative_path=args.long_transformations,
        declared_heldout_ids_relative_path=args.declared_heldout_ids,
        independent_transformations_relative_path=args.independent_transformations,
        output_root=args.output_root,
        operator_identity=args.operator_identity,
        source_commit=_source_commit(),
        timestamp=args.timestamp,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
