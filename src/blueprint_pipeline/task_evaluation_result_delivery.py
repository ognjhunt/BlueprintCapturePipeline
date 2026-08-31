"""Seal one Task Evaluation Run for private, tenant-scoped WebApp delivery.

The episode evidence index remains the scientific inventory.  This module
verifies every referenced byte, creates deterministic review/full-evidence
packages, and emits only a small secret-clean projection for the WebApp.  The
WebApp never becomes the authority for scores or evidence completeness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .adp_episode_evidence_index import INDEX_FILENAME, INDEX_SCHEMA_VERSION
from .core.security_controls import strict_identifier
from .decision_evidence_contracts import canonical_digest, canonical_json


DELIVERY_SCHEMA_VERSION = "task_evaluation_result_delivery.v1"
POLICY_CANARY_DELIVERY_SCHEMA_VERSION = "task_evaluation_result_delivery.v2"
REGISTRY_SCHEMA_VERSION = "task_evaluation_result_artifact_registry.v1"
_FIXED_ZIP_TIME = (2020, 1, 1, 0, 0, 0)


class TaskEvaluationResultDeliveryError(ValueError):
    """Fail-closed delivery construction error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _artifact_id(role: str, relative_path: str, digest: str) -> str:
    return hashlib.sha256(f"{role}\0{relative_path}\0{digest}".encode("utf-8")).hexdigest()[:32]


def _inside(root: Path, relative_path: str, *, role: str) -> Path:
    if not relative_path or relative_path.startswith("/"):
        raise TaskEvaluationResultDeliveryError(f"delivery_artifact_path_invalid:{role}")
    unresolved = root / relative_path
    if unresolved.is_symlink():
        raise TaskEvaluationResultDeliveryError(f"delivery_artifact_symlink_forbidden:{role}")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise TaskEvaluationResultDeliveryError(
            f"delivery_artifact_outside_evidence_root:{role}"
        ) from exc
    if not resolved.is_file():
        raise TaskEvaluationResultDeliveryError(f"delivery_artifact_missing:{role}")
    return resolved


def _expected_digest(value: Any) -> str:
    text = str(value or "")
    if not text.startswith("sha256:"):
        text = "sha256:" + text
    if len(text) != 71:
        raise TaskEvaluationResultDeliveryError("delivery_artifact_digest_invalid")
    return text


def _verify_record(
    root: Path,
    record: Mapping[str, Any],
    *,
    role: str,
    require_size: bool = True,
) -> dict[str, Any]:
    relative_path = str(record.get("relative_path") or "")
    path = _inside(root, relative_path, role=role)
    expected = _expected_digest(record.get("sha256") or record.get("png_sha256"))
    observed = _sha256(path)
    if observed != expected:
        raise TaskEvaluationResultDeliveryError(f"delivery_artifact_digest_mismatch:{role}")
    size = path.stat().st_size
    if require_size and int(record.get("size_bytes", -1)) != size:
        raise TaskEvaluationResultDeliveryError(f"delivery_artifact_size_mismatch:{role}")
    return {
        "role": role,
        "relative_path": relative_path,
        "sha256": observed,
        "size_bytes": size,
        "content_type": _content_type(path),
    }


def _content_type(path: Path) -> str:
    return {
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".csv": "text/csv",
        ".html": "text/html",
        ".mp4": "video/mp4",
        ".png": "image/png",
        ".zip": "application/zip",
    }.get(path.suffix.lower(), "application/octet-stream")


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise TaskEvaluationResultDeliveryError(
                f"immutable_result_delivery_conflict:{path.name}"
            )


def _write_zip_immutable(path: Path, files: list[tuple[str, bytes | Path]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for name, source in sorted(files, key=lambda row: row[0]):
                info = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                if isinstance(source, bytes):
                    archive.writestr(info, source)
                else:
                    with (
                        source.open("rb") as input_stream,
                        archive.open(info, "w", force_zip64=True) as output_stream,
                    ):
                        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        if path.exists():
            if path.stat().st_size != temporary_path.stat().st_size or _sha256(path) != _sha256(
                temporary_path
            ):
                raise TaskEvaluationResultDeliveryError(
                    f"immutable_result_delivery_conflict:{path.name}"
                )
        else:
            os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _scenario_csv(episodes: list[Mapping[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "episode_id",
            "episode_kind",
            "subject_id",
            "score_status",
            "outcome",
            "task_succeeded",
            "grader_authority",
        ]
    )
    for row in episodes:
        score = row.get("score") if isinstance(row.get("score"), Mapping) else {}
        writer.writerow(
            [
                row.get("episode_id"),
                row.get("episode_kind"),
                row.get("subject_id"),
                score.get("status"),
                score.get("outcome"),
                score.get("task_succeeded"),
                score.get("grader_authority"),
            ]
        )
    return output.getvalue().encode("utf-8")


def _blocked_delivery(
    *, run_id: str, state: str, decision_envelope: Mapping[str, Any], blocker: str
) -> dict[str, Any]:
    delivery: dict[str, Any] = {
        "schema_version": DELIVERY_SCHEMA_VERSION,
        "run_id": run_id,
        "state": state,
        "status": "blocked",
        "claim_class": "development_only",
        "decision_envelope_digest": decision_envelope.get("decision_envelope_digest"),
        "stages": [
            {"stage": "validate", "status": "blocked"},
            {"stage": "seal", "status": "waiting"},
            {"stage": "project", "status": "waiting"},
            {"stage": "package", "status": "waiting"},
            {"stage": "publish", "status": "waiting"},
        ],
        "blockers": [blocker],
        "summary": {
            "episode_count": 0,
            "learned_candidate_episode_count": 0,
            "control_episode_count": 0,
            "successful_episode_count": 0,
        },
        "episodes": [],
        "artifacts": [],
        "proof_boundary": {
            "review_video_is_authoritative_evidence": False,
            "simulation_is_physical_success": False,
            "cross_team_leaderboard_authorized": False,
        },
        "delivery_digest": "",
    }
    delivery["delivery_digest"] = canonical_digest(delivery, digest_field="delivery_digest")
    return delivery


def main(argv: list[str] | None = None) -> int:
    """Materialize one terminal Task Evaluation result-delivery projection."""

    parser = argparse.ArgumentParser(
        description="Verify and materialize a Task Evaluation result delivery."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--state",
        choices=("decided", "partially_decided", "abstained"),
        required=True,
    )
    parser.add_argument("--decision-envelope", type=Path, required=True)
    parser.add_argument("--episode-evidence-index", type=Path)
    args = parser.parse_args(argv)
    try:
        decision_envelope = json.loads(
            args.decision_envelope.expanduser().read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationResultDeliveryError(
            "result_delivery_decision_envelope_unreadable"
        ) from exc
    if not isinstance(decision_envelope, Mapping):
        raise TaskEvaluationResultDeliveryError(
            "result_delivery_decision_envelope_invalid"
        )
    result = materialize_task_evaluation_result_delivery(
        run_root=args.run_root,
        run_id=args.run_id,
        state=args.state,
        decision_envelope=decision_envelope,
        episode_evidence_index_path=args.episode_evidence_index,
    )
    print(canonical_json(result))
    return 0


def materialize_task_evaluation_result_delivery(
    *,
    run_root: str | Path,
    run_id: str,
    state: str,
    decision_envelope: Mapping[str, Any],
    episode_evidence_index_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify, seal, project, package, and prepare one result publication."""

    run = strict_identifier(run_id, field="run_id", max_length=192)
    root = Path(run_root).expanduser().resolve()
    if state not in {"decided", "partially_decided", "abstained"}:
        raise TaskEvaluationResultDeliveryError("result_delivery_state_not_terminal")
    index_path = (
        Path(episode_evidence_index_path).expanduser().resolve()
        if episode_evidence_index_path
        else root / "artifacts" / INDEX_FILENAME
    )
    if not index_path.is_file():
        return _blocked_delivery(
            run_id=run,
            state=state,
            decision_envelope=decision_envelope,
            blocker="episode_evidence_index_missing",
        )
    evidence_root = index_path.parent.resolve()
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationResultDeliveryError("episode_evidence_index_unreadable") from exc
    if not isinstance(index, dict) or index.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise TaskEvaluationResultDeliveryError("episode_evidence_index_schema_invalid")
    if index.get("index_digest") != canonical_digest(index, digest_field="index_digest"):
        raise TaskEvaluationResultDeliveryError("episode_evidence_index_digest_mismatch")
    episodes = index.get("episodes")
    if not isinstance(episodes, list) or int(index.get("episode_count", -1)) != len(episodes):
        raise TaskEvaluationResultDeliveryError("episode_evidence_index_count_mismatch")
    if state != "abstained" and not episodes:
        raise TaskEvaluationResultDeliveryError("terminal_result_episode_evidence_missing")
    if index.get("required_camera_ids") != ["external", "wrist", "overview"]:
        raise TaskEvaluationResultDeliveryError("result_delivery_camera_contract_mismatch")
    if index.get("overview_is_review_only") is not True:
        raise TaskEvaluationResultDeliveryError("result_delivery_overview_not_review_only")
    run_identity = index.get("run_identity")
    if not isinstance(run_identity, Mapping):
        raise TaskEvaluationResultDeliveryError("result_delivery_run_identity_invalid")
    claim_class = str(run_identity.get("claim_class") or "development_only")
    if claim_class not in {"development_only", "evaluation"}:
        raise TaskEvaluationResultDeliveryError("result_delivery_claim_class_invalid")

    curated: dict[str, dict[str, Any]] = {}
    supporting = index.get("supporting_evidence")
    if not isinstance(supporting, list):
        raise TaskEvaluationResultDeliveryError("result_delivery_supporting_evidence_invalid")
    for position, record in enumerate(supporting):
        if not isinstance(record, Mapping):
            raise TaskEvaluationResultDeliveryError("result_delivery_supporting_evidence_invalid")
        role = str(record.get("role") or f"supporting_evidence:{position}")
        verified = _verify_record(evidence_root, record, role=f"supporting:{role}")
        curated[verified["relative_path"]] = verified
    episode_projection: list[dict[str, Any]] = []
    for raw in episodes:
        if not isinstance(raw, Mapping):
            raise TaskEvaluationResultDeliveryError("result_delivery_episode_invalid")
        episode_id = strict_identifier(
            str(raw.get("episode_id") or ""), field="episode_id", max_length=192
        )
        episode_kind = str(raw.get("episode_kind") or "")
        if episode_kind not in {"control", "learned_candidate"}:
            raise TaskEvaluationResultDeliveryError("result_delivery_episode_kind_invalid")
        artifact_bindings: dict[str, Any] = {}
        for role, record, require_size in (
            ("receipt", raw.get("receipt"), False),
            ("frame_manifest", raw.get("frame_manifest"), True),
        ):
            if not isinstance(record, Mapping):
                raise TaskEvaluationResultDeliveryError(
                    f"result_delivery_episode_artifact_missing:{episode_id}:{role}"
                )
            verified = _verify_record(
                evidence_root, record, role=f"{episode_id}:{role}", require_size=require_size
            )
            curated[verified["relative_path"]] = verified
            artifact_bindings[role] = verified
        videos = raw.get("videos")
        if not isinstance(videos, Mapping):
            raise TaskEvaluationResultDeliveryError(
                f"result_delivery_episode_videos_missing:{episode_id}"
            )
        video_bindings: dict[str, Any] = {}
        for camera_id in ("external", "wrist", "overview"):
            record = videos.get(camera_id)
            if not isinstance(record, Mapping):
                raise TaskEvaluationResultDeliveryError(
                    f"result_delivery_camera_video_missing:{episode_id}:{camera_id}"
                )
            verified = _verify_record(evidence_root, record, role=f"{episode_id}:video:{camera_id}")
            curated[verified["relative_path"]] = verified
            video_bindings[camera_id] = verified
        for group in ("lossless_camera_frames", "exact_policy_input_frames"):
            rows = raw.get(group)
            if not isinstance(rows, list) or (episode_kind == "learned_candidate" and not rows):
                raise TaskEvaluationResultDeliveryError(
                    f"result_delivery_{group}_missing:{episode_id}"
                )
            for position, record in enumerate(rows):
                if not isinstance(record, Mapping):
                    raise TaskEvaluationResultDeliveryError(
                        f"result_delivery_{group}_invalid:{episode_id}"
                    )
                verified = _verify_record(
                    evidence_root,
                    record,
                    role=f"{episode_id}:{group}:{position}",
                )
                curated[verified["relative_path"]] = verified
        score = dict(raw.get("score") or {})
        episode_projection.append(
            {
                "episode_id": episode_id,
                "episode_kind": episode_kind,
                "subject_id": str(raw.get("subject_id") or ""),
                "score": score,
                "artifacts": {
                    "receipt": artifact_bindings["receipt"],
                    "frame_manifest": artifact_bindings["frame_manifest"],
                    "videos": video_bindings,
                },
            }
        )

    delivery_root = root / "artifacts" / "result_delivery"
    source_index_record = {
        "role": "authoritative_episode_evidence_index",
        "relative_path": index_path.relative_to(evidence_root).as_posix(),
        "sha256": _sha256(index_path),
        "size_bytes": index_path.stat().st_size,
        "content_type": "application/json",
    }
    curated[source_index_record["relative_path"]] = source_index_record
    summary = {
        "schema_version": "task_evaluation_result_summary.v1",
        "run_id": run,
        "state": state,
        "decision_envelope_digest": decision_envelope.get("decision_envelope_digest"),
        "episode_evidence_index_digest": index["index_digest"],
        "episode_count": len(episodes),
        "learned_candidate_episode_count": sum(
            row.get("episode_kind") == "learned_candidate" for row in episodes
        ),
        "control_episode_count": sum(row.get("episode_kind") == "control" for row in episodes),
        "successful_episode_count": sum(
            (row.get("score") or {}).get("task_succeeded") is True for row in episodes
        ),
        "claim_class": claim_class,
        "decision_envelope": dict(decision_envelope),
    }
    metadata_files = [
        ("bagit.txt", b"BagIt-Version: 1.0\nTag-File-Character-Encoding: UTF-8\n"),
        (
            "bag-info.txt",
            (
                f"External-Identifier: {run}\n"
                f"Source-Organization: Blueprint\n"
                f"Payload-Oxum: {sum(row['size_bytes'] for row in curated.values())}.{len(curated)}\n"
            ).encode("utf-8"),
        ),
        (
            "README.txt",
            b"Blueprint Task Evaluation Run evidence. Review videos are derived navigation aids; receipts and exact lossless frames remain authoritative simulation evidence. This package is not physical success, deployment approval, or a cross-team ranking.\n",
        ),
        ("summary.json", (canonical_json(summary) + "\n").encode("utf-8")),
        ("scenarios.csv", _scenario_csv(episodes)),
        (
            "ro-crate-metadata.json",
            (
                canonical_json(
                    {
                        "@context": "https://w3id.org/ro/crate/1.1/context",
                        "@graph": [
                            {
                                "@id": "./",
                                "@type": "Dataset",
                                "name": "Blueprint Task Evaluation Run evidence",
                                "identifier": run,
                            }
                        ],
                    }
                )
                + "\n"
            ).encode("utf-8"),
        ),
    ]
    review_files: list[tuple[str, bytes | Path]] = list(metadata_files)
    full_files: list[tuple[str, bytes | Path]] = list(metadata_files)
    for record in sorted(curated.values(), key=lambda row: row["relative_path"]):
        content = evidence_root / record["relative_path"]
        archive_name = f"data/{record['relative_path']}"
        full_files.append((archive_name, content))
        if (
            record["content_type"] == "video/mp4"
            or record["role"]
            in {
                "authoritative_episode_evidence_index",
            }
            or record["role"].endswith(":receipt")
            or record["role"].endswith(":frame_manifest")
        ):
            review_files.append((archive_name, content))

    def with_manifest(
        files: list[tuple[str, bytes | Path]],
    ) -> list[tuple[str, bytes | Path]]:
        rows = []
        for name, source in sorted(files, key=lambda row: row[0]):
            if isinstance(source, bytes):
                digest = hashlib.sha256(source).hexdigest()
            else:
                digest = _sha256(source).removeprefix("sha256:")
            rows.append(f"{digest}  {name}")
        return [*files, ("manifest-sha256.txt", ("\n".join(rows) + "\n").encode("utf-8"))]

    review_files = with_manifest(review_files)
    full_files = with_manifest(full_files)

    def source_size(source: bytes | Path) -> int:
        return len(source) if isinstance(source, bytes) else source.stat().st_size

    projected_size = sum(source_size(content) for _, content in full_files) + sum(
        source_size(content) for _, content in review_files
    )
    if shutil.disk_usage(delivery_root.parent).free < projected_size + 64 * 1024 * 1024:
        raise TaskEvaluationResultDeliveryError("result_delivery_insufficient_local_disk")
    review_path = delivery_root / "review_pack.zip"
    full_path = delivery_root / "full_evidence.zip"
    _write_zip_immutable(review_path, review_files)
    _write_zip_immutable(full_path, full_files)

    public_artifacts: list[dict[str, Any]] = []
    registry_artifacts: list[dict[str, Any]] = []
    for record in sorted(curated.values(), key=lambda row: row["relative_path"]):
        artifact = dict(record)
        artifact["artifact_id"] = _artifact_id(
            artifact["role"], artifact["relative_path"], artifact["sha256"]
        )
        artifact["evidence_root"] = str(evidence_root)
        registry_artifacts.append(artifact)
        if artifact["content_type"] in {"video/mp4", "application/json"} and (
            ":video:" in artifact["role"]
            or artifact["role"].endswith(":receipt")
            or artifact["role"].endswith(":frame_manifest")
            or artifact["role"] == "authoritative_episode_evidence_index"
        ):
            public_artifacts.append(
                {key: value for key, value in artifact.items() if key != "evidence_root"}
            )
    for role, path in (("review_package", review_path), ("full_evidence_package", full_path)):
        artifact = {
            "role": role,
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "content_type": "application/zip",
            "evidence_root": str(root),
        }
        artifact["artifact_id"] = _artifact_id(role, artifact["relative_path"], artifact["sha256"])
        registry_artifacts.append(artifact)
        public_artifacts.append(
            {key: value for key, value in artifact.items() if key != "evidence_root"}
        )

    artifact_by_path = {row["relative_path"]: row for row in public_artifacts}
    for episode in episode_projection:
        for key in ("receipt", "frame_manifest"):
            source = episode["artifacts"][key]
            episode["artifacts"][key] = artifact_by_path[source["relative_path"]]
        episode["artifacts"]["videos"] = {
            camera: artifact_by_path[source["relative_path"]]
            for camera, source in episode["artifacts"]["videos"].items()
        }

    delivery: dict[str, Any] = {
        "schema_version": DELIVERY_SCHEMA_VERSION,
        "run_id": run,
        "state": state,
        "status": "ready",
        "claim_class": claim_class,
        "decision_envelope_digest": decision_envelope.get("decision_envelope_digest"),
        "episode_evidence_index_digest": index["index_digest"],
        "stages": [
            {"stage": "validate", "status": "complete"},
            {"stage": "seal", "status": "complete"},
            {"stage": "project", "status": "complete"},
            {"stage": "package", "status": "complete"},
            {"stage": "publish", "status": "ready"},
        ],
        "blockers": [],
        "summary": {
            key: summary[key]
            for key in (
                "episode_count",
                "learned_candidate_episode_count",
                "control_episode_count",
                "successful_episode_count",
            )
        },
        "episodes": episode_projection,
        "artifacts": sorted(public_artifacts, key=lambda row: (row["role"], row["artifact_id"])),
        "proof_boundary": {
            "review_video_is_authoritative_evidence": False,
            "simulation_is_physical_success": False,
            "cross_team_leaderboard_authorized": False,
        },
        "delivery_digest": "",
    }
    delivery["delivery_digest"] = canonical_digest(delivery, digest_field="delivery_digest")
    registry: dict[str, Any] = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "run_id": run,
        "delivery_digest": delivery["delivery_digest"],
        "artifacts": registry_artifacts,
        "registry_digest": "",
    }
    registry["registry_digest"] = canonical_digest(registry, digest_field="registry_digest")
    _write_immutable(
        delivery_root / "delivery.json",
        (canonical_json(delivery) + "\n").encode("utf-8"),
    )
    _write_immutable(
        delivery_root / "artifact_registry.json",
        (canonical_json(registry) + "\n").encode("utf-8"),
    )
    return delivery


def resolve_task_evaluation_result_artifact(
    *, run_root: str | Path, run_id: str, artifact_id: str
) -> tuple[Path, dict[str, Any]]:
    """Resolve one allowlisted sealed artifact and reverify it before serving."""

    run = strict_identifier(run_id, field="run_id", max_length=192)
    requested = strict_identifier(artifact_id, field="artifact_id", max_length=64)
    root = Path(run_root).expanduser().resolve()
    registry_path = root / "artifacts" / "result_delivery" / "artifact_registry.json"
    if not registry_path.is_file():
        raise TaskEvaluationResultDeliveryError("result_delivery_registry_missing")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION or registry.get(
        "registry_digest"
    ) != canonical_digest(registry, digest_field="registry_digest"):
        raise TaskEvaluationResultDeliveryError("result_delivery_registry_invalid")
    if registry.get("run_id") != run:
        raise TaskEvaluationResultDeliveryError("result_delivery_registry_run_mismatch")
    matches = [row for row in registry.get("artifacts", []) if row.get("artifact_id") == requested]
    if len(matches) != 1:
        raise TaskEvaluationResultDeliveryError("result_delivery_artifact_not_found")
    record = matches[0]
    evidence_root = Path(str(record.get("evidence_root") or "")).resolve()
    path = _inside(evidence_root, str(record.get("relative_path") or ""), role=requested)
    if _sha256(path) != record.get("sha256") or path.stat().st_size != record.get("size_bytes"):
        raise TaskEvaluationResultDeliveryError("result_delivery_artifact_reverification_failed")
    return path, record


def materialize_policy_canary_result_delivery(
    *,
    run_root: str | Path,
    run_id: str,
    result_status: str,
    session_result: Mapping[str, Any],
    evidence_root: str | Path,
    closure_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal a private, artifact-complete internal policy-canary delivery.

    The provider result remains the source of episode truth.  This function
    verifies every exported byte, adds deterministic CSV/JSON downloads, and
    refuses a completed result unless official billing, teardown, and a fresh
    authenticated provider-zero receipt are all explicitly bound.
    """

    run = strict_identifier(run_id, field="run_id", max_length=192)
    if result_status not in {"completed_unqualified", "blocked", "cancelled"}:
        raise TaskEvaluationResultDeliveryError(
            "policy_canary_result_delivery_status_invalid"
        )
    result = json.loads(json.dumps(dict(session_result), allow_nan=False))
    if (
        result.get("run_kind") != "internal_policy_canary"
        or result.get("claim_ceiling") != "diagnostic_policy_execution"
        or result.get("result_digest")
        != canonical_digest(result, digest_field="result_digest")
    ):
        raise TaskEvaluationResultDeliveryError(
            "policy_canary_result_delivery_result_invalid"
        )
    episodes = result.get("episodes")
    if not isinstance(episodes, list) or len(episodes) > 20:
        raise TaskEvaluationResultDeliveryError(
            "policy_canary_result_delivery_episodes_invalid"
        )

    root = Path(run_root).expanduser().resolve()
    evidence = Path(evidence_root).expanduser().resolve()
    if not evidence.is_dir() or evidence.is_symlink():
        raise TaskEvaluationResultDeliveryError(
            "policy_canary_result_delivery_evidence_root_invalid"
        )
    delivery_root = root / "artifacts" / "result_delivery"
    delivery_root.mkdir(parents=True, exist_ok=True)
    registry_artifacts: list[dict[str, Any]] = []
    public_artifacts: list[dict[str, Any]] = []

    def add_artifact(*, role: str, path: Path, artifact_root: Path) -> dict[str, Any]:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(artifact_root.resolve()).as_posix()
        except ValueError as exc:
            raise TaskEvaluationResultDeliveryError(
                f"policy_canary_artifact_outside_root:{role}"
            ) from exc
        if path.is_symlink() or not resolved.is_file():
            raise TaskEvaluationResultDeliveryError(
                f"policy_canary_artifact_invalid:{role}"
            )
        digest = _sha256(resolved)
        artifact_id = _artifact_id(role, relative, digest)
        public = {
            "artifact_id": artifact_id,
            "role": role,
            "digest": digest,
            "size_bytes": resolved.stat().st_size,
            "content_type": _content_type(resolved),
        }
        public_artifacts.append(public)
        registry_artifacts.append(
            {
                **public,
                "sha256": digest,
                "relative_path": relative,
                "evidence_root": str(artifact_root.resolve()),
            }
        )
        return {
            "artifact_id": artifact_id,
            "digest": digest,
            "size_bytes": resolved.stat().st_size,
        }

    seen_paths: set[str] = set()
    inventory = result.get("artifact_inventory")
    if not isinstance(inventory, list):
        raise TaskEvaluationResultDeliveryError(
            "policy_canary_artifact_inventory_missing"
        )
    for position, row in enumerate(inventory):
        if not isinstance(row, Mapping):
            raise TaskEvaluationResultDeliveryError(
                "policy_canary_artifact_inventory_invalid"
            )
        relative = str(row.get("relative_path") or "")
        role = str(row.get("role") or f"provider_artifact_{position}")
        path = _inside(evidence, relative, role=role)
        if (
            relative in seen_paths
            or _sha256(path) != row.get("sha256")
            or path.stat().st_size != row.get("size_bytes")
        ):
            raise TaskEvaluationResultDeliveryError(
                "policy_canary_artifact_inventory_invalid"
            )
        seen_paths.add(relative)
        add_artifact(role=role, path=path, artifact_root=evidence)

    closure: dict[str, dict[str, Any]] = {}
    required_flags = {
        "billing": "official_billing_sealed",
        "teardown": "teardown_completed",
        "provider_zero": "provider_zero_verified",
    }
    for role, flag in required_flags.items():
        record = closure_records.get(role)
        if not isinstance(record, Mapping) or record.get(flag) is not True:
            raise TaskEvaluationResultDeliveryError(
                f"policy_canary_{role}_receipt_missing"
            )
        path = Path(str(record.get("path") or "")).expanduser().resolve()
        if (
            path.is_symlink()
            or not path.is_file()
            or _sha256(path) != record.get("sha256")
            or path.stat().st_size != record.get("size_bytes")
        ):
            raise TaskEvaluationResultDeliveryError(
                f"policy_canary_{role}_receipt_invalid"
            )
        descriptor = add_artifact(
            role=f"closure_{role}", path=path, artifact_root=path.parent
        )
        if role == "provider_zero":
            descriptor["provider_zero_verified"] = True
        closure[role] = descriptor

    report_path = delivery_root / "policy_canary_full_report.json"
    _write_immutable(
        report_path, (canonical_json(result) + "\n").encode("utf-8")
    )
    report_artifact = add_artifact(
        role="machine_readable_report",
        path=report_path,
        artifact_root=delivery_root,
    )
    episode_csv = io.StringIO(newline="")
    episode_writer = csv.writer(episode_csv, lineterminator="\n")
    episode_writer.writerow(
        [
            "candidate_id",
            "cell_id",
            "seed",
            "status",
            "candidate_policy_queried",
            "actions_reached_robot",
            "arm_moved",
            "policy_outcome_interpretable",
            "typed_harness_failure",
        ]
    )
    for row in episodes:
        episode_writer.writerow(
            [
                row.get("candidate_id"),
                row.get("cell_id"),
                row.get("seed"),
                row.get("status"),
                row.get("candidate_policy_queried"),
                row.get("actions_reached_robot"),
                row.get("arm_moved"),
                row.get("policy_outcome_interpretable"),
                row.get("typed_harness_failure"),
            ]
        )
    episode_csv_path = delivery_root / "policy_canary_episodes.csv"
    _write_immutable(episode_csv_path, episode_csv.getvalue().encode("utf-8"))
    add_artifact(
        role="episode_csv", path=episode_csv_path, artifact_root=delivery_root
    )

    evidence_manifest = {
        "schema_version": "task_evaluation_policy_canary_evidence_manifest.v1",
        "run_id": run,
        "result_digest": result["result_digest"],
        "artifacts": sorted(
            public_artifacts, key=lambda row: (row["role"], row["artifact_id"])
        ),
        "manifest_digest": "",
    }
    evidence_manifest["manifest_digest"] = canonical_digest(
        evidence_manifest, digest_field="manifest_digest"
    )
    evidence_manifest_path = delivery_root / "policy_canary_evidence_manifest.json"
    _write_immutable(
        evidence_manifest_path,
        (canonical_json(evidence_manifest) + "\n").encode("utf-8"),
    )
    evidence_manifest_artifact = add_artifact(
        role="evidence_manifest",
        path=evidence_manifest_path,
        artifact_root=delivery_root,
    )

    completed = sum(row.get("status") == "completed" for row in episodes)
    delivery: dict[str, Any] = {
        "schema_version": POLICY_CANARY_DELIVERY_SCHEMA_VERSION,
        "run_id": run,
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "result_status": result_status,
        "status": "ready",
        "summary": {
            "policy_count": 2,
            "episodes_per_policy": 10,
            "learned_policy_rollout_count": 20,
            "completed_learned_policy_rollout_count": completed,
        },
        "episodes": [
            {
                key: row.get(key)
                for key in (
                    "candidate_id",
                    "cell_id",
                    "seed",
                    "status",
                    "candidate_policy_queried",
                    "actions_reached_robot",
                    "arm_moved",
                    "policy_outcome_interpretable",
                    "typed_harness_failure",
                )
            }
            for row in episodes
        ],
        "artifacts": sorted(
            public_artifacts, key=lambda row: (row["role"], row["artifact_id"])
        ),
        "report": {
            "machine_readable_report": report_artifact,
            "evidence_manifest": evidence_manifest_artifact,
        },
        "closure": closure,
        "proof_boundary": {
            "controls_pending_results_unqualified": True,
            "scene_promotion_authorized": False,
            "official_policy_ranking_authorized": False,
            "winner_selection_authorized": False,
            "simulation_is_physical_success": False,
        },
        "delivery_digest": "",
    }
    delivery["delivery_digest"] = canonical_digest(
        delivery, digest_field="delivery_digest"
    )
    registry: dict[str, Any] = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "run_id": run,
        "delivery_digest": delivery["delivery_digest"],
        "artifacts": registry_artifacts,
        "registry_digest": "",
    }
    registry["registry_digest"] = canonical_digest(
        registry, digest_field="registry_digest"
    )
    _write_immutable(
        delivery_root / "delivery.json",
        (canonical_json(delivery) + "\n").encode("utf-8"),
    )
    _write_immutable(
        delivery_root / "artifact_registry.json",
        (canonical_json(registry) + "\n").encode("utf-8"),
    )
    return delivery


__all__ = [
    "DELIVERY_SCHEMA_VERSION",
    "POLICY_CANARY_DELIVERY_SCHEMA_VERSION",
    "TaskEvaluationResultDeliveryError",
    "materialize_policy_canary_result_delivery",
    "materialize_task_evaluation_result_delivery",
    "resolve_task_evaluation_result_artifact",
]


if __name__ == "__main__":  # pragma: no cover - exercised through module CLI
    raise SystemExit(main())
