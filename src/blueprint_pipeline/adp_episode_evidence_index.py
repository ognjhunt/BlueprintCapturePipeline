"""Build a portable, human-reviewable index for ADP manipulation episodes.

The episode runtime already seals lossless frames, a calibrated frame manifest,
and one review video per camera.  This module performs the missing packaging
join: it verifies those bytes from each control or learned-policy receipt and
then emits a small Finder-friendly HTML page plus a digest-bound JSON index.

The HTML is only navigation convenience.  Scores continue to come from the
deterministic simulator-state receipt, and the JSON index is the authoritative
portable inventory.
"""

from __future__ import annotations

import hashlib
import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

try:  # flat provider-bundle layout
    from decision_evidence_contracts import canonical_digest
except ModuleNotFoundError:  # repository package
    from .decision_evidence_contracts import canonical_digest

INDEX_SCHEMA_VERSION = "adp_manipulation_episode_evidence_index.v1"
INDEX_FILENAME = "episode_evidence_index.v1.json"
HTML_FILENAME = "OPEN_ME_episode_evidence_index.html"
REQUIRED_CAMERA_IDS = ("external", "wrist", "overview")
ALLOWED_RECEIPT_SCHEMAS = {
    "adp009d_control_episode.v2",
    "adp009d_policy_episode.v2",
    "adp009d_policy_episode.v3",
}


class EpisodeEvidenceIndexError(ValueError):
    """Fail-closed portable-index validation error."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EpisodeEvidenceIndexError(f"episode_receipt_unreadable:{path.name}") from exc
    if not isinstance(value, dict):
        raise EpisodeEvidenceIndexError(f"episode_receipt_not_mapping:{path.name}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, relative_path: str, *, role: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise EpisodeEvidenceIndexError(f"episode_artifact_path_outside_root:{role}") from exc
    return candidate


def _verify_artifact(
    root: Path, artifact: Mapping[str, Any], *, role: str
) -> dict[str, Any]:
    relative_path = str(artifact.get("relative_path") or "")
    if not relative_path:
        raise EpisodeEvidenceIndexError(f"episode_artifact_path_missing:{role}")
    path = _inside(root, relative_path, role=role)
    if not path.is_file():
        raise EpisodeEvidenceIndexError(f"episode_artifact_file_missing:{role}")
    expected_sha256 = str(artifact.get("sha256") or "")
    if len(expected_sha256) != 64 or _sha256(path) != expected_sha256:
        raise EpisodeEvidenceIndexError(f"episode_artifact_digest_mismatch:{role}")
    if int(artifact.get("size_bytes", -1)) != path.stat().st_size:
        raise EpisodeEvidenceIndexError(f"episode_artifact_size_mismatch:{role}")
    return {
        "relative_path": relative_path,
        "sha256": expected_sha256,
        "size_bytes": path.stat().st_size,
    }


def _episode_row(root: Path, receipt_path: Path) -> dict[str, Any]:
    receipt = _load_json(receipt_path)
    schema_version = str(receipt.get("schema_version") or "")
    if schema_version not in ALLOWED_RECEIPT_SCHEMAS:
        raise EpisodeEvidenceIndexError(
            f"episode_receipt_schema_not_admitted:{schema_version or 'missing'}"
        )
    if receipt.get("receipt_digest") != canonical_digest(
        receipt, digest_field="receipt_digest"
    ):
        raise EpisodeEvidenceIndexError("episode_receipt_digest_mismatch")

    episode_id = str(receipt.get("episode_id") or "")
    if not episode_id:
        raise EpisodeEvidenceIndexError("episode_id_missing")
    visual = receipt.get("visual_evidence")
    if not isinstance(visual, Mapping) or visual.get("status") != "complete":
        raise EpisodeEvidenceIndexError(f"episode_visual_evidence_incomplete:{episode_id}")
    if set(visual.get("required_camera_ids") or ()) != set(REQUIRED_CAMERA_IDS):
        raise EpisodeEvidenceIndexError(f"episode_required_cameras_mismatch:{episode_id}")
    if set(visual.get("review_only_camera_ids") or ()) != {"overview"}:
        raise EpisodeEvidenceIndexError(f"episode_overview_not_review_only:{episode_id}")

    artifacts = receipt.get("media_artifacts")
    if not isinstance(artifacts, list):
        raise EpisodeEvidenceIndexError(f"episode_media_artifacts_missing:{episode_id}")
    manifests = [
        row
        for row in artifacts
        if isinstance(row, Mapping)
        and row.get("role") == "multicamera_observation_frame_manifest"
    ]
    if len(manifests) != 1:
        raise EpisodeEvidenceIndexError(f"episode_frame_manifest_not_unique:{episode_id}")
    manifest = _verify_artifact(root, manifests[0], role=f"{episode_id}:manifest")

    videos: dict[str, dict[str, Any]] = {}
    for camera_id in REQUIRED_CAMERA_IDS:
        matches = [
            row
            for row in artifacts
            if isinstance(row, Mapping)
            and row.get("role") == "camera_review_video"
            and row.get("camera_id") == camera_id
        ]
        if len(matches) != 1:
            raise EpisodeEvidenceIndexError(
                f"episode_camera_video_not_unique:{episode_id}:{camera_id}"
            )
        videos[camera_id] = _verify_artifact(
            root, matches[0], role=f"{episode_id}:{camera_id}"
        )

    score = receipt.get("score")
    if not isinstance(score, Mapping) or not score.get("status"):
        raise EpisodeEvidenceIndexError(f"episode_score_missing:{episode_id}")
    receipt_relative = receipt_path.relative_to(root).as_posix()
    kind = "control" if receipt.get("control_id") else "learned_candidate"
    subject_id = str(receipt.get("control_id") or receipt.get("candidate_id") or "")
    if not subject_id:
        raise EpisodeEvidenceIndexError(f"episode_subject_missing:{episode_id}")
    return {
        "episode_id": episode_id,
        "episode_kind": kind,
        "subject_id": subject_id,
        "receipt": {
            "relative_path": receipt_relative,
            "sha256": _sha256(receipt_path),
            "receipt_digest": receipt["receipt_digest"],
        },
        "score": {
            "status": score.get("status"),
            "outcome": score.get("outcome"),
            "task_succeeded": score.get("task_succeeded"),
            "outcome_rank": score.get("outcome_rank"),
            "grader_authority": receipt.get(
                "grader_authority", "deterministic_simulator_state"
            ),
        },
        "frame_manifest": manifest,
        "videos": videos,
    }


def _render_html(payload: Mapping[str, Any]) -> str:
    rows = []
    for episode in payload["episodes"]:
        links = []
        for camera_id in REQUIRED_CAMERA_IDS:
            path = episode["videos"][camera_id]["relative_path"]
            links.append(
                f'<a href="{quote(path)}">{html.escape(camera_id)}</a>'
            )
        manifest_path = episode["frame_manifest"]["relative_path"]
        receipt_path = episode["receipt"]["relative_path"]
        score = episode["score"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(episode['episode_id'])}</td>"
            f"<td>{html.escape(episode['episode_kind'])}</td>"
            f"<td>{html.escape(episode['subject_id'])}</td>"
            f"<td>{html.escape(str(score.get('outcome')))}</td>"
            f"<td>{html.escape(str(score.get('task_succeeded')))}</td>"
            f"<td>{' | '.join(links)}</td>"
            f'<td><a href="{quote(manifest_path)}">manifest</a></td>'
            f'<td><a href="{quote(receipt_path)}">receipt</a></td>'
            "</tr>"
        )
    identity = payload["run_identity"]
    return (
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">"
        "<title>ADP episode evidence index</title>"
        "<style>body{font-family:-apple-system,sans-serif;margin:2rem}"
        "table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:.5rem}"
        "th{background:#f4f4f4;text-align:left}</style></head><body>"
        "<h1>ADP episode evidence index</h1>"
        f"<p>Scene: {html.escape(str(identity['scene_id']))} &middot; "
        f"Task: {html.escape(str(identity['task_id']))}</p>"
        "<p>Videos are derived review media. Scores come only from deterministic "
        "simulator state. Overview is review-only and was not a policy input.</p>"
        "<table><thead><tr><th>Episode</th><th>Kind</th><th>Subject</th>"
        "<th>Outcome</th><th>Success</th><th>Videos</th><th>Frames</th>"
        "<th>Receipt</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        f"<p>Authoritative JSON digest: {html.escape(payload['index_digest'])}</p>"
        f'<p><a href="{INDEX_FILENAME}">Open authoritative JSON index</a></p>'
        "</body></html>\n"
    )


def materialize_episode_evidence_index(
    *,
    run_root: str | Path,
    episode_receipt_paths: Sequence[str | Path],
    run_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify episode evidence and emit portable JSON plus HTML navigation."""

    root = Path(run_root).expanduser().resolve()
    if not root.is_dir():
        raise EpisodeEvidenceIndexError("episode_evidence_run_root_missing")
    identity = json.loads(json.dumps(dict(run_identity), allow_nan=False))
    for required in ("scene_id", "task_id", "scenario_suite_digest"):
        if not str(identity.get(required) or ""):
            raise EpisodeEvidenceIndexError(f"episode_index_identity_missing:{required}")
    if not episode_receipt_paths:
        raise EpisodeEvidenceIndexError("episode_index_receipts_missing")

    rows = []
    for raw_path in episode_receipt_paths:
        path = Path(raw_path).expanduser()
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise EpisodeEvidenceIndexError("episode_receipt_path_outside_root") from exc
        rows.append(_episode_row(root, path))
    episode_ids = [row["episode_id"] for row in rows]
    if len(set(episode_ids)) != len(episode_ids):
        raise EpisodeEvidenceIndexError("episode_index_duplicate_episode_id")

    payload: dict[str, Any] = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "run_identity": identity,
        "episodes": sorted(rows, key=lambda row: row["episode_id"]),
        "episode_count": len(rows),
        "required_camera_ids": list(REQUIRED_CAMERA_IDS),
        "overview_is_review_only": True,
        "scores_are_deterministic_simulator_state": True,
        "review_videos_are_not_physical_truth": True,
        "index_digest": "",
    }
    payload["index_digest"] = canonical_digest(payload, digest_field="index_digest")
    json_path = root / INDEX_FILENAME
    html_path = root / HTML_FILENAME
    if json_path.exists() or json_path.is_symlink() or html_path.exists() or html_path.is_symlink():
        raise EpisodeEvidenceIndexError("episode_evidence_index_overwrite_forbidden")
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    html_path.write_text(_render_html(payload), encoding="utf-8")
    return {
        "index": payload,
        "artifacts": [
            {
                "role": "authoritative_episode_evidence_index",
                "relative_path": INDEX_FILENAME,
                "sha256": _sha256(json_path),
                "size_bytes": json_path.stat().st_size,
            },
            {
                "role": "finder_friendly_episode_evidence_index",
                "relative_path": HTML_FILENAME,
                "sha256": _sha256(html_path),
                "size_bytes": html_path.stat().st_size,
                "derived_from_index_digest": payload["index_digest"],
            },
        ],
    }


__all__ = [
    "EpisodeEvidenceIndexError",
    "HTML_FILENAME",
    "INDEX_FILENAME",
    "INDEX_SCHEMA_VERSION",
    "materialize_episode_evidence_index",
]
