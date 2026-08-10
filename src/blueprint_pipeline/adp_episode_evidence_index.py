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
import os
import tempfile
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
ABSTENTION_SCHEMA_VERSION = "adp_task_evaluation_run_abstention.v1"
SUPPORTING_INVENTORY_SCHEMA_VERSION = "adp_supporting_evidence_inventory.v1"
ALLOWED_RECEIPT_SCHEMAS = {
    "adp009d_control_episode.v2",
    "adp_task_control_episode.v1",
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
    unresolved = root / relative_path
    if unresolved.is_symlink():
        raise EpisodeEvidenceIndexError(f"episode_artifact_symlink_forbidden:{role}")
    candidate = unresolved.resolve()
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


def _prefixed_sha256(path: Path) -> str:
    return "sha256:" + _sha256(path)


def _atomic_write_text(path: Path, content: str) -> None:
    """Replace one owned artifact without exposing partially written bytes."""

    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


def materialize_supporting_evidence_inventory(
    *,
    source_root: str | Path,
    output_root: str | Path,
    output_relative_path: str,
    source_root_id: str,
    artifacts: Sequence[Mapping[str, Any]],
    disclosure_class: str,
) -> dict[str, Any]:
    """Verify external construction bytes and emit a portable receipt inventory.

    Dataset-derived bytes can remain in their rights-bounded evidence root while
    the portable package carries exact, locally verified path/size/digest records.
    This deliberately does not turn a digest binding into publication authority.
    """

    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not source.is_dir():
        raise EpisodeEvidenceIndexError("supporting_evidence_source_root_missing")
    if not output.is_dir():
        raise EpisodeEvidenceIndexError("supporting_evidence_output_root_missing")
    if not source_root_id or not disclosure_class or not artifacts:
        raise EpisodeEvidenceIndexError("supporting_evidence_inventory_invalid")

    rows: list[dict[str, Any]] = []
    for index, artifact in enumerate(artifacts):
        role = str(artifact.get("role") or "")
        relative_path = str(artifact.get("relative_path") or "")
        if not role or not relative_path:
            raise EpisodeEvidenceIndexError(
                f"supporting_evidence_artifact_invalid:{index}"
            )
        path = _inside(source, relative_path, role=role)
        if path.is_symlink() or not path.is_file():
            raise EpisodeEvidenceIndexError(
                f"supporting_evidence_artifact_missing:{role}"
            )
        expected = str(artifact.get("sha256") or "")
        expected = expected.removeprefix("sha256:")
        if len(expected) != 64 or _sha256(path) != expected:
            raise EpisodeEvidenceIndexError(
                f"supporting_evidence_artifact_digest_mismatch:{role}"
            )
        if int(artifact.get("size_bytes", -1)) != path.stat().st_size:
            raise EpisodeEvidenceIndexError(
                f"supporting_evidence_artifact_size_mismatch:{role}"
            )
        rows.append(
            {
                "role": role,
                "source_root_id": source_root_id,
                "relative_path": relative_path,
                "sha256": "sha256:" + expected,
                "size_bytes": path.stat().st_size,
                "portable_link_available": False,
                "disclosure_class": disclosure_class,
            }
        )
    if len({(row["role"], row["relative_path"]) for row in rows}) != len(rows):
        raise EpisodeEvidenceIndexError("supporting_evidence_artifact_duplicate")

    receipt: dict[str, Any] = {
        "schema_version": SUPPORTING_INVENTORY_SCHEMA_VERSION,
        "status": "digest_verified_external_evidence_inventory",
        "source_root_id": source_root_id,
        "disclosure_class": disclosure_class,
        "artifacts": sorted(rows, key=lambda row: (row["role"], row["relative_path"])),
        "artifact_count": len(rows),
        "source_root_absolute_path_recorded": False,
        "artifact_bytes_embedded": False,
        "publication_authority_inferred": False,
        "inventory_digest": "",
    }
    receipt["inventory_digest"] = canonical_digest(
        receipt, digest_field="inventory_digest"
    )
    destination = _inside(output, output_relative_path, role="supporting_inventory")
    if destination.exists() or destination.is_symlink():
        raise EpisodeEvidenceIndexError("supporting_evidence_inventory_overwrite_forbidden")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


_SUPPORTING_RECEIPT_DIGEST_FIELDS = {
    "adp_gaussian_excision_attempt_receipt.v1": "receipt_digest",
    "adp_gaussian_excision_recovery_readiness.v1": "receipt_digest",
    SUPPORTING_INVENTORY_SCHEMA_VERSION: "inventory_digest",
}


def _supporting_receipt_row(root: Path, raw_path: str | Path) -> dict[str, Any]:
    path = Path(raw_path).expanduser()
    unresolved = path if path.is_absolute() else root / path
    if unresolved.is_symlink():
        raise EpisodeEvidenceIndexError("supporting_receipt_symlink_forbidden")
    path = unresolved.resolve()
    try:
        relative_path = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise EpisodeEvidenceIndexError(
            "supporting_receipt_path_outside_root"
        ) from exc
    if path.is_symlink() or not path.is_file():
        raise EpisodeEvidenceIndexError("supporting_receipt_missing")
    receipt = _load_json(path)
    schema_version = str(receipt.get("schema_version") or "")
    digest_field = _SUPPORTING_RECEIPT_DIGEST_FIELDS.get(schema_version)
    if digest_field is None:
        raise EpisodeEvidenceIndexError(
            f"supporting_receipt_schema_not_admitted:{schema_version or 'missing'}"
        )
    digest = str(receipt.get(digest_field) or "")
    if digest != canonical_digest(receipt, digest_field=digest_field):
        raise EpisodeEvidenceIndexError("supporting_receipt_digest_mismatch")
    return {
        "schema_version": schema_version,
        "relative_path": relative_path,
        "sha256": _prefixed_sha256(path),
        "size_bytes": path.stat().st_size,
        "receipt_digest": digest,
        "portable_link_available": True,
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
    abstention = payload.get("typed_abstention")
    abstention_html = (
        "<h2>Typed abstention</h2>"
        f"<p>No control or learned-policy episode exists. Smallest missing "
        f"capability: <code>{html.escape(str(abstention['smallest_missing_capability']))}</code>. "
        "This is an evidence gap, not a policy result.</p>"
        if isinstance(abstention, Mapping)
        else ""
    )
    supporting_rows = [
        "<tr>"
        f"<td>{html.escape(str(row['schema_version']))}</td>"
        f'<td><a href="{quote(str(row["relative_path"]))}">'
        f"{html.escape(str(row['relative_path']))}</a></td>"
        f"<td>{html.escape(str(row['size_bytes']))}</td>"
        f"<td><code>{html.escape(str(row['sha256']))}</code></td>"
        "</tr>"
        for row in payload.get("supporting_evidence", [])
    ]
    supporting_html = (
        "<h2>Supporting construction evidence</h2>"
        "<p>These portable receipts were digest-verified when this index was built. "
        "External dataset-derived bytes remain in their rights-bounded evidence root.</p>"
        "<table><thead><tr><th>Schema</th><th>Receipt</th><th>Bytes</th>"
        "<th>SHA-256</th></tr></thead><tbody>"
        + "".join(supporting_rows)
        + "</tbody></table>"
        if supporting_rows
        else ""
    )
    return "".join(
        [
            "<!doctype html>\n<html><head><meta charset=\"utf-8\">",
            "<title>ADP episode evidence index</title>",
            "<style>body{font-family:-apple-system,sans-serif;margin:2rem}",
            "table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:.5rem}",
            "th{background:#f4f4f4;text-align:left}</style></head><body>",
            "<h1>ADP episode evidence index</h1>",
            f"<p>Scene: {html.escape(str(identity['scene_id']))} &middot; ",
            f"Task: {html.escape(str(identity['task_id']))}</p>",
            "<p>Videos are derived review media. Scores come only from deterministic ",
            "simulator state. Overview is review-only and was not a policy input.</p>",
            abstention_html,
            supporting_html,
            "<table><thead><tr><th>Episode</th><th>Kind</th><th>Subject</th>",
            "<th>Outcome</th><th>Success</th><th>Videos</th><th>Frames</th>",
            "<th>Receipt</th></tr></thead><tbody>",
            "".join(rows),
            "</tbody></table>",
            f"<p>Authoritative JSON digest: {html.escape(payload['index_digest'])}</p>",
            f'<p><a href="{INDEX_FILENAME}">Open authoritative JSON index</a></p>',
            "</body></html>\n",
        ]
    )


def materialize_episode_evidence_index(
    *,
    run_root: str | Path,
    episode_receipt_paths: Sequence[str | Path],
    run_identity: Mapping[str, Any],
    abstention_receipt: Mapping[str, Any] | None = None,
    supporting_receipt_paths: Sequence[str | Path] = (),
    replace_existing: bool = False,
) -> dict[str, Any]:
    """Verify episode evidence and emit portable JSON plus HTML navigation."""

    root = Path(run_root).expanduser().resolve()
    if not root.is_dir():
        raise EpisodeEvidenceIndexError("episode_evidence_run_root_missing")
    identity = json.loads(json.dumps(dict(run_identity), allow_nan=False))
    for required in ("scene_id", "task_id", "scenario_suite_digest"):
        if not str(identity.get(required) or ""):
            raise EpisodeEvidenceIndexError(f"episode_index_identity_missing:{required}")
    if not episode_receipt_paths and abstention_receipt is None:
        raise EpisodeEvidenceIndexError("episode_index_receipts_missing")

    abstention = None
    if abstention_receipt is not None:
        try:
            abstention = json.loads(
                json.dumps(dict(abstention_receipt), allow_nan=False)
            )
        except (TypeError, ValueError) as exc:
            raise EpisodeEvidenceIndexError(
                "episode_index_abstention_invalid"
            ) from exc
        if (
            abstention.get("schema_version") != ABSTENTION_SCHEMA_VERSION
            or abstention.get("status") != "typed_evidence_backed_abstention"
            or abstention.get("receipt_digest")
            != canonical_digest(abstention, digest_field="receipt_digest")
            or not str(abstention.get("smallest_missing_capability") or "")
            or abstention.get("controls_executed") is not False
            or abstention.get("learned_candidate_episodes_executed") is not False
            or abstention.get("candidate_ids")
            != ["pi05_droid", "groot_n17_droid"]
        ):
            raise EpisodeEvidenceIndexError("episode_index_abstention_invalid")
        if episode_receipt_paths:
            raise EpisodeEvidenceIndexError(
                "episode_index_abstention_with_episode_receipts_forbidden"
            )

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

    supporting = [
        _supporting_receipt_row(root, path) for path in supporting_receipt_paths
    ]
    if len({row["relative_path"] for row in supporting}) != len(supporting):
        raise EpisodeEvidenceIndexError("supporting_receipt_duplicate")

    payload: dict[str, Any] = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "run_identity": identity,
        "episodes": sorted(rows, key=lambda row: row["episode_id"]),
        "episode_count": len(rows),
        "required_camera_ids": list(REQUIRED_CAMERA_IDS),
        "overview_is_review_only": True,
        "scores_are_deterministic_simulator_state": True,
        "review_videos_are_not_physical_truth": True,
        "typed_abstention": abstention,
        "supporting_evidence": sorted(
            supporting, key=lambda row: row["relative_path"]
        ),
        "index_digest": "",
    }
    payload["index_digest"] = canonical_digest(payload, digest_field="index_digest")
    json_path = root / INDEX_FILENAME
    html_path = root / HTML_FILENAME
    paths_exist = json_path.exists() or html_path.exists()
    paths_are_symlinks = json_path.is_symlink() or html_path.is_symlink()
    if paths_are_symlinks:
        raise EpisodeEvidenceIndexError("episode_evidence_index_symlink_forbidden")
    if paths_exist and not replace_existing:
        raise EpisodeEvidenceIndexError("episode_evidence_index_overwrite_forbidden")
    if replace_existing:
        if not json_path.is_file() or not html_path.is_file():
            raise EpisodeEvidenceIndexError(
                "episode_evidence_index_refresh_source_incomplete"
            )
        previous = _load_json(json_path)
        if (
            previous.get("schema_version") != INDEX_SCHEMA_VERSION
            or previous.get("index_digest")
            != canonical_digest(previous, digest_field="index_digest")
        ):
            raise EpisodeEvidenceIndexError(
                "episode_evidence_index_refresh_source_invalid"
            )
    json_content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    html_content = _render_html(payload)
    if replace_existing:
        _atomic_write_text(json_path, json_content)
        _atomic_write_text(html_path, html_content)
    else:
        json_path.write_text(json_content, encoding="utf-8")
        html_path.write_text(html_content, encoding="utf-8")
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
    "ABSTENTION_SCHEMA_VERSION",
    "HTML_FILENAME",
    "INDEX_FILENAME",
    "INDEX_SCHEMA_VERSION",
    "materialize_episode_evidence_index",
    "materialize_supporting_evidence_inventory",
]
