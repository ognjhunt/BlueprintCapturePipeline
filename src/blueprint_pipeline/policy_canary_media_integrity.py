"""Validate frozen episode media before worker completion is sealed."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Callable, Mapping

def _sha256(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"policy_canary_input_not_object:{path.name}")
    return value

def bound_media_artifact(
    output_root: Path,
    *,
    media_root: Path,
    artifacts: Any,
    role: str,
    role_match: Callable[[str], bool],
) -> dict[str, Any] | None:
    """Bind one episode media artifact into run-root-relative evidence.

    The episode runner records media rows relative to its ``media_output_dir``
    (the run's ``episodes`` directory), not to the run root.  Resolving them
    against the run root silently returned ``None`` for every frame manifest
    and review video, so paid runs shipped episode evidence without either.
    The hermetic lifecycle rehearsal pins the corrected binding.
    """

    matches = [
        row
        for row in artifacts or []
        if isinstance(row, Mapping) and role_match(str(row.get("role") or ""))
    ]
    if not matches:
        return None
    row = matches[0]
    original_path = media_root / str(row.get("relative_path") or "")
    if original_path.is_symlink():
        return None
    path = original_path.resolve()
    try:
        path.relative_to(output_root)
    except ValueError:
        return None
    if not path.is_file():
        return None
    try:
        observed_size = path.stat().st_size
        observed_digest = _sha256(path)
    except OSError:
        return None
    if (type(row.get("size_bytes")) is not int or row["size_bytes"] != observed_size
            or row.get("sha256") != observed_digest):
        return None
    return {
        "role": role,
        "relative_path": path.relative_to(output_root).as_posix(),
        "size_bytes": observed_size,
        "sha256": observed_digest,
    }


def require_completed_episode_media(output_root: Path, episode: Mapping[str, Any]) -> None:
    """Verify the producer's frozen bytes before the worker seals completion."""
    artifacts = episode.get("media_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("policy_canary_episode_media_missing")
    roles = [str(row.get("role") or "") for row in artifacts if isinstance(row, Mapping)]
    if not any("frame_manifest" in role for role in roles) or not any("video" in role for role in roles):
        raise ValueError("policy_canary_episode_media_missing")
    for row in artifacts:
        if not isinstance(row, Mapping) or bound_media_artifact(
            output_root, media_root=output_root / "episodes", artifacts=[row],
            role=str(row.get("role") or ""), role_match=lambda _name: True,
        ) is None:
            raise ValueError("policy_canary_episode_media_identity_invalid")
    # A valid inventory hash cannot hide a missing frame by omitting its row.
    # Validate the producer's manifest and every referenced observation as well.
    from .episode_visual_evidence import validate_multicamera_frame_manifest
    manifests = [row for row in artifacts if row.get("role") == "multicamera_observation_frame_manifest"]
    if len(manifests) != 1:
        raise ValueError("policy_canary_episode_multicamera_manifest_missing_or_ambiguous")
    media_root = output_root / "episodes"
    manifest = _read(media_root / manifests[0]["relative_path"])
    if (set(manifest.get("required_camera_ids") or []) != {"external", "wrist", "overview"}
            or set(manifest.get("review_only_camera_ids") or []) != {"overview"}):
        raise ValueError("policy_canary_episode_camera_contract_invalid")
    validate_multicamera_frame_manifest(manifest, output_dir=media_root, verify_files=True)
