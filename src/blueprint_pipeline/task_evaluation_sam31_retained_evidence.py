"""Read-only validation of retained paid-phase evidence; no provider imports."""
from pathlib import Path
from typing import Any, Mapping
from .task_evaluation_scene_configuration_submission_inputs import beneath, checked_file, read, sha

STAGES = {"sam31_tracking", "contribution_sweep"}

class Sam31PreparationPaidStageError(ValueError):
    """A paid precursor could not be derived or closed safely."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise Sam31PreparationPaidStageError("sam31_preparation_paid_" + code)

def _terminal_manifest_valid(
    path: Path, *, lane: str, required_paths: tuple[Path, ...], binding: Mapping[str, Any] | None = None
) -> bool:
    """Rehash retained files using the canonical allocator's actual inventory format."""
    try:
        manifest = read(path, digest_field="manifest_digest")
        rows = manifest.get("files")
        actual_binding = manifest.get("binding") or {}
        if (
            manifest.get("schema_version") != "task_evaluation_artifact_manifest.v1"
            or manifest.get("status") != "completed" or manifest.get("blockers") != []
            or actual_binding.get("allocator_lane") != lane
            or actual_binding.get("retry_cap") != 0
            or any(actual_binding.get(key) != value for key, value in (binding or {}).items())
            or not isinstance(rows, list) or not rows or manifest.get("file_count") != len(rows)
        ):
            return False
        seen: set[Path] = set()
        total = 0
        for row in rows:
            artifact = beneath(path.parent, row["relative_path"])
            size = row.get("size_bytes")
            if (artifact in seen or type(size) is not int or size < 0
                    or not artifact.is_file() or artifact.stat().st_size != size
                    or sha(artifact) != row.get("sha256")):
                return False
            seen.add(artifact)
            total += size
        return (manifest.get("total_size_bytes") == total
                and set(required_paths).issubset(seen))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def _teardown_valid(path: Path) -> bool:
    try:
        teardown = read(path)
        return (teardown.get("schema_version") == "vast_teardown_manifest.v1"
                and teardown.get("continuing_spend_from_this_run") is False)
    except (OSError, ValueError, KeyError, TypeError):
        return False

def validate_retained_paid_stage(outcome: Mapping[str, Any], *, stage_id: str) -> None:
    """Read-only validation of a completed phase's exact retained artifact set."""
    _require(stage_id in STAGES and outcome.get("stage_id") == stage_id
             and outcome.get("status") == "completed", "replay_stage_invalid")
    prefix, lane, names = (
        ("sam31", "semantic_sam31_source_tracks", ("source_tracks", "provider_zero"))
        if stage_id == "sam31_tracking"
        else ("gaussian", "adp_gaussian_excision", ("provider_execution_result", "contribution_evidence"))
    )
    required = {prefix + "_" + name for name in ("allocator_result", "teardown", "artifact_manifest", *names)}
    records = outcome.get("artifacts")
    _require(isinstance(records, Mapping) and set(records) == required,
             "replay_artifact_set_changed")
    paths = {}
    for name, row in records.items():
        _require(isinstance(row, Mapping), "replay_artifact_invalid")
        path = Path(str(row.get("path") or ""))
        _require(path.is_absolute(), "replay_artifact_invalid")
        paths[name] = checked_file(path, dict(row))
    manifest = paths[prefix + "_artifact_manifest"]
    teardown = paths[prefix + "_teardown"]
    retained = tuple(paths[prefix + "_" + name] for name in (*names, "teardown"))
    _require(_teardown_valid(teardown) and _terminal_manifest_valid(
        manifest, lane=lane, required_paths=retained,
    ), "replay_terminal_artifacts_changed")
    if "scene_owner_attempt" in outcome:
        from .task_evaluation_scene_execution_authority import require_scene_execution_authority
        reference = outcome["scene_owner_attempt"]
        owner = read(checked_file(reference["path"], reference))
        require_scene_execution_authority(owner,
            source_commit=owner["scene_attempt_binding"]["source_commit"], reopen_records=False)

