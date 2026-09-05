"""Build the server-owned profile for automatic SAM-first scene preparation.

The materializer is deliberately no-spend.  It reopens existing provider,
rights, cost, released-code, and dependency evidence and records secret *file
paths* without reading secret values.  Task-owner authority is not accepted by
this interface: paid child stages must reopen it from the immutable task
request.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .public_scene_sam31_track_selection_review import (
    AI_REVIEW_DECLARED_USE,
    AI_REVIEW_MAX_COST_USD,
    AI_REVIEW_MODEL,
    AI_RIGHTS_SCHEMA_VERSION,
)
from .scene_placement.sam31_source_track_provider import _validate_profile
from .task_evaluation_scene_configuration_sam31_plan import PROFILE_SCHEMA
from .task_evaluation_scene_configuration_submission_inputs import sha
from .task_evaluation_supervisor.openai_cost_authority import (
    validate_openai_cost_scope_attestation,
)


SAM31_SOURCE_PROFILE = "render_derived_synthetic_method_inputs"
SAM31_MAX_SPEND_USD = 1.0
SAM31_HARD_TTL_SECONDS = 1_800
CONTRIBUTION_MAX_SPEND_USD = 1.5
CONTRIBUTION_HARD_TTL_SECONDS = 3_600
MAX_HOURLY_RATE_USD = 0.5
AGGREGATE_PREPARATION_SPEND_CAP_USD = 2.5
OPENAI_REVIEW_RESOURCE_CLASS = "sam31_ai_visual_review"

_COMMIT = re.compile(r"[0-9a-f]{40}")


class Sam31PreparationProfileError(ValueError):
    """The operator inputs cannot form one production preparation profile."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise Sam31PreparationProfileError("sam31_preparation_profile_" + code)


def _safe_path(path: str | Path, *, kind: str, code: str) -> Path:
    source = Path(path).expanduser()
    _require(source.is_absolute(), code)
    _require(not any(item.is_symlink() for item in (source, *source.parents)), code)
    resolved = source.resolve()
    if kind == "file":
        _require(resolved.is_file(), code)
    elif kind == "directory":
        _require(resolved.is_dir(), code)
    else:  # pragma: no cover - private call contract
        raise AssertionError(kind)
    return resolved


def _read_json(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Sam31PreparationProfileError(
            "sam31_preparation_profile_" + code
        ) from exc
    _require(isinstance(value, dict), code)
    return value


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha(path),
        "size_bytes": path.stat().st_size,
    }


def _beneath_any(path: Path, roots: tuple[Path, ...], *, code: str) -> None:
    _require(any(path.is_relative_to(root) for root in roots), code)


def _secret_path(path: str | Path, *, group_read_allowed: bool, code: str) -> Path:
    source = _safe_path(path, kind="file", code=code)
    mode = stat.S_IMODE(source.stat().st_mode)
    forbidden = 0o027 if group_read_allowed else 0o077
    _require(mode & forbidden == 0, code)
    return source


def _git(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(  # nosec B603 - fixed executable and git arguments
            ["/usr/bin/git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Sam31PreparationProfileError(
            "sam31_preparation_profile_source_checkout_unverified"
        ) from exc
    _require(result.returncode == 0, "source_checkout_unverified")
    return result.stdout.strip()


def _validate_flashsplat(path: Path) -> dict[str, Any]:
    # Reuse the bundle builder's exact released-code/submodule validator.  The
    # paid stage repeats this check when it packages the provider runtime.
    from .adp_gaussian_excision_vast import _source_identity

    try:
        return _source_identity(path)
    except ValueError as exc:
        raise Sam31PreparationProfileError(
            "sam31_preparation_profile_flashsplat_identity_invalid"
        ) from exc


def _validate_dependency_wheelhouse(
    wheelhouse: Path, manifest_path: Path
) -> dict[str, Any]:
    # The canonical validator rehashes every pinned wheel.  Keeping the same
    # implementation here prevents a profile from postponing an offline
    # dependency failure until provider allocation.
    from .adp_gaussian_excision_vast import _verified_dependency_wheelhouse

    try:
        return _verified_dependency_wheelhouse(
            wheelhouse_path=wheelhouse,
            manifest_path=manifest_path,
        )
    except ValueError as exc:
        raise Sam31PreparationProfileError(
            "sam31_preparation_profile_dependency_wheelhouse_invalid"
        ) from exc


def _validate_provider_profile(path: Path, *, source_commit: str) -> dict[str, Any]:
    value = _read_json(path, code="sam31_provider_profile_invalid")
    blockers: list[str] = []
    _validate_profile(
        {
            "provider_profile": value,
            "allowed_evidence_uses": ["semantic_analysis"],
        },
        blockers,
    )
    _require(not blockers, "sam31_provider_profile_invalid")
    from .sam31_provider_launch_packet import validate_sam31_provider_profile_sources

    try:
        validate_sam31_provider_profile_sources(value, source_commit_sha=source_commit)
    except ValueError as exc:
        raise Sam31PreparationProfileError("sam31_preparation_profile_" + str(exc)) from exc
    return value


def _validate_review_rights(path: Path) -> dict[str, Any]:
    value = _read_json(path, code="review_rights_attestation_invalid")
    from .task_evaluation_sam31_preparation_review_authority import (
        SCHEMA as STANDING_REVIEW_SCHEMA, validate_sam31_review_authority,
    )
    if value.get("schema_version") == STANDING_REVIEW_SCHEMA:
        try:
            return validate_sam31_review_authority(path)
        except ValueError as exc:
            raise Sam31PreparationProfileError(
                "sam31_preparation_profile_review_rights_attestation_invalid"
            ) from exc
    _require(
        value.get("schema_version") == AI_RIGHTS_SCHEMA_VERSION
        and value.get("attestation_digest")
        == canonical_digest(value, digest_field="attestation_digest")
        and value.get("status") == "accepted_for_private_derived_visual_review"
        and value.get("declared_use_scope") == AI_REVIEW_DECLARED_USE
        and value.get("provider_id") == "openai"
        and value.get("runtime") == "openai_agents_sdk"
        and value.get("model") == AI_REVIEW_MODEL
        and value.get("max_inference_spend_usd") == AI_REVIEW_MAX_COST_USD
        and value.get("derived_overlay_pngs_only") is True
        and value.get("raw_source_splat_or_dataset_bytes_included") is False
        and value.get("frame_publication_authorized") is False
        and value.get("frame_redistribution_authorized") is False
        and value.get("issued_by_agent") is False
        and value.get("agent_accepted_terms") is False,
        "review_rights_attestation_invalid",
    )
    return value


def _gaussian_excision_policy() -> dict[str, Any]:
    """The existing released-code contribution/excision policy."""

    return {
        "minimum_per_view_contribution": 1.0 / 255.0,
        "owned_min_core_fraction": 0.98,
        "retained_max_core_fraction": 0.2,
        "minimum_core_camera_count": 2,
        "maximum_protected_camera_count_for_owned": 0,
        "minimum_geometry_score_owned": 0.5,
        "geometry_sigma_extent": 3.0,
        "geometry_margin_m": 0.02,
        "neighbor_count": 12,
        "neighbor_iterations": 4,
        "neighbor_radius_m": 0.12,
        "neighbor_blend": 0.35,
        "graph_owned_min_score": 0.95,
        "graph_retained_max_score": 0.2,
        "heldout_significant_alpha_threshold": 1.0 / 255.0,
        "heldout_rasterization_band_pixels": 2,
        "heldout_maximum_residual_connected_component_pixels": 4,
        "heldout_maximum_protected_significant_pixels": 0,
        "deterministic_repetitions": 2,
        "contribution_quantization_decimals": 6,
    }


def materialize_sam31_preparation_profile(
    *,
    source_commit: str,
    repo_root: str | Path,
    server_data_root: str | Path,
    runtime_root: str | Path,
    sam31_provider_profile_path: str | Path,
    sam31_review_rights_attestation_path: str | Path,
    sam31_review_cost_scope_attestation_path: str | Path,
    sam31_hf_token_file: str | Path,
    openai_admin_api_key_file: str | Path,
    openai_project_id: str,
    openai_api_key_id: str,
    flashsplat_root: str | Path,
    dependency_wheelhouse_path: str | Path,
    dependency_manifest_path: str | Path,
    approved_roots: Sequence[str | Path],
    ffmpeg_executable: str | Path = "/usr/bin/ffmpeg",
    calibrated_views_execution_site: str = "control_plane",
    calibrated_views_machine_avoidlist_path: str | Path | None = None,
) -> dict[str, Any]:
    """Derive the exact operator profile without allocating or authorizing work."""

    _require(calibrated_views_execution_site in {"control_plane", "provider_gpu"}, "calibrated_views_execution_site_invalid")
    _require(_COMMIT.fullmatch(str(source_commit)) is not None, "source_commit_invalid")
    _require(bool(str(openai_project_id).strip()), "openai_project_id_invalid")
    _require(bool(str(openai_api_key_id).strip()), "openai_api_key_id_invalid")
    roots = tuple(
        _safe_path(path, kind="directory", code="approved_root_invalid")
        for path in approved_roots
    )
    _require(bool(roots) and len(set(roots)) == len(roots), "approved_roots_invalid")
    avoidlist = None
    if calibrated_views_execution_site == "provider_gpu":
        _require(calibrated_views_machine_avoidlist_path is not None, "calibrated_views_machine_avoidlist_required")
        avoidlist = _safe_path(calibrated_views_machine_avoidlist_path, kind="file", code="calibrated_views_machine_avoidlist_invalid")
        _beneath_any(avoidlist, roots, code="calibrated_views_machine_avoidlist_outside_approved_roots")

    repo = _safe_path(repo_root, kind="directory", code="repo_root_invalid")
    data = _safe_path(server_data_root, kind="directory", code="server_data_root_invalid")
    runtime = _safe_path(runtime_root, kind="directory", code="runtime_root_invalid")
    provider_profile_path = _safe_path(
        sam31_provider_profile_path,
        kind="file",
        code="sam31_provider_profile_invalid",
    )
    rights_path = _safe_path(
        sam31_review_rights_attestation_path,
        kind="file",
        code="review_rights_attestation_invalid",
    )
    cost_path = _safe_path(
        sam31_review_cost_scope_attestation_path,
        kind="file",
        code="review_cost_scope_attestation_invalid",
    )
    flashsplat = _safe_path(
        flashsplat_root, kind="directory", code="flashsplat_root_invalid"
    )
    wheelhouse = _safe_path(
        dependency_wheelhouse_path,
        kind="directory",
        code="dependency_wheelhouse_invalid",
    )
    dependency_manifest = _safe_path(
        dependency_manifest_path,
        kind="file",
        code="dependency_manifest_invalid",
    )
    ffmpeg = _safe_path(
        ffmpeg_executable, kind="file", code="ffmpeg_executable_invalid"
    )
    for path in (
        repo,
        data,
        runtime,
        provider_profile_path,
        rights_path,
        cost_path,
        flashsplat,
        wheelhouse,
        dependency_manifest,
        ffmpeg,
    ):
        _beneath_any(path, roots, code="input_outside_approved_roots")
    _require(runtime.is_relative_to(data), "runtime_root_outside_server_data_root")

    hf_token = _secret_path(
        sam31_hf_token_file,
        group_read_allowed=True,
        code="sam31_hf_token_file_invalid",
    )
    openai_admin_key = _secret_path(
        openai_admin_api_key_file,
        group_read_allowed=False,
        code="openai_admin_api_key_file_invalid",
    )

    _require(_git(repo, "rev-parse", "HEAD") == source_commit, "source_commit_mismatch")
    _require(not _git(repo, "status", "--short"), "source_checkout_dirty")
    provider_profile = _validate_provider_profile(provider_profile_path, source_commit=source_commit)
    rights = _validate_review_rights(rights_path)
    if "authority_digest" in rights:
        _require(rights.get("source_commit") == source_commit, "review_authority_commit_mismatch")
    cost_scope = _read_json(cost_path, code="review_cost_scope_attestation_invalid")
    try:
        validate_openai_cost_scope_attestation(
            cost_scope,
            provider_id="openai",
            paid_resource_class=OPENAI_REVIEW_RESOURCE_CLASS,
            project_id=str(openai_project_id),
            api_key_id=str(openai_api_key_id),
        )
    except ValueError as exc:
        raise Sam31PreparationProfileError(
            "sam31_preparation_profile_review_cost_scope_attestation_invalid"
        ) from exc
    flashsplat_identity = _validate_flashsplat(flashsplat)
    dependency_identity = _validate_dependency_wheelhouse(wheelhouse, dependency_manifest)

    profile: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA,
        "source_commit": source_commit,
        "repo_root": str(repo),
        "server_data_root": str(data),
        "runtime_root": str(runtime),
        "ffmpeg_executable": str(ffmpeg),
        "review_model": AI_REVIEW_MODEL,
        "review_maximum_cost_usd": AI_REVIEW_MAX_COST_USD,
        "candidate_policy_queried": False,
        "calibrated_views": {"execution_site": calibrated_views_execution_site,
                             "hardware_required": calibrated_views_execution_site == "provider_gpu",
                             "max_spend_usd": 1.0, "hard_ttl_seconds": 1800,
                             "max_hourly_rate_usd": 0.5, "retry_cap": 0,
                             "maximum_resource_count": 1, "allowed_geolocation_country_codes": ["US"],
                             "machine_avoidlist": _file_record(avoidlist) if avoidlist else None},
        "artifact_references": {
            "sam31_provider_profile": _file_record(provider_profile_path),
        },
        "sam31_provider_identity": {
            "method_id": provider_profile["method_id"],
            "method_version": provider_profile["method_version"],
            "official_code_revision": provider_profile["official_code_revision"],
            "checkpoint_family": provider_profile["checkpoint_family"],
            "checkpoint_digest": provider_profile["checkpoint_digest"],
            "profile_digest": provider_profile["profile_digest"],
        },
        "sam31_visual_review": {
            "model": AI_REVIEW_MODEL,
            "maximum_cost_usd": AI_REVIEW_MAX_COST_USD,
            "rights_attestation": _file_record(rights_path),
            "rights_attestation_digest": rights.get("authority_digest", rights.get("attestation_digest")),
            "openai_cost_scope_attestation": _file_record(cost_path),
            "openai_cost_scope_attestation_digest": cost_scope[
                "scope_attestation_digest"
            ],
            "openai_admin_api_key_file": str(openai_admin_key),
            "openai_project_id": str(openai_project_id),
            "openai_api_key_id": str(openai_api_key_id),
        },
        "gaussian_excision_policy": _gaussian_excision_policy(),
        "approved_paid_input_roots": [str(path) for path in roots],
        "released_dependencies": {
            "flashsplat_root": str(flashsplat),
            "flashsplat_identity": flashsplat_identity,
            "dependency_wheelhouse_path": str(wheelhouse),
            "dependency_manifest": _file_record(dependency_manifest),
            "dependency_manifest_digest": dependency_identity["manifest_digest"],
        },
        "paid_stages": {
            "sam31_tracking": {
                "source_profile": SAM31_SOURCE_PROFILE,
                "hf_token_file": str(hf_token),
                "max_spend_usd": SAM31_MAX_SPEND_USD,
                "max_hourly_rate_usd": MAX_HOURLY_RATE_USD,
                "hard_ttl_seconds": SAM31_HARD_TTL_SECONDS,
                "retry_cap": 0,
                "allowed_active_instance_ids": [],
                "aggregate_goal_spend_before_attempt_usd": 0.0,
                "aggregate_goal_spend_cap_usd": AGGREGATE_PREPARATION_SPEND_CAP_USD,
            },
            "contribution_sweep": {
                "max_spend_usd": CONTRIBUTION_MAX_SPEND_USD,
                "max_hourly_rate_usd": MAX_HOURLY_RATE_USD,
                "hard_ttl_seconds": CONTRIBUTION_HARD_TTL_SECONDS,
                "retry_cap": 0,
                "allowed_active_instance_ids": [],
                "flashsplat_root": str(flashsplat),
                "dependency_wheelhouse_path": str(wheelhouse),
                "dependency_manifest_path": str(dependency_manifest),
            },
        },
        "authority_boundary": {
            "task_owner_authority_source": "task_request.human_authority",
            "profile_contains_task_owner_authority": False,
            "profile_grants_paid_execution_authority": False,
            "paid_stage_must_reopen_task_request": True,
            "review_rights_are_not_task_execution_authority": True,
        },
        "claim_boundary": {
            "source_profile": SAM31_SOURCE_PROFILE,
            "raw_source_upload_allowed": False,
            "evaluation_authorized": False,
            "metric_claim_upgrade_allowed": False,
            "physical_claim_upgrade_allowed": False,
            "candidate_policy_queried": False,
        },
        "paid_execution_started": False,
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
        "profile_digest": "",
    }
    profile["profile_digest"] = canonical_digest(
        profile, digest_field="profile_digest"
    )
    return profile


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _require(path.is_absolute() and not path.exists() and not path.is_symlink(), "output_invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(dict(value)) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--server-data-root", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--sam31-provider-profile", required=True)
    parser.add_argument("--sam31-review-rights-attestation", required=True)
    parser.add_argument("--sam31-review-cost-scope-attestation", required=True)
    parser.add_argument("--sam31-hf-token-file", required=True)
    parser.add_argument("--openai-admin-api-key-file", required=True)
    parser.add_argument("--openai-project-id", required=True)
    parser.add_argument("--openai-api-key-id", required=True)
    parser.add_argument("--flashsplat-root", required=True)
    parser.add_argument("--dependency-wheelhouse", required=True)
    parser.add_argument("--dependency-manifest", required=True)
    parser.add_argument("--approved-root", action="append", required=True)
    parser.add_argument("--ffmpeg-executable", default="/usr/bin/ffmpeg")
    parser.add_argument("--calibrated-views-execution-site", choices=("control_plane", "provider_gpu"), default="control_plane")
    parser.add_argument("--calibrated-views-machine-avoidlist")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        profile = materialize_sam31_preparation_profile(
            source_commit=args.source_commit,
            repo_root=args.repo_root,
            server_data_root=args.server_data_root,
            runtime_root=args.runtime_root,
            sam31_provider_profile_path=args.sam31_provider_profile,
            sam31_review_rights_attestation_path=args.sam31_review_rights_attestation,
            sam31_review_cost_scope_attestation_path=(
                args.sam31_review_cost_scope_attestation
            ),
            sam31_hf_token_file=args.sam31_hf_token_file,
            openai_admin_api_key_file=args.openai_admin_api_key_file,
            openai_project_id=args.openai_project_id,
            openai_api_key_id=args.openai_api_key_id,
            flashsplat_root=args.flashsplat_root,
            dependency_wheelhouse_path=args.dependency_wheelhouse,
            dependency_manifest_path=args.dependency_manifest,
            approved_roots=args.approved_root,
            ffmpeg_executable=args.ffmpeg_executable,
            calibrated_views_execution_site=args.calibrated_views_execution_site,
            calibrated_views_machine_avoidlist_path=args.calibrated_views_machine_avoidlist,
        )
        output = Path(args.output).expanduser().resolve()
        _write_exclusive(output, profile)
    except (OSError, Sam31PreparationProfileError) as exc:
        print(
            canonical_json(
                {
                    "schema_version": PROFILE_SCHEMA,
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "paid_execution_started": False,
                    "provider_mutations_performed": 0,
                }
            )
        )
        return 2
    print(
        canonical_json(
            {
                "schema_version": PROFILE_SCHEMA,
                "status": "materialized_no_spend",
                "profile_digest": profile["profile_digest"],
                "output": str(output),
                "paid_execution_started": False,
                "provider_mutations_performed": 0,
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PROFILE_SCHEMA",
    "SAM31_SOURCE_PROFILE",
    "Sam31PreparationProfileError",
    "main",
    "materialize_sam31_preparation_profile",
]
