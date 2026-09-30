"""Existing portable bundle receipt readers without bundle production imports."""
from __future__ import annotations

import hashlib
import json
import re
import stat
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_diagnostic_mode import (
    CHECKPOINT_RESUME_DIAGNOSTIC_BOOTSTRAP_MODE,
    FRESH_DIAGNOSTIC_BOOTSTRAP_MODE,
)
from .task_evaluation_scene_configuration_disclosure import (
    PENDING_PROVIDER_RENDER_STATUS,
    renders_on_provider,
)
from .task_evaluation_splat_render_runtime import (
    PROVIDER_RENDERER_REQUIRED_PACKAGES,
    PROVIDER_RENDERER_SCHEMA_VERSION,
)

PROBE_KIND = "task-evaluation-scene-configuration"


PROVIDER_BUNDLE_KIND = "task_evaluation_scene_configuration"


BUNDLE_SCHEMA_VERSION = "task_evaluation_scene_configuration_provider_bundle.v1"


_COMMIT = re.compile(r"[0-9a-f]{40}")


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


_PROVIDER_RENDERER_FILES = (
    "tools/splat_render/render_splat.mjs",
    "tools/splat_render/src/render_entry.mjs",
    "tools/splat_render/harness.html",
    "tools/splat_render/package.json",
    "tools/splat_render/package-lock.json",
)


class TaskEvaluationSceneConfigurationBundleError(ValueError):
    """The Website-started construction could not become a portable job."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationBundleError(code) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise TaskEvaluationSceneConfigurationBundleError(code)
    return dict(value)


def _receipt_disclosure_is_coherent(receipt: Mapping[str, Any]) -> bool:
    """True when the receipt's raw-bytes claim matches its authorizing decision.

    The old literal ``raw_interiorgs_bytes_in_provider_bundle is False`` is
    preserved for every scene whose rights do not admit the upload. A receipt
    that claims the bytes crossed must additionally carry the digest-bound
    decision that permitted it, so this narrows what a bundle may claim
    rather than widening it.
    """

    crossed = receipt.get("raw_interiorgs_bytes_in_provider_bundle")
    decision = (receipt.get("provider_disclosure_receipt") or {}).get(
        "disclosure_decision"
    ) or receipt.get("disclosure_decision")
    if crossed is False:
        return True
    return crossed is True and renders_on_provider(decision or {})


def _provider_renderer_archive_is_valid(
    archive: zipfile.ZipFile, bundle_manifest: Mapping[str, Any]
) -> bool:
    prefix = "provider_runtime/renderer/"
    manifest_member = prefix + f"{PROVIDER_RENDERER_SCHEMA_VERSION}.json"
    members = {
        info.filename: info
        for info in archive.infolist()
        if info.filename.startswith(prefix) and not info.is_dir()
    }
    required = bundle_manifest.get("provider_renderer_required") is True
    if not required:
        return not members and not any(
            key in bundle_manifest
            for key in (
                "provider_renderer_required",
                "provider_renderer_digest",
                "provider_renderer_source_runtime_digest",
            )
        )
    try:
        renderer_value = json.loads(archive.read(manifest_member).decode("utf-8"))
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(renderer_value, Mapping):
        return False
    renderer = dict(renderer_value)
    rows = renderer.get("files")
    if (
        renderer.get("schema_version") != PROVIDER_RENDERER_SCHEMA_VERSION
        or renderer.get("status") != "ready_for_provider_render"
        or renderer.get("platform") != "linux-x86_64"
        or renderer.get("source_commit") != bundle_manifest.get("source_commit")
        or renderer.get("source_runtime_digest")
        != bundle_manifest.get("provider_renderer_source_runtime_digest")
        or renderer.get("renderer_digest")
        != bundle_manifest.get("provider_renderer_digest")
        or renderer.get("renderer_digest")
        != canonical_digest(renderer, digest_field="renderer_digest")
        or not isinstance(rows, list)
        or not rows
    ):
        return False
    expected: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        relative = str(row.get("relative_path") or "")
        if (
            not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in expected
            or not isinstance(row.get("size_bytes"), int)
            or isinstance(row.get("size_bytes"), bool)
            or row.get("size_bytes") < 0
            or not isinstance(row.get("executable"), bool)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(row.get("sha256") or ""))
            is None
        ):
            return False
        expected[relative] = row
    if set(members) != {
        manifest_member,
        *(prefix + relative for relative in expected),
    }:
        return False
    for relative, row in expected.items():
        member = prefix + relative
        info = members[member]
        body = archive.read(member)
        archived_mode = info.external_attr >> 16
        if (
            len(body) != row["size_bytes"]
            or "sha256:" + hashlib.sha256(body).hexdigest() != row["sha256"]
            or not stat.S_ISREG(archived_mode)
            or bool(archived_mode & 0o111) != row["executable"]
        ):
            return False
    entrypoints = renderer.get("entrypoints")
    node = str(
        entrypoints.get("node") if isinstance(entrypoints, Mapping) else ""
    )
    browser = str(
        entrypoints.get("browser") if isinstance(entrypoints, Mapping) else ""
    )
    if any(
        relative not in expected or expected[relative].get("executable") is not True
        for relative in (node, browser)
    ):
        return False
    if any(relative not in expected for relative in _PROVIDER_RENDERER_FILES):
        return False
    return all(
        any(
            relative.startswith(
                f"tools/splat_render/node_modules/{package}/"
            )
            for relative in expected
        )
        for package in PROVIDER_RENDERER_REQUIRED_PACKAGES
    )


def load_scene_configuration_provider_bundle_receipt(
    path: str | Path,
    *,
    expected_source_commit: str | None = None,
    diagnostic_only: bool = False,
) -> dict[str, Any]:
    """Reopen the exact portable bundle and its immutable internal manifest."""

    receipt_path = Path(path).expanduser().resolve()
    receipt = _read(
        receipt_path, code="scene_configuration_bundle_receipt_invalid"
    )
    production_semantic_reuse = (
        receipt.get("production_semantic_input_reuse") is True
    )
    bundle = Path(str(receipt.get("bundle_path") or "")).expanduser().resolve()
    errors: list[str] = []
    if (
        receipt.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or receipt.get("status") != "ready"
        or receipt.get("provider_bundle_kind") != PROVIDER_BUNDLE_KIND
        or receipt.get("probe_kind") != PROBE_KIND
        or not _receipt_disclosure_is_coherent(receipt)
        or receipt.get("single_parent_allocation") is not True
        or receipt.get("nested_provider_mutations_performed") != 0
        or receipt.get("evaluation_episode_executed") is not False
        or (
            diagnostic_only
            and (
                production_semantic_reuse
                or receipt.get("diagnostic_only") is not True
                or receipt.get("qualification_eligible") is not False
                or receipt.get("executed_inside_one_parent_provider_run") is not False
                or receipt.get("configured_revision_publication_permitted") is not False
                or receipt.get("offering_publication_permitted") is not False
                or receipt.get("terminal_e2e_completion_permitted") is not False
                or _COMMIT.fullmatch(
                    str(receipt.get("construction_source_commit") or "")
                )
                is None
                or _DIGEST.fullmatch(
                    str(
                        receipt.get("diagnostic_scientific_binding_digest")
                        or ""
                    )
                )
                is None
                or not isinstance(
                    receipt.get("diagnostic_stage_sequence_ids"), list
                )
                or len(receipt.get("diagnostic_stage_sequence_ids") or []) != 6
                or len(set(receipt.get("diagnostic_stage_sequence_ids") or []))
                != 6
                or any(
                    not isinstance(stage_id, str) or not stage_id
                    for stage_id in receipt.get("diagnostic_stage_sequence_ids")
                    or []
                )
                or (
                    receipt.get("diagnostic_bootstrap_mode")
                    == FRESH_DIAGNOSTIC_BOOTSTRAP_MODE
                    and (
                        receipt.get("source_diagnostic_checkpoint_digest")
                        is not None
                        or receipt.get("carried_completed_stage_count") != 0
                    )
                )
                or (
                    receipt.get("diagnostic_bootstrap_mode")
                    != FRESH_DIAGNOSTIC_BOOTSTRAP_MODE
                    and (
                        receipt.get("diagnostic_bootstrap_mode")
                        != CHECKPOINT_RESUME_DIAGNOSTIC_BOOTSTRAP_MODE
                        or _DIGEST.fullmatch(
                            str(
                                receipt.get(
                                    "source_diagnostic_checkpoint_digest"
                                )
                                or ""
                            )
                        )
                        is None
                    )
                )
            )
        )
        or (
            not diagnostic_only
            and production_semantic_reuse
            and (
                _COMMIT.fullmatch(
                    str(receipt.get("construction_source_commit") or "")
                )
                is None
                or _DIGEST.fullmatch(
                    str(receipt.get("source_semantic_checkpoint_digest") or "")
                )
                is None
                or _DIGEST.fullmatch(
                    str(receipt.get("source_construction_envelope_digest") or "")
                )
                is None
                or _DIGEST.fullmatch(
                    str(
                        receipt.get("configuration_revision_intake_receipt_digest")
                        or ""
                    )
                )
                is None
                or re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}",
                    str(receipt.get("configuration_revision_id") or ""),
                )
                is None
                or _DIGEST.fullmatch(
                    str(
                        receipt.get("semantic_reuse_scientific_binding_digest")
                        or ""
                    )
                )
                is None
                or receipt.get("semantic_reuse_completed_stage_prefix_count") != 0
                or receipt.get("provider_render_outputs_reused") is not True
                or receipt.get("semantic_teacher_outputs_reused") is not True
                or receipt.get("full_downstream_stage_chain_required") is not True
                or receipt.get("normal_production_runner_used") is not True
                or receipt.get("configured_revision_publication_permitted") is not True
                or receipt.get("offering_publication_permitted") is not True
            )
        )
        or (
            not diagnostic_only
            and not production_semantic_reuse
            and any(
                key in receipt
                for key in (
                    "diagnostic_only",
                    "qualification_eligible",
                    "configured_revision_publication_permitted",
                    "offering_publication_permitted",
                    "terminal_e2e_completion_permitted",
                    "diagnostic_bootstrap_mode",
                    "diagnostic_scientific_binding_digest",
                    "diagnostic_stage_sequence_ids",
                    "construction_source_commit",
                    "source_construction_envelope_digest",
                    "configuration_revision_intake_receipt_digest",
                    "configuration_revision_id",
                    "production_semantic_input_reuse",
                    "source_semantic_checkpoint_digest",
                    "semantic_reuse_scientific_binding_digest",
                    "semantic_reuse_completed_stage_prefix_count",
                    "provider_render_outputs_reused",
                    "semantic_teacher_outputs_reused",
                    "full_downstream_stage_chain_required",
                    "normal_production_runner_used",
                )
            )
        )
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
        or (
            expected_source_commit is not None
            and receipt.get("source_commit") != expected_source_commit
        )
    ):
        errors.append("receipt_contract_invalid")
    internal: dict[str, Any] = {}
    if (
        bundle.is_symlink()
        or not bundle.is_file()
        or bundle.stat().st_size != receipt.get("bundle_size_bytes")
        or _sha256(bundle) != receipt.get("bundle_sha256")
    ):
        errors.append("bundle_bytes_invalid")
    else:
        try:
            with zipfile.ZipFile(bundle) as archive:
                internal_value = json.loads(
                    archive.read(
                        f"provider_runtime/{BUNDLE_SCHEMA_VERSION}.json"
                    ).decode("utf-8")
                )
                provider_renderer_valid = _provider_renderer_archive_is_valid(
                    archive,
                    internal_value if isinstance(internal_value, Mapping) else {},
                )
                if diagnostic_only:
                    names = {
                        row.filename
                        for row in archive.infolist()
                        if not row.is_dir()
                    }
                    portable_value = json.loads(
                        archive.read(
                            "provider_runtime/input/portable_construction_envelope.v1.json"
                        ).decode("utf-8")
                    )
                    diagnostic_render = (
                        portable_value.get("render_inputs_result")
                        if isinstance(portable_value, Mapping)
                        else None
                    )
                    source_appearance = (
                        diagnostic_render.get("source_appearance")
                        if isinstance(diagnostic_render, Mapping)
                        else None
                    )
                    if (
                        receipt.get("diagnostic_bootstrap_mode")
                        == FRESH_DIAGNOSTIC_BOOTSTRAP_MODE
                    ):
                        diagnostic_archive_valid = (
                            isinstance(diagnostic_render, Mapping)
                            and diagnostic_render.get("status")
                            == PENDING_PROVIDER_RENDER_STATUS
                            and diagnostic_render.get("provider_render_required")
                            is True
                            and isinstance(source_appearance, Mapping)
                            and str(source_appearance.get("path") or "").startswith(
                                "input/render/"
                            )
                            and any(
                                name.startswith("provider_runtime/renderer/")
                                for name in names
                            )
                            and any(
                                name.startswith(
                                    "provider_runtime/input/render/source_appearance"
                                )
                                for name in names
                            )
                            and not any(
                                name.startswith(
                                    "provider_runtime/input/diagnostic_checkpoint/"
                                )
                                for name in names
                            )
                        )
                    else:
                        diagnostic_archive_valid = (
                            isinstance(diagnostic_render, Mapping)
                            and diagnostic_render.get(
                                "diagnostic_checkpoint_reused"
                            )
                            is True
                            and diagnostic_render.get("provider_render_skipped")
                            is True
                            and isinstance(source_appearance, Mapping)
                            and "path" not in source_appearance
                            and not any(
                                name.startswith("provider_runtime/renderer/")  # noqa: PIE810 - preserve existing compatibility/body semantics
                                or name.startswith(
                                    "provider_runtime/input/render/source_appearance"
                                )
                                for name in names
                            )
                            and any(
                                name.startswith(
                                    "provider_runtime/input/diagnostic_checkpoint/semantic/"
                                )
                                for name in names
                            )
                            and all(
                                str(row.get("path") or "").startswith(
                                    "input/diagnostic_checkpoint/"
                                )
                                for row in diagnostic_render.get("derived_frames")
                                or []
                                if isinstance(row, Mapping)
                            )
                        )
                    if not diagnostic_archive_valid:
                        errors.append("bundle_diagnostic_archive_invalid")
                elif production_semantic_reuse:
                    names = {
                        row.filename
                        for row in archive.infolist()
                        if not row.is_dir()
                    }
                    portable_value = json.loads(
                        archive.read(
                            "provider_runtime/input/portable_construction_envelope.v1.json"
                        ).decode("utf-8")
                    )
                    reuse_render = (
                        portable_value.get("render_inputs_result")
                        if isinstance(portable_value, Mapping)
                        else None
                    )
                    reuse_archive_valid = (
                        isinstance(reuse_render, Mapping)
                        and reuse_render.get("production_semantic_input_reuse") is True
                        and reuse_render.get("provider_render_skipped") is True
                        and not any(
                            name.startswith("provider_runtime/renderer/")  # noqa: PIE810 - preserve existing compatibility/body semantics
                            or name.startswith(
                                "provider_runtime/input/render/source_appearance"
                            )
                            for name in names
                        )
                        and any(
                            name.startswith(
                                "provider_runtime/input/production_semantic_reuse_checkpoint/semantic/"
                            )
                            for name in names
                        )
                        and all(
                            str(row.get("path") or "").startswith(
                                "input/production_semantic_reuse_checkpoint/"
                            )
                            for row in reuse_render.get("derived_frames") or []
                            if isinstance(row, Mapping)
                        )
                    )
                    if not reuse_archive_valid:
                        errors.append("bundle_semantic_reuse_archive_invalid")
            internal = (
                dict(internal_value)
                if isinstance(internal_value, Mapping)
                else {}
            )
            if not provider_renderer_valid:
                errors.append("bundle_provider_renderer_invalid")
        except (
            KeyError,
            OSError,
            UnicodeError,
            ValueError,
            zipfile.BadZipFile,
            json.JSONDecodeError,
        ):
            errors.append("bundle_internal_manifest_invalid")
    compared_fields = (
        "schema_version",
        "status",
        "provider_bundle_kind",
        "probe_kind",
        "run_id",
        "source_commit",
        "construction_envelope_source_digest",
        "portable_construction_envelope_digest",
        "toolchain_digest",
        "provider_python_runtime_required",
        "replacement_authoring_backend",
        "replacement_authoring_model_provider",
        "replacement_authoring_agent_runtime",
        "replacement_authoring_model",
        "replacement_authoring_agents_api_policy",
        "provider_python_runtime_manifest",
        "provider_python_runtime_digest",
        "provider_python_runtime_python_version",
        "raw_interiorgs_bytes_in_provider_bundle",
        # Cross-compared with everything else it authorizes: without this, a
        # receipt's decision could drift from the one sealed in the bundle,
        # and the byte-crossing claim would be checkable only against itself.
        "disclosure_decision",
        "single_parent_allocation",
        "nested_provider_mutations_performed",
        "evaluation_episode_executed",
        "expected_result_filename",
        "manifest_digest",
    )
    if diagnostic_only:
        compared_fields += (
            "construction_source_commit",
            "source_diagnostic_checkpoint_digest",
            "carried_completed_stage_count",
            "diagnostic_bootstrap_mode",
            "diagnostic_scientific_binding_digest",
            "diagnostic_stage_sequence_ids",
            "normal_production_lane_used",
            "diagnostic_only",
            "qualification_eligible",
            "executed_inside_one_parent_provider_run",
            "configured_revision_publication_permitted",
            "offering_publication_permitted",
            "terminal_e2e_completion_permitted",
        )
    elif production_semantic_reuse:
        compared_fields += (
            "construction_source_commit",
            "source_construction_envelope_digest",
            "configuration_revision_intake_receipt_digest",
            "configuration_revision_id",
            "production_semantic_input_reuse",
            "source_semantic_checkpoint_digest",
            "semantic_reuse_scientific_binding_digest",
            "semantic_reuse_completed_stage_prefix_count",
            "provider_render_outputs_reused",
            "semantic_teacher_outputs_reused",
            "full_downstream_stage_chain_required",
            "normal_production_runner_used",
            "configured_revision_publication_permitted",
            "offering_publication_permitted",
        )
    if receipt.get("provider_renderer_required") is True:
        compared_fields += (
            "provider_renderer_required",
            "provider_renderer_digest",
            "provider_renderer_source_runtime_digest",
        )
    if (
        not internal
        or internal.get("manifest_digest")
        != canonical_digest(internal, digest_field="manifest_digest")
        or any(internal.get(field) != receipt.get(field) for field in compared_fields)
    ):
        errors.append("bundle_internal_manifest_invalid")
    if errors:
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_receipt_invalid:"
            + ",".join(sorted(set(errors)))
        )
    return receipt


def portable_construction_envelope(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = Path(str(receipt.get("bundle_path") or ""))
    try:
        with zipfile.ZipFile(bundle) as archive:
            value = json.loads(
                archive.read(
                    "provider_runtime/input/portable_construction_envelope.v1.json"
                ).decode("utf-8")
            )
    except (
        KeyError,
        OSError,
        UnicodeError,
        ValueError,
        zipfile.BadZipFile,
        json.JSONDecodeError,
    ) as exc:
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_publication_envelope_unavailable"
        ) from exc
    envelope = dict(value) if isinstance(value, Mapping) else {}
    if (
        envelope.get("schema_version")
        != "task_evaluation_scene_construction_envelope.v1"
        or envelope.get("envelope_digest")
        != canonical_digest(envelope, digest_field="envelope_digest")
        or envelope.get("envelope_digest")
        != receipt.get("portable_construction_envelope_digest")
        or envelope.get("expected_production_commit")
        != receipt.get("source_commit")
        or envelope.get("run_id") != receipt.get("run_id")
    ):
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_publication_envelope_invalid"
        )
    return envelope


def bundle_requires_artifixer(receipt: Mapping[str, Any]) -> bool:
    """Derive service requirements from the sealed recipe, never a caller flag."""
    envelope = portable_construction_envelope(receipt)
    return any(stage["adapter"]["id"] == "artifixer3d_observed_object_removal"
               for stage in envelope["recipe"]["stage_sequence"])

