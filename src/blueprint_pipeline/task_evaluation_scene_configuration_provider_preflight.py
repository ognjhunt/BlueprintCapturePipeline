"""Fail-closed archive preflight for scene-configuration provider bundles."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_bundle import BUNDLE_SCHEMA_VERSION
from .task_evaluation_scene_configuration_builtin_producers import (
    TOOLCHAIN_SCHEMA_VERSION,
)
from .task_evaluation_scene_configuration_disclosure import (
    PENDING_PROVIDER_RENDER_STATUS,
    render_inputs_disclosure_is_coherent,
    renders_on_provider,
)
from .task_evaluation_scene_configuration_diagnostic_mode import (
    CHECKPOINT_RESUME_DIAGNOSTIC_BOOTSTRAP_MODE,
    FRESH_DIAGNOSTIC_BOOTSTRAP_MODE,
)
from .task_evaluation_splat_render_runtime import PROVIDER_RENDERER_SCHEMA_VERSION


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def scene_configuration_bundle_contract(
    archive: zipfile.ZipFile,
) -> tuple[dict[str, Any], set[str], list[str]]:
    """Verify every declared portable input before provider mutation."""

    root = "provider_runtime/"
    manifest_member = root + f"{BUNDLE_SCHEMA_VERSION}.json"
    envelope_member = root + "input/portable_construction_envelope.v1.json"
    toolchain_member = root + "toolchain/" + TOOLCHAIN_SCHEMA_VERSION + ".json"
    required = {
        manifest_member,
        envelope_member,
        toolchain_member,
        root + "run_task_evaluation_scene_configuration_provider.sh",
        root + "task_evaluation_scene_configuration_provider_runner.py",
        root + "blueprint_pipeline/__init__.py",
        root + "blueprint_pipeline/task_evaluation_scene_configuration_provider_runtime.py",
    }
    blockers: list[str] = []
    try:
        manifest = json.loads(archive.read(manifest_member).decode("utf-8"))
        envelope = json.loads(archive.read(envelope_member).decode("utf-8"))
        toolchain = json.loads(archive.read(toolchain_member).decode("utf-8"))
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        return {}, required, ["scene_configuration_provider_manifests_invalid"]
    if not all(isinstance(value, Mapping) for value in (manifest, envelope, toolchain)):
        return {}, required, ["scene_configuration_provider_manifests_invalid"]
    manifest = dict(manifest)
    envelope = dict(envelope)
    toolchain = dict(toolchain)
    source_commit = _string(manifest.get("source_commit"))
    diagnostic_only = manifest.get("diagnostic_only") is True
    production_semantic_reuse = (
        manifest.get("production_semantic_input_reuse") is True
    )
    diagnostic_bootstrap_mode = manifest.get("diagnostic_bootstrap_mode")
    fresh_diagnostic_bootstrap = (
        diagnostic_bootstrap_mode == FRESH_DIAGNOSTIC_BOOTSTRAP_MODE
    )
    render = envelope.get("render_inputs_result")
    renders_at_provider = (
        isinstance(render, Mapping)
        and renders_on_provider(render.get("disclosure_decision") or {})
    )
    provider_renderer_required = manifest.get("provider_renderer_required") is True
    expected_provider_renderer = renders_at_provider and not production_semantic_reuse and (
        not diagnostic_only or fresh_diagnostic_bootstrap
    )
    if provider_renderer_required:
        required.update(
            {
                root + "renderer/tools/splat_render/render_splat.mjs",
                root + "renderer/" + PROVIDER_RENDERER_SCHEMA_VERSION + ".json",
            }
        )
    if (
        manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or manifest.get("status") != "ready"
        or manifest.get("provider_bundle_kind")
        != "task_evaluation_scene_configuration"
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or manifest.get("manifest_digest")
        != canonical_digest(manifest, digest_field="manifest_digest")
        or manifest.get("raw_interiorgs_bytes_in_provider_bundle")
        is not expected_provider_renderer
        or provider_renderer_required is not expected_provider_renderer
        or (
            provider_renderer_required
            and (
                re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    _string(manifest.get("provider_renderer_digest")),
                )
                is None
                or re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    _string(manifest.get("provider_renderer_source_runtime_digest")),
                )
                is None
            )
        )
        or (
            diagnostic_only
            and (
                manifest.get("qualification_eligible") is not False
                or manifest.get("normal_production_lane_used") is not False
                or manifest.get("executed_inside_one_parent_provider_run") is not False
                or manifest.get("configured_revision_publication_permitted") is not False
                or manifest.get("offering_publication_permitted") is not False
                or manifest.get("terminal_e2e_completion_permitted") is not False
                or diagnostic_bootstrap_mode
                not in {
                    FRESH_DIAGNOSTIC_BOOTSTRAP_MODE,
                    CHECKPOINT_RESUME_DIAGNOSTIC_BOOTSTRAP_MODE,
                }
                or re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    _string(
                        manifest.get("diagnostic_scientific_binding_digest")
                    ),
                )
                is None
                or (
                    fresh_diagnostic_bootstrap
                    and (
                        manifest.get("source_diagnostic_checkpoint_digest")
                        is not None
                        or manifest.get("carried_completed_stage_count") != 0
                    )
                )
                or (
                    not fresh_diagnostic_bootstrap
                    and re.fullmatch(
                        r"sha256:[0-9a-f]{64}",
                        _string(
                            manifest.get(
                                "source_diagnostic_checkpoint_digest"
                            )
                        ),
                    )
                    is None
                )
            )
        )
        or (
            production_semantic_reuse
            and (
                diagnostic_only
                or manifest.get("provider_render_outputs_reused") is not True
                or manifest.get("semantic_teacher_outputs_reused") is not True
                or manifest.get("semantic_reuse_completed_stage_prefix_count") != 0
                or manifest.get("full_downstream_stage_chain_required") is not True
                or manifest.get("normal_production_runner_used") is not True
            )
        )
        or manifest.get("single_parent_allocation") is not True
        or manifest.get("nested_provider_mutations_performed") != 0
        or manifest.get("evaluation_episode_executed") is not False
    ):
        blockers.append("scene_configuration_provider_manifest_invalid")
    references = envelope.get("materialized_references")
    configurations = envelope.get("stage_configuration_references")
    disclosure = envelope.get("provider_disclosure_receipt")
    stages = (envelope.get("recipe") or {}).get("stage_sequence")
    expected_envelope_commit = (
        manifest.get("construction_source_commit")
        if diagnostic_only
        else source_commit
    )
    if (
        envelope.get("expected_production_commit") != expected_envelope_commit
        or envelope.get("envelope_digest")
        != canonical_digest(envelope, digest_field="envelope_digest")
        or manifest.get("portable_construction_envelope_digest")
        != envelope.get("envelope_digest")
        or not isinstance(references, list)
        or any(
            not isinstance(row, Mapping)
            or row.get("contract_path") == "scene.appearance.representation"
            or row.get("materialized_path") is not None
            for row in references or []
        )
        or not isinstance(configurations, list)
        or len(configurations) != 6
        or not isinstance(stages, list)
        or len(stages) != 6
        or (
            diagnostic_only
            and manifest.get("diagnostic_stage_sequence_ids")
            != [
                str(row.get("stage_id") or "")
                for row in stages or []
                if isinstance(row, Mapping)
            ]
        )
        or not isinstance(render, Mapping)
        or not render_inputs_disclosure_is_coherent(render)
        or not isinstance(disclosure, Mapping)
        or disclosure.get("raw_interiorgs_bytes_in_provider_bundle")
        is not expected_provider_renderer
        or disclosure.get("derived_rendered_views_in_provider_bundle")
        is expected_provider_renderer
        or (
            diagnostic_only
            and not fresh_diagnostic_bootstrap
            and (
                render.get("diagnostic_checkpoint_reused") is not True
                or render.get("provider_render_skipped") is not True
            )
        )
        or (
            production_semantic_reuse
            and (
                render.get("production_semantic_input_reuse") is not True
                or render.get("provider_render_skipped") is not True
            )
        )
        or (
            diagnostic_only
            and fresh_diagnostic_bootstrap
            and (
                render.get("status")
                != PENDING_PROVIDER_RENDER_STATUS
                or render.get("provider_render_required") is not True
            )
        )
    ):
        blockers.append("scene_configuration_portable_envelope_invalid")

    bound_rows: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(references, list):
        bound_rows.extend(
            (str(row.get("provider_relative_path") or ""), row)
            for row in references
            if isinstance(row, Mapping)
        )
    if isinstance(configurations, list):
        bound_rows.extend(
            (str(row.get("relative_path") or ""), row)
            for row in configurations
            if isinstance(row, Mapping)
        )
    if isinstance(render, Mapping):
        for key in ("camera_calibration", "render_manifest"):
            row = render.get(key)
            if isinstance(row, Mapping):
                bound_rows.append((str(row.get("path") or ""), row))
        bound_rows.extend(
            (str(row.get("path") or ""), row)
            for row in render.get("derived_frames") or []
            if isinstance(row, Mapping)
        )
    seen: set[str] = set()
    for relative, row in bound_rows:
        if (
            not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in seen
        ):
            blockers.append("scene_configuration_provider_input_path_invalid")
            continue
        seen.add(relative)
        member = root + relative
        required.add(member)
        try:
            body = archive.read(member)
        except KeyError:
            blockers.append("scene_configuration_provider_input_missing")
            continue
        if (
            row.get("size_bytes") != len(body)
            or row.get("digest")
            != "sha256:" + hashlib.sha256(body).hexdigest()
        ):
            blockers.append("scene_configuration_provider_input_digest_invalid")

    files = toolchain.get("files")
    expected_toolchain_commit = _string(manifest.get("toolchain_source_commit"))
    if (
        expected_toolchain_commit != source_commit
        or toolchain.get("schema_version") != TOOLCHAIN_SCHEMA_VERSION
        or toolchain.get("status") != "published_full_byte_readback_passed"
        or toolchain.get("source_commit") != expected_toolchain_commit
        or toolchain.get("full_byte_service_account_readback_passed") is not True
        or toolchain.get("toolchain_digest")
        != canonical_digest(toolchain, digest_field="toolchain_digest")
        or manifest.get("toolchain_digest") != toolchain.get("toolchain_digest")
        or not isinstance(files, list)
        or not files
    ):
        blockers.append("scene_configuration_provider_toolchain_invalid")
    else:
        for row in files:
            if not isinstance(row, Mapping):
                blockers.append("scene_configuration_provider_toolchain_invalid")
                continue
            relative = str(row.get("relative_path") or "")
            if not relative or relative.startswith("/") or ".." in Path(relative).parts:
                blockers.append("scene_configuration_provider_toolchain_invalid")
                continue
            member = root + "toolchain/" + relative
            required.add(member)
            try:
                body = archive.read(member)
            except KeyError:
                blockers.append("scene_configuration_provider_toolchain_invalid")
                continue
            if (
                row.get("size_bytes") != len(body)
                or row.get("sha256")
                != "sha256:" + hashlib.sha256(body).hexdigest()
            ):
                blockers.append("scene_configuration_provider_toolchain_invalid")
    return manifest, required, sorted(set(blockers))


__all__ = ["scene_configuration_bundle_contract"]
