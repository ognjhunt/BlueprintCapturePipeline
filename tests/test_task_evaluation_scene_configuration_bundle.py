from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_live_profile import file_digest
from blueprint_pipeline.task_evaluation_scene_configuration_bundle import (
    BUNDLE_SCHEMA_VERSION,
    TaskEvaluationSceneConfigurationBundleError,
    build_scene_configuration_provider_bundle,
    load_scene_configuration_provider_bundle_receipt,
)
from blueprint_pipeline.task_evaluation_splat_render_runtime import (
    PROVIDER_RENDERER_REQUIRED_PACKAGES,
    PROVIDER_RENDERER_SCHEMA_VERSION,
    TaskEvaluationSplatRenderRuntimeError,
    runtime_from_provider_bundle,
)
from blueprint_pipeline import task_evaluation_scene_configuration_bundle as bundle_module
from blueprint_pipeline import task_evaluation_scene_configuration_paid_authority as authority_module
from blueprint_pipeline import task_evaluation_live_profile as live_profile_module
from blueprint_pipeline import task_evaluation_scene_configuration_vast as scene_vast
from blueprint_pipeline import (
    task_evaluation_scene_configuration_provider_artifacts as provider_artifacts,
)
from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import (
    MAX_ATTEMPT_SPEND_USD,
    MAX_HOURLY_RATE_USD,
    MAX_PROVIDER_COMPUTE_SPEND_USD,
    MIN_ARTIFIXER_VISUAL_REVIEW_SPEND_USD,
    MIN_EXTERNAL_SERVICE_SPEND_USD,
    OUTPUT_AND_CLOSURE_RESERVE_SECONDS,
    PARENT_DEADLINE_EPOCH_ENV,
    REQUIRED_PARENT_TTL_SECONDS,
    SERIAL_GPU_STAGE_TIMEOUT_SECONDS,
)
from blueprint_pipeline.task_evaluation_scene_construction_queue import (
    ENVELOPE_SCHEMA_VERSION,
    TaskEvaluationSceneConstructionQueueError,
    ensure_scene_construction_queue_root,
    preflight_scene_construction_finalization,
)
from blueprint_pipeline import task_evaluation_scene_construction_queue as scene_queue
from blueprint_pipeline import vast_provider_adapter as vpa
from scripts.build_task_evaluation_scene_configuration_live_profile import (
    build_scene_configuration_live_profile,
)
from scripts.build_task_evaluation_scene_configuration_toolchain import (
    build_published_scene_configuration_toolchain,
)
from tests.test_build_task_evaluation_scene_configuration_toolchain import (
    _component_packages,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _openai_scope_attestation(
    *, paid_resource_class: str, api_key_id: str, project_id: str = "proj_test"
) -> str:
    value: dict[str, object] = {
        "schema_version": "openai_candidate_cost_scope_attestation.v1",
        "status": "approved",
        "issued_by_agent": False,
        "operator_id": "fixture-cost-owner",
        "provider_id": "openai",
        "paid_resource_class": paid_resource_class,
        "project_id": project_id,
        "api_key_id": api_key_id,
        "exclusive_use": True,
        "candidate_reported_usage_is_authoritative": False,
        "exclusive_from": "2026-08-01T00:00:00+00:00",
        "exclusive_until": "2026-09-30T00:00:00+00:00",
        "proof_effect": "none",
    }
    value["scope_attestation_digest"] = canonical_digest(
        value, digest_field="scope_attestation_digest"
    )
    return json.dumps(value, sort_keys=True)


def _configure_scene_openai_runtime_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    visual_review_class: str = (
        "task_evaluation_scene_configuration_artifixer_visual_review"
    ),
) -> None:
    files = {
        "OPENAI_ADMIN_API_KEY_FILE": "fixture-admin-key",
        "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_FILE": "fixture-semantic-key",
        "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE": "fixture-review-key",
        "OPENAI_CONTENT_AGENTS_API_KEY_FILE": "fixture-content-key",
        "BLUEPRINT_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_COST_SCOPE_ATTESTATION_FILE": (
            _openai_scope_attestation(
                paid_resource_class=(
                    "task_evaluation_scene_configuration_artifixer_semantic_teacher"
                ),
                api_key_id="key_semantic",
            )
        ),
        "BLUEPRINT_OPENAI_ARTIFIXER_VISUAL_REVIEW_COST_SCOPE_ATTESTATION_FILE": (
            _openai_scope_attestation(
                paid_resource_class=visual_review_class,
                api_key_id="key_review",
            )
        ),
        "BLUEPRINT_OPENAI_CONTENT_AGENTS_COST_SCOPE_ATTESTATION_FILE": (
            _openai_scope_attestation(
                paid_resource_class="task_evaluation_scene_configuration_content_agents",
                api_key_id="key_content_agents",
            )
        ),
    }
    for name, payload in files.items():
        path = tmp_path / name.lower()
        path.write_text(payload, encoding="utf-8")
        path.chmod(0o640)
        monkeypatch.setenv(name, str(path))
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj_test")
    monkeypatch.setenv(
        "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_ID", "key_semantic"
    )
    monkeypatch.setenv("OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_ID", "key_review")
    monkeypatch.setenv("OPENAI_CONTENT_AGENTS_API_KEY_ID", "key_content_agents")


def _bound(path: Path, **extra: object) -> dict[str, object]:
    return {
        "materialized_path": str(path),
        "digest": _sha256(path),
        "size_bytes": path.stat().st_size,
        "full_byte_service_account_readback_passed": True,
        **extra,
    }


def _toolchain(root: Path, commit: str) -> Path:
    build_published_scene_configuration_toolchain(
        source_commit=commit,
        output_root=root,
        readback=lambda path: path.read_bytes(),
        readback_actor="service-account:test",
        component_packages=_component_packages(root.parent),
    )
    return root


def _repo(root: Path) -> Path:
    package = root / "src/blueprint_pipeline"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("\n", encoding="utf-8")
    (package / "task_evaluation_scene_configuration_provider_runtime.py").write_text(
        "def execute_scene_configuration_stage_chain(**kwargs):\n    return kwargs\n",
        encoding="utf-8",
    )
    source_root = Path(__file__).resolve().parents[1]
    (
        package / "task_evaluation_scene_configuration_python_runtime.py"
    ).write_bytes(
        (
            source_root
            / "src/blueprint_pipeline/task_evaluation_scene_configuration_python_runtime.py"
        ).read_bytes()
    )
    scripts = root / "scripts"
    scripts.mkdir()
    for name in (
        "run_task_evaluation_scene_configuration_provider.sh",
        "task_evaluation_scene_configuration_provider_runner.py",
        "task_evaluation_scene_configuration_diagnostic_provider_runner.py",
    ):
        (scripts / name).write_bytes((source_root / "scripts" / name).read_bytes())
    for relative in (
        "tools/splat_render/render_splat.mjs",
        "tools/splat_render/src/render_entry.mjs",
        "tools/splat_render/harness.html",
        "tools/splat_render/package.json",
        "tools/splat_render/package-lock.json",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"name":"fixture"}\n'
            if path.name in {"package.json", "package-lock.json"}
            else relative + "\n",
            encoding="utf-8",
        )
    return root


def _envelope(root: Path, commit: str) -> Path:
    inputs = root / "inputs"
    inputs.mkdir()
    raw_splat = inputs / "scene-secret-raw.ply"
    raw_splat.write_bytes(b"RAW_INTERIORGS_BYTES_MUST_NEVER_LEAVE_CONTROL_PLANE")
    # Production CAS materialization is digest-only even though the immutable
    # publisher URI declares an OpenUSD layer.  Keep the fixture shaped like
    # that boundary: using the local suffix would miss the provider failure.
    sage = inputs / "6c1e9a5a0f0f1d8e"
    sage.write_text("#usda 1.0\n", encoding="utf-8")
    cameras = inputs / "cameras.json"
    cameras.write_text('[{"id":"camera-0"}]\n', encoding="utf-8")
    render_manifest = inputs / "render.json"
    render_manifest.write_text('{"status":"rendered"}\n', encoding="utf-8")
    frame = inputs / "frame.png"
    Image.new("RGB", (16, 16), color=(90, 80, 70)).save(frame)
    mask = inputs / "mask.png"
    Image.new("L", (16, 16), color=255).save(mask)
    removed = inputs / "source-object-candidate.ply"
    removed.write_bytes(b"derived-source-object-candidate")
    retained = inputs / "retained-scene.ply"
    retained.write_bytes(b"derived-retained-scene")
    render = {
        "schema_version": "task_evaluation_scene_configuration_render_inputs.v1",
        "status": "derived_method_inputs_materialized",
        "run_id": "configure-scene-839873-v1",
        "source_splat_digest": _sha256(raw_splat),
        "raw_interiorgs_bytes_in_provider_packet": False,
        "camera_calibration": _bound(cameras),
        "render_manifest": _bound(render_manifest),
        "derived_frames": [
            _bound(
                frame,
                camera_id="camera-0",
                source_object_mask=_bound(
                    mask,
                    projection_kind=(
                        "registered_world_aabb_conservative_projection"
                    ),
                    observed_segmentation_truth=False,
                    pixel_bounds_xyxy=[0, 0, 16, 16],
                    foreground_pixel_count=256,
                ),
            )
        ],
        "derived_frame_count": 1,
        "source_object_masks": {
            "count": 1,
            "source": "registered_source_object_bounds_projection",
            "source_object_identity": {"publisher_instance_id": "104"},
            "observed_segmentation_truth": False,
            "all_masks_digest_bound": True,
        },
        "derived_gaussian_cutout": {
            "selection_rule": (
                "gaussian_center_inside_registered_source_object_aabb"
            ),
            "aabb_padding_m": 0.0,
            "source_count": 4,
            "removed_count": 1,
            "retained_count": 3,
            "source_object_candidate": _bound(removed),
            "retained_scene_without_source_object": _bound(retained),
            "retained_rows_byte_exact": True,
            "selection_is_candidate_not_observed_object_ownership_truth": True,
            "raw_source_bytes_in_provider_packet": False,
        },
        "result_digest": "",
    }
    render["result_digest"] = canonical_digest(render, digest_field="result_digest")
    stages = []
    configurations = []
    for index in range(6):
        stage_id = f"stage-{index + 1}"
        config = inputs / f"{stage_id}.json"
        config.write_text(json.dumps({"stage_id": stage_id}), encoding="utf-8")
        stages.append(
            {
                "stage_id": stage_id,
                "capability": f"capability-{index + 1}",
                "execution_class": "no_spend",
                "depends_on": [] if index == 0 else [f"stage-{index}"],
            }
        )
        configurations.append(_bound(config, stage_id=stage_id))
    envelope = {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "orchestration_id": "prepare-scene-839873-v1",
        "run_id": "configure-scene-839873-v1",
        "expected_production_commit": commit,
        "recipe_digest": "sha256:" + "9" * 64,
        "recipe": {"stage_sequence": stages},
        "materialized_references": [
            _bound(
                raw_splat,
                contract_path="scene.appearance.representation",
            ),
            _bound(
                sage,
                contract_path="scene.geometry.collision",
                uri="s3://blueprint/task-evaluation/sage/839873_collision.usd",
            ),
        ],
        "stage_configuration_references": configurations,
        "render_inputs_result": render,
        "envelope_digest": "",
    }
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    path = root / "envelope.json"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    return path


def _provider_render_envelope(root: Path, commit: str) -> Path:
    path = _envelope(root, commit)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    render = envelope["render_inputs_result"]
    appearance = next(
        row
        for row in envelope["materialized_references"]
        if row["contract_path"] == "scene.appearance.representation"
    )
    render.update(
        {
            "status": "derived_method_inputs_pending_provider_render",
            "source_splat_bytes_retained_on_control_plane": False,
            "raw_interiorgs_bytes_in_provider_packet": True,
            "provider_disclosure_scope": (
                "source_appearance_bytes_and_derived_views"
            ),
            "disclosure_decision": _decision(provider=True),
            "render_execution_site": "provider_gpu",
            "source_appearance": dict(appearance),
            "render_manifest": None,
            "derived_frames": [],
            "derived_frame_count": 0,
            "provider_render_required": True,
        }
    )
    render["source_object_masks"]["count"] = 0
    render["derived_gaussian_cutout"][
        "raw_source_bytes_in_provider_packet"
    ] = True
    render["result_digest"] = canonical_digest(
        render, digest_field="result_digest"
    )
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    path.write_text(json.dumps(envelope), encoding="utf-8")
    return path


def _build(tmp_path: Path, name: str) -> dict:
    commit = "a" * 40
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    envelope = source / "envelope.json"
    if not envelope.exists():
        envelope = _envelope(source, commit)
    toolchain = tmp_path / "toolchain"
    if not toolchain.exists():
        _toolchain(toolchain, commit)
    repo = tmp_path / "repo"
    if not repo.exists():
        _repo(repo)
    return build_scene_configuration_provider_bundle(
        construction_envelope_path=envelope,
        toolchain_root=toolchain,
        repository_root=repo,
        output_root=tmp_path / name,
        expected_source_commit=commit,
    )


def _bind_real_stage_three_configuration(
    envelope_path: Path, *, include_authority_denial: bool
) -> bytes:
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    subject = {"id": "scene-839873-mug-replacement", "version": "v1"}
    configuration = {
        "schema_version": "rigid_replacement_authoring_configuration.v1",
        "replacement_identity": subject,
        "metric_envelope": {
            "minimum_xyz_m": [0.0, 0.0, 0.0],
            "maximum_xyz_m": [0.2, 0.2, 0.3],
            "maximum_dimension_relative_error": 0.05,
        },
        "required_output": {
            "format": "OpenUSD",
            "rigid_body": True,
            "single_movable_root": True,
            "units": "meters",
            "up_axis": "Z",
            "mass_kg_bounds": [0.2, 0.8],
            "static_friction_bounds": [0.3, 0.9],
            "dynamic_friction_bounds": [0.2, 0.8],
            "restitution_bounds": [0.0, 0.15],
        },
    }
    if include_authority_denial:
        configuration["physics_authority_granted_by_authoring"] = False
    payload = (json.dumps(configuration, indent=1) + "\n").encode()
    path = Path(envelope["stage_configuration_references"][2]["materialized_path"])
    path.write_bytes(payload)
    envelope["recipe"]["subject_identity"] = subject
    envelope["recipe"]["stage_sequence"][2][
        "capability"
    ] = "rigid_replacement_authoring"
    envelope["stage_configuration_references"][2].update(
        {"digest": _sha256(path), "size_bytes": path.stat().st_size}
    )
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    envelope_path.write_text(json.dumps(envelope), encoding="utf-8")
    return payload


def test_bundle_refuses_invalid_stage_configuration_before_output_creation(
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    source = tmp_path / "invalid-stage-configuration-source"
    source.mkdir()
    envelope = _envelope(source, commit)
    _bind_real_stage_three_configuration(
        envelope, include_authority_denial=False
    )
    output = tmp_path / "invalid-stage-configuration-bundle"

    with pytest.raises(
        TaskEvaluationSceneConfigurationBundleError,
        match=(
            "scene_configuration_stage_configuration_preflight_failed:"
            "stage-3:rigid_replacement_authoring:"
            "physics_authority_granted_by_authoring"
        ),
    ):
        build_scene_configuration_provider_bundle(
            construction_envelope_path=envelope,
            toolchain_root=tmp_path / "toolchain-was-not-needed",
            repository_root=tmp_path / "repo-was-not-needed",
            output_root=output,
            expected_source_commit=commit,
        )

    assert not output.exists()


def test_bundle_preserves_valid_external_stage_configuration_bytes(
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    source = tmp_path / "valid-stage-configuration-source"
    source.mkdir()
    envelope = _envelope(source, commit)
    expected = _bind_real_stage_three_configuration(
        envelope, include_authority_denial=True
    )
    toolchain = _toolchain(tmp_path / "valid-stage-configuration-toolchain", commit)
    repo = _repo(tmp_path / "valid-stage-configuration-repo")
    receipt = build_scene_configuration_provider_bundle(
        construction_envelope_path=envelope,
        toolchain_root=toolchain,
        repository_root=repo,
        output_root=tmp_path / "valid-stage-configuration-bundle",
        expected_source_commit=commit,
    )

    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        actual = archive.read("provider_runtime/input/configurations/stage-3.json")
    assert actual == expected


def test_bundle_zip_is_not_group_writable_before_content_addressing(
    tmp_path: Path,
) -> None:
    original_umask = os.umask(0o002)
    try:
        receipt = _build(tmp_path, "content-addressable-bundle")
    finally:
        os.umask(original_umask)

    assert os.stat(receipt["bundle_path"]).st_mode & 0o777 == 0o644


def test_bundle_does_not_retain_its_rebuildable_expansion_tree(
    tmp_path: Path,
) -> None:
    receipt = _build(tmp_path, "bundle-without-stage-cache")

    assert Path(receipt["bundle_path"]).is_file()
    assert not (tmp_path / "bundle-without-stage-cache" / "stage").exists()


def _construction_queue(tmp_path: Path) -> Path:
    envelope_path = tmp_path / "source" / "envelope.json"
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    root = ensure_scene_construction_queue_root(tmp_path / "construction-queue")
    filename = (
        f"{envelope['orchestration_id']}-"
        f"{envelope['recipe_digest'].removeprefix('sha256:')}.json"
    )
    target = root / "pending" / filename
    if not target.exists():
        target.write_bytes(envelope_path.read_bytes())
        target.chmod(0o440)
    return root


def _provider_runtime(
    root: Path, *, repo: Path, commit: str
) -> tuple[Path, dict[str, object]]:
    runtime = root / "splat-render-runtime"
    source_renderer = runtime / "renderer"
    for relative in (
        "tools/splat_render/render_splat.mjs",
        "tools/splat_render/src/render_entry.mjs",
        "tools/splat_render/harness.html",
        "tools/splat_render/package.json",
        "tools/splat_render/package-lock.json",
    ):
        target = source_renderer / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo / relative).read_bytes())
    for package in PROVIDER_RENDERER_REQUIRED_PACKAGES:
        marker = source_renderer / "tools/splat_render/node_modules" / package / "index.js"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(package, encoding="utf-8")
    (source_renderer / "tools/splat_render/node_modules/three/empty.js").write_bytes(b"")
    browser = runtime / "browser/chrome"
    node = runtime / "node/bin/node"
    browser.parent.mkdir(parents=True)
    node.parent.mkdir(parents=True)
    browser.write_bytes(b"linux-chromium")
    node.write_bytes(b"linux-node")
    browser.chmod(0o755)
    node.chmod(0o755)
    identity = {
        "node": str(node),
        "browser_executable": str(browser),
        "renderer_root": str(source_renderer),
        "identity": {
            "mode": "immutable_host_runtime",
            "schema_version": "task_evaluation_splat_render_runtime.v1",
            "runtime_digest": "sha256:" + "9" * 64,
            "source_commit": commit,
            "platform": "linux-x86_64",
            "file_count": 17,
            "full_byte_service_account_readback_passed": True,
        },
    }
    return runtime, identity


def test_scene_construction_finalization_preflight_binds_exact_writable_queue_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build(tmp_path, "bundle")
    receipt = load_scene_configuration_provider_bundle_receipt(
        tmp_path / "bundle" / f"{BUNDLE_SCHEMA_VERSION}.receipt.json"
    )
    queue_root = _construction_queue(tmp_path)
    envelope = scene_vast._portable_construction_envelope(receipt)

    ready = preflight_scene_construction_finalization(
        queue_root=queue_root,
        envelope=envelope,
    )

    assert ready["status"] == "ready"
    assert ready["run_id"] == receipt["run_id"]
    assert ready["construction_envelope_digest"] == envelope[
        "control_plane_envelope_digest"
    ]
    assert ready["provider_mutation_performed"] is False
    assert ready["paid_execution_requested"] is False

    real_access = scene_queue.os.access
    monkeypatch.setattr(
        scene_queue.os,
        "access",
        lambda path, mode: False
        if Path(path).name == "completed"
        else real_access(path, mode),
    )
    with pytest.raises(
        TaskEvaluationSceneConstructionQueueError,
        match="scene_construction_queue_finalization_destination_unwritable",
    ):
        preflight_scene_construction_finalization(
            queue_root=queue_root,
            envelope=envelope,
        )


def _build_provider_render_bundle(
    tmp_path: Path, name: str, monkeypatch: pytest.MonkeyPatch
) -> dict:
    commit = "a" * 40
    source = tmp_path / f"{name}-source"
    source.mkdir()
    envelope = _provider_render_envelope(source, commit)
    toolchain = _toolchain(tmp_path / f"{name}-toolchain", commit)
    repo = _repo(tmp_path / f"{name}-repo")
    runtime, identity = _provider_runtime(tmp_path / name, repo=repo, commit=commit)
    monkeypatch.setattr(
        bundle_module,
        "validate_splat_render_runtime",
        lambda **_kwargs: identity,
    )
    return build_scene_configuration_provider_bundle(
        construction_envelope_path=envelope,
        toolchain_root=toolchain,
        repository_root=repo,
        splat_render_runtime_root=runtime,
        output_root=tmp_path / f"{name}-bundle",
        expected_source_commit=commit,
    )


def test_provider_render_bundle_requires_and_ships_its_exact_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    source = tmp_path / "provider-render-source"
    source.mkdir()
    envelope = _provider_render_envelope(source, commit)
    toolchain = _toolchain(tmp_path / "provider-render-toolchain", commit)
    repo = _repo(tmp_path / "provider-render-repo")

    with pytest.raises(
        TaskEvaluationSceneConfigurationBundleError,
        match="scene_configuration_bundle_provider_render_runtime_missing",
    ):
        build_scene_configuration_provider_bundle(
            construction_envelope_path=envelope,
            toolchain_root=toolchain,
            repository_root=repo,
            output_root=tmp_path / "provider-render-missing-runtime",
            expected_source_commit=commit,
        )

    runtime, identity = _provider_runtime(tmp_path, repo=repo, commit=commit)
    monkeypatch.setattr(
        bundle_module,
        "validate_splat_render_runtime",
        lambda **_kwargs: identity,
    )

    receipt = build_scene_configuration_provider_bundle(
        construction_envelope_path=envelope,
        toolchain_root=toolchain,
        repository_root=repo,
        splat_render_runtime_root=runtime,
        output_root=tmp_path / "provider-render-with-runtime",
        expected_source_commit=commit,
    )

    assert receipt["provider_renderer_required"] is True
    assert receipt["provider_renderer_source_runtime_digest"] == (
        identity["identity"]["runtime_digest"]
    )
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
        renderer_manifest = json.loads(
            archive.read(
                f"provider_runtime/renderer/{PROVIDER_RENDERER_SCHEMA_VERSION}.json"
            )
        )
    for relative in (
        "tools/splat_render/render_splat.mjs",
        "tools/splat_render/src/render_entry.mjs",
        "tools/splat_render/harness.html",
        "tools/splat_render/package.json",
        "tools/splat_render/package-lock.json",
    ):
        assert f"provider_runtime/renderer/{relative}" in names
    for package in PROVIDER_RENDERER_REQUIRED_PACKAGES:
        assert (
            f"provider_runtime/renderer/tools/splat_render/node_modules/{package}/index.js"
            in names
        )
    assert "provider_runtime/renderer/browser/chrome" in names
    assert "provider_runtime/renderer/node/bin/node" in names
    node_row = next(
        row for row in renderer_manifest["files"]
        if row["relative_path"] == "node/bin/node"
    )
    browser_row = next(
        row for row in renderer_manifest["files"]
        if row["relative_path"] == "browser/chrome"
    )
    assert node_row["executable"] is True
    assert browser_row["executable"] is True
    assert renderer_manifest["entrypoints"]["node"] == "node/bin/node"
    assert renderer_manifest["renderer_digest"] == receipt["provider_renderer_digest"]

    extracted = tmp_path / "provider-render-extracted"
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        archive.extractall(extracted)
    reopened = runtime_from_provider_bundle(
        provider_runtime_root=extracted / "provider_runtime",
    )
    assert reopened["identity"]["renderer_digest"] == receipt["provider_renderer_digest"]
    assert Path(reopened["browser_executable"]).stat().st_mode & 0o111
    assert Path(reopened["node"]).read_bytes() == b"linux-node"
    assert Path(reopened["node"]).stat().st_mode & 0o111
    assert Path(reopened["repository_root"]) == Path(reopened["renderer_root"])
    browser = Path(reopened["browser_executable"])
    browser.chmod(0o644)
    browser.write_bytes(b"tampered-browser")
    with pytest.raises(
        TaskEvaluationSplatRenderRuntimeError,
        match="provider_splat_renderer_file_invalid:browser/chrome",
    ):
        runtime_from_provider_bundle(
            provider_runtime_root=extracted / "provider_runtime",
        )
    preflight = vpa._blueprint_bundle_preflight(
        job_dir=tmp_path / "provider-render-preflight",
        generated_at="2026-08-26T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="task_evaluation_scene_configuration",
        bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://objects.example.test/provider-render.zip",
        provider_output_put_url="https://objects.example.test/output.zip",
    )
    assert preflight["blockers"] == []
    assert preflight["status"] == "passed"

    missing_renderer = tmp_path / "missing-renderer-entry.zip"
    required_renderer = "provider_runtime/renderer/tools/splat_render/render_splat.mjs"
    with (
        zipfile.ZipFile(receipt["bundle_path"]) as source,
        zipfile.ZipFile(missing_renderer, "w") as destination,
    ):
        for info in source.infolist():
            if info.filename != required_renderer:
                destination.writestr(info, source.read(info.filename))
    rebound = {
        **receipt,
        "bundle_path": str(missing_renderer),
        "bundle_sha256": _sha256(missing_renderer),
        "bundle_size_bytes": missing_renderer.stat().st_size,
        "receipt_digest": "",
    }
    rebound["receipt_digest"] = canonical_digest(
        rebound, digest_field="receipt_digest"
    )
    rebound_path = tmp_path / "missing-renderer-receipt.json"
    rebound_path.write_text(json.dumps(rebound), encoding="utf-8")
    with pytest.raises(
        TaskEvaluationSceneConfigurationBundleError,
        match="bundle_provider_renderer_invalid",
    ):
        load_scene_configuration_provider_bundle_receipt(rebound_path)
    missing_renderer_preflight = vpa._blueprint_bundle_preflight(
        job_dir=tmp_path / "provider-render-missing-entry-preflight",
        generated_at="2026-08-26T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="task_evaluation_scene_configuration",
        bundle_path=missing_renderer,
        provider_bundle_url="https://objects.example.test/provider-render.zip",
        provider_output_put_url="https://objects.example.test/output.zip",
    )
    assert missing_renderer_preflight["status"] == "blocked"
    assert (
        "provider_runtime_bundle_required_entries_missing"
        in missing_renderer_preflight["blockers"]
    )
    assert required_renderer in missing_renderer_preflight["missing_zip_entries"]


def test_diagnostic_bundle_reuses_checkpoint_without_raw_source_or_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    source = tmp_path / "diagnostic-source"
    source.mkdir()
    envelope_path = _provider_render_envelope(source, commit)
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    render = envelope["render_inputs_result"]
    toolchain = _toolchain(tmp_path / "diagnostic-toolchain", commit)
    repo = _repo(tmp_path / "diagnostic-repo")
    runtime, identity = _provider_runtime(
        tmp_path / "diagnostic-runtime", repo=repo, commit=commit
    )
    monkeypatch.setattr(
        bundle_module, "validate_splat_render_runtime", lambda **_kwargs: identity
    )
    checkpoint_renderer_identity = {
        "mode": "digest_bound_provider_bundle_renderer",
        "schema_version": PROVIDER_RENDERER_SCHEMA_VERSION,
        "renderer_digest": "sha256:" + "9" * 64,
        "source_runtime_digest": identity["identity"]["runtime_digest"],
        "source_commit": commit,
        "platform": identity["identity"]["platform"],
        # Provider packaging intentionally inventories only the staged renderer
        # subset, not every browser/runtime file in the host runtime receipt.
        "file_count": 11,
        "provider_full_byte_inventory_reopened": True,
    }
    checkpoint_root = tmp_path / "diagnostic-checkpoint"
    checkpoint_root.mkdir()
    inventory = []

    def add(role: str, relative: str, payload: bytes) -> dict[str, object]:
        path = checkpoint_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        row = {
            "role": role,
            "relative_path": relative,
            "digest": _sha256(path),
            "size_bytes": path.stat().st_size,
            "mode": 0o644,
        }
        inventory.append(row)
        return {
            "checkpoint_role": role,
            "digest": row["digest"],
            "size_bytes": row["size_bytes"],
        }

    calibration = add("camera_calibration", "render/calibration.json", b"[]\n")
    render_manifest = add("render_manifest", "render/manifest.json", b"{}\n")
    retained = add(
        "retained_scene_without_source_object", "render/retained.ply", b"retained"
    )
    frames = []
    for index in range(8):
        camera_id = f"camera-{index}"
        frame = add(
            f"raw_frame:{camera_id}",
            f"render/frames/{index}.png",
            f"frame-{index}".encode(),
        )
        frame["camera_id"] = camera_id
        frame["source_object_mask"] = add(
            f"source_object_mask:{camera_id}",
            f"render/masks/{index}.png",
            f"mask-{index}".encode(),
        )
        add(
            f"semantic_teacher_frame:{camera_id}",
            f"semantic/frames/{index}.png",
            f"semantic-{index}".encode(),
        )
        frames.append(frame)
    add("semantic_runtime_request", "semantic/request.json", b"{}\n")
    add("semantic_runtime_result", "semantic/result.json", b"{}\n")
    add("semantic_teacher_receipt", "semantic/receipt.json", b"{}\n")
    checkpoint = {
        "checkpoint_digest": "sha256:" + "c" * 64,
        "scientific_bindings": {"binding_digest": "sha256:" + "d" * 64},
        "completed_stage_prefix_count": 0,
        "inventory": inventory,
        "render_inputs_template": {
            **render,
            "status": "derived_method_inputs_materialized",
            "source_appearance": {
                "digest": render["source_appearance"]["digest"],
                "size_bytes": render["source_appearance"]["size_bytes"],
                "checkpoint_role": None,
            },
            # A checkpoint records the renderer after the provider has reopened
            # the staged renderer inventory.  A retry validates the same bytes
            # from the retained host runtime, whose receipt uses a different
            # schema and mode.
            "renderer_runtime": checkpoint_renderer_identity,
            "camera_calibration": calibration,
            "render_manifest": render_manifest,
            "derived_frames": frames,
            "derived_frame_count": 8,
            "derived_gaussian_cutout": {
                "retained_scene_without_source_object": retained,
            },
            "render_completed_on_provider": True,
            "result_digest": "sha256:" + "e" * 64,
        },
    }
    (checkpoint_root / "task_evaluation_scene_configuration_diagnostic_checkpoint.v1.json").write_text(
        "{}\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        bundle_module,
        "validate_scene_configuration_diagnostic_checkpoint",
        lambda **_kwargs: checkpoint,
    )
    def scientific_binding(**kwargs):
        assert kwargs["stage_input"]["stage"] == envelope["recipe"][
            "stage_sequence"
        ][0]
        assert kwargs["render_inputs"]["renderer_runtime"] == (
            checkpoint_renderer_identity
        )
        return checkpoint["scientific_bindings"]["binding_digest"]

    monkeypatch.setattr(
        bundle_module,
        "diagnostic_checkpoint_scientific_binding_digest",
        scientific_binding,
    )
    checkpoint_files = [path for path in checkpoint_root.rglob("*") if path.is_file()]
    checkpoint_manifest = (
        checkpoint_root
        / "task_evaluation_scene_configuration_diagnostic_checkpoint.v1.json"
    )
    advanced_reference = {
        "schema_version": (
            "task_evaluation_scene_configuration_advanced_checkpoint_reference.v1"
        ),
        "status": "validated_diagnostic_checkpoint_ready_for_next_retry",
        "checkpoint_root": str(checkpoint_root),
        "manifest_path": str(checkpoint_manifest),
        "manifest_sha256": _sha256(checkpoint_manifest),
        "checkpoint_digest": checkpoint["checkpoint_digest"],
        "completed_stage_prefix_count": 0,
        "file_count": len(checkpoint_files),
        "total_bytes": sum(path.stat().st_size for path in checkpoint_files),
        "diagnostic_only": True,
        "qualification_eligible": False,
        "source_provider_result_digest": "sha256:" + "f" * 64,
        "reference_digest": "",
    }
    advanced_reference["reference_digest"] = canonical_digest(
        advanced_reference, digest_field="reference_digest"
    )
    advanced_reference_path = tmp_path / "advanced-checkpoint-reference.json"
    advanced_reference_path.write_text(json.dumps(advanced_reference), encoding="utf-8")

    receipt = build_scene_configuration_provider_bundle(
        construction_envelope_path=envelope_path,
        toolchain_root=toolchain,
        repository_root=repo,
        splat_render_runtime_root=runtime,
        diagnostic_checkpoint_reference_path=advanced_reference_path,
        output_root=tmp_path / "diagnostic-bundle",
        expected_source_commit=commit,
    )
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
        portable = json.loads(
            archive.read(
                "provider_runtime/input/portable_construction_envelope.v1.json"
            )
        )
    assert receipt["diagnostic_only"] is True
    assert receipt["raw_interiorgs_bytes_in_provider_bundle"] is False
    assert not any(name.startswith("provider_runtime/renderer/") for name in names)
    assert not any("browser/chrome" in name for name in names)
    assert not any("node_modules/" in name for name in names)
    assert not any("input/render/source_appearance" in name for name in names)
    assert all(
        row["path"].startswith("input/diagnostic_checkpoint/")
        for row in portable["render_inputs_result"]["derived_frames"]
    )
    assert "path" not in portable["render_inputs_result"]["source_appearance"]
    assert load_scene_configuration_provider_bundle_receipt(
        tmp_path
        / "diagnostic-bundle"
        / f"{BUNDLE_SCHEMA_VERSION}.receipt.json",
        diagnostic_only=True,
    )["bundle_sha256"] == receipt["bundle_sha256"]
    preflight = vpa._blueprint_bundle_preflight(
        job_dir=tmp_path / "diagnostic-preflight",
        generated_at="2026-08-26T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="task_evaluation_scene_configuration",
        bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://objects.example.test/diagnostic.zip",
        provider_output_put_url="https://objects.example.test/output.zip",
    )
    assert preflight["blockers"] == []
    assert preflight["status"] == "passed"

    unsafe = dict(advanced_reference)
    unsafe["manifest_path"] = str(tmp_path / "outside.json")
    unsafe["reference_digest"] = canonical_digest(
        unsafe, digest_field="reference_digest"
    )
    unsafe_path = tmp_path / "unsafe-reference.json"
    unsafe_path.write_text(json.dumps(unsafe), encoding="utf-8")
    with pytest.raises(
        TaskEvaluationSceneConfigurationBundleError,
        match="scene_configuration_bundle_diagnostic_checkpoint_reference_invalid",
    ):
        build_scene_configuration_provider_bundle(
            construction_envelope_path=envelope_path,
            toolchain_root=toolchain,
            repository_root=repo,
            splat_render_runtime_root=runtime,
            diagnostic_checkpoint_reference_path=unsafe_path,
            output_root=tmp_path / "unsafe-diagnostic-bundle",
            expected_source_commit=commit,
        )


def test_fresh_diagnostic_bootstrap_ships_renderer_and_no_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    source = tmp_path / "fresh-diagnostic-source"
    source.mkdir()
    envelope = _provider_render_envelope(source, commit)
    toolchain = _toolchain(tmp_path / "fresh-diagnostic-toolchain", commit)
    repo = _repo(tmp_path / "fresh-diagnostic-repo")
    runtime, identity = _provider_runtime(tmp_path, repo=repo, commit=commit)
    monkeypatch.setattr(
        bundle_module,
        "validate_splat_render_runtime",
        lambda **_kwargs: identity,
    )

    receipt = build_scene_configuration_provider_bundle(
        construction_envelope_path=envelope,
        toolchain_root=toolchain,
        repository_root=repo,
        splat_render_runtime_root=runtime,
        output_root=tmp_path / "fresh-diagnostic-bundle",
        expected_source_commit=commit,
        fresh_diagnostic_bootstrap=True,
    )

    assert receipt["diagnostic_only"] is True
    assert receipt["qualification_eligible"] is False
    assert receipt["diagnostic_bootstrap_mode"] == "fresh"
    assert receipt["source_diagnostic_checkpoint_digest"] is None
    assert receipt["carried_completed_stage_count"] == 0
    assert receipt["diagnostic_scientific_binding_digest"].startswith(
        "sha256:"
    )
    assert receipt["raw_interiorgs_bytes_in_provider_bundle"] is True
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
        runner = archive.read(
            "provider_runtime/task_evaluation_scene_configuration_provider_runner.py"
        )
    assert any(name.startswith("provider_runtime/renderer/") for name in names)
    assert any(
        name.startswith("provider_runtime/input/render/source_appearance")
        for name in names
    )
    assert not any(
        name.startswith("provider_runtime/input/diagnostic_checkpoint/")
        for name in names
    )
    assert runner == (
        repo / "scripts/task_evaluation_scene_configuration_diagnostic_provider_runner.py"
    ).read_bytes()
    reopened = load_scene_configuration_provider_bundle_receipt(
        tmp_path
        / "fresh-diagnostic-bundle"
        / f"{BUNDLE_SCHEMA_VERSION}.receipt.json",
        diagnostic_only=True,
    )
    assert reopened["bundle_sha256"] == receipt["bundle_sha256"]
    preflight = vpa._blueprint_bundle_preflight(
        job_dir=tmp_path / "fresh-diagnostic-preflight",
        generated_at="2026-08-27T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="task_evaluation_scene_configuration",
        bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://objects.example.test/fresh-diagnostic.zip",
        provider_output_put_url="https://objects.example.test/output.zip",
    )
    assert preflight["blockers"] == []
    assert preflight["status"] == "passed"


def test_fresh_diagnostic_separates_executable_and_construction_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    construction_commit = "a" * 40
    diagnostic_commit = "b" * 40
    source = tmp_path / "split-identity-source"
    source.mkdir()
    envelope = _provider_render_envelope(source, construction_commit)
    envelope_value = json.loads(envelope.read_text(encoding="utf-8"))
    for index, row in enumerate(envelope_value["stage_configuration_references"]):
        row.pop("stage_id", None)
        row["contract_path"] = (
            f"construction.recipe.stage_sequence.{index}.configuration"
        )
    envelope_value["envelope_digest"] = canonical_digest(
        envelope_value, digest_field="envelope_digest"
    )
    envelope.write_text(json.dumps(envelope_value), encoding="utf-8")
    toolchain = _toolchain(
        tmp_path / "split-identity-toolchain", diagnostic_commit
    )
    repo = _repo(tmp_path / "split-identity-repo")
    runtime, identity = _provider_runtime(
        tmp_path, repo=repo, commit=construction_commit
    )
    monkeypatch.setattr(
        bundle_module,
        "validate_diagnostic_splat_render_runtime",
        lambda **_kwargs: identity,
    )

    receipt = build_scene_configuration_provider_bundle(
        construction_envelope_path=envelope,
        toolchain_root=toolchain,
        repository_root=repo,
        splat_render_runtime_root=runtime,
        output_root=tmp_path / "split-identity-bundle",
        expected_source_commit=diagnostic_commit,
        fresh_diagnostic_bootstrap=True,
    )

    assert receipt["source_commit"] == diagnostic_commit
    assert receipt["construction_source_commit"] == construction_commit
    assert receipt["toolchain_source_commit"] == diagnostic_commit
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        portable = json.loads(
            archive.read(
                "provider_runtime/input/portable_construction_envelope.v1.json"
            )
        )
    assert portable["expected_production_commit"] == construction_commit
    reopened = load_scene_configuration_provider_bundle_receipt(
        tmp_path
        / "split-identity-bundle"
        / f"{BUNDLE_SCHEMA_VERSION}.receipt.json",
        expected_source_commit=diagnostic_commit,
        diagnostic_only=True,
    )
    assert reopened["construction_source_commit"] == construction_commit
    preflight = vpa._blueprint_bundle_preflight(
        job_dir=tmp_path / "split-identity-preflight",
        generated_at="2026-08-27T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="task_evaluation_scene_configuration",
        bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://objects.example.test/split-identity.zip",
        provider_output_put_url="https://objects.example.test/output.zip",
    )
    assert preflight["blockers"] == []
    assert preflight["status"] == "passed"


def test_fresh_diagnostic_bootstrap_authorizes_uncarried_paid_stages() -> None:
    assert authority_module._required_external_stage_minima(
        diagnostic_only=True,
        diagnostic_bootstrap_mode="fresh",
        carried_stage_count=0,
    ) == {
        "artifixer_semantic_teacher": 2.4,
        "artifixer_visual_review": MIN_ARTIFIXER_VISUAL_REVIEW_SPEND_USD,
        "content_agents": 0.2,
    }
    assert authority_module._required_external_stage_minima(
        diagnostic_only=True,
        diagnostic_bootstrap_mode="checkpoint_resume",
        carried_stage_count=3,
    ) == {
        "artifixer_semantic_teacher": 0.0,
        "artifixer_visual_review": 0.0,
        "content_agents": 0.0,
    }
    assert authority_module._required_external_stage_minima(
        diagnostic_only=True,
        diagnostic_bootstrap_mode="checkpoint_resume",
        carried_stage_count=0,
        historical_terminal_evidence=True,
    ) == {
        "artifixer_semantic_teacher": 0.0,
        "artifixer_visual_review": 0.3,
        "content_agents": 0.2,
    }


def test_bundle_is_portable_deterministic_and_omits_raw_splat(tmp_path: Path) -> None:
    first = _build(tmp_path, "first")
    second = _build(tmp_path, "second")

    assert first["schema_version"] == BUNDLE_SCHEMA_VERSION
    assert first["bundle_sha256"] == second["bundle_sha256"]
    bundle = Path(first["bundle_path"])
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
        payloads = {name: archive.read(name) for name in names if not name.endswith("/")}
    assert all(
        b"RAW_INTERIORGS_BYTES_MUST_NEVER_LEAVE_CONTROL_PLANE" not in payload
        for payload in payloads.values()
    )
    assert not any(name.startswith("provider_runtime/renderer/") for name in names)
    assert "provider_renderer_required" not in first
    assert first["provider_python_runtime_required"] is True
    assert first["provider_python_runtime_python_version"] == "3.12"
    assert (
        "provider_runtime/"
        + first["provider_python_runtime_manifest"]
    ) in names
    assert (
        "provider_runtime/blueprint_pipeline/"
        "task_evaluation_scene_configuration_python_runtime.py"
    ) in names
    portable = json.loads(
        payloads["provider_runtime/input/portable_construction_envelope.v1.json"]
    )
    assert all(
        row["contract_path"] != "scene.appearance.representation"
        for row in portable["materialized_references"]
    )
    assert all("materialized_path" not in row for row in portable["materialized_references"])
    assert portable["render_inputs_result"]["derived_frames"][0]["path"].startswith(
        "input/render/"
    )
    assert portable["render_inputs_result"]["derived_gaussian_cutout"][
        "retained_scene_without_source_object"
    ]["path"].startswith("input/render/gaussians/")
    assert str(tmp_path).encode() not in payloads[
        "provider_runtime/input/portable_construction_envelope.v1.json"
    ]
    collision_member = "provider_runtime/input/references/0001.usd"
    assert collision_member in names
    assert payloads[collision_member] == b"#usda 1.0\n"


def test_bundle_refuses_collision_reference_without_declared_openusd_format(
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    envelope_root = tmp_path / "envelope"
    envelope_root.mkdir()
    envelope_path = _envelope(envelope_root, commit)
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    collision = next(
        row
        for row in envelope["materialized_references"]
        if row["contract_path"] == "scene.geometry.collision"
    )
    collision["uri"] = "s3://blueprint/task-evaluation/sage/collision.bin"
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    envelope_path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(
        TaskEvaluationSceneConfigurationBundleError,
        match="scene_configuration_bundle_collision_reference_format_invalid",
    ):
        build_scene_configuration_provider_bundle(
            construction_envelope_path=envelope_path,
            toolchain_root=_toolchain(tmp_path / "toolchain", commit),
            repository_root=_repo(tmp_path / "repo"),
            output_root=tmp_path / "bundle",
            expected_source_commit=commit,
        )


def test_live_run_preflights_queue_finalization_before_any_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build(tmp_path, "bundle")
    receipt_path = tmp_path / "bundle" / f"{BUNDLE_SCHEMA_VERSION}.receipt.json"
    authority_path = tmp_path / "authority.json"
    authority_path.write_text("{}", encoding="utf-8")
    authority = {
        "authority_digest": "sha256:" + "e" * 64,
        "hard_attempt_spend_cap_usd": MAX_ATTEMPT_SPEND_USD,
        "provider_compute_spend_cap_usd": MAX_PROVIDER_COMPUTE_SPEND_USD,
        "maximum_hourly_rate_usd": MAX_HOURLY_RATE_USD,
        "maximum_single_resource_ttl_seconds": REQUIRED_PARENT_TTL_SECONDS,
        "external_service_spend_caps": {
            "openai": {"maximum_cost_usd": 1.5, "maximum_requests": 32}
        },
    }
    monkeypatch.setattr(
        scene_vast,
        "validate_scene_configuration_paid_authority",
        lambda _value, **_kwargs: authority,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("mutation reached before queue finalization preflight")

    monkeypatch.setattr(scene_vast, "_provider_runtime_inputs", forbidden)
    monkeypatch.setattr(
        scene_vast, "stage_wam_provider_bundle_object_store", forbidden
    )
    monkeypatch.setattr(
        scene_vast,
        "preflight_scene_construction_finalization",
        lambda **_kwargs: (_ for _ in ()).throw(
            TaskEvaluationSceneConstructionQueueError(
                "scene_construction_queue_finalization_destination_unwritable"
            )
        ),
    )

    result = scene_vast.run_scene_configuration_vast(
        job_dir=tmp_path / "job",
        bundle_receipt_path=receipt_path,
        paid_attempt_authority_path=authority_path,
        paid_resource_admission_grant=object(),
        execute=True,
        scene_construction_queue_root=_construction_queue(tmp_path),
    )

    assert result["status"] == "blocked"
    assert result["provider_mutations_performed"] == 0
    assert result["continuing_spend_from_this_run"] is False
    assert len(result["blockers"]) == 1
    assert result["blockers"][0].startswith(
        "scene_construction_queue_finalization_preflight_failed:"
    )
    assert result["blockers"][0].endswith(
        "scene_construction_queue_finalization_destination_unwritable"
    )
    assert result["scene_construction_queue_finalization"]["queue_state"] == (
        "blocked"
    )


def test_preallocation_refusal_seals_canonical_terminal_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build(tmp_path, "bundle")
    receipt_path = tmp_path / "bundle" / f"{BUNDLE_SCHEMA_VERSION}.receipt.json"
    authority_path = tmp_path / "authority.json"
    authority_path.write_text("{}", encoding="utf-8")
    authority = {
        "authority_digest": "sha256:" + "e" * 64,
        "hard_attempt_spend_cap_usd": MAX_ATTEMPT_SPEND_USD,
        "provider_compute_spend_cap_usd": MAX_PROVIDER_COMPUTE_SPEND_USD,
        "maximum_hourly_rate_usd": MAX_HOURLY_RATE_USD,
        "maximum_single_resource_ttl_seconds": REQUIRED_PARENT_TTL_SECONDS,
        "external_service_spend_caps": {
            "openai": {"maximum_cost_usd": 1.5, "maximum_requests": 32}
        },
    }
    monkeypatch.setattr(
        scene_vast,
        "validate_scene_configuration_paid_authority",
        lambda _value, **_kwargs: authority,
    )
    monkeypatch.setattr(
        scene_vast, "_provider_runtime_inputs", lambda _authority: ({}, {})
    )
    monkeypatch.setattr(
        scene_vast,
        "require_paid_resource_admission_grant",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        scene_vast,
        "stage_wam_provider_bundle_object_store",
        lambda **_kwargs: {
            "status": "blocked",
            "blockers": ["fixture_object_store_staging_refusal"],
        },
    )
    monkeypatch.setattr(
        scene_vast,
        "cleanup_staged_wam_provider_objects",
        lambda _root: {"status": "completed", "all_objects_absent": True},
    )
    job = tmp_path / "job"

    result = scene_vast.run_scene_configuration_vast(
        job_dir=job,
        bundle_receipt_path=receipt_path,
        paid_attempt_authority_path=authority_path,
        paid_resource_admission_grant=object(),
        execute=True,
        scene_construction_queue_root=_construction_queue(tmp_path),
        disk_usage_provider=lambda _path: type(
            "Usage", (), {"free": 100_000_000_000}
        )(),
    )

    result_path = job / f"{scene_vast.RESULT_SCHEMA_VERSION}.json"
    assert result_path.is_file()
    assert json.loads(result_path.read_text(encoding="utf-8")) == result
    assert result["blockers"] == ["fixture_object_store_staging_refusal"]
    assert result["continuing_spend_from_this_run"] is False
    assert result["object_store_cleanup"]["all_objects_absent"] is True
    assert result["scene_construction_queue_finalization"]["queue_state"] == "blocked"
    teardown_path = Path(result["teardown_manifest_path"])
    artifact_manifest_path = Path(result["artifact_manifest_path"])
    teardown = json.loads(teardown_path.read_text(encoding="utf-8"))
    artifact_manifest = json.loads(
        artifact_manifest_path.read_text(encoding="utf-8")
    )
    assert teardown["status"] == "not_required_provider_adapter_never_invoked"
    assert teardown["vast_instance_ids"] == []
    assert teardown["teardown_actions_performed"] == []
    assert teardown["continuing_spend_from_this_run"] is False
    assert artifact_manifest["status"] == "completed"
    assert artifact_manifest["binding"]["allocator_lane"] == (
        scene_vast.PROVIDER_BUNDLE_KIND
    )
    assert artifact_manifest["binding"]["source_commit"] == result["source_commit"]
    assert "teardown_manifest" in artifact_manifest["required_roles"]
    assert "teardown_manifest" in artifact_manifest["observed_roles"]
    assert not list((tmp_path / "construction-queue" / "pending").glob("*.json"))
    assert len(list((tmp_path / "construction-queue" / "blocked").glob("*.json"))) == 1
    assert result["result_digest"] == canonical_digest(
        result, digest_field="result_digest"
    )


def test_insufficient_parent_runtime_authority_refuses_before_any_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The serialized stage chain must fit before staging or allocation."""

    _build(tmp_path, "bundle")
    receipt_path = tmp_path / "bundle" / f"{BUNDLE_SCHEMA_VERSION}.receipt.json"
    authority_path = tmp_path / "authority.json"
    authority_path.write_text("{}", encoding="utf-8")
    authority = {
        "authority_digest": "sha256:" + "e" * 64,
        "provider_compute_spend_cap_usd": MAX_PROVIDER_COMPUTE_SPEND_USD,
        "maximum_hourly_rate_usd": MAX_HOURLY_RATE_USD,
        "maximum_single_resource_ttl_seconds": 9_000,
    }
    monkeypatch.setattr(
        scene_vast,
        "validate_scene_configuration_paid_authority",
        lambda _value, **_kwargs: authority,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("mutation boundary reached with insufficient authority")

    monkeypatch.setattr(scene_vast, "_provider_runtime_inputs", forbidden)
    monkeypatch.setattr(
        scene_vast, "stage_wam_provider_bundle_object_store", forbidden
    )
    monkeypatch.setattr(scene_vast, "arm_independent_vast_watchdog", forbidden)
    monkeypatch.setattr(scene_vast, "run_vast_provider_adapter", forbidden)

    result = scene_vast.run_scene_configuration_vast(
        job_dir=tmp_path / "job",
        bundle_receipt_path=receipt_path,
        paid_attempt_authority_path=authority_path,
        paid_resource_admission_grant=object(),
        execute=True,
        scene_construction_queue_root=_construction_queue(tmp_path),
    )

    assert result["status"] == "blocked"
    assert result["provider_mutations_performed"] == 0
    assert result["continuing_spend_from_this_run"] is False
    assert result["blockers"] == [
        "scene_configuration_parent_runtime_budget_insufficient:25200:9000"
    ]


def test_live_terminal_result_refuses_missing_construction_queue_root(
    tmp_path: Path,
) -> None:
    result = scene_vast._seal_live_terminal_result(
        tmp_path,
        {
            "schema_version": scene_vast.RESULT_SCHEMA_VERSION,
            "status": "completed",
            "run_id": "scene-run-1",
            "source_commit": "a" * 40,
            "configuration_completed": True,
            "configured_scene_published": True,
            "full_byte_service_account_readback_passed": True,
            "continuing_spend_from_this_run": False,
            "blockers": [],
        },
        receipt={},
        scene_construction_queue_root=None,
    )

    assert result["status"] == "blocked"
    assert result["scene_construction_queue_finalization"][
        "finalization_performed"
    ] is False
    assert any(
        "scene_construction_queue_finalization_root_missing" in blocker
        for blocker in result["blockers"]
    )


def test_bundle_refuses_old_toolchain_without_provider_python_runtime(
    tmp_path: Path,
) -> None:
    """A valid historical toolchain is not sufficient for the paid path."""

    commit = "a" * 40
    packages = _component_packages(tmp_path / "packages")
    artifixer = packages["artifixer3d_observed_object_removal"]
    wheelhouse = artifixer / "python_wheelhouse"
    for directory in (wheelhouse / "wheels", wheelhouse, artifixer):
        directory.chmod(0o755)
    shutil.rmtree(wheelhouse)
    package_manifest_path = (
        artifixer
        / "task_evaluation_scene_configuration_component_package.v1.json"
    )
    package_manifest_path.chmod(0o644)
    package_manifest = json.loads(package_manifest_path.read_text(encoding="utf-8"))
    package_manifest["files"] = [
        row
        for row in package_manifest["files"]
        if not row["relative_path"].startswith("python_wheelhouse/")
    ]
    package_manifest["package_digest"] = canonical_digest(
        package_manifest, digest_field="package_digest"
    )
    package_manifest_path.write_text(json.dumps(package_manifest), encoding="utf-8")
    package_manifest_path.chmod(0o444)
    artifixer.chmod(0o555)
    toolchain = tmp_path / "old-toolchain"
    build_published_scene_configuration_toolchain(
        source_commit=commit,
        output_root=toolchain,
        readback=lambda path: path.read_bytes(),
        readback_actor="service-account:test",
        component_packages=packages,
    )
    source = tmp_path / "old-toolchain-source"
    source.mkdir()

    with pytest.raises(
        TaskEvaluationSceneConfigurationBundleError,
        match="scene_configuration_bundle_provider_python_runtime_invalid",
    ):
        build_scene_configuration_provider_bundle(
            construction_envelope_path=_envelope(source, commit),
            toolchain_root=toolchain,
            repository_root=_repo(tmp_path / "old-toolchain-repo"),
            output_root=tmp_path / "old-toolchain-bundle",
            expected_source_commit=commit,
        )


def test_provider_runner_hydrates_only_digest_bound_runtime_paths(tmp_path: Path) -> None:
    receipt = _build(tmp_path, "bundle")
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        archive.extractall(extracted)
    runner_path = extracted / "provider_runtime/task_evaluation_scene_configuration_provider_runner.py"
    spec = importlib.util.spec_from_file_location("scene_configuration_runner", runner_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = extracted / "provider_runtime"
    portable = json.loads(
        (runtime / "input/portable_construction_envelope.v1.json").read_text(
            encoding="utf-8"
        )
    )
    hydrated = module._hydrate_envelope(runtime.resolve(), portable)
    assert hydrated["portable_envelope_digest"] == portable["envelope_digest"]
    assert hydrated["envelope_digest"] == canonical_digest(
        hydrated, digest_field="envelope_digest"
    )
    assert all(
        Path(row["materialized_path"]).is_file()
        for row in hydrated["materialized_references"]
    )
    assert all(
        Path(row["path"]).is_file()
        for row in hydrated["render_inputs_result"]["derived_frames"]
    )
    assert all(
        Path(row["source_object_mask"]["path"]).is_file()
        for row in hydrated["render_inputs_result"]["derived_frames"]
    )
    assert Path(
        hydrated["render_inputs_result"]["derived_gaussian_cutout"][
            "retained_scene_without_source_object"
        ]["path"]
    ).is_file()
    assert all(
        Path(row["materialized_path"]).is_file()
        for row in hydrated["stage_configuration_references"]
    )

    frame = runtime / portable["render_inputs_result"]["derived_frames"][0]["path"]
    frame.chmod(0o644)
    frame.write_bytes(b"tampered")
    try:
        module._hydrate_envelope(runtime.resolve(), portable)
    except ValueError as exc:
        assert str(exc) == "scene_configuration_provider_bound_file_invalid"
    else:
        raise AssertionError("tampered provider input was accepted")


def test_provider_runner_accepts_the_owed_provider_render_manifest(
    tmp_path: Path,
) -> None:
    """A provider-render packet has no manifest until stage one renders it."""

    receipt = _build(tmp_path, "provider-render-bundle")
    extracted = tmp_path / "provider-render-extracted"
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        archive.extractall(extracted)
    runner_path = (
        extracted
        / "provider_runtime/task_evaluation_scene_configuration_provider_runner.py"
    )
    spec = importlib.util.spec_from_file_location(
        "scene_configuration_provider_render_runner", runner_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = extracted / "provider_runtime"
    portable_path = runtime / "input/portable_construction_envelope.v1.json"
    portable = json.loads(portable_path.read_text(encoding="utf-8"))
    render = portable["render_inputs_result"]
    render["status"] = "derived_method_inputs_pending_provider_render"
    render["render_manifest"] = None
    render["derived_frames"] = []
    render["derived_frame_count"] = 0
    render["result_digest"] = canonical_digest(
        render, digest_field="result_digest"
    )
    portable["envelope_digest"] = canonical_digest(
        portable, digest_field="envelope_digest"
    )

    hydrated = module._hydrate_envelope(runtime.resolve(), portable)

    assert hydrated["render_inputs_result"]["render_manifest"] is None
    assert hydrated["render_inputs_result"]["derived_frames"] == []
    assert Path(
        hydrated["render_inputs_result"]["camera_calibration"]["path"]
    ).is_file()


def test_vast_preflight_and_onstart_accept_only_the_sealed_scene_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _build_provider_render_bundle(tmp_path, "bundle", monkeypatch)
    job = tmp_path / "job"
    job.mkdir()
    preflight = vpa._blueprint_bundle_preflight(
        job_dir=job,
        generated_at="2026-08-25T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="task_evaluation_scene_configuration",
        bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://objects.example.test/input.zip",
        provider_output_put_url="https://objects.example.test/output.zip",
    )

    assert preflight["blockers"] == []
    assert preflight["status"] == "passed"
    assert preflight["provider_bundle_readiness_source"] == "immutable_bundle_member"
    script = vpa._probe_shell_script(
        "https://heartbeat.example.test",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        provider_bundle_kind="task_evaluation_scene_configuration",
    )
    assert "run_task_evaluation_scene_configuration_provider.sh" in script
    assert "task_evaluation_scene_configuration_provider_output.zip" in script
    assert "BLUEPRINT_SCENE_CONFIGURATION_RUNTIME_ROOT" in script
    assert "scene_configuration_runtime_toolchain_missing" in script
    assert "BLUEPRINT_VAST_SCENE_CONFIGURATION_WARM_RUNTIME_NOT_READY" in script
    assert (
        "BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:"
        "scene_configuration_warm_runtime_not_ready"
    ) not in script
    assert "timeout 300 apt-get" in script
    assert script.index(
        f" install -y {vpa.SCENE_CONFIGURATION_APT_PACKAGES}"
    ) < script.index("BLUEPRINT_VAST_PROVIDER_BUNDLE_DOWNLOADED")
    assert (
        f"for required_command in {vpa.SCENE_CONFIGURATION_REQUIRED_COMMANDS}; "
        'do command -v "$required_command"'
    ) in script
    assert 'exec /isaac-sim/python.sh "$@"' in script
    assert (
        'export PATH="/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:$PATH"'
        in script
    )
    for excluded in (
        ".artifixer-venv",
        ".hf_home",
        ".venv",
        ".ovrtx_venv",
        ".ovrtx_native_venv",
        ".ovphysx_venv",
        ".git",
        "__pycache__",
        "artifixer_bundle",
        "artifixer_execution",
        "artifixer_output",
        "content_agents_source",
    ):
        assert excluded in script
    assert "excluded_parts.isdisjoint(relative.parts)" in script
    assert "provider_output_zip_exclusions.json" in script
    subprocess.run(["bash", "-n", "-c", script], check=True)

    runtime_output = tmp_path / "runtime-output"
    work_dir = tmp_path / "provider-work"
    (runtime_output / "stages/stage-1/producer").mkdir(parents=True)
    kept = runtime_output / "stages/stage-1/producer/appearance_removal_receipt.v1.json"
    kept.write_text('{"status":"completed"}\n', encoding="utf-8")
    excluded = (
        runtime_output
        / "stages/stage-1/producer/released_artifixer_runtime/artifixer_execution/model.bin"
    )
    excluded.parent.mkdir(parents=True)
    excluded.write_bytes(b"reproducible-scratch")
    monkeypatch.setenv("BLUEPRINT_SCENE_CONFIGURATION_OUTPUT_ROOT", str(runtime_output))
    monkeypatch.setenv("BLUEPRINT_VAST_WORK_DIR", str(work_dir))
    work_dir.mkdir()
    archive_marker = "$RUNTIME_PY - <<'PY'\n"
    archive_start = script.index(
        archive_marker, script.index("BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_EXIT_CODE")
    ) + len(archive_marker)
    archive_end = script.index("\nPY\n", archive_start)
    archive_program = script[archive_start:archive_end]
    exec(compile(archive_program, "<scene-configuration-output-archive>", "exec"), {})

    with zipfile.ZipFile(
        work_dir / "task_evaluation_scene_configuration_provider_output.zip"
    ) as archive:
        archived = set(archive.namelist())
        assert kept.relative_to(runtime_output).as_posix() in archived
        assert excluded.relative_to(runtime_output).as_posix() not in archived
        exclusions = json.loads(archive.read("provider_output_zip_exclusions.json"))
    assert exclusions["schema_version"] == (
        "task_evaluation_scene_configuration_provider_output_zip_exclusions.v1"
    )
    assert "artifixer_execution" in exclusions["excluded_directory_names"]

    outside = tmp_path / "outside-provider-output.txt"
    outside.write_text("must-not-enter-provider-output\n", encoding="utf-8")
    leaked = runtime_output / "stages/stage-6/adapter/outside-link.txt"
    leaked.parent.mkdir(parents=True)
    leaked.symlink_to(outside)
    with pytest.raises(
        RuntimeError,
        match=(
            "scene_configuration_provider_output_symlink_forbidden:"
            "stages/stage-6/adapter/outside-link.txt"
        ),
    ):
        exec(
            compile(
                archive_program,
                "<scene-configuration-output-archive-symlink>",
                "exec",
            ),
            {},
        )


def test_scene_configuration_authority_binds_fresh_zero_and_project_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _build(tmp_path, "bundle")
    receipt_path = tmp_path / "bundle" / f"{BUNDLE_SCHEMA_VERSION}.receipt.json"
    assert load_scene_configuration_provider_bundle_receipt(receipt_path) == receipt
    project_path = tmp_path / "project-spend.json"
    project_path.write_text('{"project":"sealed"}\n', encoding="utf-8")
    zero_path = tmp_path / "provider-zero.json"
    zero_path.write_text('{"zero":"sealed"}\n', encoding="utf-8")

    def record(path: Path) -> dict[str, object]:
        return {
            "path": str(path),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }

    monkeypatch.setattr(
        authority_module,
        "validate_project_spend_reconciliation",
        lambda path, **_kwargs: (
            {"total_cost_usd": 500.0},
            record(Path(path).resolve()),
        ),
    )
    monkeypatch.setattr(
        live_profile_module,
        "validate_project_spend_reconciliation",
        lambda path, **_kwargs: (
            {"total_cost_usd": 500.0},
            record(Path(path).resolve()),
        ),
    )
    monkeypatch.setattr(
        live_profile_module,
        "project_spend_dependency_records",
        lambda _value: [],
    )
    monkeypatch.setattr(
        authority_module,
        "_provider_zero",
        lambda _path: {
            "observed_at_utc": "2026-08-25T12:00:00Z",
            "provider_zero_digest": "sha256:" + "c" * 64,
        },
    )
    authority_path = tmp_path / "authority.json"
    authority_arguments = dict(
        bundle_receipt_path=receipt_path,
        project_spend_reconciliation_path=project_path,
        initial_provider_zero_path=zero_path,
        authorization_reference="user-authorized-new-scene-gpu",
        authorized_by="project-owner",
        authorized_on="2026-08-25T12:05:00Z",
        source_commit="a" * 40,
        container_image=authority_module.SCENE_CONFIGURATION_PROVIDER_IMAGE,
        resource_name=(
            "adp-new-scene-simple-relocation-839873-aaaaaaaaaaaa-20260825t120500z"
        ),
        max_hourly_rate_usd=0.50,
        hard_cap_usd=MAX_ATTEMPT_SPEND_USD,
        hard_ttl_seconds=REQUIRED_PARENT_TTL_SECONDS,
        output_path=authority_path,
        provider_compute_spend_cap_usd=MAX_PROVIDER_COMPUTE_SPEND_USD,
        openai_max_cost_usd=MIN_EXTERNAL_SERVICE_SPEND_USD,
        openai_max_requests=32,
        openai_artifixer_semantic_teacher_max_cost_usd=2.4,
        openai_artifixer_visual_review_max_cost_usd=(
            MIN_ARTIFIXER_VISUAL_REVIEW_SPEND_USD
        ),
        openai_content_agents_max_cost_usd=0.2,
    )
    authority = authority_module.materialize_scene_configuration_paid_authority(
        **authority_arguments
    )
    with pytest.raises(
        authority_module.TaskEvaluationSceneConfigurationAuthorityError,
        match="scene_configuration_authority_container_image_invalid",
    ):
        authority_module.materialize_scene_configuration_paid_authority(
            **{
                **authority_arguments,
                "container_image": "nvcr.io/nvidia/isaac-sim@sha256:" + "b" * 64,
                "output_path": tmp_path / "authority-wrong-image.json",
            }
        )

    for suffix, overrides in (
        (
            "semantic-underfunded",
            {
                "openai_artifixer_semantic_teacher_max_cost_usd": 1.0,
                "openai_artifixer_visual_review_max_cost_usd": 0.95,
                "openai_content_agents_max_cost_usd": 0.95,
            },
        ),
        (
            "aggregate-underfunded",
            {
                "openai_max_cost_usd": 2.8,
                "openai_artifixer_visual_review_max_cost_usd": 0.2,
                "openai_content_agents_max_cost_usd": 0.2,
            },
        ),
        (
            "visual-review-underfunded",
            {
                "openai_artifixer_visual_review_max_cost_usd": 0.31,
                "openai_content_agents_max_cost_usd": 0.21,
            },
        ),
        (
            "content-agents-underfunded",
            {
                "openai_artifixer_visual_review_max_cost_usd": 0.33,
                "openai_content_agents_max_cost_usd": 0.19,
            },
        ),
        (
            "attempt-cap-not-canonical",
            {
                "hard_cap_usd": 9.9,
                "provider_compute_spend_cap_usd": 5.9,
            },
        ),
        (
            "compute-cap-not-canonical",
            {"provider_compute_spend_cap_usd": 5.9},
        ),
    ):
        with pytest.raises(
            authority_module.TaskEvaluationSceneConfigurationAuthorityError,
            match="scene_configuration_authority_configuration_invalid",
        ):
            authority_module.materialize_scene_configuration_paid_authority(
                **{
                    **authority_arguments,
                    **overrides,
                    "output_path": tmp_path / f"authority-{suffix}.json",
                }
            )

    assert authority["retry_cap"] == 0
    # Prior spend remains reconciled and receipt-bound, but it no longer
    # creates an unrelated lifetime ceiling for an explicitly authorized,
    # tightly bounded one-shot attempt.
    assert authority["aggregate_goal_spend_before_attempt_usd"] == 500.0
    assert "aggregate_goal_spend_cap_usd" not in authority
    assert authority["hard_attempt_spend_cap_usd"] == MAX_ATTEMPT_SPEND_USD
    assert (
        authority["provider_compute_spend_cap_usd"]
        == MAX_PROVIDER_COMPUTE_SPEND_USD
    )
    assert authority["maximum_provider_allocations"] == 1
    assert authority["maximum_automatic_retries"] == 0
    assert authority["external_service_spend_caps"]["openai"][
        "maximum_cost_usd"
    ] == MIN_EXTERNAL_SERVICE_SPEND_USD
    assert authority_module.validate_scene_configuration_paid_authority(
        authority, bundle_receipt=receipt
    ) == authority
    wrong_image = {
        **authority,
        "container_image": "nvcr.io/nvidia/isaac-sim@sha256:" + "b" * 64,
        "authority_digest": "",
    }
    wrong_image["authority_digest"] = canonical_digest(
        wrong_image, digest_field="authority_digest"
    )
    with pytest.raises(
        authority_module.TaskEvaluationSceneConfigurationAuthorityError,
        match="authority_contract_invalid",
    ):
        authority_module.validate_scene_configuration_paid_authority(
            wrong_image, bundle_receipt=receipt
        )
    for aggregate_cost, stage_overrides in (
        (
            2.8,
            {
                "artifixer_semantic_teacher": 2.4,
                "artifixer_visual_review": 0.2,
                "content_agents": 0.2,
            },
        ),
        (
            2.9,
            {
                "artifixer_semantic_teacher": 1.0,
                "artifixer_visual_review": 0.95,
                "content_agents": 0.95,
            },
        ),
        (
            2.9,
            {
                "artifixer_semantic_teacher": 2.4,
                "artifixer_visual_review": 0.29,
                "content_agents": 0.21,
            },
        ),
        (
            2.9,
            {
                "artifixer_semantic_teacher": 2.4,
                "artifixer_visual_review": 0.31,
                "content_agents": 0.19,
            },
        ),
    ):
        underfunded = json.loads(json.dumps(authority))
        openai = underfunded["external_service_spend_caps"]["openai"]
        openai["maximum_cost_usd"] = aggregate_cost
        openai["stage_max_cost_usd"] = stage_overrides
        underfunded["authority_digest"] = canonical_digest(
            underfunded, digest_field="authority_digest"
        )
        with pytest.raises(
            authority_module.TaskEvaluationSceneConfigurationAuthorityError,
            match="authority_contract_invalid",
        ):
            authority_module.validate_scene_configuration_paid_authority(
                underfunded, bundle_receipt=receipt
            )
    for field, value in (
        ("hard_attempt_spend_cap_usd", 9.9),
        ("provider_compute_spend_cap_usd", 5.9),
    ):
        noncanonical = json.loads(json.dumps(authority))
        noncanonical[field] = value
        noncanonical["authority_digest"] = canonical_digest(
            noncanonical, digest_field="authority_digest"
        )
        with pytest.raises(
            authority_module.TaskEvaluationSceneConfigurationAuthorityError,
            match="authority_contract_invalid",
        ):
            authority_module.validate_scene_configuration_paid_authority(
                noncanonical, bundle_receipt=receipt
            )
    provider_receipt = {
        **receipt,
        "raw_interiorgs_bytes_in_provider_bundle": True,
        "disclosure_decision": _decision(provider=True),
    }
    provider_authority = {
        **authority,
        "raw_interiorgs_bytes_authorized_for_provider": True,
        "provider_disclosure_decision_digest": provider_receipt[
            "disclosure_decision"
        ]["decision_digest"],
        "authority_digest": "",
    }
    provider_authority["authority_digest"] = canonical_digest(
        provider_authority, digest_field="authority_digest"
    )
    assert authority_module.validate_scene_configuration_paid_authority(
        provider_authority, bundle_receipt=provider_receipt
    ) == provider_authority
    falsely_denied = {
        **provider_authority,
        "raw_interiorgs_bytes_authorized_for_provider": False,
        "authority_digest": "",
    }
    falsely_denied["authority_digest"] = canonical_digest(
        falsely_denied, digest_field="authority_digest"
    )
    with pytest.raises(
        authority_module.TaskEvaluationSceneConfigurationAuthorityError,
        match="authority_contract_invalid",
    ):
        authority_module.validate_scene_configuration_paid_authority(
            falsely_denied, bundle_receipt=provider_receipt
        )
    for name, value in (
        ("OPENAI_ADMIN_API_KEY_FILE", "test-admin-key"),
        ("OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_FILE", "key-semantic"),
        ("OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE", "key-review"),
        ("OPENAI_CONTENT_AGENTS_API_KEY_FILE", "key-content-agents"),
        (
            "BLUEPRINT_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_COST_SCOPE_ATTESTATION_FILE",
            _openai_scope_attestation(
                paid_resource_class=(
                    "task_evaluation_scene_configuration_artifixer_semantic_teacher"
                ),
                api_key_id="key_semantic",
            ),
        ),
        (
            "BLUEPRINT_OPENAI_ARTIFIXER_VISUAL_REVIEW_COST_SCOPE_ATTESTATION_FILE",
            _openai_scope_attestation(
                paid_resource_class=(
                    "task_evaluation_scene_configuration_artifixer_visual_review"
                ),
                api_key_id="key_review",
            ),
        ),
        (
            "BLUEPRINT_OPENAI_CONTENT_AGENTS_COST_SCOPE_ATTESTATION_FILE",
            _openai_scope_attestation(
                paid_resource_class="task_evaluation_scene_configuration_content_agents",
                api_key_id="key_content_agents",
            ),
        ),
    ):
        path = tmp_path / name.lower()
        path.write_text(value, encoding="utf-8")
        path.chmod(0o640)
        monkeypatch.setenv(name, str(path))
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj_test")
    monkeypatch.setenv(
        "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_ID", "key_semantic"
    )
    monkeypatch.setenv("OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_ID", "key_review")
    monkeypatch.setenv("OPENAI_CONTENT_AGENTS_API_KEY_ID", "key_content_agents")
    monkeypatch.setattr(
        scene_vast,
        "_collect_openai_cost_snapshot",
        lambda **_kwargs: {"total_cost_usd": 0.0},
    )
    dry_run = scene_vast.run_scene_configuration_vast(
        job_dir=tmp_path / "dry-run",
        bundle_receipt_path=receipt_path,
        paid_attempt_authority_path=authority_path,
        paid_resource_admission_grant=None,
        execute=False,
    )
    assert dry_run["status"] == "dry_run_ready"
    assert dry_run["provider_mutations_performed"] == 0

    source_digest = file_digest(receipt_path)
    identity = source_digest.removeprefix("sha256:")
    manifest_publication = {
        "schema_version": "task_evaluation_immutable_manifest_publication.v1",
        "status": "published",
        "source": {
            "path": str(receipt_path.resolve()),
            "size_bytes": receipt_path.stat().st_size,
            "sha256": source_digest,
        },
        "profile_builder": (
            "build_task_evaluation_scene_configuration_live_profile.py"
        ),
        "publication_seam": "content_addressed_full_readback",
        "published_uri": (
            f"gs://fixture/sha256/{identity[:2]}/{identity}.json"
        ),
        "storage_scheme": "gs",
        "remote_size_bytes": receipt_path.stat().st_size,
        "remote_sha256": source_digest,
        "provider_full_byte_readback_verified": True,
        "content_addressed_key": True,
        "exclusive_create": True,
        "upload_receipt_digest": "sha256:" + "d" * 64,
        "provider_compute_mutation_performed": False,
        "paid_resource_allocated": False,
        "raw_secret_values_recorded": False,
        "blockers": [],
        "receipt_digest": "",
    }
    manifest_publication["receipt_digest"] = canonical_digest(
        manifest_publication, digest_field="receipt_digest"
    )
    publication_path = tmp_path / "manifest-publication.json"
    publication_path.write_text(
        json.dumps(manifest_publication), encoding="utf-8"
    )
    profile = build_scene_configuration_live_profile(
        bundle_receipt_path=receipt_path,
        attempt_authority_path=authority_path,
        source_commit="a" * 40,
        raw_manifest_uri=str(publication_path),
        revision="r1",
        max_hourly_rate_usd=0.50,
        hard_ttl_seconds=REQUIRED_PARENT_TTL_SECONDS,
        max_spend_usd=MAX_ATTEMPT_SPEND_USD,
        team_namespace="team-a",
        scene_id="interiorgs-839873",
        task_id="planar-mug-push",
    )
    assert profile["task_evaluation_run"]["run_mode"] == (
        "scene_configuration"
    )
    assert profile["task_evaluation_run"]["evaluation_episode_executed"] is False
    allocator_argv = profile["allocator"]["argv"]
    bundle_index = allocator_argv.index(
        "--scene-configuration-bundle-receipt"
    )
    assert allocator_argv[bundle_index + 1] == str(receipt_path.resolve())
    assert "execution_result_path" in profile["terminal_contract"][
        "required_path_fields"
    ]
    assert "configured_scene_revision_path" in profile["terminal_contract"][
        "required_path_fields"
    ]
    assert "publication_result_path" in profile["terminal_contract"][
        "required_path_fields"
    ]
    pod_index = allocator_argv.index("--pod-name")
    assert allocator_argv[pod_index + 1] == profile["profile_id"]

    # The allocator refuses --pod-name != authority.resource_name, and the
    # authority binds the activation id, so the builder must let the launch
    # graph pass that exact name through.
    bound = build_scene_configuration_live_profile(
        bundle_receipt_path=receipt_path,
        attempt_authority_path=authority_path,
        source_commit="a" * 40,
        raw_manifest_uri=str(publication_path),
        revision="r1",
        max_hourly_rate_usd=0.50,
        hard_ttl_seconds=REQUIRED_PARENT_TTL_SECONDS,
        max_spend_usd=MAX_ATTEMPT_SPEND_USD,
        team_namespace="team-a",
        scene_id="interiorgs-839873",
        task_id="planar-mug-push",
        pod_name=authority["resource_name"],
    )
    bound_argv = bound["allocator"]["argv"]
    assert (
        bound_argv[bound_argv.index("--pod-name") + 1]
        == authority["resource_name"]
    )
    binding_digest = hashlib.sha256(
        authority["resource_name"].encode("utf-8")
    ).hexdigest()[:12]
    assert bound["profile_id"].endswith(f"-r1-binding-{binding_digest}")

    # A retry may use the same source commit and human revision, but its paid
    # authority names a new activation-scoped provider resource. The immutable
    # profile therefore needs a distinct identity instead of conflicting with
    # the first activation's already-published bytes.
    retry_authority = dict(authority)
    retry_authority["resource_name"] = f"{authority['resource_name']}-retry"
    retry_authority["authority_digest"] = canonical_digest(
        retry_authority, digest_field="authority_digest"
    )
    retry_authority_path = tmp_path / "authority-retry.json"
    retry_authority_path.write_text(
        json.dumps(retry_authority), encoding="utf-8"
    )
    retry_bound = build_scene_configuration_live_profile(
        bundle_receipt_path=receipt_path,
        attempt_authority_path=retry_authority_path,
        source_commit="a" * 40,
        raw_manifest_uri=str(publication_path),
        revision="r1",
        max_hourly_rate_usd=0.50,
        hard_ttl_seconds=REQUIRED_PARENT_TTL_SECONDS,
        max_spend_usd=MAX_ATTEMPT_SPEND_USD,
        team_namespace="team-a",
        scene_id="interiorgs-839873",
        task_id="planar-mug-push",
        pod_name=retry_authority["resource_name"],
    )
    assert retry_bound["profile_id"] != bound["profile_id"]

    tampered = dict(authority)
    tampered["maximum_hourly_rate_usd"] = 0.81
    tampered["authority_digest"] = canonical_digest(
        tampered, digest_field="authority_digest"
    )
    with pytest.raises(
        authority_module.TaskEvaluationSceneConfigurationAuthorityError,
        match="authority_contract_invalid",
    ):
        authority_module.validate_scene_configuration_paid_authority(
            tampered, bundle_receipt=receipt
        )


def test_scene_configuration_openai_runtime_files_fail_closed_on_unsafe_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = {
        "authority_digest": "sha256:" + "a" * 64,
        "external_service_spend_caps": {
            "openai": {
                "maximum_cost_usd": 1.0,
                "maximum_requests": 3,
                "stage_max_cost_usd": {
                    "artifixer_semantic_teacher": 0.2,
                    "artifixer_visual_review": 0.5,
                    "content_agents": 0.3,
                },
            }
        },
    }
    paths = []
    for name in (
        "OPENAI_ADMIN_API_KEY_FILE",
        "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_FILE",
        "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE",
        "OPENAI_CONTENT_AGENTS_API_KEY_FILE",
        "BLUEPRINT_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_COST_SCOPE_ATTESTATION_FILE",
        "BLUEPRINT_OPENAI_ARTIFIXER_VISUAL_REVIEW_COST_SCOPE_ATTESTATION_FILE",
        "BLUEPRINT_OPENAI_CONTENT_AGENTS_COST_SCOPE_ATTESTATION_FILE",
    ):
        path = tmp_path / name.lower()
        path.write_text("private-value-" + name.lower(), encoding="utf-8")
        path.chmod(0o640)
        monkeypatch.setenv(name, str(path))
        paths.append(path)
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj_test")
    monkeypatch.setenv(
        "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_ID", "key_semantic"
    )
    monkeypatch.setenv("OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_ID", "key_review")
    monkeypatch.setenv("OPENAI_CONTENT_AGENTS_API_KEY_ID", "key_content_agents")

    paths[0].chmod(0o644)
    with pytest.raises(
        scene_vast.TaskEvaluationSceneConfigurationVastError,
        match="scene_configuration_openai_runtime_secret_configuration_invalid",
    ):
        scene_vast._provider_runtime_inputs(authority)

    paths[0].chmod(0o640)
    symlink = tmp_path / "openai-key-link"
    symlink.symlink_to(paths[0])
    monkeypatch.setenv("OPENAI_CONTENT_AGENTS_API_KEY_FILE", str(symlink))
    with pytest.raises(
        scene_vast.TaskEvaluationSceneConfigurationVastError,
        match="scene_configuration_openai_runtime_secret_configuration_invalid",
    ):
        scene_vast._provider_runtime_inputs(authority)

    monkeypatch.setenv("OPENAI_CONTENT_AGENTS_API_KEY_FILE", str(paths[3]))
    monkeypatch.setenv(
        "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_ID", "key_semantic"
    )
    with pytest.raises(
        scene_vast.TaskEvaluationSceneConfigurationVastError,
        match="scene_configuration_openai_stage_scopes_not_distinct",
    ):
        scene_vast._provider_runtime_inputs(authority)


def test_scene_configuration_resolves_pre_rename_stage_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A renamed paid resource class must not cost a production run.

    PR #1167 bound the visual-review stage to
    ``task_evaluation_scene_configuration_artifixer_visual_review``; the
    operator receipt on the production host still declared the pre-rename
    ``task_evaluation_artifixer_ai_visual_review``. Scene 839873 then refused
    at the no-spend allocator dry run for a name mismatch alone. Exclusivity is
    proven by the distinct provisioned key per stage, which
    ``scene_configuration_openai_stage_scopes_not_distinct`` still enforces
    above, so the lane now derives an equivalent receipt and continues.
    """

    authority = {
        "authority_digest": "sha256:" + "a" * 64,
        "external_service_spend_caps": {
            "openai": {
                "maximum_cost_usd": 1.0,
                "maximum_requests": 3,
                "stage_max_cost_usd": {
                    "artifixer_semantic_teacher": 0.2,
                    "artifixer_visual_review": 0.5,
                    "content_agents": 0.3,
                },
            }
        },
    }
    _configure_scene_openai_runtime_files(
        tmp_path,
        monkeypatch,
        visual_review_class="task_evaluation_artifixer_ai_visual_review",
    )

    with pytest.raises(
        scene_vast.TaskEvaluationSceneConfigurationVastError
    ) as excinfo:
        scene_vast._provider_runtime_inputs(authority)

    assert "openai_stage_scope_attestation_invalid" not in str(excinfo.value)


def test_scene_configuration_accepts_nonzero_scope_for_incremental_cost_metering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = {
        "authority_digest": "sha256:" + "a" * 64,
        "external_service_spend_caps": {
            "openai": {
                "maximum_cost_usd": 1.0,
                "maximum_requests": 3,
                "stage_max_cost_usd": {
                    "artifixer_semantic_teacher": 0.2,
                    "artifixer_visual_review": 0.5,
                    "content_agents": 0.3,
                },
            }
        },
    }
    _configure_scene_openai_runtime_files(tmp_path, monkeypatch)
    observed: list[str] = []

    def collect(**kwargs: object) -> dict[str, float]:
        observed.append(str(kwargs["api_key_id"]))
        return {"total_cost_usd": 0.877128}

    monkeypatch.setattr(scene_vast, "_collect_openai_cost_snapshot", collect)
    _secret_paths, runtime_environment = scene_vast._provider_runtime_inputs(authority)

    assert observed == ["key_semantic", "key_review", "key_content_agents"]
    assert runtime_environment["BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS"] == "true"
    assert (
        runtime_environment[
            "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_MAX_COST_USD"
        ]
        == "0.2"
    )


def test_scene_configuration_cost_preflight_uses_ephemeral_owner_only_admin_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "admin-key"
    source.write_text("fixture-admin-key-value", encoding="utf-8")
    source.chmod(0o640)
    observed: dict[str, object] = {}

    class Client:
        def __init__(
            self, *, project_id: str, api_key_id: str, admin_api_key_file: Path
        ) -> None:
            observed.update(
                {
                    "project_id": project_id,
                    "api_key_id": api_key_id,
                    "path": admin_api_key_file,
                    "mode": admin_api_key_file.stat().st_mode & 0o777,
                    "payload": admin_api_key_file.read_text(encoding="utf-8"),
                }
            )

        def snapshot(self, *, start_time: int, end_time: int) -> dict[str, float]:
            observed["window"] = (start_time, end_time)
            return {"total_cost_usd": 0.0}

    monkeypatch.setattr(scene_vast, "OpenAIOrganizationCostsClient", Client)
    snapshot = scene_vast._collect_openai_cost_snapshot(
        admin_api_key_file=str(source),
        project_id="proj_test",
        api_key_id="key_semantic",
        start_time=1,
        end_time=2,
    )

    assert snapshot == {"total_cost_usd": 0.0}
    assert observed["mode"] == 0o600
    assert observed["payload"] == "fixture-admin-key-value"
    assert observed["window"] == (1, 2)
    assert not Path(str(observed["path"])).exists()
    assert source.stat().st_mode & 0o777 == 0o640


def test_scene_configuration_provider_output_requires_complete_six_stage_chain(
    tmp_path: Path,
) -> None:
    assert vpa._provider_expected_video_count(
        "task_evaluation_scene_configuration"
    ) == 0
    stage_results = []
    for index in range(6):
        stage = {
            "schema_version": "task_evaluation_scene_configuration_stage_result.v1",
            "status": "completed",
            "stage_id": f"stage-{index}",
            "canonical_allocator": None,
            "provider_mutations_performed": 0,
            "paid_execution_requested": False,
            "executed_inside_parent_configuration_run": True,
            "raw_secret_values_recorded": False,
            "output_artifacts": [],
            "stage_result_digest": "",
        }
        stage["stage_result_digest"] = canonical_digest(
            stage, digest_field="stage_result_digest"
        )
        stage_results.append(stage)
    chain = {
        "schema_version": "task_evaluation_scene_configuration_provider_stage_chain.v1",
        "status": "completed",
        "run_id": "run-1",
        "stage_results": stage_results,
        "stage_result_digests": [
            stage["stage_result_digest"] for stage in stage_results
        ],
        "stage_count": 6,
        "executed_inside_one_parent_provider_run": True,
        "nested_provider_mutations_performed": 0,
        "nested_paid_execution_requested": False,
        "evaluation_episode_executed": False,
        "retry_cap": 0,
        "result_digest": "",
    }
    chain["result_digest"] = canonical_digest(chain, digest_field="result_digest")
    result = {
        "schema_version": "task_evaluation_scene_configuration_provider_result.v1",
        "status": "completed",
        "run_id": "run-1",
        "source_commit": "a" * 40,
        "construction_envelope_digest": "sha256:" + "b" * 64,
        "stage_chain": chain,
        "evaluation_episode_executed": False,
        "candidate_policy_queried": False,
        "provider_zero_required_after_return": True,
        "blockers": [],
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    archive = tmp_path / "output.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(
            "task_evaluation_scene_configuration_provider_result.v1.json",
            json.dumps(result),
        )

    adapter_inspection = vpa._inspect_provider_runtime_output_zip(
        archive, expected_video_count=0
    )
    assert adapter_inspection["runtime_result_present"] is True
    assert adapter_inspection["runtime_result_status"] == "completed"

    observed, blockers = scene_vast._extract_provider_output(
        archive,
        tmp_path / "extracted-output",
        maximum_archive_bytes=archive.stat().st_size,
    )
    assert blockers == []
    assert observed["stage_chain"]["stage_count"] == 6

    result["stage_chain"]["stage_results"][0]["status"] = "blocked"
    result["stage_chain"]["result_digest"] = canonical_digest(
        result["stage_chain"], digest_field="result_digest"
    )
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    invalid_archive = tmp_path / "invalid-stage-output.zip"
    with zipfile.ZipFile(invalid_archive, "w") as output:
        output.writestr(
            "task_evaluation_scene_configuration_provider_result.v1.json",
            json.dumps(result),
        )
    _observed, blockers = scene_vast._extract_provider_output(
        invalid_archive,
        tmp_path / "invalid-stage-output",
        maximum_archive_bytes=invalid_archive.stat().st_size,
    )
    assert "scene_configuration_stage_chain_invalid" in blockers


def test_provider_execution_binding_requires_the_receipt_run_identity() -> None:
    receipt = {
        "run_id": "scene-839873-run",
        "source_commit": "a" * 40,
        "construction_envelope_source_digest": "sha256:" + "c" * 64,
        "portable_construction_envelope_digest": "sha256:" + "b" * 64,
    }
    execution = {
        "run_id": receipt["run_id"],
        "source_commit": receipt["source_commit"],
        "construction_envelope_digest": receipt[
            "portable_construction_envelope_digest"
        ],
        "source_construction_envelope_digest": receipt[
            "construction_envelope_source_digest"
        ],
        "stage_chain": {"run_id": receipt["run_id"]},
    }

    assert scene_vast._provider_execution_binding_blockers(
        execution, receipt, diagnostic_only=False
    ) == []

    diagnostic_receipt = {
        **receipt,
        "source_diagnostic_checkpoint_digest": "sha256:" + "e" * 64,
        "diagnostic_bootstrap_mode": "fresh",
        "diagnostic_scientific_binding_digest": "sha256:" + "f" * 64,
    }
    minimal_diagnostic_failure = {
        "status": "blocked_diagnostic_only",
        "blockers": ["scene_configuration_diagnostic_provider_failed:fixture"],
    }
    assert scene_vast._provider_execution_binding_blockers(
        minimal_diagnostic_failure,
        diagnostic_receipt,
        diagnostic_only=True,
    ) == []
    minimal_diagnostic_failure["diagnostic_run_id"] = "different-run"
    assert scene_vast._provider_execution_binding_blockers(
        minimal_diagnostic_failure,
        diagnostic_receipt,
        diagnostic_only=True,
    ) == ["scene_configuration_provider_run_id_mismatch"]

    execution["source_construction_envelope_digest"] = "sha256:" + "d" * 64
    assert scene_vast._provider_execution_binding_blockers(
        execution, receipt, diagnostic_only=False
    ) == ["scene_configuration_provider_source_envelope_mismatch"]
    execution["source_construction_envelope_digest"] = receipt[
        "construction_envelope_source_digest"
    ]

    execution["run_id"] = "different-run"
    assert scene_vast._provider_execution_binding_blockers(
        execution, receipt, diagnostic_only=False
    ) == ["scene_configuration_provider_run_id_mismatch"]

    execution["run_id"] = receipt["run_id"]
    execution["stage_chain"] = {"run_id": "different-run"}
    assert scene_vast._provider_execution_binding_blockers(
        execution, receipt, diagnostic_only=False
    ) == ["scene_configuration_provider_run_id_mismatch"]

    execution["stage_chain"] = None
    execution["status"] = "blocked"
    assert scene_vast._provider_execution_binding_blockers(
        execution, receipt, diagnostic_only=False
    ) == []


def test_blocked_provider_result_retains_its_redacted_failure_in_terminal_blockers(
    tmp_path: Path,
) -> None:
    signed_detail = (
        "scene_configuration_provider_failed:RuntimeError:"
        "https://objects.example/output?X-Amz-Signature=do-not-retain"
    )
    result = {
        "schema_version": "task_evaluation_scene_configuration_provider_result.v1",
        "status": "blocked",
        "source_commit": "a" * 40,
        "construction_envelope_digest": "sha256:" + "b" * 64,
        "evaluation_episode_executed": False,
        "candidate_policy_queried": False,
        "provider_zero_required_after_return": True,
        "blockers": [signed_detail],
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    archive = tmp_path / "blocked-output.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(
            "task_evaluation_scene_configuration_provider_result.v1.json",
            json.dumps(result),
        )

    observed, blockers = scene_vast._extract_provider_output(
        archive,
        tmp_path / "blocked-output",
        maximum_archive_bytes=archive.stat().st_size,
    )

    assert observed["status"] == "blocked"
    assert len(blockers) == 1
    assert blockers[0].startswith(
        "provider_result_blocker:scene_configuration_provider_failed:RuntimeError:"
    )
    assert "do-not-retain" not in blockers[0]
    assert "<redacted>" in blockers[0]

    result["blockers"] = []
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    contradictory = tmp_path / "contradictory-output.zip"
    with zipfile.ZipFile(contradictory, "w") as output:
        output.writestr(
            "task_evaluation_scene_configuration_provider_result.v1.json",
            json.dumps(result),
        )
    _observed, blockers = scene_vast._extract_provider_output(
        contradictory,
        tmp_path / "contradictory-output",
        maximum_archive_bytes=contradictory.stat().st_size,
    )
    assert "scene_configuration_provider_result_contract_invalid" in blockers


def test_provider_output_extraction_seals_malformed_result_and_archive_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Corrupt paid output must become evidence, not escape terminal sealing."""

    malformed = tmp_path / "malformed.zip"
    with zipfile.ZipFile(malformed, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr(
            "task_evaluation_scene_configuration_provider_result.v1.json",
            "{not-json",
        )

    observed, blockers = scene_vast._extract_provider_output(
        malformed,
        tmp_path / "malformed-output",
        maximum_archive_bytes=malformed.stat().st_size,
    )

    assert observed == {}
    assert blockers == ["scene_configuration_provider_result_contract_invalid"]

    observed, blockers = scene_vast._extract_provider_output(
        malformed,
        tmp_path / "compressed-limit-output",
        maximum_archive_bytes=malformed.stat().st_size - 1,
    )
    assert observed == {}
    assert blockers == [
        "scene_configuration_provider_output_zip_exceeds_declared_transfer_ceiling"
    ]

    with monkeypatch.context() as member_limit:
        member_limit.setattr(scene_vast, "PROVIDER_OUTPUT_MAXIMUM_MEMBER_COUNT", 0)
        observed, blockers = scene_vast._extract_provider_output(
            malformed,
            tmp_path / "member-limit-output",
            maximum_archive_bytes=malformed.stat().st_size,
        )
    assert observed == {}
    assert blockers == [
        "scene_configuration_provider_output_archive_member_count_invalid"
    ]

    duplicate = tmp_path / "duplicate.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(duplicate, "w") as output:
            output.writestr("duplicate.txt", b"first")
            output.writestr("duplicate.txt", b"second")
    observed, blockers = scene_vast._extract_provider_output(
        duplicate,
        tmp_path / "duplicate-output",
        maximum_archive_bytes=duplicate.stat().st_size,
    )
    assert observed == {}
    assert blockers == [
        "scene_configuration_provider_output_archive_duplicate_member"
    ]

    expansion = tmp_path / "expansion.zip"
    with zipfile.ZipFile(expansion, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("highly-compressible.txt", b"0" * 10_000)
    monkeypatch.setattr(
        scene_vast,
        "PROVIDER_OUTPUT_MAXIMUM_EXPANSION_RATIO",
        1,
    )

    observed, blockers = scene_vast._extract_provider_output(
        expansion,
        tmp_path / "expansion-output",
        maximum_archive_bytes=expansion.stat().st_size,
    )

    assert observed == {}
    assert blockers == [
        "scene_configuration_provider_output_archive_expansion_invalid"
    ]
    assert not any((tmp_path / "expansion-output").iterdir())


def test_scene_configuration_provider_output_disk_formula_and_rechecks_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ZIP, its allowed expansion, and reserve must fit before paid work.

    The second check deliberately excludes the ZIP after it is resident.  Both
    boundaries are inclusive: exactly the required free bytes may proceed.
    """

    archive_ceiling = 1_000_000_000
    requirements = scene_vast._provider_output_disk_requirements(archive_ceiling)
    maximum_expanded = (
        archive_ceiling * scene_vast.PROVIDER_OUTPUT_MAXIMUM_EXPANSION_RATIO
    )
    assert requirements == {
        "maximum_archive_bytes": archive_ceiling,
        "maximum_expanded_bytes": maximum_expanded,
        "operational_reserve_bytes": (
            provider_artifacts.PROVIDER_OUTPUT_OPERATIONAL_RESERVE_BYTES
        ),
        "required_free_bytes_before_download": (
            archive_ceiling
            + maximum_expanded
            + provider_artifacts.PROVIDER_OUTPUT_OPERATIONAL_RESERVE_BYTES
        ),
        "required_free_bytes_before_extraction": (
            maximum_expanded
            + provider_artifacts.PROVIDER_OUTPUT_OPERATIONAL_RESERVE_BYTES
        ),
    }

    archive = tmp_path / "capacity-output.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("evidence.txt", b"evidence")
    extraction_required = requirements["required_free_bytes_before_extraction"]
    called = {"extract": 0}
    original_extract = scene_vast._extract_provider_output

    def observed_extract(*args, **kwargs):  # type: ignore[no-untyped-def]
        called["extract"] += 1
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(scene_vast, "_extract_provider_output", observed_extract)
    observed, blockers, capacity = (
        provider_artifacts.extract_provider_output_with_capacity_guard(
            archive,
            tmp_path / "insufficient-extraction",
            maximum_archive_bytes=archive_ceiling,
            extractor=scene_vast._extract_provider_output,
            disk_usage_provider=lambda _path: type(
                "Usage", (), {"free": extraction_required - 1}
            )(),
        )
    )
    assert observed == {}
    assert blockers == [
        "scene_configuration_provider_output_disk_capacity_insufficient"
    ]
    assert capacity["status"] == "blocked"
    assert called["extract"] == 0
    assert not (tmp_path / "insufficient-extraction").exists()

    _observed, blockers, capacity = (
        provider_artifacts.extract_provider_output_with_capacity_guard(
            archive,
            tmp_path / "exact-boundary-extraction",
            maximum_archive_bytes=archive_ceiling,
            extractor=scene_vast._extract_provider_output,
            disk_usage_provider=lambda _path: type(
                "Usage", (), {"free": extraction_required}
            )(),
        )
    )
    assert capacity["status"] == "ready"
    assert called["extract"] == 1
    assert blockers == ["scene_configuration_provider_result_missing"]

    def disk_error(_path: Path):
        raise OSError("fixture disk probe unavailable")

    unavailable = scene_vast._provider_output_disk_capacity(
        destination_directory=tmp_path,
        required_free_bytes=1,
        phase="fixture",
        disk_usage_provider=disk_error,
    )
    invalid = scene_vast._provider_output_disk_capacity(
        destination_directory=tmp_path,
        required_free_bytes=1,
        phase="fixture",
        disk_usage_provider=lambda _path: type("Usage", (), {"free": "unknown"})(),
    )
    assert unavailable["blockers"] == [
        "scene_configuration_provider_output_disk_capacity_unavailable"
    ]
    assert invalid["blockers"] == unavailable["blockers"]


def test_vast_provider_output_get_rechecks_capacity_before_fake_object_store(
    tmp_path: Path,
) -> None:
    """Insufficient or unknown space must prevent the signed object GET itself."""

    with pytest.raises(
        ValueError, match="invalid_vast_provider_output_minimum_free_bytes"
    ):
        vpa.run_vast_provider_adapter(
            job_dir=tmp_path / "invalid-capacity",
            provider_output_minimum_free_bytes=True,
        )
    output = tmp_path / "provider-output.zip"
    minimum = 5_000_000_000
    calls: list[str] = []

    def fake_object_store_get(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs["url"])
        Path(kwargs["output_path"]).write_bytes(b"provider-output")
        return {
            "status": "completed",
            "http_status_code": 200,
            "downloaded_size_bytes": len(b"provider-output"),
        }

    blocked = vpa._download_provider_output_with_capacity_guard(
        url="https://objects.example.test/output.zip?signature=fixture",
        output_path=output,
        minimum_free_bytes=minimum,
        disk_usage_provider=lambda _path: type(
            "Usage", (), {"free": minimum - 1}
        )(),
        downloader=fake_object_store_get,
    )
    assert blocked["status"] == "blocked"
    assert blocked["download_attempted"] is False
    assert blocked["blockers"] == ["provider_output_disk_capacity_insufficient"]
    assert calls == []
    assert not output.exists()

    def disk_error(_path: Path):
        raise OSError("fixture disk probe unavailable")

    unavailable = vpa._download_provider_output_with_capacity_guard(
        url="https://objects.example.test/output.zip?signature=fixture",
        output_path=output,
        minimum_free_bytes=minimum,
        disk_usage_provider=disk_error,
        downloader=fake_object_store_get,
    )
    invalid = vpa._download_provider_output_with_capacity_guard(
        url="https://objects.example.test/output.zip?signature=fixture",
        output_path=output,
        minimum_free_bytes=minimum,
        disk_usage_provider=lambda _path: type("Usage", (), {"free": None})(),
        downloader=fake_object_store_get,
    )
    assert unavailable["blockers"] == ["provider_output_disk_capacity_unavailable"]
    assert invalid["blockers"] == unavailable["blockers"]
    assert unavailable["download_attempted"] is False
    assert invalid["download_attempted"] is False
    assert calls == []

    completed = vpa._download_provider_output_with_capacity_guard(
        url="https://objects.example.test/output.zip?signature=fixture",
        output_path=output,
        minimum_free_bytes=minimum,
        disk_usage_provider=lambda _path: type("Usage", (), {"free": minimum})(),
        downloader=fake_object_store_get,
    )
    assert completed["status"] == "completed"
    assert completed["download_attempted"] is True
    assert completed["disk_capacity"]["status"] == "ready"
    assert len(calls) == 1
    assert output.read_bytes() == b"provider-output"


def test_provider_result_seals_archive_relative_artifact_paths(
    tmp_path: Path,
) -> None:
    runner_path = (
        Path(__file__).resolve().parents[1]
        / "scripts/task_evaluation_scene_configuration_provider_runner.py"
    )
    spec = importlib.util.spec_from_file_location(
        "scene_configuration_portable_runner", runner_path
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    output = tmp_path / "runtime_output"
    artifact = output / "stages/stage-1/adapter/configured.usda"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"#usda 1.0\n")
    stage = {
        "output_artifacts": [
            {
                "role": "configured_scene",
                "path": str(artifact),
                "digest": _sha256(artifact),
                "size_bytes": artifact.stat().st_size,
            }
        ],
        "stage_result_digest": "",
    }
    stage["stage_result_digest"] = canonical_digest(
        stage, digest_field="stage_result_digest"
    )
    chain = {
        "stage_results": [stage],
        "stage_result_digests": [stage["stage_result_digest"]],
        "result_digest": "",
    }
    chain["result_digest"] = canonical_digest(
        chain, digest_field="result_digest"
    )

    portable = runner._portable_stage_chain(chain, output_root=output)

    row = portable["stage_results"][0]["output_artifacts"][0]
    assert row["provider_output_relative_path"] == (
        "stages/stage-1/adapter/configured.usda"
    )
    assert portable["stage_results"][0]["stage_result_digest"] == canonical_digest(
        portable["stage_results"][0], digest_field="stage_result_digest"
    )
    assert portable["result_digest"] == canonical_digest(
        portable, digest_field="result_digest"
    )


def test_control_plane_rehydrates_only_digest_bound_provider_artifacts(
    tmp_path: Path,
) -> None:
    extraction = tmp_path / "immutable_execution"
    artifact = extraction / "stages/stage-1/adapter/configured.usda"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"#usda 1.0\n")
    stage_results = [
        {
            "output_artifacts": (
                [
                    {
                        "role": "configured_scene",
                        "path": "/workspace/runtime_output/stages/stage-1/adapter/configured.usda",
                        "provider_output_relative_path": (
                            "stages/stage-1/adapter/configured.usda"
                        ),
                        "digest": _sha256(artifact),
                        "size_bytes": artifact.stat().st_size,
                    }
                ]
                if index == 0
                else []
            )
        }
        for index in range(6)
    ]
    execution = {"stage_chain": {"stage_results": stage_results}}

    hydrated = scene_vast._publication_stage_results(
        execution, extraction_root=extraction
    )

    assert hydrated[0]["output_artifacts"][0]["path"] == str(artifact.resolve())
    execution["stage_chain"]["stage_results"][0]["output_artifacts"][0][
        "provider_output_relative_path"
    ] = "../outside.usda"
    with pytest.raises(
        scene_vast.TaskEvaluationSceneConfigurationVastError,
        match="scene_configuration_provider_artifact_portability_invalid",
    ):
        scene_vast._publication_stage_results(
            execution, extraction_root=extraction
        )


def test_completed_vast_run_cannot_finish_without_publishing_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import types

    receipt = _build(tmp_path, "bundle")
    receipt_path = tmp_path / "bundle" / f"{BUNDLE_SCHEMA_VERSION}.receipt.json"
    authority_path = tmp_path / "authority.json"
    authority_path.write_text("{}", encoding="utf-8")
    authority = {
        "authority_digest": "sha256:" + "e" * 64,
        "hard_attempt_spend_cap_usd": MAX_ATTEMPT_SPEND_USD,
        "provider_compute_spend_cap_usd": MAX_PROVIDER_COMPUTE_SPEND_USD,
        "maximum_hourly_rate_usd": MAX_HOURLY_RATE_USD,
        "maximum_single_resource_ttl_seconds": REQUIRED_PARENT_TTL_SECONDS,
        "container_image": "nvcr.io/nvidia/isaac-sim@sha256:" + "b" * 64,
        "external_service_spend_caps": {
            "openai": {"maximum_cost_usd": 1.5, "maximum_requests": 32}
        },
    }
    monkeypatch.setattr(
        scene_vast,
        "validate_scene_configuration_paid_authority",
        lambda _value, **_kwargs: authority,
    )
    monkeypatch.setattr(
        scene_vast, "_provider_runtime_inputs", lambda _authority: ({}, {})
    )
    monkeypatch.setattr(
        scene_vast, "require_paid_resource_admission_grant", lambda *_a, **_k: None
    )

    def stage(*, job_dir, **_kwargs):
        staging = Path(job_dir)
        staging.mkdir(parents=True, exist_ok=True)
        for name in (
            "provider_bundle_url.txt",
            "provider_output_put_url.txt",
            "provider_output_get_url.txt",
        ):
            (staging / name).write_text(
                f"https://objects.example.test/{name}", encoding="utf-8"
            )
        return {"status": "completed"}

    monkeypatch.setattr(scene_vast, "stage_wam_provider_bundle_object_store", stage)
    monkeypatch.setattr(
        scene_vast,
        "arm_independent_vast_watchdog",
        lambda **kwargs: (
            {"status": "armed"},
            types.SimpleNamespace(
                pod_name_prefix=kwargs["pod_name_prefix"] + "fixture-",
                started_instance_id_path=Path(kwargs["job_dir"]) / "started.json",
                deadline_epoch=9_999_999_999.0,
            ),
        ),
    )
    monkeypatch.setattr(
        scene_vast,
        "close_independent_vast_watchdog",
        lambda **_kwargs: {"status": "provider_terminal"},
    )
    monkeypatch.setattr(
        scene_vast,
        "cleanup_staged_wam_provider_objects",
        lambda _root: {"all_objects_absent": True},
    )
    monkeypatch.setattr(
        scene_vast,
        "_consume_authority_once",
        lambda _authority, **_kwargs: {"status": "consumed"},
    )
    monkeypatch.setattr(
        scene_vast,
        "_stage_owner_only_runtime_secrets",
        lambda **_kwargs: ({}, None),
    )
    from tests.test_task_evaluation_launch_preparation_contract import (
        test_configuration_request as configuration_request_fixture,
    )

    request = configuration_request_fixture()
    request["run_id"] = receipt["run_id"]
    request["expected_production_commit"] = receipt["source_commit"]
    source_envelope = json.loads(
        (tmp_path / "source" / "envelope.json").read_text(encoding="utf-8")
    )
    publication_envelope = {
        "orchestration_id": source_envelope["orchestration_id"],
        "run_id": receipt["run_id"],
        "team_namespace": request["team_namespace"],
        "expected_production_commit": receipt["source_commit"],
        "recipe_digest": source_envelope["recipe_digest"],
        "control_plane_envelope_digest": source_envelope["envelope_digest"],
        "request": request,
        "recipe": {
            "scene_identity": request["scene"]["identity"],
            "task_identity": request["task"]["identity"],
            "subject_identity": request["task"]["subject"]["identity"],
        },
        "render_inputs_result": {
            "status": "derived_method_inputs_materialized",
            "raw_interiorgs_bytes_in_provider_packet": False,
        },
        "provider_disclosure_receipt": {
            "raw_interiorgs_bytes_in_provider_bundle": False,
        },
    }
    monkeypatch.setattr(
        scene_vast,
        "_portable_construction_envelope",
        lambda _receipt: publication_envelope,
    )

    roles = {
        "configured_appearance_without_source_object": "appearance.usdc",
        "appearance_removal_receipt": "appearance-receipt.json",
        "appearance_visual_review_receipt": "appearance-review.json",
        "configured_task_thumbnail": "configured-task-thumbnail.png",
        "configured_collision_without_source_object": "collision.usda",
        "collision_excision_receipt": "collision-receipt.json",
        "statically_qualified_replacement_asset": "static.usda",
        "static_qualification_receipt": "static-receipt.json",
        "native_qualified_replacement_asset": "native.usda",
        "native_import_qualification_receipt": "native-receipt.json",
        "configured_scene_bundle_candidate_manifest": "candidate.json",
        "scene_assembly_receipt": "assembly-receipt.json",
    }
    thumbnail_payload = b"exact-reviewed-frame"
    thumbnail_digest = "sha256:" + hashlib.sha256(thumbnail_payload).hexdigest()
    appearance_review: dict[str, object] = {
        "schema_version": "task_evaluation_artifixer_ai_visual_review.v1",
        "status": "accepted",
        "review_frame_count": 8,
        "task_thumbnail_is_exact_review_frame": True,
        "task_thumbnail_selection": {
            "camera_id": "camera-3",
            "frame_sha256": thumbnail_digest,
            "rationale": "The task surface and configured scene are both clear.",
        },
        "reviewer": {
            "kind": "ai",
            "identity": "artifixer-independent-vision-reviewer-v1",
            "runtime": "openai_agents_sdk",
            "model": "gpt-5.6-terra",
        },
        "receipt_digest": "",
    }
    appearance_review["receipt_digest"] = canonical_digest(
        appearance_review, digest_field="receipt_digest"
    )
    special_payloads = {
        "appearance_visual_review_receipt": json.dumps(
            appearance_review, separators=(",", ":")
        ).encode(),
        "configured_task_thumbnail": thumbnail_payload,
    }

    def adapter(**kwargs):
        provider_run = Path(kwargs["job_dir"])
        provider_run.mkdir(parents=True, exist_ok=True)
        (provider_run / "vast_teardown_manifest.json").write_text(
            json.dumps(
                {
                    "continuing_spend_from_this_run": False,
                    "vast_instance_ids": [123],
                }
            ),
            encoding="utf-8",
        )
        (provider_run / "vast_provider_adapter_result.json").write_text(
            json.dumps({"status": "completed", "vast_instance_ids": [123]}),
            encoding="utf-8",
        )
        rows = []
        archive_members: dict[str, bytes] = {}
        for role, name in roles.items():
            relative = f"stages/stage-1/adapter/{name}"
            payload = special_payloads.get(role, (role + "\n").encode())
            archive_members[relative] = payload
            rows.append(
                {
                    "role": role,
                    "path": "/workspace/runtime_output/" + relative,
                    "provider_output_relative_path": relative,
                    "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
            )
        stages = [
            {
                "schema_version": (
                    "task_evaluation_scene_configuration_stage_result.v1"
                ),
                "status": "completed",
                "stage_id": f"stage-{index + 1}",
                "canonical_allocator": None,
                "provider_mutations_performed": 0,
                "paid_execution_requested": False,
                "executed_inside_parent_configuration_run": True,
                "raw_secret_values_recorded": False,
                "output_artifacts": rows if index == 0 else [],
                "stage_result_digest": "",
            }
            for index in range(6)
        ]
        for stage in stages:
            stage["stage_result_digest"] = canonical_digest(
                stage, digest_field="stage_result_digest"
            )
        chain = {
            "schema_version": "task_evaluation_scene_configuration_provider_stage_chain.v1",
            "status": "completed",
            "run_id": receipt["run_id"],
            "stage_results": stages,
            "stage_result_digests": [
                stage["stage_result_digest"] for stage in stages
            ],
            "stage_count": 6,
            "executed_inside_one_parent_provider_run": True,
            "nested_provider_mutations_performed": 0,
            "nested_paid_execution_requested": False,
            "evaluation_episode_executed": False,
            "retry_cap": 0,
            "result_digest": "",
        }
        chain["result_digest"] = canonical_digest(
            chain, digest_field="result_digest"
        )
        provider_result = {
            "schema_version": "task_evaluation_scene_configuration_provider_result.v1",
            "status": "completed",
            "run_id": receipt["run_id"],
            "source_commit": receipt["source_commit"],
            "source_construction_envelope_digest": receipt[
                "construction_envelope_source_digest"
            ],
            "construction_envelope_digest": receipt[
                "portable_construction_envelope_digest"
            ],
            "stage_chain": chain,
            "evaluation_episode_executed": False,
            "candidate_policy_queried": False,
            "provider_zero_required_after_return": True,
            "blockers": [],
            "result_digest": "",
        }
        provider_result["result_digest"] = canonical_digest(
            provider_result, digest_field="result_digest"
        )
        with zipfile.ZipFile(kwargs["provider_runtime_output_zip"], "w") as archive:
            for name, payload in archive_members.items():
                archive.writestr(name, payload)
            archive.writestr(
                "task_evaluation_scene_configuration_provider_result.v1.json",
                json.dumps(provider_result),
            )
        return {
            "status": "completed",
            "vast_instance_ids": [123],
            "provider_create_attempted": True,
        }

    monkeypatch.setattr(scene_vast, "run_vast_provider_adapter", adapter)
    object_store = tmp_path / "configured-object-store"
    object_store.mkdir()

    def publisher_factory():
        def publish(*, path: Path, object_name: str):
            destination = object_store / object_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            return {
                "uri": f"s3://blueprint-inputs/{object_name}",
                "digest": _sha256(path),
                "size_bytes": path.stat().st_size,
                "full_byte_service_account_readback_passed": True,
                "readback_digest": _sha256(destination),
                "readback_size_bytes": destination.stat().st_size,
            }

        return publish

    monkeypatch.setattr(
        scene_vast, "configured_scene_object_store_publisher", publisher_factory
    )

    def durable_publisher(*, path: Path, artifact_kind: str):
        digest = _sha256(path)
        destination = object_store / "artifacts" / digest.removeprefix("sha256:")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        return {
            "schema_version": "task_evaluation_scene_artifact_reference.v1",
            "status": "remote_verified",
            "artifact_kind": artifact_kind,
            "uri": f"s3://blueprint-inputs/artifacts/{destination.name}",
            "digest": digest,
            "size_bytes": path.stat().st_size,
            "cache_hit": False,
            "upload_performed": True,
            "content_addressed_key": True,
            "remote_identity_verified": True,
            "full_byte_service_account_readback_passed": True,
            "readback_digest": _sha256(destination),
            "readback_size_bytes": destination.stat().st_size,
            "raw_secret_values_recorded": False,
        }

    monkeypatch.setattr(
        provider_artifacts, "publish_configured_scene_artifact", durable_publisher
    )

    result = scene_vast.run_scene_configuration_vast(
        job_dir=tmp_path / "job",
        bundle_receipt_path=receipt_path,
        paid_attempt_authority_path=authority_path,
        paid_resource_admission_grant=object(),
        execute=True,
        scene_construction_queue_root=_construction_queue(tmp_path),
        disk_usage_provider=lambda _path: type(
            "Usage", (), {"free": 100_000_000_000}
        )(),
    )

    assert result["status"] == "completed", result["blockers"]
    assert result["configured_scene_published"] is True
    assert result["full_byte_service_account_readback_passed"] is True
    assert result["scene_construction_queue_finalization"]["queue_state"] == "completed"
    assert not list((tmp_path / "construction-queue" / "pending").glob("*.json"))
    assert len(list((tmp_path / "construction-queue" / "completed").glob("*.json"))) == 1
    assert Path(result["configured_scene_revision_path"]).is_file()
    assert Path(result["publication_result_path"]).is_file()
    assert result["configured_scene_revision_reference"]["uri"].startswith(
        "s3://blueprint-inputs/"
    )
    assert result["provider_runtime_output_remote_reference"][
        "remote_identity_verified"
    ] is True
    assert Path(result["provider_runtime_output_remote_index_path"]).is_file()
    manifest = json.loads(Path(result["artifact_manifest_path"]).read_text())
    assert "configured_scene_publication" in manifest["observed_roles"]
    assert "durable_provider_output" in manifest["observed_roles"]


def test_scene_configuration_watchdog_prefix_is_in_the_blueprint_namespace() -> None:
    """The watchdog reaps by provider name prefix, so the prefix it is armed
    with must be one the watchdog will actually accept.

    The lane once passed the paid attempt authority's ``resource_name`` here.
    That value names which *attempt* is authorized (the activation id, e.g.
    ``adp-new-scene-...-activation-a``); it is not a provider namespace, and
    ``arm_independent_vast_watchdog`` refuses anything outside ``blueprint-``
    so a watchdog can never destroy another tenant's instances. Every paid
    launch of this lane therefore died with
    ``independent_vast_watchdog_prefix_invalid`` after the provider bundle
    had already been staged to object store.
    """

    import re

    from blueprint_pipeline.task_evaluation_scene_configuration_vast import (
        WATCHDOG_POD_NAME_PREFIX,
    )

    assert re.fullmatch(
        r"blueprint-[a-z0-9-]{1,100}-", WATCHDOG_POD_NAME_PREFIX
    ), WATCHDOG_POD_NAME_PREFIX


def test_provider_output_archive_is_remote_verified_and_indexed(
    tmp_path: Path,
) -> None:
    output_zip = tmp_path / "vast_provider_runtime_output.zip"
    output_zip.write_bytes(b"exact provider output")
    observed: list[tuple[Path, str]] = []

    def publisher(*, path: Path, artifact_kind: str):
        observed.append((path, artifact_kind))
        digest = _sha256(path)
        return {
            "schema_version": "task_evaluation_scene_artifact_reference.v1",
            "status": "remote_verified",
            "artifact_kind": artifact_kind,
            "uri": (
                "s3://blueprint-inputs/artifacts/provider-output/sha256/"
                f"{digest.removeprefix('sha256:')}/{path.name}"
            ),
            "digest": digest,
            "size_bytes": path.stat().st_size,
            "cache_hit": False,
            "upload_performed": True,
            "content_addressed_key": True,
            "remote_identity_verified": True,
            "full_byte_service_account_readback_passed": True,
            "readback_digest": digest,
            "readback_size_bytes": path.stat().st_size,
            "raw_secret_values_recorded": False,
        }

    reference, index, index_path = scene_vast._publish_provider_output_archive(
        output_zip=output_zip,
        job=tmp_path,
        receipt={
            "run_id": "scene-839873-run-1",
            "source_commit": "a" * 40,
            "bundle_sha256": "sha256:" + "b" * 64,
        },
        publisher=publisher,
    )

    assert observed == [(output_zip, "provider-output")]
    assert reference["remote_identity_verified"] is True
    assert index["artifact_references"] == [reference]
    assert index["all_artifacts_remote_verified"] is True
    assert index_path.stat().st_mode & 0o777 == 0o440
    assert json.loads(index_path.read_text(encoding="utf-8")) == index


def test_scene_configuration_arms_the_watchdog_with_that_exact_prefix() -> None:
    """The constant is only worth anything if the lane actually passes it."""

    import inspect

    from blueprint_pipeline import task_evaluation_scene_configuration_vast as lane

    source = inspect.getsource(lane.run_scene_configuration_vast)
    assert "pod_name_prefix=WATCHDOG_POD_NAME_PREFIX" in source
    assert 'authority["resource_name"]) + "-"' not in source


def test_scene_configuration_prefix_is_registered_and_labels_its_instances() -> None:
    """Arming and labelling must agree, and both layers must accept the prefix.

    ``arm_independent_vast_watchdog`` enforces the ``blueprint-`` namespace,
    but the watchdog *process* additionally requires the prefix be one of the
    registered canary families -- otherwise it refuses with
    ``watchdog_pod_name_prefix_not_canary_scoped``. The registry's own
    comments record the danger of satisfying it by borrowing another lane's
    family: the name-scoped sweep then matches an empty set while still
    reporting provider zero. So the lane's prefix must be registered under its
    own name, and the Vast adapter must label instances with the very prefix
    the watchdog armed on.
    """

    import inspect

    from blueprint_pipeline.groot_oscar_runpod_watchdog import (
        CANARY_NAME_PREFIXES,
    )
    from blueprint_pipeline import task_evaluation_scene_configuration_vast as lane

    assert lane.WATCHDOG_POD_NAME_PREFIX in CANARY_NAME_PREFIXES
    assert lane.WATCHDOG_POD_NAME_PREFIX.startswith(CANARY_NAME_PREFIXES)

    source = inspect.getsource(lane.run_scene_configuration_vast)
    assert "instance_label_prefix=watchdog.pod_name_prefix" in source


def test_scene_configuration_result_reports_why_the_adapter_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An adapter failure must reach the lane result, not just the teardown note.

    When ``run_vast_provider_adapter`` raises, the lane records
    ``vast_adapter_failed:<detail>`` on its adapter dict and seals an
    unallocated teardown. If that blocker is dropped, the result carries only
    the downstream consequences -- provider result missing, output zip
    invalid, envelope mismatch -- and those look identical whether the adapter
    could not find an offer, could not reach the provider, or was never
    invoked at all.
    """

    import inspect

    from blueprint_pipeline import task_evaluation_scene_configuration_vast as lane

    source = inspect.getsource(lane.run_scene_configuration_vast)
    merge_index = source.find('adapter.get("blockers")')
    result_index = source.rfind('"blockers": sorted(set(blockers))')
    assert merge_index != -1, "adapter blockers are never merged into the result"
    assert result_index != -1
    assert merge_index < result_index, (
        "adapter blockers must be merged before the result is assembled"
    )


def test_runtime_secrets_are_staged_owner_only_for_the_adapter(
    tmp_path: Path,
) -> None:
    """The lane's own rule and the adapter's rule must both be satisfiable.

    ``_provider_runtime_inputs`` requires each source secret to be group
    readable and no wider (host convention: ``root:blueprint 0640``), while
    the Vast adapter refuses any path with ``st_mode & 0o077`` set. A
    root-owned file cannot satisfy both, so the lane must hand the adapter an
    owner-only copy it made itself.
    """

    import stat as stat_module

    from blueprint_pipeline import task_evaluation_scene_configuration_vast as lane

    source = tmp_path / "openai_cost_scope_attestation.json"
    source.write_text('{"scope":"artifixer"}\n', encoding="utf-8")
    source.chmod(0o640)
    assert source.stat().st_mode & 0o077  # the source is group readable

    job = tmp_path / "job"
    job.mkdir()
    staged, root = lane._stage_owner_only_runtime_secrets(
        job_dir=job,
        secret_paths={"BLUEPRINT_OPENAI_TEST_ATTESTATION_FILE": str(source)},
    )

    staged_path = Path(staged["BLUEPRINT_OPENAI_TEST_ATTESTATION_FILE"])
    assert staged_path.read_bytes() == source.read_bytes()
    # Exactly the adapter's rule.
    assert staged_path.stat().st_mode & 0o077 == 0
    assert stat_module.S_IMODE(staged_path.stat().st_mode) == 0o600

    lane._discard_staged_runtime_secrets(root)
    assert not staged_path.exists()
    assert not root.exists()
    assert source.read_text(encoding="utf-8") == '{"scope":"artifixer"}\n'


def test_runtime_secret_cleanup_failure_is_a_named_terminal_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blueprint_pipeline import task_evaluation_scene_configuration_vast as lane

    source = tmp_path / "openai-key"
    source.write_text("opaque-test-secret\n", encoding="utf-8")
    source.chmod(0o640)
    job = tmp_path / "job"
    job.mkdir()
    staged, root = lane._stage_owner_only_runtime_secrets(
        job_dir=job,
        secret_paths={"OPENAI_API_KEY_FILE": str(source)},
    )
    staged_path = Path(staged["OPENAI_API_KEY_FILE"])
    original_unlink = Path.unlink

    def refuse_staged_secret_unlink(path: Path, *args, **kwargs) -> None:
        if path == staged_path:
            raise PermissionError("simulated cleanup refusal")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_staged_secret_unlink)

    assert lane._discard_staged_runtime_secrets(root) == [
        "scene_configuration_openai_runtime_secret_cleanup_failed"
    ]
    assert staged_path.is_file()
    assert root is not None and root.is_dir()


def test_the_lane_hands_the_adapter_the_staged_paths_and_always_discards_them(
    tmp_path: Path,
) -> None:
    """Staging is worthless if the raw paths are still passed, or if the
    private copies outlive the run."""

    import inspect

    from blueprint_pipeline import task_evaluation_scene_configuration_vast as lane

    source = inspect.getsource(lane.run_scene_configuration_vast)
    stage_index = source.find("_stage_owner_only_runtime_secrets")
    adapter_index = source.find("runtime_secret_file_paths=runtime_secret_paths")
    assert stage_index != -1, "the lane never stages owner-only copies"
    assert adapter_index != -1
    assert stage_index < adapter_index, (
        "secrets must be staged before they reach the adapter"
    )
    assert "_discard_staged_runtime_secrets(staged_secret_root)" in source
    assert source.count("_discard_staged_runtime_secrets") >= 2, (
        "the private copies must be discarded on the failure path too"
    )
    assert "blockers.extend(runtime_secret_cleanup_blockers)" in source
    assert '"runtime_secret_cleanup_completed"' in source


def _decision(*, provider: bool) -> dict:
    from blueprint_pipeline.task_evaluation_scene_configuration_disclosure import (
        CONTROL_PLANE,
        PROVIDER_GPU,
        SCHEMA_VERSION,
    )

    decision = {
        "schema_version": SCHEMA_VERSION,
        "render_execution_site": PROVIDER_GPU if provider else CONTROL_PLANE,
        "source_appearance_bytes_to_provider": provider,
        "decision_digest": "",
    }
    decision["decision_digest"] = canonical_digest(
        decision, digest_field="decision_digest"
    )
    return decision


def test_a_provider_rendered_bundle_may_declare_the_bytes_it_carries() -> None:
    """The rule is conditional on a digest-bound decision, not a literal.

    When a scene's rights admit the upload, the packet really does carry the
    raw source bytes and the derived views are produced on the provider. A
    manifest that hardcoded ``False`` there asserted no bytes crossed on a run
    where they did -- a provenance falsehood, not a guarantee.
    """

    from blueprint_pipeline.task_evaluation_scene_configuration_bundle import (
        _receipt_disclosure_is_coherent,
    )

    assert _receipt_disclosure_is_coherent(
        {"raw_interiorgs_bytes_in_provider_bundle": False}
    )
    assert _receipt_disclosure_is_coherent(
        {
            "raw_interiorgs_bytes_in_provider_bundle": True,
            "disclosure_decision": _decision(provider=True),
        }
    )


def test_claiming_the_bytes_crossed_without_an_authorizing_decision_fails() -> None:
    """The safety property is moved, not weakened."""

    from blueprint_pipeline.task_evaluation_scene_configuration_bundle import (
        _receipt_disclosure_is_coherent,
    )

    # No decision at all.
    assert not _receipt_disclosure_is_coherent(
        {"raw_interiorgs_bytes_in_provider_bundle": True}
    )
    # A decision that names the control plane cannot authorize a crossing.
    assert not _receipt_disclosure_is_coherent(
        {
            "raw_interiorgs_bytes_in_provider_bundle": True,
            "disclosure_decision": _decision(provider=False),
        }
    )
    # A tampered decision no longer matches its own digest.
    tampered = _decision(provider=True)
    tampered["source_appearance_bytes_to_provider"] = True
    tampered["render_execution_site"] = "provider_gpu"
    tampered["decision_digest"] = "sha256:" + "0" * 64
    assert not _receipt_disclosure_is_coherent(
        {
            "raw_interiorgs_bytes_in_provider_bundle": True,
            "disclosure_decision": tampered,
        }
    )
    # Silence is never permission.
    assert not _receipt_disclosure_is_coherent(
        {"raw_interiorgs_bytes_in_provider_bundle": None}
    )


def test_the_receipt_carries_and_cross_compares_its_authorizing_decision() -> None:
    """A byte-crossing claim must travel with the decision that permitted it.

    ``_receipt_disclosure_is_coherent`` can only check a claim it can see. The
    bundle receipt is built as ``{**manifest, ...}``, so the manifest has to
    record the envelope's decision -- otherwise a provider-render receipt says
    the source bytes crossed and carries nothing to justify it, and the paid
    authority refuses the bundle with ``receipt_contract_invalid``.

    It must also be cross-compared between the receipt and the manifest sealed
    inside the bundle, or a receipt's decision could drift from the bundle's
    and the claim would be checkable only against itself.
    """

    import inspect

    from blueprint_pipeline import task_evaluation_scene_configuration_bundle as bundle

    source = inspect.getsource(bundle)
    manifest_index = source.find(
        '"raw_interiorgs_bytes_in_provider_bundle": provider_render,'
    )
    assert manifest_index != -1
    decision_index = source.find('"disclosure_decision": (', manifest_index)
    assert decision_index != -1, "the manifest never records the decision"

    compared = source[source.find("compared_fields = (") :]
    assert '"disclosure_decision",' in compared[: compared.find(")")], (
        "the decision is not cross-compared between receipt and bundle manifest"
    )


def test_provider_entrypoint_restores_toolchain_modes_before_sealing() -> None:
    """CPython's zipfile drops mode bits; the toolchain check compares them.

    The bundle is unpacked on the provider with ``python -m zipfile -e``, which
    never restores permissions, so every file lands 0644. The toolchain
    self-check then compares each file's execute bit against the ``executable``
    flag its own manifest recorded, and refuses on the first mismatch --
    before any stage runs. Anything the stages exec would also raise
    PermissionError.
    """

    from pathlib import Path as _Path

    script = _Path("scripts/run_task_evaluation_scene_configuration_provider.sh").read_text(
        encoding="utf-8"
    )
    restore = script.find("task_evaluation_scene_configuration_toolchain.v1.json")
    seal = script.find('chmod -R a-w "$RUNTIME_ROOT/toolchain"')
    assert restore != -1, "the entrypoint never restores recorded modes"
    assert seal != -1
    assert restore < seal, "modes must be restored before the tree is sealed read-only"
    assert "0o555" in script and "0o444" in script


def test_provider_entrypoint_emits_varying_progress() -> None:
    """A silent container reads as a hung one to the no-progress watchdog.

    Every stage runs its child with ``capture_output=True``, so nothing is
    printed between entrypoint start and exit. The watchdog compares log
    tails and would tear the instance down mid-run. The payload has to vary
    on its own -- the watchdog strips timestamps before comparing -- so a
    clock is not enough.
    """

    from pathlib import Path as _Path

    script = _Path("scripts/run_task_evaluation_scene_configuration_provider.sh").read_text(
        encoding="utf-8"
    )
    assert "BLUEPRINT_SCENE_CONFIGURATION_PROGRESS" in script
    assert "tick=" in script, "the emitted payload must vary, not just repeat"


def test_provider_materializes_and_preflights_sealed_python_closure() -> None:
    """Stage-only dependencies must be present before expensive stage one."""

    from pathlib import Path as _Path

    script = _Path(
        "scripts/run_task_evaluation_scene_configuration_provider.sh"
    ).read_text(encoding="utf-8")
    materialize = script.find(
        "blueprint_pipeline.task_evaluation_scene_configuration_python_runtime"
    )
    runtime_parent = script.find('mkdir -p "$(dirname "$PYTHON_RUNTIME")"')
    import_preflight = script.find("import agents")
    runner = script.find(
        '"$RUNTIME_ROOT/task_evaluation_scene_configuration_provider_runner.py"'
    )
    assert runtime_parent != -1
    assert materialize != -1
    assert import_preflight != -1
    assert runner != -1
    assert runtime_parent < materialize < import_preflight < runner
    assert "scene_configuration_provider_python_runtime_invalid" in script
    assert "scene_configuration_provider_stage_import_closure_invalid" in script
    assert "BLUEPRINT_SCENE_CONFIGURATION_PARENT_DEADLINE_EPOCH" in script
    assert "timeout --signal=TERM --kill-after=30" in script
    assert (
        "scene_configuration_parent_runtime_budget_exhausted_during_stage_chain"
        in script
    )
    assert "timeout" in vpa.SCENE_CONFIGURATION_REQUIRED_COMMANDS.split()


def test_stage_budget_exceeds_the_budget_it_grants_its_child() -> None:
    """An inverted ladder kills work the inner tool was still allowed to do."""

    from blueprint_pipeline.task_evaluation_scene_configuration_builtin_producers import (
        _STAGE_TIMEOUTS_SECONDS,
    )
    from blueprint_pipeline.task_evaluation_scene_configuration_paid_authority import (
        MAX_TTL_SECONDS,
    )

    stage_one = _STAGE_TIMEOUTS_SECONDS["artifixer3d_observed_object_removal"]
    # The stage tool grants its component 7_200s and the artifixer driver
    # grants ArtiFixer3D 7_000s.
    assert stage_one > 7_200
    assert stage_one <= MAX_TTL_SECONDS
    assert sum(_STAGE_TIMEOUTS_SECONDS.values()) == SERIAL_GPU_STAGE_TIMEOUT_SECONDS
    assert (
        SERIAL_GPU_STAGE_TIMEOUT_SECONDS + OUTPUT_AND_CLOSURE_RESERVE_SECONDS
        < MAX_TTL_SECONDS
    )


def test_scene_configuration_gets_the_cold_pull_heartbeat_window() -> None:
    """The lane admits an 18-minute cold Isaac pull, so the window must know."""

    import inspect

    from blueprint_pipeline import vast_probe_guards

    source = inspect.getsource(vast_probe_guards)
    assert '"task_evaluation_scene_configuration",' in source


def test_provider_render_resolves_inputs_against_the_runtime_root() -> None:
    """The bundle records these paths relative to the runtime, not the package.

    ``bundle.py`` writes ``source_appearance.path`` relative to
    ``provider_runtime`` ("input/render/source_appearance.ply"), and
    ``_hydrate_envelope`` never absolutises it. Joining it to the component's
    package directory looks under
    ``toolchain/components/<name>/package/input/render/…``, where the staged
    appearance has never existed.
    """

    import inspect

    from blueprint_pipeline import (
        task_evaluation_scene_configuration_artifixer_driver as driver,
    )

    source = inspect.getsource(driver)
    assert "appearance = package_root /" not in source, (
        "the appearance is still resolved against the component package root"
    )
    assert "BLUEPRINT_SCENE_CONFIGURATION_RUNTIME_ROOT" in source
    assert "input_root=runtime_root," in source


def test_provider_render_passes_the_exact_retained_gaussian_count() -> None:
    """"method_input" is a qualified class; the renderer refuses without it.

    The control-plane call passes the count. The provider call did not, and
    fell back to parsing ``element vertex N`` out of the PLY header -- absent
    whenever the source appearance is compressed or not a standard PLY. The
    packet already carries the exact number.
    """

    import inspect

    from blueprint_pipeline.task_evaluation_scene_configuration_render_inputs import (
        complete_provider_render_inputs,
    )

    source = inspect.getsource(complete_provider_render_inputs)
    assert "retained_gaussian_count=_packet_source_count(render_inputs)" in source

    from blueprint_pipeline.task_evaluation_scene_configuration_render_inputs import (
        _packet_source_count,
    )

    assert _packet_source_count(
        {"derived_gaussian_cutout": {"source_count": 1_029_923}}
    ) == 1_029_923
    # A packet with no measured count keeps the original header fallback
    # rather than raising.
    assert _packet_source_count({}) is None
    assert _packet_source_count({"derived_gaussian_cutout": {}}) is None
    assert _packet_source_count(
        {"derived_gaussian_cutout": {"source_count": 0}}
    ) is None


def test_stage_one_disclosure_is_checked_against_the_decision() -> None:
    """The last raw-bytes literal, inside stage one's own adapter.

    It demanded ``raw_interiorgs_bytes is False`` on the very run whose rights
    authorize the upload -- and it fires at the *end* of stage one, after the
    render, both OpenAI stages and the full ArtiFixer3D train have been paid
    for.
    """

    import inspect

    from blueprint_pipeline import (
        task_evaluation_scene_configuration_builtin_adapters as adapters,
    )

    source = inspect.getsource(adapters)
    assert 'get(\n            "raw_interiorgs_bytes"\n        )\n        is not False' not in source
    assert "stage_requests_upload(configuration)" in source
    assert "renders_on_provider(" in source


def test_scene_configuration_transfer_budget_is_the_receipt_s_own_byte_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The declared ceiling must cover the bundle and pinned stage bootstrap.

    Vast prices inbound and outbound bytes per GB outside the hourly rate, so
    the byte ceiling is what lets ``_offer_fits_total_cost_bound`` price an
    offer's bandwidth against the attempt's compute cap at all. The production
    non-VIBE ArtiFixer branch still installs CUDA Torch; the Torch wheel and
    only five of its mandatory Linux wheels total 2,209,255,046 bytes before
    CUDA Toolkit components, Triton, torchvision, the rest of the Python
    closure, apt packages, and source fetches. The allowance must therefore be
    a conservative ceiling above that observed floor, not the old 2 GB value.
    """

    receipt = _build(tmp_path, "bundle")
    expected_upload = max(
        2 * receipt["bundle_size_bytes"],
        provider_artifacts.PROVIDER_OUTPUT_UPLOAD_MINIMUM_BYTES,
    )

    assert scene_vast._provider_transfer_byte_budget(receipt) == (
        receipt["bundle_size_bytes"]
        + provider_artifacts.PROVISIONING_DOWNLOAD_OVERHEAD_BYTES,
        expected_upload,
    )
    assert receipt["bundle_size_bytes"] > 0
    observed_pinned_wheel_floor = 2_209_255_046
    assert provider_artifacts.ARTIFIXER_PINNED_WHEEL_DOWNLOAD_FLOOR_BYTES == (
        observed_pinned_wheel_floor
    )
    assert provider_artifacts.PROVISIONING_DOWNLOAD_OVERHEAD_BYTES >= (
        4 * observed_pinned_wheel_floor
    )
    runtime_script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_public_scene_artifixer3d.sh"
    ).read_text(encoding="utf-8")
    assert "torch==2.11.0 torchvision==0.26.0" in runtime_script
    assert "export CUDA_HOME=/usr/local/cuda" not in runtime_script
    assert 'nvcc_path="$(command -v nvcc)"' in runtime_script
    assert 'export CUDA_HOME="$(cd "$(dirname "${nvcc_path}")/.." && pwd)"' in runtime_script
    assert 'grep -Fq "release 12.8"' in runtime_script
    assert "artifixer_cuda_package_paths.py" in runtime_script
    assert 'export CPATH="${cuda_package_paths[0]}' in runtime_script
    assert 'export CPLUS_INCLUDE_PATH="${cuda_package_paths[0]}' in runtime_script
    assert 'export LIBRARY_PATH="${cuda_package_paths[1]}' in runtime_script
    assert 'export LD_LIBRARY_PATH="${cuda_package_paths[1]}' in runtime_script

    for broken in ({}, {"bundle_size_bytes": 0}, {"bundle_size_bytes": True}):
        with pytest.raises(
            scene_vast.TaskEvaluationSceneConfigurationVastError,
            match="scene_configuration_provider_transfer_budget_inputs_invalid",
        ):
            scene_vast._provider_transfer_byte_budget(broken)
    monkeypatch.setattr(
        provider_artifacts,
        "PROVISIONING_DOWNLOAD_OVERHEAD_BYTES",
        observed_pinned_wheel_floor,
    )
    with pytest.raises(
        scene_vast.TaskEvaluationSceneConfigurationVastError,
        match="scene_configuration_provider_transfer_budget_underdeclared",
    ):
        scene_vast._provider_transfer_byte_budget(receipt)
    monkeypatch.setattr(
        provider_artifacts,
        "PROVISIONING_DOWNLOAD_OVERHEAD_BYTES",
        10_000_000_000,
    )
    monkeypatch.setattr(
        provider_artifacts, "PROVIDER_OUTPUT_UPLOAD_MINIMUM_BYTES", 1
    )
    with pytest.raises(
        scene_vast.TaskEvaluationSceneConfigurationVastError,
        match="scene_configuration_provider_transfer_budget_underdeclared",
    ):
        scene_vast._provider_transfer_byte_budget(receipt)


def test_scene_configuration_declares_its_transfer_budget_to_the_allocator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The paid allocator must receive the bytes this lane will actually move.

    ``run_vast_provider_adapter`` treats a zero byte ceiling as "do not price
    transfer": ``_offer_fits_total_cost_bound`` returns ``True`` before it
    looks at anything, and the selection receipt records no projected total.
    The lane therefore admitted offers whose bandwidth price alone could
    exceed the compute cap, and left no evidence that it had not. Passing the
    receipt's own byte count is what re-arms that projection, so this asserts
    on the kwargs the allocator is actually called with -- reverting the
    caller hunk turns this red at the two assertions below.
    """

    import types

    receipt = _build(tmp_path, "bundle")
    receipt_path = tmp_path / "bundle" / f"{BUNDLE_SCHEMA_VERSION}.receipt.json"
    authority_path = tmp_path / "authority.json"
    authority_path.write_text("{}", encoding="utf-8")
    authority = {
        "authority_digest": "sha256:" + "e" * 64,
        "hard_attempt_spend_cap_usd": MAX_ATTEMPT_SPEND_USD,
        "provider_compute_spend_cap_usd": MAX_PROVIDER_COMPUTE_SPEND_USD,
        "maximum_hourly_rate_usd": MAX_HOURLY_RATE_USD,
        "maximum_single_resource_ttl_seconds": REQUIRED_PARENT_TTL_SECONDS,
        "container_image": "nvcr.io/nvidia/isaac-sim@sha256:" + "b" * 64,
        "external_service_spend_caps": {
            "openai": {"maximum_cost_usd": 1.5, "maximum_requests": 32}
        },
    }
    monkeypatch.setattr(
        scene_vast,
        "validate_scene_configuration_paid_authority",
        lambda _value, **_kwargs: authority,
    )
    monkeypatch.setattr(
        scene_vast, "_provider_runtime_inputs", lambda _authority: ({}, {})
    )
    monkeypatch.setattr(
        scene_vast, "require_paid_resource_admission_grant", lambda *_a, **_k: None
    )

    def _stage(
        *,
        job_dir,
        bundle_path,
        key_prefix,
        expiration_seconds,
        retain_content_addressed_bundle,
    ):
        del key_prefix, expiration_seconds
        assert retain_content_addressed_bundle is True
        staging = Path(job_dir)
        staging.mkdir(parents=True, exist_ok=True)
        for name in (
            "provider_bundle_url.txt",
            "provider_output_put_url.txt",
            "provider_output_get_url.txt",
        ):
            (staging / name).write_text(
                f"https://objects.example.test/{name}", encoding="utf-8"
            )
        bundle = Path(bundle_path)
        return {
            "status": "completed",
            "provider_bundle_remote_reference": {
                "schema_version": (
                    "task_evaluation_scene_artifact_reference.v1"
                ),
                "status": "remote_verified",
                "artifact_kind": "provider-bundle",
                "uri": "s3://scene-artifacts/provider-bundle.zip",
                "digest": _sha256(bundle),
                "size_bytes": bundle.stat().st_size,
                "content_addressed_key": True,
                "remote_identity_verified": True,
                "full_byte_service_account_readback_passed": True,
                "raw_secret_values_recorded": False,
            },
        }

    monkeypatch.setattr(
        scene_vast, "stage_wam_provider_bundle_object_store", _stage
    )
    monkeypatch.setattr(
        scene_vast,
        "arm_independent_vast_watchdog",
        lambda **kwargs: (
            {"status": "armed"},
            types.SimpleNamespace(
                pod_name_prefix=kwargs["pod_name_prefix"] + "abc-",
                started_instance_id_path=Path(kwargs["job_dir"]) / "started.json",
                deadline_epoch=9_999_999_999.0,
            ),
        ),
    )
    monkeypatch.setattr(
        scene_vast,
        "close_independent_vast_watchdog",
        lambda **_kwargs: {"status": "cancelled_no_allocation"},
    )
    monkeypatch.setattr(
        scene_vast,
        "cleanup_staged_wam_provider_objects",
        lambda _root: {"all_objects_absent": True},
    )
    monkeypatch.setattr(
        scene_vast,
        "_consume_authority_once",
        lambda _authority, **_kwargs: {"status": "consumed"},
    )
    monkeypatch.setattr(
        scene_vast,
        "_stage_owner_only_runtime_secrets",
        lambda **_kwargs: ({}, None),
    )
    captured: dict[str, object] = {}

    def _adapter(**kwargs):
        captured.update(kwargs)
        return {"status": "blocked", "blockers": ["stubbed_no_allocation"]}

    monkeypatch.setattr(scene_vast, "run_vast_provider_adapter", _adapter)
    expected_download = (
        receipt["bundle_size_bytes"]
        + provider_artifacts.PROVISIONING_DOWNLOAD_OVERHEAD_BYTES
    )
    expected_upload = max(
        2 * receipt["bundle_size_bytes"],
        provider_artifacts.PROVIDER_OUTPUT_UPLOAD_MINIMUM_BYTES,
    )
    required_free = scene_vast._provider_output_disk_requirements(expected_upload)[
        "required_free_bytes_before_download"
    ]

    result = scene_vast.run_scene_configuration_vast(
        job_dir=tmp_path / "job",
        bundle_receipt_path=receipt_path,
        paid_attempt_authority_path=authority_path,
        paid_resource_admission_grant=object(),
        execute=True,
        allowed_machine_ids=(21899, 44762),
        scene_construction_queue_root=_construction_queue(tmp_path),
        disk_usage_provider=lambda _path: types.SimpleNamespace(free=required_free),
    )

    assert captured["expected_provider_download_bytes"] == expected_download
    assert captured["expected_provider_upload_bytes"] == expected_upload
    assert captured["provider_output_minimum_free_bytes"] == required_free
    assert captured["provider_runtime_environment"][
        "BLUEPRINT_VAST_EXPECTED_PROVIDER_UPLOAD_BYTES"
    ] == str(expected_upload)
    assert captured["provider_runtime_environment"][PARENT_DEADLINE_EPOCH_ENV] == str(
        9_999_999_999.0
    )
    provider_all_in_cap = MAX_ATTEMPT_SPEND_USD - 1.5
    assert provider_all_in_cap + 1.5 == MAX_ATTEMPT_SPEND_USD
    assert captured["hard_cap_usd"] == provider_all_in_cap
    assert captured["target_spend_usd"] == provider_all_in_cap
    assert captured["max_live_minutes"] == 420
    assert captured["session_max_live_minutes"] == 420
    assert captured["allowed_machine_ids"] == (21899, 44762)
    assert result["expected_provider_download_bytes"] == expected_download
    assert result["expected_provider_upload_bytes"] == expected_upload


def test_scene_configuration_refuses_insufficient_output_disk_before_staging_or_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A paid run cannot begin when its declared return envelope cannot fit."""

    receipt = _build(tmp_path, "disk-capacity-bundle")
    receipt_path = (
        tmp_path
        / "disk-capacity-bundle"
        / f"{BUNDLE_SCHEMA_VERSION}.receipt.json"
    )
    authority_path = tmp_path / "disk-capacity-authority.json"
    authority_path.write_text("{}", encoding="utf-8")
    authority = {
        "authority_digest": "sha256:" + "e" * 64,
        "hard_attempt_spend_cap_usd": MAX_ATTEMPT_SPEND_USD,
        "provider_compute_spend_cap_usd": MAX_PROVIDER_COMPUTE_SPEND_USD,
        "maximum_hourly_rate_usd": MAX_HOURLY_RATE_USD,
        "maximum_single_resource_ttl_seconds": REQUIRED_PARENT_TTL_SECONDS,
        "container_image": "nvcr.io/nvidia/isaac-sim@sha256:" + "b" * 64,
        "external_service_spend_caps": {
            "openai": {"maximum_cost_usd": 1.5, "maximum_requests": 32}
        },
    }
    monkeypatch.setattr(
        scene_vast,
        "validate_scene_configuration_paid_authority",
        lambda _value, **_kwargs: authority,
    )
    monkeypatch.setattr(
        scene_vast, "_provider_runtime_inputs", lambda _authority: ({}, {})
    )
    monkeypatch.setattr(
        scene_vast, "require_paid_resource_admission_grant", lambda *_a, **_k: None
    )
    calls = {"staging": 0, "watchdog": 0, "adapter": 0}

    def must_not_stage(**_kwargs):
        calls["staging"] += 1
        raise AssertionError("object-store staging crossed disk refusal")

    def must_not_arm(**_kwargs):
        calls["watchdog"] += 1
        raise AssertionError("watchdog crossed disk refusal")

    def must_not_allocate(**_kwargs):
        calls["adapter"] += 1
        raise AssertionError("provider adapter crossed disk refusal")

    monkeypatch.setattr(
        scene_vast, "stage_wam_provider_bundle_object_store", must_not_stage
    )
    monkeypatch.setattr(scene_vast, "arm_independent_vast_watchdog", must_not_arm)
    monkeypatch.setattr(scene_vast, "run_vast_provider_adapter", must_not_allocate)
    _download, upload = scene_vast._provider_transfer_byte_budget(receipt)
    requirement = scene_vast._provider_output_disk_requirements(upload)[
        "required_free_bytes_before_download"
    ]

    result = scene_vast.run_scene_configuration_vast(
        job_dir=tmp_path / "disk-capacity-job",
        bundle_receipt_path=receipt_path,
        paid_attempt_authority_path=authority_path,
        paid_resource_admission_grant=object(),
        execute=True,
        scene_construction_queue_root=_construction_queue(tmp_path),
        disk_usage_provider=lambda _path: type(
            "Usage", (), {"free": requirement - 1}
        )(),
    )

    assert result["status"] == "blocked"
    assert result["provider_mutations_performed"] == 0
    assert result["continuing_spend_from_this_run"] is False
    assert result["blockers"] == [
        "scene_configuration_provider_output_disk_capacity_insufficient"
    ]
    assert result["provider_output_disk_capacity"]["phase"] == (
        "before_allocation_and_staging"
    )
    assert result["provider_output_disk_capacity"]["observed_free_bytes"] == (
        requirement - 1
    )
    assert calls == {"staging": 0, "watchdog": 0, "adapter": 0}


def test_scene_configuration_bundle_bytes_can_price_an_offer_out_of_the_cap(
    tmp_path: Path,
) -> None:
    """Proves the wired budget has a live admission consequence.

    With a zero upload ceiling the bound omits the provider's outbound rate
    and admits the offer; with the exact enforced ceiling the same offer is
    refused before allocation.
    """

    receipt = _build(tmp_path, "bundle")
    download_bytes, upload_bytes = scene_vast._provider_transfer_byte_budget(receipt)
    upload_gb = upload_bytes / 1_000_000_000.0
    offer = {
        "hourly_rate_usd": 0.40,
        "provider_download_cost_per_gb_usd": 0.0,
        "provider_upload_cost_per_gb_usd": 1.0 / upload_gb,
    }

    assert vpa._offer_fits_total_cost_bound(
        offer,
        hard_cap_usd=0.75,
        max_live_minutes=30,
        expected_provider_download_bytes=download_bytes,
        expected_provider_upload_bytes=0,
    )
    assert not vpa._offer_fits_total_cost_bound(
        offer,
        hard_cap_usd=0.75,
        max_live_minutes=30,
        expected_provider_download_bytes=download_bytes,
        expected_provider_upload_bytes=upload_bytes,
    )


#: Third-party packages the Isaac Sim provider image is *observed* to supply.
#: Both are in the import closure that ran to completion on instance 48833365
#: (2026-08-27T00:30Z) before the runner died further down the same chain, so
#: this is measured availability rather than an assumption about the image.
PROVIDER_RUNTIME_AVAILABLE_THIRD_PARTY = frozenset({"PIL", "numpy"})


def _provider_runner_module_scope_third_party(
    repo: Path,
) -> tuple[dict[str, set[str]], set[str]]:
    """Third-party packages the provider runner needs at *import* time.

    Walks only imports that execute when a module is imported -- module body,
    including ``try``/``if`` blocks, and excluding function bodies -- because
    those are the ones that can kill the runner before its first stage. A name
    that resolves to a file in ``src/blueprint_pipeline`` is first-party under
    either the package or the flat provider-bundle layout.
    """

    import ast
    import sys

    package_root = repo / "src" / "blueprint_pipeline"
    stdlib = set(sys.stdlib_module_names)

    def is_first_party(name: str) -> bool:
        return (package_root / f"{name}.py").is_file() or (
            package_root / name / "__init__.py"
        ).is_file()

    def module_scope_imports(tree: ast.Module) -> list[ast.stmt]:
        found: list[ast.stmt] = []

        def walk(body: list[ast.stmt]) -> None:
            for node in body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    found.append(node)
                elif isinstance(node, (ast.Try, ast.If)):
                    walk(node.body)
                    for handler in getattr(node, "handlers", []):
                        walk(handler.body)
                    walk(node.orelse)
                    walk(getattr(node, "finalbody", []))

        walk(tree.body)
        return found

    def imported_names(node: ast.stmt) -> list[str]:
        if isinstance(node, ast.Import):
            return [alias.name.split(".")[0] for alias in node.names]
        assert isinstance(node, ast.ImportFrom)
        if node.level:
            if node.module:
                return [node.module.split(".")[0]]
            return [alias.name for alias in node.names]
        if not node.module:
            return []
        top = node.module.split(".")[0]
        if top == "blueprint_pipeline" and "." in node.module:
            return [node.module.split(".")[1]]
        return [top]

    third_party: dict[str, set[str]] = {}
    pending = ["__runner__"]
    seen: set[str] = set()
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        if module == "__runner__":
            path = repo / "scripts" / "task_evaluation_scene_configuration_provider_runner.py"
        else:
            path = package_root / f"{module}.py"
            if not path.is_file():
                path = package_root / module / "__init__.py"
                if not path.is_file():
                    continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in module_scope_imports(tree):
            for name in imported_names(node):
                if name == "blueprint_pipeline":
                    continue
                if is_first_party(name):
                    pending.append(name)
                elif name not in stdlib:
                    third_party.setdefault(name, set()).add(module)
    return third_party, seen


def test_provider_runner_imports_nothing_the_provider_image_lacks() -> None:
    """The runner must not need a package the rented GPU does not have.

    Run ``adp-new-scene-simple-relocation-839873-a099de6a-r2-web-20260827T002140Z``
    rented an RTX 4090, downloaded the bundle, started the entrypoint, and died
    at ``provider_runtime/blueprint_pipeline/task_evaluation_configured_scene_revision.py``
    line 11 with ``ModuleNotFoundError: No module named 'jsonschema'`` --
    ``first_stage_started: false``, one provider mutation, one consumed
    attempt authority, and nothing configured.

    The provider never validates anything against a JSON Schema. It reached
    that module only because the stage adapters import the orchestrator for
    one string constant, ``STAGE_RESULT_SCHEMA_VERSION``. So the fix is to
    keep ``jsonschema`` out of import time, and the contract is this: whatever
    the runner needs before its first stage must be a package the provider
    image is observed to supply.
    """

    repo = Path(__file__).resolve().parents[1]
    third_party, reached_modules = _provider_runner_module_scope_third_party(repo)
    unavailable = {
        name: sorted(via)
        for name, via in third_party.items()
        if name not in PROVIDER_RUNTIME_AVAILABLE_THIRD_PARTY
    }

    assert unavailable == {}, (
        "provider runner import closure needs packages the Isaac Sim image is "
        f"not observed to supply: {unavailable}"
    )
    # The closure must still reach the actual provider runtime and both stage
    # registries, or the assertion above would pass by walking nothing. The
    # scene-specific static gate intentionally removed the generic validator's
    # eager numpy/PIL dependency, so an empty third-party closure is valid.
    assert {
        "task_evaluation_scene_configuration_provider_runtime",
        "task_evaluation_scene_configuration_builtin_adapters",
        "task_evaluation_scene_configuration_builtin_producers",
        "task_evaluation_scene_configuration_static_qualification",
    } <= reached_modules


def test_scene_configuration_validators_still_enforce_their_schemas() -> None:
    """Deferring the import must not defer the refusal."""

    from blueprint_pipeline import task_evaluation_configured_scene_revision as revision
    from blueprint_pipeline import task_evaluation_scene_construction_recipe as recipe
    from blueprint_pipeline import (
        task_evaluation_launch_preparation_contract as preparation,
    )

    with pytest.raises(revision.TaskEvaluationConfiguredSceneRevisionError):
        revision.validate_configured_scene_revision({"schema_version": "wrong"})
    with pytest.raises(recipe.TaskEvaluationSceneConstructionRecipeError):
        recipe.validate_scene_construction_recipe({"schema_version": "wrong"})
    with pytest.raises(preparation.TaskEvaluationLaunchPreparationContractError):
        preparation.validate_launch_preparation_request({"schema_version": "wrong"})


def test_provider_runner_imports_in_the_bundle_layout_without_the_repo_tree(
    tmp_path: Path,
) -> None:
    """Import the runner the way the rented GPU does, not a proxy for it.

    The provider bundle copies ``src/blueprint_pipeline`` to
    ``provider_runtime/blueprint_pipeline`` and carries no repository
    ``docs/``, ``scripts/`` or ``src/``. Two paid runs died on that difference
    and nothing else: first ``ModuleNotFoundError: No module named
    'jsonschema'``, then ``image_editor_registry_unreadable`` from a
    module-scope registry read whose ``parents[2]`` default resolves inside
    the bundle root, where the repository's ``docs/`` tree has never existed.

    A static import scan cannot see the second one -- the package is present,
    the *file it reads at import time* is not. So this actually performs the
    import, in that layout, in a subprocess whose only path entry is the
    bundle's runtime root.
    """

    repo = Path(__file__).resolve().parents[1]
    runtime = tmp_path / "task_evaluation_scene_configuration_provider_bundle" / "provider_runtime"
    runtime.mkdir(parents=True)
    shutil.copytree(
        repo / "src" / "blueprint_pipeline",
        runtime / "blueprint_pipeline",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copyfile(
        repo / "scripts" / "task_evaluation_scene_configuration_provider_runner.py",
        runtime / "task_evaluation_scene_configuration_provider_runner.py",
    )
    assert not (runtime.parent / "docs").exists()
    assert not (runtime.parent / "src").exists()

    # Every module the provider actually starts a process on. The stage
    # drivers are *not* in the runner's import closure -- the stage tool
    # execs each component's own ``run`` script, which does
    # ``python -m blueprint_pipeline.<driver>`` -- so importing only the
    # runner would have missed the registry read that killed stage 1.
    entrypoints = (
        "task_evaluation_scene_configuration_provider_runner",
        "blueprint_pipeline.task_evaluation_scene_configuration_stage_tool",
        "blueprint_pipeline.task_evaluation_scene_configuration_artifixer_driver",
        "blueprint_pipeline.task_evaluation_scene_configuration_content_agents_driver",
        "blueprint_pipeline.task_evaluation_scene_configuration_native_import_driver",
    )
    failures: dict[str, str] = {}
    for module in entrypoints:
        completed = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=tmp_path,
            env={
                "PYTHONPATH": str(runtime),
                "PATH": os.environ.get("PATH", ""),
                "HOME": str(tmp_path),
            },
            capture_output=True,
            text=True,
            timeout=600,
        )
        if completed.returncode != 0:
            failures[module] = completed.stderr[-2000:]

    assert not failures, "provider entrypoints fail to import in the bundle layout:\n" + "\n\n".join(
        f"### {module}\n{detail}" for module, detail in failures.items()
    )


def test_scene_configuration_boundary_provisions_nested_artifixer_nvcc() -> None:
    """The parent must provision the compiler demanded by nested ArtiFixer.

    Run ``adp-new-scene-simple-relocation-839873-7c0ec0c0-r2-web-20260827T015257Z``
    rented an RTX A6000, passed the Isaac smoke, and refused with
    ``BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:scene_configuration_runtime_toolchain_missing:127``
    before downloading the bundle because the image did not contain ``nvcc``.
    Removing that command from the parent check only deferred the same refusal:
    the shipped stage-1 script imports 3DGRUT's CUDA JIT graph and has its own
    correct ``nvcc`` preflight. The parent therefore installs a version-pinned
    CUDA compiler through a digest-pinned NVIDIA repository keyring, then keeps
    deriving its required commands from the packages it installs.
    """

    provided = set(vpa.SCENE_CONFIGURATION_SHIMMED_COMMANDS).union(
        *vpa.SCENE_CONFIGURATION_PROVISIONED_COMMANDS.values()
    )
    required = set(vpa.SCENE_CONFIGURATION_REQUIRED_COMMANDS.split())

    assert required, "the boundary check must verify something"
    assert required <= provided, (
        "onstart requires commands it does not provision: "
        f"{sorted(required - provided)}"
    )
    assert "nvcc" in required
    # Every package named for the apt line is actually installed by it.
    script = vpa._probe_shell_script(
        "https://heartbeat.example.test",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        provider_bundle_kind="task_evaluation_scene_configuration",
    )
    apt_packages = script.split(" install -y ", 1)[1].split(" >", 1)[0].split()
    for package in vpa.SCENE_CONFIGURATION_PROVISIONED_COMMANDS:
        assert package in apt_packages
    assert vpa.SCENE_CONFIGURATION_CUDA_NVCC_PACKAGE in apt_packages
    assert vpa.SCENE_CONFIGURATION_CUDA_KEYRING_URL in script
    assert vpa.SCENE_CONFIGURATION_CUDA_KEYRING_SHA256 in script
    assert "/usr/local/cuda-12.8/bin" in script
    assert script.index(vpa.SCENE_CONFIGURATION_CUDA_KEYRING_URL) < script.index(
        "dpkg -i"
    ) < script.index(" install -y ") < script.index("for required_command in")

    nested_stage = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_public_scene_artifixer3d.sh"
    ).read_text(encoding="utf-8")
    assert (
        'required_commands = ("git", "gcc", "g++", "cmake", "ninja", "nvcc", "slangc")'
        in nested_stage
    )


def test_scene_configuration_boundary_selects_supported_artifixer_host_compiler() -> None:
    """The Ubuntu 24.04 parent must not compile 3DGRUT with its default GCC 13.

    The exact upstream 3DGRUT-ArtiFixer commit says GCC versions above 11 are
    unsupported and directs Ubuntu 24.04 callers to install GCC/G++ 11.  The
    standalone ArtiFixer lane already does that, but the scene-configuration
    parent installed only ``build-essential`` before entering the same CUDA/JIT
    graph.  Select and read back the supported compiler before the bundle is
    downloaded, so a packaging regression refuses before the expensive stages.
    """

    script = vpa._probe_shell_script(
        "https://heartbeat.example.test",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        provider_bundle_kind="task_evaluation_scene_configuration",
    )
    apt_packages = script.split(" install -y ", 1)[1].split(" >", 1)[0].split()

    assert {"gcc-11", "g++-11"} <= set(apt_packages)
    assert vpa.SCENE_CONFIGURATION_HOST_COMPILER_MAJOR == "11"
    assert 'exec /usr/bin/gcc-11 "$@"' in script
    assert 'exec /usr/bin/g++-11 "$@"' in script
    assert "CC=/usr/bin/gcc-11" in script
    assert "CXX=/usr/bin/g++-11" in script
    assert "CUDAHOSTCXX=/usr/bin/g++-11" in script
    assert "gcc -dumpfullversion -dumpversion" in script
    assert "g++ -dumpfullversion -dumpversion" in script
    assert (
        "BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:"
        "scene_configuration_artifixer_host_compiler_invalid:$compiler_rc"
        in script
    )
    assert script.index("gcc-11") < script.index("blueprint_download_url \"$BUNDLE_URL\"")


def test_provider_entrypoint_preserves_distinct_standalone_and_isaac_paths() -> None:
    """The pre-Kit and Kit-started processes require distinct USD bindings.

    Run ``adp-new-scene-simple-relocation-839873-80fd006f-r2-web-20260827T024309Z``
    rented an RTX GPU, cleared the toolchain gate, built the stage Python
    runtime, and refused at the import preflight with
    ``ModuleNotFoundError: No module named 'pxr'`` --
    ``scene_configuration_provider_stage_import_closure_invalid``, exit 86,
    ``first_stage_started: false``.

    Preserving the inherited path did not clear the same production blocker:
    bare ``/isaac-sim/python.sh`` does not expose pxr until Kit starts. The
    parent therefore needs the sealed standalone usd-core wheel, while the
    native component must retain this original baseline so it can exclude the
    standalone wheel before SimulationApp starts Kit.
    """

    repo = Path(__file__).resolve().parents[1]
    script = (
        repo / "scripts" / "run_task_evaluation_scene_configuration_provider.sh"
    ).read_text(encoding="utf-8")

    baseline = (
        'export BLUEPRINT_SCENE_CONFIGURATION_ISAAC_PYTHONPATH="${PYTHONPATH-}"'
    )
    assert baseline in script
    assert script.index(baseline) < script.index("export PYTHONPATH=")

    exports = [
        line.strip()
        for line in script.splitlines()
        if line.strip().startswith("export PYTHONPATH=")
    ]
    assert exports, "entrypoint no longer sets PYTHONPATH; update this contract"
    discarding = [line for line in exports if "${PYTHONPATH:+" not in line]
    assert not discarding, (
        "these exports replace the image's PYTHONPATH instead of appending to "
        f"it, which removes Isaac's own runtime: {discarding}"
    )


def test_scene_configuration_onstart_provisions_the_bundled_browser_libraries() -> None:
    """The renderer ships a browser binary, not its loader dependencies.

    Run ``adp-new-scene-simple-relocation-839873-679542d9-r2-web-20260827T034953Z``
    rented a GPU, cleared the import preflight, started stage 1, and the
    bundled Chromium died before its first frame:

        chrome: error while loading shared libraries: libnspr4.so:
        cannot open shared object file: No such file or directory
        <process did exit: exitCode=127>

    The provider bundle carries the browser and its Node runtime; the shared
    libraries it links against come from the image, and the Isaac image does
    not have them. These are Playwright's published Chromium dependencies for
    Ubuntu 24.04, which is what the container reports in its own apt sources.
    None provides a command, so the onstart's command check cannot verify them
    -- the renderer's own launch is the proof, and this pins that they are at
    least requested.
    """

    required_browser_packages = {
        "libnspr4",
        "libnss3",
        "libatk1.0-0t64",
        "libatk-bridge2.0-0t64",
        "libatspi2.0-0t64",
        "libcups2t64",
        "libdrm2",
        "libxcomposite1",
        "libxdamage1",
        "libxfixes3",
        "libxrandr2",
        "libxkbcommon0",
        "libgbm1",
        "libpango-1.0-0",
        "libcairo2",
        "libasound2t64",
    }
    provisioned = set(vpa.SCENE_CONFIGURATION_PROVISIONED_COMMANDS)
    missing = sorted(required_browser_packages - provisioned)
    assert not missing, (
        "the onstart does not install these Chromium runtime libraries, so the "
        f"bundled browser cannot load: {missing}"
    )

    script = vpa._probe_shell_script(
        "https://heartbeat.example.test",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        provider_bundle_kind="task_evaluation_scene_configuration",
    )
    installed = set(
        script.split(" install -y ", 1)[1].split(" >", 1)[0].split()
    )
    assert required_browser_packages <= installed
