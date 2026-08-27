from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
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
from blueprint_pipeline.task_evaluation_scene_construction_queue import (
    ENVELOPE_SCHEMA_VERSION,
)
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
    sage = inputs / "sage.usda"
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
        "run_id": "configure-scene-839873-v1",
        "expected_production_commit": commit,
        "recipe": {"stage_sequence": stages},
        "materialized_references": [
            _bound(
                raw_splat,
                contract_path="scene.appearance.representation",
            ),
            _bound(sage, contract_path="scene.geometry.collision"),
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
            "runtime_digest": "sha256:" + "9" * 64,
            "source_commit": commit,
        },
    }
    return runtime, identity


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
    assert "provider_runtime/input/references/0001.usda" in names


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
    assert "timeout 300 apt-get" in script
    assert script.index(" install -y curl wget unzip git build-essential") < script.index(
        "BLUEPRINT_VAST_PROVIDER_BUNDLE_DOWNLOADED"
    )
    assert (
        "for required_command in python3 git gcc g++ cmake ninja nvcc Xvfb; "
        'do command -v "$required_command"'
    ) in script
    assert 'exec /isaac-sim/python.sh "$@"' in script
    assert 'export PATH="/usr/local/cuda/bin:$PATH"' in script
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
    authority = authority_module.materialize_scene_configuration_paid_authority(
        bundle_receipt_path=receipt_path,
        project_spend_reconciliation_path=project_path,
        initial_provider_zero_path=zero_path,
        authorization_reference="user-authorized-new-scene-gpu",
        authorized_by="project-owner",
        authorized_on="2026-08-25T12:05:00Z",
        source_commit="a" * 40,
        container_image="nvcr.io/nvidia/isaac-sim@sha256:" + "b" * 64,
        resource_name=(
            "adp-new-scene-simple-relocation-839873-aaaaaaaaaaaa-20260825t120500z"
        ),
        max_hourly_rate_usd=0.50,
        hard_cap_usd=2.25,
        hard_ttl_seconds=1_800,
        output_path=authority_path,
        provider_compute_spend_cap_usd=0.75,
        openai_max_cost_usd=1.5,
        openai_max_requests=32,
        openai_artifixer_semantic_teacher_max_cost_usd=0.4,
        openai_artifixer_visual_review_max_cost_usd=0.75,
        openai_content_agents_max_cost_usd=0.35,
    )

    assert authority["retry_cap"] == 0
    # Prior spend remains reconciled and receipt-bound, but it no longer
    # creates an unrelated lifetime ceiling for an explicitly authorized,
    # tightly bounded one-shot attempt.
    assert authority["aggregate_goal_spend_before_attempt_usd"] == 500.0
    assert "aggregate_goal_spend_cap_usd" not in authority
    assert authority["hard_attempt_spend_cap_usd"] == 2.25
    assert authority["provider_compute_spend_cap_usd"] == 0.75
    assert authority["maximum_provider_allocations"] == 1
    assert authority["maximum_automatic_retries"] == 0
    assert authority["external_service_spend_caps"]["openai"][
        "maximum_cost_usd"
    ] == 1.5
    assert authority_module.validate_scene_configuration_paid_authority(
        authority, bundle_receipt=receipt
    ) == authority
    for name, value in (
        ("OPENAI_ADMIN_API_KEY_FILE", "test-admin-key"),
        ("OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_FILE", "key-semantic"),
        ("OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE", "key-review"),
        ("OPENAI_CONTENT_AGENTS_API_KEY_FILE", "key-content-agents"),
        (
            "BLUEPRINT_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_COST_SCOPE_ATTESTATION_FILE",
            '{"schema_version":"openai_candidate_cost_scope_attestation.v1"}',
        ),
        (
            "BLUEPRINT_OPENAI_ARTIFIXER_VISUAL_REVIEW_COST_SCOPE_ATTESTATION_FILE",
            '{"schema_version":"openai_candidate_cost_scope_attestation.v1"}',
        ),
        (
            "BLUEPRINT_OPENAI_CONTENT_AGENTS_COST_SCOPE_ATTESTATION_FILE",
            '{"schema_version":"openai_candidate_cost_scope_attestation.v1"}',
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
        hard_ttl_seconds=1_800,
        max_spend_usd=2.25,
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
        hard_ttl_seconds=1_800,
        max_spend_usd=2.25,
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


def test_scene_configuration_provider_output_requires_complete_six_stage_chain(
    tmp_path: Path,
) -> None:
    assert vpa._provider_expected_video_count(
        "task_evaluation_scene_configuration"
    ) == 0
    chain = {
        "schema_version": "task_evaluation_scene_configuration_provider_stage_chain.v1",
        "status": "completed",
        "stage_results": [{"stage_id": f"stage-{index}"} for index in range(6)],
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
        "source_commit": "a" * 40,
        "construction_envelope_digest": "sha256:" + "b" * 64,
        "stage_chain": chain,
        "evaluation_episode_executed": False,
        "candidate_policy_queried": False,
        "provider_zero_required_after_return": True,
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
        archive, tmp_path / "extracted-output"
    )
    assert blockers == []
    assert observed["stage_chain"]["stage_count"] == 6


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
    result_index = source.find('"blockers": sorted(set(blockers))')
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
    tmp_path: Path,
) -> None:
    """The declared download ceiling must be measured, not guessed.

    Vast prices inbound and outbound bytes per GB outside the hourly rate, so
    the byte ceiling is what lets ``_offer_fits_total_cost_bound`` price an
    offer's bandwidth against the attempt's compute cap at all.
    """

    receipt = _build(tmp_path, "bundle")

    assert scene_vast._provider_transfer_byte_budget(receipt) == (
        receipt["bundle_size_bytes"],
        0,
    )
    assert receipt["bundle_size_bytes"] > 0

    for broken in ({}, {"bundle_size_bytes": 0}, {"bundle_size_bytes": True}):
        with pytest.raises(
            scene_vast.TaskEvaluationSceneConfigurationVastError,
            match="scene_configuration_provider_transfer_budget_inputs_invalid",
        ):
            scene_vast._provider_transfer_byte_budget(broken)


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
        "provider_compute_spend_cap_usd": 0.75,
        "maximum_hourly_rate_usd": 0.50,
        "maximum_single_resource_ttl_seconds": 1_800,
        "container_image": "nvcr.io/nvidia/isaac-sim@sha256:" + "b" * 64,
        "external_service_spend_caps": {
            "openai": {"maximum_cost_usd": 0.0, "maximum_requests": 0}
        },
    }
    monkeypatch.setattr(
        scene_vast,
        "validate_scene_configuration_paid_authority",
        lambda _value, **_kwargs: authority,
    )
    monkeypatch.setattr(
        scene_vast, "require_paid_resource_admission_grant", lambda *_a, **_k: None
    )

    def _stage(*, job_dir, bundle_path, key_prefix, expiration_seconds):
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

    result = scene_vast.run_scene_configuration_vast(
        job_dir=tmp_path / "job",
        bundle_receipt_path=receipt_path,
        paid_attempt_authority_path=authority_path,
        paid_resource_admission_grant=object(),
        execute=True,
    )

    assert captured["expected_provider_download_bytes"] == receipt["bundle_size_bytes"]
    assert captured["expected_provider_upload_bytes"] == 0
    # The cap the caller hands the allocator is still the authority's own,
    # unwidened by the transfer accounting.
    assert captured["hard_cap_usd"] == authority["provider_compute_spend_cap_usd"]
    assert captured["target_spend_usd"] == authority["provider_compute_spend_cap_usd"]
    assert result["expected_provider_download_bytes"] == receipt["bundle_size_bytes"]
    assert result["expected_provider_upload_bytes"] == 0


def test_scene_configuration_bundle_bytes_can_price_an_offer_out_of_the_cap(
    tmp_path: Path,
) -> None:
    """Proves the wired budget has a live admission consequence.

    With a zero ceiling the bound short-circuits to ``True`` for the very
    offer whose bandwidth alone blows the cap; with the bundle's real byte
    count the same offer is refused.
    """

    receipt = _build(tmp_path, "bundle")
    download_bytes, upload_bytes = scene_vast._provider_transfer_byte_budget(receipt)
    gb = download_bytes / 1_000_000_000.0
    offer = {
        "hourly_rate_usd": 0.40,
        "provider_download_cost_per_gb_usd": 1.0 / gb,
        "provider_upload_cost_per_gb_usd": 0.0,
    }

    assert vpa._offer_fits_total_cost_bound(
        offer,
        hard_cap_usd=0.75,
        max_live_minutes=30,
        expected_provider_download_bytes=0,
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
