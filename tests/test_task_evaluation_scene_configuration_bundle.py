from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_builtin_producers import (
    TOOLCHAIN_SCHEMA_VERSION,
)
from blueprint_pipeline.task_evaluation_scene_configuration_bundle import (
    BUNDLE_SCHEMA_VERSION,
    build_scene_configuration_provider_bundle,
)
from blueprint_pipeline.task_evaluation_scene_configuration_stage_producers import (
    ADMITTED_PRODUCER_IDENTITIES,
)
from blueprint_pipeline.task_evaluation_scene_construction_queue import (
    ENVELOPE_SCHEMA_VERSION,
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
    stages: dict[str, dict[str, object]] = {}
    files = []
    for identity in ADMITTED_PRODUCER_IDENTITIES:
        executable = root / "stages" / identity.adapter_id
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
        executable.chmod(0o555)
        stages[identity.adapter_id] = {
            "entrypoint": executable.relative_to(root).as_posix(),
            "network_policy": (
                "disabled"
                if identity.adapter_id == "simready_native_import_qualification"
                else "provider_and_openai_api"
            ),
            "secrets_via_files_only": True,
            "raw_secret_values_in_argv_or_logs": False,
        }
        files.append(
            {
                "relative_path": executable.relative_to(root).as_posix(),
                "sha256": _sha256(executable),
                "size_bytes": executable.stat().st_size,
                "executable": True,
            }
        )
    manifest = {
        "schema_version": TOOLCHAIN_SCHEMA_VERSION,
        "status": "published_full_byte_readback_passed",
        "source_commit": commit,
        "full_byte_service_account_readback_passed": True,
        "stages": stages,
        "files": files,
        "toolchain_digest": "",
    }
    manifest["toolchain_digest"] = canonical_digest(
        manifest, digest_field="toolchain_digest"
    )
    manifest_path = root / f"{TOOLCHAIN_SCHEMA_VERSION}.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() or path != manifest_path else 0o444)
    root.chmod(0o555)
    return root


def _repo(root: Path) -> Path:
    package = root / "src/blueprint_pipeline"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("\n", encoding="utf-8")
    scripts = root / "scripts"
    scripts.mkdir()
    source_root = Path(__file__).resolve().parents[1]
    for name in (
        "run_task_evaluation_scene_configuration_provider.sh",
        "task_evaluation_scene_configuration_provider_runner.py",
    ):
        (scripts / name).write_bytes((source_root / "scripts" / name).read_bytes())
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
    frame.write_bytes(b"derived-lossless-frame")
    render = {
        "schema_version": "task_evaluation_scene_configuration_render_inputs.v1",
        "status": "derived_method_inputs_materialized",
        "run_id": "configure-scene-839873-v1",
        "source_splat_digest": _sha256(raw_splat),
        "raw_interiorgs_bytes_in_provider_packet": False,
        "camera_calibration": _bound(cameras),
        "render_manifest": _bound(render_manifest),
        "derived_frames": [_bound(frame, camera_id="camera-0")],
        "derived_frame_count": 1,
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
    assert str(tmp_path).encode() not in payloads[
        "provider_runtime/input/portable_construction_envelope.v1.json"
    ]
    assert "provider_runtime/input/references/0001.usda" in names


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
    assert all(
        Path(row["materialized_path"]).is_file()
        for row in hydrated["materialized_references"]
    )
    assert all(
        Path(row["path"]).is_file()
        for row in hydrated["render_inputs_result"]["derived_frames"]
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
