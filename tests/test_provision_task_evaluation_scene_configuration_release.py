from __future__ import annotations

import hashlib
import io
from pathlib import Path

from scripts import provision_task_evaluation_scene_configuration_release as subject


def test_materializes_digest_bound_vgg16_once_without_provider_download(
    tmp_path: Path, monkeypatch
) -> None:
    body = b"pinned-vgg16-fixture"
    monkeypatch.setattr(subject, "VGG16_WEIGHTS_SIZE_BYTES", len(body))
    monkeypatch.setattr(
        subject,
        "VGG16_WEIGHTS_SHA256",
        "sha256:" + hashlib.sha256(body).hexdigest(),
    )
    calls = []

    def open_fixture(request, *, timeout):
        calls.append((request.full_url, timeout))
        return io.BytesIO(body)

    monkeypatch.setattr(subject.urllib.request, "urlopen", open_fixture)
    cache = tmp_path / "cache"
    first = subject.materialize_vgg16_weights(cache_root=cache)
    second = subject.materialize_vgg16_weights(cache_root=cache)

    assert first == second
    assert first.read_bytes() == body
    assert calls == [(subject.VGG16_WEIGHTS_SOURCE_URL, 1800)]


def test_release_provisioner_builds_scene_neutral_runtime_and_all_components(
    tmp_path: Path, monkeypatch
) -> None:
    commit = "a" * 40
    calls: list[tuple[str, dict]] = []

    def record(name: str):
        def build(**kwargs):
            calls.append((name, kwargs))
            return {"status": "published", f"{name}_digest": "sha256:" + "1" * 64}

        return build

    monkeypatch.setattr(subject, "build_published_splat_render_runtime", record("splat"))
    monkeypatch.setattr(
        subject, "build_artifixer_scene_configuration_component", record("artifixer")
    )
    monkeypatch.setattr(
        subject,
        "build_content_agents_scene_configuration_component",
        record("content_agents"),
    )
    monkeypatch.setattr(
        subject,
        "build_native_import_scene_configuration_component",
        record("native_import"),
    )
    monkeypatch.setattr(
        subject, "build_published_scene_configuration_toolchain", record("toolchain")
    )
    vgg16_weights = tmp_path / "vgg16-397923af.pth"
    vgg16_weights.write_bytes(b"fixture")
    monkeypatch.setattr(
        subject,
        "materialize_vgg16_weights",
        lambda **_kwargs: vgg16_weights,
    )

    result = subject.provision_scene_configuration_release(
        repository_root=tmp_path / "release",
        source_commit=commit,
        runtime_root=tmp_path / "system-runtimes",
        node_executable=tmp_path / "node",
        browser_root=tmp_path / "chromium",
        browser_executable=tmp_path / "chrome",
        node_modules_root=tmp_path / "node_modules",
        artifixer_root=tmp_path / "artifixer-source",
        content_agents_root=tmp_path / "content-agents-source",
        readback=lambda path: path.read_bytes(),
        readback_actor="service-account:blueprint",
    )

    assert [name for name, _kwargs in calls] == [
        "splat",
        "artifixer",
        "content_agents",
        "native_import",
        "toolchain",
    ]
    assert calls[1][1]["vgg16_weights_path"] == vgg16_weights
    assert result["status"] == "ready"
    assert result["scene_specific_artifacts_built"] is False
    assert result["provider_mutation_performed"] is False
    assert result["paid_resource_allocated"] is False
    assert result["environment"] == {
        "BLUEPRINT_TASK_EVALUATION_SPLAT_RENDER_RUNTIME_ROOT": str(
            tmp_path / "system-runtimes/splat-render" / commit
        ),
        "BLUEPRINT_TASK_EVALUATION_SCENE_CONFIGURATION_TOOLCHAIN_ROOT": str(
            tmp_path / "system-runtimes/scene-configuration" / commit
        ),
        "BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_RELEASE_WINDOW_PREFIX": (
            "s3://blueprint/task-evaluation/production-inputs/coordinator-release-windows/"
        ),
        "BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_DESTINATION_PREFIX": (
            "s3://blueprint/task-evaluation/production-inputs/task-evaluation-activations"
        ),
    }
