from __future__ import annotations

from pathlib import Path

from scripts import provision_task_evaluation_scene_configuration_release as subject


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
            "s3://blueprint-production-inputs/coordinator-release-windows/"
        ),
        "BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_DESTINATION_PREFIX": (
            "s3://blueprint-production-inputs/task-evaluation-activations"
        ),
    }
