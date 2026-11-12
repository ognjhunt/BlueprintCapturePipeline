from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import (
    GPU_STAGE_TIMEOUT_SECONDS,
    STAGE_DEADLINE_EPOCH_ENV,
    artifixer_training_timeout_seconds,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_builtin_producers import (
    builtin_scene_configuration_stage_producer_registry,
)
from blueprint_pipeline.task_evaluation_scene_configuration_stage_producers import (
    ADMITTED_PRODUCER_IDENTITIES,
    PRODUCTION_RESULT_SCHEMA_VERSION,
)
from scripts.build_task_evaluation_scene_configuration_toolchain import (
    build_published_scene_configuration_toolchain,
)
from tests.test_build_task_evaluation_scene_configuration_toolchain import (
    _component_packages,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _toolchain(tmp_path: Path, commit: str) -> Path:
    root = tmp_path / "toolchain"
    build_published_scene_configuration_toolchain(
        source_commit=commit,
        output_root=root,
        readback=lambda path: path.read_bytes(),
        readback_actor="service-account:test",
        component_packages=_component_packages(tmp_path),
    )
    return root


def test_registry_checks_its_shared_runtime_once_and_rechecks_the_next_operation(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_configuration_builtin_producers as module
    commit = "a" * 40
    root = _toolchain(tmp_path, commit)
    original = module._validate_toolchain
    calls = []
    def validate(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(module, "_validate_toolchain", validate)
    registry = module.builtin_scene_configuration_stage_producer_registry(
        toolchain_root=root, expected_source_commit=commit, environment={})
    assert registry is not None and len(calls) == 1
    # No verdict survives the operation. A changed installed entrypoint must
    # refuse even though this exact root was valid in the preceding call.
    entrypoint = next(path for path in (root / "stages").iterdir() if path.is_file())
    entrypoint.chmod(0o755)
    entrypoint.write_bytes(entrypoint.read_bytes() + b"\n# changed\n")
    entrypoint.chmod(0o555)
    with pytest.raises(module.TaskEvaluationSceneConfigurationStageProducerError):
        module.builtin_scene_configuration_stage_producer_registry(
            toolchain_root=root, expected_source_commit=commit, environment={})
    assert len(calls) == 2


def test_builtin_producer_executes_only_sealed_entrypoint_and_redacts_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("blueprint_pipeline.task_evaluation_scene_configuration_builtin_producers.time.time", lambda: 1000.0)
    commit = "a" * 40
    toolchain = _toolchain(tmp_path, commit)
    secret = tmp_path / "openai-key"
    secret.write_text("super-secret-value\n", encoding="utf-8")
    secret.chmod(0o400)
    calls: list[list[str]] = []

    def run(command, *, env, **_kwargs):
        calls.append(command)
        assert float(env[STAGE_DEADLINE_EPOCH_ENV]) == 13_000
        # A slower 9,000s first training pass fits the existing 12,000s stage.
        assert artifixer_training_timeout_seconds(env, now_epoch=1_500) >= 9_000
        # A repair shares that same deadline instead of getting a fresh window.
        assert artifixer_training_timeout_seconds(env, now_epoch=11_000) == 1_400
        output = Path(env["BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT"])
        stage_input = json.loads(
            Path(env["BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT"]).read_text(
                encoding="utf-8"
            )
        )
        assert stage_input["construction_envelope"]["run_id"] == "configure-scene"
        result_path = Path(env["BLUEPRINT_SCENE_CONFIGURATION_STAGE_RESULT"])
        roles = (
            "configured_appearance_without_source_object",
            "appearance_removal_receipt",
            "appearance_visual_review_receipt",
            "configured_task_thumbnail",
            "provider_render_reference_manifest",
        )
        artifacts = []
        for role in roles:
            path = output / f"{role}.bin"
            path.write_bytes(role.encode())
            artifacts.append(
                {
                    "role": role,
                    "path": str(path),
                    "digest": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        result = {
            "schema_version": PRODUCTION_RESULT_SCHEMA_VERSION,
            "status": "completed",
            "stage_id": "stage-1",
            "capability": "observed_appearance_object_removal",
            "provider_mutations_performed": 0,
            "paid_execution_requested": False,
            "executed_inside_parent_configuration_run": True,
            "artifacts": artifacts,
            "production_result_digest": "",
        }
        result["production_result_digest"] = canonical_digest(
            result, digest_field="production_result_digest"
        )
        result_path.write_text(json.dumps(result), encoding="utf-8")
        return subprocess.CompletedProcess(
            command, 0, stdout="key=super-secret-value", stderr=""
        )

    registry = builtin_scene_configuration_stage_producer_registry(
        expected_source_commit=commit,
        toolchain_root=toolchain,
        runner=run,
        environment={"OPENAI_API_KEY_FILE": str(secret), STAGE_DEADLINE_EPOCH_ENV: "999999999999"},
    )
    output = tmp_path / "output"
    output.mkdir()
    identity = ADMITTED_PRODUCER_IDENTITIES[0]
    configuration = tmp_path / "configuration.json"
    configuration.write_text("{}\n", encoding="utf-8")
    artifacts = registry.execute(
        stage={
            "stage_id": "stage-1",
            "capability": identity.capability,
            "adapter": {"id": identity.adapter_id, "version": identity.version},
            "execution_class": "gpu_canary",
        },
        envelope={"run_id": "configure-scene"},
        configuration={},
        configuration_path=configuration,
        dependency_results=(),
        output_root=output,
    )

    assert len(artifacts) == 5
    assert calls == [[str(toolchain / "stages" / identity.adapter_id)]]
    log = (output / "stage_producer.log").read_text(encoding="utf-8")
    assert "super-secret-value" not in log
    assert "REDACTED_SECRET" in log


@pytest.mark.parametrize("environment,refusal", [
    ({"OPENAI_API_KEY": "must-not-cross-runtime-boundary"},
     "scene_configuration_raw_secret_environment_forbidden"),
    ({"OPENAI_API_KEY_FILE": "/missing-blueprint-test-secret"},
     "scene_configuration_secret_file_invalid:OPENAI_API_KEY_FILE"),
])
def test_builtin_producer_rejects_credentials_before_entry(tmp_path: Path, environment, refusal) -> None:
    commit = "b" * 40
    registry = builtin_scene_configuration_stage_producer_registry(
        expected_source_commit=commit,
        toolchain_root=_toolchain(tmp_path, commit),
        environment=environment,
        runner=lambda *args, **kwargs: pytest.fail("credential refusal must precede producer entry"),
    )
    output = tmp_path / "output"
    output.mkdir()
    configuration = tmp_path / "configuration.json"
    configuration.write_text("{}\n", encoding="utf-8")
    identity = ADMITTED_PRODUCER_IDENTITIES[0]

    with pytest.raises(
        RuntimeError, match=refusal
    ):
        registry.execute(
            stage={
                "stage_id": "stage-1",
                "capability": identity.capability,
                "adapter": {"id": identity.adapter_id, "version": identity.version},
                "execution_class": "gpu_canary",
            },
            envelope={"run_id": "configure-scene"},
            configuration={},
            configuration_path=configuration,
            dependency_results=(),
            output_root=output,
        )
    assert {path.name for path in output.iterdir()} == {
        "stage_production_input.v1.json", "dependency_results.v1.json",
    }


def test_builtin_producer_retains_redacted_partial_output_on_timeout(
    tmp_path: Path,
) -> None:
    commit = "c" * 40
    secret = tmp_path / "openai-key"
    secret.write_text("timeout-secret-value\n", encoding="utf-8")
    secret.chmod(0o400)

    def time_out(command, **kwargs):
        raise subprocess.TimeoutExpired(
            command,
            kwargs["timeout"],
            output=b"partial stdout timeout-secret-value",
            stderr=(
                b"partial stderr timeout-secret-value "
                b"https://object.invalid/out?X-Amz-Signature=signed-timeout-value"
            ),
        )

    registry = builtin_scene_configuration_stage_producer_registry(
        expected_source_commit=commit,
        toolchain_root=_toolchain(tmp_path, commit),
        runner=time_out,
        environment={"OPENAI_API_KEY_FILE": str(secret)},
    )
    output = tmp_path / "output"
    output.mkdir()
    configuration = tmp_path / "configuration.json"
    configuration.write_text("{}\n", encoding="utf-8")
    identity = ADMITTED_PRODUCER_IDENTITIES[0]

    with pytest.raises(
        RuntimeError,
        match=(
            "scene_configuration_stage_producer_timeout:"
            "artifixer3d_observed_object_removal:"
            # Derived, so a stage budget change cannot leave this behind.
            f"{GPU_STAGE_TIMEOUT_SECONDS['artifixer3d_observed_object_removal']}"
        ),
    ):
        registry.execute(
            stage={
                "stage_id": "stage-1",
                "capability": identity.capability,
                "adapter": {"id": identity.adapter_id, "version": identity.version},
                "execution_class": "gpu_canary",
            },
            envelope={"run_id": "configure-scene"},
            configuration={},
            configuration_path=configuration,
            dependency_results=(),
            output_root=output,
        )

    log = (output / "stage_producer.log").read_text(encoding="utf-8")
    assert "partial stdout" in log
    assert "partial stderr" in log
    assert "timeout-secret-value" not in log
    assert "signed-timeout-value" not in log
    assert "<redacted>" in log
    assert log.count("REDACTED_SECRET") == 2


def test_expected_stage_roles_admit_the_destination_result_only_when_declared() -> None:
    from blueprint_pipeline.task_evaluation_scene_configuration_builtin_producers import (
        expected_stage_artifact_roles,
    )

    base = expected_stage_artifact_roles(
        "simready_native_import_qualification", supplemental_destination=False
    )
    assert base == {"native_import_runtime_result"}
    assert expected_stage_artifact_roles(
        "simready_native_import_qualification", supplemental_destination=True
    ) == {"native_import_runtime_result", "destination_native_import_runtime_result"}
    # Other adapters never grow a destination role.
    assert expected_stage_artifact_roles(
        "content_agents_rigid_replacement", supplemental_destination=True
    ) == expected_stage_artifact_roles(
        "content_agents_rigid_replacement", supplemental_destination=False
    )


# The run's own toolchain copy: restoration at the start of a stage.
#
# These run as root in CI images and here, so a read-only mode protects
# nothing: every assertion is on what the function did, not on EACCES.

_PACKAGE = "components/content_agents_rigid_replacement/package"
_SEALED_CACHE = f"{_PACKAGE}/runtime/python/lib/__pycache__"


def _run_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, commit: str = "a" * 40):
    """A toolchain laid out as a run unpacks it: ``provider_runtime/toolchain``."""
    from blueprint_pipeline import task_evaluation_scene_configuration_builtin_producers as module

    runtime = tmp_path / "provider_runtime"
    runtime.mkdir()
    root = runtime / "toolchain"
    build_published_scene_configuration_toolchain(
        source_commit=commit, output_root=root, readback=lambda path: path.read_bytes(),
        readback_actor="service-account:test",
        component_packages=_component_packages(tmp_path, bundled_bytecode=True),
    )
    (runtime / module._PROVIDER_BUNDLE_MANIFEST).write_text("{}\n", encoding="utf-8")
    # In a run this package is imported from provider_runtime/blueprint_pipeline.
    monkeypatch.setattr(module, "_loaded_runtime_root", lambda: runtime)
    digest = json.loads((root / f"{module.TOOLCHAIN_SCHEMA_VERSION}.json").read_text())["toolchain_digest"]
    return module, root.resolve(), commit, digest


def _restore(module, root, commit, digest):
    return module.restore_run_toolchain_copy(
        root=root, expected_source_commit=commit, expected_toolchain_digest=digest)


def _validate(module, root, commit):
    return module._validate_toolchain(root=root, expected_source_commit=commit, verify_bytes=True)


def _snapshot(base: Path, *, skip: Path | None = None) -> dict[str, tuple]:
    """Every entry under ``base`` (links not followed) with its kind, mode and bytes."""
    import os
    import stat as stat_module

    rows = {}
    for directory, names, files in os.walk(base, followlinks=False):
        for name in [*names, *files]:
            path = Path(directory) / name
            if skip is not None and (path == skip or path.is_relative_to(skip)):
                continue
            info = path.lstat()
            if stat_module.S_ISLNK(info.st_mode):
                rows[str(path)] = ("link", os.readlink(path))
            elif stat_module.S_ISREG(info.st_mode):
                rows[str(path)] = ("file", info.st_mode, path.read_bytes())
            else:
                rows[str(path)] = ("dir", info.st_mode)
    return rows


def test_bundle_manifest_name_matches_what_the_bundle_builder_writes() -> None:
    from blueprint_pipeline import task_evaluation_scene_configuration_builtin_producers as module
    from blueprint_pipeline.task_evaluation_scene_configuration_bundle_evidence import (
        BUNDLE_SCHEMA_VERSION,
    )

    assert module._PROVIDER_BUNDLE_MANIFEST == f"{BUNDLE_SCHEMA_VERSION}.json"


def test_restore_removes_a_foreign_pycache_and_the_full_check_then_passes(tmp_path, monkeypatch):
    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    cache = root / _PACKAGE / "__pycache__"
    cache.mkdir()  # 0755 under the default umask, as container root leaves it
    (cache / "_render_driver.cpython-311.pyc").write_bytes(b"bytecode")
    with pytest.raises(module.TaskEvaluationSceneConfigurationStageProducerError,
                       match=f"^scene_configuration_toolchain_not_read_only:{_PACKAGE}/__pycache__$"):
        _validate(module, root, commit)

    record = _restore(module, root, commit, digest)

    assert record["status"] == "restored"
    assert record["removed"] == [{"relative_path": f"{_PACKAGE}/__pycache__", "kind": "directory",
                                  "contained_entries": 1}]
    assert record["removed_count"] == 1 and record["removed_entry_count"] == 2
    assert not cache.exists()
    # Bytecode the sealed inventory lists is kept.
    assert (root / _SEALED_CACHE / "copy.cpython-311.pyc").is_file()
    manifest, _entrypoints = _validate(module, root, commit)
    assert manifest["toolchain_digest"] == digest


def test_restore_removes_a_foreign_file_in_a_sealed_directory(tmp_path, monkeypatch):
    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    stray = root / "stages" / "stray.txt"
    stray.write_text("left by a process", encoding="utf-8")
    with pytest.raises(module.TaskEvaluationSceneConfigurationStageProducerError,
                       match="^scene_configuration_toolchain_file_invalid:stages/stray.txt$"):
        _validate(module, root, commit)

    record = _restore(module, root, commit, digest)

    assert record["removed"] == [{"relative_path": "stages/stray.txt", "kind": "file", "contained_entries": 0}]
    assert not stray.exists()
    _validate(module, root, commit)


def test_restore_clears_a_write_bit_a_sealed_directory_and_the_root_gained(tmp_path, monkeypatch):
    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    (root / "components").chmod(0o755)
    root.chmod(0o755)
    with pytest.raises(module.TaskEvaluationSceneConfigurationStageProducerError,
                       match="^scene_configuration_toolchain_root_invalid:not_read_only$"):
        _validate(module, root, commit)

    record = _restore(module, root, commit, digest)

    assert record["removed_count"] == 0
    assert {row["relative_path"]: (row["before_mode"], row["after_mode"]) for row in record["modes_restored"]} == {
        ".": ("0o755", "0o555"), "components": ("0o755", "0o555")}
    assert (root / "components").stat().st_mode & 0o777 == 0o555
    assert root.stat().st_mode & 0o777 == 0o555
    _validate(module, root, commit)


def test_restore_clears_a_write_bit_on_a_sealed_file_without_touching_its_bytes(tmp_path, monkeypatch):
    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    sealed = root / _PACKAGE / "run"
    before = (sealed.read_bytes(), sealed.stat().st_ino)
    sealed.chmod(0o755)
    with pytest.raises(module.TaskEvaluationSceneConfigurationStageProducerError,
                       match=f"^scene_configuration_toolchain_file_invalid:{_PACKAGE}/run$"):
        _validate(module, root, commit)

    record = _restore(module, root, commit, digest)

    assert record["modes_restored"] == [{"relative_path": f"{_PACKAGE}/run", "kind": "file",
                                         "before_mode": "0o755", "after_mode": "0o555"}]
    assert (sealed.read_bytes(), sealed.stat().st_ino) == before
    _validate(module, root, commit)


def test_restore_unlinks_a_planted_link_and_leaves_its_target_alone(tmp_path, monkeypatch):
    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    (outside / "nested").mkdir(parents=True)
    (outside / "nested" / "keep.txt").write_text("not ours", encoding="utf-8")
    (outside / "file.txt").write_text("not ours either", encoding="utf-8")
    (root / "components" / "escape").symlink_to(outside, target_is_directory=True)
    (root / "stages" / "escape.txt").symlink_to(outside / "file.txt")
    before = _snapshot(outside)
    with pytest.raises(module.TaskEvaluationSceneConfigurationStageProducerError,
                       match="^scene_configuration_toolchain_symlink_forbidden:"):
        _validate(module, root, commit)

    record = _restore(module, root, commit, digest)

    assert sorted((row["relative_path"], row["kind"]) for row in record["removed"]) == [
        ("components/escape", "symlink"), ("stages/escape.txt", "symlink")]
    assert not (root / "components" / "escape").is_symlink()
    assert not (root / "stages" / "escape.txt").is_symlink()
    assert _snapshot(outside) == before
    _validate(module, root, commit)


def test_restore_never_repairs_a_changed_sealed_file(tmp_path, monkeypatch):
    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    sealed = root / _PACKAGE / "run"
    sealed.write_bytes(b"#!/bin/sh\nexit 0\n")  # root writes past 0555
    cache = root / _PACKAGE / "__pycache__"
    cache.mkdir()

    record = _restore(module, root, commit, digest)

    assert record["removed"][0]["relative_path"] == f"{_PACKAGE}/__pycache__"
    assert sealed.read_bytes() == b"#!/bin/sh\nexit 0\n"
    with pytest.raises(module.TaskEvaluationSceneConfigurationStageProducerError,
                       match=f"^scene_configuration_toolchain_file_invalid:{_PACKAGE}/run$"):
        _validate(module, root, commit)


def test_restore_never_supplies_a_missing_sealed_file(tmp_path, monkeypatch):
    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    missing = root / _SEALED_CACHE / "copy.cpython-311.pyc"
    missing.unlink()

    record = _restore(module, root, commit, digest)

    assert record["status"] == "unchanged" and not missing.exists()
    with pytest.raises(module.TaskEvaluationSceneConfigurationStageProducerError,
                       match=f"^scene_configuration_toolchain_inventory_incomplete:{_SEALED_CACHE}/"
                             r"copy.cpython-311.pyc:missing_count=1$"):
        _validate(module, root, commit)


def test_restore_touches_nothing_outside_the_toolchain(tmp_path, monkeypatch):
    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "target.txt").write_text("x", encoding="utf-8")
    (root / "stages" / "link").symlink_to(outside / "target.txt")
    cache = root / _PACKAGE / "__pycache__"
    cache.mkdir()
    (cache / "inner-link").symlink_to(outside, target_is_directory=True)
    (root / "components").chmod(0o755)
    before = _snapshot(tmp_path, skip=root)

    record = _restore(module, root, commit, digest)

    assert record["status"] == "restored"
    assert _snapshot(tmp_path, skip=root) == before
    _validate(module, root, commit)


def test_restore_refuses_a_toolchain_that_is_not_the_runs_own_copy(tmp_path, monkeypatch):
    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    cache = root / _PACKAGE / "__pycache__"
    cache.mkdir()
    # The deployed code is imported from the release checkout, not beside the tree.
    monkeypatch.setattr(module, "_loaded_runtime_root", lambda: tmp_path / "release" / "src")
    assert module.is_run_toolchain_copy(root) is False
    with pytest.raises(module.TaskEvaluationSceneConfigurationStageProducerError,
                       match="^scene_configuration_toolchain_restore_not_run_copy$"):
        _restore(module, root, commit, digest)
    assert cache.is_dir()
    # A published toolchain (no bundle manifest beside it) is never a run copy.
    published = _toolchain(tmp_path / "published", commit)
    assert module.is_run_toolchain_copy(published) is False
    # Every other caller still refuses, now naming the entry, and deletes nothing.
    with pytest.raises(module.TaskEvaluationSceneConfigurationStageProducerError,
                       match=f"^scene_configuration_toolchain_not_read_only:{_PACKAGE}/__pycache__$"):
        module.validate_scene_configuration_toolchain(root=root, expected_source_commit=commit)
    assert cache.is_dir()


def test_restore_changes_nothing_when_the_manifest_is_not_the_stage_inputs(tmp_path, monkeypatch):
    module, root, commit, _digest = _run_copy(tmp_path, monkeypatch)
    cache = root / _PACKAGE / "__pycache__"
    cache.mkdir()

    record = _restore(module, root, commit, "sha256:" + "0" * 64)

    assert record["status"] == "skipped_manifest_not_bound_to_stage_input"
    assert cache.is_dir()


def test_restore_reseals_a_parent_it_opened_to_remove_an_entry(tmp_path, monkeypatch):
    """Without root (the control-plane prestage) a sealed parent must be opened to unlink."""
    import os

    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    cache = root / _PACKAGE / "__pycache__"
    cache.mkdir()
    (cache / "nested").mkdir()
    (cache / "nested" / "x.pyc").write_bytes(b"x")
    (cache / "nested").chmod(0o555)
    opened = []
    real_chmod = module._chmod_nofollow

    def chmod(path, mode, *, directory):
        opened.append((Path(path), mode))
        real_chmod(path, mode, directory=directory)

    # Behave as an unprivileged owner would see the sealed modes.
    monkeypatch.setattr(os, "access", lambda path, mode: not (Path(path).stat().st_mode & 0o200 == 0))
    monkeypatch.setattr(module, "_chmod_nofollow", chmod)

    record = _restore(module, root, commit, digest)

    assert record["removed_entry_count"] == 3 and not cache.exists()
    assert (root / _PACKAGE, 0o755) in opened  # opened to remove the entry...
    assert (root / _PACKAGE).stat().st_mode & 0o777 == 0o555  # ...and sealed again
    assert record["modes_restored"] == []  # opening it was ours, not drift
    _validate(module, root, commit)
