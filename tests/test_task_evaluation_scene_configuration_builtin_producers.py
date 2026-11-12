from __future__ import annotations

import hashlib
from contextlib import contextmanager
import json
import os
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
# These run as root here and as an unprivileged user in CI. A read-only mode
# protects nothing from root, so every assertion is on what the function did,
# not on EACCES; and every setup that plants something in the sealed tree
# opens the parent for that moment only (``_unsealed``), so the tree the code
# sees is the same for any user.

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
    # A run unpacks a zip, so each file is its own inode; the builder here links
    # files it shares with its sources.
    _own_inodes(root)
    # In a run this package is imported from provider_runtime/blueprint_pipeline.
    monkeypatch.setattr(module, "_loaded_runtime_root", lambda: runtime)
    digest = json.loads((root / f"{module.TOOLCHAIN_SCHEMA_VERSION}.json").read_text())["toolchain_digest"]
    return module, root.resolve(), commit, digest


@contextmanager
def _unsealed(*paths: Path):
    """Give the owner write access to ``paths`` for a setup step, then put each mode back.

    What a process running as root does past read-only modes, done honestly by
    the owner: the modes the code later sees are exactly the sealed ones.
    """
    import os
    import stat as stat_module

    saved = []
    try:
        for path in paths:
            mode = stat_module.S_IMODE(os.lstat(path).st_mode)
            saved.append((path, mode))
            os.chmod(path, mode | (0o700 if stat_module.S_ISDIR(os.lstat(path).st_mode) else 0o600))
        yield
    finally:
        for path, mode in reversed(saved):
            if os.path.lexists(path) and not os.path.islink(path):
                os.chmod(path, mode)


def _own_inodes(root: Path) -> None:
    import os

    for directory, _names, files in os.walk(root):
        for name in files:
            path = Path(directory) / name
            info = path.lstat()
            if info.st_nlink > 1:
                data = path.read_bytes()
                with _unsealed(Path(directory)):
                    path.unlink()
                    path.write_bytes(data)
                    path.chmod(info.st_mode & 0o7777)


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
    with _unsealed(root / _PACKAGE):
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
    with _unsealed(root / "stages"):
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
    with _unsealed(root / "components", root / "stages"):
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
    with _unsealed(sealed):
        sealed.write_bytes(b"#!/bin/sh\nexit 0\n")  # root writes past 0555
    cache = root / _PACKAGE / "__pycache__"
    with _unsealed(root / _PACKAGE):
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
    with _unsealed(root / _SEALED_CACHE):
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
    with _unsealed(root / "stages", root / _PACKAGE):
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
    with _unsealed(root / _PACKAGE):
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
    with _unsealed(root / _PACKAGE):
        cache.mkdir()

    record = _restore(module, root, commit, "sha256:" + "0" * 64)

    assert record["status"] == "skipped_manifest_not_bound_to_stage_input"
    assert cache.is_dir()


def test_restore_reseals_a_parent_it_opened_to_remove_an_entry(tmp_path, monkeypatch):
    """Without the override of file modes (the control-plane prestage) a sealed parent must be opened."""
    import os

    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    cache = root / _PACKAGE / "__pycache__"
    with _unsealed(root / _PACKAGE):
        cache.mkdir()
    (cache / "nested").mkdir()
    (cache / "nested" / "x.pyc").write_bytes(b"x")
    (cache / "nested").chmod(0o555)
    package = (root / _PACKAGE).stat().st_ino
    modes = []
    real_fchmod = os.fchmod

    def fchmod(handle, mode):
        modes.append((os.fstat(handle).st_ino, mode))
        real_fchmod(handle, mode)

    # The parent is opened whatever the user id: as the owner, and as root
    # without the override of file modes, nothing else can remove the name.
    monkeypatch.setattr(os, "fchmod", fchmod)

    record = _restore(module, root, commit, digest)

    assert record["removed_entry_count"] == 3 and not cache.exists()
    assert (package, 0o755) in modes  # opened to remove the entry...
    assert (root / _PACKAGE).stat().st_mode & 0o777 == 0o555  # ...and sealed again
    assert record["modes_restored"] == []  # opening it was ours, not drift
    _validate(module, root, commit)


# Review of 5321b4372: races, mounts, hard links, deep trees and the guard.


def _writable_outside(tmp_path: Path) -> Path:
    outside = tmp_path / "outside"
    (outside / "nested").mkdir(parents=True)
    (outside / "nested" / "keep.txt").write_text("keep", encoding="utf-8")
    (outside / "file.txt").write_text("file", encoding="utf-8")
    (outside / "file.txt").chmod(0o666)
    (outside / "stray.txt").write_text("foreign only if followed", encoding="utf-8")
    outside.chmod(0o777)
    return outside


def _outside_state(outside: Path) -> tuple:
    return outside.lstat().st_mode, _snapshot(outside)


def test_guard_accepts_a_root_reached_through_a_linked_ancestor(tmp_path, monkeypatch):
    """The entrypoint passes ``$RUNTIME_ROOT/toolchain`` unresolved; ``/workspace`` may be a link."""
    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.symlink_to(tmp_path, target_is_directory=True)
    via_link = workspace / "provider_runtime" / "toolchain"
    cache = root / _PACKAGE / "__pycache__"
    with _unsealed(root / _PACKAGE):
        cache.mkdir()

    assert module.is_run_toolchain_copy(via_link) is True
    record = _restore(module, via_link, commit, digest)

    assert record["status"] == "restored" and not cache.exists()
    _validate(module, root, commit)


def test_guard_never_accepts_a_linked_root_a_missing_bundle_manifest_or_other_code(tmp_path, monkeypatch):
    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    runtime = root.parent
    assert module.is_run_toolchain_copy(root) is True
    # The root itself a link (to a real toolchain beside it).
    real = runtime / "toolchain-real"
    root.rename(real)
    root.symlink_to(real, target_is_directory=True)
    assert module.is_run_toolchain_copy(root) is False
    with pytest.raises(module.TaskEvaluationSceneConfigurationStageProducerError,
                       match="^scene_configuration_toolchain_restore_not_run_copy$"):
        _restore(module, root, commit, digest)
    root.unlink()
    real.rename(root)
    # No bundle manifest beside it, or one that is not a regular file.
    bundle = runtime / module._PROVIDER_BUNDLE_MANIFEST
    bundle.rename(runtime / "elsewhere.json")
    assert module.is_run_toolchain_copy(root) is False
    bundle.symlink_to(runtime / "elsewhere.json")
    assert module.is_run_toolchain_copy(root) is False
    bundle.unlink()
    (runtime / "elsewhere.json").rename(bundle)
    assert module.is_run_toolchain_copy(root) is True
    # This package imported from somewhere else (the release checkout).
    monkeypatch.setattr(module, "_loaded_runtime_root", lambda: tmp_path / "release" / "src")
    assert module.is_run_toolchain_copy(root) is False


def test_a_link_at_a_kept_file_name_changes_nothing_outside(tmp_path, monkeypatch):
    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    outside = _writable_outside(tmp_path)
    sealed = root / _PACKAGE / "run"
    with _unsealed(root / _PACKAGE):
        sealed.unlink()
        sealed.symlink_to(outside / "file.txt")
    before = _outside_state(outside)

    _restore(module, root, commit, digest)

    assert _outside_state(outside) == before and sealed.is_symlink()
    with pytest.raises(module.TaskEvaluationSceneConfigurationStageProducerError,
                       match=f"^scene_configuration_toolchain_symlink_forbidden:{_PACKAGE}/run$"):
        _validate(module, root, commit)


def test_a_link_at_a_kept_directory_name_changes_nothing_outside(tmp_path, monkeypatch):
    import shutil

    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    outside = _writable_outside(tmp_path)
    stages = root / "stages"
    with _unsealed(root):
        stages.chmod(0o755)
        shutil.rmtree(stages)
        stages.symlink_to(outside, target_is_directory=True)
    before = _outside_state(outside)

    _restore(module, root, commit, digest)

    assert _outside_state(outside) == before
    with pytest.raises(module.TaskEvaluationSceneConfigurationStageProducerError,
                       match="^scene_configuration_toolchain_symlink_forbidden:stages$"):
        _validate(module, root, commit)


def test_a_sealed_name_hard_linked_elsewhere_keeps_its_mode_and_is_refused(tmp_path, monkeypatch):
    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    sealed = root / _PACKAGE / "run"
    victim = outside / "victim"
    victim.write_bytes(sealed.read_bytes())
    victim.chmod(0o775)
    with _unsealed(root / _PACKAGE):
        sealed.unlink()
        os_link(victim, sealed)

    record = _restore(module, root, commit, digest)

    assert record["left"] == [{"relative_path": f"{_PACKAGE}/run", "reason": "hard_linked"}]
    assert record["modes_restored"] == []
    assert victim.stat().st_mode & 0o777 == 0o775
    with pytest.raises(module.TaskEvaluationSceneConfigurationStageProducerError,
                       match=f"^scene_configuration_toolchain_file_invalid:{_PACKAGE}/run$"):
        _validate(module, root, commit)


def test_sealed_files_linked_into_a_private_tree_still_pass(tmp_path, monkeypatch):
    """``private_blender`` hard-links sealed files after the stage-3 check; stage 5 must still run."""
    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    private = tmp_path / "packaged_blender"
    private.mkdir()
    sealed = root / _PACKAGE / "run"
    os_link(sealed, private / "run")
    cache = root / _PACKAGE / "__pycache__"
    with _unsealed(root / _PACKAGE):
        cache.mkdir()

    record = _restore(module, root, commit, digest)

    assert record["status"] == "restored" and record["left"] == []
    assert (private / "run").stat().st_ino == sealed.stat().st_ino
    _validate(module, root, commit)


def os_link(source: Path, target: Path) -> None:
    import os

    os.link(source, target)


def test_a_parent_swapped_for_a_link_mid_restore_never_reaches_outside(tmp_path, monkeypatch):
    """Review probe P10: every action is relative to a handle, so a swapped parent is not followed."""
    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    mirror = outside / Path(_PACKAGE).relative_to("components")
    mirror.mkdir(parents=True)
    (mirror / "planted.txt").write_text("OUTSIDE - must survive", encoding="utf-8")
    with _unsealed(root / _PACKAGE):
        (root / _PACKAGE / "planted.txt").write_text("foreign", encoding="utf-8")
    real_entries = module._entries
    swapped = []

    def entries(handle):
        rows = real_entries(handle)
        if not swapped and any(name == "planted.txt" for name, _info in rows):
            swapped.append(1)
            # Between listing and acting: put a link where `components` was.
            root.chmod(0o755)
            (root / "components").rename(root / "components-moved")
            (root / "components").symlink_to(outside, target_is_directory=True)
        return rows

    monkeypatch.setattr(module, "_entries", entries)

    _restore(module, root, commit, digest)

    assert swapped and (mirror / "planted.txt").read_text() == "OUTSIDE - must survive"
    # The removal went to the handle it held: the real, moved directory.
    assert not (root / "components-moved" / Path(_PACKAGE).relative_to("components") / "planted.txt").exists()


def test_an_entry_on_another_filesystem_is_left_alone(tmp_path, monkeypatch):
    """Review probe P12 without mount rights: an entry whose device differs is never touched."""
    import os

    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    mounted = root / "stages" / "mnt"
    with _unsealed(root / "stages"):
        mounted.mkdir()
    (mounted / "data.txt").write_text("other filesystem", encoding="utf-8")
    real_entries = module._entries

    def entries(handle):
        rows = []
        for name, info in real_entries(handle):
            if name == "mnt":
                values = list(info)
                values[2] = info.st_dev + 1  # st_dev
                info = os.stat_result(values)
            rows.append((name, info))
        return rows

    monkeypatch.setattr(module, "_entries", entries)

    record = _restore(module, root, commit, digest)

    assert record["left"] == [{"relative_path": "stages/mnt", "reason": "other_filesystem"}]
    assert (mounted / "data.txt").read_text() == "other filesystem"
    with pytest.raises(module.TaskEvaluationSceneConfigurationStageProducerError,
                       match="^scene_configuration_toolchain_not_read_only:stages/mnt$"):
        _validate(module, root, commit)


def test_a_real_mount_inside_the_tree_is_left_alone(tmp_path, monkeypatch):
    """Review probe P12, with a real bind mount.

    Only the mount and the unmount need root. On the disposable CI runner
    (``BLUEPRINT_DISPOSABLE_LINUX_TEST=1``, as in ``test_registered_feature_linux``)
    they run through ``sudo -n`` and a failure there fails the test; only a
    developer machine without mount rights and without that opt-in skips. The
    restore always runs as the test's own user.
    """
    import sys

    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    source = tmp_path / "mounted-source"
    source.mkdir()
    (source / "data.txt").write_text("other filesystem", encoding="utf-8")
    mounted = root / "stages" / "mnt"
    with _unsealed(root / "stages"):
        mounted.mkdir()
    prefix: list[str] = []
    bound = subprocess.run(["mount", "--bind", str(source), str(mounted)], capture_output=True, text=True)
    if bound.returncode != 0:
        if sys.platform != "linux" or os.environ.get("BLUEPRINT_DISPOSABLE_LINUX_TEST") != "1":
            pytest.skip("no mount rights here and no disposable Linux runner opt-in "
                        "(BLUEPRINT_DISPOSABLE_LINUX_TEST=1): " + bound.stderr.strip()[:200])
        prefix = ["sudo", "-n"]
        bound = subprocess.run([*prefix, "mount", "--bind", str(source), str(mounted)],
                               capture_output=True, text=True, timeout=60)
        assert bound.returncode == 0, "sudo -n mount --bind failed on the disposable runner: " + (
            bound.stdout + bound.stderr)
    try:
        record = _restore(module, root, commit, digest)
        assert (source / "data.txt").read_text() == "other filesystem"
        # A bind mount keeps the device number; the mount table still names it,
        # whoever made the mount.
        assert record["left"] == [{"relative_path": "stages/mnt", "reason": "other_filesystem"}]
        assert record["removed"] == []
    finally:
        unmounted = subprocess.run([*prefix, "umount", str(mounted)], capture_output=True, text=True, timeout=60)
        assert unmounted.returncode == 0, unmounted.stdout + unmounted.stderr


def test_a_very_deep_planted_tree_is_removed_without_recursion(tmp_path, monkeypatch):
    """Review probe P7: 3000 levels raised RecursionError out of rmtree."""
    import os

    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    base = root / _PACKAGE / "__pycache__"
    with _unsealed(root / _PACKAGE):
        base.mkdir()
    here = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for _ in range(3000):
            os.mkdir("d", dir_fd=here)
            child = os.open("d", os.O_RDONLY | os.O_DIRECTORY, dir_fd=here)
            os.close(here)
            here = child
    finally:
        os.close(here)

    record = _restore(module, root, commit, digest)

    assert record["removed"] == [{"relative_path": f"{_PACKAGE}/__pycache__", "kind": "directory",
                                  "contained_entries": 3000}]
    assert not base.exists()
    _validate(module, root, commit)


def _open_handles() -> int:
    import os

    return len(os.listdir("/proc/self/fd"))


def test_without_a_mount_table_no_directory_is_removed(tmp_path, monkeypatch):
    """A same-filesystem bind mount is visible only in the table; without it, remove no directory."""
    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    real = Path.read_bytes

    def read_bytes(self):
        if str(self) == "/proc/self/mountinfo":
            raise FileNotFoundError(self)
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    cache = root / _PACKAGE / "__pycache__"
    with _unsealed(root / _PACKAGE):
        cache.mkdir()
    (cache / "data.txt").write_text("could be a bind mount's data", encoding="utf-8")
    stray = root / "stages" / "stray.txt"
    link = root / "stages" / "link"
    with _unsealed(root / "stages"):
        stray.write_text("x", encoding="utf-8")
        link.symlink_to(tmp_path)
    (root / "components").chmod(0o755)

    record = _restore(module, root, commit, digest)

    assert record["left"] == [{"relative_path": f"{_PACKAGE}/__pycache__", "reason": "mount_table_unavailable"}]
    assert (cache / "data.txt").read_text() == "could be a bind mount's data"
    assert sorted(row["relative_path"] for row in record["removed"]) == ["stages/link", "stages/stray.txt"]
    assert not stray.exists() and not link.is_symlink()
    assert [row["relative_path"] for row in record["modes_restored"]] == ["components"]
    with pytest.raises(module.TaskEvaluationSceneConfigurationStageProducerError,
                       match=f"^scene_configuration_toolchain_not_read_only:{_PACKAGE}/__pycache__$"):
        _validate(module, root, commit)


def test_a_mount_table_with_non_utf8_paths_is_still_read(tmp_path, monkeypatch):
    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    base = os.fsencode(os.path.realpath(root))
    table = (b"36 35 98:0 / /mnt/caf\xe9\\040x rw - ext4 /dev/x rw\n"
             b"37 35 98:0 / " + base + b"/stages/m\\040nt rw - ext4 /dev/y rw\n"
             b"38 35 98:0 / " + base + b"/stages/\xff rw - ext4 /dev/z rw\n")
    real = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes",
                        lambda self: table if str(self) == "/proc/self/mountinfo" else real(self))

    assert module._mount_points_under(root) == {"stages/m nt", os.fsdecode(b"stages/\xff")}
    mounted = root / "stages" / "m nt"
    with _unsealed(root / "stages"):
        mounted.mkdir()
    (mounted / "data.txt").write_text("other filesystem", encoding="utf-8")
    record = _restore(module, root, commit, digest)
    assert record["left"] == [{"relative_path": "stages/m nt", "reason": "other_filesystem"}]
    assert (mounted / "data.txt").exists()


def test_a_failed_listing_leaks_no_handle(tmp_path, monkeypatch):
    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    with _unsealed(root / "stages"):
        (root / "stages" / "stray").write_text("x", encoding="utf-8")
    real = module._entries
    calls = []

    def entries(handle):
        calls.append(1)
        if len(calls) == 2:
            raise OSError(24, "listing failed")
        return real(handle)

    monkeypatch.setattr(module, "_entries", entries)
    before = _open_handles()
    with pytest.raises(OSError, match="listing failed"):
        _restore(module, root, commit, digest)
    assert _open_handles() == before


def test_a_kept_folder_swapped_for_an_outside_link_before_it_is_opened(tmp_path, monkeypatch):
    """Review probe P10a: refused by ``O_NOFOLLOW`` and by the device and inode comparison."""
    module, root, commit, digest = _run_copy(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    mirror = outside / Path(_PACKAGE).relative_to("components")
    mirror.mkdir(parents=True)
    (mirror / "planted.txt").write_text("OUTSIDE", encoding="utf-8")
    with _unsealed(root / _PACKAGE):
        (root / _PACKAGE / "planted.txt").write_text("foreign", encoding="utf-8")
    real = module._open_at
    swapped = []

    def open_at(name, directory, info, *, flags):
        if name == "components" and not swapped:
            swapped.append(1)
            root.chmod(0o755)
            (root / "components").rename(root / "components-moved")
            (root / "components").symlink_to(outside, target_is_directory=True)
        return real(name, directory, info, flags=flags)

    monkeypatch.setattr(module, "_open_at", open_at)

    record = _restore(module, root, commit, digest)

    assert swapped and (mirror / "planted.txt").read_text() == "OUTSIDE"
    assert {"relative_path": "components", "reason": "entry_changed"} in record["left"]
