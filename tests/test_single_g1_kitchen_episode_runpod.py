from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import importlib.metadata
import inspect
import json
import lzma
import os
import re
import shlex
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import single_g1_kitchen_episode_runpod as single_episode
from blueprint_pipeline.gpu_render_providers import RenderLaunchSpec, VastRenderProvider


def test_manipulation_policy_task_compatibility_fails_closed_without_declaration() -> None:
    report = single_episode._manipulation_policy_task_compatibility(
        plan={},
        task_id="microwave_door",
    )

    assert report["status"] == "blocked"
    assert report["blockers"] == [
        single_episode.MANIPULATION_POLICY_TASK_COMPATIBILITY_BLOCKER
    ]
    assert report["checks"]["declaration_present"] is False
    assert report["checks"]["requested_task_in_training_tasks"] is False
    assert report["claim_boundary"] == {
        "checkpoint_loadability_is_not_task_qualification": True,
        "sonic_action_shape_is_not_task_qualification": True,
        "wam_visual_prediction_is_not_registered_transition_success": True,
        "qualification_is_not_episode_success": True,
    }


def test_known_bag_checkpoint_cannot_be_redeclared_as_microwave_qualified() -> None:
    declaration = {
        "schema_version": (
            single_episode.MANIPULATION_POLICY_TASK_COMPATIBILITY_SCHEMA_VERSION
        ),
        "status": "qualified",
        "task_id": "microwave_door",
        "embodiment": single_episode.REQUIRED_SONIC_EMBODIMENT,
        "checkpoint_repo": single_episode.SEALED_SONIC_CHECKPOINT_REPO,
        "checkpoint_revision": single_episode.SEALED_SONIC_CHECKPOINT_REVISION,
        "training_dataset_repo": "blueprint/microwave-door-g1-v1",
        "training_dataset_revision": "c" * 40,
        "training_task_ids": ["microwave_door"],
        "training_task_descriptions": ["Open the microwave door."],
        "qualification_evidence": {
            "status": "passed",
            "task_id": "microwave_door",
            "environment": "isaac_sim",
            "registered_transition_id": (
                single_episode.REQUIRED_MANIPULATION_TRANSITION_ID
            ),
            "artifact_sha256": "a" * 64,
        },
    }
    report = single_episode._manipulation_policy_task_compatibility(
        plan={"manipulation_policy_task_compatibility": declaration},
        task_id="microwave_door",
    )

    assert report["status"] == "blocked"
    assert report["checks"]["checkpoint_repo_exact"] is True
    assert report["checks"]["checkpoint_revision_exact"] is True
    assert report["checks"]["training_dataset_repo_exact"] is False
    assert report["checks"]["requested_task_in_reviewed_checkpoint_tasks"] is False
    assert report["reviewed_checkpoint_training_provenance"] == {
        "dataset_repo": single_episode.SEALED_SONIC_TRAINING_DATASET_REPO,
        "dataset_revision": single_episode.SEALED_SONIC_TRAINING_DATASET_REVISION,
        "canonical_task_ids": [],
        "task_descriptions": [
            "grab the bag, turn 180 degrees and drop the bag"
        ],
        "requested_task_covered": False,
    }


def test_reviewed_microwave_checkpoint_requires_exact_sealed_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(single_episode, "SEALED_SONIC_CHECKPOINT_REPO", "blueprint/g1-microwave")
    monkeypatch.setattr(single_episode, "SEALED_SONIC_CHECKPOINT_REVISION", "b" * 40)
    monkeypatch.setattr(
        single_episode,
        "SEALED_SONIC_TRAINING_DATASET_REPO",
        "blueprint/g1-microwave-demos",
    )
    monkeypatch.setattr(
        single_episode,
        "SEALED_SONIC_TRAINING_DATASET_REVISION",
        "c" * 40,
    )
    monkeypatch.setattr(
        single_episode,
        "SEALED_SONIC_REVIEWED_TRAINING_TASK_IDS",
        ("microwave_door",),
    )
    monkeypatch.setattr(
        single_episode,
        "SEALED_SONIC_REVIEWED_TRAINING_TASK_DESCRIPTIONS",
        ("Open the microwave door.",),
    )
    declaration = {
        "schema_version": (
            single_episode.MANIPULATION_POLICY_TASK_COMPATIBILITY_SCHEMA_VERSION
        ),
        "status": "qualified",
        "task_id": "microwave_door",
        "embodiment": single_episode.REQUIRED_SONIC_EMBODIMENT,
        "checkpoint_repo": "blueprint/g1-microwave",
        "checkpoint_revision": "b" * 40,
        "training_dataset_repo": "blueprint/g1-microwave-demos",
        "training_dataset_revision": "c" * 40,
        "training_task_ids": ["microwave_door"],
        "training_task_descriptions": ["Open the microwave door."],
        "qualification_evidence": {
            "status": "passed",
            "task_id": "microwave_door",
            "environment": "isaac_sim",
            "registered_transition_id": (
                single_episode.REQUIRED_MANIPULATION_TRANSITION_ID
            ),
            "artifact_sha256": "a" * 64,
        },
    }
    report = single_episode._manipulation_policy_task_compatibility(
        plan={"manipulation_policy_task_compatibility": declaration},
        task_id="microwave_door",
    )
    assert report["status"] == "qualified"
    assert report["failed_checks"] == []
    assert report["blockers"] == []

    declaration["training_dataset_revision"] = "d" * 40
    mismatch = single_episode._manipulation_policy_task_compatibility(
        plan={"manipulation_policy_task_compatibility": declaration},
        task_id="microwave_door",
    )
    assert mismatch["status"] == "blocked"
    assert mismatch["checks"]["training_dataset_revision_exact"] is False


def test_unqualified_manipulation_policy_blocks_before_inventory_capacity_or_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        single_episode,
        "OSCAR_RUNTIME_SOURCE_SEAL_CAPABLE_IMAGE_DIGESTS",
        frozenset({single_episode.IMAGE_DIGEST}),
    )
    task_compatibility = single_episode._manipulation_policy_task_compatibility(
        plan={},
        task_id="microwave_door",
    )
    monkeypatch.setattr(
        single_episode,
        "_load_single_episode_inputs",
        lambda _path: {
            "bundle_sha256": "b" * 64,
            "manipulation_policy_task_compatibility": task_compatibility,
        },
    )
    monkeypatch.setattr(
        single_episode,
        "_require_signed_output_staging_proof",
        lambda **_kwargs: {"status": "passed", "blockers": []},
    )

    class NoProviderCalls:
        def __getattr__(self, name: str):
            raise AssertionError(f"provider method must not be called: {name}")

    monkeypatch.setattr(
        single_episode,
        "get_render_provider",
        lambda _provider_name: NoProviderCalls(),
    )
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"unused-by-mocked-loader")
    secret_files = []
    for name in ("bundle-url", "put-url", "get-url"):
        path = tmp_path / f"{name}.txt"
        path.write_text(f"https://object.invalid/{name}?signature=test", encoding="utf-8")
        path.chmod(0o600)
        secret_files.append(path)
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps({"resolved_digest_ref": single_episode.IMAGE_REF}),
        encoding="utf-8",
    )

    result = single_episode.run_single_episode(
        provider_name="runpod",
        episode_bundle=bundle,
        provider_bundle_url_file=secret_files[0],
        provider_output_put_url_file=secret_files[1],
        provider_output_get_url_file=secret_files[2],
        release_evidence=release,
        provider_launch_request=tmp_path / "provider-launch-request.json",
        preflight_bundle=tmp_path / "preflight.json",
        admission_out=tmp_path / "admission.json",
        bound_request_out=tmp_path / "bound.json",
        adapter_output=tmp_path / "result.json",
        pod_name="",
        execute=True,
    )

    assert result["status"] == "blocked"
    assert result["provider_mutations_performed"] == 0
    assert result["blockers"] == [
        single_episode.MANIPULATION_POLICY_TASK_COMPATIBILITY_BLOCKER
    ]
    assert result["admission"]["status"] == "blocked"
    assert result["preflight"]["capacity"] == {}
    assert result["preflight"]["pre_inventory"] == {}


def test_execute_without_current_spend_lock_blocks_before_episode_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        single_episode,
        "OSCAR_RUNTIME_SOURCE_SEAL_CAPABLE_IMAGE_DIGESTS",
        frozenset({single_episode.IMAGE_DIGEST}),
    )
    inputs = {
        "plan": {
            "env": {},
            "groot_server_command": ["/opt/gr00t/server"],
            "gear_sonic_controller_command": ["/opt/wbc/deploy.sh", "sim"],
            "isaac_task_executor_command": ["/isaac-sim/python.sh", "executor.py"],
        },
        "route": {"route": []},
        "seed": {"seed": 1001},
        "start_frame": b"exact-frame",
        "bootstrap_script": "upload_phase inputs_ready\n",
        "runtime_package_overlay_xz_base64": "runtime-overlay",
        "isaac_runtime_backend_overlay_gzip_base64": "backend-overlay",
        "runtime_package_overlay_sha256": "1" * 64,
        "runtime_package_overlay_source_sha256s": {},
        "bundle_sha256": single_episode.BUNDLE_SHA256,
        "manipulation_policy_task_compatibility": {
            "status": "qualified",
            "blockers": [],
        },
    }
    launch_called = False

    class Provider:
        def build_request(self, _spec, _root):
            return {}

        def billable_inventory(self, *, name_prefix: str):
            assert name_prefix == ""
            return {
                "status": "observed",
                "api_confirmed": True,
                "live_resource_count": 0,
                "resources": [],
            }

        def capacity_preflight(self, _request):
            selected = {
                "display_name": "RTX A6000",
                "gpu_ram_mb": 48_000,
                "on_demand_price_usd_per_hour": 0.5,
            }
            return {"status": "available", "viable_gpu_types": [selected]}

        def launch(self, *_args, **_kwargs):
            nonlocal launch_called
            launch_called = True
            raise AssertionError("provider launch reached without current spend lock")

    monkeypatch.setattr(single_episode, "_load_single_episode_inputs", lambda _path: inputs)
    monkeypatch.setattr(single_episode, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(
        single_episode,
        "_require_signed_output_staging_proof",
        lambda **_kwargs: {"status": "passed", "blockers": []},
    )
    for name in (
        "BLUEPRINT_LAUNCH_PROOF_MODE",
        "BLUEPRINT_REQUIRE_PAID_SPEND_ADMISSION_LOCK",
        "BLUEPRINT_PAID_SPEND_ADMISSION_LOCK_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    secret_url = tmp_path / "signed-url.txt"
    secret_url.write_text("https://objects.example/episode.zip?signature=test")
    secret_url.chmod(0o600)
    release = tmp_path / "release.json"
    release.write_text(json.dumps({"resolved_digest_ref": single_episode.IMAGE_REF}))

    result = single_episode.run_single_episode(
        provider_name="runpod",
        episode_bundle=tmp_path / "episode.zip",
        provider_bundle_url_file=secret_url,
        provider_output_put_url_file=secret_url,
        provider_output_get_url_file=secret_url,
        release_evidence=release,
        provider_launch_request=tmp_path / "provider-request.json",
        preflight_bundle=tmp_path / "preflight.json",
        admission_out=tmp_path / "admission.json",
        bound_request_out=tmp_path / "bound.json",
        adapter_output=tmp_path / "result.json",
        pod_name="",
        execute=True,
    )

    assert result["status"] == "blocked"
    assert result["provider_mutations_performed"] == 0
    assert launch_called is False
    assert any(
        blocker.startswith("qualification_pre_spend:spend_admission:")
        for blocker in result["blockers"]
    )


def test_vast_remote_bootstrap_materializes_exact_overlay_bound_artifact(
    tmp_path: Path,
) -> None:
    inputs = {
        "runtime_package_overlay_xz_base64": "runtime-overlay-payload",
        "isaac_runtime_backend_overlay_gzip_base64": "backend-overlay-payload",
        "bootstrap_script": "upload_phase inputs_ready\necho episode-started\n",
    }

    artifact = single_episode._materialize_vast_remote_bootstrap(tmp_path, inputs)
    path = Path(artifact["path"])
    payload = path.read_bytes()

    assert artifact["sha256"] == hashlib.sha256(payload).hexdigest()
    assert artifact["size_bytes"] == len(payload)
    assert artifact["mode_is_0600"] is True
    assert artifact["runtime_overlays_embedded"] is True
    assert b"runtime-overlay-payload" in payload
    assert b"backend-overlay-payload" in payload
    assert b"provider_allocation_bound" in payload
    assert f"export {single_episode.RUNTIME_PACKAGE_OVERLAY_PAYLOAD_ENV}=".encode() not in payload
    assert f"export {single_episode.ISAAC_RUNTIME_OVERLAY_PAYLOAD_ENV}=".encode() not in payload
    assert (
        f"unset {single_episode.RUNTIME_PACKAGE_OVERLAY_PAYLOAD_ENV} "
        f"{single_episode.ISAAC_RUNTIME_OVERLAY_PAYLOAD_ENV}"
    ).encode() in payload
    assert single_episode.RUNTIME_PACKAGE_OVERLAY_PAYLOAD_PATH.encode() in payload
    assert single_episode.ISAAC_RUNTIME_OVERLAY_PAYLOAD_PATH.encode() in payload
    assert (
        f"export {single_episode.OVERLAY_PAYLOAD_TRANSPORT_ENV}="
        f"{single_episode.OVERLAY_PAYLOAD_FILE_TRANSPORT}"
    ).encode() in payload
    syntax = subprocess.run(
        ["bash", "-n"],
        input=payload,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr.decode("utf-8", errors="replace")


def test_oscar_runtime_dependency_repair_binds_full_hashed_writable_closure() -> None:
    script = single_episode._oscar_runtime_dependency_repair_script()
    lock_path = (
        Path(single_episode.__file__).resolve().parents[2]
        / single_episode.OSCAR_FOUNDATION_REQUIREMENTS_LOCK_RELATIVE_PATH
    )
    lock_bytes = lock_path.read_bytes()

    assert hashlib.sha256(lock_bytes).hexdigest() == (
        single_episode.OSCAR_FOUNDATION_REQUIREMENTS_LOCK_SHA256
    )
    requirement_lines = [
        line
        for line in lock_bytes.decode("utf-8").splitlines()
        if re.match(r"^[a-zA-Z0-9_.-]+==", line)
    ]
    assert len(requirement_lines) == (
        single_episode.OSCAR_FOUNDATION_REQUIREMENTS_LOCK_PACKAGE_COUNT
    )
    compressed_match = re.search(r"compressed_lock = '([A-Za-z0-9+/=]+)'", script)
    assert compressed_match is not None
    assert lzma.decompress(base64.b64decode(compressed_match.group(1))) == lock_bytes
    assert "--require-hashes" in script
    assert '"--target",\n                str(dependency_target)' in script
    assert single_episode.OSCAR_RUNTIME_DEPENDENCY_TARGET in script
    assert 'for metadata_dir in base_site_packages.glob("*.dist-info")' in script
    assert "locked_dependency_closure_exact" in script
    assert "foundation_lock_package_count_complete" in script
    assert "einops_runtime_version_exact" in script
    assert "dependency_graph_consistent" in script
    assert b"transformer-engine==" not in lock_bytes
    assert (
        f"pytest=={single_episode.OSCAR_RUNTIME_REQUIRED_PYTEST_VERSION} \\".encode() in lock_bytes
    )
    assert "pytest_locked_to_reviewed_version" in script
    assert "pytest_runtime_importable" in script
    assert "pytest_runtime_distribution_version_exact" in script
    assert "pytest_runtime_module_version_exact" in script
    assert "pytest_runtime_module_path_trusted" in script
    assert "importlib.invalidate_caches()" in script
    assert "sys.path_importer_cache.pop(str(dependency_target), None)" in script
    install_completion = script.index("install_returncode = completed.returncode")
    cache_invalidation = script.index("importlib.invalidate_caches()")
    importer_cache_eviction = script.index(
        "sys.path_importer_cache.pop(str(dependency_target), None)"
    )
    pytest_import = script.index("import pytest as pytest_module")
    assert (
        install_completion
        < cache_invalidation
        < importer_cache_eviction
        < pytest_import
    )
    assert 'importlib.metadata.version("pytest")' in script
    assert "oscar_pytest_runtime_contract_failed" in script
    assert "pytest_fresh_subprocess_returncode_zero" in script
    assert "pytest_fresh_distribution_version_exact" in script
    assert "pytest_fresh_module_version_exact" in script
    assert "pytest_fresh_module_path_in_dependency_target" in script
    assert "pytest_fresh_module_path_trusted" in script
    assert 'checks["uv_installer_required"] = bool(mismatches_before)' in script
    assert 'if checks["uv_installer_required"] and uv_path is None:' in script
    assert "oscar_pytest_fresh_subprocess_contract_failed" in script
    assert '[oscar_python, "-c", pytest_fresh_probe_source]' in script
    assert 'pytest_fresh_env["PYTHONPATH"]' in script
    assert 'groot_clean_env["PYTHONPATH"]' in script
    assert 'os.environ.get("BLUEPRINT_GROOT_OSCAR_GROOT_VENV_PYTHON"' in script
    assert '"/opt/gr00t-venv/bin/python"' in script
    assert "import gr00t.model" in script
    assert "from gr00t.model.gr00t_n1d7.gr00t_n1d7 import get_backbone_cls" in script
    assert script.index('"import gr00t.model\\n"') < script.index(
        '"import accelerate\\n"'
    )
    assert "groot_clean_pythonpath_excludes_oscar_dependency_target" in script
    assert "groot_clean_import_probe_returncode_zero" in script
    assert "groot_clean_import_modules_outside_oscar_dependency_target" in script
    assert "groot_clean_accelerate_resolved_from_groot_venv" in script
    assert "groot_runtime_dependency_isolation_failed" in script
    assert script.index("groot_clean_probe_source =") < script.index(
        "dependency_check_returncode = None"
    )
    assert "transformer_engine-2.0.0.dist-info" in script
    assert "transformer_engine_compat_shim_marker_verified" in script
    assert "transformer_engine_shim_source_root_verified" in script
    assert "transformer_engine_attention_api_verified" in script
    assert "transformer_engine_package_code_not_overlaid" in script
    assert "transformer_engine_metadata_files_exact" in script
    assert "transformer_engine_distribution_version_exact" in script
    assert "transformer_engine_distribution_metadata_path_exact" in script
    assert "1094a00f29f0c1cdb6da87ae97d126098c129dc03ee293f22e94d6d2966772e3" in script
    assert re.search(
        r'importlib\.metadata\.distribution\(\s*"transformer-engine"\s*\)',
        script,
    )
    assert re.search(
        r'importlib\.metadata\.version\(\s*"transformer-engine"\s*\)',
        script,
    )
    assert "official_oscar_runtime_dependency_repair_failed" in script
    assert "upload_phase oscar_runtime_dependency_install_started" in script
    assert "upload_phase oscar_runtime_dependencies_bound" in script
    scoped_python = single_episode._scoped_oscar_python_command()
    assert f"export PYTHONPATH={single_episode.OSCAR_RUNTIME_DEPENDENCY_TARGET!r}" not in script
    assert script.index(f"{scoped_python} - <<'PY'") < script.index(
        "import base64"
    )
    python_marker = f"{scoped_python} - <<'PY'\n"
    python_start = script.index(python_marker) + len(python_marker)
    python_end = script.index("\nPY\n", python_start)
    compile(
        script[python_start:python_end],
        "<oscar-runtime-dependency-repair>",
        "exec",
    )
    syntax = subprocess.run(
        ["bash", "-n"],
        input=script.encode("utf-8"),
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr.decode("utf-8", errors="replace")


def test_oscar_runtime_dependency_repair_metadata_exercises_distribution_apis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = single_episode._oscar_runtime_dependency_repair_script()
    python_marker = f"{single_episode.OSCAR_PYTHON} - <<'PY'\n"
    python_start = script.index(python_marker) + len(python_marker)
    python_end = script.index("\nPY\n", python_start)
    tree = ast.parse(script[python_start:python_end])
    metadata_files = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "transformer_engine_metadata_files"
            for target in node.targets
        ):
            metadata_files = ast.literal_eval(node.value)
            break
    assert metadata_files is not None

    metadata_dir = tmp_path / "transformer_engine-2.0.0.dist-info"
    metadata_dir.mkdir()
    for filename, text in metadata_files.items():
        (metadata_dir / filename).write_text(text, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    distribution = importlib.metadata.distribution("transformer-engine")

    assert distribution.metadata["Name"] == "transformer-engine"
    assert importlib.metadata.version("transformer-engine") == "2.0.0"
    assert Path(distribution._path).resolve() == metadata_dir.resolve()
    assert not (tmp_path / "transformer_engine").exists()


def test_oscar_runtime_dependency_repair_rejects_changed_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        single_episode,
        "OSCAR_FOUNDATION_REQUIREMENTS_LOCK_SHA256",
        "0" * 64,
    )

    with pytest.raises(
        ValueError,
        match="single_episode_oscar_foundation_requirements_lock_digest_mismatch",
    ):
        single_episode._oscar_runtime_dependency_repair_script()


def test_oscar_runtime_import_preflight_verifies_pinned_pytest_before_dynamic_config() -> None:
    script = single_episode._oscar_runtime_import_preflight_script()
    python_marker = f"{single_episode.OSCAR_PYTHON} - <<'PY'\n"
    python_start = script.index(python_marker) + len(python_marker)
    python_end = script.index("\nPY\n", python_start)
    source = script[python_start:python_end]

    compile(source, "<oscar-runtime-import-preflight>", "exec")
    pytest_import = "import pytest as pytest_module"
    pytest_distribution_probe = 'importlib.metadata.version("pytest")'
    pytest_version_check = "pytest_distribution_version == required_pytest_version"
    dynamic_config_guard = "if entrypoint.is_file() and not blockers:"
    dynamic_config_probe = "config_probe = subprocess.run"

    assert (
        f"required_pytest_version = {single_episode.OSCAR_RUNTIME_REQUIRED_PYTEST_VERSION!r}"
    ) in source
    assert pytest_import in source
    assert pytest_distribution_probe in source
    assert pytest_version_check in source
    assert "oscar_pytest_importable" in source
    assert "oscar_pytest_distribution_version_exact" in source
    assert "oscar_pytest_module_version_exact" in source
    assert "oscar_pytest_runtime_contract_failed" in source
    assert source.index(pytest_import) < source.index(pytest_distribution_probe)
    assert source.index(pytest_distribution_probe) < source.index(dynamic_config_guard)
    assert source.index(dynamic_config_guard) < source.index(dynamic_config_probe)


def test_image_healthcheck_uses_only_the_baked_reason2_hf_cache() -> None:
    healthcheck = (
        "python3 /opt/blueprint/groot_oscar_closed_loop_image_healthcheck.py"
    )
    runtime_hf_home = f"{single_episode.OSCAR_RUNTIME_ASSET_CACHE_ROOT}/hf_home"
    script = f"export HF_HOME={runtime_hf_home}\n{healthcheck} --require-cuda\n"

    resolved = single_episode._pin_bootstrap_interpreters(script)
    scoped_healthcheck = (
        f"HF_HOME={single_episode.SEALED_GROOT_HF_HOME} "
        f"HF_HUB_CACHE={single_episode.SEALED_GROOT_HF_HUB_CACHE} "
        f"HUGGINGFACE_HUB_CACHE={single_episode.SEALED_GROOT_HF_HUB_CACHE} "
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
        f"COSMOS_BACKBONE_REPO={single_episode.SEALED_COSMOS_BACKBONE_REPO} "
        "COSMOS_BACKBONE_REVISION="
        f"{single_episode.SEALED_COSMOS_BACKBONE_REVISION} "
        f"{single_episode.OSCAR_PYTHON} "
        "/opt/blueprint/groot_oscar_closed_loop_image_healthcheck.py --require-cuda"
    )

    assert resolved == f"export HF_HOME={runtime_hf_home}\n{scoped_healthcheck}\n"
    assert resolved.count(scoped_healthcheck) == 1
    assert single_episode.SEALED_GROOT_HF_HOME != runtime_hf_home

    with pytest.raises(
        ValueError,
        match="single_episode_bootstrap_healthcheck_marker_ambiguous",
    ):
        single_episode._pin_bootstrap_interpreters("true\n")
    with pytest.raises(
        ValueError,
        match="single_episode_bootstrap_healthcheck_marker_ambiguous",
    ):
        single_episode._pin_bootstrap_interpreters(f"{healthcheck}\n{healthcheck}\n")


def test_oscar_runtime_assets_prepare_once_then_fail_closed_offline_preflight() -> None:
    script = single_episode._oscar_runtime_asset_prepare_and_preflight_script()
    expected_environment = single_episode.offline_runtime_environment(
        single_episode.OSCAR_RUNTIME_ASSET_CACHE_ROOT
    )

    assert "oscar_runtime_asset_contract.py" in (single_episode.RUNTIME_PACKAGE_OVERLAY_MODULES)
    assert "oscar_wam_provider_bundle.py" in (
        single_episode.RUNTIME_PACKAGE_OVERLAY_MODULES
    )
    assert single_episode.OSCAR_RUNTIME_ASSET_PREPARE_ARTIFACT in script
    assert single_episode.OSCAR_RUNTIME_ASSET_PREFLIGHT_ARTIFACT in script
    assert "runtime_evidence = prepare_runtime_asset_cache" in script
    assert "enumerate((0, 10, 30), start=1)" in script
    assert '"bounded_resume_retry_enabled": True' in script
    assert '"prepare_attempt_count": len(prepare_attempts)' in script
    assert "offline_evidence = offline_preflight(" in script
    assert "processor_probe=True" in script
    assert "dcp_metadata_probe=True" in script
    assert single_episode.OSCAR_RUNTIME_ASSET_CHECKPOINT_ROOT in script
    assert "official_oscar_runtime_asset_prepare_failed" in script
    assert "official_oscar_runtime_asset_offline_preflight_failed" in script
    assert 'BLUEPRINT_WORKER_FAILURE="official_oscar_runtime_asset_prepare_failed"' in script
    assert (
        'BLUEPRINT_WORKER_FAILURE="official_oscar_runtime_asset_offline_preflight_failed"' in script
    )
    for name, value in expected_environment.items():
        assert f"export {name}={shlex.quote(value)}" in script

    prepare_started = "upload_phase oscar_runtime_asset_prepare_started"
    prepare_call = "runtime_evidence = prepare_runtime_asset_cache"
    prepared = "upload_phase oscar_runtime_assets_prepared"
    offline_export = "export HF_HUB_OFFLINE=1"
    offline_call = "offline_evidence = offline_preflight("
    preflight_passed = "upload_phase oscar_runtime_asset_offline_preflight_passed"
    assert script.index(prepare_started) < script.index(prepare_call)
    assert script.index(prepare_call) < script.index(prepared)
    assert script.index(prepared) < script.index(offline_export)
    assert script.index(offline_export) < script.index(offline_call)
    assert script.index(offline_call) < script.index(preflight_passed)

    python_marker = f"{single_episode.OSCAR_PYTHON} - <<'PY'\n"
    embedded_sources = [tail.split("\nPY\n", 1)[0] for tail in script.split(python_marker)[1:]]
    assert len(embedded_sources) == 2
    for index, source in enumerate(embedded_sources):
        compile(source, f"<oscar-runtime-asset-{index}>", "exec")
    syntax = subprocess.run(
        ["bash", "-n"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr


def test_workspace_storage_is_sized_for_runpod_without_changing_vast_semantics() -> None:
    assert single_episode._workspace_volume_gb("runpod") == 60
    assert single_episode.RUNPOD_WORKSPACE_VOLUME_GB == 60
    assert single_episode._workspace_volume_gb("vast") == 20
    assert single_episode.VAST_IGNORED_LEGACY_VOLUME_GB == 20
    with pytest.raises(ValueError, match="single_episode_provider_unsupported"):
        single_episode._workspace_volume_gb("unknown")


def test_vast_remote_bootstrap_runs_first_external_command_with_payload_over_128k(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_path = tmp_path / "runtime-overlay.txt"
    isaac_path = tmp_path / "isaac-overlay.txt"
    first_external_marker = tmp_path / "first-external-ran.txt"
    monkeypatch.setattr(
        single_episode,
        "RUNTIME_PACKAGE_OVERLAY_PAYLOAD_PATH",
        str(runtime_path),
    )
    monkeypatch.setattr(
        single_episode,
        "ISAAC_RUNTIME_OVERLAY_PAYLOAD_PATH",
        str(isaac_path),
    )
    oversized_payload = "A" * (single_episode.LINUX_MAX_ARG_STRLEN_BYTES + 4096)
    script = single_episode._vast_remote_bootstrap_script(
        {
            "runtime_package_overlay_xz_base64": oversized_payload,
            "isaac_runtime_backend_overlay_gzip_base64": oversized_payload,
            "bootstrap_script": (
                "upload_phase() { :; }\n"
                "/bin/sh -c 'printf first-external-ran > \"$1\"' _ "
                f"{shlex.quote(str(first_external_marker))}\n"
                "upload_phase inputs_ready\n"
            ),
        }
    )

    exported_assignments = [
        line
        for line in script.splitlines()
        if re.match(r"^\s*export\s+[A-Za-z_][A-Za-z0-9_]*=", line)
    ]
    assert exported_assignments
    assert all(
        len(line.encode("utf-8")) < single_episode.LINUX_MAX_ARG_STRLEN_BYTES
        for line in exported_assignments
    )
    assert f"export {single_episode.RUNTIME_PACKAGE_OVERLAY_PAYLOAD_ENV}=" not in script
    assert f"export {single_episode.ISAAC_RUNTIME_OVERLAY_PAYLOAD_ENV}=" not in script

    result = subprocess.run(
        ["bash"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
        env={
            "CONTAINER_ID": "12345",
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            single_episode.RUNTIME_PACKAGE_OVERLAY_PAYLOAD_ENV: "inherited-exported-value",
            single_episode.ISAAC_RUNTIME_OVERLAY_PAYLOAD_ENV: "inherited-exported-value",
        },
    )

    assert result.returncode == 0, result.stderr
    assert first_external_marker.read_text(encoding="utf-8") == "first-external-ran"
    assert runtime_path.read_text(encoding="ascii") == oversized_payload
    assert isaac_path.read_text(encoding="ascii") == oversized_payload


def test_vast_remote_bootstrap_downloader_fails_closed_on_sha_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://objects.example/bootstrap.sh?signature=secret"

    class Response:
        status = 200
        headers = {"Content-Length": "12"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def geturl(self) -> str:
            return url

        def read(self, _limit: int) -> bytes:
            return b"real-payload"

    class Opener:
        def open(self, _request, *, timeout: int):
            assert timeout == 90
            return Response()

    monkeypatch.setenv(single_episode.VAST_BOOTSTRAP_URL_ENV, url)
    monkeypatch.setenv(single_episode.VAST_BOOTSTRAP_SHA256_ENV, "0" * 64)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_args: Opener())

    with pytest.raises(SystemExit, match="vast_remote_bootstrap_sha256_mismatch"):
        exec(single_episode.VAST_BOOTSTRAP_DOWNLOADER_PYTHON, {})


def test_vast_signed_bootstrap_keeps_provider_env_and_args_below_32kb(
    tmp_path: Path,
) -> None:
    downloader = single_episode._vast_signed_bootstrap_downloader_script()
    assert 'if [ "$(id -u)" -eq 0 ]; then' in downloader
    assert "vast_bootstrap_requires_root" not in downloader
    assert "vast_bootstrap_nonroot_ssh_repair_skipped" in downloader
    assert "install -d -o root -g root -m 700 /root/.ssh" in downloader
    assert "chown root:root /root/.ssh/authorized_keys" in downloader
    assert "chmod 600 /root/.ssh/authorized_keys" in downloader
    assert "vast_root_ssh_directory_symlink_forbidden" in downloader
    assert "vast_authorized_keys_not_regular_file" in downloader
    assert downloader.endswith("\nPY\n")
    syntax = subprocess.run(
        ["bash", "-n"],
        input=downloader,
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr
    assert downloader.index("chmod 600 /root/.ssh/authorized_keys") < downloader.index(
        "BLUEPRINT_VAST_REMOTE_BOOTSTRAP_SIGNED_GET_URL"
    )
    repair_prefix, separator, _download = downloader.partition(
        f"{single_episode.SYSTEM_PYTHON} - <<'PY'\n"
    )
    assert separator
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_id = fake_bin / "id"
    fake_id.write_text("#!/bin/sh\nprintf '1000\\n'\n", encoding="utf-8")
    fake_id.chmod(0o700)
    nonroot = subprocess.run(
        ["bash", "-c", repair_prefix + "printf 'download-start-reached\\n'\n"],
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert nonroot.returncode == 0, nonroot.stderr
    assert "vast_bootstrap_nonroot_ssh_repair_skipped" in nonroot.stderr
    assert nonroot.stdout == "download-start-reached\n"
    spec = RenderLaunchSpec(
        name="single-episode",
        image="example/image@sha256:" + "a" * 64,
        env={
            "BLUEPRINT_EVAL_MANIFEST_URI": "https://objects.example/bundle?sig=A",
            "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL": (
                "https://objects.example/output?sig=B"
            ),
            single_episode.VAST_BOOTSTRAP_URL_ENV: ("https://objects.example/bootstrap?sig=C"),
            single_episode.VAST_BOOTSTRAP_SHA256_ENV: "d" * 64,
        },
        bootstrap_argv=[
            "-lc",
            downloader,
        ],
    )

    request = VastRenderProvider().build_request(spec, tmp_path)
    create = request["create_payload"]
    total = len(create["args_str"].encode("utf-8")) + sum(
        len(str(key).encode("utf-8")) + len(str(value).encode("utf-8")) + 2
        for key, value in create["env"].items()
    )

    assert request["bootstrap_transport"] == "plain"
    assert total < 32 * 1024
    assert single_episode.RUNTIME_PACKAGE_OVERLAY_PAYLOAD_ENV not in create["env"]
    assert single_episode.ISAAC_RUNTIME_OVERLAY_PAYLOAD_ENV not in create["env"]
    assert "BLUEPRINT_VAST_BOOTSTRAP_GZIP_BASE64" not in create["env"]


def test_launch_session_nonce_file_matches_worker_bootstrap_session(
    tmp_path: Path,
) -> None:
    (tmp_path / "provider_bundle_url.txt").write_text(
        "https://objects.example/bundle?sig=A", encoding="utf-8"
    )
    (tmp_path / "provider_output_put_url.txt").write_text(
        "https://objects.example/output?sig=B", encoding="utf-8"
    )
    start_frame = tmp_path / "frame.png"
    start_frame.write_bytes(b"frame")
    nonce = "single-g1-kitchen-session-123"
    artifact = single_episode._materialize_launch_session_nonce(tmp_path, nonce)
    plan = {
        "env": {},
        "groot_server_command": ["true"],
        "gear_sonic_controller_command": ["true"],
        "isaac_task_executor_command": ["true"],
        "closed_loop_command": ["true"],
    }
    spec = single_episode.build_launch_spec(
        job_dir=tmp_path,
        image_ref=single_episode.IMAGE_REF,
        start_frame=start_frame,
        route_payload={},
        task_prompt=single_episode.TASK_PROMPT,
        plan=plan,
        launch_nonce=nonce,
    )

    nonce_path = Path(artifact["path"])
    assert nonce_path.read_text(encoding="utf-8") == nonce
    assert nonce_path.stat().st_mode & 0o777 == 0o600
    assert artifact["sha256"] == hashlib.sha256(nonce.encode()).hexdigest()
    assert spec.env["BLUEPRINT_LAUNCH_SESSION_ID"] == nonce
    assert "BLUEPRINT_LAUNCH_SESSION_ID" in spec.bootstrap_script


def test_run_single_episode_has_no_external_runtime_patch_transport() -> None:
    parameters = inspect.signature(single_episode.run_single_episode).parameters

    assert "runtime_patch" not in parameters
    assert "provider_runtime_patch_url_file" not in parameters
    assert {
        "provider_name",
        "episode_bundle",
        "provider_bundle_url_file",
        "provider_output_put_url_file",
        "provider_output_get_url_file",
        "provider_bootstrap_url_file",
        "release_evidence",
        "provider_launch_request",
        "preflight_bundle",
        "admission_out",
        "bound_request_out",
        "adapter_output",
        "pod_name",
        "execute",
        "qualification_checkpoint_report",
        "qualification_checkpoint_part_stage_dirs",
    } == set(parameters)


def test_legacy_direct_episode_image_blocks_before_allocation_without_source_seal() -> None:
    assert single_episode.IMAGE_DIGEST not in (
        single_episode.OSCAR_RUNTIME_SOURCE_SEAL_CAPABLE_IMAGE_DIGESTS
    )
    source = inspect.getsource(single_episode.run_single_episode)
    assert "single_episode_pinned_image_missing_oscar_runtime_source_seal" in source
    assert source.index("single_episode_pinned_image_missing_oscar_runtime_source_seal") < (
        source.index("get_render_provider")
    )


def test_qualification_checkpoint_restore_binds_ordered_parts_without_urls_in_evidence(
    tmp_path: Path,
) -> None:
    parts = [
        {"part_number": 1, "size_bytes": 4, "sha256": "a" * 64},
        {"part_number": 2, "size_bytes": 3, "sha256": "b" * 64},
    ]
    for part in parts:
        part["upload"] = {
            "status": "passed",
            "uploaded_size_bytes": part["size_bytes"],
            "uploaded_sha256": part["sha256"],
        }
    report = {
        "status": "completed",
        "open_loop_qualification": {
            "status": "passed",
            "steps": 120,
            "exact_owned_training_trajectory_only": True,
            "isaac_registered_transition_not_proven": True,
            "mse_ratio": 0.2,
            "mae_ratio": 0.3,
            "maximum_error_ratio": 0.8,
        },
        "checkpoint_archive": {
            "sha256": "c" * 64,
            "size_bytes": 7,
            "upload": {
                "status": "passed",
                "transport": "ordered_parts",
                "uploaded_sha256": "c" * 64,
                "uploaded_size_bytes": 7,
                "parts": parts,
            },
        },
    }
    report_path = tmp_path / "worker_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    stages = []
    for index in range(2):
        stage = tmp_path / f"part-{index + 1}"
        stage.mkdir()
        url_path = stage / "provider_output_get_url.txt"
        url_path.write_text(
            f"https://objects.example/checkpoint-part-{index + 1}?sig=secret",
            encoding="utf-8",
        )
        url_path.chmod(0o600)
        stages.append(stage)

    evidence, urls = single_episode._qualification_checkpoint_restore_evidence(
        worker_report_path=report_path,
        part_stage_dirs=tuple(stages),
    )

    assert evidence["status"] == "qualified_for_isaac_evaluation"
    assert evidence["checkpoint_path"] == single_episode.REMOTE_FINAL_CHECKPOINT
    assert evidence["parts"] == [
        {key: value for key, value in part.items() if key != "upload"}
        for part in parts
    ]
    assert len(urls) == 2
    assert "objects.example" not in json.dumps(evidence)
    script = single_episode._qualification_checkpoint_restore_script(evidence)
    assert single_episode.QUALIFICATION_CHECKPOINT_PART_GET_URLS_ENV in script
    assert "objects.example" not in script
    assert "qualification_checkpoint_part_digest_mismatch" in script
    assert "qualification_checkpoint_archive_digest_mismatch" in script
    assert script.index("destination.parent.mkdir") < script.index("tempfile.mkdtemp")

    preflight = single_episode.qualification_checkpoint_preflight_script()
    assert (
        "export BLUEPRINT_GROOT_OSCAR_SONIC_CHECKPOINT="
        + single_episode.REMOTE_FINAL_CHECKPOINT
    ) in preflight
    assert (
        "export BLUEPRINT_GROOT_CHECKPOINT_PREFLIGHT_ARTIFACT="
        "/workspace/closed_loop_out/qualification_checkpoint_preflight.json"
    ) in preflight
    assert "groot_sonic_checkpoint_preflight.v1" in preflight
    assert "pinned_get_backbone_cls_resolved" in preflight
    assert preflight.index("BLUEPRINT_GROOT_OSCAR_SONIC_CHECKPOINT") < preflight.index(
        "groot_sonic_checkpoint_preflight.v1"
    )
    assert preflight.index("groot_checkpoint_preflight_passed") < preflight.index(
        "qualification_checkpoint_preflight_passed"
    )


def test_qualification_checkpoint_override_pins_only_fixed_model_path() -> None:
    plan = {"groot_server_command": ["python", "serve.py", "--model-path", "/old"]}

    updated = single_episode._apply_qualification_checkpoint_override(
        plan, single_episode.REMOTE_FINAL_CHECKPOINT
    )

    assert updated["groot_server_command"][-1] == single_episode.REMOTE_FINAL_CHECKPOINT
    assert updated["qualification_checkpoint_override"]["task_compatibility_claimed"] is False
    with pytest.raises(ValueError, match="path_not_fixed"):
        single_episode._apply_qualification_checkpoint_override(plan, "/tmp/model")


def test_signed_output_staging_proof_binds_clean_sentinel_and_untouched_output(
    tmp_path: Path,
) -> None:
    put_url = (
        "https://objects.example/bucket/prefix/"
        "runpod_provider_runtime_output_0123456789abcdef0123456789abcdef.zip"
        "?X-Amz-Date=20290101T000000Z&X-Amz-Expires=43200"
        "&X-Amz-Signature=put-secret"
    )
    get_url = put_url.replace("put-secret", "get-secret")
    put_path = tmp_path / "provider_output_put_url.txt"
    get_path = tmp_path / "provider_output_get_url.txt"
    for path, value in ((put_path, put_url), (get_path, get_url)):
        path.write_text(value, encoding="utf-8")
        path.chmod(0o600)
    sentinel_sha = "a" * 64
    manifest = {
        "schema_version": single_episode.OBJECT_STORE_STAGING_SCHEMA_VERSION,
        "status": "completed",
        "blockers": [],
        "provider_output_put_url_file": {
            "path": str(put_path),
            "present": True,
            "mode_is_0600": True,
            "size_bytes": put_path.stat().st_size,
            "mtime_ns": put_path.stat().st_mtime_ns,
        },
        "provider_output_get_url_file": {
            "path": str(get_path),
            "present": True,
            "mode_is_0600": True,
            "size_bytes": get_path.stat().st_size,
            "mtime_ns": get_path.stat().st_mtime_ns,
        },
        "signed_output_round_trip": {
            "schema_version": single_episode.SIGNED_OUTPUT_ROUND_TRIP_SCHEMA_VERSION,
            "status": "passed",
            "blockers": [],
            "sentinel_sha256": sentinel_sha,
            "put": {"status": "passed"},
            "get": {
                "status": "passed",
                "exact_bytes_and_sha256": True,
                "received_sha256": sentinel_sha,
            },
            "cleanup": {"status": "passed", "absence_confirmed": True},
            "actual_output_key_was_not_used": True,
            "raw_signed_urls_recorded": False,
        },
        "fresh_output_key_absence": {
            "status": "passed",
            "absence_confirmed": True,
        },
        "output_key_run_unique": True,
        "output_key": (
            "prefix/runpod_provider_runtime_output_0123456789abcdef0123456789abcdef.zip"
        ),
        "output_url_object_binding_sha256": (
            single_episode.signed_output_object_binding_sha256(put_url, get_url)
        ),
        "presigned_url_expiry": {"expires_at": "2030-01-01T00:00:00Z"},
    }
    (tmp_path / single_episode.SIGNED_OUTPUT_STAGING_MANIFEST_NAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    proof = single_episode._require_signed_output_staging_proof(
        provider_output_put_url_file=put_path,
        provider_output_get_url_file=get_path,
        put_url=put_url,
        get_url=get_url,
        now_epoch=0,
    )

    assert proof["status"] == "passed"
    assert proof["sentinel_cleanup_absence_confirmed"] is True
    assert proof["actual_output_key_was_not_used_for_probe"] is True
    assert "put-secret" not in json.dumps(proof, sort_keys=True)
    assert "get-secret" not in json.dumps(proof, sort_keys=True)

    get_path.write_text(get_url + "&changed=1", encoding="utf-8")
    get_path.chmod(0o600)
    with pytest.raises(
        ValueError,
        match="single_episode_signed_output_get_url_file_not_staging_bound",
    ):
        single_episode._require_signed_output_staging_proof(
            provider_output_put_url_file=put_path,
            provider_output_get_url_file=get_path,
            put_url=put_url,
            get_url=get_url + "&changed=1",
            now_epoch=0,
        )


def test_single_episode_bootstrap_requires_live_isaac_frame_and_camera_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "episode.zip"
    selection_sha = "a" * 64
    scenario = tmp_path / "selected_isaac_scenario.json"
    scenario.write_text(
        json.dumps(
            {
                "task_id": "microwave_door",
                "source_selection_sha256": selection_sha,
                "perception_target_prompts": [
                    "microwave door",
                    "microwave cavity",
                ],
                "target_object_ids": ["microwave fixture"],
                "affordance_object_ids": ["handle", "door"],
                "stance_distance_candidates_m": [0.35, 0.5, 0.63, 0.78],
                "accepted_stance_contract": {
                    "status": "accepted",
                    "pose_xyz": [-1.229635, 1.471274, 0.84],
                    "stance_focus_xyz": [-1.591312, 1.471274, 1.241574],
                    "resolved_target": {
                        "prim_path": "/root/Microwave017",
                        "target_object_id": "microwave fixture",
                    },
                    "resolved_affordance": {
                        "prim_path": "/root/Microwave017/Microwave017_Door",
                        "target_object_id": "door",
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    plan = {
        "sealed_active": True,
        "blockers": [],
        "image_ref": single_episode.IMAGE_REF,
        "env": {
            "PYTHONPATH": "/opt/OSCAR",
            "BLUEPRINT_OSCAR_WAM_SOURCE_URL": "https://example.invalid/not-oscar.git",
            "BLUEPRINT_OSCAR_WAM_SOURCE_REF": "main",
        },
        "groot_server_command": ["true"],
        "isaac_task_executor_command": [
            single_episode.ISAAC_PYTHON,
            "-m",
            single_episode.ISAAC_TASK_EXECUTOR_MODULE,
            "--stage",
            "/workspace/kitchen/KitchenRoom.usd",
        ],
        "gear_sonic_controller_command": ["true"],
        "closed_loop_command": [
            "python",
            "-m",
            "blueprint_pipeline.oscar_isaac_closed_loop_eval",
            "--output-dir",
            "/workspace/closed_loop_out",
            "--oscar-repo",
            "/opt/OSCAR",
            "--checkpoint",
            "/opt/blueprint/ckpts/oscar",
            "--task-prompt",
            "stale prepared-plan prompt",
            "--perception-target-prompt",
            "stale prepared-plan object",
            "--harness-backend-kind",
            "real_provider_probe",
            "--require-real-perception-backend",
            "--require-sam3-completed",
            "--require-da3-completed",
        ],
    }
    attempt = {
        "attempt_id": "episode_001",
        "image_digest": single_episode.IMAGE_DIGEST,
        "selected_task_id": "microwave_door",
        "artifacts": {
            "selection": {"sha256": selection_sha},
            "scenario": {
                "path": str(scenario),
                "sha256": hashlib.sha256(scenario.read_bytes()).hexdigest(),
            },
        },
    }
    task_contract = {
        "schema_version": "g1_kitchen_task_success_contract.v1",
        "task_id": "microwave_door",
        "task_kind": "manipulation",
        "source_selection_sha256": selection_sha,
        "registered_criteria": [
            {
                "criterion_id": "microwave_door_open_angle",
                "observable_transition": "articulation_angle_rad",
                "comparison": "increase_at_least",
                "tolerance": 0.35,
                "unit": "rad",
            }
        ],
    }
    members: dict[str, bytes] = {
        "initial_policy_frame.png": b"exact-staged-frame",
        "route.json": json.dumps(
            {
                "route_points": [
                    [-1.229635, 1.471274, 0.84],
                    [-1.229635, 1.471274, 0.84],
                ],
                "accepted_stance_yaw_rad": 3.141593,
            }
        ).encode(),
        "seed_provenance.json": json.dumps({"seed": 1001}).encode(),
        "sealed_launch_plan.json": json.dumps(plan).encode(),
        "task_success_contract.json": json.dumps(task_contract).encode(),
        "attempt_input_manifest_episode_001.json": json.dumps(attempt).encode(),
        "episode_review_builder.py": b"",
        "kitchen/KitchenRoom.usd": b"#usda 1.0\n",
    }
    with zipfile.ZipFile(bundle, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    monkeypatch.setattr(
        single_episode,
        "BUNDLE_SHA256",
        hashlib.sha256(bundle.read_bytes()).hexdigest(),
    )
    inputs = single_episode._load_single_episode_inputs(bundle)
    script = inputs["bootstrap_script"]
    terminal_branch = script.split('if [ "$1" = "runner_done" ]', 1)[1].split(
        "else", 1
    )[0]
    assert "python3 /workspace/upload_progress.py" in terminal_branch
    assert "declare -F cleanup" in terminal_branch
    assert "while :; do sleep 300; done" in terminal_branch
    assert terminal_branch.index("python3 /workspace/upload_progress.py") < (
        terminal_branch.index("while :; do sleep 300; done")
    )
    task_compatibility = inputs["manipulation_policy_task_compatibility"]
    assert inputs["route"]["route_points"][0] == pytest.approx(
        [-0.961312, 1.471274, 0.84]
    )
    assert inputs["route"]["route_points"][1] == pytest.approx(
        [-0.961312, 1.471274, 0.84]
    )
    stance_evidence = dict(inputs["route"]["egocentric_camera_safe_stance"])
    selected_pose = stance_evidence.pop("selected_pose_xyz")
    assert selected_pose == pytest.approx([-0.961312, 1.471274, 0.84])
    assert stance_evidence == {
        "status": "selected_from_scenario_candidates",
        "source_pose_xyz": [-1.229635, 1.471274, 0.84],
        "stance_focus_xyz": [-1.591312, 1.471274, 1.241574],
        "selected_stance_distance_m": 0.63,
        "minimum_stance_distance_m": 0.63,
        "camera_mount_changed": False,
        "surrogate": False,
    }
    assert task_compatibility["status"] == "blocked"
    assert task_compatibility["blockers"] == [
        single_episode.MANIPULATION_POLICY_TASK_COMPATIBILITY_BLOCKER
    ]
    assert task_compatibility["sealed_checkpoint"] == {
        "repo": single_episode.SEALED_SONIC_CHECKPOINT_REPO,
        "revision": single_episode.SEALED_SONIC_CHECKPOINT_REVISION,
    }
    syntax = subprocess.run(
        ["bash", "-n"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    # The sealed bundle still satisfies the legacy input contract, but its
    # staged frame is never materialized or accepted by the direct runtime.
    assert inputs["start_frame"] == b"exact-staged-frame"
    assert "BLUEPRINT_INITIAL_POLICY_FRAME_B64" not in script
    assert hashlib.sha256(b"exact-staged-frame").hexdigest() not in script
    assert "live_isaac_observation_output_not_regular_file" in script
    assert script.index("live_output.unlink()") < script.index("upload_phase inputs_ready")
    assert "cp /workspace/attempt_input_manifest_episode_001.json" in script
    assert "attempt['prepared_launch_nonce'] = prepared_launch_nonce" in script
    assert "attempt['allocation_launch_session_id'] = launch_session_id" in script
    assert "attempt['launch_nonce'] = active_launch_nonce" in script
    assert "attempt['qualification_attempt_nonce'] = qualification_nonce or None" in script
    assert "launch_nonce_rebound_to_current_allocation_session" in script
    assert "launch_nonce_rebound_to_qualification_attempt" in script
    assert f"timeout {single_episode.EPISODE_TIMEOUT_SECONDS}" in script
    assert "--oscar-seed 1001" in script
    assert "/workspace/closed_loop_out/episode_001" in script
    assert (
        "/opt/oscar-venv/bin/python -m "
        "blueprint_pipeline.groot_oscar_episode_review "
        "/workspace/closed_loop_out/episode_001"
    ) in script
    assert (
        f"env PYTHONPATH={single_episode.OSCAR_RUNTIME_DEPENDENCY_TARGET}:"
        "/workspace/runtime_overlay/package:/opt/wbc:/opt/OSCAR "
        "/opt/oscar-venv/bin/python -m "
        "blueprint_pipeline.groot_oscar_episode_review "
        "/workspace/closed_loop_out/episode_001"
    ) in script
    assert "/workspace/episode_review_builder.py /workspace/closed_loop_out" not in script
    assert "attempt_input_manifest_smoke" not in script
    assert "--wam-consistency-command" not in script
    assert "--allow-wam-consistency-scoring" not in script
    assert "--require-forward-inverse-consistency" not in script
    assert "BLUEPRINT_WAM_EPISODE_CONSISTENCY_COMMAND" not in script
    assert "/opt/blueprint/ckpts/oscar" in script
    assert inputs["plan"]["env"]["PYTHONPATH"] == (
        "/workspace/runtime_overlay/package:/opt/wbc:/opt/OSCAR"
    )
    assert inputs["plan"]["env"][single_episode.OSCAR_CUDNN_LIB_DIR_ENV] == (
        single_episode.OSCAR_CUDNN_LIB_DIR
    )
    assert {
        name: inputs["plan"]["env"].get(name)
        for name in single_episode.offline_runtime_environment(
            single_episode.OSCAR_RUNTIME_ASSET_CACHE_ROOT
        )
    } == single_episode.offline_runtime_environment(single_episode.OSCAR_RUNTIME_ASSET_CACHE_ROOT)
    assert "libcudnn_graph.so.9" in script
    assert "ctypes.CDLL" in script
    assert re.search(
        r'importlib\.metadata\.version\(\s*"transformer-engine"\s*\)',
        script,
    )
    assert "oscar_transformer_engine_distribution_version_exact" in script
    assert "oscar_transformer_engine_distribution_metadata_invalid" in script
    assert 'str(entrypoint), "--help"' in script
    assert "worldsim._src.configs.agibot_control.config" in script
    assert "make_config()" in script
    assert "oscar_dynamic_config_constructible" in script
    assert single_episode.OSCAR_RUNTIME_DEPENDENCY_ARTIFACT in script
    assert "upload_phase oscar_runtime_dependency_install_started" in script
    assert "upload_phase oscar_runtime_dependencies_bound" in script
    assert "upload_phase oscar_runtime_asset_prepare_started" in script
    assert "upload_phase oscar_runtime_assets_prepared" in script
    assert "upload_phase oscar_runtime_asset_offline_preflight_passed" in script
    assert single_episode.OSCAR_RUNTIME_ASSET_PREPARE_ARTIFACT in script
    assert single_episode.OSCAR_RUNTIME_ASSET_PREFLIGHT_ARTIFACT in script
    assert single_episode.OSCAR_RUNTIME_PREFLIGHT_ARTIFACT in script
    assert "upload_phase oscar_runtime_import_preflight_passed" in script
    assert script.index("upload_phase oscar_runtime_dependencies_bound") < script.index(
        "upload_phase oscar_runtime_asset_prepare_started"
    )
    assert script.index("upload_phase oscar_runtime_asset_prepare_started") < script.index(
        "upload_phase oscar_runtime_assets_prepared"
    )
    assert script.index("upload_phase oscar_runtime_assets_prepared") < script.index(
        "upload_phase oscar_runtime_asset_offline_preflight_passed"
    )
    assert script.index("upload_phase oscar_runtime_asset_offline_preflight_passed") < script.index(
        "worldsim._src.configs.agibot_control.config"
    )
    assert script.index("upload_phase oscar_runtime_asset_offline_preflight_passed") < (
        script.index("upload_phase oscar_runtime_import_preflight_passed")
    )
    preflight_marker = f"{single_episode._scoped_oscar_python_command()} - <<'PY'\n"
    preflight_start = script.index(
        preflight_marker,
        script.index("import ctypes") - len(preflight_marker),
    )
    preflight_source_start = preflight_start + len(preflight_marker)
    preflight_source_end = script.index("\nPY\n", preflight_source_start)
    preflight_source = script[preflight_source_start:preflight_source_end]
    compile(
        preflight_source,
        "<oscar-runtime-import-preflight>",
        "exec",
    )
    distribution_probe = 'importlib.metadata.distribution("transformer-engine")'
    version_probe = 'importlib.metadata.version("transformer-engine")'
    cli_probe = '[sys.executable, str(entrypoint), "--help"]'
    assert preflight_source.index(distribution_probe) < preflight_source.index(cli_probe)
    assert preflight_source.index(version_probe) < preflight_source.index(cli_probe)
    assert 'checkpoint_loaded": False' in script
    assert 'oscar_inference_ran": False' in script
    expected_oscar_pythonpath = (
        f"{single_episode.OSCAR_RUNTIME_DEPENDENCY_TARGET}:"
        "/workspace/runtime_overlay/package:/opt/wbc:/opt/OSCAR"
    )
    expected_base_pythonpath = (
        "/workspace/runtime_overlay/package:/opt/wbc:/opt/OSCAR"
    )
    assert inputs["plan"]["env"][single_episode.GROOT_RUNTIME_PYTHONPATH_ENV] == (
        expected_base_pythonpath
    )
    assert inputs["plan"]["env"][single_episode.GROOT_VENV_ROOT_ENV] == (
        single_episode.GROOT_VENV_ROOT
    )
    assert single_episode.GROOT_VENV_ROOT == "/opt/gr00t-venv"
    assert inputs["plan"]["env"][
        single_episode.OSCAR_RUNTIME_DEPENDENCY_TARGET_ENV
    ] == single_episode.OSCAR_RUNTIME_DEPENDENCY_TARGET
    assert inputs["plan"]["groot_server_command"] == [
        "env",
        f"PYTHONPATH={expected_base_pythonpath}",
        "true",
    ]
    assert inputs["plan"]["closed_loop_command"][:7] == [
        "env",
        f"PYTHONPATH={expected_oscar_pythonpath}",
        "timeout",
        str(single_episode.EPISODE_TIMEOUT_SECONDS),
        single_episode.OSCAR_PYTHON,
        "-m",
        "blueprint_pipeline.oscar_isaac_closed_loop_eval",
    ]
    closed_loop_command = inputs["plan"]["closed_loop_command"]
    num_frames_index = closed_loop_command.index("--num-frames")
    assert closed_loop_command[num_frames_index + 1] == str(single_episode.DIRECT_OSCAR_NUM_FRAMES)
    assert single_episode.DIRECT_OSCAR_NUM_FRAMES == 81
    assert (single_episode.DIRECT_OSCAR_NUM_FRAMES - 1) % 4 == 0
    oscar_num_steps_index = closed_loop_command.index("--oscar-num-steps")
    assert closed_loop_command[oscar_num_steps_index + 1] == str(
        single_episode.DIRECT_OSCAR_NUM_STEPS
    )
    min_steps_index = closed_loop_command.index("--min-steps")
    assert closed_loop_command[min_steps_index + 1] == str(single_episode.DIRECT_EPISODE_MIN_STEPS)
    assert single_episode.DIRECT_EPISODE_MIN_STEPS == 1
    assert "--stop-on-unsafe-stance" not in closed_loop_command
    assert "--stop-on-no-progress" not in closed_loop_command
    no_progress_index = closed_loop_command.index("--no-progress-patience-steps")
    assert closed_loop_command[no_progress_index + 1] == str(
        single_episode.DIRECT_NO_PROGRESS_PATIENCE_STEPS
    )
    assert single_episode.DIRECT_NO_PROGRESS_PATIENCE_STEPS == 3
    reanchor_index = closed_loop_command.index("--clean-frame-reanchor-interval")
    assert closed_loop_command[reanchor_index + 1] == str(
        single_episode.DIRECT_CLEAN_FRAME_REANCHOR_INTERVAL
    )
    assert single_episode.DIRECT_CLEAN_FRAME_REANCHOR_INTERVAL == 1
    execution_frame_count_index = closed_loop_command.index("--groot-sonic-execution-frame-count")
    assert closed_loop_command[execution_frame_count_index + 1] == str(
        single_episode.DIRECT_GROOT_SONIC_EXECUTION_FRAME_COUNT
    )
    assert single_episode.DIRECT_GROOT_SONIC_EXECUTION_FRAME_COUNT == 40
    harness_backend_index = closed_loop_command.index("--harness-backend-kind")
    assert closed_loop_command[harness_backend_index + 1] == (
        single_episode.GENERATED_RGB_POLICY_OBSERVATION_BACKEND_KIND
    )
    for option in single_episode.DIRECT_RGB_OBSERVATION_REMOVED_REQUIREMENTS:
        assert option not in closed_loop_command
    task_prompt_index = closed_loop_command.index("--task-prompt")
    assert closed_loop_command[task_prompt_index + 1] == single_episode.TASK_PROMPT
    perception_prompts = [
        closed_loop_command[index + 1]
        for index, value in enumerate(closed_loop_command)
        if value == "--perception-target-prompt"
    ]
    assert perception_prompts == [
        "microwave",
        "microwave handle",
        "microwave door",
        "microwave cavity",
        "microwave fixture",
        "handle",
        "door",
    ]
    assert inputs["task_prompt"] == single_episode.TASK_PROMPT
    assert inputs["perception_target_prompts"] == perception_prompts
    assert inputs["plan"]["env"]["BLUEPRINT_OSCAR_WAM_SOURCE_URL"] == (
        single_episode.OFFICIAL_OSCAR_SOURCE_URL
    )
    assert inputs["plan"]["env"]["BLUEPRINT_OSCAR_WAM_SOURCE_REF"] == (
        single_episode.OFFICIAL_OSCAR_SOURCE_COMMIT
    )
    assert "blueprint_pipeline.oscar_runtime_source_provenance verify" in script
    assert "git -c safe.directory=/opt/oscar-public" not in script
    assert "--seal /opt/blueprint/oscar_source_provenance.json" in script
    assert "BLUEPRINT_FOUNDATION_OSCAR_SOURCE_URL" not in script
    assert "BLUEPRINT_FOUNDATION_OSCAR_SOURCE_REF" not in script
    assert "official_oscar_runtime_provenance_mismatch" in script
    assert single_episode.OSCAR_RUNTIME_PROVENANCE_ARTIFACT in script
    assert 'export BLUEPRINT_OSCAR_WAM_SOURCE_URL="$OSCAR_RUNTIME_SOURCE_URL"' in script
    assert 'export BLUEPRINT_OSCAR_WAM_SOURCE_REF="$OSCAR_RUNTIME_SOURCE_REF"' in script
    assert "oscar_runtime_source_provenance.py" in single_episode.RUNTIME_PACKAGE_OVERLAY_MODULES
    assert script.index("blueprint_pipeline.oscar_runtime_source_provenance verify") < (
        script.index("upload_phase inputs_ready")
    )
    assert "single_g1_kitchen_runtime_patch" not in script
    assert "GEAR_SONIC_SIM_PID" not in script
    assert "isaac_persistent_task_executor_service.py" in (
        single_episode.RUNTIME_PACKAGE_OVERLAY_MODULES
    )
    assert "isaac_persistent_task_completion_client.py" in (
        single_episode.RUNTIME_PACKAGE_OVERLAY_MODULES
    )
    assert "wam_derived_observation_harness.py" in (single_episode.RUNTIME_PACKAGE_OVERLAY_MODULES)
    assert "task_episode_baseline.py" in single_episode.RUNTIME_PACKAGE_OVERLAY_MODULES
    assert "gear_sonic_joint_order_contract.py" in (single_episode.RUNTIME_PACKAGE_OVERLAY_MODULES)
    assert "gear_sonic_process_supervisor.py" in (
        single_episode.RUNTIME_PACKAGE_OVERLAY_MODULES
    )
    assert inputs["plan"]["env"][single_episode.SNAPSHOT_ENV] == (
        single_episode.SNAPSHOT_DEFAULT_PATH
    )
    assert inputs["plan"]["env"][single_episode.BRIDGE_REQUIRED_ENV] == "true"
    assert (
        inputs["plan"]["env"][single_episode.CONTROLLER_FK_CAMERA_PROJECTION_CONTEXT_ENV]
        == single_episode.CONTROLLER_FK_CAMERA_PROJECTION_CONTEXT_PATH
    )
    isaac_command = inputs["plan"]["isaac_task_executor_command"]
    initial_frame_index = isaac_command.index("--initial-frame-output")
    projection_context_index = isaac_command.index("--camera-projection-context-output")
    assert isaac_command[initial_frame_index + 1] == single_episode.INITIAL_POLICY_FRAME_PATH
    assert isaac_command[projection_context_index + 1] == (
        single_episode.CONTROLLER_FK_CAMERA_PROJECTION_CONTEXT_PATH
    )
    assert "captured_from_live_persistent_isaac_session" in script
    assert "socket.create_connection(('127.0.0.1', 8765)" in script
    assert "isaac_task_state/initial_policy_observation.json" in script
    assert "context.get('attempt_id') == attempt.get('attempt_id')" in script
    assert "context.get('launch_nonce') == attempt.get('launch_nonce')" in script
    assert "if not context_name:\n                break" not in script
    assert 'workspace / "controller_fk_camera_projection_context.json"' in script
    assert "bundled_seed_frame_reused') is False" in script
    assert "controller_fk_camera_projection_context_sha256_mismatch" not in script
    assert "controller_fk_camera_projection_context" not in inputs
    assert "controller_fk_camera_projection_context_bytes" not in inputs
    assert "controller_fk_camera_projection_context_sha256" not in inputs
    assert "bundled_seed_frame_reused') is False" in script
    assert script.index("if identity_matches and artifact_matches and registration_ready") < (
        script.index("upload_phase isaac_task_executor_ready")
    )
    service_source = (
        Path(single_episode.__file__)
        .with_name("isaac_persistent_task_executor_service.py")
        .read_text(encoding="utf-8")
    )
    assert service_source.index("backend.capture_initial_policy_observation(") < (
        service_source.index("_write_json_atomic(args.initial_state_output, initial)")
    )
    assert "gear_sonic_isaac_dds_bridge.cpp" in script
    assert "libunitree_sdk2.a" in script
    assert "-lddscxx -lddsc -pthread" in script
    assert single_episode.BRIDGE_HEARTBEAT_PATH in script
    assert single_episode.BRIDGE_LOG_PATH in script
    assert (
        f"export GROOT_PID GEAR_SONIC_PID ISAAC_TASK_PID {single_episode.BRIDGE_PID_ENV}" in script
    )
    assert f'kill "${{{single_episode.BRIDGE_PID_ENV}}}"' in script
    assert "gear_sonic_isaac_dds_bridge_ready" in script
    assert inputs["plan"]["gear_sonic_controller_command"] == [
        single_episode.OSCAR_PYTHON,
        "-m",
        single_episode.GEAR_SONIC_PROCESS_SUPERVISOR_MODULE,
        "supervise",
        "--",
        "true",
    ]
    assert "gear_sonic_bridge_stable_before_controller_start" not in script
    assert "gear_sonic_bridge_never_stably_fresh_before_controller" not in script
    assert "heartbeat.get('source_fresh') is True" in script
    assert "heartbeat.get('holding_last_validated_snapshot') is False" in script
    assert "0 <= heartbeat_age_ns <= 500_000_000" in script
    assert "publish_count > first_publish_count" in script
    assert "'Init Done' in controller_log.read_text" in script
    controller_launch = shlex.join(inputs["plan"]["gear_sonic_controller_command"])
    assert script.index(
        f"{controller_launch} > /workspace/gear_sonic_controller.log 2>&1 &"
    ) < script.index("official_gear_sonic_controller_or_isaac_dds_bridge_not_ready")
    assert "GEAR_SONIC_PROCESS_PATTERN" not in script
    assert "stale_gear_sonic_controller_process_not_reaped" not in script
    assert script.index("gear_sonic_isaac_dds_bridge.cpp") < script.index(
        "groot_oscar_closed_loop_image_healthcheck.py --require-cuda"
    )
    runtime_overlay_payload = inputs["runtime_package_overlay_xz_base64"]
    runtime_overlay_archive = lzma.decompress(
        base64.b64decode(runtime_overlay_payload, validate=True)
    )
    assert (
        hashlib.sha256(runtime_overlay_archive).hexdigest()
        == inputs["runtime_package_overlay_sha256"]
    )
    runtime_overlay = json.loads(runtime_overlay_archive)
    assert sorted(runtime_overlay["modules"]) == sorted(
        single_episode.RUNTIME_PACKAGE_OVERLAY_MODULES
    )
    for filename in single_episode.RUNTIME_PACKAGE_OVERLAY_MODULES:
        source = Path(single_episode.__file__).with_name(filename).read_bytes()
        row = runtime_overlay["modules"][filename]
        assert base64.b64decode(row["source_base64"], validate=True) == source
        assert row["source_sha256"] == hashlib.sha256(source).hexdigest()
        assert inputs["runtime_package_overlay_source_sha256s"][filename] == (
            hashlib.sha256(source).hexdigest()
        )
    assert runtime_overlay_payload not in script
    assert single_episode.RUNTIME_PACKAGE_OVERLAY_PAYLOAD_ENV in script
    assert single_episode.RUNTIME_PACKAGE_OVERLAY_PAYLOAD_PATH in script
    assert 'payload_path.read_text(encoding="ascii")' in script
    assert single_episode.OVERLAY_PAYLOAD_TRANSPORT_ENV in script
    assert single_episode.RUNTIME_PACKAGE_OVERLAY_PAYLOAD_ENV not in inputs["plan"]["env"]
    assert (
        inputs["plan"]["env"][single_episode.RUNTIME_PACKAGE_OVERLAY_SHA256_ENV]
        == inputs["runtime_package_overlay_sha256"]
    )
    assert "runtime_package_overlay.json" in script
    assert "materialized_hash_verified_and_imported" in script
    assert "importlib.import_module(module_name)" in script
    sitecustomize_match = re.search(r"sitecustomize = base64\.b64decode\('([^']+)'", script)
    assert sitecustomize_match is not None
    sitecustomize = base64.b64decode(sitecustomize_match.group(1), validate=True).decode("utf-8")
    assert "blueprint_pipeline.__path__.insert(0, OVERLAY_PACKAGE)" in sitecustomize
    assert script.index("materialized_hash_verified_and_imported") < script.index(
        "blueprint_pipeline.oscar_runtime_source_provenance verify"
    )
    backend_source = (
        Path(single_episode.__file__).with_name("isaac_runtime_task_backend.py").read_bytes()
    )
    backend_sha = hashlib.sha256(backend_source).hexdigest()
    assert inputs["isaac_runtime_backend_overlay_sha256"] == backend_sha
    overlay_payload = inputs["isaac_runtime_backend_overlay_gzip_base64"]
    assert gzip.decompress(base64.b64decode(overlay_payload, validate=True)) == backend_source
    assert overlay_payload not in script
    assert single_episode.ISAAC_RUNTIME_OVERLAY_PAYLOAD_ENV in script
    assert single_episode.ISAAC_RUNTIME_OVERLAY_PAYLOAD_PATH in script
    assert "payload_path.read_text(encoding='ascii')" in script
    assert single_episode.ISAAC_RUNTIME_OVERLAY_PAYLOAD_ENV not in inputs["plan"]["env"]
    assert inputs["plan"]["env"]["BLUEPRINT_ISAAC_RUNTIME_BACKEND_OVERLAY_SHA256"] == backend_sha
    assert inputs["plan"]["isaac_task_executor_command"] == [
        "env",
        f"PYTHONPATH={expected_base_pythonpath}",
        single_episode.ISAAC_PYTHON,
        single_episode.ISAAC_RUNTIME_OVERLAY_WRAPPER,
        "--stage",
        "/workspace/kitchen/KitchenRoom.usd",
        "--initial-frame-output",
        single_episode.INITIAL_POLICY_FRAME_PATH,
        "--camera-projection-context-output",
        single_episode.CONTROLLER_FK_CAMERA_PROJECTION_CONTEXT_PATH,
    ]
    assert "isaac_runtime_backend_overlay.v1" in script
    assert "scene_physics_modified': False" in script
    assert backend_sha in script
    wrapper_match = re.search(r"wrapper = base64\.b64decode\('([^']+)'", script)
    assert wrapper_match is not None
    wrapper = base64.b64decode(wrapper_match.group(1), validate=True).decode("utf-8")
    assert "isaac_runtime_backend_overlay_sha256_mismatch" in wrapper
    assert "isaac_runtime_backend_overlay_plan_digest_mismatch" in wrapper
    assert "isaac_task_executor_service_overlay_path_mismatch" in wrapper
    assert single_episode.RUNTIME_PACKAGE_OVERLAY_DIR in wrapper
    assert "spec_from_file_location(MODULE_NAME, SOURCE_PATH)" in wrapper
    assert "sys.modules[MODULE_NAME] = module" in wrapper
    criterion = inputs["task_contract"]["registered_criteria"][0]
    assert criterion["articulation_prim_path"] == ("/root/Microwave017/Microwave017_Door")
    assert criterion["articulation_prim_path_resolution"] == {
        "mode": "exact_selected_scenario_affordance",
        "selected_scenario_sha256": hashlib.sha256(scenario.read_bytes()).hexdigest(),
        "selected_scenario_field": ("accepted_stance_contract.resolved_affordance.prim_path"),
    }
    assert "direct_task_contract_resolution.json" in script
    assert "derived_from_sha256" in script
    assert (
        "/opt/oscar-venv/bin/python -m blueprint_pipeline.g1_kitchen_bundle_compatibility"
    ) in script
    assert ("/opt/oscar-venv/bin/python -m blueprint_pipeline.runtime_ephemeral_trust") in script
    assert ("/opt/oscar-venv/bin/python -m blueprint_pipeline.g1_kitchen_startup_proof") in script
    assert (
        "HF_HOME=/opt/blueprint/hf_home "
        "HF_HUB_CACHE=/opt/blueprint/hf_home/hub "
        "HUGGINGFACE_HUB_CACHE=/opt/blueprint/hf_home/hub "
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
        "COSMOS_BACKBONE_REPO=nvidia/Cosmos-Reason2-2B "
        "COSMOS_BACKBONE_REVISION=9ce19a195e423419c349abfc86fd07178b230561 "
        "/opt/oscar-venv/bin/python "
        "/opt/blueprint/groot_oscar_closed_loop_image_healthcheck.py --require-cuda"
    ) in script
    assert "groot_sonic_checkpoint_preflight.v1" in script
    assert "pinned_get_backbone_cls_resolved" in script
    assert "/workspace/.blueprint-model-aliases/cosmos" in script
    assert 'selector_root / "nvidia/Cosmos-Reason2-2B"' in script
    assert 'alias = selector_anchor / "../.."' in script
    assert 'processor_kwargs["model_name"] = str(alias)' in script
    assert "groot_policy_processor_offline_constructible" in script
    assert script.index("groot_checkpoint_preflight_passed") < script.index(
        "groot_oscar_closed_loop_image_healthcheck.py --require-cuda"
    )
    assert "python3 /opt/blueprint/groot_oscar_closed_loop_image_healthcheck.py" not in script
    assert not re.search(
        r"(?<!\S)python(?=(?: -m blueprint_pipeline\.| /workspace/| - <<'PY'))",
        script,
    )


def test_collected_final_review_requires_hash_order_and_decode_evidence(
    tmp_path: Path,
) -> None:
    episode_dir = tmp_path / "closed_loop_output" / "closed_loop_out" / "episode_001"
    episode_dir.mkdir(parents=True)
    video = episode_dir / "final_review.mp4"
    video.write_bytes(b"validated-video")
    video_sha256 = hashlib.sha256(video.read_bytes()).hexdigest()
    wam_video = episode_dir / "wam_prediction_review.mp4"
    wam_video.write_bytes(b"wam-prediction-video")
    wam_validation = {
        "schema_version": "groot_oscar_wam_prediction_review_validation.v1",
        "status": "passed",
        "blockers": [],
        "path": "/workspace/closed_loop_out/episode_001/wam_prediction_review.mp4",
        "sha256": hashlib.sha256(wam_video.read_bytes()).hexdigest(),
        "trace_step_count": 3,
        "ordered_clip_count": 3,
        "ordered_step_indices": [1, 2, 3],
        "episode_order_verified": True,
        "review_source": "oscar_wam_predicted_rollout_clips",
        "video_frame_count_mode": "dynamic_from_executed_controller_duration",
        "prediction_review_timeline_mode": "executed_control_prefix_per_decision",
        "executed_prefix_duration_seconds_by_step": [1.0, 1.0, 1.0],
        "expected_executed_timeline_duration_seconds": 3.0,
        "duration_seconds": 3.0,
        "full_prediction_horizons_preserved_in_source_clips": True,
        "overlapping_unexecuted_prediction_tails_excluded": True,
    }
    (episode_dir / "wam_prediction_review_validation.json").write_text(
        json.dumps(wam_validation), encoding="utf-8"
    )
    role_videos = {}
    for role, filename in {
        "overview": "isaac_overview_review.mp4",
        "robot_pov": "isaac_robot_pov_review.mp4",
    }.items():
        role_path = episode_dir / filename
        role_path.write_bytes(f"{role}-video".encode())
        role_videos[role] = {
            "status": "passed",
            "blockers": [],
            "path": f"/workspace/closed_loop_out/episode_001/{filename}",
            "sha256": hashlib.sha256(role_path.read_bytes()).hexdigest(),
            "width": 640,
            "height": 480,
            "frame_count": 3,
        }
    (episode_dir / "final_review_validation.json").write_text(
        json.dumps(
            {
                "schema_version": "groot_oscar_episode_review_validation.v1",
                "status": "passed",
                "blockers": [],
                "sha256": video_sha256,
                "episode_order_verified": True,
                "trace_step_count": 3,
                "ordered_clip_count": 3,
                "ordered_step_indices": [1, 2, 3],
                "width": 640,
                "height": 480,
                "frame_count": 3,
                "ordered_review_frame_count": 3,
                "duration_seconds": 3.0,
                "review_source": "persistent_same_session_isaac_execution_frames",
                "execution_truth": True,
                "same_session_isaac_frames": True,
                "concat_mode": "primary_same_session_isaac_robot_pov_only",
                "primary_camera_role": "robot_pov",
                "overview_excluded_from_primary_review": True,
                "required_camera_roles": ["overview", "robot_pov"],
                "isaac_frame_evidence": {
                    "status": "passed",
                    "blockers": [],
                    "bound_steps": [{"step": value} for value in range(1, 4)],
                    "ordered_review_frame_count": 3,
                    "ordered_review_frame_indices": [0, 1, 2],
                    "ordered_review_control_frame_indices": [0, 5, 10],
                },
                "isaac_role_videos": role_videos,
                "wam_prediction_review": wam_validation,
            }
        ),
        encoding="utf-8",
    )

    evidence = single_episode._validate_collected_final_review(tmp_path)

    assert evidence["status"] == "passed"
    assert evidence["blockers"] == []
    assert evidence["video_sha256"] == video_sha256


def test_collected_final_review_accepts_one_outer_step_with_nine_sampled_frames(
    tmp_path: Path,
) -> None:
    episode_dir = tmp_path / "closed_loop_output" / "closed_loop_out" / "episode_001"
    episode_dir.mkdir(parents=True)
    final_video = episode_dir / "final_review.mp4"
    final_video.write_bytes(b"nine-sampled-final")
    wam_video = episode_dir / "wam_prediction_review.mp4"
    wam_video.write_bytes(b"one-wam-clip")
    wam_validation = {
        "schema_version": "groot_oscar_wam_prediction_review_validation.v1",
        "status": "passed",
        "blockers": [],
        "path": "/workspace/closed_loop_out/episode_001/wam_prediction_review.mp4",
        "sha256": hashlib.sha256(wam_video.read_bytes()).hexdigest(),
        "trace_step_count": 1,
        "ordered_clip_count": 1,
        "ordered_step_indices": [1],
        "episode_order_verified": True,
        "review_source": "oscar_wam_predicted_rollout_clips",
        "video_frame_count_mode": "dynamic_from_executed_controller_duration",
        "prediction_review_timeline_mode": "executed_control_prefix_per_decision",
        "executed_prefix_duration_seconds_by_step": [0.1],
        "expected_executed_timeline_duration_seconds": 0.1,
        "duration_seconds": 0.1,
        "full_prediction_horizons_preserved_in_source_clips": True,
        "overlapping_unexecuted_prediction_tails_excluded": True,
    }
    (episode_dir / "wam_prediction_review_validation.json").write_text(
        json.dumps(wam_validation), encoding="utf-8"
    )
    role_videos = {}
    for role, filename in {
        "overview": "isaac_overview_review.mp4",
        "robot_pov": "isaac_robot_pov_review.mp4",
    }.items():
        path = episode_dir / filename
        path.write_bytes(role.encode())
        role_videos[role] = {
            "status": "passed",
            "blockers": [],
            "path": f"/workspace/closed_loop_out/episode_001/{filename}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "width": 640,
            "height": 480,
            "frame_count": 9,
        }
    validation = {
        "schema_version": "groot_oscar_episode_review_validation.v1",
        "status": "passed",
        "blockers": [],
        "sha256": hashlib.sha256(final_video.read_bytes()).hexdigest(),
        "episode_order_verified": True,
        "trace_step_count": 1,
        "ordered_clip_count": 1,
        "ordered_step_indices": [1],
        "ordered_review_frame_count": 9,
        "width": 640,
        "height": 480,
        "frame_count": 9,
        "duration_seconds": 0.9,
        "review_source": "persistent_same_session_isaac_execution_frames",
        "execution_truth": True,
        "same_session_isaac_frames": True,
        "concat_mode": "primary_same_session_isaac_robot_pov_only",
        "primary_camera_role": "robot_pov",
        "overview_excluded_from_primary_review": True,
        "required_camera_roles": ["overview", "robot_pov"],
        "isaac_frame_evidence": {
            "status": "passed",
            "blockers": [],
            "bound_steps": [{"step": 1}],
            "ordered_review_frame_count": 9,
            "ordered_review_frame_indices": list(range(9)),
            "ordered_review_control_frame_indices": [
                0,
                5,
                10,
                15,
                20,
                25,
                30,
                35,
                40,
            ],
        },
        "isaac_role_videos": role_videos,
        "wam_prediction_review": wam_validation,
    }
    (episode_dir / "final_review_validation.json").write_text(
        json.dumps(validation), encoding="utf-8"
    )

    evidence = single_episode._validate_collected_final_review(tmp_path)

    assert evidence["status"] == "passed"
    assert evidence["blockers"] == []


def test_collected_final_review_fails_closed_on_hash_and_clip_count_mismatch(
    tmp_path: Path,
) -> None:
    episode_dir = tmp_path / "closed_loop_output" / "closed_loop_out" / "episode_001"
    episode_dir.mkdir(parents=True)
    (episode_dir / "final_review.mp4").write_bytes(b"actual-video")
    (episode_dir / "final_review_validation.json").write_text(
        json.dumps(
            {
                "schema_version": "groot_oscar_episode_review_validation.v1",
                "status": "passed",
                "blockers": [],
                "sha256": "0" * 64,
                "episode_order_verified": True,
                "trace_step_count": 3,
                "ordered_clip_count": 2,
                "ordered_step_indices": [1, 2],
                "width": 640,
                "height": 480,
                "frame_count": 48,
                "duration_seconds": 2.0,
            }
        ),
        encoding="utf-8",
    )

    evidence = single_episode._validate_collected_final_review(tmp_path)

    assert evidence["status"] == "blocked"
    assert "single_episode_final_review_clip_count_invalid" in evidence["blockers"]
    assert "single_episode_final_review_step_order_invalid" in evidence["blockers"]
    assert "single_episode_final_review_sha256_mismatch" in evidence["blockers"]


def test_collected_final_review_rejects_predicted_media_as_execution_truth(
    tmp_path: Path,
) -> None:
    episode_dir = tmp_path / "closed_loop_output" / "closed_loop_out" / "episode_001"
    episode_dir.mkdir(parents=True)
    video = episode_dir / "final_review.mp4"
    video.write_bytes(b"predicted-video")
    (episode_dir / "final_review_validation.json").write_text(
        json.dumps(
            {
                "schema_version": "groot_oscar_episode_review_validation.v1",
                "status": "passed",
                "blockers": [],
                "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                "episode_order_verified": True,
                "trace_step_count": 1,
                "ordered_clip_count": 1,
                "ordered_step_indices": [1],
                "width": 640,
                "height": 480,
                "frame_count": 1,
                "duration_seconds": 1.0,
                "review_source": "oscar_wam_predicted_rollout_clips",
                "execution_truth": False,
                "same_session_isaac_frames": False,
                "required_camera_roles": [],
            }
        ),
        encoding="utf-8",
    )

    evidence = single_episode._validate_collected_final_review(tmp_path)

    assert evidence["status"] == "blocked"
    assert "single_episode_final_review_source_invalid" in evidence["blockers"]
    assert "single_episode_final_review_not_execution_truth" in evidence["blockers"]
    assert "single_episode_final_review_not_same_session_isaac" in evidence["blockers"]


@pytest.mark.parametrize(
    ("provider_name", "expected_source"),
    [
        ("vast", "VAST_CONTAINERLABEL"),
        ("runpod", "RUNPOD_POD_ID"),
    ],
)
def test_provider_allocation_identity_is_bound_before_runtime_trust(
    provider_name: str,
    expected_source: str,
) -> None:
    script = "upload_phase inputs_ready\npython -m blueprint_pipeline.runtime_ephemeral_trust\n"

    resolved = single_episode._bind_provider_allocation_identity(script, provider_name)

    assert expected_source in resolved
    assert "provider_allocation_identity_unavailable" in resolved
    assert "python3 /workspace/write_result.py" in resolved
    assert resolved.index("provider_allocation_bound") < resolved.index(
        "python -m blueprint_pipeline.runtime_ephemeral_trust"
    )


def test_direct_episode_owner_is_opened_before_launch_bound_and_closed_by_teardown(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempt"
    root.mkdir()
    registry = tmp_path / "pending"
    prefix = "blueprint-groot-oscar-canary-single-episode-direct-"

    ownership = single_episode._open_direct_episode_owner(
        root=root,
        provider_name="vast",
        run_id="single-g1-kitchen-direct",
        pod_name=prefix + "pod",
        pod_name_prefix=prefix,
        watchdog_deadline_epoch=9_999_999_999.0,
        watchdog_pid=1234,
        registry_dir=registry,
    )

    pending_path = Path(ownership["pending"]["path"])
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    owner = json.loads((root / single_episode.DIRECT_OWNER_NAME).read_text(encoding="utf-8"))
    assert pending["status"] == "open"
    assert pending["instance_id"] is None
    assert pending["lane"] == "single_g1_kitchen_episode"
    assert owner["status"] == "armed_before_provider_launch"
    assert owner["campaign_ownership_required"] is False
    assert owner["owner_kind"] == ("direct_single_episode_controller_and_attempt_local_watchdog")
    assert owner["provider_mutations_performed"] == 0
    assert owner["watchdog_pid"] == 1234

    ownership = single_episode._bind_direct_episode_owner(
        root=root,
        ownership=ownership,
        instance_id="45123642",
        watchdog_pid=1234,
    )
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    owner = json.loads((root / single_episode.DIRECT_OWNER_NAME).read_text(encoding="utf-8"))
    assert pending["status"] == "open"
    assert pending["instance_id"] == "45123642"
    assert owner["status"] == "allocation_bound_to_direct_owner"
    assert owner["instance_id"] == "45123642"
    assert owner["watchdog_pid"] == 1234

    settled = single_episode._settle_direct_episode_owner(
        root=root,
        ownership=ownership,
        teardown={
            "status": "provider_terminal",
            "provider_absence_confirmed": True,
        },
        launch={"status": "launched", "instance_id": "45123642"},
    )
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    assert pending["status"] == "closed"
    assert pending["teardown_proof"]["status"] == "PASS"
    assert settled["owner"]["status"] == ("provider_terminal_and_control_plane_closed")
    assert settled["owner"]["provider_absence_confirmed"] is True


def test_launch_interruptions_continue_to_canonical_blocked_result() -> None:
    source = inspect.getsource(single_episode.run_single_episode)
    tree = ast.parse(source)
    handlers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
        and isinstance(node.type, ast.Name)
        and node.type.id == "BaseException"
    ]

    assert len(handlers) == 1
    assert not any(isinstance(node, ast.Raise) for node in ast.walk(handlers[0]))
    assert "single_episode_launch_or_watch_interrupted" in source
    assert "write_json(result_path, result)" in source


def test_direct_episode_owner_keeps_ambiguous_no_id_obligation_open(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempt"
    root.mkdir()
    ownership = single_episode._open_direct_episode_owner(
        root=root,
        provider_name="vast",
        run_id="single-g1-kitchen-ambiguous",
        pod_name="blueprint-groot-oscar-canary-single-episode-direct-pod",
        pod_name_prefix="blueprint-groot-oscar-canary-single-episode-direct-",
        watchdog_deadline_epoch=9_999_999_999.0,
        watchdog_pid=1234,
        registry_dir=tmp_path / "pending",
    )

    settled = single_episode._settle_direct_episode_owner(
        root=root,
        ownership=ownership,
        teardown={
            "status": "teardown_unverified",
            "provider_absence_confirmed": False,
        },
        launch={
            "status": "launch_or_watch_interrupted",
            "allocation_outcome_ambiguous": True,
        },
    )

    pending_path = Path(ownership["pending"]["path"])
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    assert pending["status"] == "open"
    assert pending["allocation_outcome_ambiguous"] is True
    assert settled["owner"]["status"] == (
        "allocation_outcome_ambiguous_pending_record_retained_open"
    )
