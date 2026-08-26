from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_live_profile import file_digest
from blueprint_pipeline.task_evaluation_scene_configuration_bundle import (
    BUNDLE_SCHEMA_VERSION,
    build_scene_configuration_provider_bundle,
    load_scene_configuration_provider_bundle_receipt,
)
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
    assert portable["render_inputs_result"]["derived_gaussian_cutout"][
        "retained_scene_without_source_object"
    ]["path"].startswith("input/render/gaussians/")
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


def test_vast_preflight_and_onstart_accept_only_the_sealed_scene_bundle(
    tmp_path: Path,
) -> None:
    receipt = _build(tmp_path, "bundle")
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

    assert preflight["status"] == "passed"
    assert preflight["blockers"] == []
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
    subprocess.run(["bash", "-n", "-c", script], check=True)


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
            {"total_cost_usd": 40.0},
            record(Path(path).resolve()),
        ),
    )
    monkeypatch.setattr(
        live_profile_module,
        "validate_project_spend_reconciliation",
        lambda path, **_kwargs: (
            {"total_cost_usd": 40.0},
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
    assert authority["maximum_provider_allocations"] == 1
    assert authority["aggregate_goal_spend_before_attempt_usd"] == 40.0
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
