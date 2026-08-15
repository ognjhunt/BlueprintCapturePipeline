from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from blueprint_pipeline.adp_retained_scene_render_packet import (
    RetainedSceneRenderPacketError,
    build_retained_scene_gpu_render_bundle,
    build_retained_scene_gpu_render_request,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.adp_retained_scene_render_vast import (
    _authority_environment,
    materialize_retained_scene_render_output_relocation,
    run_retained_scene_render_vast,
    validate_retained_scene_render_bundle,
    validate_retained_scene_render_paid_attempt_authority,
)
from blueprint_pipeline.gaussian_splat_decode import (
    SplatData,
    write_standard_3dgs_ply,
    write_standard_3dgs_ply_subset_exact,
)
from blueprint_pipeline.vast_provider_adapter import (
    _blueprint_bundle_preflight,
    _probe_shell_script,
)
from blueprint_pipeline.wam_provider_output import inspect_provider_runtime_output_zip


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _absolute_record(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _relative_record(root: Path, path: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def test_retained_scene_render_authority_environment_restores_retry_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_API_CALLS", "caller-api")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", "caller-launch")
    monkeypatch.setenv("BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS", "caller-retry")

    with _authority_environment():
        assert os.environ["BLUEPRINT_ALLOW_VAST_API_CALLS"] == "1"
        assert os.environ["BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH"] == "1"
        assert os.environ["BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS"] == "0"

    assert os.environ["BLUEPRINT_ALLOW_VAST_API_CALLS"] == "caller-api"
    assert os.environ["BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH"] == "caller-launch"
    assert os.environ["BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS"] == "caller-retry"


@pytest.mark.parametrize(
    ("layer", "source_role"),
    [
        ("shared_deleted_source_layer", "shared_deleted_source_union"),
        ("task_deleted_source_layer", "task_deleted_source_layer"),
    ],
)
def test_output_relocation_rebinds_container_manifest_paths_to_verified_local_files(
    tmp_path: Path, layer: str, source_role: str
) -> None:
    output = tmp_path / "immutable-extraction"
    manifest_path = (
        output
        / "renders"
        / "task_a"
        / f"task_a_{layer}_black"
        / "sealed_camera_render_manifest.v1.json"
    )
    manifest_path.parent.mkdir(parents=True)
    manifest: dict[str, object] = {
        "schema_version": "sealed_camera_render_manifest.v1",
        "status": "rendered_exact_cameras",
        "source_layer_role": source_role,
        "render_settings": {"background_rgb": "#000000"},
        "sealed_camera_render_manifest_digest": "",
    }
    manifest["sealed_camera_render_manifest_digest"] = canonical_digest(
        manifest, digest_field="sealed_camera_render_manifest_digest"
    )
    _write_json(manifest_path, manifest)
    result: dict[str, object] = {
        "schema_version": "adp009d_retained_scene_gpu_render_result.v1",
        "render_manifests": [
            {
                "task_id": "task_a",
                "layer": layer,
                "background_rgb": "#000000",
                "manifest_path": "/workspace/provider/output/manifest.json",
                "manifest_digest": manifest["sealed_camera_render_manifest_digest"],
            }
        ],
    }
    result_path = output / "adp009d_retained_scene_gpu_render_result.v1.json"
    _write_json(result_path, result)

    receipt = materialize_retained_scene_render_output_relocation(
        result_path=result_path, destination=output
    )

    assert receipt["provider_result"]["sha256"] == _sha256(result_path)
    local = receipt["render_manifests"][0]["local_manifest"]
    assert local["path"] == str(manifest_path)
    assert local["manifest_digest"] == manifest["sealed_camera_render_manifest_digest"]


def test_retained_scene_render_uses_a_watchdog_canary_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import blueprint_pipeline.adp_retained_scene_render_vast as retained_vast

    captured: dict[str, object] = {}

    def fake_arm(**kwargs: object) -> tuple[dict[str, object], SimpleNamespace]:
        captured.update(kwargs)
        return (
            {"status": "armed", "blockers": []},
            SimpleNamespace(
                started_instance_id_path=tmp_path / "started_vast_instance_id.txt",
                pod_name_prefix=kwargs["pod_name_prefix"],
            ),
        )

    monkeypatch.setattr(retained_vast, "arm_independent_vast_watchdog", fake_arm)
    receipt = {
        "bundle_path": str(tmp_path / "missing.zip"),
        "blueprint_commit": "fixture-commit",
        "status": "ready",
    }
    monkeypatch.setattr(
        retained_vast,
        "validate_retained_scene_render_bundle",
        lambda _bundle: receipt,
    )
    monkeypatch.setattr(
        retained_vast,
        "validate_retained_scene_render_paid_attempt_authority",
        lambda authority, **_kwargs: authority,
    )
    monkeypatch.setattr(
        retained_vast,
        "stage_wam_provider_bundle_object_store",
        lambda **_kwargs: {"status": "completed", "blockers": []},
    )
    monkeypatch.setattr(
        retained_vast,
        "consume_retained_scene_render_paid_attempt_authority_once",
        lambda *_args, **_kwargs: {"status": "blocked", "blockers": ["synthetic"]},
    )
    monkeypatch.setattr(
        retained_vast,
        "cleanup_staged_wam_provider_objects",
        lambda *_args, **_kwargs: {"all_objects_absent": True},
    )
    monkeypatch.setattr(
        retained_vast,
        "close_independent_vast_watchdog",
        lambda **_kwargs: {"status": "cancelled_no_allocation"},
    )
    monkeypatch.setattr(
        retained_vast,
        "require_paid_resource_admission_grant",
        lambda *_args, **_kwargs: None,
    )

    result = run_retained_scene_render_vast(
        job_dir=tmp_path / "job",
        paid_resource_admission_grant=object(),
        execute=True,
        prepared_bundle=receipt,
        paid_attempt_authority={"authorization_digest": "sha256:" + "a" * 64},
    )

    assert result["status"] == "blocked"
    assert captured["pod_name_prefix"] == "blueprint-groot-oscar-canary-adp-retained-render-"


def test_retained_scene_render_runner_retains_renderer_failure_diagnostic(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node unavailable")
    runtime = tmp_path / "runtime"
    output = tmp_path / "output"
    renderer = runtime / "renderer"
    renderer.mkdir(parents=True)
    source = runtime / "input/deleted.ply"
    retained = runtime / "input/retained.ply"
    candidate = runtime / "input/candidate.json"
    authority = runtime / "execution_authority.json"
    camera = runtime / "input/cameras.json"
    freeze = runtime / "input/freeze.json"
    for path in (source, retained):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nend_header\n")
    candidate.write_text("{}\n", encoding="utf-8")
    authority.write_text("{}\n", encoding="utf-8")
    freeze.write_text("{}\n", encoding="utf-8")
    camera.write_text(
        canonical_json(
            [
                {
                    "camera_id": "camera",
                    "T_world_camera_provider_frame": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "intrinsics": {
                        "fx": 1.0,
                        "fy": 1.0,
                        "cx": 1.0,
                        "cy": 1.0,
                        "width": 2,
                        "height": 2,
                    },
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    renderer.joinpath("render_splat.mjs").write_text(
        "process.stdout.write('renderer stdout'); process.stderr.write('renderer stderr'); process.exit(7);\n",
        encoding="utf-8",
    )
    request = {
        "shared_deleted_source_layer": {**_relative_record(runtime, source), "gaussian_count": 1},
        "shared_retained_scene": {**_relative_record(runtime, retained), "gaussian_count": 1},
        "shared_retained_gaussian_count": 1,
        "candidate_set": _relative_record(runtime, candidate),
        "execution_authority": _relative_record(runtime, authority),
        "candidate_set_digest": _digest("a"),
        "request_digest": _digest("b"),
        "renderer_identity": {},
        "lanes": [
            {
                "task_id": "task",
                "camera_contract": _relative_record(runtime, camera),
                "task_freeze": _relative_record(runtime, freeze),
                "task_deleted_source_layer": {
                    **_relative_record(runtime, source),
                    "gaussian_count": 1,
                },
                "task_retained_scene": {
                    **_relative_record(runtime, retained),
                    "gaussian_count": 1,
                },
                "dimensions": {"width": 2, "height": 2},
                "render_variants": [
                    {"layer": "task_deleted_source_layer", "background_rgb": "#000000"}
                ],
            }
        ],
    }
    runtime.joinpath("render_request.json").write_text(
        canonical_json(request) + "\n", encoding="utf-8"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    nvidia_smi = bin_dir / "nvidia-smi"
    nvidia_smi.write_text("#!/usr/bin/env bash\necho 'Fixture GPU, 1.0'\n", encoding="utf-8")
    nvidia_smi.chmod(0o700)
    runner = Path(__file__).resolve().parents[1] / "scripts/adp_retained_scene_render_provider_runner.mjs"
    environment = os.environ | {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    completed = subprocess.run(
        [node, str(runner), "--runtime", str(runtime), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    result = json.loads((output / "adp009d_retained_scene_gpu_render_result.v1.json").read_text())
    assert result["blockers"] == ["retained_scene_render_runtime_renderer_failed"]
    assert result["renderer_diagnostic"] == {
        "command": "render_splat.mjs",
        "error_code": None,
        "error_name": None,
        "exit_status": 7,
        "signal": None,
        "stderr_tail": "renderer stderr",
        "stdout_tail": "renderer stdout",
    }


def test_retained_scene_render_runner_seals_rendered_frames_with_relative_paths(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node unavailable")
    runtime = tmp_path / "runtime"
    output = tmp_path / "output"
    renderer = runtime / "renderer"
    renderer.mkdir(parents=True)
    source = runtime / "input/deleted.ply"
    retained = runtime / "input/retained.ply"
    candidate = runtime / "input/candidate.json"
    authority = runtime / "execution_authority.json"
    camera = runtime / "input/cameras.json"
    freeze = runtime / "input/freeze.json"
    for path in (source, retained):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nend_header\n")
    candidate.write_text("{}\n", encoding="utf-8")
    authority.write_text("{}\n", encoding="utf-8")
    freeze.write_text("{}\n", encoding="utf-8")
    camera.write_text(
        canonical_json(
            [
                {
                    "camera_id": "camera",
                    "T_world_camera_provider_frame": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "intrinsics": {
                        "fx": 1.0,
                        "fy": 1.0,
                        "cx": 1.0,
                        "cy": 1.0,
                        "width": 2,
                        "height": 2,
                    },
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    renderer.joinpath("render_splat.mjs").write_text(
        "import fs from 'node:fs'; import path from 'node:path'; "
        "const out = process.argv[process.argv.indexOf('--out') + 1]; "
        "fs.mkdirSync(out, { recursive: true }); "
        "fs.writeFileSync(path.join(out, 'camera.png'), 'png'); "
        "process.stdout.write(JSON.stringify({status: 'completed', "
        "graphics_diagnostics: {webgl_available: true, renderer: 'NVIDIA GPU'}}));\n",
        encoding="utf-8",
    )
    request = {
        "shared_deleted_source_layer": {**_relative_record(runtime, source), "gaussian_count": 1},
        "shared_retained_scene": {**_relative_record(runtime, retained), "gaussian_count": 1},
        "shared_retained_gaussian_count": 1,
        "candidate_set": _relative_record(runtime, candidate),
        "execution_authority": _relative_record(runtime, authority),
        "candidate_set_digest": _digest("a"),
        "request_digest": _digest("b"),
        "renderer_identity": {},
        "lanes": [
            {
                "task_id": "task",
                "camera_contract": _relative_record(runtime, camera),
                "task_freeze": _relative_record(runtime, freeze),
                "task_deleted_source_layer": {
                    **_relative_record(runtime, source),
                    "gaussian_count": 1,
                },
                "task_retained_scene": {
                    **_relative_record(runtime, retained),
                    "gaussian_count": 1,
                },
                "dimensions": {"width": 2, "height": 2},
                "render_variants": [
                    {"layer": "task_deleted_source_layer", "background_rgb": "#000000"}
                ],
            }
        ],
    }
    runtime.joinpath("render_request.json").write_text(
        canonical_json(request) + "\n", encoding="utf-8"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    nvidia_smi = bin_dir / "nvidia-smi"
    nvidia_smi.write_text("#!/usr/bin/env bash\necho 'Fixture GPU, 1.0'\n", encoding="utf-8")
    nvidia_smi.chmod(0o700)
    runner = Path(__file__).resolve().parents[1] / "scripts/adp_retained_scene_render_provider_runner.mjs"
    environment = os.environ | {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    completed = subprocess.run(
        [node, str(runner), "--runtime", str(runtime), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0
    result = json.loads((output / "adp009d_retained_scene_gpu_render_result.v1.json").read_text())
    assert result["status"] == "completed"
    manifest_path = Path(result["render_manifests"][0]["manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source_layer_role"] == "task_deleted_source_layer"
    assert manifest["splat_digest"] == _sha256(source)
    assert manifest["renders"] == [
        {
            "camera_id": "camera",
            "relative_path": "frames/camera.png",
            "size_bytes": 3,
            "digest": _sha256(manifest_path.parent / "frames/camera.png"),
            "width": 2,
            "height": 2,
        }
    ]


def test_retained_scene_render_runtime_result_is_recognized_by_provider_inspection(
    tmp_path: Path,
) -> None:
    output = tmp_path / "provider-output.zip"
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(
            "adp009d_retained_scene_gpu_render_result.v1.json",
            canonical_json(
                {
                    "status": "blocked",
                    "blockers": ["retained_scene_render_runtime_renderer_failed"],
                    "renderer_diagnostic": {"exit_status": 7},
                }
            ),
        )

    inspected = inspect_provider_runtime_output_zip(output)

    assert inspected["runtime_result_present"] is True
    assert inspected["runtime_result_status"] == "blocked"
    assert inspected["runtime_result_blockers"] == [
        "retained_scene_render_runtime_renderer_failed"
    ]


def test_retained_scene_render_reissue_rejects_estimate_only_prior_spend(
    tmp_path: Path,
) -> None:
    result: dict[str, object] = {
        "schema_version": "adp009d_retained_scene_gpu_render_vast_run.v1",
        "status": "blocked",
        "estimated_cost_usd": 0.008791,
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    prior = tmp_path / "prior_terminal_attempt.json"
    _write_json(prior, result)
    bundle = {
        "execution_authority": {"authority_digest": _digest("a")},
        "bundle_sha256": _digest("b"),
        "blueprint_commit": "fixture-commit",
        "hard_total_spend_cap_usd": 12.0,
    }
    authority: dict[str, object] = {
        "schema_version": "adp009d_retained_scene_gpu_render_paid_attempt_authority.v1",
        "authority_kind": "explicit_user_direction_in_current_goal",
        "purpose": "exact_retained_scene_gpu_render",
        "provider": "vast",
        "paid_compute_authorized": True,
        "parent_execution_authority_digest": bundle["execution_authority"]["authority_digest"],
        "bundle_sha256": bundle["bundle_sha256"],
        "blueprint_commit": bundle["blueprint_commit"],
        "maximum_paid_attempts": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "hard_attempt_spend_cap_usd": 12.0,
        "maximum_single_resource_ttl_seconds": 10_800,
        "maximum_hourly_rate_usd": 2.0,
        "external_active_instance_allowlist": [47373597],
        "manual_reissue_after_prior_terminal_attempt": True,
        "prior_terminal_attempts": [
            {
                "result_path": str(prior),
                "result_sha256": _sha256(prior),
                "receipt_digest": result["receipt_digest"],
                "estimated_cost_usd": 0.008791,
            }
        ],
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )

    with pytest.raises(
        ValueError, match="prior_terminal_attempt_reconciliation_invalid"
    ):
        validate_retained_scene_render_paid_attempt_authority(
            authority,
            prepared_bundle=bundle,
            max_hourly_rate_usd=2.0,
            hard_ttl_seconds=10_800,
            allowed_active_instance_ids=[47373597],
        )


def test_egl_graphics_arguments_are_pinned_without_running_node() -> None:
    """The flag contract, pinned hermetically so CI always enforces it.

    The subprocess test below skips wherever the renderer's dependencies are not
    installed, which is every CI runner -- so on its own it left this contract
    unenforced exactly where enforcement matters. This reads the source instead
    and runs everywhere.

    The flags matter: ``--use-gl=egl`` alone can leave WebGL disabled in
    headless Chromium even on a container exposing an NVIDIA GPU, and dropping
    ``--disable-software-rasterizer`` lets a run silently fall back to software
    and report a render that never touched the GPU.
    """
    renderer = Path(__file__).resolve().parents[1] / "tools/splat_render/render_splat.mjs"
    source = renderer.read_text(encoding="utf-8")
    egl_block = source.split('if (backend === "egl")', 1)[1].split("return [", 1)[1]
    egl_block = egl_block.split("]", 1)[0]
    flags = re.findall(r'"([^"]+)"', egl_block)

    assert flags == [
        "--use-gl=angle",
        "--use-angle=gl-egl",
        "--ignore-gpu-blocklist",
        "--disable-software-rasterizer",
        "--enable-webgl",
    ]


def test_egl_renderer_uses_angle_gl_egl_without_software_fallback() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node unavailable")
    renderer = Path(__file__).resolve().parents[1] / "tools/splat_render/render_splat.mjs"
    # `node_modules` is not committed, so on a runner that never installed them
    # the renderer exits before printing anything. That is the same class of
    # environmental absence as a missing `node`, but it was surfacing as a bare
    # CalledProcessError -- which failed CI on every PR touching this lane and
    # said nothing about the flags this test exists to pin.
    if not (renderer.parent / "node_modules").is_dir():
        pytest.skip("splat renderer dependencies not installed")
    completed = subprocess.run(
        [
            node,
            str(renderer),
            "--graphics-backend",
            "egl",
            "--print-graphics-args",
            "--out",
            str(Path.cwd()),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    # An installed renderer that cannot report its flags is a real regression,
    # so report what it said rather than raising an opaque subprocess error.
    assert completed.returncode == 0, (
        f"renderer exited {completed.returncode}: {completed.stderr.strip()[:500]}"
    )

    assert json.loads(completed.stdout) == [
        "--use-gl=angle",
        "--use-angle=gl-egl",
        "--ignore-gpu-blocklist",
        "--disable-software-rasterizer",
        "--enable-webgl",
    ]


def _task_freeze(task_id: str, slot: int) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "dual_task_task_freeze.v1",
        "task_id": task_id,
        "prompt": f"relocate observed object {slot}",
        "task_kind": "rigid_object_manipulation",
        "scene_freeze_digest": _digest("a"),
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "frozen_before_learned_policy_execution": True,
        "learned_policy_outcomes_accessed": False,
        "source_object": {
            "instance_id": f"source_{slot}",
            "semantic_label": "fixture_object",
            "observed_bounds_world_m": {
                "minimum": [0.0, 0.0, 0.0],
                "maximum": [0.1, 0.1, 0.1],
            },
            "observed_pose_world": {
                "position_world_m": [0.05, 0.05, 0.05],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "support_or_attachment_id": f"support_{slot}",
            "collision_identity_receipt_digest": _digest("b"),
            "support_receipt_digest": _digest("c"),
            "franka_placement_packet_digest": _digest("d"),
            "visibility_receipt_digest": _digest("e"),
        },
        "removal_plan": {
            "removal_id": f"removal_{slot}",
            "mask_set_id": f"mask_set_{slot}",
            "source_collider_prim_path": f"/Root/source_{slot}",
            "collider_deletion_id": f"collider_{slot}",
            "replacement_asset_id": f"replacement_{slot}",
            "replacement_qualification_id": f"qualification_{slot}",
        },
        "cameras": {
            "external": f"external_{slot}",
            "wrist": f"wrist_{slot}",
            "overview": f"overview_{slot}",
        },
        "overview_camera_policy_input": False,
        "overview_camera_deterministic_scoring_input": False,
        "execution_contract": {
            "control_frequency_hz": 20,
            "maximum_steps": 200,
            "settle_window_steps": 10,
            "seeds": [slot],
            "canonical_scenario_cell_id": f"canonical_{slot}",
            "reset_state": {"robot": "home", "object": "source_start"},
        },
        "deterministic_success_predicates": ["released", "settled"],
        "failure_rungs": ["never_moved", "collision_failure"],
        "target_configuration": {
            "kind": "pose_volume",
            "position_bounds_world_m": {"minimum": [0.2, 0.2, 0.0], "maximum": [0.3, 0.3, 0.1]},
            "orientation_reference_xyzw": [0.0, 0.0, 0.0, 1.0],
            "maximum_orientation_error_rad": 0.1,
            "support_id": f"destination_{slot}",
            "release_required": True,
        },
        "articulation_graph": None,
        "task_freeze_digest": "",
    }
    value["task_freeze_digest"] = canonical_digest(value, digest_field="task_freeze_digest")
    return value


def _source_ply(path: Path) -> Path:
    values = np.arange(10, dtype=np.float32)
    return write_standard_3dgs_ply(
        SplatData(
            count=10,
            xyz=np.stack((values, values + 10, values + 20), axis=1),
            opacity=values + 30,
            f_dc=np.stack((values + 40, values + 50, values + 60), axis=1),
            scales=np.stack((values + 70, values + 80, values + 90), axis=1),
            quats=np.stack((values + 100, values + 110, values + 120, values + 130), axis=1),
            properties=(),
            sh_rest=None,
        ),
        path,
    )


def _camera_contract(path: Path, camera_id: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "camera_id": camera_id,
            "T_world_camera_provider_frame": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0, "width": 2, "height": 2},
        }
    ]
    path.write_text(canonical_json(rows) + "\n", encoding="utf-8")
    return path


def _authority(path: Path) -> Path:
    value: dict[str, object] = {
        "schema_version": "third_scene_dual_task_execution_authority.v1",
        "program_id": "arm-decision-proof-v1",
        "publisher_scene_id": "840920",
        "private_rights_admitted_scene_derived_uploads_authorized": True,
        "raw_interiorgs_upload_authorized": False,
        "training_authorized": False,
        "public_dataset_bytes_publication_authorized": False,
        "retention": "bounded_to_goal_then_provider_zero",
        "paid_compute": {
            "provider": "vast",
            "hard_total_spend_cap_usd": 12.0,
            "zero_retry": True,
            "provider_zero_required_for_lane": True,
        },
        "authority_digest": "",
    }
    value["authority_digest"] = canonical_digest(value, digest_field="authority_digest")
    _write_json(path, value)
    return path


def _repo(root: Path) -> tuple[Path, Path]:
    repo = root / "repo"
    renderer = repo / "tools" / "splat_render"
    (renderer / "src").mkdir(parents=True)
    for relative in (
        "render_splat.mjs",
        "harness.html",
        "package.json",
        "package-lock.json",
        "src/render_entry.mjs",
    ):
        target = renderer / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "{}\n" if target.suffix == ".json" else "fixture\n",
            encoding="utf-8",
        )
    scripts = repo / "scripts"
    scripts.mkdir()
    checkout = Path(__file__).resolve().parents[1]
    for name in (
        "run_adp_retained_scene_render_provider_runtime.sh",
        "adp_retained_scene_render_provider_runner.mjs",
    ):
        shutil.copy2(checkout / "scripts" / name, scripts / name)
    vendor = root / "vendor"
    for package in ("@sparkjsdev/spark", "fflate", "playwright", "playwright-core", "three"):
        target = vendor / package
        target.mkdir(parents=True)
        (target / "index.js").write_text("fixture\n", encoding="utf-8")
    for command in (
        ("init",),
        ("add", "."),
        (
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-m",
            "fixture",
        ),
    ):
        subprocess.run(["git", "-C", str(repo), *command], check=True, capture_output=True)
    return repo, vendor


def _inputs(
    tmp_path: Path,
    *,
    candidate_schema: str = "adp009b_direct_evidence_expansion_set.v1",
) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "direct_set"
    source = _source_ply(root / "source.ply")
    shared = root / "shared_scene_union"
    shared.mkdir(parents=True)
    deleted = np.array([1, 6], dtype=np.int64)
    retained_indices = np.array([0, 2, 3, 4, 5, 7, 8, 9], dtype=np.int64)
    np.save(shared / "deleted_source_indices.npy", deleted, allow_pickle=False)
    np.save(shared / "retained_source_indices.npy", retained_indices, allow_pickle=False)
    retained = write_standard_3dgs_ply_subset_exact(
        source, shared / "retained_scene_gaussians.ply", retained_indices
    )
    deleted_splats = write_standard_3dgs_ply_subset_exact(
        source, shared / "deleted_source_gaussians.ply", deleted
    )
    tasks: list[dict[str, object]] = []
    lanes: list[dict[str, object]] = []
    for slot in (1, 2):
        task = _task_freeze(f"task_{slot}", slot)
        freeze = tmp_path / "freezes" / f"task_{slot}.json"
        _write_json(freeze, task)
        removal = task["removal_plan"]
        assert isinstance(removal, dict)
        task_row: dict[str, object] = {
            "task_id": task["task_id"],
            "task_freeze_digest": task["task_freeze_digest"],
            "removal_id": removal["removal_id"],
            "mask_set_id": removal["mask_set_id"],
            "task_freeze": _absolute_record(freeze),
        }
        if candidate_schema == "adp009d_segment_contribution_cutout_set.v1":
            task_root = root / f"task_{slot}"
            task_root.mkdir()
            task_deleted_indices = np.array([deleted[slot - 1]], dtype=np.int64)
            task_retained_indices = np.setdiff1d(
                np.arange(10, dtype=np.int64), task_deleted_indices, assume_unique=True
            )
            np.save(
                task_root / "deleted_source_indices.npy",
                task_deleted_indices,
                allow_pickle=False,
            )
            task_deleted = write_standard_3dgs_ply_subset_exact(
                source,
                task_root / "deleted_source_gaussians.ply",
                task_deleted_indices,
            )
            task_retained = write_standard_3dgs_ply_subset_exact(
                source,
                task_root / "retained_scene_gaussians.ply",
                task_retained_indices,
            )
            task_row.update(
                {
                    "counts": {"source": 10, "deleted_total": 1, "retained_total": 9},
                    "outputs": {
                        "deleted_source_indices": _relative_record(
                            root, task_root / "deleted_source_indices.npy"
                        ),
                        "deleted_source_gaussians": _relative_record(root, task_deleted),
                        "retained_scene_gaussians": _relative_record(root, task_retained),
                    },
                }
            )
        tasks.append(task_row)
        camera = _camera_contract(tmp_path / "cameras" / f"task_{slot}.json", f"camera_{slot}")
        lanes.append({"task_id": task["task_id"], "camera_contract_path": str(camera)})
    candidate: dict[str, object] = {
        "schema_version": candidate_schema,
        "task_candidates": tasks,
        "shared_scene_union": {
            "counts": {"source": 10, "deleted_total": 2, "retained_total": 8},
            "outputs": {
                "deleted_source_gaussians": _relative_record(root, deleted_splats),
                "deleted_source_indices": _relative_record(
                    root, shared / "deleted_source_indices.npy"
                ),
                "retained_source_indices": _relative_record(
                    root, shared / "retained_source_indices.npy"
                ),
                "retained_scene_gaussians": _relative_record(root, retained),
            },
        },
        "source_standard_splat": _absolute_record(source),
        "claim_boundary": {"candidate_derived_layers_only": True},
        "receipt_digest": "",
    }
    candidate["receipt_digest"] = canonical_digest(candidate, digest_field="receipt_digest")
    candidate_path = root / "candidate.json"
    _write_json(candidate_path, candidate)
    return candidate_path, {"lanes": lanes, "candidate": candidate}


def test_seals_two_task_bundle_and_rehearses_exact_uploaded_entrypoint(tmp_path: Path) -> None:
    candidate, inputs = _inputs(tmp_path)
    repo, vendor = _repo(tmp_path)
    authority = _authority(tmp_path / "authority.json")
    request: dict[str, object] = {
        "schema_version": "adp009d_retained_scene_gpu_render_request.v1",
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009D",
        "frozen_before_render_execution": True,
        "learned_policy_outcomes_accessed": False,
        "candidate_set_path": str(candidate),
        "execution_authority_path": str(authority),
        "renderer_vendor_root": str(vendor),
        "task_lanes": inputs["lanes"],
        "private_upload_policy": {
            "raw_dataset_bytes_upload": False,
            "private_derived_upload": True,
            "provider_training": False,
            "publication": False,
            "retention": "bounded_to_goal_then_provider_zero",
        },
    }
    request = build_retained_scene_gpu_render_request(request)
    request_path = tmp_path / "request.json"
    _write_json(request_path, request)

    receipt = build_retained_scene_gpu_render_bundle(
        request_path=request_path, repo_root=repo, job_dir=tmp_path / "job"
    )

    assert receipt["status"] == "ready"
    assert receipt["source_pair_per_task"] is True
    assert receipt["retained_frame_per_task"] is True
    assert receipt["exact_bundle_entrypoint_rehearsal"]["status"] == "passed"
    assert receipt["exact_bundle_entrypoint_rehearsal"]["gpu_runtime_started"] is False
    assert receipt["exact_bundle_entrypoint_rehearsal"]["provider_mutations_performed"] == 0
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
    assert "provider_runtime/input/shared_deleted_source_layer.ply" in names
    assert "provider_runtime/input/shared_retained_scene.ply" in names
    assert "provider_runtime/input/direct_evidence_successor_set.json" in names
    assert (tmp_path / "job/provider_runtime/input/shared_deleted_source_layer.ply").stat().st_ino == (
        tmp_path / "direct_set/shared_scene_union/deleted_source_gaussians.ply"
    ).stat().st_ino
    source_shell = repo / "scripts/run_adp_retained_scene_render_provider_runtime.sh"
    bundled_shell = (
        tmp_path / "job/provider_runtime/run_adp_retained_scene_render_provider_runtime.sh"
    )
    assert source_shell.stat().st_ino != bundled_shell.stat().st_ino
    assert source_shell.stat().st_mode & 0o111 == 0
    assert bundled_shell.stat().st_mode & 0o111
    assert (
        validate_retained_scene_render_bundle(receipt)["bundle_sha256"] == receipt["bundle_sha256"]
    )
    dry_run = run_retained_scene_render_vast(
        job_dir=tmp_path / "vast_dry_run",
        paid_resource_admission_grant=None,
        execute=False,
        prepared_bundle=receipt,
    )
    assert dry_run["status"] == "dry_run_ready"
    assert dry_run["provider_mutations_performed"] == 0
    attempt_authority: dict[str, object] = {
        "schema_version": "adp009d_retained_scene_gpu_render_paid_attempt_authority.v1",
        "authority_kind": "explicit_user_direction_in_current_goal",
        "purpose": "exact_retained_scene_gpu_render",
        "provider": "vast",
        "paid_compute_authorized": True,
        "parent_execution_authority_digest": receipt["execution_authority"]["authority_digest"],
        "bundle_sha256": receipt["bundle_sha256"],
        "blueprint_commit": receipt["blueprint_commit"],
        "maximum_paid_attempts": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "hard_attempt_spend_cap_usd": 12.0,
        "maximum_single_resource_ttl_seconds": 10_800,
        "maximum_hourly_rate_usd": 2.0,
        "external_active_instance_allowlist": [47373597],
        "authorization_digest": "",
    }
    attempt_authority["authorization_digest"] = canonical_digest(
        attempt_authority, digest_field="authorization_digest"
    )
    assert (
        validate_retained_scene_render_paid_attempt_authority(
            attempt_authority,
            prepared_bundle=receipt,
            max_hourly_rate_usd=2.0,
            hard_ttl_seconds=10_800,
            allowed_active_instance_ids=[47373597],
        )["authorization_digest"]
        == attempt_authority["authorization_digest"]
    )
    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "vast_preflight",
        generated_at="2026-08-11T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_retained_scene_render",
        bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://example.test/bundle",
        provider_output_put_url="https://example.test/output",
    )
    assert preflight["status"] == "passed", preflight
    probe = _probe_shell_script(
        "https://example.test/heartbeat",
        enable_blueprint_bundle=True,
        provider_bundle_kind="adp_retained_scene_render",
    )
    assert "adp_retained_scene_render_provider_bundle" in probe
    assert "apt-get" not in probe


def test_rebuilds_current_commit_bundle_from_sealed_host_predecessor(
    tmp_path: Path,
) -> None:
    candidate_path, inputs = _inputs(tmp_path)
    original_candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    repo, vendor = _repo(tmp_path)
    authority = _authority(tmp_path / "authority.json")
    request = build_retained_scene_gpu_render_request(
        {
            "schema_version": "adp009d_retained_scene_gpu_render_request.v1",
            "program_id": "arm-decision-proof-v1",
            "adp_item": "ADP-009D",
            "frozen_before_render_execution": True,
            "learned_policy_outcomes_accessed": False,
            "candidate_set_path": str(candidate_path),
            "execution_authority_path": str(authority),
            "renderer_vendor_root": str(vendor),
            "task_lanes": inputs["lanes"],
            "private_upload_policy": {
                "raw_dataset_bytes_upload": False,
                "private_derived_upload": True,
                "provider_training": False,
                "publication": False,
                "retention": "bounded_to_goal_then_provider_zero",
            },
        }
    )
    request_path = tmp_path / "predecessor-request.json"
    _write_json(request_path, request)
    predecessor = build_retained_scene_gpu_render_bundle(
        request_path=request_path,
        repo_root=repo,
        job_dir=tmp_path / "predecessor-job",
    )
    predecessor_commit = predecessor["blueprint_commit"]

    (repo / "CURRENT_RELEASE").write_text("rebuild producer fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-m",
            "current release",
        ],
        check=True,
        capture_output=True,
    )
    current_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert current_commit != predecessor_commit

    output_root = tmp_path / "host-inputs" / "retained-scene-current"
    cli_program = """
from pathlib import Path
import sys
import blueprint_pipeline.retained_scene_sealed_bundle_rebuild as sealed_rebuild
from scripts.build_retained_scene_render_bundle import main

sealed_rebuild.DEFAULT_PRODUCTION_ROOTS = (Path(sys.argv[1]).resolve(),)
raise SystemExit(main(sys.argv[2:]))
"""
    cli = subprocess.run(
        [
            sys.executable,
            "-c",
            cli_program,
            str(tmp_path),
            "--sealed-predecessor-bundle",
            predecessor["bundle_path"],
            "--source-standard-splat",
            str(tmp_path / "direct_set/source.ply"),
            "--repo-root",
            str(repo),
            "--output-root",
            str(output_root),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert cli.returncode == 0, cli.stderr
    cli_result = json.loads(cli.stdout)
    assert cli_result["status"] == "ready"
    assert cli_result["blueprint_commit"] == current_commit
    receipt_path = Path(cli_result["receipt_path"])
    assert receipt_path == (
        output_root / "adp009d_retained_scene_sealed_bundle_rebuild.v1.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert receipt["status"] == "ready"
    assert receipt["source_commit_sha"] == current_commit
    assert receipt["provider_mutation_performed"] is False
    assert receipt["paid_resource_used"] is False
    assert receipt["scientific_execution_performed"] is False
    assert receipt["website_trigger_proven"] is False
    assert receipt["reconstructed_indices"]["disjoint"] is True
    assert receipt["reconstructed_indices"]["exhaustive"] is True
    outputs = original_candidate["shared_scene_union"]["outputs"]
    assert receipt["reconstructed_indices"]["deleted"]["sha256"] == outputs[
        "deleted_source_indices"
    ]["sha256"]
    assert receipt["reconstructed_indices"]["retained"]["sha256"] == outputs[
        "retained_source_indices"
    ]["sha256"]
    rebuilt_candidate = json.loads(
        (output_root / "rehydrated_scene/direct_evidence_successor_set.json").read_text(
            encoding="utf-8"
        )
    )
    assert rebuilt_candidate["sealed_predecessor_rebuild"][
        "absolute_authoring_paths_followed"
    ] is False
    assert rebuilt_candidate["source_standard_splat"]["path"].startswith(
        str(output_root)
    )
    for row in rebuilt_candidate["task_candidates"]:
        assert row["task_freeze"]["path"].startswith(str(output_root))
    serialized = canonical_json(rebuilt_candidate)
    assert str(tmp_path / "freezes") not in serialized
    assert str(candidate_path) not in serialized
    rebuilt_request = json.loads(
        (output_root / "retained_scene_gpu_render_request.current.json").read_text(
            encoding="utf-8"
        )
    )
    assert not Path(rebuilt_request["candidate_set_path"]).is_absolute()
    assert not Path(rebuilt_request["execution_authority_path"]).is_absolute()
    assert rebuilt_request["renderer_vendor_root"] == (
        "rehydrated_renderer_vendor/node_modules"
    )
    assert not (repo / "tools/splat_render/node_modules").exists()
    vendor_receipt = json.loads(
        (output_root / "rehydrated_renderer_vendor/receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert vendor_receipt["package_file_counts"] == {
        "@sparkjsdev/spark": 1,
        "fflate": 1,
        "playwright": 1,
        "playwright-core": 1,
        "three": 1,
    }
    assert receipt["renderer_vendor"]["receipt_digest"] == vendor_receipt[
        "receipt_digest"
    ]
    assert all(
        not Path(row["camera_contract_path"]).is_absolute()
        for row in rebuilt_request["task_lanes"]
    )
    rebuilt_bundle_receipt = json.loads(
        Path(receipt["bundle_receipt"]["path"]).read_text(encoding="utf-8")
    )
    assert rebuilt_bundle_receipt["blueprint_commit"] == current_commit
    assert validate_retained_scene_render_bundle(rebuilt_bundle_receipt)[
        "bundle_sha256"
    ] == rebuilt_bundle_receipt["bundle_sha256"]


@pytest.mark.parametrize(
    "candidate_schema",
    [
        "adp009b_ownership_coverage_cutout_set.v1",
        "adp009d_segment_contribution_cutout_set.v1",
    ],
)
def test_seals_broad_ownership_coverage_cutout_for_repair_render(
    tmp_path: Path, candidate_schema: str
) -> None:
    candidate, inputs = _inputs(
        tmp_path,
        candidate_schema=candidate_schema,
    )
    repo, vendor = _repo(tmp_path)
    authority = _authority(tmp_path / "authority.json")
    request = build_retained_scene_gpu_render_request(
        {
            "schema_version": "adp009d_retained_scene_gpu_render_request.v1",
            "program_id": "arm-decision-proof-v1",
            "adp_item": "ADP-009D",
            "frozen_before_render_execution": True,
            "learned_policy_outcomes_accessed": False,
            "candidate_set_path": str(candidate),
            "execution_authority_path": str(authority),
            "renderer_vendor_root": str(vendor),
            "task_lanes": inputs["lanes"],
            "private_upload_policy": {
                "raw_dataset_bytes_upload": False,
                "private_derived_upload": True,
                "provider_training": False,
                "publication": False,
                "retention": "bounded_to_goal_then_provider_zero",
            },
        }
    )
    request_path = tmp_path / "request.json"
    _write_json(request_path, request)

    receipt = build_retained_scene_gpu_render_bundle(
        request_path=request_path,
        repo_root=repo,
        job_dir=tmp_path / "job",
    )

    assert receipt["status"] == "ready"
    assert receipt["shared_deleted_source_layer"]["deleted_gaussian_count"] == 2
    assert receipt["shared_retained_scene"]["retained_gaussian_count"] == 8
    assert receipt["exact_bundle_entrypoint_rehearsal"]["status"] == "passed"

def test_rejects_more_than_five_task_lanes() -> None:
    request = {
        "schema_version": "adp009d_retained_scene_gpu_render_request.v1",
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009D",
        "frozen_before_render_execution": True,
        "learned_policy_outcomes_accessed": False,
        "candidate_set_path": "/candidate.json",
        "execution_authority_path": "/authority.json",
        "renderer_vendor_root": "/vendor",
        "task_lanes": [
            {"task_id": f"task_{slot}", "camera_contract_path": "/camera.json"} for slot in range(6)
        ],
        "private_upload_policy": {
            "raw_dataset_bytes_upload": False,
            "private_derived_upload": True,
            "provider_training": False,
            "publication": False,
            "retention": "bounded_to_goal_then_provider_zero",
        },
    }
    with pytest.raises(RetainedSceneRenderPacketError, match="task_lane_count_invalid"):
        build_retained_scene_gpu_render_request(request)


def test_a_run_inventories_its_evidence_in_the_shared_manifest_schema(
    tmp_path: Path,
) -> None:
    """The profile's terminal contract asks the result for
    `teardown_manifest_path` and `artifact_manifest_path`. This lane named
    neither, so every run ended `allocator_terminal_artifact_missing:` for both
    regardless of what happened on the provider.

    It uses the shared `task_evaluation_artifact_manifest.v1`, not a lane-local
    schema: `adp009d_live_readiness` and every future consumer validate that
    one, and a second schema would mean each lane's evidence needed a reader
    written for it."""

    from blueprint_pipeline.task_evaluation_artifact_manifest import (
        SCHEMA_VERSION,
        build_task_evaluation_artifact_manifest,
    )

    job = tmp_path / "job"
    (job / "immutable_execution" / "renders").mkdir(parents=True)
    (job / "immutable_execution" / "renders" / "front.png").write_bytes(b"frame")
    provider_run = job / "vast_provider_run"
    provider_run.mkdir()
    (provider_run / "vast_provider_adapter_result.json").write_text("{}", encoding="utf-8")
    (provider_run / "vast_teardown_manifest.json").write_text(
        '{"continuing_spend_from_this_run": false}', encoding="utf-8"
    )

    manifest = build_task_evaluation_artifact_manifest(
        attempt_root=job,
        artifact_roots={
            "provider_runtime_evidence": job / "immutable_execution",
            "allocator_adapter_result": provider_run / "vast_provider_adapter_result.json",
            "teardown_manifest": provider_run / "vast_teardown_manifest.json",
        },
        required_roles=(
            "provider_runtime_evidence",
            "allocator_adapter_result",
            "teardown_manifest",
        ),
        binding={"allocator_lane": "adp_retained_scene_render", "retry_cap": 0},
        output_path=job / "artifact_manifest.json",
    )

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["status"] == "completed"
    assert manifest["blockers"] == []
    assert {row["relative_path"] for row in manifest["files"]} == {
        "immutable_execution/renders/front.png",
        "vast_provider_run/vast_provider_adapter_result.json",
        "vast_provider_run/vast_teardown_manifest.json",
    }
    # Each file carries the roles it satisfies, so a reader can tell render
    # evidence from teardown evidence without knowing this lane's layout.
    roles = {row["relative_path"]: row["roles"] for row in manifest["files"]}
    assert roles["vast_provider_run/vast_teardown_manifest.json"] == ["teardown_manifest"]


def test_a_missing_required_role_blocks_the_manifest(tmp_path: Path) -> None:
    """Roles state what coverage is required, rather than sweeping whatever
    happens to be on disk and calling the result complete."""

    from blueprint_pipeline.task_evaluation_artifact_manifest import (
        build_task_evaluation_artifact_manifest,
    )

    job = tmp_path / "job"
    (job / "immutable_execution").mkdir(parents=True)
    (job / "immutable_execution" / "result.json").write_text("{}", encoding="utf-8")

    manifest = build_task_evaluation_artifact_manifest(
        attempt_root=job,
        artifact_roots={"provider_runtime_evidence": job / "immutable_execution"},
        required_roles=("provider_runtime_evidence", "teardown_manifest"),
        binding={"allocator_lane": "adp_retained_scene_render", "retry_cap": 0},
        output_path=job / "artifact_manifest.json",
    )

    assert manifest["status"] == "blocked"
    assert "task_evaluation_artifact_role_missing:teardown_manifest" in manifest["blockers"]


def _portable_request(
    *, candidate: str, authority: str, vendor: str, lanes: list[dict[str, object]]
) -> dict[str, object]:
    return build_retained_scene_gpu_render_request(
        {
            "schema_version": "adp009d_retained_scene_gpu_render_request.v1",
            "program_id": "arm-decision-proof-v1",
            "adp_item": "ADP-009D",
            "frozen_before_render_execution": True,
            "learned_policy_outcomes_accessed": False,
            "candidate_set_path": candidate,
            "execution_authority_path": authority,
            "renderer_vendor_root": vendor,
            "task_lanes": lanes,
            "private_upload_policy": {
                "raw_dataset_bytes_upload": False,
                "private_derived_upload": True,
                "provider_training": False,
                "publication": False,
                "retention": "bounded_to_goal_then_provider_zero",
            },
        }
    )


def _repo_with_internal_inputs(tmp_path: Path) -> Path:
    """A checkout carrying the authority and the vendored renderer, as the real
    one does. The scene bytes stay outside it: they are private and large."""

    repo, vendor = _repo(tmp_path)
    _authority(repo / "docs" / "authority.json")
    internal_vendor = repo / "tools" / "splat_render" / "node_modules"
    shutil.copytree(vendor, internal_vendor)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, capture_output=True)
    subprocess.run(
        [
            "git", "-C", str(repo),
            "-c", "user.name=fixture",
            "-c", "user.email=fixture@example.test",
            "commit", "-m", "inputs",
        ],
        check=True,
        capture_output=True,
    )
    return repo


def test_a_request_naming_relative_inputs_rebuilds_against_a_staged_scene_root(
    tmp_path: Path,
) -> None:
    """The committed v1 request pointed at two deleted /private/tmp directories,
    so the lane could not be rebuilt anywhere. A relative request says what it
    needs; the invocation says where."""

    candidate, inputs = _inputs(tmp_path)
    repo = _repo_with_internal_inputs(tmp_path)
    lanes = [
        {
            "task_id": str(lane["task_id"]),
            "camera_contract_path": str(
                Path(str(lane["camera_contract_path"])).relative_to(tmp_path)
            ),
        }
        for lane in inputs["lanes"]
    ]
    request_path = tmp_path / "portable_request.json"
    _write_json(
        request_path,
        _portable_request(
            candidate=str(candidate.relative_to(tmp_path)),
            authority="docs/authority.json",
            vendor="tools/splat_render/node_modules",
            lanes=lanes,
        ),
    )

    receipt = build_retained_scene_gpu_render_bundle(
        request_path=request_path,
        repo_root=repo,
        job_dir=tmp_path / "portable_job",
        scene_input_root=tmp_path,
    )

    assert receipt["status"] == "ready"
    assert receipt["blockers"] == []
    # The staged pair resolves wherever it lands, so the receipt travels with it.
    assert receipt["bundle_relative_path"] == "adp_retained_scene_gpu_render_bundle.zip"
    assert receipt["execution_authority"]["relative_path"] == (
        "provider_runtime/execution_authority.json"
    )
    assert receipt["request"]["relative_path"] == "provider_runtime/source_request_manifest.json"
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
    assert "provider_runtime/source_request_manifest.json" in names
    assert "provider_runtime/execution_authority.json" in names


def test_a_relative_request_input_cannot_escape_its_roots(tmp_path: Path) -> None:
    candidate, inputs = _inputs(tmp_path)
    repo = _repo_with_internal_inputs(tmp_path)
    request_path = tmp_path / "escaping_request.json"
    _write_json(
        request_path,
        _portable_request(
            candidate="../" + candidate.relative_to(tmp_path).as_posix(),
            authority="docs/authority.json",
            vendor="tools/splat_render/node_modules",
            lanes=[
                {
                    "task_id": str(lane["task_id"]),
                    "camera_contract_path": str(
                        Path(str(lane["camera_contract_path"])).relative_to(tmp_path)
                    ),
                }
                for lane in inputs["lanes"]
            ],
        ),
    )

    with pytest.raises(RetainedSceneRenderPacketError, match="candidate_set_missing"):
        build_retained_scene_gpu_render_bundle(
            request_path=request_path,
            repo_root=repo,
            job_dir=tmp_path / "escaping_job",
            scene_input_root=tmp_path,
        )


def test_a_relative_scene_input_without_a_scene_root_fails_closed(tmp_path: Path) -> None:
    candidate, inputs = _inputs(tmp_path)
    repo = _repo_with_internal_inputs(tmp_path)
    request_path = tmp_path / "rootless_request.json"
    _write_json(
        request_path,
        _portable_request(
            candidate=str(candidate.relative_to(tmp_path)),
            authority="docs/authority.json",
            vendor="tools/splat_render/node_modules",
            lanes=[
                {
                    "task_id": str(lane["task_id"]),
                    "camera_contract_path": str(
                        Path(str(lane["camera_contract_path"])).relative_to(tmp_path)
                    ),
                }
                for lane in inputs["lanes"]
            ],
        ),
    )

    with pytest.raises(RetainedSceneRenderPacketError, match="candidate_set_missing"):
        build_retained_scene_gpu_render_bundle(
            request_path=request_path,
            repo_root=repo,
            job_dir=tmp_path / "rootless_job",
        )


def test_the_committed_portable_request_resolves_its_repository_inputs() -> None:
    """The manifest the live lane is rebuilt from must not name a path that
    exists only on one workstation."""

    checkout = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (
            checkout
            / "docs/arm_decision_proof_v1/manifests"
            / "third_scene_840920_retained_scene_gpu_render_request.v2.json"
        ).read_text(encoding="utf-8")
    )
    request = build_retained_scene_gpu_render_request(manifest)

    assert request["request_digest"] == manifest["request_digest"]
    for key in ("candidate_set_path", "execution_authority_path", "renderer_vendor_root"):
        assert not str(manifest[key]).startswith("/"), key
    for lane in manifest["task_lanes"]:
        assert not str(lane["camera_contract_path"]).startswith("/")
    # The repository-side inputs must resolve in this checkout; the scene-side
    # ones are private bytes staged separately and are not asserted here.
    assert (checkout / manifest["execution_authority_path"]).is_file()
