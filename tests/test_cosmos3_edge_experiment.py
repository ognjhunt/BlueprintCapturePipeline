from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import jsonschema

from blueprint_pipeline.cosmos3_edge_experiment import (
    ENABLE_ENV,
    EXPECTED_MODEL_ID,
    run_cosmos3_edge_experiment,
    validate_edge_configuration,
)
from blueprint_pipeline.evaluator_runtime_evidence import (
    build_wam_runtime_receipt,
    canonical_json_sha256,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _schema(name: str) -> dict[str, object]:
    return json.loads((Path(__file__).resolve().parents[1] / "docs" / "schemas" / name).read_text())


def _capture(tmp_path: Path) -> Path:
    root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write(root / "capture_descriptor.json", {"scene_id": "scene-1", "capture_id": "capture-1"})
    _write(root / "raw" / "manifest.json", {"scene_id": "scene-1", "capture_id": "capture-1"})
    return root


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path) -> Path:
    input_path = root / "pipeline" / "cosmos_training_export" / "cell-1.json"
    _write(input_path, {"frames": ["capture-derived-frame"], "actions": [[0, 0, 0, 0, 0, 0, 0]]})
    manifest = root / "pipeline" / "cosmos_training_export" / "edge_frozen_benchmark.json"
    _write(
        manifest,
        {
            "schema_version": "cosmos3_edge_frozen_benchmark.v1",
            "frozen": True,
            "privacy_safe_pipeline_derived": True,
            "cells": [
                {
                    "cell_id": "cell-1",
                    "input_path": input_path.name,
                    "input_sha256": _sha(input_path),
                    "accepted_anchor_id": "anchor-1",
                }
            ],
        },
    )
    return manifest


def _worker(tmp_path: Path) -> Path:
    worker = tmp_path / "edge-worker"
    worker.write_text(
        """#!/usr/bin/env python3
import json, pathlib, sys
output, outdir, mode, cell, input_sha, model_rev, code_rev, checkpoint_sha, config_sha = sys.argv[1:]
kind = {"forward_dynamics":"generated_video", "inverse_dynamics":"action_inference", "reasoning":"reasoning_result"}[mode]
suffix = ".mp4" if kind == "generated_video" else ".json"
artifact = pathlib.Path(outdir) / (kind + suffix)
artifact.write_bytes((kind + "-fixture-model-output").encode())
payload = {
 "schema_version":"cosmos3_edge_worker_result.v1", "status":"completed", "mode":mode,
 "model_id":"nvidia/Cosmos3-Edge", "parameter_count_billion":4,
 "model_revision":model_rev, "code_revision":code_rev, "checkpoint_sha256":checkpoint_sha,
 "configuration_sha256":config_sha, "input_sha256":input_sha,
 "outputs":[{"kind":kind,"path":artifact.name,"metadata":{"cell_id":cell}}],
 "metrics":{"wall_seconds":1.0,"peak_vram_bytes":1024},
 "grounding":{"status":"measured_fixture"}, "abstention":{"abstained":False}
}
pathlib.Path(output).write_text(json.dumps(payload) + "\\n")
""",
        encoding="utf-8",
    )
    worker.chmod(worker.stat().st_mode | stat.S_IXUSR)
    return worker


def _command(worker: Path) -> list[str]:
    return [
        str(worker),
        "{output}",
        "{output_dir}",
        "{mode}",
        "{cell_id}",
        "{input_sha256}",
        "{model_revision}",
        "{code_revision}",
        "{checkpoint_sha256}",
        "{configuration_sha256}",
    ]


def _video_ok(path: str | Path) -> dict[str, object]:
    return {
        "schema_version": "wam_generated_video_review_validation.v1",
        "status": "completed",
        "path": str(path),
        "blockers": [],
    }


def test_edge_experiment_is_distinct_dual_gated_and_runs_all_modes(tmp_path: Path) -> None:
    root = _capture(tmp_path / "capture")
    manifest = _fixture(root)
    checkpoint = tmp_path / "cosmos3-edge-checkpoint.safetensors"
    checkpoint.write_bytes(b"fixture checkpoint identity")
    worker = _worker(tmp_path)
    kwargs = {
        "capture_root": root,
        "frozen_manifest_path": manifest,
        "worker_command": _command(worker),
        "model_revision": "edge-model-revision",
        "code_revision": "edge-code-revision",
        "checkpoint_path": checkpoint,
        "configuration": {
            "inference_steps": 4,
            "precision": "bf16",
            "size": "256p",
            "fps": 12,
            "num_frames": 50,
            "action_sequence_length": 16,
            "action_embodiment": "general_camera_motion",
            "action_dimension": 9,
            "reasoning_max_tokens": 4096,
            "reasoning_video_fps": 4,
        },
        "license_compatible": True,
        "video_validator": _video_ok,
    }
    blocked = run_cosmos3_edge_experiment(**kwargs)
    assert blocked["status"] == "blocked"

    result = run_cosmos3_edge_experiment(
        **kwargs,
        allow_experiment=True,
        env={ENABLE_ENV: "true", "PATH": f"{worker.parent}:{os.defpath}"},
    )
    assert result["status"] == "completed_advisory"
    manifest_payload = json.loads(Path(result["attempt_manifest_path"]).read_text())
    adapter = json.loads(Path(result["wam_command_adapter_result_path"]).read_text())
    assert manifest_payload["attempt_count"] == 6
    assert len(manifest_payload["output_stability"]) == 3
    assert all(row["repeat_count"] == 2 for row in manifest_payload["output_stability"])
    assert {item["mode"] for item in manifest_payload["attempts"]} == {
        "forward_dynamics",
        "inverse_dynamics",
        "reasoning",
    }
    assert manifest_payload["claim_boundary"]["cosmos3_nano_sc3_qualification_inherited"] is False
    assert adapter["schema_version"] == "cosmos3_edge_wam_command_adapter.v1"
    assert adapter["adapter_id"] == "blueprint_cosmos3_edge_experimental_command_adapter"
    assert adapter["model_provenance"]["model"] == EXPECTED_MODEL_ID
    assert adapter["truth_boundary"]["rank_fidelity_result_proven"] is False
    assert result["stop_rule_evaluation"]["status"] == "stop"
    assert any(
        row["reason"] == "no_measurable_failure_or_cost_advantage"
        for row in result["stop_rule_evaluation"]["triggered_stop_rules"]
    )
    output_dir = Path(result["request_path"]).parent
    for artifact, schema in (
        (output_dir / "request.json", "cosmos3_edge_experiment_request.schema.json"),
        (output_dir / "attempt_manifest.json", "cosmos3_edge_attempt_manifest.schema.json"),
        (output_dir / "result.json", "cosmos3_edge_experiment_result.schema.json"),
    ):
        jsonschema.Draft202012Validator(_schema(schema)).validate(json.loads(artifact.read_text()))


def test_evaluator_runtime_receipt_recognizes_edge_without_inheriting_nano_profile() -> None:
    def digest(char: str) -> str:
        return "sha256:" + char * 64

    runtime_output = {
        "schema_version": "cosmos3_edge_wam_command_adapter.v1",
        "adapter_id": "blueprint_cosmos3_edge_experimental_command_adapter",
        "status": "completed",
        "blockers": [],
        "learned_wam_model_ran": True,
        "fresh_model_command_executed_this_invocation": True,
        "fresh_model_run_claimed": True,
        "fresh_model_run_steps": 1,
        "configured_inference_steps_per_model_run": 4,
        "edge_subprocess": {"status": "completed", "returncode": 0},
        "truth_boundary": {"generated_video_is_model_output": True},
        "model_provenance": {"model": EXPECTED_MODEL_ID},
        "rollouts": [
            {
                "rollout_id": "edge-cell-1",
                "generated_video_sha256": digest("a"),
                "generated_video_review_validation": {"status": "completed"},
            }
        ],
    }
    runtime_sha = canonical_json_sha256(runtime_output)
    backend = {
        "schema_version": "evaluator_backend_manifest.v1",
        "backend_id": "cosmos3_edge_experimental",
        "model_family": "cosmos3edge",
        "model_version": EXPECTED_MODEL_ID,
        "model_artifact_sha256": digest("b"),
        "backend_is_compute_provider": False,
    }
    runtime_identity = {
        "runtime_id": "edge-runtime-1",
        "provider_id": "runpod",
        "runtime_adapter_version": "1.0.0",
    }
    provider = {
        "schema_version": "evaluator_provider_execution.v1",
        "status": "succeeded",
        "execution_id": "edge-provider-exec-1",
        "runtime_id": "edge-runtime-1",
        "provider_id": "runpod",
        "runtime_output_sha256": runtime_sha,
        "model_artifact_sha256": digest("b"),
        "adapter_code_sha256": digest("c"),
        "runtime_manifest_sha256": digest("d"),
        "provider_is_evaluator_identity": False,
    }
    receipt = build_wam_runtime_receipt(
        {
            "runtime_output": runtime_output,
            "runtime_output_sha256": runtime_sha,
            "evaluator_backend": backend,
            "runtime_identity": runtime_identity,
            "adapter_code_sha256": digest("c"),
            "runtime_manifest_sha256": digest("d"),
            "license_manifest_sha256": digest("e"),
            "provider_execution": provider,
            "provider_execution_sha256": canonical_json_sha256(provider),
            "infrastructure_status": "succeeded",
            "fixture_or_proxy_model_output_used": False,
            "fallback_model_output_used": False,
            "stale_model_output_used": False,
        }
    )
    assert receipt["status"] == "validated"
    assert receipt["model_family"] == "cosmos3edge"
    assert receipt["runtime_adapter_id"] == "blueprint_cosmos3_edge_experimental_command_adapter"


def test_edge_model_card_envelope_blocks_g1_7d_action_claim() -> None:
    blockers = validate_edge_configuration(
        {
            "precision": "bf16",
            "size": "480p",
            "fps": 24,
            "num_frames": 50,
            "action_sequence_length": 16,
            "action_embodiment": "unitree_g1",
            "action_dimension": 7,
            "reasoning_max_tokens": 4096,
            "reasoning_video_fps": 4,
        }
    )
    assert "edge_action_embodiment_not_in_model_card_supported_list" in blockers


def test_official_cosmos_worker_wrapper_runs_offline_pinned_cli_contract(tmp_path: Path) -> None:
    package = tmp_path / "fake-package" / "cosmos_framework" / "scripts"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "inference.py").write_text(
        """import argparse, pathlib
p = argparse.ArgumentParser(add_help=False)
p.add_argument('-i'); p.add_argument('-o')
args, _ = p.parse_known_args()
out = pathlib.Path(args.o) / 'fixture'
out.mkdir(parents=True, exist_ok=True)
(out / 'vision.mp4').write_bytes(b'fixture-video')
""",
        encoding="utf-8",
    )
    input_path = tmp_path / "cell.json"
    config_path = tmp_path / "config.json"
    checkpoint = tmp_path / "checkpoint.bin"
    output = tmp_path / "worker-result.json"
    output_dir = tmp_path / "outputs"
    _write(
        input_path,
        {
            "mode_inputs": {
                "forward_dynamics": {
                    "name": "fixture",
                    "model_mode": "forward_dynamics",
                    "prompt": "offline fixture",
                }
            }
        },
    )
    _write(config_path, {"seed": 1, "size": "256p", "num_frames": 50, "fps": 12})
    checkpoint.write_bytes(b"pinned checkpoint")
    command = [
        sys.executable,
        str(Path(__file__).parents[1] / "scripts/run_cosmos3_edge_worker.py"),
        "--input",
        str(input_path),
        "--output",
        str(output),
        "--output-dir",
        str(output_dir),
        "--config",
        str(config_path),
        "--mode",
        "forward_dynamics",
        "--cell-id",
        "cell-1",
        "--model-revision",
        "model-rev",
        "--code-revision",
        "code-rev",
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-sha256",
        _sha(checkpoint),
        "--configuration-sha256",
        "f" * 64,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path / "fake-package")
    completed = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert result["schema_version"] == "cosmos3_edge_worker_result.v1"
    assert result["status"] == "completed"
    assert result["model_id"] == "nvidia/Cosmos3-Edge"
    assert result["outputs"][0]["kind"] == "generated_video"
