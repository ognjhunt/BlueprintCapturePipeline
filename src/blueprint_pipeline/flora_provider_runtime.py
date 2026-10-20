"""GPU worker for nvidia/Flora-Rewarder; development_only_episode_diagnostic."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _verify(root: Path, manifest: dict[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("flora_runtime_member_manifest_missing")
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
            raise ValueError("flora_runtime_member_invalid")
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for part in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(part)
        if "sha256:" + h.hexdigest() != expected:
            raise ValueError("flora_runtime_member_digest_mismatch:" + name)


def _benchmark_execution_summary(report: dict[str, Any]) -> dict[str, Any]:
    """A saved partial/fixture report is not proof that the model ran."""
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    if (report.get("report_digest") != canonical_digest(report, digest_field="report_digest")
            or report.get("runtime_execution_verified") is not True
            or report.get("inference_performed") is not True
            or report.get("fixture_episode_count") != 0 or report.get("unknown_episode_count") != 0
            or type(report.get("live_episode_count")) is not int or report["live_episode_count"] <= 0
            or report.get("status") not in {"completed", "partial"}):
        raise ValueError("flora_benchmark_real_model_execution_unverified")
    return {"runtime_execution_verified": True, "inference_performed": True,
            "benchmark_status": report["status"], "benchmark_grades_complete": report["status"] == "completed",
            "live_episode_count": report["live_episode_count"], "benchmark_metrics": report.get("metrics")}


def main() -> int:
    root = Path(os.environ["BLUEPRINT_EVALUATOR_PROVIDER_BUNDLE_DIR"]).resolve()
    output = Path(os.environ["BLUEPRINT_EVALUATOR_PROVIDER_OUTPUT_DIR"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result: dict[str, Any] = {
        "schema_version": "flora_provider_runtime.v1", "status": "blocked", "blockers": [],
        "model_id": "nvidia/Flora-Rewarder", "claim_class": "development_only_episode_diagnostic",
        "candidate_policy_queried": False, "physical_truth_claimed": False,
        "official_ranking_performed": False,
    }
    try:
        runtime = json.loads((root / "provider_runtime/evaluator_provider_runtime_manifest.json").read_text())
        _verify(root, runtime)
        inputs = json.loads((root / "provider_runtime/evaluator_input_manifest.json").read_text())
        if inputs.get("model_id") != "nvidia/Flora-Rewarder" or inputs.get("model_revision") != "7de68d22c4a9decec5ddb7f86ddcaee8f82d3367":
            raise ValueError("flora_runtime_model_identity_invalid")
        sys.path.insert(0, str(root / "provider_runtime/source/src"))
        benchmark = root / "benchmark"
        from blueprint_pipeline.flora_dependency_bootstrap import (
            FloraDependencyBootstrapError, ensure_flora_runtime_dependencies,
        )
        try:
            result["dependency_bootstrap"] = ensure_flora_runtime_dependencies(
                output_root=output / "dependency_bootstrap")
        except FloraDependencyBootstrapError as exc:
            result["dependency_bootstrap"] = exc.receipt
            raise
        versions = {}
        for package, module in {"vllm":"vllm", "transformers":"transformers", "qwen-vl-utils":"qwen_vl_utils",
                "decord":"decord", "av":"av", "numpy":"numpy", "Pillow":"PIL", "huggingface_hub":"huggingface_hub"}.items():
            versions[package] = importlib.metadata.version(package)
            importlib.import_module(module)
        for package, expected in {"vllm":"0.24.0", "transformers":"5.8.1", "qwen-vl-utils":"0.0.14", "decord":"0.6.0", "av":"16.1.0"}.items():
            if versions[package] != expected:
                raise ValueError("flora_image_dependency_version_mismatch:" + package)
        _write(output / "runtime_dependency_versions.json", versions)
        from huggingface_hub import snapshot_download
        model = snapshot_download(repo_id=inputs["model_id"], revision=inputs["model_revision"],
            cache_dir=str(root / "model_cache"), allow_patterns=["*.json", "*.safetensors", "*.py", "*.jinja", "*.txt", "*.model"])
        profile = json.loads((benchmark / inputs["profile_relative_path"]).read_text())
        config = profile["runtime_config"]
        for key, expected in {"model_path":"/opt/blueprint/flora/model", "source_root":"/opt/blueprint/flora/source"}.items():
            if config.get(key) != expected:
                raise ValueError("flora_provider_profile_runtime_path_invalid:" + key)
            link = Path(expected)
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.exists() or link.is_symlink():
                if link.resolve() != Path(model).resolve():
                    raise ValueError("flora_provider_existing_model_path_conflict")
            else:
                link.symlink_to(Path(model).resolve(), target_is_directory=True)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        _write(output / "model_snapshot_receipt.json", {"model_id": inputs["model_id"],
            "revision": inputs["model_revision"], "snapshot_directory_name": Path(model).name,
            "source_commit": runtime["source_commit"]})
        environment = {**os.environ, "PYTHONPATH": str(root / "provider_runtime/source/src")}
        command = [sys.executable, "-m", "blueprint_pipeline.flora_episode_benchmark", "run",
            "--manifest", str(benchmark / inputs["manifest_relative_path"]),
            "--profile", str(benchmark / inputs["profile_relative_path"]),
            "--rights-root", str(benchmark / inputs["rights_relative_path"]),
            "--output-dir", str(output / "benchmark")]
        with (output / "benchmark.log").open("w") as log:
            process = subprocess.run(command, cwd=benchmark, env=environment, stdout=log, stderr=subprocess.STDOUT)
        result["benchmark_exit_code"] = process.returncode
        report = output / "benchmark/benchmark_report.json"
        if process.returncode != 0 or not report.is_file():
            raise ValueError("flora_benchmark_failed_or_report_missing")
        result.update(_benchmark_execution_summary(json.loads(report.read_text())))
        result.update({"status": "completed", "benchmark_report_path": "benchmark/benchmark_report.json",
            "benchmark_report_sha256": "sha256:" + hashlib.sha256(report.read_bytes()).hexdigest(),
            "source_commit": runtime["source_commit"], "model_revision": inputs["model_revision"]})
    except Exception as exc:
        result["blockers"] = [str(exc) if isinstance(exc, ValueError) else type(exc).__name__]
    result["wall_seconds"] = round(time.time() - started, 3)
    _write(output / "evaluator_runtime_result.json", result)
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
