"""Pinned, offline FLORA batch execution on already-admitted compute.

This module neither provisions resources nor downloads models. GPU libraries are
loaded only by the configured subprocess; importing it is safe on a CPU host.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


FLORA_MODEL = "nvidia/Flora-Rewarder"
FLORA_REVISION = "7de68d22c4a9decec5ddb7f86ddcaee8f82d3367"
MODES = ("terminal", "dense", "rubric")

# SHA256 entries are the bytes at the immutable Hugging Face revision above.
# Model shard hashes come from the revision's LFS metadata; no weights are
# downloaded by this adapter. Processor and source hashes were independently read.
SOURCE_SHA256: dict[str, str] = {'batch_utils/__init__.py': 'sha256:c5914f6e9b5271b58b8b953909fd4753db75f04419a5369b1a01105dbdd23933',
 'batch_utils/core.py': 'sha256:9a003548b6830ebd4efdba89b95179d1c7ce301b51068be3c53d8d55de8f9230',
 'batch_utils/dense_reward.py': 'sha256:5f4b8977941489b06526cec72d289278272e5022d9ba9d8fe564c8ff21dcaeeb',
 'batch_utils/engine.py': 'sha256:fe2d87a52d0578722d8c3ebd010f1d39dcf2735b70e96b47cfbd9ca643d7b4bf',
 'batch_utils/rubric_reward.py': 'sha256:76f0fc04a06df6d2c7ac071f097e4cf0e265274c9c4c05a22e02a7ab1909022c',
 'batch_utils/terminal_reward.py': 'sha256:c9603b78caad1261baade92bfd6faaaa203f02e102a4719e92cfef37f93e41a0'}
MODEL_FILES: dict[str, dict[str, Any]] = {'chat_template.jinja': {'sha256': 'sha256:c5f8b92325f7afa2fe9961e9eee0fdf363c9d7806112343e80601a89f0920135',
                         'size_bytes': 7941},
 'config.json': {'sha256': 'sha256:95b6f1f65acca29a2751a90a82d67d459fb5ffae0bb565c053763b05e6336de6',
                 'size_bytes': 4733},
 'generation_config.json': {'sha256': 'sha256:fb58199b78abc0e3c284d170ecddf488501ca9fe91d9e435d6801f85b1a50ff0',
                            'size_bytes': 178},
 'merges.txt': {'sha256': 'sha256:a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d',
                'size_bytes': 3353259},
 'model-00001-of-00011.safetensors': {'sha256': 'sha256:27ddf261e7d96f167de3f7876ea2fe95ac8093465c6b63d860ec179d6f37864c',
                                      'size_bytes': 5263851892},
 'model-00002-of-00011.safetensors': {'sha256': 'sha256:cc9830c3d936d5236dd93c941e73a57516df3061cbf7a00d9b3fcac4c67cc39c',
                                      'size_bytes': 5347741707},
 'model-00003-of-00011.safetensors': {'sha256': 'sha256:186ba71e6b2bc589e63417d2e6f1f05bb8265f9fcf1f69136855b7a74dd97a28',
                                      'size_bytes': 5347741766},
 'model-00004-of-00011.safetensors': {'sha256': 'sha256:1e6d5f50fcc97a50f7bb567b0b9de215cad67959027c49a0233d546f89117728',
                                      'size_bytes': 5347741769},
 'model-00005-of-00011.safetensors': {'sha256': 'sha256:1f208f06fe351c654a0ae59828b6e29353b562efe68441a6cefc8bc865d15d8c',
                                      'size_bytes': 5347741766},
 'model-00006-of-00011.safetensors': {'sha256': 'sha256:307760b22b4c34bae029edd23ac68138599a62e36cab0280c0eb601029ab1960',
                                      'size_bytes': 5347741766},
 'model-00007-of-00011.safetensors': {'sha256': 'sha256:529ef026cf546a9fe0e7bff8e365c07040f1e366375d3763003aaabefeff70c0',
                                      'size_bytes': 5347741763},
 'model-00008-of-00011.safetensors': {'sha256': 'sha256:23ecc2ef21db674d1fa64bdaf2bace7bbebfb9a23612b1b53ec6c863ad47fc69',
                                      'size_bytes': 5368714464},
 'model-00009-of-00011.safetensors': {'sha256': 'sha256:9f84d7f0061c7c4bb937364dce43452e16c31389e338b363b3aea82f75ad63be',
                                      'size_bytes': 5347746031},
 'model-00010-of-00011.safetensors': {'sha256': 'sha256:b6156f556b04a8f0e2f35aae35aa0a663ace4c06fd7b2fc99172d1fcd61d91e1',
                                      'size_bytes': 5347749961},
 'model-00011-of-00011.safetensors': {'sha256': 'sha256:3a870800475a2d1b527bd8457ef71c48b86215d238d4dd02c1a498fca78a541a',
                                      'size_bytes': 2148561105},
 'model.safetensors.index.json': {'sha256': 'sha256:353e2167012d966777af8e1b57f275afdd4569a42c631cd6b297b1dad0e06d20',
                                  'size_bytes': 138095},
 'preprocessor_config.json': {'sha256': 'sha256:27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516',
                              'size_bytes': 390},
 'processor_config.json': {'sha256': 'sha256:5845168a99bc9130ff4dbc0f2d91eb8fca782a844b36c8d6e54e5311dd1e33fd',
                           'size_bytes': 1191},
 'tokenizer.json': {'sha256': 'sha256:06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523',
                    'size_bytes': 19989325},
 'tokenizer_config.json': {'sha256': 'sha256:7488550a894dcf741616126e322ea31d66cf9457b59ec5825f4c7efdced421f8',
                           'size_bytes': 1164},
 'video_preprocessor_config.json': {'sha256': 'sha256:7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13',
                                    'size_bytes': 385},
 'vocab.json': {'sha256': 'sha256:ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003',
                'size_bytes': 6722759}}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _offline_environment() -> dict[str, str]:
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "FLORA_REWARDER_REPO"):
        env.pop(key, None)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
               PYTHONDONTWRITEBYTECODE="1")
    return env


def _execute(argv: list[str], *, log_path: Path, timeout: float) -> dict[str, Any]:
    """Bound the entire process group, including vLLM worker children."""
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("xb") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                   env=_offline_environment(), start_new_session=True)
        timed_out = False
        try:
            code = process.wait(timeout=max(0.01, timeout))
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # A crashed parent can leave vLLM workers alive. They must not hold
            # device memory when the next mode starts or the caller returns.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                process.wait(timeout=30)
        code = process.returncode
    return {"argv": argv, "returncode": code, "timed_out": timed_out,
            "process_group_cleanup_attempted": True,
            "wall_time_seconds": time.monotonic() - started,
            "log_path": str(log_path), "log_sha256": file_digest(log_path)}


_DIAGNOSTIC = r'''
import importlib, importlib.metadata, json, os, sys
source, model, jobs_path, report_path, tensor_parallel, mode = sys.argv[1:]
sys.path.insert(0, source)
result = {"versions": {}, "devices": [], "media": [], "blockers": [], "completed": False}
def checkpoint():
    temporary = report_path + ".partial"
    with open(temporary, "w") as stream:
        json.dump(result, stream, allow_nan=False)
    os.replace(temporary, report_path)
exact = {"vllm":"0.24.0", "transformers":"5.8.1", "qwen-vl-utils":"0.0.14", "decord":"0.6.0", "av":"16.1.0"}
modules = {"vllm":"vllm", "transformers":"transformers", "qwen-vl-utils":"qwen_vl_utils",
           "decord":"decord", "av":"av", "numpy":"numpy", "Pillow":"PIL",
           "huggingface_hub":"huggingface_hub", "torch":"torch"}
for package, module in modules.items():
    try:
        version = importlib.metadata.version(package)
        result["versions"][package] = version
        importlib.import_module(module)
        if package in exact and version != exact[package]:
            result["blockers"].append("dependency_version_mismatch:" + package)
    except Exception as exc:
        result["blockers"].append("dependency_import_failed:" + package + ":" + type(exc).__name__)
try:
    from packaging.specifiers import SpecifierSet
    for package, spec in {"numpy":"<3", "Pillow":">=11,<13", "huggingface_hub":">=1.0"}.items():
        if result["versions"].get(package, "0") not in SpecifierSet(spec):
            result["blockers"].append("dependency_version_mismatch:" + package)
    if mode == "gpu":
        import torch
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if count < int(tensor_parallel):
            result["blockers"].append("insufficient_visible_cuda_devices")
        for index in range(count):
            properties = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            result["devices"].append({"index": index, "name": properties.name,
                                      "total_bytes": total, "free_bytes": free,
                                      "compute_capability": [properties.major, properties.minor]})
except Exception as exc:
    result["blockers"].append("cuda_probe_failed:" + type(exc).__name__)
if not result["blockers"]:
    try:
        from batch_utils import core, engine, rubric_reward
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        processor = engine.load_processor(model)
        SamplingParams(temperature=0.0, max_tokens=640,
                       structured_outputs=StructuredOutputsParams(regex=core.dense_regex()))
        jobs = json.load(open(jobs_path))
        result["requested_episode_count"] = len(jobs)
        checkpoint()
        for job in jobs:
            try:
                collage = core.Collage(job["videos"])
                windows = core.dense_windows(collage.num_frames, collage.fps)
                indices = core.terminal_frame_indices(collage.num_frames, collage.fps)
                # Exercise every decoded dense and terminal frame before inference.
                collage.frames(sorted(set(indices) | {i for w in windows for i in w.frame_indices}))
                engine.render(processor, core.terminal_messages(job["task"], True), thinking=False)
                for window in windows:
                    engine.render(processor, core.dense_messages(job["task"], window), thinking=False)
                prompt = core.rubric_prompt(job["task"], job["rubric"], core.rubric_header(job["robot"], job["cameras"]))
                messages = core.rubric_messages(prompt, job["videos"], fps=2.0, max_frames=166)
                engine.render(processor, messages, thinking=True, vision_ids=True)
                rubric_reward.video_inputs(messages)
                media = {"id":job["id"], "fps":collage.fps,
                    "num_frames":collage.num_frames, "duration_s":collage.duration_s,
                    "terminal_sampled_frames":len(indices), "dense_windows":len(windows)}
                collage.release()
                result["media"].append(media)
                checkpoint()
                print(json.dumps({"event":"flora_preflight_episode_complete", "episode_id":job["id"],
                    "completed_episode_count":len(result["media"]), "requested_episode_count":len(jobs),
                    "preflight_kind":mode}, sort_keys=True), flush=True)
            except Exception as exc:
                result["blockers"].append("media_preflight_failed:" + job["id"] + ":" + type(exc).__name__)
                checkpoint()
    except Exception as exc:
        result["blockers"].append("runtime_preflight_failed:" + type(exc).__name__)
result["completed"] = True
checkpoint()
'''


def _preflight_flora_runtime(
    profile: Mapping[str, Any], *, jobs: Sequence[Mapping[str, Any]] = (),
    output_root: str | Path, cpu_import_only: bool = False,
) -> dict[str, Any]:
    """Check local bytes, imports, CUDA and all input preprocessing; never infer."""
    config = profile["runtime_config"]
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []
    source = Path(config["source_root"])
    model = Path(config["model_path"])
    for name, expected in SOURCE_SHA256.items():
        path = source / name
        if not path.is_file():
            blockers.append("flora_source_missing:" + name)
        elif file_digest(path) != expected:
            blockers.append("flora_source_digest_mismatch:" + name)
    for name, expected in MODEL_FILES.items():
        if cpu_import_only and name.endswith(".safetensors"):
            continue
        path = model / name
        if not path.is_file():
            blockers.append("flora_checkpoint_missing:" + name)
        elif path.stat().st_size != expected["size_bytes"] or file_digest(path) != expected["sha256"]:
            blockers.append("flora_checkpoint_digest_mismatch:" + name)
    if not Path(config["python_executable"]).is_file():
        blockers.append("flora_python_executable_missing")
    report: dict[str, Any] = {
        "schema_version": "flora_runtime_preflight.v1", "model": FLORA_MODEL,
        "revision": FLORA_REVISION, "source_manifest_digest": canonical_digest(SOURCE_SHA256),
        "checkpoint_manifest_digest": canonical_digest(MODEL_FILES),
        "blockers": blockers, "learned_model_executed": False,
        "resource_allocation_performed": False, "network_disabled_for_model_loading": True,
        "preflight_kind": "cpu_import_and_preprocessing" if cpu_import_only else "gpu_runtime",
        "model_weight_checks_required": not cpu_import_only, "cuda_checks_required": not cpu_import_only,
    }
    if not blockers:
        # Only the six checked source files enter the Python import path. Existing
        # __pycache__ or adjacent modules cannot override them.
        verified = root / "verified_source"
        for name in SOURCE_SHA256:
            target = verified / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source / name).read_bytes())
            if file_digest(target) != SOURCE_SHA256[name]:
                raise ValueError("flora_source_changed_during_preflight")
        jobs_path, report_path = root / "jobs.json", root / "diagnostic.json"
        _write(jobs_path, list(jobs))
        execution = _execute(
            [config["python_executable"], "-I", "-B", "-c", _DIAGNOSTIC,
             str(verified), str(model), str(jobs_path), str(report_path),
             str(config["tensor_parallel_size"]), "cpu" if cpu_import_only else "gpu"],
            log_path=root / "diagnostic.log",
            timeout=min(config["timeout_seconds"], 1800 if cpu_import_only else 300),
        )
        report["diagnostic_execution"] = execution
        if execution["returncode"] or execution.get("timed_out") or not report_path.is_file():
            blockers.append("flora_runtime_diagnostic_failed")
        if execution.get("timed_out"):
            blockers.append("flora_runtime_diagnostic_timed_out")
        if report_path.is_file():
            try:
                diagnostics = json.loads(report_path.read_text())
                if not isinstance(diagnostics, dict) or not isinstance(diagnostics.get("blockers"), list):
                    raise ValueError("invalid diagnostic shape")
                report["diagnostics"] = diagnostics
                blockers.extend(diagnostics["blockers"])
                if diagnostics.get("completed") is not True:
                    blockers.append("flora_runtime_diagnostic_incomplete")
            except (ValueError, OSError):
                blockers.append("flora_runtime_diagnostic_report_invalid")
        report["verified_source_root"] = str(verified)
    report["execute_ready"] = not blockers and not cpu_import_only
    report["cpu_imports_and_preprocessing_ready"] = not blockers
    _write(root / "preflight.json", report)
    return report


def preflight_flora_runtime(
    profile: Mapping[str, Any], *, jobs: Sequence[Mapping[str, Any]] = (),
    output_root: str | Path,
) -> dict[str, Any]:
    return _preflight_flora_runtime(profile, jobs=jobs, output_root=output_root)


def preflight_flora_cpu_imports(
    profile: Mapping[str, Any], *, jobs: Sequence[Mapping[str, Any]] = (),
    output_root: str | Path,
) -> dict[str, Any]:
    """Same deterministic imports/preprocessing, without weights or CUDA admission.

    This CPU rehearsal always reports execute_ready=False. It cannot satisfy
    the GPU runtime preflight or authorize paid execution.
    """
    return _preflight_flora_runtime(profile, jobs=jobs, output_root=output_root, cpu_import_only=True)


def upstream_argv(
    profile: Mapping[str, Any], *, mode: str, source_root: Path,
    jobs_path: Path, output_path: Path,
) -> list[str]:
    if mode not in MODES:
        raise ValueError("flora_mode_invalid")
    config = profile["runtime_config"]
    bootstrap = "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); runpy.run_module('batch_utils." + mode + "_reward',run_name='__main__')"
    argv = [config["python_executable"], "-I", "-B", "-c", bootstrap, str(source_root),
            "--jobs", str(jobs_path), "--output", str(output_path),
            "--model", config["model_path"], "--revision", FLORA_REVISION,
            "--tensor-parallel-size", str(config["tensor_parallel_size"]),
            "--seed", str(config["seed"]), "--max-num-seqs", "8"]
    if mode == "terminal":
        argv.extend(["--with-rationale", "--max-model-len", "16384", "--gpu-memory-utilization", "0.88"])
    elif mode == "dense":
        argv.extend(["--frame-scores", "--tail-trim", "4", "--max-model-len", "16384",
                     "--gpu-memory-utilization", "0.88"])
    else:
        argv.extend(["--num-samples", str(config["rubric_num_samples"]), "--fps", "2",
                     "--max-frames", "166", "--temperature", "1", "--max-new-tokens", "20480",
                     "--max-model-len", "98304", "--max-num-batched-tokens", "24576",
                     "--gpu-memory-utilization", "0.85"])
    return argv


class FloraSubprocessRunner:
    """Three model loads per cohort, independent of episode count."""

    def run_batch(self, *, profile: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]],
                  output_root: Path) -> dict[str, Any]:
        started = time.monotonic()
        result: dict[str, Any] = {"records": {}, "executions": [], "errors": []}
        preflight = preflight_flora_runtime(profile, jobs=jobs, output_root=output_root / "preflight")
        result["preflight"] = preflight
        if not preflight["execute_ready"]:
            result["errors"] = ["flora_runtime_preflight_failed"]
            return result
        basic_jobs = [{key: job[key] for key in ("id", "task", "videos")} for job in jobs]
        for name, rows in (("episode_jobs", basic_jobs), ("rubric_jobs", list(jobs))):
            path = output_root / (name + ".jsonl")
            path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
        for mode in MODES:
            remaining = profile["runtime_config"]["timeout_seconds"] - (time.monotonic() - started)
            if remaining <= 0:
                result["errors"].append("flora_batch_timeout")
                break
            output = output_root / (mode + ".jsonl")
            argv = upstream_argv(profile, mode=mode, source_root=Path(preflight["verified_source_root"]),
                                 jobs_path=output_root / ("rubric_jobs.jsonl" if mode == "rubric" else "episode_jobs.jsonl"),
                                 output_path=output)
            execution = _execute(argv, log_path=output_root / (mode + ".log"), timeout=remaining)
            execution["mode"] = mode
            result["executions"].append(execution)
            if output.is_file():
                execution.update(output_path=str(output), output_sha256=file_digest(output))
                try:
                    result["records"][mode] = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
                except (json.JSONDecodeError, UnicodeError):
                    result["errors"].append("flora_output_json_invalid:" + mode)
            if execution["returncode"] or not output.is_file():
                result["errors"].append("flora_mode_execution_failed:" + mode)
                break
        result["wall_time_seconds"] = time.monotonic() - started
        return result
