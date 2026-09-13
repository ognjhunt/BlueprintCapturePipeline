"""Prepare semantic teachers on the control plane before a GPU is allocated.

The capsule retains exact bytes and their logical paths. The provider restores
those paths in its isolated filesystem, validates every file, and consumes the
admitted teacher set without repeating image generation or preliminary grading.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import stat
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest, canonical_json

CPU_PREPARATION_ENV = "BLUEPRINT_ARTIFIXER_CPU_PRETRAINING_ONLY"
CANDIDATE_SCAN_LIMIT = 4096
CANDIDATE_LIMIT = 16
GPU_PREPARATION_REQUIRED_ENV = "BLUEPRINT_ARTIFIXER_PRETRAINING_REQUIRED"
CAPSULE_URL_ENV = "BLUEPRINT_ARTIFIXER_PRETRAINING_CAPSULE_URL"
CAPSULE_SHA_ENV = "BLUEPRINT_ARTIFIXER_PRETRAINING_CAPSULE_SHA256"
CAPSULE_BYTES_ENV = "BLUEPRINT_ARTIFIXER_PRETRAINING_CAPSULE_BYTES"
CAPSULE_SCHEMA = "task_evaluation_artifixer_pretraining_capsule.v1"
STATE_SCHEMA = "task_evaluation_artifixer_pretraining_state.v1"
LOGICAL_ROOT = Path("/var/lib/blueprint/task-evaluation-inputs/semantic-pretraining")
MAX_CAPSULE_BYTES = 8 * 1024**3
REVIEW_CACHE_ENV = "BLUEPRINT_ARTIFIXER_PRETRAINING_REVIEW_CACHE"


def _require(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError("artifixer_pretraining_" + code)


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path) -> dict:
    _require(path.is_file() and not path.is_symlink(), "record_path_invalid")
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), "record_invalid")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def reuse_real_pretraining_review(*, cache_path, current_input_path, output_root):
    """Reuse a real verdict only for byte-identical current multimodal input."""
    from .task_evaluation_artifixer_ai_visual_review import (
        AI_REVIEW_MODEL, build_artifixer_ai_visual_review_input,
    )
    cache = _json(Path(cache_path))
    _require(cache.get("cache_digest") == canonical_digest(cache, digest_field="cache_digest"),
             "review_cache_digest_invalid")
    values = {}
    for name in ("review_input", "review_execution"):
        row = cache[name]
        path = Path(row["path"])
        _require(_sha(path) == row["sha256"], "review_cache_file_changed")
        values[name] = _json(path)
    execution, original = values["review_execution"], values["review_input"]
    _require(execution.get("schema_version") == "task_evaluation_artifixer_ai_visual_review_execution.v1"
             and execution.get("execution_digest") == canonical_digest(execution, digest_field="execution_digest")
             and original.get("receipt_digest") == canonical_digest(original, digest_field="receipt_digest")
             and execution.get("final_composite_receipt_digest") == original["receipt_digest"]
             and execution.get("status") == "completed" and execution.get("provider_called") is True
             and execution.get("review_phase") == "pre_training_semantic_targets"
             and original.get("review_phase") == "pre_training_semantic_targets"
             and execution.get("reviewer", {}).get("model") == AI_REVIEW_MODEL
             and execution.get("reviewer", {}).get("runtime") == "openai_agents_sdk"
             and bool(execution.get("usage", {}).get("provider_response_id"))
             and not execution.get("synthetic_test_only")
             and not execution.get("synthetic_review_verdicts")
             and execution.get("raw_secret_values_recorded") is False
             and execution.get("response_store") is False
             and execution.get("tracing_disabled") is True,
             "review_cache_not_real_pretraining_evidence")
    current, _, inventory, _ = build_artifixer_ai_visual_review_input(
        final_composite_receipt_path=current_input_path)
    if canonical_digest({"input": current}) != execution.get("input_digest"):
        return None
    _require(len(inventory) == len(execution["frames"])
             and {r["camera_id"]: r["sha256"] for r in inventory}
             == {r["camera_id"]: r["frame_sha256"] for r in execution["frames"]},
             "review_cache_inventory_changed")
    root = Path(output_root)
    paths = {}
    for name, value in values.items():
        path = root / ("retained_" + name + ".json")
        path.write_text(canonical_json(value) + "\n")
        paths[name] = path
    trace = {"status": "reused_real_pretraining_review", "provider_call_performed": False,
             "source_execution_digest": execution["execution_digest"],
             "current_multimodal_input_digest": execution["input_digest"]}
    (root / "pretraining_review_reuse.json").write_text(canonical_json(trace) + "\n")
    return {"review_input": original, "review_input_path": paths["review_input"],
            "review": {"decision": execution["decision"],
                "execution_receipt": {"path": str(paths["review_execution"]),
                                      "execution_digest": execution["execution_digest"]},
                "review_receipt": ({"retained_execution_digest": execution["execution_digest"]}
                                   if execution["decision"] == "accepted" else None)}}


def _binding(stage_input: Mapping[str, Any]) -> str:
    from .task_evaluation_scene_configuration_diagnostic_checkpoint import (
        diagnostic_checkpoint_scientific_binding_digest,
    )
    return diagnostic_checkpoint_scientific_binding_digest(
        stage_input=stage_input,
        render_inputs=stage_input["construction_envelope"]["render_inputs_result"],
    )


def write_pretraining_state(*, state, stage_input, output_path) -> dict:
    value = {
        "schema_version": STATE_SCHEMA, "status": "admitted_for_gpu_training",
        "run_id": stage_input["run_id"], "source_commit": stage_input["source_commit"],
        "scientific_binding_digest": _binding(stage_input),
        "state": _plain(state), "gpu_execution_performed": False,
        "appearance_repair_qualified": False,
    }
    value["state_digest"] = canonical_digest(value, digest_field="state_digest")
    Path(output_path).write_text(canonical_json(value) + "\n")
    return value


def _extract(archive: Path, root: Path, *, max_bytes: int = MAX_CAPSULE_BYTES) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        infos = zipped.infolist()
        _require(len(infos) <= 100_000 and sum(i.file_size for i in infos) <= max_bytes,
                 "archive_budget_exceeded")
        _require(len({i.filename for i in infos}) == len(infos), "archive_duplicate_member")
        for info in infos:
            path = Path(info.filename)
            mode = info.external_attr >> 16
            _require(not path.is_absolute() and ".." not in path.parts
                     and stat.S_IFMT(mode) in (0, stat.S_IFREG, stat.S_IFDIR), "archive_path_invalid")
        zipped.extractall(root)
        for info in infos:
            path = root / info.filename
            path.chmod((info.external_attr >> 16) & 0o777 or (0o750 if info.is_dir() else 0o640))


@contextmanager
def _environment(values):
    previous = dict(os.environ)
    os.environ.update({k: str(v) for k, v in values.items()})
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def _launch_intent_digest(launch_root: Path) -> str | None:
    try:
        profile = _json(launch_root / "launch_profile.json")
    except (OSError, ValueError, TypeError):
        return None
    binding = profile.get("scene_attempt_binding")
    digest = binding.get("intent_digest") if isinstance(binding, Mapping) else None
    return digest if isinstance(digest, str) and digest else None


def discover_completed_training_candidates(job_dir, *, limit: int = CANDIDATE_LIMIT) -> list[Path]:
    """This intent's closed scene-configuration launches whose training bytes could be reused.

    A launch qualifies only when its profile names the same intent, its launch receipt is
    terminal, its post-teardown provider-zero receipt is confirmed, and the job retained the
    pretraining capsule, its receipt and the provider runtime archive. Newest first, bounded.
    The exact scientific identity is proven later by the reuse validator, never here.
    """
    job = Path(job_dir).resolve()
    launch_root = job.parent.parent
    own = _launch_intent_digest(launch_root)
    if own is None or job.name != "scene-configuration-job" or job.parent.name != "allocator":
        return []
    rows: list[tuple[float, Path]] = []
    try:
        siblings = [p for p in launch_root.parent.iterdir() if p.is_dir() and not p.is_symlink()]
    except OSError:
        return []
    for sibling in siblings[:CANDIDATE_SCAN_LIMIT]:
        if sibling == launch_root:
            continue
        try:
            if _launch_intent_digest(sibling) != own:
                continue
            receipt = _json(sibling / "launch_receipt.json")
            zero = _json(sibling / "post_teardown_provider_zero_receipt.json")
            candidate_job = sibling / "allocator" / "scene-configuration-job"
            if (receipt.get("status") not in {"blocked", "completed"}
                    or zero.get("status") != "provider_zero_confirmed"
                    or zero.get("provider_zero_verified") is not True
                    or zero.get("continuing_spend_from_this_run") is not False
                    or not all((candidate_job / name).is_file() for name in (
                        "api_pretraining_receipt.json", "api_pretraining_capsule.zip",
                        "vast_provider_run/vast_provider_runtime_output.zip"))):
                continue
            rows.append(((sibling / "post_teardown_provider_zero_receipt.json").stat().st_mtime, sibling))
        except (OSError, ValueError, TypeError, KeyError):
            continue
    rows.sort(key=lambda row: row[0], reverse=True)
    return [path for _, path in rows[:limit]]


def prepare_semantics_before_gpu(*, bundle_receipt, authority, job_dir, environment) -> dict:
    """Run the real CPU/API prefix, archive admission, and make no GPU call."""
    from .control_plane_disk_budget import reserve_control_plane_disk
    from .task_evaluation_scene_configuration_artifixer_driver import execute_artifixer_component
    from .task_evaluation_scene_configuration_builtin_producers import _validate_toolchain

    job = Path(job_dir)
    receipt_path = job / "api_pretraining_receipt.json"
    if receipt_path.exists():
        existing = _json(receipt_path)
        _require(existing.get("receipt_digest") == canonical_digest(existing, digest_field="receipt_digest")
                 and existing.get("bundle_sha256") == bundle_receipt["bundle_sha256"]
                 and existing.get("authority_digest") == authority["authority_digest"]
                 and _sha(Path(existing["capsule_path"])) == existing["capsule_sha256"],
                 "retained_receipt_invalid")
        return existing
    bundle = Path(bundle_receipt["bundle_path"])
    _require(_sha(bundle) == bundle_receipt["bundle_sha256"], "source_bundle_changed")
    key = canonical_digest({"bundle": bundle_receipt["bundle_sha256"],
                            "authority": authority["authority_digest"]})[7:]
    root = LOGICAL_ROOT / key
    _require(not root.exists(), "partial_preparation_requires_recovery")
    with zipfile.ZipFile(bundle) as zipped:
        source_unpacked_bytes = sum(row.file_size for row in zipped.infolist())
    # Base extraction, prepared/checkpoint copies, and the outgoing archive,
    # plus bounded frame/receipt overhead. Keep the shared disk floor intact.
    peak_bytes = 3 * source_unpacked_bytes + 512 * 1024**2
    with reserve_control_plane_disk("semantic_pretraining", target_root=LOGICAL_ROOT,
                                   expected_bytes=peak_bytes):
        root.mkdir(parents=True, mode=0o700)
        extracted = root / "bundle"
        _extract(bundle, extracted)
        runtime = extracted / "provider_runtime"
        toolchain = runtime / "toolchain"
        manifest = _json(toolchain / "task_evaluation_scene_configuration_toolchain.v1.json")
        for row in manifest["files"]:
            (toolchain / row["relative_path"]).chmod(0o555 if row.get("executable") else 0o444)
        for path in [toolchain, *toolchain.rglob("*")]:
            if path.is_dir():
                path.chmod(0o555)
        manifest, _ = _validate_toolchain(root=toolchain, expected_source_commit=bundle_receipt["source_commit"])
        spec = importlib.util.spec_from_file_location(
            "_pretraining_provider_runner", Path(__file__).resolve().parents[2]
            / "scripts/task_evaluation_scene_configuration_provider_runner.py")
        _require(spec is not None and spec.loader is not None, "runner_missing")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        portable = _json(runtime / "input/portable_construction_envelope.v1.json")
        envelope = runner._hydrate_envelope(runtime, portable)
        stage = envelope["recipe"]["stage_sequence"][0]
        _require(stage["adapter"]["id"] == "artifixer3d_observed_object_removal", "first_stage_invalid")
        config_path = Path(envelope["stage_configuration_references"][0]["materialized_path"])
        config = _json(config_path)
        work = root / "output"
        work.mkdir()
        stage_input = {
            "schema_version": "task_evaluation_scene_configuration_stage_production_input.v1",
            "run_id": envelope["run_id"], "stage": stage, "configuration": config,
            "configuration_sha256": _sha(config_path), "source_commit": bundle_receipt["source_commit"],
            "construction_source_commit": envelope["expected_production_commit"],
            "execution_mode": "production", "toolchain_digest": manifest["toolchain_digest"],
            "construction_envelope": envelope,
        }
        input_path = work / "stage_production_input.v1.json"
        input_path.write_text(canonical_json(stage_input) + "\n")
        dependencies = work / "dependencies.json"
        dependencies.write_text("[]\n")
        state_path = work / "pretraining_state.json"
        component = toolchain / manifest["stages"][stage["adapter"]["id"]]["component_entrypoint"]
        from .artifixer_completed_training_reuse import CANDIDATES_ENV, SOURCE_ENV
        discovered = ([] if environment.get(SOURCE_ENV) or environment.get(CANDIDATES_ENV)
                      else discover_completed_training_candidates(job))
        values = {
            **dict(environment), CPU_PREPARATION_ENV: "1",
            **({CANDIDATES_ENV: os.pathsep.join(str(p) for p in discovered)} if discovered else {}),
            "BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS": "1",
            "BLUEPRINT_SCENE_CONFIGURATION_RUNTIME_ROOT": str(runtime),
            "BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT": str(input_path),
            "BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES": str(dependencies),
            "BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT": str(work),
            "BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_ROOT": str(component.parent),
            "BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_RESULT": str(state_path),
        }
        # Secrets stay outside the transport root; only prepared data is archived.
        for name, value in values.items():
            if name.endswith("API_KEY_FILE"):
                _require(not Path(str(value)).is_relative_to(root), "secret_inside_capsule")
        with _environment(values):
            state = execute_artifixer_component(environment=values)
        _require(state.get("status") == "admitted_for_gpu_training", "admission_missing")
        inventory = []
        for path in sorted(root.rglob("*")):
            _require(not path.is_symlink(), "symlink_forbidden")
            if path.is_file():
                inventory.append({"path": str(path.relative_to(root)), "sha256": _sha(path),
                                  "size_bytes": path.stat().st_size})
        capsule = {
            "schema_version": CAPSULE_SCHEMA, "status": "admitted_for_gpu_training",
            "logical_root": str(root), "source_bundle_sha256": bundle_receipt["bundle_sha256"],
            "authority_digest": authority["authority_digest"], "run_id": state["run_id"],
            "source_commit": state["source_commit"], "scientific_binding_digest": state["scientific_binding_digest"],
            "state_path": str(state_path.relative_to(root)), "files": inventory,
            "gpu_execution_performed": False, "appearance_repair_qualified": False,
        }
        capsule["capsule_digest"] = canonical_digest(capsule, digest_field="capsule_digest")
        (root / "capsule_manifest.json").write_text(canonical_json(capsule) + "\n")
        archive = job / "api_pretraining_capsule.zip"
        _require(not archive.exists(), "archive_exists")
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zipped:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    zipped.write(path, str(path.relative_to(root)))
        result = {
            "schema_version": CAPSULE_SCHEMA, "status": "admitted_before_gpu_allocation",
            "bundle_sha256": bundle_receipt["bundle_sha256"], "authority_digest": authority["authority_digest"],
            "capsule_path": str(archive), "capsule_sha256": _sha(archive),
            "capsule_bytes": archive.stat().st_size, "capsule_digest": capsule["capsule_digest"],
            "logical_root": str(root), "run_id": state["run_id"], "source_commit": state["source_commit"],
            "source_unpacked_bytes": source_unpacked_bytes, "reserved_peak_bytes": peak_bytes,
            "gpu_execution_performed": False, "appearance_repair_qualified": False,
        }
        result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
        receipt_path.write_text(canonical_json(result) + "\n")
        # The complete immutable archive remains retained locally and is uploaded
        # before allocation. Release only this reconstructible expanded cache.
        for path in root.rglob("*"):
            if path.is_dir():
                path.chmod(0o700)
        shutil.rmtree(root)
        return result


def consume_pretraining_capsule(*, environment, stage_input) -> dict | None:
    """Restore and verify CPU-prepared inputs inside the provider filesystem."""
    url = str(environment.get(CAPSULE_URL_ENV) or "")
    if not url:
        return None
    _require(environment.get(CPU_PREPARATION_ENV) != "1", "cpu_capsule_recursion")
    expected = str(environment.get(CAPSULE_SHA_ENV) or "")
    expected_bytes = int(environment.get(CAPSULE_BYTES_ENV) or 0)
    _require(expected.startswith("sha256:") and len(expected) == 71
             and 0 < expected_bytes <= MAX_CAPSULE_BYTES and url.startswith("https://"), "transport_invalid")
    import urllib.request
    with tempfile.TemporaryDirectory(prefix="artifixer-pretraining-") as temporary:
        archive = Path(temporary) / "capsule.zip"
        with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as target:
            remaining = expected_bytes
            while remaining:
                block = response.read(min(1024**2, remaining))
                _require(bool(block), "download_truncated")
                target.write(block)
                remaining -= len(block)
            _require(not response.read(1), "download_exceeds_binding")
        _require(_sha(archive) == expected, "archive_digest_invalid")
        with zipfile.ZipFile(archive) as zipped:
            capsule = json.loads(zipped.read("capsule_manifest.json"))
        _require(capsule.get("schema_version") == CAPSULE_SCHEMA
                 and capsule.get("status") == "admitted_for_gpu_training"
                 and capsule.get("capsule_digest") == canonical_digest(capsule, digest_field="capsule_digest")
                 and capsule.get("run_id") == stage_input["run_id"]
                 and capsule.get("source_commit") == stage_input["source_commit"]
                 and capsule.get("scientific_binding_digest") == _binding(stage_input)
                 and capsule.get("gpu_execution_performed") is False
                 and capsule.get("appearance_repair_qualified") is False, "capsule_binding_invalid")
        root = Path(capsule["logical_root"])
        _require(root.parent == LOGICAL_ROOT and len(root.name) == 64
                 and all(c in "0123456789abcdef" for c in root.name)
                 and not any(p.is_symlink() for p in (root, *root.parents)), "restore_root_invalid")
        if root.exists():
            _require(_json(root / "capsule_manifest.json") == capsule, "restore_root_conflict")
        else:
            _extract(archive, root)
    declared = {row["path"] for row in capsule["files"]}
    observed = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    _require(len(declared) == len(capsule["files"])
             and observed == declared | {"capsule_manifest.json"}, "inventory_members_invalid")
    for row in capsule["files"]:
        relative = Path(row["path"])
        _require(not relative.is_absolute() and ".." not in relative.parts, "inventory_path_invalid")
        path = root / relative
        _require(path.is_file() and not path.is_symlink() and path.stat().st_size == row["size_bytes"]
                 and _sha(path) == row["sha256"], "inventory_file_changed")
    state_relative = Path(capsule["state_path"])
    _require(not state_relative.is_absolute() and ".." not in state_relative.parts,
             "state_path_invalid")
    state = _json(root / state_relative)
    _require(state.get("schema_version") == STATE_SCHEMA
             and state.get("state_digest") == canonical_digest(state, digest_field="state_digest")
             and state.get("status") == "admitted_for_gpu_training"
             and state.get("run_id") == stage_input["run_id"]
             and state.get("source_commit") == stage_input["source_commit"]
             and state.get("gpu_execution_performed") is False
             and state.get("appearance_repair_qualified") is False
             and state.get("scientific_binding_digest") == _binding(stage_input), "state_invalid")
    # Preserve the API receipts/frames in the ordinary provider output archive.
    # Keep the existing portable checkpoint location for downstream warm reuse.
    output = Path(environment["BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT"])
    api_output = (root / state_relative).parent
    shutil.copytree(api_output / "released_artifixer_runtime", output / "api_pretraining")
    checkpoint = output / "diagnostic_checkpoint"
    shutil.copytree(api_output / "diagnostic_checkpoint", checkpoint)
    from .task_evaluation_scene_configuration_diagnostic_checkpoint import (
        validate_scene_configuration_diagnostic_checkpoint,
    )
    prepared = state["state"]
    prepared["checkpoint_root"] = str(checkpoint)
    prepared["semantic_checkpoint"] = validate_scene_configuration_diagnostic_checkpoint(
        checkpoint_root=checkpoint)
    (output / "api_pretraining_capsule_receipt.json").write_text(canonical_json(capsule) + "\n")
    return prepared
