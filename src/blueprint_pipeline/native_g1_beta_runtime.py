"""Execute selected native G1 beta cells within the allocator's absolute deadline.

The native worker owns policy queries, SONIC, task scoring and simulator close.
This wrapper preserves partial query checkpoints and a reason for every cell.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import signal
import shutil
import subprocess
import time

from .decision_evidence_contracts import canonical_digest
from .local_policy_provisioning import remaining_seconds
from .native_g1_beta_bundle import _sha
from .native_g1_beta_packet import PLAN, CONTRACT, materialize_cell_packet
from .native_g1_beta_session import RESULT_FILENAME, RESULT_SCHEMA, MAX_OUTPUT_BYTES, validate_session_authority
from .native_g1_provider_runtime import _json, _template, _preflight_g1_runtime_imports

DEADLINE_ENV = "BLUEPRINT_POLICY_CANARY_HARD_DEADLINE_EPOCH"
WORKER_RESULT = "native_g1_development_worker_result.v1.json"
EPISODE = "native_g1_built_scene_policy_episode.v1.json"
TRACE = "native_g1_shared_scene_episode_trace.v1.json"
OUTPUT_MARGIN = 128 * 1024**2
MAX_TERMINAL_BYTES = 128 * 1024**2


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    if path.name == RESULT_FILENAME and len(data.encode()) > MAX_TERMINAL_BYTES:
        raise ValueError("g1_beta_terminal_byte_bound_exceeded")
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(data)
    os.replace(temporary, path)


def _inputs(root):
    from .beta_policy_screening_g1 import validate_execution_spec
    from .native_task_arena_bundle import verify_native_task_arena_packet
    authority = validate_session_authority(_json(root / "inputs/session_authority.json"))
    spec = validate_execution_spec(_json(root / "inputs/policy_execution_spec.json"), candidate_id=authority["candidate_ids"][0])
    packet, receipt, _ = verify_native_task_arena_packet(root / "inputs/scene_packet")
    if (authority["runtime_inputs_digest"] != spec["execution_spec_digest"]
            or authority["rights_authority"] != spec["rights_authority"]
            or receipt["arena_scene_plan_digest"] != spec["scene_plan_digest"]
            or _json(packet / "native_task_arena_packet_request.v1.json").get("g1_scene_derivation", {}).get("source_packet_receipt_digest")
                != spec["rights_authority"]["source_packet_receipt_digest"]):
        raise ValueError("g1_beta_runtime_inputs_changed")
    return authority, spec, packet


def _verify_relocated_publisher(root):
    """Retain controller provenance while verifying identical provider bytes."""
    from .native_g1_publisher_source_stage import verify_g1_publisher_source
    observed = verify_g1_publisher_source(root / "publisher-source")
    staged = _json(root / "publisher-source/native_g1_publisher_source_stage.v1.json")
    if staged.get("receipt_digest") != canonical_digest(staged, digest_field="receipt_digest"):
        raise ValueError("g1_beta_runtime_publisher_changed")
    relocated = {**staged, "source_root": str(root / "publisher-source/source")}
    relocated["receipt_digest"] = canonical_digest(relocated, digest_field="receipt_digest")
    if relocated != observed:
        raise ValueError("g1_beta_runtime_publisher_changed")
    return observed


def _result(authority, episodes, *, stage, blockers):
    admission = authority["beta_screening_admission"]
    # Ship references to the retained native JSON, not multiple copies of every
    # trace and action array. The controller hydrates and verifies them before
    # scientific projection; the provider cannot assert its own closeout.
    wire_episodes = []
    for row in episodes:
        wire_episodes.append({**{key: value for key, value in row.items()
            if key not in {"worker_result", "worker_preclose", "episode", "trace", "query_checkpoints"}},
            "evidence_delivery": "retained_artifact_references.v1",
            "query_checkpoint_count": len(row.get("query_checkpoints") or [])})
    result = {"schema_version": RESULT_SCHEMA, "candidate_id": authority["candidate_ids"][0],
        **{key: admission[key] for key in ("plan_digest", "configuration_digest", "admission_digest", "runtime_identity_digest",
                                         "source", "scenario_plan_digest")},
        "task_success_contract_digest": authority["task_success_contract_digest"],
        "authority_digest": authority["authority_digest"], "runtime_inputs_digest": authority["runtime_inputs_digest"],
        "status": "completed" if all(row["status"] == "completed" for row in episodes) else "blocked",
        "episodes": wire_episodes, "stage_reached": stage, "blockers": blockers,
        "candidate_policy_queried": any(row.get("query_checkpoints") for row in episodes),
        "claim_ceiling": "development_only", "ranking_eligible": False, "qualification_permitted": False,
        "public_media_publication_authorized": False, "model_redistribution_authorized": False,
        "physical_outcome_claimed": False}
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


def _empty_cell(cell, reason="g1_beta_budget_not_attempted"):
    return {**{key: cell[key] for key in ("cell_id", "seed", "cell_spec_digest", "resolved_scenario", "resolved_scenario_digest")},
        "scene_plan_digest": None, "status": "not_attempted", "query_checkpoints": [],
        "failure_stage": "not_started", "reason": reason, "artifact_records": []}


def _bound_json(path, *, root, field, schema=None):
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("g1_beta_output_artifact_invalid")
    if path.stat().st_size > 128 * 1024**2:
        raise ValueError("g1_beta_output_json_bound_exceeded")
    value = _json(path)
    if value.get(field) != canonical_digest(value, digest_field=field) or (schema and value.get("schema_version") != schema):
        raise ValueError("g1_beta_output_artifact_digest_invalid")
    return value, {"relative_path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": _sha(path)}


def collect_cell(*, root: Path, cell_root: Path, cell, scene_plan_digest: str,
                 candidate_id: str, request_digest: str, returncode: int | None, reason: str | None):
    """Retain valid responses even when the worker never publishes its final score."""
    row = _empty_cell(cell)
    row.update(scene_plan_digest=scene_plan_digest, status="blocked", failure_stage="native_worker",
               reason=reason or "g1_beta_native_worker_incomplete")
    worker_path = cell_root / WORKER_RESULT
    if not worker_path.exists() and returncode is not None:
        from .native_g1_development_pair import _recover_worker_result_from_preclose
        _recover_worker_result_from_preclose(path=cell_root / "native_g1_development_worker_preclose.v1.json",
            result_path=worker_path, candidate_id=candidate_id, scene_plan_digest=scene_plan_digest,
            request_digest=request_digest, returncode=returncode)
    if worker_path.is_file():
        worker, record = _bound_json(worker_path, root=root, field="result_digest", schema="native_g1_development_worker_result.v1")
        if (worker.get("candidate_id") != candidate_id or worker.get("request_digest") != request_digest
                or worker.get("scene_plan_digest") not in {scene_plan_digest, None}):
            raise ValueError("g1_beta_worker_result_binding_invalid")
        row["worker_result"] = worker
        row["artifact_records"].append(record)
        row["failure_stage"] = str(worker.get("phase_reached") or row["failure_stage"])[:80]
        message = (worker.get("blocker") or {}).get("message")
        if reason is None and isinstance(message, str) and re.fullmatch(r"g1_[a-z0-9_]{1,124}", message):
            row["reason"] = message
        if worker.get("status") == "completed_development_only":
            row.update(status="completed", failure_stage=None, reason=None)
    preclose_path = cell_root / "native_g1_development_worker_preclose.v1.json"
    if preclose_path.is_file():
        value, record = _bound_json(preclose_path, root=root, field="preclose_digest", schema="native_g1_development_worker_preclose.v1")
        row["worker_preclose"] = value
        row["artifact_records"].append(record)
    checkpoint_root = cell_root / "episode/query_checkpoints"
    if checkpoint_root.exists():
        for index, path in enumerate(sorted(checkpoint_root.iterdir())):
            if index >= 3000 or path.name != f"query_{index:04d}.v1.json":
                raise ValueError("g1_beta_query_checkpoint_closure_invalid")
            value, record = _bound_json(path, root=root, field="checkpoint_digest", schema="native_g1_policy_query_checkpoint.v1")
            if (value.get("candidate_id") != candidate_id or value.get("scene_plan_digest") != scene_plan_digest
                    or value.get("query", {}).get("query_index") != index):
                raise ValueError("g1_beta_query_checkpoint_binding_invalid")
            row["query_checkpoints"].append(value)
            row["artifact_records"].append(record)
    for filename, key, field in ((EPISODE, "episode", "result_digest"), (TRACE, "trace", "trace_digest")):
        path = cell_root / "episode" / filename
        if path.is_file():
            value, record = _bound_json(path, root=root, field=field)
            if value.get("candidate_id") != candidate_id or value.get("scene_plan_digest") != scene_plan_digest:
                raise ValueError("g1_beta_episode_artifact_binding_invalid")
            row[key] = value
            row["artifact_records"].append(record)
    if row["status"] == "completed" and not {"episode", "trace", "worker_result"} <= set(row):
        row.update(status="blocked", failure_stage="evidence_collection", reason="g1_beta_completed_evidence_incomplete")
    return row


def _retained_bytes(root):
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("g1_beta_output_symlink_forbidden")
        if path.is_file() and not path.relative_to(root).as_posix().startswith((
                "models/checkpoints/", "models/sonic/", "models/torch/", "policy-runtime-build/policy-runtime/", "policy-runtime-build/wheel-downloads/", "cell_packets/")):
            total += path.stat().st_size
    return total


def _run_worker(request_path, cell_root, *, deadline, output):
    log_path = cell_root.parent / (cell_root.name + ".worker.log")
    argv = ["/isaac-sim/python.sh", "-m", "blueprint_pipeline.native_g1_development_worker",
            "--request", str(request_path), "--output-dir", str(cell_root)]
    with log_path.open("xb") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        reason = None
        try:
            while process.poll() is None:
                if time.time() >= deadline:
                    reason = "g1_beta_native_worker_deadline"
                    break
                if _retained_bytes(output) > MAX_OUTPUT_BYTES - OUTPUT_MARGIN:
                    reason = "g1_beta_output_byte_budget_exhausted"
                    break
                try:
                    process.wait(timeout=min(5, remaining_seconds(deadline)))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
    return process.returncode, reason


def _build_runtime(root, output, deadline):
    # The dedicated dependency builder is supplied by the exact offline inventory
    # integration. It must not invoke the legacy online pip/apt bootstrap.
    from .native_g1_beta_dependencies import build_policy_runtime
    return build_policy_runtime(root=root, output=output / "policy-runtime-build", deadline=deadline)



def _recover_cells(root, output, authority, spec, base, episodes):
    """Read prior child evidence after a wrapper crash; never launch or retry."""
    from .native_g1_beta_packet import verify_cell_packet
    for index, cell in enumerate(authority["cells"]):
        request_path = output / "cells" / (cell["cell_id"] + ".request.json")
        if not request_path.exists():
            continue
        request = _json(request_path)
        derivation = verify_cell_packet(base_packet=base, cell_packet=output / "cell_packets" / cell["cell_id"],
                                       cell=cell, base_authority=spec["rights_authority"])
        if (request.get("request_digest") != canonical_digest(request, digest_field="request_digest")
                or request.get("candidate_id") != authority["candidate_ids"][0]
                or request.get("rights_review") != derivation["rights_authority"]):
            raise ValueError("g1_beta_recovery_request_changed")
        episodes[index] = collect_cell(root=output, cell_root=output / "cells" / cell["cell_id"], cell=cell,
            scene_plan_digest=derivation["scene_plan_digest"], candidate_id=authority["candidate_ids"][0],
            request_digest=request["request_digest"], returncode=None, reason="g1_beta_wrapper_interrupted")


def run_session(root: Path, output: Path, *, failure_only: str | None = None):
    from .native_g1_beta_download import stage_selected_assets
    from .native_g1_beta_assets import needs_pi_tokenizer
    from .native_g1_pi_tokenizer_assets import stage_provider_tokenizer
    from .native_g1_sonic_cuda_runtime import preflight_sonic_cuda_models
    authority, spec, base = _inputs(root)
    episodes = [_empty_cell(cell) for cell in authority["cells"]]
    blockers, stage = [], "input_verification"
    if (not output.is_absolute() or output.is_symlink() or not output.is_dir()
            or (output / RESULT_FILENAME).exists()):
        raise ValueError("g1_beta_runtime_output_invalid")
    try:
        if failure_only:
            _recover_cells(root, output, authority, spec, base, episodes)
            raise ValueError(failure_only)
        deadline = float(os.environ[DEADLINE_ENV])
        remaining_seconds(deadline)
        stage = "publisher_source"
        if shutil.which("git") is None:
            raise ValueError("g1_beta_image_git_unavailable")
        _verify_relocated_publisher(root)
        stage = "native_provisioning"
        provision = _json(output / "native_task_runtime_source_provisioning.v1.json")
        source = _json(root / "native_task_runtime_sources/native_task_runtime_source_packet.v1.json")
        if (provision.get("status") != "completed" or provision.get("runtime_profile") != "unitree_g1"
                or provision.get("source_packet_sha256") != source["packet_sha256"]):
            raise ValueError("g1_beta_runtime_source_not_provisioned")
        stage = "runtime_imports"
        if _preflight_g1_runtime_imports(output)["status"] != "passed":
            raise ValueError("g1_beta_runtime_import_failure")
        stage = "policy_dependencies"
        runtime_python = Path(_build_runtime(root, output, deadline - 120)["python_executable"])
        if needs_pi_tokenizer(authority["candidate_ids"]):
            stage = "pi_tokenizer"
            tokenizer = stage_provider_tokenizer(inventory_path=root.parent / "configs/g1_paligemma_tokenizer_inventory.v1.json",
                                               bundled_dir=root / "inputs/pi_tokenizer")
            _write(output / "native_g1_pi_tokenizer_stage.v1.json", tokenizer)
        stage = "selected_model_staging"
        models = stage_selected_assets(root=root, output=output, candidate_ids=authority["candidate_ids"], deadline=deadline - 120)
        _write(output / "models/sonic_cuda_preflight.json", preflight_sonic_cuda_models(models["sonic"]))
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                          TORCH_HOME=str(output / "models/torch"))
        for index, cell in enumerate(authority["cells"]):
            stage = "cell_packet"
            if deadline - time.time() <= 180 or _retained_bytes(output) > MAX_OUTPUT_BYTES - OUTPUT_MARGIN:
                break
            packet_root = output / "cell_packets" / cell["cell_id"]
            derivation = materialize_cell_packet(base_packet=base, output_dir=packet_root,
                cell=cell, base_authority=spec["rights_authority"])
            _write(output / "cells" / (cell["cell_id"] + ".packet.json"), derivation)
            template = _json(_template(root=root, output=output, packet=packet_root, objective_name="manipulation",
                models=models, runtime_python=runtime_python, provisioning_path=output / "native_task_runtime_source_provisioning.v1.json"))
            request = {**template, "candidate_id": authority["candidate_ids"][0], "rights_review": derivation["rights_authority"]}
            if "navigation_authority" in spec["native_g1"]:
                # A scene-bound navigation authority needs its own exact derived
                # goal binding; never transplant the base approval silently.
                from .native_g1_beta_navigation import derive_navigation_authority
                request["navigation_goal_authority"] = derive_navigation_authority(
                    base_authority=_json(root / "inputs/navigation_authority.json"),
                    base_plan=_json(base / PLAN), base_contract=_json(base / CONTRACT), derived_plan=_json(packet_root / PLAN), cell_authority=derivation["rights_authority"])
            request["request_digest"] = canonical_digest(request, digest_field="request_digest")
            request_path = output / "cells" / (cell["cell_id"] + ".request.json")
            _write(request_path, request)
            stage = "native_worker"
            cell_root = output / "cells" / cell["cell_id"]
            returncode, reason = None, None
            try:
                returncode, reason = _run_worker(request_path, cell_root,
                    deadline=min(deadline - 120, time.time() + 55 * 60), output=output)
            finally:
                episodes[index] = collect_cell(root=output, cell_root=cell_root, cell=cell,
                    scene_plan_digest=derivation["scene_plan_digest"], candidate_id=authority["candidate_ids"][0],
                    request_digest=request["request_digest"], returncode=returncode, reason=reason)
            if reason or episodes[index]["status"] != "completed":
                for pending in episodes[index + 1:]:
                    pending.update(failure_stage="prior_native_cell", reason="g1_beta_prior_native_cell_blocked")
                break
    except Exception as exc:  # retain every not-attempted cell on bounded bootstrap failure
        # Preserve a bounded machine reason through the private projection;
        # arbitrary exception text may contain URLs or runtime details.
        message = str(exc)
        blockers.append(message if re.fullmatch(r"g1_[a-z0-9_]{1,120}", message)
                        else "g1_beta_" + stage + "_failed")
        for row in episodes:
            if row["status"] == "not_attempted":
                row.update(failure_stage=stage, reason=blockers[-1])
    result = _result(authority, episodes, stage=stage, blockers=blockers)
    _write(output / RESULT_FILENAME, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--failure-only")
    args = parser.parse_args(argv)
    result = run_session(args.runtime_root, args.output_dir, failure_only=args.failure_only)
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
