"""ADP-050/day 28: production queue to the bounded native allocator.

An operator-owned registry binds assets and authority to an exact capture,
task, scenario and team. Customer JSON cannot choose files or launch commands.
"""
from __future__ import annotations
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .controlled_native_policy_bundle import build_controlled_native_policy_bundle, PROBE_KIND
from .controlled_policy_outcome import validate_controlled_outcome
from .controlled_policy_configuration import canonical_request_digest
from .common import utc_now_iso

REGISTRY_ENV = "BLUEPRINT_CONTROLLED_NATIVE_REGISTRY"
TERMINAL_FILENAME = "controlled_native_terminal.json"


def _read(path: Path) -> dict[str, Any]:
    row = json.loads(path.read_text())
    if not isinstance(row, dict):
        raise ValueError("controlled_native_record_invalid")
    return row


def _write(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x") as stream:
        path.chmod(0o600)
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False))
        stream.flush()
        os.fsync(stream.fileno())


def configured_profile(request: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = os.environ.get(REGISTRY_ENV)
    if not raw:
        return None
    path = Path(raw)
    stat = path.stat()
    if not path.is_absolute() or path.is_symlink() or stat.st_mode & 0o022 or stat.st_uid not in {0, os.getuid()}:
        raise ValueError("controlled_native_registry_not_operator_owned")
    registry = _read(path)
    if (registry.get("schema_version") != "blueprint.controlled_native_registry.v1"
            or registry.get("registry_digest") != canonical_digest(registry, digest_field="registry_digest")):
        raise ValueError("controlled_native_registry_binding_invalid")
    tasks = request.get("requested_tasks") or []
    if len(tasks) != 1 or len(tasks[0].get("scenario_ids") or []) != 1:
        return None
    matches = [row for row in registry.get("profiles", []) if row.get("capture_root") == request.get("capture_root")
        and row.get("task_id") == tasks[0].get("task_id")
        and row.get("scenario_id") == tasks[0]["scenario_ids"][0]
        and request.get("customer", {}).get("id") in row.get("allowed_team_ids", [])
        and request.get("robot_profile", {}).get("robot_profile_id") in row.get("allowed_checkpoint_ids", [])]
    if len(matches) > 1:
        raise ValueError("controlled_native_profile_ambiguous")
    return matches[0] if matches else None


def routes_controlled_request(request: Mapping[str, Any]) -> bool:
    package = request.get("policy_package") or {}
    return any(isinstance(package.get(name), Mapping)
               and package[name].get("execution_profile") == "controlled_observation_v1"
               for name in ("policy_api_endpoint", "docker_container", "sim_controller_plugin"))


def _execute_staged_controlled_request(*, request: Mapping[str, Any], job_dir: Path) -> None:
    profile = configured_profile(request)
    if profile is None:
        raise ValueError("controlled_native_task_profile_required")
    modalities = [name for name in ("policy_api_endpoint", "docker_container", "sim_controller_plugin")
                  if request.get("policy_package", {}).get(name)]
    if len(modalities) != 1 or modalities[0] not in profile.get("allowed_modalities", []):
        raise ValueError("controlled_native_policy_modality_not_configured")
    job_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if (job_dir / TERMINAL_FILENAME).exists():
        return
    with (job_dir / "controlled_native_execution.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if (job_dir / "controlled_native_execution_intent.json").exists():
            # Recover sealed output, but never allocate twice after a caller crash.
            if (job_dir / "native_allocator_result.json").exists():
                intent = _read(job_dir / "controlled_native_execution_intent.json")
                seal_native_terminal(request=request, job_dir=job_dir, source_commit=intent["source_commit"])
                return
            raise ValueError("controlled_native_launch_interrupted_or_failed_without_terminal")

        from .adp_task_evaluation_abstention import collect_vast_provider_zero_receipt
        from .native_task_arena_paid_authority import materialize_native_task_arena_paid_attempt_authority
        config = _read(Path(profile["configuration_path"]))
        if modalities[0] == "policy_api_endpoint":
            from urllib.parse import urlsplit
            from .controlled_policy_session import customer_hosted_client
            endpoint = request["policy_package"]["policy_api_endpoint"]["endpoint_url"]
            customer_hosted_client(endpoint=endpoint, allowed_origins=tuple(config["allowed_origins"]),
                contract=config["contract"])
            parsed = urlsplit(endpoint)
            if not parsed.path:
                raise ValueError("controlled_native_policy_action_route_required")
        authorization = request.get("execution_authorization") or {}
        if authorization.get("episodes") != 1 or float(authorization.get("max_cost_usd", 0)) < profile["hard_cap_usd"]:
            raise ValueError("controlled_native_execution_budget_or_episode_scope_invalid")
        code_root = Path(__file__).resolve().parents[2]
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code_root, text=True).strip()
        _write(job_dir / "controlled_native_execution_intent.json", {"job_id": request["job_id"],
            "canonical_request_digest": canonical_request_digest(request), "observed_at_iso": utc_now_iso(),
            "source_commit": commit})
        observation = {"observation_id": request["job_id"] + "-native-0", "task_id": profile["task_id"],
            "scenario_id": profile["scenario_id"], "scenario_eval_run_id": request["job_id"] + "-native-0"}
        credential = None
        credential_path = profile.get("policy_credential_path")
        if credential_path:
            credential_file = Path(credential_path)
            if credential_file.is_symlink() or credential_file.stat().st_mode & 0o077:
                raise ValueError("controlled_native_policy_credential_not_private")
            credential = {**_read(credential_file), "job_id": request["job_id"]}
        bridge = None
        bridge_path = profile.get("qualified_sandbox_bridge_path")
        if modalities[0] != "policy_api_endpoint":
            if not bridge_path:
                raise ValueError("controlled_native_qualified_sandbox_not_configured")
            bridge_file = Path(bridge_path)
            if (not bridge_file.is_absolute() or bridge_file.is_symlink()
                    or bridge_file.stat().st_mode & 0o077):
                raise ValueError("controlled_native_qualified_sandbox_bridge_not_private")
            from .controlled_policy_remote_sandbox import RemoteQualifiedSandboxFactory, validate_remote_sandbox_bridge
            from .company_policy_container_contract_v2 import validate_company_policy_container_contract_v2
            bridge = validate_remote_sandbox_bridge(_read(bridge_file))
            if (bridge["job_id"] != request["job_id"]
                    or bridge["canonical_request_digest"] != canonical_request_digest(request)):
                raise ValueError("controlled_native_bridge_request_binding_mismatch")
            offered_contract = dict(config["contract"])
            offered_contract.pop("contract_digest", None)
            container = dict(offered_contract["container"])
            container["image"] = bridge["image_ref"]
            payload = request["policy_package"][modalities[0]]
            if payload.get("model_artifact") is not None:
                container.update({"serve_command": ["python", "-m", "blueprint_pipeline.policy_model_server"],
                                  "port": 8600, "run_as_uid": 65532, "run_as_gid": 65532,
                                  "gpu_required": False})
            offered_contract["container"] = container
            expected_contract = validate_company_policy_container_contract_v2(offered_contract)
            RemoteQualifiedSandboxFactory(bridge).preflight(contract=expected_contract, job_request=request)
        bundle = build_controlled_native_policy_bundle(job_dir=job_dir / "native_bundle",
            packet_dir=Path(profile["packet_dir"]), runtime_source_packet_receipt=Path(profile["runtime_source_packet_receipt"]),
            implementation_commit=commit, configuration=config, job_request=request, observations=[observation],
            policy_credential=credential, qualified_sandbox_bridge=bridge)
        zero = collect_vast_provider_zero_receipt()
        zero_path = job_dir / "initial_provider_zero.json"
        _write(zero_path, zero)
        authority_path = job_dir / "native_authority.json"
        materialize_native_task_arena_paid_attempt_authority(
            bundle_receipt_path=job_dir / "native_bundle/native_task_arena_provider_bundle_receipt.v1.json",
            project_spend_reconciliation_path=profile["project_spend_reconciliation_path"],
            initial_provider_zero_path=zero_path, authorization_reference=profile["authorization_reference"],
            authorized_by=profile["authorized_by"], authorized_on=utc_now_iso(), blueprint_commit=commit,
            max_hourly_rate_usd=profile["max_hourly_rate_usd"], hard_cap_usd=profile["hard_cap_usd"],
            hard_ttl_seconds=profile["hard_ttl_seconds"], output_path=authority_path)
        result_path = job_dir / "native_allocator_result.json"
        argv = [sys.executable, "-m", "blueprint_pipeline.paid_resource_allocator", "gpu-canary",
            "--provider", "vast", "--probe-kind", PROBE_KIND, "--execute",
            "--adp-job-dir", str(job_dir / "arena-controlled-policy-job"),
            "--native-task-arena-packet", profile["packet_dir"],
            "--native-task-arena-runtime-source-packet", profile["runtime_source_packet_receipt"],
            "--native-task-arena-bundle-receipt", bundle["bundle_path"].removesuffix(".zip") + "_receipt.v1.json",
            "--native-task-arena-attempt-authority", str(authority_path),
            "--adp-max-hourly-rate-usd", str(profile["max_hourly_rate_usd"]),
            "--adp-max-spend-usd", str(profile["hard_cap_usd"]), "--adp-hard-ttl-seconds", str(profile["hard_ttl_seconds"]),
            "--adp-machine-avoidlist", profile["machine_avoidlist_path"],
            "--admission-out", str(job_dir / "native_paid_admission.json"), "--adapter-output", str(result_path)]
        with (job_dir / "native_allocator.log").open("x") as stream:
            stream_path = job_dir / "native_allocator.log"
            stream_path.chmod(0o600)
            subprocess.run(argv, cwd=code_root, stdout=stream, stderr=subprocess.STDOUT,
                timeout=profile["hard_ttl_seconds"] + 900, check=False)
        seal_native_terminal(request=request, job_dir=job_dir, source_commit=commit)


def execute_staged_controlled_request(*, request: Mapping[str, Any], job_dir: Path) -> None:
    """A failed attempt releases its hold only after provider-zero is observed."""
    job_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        _execute_staged_controlled_request(request=request, job_dir=job_dir)
    except Exception as exc:
        from .adp_task_evaluation_abstention import collect_vast_provider_zero_receipt
        failure = job_dir / "controlled_native_failure.json"
        if not failure.exists():
            _write(failure, {"schema_version": "blueprint.controlled_native_failure.v1",
                "job_id": request["job_id"], "error_class": type(exc).__name__,
                "blockers": [str(exc)[:300]], "observed_at_iso": utc_now_iso()})
        try:
            zero = collect_vast_provider_zero_receipt()
        except Exception:
            return
        if zero.get("provider_zero") is not True or (job_dir / TERMINAL_FILENAME).exists():
            return
        if not (job_dir / "failed_provider_zero.json").exists():
            _write(job_dir / "failed_provider_zero.json", zero)
        terminal = {"schema_version": "blueprint.controlled_native_terminal.v1", "status": "blocked",
            "job_id": request["job_id"], "canonical_request_digest": canonical_request_digest(request),
            "blockers": _read(failure)["blockers"], "provider_zero_verified": True}
        terminal["terminal_digest"] = canonical_digest(terminal, digest_field="terminal_digest")
        _write(job_dir / TERMINAL_FILENAME, terminal)


def seal_native_terminal(*, request: Mapping[str, Any], job_dir: Path, source_commit: str) -> None:
    adapter = _read(job_dir / "native_allocator_result.json")
    if adapter.get("status") != "completed" or adapter.get("blockers"):
        raise ValueError("controlled_native_allocator_not_completed")
    native = _read(Path(adapter["native_control_result_path"]))
    if (native.get("result_digest") != canonical_digest(native, digest_field="result_digest")
            or native["result_digest"] != adapter.get("native_control_result_digest")
            or native.get("canonical_request_digest") != canonical_request_digest(request)
            or native.get("status") != "completed" or native.get("job_id") != request["job_id"]
            or len(native.get("attempts", [])) != 1):
        raise ValueError("controlled_native_terminal_request_binding_mismatch")
    episode = native["attempts"][0]["metrics"]
    outcome = validate_controlled_outcome(episode["independent_outcome"], executed_motor_steps=episode["executed_motor_steps"])
    from .adp_task_evaluation_abstention import collect_vast_provider_zero_receipt
    zero = collect_vast_provider_zero_receipt()
    if zero.get("provider_zero") is not True or adapter.get("continuing_spend_from_this_run") is not False:
        raise ValueError("controlled_native_provider_zero_not_verified")
    _write(job_dir / "completed_provider_zero.json", zero)
    summary = {"schema_version": "blueprint.controlled_native_private_result.v1", "evidence_scope": "development_only",
        "native_simulator": "isaac", "source_commit": source_commit, "execution_receipt_digest": native["result_digest"],
        "outcome_receipt_digest": outcome["receipt_digest"], "task_spec_digest": outcome["task_spec_digest"],
        "samples_digest": outcome["samples_digest"], "policy_queries": episode["policy_queries"],
        "executed_motor_steps": episode["executed_motor_steps"], "task_success": outcome["task_success"],
        "outcome": outcome["score"]["outcome"], "scene_files_exported": False, "scoring_harness_exported": False,
        "physical_success_proven": False, "qualification_eligible": False, "provider_zero_verified": True,
        "provider_cost_status": "pending_official_reconciliation", "observed_provider_cost_usd": None}
    terminal = {"schema_version": "blueprint.controlled_native_terminal.v1", "status": "completed",
        "job_id": request["job_id"], "canonical_request_digest": canonical_request_digest(request),
        "episodes_run": 1, "episodes_succeeded": int(outcome["task_success"]),
        "note": "Native simulator execution and independent development-task outcome; private to the submitting team",
        "artifact_uri": None, "private_execution_result": summary}
    terminal["terminal_digest"] = canonical_digest(terminal, digest_field="terminal_digest")
    _write(job_dir / TERMINAL_FILENAME, terminal)


def read_native_terminal(*, job_dir: Path, expected_job_id: str, expected_canonical_request_digest: str):
    path = job_dir / TERMINAL_FILENAME
    if not path.exists():
        return None
    row = _read(path)
    if (row.get("schema_version") != "blueprint.controlled_native_terminal.v1"
            or row.get("terminal_digest") != canonical_digest(row, digest_field="terminal_digest")
            or row.get("job_id") != expected_job_id or row.get("canonical_request_digest") != expected_canonical_request_digest):
        raise ValueError("controlled_native_delivery_binding_mismatch")
    return row
