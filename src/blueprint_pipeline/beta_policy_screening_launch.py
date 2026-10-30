"""Execute beta job-evaluation candidates through the canonical canary allocator.

Beta L5: preparation seals the existing scene and policy runtime, then one
deterministic attempt can start at most one canonical allocator. A missing
process receipt is an unresolved launch, never permission to create again.
The coordinator owns the site's budget reservation before calling launch.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_policy_canary_session import (
    BETA_ADMISSION_SCHEMA_VERSION, BETA_RUN_KIND, CLAIM_CEILING, PROBE_KIND,
    build_session_authority, validate_runtime_input_manifest, validate_session_result,
)
from .team_candidate_quick10_staging import SCENE_INPUT_FIELDS

EXECUTION_SCHEMA = "blueprint.beta_policy_screening_execution.v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("beta_screening_record_invalid")
    return value


def _record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "size_bytes": len(data),
            "sha256": "sha256:" + hashlib.sha256(data).hexdigest()}


def _bound_path(record: Mapping[str, Any]) -> Path:
    path = Path(str(record.get("path") or ""))
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or _record(path)["sha256"] != record.get("sha256")
            or ("size_bytes" in record and path.stat().st_size != record["size_bytes"])):
        raise ValueError("beta_screening_input_bytes_changed")
    return path


def _write(path: Path, value: Mapping[str, Any]) -> Path:
    """Durable exclusive publish: a crash never exposes partial JSON as intent."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # atomic no-replace; preserves launch-intent CAS
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _stage(path: Path, value: Mapping[str, Any]) -> Path:
    """Resume deterministic, unpaid staging without changing sealed inputs."""
    if path.exists():
        if _read(path) != value:
            raise ValueError("beta_screening_staged_input_changed")
        return path
    return _write(path, value)


def _identity(plan: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    from .beta_policy_screening_plan import validate_screening_plan
    validate_screening_plan(plan)
    if plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest"):
        raise ValueError("beta_screening_plan_digest_invalid")
    matches = [row for row in plan.get("candidates", []) if row.get("candidate_id") == candidate.get("candidate_id")]
    if len(matches) != 1 or matches[0] != candidate:
        raise ValueError("beta_screening_candidate_not_in_plan")
    return {"plan_digest": plan["plan_digest"], "candidate_id": candidate["candidate_id"],
            "configuration_digest": candidate["configuration_digest"]}


def _check_quote(plan: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    quote = candidate["cost_quote"]
    expires = datetime.fromisoformat(str(quote["valid_until"]).replace("Z", "+00:00"))
    cap = Decimal(str(candidate["max_cost_usd"]))
    execution = candidate["execution_spec"]
    rate = Decimal(str(execution["max_hourly_rate_usd"]))
    ttl = execution["hard_ttl_seconds"]
    if (expires.tzinfo is None or expires <= datetime.now(timezone.utc)
            or not cap.is_finite() or not Decimal("0") < cap <= Decimal(str(plan["budget_usd"])) <= Decimal("15")
            or not rate.is_finite() or rate <= 0 or type(ttl) is not int or ttl < 1800
            or rate * Decimal(ttl) / Decimal(3600) > cap
            or quote.get("quote_digest") != canonical_digest(quote, digest_field="quote_digest")
            or quote.get("configuration_digest") != candidate["configuration_digest"]
            or quote.get("scene_revision_digest") != plan["scene_revision_digest"]
            or Decimal(str(quote["maximum_cost_usd"])) != cap
            or quote.get("episode_count") != candidate["episode_count"]):
        raise ValueError("beta_screening_cost_quote_invalid_or_expired")


def prepare_candidate(*, plan: Mapping[str, Any], candidate: Mapping[str, Any],
                      attempt_dir: Path) -> dict[str, Any]:
    """Build and static-preflight a sealed provider bundle; never allocate."""
    if candidate.get("execution_spec", {}).get("backend") == "native_g1.v1":
        from .beta_policy_screening_g1_launch import prepare_candidate as prepare_g1
        return prepare_g1(plan=plan, candidate=candidate, attempt_dir=attempt_dir)
    from .beta_robolab_policy_execution import validate_robolab_policy_execution_spec
    from .native_task_arena_policy_canary_bundle import (
        build_policy_canary_session_bundle, preflight_sealed_policy_canary_bundle,
    )
    from .task_family_policy_execution import FAMILY_BACKEND, validate_task_family_policy_execution_spec
    root = Path(attempt_dir).resolve()
    identity = _identity(plan, candidate)
    _check_quote(plan, candidate)
    family = candidate["execution_spec"].get("backend") == FAMILY_BACKEND
    previous = root / "prepared.json"
    if previous.exists():
        prepared = _read(previous)
        _validate_prepared(plan, candidate, root, prepared)
        return prepared
    execution = candidate["execution_spec"]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True, check=True).stdout.strip()
    if commit != execution.get("implementation_commit"):
        raise ValueError("beta_screening_implementation_commit_changed")
    if (execution.get("schema_version") != EXECUTION_SCHEMA
            or execution.get("backend") not in {"robolab.v1", FAMILY_BACKEND}
            or execution.get("candidate_id") != candidate["candidate_id"]
            or execution.get("configuration_digest") != candidate["configuration_digest"]
            or candidate.get("admission_digest") != canonical_digest(execution)):
        raise ValueError("beta_screening_execution_binding_invalid")
    scene = _read(_bound_path(plan["scene_inputs"]))
    if (scene["scene_revision_digest"] != plan["scene_revision_digest"]
            or scene["task_success_contract_digest"] != execution["task_success_contract_digest"]):
        raise ValueError("beta_screening_scene_binding_invalid")
    if family:
        from .beta_policy_screening_family import reopen_family_planning_inputs
        static = reopen_family_planning_inputs(scene)
        if (plan.get("pre_policy_site_qualification_required") is not True
                or plan.get("family_static_readiness_digest") != scene["family_static_readiness_digest"]
                or plan.get("evaluation_backend") != FAMILY_BACKEND
                or plan.get("scenario_scope") != "native_frozen_reset_seeds"):
            raise ValueError("beta_screening_family_plan_binding_invalid")
    else:
        from .robot_free_probe_gate import require_robot_free_probe, task_success_contract_binding
        probe = require_robot_free_probe(scene.get("beta_robot_free_probe_receipt"),
            scene_revision_digest=scene["scene_revision_digest"], task_kind=scene.get("beta_task_kind"),
            task_spec_digest=scene.get("beta_probe_task_spec_digest"),
            task_success_contract_digest=task_success_contract_binding(scene.get("beta_task_kind"), scene["task_success_contract"]))
        if probe["receipt_digest"] != plan["robot_free_probe_digest"]:
            raise ValueError("beta_screening_robot_free_probe_changed")
    for key in (("base_native_packet", "runtime_source") if family else ("base_native_packet", "runtime_source", "construction_result")):
        _bound_path(scene[key])
    raw_spec = _read(_bound_path(execution["policy_execution_spec"]))
    native_spec = (validate_task_family_policy_execution_spec(raw_spec,
        candidate_id=candidate["candidate_id"], scene_plan=static["scene_plan"]) if family else
        validate_robolab_policy_execution_spec(raw_spec, candidate_id=candidate["candidate_id"]))
    if family:
        from .beta_policy_screening_family import validate_family_candidate_catalog
        validate_family_candidate_catalog(plan, candidate, native_spec, static, scene=scene)
    from .policy_interface_binding import TEAM_HTTPS_INTERFACES, TEAM_ROBOT_INTERFACES, TEAM_LOCAL_INTERFACES
    if native_spec.get("policy_interface_id") in TEAM_HTTPS_INTERFACES:
        from .beta_policy_screening_hosted import validate_hosted_delivery
        validate_hosted_delivery(native_spec, execution, episodes=candidate["episode_count"])
    elif native_spec.get("policy_interface_id") in TEAM_LOCAL_INTERFACES:
        from .team_local_policy import runtime_for_interface
        runtime_for_interface(native_spec["policy_interface_id"]).validate_execution_spec(native_spec)
    elif native_spec.get("policy_interface_id") == "blueprint_scripted_pick_place.v1":
        from .scripted_robot_baseline_runtime import validate_execution_spec
        validate_execution_spec(native_spec)
    elif native_spec.get("policy_interface_id") == "molmoact2_droid_joint8.v1":
        from .molmoact2_droid_runtime import validate_execution_spec
        validate_execution_spec(native_spec)
    elif native_spec.get("candidate_id") not in {"pi05_droid", "groot_n17_droid", "cosmos3_edge_policy_droid", "cosmos3_nano_policy_droid"}:
        raise ValueError("beta_screening_policy_provisioning_not_configured")
    if (native_spec["runtime_identity_digest"] != execution["runtime_digest"]
            or native_spec["task_success_contract_digest"] != scene["task_success_contract_digest"]):
        raise ValueError("beta_screening_policy_scene_binding_invalid")
    if native_spec.get("policy_interface_id") in TEAM_ROBOT_INTERFACES:
        from .beta_policy_screening_team import validate_team_candidate_catalog
        validate_team_candidate_catalog(plan, candidate, native_spec, scene)
    control = ({"mode": "native_family_prepolicy_required", "typed_gap": "native_site_qualification_pending",
                "policy_execution_blocked": True} if family else dict(scene["control_diagnostic"]))
    cells = [{**dict(cell), "control_diagnostic": control}
             for cell in candidate["scenario_cells"]]
    if len(cells) != candidate["episode_count"]:
        raise ValueError("beta_screening_episode_count_invalid")
    run_id = "beta-" + canonical_digest({**identity, "attempt_dir": str(root)})[7:39]
    binding = {"schema_version": BETA_ADMISSION_SCHEMA_VERSION, **identity,
               "admission_digest": candidate["admission_digest"], "evaluation_purpose": "screening",
               "backend": execution["backend"], "scene_revision_digest": plan["scene_revision_digest"],
               "task_success_contract_digest": scene["task_success_contract_digest"],
               **({"pre_policy_site_qualification_required": True,
                   "family_static_readiness_digest": scene["family_static_readiness_digest"]} if family else
                  {"robot_free_probe_receipt_digest": plan["robot_free_probe_digest"]})}
    activation = {"schema_version": "task_evaluation_policy_campaign_activation.v1", "run_id": run_id,
                  "run_kind": BETA_RUN_KIND, "claim_ceiling": CLAIM_CEILING,
                  "candidate_ids": [candidate["candidate_id"]],
                  "task_success_contract": scene["task_success_contract"],
                  "task_success_contract_digest": scene["task_success_contract_digest"],
                  "campaign_unit_count": len(cells), "campaign_units": cells, "activation_digest": ""}
    execution_kinds = ({candidate["candidate_id"]: native_spec["execution_evidence_kind"]}
                       if native_spec.get("policy_interface_id") == "blueprint_scripted_pick_place.v1" else None)
    if execution_kinds is not None:
        activation["candidate_execution_evidence_kinds"] = execution_kinds
    activation["activation_digest"] = canonical_digest(activation, digest_field="activation_digest")
    inputs = {**{key: scene[key] for key in SCENE_INPUT_FIELDS if key in scene},
              "schema_version": "task_evaluation_policy_canary_runtime_inputs.v1", "run_id": run_id,
              "run_kind": BETA_RUN_KIND, "claim_ceiling": CLAIM_CEILING,
              "candidate_ids": [candidate["candidate_id"]], "cells": cells,
              "configuration_digest": candidate["configuration_digest"], "plan_digest": plan["plan_digest"],
              "matrix_digest": plan["scenario_plan_digest"], "beta_screening_admission": binding,
              **({"pre_policy_site_qualification_required": True,
                  "family_static_readiness_digest": scene["family_static_readiness_digest"]} if family else {}),
              "activation_digest": activation["activation_digest"], "runtime_inputs_digest": ""}
    if execution_kinds is not None:
        inputs["candidate_execution_evidence_kinds"] = execution_kinds
    if family:
        inputs.pop("construction_result", None)
    inputs["runtime_inputs_digest"] = canonical_digest(inputs, digest_field="runtime_inputs_digest")
    inputs = validate_runtime_input_manifest(inputs)
    directory = root / "runtime_inputs"
    activation_path = _stage(directory / "activation.json", activation)
    input_path = _stage(directory / "policy_canary_runtime_inputs.json", inputs)
    authority = build_session_authority(activation_manifest=activation, activation_record=_record(activation_path),
        runtime_inputs=inputs, runtime_input_record=_record(input_path),
        resource_name="blueprint-native-task-policy-canary-" + run_id,
        hard_cap_usd=float(execution.get("provider_hard_cap_usd", candidate["max_cost_usd"])),
        hard_ttl_seconds=execution["hard_ttl_seconds"])
    authority_path = _stage(directory / "session_authority.json", authority)
    spec_path = _stage(directory / (candidate["candidate_id"] + ".policy_execution_spec.json"), native_spec)
    inventory = execution.get("checkpoint_inventory")
    bundle = (_read(root / "bundle_receipt.json") if (root / "bundle_receipt.json").exists() else
        build_policy_canary_session_bundle(job_dir=root / "bundle",
        packet_dir=Path(scene["base_native_packet"]["path"]).parent,
        runtime_source_packet_receipt=scene["runtime_source"]["path"],
        runtime_input_manifest_path=input_path, session_authority_path=authority_path,
        execution_spec_paths={candidate["candidate_id"]: spec_path},
        checkpoint_inventory_paths={candidate["candidate_id"]: _bound_path(inventory)} if inventory else {},
        team_robot_model_path=_bound_path(execution["team_robot_model"]) if execution.get("team_robot_model") else None,
        minimum_gpu_memory_gb=execution.get("minimum_gpu_memory_gb"),
        implementation_commit=execution["implementation_commit"]))
    bundle_path = _stage(root / "bundle_receipt.json", bundle)
    preflight = preflight_sealed_policy_canary_bundle(bundle)
    _stage(root / "static_preflight.json", preflight)
    if preflight.get("status") != "passed":
        raise ValueError("beta_screening_static_preflight_blocked:" + ",".join(preflight.get("blockers") or []))
    argv = ["gpu-canary", "--provider", "vast", "--probe-kind", PROBE_KIND,
            "--native-task-arena-policy-canary-session-authority", str(authority_path),
            "--native-task-arena-policy-canary-session-bundle-receipt", str(bundle_path),
            "--adp-job-dir", str(root / "allocator"),
            "--adp-max-hourly-rate-usd", str(execution["max_hourly_rate_usd"]),
            "--adp-max-spend-usd", str(execution.get("provider_hard_cap_usd", candidate["max_cost_usd"])),
            "--adp-hard-ttl-seconds", str(execution["hard_ttl_seconds"]),
            "--admission-out", str(root / "paid_admission.json"),
            "--adapter-output", str(root / "allocator_result.json"), "--execute"]
    if execution.get("credential_file"):
        argv += ["--beta-screening-credential-file", execution["credential_file"]]
    if execution.get("machine_avoidlist_path"):
        argv += ["--adp-machine-avoidlist", str(execution["machine_avoidlist_path"])]
    prepared = {"schema_version": "beta_policy_screening_prepared.v1", **identity, "run_id": run_id,
                "attempt_dir": str(root), "argv": argv, "authority": _record(authority_path),
                "bundle_receipt": _record(bundle_path), "prepared_digest": ""}
    prepared["prepared_digest"] = canonical_digest(prepared, digest_field="prepared_digest")
    _write(previous, prepared)
    return prepared


def _validate_prepared(plan: Mapping[str, Any], candidate: Mapping[str, Any], root: Path,
                       prepared: Mapping[str, Any]) -> None:
    identity = _identity(plan, candidate)
    if (any(prepared.get(key) != value for key, value in identity.items())
            or prepared.get("attempt_dir") != str(root)
            or prepared.get("prepared_digest") != canonical_digest(prepared, digest_field="prepared_digest")
            or _read(root / "prepared.json") != prepared):
        raise ValueError("beta_screening_prepared_binding_invalid")
    _bound_path(prepared["authority"])
    bundle = _read(_bound_path(prepared["bundle_receipt"]))
    _bound_path({"path": bundle["bundle_path"], "sha256": bundle["bundle_sha256"],
                 "size_bytes": bundle["bundle_size_bytes"]})


def preflight_candidate_capacity(*, plan: Mapping[str, Any], candidate: Mapping[str, Any],
                                 attempt_dir: Path, prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Observe canonical capacity without reserving a slot or touching a provider."""
    from .vast_launch_slots import inspect_vast_launch_admission
    _validate_prepared(plan, candidate, Path(attempt_dir).resolve(), prepared)
    admission = inspect_vast_launch_admission()
    blockers = admission.get("blockers") or []
    busy = admission.get("status") == "deferred" and blockers in (
        ["vast_paid_launch_lock_busy"], ["vast_release_deployment_in_progress"])
    return {"capacity_available": not busy, "blockers": blockers}


def launch_candidate(*, plan: Mapping[str, Any], candidate: Mapping[str, Any], attempt_dir: Path,
                     prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Consume a durable local launch intent exactly once before spawning."""
    from .team_candidate_quick10_launch import _process_start_ticks, _start_allocator
    root = Path(attempt_dir).resolve()
    _validate_prepared(plan, candidate, root, prepared)
    if (root / "allocator_launch_intent.json").exists():
        return reconcile_candidate(plan=plan, candidate=candidate, attempt_dir=root, prepared=prepared)
    try:
        _check_quote(plan, candidate)
        execution = candidate.get("execution_spec") or {}
        if execution.get("backend") == "native_g1.v1":
            from .beta_policy_screening_g1_launch import reopen_candidate
            reopen_candidate(plan, candidate)
        if execution.get("backend") == "task_family_native.v1":
            from .beta_policy_screening_family import reopen_family_planning_inputs, validate_family_candidate_catalog
            scene = _read(_bound_path(plan["scene_inputs"]))
            static = reopen_family_planning_inputs(scene)
            validate_family_candidate_catalog(plan, candidate,
                _read(_bound_path(execution["policy_execution_spec"])), static, scene=scene)
        if execution.get("credential_file"):
            from .beta_policy_screening_hosted import validate_hosted_delivery
            from .policy_interface_binding import TEAM_ROBOT_INTERFACES
            native_spec = _read(_bound_path(execution["policy_execution_spec"]))
            validate_hosted_delivery(native_spec, execution, episodes=candidate["episode_count"])
            if native_spec.get("policy_interface_id") in TEAM_ROBOT_INTERFACES:
                from .beta_policy_screening_team import validate_team_candidate_catalog
                validate_team_candidate_catalog(plan, candidate, native_spec, _read(_bound_path(plan["scene_inputs"])))
        elif execution.get("policy_execution_spec"):
            native_spec = _read(_bound_path(execution["policy_execution_spec"]))
            from .policy_interface_binding import TEAM_LOCAL_INTERFACES, TEAM_SCRIPTED_INTERFACES
            if native_spec.get("policy_interface_id") in TEAM_LOCAL_INTERFACES | TEAM_SCRIPTED_INTERFACES:
                from .beta_policy_screening_team import validate_team_candidate_catalog
                validate_team_candidate_catalog(plan, candidate, native_spec, _read(_bound_path(plan["scene_inputs"])))
        from .beta_policy_screening_worker import require_current_source
        require_current_source(plan["source"], site_id=plan["site_id"])
    except (ValueError, OSError, RuntimeError) as exc:
        # Claim the same exclusive intent slot as a launch, so a concurrent
        # caller cannot start after this refusal releases its reservation.
        refusal = {"prepared_digest": prepared["prepared_digest"], "status": "refused_before_spawn",
                   "blocker": "beta_screening_prelaunch_refused:" + type(exc).__name__}
        try:
            _write(root / "allocator_launch_intent.json", refusal)
        except FileExistsError:
            return reconcile_candidate(plan=plan, candidate=candidate, attempt_dir=root, prepared=prepared)
        return {"status": "blocked", "provider_allocation_performed": False, "provider_zero_verified": True,
                "no_allocation_proven": True, "blockers": [refusal["blocker"]]}
    try:
        _write(root / "allocator_launch_intent.json", {"prepared_digest": prepared["prepared_digest"]})
    except FileExistsError:
        return reconcile_candidate(plan=plan, candidate=candidate, attempt_dir=root, prepared=prepared)
    # A crash anywhere after intent remains uncertain. Neither a dead PID nor
    # global provider zero proves that another create cannot still occur.
    pid = _start_allocator(prepared["argv"], log_path=root / "allocator.log")
    process = {"pid": pid, "start_ticks": _process_start_ticks(pid),
               "prepared_digest": prepared["prepared_digest"]}
    _write(root / "allocator_process.json", process)
    return {"status": "running", "handle": process, "provider_zero_verified": False,
            "no_allocation_proven": False, "blockers": []}


def reconcile_candidate(*, plan: Mapping[str, Any], candidate: Mapping[str, Any], attempt_dir: Path,
                        prepared: Mapping[str, Any], handle: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Read retained allocator receipts; never start or restart an allocation."""
    from .task_evaluation_scene_execution_authority import proves_no_provider_allocation
    from .team_candidate_quick10_launch import _alive, _process_start_ticks
    from .team_candidate_quick10_closeout import validate_native_result_binding
    from .beta_policy_screening_closeout import (
        collect_scoped_closeout, scoped_closeout_verified, join_scoped_session_closeout,
    )
    root = Path(attempt_dir).resolve()
    _validate_prepared(plan, candidate, root, prepared)
    intent_path = root / "allocator_launch_intent.json"
    if intent_path.exists():
        intent = _read(intent_path)
        if (intent.get("status") == "refused_before_spawn"
                and intent.get("prepared_digest") == prepared["prepared_digest"]):
            return {"status": "blocked", "provider_allocation_performed": False, "provider_zero_verified": True,
                    "no_allocation_proven": True, "blockers": [intent["blocker"]]}
    result_path = root / "allocator_result.json"
    if result_path.is_file():
        adapter = _read(result_path)
        if proves_no_provider_allocation(adapter):
            refusal = {"status": "blocked", "provider_allocation_performed": False,
                       "provider_zero_verified": True, "no_allocation_proven": True,
                       "blockers": adapter.get("blockers") or ["beta_screening_allocator_refused"]}
            if adapter.get("status") == "blocked" and adapter.get("blockers") == ["vast_paid_launch_lock_busy"]:
                # Every hosted HTTP call occurs inside the provider session.
                # This exact canonical no-create refusal proves no policy API
                # calls occurred; a new attempt may compete for a slot later.
                refusal.update(retryable_capacity_refusal=True, no_policy_api_calls_proven=True,
                               capacity_refusal_receipt=_record(result_path))
            return refusal
        try:
            absence = collect_scoped_closeout(adapter=adapter, prepared=prepared, adapter_path=result_path)
            observed = root / "provider_absence" / (canonical_digest(absence)[7:] + ".json")
            if not observed.exists():
                _write(observed, absence)
            verified = scoped_closeout_verified(absence, adapter=adapter, prepared=prepared, adapter_path=result_path)
        except (OSError, ValueError, KeyError, TypeError):
            return {"status": "reconciliation_required", "provider_zero_verified": False,
                    "no_allocation_proven": False, "blockers": ["beta_screening_scoped_closeout_unproven"]}
        if not verified:
            return {"status": "reconciliation_required", "provider_zero_verified": False,
                    "no_allocation_proven": False,
                    "blockers": absence.get("blockers") or ["beta_screening_scoped_closeout_unproven"]}
        if candidate.get("execution_spec", {}).get("backend") == "native_g1.v1":
            try:
                from .beta_policy_screening_g1_launch import join_result
                joined = join_result(plan=plan, candidate=candidate, adapter=adapter, prepared=prepared,
                                     receipt=absence, adapter_path=result_path)
            except (OSError, ValueError, KeyError, TypeError):
                return {"status": "blocked", "provider_allocation_performed": True,
                        "provider_zero_verified": True, "no_allocation_proven": False,
                        "blockers": ["g1_beta_native_result_missing_or_invalid"]}
            return {"status": "completed", "provider_allocation_performed": True,
                    "provider_zero_verified": True, "no_allocation_proven": False,
                    "result": joined, "blockers": joined.get("blockers") or []}
        try:
            native_path = Path(str(adapter.get("native_control_result_path") or ""))
            native = _read(native_path)
            validate_native_result_binding(native, expected_digest=adapter.get("native_control_result_digest"))
            joined = join_scoped_session_closeout(inner=native, adapter=adapter, prepared=prepared,
                                                  receipt=absence, adapter_path=result_path)
            joined = validate_session_result(joined)
            if joined.get("beta_screening_admission", {}).get("plan_digest") != plan["plan_digest"]:
                raise ValueError("beta_screening_native_result_plan_mismatch")
        except (OSError, ValueError, KeyError, TypeError):
            # The provider's lifecycle is resolved even if scientific output is
            # missing or invalid. Debit the full reservation and publish no score.
            return {"status": "blocked", "provider_allocation_performed": True,
                    "provider_zero_verified": True, "no_allocation_proven": False,
                    "blockers": ["beta_screening_native_result_missing_or_invalid"]}
        return {"status": "completed", "provider_allocation_performed": True,
                "provider_zero_verified": True, "no_allocation_proven": False,
                "result": joined, "blockers": joined.get("blockers") or []}
    process_path = root / "allocator_process.json"
    if process_path.exists():
        process = _read(process_path)
        if (process.get("prepared_digest") == prepared["prepared_digest"] and process.get("start_ticks")
                and _process_start_ticks(process["pid"]) == process["start_ticks"] and _alive(process["pid"])):
            return {"status": "running", "handle": process, "provider_zero_verified": False,
                    "no_allocation_proven": False, "blockers": []}
    return {"status": "reconciliation_required", "provider_zero_verified": False,
            "no_allocation_proven": False, "blockers": ["beta_screening_allocator_identity_unresolved"]}


def build_planning_inputs(**kwargs: Any) -> dict[str, Any]:
    """Derive planning capabilities from retained operator and scene records."""
    from .beta_policy_screening_capabilities import build_planning_inputs as build
    return build(**kwargs)
