"""Fixed-arm/DROID full-matrix cell execution on the existing native runtime.

Each cell uses one isolated interpreter because closing/rebuilding Isaac in a
shared interpreter invalidates renderer state. No provider is allocated here.
The caller retains all existing allocator, watchdog, hold, spend and teardown
responsibilities; each episode asks its parent for permission before it starts.
"""

from __future__ import annotations

import json
import multiprocessing
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .exact_workcell_evaluation_session import (
    _atomic,
    artifact_record,
    execute_evaluation_schedule,
    safe_execution_blockers,
)
from .core.immutable_file import write_file_exclusive
from .exact_workcell_native_runtime import prepare_native_runtime_cell, read_native_runtime_cell
from .exact_workcell_variation_matrix import ExactWorkcellVariationError


def _receipt_path(root: Path, episode_id: str, cell_id: str) -> Path:
    # The existing evidence index resolves media relative to this episode's
    # media root and requires its receipt to be inside that same root.
    return (
        root
        / "cells"
        / canonical_digest({"cell_id": cell_id})[7:]
        / "episodes"
        / "receipts"
        / (canonical_digest({"episode_id": episode_id})[7:] + ".json")
    )


class _ResetBoundEnvironment:
    def __init__(self, environment: Any, reset_reader: Callable[[str], Mapping[str, Any]]):
        self._environment, self.scientific_reset_readback = environment, reset_reader

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)


def run_native_evaluation_cell(
    *,
    config: Mapping[str, Any],
    cell: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    begin: Callable[[Mapping[str, Any]], None],
    commit: Callable[[Mapping[str, Any], Mapping[str, Any]], None],
    runtime: Any = None,
) -> None:
    """Use real native control/policy loops; injectable Isaac seam is test-only.

    Construction results must be real, qualified, and bound to each prepared
    cell. The controls compiler refuses a copied canonical construction receipt
    after a scene/task transform. Missing native inputs remain a precise block.
    """
    from .adp009d_control_episode import run_task_neutral_control
    from .adp009d_droid_action_execution import GripperConvention
    from .native_task_arena_controls_worker import _construction_global_ik_joint_targets
    from .native_task_arena_policy_canary_worker import isaac_cell_runtime
    from .native_task_arena_policy_worker import _PolicyQueryTracker
    from .native_task_control_plan import materialize_native_task_control_plan
    from .native_task_arena_policy_bundle import validate_native_task_policy_execution_spec
    from .exact_workcell_native_admission import (
        validate_native_source_policy_candidates,
        derive_native_cell_policy_admission,
    )
    from .policy_interface_binding import observation_interface
    from .policy_scientific_reset import (
        compare_reset_readbacks,
        read_native_reset_channels,
        seal_reset_readback,
    )

    root = Path(config["evidence_root"]).resolve()
    matrix, request = config["matrix"], config["request"]
    source = config["source_scene_plan"]
    plan = prepare_native_runtime_cell(
        matrix=matrix,
        request=request,
        cell_id=cell["cell_id"],
        source_scene_plan=source,
        source_plan_digest=config["source_binding"]["source_plan_digest"],
    )
    if plan["task_kind"] != "rigid_pick_place" or plan["robot"]["robot_id"] != "franka_panda":
        raise ExactWorkcellVariationError(["full_matrix_native_profile_unavailable"])
    if plan["task_spec"].get("astra_asset_adoption") is not None:
        # This specialized profile has an additional existing monitor and
        # feasibility admission. Never skip it through the rigid-task port.
        raise ExactWorkcellVariationError(["full_matrix_native_asset_profile_unavailable"])
    construction = config["construction_result"]
    control_plan = materialize_native_task_control_plan(
        scene_plan=plan, construction_result=construction
    )
    candidate_specs = config["candidate_specs"]
    identities = config["schedule_request"]["candidate_set"]["candidate_identity_digests"]
    if set(candidate_specs) != {"pi05_droid", "groot_n17_droid"} or set(identities) != set(
        candidate_specs
    ):
        raise ExactWorkcellVariationError(["full_matrix_droid_pair_required"])
    for candidate, spec in candidate_specs.items():
        validate_native_task_policy_execution_spec(spec)
        if (
            spec.get("candidate_id") != candidate
            or spec.get("execution_spec_digest")
            != canonical_digest(spec, digest_field="execution_spec_digest")
            or spec["execution_spec_digest"] != identities[candidate]
            or spec.get("scene_plan_digest") != config["source_binding"]["source_plan_digest"]
            or spec.get("task_id") != source["task_id"]
        ):
            raise ExactWorkcellVariationError(["full_matrix_frozen_candidate_binding_mismatch"])
    validate_native_source_policy_candidates(
        source_scene_plan=source,
        candidate_specs=candidate_specs,
        candidate_identity_digests=identities,
        readiness=config["policy_readiness"],
        scenario_suite=config["scenario_suite"],
    )
    cell_admission = None
    bound = runtime if runtime is not None else isaac_cell_runtime()
    cell_root = root / "cells" / canonical_digest({"cell_id": cell["cell_id"]})[7:]
    cell_root.mkdir(parents=True, exist_ok=True)
    app = built = None
    try:
        from .native_task_arena_policy_canary_worker import appearance_render_backend_from_plan

        appearance = appearance_render_backend_from_plan(
            plan, packet_request=config["packet_request"]
        )
        app, launch = bound.launch_isaac(
            cell_root / "runtime_source_provisioning.json",
            device=bound.device,
            appearance_render_path=appearance["launch_render_path"],
        )
        _atomic(cell_root / "launch.json", dict(launch))
        dependencies = bound.preflight_dependency_matrix(robot_id=plan["robot"]["robot_id"])
        if dependencies["all_required_available"] is not True:
            raise ExactWorkcellVariationError(["full_matrix_native_dependencies_unavailable"])
        preconstruction = bound.prepare_preconstruction(expected_device=bound.device)
        if preconstruction["passed"] is not True:
            raise ExactWorkcellVariationError(["full_matrix_native_preconstruction_failed"])
        built = bound.build_environment(
            plan,
            device=bound.device,
            bundle_root=Path(config["native_packet_root"]),
            preconstruction_receipt=preconstruction,
        )
        renderer = bound.prepare_appearance_renderer(simulation_app=app, plan=plan)
        if (
            renderer.get("passed") is not True
            or bound.read_device_binding(built, expected_device=bound.device)["passed"] is not True
        ):
            raise ExactWorkcellVariationError(["full_matrix_renderer_or_device_unproven"])
        env = built.env
        env.reset(seed=cell["seed"])
        robot = env.unwrapped.scene["robot"]
        gripper = bound.gripper_probe(env=env, robot=robot, seed=cell["seed"])
        if gripper.get("status") != "measured":
            raise ExactWorkcellVariationError(["full_matrix_gripper_convention_unproven"])
        env.reset(seed=cell["seed"])
        servo = bound.make_servo(env=env, robot=robot, gripper_convention=gripper)
        if bound.make_rigid_task_readback is None or bound.wrap_rigid_scoring_environment is None:
            raise ExactWorkcellVariationError(["full_matrix_rigid_readback_unavailable"])
        task_readback = bound.make_rigid_task_readback(built)
        targets = _construction_global_ik_joint_targets(
            construction=construction, control_plan=control_plan
        )
        environment, _ = bound.build_episode_environment(
            built=built,
            gripper_convention=gripper,
            servo=servo,
            task_readback=task_readback,
            to_tensor=bound.to_tensor,
            scripted_pose_joint_targets=targets,
            scripted_pose_phase_targets=[
                row
                for row in control_plan["scripted_positive_actions"]
                if row.get("mode") == "ik_pose"
            ],
        )
        environment = bound.wrap_rigid_scoring_environment(
            environment=environment, task_readback=task_readback, task_spec=plan["task_spec"]
        )
        if bound.prepolicy_camera_gate is None:
            raise ExactWorkcellVariationError(["full_matrix_observation_gate_unavailable"])
        gate = bound.prepolicy_camera_gate(
            simulation_app=app,
            built=built,
            packet_request=config["packet_request"],
            plan=plan,
            output_root=cell_root,
        )
        if gate.get("policy_observation_integrity_passed") is not True:
            raise ExactWorkcellVariationError(["full_matrix_observation_gate_failed"])
        if bound.configure_post_gate_renderer is None:
            raise ExactWorkcellVariationError(["full_matrix_post_gate_renderer_unavailable"])
        renderer_guard = bound.configure_post_gate_renderer(
            observation_gate=gate, output_path=cell_root / "post_gate_renderer.json"
        )
        if renderer_guard.get("status") != "configured":
            raise ExactWorkcellVariationError(["full_matrix_post_gate_renderer_unproven"])
        _atomic(cell_root / "post_gate_renderer.json", dict(renderer_guard))
        reference = None
        for scheduled in config["schedule"]["rows"]:
            previous = _receipt_path(root, scheduled["episode_id"], scheduled["cell_id"])
            if scheduled["cell_id"] == cell["cell_id"] and previous.exists():
                retained = json.loads(previous.read_text())
                reset = retained.get("scientific_reset")
                if reset is not None:
                    reference = reset
                    break
        for row in rows:
            subject = row["subject_id"]
            application = None

            def reset_reader(subject_id: str = subject) -> Mapping[str, Any]:
                nonlocal application, reference
                application = read_native_runtime_cell(
                    built=built,
                    matrix=matrix,
                    request=request,
                    cell_id=cell["cell_id"],
                    prepared_plan_digest=plan["plan_digest"],
                )
                if application["status"] != "applied_and_readback_verified":
                    raise ExactWorkcellVariationError(["full_matrix_native_parameters_unproven"])
                reset = seal_reset_readback(
                    binding={
                        "candidate_id": subject_id,
                        "cell_id": cell["cell_id"],
                        "seed": cell["seed"],
                        "resolved_scenario_digest": canonical_digest(cell["resolved_values"]),
                        "task_spec_digest": canonical_digest(plan["task_spec"]),
                        "matrix_cell_digest": cell["cell_digest"],
                        "matrix_reset_digest": cell["reset_digest"],
                    },
                    **read_native_reset_channels(built, environment),
                )
                if reset["complete"] is not True:
                    raise ExactWorkcellVariationError(["full_matrix_scientific_reset_incomplete"])
                if (
                    reference is not None
                    and not compare_reset_readbacks(reference, reset)["comparison_eligible"]
                ):
                    raise ExactWorkcellVariationError(["full_matrix_scientific_reset_mismatch"])
                reference = reset
                return reset

            episode_env = _ResetBoundEnvironment(environment, reset_reader)
            if row["subject_type"] == "control":
                begin(row)  # Durable intent and authority recheck before action.
                episode = run_task_neutral_control(
                    environment=episode_env,
                    task_spec=plan["task_spec"],
                    control_plan=control_plan,
                    control_id=subject,
                    episode_id=row["episode_id"],
                    gripper_open_command=float(gripper["open_command"]),
                    gripper_closed_command=float(gripper["closed_command"]),
                    output_dir=cell_root / "episodes",
                )
            else:
                # Both controls must have genuine retained passing receipts on
                # this exact cell before a policy endpoint is even loaded.
                controls = [
                    r
                    for r in config["schedule"]["rows"]
                    if r["cell_id"] == cell["cell_id"] and r["subject_type"] == "control"
                ]
                if not all(
                    _receipt_path(root, r["episode_id"], r["cell_id"]).exists()
                    and json.loads(
                        _receipt_path(root, r["episode_id"], r["cell_id"]).read_text()
                    ).get("control_passed")
                    is True
                    for r in controls
                ):
                    raise ExactWorkcellVariationError(["full_matrix_cell_controls_not_passed"])
                if cell_admission is None:
                    cell_admission = derive_native_cell_policy_admission(
                        source_scene_plan=source,
                        source_candidate_specs=candidate_specs,
                        candidate_identity_digests=identities,
                        scene_plan=plan,
                        construction_result=construction,
                        control_plan=control_plan,
                        control_receipt_paths={
                            r["subject_id"]: _receipt_path(root, r["episode_id"], r["cell_id"])
                            for r in controls
                        },
                        media_root=cell_root / "episodes",
                        readiness=config["policy_readiness"],
                        scenario_suite=config["scenario_suite"],
                    )
                    _atomic(cell_root / "cell_policy_admission.json", cell_admission)
                spec = cell_admission["candidate_specs"][subject]
                groot = None
                if subject == "groot_n17_droid":
                    groot, _ = bound.groot_worker_identity(
                        output_root=Path(config["policy_runtime_evidence_root"]), spec=spec
                    )
                begin(
                    row
                )  # Controls/rights are admitted; no endpoint was called before this intent.
                client = None
                progress: dict[str, Any] = {}
                try:
                    client = bound.policy_client(spec, groot_worker_identity_receipt=groot)
                    episode = bound.run_policy_episode(
                        environment=episode_env,
                        policy=_PolicyQueryTracker(client),
                        candidate_id=subject,
                        interface_candidate_id=observation_interface(spec),
                        prompt=spec["prompt"],
                        task_spec=plan["task_spec"],
                        max_policy_queries=spec["max_policy_queries"],
                        settle_window_samples=plan["task_spec"]["settle_window_samples"],
                        open_loop_horizon=spec["open_loop_horizon"],
                        gripper=GripperConvention(
                            closed_command=float(gripper["closed_command"]),
                            open_command=float(gripper["open_command"]),
                            measured_by_probe=True,
                        ),
                        media_output_dir=cell_root / "episodes",
                        episode_id=row["episode_id"],
                        scoring_authorized=True,
                        require_complete_multicamera_media=True,
                        require_prestart_readiness=True,
                        observation_integrity={
                            "authority": config["observation_integrity_authority"],
                            "appearance_render_backend_receipt_digest": appearance[
                                "receipt_digest"
                            ],
                            "runtime_gate": gate,
                        },
                        scientific_reset_reader=reset_reader,
                        progress=progress,
                    )
                except Exception as exc:
                    finalizer = progress.get("_legitimate_early_terminal_finalizer")
                    episode = None
                    if progress.get("episode_started") is True and callable(finalizer):
                        try:
                            episode = finalizer(exc)
                        except Exception:
                            episode = None
                    if episode is None:
                        episode = retain_native_policy_failure(
                            root=root,
                            cell_root=cell_root,
                            row=row,
                            progress=progress,
                            failure=exc,
                            run_id=config["source_binding"]["job_id"],
                        )
                finally:
                    close = getattr(client, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception as exc:
                            # Closing a transport must not discard a sealed
                            # episode or replace its scientific terminal state.
                            _atomic(
                                cell_root / (subject + ".policy_client_close_failure.json"),
                                {
                                    "failure_type": type(exc).__name__,
                                    "automatic_retry_allowed": False,
                                },
                            )
            path = _receipt_path(root, row["episode_id"], row["cell_id"])
            if path.exists():
                raise ExactWorkcellVariationError(["full_matrix_episode_receipt_already_exists"])
            write_file_exclusive(
                path, (json.dumps(dict(episode), sort_keys=True, allow_nan=False) + "\n").encode()
            )
            record = {
                "episode_id": row["episode_id"],
                "episode_binding_digest": row["episode_binding_digest"],
                "artifact": artifact_record(root, path),
                "reset_application": application,
                "media_root_relative": (cell_root / "episodes").relative_to(root).as_posix(),
            }
            if row["subject_type"] == "policy":
                record["candidate_identity_digest"] = identities[subject]
            write_file_exclusive(
                root
                / "receipts"
                / (canonical_digest({"episode_id": row["episode_id"]})[7:] + ".record.json"),
                (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode(),
            )
            commit(row, record)
    finally:
        # Seal close intent before SimulationApp.close, which may terminate
        # the child interpreter rather than return to Python. A failed journal
        # write must not prevent either runtime resource from being closed.
        try:
            _atomic(
                cell_root / "native_close_intent.json",
                {"close_required": True, "provider_teardown_owned_by_allocator": True},
            )
        finally:
            try:
                if built is not None:
                    built.env.close()
            finally:
                if app is not None:
                    app.close()


def retain_native_policy_failure(
    *,
    root: Path,
    cell_root: Path,
    row: Mapping[str, Any],
    progress: Mapping[str, Any],
    failure: Exception,
    run_id: str,
) -> dict[str, Any]:
    """Reuse the existing failure/media sealer and keep its original identity.

    The full-schedule wrapper binds the scheduled identity to that retained
    failure artifact; it never presents a transport failure as a scored episode.
    """
    from .policy_canary_worker_evidence import _write_episode_failure_gap

    original = _write_episode_failure_gap(
        output_root=cell_root,
        run_id=run_id,
        context={**row, "candidate_id": row["subject_id"]},
        failure=failure,
        progress=progress,
    )
    retained = json.loads(original.read_text())
    result = {
        "schema_version": "exact_workcell_native_episode_failure.v1",
        "episode_id": row["episode_id"],
        "candidate_id": row["subject_id"],
        "episode_binding_digest": row["episode_binding_digest"],
        "status": "failed",
        "failure_evidence": artifact_record(root, original),
        "scientific_reset": retained.get("scientific_reset"),
        "candidate_policy_queried": retained["candidate_policy_queried"],
        "candidate_policy_query_attempted": retained["candidate_policy_query_attempted"],
        "visual_evidence": retained["visual_evidence"],
        "score": {"status": "not_scored"},
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    return result


def _native_child(
    pipe: Any, config: Mapping[str, Any], cell: Mapping[str, Any], rows: list[Mapping[str, Any]]
) -> None:
    def begin(row: Mapping[str, Any]) -> None:
        pipe.send(("begin", row))
        if pipe.recv() != "ok":
            raise RuntimeError("full_matrix_parent_refused")

    def commit(row: Mapping[str, Any], record: Mapping[str, Any]) -> None:
        pipe.send(("commit", row, record))
        if pipe.recv() != "ok":
            raise RuntimeError("full_matrix_parent_refused")

    try:
        run_native_evaluation_cell(config=config, cell=cell, rows=rows, begin=begin, commit=commit)
        pipe.send(("closed",))
    except Exception as exc:
        pipe.send(("failure", type(exc).__name__, safe_execution_blockers(exc)))
    finally:
        pipe.close()


@dataclass(frozen=True)
class NativeCellProcessRunner:
    """Production per-cell process port for execute_evaluation_schedule.

    ``config`` contains retained source/packet/spec paths and exact per-cell
    qualified construction results. No API token or spend authority is added.
    The parent's existing watchdog remains the enclosing allocation limit.
    """

    config: Mapping[str, Any]
    timeout_seconds: float

    def __call__(
        self,
        *,
        cell: Mapping[str, Any],
        rows: list[Mapping[str, Any]],
        begin: Callable[..., None],
        commit: Callable[..., None],
    ) -> None:
        if not 0 < self.timeout_seconds <= 86400:
            raise ExactWorkcellVariationError(["full_matrix_cell_timeout_invalid"])
        construction = self.config["construction_results"].get(cell["cell_id"])
        if construction is None:
            raise ExactWorkcellVariationError(["full_matrix_cell_construction_required"])
        config = {key: value for key, value in self.config.items() if key != "construction_results"}
        config["construction_result"] = construction
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=_native_child, args=(child, config, cell, rows))
        process.start()
        child.close()
        deadline = time.monotonic() + self.timeout_seconds
        try:
            while process.is_alive() or parent.poll():
                if time.monotonic() > deadline:
                    raise TimeoutError("full_matrix_cell_deadline")
                if not parent.poll(0.1):
                    continue
                try:
                    message = parent.recv()
                except EOFError:
                    break
                if message[0] == "begin":
                    begin(message[1])
                    parent.send("ok")
                elif message[0] == "commit":
                    commit(message[1], message[2])
                    parent.send("ok")
                elif message[0] == "failure":
                    if message[2]:
                        raise ExactWorkcellVariationError(message[2])
                    raise RuntimeError("full_matrix_native_cell_failed:" + message[1])
            process.join(timeout=1)
            if process.exitcode != 0:
                raise RuntimeError("full_matrix_native_cell_exit_unconfirmed")
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
            parent.close()


def execute_native_evaluation_schedule(
    *, config: Mapping[str, Any], assert_execution_allowed: Callable[[], None]
) -> dict[str, Any]:
    """Full production port inside an existing, externally admitted allocation.

    This API intentionally has no allocator or auto-activation flag. The
    existing managed run host supplies its live authority recheck and owns
    watchdog/teardown/provider-zero after the result returns.
    """

    # Snapshot every science/runtime input before handing cells to spawned
    # interpreters. A changed authority, endpoint, construction or native packet
    # reference cannot silently resume the same retained execution identity.
    def portable(value: Any) -> str:
        if isinstance(value, Path):
            return str(value)
        raise TypeError("full_matrix_config_requires_retained_json_inputs")

    try:
        config = json.loads(json.dumps(dict(config), default=portable, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ExactWorkcellVariationError(["full_matrix_config_not_immutable_json"]) from exc
    execution_inputs_digest = canonical_digest(config)
    if (
        config["source_binding"]["site_id"] != config["request"]["scene_binding"]["scene_id"]
        or config["source_binding"]["source_plan_digest"]
        != config["source_scene_plan"]["plan_digest"]
    ):
        raise ExactWorkcellVariationError(["full_matrix_source_identity_mismatch"])
    runner = NativeCellProcessRunner(config=config, timeout_seconds=config["cell_timeout_seconds"])
    return execute_evaluation_schedule(
        request=config["request"],
        schedule_request=config["schedule_request"],
        matrix=config["matrix"],
        schedule=config["schedule"],
        source_binding=config["source_binding"],
        evidence_root=Path(config["evidence_root"]),
        run_cell=runner,
        assert_execution_allowed=assert_execution_allowed,
        execution_inputs_digest=execution_inputs_digest,
    )


@dataclass(frozen=True)
class ExactWorkcellNativeEvaluationRunExecutor:
    """Registered full-schedule runtime port; never an allocation bypass."""

    adapter_id: str = "exact_workcell_native_droid"

    def execute(
        self, *, spec: Any, output_dir: Path, context: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        config = context.get("native_exact_workcell_config")
        recheck = context.get("assert_native_execution_allowed")
        binding = context.get("_evaluation_run_binding") or {}
        if not isinstance(config, Mapping) or not callable(recheck):
            return {
                "schema_version": "exact_workcell_execution_adapter.v1",
                "status": "blocked",
                "blockers": ["existing_native_runtime_authority_and_immutable_inputs_required"],
            }
        if (
            config["source_binding"]["configuration_digest"] != binding.get("spec_digest")
            or binding.get("run_id") != spec.run_id
            or spec.task_scenario_pack.get("adapter_id") != "exact_workcell_variation_matrix"
            or spec.task_scenario_pack.get("matrix_digest") != config["matrix"]["matrix_digest"]
            or spec.scene_bundle.get("content_digest")
            != config["request"]["scene_binding"]["scene_digest"]
            or spec.robot_adapter.get("robot_profile_id")
            != config["request"]["embodiment_binding"]["embodiment_id"]
        ):
            raise ExactWorkcellVariationError(["full_matrix_evaluation_run_binding_mismatch"])
        expected_root = (output_dir / "native_exact_workcell").resolve()
        if Path(config["evidence_root"]).resolve() != expected_root:
            raise ExactWorkcellVariationError(["full_matrix_output_binding_mismatch"])
        result = execute_native_evaluation_schedule(config=config, assert_execution_allowed=recheck)
        return {
            "schema_version": "exact_workcell_execution_adapter.v1",
            "status": "completed"
            if result["status"] == "complete_simulator_evidence"
            else "blocked",
            "blockers": result.get("blockers", [])
            or result.get("evidence", {}).get("blockers", [])
            or ([] if result["status"] == "complete_simulator_evidence" else [result["status"]]),
            "session_result": result,
            "qualification_authorized": False,
            "provider_allocation_performed": False,
            "provider_teardown_owned_by_allocator": True,
        }
