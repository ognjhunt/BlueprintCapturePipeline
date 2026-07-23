"""Frozen, benchmark-grade policy evaluation protocol and reporting.

This module makes benchmark conventions explicit without coupling Blueprint to
BEHAVIOR, RoboCasa, or any other benchmark runtime.  It compiles a private,
frozen split and execution plan, a redacted public benchmark card, an exact
baseline registry, and evidence-backed aggregate/external-fidelity reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import read_json_any, utc_now_iso, write_json
from .external_tool_runtime import PUBLIC_CLAIM_UPGRADE_KEY


SPEC_SCHEMA_VERSION = "blueprint_benchmark_spec.v1"
SPLIT_SCHEMA_VERSION = "blueprint_benchmark_split_manifest.v1"
CARD_SCHEMA_VERSION = "blueprint_benchmark_card.v1"
BASELINE_REGISTRY_SCHEMA_VERSION = "blueprint_public_baseline_registry.v1"
PLAN_SCHEMA_VERSION = "blueprint_benchmark_execution_plan.v1"
RESULTS_SCHEMA_VERSION = "blueprint_benchmark_results.v1"
REPORT_SCHEMA_VERSION = "blueprint_benchmark_report.v1"
EXTERNAL_REFERENCE_SCHEMA_VERSION = "external_reference_results.v1"
EXTERNAL_REPORT_SCHEMA_VERSION = "blueprint_external_rank_fidelity_report.v1"
WEBAPP_PROJECTION_SCHEMA_VERSION = "blueprint_webapp_benchmark_projection.v1"
BENCHMARK_REQUEST_STATUS_SCHEMA_VERSION = "blueprint_benchmark_protocol_request_status.v1"
EVIDENCE_INDEX_SCHEMA_VERSION = "blueprint_benchmark_evidence_index.v1"

BOOTSTRAP_REPLICATES = 10_000
# Kept separate so focused unit tests can shorten sampling without weakening the
# validated public protocol, which always requires BOOTSTRAP_REPLICATES.
_BOOTSTRAP_EXECUTION_REPLICATES = BOOTSTRAP_REPLICATES
GENERALIZATION_AXES = (
    "task",
    "scene",
    "object",
    "camera",
    "lighting",
    "embodiment",
)
SPLITS = ("train", "dev", "public_test", "hidden_test")
SCORED_SPLITS = ("public_test", "hidden_test")
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _strict_rows(value: Any) -> tuple[list[dict[str, Any]], bool]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return [], False
    if any(not isinstance(row, Mapping) for row in value):
        return [], False
    return [dict(row) for row in value], True


def _string(value: Any) -> str:
    return str(value or "").strip()


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any, *, minimum: int = 0) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return value


def _digest(value: Any) -> str:
    text = _string(value).lower()
    return text.removeprefix("sha256:") if _SHA256_RE.fullmatch(text) else ""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_id(value: Any) -> bool:
    return bool(_ID_RE.fullmatch(_string(value)))


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _artifact_ref_valid(value: Any) -> bool:
    row = _mapping(value)
    return bool(_string(row.get("uri")) and _digest(row.get("sha256")))


def _policy_errors(row: Mapping[str, Any], *, public_baseline: bool) -> list[str]:
    errors: list[str] = []
    policy_id = _string(row.get("policy_id"))
    if not _valid_id(policy_id):
        errors.append("policy_id_missing_or_invalid")
    for field in ("policy_family", "checkpoint_id", "embodiment_id"):
        if not _string(row.get(field)):
            errors.append(f"{field}_missing")
    for field in ("checkpoint_sha256", "adapter_code_sha256"):
        if not _digest(row.get(field)):
            errors.append(f"{field}_missing_or_invalid")
    runner = _mapping(row.get("runner"))
    if runner.get("schema_version") != "reproducible_policy_runner.v1":
        errors.append("runner_schema_missing_or_invalid")
    if not _string(runner.get("command")):
        errors.append("runner_command_missing")
    if not _digest(runner.get("runner_manifest_sha256")):
        errors.append("runner_manifest_sha256_missing_or_invalid")
    image = _string(runner.get("container_image_digest"))
    source_revision = _string(runner.get("source_revision"))
    if not image and not source_revision:
        errors.append("runner_container_digest_or_source_revision_required")
    if image and not re.fullmatch(r".+@sha256:[0-9a-f]{64}", image):
        errors.append("runner_container_image_digest_invalid")
    if public_baseline:
        if row.get("public") is not True:
            errors.append("public_baseline_must_be_public")
        if not _string(row.get("source_uri")):
            errors.append("public_baseline_source_uri_missing")
        if not _string(row.get("license")):
            errors.append("public_baseline_license_missing")
    return [f"{policy_id or 'unknown'}:{error}" for error in errors]


def validate_benchmark_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a complete frozen benchmark definition without writing files."""

    blockers: list[str] = []
    if spec.get("schema_version") != SPEC_SCHEMA_VERSION:
        blockers.append("benchmark_spec_schema_missing_or_unsupported")
    for field in ("benchmark_id", "benchmark_version", "protocol_version"):
        if not _valid_id(spec.get(field)):
            blockers.append(f"benchmark_identity_missing_or_invalid:{field}")
    if spec.get("frozen") is not True:
        blockers.append("benchmark_must_be_frozen")
    if not _digest(spec.get("preregistration_sha256")):
        blockers.append("preregistration_sha256_missing_or_invalid")

    tasks, tasks_valid = _strict_rows(spec.get("tasks"))
    if not tasks_valid or not tasks:
        blockers.append("benchmark_tasks_missing_or_invalid")
    task_ids: list[str] = []
    for index, task in enumerate(tasks):
        task_id = _string(task.get("task_id"))
        task_ids.append(task_id)
        if not _valid_id(task_id):
            blockers.append(f"task_identity_missing_or_invalid:{index}")
        for field in ("instruction", "reset_protocol", "success_definition"):
            if not _string(task.get(field)):
                blockers.append(f"task_field_missing:{index}:{field}")
        timeout = _number(task.get("timeout_seconds"))
        if timeout is None or timeout <= 0:
            blockers.append(f"task_timeout_missing_or_invalid:{index}")
        predicates = _rows(task.get("partial_progress_predicates"))
        if not predicates:
            blockers.append(f"task_partial_progress_predicates_missing:{index}")
        elif any(
            not _valid_id(item.get("predicate_id"))
            or (_number(item.get("weight")) is None or float(item.get("weight")) <= 0)
            for item in predicates
        ):
            blockers.append(f"task_partial_progress_predicates_invalid:{index}")
    if len(task_ids) != len(set(task_ids)):
        blockers.append("duplicate_task_id")

    action_space = _mapping(spec.get("action_space"))
    for field in ("schema_ref", "coordinate_frame", "timestamp_semantics"):
        if not _string(action_space.get(field)):
            blockers.append(f"action_space_field_missing:{field}")
    if _integer(action_space.get("dimension"), minimum=1) is None:
        blockers.append("action_space_dimension_missing_or_invalid")
    if not _digest(action_space.get("normalization_manifest_sha256")):
        blockers.append("action_space_normalization_digest_missing_or_invalid")

    environment = _mapping(spec.get("environment"))
    if not _valid_id(environment.get("site_id")):
        blockers.append("environment_site_id_missing_or_invalid")
    for field in ("site_package_sha256", "observation_calibration_sha256"):
        if not _digest(environment.get(field)):
            blockers.append(f"environment_digest_missing_or_invalid:{field}")
    if environment.get("representation_type") not in {
        "captured_3dgs_site_memory",
        "simready_usd",
        "hybrid",
        "other",
    }:
        blockers.append("environment_representation_type_invalid")
    if environment.get("physics_authority") not in {
        "none",
        "mujoco",
        "isaac",
        "newton",
        "real_robot",
    }:
        blockers.append("environment_physics_authority_invalid")
    if (
        environment.get("physics_authority") != "none"
        and not _digest(environment.get("physics_asset_sha256"))
    ):
        blockers.append("environment_physics_asset_digest_required")
    if not isinstance(environment.get("same_site_capture"), bool):
        blockers.append("environment_same_site_capture_boolean_required")

    evaluator_runtime = _mapping(spec.get("evaluator_runtime"))
    for field in ("evaluator_id", "evaluator_version"):
        if not _valid_id(evaluator_runtime.get(field)):
            blockers.append(f"evaluator_runtime_identity_missing_or_invalid:{field}")
    for field in (
        "runner_manifest_sha256",
        "success_evaluator_sha256",
        "robot_adapter_sha256",
        "observation_adapter_sha256",
        "action_adapter_sha256",
    ):
        if not _digest(evaluator_runtime.get(field)):
            blockers.append(f"evaluator_runtime_digest_missing_or_invalid:{field}")
    if not (
        _string(evaluator_runtime.get("source_revision"))
        or re.fullmatch(
            r".+@sha256:[0-9a-f]{64}",
            _string(evaluator_runtime.get("container_image_digest")),
        )
    ):
        blockers.append("evaluator_runtime_exact_source_or_container_required")
    if evaluator_runtime.get("deterministic_seeding") is not True:
        blockers.append("evaluator_runtime_deterministic_seeding_required")

    scenarios, scenarios_valid = _strict_rows(spec.get("scenarios"))
    if not scenarios_valid or not scenarios:
        blockers.append("benchmark_scenarios_missing_or_invalid")
    scenario_ids: list[str] = []
    axis_values: dict[str, set[str]] = defaultdict(set)
    split_counts: dict[str, int] = defaultdict(int)
    for index, scenario in enumerate(scenarios):
        scenario_id = _string(scenario.get("scenario_id"))
        scenario_ids.append(scenario_id)
        if not _valid_id(scenario_id):
            blockers.append(f"scenario_identity_missing_or_invalid:{index}")
        if _string(scenario.get("task_id")) not in set(task_ids):
            blockers.append(f"scenario_unknown_task:{index}")
        split = _string(scenario.get("split"))
        if split not in SPLITS:
            blockers.append(f"scenario_split_invalid:{index}")
        else:
            split_counts[split] += 1
        if _integer(scenario.get("seed"), minimum=0) is None:
            blockers.append(f"scenario_seed_missing_or_invalid:{index}")
        if not _digest(scenario.get("initial_condition_sha256")):
            blockers.append(f"scenario_initial_condition_digest_missing:{index}")
        axes = _mapping(scenario.get("generalization"))
        for axis in GENERALIZATION_AXES:
            value = _string(axes.get(axis))
            if value not in {"seen", "unseen"}:
                blockers.append(f"scenario_generalization_axis_invalid:{index}:{axis}")
            else:
                axis_values[axis].add(value)
    if len(scenario_ids) != len(set(scenario_ids)):
        blockers.append("duplicate_scenario_id")
    for split in SPLITS:
        if split_counts.get(split, 0) == 0:
            blockers.append(f"required_split_missing:{split}")
    for axis in GENERALIZATION_AXES:
        if axis_values.get(axis, set()) != {"seen", "unseen"}:
            blockers.append(f"seen_unseen_coverage_missing:{axis}")

    rollout = _mapping(spec.get("rollout_protocol"))
    count = _integer(rollout.get("fixed_rollouts_per_scenario_policy"), minimum=1)
    if count is None:
        blockers.append("fixed_rollout_count_missing_or_invalid")
    if rollout.get("cherry_picking_prohibited") is not True:
        blockers.append("cherry_picking_must_be_prohibited")
    if rollout.get("result_replacement_prohibited") is not True:
        blockers.append("result_replacement_must_be_prohibited")
    if rollout.get("infrastructure_retries_scored_as_new_attempts") is not True:
        blockers.append("infrastructure_retries_must_be_scored_as_new_attempts")

    evidence = _mapping(spec.get("required_episode_evidence"))
    for key in ("video", "action_trace", "evaluator_output"):
        if evidence.get(key) is not True:
            blockers.append(f"required_episode_evidence_missing:{key}")
    if evidence.get("content_digests") is not True:
        blockers.append("episode_evidence_content_digests_required")

    scoring = _mapping(spec.get("scoring"))
    required_metrics = {
        "full_task_success",
        "partial_progress",
        "efficiency",
        "safety_interventions",
        "evaluator_abstention",
    }
    metrics = set(_string(item) for item in scoring.get("metrics", []))
    if metrics != required_metrics:
        blockers.append("scoring_metrics_missing_or_mismatched")
    if scoring.get("confidence_intervals_required") is not True:
        blockers.append("confidence_intervals_must_be_required")
    if scoring.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES:
        blockers.append("bootstrap_replicates_must_equal_10000")

    baselines, baselines_valid = _strict_rows(spec.get("public_baselines"))
    candidates, candidates_valid = _strict_rows(spec.get("candidate_policies", []))
    if not baselines_valid or not baselines:
        blockers.append("public_baseline_registry_missing_or_invalid")
    if not candidates_valid:
        blockers.append("candidate_policy_registry_invalid")
    for row in baselines:
        blockers.extend(f"public_baseline:{error}" for error in _policy_errors(row, public_baseline=True))
    for row in candidates:
        blockers.extend(f"candidate_policy:{error}" for error in _policy_errors(row, public_baseline=False))
    policy_ids = [_string(row.get("policy_id")) for row in baselines + candidates]
    checkpoint_ids = [_digest(row.get("checkpoint_sha256")) for row in baselines + candidates]
    if len(policy_ids) != len(set(policy_ids)):
        blockers.append("duplicate_policy_id")
    if len(checkpoint_ids) != len(set(checkpoint_ids)):
        blockers.append("duplicate_checkpoint_digest")
    if len(policy_ids) < 2:
        blockers.append("benchmark_requires_at_least_two_policies")

    blockers = sorted(set(blockers))
    return {
        "schema_version": "blueprint_benchmark_spec_validation.v1",
        "status": "validated" if not blockers else "blocked",
        "benchmark_id": spec.get("benchmark_id"),
        "task_count": len(tasks),
        "scenario_count": len(scenarios),
        "policy_count": len(policy_ids),
        "split_counts": {split: split_counts.get(split, 0) for split in SPLITS},
        "blockers": blockers,
    }


def _chmod_private(path: Path) -> None:
    path.chmod(0o600)


def compile_benchmark_protocol(
    spec: Mapping[str, Any], *, output_dir: Path, generated_at: str | None = None
) -> dict[str, Any]:
    """Compile frozen private/public benchmark artifacts and a fixed run plan."""

    validation = validate_benchmark_spec(spec)
    if validation["blockers"]:
        raise ValueError("benchmark_spec_blocked:" + ",".join(validation["blockers"]))
    generated_at = generated_at or utc_now_iso()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = _rows(spec.get("scenarios"))
    baselines = _rows(spec.get("public_baselines"))
    candidates = _rows(spec.get("candidate_policies", []))
    policies = baselines + candidates
    rollout_count = int(_mapping(spec.get("rollout_protocol"))["fixed_rollouts_per_scenario_policy"])
    environment_sha256 = canonical_sha256(_mapping(spec.get("environment")))
    evaluator_runtime_sha256 = canonical_sha256(_mapping(spec.get("evaluator_runtime")))

    split_payload = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "benchmark_id": spec["benchmark_id"],
        "benchmark_version": spec["benchmark_version"],
        "generated_at": generated_at,
        "frozen": True,
        "preregistration_sha256": _digest(spec.get("preregistration_sha256")),
        "generalization_axes": list(GENERALIZATION_AXES),
        "scenarios": scenarios,
    }
    split_payload["frozen_split_sha256"] = canonical_sha256(split_payload)
    split_path = output_dir / "benchmark_split_manifest.private.json"
    write_json(split_path, split_payload)
    _chmod_private(split_path)

    registry = {
        "schema_version": BASELINE_REGISTRY_SCHEMA_VERSION,
        "benchmark_id": spec["benchmark_id"],
        "benchmark_version": spec["benchmark_version"],
        "generated_at": generated_at,
        "baselines": baselines,
    }
    registry["registry_sha256"] = canonical_sha256(registry)
    registry_path = output_dir / "public_baseline_registry.json"
    write_json(registry_path, registry)

    attempts: list[dict[str, Any]] = []
    for scenario in scenarios:
        if scenario["split"] not in SCORED_SPLITS:
            continue
        for rollout_index in range(rollout_count):
            for policy in policies:
                attempt_key = {
                    "benchmark_id": spec["benchmark_id"],
                    "benchmark_version": spec["benchmark_version"],
                    "policy_id": policy["policy_id"],
                    "checkpoint_sha256": _digest(policy["checkpoint_sha256"]),
                    "scenario_id": scenario["scenario_id"],
                    "split": scenario["split"],
                    "seed": scenario["seed"],
                    "rollout_index": rollout_index,
                    "initial_condition_sha256": _digest(scenario["initial_condition_sha256"]),
                    "environment_sha256": environment_sha256,
                    "evaluator_runtime_sha256": evaluator_runtime_sha256,
                }
                attempts.append(
                    {
                        "attempt_id": "bench-" + canonical_sha256(attempt_key)[:24],
                        **attempt_key,
                        "task_id": scenario["task_id"],
                        "generalization": dict(scenario["generalization"]),
                        "selected_for_reporting": True,
                        "replacement_allowed": False,
                    }
                )
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "benchmark_id": spec["benchmark_id"],
        "benchmark_version": spec["benchmark_version"],
        "generated_at": generated_at,
        "frozen_split_sha256": split_payload["frozen_split_sha256"],
        "baseline_registry_sha256": registry["registry_sha256"],
        "environment_sha256": environment_sha256,
        "evaluator_runtime_sha256": evaluator_runtime_sha256,
        "fixed_rollouts_per_scenario_policy": rollout_count,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "anti_cherry_picking": {
            "exact_attempt_coverage_required": True,
            "result_replacement_prohibited": True,
            "infrastructure_failures_retained": True,
            "all_scheduled_attempts_selected_for_reporting": True,
        },
    }
    plan["execution_plan_sha256"] = canonical_sha256(plan)
    plan_path = output_dir / "benchmark_execution_plan.private.json"
    write_json(plan_path, plan)
    _chmod_private(plan_path)

    split_counts = {split: sum(row["split"] == split for row in scenarios) for split in SPLITS}
    generalization_counts = {
        axis: {
            label: sum(row["generalization"][axis] == label for row in scenarios)
            for label in ("seen", "unseen")
        }
        for axis in GENERALIZATION_AXES
    }
    card = {
        "schema_version": CARD_SCHEMA_VERSION,
        "benchmark_id": spec["benchmark_id"],
        "benchmark_version": spec["benchmark_version"],
        "protocol_version": spec["protocol_version"],
        "title": spec.get("title"),
        "description": spec.get("description"),
        "generated_at": generated_at,
        "frozen": True,
        "frozen_split_sha256": split_payload["frozen_split_sha256"],
        "preregistration_sha256": _digest(spec.get("preregistration_sha256")),
        "tasks": _rows(spec.get("tasks")),
        "action_space": _mapping(spec.get("action_space")),
        "environment_summary": {
            "site_id": _mapping(spec.get("environment")).get("site_id"),
            "representation_type": _mapping(spec.get("environment")).get(
                "representation_type"
            ),
            "physics_authority": _mapping(spec.get("environment")).get(
                "physics_authority"
            ),
            "same_site_capture": _mapping(spec.get("environment")).get(
                "same_site_capture"
            ),
            "environment_sha256": environment_sha256,
        },
        "evaluator_runtime_summary": {
            "evaluator_id": _mapping(spec.get("evaluator_runtime")).get(
                "evaluator_id"
            ),
            "evaluator_version": _mapping(spec.get("evaluator_runtime")).get(
                "evaluator_version"
            ),
            "evaluator_runtime_sha256": evaluator_runtime_sha256,
        },
        "rollout_protocol": _mapping(spec.get("rollout_protocol")),
        "scoring": _mapping(spec.get("scoring")),
        "required_episode_evidence": _mapping(spec.get("required_episode_evidence")),
        "split_summary": {
            "counts": split_counts,
            "generalization_counts": generalization_counts,
            "hidden_test_identifiers_redacted": True,
            "hidden_test_content_digest_committed": True,
        },
        "public_baseline_registry_sha256": registry["registry_sha256"],
        "public_baseline_count": len(baselines),
        "candidate_policy_count": len(candidates),
        "claim_boundary": {
            "benchmark_protocol_is_not_execution_proof": True,
            "hidden_scenario_identity_not_public": True,
            "simulator_results_are_not_real_world_results": True,
            "external_rank_fidelity_requires_independent_anchor_results": True,
        },
    }
    card["benchmark_card_sha256"] = canonical_sha256(card)
    card_path = output_dir / "benchmark_card.json"
    write_json(card_path, card)

    task_pack = {
        "adapter_id": "benchmark_task_scenario_pack",
        "adapter_version": "1",
        "pack_id": f"{spec['benchmark_id']}:{spec['benchmark_version']}",
        "tasks": [{"task_id": row["task_id"]} for row in _rows(spec.get("tasks"))],
        "scenarios": [
            {"scenario_id": row["scenario_id"], "task_id": row["task_id"]}
            for row in scenarios
            if row["split"] in SCORED_SPLITS
        ],
        "frozen_split_sha256": split_payload["frozen_split_sha256"],
        "execution_plan_sha256": plan["execution_plan_sha256"],
    }
    task_pack_path = output_dir / "evaluation_run_task_scenario_pack.private.json"
    write_json(task_pack_path, task_pack)
    _chmod_private(task_pack_path)

    projection = _webapp_projection(card=card, report=None, external_report=None)
    projection_path = output_dir / "webapp_benchmark_projection.json"
    write_json(projection_path, projection)
    return {
        "validation": validation,
        "benchmark_card": card,
        "split_manifest": split_payload,
        "baseline_registry": registry,
        "execution_plan": plan,
        "evaluation_run_task_scenario_pack": task_pack,
        "webapp_projection": projection,
        "artifacts": {
            "benchmark_card": str(card_path),
            "private_split_manifest": str(split_path),
            "public_baseline_registry": str(registry_path),
            "private_execution_plan": str(plan_path),
            "private_evaluation_run_task_scenario_pack": str(task_pack_path),
            "webapp_projection": str(projection_path),
        },
    }


def _metric_value(row: Mapping[str, Any], metric: str) -> float | None:
    status = _string(row.get("status"))
    abstained = row.get("evaluator_abstained") is True or status == "abstained"
    if metric == "coverage":
        return 1.0 if status in {"completed", "abstained"} else 0.0
    if metric == "infrastructure_failure_rate":
        return 1.0 if status == "infrastructure_failed" else 0.0
    if metric == "evaluator_abstention":
        return 1.0 if abstained else 0.0
    if status != "completed" or abstained:
        return None
    if metric == "full_task_success":
        value = row.get("full_task_success")
        return float(value) if isinstance(value, bool) else None
    if metric == "partial_progress":
        return _number(row.get("partial_progress"))
    if metric == "efficiency":
        return _number(_mapping(row.get("efficiency")).get("normalized_score"))
    safety = _mapping(row.get("safety"))
    if metric == "safety_interventions":
        return _number(safety.get("intervention_count"))
    if metric == "collision_free_rate":
        collision_count = _integer(safety.get("collision_count"), minimum=0)
        return None if collision_count is None else float(collision_count == 0)
    return None


def _aggregate_metric(
    rows: Sequence[Mapping[str, Any]], *, metric: str, seed: int
) -> dict[str, Any]:
    values = [value for row in rows if (value := _metric_value(row, metric)) is not None]
    estimate = sum(values) / len(values) if values else None
    samples: list[float] = []
    if values:
        rng = random.Random(seed)
        for _ in range(_BOOTSTRAP_EXECUTION_REPLICATES):
            sample = [rng.choice(values) for _ in values]
            samples.append(sum(sample) / len(sample))
    return {
        "estimate": round(estimate, 6) if estimate is not None else None,
        "confidence_interval_95": [
            round(lower, 6) if (lower := _percentile(samples, 0.025)) is not None else None,
            round(upper, 6) if (upper := _percentile(samples, 0.975)) is not None else None,
        ],
        "sample_count": len(values),
        "method": "episode_percentile_bootstrap.v1",
        "bootstrap_replicates": _BOOTSTRAP_EXECUTION_REPLICATES,
    }


def _metric_bundle(rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    metrics = (
        "full_task_success",
        "partial_progress",
        "efficiency",
        "safety_interventions",
        "collision_free_rate",
        "evaluator_abstention",
        "coverage",
        "infrastructure_failure_rate",
    )
    return {
        metric: _aggregate_metric(rows, metric=metric, seed=seed + index * 7919)
        for index, metric in enumerate(metrics)
    }


def _result_errors(row: Mapping[str, Any], *, index: int) -> list[str]:
    errors: list[str] = []
    status = _string(row.get("status"))
    if status not in {"completed", "abstained", "infrastructure_failed"}:
        errors.append(f"result_status_invalid:{index}")
    if row.get("selected_for_reporting") is not True:
        errors.append(f"result_not_selected_for_reporting:{index}")
    if row.get("replacement_attempt") is not False:
        errors.append(f"replacement_attempt_forbidden:{index}")
    evidence = _mapping(row.get("evidence"))
    for key in ("video", "action_trace", "evaluator_output"):
        if not _artifact_ref_valid(evidence.get(key)):
            errors.append(f"result_evidence_missing_or_invalid:{index}:{key}")
    if status == "completed":
        if not isinstance(row.get("full_task_success"), bool):
            errors.append(f"result_full_task_success_missing:{index}")
        partial = _number(row.get("partial_progress"))
        if partial is None or not 0 <= partial <= 1:
            errors.append(f"result_partial_progress_invalid:{index}")
        efficiency = _mapping(row.get("efficiency"))
        score = _number(efficiency.get("normalized_score"))
        if score is None or not 0 <= score <= 1:
            errors.append(f"result_efficiency_score_invalid:{index}")
        for field in ("duration_seconds", "path_length_m"):
            value = _number(efficiency.get(field))
            if value is None or value < 0:
                errors.append(f"result_efficiency_field_invalid:{index}:{field}")
        safety = _mapping(row.get("safety"))
        for field in ("intervention_count", "collision_count", "unsafe_event_count"):
            if _integer(safety.get(field), minimum=0) is None:
                errors.append(f"result_safety_field_invalid:{index}:{field}")
        if row.get("evaluator_abstained") is not False:
            errors.append(f"completed_result_cannot_abstain:{index}")
    if status == "abstained":
        if row.get("evaluator_abstained") is not True:
            errors.append(f"abstained_result_flag_missing:{index}")
        if not _string(row.get("abstention_reason")):
            errors.append(f"abstained_result_reason_missing:{index}")
    return errors


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    offset = 0
    while offset < len(ordered):
        end = offset + 1
        while end < len(ordered) and ordered[end][0] == ordered[offset][0]:
            end += 1
        average_rank = (offset + 1 + end) / 2.0
        for _, index in ordered[offset:end]:
            ranks[index] = average_rank
        offset = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_scale = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_scale = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    denominator = left_scale * right_scale
    return numerator / denominator if denominator else None


def _kendall_tau_b(left: Sequence[float], right: Sequence[float]) -> float | None:
    concordant = discordant = left_ties = right_ties = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            a = left[first] - left[second]
            b = right[first] - right[second]
            if a == 0 and b == 0:
                continue
            if a == 0:
                left_ties += 1
            elif b == 0:
                right_ties += 1
            elif a * b > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + left_ties)
        * (concordant + discordant + right_ties)
    )
    return (concordant - discordant) / denominator if denominator else None


def _pairwise_accuracy(predicted: Sequence[float], reference: Sequence[float]) -> float | None:
    correct = total = 0
    for first in range(len(predicted)):
        for second in range(first + 1, len(predicted)):
            external_delta = reference[first] - reference[second]
            if external_delta == 0:
                continue
            predicted_delta = predicted[first] - predicted[second]
            total += 1
            if predicted_delta * external_delta > 0:
                correct += 1
            elif predicted_delta == 0:
                correct += 0.5
    return correct / total if total else None


def _mmrv(predicted: Sequence[float], reference: Sequence[float]) -> float | None:
    if len(predicted) < 2 or len(predicted) != len(reference):
        return None
    worst: list[float] = []
    for first in range(len(predicted)):
        violations = [0.0]
        for second in range(len(predicted)):
            if first == second:
                continue
            real_delta = reference[first] - reference[second]
            predicted_delta = predicted[first] - predicted[second]
            if real_delta * predicted_delta < 0 or (real_delta != 0 and predicted_delta == 0):
                violations.append(abs(real_delta))
        worst.append(max(violations))
    return sum(worst) / len(worst)


def _external_metrics(predicted: Sequence[float], reference: Sequence[float]) -> dict[str, float | None]:
    return {
        "pearson": _pearson(predicted, reference),
        "spearman": _pearson(_ranks(predicted), _ranks(reference)),
        "kendall_tau_b": _kendall_tau_b(predicted, reference),
        "pairwise_ordering_accuracy": _pairwise_accuracy(predicted, reference),
        "mmrv": _mmrv(predicted, reference),
    }


def external_rank_metrics(
    predicted: Sequence[float], reference: Sequence[float]
) -> dict[str, float | None]:
    """Return Blueprint's canonical policy-rank agreement metrics.

    This public wrapper keeps evaluator studies and benchmark reports on the
    same Pearson, Spearman, Kendall tau-b, pairwise-ordering, and MMRV
    definitions without requiring those studies to reach into private helpers.
    """

    return _external_metrics(predicted, reference)


def build_external_rank_fidelity_report(
    *,
    reference: Mapping[str, Any],
    policy_aggregates: Sequence[Mapping[str, Any]],
    policy_registry: Sequence[Mapping[str, Any]],
    seed: int,
) -> dict[str, Any]:
    blockers: list[str] = []
    if reference.get("schema_version") != EXTERNAL_REFERENCE_SCHEMA_VERSION:
        blockers.append("external_reference_schema_missing_or_unsupported")
    if reference.get("reference_type") not in {"real_robot", "simulator", "world_model"}:
        blockers.append("external_reference_type_invalid")
    if reference.get("site_alignment") not in {"same_site", "different_site", "aggregate_only"}:
        blockers.append("external_reference_site_alignment_invalid")
    for field in ("source_artifact_sha256", "task_mapping_sha256"):
        if not _digest(reference.get(field)):
            blockers.append(f"external_reference_digest_missing:{field}")
    if not _string(reference.get("source_uri")):
        blockers.append("external_reference_source_uri_missing")
    if reference.get("independently_accepted") is not True:
        blockers.append("external_reference_not_independently_accepted")
    external_rows, rows_valid = _strict_rows(reference.get("policy_results"))
    if not rows_valid:
        blockers.append("external_policy_results_invalid")
    aggregate_by_policy = {str(row.get("policy_id")): row for row in policy_aggregates}
    registry_by_policy = {str(row.get("policy_id")): row for row in policy_registry}
    matched: list[dict[str, Any]] = []
    for index, row in enumerate(external_rows):
        policy_id = _string(row.get("policy_id"))
        score = _number(row.get("score"))
        if score is None:
            blockers.append(f"external_policy_score_invalid:{index}")
            continue
        registry = registry_by_policy.get(policy_id)
        aggregate = aggregate_by_policy.get(policy_id)
        if not registry or not aggregate:
            continue
        if _digest(row.get("checkpoint_sha256")) != _digest(registry.get("checkpoint_sha256")):
            blockers.append(f"external_checkpoint_digest_mismatch:{policy_id}")
            continue
        predicted = _number(
            _mapping(_mapping(aggregate.get("metrics")).get("full_task_success")).get("estimate")
        )
        if predicted is None:
            blockers.append(f"blueprint_policy_score_missing:{policy_id}")
            continue
        matched.append(
            {
                "policy_id": policy_id,
                "checkpoint_sha256": _digest(registry.get("checkpoint_sha256")),
                "blueprint_score": predicted,
                "external_score": score,
            }
        )
    if len(matched) < 3:
        blockers.append("external_rank_fidelity_requires_three_exact_checkpoint_matches")
    predicted_values = [row["blueprint_score"] for row in matched]
    reference_values = [row["external_score"] for row in matched]
    estimates = _external_metrics(predicted_values, reference_values) if len(matched) >= 2 else {}
    bootstrap: dict[str, list[float]] = defaultdict(list)
    if len(matched) >= 3:
        rng = random.Random(seed)
        for _ in range(_BOOTSTRAP_EXECUTION_REPLICATES):
            sample = [rng.choice(matched) for _ in matched]
            sample_predicted = [row["blueprint_score"] for row in sample]
            sample_reference = [row["external_score"] for row in sample]
            for metric, value in _external_metrics(sample_predicted, sample_reference).items():
                if value is not None and math.isfinite(value):
                    bootstrap[metric].append(value)
    metrics = {
        metric: {
            "estimate": round(value, 6) if value is not None else None,
            "confidence_interval_95": [
                round(lower, 6) if (lower := _percentile(bootstrap.get(metric, []), 0.025)) is not None else None,
                round(upper, 6) if (upper := _percentile(bootstrap.get(metric, []), 0.975)) is not None else None,
            ],
            "sample_count": len(matched),
            "method": "exact_checkpoint_policy_bootstrap.v1",
            "bootstrap_replicates": _BOOTSTRAP_EXECUTION_REPLICATES,
        }
        for metric, value in estimates.items()
    }
    reference_type = _string(reference.get("reference_type"))
    site_alignment = _string(reference.get("site_alignment"))
    if reference_type == "real_robot" and site_alignment == "same_site":
        measurement_scope = "same_site_real_robot_rank_fidelity"
    elif reference_type == "real_robot":
        measurement_scope = "cross_site_real_robot_rank_concordance"
    else:
        measurement_scope = "cross_evaluator_concordance"
    blockers = sorted(set(blockers))
    measured = not blockers
    same_site_real_robot_fidelity = (
        measured and reference_type == "real_robot" and site_alignment == "same_site"
    )
    cross_site_real_robot_concordance = (
        measured and reference_type == "real_robot" and site_alignment == "different_site"
    )
    return {
        "schema_version": EXTERNAL_REPORT_SCHEMA_VERSION,
        "status": "measured" if not blockers else "blocked",
        "measurement_scope": measurement_scope,
        "reference_id": reference.get("reference_id"),
        "reference_type": reference_type,
        "site_alignment": site_alignment,
        "independently_accepted": reference.get("independently_accepted") is True,
        "source_uri": reference.get("source_uri"),
        "source_artifact_sha256": _digest(reference.get("source_artifact_sha256")),
        "task_mapping_sha256": _digest(reference.get("task_mapping_sha256")),
        "matched_policies": matched,
        "metrics": metrics,
        "blockers": blockers,
        "claim_boundary": {
            "different_site_comparison_is_not_site_specific_validation": site_alignment != "same_site",
            "simulator_agreement_is_not_real_world_validation": reference_type != "real_robot",
            "exact_checkpoint_matching_required": True,
            PUBLIC_CLAIM_UPGRADE_KEY: False,
            "scoped_external_comparison_measured": measured,
            "rank_fidelity_result_proven": same_site_real_robot_fidelity,
            "cross_site_rank_concordance_proven": cross_site_real_robot_concordance,
        },
    }


def _webapp_projection(
    *, card: Mapping[str, Any], report: Mapping[str, Any] | None, external_report: Mapping[str, Any] | None
) -> dict[str, Any]:
    aggregates = _rows((report or {}).get("policy_aggregates"))
    safe_aggregates = [
        {
            "policy_id": row.get("policy_id"),
            "checkpoint_sha256": row.get("checkpoint_sha256"),
            "metrics": row.get("metrics"),
        }
        for row in aggregates
    ]
    report_breakdowns = _mapping((report or {}).get("breakdowns"))
    split_breakdowns = _mapping(report_breakdowns.get("split"))
    generalization_breakdowns = _mapping(report_breakdowns.get("generalization"))
    safe_breakdowns = {
        "split": {
            split: _mapping(split_breakdowns.get(split))
            for split in SCORED_SPLITS
            if split in split_breakdowns
        },
        "generalization": {
            axis: {
                label: _mapping(_mapping(generalization_breakdowns.get(axis)).get(label))
                for label in ("seen", "unseen")
                if label in _mapping(generalization_breakdowns.get(axis))
            }
            for axis in GENERALIZATION_AXES
            if axis in generalization_breakdowns
        },
    }
    return {
        "schema_version": WEBAPP_PROJECTION_SCHEMA_VERSION,
        "benchmark_id": card.get("benchmark_id"),
        "benchmark_version": card.get("benchmark_version"),
        "benchmark_card_sha256": card.get("benchmark_card_sha256"),
        "status": (report or {}).get("status", "planned"),
        "split_summary": card.get("split_summary"),
        "rollout_protocol": card.get("rollout_protocol"),
        "scoring": card.get("scoring"),
        "environment_summary": card.get("environment_summary"),
        "evaluator_runtime_summary": card.get("evaluator_runtime_summary"),
        "policy_aggregates": safe_aggregates,
        "breakdowns": safe_breakdowns,
        "evidence_summary": (report or {}).get("evidence_summary"),
        "evidence_index_sha256": (report or {}).get("evidence_index_sha256"),
        "external_rank_fidelity": external_report,
        "hidden_scenario_identifiers_included": False,
        "claim_boundary": {
            "owner_system_artifacts_required": True,
            "different_site_comparison_is_not_site_specific_validation": True,
            PUBLIC_CLAIM_UPGRADE_KEY: False,
        },
    }


def build_benchmark_report(
    *,
    spec: Mapping[str, Any],
    plan: Mapping[str, Any],
    results: Mapping[str, Any],
    external_reference: Mapping[str, Any] | None = None,
    seed: int = 20260721,
) -> dict[str, Any]:
    """Validate complete attempt coverage and compute all benchmark aggregates."""

    blockers: list[str] = []
    validation = validate_benchmark_spec(spec)
    blockers.extend(f"benchmark_spec:{item}" for item in validation["blockers"])
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        blockers.append("benchmark_execution_plan_schema_invalid")
    if results.get("schema_version") != RESULTS_SCHEMA_VERSION:
        blockers.append("benchmark_results_schema_invalid")
    if _string(plan.get("benchmark_id")) != _string(spec.get("benchmark_id")):
        blockers.append("benchmark_plan_identity_mismatch")
    if _digest(plan.get("environment_sha256")) != canonical_sha256(
        _mapping(spec.get("environment"))
    ):
        blockers.append("benchmark_plan_environment_mismatch")
    if _digest(plan.get("evaluator_runtime_sha256")) != canonical_sha256(
        _mapping(spec.get("evaluator_runtime"))
    ):
        blockers.append("benchmark_plan_evaluator_runtime_mismatch")
    plan_rows, plan_valid = _strict_rows(plan.get("attempts"))
    result_rows, results_valid = _strict_rows(results.get("attempts"))
    if not plan_valid:
        blockers.append("benchmark_plan_attempts_invalid")
    if not results_valid:
        blockers.append("benchmark_result_attempts_invalid")
    expected = {_string(row.get("attempt_id")): row for row in plan_rows}
    observed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(result_rows):
        attempt_id = _string(row.get("attempt_id"))
        if not attempt_id or attempt_id not in expected:
            blockers.append(f"unknown_result_attempt:{index}")
        if attempt_id in observed:
            blockers.append(f"duplicate_result_attempt:{index}")
        observed[attempt_id] = row
        blockers.extend(_result_errors(row, index=index))
        scheduled = expected.get(attempt_id, {})
        for field in (
            "policy_id",
            "checkpoint_sha256",
            "scenario_id",
            "split",
            "seed",
            "rollout_index",
            "initial_condition_sha256",
            "environment_sha256",
            "evaluator_runtime_sha256",
        ):
            left = _digest(row.get(field)) if field.endswith("sha256") else row.get(field)
            right = _digest(scheduled.get(field)) if field.endswith("sha256") else scheduled.get(field)
            if left != right:
                blockers.append(f"result_attempt_binding_mismatch:{index}:{field}")
    if set(observed) != set(expected):
        blockers.append("result_attempt_coverage_not_exact")

    enriched_rows = [
        {
            **row,
            "generalization": expected.get(_string(row.get("attempt_id")), {}).get(
                "generalization", {}
            ),
            "task_id": expected.get(_string(row.get("attempt_id")), {}).get("task_id"),
        }
        for row in result_rows
    ]
    registry = _rows(spec.get("public_baselines")) + _rows(spec.get("candidate_policies", []))
    registry_by_id = {_string(row.get("policy_id")): row for row in registry}
    policy_aggregates: list[dict[str, Any]] = []
    for policy_index, policy_id in enumerate(sorted(registry_by_id)):
        rows = [row for row in enriched_rows if _string(row.get("policy_id")) == policy_id]
        policy_aggregates.append(
            {
                "policy_id": policy_id,
                "checkpoint_sha256": _digest(registry_by_id[policy_id].get("checkpoint_sha256")),
                "metrics": _metric_bundle(rows, seed=seed + policy_index * 104729),
            }
        )

    breakdowns: dict[str, Any] = {"split": {}, "generalization": {}}
    for split in SCORED_SPLITS:
        split_rows = [row for row in enriched_rows if row.get("split") == split]
        breakdowns["split"][split] = {
            policy_id: _metric_bundle(
                [row for row in split_rows if row.get("policy_id") == policy_id],
                seed=seed + int(canonical_sha256(["split", split, policy_id])[:8], 16),
            )
            for policy_id in sorted(registry_by_id)
        }
    for axis in GENERALIZATION_AXES:
        breakdowns["generalization"][axis] = {}
        for label in ("seen", "unseen"):
            axis_rows = [
                row
                for row in enriched_rows
                if _mapping(row.get("generalization")).get(axis) == label
            ]
            breakdowns["generalization"][axis][label] = {
                policy_id: _metric_bundle(
                    [row for row in axis_rows if row.get("policy_id") == policy_id],
                    seed=seed
                    + int(canonical_sha256(["axis", axis, label, policy_id])[:8], 16),
                )
                for policy_id in sorted(registry_by_id)
            }

    external_report = None
    if external_reference is not None:
        external_report = build_external_rank_fidelity_report(
            reference=external_reference,
            policy_aggregates=policy_aggregates,
            policy_registry=registry,
            seed=seed,
        )
    evidence_index = _private_evidence_index(
        spec=spec,
        plan=plan,
        results=results,
    )
    evidence_rows = _rows(evidence_index.get("attempt_evidence"))
    evidence_summary = {
        "attempt_count": len(evidence_rows),
        "video_count": sum(
            _artifact_ref_valid(_mapping(row.get("evidence")).get("video"))
            for row in evidence_rows
        ),
        "action_trace_count": sum(
            _artifact_ref_valid(_mapping(row.get("evidence")).get("action_trace"))
            for row in evidence_rows
        ),
        "evaluator_output_count": sum(
            _artifact_ref_valid(_mapping(row.get("evidence")).get("evaluator_output"))
            for row in evidence_rows
        ),
        "all_attempts_digest_bound": all(
            all(
                _artifact_ref_valid(_mapping(row.get("evidence")).get(key))
                for key in ("video", "action_trace", "evaluator_output")
            )
            for row in evidence_rows
        )
        and len(evidence_rows) == len(expected),
    }
    blockers = sorted(set(blockers))
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "benchmark_id": spec.get("benchmark_id"),
        "benchmark_version": spec.get("benchmark_version"),
        "status": "complete" if not blockers else "blocked",
        "execution_plan_sha256": plan.get("execution_plan_sha256"),
        "attempt_count_expected": len(expected),
        "attempt_count_observed": len(observed),
        "anti_cherry_picking_verified": set(observed) == set(expected) and len(observed) == len(result_rows),
        "policy_aggregates": policy_aggregates,
        "breakdowns": breakdowns,
        "evidence_summary": evidence_summary,
        "evidence_index_sha256": evidence_index["evidence_index_sha256"],
        "external_rank_fidelity": external_report,
        "blockers": blockers,
        "claim_boundary": {
            "simulator_results_are_not_real_world_results": True,
            "external_report_does_not_auto_upgrade_public_claims": True,
            "physical_robot_readiness_proven": False,
            "safety_validation_proven": False,
            PUBLIC_CLAIM_UPGRADE_KEY: False,
        },
    }
    report["benchmark_report_sha256"] = canonical_sha256(report)
    return report


def _private_evidence_index(
    *,
    spec: Mapping[str, Any],
    plan: Mapping[str, Any],
    results: Mapping[str, Any],
) -> dict[str, Any]:
    attempts_by_id = {
        _string(row.get("attempt_id")): row for row in _rows(plan.get("attempts"))
    }
    attempt_evidence: list[dict[str, Any]] = []
    for result in _rows(results.get("attempts")):
        attempt_id = _string(result.get("attempt_id"))
        scheduled = _mapping(attempts_by_id.get(attempt_id))
        attempt_evidence.append(
            {
                "attempt_id": attempt_id,
                "policy_id": scheduled.get("policy_id") or result.get("policy_id"),
                "checkpoint_sha256": scheduled.get("checkpoint_sha256")
                or result.get("checkpoint_sha256"),
                "scenario_id": scheduled.get("scenario_id") or result.get("scenario_id"),
                "split": scheduled.get("split") or result.get("split"),
                "seed": scheduled.get("seed") if scheduled else result.get("seed"),
                "rollout_index": scheduled.get("rollout_index")
                if scheduled
                else result.get("rollout_index"),
                "status": result.get("status"),
                "evidence": _mapping(result.get("evidence")),
            }
        )
    payload = {
        "schema_version": EVIDENCE_INDEX_SCHEMA_VERSION,
        "benchmark_id": spec.get("benchmark_id"),
        "benchmark_version": spec.get("benchmark_version"),
        "execution_plan_sha256": plan.get("execution_plan_sha256"),
        "private": True,
        "attempt_evidence": attempt_evidence,
        "claim_boundary": {
            "contains_hidden_scenario_material": True,
            "webapp_export_allowed": False,
            "public_export_allowed": False,
        },
    }
    payload["evidence_index_sha256"] = canonical_sha256(payload)
    return payload


def write_benchmark_report(
    *,
    spec: Mapping[str, Any],
    plan: Mapping[str, Any],
    results: Mapping[str, Any],
    output_dir: Path,
    external_reference: Mapping[str, Any] | None = None,
    seed: int = 20260721,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_benchmark_report(
        spec=spec,
        plan=plan,
        results=results,
        external_reference=external_reference,
        seed=seed,
    )
    report_path = output_dir / "benchmark_report.json"
    write_json(report_path, report)
    evidence_index_path = output_dir / "benchmark_evidence_index.private.json"
    write_json(
        evidence_index_path,
        _private_evidence_index(spec=spec, plan=plan, results=results),
    )
    _chmod_private(evidence_index_path)
    external = report.get("external_rank_fidelity")
    if isinstance(external, Mapping):
        write_json(output_dir / "external_rank_fidelity_report.json", external)
    card_path = output_dir / "benchmark_card.json"
    card = _mapping(read_json_any(card_path)) if card_path.is_file() else {}
    if not card:
        compiled = compile_benchmark_protocol(spec, output_dir=output_dir)
        card = compiled["benchmark_card"]
    projection = _webapp_projection(
        card=card,
        report=report,
        external_report=_mapping(external) if isinstance(external, Mapping) else None,
    )
    write_json(output_dir / "webapp_benchmark_projection.json", projection)
    return {"report": report, "webapp_projection": projection}


def _request_artifact_path(
    value: Any, *, allowed_root: Path, field: str
) -> tuple[Path | None, str | None]:
    uri = _string(value)
    if not uri:
        return None, f"{field}_missing"
    if uri.startswith("file://"):
        candidate = Path(uri[7:])
    elif "://" in uri:
        return None, f"{field}_requires_staged_local_artifact"
    else:
        candidate = Path(uri)
    if not candidate.is_absolute():
        candidate = allowed_root / candidate
    resolved = candidate.resolve()
    root = allowed_root.resolve()
    if not resolved.is_relative_to(root):
        return None, f"{field}_outside_allowed_root"
    if not resolved.is_file():
        return None, f"{field}_not_found"
    return resolved, None


def execute_benchmark_protocol_request(
    request: Mapping[str, Any], *, output_dir: Path, allowed_root: Path
) -> dict[str, Any]:
    """Compile/report a benchmark request without exposing private split material.

    Remote artifacts must first be staged beneath ``allowed_root`` by an
    authenticated owner-system adapter. This function never downloads URLs or
    resolves customer credentials.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = _mapping(request.get("benchmark_protocol_request"))
    mode = _string(protocol.get("mode") or "standard")
    status_path = output_dir / "benchmark_protocol_status.json"
    if mode != "benchmark_grade":
        status = {
            "schema_version": BENCHMARK_REQUEST_STATUS_SCHEMA_VERSION,
            "mode": "standard",
            "status": "not_requested",
            "blockers": [],
            "artifacts": {},
            "claim_boundary": {
                "benchmark_execution_proven": False,
                "private_split_material_exported": False,
                PUBLIC_CLAIM_UPGRADE_KEY: False,
            },
        }
        write_json(status_path, status)
        return status

    blockers: list[str] = []
    if protocol.get("schema_version") != "blueprint_benchmark_protocol_request.v1":
        blockers.append("benchmark_protocol_request_schema_missing_or_unsupported")
    for field in (
        "frozen_hidden_splits_required",
        "fixed_rollouts_required",
        "confidence_intervals_required",
        "exact_checkpoint_digests_required",
    ):
        if protocol.get(field) is not True:
            blockers.append(f"benchmark_protocol_requirement_missing:{field}")
    if protocol.get("private_split_material_allowed_in_webapp") is not False:
        blockers.append("private_split_material_must_not_enter_webapp")
    if protocol.get("scheduler_owner") != "BlueprintCapturePipeline":
        blockers.append("benchmark_scheduler_owner_must_be_pipeline")

    spec_path, path_error = _request_artifact_path(
        protocol.get("benchmark_spec_uri"),
        allowed_root=allowed_root,
        field="benchmark_spec_uri",
    )
    if path_error:
        blockers.append(path_error)
    compiled: dict[str, Any] | None = None
    report_outcome: dict[str, Any] | None = None
    if not blockers and spec_path is not None:
        spec = _load_mapping(spec_path)
        declared_digest = _digest(protocol.get("benchmark_spec_sha256"))
        if not declared_digest or declared_digest != canonical_sha256(spec):
            blockers.append("benchmark_spec_sha256_mismatch")
        else:
            try:
                compiled = compile_benchmark_protocol(spec, output_dir=output_dir)
            except ValueError:
                blockers.append("benchmark_spec_validation_failed")

    results_path: Path | None = None
    external_path: Path | None = None
    if compiled and _string(protocol.get("benchmark_results_uri")):
        results_path, results_error = _request_artifact_path(
            protocol.get("benchmark_results_uri"),
            allowed_root=allowed_root,
            field="benchmark_results_uri",
        )
        if results_error:
            blockers.append(results_error)
        if _string(protocol.get("external_reference_uri")):
            external_path, external_error = _request_artifact_path(
                protocol.get("external_reference_uri"),
                allowed_root=allowed_root,
                field="external_reference_uri",
            )
            if external_error:
                blockers.append(external_error)
        if not blockers and results_path is not None:
            report_outcome = write_benchmark_report(
                spec=_load_mapping(spec_path),
                plan=compiled["execution_plan"],
                results=_load_mapping(results_path),
                output_dir=output_dir,
                external_reference=(
                    _load_mapping(external_path) if external_path is not None else None
                ),
            )

    if blockers:
        request_status = "blocked"
    elif report_outcome:
        request_status = str(report_outcome["report"].get("status") or "blocked")
    else:
        request_status = "planned"
    artifacts = {
        key: str(Path(value).relative_to(output_dir))
        for key, value in (compiled or {}).get("artifacts", {}).items()
    }
    if report_outcome:
        artifacts.update(
            {
                "benchmark_report": "benchmark_report.json",
                "private_evidence_index": "benchmark_evidence_index.private.json",
                "webapp_projection": "webapp_benchmark_projection.json",
            }
        )
        if report_outcome["report"].get("external_rank_fidelity"):
            artifacts["external_rank_fidelity_report"] = (
                "external_rank_fidelity_report.json"
            )
    status = {
        "schema_version": BENCHMARK_REQUEST_STATUS_SCHEMA_VERSION,
        "mode": "benchmark_grade",
        "status": request_status,
        "blockers": sorted(set(blockers)),
        "artifacts": artifacts,
        "benchmark_id": (compiled or {}).get("benchmark_card", {}).get("benchmark_id"),
        "benchmark_version": (compiled or {}).get("benchmark_card", {}).get(
            "benchmark_version"
        ),
        "claim_boundary": {
            "benchmark_execution_proven": request_status == "complete",
            "private_split_material_exported": False,
            "webapp_projection_contains_private_split_material": False,
            PUBLIC_CLAIM_UPGRADE_KEY: False,
        },
    }
    write_json(status_path, status)
    return status


def _load_mapping(path: str | Path) -> dict[str, Any]:
    payload = read_json_any(Path(path))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected_json_object:{path}")
    return dict(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--spec", required=True)
    compile_parser.add_argument("--output-dir", required=True)
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--spec", required=True)
    report_parser.add_argument("--plan", required=True)
    report_parser.add_argument("--results", required=True)
    report_parser.add_argument("--output-dir", required=True)
    report_parser.add_argument("--external-reference")
    report_parser.add_argument("--bootstrap-seed", type=int, default=20260721)
    args = parser.parse_args(argv)
    if args.command == "compile":
        compile_benchmark_protocol(
            _load_mapping(args.spec), output_dir=Path(args.output_dir)
        )
        return 0
    outcome = write_benchmark_report(
        spec=_load_mapping(args.spec),
        plan=_load_mapping(args.plan),
        results=_load_mapping(args.results),
        output_dir=Path(args.output_dir),
        external_reference=(
            _load_mapping(args.external_reference) if args.external_reference else None
        ),
        seed=args.bootstrap_seed,
    )
    return 0 if outcome["report"]["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
