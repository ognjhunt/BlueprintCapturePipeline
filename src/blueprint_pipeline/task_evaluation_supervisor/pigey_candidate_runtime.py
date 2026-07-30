"""Frozen external-checkout runtime for Pigey's public LIBERO harness.

Pigey remains third-party candidate code. Blueprint invokes an exact checkout,
normalizes its public ``trial.json`` artifacts, and never accepts Pigey's own
success field as an evaluator verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
# Candidate execution uses an exact argv vector, no shell, and a frozen entrypoint.
import subprocess  # nosec B404
from typing import Any, Callable, Mapping, Sequence

from ..common import write_json
from ..decision_evidence_contracts import canonical_digest
from ..paid_resource_admission import PaidResourceAdmissionGrant
from .candidate_policy import CandidatePolicyError
from .openai_cost_authority import openai_cost_authority_binding_digest


PIGEY_TRACE_SCHEMA_VERSION = "candidate_policy_trace.v1"
PIGEY_RUNTIME_RESULT_SCHEMA_VERSION = "candidate_policy_runtime_result.v1"
PIGEY_UPSTREAM_URL = "https://github.com/lianegalanti/Pigey"
PIGEY_LICENSE_ATTESTATION_SCHEMA_VERSION = "pigey_license_attestation.v1"
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SCENARIO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ALLOWED_ENVIRONMENT_KEYS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "CORTEX_NO_PHASE0",
        "EXTRA_PYTHONPATH",
        "GEMINI_API_KEY",
        "HARNESS_PERCEPTION",
        "LIBERO_CONFIG_PATH",
        "LIBERO_ENV_RES",
        "LIBERO_HORIZON",
        "MAX_ENV_STEPS",
        "OPENAI_API_KEY",
        "OPENAI_PROJECT",
        "PATH",
        "PHASE0_MAX_STEPS",
        "PYTHONPATH",
        "TOGETHER_API_KEY",
        "XLA_FLAGS",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _nonnegative_number(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CandidatePolicyError(f"{field}:invalid") from exc
    if not math.isfinite(number) or number < 0:
        raise CandidatePolicyError(f"{field}:invalid")
    return number


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise CandidatePolicyError(f"{field}:invalid")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise CandidatePolicyError(f"{field}:invalid") from exc
    if number < 0 or str(number) != str(value):
        raise CandidatePolicyError(f"{field}:invalid")
    return number


def validate_pigey_license_attestation(
    value: Mapping[str, Any],
    *,
    expected_commit_sha: str,
) -> dict[str, Any]:
    """Require independent commercial-use permission for unlicensed Pigey code."""

    attestation = dict(value)
    expected_digest = canonical_digest(
        attestation,
        digest_field="license_attestation_digest",
    )
    if (
        attestation.get("schema_version") != PIGEY_LICENSE_ATTESTATION_SCHEMA_VERSION
        or attestation.get("license_attestation_digest") != expected_digest
        or attestation.get("status") not in {"license_verified", "permission_granted"}
        or attestation.get("source_repository") != PIGEY_UPSTREAM_URL
        or attestation.get("source_commit_sha") != expected_commit_sha
        or attestation.get("commercial_use_authorized") is not True
        or attestation.get("code_execution_authorized") is not True
        or attestation.get("issued_by_agent") is not False
        or not str(attestation.get("reviewer_id") or "").strip()
        or attestation.get("proof_effect") != "none"
    ):
        raise CandidatePolicyError("pigey_license_or_permission_attestation_invalid")
    return attestation


@dataclass(frozen=True)
class PigeyScenarioBinding:
    scenario_id: str
    suite: str
    task_id: int
    episode: int
    seed: int = 7

    def __post_init__(self) -> None:
        if (
            not _SCENARIO_ID.fullmatch(self.scenario_id)
            or not self.suite.strip()
            or self.task_id < 0
            or self.episode < 0
            or self.seed < 0
        ):
            raise CandidatePolicyError("pigey_scenario_binding_invalid")


@dataclass
class PigeySimCandidateRuntime:
    """Execute one frozen Pigey stack across the suite's public scenario IDs."""

    candidate_id: str
    candidate_policy_manifest_digest: str
    checkout_root: Path
    expected_commit_sha: str
    expected_agent_sim_digest: str
    runtime_environment_digest: str
    terminal_signal_policy: str
    python_executable: Path
    mode: str
    model_id: str
    policy_host: str
    policy_port: int
    scenario_bindings: tuple[PigeyScenarioBinding, ...]
    observation_schema_ref: str
    action_schema_ref: str
    max_steps_per_rollout: int
    max_llm_steps: int
    replan_steps: int
    timeout_seconds_per_scenario: float
    max_cost_usd: float
    input_cost_per_million_tokens_usd: float
    output_cost_per_million_tokens_usd: float
    environment: Mapping[str, str]
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None
    openai_project_id: str
    openai_api_key_id: str
    openai_api_key_scope_attestation_digest: str
    license_attestation: Mapping[str, Any]
    provider_id: str = "pigey_external_candidate"
    provider_execution_planned: bool = True
    cost_accounting_authoritative: bool = False
    paid_resource_class: str | None = "openai_api_candidate"
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run

    @property
    def runtime_configuration_digest(self) -> str:
        """Digest every execution-relevant setting supplied to Pigey.

        Secret values are intentionally excluded; their variable names are bound,
        while the separately attested runtime-environment digest binds the sealed
        dependency environment without persisting credential fingerprints.
        """

        secret_keys = sorted(
            key for key in self.environment if key.endswith("_KEY") or key.endswith("_TOKEN")
        )
        nonsecret_environment = {
            key: value for key, value in sorted(self.environment.items()) if key not in secret_keys
        }
        return canonical_digest(
            {
                "schema_version": "pigey_runtime_configuration.v1",
                "source_repository": PIGEY_UPSTREAM_URL,
                "source_commit_sha": self.expected_commit_sha,
                "source_agent_sim_digest": self.expected_agent_sim_digest,
                "runtime_environment_digest": self.runtime_environment_digest,
                "python_executable": str(self.python_executable),
                "terminal_signal_policy": self.terminal_signal_policy,
                "candidate_id": self.candidate_id,
                "mode": self.mode,
                "model_id": self.model_id,
                "policy_host": self.policy_host,
                "policy_port": self.policy_port,
                "scenario_bindings": [
                    {
                        "scenario_id": row.scenario_id,
                        "suite": row.suite,
                        "task_id": row.task_id,
                        "episode": row.episode,
                        "seed": row.seed,
                    }
                    for row in sorted(
                        self.scenario_bindings,
                        key=lambda binding: binding.scenario_id,
                    )
                ],
                "observation_schema_ref": self.observation_schema_ref,
                "action_schema_ref": self.action_schema_ref,
                "max_steps_per_rollout": self.max_steps_per_rollout,
                "max_llm_steps": self.max_llm_steps,
                "replan_steps": self.replan_steps,
                "timeout_seconds_per_scenario": self.timeout_seconds_per_scenario,
                "max_cost_usd": self.max_cost_usd,
                "input_cost_per_million_tokens_usd": (self.input_cost_per_million_tokens_usd),
                "output_cost_per_million_tokens_usd": (self.output_cost_per_million_tokens_usd),
                "provider_id": self.provider_id,
                "provider_execution_planned": self.provider_execution_planned,
                "cost_accounting_authoritative": self.cost_accounting_authoritative,
                "paid_resource_class": self.paid_resource_class,
                "openai_project_id": self.openai_project_id,
                "openai_api_key_id": self.openai_api_key_id,
                "openai_api_key_scope_attestation_digest": (
                    self.openai_api_key_scope_attestation_digest
                ),
                "license_attestation_digest": self.license_attestation[
                    "license_attestation_digest"
                ],
                "cost_authority_binding_digest": self.cost_authority_binding_digest,
                "secret_environment_keys": secret_keys,
                "nonsecret_environment": nonsecret_environment,
            }
        )

    @property
    def cost_authority_binding_digest(self) -> str:
        return openai_cost_authority_binding_digest(
            provider_id=self.provider_id,
            paid_resource_class=str(self.paid_resource_class or ""),
            project_id=self.openai_project_id,
            api_key_id=self.openai_api_key_id,
            scope_attestation_digest=self.openai_api_key_scope_attestation_digest,
        )

    def __post_init__(self) -> None:
        self.checkout_root = self.checkout_root.expanduser().resolve()
        self.python_executable = self.python_executable.expanduser().resolve()
        if not _COMMIT_SHA.fullmatch(self.expected_commit_sha):
            raise CandidatePolicyError("pigey_commit_sha_invalid")
        if not _SHA256_DIGEST.fullmatch(self.expected_agent_sim_digest):
            raise CandidatePolicyError("pigey_agent_sim_digest_invalid")
        if not _SHA256_DIGEST.fullmatch(self.runtime_environment_digest):
            raise CandidatePolicyError("pigey_runtime_environment_digest_invalid")
        self.license_attestation = validate_pigey_license_attestation(
            self.license_attestation,
            expected_commit_sha=self.expected_commit_sha,
        )
        if self.terminal_signal_policy != "shared_libero_task_done":
            raise CandidatePolicyError("pigey_terminal_signal_policy_invalid")
        if self.mode not in {"raw", "harness"} or not self.model_id.strip():
            raise CandidatePolicyError("pigey_runtime_mode_or_model_invalid")
        if (
            not self.policy_host.strip()
            or not 1 <= self.policy_port <= 65535
            or not self.scenario_bindings
            or len({row.scenario_id for row in self.scenario_bindings})
            != len(self.scenario_bindings)
        ):
            raise CandidatePolicyError("pigey_runtime_binding_invalid")
        if (
            not self.observation_schema_ref.strip()
            or not self.action_schema_ref.strip()
            or self.max_steps_per_rollout < 1
            or self.max_llm_steps < 1
            or self.replan_steps < 1
            or not math.isfinite(self.timeout_seconds_per_scenario)
            or self.timeout_seconds_per_scenario <= 0
        ):
            raise CandidatePolicyError("pigey_runtime_envelope_invalid")
        for value in (
            self.max_cost_usd,
            self.input_cost_per_million_tokens_usd,
            self.output_cost_per_million_tokens_usd,
        ):
            if not math.isfinite(value) or value < 0:
                raise CandidatePolicyError("pigey_runtime_cost_invalid")
        unknown_env = set(self.environment) - _ALLOWED_ENVIRONMENT_KEYS
        if unknown_env or any(not isinstance(value, str) for value in self.environment.values()):
            raise CandidatePolicyError("pigey_runtime_environment_not_allowlisted")
        if (
            not self.openai_project_id.strip()
            or not self.openai_api_key_id.strip()
            or not self.openai_api_key_scope_attestation_digest.startswith("sha256:")
            or len(self.openai_api_key_scope_attestation_digest) != 71
            or self.environment.get("OPENAI_PROJECT") != self.openai_project_id
            or not str(self.environment.get("OPENAI_API_KEY") or "").strip()
        ):
            raise CandidatePolicyError("pigey_openai_cost_scope_not_bound")

    def _run(self, command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return self.command_runner(list(command), **kwargs)

    def _verify_checkout(self) -> Path:
        script = (self.checkout_root / "sim" / "agent_sim.py").resolve()
        if self.checkout_root not in script.parents or not script.is_file():
            raise CandidatePolicyError("pigey_agent_sim_missing")
        commit = self._run(
            ["git", "-C", str(self.checkout_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if commit.returncode != 0 or commit.stdout.strip() != self.expected_commit_sha:
            raise CandidatePolicyError("pigey_checkout_commit_mismatch")
        status = self._run(
            [
                "git",
                "-C",
                str(self.checkout_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if status.returncode != 0 or status.stdout.strip():
            raise CandidatePolicyError("pigey_checkout_not_clean")
        if _sha256_file(script) != self.expected_agent_sim_digest:
            raise CandidatePolicyError("pigey_agent_sim_digest_mismatch")
        return script

    def _scenario_ids(self, spec: Mapping[str, Any]) -> tuple[str, ...]:
        policy = dict(spec.get("policy_adapter") or {})
        if (
            policy.get("policy_id") != self.candidate_id
            or policy.get("observation_schema_ref") != self.observation_schema_ref
            or policy.get("action_schema_ref") != self.action_schema_ref
        ):
            raise CandidatePolicyError("pigey_evaluation_spec_binding_mismatch")
        pack = dict(spec.get("task_scenario_pack") or {})
        scenario_ids = tuple(sorted(str(row) for row in pack.get("scenario_ids") or []))
        bound_ids = tuple(sorted(row.scenario_id for row in self.scenario_bindings))
        if not scenario_ids or scenario_ids != bound_ids:
            raise CandidatePolicyError("pigey_scenario_set_mismatch")
        if pack.get("hidden_labels_included") is not False:
            raise CandidatePolicyError("pigey_spec_exposes_hidden_labels")
        return scenario_ids

    def execute(
        self,
        *,
        evaluation_run_spec: Mapping[str, Any],
        output_dir: Path,
    ) -> Mapping[str, Any]:
        script = self._verify_checkout()
        scenario_ids = self._scenario_ids(evaluation_run_spec)
        bindings = {row.scenario_id: row for row in self.scenario_bindings}
        root = output_dir.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        env = {key: value for key, value in self.environment.items()}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        normalized_trials: list[dict[str, Any]] = []
        total_cost = 0.0
        total_duration = 0.0
        provider_started = False
        for scenario_id in scenario_ids:
            binding = bindings[scenario_id]
            scenario_root = (root / "pigey" / scenario_id).resolve()
            if root not in scenario_root.parents:
                raise CandidatePolicyError("pigey_output_path_escape")
            scenario_root.mkdir(parents=True, exist_ok=False)
            command = [
                str(self.python_executable),
                str(script),
                "--mode",
                self.mode,
                "--suite",
                binding.suite,
                "--task",
                str(binding.task_id),
                "--episode",
                str(binding.episode),
                "--seed",
                str(binding.seed),
                "--policy-host",
                self.policy_host,
                "--policy-port",
                str(self.policy_port),
                "--model",
                self.model_id,
                "--max-steps-per-rollout",
                str(self.max_steps_per_rollout),
                "--max-llm-steps",
                str(self.max_llm_steps),
                "--replan-steps",
                str(self.replan_steps),
                "--out-dir",
                str(scenario_root),
                "--no-video",
            ]
            provider_started = True
            try:
                completed = self._run(
                    command,
                    cwd=self.checkout_root / "sim",
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds_per_scenario,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return self._failed_result(
                    blocker="pigey_scenario_timeout",
                    cost_usd=total_cost,
                    duration_seconds=total_duration,
                    provider_started=provider_started,
                )
            if completed.returncode != 0:
                return self._failed_result(
                    blocker="pigey_process_failed",
                    cost_usd=total_cost,
                    duration_seconds=total_duration,
                    provider_started=provider_started,
                )
            trial_paths = sorted(scenario_root.rglob("trial.json"))
            if len(trial_paths) != 1:
                return self._failed_result(
                    blocker="pigey_trial_artifact_count_invalid",
                    cost_usd=total_cost,
                    duration_seconds=total_duration,
                    provider_started=provider_started,
                )
            trial_path = trial_paths[0].resolve()
            if scenario_root not in trial_path.parents:
                raise CandidatePolicyError("pigey_trial_path_escape")
            try:
                trial = json.loads(trial_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CandidatePolicyError("pigey_trial_artifact_invalid") from exc
            if not isinstance(trial, Mapping):
                raise CandidatePolicyError("pigey_trial_artifact_invalid")
            if (
                trial.get("mode") != self.mode
                or trial.get("model") != self.model_id
                or trial.get("suite") != binding.suite
                or _nonnegative_int(trial.get("task_id"), field="pigey_trial_task_id")
                != binding.task_id
                or _nonnegative_int(trial.get("episode"), field="pigey_trial_episode")
                != binding.episode
            ):
                raise CandidatePolicyError("pigey_trial_binding_mismatch")
            serialized_trial = json.dumps(trial, sort_keys=True)
            secret_values = [
                value
                for key, value in self.environment.items()
                if (key.endswith("_KEY") or key.endswith("_TOKEN")) and len(value) >= 8
            ]
            if any(secret in serialized_trial for secret in secret_values):
                raise CandidatePolicyError("pigey_trial_contains_secret")
            usage = dict(trial.get("usage") or {})
            input_tokens = _nonnegative_number(
                usage.get("input_tokens") or 0,
                field="pigey_input_tokens",
            )
            output_tokens = _nonnegative_number(
                usage.get("output_tokens") or 0,
                field="pigey_output_tokens",
            )
            trial_cost = (
                input_tokens * self.input_cost_per_million_tokens_usd
                + output_tokens * self.output_cost_per_million_tokens_usd
            ) / 1_000_000
            total_cost += trial_cost
            if total_cost > self.max_cost_usd:
                raise CandidatePolicyError("pigey_runtime_cost_exceeded")
            duration = _nonnegative_number(
                trial.get("duration_s") or 0,
                field="pigey_duration_seconds",
            )
            total_duration += duration
            transcript = trial.get("transcript")
            if not isinstance(transcript, list):
                raise CandidatePolicyError("pigey_transcript_invalid")
            normalized_trials.append(
                {
                    "scenario_id": scenario_id,
                    "suite": binding.suite,
                    "task_id": binding.task_id,
                    "episode": binding.episode,
                    "seed": binding.seed,
                    "trial_artifact_digest": canonical_digest(trial),
                    "transcript": transcript,
                    "usage": {
                        "input_tokens": int(input_tokens),
                        "output_tokens": int(output_tokens),
                    },
                    "env_steps": _nonnegative_int(
                        trial.get("env_steps", 0),
                        field="pigey_env_steps",
                    ),
                    "llm_steps": _nonnegative_int(
                        trial.get("llm_steps", 0),
                        field="pigey_llm_steps",
                    ),
                    "candidate_reported_success_present": "success" in trial,
                    "candidate_reported_success_value_excluded": True,
                }
            )

        trace = {
            "schema_version": PIGEY_TRACE_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "source_project": "Pigey",
            "source_repository": PIGEY_UPSTREAM_URL,
            "source_commit_sha": self.expected_commit_sha,
            "source_agent_sim_digest": self.expected_agent_sim_digest,
            "runtime_environment_digest": self.runtime_environment_digest,
            "runtime_configuration_digest": self.runtime_configuration_digest,
            "terminal_signal_policy": self.terminal_signal_policy,
            "mode": self.mode,
            "model_id": self.model_id,
            "observation_schema_ref": self.observation_schema_ref,
            "action_schema_ref": self.action_schema_ref,
            "scenario_trials": normalized_trials,
            "candidate_reported_success_accepted_as_verdict": False,
            "hidden_labels_received": False,
            "evaluator_authority": False,
            "proof_effect": "none",
        }
        trace_path = root / "pigey_candidate_trace.json"
        write_json(trace_path, trace)
        return {
            "schema_version": PIGEY_RUNTIME_RESULT_SCHEMA_VERSION,
            "status": "completed",
            "trace_artifact_path": trace_path.relative_to(root).as_posix(),
            "trace_artifact_digest": canonical_digest(trace),
            "blockers": [],
            "cost_usd": round(total_cost, 6),
            "duration_seconds": round(total_duration, 6),
            "provider_execution_started": provider_started,
            "attempt_count": 1,
        }

    @staticmethod
    def _failed_result(
        *,
        blocker: str,
        cost_usd: float,
        duration_seconds: float,
        provider_started: bool,
    ) -> dict[str, Any]:
        return {
            "schema_version": PIGEY_RUNTIME_RESULT_SCHEMA_VERSION,
            "status": "failed",
            "blockers": [blocker],
            "cost_usd": round(cost_usd, 6),
            "duration_seconds": round(duration_seconds, 6),
            "provider_execution_started": provider_started,
            "attempt_count": 1,
        }


__all__ = [
    "PIGEY_LICENSE_ATTESTATION_SCHEMA_VERSION",
    "PIGEY_RUNTIME_RESULT_SCHEMA_VERSION",
    "PIGEY_TRACE_SCHEMA_VERSION",
    "PIGEY_UPSTREAM_URL",
    "PigeyScenarioBinding",
    "PigeySimCandidateRuntime",
    "validate_pigey_license_attestation",
]
