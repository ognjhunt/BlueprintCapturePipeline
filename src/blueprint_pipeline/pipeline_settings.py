"""Typed entrypoint settings loaded once from environment mappings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}


def _strict_bool(env: Mapping[str, str], name: str, *, default: bool = False) -> bool:
    if name not in env:
        return default
    value = str(env.get(name) or "").strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(f"invalid_boolean_environment_value:{name}")


@dataclass(frozen=True)
class PipelineSettings:
    environment: str
    gcs_root: Path
    allow_simulator_execution: bool
    allow_gpu_provisioning: bool
    allow_cosmos_training: bool
    allow_live_agents_sdk_operators: bool
    allow_legacy_agents_sdk_job_orchestration: bool
    sim_only_beta_autonomy: bool
    sim_only_beta_default_task_eval: bool

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "PipelineSettings":
        source = os.environ if env is None else env
        environment = str(source.get("BLUEPRINT_ENV") or "development").strip().lower()
        if environment not in {"development", "test", "staging", "production"}:
            raise ValueError(f"unsupported_blueprint_environment:{environment}")
        return cls(
            environment=environment,
            gcs_root=Path(str(source.get("GCS_ROOT") or "/mnt/gcs")).expanduser(),
            allow_simulator_execution=_strict_bool(
                source,
                "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION",
            ),
            allow_gpu_provisioning=_strict_bool(
                source,
                "BLUEPRINT_ALLOW_GPU_PROVISIONING",
            ),
            allow_cosmos_training=_strict_bool(
                source,
                "BLUEPRINT_ALLOW_COSMOS_TRAINING",
            ),
            allow_live_agents_sdk_operators=_strict_bool(
                source,
                "BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS",
            ),
            allow_legacy_agents_sdk_job_orchestration=_strict_bool(
                source,
                "BLUEPRINT_ALLOW_AGENTS_SDK_JOB_ORCHESTRATION",
            ),
            sim_only_beta_autonomy=_strict_bool(
                source,
                "BLUEPRINT_SIM_ONLY_BETA_AUTONOMY",
            ),
            sim_only_beta_default_task_eval=_strict_bool(
                source,
                "BLUEPRINT_SIM_ONLY_BETA_DEFAULT_TASK_EVAL",
            ),
        )

    def validate_cli_admission(
        self,
        *,
        allow_gpu_provisioning: bool = False,
        allow_simulator_execution: bool = False,
        allow_cosmos_training: bool = False,
        allow_live_agents_sdk_operator: bool = False,
    ) -> None:
        """Require environment approval for every mutating CLI admission flag."""

        missing: list[str] = []
        if allow_gpu_provisioning and not self.allow_gpu_provisioning:
            missing.append("BLUEPRINT_ALLOW_GPU_PROVISIONING")
        if allow_simulator_execution and not self.allow_simulator_execution:
            missing.append("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION")
        if allow_cosmos_training and not self.allow_cosmos_training:
            missing.append("BLUEPRINT_ALLOW_COSMOS_TRAINING")
        if allow_live_agents_sdk_operator and not (
            self.allow_live_agents_sdk_operators
            or self.allow_legacy_agents_sdk_job_orchestration
        ):
            missing.append("BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS")
        if missing:
            raise ValueError(
                "cli_admission_missing_environment_approval:" + ",".join(missing)
            )
