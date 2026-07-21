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
            sim_only_beta_autonomy=_strict_bool(
                source,
                "BLUEPRINT_SIM_ONLY_BETA_AUTONOMY",
            ),
            sim_only_beta_default_task_eval=_strict_bool(
                source,
                "BLUEPRINT_SIM_ONLY_BETA_DEFAULT_TASK_EVAL",
            ),
        )
