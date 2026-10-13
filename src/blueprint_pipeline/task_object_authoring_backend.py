"""New-request policy; issued configurations retain their explicit backend/budget.

ADP-009 day-14/day-21 owner-directed migration, 2026-10-12. This module
contains no authoring engine, provider client, or source-measurement inference.

One setting, ``BLUEPRINT_TASK_OBJECT_AUTHORING_BACKEND``, selects the builder
for new requests and for deploys. Owner decision 2026-10-13: ARTISAN is merged
but switched off (Astra) until its runtime is installed on the production host
and a retained canary is valid; go-live sets it to ``artisan_v1`` for the
deploy and every service. Explicit request backends are never rewritten.
"""
from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Any

ARTISAN_BACKEND = "artisan_v1"
BACKEND_SETTING_ENV = "BLUEPRINT_TASK_OBJECT_AUTHORING_BACKEND"
SWITCHED_OFF_BACKEND = "astra_cad_blender_v1"
LEGACY_BACKENDS = frozenset({"content_agents", "astra_cad_blender_v1"})
AUTHORING_BACKENDS = LEGACY_BACKENDS | {ARTISAN_BACKEND}
ARTISAN_REVISION = "ad322b5b31e50bdb84155520f09622c00c6056c3"
ARTISAN_TREE = "6ed9c860e74afdf90bade4bc5ec4ac592cb1e920"
ARTISAN_BUDGET_ENV = "BLUEPRINT_ARTISAN_AUTHORING_MAX_COST_USD"
BUDGET_SCHEMA = "task_object_authoring_budget.v1"
ARTISAN_MODELS = {"provider": "openai", "model": "gpt-6-astra", "thinking": "high",
                  "compaction_model": "gpt-5.6-sol", "image_model": "gpt-6-astra",
                  "image_tool_model": "gpt-image-2.5-sunburst"}


def default_backend(environment: Mapping[str, str] | None = None) -> str:
    """The builder a new request or deploy selects when none is named."""
    values = os.environ if environment is None else environment
    value = str(values.get(BACKEND_SETTING_ENV) or SWITCHED_OFF_BACKEND).strip()
    if value not in {ARTISAN_BACKEND, SWITCHED_OFF_BACKEND}:
        raise ValueError("task_object_authoring_backend_setting_invalid")
    return value


def positive_cost(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("artisan_authoring_budget_invalid")
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("artisan_authoring_budget_invalid") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError("artisan_authoring_budget_invalid")
    return result


def new_authoring_budget(environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    values = os.environ if environment is None else environment
    return {"schema_version": BUDGET_SCHEMA, "mode": "finite_cap",
            "maximum_cost_usd": positive_cost(values.get(ARTISAN_BUDGET_ENV, "15")),
            "scope": "whole_authoring_stage_cumulative_including_resume"}


def frozen_authoring_cap(configuration: Mapping[str, Any]) -> float:
    budget = configuration.get("authoring_budget")
    if (not isinstance(budget, Mapping) or budget.get("schema_version") != BUDGET_SCHEMA
            or budget.get("mode") != "finite_cap"
            or budget.get("scope") != "whole_authoring_stage_cumulative_including_resume"):
        raise ValueError("artisan_frozen_authoring_budget_required")
    return positive_cost(budget.get("maximum_cost_usd"))


def request_authoring_cap(request: Mapping[str, Any]) -> float | None:
    if request.get("replacement_authoring_backend") != ARTISAN_BACKEND:
        return None
    return positive_cost(request["spend"]["external_service_caps"]["openai"]
                         ["stage_max_cost_usd"]["content_agents"])


def bundle_authoring_cap(receipt: Mapping[str, Any]) -> float | None:
    if receipt.get("replacement_authoring_backend") != ARTISAN_BACKEND:
        return None
    return positive_cost(receipt.get("authoring_max_cost_usd"))


def backend_configuration(backend: str) -> dict[str, Any]:
    if backend not in AUTHORING_BACKENDS:
        raise ValueError("replacement_authoring_backend_invalid")
    if backend != ARTISAN_BACKEND:
        return {}
    return {"authoring_budget": new_authoring_budget(),
            "authoring_models": dict(ARTISAN_MODELS),
            "authoring_framework": {"repository": "https://github.com/physicalvision/Artisan",
                                    "revision": ARTISAN_REVISION},
            "maximum_export_error_m": 0.001}
