"""Private delivery and bounded API charges for configured HTTPS policies.

Only operator-owned credential paths enter local execution records. Secret
contents never enter the plan, provider bundle or public result. The existing
HTTPS client allows no redirect or retry: every inference sends one metadata
GET plus one POST, with four extra readiness GETs reserved per episode.
"""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

HOSTED_FIELDS = ("credential_file", "credential_origin", "provider_hard_cap_usd",
                 "hosted_inference_maximum_usd", "hosted_inference_payment",
                 "hosted_max_cost_per_request_usd", "hosted_http_request_limit")


def request_limit(spec: Mapping[str, Any], episodes: int) -> int:
    queries = spec.get("max_policy_queries")
    if type(queries) is not int or queries < 1 or type(episodes) is not int or not 1 <= episodes <= 10:
        raise ValueError("beta_screening_hosted_request_bound_invalid")
    return (2 * queries + 4) * episodes


def validate_hosted_delivery(spec: Mapping[str, Any], settings: Mapping[str, Any], *, episodes: int) -> None:
    """Validate exact endpoint/identity/secret origin and worst-case HTTP charges."""
    from .droid_https_policy_client import DroidHttpsPolicySpec
    from .team_candidate_quick10_vast import credential_file_blockers

    interface = spec.get("policy_interface_id")
    if interface == "openpi_droid_https.v1":
        policy = DroidHttpsPolicySpec(**spec["policy_spec"])
    elif interface == "task_family_policy_https.v1" and spec.get("evaluation_backend") == "task_family_native.v1":
        from .task_family_policy_https_client import TaskFamilyPolicySpec

        policy = TaskFamilyPolicySpec(**spec["policy_spec"])
    else:
        raise ValueError("beta_screening_hosted_interface_not_supported")
    policy.validate()
    if any(urlsplit(url).query for url in (policy.endpoint_url, policy.metadata_url)):
        raise ValueError("beta_screening_hosted_query_parameters_forbidden")
    if (policy.allowed_origin != settings.get("credential_origin")
            or policy.model_identity_digest != spec.get("checkpoint_digest")
            or policy.open_loop_horizon != spec.get("open_loop_horizon")):
        raise ValueError("beta_screening_hosted_origin_or_identity_mismatch")
    blockers = credential_file_blockers(settings.get("credential_file"))
    if blockers:
        raise ValueError("beta_screening_hosted_credential_not_private")
    # Validate the private text without retaining or rendering it anywhere.
    token = Path(settings["credential_file"]).read_text(encoding="utf-8").strip()
    if not token or any(character in token for character in "\r\n"):
        raise ValueError("beta_screening_hosted_credential_invalid")
    if set(HOSTED_FIELDS) - set(settings):
        raise ValueError("beta_screening_hosted_cost_unknown")
    if settings["hosted_http_request_limit"] != request_limit(spec, episodes):
        raise ValueError("beta_screening_hosted_request_bound_mismatch")
    maximum = Decimal(settings["hosted_inference_maximum_usd"])
    price = Decimal(settings["hosted_max_cost_per_request_usd"])
    if (not maximum.is_finite() or not price.is_finite() or maximum < 0 or price < 0
            or price * settings["hosted_http_request_limit"] > maximum):
        raise ValueError("beta_screening_hosted_cost_below_request_bound")
    payer = settings["hosted_inference_payment"]
    if (payer not in {"blueprint", "free", "caller_paid"}
            or (payer == "blueprint" and (price <= 0 or maximum <= 0))
            or (payer in {"free", "caller_paid"} and (price != 0 or maximum != 0))):
        raise ValueError("beta_screening_hosted_payment_unproven")
