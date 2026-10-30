"""Root-door admission for one sealed, explicitly funded public asset batch.

This stdlib-only boundary cannot import code from the active pipeline release.
Parity tests keep its allowlist and digest identical to the worker contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from typing import Any

PUBLIC_EMBODIMENT_IDS = frozenset({
    "yaskawa_gp12_robotiq_2f85_sim", "yaskawa_gp20_robotiq_2f85_sim",
    "kuka_lbr_iiwa_14_robotiq_2f85_sim",
    "fanuc_crx_10ia_robotiq_2f85_sim",
    "fanuc_crx_10ia_l_robotiq_2f85_sim",
    "fanuc_m710ic_45m_robotiq_2f85_sim",
    "abb_gofa_crb15000_5_95_robotiq_2f85_sim",
    "abb_gofa_crb15000_12_127_robotiq_2f85_sim",
    "universal_robots_ur5e_robotiq_2f85_sim", "universal_robots_ur10e_robotiq_2f85_sim",
    "abb_gofa_crb15000_5_95", "abb_gofa_crb15000_12_127",
    "franka_research_3_robotiq_2f85_sim", "agilex_piper_bimanual", "almond_axol", "fanuc_crx_10ia", "fanuc_crx_10ia_l",
    "fanuc_m710ic_45m", "franka_research_3_bimanual", "i2rt_yam_bimanual", "kinova_gen3",
    "kinova_gen3_bimanual", "kuka_lbr_iiwa_14", "trlc_dk1_bimanual", "ufactory_xarm7", "ufactory_xarm7_bimanual",
    "universal_robots_ur10e", "universal_robots_ur5e", "universal_robots_ur5e_bimanual", "yaskawa_gp12", "yaskawa_gp20",
})
REQUEST_FIELDS = frozenset({
    "schema_version", "authorization_id", "release_commit", "embodiment_ids", "max_spend_usd",
    "max_hourly_rate_usd", "ttl_seconds", "expires_at_epoch", "authorization_reference", "request_digest",
})


def validate_request(value: Any, *, now: float | None = None) -> dict[str, Any]:
    """Preserve the exact sealed request; never supply or broaden its authority."""
    def require(condition: Any, reason: str) -> None:
        if not condition:
            raise ValueError("embodiment_preparation_" + reason)

    require(isinstance(value, dict) and set(value) == REQUEST_FIELDS
            and value.get("schema_version") == "embodiment_asset_preparation_request.v1", "request_schema_invalid")
    request = dict(value)
    for key, pattern in (("authorization_id", r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}"),
                         ("release_commit", r"[0-9a-f]{40}")):
        require(isinstance(request[key], str) and re.fullmatch(pattern, request[key]), key + "_invalid")
    ids = request["embodiment_ids"]
    require(isinstance(ids, list) and 1 <= len(ids) <= 24 and all(isinstance(x, str) for x in ids)
            and len(set(ids)) == len(ids) and set(ids) <= PUBLIC_EMBODIMENT_IDS, "catalog_selection_invalid")
    rate, cap, ttl, expires = (request[k] for k in
                             ("max_hourly_rate_usd", "max_spend_usd", "ttl_seconds", "expires_at_epoch"))
    require(all(type(x) in (int, float) and (type(x) is int or math.isfinite(x)) for x in (rate, cap, expires))
            and 0 < rate <= .80 and 0 < cap <= 4.0 and type(ttl) is int
            and 1800 <= ttl <= 14_400 and rate * ttl / 3600 <= cap, "budget_invalid")
    current = time.time() if now is None else now
    require(current + ttl + 300 <= expires <= current + 7 * 86400, "authorization_expiry_invalid")
    reference = request["authorization_reference"]
    require(isinstance(reference, str) and 1 <= len(reference) <= 300 and reference.isprintable(),
            "authorization_reference_invalid")
    try:
        payload = json.dumps({k: v for k, v in request.items() if k != "request_digest"},
                             sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, UnicodeError) as error:
        raise ValueError("embodiment_preparation_request_digest_invalid") from error
    require(request["request_digest"] == "sha256:" + hashlib.sha256(payload).hexdigest(), "request_digest_invalid")
    return request
