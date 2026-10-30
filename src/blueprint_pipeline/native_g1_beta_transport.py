"""Terminal G1 beta transport identity; no score or model-loading authority."""
from __future__ import annotations

from collections.abc import Mapping
import re

from .decision_evidence_contracts import canonical_digest

RESULT_SCHEMA = "native_g1_beta_session_result.v1"
RESULT_FILENAME = RESULT_SCHEMA + ".json"
MAX_OUTPUT_BYTES = 4 * 1024**3
MAX_TERMINAL_BYTES = 512 * 1024**2


def transport_completed(result: Mapping, *, manifest: Mapping | None = None) -> bool:
    """A sealed blocked session still arrived; final closeout verifies its bytes."""
    try:
        digests = ("authority_digest", "runtime_inputs_digest", "plan_digest", "configuration_digest",
            "admission_digest", "runtime_identity_digest", "scenario_plan_digest", "task_success_contract_digest")
        episodes = result.get("episodes")
        if (result.get("schema_version") != RESULT_SCHEMA
                or result.get("status") not in {"completed", "blocked"}
                or result.get("claim_ceiling") != "development_only"
                or any(result.get(key) is not False for key in (
                    "ranking_eligible", "qualification_permitted", "public_media_publication_authorized",
                    "model_redistribution_authorized", "physical_outcome_claimed"))
                or type(result.get("candidate_policy_queried")) is not bool
                or any(re.fullmatch(r"sha256:[0-9a-f]{64}", str(result.get(key, ""))) is None for key in digests)
                or not isinstance(result.get("candidate_id"), str) or not result["candidate_id"]
                or not isinstance(result.get("blockers"), list)
                or any(not isinstance(reason, str) for reason in result["blockers"])
                or not isinstance(episodes, list) or not 1 <= len(episodes) <= 10
                or any(not isinstance(row, Mapping) or not isinstance(row.get("cell_id"), str)
                       or not row["cell_id"] or row.get("status") not in {"completed", "blocked", "not_attempted"}
                       for row in episodes)
                or len({row["cell_id"] for row in episodes}) != len(episodes)
                or (result["status"] == "completed" and any(row["status"] != "completed" for row in episodes))
                or result.get("result_digest") != canonical_digest(result, digest_field="result_digest")):
            return False
        if manifest is not None and (
                manifest.get("candidate_ids") != [result["candidate_id"]]
                or manifest.get("cell_ids") != [row["cell_id"] for row in episodes]
                or any(result[key] != manifest.get(key) for key in (
                    "authority_digest", "runtime_inputs_digest", "task_success_contract_digest"))):
            return False
    except (KeyError, TypeError, ValueError):
        return False
    return True
