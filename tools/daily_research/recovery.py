"""Narrow offline derivation; retained research is never promoted to approved fact."""
from copy import deepcopy


def quarantine_null_operator_deltas(output):
    """Keep every candidate/claim unchanged; exclude whole optional proposals.

    Only the observed all-operator/all-null evidence pattern is recoverable.
    No null is translated into vendor/deployment/current/unknown evidence.
    All remaining output still passes the original strict validator and agent QA.
    """
    values = output.get("proposed_knowledge_deltas")
    if not isinstance(values, list) or len(values) > 10:
        raise ValueError("output_recovery_delta_shape_invalid")
    derived, quarantine = deepcopy(output), []
    kept = []
    for index, delta in enumerate(values):
        evidence = delta.get("evidence") if isinstance(delta, dict) else None
        matched = (isinstance(evidence, list) and 1 <= len(evidence) <= 4
                   and all(isinstance(item, dict) and item.get("classification") == "operator"
                           and "evidence_level" in item and item["evidence_level"] is None for item in evidence))
        if matched:
            quarantine.append({"delta_index": index, "proposal": deepcopy(delta),
                               "reason": "operator_delta_null_evidence_level_requires_agent_correction",
                               "invalid_fields": [f"/proposed_knowledge_deltas/{index}/evidence/{i}/evidence_level"
                                                  for i in range(len(evidence))],
                               "approved": False})
        else:
            kept.append(deepcopy(delta))
    if not quarantine:
        raise ValueError("output_recovery_no_matching_proposal")
    derived["proposed_knowledge_deltas"] = kept
    return derived, quarantine
