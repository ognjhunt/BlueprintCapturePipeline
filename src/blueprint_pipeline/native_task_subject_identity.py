"""Pure runtime-safe subject identity normalization shared by compiler and probes."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest


def runtime_subject_task_spec(value: Mapping[str, Any], *, error_type: type[Exception] = ValueError) -> dict[str, Any]:
    """Derive a native-safe asset ID while retaining product identity."""

    task_spec = json.loads(json.dumps(dict(value)))
    # An adapter that already bound both spellings has nothing to derive; doing
    # it again would overwrite the product identity with the runtime name.
    declared_source = task_spec.get("source_subject_identity")
    if isinstance(declared_source, str) and declared_source.strip():
        source_subject_id = declared_source
        runtime_subject_id = re.sub(r"[^A-Za-z0-9_]", "_", source_subject_id)
        affordance = task_spec.get("interaction_affordance")
        if (
            task_spec.get("subject_asset_id") != runtime_subject_id
            or not isinstance(affordance, Mapping)
            or affordance.get("subject_asset_id") != runtime_subject_id
        ):
            raise error_type(
                "episode_compiler_interaction_affordance_invalid"
            )
        return task_spec
    source_subject_id = str(task_spec.get("subject_asset_id") or "")
    runtime_subject_id = re.sub(r"[^A-Za-z0-9_]", "_", source_subject_id)
    if not runtime_subject_id or not runtime_subject_id.replace("_", "a").isalnum():
        raise error_type(
            "episode_compiler_subject_runtime_id_invalid"
        )
    interaction_affordance = task_spec.get("interaction_affordance")
    if (
        not isinstance(interaction_affordance, Mapping)
        or interaction_affordance.get("subject_asset_id") != source_subject_id
    ):
        raise error_type(
            "episode_compiler_interaction_affordance_invalid"
        )
    task_spec["subject_asset_id"] = runtime_subject_id
    task_spec["source_subject_identity"] = source_subject_id
    interaction_affordance = dict(interaction_affordance)
    interaction_affordance["subject_asset_id"] = runtime_subject_id
    interaction_affordance["affordance_digest"] = canonical_digest(
        interaction_affordance, digest_field="affordance_digest"
    )
    task_spec["interaction_affordance"] = interaction_affordance
    return task_spec

