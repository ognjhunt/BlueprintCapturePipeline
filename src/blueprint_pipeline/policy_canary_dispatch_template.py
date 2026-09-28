"""ADP-009D/day-21: dispatch the setup retained for this Website evaluation."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_policy_canary_scene_setup import PolicyCanarySetupError


def resolve_execution_template(*, queue_root: Path, envelope: Mapping[str, Any],
                               legacy_template: str | Path | None) -> Path | None:
    """Select by sealed handoff identity; never use a global scene for owner runs.

    The existing materializer still verifies template bytes, owner reservation,
    scene revision, task contract and runtime inputs before dispatch.
    """
    if not envelope.get("scene_intent_digest"):
        return Path(legacy_template) if legacy_template is not None else None
    source = str(envelope.get("capture_session_id") or "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}", source):
        raise PolicyCanarySetupError(["policy_canary_handoff_source_invalid"])
    root = queue_root.parent / "task-evaluation-configured-controls" / source
    matches = []
    for path in sorted(root.glob("*/policy_canary_handoff_progression.json")):
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise PolicyCanarySetupError(["policy_canary_handoff_path_unsafe"])
        handoff = json.loads(path.read_text())
        if (str(handoff.get("run_id")) + "-activation" != envelope.get("activation_id")
                or handoff.get("expected_production_commit") != envelope.get("source_commit")):
            continue
        authority = handoff.get("evaluation_authority") or {}
        if (handoff.get("schema_version") != "task_evaluation_policy_canary_handoff_progression.v1"
                or handoff.get("status") != "canary_launch_submitted"
                or handoff.get("source_launch_id") != source
                or authority.get("scene_intent_digest") != envelope["scene_intent_digest"]
                or handoff.get("progression_digest") != canonical_digest(handoff, digest_field="progression_digest")):
            raise PolicyCanarySetupError(["policy_canary_handoff_binding_invalid"])
        directory = path.parent / "policy-canary-presubmission"
        wrapper = directory / "task_evaluation_policy_canary_profile_materialization_input.v1.json"
        if str(wrapper) != handoff.get("profile_materialization_input_path"):
            raise PolicyCanarySetupError(["policy_canary_handoff_wrapper_path_invalid"])
        matches.append(directory / "task_evaluation_policy_canary_execution_setup_template.v1.json")
    if len(matches) != 1:
        raise PolicyCanarySetupError(["policy_canary_handoff_template_missing_or_ambiguous"])
    return matches[0]
