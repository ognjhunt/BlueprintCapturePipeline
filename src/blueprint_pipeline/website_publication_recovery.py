"""Controller-owned, no-provider publication recovery for completed website builds."""

import json
import os
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_publication_recovery import (
    recover_completed_configuration_publication,
    activate_recovered_launch_receipt,
    RECOVERED_LAUNCH_RECEIPT_FILENAME,
)
from .task_evaluation_scene_progression_state import safe_path


def reconcile_website_publication(*, intent, config, release):
    if not intent["request"]["task"]["task_id"].startswith("website-"):
        return None
    root = safe_path(config["launch_execution_root"])
    for path in sorted(
        root.glob("*/launch_profile.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True
    ):
        profile = json.loads(safe_path(path).read_text())
        binding = profile.get("scene_attempt_binding") or {}
        if any(binding.get(k) != intent[k] for k in ("intent_id", "intent_digest")):
            continue
        launch_path = path.parent / "launch_receipt.json"
        if not launch_path.is_file():
            continue
        launch = json.loads(launch_path.read_text())
        if launch.get("status") != "blocked":
            continue
        result_path = (launch.get("terminal_evidence", {}).get("result") or {}).get("path")
        if not result_path:
            continue
        result = json.loads(safe_path(result_path).read_text())
        if (
            result.get("configuration_completed") is not True
            or result.get("configured_scene_published") is not False
        ):
            continue
        output = path.parent / ("publication-recovery-" + release["source_commit"])
        recovered = output / RECOVERED_LAUNCH_RECEIPT_FILENAME
        failure = path.parent / (output.name + ".failure.json")
        try:
            if failure.exists():
                return {
                    "status": "blocked",
                    "phase": "scene_publication",
                    "blockers": json.loads(failure.read_text())["blockers"],
                }
            from .task_evaluation_scene_attempt_binding import require_scene_execution_binding

            require_scene_execution_binding(profile, source_commit=profile["source_commit"])
            if profile.get("profile_digest") != canonical_digest(
                profile, digest_field="profile_digest"
            ) or launch.get("launch_profile_digest") != profile.get("profile_digest"):
                raise ValueError("website_publication_profile_changed")
            attempt_path = (
                Path(config["intent_root"])
                / intent["intent_id"]
                / "attempts"
                / (binding["attempt_id"] + ".json")
            )
            attempt = json.loads(safe_path(attempt_path).read_text())
            if attempt.get("attempt_digest") != canonical_digest(
                attempt, digest_field="attempt_digest"
            ) or any(attempt.get(k) != v for k, v in binding.items() if k != "schema_version"):
                raise ValueError("website_publication_owner_mismatch")
            if not recovered.exists():
                argv = profile["allocator"]["argv"]
                flag = "--scene-configuration-bundle-receipt"
                if argv.count(flag) != 1:
                    raise ValueError("website_publication_bundle_ambiguous")
                recover_completed_configuration_publication(
                    bundle_receipt_path=argv[argv.index(flag) + 1],
                    provider_result_path=result["execution_result_path"],
                    original_result_path=result_path,
                    original_launch_receipt_path=launch_path,
                    queue_root=os.environ[
                        "BLUEPRINT_TASK_EVALUATION_SCENE_CONSTRUCTION_QUEUE_ROOT"
                    ],
                    output_root=output,
                    recovery_source_commit=release["source_commit"],
                )
            # A transient Website delivery failure retries this exact sealed
            # recovery, never republishing or rerunning the provider stages.
            activate_recovered_launch_receipt(
                original_launch_receipt_path=launch_path, recovered_launch_receipt_path=recovered
            )
            return {"status": "awaiting_execution", "phase": "scene_publication", "blockers": []}
        except (ValueError, RuntimeError, OSError, KeyError) as exc:
            blockers = [str(exc)[:400]]
            if not recovered.exists():
                failure.write_text(
                    json.dumps({"blockers": blockers, "provider_execution_repeated": False})
                )
            return {"status": "blocked", "phase": "scene_publication", "blockers": blockers}
    return None
