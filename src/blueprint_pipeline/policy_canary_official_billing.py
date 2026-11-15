"""Terminal evidence adapter for official billing of policy canary sessions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .provider_output_promotion_records import (
    ABSENCE_PROOF_FILENAME,
    ProviderOutputPromotionRecordError,
    load_promotion_receipt,
    load_staged_object_absence_proof,
)

# The arena lane's staging directory under its attempt root.
STAGING_DIRNAME = "object_store_staging"
NOT_DURABLE = "policy_canary_provider_output_not_durable"
NOT_FINAL = "policy_canary_provider_output_promotion_not_final"


def _sealed_absent(lane_result: Mapping[str, Any]) -> bool:
    closeout = lane_result.get("provider_closeout")
    return isinstance(closeout, Mapping) and closeout.get("all_staged_objects_absent") is True


def _checkpoint(receipt: Mapping[str, Any]) -> bool:
    """Whether a promotion receipt is a checkpoint (witness pending): a promotion mid-run."""
    section = (receipt.get("staged_objects") or {}).get("paired_witness") or {}
    return (receipt.get("witness") or {}).get("disposition") == "pending" or section.get("state") == "pending"


def staged_object_absence_proof(lane_result: Mapping[str, Any]) -> tuple[dict, Path, bool] | None:
    """The attempt's validated absence proof, its path and whether promotion is final; else None.

    Only a proof for a staging manifest that required promotion counts. A
    proof file that does not validate, or that is bound to a promotion receipt
    other than the staging dir's current, final one, raises
    ``ProviderOutputPromotionRecordError``.
    """
    attempt = str(lane_result.get("attempt_root") or "").strip()
    if not attempt:
        return None
    staging = Path(attempt) / STAGING_DIRNAME
    proof = load_staged_object_absence_proof(staging)
    if proof is None or proof.get("output_promotion_required") is not True:
        return None
    path = staging / ABSENCE_PROOF_FILENAME
    receipt = load_promotion_receipt(staging, staging_manifest_sha256=proof.get("staging_manifest_sha256"))
    if receipt is not None and _checkpoint(receipt):
        return proof, path, False
    if receipt is None or receipt.get("receipt_digest") != proof.get("promotion_receipt_digest"):
        raise ProviderOutputPromotionRecordError("staged_object_absence_proof_receipt_not_current")
    return proof, path, True


def billing_staged_objects_absent(lane_result: Mapping[str, Any]) -> tuple[bool, Path | None]:
    """Billing's answer: absent, and the proof path when the sealed flag did not say so."""
    if _sealed_absent(lane_result):
        return True, None
    try:
        found = staged_object_absence_proof(lane_result)
    except ProviderOutputPromotionRecordError:
        return False, None
    return (True, found[1]) if found is not None and found[2] else (False, None)


def late_watchdog_instance(
    *,
    result_path: Path,
    result: Mapping[str, Any],
    read_json: Callable[[Path, str], Mapping[str, Any]],
    error_factory: Callable[[str], Exception],
) -> tuple[int, Path] | None:
    snapshot = result.get("independent_watchdog")
    if not isinstance(snapshot, Mapping) or snapshot.get("status") != "retained_until_hard_ttl":
        return None
    ids = snapshot.get("instance_ids")
    if (not isinstance(ids, list) or len(ids) != 1 or isinstance(ids[0], bool)
            or not isinstance(ids[0], int) or ids[0] <= 0):
        raise error_factory("policy_canary_late_watchdog_identity_invalid")
    instance_id = ids[0]
    adapter_path = Path(str(result.get("adapter_result_path") or ""))
    if not adapter_path.is_absolute() or adapter_path.name != "vast_provider_adapter_result.json":
        raise error_factory("policy_canary_late_watchdog_identity_invalid")
    attempt_root = adapter_path.parent.parent
    if adapter_path.parent.name != "vast_provider_run" or attempt_root.parent.name != "attempts":
        raise error_factory("policy_canary_late_watchdog_identity_invalid")
    if not attempt_root.is_relative_to(result_path.parent):
        raise error_factory("policy_canary_late_watchdog_identity_invalid")
    terminal_path = (
        attempt_root / "independent_vast_watchdog" / "groot_oscar_runpod_canary_watchdog.json"
    )
    if (snapshot.get("watchdog_evidence_path") != str(terminal_path)
            or snapshot.get("watchdog_out_dir") != str(terminal_path.parent)
            or terminal_path.is_symlink()):
        raise error_factory("policy_canary_late_watchdog_identity_invalid")
    if not terminal_path.is_file():
        return None
    terminal = read_json(terminal_path, "policy_canary_late_watchdog_terminal_invalid")
    recorded = terminal.get("recorded_vast_instance")
    teardown = terminal.get("recorded_vast_instance_teardown")
    if (
        terminal.get("schema_version") != "groot_oscar_runpod_canary_watchdog.v1"
        or terminal.get("status") != "provider_terminal"
        or terminal.get("provider_absence_confirmed") is not True
        or terminal.get("provider_absence_scope") != "recorded_instance_and_lane_prefix"
        or terminal.get("raw_secret_values_recorded") is not False
        or terminal.get("pod_name_prefix") != snapshot.get("pod_name_prefix")
        or terminal.get("resource_name_exact") != snapshot.get("resource_name_exact")
        or terminal.get("watchdog_out_dir") != str(terminal_path.parent)
        or not isinstance(recorded, Mapping)
        or recorded.get("instance_id") != str(instance_id)
        or recorded.get("scope_confirmed") is not True
        or recorded.get("pod_name_prefix") != snapshot.get("pod_name_prefix")
        or not isinstance(teardown, Mapping)
        or teardown.get("instance_id") != str(instance_id)
        or teardown.get("status") != "absent"
        or teardown.get("provider_absence_confirmed") is not True
    ):
        raise error_factory("policy_canary_late_watchdog_terminal_invalid")
    return instance_id, terminal_path



def policy_canary_terminal_evidence(
    *,
    instance_id: int,
    result_path: Path,
    result: Mapping[str, Any],
    result_bytes: bytes,
    json_file: Callable[..., tuple[Path, dict[str, Any], bytes]],
    record: Callable[[Path, bytes], dict[str, Any]],
    error_factory: Callable[[str], Exception],
) -> dict[str, Any] | None:
    if (
        result.get("schema_version")
        != "native_task_arena_policy_canary_session_result.v1"
        or result_path.name != "allocator_result.json"
    ):
        return None
    closeout = result.get("provider_closeout")
    watchdog = result.get("independent_watchdog")
    instance_ids = result.get("vast_instance_ids")
    watchdog_instance_lineage_valid = True
    late_watchdog_path: Path | None = None
    if instance_ids is None and isinstance(watchdog, Mapping):
        # The canonical paid allocator owns the provider identity and already
        # seals it in the caller-surviving watchdog close receipt.  Early
        # policy-canary results did not duplicate that identity at top level,
        # so billing must consume the authoritative closure field instead of
        # waiting forever for a redundant projection that cannot appear after
        # teardown.
        if watchdog.get("status") == "retained_until_hard_ttl":
            late = late_watchdog_instance(
                result_path=result_path, result=result,
                read_json=lambda path, code: json_file(path, code=code)[1],
                error_factory=error_factory,
            )
            instance_ids = [late[0]] if late is not None else None
            late_watchdog_path = late[1] if late is not None else None
            watchdog_instance_lineage_valid = late is not None
        else:
            instance_ids = watchdog.get("instance_ids")
            watchdog_instance_lineage_valid = bool(
                watchdog.get("status") == "provider_terminal"
                and watchdog.get("provider_absence_confirmed") is True
            )
    elif instance_ids is None:
        watchdog_instance_lineage_valid = False
    # A streamed attempt whose lane deferred its cleanup is proven absent by
    # the digest-bound proof resume writes (review C2).
    staged_absent, absence_proof_path = billing_staged_objects_absent(result)
    if (
        instance_ids != [instance_id]
        or result.get("status") not in {"completed", "blocked"}
        or result.get("retry_cap") != 0
        or result.get("continuing_spend_from_this_run") is not False
        or not watchdog_instance_lineage_valid
        or not isinstance(closeout, Mapping)
        or closeout.get("provider_zero_confirmed") is not True
        or closeout.get("warm_session_retained") is not False
        or not staged_absent
    ):
        raise error_factory("vast_official_terminal_result_invalid")
    paths = {
        "provider_adapter_result": Path(str(result.get("adapter_result_path") or "")),
        "teardown_manifest": Path(str(result.get("teardown_manifest_path") or "")),
        "artifact_manifest": Path(str(result.get("artifact_manifest_path") or "")),
        "post_teardown_provider_zero": (
            result_path.parent / "post_teardown_global_provider_zero.json"
        ),
    }
    loaded = {
        role: json_file(path, code=f"vast_official_policy_canary_{role}_invalid")
        for role, path in paths.items()
    }
    adapter = loaded["provider_adapter_result"][1]
    teardown = loaded["teardown_manifest"][1]
    zero = loaded["post_teardown_provider_zero"][1]
    if (
        adapter.get("vast_instance_ids") != [instance_id]
        or adapter.get("continuing_spend_from_this_run") is not False
        or teardown.get("vast_instance_ids") != [instance_id]
        or teardown.get("continuing_spend_from_this_run") is not False
        or teardown.get("runner_gpu_teardown_completed") is not True
        or zero.get("schema_version")
        != "task_evaluation_policy_canary_vast_provider_zero.v1"
        or zero.get("provider_zero_verified") is not True
        or zero.get("live_instance_count") != 0
    ):
        raise error_factory("vast_official_terminal_result_invalid")
    terminal = {
        "terminal_status": result["status"],
        "provider_absence_confirmed": True,
        "provider_zero_verified": True,
        "continuing_spend_from_this_run": False,
        "retry_cap": 0,
        "terminal_result": record(result_path, result_bytes),
    }
    for role, (path, _value, payload) in loaded.items():
        terminal[role] = record(path, payload)
    if late_watchdog_path is not None:
        path, _value, payload = json_file(
            late_watchdog_path, code="vast_official_policy_canary_late_watchdog_invalid"
        )
        terminal["independent_watchdog_terminal"] = record(path, payload)
    if absence_proof_path is not None:
        path, _value, payload = json_file(
            absence_proof_path, code="vast_official_policy_canary_staged_object_absence_proof_invalid"
        )
        terminal["staged_object_absence_proof"] = record(path, payload)
    return terminal


__all__ = ["policy_canary_terminal_evidence"]
