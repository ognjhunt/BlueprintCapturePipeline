"""Read-only strict-family closure supervisor for one production scene.

Reachability answers whether a lane *can* be launched.  This module answers a
different question: which governed families have exact terminal production
evidence for one scene and its tasks?  It never invokes the allocator, issues
authority, publishes a profile, or treats startup/provider-zero alone as
success.

The external interface is intentionally small: :func:`audit_scene_families`
reads a scene evidence root and the canonical launch-state root, validates the
full terminal tuple, and optionally appends one self-digested checkpoint to an
immutable hash chain.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_launch_dispatcher import (
    CANONICAL_ALLOCATOR_ENTRYPOINT,
    validate_launch_profile,
    validate_launch_request,
)
from .task_evaluation_launch_reconciler import (
    validated_succeeded_webapp_sync_row,
)


SCHEMA_VERSION = "scene_strict_family_checkpoint.v1"
CHAIN_SCHEMA_VERSION = "scene_strict_family_checkpoint_chain.v1"
AI_REVIEW_PROBE_KIND = "non-probe:openai-sam31-visual-review"
EXPECTED_WEBSITE_PROBE_COUNT = 17
EXPECTED_FAMILY_COUNT = 15
MAX_EVIDENCE_JSON_BYTES = 32 * 1024 * 1024
MAX_SCENE_LAUNCHES = 4096
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_REACHABILITY_ROW = re.compile(r"^\| `([^`]+)` \| `([^`]+)` \|$")
_SCENE_IN_TEXT = re.compile(r"(?:scene[-_]?|publisher[-_]?scene[-_]?)(\d+)", re.I)
_KNOWN_PUBLIC_SCENE_IN_TEXT = re.compile(r"(?<!\d)(840\d{3})(?!\d)")


class SceneStrictFamilyError(ValueError):
    """The governed inventory or append-only ledger is invalid."""


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    probe_kinds: tuple[str, ...]
    task_split: bool = False


# Only grouping and ordering live here.  Membership is derived from the live
# reachability table, whose own test derives it from allocator dispatch plus
# profile builders.  A newly reachable probe is therefore a typed inventory
# blocker, not an implicitly expanded or optimized denominator.
_FAMILY_ORDER: tuple[FamilySpec, ...] = (
    FamilySpec("semantic_source_tracks", ("semantic-sam31-source-tracks",), True),
    FamilySpec("production_ai_visual_review", (AI_REVIEW_PROBE_KIND,)),
    FamilySpec("gaussian_excision", ("adp-gaussian-excision",), True),
    FamilySpec("retained_scene_render", ("adp-retained-scene-gpu-render",)),
    FamilySpec("semantic_teacher_image_edit", ("semantic-teacher-image-edit",)),
    FamilySpec(
        "artifixer3d_paired_native_import",
        ("adp-artifixer3d-exact-support", "adp-paired-target-native-import"),
    ),
    FamilySpec("usd_content_agents", ("adp-usd-content-agents",), True),
    FamilySpec("usd_joint_agent", ("adp-usd-joint-agent",)),
    FamilySpec("simready_isaac", ("adp009b-exact-simready-isaac",)),
    FamilySpec(
        "isaac_lab_arena_native_control",
        ("adp-isaac-lab-arena-native-control",),
    ),
    FamilySpec("franka_native_microcheck", ("adp009d-franka-native-microcheck",)),
    FamilySpec(
        "native_task_arena",
        (
            "native-task-arena-construction",
            "native-task-arena-controls",
            "native-task-arena-policy",
        ),
    ),
    FamilySpec("new_site_diagnostic_canary", ("new-site-diagnostic-canary",)),
    FamilySpec("new_site_native_camera", ("new-site-native-camera",)),
    FamilySpec("reconstruction_worker_smoke", ("reconstruction-worker-smoke",)),
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_reachability_path() -> Path:
    return (
        _repository_root()
        / "docs"
        / "arm_decision_proof_v1"
        / "LIVE_LANE_REACHABILITY.md"
    )


def _json(path: Path) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > MAX_EVIDENCE_JSON_BYTES
    ):
        raise SceneStrictFamilyError(f"evidence_file_invalid:{path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SceneStrictFamilyError(f"evidence_json_invalid:{path.name}") from exc
    if not isinstance(value, Mapping):
        raise SceneStrictFamilyError(f"evidence_json_not_object:{path.name}")
    return dict(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _self_digest(value: Mapping[str, Any], field: str) -> bool:
    return _valid_digest(value.get(field)) and value.get(field) == canonical_digest(
        value, digest_field=field
    )


def _timestamp(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _arg(argv: Any, flag: str) -> str | None:
    if not isinstance(argv, list):
        return None
    try:
        index = argv.index(flag)
    except ValueError:
        return None
    return str(argv[index + 1]) if index + 1 < len(argv) else None


def _scene_ids(value: Any) -> set[str]:
    found: set[str] = set()

    def walk(current: Any, key: str = "") -> None:
        if isinstance(current, Mapping):
            for child_key, child in current.items():
                walk(child, str(child_key))
            return
        if isinstance(current, list):
            for child in current:
                walk(child, key)
            return
        text = str(current or "")
        normalized_key = key.lower().replace("-", "_")
        if normalized_key in {"scene_id", "publisher_scene_id", "source_scene_id"}:
            if text:
                found.add(text.removeprefix("scene-"))
        found.update(_SCENE_IN_TEXT.findall(text))
        found.update(_KNOWN_PUBLIC_SCENE_IN_TEXT.findall(text))

    walk(value)
    return found


def _task_marker(task_id: str) -> str:
    match = re.match(r"task[_-]([a-z0-9]+)", task_id.lower())
    return f"task-{match.group(1)}" if match else task_id.lower().replace("_", "-")


def _task_ids(value: Any, expected_tasks: Sequence[str]) -> set[str]:
    serialized = json.dumps(value, sort_keys=True).lower().replace("_", "-")
    return {
        task
        for task in expected_tasks
        if task.lower().replace("_", "-") in serialized
        or _task_marker(task) in serialized
    }


def derive_governed_families(
    reachability_path: str | Path | None = None,
) -> tuple[tuple[FamilySpec, ...], dict[str, Any]]:
    """Derive the strict 15-family set from the live 17-probe inventory."""

    source = Path(reachability_path or _default_reachability_path()).resolve()
    text = source.read_text(encoding="utf-8")
    reachable_section = text.split("## Website-reachable probe kinds", 1)
    if len(reachable_section) != 2:
        raise SceneStrictFamilyError("website_reachability_section_missing")
    table = reachable_section[1].split("## Named non-reachable probe kinds", 1)[0]
    rows = [_REACHABILITY_ROW.match(line) for line in table.splitlines()]
    probes = tuple(match.group(1) for match in rows if match)
    expected = {
        probe
        for family in _FAMILY_ORDER
        for probe in family.probe_kinds
        if probe != AI_REVIEW_PROBE_KIND
    }
    actual = set(probes)
    if len(probes) != EXPECTED_WEBSITE_PROBE_COUNT or actual != expected:
        raise SceneStrictFamilyError(
            "strict_family_probe_inventory_mismatch:"
            f"missing={sorted(expected - actual)}:unexpected={sorted(actual - expected)}"
        )
    if len(_FAMILY_ORDER) != EXPECTED_FAMILY_COUNT:
        raise SceneStrictFamilyError("strict_family_denominator_mismatch")
    derivation = {
        "reachability_path": str(source),
        "reachability_sha256": _sha256_file(source),
        "website_reachable_probe_count": len(probes),
        "website_reachable_probe_kinds": list(probes),
        "grouping_rules": {
            "native_task_arena": list(_FAMILY_ORDER[11].probe_kinds),
            "artifixer3d_paired_native_import": list(_FAMILY_ORDER[5].probe_kinds),
            "semantic_source_tracks_subruns_form_one_family": True,
        },
        "non_probe_families": [AI_REVIEW_PROBE_KIND],
        "strict_family_count": len(_FAMILY_ORDER),
    }
    derivation["derivation_digest"] = canonical_digest(
        derivation, digest_field="derivation_digest"
    )
    return _FAMILY_ORDER, derivation


def _contained(path_value: Any, root: Path, *, code: str) -> Path:
    path = Path(str(path_value or "")).expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SceneStrictFamilyError(code) from exc
    if path.is_symlink() or not path.is_file():
        raise SceneStrictFamilyError(code)
    return path


def _descriptor_file(
    descriptor: Any, root: Path, *, code: str
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(descriptor, Mapping) or descriptor.get("exists") is not True:
        raise SceneStrictFamilyError(code)
    path = _contained(descriptor.get("path"), root, code=code)
    if descriptor.get("digest") != _sha256_file(path):
        raise SceneStrictFamilyError(f"{code}_digest_mismatch")
    return path, _json(path)


def _validate_admission(
    *, run_root: Path, probe_kind: str, expected_commit: str
) -> tuple[dict[str, Any], str]:
    admission = _json(run_root / "allocator" / "admission.json")
    identity = admission.get("control_plane_identity")
    allocation = admission.get("allocation_binding")
    if (
        admission.get("schema_version") != "paid_lane_admission.v1"
        or admission.get("status") != "admitted"
        or admission.get("blockers") != []
        or admission.get("probe_kind") != probe_kind
        or admission.get("retry_cap") != 0
        or not isinstance(identity, Mapping)
        or identity.get("checkout_clean") is not True
        or identity.get("identity_probe_ran") is not True
        or identity.get("orchestrator_equals_origin_main") is not True
        or identity.get("orchestrator_equals_remote_main") is not True
        or any(
            identity.get(field) != expected_commit
            for field in (
                "orchestrator_source_commit",
                "origin_main_commit",
                "remote_main_commit",
            )
        )
        or not isinstance(allocation, Mapping)
        or allocation.get("probe_kind") != probe_kind
        or allocation.get("orchestrator_source_commit") != expected_commit
        or allocation.get("retry_cap") != 0
        or admission.get("allocation_binding_digest") != canonical_digest(allocation)
        or not _valid_digest(allocation.get("paid_attempt_authority_digest"))
    ):
        raise SceneStrictFamilyError("exact_deployed_profile_admission_invalid")
    return admission, str(allocation["paid_attempt_authority_digest"])


def _json_path(value: Any, path: Any) -> Any:
    current = value
    if not isinstance(path, list) or not path:
        raise SceneStrictFamilyError("official_billing_binding_path_invalid")
    for part in path:
        if isinstance(part, str) and isinstance(current, Mapping) and part in current:
            current = current[part]
        elif (
            isinstance(part, int)
            and not isinstance(part, bool)
            and isinstance(current, list)
            and 0 <= part < len(current)
        ):
            current = current[part]
        else:
            raise SceneStrictFamilyError("official_billing_binding_path_invalid")
    return current


def _validate_billing_entry(
    *,
    candidate_path: Path,
    candidate: Mapping[str, Any],
    launch_id: str,
    run_root: Path,
    authority_digest: str,
    result_path: Path,
    teardown_path: Path,
    zero_path: Path,
) -> dict[str, Any] | None:
    entries = candidate.get("entries")
    if not isinstance(entries, list):
        return None
    entry = next(
        (
            row
            for row in entries
            if isinstance(row, Mapping) and row.get("attempt_id") == launch_id
        ),
        None,
    )
    if entry is None:
        return None
    if (
        candidate.get("blockers") != []
        or candidate.get("entry_count") != len(entries)
        or not _self_digest(candidate, "receipt_digest")
        or entry.get("evidence_kind") != "fully_bound_official_billing"
        or entry.get("continuing_spend_from_this_run") is not False
        or entry.get("provider_zero_confirmed") is not True
        or entry.get("authority_digest") != authority_digest
        or sum(
            isinstance(row, Mapping)
            and row.get("authority_digest") == authority_digest
            for row in entries
        )
        != 1
        or not isinstance(entry.get("cost_usd"), (int, float))
        or isinstance(entry.get("cost_usd"), bool)
        or not math.isfinite(float(entry["cost_usd"]))
        or float(entry["cost_usd"]) < 0
        or not _self_digest(entry, "entry_digest")
    ):
        raise SceneStrictFamilyError("official_billing_entry_invalid")
    sources = entry.get("source_receipts")
    if not isinstance(sources, list):
        raise SceneStrictFamilyError("official_billing_sources_invalid")
    reopened: dict[str, Any] = {}
    expected_paths = {
        "terminal_result": result_path,
        "teardown_manifest": teardown_path,
        "provider_zero": zero_path,
    }
    for source in sources:
        if not isinstance(source, Mapping):
            raise SceneStrictFamilyError("official_billing_sources_invalid")
        role = str(source.get("role") or "")
        record = source.get("record")
        if not role or role in reopened or not isinstance(record, Mapping):
            raise SceneStrictFamilyError("official_billing_sources_invalid")
        path = Path(str(record.get("path") or "")).expanduser().resolve()
        if path.is_symlink() or not path.is_file() or record.get("sha256") != _sha256_file(path):
            raise SceneStrictFamilyError("official_billing_source_digest_invalid")
        if role in expected_paths and path != expected_paths[role]:
            raise SceneStrictFamilyError(f"official_billing_{role}_binding_invalid")
        reopened[role] = _json(path)
    if not {
        "terminal_result",
        "teardown_manifest",
        "provider_zero",
        "official_billing_response",
        "admission",
    }.issubset(reopened):
        raise SceneStrictFamilyError("official_billing_terminal_tuple_incomplete")
    for binding in entry.get("bindings") or []:
        if not isinstance(binding, Mapping):
            raise SceneStrictFamilyError("official_billing_binding_invalid")
        role = str(binding.get("source_role") or "")
        if role not in reopened or _json_path(reopened[role], binding.get("json_path")) != binding.get(
            "expected_value"
        ):
            raise SceneStrictFamilyError("official_billing_binding_invalid")
    return {
        "path": str(candidate_path),
        "sha256": _sha256_file(candidate_path),
        "receipt_digest": candidate.get("receipt_digest"),
        "entry_digest": entry.get("entry_digest"),
        "official_cost_usd": float(entry["cost_usd"]),
    }


def _official_billing(
    *,
    evidence_root: Path,
    run_root: Path,
    launch_id: str,
    authority_digest: str,
    result_path: Path,
    teardown_path: Path,
    zero_path: Path,
) -> dict[str, Any]:
    def matching(paths: Any) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        seen: set[Path] = set()
        for path in paths:
            resolved = path.resolve()
            if resolved in seen or not resolved.is_file():
                continue
            seen.add(resolved)
            value = _json(resolved)
            result = _validate_billing_entry(
                candidate_path=resolved,
                candidate=value,
                launch_id=launch_id,
                run_root=run_root,
                authority_digest=authority_digest,
                result_path=result_path,
                teardown_path=teardown_path,
                zero_path=zero_path,
            )
            if result is not None:
                results.append(result)
        return results

    local = matching((run_root / "official_billing").glob("*.json"))
    if len(local) == 1:
        return local[0]
    if len(local) > 1:
        raise SceneStrictFamilyError("official_billing_reconciliation_ambiguous")
    external = matching(evidence_root.glob("**/*reconciliation*.json"))
    if len(external) == 1:
        return external[0]
    if len(external) > 1:
        raise SceneStrictFamilyError("official_billing_reconciliation_ambiguous")
    raise SceneStrictFamilyError("official_billing_reconciliation_missing")


def _validate_openai_settlement(root: Path, *, run_id: str | None = None) -> dict[str, Any]:
    for path in root.glob("**/openai_official_cost_run_settlement.v1.json"):
        value = _json(path)
        if run_id is not None and value.get("run_id") != run_id:
            continue
        if (
            value.get("schema_version") != "openai_official_cost_run_settlement.v1"
            or value.get("status") != "reconciled"
            or value.get("cost_is_final") is not True
            or value.get("strict_official_billing_satisfied") is not True
            or value.get("candidate_reported_cost_accepted") is not False
            or not _self_digest(value, "settlement_receipt_digest")
        ):
            raise SceneStrictFamilyError("openai_official_billing_settlement_invalid")
        return {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path.resolve()),
            "settlement_receipt_digest": value["settlement_receipt_digest"],
            "actual_cost_usd": value.get("actual_cost_usd"),
        }
    raise SceneStrictFamilyError("openai_official_billing_settlement_missing")


def _validate_launch(
    *, run_root: Path, evidence_root: Path, scene_id: str, task_ids: Sequence[str]
) -> dict[str, Any]:
    request = _json(run_root / "launch_request.json")
    profile = _json(run_root / "launch_profile.json")
    receipt = _json(run_root / "launch_receipt.json")
    binding = _json(run_root / "launch_binding.json")
    launch_id = str(receipt.get("launch_id") or "")
    argv = (profile.get("allocator") or {}).get("argv")
    probe_kind = str(_arg(argv, "--probe-kind") or "")
    expected_commit = str(_arg(argv, "--expected-source-commit") or "")
    blockers: list[str] = []
    blockers.extend(validate_launch_request(request))
    blockers.extend(validate_launch_profile(profile))
    if (
        not launch_id
        or receipt.get("schema_version") != "task_evaluation_launch_receipt.v1"
        or receipt.get("status") != "completed"
        or receipt.get("blockers") != []
        or receipt.get("execute_requested") is not True
        or receipt.get("allocator_exit_code") != 0
        or receipt.get("canonical_allocator") != CANONICAL_ALLOCATOR_ENTRYPOINT
        or not _self_digest(receipt, "receipt_digest")
        or request.get("launch_id") != launch_id
        or request.get("run_id") != receipt.get("run_id")
        or request.get("request_digest") != receipt.get("request_digest")
        or request.get("source_bundle") != profile.get("source_bundle")
        or request.get("evaluation_run_spec") != profile.get("evaluation_run_spec")
        or request.get("required_controls") != profile.get("required_controls")
        or request.get("claim_ceiling") != profile.get("claim_ceiling")
        or request.get("launch_profile_digest") != profile.get("profile_digest")
        or receipt.get("launch_profile_digest") != profile.get("profile_digest")
        or not _self_digest(profile, "profile_digest")
        or not _COMMIT.fullmatch(expected_commit)
        or binding.get("schema_version") != "task_evaluation_launch_binding.v1"
        or binding.get("launch_id") != launch_id
        or binding.get("run_id") != receipt.get("run_id")
        or binding.get("request_digest") != receipt.get("request_digest")
        or binding.get("profile_digest") != profile.get("profile_digest")
        or receipt.get("binding_digest") != binding.get("binding_digest")
        or binding.get("execute_requested") is not True
        or not _self_digest(binding, "binding_digest")
    ):
        blockers.append("launch_identity_or_terminal_receipt_invalid")
    observed_scenes = _scene_ids({"request": request, "profile": profile})
    if scene_id not in observed_scenes or observed_scenes - {scene_id}:
        blockers.append("launch_scene_identity_mismatch")
    terminal = receipt.get("terminal_evidence")
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("status") != "passed"
        or terminal.get("blockers") != []
    ):
        blockers.append("terminal_result_contract_not_passed")
    if blockers:
        raise SceneStrictFamilyError(",".join(sorted(set(blockers))))
    result_path, result = _descriptor_file(
        terminal.get("result"), run_root, code="terminal_result_invalid"
    )
    if (
        result.get("status") != "completed"
        or result.get("blockers") != []
        or result.get("continuing_spend_from_this_run") is not False
        or result.get("retry_cap") != 0
    ):
        raise SceneStrictFamilyError("terminal_result_values_invalid")
    artifacts = terminal.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise SceneStrictFamilyError("terminal_artifacts_invalid")
    manifest_path, manifest = _descriptor_file(
        artifacts.get("artifact_manifest_path"),
        run_root,
        code="artifact_manifest_invalid",
    )
    if (
        manifest.get("schema_version") != "task_evaluation_artifact_manifest.v1"
        or manifest.get("status") != "completed"
        or manifest.get("blockers") != []
        or manifest.get("file_count", 0) < 1
        or not _self_digest(manifest, "manifest_digest")
        or not set(manifest.get("required_roles") or []).issubset(
            set(manifest.get("observed_roles") or [])
        )
        or (manifest.get("binding") or {}).get("retry_cap") != 0
    ):
        raise SceneStrictFamilyError("artifact_manifest_values_invalid")
    attempt_root = manifest_path.parent.resolve()
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != manifest.get("file_count"):
        raise SceneStrictFamilyError("artifact_manifest_files_invalid")
    observed_total = 0
    for item in files:
        if not isinstance(item, Mapping):
            raise SceneStrictFamilyError("artifact_manifest_files_invalid")
        artifact = (attempt_root / str(item.get("relative_path") or "")).resolve()
        try:
            artifact.relative_to(attempt_root)
        except ValueError as exc:
            raise SceneStrictFamilyError("artifact_manifest_file_outside_attempt") from exc
        if (
            artifact.is_symlink()
            or not artifact.is_file()
            or artifact.stat().st_size != item.get("size_bytes")
            or _sha256_file(artifact) != item.get("sha256")
        ):
            raise SceneStrictFamilyError("artifact_manifest_file_digest_invalid")
        observed_total += artifact.stat().st_size
    if observed_total != manifest.get("total_size_bytes"):
        raise SceneStrictFamilyError("artifact_manifest_total_size_invalid")
    teardown_path, teardown = _descriptor_file(
        artifacts.get("teardown_manifest_path"),
        run_root,
        code="teardown_manifest_invalid",
    )
    teardown_at = _timestamp(teardown.get("generated_at"))
    if (
        teardown.get("schema_version") != "vast_teardown_manifest.v1"
        or teardown.get("status") != "completed"
        or teardown.get("continuing_spend_from_this_run") is not False
        or teardown_at is None
    ):
        raise SceneStrictFamilyError("teardown_manifest_values_invalid")
    zero_path = run_root / "post_teardown_provider_zero_receipt.json"
    zero = _json(zero_path)
    zero_at = _timestamp(zero.get("observed_at"))
    guard_record = zero.get("independent_guard_snapshot")
    if (
        zero.get("schema_version") != "task_evaluation_post_teardown_provider_zero.v1"
        or zero.get("status") != "provider_zero_confirmed"
        or zero.get("provider_zero_verified") is not True
        or zero.get("continuing_spend_from_this_run") is not False
        or zero.get("blockers") != []
        or zero.get("launch_id") != launch_id
        or zero.get("run_id") != receipt.get("run_id")
        or zero.get("request_digest") != receipt.get("request_digest")
        or zero.get("receipt_digest") != receipt.get("receipt_digest")
        or zero.get("launch_profile_digest") != profile.get("profile_digest")
        or not _self_digest(zero, "provider_zero_receipt_digest")
        or zero_at is None
        or zero_at < teardown_at
        or (zero.get("teardown_manifest") or {}).get("digest") != _sha256_file(teardown_path)
        or not isinstance(guard_record, Mapping)
    ):
        raise SceneStrictFamilyError("post_teardown_provider_zero_invalid")
    guard_path = _contained(
        guard_record.get("path"), run_root, code="provider_zero_guard_snapshot_invalid"
    )
    guard = _json(guard_path)
    guard_at = _timestamp(guard_record.get("source_guard_generated_at"))
    if (
        guard_record.get("snapshot_digest") != guard.get("snapshot_digest")
        or not _self_digest(guard, "snapshot_digest")
        or guard_at is None
        or guard_at < teardown_at
        or guard_at > zero_at
    ):
        raise SceneStrictFamilyError("provider_zero_guard_snapshot_invalid")
    sync = _json(run_root / "webapp_sync_succeeded.json")
    try:
        sync_row = validated_succeeded_webapp_sync_row(receipt=receipt, attempt=sync)
    except (OSError, ValueError) as exc:
        raise SceneStrictFamilyError("webapp_terminal_binding_invalid") from exc
    if (
        sync_row.get("webapp_record_bound") is not True
        or sync_row.get("website_trigger_proven") is not True
    ):
        raise SceneStrictFamilyError("webapp_terminal_binding_invalid")
    _admission, authority_digest = _validate_admission(
        run_root=run_root, probe_kind=probe_kind, expected_commit=expected_commit
    )
    billing = _official_billing(
        evidence_root=evidence_root,
        run_root=run_root,
        launch_id=launch_id,
        authority_digest=authority_digest,
        result_path=result_path,
        teardown_path=teardown_path,
        zero_path=zero_path,
    )
    openai_billing = None
    if probe_kind == "semantic-teacher-image-edit":
        openai_billing = _validate_openai_settlement(run_root)
    return {
        "status": "strict_terminal_tuple_valid",
        "probe_kind": probe_kind,
        "launch_id": launch_id,
        "run_id": receipt.get("run_id"),
        "request_digest": receipt.get("request_digest"),
        "profile_id": profile.get("profile_id"),
        "profile_digest": profile.get("profile_digest"),
        "deployed_source_commit": expected_commit,
        "scene_id": scene_id,
        "task_ids": sorted(_task_ids({"request": request, "profile": profile}, task_ids)),
        "terminal_result_sha256": _sha256_file(result_path),
        "artifact_manifest_digest": manifest.get("manifest_digest"),
        "teardown_sha256": _sha256_file(teardown_path),
        "provider_zero_receipt_digest": zero.get("provider_zero_receipt_digest"),
        "webapp_sync_result_digest": sync.get("sync_result_digest"),
        "authority_digest": authority_digest,
        "official_billing": billing,
        "openai_official_billing": openai_billing,
        "tuple": {
            "exact_deployed_profile_identity": True,
            "terminal_result_blockers_empty": True,
            "artifact_manifest_valid": True,
            "teardown_terminal": True,
            "official_billing_reconciled": True,
            "fresh_provider_zero": True,
            "webapp_api_causality_and_binding": True,
            "authority_consumed_exactly_once": True,
        },
    }


def _validate_ai_review(evidence_root: Path, scene_id: str) -> dict[str, Any]:
    for path in evidence_root.glob("**/*.json"):
        try:
            receipt = _json(path)
        except SceneStrictFamilyError:
            continue
        if receipt.get("schema_version") != "public_scene_sam31_track_selection_ai_visual_review.v1":
            continue
        reviewer = receipt.get("reviewer")
        candidate_record = receipt.get("candidate")
        execution_record = receipt.get("review_execution_receipt")
        if (
            receipt.get("status") != "selected_tracks_ai_visual_review_accepted"
            or receipt.get("decision") != "accepted"
            or receipt.get("all_selected_tracks_accepted") is not True
            or receipt.get("task_count") != 2
            or (receipt.get("review_scope") or {}).get("review_frame_count") != 16
            or (receipt.get("claim_boundary") or {}).get("ai_visual_review_completed") is not True
            or not isinstance(reviewer, Mapping)
            or reviewer.get("kind") != "ai"
            or reviewer.get("runtime") != "openai_agents_sdk"
            or not isinstance(candidate_record, Mapping)
            or not isinstance(execution_record, Mapping)
            or not _self_digest(receipt, "receipt_digest")
        ):
            raise SceneStrictFamilyError("production_ai_visual_review_invalid")
        candidate_path = Path(str(candidate_record.get("path") or "")).resolve()
        execution_path = Path(str(execution_record.get("path") or "")).resolve()
        if (
            not candidate_path.is_file()
            or candidate_record.get("sha256") != _sha256_file(candidate_path)
            or not execution_path.is_file()
            or execution_record.get("sha256") != _sha256_file(execution_path)
        ):
            raise SceneStrictFamilyError("production_ai_visual_review_binding_invalid")
        candidate = _json(candidate_path)
        execution = _json(execution_path)
        if (
            scene_id not in _scene_ids({"receipt": receipt, "candidate": candidate})
            or _scene_ids({"receipt": receipt, "candidate": candidate}) - {scene_id}
            or candidate.get("candidate_digest") != candidate_record.get("candidate_digest")
            or not _self_digest(candidate, "candidate_digest")
            or execution.get("execution_receipt_digest")
            != execution_record.get("execution_receipt_digest")
            or not _self_digest(execution, "execution_receipt_digest")
            or execution.get("provider_called") is not True
            or execution.get("response_store") is not False
            or execution.get("tracing_disabled") is not True
            or execution.get("trace_sensitive_data_included") is not False
            or execution.get("decision") != "accepted"
        ):
            raise SceneStrictFamilyError("production_ai_visual_review_execution_invalid")
        inventory = execution.get("frame_inventory")
        overlay_digests = [
            row.get("overlay_sha256")
            for row in inventory or []
            if isinstance(row, Mapping)
        ]
        if (
            not isinstance(inventory, list)
            or len(inventory) != 16
            or len(overlay_digests) != 16
            or len(set(overlay_digests)) != 16
            or any(not _valid_digest(item) for item in overlay_digests)
        ):
            raise SceneStrictFamilyError("production_ai_visual_review_frames_invalid")
        settlement = _validate_openai_settlement(
            execution_path.parent, run_id=str(execution.get("run_id") or "")
        )
        return {
            "status": "strict_terminal_tuple_valid",
            "probe_kind": AI_REVIEW_PROBE_KIND,
            "receipt_path": str(path.resolve()),
            "receipt_digest": receipt["receipt_digest"],
            "candidate_digest": candidate["candidate_digest"],
            "execution_receipt_digest": execution["execution_receipt_digest"],
            "openai_official_billing": settlement,
            "tuple": {
                "exact_rights_and_candidate_identity": True,
                "typed_ai_decision_accepted": True,
                "sixteen_unique_frames_reviewed": True,
                "official_billing_reconciled": True,
                "webapp_api_causality_and_binding": "not_applicable_non_probe",
                "provider_zero": "not_applicable_non_gpu_review",
            },
        }
    raise SceneStrictFamilyError("production_ai_visual_review_terminal_evidence_missing")


def _verify_checkpoint_chain(ledger_root: Path, *, scene_id: str) -> tuple[int, str | None]:
    checkpoints = sorted((ledger_root / scene_id).glob("*.json"))
    previous: str | None = None
    for index, path in enumerate(checkpoints, start=1):
        value = _json(path)
        if (
            value.get("schema_version") != SCHEMA_VERSION
            or value.get("chain_schema_version") != CHAIN_SCHEMA_VERSION
            or value.get("sequence") != index
            or value.get("scene_id") != scene_id
            or value.get("previous_checkpoint_digest") != previous
            or not _self_digest(value, "checkpoint_digest")
            or path.name != f"{index:06d}-{value['checkpoint_digest'][7:23]}.json"
        ):
            raise SceneStrictFamilyError("strict_family_checkpoint_chain_invalid")
        previous = str(value["checkpoint_digest"])
    return len(checkpoints), previous


def _append_checkpoint(ledger_root: Path, checkpoint: Mapping[str, Any]) -> Path:
    scene_id = str(checkpoint["scene_id"])
    scene_root = ledger_root / scene_id
    scene_root.mkdir(parents=True, exist_ok=True)
    lock_descriptor = os.open(scene_root / ".append.lock", os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(lock_descriptor, "r+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        count, previous = _verify_checkpoint_chain(ledger_root, scene_id=scene_id)
        value = dict(checkpoint)
        value.update(
            {
                "schema_version": SCHEMA_VERSION,
                "chain_schema_version": CHAIN_SCHEMA_VERSION,
                "sequence": count + 1,
                "previous_checkpoint_digest": previous,
            }
        )
        value["checkpoint_digest"] = canonical_digest(
            value, digest_field="checkpoint_digest"
        )
        destination = (
            scene_root / f"{count + 1:06d}-{value['checkpoint_digest'][7:23]}.json"
        )
        payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
        try:
            descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o440)
        except FileExistsError as exc:
            raise SceneStrictFamilyError("strict_family_checkpoint_append_conflict") from exc
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(scene_root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        _verify_checkpoint_chain(ledger_root, scene_id=scene_id)
        return destination


def audit_scene_families(
    *,
    scene_id: str,
    task_ids: Sequence[str],
    evidence_root: str | Path,
    launch_state_root: str | Path,
    ledger_root: str | Path | None = None,
    reachability_path: str | Path | None = None,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Validate one scene and optionally append its honest closure checkpoint."""

    scene = str(scene_id or "").strip()
    tasks = tuple(dict.fromkeys(str(item).strip() for item in task_ids if str(item).strip()))
    evidence = Path(evidence_root).expanduser().resolve()
    launches = Path(launch_state_root).expanduser().resolve()
    if not scene or not tasks or not evidence.is_dir() or not launches.is_dir():
        raise SceneStrictFamilyError("scene_supervisor_input_invalid")
    families, derivation = derive_governed_families(reachability_path)
    probes = {probe for family in families for probe in family.probe_kinds}
    valid_launches: list[dict[str, Any]] = []
    rejected_launches: list[dict[str, Any]] = []
    receipt_paths = sorted(launches.glob("*/launch_receipt.json"))
    if len(receipt_paths) > MAX_SCENE_LAUNCHES:
        raise SceneStrictFamilyError("scene_launch_history_limit_exceeded")
    for receipt_path in receipt_paths:
        run_root = receipt_path.parent.resolve()
        try:
            profile = _json(run_root / "launch_profile.json")
            probe = str(_arg((profile.get("allocator") or {}).get("argv"), "--probe-kind") or "")
        except SceneStrictFamilyError:
            continue
        if probe not in probes or probe == AI_REVIEW_PROBE_KIND:
            continue
        if scene not in _scene_ids(profile) and scene not in run_root.name:
            continue
        try:
            valid_launches.append(
                _validate_launch(
                    run_root=run_root,
                    evidence_root=evidence,
                    scene_id=scene,
                    task_ids=tasks,
                )
            )
        except SceneStrictFamilyError as exc:
            rejected_launches.append(
                {
                    "launch_root": str(run_root),
                    "probe_kind": probe,
                    "blockers": sorted(set(str(exc).split(","))),
                }
            )
    ai_review: dict[str, Any] | None = None
    ai_blocker: str | None = None
    try:
        ai_review = _validate_ai_review(evidence, scene)
    except SceneStrictFamilyError as exc:
        ai_blocker = str(exc)

    family_rows: list[dict[str, Any]] = []
    next_checkpoint: dict[str, Any] | None = None
    for family in families:
        blockers: list[str] = []
        qualifying: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        if family.probe_kinds == (AI_REVIEW_PROBE_KIND,):
            if ai_review is None:
                blockers.append(ai_blocker or "production_ai_visual_review_unproven")
                missing.append({"probe_kind": AI_REVIEW_PROBE_KIND})
            else:
                qualifying.append(ai_review)
        else:
            for probe in family.probe_kinds:
                candidates = [row for row in valid_launches if row["probe_kind"] == probe]
                if family.task_split:
                    for task in tasks:
                        task_candidates = [row for row in candidates if task in row["task_ids"]]
                        if task_candidates:
                            qualifying.append(task_candidates[-1])
                        else:
                            missing.append({"probe_kind": probe, "task_id": task})
                            blockers.append(f"strict_terminal_launch_missing:{probe}:{task}")
                elif candidates:
                    qualifying.append(candidates[-1])
                else:
                    missing.append({"probe_kind": probe})
                    blockers.append(f"strict_terminal_launch_missing:{probe}")
        completed = not blockers
        row = {
            "family_id": family.family_id,
            "status": "strict_terminal_complete" if completed else "unproven",
            "ordered_probe_kinds": list(family.probe_kinds),
            "task_split": family.task_split,
            "qualified_launch_ids": sorted(
                {
                    str(item["launch_id"])
                    for item in qualifying
                    if item.get("launch_id")
                }
            ),
            "evidence": qualifying,
            "missing_checkpoints": missing,
            "blockers": sorted(set(blockers)),
        }
        family_rows.append(row)
        if not completed and next_checkpoint is None:
            next_checkpoint = {
                "family_id": family.family_id,
                **missing[0],
                "blockers": row["blockers"],
            }

    completed_count = sum(row["status"] == "strict_terminal_complete" for row in family_rows)
    checkpoint: dict[str, Any] = {
        "scene_id": scene,
        "task_ids": list(tasks),
        "observed_at": observed_at or datetime.now(timezone.utc).isoformat(),
        "governed_family_derivation": derivation,
        "strict_completed_family_count": completed_count,
        "strict_family_denominator": len(family_rows),
        "status": "strict_terminal_complete" if completed_count == len(family_rows) else "in_progress",
        "families": family_rows,
        "next_unproven_checkpoint": next_checkpoint,
        "rejected_candidate_launches": rejected_launches,
        "authority_boundary": {
            "read_only": True,
            "allocator_invoked": False,
            "provider_mutation_performed": False,
            "publication_performed": False,
            "authority_issued": False,
            "automatic_retry_performed": False,
            "reachable_or_started_counts_as_complete": False,
        },
    }
    checkpoint["audit_digest"] = canonical_digest(checkpoint, digest_field="audit_digest")
    if ledger_root is not None:
        destination = _append_checkpoint(Path(ledger_root).expanduser().resolve(), checkpoint)
        return _json(destination)
    return checkpoint


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--launch-state-root", required=True)
    parser.add_argument("--ledger-root", required=True)
    parser.add_argument("--reachability-path")
    args = parser.parse_args(argv)
    try:
        result = audit_scene_families(
            scene_id=args.scene_id,
            task_ids=args.task_id,
            evidence_root=args.evidence_root,
            launch_state_root=args.launch_state_root,
            ledger_root=args.ledger_root,
            reachability_path=args.reachability_path,
        )
    except (OSError, SceneStrictFamilyError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "allocator_invoked": False,
                    "provider_mutation_performed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "strict_terminal_complete" else 1


__all__ = [
    "AI_REVIEW_PROBE_KIND",
    "EXPECTED_FAMILY_COUNT",
    "FamilySpec",
    "SceneStrictFamilyError",
    "audit_scene_families",
    "derive_governed_families",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
