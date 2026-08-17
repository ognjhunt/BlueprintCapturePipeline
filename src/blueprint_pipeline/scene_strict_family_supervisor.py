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
from collections import deque
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
from .vast_official_billing_extractor import (
    VastOfficialBillingExtractionError,
    validate_vast_official_same_goal_reconciliation,
)


SCHEMA_VERSION = "scene_strict_family_checkpoint.v1"
CHAIN_SCHEMA_VERSION = "scene_strict_family_checkpoint_chain.v1"
AI_REVIEW_PROBE_KIND = "non-probe:openai-sam31-visual-review"
EXPECTED_WEBSITE_PROBE_COUNT = 17
EXPECTED_FAMILY_COUNT = 15
MAX_EVIDENCE_JSON_BYTES = 32 * 1024 * 1024
MAX_SCENE_LAUNCHES = 4096
MAX_TRUSTED_EVIDENCE_FILES = 8192
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_REACHABILITY_ROW = re.compile(r"^\| `([^`]+)` \| `([^`]+)` \|$")
_SCENE_IN_TEXT = re.compile(r"(?:scene[-_]?|publisher[-_]?scene[-_]?)(\d+)", re.I)
_NUMERIC_ID_TOKEN = re.compile(r"(?<![A-Za-z0-9])(\d{6})(?![A-Za-z0-9])")
_TASK_TOKEN = re.compile(r"(?<![A-Za-z0-9])task[-_]([a-z])(?![A-Za-z0-9])", re.I)


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

    walk(value)
    return found


def _identity_text_scene_ids(
    values: Sequence[Any], *, expected_scene_id: str | None = None
) -> set[str]:
    found: set[str] = set()
    for value in values:
        text = str(value or "")
        found.update(_SCENE_IN_TEXT.findall(text))
        if expected_scene_id is not None and expected_scene_id in _NUMERIC_ID_TOKEN.findall(text):
            found.add(expected_scene_id)
    return found


def _identity_text_task_markers(values: Sequence[Any]) -> set[str]:
    found: set[str] = set()
    for value in values:
        found.update(f"task-{item.lower()}" for item in _TASK_TOKEN.findall(str(value or "")))
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


def _profile_identity_values(
    *, run_root: Path, profile: Mapping[str, Any], request: Mapping[str, Any]
) -> list[Any]:
    """Return current-launch identifiers, excluding historical spend inputs."""

    values: list[Any] = [run_root.name]
    for item in (profile, request):
        values.extend(
            [
                item.get("profile_id"),
                item.get("launch_id"),
                item.get("run_id"),
            ]
        )
        source = item.get("source_bundle")
        spec = item.get("evaluation_run_spec")
        if isinstance(source, Mapping):
            values.extend([source.get("bundle_id"), source.get("uri")])
        if isinstance(spec, Mapping):
            values.append(spec.get("uri"))
    argv = (profile.get("allocator") or {}).get("argv")
    if isinstance(argv, list):
        # Paths and pod/profile identities carry the current scene/task. A
        # commit is deliberately excluded: a SHA such as 53b840157... is not a
        # scene, which was the production false mismatch this helper repairs.
        for index, value in enumerate(argv):
            if index and argv[index - 1] == "--expected-source-commit":
                continue
            values.append(value)
    return values


def _verified_core_identity_inputs(
    profile: Mapping[str, Any], *, required: bool
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for item in profile.get("immutable_inputs") or []:
        if not isinstance(item, Mapping) or item.get("name") not in {
            "source_bundle_manifest",
            "evaluation_run_spec",
        }:
            continue
        path = Path(str(item.get("path") or "")).expanduser().resolve()
        if (
            path.is_symlink()
            or not path.is_file()
            or item.get("digest") != _sha256_file(path)
            or path.suffix.lower() != ".json"
        ):
            if required:
                raise SceneStrictFamilyError("launch_core_identity_input_invalid")
            continue
        values.append(_json(path))
    return values


def _launch_identity(
    *,
    run_root: Path,
    profile: Mapping[str, Any],
    request: Mapping[str, Any],
    expected_scene_id: str,
    expected_tasks: Sequence[str],
    require_core_inputs: bool = True,
) -> tuple[set[str], set[str], set[str]]:
    values = _profile_identity_values(run_root=run_root, profile=profile, request=request)
    scenes = _identity_text_scene_ids(values, expected_scene_id=expected_scene_id)
    markers = _identity_text_task_markers(values)
    for core in _verified_core_identity_inputs(profile, required=require_core_inputs):
        scenes.update(_scene_ids(core))
        markers.update(_identity_text_task_markers([json.dumps(core, sort_keys=True)]))
    tasks = {task for task in expected_tasks if _task_marker(task) in markers}
    return scenes, tasks, markers


def _path_records(value: Any) -> list[tuple[Path, str]]:
    records: list[tuple[Path, str]] = []

    def walk(current: Any) -> None:
        if isinstance(current, Mapping):
            path_value = current.get("path")
            digest = current.get("sha256")
            if isinstance(path_value, str) and _valid_digest(digest):
                records.append((Path(path_value).expanduser().resolve(), str(digest)))
            for child in current.values():
                walk(child)
        elif isinstance(current, list):
            for child in current:
                walk(child)

    walk(value)
    return records


def _evidence_carrier(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() == ".json" and any(
        token in name
        for token in (
            "reconciliation",
            "official",
            "authority",
            "bundle_receipt",
            "bundle-receipt",
            "receipt",
        )
    )


def _billing_evidence_carrier(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() == ".json" and (
        "reconciliation" in name or "official" in name
    )


def _trusted_evidence_catalog(
    *,
    launch_state_root: Path,
    scene_evidence_root: Path,
    scene_id: str,
    task_ids: Sequence[str],
    probe_kinds: set[str],
) -> dict[Path, str]:
    """Follow only digest-bound evidence reachable from canonical launch state."""

    expected_markers = {_task_marker(task) for task in task_ids}
    queue: deque[tuple[Path, str]] = deque()
    trusted: dict[Path, str] = {}
    input_parent = scene_evidence_root.parent.resolve()
    if scene_evidence_root != input_parent and input_parent.is_dir():
        for candidate_root in sorted(
            path
            for path in input_parent.iterdir()
            if path.is_dir() and not path.is_symlink()
        ):
            scenes = _identity_text_scene_ids(
                [candidate_root.name], expected_scene_id=scene_id
            )
            markers = _identity_text_task_markers([candidate_root.name])
            if scenes != {scene_id} or markers - expected_markers:
                continue
            for path in candidate_root.glob("**/*.json"):
                if (
                    _billing_evidence_carrier(path)
                    and not path.is_symlink()
                    and path.stat().st_size <= MAX_EVIDENCE_JSON_BYTES
                ):
                    queue.append((path.resolve(), _sha256_file(path.resolve())))
    for profile_path in sorted(launch_state_root.glob("*/launch_profile.json")):
        run_root = profile_path.parent.resolve()
        request_path = run_root / "launch_request.json"
        if not request_path.is_file():
            continue
        try:
            profile = _json(profile_path)
            request = _json(request_path)
        except SceneStrictFamilyError:
            continue
        probe = str(_arg((profile.get("allocator") or {}).get("argv"), "--probe-kind") or "")
        if probe not in probe_kinds or not _self_digest(profile, "profile_digest"):
            continue
        try:
            scenes, _tasks, markers = _launch_identity(
                run_root=run_root,
                profile=profile,
                request=request,
                expected_scene_id=scene_id,
                expected_tasks=task_ids,
                require_core_inputs=False,
            )
        except SceneStrictFamilyError:
            continue
        if scenes != {scene_id} or markers - expected_markers:
            continue
        # Reconciliations retained with the canonical launch are trusted seeds,
        # regardless of whether the lane used the later official_billing/
        # convention or the original run-root convention.
        for path in [
            *run_root.glob("*reconciliation*.json"),
            *(run_root / "official_billing").glob("*.json"),
        ]:
            queue.append((path.resolve(), _sha256_file(path.resolve())))
        for item in profile.get("immutable_inputs") or []:
            if not isinstance(item, Mapping) or not _valid_digest(item.get("digest")):
                continue
            path = Path(str(item.get("path") or "")).expanduser().resolve()
            if _evidence_carrier(path):
                queue.append((path, str(item["digest"])))

    while queue:
        path, expected_digest = queue.popleft()
        if path in trusted:
            continue
        if len(trusted) >= MAX_TRUSTED_EVIDENCE_FILES:
            raise SceneStrictFamilyError("trusted_evidence_catalog_limit_exceeded")
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected_digest:
            # Historical launch profiles are immutable records and may outlive
            # host-retained inputs. A broken edge is not evidence; it also must
            # not poison later valid profiles for the same scene.
            continue
        trusted[path] = expected_digest
        if path.suffix.lower() != ".json" or path.stat().st_size > MAX_EVIDENCE_JSON_BYTES:
            continue
        value = _json(path)
        for child_path, child_digest in _path_records(value):
            if _evidence_carrier(child_path):
                queue.append((child_path, child_digest))
    return dict(sorted(trusted.items()))


def _billing_evidence_index(paths: Sequence[Path]) -> dict[str, tuple[Path, ...]]:
    indexed: dict[str, set[Path]] = {}
    for path in paths:
        if "reconciliation" not in path.name.lower() and "official" not in path.name.lower():
            continue
        try:
            value = _json(path)
        except SceneStrictFamilyError:
            continue
        for entry in value.get("entries") or []:
            if not isinstance(entry, Mapping):
                continue
            launch_id = entry.get("attempt_id") or entry.get("launch_id")
            terminal = entry.get("terminal_execution_evidence")
            if not launch_id and isinstance(terminal, Mapping):
                launch_id = terminal.get("launch_id")
            if isinstance(launch_id, str) and launch_id:
                indexed.setdefault(launch_id, set()).add(path.resolve())
    return {key: tuple(sorted(values)) for key, values in indexed.items()}


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
    *,
    run_root: Path,
    probe_kind: str,
    expected_commit: str,
    terminal_result: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    admission = _json(run_root / "allocator" / "admission.json")
    if (
        probe_kind == "semantic-sam31-source-tracks"
        and admission.get("schema_version") == "semantic_sam31_gpu_canary_admission.v1"
    ):
        consumption = terminal_result.get("authorization_consumption")
        if (
            admission.get("schema_version") != "semantic_sam31_gpu_canary_admission.v1"
            or admission.get("status") != "execute_ready"
            or admission.get("blockers") != []
            or admission.get("probe_kind") != probe_kind
            or admission.get("source_commit_sha") != expected_commit
            or admission.get("retry_cap") != 0
            or admission.get("watchdog_armed") is not True
            or admission.get("provider_zero_verified") is not True
            or admission.get("provider_mutations_performed") != 0
            or not _self_digest(admission, "admission_digest")
            or not isinstance(consumption, Mapping)
            or consumption.get("status") != "consumed"
            or not _valid_digest(consumption.get("authorization_digest"))
            or not _valid_digest(consumption.get("consumption_record_sha256"))
        ):
            raise SceneStrictFamilyError("exact_deployed_profile_admission_invalid")
        return admission, str(consumption["authorization_digest"])
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
    zero_path: Path | None,
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
    expected_paths: dict[str, Path] = {
        "terminal_result": result_path,
        "teardown_manifest": teardown_path,
    }
    if zero_path is not None:
        expected_paths["provider_zero"] = zero_path
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
    }.issubset(reopened) or not {
        "admission",
        "provider_billing_source_receipt",
    }.intersection(reopened):
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


def _record_path(record: Any, *, code: str, expected: Path | None = None) -> Path:
    if not isinstance(record, Mapping):
        raise SceneStrictFamilyError(code)
    path = Path(str(record.get("path") or "")).expanduser().resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or record.get("sha256") != _sha256_file(path)
        or (
            isinstance(record.get("size_bytes"), int)
            and record.get("size_bytes") != path.stat().st_size
        )
        or (expected is not None and path != expected)
    ):
        raise SceneStrictFamilyError(code)
    return path


def _authority_consumption_matches(value: Any, authority_digest: str) -> bool:
    if isinstance(value, Mapping):
        if (
            value.get("status") == "consumed"
            and value.get("authorization_digest") == authority_digest
        ):
            return True
        return any(
            _authority_consumption_matches(child, authority_digest)
            for child in value.values()
        )
    if isinstance(value, list):
        return any(_authority_consumption_matches(child, authority_digest) for child in value)
    return False


def _validate_vast_billing_entry(
    *,
    candidate_path: Path,
    candidate: Mapping[str, Any],
    launch_id: str,
    run_root: Path,
    authority_digest: str,
    teardown_path: Path,
) -> dict[str, Any] | None:
    if candidate.get("schema_version") != "blueprint.vast_official_same_goal_reconciliation.v1":
        return None
    try:
        validated = validate_vast_official_same_goal_reconciliation(candidate_path)
    except VastOfficialBillingExtractionError as exc:
        raise SceneStrictFamilyError("vast_official_billing_reconciliation_invalid") from exc
    matches = [
        entry
        for entry in validated.get("entries") or []
        if isinstance(entry, Mapping)
        and isinstance(entry.get("terminal_execution_evidence"), Mapping)
        and entry["terminal_execution_evidence"].get("launch_id") == launch_id
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise SceneStrictFamilyError("official_billing_reconciliation_ambiguous")
    entry = matches[0]
    terminal = entry["terminal_execution_evidence"]
    launch_receipt = _record_path(
        terminal.get("launch_receipt"),
        code="vast_official_launch_receipt_binding_invalid",
        expected=run_root / "launch_receipt.json",
    )
    _record_path(
        terminal.get("launch_profile"),
        code="vast_official_launch_profile_binding_invalid",
        expected=run_root / "launch_profile.json",
    )
    _record_path(
        terminal.get("launch_request"),
        code="vast_official_launch_request_binding_invalid",
        expected=run_root / "launch_request.json",
    )
    _record_path(
        terminal.get("teardown_manifest"),
        code="vast_official_teardown_binding_invalid",
        expected=teardown_path,
    )
    internal_result_path = _record_path(
        terminal.get("terminal_result"), code="vast_official_terminal_result_invalid"
    )
    internal_result = _json(internal_result_path)
    if (
        _json(launch_receipt).get("status") != "completed"
        or terminal.get("terminal_status") != "completed"
        or terminal.get("provider_zero_verified") is not True
        or terminal.get("provider_absence_confirmed") is not True
        or terminal.get("continuing_spend_from_this_run") is not False
        or terminal.get("retry_cap") != 0
        or internal_result.get("status") != "completed"
        or not _authority_consumption_matches(internal_result, authority_digest)
        or entry.get("official_charge_posted") is not True
        or not isinstance(entry.get("official_charge_usd"), (int, float))
    ):
        raise SceneStrictFamilyError("vast_official_terminal_tuple_invalid")
    return {
        "path": str(candidate_path),
        "sha256": _sha256_file(candidate_path),
        "receipt_digest": validated.get("receipt_digest"),
        "entry_digest": entry.get("entry_digest"),
        "official_cost_usd": float(entry["official_charge_usd"]),
        "evidence_schema": validated.get("schema_version"),
        "legacy_provider_zero_binding": None,
    }


def _validate_sam_prior_spend_entry(
    *,
    candidate_path: Path,
    candidate: Mapping[str, Any],
    launch_id: str,
    run_root: Path,
    authority_digest: str,
    result_path: Path,
    teardown_path: Path,
) -> dict[str, Any] | None:
    if candidate.get("schema_version") != "adp009d_prior_spend_reconciliation.v1":
        return None
    entries = candidate.get("entries")
    if not isinstance(entries, list):
        raise SceneStrictFamilyError("sam_official_billing_reconciliation_invalid")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("launch_id") == launch_id
    ]
    if not matches:
        return None
    if (
        len(matches) != 1
        or candidate.get("status") != "all_supplemental_spend_terminal_and_provider_zero"
        or not _self_digest(candidate, "receipt_digest")
    ):
        raise SceneStrictFamilyError("sam_official_billing_reconciliation_invalid")
    entry = matches[0]
    evidence = entry.get("supporting_evidence")
    if (
        entry.get("terminal_status") != "completed"
        or entry.get("authority_consumed") is not True
        or entry.get("provider_zero") is not True
        or entry.get("continuing_spend_from_this_run") is not False
        or entry.get("provider_statement_pending") is not False
        or entry.get("cost_basis") != "official_vast_posted_charge"
        or not isinstance(entry.get("actual_provider_charge_usd"), (int, float))
        or entry.get("cost_usd") != entry.get("actual_provider_charge_usd")
        or not isinstance(evidence, Mapping)
    ):
        raise SceneStrictFamilyError("sam_official_billing_entry_invalid")
    _record_path(
        evidence.get("allocator_terminal_result"),
        code="sam_terminal_result_binding_invalid",
        expected=result_path,
    )
    _record_path(
        evidence.get("launch_receipt"),
        code="sam_launch_receipt_binding_invalid",
        expected=run_root / "launch_receipt.json",
    )
    _record_path(
        evidence.get("artifact_manifest"),
        code="sam_artifact_manifest_binding_invalid",
        expected=Path(str(_json(result_path).get("artifact_manifest_path") or "")).resolve(),
    )
    _record_path(
        evidence.get("provider_teardown_manifest"),
        code="sam_teardown_binding_invalid",
        expected=teardown_path,
    )
    sync_path = _record_path(
        evidence.get("webapp_terminal_sync"),
        code="sam_webapp_binding_invalid",
        expected=run_root / "webapp_sync_succeeded.json",
    )
    guard_path = _record_path(
        evidence.get("post_terminal_global_provider_zero"),
        code="sam_global_provider_zero_invalid",
    )
    source_teardown_path = _record_path(
        evidence.get("source_teardown"), code="sam_source_teardown_invalid"
    )
    source_zero_path = _record_path(
        evidence.get("source_provider_zero"), code="sam_source_provider_zero_invalid"
    )
    billing_path = _record_path(
        evidence.get("official_vast_billing_response"),
        code="sam_official_billing_response_invalid",
    )
    source_receipt_path = _record_path(
        evidence.get("provider_billing_source_receipt"),
        code="sam_billing_source_receipt_invalid",
    )
    result = _json(result_path)
    guard = _json(guard_path)
    source_teardown = _json(source_teardown_path)
    source_zero = _json(source_zero_path)
    billing = _json(billing_path)
    billing_source = _json(source_receipt_path)
    selector = (evidence.get("official_vast_billing_response") or {}).get(
        "billing_row_selector"
    )
    expected_amount = (evidence.get("official_vast_billing_response") or {}).get(
        "expected_amount_usd"
    )
    billing_rows = billing.get("results") if isinstance(billing.get("results"), list) else []
    billing_matches = [
        row
        for row in billing_rows
        if isinstance(row, Mapping)
        and isinstance(selector, Mapping)
        and row.get("source") == selector.get("source")
        and row.get("amount") == expected_amount
        and (row.get("metadata") or {}).get("label") == selector.get("metadata_label")
    ]
    if (
        not _authority_consumption_matches(result, authority_digest)
        or result.get("provider_zero_verified") is not True
        or result.get("provider_zero_digest") != source_zero.get("provider_zero_digest")
        or guard.get("status") != "passed"
        or guard.get("blockers") != []
        or guard.get("provider_zero_verified") is not True
        or guard.get("live_instance_count") != 0
        or _timestamp(guard.get("generated_at")) is None
        or source_teardown.get("status") != "PASS"
        or source_teardown.get("provider_zero_verified") is not True
        or _timestamp(source_teardown.get("timestamp")) is None
        or _timestamp(guard.get("generated_at")) < _timestamp(source_teardown.get("timestamp"))
        or billing_source.get("status") != "reconciled"
        or not _self_digest(billing_source, "receipt_digest")
        or len(billing_matches) != 1
        or expected_amount != entry.get("actual_provider_charge_usd")
        or _json(sync_path).get("status") != "succeeded"
    ):
        raise SceneStrictFamilyError("sam_official_terminal_tuple_invalid")
    return {
        "path": str(candidate_path),
        "sha256": _sha256_file(candidate_path),
        "receipt_digest": candidate.get("receipt_digest"),
        "entry_digest": None,
        "official_cost_usd": float(entry["actual_provider_charge_usd"]),
        "evidence_schema": candidate.get("schema_version"),
        "legacy_provider_zero_binding": {
            "guard_path": str(guard_path),
            "guard_sha256": _sha256_file(guard_path),
            "provider_zero_digest": result.get("provider_zero_digest"),
        },
    }


def _official_billing(
    *,
    trusted_evidence_paths: Sequence[Path],
    run_root: Path,
    launch_id: str,
    authority_digest: str,
    result_path: Path,
    teardown_path: Path,
    zero_path: Path | None,
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
            if result is None:
                result = _validate_vast_billing_entry(
                    candidate_path=resolved,
                    candidate=value,
                    launch_id=launch_id,
                    run_root=run_root,
                    authority_digest=authority_digest,
                    teardown_path=teardown_path,
                )
            if result is None:
                result = _validate_sam_prior_spend_entry(
                    candidate_path=resolved,
                    candidate=value,
                    launch_id=launch_id,
                    run_root=run_root,
                    authority_digest=authority_digest,
                    result_path=result_path,
                    teardown_path=teardown_path,
                )
            if result is not None:
                results.append(result)
        return results

    local = matching(
        [
            *run_root.glob("*reconciliation*.json"),
            *(run_root / "official_billing").glob("*.json"),
        ]
    )
    if len(local) == 1:
        return local[0]
    if len(local) > 1:
        # A later lane can retain both the generic same-goal reconciliation and
        # the provider-specific charge seal for the same exact instance. They
        # are corroborating encodings, not two charges, when the posted amount
        # agrees exactly.
        if len({row["official_cost_usd"] for row in local}) == 1:
            return local[0]
        raise SceneStrictFamilyError("official_billing_reconciliation_ambiguous")
    external = matching(
        path
        for path in trusted_evidence_paths
        if "reconciliation" in path.name.lower() or "official" in path.name.lower()
    )
    if len(external) == 1:
        return external[0]
    if len(external) > 1:
        if len({row["official_cost_usd"] for row in external}) == 1:
            return external[0]
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
    *,
    run_root: Path,
    billing_evidence_paths: Sequence[Path],
    scene_id: str,
    task_ids: Sequence[str],
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
    observed_scenes, observed_tasks, observed_task_markers = _launch_identity(
        run_root=run_root,
        profile=profile,
        request=request,
        expected_scene_id=scene_id,
        expected_tasks=task_ids,
    )
    allowed_task_markers = {_task_marker(task) for task in task_ids}
    if observed_scenes != {scene_id}:
        blockers.append("launch_scene_identity_mismatch")
    if observed_task_markers - allowed_task_markers:
        blockers.append("launch_task_identity_mismatch")
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
        or (teardown_at is None and probe_kind != "semantic-sam31-source-tracks")
    ):
        raise SceneStrictFamilyError("teardown_manifest_values_invalid")
    zero_path: Path | None = run_root / "post_teardown_provider_zero_receipt.json"
    zero: dict[str, Any] | None = None
    if zero_path.is_file():
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
            or teardown_at is None
            or zero_at < teardown_at
            or (zero.get("teardown_manifest") or {}).get("digest")
            != _sha256_file(teardown_path)
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
    elif (
        probe_kind != "semantic-sam31-source-tracks"
        or result.get("provider_zero_verified") is not True
        or not _valid_digest(result.get("provider_zero_digest"))
        or result.get("teardown_manifest_path") != str(teardown_path)
    ):
        raise SceneStrictFamilyError("post_teardown_provider_zero_invalid")
    else:
        zero_path = None
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
        run_root=run_root,
        probe_kind=probe_kind,
        expected_commit=expected_commit,
        terminal_result=result,
    )
    billing = _official_billing(
        trusted_evidence_paths=billing_evidence_paths,
        run_root=run_root,
        launch_id=launch_id,
        authority_digest=authority_digest,
        result_path=result_path,
        teardown_path=teardown_path,
        zero_path=zero_path,
    )
    if zero is None and not isinstance(billing.get("legacy_provider_zero_binding"), Mapping):
        raise SceneStrictFamilyError("post_teardown_provider_zero_invalid")
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
        "task_ids": sorted(observed_tasks),
        "terminal_result_sha256": _sha256_file(result_path),
        "artifact_manifest_digest": manifest.get("manifest_digest"),
        "teardown_sha256": _sha256_file(teardown_path),
        "provider_zero_receipt_digest": (
            zero.get("provider_zero_receipt_digest")
            if zero is not None
            else billing["legacy_provider_zero_binding"].get("provider_zero_digest")
        ),
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
    trusted_evidence = _trusted_evidence_catalog(
        launch_state_root=launches,
        scene_evidence_root=evidence,
        scene_id=scene,
        task_ids=tasks,
        probe_kinds=probes,
    )
    trusted_evidence_paths = tuple(trusted_evidence)
    billing_evidence_index = _billing_evidence_index(trusted_evidence_paths)
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
        try:
            request = _json(run_root / "launch_request.json")
            observed_scenes, _observed_tasks, observed_markers = _launch_identity(
                run_root=run_root,
                profile=profile,
                request=request,
                expected_scene_id=scene,
                expected_tasks=tasks,
            )
        except SceneStrictFamilyError:
            continue
        if scene not in observed_scenes and scene not in _identity_text_scene_ids(
            [run_root.name], expected_scene_id=scene
        ):
            continue
        try:
            valid_launches.append(
                _validate_launch(
                    run_root=run_root,
                    billing_evidence_paths=billing_evidence_index.get(
                        str(_json(run_root / "launch_receipt.json").get("launch_id") or ""), ()
                    ),
                    scene_id=scene,
                    task_ids=tasks,
                )
            )
        except SceneStrictFamilyError as exc:
            rejected_launches.append(
                {
                    "launch_root": str(run_root),
                    "probe_kind": probe,
                    "observed_scene_ids": sorted(observed_scenes),
                    "observed_task_markers": sorted(observed_markers),
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
        "trusted_evidence_catalog": {
            "source": "canonical_launch_state_and_allowed_scene_task_roots",
            "file_count": len(trusted_evidence_paths),
            "catalog_digest": canonical_digest(
                [
                    {"path": str(path), "sha256": digest}
                    for path, digest in trusted_evidence.items()
                ]
            ),
            "arbitrary_evidence_root_files_admitted": False,
        },
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
