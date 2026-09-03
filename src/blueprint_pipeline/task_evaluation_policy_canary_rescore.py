"""Offline, evidence-bound rescoring for retained policy-canary results.

This module never edits the recovered provider result or its original score
receipts.  It verifies those bytes, replays the deployed deterministic scorer
from each episode's frozen task spec and native state trace, and emits a
derived correction overlay that a publisher can validate against the original
``completed_unqualified`` result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .adp_task_scoring import score_task_episode_from_spec
from .decision_evidence_contracts import (
    canonical_digest,
    canonical_json,
    cross_runtime_canonical_digest,
)
from .native_task_arena_policy_canary_session import validate_session_result


EPISODE_RECEIPT_SCHEMA_VERSION = "task_evaluation_policy_canary_rescore_episode.v1"
CORRECTION_SCHEMA_VERSION = "task_evaluation_policy_canary_score_correction.v1"
SCORER_IDENTITY_SCHEMA_VERSION = "task_evaluation_deterministic_scorer_identity.v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SCORER_SOURCES = (
    "src/blueprint_pipeline/adp_task_scoring.py",
    "src/blueprint_pipeline/adp009d_task_scoring.py",
    "src/blueprint_pipeline/articulation_graph_contract.py",
    "src/blueprint_pipeline/decision_evidence_contracts.py",
)
_PUBLICATION_CONSTRAINTS = {
    "source_status": "completed_unqualified",
    "corrected_status": "completed_unqualified",
    "episode_identity_must_be_unchanged": True,
    "artifact_inventory_must_be_unchanged": True,
    "original_provider_result_must_be_retained": True,
    "original_score_receipts_must_be_retained": True,
    "allowed_score_overlay_fields": [
        "episode.score",
        "deterministic_score_digest",
        "scoring_version_digest",
    ],
    "correction_authority": "derived_deterministic_rescore_receipt",
}


class PolicyCanaryRescoreError(ValueError):
    """The retained result cannot safely be rescored or published."""


def _mapping(value: Any, *, code: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyCanaryRescoreError(code)
    return dict(value)


def _digest(value: Any) -> bool:
    return bool(_DIGEST.fullmatch(str(value or "")))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _inside(root: Path, relative_path: Any, *, code: str) -> Path:
    relative = str(relative_path or "")
    if not relative or relative.startswith("/"):
        raise PolicyCanaryRescoreError(code)
    unresolved = root / relative
    if unresolved.is_symlink():
        raise PolicyCanaryRescoreError(code)
    path = unresolved.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PolicyCanaryRescoreError(code) from exc
    if not path.is_file():
        raise PolicyCanaryRescoreError(code)
    return path


def _load_object(path: Path, *, code: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PolicyCanaryRescoreError(code)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyCanaryRescoreError(code) from exc
    return _mapping(value, code=code)


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes() != payload:
            raise PolicyCanaryRescoreError(f"policy_canary_rescore_immutable_conflict:{path.name}")


def _source_episode_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    episode = _mapping(row.get("episode"), code="policy_canary_rescore_source_episode_missing")
    return {
        "candidate_id": row.get("candidate_id"),
        "cell_id": row.get("cell_id"),
        "seed": row.get("seed"),
        "run_kind": row.get("run_kind"),
        "claim_ceiling": row.get("claim_ceiling"),
        "family": row.get("family"),
        "checkpoint_digest": row.get("checkpoint_digest"),
        "runtime_identity_digest": row.get("runtime_identity_digest"),
        "reset_state_digest": row.get("reset_state_digest"),
        "scene_revision_digest": row.get("scene_revision_digest"),
        "container_identity_digest": row.get("container_identity_digest"),
        "episode_id": episode.get("episode_id"),
        "source_episode_digest": canonical_digest({"value": row}),
        "source_episode_receipt_digest": episode.get("receipt_digest"),
        "task_spec_digest": episode.get("task_spec_digest"),
        "state_trace_digest": row.get("state_trace_digest"),
        "evidence_artifact_bindings_digest": canonical_digest(row.get("evidence_artifacts") or {}),
    }


def _episode_key(row: Mapping[str, Any]) -> str:
    return "\0".join(
        (
            str(row.get("candidate_id") or ""),
            str(row.get("cell_id") or ""),
            str(row.get("seed") if row.get("seed") is not None else ""),
        )
    )


def _verify_artifact_inventory(
    result: Mapping[str, Any], *, evidence_root: Path
) -> dict[str, dict[str, Any]]:
    inventory = result.get("artifact_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise PolicyCanaryRescoreError("policy_canary_rescore_artifact_inventory_invalid")
    if result.get("artifact_inventory_digest") != canonical_digest({"value": inventory}):
        raise PolicyCanaryRescoreError("policy_canary_rescore_artifact_inventory_digest_mismatch")
    records: dict[str, dict[str, Any]] = {}
    for raw in inventory:
        record = _mapping(raw, code="policy_canary_rescore_artifact_inventory_invalid")
        relative = str(record.get("relative_path") or "")
        if relative in records:
            raise PolicyCanaryRescoreError(
                "policy_canary_rescore_artifact_inventory_duplicate_path"
            )
        path = _inside(
            evidence_root,
            relative,
            code="policy_canary_rescore_artifact_missing",
        )
        if (
            not _digest(record.get("sha256"))
            or _sha256(path) != record.get("sha256")
            or isinstance(record.get("size_bytes"), bool)
            or record.get("size_bytes") != path.stat().st_size
        ):
            raise PolicyCanaryRescoreError(
                f"policy_canary_rescore_artifact_digest_mismatch:{relative}"
            )
        records[relative] = record
    return records


def _verified_episode_artifact(
    *,
    row: Mapping[str, Any],
    role: str,
    inventory: Mapping[str, Mapping[str, Any]],
    evidence_root: Path,
) -> tuple[dict[str, Any], Path]:
    evidence = _mapping(
        row.get("evidence_artifacts"),
        code="policy_canary_rescore_episode_artifacts_missing",
    )
    record = _mapping(
        evidence.get(role),
        code=f"policy_canary_rescore_episode_artifact_missing:{role}",
    )
    relative = str(record.get("relative_path") or "")
    indexed = inventory.get(relative)
    if indexed is None or any(
        record.get(field) != indexed.get(field)
        for field in ("role", "relative_path", "size_bytes", "sha256")
    ):
        raise PolicyCanaryRescoreError(f"policy_canary_rescore_episode_artifact_unbound:{role}")
    return record, _inside(
        evidence_root,
        relative,
        code=f"policy_canary_rescore_episode_artifact_missing:{role}",
    )


def _verify_source_episode(
    row: Mapping[str, Any],
    *,
    inventory: Mapping[str, Mapping[str, Any]],
    evidence_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if row.get("status") != "completed":
        raise PolicyCanaryRescoreError("policy_canary_rescore_requires_completed_episode")
    episode = _mapping(row.get("episode"), code="policy_canary_rescore_source_episode_missing")
    receipt_source = dict(episode)
    parity = receipt_source.pop("embodiment_parity_diagnostic", None)
    receipt_valid = episode.get("receipt_digest") == canonical_digest(
        episode, digest_field="receipt_digest"
    ) or episode.get("receipt_digest") == canonical_digest(
        receipt_source, digest_field="receipt_digest"
    )
    if not receipt_valid:
        raise PolicyCanaryRescoreError("policy_canary_rescore_source_episode_digest_mismatch")
    if parity is not None:
        parity_receipt = _mapping(
            parity,
            code="policy_canary_rescore_episode_parity_receipt_invalid",
        )
        if parity_receipt.get("receipt_digest") != canonical_digest(
            parity_receipt, digest_field="receipt_digest"
        ):
            raise PolicyCanaryRescoreError("policy_canary_rescore_episode_parity_receipt_invalid")
    if episode.get("candidate_id") != row.get("candidate_id"):
        raise PolicyCanaryRescoreError("policy_canary_rescore_source_episode_identity_mismatch")
    task_spec = _mapping(episode.get("task_spec"), code="policy_canary_rescore_task_spec_missing")
    task_spec_digest = canonical_digest(task_spec)
    if episode.get("task_spec_digest") != task_spec_digest:
        raise PolicyCanaryRescoreError("policy_canary_rescore_task_spec_digest_mismatch")
    state_trace = _mapping(
        episode.get("state_trace"), code="policy_canary_rescore_state_trace_missing"
    )
    samples = state_trace.get("task_state_samples")
    if not isinstance(samples, list) or not samples:
        raise PolicyCanaryRescoreError("policy_canary_rescore_state_trace_samples_missing")
    if row.get("state_trace_digest") != canonical_digest({"value": state_trace}):
        raise PolicyCanaryRescoreError("policy_canary_rescore_state_trace_digest_mismatch")
    _, state_path = _verified_episode_artifact(
        row=row,
        role="state_trace",
        inventory=inventory,
        evidence_root=evidence_root,
    )
    if (
        _load_object(state_path, code="policy_canary_rescore_state_trace_artifact_invalid")
        != state_trace
    ):
        raise PolicyCanaryRescoreError(
            "policy_canary_rescore_state_trace_artifact_content_mismatch"
        )
    old_score = _mapping(episode.get("score"), code="policy_canary_rescore_old_score_missing")
    if row.get("deterministic_score_digest") != canonical_digest({"value": old_score}):
        raise PolicyCanaryRescoreError("policy_canary_rescore_old_score_digest_mismatch")
    if old_score.get("report_digest") != canonical_digest(old_score, digest_field="report_digest"):
        raise PolicyCanaryRescoreError("policy_canary_rescore_old_score_report_digest_mismatch")
    _, score_path = _verified_episode_artifact(
        row=row,
        role="score_receipt",
        inventory=inventory,
        evidence_root=evidence_root,
    )
    if (
        _load_object(score_path, code="policy_canary_rescore_old_score_artifact_invalid")
        != old_score
    ):
        raise PolicyCanaryRescoreError("policy_canary_rescore_old_score_artifact_content_mismatch")
    return task_spec, old_score, samples


def resolve_scorer_identity(*, expected_commit: str) -> dict[str, Any]:
    """Bind the imported deterministic scorer to clean bytes at one Git commit."""

    if not _COMMIT.fullmatch(expected_commit):
        raise PolicyCanaryRescoreError("policy_canary_rescore_scorer_commit_invalid")
    repo = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo), *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise PolicyCanaryRescoreError(
                "policy_canary_rescore_scorer_git_identity_unavailable"
            ) from exc
        return completed.stdout.strip()

    observed = git("rev-parse", "HEAD")
    if observed != expected_commit:
        raise PolicyCanaryRescoreError("policy_canary_rescore_scorer_commit_mismatch")
    if (
        subprocess.run(
            ["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", *_SCORER_SOURCES]
        ).returncode
        != 0
    ):
        raise PolicyCanaryRescoreError("policy_canary_rescore_scorer_sources_dirty")
    sources = []
    for relative in _SCORER_SOURCES:
        path = repo / relative
        sources.append({"path": relative, "sha256": _sha256(path)})
    identity: dict[str, Any] = {
        "schema_version": SCORER_IDENTITY_SCHEMA_VERSION,
        "scorer": "blueprint_pipeline.adp_task_scoring.score_task_episode_from_spec",
        "scorer_commit": observed,
        "source_files": sources,
        "source_files_digest": canonical_digest({"value": sources}),
        "scoring_version_digest": "",
    }
    identity["scoring_version_digest"] = cross_runtime_canonical_digest(
        identity, digest_field="scoring_version_digest"
    )
    return identity


def _validate_scorer_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    identity = _mapping(value, code="policy_canary_rescore_scorer_identity_invalid")
    source_files = identity.get("source_files")
    if (
        set(identity)
        != {
            "schema_version",
            "scorer",
            "scorer_commit",
            "source_files",
            "source_files_digest",
            "scoring_version_digest",
        }
        or identity.get("schema_version") != SCORER_IDENTITY_SCHEMA_VERSION
        or identity.get("scorer")
        != "blueprint_pipeline.adp_task_scoring.score_task_episode_from_spec"
        or not _COMMIT.fullmatch(str(identity.get("scorer_commit") or ""))
        or not isinstance(source_files, list)
        or not source_files
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256"}
            or not str(row.get("path") or "")
            or not _digest(row.get("sha256"))
            for row in source_files
        )
        or not _digest(identity.get("source_files_digest"))
        or identity.get("source_files_digest")
        != canonical_digest({"value": identity.get("source_files") or []})
        or identity.get("scoring_version_digest")
        != cross_runtime_canonical_digest(identity, digest_field="scoring_version_digest")
    ):
        raise PolicyCanaryRescoreError("policy_canary_rescore_scorer_identity_invalid")
    return identity


def rescore_policy_canary_result(
    *,
    source_result_path: str | Path,
    evidence_root: str | Path,
    output_root: str | Path,
    expected_run_id: str,
    scorer_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify one retained run and write immutable derived correction receipts."""

    unresolved_source = Path(source_result_path).expanduser()
    unresolved_evidence = Path(evidence_root).expanduser()
    unresolved_output = Path(output_root).expanduser()
    if unresolved_source.is_symlink():
        raise PolicyCanaryRescoreError("policy_canary_rescore_source_result_invalid")
    if unresolved_evidence.is_symlink():
        raise PolicyCanaryRescoreError("policy_canary_rescore_evidence_root_invalid")
    if unresolved_output.is_symlink():
        raise PolicyCanaryRescoreError("policy_canary_rescore_output_root_invalid")
    source_path = unresolved_source.resolve()
    evidence = unresolved_evidence.resolve()
    output = unresolved_output.resolve()
    if not evidence.is_dir():
        raise PolicyCanaryRescoreError("policy_canary_rescore_evidence_root_invalid")
    source = _load_object(source_path, code="policy_canary_rescore_source_result_invalid")
    try:
        validate_session_result(source)
    except ValueError as exc:
        raise PolicyCanaryRescoreError("policy_canary_rescore_source_result_invalid") from exc
    if source.get("status") != "completed_unqualified" or source.get("run_id") != expected_run_id:
        raise PolicyCanaryRescoreError("policy_canary_rescore_source_run_identity_invalid")
    source_file_sha256 = _sha256(source_path)
    source_result_digest = str(source["result_digest"])
    inventory = _verify_artifact_inventory(source, evidence_root=evidence)
    identity = _validate_scorer_identity(scorer_identity)
    scoring_version_digest = str(identity["scoring_version_digest"])
    correction_id = cross_runtime_canonical_digest(
        {
            "source_result_digest": source_result_digest,
            "source_result_file_sha256": source_file_sha256,
            "scoring_version_digest": scoring_version_digest,
        }
    )[7:31]
    correction_root = output / correction_id
    episodes = source.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise PolicyCanaryRescoreError("policy_canary_rescore_source_episodes_invalid")
    updates: list[dict[str, Any]] = []
    source_identities: list[dict[str, Any]] = []
    old_scores: list[dict[str, Any]] = []
    new_scores: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(episodes):
        row = _mapping(raw, code="policy_canary_rescore_source_episode_invalid")
        key = _episode_key(row)
        if key in seen:
            raise PolicyCanaryRescoreError("policy_canary_rescore_source_episode_duplicate")
        seen.add(key)
        task_spec, old_score, samples = _verify_source_episode(
            row,
            inventory=inventory,
            evidence_root=evidence,
        )
        new_score = score_task_episode_from_spec(
            task_spec=task_spec,
            samples=samples,
        )
        success_contract = _mapping(
            new_score.get("task_success_contract"),
            code="policy_canary_rescore_new_success_contract_missing",
        )
        success_contract_digest = new_score.get("task_success_contract_digest")
        if (
            not _digest(success_contract_digest)
            or success_contract.get("contract_digest") != success_contract_digest
            or success_contract_digest
            != canonical_digest(success_contract, digest_field="contract_digest")
            or new_score.get("report_digest")
            != canonical_digest(new_score, digest_field="report_digest")
        ):
            raise PolicyCanaryRescoreError("policy_canary_rescore_new_score_digest_invalid")
        source_identity = _source_episode_identity(row)
        source_identity_digest = canonical_digest(source_identity)
        old_score_digest = canonical_digest({"value": old_score})
        new_score_digest = canonical_digest({"value": new_score})
        receipt: dict[str, Any] = {
            "schema_version": EPISODE_RECEIPT_SCHEMA_VERSION,
            "status": "rescored",
            "derived_only": True,
            "source_run_id": expected_run_id,
            "source_result_digest": source_result_digest,
            "source_result_file_sha256": source_file_sha256,
            "candidate_id": row.get("candidate_id"),
            "cell_id": row.get("cell_id"),
            "seed": row.get("seed"),
            "source_episode_identity_digest": source_identity_digest,
            "source_episode_digest": source_identity["source_episode_digest"],
            "source_episode_receipt_digest": source_identity["source_episode_receipt_digest"],
            "source_evidence_artifact_bindings_digest": source_identity[
                "evidence_artifact_bindings_digest"
            ],
            "task_spec_digest": source_identity["task_spec_digest"],
            "success_contract_digest": success_contract_digest,
            "state_trace_digest": source_identity["state_trace_digest"],
            "old_score_digest": old_score_digest,
            "new_score_digest": new_score_digest,
            "old_score": old_score,
            "new_score": new_score,
            "scorer_commit": identity["scorer_commit"],
            "scorer_source_files_digest": identity["source_files_digest"],
            "scoring_version_digest": scoring_version_digest,
            "original_provider_output_overwritten": False,
            "original_score_receipt_overwritten": False,
            "receipt_digest": "",
        }
        receipt["receipt_digest"] = cross_runtime_canonical_digest(
            receipt, digest_field="receipt_digest"
        )
        relative = f"episodes/{index:02d}.json"
        receipt_path = correction_root / relative
        _write_immutable(receipt_path, receipt)
        receipt_record = {
            "relative_path": relative,
            "sha256": _sha256(receipt_path),
            "size_bytes": receipt_path.stat().st_size,
            "receipt_digest": receipt["receipt_digest"],
        }
        updates.append(
            {
                "candidate_id": row.get("candidate_id"),
                "cell_id": row.get("cell_id"),
                "seed": row.get("seed"),
                "source_episode_identity_digest": source_identity_digest,
                "source_evidence_artifact_bindings_digest": source_identity[
                    "evidence_artifact_bindings_digest"
                ],
                "old_score_digest": old_score_digest,
                "new_score_digest": new_score_digest,
                "new_score": new_score,
                "success_contract_digest": success_contract_digest,
                "scoring_version_digest": scoring_version_digest,
                "derived_rescore_receipt": receipt_record,
            }
        )
        source_identities.append(source_identity)
        old_scores.append({"episode_key": key, "score_digest": old_score_digest})
        new_scores.append({"episode_key": key, "score_digest": new_score_digest})
    correction: dict[str, Any] = {
        "schema_version": CORRECTION_SCHEMA_VERSION,
        "status": "completed_unqualified_score_correction_ready",
        "correction_id": correction_id,
        "source_run_id": expected_run_id,
        "source_result_status": "completed_unqualified",
        "corrected_result_status": "completed_unqualified",
        "source_result_digest": source_result_digest,
        "source_result_file_sha256": source_file_sha256,
        "source_artifact_inventory_digest": source["artifact_inventory_digest"],
        "source_episode_identity_set_digest": canonical_digest({"value": source_identities}),
        "source_score_set_digest": canonical_digest({"value": old_scores}),
        "corrected_score_set_digest": canonical_digest({"value": new_scores}),
        "success_contract_set_digest": canonical_digest(
            {"value": [row["success_contract_digest"] for row in updates]}
        ),
        "scorer_identity": identity,
        "scoring_version_digest": scoring_version_digest,
        "episode_count": len(updates),
        "score_updates": updates,
        "publication_constraints": _PUBLICATION_CONSTRAINTS,
        "correction_digest": "",
    }
    correction["correction_digest"] = cross_runtime_canonical_digest(
        correction, digest_field="correction_digest"
    )
    _write_immutable(correction_root / "score_correction.json", correction)
    return validate_policy_canary_score_correction(
        source_result=source,
        correction=correction,
        receipt_root=correction_root,
    )


def validate_policy_canary_score_correction(
    *,
    source_result: Mapping[str, Any],
    correction: Mapping[str, Any],
    receipt_root: str | Path,
) -> dict[str, Any]:
    """Validate the narrow overlay a Website publisher is allowed to apply."""

    source = _mapping(source_result, code="policy_canary_score_correction_source_invalid")
    value = _mapping(correction, code="policy_canary_score_correction_invalid")
    root = Path(receipt_root).expanduser().resolve()
    expected_fields = {
        "schema_version",
        "status",
        "correction_id",
        "source_run_id",
        "source_result_status",
        "corrected_result_status",
        "source_result_digest",
        "source_result_file_sha256",
        "source_artifact_inventory_digest",
        "source_episode_identity_set_digest",
        "source_score_set_digest",
        "corrected_score_set_digest",
        "success_contract_set_digest",
        "scorer_identity",
        "scoring_version_digest",
        "episode_count",
        "score_updates",
        "publication_constraints",
        "correction_digest",
    }
    try:
        validate_session_result(source)
    except ValueError as exc:
        raise PolicyCanaryRescoreError("policy_canary_score_correction_source_invalid") from exc
    scorer_identity = _validate_scorer_identity(
        _mapping(
            value.get("scorer_identity"),
            code="policy_canary_score_correction_scorer_identity_invalid",
        )
    )
    if (
        set(value) != expected_fields
        or value.get("schema_version") != CORRECTION_SCHEMA_VERSION
        or value.get("status") != "completed_unqualified_score_correction_ready"
        or value.get("source_result_status") != "completed_unqualified"
        or value.get("corrected_result_status") != "completed_unqualified"
        or source.get("status") != "completed_unqualified"
        or value.get("source_run_id") != source.get("run_id")
        or value.get("source_result_digest") != source.get("result_digest")
        or not _digest(value.get("source_result_file_sha256"))
        or value.get("source_artifact_inventory_digest") != source.get("artifact_inventory_digest")
        or value.get("scoring_version_digest") != scorer_identity.get("scoring_version_digest")
        or value.get("publication_constraints") != _PUBLICATION_CONSTRAINTS
        or value.get("correction_digest")
        != cross_runtime_canonical_digest(value, digest_field="correction_digest")
    ):
        raise PolicyCanaryRescoreError("policy_canary_score_correction_invalid")
    expected_correction_id = cross_runtime_canonical_digest(
        {
            "source_result_digest": value["source_result_digest"],
            "source_result_file_sha256": value["source_result_file_sha256"],
            "scoring_version_digest": value["scoring_version_digest"],
        }
    )[7:31]
    if value.get("correction_id") != expected_correction_id or root.name != expected_correction_id:
        raise PolicyCanaryRescoreError("policy_canary_score_correction_identity_invalid")
    episodes = source.get("episodes")
    updates = value.get("score_updates")
    if (
        not isinstance(episodes, list)
        or not isinstance(updates, list)
        or len(episodes) != len(updates)
        or value.get("episode_count") != len(episodes)
    ):
        raise PolicyCanaryRescoreError("policy_canary_score_correction_episode_count_mismatch")
    source_identities = [
        _source_episode_identity(
            _mapping(row, code="policy_canary_score_correction_source_episode_invalid")
        )
        for row in episodes
    ]
    if value.get("source_episode_identity_set_digest") != canonical_digest(
        {"value": source_identities}
    ):
        raise PolicyCanaryRescoreError("policy_canary_score_correction_episode_identity_changed")
    source_by_key = {_episode_key(row): row for row in episodes}
    seen: set[str] = set()
    observed_old_scores: list[dict[str, Any]] = []
    observed_new_scores: list[dict[str, Any]] = []
    for update in updates:
        row = _mapping(update, code="policy_canary_score_correction_update_invalid")
        if set(row) != {
            "candidate_id",
            "cell_id",
            "seed",
            "source_episode_identity_digest",
            "source_evidence_artifact_bindings_digest",
            "old_score_digest",
            "new_score_digest",
            "new_score",
            "success_contract_digest",
            "scoring_version_digest",
            "derived_rescore_receipt",
        }:
            raise PolicyCanaryRescoreError("policy_canary_score_correction_update_fields_invalid")
        key = _episode_key(row)
        source_row = source_by_key.get(key)
        if source_row is None or key in seen:
            raise PolicyCanaryRescoreError(
                "policy_canary_score_correction_episode_identity_changed"
            )
        seen.add(key)
        source_identity = _source_episode_identity(source_row)
        old_score = _mapping(
            _mapping(
                source_row.get("episode"),
                code="policy_canary_score_correction_source_episode_invalid",
            ).get("score"),
            code="policy_canary_score_correction_source_score_invalid",
        )
        new_score = _mapping(
            row.get("new_score"), code="policy_canary_score_correction_new_score_invalid"
        )
        success_contract = _mapping(
            new_score.get("task_success_contract"),
            code="policy_canary_score_correction_new_success_contract_invalid",
        )
        if (
            row.get("source_episode_identity_digest") != canonical_digest(source_identity)
            or row.get("source_evidence_artifact_bindings_digest")
            != source_identity["evidence_artifact_bindings_digest"]
            or row.get("old_score_digest") != canonical_digest({"value": old_score})
            or row.get("new_score_digest") != canonical_digest({"value": new_score})
            or new_score.get("report_digest")
            != canonical_digest(new_score, digest_field="report_digest")
            or row.get("success_contract_digest") != new_score.get("task_success_contract_digest")
            or row.get("success_contract_digest") != success_contract.get("contract_digest")
            or row.get("success_contract_digest")
            != canonical_digest(success_contract, digest_field="contract_digest")
            or row.get("scoring_version_digest") != value.get("scoring_version_digest")
        ):
            raise PolicyCanaryRescoreError("policy_canary_score_correction_update_binding_invalid")
        observed_old_scores.append({"episode_key": key, "score_digest": row["old_score_digest"]})
        observed_new_scores.append({"episode_key": key, "score_digest": row["new_score_digest"]})
        record = _mapping(
            row.get("derived_rescore_receipt"),
            code="policy_canary_score_correction_receipt_invalid",
        )
        path = _inside(
            root,
            record.get("relative_path"),
            code="policy_canary_score_correction_receipt_invalid",
        )
        if _sha256(path) != record.get("sha256") or path.stat().st_size != record.get("size_bytes"):
            raise PolicyCanaryRescoreError("policy_canary_score_correction_receipt_digest_mismatch")
        receipt = _load_object(path, code="policy_canary_score_correction_receipt_invalid")
        if (
            set(receipt)
            != {
                "schema_version",
                "status",
                "derived_only",
                "source_run_id",
                "source_result_digest",
                "source_result_file_sha256",
                "candidate_id",
                "cell_id",
                "seed",
                "source_episode_identity_digest",
                "source_episode_digest",
                "source_episode_receipt_digest",
                "source_evidence_artifact_bindings_digest",
                "task_spec_digest",
                "success_contract_digest",
                "state_trace_digest",
                "old_score_digest",
                "new_score_digest",
                "old_score",
                "new_score",
                "scorer_commit",
                "scorer_source_files_digest",
                "scoring_version_digest",
                "original_provider_output_overwritten",
                "original_score_receipt_overwritten",
                "receipt_digest",
            }
            or receipt.get("schema_version") != EPISODE_RECEIPT_SCHEMA_VERSION
            or receipt.get("status") != "rescored"
            or receipt.get("derived_only") is not True
            or receipt.get("source_run_id") != source.get("run_id")
            or receipt.get("source_result_digest") != source.get("result_digest")
            or receipt.get("source_result_file_sha256") != value.get("source_result_file_sha256")
            or receipt.get("candidate_id") != row.get("candidate_id")
            or receipt.get("cell_id") != row.get("cell_id")
            or receipt.get("seed") != row.get("seed")
            or receipt.get("receipt_digest") != record.get("receipt_digest")
            or receipt.get("receipt_digest")
            != cross_runtime_canonical_digest(receipt, digest_field="receipt_digest")
            or receipt.get("source_episode_identity_digest")
            != row.get("source_episode_identity_digest")
            or receipt.get("source_episode_digest") != source_identity["source_episode_digest"]
            or receipt.get("source_episode_receipt_digest")
            != source_identity["source_episode_receipt_digest"]
            or receipt.get("source_evidence_artifact_bindings_digest")
            != row.get("source_evidence_artifact_bindings_digest")
            or receipt.get("task_spec_digest") != source_identity["task_spec_digest"]
            or receipt.get("success_contract_digest") != row.get("success_contract_digest")
            or receipt.get("state_trace_digest") != source_identity["state_trace_digest"]
            or receipt.get("old_score_digest") != row.get("old_score_digest")
            or receipt.get("old_score") != old_score
            or receipt.get("new_score_digest") != row.get("new_score_digest")
            or receipt.get("new_score") != new_score
            or receipt.get("scorer_commit") != scorer_identity["scorer_commit"]
            or receipt.get("scorer_source_files_digest") != scorer_identity["source_files_digest"]
            or receipt.get("scoring_version_digest") != row.get("scoring_version_digest")
            or receipt.get("original_provider_output_overwritten") is not False
            or receipt.get("original_score_receipt_overwritten") is not False
        ):
            raise PolicyCanaryRescoreError("policy_canary_score_correction_receipt_binding_invalid")
    if (
        value.get("source_score_set_digest") != canonical_digest({"value": observed_old_scores})
        or value.get("corrected_score_set_digest")
        != canonical_digest({"value": observed_new_scores})
        or value.get("success_contract_set_digest")
        != canonical_digest({"value": [row["success_contract_digest"] for row in updates]})
    ):
        raise PolicyCanaryRescoreError("policy_canary_score_correction_aggregate_binding_invalid")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scorer-commit", required=True)
    args = parser.parse_args(argv)
    try:
        result = rescore_policy_canary_result(
            source_result_path=args.source_result,
            evidence_root=args.evidence_root,
            output_root=args.output_root,
            expected_run_id=args.run_id,
            scorer_identity=resolve_scorer_identity(expected_commit=args.scorer_commit),
        )
    except (OSError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "blocked", "blockers": [str(exc)]},
                sort_keys=True,
            )
        )
        return 2
    print(canonical_json(result))
    return 0


__all__ = [
    "CORRECTION_SCHEMA_VERSION",
    "EPISODE_RECEIPT_SCHEMA_VERSION",
    "PolicyCanaryRescoreError",
    "rescore_policy_canary_result",
    "resolve_scorer_identity",
    "validate_policy_canary_score_correction",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
