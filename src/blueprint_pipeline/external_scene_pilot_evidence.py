"""Finalize a multi-lane external-scene-to-Isaac pilot from hash-bound evidence.

This module deliberately does not execute Isaac or infer scientific success.  It
verifies the evidence files produced by those systems, applies the pilot's
completion rules, and emits the small reports needed to review or replay a run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import write_json
from .decision_evidence_contracts import canonical_digest


REQUEST_SCHEMA_VERSION = "external_scene_pilot_finalization_request.v1"
MASTER_MANIFEST_SCHEMA_VERSION = "external_scene_pilot_master_manifest.v1"
COMPARISON_SCHEMA_VERSION = "external_scene_pilot_comparison.v1"
CLAIM_LEDGER_SCHEMA_VERSION = "external_scene_pilot_claim_ledger.v1"
TERMINAL_SUMMARY_SCHEMA_VERSION = "external_scene_pilot_terminal_summary.v1"
REPRODUCTION_SCHEMA_VERSION = "external_scene_pilot_reproduction.v1"
GIT_STATE_SCHEMA_VERSION = "external_scene_pilot_git_state.v1"

_SHA256_PREFIX = "sha256:"
_STATUSES = {"passed", "failed", "abstained", "not_run", "not_applicable"}
_CLAIM_STATUSES = {"supported", "rejected", "abstained"}

# These are the shortest credible downstream-harness gates.  Descriptive rows
# may be added by callers, but a lane cannot disappear a mandatory gate.
MANDATORY_GATE_IDS = (
    "source_frozen",
    "strict_import",
    "cpu_qualification",
    "task_target_binding",
    "robot_placement",
    "isaac_stage_load",
    "nonblank_render",
    "collision_contact",
    "scenario_execution",
    "robot_spawn",
    "robot_camera",
    "two_candidate_traces",
    "distinct_trace_pair",
    "task_evaluation_run",
    "digest_linkage",
    "provider_zero",
)

# A CPU qualification or a Task Evaluation Run may correctly abstain while the
# lower-level harness remains complete.  Missing live execution evidence may not.
_ABSTENTION_COMPLETES_GATE = {"cpu_qualification", "task_evaluation_run"}

_COMPARISON_FIELDS = (
    "asset_digest",
    "source_class",
    "capture_modality",
    "archive_structure",
    "nurec_representation",
    "mesh_collider_presence",
    "metric_scale",
    "up_axis",
    "bounds",
    "unresolved_references",
    "runtime_cost_usd",
    "manual_operator_minutes",
    "failure_points",
)


class ExternalScenePilotEvidenceError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith(_SHA256_PREFIX)
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return _SHA256_PREFIX + digest.hexdigest()


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ExternalScenePilotEvidenceError(["pilot_request_not_json"]) from exc


def _resolve_artifact(
    *, artifact_root: Path, reference: Mapping[str, Any], lane_id: str, gate_id: str
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    raw_path = reference.get("path")
    expected = reference.get("sha256")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None, [f"artifact_path_missing:{lane_id}:{gate_id}"]
    if not _is_digest(expected):
        errors.append(f"artifact_digest_invalid:{lane_id}:{gate_id}:{raw_path}")
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = artifact_root / candidate
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(artifact_root)
    except (FileNotFoundError, RuntimeError, ValueError):
        errors.append(f"artifact_path_unsafe_or_missing:{lane_id}:{gate_id}:{raw_path}")
        return None, errors
    if candidate.is_symlink() or not resolved.is_file():
        errors.append(f"artifact_not_regular_file:{lane_id}:{gate_id}:{raw_path}")
        return None, errors
    observed = _file_digest(resolved)
    if observed != expected:
        errors.append(f"artifact_digest_mismatch:{lane_id}:{gate_id}:{raw_path}")
    return (
        {
            "path": resolved.relative_to(artifact_root).as_posix(),
            "sha256": observed,
            "size_bytes": resolved.stat().st_size,
        },
        errors,
    )


def _gate_complete(gate_id: str, status: str, *, remote_runtime_used: bool) -> bool:
    if status == "passed":
        return True
    if status == "abstained" and gate_id in _ABSTENTION_COMPLETES_GATE:
        return True
    if gate_id == "provider_zero" and not remote_runtime_used and status == "not_applicable":
        return True
    return False


def _git(*args: str, repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _build_git_state(repo_root: Path, expected_commit: str) -> dict[str, Any]:
    try:
        commit = _git("rev-parse", "HEAD", repo_root=repo_root)
        branch = _git("branch", "--show-current", repo_root=repo_root) or None
        porcelain = _git("status", "--porcelain", repo_root=repo_root)
        upstream = _git("rev-parse", "--abbrev-ref", "@{upstream}", repo_root=repo_root)
        ahead_behind = _git(
            "rev-list", "--left-right", "--count", "HEAD...@{upstream}", repo_root=repo_root
        ).split()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExternalScenePilotEvidenceError(["pilot_git_state_unavailable"]) from exc
    payload: dict[str, Any] = {
        "schema_version": GIT_STATE_SCHEMA_VERSION,
        "repo_root": str(repo_root),
        "head_commit": commit,
        "expected_source_commit": expected_commit,
        "expected_commit_matches_head": commit == expected_commit,
        "branch": branch,
        "clean": not porcelain,
        "status_porcelain": porcelain.splitlines(),
        "upstream": upstream,
        "ahead": int(ahead_behind[0]),
        "behind": int(ahead_behind[1]),
    }
    payload["git_state_digest"] = canonical_digest(payload)
    return payload


def compile_external_scene_pilot_evidence(
    request: Mapping[str, Any], *, artifact_root: str | Path, repo_root: str | Path
) -> dict[str, dict[str, Any]]:
    """Verify a finalization request and compile all review-facing artifacts."""

    value = _clone(request)
    errors: list[str] = []
    if value.get("schema_version") != REQUEST_SCHEMA_VERSION:
        errors.append("pilot_request_schema_invalid")
    run_id = value.get("run_id")
    source_commit = value.get("source_commit_sha")
    if not isinstance(run_id, str) or not run_id.strip():
        errors.append("pilot_run_id_missing")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        errors.append("pilot_source_commit_invalid")
    lanes = value.get("lanes")
    if not isinstance(lanes, list) or not lanes:
        errors.append("pilot_lanes_missing")
        lanes = []
    artifact_root_path = Path(artifact_root).expanduser().resolve()
    repo_root_path = Path(repo_root).expanduser().resolve()
    if not artifact_root_path.is_dir():
        errors.append("pilot_artifact_root_invalid")
    if not (repo_root_path / ".git").exists() and not _is_git_worktree(repo_root_path):
        errors.append("pilot_repo_root_invalid")

    normalized_lanes: list[dict[str, Any]] = []
    seen_lane_ids: set[str] = set()
    verified_artifacts: list[dict[str, Any]] = []
    for index, lane in enumerate(lanes):
        if not isinstance(lane, Mapping):
            errors.append(f"pilot_lane_invalid:{index}")
            continue
        lane_id = str(lane.get("lane_id") or "")
        if not lane_id or lane_id in seen_lane_ids:
            errors.append(f"pilot_lane_id_invalid_or_duplicate:{index}")
            continue
        seen_lane_ids.add(lane_id)
        profile = lane.get("comparison_profile")
        if not isinstance(profile, Mapping):
            errors.append(f"pilot_comparison_profile_missing:{lane_id}")
            profile = {}
        for field in _COMPARISON_FIELDS:
            if field not in profile:
                errors.append(f"pilot_comparison_field_missing:{lane_id}:{field}")
        if not _is_digest(profile.get("asset_digest")):
            errors.append(f"pilot_asset_digest_invalid:{lane_id}")
        remote_runtime_used = lane.get("remote_runtime_used") is True
        gates_value = lane.get("gates")
        if not isinstance(gates_value, list):
            errors.append(f"pilot_gates_missing:{lane_id}")
            gates_value = []
        gate_rows: dict[str, dict[str, Any]] = {}
        for gate in gates_value:
            if not isinstance(gate, Mapping):
                errors.append(f"pilot_gate_invalid:{lane_id}")
                continue
            gate_id = str(gate.get("gate_id") or "")
            status = str(gate.get("status") or "")
            if not gate_id or gate_id in gate_rows:
                errors.append(f"pilot_gate_id_invalid_or_duplicate:{lane_id}:{gate_id}")
                continue
            if status not in _STATUSES:
                errors.append(f"pilot_gate_status_invalid:{lane_id}:{gate_id}")
            blocker_codes = gate.get("blocker_codes")
            if not isinstance(blocker_codes, list) or any(
                not isinstance(code, str) or not code for code in blocker_codes
            ):
                errors.append(f"pilot_gate_blockers_invalid:{lane_id}:{gate_id}")
                blocker_codes = []
            refs = gate.get("artifacts")
            if not isinstance(refs, list):
                errors.append(f"pilot_gate_artifacts_invalid:{lane_id}:{gate_id}")
                refs = []
            if status in {"passed", "failed", "abstained"} and not refs:
                errors.append(f"pilot_gate_evidence_missing:{lane_id}:{gate_id}")
            observed_refs: list[dict[str, Any]] = []
            for ref in refs:
                if not isinstance(ref, Mapping):
                    errors.append(f"pilot_artifact_reference_invalid:{lane_id}:{gate_id}")
                    continue
                observed, ref_errors = _resolve_artifact(
                    artifact_root=artifact_root_path,
                    reference=ref,
                    lane_id=lane_id,
                    gate_id=gate_id,
                )
                errors.extend(ref_errors)
                if observed is not None:
                    observed_refs.append(observed)
                    verified_artifacts.append({"lane_id": lane_id, "gate_id": gate_id, **observed})
            gate_rows[gate_id] = {
                "gate_id": gate_id,
                "status": status,
                "completion_accepted": _gate_complete(
                    gate_id, status, remote_runtime_used=remote_runtime_used
                ),
                "blocker_codes": sorted(set(blocker_codes)),
                "artifacts": observed_refs,
            }
        for gate_id in MANDATORY_GATE_IDS:
            if gate_id not in gate_rows:
                errors.append(f"pilot_mandatory_gate_missing:{lane_id}:{gate_id}")
        missing_completion = sorted(
            gate_id
            for gate_id in MANDATORY_GATE_IDS
            if gate_id not in gate_rows or not gate_rows[gate_id]["completion_accepted"]
        )
        normalized_lanes.append(
            {
                "lane_id": lane_id,
                "comparison_profile": dict(profile),
                "remote_runtime_used": remote_runtime_used,
                "gates": [gate_rows[key] for key in sorted(gate_rows)],
                "completion_status": "complete" if not missing_completion else "incomplete",
                "incomplete_gate_ids": missing_completion,
            }
        )

    claims_value = value.get("claims")
    if not isinstance(claims_value, list):
        errors.append("pilot_claims_missing")
        claims_value = []
    claims: list[dict[str, Any]] = []
    lane_by_id = {lane["lane_id"]: lane for lane in normalized_lanes}
    gate_by_lane = {
        lane_id: {row["gate_id"]: row for row in lane["gates"]}
        for lane_id, lane in lane_by_id.items()
    }
    for index, claim in enumerate(claims_value):
        if not isinstance(claim, Mapping):
            errors.append(f"pilot_claim_invalid:{index}")
            continue
        claim_id = str(claim.get("claim_id") or "")
        lane_id = str(claim.get("lane_id") or "")
        status = str(claim.get("status") or "")
        required = claim.get("required_gate_ids")
        blockers = claim.get("blocker_codes")
        if not claim_id:
            errors.append(f"pilot_claim_id_missing:{index}")
        if lane_id not in gate_by_lane:
            errors.append(f"pilot_claim_lane_unknown:{claim_id}:{lane_id}")
        if status not in _CLAIM_STATUSES:
            errors.append(f"pilot_claim_status_invalid:{claim_id}")
        if not isinstance(required, list) or not required:
            errors.append(f"pilot_claim_required_gates_missing:{claim_id}")
            required = []
        if not isinstance(blockers, list):
            errors.append(f"pilot_claim_blockers_invalid:{claim_id}")
            blockers = []
        unsatisfied = sorted(
            gate_id
            for gate_id in required
            if gate_id not in gate_by_lane.get(lane_id, {})
            or gate_by_lane[lane_id][gate_id]["status"] != "passed"
        )
        if status == "supported" and unsatisfied:
            errors.append(f"pilot_claim_support_unqualified:{claim_id}")
        claims.append(
            {
                "claim_id": claim_id,
                "lane_id": lane_id,
                "statement": str(claim.get("statement") or ""),
                "status": status,
                "required_gate_ids": list(required),
                "unsatisfied_gate_ids": unsatisfied,
                "blocker_codes": sorted(set(str(code) for code in blockers if str(code))),
            }
        )

    if errors:
        raise ExternalScenePilotEvidenceError(errors)

    git_state = _build_git_state(repo_root_path, str(source_commit))
    if not git_state["expected_commit_matches_head"]:
        raise ExternalScenePilotEvidenceError(["pilot_source_commit_head_mismatch"])

    comparison = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "run_id": run_id,
        "columns": list(_COMPARISON_FIELDS) + list(MANDATORY_GATE_IDS),
        "lanes": [
            {
                "lane_id": lane["lane_id"],
                **lane["comparison_profile"],
                **{row["gate_id"]: row["status"] for row in lane["gates"]},
                "completion_status": lane["completion_status"],
            }
            for lane in normalized_lanes
        ],
    }
    comparison["comparison_digest"] = canonical_digest(comparison)
    claim_ledger = {
        "schema_version": CLAIM_LEDGER_SCHEMA_VERSION,
        "run_id": run_id,
        "claims": claims,
        "unsupported_global_claims": [
            "physical_robot_success",
            "zero_shot_sim_to_real_transfer",
            "customer_site_readiness",
            "iphone_360_capture_parity",
            "blueprint_raw_capture_authority",
        ],
    }
    claim_ledger["claim_ledger_digest"] = canonical_digest(claim_ledger)
    lane_status = {lane["lane_id"]: lane["completion_status"] for lane in normalized_lanes}
    terminal = {
        "schema_version": TERMINAL_SUMMARY_SCHEMA_VERSION,
        "run_id": run_id,
        "status": (
            "dual_asset_harness_complete"
            if len(normalized_lanes) >= 2
            and all(value == "complete" for value in lane_status.values())
            else "partial_harness_evidence"
        ),
        "lane_status": lane_status,
        "remaining_gate_ids": {
            lane["lane_id"]: lane["incomplete_gate_ids"] for lane in normalized_lanes
        },
        "scientific_thesis_state": "inconclusive",
        "physical_success_proven": False,
        "sim_to_real_transfer_proven": False,
    }
    terminal["terminal_summary_digest"] = canonical_digest(terminal)
    reproduction = {
        "schema_version": REPRODUCTION_SCHEMA_VERSION,
        "run_id": run_id,
        "commands": list(value.get("reproduction_commands") or []),
        "source_commit_sha": source_commit,
    }
    if not reproduction["commands"] or any(
        not isinstance(command, str) or not command.strip() for command in reproduction["commands"]
    ):
        raise ExternalScenePilotEvidenceError(["pilot_reproduction_commands_invalid"])
    reproduction["reproduction_digest"] = canonical_digest(reproduction)
    master = {
        "schema_version": MASTER_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "source_commit_sha": source_commit,
        "request_digest": canonical_digest(value),
        "lanes": normalized_lanes,
        "verified_artifacts": sorted(
            verified_artifacts,
            key=lambda item: (item["lane_id"], item["gate_id"], item["path"]),
        ),
        "report_digests": {
            "comparison": comparison["comparison_digest"],
            "claim_ledger": claim_ledger["claim_ledger_digest"],
            "terminal_summary": terminal["terminal_summary_digest"],
            "reproduction": reproduction["reproduction_digest"],
            "git_state": git_state["git_state_digest"],
        },
    }
    master["master_manifest_digest"] = canonical_digest(master)
    return {
        "master_manifest": master,
        "comparison": comparison,
        "claim_ledger": claim_ledger,
        "terminal_summary": terminal,
        "reproduction": reproduction,
        "git_state": git_state,
    }


def _is_git_worktree(path: Path) -> bool:
    try:
        return _git("rev-parse", "--is-inside-work-tree", repo_root=path) == "true"
    except (OSError, subprocess.CalledProcessError):
        return False


def write_external_scene_pilot_evidence(
    reports: Mapping[str, Mapping[str, Any]], output_root: str | Path
) -> dict[str, str]:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    filenames = {
        "master_manifest": "master_run_manifest.v1.json",
        "comparison": "provider_neutral_comparison.v1.json",
        "claim_ledger": "claim_ledger.v1.json",
        "terminal_summary": "terminal_summary.v1.json",
        "reproduction": "reproduction_commands.v1.json",
        "git_state": "git_state_report.v1.json",
    }
    written: dict[str, str] = {}
    for key, filename in filenames.items():
        write_json(root / filename, reports[key])
        written[key] = str(root / filename)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    reports = compile_external_scene_pilot_evidence(
        request, artifact_root=args.artifact_root, repo_root=args.repo_root
    )
    written = write_external_scene_pilot_evidence(reports, args.output_root)
    print(json.dumps({"status": "completed", "outputs": written}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
