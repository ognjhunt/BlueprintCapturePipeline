"""Reopen G1 worker, query and scorer bytes before private beta projection."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import zipfile

from .decision_evidence_contracts import canonical_digest
from .native_g1_beta_bundle import _sha, validate_bundle_archive
from .native_g1_beta_packet import REQUEST, CONTRACT, PLAN, RECEIPT, derive_documents, expected_cell_packet_receipt
from .native_g1_beta_session import MANIFEST, RESULT_FILENAME, RESULT_SCHEMA, validate_session_authority


def verify_output(*, output_dir: Path, bundle: dict):
    from .beta_policy_screening_g1 import _episode_outcome
    root = Path(output_dir)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("g1_beta_output_root_invalid")
    with zipfile.ZipFile(bundle["bundle_path"]) as archive:
        manifest = validate_bundle_archive(archive, json.loads(archive.read(MANIFEST)))
        if manifest["authority_digest"] != bundle["authority_digest"]:
            raise ValueError("g1_beta_output_bundle_mismatch")
        authority = validate_session_authority(json.loads(archive.read("provider_runtime/inputs/session_authority.json")))
        base_receipt = json.loads(archive.read("provider_runtime/inputs/scene_packet/" + RECEIPT))
        base_navigation = (json.loads(archive.read("provider_runtime/inputs/navigation_authority.json"))
                           if "provider_runtime/inputs/navigation_authority.json" in archive.namelist() else None)
        docs = {key: json.loads(archive.read("provider_runtime/inputs/scene_packet/" + name))
                for key, name in (("base_request", REQUEST), ("base_contract", CONTRACT), ("base_scene", PLAN))}
    path = root / RESULT_FILENAME
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128 * 1024**2:
        raise ValueError("g1_beta_output_result_missing_or_oversize")
    result = json.loads(path.read_text())
    admission = authority["beta_screening_admission"]
    expected = {"schema_version": RESULT_SCHEMA, "candidate_id": authority["candidate_ids"][0],
        **{key: admission[key] for key in ("plan_digest", "configuration_digest", "admission_digest", "runtime_identity_digest",
                                         "source", "scenario_plan_digest")},
        "task_success_contract_digest": authority["task_success_contract_digest"],
        "authority_digest": authority["authority_digest"], "runtime_inputs_digest": authority["runtime_inputs_digest"], "claim_ceiling": "development_only"}
    if (any(result.get(key) != value for key, value in expected.items())
            or result.get("result_digest") != canonical_digest(result, digest_field="result_digest")
            or "session_closeout" in result or "source_result_digest" in result
            or not isinstance(result.get("episodes"), list)
            or len(result["episodes"]) != len(authority["cells"])
            or any(result.get(key) is not False for key in ("ranking_eligible", "qualification_permitted",
                "public_media_publication_authorized", "model_redistribution_authorized", "physical_outcome_claimed"))):
        raise ValueError("g1_beta_output_identity_invalid")
    source_result_digest = result["result_digest"]
    witnessed = 0
    for row, cell in zip(result["episodes"], authority["cells"], strict=True):
        if any(row.get(key) != cell[key] for key in ("cell_id", "seed", "cell_spec_digest", "resolved_scenario", "resolved_scenario_digest")):
            raise ValueError("g1_beta_output_cell_changed")
        derived = derive_documents(**docs, cell=cell)
        if row.get("scene_plan_digest") != derived["scene"]["plan_digest"] and not (
                row.get("status") == "not_attempted" and row.get("scene_plan_digest") is None):
            raise ValueError("g1_beta_output_scene_changed")
        if (row.get("evidence_delivery") != "retained_artifact_references.v1"
                or any(key in row for key in ("worker_result", "worker_preclose", "episode", "trace", "query_checkpoints"))
                or type(row.get("query_checkpoint_count")) is not int or not 0 <= row["query_checkpoint_count"] <= 3000):
            raise ValueError("g1_beta_output_delivery_invalid")
        records = row.get("artifact_records")
        if not isinstance(records, list) or len(records) != len({r.get("relative_path") for r in records}):
            raise ValueError("g1_beta_output_artifact_records_invalid")
        observed = {}
        prefix = PurePosixPath("cells") / cell["cell_id"]
        for record in records:
            relative = PurePosixPath(str(record.get("relative_path") or ""))
            artifact = root / relative
            if (relative.is_absolute() or ".." in relative.parts or not relative.is_relative_to(prefix)
                    or artifact.is_symlink() or not artifact.is_file() or artifact.resolve() != artifact
                    or type(record.get("size_bytes")) is not int or not 0 < record["size_bytes"] <= 128 * 1024**2
                    or artifact.stat().st_size != record["size_bytes"] or _sha(artifact) != record.get("sha256")):
                raise ValueError("g1_beta_output_artifact_bytes_changed")
            observed[relative.as_posix()] = json.loads(artifact.read_text())
        for key, filename in (("worker_result", "native_g1_development_worker_result.v1.json"),
                ("worker_preclose", "native_g1_development_worker_preclose.v1.json"),
                ("episode", "episode/native_g1_built_scene_policy_episode.v1.json"),
                ("trace", "episode/native_g1_shared_scene_episode_trace.v1.json")):
            name = (prefix / filename).as_posix()
            if name in observed:
                row[key] = observed[name]
        expected_paths = [(prefix / "episode/query_checkpoints" / f"query_{i:04d}.v1.json").as_posix()
                          for i in range(row["query_checkpoint_count"])]
        checkpoints = [observed.get(name) for name in expected_paths]
        if any(not isinstance(value, dict) for value in checkpoints):
            raise ValueError("g1_beta_output_checkpoints_missing")
        row["query_checkpoints"] = checkpoints
        checkpoint_root = root / prefix / "episode/query_checkpoints"
        actual_paths = sorted(p.relative_to(root).as_posix() for p in checkpoint_root.iterdir()) if checkpoint_root.exists() else []
        if actual_paths != expected_paths or any(observed.get(name) != value for name, value in zip(expected_paths, checkpoints, strict=True)):
            raise ValueError("g1_beta_output_query_file_closure_changed")
        from .native_g1_beta_authority import derive_cell_authority
        derived_receipt = expected_cell_packet_receipt(base_receipt, derived, cell)
        cell_authority = derive_cell_authority(base_authority=authority["rights_authority"], cell=cell,
            scene_plan_digest=derived["scene"]["plan_digest"], packet_receipt_digest=derived_receipt["receipt_digest"])
        worker = row.get("worker_result")
        if worker is not None and not isinstance(worker, dict):
            raise ValueError("g1_beta_output_worker_invalid")
        if worker and worker.get("rights_review_digest") not in {None, cell_authority["authority_digest"]}:
            raise ValueError("g1_beta_output_worker_authority_changed")
        if row["status"] == "completed":
            if not all(isinstance(row.get(key), dict) for key in ("worker_result", "episode", "trace")):
                raise ValueError("g1_beta_output_completed_evidence_missing")
            if worker.get("rights_review_digest") != cell_authority["authority_digest"]:
                raise ValueError("g1_beta_output_worker_authority_missing")
            samples = [row["trace"]["initial_task_sample"], *[step["task_sample"] for step in row["trace"]["steps"]]]
            if expected["candidate_id"].endswith("_vision_navi"):
                from .native_g1_beta_navigation import derive_navigation_authority
                from .native_g1_navigation_goal import score_g1_navigation_episode
                navigation = derive_navigation_authority(base_navigation, docs["base_scene"], derived["scene"],
                    cell_authority, base_contract=docs["base_contract"])
                if worker.get("navigation_goal_authority_digest") != navigation["authority_digest"]:
                    raise ValueError("g1_beta_output_navigation_authority_changed")
                score = score_g1_navigation_episode(task_spec=derived["scene"]["task_spec"], samples=samples)
            else:
                from .adp_task_scoring import score_task_episode_from_spec
                score = score_task_episode_from_spec(task_spec=derived["scene"]["task_spec"], samples=samples)
            if score != row["episode"]["score"]:
                raise ValueError("g1_beta_output_independent_score_changed")
        _outcome, _reason, queried = _episode_outcome(row, candidate_id=expected["candidate_id"],
            contract_digest=authority["task_success_contract_digest"])
        witnessed += int(queried)
    if result.get("candidate_policy_queried") is not bool(witnessed):
        raise ValueError("g1_beta_output_query_claim_changed")
    result["source_result_digest"] = source_result_digest
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result
