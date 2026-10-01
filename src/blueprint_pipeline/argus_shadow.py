"""Offline ADP-009D shadow comparison; never a production evaluator or launcher.

Validate a frozen retained corpus, prepare blinded request specifications, replay
the CURRENT deterministic grader, and compare imported, byte-bound responses.
There is deliberately no HTTP client, credential lookup, decoder, or model call.
Synthetic tests require an explicit flag and cannot produce real-corpus results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "argus_shadow_manifest.v1"
ARGUS_COMMIT = "6c99686a3d93027c517f02f37f24b5b68532e12a"
TARGET_MODEL = "openai/gpt-6.1-sol"
ARMS = ("current", "argus_adapted", "argus_vanilla")
OUTCOMES = {"success", "failure", "unclear"}
ARTIFACTS = {"task_spec", "task_success_contract", "state_trace", "baseline_output",
             "independent_label", "review_video", "frame_manifest"}
EVIDENCE_ROLES = {"task_success_contract", "state_trace", "contact_force_trace",
                  "review_video", "frame_manifest"}
SHA = re.compile(r"^sha256:[0-9a-f]{64}$")
CURRENT_SOURCES = {f"src/blueprint_pipeline/{name}.py" for name in (
    "adp_task_scoring", "adp009d_task_scoring", "adp_rigid_task_scoring",
    "adp_rigid_retreat_scoring", "adp_articulated_task_success_contract",
    "task_evaluation_surface_target", "decision_evidence_contracts", "articulation_graph_contract")}


class ShadowError(ValueError):
    """Malformed, unbound, circular, or inadmissible comparison evidence."""


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def seal(value: dict, key: str) -> dict:
    return {**value, key: digest(canonical({k: v for k, v in value.items() if k != key}))}


def check_seal(value: dict, key: str) -> None:
    if value.get(key) != seal(value, key)[key]:
        raise ShadowError(f"invalid_{key}")


def number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ShadowError(f"invalid_number:{name}")
    return float(value)


def bound_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ShadowError("artifact_path_outside_root")
    current = root.resolve()
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ShadowError("artifact_symlink")
    if not current.is_file():
        raise ShadowError(f"artifact_missing:{relative}")
    return current


def binding(root: Path, relative: str) -> dict:
    data = bound_path(root, relative).read_bytes()
    return {"path": relative, "sha256": digest(data), "size_bytes": len(data)}


def read_bound(root: Path, ref: dict) -> bytes:
    if not isinstance(ref, dict) or not SHA.fullmatch(str(ref.get("sha256", ""))):
        raise ShadowError("invalid_artifact_binding")
    data = bound_path(root, ref["path"]).read_bytes()
    if digest(data) != ref["sha256"] or len(data) != ref.get("size_bytes"):
        raise ShadowError(f"artifact_changed:{ref['path']}")
    return data


def read_json(root: Path, ref: dict) -> Any:
    return json.loads(read_bound(root, ref))


def episode_digest(episode: dict) -> str:
    return digest(canonical(episode))


def current_outcome(score: dict) -> str:
    if not isinstance(score, dict):
        raise ShadowError("current_score_not_object")
    if score.get("status") == "undetermined":
        return "unclear"
    if score.get("status") != "scored" or type(score.get("task_succeeded")) is not bool:
        raise ShadowError("invalid_current_score")
    return "success" if score["task_succeeded"] else "failure"


def _label(episode: dict, root: Path) -> dict:
    label = read_json(root, episode["artifacts"]["independent_label"])
    if not isinstance(label, dict):
        raise ShadowError("independent_label_not_object")
    if label.get("outcome") not in OUTCOMES:
        raise ShadowError("invalid_independent_outcome")
    if label.get("authority") not in {"human_review", "simulator_predicate_review"}:
        raise ShadowError("ground_truth_must_be_independent")
    if not label.get("reviewer_id") or label.get("review_status") != "confirmed":
        raise ShadowError("independent_review_missing")
    if label.get("episode_id") != episode["episode_id"]:
        raise ShadowError("ground_truth_episode_mismatch")
    contract = read_json(root, episode["artifacts"]["task_success_contract"])
    if set(label.get("reviewed_criteria", [])) != set(contract["criteria"]):
        raise ShadowError("ground_truth_acceptance_review_incomplete")
    if episode["artifacts"]["independent_label"]["sha256"] == episode["artifacts"]["baseline_output"]["sha256"]:
        raise ShadowError("ground_truth_is_baseline_output")
    refs = label.get("evidence", [])
    roles = {ref.get("role") for ref in refs}
    if not refs or not roles <= EVIDENCE_ROLES:
        raise ShadowError("ground_truth_circular_or_missing_evidence")
    if "task_success_contract" not in roles:
        raise ShadowError("ground_truth_contract_missing")
    required = {"state_trace"} if label["authority"] == "simulator_predicate_review" else {
        "review_video", "frame_manifest"}
    if not roles & required:
        raise ShadowError("ground_truth_observation_missing")
    for ref in refs:
        if ref.get("sha256") != episode["artifacts"][ref["role"]]["sha256"]:
            raise ShadowError("ground_truth_evidence_mismatch")
        if not ref.get("finding"):
            raise ShadowError("ground_truth_finding_missing")
    if label["authority"] == "simulator_predicate_review":
        checks = label.get("predicate_checks", [])
        if not checks or not label.get("independent_predicate_method"):
            raise ShadowError("independent_predicate_checks_missing")
        for check in checks:
            role = check.get("role")
            if role not in {"state_trace", "contact_force_trace"}:
                raise ShadowError("predicate_must_reference_native_trace")
            value = read_json(root, episode["artifacts"][role])
            pointer = check.get("json_pointer", "")
            if not pointer.startswith("/"):
                raise ShadowError("predicate_pointer_missing")
            for part in pointer[1:].split("/"):
                part = part.replace("~1", "/").replace("~0", "~")
                value = value[int(part)] if isinstance(value, list) else value[part]
            if value != check.get("observed") or not check.get("finding"):
                raise ShadowError("independent_predicate_readback_mismatch")
    for name, value in label.get("events", {}).items():
        if not 0 <= number(value, name) <= episode["duration_s"]:
            raise ShadowError("ground_truth_event_out_of_range")
    return label


def frame_rows(episode: dict, root: Path) -> list[dict]:
    """Read existing native media contracts without re-encoding or inventing time.

    Native frame paths are relative to their original run output directory;
    media_root supplies that retained directory relative to evidence-root.
    """
    manifest = read_json(root, episode["artifacts"]["frame_manifest"])
    if not isinstance(manifest, dict):
        raise ShadowError("frame_manifest_not_object")
    if "frames" in manifest:  # explicit normalized comparison sidecar
        return manifest["frames"]
    observations = []
    if manifest.get("schema_version") == "adp_multicamera_observation_frame_manifest.v1":
        observations = [*manifest["policy_input_observations"],
                        *manifest.get("review_observations", []), manifest["terminal_observation"]]
        raw = [frame for observation in observations for frame in observation["views"].values()]
    elif manifest.get("schema_version") == "adp_observation_frame_manifest.v1":
        raw = [*manifest["policy_input_frames"], manifest["terminal_observation"]]
    else:
        raise ShadowError("unsupported_native_frame_manifest")
    result = {}
    for frame in raw:
        if "simulation_time_s" not in frame or "camera_id" not in frame:
            raise ShadowError("native_frame_time_or_camera_missing_no_interpolation")
        relative = str(Path(episode.get("media_root", "")) / frame["relative_path"])
        row = {"path": relative, "sha256": frame["png_sha256"],
               "size_bytes": frame["size_bytes"], "timestamp_s": frame["simulation_time_s"],
               "camera_id": frame["camera_id"]}
        identity = (row["timestamp_s"], row["camera_id"])
        if identity in result and result[identity] != row:
            raise ShadowError("conflicting_native_frame_identity")
        result[identity] = row
    return list(result.values())


def validate_manifest(manifest: dict, root: Path, *, allow_fixtures: bool = False) -> list[dict]:
    if not isinstance(manifest, dict):
        raise ShadowError("manifest_not_object")
    check_seal(manifest, "manifest_digest")
    if manifest.get("schema_version") != SCHEMA:
        raise ShadowError("invalid_manifest_schema")
    if manifest.get("argus_commit") != ARGUS_COMMIT:
        raise ShadowError("argus_source_pin_mismatch")
    if manifest.get("argus_model") != TARGET_MODEL:
        raise ShadowError("argus_model_pin_mismatch")
    fixture = manifest.get("corpus_kind") == "synthetic_fixture"
    if manifest.get("corpus_kind") not in {"retained_episodes", "synthetic_fixture"}:
        raise ShadowError("invalid_corpus_kind")
    if fixture and not allow_fixtures:
        raise ShadowError("fixtures_require_explicit_flag")
    if not manifest.get("current_grader_commit") or not manifest.get("current_grader_sources"):
        raise ShadowError("current_grader_pin_missing")
    if manifest.get("current_grader_kind") != "adp_deterministic":
        raise ShadowError("unsupported_current_grader_requires_exact_adapter")
    if not fixture:
        if not re.fullmatch(r"[0-9a-f]{40}", manifest["current_grader_commit"]):
            raise ShadowError("current_grader_commit_not_immutable")
        if {ref["path"] for ref in manifest["current_grader_sources"]} != CURRENT_SOURCES:
            raise ShadowError("current_grader_source_closure_incomplete")
        for ref in manifest["current_grader_sources"]:
            actual = Path(__file__).parent / Path(ref["path"]).name
            if digest(actual.read_bytes()) != ref["sha256"]:
                raise ShadowError("executing_grader_differs_from_pin")
        read_bound(root, manifest["argus_source_pins"])
    for ref in manifest.get("evidence_inventories", []):
        read_bound(root, ref)
    for ref in manifest["current_grader_sources"]:
        read_bound(root, ref)
    if set(manifest["prompts"]) != set(ARMS[1:]):
        raise ShadowError("paired_prompt_pins_missing")
    for ref in manifest["prompts"].values():
        read_bound(root, ref)
    episodes = manifest["episodes"]
    if not isinstance(episodes, list):
        raise ShadowError("invalid_episode_list")
    ids = [ep["episode_id"] for ep in episodes]
    if len(set(ids)) != len(ids):
        raise ShadowError("duplicate_episode_id")
    if manifest.get("selection", {}).get("episode_ids") != ids:
        raise ShadowError("selection_changed")
    for ep in episodes:
        kind = ep.get("evidence_kind")
        if (fixture and kind != "synthetic_fixture") or (
            not fixture and kind not in {"simulator_recording", "physical_recording"}
        ):
            raise ShadowError("fixture_or_generated_media_cannot_be_real_episode")
        if not ep.get("intended_task") or not ep.get("provenance"):
            raise ShadowError("task_or_provenance_missing")
        if not fixture and (
            ep["provenance"].get("source_kind") != kind
            or ep["provenance"].get("synthetic") is not False
            or ep["provenance"].get("generated") is not False
            or any(ref["path"].startswith("tests/fixtures/") for ref in ep["artifacts"].values())
        ):
            raise ShadowError("real_episode_provenance_missing_or_fixture")
        if number(ep["duration_s"], "duration") <= 0:
            raise ShadowError("invalid_duration")
        rights = ep.get("rights", {})
        if rights.get("offline_review_allowed") is not True or not rights.get("basis"):
            raise ShadowError("offline_rights_missing")
        # This metadata is a retained rights finding, never new upload authority.
        if not ARTIFACTS <= ep["artifacts"].keys():
            raise ShadowError("episode_evidence_incomplete")
        for ref in ep["artifacts"].values():
            read_bound(root, ref)
        video = read_bound(root, ep["artifacts"]["review_video"])
        if not fixture and b"ftyp" not in video[:32]:
            raise ShadowError("review_video_not_mp4_or_mov")
        contract = read_json(root, ep["artifacts"]["task_success_contract"])
        if contract.get("provenance", {}).get("confirmation_status") != "confirmed":
            raise ShadowError("intended_acceptance_contract_unconfirmed")
        from .adp_articulated_task_success_contract import validate_task_success_contract
        spec = read_json(root, ep["artifacts"]["task_spec"])
        validate_task_success_contract(contract, task_kind=spec["task_kind"], require_confirmed=True)
        if spec["task_kind"] == "rigid_pick_place" and spec.get("task_success_contract") != contract:
            raise ShadowError("grader_and_argus_acceptance_contract_differ")
        if spec["task_kind"] == "articulated_open_close":
            from .adp_articulated_task_success_contract import compatibility_articulated_success_criteria
            if contract["criteria"] != compatibility_articulated_success_criteria(spec):
                raise ShadowError("grader_and_argus_acceptance_contract_differ")
        frames = frame_rows(ep, root)
        if not frames:
            raise ShadowError("lossless_frames_missing")
        identities = [(f["timestamp_s"], f["camera_id"]) for f in frames]
        if len(set(identities)) != len(identities):
            raise ShadowError("duplicate_frame_identity")
        for frame in frames:
            if not 0 <= number(frame["timestamp_s"], "frame_time") <= ep["duration_s"]:
                raise ShadowError("frame_time_out_of_range")
            if not frame.get("camera_id"):
                raise ShadowError("frame_camera_missing")
            read_bound(root, frame)
        _label(ep, root)
        current_outcome(read_json(root, ep["artifacts"]["baseline_output"]))
    return episodes


def replay_current(episode: dict, root: Path) -> dict:
    """Use the actual current dispatcher, with no duplicated scoring logic."""
    from .adp_task_scoring import score_task_episode_from_spec

    trace = read_json(root, episode["artifacts"]["state_trace"])
    spec = read_json(root, episode["artifacts"]["task_spec"])
    if not isinstance(trace, dict):
        raise ShadowError("state_trace_not_object")
    samples = trace.get("task_state_samples", trace.get("samples"))
    if not isinstance(samples, list) or not samples:
        raise ShadowError("native_task_state_samples_missing")
    result = score_task_episode_from_spec(task_spec=spec, samples=samples)
    retained = read_json(root, episode["artifacts"]["baseline_output"])
    if canonical(result) != canonical(retained):
        raise ShadowError("current_grader_replay_differs_from_retained_score")
    return result


def prepare(manifest: dict, root: Path, *, allow_fixtures: bool = False) -> dict:
    episodes = validate_manifest(manifest, root, allow_fixtures=allow_fixtures)
    requests = []
    for ep in episodes:
        frames = frame_rows(ep, root)
        ordered = sorted(frames, key=lambda row: (row["timestamp_s"], row["camera_id"]))
        times = sorted({row["timestamp_s"] for row in ordered})
        # Deterministic teleop cadence, first and last; no routing model call.
        selected = {times[0], times[-1]}
        for index in range(1, math.ceil(ep["duration_s"] / 1.5)):
            target = index * 1.5
            selected.add(min(times, key=lambda t: (abs(t - target), t)))
        chosen = [frame for frame in ordered if frame["timestamp_s"] in selected]
        blind = {role: ref for role, ref in ep["artifacts"].items()
                 if role in EVIDENCE_ROLES - {"frame_manifest"} or role == "action_trace"}
        for arm in ARMS[1:]:
            request = {"episode_id": ep["episode_id"], "arm": arm,
                       "manifest_digest": manifest["manifest_digest"],
                       "episode_digest": episode_digest(ep),
                       "intended_task": ep["intended_task"], "duration_s": ep["duration_s"],
                       "prompt": manifest["prompts"][arm], "artifacts": blind,
                       "frames": chosen, "source_frame_count": len(ordered),
                       "sampling_interval_s": 1.5, "grid_cell_width_px": 448,
                       "sampling_can_miss_transients": True,
                       "model": manifest["argus_model"], "reasoning_effort": "medium",
                       "max_input_tokens": 120000, "max_output_tokens": 64000, "model_calls": 0,
                       "request_kind": "offline_specification_not_provider_payload",
                       "target_inference_rights_admitted": ep["rights"].get(
                           "external_inference_allowed") is True}
            requests.append(seal(request, "request_digest"))
    return {"schema_version": "argus_shadow_plan.v1", "manifest_digest": manifest["manifest_digest"],
            "corpus_kind": manifest["corpus_kind"], "episode_count": len(episodes),
            "requests": requests, "model_calls": 0, "production_grader_changed": False}


def normalize_argus(raw: dict, duration_s: float) -> dict:
    """Import both upstream labels.completion and the adapted completion schema."""
    if not isinstance(raw, dict):
        raise ShadowError("argus_response_not_object")
    labels = raw.get("labels", raw)
    if not isinstance(labels, dict):
        raise ShadowError("argus_labels_not_object")
    if raw.get("parse_ok") is False or "_raw" in labels:
        return {"outcome": "unclear", "original_outcome": "parse_failure", "events": {},
                "explanation": "Unparseable retained response", "parse_failed": True,
                "confidence": None}
    completion = labels.get("completion", {})
    if not isinstance(completion, dict):
        raise ShadowError("argus_completion_not_object")
    original = completion.get("task_completed")
    if original not in {"success", "failure", "partial", "unclear", "success_then_undone"}:
        raise ShadowError("invalid_argus_outcome")
    outcome = original if original in OUTCOMES else "failure"
    events = {}
    for name in ("completed_at_s", "goal_reached_at_s", "undone_at_s"):
        value = completion.get(name)
        if value is not None:
            if not 0 <= number(value, name) <= duration_s:
                raise ShadowError("argus_event_out_of_range")
            events[name] = value
    if original == "success_then_undone" and (
        "goal_reached_at_s" not in events or "undone_at_s" not in events
        or events["undone_at_s"] < events["goal_reached_at_s"]
        or "completed_at_s" in events
    ):
        raise ShadowError("invalid_undo_timeline")
    if original == "success" and "undone_at_s" in events:
        raise ShadowError("success_contradicts_undo")
    if original != "success" and "completed_at_s" in events:
        raise ShadowError("non_success_has_completion_time")
    confidence = completion.get("confidence")
    if confidence is not None and not 0 <= number(confidence, "confidence") <= 1:
        raise ShadowError("invalid_confidence")
    return {"outcome": outcome, "original_outcome": original, "events": events,
            "explanation": completion.get("reason", ""), "confidence": confidence,
            "parse_failed": False}


def _rate(count: int, denominator: int) -> dict:
    return {"count": count, "denominator": denominator,
            "rate": count / denominator if denominator else None}


def metrics(rows: list[dict], arm: str) -> dict:
    confusion = Counter((r["truth"]["outcome"], r["arms"][arm]["outcome"]) for r in rows)
    failures = sum(r["truth"]["outcome"] == "failure" for r in rows)
    successes = sum(r["truth"]["outcome"] == "success" for r in rows)
    unclear = sum(r["truth"]["outcome"] == "unclear" for r in rows)
    brier, timestamp_errors = [], []
    truth_event_count = sum(len(r["truth"].get("events", {})) for r in rows)
    for row in rows:
        result, truth = row["arms"][arm], row["truth"]
        confidence = result.get("confidence")
        # Confidence in an annotation is not a probability of task success.
        if result.get("probability_semantics") == "p_contract_success" and confidence is not None:
            if truth["outcome"] != "unclear":
                brier.append((confidence - int(truth["outcome"] == "success")) ** 2)
        for event, value in truth.get("events", {}).items():
            if event in result["events"]:
                timestamp_errors.append(abs(value - result["events"][event]))
    return {"confusion": {t: {p: confusion[t, p] for p in sorted(OUTCOMES)}
                          for t in sorted(OUTCOMES)},
            "false_success": _rate(confusion["failure", "success"], failures),
            "missed_failure": _rate(confusion["failure", "success"] + confusion["failure", "unclear"], failures),
            "missed_success": _rate(confusion["success", "failure"], successes),
            "failure_abstention": _rate(confusion["failure", "unclear"], failures),
            "unsupported_success_on_ambiguous_truth": _rate(confusion["unclear", "success"], unclear),
            "abstention": _rate(sum(r["arms"][arm]["outcome"] == "unclear" for r in rows), len(rows)),
            "timestamp_matched_event_count": len(timestamp_errors),
            "timestamp_reference_event_count": truth_event_count,
            "timestamp_missing_event_count": truth_event_count - len(timestamp_errors),
            "timestamp_mean_absolute_error_s": sum(timestamp_errors) / len(timestamp_errors) if timestamp_errors else None,
            "brier_score": sum(brier) / len(brier) if brier else None,
            "calibration_sample_count": len(brier),
            "parse_failure_count": sum(bool(r["arms"][arm].get("parse_failed")) for r in rows),
            "calibration_limit": "No calibrated-probability claim; unclear truth excluded; small selected samples are descriptive.",
            "explanation_usefulness": "Requires blinded human review against cited evidence; text presence is not usefulness."}


def compare(manifest: dict, root: Path, records: list[dict], *, allow_fixtures: bool = False) -> dict:
    episodes = validate_manifest(manifest, root, allow_fixtures=allow_fixtures)
    plan = prepare(manifest, root, allow_fixtures=allow_fixtures)
    requests = {(r["episode_id"], r["arm"]): r for r in plan["requests"]}
    by_pair = {}
    if not isinstance(records, list):
        raise ShadowError("response_records_not_array")
    for record in records:
        if not isinstance(record, dict):
            raise ShadowError("response_record_not_object")
        pair = (record["episode_id"], record["arm"])
        if pair in by_pair or pair not in requests:
            raise ShadowError("duplicate_or_unselected_response")
        request = requests[pair]
        if (record.get("request_digest") != request["request_digest"]
            or record.get("episode_digest") != request["episode_digest"]
            or record.get("manifest_digest") != manifest["manifest_digest"]):
            raise ShadowError("response_not_bound_to_paired_input")
        identity = record.get("inference_identity", {})
        if not isinstance(identity, dict):
            raise ShadowError("response_inference_identity_not_object")
        if not all(identity.get(k) for k in ("model_requested", "model_served", "provider", "generation_id")):
            raise ShadowError("response_inference_identity_missing")
        if identity["model_requested"] != request["model"]:
            raise ShadowError("response_model_mismatch")
        if manifest["corpus_kind"] != "synthetic_fixture" and identity["model_served"] not in {
            request["model"], request["model"].split("/", 1)[-1]
        }:
            raise ShadowError("unadmitted_served_model_requires_new_manifest")
        raw = read_json(root, record["raw_response"])
        by_pair[pair] = (raw, record)
    if set(by_pair) != set(requests):
        raise ShadowError("paired_responses_missing")
    rows = []
    for ep in episodes:
        identities = [by_pair[ep["episode_id"], arm][1]["inference_identity"] for arm in ARMS[1:]]
        if len({(identity["provider"], identity["model_served"]) for identity in identities}) != 1:
            raise ShadowError("paired_served_model_or_provider_mismatch")
        current = replay_current(ep, root)
        arms = {"current": {"outcome": current_outcome(current), "events": {},
                            "explanation": current.get("failure_reason_plain_english", ""),
                            "confidence": None}}
        for arm in ARMS[1:]:
            raw, record = by_pair[ep["episode_id"], arm]
            arms[arm] = {**normalize_argus(raw, ep["duration_s"]),
                         "raw_response": record["raw_response"],
                         "request_digest": record["request_digest"],
                         "inference_identity": {k: record["inference_identity"].get(k)
                             for k in ("model_requested", "model_served", "provider", "generation_id",
                                       "system_fingerprint")}}
        rows.append({"episode_id": ep["episode_id"], "episode_digest": episode_digest(ep),
                     "categories": ep.get("categories", []), "truth": _label(ep, root), "arms": arms})
    fixture = manifest["corpus_kind"] == "synthetic_fixture"
    return {"schema_version": "argus_shadow_report.v1", "manifest_digest": manifest["manifest_digest"],
            "status": "fixture_only" if fixture else "descriptive_shadow" if rows else "blocked_no_retained_episodes",
            "episode_count": len(rows), "real_episode_count": 0 if fixture else len(rows),
            "ground_truth_counts": dict(Counter(r["truth"]["outcome"] for r in rows)),
            "metrics": {arm: metrics(rows, arm) for arm in ARMS}, "episodes": rows,
            "category_metrics": {category: {arm: metrics(
                [r for r in rows if category in r["categories"]], arm) for arm in ARMS}
                for category in sorted({c for row in rows for c in row["categories"]})},
            "explanation_review_required": {
                "blinding": "Randomize arm order and conceal model identity before human review",
                "rubric": ["evidence accuracy", "criterion coverage", "useful failure cause",
                           "timestamp accuracy", "uncertainty honesty"],
                "reviewed_explanations": 0},
            "policy_ranking": {"status": "not_supported", "reason": "No preregistered, powered, paired policy sample admitted."},
            "claim_ceiling": "Offline fixture correctness" if fixture else "Retained evidence comparison; no physical validity or evaluator replacement",
            "production_grader_changed": False, "model_calls": 0}


def cost_proposal(manifest: dict, root: Path, *, allow_fixtures: bool = False) -> dict:
    episodes = validate_manifest(manifest, root, allow_fixtures=allow_fixtures)
    seconds = sum(ep["duration_s"] for ep in episodes)
    # Research page reports GPT-6.1 Sol medium at ~$5/footage hour, not a quote.
    estimate = seconds / 3600 * 5 * 2
    return {"schema_version": "argus_shadow_cost_proposal.v1", "admitted_episode_count": len(episodes),
            "footage_seconds": seconds, "argus_arms": 2, "current_replay_usd": 0,
            "estimated_argus_usd": round(estimate, 6), "paid_execution_authorized": False,
            "target_model": TARGET_MODEL, "reasoning_effort": "medium",
            "spend_status": "deferred_billing_incident_no_new_paid_approval_requested",
            "illustrative_rate_comparison_only": {"episodes": 12, "seconds_per_episode_assumed": 60,
                                  "success_failure_ambiguous_each": 4,
                                  "argus_requests": 24, "sol61_research_page_estimated_usd": 2.00,
                                  "sol61_readme_ratio_estimated_usd": 2.08,
                                  "astra_research_page_estimated_usd": 10.80,
                                  "astra_readme_teleop_estimated_usd": 10.40,
                                  "proposed_aggregate_cap_usd": None,
                                  "sol61_token_bound_usd": None,
                                  "retry_count": 0},
            "estimate_basis": "Pantheon research page: GPT-6.1 Sol medium ~$5/footage hour vs Astra ~$27. Pinned README: Astra teleop ~$26/hour and Sol6.1 medium 20% of Astra per episode. Neither is an official quote.",
            "execution_gate": "Paid work is deferred during billing incident. Trace/admit actual footage and independent labels first; current official Sol6.1 quote, disclosure terms and aggregate/request reservations are required before any later approval or send.",
            "unrelated_budgets_available_usd": 0, "model_calls": 0}


def write_new_output(path: Path, result: dict) -> None:
    """Idempotent outputs without overwriting any retained evidence or source."""
    content = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    absolute = path.absolute()
    if any(p.is_symlink() for p in (absolute, *absolute.parents)):
        raise ShadowError("output_symlink")
    if absolute.exists():
        if absolute.is_file() and absolute.read_bytes() == content:
            return
        raise ShadowError("output_exists_no_overwrite")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    with absolute.open("xb") as stream:
        stream.write(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "prepare", "compare", "cost"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-synthetic-fixtures", action="store_true")
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text())
        kwargs = {"allow_fixtures": args.allow_synthetic_fixtures}
        if args.command == "validate":
            episodes = validate_manifest(manifest, args.evidence_root, **kwargs)
            result = {"valid": True, "episode_count": len(episodes), "model_calls": 0}
        elif args.command == "prepare":
            result = prepare(manifest, args.evidence_root, **kwargs)
        elif args.command == "cost":
            result = cost_proposal(manifest, args.evidence_root, **kwargs)
        else:
            if args.responses is None:
                raise ShadowError("responses_required")
            result = compare(manifest, args.evidence_root,
                             json.loads(args.responses.read_text()), **kwargs)
        write_new_output(args.output, result)
    except (ShadowError, KeyError, IndexError, TypeError, ValueError, OSError) as exc:
        parser.exit(2, f"argus_shadow_refused:{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
