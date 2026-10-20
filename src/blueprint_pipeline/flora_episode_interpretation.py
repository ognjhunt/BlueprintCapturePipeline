"""Research-only FLORA visual rewards behind the existing interpreter contract.

ADP-009D / Day-28: optional, replayable interpretation of already sealed episode
evidence. None of these learned scores can change deterministic task success.
Reference scores, policy identity, state traces and contact traces are withheld.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

from .adp_task_scoring import validate_rigid_task_success_contract
from .decision_evidence_contracts import canonical_digest
from .episode_interpretation import (
    EpisodeInterpretationError,
    EpisodeInterpretationRequest,
    EpisodeInterpreterOutput,
    InterpretationEvidenceRef,
    InterpretedEpisodeEvent,
    InterpreterIdentity,
    PossibleMissedEvent,
    validate_episode_interpretation_rights,
)
from .flora_rewarder_runtime import (
    FLORA_MODEL,
    FLORA_REVISION,
    MODES,
    SOURCE_SHA256,
    FloraSubprocessRunner,
    file_digest,
    preflight_flora_runtime,
)


PROFILE_SCHEMA = "policy_canary_episode_interpreter_profile.flora.v1"
INTERPRETER_ID = "flora_episode_interpreter_v1"
RUNTIME = "flora_rewarder_batch_v1"
PROMPT_VERSION = "flora_rewarder_prompts.v1"
SAMPLING = {
    "terminal": {"max_frames": 64, "max_fps": 10, "temperature": 0.0, "with_rationale": True},
    "dense": {"frames_per_window": 64, "fps": 4, "window_seconds": 16,
              "stride_seconds": 10, "tail_trim_seconds": 4, "temperature": 0.0},
    "rubric": {"fps": 2, "max_frames_per_camera": 166, "temperature": 1.0,
               "max_new_tokens": 20480, "selection": "medoid"},
}
_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _fail(reason: str) -> None:
    raise EpisodeInterpretationError(reason)


def _number(value: Any, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("flora_numeric_value_invalid")
    if not math.isfinite(value) or not minimum <= value <= maximum:
        _fail("flora_numeric_value_invalid")
    return float(value)


def _integer(value: Any, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail("flora_integer_value_invalid")
    return value


def _text(value: Any, maximum: int = 4000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        _fail("flora_text_value_invalid")
    return value


def validate_flora_interpreter_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a sealed configuration without loading a model or touching compute."""
    value = copy.deepcopy(dict(profile))
    fixed = {"schema_version": PROFILE_SCHEMA, "status": "configured",
             "interpreter_id": INTERPRETER_ID, "provider_id": "nvidia", "runtime": RUNTIME,
             "model": FLORA_MODEL, "model_version": FLORA_REVISION,
             "source_revision": FLORA_REVISION, "research_only": True, "modes": list(MODES)}
    if any(value.get(key) != expected for key, expected in fixed.items()):
        _fail("flora_profile_identity_invalid")
    if value.get("research_only") is not True or value.get("execution_site") not in {"local", "external_provider"}:
        _fail("flora_profile_usage_invalid")
    if (value["execution_site"] == "external_provider"
            and not re.fullmatch(r"[a-z][a-z0-9_-]{1,63}", str(value.get("compute_provider_id", "")))):
        _fail("flora_external_compute_provider_missing")
    if value.get("profile_digest") != canonical_digest(value, digest_field="profile_digest"):
        _fail("flora_profile_digest_invalid")
    config, task = value.get("runtime_config"), value.get("task")
    if not isinstance(config, dict) or not isinstance(task, dict):
        _fail("flora_profile_configuration_missing")
    for key in ("python_executable", "source_root", "model_path", "output_root"):
        if not Path(_text(config.get(key))).is_absolute():
            _fail("flora_runtime_path_must_be_absolute")
    _integer(config.get("timeout_seconds"), 1, 3600)
    _integer(config.get("tensor_parallel_size"), 1, 8)
    _integer(config.get("seed"), 0, 2**31 - 1)
    _integer(config.get("rubric_num_samples"), 1, 32)
    _text(task.get("instruction"))
    _text(task.get("robot"), 500)
    if not _SHA.fullmatch(str(task.get("task_success_contract_digest", ""))):
        _fail("flora_task_contract_digest_invalid")
    cameras = task.get("cameras")
    if not isinstance(cameras, list) or not 1 <= len(cameras) <= 4:
        _fail("flora_camera_inventory_invalid")
    for camera in cameras:
        _text(camera, 500)
    rubric = task.get("rubric")
    if not isinstance(rubric, list) or not 1 <= len(rubric) <= 20:
        _fail("flora_rubric_invalid")
    maxima = []
    for step in rubric:
        if not isinstance(step, dict) or set(step) != {"action", "grading"}:
            _fail("flora_rubric_step_invalid")
        _text(step["action"])
        grading = step["grading"]
        if not isinstance(grading, list) or not 2 <= len(grading) <= 5:
            _fail("flora_rubric_grading_invalid")
        points = []
        for level in grading:
            if not isinstance(level, dict) or set(level) != {"label", "points"}:
                _fail("flora_rubric_level_invalid")
            _text(level["label"])
            point = _number(level["points"])
            # Upstream renders allowed levels with two decimal places.
            if round(point, 2) != point:
                _fail("flora_rubric_level_not_exactly_renderable")
            points.append(point)
        if len(set(points)) != len(points) or min(points) != 0 or max(points) <= 0:
            _fail("flora_rubric_levels_invalid")
        maxima.append(max(points))
    if not math.isclose(sum(maxima), 1.0, abs_tol=1e-9):
        _fail("flora_rubric_maxima_must_sum_to_one")
    return value


def _finite_tree(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _fail("flora_nonfinite_output")
    if isinstance(value, Mapping):
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            _finite_tree(child)


def _terminal(row: Mapping[str, Any]) -> dict[str, Any]:
    score = _integer(row.get("score"), 1, 5)
    raw = _text(row.get("output"), 8000)
    match = re.fullmatch(r"Rationale:\s*(.*?)\n\s*\nScore:\s*([1-5])\s*", raw.strip(), re.DOTALL)
    if not match or int(match[2]) != score or " ".join(match[1].split()) != row.get("rationale"):
        _fail("flora_terminal_raw_answer_invalid")
    _text(row.get("rationale"), 4000)
    expected = _number(row.get("expected_score"), minimum=1, maximum=5)
    progress = _number(row.get("progress"))
    probabilities = row.get("probabilities")
    if not isinstance(probabilities, Mapping) or set(probabilities) != set("12345"):
        _fail("flora_terminal_probabilities_invalid")
    for probability in probabilities.values():
        _number(probability)
    if (abs(sum(probabilities.values()) - 1) > 0.0003
            or abs(sum(int(k) * p for k, p in probabilities.items()) - expected) > 0.002
            or abs((expected - 1) / 4 - progress) > 0.0002):
        _fail("flora_terminal_summary_inconsistent")
    _number(row.get("duration_s"), minimum=0.001, maximum=86400)
    _integer(row.get("sampled_frames"), 2, 64)
    return {"score": score, "expected_score": expected, "progress_uncalibrated": progress,
            "score_token_probabilities_uncalibrated": dict(probabilities), "apparent_pass": score == 5}


def _runs(scores: Sequence[int]) -> list[tuple[int, int, int]]:
    result = []
    start = 0
    for index in range(1, len(scores) + 1):
        if index == len(scores) or scores[index] != scores[start]:
            result.append((start, index, scores[start]))
            start = index
    return result


def _dense(row: Mapping[str, Any]) -> dict[str, Any]:
    if type(row.get("invalid_clips")) is not int or row["invalid_clips"] != 0:
        _fail("flora_dense_invalid_clips")
    clips = row.get("clips")
    if not isinstance(clips, list) or not clips:
        _fail("flora_dense_clips_missing")
    for clip in clips:
        if not isinstance(clip, Mapping) or clip.get("valid") is not True:
            _fail("flora_dense_normal_fallback_forbidden")
        quality = clip.get("quality")
        if not isinstance(quality, str) or not re.fullmatch("[123]{32}", quality):
            _fail("flora_dense_quality_invalid")
        raw = _text(clip.get("output"), 64000)
        match = re.fullmatch(r"Worst: ([123])\nQuality:\n([123]{32})\nRationales:(.*)", raw.strip(), re.DOTALL)
        if not match or match[2] != quality or int(match[1]) != min(map(int, quality)):
            _fail("flora_dense_raw_answer_invalid")
        expected = {(start + 1, end) for start, end, score in _runs(list(map(int, quality))) if score < 3}
        found = set()
        for line in match[3].strip().splitlines():
            rationale = re.fullmatch(r"(\d+)-(\d+) (\S.*)", line)
            if not rationale or (int(rationale[1]), int(rationale[2])) in found:
                _fail("flora_dense_rationale_invalid")
            found.add((int(rationale[1]), int(rationale[2])))
        if found != expected:
            _fail("flora_dense_rationale_coverage_invalid")
    frames = row.get("frame_scores")
    count = _integer(row.get("num_frames"), 1, 10_000_000)
    fps = _number(row.get("fps"), minimum=0.001, maximum=1000)
    duration = _number(row.get("duration_s"), minimum=0.001, maximum=86400)
    if not isinstance(frames, list) or len(frames) != count:
        _fail("flora_dense_frame_scores_missing")
    for score in frames:
        _integer(score, 1, 3)
    mean = _number(row.get("mean_score"), minimum=1, maximum=3)
    worst = _integer(row.get("worst_score"), 1, 3)
    if worst != min(frames) or abs(mean - sum(frames) / count) > 0.0001 or abs(duration - count / fps) > 0.001:
        _fail("flora_dense_summary_inconsistent")
    fractions = row.get("fraction")
    if not isinstance(fractions, Mapping) or set(fractions) != {"failure", "suboptimal", "normal"}:
        _fail("flora_dense_fraction_invalid")
    for score, label in ((1, "failure"), (2, "suboptimal"), (3, "normal")):
        if abs(_number(fractions[label]) - frames.count(score) / count) > 0.0001:
            _fail("flora_dense_fraction_invalid")
    segments = row.get("segments")
    runs = _runs(frames)
    if not isinstance(segments, list) or len(segments) != len(runs):
        _fail("flora_dense_segment_coverage_invalid")
    for segment, (start, end, score) in zip(segments, runs):
        if (not isinstance(segment, Mapping) or segment.get("start_frame") != start
                or segment.get("end_frame") != end or segment.get("score") != score
                or abs(_number(segment.get("start_s"), maximum=86400) - start / fps) > 0.001
                or abs(_number(segment.get("end_s"), maximum=86400) - end / fps) > 0.001):
            _fail("flora_dense_segment_invalid")
        if score < 3:
            _text(segment.get("rationale"), 4000)
    per_second = [frames[min(count - 1, int((t + 0.5) * fps))] for t in range(math.ceil(count / fps))]
    if row.get("per_second") != per_second:
        _fail("flora_dense_per_second_invalid")
    return {"worst_score": worst, "mean_score": mean, "fraction": dict(fractions),
            "invalid_clips": 0, "failure_spans": [
                {key: segment[key] for key in ("start_s", "end_s", "rationale")}
                for segment in segments if segment["score"] == 1],
            "apparent_pass": worst > 1}


def _rubric(row: Mapping[str, Any], task: Mapping[str, Any], sample_count: int) -> dict[str, Any]:
    rubric = task["rubric"]
    allowed = [{level["points"] for level in step["grading"]} for step in rubric]
    samples = row.get("samples")
    if not isinstance(samples, list) or len(samples) != sample_count or row.get("parsed_samples") != sample_count:
        _fail("flora_rubric_samples_missing_or_invalid")
    vectors = []
    for sample in samples:
        if not isinstance(sample, Mapping) or sample.get("parsed") is not True:
            _fail("flora_rubric_sample_parse_failed")
        raw = _text(sample.get("answer"), 100000)
        match = re.search(r"ANSWER:\s*(\[[^\[\]]*\])\s*\Z", raw)
        if not match:
            _fail("flora_rubric_answer_invalid")
        try:
            scores = json.loads(match[1])
        except json.JSONDecodeError:
            _fail("flora_rubric_answer_invalid")
        if not isinstance(scores, list) or len(scores) != len(allowed):
            _fail("flora_rubric_answer_invalid")
        for score, values in zip(scores, allowed):
            if _number(score) not in values:
                _fail("flora_rubric_score_not_allowed")
        if sample.get("scores") != scores:
            _fail("flora_rubric_clamped_score_forbidden")
        vectors.append(scores)
    medoid = min(range(sample_count), key=lambda i: (
        sum(sum(abs(a - b) for a, b in zip(vectors[i], vector)) for vector in vectors), i))
    scores = vectors[medoid]
    if (type(row.get("medoid_sample")) is not int or row["medoid_sample"] != medoid
            or row.get("scores") != scores or row.get("steps") != [step["action"] for step in rubric]
            or abs(_number(row.get("total")) - sum(scores)) > 0.0001):
        _fail("flora_rubric_summary_invalid")
    return {"scores": scores, "total": sum(scores), "sample_count": sample_count,
            "medoid_sample": medoid, "apparent_pass": all(s == max(a) for s, a in zip(scores, allowed))}


def normalize_flora_rewards(records: Mapping[str, Any], *, job: Mapping[str, Any],
                            sample_count: int) -> dict[str, Any]:
    """Validate the released scripts' actual JSON, including unclamped raw text."""
    _finite_tree(records)
    if set(records) != set(MODES):
        _fail("flora_required_mode_missing")
    for row in records.values():
        if not isinstance(row, Mapping) or row.get("id") != job["id"] or row.get("task") != job["task"]:
            _fail("flora_result_identity_mismatch")
    rewards = {"terminal": _terminal(records["terminal"]), "dense": _dense(records["dense"]),
               "rubric": _rubric(records["rubric"], job, sample_count)}
    if abs(records["terminal"]["duration_s"] - records["dense"]["duration_s"]) > 0.002:
        _fail("flora_mode_duration_mismatch")
    return rewards


def _safe_raw(value: Any) -> Any:
    """Retain a typed representation even when invalid output contains NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_nonfinite_number": repr(value)}
    if isinstance(value, Mapping):
        return {str(key): _safe_raw(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_raw(child) for child in value]
    return value


def verify_flora_raw_artifacts(
    metadata: Mapping[str, Any], *, artifact_root: str | Path | None = None,
) -> None:
    """Reopen retained raw files, optionally beneath a relocated raw-output root.

    ``artifact_root`` replaces only the recorded ``raw_output_root``. Sealed
    relative paths, original absolute paths and digests remain unchanged.
    """
    origin = Path(str(metadata.get("raw_output_root", "")))
    if not origin.is_absolute() or metadata.get("raw_artifact_schema_version") != "flora_raw_artifacts.v1":
        _fail("flora_raw_artifact_origin_invalid")
    root = Path(artifact_root).expanduser().absolute() if artifact_root is not None else origin
    artifacts = metadata.get("raw_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        _fail("flora_raw_artifacts_missing")
    indexed: dict[tuple[str, str | None], tuple[Mapping[str, Any], Path]] = {}
    paths: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            _fail("flora_raw_artifact_invalid")
        relative = Path(str(artifact.get("relative_path", "")))
        kind, mode = artifact.get("kind"), artifact.get("mode")
        if (relative.is_absolute() or not relative.parts or ".." in relative.parts
                or kind not in {"episode_raw", "native_output", "runtime_log"}
                or (kind == "episode_raw" and mode is not None)
                or (kind != "episode_raw" and mode not in MODES)
                or artifact.get("path") != str(origin / relative)
                or not _SHA.fullmatch(str(artifact.get("sha256", "")))
                or (kind, mode) in indexed or str(relative) in paths):
            _fail("flora_raw_artifact_binding_invalid")
        target = root / relative
        if (any(path.is_symlink() for path in (target, *target.parents))
                or not target.is_file() or file_digest(target) != artifact["sha256"]):
            _fail("flora_raw_artifact_missing_or_changed")
        indexed[(kind, mode)] = (artifact, target)
        paths.add(str(relative))
    raw = indexed.get(("episode_raw", None))
    if raw is None:
        _fail("flora_episode_raw_artifact_missing")
    try:
        if json.loads(raw[1].read_text()) != metadata.get("raw_outputs"):
            _fail("flora_episode_raw_artifact_mismatch")
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EpisodeInterpretationError("flora_episode_raw_artifact_invalid") from exc
    executions = metadata.get("runtime_execution")
    if not isinstance(executions, list):
        _fail("flora_raw_execution_bindings_invalid")
    claimed = {("episode_raw", None)}
    seen_modes: set[str] = set()
    for execution in executions:
        if not isinstance(execution, Mapping) or execution.get("mode") not in MODES:
            _fail("flora_raw_execution_bindings_invalid")
        mode = execution["mode"]
        if mode in seen_modes:
            _fail("flora_raw_execution_bindings_invalid")
        seen_modes.add(mode)
        for kind, path_key, digest_key in (("native_output", "output_path", "output_sha256"),
                                          ("runtime_log", "log_path", "log_sha256")):
            if path_key not in execution and digest_key not in execution:
                if metadata.get("runtime_kind") == "live_subprocess" and (
                    kind == "runtime_log" or execution.get("returncode") == 0
                ):
                    _fail("flora_raw_execution_artifact_missing")
                continue
            artifact = indexed.get((kind, mode))
            if (artifact is None or artifact[0]["path"] != execution.get(path_key)
                    or artifact[0]["sha256"] != execution.get(digest_key)):
                _fail("flora_raw_execution_artifact_mismatch")
            claimed.add((kind, mode))
            if kind != "native_output":
                continue
            expected = metadata.get("raw_outputs", {}).get(mode)
            try:
                rows = [json.loads(line) for line in artifact[1].read_text().splitlines() if line.strip()]
            except (UnicodeError, json.JSONDecodeError) as exc:
                # Malformed native JSON is retained failure evidence. It cannot
                # support a successful mode, but must not erase an abstention.
                if metadata.get("status") == "abstained" and expected is None:
                    continue
                raise EpisodeInterpretationError("flora_native_raw_output_invalid") from exc
            if expected is not None:
                matches = [row for row in rows if isinstance(row, Mapping) and row.get("id") == expected.get("id")]
                if len(matches) != 1 or _safe_raw(matches[0]) != expected:
                    _fail("flora_native_raw_output_mismatch")
    if set(indexed) != claimed:
        _fail("flora_raw_artifact_unclaimed")


class FloraEpisodeInterpreter:
    def __init__(self, profile: Mapping[str, Any], *, output_root: str | Path | None = None,
                 runner: Any = None) -> None:
        self.profile = validate_flora_interpreter_profile(profile)
        self.output_root = Path(output_root or self.profile["runtime_config"]["output_root"]).expanduser().resolve()
        self.runner = runner if runner is not None else FloraSubprocessRunner()
        self._live_runner = type(self.runner) is FloraSubprocessRunner
        self._cache: dict[str, tuple[EpisodeInterpreterOutput, dict[str, Any]]] = {}
        self._metadata: dict[str, Any] = {}

    @classmethod
    def from_profile(cls, profile: Mapping[str, Any], *, output_root: str | Path | None = None,
                     runner: Any = None) -> FloraEpisodeInterpreter:
        return cls(profile, output_root=output_root, runner=runner)

    @property
    def identity(self) -> InterpreterIdentity:
        provider = self.profile["compute_provider_id"] if self.profile["execution_site"] == "external_provider" else "nvidia"
        return InterpreterIdentity(INTERPRETER_ID, "independent_interpreter", provider,
                                   self.profile["execution_site"], RUNTIME, FLORA_MODEL, FLORA_REVISION)

    def disclosed_artifact_roles(self, request: EpisodeInterpretationRequest) -> Sequence[str]:
        del request
        return ("task_success_contract", "review_video")

    def prompt_identity(self) -> dict[str, str]:
        return {"prompt_contract_version": PROMPT_VERSION, "prompt_digest": canonical_digest({
            "version": PROMPT_VERSION, "source_sha256": SOURCE_SHA256,
            "task": self.profile["task"], "sampling": SAMPLING,
            "rubric_num_samples": self.profile["runtime_config"]["rubric_num_samples"]})}

    def adapter_metadata(self) -> dict[str, Any]:
        return copy.deepcopy(self._metadata)

    def preflight(self, *, output_root: str | Path | None = None) -> dict[str, Any]:
        return preflight_flora_runtime(self.profile, output_root=output_root or self.output_root / "runtime_preflight")

    def _input(self, request: EpisodeInterpretationRequest) -> tuple[dict[str, Any], str]:
        contract = validate_rigid_task_success_contract(request.task_success_contract, require_confirmed=True)
        if contract["contract_digest"] != self.profile["task"]["task_success_contract_digest"]:
            _fail("flora_task_contract_mismatch")
        receipt = request.input_receipt
        if receipt.get("input_bundle_digest") != canonical_digest(receipt, digest_field="input_bundle_digest"):
            _fail("flora_input_bundle_digest_invalid")
        bindings = receipt["artifacts"]["review_videos"]
        if (not request.review_video_paths or len(bindings) != len(request.review_video_paths)
                or len(bindings) != len(self.profile["task"]["cameras"])):
            _fail("flora_video_camera_count_mismatch")
        for index, (path, binding) in enumerate(zip(request.review_video_paths, bindings)):
            if (path.is_symlink() or not path.is_file() or path.stat().st_size <= 0
                    or path.resolve().is_relative_to(request.evidence_root.resolve()) is False
                    or file_digest(path) != binding["sha256"]
                    or binding.get("video_index") != index):
                _fail("flora_video_digest_invalid")
        task = self.profile["task"]
        selected = {key: copy.deepcopy(task[key]) for key in ("instruction", "robot", "cameras", "rubric")}
        selected["video_digests"] = [row["sha256"] for row in bindings]
        selected["sampling"] = SAMPLING
        selected["generation_settings"] = {
            key: self.profile["runtime_config"][key]
            for key in ("seed", "rubric_num_samples", "tensor_parallel_size")
        }
        selected["prompt_digest"] = self.prompt_identity()["prompt_digest"]
        return selected, canonical_digest(selected)

    def run_batch(self, requests: Sequence[EpisodeInterpretationRequest], *,
                  rights_attestation_paths: Mapping[str, str | Path]) -> dict[str, EpisodeInterpreterOutput]:
        """Validate every rights binding, then infer the whole cohort three times."""
        if len({request.episode_id for request in requests}) != len(requests):
            _fail("flora_duplicate_episode_id")
        for request in requests:
            path = rights_attestation_paths.get(request.episode_id)
            if path is None:
                _fail("episode_interpretation_rights_missing")
            validate_episode_interpretation_rights(rights_path=path, request=request, interpreter=self)
        self._execute_batch(requests)
        return {request.episode_id: self._cache[request.input_receipt["input_bundle_digest"]][0]
                for request in requests}

    def interpret(self, request: EpisodeInterpretationRequest) -> EpisodeInterpreterOutput:
        key = request.input_receipt["input_bundle_digest"]
        if key not in self._cache:
            self._execute_batch([request])
        output, metadata = self._cache[key]
        # A cached result is valid only for the same unchanged video bytes and
        # retained raw output. No stale-result reuse after evidence mutation.
        _, digest = self._input(request)
        if digest != metadata["model_input_digest"]:
            _fail("flora_cached_input_changed")
        verify_flora_raw_artifacts(metadata)
        self._metadata = metadata
        return output.model_copy(deep=True)

    def _execute_batch(self, requests: Sequence[EpisodeInterpretationRequest]) -> None:
        pending = [r for r in requests if r.input_receipt["input_bundle_digest"] not in self._cache]
        if not pending:
            return
        cohort_digest = canonical_digest({"inputs": [r.input_receipt["input_bundle_digest"] for r in pending]})
        root = self.output_root / ("batch-" + cohort_digest.split(":")[1][:24])
        if root.exists():
            _fail("flora_batch_output_exists")
        root.mkdir(parents=True)
        jobs, prepared = [], []
        for request in pending:
            selected, digest = self._input(request)
            opaque_id = "episode-" + request.input_receipt["input_bundle_digest"].split(":")[1][:32]
            videos = []
            for index, source in enumerate(request.review_video_paths):
                target = root / "media" / opaque_id / (f"camera-{index}" + source.suffix.lower())
                target.parent.mkdir(parents=True, exist_ok=True)
                # Independent bytes protect retained evidence even if a model
                # process writes to a staged input. Never hardlink source media.
                shutil.copyfile(source, target)
                target.chmod(0o444)
                if file_digest(target) != selected["video_digests"][index]:
                    _fail("flora_staged_video_digest_invalid")
                videos.append(str(target))
            job = {"id": opaque_id, "task": selected["instruction"], "robot": selected["robot"],
                   "cameras": selected["cameras"], "rubric": selected["rubric"], "videos": videos}
            jobs.append(job)
            prepared.append((request, job, digest))
        try:
            result = self.runner.run_batch(profile=self.profile, jobs=jobs, output_root=root)
        except Exception as exc:
            result = {"records": {}, "errors": ["flora_runtime_exception:" + type(exc).__name__], "executions": []}
        if not isinstance(result, Mapping):
            result = {"records": {}, "errors": ["flora_batch_output_invalid"], "executions": []}
        records_by_id: dict[str, dict[str, Any]] = {job["id"]: {} for job in jobs}
        errors = list(result.get("errors", []))
        raw_records = result.get("records", {})
        if not isinstance(raw_records, Mapping):
            errors.append("flora_batch_output_invalid")
            raw_records = {}
        for mode, records in raw_records.items():
            if mode not in MODES or not isinstance(records, list):
                errors.append("flora_batch_output_invalid")
                continue
            seen = set()
            for record in records:
                identifier = record.get("id") if isinstance(record, Mapping) else None
                if identifier not in records_by_id or identifier in seen:
                    errors.append("flora_batch_result_identity_invalid")
                    continue
                seen.add(identifier)
                records_by_id[identifier][mode] = record
        for request, job, input_digest in prepared:
            records = records_by_id[job["id"]]
            metadata = self._base_metadata(request, input_digest)
            metadata.update(raw_outputs=_safe_raw(records), runtime_execution=_safe_raw(result.get("executions", [])),
                            runtime_preflight=_safe_raw(result.get("preflight", {})),
                            batch_wall_time_seconds=result.get("wall_time_seconds"),
                            batch_episode_count=len(jobs))
            raw_path = root / (job["id"] + ".raw.json")
            raw_path.write_text(json.dumps(_safe_raw(records), sort_keys=True, allow_nan=False) + "\n")
            metadata["raw_artifacts"] = [{"path": str(raw_path), "sha256": file_digest(raw_path),
                "relative_path": raw_path.relative_to(self.output_root).as_posix(), "kind": "episode_raw"}]
            for execution in result.get("executions", []):
                for kind, path_key, digest_key in (("native_output", "output_path", "output_sha256"),
                                                  ("runtime_log", "log_path", "log_sha256")):
                    if path_key in execution and digest_key in execution:
                        metadata["raw_artifacts"].append({"path": execution[path_key], "sha256": execution[digest_key],
                            "relative_path": Path(execution[path_key]).relative_to(self.output_root).as_posix(),
                            "kind": kind, "mode": execution["mode"]})
            try:
                if errors:
                    _fail(str(errors[0]).split(":")[0])
                _, current_digest = self._input(request)
                if current_digest != input_digest:
                    _fail("flora_source_changed_during_inference")
                bindings = request.input_receipt["artifacts"]["review_videos"]
                if any(file_digest(Path(path)) != binding["sha256"]
                       for path, binding in zip(job["videos"], bindings)):
                    _fail("flora_staged_video_changed_during_inference")
                rewards = normalize_flora_rewards(records, job=job,
                    sample_count=self.profile["runtime_config"]["rubric_num_samples"])
                metadata.update(rewards=rewards, apparent_pass=all(r["apparent_pass"] for r in rewards.values()),
                                status="completed", learned_model_executed=self._live_runner,
                                inference_performed=self._live_runner,
                                inference_mode_count=len(result.get("executions", [])) if self._live_runner else 0)
                output = self._output(request, records, rewards)
            except (ValueError, KeyError, TypeError) as exc:
                reason = str(exc) if isinstance(exc, EpisodeInterpretationError) else "flora_output_schema_invalid"
                metadata.update(status="abstained", abstention_reason=reason,
                                errors=errors or [reason], rewards={}, apparent_pass=None,
                                learned_model_executed=self._live_runner and bool(records),
                                inference_performed=self._live_runner and bool(records),
                                inference_mode_count=len(records) if self._live_runner else 0)
                output = EpisodeInterpreterOutput(episode_outcome="unclear",
                    summary="FLORA visual interpretation abstained: " + reason,
                    confidence=0.0, possible_missed_events=[PossibleMissedEvent(
                        description="No complete validated set of terminal, dense and rubric rewards is available.",
                        reason=reason)])
            self._cache[request.input_receipt["input_bundle_digest"]] = (output, metadata)

    def _base_metadata(self, request: EpisodeInterpretationRequest, digest: str) -> dict[str, Any]:
        return {"schema_version": "flora_episode_interpretation_metadata.v1",
                "input_bundle_digest": request.input_receipt["input_bundle_digest"],
                "interpreter_identity_digest": canonical_digest(self.identity.__dict__),
                "model_input_digest": digest, "profile_digest": self.profile["profile_digest"],
                "source_revision": FLORA_REVISION, "source_manifest_digest": canonical_digest(SOURCE_SHA256),
                "model_provider_id": "nvidia", "compute_provider_id": self.profile.get("compute_provider_id", "local"),
                "research_only": True, "learned_interpretation_only": True,
                "ranking_or_promotion_effect": "none", "confidence_calibrated": False,
                "confidence_zero_means_unavailable": True,
                "reference_scores_disclosed": False, "candidate_identity_disclosed": False,
                "state_and_contact_traces_disclosed": False,
                "sampling": copy.deepcopy(SAMPLING),
                "generation_settings": {key: self.profile["runtime_config"][key]
                                        for key in ("seed", "rubric_num_samples", "tensor_parallel_size")},
                "runtime_kind": "live_subprocess" if self._live_runner else "hermetic_injected_runner",
                "inference_execution_kind": "real_model" if self._live_runner else "hermetic_test",
                "raw_output_root": str(self.output_root), "raw_artifact_schema_version": "flora_raw_artifacts.v1",
                "raw_artifacts": []}

    def _output(self, request: EpisodeInterpretationRequest, records: Mapping[str, Any],
                rewards: Mapping[str, Any]) -> EpisodeInterpreterOutput:
        refs = [InterpretationEvidenceRef(artifact_role="review_video", artifact_digest=row["sha256"],
                                         note="Sampled derived visual evidence; not physical ground truth.")
                for row in request.input_receipt["artifacts"]["review_videos"]]
        terminal = records["terminal"]
        all_pass = all(reward["apparent_pass"] for reward in rewards.values())
        outcome = "appears_complete" if all_pass else (
            "appears_incomplete" if not rewards["terminal"]["apparent_pass"]
            and not rewards["rubric"]["apparent_pass"] else "unclear")
        events = [InterpretedEpisodeEvent(event_type="visual_terminal_assessment",
            start_time_seconds=terminal["duration_s"], description=terminal["rationale"],
            evidence_refs=refs, confidence=0.0)]
        for segment in records["dense"]["segments"]:
            if segment["score"] < 3 and len(events) < 200:
                events.append(InterpretedEpisodeEvent(event_type="visual_" + segment["label"],
                    start_time_seconds=segment["start_s"], end_time_seconds=segment["end_s"],
                    description=segment["rationale"], evidence_refs=refs, confidence=0.0))
        return EpisodeInterpreterOutput(episode_outcome=outcome,
            summary=(f"Research-only FLORA visual assessment: terminal {terminal['score']}/5; "
                     f"rubric {rewards['rubric']['total']:.2f}/1; "
                     f"dense worst quality {rewards['dense']['worst_score']}/3. "
                     "Scores are uncalibrated and do not determine official pass/fail."),
            events=events, confidence=0.0,
            possible_missed_events=[PossibleMissedEvent(
                description="Subsampled video can omit brief events, hidden contacts and physical state.",
                reason="visual_sampling_and_occlusion", evidence_refs=refs)],
            contract_considerations=["Deterministic success and task constraints remain authoritative.",
                                     "Zero confidence denotes unavailable calibration, not an estimated probability.",
                                     "Dense labels use later frames and are intended for offline review."])


def build_flora_episode_interpreter(profile: Mapping[str, Any], *,
                                    output_root: str | Path | None = None,
                                    runner: Any = None) -> FloraEpisodeInterpreter:
    return FloraEpisodeInterpreter.from_profile(profile, output_root=output_root, runner=runner)
