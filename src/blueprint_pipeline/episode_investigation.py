"""Read-only evidence tools for adaptive interpretation of a sealed episode."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .agent_execution.contracts import AgentExecutionError, AgentTool, ToolContext, canonical_json
from .agent_execution.evidence import ImageEvidence, ImageEvidenceCatalog
from .decision_evidence_contracts import canonical_digest
from .episode_interpretation import EpisodeInterpretationRequest


class EpisodeEvidenceTools:
    """The agent can inspect sealed evidence but cannot reach the scorer or policy."""

    def __init__(self, request: EpisodeInterpretationRequest, *, admitted_digests: frozenset[str],
                 descriptors_only: bool = False) -> None:
        self.root = request.evidence_root.resolve()
        self.receipt = json.loads(canonical_json(request.input_receipt))
        self._receipt_identity = canonical_json(self.receipt)
        if self.receipt.get("input_bundle_digest") != canonical_digest(
            self.receipt, digest_field="input_bundle_digest",
        ):
            raise AgentExecutionError("episode_investigation_input_receipt_invalid")
        self.episode_id = request.episode_id
        if self.receipt.get("episode_id") != self.episode_id:
            raise AgentExecutionError("episode_investigation_identity_mismatch")
        self.admitted_digests = frozenset(admitted_digests)
        self.records = self.receipt["artifacts"]
        self.descriptors_only = descriptors_only
        manifest = {} if descriptors_only else self._read("frame_manifest")
        self.required_cameras = tuple(manifest.get("required_camera_ids") or ())
        images = []
        for frame in self.records["lossless_frames"]:
            images.append(ImageEvidence(
                image_id=f"episode_frame_{frame['frame_index']}",
                path=self.root / frame["relative_path"], sha256=frame["sha256"],
                camera_id=frame.get("camera_id") or "unrecorded_camera",
                role=frame.get("kind") or "lossless_frame",
                time_seconds=frame.get("simulation_time_s"),
                observation_index=frame.get("source_frame_index"),
            ))
        self.images = ImageEvidenceCatalog(
            root=self.root, images=images, admitted_digests=self.admitted_digests,
            defer_path_validation=descriptors_only,
        )

    def _read(self, role: str) -> dict[str, Any]:
        if self.descriptors_only:
            raise AgentExecutionError("episode_descriptor_cannot_read_evidence")
        self._check_identity()
        if role not in {"task_success_contract", "deterministic_score", "state_trace",
                        "contact_force_trace", "frame_manifest"}:
            raise AgentExecutionError("episode_investigation_role_not_admitted")
        record = self.records[role]
        if record["sha256"] not in self.admitted_digests:
            raise AgentExecutionError("episode_investigation_disclosure_not_admitted")
        path = self.root / record["relative_path"]
        if not path.resolve().is_relative_to(self.root) or not path.is_file():
            raise AgentExecutionError("episode_investigation_artifact_path_invalid")
        if any(parent.is_symlink() for parent in (path, *path.parents)
               if parent != self.root and parent.is_relative_to(self.root)):
            raise AgentExecutionError("episode_investigation_artifact_symlink")
        with path.open("rb") as stream:
            raw = stream.read(64_000_001)
        if len(raw) > 64_000_000:
            raise AgentExecutionError("episode_investigation_artifact_size_limit")
        if "sha256:" + hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise AgentExecutionError("episode_investigation_artifact_changed")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise AgentExecutionError("episode_investigation_artifact_invalid")
        canonical_json(value)
        return value

    def _check_identity(self) -> None:
        if (canonical_json(self.receipt) != self._receipt_identity
                or canonical_json(self.records) != canonical_json(self.receipt["artifacts"])):
            raise AgentExecutionError("episode_investigation_receipt_mutated")

    def context(self) -> dict[str, Any]:
        score = self._read("deterministic_score")
        contract = self._read("task_success_contract")
        return {
            "schema_version": "episode_investigation_context.v1",
            "episode_id": self.episode_id,
            "input_bundle_digest": self.receipt["input_bundle_digest"],
            "task_success_contract": contract,
            "deterministic_score": score,
            "score_evidence_digest": self.records["deterministic_score"]["logical_digest"],
            "required_camera_ids": list(self.required_cameras),
            "frame_count": len(self.images.images),
            "frames_without_recorded_time": sum(
                image.time_seconds is None for image in self.images.images.values()
            ),
            "authority": "interpretation_only_scoring_unchanged",
        }

    def trace(self, role: str, *, start_step: int, end_step: int, limit: int = 100) -> dict[str, Any]:
        if role not in {"state_trace", "contact_force_trace"}:
            raise AgentExecutionError("episode_investigation_trace_role_invalid")
        if (type(start_step) is not int or type(end_step) is not int
                or start_step < 0 or end_step < start_step
                or type(limit) is not int or not 1 <= limit <= 200):
            raise AgentExecutionError("episode_investigation_trace_window_invalid")
        value = self._read(role)
        fields = ("task_state_samples", "joint_states") if role == "state_trace" else ("samples",)
        result = {}
        for field in fields:
            rows = value.get(field) or []
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise AgentExecutionError("episode_investigation_trace_rows_invalid")
            unindexed = [row for row in rows if type(row.get("step_index")) is not int]
            selected = [row for row in rows if type(row.get("step_index")) is int
                        and start_step <= row["step_index"] <= end_step]
            selected.sort(key=lambda row: row["step_index"])
            result[field] = {
                "rows": selected[:limit], "matching_row_count": len(selected),
                "truncated": len(selected) > limit, "unindexed_row_count": len(unindexed),
                "next_step": selected[limit]["step_index"] if len(selected) > limit else None,
            }
        return {
            "schema_version": "episode_investigation_trace_window.v1",
            "episode_id": self.episode_id, "artifact_role": role,
            "evidence_digest": self.records[role]["logical_digest"],
            "start_step": start_step, "end_step": end_step,
            "fields": result, "typed_gap": value.get("typed_gap"),
        }

    def interval(
        self, *, start_seconds: float, end_seconds: float, max_observations: int = 4,
    ) -> list[dict[str, Any]]:
        self._check_identity()
        if (any(type(value) not in (int, float) or not math.isfinite(value)
                for value in (start_seconds, end_seconds))
                or not 0 <= start_seconds <= end_seconds
                or type(max_observations) is not int or not 1 <= max_observations <= 8):
            raise AgentExecutionError("episode_investigation_frame_window_invalid")
        grouped: dict[float, list[ImageEvidence]] = {}
        for image in self.images.images.values():
            if image.time_seconds is not None and start_seconds <= image.time_seconds <= end_seconds:
                grouped.setdefault(image.time_seconds, []).append(image)
        times = sorted(grouped)
        selected_times = times
        if len(times) > max_observations:
            if max_observations == 1:
                selected_times = [times[len(times) // 2]]
            else:
                selected_times = [times[round(i * (len(times) - 1) / (max_observations - 1))]
                                  for i in range(max_observations)]
        groups = []
        content = []
        for moment in selected_times:
            images = sorted(grouped[moment], key=lambda image: (image.camera_id, image.image_id))
            camera_ids = [image.camera_id for image in images]
            groups.append({
                "time_seconds": moment, "image_ids": [image.image_id for image in images],
                "camera_ids": camera_ids,
                "missing_required_camera_ids": sorted(set(self.required_cameras) - set(camera_ids)),
                "duplicate_camera_records": len(set(camera_ids)) != len(camera_ids),
            })
            for image in images:
                content.extend(self.images.inspect(image.image_id))
        summary = {
            "schema_version": "episode_investigation_frame_window.v1",
            "episode_id": self.episode_id,
            "frame_manifest_digest": self.records["frame_manifest"]["logical_digest"],
            "grouping": "recorded_simulation_time",
            "requested_interval_seconds": [start_seconds, end_seconds],
            "matching_observation_count": len(times), "returned_observation_count": len(selected_times),
            "sampled": len(times) > len(selected_times), "groups": groups,
            "no_recorded_frames_in_interval": not times,
            "frames_without_recorded_time": sum(
                image.time_seconds is None for image in self.images.images.values()
            ),
        }
        return [{"type": "input_text", "text": canonical_json(summary)}, *content]

    def tools(self) -> tuple[AgentTool, ...]:
        self._check_identity()
        bundle = self.receipt["input_bundle_digest"]

        def check(context: ToolContext, roles: tuple[str, ...]) -> None:
            required = {self.records[role]["sha256"] for role in roles}
            if not required <= set(context.allowed_input_digests):
                raise AgentExecutionError("episode_investigation_task_disclosure_mismatch")

        def read_context(_args, context):
            check(context, ("task_success_contract", "deterministic_score", "frame_manifest"))
            return self.context()

        def trace(args, context):
            check(context, (args["role"],))
            return self.trace(args["role"], start_step=args["start_step"], end_step=args["end_step"],
                              limit=args["limit"])

        def interval(args, context):
            check(context, ("frame_manifest",))
            if not {image.sha256 for image in self.images.images.values()} <= set(context.allowed_input_digests):
                raise AgentExecutionError("episode_investigation_task_disclosure_mismatch")
            return self.interval(**args)

        return (*self.images.tools(),
            AgentTool(
                "read_episode_context", "1", "Read the frozen contract and sealed independent score. " + bundle,
                {"type": "object", "properties": {}, "additionalProperties": False},
                "read_only", read_context,
            ),
            AgentTool(
                "read_episode_trace", "1", "Read a bounded step interval from a sealed trace. " + bundle,
                {"type": "object", "properties": {
                    "role": {"type": "string", "enum": ["state_trace", "contact_force_trace"]},
                    "start_step": {"type": "integer", "minimum": 0},
                    "end_step": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                }, "required": ["role", "start_step", "end_step", "limit"], "additionalProperties": False},
                "read_only", trace,
            ),
            AgentTool(
                "inspect_episode_interval", "1",
                "Inspect all recorded cameras at selected timestamps in a time interval. "
                "Narrow sampled intervals to investigate transient events. " + bundle,
                {"type": "object", "properties": {
                    "start_seconds": {"type": "number", "minimum": 0},
                    "end_seconds": {"type": "number", "minimum": 0},
                    "max_observations": {"type": "integer", "minimum": 1, "maximum": 8},
                }, "required": ["start_seconds", "end_seconds", "max_observations"],
                    "additionalProperties": False},
                "read_only", interval,
            ),
        )
