"""ADP-050/day 28: retain native measurements and apply the independent scorer.

This object stays beside Blueprint's simulator. Neither policy responses nor
uploaded skill traces can supply its samples, frozen task spec or outcome.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .decision_evidence_contracts import canonical_digest


def validate_controlled_outcome(value: Mapping[str, Any], *, executed_motor_steps: int) -> dict[str, Any]:
    row = dict(value)
    if (row.get("schema_version") != "blueprint.controlled_policy_outcome.v1"
            or row.get("source") != "blueprint_native_simulator_readback"
            or type(row.get("task_success")) is not bool
            or type(row.get("sample_count")) is not int or row["sample_count"] < 2
            or row.get("executed_motor_steps") != executed_motor_steps or executed_motor_steps < 1
            or row.get("policy_supplied_outcome") is not False
            or row.get("physical_success_proven") is not False
            or not isinstance(row.get("score"), Mapping)
            or row["score"].get("task_succeeded") is not row["task_success"]
            or row["score"].get("status") != "scored"
            or row.get("receipt_digest") != canonical_digest(row, digest_field="receipt_digest")):
        raise ValueError("controlled_policy_independent_outcome_invalid")
    for field in ("task_spec_digest", "samples_digest"):
        digest = row.get(field)
        if not isinstance(digest, str) or len(digest) != 71 or not digest.startswith("sha256:"):
            raise ValueError("controlled_policy_independent_outcome_binding_invalid")
    return row


class NativeOutcomeRecorder:
    def __init__(self, *, task_spec: Mapping[str, Any], read_sample: Callable[[], Mapping[str, Any]],
                 evidence_path: Path):
        self.task_spec = json.loads(json.dumps(task_spec, allow_nan=False))
        self.read_sample = read_sample
        self.evidence_path = evidence_path
        self.samples: list[dict[str, Any]] = []
        self.steps = 0
        self.closed = False
        evidence_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.stream = evidence_path.open("x", encoding="utf-8")
        evidence_path.chmod(0o600)
        self.capture()

    def capture(self, *, motor_step: bool = False) -> None:
        if self.closed:
            raise ValueError("controlled_policy_native_measurements_closed")
        if motor_step:
            self.steps += 1
        row = json.loads(json.dumps(dict(self.read_sample()), allow_nan=False))
        row["step_index"] = self.steps
        self.samples.append(row)
        self.stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self.stream.flush()

    def finish(self, *, executed_motor_steps: int) -> dict[str, Any]:
        if self.closed or executed_motor_steps != self.steps or self.steps < 1:
            raise ValueError("controlled_policy_native_step_accounting_mismatch")
        from .adp_task_scoring import score_task_episode_from_spec
        try:
            score = score_task_episode_from_spec(task_spec=self.task_spec, samples=self.samples)
            if type(score.get("task_succeeded")) is not bool or score.get("status") != "scored":
                raise ValueError("controlled_policy_native_score_missing")
            row = {"schema_version": "blueprint.controlled_policy_outcome.v1",
                   "source": "blueprint_native_simulator_readback", "task_success": score["task_succeeded"],
                   "score": score, "executed_motor_steps": self.steps, "sample_count": len(self.samples),
                   "task_spec_digest": canonical_digest(self.task_spec), "samples_digest": canonical_digest({"samples": self.samples}),
                   "policy_supplied_outcome": False, "physical_success_proven": False,
                   "qualification_eligible": False, "evidence_scope": "development_only",
                   "samples_path": str(self.evidence_path)}
            row["receipt_digest"] = canonical_digest(row, digest_field="receipt_digest")
            return validate_controlled_outcome(row, executed_motor_steps=self.steps)
        finally:
            self.close()

    def close(self) -> None:
        if not self.closed:
            self.stream.close()
            self.closed = True
