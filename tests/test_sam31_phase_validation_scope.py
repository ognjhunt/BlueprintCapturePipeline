"""Production phases share pure measurements only until that phase returns."""
from __future__ import annotations

import json

import pytest

from blueprint_pipeline import sam31_source_calibration_stage as calibration
from blueprint_pipeline import task_evaluation_sam31_preparation_stages as stages
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.validation_file_digests import scoped_measurement


@pytest.mark.parametrize("first_phase_fails", [False, True])
def test_calibration_measurement_reuse_is_per_production_invocation(tmp_path, monkeypatch, first_phase_fails):
    # The hardware-phase dispatcher is real. Replace its expensive render/CPU
    # body with a pure measurement so this exercises the production scope
    # without a GPU, an allocator, or a second copy of USD validation logic.
    commit = "a" * 40
    profile = {"schema_version": stages.PROFILE_SCHEMA, "source_commit": commit,
               "calibrated_views": {"hardware_required": True}}
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    monkeypatch.setattr(stages, "resolve_sam31_profile", lambda _: profile_path)
    job = {"phase": "calibrated_views", "plan": {"source_commit": commit},
           "expected_source_commit": commit, "output_root": str(tmp_path), "job_digest": "sha256:" + "b" * 64}
    calls, measured = [], []

    def measure():
        calls.append(1)
        return {"measurement": len(calls)}

    def calibration_body(context):
        assert context["stage_id"] == "calibrated_views"
        first = scoped_measurement("same-current-inputs", measure)
        second = scoped_measurement("same-current-inputs", measure)
        measured.append((first["measurement"], second["measurement"]))
        if first_phase_fails and len(measured) == 1:
            raise ValueError("retained evidence refused")
        return {"status": "waiting_for_external_result", "artifacts": {}}

    monkeypatch.setattr(calibration, "execute_source_calibration_stage", calibration_body)
    if first_phase_fails:
        with pytest.raises(ValueError, match="retained evidence refused"):
            stages.execute_stage(job)
    else:
        stages.execute_stage(job)
    stages.execute_stage(job)
    assert measured == [(1, 1), (2, 2)]
    assert scoped_measurement("same-current-inputs", measure) == {"measurement": 3}
