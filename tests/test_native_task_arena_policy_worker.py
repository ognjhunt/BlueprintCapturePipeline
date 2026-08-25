

def test_persist_survives_values_json_cannot_encode() -> None:
    """A receipt that cannot be written destroys the diagnosis of a paid run.

    `_persist` is called from a `finally`. Without `default=str` a value json
    cannot encode raises *inside* the handler, replacing the real exception and
    leaving the run with no receipt at all. The construction and controls
    workers already pass `default=str`; this one did not.
    """

    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from blueprint_pipeline.native_task_arena_policy_worker import _persist

    class _Unencodable:
        def __repr__(self) -> str:
            return "<warp array>"

    with TemporaryDirectory() as directory:
        target = Path(directory) / "native_task_arena_policy_result.v1.json"
        _persist(target, {"status": "blocked", "stray": _Unencodable()})

        written = json.loads(target.read_text(encoding="utf-8"))

    assert written["status"] == "blocked"
    assert written["stray"] == "<warp array>"
    assert written["result_digest"].startswith("sha256:")


def test_policy_query_tracker_records_returned_response_before_later_failure() -> None:
    from blueprint_pipeline.native_task_arena_policy_worker import _PolicyQueryTracker

    class _Client:
        action_space = "joint_position"

        def infer(self, observation):
            assert observation == {"prompt": "open"}
            return [[0.0] * 8]

    tracker = _PolicyQueryTracker(_Client())

    assert tracker.candidate_policy_queried is False
    assert tracker.infer({"prompt": "open"}) == [[0.0] * 8]
    assert tracker.candidate_policy_queried is True
    assert tracker.action_space == "joint_position"


def test_policy_query_tracker_does_not_claim_failed_server_query() -> None:
    import pytest

    from blueprint_pipeline.native_task_arena_policy_worker import _PolicyQueryTracker

    class _Client:
        def infer(self, observation):
            del observation
            raise RuntimeError("server_query_failed")

    tracker = _PolicyQueryTracker(_Client())

    with pytest.raises(RuntimeError, match="server_query_failed"):
        tracker.infer({})
    assert tracker.candidate_policy_queried is False


def test_policy_query_tracker_preserves_completed_query_on_response_refusal() -> None:
    import pytest

    from blueprint_pipeline.native_task_arena_policy_worker import _PolicyQueryTracker

    class _Client:
        candidate_policy_queried = False

        def infer(self, observation):
            del observation
            self.candidate_policy_queried = True
            raise ValueError("response_refused")

    tracker = _PolicyQueryTracker(_Client())

    with pytest.raises(ValueError, match="response_refused"):
        tracker.infer({})
    assert tracker.candidate_policy_queried is True


def test_ready_policy_server_teardown_waits_for_exact_pid_after_sigterm(
    tmp_path,
) -> None:
    import json
    import signal

    from blueprint_pipeline.adp009d_policy_server_worker import (
        terminate_ready_server_from_receipt,
    )

    command = ["/policy/bin/python", "serve.py", "--port", "8000"]
    receipt = tmp_path / "server.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "ready",
                "candidate_id": "pi05_droid",
                "server_pid": 417,
                "command": command,
            }
        ),
        encoding="utf-8",
    )
    state = {"command": command}
    signals = []

    def signal_process(pid, sent_signal):
        assert pid == 417
        signals.append(sent_signal)
        state["command"] = None

    result = terminate_ready_server_from_receipt(
        receipt,
        command_reader=lambda pid: state["command"],
        signal_process=signal_process,
        monotonic=lambda: 0.0,
        sleep=lambda seconds: None,
    )

    assert signals == [signal.SIGTERM]
    assert result["exact_process_identity_verified"] is True
    assert result["policy_server_process_terminated"] is True
    assert result["termination_method"] == "sigterm"


def test_ready_policy_server_teardown_refuses_reused_pid(tmp_path) -> None:
    import json

    from blueprint_pipeline.adp009d_policy_server_worker import (
        terminate_ready_server_from_receipt,
    )

    receipt = tmp_path / "server.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "ready",
                "candidate_id": "groot_n17_droid",
                "server_pid": 418,
                "command": ["/policy/bin/python", "serve.py"],
            }
        ),
        encoding="utf-8",
    )
    signals = []

    result = terminate_ready_server_from_receipt(
        receipt,
        command_reader=lambda pid: ["/usr/bin/unrelated"],
        signal_process=lambda pid, sent_signal: signals.append((pid, sent_signal)),
    )

    assert signals == []
    assert result["policy_server_process_terminated"] is False
    assert result["blockers"] == ["policy_server_teardown_pid_identity_mismatch"]


def test_ready_policy_server_teardown_escalates_and_observes_sigkill(
    tmp_path,
) -> None:
    import json
    import signal

    from blueprint_pipeline.adp009d_policy_server_worker import (
        terminate_ready_server_from_receipt,
    )

    command = ["/policy/bin/python", "serve.py"]
    receipt = tmp_path / "server.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "ready",
                "candidate_id": "pi05_droid",
                "server_pid": 419,
                "command": command,
            }
        ),
        encoding="utf-8",
    )
    state = {"command": command}
    signals = []

    def signal_process(pid, sent_signal):
        assert pid == 419
        signals.append(sent_signal)
        if sent_signal == signal.SIGKILL:
            state["command"] = None

    result = terminate_ready_server_from_receipt(
        receipt,
        command_reader=lambda pid: state["command"],
        signal_process=signal_process,
        monotonic=lambda: 0.0,
        sleep=lambda seconds: None,
        terminate_timeout_seconds=0.0,
    )

    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert result["policy_server_process_terminated"] is True
    assert result["termination_method"] == "sigkill"


def test_terminal_result_binds_only_observed_policy_server_teardown(
    tmp_path, monkeypatch
) -> None:
    import json

    from blueprint_pipeline import adp009d_policy_server_worker as worker

    result_path = tmp_path / "native_task_arena_policy_result.v1.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": "native_task_arena_policy_result.v1",
                "status": "completed",
                "blockers": [],
                "scientific_outcome_admitted": True,
                "ranking_eligible": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        worker,
        "terminate_ready_server_from_receipt",
        lambda receipt_path: {
            "schema_version": "adp009d_policy_server_teardown.v1",
            "status": "blocked",
            "policy_server_process_terminated": False,
            "server_pid": 420,
            "exact_process_identity_verified": True,
            "termination_method": None,
            "blockers": ["policy_server_teardown_process_survived_sigkill"],
        },
    )

    assert (
        worker._seal_ready_server_teardown(
            receipt_path=tmp_path / "server.json",
            result_path=result_path,
            result_schema="native_task_arena_policy_result.v1",
        )
        == 1
    )
    sealed = json.loads(result_path.read_text(encoding="utf-8"))

    assert sealed["status"] == "blocked"
    assert sealed["scientific_outcome_admitted"] is False
    assert sealed["ranking_eligible"] is False
    assert sealed["teardown"]["policy_server_process_terminated"] is False
    assert "native_task_policy_server_teardown_unproven" in sealed["blockers"]
    assert sealed["result_digest"] == worker._canonical_digest(
        sealed, digest_field="result_digest"
    )


def test_groot_episode_consumes_runtime_measured_worker_identity(tmp_path) -> None:
    """The immutable request may not impersonate a later runtime measurement."""

    import json

    from blueprint_pipeline.groot_n17_droid_policy_runtime import (
        GrootN17DroidPolicySpec,
    )
    from blueprint_pipeline.adp009d_groot_worker_identity import (
        expected_checkpoint_content_binding,
    )
    from blueprint_pipeline.native_task_arena_policy_worker import (
        GROOT_RUNTIME_IDENTITY_FILENAME,
        _runtime_groot_worker_identity,
    )

    policy_spec = GrootN17DroidPolicySpec()
    receipt = {
        "status": "verified",
        "model_id": policy_spec.model_id,
        "embodiment_tag": policy_spec.embodiment_tag,
        "groot_source_revision": policy_spec.groot_source_revision,
        "checkpoint_revision": policy_spec.checkpoint_revision,
        "checkpoint_files_sha256": "4" * 64,
        "checkpoint_content_manifest_digest": expected_checkpoint_content_binding()[
            "file_manifest_digest"
        ],
        "environment_lock_sha256": "5" * 64,
    }
    path = tmp_path / GROOT_RUNTIME_IDENTITY_FILENAME
    path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    observed, evidence = _runtime_groot_worker_identity(
        output_root=tmp_path,
        spec={
            "policy_spec": {
                "model_id": policy_spec.model_id,
                "embodiment_tag": policy_spec.embodiment_tag,
                "groot_source_revision": policy_spec.groot_source_revision,
                "checkpoint_revision": policy_spec.checkpoint_revision,
                "open_loop_horizon": policy_spec.open_loop_horizon,
            }
        },
    )

    assert observed == receipt
    assert evidence["source"] == "runtime_provisioning_measurement"
    assert evidence["relative_path"] == GROOT_RUNTIME_IDENTITY_FILENAME
    assert evidence["file_sha256"].startswith("sha256:")
    assert evidence["receipt_digest"].startswith("sha256:")


def test_groot_episode_refuses_missing_runtime_identity(tmp_path) -> None:
    from blueprint_pipeline.groot_n17_droid_policy_runtime import (
        GrootN17DroidPolicySpec,
    )
    from blueprint_pipeline.native_task_arena_policy_worker import (
        _runtime_groot_worker_identity,
    )
    import pytest

    policy_spec = GrootN17DroidPolicySpec()
    with pytest.raises(
        RuntimeError, match="groot_runtime_worker_identity_receipt_missing"
    ):
        _runtime_groot_worker_identity(
            output_root=tmp_path,
            spec={"policy_spec": policy_spec.__dict__},
        )


def _bundled_policy_inputs(tmp_path) -> dict[str, dict]:
    """Read back exactly what the policy worker reads on the provider.

    Packet, construction receipt, control result, execution spec and manifest
    all come from their real producers and are read out of a real bundle.
    """

    import json
    import zipfile

    from tests.test_native_task_arena_bundle import (
        _articulated_packet,
        _policy_spec,
        _qualified_construction,
        _qualified_controls,
        _runtime_source_packet,
    )
    from blueprint_pipeline.native_task_arena_policy_bundle import (
        build_native_task_arena_policy_bundle,
    )

    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    controls = _qualified_controls(tmp_path, scene, construction)
    bundle = build_native_task_arena_policy_bundle(
        job_dir=tmp_path / "policy-bundle",
        packet_dir=packet,
        construction_result_path=construction,
        control_result_path=controls,
        policy_execution_spec=_policy_spec(scene, construction, controls),
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        implementation_commit="d" * 40,
        generated_at="fixed",
    )
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(bundle["bundle_path"]) as archive:
        archive.extractall(extracted)
    runtime = extracted / "provider_runtime"
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))  # noqa: E731
    return {
        "manifest": read(runtime / "adp_arena_provider_manifest.json"),
        "spec": read(
            runtime
            / "runtime_inputs/native_task_arena_policy_execution_spec.v1.json"
        ),
        "construction": read(
            runtime
            / "runtime_inputs/native_task_arena_construction_result.v1.json"
        ),
        "controls": read(
            runtime / "runtime_inputs/native_task_arena_control_result.v1.json"
        ),
        "scene_plan": read(
            runtime / "native_task_packet/native_task_arena_scene_plan.v1.json"
        ),
    }


def test_real_producers_satisfy_every_policy_admission_relation(tmp_path) -> None:
    """The admission gate must be satisfiable by what its producers emit."""

    from blueprint_pipeline.native_task_arena_policy_worker import (
        _admission_binding_mismatches,
    )

    assert _admission_binding_mismatches(**_bundled_policy_inputs(tmp_path)) == []


def test_pi05_worker_accepts_the_checkpoint_inventory_required_by_its_bundle(
    tmp_path,
) -> None:
    """The provider worker and bundle verifier must require the same inputs."""

    import json
    import zipfile

    from blueprint_pipeline.native_task_arena_policy_bundle import (
        build_native_task_arena_policy_bundle,
    )
    from blueprint_pipeline.native_task_arena_policy_worker import _inputs
    from tests.test_native_task_arena_bundle import (
        _articulated_packet,
        _policy_spec,
        _qualified_construction,
        _qualified_controls,
        _runtime_source_packet,
    )

    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    controls = _qualified_controls(tmp_path, scene, construction)
    bundle = build_native_task_arena_policy_bundle(
        job_dir=tmp_path / "pi05-policy-bundle",
        packet_dir=packet,
        construction_result_path=construction,
        control_result_path=controls,
        policy_execution_spec=_policy_spec(scene, construction, controls),
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        implementation_commit="d" * 40,
        generated_at="fixed",
    )
    extracted = tmp_path / "pi05-extracted"
    with zipfile.ZipFile(bundle["bundle_path"]) as archive:
        archive.extractall(extracted)
    runtime = extracted / "provider_runtime"
    manifest = json.loads(
        (runtime / "adp_arena_provider_manifest.json").read_text(encoding="utf-8")
    )

    assert set(_inputs(runtime, manifest)) == {
        "adp009d_scene_840920_policy_readiness.v1.json",
        "third_scene_840920_task_a_scenario_suite.v1.json",
        "native_task_arena_construction_result.v1.json",
        "native_task_arena_control_result.v1.json",
        "native_task_arena_policy_execution_spec.v1.json",
        "openpi_polaris_checkpoint_inventory.json",
    }


def test_each_policy_admission_relation_reports_which_one_failed(
    tmp_path,
) -> None:
    """This gate stands in front of a paid run; a refusal must be readable."""

    from blueprint_pipeline.native_task_arena_policy_worker import (
        _admission_binding_mismatches,
    )

    inputs = _bundled_policy_inputs(tmp_path)
    other = "sha256:" + "0" * 64
    breakages = {
        "execution_spec_candidate_id_vs_manifest": (
            "manifest",
            "policy_candidate_id",
            "groot_n17_droid",
        ),
        "execution_spec_digest_vs_manifest": (
            "manifest",
            "policy_execution_spec_digest",
            other,
        ),
        "execution_authority_vs_manifest": (
            "manifest",
            "policy_execution_authority",
            "wrong_authority",
        ),
        "candidate_rights_binding_vs_manifest": (
            "manifest",
            "policy_rights_binding",
            {"rights_receipt_digest": other},
        ),
        "construction_result_digest_vs_execution_spec": (
            "construction",
            "result_digest",
            other,
        ),
        "control_result_digest_vs_execution_spec": (
            "controls",
            "result_digest",
            other,
        ),
        "scene_plan_digest_vs_execution_spec": (
            "scene_plan",
            "plan_digest",
            other,
        ),
        "construction_gate_qualified": (
            "construction",
            "construction_gate_qualified",
            False,
        ),
        "controls_qualified": ("controls", "controls_qualified", False),
    }
    for relation, (artifact, field, value) in breakages.items():
        broken = {key: dict(item) for key, item in inputs.items()}
        broken[artifact][field] = value
        assert set(_admission_binding_mismatches(**broken)) == {relation}, relation

    broken = {key: dict(item) for key, item in inputs.items()}
    diagnostic_authority = (
        "development_only_unqualified_controls_canonical_diagnostic"
    )
    broken["spec"]["execution_authority"] = diagnostic_authority
    broken["manifest"]["policy_execution_authority"] = diagnostic_authority
    assert set(_admission_binding_mismatches(**broken)) == {
        "qualified_execution_authority"
    }

    broken = {key: dict(item) for key, item in inputs.items()}
    broken["controls"]["control_pair"] = {
        **inputs["controls"]["control_pair"],
        "cell_admitted_for_policy_execution": False,
    }
    assert set(_admission_binding_mismatches(**broken)) == {
        "control_pair_cell_admitted_for_policy_execution"
    }

    broken = {key: dict(item) for key, item in inputs.items()}
    broken["spec"]["prompt"] = "Open a different appliance."
    assert set(_admission_binding_mismatches(**broken)) == {
        "execution_spec_prompt_vs_task_spec"
    }

    broken = {key: dict(item) for key, item in inputs.items()}
    broken["spec"]["max_policy_queries"] -= 1
    assert set(_admission_binding_mismatches(**broken)) == {
        "execution_spec_query_budget_vs_task_spec"
    }

    broken = {key: dict(item) for key, item in inputs.items()}
    broken["manifest"]["policy_rights_binding"] = {"binding_digest": "tampered"}
    assert set(_admission_binding_mismatches(**broken)) == {
        "candidate_rights_binding_vs_manifest"
    }


def test_policy_admission_refuses_two_absent_digests() -> None:
    """Absent digests are refusals, not agreements.

    Every digest relation here previously held vacuously as `None == None`
    whenever both artifacts lacked the field.
    """

    from blueprint_pipeline.native_task_arena_policy_worker import (
        _admission_binding_mismatches,
    )

    mismatches = _admission_binding_mismatches(
        manifest={}, spec={}, construction={}, controls={}, scene_plan={}
    )

    assert set(mismatches) == {
        "execution_spec_candidate_id_vs_manifest",
        "execution_spec_digest_vs_manifest",
        "execution_authority_vs_manifest",
        "candidate_rights_binding_vs_manifest",
        "construction_result_digest_vs_execution_spec",
        "control_result_digest_vs_execution_spec",
        "control_pair_digest_vs_execution_spec",
        "scene_plan_digest_vs_execution_spec",
        "construction_gate_qualified",
        "controls_qualified",
        "qualified_execution_authority",
        "control_pair_cell_admitted_for_policy_execution",
        "qualified_execution_authority",
        "candidate_rights_binding_vs_manifest",
        "execution_spec_prompt_vs_task_spec",
        "execution_spec_query_budget_vs_task_spec",
    }


def test_episode_progress_is_monotonic_and_phase_ordered() -> None:
    from blueprint_pipeline.native_task_arena_policy_worker import (
        _apply_episode_progress,
    )

    result = {
        "phase_reached": "policy_client_verified",
        "candidate_policy_queried": False,
    }
    _apply_episode_progress(
        result,
        {
            "phase": "policy_response_received",
            "candidate_policy_queried": True,
            "candidate_action_returned": True,
            "policy_inference_evidence": {
                "server_response_received": True,
                "raw_vendor_action_response_digest": "sha256:" + "a" * 64,
            },
        },
    )
    _apply_episode_progress(
        result,
        {
            "phase": "policy_action_bounds_refused",
            "candidate_action_shape_validated": True,
            "candidate_action_finite_validated": True,
            "candidate_action_bounds_validated": False,
        },
    )
    _apply_episode_progress(
        result,
        {
            "phase": "first_observation",
            "candidate_policy_queried": False,
            "candidate_action_returned": False,
        },
    )

    assert result["phase_reached"] == "policy_action_bounds_refused"
    assert result["candidate_policy_queried"] is True
    assert result["candidate_action_returned"] is True
    assert result["candidate_action_shape_validated"] is True
    assert result["candidate_action_finite_validated"] is True
    assert result["candidate_action_bounds_validated"] is False
    assert result["policy_inference_evidence"] == {
        "server_response_received": True,
        "raw_vendor_action_response_digest": "sha256:" + "a" * 64,
    }


def test_blocked_result_before_first_observation_retains_typed_media_gap(
    tmp_path,
) -> None:
    """A failure before the first observation must retain a typed media gap.

    The doctrine (AGENTS.md, enforced downstream by
    ``adp_prospective_design``) refuses completed-episode claims without media
    and refuses pre-observation failures without an explicit typed gap.  A
    bare blocked receipt is indistinguishable from lost media.
    """

    from blueprint_pipeline.native_task_arena_policy_worker import (
        _typed_media_gap_for_blocked_result,
    )

    result = {
        "status": "blocked",
        "blockers": ["native_task_policy_failed_at_start:RuntimeError:boom"],
        "phase_reached": "start",
        "candidate_policy_queried": False,
    }

    gap = _typed_media_gap_for_blocked_result(output_root=tmp_path, result=result)

    assert gap == {
        "status": "unavailable_before_first_observation",
        "media_gap": {
            "type": "before_first_observation",
            "reason": "native_task_policy_failed_at_start:RuntimeError:boom",
        },
    }


def test_arbitrary_media_bytes_do_not_refute_before_first_observation(tmp_path) -> None:
    """A loose PNG is not a retained, digest-bound first observation."""

    from blueprint_pipeline.native_task_arena_policy_worker import (
        _typed_media_gap_for_blocked_result,
    )

    frames = tmp_path / "episodes" / "e1" / "frames" / "external"
    frames.mkdir(parents=True)
    (frames / "000000-policy-input.png").write_bytes(b"\x89PNG")
    result = {
        "status": "blocked",
        "blockers": ["native_task_policy_failed_at_policy_client_verified:X:y"],
        "phase_reached": "policy_client_verified",
        "candidate_policy_queried": False,
    }

    gap = _typed_media_gap_for_blocked_result(output_root=tmp_path, result=result)
    assert gap is not None
    assert gap["status"] == "unavailable_before_first_observation"


def test_first_observation_without_sealed_media_gets_typed_partial_gap(
    tmp_path,
) -> None:
    from blueprint_pipeline.native_task_arena_policy_worker import (
        _typed_media_gap_for_blocked_result,
    )

    result = {
        "status": "blocked",
        "blockers": ["native_task_policy_failed_at_policy_response_received:X:y"],
        "phase_reached": "policy_response_received",
        "first_observation_retained": True,
        "candidate_policy_queried": True,
    }

    gap = _typed_media_gap_for_blocked_result(output_root=tmp_path, result=result)
    assert gap == {
        "status": "incomplete_after_first_observation",
        "media_gap": {
            "type": "after_first_observation_evidence_incomplete",
            "reason": "native_task_policy_failed_at_policy_response_received:X:y",
        },
    }


def test_detailed_post_observation_media_gap_preserves_retained_artifacts(
    tmp_path,
) -> None:
    from blueprint_pipeline.native_task_arena_policy_worker import (
        _typed_media_gap_for_blocked_result,
    )

    visual = {
        "status": "incomplete_after_first_observation",
        "episode_terminal_status": "failed_after_first_observation",
        "exact_policy_observation_retained": True,
        "multicamera_policy_observation_retained": False,
        "frame_manifest_digest": "sha256:" + "1" * 64,
        "video": {"relative_path": "media/e1/episode.mp4"},
        "media_gap": {
            "type": "after_first_observation_evidence_incomplete",
            "reason": "RuntimeError:native cameras failed",
        },
    }
    result = {
        "status": "blocked",
        "blockers": ["native cameras failed"],
        "phase_reached": "first_observation",
        "first_observation_retained": True,
        "visual_evidence": visual,
    }

    assert _typed_media_gap_for_blocked_result(
        output_root=tmp_path, result=result
    ) == visual


def test_sealed_failure_media_satisfies_post_observation_contract(tmp_path) -> None:
    from blueprint_pipeline.native_task_arena_policy_worker import (
        _typed_media_gap_for_blocked_result,
    )

    result = {
        "status": "blocked",
        "blockers": ["boom"],
        "phase_reached": "policy_response_received",
        "first_observation_retained": True,
        "candidate_policy_queried": True,
        "visual_evidence": {
            "status": "complete",
            "episode_terminal_status": "failed_after_first_observation",
        },
    }

    assert (
        _typed_media_gap_for_blocked_result(output_root=tmp_path, result=result)
        is None
    )


def test_media_gap_not_asserted_on_completed_result(tmp_path) -> None:
    from blueprint_pipeline.native_task_arena_policy_worker import (
        _typed_media_gap_for_blocked_result,
    )

    result = {
        "status": "completed",
        "blockers": [],
        "phase_reached": "episode_complete",
        "candidate_policy_queried": True,
        "episode": {"schema_version": "adp009d_policy_episode.v3"},
    }

    assert (
        _typed_media_gap_for_blocked_result(output_root=tmp_path, result=result)
        is None
    )


def test_media_gap_reason_falls_back_to_phase_when_blockers_empty(
    tmp_path,
) -> None:
    """The contract requires a non-empty reason string."""

    from blueprint_pipeline.native_task_arena_policy_worker import (
        _typed_media_gap_for_blocked_result,
    )

    result = {
        "status": "blocked",
        "blockers": [],
        "phase_reached": "inputs_verified",
        "candidate_policy_queried": False,
    }

    gap = _typed_media_gap_for_blocked_result(output_root=tmp_path, result=result)

    assert gap is not None
    assert gap["media_gap"]["reason"] == "failed_at_inputs_verified"
