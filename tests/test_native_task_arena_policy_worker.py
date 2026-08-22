

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


def test_groot_episode_consumes_runtime_measured_worker_identity(tmp_path) -> None:
    """The immutable request may not impersonate a later runtime measurement."""

    import json

    from blueprint_pipeline.groot_n17_droid_policy_runtime import (
        GrootN17DroidPolicySpec,
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
        "construction_result_digest_vs_execution_spec",
        "control_result_digest_vs_execution_spec",
        "control_pair_digest_vs_execution_spec",
        "scene_plan_digest_vs_execution_spec",
        "construction_gate_qualified",
        "controls_qualified",
        "control_pair_cell_admitted_for_policy_execution",
        "execution_spec_prompt_vs_task_spec",
        "execution_spec_query_budget_vs_task_spec",
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


def test_media_gap_not_asserted_when_episode_media_exists(tmp_path) -> None:
    """Retained frames refute 'before first observation'; assert no gap."""

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
