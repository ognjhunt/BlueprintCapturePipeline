

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
    }
