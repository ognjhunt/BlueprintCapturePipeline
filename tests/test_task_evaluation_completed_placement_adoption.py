import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_completed_placement_adoption as adoption
from blueprint_pipeline import task_evaluation_configured_controls_autostart as auto
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


@pytest.mark.parametrize('evaluation_id', [None, 'team-eval-one'])
@pytest.mark.parametrize(
    'provenance_rebound,retained_lineage_rebound',
    [(False, False), (True, False), (True, True)],
)
def test_completed_placement_rebinds_native_plan_without_new_model_or_search(
    tmp_path, monkeypatch, evaluation_id, provenance_rebound, retained_lineage_rebound,
):
    scene = {"scene": "fixture"}
    task = {"task": "move-object"}
    trajectory = {"trajectory_digest": "sha256:" + "a" * 64}
    prior_task = task
    if provenance_rebound:
        trajectory = {"trajectory_digest": "sha256:" + "b" * 64}
        task = {**task, "trajectory_digest": trajectory["trajectory_digest"]}
        prior_task = {**task, "trajectory_digest": "sha256:" + "a" * 64}
    original_task = prior_task
    if retained_lineage_rebound:
        original_task = {**task, "trajectory_digest": "sha256:" + "c" * 64}
    revision = {"revision_digest": "sha256:" + "b" * 64}
    cameras = tmp_path / "old-cameras.json"
    cameras.write_text(json.dumps({"cameras": [{"pose": "fixed"}]}))
    universe = tmp_path / "old-universe.json"
    universe.write_text(json.dumps({"run_id": "retained-run"}))
    old = {
        "scene_binding_digest": canonical_digest(scene),
        "task_binding_digest": canonical_digest(prior_task),
        "trajectory_digest": prior_task.get("trajectory_digest", trajectory["trajectory_digest"]),
        "configured_scene_revision_digest": revision["revision_digest"],
        "native_construction_candidate_universe": {"path": str(universe)},
        "placement_agent_receipt_digest": "sha256:" + "c" * 64,
        "official_openai_cost_evidence": {"retained": True},
        "cpu_placement_checkpoint_binding_digest": "sha256:" + "d" * 64,
    }
    placement = {
        "accepted_pose": {"position_world_m": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1]},
        "accepted_candidate_id": "selected",
        "task_binding_digest": canonical_digest(original_task),
        "task_trajectory_digest": original_task.get("trajectory_digest"),
    }
    files = {
        k: {"digest": "sha256:" + "e" * 64}
        for k in (
            "robot_asset_usd_path",
            "robot_mount_interface_path",
            "scene_camera_calibration_path",
        )
    }
    source = {
        "result": old,
        "inventory": {},
        "placement": placement,
        "intent": {"artifact_inventory": files},
        "plan": {"cameras_path": str(cameras)},
    }
    packet = {"source_result": {"digest": "sha256:" + "f" * 64}}
    monkeypatch.setattr(adoption, "validate_adoption", lambda value: source)
    if provenance_rebound:
        from blueprint_pipeline import task_evaluation_robot_placement_trajectory as projector

        old_binding = tmp_path / "old-binding"
        old_plan = old_binding / "deferred-inputs" / "original" / "native_trajectory_plan.v1.json"
        old_plan.parent.mkdir(parents=True)
        old_plan.write_text(json.dumps({"adapter_digest": "old", "plan_digest": "old-plan", "phases": ["same"]}))
        new_plan = tmp_path / "new-native-plan.json"
        new_plan.write_text(json.dumps({"adapter_digest": "new", "plan_digest": "new-plan", "phases": ["same"]}))
        old["base_pose_candidate_path"] = str(old_binding / "candidate.json")
        monkeypatch.setattr(projector, "placement_trajectory_from_native_plan", lambda plan: {
            "trajectory_digest": "sha256:" + ("a" if plan["adapter_digest"] == "old" else "b") * 64,
        })
    seen = []

    def forbid(**kwargs):
        raise AssertionError("completed placement must not rerun")

    monkeypatch.setattr(auto, "run_robot_placement_cli", forbid)

    def native_universe(**kwargs):
        assert kwargs["run_id"] == "retained-run"
        seen.append("universe")
        return universe, {"inventory_digest": "sha256:" + "0" * 64, "candidates": [{}]}

    monkeypatch.setattr(auto, "_materialize_native_feedback_candidate_universe", native_universe)
    monkeypatch.setattr(auto, "_materialize_placement_aware_cameras", lambda **kwargs: cameras)

    def readiness(**kwargs):
        assert kwargs["placement_receipt"] is placement
        assert kwargs["task_binding"] == original_task
        Path(kwargs["output_path"]).write_text("{}")
        seen.append("readiness")

    def plan(**kwargs):
        assert set(kwargs["bindings"]["phases"]) == {"destination", "construction", "controls"}
        assert kwargs["expected_production_commit"] == "1" * 40
        seen.append("plan")
        return {"plan_path": str(tmp_path / "new-plan.json"), "plan_digest": "sha256:" + "2" * 64}

    intent = {
        "completed_placement_adoption": packet,
        "expected_production_commit": "1" * 40,
        "artifact_inventory": files,
        "placement": {"candidate_inventory_cap": 24},
        "intent_digest": "sha256:" + "3" * 64,
        "phases": {"destination": {}, "construction": {}, "controls": {}},
        "profile_dir": str(tmp_path),
        "submitted_by": "fixture",
    }
    if evaluation_id:
        binding = {'evaluation_run_id':evaluation_id, 'source_launch_id':'source',
            'source_profile_digest':'sha256:'+'4'*64,
            'configured_scene_revision_digest':revision['revision_digest'], 'scene_intent_digest':'sha256:'+'5'*64}
        intent.update(evaluation_run_id=evaluation_id, evaluation_authority=binding)
        source['intent'].update(evaluation_run_id=evaluation_id, evaluation_authority=binding)
    paths = {
        k: str(tmp_path / k)
        for k in (
            "cameras_path",
            "robot_mount_interface_path",
            "scene_camera_calibration_path",
            "runtime_binding_path",
        )
    }
    if provenance_rebound:
        paths["native_trajectory_plan_path"] = str(new_plan)
    kwargs = dict(
        intent=intent,
        root=tmp_path,
        source_launch_id="source",
        launch_root=tmp_path,
        paths=paths,
        revision=revision,
        scene_binding=scene,
        task_binding=task,
        trajectory=trajectory,
        plan_root=tmp_path,
        readiness_materializer=readiness,
        plan_materializer=plan,
    )
    result = adoption.materialize(**kwargs)
    assert result["placement_agent_receipt_digest"] == old["placement_agent_receipt_digest"]
    assert (
        result["cpu_placement_checkpoint_binding_digest"]
        == old["cpu_placement_checkpoint_binding_digest"]
    )
    assert result["official_openai_cost_evidence"] == old["official_openai_cost_evidence"]
    assert result["placement_calls_reexecuted"] is False and seen == [
        "universe",
        "readiness",
        "plan",
    ]
    if provenance_rebound:
        assert result["trajectory_digest"] == trajectory["trajectory_digest"]
        assert result["task_binding_digest"] == canonical_digest(task)
        assert result["trajectory_adapter_provenance_rebound"]["physical_plan_fields_identical"] is True
        new_plan.write_text(json.dumps({"adapter_digest": "new", "plan_digest": "new-plan", "phases": ["changed"]}))
        with pytest.raises(ValueError, match="scientific_binding_changed"):
            adoption.materialize(**{**kwargs, "root": tmp_path / "changed"})
    with pytest.raises(ValueError, match="scientific_binding_changed"):
        adoption.materialize(**{**kwargs, "task_binding": {"task": "different"}})
    if evaluation_id:
        with pytest.raises(ValueError, match='evaluation_authority_changed'):
            adoption.materialize(**{**kwargs, 'intent':{**intent, 'evaluation_run_id':'foreign'}})


@pytest.mark.parametrize('evaluation_id', [None, 'team-eval-one'])
def test_native_submission_prevents_budget_retirement(tmp_path, evaluation_id):
    config = {
        "scene_root": str(tmp_path / "owners"),
        "progression_root": str(tmp_path / "progression"),
        "launch_state_root": str(tmp_path / "launches"),
    }
    plan = {
        "source_launch_id": "source",
        "expected_production_commit": "a" * 40,
        "future_outputs": {"construction": {"expected_activation_id": "activation"}},
    }
    if evaluation_id:
        plan['evaluation_run_id'] = evaluation_id
    assert adoption.native_submission_absent(config=config, plan=plan)
    marker = (
        tmp_path
        / "progression"
        / "source"
        / adoption.progression_directory('a'*40, evaluation_id)
        / "construction_activation_progression.json"
    )
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    assert not adoption.native_submission_absent(config=config, plan=plan)
    marker.unlink()
    (tmp_path / "launches" / "activation-auto-launch").mkdir(parents=True)
    assert not adoption.native_submission_absent(config=config, plan=plan)


def test_repeated_adoption_uses_original_checkpoint_without_new_model_file(tmp_path, monkeypatch):
    original = tmp_path / "accepted-checkpoint.json"
    original.write_text('{"original_intent": "first-model-call"}\n')
    reference = adoption._file(original)
    inherited = {"source_agent_checkpoint": reference}
    validated = []
    monkeypatch.setattr(adoption, "validate_adoption", lambda value: validated.append(value))
    result = {"completed_placement_adoption": inherited, "placement_calls_reexecuted": False}
    assert (
        adoption.checkpoint_reference(result=result, binding=tmp_path, token="new-intent")
        == reference
    )
    assert validated == [inherited]
    assert not (tmp_path / "agent-placement-checkpoint-new-intent.v1.json").exists()
    original.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="reference_changed"):
        adoption.checkpoint_reference(result=result, binding=tmp_path, token="new-intent")


def test_retained_placement_provenance_accepts_identical_cached_plans_only(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_retained_controls_evidence as evidence
    from blueprint_pipeline import task_evaluation_robot_placement_trajectory as projector

    root = tmp_path / "binding"
    old = {"adapter_digest": "old", "phases": ["same"]}
    new = {"adapter_digest": "new", "phases": ["same"]}
    for plan in (old, new):
        plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")

    def write_plan(directory, plan):
        path = root / "deferred-inputs" / directory / "native_trajectory_plan.v1.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan))
        return path

    write_plan("old-a", old)
    write_plan("old-b", old)
    write_plan("new-a", new)
    duplicate = write_plan("new-b", new)
    monkeypatch.setattr(projector, "placement_trajectory_from_native_plan", lambda plan: {
        "trajectory_digest": "sha256:" + ("a" if plan["adapter_digest"] == "old" else "b") * 64,
    })
    previous = {"base_pose_candidate_path": str(root / "candidate.json"),
                "trajectory_digest": "sha256:" + "a" * 64}
    result = {"base_pose_candidate_path": str(root / "candidate.json"),
              "trajectory_digest": "sha256:" + "b" * 64,
              "trajectory_adapter_provenance_rebound": {
                  "source_plan_digest": old["plan_digest"],
                  "successor_plan_digest": new["plan_digest"],
                  "physical_plan_fields_identical": True,
              }}
    evidence._validated_trajectory_provenance_rebound(result=result, ancestor={"result": previous})

    changed = {**new, "phases": ["different"]}
    changed["plan_digest"] = new["plan_digest"]
    duplicate.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="trajectory_provenance_invalid"):
        evidence._validated_trajectory_provenance_rebound(result=result, ancestor={"result": previous})
    duplicate.write_text(json.dumps(new))
    changed["plan_digest"] = canonical_digest(changed, digest_field="plan_digest")
    result["trajectory_adapter_provenance_rebound"]["successor_plan_digest"] = changed["plan_digest"]
    with pytest.raises(ValueError, match="trajectory_provenance_invalid"):
        evidence._validated_trajectory_provenance_rebound(result=result, ancestor={"result": previous})


def test_retained_placement_finds_original_receipt_binding_in_linear_reads(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_retained_controls_evidence as evidence

    parent = None
    for name in ("original", "successor-one", "successor-two"):
        value = {"name": name}
        if parent is not None:
            value["completed_placement_adoption"] = {"source_result": parent}
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value))
        parent = evidence._file(path)

    reads = []
    original_reader = evidence._placement_ref

    def counted(ref):
        reads.append(ref["digest"])
        return original_reader(ref)

    monkeypatch.setattr(evidence, "_placement_ref", counted)
    previous, original = evidence._original_placement_result(
        {"source_result": parent}, {"digest": "sha256:" + "f" * 64}
    )
    assert previous["name"] == "successor-two"
    assert original["name"] == "original"
    assert len(reads) == 3


def test_legacy_checkpoint_alias_is_byte_identical_idempotent_and_never_overwrites(
    tmp_path, monkeypatch
):
    binding = tmp_path / "source-launch" / "cpu-robot-binding"
    binding.mkdir(parents=True)
    original = tmp_path / "accepted.json"
    original.write_text('{ "original_intent": "accepted-model-call" }\n')
    reference = adoption._file(original)
    inherited = {"source_agent_checkpoint": reference, "source_launch_id": "source-launch"}
    intent = {"intent_digest": "sha256:" + "a" * 64, "completed_placement_adoption": inherited}
    intent_path = tmp_path / "intent.json"
    intent_path.write_text(json.dumps(intent))
    result = {
        "completed_placement_adoption": inherited,
        "placement_calls_reexecuted": False,
        "scene_binding_digest": "scene",
        "task_binding_digest": "task",
        "cpu_placement_checkpoint_binding_digest": "cpu",
    }
    result_path = auto._autostart_result_path(root=binding, intent_digest=intent["intent_digest"])
    result_path.write_text(json.dumps(result))
    monkeypatch.setattr(auto, "validate_configured_controls_autostart_intent", lambda value: value)
    monkeypatch.setattr(auto, "_validate_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(adoption, "validate_adoption", lambda value: None)
    first = adoption.materialize_legacy_checkpoint_alias(
        intent_path=intent_path, binding_root=binding
    )
    assert Path(first["target"]["path"]).read_bytes() == original.read_bytes()
    assert first["target"]["digest"] == reference["digest"]
    assert first["placement_calls_reexecuted"] is False
    assert (
        adoption.materialize_legacy_checkpoint_alias(intent_path=intent_path, binding_root=binding)[
            "status"
        ]
        == "already_present"
    )
    target = Path(first["target"]["path"])
    target.chmod(0o640)
    target.write_text('{"unrelated": true}')
    with pytest.raises(ValueError, match="checkpoint_alias_conflict"):
        adoption.materialize_legacy_checkpoint_alias(intent_path=intent_path, binding_root=binding)
    assert target.read_text() == '{"unrelated": true}'


def _lineage_packet(tmp_path, name, parent=None, **changes):
    result = {'name':name}
    if parent is not None:
        result.update(completed_placement_adoption=parent, placement_calls_reexecuted=False)
    result.update(changes)
    path = tmp_path/(name+'.json')
    path.write_text(json.dumps(result))
    packet = {'source_result':adoption._file(path), 'source_launch_id':'source',
              'owner_intent_digest':'sha256:'+'a'*64,
              'source_agent_checkpoint':{'digest':'sha256:'+'b'*64,'path':'retained-checkpoint'}}
    packet['adoption_digest'] = canonical_digest(packet, digest_field='adoption_digest')
    return packet


def test_discovery_chooses_exact_descendant_after_multiple_release_adoptions(tmp_path):
    first = _lineage_packet(tmp_path, 'first')
    middle = _lineage_packet(tmp_path, 'middle', first)
    last = _lineage_packet(tmp_path, 'last', middle)
    for matches in ([first,last], [last,first], [first,middle,last,last]):
        assert adoption._latest_verified_descendant(matches) == last
    assert adoption._latest_verified_descendant([first,first]) == first
    assert adoption._latest_verified_descendant([]) is None


def test_cancelled_predecessor_rebinds_only_when_intermediate_release_has_no_plan(
    tmp_path, monkeypatch,
):
    from blueprint_pipeline import task_evaluation_controls_autoprovision as provisioner
    from blueprint_pipeline import task_evaluation_scene_intake as intake
    from blueprint_pipeline import task_evaluation_retained_controls_evidence as evidence

    intent_id = "scene-one"
    previous_commit = "1" * 40
    next_commit = "2" * 40
    controls = tmp_path / "controls" / "terminal-adoptions" / intent_id
    scene = tmp_path / "scenes" / intent_id
    scene.mkdir(parents=True)
    plans = tmp_path / "plans"
    plans.mkdir()
    retained = {
        "execution_commit": previous_commit,
        "source_plan": {"path": str(plans / "original.json")},
        "adoption_digest": "sha256:" + "a" * 64,
    }
    for name, attempt_id in (("original", "original"), ("intermediate", "intermediate")):
        directory = controls / name
        directory.mkdir(parents=True)
        authorization = directory / "authorization.json"
        authorization.write_text(json.dumps({
            "scene_owner_attempt": {"scene_attempt_binding": {"attempt_id": attempt_id}}
        }))
        intent_path = directory / "intent.json"
        intent_path.write_text(json.dumps({
            "intent_digest": "sha256:" + ("b" if name == "original" else "c") * 64,
            "evaluation_authority": None,
            "evaluation_run_id": None,
            "phases": {"construction": {"authorization_path": str(authorization)}},
            **({"completed_placement_adoption": retained} if name == "intermediate" else {}),
        }))
        (directory / "terminal_adoption_provisioning.json").write_text(json.dumps({
            "execution_source_commit": previous_commit if name == "intermediate" else "0" * 40,
            "provisioning": {"intent_path": str(intent_path)},
        }))
    monkeypatch.setattr(provisioner, "_sealed", lambda path, field: json.loads(Path(path).read_text()))
    monkeypatch.setattr(intake, "_read", lambda path, field: {"attempt_id": Path(path).stem})
    monkeypatch.setattr(evidence, "validated_cancellation", lambda directory, attempt: (
        {"completed_placement_adoption": retained}
        if attempt["attempt_id"] == "original" else None
    ))
    monkeypatch.setattr(auto, "validate_configured_controls_autostart_intent", lambda value: value)
    monkeypatch.setattr(adoption, "validate_adoption", lambda packet: {"plan": {"source_launch_id": "source"}})
    monkeypatch.setattr(adoption, "native_submission_absent", lambda **kwargs: True)
    config = {
        "controls_root": str(tmp_path / "controls"),
        "scene_root": str(tmp_path / "scenes"),
        "progression_root": str(tmp_path / "state"),
    }
    source = {"launch_id": "source", "evaluation_authority": None}
    rebound = adoption.discover(config=config, intent_id=intent_id, source=source,
        expected_commit=next_commit)
    assert rebound is not None and rebound["execution_commit"] == next_commit
    assert rebound["adoption_digest"] == canonical_digest(rebound, digest_field="adoption_digest")
    (controls / "intermediate" / "terminal_adoption_provisioning.json").unlink()
    assert adoption.discover(config=config, intent_id=intent_id, source=source,
        expected_commit=next_commit) == rebound
    binding = tmp_path / "state" / "source" / adoption.scoped_identity("cpu-robot-binding", None)
    binding.mkdir(parents=True)
    prior_result = binding / "task_evaluation_configured_controls_autostart.v3-prior.json"
    prior_result.write_text(json.dumps({
        "completed_placement_adoption": {"execution_commit": previous_commit},
    }))
    assert adoption.discover(config=config, intent_id=intent_id, source=source,
        expected_commit=next_commit) is None
    prior_result.unlink()
    (plans / "intermediate.json").write_text(json.dumps({
        "expected_production_commit": previous_commit, "source_launch_id": "source",
    }))
    assert adoption.discover(config=config, intent_id=intent_id, source=source,
        expected_commit=next_commit) is None


@pytest.mark.parametrize('defect', ['unrelated', 'branch', 'rerun', 'changed_checkpoint', 'changed_reference'])
def test_discovery_never_collapses_different_or_changed_placement_lineages(tmp_path, defect):
    first = _lineage_packet(tmp_path, 'first')
    second = _lineage_packet(tmp_path, 'second', first)
    if defect == 'unrelated':
        second = _lineage_packet(tmp_path, 'second', None)
    elif defect == 'branch':
        first = _lineage_packet(tmp_path, 'other-branch', first)
    elif defect == 'rerun':
        second = _lineage_packet(tmp_path, 'second', first, placement_calls_reexecuted=True)
    elif defect == 'changed_checkpoint':
        second['source_agent_checkpoint'] = {'digest':'sha256:'+'c'*64,'path':'different'}
        second['adoption_digest'] = canonical_digest(second, digest_field='adoption_digest')
    elif defect == 'changed_reference':
        Path(first['source_result']['path']).write_text('{}')
    with pytest.raises(ValueError, match='ambiguous_sources|lineage_changed|reference_changed'):
        adoption._latest_verified_descendant([first,second])
