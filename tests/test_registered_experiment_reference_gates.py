"""Unsupported publishers refuse decoded registered paths before persistence."""

import importlib
import pytest

TARGET = "/mnt/blueprint-work/lanes/g1/registered-" + "a" * 32
VALUE = {"nested": [{"path": TARGET + "/payload.bin"}]}
CASES = [
    (
        "task_evaluation_launch_preparation_queue",
        "write_launch_preparation_record_exclusive",
        {"path": None, "value": VALUE},
    ),
    (
        "task_evaluation_launch_preparation_queue",
        "stage_launch_preparation_request",
        {"value": VALUE, "queue_root": "/unused", "submitted_by": "owner"},
    ),
    (
        "task_evaluation_launch_activation_queue",
        "stage_launch_activation_request",
        {"value": VALUE, "queue_root": "/unused", "submitted_by": "owner"},
    ),
    (
        "task_evaluation_scene_intake",
        "stage_scene_intent",
        {
            "value": VALUE,
            "queue_root": "/unused",
            "authenticated_client": "owner",
            "trusted_clients": {"owner"},
        },
    ),
    (
        "task_evaluation_scene_construction_queue",
        "stage_scene_construction",
        {
            "request": VALUE,
            "preparation_result": {},
            "recipe": {},
            "recipe_configuration_references": (),
            "render_inputs_result": {},
            "queue_root": "/unused",
        },
    ),
    (
        "task_evaluation_scene_construction_queue",
        "stage_scene_configuration_revision",
        {
            "queue_root": "/unused",
            "source_envelope": VALUE,
            "expected_production_commit": "a" * 40,
            "revision_id": "revision",
            "semantic_checkpoint_digest": "sha256:" + "b" * 64,
        },
    ),
    (
        "task_evaluation_episode_compilation_queue",
        "stage_episode_compilation",
        {
            "request": VALUE,
            "preparation_result": {},
            "configured_revision": {},
            "configured_scene_bundle_reference": {},
            "queue_root": "/unused",
        },
    ),
    (
        "task_evaluation_launch_dispatcher",
        "stage_launch_request",
        {"value": VALUE, "queue_root": "/unused"},
    ),
    (
        "task_evaluation_policy_canary_handoff_state",
        "write_immutable",
        {"path": None, "value": VALUE},
    ),
    ("task_evaluation_policy_canary_handoff_state", "seal_state", {"path": None, "value": VALUE}),
    (
        "task_evaluation_sam31_preparation_queue",
        "advance_sam31_for_preparation",
        {"queue_root": None, "envelope_context": VALUE, "approved_roots": ()},
    ),
]


@pytest.mark.parametrize("module,name,kwargs", CASES)
def test_actual_queue_publisher_refuses_enrolled_reference_before_disk(
    module, name, kwargs, monkeypatch
):
    from pathlib import Path

    monkeypatch.setattr(
        Path, "mkdir", lambda *a, **kw: pytest.fail("unsupported persistent consumer touched disk")
    )
    code = importlib.import_module("blueprint_pipeline." + module)
    with pytest.raises(ValueError, match="experiment_external_publisher_unsupported"):
        getattr(code, name)(**kwargs)


def test_pin_cannot_silently_publish_registered_path_without_current_lifetime(tmp_path):
    from blueprint_pipeline.control_plane_storage_pins import write_storage_pin

    with pytest.raises(ValueError, match="experiment_external_publisher_unsupported"):
        write_storage_pin(
            pins_root=tmp_path / "pins", kind="activation", owner_id="run", paths=[TARGET]
        )
    assert not (tmp_path / "pins").exists()


def test_decoded_escape_and_relative_registered_names_are_reserved():
    import json
    from blueprint_pipeline.control_plane_registered_reference_gate import (
        refuse_registered_references,
    )

    for value in (
        json.loads('"g1\\u002fregistered-' + "a" * 32 + '\\u002fpayload"'),
        "../g1/registered-" + "b" * 32,
    ):
        with pytest.raises(ValueError, match="experiment_external_publisher_unsupported"):
            refuse_registered_references({"reference": value})
    refuse_registered_references(
        {"path": "/work/g1/legacy-run/payload", "evidence": [1, True, None]}
    )


@pytest.mark.parametrize("spelling", ["direct", "dotdot", "file_uri"])
def test_native_immutable_queue_destination_is_gated_before_mutation(tmp_path, spelling):
    from pathlib import Path
    from blueprint_pipeline.task_evaluation_launch_preparation_queue import (
        write_launch_preparation_record_exclusive,
    )

    target = tmp_path / "lanes" / "g1" / ("registered-" + "a" * 32)
    target.mkdir(parents=True)
    path = target / "published.json"
    if spelling == "dotdot":
        (target.parent / "ordinary").mkdir()
        path = target.parent / "ordinary" / ".." / target.name / "published.json"
    if spelling == "file_uri":
        from blueprint_pipeline.control_plane_registered_reference_gate import (
            refuse_registered_references,
        )

        with pytest.raises(ValueError, match="experiment_external_publisher_unsupported"):
            refuse_registered_references({"uri": path.as_uri()})
        return
    with pytest.raises(ValueError, match="experiment_external_publisher_unsupported"):
        write_launch_preparation_record_exclusive(path=Path(path), value={"version": 1})
    assert not (target / "published.json").exists()


@pytest.mark.parametrize("module,name,kwargs", CASES[1:8])
def test_actual_native_queue_root_is_observed_before_validator_or_mutation(
    module, name, kwargs, monkeypatch
):
    from pathlib import Path

    kwargs = dict(kwargs)
    kwargs["queue_root"] = Path(TARGET) / "queue"
    for field in ("value", "request", "source_envelope"):
        if field in kwargs:
            kwargs[field] = {}
    code = importlib.import_module("blueprint_pipeline." + module)
    monkeypatch.setattr(Path, "mkdir", lambda *a, **kw: pytest.fail("queue root mutated"))
    with pytest.raises(ValueError, match="experiment_external_publisher_unsupported"):
        getattr(code, name)(**kwargs)


@pytest.mark.parametrize("payload", ["a" * 65537, list(range(10001))])
def test_publisher_respects_native_lower_raw_and_value_allowances(payload):
    from blueprint_pipeline.control_plane_registered_reference_gate import (
        refuse_registered_references,
    )

    with pytest.raises(ValueError, match="experiment_publisher_input_limit"):
        refuse_registered_references(payload)


def test_publisher_one_clock_covers_all_roots_before_growth(monkeypatch):
    from blueprint_pipeline import control_plane_registered_reference_gate as gate
    from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget

    clock = iter([0.0, 0.0, 6.0])
    monkeypatch.setattr(
        gate,
        "ReferenceCollectionBudget",
        lambda **kw: ReferenceCollectionBudget(monotonic=lambda: next(clock), **kw),
        raising=False,
    )
    with pytest.raises(ValueError, match="experiment_publisher_input_limit"):
        gate.refuse_registered_references({"one": ["safe"]}, {"two": ["safe"]})


def test_native_installed_uri_resolution_observes_actual_selected_source_before_payload(
    tmp_path, monkeypatch
):
    from blueprint_pipeline.task_evaluation_installed_source_bindings import (
        InstalledSource,
        InstalledSourceBindings,
    )
    from pathlib import Path

    source = InstalledSource(
        path=tmp_path / "g1" / ("registered-" + "a" * 32) / "payload",
        digest="sha256:" + "b" * 64,
        size_bytes=7,
        installation_receipt_digest="receipt",
        publisher_intake_sha256="intake",
    )
    source.path.parent.mkdir(parents=True)
    source.path.write_bytes(b"payload")
    mapping = InstalledSourceBindings({"https://publisher.example/pinned": source})
    monkeypatch.setattr(
        Path, "open", lambda *args, **kwargs: pytest.fail("resolved registered payload opened")
    )
    with pytest.raises(ValueError, match="experiment_external_publisher_unsupported"):
        mapping.resolve("https://publisher.example/pinned", source.digest, source.size_bytes)


@pytest.mark.parametrize("value", [float("inf"), 1 << 5000], ids=["nonfinite", "oversize_integer"])
def test_publisher_native_scalar_size_is_refused_before_encoding(value):
    from blueprint_pipeline.control_plane_registered_reference_gate import (
        refuse_registered_references,
    )

    with pytest.raises(ValueError, match="experiment_publisher_input_limit"):
        refuse_registered_references(value)


def test_publisher_overlong_native_path_is_rejected_before_string_growth(monkeypatch):
    from pathlib import Path
    from blueprint_pipeline.control_plane_registered_reference_gate import (
        refuse_registered_references,
    )

    selected = Path("/" + "x" * 5000)
    original = Path.__str__

    def guarded(path):
        if path is selected:
            pytest.fail("overlong native Path was expanded before bounds")
        return original(path)

    monkeypatch.setattr(Path, "__str__", guarded)
    with pytest.raises(ValueError, match="experiment_publisher_input_limit"):
        refuse_registered_references(selected)


@pytest.mark.parametrize("operation", ["write", "release"])
def test_actual_pin_storage_destination_cannot_select_registered_target(tmp_path, operation):
    from blueprint_pipeline.control_plane_storage_pins import write_storage_pin, release_storage_pin

    target = tmp_path / "g1" / ("registered-" + "a" * 32)
    target.mkdir(parents=True)
    with pytest.raises(ValueError, match="experiment_external_publisher_unsupported"):
        if operation == "write":
            write_storage_pin(
                pins_root=target, kind="activation", owner_id="run", paths=[tmp_path / "ordinary"]
            )
        else:
            release_storage_pin(pins_root=target, kind="activation", owner_id="run")
    assert list(target.iterdir()) == []


def test_native_launch_selected_profile_root_is_gated_before_request_payload(tmp_path, monkeypatch):
    from pathlib import Path
    from blueprint_pipeline.task_evaluation_launch_dispatcher import dispatch_launch_request

    target = tmp_path / "g1" / ("registered-" + "a" * 32)
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *a, **kw: pytest.fail("native dispatch read payload before path guard"),
    )
    with pytest.raises(ValueError, match="experiment_external_publisher_unsupported"):
        dispatch_launch_request(
            request_path=tmp_path / "ordinary.json",
            profile_dir=target,
            state_root=tmp_path / "state",
        )


def test_installed_source_cannot_copy_needed_cache_without_registered_use(tmp_path):
    import hashlib
    from blueprint_pipeline.task_evaluation_installed_source_bindings import (
        InstalledSource,
        InstalledSourceBindings,
    )

    path = tmp_path / "g1-checkpoint" / "cache" / "payload"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"payload")
    digest = "sha256:" + hashlib.sha256(b"payload").hexdigest()
    source = InstalledSource(
        path=path,
        digest=digest,
        size_bytes=7,
        installation_receipt_digest="receipt",
        publisher_intake_sha256="intake",
    )
    with pytest.raises(ValueError, match="experiment_external_publisher_unsupported"):
        InstalledSourceBindings({"https://publisher.example/pinned": source}).resolve(
            "https://publisher.example/pinned", digest, 7
        )


def test_actual_nested_queue_publishers_share_one_native_allowance(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_registered_reference_gate as gate
    from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
    from blueprint_pipeline.task_evaluation_launch_preparation_queue import (
        stage_launch_preparation_request,
    )
    from tests.test_task_evaluation_launch_preparation_contract import request

    budgets = []

    def create(**kw):
        budget = ReferenceCollectionBudget(**kw)
        budgets.append(budget)
        return budget

    monkeypatch.setattr(gate, "ReferenceCollectionBudget", create)
    result = stage_launch_preparation_request(
        value=request(), queue_root=tmp_path / "queue", submitted_by="webapp-service"
    )
    assert result["accepted"] is True
    assert len(budgets) == 1
    assert budgets[0].closed is True
    assert budgets[0].counts["values"] > 0


def test_actual_nested_queue_cannot_restart_deadline_before_publication(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_registered_reference_gate as gate
    from blueprint_pipeline import task_evaluation_launch_preparation_queue as queue
    from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
    from tests.test_task_evaluation_launch_preparation_contract import request

    clock = [0.0]
    budgets = []

    def create(**kw):
        budget = ReferenceCollectionBudget(monotonic=lambda: clock[0], **kw)
        budgets.append(budget)
        return budget

    original = queue.validate_launch_preparation_request

    def delayed(value):
        result = original(value)
        clock[0] = 6.0
        return result

    monkeypatch.setattr(gate, "ReferenceCollectionBudget", create)
    monkeypatch.setattr(queue, "validate_launch_preparation_request", delayed)
    with pytest.raises(ValueError, match="experiment_publisher_input_limit"):
        queue.stage_launch_preparation_request(
            value=request(), queue_root=tmp_path / "queue", submitted_by="webapp-service"
        )
    assert not (tmp_path / "queue").exists()
    assert not list(tmp_path.rglob("*.json"))
    assert len(budgets) == 1
    assert budgets[0].closed is True


def test_actual_scene_validation_cannot_publish_after_original_deadline(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_registered_reference_gate as gate
    from blueprint_pipeline import task_evaluation_scene_intake as intake
    from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
    from tests.test_task_evaluation_scene_intake import request

    clock = [0.0]
    budgets = []

    def create(**kw):
        value = ReferenceCollectionBudget(monotonic=lambda: clock[0], **kw)
        budgets.append(value)
        return value

    original = intake.validate_request

    def delayed(*args, **kwargs):
        value = original(*args, **kwargs)
        clock[0] = 6.0
        return value

    monkeypatch.setattr(gate, "ReferenceCollectionBudget", create)
    monkeypatch.setattr(intake, "validate_request", delayed)
    destination = tmp_path / "no-output"
    with pytest.raises(intake.SceneIntakeError, match="experiment_publisher_input_limit"):
        intake.stage_scene_intent(
            value=request(),
            queue_root=destination,
            authenticated_client="webapp-service",
            trusted_clients={"webapp-service"},
            now=1000,
        )
    assert not destination.exists()
    assert len(budgets) == 1 and budgets[0].closed


def test_actual_private_queue_encoding_cannot_open_after_original_deadline(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_registered_reference_gate as gate
    from blueprint_pipeline import task_evaluation_launch_preparation_queue as queue
    from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget

    clock = [0.0]
    monkeypatch.setattr(
        gate,
        "ReferenceCollectionBudget",
        lambda **kw: ReferenceCollectionBudget(monotonic=lambda: clock[0], **kw),
    )
    original = queue.json.dumps

    def delayed(*args, **kwargs):
        value = original(*args, **kwargs)
        clock[0] = 6.0
        return value

    monkeypatch.setattr(queue.json, "dumps", delayed)
    destination = tmp_path / "one" / "two" / "receipt.json"
    destination.parent.mkdir(parents=True)
    with pytest.raises(ValueError, match="experiment_publisher_input_limit"):
        queue.write_launch_preparation_record_exclusive(destination, {"version": 1})
    assert not destination.exists()
    assert not list(destination.parent.iterdir())
