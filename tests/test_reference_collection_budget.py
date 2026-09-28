# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_reference_budget.py
#   src/blueprint_pipeline/control_plane_queue_observation.py
#   src/blueprint_pipeline/control_plane_queue_auxiliary_observation.py
#   src/blueprint_pipeline/control_plane_storage_pin_observation.py
#   src/blueprint_pipeline/control_plane_preparation_activation_references.py
"""Public shared budgets do not multiply clock, acquisition or allocation limits."""

import pytest

from blueprint_pipeline import control_plane_reference_budget as budgets
from blueprint_pipeline.control_plane_queue_observation import QueueRootContract, observe_queue_states, QueueObservationError
from blueprint_pipeline.control_plane_storage_pin_observation import observe_storage_pins, StoragePinObservationError
from blueprint_pipeline.control_plane_queue_auxiliary_observation import AuxiliaryQueueContract, observe_preparation_sam_auxiliaries, AuxiliaryQueueObservationError
from blueprint_pipeline.control_plane_preparation_activation_references import interpret_preparation_activation_references, PreparationActivationReferenceError
from tests.test_preparation_activation_reference_records import record
from blueprint_pipeline.control_plane_preparation_activation_references import ReferenceFamilyContract


def contracts():
    return [ReferenceFamilyContract("preparation", "/queues/preparation")]


def test_cross_call_directory_entries_never_reset(tmp_path, monkeypatch):
    root = tmp_path / "queue"
    (root / "pending").mkdir(parents=True)
    monkeypatch.setattr(budgets, "MAX_ENTRIES", 3)
    shared = budgets.ReferenceCollectionBudget(monotonic=lambda: 0)
    first = observe_queue_states([QueueRootContract(str(root), ("pending",))], observed_at_epoch=100, budget=shared)
    assert first.complete
    second = observe_queue_states([QueueRootContract(str(root), ("pending",))], observed_at_epoch=100, budget=shared)
    assert not second.complete and "reference_entries_limit" in second.blockers


def test_leaf_composition_uses_one_deadline_not_new_five_second_windows(tmp_path):
    root = tmp_path / "queue"
    (root / "pending").mkdir(parents=True)
    pins = tmp_path / "pins"
    pins.mkdir()
    time = [0.0]
    shared = budgets.ReferenceCollectionBudget(monotonic=lambda: time[0])
    assert observe_storage_pins(str(pins), observed_at_epoch=100, budget=shared).complete
    time[0] = 5.1
    result = observe_queue_states([QueueRootContract(str(root), ("pending",))], observed_at_epoch=100, budget=shared)
    assert not result.complete and "reference_deadline_exceeded" in result.blockers


@pytest.mark.parametrize("leaf,error", [
    (lambda root, **options: observe_queue_states([QueueRootContract(root, ("pending",))], observed_at_epoch=100, **options), QueueObservationError),
    (lambda root, **options: observe_storage_pins(root, observed_at_epoch=100, **options), StoragePinObservationError),
    (lambda root, **options: observe_preparation_sam_auxiliaries([AuxiliaryQueueContract("preparation", root)], observed_at_epoch=100, **options), AuxiliaryQueueObservationError),
    (lambda root, **options: interpret_preparation_activation_references(contracts(), [record()], **options), PreparationActivationReferenceError),
])
def test_conflicting_nondefault_clock_settings_are_fixed_api_refusal(tmp_path, leaf, error):
    with pytest.raises(error, match="parameters_invalid"):
        leaf(str(tmp_path), budget=budgets.ReferenceCollectionBudget(monotonic=lambda: 0), monotonic=lambda: 1)


def test_closed_or_exhausted_budget_never_reopens_into_complete_evidence(tmp_path):
    root = tmp_path / "pins"
    root.mkdir()
    shared = budgets.ReferenceCollectionBudget(monotonic=lambda: 0)
    shared.close()
    result = observe_storage_pins(str(root), observed_at_epoch=100, budget=shared)
    assert not result.complete and "reference_budget_closed" in result.blockers


def test_pure_interpreter_shared_byte_limit_precedes_parser_and_hash(monkeypatch):
    from blueprint_pipeline import control_plane_preparation_activation_references as interpreter
    shared = budgets.ReferenceCollectionBudget(monotonic=lambda: 0)
    shared.charge("raw_bytes", budgets.MAX_RAW_BYTES)
    retained = record()
    monkeypatch.setattr(interpreter.json, "loads", lambda *args, **kwargs: pytest.fail("parsed after exhaustion"))
    monkeypatch.setattr(interpreter.hashlib, "sha256", lambda *args, **kwargs: pytest.fail("hashed after exhaustion"))
    result = interpreter.interpret_preparation_activation_references(contracts(), [retained], budget=shared)
    assert not result.complete_supplied_supported_projection
    assert "reference_raw_bytes_limit" in result.blockers


def test_budget_has_no_filesystem_or_observation_operations():
    shared = budgets.ReferenceCollectionBudget(monotonic=lambda: 0)
    assert not any(hasattr(shared, name) for name in ("open", "read", "walk", "scandir", "stat"))


def test_shared_output_refusal_precedes_queue_dataclass_conversion(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_queue_observation as queues
    root = tmp_path / "queue"
    (root / "pending").mkdir(parents=True)
    (root / "pending" / "row.json").write_text('{"value":"tiny"}')
    monkeypatch.setattr(budgets, "MAX_OUTPUT_BYTES", 10)
    shared = budgets.ReferenceCollectionBudget(monotonic=lambda: 0)
    monkeypatch.setattr(queues, "asdict", lambda value: pytest.fail("converted after shared output exhaustion"))
    result = queues.observe_queue_states([QueueRootContract(str(root), ("pending",))], observed_at_epoch=100, budget=shared)
    assert not result.complete and "reference_output_bytes_limit" in result.blockers


def test_shared_pin_lexical_bound_precedes_parser(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_storage_pin_observation as pins
    root = tmp_path / "pins"
    (root / "preparation").mkdir(parents=True)
    (root / "preparation" / "owner.json").write_text('{"unknown":' + '[' * 3 + '0' + ']' * 3 + '}')
    monkeypatch.setattr(budgets, "MAX_DEPTH", 2)
    monkeypatch.setattr(pins.json, "loads", lambda *args, **kwargs: pytest.fail("parsed after lexical depth cap"))
    result = pins.observe_storage_pins(str(root), observed_at_epoch=100, budget=budgets.ReferenceCollectionBudget(monotonic=lambda: 0))
    assert not result.complete and "reference_depth_limit" in result.blockers


def test_shared_row_allowance_is_charged_across_interpretation_calls(monkeypatch):
    monkeypatch.setattr(budgets, "MAX_ROWS", 1)
    shared = budgets.ReferenceCollectionBudget(monotonic=lambda: 0)
    first = interpret_preparation_activation_references(contracts(), [record()], budget=shared)
    assert first.records
    second = interpret_preparation_activation_references(contracts(), [record()], budget=shared)
    assert not second.records and "reference_rows_limit" in second.blockers


def test_shared_preflight_checks_clock_mid_string_before_parser():
    calls = [0]
    def clock():
        calls[0] += 1
        return 0 if calls[0] < 3 else 6
    shared = budgets.ReferenceCollectionBudget(monotonic=clock)
    with pytest.raises(budgets.ReferenceCollectionBudgetError, match="deadline_exceeded"):
        shared.preflight('{"key":"' + 'x' * 2048 + '"}')


@pytest.mark.parametrize("clock", [lambda: float("nan"), lambda: True, lambda: "0"])
def test_shared_invalid_clock_is_empty_fixed_unknown(clock, tmp_path):
    root = tmp_path / "pins"
    root.mkdir()
    result = observe_storage_pins(str(root), observed_at_epoch=100, budget=budgets.ReferenceCollectionBudget(monotonic=clock))
    assert not result.complete and not result.rows and "reference_clock_invalid" in result.blockers


def test_shared_large_integer_measure_refuses_before_decimal_conversion():
    shared = budgets.ReferenceCollectionBudget(monotonic=lambda: 0)
    with pytest.raises(budgets.ReferenceCollectionBudgetError, match="output_bytes_limit"):
        shared.measure(10 ** 5000)


def test_public_state_cannot_reset_counters_limits_or_closed_budget():
    shared = budgets.ReferenceCollectionBudget(monotonic=lambda: 0)
    shared.charge("rows")
    with pytest.raises(TypeError):
        shared.counts["rows"] = 0
    with pytest.raises(TypeError):
        shared.limits["rows"] = 100000000
    shared.close()
    with pytest.raises(AttributeError):
        shared.closed = False


def test_discovered_auxiliary_groups_share_static_group_allowance(tmp_path, monkeypatch):
    from tests.test_queue_auxiliary_layouts import root_for, STEM
    root = root_for(tmp_path, "preparation")
    (root / "source-progress" / STEM).mkdir()
    monkeypatch.setattr(budgets, "MAX_GROUPS", 8)
    result = observe_preparation_sam_auxiliaries(
        [AuxiliaryQueueContract("preparation", str(root))], observed_at_epoch=100,
        budget=budgets.ReferenceCollectionBudget(monotonic=lambda: 0))
    assert not result.complete and "reference_groups_limit" in result.blockers


def test_auxiliary_directory_evidence_is_charged_before_typed_construction(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_queue_auxiliary_observation as auxiliary
    from tests.test_queue_auxiliary_layouts import root_for
    root = root_for(tmp_path, "sam")
    monkeypatch.setattr(budgets, "MAX_OUTPUT_BYTES", 80)
    monkeypatch.setattr(auxiliary, "ObservedAuxiliaryDirectory", lambda *args: pytest.fail("constructed unbudgeted directory evidence"))
    result = auxiliary.observe_preparation_sam_auxiliaries(
        [auxiliary.AuxiliaryQueueContract("sam", str(root))], observed_at_epoch=100,
        budget=budgets.ReferenceCollectionBudget(monotonic=lambda: 0))
    assert not result.complete and "reference_output_bytes_limit" in result.blockers


def test_public_deadline_state_cannot_reset_single_use_budget():
    shared = budgets.ReferenceCollectionBudget(monotonic=lambda: 0)
    shared.tick()
    for name in ("deadline", "last", "duration", "monotonic"):
        with pytest.raises(AttributeError):
            setattr(shared, name, None)
