# Covers (for impacted-test selection): src/blueprint_pipeline/control_plane_preparation_activation_references.py src/blueprint_pipeline/control_plane_queue_observation.py
"""Invocation-wide allocation bounds and cold pure compatibility."""
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from blueprint_pipeline import control_plane_queue_observation as primary
from tests.test_preparation_activation_reference_records import observe, record, subject
from tests.test_preparation_activation_reference_edges import preparation_set


@pytest.mark.parametrize("field", ["family", "queue_root", "role", "row_path"])
@pytest.mark.parametrize("value", [[], {}, None, 1])
def test_dataclass_primitive_errors_never_leak_membership_typeerror(field, value):
    with pytest.raises(subject.PreparationActivationReferenceError, match="reference_parameters_invalid"):
        subject.interpret_preparation_activation_references(
            [subject.ReferenceFamilyContract("preparation", "/queues/preparation")], [replace(record(), **{field: value})])


def forbidden(*args, **kwargs):
    raise AssertionError("work before global bound proof")


def test_global_lexical_cap_precedes_every_json_parse_hash_and_canonical_encoding(monkeypatch):
    first = record()
    second = replace(first, row_path=first.row_path.replace("materialized", "pending"), raw_bytes=b'{"many":[0,0,0,0,0,0,0,0]}')
    scanner = primary._Scan((), 0, lambda: 0.0, 5.0)
    scanner.preflight(first.raw_bytes.decode())
    monkeypatch.setattr(primary, "MAX_VALUES", scanner.values + 1)
    monkeypatch.setattr(subject.json, "loads", forbidden)
    monkeypatch.setattr(subject.hashlib, "sha256", forbidden)
    monkeypatch.setattr(subject.json.JSONEncoder, "iterencode", forbidden)
    result = observe(first, second)
    assert not result.records and not result.complete_supplied_supported_projection


@pytest.mark.parametrize("contract", [subject.ReferenceFamilyContract([], "/queues/preparation"),
                                     subject.ReferenceFamilyContract("preparation", [])])
def test_contract_primitive_errors_are_fixed_refusals(contract):
    with pytest.raises(subject.PreparationActivationReferenceError, match="reference_parameters_invalid"):
        subject.interpret_preparation_activation_references([contract], [])


def test_inclusive_raw_byte_and_record_caps_are_shared_between_families(monkeypatch):
    first, second = record(), record("activation", state="prepared")
    monkeypatch.setattr(subject, "MAX_RECORDS", 2)
    monkeypatch.setattr(subject, "MAX_RECORD_BYTES", max(len(first.raw_bytes), len(second.raw_bytes)))
    monkeypatch.setattr(subject, "MAX_TOTAL_BYTES", len(first.raw_bytes) + len(second.raw_bytes))
    assert len(observe(first, second).records) == 2
    monkeypatch.setattr(subject, "MAX_TOTAL_BYTES", len(first.raw_bytes) + len(second.raw_bytes) - 1)
    assert not observe(first, second).records


def test_root_length_refusal_precedes_slicing(monkeypatch):
    class SliceSpy(str):
        def __getitem__(self, key):
            raise AssertionError("slice before bound")
    monkeypatch.setattr(subject, "MAX_PATH_BYTES", 4)
    with pytest.raises(subject.PreparationActivationReferenceError):
        subject.interpret_preparation_activation_references([subject.ReferenceFamilyContract("preparation", SliceSpy("/too-long"))], [])


def test_invalid_later_decoded_string_precedes_any_hash_or_canonical_encoding(monkeypatch):
    first = record()
    second = replace(first, row_path=first.row_path.replace("materialized", "pending"), raw_bytes=b'{"bad":"\\ud800"}')
    calls = []
    original = subject.hashlib.sha256
    def hash_spy(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    original_parse = primary._Scan.parse
    def parse_spy(self, raw):
        if raw == second.raw_bytes:
            assert not calls
        return original_parse(self, raw)
    monkeypatch.setattr(subject.hashlib, "sha256", hash_spy)
    monkeypatch.setattr(primary._Scan, "parse", parse_spy)
    result = observe(first, second)
    assert len(result.records) == 2 and not result.complete_supplied_supported_projection


def test_output_cap_refuses_before_record_dataclass_or_asdict_allocation(monkeypatch):
    row = record()
    monkeypatch.setattr(subject, "MAX_OUTPUT_BYTES", 1)
    monkeypatch.setattr(subject, "RawReferenceProvenance", forbidden)
    monkeypatch.setattr(subject, "ReferenceRecordDisposition", forbidden)
    monkeypatch.setattr(subject, "asdict", forbidden)
    result = observe(row)
    assert result.blockers == ("reference_output_limit",)
    assert not result.records


def test_fact_output_charge_precedes_constructor(monkeypatch):
    rows = preparation_set()
    original = subject.ReferenceFact
    calls = []
    def spy(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(subject, "ReferenceFact", spy)
    monkeypatch.setattr(subject, "MAX_OUTPUT_BYTES", 3500)
    result = observe(*rows)
    assert not result.records and not result.local_path_protections
    assert "reference_output_limit" in result.blockers
    assert len(calls) < 12


@pytest.mark.parametrize("cap", ["MAX_RECORDS", "MAX_RECORD_BYTES", "MAX_TOTAL_BYTES", "MAX_FACTS"])
def test_global_bounds_return_fixed_empty_incomplete(monkeypatch, cap):
    rows = preparation_set()
    monkeypatch.setattr(subject, cap, 0)
    result = observe(*rows)
    assert not result.records and not result.local_path_protections
    assert not result.complete_supplied_supported_projection
    assert result.blockers


@pytest.mark.parametrize("clock", [lambda: float("nan"), lambda: True, lambda: float("inf")])
def test_invalid_clock_returns_fixed_empty_incomplete(clock):
    result = observe(record(), monotonic=clock)
    assert result.blockers == ("queue_clock_invalid",)
    assert not result.records


def test_global_deadline_is_not_reset_for_each_record():
    ticks = iter([0.0, 0.0, 0.0, 0.0, 1.0])
    result = observe(record(), monotonic=lambda: next(ticks, 2.0), time_budget_seconds=0.5)
    assert result.blockers == ("queue_deadline_exceeded",)
    assert not result.records


def test_mid_preflight_clock_expiry_happens_before_json_loads(monkeypatch):
    row = replace(record(), raw_bytes=json.dumps({"text": "x" * 3000}).encode())
    calls = iter([0.0] * 6 + [1.0])
    monkeypatch.setattr(subject.json, "loads", forbidden)
    result = observe(row, monotonic=lambda: next(calls, 1.0), time_budget_seconds=0.5)
    assert result.blockers == ("queue_deadline_exceeded",)


def test_canonical_size_proof_precedes_encoder(monkeypatch):
    row = record()
    original = primary._Scan.output_size
    proved = set()
    def proof(self, document):
        original(self, document)
        proved.add(id(document))
    original_encode = subject.json.JSONEncoder.iterencode
    def encode(self, document, *args, **kwargs):
        assert id(document) in proved
        return original_encode(self, document, *args, **kwargs)
    monkeypatch.setattr(primary._Scan, "output_size", proof)
    monkeypatch.setattr(subject.json.JSONEncoder, "iterencode", encode)
    observe(row)


@pytest.mark.parametrize("root", ["/queues/../x", "/" + "x" * 256, "/" + "/".join(["x"] * 65), "relative"])
def test_roots_are_lexical_and_bounded_before_component_allocation(root):
    with pytest.raises(subject.PreparationActivationReferenceError):
        subject.interpret_preparation_activation_references([subject.ReferenceFamilyContract("preparation", root)], [])


@pytest.mark.slow
def test_first_call_in_fresh_process_has_no_runtime_schema_env_or_filesystem_access():
    code = '''
import importlib.abc,sys
sys.path.insert(0,SOURCE_PATH)
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, *args):
  if fullname.startswith("blueprint_pipeline.") and fullname not in {"blueprint_pipeline.control_plane_queue_observation", "blueprint_pipeline.control_plane_preparation_activation_references"}:
   raise AssertionError(fullname)
sys.meta_path.insert(0,Guard())
from blueprint_pipeline.control_plane_preparation_activation_references import ReferenceFamilyContract, interpret_preparation_activation_references
import os,pathlib,builtins
def forbidden(*args,**kwargs): raise AssertionError("external access")
os.getenv=forbidden
os.stat=forbidden
os.open=forbidden
os.scandir=forbidden
os.environ.get=forbidden
builtins.open=forbidden
pathlib.Path.read_text=forbidden
pathlib.Path.resolve=forbidden
from blueprint_pipeline.control_plane_preparation_activation_references import RetainedReferenceRecord
r=interpret_preparation_activation_references([ReferenceFamilyContract("preparation","/queues/preparation")], [RetainedReferenceRecord("preparation","/queues/preparation","result","/queues/preparation/results/prep-"+"1"*64+".json",b"{}")])
assert r.mutations == 0 and not r.references_clear and not r.general_reference_inventory_complete
'''
    code = code.replace("SOURCE_PATH", repr(str(Path(__file__).resolve().parents[1] / "src")))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
