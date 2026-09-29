# Covers (for impacted-test selection):
#   src/blueprint_pipeline/policy_canary_output_members.py
#   src/blueprint_pipeline/native_task_arena_vast.py
#   src/blueprint_pipeline/adp_isaac_lab_arena_vast.py
#   src/blueprint_pipeline/provider_output_member_view.py
"""The Quick-10 output delivery mode is resolved before any paid mutation (plan 15, PR C, 15.C1)."""

from __future__ import annotations

import ast
import inspect
import zipfile
from pathlib import Path, PurePosixPath

import pytest

from blueprint_pipeline import adp_isaac_lab_arena_vast as arena
from blueprint_pipeline import native_task_arena_vast as native
from blueprint_pipeline import policy_canary_output_members as members
from blueprint_pipeline.native_task_arena_policy_canary_session import (
    PROVIDER_RESULT_FILENAME,
    RESULT_SCHEMA_VERSION,
)
from blueprint_pipeline.provider_output_member_index import build_member_index, validate_member_selection
from blueprint_pipeline.provider_output_member_view import CONTRACTS, POLICY_CANARY_CONTRACT
from blueprint_pipeline.task_evaluation_configured_scene_object_store import _ARTIFACT_STORE_FILE_ENV
from blueprint_pipeline.task_evaluation_scene_execution_authority import proves_no_provider_allocation
from tests.provider_output_fixtures import (
    RangeStore,
    VirtualFile,
    quick10_production_shaped_archive,
    quick10_shaped_archive,
)
from tests.test_native_task_arena_paired_witness_staging import context as context  # noqa: F401

SRC = Path(__file__).resolve().parents[1] / "src" / "blueprint_pipeline"
SMALL = {"cells": 10, "frames_per_camera": 2, "png_bytes": 8 * 1024, "mp4_bytes": 64 * 1024,
         "policy_request_bytes": 4 * 1024}


def _run_session(context, tmp_path, monkeypatch, *, execute=True, lane=None):
    """The Quick-10 session up to its lane call, with the authority and bundle checks stubbed."""
    _, bundle_path, binding = context
    authority = {**binding, "hard_cap_usd": 4.0, "hard_ttl_seconds": 9000,
                 "resource_name": "blueprint-native-task-policy-canary-test"}
    bundle = {"implementation_commit": binding["implementation_commit"],
              "bundle_sha256": binding["provider_bundle_sha256"], "container_image": "immutable-image",
              "bundle_path": str(bundle_path), "bundle_size_bytes": bundle_path.stat().st_size}
    monkeypatch.setattr(native, "validate_policy_canary_session_authority", lambda value: value)
    monkeypatch.setattr(native, "validate_policy_canary_provider_bundle", lambda *_a, **_kw: bundle)
    monkeypatch.setattr(native, "_policy_provider_transfer_byte_budget", lambda _candidate: (100, 20))
    monkeypatch.setattr(native, "run_arena_native_control_vast", lane or (lambda **kwargs: kwargs))
    return native.run_native_task_arena_policy_canary_session_vast(
        job_dir=tmp_path / "allocator", prepared_bundle=bundle, session_authority=authority,
        paid_resource_admission_grant=None, execute=execute, hard_ttl_seconds=9000,
        provider_runtime_environment={"BLUEPRINT_ADP009D_CAMERA_RESOLUTION": "640x360"})


def _forbidden(*_args, **_kwargs):
    raise AssertionError("the session authority was consumed or the lane ran before the mode was resolved")


@pytest.mark.parametrize("value", ["STREAM", "stream ", "upload", "1"])
def test_invalid_delivery_mode_refuses_before_consumption_with_zero_mutations(context, tmp_path, monkeypatch, value):
    monkeypatch.setenv(members.DELIVERY_ENV, value)
    monkeypatch.setattr(native, "consume_session_authority_once", _forbidden)

    result = _run_session(context, tmp_path, monkeypatch, lane=_forbidden)

    assert result == {"schema_version": RESULT_SCHEMA_VERSION, "status": "blocked",
                      "provider_mutations_performed": 0, "provider_allocations_observed": 0, "retry_cap": 0,
                      "blockers": ["policy_canary_output_delivery_mode_invalid"]}
    assert not (tmp_path / "allocator" / "policy_canary_session_consumption.json").exists()
    # The dispatcher closes it as a run that allocated nothing.
    assert proves_no_provider_allocation(result)


def test_download_mode_forwards_exactly_today_s_lane_arguments(context, tmp_path, monkeypatch):
    monkeypatch.delenv(members.DELIVERY_ENV, raising=False)
    unset = _run_session(context, tmp_path / "unset", monkeypatch, execute=False)
    for value in ("", "download"):
        monkeypatch.setenv(members.DELIVERY_ENV, value)
        forwarded = _run_session(context, tmp_path / "unset", monkeypatch, execute=False)
        assert forwarded == unset
    # The lane's own defaults: nothing streams.
    assert (unset["provider_output_delivery"], unset["provider_output_member_contract"]) == ("download", None)
    defaults = inspect.signature(arena.run_arena_native_control_vast).parameters
    assert all(defaults[key].default == unset[key] for key in unset if key.startswith("provider_output_"))


def test_stream_refuses_before_consumption_without_the_b2_artifact_store(context, tmp_path, monkeypatch):
    """Review I4: the artifact client would otherwise borrow the staging store's credentials."""
    monkeypatch.setenv(members.DELIVERY_ENV, "stream")
    for name in _ARTIFACT_STORE_FILE_ENV.values():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(next(iter(_ARTIFACT_STORE_FILE_ENV.values())), str(tmp_path / "only-one-of-five"))
    monkeypatch.setattr(native, "consume_session_authority_once", _forbidden)

    result = _run_session(context, tmp_path, monkeypatch, lane=_forbidden)

    assert result["status"] == "blocked" and result["provider_mutations_performed"] == 0
    assert result["blockers"] == ["policy_canary_output_stream_artifact_store_not_configured"]


def test_contract_materializes_json_outside_policy_requests_only():
    shaped = quick10_shaped_archive(**SMALL)
    index = build_member_index(RangeStore(shaped.archive).reader(block_bytes=128 * 1024),
                               maximum_expanded_bytes=8 * shaped.archive.size)
    contract = members.POLICY_CANARY_OUTPUT_CONTRACT

    selection = contract.selection(index)

    rows = validate_member_selection(selection, index)
    chosen = [row["path"] for row in rows]
    children = [f"cell_runs/{cell:02d}/{PROVIDER_RESULT_FILENAME}" for cell in range(10)]
    assert selection["selection_version"] == members.CONTRACT_VERSION == POLICY_CANARY_CONTRACT
    assert chosen == sorted(chosen) and chosen, "rows come back in archive order"
    assert set(chosen) == {path for path in shaped.json_members if path not in children}
    assert not set(chosen) & set(shaped.bulk_members)
    # Review I7: the ten child results stay in the archive; the aggregate carries their receipts.
    assert all(path in {row["path"] for row in index["members"]} for path in children)
    assert not set(children) & set(chosen) and PROVIDER_RESULT_FILENAME in chosen
    for path in chosen:
        parts = PurePosixPath(path).parts
        assert path.endswith(".json") and "policy-requests" not in parts
    assert contract.needed_bytes(index) == sum(row["size"] for row in rows)
    assert members.CHILD_RESULT_NAME == PROVIDER_RESULT_FILENAME
    # The member view's read-only planner applies the very same rule.
    assert CONTRACTS[POLICY_CANARY_CONTRACT] is contract.needed


def test_needed_set_budget_covers_the_measured_quick10_shape():
    """The budget is sized from the rehearsal-derived, production-scaled Quick-10 shape."""
    archive = quick10_production_shaped_archive()
    with zipfile.ZipFile(VirtualFile(archive)) as zipped:
        infos = [info for info in zipped.infolist() if not info.is_dir()]
    contract = members.POLICY_CANARY_OUTPUT_CONTRACT
    needed = sum(info.file_size for info in infos if contract.needed(info.filename))
    children = sum(info.file_size for info in infos
                   if PurePosixPath(info.filename).parts[:1] == ("cell_runs",)
                   and PurePosixPath(info.filename).name == PROVIDER_RESULT_FILENAME)

    assert (len(infos), sum(info.filename.endswith(".mp4") for info in infos)) == (6_745, 120)
    assert archive.size > 4 * 1024**3  # ZIP64 offsets, like the 4.2 GB production archive's end
    # The measurement the budget is sized from: 436.5 MB (416 MiB); the children add 191.2 MB.
    assert (needed, children) == (436_485_098, 191_208_330)
    assert needed <= contract.needed_set_budget_bytes == members.NEEDED_SET_BUDGET_BYTES
    assert contract.needed_set_budget_bytes >= 1.5 * needed
    # The ingestion hold for the budget itself, with this shape's metadata, stays under 1 GiB.
    assert contract.hold_bytes(needed_bytes=contract.needed_set_budget_bytes, member_count=len(infos),
                               index_file_bytes=8 * 1024**2) < 1024**3


def _keyword_names(call: ast.Call, where) -> list[str]:
    """Every keyword a call passes, including the literal keys of a ``**{...}`` expansion."""
    names = []
    for keyword in call.keywords:
        if keyword.arg is not None:
            names.append(keyword.arg)
            continue
        literals = [node for node in ast.walk(keyword.value) if isinstance(node, ast.Dict)]
        assert literals, f"{where}: an opaque ** expansion could pass anything"
        names.extend(key.value for literal in literals for key in literal.keys
                     if isinstance(key, ast.Constant) and isinstance(key.value, str))
    return names


def test_other_arena_callers_never_stream():
    """Only the Quick-10 session passes a delivery mode; every other caller keeps the download default."""
    parameters = inspect.signature(arena.run_arena_native_control_vast).parameters
    assert parameters["provider_output_delivery"].default == "download"
    assert parameters["provider_output_member_contract"].default is None
    streaming_calls, lane_calls = [], 0
    sources = {path: path.read_text(encoding="utf-8") for path in sorted(SRC.rglob("*.py"))}
    for path, source in sources.items():
        if "run_arena_native_control_vast(" not in source:
            continue
        tree = ast.parse(source)
        enclosing = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                enclosing[child] = node if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else (
                    enclosing.get(node))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "run_arena_native_control_vast"):
                continue
            lane_calls += 1
            function = getattr(enclosing.get(node), "name", "<module>")
            if any(name.startswith("provider_output_") for name in _keyword_names(node, (path.name, function))):
                streaming_calls.append((path.name, function))
    assert lane_calls >= 10
    assert streaming_calls == [("native_task_arena_vast.py", "run_native_task_arena_policy_canary_session_vast")]
    # Only the contract module reads the flag: no other module holds its name as a value.
    readers = sorted(path.name for path, source in sources.items() if members.DELIVERY_ENV in source and any(
        isinstance(node, ast.Constant) and node.value == members.DELIVERY_ENV for node in ast.walk(ast.parse(source))))
    assert readers == ["policy_canary_output_members.py"]


def test_resolve_output_delivery_defaults_to_download():
    assert members.resolve_output_delivery({}) == "download"
    assert members.resolve_output_delivery({members.DELIVERY_ENV: ""}) == "download"
    assert members.resolve_output_delivery({members.DELIVERY_ENV: "download"}) == "download"
    assert members.resolve_output_delivery({members.DELIVERY_ENV: "stream"}) == "stream"
    with pytest.raises(members.PolicyCanaryOutputDeliveryError, match="^policy_canary_output_delivery_mode_invalid$"):
        members.resolve_output_delivery({members.DELIVERY_ENV: "Stream"})
