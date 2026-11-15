# Covers (for impacted-test selection):
#   src/blueprint_pipeline/policy_canary_output_members.py
#   src/blueprint_pipeline/native_task_arena_vast.py
#   src/blueprint_pipeline/adp_isaac_lab_arena_vast.py
#   src/blueprint_pipeline/provider_output_member_view.py
"""The Quick-10 output delivery mode is resolved before any paid mutation (plan 15, PR C, 15.C1).

Unset or empty means auto (2026-09-30): stream only when promotion would accept the dedicated B2
store and the host holds a sealed needed-set measurement within the contract's budget; otherwise
download. The session records the effective mode and why as ``provider_output_delivery_resolution``.
"""

from __future__ import annotations

import ast
import inspect
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from blueprint_pipeline import adp_isaac_lab_arena_vast as arena
from blueprint_pipeline import policy_canary_output_members as members
from blueprint_pipeline import task_evaluation_configured_scene_object_store as scene_store
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.provider_output_member_index import build_member_index, validate_member_selection
from blueprint_pipeline.provider_output_member_view import CONTRACTS, POLICY_CANARY_CONTRACT
from blueprint_pipeline.task_evaluation_configured_scene_object_store import _ARTIFACT_STORE_FILE_ENV
from tests.provider_output_fixtures import (
    RangeStore,
    VirtualFile,
    quick10_production_shaped_archive,
    quick10_shaped_archive,
)

#: ``native_task_arena_policy_canary_session.PROVIDER_RESULT_FILENAME`` and ``EPISODES_PER_POLICY`` when the
#: canary left: the names and shape retained archives carry.
PROVIDER_RESULT_FILENAME = "native_task_arena_policy_canary_session_result.v1.json"
EPISODES_PER_POLICY = 10

SRC = Path(__file__).resolve().parents[1] / "src" / "blueprint_pipeline"
SMALL = {"cells": 10, "frames_per_camera": 2, "png_bytes": 8 * 1024, "mp4_bytes": 64 * 1024,
         "policy_request_bytes": 4 * 1024}
EXPECTED_BUCKET_ENV = "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_EXPECTED_BUCKET"
AUTO_STREAM = {"mode": "stream", "reason": "auto_needed_set_within_budget"}
AUTO_DOWNLOAD = {"mode": "download", "reason": "auto_artifact_store_not_configured"}
INVALID_STORE = {"mode": "download", "reason": "auto_artifact_store_invalid"}
UNMEASURED = {"mode": "download", "reason": "auto_needed_set_unmeasured"}
OVER_BUDGET = {"mode": "download", "reason": "auto_needed_set_over_budget"}
FOOTPRINT_ENV = "BLUEPRINT_CONTROL_PLANE_DISK_FOOTPRINT_POLICY_CANARY_OUTPUT_BYTES"
# Placeholder settings a real botocore client builds from without any network call.
B2_VALUES = {"access_key": "005placeholderkeyid", "secret_key": "K005placeholdersecret",
             "bucket": "blueprint-task-evaluation-artifacts-dev",
             "endpoint": "https://s3.us-east-005.backblazeb2.com", "region": "us-east-005"}


def _b2_store(root: Path, *, configured: bool = True) -> dict[str, str]:
    """The five dedicated B2 settings as promotion accepts them (private, non-empty UTF-8 files that
    build its client); none at all when not configured."""
    if not configured:
        return {}
    root.mkdir(parents=True, exist_ok=True)
    for key in _ARTIFACT_STORE_FILE_ENV:
        (root / key).write_text(B2_VALUES[key] + "\n", encoding="utf-8")
        (root / key).chmod(0o600)
    return {name: str(root / key) for key, name in _ARTIFACT_STORE_FILE_ENV.items()}


def _measurement(path: Path, *, materialized_bytes: int = 436_485_098, archive_members: int = 6_745,
                 drop: tuple[str, ...] = (), **fields) -> Path:
    """A sealed needed-set measurement of a Quick-10 at ``path``; ``fields`` override it and
    ``drop`` removes fields, each resealed so only the gate's own checks can refuse it."""
    record = members.seal_needed_set_measurement(
        contract=members.CONTRACT_VERSION, materialized_members=1_200, materialized_bytes=materialized_bytes,
        archive={"name": "vast_provider_runtime_output.zip", "size_bytes": 4_200_000_000,
                 "sha256": "sha256:" + "a" * 64, "members": archive_members},
        quick10_shape=members.QUICK10_SHAPE, measured_at="2026-09-30T00:00:00+00:00")
    if fields or drop:
        record = {key: value for key, value in {**record, **fields}.items() if key not in drop}
        record["record_digest"] = canonical_digest(record, digest_field="record_digest")
    members.write_needed_set_measurement(record, path)
    return path


def _set_delivery(monkeypatch, tmp_path: Path, setting: str | None, *, configured: bool,
                  measured: bool = False) -> None:
    """The host the session reads: the setting (None leaves it unset), the B2 store, and the
    needed-set measurement at the (redirected) fixed record path."""
    for name in (*_ARTIFACT_STORE_FILE_ENV.values(), EXPECTED_BUCKET_ENV):
        monkeypatch.delenv(name, raising=False)
    for name, value in _b2_store(tmp_path / "b2", configured=configured).items():
        monkeypatch.setenv(name, value)
    record = tmp_path / "policy-canary-output" / "needed-set-measurement.v1.json"
    monkeypatch.setattr(members, "MEASUREMENT_PATH", _measurement(record) if measured else record)
    if setting is None:
        monkeypatch.delenv(members.DELIVERY_ENV, raising=False)
    else:
        monkeypatch.setenv(members.DELIVERY_ENV, setting)


@pytest.mark.parametrize(("defect", "accepted"), [
    ("none", True), ("expected_bucket", True), ("symlink", False), ("mode_0644", False), ("empty", False),
    ("over_4_kib", False), ("not_utf8", False), ("wrong_bucket", False), ("four_of_five", False),
])
def test_configured_means_exactly_what_promotion_accepts(tmp_path, monkeypatch, defect, accepted):
    """Review: a store promotion's own reader refuses must never resolve auto, or pass an explicit
    ``stream``, into a paid run whose promotion then fails and holds the output. The oracle is
    promotion's own client (boto3 stubbed): configured means exactly what it accepts."""
    _set_delivery(monkeypatch, tmp_path, None, configured=True, measured=True)
    settings = tmp_path / "b2"
    secret, bucket = settings / "secret_key", settings / "bucket"
    if defect == "expected_bucket":
        monkeypatch.setenv(EXPECTED_BUCKET_ENV, B2_VALUES["bucket"])
    elif defect == "symlink":
        (tmp_path / "elsewhere").write_text("secret\n", encoding="utf-8")
        (tmp_path / "elsewhere").chmod(0o600)
        secret.unlink()
        secret.symlink_to(tmp_path / "elsewhere")
    elif defect == "mode_0644":
        secret.chmod(0o644)
    elif defect == "empty":
        secret.write_text("", encoding="utf-8")
    elif defect == "over_4_kib":
        secret.write_text("s" * 4097, encoding="utf-8")
    elif defect == "not_utf8":
        secret.write_bytes(b"\xff\xfe\xfd\n")
    elif defect == "wrong_bucket":
        monkeypatch.setenv(EXPECTED_BUCKET_ENV, "blueprint-task-evaluation-artifacts-prod")
        assert bucket.read_text(encoding="utf-8").strip() != "blueprint-task-evaluation-artifacts-prod"
    elif defect == "four_of_five":
        monkeypatch.delenv(_ARTIFACT_STORE_FILE_ENV["region"])
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda *_args, **_kwargs: object()))
    monkeypatch.setitem(sys.modules, "botocore", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "botocore.client", SimpleNamespace(Config=lambda **_kwargs: object()))
    try:
        scene_store._artifact_object_store_client()
        promotion_accepts = True
    except scene_store.TaskEvaluationConfiguredSceneObjectStoreError:
        promotion_accepts = False

    assert promotion_accepts is accepted
    assert members.artifact_store_configured() is accepted
    assert members.resolve_output_delivery().record() == (AUTO_STREAM if accepted else INVALID_STORE)


@pytest.mark.parametrize(("setting", "value"), [
    ("endpoint", "endpoint-value"), ("endpoint", "not a url"), ("region", "us east 005!"),
])
def test_the_gate_builds_the_client_exactly_as_promotion_does(tmp_path, monkeypatch, setting, value):
    """Review minor: botocore refuses a malformed endpoint (``Invalid endpoint``) or region
    (``InvalidRegionError``) only when it builds the client, so a gate that never built one passed
    stores promotion then refused after the run was paid. With real botocore (skipped where it is
    not installed), the gate refuses exactly the stores promotion's own client does."""
    pytest.importorskip("boto3")
    pytest.importorskip("botocore")
    _set_delivery(monkeypatch, tmp_path, None, configured=True, measured=True)
    scene_store._artifact_object_store_client()  # the placeholder store builds, without any network call
    assert members.resolve_output_delivery().record() == AUTO_STREAM
    (tmp_path / "b2" / setting).write_text(value + "\n", encoding="utf-8")

    with pytest.raises(ValueError):  # botocore's own refusal (InvalidRegionError is a ValueError)
        scene_store._artifact_object_store_client()
    assert members.artifact_store_configured() is False
    assert members.resolve_output_delivery().record() == INVALID_STORE


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


def test_forecast_hold_fits_dense_observations_inside_the_existing_role():
    """Reserve dense metadata before the run, with no role increase or later hold growth."""
    from blueprint_pipeline.control_plane_disk_ledger import ROLE_FOOTPRINT_BYTES

    contract = members.POLICY_CANARY_OUTPUT_CONTRACT
    forecast = contract.forecast_hold_bytes()

    assert forecast == contract.hold_bytes(
        needed_bytes=contract.needed_set_budget_bytes, member_count=members.FORECAST_MEMBER_COUNT,
        index_file_bytes=members.FORECAST_MEMBER_COUNT * members.FORECAST_INDEX_ROW_BYTES)
    assert members.FORECAST_MEMBER_COUNT >= 3 * 6_745 - 300 and members.FORECAST_INDEX_ROW_BYTES >= 2 * 484
    assert forecast == ROLE_FOOTPRINT_BYTES[members.OUTPUT_ROLE]
    # The retained dense-observation shape exceeds the old 640 MiB estimate.
    assert contract.admits(needed_bytes=981_095_202, member_count=9_940,
                           hold_cap_bytes=forecast)
    assert not contract.admits(needed_bytes=forecast + 1, member_count=9_940,
                               hold_cap_bytes=forecast)
    # Any run the contract admits (a needed set within budget, members and index rows within the
    # allowance) shrinks the forecast in place: its exact hold never needs growth.
    assert contract.hold_bytes(needed_bytes=contract.needed_set_budget_bytes, member_count=6_745,
                               index_file_bytes=3_265_281) <= forecast
    assert contract.hold_bytes(needed_bytes=436_485_098, member_count=members.FORECAST_MEMBER_COUNT,
                               index_file_bytes=members.FORECAST_MEMBER_COUNT * 484) <= forecast


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
    """The frozen pair resolves delivery; the team explicitly downloads and other callers use the default."""
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
    assert lane_calls >= 4
    # The policy-canary session runner and the team session left with the early trial (2026-11-14):
    # no caller asks the transport for stream delivery.
    assert streaming_calls == []
    # Only the contract module reads the flag: no other module holds its name as a value.
    readers = sorted(path.name for path, source in sources.items() if members.DELIVERY_ENV in source and any(
        isinstance(node, ast.Constant) and node.value == members.DELIVERY_ENV for node in ast.walk(ast.parse(source))))
    assert readers == ["policy_canary_output_members.py"]


@pytest.mark.parametrize(("setting", "configured", "measured", "mode", "reason"), [
    (None, True, True, "stream", "auto_needed_set_within_budget"),
    ("", True, True, "stream", "auto_needed_set_within_budget"),
    (None, True, False, "download", "auto_needed_set_unmeasured"),
    (None, False, True, "download", "auto_artifact_store_not_configured"),
    ("", False, False, "download", "auto_artifact_store_not_configured"),
    ("download", True, True, "download", "explicit"),
    ("download", False, False, "download", "explicit"),
    ("stream", True, True, "stream", "explicit"),
    # An explicit stream needs no measurement; the session, not resolution, refuses it without the store.
    ("stream", True, False, "stream", "explicit"),
    ("stream", False, False, "stream", "explicit"),
])
def test_resolve_output_delivery_is_auto_when_unset(tmp_path, setting, configured, measured, mode, reason):
    environ = _b2_store(tmp_path / "b2", configured=configured)
    if setting is not None:
        environ[members.DELIVERY_ENV] = setting
    record = tmp_path / "needed-set-measurement.v1.json"
    if measured:
        _measurement(record)

    resolved = members.resolve_output_delivery(environ, measurement_path=record)

    assert (resolved.mode, resolved.reason) == (mode, reason)
    assert resolved.record() == {"mode": mode, "reason": reason}
    assert resolved.mode in members.REASON_MODES[resolved.reason]


@pytest.mark.parametrize("value", ["Stream", "auto", " download"])
def test_resolve_output_delivery_refuses_a_value_that_is_not_a_mode(tmp_path, value):
    environ = {**_b2_store(tmp_path / "b2", configured=True), members.DELIVERY_ENV: value}

    with pytest.raises(members.PolicyCanaryOutputDeliveryError, match="^policy_canary_output_delivery_mode_invalid$"):
        members.resolve_output_delivery(environ, measurement_path=_measurement(tmp_path / "measured.json"))


@pytest.mark.parametrize(("case", "reason"), [
    ("within_budget", "auto_needed_set_within_budget"),
    ("at_the_budget", "auto_needed_set_within_budget"),
    ("absent", "auto_needed_set_unmeasured"),
    ("over_budget", "auto_needed_set_over_budget"),
    # More members than the forecast hold allows: the run would block after it paid, like an over-budget set.
    ("too_many_members", "auto_needed_set_over_budget"),
    ("stale_contract", "auto_needed_set_record_invalid"),
    ("stale_selection", "auto_needed_set_record_invalid"),
    # Review: a record not tied to a Quick-10 -- sealed without the shape facts, with a partial
    # shape, with a truthy stand-in for a fact, or with an empty needed set -- never streams.
    ("shape_missing", "auto_needed_set_record_invalid"),
    ("shape_partial", "auto_needed_set_record_invalid"),
    ("shape_not_boolean", "auto_needed_set_record_invalid"),
    ("needed_set_empty", "auto_needed_set_record_invalid"),
    ("tampered", "auto_needed_set_record_invalid"),
    ("unsealed_field", "auto_needed_set_record_invalid"),
    ("not_json", "auto_needed_set_record_invalid"),
    ("symlink", "auto_needed_set_record_invalid"),
    ("directory", "auto_needed_set_record_invalid"),
    ("oversized", "auto_needed_set_record_invalid"),
])
def test_auto_streams_only_on_a_sealed_measurement_within_budget(tmp_path, case, reason):
    """Review: deploying must not be the flip. With the store configured, auto streams only once the
    host holds a sealed, digest-bound needed-set measurement naming the current contract and
    selection version and fitting the contract's budget; anything else downloads and says why."""
    environ = _b2_store(tmp_path / "b2")
    record = tmp_path / "needed-set-measurement.v1.json"
    budget = members.POLICY_CANARY_OUTPUT_CONTRACT.needed_set_budget_bytes
    if case == "within_budget":
        _measurement(record)
    elif case == "at_the_budget":
        _measurement(record, materialized_bytes=budget)
    elif case == "over_budget":
        _measurement(record, materialized_bytes=budget + 1)
    elif case == "too_many_members":
        _measurement(record, archive_members=10 * members.FORECAST_MEMBER_COUNT)
    elif case == "stale_contract":
        _measurement(record, contract="policy_canary_output_member_contract.v0")
    elif case == "stale_selection":
        _measurement(record, selection_version="policy_canary_output_member_contract.v0")
    elif case == "shape_missing":
        _measurement(record, drop=("quick10_shape",))
    elif case == "shape_partial":
        _measurement(record, quick10_shape={**members.QUICK10_SHAPE, "cell_results": 9})
    elif case == "shape_not_boolean":
        _measurement(record, quick10_shape={**members.QUICK10_SHAPE, "aggregate": 1})
    elif case == "needed_set_empty":
        _measurement(record, materialized_bytes=0)
    elif case == "tampered":
        # An over-budget measurement edited to look within budget, without resealing.
        sealed = json.loads(_measurement(record, materialized_bytes=budget + 1).read_text(encoding="utf-8"))
        record.write_text(json.dumps({**sealed, "materialized_bytes": 1}), encoding="utf-8")
    elif case == "unsealed_field":
        sealed = json.loads(_measurement(record).read_text(encoding="utf-8"))
        record.write_text(json.dumps({**sealed, "operator_note": "trust me"}), encoding="utf-8")
    elif case == "not_json":
        record.write_text("needed set: small\n", encoding="utf-8")
    elif case == "symlink":
        record.symlink_to(_measurement(tmp_path / "elsewhere.json"))
    elif case == "directory":
        record.mkdir()
    elif case == "oversized":
        sealed = json.loads(_measurement(record).read_text(encoding="utf-8"))
        record.write_text(json.dumps(sealed) + " " * members.MEASUREMENT_MAXIMUM_BYTES, encoding="utf-8")

    resolved = members.resolve_output_delivery(environ, measurement_path=record)

    assert resolved.record() == {"mode": "stream" if reason == "auto_needed_set_within_budget" else "download",
                                 "reason": reason}
    assert members.needed_set_measurement_refusal(record) == (
        None if reason == "auto_needed_set_within_budget" else reason)
    # The owner's explicit override needs no record at all.
    override = members.resolve_output_delivery({**environ, members.DELIVERY_ENV: "stream"}, measurement_path=record)
    assert override.record() == {"mode": "stream", "reason": "explicit"}


def test_the_fixed_record_path_is_hot_evidence_beside_the_policy_canary_state():
    """The one host record auto reads: under the control plane's state tree, where the dispatcher
    (``blueprint``) reads it, and classified so no reclaim tool ever takes it."""
    from blueprint_pipeline.control_plane_storage_roots import classify_path

    path = members.MEASUREMENT_PATH
    assert path == Path(
        "/var/lib/blueprint/pipeline-control-plane/policy-canary-output/needed-set-measurement.v1.json")
    root = classify_path(str(path))
    assert root is not None and root.storage_class == "evidence_hot" and root.owner == "blueprint"
    assert root.path == "/var/lib/blueprint/pipeline-control-plane/policy-canary-output"


def test_the_quick10_shape_names_the_session_s_own_members():
    """The facts a measurement is tied to are the contract's own member paths: the aggregate result,
    one child result per cell (one cell per episode of each policy), and the first cell's static
    startup preflight, which also binds the streamed native inventory."""
    from blueprint_pipeline import arena_provider_output_streaming as streaming
    from tests.provider_output_fixtures import quick10_production_shape

    assert members.QUICK10_CELL_COUNT == EPISODES_PER_POLICY
    assert streaming.IDENTITY_DOCUMENT == members.STARTUP_PREFLIGHT_MEMBER
    assert members.QUICK10_SHAPE == {"aggregate": True, "cell_results": 10, "startup_preflight": True}
    names = [PROVIDER_RESULT_FILENAME, *(f"cell_runs/{cell:02d}/{PROVIDER_RESULT_FILENAME}" for cell in range(10)),
             "cell_runs/00/policy_canary_static_startup_preflight.v1.json"]
    assert members.quick10_shape(names) == members.QUICK10_SHAPE
    assert members.quick10_shape(quick10_production_shape()) == members.QUICK10_SHAPE
    assert members.quick10_shape(names[:-1]) == {**members.QUICK10_SHAPE, "startup_preflight": False}
    assert members.quick10_shape(names[1:]) == {**members.QUICK10_SHAPE, "aggregate": False}
    assert members.quick10_shape(names[:6]) == {"aggregate": True, "cell_results": 5, "startup_preflight": False}
    # A child result outside the ten cells, or nested deeper, is not one of them.
    assert members.quick10_shape(["cell_runs/10/" + PROVIDER_RESULT_FILENAME,
                                  "cell_runs/00/nested/" + PROVIDER_RESULT_FILENAME])["cell_results"] == 0


def test_auto_holds_the_needed_set_to_the_hold_the_session_can_take(tmp_path, monkeypatch):
    """Review nit: the session's hold before the run is the forecast capped at the
    ``policy_canary_output`` footprint, and an operator override below the forecast caps it
    further. A measurement that fits the forecast but not that hold downloads; an override that
    cannot be read never raises, and downloads too."""
    from blueprint_pipeline import arena_provider_output_streaming as streaming

    environ = _b2_store(tmp_path / "b2")
    record = _measurement(tmp_path / "needed-set-measurement.v1.json")
    monkeypatch.delenv(FOOTPRINT_ENV, raising=False)
    forecast = members.POLICY_CANARY_OUTPUT_CONTRACT.forecast_hold_bytes()
    assert members.forecast_hold_cap_bytes() == forecast
    assert members.resolve_output_delivery(environ, measurement_path=record).record() == AUTO_STREAM

    monkeypatch.setenv(FOOTPRINT_ENV, str(256 * 1024**2))
    assert members.forecast_hold_cap_bytes() == 256 * 1024**2 < forecast
    assert members.resolve_output_delivery(environ, measurement_path=record).record() == OVER_BUDGET
    # The session reserves exactly that capped hold before it consumes its authority.
    seen = {}
    monkeypatch.setattr(streaming, "reserve_control_plane_disk",
                        lambda *_args, **kwargs: seen.update(kwargs) or "hold")
    assert streaming.reserve_forecast_hold(job_dir=tmp_path) == "hold"
    assert seen["expected_bytes"] == members.forecast_hold_cap_bytes()

    monkeypatch.setenv(FOOTPRINT_ENV, "lots")
    assert members.resolve_output_delivery(environ, measurement_path=record).record() == OVER_BUDGET
