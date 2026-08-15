"""The Arena chain has to be orderable from the website, not just runnable.

Three probe kinds over one transport, and they are ordered: controls consumes
the construction result, policy consumes both. The allocator enforces that
itself, but only after a provider has been handed over -- so a profile that
omits a predecessor costs a paid allocation to discover. Refusing at build time
costs nothing.

All three had bundles, a transport, and an allocator branch, and no launch
profile.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_launch_dispatcher import TaskEvaluationLaunchError

pytestmark = pytest.mark.usefixtures(
    "_materialize_generated_manifest_publication_fixture"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
URI = f"https://raw.githubusercontent.com/example/repo/{COMMIT}/arena.json"


def _load():
    name = "build_native_task_arena_live_profile"
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = _load()


@pytest.fixture()
def lane(tmp_path: Path) -> dict:
    packet = tmp_path / "packet"
    packet.mkdir()
    sealed = packet / "native_task_arena_packet.zip"
    sealed.write_bytes(b"arena-packet")
    digest = "sha256:" + hashlib.sha256(sealed.read_bytes()).hexdigest()
    (packet / builder.PACKET_RECEIPT_NAME).write_text(
        json.dumps(
            {
                "status": "ready",
                "implementation_commit": COMMIT,
                "bundle_path": str(sealed),
                "bundle_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    source_packet = tmp_path / "runtime_source_packet.json"
    source_packet.write_text(json.dumps({"status": "ready"}), encoding="utf-8")
    construction = tmp_path / "construction_result.json"
    construction.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    control = tmp_path / "control_result.json"
    control.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    policy_spec = tmp_path / "policy_execution_spec.json"
    policy_spec.write_text(json.dumps({"candidates": []}), encoding="utf-8")
    return {
        "packet": packet,
        "source_packet": source_packet,
        "construction": construction,
        "control": control,
        "policy_spec": policy_spec,
    }


def _build(lane, link: str, **overrides):
    arguments = {
        "link": link,
        "packet_dir": lane["packet"],
        "runtime_source_packet_path": lane["source_packet"],
        "source_commit": overrides.pop("source_commit", COMMIT),
        "raw_manifest_uri": URI,
    }
    if link in {"controls", "policy"}:
        arguments["construction_result_path"] = lane["construction"]
    if link == "policy":
        arguments["control_result_path"] = lane["control"]
        arguments["policy_execution_spec_path"] = lane["policy_spec"]
    arguments.update(overrides)
    return builder.build_native_task_arena_live_profile(**arguments)


@pytest.mark.parametrize(
    "link,probe_kind",
    [
        ("construction", "native-task-arena-construction"),
        ("controls", "native-task-arena-controls"),
        ("policy", "native-task-arena-policy"),
    ],
)
def test_each_link_routes_its_own_probe_kind(lane, link: str, probe_kind: str) -> None:
    argv = _build(lane, link)["allocator"]["argv"]

    assert argv[argv.index("--probe-kind") + 1] == probe_kind
    assert "--native-task-arena-packet" in argv
    assert "--native-task-arena-runtime-source-packet" in argv


@pytest.mark.parametrize("link", ["construction", "controls", "policy"])
def test_the_shared_controls_are_present(lane, link: str) -> None:
    profile = _build(lane, link)

    assert profile["required_controls"]["provider_zero_required"] is True
    assert profile["required_controls"]["teardown_required"] is True
    assert profile["allocator"]["retry_cap"] == 0
    assert sorted(profile["terminal_contract"]["required_path_fields"]) == [
        "artifact_manifest_path",
        "teardown_manifest_path",
    ]


@pytest.mark.parametrize(
    "link,omitted",
    [
        ("controls", "construction_result_path"),
        ("policy", "control_result_path"),
        ("policy", "policy_execution_spec_path"),
    ],
)
def test_a_link_without_its_predecessor_is_refused_before_allocation(
    lane, link: str, omitted: str
) -> None:
    """The allocator names the same absence, but only after it has rented."""

    with pytest.raises(TaskEvaluationLaunchError) as excinfo:
        _build(lane, link, **{omitted: None})

    assert "native_task_arena_predecessor_required" in str(excinfo.value)


def test_construction_needs_no_predecessor(lane) -> None:
    """It is the head of the chain; demanding one would deadlock it."""

    assert _build(lane, "construction")["allocator"]["argv"]


@pytest.mark.parametrize("ttl", [900, 20_000], ids=["under-band", "over-band"])
def test_a_ttl_outside_the_allocator_band_is_refused_here(lane, ttl: int) -> None:
    with pytest.raises(TaskEvaluationLaunchError) as excinfo:
        _build(lane, "construction", hard_ttl_seconds=ttl)

    assert "hard_ttl_out_of_band" in str(excinfo.value)


def test_a_packet_from_another_commit_is_refused(lane) -> None:
    with pytest.raises(TaskEvaluationLaunchError) as excinfo:
        _build(lane, "construction", source_commit="b" * 40)

    assert "bundle_commit_not_source_commit" in str(excinfo.value)


def test_each_predecessor_result_is_pinned_by_digest(lane) -> None:
    """This link's verdict is only about the packet it actually consumed."""

    inputs = {row["name"]: row for row in _build(lane, "policy")["immutable_inputs"]}

    for name, path in (
        ("native_task_arena_construction_result", lane["construction"]),
        ("native_task_arena_control_result", lane["control"]),
    ):
        expected = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        assert inputs[name]["digest"] == expected


def test_the_allocator_still_demands_the_predecessors_this_builder_carries() -> None:
    """Guards the ordering itself, not just this profile's rendering."""

    source = (
        REPO_ROOT / "src" / "blueprint_pipeline" / "paid_resource_allocator.py"
    ).read_text(encoding="utf-8")

    assert "native_task_arena_construction_result" in source
    assert "native_task_arena_control_result" in source
    assert "native_task_arena_policy_execution_spec" in source
