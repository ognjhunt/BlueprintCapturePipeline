"""Every paid lane must seal the terminal artifacts its profile asks it for.

A production launch profile's terminal contract reads two fields off the
allocator result, `artifact_manifest_path` and `teardown_manifest_path`. Eight
paid lanes emitted neither or only the second, so each of them ended
`allocator_terminal_artifact_missing:` regardless of what happened on the
provider -- "all controls passed" was unreachable by construction, and the
evidence they had retained sat on disk with nothing pointing at it.

Fixing that lane by lane leaves the next lane to rediscover it. This is the test
that makes the shared seal the default: a lane added tomorrow either calls it or
fails here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "blueprint_pipeline"

#: Modules that run a paid provider attempt and return an allocator result.
#: Adding a lane here is the point: it is cheaper than discovering at the paid
#: boundary that its evidence was never inventoried.
PAID_LANE_MODULES = (
    "adp009d_aura_native_vast.py",
    "adp009d_franka_vast.py",
    "adp009d_ovrtx_vast.py",
    "adp_aura_author_smoke_vast.py",
    "adp_aura_interiorgs_vast.py",
    "adp_content_agents_vast.py",
    "adp_gaussian_excision_vast.py",
    "adp_inpaint360_interiorgs_vast.py",
    "adp_joint_agent_vast.py",
    "adp_retained_scene_render_vast.py",
    "native_task_arena_vast.py",
    "public_scene_aura_exact_residual_vast.py",
    "public_scene_simready_isaac_vast.py",
    "simpler_public_vast.py",
)

#: The shared Arena transport seals on behalf of every lane that runs through
#: it, so delegating to it satisfies the contract as completely as calling the
#: seal directly -- and is how the Franka lane already satisfies it.
DELEGATING_ENTRYPOINTS = ("run_arena_native_control_vast",)

#: These build their manifest directly with the shared builder rather than the
#: convenience seal, which satisfies the same contract.
DIRECT_BUILDER_MODULES = (
    "adp_isaac_lab_arena_vast.py",
    "adp_retained_scene_render_vast.py",
)


def _calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


@pytest.mark.parametrize("module", PAID_LANE_MODULES)
def test_a_paid_lane_seals_its_terminal_artifacts(module: str) -> None:
    path = SOURCE_ROOT / module
    assert path.is_file(), f"paid lane module missing: {module}"
    calls = _calls(path)

    sealed = "seal_lane_terminal_artifacts" in calls
    built = "build_task_evaluation_artifact_manifest" in calls
    delegated = any(entrypoint in calls for entrypoint in DELEGATING_ENTRYPOINTS)

    assert sealed or built or delegated, (
        f"{module} returns an allocator result without sealing "
        "artifact_manifest_path/teardown_manifest_path. Call "
        "seal_lane_terminal_artifacts(result, attempt_root=job, lane=...) before "
        "writing the terminal result, or build the manifest directly."
    )


@pytest.mark.parametrize("module", DIRECT_BUILDER_MODULES)
def test_a_direct_builder_lane_uses_the_shared_manifest(module: str) -> None:
    """One schema, so a validator that knows one lane can check another."""

    path = SOURCE_ROOT / module
    assert "build_task_evaluation_artifact_manifest" in _calls(path)


def test_the_lane_list_is_the_set_of_modules_that_run_a_paid_attempt() -> None:
    """The list is only a gate if it stays complete.

    A module that runs a provider attempt is identifiable: it takes a paid
    resource admission grant. Anything matching that and absent from the list
    would slip through the parametrised check above without this.
    """

    discovered = {
        path.name
        for path in sorted(SOURCE_ROOT.glob("*_vast.py"))
        if "paid_resource_admission_grant" in path.read_text(encoding="utf-8")
    }
    known = set(PAID_LANE_MODULES) | set(DIRECT_BUILDER_MODULES)

    assert discovered <= known, (
        "these modules run a paid attempt but are not covered by the terminal "
        f"artifact contract: {sorted(discovered - known)}"
    )


def test_the_seal_leaves_a_dry_run_alone(tmp_path) -> None:
    """A dry run never reaches a provider, so it has no terminal artifacts to
    seal and must not be made to look like a live attempt that lost them."""

    from blueprint_pipeline.task_evaluation_artifact_manifest import (
        seal_lane_terminal_artifacts,
    )

    result = {"status": "dry_run_ready", "blockers": ["b", "a"]}

    sealed = seal_lane_terminal_artifacts(
        result, attempt_root=tmp_path, lane="adp_example"
    )

    assert sealed == result
    # Blocker order is the lane's own, not re-sorted behind its back.
    assert sealed["blockers"] == ["b", "a"]


def test_the_seal_names_both_artifacts_after_a_real_attempt(tmp_path) -> None:
    from blueprint_pipeline.task_evaluation_artifact_manifest import (
        seal_lane_terminal_artifacts,
    )

    provider_run = tmp_path / "vast_provider_run"
    provider_run.mkdir()
    (provider_run / "vast_provider_adapter_result.json").write_text("{}", encoding="utf-8")
    (provider_run / "vast_teardown_manifest.json").write_text(
        '{"continuing_spend_from_this_run": false}', encoding="utf-8"
    )
    (tmp_path / "immutable_execution").mkdir()
    (tmp_path / "immutable_execution" / "result.json").write_text("{}", encoding="utf-8")

    sealed = seal_lane_terminal_artifacts(
        {"status": "completed", "blockers": []},
        attempt_root=tmp_path,
        lane="adp_example",
    )

    assert sealed["status"] == "completed"
    assert sealed["artifact_manifest_path"] == str(tmp_path / "artifact_manifest.json")
    assert sealed["teardown_manifest_path"] == str(
        provider_run / "vast_teardown_manifest.json"
    )
    assert (tmp_path / "artifact_manifest.json").is_file()


def test_an_attempt_that_retained_nothing_is_blocked_by_its_own_manifest(tmp_path) -> None:
    """An attempt that reached a provider and kept no evidence must not report a
    terminal status implying it did."""

    from blueprint_pipeline.task_evaluation_artifact_manifest import (
        seal_lane_terminal_artifacts,
    )

    (tmp_path / "vast_provider_run").mkdir()

    sealed = seal_lane_terminal_artifacts(
        {"status": "completed", "blockers": []},
        attempt_root=tmp_path,
        lane="adp_example",
    )

    assert sealed["status"] == "blocked"
    assert sealed["blockers"]
