"""Guard the articulated harness against another scene's assumptions leaking in.

Roughly fifteen 840313 hardcodings were found one paid run at a time - input
filenames, prim paths, VLM prompts describing a beverage can, a probe set of
drop/slide/tip/gripper. Each was individually reasonable when written and
individually invisible until a different scene hit it. This test makes the
class of defect cheap to catch: no module in the articulated or suppression
lane may name a scene, an instance, or a specific object in executable code.

Docstrings are exempt on purpose. Explaining which run motivated a contract is
worth keeping; depending on that run at runtime is not.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LANE_MODULES = (
    "articulated_controls_probe.py",
    "articulated_runtime_composition.py",
    "articulated_control_planner.py",
    "articulated_scripted_ik_preflight.py",
    "articulated_dynamics_realism.py",
    "gaussian_suppression_volume.py",
    "gaussian_suppression_render.py",
    "articulated_interior_exposure.py",
    "articulated_interior_authoring.py",
    "articulated_support_aperture.py",
    "articulated_render_materials.py",
    "articulated_native_probe.py",
    "articulated_isaac_bundle.py",
    "agent_enrichment_acceptance.py",
    "replacement_colour_fidelity.py",
)
# The older paid lanes were each written against the 840313 can and audited
# only after a different scene hit them. These are the modules whose job is
# transport or task execution - they must stay object-agnostic. Registries that
# deliberately record the 840313 asset's identity are not in this list.
OLDER_LANE_MODULES = (
    "adp_aura_interiorgs_vast.py",
    "public_scene_aura_adapter.py",
    "public_scene_aura_execution.py",
    "adp_joint_agent_vast.py",
    "joint_agent_articulation_review.py",
    "joint_agent_execution_receipt.py",
    "adp009d_policy_episode.py",
    "adp009d_episode_batch.py",
    "adp009d_isaac_episode_adapter.py",
    "adp_episode_evidence_index.py",
    "adp_isaac_lab_arena_vast.py",
    "adp_isaac_lab_arena_request.py",
    "arena_result_ingest.py",
)
# A task-object name in an execution seam is the defect that hid longest: the
# Isaac adapter took an ``approved_can`` even after it had been generalized.
OBJECT_NAMED_PARAMETERS = ("approved_can:", "the_can", "can_prim", "can_path")

FORBIDDEN = (
    "840796",
    "840313",
    "840411",
    "refrigerator",
    "fridge",
    "canned_beverage",
    "beverage",
    "ins123",
    "ins160",
    "upper_door",
    "lower_door",
)


def _executable_source(path: Path) -> str:
    """The module's source with every docstring removed."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    spans: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            spans.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(
        line for number, line in enumerate(lines, start=1) if number not in spans
    )


@pytest.mark.parametrize("module", LANE_MODULES)
def test_lane_module_names_no_scene_or_object(module: str) -> None:
    path = ROOT / "src/blueprint_pipeline" / module
    assert path.is_file(), module
    source = _executable_source(path).lower()

    offenders = [token for token in FORBIDDEN if token in source]

    assert offenders == [], f"{module} hardcodes {offenders}"


def test_the_isaac_worker_names_no_scene_or_object() -> None:
    path = ROOT / "scripts/run_adp009d_articulated_isaac_worker.py"
    source = _executable_source(path).lower()

    offenders = [token for token in FORBIDDEN if token in source]

    assert offenders == [], f"worker hardcodes {offenders}"


@pytest.mark.parametrize("module", OLDER_LANE_MODULES)
def test_older_lane_module_names_no_scene(module: str) -> None:
    path = ROOT / "src/blueprint_pipeline" / module
    assert path.is_file(), module
    source = _executable_source(path).lower()

    offenders = [
        token
        for token in ("840796", "840313", "840411", "canned_beverage", "ins160")
        if token in source
    ]

    assert offenders == [], f"{module} hardcodes {offenders}"


@pytest.mark.parametrize("module", OLDER_LANE_MODULES)
def test_older_lane_execution_seams_take_no_can_shaped_parameter(module: str) -> None:
    """An execution seam that names the object cannot serve another task."""

    source = _executable_source(ROOT / "src/blueprint_pipeline" / module)

    offenders = [token for token in OBJECT_NAMED_PARAMETERS if token in source]

    assert offenders == [], f"{module} declares {offenders}"


def test_no_lane_module_assumes_a_default_prim_name() -> None:
    """`/Asset` is this scene's root, not every asset's root."""

    offenders = []
    for module in LANE_MODULES:
        source = _executable_source(ROOT / "src/blueprint_pipeline" / module)
        if '"/Asset' in source or "'/Asset" in source:
            offenders.append(module)

    assert offenders == []
