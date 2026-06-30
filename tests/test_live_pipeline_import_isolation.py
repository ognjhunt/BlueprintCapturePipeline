"""Static guard: the live-pipeline control-plane / intake lane must never import the
GPU/provider hot lane.

The control plane and intake service run on CPU-only hosts (FastAPI ingest, manifest
emission) and must stay importable without dragging in provider adapters, Isaac/MuJoCo
runtimes, or async GPU runners. This test ast-parses the three entrypoint modules,
follows every relative / ``blueprint_pipeline``-prefixed import transitively, and asserts
the reachable module set is disjoint from the hardcoded hot-lane module names.
"""

from __future__ import annotations

import ast
from pathlib import Path

import blueprint_pipeline

PKG_DIR = Path(blueprint_pipeline.__file__).resolve().parent

# CPU-only entrypoints under guard.
SEED_MODULES = (
    "live_pipeline_control_plane",
    "live_pipeline_input_intake",
    "live_pipeline_intake_service",
)

# GPU / provider / simulator hot-lane modules that must remain unreachable from the
# control-plane import graph. Kept as a literal list so a new accidental import shows up
# as a failure here rather than as a heavy CPU-host import at runtime.
HOT_LANE_MODULES = frozenset(
    {
        "oscar_isaac_closed_loop_eval",
        "oscar_wam_provider_bundle",
        "oscar_wam_provider_command_adapter",
        "oscar_wam_command_adapter",
        "oscar_cosmos_wam_evaluator",
        "unitree_groot_n17_sonic_vast_persistent_session",
        "unitree_groot_n17_sonic_provider_smoke",
        "unitree_groot_n17_sonic_policy_runtime",
        "persistent_wam_short_visual_sanity",
        "wam_compute_providers",
        "wam_generated_video_review",
        "vast_provider_adapter",
        "vast_wam_async_runner",
        "runpod_wam_async_runner",
        "runpod_provider_adapter",
        "lambda_provider_adapter",
        "provider_race",
        "gpu_render_providers",
        "launch_provenance",
        "isaac_worker_runtime_preflight",
        "isaac_g1_policy",
        "isaac_g1_kitchen_parity_job",
        "isaac_particlefield_render_job",
        "mujoco_g1_wam_vla_policy_endpoint_eval",
        "mujoco_g1_simulator_command",
        "mujoco_worker_runtime_preflight",
    }
)


def _local_imported_modules(path: Path) -> set[str]:
    """Return the set of in-package module names imported by ``path``.

    Covers both relative imports (``from .foo import x`` / ``from . import foo``) and
    absolute ``blueprint_pipeline.foo`` imports, including imports nested inside
    functions, so deferred imports cannot smuggle the hot lane in.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level > 0:
                # Relative import: the first dotted component is the sibling module.
                if module:
                    names.add(module.split(".", 1)[0])
                else:
                    # ``from . import foo, bar`` -> the imported names are modules.
                    for alias in node.names:
                        names.add(alias.name.split(".", 1)[0])
            elif module.startswith("blueprint_pipeline."):
                names.add(module.split(".")[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("blueprint_pipeline."):
                    names.add(alias.name.split(".")[1])
    return names


def _transitive_local_modules(seeds: tuple[str, ...]) -> set[str]:
    reachable: set[str] = set()
    stack = list(seeds)
    while stack:
        current = stack.pop()
        module_path = PKG_DIR / f"{current}.py"
        if not module_path.is_file():
            continue
        for dependency in _local_imported_modules(module_path):
            if dependency not in reachable:
                reachable.add(dependency)
                stack.append(dependency)
    return reachable


def test_hot_lane_module_names_unique() -> None:
    # The literal list must stay a 26-name set; a typo collapsing two entries would
    # silently weaken the guard.
    assert len(HOT_LANE_MODULES) == 26


def test_seed_modules_exist() -> None:
    for seed in SEED_MODULES:
        assert (PKG_DIR / f"{seed}.py").is_file(), seed


def test_control_plane_lane_does_not_import_hot_lane() -> None:
    reachable = _transitive_local_modules(SEED_MODULES)
    # Sanity: the graph is non-trivial (we actually walked beyond the seeds).
    assert "common" in reachable
    assert reachable - set(SEED_MODULES)

    leaked = sorted(reachable & HOT_LANE_MODULES)
    assert not leaked, (
        "Control-plane / intake lane transitively imports hot-lane modules: "
        f"{leaked}"
    )
