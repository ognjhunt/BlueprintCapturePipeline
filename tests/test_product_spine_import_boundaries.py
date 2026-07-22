"""Mechanical import direction for the sellable artifact-contract spine."""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "blueprint_pipeline"
PRODUCT_SPINE_MODULES = (
    "artifact_contracts",
    "robot_eval_dataset",
    "evaluation_run",
    "evaluation_run_contract",
    "evaluation_run_execution",
    "post_training_data_package",
    "buyer_package_readout",
    "attempt_closure_projection",
)
FORBIDDEN_DEPENDENCY_PREFIXES = (
    "agent_runtime",
    "alpha_readiness",
    "g1_",
    "gear_",
    "gpu_render",
    "groot_",
    "kitchen_",
    "paid_",
    "production_gpu",
    "runpod",
    "single_g1_",
    "vast_",
)


def _local_import_roots(module_name: str) -> set[str]:
    tree = ast.parse((PACKAGE_ROOT / f"{module_name}.py").read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level and node.module:
                roots.add(node.module.split(".", 1)[0])
            elif node.module and node.module.startswith("blueprint_pipeline."):
                roots.add(node.module.split(".", 2)[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("blueprint_pipeline."):
                    roots.add(alias.name.split(".", 2)[1])
    return roots


def test_sellable_product_spine_does_not_import_campaign_readiness_or_provider_code() -> None:
    violations: list[str] = []
    for module_name in PRODUCT_SPINE_MODULES:
        for dependency in sorted(_local_import_roots(module_name)):
            if dependency.startswith(FORBIDDEN_DEPENDENCY_PREFIXES):
                violations.append(f"{module_name}->{dependency}")

    assert violations == [], (
        "Product artifact contracts must depend on neutral contracts, not campaign, "
        "readiness, or paid-provider implementations: " + ", ".join(violations)
    )
