"""Every step of the input and selected-native chains can be run from a command line.

`materialize_*` functions produce the receipts every paid lane consumes. A step
whose materializer takes a keyword its flag table cannot supply can only be run
from a Python session, and the lane downstream of it dead-ends on a missing input.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


@pytest.mark.parametrize(
    "step",
    [
        "object-absent-reference",
        "candidate-inputs",
        "semantic-teacher",
        "dual-target",
        "use-attestation",
    ],
)
def test_every_input_step_can_supply_its_materializer(step: str) -> None:
    """Same rule as the authority issuer: the flag table is the call."""

    import importlib.util
    import inspect
    import sys

    name = "prepare_artifixer3d_inputs"
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)

    entry = module.STEPS[step]
    upstream = {
        parameter
        for parameter, value in inspect.signature(entry.materialize).parameters.items()
        if value.kind is inspect.Parameter.KEYWORD_ONLY
    }

    assert not upstream - set(entry.params), (
        f"{step} cannot supply {sorted(upstream - set(entry.params))} from a "
        "command line"
    )


@pytest.mark.parametrize(
    "step",
    [
        "source-collider-batch-removal",
        "simready-graph-asset",
        "simready-static-qualification",
        "visual-binding",
        "visual-composition",
        "registered-asset",
        "interaction-affordance",
        "articulated-kinematic-path",
        "camera-rig",
        "construction-bindings",
        "arena-requests",
        "arena-packet",
        "policy-execution-spec",
    ],
)
def test_every_selected_replacement_native_step_can_supply_its_materializer(
    step: str,
) -> None:
    """Every selected-path producer must be callable without a Python session."""

    import importlib.util
    import inspect
    import sys

    name = "materialize_paired_target_native_inputs"
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)

    entry = module.STEPS[step]
    upstream = {
        parameter
        for parameter, value in inspect.signature(entry.materialize).parameters.items()
        if value.kind is inspect.Parameter.KEYWORD_ONLY
    }

    assert not upstream - set(entry.params), (
        f"{step} cannot supply {sorted(upstream - set(entry.params))} from a "
        "command line"
    )
