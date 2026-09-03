"""The sealed policy-canary bundle must import on a provider before any GPU is paid for."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import zipfile

import pytest

# Dotted import on purpose: a bundle-only change must select this suite (pinned
# in tests/test_impacted_test_selection.py).
import blueprint_pipeline.native_task_arena_policy_canary_bundle as bundle
from blueprint_pipeline import vast_provider_adapter
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_bundle import POLICY_RUNTIME_ROOT_MODULE_NAMES
from blueprint_pipeline.native_task_arena_execution_contract import (
    POLICY_RUNTIME_MODULE_NAMES,
)
from blueprint_pipeline.native_task_arena_policy_canary_session import (
    CANDIDATE_IDS,
    build_session_authority,
)
from blueprint_pipeline.provider_runtime_import_closure import (
    CONTROL_PLANE_ONLY_LAZY_IMPORTS,
    assert_provider_runtime_import_closure,
    provider_runtime_import_closure_blockers,
)
from tests.test_native_task_arena_bundle import _packet, _runtime_source_packet
from tests.test_task_evaluation_policy_canary_setup import _setup as public_setup


REPO_PACKAGE = Path(bundle.__file__).resolve().parent
CANARY_SHIPPED_MODULES = sorted(
    set(POLICY_RUNTIME_MODULE_NAMES)
    | {
        "native_task_arena_policy_worker.py",
        "native_task_arena_policy_canary_session.py",
        "native_task_arena_policy_canary_worker.py",
    }
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha(path)}


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _spec(candidate: str) -> dict[str, object]:
    setup = public_setup()
    value: dict[str, object] = {
        "schema_version": "native_task_arena_policy_canary_execution_spec.v1",
        "candidate_id": candidate,
        "execution_authority": "internal_policy_canary_unqualified",
        "claim_ceiling": "diagnostic_policy_execution",
        "ranking_permitted": False,
        "qualification_permitted": False,
        "scene_promotion_permitted": False,
        "policy_endpoint": {"host": "127.0.0.1", "port": 8000},
        "policy_spec": {"candidate_id": candidate},
        "candidate_rights_binding": {"status": "admitted"},
        "checkpoint_digest": "sha256:" + "1" * 64,
        "runtime_identity_digest": "sha256:" + "2" * 64,
        "task_success_contract": setup["task_success_contract"],
        "task_success_contract_digest": setup["task_success_contract_digest"],
        "prompt": "Move the object",
        "max_policy_queries": 10,
        "open_loop_horizon": 8,
        "execution_spec_digest": "",
    }
    value["execution_spec_digest"] = canonical_digest(
        value, digest_field="execution_spec_digest"
    )
    return value


def _build_real_canary_bundle(tmp_path: Path) -> dict[str, object]:
    public = public_setup()
    packet = _packet(tmp_path, scene_id="839873")
    runtime_receipt = _runtime_source_packet(tmp_path)
    construction = _write(tmp_path / "construction.json", {"status": "completed"})
    run_id = "scene-839873-canary-closure"
    activation: dict[str, object] = {
        "schema_version": "task_evaluation_policy_campaign_activation.v1",
        "run_id": run_id,
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "candidate_ids": list(CANDIDATE_IDS),
        "task_success_contract": public["task_success_contract"],
        "task_success_contract_digest": public["task_success_contract_digest"],
        "campaign_unit_count": 10,
        "campaign_units": [
            {
                "campaign_unit_id": f"unit-{index}",
                "cell_id": f"cell-{index}",
                "seed": 3100 + index,
                "candidate_ids": list(CANDIDATE_IDS),
            }
            for index in range(10)
        ],
        "activation_digest": "",
    }
    activation["activation_digest"] = canonical_digest(
        activation, digest_field="activation_digest"
    )
    activation_path = _write(tmp_path / "activation.json", activation)
    cells = []
    for index in range(10):
        scenario = {"family": "canonical", "ordinal": index}
        cells.append(
            {
                "cell_id": f"cell-{index}",
                "seed": 3100 + index,
                "family": "canonical_anchor",
                "cell_spec_digest": "sha256:" + f"{index:064x}",
                "resolved_scenario": scenario,
                "resolved_scenario_digest": canonical_digest(scenario),
                "control_diagnostic": {
                    "mode": "nonblocking_diagnostic_pending",
                    "typed_gap": "controls_pending_at_submission",
                    "policy_execution_blocked": False,
                },
            }
        )
    inputs: dict[str, object] = {
        "schema_version": "task_evaluation_policy_canary_runtime_inputs.v1",
        "run_id": run_id,
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "scene_revision_digest": "sha256:" + "9" * 64,
        "matrix_digest": "sha256:" + "8" * 64,
        "configuration_digest": "sha256:" + "1" * 64,
        "plan_digest": "sha256:" + "2" * 64,
        "activation_digest": activation["activation_digest"],
        "base_native_packet": _record(packet / "native_task_arena_packet_receipt.v1.json"),
        "runtime_source": _record(runtime_receipt),
        "construction_result": _record(construction),
        "candidate_ids": list(CANDIDATE_IDS),
        "task_success_contract": public["task_success_contract"],
        "task_success_contract_digest": public["task_success_contract_digest"],
        "cells": cells,
        "execution_authority": {
            "maximum_provider_allocations": 1,
            "retry_cap": 0,
            "single_warm_provider_session_required": True,
            "caller_surviving_watchdog_required": True,
            "billing_teardown_provider_zero_required": True,
        },
        "runtime_inputs_digest": "",
    }
    inputs["runtime_inputs_digest"] = canonical_digest(
        inputs, digest_field="runtime_inputs_digest"
    )
    inputs_path = _write(tmp_path / "runtime_inputs.json", inputs)
    authority = build_session_authority(
        activation_manifest=activation,
        activation_record=_record(activation_path),
        runtime_inputs=inputs,
        runtime_input_record=_record(inputs_path),
        resource_name="blueprint-native-task-policy-canary-0123456789abcdef",
        hard_cap_usd=4.0,
        hard_ttl_seconds=9_000,
    )
    authority_path = _write(tmp_path / "authority.json", authority)
    inventory = (
        REPO_PACKAGE.parents[1]
        / "docs/experiments/policy_ranking_thesis_20260726/openpi_polaris_checkpoint_inventory.json"
    )
    return bundle.build_policy_canary_session_bundle(
        job_dir=tmp_path / "job",
        packet_dir=packet,
        runtime_source_packet_receipt=runtime_receipt,
        runtime_input_manifest_path=inputs_path,
        session_authority_path=authority_path,
        pi05_execution_spec_path=_write(tmp_path / "pi05.json", _spec("pi05_droid")),
        groot_execution_spec_path=_write(tmp_path / "groot.json", _spec("groot_n17_droid")),
        pi05_checkpoint_inventory_path=inventory,
        implementation_commit="a" * 40,
        generated_at="fixed",
    )


def test_canary_shipped_package_is_statically_import_closed() -> None:
    assert (
        provider_runtime_import_closure_blockers(
            package_source_dir=REPO_PACKAGE, shipped_module_names=CANARY_SHIPPED_MODULES
        )
        == []
    )
    # The runtime imports the GR00T identity helper package-relatively; the
    # flat root copy alone cannot satisfy that on the provider.
    assert "adp009d_groot_worker_identity.py" in POLICY_RUNTIME_MODULE_NAMES
    assert "adp009d_groot_worker_identity.py" in POLICY_RUNTIME_ROOT_MODULE_NAMES


def test_every_exemption_is_a_lazy_import_that_still_exists() -> None:
    for module_name, target in sorted(CONTROL_PLANE_ONLY_LAZY_IMPORTS):
        source = (REPO_PACKAGE / module_name).read_text(encoding="utf-8")
        assert target in source, (module_name, target)
        blockers = provider_runtime_import_closure_blockers(
            package_source_dir=REPO_PACKAGE,
            shipped_module_names=[module_name],
            exemptions=frozenset(),
        )
        assert any(
            blocker.endswith(f"{module_name}->{target}:lazy") for blocker in blockers
        ), (module_name, target, blockers)


def test_validator_flags_unshipped_module_level_lazy_and_subpackage_imports(
    tmp_path: Path,
) -> None:
    package = tmp_path / "blueprint_pipeline"
    (package / "core").mkdir(parents=True)
    (package / "core" / "__init__.py").write_text("", encoding="utf-8")
    (package / "shipped.py").write_text(
        textwrap.dedent(
            """
            from .present import value
            from .absent_module import other

            def lazy():
                from .absent_lazy import thing
                from blueprint_pipeline.absent_absolute import more
                return thing, more

            def uses_subpackage():
                from .core import common
                return common
            """
        ),
        encoding="utf-8",
    )
    (package / "present.py").write_text("value = 1\n", encoding="utf-8")
    (package / "exempt_module_level.py").write_text(
        "from .exempted import x\n", encoding="utf-8"
    )

    blockers = provider_runtime_import_closure_blockers(
        package_source_dir=package,
        shipped_module_names=["shipped.py", "present.py", "exempt_module_level.py", "gone.py"],
        exemptions=frozenset({("exempt_module_level.py", "exempted")}),
    )

    assert blockers == [
        "provider_runtime_import_exemption_not_lazy:exempt_module_level.py->exempted",
        "provider_runtime_import_of_unshipped_subpackage:shipped.py->core",
        "provider_runtime_import_unshipped:shipped.py->absent_absolute:lazy",
        "provider_runtime_import_unshipped:shipped.py->absent_lazy:lazy",
        "provider_runtime_import_unshipped:shipped.py->absent_module",
        "provider_runtime_module_missing:gone.py",
    ]
    with pytest.raises(ValueError, match="^fixture_code:provider_runtime_import"):
        assert_provider_runtime_import_closure(
            package_source_dir=package,
            shipped_module_names=["shipped.py"],
            code="fixture_code",
        )


def test_real_canary_bundle_passes_vast_preflight_and_imports_in_isolation(
    tmp_path: Path,
) -> None:
    receipt = _build_real_canary_bundle(tmp_path)
    assert receipt["status"] == "ready"
    bundle_path = Path(str(receipt["bundle_path"]))

    preflight = vast_provider_adapter._blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight",
        generated_at="fixed",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="native_task_arena_policy_canary_session",
        bundle_path=bundle_path,
        provider_bundle_url="https://storage.example.com/bundle.zip",
        provider_output_put_url="https://storage.example.com/output.zip",
    )
    assert preflight["status"] == "passed", preflight.get("blockers")
    assert preflight.get("blockers") in (None, [])

    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(bundle_path) as archive:
        archive.extractall(extracted)
    runtime = extracted / "provider_runtime"
    package = runtime / "blueprint_pipeline"
    shipped = sorted(path.name for path in package.glob("*.py") if path.name != "__init__.py")
    assert "adp009d_groot_worker_identity.py" in shipped
    assert (runtime / "adp009d_groot_worker_identity.py").is_file()
    assert (
        provider_runtime_import_closure_blockers(
            package_source_dir=package, shipped_module_names=shipped
        )
        == []
    )
    root_modules = sorted(
        path.stem
        for path in runtime.glob("*.py")
        if path.name != "adp_arena_provider_runner.py"
    )
    probe = textwrap.dedent(
        """
        import ast, importlib, sys
        runtime = sys.argv[1]
        sys.path[:0] = [runtime]
        package_modules = sys.argv[2].split(",")
        root_modules = sys.argv[3].split(",")
        for name in package_modules:
            importlib.import_module("blueprint_pipeline." + name)
        for name in root_modules:
            importlib.import_module(name)
        source = open(runtime + "/adp_arena_provider_runner.py", encoding="utf-8").read()
        for node in ast.parse(source).body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                exec(compile(ast.Module(body=[node], type_ignores=[]), "runner", "exec"), {})
        print("IMPORTS_OK", len(package_modules), len(root_modules))
        """
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            probe,
            str(runtime),
            ",".join(Path(name).stem for name in shipped),
            ",".join(root_modules),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("IMPORTS_OK")


def test_bundle_builder_refuses_an_unclosed_shipped_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        bundle,
        "POLICY_RUNTIME_MODULE_NAMES",
        tuple(
            name
            for name in POLICY_RUNTIME_MODULE_NAMES
            if name != "adp009d_groot_worker_identity.py"
        ),
    )
    with pytest.raises(
        ValueError,
        match=(
            "policy_canary_bundle_import_closure_incomplete:.*"
            "groot_n17_droid_policy_runtime.py->adp009d_groot_worker_identity"
        ),
    ):
        _build_real_canary_bundle(tmp_path)
