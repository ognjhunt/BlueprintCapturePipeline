"""The rigid robot-free probe's sealed bundle ships exactly what its GPU worker imports, and nothing else.

Beta journey step: L5 team policy evaluation. A real probe bundle is sealed from
a real scene packet and checked by the same Vast bundle preflight and runtime
contract a live launch runs; the shipped package is pinned to the worker's
static import closure and imported from the bundle's own bytes in an isolated
interpreter. A defect these can catch must never be discovered on a rented GPU.
No GPU, provider, network or spend.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import rigid_robot_free_probe_worker as worker
from blueprint_pipeline import vast_provider_adapter
from blueprint_pipeline.native_task_arena_execution_contract import (
    EXECUTION_MODE_CONTRACTS,
    RIGID_ROBOT_FREE_PROBE_RUNTIME_MODULE_NAMES,
)
from blueprint_pipeline.provider_runtime_bundle_contract import provider_runtime_contract_blockers
from blueprint_pipeline.provider_runtime_import_closure import (
    CONTROL_PLANE_ONLY_LAZY_IMPORTS,
    _import_targets,
    provider_runtime_import_closure_blockers,
)
from blueprint_pipeline.rigid_robot_free_probe_bundle import (
    BUNDLE_RECEIPT_FILENAME,
    RigidRobotFreeProbeBundleError,
    build_rigid_robot_free_probe_bundle,
    current_probe_state,
    load_verified_rigid_robot_free_probe_bundle,
    main,
)
from tests.test_rigid_robot_free_probe_lane import COMMIT, build_world, staged_attempt

PACKAGE = Path(worker.__file__).resolve().parent
# The worker, the two modules its entrypoint and the Vast bootstrap run by name,
# and the modules the Vast adapter requires of every native Arena bundle.
CLOSURE_ROOTS = (
    "rigid_robot_free_probe_worker.py", "native_task_runtime_source_provision.py",
    "native_task_runtime_dependency_download.py", "decision_evidence_contracts.py",
    "native_articulated_motion_geometry.py", "native_articulated_task_state.py", "native_franka_pose_servo.py",
    "native_franka_action_math.py", "native_pose_transforms.py", "native_task_arena_readback.py",
    "native_task_arena_runtime.py", "native_task_camera_observability.py",
)


def _closure(roots: tuple[str, ...]) -> set[str]:
    seen: set[str] = set()
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        for target, lazy in _import_targets((PACKAGE / name).read_text(encoding="utf-8")):
            if not ((name, target) in CONTROL_PLANE_ONLY_LAZY_IMPORTS and lazy):
                pending.append(f"{target}.py")
    return seen


def test_the_shipped_package_is_exactly_the_workers_static_import_closure() -> None:
    assert set(RIGID_ROBOT_FREE_PROBE_RUNTIME_MODULE_NAMES) == _closure(CLOSURE_ROOTS)
    assert len(set(RIGID_ROBOT_FREE_PROBE_RUNTIME_MODULE_NAMES)) == len(RIGID_ROBOT_FREE_PROBE_RUNTIME_MODULE_NAMES)
    assert provider_runtime_import_closure_blockers(
        package_source_dir=PACKAGE, shipped_module_names=RIGID_ROBOT_FREE_PROBE_RUNTIME_MODULE_NAMES) == []
    contract = EXECUTION_MODE_CONTRACTS[worker.EXECUTION_MODE]
    assert contract.expected_output_filename == worker.RESULT_FILENAME
    assert contract.policy_candidate_required is False
    # No policy server, client or checkpoint machinery is reachable from the probe.
    assert not {"native_task_arena_policy_canary_worker.py", "team_candidate_policy_client.py",
                "adp009d_policy_server_worker.py", "droid_https_policy_client.py",
                "beta_robolab_policy_execution.py", "openwam_piper_runtime.py",
                "openwam_piper_policy.py", "openwam_piper_assets.py"} & set(
        RIGID_ROBOT_FREE_PROBE_RUNTIME_MODULE_NAMES)


@pytest.fixture
def sealed(tmp_path, monkeypatch) -> dict:
    world = build_world(tmp_path, monkeypatch)
    staged = staged_attempt(world)
    receipt_path = staged["root"] / "bundle" / "native_task_arena_provider_bundle_receipt.v1.json"
    return {"world": world, "staged": staged, "receipt_path": receipt_path,
            "receipt": json.loads(receipt_path.read_text())}


def test_cli_rebuilds_retained_inputs_at_a_new_commit_without_spending(sealed, tmp_path, capsys):
    state = current_probe_state(sealed["world"]["capture"])
    rebuilt = tmp_path / "rebuilt"
    commit = "b" * 40
    assert main([
        "--job-dir", str(rebuilt), "--packet-dir", str(state["facts"]["packet_dir"]),
        "--runtime-source-packet-receipt", str(state["facts"]["runtime_source_packet_receipt"]),
        "--inputs-path", str(sealed["staged"]["root"] / worker.INPUTS_FILENAME),
        "--implementation-commit", commit,
        "--generated-at", "2026-10-27T18:00:00+00:00",
    ]) == 0
    receipt = load_verified_rigid_robot_free_probe_bundle(
        rebuilt / BUNDLE_RECEIPT_FILENAME, expected_implementation_commit=commit)
    assert json.loads(capsys.readouterr().out)["bundle_sha256"] == receipt["bundle_sha256"]
    assert receipt["bound_runtime_inputs"] == sealed["receipt"]["bound_runtime_inputs"]
    assert sealed["world"]["allocator_calls"] == []
    assert sealed["world"]["webapp_calls"] == []


def test_the_sealed_bundle_passes_the_vast_preflight_and_carries_no_policy(sealed, tmp_path) -> None:
    receipt = sealed["receipt"]
    assert receipt["status"] == "ready" and receipt["execution_mode"] == worker.EXECUTION_MODE
    assert receipt["program_id"] == "blueprint-free-beta-evaluated-pilot"
    assert receipt["candidate_policy_queried"] is False and receipt["policy_candidate_id"] is None
    assert receipt["policy_provisioning_script"] is None
    from blueprint_pipeline.native_task_official_nurec import runtime_data_files
    data = runtime_data_files()
    assert {row["relative_path"] for row in receipt["runtime_root_modules"]} == set(data)
    bundle_path = Path(receipt["bundle_path"])
    preflight = vast_provider_adapter._blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight", generated_at="fixed", enable_blueprint_bundle=True,
        enable_isaac_smoke=True, provider_bundle_kind="native_task_arena", bundle_path=bundle_path,
        provider_bundle_url="https://storage.example.com/bundle.zip",
        provider_output_put_url="https://storage.example.com/output.zip")
    assert preflight["status"] == "passed", preflight.get("blockers")
    with zipfile.ZipFile(bundle_path) as archive:
        names = set(archive.namelist())
        entrypoint = archive.read("provider_runtime/run_adp_arena_provider_runtime.sh").decode()
        runner_source = archive.read("provider_runtime/adp_arena_provider_runner.py").decode()
        import hashlib
        for row in receipt["runtime_root_modules"]:
            payload = archive.read("provider_runtime/" + row["relative_path"])
            assert payload == data[row["relative_path"]].read_bytes()
            assert row["sha256"] == "sha256:" + hashlib.sha256(payload).hexdigest()
    assert f"provider_runtime/runtime_inputs/{worker.INPUTS_FILENAME}" in names
    assert not any("provisioning" in name or "policy_execution_spec" in name or "policy_server" in name
                   for name in names)
    assert runner_source == (Path(worker.__file__)).read_text()
    assert provider_runtime_contract_blockers(provider_bundle_kind="native_task_arena",
                                              entrypoint_text=entrypoint, runner_text=runner_source) == []
    # A runner that does not record the probe is not this mode's worker.
    imposter = runner_source.replace("record_arena_robot_free_probe", "record_something_else")
    assert "provider_runner_missing_native_task_arena_runtime_contract" in provider_runtime_contract_blockers(
        provider_bundle_kind="native_task_arena", entrypoint_text=entrypoint, runner_text=imposter)


def test_the_paid_lane_reverifies_the_receipt_and_bytes(sealed) -> None:
    path = sealed["receipt_path"]
    assert load_verified_rigid_robot_free_probe_bundle(path, expected_implementation_commit=COMMIT)[
        "bundle_sha256"] == sealed["receipt"]["bundle_sha256"]
    with pytest.raises(RigidRobotFreeProbeBundleError, match="commit_mismatch"):
        load_verified_rigid_robot_free_probe_bundle(path, expected_implementation_commit="b" * 40)
    with Path(sealed["receipt"]["bundle_path"]).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(RigidRobotFreeProbeBundleError, match="identity_invalid"):
        load_verified_rigid_robot_free_probe_bundle(path, expected_implementation_commit=COMMIT)


def test_the_bundle_refuses_an_input_for_another_packet(sealed, tmp_path) -> None:
    inputs_path = sealed["staged"]["root"] / worker.INPUTS_FILENAME
    value = json.loads(inputs_path.read_text())
    value["packet_receipt_digest"] = "sha256:" + "0" * 64
    value["inputs_digest"] = ""
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    value["inputs_digest"] = canonical_digest(value, digest_field="inputs_digest")
    other = tmp_path / "other-inputs.json"
    other.write_text(json.dumps(value))
    facts = sealed["world"]["facts"]
    with pytest.raises(RigidRobotFreeProbeBundleError, match="packet_binding_invalid"):
        build_rigid_robot_free_probe_bundle(job_dir=tmp_path / "other-bundle", packet_dir=facts["packet_dir"],
                                            runtime_source_packet_receipt=facts["runtime_source_packet_receipt"],
                                            inputs_path=other, implementation_commit=COMMIT)


def test_the_worker_imports_and_verifies_its_inputs_from_the_bundle_alone(sealed, tmp_path) -> None:
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(sealed["receipt"]["bundle_path"]) as archive:
        archive.extractall(extracted)
    runtime = extracted / "provider_runtime"
    shipped = sorted(path.stem for path in (runtime / "blueprint_pipeline").glob("*.py") if path.name != "__init__.py")
    assert sorted(f"{name}.py" for name in shipped) == sorted(RIGID_ROBOT_FREE_PROBE_RUNTIME_MODULE_NAMES)
    probe = textwrap.dedent(
        """
        import importlib, json, pathlib, sys
        runtime = pathlib.Path(sys.argv[1])
        sys.path[:0] = [str(runtime)]
        for name in sys.argv[2].split(","):
            importlib.import_module("blueprint_pipeline." + name)
        import adp_arena_provider_runner as runner
        assert pathlib.Path(runner.__file__).resolve().parent == runtime.resolve()
        manifest, inputs = runner.verified_runtime_inputs(runtime)
        print("PROBE_INPUTS_VERIFIED", inputs["probe_request"]["request_digest"], len(sys.argv[2].split(",")))
        """
    )
    completed = subprocess.run([sys.executable, "-I", "-c", probe, str(runtime), ",".join(shipped)],
                               check=False, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    request = json.loads((sealed["staged"]["root"] / worker.INPUTS_FILENAME).read_text())["probe_request"]
    assert completed.stdout.split() == ["PROBE_INPUTS_VERIFIED", request["request_digest"], str(len(shipped))]


def test_the_worker_never_launches_isaac_for_inputs_the_manifest_does_not_bind(sealed, tmp_path) -> None:
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(sealed["receipt"]["bundle_path"]) as archive:
        archive.extractall(extracted)
    runtime = extracted / "provider_runtime"
    inputs = runtime / "runtime_inputs" / worker.INPUTS_FILENAME
    value = json.loads(inputs.read_text())
    value["scene_plan"]["scenario"]["seed"] = 7
    inputs.write_text(json.dumps(value))
    launched: list[str] = []

    def never(*_args, **_kwargs):
        launched.append("isaac")
        raise AssertionError("Isaac was launched for unbound inputs")

    runtime_seam = worker.ProbeRuntime(device="cuda:0", resolve_render_path=never, launch_isaac=never,
                                       prepare_preconstruction=never, build_environment=never,
                                       read_device_binding=never, make_rigid_task_readback=never, record_probe=never)
    result, app = worker.run_rigid_robot_free_probe(runtime_dir=runtime, output_root=tmp_path / "out",
                                                    runtime=runtime_seam)
    assert result["status"] == "blocked" and result["phase_reached"] == "not_started" and app is None
    assert any("rigid_robot_free_probe_inputs_binding_invalid" in item for item in result["blockers"])
    assert launched == [] and result["candidate_policy_queried"] is False
    sealed_result = json.loads((tmp_path / "out" / worker.RESULT_FILENAME).read_text())
    assert sealed_result == result
