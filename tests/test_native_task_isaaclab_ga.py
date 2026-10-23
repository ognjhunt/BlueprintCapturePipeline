import hashlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from blueprint_pipeline import native_task_isaaclab_ga as ga
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_isaaclab_launch import NativeTaskIsaacLabLaunchError
from blueprint_pipeline.native_task_runtime_source_packet import (
    NativeTaskRuntimeSourcePacketError, materialize_native_task_runtime_source_packet,
)


def provisioning(tmp_path):
    experience = tmp_path / "apps/headless.kit"
    experience.parent.mkdir()
    experience.write_text("exact release experience")
    value = {
        "schema_version": "native_task_runtime_source_provisioning.v1", "status": "completed",
        "runtime_profile": ga.PROFILE, "container_image": ga.IMAGE, "python_executable": ga.PYTHON,
        "extraction_dir": str(tmp_path), "runtime_import_probe_returncode": 0,
        "runtime_experience": {"path": str(experience), "source_revision": ga.COMMIT,
            "source_tree": ga.TREE, "sha256": "sha256:" + hashlib.sha256(experience.read_bytes()).hexdigest()},
        "runtime_import_probes": [{"module": c["module"], "observed_version": c["expected_version"],
            "version_matches": True, "cuda_tensor_operation_passed": True,
            "observed_cuda_version": c.get("expected_cuda_version")} for c in ga.PROBES],
        "installed_runtime_dependencies": [{"package": c["package"], "wheel_sha256": c["sha256"]}
                                           for c in ga.GA_SUPPLEMENTAL_WHEELS],
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    return value


def test_ga_requires_exact_experience_runtime_and_supplemental_closure(tmp_path):
    value = provisioning(tmp_path)
    assert ga.verify_ga_launch(value)["runtime_profile"] == ga.PROFILE
    value["installed_runtime_dependencies"].pop()
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    with pytest.raises(NativeTaskIsaacLabLaunchError, match="supplemental_dependencies_invalid"):
        ga.verify_ga_launch(value)


@pytest.mark.parametrize("mutation", ["cuda", "image", "experience"])
def test_ga_readback_cannot_be_promoted_after_drift(tmp_path, mutation):
    value = provisioning(tmp_path)
    if mutation == "cuda":
        value["runtime_import_probes"][1]["cuda_tensor_operation_passed"] = False
    elif mutation == "image":
        value["container_image"] = "nvcr.io/nvidia/isaac-lab:latest"
    else:
        from pathlib import Path
        Path(value["runtime_experience"]["path"]).write_text("changed")
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    with pytest.raises(NativeTaskIsaacLabLaunchError):
        ga.verify_ga_launch(value)


def test_ga_materialization_does_not_accept_missing_supplemental_dependencies(tmp_path):
    with pytest.raises(NativeTaskRuntimeSourcePacketError, match="supplemental_wheel_dir_required"):
        materialize_native_task_runtime_source_packet(output_dir=tmp_path / "out",
            isaaclab_repo=tmp_path, arena_repo=tmp_path, dependency_wheel_dir=None,
            runtime_profile=ga.PROFILE)
    with pytest.raises(ValueError, match="supplemental_wheel_identity_mismatch"):
        ga.materialize_ga_source_packet(output_dir=tmp_path / "out", isaaclab_repo=tmp_path,
            arena_repo=tmp_path, dependency_wheel_dir=tmp_path)


def test_ga_uses_explicit_physx_context_lifecycle(monkeypatch):
    calls = []
    context = SimpleNamespace(__enter__=lambda: calls.append("enter"),
        __exit__=lambda kind, exc, tb: calls.append((kind, exc.code)))
    lab_app = ModuleType("isaaclab.app")
    def launch(config, arguments):
        calls.append(arguments)
        return context
    lab_app.launch_simulation = launch
    physics = ModuleType("isaaclab_physx.physics")
    physics.PhysxCfg = lambda: "physx"
    omni, kit, app = ModuleType("omni"), ModuleType("omni.kit"), ModuleType("omni.kit.app")
    app.get_app = lambda: SimpleNamespace(is_running=lambda: True)
    omni.kit, kit.app = kit, app
    for key, value in {"isaaclab.app": lab_app, "isaaclab_physx.physics": physics,
                       "omni": omni, "omni.kit": kit, "omni.kit.app": app}.items():
        monkeypatch.setitem(sys.modules, key, value)
    result = ga.ga_app_launcher(headless=True, enable_cameras=True, device="cuda:0")
    assert calls[0] == {"headless": True, "enable_cameras": True, "device": "cuda:0",
                        "physics": "isaacsim_physx", "require_kit": True}
    assert calls[1] == "enter"
    assert result.app.is_exiting() is False
    result.app.close(exit_code=3)
    assert calls[-1] == (SystemExit, 3)
