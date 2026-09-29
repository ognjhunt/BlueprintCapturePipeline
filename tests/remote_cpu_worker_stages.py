"""Stage handlers the remote CPU worker tests run inside a real stage child (plan 14 PR 3).

The worker tests ship this file in the fixture release as ``src/remote_cpu_worker_stages.py``, so each
handler runs from the extracted archive in a spawned interpreter, exactly as a registered stage would.
A handler takes the sealed descriptor and the stage roots and returns the stage's sealed result.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

from blueprint_pipeline.decision_evidence_contracts import canonical_digest

NEW_BYTES = b'{"adapter": "result"}\n'


def sealed_result(descriptor: dict[str, Any], *, blockers: list[str], **extra: Any) -> dict[str, Any]:
    result = {"schema_version": "task_evaluation_episode_compilation_result.v1",
              "status": "blocked" if blockers else "compiled_for_production_launch",
              "compilation_id": descriptor["outputs"]["output_root"].rsplit("/", 1)[-1],
              "source_commit": descriptor["code"]["source_commit"], "provider_mutation_performed": False,
              "paid_execution_requested": False, "blockers": blockers, **extra, "result_digest": ""}
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


def _identity() -> dict[str, Any]:
    import blueprint_pipeline

    return {"pid": os.getpid(), "ppid": os.getppid(), "argv": sys.argv[1:], "pytest_loaded": "pytest" in sys.modules,
            "package_file": blueprint_pipeline.__file__, "working_directory": os.getcwd(),
            "environment_names": sorted(os.environ),
            "presigned_in_environment": any("X-Amz-" in value for value in os.environ.values())}


def compiled(descriptor: dict[str, Any], roots: Any) -> dict[str, Any]:
    """Write new bytes, a copy of an input and a bundle member under the output root, and a scratch member."""

    output = roots.local(descriptor["outputs"]["output_root"])
    adapter = output / "native-arena-adapter"
    adapter.mkdir(parents=True)
    (adapter / "result.json").write_bytes(NEW_BYTES)
    inputs = {Path(item["materialize_at"]).name: roots.local(item["materialize_at"]) for item in descriptor["inputs"]}
    (output / "robot.json").write_bytes(inputs["robot.json"].read_bytes())
    with zipfile.ZipFile(inputs["runtime.zip"]) as bundle:
        member = bundle.read("runtime/model.bin")
    (adapter / "model.bin").write_bytes(member)
    scratch = roots.local(descriptor["outputs"]["declared_scratch"][0]) / "adapter-members"
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "model.bin").write_bytes(member)
    return sealed_result(descriptor, blockers=[], identity=_identity())


def slow_compiled(descriptor: dict[str, Any], roots: Any) -> dict[str, Any]:
    time.sleep(3.5)
    return compiled(descriptor, roots)


def hangs(descriptor: dict[str, Any], roots: Any) -> dict[str, Any]:
    """Record this child and a grandchild in its session, then never return."""

    grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    marker = Path(os.environ["REMOTE_CPU_TEST_PIDS"])
    marker.write_text(f"{os.getpid()} {grandchild.pid}\n", encoding="utf-8")
    threading.Event().wait()
    raise AssertionError("unreachable")


def reads_success_schema(descriptor: dict[str, Any], roots: Any) -> dict[str, Any]:
    """Read a schema through ``parents[2]``, as the compile does, and turn an unreadable one into a
    deterministic blocked result, as the host's envelope load does (plan 14 Facts)."""

    from blueprint_pipeline.rigid_task_success_contract_schema import rigid_task_success_contract_schema

    try:
        rigid_task_success_contract_schema()
        blocker = "episode_compilation_envelope_invalid"
    except OSError:
        blocker = "rigid_task_success_contract_schema_unavailable"
    return sealed_result(descriptor, blockers=[blocker], automatic_retry_performed=False)


# Plan 14 PR 4: the episode-compilation stage with the compiler's own test stand-ins.  The compiler's tests
# (``tests/test_task_evaluation_native_arena_episode_compiler.py``) replace its three runtime-bound steps; these
# do the same from their inputs alone, so a host compile and a worker compile at the same paths agree byte for
# byte.  The configured-scene extraction, task adapter, request writing, adapter bundle zip and result sealing
# stay the compiler's own.
GROUNDED_PROMPT = ("Pick up the open book, place it fully inside the blue document tray, release it, "
                   "and move the gripper clear.")


def _write_sealed(path: Path, value: dict[str, Any], field: str) -> dict[str, Any]:
    import json

    value[field] = canonical_digest(value, digest_field=field)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return value


def stand_in_appearance(*, source_path: Any, output_root: Path) -> dict[str, Any]:
    """What the compiler's tests pass as ``native_appearance_materializer``: a copied field and its backend."""

    import hashlib

    output_root.mkdir()
    output = output_root / "scene_appearance.usdc"
    output.write_bytes(Path(source_path).read_bytes())
    digest, size = "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest(), output.stat().st_size
    backend = {"schema_version": "task_evaluation_appearance_render_backend.v1",
               "kind": "nvidia_3dgrut_direct_nurec_transcode", "source_configured_appearance_digest": digest,
               "particlefield_digest": digest, "authoring_receipt_digest": "sha256:" + "9" * 64,
               "upstream_converter": {"source_revision": "test"}, "projection_mode_hint": "perspective",
               "sorting_mode_hint": "cameraDistance", "color_space": "srgb_rec709_display", "backend_digest": ""}
    backend["backend_digest"] = canonical_digest(backend, digest_field="backend_digest")
    return {"status": "nurec_converted_to_particlefield", "path": str(output),
            "representation": "particlefield_3d_gaussian_splat", "source_configured_appearance_digest": digest,
            "source_configured_appearance_size_bytes": size, "particlefield_digest": digest,
            "particlefield_size_bytes": size, "representation_conversion_performed": True,
            "exact_learned_arrays_preserved": True,
            "gaussian_field_quality": {"schema_version": "gaussian_field_quality.v1", "status": "qualified",
                                       "blockers": [], "learned_tensors_mutated": False},
            "appearance_render_backend": backend}


def stand_in_packet(*, request: dict[str, Any], evidence_root: Any, output_dir: Any, link_sources_within: Any) -> None:
    """A packet directory from the packet request: its documents, and the configured assets hard-linked in."""

    output = Path(output_dir)
    output.mkdir()
    (output / "native_task_arena_packet_request.v1.json").write_text(
        __import__("json").dumps(request, sort_keys=True) + "\n", encoding="utf-8")
    task_digest = canonical_digest(request["task_spec"])
    _write_sealed(output / "native_task_runtime_contract.v1.json",
                  {"task_spec_digest": task_digest, "robot": {"robot_id": "franka", "joint_count": 7}}, "contract_digest")
    _write_sealed(output / "native_task_arena_scene_plan.v1.json",
                  {"schema_version": "native_task_arena_scene_plan.v1", "task_spec_digest": task_digest}, "plan_digest")
    _write_sealed(output / "native_task_arena_packet_receipt.v1.json",
                  {"schema_version": "native_task_arena_packet_receipt.v1", "task_spec_digest": task_digest},
                  "receipt_digest")
    assets = Path(evidence_root) / "configured-scene"
    (output / "assets").mkdir()
    for source in sorted(assets.iterdir()) if assets.is_dir() else []:
        os.link(source, output / "assets" / source.name)


def _extract_through_store(bundle: Any, destination: Path, store: Any) -> dict[str, Any]:
    """Extract a manifest-bound bundle through the shared member store, one hard link per member."""

    import hashlib
    import json

    manifest_name = "task_evaluation_adapter_bundle_manifest.v1.json"
    destination.mkdir(parents=True, mode=0o750)
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read(manifest_name))
        for row in manifest["entries"]:
            data = archive.read(row["relative_path"])
            digest = "sha256:" + hashlib.sha256(data).hexdigest()
            target = destination.joinpath(*row["relative_path"].split("/")[1:])
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
            cached = Path(store) / digest.removeprefix("sha256:")
            if not cached.exists():
                cached.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
                cached.write_bytes(data)
                cached.chmod(0o440)
            os.link(cached, target)
    return {**manifest, "manifest_digest": manifest.get("manifest_digest") or canonical_digest(manifest)}


def stand_in_adapter(*, request: dict[str, Any], compiled_episode_packet_path: Any,
                     compiled_episode_packet_reference: dict[str, Any], configured_revision: dict[str, Any],
                     runtime_source_bundle_path: Any, output_root: Any, content_store_root: Any = None,
                     external_layers: Any = None) -> dict[str, Any]:
    """The adapter's layout: both bundles extracted through the member store, then its sealed result."""

    import json

    from blueprint_pipeline.task_evaluation_launch_preparation_queue import write_launch_preparation_record_exclusive

    root = Path(output_root)
    root.mkdir(mode=0o750)
    construction = _extract_through_store(compiled_episode_packet_path, root / "construction-packet",
                                          content_store_root)
    runtime = _extract_through_store(runtime_source_bundle_path, root / "runtime-source", content_store_root)
    receipt_path = root / "runtime-source" / "native_task_runtime_source_packet.v1.json"
    receipt = _write_sealed(receipt_path, {"schema_version": "native_task_runtime_source_packet.v1",
                                           "packet_path": "packet.bin"}, "receipt_digest")
    packet_receipt = json.loads((root / "construction-packet" / "native_task_arena_packet_receipt.v1.json").read_text())
    result = {"schema_version": "task_evaluation_native_arena_adapter_result.v1",
              "status": "native_arena_adapter_materialized", "preparation_id": request["preparation_id"],
              "source_commit": request["expected_production_commit"], "adapter_kind": "native_task_arena",
              "adapter_version": "v1", "construction_manifest_digest": construction["manifest_digest"],
              "configured_scene_revision_digest": configured_revision["revision_digest"],
              "runtime_source_manifest_digest": runtime["manifest_digest"],
              "packet_receipt_digest": packet_receipt["receipt_digest"],
              "runtime_source_receipt_digest": receipt["receipt_digest"],
              "packet_root": str(root / "construction-packet"), "runtime_source_receipt": str(receipt_path),
              "provider_mutation_performed": False, "catalog_mutation_performed": False,
              "paid_execution_requested": False, "result_digest": ""}
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    write_launch_preparation_record_exclusive(root / "task_evaluation_native_arena_adapter_result.v1.json", result)
    return result


def install_compile_stand_ins(assign: Any = setattr) -> Any:
    """Patch the compiler's runtime-bound steps through ``assign`` (a test passes ``monkeypatch.setattr``) and
    return the compiler to call; installing twice leaves the first installation in place."""

    import functools

    from blueprint_pipeline import task_evaluation_native_arena_episode_compiler as compiler

    if compiler.materialize_native_arena_adapter is not stand_in_adapter:
        template = compiler.adapt_rigid_relocation_task_template

        def grounded(**kwargs: Any) -> dict[str, Any]:
            result = template(**kwargs)
            if kwargs["request"]["task"].get("strategy") == "pick_and_place":
                result["native_task_definition"]["task_spec"].update(instruction_subject_label="open book",
                                                                     prompt=GROUNDED_PROMPT)
                result["adapter_digest"] = canonical_digest(result, digest_field="adapter_digest")
            return result

        assign(compiler, "adapt_rigid_relocation_task_template", grounded)
        assign(compiler, "materialize_native_task_arena_packet", stand_in_packet)
        assign(compiler, "materialize_native_arena_adapter", stand_in_adapter)
    return functools.partial(compiler.compile_native_arena_episode, native_appearance_materializer=stand_in_appearance)


def compiles_episode(descriptor: dict[str, Any], roots: Any) -> dict[str, Any]:
    """The registered ``episode_compilation`` handler, run with the compiler's stand-ins (plan 14 task 4.3)."""

    from blueprint_pipeline.task_evaluation_episode_compilation_remote import run_episode_compilation_in_worker

    return run_episode_compilation_in_worker(descriptor, roots, episode_compiler=install_compile_stand_ins())
