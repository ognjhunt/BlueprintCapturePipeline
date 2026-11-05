"""Release bytes, interpreter ABI and money boundaries without model downloads."""

import ast
import re
import zipfile

import pytest

from blueprint_pipeline.cosmos_droid_bundle import (
    CANDIDATES,
    CHECKPOINT_BYTES,
    IMAGE_DOWNLOAD_ALLOWANCE_BYTES,
    MAXIMUM_OUTPUT_BYTES,
    profile,
    runtime_contract,
    stage_runtime,
    transfer_byte_budget,
)
from blueprint_pipeline.cosmos_droid_dependencies import (
    TOTAL_DOWNLOAD_BYTES,
    verified_dependency_manifest,
)
from blueprint_pipeline.cosmos_droid_provisioning import client_wheels, server_environment
from blueprint_pipeline.cosmos_droid_source import (
    CHECKPOINT_REGISTRY_FILES,
    SOURCE_ARCHIVE,
    SOURCE_MANIFEST,
    checkpoint_registry_downloads,
    checkpoint_registry_path,
    extract_source,
    verify_source_directory,
)


@pytest.mark.parametrize("candidate", CANDIDATES)
def test_release_stages_verified_source_and_counts_every_external_byte(tmp_path, candidate):
    contract = stage_runtime(tmp_path, candidate)
    assert all((tmp_path / name).is_file() for name in contract["required_entries"])
    dependencies = verified_dependency_manifest(tmp_path / "cosmos-droid-dependencies.v1.json")
    external = {
        row["url"]: row["size_bytes"]
        for row in [
            dependencies["interpreter"],
            *dependencies["server_wheels"],
            *dependencies["thin_client_wheels"],
        ]
        if row.get("url")
    }
    assert sum(external.values()) == TOTAL_DOWNLOAD_BYTES == 5_652_022_212
    bundle = {
        "cosmos_runtime": contract,
        "bundle_size_bytes": 1234,
        "runtime_source_packet": {
            "transport": "content_addressed_external_layer.v1",
            "packet_size_bytes": 5678,
        },
    }
    download, upload = transfer_byte_budget(bundle)
    assert (
        download
        == 1234
        + 5678
        + CHECKPOINT_BYTES[candidate]
        + TOTAL_DOWNLOAD_BYTES
        + IMAGE_DOWNLOAD_ALLOWANCE_BYTES
    )
    assert upload == MAXIMUM_OUTPUT_BYTES
    bundle["cosmos_runtime"]["checkpoint_download_bytes"] -= 1
    with pytest.raises(ValueError, match="sealed_runtime_identity"):
        transfer_byte_budget(bundle)


_PROCESSORS = "cosmos_framework/data/generator/processors/"
# Model-directory files the pinned transformers 4.57.6 loaders read for these
# publisher snapshots (traced on both). Optional probes for files neither
# publisher ships, such as added_tokens.json, are not required.
_LOADER_FILES = {
    "AutoTokenizer": {"tokenizer_config.json", "tokenizer.json", "config.json", "chat_template.jinja"},
    "AutoProcessor": {"tokenizer_config.json", "tokenizer.json", "config.json", "chat_template.json",
                      "preprocessor_config.json", "video_preprocessor_config.json"},
}
# Edge's processor_config.json names Cosmos3EdgeProcessor, so the pinned
# dispatch builds the native processor; Nano's snapshot takes AutoProcessor.
_NATIVE_EDGE_PROCESSOR = {"cosmos3_edge_policy_droid": True, "cosmos3_nano_policy_droid": False}
_MODEL_FILE = re.compile(r"[\w./-]+\.(?:json|jinja|txt|model)")


def _model_files_opened(scope, functions):
    """Files a pinned source block opens from the model directory, following its local calls."""
    opened, pending, followed = set(), list(scope), set()
    while pending:
        for call in (node for node in ast.walk(pending.pop()) if isinstance(node, ast.Call)):
            if isinstance(call.func, ast.Attribute) and call.func.attr == "from_pretrained":
                loader = ast.unparse(call.func.value)
                assert loader in _LOADER_FILES, f"unaudited processor loader: {loader}"
                opened |= _LOADER_FILES[loader]
            name = getattr(call.func, "id", None)
            if name in functions and name not in followed:
                followed.add(name)
                pending.append(functions[name])
            opened |= {arg.value for arg in call.args if isinstance(arg, ast.Constant)
                       and isinstance(arg.value, str) and _MODEL_FILE.fullmatch(arg.value)}
    return opened


@pytest.mark.parametrize("candidate", CANDIDATES)
def test_snapshot_downloads_every_file_the_pinned_processor_opens(tmp_path, candidate):
    # Provisioning fetches exactly required_files, so a file the pinned processor
    # opens and the manifest omits stops the policy server at startup.
    selected = profile(candidate)
    stage_runtime(tmp_path, candidate)
    with zipfile.ZipFile(tmp_path / SOURCE_ARCHIVE) as archive:
        dispatch_module = ast.parse(archive.read(_PROCESSORS + "base.py"))
        edge_module = ast.parse(archive.read(_PROCESSORS + "cosmos3_edge_processing.py"))
    functions = {node.name: node for node in edge_module.body if isinstance(node, ast.FunctionDef)}
    dispatch = next(node for node in ast.walk(dispatch_module) if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Call)
                    and getattr(node.test.func, "id", None) == "is_cosmos3_edge_native_snapshot")
    scope = [dispatch.test, *dispatch.body] if _NATIVE_EDGE_PROCESSOR[candidate] else dispatch.orelse
    opened = _model_files_opened(scope, functions)
    manifest = selected.verified_snapshot_manifest(tmp_path / selected.SNAPSHOT_STAGED_NAME)
    assert opened, "no processor file reads found in the pinned source"
    assert sorted(opened - {row["path"] for row in manifest["required_files"]}) == []


_SOURCE = "cosmos_framework/"
# Pinned-source modules the policy server's model load runs for these snapshots,
# traced from action_policy_server_robolab through OmniInference.create into the
# hydra targets each snapshot's config.json names.
_LOAD_MODULES = (
    "scripts/action_policy_server_robolab.py", "inference/inference.py", "inference/common/inference.py",
    "inference/common/args.py", "inference/args.py", "inference/model.py", "model/generator/omni_mot_model.py",
    "model/generator/tokenizers/wan2pt2_vae_4x16x16.py", "data/generator/processors/__init__.py",
    "data/generator/processors/base.py", "model/generator/mot/unified_mot.py", "configs/base/defaults/reasoner.py",
)
# Names outside checkpoint_db itself that lead into the checkpoint registry.
_REGISTRY_ROUTES = {"maybe_download_hf_model_from_s3", "load_language_model_safetensors", "GuardrailRunners",
                    "download_checkpoint", "checkpoint_hf"}
# Every function in those modules that can reach the registry, with the traced
# reason these loads skip it. None marks the one call they make.
_REGISTRY_CALLERS = {
    "scripts/action_policy_server_robolab.py:_resolve_checkpoint_path": "checkpoint path is a local directory",
    "inference/inference.py:_create": "local directory whose bundled processor serves the tokenizer node",
    "inference/common/inference.py:create": "the wrapper disables guardrails",
    "inference/common/args.py:download_checkpoint": "checkpoint_hf is unset for a local directory",
    "inference/common/args.py:_build_checkpoint": "sets checkpoint_hf only for registered checkpoint names",
    "model/generator/omni_mot_model.py:load_pretrained_model_if_needed": "training hook; pretrained weights off",
    "model/generator/omni_mot_model.py:_load_language_model": "training hook; pretrained weights off",
    "model/generator/tokenizers/wan2pt2_vae_4x16x16.py:_video_vae": None,
    "data/generator/processors/__init__.py:_download_llm_tokenizer": "text-only tokenizer types",
    "data/generator/processors/__init__.py:build_processor_lazy": "repository is unset once the node points at the checkpoint",
    "data/generator/processors/base.py:__init__": "the processor name is the checkpoint directory",
    "configs/base/defaults/reasoner.py:download_tokenizer_files": "only the Qwen2 tokenizer node calls it",
}
# Each snapshot's config.json, by the sha256 its manifest pins, configures the
# video tokenizer wan2pt2_vae_interface with this (bucket_name, vae_path).
_VIDEO_TOKENIZER = {
    "da6a23cbf4477aafda3e773874bf2c98d6869156e8a450d944e2f28c94eee00b":
        ("", "pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth"),
    "11577909f49e95d62456b6a497cf7b883f8987a3186d720d838ea5de0e493366":
        ("bucket", "pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth"),
}


def _registry_callers(modules):
    found = set()
    for name, module in modules.items():
        names = set(_REGISTRY_ROUTES)
        for node in ast.walk(module):
            if isinstance(node, ast.ImportFrom) and node.module == "cosmos_framework.utils.checkpoint_db":
                names |= {alias.asname or alias.name for alias in node.names}
        for function in (node for node in ast.walk(module) if isinstance(node, ast.FunctionDef)):
            if any(getattr(node, "id", getattr(node, "attr", None)) in names for node in ast.walk(function)):
                found.add(f"{name}:{function.name}")
    return found


def _registered_files(module):
    """s3 URI to CheckpointFileHf keywords, for each registration in register_checkpoints."""
    register = next(node for node in module.body
                    if isinstance(node, ast.FunctionDef) and node.name == "register_checkpoints")
    files = {}
    for call in ast.walk(register):
        if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "CheckpointConfig":
            fields = {keyword.arg: keyword.value for keyword in call.keywords}
            if getattr(fields["hf"].func, "id", None) == "CheckpointFileHf":
                uri = next(keyword.value for keyword in fields["s3"].keywords if keyword.arg == "uri")
                files[ast.literal_eval(uri)] = {keyword.arg: ast.literal_eval(keyword.value)
                                                for keyword in fields["hf"].keywords}
    return files


@pytest.mark.parametrize("candidate", CANDIDATES)
def test_every_registry_file_the_pinned_model_load_fetches_is_pinned_and_provisioned(tmp_path, candidate):
    # The pinned registry shells out to `uvx hf download`, which the GPU image
    # lacks and the offline server must not run, so a registry file that is not
    # pinned and provisioned stops the policy server during model load.
    selected = profile(candidate)
    stage_runtime(tmp_path, candidate)
    with zipfile.ZipFile(tmp_path / SOURCE_ARCHIVE) as archive:
        modules = {name: ast.parse(archive.read(_SOURCE + name)) for name in _LOAD_MODULES}
        registry = ast.unparse(ast.parse(archive.read(_SOURCE + "utils/checkpoint_db.py")))
        registered = _registered_files(ast.parse(archive.read(_SOURCE + "inference/common/checkpoints.py")))
    assert sorted(_registry_callers(modules) ^ set(_REGISTRY_CALLERS)) == [], "unaudited registry caller"
    # The tokenizer prefixes a bucket only when one is set, and _video_vae
    # resolves the result with download_checkpoint_v2. An s3 URI's bucket is
    # rewritten to "bucket"; a bare relative path is looked up under s3://bucket/.
    vae = ast.unparse(modules["model/generator/tokenizers/wan2pt2_vae_4x16x16.py"])
    assert "vae_path_full = f's3://{bucket_name}/{vae_path}' if bucket_name else vae_path" in vae
    assert "pretrained_path = download_checkpoint_v2(pretrained_path)" in vae
    assert "sanitize_uri(f's3://bucket/{checkpoint_uri}')" in registry
    # Every registry file download passes the argument list the server answers.
    assert ("cmd_args = [self.repository, '--repo-type', self.repository_type.value, '--revision', "
            "self.revision, self.filename]\n        path = _hf_download(cmd_args)") in registry
    manifest = selected.verified_snapshot_manifest(tmp_path / selected.SNAPSHOT_STAGED_NAME)
    config_sha256 = next(row["sha256"] for row in manifest["required_files"] if row["path"] == "config.json")
    reached = registered["s3://bucket/" + _VIDEO_TOKENIZER[config_sha256][1]]
    assert "repository_type" not in reached  # a model repository
    pinned = [{key: row[key] for key in ("repository", "revision", "filename")} for row in CHECKPOINT_REGISTRY_FILES]
    assert pinned == [reached]
    # Provisioned: inside the transfer bound and fetched where the server reads.
    assert CHECKPOINT_BYTES[candidate] == sum(row["size_bytes"] for row in
                                              [*manifest["required_files"], *CHECKPOINT_REGISTRY_FILES])
    assert runtime_contract(candidate)["checkpoint_registry_files"] == [dict(row) for row in CHECKPOINT_REGISTRY_FILES]
    for row, destination in checkpoint_registry_downloads(tmp_path / "store"):
        assert row["url"].startswith(f"https://huggingface.co/{row['repository']}/resolve/{row['revision']}/")
        assert destination == checkpoint_registry_path(tmp_path / "store", row)
        assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) and row["size_bytes"] > 0


def test_fixed_source_archive_cannot_gain_an_unverified_import(tmp_path):
    stage_runtime(tmp_path, CANDIDATES[1])
    source = extract_source(
        tmp_path / SOURCE_ARCHIVE, tmp_path / SOURCE_MANIFEST, tmp_path / "source"
    )
    assert (source / "cosmos_framework/scripts/action_policy_server_robolab.py").is_file()
    (source / "unexpected_policy_import.py").write_text("raise RuntimeError('unexpected source')\n")
    with pytest.raises(ValueError, match="members_changed"):
        verify_source_directory(source, tmp_path / SOURCE_MANIFEST)


def test_standalone_server_and_thin_native_client_have_distinct_exact_interpreters():
    manifest = verified_dependency_manifest()
    assert manifest["interpreter"]["version"] == "3.13.9"
    names = {row.get("package", row.get("name")) for row in manifest["server_wheels"]}
    assert {"flash-attn", "natten", "torch", "boto3", "webdataset"}.issubset(names)
    assert not {"jupyterlab", "ray", "lerobot", "megatron-core", "torchtitan"}.intersection(names)
    for minor, abi in [("3.11", "cp311"), ("3.12", "cp312")]:
        rows = client_wheels(manifest, minor)
        assert {row["package"] for row in rows} == {
            "openpi-client",
            "websockets",
            "msgpack",
            "typing-extensions",
        }
        assert abi in next(row["filename"] for row in rows if row["package"] == "msgpack")
    with pytest.raises(ValueError, match="interpreter_unsupported"):
        client_wheels(manifest, "3.13")


def test_runtime_rejects_too_small_nano_gpu_after_resealing():
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.policy_canary_session_bundle_identity import (
        validate_policy_canary_provider_manifest,
    )

    value = {
        "candidate_ids": [CANDIDATES[1]],
        "cosmos_runtime": runtime_contract(CANDIDATES[1]),
        "minimum_gpu_memory_gb": 48,
    }
    value["input_digest"] = canonical_digest(value, digest_field="input_digest")
    with pytest.raises(ValueError, match="cosmos_gpu_memory"):
        validate_policy_canary_provider_manifest(value)


def test_server_environment_cannot_import_isaac_or_use_ambient_hub_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", "/isaac-sim/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/isaac-sim")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/isaac-sim/kit:/usr/lib")
    monkeypatch.setenv("HF_TOKEN", "dummy-not-a-secret")
    env = server_environment(tmp_path / "runtime", tmp_path / "root")
    assert env["PYTHONPATH"] == str(tmp_path / "runtime")
    assert env["LD_LIBRARY_PATH"] == "/usr/lib"
    assert "PYTHONHOME" not in env and "HF_TOKEN" not in env
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"


@pytest.mark.parametrize("mutation", ["changed_bytes", "oversized_part", "missing_part"])
def test_source_parts_fit_privileged_leaf_cap_and_reject_changes(tmp_path, monkeypatch, mutation):
    import shutil
    from pathlib import Path
    from blueprint_pipeline import cosmos_droid_source as source
    vendor = Path(source.__file__).parent / "_vendor"
    target_vendor = tmp_path / "package" / "_vendor"
    target_vendor.mkdir(parents=True)
    for name in (*source.SOURCE_PARTS, source.SOURCE_MANIFEST):
        shutil.copyfile(vendor / name, target_vendor / name)
    assert len(source.SOURCE_PARTS) == 5
    assert all((target_vendor / name).stat().st_size <= 1024**2 for name in source.SOURCE_PARTS)
    monkeypatch.setattr(source, "__file__", str(target_vendor.parent / "cosmos_droid_source.py"))
    staged = tmp_path / "staged"
    staged.mkdir()
    source.stage_source(staged)
    assert (staged / source.SOURCE_ARCHIVE).stat().st_size == source.SOURCE_ARCHIVE_BYTES
    part = target_vendor / source.SOURCE_PARTS[0]
    if mutation == "changed_bytes":
        value = bytearray(part.read_bytes())
        value[0] ^= 1
        part.write_bytes(value)
    elif mutation == "oversized_part":
        with part.open("ab") as stream:
            stream.write(b"x")
    else:
        part.unlink()
    with pytest.raises(ValueError, match="source_archive_changed"):
        source.stage_source(staged)
