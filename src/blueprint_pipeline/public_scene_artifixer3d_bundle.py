"""Seal one exact-support ArtiFixer -> ArtiFixer3D -> ArtiFixer3D+ bundle.

The bundle contains only private-derived calibrated frames, masks, and a
deterministic COLMAP seed derived from the retained splat.  It deliberately
does not contain the publisher's raw dataset bytes or multi-gigabyte model
weights.  Released model artifacts are fetched on the worker at immutable
revisions and verified against this module's exact byte inventory before any
inference is permitted.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
from typing import Any, Mapping, Sequence
import zipfile

from .decision_evidence_contracts import canonical_digest, canonical_json
from .provider_bundle_rehearsal import rehearse_provider_bundle_entrypoint
from .public_scene_artifixer3d_candidate_inputs import (
    SCHEMA_VERSION as CANDIDATE_INPUT_SCHEMA,
)


SCHEMA_VERSION = "public_scene_artifixer3d_bundle.v1"
RUNTIME_REQUEST_SCHEMA_VERSION = "public_scene_artifixer3d_runtime_request.v1"
RUNTIME_RESULT_SCHEMA_VERSION = "public_scene_artifixer3d_runtime_result.v1"
ENTRYPOINT = "provider_runtime/run_public_scene_artifixer3d.sh"
RUNNER = "provider_runtime/public_scene_artifixer3d_runner.py"
MANIFEST = "provider_runtime/artifixer3d_bundle_manifest.json"
RUNTIME_REQUEST = "provider_runtime/artifixer3d_runtime_request.json"
INPUT_RECEIPT = "provider_runtime/input/public_scene_artifixer3d_candidate_inputs.v3.json"

ARTIFIXER_REPOSITORY = "https://github.com/nv-tlabs/ArtiFixer.git"
ARTIFIXER_COMMIT = "a392c4dfe17459ef9952407accdb9fcdcdddba98"
ARTIFIXER_TREE = "f9283bfe5e3a6cc160fd418f4e66412746a19a07"
ARTIFIXER_SUBMODULE_REPOSITORY = "https://github.com/nv-tlabs/3DGRUT-ArtiFixer.git"
ARTIFIXER_SUBMODULE_COMMIT = "62e1038b74b2edc01440fd4ddf5f080109b6faba"
ARTIFIXER_SUBMODULE_TREE = "494ecc2dd0834fcf71bf0124de152940e0c6d845"
ARTIFIXER_MODEL_REPOSITORY = "nvidia/ArtiFixer"
ARTIFIXER_MODEL_REVISION = "f96352ad72c84a628d5844b6543e94ae8c4479b3"
WAN_MODEL_REPOSITORY = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN_MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
QWEN_IMAGE_EDIT_REPOSITORY = "Qwen/Qwen-Image-Edit-2511"
QWEN_IMAGE_EDIT_REVISION = "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
QWEN_IMAGE_EDIT_LICENSE = "Apache-2.0"
DIRECT_EDITOR_BACKENDS = frozenset({"artifixer", "qwen_image_edit_2511"})
MODEL_LICENSE_URL = (
    "https://developer.download.nvidia.com/licenses/"
    "NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf"
)
MODEL_LICENSE_SHA256 = (
    "sha256:4ff203c3f7997c7fed287a463d733f794934a79cfabb2936008fca0bcc8ad3d6"
)
USE_ATTESTATION_SCHEMA_VERSION = (
    "public_scene_artifixer3d_noncommercial_use_attestation.v1"
)
DEFAULT_IMAGE = (
    "nvcr.io/nvidia/pytorch@"
    "sha256:0981807f1a51a156563e28b59dc2e7a9b5c1c7d85d1169d4965c5fd91fa38bcb"
)
ARTIFIXER_MODEL_FILES = (
    {
        "path": "artifixer-1.3b.pt",
        "size_bytes": 6_715_346_651,
        "sha256": "sha256:23e909fb4232c6a74a1c59eaf0ebfd419dd188e601aa0ab0145b9aaea821e059",
    },
)
WAN_RUNTIME_FILES = (
    {
        "path": "scheduler/scheduler_config.json",
        "size_bytes": 751,
        "sha256": "sha256:3fed2abbd9bbc301a74db01947198057ec5049808910dccab320925bf27bea6e",
    },
    {
        "path": "transformer/config.json",
        "size_bytes": 465,
        "sha256": "sha256:0b093fa072e9ff28763febe9b964ee582f566733a6d6709deb9dfba1bde16b81",
    },
    {
        "path": "vae/config.json",
        "size_bytes": 724,
        "sha256": "sha256:f0c1cc1d7decb5badc384f54691746a27a9aeff49f7ebca974e583389342d527",
    },
    {
        "path": "vae/diffusion_pytorch_model.safetensors",
        "size_bytes": 507_591_892,
        "sha256": "sha256:d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793",
    },
)
QWEN_IMAGE_EDIT_LARGE_FILES = (
    {
        "path": "text_encoder/model-00001-of-00004.safetensors",
        "size_bytes": 4_968_243_304,
        "sha256": "sha256:d725335e4ea2399be706469e4b8807716a8fa64bd03468252e9f7acf2415fee4",
    },
    {
        "path": "text_encoder/model-00002-of-00004.safetensors",
        "size_bytes": 4_991_495_816,
        "sha256": "sha256:b1830db6908dcc76df3a71492acbcf2b8cac130114cf1f3c2d9edae8de8c6de3",
    },
    {
        "path": "text_encoder/model-00003-of-00004.safetensors",
        "size_bytes": 4_932_751_040,
        "sha256": "sha256:09c1807c6d00d7cab94f7db39d4c02ebb8537225ccde383861ac48db97945aa6",
    },
    {
        "path": "text_encoder/model-00004-of-00004.safetensors",
        "size_bytes": 1_691_924_384,
        "sha256": "sha256:5dd068336d14d45ffb43cef374d286cc6ba9d8741b028f90a7d040d847961f4a",
    },
    {
        "path": "transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
        "size_bytes": 9_973_578_592,
        "sha256": "sha256:2a0c30c9ba44a5f11c21ca139e37951430bbde814ff4e0b5b1a68b80530e7a1a",
    },
    {
        "path": "transformer/diffusion_pytorch_model-00002-of-00005.safetensors",
        "size_bytes": 9_987_326_072,
        "sha256": "sha256:54ec249b07b4376e19cf16b764054f03ca03ae2cfbd9939453e2085f4e9bd259",
    },
    {
        "path": "transformer/diffusion_pytorch_model-00003-of-00005.safetensors",
        "size_bytes": 9_987_307_440,
        "sha256": "sha256:c55157843525653161e8f6af5acc670ba3aceff04284f7cf657199d24d065e16",
    },
    {
        "path": "transformer/diffusion_pytorch_model-00004-of-00005.safetensors",
        "size_bytes": 9_930_685_712,
        "sha256": "sha256:ffcfb5a4895702635890a67bad183591e0ae515d794bdcb26e217b27a7f6d12d",
    },
    {
        "path": "transformer/diffusion_pytorch_model-00005-of-00005.safetensors",
        "size_bytes": 982_130_472,
        "sha256": "sha256:2b2556b736629e10a5a0dfa14606f2057f4f81c2ba53f94103682c7ac42d4940",
    },
    {
        "path": "vae/diffusion_pytorch_model.safetensors",
        "size_bytes": 253_806_966,
        "sha256": "sha256:0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344",
    },
)
MAX_MEMBER_BYTES = 2 * 1024**3
MAX_TOTAL_BYTES = 4 * 1024**3
MAX_MEMBER_COUNT = 20_000


class ArtiFixer3DBundleError(ValueError):
    """Stable fail-closed bundle materialization errors."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtiFixer3DBundleError([code]) from exc
    if not isinstance(value, dict):
        raise ArtiFixer3DBundleError([code])
    return value


def _file(value: str | Path, *, code: str) -> Path:
    unresolved = Path(value).expanduser()
    if unresolved.is_symlink():
        raise ArtiFixer3DBundleError([code])
    path = unresolved.resolve()
    if not path.is_file():
        raise ArtiFixer3DBundleError([code])
    return path


def _record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    key = "relative_path" if root is not None else "path"
    value = path.relative_to(root).as_posix() if root is not None else str(path)
    return {
        key: value,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _portable(value: str) -> PurePosixPath:
    path = PurePosixPath(value.replace("\\", "/"))
    if (
        not value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ArtiFixer3DBundleError(["artifixer3d_bundle_member_path_invalid"])
    return path


def _candidate_receipt(path: Path) -> dict[str, Any]:
    value = _read(path, code="artifixer3d_bundle_candidate_receipt_unreadable")
    tasks = value.get("tasks")
    semantics = value.get("repair_target_semantics")
    if (
        value.get("schema_version") != CANDIDATE_INPUT_SCHEMA
        or value.get("status") != "candidate_inputs_prepared_no_model_no_execution"
        or value.get("receipt_digest")
        != canonical_digest(value, digest_field="receipt_digest")
        or not isinstance(tasks, list)
        or not 1 <= len(tasks) <= 5
        or value.get("replacement_object_count") != len(tasks)
        or not isinstance(semantics, Mapping)
        or semantics.get("source_washer_or_notebook_restoration_permitted") is not False
        or semantics.get("black_unknown_placeholder_preservation_permitted") is not False
        or semantics.get("outside_exact_support_changed_pixels_permitted") != 0
        or value.get("execution", {}).get("provider_mutations_performed") != 0
    ):
        raise ArtiFixer3DBundleError(
            ["artifixer3d_bundle_candidate_receipt_invalid"]
        )
    for task in tasks:
        if (
            not isinstance(task, Mapping)
            or task.get("artifixer3d_distillation", {}).get(
                "camera_partition_eligible"
            )
            is not True
            or task.get("artifixer3d_distillation", {}).get("execution_eligible")
            is not False
            or task.get("direct_prediction_coverage_indices")
            != list(range(int(task.get("camera_count") or 0)))
            or any(
                frame.get("outside_support_changed_pixels") != 0
                for frame in task.get("frames") or []
                if isinstance(frame, Mapping)
            )
        ):
            raise ArtiFixer3DBundleError(
                ["artifixer3d_bundle_candidate_task_invalid"]
            )
    return value


def _absolute_bound(record: Any, *, code: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, Mapping):
        raise ArtiFixer3DBundleError([code])
    path = _file(record.get("path") or "", code=code)
    if (
        path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise ArtiFixer3DBundleError([code])
    return path, _read(path, code=code)


def materialize_artifixer3d_use_attestation(
    *,
    candidate_inputs_receipt_path: str | Path,
    output_path: str | Path,
    authorized_by: str,
    authorization_kind: str = "explicit_user_direction_in_current_goal",
) -> dict[str, Any]:
    """Turn explicit noncommercial direction into a digest-bound authority file."""

    receipt_path = _file(
        candidate_inputs_receipt_path,
        code="artifixer3d_attestation_candidate_receipt_missing",
    )
    candidate = _candidate_receipt(receipt_path)
    authority_path, authority = _absolute_bound(
        candidate.get("execution_authority"),
        code="artifixer3d_attestation_execution_authority_invalid",
    )
    if (
        not authorized_by.strip()
        or authorization_kind != "explicit_user_direction_in_current_goal"
        or authority.get("authority_digest")
        != canonical_digest(authority, digest_field="authority_digest")
    ):
        raise ArtiFixer3DBundleError(["artifixer3d_attestation_authority_invalid"])
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise ArtiFixer3DBundleError(["artifixer3d_attestation_output_exists"])
    value: dict[str, Any] = {
        "schema_version": USE_ATTESTATION_SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "publisher_scene_id": candidate["publisher_scene_id"],
        "authorization_kind": authorization_kind,
        "authorized_by": authorized_by.strip(),
        "candidate_input_receipt": {
            **_record(receipt_path),
            "receipt_digest": candidate["receipt_digest"],
        },
        "parent_execution_authority": {
            **_record(authority_path),
            "authority_digest": authority["authority_digest"],
        },
        "artifixer_source_license": "Apache-2.0",
        "artifixer_model_license": {
            "name": "NVIDIA One-Way Noncommercial License",
            "url": MODEL_LICENSE_URL,
            "sha256": MODEL_LICENSE_SHA256,
        },
        "internal_noncommercial_research_and_development_only": True,
        "private_derived_input_upload_authorized": True,
        "raw_dataset_bytes_upload_authorized": False,
        "provider_training_authorized": False,
        "commercial_use_authorized": False,
        "redistribution_authorized": False,
        "publication_authorized": False,
        "simulator_or_generated_output_is_physical_evidence": False,
        "attestation_digest": "",
    }
    value["attestation_digest"] = canonical_digest(
        value, digest_field="attestation_digest"
    )
    _write_json(output, value)
    return value


def _validated_use_attestation(
    path: Path, *, candidate: Mapping[str, Any]
) -> dict[str, Any]:
    value = _read(path, code="artifixer3d_use_attestation_unreadable")
    license_row = value.get("artifixer_model_license")
    if (
        value.get("schema_version") != USE_ATTESTATION_SCHEMA_VERSION
        or value.get("attestation_digest")
        != canonical_digest(value, digest_field="attestation_digest")
        or value.get("publisher_scene_id") != candidate.get("publisher_scene_id")
        or value.get("candidate_input_receipt", {}).get("receipt_digest")
        != candidate.get("receipt_digest")
        or not isinstance(license_row, Mapping)
        or license_row.get("url") != MODEL_LICENSE_URL
        or license_row.get("sha256") != MODEL_LICENSE_SHA256
        or value.get("internal_noncommercial_research_and_development_only")
        is not True
        or value.get("private_derived_input_upload_authorized") is not True
        or value.get("raw_dataset_bytes_upload_authorized") is not False
        or value.get("provider_training_authorized") is not False
        or value.get("commercial_use_authorized") is not False
        or value.get("redistribution_authorized") is not False
        or value.get("publication_authorized") is not False
        or value.get("simulator_or_generated_output_is_physical_evidence") is not False
    ):
        raise ArtiFixer3DBundleError(["artifixer3d_use_attestation_invalid"])
    _absolute_bound(
        value.get("candidate_input_receipt"),
        code="artifixer3d_use_attestation_candidate_unbound",
    )
    _absolute_bound(
        value.get("parent_execution_authority"),
        code="artifixer3d_use_attestation_parent_unbound",
    )
    return value


def _git_value(root: Path, *args: str, code: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ArtiFixer3DBundleError([code]) from exc


def _repository_identity(root: Path) -> dict[str, Any]:
    commit = _git_value(root, "rev-parse", "HEAD", code="artifixer3d_repository_invalid")
    tree = _git_value(
        root, "rev-parse", "HEAD^{tree}", code="artifixer3d_repository_invalid"
    )
    dirty = _git_value(
        root, "status", "--porcelain=v1", code="artifixer3d_repository_invalid"
    )
    if dirty or len(commit) != 40 or len(tree) != 40:
        raise ArtiFixer3DBundleError(["artifixer3d_repository_not_clean_immutable"])
    return {"commit": commit, "tree": tree, "tracked_files_clean": True}


def _copy_source_release(source: Path, destination: Path) -> list[dict[str, Any]]:
    if (
        _git_value(source, "rev-parse", "HEAD", code="artifixer3d_source_invalid")
        != ARTIFIXER_COMMIT
        or _git_value(
            source, "rev-parse", "HEAD^{tree}", code="artifixer3d_source_invalid"
        )
        != ARTIFIXER_TREE
        or _git_value(
            source, "status", "--porcelain=v1", code="artifixer3d_source_invalid"
        )
    ):
        raise ArtiFixer3DBundleError(["artifixer3d_source_invalid"])
    names = _git_value(
        source, "ls-files", code="artifixer3d_source_index_invalid"
    ).splitlines()
    records: list[dict[str, Any]] = []
    for name in sorted(names):
        if name == "thirdparty/3DGRUT-ArtiFixer":
            continue
        relative = _portable(name)
        item = source.joinpath(*relative.parts)
        if not item.is_file() or item.is_symlink():
            raise ArtiFixer3DBundleError(["artifixer3d_source_member_invalid"])
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item, target)
        if _sha256(item) != _sha256(target):
            raise ArtiFixer3DBundleError(["artifixer3d_source_copy_invalid"])
        records.append(_record(target, root=destination))
    if not records:
        raise ArtiFixer3DBundleError(["artifixer3d_source_empty"])
    return records


def _copy_candidate_tree(source: Path, destination: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in sorted(source.rglob("*")):
        if item.is_dir() and not item.is_symlink():
            continue
        if not item.is_file() or item.is_symlink():
            raise ArtiFixer3DBundleError(
                ["artifixer3d_candidate_tree_member_invalid"]
            )
        relative = item.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(item, target)
        except OSError:
            shutil.copyfile(item, target)
        if _sha256(item) != _sha256(target):
            raise ArtiFixer3DBundleError(
                ["artifixer3d_candidate_tree_copy_invalid"]
            )
        records.append(_record(target, root=destination))
    if not records:
        raise ArtiFixer3DBundleError(["artifixer3d_candidate_tree_empty"])
    return records


def _zip_tree(source: Path, destination: Path) -> None:
    total = 0
    count = 0
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(source.rglob("*")):
            if item.is_dir() and not item.is_symlink():
                continue
            if not item.is_file() or item.is_symlink():
                raise ArtiFixer3DBundleError(
                    ["artifixer3d_bundle_archive_member_invalid"]
                )
            relative = item.relative_to(source).as_posix()
            size = item.stat().st_size
            total += size
            count += 1
            if (
                size > MAX_MEMBER_BYTES
                or total > MAX_TOTAL_BYTES
                or count > MAX_MEMBER_COUNT
            ):
                raise ArtiFixer3DBundleError(
                    ["artifixer3d_bundle_archive_limit_exceeded"]
                )
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            with item.open("rb") as input_stream, archive.open(info, "w") as output:
                shutil.copyfileobj(input_stream, output, length=1024 * 1024)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def build_artifixer3d_bundle(
    *,
    candidate_inputs_receipt_path: str | Path,
    use_attestation_path: str | Path,
    artifixer_source_directory: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
    allowed_active_instance_ids: Sequence[int] = (),
    artifixer3d_steps: int = 30_000,
    random_seed: int = 840_920,
    direct_editor_backend: str = "artifixer",
    semantic_editor_only: bool = False,
) -> dict[str, Any]:
    """Build and locally rehearse one immutable no-upload provider bundle."""

    receipt_path = _file(
        candidate_inputs_receipt_path,
        code="artifixer3d_bundle_candidate_receipt_missing",
    )
    candidate = _candidate_receipt(receipt_path)
    attestation_path = _file(
        use_attestation_path, code="artifixer3d_bundle_use_attestation_missing"
    )
    attestation = _validated_use_attestation(
        attestation_path, candidate=candidate
    )
    source = Path(artifixer_source_directory).expanduser().resolve()
    repo = Path(repository_root).expanduser().resolve()
    if (
        source.is_symlink()
        or not source.is_dir()
        or repo.is_symlink()
        or not (repo / "scripts" / "run_public_scene_artifixer3d.sh").is_file()
        or not (repo / "scripts" / "public_scene_artifixer3d_runner.py").is_file()
        or not isinstance(artifixer3d_steps, int)
        or isinstance(artifixer3d_steps, bool)
        or not 1 <= artifixer3d_steps <= 30_000
        or not isinstance(random_seed, int)
        or isinstance(random_seed, bool)
        or direct_editor_backend not in DIRECT_EDITOR_BACKENDS
        or not isinstance(semantic_editor_only, bool)
        or (semantic_editor_only and direct_editor_backend != "qwen_image_edit_2511")
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in allowed_active_instance_ids
        )
    ):
        raise ArtiFixer3DBundleError(["artifixer3d_bundle_configuration_invalid"])
    repository_identity = _repository_identity(repo)
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ArtiFixer3DBundleError(["artifixer3d_bundle_output_not_empty"])
    output.mkdir(parents=True, exist_ok=True)
    stage = output / "stage"
    runtime = stage / "provider_runtime"
    input_root = runtime / "input"
    source_root = runtime / "ArtiFixer_official"
    runtime.mkdir(parents=True)
    candidate_files = _copy_candidate_tree(receipt_path.parent, input_root)
    source_files = _copy_source_release(source, source_root)
    shutil.copyfile(repo / "scripts" / Path(ENTRYPOINT).name, runtime / Path(ENTRYPOINT).name)
    shutil.copyfile(repo / "scripts" / Path(RUNNER).name, runtime / Path(RUNNER).name)
    shutil.copyfile(attestation_path, runtime / "artifixer3d_use_attestation.json")

    runtime_request: dict[str, Any] = {
        "schema_version": RUNTIME_REQUEST_SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "publisher_scene_id": candidate["publisher_scene_id"],
        "candidate_input_receipt_digest": candidate["receipt_digest"],
        "replacement_object_count": candidate["replacement_object_count"],
        "task_ids": [task["task_id"] for task in candidate["tasks"]],
        "repair_target": (
            "plausible_object_free_background_inside_exact_support_only"
        ),
        "source_object_restoration_permitted": False,
        "outside_exact_support_changed_pixels_permitted": 0,
        "artifixer": {
            "repository": ARTIFIXER_REPOSITORY,
            "commit": ARTIFIXER_COMMIT,
            "tree": ARTIFIXER_TREE,
            "submodule_repository": ARTIFIXER_SUBMODULE_REPOSITORY,
            "submodule_commit": ARTIFIXER_SUBMODULE_COMMIT,
            "submodule_tree": ARTIFIXER_SUBMODULE_TREE,
        },
        "model": {
            "repository": ARTIFIXER_MODEL_REPOSITORY,
            "revision": ARTIFIXER_MODEL_REVISION,
            "files": list(ARTIFIXER_MODEL_FILES),
            "license": "NVIDIA One-Way Noncommercial License",
            "use_scope": "internal_noncommercial_research_and_development_only",
        },
        "wan_base": {
            "repository": WAN_MODEL_REPOSITORY,
            "revision": WAN_MODEL_REVISION,
            "files": list(WAN_RUNTIME_FILES),
            "omitted_unneeded_components": ["text_encoder", "tokenizer", "transformer_weights"],
        },
        "container_image": DEFAULT_IMAGE,
        "blueprint_source_identity": repository_identity,
        "use_attestation": {
            "schema_version": USE_ATTESTATION_SCHEMA_VERSION,
            "attestation_digest": attestation["attestation_digest"],
        },
        "direct_inference": {
            "checkpoint_filename": "artifixer-1.3b.pt",
            "inference_pipeline": "kv_cache",
            "num_inference_steps": 4,
            "frames_per_block": 7,
            "neighbor_selection_mode": "evenly_spaced",
            "fold_policy": "two_complementary_target_folds",
            "prompt": "unconditioned_zero_embedding",
        },
        "artifixer3d": {
            "steps": artifixer3d_steps,
            "config_name": "apps/colmap_3dgut_sparse_mcmc_lpips",
            "use_wandb": False,
        },
        "random_seed": random_seed,
        "direct_editor_backend": direct_editor_backend,
        "phases": [
            (
                "semantic_editor_inference"
                if direct_editor_backend == "qwen_image_edit_2511"
                else "direct_inference"
            ),
            "exact_support_composite",
            "artifixer3d_distillation",
            "artifixer3d_render",
            "artifixer3d_plus_inference",
            "artifixer3d_plus_exact_support_composite",
        ],
        "runtime_request_digest": "",
    }
    if direct_editor_backend == "qwen_image_edit_2511":
        runtime_request["semantic_editor"] = {
            "backend": direct_editor_backend,
            "repository": QWEN_IMAGE_EDIT_REPOSITORY,
            "revision": QWEN_IMAGE_EDIT_REVISION,
            "license": QWEN_IMAGE_EDIT_LICENSE,
            "large_files": list(QWEN_IMAGE_EDIT_LARGE_FILES),
            "diffusers_version": "0.37.1",
            "num_inference_steps": 40,
            "true_cfg_scale": 4.0,
            "guidance_scale": 1.0,
            "enable_model_cpu_offload": True,
            "input_role": "black_exact_mask_hole_with_original_context",
            "prompt_policy": "generic_object_absent_background_completion_v1",
            "output_must_be_exact_support_composited": True,
        }
        runtime_request["semantic_editor_only"] = semantic_editor_only
        if semantic_editor_only:
            runtime_request["phases"] = [
                "semantic_editor_inference",
                "exact_support_composite",
                "external_visual_and_multiview_review",
            ]
    runtime_request["runtime_request_digest"] = canonical_digest(
        runtime_request, digest_field="runtime_request_digest"
    )
    _write_json(runtime / Path(RUNTIME_REQUEST).name, runtime_request)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "sealed_no_upload_no_execution",
        "entrypoint": ENTRYPOINT,
        "runner": RUNNER,
        "runtime_request": {
            **_record(runtime / Path(RUNTIME_REQUEST).name, root=stage),
            "runtime_request_digest": runtime_request["runtime_request_digest"],
        },
        "candidate_input_receipt": {
            **_record(input_root / receipt_path.name, root=stage),
            "receipt_digest": candidate["receipt_digest"],
        },
        "candidate_files": candidate_files,
        "source_files": source_files,
        "source_identity": runtime_request["artifixer"],
        "model_identity": runtime_request["model"],
        "wan_base_identity": runtime_request["wan_base"],
        "direct_editor_backend": direct_editor_backend,
        "semantic_editor_only": semantic_editor_only,
        "semantic_editor_model_identity": runtime_request.get("semantic_editor"),
        "container_image": DEFAULT_IMAGE,
        "blueprint_source_identity": repository_identity,
        "use_attestation": {
            **_record(runtime / "artifixer3d_use_attestation.json", root=stage),
            "attestation_digest": attestation["attestation_digest"],
        },
        "allowed_active_instance_ids": sorted(set(allowed_active_instance_ids)),
        "contains_raw_dataset_bytes": False,
        "contains_private_derived_inputs": True,
        "contains_model_weights": False,
        "provider_mutations_performed": 0,
        "expected_runtime_result_schema": RUNTIME_RESULT_SCHEMA_VERSION,
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    _write_json(runtime / Path(MANIFEST).name, manifest)
    bundle_path = output / "public_scene_artifixer3d_provider_bundle.zip"
    _zip_tree(stage, bundle_path)
    rehearsal = rehearse_provider_bundle_entrypoint(
        bundle_path=bundle_path,
        entrypoint_relative_path=ENTRYPOINT,
        evidence_path=output / "artifixer3d_exact_bundle_rehearsal.json",
    )
    if (
        rehearsal.get("status") != "passed"
        or rehearsal.get("provider_mutations_performed") != 0
    ):
        raise ArtiFixer3DBundleError(["artifixer3d_bundle_rehearsal_failed"])
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "sealed_rehearsal_passed_no_upload_no_execution",
        "bundle": _record(bundle_path),
        "manifest_digest": manifest["manifest_digest"],
        "runtime_request_digest": runtime_request["runtime_request_digest"],
        "candidate_input_receipt_digest": candidate["receipt_digest"],
        "replacement_object_count": candidate["replacement_object_count"],
        "task_ids": runtime_request["task_ids"],
        "direct_editor_backend": direct_editor_backend,
        "semantic_editor_only": semantic_editor_only,
        "container_image": DEFAULT_IMAGE,
        "blueprint_source_identity": repository_identity,
        "use_attestation_digest": attestation["attestation_digest"],
        "allowed_active_instance_ids": manifest["allowed_active_instance_ids"],
        "local_rehearsal": rehearsal,
        "provider_mutations_performed": 0,
        "private_derived_upload_performed": False,
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(
        result, digest_field="receipt_digest"
    )
    receipt = output / "public_scene_artifixer3d_bundle_receipt.json"
    _write_json(receipt, result)
    return result


__all__ = [
    "ARTIFIXER_COMMIT",
    "ARTIFIXER_MODEL_FILES",
    "ARTIFIXER_MODEL_REVISION",
    "ARTIFIXER_SUBMODULE_COMMIT",
    "ARTIFIXER_SUBMODULE_TREE",
    "ARTIFIXER_TREE",
    "ArtiFixer3DBundleError",
    "DEFAULT_IMAGE",
    "DIRECT_EDITOR_BACKENDS",
    "QWEN_IMAGE_EDIT_LARGE_FILES",
    "QWEN_IMAGE_EDIT_REPOSITORY",
    "QWEN_IMAGE_EDIT_REVISION",
    "SCHEMA_VERSION",
    "WAN_MODEL_REVISION",
    "WAN_RUNTIME_FILES",
    "build_artifixer3d_bundle",
    "materialize_artifixer3d_use_attestation",
]
