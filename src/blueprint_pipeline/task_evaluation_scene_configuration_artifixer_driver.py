"""Bridge a generic scene-configuration stage to released ArtiFixer runtimes.

This module contains no image-editing or reconstruction algorithm.  It converts
the Website-bound scene/camera/mask contract into the existing semantic-teacher,
paired-target ArtiFixer3D, native-export, and independent-review contracts, then
runs those released implementations inside the already allocated parent GPU.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess  # nosec B404 - package and entrypoint are digest-bound
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from PIL import Image

from .decision_evidence_contracts import canonical_digest, canonical_json
from .fresh_scene_semantic_teacher_image_edit import (
    PROMPT_POLICY,
    REQUEST_SCHEMA_VERSION,
    RIGHTS_SCHEMA_VERSION as SEMANTIC_RIGHTS_SCHEMA_VERSION,
    materialize_semantic_teacher_image_edit_packet,
)
from .provider_archive import extract_provider_archive
from .public_scene_artifixer3d_bundle import (
    DUAL_TARGET_PIPELINE_MODE,
    build_artifixer3d_bundle,
    materialize_artifixer3d_use_attestation,
)
from .public_scene_artifixer3d_candidate_inputs import (
    materialize_artifixer3d_candidate_inputs,
)
from .public_scene_artifixer3d_dual_target_inputs import (
    materialize_dual_target_artifixer3d_inputs,
    materialize_whole_frame_semantic_teacher_receipt,
)
from .semantic_teacher_image_edit_worker import (
    RUNTIME_REQUEST_SCHEMA_VERSION as SEMANTIC_RUNTIME_REQUEST_SCHEMA_VERSION,
    execute_semantic_teacher_image_edits,
)
from .task_evaluation_artifixer_ai_visual_review import (
    DUAL_TARGET_REVIEW_SCHEMA_VERSION,
    materialize_artifixer_ai_visual_review_rights,
    run_artifixer_ai_visual_review,
)
from .task_evaluation_scene_configuration_component_package import (
    SCHEMA_VERSION as COMPONENT_PACKAGE_SCHEMA_VERSION,
)
from .task_evaluation_scene_configuration_disclosure import (
    PENDING_PROVIDER_RENDER_STATUS,
)
from .task_evaluation_scene_configuration_render_inputs import (
    complete_provider_render_inputs,
)
from .task_evaluation_scene_configuration_openai_gate import (
    scene_configuration_openai_stage_gate,
    scene_configuration_openai_stage_scope,
)
from .task_evaluation_scene_configuration_render_handoff import (
    materialize_provider_render_handoff,
)
from .task_evaluation_scene_configuration_stage_tool import (
    COMPONENT_RESULT_SCHEMA_VERSION,
)


_INPUT_ENV = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT"
_DEPENDENCIES_ENV = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES"
_OUTPUT_ENV = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT"
_RESULT_ENV = "BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_RESULT"
_PACKAGE_ENV = "BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_ROOT"
_ADAPTER_ID = "artifixer3d_observed_object_removal"
_VISUAL_REVIEW_COST_SCOPE = (
    "task_evaluation_scene_configuration_artifixer_visual_review"
)
_SEMANTIC_BACKEND_ID = "openai_gpt_image_2_2026_04_21_semantic_teacher"


class TaskEvaluationSceneConfigurationArtifixerError(RuntimeError):
    """The released ArtiFixer chain could not satisfy the generic stage."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationArtifixerError(code) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise TaskEvaluationSceneConfigurationArtifixerError(code)
    return dict(value)


def _required_path(environment: Mapping[str, str], name: str) -> Path:
    value = str(environment.get(name) or "").strip()
    if not value:
        raise TaskEvaluationSceneConfigurationArtifixerError(
            f"scene_configuration_artifixer_environment_missing:{name}"
        )
    return Path(value).expanduser().resolve()


def _artifixer_tuning(configuration: Mapping[str, Any]) -> dict[str, int]:
    """Resolve nullable website tuning before any paid semantic edit."""

    supplied = {
        "transition_radius_pixels": configuration.get("transition_radius_pixels"),
        "artifixer3d_steps": configuration.get("artifixer3d_steps"),
        "random_seed": configuration.get("random_seed"),
    }
    defaults = {
        "transition_radius_pixels": 3,
        "artifixer3d_steps": 30_000,
        "random_seed": 839_873,
    }
    resolved = {
        name: defaults[name] if value is None else value
        for name, value in supplied.items()
    }
    if (
        isinstance(resolved["transition_radius_pixels"], bool)
        or not isinstance(resolved["transition_radius_pixels"], int)
        or resolved["transition_radius_pixels"] < 0
        or isinstance(resolved["artifixer3d_steps"], bool)
        or not isinstance(resolved["artifixer3d_steps"], int)
        or not 1 <= resolved["artifixer3d_steps"] <= 30_000
        or isinstance(resolved["random_seed"], bool)
        or not isinstance(resolved["random_seed"], int)
    ):
        raise TaskEvaluationSceneConfigurationArtifixerError(
            "scene_configuration_artifixer_tuning_invalid"
        )
    return resolved


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _component_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "digest": _sha256(path),
    }


def _materialized(envelope: Mapping[str, Any], contract_path: str) -> tuple[dict[str, Any], Path]:
    rows = [
        row
        for row in envelope.get("materialized_references") or []
        if isinstance(row, Mapping) and row.get("contract_path") == contract_path
    ]
    if len(rows) != 1:
        raise TaskEvaluationSceneConfigurationArtifixerError(
            f"scene_configuration_artifixer_reference_missing:{contract_path}"
        )
    row = dict(rows[0])
    path = Path(str(row.get("materialized_path") or "")).expanduser().resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != row.get("size_bytes")
        or _sha256(path) != row.get("digest")
        or row.get("full_byte_service_account_readback_passed") is not True
    ):
        raise TaskEvaluationSceneConfigurationArtifixerError(
            f"scene_configuration_artifixer_reference_invalid:{contract_path}"
        )
    return row, path


def _human_authority(configuration: Mapping[str, Any]) -> dict[str, Any]:
    value = configuration.get("human_authority")
    if (
        not isinstance(value, Mapping)
        or not str(value.get("accepted_by") or "").strip()
        or not str(value.get("accepted_on") or "").strip()
        or not str(value.get("authority_reference") or "").strip()
        or value.get("private_derived_frame_disclosure_authorized") is not True
        or value.get("provider_retention_terms_accepted") is not True
        or value.get("provider_training_terms_accepted") is not True
        or value.get("provider_training_authorized") is not False
    ):
        raise TaskEvaluationSceneConfigurationArtifixerError(
            "scene_configuration_artifixer_human_authority_invalid"
        )
    return dict(value)


def _write_execution_authority(
    *,
    envelope: Mapping[str, Any],
    configuration: Mapping[str, Any],
    destination: Path,
) -> tuple[dict[str, Any], Path, str]:
    human = _human_authority(configuration)
    rights_row, rights_path = _materialized(envelope, "scene.rights.admission")
    rights = _read(rights_path, code="scene_configuration_artifixer_rights_admission_invalid")
    publisher_scene_id = str(rights.get("publisher_scene_id") or rights.get("scene_id") or "")
    if (
        not publisher_scene_id
        or rights.get("status") != "admitted_for_internal_development"
        or rights.get("private_provider_processing_allowed") is not True
        or rights.get("provider_training_allowed") is not False
        or rights.get("public_redistribution_allowed") is not False
    ):
        raise TaskEvaluationSceneConfigurationArtifixerError(
            "scene_configuration_artifixer_rights_admission_invalid"
        )
    authority: dict[str, Any] = {
        "schema_version": "third_scene_dual_task_execution_authority.v1",
        "program_id": "arm-decision-proof-v1",
        "publisher_scene_id": publisher_scene_id,
        "authority_kind": "website_human_scene_configuration_authority",
        "authorized_by": human["accepted_by"],
        "accepted_on": human["accepted_on"],
        "authority_reference": human["authority_reference"],
        "source_rights_admission": {
            **_record(rights_path),
            "admission_digest": rights_row["digest"],
        },
        "terms": {
            "internal_noncommercial_research_and_development_only": True,
            "private_derived_frame_disclosure_authorized": True,
            "provider_retention_terms_accepted": True,
            "provider_training_terms_accepted": True,
            "provider_training_authorized": False,
            "raw_source_bytes_disclosure_authorized": False,
            "public_redistribution_authorized": False,
        },
        "authority_digest": "",
    }
    authority["authority_digest"] = canonical_digest(authority, digest_field="authority_digest")
    destination.write_text(canonical_json(authority) + "\n", encoding="utf-8")
    return authority, rights_path, publisher_scene_id


def _materialize_preflight(
    *,
    envelope: Mapping[str, Any],
    configuration: Mapping[str, Any],
    authority: Mapping[str, Any],
    authority_path: Path,
    output_path: Path,
) -> tuple[dict[str, Any], str]:
    render = envelope.get("render_inputs_result")
    source_object = configuration.get("source_object")
    if not isinstance(render, Mapping) or not isinstance(source_object, Mapping):
        raise TaskEvaluationSceneConfigurationArtifixerError(
            "scene_configuration_artifixer_render_inputs_invalid"
        )
    calibration_path = Path(str((render.get("camera_calibration") or {}).get("path") or ""))
    calibration_rows = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibrations = {
        str(row.get("id") or ""): row for row in calibration_rows if isinstance(row, Mapping)
    }
    task_id = "remove-source-object-" + str(source_object["publisher_instance_id"])
    camera_inputs: list[dict[str, Any]] = []
    for row in render.get("derived_frames") or []:
        camera_id = str(row.get("camera_id") or "") if isinstance(row, Mapping) else ""
        frame = Path(str((row or {}).get("path") or "")).resolve()
        mask_row = (row or {}).get("source_object_mask") or {}
        mask = Path(str(mask_row.get("path") or "")).resolve()
        calibration = calibrations.get(camera_id)
        if not camera_id or calibration is None or not frame.is_file() or not mask.is_file():
            raise TaskEvaluationSceneConfigurationArtifixerError(
                "scene_configuration_artifixer_render_inputs_invalid"
            )
        with Image.open(mask) as image:
            pixel_count = sum(1 for value in image.convert("L").getdata() if value > 0)
        camera_inputs.append(
            {
                "task_id": task_id,
                "camera_id": camera_id,
                "calibration": calibration,
                "retained_scene_before": _record(frame),
                "exact_residual_mask": {**_record(mask), "pixel_count": pixel_count},
            }
        )
    retained_row = (render.get("derived_gaussian_cutout") or {}).get(
        "retained_scene_without_source_object"
    ) or {}
    retained = Path(str(retained_row.get("path") or "")).resolve()
    if not camera_inputs or not retained.is_file():
        raise TaskEvaluationSceneConfigurationArtifixerError(
            "scene_configuration_artifixer_render_inputs_invalid"
        )
    preflight: dict[str, Any] = {
        "schema_version": "public_scene_calibrated_exact_segment_repair_preflight.v1",
        "status": "prepared_no_upload_no_execution",
        "replacement_object_count": 1,
        "lanes": [{"task_id": task_id}],
        "execution": {
            "provider_mutations_performed": 0,
            "aura_inpainting_executed": False,
        },
        "required_result_checks": {
            "outside_mask_pixel_delta_required": 0,
            "locality_mask_dilation_pixels": 0,
        },
        "backend_admission": {
            "execution_authority": {
                **_record(authority_path),
                "authority_digest": authority["authority_digest"],
            }
        },
        "shared_retained_scene": {
            **_record(retained),
            "retained_gaussian_count": (render.get("derived_gaussian_cutout") or {}).get(
                "retained_count"
            ),
        },
        "camera_inputs": camera_inputs,
        "preflight_digest": "",
    }
    preflight["preflight_digest"] = canonical_digest(preflight, digest_field="preflight_digest")
    output_path.write_text(canonical_json(preflight) + "\n", encoding="utf-8")
    return preflight, task_id


def _semantic_rights_and_request(
    *,
    candidate: Mapping[str, Any],
    candidate_path: Path,
    registry_path: Path,
    configuration: Mapping[str, Any],
    publisher_scene_id: str,
    output_root: Path,
) -> Path:
    human = _human_authority(configuration)
    registry = _read(
        registry_path,
        code="scene_configuration_artifixer_backend_registry_invalid",
    )
    matches = [
        row
        for row in registry.get("backends") or []
        if isinstance(row, Mapping) and row.get("backend_id") == _SEMANTIC_BACKEND_ID
    ]
    if len(matches) != 1:
        raise TaskEvaluationSceneConfigurationArtifixerError(
            "scene_configuration_artifixer_backend_registry_invalid"
        )
    backend = matches[0]
    execution = backend["execution"]
    backend_digest = canonical_digest(backend)
    rights: dict[str, Any] = {
        "schema_version": SEMANTIC_RIGHTS_SCHEMA_VERSION,
        "status": "accepted_for_private_derived_semantic_edit",
        "source_candidate_inputs_receipt_digest": candidate["receipt_digest"],
        "publisher_scene_id": publisher_scene_id,
        "backend_id": _SEMANTIC_BACKEND_ID,
        "backend_entry_digest": backend_digest,
        "provider_id": execution["provider_id"],
        "model_snapshot": execution["model_snapshot"],
        "raw_nonredistributable_source_bytes_included": False,
        "private_derived_frame_disclosure_authorized": True,
        "provider_retention_terms_accepted": True,
        "provider_training_terms_accepted": True,
        "issued_by_agent": False,
        "accepted_by": human["accepted_by"],
        "accepted_on": human["accepted_on"],
        "human_authority_reference": human["authority_reference"],
        "attestation_digest": "",
    }
    rights["attestation_digest"] = canonical_digest(rights, digest_field="attestation_digest")
    rights_path = output_root / "semantic_teacher_rights.v1.json"
    rights_path.write_text(canonical_json(rights) + "\n", encoding="utf-8")
    request: dict[str, Any] = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "source_candidate_inputs_receipt_path": str(candidate_path),
        "backend_registry_path": str(registry_path),
        "backend_id": _SEMANTIC_BACKEND_ID,
        "rights_attestation_path": str(rights_path),
        "selected_task_ids": [candidate["tasks"][0]["task_id"]],
        "prompt_policy": PROMPT_POLICY,
        "output_format": "png",
        "retry_count": 0,
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    packet_root = output_root / "semantic_teacher_packet"
    materialize_semantic_teacher_image_edit_packet(request=request, output_root=packet_root)
    return packet_root


def _semantic_runtime_request(
    *, packet_root: Path, source_commit: str, maximum_cost_usd: float | None = None
) -> Path:
    packet = _read(
        packet_root / "fresh_scene_semantic_teacher_image_edit_packet.v1.json",
        code="scene_configuration_artifixer_semantic_packet_invalid",
    )
    tasks = []
    for task in packet["tasks"]:
        tasks.append(
            {
                "task_id": task["task_id"],
                "frames": [
                    {
                        "frame_index": frame["frame_index"],
                        "camera_id": frame["camera_id"],
                        "input_rgb": frame["staged_input_rgb"],
                        "edit_mask": frame["staged_edit_mask"],
                    }
                    for frame in task["frames"]
                ],
            }
        )
    request: dict[str, Any] = {
        "schema_version": SEMANTIC_RUNTIME_REQUEST_SCHEMA_VERSION,
        "source_commit_sha": source_commit,
        "source_packet_digest": packet["packet_digest"],
        "backend": {
            "registry_entry": packet["backend"]["registry_entry"],
            "backend_entry_digest": packet["backend"]["backend_entry_digest"],
            "execution": packet["backend"]["execution"],
        },
        "prompt_policy": packet["backend"]["prompt_policy"],
        "prompt": packet["backend"]["prompt"],
        "tasks": tasks,
        "max_parallel_requests": 2,
        # The stage's own cap, so the worker can stop issuing frame requests
        # once the observed spend would carry past it. Without this the cap is
        # only checked at settlement, two days after the money is gone: run
        # ...4dfc5f8e-r3-web-20260827T050053Z billed $0.877128 against a $0.40
        # reservation and nothing refused it.
        "maximum_cost_usd": maximum_cost_usd,
        "retry_count": 0,
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    path = packet_root / "semantic_teacher_image_edit_runtime_request.v1.json"
    path.write_text(canonical_json(request) + "\n", encoding="utf-8")
    return path


def _stage_openai_token(environment: Mapping[str, str], *, stage: str) -> str:
    """Read one stage's exclusive OpenAI key so per-stage attribution holds."""

    scope = scene_configuration_openai_stage_scope(environment, stage=stage)
    token_path = Path(scope["api_key_file"]).expanduser()
    if token_path.is_symlink() or not token_path.is_file() or (
        token_path.stat().st_mode & 0o077
    ):
        raise TaskEvaluationSceneConfigurationArtifixerError(
            "scene_configuration_artifixer_secret_file_invalid"
        )
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise TaskEvaluationSceneConfigurationArtifixerError(
            "scene_configuration_artifixer_secret_file_invalid"
        )
    return token


@contextmanager
def _temporary_openai_key(token: str):
    previous = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = token
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = previous


def execute_artifixer_component(
    *,
    environment: Mapping[str, str] | None = None,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    """Run the released production chain once inside its parent GPU."""

    values = dict(os.environ if environment is None else environment)
    stage_input = _read(
        _required_path(values, _INPUT_ENV),
        code="scene_configuration_artifixer_input_invalid",
    )
    dependencies = json.loads(_required_path(values, _DEPENDENCIES_ENV).read_text(encoding="utf-8"))
    stage = stage_input.get("stage") or {}
    configuration = stage_input.get("configuration") or {}
    envelope = stage_input.get("construction_envelope") or {}
    if (
        stage.get("adapter", {}).get("id") != _ADAPTER_ID
        or configuration.get("schema_version")
        != "observed_appearance_object_removal_configuration.v1"
        or dependencies != []
    ):
        raise TaskEvaluationSceneConfigurationArtifixerError(
            "scene_configuration_artifixer_input_invalid"
        )
    tuning = _artifixer_tuning(configuration)
    output_root = _required_path(values, _OUTPUT_ENV)
    package_root = _required_path(values, _PACKAGE_ENV)
    component_result_path = _required_path(values, _RESULT_ENV)
    work = output_root / "released_artifixer_runtime"
    work.mkdir(mode=0o700)
    authority, rights_path, publisher_scene_id = _write_execution_authority(
        envelope=envelope,
        configuration=configuration,
        destination=work / "execution_authority.v1.json",
    )
    # A rights-admitted scene arrives with its render still owed. Finish it
    # here, on the GPU this stage already occupies, before anything reads
    # the frames.
    render_inputs = envelope.get("render_inputs_result")
    if (
        isinstance(render_inputs, Mapping)
        and render_inputs.get("status") == PENDING_PROVIDER_RENDER_STATUS
    ):
        # The bundle records these paths relative to the provider *runtime*
        # root ("input/render/source_appearance.ply"), not to this component's
        # package directory. Joining them to package_root looks under
        # toolchain/components/<name>/package/, where the staged appearance
        # has never existed.
        runtime_root = Path(
            os.environ.get("BLUEPRINT_SCENE_CONFIGURATION_RUNTIME_ROOT")
            or Path(__file__).resolve().parents[1]
        )
        appearance = runtime_root / str(
            (render_inputs.get("source_appearance") or {}).get("path") or ""
        )
        envelope = {
            **envelope,
            "render_inputs_result": complete_provider_render_inputs(
                render_inputs=render_inputs,
                appearance_path=appearance,
                source_object=configuration["source_object"],
                output_root=work / "provider_render",
                input_root=runtime_root,
            ),
        }
    render_handoff = materialize_provider_render_handoff(
        render_inputs=envelope["render_inputs_result"],
        output_root=output_root,
    )
    _preflight, task_id = _materialize_preflight(
        envelope=envelope,
        configuration=configuration,
        authority=authority,
        authority_path=work / "execution_authority.v1.json",
        output_path=work / "calibrated_preflight.v1.json",
    )
    candidate_root = work / "candidate_inputs"
    candidate = materialize_artifixer3d_candidate_inputs(
        calibrated_residual_preflight_path=work / "calibrated_preflight.v1.json",
        output_root=candidate_root,
    )
    candidate_path = candidate_root / "public_scene_artifixer3d_candidate_inputs.v3.json"
    registry_path = (
        package_root
        / "blueprint_runtime/docs/arm_decision_proof_v1/manifests/image_editor_backends.v1.json"
    )
    packet_root = _semantic_rights_and_request(
        candidate=candidate,
        candidate_path=candidate_path,
        registry_path=registry_path,
        configuration=configuration,
        publisher_scene_id=publisher_scene_id,
        output_root=work,
    )
    token = _stage_openai_token(values, stage="artifixer_semantic_teacher")
    semantic_output = work / "semantic_teacher_output"
    semantic_cap_raw = values.get(
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_MAX_COST_USD"
    )
    semantic_cap = float(semantic_cap_raw) if semantic_cap_raw else None
    semantic_request = _semantic_runtime_request(
        packet_root=packet_root,
        source_commit=str(stage_input["source_commit"]),
        maximum_cost_usd=semantic_cap,
    )
    semantic_cost_gate = scene_configuration_openai_stage_gate(
        environment=values,
        stage="artifixer_semantic_teacher",
        run_id=f"{stage_input['run_id']}-artifixer-semantic-teacher",
        request_digest=_sha256(semantic_request),
        candidate_digest=str(candidate["receipt_digest"]),
        output_root=work / "semantic_teacher_official_openai_cost",
    )
    semantic_cost_gate.reserve()
    try:
        semantic_result = execute_semantic_teacher_image_edits(
            runtime_request_path=semantic_request,
            output_root=semantic_output,
            token=token,
        )
    except Exception as exc:
        semantic_cost_gate.complete(
            provider_call_performed=True,
            runtime_result_digest=None,
            runtime_exception_type=type(exc).__name__,
        )
        raise
    semantic_cost_gate.complete(
        provider_call_performed=True,
        runtime_result_digest=str(semantic_result.get("result_digest") or "") or None,
        runtime_exception_type=None,
    )
    if semantic_result.get("status") != "completed_unreviewed_semantic_teacher_candidates":
        raise TaskEvaluationSceneConfigurationArtifixerError(
            "scene_configuration_artifixer_semantic_teacher_failed"
        )
    teacher_receipt_path = work / "whole_frame_semantic_teacher.v1.json"
    materialize_whole_frame_semantic_teacher_receipt(
        source_candidate_inputs_receipt_path=candidate_path,
        task_id=task_id,
        semantic_teacher_frames_root=semantic_output / "tasks" / task_id,
        editor_identity={
            "backend_id": semantic_result["backend_id"],
            "model_snapshot": semantic_result["model_snapshot"],
            "result_digest": semantic_result["result_digest"],
        },
        prompt_policy=PROMPT_POLICY,
        output_path=teacher_receipt_path,
    )
    dual_root = work / "dual_target_inputs"
    materialize_dual_target_artifixer3d_inputs(
        source_candidate_inputs_receipt_path=candidate_path,
        semantic_teacher_receipt_paths=[teacher_receipt_path],
        output_root=dual_root,
        transition_radius_pixels=tuning["transition_radius_pixels"],
    )
    dual_path = dual_root / "public_scene_artifixer3d_dual_target_inputs.v1.json"
    use_attestation_path = work / "artifixer3d_use_attestation.v1.json"
    materialize_artifixer3d_use_attestation(
        candidate_inputs_receipt_path=dual_path,
        output_path=use_attestation_path,
        authorized_by=_human_authority(configuration)["accepted_by"],
    )
    package_manifest = _read(
        package_root / f"{COMPONENT_PACKAGE_SCHEMA_VERSION}.json",
        code="scene_configuration_artifixer_package_invalid",
    )
    blueprint_receipt = _read(
        package_root / "blueprint_source_receipt.json",
        code="scene_configuration_artifixer_package_invalid",
    )
    bundle_root = work / "artifixer_bundle"
    bundle = build_artifixer3d_bundle(
        candidate_inputs_receipt_path=dual_path,
        use_attestation_path=use_attestation_path,
        artifixer_source_directory=package_root / "artifixer_source",
        artifixer_source_receipt_path=package_root / "artifixer_source_receipt.json",
        output_root=bundle_root,
        repository_root=package_root / "blueprint_runtime",
        blueprint_source_identity={
            "commit": stage_input["source_commit"],
            "tree": blueprint_receipt["tree"],
            "tracked_files_clean": True,
            "full_byte_component_package_verified": True,
            "component_package_digest": package_manifest["package_digest"],
        },
        pipeline_mode=DUAL_TARGET_PIPELINE_MODE,
        artifixer3d_steps=tuning["artifixer3d_steps"],
        random_seed=tuning["random_seed"],
    )
    extracted = work / "artifixer_execution"
    extract_provider_archive(Path(bundle["bundle"]["path"]), extracted)
    artifixer_output = work / "artifixer_output"
    completed = runner(
        [str(extracted / "provider_runtime/run_public_scene_artifixer3d.sh")],
        cwd=extracted,
        env={
            **values,
            "BLUEPRINT_PUBLIC_SCENE_ARTIFIXER3D_OUTPUT_DIR": str(artifixer_output),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=7_000,
    )
    runtime_result = _read(
        artifixer_output / "public_scene_artifixer3d_runtime_result.json",
        code="scene_configuration_artifixer_runtime_result_missing",
    )
    if completed.returncode != 0 or runtime_result.get("status") != (
        "raw_artifixer3d_candidate_completed_requires_visual_and_multiview_review"
    ):
        raise TaskEvaluationSceneConfigurationArtifixerError(
            "scene_configuration_artifixer_runtime_failed"
        )
    runtime_task = runtime_result["tasks"][0]
    source_task = candidate["tasks"][0]
    review_frames = []
    for source_frame, generated in zip(
        source_task["frames"], runtime_task["artifixer3d_review_frames"], strict=True
    ):
        review_frames.append(
            {
                "frame_index": source_frame["frame_index"],
                "camera_id": source_frame["camera_id"],
                "source_frame": source_frame["input_retained_frame"],
                "exact_repair_mask": source_frame["input_exact_repair_mask"],
                "final_frame": _record(Path(generated["path"])),
            }
        )
    review_input: dict[str, Any] = {
        "schema_version": DUAL_TARGET_REVIEW_SCHEMA_VERSION,
        "status": "paired_target_frames_pending_independent_visual_review",
        "publisher_scene_id": publisher_scene_id,
        "review_scope": "source_anchor_exact_mask_and_generated_full_frame_comparison",
        "tasks": [
            {
                "task_id": task_id,
                "physical_camera_count": len(review_frames),
                "frames": review_frames,
            }
        ],
        "outside_support_invariance_proven": False,
        "outside_support_invariance_claimed": False,
        "semantic_object_absence_review_passed": False,
        "multiview_consistency_review_passed": False,
        "appearance_repair_qualified": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "receipt_digest": "",
    }
    review_input["receipt_digest"] = canonical_digest(review_input, digest_field="receipt_digest")
    review_input_path = work / f"{DUAL_TARGET_REVIEW_SCHEMA_VERSION}.json"
    review_input_path.write_text(canonical_json(review_input) + "\n", encoding="utf-8")
    rights_digest = _sha256(rights_path)
    review_rights_path = work / "artifixer_ai_visual_review_rights.v1.json"
    human = _human_authority(configuration)
    materialize_artifixer_ai_visual_review_rights(
        configuration_run_id=str(stage_input["run_id"]),
        source_scene_rights_admission_digest=rights_digest,
        accepted_by=human["accepted_by"],
        accepted_on=human["accepted_on"],
        human_authority_reference=human["authority_reference"],
        output_path=review_rights_path,
    )
    review_scope = scene_configuration_openai_stage_scope(
        values, stage="artifixer_visual_review"
    )
    review_token = _stage_openai_token(values, stage="artifixer_visual_review")
    with _temporary_openai_key(review_token):
        visual_review_cap = float(
            values.get(
                "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_ARTIFIXER_VISUAL_REVIEW_MAX_COST_USD"
            )
            or 0
        )
        review = run_artifixer_ai_visual_review(
            final_composite_receipt_path=review_input_path,
            rights_attestation_path=review_rights_path,
            configuration_run_id=str(stage_input["run_id"]),
            publisher_instance_id=str(configuration["source_object"]["publisher_instance_id"]),
            minimum_review_frames=int(configuration["required_views"]["minimum"]),
            output_root=work / "independent_visual_review",
            openai_cost_scope_attestation_path=Path(
                review_scope["attestation_file"]
            ),
            openai_admin_api_key_file=_required_path(values, "OPENAI_ADMIN_API_KEY_FILE"),
            openai_project_id=str(values.get("OPENAI_PROJECT_ID") or ""),
            openai_api_key_id=review_scope["api_key_id"],
            max_cost_usd=visual_review_cap,
            cost_lane_id=_VISUAL_REVIEW_COST_SCOPE,
            paid_resource_class=_VISUAL_REVIEW_COST_SCOPE,
            # The scene lane binds the call to its pre-call official-cost
            # snapshot and settles only the attributable delta.  Keeping the
            # generic reviewer's zero-baseline default would make this stage
            # usable only once per UTC day after its first successful call.
            require_zero_baseline=False,
        )
    if review.get("decision") != "accepted" or not review.get("review_receipt"):
        raise TaskEvaluationSceneConfigurationArtifixerError(
            "scene_configuration_artifixer_visual_review_rejected"
        )
    review_receipt_path = Path(review["review_receipt"]["path"])
    native = runtime_task["native_appearance"]
    appearance_source = Path(native["isaac_nurec_usdz"]["path"])
    appearance = output_root / "configured_appearance_without_source_object.usdz"
    shutil.copyfile(appearance_source, appearance)
    copied_review = output_root / "appearance_visual_review_receipt.v1.json"
    shutil.copyfile(review_receipt_path, copied_review)
    removal: dict[str, Any] = {
        "schema_version": "task_evaluation_artifixer_object_removal_result.v1",
        "status": "qualified_generated_appearance_edit",
        "publisher_instance_id": configuration["source_object"]["publisher_instance_id"],
        "raw_interiorgs_bytes_sent_to_external_provider": False,
        "visual_review_receipt_digest": review["review_receipt"]["receipt_digest"],
        "visual_review_receipt_sha256": _sha256(copied_review),
        "semantic_object_free_visual_review_passed": True,
        "multiview_consistency_review_passed": True,
        "generated_pixels_labeled": True,
        "appearance_authority": "generated_support_not_observed_source_or_physics_truth",
        "result_digest": "",
    }
    removal["result_digest"] = canonical_digest(removal, digest_field="result_digest")
    removal_path = output_root / "appearance_removal_receipt.v1.json"
    removal_path.write_text(canonical_json(removal) + "\n", encoding="utf-8")
    artifacts = [
        {"role": "configured_appearance_without_source_object", **_component_record(appearance)},
        {"role": "appearance_removal_receipt", **_component_record(removal_path)},
        {"role": "appearance_visual_review_receipt", **_component_record(copied_review)},
        render_handoff,
    ]
    result = {
        "schema_version": COMPONENT_RESULT_SCHEMA_VERSION,
        "status": "completed",
        "adapter_id": _ADAPTER_ID,
        "stage_id": stage["stage_id"],
        "provider_mutations_performed": 0,
        "nested_paid_execution_requested": False,
        "artifacts": artifacts,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    component_result_path.write_text(canonical_json(result) + "\n", encoding="utf-8")
    return result


def main() -> int:
    execute_artifixer_component()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TaskEvaluationSceneConfigurationArtifixerError",
    "execute_artifixer_component",
    "main",
]
