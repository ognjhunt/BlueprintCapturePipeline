"""Release only model allowances proven unentered by retained terminal evidence.

Native billing stays at its full authorized allowance. These two early refusal
contracts precede the first model gate; an ordinary failure, a partial authoring
directory, or an incomplete archive is not evidence of zero model spending.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from .decision_evidence_contracts import canonical_digest


def pretraining_never_entered(result):
    return (result.get("api_pretraining") is None
            and result.get("cpu_prestage") is None
            and result.get("provider_mutations_performed") == 0
            and result.get("provider_runtime_output_zip_path") is None
            and "vast_adapter_failed:ValueError:artifixer_pretraining_first_stage_invalid"
            in result.get("blockers", []))


def _require(condition):
    if not condition:
        raise ValueError("unentered_authoring_evidence_invalid")


def _sealed(value, field):
    _require(value.get(field) == canonical_digest(value, digest_field=field))
    return value


def authoring_never_entered(result, request):
    """Recognize the initial Astra disclosure refusal, never a later failure.

Read the digest-bound complete output archive, not mutable extracted files.
Only the input, dependency list, lock, and initial refusal log may exist in
stage 3. A cost ledger, resumed authoring, or any later stage keeps the hold.
"""
    try:
        _sealed(result, "result_digest")
        _require(result.get("status") == "blocked" and result.get("retry_cap") == 0
                 and result.get("api_pretraining") is None and result.get("cpu_prestage") is None)
        caps = request["spend"]["external_service_caps"]["openai"]["stage_max_cost_usd"]
        _require(set(caps) == {"artifixer_semantic_teacher", "artifixer_visual_review", "content_agents"}
                 and caps["artifixer_semantic_teacher"] == caps["artifixer_visual_review"] == 0)
        path = Path(result["provider_runtime_output_zip_path"])
        _require(not path.is_symlink() and path.is_file())
        with path.open("rb") as stream:
            digest = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
        _require(digest == result["provider_runtime_output_zip_sha256"])
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            _require(len(names) == len(set(names)))
            prefix = "stages/stage-3/producer/"
            expected = {prefix + name for name in (
                ".astra-component.lock", "dependency_results.v1.json",
                "stage_production_input.v1.json", "stage_producer.log")}
            stage3 = {name for name in names if name.startswith("stages/stage-3/")}
            _require(stage3 == expected and not any(
                name.startswith("stages/stage-") and not name.startswith(
                    ("stages/stage-1/", "stages/stage-2/", "stages/stage-3/")) for name in names))

            def read(name):
                _require(archive.getinfo(name).file_size <= 2 * 1024**2)
                return archive.read(name)

            exclusions = json.loads(read("provider_output_zip_exclusions.json"))
            from .task_evaluation_scene_configuration_output_archive import EXCLUDED_PARTS
            _require(exclusions == {
                "schema_version": "task_evaluation_scene_configuration_provider_output_zip_exclusions.v1",
                "excluded_directory_names": sorted(EXCLUDED_PARTS)})
            provider = _sealed(json.loads(read("task_evaluation_scene_configuration_provider_result.v1.json")), "result_digest")
            stage_input = json.loads(read(prefix + "stage_production_input.v1.json"))
            _require(provider.get("status") == "blocked" and provider.get("first_stage_started") is True)
            for value in (provider, stage_input):
                _require(value.get("run_id") == result["run_id"] == request["run_id"]
                         and value.get("source_commit") == result["source_commit"] == request["expected_production_commit"])
            config = stage_input["configuration"]
            envelope = stage_input["construction_envelope"]
            _require(config.get("authoring_backend") == "astra_cad_blender_v1"
                     and config.get("source_observation_kind") == "website_capture_frames"
                     and config.get("astra_phase_adoption") is None
                     and envelope.get("partial_astra_successor") is None)
            dependencies = json.loads(read(prefix + "dependency_results.v1.json"))
            _require([row.get("stage_id") for row in dependencies] == ["stage-1", "stage-2"])
            for row in dependencies:
                _sealed(row, "stage_result_digest")
                _require(row.get("status") == "completed" and row.get("execution_class") == "no_spend"
                         and row.get("paid_execution_requested") is False and row.get("provider_mutations_performed") == 0)
            log = read(prefix + "stage_producer.log").decode("utf-8")
            _require("in build_authoring_request\n" in log and
                     "blueprint_pipeline.task_evaluation_scene_configuration_astra_driver.AstraStageError: astra_derived_disclosure_not_admitted\n" in log)
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError, zipfile.BadZipFile):
        return False
