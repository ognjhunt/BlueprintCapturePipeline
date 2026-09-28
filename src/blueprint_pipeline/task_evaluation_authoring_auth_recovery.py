"""Recognize a rejected first authoring request and require a changed, working key."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import urllib.error
import urllib.request
import zipfile

from .task_evaluation_unentered_authoring_budget import _require, _sealed

KIND = "initial_authoring_authentication"
PREFIX = "stages/stage-3/producer/astra_cad_blender_runtime/"


def authenticate_key(path, *, opener=None):
    """Read-only authentication; no inference and no credential in error output."""
    opener = opener or urllib.request.urlopen
    try:
        request = urllib.request.Request("https://api.openai.com/v1/models",
            headers={"Authorization": "Bearer " + Path(path).read_text().strip()})
        with opener(request, timeout=30) as response:
            value = json.load(response)
        if not isinstance(value.get("data"), list):
            raise ValueError("openai_key_authentication_response_invalid")
    except urllib.error.HTTPError as exc:
        raise ValueError(f"openai_key_authentication_http_{exc.code}") from None
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("openai_key_authentication_unavailable") from None


def initial_authentication_failure(result):
    """A complete archive proves one reserved source-analysis call was rejected.

    Keep that call's full projected maximum pending official billing. Never
    infer zero cost from a missing response, a timeout, or an arbitrary failure.
    """
    try:
        _sealed(result, "result_digest")
        _require(result.get("status") == "blocked" and result.get("retry_cap") == 0
                 and result.get("api_pretraining") is None and result.get("cpu_prestage") is None
                 and type(result.get("provider_mutations_performed")) is int
                 and result["provider_mutations_performed"] == 0
                 and result.get("continuing_spend_from_this_run") is False)
        path = Path(result["provider_runtime_output_zip_path"])
        _require(not path.is_symlink() and path.is_file())
        with path.open("rb") as stream:
            _require("sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
                     == result["provider_runtime_output_zip_sha256"])
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            _require(len(names) == len(set(names)))
            def read(name):
                _require(archive.getinfo(name).file_size <= 2 * 1024**2)
                return json.loads(archive.read(name))
            from .task_evaluation_scene_configuration_output_archive import EXCLUDED_PARTS
            _require(read("provider_output_zip_exclusions.json") == {
                "schema_version": "task_evaluation_scene_configuration_provider_output_zip_exclusions.v1",
                "excluded_directory_names": sorted(EXCLUDED_PARTS)})
            _require(not any(n.startswith("stages/stage-") and not n.startswith(
                ("stages/stage-1/", "stages/stage-2/", "stages/stage-3/")) for n in names))
            stage = read("stages/stage-3/producer/stage_production_input.v1.json")
            _require(stage["run_id"] == result["run_id"] and stage["source_commit"] == result["source_commit"]
                     and stage["configuration"].get("authoring_backend") == "astra_cad_blender_v1"
                     and stage["configuration"].get("astra_phase_adoption") is None
                     and stage["construction_envelope"].get("partial_astra_successor") is None)
            provider = _sealed(read("task_evaluation_scene_configuration_provider_result.v1.json"), "result_digest")
            _require(provider["run_id"] == result["run_id"] and provider["source_commit"] == result["source_commit"]
                     and provider["status"] == "blocked" and provider["first_stage_started"] is True)
            caps = stage["construction_envelope"]["request"]["spend"]["external_service_caps"]["openai"]["stage_max_cost_usd"]
            _require(set(caps) == {"artifixer_semantic_teacher", "artifixer_visual_review", "content_agents"})
            _require(caps["artifixer_semantic_teacher"] == caps["artifixer_visual_review"] == 0)
            deps = read("stages/stage-3/producer/dependency_results.v1.json")
            _require([d["stage_id"] for d in deps] == ["stage-1", "stage-2"])
            for dep in deps:
                _sealed(dep, "stage_result_digest")
                _require(dep["status"] == "completed" and dep["paid_execution_requested"] is False
                         and dep["execution_class"] == "no_spend" and dep["provider_mutations_performed"] == 0)
            authored = {n.removeprefix(PREFIX + "authoring/") for n in names if n.startswith(PREFIX + "authoring/")}
            _require({"request.json", "failure.json"} <= authored
                     and all(n in {"request.json", "failure.json"} or n.startswith("cache/") for n in authored))
            request, failure = read(PREFIX + "authoring/request.json"), read(PREFIX + "authoring/failure.json")
            _sealed(request, "request_digest")
            _require(request["run_id"] == result["run_id"] and failure["request_digest"] == request["request_digest"]
                     and failure["exception_type"] == "AuthenticationError" and failure["status"] == "blocked"
                     and str(failure["blocker"]).startswith("Error code: 401 - ")
                     and "expired_secret_key" in failure["blocker"])
            inference = [n for n in names if n.startswith(PREFIX + "inference/")]
            _require(len(inference) == 1 and inference[0].startswith(PREFIX + "inference/inference_reservations/reserved/"))
            reservation = _sealed(read(inference[0]), "inference_reservation_digest")
            _require(reservation["run_id"] == result["run_id"] and reservation["capability"] == request["object_id"] + "_source_analysis"
                     and reservation["max_turns"] == 1 and reservation["billing_status"] == "worst_case_reserved_before_provider_call")
            cost = _sealed(read(PREFIX + "official_openai_cost/openai_official_cost_run_completion.v1.json"), "completion_receipt_digest")
            _require(cost["run_id"] == result["run_id"] and cost["runtime_exception_type"] == "AuthenticationError"
                     and cost["provider_call_performed"] is True and cost["runtime_result_digest"] is None)
            snapshot = _sealed(cost["official_completion_snapshot"], "openai_cost_snapshot_digest")
            official = _sealed(read(PREFIX + "official_openai_cost/openai_official_cost_run_reservation.v1.json"), "reservation_receipt_digest")
            scope = _sealed(official["provider_reservation"], "cost_reservation_digest")
            _require(official["run_id"] == result["run_id"] == scope["candidate_id"]
                     and official["reservation_receipt_digest"] == cost["reservation_receipt_digest"]
                     and scope["cost_reservation_digest"] == cost["cost_reservation_digest"]
                     and all(official[k] == cost[k] for k in ("request_digest", "candidate_digest", "authorization_receipt_digest"))
                     and all(scope[k] == snapshot[k] for k in ("api_key_id", "project_id"))
                     and official["maximum_cost_usd"] == scope["reserved_max_cost_usd"] == caps["content_agents"])
            bound = float(reservation["projected_max_cost_usd"])
            _require(math.isfinite(bound) and 0 < bound <= float(caps["content_agents"]))
            return {"api_key_id": snapshot["api_key_id"], "project_id": snapshot["project_id"],
                    "retained_spend_usd": bound, "official_billing_final": False}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, zipfile.BadZipFile):
        return None


def replacement_admission(result, environment=None):
    evidence = initial_authentication_failure(result)
    env = os.environ if environment is None else environment
    key_id = env.get("OPENAI_CONTENT_AGENTS_API_KEY_ID")
    if (evidence is None or not key_id or key_id == evidence["api_key_id"]
            or env.get("OPENAI_PROJECT_ID") != evidence["project_id"]):
        return {"status": "blocked", "blockers": ["authoring_replacement_key_required"]}
    try:
        authenticate_key(env["OPENAI_CONTENT_AGENTS_API_KEY_FILE"])
    except (KeyError, ValueError) as exc:
        return {"status": "blocked", "blockers": [str(exc)]}
    return {"status": "admitted", "prior_api_key_id": evidence["api_key_id"],
            "replacement_api_key_id": key_id, "provider_mutation_performed": False}
