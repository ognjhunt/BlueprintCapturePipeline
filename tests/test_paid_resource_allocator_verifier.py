from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_paid_resource_allocator.py"
SPEC = importlib.util.spec_from_file_location("verify_paid_resource_allocator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)

GEMINI_EVALUATOR_PAID_SURFACES = {
    "src/blueprint_pipeline/policy_ranking_evaluator_diagnostic_gemini_matrix.py",
    "src/blueprint_pipeline/policy_ranking_evaluator_diagnostic_gemini_transport_canary.py",
}
SCENE_CONFIGURATION_ALLOCATOR = (
    "src/blueprint_pipeline/task_evaluation_scene_configuration_allocator.py"
)
CONFIGURED_SCENE_OBJECT_STORE = (
    "src/blueprint_pipeline/task_evaluation_configured_scene_object_store.py"
)
VAST_PROVIDER_ADAPTER = "src/blueprint_pipeline/vast_provider_adapter.py"
VAST_PROVIDER_ADAPTER_CLI = "src/blueprint_pipeline/vast_provider_adapter_cli.py"


def test_selected_g1_controller_is_a_registered_canonical_admission_surface():
    path = "src/blueprint_pipeline/native_g1_team_paid_policy.py"
    manifest = json.loads(verifier.MUTATION_SURFACE_MANIFEST.read_text())
    assert path in verifier.APPROVED_ADMISSION_ISSUERS
    assert path in verifier.APPROVED_LANE_ADMISSION_BUILDERS
    assert path in manifest["issuer_allowlist"]["require_paid_resource_admission"]
    assert path in manifest["issuer_allowlist"]["build_paid_lane_admission"]


def test_request_dict_runpod_create_is_discovered_and_unclassified() -> None:
    source = """
RUNPOD_REST_API_BASE = "https://rest.runpod.io/v1"
request = {"url": f"{RUNPOD_REST_API_BASE}/pods", "method": "POST"}
"""
    assert verifier._direct_paid_mutation_signals(source) == {"runpod_pod_create"}
    assert verifier._unclassified_direct_mutators(
        {"src/blueprint_pipeline/new_bypass.py": source}, set()
    ) == {"src/blueprint_pipeline/new_bypass.py"}


def test_request_dict_runpod_create_is_accepted_only_when_manifested() -> None:
    path = "src/blueprint_pipeline/canonical_adapter.py"
    source = """
RUNPOD_REST_API_BASE = "https://rest.runpod.io/v1"
request = {"url": f"{RUNPOD_REST_API_BASE}/pods", "method": "POST"}
"""
    assert verifier._unclassified_direct_mutators({path: source}, {path}) == set()


def test_s3_write_or_delete_is_discovered_and_unclassified() -> None:
    source = "client.upload_file(path, bucket, key)\nclient.delete_object(Bucket=bucket, Key=key)"
    assert verifier._direct_paid_mutation_signals(source) == {"s3_object_write_or_delete"}
    assert verifier._unclassified_direct_mutators(
        {"src/blueprint_pipeline/new_s3_bypass.py": source}, set()
    ) == {"src/blueprint_pipeline/new_s3_bypass.py"}


def test_findall_executable_mutations_are_discovered_but_offline_preparation_is_not():
    sources = [
        'client.beta.findall.create(objective="fixture")',
        'client.beta.findall.create (objective="fixture")',
        'client.beta.findall.ingest(objective="fixture")',
        'client.beta.findall.extend("findall_fixture", match_limit=10)',
        'from .parallel_findall import RUNS_URL\nself._post(RUNS_URL, body)',
        'urllib.request.Request("https://api.parallel.ai/v1beta/findall/runs", method="POST")',
        'from .parallel_findall import RUNS_URL\nrequests.post(RUNS_URL, json=spec)',
        'from .parallel_findall import RUNS_URL as target\nRequest(target, method="POST")',
        'from . import parallel_findall as findall\nrequests.post(url=findall.RUNS_URL, json=spec)',
        'import blueprint_pipeline.parallel_findall\nrequests.post(blueprint_pipeline.parallel_findall.RUNS_URL, json=spec)',
    ]
    for source in sources:
        assert "parallel_findall_paid_mutation" in verifier._direct_paid_mutation_signals(source)
        assert verifier._unclassified_direct_mutators({"scripts/bypass.py": source}, set()) == {"scripts/bypass.py"}
    reader = SCRIPT.parent.parent / "src/blueprint_pipeline/parallel_findall.py"
    assert verifier._direct_paid_mutation_signals(reader.read_text()) == set()


def test_findall_adapter_cannot_issue_its_own_grant():
    path = "src/blueprint_pipeline/parallel_findall_execution.py"
    source = (SCRIPT.parent.parent / path).read_text()
    calls = verifier._all_calls(SCRIPT.parent.parent / path)
    assert "require_paid_resource_admission_grant" in verifier._reachable_calls(source, "create")
    assert "claim_submission" in verifier._reachable_calls(source, "create")
    assert not calls & {"require_paid_resource_admission", "build_paid_lane_admission"}
    manifest = json.loads(verifier.MUTATION_SURFACE_MANIFEST.read_text())
    rows = {row["path"]: row for row in manifest["surfaces"]}
    assert rows[path]["classification"] == "grant_gated_legacy_adapter"


def test_provider_hostname_signals_use_exact_regex_matching() -> None:
    source = """
request = {"url": "https://api.runpod.io/v2/pods", "method": "POST"}
"""
    assert verifier._direct_paid_mutation_signals(source) == {"runpod_pod_create"}


def test_unmanifested_script_mutator_is_rejected() -> None:
    path = "scripts/new_paid_bypass.py"
    source = 'client.upload_file("source", "bucket", "key")'
    assert verifier._unclassified_direct_mutators({path: source}, set()) == {path}


def test_third_s3_capability_or_transport_caller_is_rejected() -> None:
    sources = {
        "src/blueprint_pipeline/groot_oscar_runpod_s3_model_cache.py": """
def upload_and_verify_model_cache():
    _issue_transport_execution_capability()
    _upload_and_verify_model_cache_impl()
""",
        "src/blueprint_pipeline/groot_oscar_model_cache_s3_remote_executor.py": """
def execute_remote_packet():
    _issue_transport_execution_capability()
    _upload_and_verify_model_cache_impl()
""",
        "scripts/new_bypass.py": """
from blueprint_pipeline.groot_oscar_runpod_s3_model_cache import (
    _issue_transport_execution_capability as mint,
    _upload_and_verify_model_cache_impl as mutate,
)
def bypass():
    mint()
    mutate()
""",
    }
    assert verifier._s3_transport_capability_callers(sources) != (
        verifier.APPROVED_S3_TRANSPORT_CAPABILITY_CALLERS
    )


def test_production_scan_recurses_through_source_and_scripts(tmp_path: Path) -> None:
    source = tmp_path / "src/blueprint_pipeline/nested/adapter.py"
    script = tmp_path / "scripts/nested/executor.py"
    source.parent.mkdir(parents=True)
    script.parent.mkdir(parents=True)
    source.write_text("", encoding="utf-8")
    script.write_text("", encoding="utf-8")
    assert verifier._production_python_paths(tmp_path) == [script, source]


def test_operator_docs_reject_legacy_paid_commands_and_allow_canonical() -> None:
    forbidden = """
blueprint-run-runpod-provider-adapter --request x --mode on-demand-pod
"""
    assert verifier._forbidden_operator_doc_commands(forbidden) == {
        "legacy_runpod_adapter_paid_mode"
    }
    canonical = "python -m blueprint_pipeline.paid_resource_allocator gpu-canary --execute"
    assert verifier._forbidden_operator_doc_commands(canonical) == set()


def test_gemini_evaluator_paid_paths_are_registered_as_canonical_surfaces() -> None:
    manifest = json.loads(verifier.MUTATION_SURFACE_MANIFEST.read_text(encoding="utf-8"))
    issuer_allowlist = manifest["issuer_allowlist"]
    assert GEMINI_EVALUATOR_PAID_SURFACES <= verifier.APPROVED_ADMISSION_ISSUERS
    assert GEMINI_EVALUATOR_PAID_SURFACES <= verifier.APPROVED_LANE_ADMISSION_BUILDERS
    assert GEMINI_EVALUATOR_PAID_SURFACES <= set(
        issuer_allowlist["require_paid_resource_admission"]
    )
    assert GEMINI_EVALUATOR_PAID_SURFACES <= set(issuer_allowlist["build_paid_lane_admission"])
    surfaces = {row["path"]: row for row in manifest["surfaces"]}
    for path in GEMINI_EVALUATOR_PAID_SURFACES:
        assert surfaces[path]["classification"] == "canonical_adapter"
        assert {
            "build_paid_lane_admission",
            "require_paid_resource_admission",
            "require_paid_resource_admission_grant",
        } <= set(surfaces[path]["required_markers"])


def test_scene_configuration_paid_surfaces_are_exactly_classified() -> None:
    manifest = json.loads(verifier.MUTATION_SURFACE_MANIFEST.read_text(encoding="utf-8"))
    issuer_allowlist = manifest["issuer_allowlist"]
    assert SCENE_CONFIGURATION_ALLOCATOR in verifier.APPROVED_ADMISSION_ISSUERS
    assert SCENE_CONFIGURATION_ALLOCATOR in verifier.APPROVED_LANE_ADMISSION_BUILDERS
    assert SCENE_CONFIGURATION_ALLOCATOR in issuer_allowlist[
        "require_paid_resource_admission"
    ]
    assert SCENE_CONFIGURATION_ALLOCATOR in issuer_allowlist[
        "build_paid_lane_admission"
    ]
    surfaces = {row["path"]: row for row in manifest["surfaces"]}
    assert surfaces[SCENE_CONFIGURATION_ALLOCATOR] == {
        "path": SCENE_CONFIGURATION_ALLOCATOR,
        "classification": "canonical_adapter",
        "required_markers": [
            "build_paid_lane_admission",
            "require_paid_resource_admission",
            "run_scene_configuration_vast",
        ],
    }
    assert surfaces[CONFIGURED_SCENE_OBJECT_STORE] == {
        "path": CONFIGURED_SCENE_OBJECT_STORE,
        "classification": "metered_object_storage_data_plane",
        "required_markers": [
            "upload_file",
            "get_object",
            "full_byte_service_account_readback_passed",
        ],
    }


def test_extracted_vast_cli_keeps_the_disable_gate_separately_classified() -> None:
    manifest = json.loads(verifier.MUTATION_SURFACE_MANIFEST.read_text(encoding="utf-8"))
    surfaces = {row["path"]: row for row in manifest["surfaces"]}

    assert surfaces[VAST_PROVIDER_ADAPTER] == {
        "path": VAST_PROVIDER_ADAPTER,
        "classification": "grant_gated_legacy_adapter",
        "required_markers": ["require_paid_resource_admission_grant"],
    }
    assert surfaces[VAST_PROVIDER_ADAPTER_CLI] == {
        "path": VAST_PROVIDER_ADAPTER_CLI,
        "classification": "hard_disabled_legacy_launcher",
        "required_markers": ["legacy_vast_provider_mutation_cli_disabled"],
    }
    assert "paid_resource_mutation_surface_marker_missing:" + VAST_PROVIDER_ADAPTER not in (
        verifier.verify()
    )


def test_model_volume_watchdog_handoff_is_machine_enforced() -> None:
    blockers = set(verifier.verify())
    assert "paid_resource_admission_issuer_set_mismatch" not in blockers
    assert "model_volume_watchdog_handoff_schema_missing" not in blockers
    assert "model_volume_watchdog_process_handoff_missing" not in blockers
    assert "model_volume_missing_key_terminal_evidence_missing" not in blockers
    assert "model_volume_ready_handoff_liveness_guard_missing" not in blockers
    assert "gpu_preflight_model_volume_watchdog_handoff_guard_missing" not in blockers
    assert "gpu_launch_refresh_drops_model_volume_watchdog_handoff" not in blockers
    assert "runbook_model_volume_watchdog_handoff_missing" not in blockers
    assert "remote_build_final_tag_promotion_guard_missing" not in blockers
    assert "remote_build_final_tag_promotion_order_invalid" not in blockers
    assert "remote_build_pushes_unvalidated_final_release_tag" not in blockers
    assert "remote_build_pushes_unvalidated_final_foundation_tag" not in blockers
    assert "lambda_termination_shared_admission_guard_missing" not in blockers


REMOTE_CPU_ALLOCATOR = "src/blueprint_pipeline/remote_cpu_job_allocator.py"
CLOUD_RUN_JOBS_CLIENT = "src/blueprint_pipeline/cloud_run_jobs_client.py"
PAIRED_WITNESS_STAGING = "src/blueprint_pipeline/native_task_arena_paired_witness_staging.py"
CANONICAL_ALLOCATOR = "src/blueprint_pipeline/paid_resource_allocator.py"


def _source(relative: str) -> str:
    return (verifier.ROOT / relative).read_text(encoding="utf-8")


def test_cloud_run_job_mutation_is_discovered_including_run_v2_clients() -> None:
    mutations = (
        'API = "https://run.googleapis.com"\nurl = f"{API}/v2/{job}:run"\n',
        'url = f"https://run.googleapis.com/v2/{execution}:cancel"\n',
        "from google.cloud import run_v2\nrun_v2.JobsClient().run_job(name=job)\n",
        "import google.cloud.run_v2 as cloud_run\n",
        "client = ExecutionsClient()\nclient.cancel_execution(name=name)\n",
        "request = RunJobRequest(name=job, overrides=overrides)\n",
    )
    for source in mutations:
        assert "gcp_cloud_run_job_mutation" in verifier._direct_paid_mutation_signals(source), source
        assert verifier._unclassified_direct_mutators(
            {"src/blueprint_pipeline/new_cloud_run_launcher.py": source}, set()
        ) == {"src/blueprint_pipeline/new_cloud_run_launcher.py"}
    # Identifiers that merely contain run_v2, and a read-only mention of the host, are not clients.
    for source in (
        "def compile_new_site_task_evaluation_run_v2(value):\n    return value\n",
        'run_v2 = sub.add_parser("run-v2")\n',
        'DOCS = "https://run.googleapis.com/v2/projects/p/locations/l/jobs/j"\n',
    ):
        assert verifier._direct_paid_mutation_signals(source) == set(), source
    assert "gcp_cloud_run_job_mutation" in verifier._direct_paid_mutation_signals(_source(CLOUD_RUN_JOBS_CLIENT))


def test_presigned_put_and_copy_object_are_discovered_and_unclassified() -> None:
    writers = (
        ('url = client.generate_presigned_url("put_object", Params=params, ExpiresIn=60)',
         "s3_presigned_write_authority"),
        ("url = client.generate_presigned_url(\n    ClientMethod='upload_part', Params=params)",
         "s3_presigned_write_authority"),
        ('method = "put_object"\nurl = client.generate_presigned_url(method, Params=params)',
         "s3_presigned_write_authority"),
        ("client.copy_object(Bucket=bucket, Key=key, CopySource=source)", "s3_object_write_or_delete"),
        ("client.upload_part_copy(Bucket=bucket, Key=key, UploadId=upload, PartNumber=1, CopySource=source)",
         "s3_object_write_or_delete"),
    )
    for source, signal in writers:
        assert signal in verifier._direct_paid_mutation_signals(source), source
        assert verifier._unclassified_direct_mutators(
            {"scripts/new_object_writer.py": source}, set()
        ) == {"scripts/new_object_writer.py"}
    # A presigned GET is read authority only.
    assert verifier._direct_paid_mutation_signals(
        'url = client.generate_presigned_url("get_object", Params=params, ExpiresIn=60)'
    ) == set()

    manifest = json.loads(verifier.MUTATION_SURFACE_MANIFEST.read_text(encoding="utf-8"))
    classified = {row["path"] for row in manifest["surfaces"]}
    production = {
        path.relative_to(verifier.ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in verifier._production_python_paths()
    }
    writers_in_tree = {
        relative for relative, source in production.items()
        if verifier._direct_paid_mutation_signals(source)
        & {"s3_presigned_write_authority", "s3_object_write_or_delete"}
    }
    assert {PAIRED_WITNESS_STAGING, CONFIGURED_SCENE_OBJECT_STORE} <= writers_in_tree
    assert verifier._unclassified_direct_mutators(production, classified) == set()


def test_remote_cpu_lane_requires_reconcile_pagination_and_zero_proofs() -> None:
    allocator, client = _source(REMOTE_CPU_ALLOCATOR), _source(CLOUD_RUN_JOBS_CLIENT)
    assert verifier._remote_cpu_lane_blockers(allocator, client) == []
    for call, blocker in verifier.REMOTE_CPU_REQUIRED_CALLS.items():
        mutated = allocator.replace(f"{call}(", f"{call}_removed(")
        assert blocker in verifier._remote_cpu_lane_blockers(mutated, client), call
    for marker in ("remote_cpu_provider_zero_unproven", "remote_cpu_ambiguous_dispatch_unresolved"):
        assert f"remote_cpu_lane_marker_missing:{marker}" in verifier._remote_cpu_lane_blockers(
            allocator.replace(marker, "remote_cpu_renamed"), client)
    assert "remote_cpu_run_job_grant_validation_missing" in verifier._remote_cpu_lane_blockers(
        allocator, client.replace("require_paid_resource_admission_grant(", "_grant_left_unchecked("))
    assert "remote_cpu_execution_listing_pagination_missing" in verifier._remote_cpu_lane_blockers(
        allocator, client.replace("nextPageToken", "pageTokenIgnored"))

    # Reachability, not mere presence: a proof defined but never reached from the entry point fails.
    reached = (
        "def run_remote_cpu_job(args):\n    return _dispatch(args)\n"
        "def _dispatch(args):\n    require_paid_resource_admission(args)\n    reconcile_ambiguous_dispatch(args)\n"
        "    client.list_all_executions(job)\n    return _teardown(args)\n"
        "def _teardown(args):\n    prove_compute_zero(args)\n    prove_provider_zero(args)\n"
        "BLOCKERS = ('remote_cpu_provider_zero_unproven', 'remote_cpu_ambiguous_dispatch_unresolved')\n"
    )
    assert verifier._remote_cpu_lane_blockers(reached, client) == []
    unreached = reached.replace("    return _teardown(args)\n", "    return args\n")
    assert set(verifier._remote_cpu_lane_blockers(unreached, client)) == {
        "remote_cpu_compute_zero_proof_missing", "remote_cpu_provider_zero_proof_missing"}


def test_remote_cpu_job_surfaces_are_exactly_classified() -> None:
    manifest = json.loads(verifier.MUTATION_SURFACE_MANIFEST.read_text(encoding="utf-8"))
    surfaces = {row["path"]: row for row in manifest["surfaces"]}
    assert surfaces[REMOTE_CPU_ALLOCATOR] == {
        "path": REMOTE_CPU_ALLOCATOR,
        "classification": "canonical_adapter",
        "required_markers": [
            "build_paid_lane_admission",
            "require_paid_resource_admission",
            "run_remote_cpu_job",
        ],
    }
    assert surfaces[CLOUD_RUN_JOBS_CLIENT] == {
        "path": CLOUD_RUN_JOBS_CLIENT,
        "classification": "grant_gated_legacy_adapter",
        "required_markers": [
            "require_paid_resource_admission_grant",
            'CLOUD_RUN_CPU_JOB_RESOURCE_CLASS = "cloud_run_cpu_job"',
        ],
    }
    assert surfaces[PAIRED_WITNESS_STAGING] == {
        "path": PAIRED_WITNESS_STAGING,
        "classification": "metered_object_storage_data_plane",
        "required_markers": ["stage_paired_witness_slot", "signed_output_object_binding_sha256"],
    }
    assert surfaces[CONFIGURED_SCENE_OBJECT_STORE]["classification"] == "metered_object_storage_data_plane"
    issuers = manifest["issuer_allowlist"]
    for allowlist in (verifier.APPROVED_ADMISSION_ISSUERS, verifier.APPROVED_LANE_ADMISSION_BUILDERS,
                      issuers["require_paid_resource_admission"], issuers["build_paid_lane_admission"]):
        assert REMOTE_CPU_ALLOCATOR in allowlist and CLOUD_RUN_JOBS_CLIENT not in allowlist
    for relative in (REMOTE_CPU_ALLOCATOR, CLOUD_RUN_JOBS_CLIENT, PAIRED_WITNESS_STAGING):
        assert all(marker in _source(relative) for marker in surfaces[relative]["required_markers"]), relative
    issuing = {"require_paid_resource_admission", "build_paid_lane_admission"}
    client_calls = verifier._all_calls(verifier.ROOT / CLOUD_RUN_JOBS_CLIENT)
    assert "require_paid_resource_admission_grant" in client_calls and not client_calls & issuing
    assert issuing <= verifier._all_calls(verifier.ROOT / REMOTE_CPU_ALLOCATOR)
    assert not issuing & verifier._all_calls(verifier.ROOT / PAIRED_WITNESS_STAGING)
    # The seam module itself holds no direct provider write; that stays in the client and the data plane.
    assert verifier._direct_paid_mutation_signals(_source(REMOTE_CPU_ALLOCATOR)) == set()


def test_canonical_allocator_requires_the_remote_cpu_job_subcommand() -> None:
    manifest = json.loads(verifier.MUTATION_SURFACE_MANIFEST.read_text(encoding="utf-8"))
    surfaces = {row["path"]: row for row in manifest["surfaces"]}
    assert "remote-cpu-job" in verifier.CANONICAL_SUBCOMMANDS
    assert surfaces[CANONICAL_ALLOCATOR]["required_markers"] == list(verifier.CANONICAL_SUBCOMMANDS)
    canonical = _source(CANONICAL_ALLOCATOR)
    assert verifier._canonical_subcommand_blockers(canonical, _source("AGENTS.md")) == []
    assert verifier._canonical_subcommand_blockers(
        canonical.replace("remote-cpu-job", "remote-cpu-other"), _source("AGENTS.md")
    ) == ["canonical_allocator_subcommands_missing"]
    assert verifier._canonical_subcommand_blockers(
        canonical, _source("AGENTS.md").replace("paid_resource_allocator remote-cpu-job", "remote_cpu_job")
    ) == ["agents_md_paid_allocator_command_missing:remote-cpu-job"]


# Shapes a new Cloud Run launcher or object-store writer could take (plan 14 PR 2 review).
REVIEW_PROBES = {
    "gcloud_argv": ('import subprocess\nsubprocess.run(["gcloud", "run", "jobs", "execute", job, "--region", "us-central1"])\n',
                    "gcp_cloud_run_job_mutation"),
    "gcloud_text": ('os.system(f"gcloud beta run jobs execute {job} --wait")\n', "gcp_cloud_run_job_mutation"),
    "discovery": ('from googleapiclient import discovery\n'
                  'discovery.build("run", "v2").projects().locations().jobs().run(name=job, body={}).execute()\n',
                  "gcp_cloud_run_job_mutation"),
    "rest_query": ('API = "https://run.googleapis.com"\nurl = f"{API}/v2/{job}:run?alt=json"\n',
                   "gcp_cloud_run_job_mutation"),
    "start_token": ('body = json.dumps({"template": template, "startExecutionToken": "t1"})\n',
                    "gcp_cloud_run_job_mutation"),
    "delete_objects": ('client.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": key, "VersionId": v}]})\n',
                       "s3_object_write_or_delete"),
    "presigned_post": ("client.generate_presigned_post(Bucket=bucket, Key=key, ExpiresIn=600)\n",
                       "s3_presigned_write_authority"),
    "discovery_build_import": ('from googleapiclient.discovery import build\n'
                               'build("run", "v2").projects().locations().jobs().run(name=job, body={}).execute()\n',
                               "gcp_cloud_run_job_mutation"),
    "gcloud_global_flag_argv": ('import subprocess\n'
                                'subprocess.run(["gcloud", "--project", project, "run", "jobs", "execute", job])\n',
                                "gcp_cloud_run_job_mutation"),
    "gcloud_global_flag_text": ('os.system(f"gcloud --project {project} --quiet run jobs execute {job}")\n',
                                "gcp_cloud_run_job_mutation"),
}


def test_review_probe_shapes_are_discovered_and_unclassified() -> None:
    for name, (source, signal) in REVIEW_PROBES.items():
        path = f"src/blueprint_pipeline/probe_{name}.py"
        assert signal in verifier._direct_paid_mutation_signals(source), name
        assert verifier._unclassified_direct_mutators({path: source}, set()) == {path}, name
    for source in (
        'subprocess.run(["gcloud", "run", "jobs", "describe", job])\n',
        'subprocess.run(["gcloud", "run", "jobs", "executions", "list", "--job", job])\n',
        'discovery.build("storage", "v1")\n',
        'from googleapiclient.discovery import build\nbuild("storage", "v1")\n',
        'subprocess.run(["gcloud", "--project", project, "run", "jobs", "describe", job])\n',
        "client.generate_presigned_url('get_object', Params=params)\n",
    ):
        assert verifier._direct_paid_mutation_signals(source) == set(), source


def test_remote_cpu_object_store_writers_have_only_approved_callers() -> None:
    production = {
        path.relative_to(verifier.ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in verifier._production_python_paths()
    }
    writers = verifier.REMOTE_CPU_OBJECT_STORE_WRITERS
    assert writers == {"presign_remote_cpu_put", "copy_remote_cpu_staging_to_cas",
                       "delete_remote_cpu_staging_versions", "discard_remote_cpu_output_object",
                       "remote_cpu_object_store_sentinel"}
    approved = verifier.APPROVED_REMOTE_CPU_OBJECT_STORE_CALLERS
    assert verifier._s3_transport_capability_callers(production, writers) == approved
    # Plan 14 PR 4: the episode-compilation collector promotes (and discards a failed readback) in one function.
    assert {path for path, _ in approved} == {REMOTE_CPU_ALLOCATOR, CONFIGURED_SCENE_OBJECT_STORE,
                                              "src/blueprint_pipeline/task_evaluation_episode_compilation_collector.py"}
    wrapper = (
        "from .task_evaluation_configured_scene_object_store import (\n"
        "    copy_remote_cpu_staging_to_cas, delete_remote_cpu_staging_versions as purge,\n"
        "    presign_remote_cpu_put, remote_cpu_object_store_sentinel,\n"
        ")\n\n\n"
        "def ungated(client, bucket, staging, prefix):\n"
        "    purge(staging_prefix=prefix, client=client, bucket=bucket)\n"
        "    return copy_remote_cpu_staging_to_cas\n"
    )
    observed = verifier._s3_transport_capability_callers(
        {**production, "src/blueprint_pipeline/new_writer.py": wrapper}, writers)
    assert observed - approved == {("src/blueprint_pipeline/new_writer.py", "ungated")}


def test_module_and_class_level_calls_to_object_store_writers_are_callers() -> None:
    writers = verifier.REMOTE_CPU_OBJECT_STORE_WRITERS
    module_level = (
        "from .task_evaluation_configured_scene_object_store import copy_remote_cpu_staging_to_cas\n"
        "RESULT = copy_remote_cpu_staging_to_cas(client=None, bucket='b', staging='s', cas='c')\n"
    )
    class_level = (
        "from .task_evaluation_configured_scene_object_store import delete_remote_cpu_staging_versions as purge\n"
        "class Sweeper:\n"
        "    DONE = purge(staging_prefix='p', client=None, bucket='b')\n"
    )
    observed = verifier._s3_transport_capability_callers(
        {"src/blueprint_pipeline/module_writer.py": module_level,
         "src/blueprint_pipeline/class_writer.py": class_level}, writers)
    assert observed == {("src/blueprint_pipeline/module_writer.py", "<module>"),
                        ("src/blueprint_pipeline/class_writer.py", "Sweeper")}
