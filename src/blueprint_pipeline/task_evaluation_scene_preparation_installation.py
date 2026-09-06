"""Provision release-independent completed-scene preparation, with execution off.

One operator-owned bootstrap binds the existing asset catalog and storage
roots. Deployment recompiles it; scenes and release SHAs are never hand-edited
into the worker configuration. This module neither starts units nor submits a
run, and the installed configuration cannot activate a paid execution.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pwd
import tempfile

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_public_scene_attempt_factory import record
from .task_evaluation_scene_configuration_submission_inputs import checked_file, read
from .task_evaluation_scene_progression_state import atomic_json, require, safe_path

BOOTSTRAP_SCHEMA = "task_evaluation_scene_preparation_bootstrap.v1"
MANAGED_BY = "blueprint_pipeline.task_evaluation_scene_preparation_installation"
DEFAULT_BOOTSTRAP = "/etc/blueprint/task-evaluation-scene-preparation-bootstrap.json"
DEFAULT_PHYSICS_BOUNDS = {"mass_kg_bounds": [0.1, 1.2], "static_friction_bounds": [0.4, 0.8],
                        "dynamic_friction_bounds": [0.3, 0.6], "restitution_bounds": [0.0, 0.1]}


def validate_destination_catalog(rows):
    """Requalify the existing exact USDs; catalog flags alone are not admission."""
    from .task_evaluation_scene_configuration_submission import _destination
    from .task_evaluation_scene_configuration_static_qualification import qualify_scene_configuration_rigid_asset_static
    from .task_evaluation_scene_configuration_supplemental_destination import _requalification_comparable
    require(isinstance(rows, list) and 1 <= len(rows) <= 64, "destination_catalog_invalid")
    names, aliases = set(), set()
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"binding_id", "owner_description_aliases", "simready_result"},
                "destination_catalog_row_invalid")
        name = row["binding_id"]
        require(isinstance(name, str) and name and name not in names, "destination_catalog_identity_invalid")
        names.add(name)
        labels = row["owner_description_aliases"]
        require(isinstance(labels, list) and labels and len(labels) <= 16
                and all(isinstance(label, str) and 0 < len(label.strip()) <= 160 for label in labels),
                "destination_catalog_alias_invalid")
        normalized = {label.strip().casefold() for label in labels}
        require(len(normalized) == len(labels) and not aliases.intersection(normalized), "destination_catalog_alias_ambiguous")
        aliases.update(normalized)
        reference = row["simready_result"]
        result, static, paths = _destination(checked_file(reference["path"], reference), [0, 0, 0], [1e-9] * 3)
        authoring = read(paths["authoring_receipt"], digest_field="result_digest")
        identity = result["destination_identity"]
        graph = {"schema_version": "task_evaluation_rigid_replacement_graph.v1", "asset_id": identity["id"],
            "asset_version": identity["version"], "articulation_graph": {"joints": []}, "single_rigid_candidate": True,
            "physics_bounds": authoring["candidate_physics_completion"]["physics_bounds"], "physics_authority_granted": False}
        with tempfile.TemporaryDirectory(prefix="scene-catalog-check-") as temporary:
            actual = qualify_scene_configuration_rigid_asset_static(asset_path=paths["asset"], graph_spec=graph,
                authoring_receipt=authoring, replacement_identity=identity,
                output_path=Path(temporary) / "static.json")
        require(_requalification_comparable(actual) == _requalification_comparable(static),
                "destination_catalog_static_requalification_changed")
    return rows


#: Owner-upload sources the preparation service admits by default.
OWNER_UPLOAD_SOURCE_KINDS = ["mesh", "gaussian_splat"]
#: With public_scene_enabled the service additionally admits rights-admitted
#: public-scene persistent intents (Spec A: the legacy public-scene path for
#: scenes such as 841757). It stays a separate no-spend preparation phase; the
#: per-scene public-source binding and public-scene machinery are materialized
#: by the public-scene provisioner, not manufactured here.
PUBLIC_SCENE_SOURCE_KINDS = ["mesh", "gaussian_splat", "public_scene"]


def build_bootstrap(*, destination_catalog, config_root="/etc/blueprint",
                    state_root="/var/lib/blueprint/pipeline-control-plane",
                    inputs_root="/var/lib/blueprint/task-evaluation-inputs",
                    capture_store_root="/var/lib/blueprint/capture-intake",
                    running_repo_root="/opt/blueprint/task-evaluation-control-plane", service_account="blueprint",
                    public_scene_enabled=False):
    rows = validate_destination_catalog(destination_catalog)
    roots = {key: str(safe_path(value)) for key, value in {
        "config_root": config_root, "state_root": state_root, "inputs_root": inputs_root,
        "capture_store_root": capture_store_root}.items()}
    require(Path(running_repo_root).is_absolute(), "scene_preparation_running_repo_invalid")
    pwd.getpwnam(service_account)
    value = {"schema_version": BOOTSTRAP_SCHEMA, "managed_by": MANAGED_BY, **roots,
        "running_repo_root": str(running_repo_root), "service_account": service_account,
        "destination_catalog": rows, "simulation_physics_bounds": {key: list(value) for key, value in DEFAULT_PHYSICS_BOUNDS.items()},
        "execution_activation_enabled": False,
        "supported_source_kinds": PUBLIC_SCENE_SOURCE_KINDS if public_scene_enabled else OWNER_UPLOAD_SOURCE_KINDS}
    value["bootstrap_digest"] = canonical_digest(value, digest_field="bootstrap_digest")
    return value


def _managed_json(path, value, account):
    path = safe_path(path)
    unchanged = False
    if path.exists():
        old = read(path)
        require(old.get("managed_by") == MANAGED_BY, "scene_preparation_unmanaged_file")
        unchanged = old == value
    if not unchanged:
        atomic_json(path, value)
    if os.geteuid() == 0:
        os.chown(path, 0, account.pw_gid)
    path.chmod(0o640)


def install_scene_preparation(*, bootstrap_path):
    bootstrap_path = safe_path(bootstrap_path)
    require(not bootstrap_path.stat().st_mode & 0o022, "scene_preparation_bootstrap_writable")
    bootstrap = read(bootstrap_path, digest_field="bootstrap_digest")
    require(bootstrap.get("schema_version") == BOOTSTRAP_SCHEMA
            and bootstrap.get("managed_by") == MANAGED_BY
            and bootstrap.get("execution_activation_enabled") is False
            and bootstrap.get("supported_source_kinds") in (OWNER_UPLOAD_SOURCE_KINDS, PUBLIC_SCENE_SOURCE_KINDS),
            "scene_preparation_bootstrap_scope_invalid")
    validate_destination_catalog(bootstrap["destination_catalog"])
    from .task_evaluation_scene_configuration_content_agents_driver import _physics_bounds
    _physics_bounds({"required_output": bootstrap["simulation_physics_bounds"]})
    account = pwd.getpwnam(bootstrap["service_account"])
    state, inputs, config_root = (safe_path(bootstrap[key]) for key in ("state_root", "inputs_root", "config_root"))
    owner_queue = state / "task-evaluation-owned-scene-preparations"
    public_scene_enabled = "public_scene" in bootstrap["supported_source_kinds"]
    public_binding_root = inputs / "public-source-bindings"
    directories = [state / "task-evaluation-scene-intents", owner_queue,
        inputs / "owner-source-store", inputs / "completed-scene-preparation",
        inputs / "completed-scene-preparation-inputs", state / "scene-preparation-release-bindings",
        state / "scene-preparation-service", state / "submission-publication-locks",
        state / "disk-reservations", state / "storage-pins"]
    if public_scene_enabled:
        # The public-scene binding directory the per-scene provisioner writes
        # <binding_id>.json into; the scene-progression _source resolver reads
        # public_source_binding_root/<binding_id>.json for a public_scene intent.
        directories.append(public_binding_root)
    for directory in directories:
        safe_path(directory)
        if not directory.exists():
            directory.mkdir(parents=True, mode=0o750)
            if os.geteuid() == 0:
                os.chown(directory, account.pw_uid, account.pw_gid)
    from .task_evaluation_launch_preparation_queue import ensure_launch_preparation_queue_root
    ensure_launch_preparation_queue_root(owner_queue)
    for directory in owner_queue.iterdir():
        if directory.is_dir() and os.geteuid() == 0:
            os.chown(directory, account.pw_uid, account.pw_gid)
    machinery = {"schema_version": "task_evaluation_completed_scene_machinery.v1", "managed_by": MANAGED_BY,
        "maximum_preparation_spend_usd": 0, "provider": "control_plane",
        "destination_catalog": bootstrap["destination_catalog"], "simulation_physics_bounds": bootstrap["simulation_physics_bounds"]}
    machinery["machinery_digest"] = canonical_digest(machinery, digest_field="machinery_digest")
    machinery_path = config_root / "task-evaluation-completed-scene-machinery.json"
    _managed_json(machinery_path, machinery, account)
    config = {"schema_version": "task_evaluation_scene_progression_config.v1", "managed_by": MANAGED_BY,
        "intent_root": str(directories[0]), "capture_store_root": bootstrap["capture_store_root"],
        "factory_output_root": str(inputs / "completed-scene-preparation"),
        "completed_source_machinery_path": str(machinery_path),
        "deployment_receipt_root": str(state / "deploy-receipts"),
        "release_binding_root": str(state / "scene-preparation-release-bindings"),
        "running_repo_root": bootstrap["running_repo_root"], "runtime_publication_root": str(inputs / "system-runtimes"),
        "trusted_clients": ["blueprint-webapp"], "supported_source_kinds": bootstrap["supported_source_kinds"],
        "maximum_intents_per_pass": 16, "maximum_http_submission_attempts": 2,
        "submission_enabled": True, "submission_transport": "local_owned_queue", "activation_enabled": False,
        "service_account": bootstrap["service_account"], "preparation_queue_root": str(owner_queue),
        "publication_lock_root": str(state / "submission-publication-locks"),
        "service_status_path": str(state / "scene-preparation-service/latest.json"),
        "preparation_worker": {"input_root": str(inputs / "completed-scene-preparation-inputs"),
            "construction_queue_root": str(state / "task-evaluation-scene-constructions"),
            "disk_reservation_root": str(state / "disk-reservations"), "storage_pins_root": str(state / "storage-pins"),
            "max_messages": 1, "allowed_uri_prefixes": ["s3://blueprint/task-evaluation/production-inputs/",
                                                         "s3://blueprint/task-evaluation/host-only-owner-sources/"]}}
    if public_scene_enabled:
        # public_source_binding_root + machinery_path let the scene-progression
        # _source resolver bind a public_scene intent. The public-scene machinery
        # (task_evaluation_public_scene_machinery.v1) is release/provider bound and
        # is materialized per scene by the public-scene provisioner, not here — the
        # worker only reads machinery_path once a public_scene intent exists.
        config["public_source_binding_root"] = str(public_binding_root)
        config["machinery_path"] = str(config_root / "task-evaluation-public-scene-machinery.json")
    config["config_digest"] = canonical_digest(config, digest_field="config_digest")
    config_path = config_root / "task-evaluation-scene-progression.json"
    _managed_json(config_path, config, account)
    environment = {"BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT": config["intent_root"],
        "BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS": "blueprint-webapp",
        "BLUEPRINT_TASK_EVALUATION_OWNER_SOURCE_STORE_ROOT": str(inputs / "owner-source-store"),
        "PIPELINE_CAPTURE_INTAKE_STORE_ROOT": bootstrap["capture_store_root"]}
    require(all(not any(c.isspace() for c in value) for value in environment.values()), "scene_preparation_environment_path_invalid")
    content = "# Managed by " + MANAGED_BY + ".\n" + "".join(f"{key}={value}\n" for key, value in sorted(environment.items()))
    env_path = safe_path(config_root / "task-evaluation-scene-progression.env")
    if env_path.exists():
        require(env_path.read_text().startswith("# Managed by " + MANAGED_BY + ".\n"), "scene_preparation_unmanaged_environment")
    if not env_path.exists() or env_path.read_text() != content:
        with tempfile.NamedTemporaryFile(mode="w", dir=config_root, prefix=".scene-preparation-env-", delete=False) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        temporary.chmod(0o640)
        if os.geteuid() == 0:
            os.chown(temporary, 0, account.pw_gid)
        os.replace(temporary, env_path)
    return {"schema_version": "task_evaluation_scene_preparation_installation.v1", "status": "installed",
        "bootstrap": record(bootstrap_path), "config": record(config_path), "environment": record(env_path),
        "machinery": record(machinery_path), "execution_activation_enabled": False,
        "provider_mutation_performed": False, "service_start_requested": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--destination-simready")
    parser.add_argument("--destination-alias", action="append", default=[])
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)
    if args.destination_simready:
        path = safe_path(args.destination_simready)
        identity = read(path, digest_field="result_digest")["destination_identity"]
        bootstrap = build_bootstrap(destination_catalog=[{"binding_id": identity["id"] + "-" + identity["version"],
            "owner_description_aliases": args.destination_alias, "simready_result": record(path)}])
        _managed_json(safe_path(args.bootstrap), bootstrap, pwd.getpwnam(bootstrap["service_account"]))
    result = install_scene_preparation(bootstrap_path=args.bootstrap) if args.install else {"bootstrap": record(args.bootstrap)}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
