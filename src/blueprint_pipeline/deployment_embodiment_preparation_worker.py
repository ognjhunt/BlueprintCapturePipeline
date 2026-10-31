"""Fixed, isolated Isaac imports of a sealed public source batch; no policies."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

from .deployment_embodiment_preparation_contract import (
    INPUT_FILENAME, INPUT_SCHEMA, MAX_OUTPUT_BYTES, RESULT_FILENAME, RESULT_SCHEMA,
    read, require, sealed, sha, validate_request, write_once,
)


def verify_inputs(runtime: Path) -> dict:
    from .decision_evidence_contracts import canonical_digest
    from .deployment_embodiment_assets import verify_source_bundle
    inputs = read(runtime / INPUT_FILENAME)
    require(inputs.get("schema_version") == INPUT_SCHEMA and inputs.get("inputs_digest")
            == canonical_digest(inputs, digest_field="inputs_digest"), "worker_inputs_invalid")
    request = validate_request(inputs["request"], allow_expired=True)
    require([row["embodiment_id"] for row in inputs["sources"]] == request["embodiment_ids"], "worker_sources_mismatch")
    for row in inputs["sources"]:
        if row.get("status") == "source_unavailable":
            continue
        source = verify_source_bundle(runtime / "robot_sources" / row["embodiment_id"])
        require(source["bundle_digest"] == row["bundle_digest"], "worker_source_changed")
    return inputs


def import_one(runtime: Path, output: Path, identity: str) -> dict:
    from .deployment_embodiment_runtime_configuration import prepare_runtime_configuration, bind_prepared_configuration
    from .deployment_embodiment_asset_import import import_source_asset
    from .deployment_embodiment_isaac_launch import launch_embodiment_isaaclab

    inputs = verify_inputs(runtime)
    require(identity in inputs["request"]["embodiment_ids"], "worker_identity_unbound")
    output.mkdir(parents=True, exist_ok=True)
    config = prepare_runtime_configuration(runtime / "robot_sources" / identity, output / "configuration")
    app, launch = launch_embodiment_isaaclab(Path(os.environ["BLUEPRINT_EMBODIMENT_PROVISIONING_RECEIPT"]),
        output=output / "import-launch", source_format=config.get("source_model", {}).get("format", "urdf"))
    try:
        imported = import_source_asset(output / "configuration", output / "prepared", fix_base=True,
                                        station_parameters=config["station_parameters"])
        # This is an operator-store identity, not a public download or an upload.
        # The controller stages the verified prepared closure for this exact URL.
        url = "https://robot-models.blueprint.internal/" + imported["model"]["sha256"][7:] + "/robot.usda"
        binding = bind_prepared_configuration(output / "configuration", output / "prepared", url)
        write_once(output / "binding-inputs.json", binding)
        row = {"embodiment_id": identity, "status": "imported", "source_bundle_digest": config["source_bundle_digest"],
               "configuration_digest": config["configuration_digest"], "import_digest": imported["import_digest"],
               "model_sha256": imported["model"]["sha256"], "source_dynamics": config["source_dynamics"],
               "source_limitations": config["source_limitations"], "native_import_executed": True,
               "task_executed": False, "policy_executed": False, "launcher": launch}
        from .decision_evidence_contracts import canonical_digest
        write_once(output / "import-checkpoint.json", sealed({
            "schema_version": "embodiment_import_checkpoint.v1", "body_result": row,
            "binding_digest": canonical_digest(binding)}, "checkpoint_digest"))
        seal_verified_import(output, identity)
    except Exception:
        # Fast shutdown can suppress the pending exception as well as return
        # code. Preserve its exact diagnostic in the parent's bounded log.
        import traceback
        traceback.print_exc()
        print_imported_structure(output / "prepared")
        sys.stderr.flush()
        raise
    finally:
        # Isaac's default fast shutdown terminates this process. The batch
        # parent owns diagnostics and the final body receipt after verifying
        # the checkpoint above; nothing required may follow this close.
        app.close()
    return row


def print_imported_structure(prepared: Path) -> None:
    """A binding mismatch is diagnosable only from the names the importer produced.

    Failed bodies are not retained, so their bounded log carries the imported
    body paths, joint names, world anchors and any source names the importer
    recorded for prims it renamed (public asset names only).
    """
    import json
    try:
        structure = json.loads((prepared / "asset-import.json").read_text(encoding="utf-8"))["structure"]
        summary = {key: list(structure.get(key) or [])[:400]
                   for key in ("body_paths", "joint_names", "world_anchor_links")}
        summary.update({key: dict(list(structure[key].items())[:400])
                        for key in ("body_source_names", "joint_source_names") if structure.get(key)})
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return
    print("imported_structure " + json.dumps(summary, sort_keys=True), file=sys.stderr)


def _closure_digest(body: Path) -> str:
    """Every configuration, prepared-model and binding byte, recomputable without USD."""
    from .decision_evidence_contracts import canonical_digest
    files = [p for root in ("configuration", "prepared") for p in sorted((body / root).rglob("*")) if p.is_file()]
    return canonical_digest({"files": [[p.relative_to(body).as_posix(), sha(p)]
                                       for p in [*files, body / "binding-inputs.json"]]})


def seal_verified_import(body: Path, identity: str) -> dict:
    """Reopen the closure with Kit's USD and seal what was verified for the batch parent.

    Isaac Sim 6's python.sh no longer puts pxr on the path, so the batch parent
    cannot reopen USD (public-robot-imports-20261030-07). The import process can,
    while Kit is up; the parent then recomputes the closure digest from disk.
    """
    from .decision_evidence_contracts import canonical_digest
    row = verified_import_checkpoint(body, identity)
    checkpoint = read(body / "import-checkpoint.json")
    value = sealed({
        "schema_version": "embodiment_verified_import.v1", "checkpoint_digest": checkpoint["checkpoint_digest"],
        "binding_digest": canonical_digest(read(body / "binding-inputs.json")), "body_result": row,
        "closure_digest": _closure_digest(body)}, "verification_digest")
    write_once(body / "verified-import.json", value)
    return value


def checkpoint_row(body: Path, identity: str) -> dict:
    """The batch parent's check before any diagnostic or retention; needs no USD."""
    from .decision_evidence_contracts import canonical_digest
    checkpoint = read(body / "import-checkpoint.json")
    verified = read(body / "verified-import.json")
    require(checkpoint.get("schema_version") == "embodiment_import_checkpoint.v1"
            and checkpoint.get("checkpoint_digest") == canonical_digest(checkpoint, digest_field="checkpoint_digest")
            and verified.get("schema_version") == "embodiment_verified_import.v1"
            and verified.get("verification_digest") == canonical_digest(verified, digest_field="verification_digest")
            and verified["checkpoint_digest"] == checkpoint["checkpoint_digest"], "import_checkpoint_invalid")
    require(canonical_digest(read(body / "binding-inputs.json")) == checkpoint["binding_digest"] == verified["binding_digest"]
            and _closure_digest(body) == verified["closure_digest"], "import_checkpoint_binding_changed")
    row = dict(checkpoint["body_result"])
    require(row == verified["body_result"] and row.get("embodiment_id") == identity and row.get("status") == "imported"
            and row.get("task_executed") is False and row.get("policy_executed") is False,
            "import_checkpoint_identity_changed")
    return row


def verified_import_checkpoint(body: Path, identity: str) -> dict:
    """Reopen the actual source/model closure (needs Kit's USD) before sealing it."""
    from .decision_evidence_contracts import canonical_digest
    from .deployment_embodiment_runtime_configuration import bind_prepared_configuration
    checkpoint = read(body / "import-checkpoint.json")
    require(checkpoint.get("schema_version") == "embodiment_import_checkpoint.v1"
            and checkpoint.get("checkpoint_digest") == canonical_digest(checkpoint, digest_field="checkpoint_digest"),
            "import_checkpoint_invalid")
    recorded = read(body / "binding-inputs.json")
    verified = bind_prepared_configuration(body / "configuration", body / "prepared",
        recorded["robot_embodiment"]["model"]["url"], runtime_settings=recorded["runtime_settings"])
    require(recorded == verified and canonical_digest(verified) == checkpoint["binding_digest"], "import_checkpoint_binding_changed")
    row = dict(checkpoint["body_result"])
    provenance = verified["model_record"]["source_import_provenance"]
    require(row.get("embodiment_id") == identity == verified["robot_embodiment"]["embodiment_id"]
            and row.get("status") == "imported" and row.get("task_executed") is False and row.get("policy_executed") is False
            and row["model_sha256"] == verified["model_record"]["sha256"]
            and row["configuration_digest"] == verified["configuration_digest"]
            and row["import_digest"] == provenance["import_digest"]
            and row["source_bundle_digest"] == provenance["source_bundle_digest"], "import_checkpoint_identity_changed")
    return row


_LOGGED_FAILURE = re.compile(rb"^[A-Za-z_.]*(?:Error|Exception): ([a-z][a-z0-9_]{1,127})(?=[:\s]|$)", re.M)


def logged_failure(log: Path) -> str | None:
    """Isaac's fast shutdown can exit 0 after a traceback; keep its fixed code.

    Only a lowercase error code is published, never the message detail after it.
    """
    try:
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 256 * 1024))
            found = _LOGGED_FAILURE.findall(stream.read())
    except OSError:
        return None
    return found[-1].decode() if found else None


def recover_import_checkpoint(body: Path, identity: str, reason: str) -> dict:
    """An interrupted diagnostic cannot erase an already verified import."""
    row = checkpoint_row(body, identity)
    row["native_controller_smoke"] = {"status": "failed", "reasons": ["import_process_interrupted_after_checkpoint:" + reason],
                                      "smoke_digest": None, "checkpoint_recovered": True}
    prior = body / "body-result.json"
    if prior.exists():
        prior.rename(body / "body-result-before-interruption.json")
    write_once(prior, row)
    return row

def run_batch(runtime: Path, output: Path, *, run=subprocess.run, now=time.monotonic, wall=time.time) -> dict:
    from .deployment_embodiment_native_smoke import run_bounded_smoke_process

    inputs = verify_inputs(runtime)
    output.mkdir(parents=True, exist_ok=True)
    provider_deadline = float(os.environ["BLUEPRINT_POLICY_CANARY_HARD_DEADLINE_EPOCH"])
    deadline = now() + max(0, min(provider_deadline, inputs["request"]["expires_at_epoch"]) - wall() - 600)
    rows, retained_bytes = [], 0
    for identity in inputs["request"]["embodiment_ids"]:
        source = next(row for row in inputs["sources"] if row["embodiment_id"] == identity)
        if source.get("status") == "source_unavailable":
            rows.append({"embodiment_id": identity, "status": "not_imported",
                         "reason": "source_unavailable", "source_reason": source["reason"]})
            continue
        remaining = deadline - now()
        if remaining < 60:
            rows.append({"embodiment_id": identity, "status": "not_imported", "reason": "batch_time_budget_exhausted"})
            continue
        with tempfile.TemporaryDirectory(prefix="robot-import-") as temporary:
            body = Path(temporary) / identity
            log = output / (identity + ".log")
            try:
                body_timeout = min(1200, remaining - 30)
                body_deadline = wall() + body_timeout
                with log.open("wb") as stream:
                    process = run([sys.executable, "-m", "blueprint_pipeline.deployment_embodiment_preparation_worker",
                                   "one", "--runtime", str(runtime),
                                   "--output", str(body), "--embodiment", identity], stdout=stream, stderr=subprocess.STDOUT,
                                  timeout=body_timeout, check=False,
                                  env={**os.environ, "BLUEPRINT_EMBODIMENT_BODY_DEADLINE_EPOCH": str(body_deadline)})
                if process.returncode != 0:
                    row = {"embodiment_id": identity, "status": "import_failed", "reason": "isaac_import_process_failed",
                           "exit_code": process.returncode, "log": log.name}
                else:
                    row = checkpoint_row(body, identity)
                    # Keep the original per-body deadline and both provider
                    # clocks. SDK shutdown must not grant a fresh smoke window.
                    smoke = run_bounded_smoke_process(body, wall_seconds=max(0., min(
                        90., deadline - now(), body_deadline - wall() - 5,
                        min(provider_deadline, inputs["request"]["expires_at_epoch"]) - wall() - 600)))
                    row["native_controller_smoke"] = {key: smoke[key] for key in (
                        "status", "process_digest", "smoke_digest", "reasons")}
                    write_once(body / "body-result.json", row)
            except subprocess.TimeoutExpired:
                row = {"embodiment_id": identity, "status": "import_failed", "reason": "isaac_import_timeout", "log": log.name}
            except Exception as exc:  # noqa: BLE001 - one body's failure never stops the rest of the batch
                reason = (logged_failure(log) if not (body / "import-checkpoint.json").exists() else None)
                row = {"embodiment_id": identity, "status": "import_failed", "reason": reason or type(exc).__name__,
                       "log": log.name}
            if row["status"] != "imported" and (body / "import-checkpoint.json").is_file():
                try:
                    row = recover_import_checkpoint(body, identity, row["reason"])
                except Exception as exc:  # noqa: BLE001 - an unverifiable checkpoint is this body's failure only
                    row["checkpoint_recovery"] = "unverified:" + type(exc).__name__
            if row["status"] == "imported":
                try:
                    size = sum(p.stat().st_size for p in body.rglob("*") if p.is_file())
                    if retained_bytes + size > MAX_OUTPUT_BYTES - 64 * 1024**2:
                        row = {"embodiment_id": identity, "status": "not_retained", "reason": "batch_output_budget_exhausted"}
                    else:
                        destination = output / "bodies" / identity
                        destination.parent.mkdir(exist_ok=True)
                        shutil.move(str(body), destination)
                        retained_bytes += size
                except OSError:
                    row = {"embodiment_id": identity, "status": "not_retained", "reason": "import_retention_io_failed"}
            if row["status"] != "imported" and (body / "configuration/runtime-configuration.json").is_file():
                try:
                    configuration = read(body / "configuration/runtime-configuration.json")
                    row.update(source_dynamics=configuration["source_dynamics"], source_limitations=configuration["source_limitations"])
                except (OSError, ValueError, KeyError, TypeError):
                    row["configuration_diagnostic"] = "source_dynamics_receipt_unreadable"
            # Provider startup logs can be verbose. Retain a bounded tail per body.
            if log.exists() and log.stat().st_size > 1024**2:
                with log.open("rb") as stream:
                    stream.seek(-1024**2, os.SEEK_END)
                    tail = stream.read()
                log.write_bytes(tail)
            rows.append(row)
    files = [{"path": p.relative_to(output).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha(p)}
             for p in sorted(output.rglob("*")) if p.is_file() and p.name != RESULT_FILENAME]
    require(sum(row["size_bytes"] for row in files) <= MAX_OUTPUT_BYTES, "output_bound_exceeded")
    result = sealed({"schema_version": RESULT_SCHEMA, "status": "completed", "claim_ceiling": "robot_asset_import_only",
        "candidate_policy_queried": False, "task_executed": False, "inputs_digest": inputs["inputs_digest"],
        "request_digest": inputs["request"]["request_digest"], "results": rows, "files": files,
        "provider_teardown_verified": False}, "result_digest")
    write_once(output / RESULT_FILENAME, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("one", "batch"))
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embodiment")
    args = parser.parse_args(argv)
    if args.mode == "one":
        import_one(args.runtime.resolve(), args.output, args.embodiment)
    else:
        run_batch(args.runtime.resolve(), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
