"""Offline consent-withdrawal drill: one synthetic site, from the WebApp's withdrawal vector to a sealed receipt.

Beta journey step: site withdrawal (tracker G-02, exit criterion E5). It runs at $0
with no network: every root is under one temporary work root, the B2 artifact store
and the capture source bucket are in-memory fakes driven through the same clients
the operator command uses, and nothing outside the work root is read or written.

The drill walks the real path:

1. the WebApp's withdrawal command (``tests/fixtures/website-withdrawal-vector.json``,
   shared with the WebApp) is acknowledged as a tombstone (``acknowledge_withdrawal``);
2. the plan the intake starts by itself after an acknowledgement is recorded beside
   the tombstone (``request_automatic_deletion_plan`` running the deletion CLI's
   ``--record-automatic-plan``), and nothing is deleted;
3. the paid screening launch's current-source check for the site now refuses
   (``beta_policy_screening_worker.require_current_source``, which resolves the
   capture through ``team_scene_capture_root.resolve_capture_root``);
4. an operator plan with the object store and the source bucket, then ``apply`` with
   that plan's digest and the explicit authorization, then the sealed receipt.

It checks that every data class of the beta retention policy appears in the receipt
as removed or remaining with a reason, and that nothing outside the site's copies
changed, and prints one row per copy class: count before, count after, and why
anything remains.

``python -m blueprint_pipeline.website_capture_withdrawal_drill [--work-root DIR]``
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

BUCKET = "synthetic-artifacts"
SOURCE_BUCKET = "synthetic-source"
_PREFIX = "blueprint/arm-decision-proof-v1/configured-scenes/artifacts"
DEFAULT_VECTOR = Path(__file__).resolve().parents[2] / "tests/fixtures/website-withdrawal-vector.json"


class DrillFailure(AssertionError):
    """The drill found a withdrawal property that does not hold."""


def _check(condition: Any, message: str) -> None:
    if not condition:
        raise DrillFailure(message)


class MemoryB2:
    """The boto3 S3 version calls the existing ``B2ObjectVersions`` client makes, in memory."""

    def __init__(self) -> None:
        self.objects: dict[str, list[dict[str, Any]]] = {}
        self.deleted: list[tuple[str, str]] = []

    def put(self, uri: str, body: bytes, *, versions: int = 1) -> None:
        self.objects[self.key(uri)] = [{"VersionId": f"v{index}", "Body": body} for index in range(versions)]

    @staticmethod
    def key(uri: str) -> str:
        return uri.split(f"s3://{BUCKET}/", 1)[1]

    def versions_of(self, uri: str) -> int:
        return len(self.objects.get(self.key(uri), []))

    def list_object_versions(self, *, Bucket: str, Prefix: str, **_markers: Any) -> dict[str, Any]:
        _check(Bucket == BUCKET, "drill_b2_bucket_unexpected")
        versions = [{"Key": key, "VersionId": row["VersionId"]}
                    for key, rows in self.objects.items() if key.startswith(Prefix) for row in rows]
        return {"Versions": versions, "DeleteMarkers": [], "IsTruncated": False}

    def delete_object(self, *, Bucket: str, Key: str, VersionId: str) -> None:
        _check(Bucket == BUCKET, "drill_b2_bucket_unexpected")
        self.deleted.append((Key, VersionId))
        self.objects[Key] = [row for row in self.objects[Key] if row["VersionId"] != VersionId]

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        return {"Body": io.BytesIO(self.objects[Key][0]["Body"])}


class MemoryGcs:
    """The existing exact GCS cleanup storage's calls, in memory."""

    bucket_name = SOURCE_BUCKET

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def objects(self, prefix: str) -> list[dict[str, Any]]:
        return [dict(row) for row in self.rows if row["name"].startswith(prefix)]

    def soft_deleted_objects(self, prefix: str) -> list[dict[str, Any]]:
        return []

    def delete(self, name: str, generation: str) -> None:
        self.rows = [row for row in self.rows if (row["name"], row["generation"]) != (name, generation)]


def _write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
    return path


def _sealed(value: dict[str, Any], field: str) -> dict[str, Any]:
    value = {**value, field: ""}
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _uri(kind: str, digit: str, name: str) -> str:
    return f"s3://{BUCKET}/{_PREFIX}/{kind}/sha256/{digit * 64}/{name}"


def _capture(partition: Path, request_id: str) -> tuple[Path, dict[str, Any]]:
    scene_id, capture_id = f"site-{request_id}", f"walkthrough-{request_id}"
    capture = partition / "scenes" / scene_id / "captures" / capture_id
    _write(capture / "raw/manifest.json", {"site_submission_id": request_id, "scene_id": scene_id,
                                           "capture_id": capture_id, "consent_status": "granted"})
    _write(capture / "raw/video.mp4", b"synthetic-video " + request_id.encode())
    _write(capture / "capture_descriptor.json", {"scene_id": scene_id, "capture_id": capture_id})
    context = _sealed({"request_id": request_id, "scene_id": scene_id, "capture_id": capture_id,
                       "task": "synthetic drill task"}, "context_digest")
    _write(capture / "pipeline/website_task_context.json", context)
    _write(capture / "pipeline/website_scene_preparation/preparation.json", {"scene_id": scene_id})
    source = {"capture_id": capture_id, "task_context_digest": context["context_digest"],
              "task_id": "website-" + context["context_digest"][7:27], "scene_intent_id": "synthetic"}
    return capture, source


def build_world(work: Path, request_id: str) -> dict[str, Any]:
    """The withdrawn site, one other site, and every root the deletion reads, all under ``work``."""

    partition, control, inputs = work / "capture-partition", work / "control-plane", work / "inputs"
    capture, source = _capture(partition, request_id)
    other_capture, _ = _capture(partition, "drill-other")
    launches, canaries = control / "task-evaluation-launch-runs", control / "task-evaluation-policy-canaries"
    prepared = inputs / "completed-scene-preparation"
    for root in (launches, canaries, prepared):
        root.mkdir(parents=True)
    # Lineage: the site's sealed website source registration and the scene intent it serves.
    request = {"owner": {"tenant": "drill-tenant"}, "submission_id": f"drill-{request_id}",
               "source": {"kind": "website"}}
    intent_id = "scene-" + hashlib.sha256(request_id.encode()).hexdigest()
    _write(control / "task-evaluation-scene-intents" / intent_id / "intent.json",
           {"intent_id": intent_id, "request": request})
    _write(control / "website-source-bindings" / "registration.json", _sealed({
        "schema_version": "website_scene_source_registration.v1",
        "request_digest": cross_runtime_canonical_digest(request),
        "capture_source": {"request_id": request_id, "scene_id": f"site-{request_id}",
                           "capture_id": f"walkthrough-{request_id}"},
        "references": {"preparation": {"path": str(capture / "pipeline/website_scene_preparation/preparation.json")}}},
        "registration_digest"))
    # A launch run that names only the site's scene intent, and a team run naming that launch whose
    # result artifact is on B2.
    launch = launches / f"launch-{request_id}-construction-0001"
    _write(launch / "launch_profile.json", {"scene_attempt_binding": {"intent_id": intent_id}})
    _write(launch / "frames/0001.png", b"\x89PNG synthetic frame")
    canary = canaries / "team-scored-run-drill-0001"
    _write(canary / "dispatch_receipt.json", {"source_launch_id": launch.name})
    review = _uri("task-evaluation-result", "a", "review.mp4")
    _write(canary / "artifacts/result_delivery/remote_artifacts/review.json",
           {"schema_version": "task_evaluation_result_artifact_remote_reference.v1", "uri": review})
    # An offloaded launch run whose sealed pointer names the site.
    offloaded = _uri("control-plane-evidence", "b", "evidence.tar")
    receipt = json.dumps({"launch_id": f"launch-{request_id}-controls-0002", "scene_id": f"site-{request_id}"})
    _write(launches / f"launch-{request_id}-controls-0002.offloaded.v1.json", _sealed({
        "schema_version": "control_plane_evidence_offload_pointer.v1",
        "directory": f"launch-{request_id}-controls-0002", "uri": offloaded, "members": [],
        "retained_receipt_bytes": {"launch_receipt.json": base64.b64encode(receipt.encode()).decode()}},
        "pointer_digest"))
    _write(prepared / intent_id / "website-source/binding.json", {"intent_digest": "sha256:" + "e" * 64})
    # The other site's run and B2 object: never touched.
    other_run = launches / "launch-drill-other-construction-0003"
    _write(other_run / "launch_profile.json", {"capture_root": str(other_capture)})
    other_uri = _uri("task-evaluation-result", "f", "review.mp4")
    _write(other_run / "artifacts/result_delivery/remote_artifacts/review.json",
           {"schema_version": "task_evaluation_result_artifact_remote_reference.v1", "uri": other_uri})
    store = MemoryB2()
    store.put(review, b"review bytes")
    store.put(offloaded, b"offloaded evidence", versions=2)
    store.put(_uri("atlas-scene-bundle", "3", f"walkthrough-{request_id}.zip"), b"atlas preview", versions=2)
    store.put(other_uri, b"another site's review")
    gcs = MemoryGcs([{"name": f"scenes/site-{request_id}/captures/walkthrough-{request_id}/raw/video.mp4",
                      "generation": "7", "size": 15, "crc32c": "AAAAAA=="},
                     {"name": "scenes/site-drill-other/captures/walkthrough-drill-other/raw/video.mp4",
                      "generation": "8", "size": 15, "crc32c": "AAAAAA=="}])
    # A Pipeline root the deletion does not cover yet; the receipt names it with its reason.
    uncovered = inputs / "sam31-preparations"
    _write(uncovered / "drill-entry/record.json", {"note": "synthetic"})
    resolution = {"run_roots": (launches, canaries, prepared), "intent_root": control / "task-evaluation-scene-intents",
                  "binding_root": control / "website-source-bindings", "uncovered_roots": (uncovered,),
                  "screening_queue_root": control / "beta-policy-screening",
                  "screening_work_root": work / "work/beta-policy-screening-work",
                  "native_g1_registration_root": control / "native-g1-site-registrations",
                  "team_scene_halves_root": inputs / "team-scene-halves",
                  "team_registry": work / "etc/team-evaluation-registry.json",
                  "team_scene_packet_queue_root": control / "team-scene-packets",
                  "rigid_probe_queue_root": control / "rigid-robot-free-probes",
                  "rigid_probe_work_root": work / "work/rigid-robot-free-probe-work"}
    return {"work": work, "partition": partition, "capture": capture, "site": capture.parents[1],
            "source": source, "store": store, "gcs": gcs, "resolution": resolution,
            "site_runs": [launch, canary, prepared / intent_id], "other_run": other_run,
            "other_capture": other_capture, "other_uri": other_uri}


def _cli_root_arguments(resolution: dict[str, Any]) -> list[str]:
    arguments = []
    for root in resolution["run_roots"]:
        arguments += ["--run-root", str(root)]
    for root in resolution["uncovered_roots"]:
        arguments += ["--uncovered-root", str(root)]
    for key in ("intent_root", "binding_root", "screening_queue_root", "screening_work_root",
                "native_g1_registration_root", "team_scene_halves_root", "team_registry",
                "team_scene_packet_queue_root", "rigid_probe_queue_root", "rigid_probe_work_root"):
        arguments += ["--" + key.replace("_", "-"), str(resolution[key])]
    return arguments


def _snapshot(paths: list[Path]) -> dict[str, str]:
    rows = {}
    for base in paths:
        for path in sorted(base.rglob("*")):
            if path.is_file():
                rows[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows


@contextlib.contextmanager
def _environment(name: str, value: str) -> Iterator[None]:
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _site_files(site: Path) -> int:
    return sum(1 for path in site.rglob("*") if path.is_file() and "website_withdrawal" not in path.parts)


# Audit events that change the file system, and the positions of their (path, dir_fd) pairs.
_MUTATING = {"os.remove": ((0, 1),), "os.rmdir": ((0, 1),), "os.mkdir": ((0, 2),), "shutil.rmtree": ((0, 1),),
             "os.rename": ((0, 2), (1, 3)), "os.link": ((1, 3),), "os.symlink": ((1, 2),),
             "os.chmod": ((0, 2),), "os.utime": ((0, 3),), "os.truncate": ((0, None),)}
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_CONFINED: list[dict[str, Any]] = []
_HOOKED: list[bool] = []


def _resolve(path: Any, dir_fd: Any) -> str | None:
    if isinstance(path, int) or path is None:
        return None
    path = os.fsdecode(path)
    if os.path.isabs(path):
        return os.path.normpath(path)
    base = os.readlink(f"/proc/self/fd/{dir_fd}") if isinstance(dir_fd, int) and dir_fd >= 0 else os.getcwd()
    return os.path.normpath(os.path.join(base, path))


def _audit(event: str, arguments: tuple[Any, ...]) -> None:
    if not _CONFINED:
        return
    confinement = _CONFINED[-1]
    if event == "open":
        path, mode, flags = (tuple(arguments) + (None, None, None))[:3]
        writing = (isinstance(mode, str) and any(char in mode for char in "wax+")) or (
            isinstance(flags, int) and flags & _WRITE_FLAGS)
        pairs = [(path, None)] if writing else []
    else:
        pairs = [(arguments[index], arguments[fd] if fd is not None else None)
                 for index, fd in _MUTATING.get(event, ())]
    for path, dir_fd in pairs:
        target = _resolve(path, dir_fd)
        # Bytecode caches are the interpreter's, not the drill's.
        if target and not (target == confinement["root"] or target.startswith(confinement["root"] + os.sep)
                           or f"{os.sep}__pycache__{os.sep}" in target):
            confinement["violations"].append(f"{event}:{target}")


@contextlib.contextmanager
def _confined(work: Path) -> Iterator[None]:
    """Fail the drill if it writes, removes, renames or creates anything outside ``work``.

    Every file-system change Python audits is checked (relative names against their
    directory descriptor, or against ``work``, which the drill runs in), and the work
    root's parent directory must list the same entries afterwards. The host lock is
    redirected into ``work``.
    """
    from .website_capture_withdrawal_automatic_plan import LOCK_ENV

    root = str(Path(work).resolve())
    if not _HOOKED:
        import sys
        sys.addaudithook(_audit)
        _HOOKED.append(True)
    parent = Path(root).parent
    siblings = sorted(os.listdir(parent))
    previous = os.getcwd()
    confinement = {"root": root, "violations": []}
    with _environment(LOCK_ENV, str(Path(root) / "host-lock/website-withdrawal-automatic-plan.lock")):
        os.chdir(root)
        _CONFINED.append(confinement)
        try:
            yield
        finally:
            _CONFINED.remove(confinement)
            os.chdir(previous)
    _check(not confinement["violations"], "drill_wrote_outside_work_root: " + ", ".join(confinement["violations"][:5]))
    _check(sorted(os.listdir(parent)) == siblings, "drill_changed_the_work_roots_parent")


def run_drill(work: Path, *, vector_path: Path = DEFAULT_VECTOR) -> dict[str, Any]:
    """Run the whole drill under ``work``; raises ``DrillFailure`` when a property does not hold."""

    with _confined(Path(work)):
        result = _run_drill(Path(work), vector_path=vector_path)
    result["outside_work_root_unchanged"] = True
    return result


def _run_drill(work: Path, *, vector_path: Path) -> dict[str, Any]:
    from . import website_capture_withdrawal as withdrawal
    from . import website_capture_withdrawal_deletion as deletion
    from .beta_policy_screening_worker import require_current_source
    from .team_scene_capture_root import (
        CAPTURE_PARTITION_ENV,
        TeamScenePacketRequestRefused,
        resolve_capture_root,
    )

    vector = json.loads(Path(vector_path).read_text())
    command = vector["command"]
    _check(canonical_digest(command) == vector["command_digest"], "drill_vector_command_digest_mismatch")
    world = build_world(Path(work), command["request_id"])
    capture, site, store, gcs = world["capture"], world["site"], world["store"], world["gcs"]
    resolution = {**world["resolution"], "object_store": deletion.B2ObjectVersions(store, BUCKET)}
    outside = [world["other_capture"], world["other_run"]]
    outside_before = _snapshot(outside)
    other_before = (store.versions_of(world["other_uri"]), [row for row in gcs.rows if "drill-other" in row["name"]])

    # Before the withdrawal the paid launch path resolves the site's capture.
    _check(resolve_capture_root(world["source"], partition=world["partition"]) == capture.resolve(),
           "drill_capture_not_admissible_before_withdrawal")

    # 1. The WebApp's command becomes the Pipeline's tombstone and acknowledgement.
    acknowledgement = withdrawal.acknowledge_withdrawal(capture_root=capture, command=command)
    _check(acknowledgement["command_digest"] == vector["command_digest"]
           and acknowledgement["pipeline_acknowledged"] is True and acknowledgement["deletion_confirmed"] is False
           and acknowledgement["future_processing_allowed"] is False, "drill_acknowledgement_unexpected")

    # 2. The plan the intake starts by itself: the exact CLI command, run here with the drill's roots.
    started: list[list[str]] = []

    def launch(arguments: list[str]) -> None:
        started.append(arguments)
        _check("--apply" not in arguments and "--authorize-complete-deletion" not in arguments,
               "drill_automatic_plan_would_apply")
        deletion.main([*arguments[3:], *_cli_root_arguments(world["resolution"])])

    with contextlib.redirect_stdout(io.StringIO()):
        state = withdrawal.request_automatic_deletion_plan(capture_root=capture, launch=launch)
    automatic = json.loads((site / "website_withdrawal" / withdrawal.AUTOMATIC_PLAN_FILE).read_text())
    _check(state == "started" and len(started) == 1, "drill_automatic_plan_not_started")
    _check(automatic["applied"] is False and automatic["tombstone_digest"] == acknowledgement["tombstone_digest"],
           "drill_automatic_plan_unexpected")
    _check(store.deleted == [] and _site_files(site) == automatic["summary"]["scene_files"],
           "drill_automatic_plan_deleted_something")
    _check(withdrawal.request_automatic_deletion_plan(capture_root=capture, launch=launch) == "recorded"
           and len(started) == 1, "drill_automatic_plan_not_idempotent")

    # 3. A new paid admission for the site is refused once the tombstone exists.
    with _environment(CAPTURE_PARTITION_ENV, str(world["partition"])):
        try:
            require_current_source(world["source"], site_id=f"site-{command['request_id']}")
        except TeamScenePacketRequestRefused as error:
            admission_refusal = str(error)
        else:
            raise DrillFailure("drill_paid_admission_not_refused")
    _check(admission_refusal == "team_scene_capture_withdrawn", "drill_paid_admission_refusal_unexpected")

    # 4. The operator's plan with the object store and the source bucket, then apply exactly that plan.
    plan = deletion.plan_withdrawal_deletion(capture_root=capture, cloud_storage=gcs, **resolution)
    _check(plan["unresolved"] == [], "drill_plan_unresolved")
    before = {"scene_files": plan["scene_files"]["count"],
              "run_directories": sum(1 for run in plan["runs"] if run["directory"]),
              "run_pointers": sum(len(run["pointers"]) for run in plan["runs"]),
              "b2_object_versions": sum(len(row["versions"]) for row in plan["objects"]),
              "capture_source_bucket_objects": len(gcs.objects(f"scenes/{plan['scene_id']}/"))}
    receipt = deletion.apply_withdrawal_deletion(capture_root=capture, expected_plan_digest=plan["digest"],
                                                 authorize_complete_deletion=True, cloud_storage=gcs, **resolution)
    after = {"scene_files": _site_files(site),
             "run_directories": sum(1 for run in plan["runs"] if run["directory"] and Path(run["directory"]).exists()),
             "run_pointers": sum(Path(path).exists() for run in plan["runs"] for path in run["pointers"]),
             "b2_object_versions": sum(store.versions_of(row["uri"]) for row in plan["objects"]),
             "capture_source_bucket_objects": len(gcs.objects(f"scenes/{plan['scene_id']}/"))}
    _check(all(value == 0 for value in after.values()), "drill_covered_copies_remaining")

    # Every data class the retention policy names is in the receipt, removed or remaining with a reason.
    classes = {row["class"]: row for row in receipt["policy_copy_classes"]}
    _check(set(classes) == set(deletion.POLICY_COPY_CLASSES), "drill_policy_class_missing")
    _check(all(row["removed"] or all(item["reason"] for item in row["remaining"]) for row in classes.values()),
           "drill_policy_class_without_reason")
    _check(all(row.get("reason") for row in receipt["remaining"]), "drill_remaining_without_reason")

    # Nothing outside the site's copies changed (and ``_confined`` checks nothing outside ``work`` did).
    _check(_snapshot(outside) == outside_before, "drill_touched_another_site")
    _check((store.versions_of(world["other_uri"]),
            [row for row in gcs.rows if "drill-other" in row["name"]]) == other_before, "drill_touched_another_object")
    work_root = str(Path(work).absolute())
    _check(all(str(path).startswith(work_root) for run in plan["runs"] for path in
               (run["directory"] or work_root, *run["pointers"], *run["leftovers"])), "drill_path_outside_work_root")

    rows = [{"copy_class": name, "before": before[name], "after": after[name], "remaining_reason": None}
            for name in before]
    rows += [{"copy_class": f"policy:{name}", "before": None, "after": None,
              "remaining_reason": ("removed: " + ", ".join(row["removed"]) + "; remaining: " if row["removed"] else "")
              + "; ".join(item["reason"] for item in row["remaining"])}
             for name, row in classes.items()]
    rows += [{"copy_class": f"remaining:{row['holder']}", "before": None, "after": None,
              "remaining_reason": row["reason"]} for row in receipt["remaining"]]
    return {"vector_command_digest": vector["command_digest"], "acknowledgement": acknowledgement,
            "automatic_plan": automatic, "paid_admission_refusal": admission_refusal, "plan": plan,
            "receipt": receipt, "record": rows}


def format_record(rows: list[dict[str, Any]]) -> str:
    lines = ["copy_class | before | after | remaining_reason"]
    for row in rows:
        counts = "-" if row["before"] is None else str(row["before"])
        after = "-" if row["after"] is None else str(row["after"])
        lines.append(f"{row['copy_class']} | {counts} | {after} | {row['remaining_reason'] or '-'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-root", type=Path, help="An empty directory to run in (default: a new temporary one)")
    parser.add_argument("--vector", type=Path, default=DEFAULT_VECTOR)
    args = parser.parse_args(argv)
    if args.work_root is not None:
        args.work_root.mkdir(parents=True, exist_ok=True)
        if any(args.work_root.iterdir()):
            parser.error("--work-root must be empty")
        result = run_drill(args.work_root, vector_path=args.vector)
    else:
        with tempfile.TemporaryDirectory(prefix="withdrawal-drill-") as work:
            result = run_drill(Path(work), vector_path=args.vector)
    print(format_record(result["record"]))
    print(f"paid admission after tombstone: refused ({result['paid_admission_refusal']})")
    print(f"automatic plan: {result['automatic_plan']['plan_digest']} (applied: false)")
    print(f"receipt: {result['receipt']['digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
