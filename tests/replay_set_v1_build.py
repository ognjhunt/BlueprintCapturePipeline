"""Build the synthetic records of ``tests/fixtures/replay_set_v1`` from the existing test builders.

Every record here is synthetic: it is made by the same builders the replay tools' own unit tests use
(``tests/test_task_evaluation_downstream_replay.website_job`` for a rigid pick-and-place job,
``tests/test_website_navigation_intake`` for a navigation job). None of it comes from a host, a capture
or a retained job.

The records name absolute paths (a launch receipt names its result by path, and that path is under the
receipt's digest), so each record is built at ``<materialize_root>/<record_id>/host`` and the replay set
runner (``python -m blueprint_pipeline.replay_set``) puts it back at exactly that path before replaying it.

Rebuild (rewrites the records and the manifest; review the diff)::

    PYTHONPATH=src python -m tests.replay_set_v1_build --out tests/fixtures/replay_set_v1
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from blueprint_pipeline.replay_set import (
    CLOCK_FIELD,
    DEFAULT_MATERIALIZE_ROOT,
    GZIP_SUFFIX,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA,
)

DOWNSTREAM = "task_evaluation_downstream_replay"
PROVENANCE = ("synthetic: built by tests/replay_set_v1_build.py from the replay tools' unit-test builders; "
              "no real capture, clip or job record is in it")
# JSON larger than this is committed gzipped (deterministically) and decompressed when the record is put back.
GZIP_ABOVE_BYTES = 32 * 1024


def _rigid(host_parent: Path, *, trace: bool, drop: tuple[str, ...] = ()) -> dict[str, Any]:
    from tests.test_task_evaluation_downstream_replay import website_job

    runtime = None
    if not trace:
        # A scene without a retained probe trace needs the probe run; its compile reads the runtime sources.
        from tests.test_native_task_arena_bundle import _runtime_source_packet

        runtime = _runtime_source_packet(host_parent / "host" / "runtime-sources")
    job = website_job(host_parent, trace=trace, runtime_source=runtime)
    if runtime is not None:
        # The packet's receipt and archive are what the compile reads; the scratch git checkouts are not.
        shutil.rmtree(runtime.parents[1] / "runtime-source-repos", ignore_errors=True)
    for relative in drop:
        (job["host"] / relative).unlink()
    host = job["host"]
    return {"launch_run": job["run"].relative_to(host).as_posix(),
            "intent_dir": job["intent_dir"].relative_to(host).as_posix(),
            "capture_dir": job["capture"].relative_to(host).as_posix(),
            "registry": job["registry"].relative_to(host).as_posix(),
            "mirror": job["mirror"].relative_to(host).as_posix(),
            "webapp_snapshots": job["webapp"].relative_to(host).as_posix()}


def _navigation(host_parent: Path, *, submission_id: str) -> dict[str, Any]:
    """A navigation job: its intent, prepared by the real progression with no launch (kept as filmed)."""
    from tests.test_website_navigation_intake import _advance, _request, _staged

    host = host_parent / "host"
    host.mkdir(parents=True)
    staged, intent = _staged(host, _request(submission_id=submission_id))
    progress = _advance(host, staged, intent)
    if progress.get("status") != "completed":
        raise RuntimeError(f"navigation progression did not complete: {progress.get('status')}")
    shutil.rmtree(host / "factory-output", ignore_errors=True)
    return {"launch_run": "launch-runs/none", "intent_dir": staged["directory"].relative_to(host).as_posix(),
            "empty_dirs": ["launch-runs/none"]}


RECORDS: list[dict[str, Any]] = [
    {"record_id": "rigid-pick-place-1080p-short", "job_kind": "rigid_pick_and_place",
     "input_variety": {"video_resolution": "1080p", "clip_length": "short"},
     "description": "A completed rigid pick-and-place preparation with its robot-free probe trace retained.",
     "build": lambda root: _rigid(root, trace=True),
     "until": "screening_plan",
     "expected_stop": {"kind": "completed", "stage": "screening_plan", "code": "until_screening_plan"}},
    {"record_id": "rigid-pick-place-4k-long", "job_kind": "rigid_pick_and_place",
     "input_variety": {"video_resolution": "4k", "clip_length": "long"},
     "description": "A completed rigid pick-and-place preparation with no probe trace: the probe must run, "
                    "so the chain stops where the probe lane would buy its GPU run.",
     "build": lambda root: _rigid(root, trace=False),
     "until": "screening_plan",
     "expected_stop": {"kind": "would_buy", "stage": "probe_enqueue",
                       "code": "paid_resource_allocator:gpu-canary:rigid-robot-free-probe"}},
    {"record_id": "rigid-pick-place-1080p-long-closure-missing", "job_kind": "rigid_pick_and_place",
     "input_variety": {"video_resolution": "1080p", "clip_length": "long"},
     "description": "A rigid pick-and-place launch whose post-teardown provider-zero receipt was not retained: "
                    "the join names the file to pull.",
     "build": lambda root: _rigid(root, trace=True, drop=(
         "launch-runs/scene-website-qualifying/post_teardown_provider_zero_receipt.json",)),
     "until": "screening_plan",
     "expected_stop": {"kind": "missing_input", "stage": "completion",
                       "code": "{host}/launch-runs/scene-website-qualifying/post_teardown_provider_zero_receipt.json"}},
    {"record_id": "navigation-4k-short", "job_kind": "navigation",
     "input_variety": {"video_resolution": "4k", "clip_length": "short"},
     "description": "A navigation job prepared with its scene kept as filmed (no launch). The post-completion "
                    "chain is the launched preparation's; its join does not adopt this job, by name.",
     "build": lambda root: _navigation(root, submission_id="walkthrough-nav-4k-short"),
     "until": "screening_plan",
     "expected_stop": {"kind": "refusal", "stage": "completion", "code": "preparation_completion_not_adopted"}},
    {"record_id": "navigation-1080p-long", "job_kind": "navigation",
     "input_variety": {"video_resolution": "1080p", "clip_length": "long"},
     "description": "A second navigation job, as above, under its own submission.",
     "build": lambda root: _navigation(root, submission_id="walkthrough-nav-1080p-long"),
     "until": "screening_plan",
     "expected_stop": {"kind": "refusal", "stage": "completion", "code": "preparation_completion_not_adopted"}},
]


def _commit_tree(source: Path, target: Path) -> list[str]:
    """Copy a built host tree into the fixture, gzipping large JSON; return the committed relative paths."""
    written = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        data = path.read_bytes()
        if path.suffix == ".json" and len(data) > GZIP_ABOVE_BYTES:
            relative = relative.with_name(relative.name + GZIP_SUFFIX)
            data = gzip.compress(data, mtime=0)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        destination.chmod(path.stat().st_mode & 0o777)
        written.append(relative.as_posix())
    return written


def build(out: Path, *, materialize_root: Path = Path(DEFAULT_MATERIALIZE_ROOT)) -> dict[str, Any]:
    records_dir = out / "records"
    shutil.rmtree(records_dir, ignore_errors=True)
    entries = []
    for spec in RECORDS:
        record_id = spec["record_id"]
        parent = materialize_root / record_id
        shutil.rmtree(parent, ignore_errors=True)
        parent.mkdir(parents=True)
        inputs: Callable[[Path], dict[str, Any]] = spec["build"]
        built = inputs(parent)
        empty_dirs = built.pop("empty_dirs", [])
        files = _commit_tree(parent / "host", records_dir / record_id)
        shutil.rmtree(parent, ignore_errors=True)
        entries.append({
            "record_id": record_id, "synthetic": True, "provenance": PROVENANCE, "job_kind": spec["job_kind"],
            "description": spec["description"],
            "input_variety": {**spec["input_variety"], "read_by_replayed_code": False},
            "empty_dirs": empty_dirs, "file_count": len(files),
            "tools": [{"tool": DOWNSTREAM, "until": spec["until"], "inputs": built,
                       "expected_stop": spec["expected_stop"]}]})
    # The builders stamp consent and expiry from the wall clock, so the set is replayed at this instant.
    manifest = {"schema_version": MANIFEST_SCHEMA, "set_id": out.name, "synthetic": True,
                "materialize_root": str(materialize_root), CLOCK_FIELD: int(time.time()),
                "built_by": "PYTHONPATH=src python -m tests.replay_set_v1_build --out " + out.as_posix(),
                "records": entries}
    (out / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = build(args.out)
    print(f"built {len(manifest['records'])} synthetic records in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
