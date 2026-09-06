"""Publish a complete preparation directory atomically; retain source inputs."""
from __future__ import annotations

from functools import wraps
import os
from pathlib import Path
import tempfile

from .task_evaluation_scene_configuration_submission_inputs import read, require


def completed_submission_transaction(builder):
    @wraps(builder)
    def materialize(**kwargs):
        from .task_evaluation_scene_configuration_submission_publication import _validated_inventory, _namespace_lock
        root = Path(kwargs["staging_root"])
        require(root.is_absolute() and not any(path.is_symlink() for path in (root, *root.parents)),
                "completed_submission_root_unsafe")
        root.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
        with _namespace_lock(root.parent / "submission-build-locks", root.name):
            if root.exists():
                manifest, _ = _validated_inventory(root, kwargs["expected_production_commit"])
                require(read(root / "provenance/completed_task_request.v1.json") == kwargs["task"],
                        "completed_submission_task_conflict")
                return {"staging_root": str(root), "input_namespace": manifest["input_namespace"],
                    "request_digest": manifest["request_digest"], "manifest_digest": manifest["manifest_digest"],
                    "status": manifest["status"]}
            with tempfile.TemporaryDirectory(prefix=".completed-submission-", dir=root.parent) as temporary:
                staged = Path(temporary) / "submission"
                result = builder(**{**kwargs, "staging_root": staged})
                _validated_inventory(staged, kwargs["expected_production_commit"])
                os.rename(staged, root)
                descriptor = os.open(root.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                return {**result, "staging_root": str(root)}
    return materialize
