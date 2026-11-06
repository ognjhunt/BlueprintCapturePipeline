"""Where the rigid robot-free probe queue keeps a capture's entries, for readers that must not launch.

Split out of ``rigid_robot_free_probe_runner``, which reaches the paid allocator, so withdrawal
deletion can find a withdrawn capture's probe entries without importing it. The runner
re-exports every name here, so its callers are unchanged.

A capture's entry is ``<queue>/pending/<capture key>.json`` until a pass retires it to
``<queue>/done/<capture key>.json``; each probe request's attempts sit under
``<queue>/jobs/<capture key>-<request digest prefix>/``. The capture key is
``team_scene_capture_root.capture_key`` of the capture's resolved root.
"""

from __future__ import annotations

QUEUE_ENV = "BLUEPRINT_RIGID_ROBOT_FREE_PROBE_QUEUE"
DEFAULT_QUEUE = "/var/lib/blueprint/pipeline-control-plane/rigid-robot-free-probes"
ENTRY_DIRECTORIES = ("pending", "done")
JOBS_DIRECTORY = "jobs"

__all__ = ["DEFAULT_QUEUE", "ENTRY_DIRECTORIES", "JOBS_DIRECTORY", "QUEUE_ENV"]
