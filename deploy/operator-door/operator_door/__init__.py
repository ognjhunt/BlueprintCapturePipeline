"""Operator door: a narrow, token-gated HTTPS window onto the control-plane host.

Cloud agent sessions cannot SSH to the host (their egress is an HTTPS proxy),
so this service gives them the reads a scene run needs (status, run-state
files, journals, unit state) and a spool through which a root oneshot performs
the few privileged operations that are allowed (deploy a pushed commit, start
or pause controller units, replay a failed stage in isolation).

Standard library only: it runs on the host's system Python so that a broken
pipeline release or virtualenv cannot take the door down with it.
"""

VERSION = "1.0.0"
API_PREFIX = "/api/live-pipeline/operator/v1"
