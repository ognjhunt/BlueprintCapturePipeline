#!/usr/bin/env python3
"""Move every control-plane surface to one commit, and prove they all moved.

A deploy here is not one thing. `/opt/blueprint/BlueprintCapturePipeline` is a
mutable clone that runs intake and is the only surface with Git history; the
release symlink is the detached checkout the allocator runs from. Moving one
and not the other is silent -- the website still answers, launches still queue,
and the allocator refuses at the paid boundary on a commit mismatch, several
minutes and one consumed attempt authority later.

The failure that forced this was worse than a stale surface. A release tree
created with `git archive | tar -x` has the right bytes and no `.git`, so
`git rev-parse HEAD` fails inside it and the allocator's orchestrator identity
probe comes back empty. Two unrelated-looking admission blockers
(`gpu_canary_orchestrator_identity_probe_failed` and
`adp_content_agents_config_preflight_binding_invalid`, the latter because the
expected commit compared against was the empty string) had that one cause.

So the check that matters is not "did the command succeed" but "does every
surface now answer `git rev-parse HEAD` with the same commit, and does the
restarted intake process report that commit from its version endpoint". A
surface that cannot answer at all, or a service still bound to an archived
checkout by its environment file, fails here rather than at the paid boundary.

Runs Git and systemd on this host. Contacts no provider, reads no credential,
and rents nothing.
"""

from __future__ import annotations

import argparse
import functools
import selectors
import socket
import http.client
import base64
import contextlib
import dataclasses
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess  # nosec B404 - fixed git/systemctl argv over validated paths
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import sys

_SOURCE_MAX_RESOLVER_BYTES = 64 * 1024
_SOURCE_MAX_RESOLVER_ADDRESSES = 256


_SOURCE_RESOLVER_SCRIPT = """
import json, socket, sys
host, port = json.loads(sys.stdin.buffer.read().decode("utf-8"))
try:
    addresses = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    result = ({"addresses": addresses} if len(addresses) <= 256 else
              {"refusal": "controlled_http_resolution_oversized"})
except socket.gaierror as error:
    result = {"error": [error.errno, error.strerror]}
sys.stdout.buffer.write(json.dumps(result).encode("utf-8"))
"""


def _source_connection_remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("controlled_http_response_timeout")
    return remaining


def _source_resolve_addresses(address, *, deadline):
    """Bound libc resolution without accumulating uncancellable threads.

    Socket timeouts do not cover getaddrinfo. The isolated child receives only
    the hostname and port, and is killed and reaped on timeout or interruption.
    It never receives the URL, headers, credentials, or request body.
    """
    _source_connection_remaining(deadline)
    payload = json.dumps(address).encode("utf-8")
    if len(payload) > 4096:
        raise ValueError("controlled_http_resolution_oversized")
    _source_connection_remaining(deadline)
    process = subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", _SOURCE_RESOLVER_SCRIPT],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        env={},
    )
    try:
        raw = bytearray()
        written = 0
        with selectors.DefaultSelector() as selector:
            os.set_blocking(process.stdin.fileno(), False)
            os.set_blocking(process.stdout.fileno(), False)
            selector.register(process.stdin, selectors.EVENT_WRITE)
            selector.register(process.stdout, selectors.EVENT_READ)
            while not process.stdout.closed:
                events = selector.select(_source_connection_remaining(deadline))
                if not events:
                    raise TimeoutError("controlled_http_response_timeout")
                for key, _event in events:
                    if key.fileobj is process.stdin:
                        written += os.write(process.stdin.fileno(), payload[written:written + 512])
                        if written == len(payload):
                            selector.unregister(process.stdin)
                            process.stdin.close()
                    else:
                        chunk = os.read(process.stdout.fileno(),
                                        min(8192, _SOURCE_MAX_RESOLVER_BYTES + 1 - len(raw)))
                        if not chunk:
                            selector.unregister(process.stdout)
                            process.stdout.close()
                        else:
                            raw.extend(chunk)
                            if len(raw) > _SOURCE_MAX_RESOLVER_BYTES:
                                raise ValueError("controlled_http_resolution_oversized")
                _source_connection_remaining(deadline)
        try:
            process.wait(timeout=_source_connection_remaining(deadline))
        except subprocess.TimeoutExpired:
            raise TimeoutError("controlled_http_response_timeout") from None
        _source_connection_remaining(deadline)
        if process.returncode != 0:
            raise OSError("controlled_http_resolution_failed")
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeDecodeError, RecursionError):
            raise ValueError("controlled_http_resolution_invalid") from None
        _source_connection_remaining(deadline)
        addresses = _source_validated_resolver_result(result)
        _source_connection_remaining(deadline)
        return addresses
    finally:
        if process.poll() is None:
            process.kill()
        process.stdin.close()
        process.stdout.close()
        process.wait()


def _source_validated_resolver_result(result):
    if type(result) is not dict:
        raise ValueError("controlled_http_resolution_invalid")
    if set(result) == {"refusal"} and result["refusal"] == "controlled_http_resolution_oversized":
        raise ValueError("controlled_http_resolution_oversized")
    if set(result) == {"error"}:
        error = result["error"]
        if (type(error) is list and len(error) == 2
                and type(error[0]) is int and type(error[1]) is str):
            raise socket.gaierror(*error)
        raise ValueError("controlled_http_resolution_invalid")
    rows = result.get("addresses")
    if set(result) != {"addresses"} or type(rows) is not list:
        raise ValueError("controlled_http_resolution_invalid")
    if len(rows) > _SOURCE_MAX_RESOLVER_ADDRESSES:
        raise ValueError("controlled_http_resolution_oversized")
    addresses = []
    for row in rows:
        if type(row) is not list or len(row) != 5:
            raise ValueError("controlled_http_resolution_invalid")
        family, kind, protocol, canonical, sockaddr = row
        if (type(family) is not int or family not in {socket.AF_INET, socket.AF_INET6}
                or type(kind) is not int or kind != socket.SOCK_STREAM
                or type(protocol) is not int or protocol not in {0, socket.IPPROTO_TCP}
                or type(canonical) is not str or type(sockaddr) is not list
                or len(sockaddr) != (2 if family == socket.AF_INET else 4)
                or type(sockaddr[0]) is not str or type(sockaddr[1]) is not int
                or not 0 <= sockaddr[1] <= 65535
                or any(type(value) is not int or not 0 <= value <= 0xffffffff
                       for value in sockaddr[2:])):
            raise ValueError("controlled_http_resolution_invalid")
        try:
            socket.inet_pton(family, sockaddr[0])
        except (OSError, ValueError):
            raise ValueError("controlled_http_resolution_invalid") from None
        addresses.append((family, kind, protocol, canonical, tuple(sockaddr)))
    return addresses


def _source_create_deadline_connection(address, _timeout=None, source_address=None, *, deadline):
    addresses = _source_resolve_addresses(address, deadline=deadline)
    sources = (_source_resolve_addresses(source_address, deadline=deadline)
               if source_address else None)
    last_error = None
    for family, kind, protocol, _canonical, sockaddr in addresses:
        _source_connection_remaining(deadline)
        peer = socket.socket(family, kind, protocol)
        try:
            peer.settimeout(_source_connection_remaining(deadline))
            if sources:
                source = next((item[4] for item in sources if item[0] == family), None)
                if source is None:
                    raise OSError("controlled_http_source_address_unavailable")
                peer.bind(source)
            peer.connect(sockaddr)
            _source_connection_remaining(deadline)
            # CONNECT status/headers must share the budget before TLS starts.
            return _SourceDeadlineSocket(peer, deadline, 30)
        except OSError as error:
            last_error = error
            peer.close()
        except BaseException:
            peer.close()
            raise
    _source_connection_remaining(deadline)
    if last_error is not None:
        raise last_error
    raise OSError("getaddrinfo returns an empty list")


class _SourceDeadlineSocket:
    """Clamp every raw receive, including status/header/chunk line refills."""

    def __init__(self, socket, deadline: float, maximum_timeout: float):
        self._socket, self._deadline, self._maximum_timeout = socket, deadline, maximum_timeout

    def __getattr__(self, name):
        return getattr(self._socket, name)

    def _remaining(self):
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("controlled_http_response_timeout")
        return min(remaining,self._maximum_timeout)

    def recv_into(self, target, *args):
        self._socket.settimeout(self._remaining())
        result = self._socket.recv_into(target, *args)
        self._remaining()
        return result

    def sendall(self, data, *args):
        self._socket.settimeout(self._remaining())
        self._socket.sendall(data, *args)
        self._remaining()

    def makefile(self, *args, **kwargs):
        stream = self._socket.makefile(*args, **kwargs)
        raw = getattr(stream, "raw", None)
        if raw is None or getattr(raw, "_sock", None) is not self._socket:
            stream.close()
            raise ValueError("controlled_http_response_reader_invalid")
        raw._sock = self
        return stream

class _SourceDeadlineHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args, deadline, **kwargs):
        self._deadline = deadline
        super().__init__(*args, **kwargs)

    def connect(self):
        # The setup phases share one <=30-second budget, itself clamped to
        # the original installation deadline. Body reads retain global900
        # and the original per-read timeout after the verified TLS handshake.
        if type(self.timeout) not in {int,float} or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError('controlled_http_connection_timeout_invalid')
        maximum_timeout = min(30,self.timeout)
        setup_deadline = min(self._deadline,time.monotonic()+maximum_timeout)
        _source_connection_remaining(setup_deadline)
        self._create_connection = functools.partial(_source_create_deadline_connection,deadline=setup_deadline)
        try:
            http.client.HTTPConnection.connect(self)
            peer = self.sock._socket
            peer.settimeout(_source_connection_remaining(setup_deadline))
            hostname = self._tunnel_host or self.host
            self.sock = self._context.wrap_socket(peer,server_hostname=hostname,do_handshake_on_connect=False)
            self.sock.settimeout(_source_connection_remaining(setup_deadline))
            self.sock.do_handshake()
            _source_connection_remaining(setup_deadline)
            _source_connection_remaining(self._deadline)
            self.sock = _SourceDeadlineSocket(self.sock,self._deadline,maximum_timeout)
        except BaseException:
            self.close()
            raise

class _SourceDeadlineHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, deadline, context):
        super().__init__(context=context)
        self.deadline = deadline

    def https_open(self, request):
        return self.do_open(lambda *args, **kwargs: _SourceDeadlineHTTPSConnection(
            *args, deadline=self.deadline, **kwargs), request, context=self._context)

class _SourceClosingHTTPErrorProcessor(urllib.request.HTTPErrorProcessor):
    def http_response(self, request, response):
        try:
            return super().http_response(request, response)
        except urllib.error.HTTPError as error:
            error.close()
            raise

    https_response = http_response

class _SourceNoRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self,error):
        super().__init__()
        self.error = error

    def redirect_request(self,request,response,code,message,headers,url):
        raise ValueError(self.error)

    def http_error_302(self,request,response,code,message,headers):
        try:
            raise ValueError(self.error)
        finally:
            response.close()

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


def _source_response_opener(deadline,*,error):
    if type(deadline) not in {int,float} or not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise ValueError(error)
    return urllib.request.build_opener(urllib.request.ProxyHandler({}),
        _SourceNoRedirect(error),_SourceDeadlineHTTPSHandler(deadline,None),_SourceClosingHTTPErrorProcessor())


sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(2, str(Path(__file__).resolve().parents[1]))

from stage_task_evaluation_control_plane_release import (  # noqa: E402
    ControlPlaneReleaseError,
    stage_task_evaluation_control_plane_release,
)
from bootstrap_task_evaluation_splat_render_prerequisites import (  # noqa: E402
    validate_splat_render_prerequisites,
)
from blueprint_pipeline.production_blender_runtime import (  # noqa: E402
    ARCHIVE_NAME as BLENDER_ARCHIVE_NAME,
    DEFAULT_ROOT as BLENDER_INSTALL_ROOT,
)
from blueprint_pipeline.task_evaluation_configured_controls_autostart import (  # noqa: E402
    configured_controls_autostart_registry_name,
    validate_configured_controls_autostart_intent,
)
from blueprint_pipeline.control_plane_disk_budget import (  # noqa: E402
    DEFAULT_RESERVATION_ROOT,
    FOOTPRINT_HISTORY_DIRNAME,
    ControlPlaneDiskBudgetError,
    DiskReservation,
    measured_footprint,
    reserve_control_plane_disk,
)
from blueprint_pipeline.control_plane_disk_usage import TreeUsage, tree_usage  # noqa: E402
from blueprint_pipeline.control_plane_storage_pins import (  # noqa: E402
    DEFAULT_PINS_ROOT,
)
from blueprint_pipeline.control_plane_release_leases import (  # noqa: E402
    DEFAULT_PROTECTION_SOURCES,
    ProtectionSources,
    collect_release_protections,
)
from blueprint_pipeline.control_plane_release_retirement import (  # noqa: E402
    DEFAULT_LOCK_TIMEOUT_SECONDS as DEFAULT_RELEASE_LOCK_TIMEOUT_SECONDS,
    EXECUTE_ACK as RELEASE_RETIREMENT_ACK,
    RUNTIME_COMPONENTS as RELEASE_RUNTIME_COMPONENTS,
    apply_release_retirement_plan,
    build_release_retirement_plan,
    delete_retiring_trees,
    live_release_commits as _live_release_commits,
    publisher_lock_roots,
)
from blueprint_pipeline.task_evaluation_release_reference_lock import (  # noqa: E402
    ReleaseReferenceLockError,
    release_reference_lock,
)
from blueprint_pipeline.production_cad_skill_sources import (  # noqa: E402
    DEFAULT_ROOT as DEFAULT_CAD_SKILL_SOURCE_ROOT,
    ProductionCadSkillSourcesError,
    provision_production_cad_skill_sources,
)
from blueprint_pipeline.active_deployed_release_admission import (  # noqa: E402
    trusted_deploy_source,
)
from blueprint_pipeline.control_plane_break_glass import (  # noqa: E402
    DEFAULT_NOTES_ROOT as DEFAULT_BREAK_GLASS_NOTES_ROOT,
    DEPLOY_FROM_UNTRUSTED_SOURCE,
    BreakGlassNoteError,
    mark_reported as mark_break_glass_notes_reported,
    note_summary as break_glass_note_summary,
    refusal_code as break_glass_refusal_code,
    unreported_notes as unreported_break_glass_notes,
    verify_note as verify_break_glass_note,
)

SCHEMA_VERSION = "control_plane_commit_deploy_receipt.v1"
_DEPLOY_ACTIVE_TRANSITION = False
#: How long a break-glass note authorizes a deploy from an untrusted source.
BREAK_GLASS_DEPLOY_NOTE_MAX_AGE_SECONDS = 24 * 3600
UNTRUSTED_SOURCE_REFUSAL = (
    "Merged commit: deploy through the operator door. Unmerged canary or "
    "iteration: use deploy_control_plane_canary.sh or "
    "deploy_control_plane_iteration.sh from the canonical checkout, or a "
    "root-owned, non-group/world-writable clone directly under the config-tools "
    "root. A break-glass note (python -m blueprint_pipeline.control_plane_break_glass "
    "record --action deploy-from-untrusted-source --reason ...) lets this deploy "
    "proceed, but GPU admission will still refuse the release until a trusted "
    "deploy replaces it."
)

#: The single-flight guard a lane holds for the whole life of a paid instance.
#: `vast_provider_adapter` writes it before the launch API call and clears it on
#: teardown, and it records the holding pid and job directory.
DEFAULT_PAID_LAUNCH_LOCKS = (
    "/var/lib/blueprint/pipeline-control-plane/provider-locks/vast_paid_launch.lock",
)
DEFAULT_RESTART_UNITS = ("blueprint-pipeline-intake.service",)
DEFAULT_DOOR_HOLDS_DIR = "/var/lib/blueprint-operator-door/requests/holds"
DEFAULT_DEPLOYED_SYSTEMD_UNITS = (
    "blueprint-pubsub-handoff-listener.service",
    "blueprint-pubsub-handoff-listener.timer",
    "blueprint-agent-execution.service",
    "blueprint-agent-run-dispatcher.service",
    "blueprint-agent-run-dispatcher.timer",
    "blueprint-agent-stage-replay.service",
    "blueprint-agent-stage-replay.timer",
    "blueprint-task-evaluation-launch-dispatcher.service",
    "blueprint-task-evaluation-launch-dispatcher.path",
    "blueprint-task-evaluation-launch-preparation.service",
    "blueprint-task-evaluation-launch-preparation.path",
    "blueprint-task-evaluation-launch-preparation.timer",
    "blueprint-task-evaluation-sam31-preparation-execution.service",
    "blueprint-task-evaluation-sam31-preparation-execution.path",
    "blueprint-task-evaluation-sam31-preparation-execution.timer",
    "blueprint-task-evaluation-episode-compilation.service",
    "blueprint-task-evaluation-episode-compilation.path",
    # Plan 14 §1: backstops the path unit for fallback markers and deferred handed-back rows.
    "blueprint-task-evaluation-episode-compilation.timer",
    # Plan 14 §1: the paid remote-compilation unit, woken by its timer and by hand-offs.
    "blueprint-task-evaluation-episode-compilation-remote.service",
    "blueprint-task-evaluation-episode-compilation-remote.timer",
    "blueprint-task-evaluation-episode-compilation-remote.path",
    "blueprint-task-evaluation-launch-activation.service",
    "blueprint-task-evaluation-launch-activation.path",
    "blueprint-task-evaluation-policy-canary-dispatcher.service",
    "blueprint-task-evaluation-policy-canary-dispatcher.path",
    "blueprint-native-g1-team-campaign-dispatcher.service",
    "blueprint-native-g1-team-campaign-dispatcher.timer",
    "blueprint-native-g1-team-campaign-settlement.service",
    "blueprint-native-g1-team-campaign-settlement.timer",
    "blueprint-scene-object-discovery.service",
    "blueprint-scene-object-discovery.path",
    "blueprint-task-evaluation-configured-controls-progression.service",
    "blueprint-task-evaluation-configured-controls-progression.timer",
    "blueprint-task-evaluation-configured-controls-progression.path",
    "blueprint-task-evaluation-scene-progression.service",
    "blueprint-task-evaluation-scene-progression.timer",
    "blueprint-task-evaluation-launch-supervisor.service",
    "blueprint-task-evaluation-launch-supervisor.timer",
    "blueprint-task-evaluation-launch-reconciler.service",
    "blueprint-task-evaluation-launch-reconciler.timer",
    # Provider billing reconciliation is the accounting-closure feed the
    # capacity/credit guard reads; deploying its timer (not merely enabling it
    # once by hand) keeps the ten-minute reconciliation running across deploys
    # and reboots instead of drifting inactive.
    "blueprint-provider-billing-reconciler.service",
    "blueprint-provider-billing-reconciler.timer",
    "blueprint-task-evaluation-terminal-resource-release.service",
    "blueprint-task-evaluation-terminal-resource-release.path",
    "blueprint-gpu-spend-guard.service",
    "blueprint-gpu-spend-guard.timer",
    "blueprint-completed-replay-cache-gc.service",
    "blueprint-completed-replay-cache-gc.timer",
    "blueprint-scene-project-spend-refresh.service",
    "blueprint-control-plane-storage-gc.service",
    "blueprint-control-plane-storage-gc.timer",
    "blueprint-control-plane-capacity.service",
    "blueprint-control-plane-capacity.timer",
    "blueprint-control-plane-preflight.service",
    "blueprint-control-plane-preflight.timer",
    "blueprint-pipeline-control-plane.service",
    "blueprint-pipeline-intake.service",
)
#: Watchers whose execution surface is provably no-spend and may be armed on a
#: fresh host without widening provider authority.  The paid dispatcher is
#: deliberately absent: its operator freeze must survive every deploy unless
#: ``--arm-path-units`` is explicitly supplied.
DEFAULT_ALWAYS_ARM_PATH_UNITS = (
    "blueprint-task-evaluation-launch-preparation.path",
    "blueprint-task-evaluation-episode-compilation.path",
    "blueprint-task-evaluation-launch-activation.path",
    "blueprint-scene-object-discovery.path",
)
#: Paid execution is still impossible without a consumed, digest-bound
#: activation authority, a clear global spend guard, and the provider-zero
#: preflight inside the dispatcher.  Keeping this watcher active therefore
#: restores Website-to-GPU liveness without granting spend authority by
#: itself.  It is separated from the no-spend watchers so deploy receipts do
#: not blur that distinction.
DEFAULT_ALWAYS_ARM_AUTHORITY_GATED_PATH_UNITS = (
    "blueprint-task-evaluation-sam31-preparation-execution.path",
    "blueprint-task-evaluation-policy-canary-dispatcher.path",
    # Remote episode compilation dispatches only in a remote mode (set, or auto
    # once the remote CPU config is there), with that config, the owner's
    # standing authority and the dispatcher credential; in host mode, which an
    # unset flag without a config is, its ExecCondition skips it except to drain.
    "blueprint-task-evaluation-episode-compilation-remote.path",
)
#: This fixed timer advances only a sealed, qualifying configured-scene plan
#: through the canonical Website APIs.  It cannot be supplied by a request or
#: launch profile and never invokes an allocator directly, but unlike the
#: no-spend queue watchers above it may eventually reach already-authorized
#: downstream spend.  Keep that authority distinct in the deployment receipt.
#: The compilation-result path watcher wakes the same oneshot service the
#: moment a no-spend canary compiles, so it carries the same progression
#: authority rather than the no-spend watcher category.
DEFAULT_ALWAYS_ARM_TIMER_UNITS = (
    "blueprint-task-evaluation-launch-preparation.timer",
    "blueprint-native-g1-team-campaign-dispatcher.timer",
    "blueprint-native-g1-team-campaign-settlement.timer",
    # The service retains its explicit enable flag and capture scope. Installing
    # its timer makes admitted website runs progress after deployment/reboot.
    "blueprint-agent-run-dispatcher.timer",
    "blueprint-agent-stage-replay.timer",
    "blueprint-task-evaluation-scene-progression.timer",
    "blueprint-task-evaluation-sam31-preparation-execution.timer",
    "blueprint-task-evaluation-episode-compilation.timer",
    "blueprint-task-evaluation-episode-compilation-remote.timer",
    "blueprint-task-evaluation-configured-controls-progression.timer",
    "blueprint-task-evaluation-configured-controls-progression.path",
    # The storage reaper is no-spend housekeeping. By default it removes
    # unpinned cache and replay bytes, releases stale pins, offloads sealed
    # evidence and registry residue behind pointers, and retires finished
    # website scene workspaces, each behind its own proofs (docs/CONTROL_PLANE_STORAGE.md).
    "blueprint-control-plane-storage-gc.timer",
    "blueprint-completed-replay-cache-gc.timer",
    "blueprint-control-plane-capacity.timer",
    "blueprint-control-plane-preflight.timer",
)

CONFIGURED_CONTROLS_AUTOMATION_UNITS = (
    "blueprint-task-evaluation-configured-controls-progression.timer",
    "blueprint-task-evaluation-configured-controls-progression.path",
)
#: Where typed release protection is read: live queue envelopes, standing
#: authorizations, retention bindings (with their sidecar leases) and the
#: configuration files that name runtime paths.  Nothing is grepped.
DEFAULT_RELEASE_PROTECTION_SOURCES = DEFAULT_PROTECTION_SOURCES
#: The latest actual retirement summary; the historical deploy-named file is retained.
#: Written under ``<state_root>/release-retention``
#: (0644) so capacity paging can read its alerts without root.
RELEASE_RETIREMENT_SUMMARY_NAME = "latest-deploy-retirement.json"
RELEASE_RETIREMENT_SUMMARY_SCHEMA = "control_plane_release_retirement_summary.v1"
DEFAULT_RELEASE_RETIREMENT_KEEP_LAST = 3
#: The only unit kinds a release may install.  Services and their queue-watching
#: paths stay paired, while the one fixed progression timer (and its
#: compilation-result path watcher) stays paired with its oneshot service.
#: Sockets, mounts, and anything else remain refused.
DEPLOYED_SYSTEMD_UNIT_SUFFIXES = (".service", ".path", ".timer")
DEFAULT_SYSTEMD_DIR = "/etc/systemd/system"
DEFAULT_SCENE_OBJECT_DISCOVERY_QUEUE_ROOT = (
    "/var/lib/blueprint/pipeline-control-plane/scene-object-discoveries"
)
DEFAULT_SCENE_OBJECT_DISCOVERY_RUNTIME_DIRECTORIES = (
    DEFAULT_SCENE_OBJECT_DISCOVERY_QUEUE_ROOT,
    *(f"{DEFAULT_SCENE_OBJECT_DISCOVERY_QUEUE_ROOT}/{name}" for name in (
        "pending",
        "processing",
        "blocked",
        "results",
        "identities",
        "selections",
    )),
    "/var/lib/blueprint/task-evaluation-inputs/scene-object-discoveries",
    "/var/lib/blueprint/task-evaluation-inputs/scene-object-discovery-outputs",
)
DEFAULT_EPISODE_COMPILATION_QUEUE_ROOT = (
    "/var/lib/blueprint/pipeline-control-plane/task-evaluation-episode-compilations"
)
DEFAULT_POLICY_DISPATCH_QUEUE_ROOT = (
    "/var/lib/blueprint/pipeline-control-plane/task-evaluation-policy-canary-dispatches"
)
DEFAULT_REMOTE_CPU_JOBS_ROOT = "/var/lib/blueprint/pipeline-control-plane/remote-cpu-jobs"
DEFAULT_EPISODE_COMPILATION_RUNTIME_DIRECTORIES = (
    DEFAULT_EPISODE_COMPILATION_QUEUE_ROOT,
    *(
        f"{DEFAULT_EPISODE_COMPILATION_QUEUE_ROOT}/{name}"
        for name in ("pending", "processing", "completed", "blocked")
    ),
    # Plan 14 §1: the remote episode-compilation markers, so a host set up
    # before PR 4 has them owned by the service before any path unit watches.
    DEFAULT_REMOTE_CPU_JOBS_ROOT,
    *(f"{DEFAULT_REMOTE_CPU_JOBS_ROOT}/{kind}{stage}" for kind in
      ("handoffs", "shadow", "fallback", "recovery") for stage in ("", "/episode_compilation")),
    # Older root-run GC created stranded/ under umask 0077. The owning
    # dispatcher must be able to discover and resume its sealed deliveries.
    DEFAULT_POLICY_DISPATCH_QUEUE_ROOT,
    *(f"{DEFAULT_POLICY_DISPATCH_QUEUE_ROOT}/{name}" for name in
      ("pending", "processing", "completed", "blocked", "stranded")),
)
DEFAULT_CONFIGURED_CONTROLS_PLAN_ROOT = (
    "/etc/blueprint/task-evaluation-configured-controls-plans"
)
DEFAULT_CONFIGURED_CONTROLS_AUTOSTART_INTENT_ROOT = (
    "/etc/blueprint/task-evaluation-configured-controls-intents"
)
DEFAULT_CONFIGURED_CONTROLS_WEBAPP_SECRET = (
    "/etc/blueprint/provider-secrets/blueprint_task_evaluation_launch_submit_secret"
)
DEFAULT_INTAKE_RUNTIME_DROP_IN = (
    "/etc/systemd/system/blueprint-pipeline-intake.service.d/"
    "90-blueprint-deploy-identity.conf"
)
DEFAULT_INTAKE_VERSION_URL = "http://127.0.0.1:8765/api/live-pipeline/version"
DEFAULT_SCENE_CONFIGURATION_ENVIRONMENT_FILE = (
    "/etc/blueprint/task-evaluation-scene-configuration-release.env"
)
DEFAULT_SCENE_CONFIGURATION_RUNTIME_ROOT = (
    "/var/lib/blueprint/task-evaluation-inputs/system-runtimes"
)
DEFAULT_SPLAT_RENDER_PREREQUISITE_ROOT = (
    "/var/lib/blueprint/task-evaluation-inputs/system-runtime-prerequisites/"
    "splat-render-v1"
)
DEFAULT_ARTIFIXER_SOURCE_ROOT = (
    "/var/lib/blueprint/task-evaluation-inputs/sources/artifixer-a392c4df"
)
DEFAULT_CONTENT_AGENTS_SOURCE_ROOT = (
    "/var/lib/blueprint/task-evaluation-inputs/sources/"
    "usd-content-agents-v0.5.2-36dbf3f2"
)
INTAKE_START_TIMEOUT_SECONDS = 300
DEPLOY_RELEASE_PROVENANCE_NAME = "deploy-release-provenance.json"
SUPERSEDED_ITERATION_PROVENANCE_NAME = (
    "deploy-release-provenance.iteration-superseded.json"
)


class ControlPlaneDeployError(ValueError):
    """A surface did not reach the requested commit, or cannot say that it did."""


class UntrustedDeploySourceError(ControlPlaneDeployError):
    """A typed refusal with a separate remedy, never a host path in the code."""

    def __init__(self, code: str, *, door_clone: bool = False) -> None:
        super().__init__(f"deploy_source_repo_untrusted:{code}")
        self.remedy = UNTRUSTED_SOURCE_REFUSAL + (
            " Repair the door source clone to root:root 0755 before retrying."
            if door_clone else ""
        )


def _provision_scene_configuration_from_release(*, repository_root, source_commit,
        readback_user, **paths) -> dict[str, Any]:
    """Run target-release builders, never modules imported by the old deployer."""
    repository = Path(repository_root).resolve()
    command = [sys.executable,
        str(repository / "scripts/provision_task_evaluation_scene_configuration_release.py"),
        "--repository-root", str(repository), "--source-commit", source_commit,
        "--readback-user", readback_user]
    for key, value in paths.items():
        if value is not None:
            flag = "astra-blender-archive" if key == "astra_blender_archive_path" else key.replace("_", "-")
            command.extend(["--" + flag, str(value)])
    environment = dict(os.environ)
    environment.update(PYTHONPATH=os.pathsep.join([str(repository / "src"), str(repository)]),
                       PYTHONDONTWRITEBYTECODE="1")
    try:
        result = subprocess.run(command, cwd=repository, env=environment,
            check=True, capture_output=True, text=True, timeout=1800)
        value = json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        raise ValueError("scene_configuration_target_release_provision_failed") from exc
    if (not isinstance(value, dict) or value.get("status") != "ready"
            or value.get("source_commit") != source_commit
            or not isinstance(value.get("environment"), dict)):
        raise ValueError("scene_configuration_target_release_provision_receipt_invalid")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _validated_release_provenance(
    path: str | Path, *, source_commit: str
) -> tuple[Path, bytes, dict[str, Any]]:
    """Open the exact live-verified production-promotion receipt."""

    raw_source = Path(path).expanduser()
    try:
        if raw_source.is_symlink():
            raise ControlPlaneDeployError("deploy_release_provenance_invalid")
        source = raw_source.resolve()
        if not source.is_file():
            raise ControlPlaneDeployError("deploy_release_provenance_invalid")
        payload = source.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ControlPlaneDeployError("deploy_release_provenance_invalid") from exc
    if not isinstance(value, Mapping):
        raise ControlPlaneDeployError("deploy_release_provenance_invalid")
    collection = value.get("collection")
    claim_boundary = value.get("claim_boundary")
    if not (
        value.get("schema_version") == "blueprint.deploy_release_provenance.v1"
        and value.get("status") == "verified"
        and value.get("git_sha") == source_commit
        and value.get("workflow_name") == "Full Test Lane"
        and value.get("workflow_path") == ".github/workflows/full-test-lane.yml"
        and value.get("job_name") == "Full pytest lane on CPU runner"
        and type(value.get("run_id")) is int
        and value.get("run_id", 0) > 0
        and isinstance(collection, Mapping)
        and type(collection.get("test_count")) is int
        and collection.get("test_count", 0) > 0
        and isinstance(claim_boundary, Mapping)
        and claim_boundary.get("canonical_full_lane_verified") is True
    ):
        raise ControlPlaneDeployError("deploy_release_provenance_mismatch")
    return source, payload, dict(value)


def _install_release_provenance(
    *, payload: bytes, state_root: Path, source_commit: str, receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Install promotion proof, permitting only documented one-way upgrades.

    A pushed canary may later merge to main at the same commit. Its canonical
    iteration deploy must preserve the earlier canary receipt rather than fail
    on the different provenance label. A development-only receipt may also be
    promoted after the full lane. Preserve either predecessor beside the
    canonical path, then atomically replace it. Every other change conflicts.
    """

    destination = state_root / source_commit / DEPLOY_RELEASE_PROVENANCE_NAME
    superseded_iteration = (
        destination.parent / SUPERSEDED_ITERATION_PROVENANCE_NAME
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ControlPlaneDeployError("deploy_release_provenance_destination_symlink")
    superseded_receipt: dict[str, Any] | None = None
    try:
        if destination.exists():
            if not destination.is_file():
                raise ControlPlaneDeployError("deploy_release_provenance_conflict")
            existing_payload = destination.read_bytes()
            if existing_payload != payload:
                try:
                    existing_receipt = json.loads(existing_payload)
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise ControlPlaneDeployError(
                        "deploy_release_provenance_conflict"
                    ) from exc
                existing_claim = (
                    existing_receipt.get("claim_boundary")
                    if isinstance(existing_receipt, Mapping)
                    else None
                )
                incoming_claim = receipt.get("claim_boundary")
                is_same_commit_development = (
                    isinstance(existing_receipt, Mapping)
                    and existing_receipt.get("schema_version")
                    == "blueprint.deploy_release_provenance.v1"
                    and existing_receipt.get("status") in {"iteration", "canary"}
                    and existing_receipt.get("git_sha") == source_commit
                    and existing_receipt.get("promotion_eligible") is False
                    and isinstance(existing_claim, Mapping)
                    and set(existing_claim) == {
                        "canonical_full_lane_verified", "promotion_eligible", "evidence_grade"
                    }
                    and existing_claim.get("canonical_full_lane_verified") is False
                    and existing_claim.get("promotion_eligible") is False
                    and existing_claim.get("evidence_grade") == "development_only"
                )
                is_canary_to_iteration = (
                    is_same_commit_development
                    and existing_receipt.get("status") == "canary"
                    and receipt.get("schema_version")
                    == "blueprint.deploy_release_provenance.v1"
                    and receipt.get("status") == "iteration"
                    and receipt.get("git_sha") == source_commit
                    and receipt.get("promotion_eligible") is False
                    and incoming_claim == existing_claim
                )
                is_verified_upgrade = (
                    receipt.get("schema_version")
                    == "blueprint.deploy_release_provenance.v1"
                    and receipt.get("status") == "verified"
                    and receipt.get("git_sha") == source_commit
                    and receipt.get("promotion_eligible") is True
                    and isinstance(incoming_claim, Mapping)
                    and incoming_claim.get("canonical_full_lane_verified") is True
                )
                if not (is_canary_to_iteration or (
                    is_same_commit_development and is_verified_upgrade
                )):
                    raise ControlPlaneDeployError(
                        "deploy_release_provenance_conflict"
                    )
                if superseded_iteration.is_symlink():
                    raise ControlPlaneDeployError(
                        "deploy_release_provenance_supersession_conflict"
                    )
                if superseded_iteration.exists():
                    if (
                        not superseded_iteration.is_file()
                        or superseded_iteration.read_bytes() != existing_payload
                    ):
                        raise ControlPlaneDeployError(
                            "deploy_release_provenance_supersession_conflict"
                        )
                else:
                    with superseded_iteration.open("xb") as handle:
                        handle.write(existing_payload)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.chmod(superseded_iteration, 0o440)

                temporary_fd, temporary_name = tempfile.mkstemp(
                    prefix=f".{DEPLOY_RELEASE_PROVENANCE_NAME}.",
                    suffix=".tmp",
                    dir=destination.parent,
                )
                temporary = Path(temporary_name)
                try:
                    with os.fdopen(temporary_fd, "wb") as handle:
                        handle.write(payload)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.chmod(temporary, 0o440)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
                superseded_receipt = {
                    "path": str(superseded_iteration),
                    "sha256": _sha256_bytes(existing_payload),
                    "size_bytes": len(existing_payload),
                    "git_sha": source_commit,
                    "status": existing_receipt["status"],
                    "mode": "0440",
                }
        else:
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.chmod(destination, 0o440)
        reopened = destination.read_bytes()
    except OSError as exc:
        raise ControlPlaneDeployError("deploy_release_provenance_install_failed") from exc
    if reopened != payload:
        raise ControlPlaneDeployError("deploy_release_provenance_readback_mismatch")
    # Inside the installer, not at the call site: promotion proof the reader
    # cannot open is indistinguishable from proof that was never installed, so
    # no caller gets to skip this.
    provenance_access = _install_release_provenance_access(
        destination,
        superseded_iteration if superseded_receipt is not None else None,
    )
    installed = {
        "path": str(destination),
        "sha256": _sha256_bytes(reopened),
        "size_bytes": len(reopened),
        "git_sha": source_commit,
        "run_id": receipt.get("run_id"),
        "run_url": receipt.get("run_url"),
        # Report what the installed receipt actually claims. Hardcoding
        # True told every reader of a deploy receipt that an iteration
        # release had passed the canonical Full Test Lane, while the
        # provenance file it summarised correctly said it had not.
        "canonical_full_lane_verified": bool(
            (receipt.get("claim_boundary") or {}).get(
                "canonical_full_lane_verified"
            )
        ),
        "promotion_eligible": bool(receipt.get("promotion_eligible")),
        "provenance_status": receipt.get("status"),
        "mode": "0440",
        "service_account_access": provenance_access,
    }
    if superseded_receipt is not None:
        installed["superseded_iteration_provenance"] = superseded_receipt
    return installed


def _expanded_slots(lock_paths: Sequence[str]) -> list[Path]:
    """Every concurrency slot, not just slot 0.

    The launch lock became an N-slot semaphore so lanes stop queueing behind
    the slowest run. A deploy that held only the historical filename would be
    exclusive with slot 0 and blind to the rest -- which is worse than the
    check it replaced, because it would look correct while swapping the release
    under two live GPUs.
    """

    from blueprint_pipeline.vast_provider_adapter import vast_launch_lock_paths

    expanded: list[Path] = []
    for raw in lock_paths:
        for slot in vast_launch_lock_paths(Path(raw).expanduser()):
            if slot not in expanded:
                expanded.append(slot)
    return expanded


DEFAULT_SERVICE_ACCOUNT = "blueprint"


def _repair_paid_launch_lock_slots(
    lock_paths: Sequence[str],
    *,
    owner_uid: int,
    owner_gid: int,
    chown: Any = os.chown,
) -> dict[str, Any]:
    """Give every existing lock slot back to the account that launches.

    The launch lock is an N-slot semaphore, and a slot only counts if the
    service account can open it. A paid-lane tool run once as root left
    `slot1`/`slot2` owned `root:root` at 0644 on the live control plane, so the
    authorized N=3 was really N=1 -- invisibly, because the lane holding slot 0
    kept succeeding.

    The runtime guard detects this on every service start but runs as the
    service account and can never repair it. This is the only root-run repo
    code that touches these files on every deploy, so the repair belongs here
    rather than in an operator's remembered `chown`, which a rebuilt host never
    performs. With the guard blocking on an unusable slot, an unrepaired host
    refuses to start intake at all.

    Absent slots are left alone: the guard creates those as the service account
    at 0600, and `open("a+")` never changes an existing file's owner, so a slot
    created correctly once survives every later root-run tool.
    """

    from blueprint_pipeline.vast_provider_adapter import vast_launch_gate_path

    repaired: list[str] = []
    gates = [vast_launch_gate_path(Path(raw).expanduser()) for raw in lock_paths]
    for path in [*_expanded_slots(lock_paths), *gates]:
        if not path.is_file():
            continue
        metadata = path.stat()
        changed = False
        if metadata.st_uid != owner_uid or metadata.st_gid != owner_gid:
            chown(path, owner_uid, owner_gid)
            changed = True
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            path.chmod(0o600)
            changed = True
        if changed:
            repaired.append(str(path))
    return {"repaired_slots": repaired, "owner_uid": owner_uid, "owner_gid": owner_gid}


def _service_account_ids(account: str) -> tuple[int, int] | None:
    """Resolve the service account, or report that this host has none."""

    try:
        import pwd

        entry = pwd.getpwnam(account)
    except (ImportError, KeyError):
        return None
    return entry.pw_uid, entry.pw_gid



def _service_account_read_blocker(
    path: Path, *, owner_uid: int, owner_gid: int
) -> str | None:
    """Report why `owner_uid:owner_gid` cannot read `path`, or None if it can.

    Every ancestor directory is checked for traverse permission, not just the
    file's own read bit. A directory the service account cannot enter hides a
    perfectly readable file beneath it, and that is the shape the live control
    plane actually took: 304 root-owned directories, no `o+x`, holding
    promotion proof the blueprint-run services could not reach.
    """

    def _granted(metadata: os.stat_result, want: int) -> bool:
        mode = stat.S_IMODE(metadata.st_mode)
        if metadata.st_uid == owner_uid:
            bits = (mode >> 6) & 0o7
        elif metadata.st_gid == owner_gid:
            bits = (mode >> 3) & 0o7
        else:
            bits = mode & 0o7
        return bool(bits & want)

    for ancestor in reversed(path.parents):
        try:
            metadata = ancestor.stat()
        except OSError:
            return f"unstatable_directory:{ancestor}"
        if not _granted(metadata, 0o1):
            return f"untraversable_directory:{ancestor}"
    try:
        metadata = path.stat()
    except OSError:
        return f"unstatable_file:{path}"
    if not _granted(metadata, 0o4):
        return f"unreadable_file:{path}"
    return None


def _grant_service_account_read(
    paths: Sequence[Path], *, owner_gid: int, chown: Any = os.chown
) -> list[str]:
    """Give the service account group-read on what deploy just wrote.

    Only the group is moved; the owning uid is left alone deliberately. The
    provenance receipt is installed 0440 precisely so that nothing can rewrite
    it, and a root-owned file the service account merely reads keeps that
    property. Chowning it to the service account would hand the reader the
    power to chmod its own promotion proof.
    """

    adjusted: list[str] = []
    for path in paths:
        try:
            metadata = path.stat()
        except OSError:
            continue
        changed = False
        if metadata.st_gid != owner_gid:
            chown(path, -1, owner_gid)
            changed = True
            metadata = path.stat()
        wanted = 0o050 if path.is_dir() else 0o040
        mode = stat.S_IMODE(metadata.st_mode)
        if mode & wanted != wanted:
            path.chmod(mode | wanted)
            changed = True
        if changed:
            adjusted.append(str(path))
    return adjusted


def _install_release_provenance_access(
    destination: Path,
    superseded: Path | None,
    *,
    account: str = DEFAULT_SERVICE_ACCOUNT,
    chown: Any = os.chown,
) -> dict[str, Any]:
    """Make the promotion proof readable by the account that consumes it, then prove it.

    Deploy runs as root; every service that reads this file runs as
    `blueprint`. `os.chmod(..., 0o440)` alone therefore installed `root:root`
    promotion proof that the reader could not open, so a host could pass its
    whole deploy and still have every launch fall back to `development_only`
    -- with no error anywhere, because nothing asserted the reader's side.

    The grant is not trusted on its own. The gate below re-derives readability
    from the installed inode and fails the deploy if the service account still
    cannot reach it, so this can never silently regress to a no-op.
    """

    account_ids = _service_account_ids(account)
    if account_ids is None:
        return {
            "status": "not_applicable_no_service_account",
            "account": account,
            "adjusted_paths": [],
        }
    owner_uid, owner_gid = account_ids
    targets: list[Path] = [destination.parent, destination]
    if superseded is not None:
        targets.append(superseded)
    adjusted = _grant_service_account_read(
        targets, owner_gid=owner_gid, chown=chown
    )
    for path in targets:
        blocker = _service_account_read_blocker(
            path, owner_uid=owner_uid, owner_gid=owner_gid
        )
        if blocker is not None:
            raise ControlPlaneDeployError(
                "deploy_release_provenance_unreadable_by_service_account:"
                f"{account}:{blocker}"
            )
    return {
        "status": "readable",
        "account": account,
        "owner_uid": owner_uid,
        "owner_gid": owner_gid,
        "adjusted_paths": adjusted,
        "verified_paths": [str(path) for path in targets],
    }


def _install_disk_reservation_runtime_prerequisites(
    reservation_root: str | Path,
    *,
    account: str = DEFAULT_SERVICE_ACCOUNT,
    root_uid: int = 0,
    chown: Any = os.chown,
    stat_reader: Any = lambda path: path.stat(),
) -> dict[str, Any]:
    """Install the shared disk ledger so root and runtime workers can use it.

    Deploy reserves disk as root, while every queue worker reserves disk as the
    ``blueprint`` service account. Merely creating the ledger with mode 2770 is
    insufficient: a root-created directory and lock retain ``root:root``
    ownership, leaving the runtime unable to enter the directory or open the
    lock after an otherwise successful deploy.

    Reconcile these inodes before any service restart and verify their installed
    ownership and modes instead of trusting the privileged mutations.  The
    footprint ``history`` directory gets the same treatment: root appends the
    deploy's samples and the runtime account appends every worker's.
    """

    account_ids = _service_account_ids(account)
    if account_ids is None:
        raise ControlPlaneDeployError(
            f"deploy_disk_reservation_account_missing:{account}"
        )
    _owner_uid, owner_gid = account_ids
    root = Path(reservation_root).expanduser()
    lock = root / ".lock"
    history = root / FOOTPRINT_HISTORY_DIRNAME
    if not root.is_absolute():
        raise ControlPlaneDeployError(
            "deploy_disk_reservation_directory_not_absolute"
        )
    # A refusal is a typed code: it names the ledger item (directory, lock or
    # history), never the host path it lives at.
    items = (("directory", root, 0o2770), ("lock", lock, 0o660), ("history", history, 0o2770))
    for item, path, _mode in items:
        if path.is_symlink():
            raise ControlPlaneDeployError(
                f"deploy_disk_reservation_runtime_symlink:{item}"
            )

    repaired: list[str] = []
    item = "directory"
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o2770)
        if root.is_symlink() or not root.is_dir():
            raise ControlPlaneDeployError(
                "deploy_disk_reservation_directory_invalid"
            )
        item = "lock"
        lock.touch(mode=0o660, exist_ok=True)
        if lock.is_symlink() or not lock.is_file():
            raise ControlPlaneDeployError(
                "deploy_disk_reservation_lock_invalid"
            )
        item = "history"
        history.mkdir(exist_ok=True, mode=0o2770)
        if history.is_symlink() or not history.is_dir():
            raise ControlPlaneDeployError(
                "deploy_disk_reservation_history_directory_invalid"
            )
        for item, path, wanted_mode in items:
            metadata = stat_reader(path)
            changed = False
            if metadata.st_uid != root_uid or metadata.st_gid != owner_gid:
                chown(path, root_uid, owner_gid)
                changed = True
                metadata = stat_reader(path)
            if stat.S_IMODE(metadata.st_mode) != wanted_mode:
                path.chmod(wanted_mode)
                changed = True
            if changed:
                repaired.append(str(path))
    except ControlPlaneDeployError:
        raise
    except OSError as exc:
        raise ControlPlaneDeployError(
            f"deploy_disk_reservation_runtime_install_failed:{item}"
        ) from exc

    installed: list[dict[str, Any]] = []
    for item, path, wanted_mode in items:
        metadata = stat_reader(path)
        if (
            metadata.st_uid != root_uid
            or metadata.st_gid != owner_gid
            or stat.S_IMODE(metadata.st_mode) != wanted_mode
        ):
            raise ControlPlaneDeployError(
                f"deploy_disk_reservation_runtime_readback_mismatch:{item}"
            )
        kind = "history_directory" if item == "history" else item
        installed.append(
            {
                "kind": kind,
                "path": str(path),
                "owner": "root",
                "group": account,
                "owner_uid": root_uid,
                "owner_gid": owner_gid,
                "mode": f"{wanted_mode:04o}",
            }
        )
    return {
        "status": "ready",
        "account": account,
        "repaired_paths": repaired,
        "installed": installed,
    }


def _install_release_lease_root(lease_root: str | Path) -> dict[str, Any]:
    """Create the directory that holds the sidecar leases of immutable bindings.

    Only the root deploy writes sidecars, so the tree is root:root 0750.  A
    symlink or a file where the tree belongs is refused rather than repaired.
    """

    root = Path(lease_root)
    installed = os.geteuid() == 0
    for path in (root, root / "bindings"):
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ControlPlaneDeployError("deploy_release_lease_root_unsafe")
        path.mkdir(mode=0o750, exist_ok=True)
        if installed:
            os.chown(path, 0, 0)
        path.chmod(0o750)
    return {"status": "ready", "path": str(root / "bindings"), "mode": "0750",
            "owner": "root" if installed else "deploying_user"}


class _ProtectionRootCreated(Exception):
    """Retirement created a missing protection root and must stop this time."""

    def __init__(self, codes: list[str], roots: list[str]) -> None:
        super().__init__(",".join(codes))
        self.codes = codes
        self.roots = roots


def _install_release_protection_roots(
    sources: ProtectionSources, *, account: str = DEFAULT_SERVICE_ACCOUNT
) -> list[str]:
    """Create, empty, the authorization and binding roots a host lacks.

    Retirement treats a missing root as a source it cannot read.  On a fresh
    host they may simply not exist yet (the host installer creates the
    standing-authorization directory, but nothing creates the binding root
    until a binding is published), so deploy creates them with the service
    account's ownership (0750).  An existing root is never touched, and a
    symlink or file in its place refuses.  A deploy that had to create one
    retires nothing: an empty root proves nothing about what used to be in it.
    """

    created: list[str] = []
    for root in (Path(sources.standing_authorization_dir), Path(sources.binding_root)):
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise ControlPlaneDeployError("deploy_release_protection_root_unsafe")
        if root.exists():
            continue
        root.mkdir(mode=0o750)
        ids = _service_account_ids(account) if os.geteuid() == 0 else None
        if ids is not None:
            os.chown(root, ids[0], ids[1])
        root.chmod(0o750)
        created.append(str(root))
    return created


def _write_release_retirement_summary(
    path: Path, result: Mapping[str, Any], *, source_commit: str, generated_at: float
) -> dict[str, Any]:
    """Replace the latest retirement summary atomically; never fail the deploy."""

    summary = {
        "schema_version": RELEASE_RETIREMENT_SUMMARY_SCHEMA,
        "generated_at_epoch": generated_at,
        "source_commit": source_commit,
        **{key: value for key, value in result.items() if key != "summary"},
    }
    payload = (json.dumps(summary, indent=1, sort_keys=True) + "\n").encode("utf-8")
    directory = path.parent
    try:
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise OSError("release retention directory unsafe")
        directory.mkdir(mode=0o750, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=directory
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fchmod(stream.fileno(), 0o644)
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise
    except OSError as exc:
        return {"status": "write_failed", "path": str(path), "error": type(exc).__name__}
    return {"status": "written", "path": str(path), "mode": "0644"}


def _retirement_protection_summary(
    plan: Mapping[str, Any], protections: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "protected_commit_count": len(plan["protected_commits"]),
        "protected_by_kind": {
            kind: len(commits) for kind, commits in plan["protected_by_kind"].items()
        },
        "protected_tree_count": plan["protected_tree_count"],
        "lease_protected_tree_count": plan["lease_protected_tree_count"],
        "lapsed_count": plan["lapsed_count"],
        "migrated_binding_count": len(protections["migrated"]),
        "renewed_lease_count": len(protections["renewed"]),
        "warning_count": plan["warning_count"],
        "alerts": list(plan["alerts"]),
    }


def _retire_superseded_release_trees(
    *,
    release_root: str | Path,
    runtime_root: str | Path,
    active_link: str | Path,
    current_commit: str,
    protection_sources: ProtectionSources,
    keep_last: int,
    state_root: str | Path | None = None,
    extra_config_files: Sequence[str | Path] = (),
    now: Callable[[], float] = time.time,
    proc_root: str | Path = "/proc",
    summary_path: str | Path | None = None,
    lock_timeout_seconds: float = DEFAULT_RELEASE_LOCK_TIMEOUT_SECONDS,
    source_repo: str | Path | None = None,
    defer_deletion: bool = False,
) -> dict[str, Any]:
    """Explicitly retire superseded release and runtime trees.

    This compatibility helper requires separate retirement authority; ordinary
    deployment never calls it. Every release-reference publisher takes the reference
    lock shared on its root: queue writers, the launch-profile publisher, the
    standing-authorization materializer, release activation and the SAM
    prefix binding writer.  Retirement holds each of those roots exclusively
    only while it collects typed protection, plans, re-checks live processes
    and renames each candidate into ``<root>/.retiring``, so no reference can
    appear in between; walking and deleting the moved trees happens after
    the locks are released (``_finish_release_retirement``; with
    ``defer_deletion`` the caller runs it later, outside its own locks too),
    and leftovers of an interrupted retirement are swept before they are
    taken.  Anything the plan cannot prove safe is left in place and
    reported; a retirement failure never fails a deploy whose surfaces
    already moved.
    """

    roots: list[Path] = []
    staged_roots = _release_retiring_roots(release_root, runtime_root)
    renamed: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    swept: dict[str, Any] = {"deleted": [], "failed": []}

    def in_use() -> list[str]:
        return _live_release_commits(release_root, runtime_root=runtime_root, proc_root=proc_root)

    try:
        sources = protection_sources
        extra = [Path(path) for path in extra_config_files if Path(path) not in sources.config_files]
        if extra:
            # The bootstrap this deploy installed may name a non-default
            # machinery file; retirement must read the same configuration.
            sources = dataclasses.replace(sources, config_files=(*sources.config_files, *extra))
        # A previous retirement that stopped after renaming left trees that no
        # reference can reach any more; they go before any lock is taken.
        swept = _sweep_retiring_trees(staged_roots)
        # The same directories publishers lock shared, plus this deploy's own
        # state root (release activation locks it), each exactly once.
        roots.extend(
            publisher_lock_roots(sources, *([] if state_root is None else [state_root]))
        )
        with contextlib.ExitStack() as held:
            for root in roots:
                held.enter_context(
                    release_reference_lock(
                        root, exclusive=True, timeout_seconds=lock_timeout_seconds
                    )
                )
            created_roots = _install_release_protection_roots(sources)
            if created_roots:
                created = [
                    f"release_protection_root_created:{Path(path).name}" for path in created_roots
                ]
                raise _ProtectionRootCreated(created, created_roots)
            _install_release_lease_root(sources.lease_root)
            protections = collect_release_protections(sources, now=float(now()), migrate=True)
            plan = build_release_retirement_plan(
                release_root=release_root,
                runtime_root=runtime_root,
                active_link=active_link,
                current_commit=current_commit,
                protections=protections,
                keep_last=keep_last,
                now=now,
                in_use_commits=in_use(),
                measure_sizes=False,
            )
            if plan["status"] != "dry_run":
                result: dict[str, Any] = {
                    "status": "skipped",
                    "blockers": list(plan["blockers"]),
                    "plan_digest": plan["plan_digest"],
                    "created_protection_roots": created_roots,
                    **_retirement_protection_summary(plan, protections),
                }
            else:
                receipt = apply_release_retirement_plan(
                    plan,
                    ack=RELEASE_RETIREMENT_ACK,
                    active_link=active_link,
                    release_root=release_root,
                    in_use_now=lambda: set(in_use()),
                )
                renamed = list(receipt["renamed"])
                fallback = list(receipt["direct_delete_fallback"])
                result = {
                    "status": "applied",
                    "plan_digest": plan["plan_digest"],
                    "receipt_digest": receipt["result_digest"],
                    "unmanaged_children": list(plan["unmanaged_children"]),
                    "skipped": list(receipt["skipped"]),
                    "created_protection_roots": created_roots,
                    **_retirement_protection_summary(plan, protections),
                }
    except _ProtectionRootCreated as created:
        # The roots now exist for the next deploy, which retires normally.
        result = {
            "status": "skipped",
            "reason": "protection_root_created",
            "blockers": list(created.codes),
            "created_protection_roots": list(created.roots),
            "alerts": list(created.codes),
        }
    except (ReleaseReferenceLockError, ControlPlaneDeployError) as exc:
        # Both carry typed codes, never host paths.
        result = {"status": "blocked", "blockers": [str(exc)]}
    except Exception as exc:  # the surfaces already moved; report, never fail the deploy
        partial = getattr(exc, "release_retirement_receipt", None)
        if isinstance(partial, Mapping):
            renamed = list(partial.get("renamed") or [])
            fallback = list(partial.get("direct_delete_fallback") or [])
        result = {
            "status": "blocked",
            "blockers": [f"deploy_release_retirement_failed:{type(exc).__name__}"],
        }
    result["renamed"] = renamed
    result["direct_delete_fallback"] = fallback
    result["retired_commits"] = sorted({str(row["commit"]) for row in (*renamed, *fallback)})
    result["swept"] = swept["deleted"]
    result["deletion_failures"] = list(swept["failed"])
    if result["status"] == "blocked":
        result["alerts"] = [f"release_retirement_blocked:{result['blockers'][0]}"]
    result["lock_roots"] = [str(root) for root in roots]
    if defer_deletion:
        return result
    return _finish_release_retirement(
        result,
        retiring_roots=staged_roots,
        source_repo=source_repo,
        current_commit=current_commit,
        summary_path=summary_path,
        now=now,
    )


def _release_retiring_roots(release_root: str | Path, runtime_root: str | Path) -> list[Path]:
    """Every managed root whose ``.retiring`` directory holds trees moved aside."""

    return [
        Path(release_root),
        *(Path(runtime_root) / component for component in RELEASE_RUNTIME_COMPONENTS),
    ]


def _sweep_retiring_trees(roots: Sequence[Path]) -> dict[str, Any]:
    """Delete what earlier retirements moved aside; never raises."""

    try:
        return delete_retiring_trees(roots)
    except Exception as exc:
        return {
            "deleted": [],
            "deleted_bytes": 0,
            "shared_bytes": 0,
            "failed": [{"path": "", "reason": f"deletion_failed:{type(exc).__name__}"}],
        }


def _prune_release_worktrees(source_repo: str | Path) -> dict[str, Any]:
    """Make Git forget the release worktrees retirement deleted.  Never raises.

    Release trees are worktrees of the source clone.  Retirement renames and
    deletes them without ``git worktree remove``, so their registrations stay
    behind and ``git worktree add`` later refuses the same path as "missing
    but already registered", which would break a rollback to a retired
    commit.  ``git worktree prune`` drops only registrations whose
    directories are gone.
    """

    checkout = Path(source_repo).resolve()
    argv = ["git", "-c", f"safe.directory={checkout}", "-C", str(checkout), "worktree", "prune"]
    try:
        completed = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
            argv, capture_output=True, text=True, check=False, timeout=120
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "failed", "error": type(exc).__name__}
    if completed.returncode != 0:
        return {"status": "failed", "returncode": completed.returncode}
    return {"status": "pruned"}


def _finish_release_retirement(
    result: dict[str, Any],
    *,
    retiring_roots: Sequence[Path],
    source_repo: str | Path | None,
    current_commit: str,
    summary_path: str | Path | None,
    now: Callable[[], float] = time.time,
    startup_sweep: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Outside every lock: delete and measure what retirement moved aside, then report.

    Runs after the publishers' locks and, in a deploy, after the paid-launch
    gate and disk reservation are released, so no launch or publisher waits
    while about ninety trees are deleted.  Then prunes the source clone's
    registrations of the deleted worktrees, adds the deletion, the prune and
    the alerts to the receipt, and writes the retirement summary.  Never
    raises.
    """

    deletion = _sweep_retiring_trees(retiring_roots)
    fallback = list(result.get("direct_delete_fallback") or [])
    result["deleted"] = deletion["deleted"]
    result["retired_bytes"] = deletion["deleted_bytes"] + sum(row["bytes"] for row in fallback)
    result["shared_bytes"] = deletion.get("shared_bytes", 0) + sum(
        row["shared_bytes"] for row in fallback
    )
    failures = [*result.get("deletion_failures", []), *deletion["failed"]]
    if startup_sweep is not None:
        result["startup_swept"] = list(startup_sweep["deleted"])
        failures = [*startup_sweep["failed"], *failures]
    result["deletion_failures"] = failures
    result["worktree_prune"] = (
        _prune_release_worktrees(source_repo)
        if source_repo is not None
        else {"status": "not_requested"}
    )
    alerts = list(result.get("alerts") or [])
    rename_failures = sum(
        1
        for row in result.get("skipped") or []
        if str(row.get("reason", "")).startswith(("rename_failed:", "direct_delete_failed:"))
    )
    if rename_failures:
        alerts.append(f"release_retirement_rename_failed:{rename_failures}")
    if failures:
        alerts.append(f"release_retirement_deletion_failed:{len(failures)}")
    result["alerts"] = alerts
    if summary_path is not None:
        result["summary"] = _write_release_retirement_summary(
            Path(summary_path), result, source_commit=current_commit, generated_at=float(now())
        )
    return result


_SANDBOX_DIRECTIVES = ("ReadWritePaths=", "ReadOnlyPaths=")
_UNIT_PROVISIONABLE_PREFIX = "/var/lib/blueprint/"
_UNIT_FILE_SUFFIXES = (".json", ".jsonl", ".lock", ".env", ".sqlite", ".log", ".txt")


def _unit_sandbox_entries(unit_text: str) -> list[tuple[str, bool, str]]:
    """Every ``(path, optional, directive)`` a unit's filesystem sandbox names."""

    entries: list[tuple[str, bool, str]] = []
    for raw_line in unit_text.splitlines():
        line = raw_line.strip()
        for directive in _SANDBOX_DIRECTIVES:
            if not line.startswith(directive):
                continue
            for token in line[len(directive):].split():
                optional = token.startswith("-")
                path_text = token[1:] if optional else token
                if not path_text.startswith("/"):
                    continue
                entries.append((path_text.rstrip("/") or "/", optional, directive[:-1]))
    return entries


def _install_retention_plan_reader_access(root: Path, *, owner_gid: int) -> dict[str, Any]:
    """Repair only directory traversal for retained deployment plans, not file authority."""
    if not root.exists():
        return {"status": "absent", "path": str(root)}
    if not root.is_absolute() or any(p.is_symlink() for p in (root, *root.parents)) or not root.is_dir():
        raise ControlPlaneDeployError("deploy_retention_reader_root_unsafe")
    metadata = root.stat()
    if metadata.st_gid != owner_gid:
        os.chown(root, -1, owner_gid)
    root.chmod(0o750)
    observed = root.stat()
    if observed.st_uid != metadata.st_uid or observed.st_gid != owner_gid or stat.S_IMODE(observed.st_mode) != 0o750:
        raise ControlPlaneDeployError("deploy_retention_reader_readback_failed")
    return {"status": "readable_directory", "path": str(root), "owner_uid": observed.st_uid,
            "reader_gid": observed.st_gid, "mode": "0750", "file_permissions_changed": False}


def _install_scene_retirement_stores(*, root_prefix: str | Path | None = None) -> dict[str, Any]:
    """Provision only the disabled stores already declared by first installation.

    Existing authority/data is verified, never reowned or repaired. No policy,
    consent, generation, cleanup flag or new privileged operation is issued.
    """
    from blueprint_pipeline.task_evaluation_scene_retirement_access import (
        _close_owned, _identity, _open_owned, _opened,
    )

    ids = _service_account_ids(DEFAULT_SERVICE_ACCOUNT)
    if ids is None:
        raise ControlPlaneDeployError("deploy_scene_retirement_store_account_missing")
    service_uid, service_gid = ids
    root = Path('/var/lib/blueprint/scene-retirement')
    if root_prefix is not None:
        root = Path(root_prefix) / root.relative_to('/')
    rows = [(root / relative if relative else root, mode,
             service_uid if service else _SCENE_RUNTIME_OWNER, service_gid)
            for relative, mode, service in (
                ('', 0o755, False), ('coordinator', 0o755, False),
                ('generations', 0o700, True), ('journals', 0o700, False),
                ('journals/processes', 0o700, False), ('journals/retired', 0o700, False),
                ('journals.metadata', 0o750, False), ('consents', 0o700, False))]
    error = 'deploy_scene_retirement_store_unsafe'
    def require(value):
        if not value:
            raise ControlPlaneDeployError(error)
    def verify(path, mode, uid, gid):
        with _opened(path, directory=True) as (_, info):
            require((info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) == (uid, gid, mode))
    try:
        # Complete preflight precedes the first write. The fixed parent must
        # already be protected; this step never creates arbitrary ancestry.
        with _opened(root.parent, directory=True, protected=True):
            pass
        existing = set()
        for path, mode, uid, gid in rows:
            try:
                verify(path, mode, uid, gid)
            except FileNotFoundError:
                continue
            existing.add(path)
        created = []
        for path, mode, uid, gid in rows:
            if path in existing:
                verify(path, mode, uid, gid)
                continue
            with _opened(path.parent, directory=True, protected=True) as (parent, parent_info):
                expected = _identity(parent_info)
                require(_identity(os.fstat(parent)) == expected)
                os.mkdir(path.name, mode, dir_fd=parent)
                # Only the directory created by this invocation can be chowned.
                before = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
                require(stat.S_ISDIR(before.st_mode) and before.st_uid == os.geteuid())
                fd, info = _open_owned(path.name, os.O_RDONLY | os.O_DIRECTORY, dir_fd=parent)
                owned = _identity(info)
                try:
                    require(owned == _identity(before))
                    require(_identity(os.fstat(parent)) == expected)
                    os.fchown(fd, uid, gid)
                    require(_identity(os.fstat(fd)) == owned
                            and _identity(os.stat(path.name, dir_fd=parent, follow_symlinks=False)) == owned)
                    os.fchmod(fd, mode)
                    named = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
                    require(_identity(named)[:2] == owned[:2]
                            and stat.S_IMODE(named.st_mode) == mode)
                    owned = _identity(named)
                    info = os.fstat(fd)
                    require(_identity(info) == owned
                            and (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) == (uid, gid, mode))
                    require(_identity(os.stat(path.name, dir_fd=parent, follow_symlinks=False)) == owned
                            and _identity(os.fstat(parent)) == expected)
                    os.fsync(fd)
                    require(_identity(os.fstat(parent)) == expected)
                    os.fsync(parent)
                finally:
                    require(_close_owned(fd, owned) is None)
            verify(path, mode, uid, gid)
            created.append({'path': str(path), 'owner_uid': uid, 'owner_gid': gid, 'mode': f'{mode:04o}'})
        return {'status': 'ready', 'created': created, 'created_count': len(created),
                'verified_count': len(rows), 'authority_issued': False, 'cleanup_enabled': False}
    except (OSError, ValueError) as exc:
        raise ControlPlaneDeployError(error) from exc


def _install_unit_sandbox_paths(
    *,
    release_path: str | Path,
    units: Sequence[str] = DEFAULT_DEPLOYED_SYSTEMD_UNITS,
    root_prefix: str | Path | None = None,
    account: str = DEFAULT_SERVICE_ACCOUNT,
    owner_ids: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Make every path a deployed unit's sandbox names exist before the release moves.

    ``ProtectSystem=strict`` units fail to start when a ``ReadWritePaths`` or
    ``ReadOnlyPaths`` entry does not exist.  Twice (the disk-reservation ledger,
    then the storage-pin ledger) a unit gained a path that no deploy step
    created, and the dead worker was discovered only when a Website run
    stalled; each time the fix was one more hand-written installer.  This step
    reads the staged release's own unit files, creates any missing
    service-owned directory under ``/var/lib/blueprint`` (never repairing one
    that exists), and refuses the deploy when a path it may not create -- host
    configuration under ``/etc``, a file, another tree -- is absent and not
    marked optional with a leading ``-``.
    """

    release = Path(release_path).expanduser()
    retention_path = Path("/var/lib/blueprint/pipeline-control-plane/release-retention")
    if root_prefix is not None:
        retention_path = Path(root_prefix) / retention_path.relative_to("/")
    if retention_path.exists():
        reader_ids = owner_ids or _service_account_ids(account)
        if reader_ids is None:
            raise ControlPlaneDeployError("deploy_retention_reader_account_missing")
        _install_retention_plan_reader_access(retention_path, owner_gid=reader_ids[1])
    created: list[dict[str, Any]] = []
    verified: list[dict[str, str]] = []
    blockers: list[str] = []
    pending: list[tuple[str, str, Path]] = []
    seen: set[str] = set()
    for unit in units:
        source = release / "deploy" / "systemd" / unit
        if source.is_symlink() or not source.is_file():
            # The unit installer refuses an absent release unit on its own.
            continue
        for path_text, optional, directive in _unit_sandbox_entries(
            source.read_text(encoding="utf-8")
        ):
            if path_text in seen:
                continue
            seen.add(path_text)
            host_path = (
                Path(path_text)
                if root_prefix is None
                else Path(root_prefix).expanduser() / path_text.lstrip("/")
            )
            if host_path.exists():
                verified.append({"unit": unit, "path": path_text, "directive": directive})
                continue
            if optional:
                continue
            file_like = Path(path_text).suffix in _UNIT_FILE_SUFFIXES
            if path_text.startswith(_UNIT_PROVISIONABLE_PREFIX) and not file_like:
                pending.append((unit, path_text, host_path))
                continue
            blockers.append(f"deploy_unit_sandbox_path_missing:{unit}:{path_text}")
    if blockers:
        raise ControlPlaneDeployError(",".join(sorted(blockers)))
    if pending:
        ids = owner_ids or _service_account_ids(account)
        if ids is None:
            raise ControlPlaneDeployError(f"deploy_unit_sandbox_account_missing:{account}")
        owner_uid, owner_gid = ids
        for unit, path_text, host_path in pending:
            try:
                host_path.mkdir(parents=True, exist_ok=True, mode=0o750)
                if host_path.is_symlink() or not host_path.is_dir():
                    raise ControlPlaneDeployError(
                        f"deploy_unit_sandbox_path_invalid:{unit}:{path_text}"
                    )
                os.chown(host_path, owner_uid, owner_gid)
                host_path.chmod(0o750)
            except OSError as exc:
                raise ControlPlaneDeployError(
                    f"deploy_unit_sandbox_path_install_failed:{unit}:{path_text}"
                ) from exc
            created.append(
                {
                    "unit": unit,
                    "path": path_text,
                    "mode": "0750",
                    "owner_uid": owner_uid,
                    "owner_gid": owner_gid,
                }
            )
    return {
        "status": "ready",
        "unit_count": len(units),
        "verified_count": len(verified),
        "created_count": len(created),
        "created": created,
    }


def _install_scene_object_discovery_runtime_directories(
    *,
    directories: Sequence[str] = DEFAULT_SCENE_OBJECT_DISCOVERY_RUNTIME_DIRECTORIES,
    account: str = DEFAULT_SERVICE_ACCOUNT,
) -> list[dict[str, Any]]:
    """Install the no-spend discovery queue and materialization roots.

    Exact-release deployment must be sufficient on an already-provisioned host;
    requiring an operator to remember to rerun the broad bootstrap installer
    leaves the newly installed path unit watching an absent directory.
    """

    account_ids = _service_account_ids(account)
    if account_ids is None:
        raise ControlPlaneDeployError(
            f"deploy_scene_object_discovery_account_missing:{account}"
        )
    owner_uid, owner_gid = account_ids
    receipts: list[dict[str, Any]] = []
    for raw_path in directories:
        path = Path(raw_path)
        if not path.is_absolute():
            raise ControlPlaneDeployError(
                "deploy_scene_object_discovery_directory_not_absolute"
            )
        if path.is_symlink():
            raise ControlPlaneDeployError(
                f"deploy_scene_object_discovery_directory_symlink:{path}"
            )
        try:
            path.mkdir(parents=True, exist_ok=True, mode=0o750)
            if path.is_symlink() or not path.is_dir():
                raise ControlPlaneDeployError(
                    f"deploy_scene_object_discovery_directory_invalid:{path}"
                )
            os.chown(path, owner_uid, owner_gid)
            path.chmod(0o750)
            stat_result = path.stat()
        except OSError as exc:
            raise ControlPlaneDeployError(
                f"deploy_scene_object_discovery_directory_install_failed:{path}"
            ) from exc
        if (
            stat_result.st_uid != owner_uid
            or stat_result.st_gid != owner_gid
            or stat.S_IMODE(stat_result.st_mode) != 0o750
        ):
            raise ControlPlaneDeployError(
                f"deploy_scene_object_discovery_directory_readback_mismatch:{path}"
            )
        receipts.append(
            {
                "path": str(path),
                "account": account,
                "owner_uid": owner_uid,
                "owner_gid": owner_gid,
                "mode": "0750",
            }
        )
    return receipts


def _install_episode_compilation_runtime_directories(
    *,
    directories: Sequence[str] = DEFAULT_EPISODE_COMPILATION_RUNTIME_DIRECTORIES,
    account: str = DEFAULT_SERVICE_ACCOUNT,
) -> list[dict[str, Any]]:
    """Install every state watched or consumed by the episode queue units."""

    account_ids = _service_account_ids(account)
    if account_ids is None:
        raise ControlPlaneDeployError(
            f"deploy_episode_compilation_account_missing:{account}"
        )
    owner_uid, owner_gid = account_ids
    receipts: list[dict[str, Any]] = []
    for raw_path in directories:
        path = Path(raw_path)
        if not path.is_absolute():
            raise ControlPlaneDeployError(
                "deploy_episode_compilation_directory_not_absolute"
            )
        if path.is_symlink():
            raise ControlPlaneDeployError(
                f"deploy_episode_compilation_directory_symlink:{path}"
            )
        try:
            path.mkdir(parents=True, exist_ok=True, mode=0o750)
            if path.is_symlink() or not path.is_dir():
                raise ControlPlaneDeployError(
                    f"deploy_episode_compilation_directory_invalid:{path}"
                )
            metadata = path.stat()
            if metadata.st_uid != owner_uid or metadata.st_gid != owner_gid:
                os.chown(path, owner_uid, owner_gid)
            if stat.S_IMODE(metadata.st_mode) != 0o750:
                path.chmod(0o750)
            readback = path.stat()
        except OSError as exc:
            raise ControlPlaneDeployError(
                f"deploy_episode_compilation_directory_install_failed:{path}"
            ) from exc
        if (
            readback.st_uid != owner_uid
            or readback.st_gid != owner_gid
            or stat.S_IMODE(readback.st_mode) != 0o750
        ):
            raise ControlPlaneDeployError(
                f"deploy_episode_compilation_directory_readback_mismatch:{path}"
            )
        receipts.append(
            {
                "path": str(path),
                "account": account,
                "owner_uid": owner_uid,
                "owner_gid": owner_gid,
                "mode": "0750",
            }
        )
    return receipts


def _install_storage_pins_runtime_root(
    *,
    pins_root: str | Path = DEFAULT_PINS_ROOT,
    account: str = DEFAULT_SERVICE_ACCOUNT,
    chown: Any = os.chown,
    stat_reader: Any = lambda path: path.stat(),
) -> dict[str, Any]:
    """Install the service-owned root required by every storage-pin writer."""

    account_ids = _service_account_ids(account)
    if account_ids is None:
        raise ControlPlaneDeployError(
            f"deploy_storage_pins_account_missing:{account}"
        )
    owner_uid, owner_gid = account_ids
    root = Path(pins_root).expanduser()
    if not root.is_absolute():
        raise ControlPlaneDeployError("deploy_storage_pins_root_not_absolute")
    if root.is_symlink():
        raise ControlPlaneDeployError(
            f"deploy_storage_pins_root_symlink:{root}"
        )
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o750)
        if root.is_symlink() or not root.is_dir():
            raise ControlPlaneDeployError(
                f"deploy_storage_pins_root_invalid:{root}"
            )
        metadata = stat_reader(root)
        repaired = False
        if metadata.st_uid != owner_uid or metadata.st_gid != owner_gid:
            chown(root, owner_uid, owner_gid)
            repaired = True
            metadata = stat_reader(root)
        if stat.S_IMODE(metadata.st_mode) != 0o750:
            root.chmod(0o750)
            repaired = True
        readback = stat_reader(root)
    except ControlPlaneDeployError:
        raise
    except OSError as exc:
        raise ControlPlaneDeployError(
            f"deploy_storage_pins_root_install_failed:{root}"
        ) from exc
    if (
        readback.st_uid != owner_uid
        or readback.st_gid != owner_gid
        or stat.S_IMODE(readback.st_mode) != 0o750
    ):
        raise ControlPlaneDeployError(
            f"deploy_storage_pins_root_readback_mismatch:{root}"
        )
    return {
        "status": "ready",
        "path": str(root),
        "account": account,
        "owner_uid": owner_uid,
        "owner_gid": owner_gid,
        "mode": "0750",
        "repaired": repaired,
    }


def _install_configured_controls_runtime_prerequisites(
    *,
    plan_root: str = DEFAULT_CONFIGURED_CONTROLS_PLAN_ROOT,
    webapp_secret: str = DEFAULT_CONFIGURED_CONTROLS_WEBAPP_SECRET,
    account: str = DEFAULT_SERVICE_ACCOUNT,
    root_uid: int = 0,
    chown: Any = os.chown,
    stat_reader: Any = lambda path: path.stat(),
) -> dict[str, Any]:
    """Provision the timer plan root and secret permission without reading it."""

    account_ids = _service_account_ids(account)
    if account_ids is None:
        raise ControlPlaneDeployError(
            f"deploy_configured_controls_account_missing:{account}"
        )
    owner_uid, owner_gid = account_ids
    root = Path(plan_root)
    secret = Path(webapp_secret)
    if (
        not root.is_absolute()
        or not secret.is_absolute()
        or root.is_symlink()
        or secret.is_symlink()
    ):
        raise ControlPlaneDeployError(
            "deploy_configured_controls_runtime_path_invalid"
        )
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o750)
        if not root.is_dir() or not secret.is_file():
            raise ControlPlaneDeployError(
                "deploy_configured_controls_runtime_prerequisite_missing"
            )
        root_metadata = stat_reader(root)
        if root_metadata.st_uid != owner_uid or root_metadata.st_gid != owner_gid:
            chown(root, owner_uid, owner_gid)
        if stat.S_IMODE(root_metadata.st_mode) != 0o750:
            root.chmod(0o750)
        secret_metadata = stat_reader(secret)
        if secret_metadata.st_uid != root_uid or secret_metadata.st_gid != owner_gid:
            chown(secret, root_uid, owner_gid)
        if stat.S_IMODE(secret_metadata.st_mode) != 0o440:
            secret.chmod(0o440)
        root_readback = stat_reader(root)
        secret_readback = stat_reader(secret)
    except OSError as exc:
        raise ControlPlaneDeployError(
            "deploy_configured_controls_runtime_prerequisite_install_failed"
        ) from exc
    if (
        root_readback.st_uid != owner_uid
        or root_readback.st_gid != owner_gid
        or stat.S_IMODE(root_readback.st_mode) != 0o750
        or secret_readback.st_uid != root_uid
        or secret_readback.st_gid != owner_gid
        or stat.S_IMODE(secret_readback.st_mode) != 0o440
    ):
        raise ControlPlaneDeployError(
            "deploy_configured_controls_runtime_prerequisite_readback_mismatch"
        )
    return {
        "plan_root": str(root),
        "plan_root_owner_uid": owner_uid,
        "plan_root_owner_gid": owner_gid,
        "plan_root_mode": "0750",
        "webapp_secret": str(secret),
        "webapp_secret_owner_uid": root_uid,
        "webapp_secret_owner_gid": owner_gid,
        "webapp_secret_mode": "0440",
        "secret_bytes_read": False,
    }


def _install_configured_controls_autostart_registry(
    *,
    intent_root: str = DEFAULT_CONFIGURED_CONTROLS_AUTOSTART_INTENT_ROOT,
    intent_sources: Sequence[str] = (),
    source_commit: str,
    account: str = DEFAULT_SERVICE_ACCOUNT,
    root_uid: int = 0,
) -> dict[str, Any]:
    """Install immutable per-scene continuation intent with readback proof.

    This registry is a pre-admission boundary, not universal task inference: a
    scene-configuration activation is admitted only when its exact team/scene/
    task intent was provisioned at the same production commit.
    """

    account_ids = _service_account_ids(account)
    if account_ids is None:
        raise ControlPlaneDeployError(
            f"deploy_configured_controls_account_missing:{account}"
        )
    _owner_uid, owner_gid = account_ids
    root = Path(intent_root).expanduser()
    if not root.is_absolute() or root.is_symlink():
        raise ControlPlaneDeployError(
            "deploy_configured_controls_autostart_intent_root_invalid"
        )
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o750)
        if root.is_symlink() or not root.is_dir():
            raise ControlPlaneDeployError(
                "deploy_configured_controls_autostart_intent_root_invalid"
            )
        metadata = root.stat()
        if metadata.st_uid != root_uid or metadata.st_gid != owner_gid:
            os.chown(root, root_uid, owner_gid)
        if stat.S_IMODE(metadata.st_mode) != 0o750:
            root.chmod(0o750)
        readback = root.stat()
    except OSError as exc:
        raise ControlPlaneDeployError(
            "deploy_configured_controls_autostart_intent_root_install_failed"
        ) from exc
    if (
        readback.st_uid != root_uid
        or readback.st_gid != owner_gid
        or stat.S_IMODE(readback.st_mode) != 0o750
    ):
        raise ControlPlaneDeployError(
            "deploy_configured_controls_autostart_intent_root_readback_mismatch"
        )

    entries: list[dict[str, Any]] = []
    for raw_source in intent_sources:
        source = Path(raw_source).expanduser()
        if not source.is_absolute() or source.is_symlink() or not source.is_file():
            raise ControlPlaneDeployError(
                "deploy_configured_controls_autostart_intent_source_invalid"
            )
        try:
            payload = source.read_bytes()
            value = validate_configured_controls_autostart_intent(
                json.loads(payload)
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ControlPlaneDeployError(
                "deploy_configured_controls_autostart_intent_source_invalid"
            ) from exc
        if value["expected_production_commit"] != source_commit:
            raise ControlPlaneDeployError(
                "deploy_configured_controls_autostart_intent_commit_mismatch"
            )
        adoption = value["configuration_adoption"]
        if adoption["mode"] == "explicit_terminal_adoption":
            from blueprint_pipeline.task_evaluation_configured_controls_autostart import (
                configured_controls_autostart_adoption_registry_name,
            )

            destination = root / configured_controls_autostart_adoption_registry_name(
                team_namespace=value["team_namespace"],
                scene_id=value["scene_id"],
                task_id=value["task_id"],
                source_launch_id=adoption["source_launch_id"],
            )
        else:
            destination = root / configured_controls_autostart_registry_name(
                team_namespace=value["team_namespace"],
                scene_id=value["scene_id"],
                task_id=value["task_id"],
            )
        previous_sha256: str | None = None
        try:
            if destination.exists():
                if destination.is_symlink() or not destination.is_file():
                    raise ControlPlaneDeployError(
                        "deploy_configured_controls_autostart_intent_conflict"
                    )
                previous_payload = destination.read_bytes()
                try:
                    previous = validate_configured_controls_autostart_intent(
                        json.loads(previous_payload)
                    )
                except (
                    UnicodeError,
                    json.JSONDecodeError,
                    ValueError,
                ) as exc:
                    raise ControlPlaneDeployError(
                        "deploy_configured_controls_autostart_existing_intent_invalid"
                    ) from exc
                if (
                    previous["team_namespace"] != value["team_namespace"]
                    or previous["scene_id"] != value["scene_id"]
                    or previous["task_id"] != value["task_id"]
                ):
                    raise ControlPlaneDeployError(
                        "deploy_configured_controls_autostart_intent_conflict"
                    )
                previous_sha256 = _sha256_bytes(previous_payload)
            if not destination.exists() or destination.read_bytes() != payload:
                temporary: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        dir=root,
                        prefix=f".{destination.name}.",
                        delete=False,
                    ) as stream:
                        temporary = Path(stream.name)
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.chown(temporary, root_uid, owner_gid)
                    temporary.chmod(0o440)
                    os.replace(temporary, destination)
                    temporary = None
                    directory_fd = os.open(root, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                finally:
                    if temporary is not None:
                        with contextlib.suppress(OSError):
                            temporary.unlink()
            else:
                os.chown(destination, root_uid, owner_gid)
                destination.chmod(0o440)
            destination_payload = destination.read_bytes()
            destination_metadata = destination.stat()
        except OSError as exc:
            raise ControlPlaneDeployError(
                "deploy_configured_controls_autostart_intent_install_failed"
            ) from exc
        if (
            destination.is_symlink()
            or destination_payload != payload
            or destination_metadata.st_uid != root_uid
            or destination_metadata.st_gid != owner_gid
            or stat.S_IMODE(destination_metadata.st_mode) != 0o440
        ):
            raise ControlPlaneDeployError(
                "deploy_configured_controls_autostart_intent_readback_mismatch"
            )
        entries.append(
            {
                "path": str(destination),
                "sha256": _sha256_bytes(payload),
                "size_bytes": len(payload),
                "mode": "0440",
                "expected_production_commit": source_commit,
                "team_namespace": value["team_namespace"],
                "scene_id": value["scene_id"],
                "task_id": value["task_id"],
                "intent_digest": value["intent_digest"],
                "configuration_adoption_mode": adoption["mode"],
                "replaced_previous_sha256": (
                    previous_sha256
                    if previous_sha256 != _sha256_bytes(payload)
                    else None
                ),
            }
        )
    return {
        "root": str(root),
        "root_owner_uid": root_uid,
        "root_owner_gid": owner_gid,
        "root_mode": "0750",
        "status": "provisioned" if entries else "empty_pre_admission_required",
        "pre_admission_required": True,
        "entry_count": len(entries),
        "entries": entries,
    }


@contextlib.contextmanager
def _holding_paid_launch_locks(lock_paths: Sequence[str]):
    """Hold the provider's own launch lock for the whole deploy.

    Checking whether a lock is held and then deploying is two steps, and a
    launch can start between them -- which is exactly what happened on
    2026-08-13: the check passed and the parallel lane acquired the lock 20
    seconds later, mid-deploy.

    `vast_provider_adapter` guards a paid launch with `fcntl.flock` on this
    file, so taking the same lock makes deploy and launch genuinely exclusive
    rather than politely sequenced. While the deploy holds it a launch refuses
    with `vast_paid_launch_lock_busy`, which is the correct outcome: a run must
    not start against a release that is being swapped underneath it.

    Opened read-only and never created. The adapter creates this file as the
    service account at 0600; a deploy running as root that created it first
    would leave a file the service can never open again, taking every paid lane
    down. A lock that does not exist yet means no adapter has ever launched
    here, so there is nothing to be exclusive with.
    """

    handles: list[Any] = []
    try:
        for path in _expanded_slots(lock_paths):
            try:
                handle = path.open("r", encoding="utf-8")
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise ControlPlaneDeployError(
                    f"deploy_paid_launch_lock_unreadable:{path.name}"
                ) from exc
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.seek(0)
                holder = handle.read(1000)
                handle.close()
                # Slots already taken are released by the `finally` below.
                # Closing them here too made the unlock operate on a closed
                # file, which raises ValueError -- not the OSError the cleanup
                # suppresses -- so a refusal crashed instead of refusing.
                raise ControlPlaneDeployError(
                    "deploy_refused_paid_launch_in_flight:" + _holder_summary(holder)
                ) from None
            handles.append(handle)
        yield
    finally:
        for handle in handles:
            with contextlib.suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()


#: A launch holds the gate shared only while taking a slot, so the wait is short.
PAID_LAUNCH_GATE_WAIT_SECONDS = 120


@contextlib.contextmanager
def _holding_paid_launch_gate(
    lock_paths: Sequence[str],
    *,
    wait_seconds: float | None = None,
    sleeper: Any = time.sleep,
    clock: Any = time.monotonic,
):
    """Close paid launches for the deploy without waiting out runs in flight.

    Holding every slot made a deploy wait for the longest paid run, and with
    three slots busy a deploy could wait for hours. What the deploy must
    exclude is a launch *starting* while the release is swapped (2026-08-13).
    A run that started earlier runs on its own immutable release tree, which
    retirement never removes while a live process uses it.

    So the deploy takes the gate exclusively, which new launches take shared
    while they acquire a slot, and also holds every free slot, so a process
    still running pre-gate adapter code cannot start a launch either. Busy
    slots are reported as in flight, not refused. Files are opened read-only
    and never created, for the same ownership reason as the slots.
    """

    from blueprint_pipeline.vast_provider_adapter import vast_launch_gate_path

    handles: list[Any] = []
    in_flight: list[dict[str, Any]] = []
    try:
        for raw in lock_paths:
            gate = vast_launch_gate_path(Path(raw).expanduser())
            try:
                handle = gate.open("r", encoding="utf-8")
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise ControlPlaneDeployError(
                    f"deploy_paid_launch_gate_unreadable:{gate.name}"
                ) from exc
            deadline = clock() + (
                PAID_LAUNCH_GATE_WAIT_SECONDS if wait_seconds is None else wait_seconds
            )
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if clock() >= deadline:
                        handle.close()
                        raise ControlPlaneDeployError(
                            "deploy_refused_paid_launch_gate_busy"
                        ) from None
                    sleeper(1.0)
            handles.append(handle)
        for path in _expanded_slots(lock_paths):
            try:
                handle = path.open("r", encoding="utf-8")
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise ControlPlaneDeployError(
                    f"deploy_paid_launch_lock_unreadable:{path.name}"
                ) from exc
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.seek(0)
                holder = handle.read(1000)
                handle.close()
                in_flight.append({
                    "slot": path.name,
                    "holder": _holder_summary(holder),
                    "pid": _holder_pid(holder),
                })
                continue
            handles.append(handle)
        yield in_flight
    finally:
        for handle in handles:
            with contextlib.suppress(OSError, ValueError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()


def _holder_pid(holder: str) -> int | None:
    try:
        record = json.loads(holder)
    except (ValueError, json.JSONDecodeError):
        return None
    pid = record.get("pid") if isinstance(record, Mapping) else None
    return pid if type(pid) is int and pid > 0 else None


def _require_in_flight_runs_outside_units(
    in_flight: Sequence[Mapping[str, Any]],
    units: Sequence[str],
    *,
    proc_root: str | Path = "/proc",
) -> None:
    """A restart must never take down the controller of a live paid instance."""

    for run in in_flight:
        pid = run.get("pid")
        if pid is None:
            # A holder that names no process cannot be placed outside a unit.
            raise ControlPlaneDeployError(
                "deploy_refused_paid_launch_holder_unknown:" + str(run.get("holder"))
            )
        try:
            cgroup = (Path(proc_root) / str(pid) / "cgroup").read_text(encoding="utf-8")
        except OSError:
            continue  # Gone: its slot is free again and nothing is restarted under it.
        for unit in units:
            if f"/{unit}" in cgroup:
                raise ControlPlaneDeployError(
                    f"deploy_refused_paid_launch_in_restarted_unit:{unit}:{run.get('holder')}"
                )


def _holder_summary(holder: str) -> str:
    """Name the run that holds the lock, not the file that records it."""

    try:
        record = json.loads(holder)
    except (ValueError, json.JSONDecodeError):
        return "unparseable_holder"
    if not isinstance(record, Mapping):
        return "unparseable_holder"
    return str(record.get("job_dir") or record.get("pid") or "unknown_holder")


def _git(repo: Path, *arguments: str) -> tuple[int, str]:
    result = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout.strip()


RELEASE_ESTIMATE_HEADROOM = 1.25
RELEASE_ESTIMATE_BLOCK_BYTES = 4096
RELEASE_ESTIMATE_OVERHEAD_BYTES = 256 * 1024**2


def _release_footprint_estimate(
    source: Path,
    commit: str,
    *,
    reservation_root: str | Path = DEFAULT_RESERVATION_ROOT,
) -> dict[str, Any]:
    """What staging ``commit`` costs on disk, read from its git tree.

    The blob bytes plus a quarter for block rounding, one block per file, and
    256 MiB for the checkout's index and the runtime trees provisioned beside
    it.  When the tree cannot be listed, the deploy role's measured footprint
    (or its declared ceiling while the history is short) stands in.
    """

    code, listing = _git(Path(source), "ls-tree", "-r", "-l", "--full-tree", commit)
    tree_bytes = 0
    file_count = 0
    readable = code == 0
    if readable:
        for line in listing.splitlines():
            # "<mode> <type> <object> <size>\t<path>"; a submodule's size is "-".
            fields = line.split("\t", 1)[0].split()
            if len(fields) != 4 or not (fields[3] == "-" or fields[3].isdigit()):
                readable = False
                break
            file_count += 1
            tree_bytes += 0 if fields[3] == "-" else int(fields[3])
    if not readable:
        measured = measured_footprint(
            "control_plane_deploy", reservation_root=reservation_root
        )
        return {
            "bytes": int(measured["bytes"]),
            "basis": measured["basis"],
            "sample_count": measured["sample_count"],
            "tree_bytes": None,
            "file_count": None,
        }
    return {
        "bytes": math.ceil(tree_bytes * RELEASE_ESTIMATE_HEADROOM)
        + RELEASE_ESTIMATE_BLOCK_BYTES * file_count
        + RELEASE_ESTIMATE_OVERHEAD_BYTES,
        "basis": "git_tree_estimate",
        "tree_bytes": tree_bytes,
        "file_count": file_count,
    }


def _release_runtime_trees(runtime_root: Path, commit: str) -> set[Path]:
    """The per-release runtime trees, ``<runtime_root>/<component>/<commit>``, present now."""

    try:
        return {
            path
            for path in runtime_root.glob(f"*/{commit}")
            if path.is_dir() and not path.is_symlink()
        }
    except OSError:
        return set()


def _created_release_usage(
    *,
    created_release_checkout: bool,
    release_path: str | Path,
    runtime_root: Path,
    commit: str,
    runtime_trees_before: set[Path],
) -> TreeUsage | None:
    """Allocated bytes and scan completeness, or None when nothing was created.

    A redeploy that reuses an existing checkout and runtime trees costs nothing
    to stage; recording it as a zero-byte sample would only drag the role's
    measured footprint down, so it records no sample at all.
    """

    created_trees = _release_runtime_trees(runtime_root, commit) - runtime_trees_before
    if not created_release_checkout and not created_trees:
        return None
    usages = ([tree_usage(release_path)] if created_release_checkout else []) + [
        tree_usage(path) for path in sorted(created_trees)
    ]
    return TreeUsage(
        allocated_bytes=sum(usage.allocated_bytes for usage in usages),
        unreadable=sum(usage.unreadable for usage in usages),
    )


def _observe_created_release_usage(
    reservation: DiskReservation, usage: TreeUsage | None,
) -> None:
    if usage is None:
        return
    reservation.observe(usage.allocated_bytes)
    if usage.unreadable:
        reservation.measurement_incomplete = True


def _surface_commit(path: Path, *, name: str) -> str:
    """What commit does this surface *say* it is? Refuse if it cannot say."""

    code, head = _git(path, "rev-parse", "HEAD")
    if code != 0 or not head:
        # A tree extracted from an archive lands here: correct bytes, no
        # identity, and every downstream identity probe silently empty.
        raise ControlPlaneDeployError(f"deploy_surface_has_no_git_identity:{name}")
    code, dirty = _git(path, "status", "--porcelain")
    if code != 0:
        raise ControlPlaneDeployError(f"deploy_surface_status_unavailable:{name}")
    if dirty:
        raise ControlPlaneDeployError(f"deploy_surface_checkout_dirty:{name}")
    return head


def _move_source_checkout(repo: Path, commit: str) -> None:
    if _git(repo, "status", "--porcelain")[1]:
        # Never carry a local edit across a deploy: it would be running code
        # that is on no commit at all.
        raise ControlPlaneDeployError("deploy_source_checkout_dirty")
    if _git(repo, "fetch", "--quiet", "origin", "main")[0] != 0:
        raise ControlPlaneDeployError("deploy_source_fetch_failed")
    if _git(repo, "checkout", "--quiet", commit)[0] != 0:
        raise ControlPlaneDeployError(f"deploy_source_checkout_failed:{commit}")


def _restart_units(units: Sequence[str]) -> list[dict[str, Any]]:
    reload_result = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
        ["systemctl", "daemon-reload"], capture_output=True, text=True, check=False
    )
    if reload_result.returncode != 0:
        raise ControlPlaneDeployError("deploy_systemd_daemon_reload_failed")
    restarted: list[dict[str, Any]] = []
    for unit in units:
        result = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
            ["systemctl", "restart", unit], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise ControlPlaneDeployError(f"deploy_unit_restart_failed:{unit}")
        active = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
            ["systemctl", "is-active", unit], capture_output=True, text=True, check=False
        )
        state = active.stdout.strip()
        if state != "active":
            raise ControlPlaneDeployError(f"deploy_unit_not_active:{unit}:{state}")
        restarted.append({"unit": unit, "state": state})
    return restarted


def _install_release_systemd_units(
    *,
    release_path: str | Path,
    systemd_dir: str | Path,
    units: Sequence[str] = DEFAULT_DEPLOYED_SYSTEMD_UNITS,
) -> list[dict[str, Any]]:
    """Install exact release-owned unit bytes before daemon reload.

    Promoting a detached release without refreshing its installed unit left the
    dispatcher on older concurrency and watchdog-survival semantics.  The
    allocator then ran exact new Python under stale systemd controls and failed
    before provider allocation.  Install only the release-owned Task Evaluation
    queue pairs here; the ordinary restart seam immediately daemon-reloads them,
    and the next queue activation therefore uses the same release that authored
    the request or profile.

    The pair includes the queue-watching ``.path`` unit: a release that changed
    how the queue wakes the dispatcher (PR #1057 added ``PathChanged=``) was
    otherwise deployed with only its ``.service`` refreshed, leaving the
    watcher on whatever bytes an operator had once copied by hand.
    """

    release = Path(release_path).expanduser().resolve()
    destination_root = Path(systemd_dir).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []
    for unit in units:
        if Path(unit).name != unit or not unit.endswith(
            DEPLOYED_SYSTEMD_UNIT_SUFFIXES
        ):
            raise ControlPlaneDeployError("deploy_systemd_unit_name_invalid")
        source = release / "deploy" / "systemd" / unit
        destination = destination_root / unit
        if source.is_symlink() or not source.is_file():
            raise ControlPlaneDeployError(f"deploy_systemd_unit_source_invalid:{unit}")
        if destination.is_symlink():
            raise ControlPlaneDeployError(
                f"deploy_systemd_unit_destination_symlink:{unit}"
            )
        try:
            payload = source.read_bytes()
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{unit}.", suffix=".tmp", dir=destination_root
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, 0o644)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            reopened = destination.read_bytes()
        except OSError as exc:
            raise ControlPlaneDeployError(
                f"deploy_systemd_unit_install_failed:{unit}"
            ) from exc
        if reopened != payload or destination.stat().st_mode & 0o777 != 0o644:
            raise ControlPlaneDeployError(
                f"deploy_systemd_unit_readback_mismatch:{unit}"
            )
        receipts.append(
            {
                "unit": unit,
                "source_path": str(source),
                "installed_path": str(destination),
                "sha256": _sha256_bytes(reopened),
                "size_bytes": len(reopened),
                "mode": "0644",
            }
        )
    return receipts


def _systemd_unit_state(unit: str, *, deadline: float | None = None) -> dict[str, str]:
    """Read enabled/active state without changing the unit."""

    states: dict[str, str] = {}
    for probe in ("is-enabled", "is-active"):
        timeout = 15.0 if deadline is None else min(15.0, deadline - time.monotonic())
        if timeout <= 0:
            raise ControlPlaneDeployError(f"deploy_systemd_state_probe_failed:{unit}:{probe}")
        try:
            result = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
                ["systemctl", probe, unit],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ControlPlaneDeployError(
                f"deploy_systemd_state_probe_failed:{unit}:{probe}"
            ) from exc
        if deadline is not None and time.monotonic() >= deadline:
            raise ControlPlaneDeployError(f"deploy_systemd_state_probe_failed:{unit}:{probe}")
        state = result.stdout.strip() or (
            "disabled" if probe == "is-enabled" else "inactive"
        )
        if state in {"not-found", "unknown"}:
            state = "disabled" if probe == "is-enabled" else "inactive"
        states[probe] = state
    return {"enabled": states["is-enabled"], "state": states["is-active"]}


def _installed_path_unit_states(
    installed_units: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    """Snapshot path/timer intent before unit bytes or daemon state move."""

    return {
        unit: _systemd_unit_state(unit)
        for entry in installed_units
        if (unit := str(entry.get("unit") or "")).endswith((".path", ".timer"))
    }


def _quiesce_active_path_units(
    before: Mapping[str, Mapping[str, str]],
) -> list[dict[str, str]]:
    """Stop only watchers that were active, before release surfaces move.

    Paid-launch locks stop provider allocation, but an armed watcher could
    still claim a newly published request while source and release identities
    are changing.  Quiesce it first, retain the prior state, and restore that
    exact intent only after intake proves the new commit.
    """

    stopped: list[dict[str, str]] = []
    for unit, state in before.items():
        if state.get("state") != "active":
            continue
        result = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
            ["systemctl", "stop", unit],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise ControlPlaneDeployError(
                f"deploy_path_unit_quiesce_failed:{unit}"
            )
        observed = _systemd_unit_state(unit)
        if observed["state"] != "inactive":
            raise ControlPlaneDeployError(
                f"deploy_path_unit_quiesce_state_mismatch:{unit}:"
                f"{observed['state']}"
            )
        stopped.append({"unit": unit, "state": observed["state"]})
    return stopped


def _active_door_holds(root: str | Path, *, now: float | None = None) -> tuple[dict[str, dict[str, Any]], str | None]:
    """Read the root-owned hold records without trusting symlinks or malformed JSON."""

    directory = Path(root)
    if directory.is_symlink():
        return {}, "door_holds_unreadable"
    if not directory.exists():
        return {}, None
    if not directory.is_dir():
        return {}, "door_holds_unreadable"
    moment = time.time() if now is None else now
    found: dict[str, dict[str, Any]] = {}
    try:
        paths = list(directory.glob("*.json"))
        if len(paths) > 1024:
            return {}, "door_holds_unreadable"
        for path in paths:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_size > 16 * 1024:
                    raise ValueError("hold_record_unsafe")
                raw = os.read(fd, 16 * 1024 + 1)
            finally:
                os.close(fd)
            if len(raw) > 16 * 1024:
                raise ValueError("hold_record_unsafe")
            record = json.loads(raw)
            unit = path.name.removesuffix(".json")
            if (not isinstance(record, dict) or record.get("schema") != "blueprint_operator_door_hold.v1"
                    or record.get("unit") != unit or not re.fullmatch(r"blueprint-[A-Za-z0-9_.@-]+\.(timer|path)", unit)
                    or not isinstance(record.get("owner"), str) or not isinstance(record.get("reason"), str)
                    or not isinstance(record.get("expires_at"), str)
                    or type(record.get("expires_at_epoch")) is not int
                    or type(record.get("require_explicit_release", False)) is not bool
                    or (record.get("require_explicit_release") is True
                        and unit != "blueprint-agent-run-dispatcher.timer")):
                raise ValueError("hold_record_invalid")
            if record.get("status") == "active" and (record["expires_at_epoch"] > moment
                                                    or record.get("require_explicit_release") is True):
                found[unit] = record
    except (OSError, ValueError, UnicodeDecodeError):
        return {}, "door_holds_unreadable"
    return found, None


@contextlib.contextmanager
def _locked_door_holds(root: str | Path, *, deadline: float | None = None):
    """Keep a matching expiry or new hold from racing the deploy's unit restore."""

    directory = Path(root)
    if directory.is_symlink() or not directory.exists() or not directory.is_dir():
        yield _active_door_holds(directory)
        return
    fd: int | None = None
    try:
        fd = os.open(directory / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, "unsafe hold lock")
        if deadline is None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        else:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ControlPlaneDeployError("deploy_door_holds_lock_timeout")
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(min(0.1, remaining))
    except ControlPlaneDeployError:
        if fd is not None:
            os.close(fd)
        raise
    except OSError:
        if fd is not None:
            os.close(fd)
        yield {}, "door_holds_unreadable"
        return
    try:
        yield _active_door_holds(directory)
    finally:
        os.close(fd)


def _restore_installed_path_units(
    installed_units: Sequence[Mapping[str, Any]],
    *,
    before: Mapping[str, Mapping[str, str]],
    arm_path_units: bool,
    always_arm_units: Sequence[str] = (),
    always_arm_authority_gated_units: Sequence[str] = (),
    always_arm_timer_units: Sequence[str] = (),
    preserve_configured_controls_state: bool = False,
    held_units: Mapping[str, Mapping[str, Any]] | None = None,
    defer_start_verification: bool = False,
) -> list[dict[str, Any]]:
    """Restore path/timer intent without widening arbitrary launch authority.

    A stopped paid watcher is an operational freeze, so paths preserve both boot
    and active state unless explicitly armed or fixed as no-spend.  The one
    configured-controls timer is a separate fixed category: it only consumes
    sealed qualifying plans through canonical APIs, and its receipt names that
    progression authority instead of misreporting it as no-spend.
    """

    if preserve_configured_controls_state and arm_path_units:
        raise ControlPlaneDeployError("deploy_conflicting_configured_controls_intent")
    receipts: list[dict[str, Any]] = []
    for entry in installed_units:
        unit = str(entry.get("unit") or "")
        if not unit.endswith((".path", ".timer")):
            continue
        prior = dict(before.get(unit) or {"enabled": "disabled", "state": "inactive"})
        hold = (held_units or {}).get(unit)
        if hold is not None:
            # Keep the existing boot policy; do not enable, restart or start a
            # trigger whose owner has explicitly held it through this deploy.
            result = subprocess.run(  # nosec B603 B607 - fixed systemctl argv
                ["systemctl", "stop", unit], capture_output=True, text=True, check=False,
            )
            after = _systemd_unit_state(unit)
            if result.returncode != 0 and after["state"] != "inactive":
                raise ControlPlaneDeployError(f"deploy_held_unit_stop_failed:{unit}")
            if after["state"] != "inactive":
                raise ControlPlaneDeployError(f"deploy_held_unit_active_state_mismatch:{unit}:{after['state']}")
            receipts.append({"unit": unit, "before": prior, "requested_intent": "hold", "after": after,
                             "operator_freeze_preserved": True, "held": True,
                             "owner": hold["owner"], "reason": hold["reason"],
                             "expires_at": hold["expires_at"],
                             **({"require_explicit_release": True}
                                if hold.get("require_explicit_release") is True else {})})
            continue
        arm_no_spend = unit in always_arm_units
        arm_authority_gated = unit in always_arm_authority_gated_units
        arm_progression = unit in always_arm_timer_units and not (
            preserve_configured_controls_state
            and unit in CONFIGURED_CONTROLS_AUTOMATION_UNITS
        )
        if sum((arm_no_spend, arm_authority_gated, arm_progression)) > 1:
            raise ControlPlaneDeployError(
                f"deploy_automation_unit_authority_ambiguous:{unit}"
            )
        explicit_path_arm = arm_path_units and unit.endswith(".path")
        should_enable = (
            explicit_path_arm
            or arm_no_spend
            or arm_authority_gated
            or arm_progression
            or prior.get("enabled") == "enabled"
        )
        should_start = (
            explicit_path_arm
            or arm_no_spend
            or arm_authority_gated
            or arm_progression
            or prior.get("state") == "active"
        )
        commands = ["enable" if should_enable else "disable"]
        commands.append("restart" if should_start else "stop")
        for verb in commands:
            deferred = defer_start_verification and verb == "restart"
            result = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
                ["systemctl", *(["--no-block"] if deferred else []), verb, unit],
                capture_output=True,
                text=True,
                check=False,
                **({"timeout": 15} if deferred else {}),
            )
            if result.returncode != 0:
                observed = _systemd_unit_state(unit)
                already_restored = (
                    verb == "disable" and observed["enabled"] == "disabled"
                ) or (verb == "stop" and observed["state"] == "inactive")
                if already_restored:
                    continue
                raise ControlPlaneDeployError(
                    f"deploy_path_unit_state_restore_failed:{unit}:{verb}"
                )
        after = _systemd_unit_state(unit)
        expected_enabled = "enabled" if should_enable else "disabled"
        expected_state = "active" if should_start else "inactive"
        if after["enabled"] != expected_enabled:
            raise ControlPlaneDeployError(
                f"deploy_path_unit_enabled_state_mismatch:{unit}:"
                f"{after['enabled']}:{expected_enabled}"
            )
        if after["state"] != expected_state and not (defer_start_verification and should_start):
            raise ControlPlaneDeployError(
                f"deploy_path_unit_active_state_mismatch:{unit}:"
                f"{after['state']}:{expected_state}"
            )
        receipts.append(
            {
                "unit": unit,
                "before": prior,
                "requested_intent": (
                    "arm"
                    if explicit_path_arm
                    else "arm_no_spend"
                    if arm_no_spend
                    else "arm_authority_gated_paid_dispatch"
                    if arm_authority_gated
                    else "arm_configured_controls_progression"
                    if arm_progression
                    else "preserve"
                ),
                "after": after,
                "operator_freeze_preserved": (
                    not explicit_path_arm
                    and not arm_no_spend
                    and not arm_authority_gated
                    and not arm_progression
                    and not should_start
                ),
                **({"_start_pending": True} if defer_start_verification and should_start else {}),
            }
        )
    return receipts


def _verify_deferred_path_unit_starts(
    receipts: Sequence[dict[str, Any]], *, door_holds_dir: str | Path,
    timeout_seconds: float = 60,
) -> str | None:
    """Wait outside the hold lock so a timer's hold-sweep dependency can run.

    Starts are enqueued under the lock to serialize them with new holds. Each
    probe rechecks current holds under that same lock: a subsequent owner hold
    may cancel a queued start and must still be reported as held, never rearmed.
    """
    pending = [row for row in receipts if row.get("_start_pending")]
    deadline = time.monotonic() + timeout_seconds
    warning = None
    while pending:
        with _locked_door_holds(door_holds_dir, deadline=deadline) as (held_units, hold_warning):
            warning = warning or hold_warning
            for row in pending[:]:
                unit = row["unit"]
                hold = held_units.get(unit)
                if time.monotonic() >= deadline:
                    raise ControlPlaneDeployError(f"deploy_path_unit_start_timeout:{unit}")
                after = _systemd_unit_state(unit, deadline=deadline)
                if time.monotonic() >= deadline:
                    raise ControlPlaneDeployError(f"deploy_path_unit_start_timeout:{unit}")
                if hold is not None:
                    if after["state"] != "inactive":
                        continue
                    row.update(requested_intent="hold", operator_freeze_preserved=True,
                               held=True, owner=hold["owner"], reason=hold["reason"],
                               expires_at=hold["expires_at"])
                else:
                    expected_enabled = row["after"]["enabled"]
                    if after["enabled"] != expected_enabled:
                        raise ControlPlaneDeployError(
                            f"deploy_path_unit_enabled_state_mismatch:{unit}:"
                            f"{after['enabled']}:{expected_enabled}"
                        )
                    if after["state"] != "active":
                        continue
                row["after"] = after
                row.pop("_start_pending")
                pending.remove(row)
        if pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ControlPlaneDeployError(
                    "deploy_path_unit_start_timeout:" + ",".join(row["unit"] for row in pending)
                )
            time.sleep(min(0.1, remaining))
    return warning


@contextlib.contextmanager
def _restore_path_unit_states_on_deploy_failure(
    installed_units: Sequence[Mapping[str, Any]],
    *,
    door_holds_dir: str | Path = DEFAULT_DOOR_HOLDS_DIR,
):
    """Quiesce watchers and restore their exact prior state on any failure.

    A deploy intentionally stops queue watchers before validating and
    publishing a release.  The success path restores them after the intake
    proves the new commit, but a failed prerequisite or restart previously
    exited with every formerly active watcher still stopped.  Recovery runs
    while the paid-launch locks remain held and never applies the no-spend or
    explicit-arm widening rules: it restores only the state observed before
    this attempt.
    """

    before = _installed_path_unit_states(installed_units)
    try:
        quiesced = _quiesce_active_path_units(before)
        yield before, quiesced
    except BaseException as deployment_error:
        try:
            with _locked_door_holds(door_holds_dir) as (held_units, _warning):
                restored = _restore_installed_path_units(
                    installed_units,
                    before=before,
                    arm_path_units=False,
                    always_arm_units=(),
                    held_units=held_units,
                    defer_start_verification=True,
                )
            _verify_deferred_path_unit_starts(restored, door_holds_dir=door_holds_dir)
        except Exception as restore_error:
            raise ControlPlaneDeployError(
                "deploy_failed_path_unit_restore_failed:"
                f"{type(deployment_error).__name__}:{restore_error}"
            ) from deployment_error
        raise


def _required_restart_units(units: Sequence[str]) -> tuple[str, ...]:
    """The intake restart is mandatory; callers may only add units."""

    required = list(DEFAULT_RESTART_UNITS)
    for unit in units:
        if unit not in required:
            required.append(unit)
    return tuple(required)


def _drain_agent_execution_before_release_switch(
    *, expected_commit: str, config_path: str | Path = "/etc/blueprint/agent-execution.json"
) -> dict[str, Any]:
    """Quiesce the old agent while its matching worker can still perform cleanup."""
    path = Path(config_path)
    if not path.exists():
        return {"status": "not_configured"}
    from blueprint_pipeline.agent_execution.production import ProductionAgentService, ProductionConfig, _read_private
    from blueprint_pipeline.agent_execution.release import drain, pending_cleanup
    from blueprint_pipeline.agent_execution.journal import AgentJournal

    config = ProductionConfig.model_validate_json(_read_private(path))
    if config.source_commit == expected_commit:
        return {"status": "already_bound", "source_commit": expected_commit}
    if pending_cleanup(AgentJournal(config.state_root)) and (
        _systemd_unit_state("blueprint-agent-execution.service").get("state") != "active"
    ):
        raise ControlPlaneDeployError("deploy_agent_cleanup_worker_not_active")
    try:
        service = ProductionAgentService(path, source_commit=config.source_commit)
        result = drain(service, target_source_commit=expected_commit, timeout_seconds=30, drive_worker=False)
    except Exception as exc:
        raise ControlPlaneDeployError("deploy_agent_configuration_requires_clean_drain") from exc
    if result["status"] != "drained":
        raise ControlPlaneDeployError("deploy_agent_cleanup_reconciliation_pending")
    return result


def _activate_agent_execution(*, expected_commit: str, config_path: str | Path = "/etc/blueprint/agent-execution.json") -> dict[str, Any]:
    """An admitted config activates the worker; absent config never grants inference."""
    path = Path(config_path)
    if not path.exists():
        return {"status": "not_configured", "activated": False}
    from blueprint_pipeline.agent_execution.production import ProductionAgentService, ProductionConfig, _read_private
    config = ProductionConfig.model_validate_json(_read_private(path))
    from blueprint_pipeline.agent_execution.release import adopt_drained_config, drain, pending_cleanup
    from blueprint_pipeline.agent_execution.journal import AgentJournal
    cleanup = {"status": "not_required"}
    try:
        if config.source_commit != expected_commit and pending_cleanup(AgentJournal(config.state_root)):
            # A completed observer can still own its provider session. Drain
            # through the journal's normal cancellation/cleanup protocol; never
            # rewrite terminal records or infer provider deletion from completion.
            if _systemd_unit_state("blueprint-agent-execution.service").get("state") != "active":
                raise ControlPlaneDeployError("deploy_agent_cleanup_worker_not_active")
            service = ProductionAgentService(path, source_commit=config.source_commit)
            cleanup = drain(service, target_source_commit=expected_commit, timeout_seconds=30, drive_worker=False)
            if cleanup["status"] != "drained":
                raise ControlPlaneDeployError("deploy_agent_cleanup_reconciliation_pending")
        adoption = adopt_drained_config(path, expected_commit=expected_commit)
    except Exception as exc:
        raise ControlPlaneDeployError("deploy_agent_configuration_requires_clean_drain") from exc
    unit = "blueprint-agent-execution.service"
    subprocess.run(["systemctl", "enable", unit], check=True, capture_output=True, text=True, timeout=20)
    _restart_units((unit,))
    observed = _systemd_unit_state(unit)
    if observed.get("enabled") != "enabled" or observed.get("state") != "active":
        raise ControlPlaneDeployError("deploy_agent_worker_not_enabled_and_active")
    return {"status": "active", "activated": True, "unit": unit, "source_commit": expected_commit,
            "enabled": observed["enabled"], "state": observed["state"], "configuration_adoption": adoption,
            "release_cleanup": cleanup}


def _require_terminal_controls_quiescence(*, wait_seconds: float = 0) -> None:
    """With triggers stopped, allow an existing materializer to finish naturally."""
    deadline = time.monotonic() + wait_seconds
    for suffix in ('path', 'timer', 'service'):
        unit = 'blueprint-task-evaluation-configured-controls-progression.' + suffix
        while True:
            observed = subprocess.run(['systemctl', 'show', unit, '-p', 'LoadState', '-p', 'ActiveState', '-p', 'MainPID'],
                                      check=True, capture_output=True, text=True, timeout=10)
            fields = dict(line.split('=', 1) for line in observed.stdout.splitlines() if '=' in line)
            if (fields.get('LoadState') == 'loaded' and fields.get('ActiveState') in {'inactive', 'failed'}
                    and fields.get('MainPID', '0') == '0'):
                break
            remaining = deadline - time.monotonic()
            if suffix != 'service' or fields.get('LoadState') != 'loaded' or remaining <= 0:
                raise ControlPlaneDeployError('deploy_terminal_controls_worker_not_quiescent')
            time.sleep(min(2, remaining))
        if fields['ActiveState'] == 'failed':
            subprocess.run(['systemctl', 'reset-failed', unit], check=True, capture_output=True, text=True, timeout=10)


def _terminal_controls_artifact_store_env(release: Path) -> list[str]:
    """Give the deploy helper the same scoped artifact store as its worker."""
    unit = release / 'deploy/systemd/blueprint-task-evaluation-configured-controls-progression.service'
    required = (
        'BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_ACCESS_KEY_ID_FILE',
        'BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_SECRET_ACCESS_KEY_FILE',
        'BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_BUCKET_FILE',
        'BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_ENDPOINT_URL_FILE',
        'BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_REGION_FILE',
        'BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_EXPECTED_BUCKET',
    )
    values: dict[str, str] = {}
    for line in unit.read_text().splitlines():
        if not line.startswith('Environment=') or '=' not in line[len('Environment='):]:
            continue
        name, value = line[len('Environment='):].split('=', 1)
        if name in required:
            if name in values or not value or (name.endswith('_FILE') and
                    not value.startswith('/etc/blueprint/provider-secrets/')):
                raise ControlPlaneDeployError('deploy_terminal_controls_artifact_store_env_invalid')
            values[name] = value
    if set(values) != set(required):
        raise ControlPlaneDeployError('deploy_terminal_controls_artifact_store_env_missing')
    return [f'--setenv={name}={values[name]}' for name in required]


def _prepare_terminal_controls_adoptions(*, release_path: str | Path, commit: str,
                                        config_path: str | Path) -> dict[str, Any]:
    """Use the registry's required quiescence before restoring its worker.

    A worker cannot supersede its own live registration: doing this after the
    timers restart either refuses or races an old materialization. This step
    runs only under the deploy's provider locks and stopped automation triggers.
    """
    _require_terminal_controls_quiescence()
    release = Path(release_path)
    argv = ['systemd-run', '--quiet', '--wait', '--pipe', '--collect',
        '--property=Type=exec', '--property=User=blueprint', '--property=Group=blueprint',
        '--property=EnvironmentFile=-/etc/blueprint/pipeline-control-plane.env',
        '--property=EnvironmentFile=-/etc/blueprint/task-evaluation-scene-progression.env',
        *_terminal_controls_artifact_store_env(release),
        '/usr/bin/env', f'PYTHONPATH={release / "src"}', 'PYTHONDONTWRITEBYTECODE=1',
        'GIT_CONFIG_COUNT=1', 'GIT_CONFIG_KEY_0=safe.directory', f'GIT_CONFIG_VALUE_0={release}',
        sys.executable, '-m', 'blueprint_pipeline.task_evaluation_terminal_controls_deploy',
        '--config', str(config_path), '--expected-commit', commit]
    # Retained placement validation exceeded ten minutes on the 4-vCPU host.
    # Keep the deploy locks and quiescence while the existing worker finishes.
    try:
        result = subprocess.run(argv, check=True, capture_output=True, text=True, timeout=1800)
    except subprocess.CalledProcessError as exc:
        # The helper's stdout/stderr are captured by systemd-run, not journald.
        # Surface only a bounded typed error; never echo a child traceback or
        # its environment into an operator-door deployment receipt.
        output = (exc.stderr or "") + "\n" + (exc.stdout or "")
        codes = re.findall(r"(?:[A-Za-z0-9_.]+(?:Error|Exception)):\s*([a-z][a-z0-9_]+)", output)
        code = codes[-1] if codes else "unclassified"
        digest = hashlib.sha256(output.encode("utf-8", errors="replace")).hexdigest()
        raise ControlPlaneDeployError(
            f"deploy_terminal_controls_adoptions_failed:{code}:exit_{exc.returncode}:stderr_sha256_{digest}"
        ) from None
    try:
        report = json.loads(result.stdout)
    except ValueError as exc:
        raise ControlPlaneDeployError('deploy_terminal_controls_report_invalid') from exc
    if (report.get('status') != 'prepared' or report.get('source_commit') != commit
            or report.get('provider_mutation_performed') is not False
            or report.get('model_called') is not False or report.get('placement_materialized') is not False):
        raise ControlPlaneDeployError('deploy_terminal_controls_report_invalid')
    return report


def _install_intake_runtime_identity_drop_in(
    drop_in: Path, *, source_repo: Path, source_commit: str
) -> dict[str, Any]:
    """Install a final non-secret env file without opening the credential file.

    The base unit loads ``/etc/blueprint/pipeline-control-plane.env``.  systemd
    gives values loaded from ``EnvironmentFile=`` precedence over values from
    ``Environment=``, even when the latter appears in a later drop-in.  The
    first production version of this deploy guard therefore restarted the
    service while the archived checkout named in the credential env file still
    won.  Load a second, identity-only env file last so it overrides those two
    non-secret identity and import-path keys while leaving every credential in
    the original file alone.
    """

    if drop_in.is_symlink():
        raise ControlPlaneDeployError("deploy_intake_runtime_drop_in_symlink")
    if drop_in.exists() and not stat.S_ISREG(drop_in.stat().st_mode):
        raise ControlPlaneDeployError("deploy_intake_runtime_drop_in_not_regular")
    identity_env = drop_in.with_suffix(".env")
    if identity_env.is_symlink():
        raise ControlPlaneDeployError("deploy_intake_runtime_identity_env_symlink")
    if identity_env.exists() and not stat.S_ISREG(identity_env.stat().st_mode):
        raise ControlPlaneDeployError("deploy_intake_runtime_identity_env_not_regular")
    if any(character.isspace() for character in str(source_repo)):
        raise ControlPlaneDeployError("deploy_intake_source_repo_contains_whitespace")
    if len(source_commit) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ControlPlaneDeployError("deploy_intake_source_commit_invalid")
    env_content = (
        "# Managed by scripts/deploy_control_plane_commit.py.\n"
        "# Contains deployment identity only; no credentials.\n"
        f"BLUEPRINT_PIPELINE_REPO={source_repo}\n"
        f"BLUEPRINT_SOURCE_COMMIT={source_commit}\n"
        # Preserve the virtualenv entrypoint. Resolving this symlink selects
        # the system interpreter and silently drops production dependencies.
        f"BLUEPRINT_PIPELINE_PYTHON={Path(sys.executable).absolute()}\n"
        f"PYTHONPATH={source_repo / 'src'}\n"
        "BLUEPRINT_SCENE_OBJECT_DISCOVERY_QUEUE_ROOT="
        f"{DEFAULT_SCENE_OBJECT_DISCOVERY_QUEUE_ROOT}\n"
    )
    drop_in_content = (
        "# Managed by scripts/deploy_control_plane_commit.py.\n"
        "# Loaded after the base unit credential EnvironmentFile.\n"
        "[Service]\n"
        f"EnvironmentFile={identity_env}\n"
        f"TimeoutStartSec={INTAKE_START_TIMEOUT_SECONDS}s\n"
    )

    def atomic_write(path: Path, content: str) -> None:
        temp_path: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_path, 0o644)
            os.replace(temp_path, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            temp_path = None
        finally:
            if temp_path is not None:
                with contextlib.suppress(OSError):
                    temp_path.unlink()

    try:
        # Publish the referenced file first.  A concurrent daemon-reload can
        # therefore see the old complete pair or the new complete env file,
        # never a drop-in pointing at an absent file.
        atomic_write(identity_env, env_content)
        atomic_write(drop_in, drop_in_content)
    except OSError as exc:
        raise ControlPlaneDeployError("deploy_intake_runtime_drop_in_update_failed") from exc
    return {
        "path": str(drop_in),
        "identity_environment_file": str(identity_env),
        "source_repo": str(source_repo),
        "source_commit": source_commit,
        "pythonpath": str(source_repo / "src"),
        "timeout_start_seconds": INTAKE_START_TIMEOUT_SECONDS,
        "credential_environment_file_opened": False,
        "credential_values_recorded": False,
    }


def _install_scene_configuration_environment(
    path: Path, *, environment: Mapping[str, str]
) -> dict[str, Any]:
    """Atomically install exact-release, non-secret scene runtime bindings."""

    expected_names = {
        "BLUEPRINT_TASK_EVALUATION_SPLAT_RENDER_RUNTIME_ROOT",
        "BLUEPRINT_TASK_EVALUATION_SCENE_CONFIGURATION_TOOLCHAIN_ROOT",
        "BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_RELEASE_WINDOW_PREFIX",
        "BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_DESTINATION_PREFIX",
    }
    allowed_names = expected_names | {"BLUEPRINT_WEBSITE_MAPANYTHING_PROFILE"}
    if (
        not expected_names <= set(environment) <= allowed_names
        or path.is_symlink()
        or (path.exists() and not stat.S_ISREG(path.stat().st_mode))
        or any(
            not value
            or "\n" in value
            or "\r" in value
            or any(character.isspace() for character in value)
            for value in environment.values()
        )
    ):
        raise ControlPlaneDeployError("deploy_scene_configuration_environment_invalid")
    content = (
        "# Managed by scripts/deploy_control_plane_commit.py.\n"
        "# Exact-release paths and public object-store prefixes only; no credentials.\n"
        + "".join(f"{name}={environment[name]}\n" for name in sorted(environment))
    )
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        temporary = None
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ControlPlaneDeployError(
            "deploy_scene_configuration_environment_update_failed"
        ) from exc
    finally:
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink()
    payload = path.read_bytes()
    if payload != content.encode("utf-8"):
        raise ControlPlaneDeployError(
            "deploy_scene_configuration_environment_readback_mismatch"
        )
    return {
        "path": str(path),
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
        "mode": "0644",
        "credential_values_recorded": False,
        "environment_names": sorted(environment),
    }


def _verify_intake_runtime(
    url: str,
    *,
    expected_commit: str,
    attempts: int = 30,
    retry_delay_seconds: float = 1.0,
) -> dict[str, Any]:
    """Require the restarted process—not just its files—to report the SHA."""

    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ControlPlaneDeployError("deploy_intake_version_url_not_loopback_http")
    if attempts < 1:
        raise ControlPlaneDeployError("deploy_intake_version_probe_attempts_invalid")
    payload: Any = None
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=15) as response:  # nosec B310
                payload = json.load(response)
            break
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(retry_delay_seconds)
    else:
        raise ControlPlaneDeployError("deploy_intake_version_probe_failed") from last_error
    if not isinstance(payload, Mapping):
        raise ControlPlaneDeployError("deploy_intake_version_payload_invalid")
    observed = str(payload.get("source_commit") or "")
    if payload.get("commit_proven") is not True or observed != expected_commit:
        raise ControlPlaneDeployError(
            f"deploy_intake_runtime_commit_mismatch:{observed or 'missing'}"
        )
    return {
        "url": url,
        "commit_proven": True,
        "source_commit": observed,
        "service_schema_version": payload.get("service_schema_version"),
    }


def _report_break_glass_notes(
    receipt: dict[str, Any], *, root: str | Path, deploy_commit: str
) -> None:
    """Add unreported break-glass notes to a receipt without marking them.

    Operators and agents changed the host by hand over SSH and nothing
    recorded it; a note now does, and this puts every unreported note in the
    receipt. The CLI marks them only after its receipt is safely written. It
    runs after every surface moved and never fails the deploy, which has
    already happened: an unreadable note stays unreported for the next deploy.
    """

    try:
        notes = unreported_break_glass_notes(root)
    except Exception as exc:
        code = break_glass_refusal_code(exc)
        receipt["break_glass_notes"] = None
        receipt["break_glass_notes_error"] = code
        receipt.setdefault("alerts", []).append(f"break_glass_notes_unreadable:{code}")
        return
    receipt["break_glass_notes"] = [break_glass_note_summary(note) for note in notes]
    if not notes:
        return
    alerts = receipt.setdefault("alerts", [])
    alerts.append(f"break_glass_notes_reported:{len(notes)}")


_SCENE_RUNTIME_BOOT_ROOT = Path("/usr/lib/blueprint/scene-retirement-runtime")
_SCENE_RUNTIME_OWNER = 0
_SCENE_RUNTIME_INSTALL_SECONDS = 900


# The service-owned state directory cannot be an ancestor of root-admitted proof.
_SCENE_SOURCE_ATTESTATIONS = _SCENE_RUNTIME_BOOT_ROOT / "source-attestations"
_SCENE_SOURCE_GH = Path("/usr/bin/gh")


def _scene_source_snappy(raw: bytes, *, deadline: float, cap: int) -> bytes:
    """Bound GitHub's raw Snappy bundle encoding; decoded bytes stay untrusted.

    Format: https://github.com/google/snappy/blob/main/format_description.txt
    No framed streams, allocation from unchecked lengths, or external codec.
    """
    error = "deploy_scene_retirement_runtime_unproven"
    def require(value):
        if not value:
            raise ControlPlaneDeployError(error)
    require(type(raw) is bytes and 0 < len(raw) <= 32*1024*1024 and 0 < cap <= 32*1024*1024)
    declared, cursor = 0, 0
    for shift in range(0,35,7):
        require(cursor < len(raw) and time.monotonic() <= deadline)
        value = raw[cursor]
        cursor += 1
        declared |= (value & 127) << shift
        if value < 128:
            break
    else:
        raise ControlPlaneDeployError(error)
    require(0 < declared <= cap)
    output = bytearray()
    while cursor < len(raw):
        require(time.monotonic() <= deadline)
        tag = raw[cursor]
        cursor += 1
        kind = tag & 3
        if kind == 0:
            length = tag >> 2
            if length >= 60:
                count = length-59
                require(cursor+count <= len(raw))
                length = int.from_bytes(raw[cursor:cursor+count],'little')
                cursor += count
            length += 1
            require(len(output)+length <= declared and cursor+length <= len(raw))
            output.extend(raw[cursor:cursor+length])
            cursor += length
        else:
            count = 1 if kind == 1 else 2 if kind == 2 else 4
            require(cursor+count <= len(raw))
            offset = int.from_bytes(raw[cursor:cursor+count],'little')
            cursor += count
            if kind == 1:
                offset |= (tag & 224) << 3
                length = 4+((tag >> 2) & 7)
            else:
                length = 1+(tag >> 2)
            require(0 < offset <= len(output) and len(output)+length <= declared)
            start = len(output)-offset
            # Snappy copies may overlap. Read only a <=64-byte seed instead
            # of duplicating the entire output for a large back-reference.
            seed = bytes(output[start:start+min(length,offset)])
            output.extend((seed*((length+len(seed)-1)//len(seed)))[:length])
    require(len(output) == declared and time.monotonic() <= deadline)
    return bytes(output)


def _scene_source_selector_bytes(source_commit: str) -> bytes:
    """Derive public lookup data without executing any candidate source."""
    if type(source_commit) is not str or re.fullmatch('[0-9a-f]{40}', source_commit) is None:
        raise ControlPlaneDeployError("deploy_scene_retirement_runtime_unproven")
    return (json.dumps({'schema_version':'blueprint.source_commit_selector.v1',
        'repository':'ognjhunt/BlueprintCapturePipeline', 'source_commit':source_commit},
        sort_keys=True, separators=(',', ':'), ensure_ascii=False)+'\n').encode()


def _scene_source_delivery(source_commit: str, *, deadline: float) -> None:
    """Acquire public proof as untrusted data; this function grants no admission."""
    error = "deploy_scene_retirement_runtime_unproven"
    repository = 'ognjhunt/BlueprintCapturePipeline'
    predicate_type = 'https://github.com/ognjhunt/BlueprintCapturePipeline/attestations/source-sha256-manifest/v1'
    def require(value):
        if not value:
            raise ControlPlaneDeployError(error)
    require(re.fullmatch('[0-9a-f]{40}', source_commit) is not None)
    opener = _source_response_opener(deadline,error=error)
    transfer_remaining = [48*1024*1024]
    decoded_remaining = [48*1024*1024]
    def fetch(url, cap, *, blob=False):
        require(time.monotonic() <= deadline)
        cap = min(cap, transfer_remaining[0])
        require(cap > 0)
        request = urllib.request.Request(url, headers={'Accept':'application/vnd.github+json',
            'X-GitHub-Api-Version':'2022-11-28', 'User-Agent':'Blueprint-source-admission'})
        with opener.open(request, timeout=min(30, max(.001, deadline-time.monotonic()))) as response:
            require(response.status == 200 and response.url == url)
            body = bytearray()
            while True:
                require(time.monotonic() <= deadline)
                # Each blocking read shares the absolute deadline, including a
                # slow peer that repeatedly delivers a tiny successful prefix.
                if response.fp is None:
                    break
                response.fp.raw._sock.settimeout(min(30, max(.001, deadline-time.monotonic())))
                chunk = response.read1(min(65536, cap+1-len(body)))
                if not chunk:
                    break
                require(len(body)+len(chunk) <= cap)
                body.extend(chunk)
                transfer_remaining[0] -= len(chunk)
            require(time.monotonic() <= deadline)
            raw = bytes(body)
            if blob:
                content_type = response.headers.get('Content-Type','').split(';',1)[0].lower()
                if content_type == 'application/x-snappy':
                    raw = _scene_source_snappy(raw,deadline=deadline,cap=min(32*1024*1024,decoded_remaining[0]))
                else:
                    require(content_type in {'application/json','application/vnd.dev.sigstore.bundle.v0.3+json'})
            require(len(raw) <= decoded_remaining[0])
            decoded_remaining[0] -= len(raw)
            return json.loads(raw)
    # GitHub's digest-addressed attestation API retains signed statements
    # independently of Actions artifacts. The known selector is a second
    # subject of the manifest attestation; it supplies no source authority.
    selector_digest = hashlib.sha256(_scene_source_selector_bytes(source_commit)).hexdigest()
    evidence = fetch('https://api.github.com/repos/'+repository+'/attestations/sha256:'+selector_digest+'?per_page=20', 48*1024*1024)
    require(type(evidence) is dict and type(evidence.get('attestations')) is list and 0 < len(evidence['attestations']) <= 20)
    candidates = []
    for item in evidence['attestations']:
        if type(item) is not dict:
            continue
        bundle = item.get('bundle')
        if type(bundle) is not dict:
            # The documented API can omit inline bytes and return a fresh
            # Azure blob URL. Restrict its origin/path before any request;
            # never follow redirects or forward authorization/proxy state.
            url = item.get('bundle_url')
            if type(url) is not str or len(url) > 16384 or any(ord(char) < 33 or ord(char) == 127 for char in url):
                continue
            try:
                location = urllib.parse.urlsplit(url)
            except ValueError:
                continue
            if (location.scheme != 'https' or location.netloc != 'tmaproduction.blob.core.windows.net'
                    or location.fragment or not re.fullmatch(
                        r'/attestations/[0-9]+/[0-9]{4}/[0-9]{2}/[0-9]{2}/[0-9]+\.json\.sn',location.path)):
                continue
            try:
                bundle = fetch(url,32*1024*1024,blob=True)
            except (OSError, ValueError):
                # Do not expose temporary signed query strings in errors.
                require(time.monotonic() <= deadline)
                continue
            if type(bundle) is not dict:
                continue
        envelope = bundle.get('dsseEnvelope')
        if (type(envelope) is not dict or envelope.get('payloadType') != 'application/vnd.in-toto+json'
                or type(envelope.get('payload')) is not str):
            continue
        require(len(envelope['payload']) <= 32*1024*1024)
        payload = base64.b64decode(envelope['payload'], validate=True)
        require(len(payload) <= 24*1024*1024)
        statement = json.loads(payload)
        if (type(statement) is not dict or statement.get('_type') != 'https://in-toto.io/Statement/v1'
                or statement.get('predicateType') != predicate_type or type(statement.get('predicate')) is not dict
                or type(statement.get('subject')) is not list or len(statement['subject']) != 2
                or not all(type(subject) is dict for subject in statement['subject'])):
            continue
        raw = (json.dumps(statement['predicate'], sort_keys=True, separators=(',', ':'), ensure_ascii=False)+'\n').encode()
        require(0 < len(raw) <= 16*1024*1024)
        digest = hashlib.sha256(raw).hexdigest()
        if (sum(subject.get('digest') == {'sha256':digest} for subject in statement['subject']) != 1
                or sum(subject.get('digest') == {'sha256':selector_digest} for subject in statement['subject']) != 1):
            continue
        proof = (json.dumps(bundle, sort_keys=True, separators=(',', ':'))+'\n').encode()
        require(len(proof) <= 32*1024*1024)
        candidates.append((raw, proof))
    require(candidates)
    root = _SCENE_SOURCE_ATTESTATIONS / source_commit
    if all((root/name).exists() for name in ('source-sha256-manifest.json','source-provenance.sigstore.json')):
        _scene_source_attestation(source_commit,deadline=deadline,_proof_root=root)
        return
    selection = _scene_source_selected_proof(root, deadline=deadline)
    selected = None
    # CI reruns at one SHA legitimately produce several signatures over the
    # same deterministic subject. List order is only a bounded trial order;
    # no candidate is admitted until the protected verifier checks its policy.
    for raw, proof in candidates:
        require(time.monotonic() <= deadline)
        if selection is not None and not (hashlib.sha256(proof).hexdigest()+'\n').encode().startswith(selection):
            continue
        candidate = root/'candidates'/hashlib.sha256(proof).hexdigest()
        _scene_source_cache_publish(candidate, raw, proof, deadline=deadline)
        try:
            _scene_source_attestation(source_commit, deadline=deadline, _proof_root=candidate)
        except (ControlPlaneDeployError, OSError, ValueError, subprocess.SubprocessError):
            require(time.monotonic() <= deadline)
            continue
        selected = (raw,proof)
        break
    require(selected is not None)
    _scene_source_cache_publish(root, *selected, deadline=deadline)


def _scene_source_selected_proof(root: Path, *, deadline: float) -> bytes | None:
    """Retain a protected selection across interrupted publication and CI reruns."""
    error = "deploy_scene_retirement_runtime_unproven"
    def require(value):
        if not value:
            raise ControlPlaneDeployError(error)
    require(root.is_absolute() and '..' not in root.parts and time.monotonic() <= deadline)
    held, identities = [], []
    def identity(info):
        return (info.st_dev,info.st_ino,info.st_mode,info.st_uid,info.st_gid,
                info.st_nlink,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
    try:
        directory = os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_CLOEXEC)
        held.append(directory)
        for part in root.parts[1:]:
            before = os.fstat(directory)
            require(stat.S_ISDIR(before.st_mode) and before.st_uid in {0,_SCENE_RUNTIME_OWNER}
                    and not before.st_mode & 0o022)
            identities.append((directory,before))
            try:
                directory = os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=directory)
            except FileNotFoundError:
                return None
            held.append(directory)
        before = os.fstat(directory)
        require(stat.S_ISDIR(before.st_mode) and before.st_uid in {0,_SCENE_RUNTIME_OWNER} and not before.st_mode & 0o022)
        identities.append((directory,before))
        for name, complete in (('.public-selected-proof.sha256',True),('.public-selected-proof.sha256.public-pending',False)):
            try:
                fd = os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=directory)
            except FileNotFoundError:
                continue
            held.append(fd)
            before = os.fstat(fd)
            require(stat.S_ISREG(before.st_mode) and before.st_uid == _SCENE_RUNTIME_OWNER and before.st_nlink == 1
                    and stat.S_IMODE(before.st_mode) in {0o600,0o444} and 0 <= before.st_size <= 65
                    and (not complete or before.st_size == 65))
            raw = os.read(fd,66)
            require(len(raw) == before.st_size and re.fullmatch(rb'[0-9a-f]{0,64}\n?',raw) is not None
                    and (b'\n' not in raw or len(raw) == 65)
                    and identity(before) == identity(os.fstat(fd)) == identity(os.stat(name,dir_fd=directory,follow_symlinks=False)))
            for retained, original in identities:
                require(identity(os.fstat(retained)) == identity(original))
            return raw
        return None
    finally:
        for fd in reversed(held):
            os.close(fd)


def _scene_source_cache_publish(root: Path, raw: bytes, proof: bytes, *, deadline: float) -> None:
    """Publish bounded proof DATA without issuing source or execution authority."""
    error = "deploy_scene_retirement_runtime_unproven"
    def require(value):
        if not value:
            raise ControlPlaneDeployError(error)
    require(root.is_absolute() and '..' not in root.parts and 0 < len(raw) <= 16*1024*1024
            and 0 < len(proof) <= 32*1024*1024 and time.monotonic() <= deadline)
    def identity(info):
        # Reading immutable proof bytes may update atime. Bind all mutation and
        # ownership metadata, including nanosecond mtime/ctime, instead.
        return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
                info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    def directory(path):
        if path != Path('/'):
            directory(path.parent)
        if not path.exists() and not path.is_symlink():
            path.mkdir(mode=0o755)
        info = path.lstat()
        require(stat.S_ISDIR(info.st_mode) and info.st_uid in {0,_SCENE_RUNTIME_OWNER} and not info.st_mode & 0o022)
    directory(root)
    directories = []
    root_fd = os.open('/', os.O_RDONLY|os.O_DIRECTORY|os.O_CLOEXEC)
    directories.append((root_fd,os.fstat(root_fd)))
    lock = None
    try:
        for part in root.parts[1:]:
            parent = root_fd
            before = os.stat(part,dir_fd=parent,follow_symlinks=False)
            require(stat.S_ISDIR(before.st_mode) and before.st_uid in {0,_SCENE_RUNTIME_OWNER} and not before.st_mode & 0o022)
            root_fd = os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=parent)
            directories.append((root_fd,before))
            require(os.fstat(root_fd) == before)
        lock = os.open('.public-source-proof.lock', os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW|os.O_CLOEXEC, 0o600,dir_fd=root_fd)
        try:
            info = os.fstat(lock)
            require(stat.S_ISREG(info.st_mode) and info.st_uid == _SCENE_RUNTIME_OWNER and info.st_nlink == 1
                    and stat.S_IMODE(info.st_mode) == 0o600)
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            # Bind the selected bundle before any fixed manifest/proof bytes.
            # Its protected prefix also pins a retry interrupted in this claim.
            selection = (hashlib.sha256(proof).hexdigest()+'\n').encode()
            for name, raw in zip(('.public-selected-proof.sha256','source-sha256-manifest.json','source-provenance.sigstore.json'), (selection,raw,proof)):
                require(time.monotonic() <= deadline)
                target = root/name
                if target.exists() or target.is_symlink():
                    fd = os.open(target.name, os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=root_fd)
                    try:
                        before = os.fstat(fd)
                        require(stat.S_ISREG(before.st_mode) and before.st_uid == _SCENE_RUNTIME_OWNER
                                and before.st_nlink == 1 and not before.st_mode & 0o022
                                and before.st_size == len(raw) and os.read(fd,len(raw)+1) == raw
                                and identity(before) == identity(os.fstat(fd))
                                == identity(os.stat(target.name,dir_fd=root_fd,follow_symlinks=False)))
                    finally:
                        os.close(fd)
                    continue
                pending = root/(name+'.public-pending')
                flags = os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC
                try:
                    os.stat(pending.name,dir_fd=root_fd,follow_symlinks=False)
                except FileNotFoundError:
                    flags |= os.O_CREAT|os.O_EXCL
                fd = os.open(pending.name, flags, 0o600,dir_fd=root_fd)
                try:
                    before = os.fstat(fd)
                    require(stat.S_ISREG(before.st_mode) and before.st_uid == _SCENE_RUNTIME_OWNER and before.st_nlink == 1
                            and stat.S_IMODE(before.st_mode) in {0o600,0o444} and before.st_size <= len(raw)
                            and (stat.S_IMODE(before.st_mode) != 0o444 or before.st_size == len(raw))
                            and os.read(fd,before.st_size) == raw[:before.st_size])
                    view = memoryview(raw)[before.st_size:]
                    while view:
                        require(time.monotonic() <= deadline)
                        written = os.write(fd, view)
                        require(written > 0)
                        view = view[written:]
                    os.fsync(fd)
                    os.fchmod(fd,0o444)
                    require(os.stat(pending.name,dir_fd=root_fd,follow_symlinks=False).st_ino == before.st_ino)
                    os.link(pending.name,target.name,src_dir_fd=root_fd,dst_dir_fd=root_fd,follow_symlinks=False)
                    os.unlink(pending.name,dir_fd=root_fd)
                finally:
                    os.close(fd)
            os.fsync(root_fd)
            for fd, before in directories:
                current = os.fstat(fd)
                require(stat.S_ISDIR(current.st_mode) and not current.st_mode & 0o022
                        and (current.st_dev,current.st_ino,current.st_mode,current.st_uid,current.st_gid)
                        == (before.st_dev,before.st_ino,before.st_mode,before.st_uid,before.st_gid))
            require(os.fstat(root_fd).st_ino == root.lstat().st_ino)
        finally:
            os.close(lock)
    finally:
        for fd, _ in reversed(directories):
            os.close(fd)


def _scene_source_attestation(source_commit: str, *, deadline: float, _proof_root: Path | None = None) -> tuple[dict[str, Any], bytes, bytes]:
    """Admit signed data with the trusted host tool before executing candidate code."""
    import selectors
    error = "deploy_scene_retirement_runtime_unproven"
    def require(value):
        if not value:
            raise ControlPlaneDeployError(error)
    require(re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None and time.monotonic() <= deadline)
    held, identities = [], []
    verifier_cache = None
    def identity(info):
        return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
                info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    def protected_read(path, cap, *, executable=False):
        require(path.is_absolute() and '..' not in path.parts)
        directory = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        held.append(directory)
        for part in path.parts[1:-1]:
            info = os.fstat(directory)
            require(stat.S_ISDIR(info.st_mode) and info.st_uid in {0, _SCENE_RUNTIME_OWNER} and not info.st_mode & 0o022)
            identities.append((directory, info))
            directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
            held.append(directory)
        info = os.fstat(directory)
        require(stat.S_ISDIR(info.st_mode) and info.st_uid in {0, _SCENE_RUNTIME_OWNER} and not info.st_mode & 0o022)
        identities.append((directory, info))
        before = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        require(stat.S_ISREG(before.st_mode) and before.st_uid == _SCENE_RUNTIME_OWNER
                and not before.st_mode & 0o022 and before.st_nlink >= 1
                and (executable or before.st_nlink == 1) and 0 < before.st_size <= cap
                and (not executable or before.st_mode & 0o111))
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
        held.append(fd)
        require(identity(os.fstat(fd)) == identity(before))
        identities.append((fd, before))
        raw = bytearray()
        if not executable:
            while len(raw) < before.st_size:
                require(time.monotonic() <= deadline)
                block = os.read(fd, min(1024*1024, before.st_size-len(raw)))
                require(block)
                raw.extend(block)
            require(len(raw) == before.st_size and identity(os.fstat(fd)) == identity(before)
                    == identity(os.stat(path.name, dir_fd=directory, follow_symlinks=False)))
        return bytes(raw), fd
    try:
        root = _SCENE_SOURCE_ATTESTATIONS / source_commit if _proof_root is None else _proof_root
        _, gh_fd = protected_read(_SCENE_SOURCE_GH, 256*1024*1024, executable=True)
        if _proof_root is None and not all((root/name).exists() for name in ('source-sha256-manifest.json','source-provenance.sigstore.json')):
            _scene_source_delivery(source_commit, deadline=deadline)
        raw, manifest_fd = protected_read(root / 'source-sha256-manifest.json', 16*1024*1024)
        bundle, bundle_fd = protected_read(root / 'source-provenance.sigstore.json', 32*1024*1024)
        for fd in (manifest_fd, bundle_fd):
            os.lseek(fd, 0, os.SEEK_SET)
        # gh selects the bundle parser by suffix: a bare /proc/self/fd/N is
        # refused before crypto. Give the admitted bytes a .json name through
        # a pinned private directory; retain original proof and copy identities.
        verifier_cache = tempfile.TemporaryDirectory(prefix='blueprint-source-verifier-')
        proof_directory = Path(verifier_cache.name) / 'proof'
        proof_directory.mkdir(mode=0o700)
        proof_directory_fd = os.open(proof_directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        held.append(proof_directory_fd)
        proof_info = os.fstat(proof_directory_fd)
        require(stat.S_ISDIR(proof_info.st_mode) and proof_info.st_uid == _SCENE_RUNTIME_OWNER
                and stat.S_IMODE(proof_info.st_mode) == 0o700)
        proof_copy_fd = os.open('source-provenance.sigstore.json',
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=proof_directory_fd)
        held.append(proof_copy_fd)
        view = memoryview(bundle)
        while view:
            require(time.monotonic() <= deadline)
            written = os.write(proof_copy_fd, view)
            require(written > 0)
            view = view[written:]
        os.fsync(proof_copy_fd)
        os.fchmod(proof_copy_fd, 0o400)
        proof_copy_info = os.fstat(proof_copy_fd)
        require(stat.S_ISREG(proof_copy_info.st_mode) and proof_copy_info.st_uid == _SCENE_RUNTIME_OWNER
                and proof_copy_info.st_nlink == 1 and proof_copy_info.st_size == len(bundle))
        identities.extend(((proof_directory_fd, os.fstat(proof_directory_fd)), (proof_copy_fd, proof_copy_info)))
        command = [f'/proc/self/fd/{gh_fd}', 'attestation', 'verify', f'/proc/self/fd/{manifest_fd}',
                   '--bundle', f'/proc/self/fd/{proof_directory_fd}/source-provenance.sigstore.json',
                   '--repo', 'ognjhunt/BlueprintCapturePipeline', '--hostname', 'github.com',
                   '--cert-identity', 'https://github.com/ognjhunt/BlueprintCapturePipeline/.github/workflows/ci.yml@refs/heads/main',
                   '--cert-oidc-issuer', 'https://token.actions.githubusercontent.com',
                   '--source-ref', 'refs/heads/main', '--source-digest', source_commit,
                   '--signer-digest', source_commit, '--deny-self-hosted-runners',
                   '--digest-alg', 'sha256', '--predicate-type', 'https://github.com/ognjhunt/BlueprintCapturePipeline/attestations/source-sha256-manifest/v1', '--limit', '1', '--format', 'json']
        # Sigstore initializes authenticated trust metadata in a writable cache.
        # Keep it private and disposable; HOME/config remain credential-free.
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, pass_fds=(manifest_fd, bundle_fd, gh_fd, proof_directory_fd), start_new_session=True,
            env={'PATH':'/usr/bin:/bin', 'HOME':'/nonexistent', 'GH_CONFIG_DIR':'/nonexistent', 'GH_HOST':'github.com', 'LC_ALL':'C',
                 'XDG_CACHE_HOME':verifier_cache.name})
        output, errors = bytearray(), bytearray()
        verify_deadline = min(deadline, time.monotonic()+120)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, (output, 64*1024*1024))
                selector.register(process.stderr, selectors.EVENT_READ, (errors, 65536))
                while selector.get_map():
                    require(time.monotonic() <= verify_deadline)
                    for key, _ in selector.select(min(.1, max(0, verify_deadline-time.monotonic()))):
                        target, cap = key.data
                        block = os.read(key.fd, min(65536, cap+1-len(target)))
                        if block:
                            require(len(target)+len(block) <= cap)
                            target.extend(block)
                        else:
                            selector.unregister(key.fileobj)
                require(process.wait(timeout=max(.001, verify_deadline-time.monotonic())) == 0)
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    process.kill()
            process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()
        for fd, original in identities:
            require(identity(os.fstat(fd)) == identity(original))
        proofs = json.loads(output)
        require(type(proofs) is list and len(proofs) == 1 and type(proofs[0]) is dict)
        verified = proofs[0].get('verificationResult', {})
        require(type(verified) is dict)
        signature = verified.get('signature', {})
        require(type(signature) is dict and type(signature.get('certificate')) is dict and bool(signature['certificate'])
                and type(verified.get('verifiedTimestamps')) is list and bool(verified['verifiedTimestamps'])
                and all(type(value) is dict and value for value in verified['verifiedTimestamps']))
        statement = verified.get('statement', {})
        subjects = statement.get('subject') if type(statement) is dict else None
        manifest_digest = {'sha256':hashlib.sha256(raw).hexdigest()}
        selector_digest = {'sha256':hashlib.sha256(_scene_source_selector_bytes(source_commit)).hexdigest()}
        require(type(statement) is dict and statement.get('_type') == 'https://in-toto.io/Statement/v1'
                and statement.get('predicateType') == 'https://github.com/ognjhunt/BlueprintCapturePipeline/attestations/source-sha256-manifest/v1'
                and type(subjects) is list and len(subjects) in {1,2}
                and all(type(subject) is dict for subject in subjects)
                and sum(subject.get('digest') == manifest_digest for subject in subjects) == 1
                and (len(subjects) == 1 or sum(subject.get('digest') == selector_digest for subject in subjects) == 1)
                and (json.dumps(statement.get('predicate'), sort_keys=True, separators=(',', ':'), ensure_ascii=False)+'\n').encode() == raw)
        manifest = json.loads(raw)
        require(type(manifest) is dict and set(manifest) == {'schema_version', 'sources'}
                and manifest.get('schema_version') == 'blueprint.source_sha256_manifest.v1'
                and type(manifest.get('sources')) is list and len(manifest['sources']) == 2
                and raw == (json.dumps(manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=False)+'\n').encode())
        roots = (('src/blueprint_pipeline', 'scripts', 'deploy/systemd', 'uv.lock', 'pyproject.toml'),
                 ('src/blueprint_contracts', 'blueprint_contracts'))
        repositories = ('ognjhunt/BlueprintCapturePipeline', 'ognjhunt/BlueprintContracts')
        commits = (source_commit, '7708a4e4c5dedeeb39cc73d3f6869304de295b81')
        count = total = 0
        for source, repository, commit, wanted in zip(manifest['sources'], repositories, commits, roots):
            require(type(source) is dict and set(source) == {'repository','commit','tree','roots','files'}
                    and source['repository'] == repository and source['commit'] == commit
                    and type(source['tree']) is str and re.fullmatch('[0-9a-f]{40}', source['tree'])
                    and source['roots'] == list(wanted) and type(source['files']) is list and bool(source['files']))
            previous, present = '', set()
            for row in source['files']:
                require(type(row) is dict and set(row) == {'path','git_blob_oid','mode','size','sha256'})
                path = row['path']
                require(type(path) is str and path > previous and len(path.encode()) <= 4096
                        and not path.startswith('/') and '\\' not in path and len(path.split('/')) <= 32
                        and all(part not in {'','.','..'} for part in path.split('/'))
                        and not any(ord(char) < 32 or ord(char) == 127 for char in path)
                        and any(path == root or path.startswith(root+'/') for root in wanted)
                        and row['mode'] in {'100644','100755'} and type(row['git_blob_oid']) is str
                        and re.fullmatch('[0-9a-f]{40}',row['git_blob_oid']) and type(row['sha256']) is str
                        and re.fullmatch('[0-9a-f]{64}',row['sha256']) and type(row['size']) is int
                        and 0 <= row['size'] <= (16*1024*1024 if path == 'uv.lock' else 1024*1024))
                previous = path
                present.add(path)
                count += 1
                total += row['size']
                require(count <= 32768 and total <= 4*1024**3)
            if repository == repositories[0]:
                require(all(any(path == root or path.startswith(root+'/') for path in present) for root in wanted)
                        and {'scripts/install_scene_retirement_runtime.py', 'scripts/deploy_control_plane_commit.py',
                             'scripts/release_source_manifest.py'} <= present)
            else:
                require(any(root+'/__init__.py' in present for root in wanted))
        pipeline = manifest['sources'][0]
        require(pipeline.get('repository') == 'ognjhunt/BlueprintCapturePipeline' and pipeline.get('commit') == source_commit
                and type(pipeline.get('files')) is list and 0 < len(pipeline['files']) <= 32768)
        return manifest, raw, bundle
    finally:
        for fd in reversed(held):
            os.close(fd)
        if verifier_cache is not None:
            verifier_cache.cleanup()


def _bootstrap_scene_retirement_installer(source_repo: Path, source_commit: str, *, deadline: float,
                                         destination: Path | None = None) -> dict[str, Path]:
    """Authenticate installer Git data before the first privileged execution."""
    import fcntl
    import selectors
    error = "deploy_scene_retirement_runtime_unproven"
    shared_deadline = deadline
    def require(value):
        if not value:
            raise ControlPlaneDeployError(error)
    require(re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None)
    require(source_repo.is_absolute() and ".." not in source_repo.parts and not source_repo.is_symlink())
    def object_bytes(digest, cap):
        deadline = min(shared_deadline, time.monotonic()+30)
        command = ["/usr/bin/git", "--no-replace-objects", "-C", str(source_repo),
                   "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
                   "-c", "safe.directory=" + str(source_repo), "cat-file", "blob", digest]
        environment = {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "GIT_CONFIG_NOSYSTEM": "1",
                       "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_NO_LAZY_FETCH": "1", "GIT_ALLOW_PROTOCOL": "",
                       "GIT_PROTOCOL_FROM_USER": "0", "GIT_TERMINAL_PROMPT": "0"}
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env=environment)
        output, errors = bytearray(), bytearray()
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, (output, cap))
                selector.register(process.stderr, selectors.EVENT_READ, (errors, 4096))
                while selector.get_map():
                    require(time.monotonic() <= deadline)
                    for key, _ in selector.select(min(.1, max(0, deadline-time.monotonic()))):
                        buffer, limit = key.data
                        raw = os.read(key.fd, min(65536, limit+1-len(buffer)))
                        if not raw:
                            selector.unregister(key.fileobj)
                        else:
                            require(len(buffer)+len(raw) <= limit)
                            buffer.extend(raw)
            require(process.wait(timeout=max(.001, deadline-time.monotonic())) == 0)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()
            process.stderr.close()
        raw = bytes(output)
        return raw
    manifest, manifest_raw, bundle_raw = _scene_source_attestation(source_commit, deadline=shared_deadline)
    files = manifest['sources'][0]['files']
    names = ['scripts/install_scene_retirement_runtime.py', 'scripts/release_source_manifest.py']
    admitted = {}
    for name in names:
        matches = [row for row in files if type(row) is dict and row.get('path') == name]
        require(len(matches) == 1)
        row = matches[0]
        require(set(row) == {'path', 'git_blob_oid', 'mode', 'size', 'sha256'}
                and row['mode'] in {'100644', '100755'} and type(row['size']) is int and 0 < row['size'] <= 1024*1024
                and type(row['git_blob_oid']) is str and re.fullmatch('[0-9a-f]{40}', row['git_blob_oid'])
                and type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}', row['sha256']))
        raw = object_bytes(row['git_blob_oid'], row['size'])
        require(len(raw) == row['size'] and hashlib.sha256(raw).hexdigest() == row['sha256'])
        admitted[name] = raw
    body = admitted[names[0]]
    root = destination if destination is not None else _SCENE_RUNTIME_BOOT_ROOT
    def protected_directory(path):
        if not path.exists() and not path.is_symlink():
            protected_directory(path.parent)
            path.mkdir(mode=0o755)
        info = path.lstat()
        require(stat.S_ISDIR(info.st_mode) and info.st_uid in {0,_SCENE_RUNTIME_OWNER} and not info.st_mode & 0o022)
        if path != Path('/'):
            protected_directory(path.parent)
    protected_directory(root)
    lock = os.open(root / ".installer-bootstrap.lock",os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW|os.O_CLOEXEC,0o600)
    try:
        info = os.fstat(lock)
        require(stat.S_ISREG(info.st_mode) and info.st_uid == _SCENE_RUNTIME_OWNER and info.st_nlink == 1
                and stat.S_IMODE(info.st_mode) == 0o600)
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        target = root / "runtime_installer.py"
        value = json.dumps({"schema":"scene-retirement-runtime-installer.v1",
                 "sha256":"sha256:"+hashlib.sha256(body).hexdigest(),"size_bytes":len(body)},
                 sort_keys=True,separators=(",", ":" )).encode()
        def publish(path, raw):
            if path.exists() or path.is_symlink():
                check = os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
                try:
                    old = os.fstat(check)
                    require(stat.S_ISREG(old.st_mode) and old.st_uid == _SCENE_RUNTIME_OWNER
                            and old.st_nlink == 1 and not old.st_mode & 0o022
                            and old.st_size == len(raw) and os.read(check,len(raw)+1) == raw)
                finally:
                    os.close(check)
                return
            partial = path.with_name(path.name+".bootstrap-pending")
            flags = os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC
            if not partial.exists() and not partial.is_symlink():
                flags |= os.O_CREAT|os.O_EXCL
            fd = os.open(partial,flags,0o600)
            try:
                original = os.fstat(fd)
                require(stat.S_ISREG(original.st_mode) and original.st_uid == _SCENE_RUNTIME_OWNER
                        and original.st_nlink == 1 and stat.S_IMODE(original.st_mode) in {0o600,0o644}
                        and original.st_size <= len(raw) and os.read(fd,original.st_size) == raw[:original.st_size])
                view = memoryview(raw)[original.st_size:]
                while view:
                    require(time.monotonic() <= deadline)
                    written = os.write(fd,view)
                    require(written>0)
                    view = view[written:]
                os.fsync(fd)
                os.fchmod(fd,0o644)
                require(partial.lstat().st_ino == original.st_ino)
                os.link(partial,path,follow_symlinks=False)
                os.unlink(partial)
                parent = os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
                try:
                    os.fsync(parent)
                finally:
                    os.close(parent)
            finally:
                os.close(fd)
        pending = root / "runtime-installer-pending.json"
        if target.exists() or target.is_symlink():
            require(pending.exists())
            publish(pending,value)
            publish(target,body)
        else:
            publish(pending,value)
            publish(target,body)
        publish(root/"runtime-installer.json",value)
        verifier = admitted[names[1]]
        publish(root/'release_source_manifest.py', verifier)
        publish(root/'source-manifest-verifier.json', json.dumps({
            'schema':'scene-retirement-source-manifest-verifier.v1', 'sha256':hashlib.sha256(verifier).hexdigest(),
            'size':len(verifier)}, sort_keys=True, separators=(',', ':')).encode())
        publish(root/'source-sha256-manifest.json', manifest_raw)
        publish(root/'source-provenance.sigstore.json', bundle_raw)
        return {'source_manifest':root/'source-sha256-manifest.json',
                'source_attestation':root/'source-provenance.sigstore.json', 'manifest_verifier':root/'release_source_manifest.py'}
    finally:
        os.close(lock)


def _scene_runtime_diagnostic(stderr: bytes | str | None, *, phase: str, reason: str) -> dict[str, str]:
    """Admit only bounded fixed markers; child output is never forwarded."""
    phases = {"source_attestation", "signed_release", "build_sdk", "resume_initial_intent", "refresh", "prepare", "publish_installer"}
    reasons = {"deadline", "validation", "io", "unexpected"}
    if isinstance(stderr, (bytes, str)) and len(stderr) <= 65536:
        text = stderr.decode("ascii", errors="replace") if isinstance(stderr, bytes) else stderr
        for line in text.splitlines():
            if line.startswith("scene_retirement_runtime_phase:") and line.split(":", 1)[1] in phases:
                phase = line.split(":", 1)[1]
            if line.startswith("scene_retirement_runtime_failure:") and line.split(":", 1)[1] in reasons:
                reason = line.split(":", 1)[1]
    return {"phase": phase, "reason": reason}


def _prepare_scene_retirement_runtime(*, source_repo: Path, source_commit: str, _deadline: float | None = None) -> dict[str, Any]:
    """Authenticate the selected release installer before exposing new units."""
    # One installation-only origin covers authentication, SDK and durable copy.
    started = time.monotonic()
    if _deadline is not None and (type(_deadline) is not float or not math.isfinite(_deadline)):
        raise ControlPlaneDeployError("deploy_scene_retirement_runtime_unproven")
    deadline = started + _SCENE_RUNTIME_INSTALL_SECONDS if _deadline is None else min(started + _SCENE_RUNTIME_INSTALL_SECONDS, _deadline)
    if started > deadline:
        raise ControlPlaneDeployError("deploy_scene_retirement_runtime_unproven")
    root = _SCENE_RUNTIME_BOOT_ROOT
    helper = root / "runtime_installer.py"
    error = "deploy_scene_retirement_runtime_unproven"
    phase = "retained_installer"
    held: list[int] = []
    def identity(info: os.stat_result) -> tuple[int, ...]:
        return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    def read(path: Path, cap: int) -> tuple[bytes, int]:
        for ancestor in (*reversed(path.parent.parents), path.parent):
            info = ancestor.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid not in {0, _SCENE_RUNTIME_OWNER} or info.st_mode & 0o022:
                raise ControlPlaneDeployError(error)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        held.append(fd)
        first = os.fstat(fd)
        if (not stat.S_ISREG(first.st_mode) or first.st_uid != _SCENE_RUNTIME_OWNER or first.st_mode & 0o022
                or first.st_nlink != 1 or first.st_size > cap):
            raise ControlPlaneDeployError(error)
        raw = os.read(fd, cap + 1)
        if len(raw) != first.st_size or identity(os.fstat(fd)) != identity(first) or identity(path.lstat()) != identity(first):
            raise ControlPlaneDeployError(error)
        return raw, fd
    def verified_installer(directory: Path) -> int:
        raw, fd = read(directory / "runtime_installer.py", 1024 * 1024)
        selected = {"sha256": "sha256:" + hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
        matched = False
        for name in ("runtime-installer.json", "runtime-installer-pending.json"):
            path = directory / name
            if path.exists() or path.is_symlink():
                body, _ = read(path, 4096)
                value = json.loads(body)
                if (type(value) is dict and set(value) == {"schema", "sha256", "size_bytes"}
                        and value["schema"] == "scene-retirement-runtime-installer.v1"
                        and {key: value[key] for key in selected} == selected):
                    matched = True
        if not matched or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
            raise ControlPlaneDeployError(error)
        return fd
    try:
        # Unknown retained state still refuses before staging or execution. A
        # valid old helper must not prevent a reviewed installer repair from
        # running: stage authenticated Git bytes separately and preserve it.
        retained = (helper, root / "runtime-installer.json", root / "runtime-installer-pending.json")
        if any(path.exists() or path.is_symlink() for path in retained):
            verified_installer(root)
        candidate = root / "installers" / source_commit
        phase = "authenticate_installer"
        attestation = _bootstrap_scene_retirement_installer(
            source_repo, source_commit, deadline=deadline, destination=candidate,
        )
        fd = verified_installer(candidate)
        os.lseek(fd, 0, os.SEEK_SET)
        command = ["/usr/bin/python3", "-I", "-S", f"/proc/self/fd/{fd}",
                   "--source", str(source_repo), "--source-commit", source_commit, "--locked-sdk",
                   "--deadline-monotonic", str(deadline),
                   "--source-manifest", str(attestation['source_manifest']),
                   "--source-attestation", str(attestation['source_attestation']),
                   "--manifest-verifier", str(attestation['manifest_verifier'])]
        phase = "execute_installer"
        result = subprocess.run(command, pass_fds=(fd,), stdin=subprocess.DEVNULL,
                                capture_output=True, timeout=max(.001, deadline-time.monotonic()), check=False,
                                env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "LC_ALL": "C"})
        if result.returncode != 0 or len(result.stdout) > 65536 or len(result.stderr) > 65536:
            failure = ControlPlaneDeployError(error)
            setattr(failure, "runtime_diagnostic", _scene_runtime_diagnostic(result.stderr, phase=phase, reason="validation"))
            raise failure
        phase = "verify_result"
        value = json.loads(result.stdout)
        if (type(value) is not dict or value.get("status") not in {"prepared", "refreshed"}
                or value.get("source_commit") != source_commit
                or value.get("authority_issued") is not False or value.get("cleanup_enabled") is not False):
            raise ControlPlaneDeployError(error)
        return value
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        failure = ControlPlaneDeployError(error)
        diagnostic = getattr(exc, "runtime_diagnostic", None)
        if diagnostic is None:
            timed_out = isinstance(exc, subprocess.TimeoutExpired)
            diagnostic = _scene_runtime_diagnostic(
                exc.stderr if isinstance(exc, subprocess.TimeoutExpired) else None, phase=phase,
                reason="deadline" if timed_out else "io" if isinstance(exc, OSError) else "validation",
            )
            if timed_out:
                diagnostic["reason"] = "deadline"
        setattr(failure, "runtime_diagnostic", diagnostic)
        raise failure from exc
    finally:
        for fd in reversed(held):
            os.close(fd)


def deploy_control_plane_commit(
    *,
    source_repo: str | Path,
    source_commit: str,
    release_root: str | Path,
    state_root: str | Path,
    active_link: str | Path,
    release_provenance: str | Path | None = None,
    iteration: bool = False,
    canary: bool = False,
    restart_units: Sequence[str] = DEFAULT_RESTART_UNITS,
    paid_launch_locks: Sequence[str] = DEFAULT_PAID_LAUNCH_LOCKS,
    intake_runtime_drop_in: str | Path = DEFAULT_INTAKE_RUNTIME_DROP_IN,
    intake_version_url: str = DEFAULT_INTAKE_VERSION_URL,
    systemd_dir: str | Path = DEFAULT_SYSTEMD_DIR,
    scene_configuration_environment_file: str | Path = (
        DEFAULT_SCENE_CONFIGURATION_ENVIRONMENT_FILE
    ),
    scene_configuration_runtime_root: str | Path = (
        DEFAULT_SCENE_CONFIGURATION_RUNTIME_ROOT
    ),
    splat_render_prerequisite_root: str | Path = (
        DEFAULT_SPLAT_RENDER_PREREQUISITE_ROOT
    ),
    artifixer_source_root: str | Path = DEFAULT_ARTIFIXER_SOURCE_ROOT,
    content_agents_source_root: str | Path = DEFAULT_CONTENT_AGENTS_SOURCE_ROOT,
    cad_skill_source_root: str | Path = DEFAULT_CAD_SKILL_SOURCE_ROOT,
    astra_blender_archive_path: str | Path | None = BLENDER_INSTALL_ROOT.parent / 'archives' / BLENDER_ARCHIVE_NAME,
    configured_controls_autostart_intent_root: str | Path = (
        DEFAULT_CONFIGURED_CONTROLS_AUTOSTART_INTENT_ROOT
    ),
    configured_controls_autostart_intent_sources: Sequence[str] = (),
    scene_preparation_bootstrap_file: str | Path = "/etc/blueprint/task-evaluation-scene-preparation-bootstrap.json",
    controls_autoprovision_bootstrap_file: str | Path = "/etc/blueprint/task-evaluation-controls-autoprovision-bootstrap.json",
    arm_path_units: bool = False,
    preserve_configured_controls_state: bool = False,
    disk_reservation_root: str | Path | None = None,
    release_protection_sources: ProtectionSources = DEFAULT_RELEASE_PROTECTION_SOURCES,
    release_retirement_keep_last: int = DEFAULT_RELEASE_RETIREMENT_KEEP_LAST,
    break_glass_notes_root: str | Path | None = None,
    door_holds_dir: str | Path = DEFAULT_DOOR_HOLDS_DIR,
) -> dict[str, Any]:
    """Move the mutable clone and the release link, then verify both.

    With ``break_glass_notes_root`` (the CLI always passes it), the receipt
    also reports every break-glass note no earlier deploy reported.

    Deployment never retires existing release/runtime trees, including
    interrupted ``.retiring`` trees. Retirement is a separate explicit action;
    its protection/keep-last arguments remain accepted for caller compatibility.
    """

    global _DEPLOY_ACTIVE_TRANSITION
    _DEPLOY_ACTIVE_TRANSITION = False

    if preserve_configured_controls_state and arm_path_units:
        raise ControlPlaneDeployError("deploy_conflicting_configured_controls_intent")
    source = Path(source_repo).expanduser().resolve()
    active = Path(active_link).expanduser()
    releases = Path(release_root).expanduser().resolve()
    raw_state = Path(state_root).expanduser()
    if (
        len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ControlPlaneDeployError("deploy_source_commit_invalid")
    if not raw_state.is_absolute():
        raise ControlPlaneDeployError("deploy_state_root_must_be_absolute")
    state = raw_state.resolve()
    if (
        state == source
        or state in source.parents
        or source in state.parents
        or state == releases
        or state in releases.parents
    ):
        raise ControlPlaneDeployError("deploy_state_root_overlaps_checkout")
    if canary and not iteration:
        raise ControlPlaneDeployError("deploy_canary_requires_iteration")
    if iteration:
        # An iteration deploy trades promotion evidence for cycle time. The
        # full lane takes ~15 minutes; a fix-and-fire loop that waits for it
        # costs ~18 minutes per attempt, which dominates a campaign of dozens
        # of GPU runs. What must NOT be traded away is knowing exactly which
        # bytes ran: the release is still built from a real pushed commit, so
        # the running code and main can never silently diverge.
        #
        # The receipt says plainly that no lane verified it.
        # `_commit_has_verified_production_promotion` requires
        # status == "verified", so this can never be mistaken for a promoted
        # release, and paid admission still refuses it as an ancestor.
        if release_provenance is not None:
            raise ControlPlaneDeployError("deploy_iteration_provenance_conflict")
        # The guard belongs here, not in a wrapper script. Within an hour of
        # the wrapper being written its `git fetch` hit a permission error and
        # the obvious workaround was to call this tool directly -- which then
        # had no ancestry check at all. That is how a guard dies: the wrapper
        # is inconvenient once and the bypass becomes the habit.
        #
        # Iteration exists to skip the LANE, never to skip main. A commit that
        # is not an ancestor of origin/main is exactly the local-only drift
        # this mode replaces, so refuse it however this tool was invoked.
        # Fail closed: a stale or missing origin/main refuses rather than
        # admits, and the operator fetches and retries.
        if canary:
            # A canary deploy exists for the fix-and-fire loop on a lane that
            # is already development_only: waiting for a merge (review, CI,
            # rebase churn against a fast-moving main) costs 5-15 minutes per
            # attempt and buys nothing the canary evidence can use. What is
            # still NOT traded away is immutability: the commit must be
            # reachable from some pushed origin ref, so the running bytes are
            # publicly recorded and can never silently diverge from the
            # repository. Local-only or dirty-tree commits still refuse.
            code, reachable = _git(
                source,
                "branch",
                "--remotes",
                "--contains",
                source_commit,
            )
            if code != 0 or not reachable.strip():
                raise ControlPlaneDeployError(
                    "deploy_canary_commit_not_pushed_to_origin"
                )
        else:
            code, _ = _git(source, "merge-base", "--is-ancestor", source_commit, "origin/main")
            if code != 0:
                raise ControlPlaneDeployError("deploy_iteration_commit_not_on_origin_main")
        provenance_receipt = {
            "schema_version": "blueprint.deploy_release_provenance.v1",
            "status": "canary" if canary else "iteration",
            "git_sha": source_commit,
            "promotion_eligible": False,
            "claim_boundary": {
                "canonical_full_lane_verified": False,
                "promotion_eligible": False,
                "evidence_grade": "development_only",
            },
        }
        provenance_payload = (
            json.dumps(provenance_receipt, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    else:
        if release_provenance is None:
            raise ControlPlaneDeployError("deploy_release_provenance_missing")
        _provenance_source, provenance_payload, provenance_receipt = (
            _validated_release_provenance(
                release_provenance, source_commit=source_commit
            )
        )
        provenance_receipt = dict(provenance_receipt)
        provenance_receipt.setdefault("promotion_eligible", True)

    disk_reservation = None
    disk_reservation_runtime = None
    disk_reservation_estimate = None
    if disk_reservation_root is not None:
        disk_reservation_runtime = _install_disk_reservation_runtime_prerequisites(
            disk_reservation_root
        )
        try:
            # Reserve what this release costs to stage, not a flat 2 GiB: on
            # 2026-09-26 the flat figure refused a deploy the disk could hold.
            disk_reservation_estimate = _release_footprint_estimate(
                source, source_commit, reservation_root=disk_reservation_root
            )
            disk_reservation = reserve_control_plane_disk(
                "control_plane_deploy",
                target_root=releases,
                expected_bytes=disk_reservation_estimate["bytes"],
                reservation_root=disk_reservation_root,
                workload="control_plane_release",
            )
        except ControlPlaneDiskBudgetError as exc:
            raise ControlPlaneDeployError(
                f"deploy_disk_budget_exceeded:{exc}"
            ) from exc
    disk_reservation_receipt = (
        disk_reservation.receipt() if disk_reservation is not None else None
    )
    automation_unit_names = [
        {"unit": unit}
        for unit in DEFAULT_DEPLOYED_SYSTEMD_UNITS
        if unit.endswith((".path", ".timer"))
    ]
    # Where a deploy's minutes go is otherwise invisible; the receipt records
    # each stage so a slow deploy is a measurement, not a feeling.
    stage_timings: dict[str, float] = {}
    stage_clock = [time.monotonic()]

    def _mark_stage(name: str) -> None:
        now = time.monotonic()
        stage_timings[name] = round(now - stage_clock[0], 3)
        stage_clock[0] = now

    # Held for the whole deploy, not sampled before it: a launch that starts
    # mid-deploy would read a release being swapped underneath it.  The nested
    # guard restores exact watcher intent before the paid locks are released if
    # any later deploy step fails.
    with (
        disk_reservation or contextlib.nullcontext(),
        _holding_paid_launch_gate(paid_launch_locks) as paid_runs_in_flight,
        _restore_path_unit_states_on_deploy_failure(
            automation_unit_names, door_holds_dir=door_holds_dir,
        ) as (
            automation_unit_states_before,
            quiesced_automation_units,
        ),
    ):
        if Path(controls_autoprovision_bootstrap_file).expanduser().exists():
            _require_terminal_controls_quiescence(wait_seconds=600)
        installed_provenance = _install_release_provenance(
            payload=provenance_payload,
            state_root=state,
            source_commit=source_commit,
            receipt=provenance_receipt,
        )
        staged_release = stage_task_evaluation_control_plane_release(
            source_repo=source,
            source_commit=source_commit,
            release_root=release_root,
            state_root=state_root,
            active_link=active,
            activate=False,
            allow_unmerged_remote_commit=canary,
        )
        scene_retirement_runtime = _prepare_scene_retirement_runtime(
            source_repo=source, source_commit=source_commit
        )
        scene_retirement_stores = _install_scene_retirement_stores()
        _mark_stage("release_staged")
        # Record what this deploy really stages (the release checkout it created
        # and the runtime trees it provisions) as the deploy role's footprint.
        runtime_trees_root = Path(scene_configuration_runtime_root).expanduser()
        runtime_trees_before: set[Path] = set()
        if disk_reservation is not None:
            runtime_trees_before = _release_runtime_trees(runtime_trees_root, source_commit)
        # Every path the new units' sandboxes name must exist before the
        # release link moves, or the first worker to start after the switch
        # dies on mount setup and the deploy still reports success.
        unit_sandbox_paths = _install_unit_sandbox_paths(
            release_path=staged_release["release_path"]
        )
        _mark_stage("unit_sandbox_paths")
        try:
            cad_skill_sources = provision_production_cad_skill_sources(
                cad_skill_source_root
            )
            cad_sources_by_id = {
                str(row["id"]): str(row["path"])
                for row in cad_skill_sources["sources"]
            }
            prerequisite = validate_splat_render_prerequisites(
                root=splat_render_prerequisite_root,
                repository_root=staged_release["release_path"],
            )
            prerequisite_entrypoints = prerequisite["entrypoints"]
            scene_configuration_runtime = _provision_scene_configuration_from_release(
                repository_root=staged_release["release_path"],
                source_commit=source_commit,
                runtime_root=scene_configuration_runtime_root,
                node_executable=prerequisite_entrypoints["node"],
                browser_root=prerequisite_entrypoints["browser_root"],
                browser_executable=prerequisite_entrypoints["browser"],
                node_modules_root=prerequisite_entrypoints["node_modules"],
                artifixer_root=artifixer_source_root,
                content_agents_root=content_agents_source_root,
                text_to_cad_root=cad_sources_by_id["text-to-cad"],
                multi_agent_cad_root=cad_sources_by_id["multi-agent-cad"],
                astra_blender_archive_path=astra_blender_archive_path,
                readback_user=DEFAULT_SERVICE_ACCOUNT,
            )
        except (ValueError, ProductionCadSkillSourcesError) as exc:
            raise ControlPlaneDeployError(
                f"deploy_scene_configuration_runtime_invalid:{exc}"
            ) from exc
        scene_configuration_environment = _install_scene_configuration_environment(
            Path(scene_configuration_environment_file).expanduser(),
            environment=scene_configuration_runtime["environment"],
        )
        bootstrap = Path(scene_preparation_bootstrap_file).expanduser()
        if bootstrap.exists():
            from blueprint_pipeline.task_evaluation_scene_preparation_installation import install_scene_preparation
            scene_preparation_installation = install_scene_preparation(bootstrap_path=bootstrap)
        else:
            scene_preparation_installation = {"status": "not_configured", "bootstrap_path": str(bootstrap),
                                              "provider_mutation_performed": False}
        if disk_reservation is not None:
            created_usage = _created_release_usage(
                created_release_checkout=bool(staged_release.get("created_release_checkout")),
                release_path=staged_release["release_path"],
                runtime_root=runtime_trees_root,
                commit=source_commit,
                runtime_trees_before=runtime_trees_before,
            )
            _observe_created_release_usage(disk_reservation, created_usage)
        _mark_stage("runtime_trees_provisioned")
        agent_pre_activation_drain = _drain_agent_execution_before_release_switch(
            expected_commit=source_commit
        )
        _move_source_checkout(source, source_commit)
        # A termination here must not strand the new active link with old
        # services. The CLI defers SIGTERM until the release is proven live.
        _DEPLOY_ACTIVE_TRANSITION = True
        release = stage_task_evaluation_control_plane_release(
            source_repo=source,
            source_commit=source_commit,
            release_root=release_root,
            state_root=state_root,
            active_link=active,
            activate=True,
            allow_unmerged_remote_commit=canary,
        )
        commit = str(release["source_commit"])
        _mark_stage("release_activated")

        surfaces = {
            "source_checkout": source,
            "active_release": active.resolve(),
        }
        observed: dict[str, str] = {}
        for name, path in surfaces.items():
            observed[name] = _surface_commit(path, name=name)
        disagreeing = sorted(
            name for name, head in observed.items() if head != commit
        )
        if disagreeing:
            # The whole point. One surface moving is not a deploy.
            raise ControlPlaneDeployError(
                "deploy_surfaces_disagree:" + ",".join(disagreeing)
            )

        installed_systemd_units = _install_release_systemd_units(
            release_path=release["release_path"],
            systemd_dir=systemd_dir,
        )
        scene_object_discovery_runtime_directories = (
            _install_scene_object_discovery_runtime_directories()
        )
        episode_compilation_runtime_directories = (
            _install_episode_compilation_runtime_directories()
        )
        storage_pins_runtime = _install_storage_pins_runtime_root()
        configured_controls_runtime = (
            _install_configured_controls_runtime_prerequisites()
        )
        configured_controls_autostart_registry = (
            _install_configured_controls_autostart_registry(
                intent_root=str(configured_controls_autostart_intent_root),
                intent_sources=configured_controls_autostart_intent_sources,
                source_commit=commit,
            )
        )
        # Controls-autoprovision config + sealed content catalog + env pointer
        # (Spec B). Runs AFTER the autostart registry so the group-writable
        # registry permission wins, and BEFORE the unit restart so the config the
        # env pointer names already exists when the worker reads it. The bootstrap
        # is operator-owned; without it the worker stays in the legacy lane and
        # BLUEPRINT_TASK_EVALUATION_CONTROLS_AUTOPROVISION_CONFIG stays unset, so
        # progression is never pointed at a missing config (which fails closed).
        controls_autoprovision_bootstrap = Path(
            controls_autoprovision_bootstrap_file
        ).expanduser()
        if controls_autoprovision_bootstrap.exists():
            from blueprint_pipeline.task_evaluation_controls_autoprovision_installation import (
                install_controls_autoprovision,
            )

            controls_autoprovision_installation = install_controls_autoprovision(
                bootstrap_path=controls_autoprovision_bootstrap
            )
            terminal_controls_adoptions = _prepare_terminal_controls_adoptions(
                release_path=staged_release['release_path'], commit=commit,
                config_path=controls_autoprovision_installation['config']['path'],
            )
        else:
            controls_autoprovision_installation = {
                "status": "not_configured",
                "bootstrap_path": str(controls_autoprovision_bootstrap),
                "provider_mutation_performed": False,
            }
            terminal_controls_adoptions = {'status': 'not_configured', 'provider_mutation_performed': False}

        runtime_binding = _install_intake_runtime_identity_drop_in(
            Path(intake_runtime_drop_in).expanduser(),
            source_repo=source,
            source_commit=commit,
        )
        # Before the restart, not after: the runtime guard blocks a unit from
        # starting on a lock slot the service account cannot use, so a host
        # carrying a root-created slot would fail its own restart here.
        service_ids = _service_account_ids(DEFAULT_SERVICE_ACCOUNT)
        if service_ids is None:
            lock_repair: dict[str, Any] = {
                "status": "not_applicable_no_service_account",
                "account": DEFAULT_SERVICE_ACCOUNT,
                "repaired_slots": [],
            }
        else:
            lock_repair = _repair_paid_launch_lock_slots(
                paid_launch_locks, owner_uid=service_ids[0], owner_gid=service_ids[1]
            )
            lock_repair["status"] = "repaired"
            lock_repair["account"] = DEFAULT_SERVICE_ACCOUNT
        _mark_stage("units_and_directories_installed")
        _require_in_flight_runs_outside_units(
            paid_runs_in_flight, _required_restart_units(restart_units)
        )
        restarted = _restart_units(_required_restart_units(restart_units))
        runtime = _verify_intake_runtime(
            intake_version_url, expected_commit=commit
        )
        _mark_stage("intake_restarted_and_proven")
        agent_execution = _activate_agent_execution(expected_commit=commit)
        agent_execution["pre_activation_drain"] = agent_pre_activation_drain
        # Last inside the held locks: the queue watcher only starts watching
        # once the restarted intake has proven the new commit, and no launch
        # can slip in between the watcher restart and the lock release.
        with _locked_door_holds(door_holds_dir) as (held_units, door_holds_warning):
            automation_unit_state_receipts = _restore_installed_path_units(
                installed_systemd_units,
                before=automation_unit_states_before,
                arm_path_units=arm_path_units,
                always_arm_units=DEFAULT_ALWAYS_ARM_PATH_UNITS,
                always_arm_authority_gated_units=(
                    DEFAULT_ALWAYS_ARM_AUTHORITY_GATED_PATH_UNITS
                ),
                always_arm_timer_units=DEFAULT_ALWAYS_ARM_TIMER_UNITS,
                preserve_configured_controls_state=preserve_configured_controls_state,
                held_units=held_units,
                defer_start_verification=True,
            )
        verification_warning = _verify_deferred_path_unit_starts(
            automation_unit_state_receipts, door_holds_dir=door_holds_dir,
        )
        door_holds_warning = door_holds_warning or verification_warning

    # Deployment authority does not authorize deletion. Leave even interrupted
    # retirement trees untouched, and preserve the last actual retirement summary.
    release_retirement = {
        "status": "not_requested",
        "reason": "requires_separate_action",
        "retired_bytes": 0,
        "retired_commits": [],
        "renamed": [],
        "direct_delete_fallback": [],
        "deleted": [],
        "swept": [],
        "startup_swept": [],
        "worktree_prune": {"status": "not_requested"},
        "alerts": [],
    }

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "deployed",
        "preserve_configured_controls_state": preserve_configured_controls_state,
        "release_retirement": release_retirement,
        "unit_sandbox_paths": unit_sandbox_paths,
        "stage_timings_seconds": stage_timings,
        "source_commit": commit,
        "disk_reservation": disk_reservation_receipt,
        "disk_reservation_estimate": disk_reservation_estimate,
        "disk_reservation_runtime": disk_reservation_runtime,
        "scene_retirement_runtime": scene_retirement_runtime,
        "scene_retirement_stores": scene_retirement_stores,
        "surfaces": [
            {"name": name, "path": str(path), "head": observed[name]}
            for name, path in sorted(surfaces.items())
        ],
        "release_path": release["release_path"],
        "release_provenance": installed_provenance,
        "created_release_checkout": release["created_release_checkout"],
        "restarted_units": restarted,
        "agent_execution": agent_execution,
        "installed_systemd_units": installed_systemd_units,
        "scene_object_discovery_runtime_directories": (
            scene_object_discovery_runtime_directories
        ),
        "episode_compilation_runtime_directories": (
            episode_compilation_runtime_directories
        ),
        "storage_pins_runtime": storage_pins_runtime,
        "configured_controls_runtime": configured_controls_runtime,
        "configured_controls_autostart_registry": (
            configured_controls_autostart_registry
        ),
        "path_unit_states": [
            row for row in automation_unit_state_receipts if row["unit"].endswith(".path")
        ],
        "timer_unit_states": [
            row for row in automation_unit_state_receipts if row["unit"].endswith(".timer")
        ],
        "quiesced_path_units": [
            row for row in quiesced_automation_units if row["unit"].endswith(".path")
        ],
        "quiesced_timer_units": [
            row for row in quiesced_automation_units if row["unit"].endswith(".timer")
        ],
        # Compatibility projection for readers that predate the state-preserving
        # receipt. It contains only watchers that are active after this deploy.
        "activated_path_units": [
            {
                "unit": row["unit"],
                "enabled": row["after"]["enabled"],
                "state": row["after"]["state"],
            }
            for row in automation_unit_state_receipts
            if row["unit"].endswith(".path") and row["after"]["state"] == "active"
        ],
        "activated_timer_units": [
            {
                "unit": row["unit"],
                "enabled": row["after"]["enabled"],
                "state": row["after"]["state"],
            }
            for row in automation_unit_state_receipts
            if row["unit"].endswith(".timer") and row["after"]["state"] == "active"
        ],
        "intake_runtime_binding": runtime_binding,
        "intake_runtime": runtime,
        "scene_configuration_runtime": scene_configuration_runtime,
        "cad_skill_sources": cad_skill_sources,
        "scene_configuration_environment": scene_configuration_environment,
        "scene_preparation_installation": scene_preparation_installation,
        "controls_autoprovision_installation": controls_autoprovision_installation,
        "terminal_controls_adoptions": terminal_controls_adoptions,
        # Every slot actually held, not the one base path the caller named.
        # The lock is an N-slot semaphore, so recording the input would
        # under-report what this deploy was exclusive with -- and a receipt
        # that under-reports its own guarantee is the thing a later reader
        # trusts.
        "paid_launch_locks_held": [
            str(path) for path in _expanded_slots(paid_launch_locks)
            if path.name not in {run["slot"] for run in paid_runs_in_flight}
        ],
        # Runs that started before the deploy and kept their own release tree.
        "paid_runs_in_flight": list(paid_runs_in_flight),
        "paid_launch_lock_repair": lock_repair,
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
        "claim_boundary": (
            "This receipt proves every named filesystem surface reports this "
            "commit and is clean, and that the restarted intake process reports "
            "the same commit. It says nothing about whether any launch profile, "
            "bundle, or preflight built at an earlier commit is still valid -- "
            "those bind the deployed commit and are rebuilt after a deploy, not "
            "before."
        ),
    }
    if door_holds_warning:
        receipt.setdefault("alerts", []).append(door_holds_warning)
    # Last, once every surface has moved: host changes made outside the door
    # since the previous deploy are reported here, and never fail this one.
    if break_glass_notes_root is not None:
        _report_break_glass_notes(
            receipt, root=break_glass_notes_root, deploy_commit=commit
        )
    return receipt


def _require_trusted_deploy_source(
    source_repo: str | Path, break_glass_note: str | Path | None
) -> dict[str, Any] | None:
    """Refuse a source GPU admission would not trust, unless a fresh note allows it.

    On 2026-09-26 a deploy ran from a scratch checkout. It succeeded, and GPU
    admission then refused every sponsored step for hours, because the receipt
    named a source admission does not trust. The check is admission's own
    predicate, asked of the path the receipt will record, so the two cannot
    drift. It runs before anything on the host changes; a note must be sealed,
    at most a day old and name deploy-from-untrusted-source. Returns what the
    receipt records about that note, or None for a trusted source.
    """

    if trusted_deploy_source(Path(source_repo).expanduser().resolve()):
        return None
    reason = "break_glass_note_missing"
    if break_glass_note is not None:
        supplied_path = Path(break_glass_note).expanduser()
        path = supplied_path.resolve()
        notes_root = Path(DEFAULT_BREAK_GLASS_NOTES_ROOT).expanduser()
        if notes_root.is_symlink():
            reason = "break_glass_notes_root_unsafe"
        elif path.parent != notes_root.resolve():
            reason = "break_glass_note_outside_notes_root"
        else:
            try:
                note = verify_break_glass_note(
                    supplied_path, max_age_seconds=BREAK_GLASS_DEPLOY_NOTE_MAX_AGE_SECONDS
                )
            except BreakGlassNoteError as exc:
                reason = str(exc)
            else:
                if DEPLOY_FROM_UNTRUSTED_SOURCE in note["actions"]:
                    summary = break_glass_note_summary({"name": path.name, **note})
                    return {**summary, "path": str(path), "actions": list(note["actions"])}
                reason = "break_glass_note_action_missing"
    door_clone = Path(source_repo).expanduser().resolve() == (
        Path("/opt/blueprint/control-plane-config-tools") / "operator-door-source"
    )
    raise UntrustedDeploySourceError(reason, door_clone=door_clone)


def _run_deploy_with_signal_cleanup(callback: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Turn a transient unit stop into an exception so deploy rollback runs."""

    global _DEPLOY_ACTIVE_TRANSITION
    _DEPLOY_ACTIVE_TRANSITION = False
    deferred: set[str] = set()
    previous = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}

    def interrupted(number: int, _frame: Any) -> None:
        if _DEPLOY_ACTIVE_TRANSITION:
            deferred.add(signal.Signals(number).name)
            return
        signal.signal(number, signal.SIG_IGN)
        raise ControlPlaneDeployError(f"deploy_interrupted:{signal.Signals(number).name}")

    try:
        for number in previous:
            signal.signal(number, interrupted)
        result = callback()
        if deferred:
            print("deploy_signal_deferred_after_activation:" + ",".join(sorted(deferred)), file=sys.stderr)
        return result
    finally:
        _DEPLOY_ACTIVE_TRANSITION = False
        for number, handler in previous.items():
            signal.signal(number, handler)


def _write_receipt_and_return(receipt: dict[str, Any], path: str | None) -> dict[str, Any]:
    if path:
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(receipt, indent=1, sort_keys=True) + "\n").encode("utf-8")
        descriptor, temporary = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=out.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                # The operator door reads deploy receipts as the blueprint user.
                os.fchmod(stream.fileno(), 0o644)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, out)
            directory = os.open(out.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise
    return receipt


def _finalize_deploy_receipt(
    receipt: dict[str, Any], *, path: str | None, notes_root: Path, commit: str,
    note_was_passed: bool = False,
) -> dict[str, Any]:
    """Persist before marking notes, keeping the deployed receipt on write failure."""

    if receipt.get("break_glass_note") is not None:
        receipt.setdefault("alerts", []).append("break_glass_deploy_from_untrusted_source")
    elif note_was_passed:
        receipt.setdefault("alerts", []).append("break_glass_note_ignored_trusted_source")
    notes = receipt.get("break_glass_notes")
    if not path and isinstance(notes, list) and notes:
        receipt.setdefault("alerts", []).append("break_glass_notes_not_marked:no_receipt_out")
    try:
        _write_receipt_and_return(receipt, path)
    except OSError as exc:
        code = errno.errorcode.get(exc.errno, type(exc).__name__)
        receipt.setdefault("blockers", []).append(f"deploy_receipt_write_failed:{code}")
        return receipt
    if not path or not isinstance(notes, list) or not notes:
        return receipt
    rows = [{"name": row.get("name"), "note_digest": row.get("digest")}
            for row in notes if isinstance(row, Mapping)]
    try:
        mark_break_glass_notes_reported(notes_root, rows, deploy_commit=commit)
    except Exception as exc:  # noqa: BLE001 - marking never invalidates an applied deploy
        code = break_glass_refusal_code(exc)
        receipt["break_glass_notes_error"] = code
        receipt.setdefault("alerts", []).append(f"break_glass_notes_not_marked:{code}")
        try:
            _write_receipt_and_return(receipt, path)
        except OSError as rewrite_error:
            rewrite_code = errno.errorcode.get(rewrite_error.errno, type(rewrite_error).__name__)
            receipt.setdefault("blockers", []).append(f"deploy_receipt_rewrite_failed:{rewrite_code}")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument(
        "--break-glass-note",
        default=None,
        help=(
            "A sealed break-glass note, at most 24 hours old, whose actions "
            "include deploy-from-untrusted-source (record it with python -m "
            "blueprint_pipeline.control_plane_break_glass record). Required "
            "only when --source-repo is not a source GPU admission trusts: "
            "the canonical checkout, or a root-owned clone directly under the "
            "config-tools root."
        ),
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--active-link", required=True)
    parser.add_argument(
        "--release-provenance",
        default=None,
        help=(
            "Exact verified blueprint.deploy_release_provenance.v1 receipt "
            "for --source-commit. Required unless --iteration is given."
        ),
    )
    parser.add_argument(
        "--canary",
        action="store_true",
        help=(
            "With --iteration: accept any commit pushed to origin (not only "
            "origin/main ancestors). For fix-and-fire debugging on a "
            "development_only lane; the release is stamped status=canary and "
            "promotion_eligible=false. Unpushed or dirty-tree commits still "
            "refuse."
        ),
    )
    parser.add_argument(
        "--iteration",
        action="store_true",
        help=(
            "Deploy a pushed commit without waiting for the Full Test Lane. "
            "The release is stamped promotion_eligible=false and evidence "
            "grade development_only. Use for fix-and-fire iteration; promote "
            "with a lane-verified deploy before sealing evidence."
        ),
    )
    parser.add_argument(
        "--systemd-dir",
        default=DEFAULT_SYSTEMD_DIR,
        help="systemd unit directory receiving exact release-owned unit bytes",
    )
    parser.add_argument(
        "--restart-unit",
        action="append",
        default=None,
        help=(
            "An additional systemd unit to restart and confirm active. "
            "Repeatable; the canonical intake unit is always restarted."
        ),
    )
    parser.add_argument(
        "--paid-launch-lock",
        action="append",
        default=None,
        help=(
            "A provider single-flight lock to check before activating. "
            "Repeatable. Defaults to the canonical Vast paid-launch lock."
        ),
    )
    parser.add_argument("--receipt-out")
    parser.add_argument(
        "--intake-runtime-drop-in", default=DEFAULT_INTAKE_RUNTIME_DROP_IN
    )
    parser.add_argument("--intake-version-url", default=DEFAULT_INTAKE_VERSION_URL)
    parser.add_argument(
        "--scene-configuration-environment-file",
        default=DEFAULT_SCENE_CONFIGURATION_ENVIRONMENT_FILE,
    )
    parser.add_argument(
        "--scene-configuration-runtime-root",
        default=DEFAULT_SCENE_CONFIGURATION_RUNTIME_ROOT,
    )
    parser.add_argument("--scene-preparation-bootstrap-file",
                        default="/etc/blueprint/task-evaluation-scene-preparation-bootstrap.json")
    parser.add_argument("--controls-autoprovision-bootstrap-file",
                        default="/etc/blueprint/task-evaluation-controls-autoprovision-bootstrap.json")
    parser.add_argument(
        "--splat-render-prerequisite-root",
        default=DEFAULT_SPLAT_RENDER_PREREQUISITE_ROOT,
    )
    parser.add_argument("--artifixer-source-root", default=DEFAULT_ARTIFIXER_SOURCE_ROOT)
    parser.add_argument(
        "--content-agents-source-root", default=DEFAULT_CONTENT_AGENTS_SOURCE_ROOT
    )
    parser.add_argument(
        "--cad-skill-source-root", default=DEFAULT_CAD_SKILL_SOURCE_ROOT
    )
    parser.add_argument(
        "--astra-blender-archive", type=Path,
        default=BLENDER_INSTALL_ROOT.parent / 'archives' / BLENDER_ARCHIVE_NAME,
        help="Verified official Linux Blender archive to seal with the Astra authoring runtime.",
    )
    parser.add_argument(
        "--configured-controls-autostart-intent-root",
        default=DEFAULT_CONFIGURED_CONTROLS_AUTOSTART_INTENT_ROOT,
    )
    parser.add_argument(
        "--configured-controls-autostart-intent",
        action="append",
        default=None,
        help=(
            "Exact digest-bound per-scene continuation intent to provision. "
            "Repeatable; omitting it leaves scene configuration fail-closed "
            "until pre-admission is provisioned."
        ),
    )
    parser.add_argument(
        "--preserve-configured-controls-state",
        action="store_true",
        help=(
            "Preserve the current enabled and active state of the configured-controls "
            "progression timer and path, including an operator pause. Other automation "
            "retains its normal deployment behavior. Incompatible with --arm-path-units."
        ),
    )
    parser.add_argument(
        "--arm-path-units",
        action="store_true",
        help=(
            "Explicitly enable and start release-owned path watchers. By default "
            "deploy preserves the prior enabled/active state and leaves fresh "
            "installations disarmed."
        ),
    )
    args = parser.parse_args(argv)

    try:
        # Before the deploy function, so a refused source reserves no disk,
        # writes no provenance and moves nothing.
        break_glass_note = _require_trusted_deploy_source(
            args.source_repo, args.break_glass_note
        )
        receipt = _run_deploy_with_signal_cleanup(lambda: _finalize_deploy_receipt({
            **deploy_control_plane_commit(
            source_repo=args.source_repo,
            source_commit=args.source_commit,
            release_root=args.release_root,
            state_root=args.state_root,
            active_link=args.active_link,
            release_provenance=args.release_provenance,
            iteration=args.iteration,
            canary=args.canary,
            restart_units=tuple(args.restart_unit or ()),
            paid_launch_locks=tuple(args.paid_launch_lock or DEFAULT_PAID_LAUNCH_LOCKS),
            intake_runtime_drop_in=args.intake_runtime_drop_in,
            intake_version_url=args.intake_version_url,
            systemd_dir=args.systemd_dir,
            scene_configuration_environment_file=(
                args.scene_configuration_environment_file
            ),
            scene_configuration_runtime_root=args.scene_configuration_runtime_root,
            scene_preparation_bootstrap_file=args.scene_preparation_bootstrap_file,
            controls_autoprovision_bootstrap_file=args.controls_autoprovision_bootstrap_file,
            splat_render_prerequisite_root=args.splat_render_prerequisite_root,
            artifixer_source_root=args.artifixer_source_root,
            content_agents_source_root=args.content_agents_source_root,
            cad_skill_source_root=args.cad_skill_source_root,
            astra_blender_archive_path=args.astra_blender_archive,
            configured_controls_autostart_intent_root=(
                args.configured_controls_autostart_intent_root
            ),
            configured_controls_autostart_intent_sources=tuple(
                args.configured_controls_autostart_intent or ()
            ),
            arm_path_units=args.arm_path_units,
            preserve_configured_controls_state=args.preserve_configured_controls_state,
            disk_reservation_root=(
                Path(args.state_root).expanduser() / "disk-reservations"
            ),
            break_glass_notes_root=DEFAULT_BREAK_GLASS_NOTES_ROOT,
            ),
            "break_glass_note": break_glass_note,
        }, path=args.receipt_out, notes_root=DEFAULT_BREAK_GLASS_NOTES_ROOT,
           commit=args.source_commit, note_was_passed=args.break_glass_note is not None))
    except (OSError, ValueError, ControlPlaneReleaseError) as exc:
        blocked = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                }
        if isinstance(exc, UntrustedDeploySourceError):
            blocked["remedy"] = exc.remedy
        diagnostic = getattr(exc, "runtime_diagnostic", None)
        if (str(exc) == "deploy_scene_retirement_runtime_unproven" and isinstance(diagnostic, dict)
                and diagnostic.get("phase") in {"retained_installer", "authenticate_installer", "execute_installer", "verify_result",
                    "signed_release", "build_sdk", "resume_initial_intent", "refresh", "prepare", "publish_installer"}
                and diagnostic.get("reason") in {"deadline", "validation", "io", "unexpected"}):
            blocked["runtime_diagnostic"] = {key: diagnostic[key] for key in ("phase", "reason")}
        print(json.dumps(blocked, indent=1, sort_keys=True))
        return 2

    print(json.dumps(receipt, indent=1, sort_keys=True))
    return 2 if any(str(code).startswith("deploy_receipt_write_failed:") or
                    str(code).startswith("deploy_receipt_rewrite_failed:")
                    for code in receipt.get("blockers", [])) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
