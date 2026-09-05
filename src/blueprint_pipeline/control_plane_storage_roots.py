"""Classify every production storage root on the control-plane host by retention law.

The host filled its disk four times because bytes of very different kinds were
written under the same roots with no rule about which could ever be removed.
This module is the single table of production roots and their storage class:

``evidence_hot``
    Append-only and read by live services (spend guard ledgers, deploy
    receipts, standing authorizations, the launch profile catalog).  Never
    evicted, never offloaded.
``evidence_cold``
    Append-only run evidence.  A sealed run directory may be offloaded to the
    artifact store after its hot window and replaced by a digest-bound pointer;
    bytes are migrated, never deleted.
``cache``
    Content-addressed or derived bytes reproducible from immutable inputs.
    Evictable when no live pin, queue message, or hardlink references them.
``work``
    Queues and per-run scratch owned by their queue contract.
``release``
    Per-commit trees created by deploy; retired by deploy when superseded and
    unreferenced.
``ledger``
    Coordination state: reservations, pins, locks.
``container``
    A parent directory that only holds classified children.
``staging``
    The isolated staging intake tree.
``scratch``
    Reproducible diagnostics and engineering scratch.  Nothing references a
    scratch tree from a queue or a receipt, so it is reaped by idle age alone.

The storage reclaim tools validate their configured roots against this table,
so a unit environment that points a reaper at an evidence root is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath


STORAGE_CLASSES = frozenset(
    {
        "evidence_hot",
        "evidence_cold",
        "cache",
        "work",
        "release",
        "ledger",
        "container",
        "staging",
        "scratch",
    }
)


@dataclass(frozen=True)
class StorageRoot:
    path: str
    storage_class: str
    owner: str
    note: str


_CONTROL_PLANE = "/var/lib/blueprint/pipeline-control-plane"
_INPUTS = "/var/lib/blueprint/task-evaluation-inputs"

STORAGE_ROOTS: tuple[StorageRoot, ...] = (
    StorageRoot("/var/lib/blueprint", "container", "blueprint", "service state tree"),
    StorageRoot(_CONTROL_PLANE, "container", "blueprint", "control-plane state"),
    StorageRoot(_INPUTS, "container", "blueprint", "immutable and derived launch inputs"),
    StorageRoot("/var/lib/blueprint-staging", "staging", "blueprint", "isolated staging intake"),
    # --- evidence that live services read; never evicted or offloaded
    StorageRoot(f"{_CONTROL_PLANE}/gpu_spend_guard", "evidence_hot", "blueprint", "spend ledger, billing audit, admission lock"),
    StorageRoot(f"{_CONTROL_PLANE}/deploy-receipts", "evidence_hot", "root", "deploy receipts"),
    StorageRoot(f"{_CONTROL_PLANE}/standing-authorizations", "evidence_hot", "blueprint", "standing launch authorizations"),
    StorageRoot(f"{_CONTROL_PLANE}/task-evaluation-launch-profile-catalog.json", "evidence_hot", "blueprint", "public launch profile catalog"),
    StorageRoot(f"{_CONTROL_PLANE}/live_pipeline_control_plane_manifest.json", "evidence_hot", "blueprint", "control-plane manifest"),
    StorageRoot(f"{_CONTROL_PLANE}/task-evaluation-launch-reconciliation", "evidence_hot", "blueprint", "reconciler reports"),
    StorageRoot(f"{_CONTROL_PLANE}/task-evaluation-launch-supervision", "evidence_hot", "blueprint", "launch supervisor reports"),
    StorageRoot(f"{_CONTROL_PLANE}/storage-gc", "evidence_hot", "blueprint", "storage reclaim reports"),
    StorageRoot(f"{_CONTROL_PLANE}/episode-interpretation-rights", "evidence_hot", "blueprint", "human-approved per-episode disclosure rights"),
    StorageRoot("/var/lib/blueprint/production-gpu-campaigns.sqlite", "evidence_hot", "blueprint", "gpu campaign ledger"),
    StorageRoot("/var/lib/blueprint/production-gpu-worker-pool.sqlite", "evidence_hot", "blueprint", "gpu worker pool ledger"),
    # --- sealed run evidence; offloadable after the hot window
    StorageRoot(f"{_CONTROL_PLANE}/task-evaluation-launch-runs", "evidence_cold", "blueprint", "launch run directories"),
    StorageRoot(f"{_CONTROL_PLANE}/task-evaluation-policy-canaries", "evidence_cold", "blueprint", "policy canary run directories"),
    StorageRoot(f"{_CONTROL_PLANE}/capture-reconstruction-runs", "evidence_cold", "blueprint", "capture reconstruction runs"),
    StorageRoot("/var/lib/blueprint/production-gpu-artifacts", "evidence_cold", "blueprint", "gpu campaign artifacts"),
    # --- coordination ledgers
    StorageRoot(f"{_CONTROL_PLANE}/disk-reservations", "ledger", "blueprint", "disk admission ledger"),
    StorageRoot(f"{_CONTROL_PLANE}/storage-pins", "ledger", "blueprint", "derived-directory pins"),
    StorageRoot(f"{_CONTROL_PLANE}/provider-locks", "ledger", "blueprint", "paid launch lock slots"),
    # --- queues and scratch
    StorageRoot(f"{_CONTROL_PLANE}/task-evaluation-launches", "work", "blueprint", "launch queue"),
    StorageRoot(f"{_CONTROL_PLANE}/task-evaluation-launch-preparations", "work", "blueprint", "preparation queue"),
    StorageRoot(f"{_CONTROL_PLANE}/sam31-preparation-executions", "work", "blueprint", "SAM preparation child queue"),
    StorageRoot(f"{_CONTROL_PLANE}/task-evaluation-episode-compilations", "work", "blueprint", "compilation queue"),
    StorageRoot(f"{_CONTROL_PLANE}/task-evaluation-launch-activations", "work", "blueprint", "activation queue"),
    StorageRoot(f"{_CONTROL_PLANE}/task-evaluation-policy-canary-dispatches", "work", "blueprint", "canary dispatch queue"),
    StorageRoot(f"{_CONTROL_PLANE}/task-evaluation-scene-constructions", "work", "blueprint", "scene construction queue"),
    StorageRoot(f"{_CONTROL_PLANE}/scene-object-discoveries", "work", "blueprint", "scene object discovery queue"),
    StorageRoot(f"{_CONTROL_PLANE}/task-evaluation-terminal-resource-releases", "work", "blueprint", "terminal resource release queue"),
    StorageRoot(f"{_CONTROL_PLANE}/task-evaluation-terminal-resource-releases-state", "work", "blueprint", "terminal resource release state"),
    StorageRoot(f"{_CONTROL_PLANE}/task-evaluation-configured-controls", "work", "blueprint", "configured controls progression state"),
    StorageRoot(f"{_CONTROL_PLANE}/capture-reconstruction-queue", "work", "blueprint", "capture reconstruction queue"),
    StorageRoot(f"{_CONTROL_PLANE}/capture-reconstruction-derived", "cache", "blueprint", "derived reconstruction bytes"),
    StorageRoot(f"{_CONTROL_PLANE}/profile-install-staging", "cache", "blueprint", "reproducible launch-profile installation staging"),
    StorageRoot(f"{_CONTROL_PLANE}/policy-canary-presubmission", "cache", "blueprint", "reproducible policy-canary presubmission packets"),
    StorageRoot("/var/lib/blueprint/pubsub-handoffs", "work", "blueprint", "pubsub handoff spool"),
    # --- reproducible derived inputs
    StorageRoot(f"{_INPUTS}/prepared-references", "cache", "blueprint", "materialized references and content store"),
    StorageRoot(f"{_INPUTS}/sam31-preparations", "cache", "blueprint", "reproducible SAM preparation artifacts"),
    StorageRoot(f"{_INPUTS}/compiled-episodes", "cache", "blueprint", "compiled episodes and adapter member store"),
    StorageRoot(f"{_INPUTS}/launch-activations", "cache", "blueprint", "activation launch sets"),
    StorageRoot(f"{_INPUTS}/scene-object-discoveries", "cache", "blueprint", "scene object discovery inputs"),
    StorageRoot(f"{_INPUTS}/scene-object-discovery-outputs", "cache", "blueprint", "scene object discovery outputs"),
    StorageRoot(f"{_INPUTS}/policy-canary-execution-setups", "cache", "blueprint", "canary execution setups"),
    StorageRoot(f"{_INPUTS}/policy-canary-execution-setup-template.json", "cache", "blueprint", "canary execution setup template"),
    # --- per-commit trees
    StorageRoot(f"{_INPUTS}/system-runtimes", "release", "root", "per-commit renderer and toolchain trees"),
    StorageRoot("/opt/blueprint/task-evaluation-control-plane-releases", "release", "root", "per-commit release worktrees"),
    StorageRoot("/opt/blueprint/task-evaluation-control-plane", "release", "root", "active release link"),
    StorageRoot("/opt/blueprint/BlueprintCapturePipeline", "release", "root", "mutable source checkout and venv"),
    StorageRoot("/opt/blueprint/BlueprintCapturePipeline-staging", "staging", "root", "staging checkout"),
    StorageRoot(f"{_CONTROL_PLANE}/release-retention", "evidence_hot", "root", "release retirement plans and receipts"),
    StorageRoot(f"{_CONTROL_PLANE}/capacity", "evidence_hot", "root", "capacity controller reports"),
    StorageRoot(f"{_CONTROL_PLANE}/preflight", "evidence_hot", "root", "chain preflight reports"),
    StorageRoot(f"{_CONTROL_PLANE}/launch-materialization", "evidence_hot", "blueprint", "spend reconciliations and materialized launch inputs"),
    StorageRoot(f"{_CONTROL_PLANE}/episode-interpretation-backfills", "evidence_cold", "root", "episode interpretation backfill runs"),
    StorageRoot(f"{_CONTROL_PLANE}/policy-canary-preprovider-audits", "evidence_cold", "root", "policy canary pre-provider audits"),
    StorageRoot(f"{_CONTROL_PLANE}/scene-configuration-diagnostics", "cache", "blueprint", "sealed diagnostic bundles with their own retention tool"),
    StorageRoot(f"{_CONTROL_PLANE}/submission-publication-locks", "ledger", "blueprint", "submission publication locks"),
    StorageRoot("/var/lib/blueprint/spend-authority", "ledger", "blueprint", "single-use spend authority consumption ledger"),
    StorageRoot(f"{_CONTROL_PLANE}/engineering", "scratch", "blueprint", "agent engineering scratch"),
    StorageRoot(f"{_CONTROL_PLANE}/render-probes", "scratch", "root", "render diagnostics"),
    StorageRoot(f"{_CONTROL_PLANE}/diagnostic-checkouts", "scratch", "blueprint", "diagnostic source checkouts"),
    StorageRoot(f"{_CONTROL_PLANE}/release-builds", "scratch", "root", "release build staging"),
    StorageRoot(f"{_INPUTS}/render-probes", "scratch", "root", "render diagnostic inputs"),
    StorageRoot(f"{_INPUTS}/stage-replays", "scratch", "blueprint", "stage replay outputs (task_evaluation_stage_replay)"),
)


def classify_path(path: str) -> StorageRoot | None:
    """Return the most specific classified root containing ``path``."""

    candidate = PurePosixPath(str(path))
    best: StorageRoot | None = None
    for root in STORAGE_ROOTS:
        root_path = PurePosixPath(root.path)
        if candidate == root_path or root_path in candidate.parents:
            if best is None or len(root_path.parts) > len(PurePosixPath(best.path).parts):
                best = root
    return best


def roots_of_class(storage_class: str) -> tuple[str, ...]:
    if storage_class not in STORAGE_CLASSES:
        raise ValueError(f"control_plane_storage_class_invalid:{storage_class}")
    return tuple(root.path for root in STORAGE_ROOTS if root.storage_class == storage_class)


def require_storage_class(path: str, *, expected: str, code: str) -> StorageRoot:
    """Refuse a tool root whose classification is not ``expected``."""

    root = classify_path(path)
    if root is None or root.storage_class != expected:
        observed = root.storage_class if root is not None else "unclassified"
        raise ValueError(f"{code}:{observed}")
    return root


__all__ = [
    "STORAGE_CLASSES",
    "STORAGE_ROOTS",
    "StorageRoot",
    "classify_path",
    "require_storage_class",
    "roots_of_class",
]
