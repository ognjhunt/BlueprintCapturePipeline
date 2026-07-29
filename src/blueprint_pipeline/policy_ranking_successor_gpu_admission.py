"""Fail-closed paid-GPU admission for the Cosmos3 successor smoke canary.

This module is an adapter behind ``paid_resource_allocator gpu-canary``.  It
cannot be used as a standalone launcher.  The first paid phase is restricted to
one Vast RTX PRO 6000 Blackwell instance: two reduced-step qualification
generations must prove the exact direct and wrapper stack before the
already-frozen ten-rollout scientific matrix is allowed to continue.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import utc_now_iso, write_json
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionBlocked,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from .policy_ranking_successor_cosmos import (
    CHECKPOINT_REPOSITORY,
    CHECKPOINT_REVISION,
    COSMOS_FRAMEWORK_REVISION,
    COSMOS_REVISION,
    VLLM_IMAGE,
    VLLM_IMAGE_DIGEST,
    canonical_sha256,
    validate_smoke_inventory_manifest,
)
from .provider_runtime_bundle_contract import provider_runtime_contract_blockers
from .gpu_render_providers import get_render_provider
from .groot_oscar_runpod_watchdog import EVIDENCE_NAME as RUNPOD_WATCHDOG_EVIDENCE_NAME
from .runpod_wam_async_runner import (
    RUNPOD_WAM_DISABLE_WARM_CANDIDATE_ENV,
    RUNPOD_WAM_TERMINAL_HOLD_SECONDS_ENV,
    RUNPOD_WAM_TEARDOWN_ACTION_ENV,
    create_runpod_wam_async_run,
    poll_runpod_wam_async_run,
)
from .vast_wam_authorized_runner import run_vast_wam_authorized_runner
from .vast_provider_adapter import (
    VAST_API_GATE_ENV,
    VAST_INSTANCE_LAUNCH_GATE_ENV,
    _env_truthy as _vast_env_truthy,
)
from .vast_session_budget_contract import (
    build_vast_session_budget_guard,
    successor_session_live_limit_minutes,
)
from .watchdog_owner_teardown_contract import write_owner_teardown_cancel_request


PROBE_KIND = "policy-ranking-successor-cosmos"
FOLLOWUP_EXPERIMENT_ID = "policy_ranking_cosmos3_followup_20260728"
SCHEMA_VERSION = "policy_ranking_successor_gpu_admission.v1"
AUTHORIZATION_SCHEMA = "policy_ranking_successor_compute_authorization.v1"
PREFLIGHT_SCHEMA = "policy_ranking_successor_vast_preflight.v1"
BUNDLE_SCHEMA = "policy_ranking_successor_cosmos_bundle.v1"
PUBLIC_IMAGE = f"{VLLM_IMAGE}@{VLLM_IMAGE_DIGEST}"


@dataclass(frozen=True)
class SuccessorGPUProfile:
    """Frozen identities and spend limits for one Cosmos successor experiment."""

    experiment_id: str
    admission_schema: str
    authorization_schema: str
    preflight_schema: str
    receipt_schema: str
    authorization_ids_by_allocation_index: Mapping[int, str]
    cost_authorization_binding_sha256: str
    expected_bundle_sha256: str
    expected_bundle_size_bytes: int
    expected_embedded_input_hashes: Mapping[str, str]
    qualification_canary_request_count: int
    scientific_matrix_request_count: int
    total_initial_generation_request_count: int
    request_budget_amendment_sha256: str | None
    max_compute_cap_usd: float
    max_hourly_rate_usd: float
    target_spend_usd: float
    hard_ttl_seconds: int
    reference_bundle: bool = False
    powered_bundle: bool = False
    cosmos_revision: str = COSMOS_REVISION
    cosmos_framework_revision: str = COSMOS_FRAMEWORK_REVISION
    vllm_omni_revision: str | None = None
    allowed_providers: tuple[str, ...] = ("vast",)
    compatible_gpu_keywords: tuple[str, ...] = ("RTX PRO 6000",)


MAX_COMPUTE_CAP_USD = 6.0
MAX_HOURLY_RATE_USD = 1.05
TARGET_SPEND_USD = 3.25
HARD_TTL_SECONDS = 10_800
DISK_GB = 250
MIN_GPU_RAM_MB = 95_000
MIN_RELIABILITY = 0.98
MAX_PREFLIGHT_AGE_SECONDS = 900
AUTHORIZATION_IDS_BY_ALLOCATION_INDEX = {
    2: "policy-ranking-cosmos3-followup-20260728-allocation-2",
    3: "policy-ranking-cosmos3-followup-20260728-allocation-3",
}
GOAL_COST_AUTHORIZATION_SHA256 = "7f2ebe7ae1d176f9eea6b97a2b2f0ce235e7c5ff6af0ddd3baaef9000ab92cc0"
AUTHORIZATION_CONSUMPTION_ROOT = Path.home() / ".blueprint-spend-authority" / "consumed"
EXPECTED_BUNDLE_SHA256 = "0e938e1674ff2efc043363ab9b7e2724ae2f9bc264e289895d7759a9eb8173fd"
EXPECTED_BUNDLE_SIZE_BYTES = 301_649
QUALIFICATION_CANARY_REQUEST_COUNT = 2
SCIENTIFIC_MATRIX_REQUEST_COUNT = 10
TOTAL_INITIAL_GENERATION_REQUEST_COUNT = 12
REQUEST_BUDGET_AMENDMENT_SHA256 = "e67226e16318a073e9190915554dc37b1d378fc155c6eb6bec7ecc79fb27786a"
EXPECTED_EMBEDDED_INPUT_HASHES = {
    "initial_observation_sha256": (
        "ed17ad901b3f1779d5282e328bafaac1be8e42a2407c668e188f6e24261cc23b"
    ),
    "smoke_inventory_sha256": ("1e1a6eb0be7d31067ad91e871430e3f87523b64e414ae8cccad52dbf52b58f0f"),
    "action_streams_sha256": ("1deafb10863b46895354ac274dcbe7243e1ab199e80d73f9b921f2a967b6e3e2"),
}

LEGACY_PROFILE = SuccessorGPUProfile(
    experiment_id=FOLLOWUP_EXPERIMENT_ID,
    admission_schema=SCHEMA_VERSION,
    authorization_schema=AUTHORIZATION_SCHEMA,
    preflight_schema=PREFLIGHT_SCHEMA,
    receipt_schema="policy_ranking_successor_cosmos_bundle_receipt.v1",
    authorization_ids_by_allocation_index=AUTHORIZATION_IDS_BY_ALLOCATION_INDEX,
    cost_authorization_binding_sha256=GOAL_COST_AUTHORIZATION_SHA256,
    expected_bundle_sha256=EXPECTED_BUNDLE_SHA256,
    expected_bundle_size_bytes=EXPECTED_BUNDLE_SIZE_BYTES,
    expected_embedded_input_hashes=EXPECTED_EMBEDDED_INPUT_HASHES,
    qualification_canary_request_count=QUALIFICATION_CANARY_REQUEST_COUNT,
    scientific_matrix_request_count=SCIENTIFIC_MATRIX_REQUEST_COUNT,
    total_initial_generation_request_count=TOTAL_INITIAL_GENERATION_REQUEST_COUNT,
    request_budget_amendment_sha256=REQUEST_BUDGET_AMENDMENT_SHA256,
    max_compute_cap_usd=MAX_COMPUTE_CAP_USD,
    max_hourly_rate_usd=MAX_HOURLY_RATE_USD,
    target_spend_usd=TARGET_SPEND_USD,
    hard_ttl_seconds=HARD_TTL_SECONDS,
)

PHASE_B_PROFILE = SuccessorGPUProfile(
    experiment_id="policy_ranking_roboarena_disjoint_reasoner_successor_20260728",
    admission_schema="policy_ranking_phase_b_native_cosmos_gpu_admission.v1",
    authorization_schema="policy_ranking_phase_b_native_cosmos_compute_authorization.v1",
    preflight_schema="policy_ranking_phase_b_native_cosmos_vast_preflight.v1",
    receipt_schema="policy_ranking_phase_b_native_cosmos_bundle_receipt.v1",
    authorization_ids_by_allocation_index={
        1: "policy-ranking-roboarena-phase-b-native-cosmos-20260728-allocation-1",
        2: "policy-ranking-roboarena-phase-b-high-motion-cosmos-20260728-allocation-2",
    },
    cost_authorization_binding_sha256=(
        "b2411e5981af473ff992b13f253253674a33293fa396012e9cf633695f3aa196"
    ),
    expected_bundle_sha256="56e0b86e1070140587b132bbe04178382a4f1478b200c282958e59d7d935f823",
    expected_bundle_size_bytes=452_403,
    expected_embedded_input_hashes={
        "initial_observation_sha256": (
            "c1d89dd07b597796ad7620661dd2eacd4d4f58aad03d8860a52e13612bf0d99a"
        ),
        "smoke_inventory_sha256": (
            "9acdfb578d1c595970e3d33a0daade2ecfa1fc0ea6b87cddd163a8bf374747fb"
        ),
        "action_streams_sha256": (
            "12b572607ec3f4f68c2514f15553dfd1ef8c420fa0e7e601cefe8e12447a59b4"
        ),
    },
    qualification_canary_request_count=2,
    scientific_matrix_request_count=12,
    total_initial_generation_request_count=14,
    request_budget_amendment_sha256=None,
    max_compute_cap_usd=5.0,
    max_hourly_rate_usd=1.25,
    target_spend_usd=2.5,
    hard_ttl_seconds=7_200,
)
PHASE_B_POSITIVE_CONTROL_PROFILE = SuccessorGPUProfile(
    experiment_id="policy_ranking_roboarena_disjoint_reasoner_successor_20260728",
    admission_schema="policy_ranking_phase_b_native_cosmos_positive_control_gpu_admission.v1",
    authorization_schema=(
        "policy_ranking_phase_b_native_cosmos_positive_control_compute_authorization.v1"
    ),
    preflight_schema="policy_ranking_phase_b_native_cosmos_positive_control_vast_preflight.v1",
    receipt_schema=("policy_ranking_phase_b_native_cosmos_positive_control_bundle_receipt.v1"),
    authorization_ids_by_allocation_index={
        4: "policy-ranking-roboarena-phase-b-positive-control-20260728-allocation-4",
    },
    cost_authorization_binding_sha256=(
        "ae034d9ebd976c4f4d5540d340182d69568555bf772b18fd4b6bc6544bfbc7fc"
    ),
    expected_bundle_sha256="0daebe3845c4ad70effce16e793abdb1f5307a12a28840cce97578bcaad81229",
    expected_bundle_size_bytes=1_776_383,
    expected_embedded_input_hashes={
        "initial_observation_sha256": (
            "c1d89dd07b597796ad7620661dd2eacd4d4f58aad03d8860a52e13612bf0d99a"
        ),
        "smoke_inventory_sha256": (
            "9acdfb578d1c595970e3d33a0daade2ecfa1fc0ea6b87cddd163a8bf374747fb"
        ),
        "action_streams_sha256": (
            "12b572607ec3f4f68c2514f15553dfd1ef8c420fa0e7e601cefe8e12447a59b4"
        ),
        "positive_control_manifest_sha256": (
            "326efd06d1659c57b86e979eee7a7b30611fd22eda00f37c11e0f3ddbd0c3584"
        ),
    },
    qualification_canary_request_count=2,
    scientific_matrix_request_count=12,
    total_initial_generation_request_count=18,
    request_budget_amendment_sha256=(
        "326efd06d1659c57b86e979eee7a7b30611fd22eda00f37c11e0f3ddbd0c3584"
    ),
    max_compute_cap_usd=5.0,
    max_hourly_rate_usd=1.25,
    target_spend_usd=2.5,
    hard_ttl_seconds=7_200,
)
DROID_REFERENCE_PROFILE = SuccessorGPUProfile(
    experiment_id="policy_ranking_roboarena_droid_reference_confirmation_20260729",
    admission_schema="policy_ranking_cosmos3_droid_reference_gpu_admission.v1",
    authorization_schema="policy_ranking_cosmos3_droid_reference_compute_authorization.v1",
    preflight_schema="policy_ranking_cosmos3_droid_reference_vast_preflight.v1",
    receipt_schema="policy_ranking_cosmos3_droid_reference_bundle_receipt.v1",
    authorization_ids_by_allocation_index={
        1: "policy-ranking-droid-reference-20260729-allocation-1",
        2: "policy-ranking-droid-reference-20260729-allocation-2",
        3: "policy-ranking-droid-reference-20260729-allocation-3",
        4: "policy-ranking-droid-reference-20260729-allocation-4",
    },
    cost_authorization_binding_sha256=(
        "305668fe34d4524caa0d7dc5ce301e44a1a04e0b66176719c02a8cab76373cb4"
    ),
    expected_bundle_sha256="d8378dda5c21757c35cb010506615cdb2886c11fbe4c6c9dbd97ff7aef8b044f",
    expected_bundle_size_bytes=420_221,
    expected_embedded_input_hashes={
        "reference_manifest_sha256": (
            "7d7ec85a0976ed5ede44db53a8566bd50f6596b99f2bf22f0383593d20b08ffe"
        ),
        "initial_observation_sha256": (
            "e8f2735942986934a77a47a9c1f50fd5b55ade03bc241e3950193aaf1137004f"
        ),
        "action_streams_sha256": (
            "acbccfdbea8a645cd8b211109dba5ab434f530427087ea3a3b1301eff71a8263"
        ),
        "provider_runtime_runner_sha256": (
            "0e7ea3d4de04c50a548d7f6b1a65f515a0de126cd51df6863bab4e341b9eeabd"
        ),
    },
    qualification_canary_request_count=2,
    scientific_matrix_request_count=0,
    total_initial_generation_request_count=2,
    request_budget_amendment_sha256=None,
    max_compute_cap_usd=5.0,
    max_hourly_rate_usd=2.05,
    target_spend_usd=2.5,
    hard_ttl_seconds=7_200,
    reference_bundle=True,
    cosmos_revision="0299468993d8bcd8f6a95b0d8427b1221fccfced",
    cosmos_framework_revision="9726697a83315540c6885baefd2fe353d9c74920",
    vllm_omni_revision="1c6e7313394923000215a3299f4f79ede3873ecc",
    allowed_providers=("vast", "runpod"),
    compatible_gpu_keywords=("RTX PRO 6000", "H100"),
)
POWERED_DROID_PROFILE = SuccessorGPUProfile(
    experiment_id="policy_ranking_roboarena_powered_droid_confirmation_20260729",
    admission_schema="policy_ranking_powered_droid_gpu_admission.v1",
    authorization_schema="policy_ranking_powered_droid_compute_authorization.v1",
    preflight_schema="policy_ranking_powered_droid_provider_preflight.v1",
    receipt_schema="policy_ranking_powered_droid_bundle_receipt.v1",
    authorization_ids_by_allocation_index={
        1: "policy-ranking-powered-droid-20260729-allocation-1",
        2: "policy-ranking-powered-droid-20260729-allocation-2",
    },
    cost_authorization_binding_sha256=(
        "6a7486b8eda01057934106206c5ecf3de808d19f3b3124a4a829c7a144f4c689"
    ),
    expected_bundle_sha256="bae438e48fa4ac2544840c91e713cdfc1274334820f1d7649d0c772876f1831a",
    expected_bundle_size_bytes=17_986_110,
    expected_embedded_input_hashes={
        "provider_packet_sha256": (
            "e9eceb32f9875cf2399877b91d43a709fb58222c944f0a16ed35c2f33120d9ee"
        ),
        "image_manifest_sha256": (
            "1804981b2c05695a04dc1816dd0d3e3bba0c59333dd3058db18b55463804c4cc"
        ),
        "official_canary_manifest_sha256": (
            "7d7ec85a0976ed5ede44db53a8566bd50f6596b99f2bf22f0383593d20b08ffe"
        ),
        "provider_runtime_runner_sha256": (
            "53b3627ecc444dd6ca3cc1468b1ac7149ce674c7966fdb64e89303114343fcbc"
        ),
    },
    qualification_canary_request_count=1,
    scientific_matrix_request_count=612,
    total_initial_generation_request_count=613,
    request_budget_amendment_sha256=None,
    max_compute_cap_usd=10.0,
    max_hourly_rate_usd=2.05,
    target_spend_usd=6.2,
    hard_ttl_seconds=10_800,
    powered_bundle=True,
    cosmos_revision="0299468993d8bcd8f6a95b0d8427b1221fccfced",
    cosmos_framework_revision="9726697a83315540c6885baefd2fe353d9c74920",
    vllm_omni_revision="1c6e7313394923000215a3299f4f79ede3873ecc",
    allowed_providers=("vast", "runpod"),
    compatible_gpu_keywords=("RTX PRO 6000", "H100"),
)
RTX_ALLOWED_KEYWORDS = ("RTX PRO 6000",)
RTX_SELECTION_POLICY: Mapping[str, Any] = {
    "policy_id": "policy_ranking_successor_rtx_pro_6000_blackwell_preflight",
    "denied_gpu_keywords": (),
    "allowed_gpu_keywords": RTX_ALLOWED_KEYWORDS,
    "reason": (
        "provisional exact-stack compatibility preflight; scientific rollouts "
        "continue only after the Blackwell stack passes"
    ),
}

REQUIRED_BUNDLE_ENTRIES = frozenset(
    {
        "provider_runtime/wam_provider_runtime_runner.py",
        "provider_runtime/run_wam_provider_runtime.sh",
        "provider_runtime/successor_retained_control.py",
        "provider_runtime/wam_provider_runtime_manifest.json",
        "provider_runtime/wam_rollout_input_manifest.json",
        "provider_runtime/cosmos3_input/initial_observation.png",
        "provider_runtime/cosmos3_input/smoke_request_inventory.json",
        "provider_runtime/cosmos3_input/action_streams.json",
    }
)
REFERENCE_BUNDLE_ENTRIES = frozenset(
    {
        "provider_runtime/wam_provider_runtime_runner.py",
        "provider_runtime/run_wam_provider_runtime.sh",
        "provider_runtime/successor_retained_control.py",
        "provider_runtime/wam_provider_runtime_manifest.json",
        "provider_runtime/wam_rollout_input_manifest.json",
        "provider_runtime/cosmos3_droid_reference/canary_manifest.json",
        "provider_runtime/cosmos3_droid_reference/initial_observation.png",
        "provider_runtime/cosmos3_droid_reference/action_streams.json",
    }
)
POWERED_BUNDLE_ENTRIES = frozenset(
    {
        "provider_runtime/wam_provider_runtime_runner.py",
        "provider_runtime/run_wam_provider_runtime.sh",
        "provider_runtime/successor_retained_control.py",
        "provider_runtime/wam_provider_runtime_manifest.json",
        "provider_runtime/wam_rollout_input_manifest.json",
        "provider_runtime/cosmos3_powered_droid/packet.json",
        "provider_runtime/cosmos3_powered_droid/official_canary/canary_manifest.json",
        "provider_runtime/cosmos3_powered_droid/official_canary/initial_observation.png",
        "provider_runtime/cosmos3_powered_droid/official_canary/action_streams.json",
    }
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected_json_object:{path}")
    return dict(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_successor_vast_preflight(
    *,
    name_prefix: str,
    profile: SuccessorGPUProfile = PHASE_B_PROFILE,
) -> dict[str, Any]:
    """Perform a read-only, zero-inventory Vast capacity preflight."""

    provider = get_render_provider("vast")
    request = {
        "max_hourly_rate_usd": profile.max_hourly_rate_usd,
        "min_gpu_ram_mb": MIN_GPU_RAM_MB,
        "min_reliability": MIN_RELIABILITY,
        "require_avx": True,
        "require_known_supported_isaac_driver": False,
        "require_direct_port": False,
        "preferred_gpu_keywords": list(RTX_ALLOWED_KEYWORDS),
        "min_compute_cap": 1200,
        "max_compute_cap": 0,
        "prefer_isaac_rt": False,
        "gpu_selection_policy": RTX_SELECTION_POLICY,
    }
    capacity = provider.capacity_preflight(request)
    task_inventory = provider.billable_inventory(name_prefix=name_prefix)
    inventory = provider.billable_inventory(name_prefix="")
    viable = [
        dict(row)
        for row in capacity.get("viable_gpu_types", [])
        if isinstance(row, Mapping)
        and "RTX PRO 6000" in str(row.get("gpu_name") or "").upper()
        and int(row.get("num_gpus") or 0) == 1
        and int(row.get("gpu_ram_mb") or 0) >= MIN_GPU_RAM_MB
        and 0 < float(row.get("hourly_rate_usd") or 0) <= profile.max_hourly_rate_usd
        and float(row.get("reliability") or 0) >= MIN_RELIABILITY
    ]
    viable.sort(key=lambda row: (float(row["hourly_rate_usd"]), -float(row["reliability"])))
    selected = viable[0] if viable else {}
    inventory_zero = bool(
        inventory.get("api_confirmed") is True and inventory.get("live_resource_count") == 0
    )
    api_verified = bool(
        capacity.get("status") == "available"
        and inventory.get("api_confirmed") is True
        and task_inventory.get("api_confirmed") is True
    )
    blockers: list[str] = []
    if not api_verified:
        blockers.append("successor_vast_api_not_verified")
    if not inventory_zero:
        blockers.append("successor_vast_inventory_not_zero")
    if not selected:
        blockers.append("successor_compatible_single_rtx_pro_6000_offer_unavailable")
    result: dict[str, Any] = {
        "schema_version": profile.preflight_schema,
        "status": "verified" if not blockers else "blocked",
        "experiment_id": profile.experiment_id,
        "provider": "vast",
        "observed_at_epoch": time.time(),
        "blockers": blockers,
        "provider_api_verified": api_verified,
        "provider_inventory_verified_zero": inventory_zero,
        "selected_offer": selected or None,
        "capacity_request": request,
        "capacity_snapshot": capacity,
        "task_billable_inventory": task_inventory,
        "billable_inventory": inventory,
        "provider_mutations_performed": 0,
        "reservation_proven": False,
        "raw_secret_values_recorded": False,
    }
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def collect_successor_runpod_preflight(
    *,
    name_prefix: str,
    profile: SuccessorGPUProfile = DROID_REFERENCE_PROFILE,
    gpu_type_ids: Sequence[str] = (
        "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "NVIDIA H100 80GB HBM3",
        "NVIDIA H100 PCIe",
    ),
) -> dict[str, Any]:
    """Read-only RunPod capacity and provider-zero preflight for Cosmos inference."""

    provider = get_render_provider("runpod")
    request = {
        "gpuTypeIds": list(gpu_type_ids),
        "cloudType": "SECURE",
        "min_gpu_ram_mb": 80_000,
        "requires_rtx": False,
    }
    capacity = provider.capacity_preflight(request)
    task_inventory = provider.billable_inventory(name_prefix=name_prefix)
    inventory = provider.billable_inventory(name_prefix="")
    viable = []
    for row in capacity.get("viable_gpu_types", []):
        if not isinstance(row, Mapping):
            continue
        gpu_name = str(row.get("gpu_type_id") or row.get("display_name") or "")
        memory_gb = int(row.get("memory_in_gb") or 0)
        price = float(row.get("on_demand_price_usd_per_hour") or 0.0)
        if (
            any(keyword in gpu_name.upper() for keyword in ("RTX PRO 6000", "H100"))
            and memory_gb >= 80
            and 0.0 < price <= profile.max_hourly_rate_usd
            and row.get("capacity_confidence") == "advisory"
        ):
            viable.append(
                {
                    **dict(row),
                    "gpu_name": gpu_name,
                    "gpu_ram_mb": memory_gb * 1000,
                    "hourly_rate_usd": price,
                }
            )
    viable.sort(key=lambda row: float(row["hourly_rate_usd"]))
    selected = viable[0] if viable else {}
    inventory_zero = bool(
        inventory.get("api_confirmed") is True and inventory.get("live_resource_count") == 0
    )
    api_verified = bool(
        capacity.get("status") == "available"
        and inventory.get("api_confirmed") is True
        and task_inventory.get("api_confirmed") is True
    )
    blockers: list[str] = []
    if not api_verified:
        blockers.append("successor_runpod_api_not_verified")
    if not inventory_zero:
        blockers.append("successor_runpod_inventory_not_zero")
    if not selected:
        blockers.append("successor_compatible_runpod_offer_unavailable")
    result: dict[str, Any] = {
        "schema_version": profile.preflight_schema,
        "status": "verified" if not blockers else "blocked",
        "experiment_id": profile.experiment_id,
        "provider": "runpod",
        "observed_at_epoch": time.time(),
        "blockers": blockers,
        "provider_api_verified": api_verified,
        "provider_inventory_verified_zero": inventory_zero,
        "selected_offer": selected or None,
        "capacity_request": request,
        "capacity_snapshot": capacity,
        "task_billable_inventory": task_inventory,
        "billable_inventory": inventory,
        "provider_mutations_performed": 0,
        "reservation_proven": False,
        "raw_secret_values_recorded": False,
    }
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def inspect_successor_bundle(
    path: str | Path,
    *,
    receipt: Mapping[str, Any] | None = None,
    smoke_inventory: Mapping[str, Any] | None = None,
    profile: SuccessorGPUProfile = LEGACY_PROFILE,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    blockers: list[str] = []
    names: set[str] = set()
    manifest: Mapping[str, Any] = {}
    embedded_hashes: dict[str, str] = {}
    entrypoint_text = ""
    runner_text = ""
    if not resolved.is_file():
        blockers.append("successor_cosmos_provider_bundle_missing")
    else:
        try:
            with zipfile.ZipFile(resolved) as archive:
                names = set(archive.namelist())
                bad_member = archive.testzip()
                if bad_member is not None:
                    blockers.append("successor_cosmos_provider_bundle_zip_invalid")
                if "provider_runtime/wam_provider_runtime_manifest.json" in names:
                    manifest_value = json.loads(
                        archive.read("provider_runtime/wam_provider_runtime_manifest.json").decode(
                            "utf-8"
                        )
                    )
                    manifest = _mapping(manifest_value)
                if "provider_runtime/run_wam_provider_runtime.sh" in names:
                    entrypoint_text = archive.read(
                        "provider_runtime/run_wam_provider_runtime.sh"
                    ).decode("utf-8")
                if "provider_runtime/wam_provider_runtime_runner.py" in names:
                    runner_text = archive.read(
                        "provider_runtime/wam_provider_runtime_runner.py"
                    ).decode("utf-8")
                if profile.powered_bundle:
                    powered_packet = json.loads(
                        archive.read("provider_runtime/cosmos3_powered_droid/packet.json").decode(
                            "utf-8"
                        )
                    )
                    recorded_packet_sha256 = powered_packet.get("manifest_sha256")
                    computed_packet_sha256 = canonical_sha256(
                        {
                            key: value
                            for key, value in powered_packet.items()
                            if key != "manifest_sha256"
                        }
                    )
                    if recorded_packet_sha256 != computed_packet_sha256:
                        blockers.append("successor_powered_packet_hash_invalid")
                    image_manifest: list[dict[str, str]] = []
                    for row in powered_packet.get("rows", []):
                        relative = str(row.get("initial_observation_relative_path") or "")
                        expected = str(row.get("initial_observation_sha256") or "")
                        archive_name = f"provider_runtime/cosmos3_powered_droid/{relative}"
                        if archive_name not in names:
                            blockers.append("successor_powered_image_missing")
                            continue
                        observed_image = hashlib.sha256(archive.read(archive_name)).hexdigest()
                        if observed_image != expected:
                            blockers.append("successor_powered_image_hash_invalid")
                        image_manifest.append({"relative_path": relative, "sha256": expected})
                    canary_manifest = json.loads(
                        archive.read(
                            "provider_runtime/cosmos3_powered_droid/official_canary/"
                            "canary_manifest.json"
                        ).decode("utf-8")
                    )
                    computed_canary_sha256 = canonical_sha256(
                        {
                            key: value
                            for key, value in canary_manifest.items()
                            if key != "manifest_sha256"
                        }
                    )
                    if canary_manifest.get("manifest_sha256") != computed_canary_sha256:
                        blockers.append("successor_powered_canary_manifest_hash_invalid")
                    embedded_hashes = {
                        "provider_packet_sha256": computed_packet_sha256,
                        "image_manifest_sha256": canonical_sha256(image_manifest),
                        "official_canary_manifest_sha256": computed_canary_sha256,
                        "provider_runtime_runner_sha256": hashlib.sha256(
                            archive.read("provider_runtime/wam_provider_runtime_runner.py")
                        ).hexdigest(),
                    }
                elif profile.reference_bundle:
                    reference_manifest = json.loads(
                        archive.read(
                            "provider_runtime/cosmos3_droid_reference/canary_manifest.json"
                        ).decode("utf-8")
                    )
                    recorded_reference_sha256 = reference_manifest.get("manifest_sha256")
                    computed_reference_sha256 = canonical_sha256(
                        {
                            key: value
                            for key, value in reference_manifest.items()
                            if key != "manifest_sha256"
                        }
                    )
                    if recorded_reference_sha256 != computed_reference_sha256:
                        blockers.append("successor_droid_reference_manifest_hash_invalid")
                    embedded_hashes = {
                        "reference_manifest_sha256": computed_reference_sha256,
                        "initial_observation_sha256": hashlib.sha256(
                            archive.read(
                                "provider_runtime/cosmos3_droid_reference/initial_observation.png"
                            )
                        ).hexdigest(),
                        "action_streams_sha256": canonical_sha256(
                            json.loads(
                                archive.read(
                                    "provider_runtime/cosmos3_droid_reference/action_streams.json"
                                ).decode("utf-8")
                            )
                        ),
                        "provider_runtime_runner_sha256": hashlib.sha256(
                            archive.read("provider_runtime/wam_provider_runtime_runner.py")
                        ).hexdigest(),
                    }
                else:
                    embedded_hashes = {
                        "initial_observation_sha256": hashlib.sha256(
                            archive.read("provider_runtime/cosmos3_input/initial_observation.png")
                        ).hexdigest(),
                        "smoke_inventory_sha256": canonical_sha256(
                            json.loads(
                                archive.read(
                                    "provider_runtime/cosmos3_input/smoke_request_inventory.json"
                                ).decode("utf-8")
                            )
                        ),
                        "action_streams_sha256": canonical_sha256(
                            json.loads(
                                archive.read(
                                    "provider_runtime/cosmos3_input/action_streams.json"
                                ).decode("utf-8")
                            )
                        ),
                    }
                positive_control_manifest_entry = (
                    "provider_runtime/cosmos3_positive_control/manifest.json"
                )
                if positive_control_manifest_entry in names:
                    positive_control_manifest = json.loads(
                        archive.read(positive_control_manifest_entry).decode("utf-8")
                    )
                    recorded_positive_control_sha256 = positive_control_manifest.get(
                        "manifest_sha256"
                    )
                    computed_positive_control_sha256 = canonical_sha256(
                        {
                            key: value
                            for key, value in positive_control_manifest.items()
                            if key != "manifest_sha256"
                        }
                    )
                    if recorded_positive_control_sha256 != computed_positive_control_sha256:
                        blockers.append("successor_positive_control_manifest_hash_invalid")
                    embedded_hashes["positive_control_manifest_sha256"] = (
                        computed_positive_control_sha256
                    )
        except (
            OSError,
            KeyError,
            ValueError,
            zipfile.BadZipFile,
            json.JSONDecodeError,
        ):
            blockers.append("successor_cosmos_provider_bundle_unreadable")
    required_entries = (
        POWERED_BUNDLE_ENTRIES
        if profile.powered_bundle
        else REFERENCE_BUNDLE_ENTRIES
        if profile.reference_bundle
        else REQUIRED_BUNDLE_ENTRIES
    )
    missing = sorted(required_entries - names)
    if "positive_control_manifest_sha256" in profile.expected_embedded_input_hashes:
        positive_control_entries = {
            "provider_runtime/cosmos3_positive_control/manifest.json",
            "provider_runtime/cosmos3_positive_control/first_frame.png",
            "provider_runtime/cosmos3_positive_control/action_chunks.json",
            "provider_runtime/cosmos3_positive_control/reference_output.mp4",
        }
        missing.extend(sorted(positive_control_entries - names))
    if missing:
        blockers.append("successor_cosmos_provider_bundle_entries_missing")
    blockers.extend(
        provider_runtime_contract_blockers(
            provider_bundle_kind="wam",
            entrypoint_text=entrypoint_text,
            runner_text=runner_text,
        )
    )
    if manifest.get("schema_version") != BUNDLE_SCHEMA:
        blockers.append("successor_cosmos_provider_bundle_manifest_invalid")
    if manifest.get("experiment_id") != profile.experiment_id:
        blockers.append("successor_cosmos_provider_bundle_experiment_mismatch")
    if manifest.get("checkpoint_revision") != CHECKPOINT_REVISION:
        blockers.append("successor_cosmos_provider_bundle_checkpoint_mismatch")
    if manifest.get("public_image") != PUBLIC_IMAGE:
        blockers.append("successor_cosmos_provider_bundle_image_mismatch")
    expected_request_budget = {
        "qualification_canary_request_count": profile.qualification_canary_request_count,
        "scientific_matrix_request_count": profile.scientific_matrix_request_count,
        "total_initial_generation_request_count": profile.total_initial_generation_request_count,
    }
    if profile.request_budget_amendment_sha256 is not None:
        expected_request_budget["request_budget_amendment_sha256"] = (
            profile.request_budget_amendment_sha256
        )
    if any(manifest.get(key) != value for key, value in expected_request_budget.items()):
        blockers.append("successor_cosmos_provider_bundle_request_budget_mismatch")
    receipt_value = _mapping(receipt)
    bundle_sha256 = _sha256_file(resolved) if resolved.is_file() else None
    bundle_size_bytes = resolved.stat().st_size if resolved.is_file() else 0
    if receipt_value.get("schema_version") != profile.receipt_schema:
        blockers.append("successor_cosmos_provider_bundle_receipt_invalid")
    if receipt_value.get("experiment_id") != profile.experiment_id:
        blockers.append("successor_cosmos_provider_bundle_receipt_experiment_mismatch")
    if receipt_value.get("bundle_sha256") != bundle_sha256:
        blockers.append("successor_cosmos_provider_bundle_receipt_hash_mismatch")
    if receipt_value.get("bundle_size_bytes") != bundle_size_bytes:
        blockers.append("successor_cosmos_provider_bundle_receipt_size_mismatch")
    if bundle_sha256 != profile.expected_bundle_sha256:
        blockers.append("successor_cosmos_provider_bundle_frozen_hash_mismatch")
    if bundle_size_bytes != profile.expected_bundle_size_bytes:
        blockers.append("successor_cosmos_provider_bundle_frozen_size_mismatch")
    for key, expected in profile.expected_embedded_input_hashes.items():
        if embedded_hashes.get(key) != expected or receipt_value.get(key) != expected:
            blockers.append(f"successor_cosmos_provider_bundle_receipt_{key}_mismatch")
    if (
        not profile.reference_bundle
        and not profile.powered_bundle
        and smoke_inventory is not None
        and receipt_value.get("smoke_inventory_sha256") != canonical_sha256(smoke_inventory)
    ):
        blockers.append("successor_cosmos_external_smoke_inventory_hash_mismatch")
    return {
        "status": "passed" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "bundle_path": str(resolved),
        "bundle_sha256": bundle_sha256,
        "bundle_size_bytes": bundle_size_bytes,
        "required_entry_count": len(required_entries),
        "manifest": dict(manifest),
        "embedded_input_hashes": embedded_hashes,
    }


def build_successor_gpu_admission(
    *,
    authorization: Mapping[str, Any],
    environment: Mapping[str, Any],
    smoke_inventory: Mapping[str, Any],
    provider_preflight: Mapping[str, Any],
    bundle_inspection: Mapping[str, Any],
    expected_source_commit: str,
    execute: bool,
    observed_now_epoch: float | None = None,
    initial_blockers: Sequence[str] = (),
    profile: SuccessorGPUProfile = LEGACY_PROFILE,
) -> dict[str, Any]:
    blockers = list(initial_blockers)
    checkpoint = _mapping(environment.get("checkpoint"))
    upstream = _mapping(environment.get("upstream_source"))
    cosmos = _mapping(upstream.get("cosmos"))
    framework = _mapping(upstream.get("cosmos_framework"))
    vllm = _mapping(upstream.get("vllm_omni"))
    if environment.get("experiment_id") != profile.experiment_id:
        blockers.append("successor_environment_experiment_mismatch")
    if cosmos.get("revision") != profile.cosmos_revision:
        blockers.append("successor_cosmos_revision_mismatch")
    if framework.get("revision") != profile.cosmos_framework_revision:
        blockers.append("successor_framework_revision_mismatch")
    if checkpoint.get("repository") != CHECKPOINT_REPOSITORY:
        blockers.append("successor_checkpoint_repository_mismatch")
    if checkpoint.get("revision") != CHECKPOINT_REVISION:
        blockers.append("successor_checkpoint_revision_mismatch")
    if vllm.get("runtime_image") != PUBLIC_IMAGE:
        blockers.append("successor_runtime_image_mismatch")
    if (
        profile.vllm_omni_revision is not None
        and vllm.get("revision") != profile.vllm_omni_revision
    ):
        blockers.append("successor_vllm_omni_revision_mismatch")
    if checkpoint.get("remote_code_policy") != (
        "no_unpinned_remote_code_and_trust_remote_code_false"
    ):
        blockers.append("successor_remote_code_policy_invalid")

    if profile.reference_bundle or profile.powered_bundle:
        inventory_validation = {
            "status": "not_applicable",
            "reason": (
                "powered_droid_inputs_are_frozen_inside_the_bundle"
                if profile.powered_bundle
                else "official_droid_reference_inputs_are_frozen_inside_the_bundle"
            ),
        }
    else:
        try:
            inventory_validation = validate_smoke_inventory_manifest(smoke_inventory)
        except ValueError as exc:
            inventory_validation = {"status": "blocked", "reason": str(exc)}
            blockers.append("successor_smoke_inventory_invalid")
    if bundle_inspection.get("status") != "passed":
        blockers.extend(str(item) for item in bundle_inspection.get("blockers") or [])

    provider_name = str(provider_preflight.get("provider") or "").strip().lower()
    if provider_preflight.get("schema_version") != profile.preflight_schema:
        blockers.append("successor_provider_preflight_schema_invalid")
    if provider_preflight.get("experiment_id") != profile.experiment_id:
        blockers.append("successor_provider_preflight_experiment_mismatch")
    if provider_preflight.get("status") != "verified":
        blockers.append("successor_provider_preflight_not_verified")
    if provider_name not in profile.allowed_providers:
        blockers.append("successor_provider_preflight_provider_invalid")
    if provider_preflight.get("provider_inventory_verified_zero") is not True:
        blockers.append("successor_provider_inventory_not_zero")
    if provider_preflight.get("provider_mutations_performed") != 0:
        blockers.append("successor_provider_preflight_mutation_boundary_invalid")
    offer = _mapping(provider_preflight.get("selected_offer"))
    gpu_name = str(
        offer.get("gpu_name") or offer.get("gpu_type_id") or offer.get("display_name") or ""
    )
    if not any(keyword in gpu_name.upper() for keyword in profile.compatible_gpu_keywords):
        blockers.append("successor_provider_preflight_gpu_not_compatible")
    try:
        gpu_ram_mb = int(offer.get("gpu_ram_mb") or 0)
        hourly_rate = float(
            offer.get("hourly_rate_usd") or offer.get("on_demand_price_usd_per_hour") or 0.0
        )
        reliability = float(offer.get("reliability") or 0.0)
    except (TypeError, ValueError):
        gpu_ram_mb, hourly_rate, reliability = 0, 0.0, 0.0
    minimum_gpu_ram_mb = 80_000 if provider_name == "runpod" else MIN_GPU_RAM_MB
    if gpu_ram_mb < minimum_gpu_ram_mb:
        blockers.append("successor_provider_preflight_gpu_ram_below_minimum")
    if not 0.0 < hourly_rate <= profile.max_hourly_rate_usd:
        blockers.append("successor_provider_preflight_hourly_rate_above_frozen_ceiling")
    if provider_name == "vast" and reliability < MIN_RELIABILITY:
        blockers.append("successor_vast_preflight_reliability_below_frozen_floor")
    if provider_name == "runpod" and (
        offer.get("cloud_type") != "SECURE" or offer.get("capacity_confidence") != "advisory"
    ):
        blockers.append("successor_runpod_secure_advisory_capacity_missing")
    observed = provider_preflight.get("observed_at_epoch")
    now = time.time() if observed_now_epoch is None else float(observed_now_epoch)
    if type(observed) not in {int, float} or not math.isfinite(float(observed)):
        blockers.append("successor_provider_preflight_timestamp_invalid")
    elif execute and not 0.0 <= now - float(observed) <= MAX_PREFLIGHT_AGE_SECONDS:
        blockers.append("successor_provider_preflight_stale_or_future")

    authorization_blockers: list[str] = []
    if authorization.get("schema_version") != profile.authorization_schema:
        authorization_blockers.append("successor_compute_authorization_schema_invalid")
    if authorization.get("experiment_id") != profile.experiment_id:
        authorization_blockers.append("successor_compute_authorization_experiment_mismatch")
    allocation_index = authorization.get("allocation_index")
    expected_authorization_id = (
        profile.authorization_ids_by_allocation_index.get(allocation_index)
        if type(allocation_index) is int
        else None
    )
    if authorization.get("authorization_id") != expected_authorization_id:
        authorization_blockers.append("successor_compute_authorization_id_invalid")
    if authorization.get("maximum_provider_allocations") != 1:
        authorization_blockers.append("successor_compute_authorization_allocation_limit_invalid")
    if authorization.get("single_use_consumption_required") is not True:
        authorization_blockers.append("successor_compute_authorization_single_use_invalid")
    if allocation_index not in profile.authorization_ids_by_allocation_index:
        authorization_blockers.append("successor_compute_authorization_allocation_index_invalid")
    if authorization.get("goal_cost_authorization_amendment_sha256") != (
        profile.cost_authorization_binding_sha256
    ):
        authorization_blockers.append("successor_goal_cost_authorization_binding_invalid")
    if authorization.get("prior_cumulative_compute_cap_superseded") is not True:
        authorization_blockers.append("successor_goal_cost_ceiling_amendment_missing")
    if authorization.get("per_allocation_maximum_spend_required") is not True:
        authorization_blockers.append("successor_per_allocation_spend_limit_missing")
    if authorization.get("paid_mutation_authorized") is not True:
        authorization_blockers.append("successor_compute_not_explicitly_authorized")
    try:
        authorized_cap = float(authorization.get("authorized_compute_cap_usd"))
    except (TypeError, ValueError):
        authorized_cap = math.nan
    if not math.isfinite(authorized_cap) or authorized_cap != profile.max_compute_cap_usd:
        authorization_blockers.append("successor_compute_cap_mismatch")
    required_controls = {
        "one_resource_limit": True,
        "independent_teardown_watchdog": True,
        "watchdog_armed_before_allocation": True,
        "automatic_spend_cutoff": True,
        "teardown_required": True,
        "provider_zero_verification_required": True,
        "physical_robot_endpoint_access_allowed": False,
        "hard_ttl_seconds": profile.hard_ttl_seconds,
    }
    for key, expected in required_controls.items():
        if authorization.get(key) != expected:
            authorization_blockers.append(f"successor_compute_authorization_{key}_invalid")
    blockers.extend(authorization_blockers)

    source_commit = str(expected_source_commit or "").strip().lower()
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        blockers.append("successor_expected_source_commit_invalid")

    resource_class = "runpod_wam_async" if provider_name == "runpod" else "vast_provider_adapter"
    shared = build_paid_lane_admission(resource_class=resource_class, blockers=blockers)
    result: dict[str, Any] = {
        "schema_version": profile.admission_schema,
        "status": "admitted" if not blockers else "blocked",
        "probe_kind": PROBE_KIND,
        "experiment_id": profile.experiment_id,
        "execute_requested": bool(execute),
        "blockers": sorted(set(blockers)),
        "source_commit": source_commit or None,
        "public_image": PUBLIC_IMAGE,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "smoke_inventory_validation": inventory_validation,
        "smoke_inventory_sha256": canonical_sha256(smoke_inventory),
        "provider_preflight_sha256": canonical_sha256(provider_preflight),
        "provider_bundle_sha256": bundle_inspection.get("bundle_sha256"),
        "request_budget": {
            "qualification_canary_request_count": profile.qualification_canary_request_count,
            "scientific_matrix_request_count": profile.scientific_matrix_request_count,
            "total_initial_generation_request_count": profile.total_initial_generation_request_count,
            "amendment_sha256": profile.request_budget_amendment_sha256,
        },
        "selected_offer": dict(offer),
        "provider": provider_name or None,
        "authorization": {
            "status": "accepted" if not authorization_blockers else "blocked",
            "authorized_compute_cap_usd": (
                authorized_cap if math.isfinite(authorized_cap) else None
            ),
            "hard_ttl_seconds": authorization.get("hard_ttl_seconds"),
        },
        "limits": {
            "hard_cap_usd": profile.max_compute_cap_usd,
            "target_spend_usd": profile.target_spend_usd,
            "max_hourly_rate_usd": profile.max_hourly_rate_usd,
            "hard_ttl_seconds": profile.hard_ttl_seconds,
            "one_resource": True,
            "disk_gb": DISK_GB,
            "min_gpu_ram_mb": minimum_gpu_ram_mb,
            "min_reliability": MIN_RELIABILITY if provider_name == "vast" else None,
            "allowed_gpu_keywords": list(profile.compatible_gpu_keywords),
        },
        "shared_paid_lane_admission": shared,
        "provider_mutations_performed": 0,
        "claim_boundary": {
            "admission_is_not_gpu_execution": True,
            "admission_is_not_generated_media": True,
            "admission_is_not_wam_causal_validity": True,
            "physical_robot_access_forbidden": True,
        },
    }
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def _consume_authorization_once(
    authorization: Mapping[str, Any],
    *,
    expected_source_commit: str,
    profile: SuccessorGPUProfile = LEGACY_PROFILE,
) -> dict[str, Any]:
    authorization_id = str(authorization.get("authorization_id") or "")
    allocation_index = authorization.get("allocation_index")
    expected_authorization_id = (
        profile.authorization_ids_by_allocation_index.get(allocation_index)
        if type(allocation_index) is int
        else None
    )
    if authorization_id != expected_authorization_id or not re.fullmatch(
        r"[a-z0-9][a-z0-9-]{15,127}", authorization_id
    ):
        return {
            "status": "blocked",
            "blockers": ["successor_compute_authorization_id_invalid"],
        }
    root = AUTHORIZATION_CONSUMPTION_ROOT
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root_mode = root.stat().st_mode
    except OSError:
        return {
            "status": "blocked",
            "blockers": ["successor_authorization_consumption_root_unavailable"],
        }
    if root_mode & 0o077:
        return {
            "status": "blocked",
            "blockers": ["successor_authorization_consumption_root_insecure"],
        }
    record_path = root / f"{authorization_id}.json"
    record = {
        "schema_version": "policy_ranking_successor_authorization_consumption.v1",
        "authorization_id": authorization_id,
        "authorization_sha256": canonical_sha256(authorization),
        "experiment_id": profile.experiment_id,
        "source_commit": expected_source_commit,
        "consumed_at_epoch": time.time(),
        "maximum_provider_allocations": 1,
    }
    record_bytes = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    record_sha256 = hashlib.sha256(record_bytes).hexdigest()
    temporary_path = root / (f".{authorization_id}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    try:
        descriptor = os.open(
            temporary_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except OSError:
        return {
            "status": "blocked",
            "blockers": ["successor_authorization_consumption_write_failed"],
            "authorization_id": authorization_id,
        }
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(record_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, record_path)
    except FileExistsError:
        return {
            "status": "blocked",
            "blockers": ["successor_compute_authorization_already_consumed"],
            "authorization_id": authorization_id,
        }
    except OSError:
        return {
            "status": "blocked",
            "blockers": ["successor_authorization_consumption_write_failed"],
            "authorization_id": authorization_id,
        }
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
    return {
        "status": "consumed",
        "authorization_id": authorization_id,
        "consumption_record_sha256": record_sha256,
        "record_location_disclosed": False,
    }


RUNPOD_DROID_REFERENCE_PREFIX = "blueprint-groot-oscar-canary-droid-reference-"


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _arm_runpod_successor_watchdog(
    *, job_dir: Path, deadline_epoch: float
) -> tuple[dict[str, Any], subprocess.Popen[str] | None, Path]:
    out_dir = job_dir / "independent_runpod_watchdog"
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "blueprint_pipeline.groot_oscar_runpod_watchdog",
        "--out-dir",
        str(out_dir),
        "--pod-name-prefix",
        RUNPOD_DROID_REFERENCE_PREFIX,
        "--deadline-epoch",
        str(deadline_epoch),
        "--provider",
        "runpod",
    ]
    try:
        process = subprocess.Popen(  # noqa: S603  # nosec B603 - fixed module argv
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        return (
            {
                "status": "blocked",
                "blockers": ["successor_runpod_watchdog_process_start_failed"],
                "error_type": type(exc).__name__,
                "provider_mutations_performed": 0,
            },
            None,
            out_dir,
        )
    evidence_path = out_dir / RUNPOD_WATCHDOG_EVIDENCE_NAME
    until = time.monotonic() + 10.0
    evidence: dict[str, Any] = {}
    while time.monotonic() < until:
        evidence = _read_json_object(evidence_path)
        if (
            evidence.get("status") == "armed"
            and evidence.get("independent_process") is True
            and evidence.get("pid") == process.pid
            and evidence.get("provider") == "runpod"
            and evidence.get("pod_name_prefix") == RUNPOD_DROID_REFERENCE_PREFIX
            and process.poll() is None
        ):
            return evidence, process, out_dir
        if process.poll() is not None:
            break
        time.sleep(0.1)
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    return (
        {
            "status": "blocked",
            "blockers": ["successor_runpod_watchdog_not_confirmed_armed"],
            "provider_mutations_performed": 0,
        },
        None,
        out_dir,
    )


def _stop_unallocated_runpod_watchdog(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _close_runpod_watchdog_after_provider_zero(
    *,
    process: subprocess.Popen[str],
    out_dir: Path,
    pod_id: str,
    provider_zero: bool,
) -> dict[str, Any]:
    if not pod_id or not provider_zero:
        return {
            "status": "retained_until_hard_ttl",
            "provider_absence_confirmed": False,
            "watchdog_process_alive": process.poll() is None,
        }
    write_owner_teardown_cancel_request(
        root=out_dir,
        pod_name_prefix=RUNPOD_DROID_REFERENCE_PREFIX,
        provider_name="runpod",
        instance_id=pod_id,
    )
    until = time.monotonic() + 45.0
    while process.poll() is None and time.monotonic() < until:
        time.sleep(0.2)
    terminal = _read_json_object(out_dir / RUNPOD_WATCHDOG_EVIDENCE_NAME)
    passed = bool(
        terminal.get("status") == "provider_terminal"
        and terminal.get("provider_absence_confirmed") is True
    )
    return {
        "status": "provider_terminal" if passed else "retained_until_hard_ttl",
        "provider_absence_confirmed": passed,
        "watchdog_process_exit_code": process.poll(),
        "watchdog_process_alive": process.poll() is None,
    }


def _run_successor_runpod(
    *,
    job_dir: str | Path,
    provider_bundle_path: str | Path,
    public_base_url: str | None,
    token_file: str | Path | None,
    secret_env_file: str | Path | None,
    provider_bundle_url_file: str | Path | None,
    provider_output_put_url_file: str | Path | None,
    provider_output_get_url_file: str | Path | None,
    output_path: str | Path | None,
    profile: SuccessorGPUProfile,
    selected_offer: Mapping[str, Any],
    session_max_live_minutes: int,
    paid_resource_admission_grant: Any,
    pre_provider_mutation_hook: Any,
) -> dict[str, Any]:
    root = Path(job_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    authorized_live_minutes = min(int(session_max_live_minutes), profile.hard_ttl_seconds // 60)
    deadline = time.time() + authorized_live_minutes * 60
    watchdog, process, watchdog_dir = _arm_runpod_successor_watchdog(
        job_dir=root, deadline_epoch=deadline
    )
    write_json(root / "runpod_successor_watchdog_handoff.json", watchdog)
    if watchdog.get("status") != "armed" or process is None:
        return {
            "status": "blocked",
            "blockers": list(watchdog.get("blockers") or []),
            "provider_mutations_performed": 0,
            "independent_watchdog_handoff": watchdog,
        }
    gpu_name = str(selected_offer.get("gpu_name") or selected_offer.get("gpu_type_id") or "")
    pod_name = f"{RUNPOD_DROID_REFERENCE_PREFIX}{int(time.time())}"
    prior_disable_warm = os.environ.get(RUNPOD_WAM_DISABLE_WARM_CANDIDATE_ENV)
    prior_teardown = os.environ.get(RUNPOD_WAM_TEARDOWN_ACTION_ENV)
    prior_terminal_hold = os.environ.get(RUNPOD_WAM_TERMINAL_HOLD_SECONDS_ENV)
    os.environ[RUNPOD_WAM_DISABLE_WARM_CANDIDATE_ENV] = "1"
    os.environ[RUNPOD_WAM_TEARDOWN_ACTION_ENV] = "delete"
    os.environ[RUNPOD_WAM_TERMINAL_HOLD_SECONDS_ENV] = str(profile.hard_ttl_seconds)
    try:
        create = create_runpod_wam_async_run(
            job_dir=root,
            bundle_path=provider_bundle_path,
            public_base_url=public_base_url or "",
            provider_bundle_url_file=provider_bundle_url_file,
            provider_output_put_url_file=provider_output_put_url_file,
            provider_output_get_url_file=provider_output_get_url_file,
            token_file=token_file,
            secret_env_file=secret_env_file,
            output_path=output_path,
            max_spend_usd=profile.max_compute_cap_usd,
            allow_paid_runpod_launch=True,
            gpu_type_ids=(gpu_name,),
            image_name=PUBLIC_IMAGE,
            provider_bundle_kind="wam",
            container_disk_gb=100,
            volume_gb=20,
            cloud_type="SECURE",
            min_vcpu_per_gpu=8,
            min_ram_per_gpu=32,
            pod_name=pod_name,
            paid_resource_admission_grant=paid_resource_admission_grant,
            forward_model_secret_env=False,
            pre_provider_mutation_hook=pre_provider_mutation_hook,
        )
    finally:
        if prior_disable_warm is None:
            os.environ.pop(RUNPOD_WAM_DISABLE_WARM_CANDIDATE_ENV, None)
        else:
            os.environ[RUNPOD_WAM_DISABLE_WARM_CANDIDATE_ENV] = prior_disable_warm
        if prior_teardown is None:
            os.environ.pop(RUNPOD_WAM_TEARDOWN_ACTION_ENV, None)
        else:
            os.environ[RUNPOD_WAM_TEARDOWN_ACTION_ENV] = prior_teardown
        if prior_terminal_hold is None:
            os.environ.pop(RUNPOD_WAM_TERMINAL_HOLD_SECONDS_ENV, None)
        else:
            os.environ[RUNPOD_WAM_TERMINAL_HOLD_SECONDS_ENV] = prior_terminal_hold
    if create.get("status") != "pod_created":
        _stop_unallocated_runpod_watchdog(process)
        return {
            "status": "blocked",
            "blockers": list(create.get("blockers") or ["successor_runpod_create_failed"]),
            "provider_mutations_performed": int(create.get("provider_mutations_performed") or 0),
            "runpod_create": create,
            "independent_watchdog_handoff": watchdog,
        }
    state = _read_json_object(root / "runpod_wam_async_state.json")
    pod_id = str(state.get("pod_id") or create.get("pod_id") or "")
    poll = poll_runpod_wam_async_run(
        job_dir=root,
        max_wait_seconds=max(60, authorized_live_minutes * 60),
        retry_interval_seconds=5,
        teardown=True,
    )
    provider = get_render_provider("runpod")
    task_inventory = provider.billable_inventory(name_prefix=RUNPOD_DROID_REFERENCE_PREFIX)
    global_inventory = provider.billable_inventory(name_prefix="")
    provider_zero = bool(
        task_inventory.get("api_confirmed") is True
        and task_inventory.get("live_resource_count") == 0
        and global_inventory.get("api_confirmed") is True
        and global_inventory.get("live_resource_count") == 0
    )
    watchdog_close = _close_runpod_watchdog_after_provider_zero(
        process=process,
        out_dir=watchdog_dir,
        pod_id=pod_id,
        provider_zero=provider_zero,
    )
    created_at = float(state.get("created_at_epoch") or time.time())
    runtime_seconds = max(0.0, time.time() - created_at)
    hourly_rate = float(selected_offer.get("hourly_rate_usd") or 0.0)
    estimated_cost = runtime_seconds * hourly_rate / 3600.0
    completed = bool(
        poll.get("status") == "completed"
        and poll.get("continuing_spend_from_this_run") is False
        and provider_zero
        and watchdog_close.get("status") == "provider_terminal"
    )
    return {
        "status": "completed" if completed else "blocked",
        "blockers": []
        if completed
        else [
            *list(poll.get("blockers") or []),
            *([] if provider_zero else ["successor_runpod_provider_zero_not_proven"]),
            *(
                []
                if watchdog_close.get("status") == "provider_terminal"
                else ["successor_runpod_watchdog_not_terminal"]
            ),
        ],
        "provider": "runpod",
        "pod_id": pod_id or None,
        "runpod_create": create,
        "runpod_poll": poll,
        "task_inventory": task_inventory,
        "global_inventory": global_inventory,
        "provider_zero_verified": provider_zero,
        "continuing_hourly_burn": not provider_zero,
        "runtime_seconds": runtime_seconds,
        "selected_hourly_rate_usd": hourly_rate,
        "estimated_gpu_cost_usd": estimated_cost,
        "independent_watchdog_handoff": watchdog,
        "independent_watchdog_close": watchdog_close,
        "provider_mutations_performed": 1,
        "raw_secret_values_recorded": False,
    }


def run_successor_gpu_lane(
    *,
    authorization_path: str | Path,
    environment_path: str | Path,
    smoke_inventory_path: str | Path,
    provider_preflight_path: str | Path,
    provider_bundle_path: str | Path,
    provider_bundle_receipt_path: str | Path,
    admission_out: str | Path,
    bound_request_out: str | Path,
    adapter_output: str | Path,
    job_dir: str | Path,
    public_base_url: str | None,
    token_file: str | Path | None,
    secret_env_file: str | Path | None,
    output_path: str | Path | None,
    session_budget_ledger: str | Path | None,
    expected_source_commit: str,
    execute: bool,
    provider_bundle_url_file: str | Path | None = None,
    provider_output_put_url_file: str | Path | None = None,
    provider_output_get_url_file: str | Path | None = None,
    observed_now_epoch: float | None = None,
) -> dict[str, Any]:
    input_blockers: list[str] = []

    def load_input(label: str, path: str | Path) -> dict[str, Any]:
        try:
            return _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            input_blockers.append(f"successor_{label}_unreadable")
            return {}

    authorization = load_input("authorization", authorization_path)
    environment = load_input("environment", environment_path)
    smoke_inventory = load_input("smoke_inventory", smoke_inventory_path)
    provider_preflight = load_input("provider_preflight", provider_preflight_path)
    bundle_receipt = load_input("bundle_receipt", provider_bundle_receipt_path)
    receipt_schema = bundle_receipt.get("schema_version")
    if receipt_schema == POWERED_DROID_PROFILE.receipt_schema:
        profile = POWERED_DROID_PROFILE
    elif receipt_schema == DROID_REFERENCE_PROFILE.receipt_schema:
        profile = DROID_REFERENCE_PROFILE
    elif receipt_schema == PHASE_B_POSITIVE_CONTROL_PROFILE.receipt_schema:
        profile = PHASE_B_POSITIVE_CONTROL_PROFILE
    elif receipt_schema == PHASE_B_PROFILE.receipt_schema:
        profile = PHASE_B_PROFILE
    else:
        profile = LEGACY_PROFILE
    bundle = inspect_successor_bundle(
        provider_bundle_path,
        receipt=bundle_receipt,
        smoke_inventory=smoke_inventory,
        profile=profile,
    )
    requested_live_minutes = profile.hard_ttl_seconds // 60
    session_limit: dict[str, Any] = {
        "schema_version": "vast_successor_session_live_limit.v1",
        "status": "blocked",
        "blockers": ["successor_session_budget_ledger_missing"],
        "session_max_live_runtime_minutes": requested_live_minutes,
        "raw_secret_values_recorded": False,
    }
    session_guard: dict[str, Any] = {
        "schema_version": "vast_session_budget_guard.v1",
        "status": "blocked",
        "blockers": ["successor_session_budget_ledger_missing"],
        "raw_secret_values_recorded": False,
    }
    if session_budget_ledger is not None:
        budget_path = Path(session_budget_ledger).expanduser().resolve()
        session_limit = successor_session_live_limit_minutes(
            budget_path=budget_path,
            requested_max_live_minutes=requested_live_minutes,
        )
        session_guard = build_vast_session_budget_guard(
            generated_at=utc_now_iso(),
            budget_path=budget_path,
            session_max_live_minutes=int(session_limit["session_max_live_runtime_minutes"]),
            requested_max_live_minutes=requested_live_minutes,
            target_spend_usd=profile.target_spend_usd,
            hard_cap_usd=profile.max_compute_cap_usd,
            max_hourly_rate=profile.max_hourly_rate_usd,
        )
        write_json(
            Path(job_dir).expanduser().resolve() / "vast_session_budget_guard.json",
            session_guard,
        )
        if not budget_path.is_file():
            input_blockers.append("successor_session_budget_ledger_missing")
        input_blockers.extend(str(item) for item in session_limit.get("blockers") or [])
        input_blockers.extend(str(item) for item in session_guard.get("blockers") or [])
    else:
        input_blockers.append("successor_session_budget_ledger_missing")
    admission = build_successor_gpu_admission(
        authorization=authorization,
        environment=environment,
        smoke_inventory=smoke_inventory,
        provider_preflight=provider_preflight,
        bundle_inspection=bundle,
        expected_source_commit=expected_source_commit,
        execute=execute,
        observed_now_epoch=observed_now_epoch,
        initial_blockers=input_blockers,
        profile=profile,
    )
    admission["session_live_limit"] = session_limit
    admission["session_budget_preflight"] = session_guard
    admission["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in admission.items() if key != "manifest_sha256"}
    )
    write_json(Path(admission_out), admission)
    bound = {
        "schema_version": "policy_ranking_successor_bound_gpu_request.v1",
        "status": "bound" if admission["status"] == "admitted" else "blocked",
        "experiment_id": profile.experiment_id,
        "source_commit": expected_source_commit,
        "provider": str(provider_preflight.get("provider") or "").strip().lower(),
        "probe_kind": PROBE_KIND,
        "public_image": PUBLIC_IMAGE,
        "provider_bundle_sha256": bundle.get("bundle_sha256"),
        "smoke_inventory_sha256": canonical_sha256(smoke_inventory),
        "request_budget": admission.get("request_budget"),
        "selected_offer_id": _mapping(admission.get("selected_offer")).get("ask_contract_id"),
        "limits": admission["limits"],
        "session_live_limit": session_limit,
        "session_budget_preflight": session_guard,
        "blockers": admission["blockers"],
        "provider_mutations_performed": 0,
    }
    bound["manifest_sha256"] = canonical_sha256(bound)
    write_json(Path(bound_request_out), bound)
    if not execute:
        dry_ready = admission["status"] == "admitted"
        result = {
            "status": "dry_run_ready" if dry_ready else "blocked",
            "reason": (
                "admission_validated_without_provider_mutation"
                if dry_ready
                else "paid_execution_not_requested_and_admission_blocked"
            ),
            "blockers": admission["blockers"],
            "provider_mutations_performed": 0,
            "admission_status": admission["status"],
        }
        write_json(Path(adapter_output), result)
        return result
    try:
        provider_name = str(provider_preflight.get("provider") or "").strip().lower()
        resource_class = (
            "runpod_wam_async" if provider_name == "runpod" else "vast_provider_adapter"
        )
        grant = require_paid_resource_admission(
            admission["shared_paid_lane_admission"],
            resource_class=resource_class,
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
        )
    except PaidResourceAdmissionBlocked as exc:
        result = {
            "status": "blocked",
            "reason": "shared_paid_resource_admission_blocked",
            "blockers": [*admission["blockers"], *exc.blockers],
            "provider_mutations_performed": 0,
        }
        write_json(Path(adapter_output), result)
        return result
    if provider_name == "runpod":
        consumption: dict[str, Any] = {
            "status": "not_consumed",
            "reason": "awaiting_verified_staging_and_selected_offer",
            "provider_mutations_performed": 0,
        }

        def consume_immediately_before_runpod_mutation() -> Mapping[str, Any]:
            nonlocal consumption
            observed_preflight_epoch = provider_preflight.get("observed_at_epoch")
            mutation_now_epoch = time.time()
            if (
                type(observed_preflight_epoch) not in {int, float}
                or not math.isfinite(float(observed_preflight_epoch))
                or not 0.0
                <= mutation_now_epoch - float(observed_preflight_epoch)
                <= MAX_PREFLIGHT_AGE_SECONDS
            ):
                consumption = {
                    "status": "blocked",
                    "blockers": ["successor_runpod_preflight_stale_or_future_at_provider_mutation"],
                    "provider_mutations_performed": 0,
                }
                return consumption
            consumption = _consume_authorization_once(
                authorization,
                expected_source_commit=expected_source_commit,
                profile=profile,
            )
            return consumption

        result = _run_successor_runpod(
            job_dir=job_dir,
            provider_bundle_path=provider_bundle_path,
            public_base_url=public_base_url,
            token_file=token_file,
            secret_env_file=secret_env_file,
            provider_bundle_url_file=provider_bundle_url_file,
            provider_output_put_url_file=provider_output_put_url_file,
            provider_output_get_url_file=provider_output_get_url_file,
            output_path=output_path,
            profile=profile,
            selected_offer=_mapping(admission.get("selected_offer")),
            session_max_live_minutes=int(session_limit["session_max_live_runtime_minutes"]),
            paid_resource_admission_grant=grant,
            pre_provider_mutation_hook=consume_immediately_before_runpod_mutation,
        )
        if consumption["status"] != "consumed":
            blockers = [str(item) for item in result.get("blockers") or []]
            if result.get("status") in {"completed", "retained_owned"}:
                blockers.append(
                    "successor_compute_authorization_not_consumed_before_provider_mutation"
                )
                result["status"] = "blocked"
            result["blockers"] = sorted(set(blockers))
        else:
            admission["authorization_consumption"] = consumption
            admission["manifest_sha256"] = canonical_sha256(
                {key: value for key, value in admission.items() if key != "manifest_sha256"}
            )
            write_json(Path(admission_out), admission)
            bound["authorization_consumption"] = consumption
            bound["manifest_sha256"] = canonical_sha256(
                {key: value for key, value in bound.items() if key != "manifest_sha256"}
            )
            write_json(Path(bound_request_out), bound)
        result["authorization_consumption"] = consumption
        write_json(Path(adapter_output), result)
        return result
    if provider_name != "vast":
        result = {
            "status": "blocked",
            "reason": "successor_provider_executor_not_supported",
            "blockers": ["successor_provider_executor_not_supported"],
            "authorization_consumed": False,
            "provider_mutations_performed": 0,
        }
        write_json(Path(adapter_output), result)
        return result
    launch_gate_blockers = [
        f"missing_env_{name}"
        for name in (VAST_API_GATE_ENV, VAST_INSTANCE_LAUNCH_GATE_ENV)
        if not _vast_env_truthy(name)
    ]
    if launch_gate_blockers:
        result = {
            "status": "blocked",
            "reason": "provider_launch_env_gate_blocked_before_authorization_consumption",
            "blockers": launch_gate_blockers,
            "authorization_consumed": False,
            "provider_mutations_performed": 0,
        }
        write_json(Path(adapter_output), result)
        return result
    consumption: dict[str, Any] = {
        "status": "not_consumed",
        "reason": "awaiting_verified_staging_and_selected_offer",
        "provider_mutations_performed": 0,
    }

    def consume_immediately_before_provider_mutation() -> Mapping[str, Any]:
        nonlocal consumption
        observed_preflight_epoch = provider_preflight.get("observed_at_epoch")
        mutation_now_epoch = time.time()
        if (
            type(observed_preflight_epoch) not in {int, float}
            or not math.isfinite(float(observed_preflight_epoch))
            or not 0.0
            <= mutation_now_epoch - float(observed_preflight_epoch)
            <= MAX_PREFLIGHT_AGE_SECONDS
        ):
            consumption = {
                "status": "blocked",
                "blockers": ["successor_vast_preflight_stale_or_future_at_provider_mutation"],
                "provider_mutations_performed": 0,
            }
            return consumption
        consumption = _consume_authorization_once(
            authorization,
            expected_source_commit=expected_source_commit,
            profile=profile,
        )
        return consumption

    result = run_vast_wam_authorized_runner(
        job_dir=job_dir,
        bundle_path=provider_bundle_path,
        public_base_url=public_base_url,
        token_file=token_file,
        secret_env_file=secret_env_file,
        provider_bundle_url_file=provider_bundle_url_file,
        provider_output_put_url_file=provider_output_put_url_file,
        provider_output_get_url_file=provider_output_get_url_file,
        output_path=output_path,
        session_budget_ledger=session_budget_ledger,
        allow_paid_vast_launch=True,
        max_hourly_rate=profile.max_hourly_rate_usd,
        target_spend_usd=profile.target_spend_usd,
        hard_cap_usd=profile.max_compute_cap_usd,
        max_live_minutes=requested_live_minutes,
        session_max_live_minutes=int(session_limit["session_max_live_runtime_minutes"]),
        startup_timeout_seconds=3600,
        public_image=PUBLIC_IMAGE,
        disk_gb=DISK_GB,
        min_gpu_ram_mb=MIN_GPU_RAM_MB,
        min_compute_cap=1200,
        max_compute_cap=0,
        min_reliability=MIN_RELIABILITY,
        preferred_gpu_keywords=RTX_ALLOWED_KEYWORDS,
        prefer_isaac_rt=False,
        gpu_selection_policy=RTX_SELECTION_POLICY,
        require_independent_watchdog=True,
        retain_instance_on_runtime_failure=True,
        # Cosmos3-Nano at the frozen revision is publicly downloadable.  Do not
        # put an unrelated account credential into provider-visible instance
        # environment metadata.
        forward_hf_token=False,
        retention_binding={
            "source_commit": expected_source_commit,
            "dirty_state_declaration": "clean_exact_commit",
            "bundle_sha256": bundle["bundle_sha256"],
            "authorization_receipt_sha256": _sha256_file(
                Path(provider_bundle_receipt_path).expanduser().resolve()
            ),
            "image_digest": VLLM_IMAGE_DIGEST,
            "checkpoint": CHECKPOINT_REPOSITORY,
            "checkpoint_revision": CHECKPOINT_REVISION,
        },
        paid_resource_admission_grant=grant,
        pre_provider_mutation_hook=consume_immediately_before_provider_mutation,
    )
    if consumption["status"] != "consumed":
        blockers = [str(item) for item in result.get("blockers") or []]
        if result.get("status") in {"completed", "retained_owned"}:
            blockers.append("successor_compute_authorization_not_consumed_before_provider_mutation")
            result["status"] = "blocked"
        result["blockers"] = sorted(set(blockers))
    else:
        admission["authorization_consumption"] = consumption
        admission["manifest_sha256"] = canonical_sha256(
            {key: value for key, value in admission.items() if key != "manifest_sha256"}
        )
        write_json(Path(admission_out), admission)
        bound["authorization_consumption"] = consumption
        bound["manifest_sha256"] = canonical_sha256(
            {key: value for key, value in bound.items() if key != "manifest_sha256"}
        )
        write_json(Path(bound_request_out), bound)
    result["authorization_consumption"] = consumption
    write_json(Path(adapter_output), result)
    return result


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "HARD_TTL_SECONDS",
    "MAX_COMPUTE_CAP_USD",
    "PREFLIGHT_SCHEMA",
    "DROID_REFERENCE_PROFILE",
    "POWERED_DROID_PROFILE",
    "PHASE_B_PROFILE",
    "PHASE_B_POSITIVE_CONTROL_PROFILE",
    "PROBE_KIND",
    "PUBLIC_IMAGE",
    "SuccessorGPUProfile",
    "build_successor_gpu_admission",
    "collect_successor_vast_preflight",
    "collect_successor_runpod_preflight",
    "inspect_successor_bundle",
    "run_successor_gpu_lane",
]
