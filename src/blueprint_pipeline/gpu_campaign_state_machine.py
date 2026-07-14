"""Provider-neutral, checkpointed control plane for paid GPU campaigns.

The public interface is intentionally small: construct :class:`CampaignConfig`,
provide a :class:`CampaignProviderAdapter`, and call :meth:`CampaignMachine.run`.
Every externally visible transition is persisted before the next stage starts.

The module proves control-plane behavior only. A completed stage is not policy,
media, semantic-task, safety, deployment, or physical-robot proof.
"""

from __future__ import annotations

import hashlib
import json
import fcntl
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


SCHEMA_VERSION = "provider_neutral_gpu_campaign.v1"
CONFIG_SCHEMA_VERSION = "provider_neutral_gpu_campaign_config.v1"
LARGE_IMAGE_PRELOAD_THRESHOLD_BYTES = 20 * 1024**3


class CampaignBlocked(RuntimeError):
    """A fail-closed campaign terminal condition."""


class CampaignProviderAdapter(Protocol):
    """The provider seam. Production and in-memory adapters share it."""

    provider_name: str

    def inventory(self, allocation_key: str) -> Sequence[Mapping[str, Any]]: ...

    def allocate(self, config: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def run_stage(
        self,
        allocation_id: str,
        stage: str,
        *,
        deadline_seconds: int,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def retrieve(self, allocation_id: str, config: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def terminate(self, allocation_id: str) -> Mapping[str, Any]: ...

    def inspect(self, allocation_id: str) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class CampaignConfig:
    campaign_id: str
    allocation_key: str
    source_sha: str
    image_digest: str
    hourly_rate_usd: float
    max_provider_seconds: int
    spend_authorization_usd: float
    image_total_compressed_bytes: int
    image_largest_layer_bytes: int
    image_residency_evidence: Mapping[str, Any] | None
    prior_exposure_usd: float = 0.0
    smoke_seed: int = 1000
    episode_seeds: tuple[int, ...] = (1001, 1002, 1003)
    stage_deadlines_seconds: Mapping[str, int] = field(
        default_factory=lambda: {
            "host_ready": 600,
            "image_ready": 1200,
            "runtime_health": 300,
            "canary": 300,
            "smoke": 300,
            "episodes": 2700,
            "artifact_retrieval": 300,
            "teardown": 300,
        }
    )
    reuse_validated_same_allocation_canary: bool = False
    canary_handoff: Mapping[str, Any] | None = None

    def payload(self) -> dict[str, Any]:
        result = asdict(self)
        result["schema_version"] = CONFIG_SCHEMA_VERSION
        result["episode_seeds"] = list(self.episode_seeds)
        return result

    def validate(self) -> list[str]:
        blockers: list[str] = []
        if not self.campaign_id.strip():
            blockers.append("campaign_id_missing")
        if not self.allocation_key.strip():
            blockers.append("allocation_key_missing")
        if len(self.source_sha) != 40:
            blockers.append("source_sha_invalid")
        if not self.image_digest.startswith("sha256:") or len(self.image_digest) != 71:
            blockers.append("image_digest_invalid")
        if self.hourly_rate_usd <= 0 or self.max_provider_seconds <= 0:
            blockers.append("paid_runtime_bound_invalid")
        if self.image_total_compressed_bytes <= 0 or self.image_largest_layer_bytes <= 0:
            blockers.append("image_closure_size_missing")
        elif self.image_largest_layer_bytes > self.image_total_compressed_bytes:
            blockers.append("image_largest_layer_exceeds_total")
        if self.image_total_compressed_bytes >= LARGE_IMAGE_PRELOAD_THRESHOLD_BYTES:
            blockers.extend(validate_preloaded_image_evidence(self, self.image_residency_evidence))
        maximum = self.prior_exposure_usd + (
            self.hourly_rate_usd * self.max_provider_seconds / 3600
        )
        if maximum > self.spend_authorization_usd:
            blockers.append("campaign_maximum_exceeds_spend_authorization")
        required = {
            "host_ready",
            "image_ready",
            "runtime_health",
            "canary",
            "smoke",
            "episodes",
            "artifact_retrieval",
            "teardown",
        }
        if required - set(self.stage_deadlines_seconds):
            blockers.append("stage_deadlines_incomplete")
        if any(int(value) <= 0 for value in self.stage_deadlines_seconds.values()):
            blockers.append("stage_deadline_invalid")
        if len(set((self.smoke_seed, *self.episode_seeds))) != 1 + len(self.episode_seeds):
            blockers.append("campaign_seeds_not_independent")
        if self.reuse_validated_same_allocation_canary:
            blockers.extend(validate_same_allocation_canary_handoff(self, self.canary_handoff))
        return blockers


def validate_preloaded_image_evidence(
    config: CampaignConfig, evidence: Mapping[str, Any] | None
) -> list[str]:
    """Require a large exact digest to be resident before paid allocation.

    A registry manifest is intentionally insufficient: it proves availability,
    not that a provider host can start the image without a paid cold pull.
    """

    if not isinstance(evidence, Mapping):
        return ["large_image_preload_evidence_missing"]
    blockers: list[str] = []
    if evidence.get("schema_version") != "preloaded_worker_image.v1":
        blockers.append("large_image_preload_schema_invalid")
    for key, expected in {
        "source_sha": config.source_sha,
        "image_digest": config.image_digest,
        "allocation_key": config.allocation_key,
    }.items():
        if evidence.get(key) != expected:
            blockers.append(f"large_image_preload_{key}_mismatch")
    if not str(evidence.get("host_image_id") or "").strip():
        blockers.append("large_image_preload_host_image_id_missing")
    for field_name in (
        "image_present_before_allocation",
        "local_digest_inspect_passed",
        "runtime_health_preflight_passed",
    ):
        if evidence.get(field_name) is not True:
            blockers.append(f"large_image_preload_{field_name}_not_proven")
    if evidence.get("cold_pull_required_during_campaign") is not False:
        blockers.append("large_image_cold_pull_still_required")
    return blockers


def validate_image_ready_result(
    config: CampaignConfig, result: Mapping[str, Any]
) -> list[str]:
    """Re-verify residency on the allocated host before runtime startup."""

    blockers: list[str] = []
    if result.get("image_digest") != config.image_digest:
        blockers.append("image_ready_digest_mismatch")
    if result.get("local_digest_inspect_passed") is not True:
        blockers.append("image_ready_local_digest_inspect_not_passed")
    if config.image_total_compressed_bytes >= LARGE_IMAGE_PRELOAD_THRESHOLD_BYTES:
        if result.get("digest_already_local_at_allocation") is not True:
            blockers.append("large_image_not_local_at_allocation")
        if result.get("cold_pull_performed_during_campaign") is not False:
            blockers.append("large_image_paid_cold_pull_detected")
        evidence = config.image_residency_evidence or {}
        if result.get("host_image_id") != evidence.get("host_image_id"):
            blockers.append("large_image_host_image_identity_mismatch")
    return blockers


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_same_allocation_canary_handoff(
    config: CampaignConfig, evidence: Mapping[str, Any] | None
) -> list[str]:
    """Validate the explicit schema for canary-to-smoke allocation reuse."""

    if not isinstance(evidence, Mapping):
        return ["same_allocation_canary_handoff_missing"]
    blockers: list[str] = []
    if evidence.get("schema_version") != "same_allocation_canary_handoff.v1":
        blockers.append("same_allocation_canary_handoff_schema_invalid")
    expected = {
        "source_sha": config.source_sha,
        "image_digest": config.image_digest,
        "allocation_key": config.allocation_key,
    }
    for key, value in expected.items():
        if evidence.get(key) != value:
            blockers.append(f"same_allocation_canary_{key}_mismatch")
    if not str(evidence.get("allocation_id") or "").strip():
        blockers.append("same_allocation_canary_allocation_id_missing")
    if not str(evidence.get("launch_nonce") or "").strip():
        blockers.append("same_allocation_canary_launch_nonce_missing")
    if not str(evidence.get("teardown_owner") or "").strip():
        blockers.append("same_allocation_canary_teardown_owner_missing")
    if evidence.get("runtime_health_passed") is not True:
        blockers.append("same_allocation_canary_runtime_health_not_passed")
    if evidence.get("review_media_valid") is not True:
        blockers.append("same_allocation_canary_review_media_not_valid")
    if evidence.get("allocation_still_owned") is not True:
        blockers.append("same_allocation_canary_allocation_not_owned")
    if evidence.get("teardown_requested") is not False:
        blockers.append("same_allocation_canary_teardown_already_requested")
    return blockers


def validate_smoke_result(result: Mapping[str, Any]) -> list[str]:
    """Require real simulator/policy evidence before episode admission."""

    blockers: list[str] = []
    if result.get("status") not in {"passed", "completed"}:
        blockers.append("smoke_status_not_passed")
    if result.get("command_return_code") != 0:
        blockers.append("smoke_command_return_code_not_zero")
    if int(result.get("simulator_steps") or 0) < 3:
        blockers.append("smoke_real_simulator_steps_below_three")
    for evidence_field in (
        "manifest_valid",
        "learned_policy_request_response_valid",
        "fresh_policy_conditioning_valid",
        "action_trace_nonempty",
        "real_task_executor_measurement",
        "artifact_output_present",
    ):
        if result.get(evidence_field) is not True:
            blockers.append(f"smoke_{evidence_field}_not_proven")
    if int(result.get("learned_policy_action_count") or 0) < 3:
        blockers.append("smoke_learned_policy_actions_below_three")
    sources = [str(source).lower() for source in result.get("action_sources") or []]
    if not sources or any("surrogate" in source or "fixture" in source for source in sources):
        blockers.append("smoke_action_sources_not_real")
    return blockers


class CampaignMachine:
    """Run or resume one campaign while preserving one teardown owner."""

    _STAGES = (
        "host_ready",
        "image_ready",
        "runtime_health",
        "canary",
        "smoke",
        "episodes",
        "artifact_retrieval",
    )

    def __init__(
        self,
        *,
        config: CampaignConfig,
        adapter: CampaignProviderAdapter,
        state_dir: str | Path,
        teardown_owner: str | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        self.config = config
        self.adapter = adapter
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "campaign_state.json"
        self.config_path = self.state_dir / "immutable_config_manifest.json"
        recorded_owner = ""
        if teardown_owner is None and self.state_path.exists():
            try:
                recorded_owner = str(
                    json.loads(self.state_path.read_text()).get("teardown_owner") or ""
                )
            except (OSError, json.JSONDecodeError):
                recorded_owner = ""
        handoff_owner = ""
        if self.config.reuse_validated_same_allocation_canary and isinstance(
            self.config.canary_handoff, Mapping
        ):
            handoff_owner = str(self.config.canary_handoff.get("teardown_owner") or "")
        self.teardown_owner = (
            teardown_owner or recorded_owner or handoff_owner or str(uuid.uuid4())
        )
        self.clock = clock

    def _write(self, path: Path, payload: Mapping[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)

    def _initial_state(self, config_sha: str) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": self.config.campaign_id,
            "provider": self.adapter.provider_name,
            "config_sha256": config_sha,
            "teardown_owner": self.teardown_owner,
            "allocation_id": None,
            "completed_stages": [],
            "stage_results": {},
            "status": "running",
            "blockers": [],
            "teardown": None,
            "final_inventory": None,
            "claim_boundary": {
                "control_plane_completion_is_not_semantic_task_success": True,
                "simulator_evidence_is_not_physical_robot_readiness": True,
            },
        }

    def _load(self) -> tuple[dict[str, Any], str]:
        config_payload = self.config.payload()
        config_sha = _canonical_sha(config_payload)
        if self.config_path.exists():
            recorded = json.loads(self.config_path.read_text())
            if recorded.get("config_sha256") != config_sha:
                raise CampaignBlocked("immutable_campaign_config_changed")
        else:
            self._write(
                self.config_path,
                {"config": config_payload, "config_sha256": config_sha},
            )
        if not self.state_path.exists():
            return self._initial_state(config_sha), config_sha
        state = json.loads(self.state_path.read_text())
        if state.get("config_sha256") != config_sha:
            raise CampaignBlocked("campaign_checkpoint_config_mismatch")
        if state.get("teardown_owner") != self.teardown_owner:
            raise CampaignBlocked("campaign_teardown_owned_by_another_controller")
        return state, config_sha

    def _checkpoint(self, state: dict[str, Any], stage: str, result: Mapping[str, Any]) -> None:
        state["stage_results"][stage] = dict(result)
        if stage not in state["completed_stages"]:
            state["completed_stages"].append(stage)
        self._write(self.state_path, state)

    @staticmethod
    def _stage_passed(result: Mapping[str, Any]) -> bool:
        return result.get("status") in {"passed", "completed", "ready", "retrieved"}

    def _teardown(self, state: dict[str, Any]) -> None:
        allocation_id = str(state.get("allocation_id") or "")
        if not allocation_id:
            state["teardown"] = {"status": "not_required", "billing_stopped": True}
            state["final_inventory"] = list(self.adapter.inventory(self.config.allocation_key))
            self._write(self.state_path, state)
            return
        teardown = dict(self.adapter.terminate(allocation_id))
        inspection = dict(self.adapter.inspect(allocation_id))
        absent = inspection.get("absent") is True or inspection.get("http") == 404
        final_inventory = list(self.adapter.inventory(self.config.allocation_key))
        state["teardown"] = {
            "status": "passed" if absent and not final_inventory else "blocked",
            "delete_request": teardown,
            "provider_inspection": inspection,
            "billing_stopped": absent,
            "teardown_owner": self.teardown_owner,
        }
        state["final_inventory"] = final_inventory
        if not absent:
            state["blockers"].append("provider_teardown_ambiguous")
        if final_inventory:
            state["blockers"].append("provider_final_inventory_not_zero")
        self._write(self.state_path, state)

    def run(self) -> dict[str, Any]:
        """Run with an OS-released exclusive lock as the teardown owner."""

        lock_path = self.state_dir / "campaign_controller.lock"
        lock_handle = lock_path.open("a+")
        try:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CampaignBlocked("campaign_controller_already_running") from exc
            return self._run_owned()
        finally:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                lock_handle.close()

    def _run_owned(self) -> dict[str, Any]:
        blockers = self.config.validate()
        if blockers:
            raise CampaignBlocked(",".join(blockers))
        state, _ = self._load()
        if state.get("status") in {"completed", "blocked"}:
            return state
        started = self.clock()
        try:
            if not state.get("allocation_id"):
                inventory = list(self.adapter.inventory(self.config.allocation_key))
                if self.config.reuse_validated_same_allocation_canary:
                    handoff = self.config.canary_handoff or {}
                    allocation_id = str(handoff.get("allocation_id") or "").strip()
                    if str(handoff.get("teardown_owner") or "") != self.teardown_owner:
                        raise CampaignBlocked("same_allocation_teardown_owner_mismatch")
                    inventory_ids = {
                        str(item.get("allocation_id") or item.get("id") or "").strip()
                        for item in inventory
                    }
                    if inventory_ids != {allocation_id}:
                        raise CampaignBlocked(
                            "same_allocation_handoff_inventory_mismatch"
                        )
                    allocation = {
                        "status": "completed",
                        "allocation_id": allocation_id,
                        "adopted_canary_handoff": True,
                        "launch_nonce": handoff.get("launch_nonce"),
                        "teardown_owner": self.teardown_owner,
                    }
                else:
                    if inventory:
                        raise CampaignBlocked("duplicate_paid_allocation_detected")
                    allocation = dict(self.adapter.allocate(self.config.payload()))
                    allocation_id = str(allocation.get("allocation_id") or "").strip()
                    if not allocation_id:
                        raise CampaignBlocked("provider_allocation_id_missing")
                state["allocation_id"] = allocation_id
                self._checkpoint(state, "allocation", allocation)

            allocation_id = str(state["allocation_id"])
            if self.config.reuse_validated_same_allocation_canary:
                handoff = self.config.canary_handoff or {}
                if allocation_id != str(handoff.get("allocation_id") or ""):
                    raise CampaignBlocked("same_allocation_handoff_checkpoint_mismatch")
                if self.teardown_owner != str(handoff.get("teardown_owner") or ""):
                    raise CampaignBlocked("same_allocation_teardown_owner_mismatch")
            for stage in self._STAGES:
                if stage in state["completed_stages"]:
                    continue
                stage_started = self.clock()
                if stage == "canary" and self.config.reuse_validated_same_allocation_canary:
                    result = {"status": "passed", "reused_handoff": True}
                elif stage == "artifact_retrieval":
                    result = dict(self.adapter.retrieve(allocation_id, self.config.payload()))
                else:
                    result = dict(
                        self.adapter.run_stage(
                            allocation_id,
                            stage,
                            deadline_seconds=int(self.config.stage_deadlines_seconds[stage]),
                            config=self.config.payload(),
                        )
                    )
                elapsed = self.clock() - stage_started
                if elapsed > int(self.config.stage_deadlines_seconds[stage]):
                    raise CampaignBlocked(f"campaign_stage_deadline_exceeded:{stage}")
                result.setdefault("elapsed_seconds", round(elapsed, 3))
                if not self._stage_passed(result):
                    raise CampaignBlocked(f"campaign_stage_failed:{stage}")
                if stage == "image_ready":
                    image_blockers = validate_image_ready_result(self.config, result)
                    if image_blockers:
                        raise CampaignBlocked(
                            "image_ready_gate_blocked:" + ",".join(image_blockers)
                        )
                if stage == "smoke":
                    smoke_blockers = validate_smoke_result(result)
                    if smoke_blockers:
                        raise CampaignBlocked("smoke_gate_blocked:" + ",".join(smoke_blockers))
                self._checkpoint(state, stage, result)
                if self.clock() - started > self.config.max_provider_seconds:
                    raise CampaignBlocked("campaign_provider_lifetime_exceeded")
            state["status"] = "completed"
        except Exception as exc:  # fail closed, including provider ambiguity
            state["status"] = "blocked"
            blocker = (
                str(exc) if isinstance(exc, CampaignBlocked) else f"{type(exc).__name__}:{exc}"
            )
            if blocker not in state["blockers"]:
                state["blockers"].append(blocker)
        finally:
            self._write(self.state_path, state)
            self._teardown(state)
            if state["teardown"]["status"] != "passed":
                state["status"] = "blocked"
            self._write(self.state_path, state)
        return state
