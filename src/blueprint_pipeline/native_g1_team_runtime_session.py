"""Open one reviewed team policy through the same G1 observation/action client.

This is the worker-side transport seam. Synthetic conformance runs before the
client can receive site observations. The caller still owns rights, site-data
authorization, paid admission, scene scoring, and provider teardown.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_g1_team_artifact_runtime import launch_g1_team_artifact_synthetic_probe
from .native_g1_team_container_runtime import launch_g1_team_container_synthetic_probe
from .native_g1_team_policy_conformance import run_g1_team_policy_synthetic_conformance
from .native_g1_team_policy_https_client import NativeG1TeamPolicyHttpsClient
from .native_g1_shared_scene_episode import team_policy_candidate_id
from .team_policy_delivery_profile import validate_team_policy_delivery_profile


SESSION_SCHEMA = "native_g1_team_runtime_session.v1"
CONFORMANCE_FILENAME = "native_g1_team_policy_synthetic_conformance.v1.json"


class NativeG1TeamRuntimeSession:
    """Keep the qualified client and its child-process cleanup together."""

    def __init__(
        self, *, client: Any, conformance: dict[str, Any],
        profile_digest: str, delivery_mode: str, output_dir: Path, lease: Any = None,
    ) -> None:
        self.client = client
        self.conformance = conformance
        self.profile_digest = profile_digest
        self.delivery_mode = delivery_mode
        self.output_dir = output_dir
        self._lease = lease
        self._closed: dict[str, Any] | None = None
        self._linked_episode_digest: str | None = None

    def link_scored_episode(self, result: Mapping[str, Any]) -> None:
        """Bind a caller-produced episode digest without claiming to verify media."""

        if (
            self._closed is not None or self._linked_episode_digest is not None
            or not isinstance(result, Mapping)
            or result.get("schema_version") != "native_g1_team_scored_scene_episode.v1"
            or result.get("status") != "development_only_scored_episode"
            or result.get("profile_digest") != self.profile_digest
            or result.get("candidate_id") != team_policy_candidate_id(self.profile_digest)
            or result.get("source_setup_digest") != self.conformance["source_setup_digest"]
            or result.get("delivery_mode") != self.delivery_mode
            or type(result.get("policy_query_count")) is not int
            or result["policy_query_count"] < 1
            or result.get("result_digest")
            != canonical_digest(result, digest_field="result_digest")
        ):
            raise ValueError("g1_team_runtime_episode_link_invalid")
        self._linked_episode_digest = result["result_digest"]

    def close(self) -> dict[str, Any]:
        if self._closed is not None:
            return self._closed
        teardown = self._lease.close() if self._lease is not None else None
        closed = {
            "schema_version": SESSION_SCHEMA,
            "status": (
                "closed" if teardown is None or teardown.get("status") in
                {"container_removed", "process_exited"} else "teardown_blocked"
            ),
            "profile_digest": self.profile_digest,
            "delivery_mode": self.delivery_mode,
            "synthetic_conformance_digest": self.conformance["receipt_digest"],
            "child_teardown_digest": teardown.get("receipt_digest") if teardown else None,
            "child_teardown_required": teardown is not None,
            "linked_scored_episode_result_digest": self._linked_episode_digest,
            "linked_episode_media_verified_by_session": False,
            "provider_teardown_verified": False,
            "claim_ceiling": "planning_only",
        }
        closed["receipt_digest"] = canonical_digest(closed, digest_field="receipt_digest")
        with (self.output_dir / (SESSION_SCHEMA + ".json")).open("x", encoding="utf-8") as stream:
            json.dump(closed, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        self._closed = closed
        return closed

    def __enter__(self) -> "NativeG1TeamRuntimeSession":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def open_g1_team_runtime_session(
    *, profile: Mapping[str, Any], trusted_setup: Mapping[str, Any],
    authenticated_owner: Mapping[str, str], approved_binding: Mapping[str, Any],
    output_dir: Path, credential: str | None = None, fetcher: Any = None,
) -> NativeG1TeamRuntimeSession:
    """Probe exactly one operator-bound endpoint, image, or staged artifact."""

    bound = validate_team_policy_delivery_profile(
        profile, trusted_setup=trusted_setup, authenticated_owner=authenticated_owner
    )
    mode = bound["delivery"]["mode"]
    if (
        not isinstance(output_dir, Path) or not output_dir.is_absolute()
        or output_dir.exists() or output_dir.is_symlink()
        or not isinstance(approved_binding, Mapping)
        or approved_binding.get("mode") != mode
        or approved_binding.get("profile_digest") != bound["profile_digest"]
    ):
        raise ValueError("g1_team_runtime_session_binding_invalid")

    if mode == "authenticated_endpoint":
        if (
            set(approved_binding) != {
                "mode", "profile_digest", "approved_origin", "resolved_secret_ref"
            }
            or not isinstance(credential, str)
            or not credential
        ):
            raise ValueError("g1_team_runtime_endpoint_binding_invalid")
        robots = [
            robot for robot in trusted_setup["robot_presets"]
            if robot["robot_preset_id"] == bound["robot_preset_id"]
        ]
        robot = robots[0]
        options = {"fetcher": fetcher} if fetcher is not None else {}
        client = NativeG1TeamPolicyHttpsClient(
            profile=bound, expected_owner=authenticated_owner,
            expected_setup_digest=trusted_setup["setup_digest"],
            expected_interface={
                "robot_preset_id": robot["robot_preset_id"],
                "embodiment_id": robot["embodiment_id"],
                "observation_schema_id": robot["observation_schema"]["schema_id"],
                "action_schema_id": robot["action_schema"]["schema_id"],
            },
            approved_origin=approved_binding["approved_origin"],
            resolved_secret_ref=approved_binding["resolved_secret_ref"],
            credential=credential, **options,
        )
        conformance = run_g1_team_policy_synthetic_conformance(
            profile=bound, trusted_setup=trusted_setup,
            authenticated_owner=authenticated_owner, policy_client=client,
        )
        output_dir.mkdir(mode=0o700)
        with (output_dir / CONFORMANCE_FILENAME).open("x", encoding="utf-8") as stream:
            json.dump(conformance, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        lease = None
    elif mode == "container":
        if (
            set(approved_binding) != {"mode", "profile_digest", "image_ref", "gpu_device"}
            or credential is not None
        ):
            raise ValueError("g1_team_runtime_container_binding_invalid")
        lease, conformance = launch_g1_team_container_synthetic_probe(
            profile=bound, trusted_setup=trusted_setup,
            authenticated_owner=authenticated_owner,
            operator_approved_profile_digest=bound["profile_digest"],
            operator_approved_image_ref=approved_binding["image_ref"],
            output_dir=output_dir, gpu_device=approved_binding["gpu_device"],
        )
        client = lease.client
    elif mode == "noncontainer_artifact":
        if (
            set(approved_binding) != {
                "mode", "profile_digest", "artifact_sha256", "staged_artifact_path"
            }
            or credential is not None
        ):
            raise ValueError("g1_team_runtime_artifact_binding_invalid")
        lease, conformance = launch_g1_team_artifact_synthetic_probe(
            profile=bound, trusted_setup=trusted_setup,
            authenticated_owner=authenticated_owner,
            operator_approved_profile_digest=bound["profile_digest"],
            operator_approved_artifact_sha256=approved_binding["artifact_sha256"],
            staged_artifact_path=Path(approved_binding["staged_artifact_path"]),
            output_dir=output_dir,
        )
        client = lease.client
    else:
        raise ValueError("g1_team_runtime_delivery_mode_invalid")

    if conformance.get("status") != "synthetic_wire_compatible":
        if lease is not None:
            lease.close()
        raise ValueError("g1_team_runtime_synthetic_conformance_invalid")
    return NativeG1TeamRuntimeSession(
        client=client, conformance=conformance,
        profile_digest=bound["profile_digest"], delivery_mode=mode,
        output_dir=output_dir, lease=lease,
    )


__all__ = ["NativeG1TeamRuntimeSession", "open_g1_team_runtime_session"]
